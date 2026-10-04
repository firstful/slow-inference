#!/usr/bin/env python3
"""Generate tiny deterministic synthetic GGUF fixtures for llama.cpp parity.

The parity project was blocked on "we have no GGUF for architecture X, and
downloading one is not an option". These fixtures remove that blocker: a
model only needs to be *shaped* correctly, not *trained*, to serve as a
differential test between our forward pass and llama.cpp's.

Design constraints, all deliberate:

* **F32 everywhere.** Quantization noise would swamp the signal we are trying
  to measure. The quant kernels have their own bit-exact golden-vector test
  (`engine/tests/quant_golden.rs`); this harness is about *architecture
  semantics* — tensor naming, norms, RoPE, GQA, residual order — which is a
  different failure mode entirely.
* **Deterministic pseudo-random weights** from a fixed seed. A degenerate
  fixture (all ones, identity) can pass by accident; pseudo-random weights
  expose any transposition, mis-slicing, or missing scale factor.
* **Non-power-of-two and small dimensions** where cheap. Dimensions that are
  all nice powers of two hide off-by-one and stride bugs.
* **The same seed for every arch**, so fixtures stay comparable across runs.

Usage:
    python3 scripts/generate_synthetic_fixtures.py --arch llama --out models/synthetic-llama.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def import_gguf() -> object:
    try:
        import gguf  # type: ignore
    except ModuleNotFoundError:
        candidate = Path(
            sys.env.get("LLAMA_CPP_GGUF_PY", Path.home() / "llama.cpp" / "gguf-py")
        ) if (sys := __import__("sys")) else None
        if candidate.is_dir():
            __import__("sys").path.insert(0, str(candidate))
        try:
            import gguf  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "gguf-py is required. Set LLAMA_CPP_GGUF_PY or PYTHONPATH to llama.cpp/gguf-py."
            ) from exc
    return gguf


class Rng:
    """splitmix64 — same generator the C golden-vector tool uses."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def normal(self, n: int, scale: float = 1.0) -> np.ndarray:
        """Box-Muller over splitmix64 draws. Deterministic across platforms."""
        out = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            u1 = (self.next_u64() >> 11) / float(1 << 53)
            u2 = (self.next_u64() >> 11) / float(1 << 53)
            u1 = max(u1, 1e-12)
            r = float(np.sqrt(-2.0 * np.log(u1)))
            out[i] = r * float(np.cos(2.0 * np.pi * u2))
            if i + 1 < n:
                out[i + 1] = r * float(np.sin(2.0 * np.pi * u2))
            i += 2
        return (out * scale).astype(np.float32)

    def uniform(self, n: int, lo: float, hi: float) -> np.ndarray:
        d = (self.next_u64() >> 11) / float(1 << 53)
        return np.full(n, lo + (hi - lo) * d, dtype=np.float32)

    def mat(self, rows: int, cols: int, scale: float = 1.0) -> np.ndarray:
        # Fan-in scaled init, like a real model, so activations stay in range.
        s = scale / float(np.sqrt(cols))
        return self.normal(rows * cols, s).reshape(rows, cols)

    def cube(self, d0: int, d1: int, d2: int) -> np.ndarray:
        """3D pseudo-random weights. `mat`'s third argument is a scale, not a shape,
        so expert/absorbed-K tensors need their own constructor to avoid silently
        producing a 2D array with an absurd scale."""
        return self.normal(d0 * d1 * d2, 1.0 / float(np.sqrt(d2))).reshape(d0, d1, d2)


def add_tensor(writer: object, gguf: object, name: str, arr: np.ndarray) -> None:
    data = np.ascontiguousarray(arr, dtype=np.float32)
    if hasattr(writer, "add_tensor"):
        writer.add_tensor(name, data)
    else:
        writer.add_tensor_info(name, list(data.shape), data.dtype, data.nbytes)
        writer.write_tensor_data(data)


def write_common_vocab(
    writer: object,
    n_vocab: int,
    tokens: list[str] | None = None,
    types: list[int] | None = None,
    pre: str | None = None,
) -> None:
    """The `tok{i}` vocabulary every fixture shares, or an explicit one.

    `tokens`/`types` exist for `gpt2-strip`, the one fixture whose *vocabulary*
    is the subject rather than the vehicle: `tokenizer_st_partition`'s strip
    rules are keyed on the literal token texts `<mask>`, `[MASK]`, `</s>`,
    `<unk>` and `<s>`, and none of them appears in any GGUF on disk, so there is
    nothing to test the rules against until one of those vocabularies exists.
    """
    writer.add_string("tokenizer.ggml.model", "gpt2")
    # llama.cpp defaults an absent `tokenizer.ggml.pre` to `default` and carries
    # on; this engine refuses the absent case (see `pre_type_from_str`), because
    # `default` is a four-expression regex it has not implemented. So a fixture
    # that is meant to be *loaded by the engine* has to declare one. Only
    # `gpt2-strip` is, and the reason is its own docstring: the strip rules are
    # about attributes, and an engine that will not load the vocabulary cannot
    # demonstrate them.
    if pre is not None:
        writer.add_string("tokenizer.ggml.pre", pre)
    if tokens is None:
        tokens = [f"tok{i}" for i in range(n_vocab)]
    if types is None:
        types = [1] * len(tokens)
    if len(types) != len(tokens):
        raise SystemExit("write_common_vocab: tokens/types length mismatch")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.token_type", types)
    # llama.cpp refuses to build a gpt2 vocab with an empty merges list, so
    # give it one trivial (unused) rule. We only ever feed raw token ids, so
    # the merges never actually run.
    writer.add_array("tokenizer.ggml.merges", ["t o"])
    writer.add_uint32("tokenizer.ggml.bos_token_id", 1)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 2)
    writer.add_uint32("tokenizer.ggml.unknown_token_id", 0)
    writer.add_bool("tokenizer.ggml.add_bos_token", False)
    writer.add_bool("tokenizer.ggml.add_eos_token", False)


def gen_llama(
    path: Path,
    gguf: object,
    seed: int,
    freq_base: float = 10000.0,
    eps: float = 1e-5,
    bias: bool = False,
    arch_name: str = "llama",
) -> None:
    """llama / qwen2 family: RMSNorm, RoPE(NORM), GQA, SwiGLU, parallel-safe.

    `freq_base`, `eps` and `bias` are exposed so a single feature can be varied
    at a time when bisecting a parity mismatch — that is how the bias-vs-RoPE
    ordering bug was isolated.
    """
    rng = Rng(seed)
    n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab = 2, 64, 8, 2, 96, 128
    head_dim = n_embd // n_head
    arch = arch_name

    w = gguf.GGUFWriter(path, arch)
    w.add_string("general.name", f"synthetic-{arch}")
    w.add_uint32("general.alignment", 32)
    w.add_uint32(f"{arch}.block_count", n_layer)
    w.add_uint32(f"{arch}.embedding_length", n_embd)
    w.add_uint32(f"{arch}.feed_forward_length", n_ff)
    w.add_uint32(f"{arch}.attention.head_count", n_head)
    w.add_uint32(f"{arch}.attention.head_count_kv", n_head_kv)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", eps)
    w.add_float32(f"{arch}.rope.freq_base", freq_base)
    w.add_uint32(f"{arch}.rope.dimension_count", head_dim)
    w.add_uint32(f"{arch}.vocab_size", n_vocab)
    w.add_uint32(f"{arch}.context_length", 128)
    write_common_vocab(w, n_vocab)

    add_tensor(w, gguf, "token_embd.weight", rng.mat(n_vocab, n_embd))
    add_tensor(w, gguf, "output_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
    # Tied embedding deliberately: no output.weight, so llama.cpp duplicates
    # token_embd. Weight-tying is a common source of silent divergence.
    for i in range(n_layer):
        p = f"blk.{i}"
        add_tensor(w, gguf, f"{p}.attn_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
        add_tensor(w, gguf, f"{p}.attn_q.weight", rng.mat(n_head * head_dim, n_embd))
        add_tensor(w, gguf, f"{p}.attn_k.weight", rng.mat(n_head_kv * head_dim, n_embd))
        add_tensor(w, gguf, f"{p}.attn_v.weight", rng.mat(n_head_kv * head_dim, n_embd))
        if bias:
            add_tensor(w, gguf, f"{p}.attn_q.bias", rng.mat(1, n_head * head_dim).ravel())
            add_tensor(w, gguf, f"{p}.attn_k.bias", rng.mat(1, n_head_kv * head_dim).ravel())
            add_tensor(w, gguf, f"{p}.attn_v.bias", rng.mat(1, n_head_kv * head_dim).ravel())
        add_tensor(w, gguf, f"{p}.attn_output.weight", rng.mat(n_embd, n_head * head_dim))
        add_tensor(w, gguf, f"{p}.ffn_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
        add_tensor(w, gguf, f"{p}.ffn_gate.weight", rng.mat(n_ff, n_embd))
        add_tensor(w, gguf, f"{p}.ffn_up.weight", rng.mat(n_ff, n_embd))
        add_tensor(w, gguf, f"{p}.ffn_down.weight", rng.mat(n_embd, n_ff))
    _finish(w)


def gen_llama_moe(path: Path, gguf: object, seed: int, expert_weights_scale: float | None = None) -> None:
    """llama arch, MoE FFN, `expert_used_count < expert_count`, no shared expert.

    `expert_weights_scale`, when given, writes `{arch}.expert_weights_scale`.
    The `llama` arch does **not** read that key (`src/models/llama.cpp`'s
    `load_hparams` never calls `get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, ...)`),
    so `hparams.expert_weights_scale` stays at its 0.0 "unset" default whatever
    the file says — measured, not inferred: llama.cpp's own logits on
    `llama-moe-wscale` are byte-identical to its logits on `llama-moe`. The
    engine applied the key unconditionally, which is a real divergence on any
    file that carries it — and `llama-model-saver.cpp:226` writes exactly that
    key, so llama.cpp's own model saver produces such files.

    This is the fixture the whole set was missing. Tick 22 found that `llama`
    passes a literal `true` for `norm_w` in `build_moe_ffn` (llama.cpp:209), so
    the selected experts' weights are renormalised by their own sum — and that
    every `llama`-arch MoE model *on disk* has `expert_used_count ==
    expert_count` (Tiny-MoE is 2 of 2), where that division is a no-op up to one
    ulp. So the fix could only be pinned by a unit test.

    Here 4 experts are declared and 2 are used, so the correction is
    `sum(all probs) / sum(top-2 probs)` — a factor of roughly 2 on a router that
    does not strongly prefer any expert, and therefore several percent on the
    FFN output. Any engine that skips `norm_w` is wrong by that much, and the
    float64 arbiter in `arbitrate_logits.py::moe_ffn` computes it independently.

    Three further properties this fixture is built to have:

    * **`expert_weights_norm` is written as `false`,** which llama.cpp's `llama`
      arch *ignores* — it passes a literal, not `hparams.expert_weights_norm`.
      So the file contains a key whose value is the wrong answer for this arch,
      and a parser that trusts it diverges from llama.cpp. That is the shape of
      the bug, not a coincidence of it. (`moe_norm_w_for` is the per-arch
      constant table that gets this right.)
    * **The router is deliberately *unconfident*.** `ffn_gate_inp` is drawn at
      half the usual fan-in scale. This is the whole design of the fixture, and
      it is the sharpened form of the trap tick 22 walked into: what hides a
      missing `norm_w` is not only "the model uses every expert" but **"the
      selected weights already sum to ~1"**. A confident router picks two of four
      experts holding 0.99 of the probability mass and the renormalisation is a
      no-op there too — so `scale=3.0`, which spreads the top-2 over ten distinct
      subsets, still leaves the *worst* of 24 routing decisions at a factor of
      1.0000 and could pass on one lucky position. At `scale=0.5` the softmax is
      nearly uniform and **every** routing decision has a factor >= 1.21 (median
      1.41), across all six unordered expert pairs. A router whose four columns
      were identical would instead route to experts {0,1} at every position and
      could not see a wrong *selection* at all, only a wrong weight.

      Measured by `scripts/arb_route_power.py`, which prints the factor table and
      re-runs the engine against a `norm_w`-ablated reference.
    * **There is deliberately no fused variant of this fixture.** The fused
      `ffn_gate_up_exps` form is real and `create_tensor_gate_up_exps`
      (llama-model.cpp:2881) prefers it, but the `llama` arch does not call that
      helper -- `src/models/llama.cpp:78-80` creates `ffn_gate_exps` and
      `ffn_up_exps` directly, with `ffn_up_exps` at `flags = 0` i.e. required. A
      `llama`-arch GGUF carrying only `ffn_gate_up_exps` is therefore rejected by
      llama.cpp before any token is evaluated:

          llama_model_load: error loading model: check_tensor_dims:
              tensor 'blk.0.ffn_up_exps.weight' not found

      (measured, not inferred, with `tools/gen_ref_logits` on a file this
      generator produced). `deepseek2` *does* call the helper and takes both
      forms, so `gen_deepseek2(fused=True)` is where the fused path is tested.

    No shared expert: llama.cpp only creates `ffn_{up,gate,down}_shexp` when
    `hparams.n_ff_shexp > 0` (llama.cpp:86), and the `llama` arch never reads
    `expert_shared_feed_forward_length`, so on this arch it is always 0. A real
    `llama`-arch MoE file (Tiny-MoE) carries exactly the same key set: just
    `expert_count` and `expert_used_count`.
    """
    rng = Rng(seed)
    n_layer, n_embd, n_head, n_head_kv = 3, 32, 4, 2
    head_dim = 8
    n_ff = 14  # the expert FFN width; also `feed_forward_length`
    n_expert, n_expert_used = 4, 2
    n_vocab, n_ctx = 128, 128
    eps = 1e-6
    arch = "llama"

    w = gguf.GGUFWriter(path, arch)
    w.add_string("general.name", "synthetic-llama-moe")
    w.add_uint32("general.alignment", 32)
    w.add_uint32(f"{arch}.block_count", n_layer)
    w.add_uint32(f"{arch}.embedding_length", n_embd)
    # llama.cpp's llama.cpp loader sizes the expert tensors from `n_ff`
    # (llama.cpp:80-82), *not* from `expert_feed_forward_length`, so the key is
    # omitted here exactly as Tiny-MoE omits it. The engine infers the expert
    # width from the tensor itself (`up_t.n_rows / n_experts`).
    w.add_uint32(f"{arch}.feed_forward_length", n_ff)
    w.add_uint32(f"{arch}.vocab_size", n_vocab)
    w.add_uint32(f"{arch}.context_length", n_ctx)
    w.add_uint32(f"{arch}.attention.head_count", n_head)
    w.add_uint32(f"{arch}.attention.head_count_kv", n_head_kv)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", eps)
    w.add_float32(f"{arch}.rope.freq_base", 10000.0)
    w.add_uint32(f"{arch}.rope.dimension_count", head_dim)
    # MoE. Only these three keys; no gating func, no expert_groups — all of
    # which the `llama` arch supplies as literals or leaves at their defaults,
    # so a fixture that set them would test the wrong thing (see the docstring
    # on `expert_weights_norm` below). `expert_weights_scale` is absent here and
    # present with a value the arch must ignore on the `-wscale` variant.
    w.add_uint32(f"{arch}.expert_count", n_expert)
    w.add_uint32(f"{arch}.expert_used_count", n_expert_used)
    # Deliberately the *wrong* value for this arch. llama.cpp passes a literal
    # `true` for norm_w at llama.cpp:209 and never reads this key, so a correct
    # engine renormalises anyway. Anything that reads the flag is wrong.
    w.add_bool(f"{arch}.expert_weights_norm", False)
    if expert_weights_scale is not None:
        # The same trick one key over, for `expert_weights_scale`: a value the
        # `llama` arch must ignore. Present only on the `-wscale` variant, so
        # the plain fixture keeps exercising the *absent*-key path.
        w.add_float32(f"{arch}.expert_weights_scale", expert_weights_scale)
    write_common_vocab(w, n_vocab)

    add_tensor(w, gguf, "token_embd.weight", rng.mat(n_vocab, n_embd))
    add_tensor(w, gguf, "output_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
    # Tied output (no output.weight), like every other llama-arch fixture.
    for i in range(n_layer):
        p = f"blk.{i}"
        add_tensor(w, gguf, f"{p}.attn_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
        add_tensor(w, gguf, f"{p}.attn_q.weight", rng.mat(n_head * head_dim, n_embd))
        add_tensor(w, gguf, f"{p}.attn_k.weight", rng.mat(n_head_kv * head_dim, n_embd))
        add_tensor(w, gguf, f"{p}.attn_v.weight", rng.mat(n_head_kv * head_dim, n_embd))
        add_tensor(w, gguf, f"{p}.attn_output.weight", rng.mat(n_embd, n_head * head_dim))
        add_tensor(w, gguf, f"{p}.ffn_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
        # Router at HALF the usual fan-in scale -- see the docstring: the
        # discriminating power of this fixture comes from an unconfident router,
        # because `norm_w` is unobservable exactly when the selected weights
        # already sum to 1.
        add_tensor(w, gguf, f"{p}.ffn_gate_inp.weight", rng.mat(n_expert, n_embd, scale=0.5))
        add_tensor(w, gguf, f"{p}.ffn_down_exps.weight", rng.cube(n_expert, n_embd, n_ff))
        add_tensor(w, gguf, f"{p}.ffn_up_exps.weight", rng.cube(n_expert, n_ff, n_embd))
        add_tensor(w, gguf, f"{p}.ffn_gate_exps.weight", rng.cube(n_expert, n_ff, n_embd))
    _finish(w)


def gen_qwen2(path: Path, gguf: object, seed: int) -> None:
    """qwen2 = llama shape plus per-tensor QKV biases (and its own arch key)."""
    gen_llama(
        path,
        gguf,
        seed,
        freq_base=1_000_000.0,
        eps=1e-6,
        bias=True,
        arch_name="qwen2",
    )


#: Index of each strip-rule token in the `gpt2-strip` vocabulary, so no test
#: re-types the text. Tick 29 lost four literals to the harness rendering
#: ``<|tool_response|>`` with a closing pipe upstream does not have; an index
#: cannot be misread, a spelling can.
STRIP_MASK = 0
STRIP_BRACKET_MASK = 1
STRIP_S = 2
STRIP_UNK = 3
STRIP_BOS = 4
STRIP_SPECIAL_COUNT = 5

#: `tokenizer.ggml.pre` for the `gpt2-strip` fixture. It has to satisfy three
#: things at once, and the first two are easy to get wrong:
#:
#:   * llama.cpp must *accept* the spelling, or it refuses to load the file
#:     ("unknown pre-tokenizer type") and there is no reference to compare to.
#:     Measured: `gpt2` is rejected — upstream spells it `gpt-2`
#:     (llama-vocab.cpp:2174), while this engine accepts `gpt2`. So the spelling
#:     the engine knows is one upstream does not.
#:   * the spelling must not itself trigger a strip rule, or the "no label"
#:     baseline stops being a baseline. That rules out every `jina-v2-*` and
#:     `modern-bert` value.
#:   * this engine must accept it too, or the fixture cannot be loaded at all.
#:
#: `mpt` is the one value in all three: upstream routes it to
#: `LLAMA_VOCAB_PRE_TYPE_MPT` (llama-vocab.cpp:2168), which shares the GPT-2
#: `regex_exprs` entry this engine's `PreType::Gpt2` implements, and it is in the
#: engine's match arm.
STRIP_FIXTURE_PRE = "mpt"


def strip_vocab(n_vocab: int) -> tuple[list[str], list[int]]:
    """The `gpt2-strip` token list: the five strip-rule tokens, then `w{i}`.

    `tokenizer_st_partition`'s LSTRIP/RSTRIP rules name literal token texts, and
    every branch needs a token that no vocabulary on disk has:

      * ``<mask>``    — LSTRIP, when `tokenizer.ggml.pre` says jina-v2-* or the
                        architecture is nomic-bert-moe / jina-bert-v3;
      * ``[MASK]``    — LSTRIP, when `general.name` says modern-bert;
      * ``</s>``       — RSTRIP, in the phi-3 branch;
      * ``<unk>``, ``<s>`` — RSTRIP *cleared*, in the same branch, after it has
                        been set on every `cache_special_tokens` id.

    Types are chosen to be discriminating rather than convenient. ``<mask>`` and
    ``[MASK]`` are CONTROL, so the partition cuts on them only under
    `parse_special = true`; ``<unk>`` is UNKNOWN, so it is skipped under
    `parse_special = false` as well. The phi-3 branch's clearing loop is the only
    thing that ever distinguishes an UNKNOWN id from a CONTROL one, and only if
    the ids differ — which is the whole reason ``<unk>`` is not CONTROL here.

    ``<empty>`` is deliberately **absent**, and its absence is the finding.
    Upstream's phi-3 branch calls `token_to_id.at("<empty>")` unconditionally,
    and a GGUF's empty token is rewritten to ``[EMPTY_<i>]`` at load
    (llama-vocab.cpp:2438), so that ``.at`` throws for every vocabulary that does
    not literally contain those eight characters. No real tokenizer has such a
    token, so no real Phi-3 GGUF loads at all — measured, in
    ``scripts/gen_strip_refs.sh``.
    """
    specials = [
        ("<mask>", 3),  # CONTROL
        ("[MASK]", 3),  # CONTROL
        ("</s>", 3),  # CONTROL
        ("<unk>", 2),  # UNKNOWN
        ("<s>", 3),  # CONTROL
    ]
    tokens = [t for t, _ in specials]
    types = [ty for _, ty in specials]
    # Plain text tokens either side so a corpus has something to strip *around*,
    # and so the fixture keeps `gpt2`'s vocabulary size and therefore `gpt2`'s
    # tensor shapes — the two files differ in exactly one list.
    for i in range(n_vocab - STRIP_SPECIAL_COUNT):
        tokens.append(f"w{i}")
        types.append(1)
    return tokens, types


def gen_gpt2(
    path: Path,
    gguf: object,
    seed: int,
    tokens: list[str] | None = None,
    types: list[int] | None = None,
    pre: str | None = None,
) -> None:
    """gpt2: LayerNorm with bias, learned positional embeddings, GELU, no RoPE.

    `tokens`/`types` override the shared `tok{i}` vocabulary. Only `gpt2-strip`
    uses that, and the tensors are byte-for-byte the same function either way:
    a difference in its tokenizer output is then attributable to the vocabulary
    and to nothing else.
    """
    rng = Rng(seed)
    n_layer, n_embd, n_head, n_ff, n_vocab, n_ctx = 2, 64, 8, 96, 128, 128
    head_dim = n_embd // n_head

    w = gguf.GGUFWriter(path, "gpt2")
    w.add_string("general.name", "synthetic-gpt2")
    w.add_uint32("general.alignment", 32)
    w.add_uint32("gpt2.block_count", n_layer)
    w.add_uint32("gpt2.embedding_length", n_embd)
    w.add_uint32("gpt2.feed_forward_length", n_ff)
    w.add_uint32("gpt2.attention.head_count", n_head)
    w.add_float32("gpt2.attention.layer_norm_epsilon", 1e-5)
    w.add_uint32("gpt2.vocab_size", n_vocab)
    w.add_uint32("gpt2.context_length", n_ctx)
    write_common_vocab(w, n_vocab, tokens, types, pre=pre)

    add_tensor(w, gguf, "token_embd.weight", rng.mat(n_vocab, n_embd))
    add_tensor(w, gguf, "position_embd.weight", rng.mat(n_ctx, n_embd, scale=0.5))
    add_tensor(w, gguf, "output_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
    add_tensor(w, gguf, "output_norm.bias", rng.uniform(n_embd, -0.1, 0.1))
    for i in range(n_layer):
        p = f"blk.{i}"
        add_tensor(w, gguf, f"{p}.attn_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
        add_tensor(w, gguf, f"{p}.attn_norm.bias", rng.uniform(n_embd, -0.1, 0.1))
        # fused QKV: [n_embd + 2*n_embd, n_embd]
        add_tensor(w, gguf, f"{p}.attn_qkv.weight", rng.mat(n_embd + 2 * n_embd, n_embd))
        add_tensor(w, gguf, f"{p}.attn_qkv.bias", rng.mat(1, n_embd + 2 * n_embd).ravel())
        add_tensor(w, gguf, f"{p}.attn_output.weight", rng.mat(n_embd, n_embd))
        add_tensor(w, gguf, f"{p}.attn_output.bias", rng.uniform(n_embd, -0.1, 0.1))
        add_tensor(w, gguf, f"{p}.ffn_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
        add_tensor(w, gguf, f"{p}.ffn_norm.bias", rng.uniform(n_embd, -0.1, 0.1))
        add_tensor(w, gguf, f"{p}.ffn_up.weight", rng.mat(n_ff, n_embd))
        add_tensor(w, gguf, f"{p}.ffn_up.bias", rng.uniform(n_ff, -0.1, 0.1))
        add_tensor(w, gguf, f"{p}.ffn_down.weight", rng.mat(n_embd, n_ff))
        add_tensor(w, gguf, f"{p}.ffn_down.bias", rng.uniform(n_embd, -0.1, 0.1))
    _finish(w)


def gen_deepseek2(
    path: Path,
    gguf: object,
    seed: int,
    fused: bool = False,
    expert_weights_norm: bool = False,
) -> None:
    """deepseek2: MLA with absorption, mixed dense-lead / MoE layers.

    This is the fixture ROADMAP 9.15 has been waiting on. `models/synthetic-deepseek2-mla.gguf`
    (from `generate_deepseek2_mla_fixture.py`) is the older structural smoke test and is
    deliberately degenerate -- identity projections, ones for every norm, zeros for the whole
    FFN. That shape can pass while the arithmetic is wrong: with `ffn_up/down` identically zero
    the FFN contributes nothing, with identity weights a transposed slice still lands in the
    right place, and with every RMSNorm weight equal to 1 the norm cannot be checked either.

    So this one is pseudo-random like the llama/qwen2/gpt2 fixtures, and it exercises the three
    things that make DeepSeek-V2 its own architecture rather than a llama variant:

    * the split Q projection (`attn_q_a` -> RMSNorm -> `attn_q_b`) with per-head
      `nope`/`pe` interleaving inside one row of length `key_length_mla`;
    * the absorbed-MLA attention: `attn_k_b` folds the per-head K into Q so the attention
      collapses to MQA, and `attn_v_b` expands the weighted latent back out per head;
    * a dense-lead block followed by MoE blocks with a shared expert, so both FFN branches
      and their sum into the residual are covered.

    Dimensions are deliberately small and mostly not powers of two (`key_length_mla = 6`,
    `kv_lora_rank = 6`, `n_ff_exp = 14`, `n_expert = 4`): nice round numbers hide stride
    bugs. `eps = 1e-6` differs from the usual 1e-5 so a fixture with the wrong epsilon
    cannot agree.

    **`fused=True` writes the same weights in llama.cpp's preferred single-tensor
    form.** `create_tensor_gate_up_exps` (llama-model.cpp:2881) tries
    `ffn_gate_up_exps` = `{n_embd, n_ff_exp*2, n_expert}` first and only falls back to
    the split pair, and `build_moe_ffn` then takes two views of one matmul result:
    gate at row offset 0, up at row offset `n_ff_exp` (llama-graph.cpp:2100-2107). So
    expert `e` owns rows `[2*e*n_ff_exp, 2*e*n_ff_exp + 2*n_ff_exp)` with **gate rows
    first, then up rows** -- which is what `convert_hf_to_gguf.py --fuse-gate-up-exps`
    produces with `torch.cat([gate, up], dim=1)`.

    The two draws are hoisted out of the `if` and in the same order for both forms, so
    `synthetic-deepseek2-fused.gguf` is the split fixture with two tensors renamed and
    concatenated -- nothing else differs, not one weight bit. That is what makes
    `tests/ref_parity.rs::the_fused_and_split_expert_forms_agree_bit_for_bit` a real
    identity rather than two unrelated models that happen to be close: it is the only
    check that the row order inside the fused tensor is gate-then-up, since reading it
    backwards produces a *different* model that the tolerance gate catches as well.

    deepseek2 is the architecture that supports this fixture and the only one in the
    engine's supported set that does -- see `gen_llama_moe` for why the `llama` arch
    cannot, measured rather than inferred.
    """
    rng = Rng(seed)
    n_layer, n_embd, n_head = 3, 32, 4
    n_lora_q, n_lora_kv = 12, 6
    key_length_mla, value_length_mla, rope_dim = 6, 6, 4
    n_ff, n_ff_exp, n_expert, n_expert_used, n_expert_shared = 20, 14, 4, 2, 1
    n_vocab, n_ctx = 128, 128
    n_dense_lead = 1
    eps = 1e-6
    arch = "deepseek2"

    w = gguf.GGUFWriter(path, arch)
    w.add_string("general.name", "synthetic-deepseek2-fused" if fused else "synthetic-deepseek2")
    w.add_uint32("general.alignment", 32)
    w.add_uint32(f"{arch}.block_count", n_layer)
    w.add_uint32(f"{arch}.embedding_length", n_embd)
    w.add_uint32(f"{arch}.feed_forward_length", n_ff)
    w.add_uint32(f"{arch}.vocab_size", n_vocab)
    w.add_uint32(f"{arch}.context_length", n_ctx)
    w.add_uint32(f"{arch}.attention.head_count", n_head)
    # MLA ignores head_count_kv entirely (K/V are the latent, shared by all heads);
    # real DeepSeek-V2 GGUFs still carry 1. The engine must not depend on it.
    w.add_uint32(f"{arch}.attention.head_count_kv", 1)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", eps)
    # MLA geometry. `attention.key_length` is *not* the MLA head width for these
    # models -- it is the width of one cached latent row, i.e. kv_lora_rank + rope
    # dims, and llama.cpp sizes the KV cache from it (n_embd_k_gqa =
    # n_embd_head_k * n_head_kv). Setting it to key_length_mla makes llama.cpp
    # allocate a 8-wide cache and abort in ggml_set_rows.
    w.add_uint32(f"{arch}.attention.key_length", n_lora_kv + rope_dim)
    w.add_uint32(f"{arch}.attention.key_length_mla", key_length_mla)
    w.add_uint32(f"{arch}.attention.value_length_mla", value_length_mla)
    w.add_uint32(f"{arch}.attention.q_lora_rank", n_lora_q)
    w.add_uint32(f"{arch}.attention.kv_lora_rank", n_lora_kv)
    w.add_uint32(f"{arch}.leading_dense_block_count", n_dense_lead)
    # MoE
    w.add_uint32(f"{arch}.expert_count", n_expert)
    w.add_uint32(f"{arch}.expert_used_count", n_expert_used)
    w.add_uint32(f"{arch}.expert_feed_forward_length", n_ff_exp)
    w.add_uint32(f"{arch}.expert_shared_count", n_expert_shared)
    # `deepseek2` is in the group that forwards `hparams.expert_weights_norm` to
    # `build_moe_ffn` AND reads the key, so for this architecture the flag is
    # live metadata rather than a hint. The plain fixture keeps it false (the
    # value llama.cpp's default gives, and the one every synthetic case so far
    # has carried); `--arch deepseek2-normw` sets it true so the DEFER branch of
    # `moe_norm_w_for` has a fixture behind it instead of only an oracle.
    w.add_bool(f"{arch}.expert_weights_norm", expert_weights_norm)
    w.add_uint32(f"{arch}.expert_gating_func", 1)  # 1 == SOFTMAX
    # RoPE: NEOX is wrong here (deepseek2 is llama.cpp's NORM), and a real yarn
    # extension would change both kq_scale and the frequency ladder, so scaling
    # stays off. attn_factor/yarn_log_multiplier stay at 1/0, which makes the
    # pre-scaled kq_scale collapse to 1/sqrt(key_length_mla) -- note that is the
    # *full* head width including the rope part, not the nope part.
    w.add_float32(f"{arch}.rope.freq_base", 10000.0)
    w.add_uint32(f"{arch}.rope.dimension_count", rope_dim)
    w.add_float32(f"{arch}.rope.scaling.factor", 1.0)
    write_common_vocab(w, n_vocab)

    add_tensor(w, gguf, "token_embd.weight", rng.mat(n_vocab, n_embd))
    add_tensor(w, gguf, "output.weight", rng.mat(n_vocab, n_embd))
    add_tensor(w, gguf, "output_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))

    for i in range(n_layer):
        p = f"blk.{i}"
        add_tensor(w, gguf, f"{p}.attn_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
        # split Q projection: low-rank bottleneck then RMSNorm then up-project
        add_tensor(w, gguf, f"{p}.attn_q_a.weight", rng.mat(n_lora_q, n_embd))
        add_tensor(w, gguf, f"{p}.attn_q_a_norm.weight", 1.0 + rng.uniform(n_lora_q, -0.2, 0.2))
        add_tensor(w, gguf, f"{p}.attn_q_b.weight", rng.mat(n_head * key_length_mla, n_lora_q))
        # KV down-projection: latent + shared rope part in one fused tensor
        add_tensor(w, gguf, f"{p}.attn_kv_a_mqa.weight", rng.mat(n_lora_kv + rope_dim, n_embd))
        add_tensor(w, gguf, f"{p}.attn_kv_a_norm.weight", 1.0 + rng.uniform(n_lora_kv, -0.2, 0.2))
        # absorbed-K and V-expansion are 3D in GGUF: (nope, kv_lora, head) and
        # (kv_lora, v_head, head). They are read here as if they were 2D slices,
        # which is the property under test -- see model.rs:1343.
        qk_nope = key_length_mla - rope_dim
        add_tensor(w, gguf, f"{p}.attn_k_b.weight", rng.cube(n_head, n_lora_kv, qk_nope))
        add_tensor(w, gguf, f"{p}.attn_v_b.weight", rng.cube(n_head, value_length_mla, n_lora_kv))
        # {n_head * n_embd_head_v_mla, n_embd} -- note the order: the attention
        # output projects *from* the head dimension *into* the model dimension.
        add_tensor(w, gguf, f"{p}.attn_output.weight", rng.mat(n_embd, n_head * value_length_mla))
        add_tensor(w, gguf, f"{p}.ffn_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
        if i < n_dense_lead:
            add_tensor(w, gguf, f"{p}.ffn_gate.weight", rng.mat(n_ff, n_embd))
            add_tensor(w, gguf, f"{p}.ffn_up.weight", rng.mat(n_ff, n_embd))
            add_tensor(w, gguf, f"{p}.ffn_down.weight", rng.mat(n_embd, n_ff))
        else:
            add_tensor(w, gguf, f"{p}.ffn_gate_inp.weight", rng.mat(n_expert, n_embd))
            # Drawn in one order for both forms so `fused=True` is the split file
            # with these two tensors concatenated -- see the docstring.
            up = rng.cube(n_expert, n_ff_exp, n_embd)
            gate = rng.cube(n_expert, n_ff_exp, n_embd)
            if fused:
                # GGUF ne {n_embd, n_ff_exp*2, n_expert}; numpy (n_expert, 2*n_ff_exp,
                # n_embd). Gate rows first, then up rows, per expert.
                add_tensor(
                    w,
                    gguf,
                    f"{p}.ffn_gate_up_exps.weight",
                    np.concatenate([gate, up], axis=1),
                )
            else:
                add_tensor(w, gguf, f"{p}.ffn_up_exps.weight", up)
                add_tensor(w, gguf, f"{p}.ffn_gate_exps.weight", gate)
            add_tensor(w, gguf, f"{p}.ffn_down_exps.weight", rng.cube(n_expert, n_embd, n_ff_exp))
            n_sh = n_ff_exp * n_expert_shared
            add_tensor(w, gguf, f"{p}.ffn_up_shexp.weight", rng.mat(n_sh, n_embd))
            add_tensor(w, gguf, f"{p}.ffn_gate_shexp.weight", rng.mat(n_sh, n_embd))
            add_tensor(w, gguf, f"{p}.ffn_down_shexp.weight", rng.mat(n_embd, n_sh))
    _finish(w)


def gen_gemma3(path: Path, gguf: object, seed: int) -> None:
    """gemma3: sliding-window / global alternating attention, post-norms, GEGLU.

    Four properties of this architecture have no analogue in the llama-family
    fixtures, and each one is something a forward pass can get wrong in a way
    that still produces plausible-looking logits:

    * **The layer classification has to be observable.** `sliding_window` is 4, the
      prompt is 8 tokens, and `n_layer = 8` so layers 0-4 are SWA and 5-7 are global
      under llama.cpp's default pattern of 6 (`set_swa_pattern`, llama-hparams.cpp:8,
      `is_swa(il) = il % n_pattern < n_pattern - 1`). An SWA layer that silently
      attended to the whole prefix would produce different numbers, not the same.
    * **`n_head * key_length != n_embd`.** 4 heads of width 14 is 56 against an
      `embedding_length` of 48 — this is not an accident of the fixture, it is
      Gemma-3's real geometry (1B: 4 heads x 256 = 1024 vs n_embd 1152). llama.cpp
      therefore sizes `attn_output.weight` as `{n_embd_head_k * n_head, n_embd}`
      (gemma3.cpp), so the attention output buffer is 56 wide, not 48. A fixture
      with `n_embd == n_head * key_length` cannot tell those apart.
    * **Per-head QK-norm with a *shared* weight vector.** `attn_q_norm` and
      `attn_k_norm` are each `{n_embd_head_k}` long and each is applied to *every*
      head independently — they are not tiled per head.
    * **Post-norm ordering.** `attn_post_norm` and `ffn_post_norm` run on the
      branch output *before* it is added to the residual, so the residual stream is
      a sum of two normalised vectors rather than a normalised sum. (The Gemma3RMSNorm
      `+1.0` shift is applied by the converter, so the weights here already include
      it and the runtime norm is plain RMSNorm.)

    Metadata is deliberately **converter-faithful**, which turns out to be the most
    interesting part. `convert_hf_to_gguf.py`'s `Gemma3Model.set_gguf_parameters`
    calls `add_sliding_window` but never `add_sliding_window_pattern`,
    `add_rope_freq_base_swa` or `add_key_length_swa`, even though
    `gguf_writer.py` implements all three. A real Gemma-3 GGUF therefore leans on
    three llama.cpp defaults:

    | key | absent | llama.cpp default | source |
    |---|---|---|---|
    | `attention.sliding_window_pattern` | yes | **6** | gemma3.cpp:8 `swa_period = 6` |
    | `rope.freq_base_swa` | yes | **10000.0** | llama-hparams.h:127 |
    | `attention.key_length_swa` | yes | = `key_length` | llama-model.cpp:1219 |

    All three are omitted here for fidelity. That makes `rope.freq_base` (1e6, the
    value the converter writes for global layers) and the SWA layers' actual base
    10000.0 *differ by two orders of magnitude in the fixture*, so an engine that
    defaults `freq_base_swa` to "unset, fall back to `freq_base`" instead of to
    10000.0 is caught rather than merely noted.

    `final_logit_softcapping` is left out: it is a near-identity map at these logit
    magnitudes and would only add a tanh rounding floor to the parity gate. The
    engine's unit test covers it.
    """
    rng = Rng(seed)
    n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab, n_ctx = 8, 48, 4, 2, 70, 128, 128
    head_dim = 14  # n_head * head_dim == 56 != n_embd, on purpose
    n_swa = 4
    eps = 1e-6
    arch = "gemma3"

    w = gguf.GGUFWriter(path, arch)
    w.add_string("general.name", "synthetic-gemma3")
    w.add_uint32("general.alignment", 32)
    w.add_uint32(f"{arch}.block_count", n_layer)
    w.add_uint32(f"{arch}.embedding_length", n_embd)
    w.add_uint32(f"{arch}.feed_forward_length", n_ff)
    w.add_uint32(f"{arch}.vocab_size", n_vocab)
    w.add_uint32(f"{arch}.context_length", n_ctx)
    w.add_uint32(f"{arch}.attention.head_count", n_head)
    w.add_uint32(f"{arch}.attention.head_count_kv", n_head_kv)
    w.add_uint32(f"{arch}.attention.key_length", head_dim)
    w.add_uint32(f"{arch}.attention.value_length", head_dim)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", eps)
    # The only SWA key a converted file carries.
    w.add_uint32(f"{arch}.attention.sliding_window", n_swa)
    w.add_float32(f"{arch}.rope.freq_base", 1_000_000.0)
    w.add_uint32(f"{arch}.rope.dimension_count", head_dim)
    write_common_vocab(w, n_vocab)

    add_tensor(w, gguf, "token_embd.weight", rng.mat(n_vocab, n_embd))
    add_tensor(w, gguf, "output_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
    for i in range(n_layer):
        p = f"blk.{i}"
        add_tensor(w, gguf, f"{p}.attn_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
        add_tensor(w, gguf, f"{p}.attn_q.weight", rng.mat(n_head * head_dim, n_embd))
        add_tensor(w, gguf, f"{p}.attn_k.weight", rng.mat(n_head_kv * head_dim, n_embd))
        add_tensor(w, gguf, f"{p}.attn_v.weight", rng.mat(n_head_kv * head_dim, n_embd))
        # One weight vector of length key_length, reused by every head.
        add_tensor(w, gguf, f"{p}.attn_q_norm.weight", 1.0 + rng.uniform(head_dim, -0.2, 0.2))
        add_tensor(w, gguf, f"{p}.attn_k_norm.weight", 1.0 + rng.uniform(head_dim, -0.2, 0.2))
        # {n_embd_head_k * n_head, n_embd} == {56, 48}, i.e. NOT {n_embd, n_embd}.
        add_tensor(w, gguf, f"{p}.attn_output.weight", rng.mat(n_embd, n_head * head_dim))
        add_tensor(w, gguf, f"{p}.post_attention_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
        add_tensor(w, gguf, f"{p}.ffn_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
        add_tensor(w, gguf, f"{p}.ffn_gate.weight", rng.mat(n_ff, n_embd))
        add_tensor(w, gguf, f"{p}.ffn_up.weight", rng.mat(n_ff, n_embd))
        add_tensor(w, gguf, f"{p}.ffn_down.weight", rng.mat(n_embd, n_ff))
        add_tensor(w, gguf, f"{p}.post_ffw_norm.weight", 1.0 + rng.uniform(n_embd, -0.2, 0.2))
    _finish(w)


def _finish(w: object) -> None:
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


ARCHS = {
    "llama": gen_llama,
    "llama-moe": gen_llama_moe,
    "llama-moe-wscale": lambda path, gguf, seed: gen_llama_moe(path, gguf, seed, expert_weights_scale=2.5),
    "qwen2": gen_qwen2,
    "gpt2": gen_gpt2,
    "gpt2-strip": lambda path, gguf, seed: gen_gpt2(
        path, gguf, seed, *strip_vocab(128), pre=STRIP_FIXTURE_PRE
    ),
    "deepseek2": gen_deepseek2,
    "deepseek2-fused": lambda path, gguf, seed: gen_deepseek2(path, gguf, seed, fused=True),
    "deepseek2-normw": lambda path, gguf, seed: gen_deepseek2(
        path, gguf, seed, expert_weights_norm=True
    ),
    "gemma3": gen_gemma3,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    ap.add_argument("--freq-base", type=float, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0x5EED)
    args = ap.parse_args()

    gguf = import_gguf()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    if args.arch == "llama" and (args.freq_base or args.eps or args.bias):
        gen_llama(
            args.out, gguf, args.seed,
            freq_base=args.freq_base if args.freq_base is not None else 10000.0,
            eps=args.eps if args.eps is not None else 1e-5,
            bias=args.bias,
        )
    else:
        ARCHS[args.arch](args.out, gguf, args.seed)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, arch={args.arch}, seed={args.seed:#x})")


if __name__ == "__main__":
    main()
