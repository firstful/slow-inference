#!/usr/bin/env python3
"""Arbitrate whose logit error is real: ours, or llama.cpp's.

`scripts/compare_logits.py` reports engine-vs-llama.cpp disagreement. On the F32
synthetic fixtures that disagreement is ~2e-4 relative — far larger than f32
rounding for these tiny dimensions, so *something* is wrong, but a two-way diff
cannot say which side is at fault. This script supplies the missing third
opinion: an independent forward pass evaluated in **float64**, which removes
accumulation-order error from the question entirely.

For each fixture we compute

    d_engine = ||logits_engine - logits_f64|| / ||logits_f64||
    d_llama  = ||logits_llama  - logits_f64|| / ||logits_f64||

The side with the smaller number is the one accumulating in f32; the other has a
real semantic bug. If they are comparable, the 2e-4 is neither's fault and comes
from something the f64 model deliberately does not reproduce.

This is deliberately a *third* implementation rather than a re-reading of the
Rust source: it is written from llama.cpp's `src/models/<arch>.cpp` graph order,
so agreement between all three is real evidence and a shared misreading is still
possible but has to be made twice.

Usage:
    python3 scripts/arbitrate_logits.py --arch gpt2 \
        --engine /tmp/gpt2.eng.ref --llama engine/tests/data/fixtures/gpt2.ref
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

SQRT_2_OVER_PI = 0.7978845608028654
GELU_COEF_A = 0.044715


def f16_bits(x: np.ndarray) -> np.ndarray:
    """Round float32 values to float16 precision, returned as float64.

    Used to model ggml's precomputed 128 KB activation tables, which are keyed
    by the float16 bit pattern of the *input* and store float16 outputs.
    """
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float64)


def gelu(x, f16_table=False):
    # ggml's `ggml_gelu_f32`: 0.5*x*(1+tanh(sqrt(2/pi)*x*(1+a*x*x)))
    def exact(v):
        return 0.5 * v * (1.0 + np.tanh(SQRT_2_OVER_PI * v * (1.0 + GELU_COEF_A * v * v)))

    if not f16_table:
        return exact(x)
    # `GGML_GELU_FP16` is defined unconditionally (ggml-cpu/vec.h:46), so
    # ggml_vec_gelu_f32 looks the answer up in a 64K-entry float16 table keyed
    # by the float16 bits of the input. Both the key and the stored value are
    # float16, which is worth ~5e-4 relative error per element.
    return f16_bits(exact(f16_bits(x)))


def import_gguf():
    try:
        import gguf  # type: ignore
    except ModuleNotFoundError:
        candidate = Path(
            __import__("os").environ.get("LLAMA_CPP_GGUF_PY", Path.home() / "llama.cpp" / "gguf-py")
        )
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
        try:
            import gguf  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "gguf-py is required. Set LLAMA_CPP_GGUF_PY or PYTHONPATH=~/llama.cpp/gguf-py."
            ) from exc
    return gguf


def load_ref(path: Path):
    buf = path.read_bytes()
    if buf[:8] != b"SLDREFLG":
        raise SystemExit(f"{path}: not a SLDREFLG file")
    (version,) = struct.unpack_from("<I", buf, 8)
    if version != 1:
        raise SystemExit(f"{path}: unsupported version {version}")
    (n_tok,) = struct.unpack_from("<I", buf, 12)
    off = 16
    toks = list(struct.unpack_from(f"<{n_tok}i", buf, off))
    off += 4 * n_tok
    (n_vocab,) = struct.unpack_from("<I", buf, off)
    off += 4
    flat = struct.unpack_from(f"<{n_tok * n_vocab}f", buf, off)
    rows = [np.array(flat[i * n_vocab : (i + 1) * n_vocab], dtype=np.float64) for i in range(n_tok)]
    return toks, n_vocab, rows


# ---------------------------------------------------------------- gguf access


class Weights:
    """Reads tensors out of a GGUF file as float64 arrays.

    Quantized tensors are widened with gguf-py's `dequantize()`, which is a
    transcription of ggml's own reference `dequantize_row_*` — the same authority
    `engine/tests/quant_golden.rs` pins the Rust kernels against, bit for bit.
    That matters for what this file is *for*: it is a third opinion on the
    **forward pass**, so it must see exactly the f32 weight values both engines
    see, and differ from them only in arithmetic. Re-deriving the weights with
    an independent implementation would trade one source of noise (ggml's
    activation quantization, which is the thing being measured) for another
    (whether our dequantizer agrees), which is strictly worse.
    """

    def __init__(self, path: Path):
        self.gguf = import_gguf()
        self.reader = self.gguf.GGUFReader(path, "r")
        self.map = {}
        self.types = {}
        for t in self.reader.tensors:
            # GGUF stores dims reversed relative to numpy, and gguf-py's
            # `data` is already the flattened row-major block (raw bytes for a
            # quantized type, hence the widen below).
            shape = [int(d) for d in reversed(t.shape)]
            data = np.asarray(t.data)
            if str(t.tensor_type).startswith("F16") or str(t.tensor_type) in ("F32", "F64", "BF16"):
                arr = data.astype(np.float32).reshape(shape)
            elif str(t.tensor_type) in ("I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64"):
                arr = data.astype(np.float64).reshape(shape)
            else:
                arr = self.gguf.quants.dequantize(data, t.tensor_type).reshape(shape)
            self.map[t.name] = arr.astype(np.float64)
            self.types[t.name] = str(t.tensor_type)
        self.meta = {}
        for kv in self.reader.fields.values():
            try:
                self.meta[kv.name] = kv.contents()
            except Exception:
                pass

    def get(self, name):
        if name not in self.map:
            raise SystemExit(f"fixture is missing tensor {name}")
        return self.map[name]

    def has(self, name):
        return name in self.map

    def meta_f(self, arch, suffix, default):
        key = f"{arch}.{suffix}"
        if key in self.meta:
            v = self.meta[key]
            return float(v) if not isinstance(v, np.ndarray) else float(v.flat[0])
        return default


# --------------------------------------------------------- MoE key authority
#
# `build_moe_ffn` receives `w_scale` as a plain argument, so whether a GGUF's
# `expert_weights_scale` reaches the forward pass depends on the *architecture*:
# ~50 model files pass `hparams.expert_weights_scale` through, but only 22 ever
# call `get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, ...)`. For the rest the field keeps
# its `0.0f` initialiser whatever the file says, and 0.0 means "unset" in
# build_moe_ffn, not "scale by zero". Reading the key here unconditionally makes
# this reference agree with a *mis-implemented* engine, which is precisely the
# two-way diff that tick 16 exists to replace.
#
# The answer is not re-derived in this file: it is read from the same oracle
# `scripts/gen_moe_key_authority.py` generates and `engine/tests/moe_keys.rs`
# pins the Rust tables against, so all three agree by construction. If the JSON
# is missing, the fallback list is the 22 names as of llama.cpp 3653e6d.
_KEY_AUTHORITY_PATH = (
    Path(__file__).resolve().parents[1] / "engine/tests/data/moe_key_authority.json"
)
_EXPERT_WEIGHTS_SCALE_READERS_FALLBACK = {
    "afmoe", "bailingmoe", "bailingmoe2", "cohere2moe", "deepseek", "deepseek2",
    "deepseek2-ocr", "deepseek32", "deepseek4", "dflash", "dots1", "exaone-moe",
    "glm-dsa", "glm4moe", "hy_v3", "kimi-linear", "laguna", "minimax-m3",
    "mistral4", "nemotron_h", "nemotron_h_moe", "step35",
}


def expert_weights_scale_readers() -> set:
    if _KEY_AUTHORITY_PATH.exists():
        archs = json.loads(_KEY_AUTHORITY_PATH.read_text())["archs"]
        return {a for a, e in archs.items() if e["expert_weights_scale"]}
    return set(_EXPERT_WEIGHTS_SCALE_READERS_FALLBACK)


def w_scale_for(arch: str, value: float) -> float:
    """`w_scale` as llama.cpp would see it for this architecture."""
    return value if arch in expert_weights_scale_readers() else 0.0


# ------------------------------------------------------------- f64 primitives


def rmsnorm(x, w, eps):
    inv = 1.0 / math.sqrt(float(np.dot(x, x)) / len(x) + eps)
    return (x * inv) * w


def layernorm(x, w, b, eps):
    mean = float(np.mean(x))
    var = float(np.mean((x - mean) ** 2))
    inv = 1.0 / math.sqrt(var + eps)
    return (x - mean) * inv * w + b


def softmax(x):
    m = float(np.max(x))
    e = np.exp(x - m)
    return e / float(np.sum(e))


def rope(x, pos, head_dim, theta, neox):
    """x is (n_heads, head_dim). Returns rotated copy."""
    out = x.copy()
    n_rot = head_dim
    if neox:
        half = head_dim // 2
        for i in range(half):
            freq = 1.0 / (theta ** (2.0 * i / n_rot))
            theta_p = pos * freq
            c, s = math.cos(theta_p), math.sin(theta_p)
            for h in range(x.shape[0]):
                a = x[h, i]
                b = x[h, i + half]
                out[h, i] = a * c - b * s
                out[h, i + half] = a * s + b * c
    else:
        for i in range(0, head_dim, 2):
            freq = 1.0 / (theta ** (2.0 * (i // 2) / n_rot))
            theta_p = pos * freq
            c, s = math.cos(theta_p), math.sin(theta_p)
            for h in range(x.shape[0]):
                a = x[h, i]
                b = x[h, i + 1]
                out[h, i] = a * c - b * s
                out[h, i + 1] = a * s + b * c
    return out


def attention(q, K, V, n_heads, head_dim, n_kv_heads, scale):
    """q: (n_heads, head_dim); K, V: (T, n_kv_heads, head_dim). Returns (n_embd,).

    Only positions 0..T-1 are in the cache, so attending to all of them *is* the
    causal mask — no separate mask is needed.
    """
    kv_group = n_heads // n_kv_heads
    T = K.shape[0]
    scores = np.zeros((n_heads, T), dtype=np.float64)
    for h in range(n_heads):
        kv_h = h // kv_group
        for t in range(T):
            scores[h, t] = float(np.dot(q[h], K[t, kv_h])) * scale
    out = np.zeros(n_heads * head_dim, dtype=np.float64)
    for h in range(n_heads):
        w = softmax(scores[h])
        kv_h = h // kv_group
        for t in range(T):
            out[h * head_dim : (h + 1) * head_dim] += w[t] * V[t, kv_h]
    return out


# ------------------------------------------------------------ arch forward(s)


def forward_llama_family(w: Weights, arch: str, tokens):
    """llama / qwen2: RMSNorm pre-norm, RoPE, GQA, SwiGLU, tied output.

    The FFN branches on `ffn_gate_inp` being present, which is what
    `src/models/llama.cpp:182` does — not on a metadata flag. Every `llama`-arch
    model with `expert_count > 0` and no router tensor would take the dense
    branch in llama.cpp too, so keying off the tensor is the faithful test.
    """
    n_layer = int(w.meta_f(arch, "block_count", 0))
    n_embd = int(w.meta_f(arch, "embedding_length", 0))
    n_head = int(w.meta_f(arch, "attention.head_count", 0))
    n_kv = int(w.meta_f(arch, "attention.head_count_kv", n_head))
    eps = w.meta_f(arch, "attention.layer_norm_rms_epsilon", 1e-5)
    theta = w.meta_f(arch, "rope.freq_base", 10000.0)
    n_embd_head = n_embd // n_head
    n_expert = int(w.meta_f(arch, "expert_count", 0))
    n_expert_used = int(w.meta_f(arch, "expert_used_count", 0))
    expert_w_scale = w_scale_for(arch, w.meta_f(arch, "expert_weights_scale", 1.0))
    # ggml seeds theta = freq_scale, then *= freq_base^(-2i/n_dims)
    freq_scale = w.meta_f(arch, "rope.freq_scale", 1.0)
    neox = arch == "qwen2"  # llama_model_rope_type: qwen2 -> NEOX, llama -> NORMAL

    tok_embd = w.get("token_embd.weight")
    out_norm = w.get("output_norm.weight")
    out_w = w.map.get("output.weight", tok_embd)

    rows = []
    # K/V caches are per-layer and grow one entry per position; attention at
    # position p attends to keys 0..=p (causal).
    kcache = [[] for _ in range(n_layer)]
    vcache = [[] for _ in range(n_layer)]
    for pos, tok in enumerate(tokens):
        x = tok_embd[tok].copy()
        for il in range(n_layer):
            p = f"blk.{il}"
            t = rmsnorm(x, w.get(f"{p}.attn_norm.weight"), eps)
            qw = w.get(f"{p}.attn_q.weight")
            kw = w.get(f"{p}.attn_k.weight")
            vw = w.get(f"{p}.attn_v.weight")
            q = (qw @ t).reshape(n_head, n_embd_head)
            k = (kw @ t).reshape(n_kv, n_embd_head)
            v = (vw @ t).reshape(n_kv, n_embd_head)
            if f"{p}.attn_q.bias" in w.map:
                q = q + w.get(f"{p}.attn_q.bias").reshape(n_head, n_embd_head)
                k = k + w.get(f"{p}.attn_k.bias").reshape(n_kv, n_embd_head)
                v = v + w.get(f"{p}.attn_v.bias").reshape(n_kv, n_embd_head)
            q = rope(q, pos, n_embd_head, theta * freq_scale, neox)
            k = rope(k, pos, n_embd_head, theta * freq_scale, neox)
            kcache[il].append(k)
            vcache[il].append(v)
            K = np.stack(kcache[il])
            V = np.stack(vcache[il])
            a = attention(q, K, V, n_head, n_embd_head, n_kv, 1.0 / math.sqrt(n_embd_head))
            cur = w.get(f"{p}.attn_output.weight") @ a
            if f"{p}.attn_output.bias" in w.map:
                cur = cur + w.get(f"{p}.attn_output.bias")
            x = x + cur

            t2 = rmsnorm(x, w.get(f"{p}.ffn_norm.weight"), eps)
            if n_expert > 0 and w.has(f"{p}.ffn_gate_inp.weight"):
                x = x + moe_ffn(w, p, t2, n_expert, n_expert_used, expert_w_scale, "softmax")
            else:
                gate = w.get(f"{p}.ffn_gate.weight") @ t2
                up = w.get(f"{p}.ffn_up.weight") @ t2
                h = (gate / (1.0 + np.exp(-gate))) * up
                x = x + w.get(f"{p}.ffn_down.weight") @ h

        x = rmsnorm(x, out_norm, eps)
        rows.append(out_w @ x)
    return rows


def forward_gpt2(w: Weights, arch: str, tokens, f16_gelu: bool = False):
    """gpt2: LayerNorm w/ bias, learned positions, fused QKV, GELU, no RoPE."""
    n_layer = int(w.meta_f(arch, "block_count", 0))
    n_embd = int(w.meta_f(arch, "embedding_length", 0))
    n_head = int(w.meta_f(arch, "attention.head_count", 0))
    eps = w.meta_f(arch, "attention.layer_norm_epsilon", 1e-5)
    n_embd_head = n_embd // n_head

    tok_embd = w.get("token_embd.weight")
    pos_embd = w.get("position_embd.weight")
    out_norm = w.get("output_norm.weight")
    out_norm_b = w.get("output_norm.bias")
    out_w = w.map.get("output.weight", tok_embd)

    rows = []
    kcache = [[] for _ in range(n_layer)]
    vcache = [[] for _ in range(n_layer)]
    for pos, tok in enumerate(tokens):
        x = tok_embd[tok] + pos_embd[pos]
        for il in range(n_layer):
            p = f"blk.{il}"
            t = layernorm(x, w.get(f"{p}.attn_norm.weight"), w.get(f"{p}.attn_norm.bias"), eps)
            qkv = w.get(f"{p}.attn_qkv.weight") @ t + w.get(f"{p}.attn_qkv.bias")
            q = qkv[:n_embd].reshape(n_head, n_embd_head)
            k = qkv[n_embd : 2 * n_embd].reshape(n_head, n_embd_head)
            v = qkv[2 * n_embd :].reshape(n_head, n_embd_head)
            kcache[il].append(k)
            vcache[il].append(v)
            K = np.stack(kcache[il])
            V = np.stack(vcache[il])
            a = attention(q, K, V, n_head, n_embd_head, n_head, 1.0 / math.sqrt(n_embd_head))
            cur = w.get(f"{p}.attn_output.weight") @ a + w.get(f"{p}.attn_output.bias")
            ffn_in = x + cur

            t2 = layernorm(
                ffn_in, w.get(f"{p}.ffn_norm.weight"), w.get(f"{p}.ffn_norm.bias"), eps
            )
            h = gelu(w.get(f"{p}.ffn_up.weight") @ t2 + w.get(f"{p}.ffn_up.bias"), f16_gelu)
            cur = w.get(f"{p}.ffn_down.weight") @ h + w.get(f"{p}.ffn_down.bias")
            x = ffn_in + cur

        x = layernorm(x, out_norm, out_norm_b, eps)
        rows.append(out_w @ x)
    return rows


def forward_deepseek2(w: Weights, arch: str, tokens):
    """deepseek2: Multi-head Latent Attention with absorption, dense lead + MoE.

    Written from llama.cpp's `src/models/deepseek2.cpp` `graph::graph` graph order.
    Three things here are easy to get wrong and are the reason this function exists
    at all rather than reusing `forward_llama_family`:

    * **Q is split low-rank.** `wq_a` projects to `q_lora_rank`, that is RMSNormed by
      `attn_q_a_norm`, and only then does `wq_b` produce `n_head * key_length_mla`.
      Skipping the middle norm gives a plausible-but-wrong model.
    * **The head width is split nope/rope inside one row**, `nope` first then `rope`,
      and only the `rope` part gets positional rotation.
    * **Attention is MQA, not GQA.** `wk_b` absorbs the per-head K into Q, so all
      heads share one latent K; `wv_b` then expands the weighted latent back out.
      Running the textbook MHA over `wk_b`/`wv_b` as if they were K/V projections
      produces the same numbers but through entirely different code, which defeats
      the purpose of an independent implementation.

    `kq_scale` follows llama.cpp's pre-scaling: it divides by the *full* MLA head
    width (`key_length_mla`, i.e. nope+rope), not by the nope width.
    """
    n_layer = int(w.meta_f(arch, "block_count", 0))
    n_embd = int(w.meta_f(arch, "embedding_length", 0))
    n_head = int(w.meta_f(arch, "attention.head_count", 0))
    eps = w.meta_f(arch, "attention.layer_norm_rms_epsilon", 1e-5)
    n_lora_q = int(w.meta_f(arch, "attention.q_lora_rank", 0))
    n_lora_kv = int(w.meta_f(arch, "attention.kv_lora_rank", 0))
    key_length = int(w.meta_f(arch, "attention.key_length_mla", 0))
    value_length = int(w.meta_f(arch, "attention.value_length_mla", 0))
    rope_dim = int(w.meta_f(arch, "rope.dimension_count", 0))
    theta = w.meta_f(arch, "rope.freq_base", 10000.0)
    freq_scale = w.meta_f(arch, "rope.freq_scale", 1.0)
    attn_factor = w.meta_f(arch, "rope.scaling.attn_factor", 1.0)
    yarn_log_mul = w.meta_f(arch, "rope.scaling.yarn_log_multiplier", 0.0)
    rope_scaling = w.meta_f(arch, "rope.scaling.factor", 1.0)
    n_ff = int(w.meta_f(arch, "feed_forward_length", 0))
    n_layer_dense_lead = int(w.meta_f(arch, "leading_dense_block_count", 0))
    n_expert = int(w.meta_f(arch, "expert_count", 0))
    n_expert_used = int(w.meta_f(arch, "expert_used_count", 1))
    n_ff_exp = int(w.meta_f(arch, "expert_feed_forward_length", 0))
    n_expert_shared = int(w.meta_f(arch, "expert_shared_count", 0))
    norm_w = w.meta_f(arch, "expert_weights_norm", 0.0) != 0.0
    w_scale = w_scale_for(arch, w.meta_f(arch, "expert_weights_scale", 0.0))

    if key_length <= rope_dim:
        raise SystemExit("deepseek2: key_length_mla must exceed rope.dimension_count")

    qk_nope = key_length - rope_dim

    # [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]: llama.cpp divides by 0.1 at load time to
    # cancel the factor baked in by the convert script.
    if yarn_log_mul != 0.0:
        yarn_log_mul /= 0.1
    log_inv_scale = math.log(1.0 / (rope_scaling * freq_scale))
    attn_factor_org = attn_factor * (1.0 + 0.1 * log_inv_scale)
    mscale = attn_factor_org * (1.0 + 0.1 * yarn_log_mul * log_inv_scale)
    kq_scale = mscale * mscale / math.sqrt(float(key_length))

    tok_embd = w.get("token_embd.weight")
    out_norm = w.get("output_norm.weight")
    out_w = w.map.get("output.weight", tok_embd)

    rows = []
    kcache = [[] for _ in range(n_layer)]  # latent K = [kv_cmpr, k_pe]
    for pos, tok in enumerate(tokens):
        x = tok_embd[tok].copy()
        for il in range(n_layer):
            p = f"blk.{il}"
            cur = rmsnorm(x, w.get(f"{p}.attn_norm.weight"), eps)

            if n_lora_q > 0:
                qa = rmsnorm(w.get(f"{p}.attn_q_a.weight") @ cur, w.get(f"{p}.attn_q_a_norm.weight"), eps)
                q = w.get(f"{p}.attn_q_b.weight") @ qa
            else:
                q = w.get(f"{p}.attn_q.weight") @ cur
            q = q.reshape(n_head, key_length)
            q_nope = q[:, :qk_nope]
            q_pe = rope(q[:, qk_nope:], pos, rope_dim, theta * freq_scale, neox=False)

            kv_cmpr_pe = w.get(f"{p}.attn_kv_a_mqa.weight") @ cur
            kv_cmpr = rmsnorm(kv_cmpr_pe[:n_lora_kv], w.get(f"{p}.attn_kv_a_norm.weight"), eps)
            # k_pe is a single shared rope block (one "head"), not per-head like q_pe.
            k_pe = rope(kv_cmpr_pe[n_lora_kv:][None, :], pos, rope_dim, theta * freq_scale, neox=False)[0]

            # wk_b / wv_b are 3D in GGUF: (qk_nope, kv_lora, head) and
            # (kv_lora, v_head, head). Read them as per-head 2D slices.
            wk_b = w.get(f"{p}.attn_k_b.weight")
            wv_b = w.get(f"{p}.attn_v_b.weight")
            kcache[il].append(np.concatenate([kv_cmpr, k_pe]))
            K = np.stack(kcache[il])  # (T, kv_lora + rope)

            attn = np.zeros((n_head, value_length), dtype=np.float64)
            for h in range(n_head):
                absorbed = np.einsum("ki,i->k", wk_b[h], q_nope[h])
                Qcur = np.concatenate([absorbed, q_pe[h]])
                scores = (K @ Qcur) * kq_scale
                wgt = softmax(scores)
                cmpr = wgt @ K[:, :n_lora_kv]
                attn[h] = wv_b[h] @ cmpr
            cur = w.get(f"{p}.attn_output.weight") @ attn.reshape(-1)
            ffn_inp = x + cur

            cur = rmsnorm(ffn_inp, w.get(f"{p}.ffn_norm.weight"), eps)
            if il < n_layer_dense_lead:
                gate = w.get(f"{p}.ffn_gate.weight") @ cur
                up = w.get(f"{p}.ffn_up.weight") @ cur
                h_act = (gate / (1.0 + np.exp(-gate))) * up
                cur = w.get(f"{p}.ffn_down.weight") @ h_act
            else:
                router = w.get(f"{p}.ffn_gate_inp.weight") @ cur
                all_probs = softmax(router)
                order = np.argsort(-all_probs, kind="stable")[:n_expert_used]
                sel = all_probs[order]
                # deepseek2 is in the group that *forwards* both
                # `hparams.expert_weights_norm` and `hparams.expert_weights_scale`
                # and *reads* both keys, so unlike `llama` the metadata is live
                # here. Omitting either silently makes this reference agree with
                # an engine that also omits it, which is the two-way diff this
                # script exists to avoid -- DeepSeek-V2's real
                # `routed_scaling_factor` is 16.0.
                if norm_w:
                    total = float(np.sum(sel))
                    if total < 6.103515625e-5:
                        total = 6.103515625e-5
                    sel = sel / total
                if w_scale != 0.0 and w_scale != 1.0:
                    sel = sel * w_scale
                out = np.zeros(n_embd, dtype=np.float64)
                for weight, e in zip(sel, order):
                    gate_m, up_m = expert_gate_up(w, p, e, n_ff_exp)
                    up = up_m @ cur
                    gate = gate_m @ cur
                    h_act = (gate / (1.0 + np.exp(-gate))) * up
                    out = out + weight * (w.get(f"{p}.ffn_down_exps.weight")[e] @ h_act)
                if n_expert_shared > 0:
                    up = w.get(f"{p}.ffn_up_shexp.weight") @ cur
                    gate = w.get(f"{p}.ffn_gate_shexp.weight") @ cur
                    h_act = (gate / (1.0 + np.exp(-gate))) * up
                    out = out + w.get(f"{p}.ffn_down_shexp.weight") @ h_act
                cur = out
            x = ffn_inp + cur

        x = rmsnorm(x, out_norm, eps)
        rows.append(out_w @ x)
    return rows


def forward_gemma3(w: Weights, arch: str, tokens, f16_gelu: bool = False):
    """gemma3: SWA/global alternating attention, per-head QK-norm, post-norms, GEGLU.

    Written from llama.cpp's `src/models/gemma3.cpp` `graph<iswa>::graph` in file order,
    including the three hparams defaults the converter never writes (see
    `gen_gemma3` in scripts/generate_synthetic_fixtures.py for the full table):

        swa_period            = 6                    gemma3.cpp:8
        rope_freq_base_train_swa = 10000.0           llama-hparams.h:127
        n_embd_head_{k,v}_swa = n_embd_head_{k,v}_full  llama-model.cpp:1219

    Graph order, and the four things in it that a llama-shaped reading gets wrong:

    * `inpL = embd * sqrt(n_embd)` — the Gemma embedding scale. It is the single
      largest numerical difference in the whole architecture and it is one line.
    * `attn_output.weight` is `{n_embd_head_k * n_head, n_embd}`, so the attention
      output is `n_head * key_length` wide, which is 56 here against an
      `embedding_length` of 48. Indexing it as if it were `n_embd` wide is wrong on
      every Gemma-3 checkpoint released so far.
    * `Qcur = ggml_scale(Qcur, f_attention_scale)` happens **after** RoPE and
      **after** the QK-norm, and only on Q.
    * `attn_post_norm` / `ffn_post_norm` normalise the *branch output* before it is
      added to the residual, so the residual is `post_norm(attn) + x`, not
      `post_norm(attn + x)`.
    * RoPE pairing is **NEOX, not NORM** (`llama_model_rope_type` returns
      `LLAMA_ROPE_TYPE_NEOX` for gemma3 — the `llama.h` enum value is 2, and
      `print_info: rope type = 2` says so out loud). NEOX pairs element `i` with
      element `i + n_rot/2`; NORM pairs `2i` with `2i+1`. This is invisible at
      position 0, where the rotation is the identity, and the fixture only
      disagrees with llama.cpp from position 1 on — which is exactly the shape of a
      pairing bug and the reason `PROMPT` in `tests/ref_parity.rs` is 8 tokens
      rather than 1.

    `f_attention_scale` is not read from metadata: gemma3.cpp *computes* it, and
    only the 27B checkpoint (`n_layer == 62`) uses `1/sqrt(n_embd/n_head)` instead of
    `1/sqrt(n_embd_head_k(0))`. Reproduced here from `n_layer` alone, because an
    engine that trusts a `gemma3.attention.scale` key is trusting a key llama.cpp
    ignores.

    The FFN is `LLM_FFN_GELU` with `LLM_FFN_PAR`, so `build_ffn` puts the gate
    projection in `cur` and the up projection in `tmp`, and `ggml_geglu_split(cur, tmp)`
    is `gelu(gate) * up` — the gelu is on the *gate*, not on the up. Swapping the two
    is invisible on a symmetric activation and is not on this one.
    """
    n_layer = int(w.meta_f(arch, "block_count", 0))
    n_embd = int(w.meta_f(arch, "embedding_length", 0))
    n_head = int(w.meta_f(arch, "attention.head_count", 0))
    n_kv = int(w.meta_f(arch, "attention.head_count_kv", n_head))
    eps = w.meta_f(arch, "attention.layer_norm_rms_epsilon", 1e-5)
    theta = w.meta_f(arch, "rope.freq_base", 10000.0)
    theta_swa = w.meta_f(arch, "rope.freq_base_swa", 10000.0)
    # `get_rope_freq_scale` returns hparams.rope_freq_scale_train_swa for SWA layers
    # and cparams.rope_freq_scale for global ones. There is no GGUF key for the SWA
    # one in this llama.cpp (no LLM_KV_ROPE_FREQ_SCALE_SWA in llama-arch.cpp), so it
    # is always 1.0 and only the global half is scaled.
    freq_scale = w.meta_f(arch, "rope.freq_scale", 1.0)
    head_dim = int(w.meta_f(arch, "attention.key_length", n_embd // n_head))
    n_swa = int(w.meta_f(arch, "attention.sliding_window", 0))
    # gemma3.cpp:8 — `uint32_t swa_period = 6` then get_key_or_arr(... , false).
    swa_period = int(w.meta_f(arch, "attention.sliding_window_pattern", 6))
    softcap = w.meta_f(arch, "final_logit_softcapping", 0.0)
    # gemma3.cpp:76 — computed, not read. Only LLM_TYPE_27B (n_layer == 62) takes the
    # n_embd/n_head(0) branch; everything else is 1/sqrt(n_embd_head_k).
    attn_scale = (
        1.0 / math.sqrt(n_embd / n_head) if n_layer == 62 else 1.0 / math.sqrt(head_dim)
    )

    def is_swa(il: int) -> bool:
        # llama_hparams::set_swa_pattern(n, dense_first=false):
        #   is_swa(il) = n == 0 || (il % n < n - 1)
        # note `n == 0` means *every* layer is SWA, which is a legal value for the
        # pattern key and the opposite of "unset".
        if n_swa <= 0:
            return False
        return swa_period == 0 or (il % swa_period < swa_period - 1)

    tok_embd = w.get("token_embd.weight")
    out_norm = w.get("output_norm.weight")
    out_w = w.map.get("output.weight", tok_embd)
    embd_scale = math.sqrt(float(n_embd))

    rows = []
    kcache = [[] for _ in range(n_layer)]
    vcache = [[] for _ in range(n_layer)]
    for pos, tok in enumerate(tokens):
        x = tok_embd[tok] * embd_scale
        for il in range(n_layer):
            p = f"blk.{il}"
            layer_theta = theta_swa if is_swa(il) else theta * freq_scale
            t = rmsnorm(x, w.get(f"{p}.attn_norm.weight"), eps)
            q = (w.get(f"{p}.attn_q.weight") @ t).reshape(n_head, head_dim)
            k = (w.get(f"{p}.attn_k.weight") @ t).reshape(n_kv, head_dim)
            v = (w.get(f"{p}.attn_v.weight") @ t).reshape(n_kv, head_dim)
            # One weight vector, applied to every head. ggml normalises the whole
            # {n_embd_head_k * n_head, n_embd_head_k} view in one shot, which is
            # exactly a per-head RMSNorm with a shared weight.
            qw = w.get(f"{p}.attn_q_norm.weight")
            kw = w.get(f"{p}.attn_k_norm.weight")
            q = np.stack([rmsnorm(q[h], qw, eps) for h in range(n_head)])
            k = np.stack([rmsnorm(k[h], kw, eps) for h in range(n_kv)])
            q = rope(q, pos, head_dim, layer_theta, neox=True) * attn_scale
            k = rope(k, pos, head_dim, layer_theta, neox=True)
            kcache[il].append(k)
            vcache[il].append(v)
            K = np.stack(kcache[il])
            V = np.stack(vcache[il])
            if is_swa(il):
                # An SWA layer sees only the last n_swa positions of the same
                # stream; llama.cpp gets this by giving the iswa graph its own
                # n_swa-cell cache that is shifted each step.
                keep = min(n_swa, pos + 1)
                K = K[pos + 1 - keep :]
                V = V[pos + 1 - keep :]
            a = attention(q, K, V, n_head, head_dim, n_kv, 1.0)
            # n_head * head_dim == 56 here, not n_embd.
            cur = w.get(f"{p}.attn_output.weight") @ a.reshape(-1)
            cur = rmsnorm(cur, w.get(f"{p}.post_attention_norm.weight"), eps)
            sa_out = x + cur

            t2 = rmsnorm(sa_out, w.get(f"{p}.ffn_norm.weight"), eps)
            gate = w.get(f"{p}.ffn_gate.weight") @ t2
            up = w.get(f"{p}.ffn_up.weight") @ t2
            h = gelu(gate, f16_gelu) * up
            cur = w.get(f"{p}.ffn_down.weight") @ h
            cur = rmsnorm(cur, w.get(f"{p}.post_ffw_norm.weight"), eps)
            x = sa_out + cur

        x = rmsnorm(x, out_norm, eps)
        logits = out_w @ x
        if softcap:
            logits = np.tanh(logits / softcap) * softcap
        rows.append(logits)
    return rows


def expert_gate_up(w: Weights, p: str, e: int, n_ff_exp: int):
    """One expert's gate and up projection matrices, in either container form.

    llama.cpp's `create_tensor_gate_up_exps` (llama-model.cpp:2881) prefers a single
    `ffn_gate_up_exps` = `{n_embd, n_ff_exp*2, n_expert}` tensor and only creates the
    split `ffn_{gate,up}_exps` pair when the fused one is absent. `build_moe_ffn` then
    takes two views of a single matmul result (llama-graph.cpp:2100-2107):

        cur = ggml_view_3d(gate_up, n_ff, ..., nb[1], nb[2], 0)             # gate
        up  = ggml_view_3d(gate_up, n_ff, ..., nb[1], nb[2], n_ff*nb[0])   # up

    so expert `e` owns rows `[2*e*n_ff_exp, 2*e*n_ff_exp + 2*n_ff_exp)` with **gate
    rows first, then up rows** -- which is what `convert_hf_to_gguf.py
    --fuse-gate-up-exps` writes via `torch.cat([gate, up], dim=1)`.

    That order is measured, not assumed: generating the fused fixture with the
    concatenation reversed (`[up, gate]`) changes llama.cpp's own logits, so a
    reference that got it backwards would be visibly wrong rather than silently
    agreeing with a second wrong implementation.

    Returns `(gate_matrix, up_matrix)`, each `n_ff_exp x n_embd` in numpy order.
    """
    fused = f"{p}.ffn_gate_up_exps.weight"
    if w.has(fused):
        block = w.get(fused)[e]
        if block.shape[0] != 2 * n_ff_exp:
            raise SystemExit(
                f"{fused}: expert {e} has {block.shape[0]} rows, expected "
                f"2*n_ff_exp = {2 * n_ff_exp}"
            )
        return block[:n_ff_exp], block[n_ff_exp:]
    return w.get(f"{p}.ffn_gate_exps.weight")[e], w.get(f"{p}.ffn_up_exps.weight")[e]


def moe_ffn(w: Weights, p: str, x, n_expert, n_expert_used, w_scale, gating: str):
    """llama.cpp `llm_graph_context::build_moe_ffn`, in graph order.

    Written from `llama-graph.cpp:1914`, not from the Rust engine, because the
    MoE FFN is where three separate normalisations could hide and only one of
    them is in the engine:

    * **The router softmax is over *all* experts**, and the per-expert weights
      are `get_rows(probs, selected)` — i.e. the selected experts' *already
      normalised* probabilities, never a softmax over the selected subset.
      `build_moe_ffn` only re-softmaxes when `gating_op ==
      SOFTMAX_WEIGHT` (Mixtral, OLMoE), which is a different arch string.
    * **`norm_w` is `true` for the `llama` arch** (`llama.cpp:207` passes a
      literal `true`, not a metadata flag), so the selected weights are divided
      by their own sum, clamped at the smallest f16 `6.103515625e-5`. When
      `n_expert_used == n_expert` that division is a no-op up to one ulp,
      because the softmax already sums to 1 — which is exactly why Tiny-MoE
      cannot tell you whether you implemented it, and why it is written down.
    * **The expert weights are applied to the expert *outputs*, not to the
      inputs** (`ggml_mul(experts, weights)` after `down_exps`). Only
      `LLM_ARCH_LLAMA4` does it the other way (`weight_before_ffn`), and that is
      a per-arch `const` in llama.cpp rather than a config knob.

    `ggml_argsort_top_k` breaks ties towards the lower index; the reference uses
    a stable descending sort, which is the same tie-break.
    """
    if gating != "softmax":
        raise SystemExit(f"moe_ffn: only softmax gating is transcribed, got {gating!r}")
    router = w.get(f"{p}.ffn_gate_inp.weight") @ x
    probs = softmax(router)
    order = np.argsort(-probs, kind="stable")[:n_expert_used]
    sel = probs[order]
    total = float(np.sum(sel))
    if total < 6.103515625e-5:
        total = 6.103515625e-5
    sel = sel / total
    if w_scale != 0.0 and w_scale != 1.0:
        sel = sel * w_scale
    out = np.zeros_like(x)
    down_exps = w.get(f"{p}.ffn_down_exps.weight")
    n_ff_exp = w.get(f"{p}.ffn_down_exps.weight").shape[0]
    for weight, e in zip(sel, order):
        gate_m, up_m = expert_gate_up(w, p, e, n_ff_exp)
        up = up_m @ x
        gate = gate_m @ x
        h = (gate / (1.0 + np.exp(-gate))) * up  # ggml_swiglu_split(gate, up)
        out = out + weight * (down_exps[e] @ h)
    return out


FORWARD = {
    "llama": forward_llama_family,
    "qwen2": forward_llama_family,
    "gpt2": forward_gpt2,
    "deepseek2": forward_deepseek2,
    "gemma3": forward_gemma3,
}


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(FORWARD))
    ap.add_argument("--gguf", type=Path, default=None)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--llama", type=Path, required=True)
    ap.add_argument(
        "--f16-gelu",
        action="store_true",
        help="round GELU through float16 like ggml's lookup table "
        "(GGML_GELU_FP16, ggml-cpu/vec.h:46)",
    )
    ap.add_argument(
        "--emit-ref",
        type=Path,
        default=None,
        help="write this float64 forward pass as a SLDREFLG file, for tests/ref_parity.rs",
    )
    args = ap.parse_args()

    gguf = args.gguf or REPO / "engine" / "tests" / "data" / "fixtures" / f"synthetic-{args.arch}.gguf"
    etoks, evocab, erows = load_ref(args.engine)
    rtoks, rvocab, rrows = load_ref(args.llama)
    if etoks != rtoks:
        raise SystemExit(f"token sequences differ: {etoks} vs {rtoks}")

    w = Weights(gguf)
    fn = FORWARD[args.arch]
    f64rows = fn(w, args.arch, etoks, args.f16_gelu) if args.f16_gelu else fn(w, args.arch, etoks)
    if args.f16_gelu:
        print("   (modelling ggml's float16 GELU lookup table)")

    print(f"== {args.arch}: {len(etoks)} tokens, vocab {evocab} ==")
    print("     rel L2 error vs an independent float64 forward pass")
    worst_e = worst_l = 0.0
    for i, (e, r, f) in enumerate(zip(erows, rrows, f64rows)):
        de = rel_err(e, f)
        dl = rel_err(r, f)
        worst_e = max(worst_e, de)
        worst_l = max(worst_l, dl)
        ae = int(np.argmax(e))
        ar = int(np.argmax(r))
        af = int(np.argmax(f))
        print(
            f"  pos {i:>3}  engine={de:.3e}  llama={dl:.3e}  "
            f"ratio={de / dl if dl else float('nan'):>6.1f}x  "
            f"argmax e/l/f = {ae}/{ar}/{af}"
        )
    verdict = "engine is closer" if worst_e < worst_l else "llama.cpp is closer"
    if abs(worst_e - worst_l) <= 1e-12:
        verdict = "tie"
    print(f"   worst: engine={worst_e:.3e} llama={worst_l:.3e}  ->  {verdict}")

    if args.emit_ref:
        # SLDREFLG v1, same layout tools/gen_ref_logits.c writes, so the Rust
        # test reads both files with one parser. This is the *exact-math*
        # oracle: unlike llama.cpp's own forward pass it carries none of the
        # f16 table / f16 K-V cast error, so the gate built on it is tight
        # enough to catch a wrong activation constant rather than only a wrong
        # tensor layout.
        out = bytearray(b"SLDREFLG")
        out += struct.pack("<I", 1)
        out += struct.pack("<I", len(etoks))
        for t in etoks:
            out += struct.pack("<i", int(t))
        out += struct.pack("<I", evocab)
        for row in f64rows:
            for v in row[:evocab]:
                out += struct.pack("<f", float(v))
        args.emit_ref.write_bytes(bytes(out))
        print(f"   wrote {args.emit_ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())