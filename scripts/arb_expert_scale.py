#!/usr/bin/env python3
"""Which MoE hparam keys reach the forward pass, and what that costs you?

`build_moe_ffn` receives `norm_w` and `w_scale` as plain arguments, so whether a
GGUF's `{arch}.expert_weights_norm` / `{arch}.expert_weights_scale` means anything
is decided per architecture, in the *loader* rather than the graph:

* ~50 model files forward `hparams.expert_weights_scale` to `build_moe_ffn`, but
  only **22 of 137** ever call `get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, ...)`. For
  the rest the field keeps its `0.0f` initialiser whatever the file says — and
  `build_moe_ffn` reads `0.0` as "no scale", not "scale by zero".
* 28 architectures pass a **literal** `true` for `norm_w` and 12 a literal
  `false`; the remaining 20 forward the key, and are exactly the 20 that read it.

So an engine that reads either key unconditionally computes a different model on
115 of 137 architectures, and one that hard-codes `norm_w` wrong way computes a
different model on 40. Both failures are silent: no error, plausible text.

This script measures the four things that make the claim real, all by running
llama.cpp rather than by reading it:

1. **Who reads each key.** From `engine/tests/data/moe_key_authority.json`, which
   `scripts/gen_moe_key_authority.py` derives from a llama.cpp checkout.
2. **Is the key observable at all on the `llama` arch?** Two fixtures differing
   only in `{arch}.expert_weights_scale = 2.5`; llama.cpp's logits are compared
   as **bytes**, so "the arch ignores the key" is a statement about llama.cpp's
   own output rather than about a tolerance.
3. **How far apart are honouring and ignoring it?** The engine run with the key
   ignored (correct) against the same weights with it applied (the defect this
   tick fixed), as relative L2. Reported for `llama` (ignored upstream, 6.8e-1)
   and for `deepseek2` (honoured upstream, so applying it is *correct*).
4. **Same question for `expert_weights_norm`.** Whether honouring the flag changes
   the logits at all on `deepseek2`, and by how much.

    python3 scripts/arb_expert_scale.py --json results/expert_key_authority.json

Needs a built llama.cpp (`$LLAMA_CPP/build/bin/libllama.so`) for the oracle
columns; `--no-llama` runs the engine and float64 columns alone.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ENGINE_BIN = REPO / "engine" / "target" / "release" / "slow-inference"
ORACLE = REPO / "engine" / "tests" / "data" / "moe_key_authority.json"
sys.path.insert(0, str(HERE))
import arbitrate_logits as A  # noqa: E402
import generate_synthetic_fixtures as G  # noqa: E402

SEED = 0x5EED
PROMPT = "3,17,42,99,7,61,127,24"


def rel_l2(a: Path, b: Path) -> float | None:
    """Worst per-position relative L2 between two SLDREFLG files."""
    if not (a.exists() and b.exists()):
        return None
    x, y = A.load_ref(a)[2], A.load_ref(b)[2]
    if len(x) != len(y):
        return None
    return max(A.rel_err(p, q) for p, q in zip(x, y))


def build_gen_ref(llama_cpp: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gcc", "-O2", "-std=c11", "-o", str(dest),
            f"-I{llama_cpp}/include", f"-I{llama_cpp}/ggml/include",
            str(REPO / "engine/tools/gen_ref_logits.c"),
            f"-L{llama_cpp}/build/bin", "-lllama", "-lggml", "-lggml-base", "-lm",
            f"-Wl,-rpath,{llama_cpp}/build/bin",
        ],
        check=True,
    )
    return dest


def llama_run(bin_: Path, gguf: Path, out: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        [str(bin_), str(gguf), str(out), PROMPT],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stderr.strip().splitlines()[-1] if proc.returncode else ""


def f64_run(arch: str, gguf: Path, ref: Path, out: Path) -> None:
    subprocess.run(
        [
            sys.executable, str(HERE / "arbitrate_logits.py"),
            "--arch", arch, "--gguf", str(gguf),
            "--engine", str(ref), "--llama", str(ref),
            "--emit-ref", str(out),
        ],
        check=True,
        capture_output=True,
    )


def f64_rows(arch: str, gguf: Path, ref: Path, readers=None):
    """The float64 forward pass, in-process, with the reader set optionally forced.

    In-process because the question is "what if this architecture *were* a reader",
    and the only way to ask it is to change the reader set for one run. Shelling
    out to `arbitrate_logits.py` cannot do that, which is how the first version of
    this script measured 0.0 for a difference that is really 0.68.
    """
    saved = A.expert_weights_scale_readers
    if readers is not None:
        A.expert_weights_scale_readers = lambda: set(readers)
    try:
        toks = A.load_ref(ref)[0]
        return A.FORWARD[arch](A.Weights(gguf), arch, toks)
    finally:
        A.expert_weights_scale_readers = saved


def worst_rel_l2(x, y) -> float:
    return max(A.rel_err(np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64))
               for p, q in zip(x, y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--no-llama", action="store_true", help="skip the llama.cpp oracle")
    args = ap.parse_args()

    llama_cpp = Path(os.environ.get("LLAMA_CPP", str(Path.home() / "llama.cpp")))
    if not args.no_llama and not (llama_cpp / "build" / "bin" / "libllama.so").exists():
        raise SystemExit(f"no {llama_cpp}/build/bin/libllama.so — build it, or pass --no-llama")
    if not ENGINE_BIN.exists():
        raise SystemExit(f"no {ENGINE_BIN} — `cargo build --release` first")

    oracle = json.loads(ORACLE.read_text())
    archs = oracle["archs"]
    routing = {a: e for a, e in archs.items() if e["builds_moe"]}
    scale_readers = sorted(a for a, e in archs.items() if e["expert_weights_scale"])
    norm_readers = sorted(a for a, e in archs.items() if e["expert_weights_norm"])
    norm_literal_true = sorted(a for a, e in routing.items() if e["norm_w"] == ["literal_true"])
    norm_literal_false = sorted(a for a, e in routing.items() if e["norm_w"] == ["literal_false"])
    norm_key = sorted(a for a, e in routing.items()
                      if e["norm_w"] == ["key:expert_weights_norm"])

    out: dict[str, object] = {
        "llama_cpp_commit": oracle["llama_cpp_commit"],
        "prompt": PROMPT,
        "seed": hex(SEED),
        "counts": {
            "architectures_in_oracle": len(archs),
            "architectures_calling_build_moe_ffn": len(routing),
            "read_expert_weights_scale": len(scale_readers),
            "read_expert_weights_norm": len(norm_readers),
            "norm_w_literal_true": len(norm_literal_true),
            "norm_w_literal_false": len(norm_literal_false),
            "norm_w_from_key": len(norm_key),
        },
        "lists": {
            "expert_weights_scale_readers": scale_readers,
            "norm_w_literal_true": norm_literal_true,
            "norm_w_literal_false": norm_literal_false,
            "norm_w_from_key": norm_key,
            "forward_w_scale_but_do_not_read_it": sorted(
                a for a, e in routing.items()
                if e["w_scale"] == ["key:expert_weights_scale"]
                and not e["expert_weights_scale"]
            ),
        },
    }
    print("== 1. who reads each key (llama.cpp {}) ==".format(oracle["llama_cpp_commit"]))
    for label, n, names in (
        ("expert_weights_scale", len(scale_readers), scale_readers),
        ("expert_weights_norm ", len(norm_readers), norm_readers),
    ):
        print(f"   {label}: {n}/{len(archs)} architectures -> {' '.join(names)}")
    print(f"   norm_w passed as a literal true : {len(norm_literal_true)}")
    print(f"   norm_w passed as a literal false: {len(norm_literal_false)}")
    print(f"   norm_w passed as the key        : {len(norm_key)}")
    print(
        "   forward w_scale but never read it: "
        f"{out['lists']['forward_w_scale_but_do_not_read_it']}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        gguf = G.import_gguf()
        bin_ = None if args.no_llama else build_gen_ref(llama_cpp, td / "gen_ref_logits")

        lm = td / "llama-moe.gguf"
        lmw = td / "llama-moe-wscale.gguf"
        ds = td / "deepseek2.gguf"
        dsn = td / "deepseek2-normw.gguf"
        G.gen_llama_moe(lm, gguf, SEED)
        G.gen_llama_moe(lmw, gguf, SEED, expert_weights_scale=2.5)
        G.gen_deepseek2(ds, gguf, SEED)
        G.gen_deepseek2(dsn, gguf, SEED, expert_weights_norm=True)

        measured: dict[str, dict] = {}

        print()
        print("== 2/3. expert_weights_scale on the `llama` arch (which ignores it) ==")
        lm_ref, lmw_ref = td / "lm.ref", td / "lmw.ref"
        if bin_ is not None:
            ok, err = llama_run(bin_, lm, lm_ref)
            if not ok:
                print(f"   llama-moe: llama.cpp refused the fixture: {err}")
            llama_run(bin_, lmw, lmw_ref)
            same = lm_ref.read_bytes() == lmw_ref.read_bytes() if lm_ref.exists() else None
            print(f"   llama.cpp logits identical with and without the key: {same}")
            out["llama_arch_llama_cpp_ignores_the_key"] = same
        for name, src, ref in (("llama-moe", lm, lm_ref), ("llama-moe-wscale", lmw, lmw_ref)):
            got = td / f"{name}.engine.ref"
            subprocess.run(
                # `--prompt-tokens` rather than `--prompt`: the synthetic
                # fixtures carry a vocab but no usable merges table, so the CLI's
                # tokenizer refuses them. The library path (`prefill_logits`) does
                # not need one, which is what the parity tests use.
                [str(ENGINE_BIN), "--model", str(src), "--prompt-tokens", PROMPT,
                 "--dump-logits", str(got)],
                check=True, capture_output=True,
            )
            measured[name] = {"engine_vs_llama_cpp": rel_l2(got, ref) if ref.exists() else None}
            d = measured[name]["engine_vs_llama_cpp"]
            print(f"   engine({name:18}) vs llama.cpp: {'n/a' if d is None else f'{d:.3e}'}")
        # The defect, measured: the same weights with the key applied instead of
        # ignored. Both runs read the *same* fixture, so the only difference is
        # whether `llama` is in the reader set -- which is exactly the question.
        # The fixture has to be the one that *carries* the key: on `llama-moe` the
        # key is absent, `meta_f` falls back to 1.0, and build_moe_ffn skips 1 --
        # so both readings would agree and the measurement would be 0.0 for a
        # difference that is really 0.68. (It was, on the first run.)
        base = lmw if lmw_ref.exists() else lm
        ignored_rows = f64_rows("llama", base, lmw_ref if lmw_ref.exists() else lm_ref, set())
        applied_rows = f64_rows("llama", base, lmw_ref if lmw_ref.exists() else lm_ref, {"llama"})
        defect = worst_rel_l2(ignored_rows, applied_rows)
        out["llama_arch_honouring_the_key_moves_logits_by_rel_l2"] = defect
        print(f"   same weights, key ignored vs applied: rel L2 {defect:.3e}")

        print()
        print("== 4. expert_weights_norm on the `deepseek2` arch (which honours it) ==")
        ds_ref, dsn_ref = td / "ds.ref", td / "dsn.ref"
        if bin_ is not None:
            ok, err = llama_run(bin_, ds, ds_ref)
            if not ok:
                print(f"   deepseek2: llama.cpp refused the fixture: {err}")
            llama_run(bin_, dsn, dsn_ref)
            moved = rel_l2(ds_ref, dsn_ref)
            out["deepseek2_norm_w_moves_llama_cpp_by_rel_l2"] = moved
            print(f"   llama.cpp logits, expert_weights_norm false -> true: {moved:.3e}")
        for name, src, ref in (("deepseek2", ds, ds_ref), ("deepseek2-normw", dsn, dsn_ref)):
            got = td / f"{name}.engine.ref"
            subprocess.run(
                # `--prompt-tokens` rather than `--prompt`: the synthetic
                # fixtures carry a vocab but no usable merges table, so the CLI's
                # tokenizer refuses them. The library path (`prefill_logits`) does
                # not need one, which is what the parity tests use.
                [str(ENGINE_BIN), "--model", str(src), "--prompt-tokens", PROMPT,
                 "--dump-logits", str(got)],
                check=True, capture_output=True,
            )
            measured[name] = {"engine_vs_llama_cpp": rel_l2(got, ref) if ref.exists() else None}
            d = measured[name]["engine_vs_llama_cpp"]
            print(f"   engine({name:18}) vs llama.cpp: {'n/a' if d is None else f'{d:.3e}'}")
        out["engine_vs_llama_cpp"] = measured

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
