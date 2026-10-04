#!/usr/bin/env python3
"""Is `ffn_gate_up_exps` -- the fused MoE expert tensor -- a different model?

llama.cpp's `create_tensor_gate_up_exps` (llama-model.cpp:2881) prefers a single
`ffn_gate_up_exps` of shape `{n_embd, n_ff*2, n_expert}` and only creates the
split `ffn_{gate,up}_exps` pair when it is absent. `build_moe_ffn` then takes two
`ggml_view_3d`s of one matmul result (llama-graph.cpp:2100-2107): gate at row
offset 0, up at `n_ff*nb[0]`. So per expert the rows are **gate first, then up**.

That is a three-line claim with three ways of being wrong, and this script
settles each by running llama.cpp rather than by reading it:

1. **Which architectures can carry the fused tensor at all.** The `llama` arch
   does not call the helper -- `src/models/llama.cpp:78-80` creates the split
   pair directly and passes `flags = 0` for `ffn_up_exps`, i.e. required -- so a
   `llama`-arch file holding only the fused tensor should be rejected at load.
2. **Is gate-then-up the right order?** A third file is written with the
   concatenation reversed, `[up, gate]`, and llama.cpp's logits are compared.
   If the two orders were interchangeable the fused tensor would be a pure
   rename and this would be a naming detail rather than a load-bearing layout.
3. **Do the two container forms compute the same function?** For llama.cpp and
   for the engine, compared as raw bytes rather than through a tolerance.

(3) is the one that pins the order. `Model::expert` resolves the row layout
before doing any arithmetic, so both paths run the same float operations over
the same values and the identity is exact; an exact assertion is the only thing
that can tell "read the rows in the right order" from "read them in a nearby
wrong order". A tolerance gate passes both.

    python3 scripts/arb_fused_gate_up.py --json results/fused_gate_up_exps.json

Needs a built llama.cpp (`$LLAMA_CPP/build/bin/libllama.so`) for the oracle
columns; `--no-llama` runs the engine side alone.
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
ENGINE = REPO / "engine" / "target" / "release" / "slow-inference"
GEN_REF = REPO / "engine" / "tools" / "gen_ref_logits.c"

sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("LLAMA_CPP_GGUF_PY", str(Path.home() / "llama.cpp" / "gguf-py")))

import arbitrate_logits as A  # noqa: E402
import generate_synthetic_fixtures as G  # noqa: E402

PROMPT = [3, 17, 42, 99, 7, 61, 127, 24]
SEED = 0x5EED


def build_gen_ref(llama_cpp: Path, out: Path) -> Path:
    """Compile `tools/gen_ref_logits.c` against the libllama.so the .ref files came from."""
    subprocess.run(
        [
            "gcc", "-O2", "-std=c11", "-o", str(out),
            "-I", str(llama_cpp / "include"),
            "-I", str(llama_cpp / "ggml" / "include"),
            str(GEN_REF),
            "-L", str(llama_cpp / "build" / "bin"),
            "-lllama", "-lggml", "-lggml-base", "-lm",
            f"-Wl,-rpath,{llama_cpp / 'build' / 'bin'}",
        ],
        check=True,
    )
    return out


def llama_run(bin_: Path, gguf: Path, ref: Path) -> tuple[bool, str]:
    """Run llama.cpp over `gguf`. Returns (loaded, first error line if not)."""
    proc = subprocess.run([str(bin_), str(gguf), str(ref), ",".join(map(str, PROMPT))],
                          capture_output=True, text=True)
    errs = [ln for ln in proc.stderr.splitlines() if "error" in ln]
    return (proc.returncode == 0 and ref.exists()), (errs[0].strip() if errs else "")


def engine_run(gguf: Path, ref: Path) -> None:
    subprocess.run(
        [str(ENGINE), "--model", str(gguf), "--prompt-tokens", ",".join(map(str, PROMPT)),
         "--dump-logits", str(ref)],
        check=True, capture_output=True,
    )


def rows(ref: Path) -> list[np.ndarray]:
    return [np.asarray(r, dtype=np.float64) for r in A.load_ref(ref)[2]]


def identical(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()


def rel_l2(a: Path, b: Path) -> float | None:
    """Worst per-position relative L2 between two SLDREFLG files."""
    if not (a.exists() and b.exists()):
        return None
    x, y = rows(a), rows(b)
    if len(x) != len(y):
        return None
    return max(A.rel_err(p, q) for p, q in zip(x, y))


def llama_moe_fused(src: Path, dst: Path, gguf: object) -> None:
    """Rewrite a `llama`-arch MoE fixture into the fused expert form.

    Not something the fixture generator offers, because llama.cpp cannot load
    the result -- which is the point of measuring rather than assuming. The
    rewrite is `torch.cat([gate, up], dim=1)` over `(n_expert, n_ff, n_embd)`,
    i.e. what `convert_hf_to_gguf.py --fuse-gate-up-exps` writes.
    """
    r = gguf.GGUFReader(src, "r")
    w = gguf.GGUFWriter(dst, "llama")
    for kv in r.fields.values():
        # `GGUF.version` / `tensor_count` / `kv_count` are synthesised by the
        # reader from the header and are not in the file; copying them back
        # emits each one twice and the result cannot be read at all.
        # `general.architecture` is set by `GGUFWriter(path, arch)` itself.
        if kv.name.startswith("GGUF.") or kv.name == "general.architecture":
            continue
        try:
            w.add_key_value(kv.name, kv.contents(), kv.types[0])
        except Exception:  # an array or a type this writer cannot re-emit
            continue
    for t in r.tensors:
        name, arr = t.name, np.asarray(t.data)
        if t.name.endswith("ffn_up_exps.weight"):
            gate = np.asarray(
                r.tensors[[x.name for x in r.tensors].index(t.name.replace("up", "gate"))].data
            )
            name = t.name.replace("ffn_up_exps", "ffn_gate_up_exps")
            arr = np.concatenate([gate, arr], axis=1)
        elif t.name.endswith("ffn_gate_exps.weight"):
            continue
        w.add_tensor(name, arr)
    G._finish(w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--no-llama", action="store_true", help="skip the llama.cpp oracle")
    args = ap.parse_args()

    llama_cpp = Path(os.environ.get("LLAMA_CPP", str(Path.home() / "llama.cpp")))
    if not args.no_llama and not (llama_cpp / "build" / "bin" / "libllama.so").exists():
        raise SystemExit(f"no {llama_cpp}/build/bin/libllama.so — build llama.cpp, or pass --no-llama")

    gguf = G.import_gguf()
    out: dict[str, object] = {"prompt": PROMPT, "seed": hex(SEED)}

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        split, fused = td / "ds2.gguf", td / "ds2fused.gguf"
        rev = td / "ds2rev.gguf"
        G.gen_deepseek2(split, gguf, SEED)
        G.gen_deepseek2(fused, gguf, SEED, fused=True)

        saved = np.concatenate
        np.concatenate = lambda a, axis=None, *r, **k: saved(  # noqa: E731
            list(a)[::-1] if axis == 1 else a, axis=axis, *r, **k
        )
        try:
            G.gen_deepseek2(rev, gguf, SEED, fused=True)
        finally:
            np.concatenate = saved

        bin_ = None if args.no_llama else build_gen_ref(llama_cpp, td / "gen_ref_logits")

        print("== which architectures accept the fused tensor ==")
        if bin_ is not None:
            lm = td / "llama-moe.gguf"
            G.gen_llama_moe(lm, gguf, SEED)
            lmf = td / "llama-moe-fused.gguf"
            llama_moe_fused(lm, lmf, gguf)
            ok_split, _ = llama_run(bin_, lm, td / "lm.ref")
            ok_fused, err = llama_run(bin_, lmf, td / "lmf.ref")
            ok_ds, _ = llama_run(bin_, fused, td / "ds2fused.ref")
            llama_run(bin_, split, td / "ds2.ref")
            llama_run(bin_, rev, td / "ds2rev.ref")
            print(f"   llama    arch, split ffn_[gate,up]_exps : {'loads' if ok_split else 'FAILS'}")
            print(f"   llama    arch, fused ffn_gate_up_exps   : {'loads' if ok_fused else 'REJECTED'}")
            print(f"   deepseek2 arch, fused ffn_gate_up_exps   : {'loads' if ok_ds else 'FAILS'}")
            if not ok_fused:
                print(f"      {err}")
            out["llama_cpp_loads"] = {"llama_split": ok_split, "llama_fused": ok_fused,
                                      "deepseek2_fused": ok_ds}
            if not ok_fused:
                out["llama_arch_fused_rejected_with"] = err
                out["why"] = (
                    "the llama arch does not call create_tensor_gate_up_exps "
                    "(src/models/llama.cpp:78-80): it creates ffn_gate_exps / ffn_up_exps "
                    "directly and passes flags=0 for ffn_up_exps, i.e. required. So a "
                    "llama-arch file carrying only ffn_gate_up_exps cannot load at all, and "
                    "the fused form is testable only on an arch that calls the helper."
                )

            print("== is the row order inside the fused tensor observable? ==")
            same = identical(td / "ds2fused.ref", td / "ds2rev.ref")
            print(f"   llama.cpp, [gate,up] vs [up,gate] concat: "
                  f"{'IDENTICAL (order would be a detail)' if same else 'DIFFER'}"
                  f"  rel L2 {rel_l2(td / 'ds2rev.ref', td / 'ds2fused.ref'):.3e}")
            out["fused_row_order_is_observable"] = not same
            out["rel_l2_reversed_vs_gate_up"] = rel_l2(td / "ds2rev.ref", td / "ds2fused.ref")

        print("== split vs fused container, identical weights ==")
        engine_run(split, td / "eng_split.ref")
        engine_run(fused, td / "eng_fused.ref")
        eng_same = identical(td / "eng_split.ref", td / "eng_fused.ref")
        print(f"   engine,    split vs fused : {'BIT IDENTICAL' if eng_same else 'DIFFER'}")
        out["engine_split_vs_fused_bit_identical"] = eng_same
        if bin_ is not None:
            ll_same = identical(td / "ds2.ref", td / "ds2fused.ref")
            print(f"   llama.cpp, split vs fused : {'BIT IDENTICAL' if ll_same else 'DIFFER'}")
            print(f"   engine vs llama.cpp, fused: rel L2 {rel_l2(td / 'eng_fused.ref', td / 'ds2fused.ref'):.3e}")
            out["llama_cpp_split_vs_fused_bit_identical"] = ll_same

    print("\nfixtures were regenerated into a temporary directory; the checked-in pair is\n"
          "  python3 scripts/generate_synthetic_fixtures.py --arch deepseek2       --out engine/tests/data/fixtures/synthetic-deepseek2.gguf\n"
          "  python3 scripts/generate_synthetic_fixtures.py --arch deepseek2-fused --out engine/tests/data/fixtures/synthetic-deepseek2-fused.gguf")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
