#!/usr/bin/env python3
"""Decompose the engine-vs-llama.cpp gap on a quantized model into named terms.

The question this answers
-------------------------
Since tick 14 the parity suite could only run on synthetic **F32** fixtures, and
every real model on disk is quantized. On the real ones the engine and
llama.cpp disagreed (cos 0.996-0.9993 on Qwen2.5-3B Q4_K_M, argmax diverging) and
a two-way diff could not say which side was wrong. Tick 16 named three of
llama.cpp's precision shortcuts and showed they were 2e-4-class on the tiny
fixtures; they were not enough to explain 1e-1 on a real 3B model.

`scripts/requantize.py` produced the lever that closes it: a metadata-identical
F32 copy of the same quantized file. Four runs then name every term.

    e_q  engine,   quantized file        (what we ship)
    e_f  engine,   F32 widening          (same numbers, exact f32 storage)
    l_q  llama.cpp, quantized file
    l_f  llama.cpp, F32 widening

| pair          | what it isolates                                              |
|---------------|---------------------------------------------------------------|
| e_f vs l_f    | semantics, with quantization removed from both sides          |
| e_q vs e_f    | our own sensitivity to 4-bit weights                          |
| l_q vs l_f    | llama.cpp's own sensitivity to 4-bit weights                  |
| e_q vs l_q    | the gap, as originally observed                                |

The result on Tiny-Moe (a real 148M-parameter converted GGUF):

    e_f vs l_f : rel L2 4.1e-6   cos 1.000000000   argmax exact at 8/8 positions
    e_q vs e_f : rel L2 0.0      byte-identical files
    l_q vs l_f : rel L2 1.2e-1   cos 0.9972
    e_q vs l_q : rel L2 1.2e-1   cos 0.9972   -- the same numbers to 4 s.f.

`e_q == e_f` makes the algebra close: the engine computes the exact forward pass
over the dequantized weights, so `l_q - e_q == l_q - e_f == l_q - exact`, i.e.
the observed gap *is* llama.cpp's own sensitivity to 4-bit weights. There was no
engine bug behind it. `engine/tests/quant_gap.rs` asserts the `e_q == e_f` half
of that identity on a 68 KB checked-in fixture so it cannot rot.

Usage:
    python3 scripts/quant_gap_report.py \\
        --model models/Tiny-Moe.Q4_K_M.gguf \\
        --prompt 13,120,391,42,7,260,511,88 \\
        --out results/quant_gap_findings.json

Requires numpy + gguf-py, a built engine (`--engine-bin`) and, for the llama.cpp
columns, a built llama.cpp with `engine/tools/gen_ref_logits.c` compiled
(`--gen-ref-logits`).
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


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


def rel_l2(a, b) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cosine(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0.0 or nb == 0.0 else float(np.dot(a, b) / (na * nb))


def run(cmd: list[str], **kw) -> None:
    subprocess.run(cmd, check=True, **kw)


def engine_dump(binary: Path, gguf: Path, prompt: str, out: Path) -> None:
    run([str(binary), "--model", str(gguf), "--prompt-tokens", prompt, "--dump-logits", str(out)],
        stdout=subprocess.DEVNULL)


def build_gen_ref(llama_cpp: Path, src: Path, out: Path) -> None:
    run([
        "gcc", "-O2", "-std=c11", "-o", str(out),
        f"-I{llama_cpp / 'include'}", f"-I{llama_cpp / 'ggml' / 'include'}",
        str(src),
        f"-L{llama_cpp / 'build' / 'bin'}",
        "-lllama", "-lggml", "-lggml-base", "-lm",
        f"-Wl,-rpath,{llama_cpp / 'build' / 'bin'}",
    ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--prompt", required=True, help="comma-separated token ids")
    ap.add_argument("--out", type=Path, default=None, help="write findings JSON here")
    ap.add_argument("--engine-bin", type=Path, default=REPO / "engine/target/release/slow-inference")
    ap.add_argument("--llama-cpp", type=Path, default=Path.home() / "llama.cpp")
    ap.add_argument("--workdir", type=Path, default=None, help="keep the intermediate .ref files")
    ap.add_argument("--threads", default="1", help="llama.cpp n_threads; ggml's reduction order depends on it")
    args = ap.parse_args()

    prompt = args.prompt
    work = args.workdir or Path(tempfile.mkdtemp(prefix="quantgap-"))
    work.mkdir(parents=True, exist_ok=True)
    f32_model = work / (args.model.stem + "-f32.gguf")

    if not args.engine_bin.exists():
        raise SystemExit(f"{args.engine_bin} not found — cargo build --release first")
    have_llama = (args.llama_cpp / "build" / "bin" / "libllama.so").exists()

    print(f"== {args.model} ==")
    print(f"widening to F32 -> {f32_model}")
    env = {**__import__("os").environ, "PYTHONPATH": str(args.llama_cpp / "gguf-py")}
    run([sys.executable, str(REPO / "scripts" / "requantize.py"),
         "--to", "f32", "--in", str(args.model), "--out", str(f32_model), "--quiet"],
        env=env)

    refs = {}
    for tag, gguf in (("q", args.model), ("f32", f32_model)):
        refs[tag + "_eng"] = work / f"{tag}.eng.ref"
        engine_dump(args.engine_bin, gguf, prompt, refs[tag + "_eng"])
        print(f"  engine {tag:>3}: {refs[tag + '_eng'].name}")
    if have_llama:
        gen = work / "gen_ref_logits"
        build_gen_ref(args.llama_cpp, REPO / "engine" / "tools" / "gen_ref_logits.c", gen)
        for tag, gguf in (("q", args.model), ("f32", f32_model)):
            refs[tag + "_llama"] = work / f"{tag}.llama.ref"
            run([str(gen), str(gguf), str(refs[tag + "_llama"]), prompt],
                env={**__import__("os").environ, "LLAMA_REF_THREADS": args.threads},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  llama  {tag:>3}: {refs[tag + '_llama'].name}")
    else:
        print("  (no libllama.so — engine-only columns; the llama.cpp half needs a build)")

    def rows(p: Path):
        _, n_vocab, r = load_ref(p)
        return n_vocab, r

    nv, e_q = rows(refs["q_eng"])
    _, e_f = rows(refs["f32_eng"])

    def cmp(a, b):
        return {
            "worst_rel_l2": max(rel_l2(x, y) for x, y in zip(a, b)),
            "worst_cosine": min(cosine(x, y) for x, y in zip(a, b)),
            "argmax_agreement": sum(int(np.argmax(x) == np.argmax(y)) for x, y in zip(a, b)),
            "n_positions": len(a),
        }

    out = {
        "model": str(args.model),
        "prompt": [int(t) for t in prompt.split(",")],
        "vocab": nv,
        "llama_cpp": str(args.llama_cpp) if have_llama else None,
        "pairs": {},
    }

    out["pairs"]["engine_f32_vs_llama_f32 (semantics, quantization removed)"] = cmp(e_f, load_ref(refs["f32_llama"])[2]) if have_llama else None
    out["pairs"]["engine_q_vs_engine_f32 (our sensitivity to 4-bit weights)"] = cmp(e_q, e_f)
    if have_llama:
        _, _, l_q = load_ref(refs["q_llama"])
        _, _, l_f = load_ref(refs["f32_llama"])
        out["pairs"]["llama_q_vs_llama_f32 (llama.cpp's own sensitivity)"] = cmp(l_q, l_f)
        out["pairs"]["engine_q_vs_llama_q (the gap as originally observed)"] = cmp(e_q, l_q)

    print()
    for label, r in out["pairs"].items():
        if r is None:
            print(f"  {label:<58} (skipped)")
            continue
        print(
            f"  {label:<58} rel L2 {r['worst_rel_l2']:.3e}  cos {r['worst_cosine']:.9f}  "
            f"argmax {r['argmax_agreement']}/{r['n_positions']}"
        )

    bitwise = out["pairs"]["engine_q_vs_engine_f32 (our sensitivity to 4-bit weights)"]["worst_rel_l2"]
    print()
    if bitwise == 0.0:
        print("VERDICT: engine(Q) == engine(F32 widening) exactly, so")
        print("         engine(Q) - llama(Q) == llama(Q) - llama(F32 widening):")
        print("         the quantized-model gap IS llama.cpp's own sensitivity to 4-bit weights.")
    else:
        print(f"VERDICT: the engine moved by {bitwise:.3e} across the widening. The identity")
        print("         engine(Q) == engine(F32) does not hold, so engine/tests/quant_gap.rs")
        print("         should be failing too — investigate that before reading the rest.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    else:
        print(f"\nintermediate files left in {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
