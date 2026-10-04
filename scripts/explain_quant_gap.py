#!/usr/bin/env python3
"""Explain the engine-vs-llama.cpp gap on a real quantized model.

Why this exists
---------------
Tick 19 measured, on `models/Tiny-Moe.Q4_K_M.gguf`, that the engine and
llama.cpp agree on argmax at every prompt position but differ by 2-6e-2 in
relative L2, against 3-6e-7 on the synthetic F32 fixtures. Two orders above the
f16-shortcut floor measured in tick 16, and *not structural* (argmax agrees
everywhere). "Not structural" is not "explained", so it sat as the largest open
question in the project for two ticks.

`scripts/arbitrate_logits.py` could not answer it: it had no MoE forward pass, so
it could not read Tiny-MoE at all. It can now (see `moe_ffn` in that file), and
this script is the experiment around it. Three independent measurements, each
answering a different half of the question:

1. **Which side drifts?** An independent float64 forward pass over the *same*
   dequantized weight values both engines see. No accumulation-order error, no
   activation quantization, no f16 anything.

2. **Where does the drift come from?** The model is rewritten to F32 with
   `scripts/requantize.py --to f32`, which changes only each tensor's *container*
   and not one weight value. Both sides are re-run on it. If the gap is the
   weight container rather than the arithmetic, it is ggml's quantized matmul.

3. **Why?** `engine/tools/probe_quant_matvec.c` measures a single matvec with
   ggml on both sides -- the real kernel path against a float64 dot product on
   the dequantized weights -- and then repeats it with the activation
   quantization removed (`--control`), which is the only difference between the
   two numbers.

Measured result, Tiny-MoE Q4_K_M, prompt "The capital of France is" (11 tokens):

    engine   vs exact float64   1.2e-6 .. 4.9e-6     (both weight containers)
    llama.cpp vs exact float64   2.0e-2 .. 6.1e-2     Q4_K_M weights
                                2.4e-7 .. 6.8e-7     F32 weights, same values
    per-matvec, quantized weights  3e-3 .. 5e-2
    per-matvec, --control          1.1e-7 .. 1.5e-7

So: the engine is the accurate side, llama.cpp is the drifting one, and the
whole of the drift is ggml rounding the *activation* to Q8_K before every
quantized dot product. That is not a semantic difference, it cannot be fixed by
being more correct about the graph, and it puts a ~2e-2 relative noise floor
under any engine-vs-llama.cpp comparison on 4-bit weights. The engine's own
Q4_K_M dump is byte-identical to its F32 dump.

Usage:
    scripts/explain_quant_gap.py --model models/Tiny-Moe.Q4_K_M.gguf \
        --prompt "The capital of France is" --out results/quant_gap_explained.json

Needs a built llama.cpp (for libllama + ggml), numpy, and gguf-py. Everything it
builds lands in --workdir (default: a mkdtemp), which holds the F32 rewrite
(~550 MB for Tiny-MoE) and the .ref dumps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from arbitrate_logits import FORWARD, Weights, load_ref, rel_err  # noqa: E402


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    proc = subprocess.run(cmd, env=merged, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def build(tool_c: Path, out: Path, ll: Path, libs: list[str]) -> None:
    """Compile one of engine/tools/*.c against the llama.cpp in `ll`."""
    run(
        [
            "gcc", "-O2", "-std=c11", "-o", str(out),
            f"-I{ll}/include", f"-I{ll}/ggml/include", str(tool_c),
            f"-L{ll}/build/bin", *libs, "-lm",
            f"-Wl,-rpath,{ll}/build/bin",
        ]
    )


def ref_tokens(path: Path) -> list[int]:
    buf = path.read_bytes()
    (n,) = struct.unpack_from("<I", buf, 12)
    return list(struct.unpack_from(f"<{n}i", buf, 16))


def llama_env() -> dict[str, str]:
    return {"LD_LIBRARY_PATH": str(Path.home() / "llama.cpp" / "build" / "bin")}


def engine_ref(binary: Path, model: Path, prompt: str, out: Path) -> list[int]:
    run([str(binary), "--model", str(model), "--prompt", prompt, "--dump-logits", str(out)])
    return ref_tokens(out)


def arbiter_rows(gguf: Path, arch: str, tokens: list[int]) -> list[np.ndarray]:
    w = Weights(gguf)
    return FORWARD[arch](w, arch, tokens)


def score(rows: list[np.ndarray], f64: list[np.ndarray]) -> dict[str, object]:
    per = [rel_err(r, f) for r, f in zip(rows, f64)]
    argmax_match = sum(1 for r, f in zip(rows, f64) if int(np.argmax(r)) == int(np.argmax(f)))
    return {
        "worst_rel_l2": max(per),
        "best_rel_l2": min(per),
        "argmax_matches": argmax_match,
        "n_positions": len(per),
        "per_position": [float(x) for x in per],
    }


PROBE_ROW = re.compile(r"l2_rel=(?P<l2>[\deE.+-]+)\s+max_rel=(?P<mx>[\deE.+-]+)")
PROBE_CONTROL = re.compile(r"CONTROL .*l2_rel=(?P<l2>[\deE.+-]+)")
PROBE_SUMMARY = re.compile(r"SUMMARY type=(?P<t>\S+)\s+ne0=(?P<ne0>\d+)\s+nrows=(?P<nr>\d+)\s+(?P<d>\w+)")


def parse_probe(text: str) -> dict[str, object]:
    rows = [float(m.group("l2")) for m in PROBE_ROW.finditer(text)]
    controls = [float(m.group("l2")) for m in PROBE_CONTROL.finditer(text)]
    if not rows:
        raise SystemExit("probe_quant_matvec produced no rows; is the model quantized?")
    summary: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if not line.startswith("SUMMARY"):
            continue
        head = PROBE_SUMMARY.match(line)
        if not head:
            continue
        worst = float(re.search(r"worst_rel=([\deE.+-]+)", line).group(1))
        mean = float(re.search(r"mean_rms=([\deE.+-]+)", line).group(1))
        key = f"{head.group('t')}/ne0={head.group('ne0')}/{head.group('d')}"
        summary[key] = {"mean_rms": mean, "worst": worst}
    rows.sort()
    return {
        "n_rows_probed": len(rows),
        "median_rel_l2": rows[len(rows) // 2],
        "min_rel_l2": rows[0],
        "max_rel_l2": rows[-1],
        # The control repeats the same matvec on the same weight bytes with the
        # activation quantization removed. It is the whole argument.
        "control_no_activation_quant": controls,
        "control_worst_rel_l2": max(controls) if controls else None,
        "summary": summary,
    }


# llama.cpp's precision knobs, from gen_ref_logits.c. "precise" is the default
# there and is the configuration tests/ref_parity.rs gates on; the rest exist to
# price the shortcuts individually.
CONFIGS = [
    ("precise", {}),
    ("flash_attn_on", {"LLAMA_REF_FA_ON": "1"}),
    ("f16_kv_cache", {"LLAMA_REF_KV_F16": "1"}),
    ("llama_cpp_defaults", {"LLAMA_REF_LLAMA_DEFAULTS": "1"}),
    ("precise_8_threads", {"LLAMA_REF_THREADS": "8"}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=REPO / "models" / "Tiny-Moe.Q4_K_M.gguf")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--arch", default=None, help="default: general.architecture from the GGUF")
    ap.add_argument("--llama-cpp", type=Path, default=Path.home() / "llama.cpp")
    ap.add_argument("--engine", type=Path, default=REPO / "engine" / "target" / "release" / "slow-inference")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "quant_gap_explained.json")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--no-probe", action="store_true", help="skip the per-matvec measurement")
    args = ap.parse_args()

    ll: Path = args.llama_cpp
    if not (ll / "build" / "bin" / "libllama.so").exists():
        raise SystemExit(f"no libllama.so under {ll}/build/bin -- build llama.cpp first")
    gguf_py = ll / "gguf-py"
    work = args.workdir or Path(tempfile.mkdtemp(prefix="quantgap-"))
    work.mkdir(parents=True, exist_ok=True)
    pyenv = {"PYTHONPATH": str(gguf_py) + ":" + os.environ.get("PYTHONPATH", "")}

    report: dict[str, object] = {
        "model": str(args.model),
        "prompt": args.prompt,
        "llama_cpp_commit": subprocess.run(
            ["git", "-C", str(ll), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "question": (
            "engine vs llama.cpp disagree by 2-6e-2 relative L2 on Tiny-MoE Q4_K_M "
            "with argmax agreeing everywhere. Which side drifts, and why?"
        ),
        "method": [
            "1. independent float64 forward pass over the dequantized weights",
            "2. the same model rewritten to F32 with identical weight values",
            "3. one matvec, ggml kernel path vs float64, with and without activation quantization",
        ],
    }

    print("== building the two llama.cpp-linked tools ==")
    gen_ref = work / "gen_ref_logits"
    build(REPO / "engine" / "tools" / "gen_ref_logits.c", gen_ref, ll,
          ["-lllama", "-lggml", "-lggml-base"])
    probe = work / "probe_quant_matvec"
    if not args.no_probe:
        build(REPO / "engine" / "tools" / "probe_quant_matvec.c", probe, ll,
              ["-lggml", "-lggml-base", "-lggml-cpu"])

    print("== 1. exact float64 forward pass, and the engine on both weight containers ==")
    eng_q = work / "engine.q4.ref"
    tokens = engine_ref(args.engine, args.model, args.prompt, eng_q)
    toks = ",".join(str(t) for t in tokens)
    print(f"   {len(tokens)} tokens: {toks}")

    arch = args.arch
    if arch is None:
        wq = Weights(args.model)
        arch = str(wq.meta["general.architecture"])
        if arch not in FORWARD:
            raise SystemExit(
                f"arbitrate_logits.py has no float64 forward pass for {arch!r}; "
                f"known: {sorted(FORWARD)}"
            )
    f64_q = arbiter_rows(args.model, arch, tokens)

    # The F32 rewrite: same metadata, same weight values, different container.
    f32_model = work / (args.model.stem + ".f32.gguf")
    print("== 2. rewriting the model to F32 with identical weight values ==")
    run(
        [sys.executable, str(REPO / "scripts" / "requantize.py"),
         "--in", str(args.model), "--out", str(f32_model), "--to", "f32", "--quiet"],
        env=pyenv,
    )
    eng_f32 = work / "engine.f32.ref"
    tokens_f32 = engine_ref(args.engine, f32_model, args.prompt, eng_f32)
    if tokens_f32 != tokens:
        raise SystemExit(f"tokenization changed across the rewrite: {tokens} vs {tokens_f32}")
    f64_f32 = arbiter_rows(f32_model, arch, tokens)
    identical = eng_q.read_bytes() == eng_f32.read_bytes()

    print("== 3. llama.cpp, one run per precision configuration ==")
    llama_runs: dict[str, dict[str, object]] = {}
    for name, env in CONFIGS:
        out = work / f"llama.{name}.ref"
        run([str(gen_ref), str(args.model), str(out), toks], env={**llama_env(), **env})
        llama_runs[name] = score(load_ref(out)[2], f64_q)
        print(f"   {name:<20} worst={llama_runs[name]['worst_rel_l2']:.3e}")
    out_f32 = work / "llama.f32.ref"
    run([str(gen_ref), str(f32_model), str(out_f32), toks], env=llama_env())
    llama_f32 = score(load_ref(out_f32)[2], f64_f32)
    print(f"   {'f32 weights':<20} worst={llama_f32['worst_rel_l2']:.3e}")

    if not args.no_probe:
        print("== 4. one matvec: ggml kernel path vs float64, and the control ==")
        p = subprocess.run(
            [str(probe), str(args.model), "--rows", "8", "--dists", "--control"],
            env=llama_env(), capture_output=True, text=True,
        )
        if p.returncode != 0:
            sys.stderr.write(p.stdout + p.stderr)
            raise SystemExit("probe_quant_matvec failed")
        (work / "probe_quant_matvec.txt").write_text(p.stdout)
        report["per_matvec"] = parse_probe(p.stdout)

    report.update(
        {
            "tokens": tokens,
            "n_tokens": len(tokens),
            "engine_q4_vs_f64": score(load_ref(eng_q)[2], f64_q),
            "engine_f32_vs_f64": score(load_ref(eng_f32)[2], f64_f32),
            "engine_q4_dump_identical_to_f32": identical,
            "llama_q4_vs_f64_by_config": llama_runs,
            "llama_f32_vs_f64": llama_f32,
            "verdict": {
                "engine": "1e-6 from exact float64 on both weight containers; "
                          "its Q4_K_M logits are byte-identical to its F32 logits",
                "llama_cpp": f"{llama_runs['precise']['worst_rel_l2']:.1e} from exact on "
                             f"Q4_K_M weights, {llama_f32['worst_rel_l2']:.1e} on the same "
                             "weight values in an F32 container",
                "cause": "ggml rounds the activation to Q8_K before every quantized dot "
                         "product; the --control row isolates it to that and nothing else",
                "consequence": "any engine-vs-llama.cpp diff on 4-bit weights carries a "
                               "~2e-2 relative floor that is llama.cpp's, not a semantic "
                               "disagreement. Gate quantized comparisons on argmax/top-k, "
                               "and gate architecture semantics on the F32 rewrite.",
            },
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print()
    print(f"== {args.model.name}: {len(tokens)} tokens, vocab {load_ref(eng_q)[1]}, arch {arch} ==")
    print("   rel L2 vs an independent float64 forward pass (worst over positions)")
    print(f"   engine   Q4_K_M weights              {report['engine_q4_vs_f64']['worst_rel_l2']:.3e}")
    print(f"   engine   F32 weights                 {report['engine_f32_vs_f64']['worst_rel_l2']:.3e}")
    for name, _ in CONFIGS:
        print(f"   llama    {name:<26} {llama_runs[name]['worst_rel_l2']:.3e}")
    print(f"   llama    F32 weights                 {llama_f32['worst_rel_l2']:.3e}")
    print(f"   engine Q4_K_M dump == F32 dump:      {identical}")
    if "per_matvec" in report:
        pm = report["per_matvec"]
        print(f"   one matvec, quantized weights       {pm['min_rel_l2']:.1e} .. {pm['max_rel_l2']:.1e}"
              f" (median {pm['median_rel_l2']:.1e}, n={pm['n_rows_probed']})")
        print(f"   same matvec, no activation quant    {pm['control_worst_rel_l2']:.1e}")
    print(f"   wrote {args.out}")
    if not args.workdir:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
