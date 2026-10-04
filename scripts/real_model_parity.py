#!/usr/bin/env python3
"""Engine vs llama.cpp logits on a REAL quantized GGUF, with the GQA geometry named.

Why this exists
---------------
Every architecture-parity claim in this project has, until now, been made on
synthetic F32 fixtures, because the real models on disk are all 4-bit and
`scripts/quant_gap_report.py` needs an F32 widening (12.4 GB for Qwen2.5-3B) to
attribute a gap. Tick 28 found a GQA pairing bug live in the engine — query head
`h` paired with KV head `h % n_kv_heads` instead of llama.cpp's `h / n_rep` —
and the fix was verified only against those fixtures. A fixture's head geometry
is part of what it tests: 8 heads over 1 KV head is MQA, both conventions agree
there, and `ref_parity` goes green with the bug back in. So "the fix is right"
was, strictly, a statement about synthetic fixtures of a particular geometry.

This script closes that without widening anything. It needs no F32 copy, no
numpy model, and ~5 GB of RAM (the engine streams weights; llama.cpp mmaps):

  * run `engine/tools/gen_ref_logits` (llama.cpp's own forward pass, flash
    attention off and f32 KV cache — the most precise configuration it has) on a
    real GGUF for an explicit token list,
  * run the engine's `--dump-logits` on the same tokens (same SLDREFLG format),
  * diff them per position: relative L2, cosine, argmax, top-k overlap, and the
    top-1/top-2 margin so an argmax disagreement can be read as a near-tie or as
    a real one,
  * report the model's GQA geometry (`head_count`, `head_count_kv`, `n_rep`) and
    whether `h / n_rep` and `h % n_kv_heads` even disagree for it, because a
    model whose geometry cannot tell the two apart proves nothing about GQA and
    must not be reported as if it does,
  * print the reference's and the engine's top-k next tokens *as text*, which is
    the one form of this evidence a human can check without a debugger.

A quantized comparison cannot be gated at 1e-6: `results/quant_gap_explained.json`
measured llama.cpp's own sensitivity to Q4_K_M weights at 2.0e-2..6.1e-2 relative
L2 on Tiny-MoE, caused by ggml rounding activations to Q8_K before every
quantized dot product (ggml's integer accumulation is exact). Anything at or
below that band is llama.cpp's noise floor, not a semantic disagreement. A real
semantic bug is not near it: the tick-28 pairing error produced rel L2 ~1.2 and
cos ~0.35, i.e. a different function rather than a drift. `--floor` defaults to
twice the worst measured floor (1.2e-1) so the gate sits ~10x below the bug
signature and ~2x above the noise.

Can this check fail? Prove it, do not assume it
----------------------------------------------
`--build LABEL=PATH[:MODEL.gguf]` may be given more than once. Two uses:

  * a *second engine binary* built with a known semantic change -- e.g. the
    tick-28 pairing error, all seven sites -- compared against the same
    llama.cpp reference, and
  * the *same binary over a second weight file*, which is the only way this
    script can measure engine-vs-engine. `engine(Q4_K_M)` against
    `engine(F32 widening)` is the F32 identity, and it is the row that decides
    whether a real-model gap is attributable to llama.cpp at all; the report
    prints it under "build vs build" and states bit-identity rather than a
    relative error.

A parity harness that has only ever reported PASS has not been shown to be able
to report FAIL. That is the same discipline `tests/gqa_mapping.rs` and
`scripts/mutation_check.py` are built on, and the second build is why this
script takes a list.

Usage:
    python3 scripts/real_model_parity.py \\
        --model models/qwen2.5-3b-instruct-q4_k_m.gguf \\
        --case capital=785,6722,315,9625,374 \\
        --out results/real_model_gqa_parity.json

    # two builds, to show the check can fail:
    python3 scripts/real_model_parity.py --model ... --case ... \\
        --build fixed=engine/target/release/slow-inference \\
        --build kv_head_modulo=/tmp/mutant/target/release/slow-inference \\
        --out results/real_model_gqa_parity.json

Token ids must come from llama.cpp's own tokenizer
(`engine/tools/gen_ref_tokens`, or the checked-in
`engine/tests/data/ref_tokens/*.ref`), not from the engine's: feeding the two
sides the same ids is what makes the logits comparable, and a shared tokenization
bug would otherwise show up here as a semantic one. `--case` takes ids, never a
string, for that reason.

Requires numpy + gguf-py (for the metadata and the piece strings), a built
engine and a built llama.cpp with `engine/tools/gen_ref_logits.c` compilable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

#: Twice the worst relative L2 that `results/quant_gap_explained.json` measured
#: for llama.cpp's own sensitivity to Q4_K_M weights (6.1e-2, on Tiny-MoE). The
#: gate has to clear that floor and still sit far below the ~1.2 rel L2 that a
#: semantic bug produces.
#:
#: This is a fallback, and it is a *carried* number: 6.1e-2 was measured on a
#: 12-layer MHA model with 768-wide embeddings. On the 3B Q4_K_M model (36
#: layers, 2048-wide) the correct build measures 1.29e-1, so this floor sits
#: BELOW the right answer and reports a correct build as 7% over the line. Pass
#: `--ref-model` with the other weight file and `--floor auto` to have the floor
#: measured on the model in hand instead.
DEFAULT_FLOOR = 1.2e-1

#: Multiplier applied to the measured reference sensitivity when `--floor auto`.
#: A gate exactly at the noise has a 50% chance of failing on a correct build, so
#: it wants headroom; it must also stay far below the ~1.2 rel L2 of a semantic
#: bug, and 2x buys 500x of that on the 3B model (2.6e-1 against 1.2).
FLOOR_SAFETY = 2.0


def load_ref(path: Path):
    """Read a SLDREFLG v1 dump: (tokens, n_vocab, rows as float64)."""
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
    if len(buf) - off != n_tok * n_vocab * 4:
        raise SystemExit(f"{path}: truncated logits")
    flat = np.frombuffer(buf, dtype="<f4", count=n_tok * n_vocab, offset=off)
    rows = [flat[i * n_vocab : (i + 1) * n_vocab].astype(np.float64) for i in range(n_tok)]
    return toks, n_vocab, rows


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / nb) if nb else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return 0.0 if na == 0.0 or nb == 0.0 else float(np.dot(a, b) / (na * nb))


def topk(row: np.ndarray, k: int) -> list[int]:
    # argpartition then sort: a full argsort of a 151936-wide row is 20 ms of
    # nothing, but this runs once per position per build and the intent reads
    # better without the noise.
    k = min(k, row.size)
    idx = np.argpartition(-row, k - 1)[:k]
    return sorted(idx.tolist(), key=lambda i: -row[i])


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    return len(set(topk(a, k)) & set(topk(b, k))) / float(min(k, a.size))


def margin(row: np.ndarray) -> float:
    """Gap between a row's best and second-best logit, relative to its scale."""
    t = topk(row, 2)
    if len(t) < 2:
        return 0.0
    scale = float(np.max(np.abs(row))) or 1.0
    return float(row[t[0]] - row[t[1]]) / scale


def piece(pieces: list[str], i: int) -> str:
    """A BPE token as readable text (GPT-2 byte-level 'Ġ' is a space)."""
    return pieces[i].replace("Ġ", " ").replace("Ċ", "\\n").replace("▁", " ")


def pairwise_diff(rows_by_build: dict[str, list[np.ndarray]]) -> dict:
    """Every unordered pair of builds, diffed against each other directly.

    This is the row the reference column cannot produce. When two entries are
    the same binary over two different weight files, the interesting claim is
    not "how far is this from llama.cpp" but "are these the same function" --
    and for a quantized model that claim should come out as *bit-identical*,
    because the engine dequantizes a Q4_K tensor to the f32 values it then
    multiplies in f32, so the weight format reaches the arithmetic only through
    those values. When that comes out non-zero it means the engine's compute
    does depend on how the weights are stored, and every real-model gap
    measured against llama.cpp stops being attributable to llama.cpp.

    Byte-identity is checked on the raw f32 payloads rather than through
    `rel_l2`, so an exact-zero print cannot be a rounding artefact of the
    metric.
    """
    out: dict = {}
    labels = list(rows_by_build)
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            a, b = rows_by_build[la], rows_by_build[lb]
            if len(a) != len(b):
                raise SystemExit(f"{la} has {len(a)} positions, {lb} has {len(b)}")
            per = [
                {
                    "rel_l2": rel_l2(x, y),
                    "cosine": cosine(x, y),
                    "argmax_match": int(np.argmax(x)) == int(np.argmax(y)),
                    "rows_bit_identical": bool(np.array_equal(x, y)),
                }
                for x, y in zip(a, b)
            ]
            out[f"{la} vs {lb}"] = {
                "n_positions": len(per),
                "worst_rel_l2": max(p["rel_l2"] for p in per),
                "worst_cosine": min(p["cosine"] for p in per),
                "argmax_agreement": sum(p["argmax_match"] for p in per),
                "bit_identical_rows": sum(p["rows_bit_identical"] for p in per),
                "byte_identical": all(p["rows_bit_identical"] for p in per),
                "per_position": per,
            }
    return out


def ensure_gguf(llama_cpp: Path) -> None:
    """Make llama.cpp's `gguf` importable, the way its own scripts do it.

    gguf-py is not an installed package; it lives in the llama.cpp checkout and is
    normally reached through PYTHONPATH. A tool that only runs when the caller
    remembered an environment variable is a tool that gets re-implemented instead.
    """
    try:
        import gguf  # noqa: F401
    except ImportError:
        path = str(llama_cpp / "gguf-py")
        if not Path(path).is_dir():
            raise SystemExit(
                f"cannot import gguf and {path} does not exist — pass --llama-cpp "
                f"or set PYTHONPATH to a llama.cpp checkout's gguf-py"
            )
        sys.path.insert(0, path)


def read_geometry(path: Path) -> dict:
    # Not `gguf` for the parameter: the module is imported under that name below
    # and a parameter would shadow it at the one line that uses it.
    import gguf

    r = gguf.GGUFReader(str(path), "r")
    fields = r.fields
    arch = fields["general.architecture"].contents()
    n_head = int(fields[f"{arch}.attention.head_count"].contents())
    kv_key = f"{arch}.attention.head_count_kv"
    n_head_kv = int(fields[kv_key].contents()) if kv_key in fields else n_head
    # MLA stores no KV heads at all: K and V are absorbed into a low-rank latent.
    mla = f"{arch}.attention.key_length_mla" in fields
    if n_head_kv == 0:
        # some archs store 0 to mean "same as head_count"
        n_head_kv = n_head
    n_rep = n_head // n_head_kv if n_head_kv else 0
    grouped = [h // n_rep for h in range(n_head)] if n_rep else []
    interleaved = [h % n_head_kv for h in range(n_head)] if n_head_kv else []
    pieces = [str(t) for t in fields["tokenizer.ggml.tokens"].contents()]
    return {
        "arch": str(arch),
        "n_layer": int(fields[f"{arch}.block_count"].contents()),
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "n_rep": n_rep,
        "mla": mla,
        "mha": n_head_kv == n_head,
        "single_kv_head": n_head_kv == 1,
        # The property that decides whether this model can tell the two GQA
        # conventions apart at all. False here means a green result on this
        # model says nothing about GQA, and must not be reported as if it did.
        "gqa_conventions_disagree": bool(n_rep) and grouped != interleaved,
        "grouped_pairing": grouped,
        "interleaved_pairing": interleaved,
        "n_vocab": len(pieces),
        "pieces": pieces,
    }


def build_gen_ref(llama_cpp: Path, out: Path) -> None:
    subprocess.run(
        [
            "gcc", "-O2", "-std=c11", "-o", str(out),
            f"-I{llama_cpp / 'include'}", f"-I{llama_cpp / 'ggml' / 'include'}",
            str(REPO / "engine" / "tools" / "gen_ref_logits.c"),
            f"-L{llama_cpp / 'build' / 'bin'}",
            "-lllama", "-lggml", "-lggml-base", "-lm",
            f"-Wl,-rpath,{llama_cpp / 'build' / 'bin'}",
        ],
        check=True,
    )


def parse_cases(raw: list[str]) -> list[tuple[str, str]]:
    out = []
    for item in raw:
        name, _, ids = item.partition("=")
        if not ids:
            raise SystemExit(f"--case {item!r} must be NAME=id,id,id")
        for tok in ids.split(","):
            if not tok.strip().lstrip("-").isdigit():
                raise SystemExit(f"--case {item!r}: {tok!r} is not a token id")
        out.append((name, ids.replace(" ", "")))
    return out


def parse_builds(raw: list[str], default: Path) -> list[tuple[str, Path, Path | None]]:
    """`LABEL=BINARY[:MODEL.gguf]` -> (label, binary, model or None for --model).

    The model override exists for the widening experiment. Everything this
    script can say is "one llama.cpp reference vs N engine runs", and that
    cannot express the one row of the table that decides attribution:

        engine(Q4_K_M)  vs  engine(F32 widening)

    Both sides are the engine, so there is no reference to diff them against --
    and that comparison is exactly the F32 identity, the thing that makes a
    quantized gap attributable to *somebody*. Running the same binary over two
    files puts it in the same table as the reference diff, where the pairwise
    summary below reports it directly instead of leaving it to a second script
    that would be free to get its own labels wrong.
    """
    if not raw:
        return [("engine", default, None)]
    out = []
    for item in raw:
        label, _, rest = item.partition("=")
        if not rest:
            raise SystemExit(f"--build {item!r} must be LABEL=/path/to/binary[:model.gguf]")
        # Split on the LAST colon, and only when the tail looks like a gguf, so a
        # binary or directory containing a colon is not silently mangled.
        model: Path | None = None
        head, sep, tail = rest.rpartition(":")
        if sep and tail.endswith(".gguf"):
            model, rest = Path(tail), head
        out.append((label, Path(rest), model))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="NAME=IDS",
        help="token ids from llama.cpp's tokenizer; repeatable",
    )
    ap.add_argument(
        "--build",
        action="append",
        default=[],
        metavar="LABEL=PATH[:MODEL.gguf]",
        help="an engine binary; repeatable, so a second build with a known "
        "semantic change can be compared against the same reference. The "
        "optional model override runs the same binary over a different weight "
        "file, which is how engine(Q4_K_M) vs engine(F32 widening) -- the F32 "
        "identity -- is measured here rather than in a second script",
    )
    ap.add_argument("--out", type=Path, default=None, help="write findings JSON here")
    ap.add_argument(
        "--engine-bin",
        type=Path,
        default=REPO / "engine/target/release/slow-inference",
        help="used when --build is not given",
    )
    ap.add_argument("--llama-cpp", type=Path, default=Path.home() / "llama.cpp")
    ap.add_argument("--workdir", type=Path, default=None, help="keep the intermediate .ref files")
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-run every engine build even if its .ref is already in --workdir",
    )
    ap.add_argument("--threads", default="1", help="llama.cpp n_threads; ggml's reduction order depends on it")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument(
        "--ref-model",
        type=Path,
        default=None,
        help="a second weight file for the ORACLE to run on: same tokens, same "
        "precision settings, different storage. The gap between the two "
        "references is llama.cpp's own sensitivity to the weight format, which "
        "is the quantity a gap has to beat before it means a semantic "
        "disagreement. Pair with --floor auto.",
    )
    ap.add_argument(
        "--ref-label",
        default="alt",
        help="filename tag for the second reference's .ref (default 'alt')",
    )
    ap.add_argument(
        "--floor",
        default=str(DEFAULT_FLOOR),
        help=f"relative L2 at or below which a gap is llama.cpp's quantization "
        f"noise rather than a semantic disagreement. A number, or 'auto' to use "
        f"{FLOOR_SAFETY:g} x the sensitivity measured from --ref-model on these "
        f"exact positions. The default {DEFAULT_FLOOR:g} is carried from a "
        f"12-layer model and is too low for a 36-layer one.",
    )
    args = ap.parse_args()
    if args.floor != "auto":
        try:
            args.floor = float(args.floor)
        except ValueError:
            raise SystemExit(f"--floor {args.floor!r}: expected a number or 'auto'")

    cases = parse_cases(args.case)
    builds = parse_builds(args.build, args.engine_bin)
    for _, path, model in builds:
        if not path.exists():
            raise SystemExit(f"{path} not found — cargo build --release first")
        if model is not None and not model.exists():
            raise SystemExit(f"--build model {model} not found")
    if not (args.llama_cpp / "build" / "bin" / "libllama.so").exists():
        raise SystemExit(f"no libllama.so under {args.llama_cpp} — the reference side needs llama.cpp")

    work = args.workdir or Path(tempfile.mkdtemp(prefix="realparity-"))
    work.mkdir(parents=True, exist_ok=True)
    gen = work / "gen_ref_logits"
    if not gen.exists():
        build_gen_ref(args.llama_cpp, gen)

    print(f"== {args.model} ==")
    ensure_gguf(args.llama_cpp)
    meta = read_geometry(args.model)
    pieces = meta.pop("pieces")
    print(
        f"  arch={meta['arch']} layers={meta['n_layer']} heads={meta['n_head']} "
        f"kv={meta['n_head_kv']} n_rep={meta['n_rep']} vocab={meta['n_vocab']}"
    )
    kind = (
        "MLA (no KV heads to pair)"
        if meta["mla"]
        else "MHA (both conventions agree)"
        if meta["mha"]
        else "MQA, single KV head (both conventions agree)"
        if meta["single_kv_head"]
        else "grouped-query"
    )
    print(f"  attention: {kind}; h/n_rep and h%n_kv_heads {'DISAGREE' if meta['gqa_conventions_disagree'] else 'AGREE'} on this model")
    if not meta["gqa_conventions_disagree"] and not meta["mla"]:
        print(
            "  NOTE: this geometry cannot distinguish the two GQA pairings, so a pass "
            "here is evidence about everything EXCEPT the pairing."
        )

    env = {**os.environ, "LLAMA_REF_THREADS": args.threads}
    findings: dict = {
        "model": str(args.model),
        "geometry": meta,
        "llama_cpp": str(args.llama_cpp),
        "llama_cpp_commit": subprocess.run(
            ["git", "-C", str(args.llama_cpp), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip() or None,
        "threads": args.threads,
        "floor": args.floor,
        "topk": args.topk,
        "builds": {
            label: {"binary": str(path), "model": str(model or args.model)}
            for label, path, model in builds
        },
        "cases": {},
    }

    def run_oracle(model: Path, path: Path, ids: str, label: str) -> tuple[list[int], int, list]:
        """Run the oracle, or reuse a dump already in the workdir.

        Cached for the same reason the engine dumps are: on a 12.7 GB F32 model
        the oracle takes minutes per case, and iterating on the reporting above
        it should not re-pay that. Reuse is not trusted blindly -- `load_ref`
        returns the ids the oracle consumed and they are checked against the
        case below -- but the id echo cannot catch a dump made from a *different
        weight file*, so the model is named in the filename (`llama-q4km-*.ref`
        vs `llama-*.ref`) and `--force` remains the escape hatch.
        """
        if path.exists() and not args.force:
            print(f"  {label}: reusing {path.name}")
        else:
            subprocess.run(
                [str(gen), str(model), str(path), ids],
                env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        toks, vocab, rows = load_ref(path)
        want = [int(t) for t in ids.split(",")]
        if toks != want:
            # The oracle echoes the ids it consumed; a mismatch means the two
            # sides would not be looking at the same sequence, which would make
            # every number below meaningless.
            raise SystemExit(f"{label} oracle consumed {toks}, expected {want}")
        return toks, vocab, rows

    for name, ids in cases:
        print(f"\n-- case {name}: {ids}")
        rtoks, rvocab, rrows = run_oracle(args.model, work / f"llama-{name}.ref", ids, "reference")
        print(f"  reference: {len(rtoks)} positions x {rvocab} vocab")

        # The reference's OWN sensitivity to the weight format: the same oracle,
        # the same tokens, the same precision settings, on a second weight file.
        #
        # This is the number the floor should be built from, and it is the one
        # that cannot be argued into existence. A quantized engine-vs-llama.cpp
        # gap is only evidence of a semantic disagreement if it is bigger than
        # what the reference alone does when the weights are re-quantized. A
        # floor carried as a constant from another model is a guess about a
        # quantity that is a property of this model's weights, this model's
        # depth and this quantization type; on the 3B Q4_K_M model the carried
        # 1.2e-1 was measured against Tiny-MoE's 12 layers, and it sat BELOW
        # the correct build's own 1.29e-1 -- a gate that fails the right answer.
        r2rows = None
        if args.ref_model is not None:
            _, v2, r2rows = run_oracle(
                args.ref_model, work / f"llama-{args.ref_label}-{name}.ref", ids,
                f"reference on {args.ref_label}",
            )
            if v2 != rvocab:
                raise SystemExit(f"reference on {args.ref_model}: vocab {v2} against {rvocab}")

        # Resolved per case rather than once, so an automatic floor is measured
        # on exactly the positions being gated instead of on a sibling prompt.
        if args.floor == "auto":
            if r2rows is None:
                raise SystemExit(
                    "--floor auto needs --ref-model: the floor is the reference's "
                    "own sensitivity to the weight format, and there is nothing to "
                    "measure it from without a second weight file"
                )
            floor_here = FLOOR_SAFETY * max(
                rel_l2(a, b) for a, b in zip(rrows, r2rows)
            )
            print(
                f"  floor: {floor_here:.3e} = {FLOOR_SAFETY:g} x the reference's own "
                f"movement on the other weight file (measured, not carried)"
            )
        else:
            floor_here = args.floor

        case_out: dict = {"tokens": rtoks, "builds": {}, "floor": floor_here}
        # Kept for the pairwise diff below. 17 x 151936 f32 is ~10 MB per build,
        # and it is the only way this script can report engine-vs-engine.
        rows_by_build: dict[str, list[np.ndarray]] = {}
        for label, binary, override in builds:
            model_path = override or args.model
            eng_path = work / f"{label}-{name}.ref"
            # Reuse a dump that is already there. On a 3B model one engine run is
            # 30-70 s, and re-deriving identical bytes to re-print a table is the
            # kind of cost that makes a harness get abandoned. The tokens are
            # checked below against the reference, so a stale file is caught
            # rather than silently compared.
            if eng_path.exists() and not args.force:
                print(f"  {label}: reusing {eng_path.name}")
            else:
                subprocess.run(
                    [str(binary), "--model", str(model_path), "--prompt-tokens", ids,
                     "--dump-logits", str(eng_path)],
                    check=True, stdout=subprocess.DEVNULL,
                )
            etoks, evocab, erows = load_ref(eng_path)
            if etoks != rtoks or evocab != rvocab:
                raise SystemExit(f"{label}: token/vocab mismatch against the reference")
            if len(erows) != len(rrows):
                raise SystemExit(f"{label}: {len(erows)} rows against the reference's {len(rrows)}")

            per = []
            for e, r in zip(erows, rrows):
                ae, ar = int(np.argmax(e)), int(np.argmax(r))
                per.append({
                    "rel_l2": rel_l2(e, r),
                    "cosine": cosine(e, r),
                    "argmax_engine": ae,
                    "argmax_reference": ar,
                    "argmax_match": ae == ar,
                    f"top{args.topk}_overlap": top_k_overlap(e, r, args.topk),
                    # Only meaningful where the argmaxes differ, but recorded
                    # everywhere: a disagreement inside the reference's own
                    # top-1/top-2 gap is a coin flip, not a semantic claim.
                    "engine_top1_top2_margin": margin(e),
                    "reference_top1_top2_margin": margin(r),
                })
            worst = max(p["rel_l2"] for p in per)
            worst_cos = min(p["cosine"] for p in per)
            agree = sum(p["argmax_match"] for p in per)
            case_out["builds"][label] = {
                "worst_rel_l2": worst,
                "worst_cosine": worst_cos,
                "argmax_agreement": agree,
                "n_positions": len(per),
                "within_floor": worst <= floor_here,
                "per_position": per,
            }
            verdict = "within the quantization floor" if worst <= floor_here else "ABOVE the floor"
            print(
                f"  {label:<16} rel L2 worst {worst:.3e}  cos worst {worst_cos:.9f}  "
                f"argmax {agree}/{len(per)}  -> {verdict}"
            )

            # The human-checkable version of the same comparison: what does each
            # side think comes next, as text. Inside the build loop on purpose --
            # the first version of this script computed it after the loop, from
            # whichever rows happened to be bound last, so with two builds it
            # printed the MUTANT's next tokens under a heading that said "engine"
            # while the table above it said the fixed build matched 12/12. A
            # mislabelled number in a parity harness is worse than no number.
            last = len(rrows) - 1
            rt, et = topk(rrows[last], args.topk), topk(erows[last], args.topk)
            case_out["builds"][label]["next_token_at_last_position"] = {
                "reference": [piece(pieces, i) for i in rt],
                "reference_ids": rt,
                "engine": [piece(pieces, i) for i in et],
                "engine_ids": et,
                "top1_match": rt[0] == et[0],
                "set_overlap": top_k_overlap(erows[last], rrows[last], args.topk),
            }
            print(f"  {label}: next token, llama.cpp: " + " | ".join(piece(pieces, i) for i in rt))
            print(f"  {label}: next token, engine   : " + " | ".join(piece(pieces, i) for i in et))
            rows_by_build[label] = erows

        case_out["pairwise"] = pairwise_diff(rows_by_build)
        if case_out["pairwise"]:
            print("  -- build vs build (no reference involved)")
            for pair, val in case_out["pairwise"].items():
                print(
                    f"     {pair:<34} rel L2 worst {val['worst_rel_l2']:.3e}  "
                    f"byte-identical: {val['byte_identical']}"
                )

        if r2rows is not None:
            per = [
                {
                    "rel_l2": rel_l2(a, b),
                    "cosine": cosine(a, b),
                    "argmax_match": int(np.argmax(a)) == int(np.argmax(b)),
                }
                for a, b in zip(rrows, r2rows)
            ]
            worst = max(p["rel_l2"] for p in per)
            case_out["reference_self_sensitivity"] = {
                "reference_model": str(args.model),
                "other_weight_file": str(args.ref_model),
                "note": "llama.cpp vs llama.cpp, identical tokens and precision "
                "settings; the only difference is how the weights are stored. "
                "An engine-vs-llama.cpp gap at or below this is the reference's "
                "own noise, not a semantic disagreement.",
                "worst_rel_l2": worst,
                "worst_cosine": min(p["cosine"] for p in per),
                "argmax_agreement": sum(p["argmax_match"] for p in per),
                "n_positions": len(per),
                "per_position": per,
            }
            print(
                f"  -- the reference against itself on the other weight file: "
                f"rel L2 worst {worst:.3e}  cos worst "
                f"{min(p['cosine'] for p in per):.9f}  argmax "
                f"{sum(p['argmax_match'] for p in per)}/{len(per)}"
            )
        findings["cases"][name] = case_out

    # Cross-build summary: the whole point of accepting more than one --build is
    # the ratio between them, which is the check's dynamic range on this model.
    summary = {}
    if len(builds) > 1:
        print("\n-- separation between builds")
        for name in findings["cases"]:
            per_build = findings["cases"][name]["builds"]
            # Closest and furthest, over however many builds were given. This
            # used to unpack the first two entries of the sort, which is right
            # for two builds and raises ValueError on the third — and the third
            # build is the one that makes this worth reporting, since it is the
            # mutant the check has to be shown able to reject.
            ranked = sorted(per_build.items(), key=lambda kv: kv[1]["worst_rel_l2"])
            (lo_label, lo), (hi_label, hi) = ranked[0], ranked[-1]
            ratio = hi["worst_rel_l2"] / lo["worst_rel_l2"] if lo["worst_rel_l2"] else float("inf")
            case_floor = findings["cases"][name].get("floor", args.floor)
            margin_of_closest = (
                case_floor / lo["worst_rel_l2"] if lo["worst_rel_l2"] else float("inf")
            )
            summary[name] = {
                "closest_build": lo_label,
                "closest_worst_rel_l2": lo["worst_rel_l2"],
                "furthest_build": hi_label,
                "furthest_worst_rel_l2": hi["worst_rel_l2"],
                "ratio": ratio,
                "floor": case_floor,
                "floor_measured": findings["cases"][name].get("reference_self_sensitivity") is not None,
                "floor_margin_of_closest": margin_of_closest,
            }
            print(
                f"  {name}: {hi_label} {hi['worst_rel_l2']:.3e} vs {lo_label} "
                f"{lo['worst_rel_l2']:.3e}  ({ratio:.1f}x separation); the closest build "
                f"sits {margin_of_closest:.2f}x {'under' if margin_of_closest >= 1 else 'OVER'} "
                f"the {case_floor:g} floor"
            )
    findings["separation"] = summary

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(findings, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    else:
        print(f"\nintermediate files left in {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
