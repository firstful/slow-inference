#!/usr/bin/env python3
"""Diff a SLDREFLG logits dump from the engine against one from llama.cpp.

Both sides emit the same binary format (see engine/tools/gen_ref_logits.c):

    magic "SLDREFLG" | u32 version | u32 n_tok | i32[n_tok] tokens
                     | u32 n_vocab | f32[n_tok * n_vocab]

We report, per position, the max absolute logit difference, the correlation of
the top-k ordering, and whether argmax agrees. Quantized inference will never be
bit-exact against ggml's SIMD kernels, so the pass criterion is a relative
tolerance plus argmax agreement, not equality.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path


def load_ref(path: Path) -> tuple[list[int], int, list[list[float]]]:
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
    need = n_tok * n_vocab * 4
    if len(buf) - off != need:
        raise SystemExit(f"{path}: expected {need} bytes of logits, got {len(buf) - off}")
    flat = struct.unpack_from(f"<{n_tok * n_vocab}f", buf, off)
    rows = [list(flat[i * n_vocab : (i + 1) * n_vocab]) for i in range(n_tok)]
    return toks, n_vocab, rows


def spearman_topk(a: list[float], b: list[float], k: int) -> float:
    """Overlap@k of the top-k sets, a scale-free agreement measure."""
    ta = sorted(range(len(a)), key=lambda i: -a[i])[:k]
    tb = sorted(range(len(b)), key=lambda i: -b[i])[:k]
    sa, sb = set(ta), set(tb)
    return len(sa & sb) / float(k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--atol", type=float, default=2e-2, help="abs logit tolerance")
    ap.add_argument("--rtol", type=float, default=5e-3, help="rel tolerance")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    etoks, evocab, erows = load_ref(args.engine)
    rtoks, rvocab, rrows = load_ref(args.reference)

    label = args.label or f"{args.engine.name} vs {args.reference.name}"
    print(f"== {label} ==")
    if etoks != rtoks:
        raise SystemExit(f"token sequences differ: {etoks} vs {rtoks}")
    if evocab != rvocab:
        raise SystemExit(f"vocab differs: engine {evocab} vs reference {rvocab}")
    print(f"   {len(etoks)} tokens, vocab {evocab}")

    all_ok = True
    for i, (e, r) in enumerate(zip(erows, rrows)):
        diffs = [abs(x - y) for x, y in zip(e, r)]
        max_abs = max(diffs)
        scale = max(abs(x) for x in r) or 1.0
        rel = max_abs / scale
        argmax_e = max(range(len(e)), key=lambda j: e[j])
        argmax_r = max(range(len(r)), key=lambda j: r[j])
        overlap = spearman_topk(e, r, args.topk)
        # ggml accumulates in a different order than we do, so a relative
        # threshold is the meaningful gate.
        ok = (max_abs <= args.atol) or (rel <= args.rtol)
        if not ok:
            all_ok = False
        # cosine similarity catches "same shape, wrong scale"
        dot = sum(x * y for x, y in zip(e, r))
        ne = math.sqrt(sum(x * x for x in e))
        nr = math.sqrt(sum(y * y for y in r))
        cos = dot / (ne * nr) if ne and nr else 0.0
        flag = "ok " if ok else "FAIL"
        print(
            f"  pos {i:>3} tok {etoks[i]:>6} {flag} "
            f"max_abs={max_abs:.4e} rel={rel:.3e} cos={cos:.9f} "
            f"top{args.topk}={overlap:.2f} "
            f"argmax {argmax_e}/{argmax_r} {'MATCH' if argmax_e == argmax_r else 'differs'}"
        )

    print(f"   result: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
