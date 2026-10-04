"""Validate dequant kernels against gguf-py reference on REAL model data.

Reads tensors from the downloaded Tiny-Moe Q4_K_M GGUF, dequantizes with
gguf-py's dequantize (authoritative reference), and compares against our
row-wise kernels. Q8_0/Q4_K/Q6_K all validated against real weights.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize

from src import quant
from src.gguf import GGML_F16, GGML_Q8_0, GGML_Q4K, GGML_Q6K, parse

MODEL = "models/Tiny-Moe.Q4_K_M.gguf"

TYPE_MAP = {GGML_Q8_0: GGMLQuantizationType.Q8_0,
            GGML_Q4K: GGMLQuantizationType.Q4_K,
            GGML_Q6K: GGMLQuantizationType.Q6_K}


def check(name, ours, ref, tol=1e-4):
    ours = np.asarray(ours, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if ours.shape != ref.shape:
        print(f"FAIL {name}: shape {ours.shape} vs {ref.shape}")
        return False
    err = np.abs(ours - ref).max()
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs err {err:.2e}")
    return ok


def main():
    gf = parse(MODEL)
    print(f"{MODEL}: {gf.tensor_count} tensors, {len(gf.metadata)} metadata kv")
    arch = gf.metadata.get("general.architecture")
    print(f"arch={arch}")

    # find one tensor of each interesting dtype
    wanted = {}
    for t in gf.tensors.values():
        if t.dtype in TYPE_MAP and t.dtype not in wanted:
            if len(t.dims) == 2 and t.dims[0] % 256 == 0:
                wanted[t.dtype] = t
    if len(wanted) < len(TYPE_MAP):
        print(f"note: dtypes present in file: "
              f"{sorted(set(t.dtype for t in gf.tensors.values()))}")

    all_ok = True
    with open(MODEL, "rb") as f:
        for dtype, t in sorted(wanted.items()):
            # read the FIRST ROW only (that's what our kernel takes)
            row_elems = t.dims[0]
            row_bytes = t.nbytes * row_elems // (t.dims[0] * t.dims[1]) if len(t.dims) == 2 else t.nbytes
            rows = t.dims[1] if len(t.dims) == 2 else 1
            row_bytes = t.nbytes // rows
            f.seek(gf.data_start + t.offset)
            raw_row = f.read(row_bytes)

            # reference: read whole tensor, dequantize with gguf-py
            f.seek(gf.data_start + t.offset)
            whole = f.read(t.nbytes)
            ref = dequantize(np.frombuffer(whole, dtype=np.uint8), TYPE_MAP[dtype])
            ref_row = ref.reshape(rows, row_elems)[0]

            ours = quant.dequant_row(raw_row, dtype, row_elems)
            all_ok &= check(f"{t.name} ({t.dtype})", ours, ref_row)

    print("ALL OK" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
