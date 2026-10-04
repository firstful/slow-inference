"""Validate our dequant kernels against gguf-py's reference implementation.

Builds random float vectors, quantizes with gguf-py, then checks our
dequant_row matches gguf-py's dequantize exactly.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from gguf.quants import quantize, dequantize
from gguf.constants import GGMLQuantizationType

from src import quant
from src.gguf import GGML_Q8_0, GGML_Q4K, GGML_Q6K

TYPE_MAP = {GGML_Q8_0: GGMLQuantizationType.Q8_0,
            GGML_Q4K: GGMLQuantizationType.Q4_K,
            GGML_Q6K: GGMLQuantizationType.Q6_K}


def check(name, ours, ref, tol=1e-4):
    ours = np.asarray(ours, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if ours.shape != ref.shape:
        print(f"FAIL {name}: shape {ours.shape} vs {ref.shape}")
        return False
    err = np.abs(ours - ref).max()
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs err {err:.2e}")
    return ok


def main():
    rng = np.random.default_rng(0)
    n = 256 * 8
    all_ok = True

    for name, our_type in [("q8_0", GGML_Q8_0), ("q4_k", GGML_Q4K), ("q6_k", GGML_Q6K)]:
        x = (rng.standard_normal(n) * 3).astype(np.float32)
        q = quantize(x, TYPE_MAP[our_type])
        ref = dequantize(q, TYPE_MAP[our_type])
        # our kernels work on the flat block stream
        ours = quant.dequant_row(bytes(q), our_type, n)
        all_ok &= check(name, ours, ref)

    print("ALL OK" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
