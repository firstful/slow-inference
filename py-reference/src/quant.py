"""Dequantization kernels for GGML quant formats -> float32 numpy.

Row-wise so the runner can dequantize one weight row at a time (minimal RSS).
Layouts follow ggml/src/ggml-quants.c dequantize_row_* exactly.
Implemented: F32, F16, Q8_0, Q4_K, Q6_K. (Q2_K/Q3_K later if a model needs it.)
"""
import numpy as np

from .gguf import GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q4K, GGML_Q6K

BLK = {  # type -> (elements per block, bytes per block)
    GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_Q8_0: (32, 34),
    GGML_Q4K: (256, 144), GGML_Q6K: (256, 210),
}


def dequant_row(raw: bytes, dtype: int, ne0: int) -> np.ndarray:
    """Dequantize ONE row of `ne0` elements. raw holds exactly the row's bytes."""
    if dtype == GGML_F32:
        return np.frombuffer(raw, dtype="<f4").copy()
    if dtype == GGML_F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if dtype == GGML_Q8_0:
        return _q8_0(raw, ne0)
    if dtype == GGML_Q4K:
        return _q4_k(raw, ne0)
    if dtype == GGML_Q6K:
        return _q6_k(raw, ne0)
    raise NotImplementedError(f"dequant for ggml type {dtype}")


def _q8_0(raw: bytes, ne0: int) -> np.ndarray:
    nblk = ne0 // 32
    raw_a = np.frombuffer(raw, dtype=np.uint8).reshape(nblk, 34)
    d = raw_a[:, 0:2].copy().view(np.float16).astype(np.float32)      # (nblk,1)
    qs = raw_a[:, 2:34].copy().view(np.int8).astype(np.float32)       # (nblk,32)
    return (qs * d).reshape(ne0)


def _get_scale_min_k4(j: int, q: np.ndarray):
    """ggml-quants.c get_scale_min_k4: q = 12-byte scales array."""
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), \
           (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def _q4_k(raw: bytes, ne0: int) -> np.ndarray:
    """Q4_K: 256-elem blocks. 2B d, 2B dmin, 12B scales, 128B nibbles.
    Per 64-elem group: 32 bytes; low nibbles -> first 32 outputs (scale 2g),
    high nibbles -> next 32 outputs (scale 2g+1)."""
    nblk = ne0 // 256
    out = np.empty(ne0, dtype=np.float32)
    pos = 0
    for b in range(nblk):
        off = b * 144
        buf = np.frombuffer(raw[off:off + 144], dtype=np.uint8)
        d = float(buf[0:2].copy().view(np.float16)[0])
        dmin = float(buf[2:4].copy().view(np.float16)[0])
        sc_raw = buf[4:16]
        qs = buf[16:144]
        for g in range(4):
            s0, m0 = _get_scale_min_k4(2 * g, sc_raw)
            s1, m1 = _get_scale_min_k4(2 * g + 1, sc_raw)
            d1, m1f = d * s0, dmin * m0
            d2, m2f = d * s1, dmin * m1
            q = qs[g * 32:(g + 1) * 32].astype(np.uint8)
            out[pos:pos + 32] = d1 * (q & 0xF) - m1f
            out[pos + 32:pos + 64] = d2 * (q >> 4) - m2f
            pos += 64
    return out


def _q6_k(raw: bytes, ne0: int) -> np.ndarray:
    """Q6_K: 256-elem blocks, 210 bytes = 2 superblocks of 128 elements.
    Block layout: 128B ql | 64B qh | 16B scales (int8) | 2B d (f16).
    Per superblock s: ql = ql[s*64:(s+1)*64], qh = qh[s*32:(s+1)*32],
    8 scales from scales[s*8:(s+1)*8].
    Values: lo nibbles of ql (64) then hi nibbles (64); qh gives 4 bit-planes
    of 2 bits each (b&3, b>>2&3, b>>4&3, b>>6&3). q = ql|(qh<<4) - 32,
    8 groups of 16, each scaled by d*scale[g]."""
    nblk = ne0 // 256
    out = np.empty(ne0, dtype=np.float32)
    pos = 0
    for b in range(nblk):
        off = b * 210
        buf = np.frombuffer(raw[off:off + 210], dtype=np.uint8)
        ql_all = buf[0:128]
        qh_all = buf[128:192]
        scales_all = buf[192:208].copy().view(np.int8).astype(np.float32)
        d = float(buf[208:210].copy().view(np.float16)[0])
        vals = np.empty(256, dtype=np.float32)
        for sb in range(2):
            ql = ql_all[sb * 64:(sb + 1) * 64]
            qh = qh_all[sb * 32:(sb + 1) * 32]
            scales = scales_all[sb * 8:(sb + 1) * 8]
            ql_nib = np.concatenate([(ql & 0xF).astype(np.uint8),
                                     (ql >> 4).astype(np.uint8)])
            qh_bits = np.concatenate([
                (qh & 3).astype(np.uint8), ((qh >> 2) & 3).astype(np.uint8),
                ((qh >> 4) & 3).astype(np.uint8), ((qh >> 6) & 3).astype(np.uint8)])
            q = (ql_nib | (qh_bits << np.uint8(4))).astype(np.int8) - np.int8(32)
            q = q.astype(np.float32).reshape(8, 16)
            vals[sb * 128:(sb + 1) * 128] = (d * scales.reshape(8, 1) * q).ravel()
        out[pos:pos + 256] = vals
        pos += 256
    return out
