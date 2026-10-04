"""Minimal GGUF container parser.

Reads ONLY the header + tensor directory into memory (a few KB for typical
models). Tensor data is left on disk; callers get (offset, size) and read
what they need with file.seek()/read(). No mmap of the data section, ever.

GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""
import struct
from dataclasses import dataclass

GGUF_MAGIC = 0x46554747  # "GGUF"

# GGUF metadata value types
T_U8, T_I8, T_U16, T_I16 = 0, 1, 2, 3
T_U32, T_I32, T_F32 = 4, 5, 6
T_BOOL = 7
T_STR = 8
T_ARRAY = 9
T_U64, T_I64, T_F64 = 10, 11, 12

_FMT = {
    T_U8: ("B", 1), T_I8: ("b", 1), T_U16: ("H", 2), T_I16: ("h", 2),
    T_U32: ("I", 4), T_I32: ("i", 4), T_F32: ("f", 4), T_BOOL: ("B", 1),
    T_U64: ("Q", 8), T_I64: ("q", 8), T_F64: ("d", 8),
}

# GGML element type -> numpy dtype + block structure
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGML_Q2K = 10
GGML_Q3K = 11
GGML_Q4K = 12
GGML_Q5K = 13
GGML_Q6K = 14
GGML_Q8K = 16


@dataclass
class TensorInfo:
    name: str
    dims: tuple          # ne[0], ne[1], ... (ne[0] is fastest-varying)
    dtype: int           # GGML element type
    offset: int          # absolute file offset of tensor data
    nbytes: int          # bytes on disk


@dataclass
class GGUFFile:
    path: str
    version: int
    tensor_count: int
    metadata: dict
    tensors: dict        # name -> TensorInfo
    data_start: int      # file offset where tensor data begins


class Reader:
    """Little cursor over a byte buffer of the header region."""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def _take(self, n: int) -> bytes:
        b = self.buf[self.pos:self.pos + n]
        if len(b) < n:
            raise EOFError("truncated GGUF header (read header region fully first)")
        self.pos += n
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._take(8))[0]

    def value(self, vtype: int):
        if vtype == T_STR:
            n = self.u64()
            return self._take(n).decode("utf-8", errors="replace")
        if vtype == T_ARRAY:
            etype = self.u32()
            count = self.u64()
            return [self.value(etype) for _ in range(count)]
        fmt, size = _FMT[vtype]
        return struct.unpack("<" + fmt, self._take(size))[0]


def _tensor_type_size(etype: int, ne: tuple) -> tuple:
    """Return (bytes_per_element_block, elements_per_block) for a GGML type."""
    ne0 = ne[0]
    # type sizes from ggml.h ggml_blck_size / ggml_type_size
    if etype == GGML_F32:
        return 4, 1
    if etype == GGML_F16:
        return 2, 1
    if etype == GGML_Q4_0:
        return 18, 32
    if etype == GGML_Q8_0:
        return 34, 32
    if etype == GGML_Q2K:
        return 84, 256
    if etype == GGML_Q3K:
        return 110, 256
    if etype == GGML_Q4K:
        return 144, 256
    if etype == GGML_Q5K:
        return 176, 256
    if etype == GGML_Q6K:
        return 210, 256
    if etype == GGML_Q8K:
        return 4, 8
    raise NotImplementedError(f"GGML type {etype} not supported yet")


def parse(path: str) -> GGUFFile:
    """Parse GGUF header + tensor directory. Reads only the header region."""
    with open(path, "rb") as f:
        head = f.read(24)
        magic, version = struct.unpack("<II", head[:8])
        tensor_count, kv_count = struct.unpack("<QQ", head[8:24])
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")

        # Read a generous header region, then fix up. Tensor dirs are small
        # (a few hundred tensors x ~100 bytes) but metadata arrays (vocab!)
        # can be ~1MB for a 100k vocab. Read in chunks until it parses.
        region_size = 1 << 20
        while True:
            f.seek(0)
            buf = f.read(region_size)
            r = Reader(buf)
            r._take(24)  # skip header, already parsed
            try:
                metadata = {}
                for _ in range(kv_count):
                    key = r.value(T_STR)
                    vtype = r.u32()
                    metadata[key] = r.value(vtype)

                tensors = {}
                while len(tensors) < tensor_count:
                    name = r.value(T_STR)
                    n_dims = r.u32()
                    dims = tuple(r.u64() for _ in range(n_dims))
                    dtype = r.u32()
                    offset = r.u64()
                    tensors[name] = TensorInfo(name, dims, dtype, offset, 0)
                break
            except EOFError:
                region_size *= 4
                if region_size > (1 << 30):
                    raise
                # retry with a bigger window

        # data section starts after tensor dir, aligned to 32 bytes
        data_start = (r.pos + 31) & ~31
        # recompute exact tensor byte counts now that we can
        for t in tensors.values():
            t.nbytes = _tensor_nbytes(t)

    return GGUFFile(path, version, tensor_count, metadata, tensors, data_start)


def _tensor_nbytes(t: TensorInfo) -> int:
    blk_size, blk_elems = _tensor_type_size(t.dtype, t.dims)
    numel = 1
    for d in t.dims:
        numel *= d
    return (numel // blk_elems) * blk_size


def dequant_shape(t: TensorInfo) -> tuple:
    """Shape of the tensor once dequantized, in (rows, cols) for 2D."""
    if len(t.dims) == 2:
        return t.dims[1], t.dims[0]   # GGML stores (ne0=cols, ne1=rows)
    return tuple(reversed(t.dims))
