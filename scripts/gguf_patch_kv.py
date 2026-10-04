#!/usr/bin/env python3
"""Rewrite string/uint metadata keys in a GGUF, byte-exactly, without a writer.

Why this exists
---------------
Three of llama.cpp's `tokenizer_st_partition` attribute rules are keyed on
metadata that has nothing to do with the vocabulary itself
(`llama-vocab.cpp`, `llama_vocab::impl::load()`):

  * `LLAMA_TOKEN_ATTR_LSTRIP` on `<mask>` when `tokenizer.ggml.pre` contains
    `jina-v2-de` / `jina-v2-es` / `jina-v2-code`, **or** `general.architecture`
    contains `nomic-bert-moe` / `jina-bert-v3`;
  * `LLAMA_TOKEN_ATTR_RSTRIP` on every `cache_special_tokens` id plus `</s>`
    when the lowercased `general.name` contains `phi-3` or `phi3`;
  * `LLAMA_TOKEN_ATTR_LSTRIP` on `[MASK]` when `general.name` contains
    `modern-bert`.

Neither of the reachable two is reachable on the three GGUFs in `models/`, so the
rules have to be tested on *the same vocabulary relabelled* — which is what this
script is for. It produces `general.name = "synthetic-gpt2"` ->
`general.name = "modern-bert-synthetic"` with every byte of the tensors
untouched, so any difference in `llama_tokenize`'s output is attributable to the
relabelling and to nothing else.

Regenerating a fixture per label instead would change the tensor bytes too, and
in a tokenizer harness a difference in the *vocabulary* silently explains away a
difference in the *rule*. That is the exact confusion this file exists to make
impossible.

Layout (ggml/src/gguf.cpp)
-------------------------
    magic "GGUF" | version u32 | n_tensors u64 | n_kv u64
    kv pairs:     n_kv * (key, type u32, value)            <- read at gguf.cpp:545
    tensor infos: n_tensors * (name, n_dims u32, dims u64[n_dims],
                               type u32, offset u64)         <- read at gguf.cpp:624
    pad to general.alignment (default 32)
    tensor data

The KV block comes *first*. That is easy to get backwards — the counts appear in
the other order in the header — and getting it wrong reads as a truncated file
rather than as a wrong field, because the tensor-name length prefixes still
parse as plausible integers for a while.

Two properties make the rewrite a pure prefix swap:

  * a tensor's `offset` is **relative to the tensor-data base**
    (`gguf.cpp:214`, applied at `gguf.cpp:882` as `data->data + info.offset`),
    so moving that base leaves every offset correct;
  * nothing in the KV block addresses the tensor data.

So the tensor data is copied verbatim from the old base to the new one and
nothing inside it is touched. Only string, uint32 and bool values may be edited;
anything else (an array or a blob) is refused rather than mangled.

Usage:
    python3 scripts/gguf_patch_kv.py models/synthetic-gpt2.gguf --print general.name
    python3 scripts/gguf_patch_kv.py in.gguf out.gguf --set general.name=modern-bert-synth
    python3 scripts/gguf_patch_kv.py in.gguf out.gguf --set general.alignment:64
    python3 scripts/gguf_patch_kv.py in.gguf out.gguf --rename-token w5='<|endoftext|>'

Why `--rename-token` exists
---------------------------
llama.cpp's phi-3 attribute branch ends with three lookups by token *text*
(`llama-vocab.cpp:3039`), and `_set_token_attr` is `token_to_id.at(token)` — a
throw, not a miss. A vocabulary missing any of them cannot be loaded with that
branch switched on, so the branch cannot be observed without one that has all
three. No vocabulary on disk has `<|endoftext|>` *and* `<unk>` *and* `<s>` except
Qwen2.5-3B, which is 1.9 GB. Renaming one otherwise-unused token of the
128-token strip fixture keeps the experiment at a few kilobytes and changes
nothing the attribute dump depends on: which ids carry RSTRIP is decided by
`cache_special_tokens` and by the three names, not by the vocabulary's contents.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10

#: gguf_type -> (struct format, width). Types whose raw bytes can be carried
#: through a rewrite untouched.
_SCALARS = {
    0: ("<B", 1),  # UINT8
    1: ("<b", 1),  # INT8
    2: ("<H", 2),  # UINT16
    3: ("<h", 2),  # INT16
    4: ("<I", 4),  # UINT32
    5: ("<i", 4),  # INT32
    6: ("<f", 4),  # FLOAT32
    7: ("<?", 1),  # BOOL
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}


class Reader:
    def __init__(self, buf: bytes, pos: int = 0) -> None:
        self.buf = buf
        self.pos = pos

    def take(self, n: int) -> bytes:
        out = self.buf[self.pos : self.pos + n]
        if len(out) != n:
            raise ValueError("truncated GGUF")
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> bytes:
        return self.take(self.u64())


class Kv:
    """One KV pair, kept as raw bytes so an untouched key re-encodes identically."""

    __slots__ = ("key", "vtype", "raw", "value")

    def __init__(self, key: bytes, vtype: int, raw: bytes, value: object) -> None:
        self.key = key
        self.vtype = vtype
        self.raw = raw
        self.value = value

    def encode(self) -> bytes:
        return struct.pack("<Q", len(self.key)) + self.key + struct.pack("<I", self.vtype) + self.raw

    def set_string(self, text: str) -> None:
        b = text.encode("utf-8")
        self.vtype = GGUF_TYPE_STRING
        self.raw = struct.pack("<Q", len(b)) + b
        self.value = text

    def set_string_array(self, values: list[bytes]) -> None:
        """Re-encode a whole string array. `self.value` stays None: arrays are
        never decoded, and decoding them here would be a second parser to keep
        in step with `_read_value`."""
        out = bytearray(struct.pack("<I", GGUF_TYPE_STRING) + struct.pack("<Q", len(values)))
        for v in values:
            out += struct.pack("<Q", len(v)) + v
        self.vtype = GGUF_TYPE_ARRAY
        self.raw = bytes(out)


class Tensor:
    __slots__ = ("name", "dims", "ttype", "offset")

    def __init__(self, name: bytes, dims: list[int], ttype: int, offset: int) -> None:
        self.name = name
        self.dims = dims
        self.ttype = ttype
        self.offset = offset

    def encode(self) -> bytes:
        out = bytearray(struct.pack("<Q", len(self.name)))
        out += self.name
        out += struct.pack("<I", len(self.dims))
        for d in self.dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", self.ttype)
        # Relative to the tensor-data base, which is what ggml.cpp:214 documents
        # and what makes the whole rewrite a prefix swap.
        out += struct.pack("<Q", self.offset)
        return bytes(out)


class Gguf:
    def __init__(self, version: int, tensors: list[Tensor], kvs: list[Kv], data: bytes) -> None:
        self.version = version
        self.tensors = tensors
        self.kvs = kvs
        self.data = data

    @property
    def alignment(self) -> int:
        for kv in self.kvs:
            if kv.key == b"general.alignment" and kv.vtype == GGUF_TYPE_UINT32:
                return int(kv.value) or 32
        return 32

    def get(self, key: str) -> object:
        kb = key.encode()
        for kv in self.kvs:
            if kv.key == kb:
                return kv.value
        return None

    def encode(self) -> bytes:
        out = bytearray()
        out += GGUF_MAGIC
        out += struct.pack("<I", self.version)
        out += struct.pack("<Q", len(self.tensors))
        out += struct.pack("<Q", len(self.kvs))
        for kv in self.kvs:
            out += kv.encode()
        for t in self.tensors:
            out += t.encode()
        pad = self.alignment - (len(out) % self.alignment)
        out += b"\x00" * (pad % self.alignment)
        out += self.data
        return bytes(out)


def _read_value(r: Reader, vtype: int) -> tuple[bytes, object]:
    """Return (raw bytes, decoded value) for one KV value, advancing `r`."""
    if vtype in _SCALARS:
        _, width = _SCALARS[vtype]
        raw = r.take(width)
        fmt, _ = _SCALARS[vtype]
        return raw, struct.unpack(fmt, raw)[0]
    if vtype == GGUF_TYPE_STRING:
        # `Kv.raw` must round-trip, and the value's own length prefix is part of
        # the encoding, so it is captured rather than re-derived. (Reader.string()
        # returns payload only, which is what makes this easy to get wrong: the
        # output still parses as a string, just with the *next* key's bytes
        # eaten.)
        n = r.u64()
        payload = r.take(n)
        return struct.pack("<Q", n) + payload, payload.decode("utf-8", "replace")
    if vtype == GGUF_TYPE_ARRAY:
        elem = r.u32()
        count = r.u64()
        out = bytearray(struct.pack("<I", elem) + struct.pack("<Q", count))
        if elem in _SCALARS:
            _, width = _SCALARS[elem]
            out += r.take(count * width)
        elif elem == GGUF_TYPE_STRING:
            for _ in range(count):
                n = r.u64()
                out += struct.pack("<Q", n) + r.take(n)
        else:
            raise ValueError(f"unsupported array element type {elem}")
        # Arrays are copied verbatim and never decoded: no rule this script
        # serves needs one, and a half-decoded array is a wrong answer.
        return bytes(out), None
    raise ValueError(f"unsupported KV type {vtype}")


def _decode_string_array(raw: bytes) -> list[bytes]:
    """The inverse of `Kv.set_string_array` for a string-array payload.

    Only the element type and the elements are interpreted; the count is taken
    from the header rather than from the buffer length, so a truncated array
    raises instead of silently yielding a short vocabulary.
    """
    if len(raw) < 12:
        raise ValueError("string array too short")
    elem, count = struct.unpack_from("<IQ", raw, 0)
    if elem != GGUF_TYPE_STRING:
        raise ValueError(f"array element type {elem} is not a string")
    pos = 12
    out = []
    for _ in range(count):
        n = struct.unpack_from("<Q", raw, pos)[0]
        pos += 8
        if pos + n > len(raw):
            raise ValueError("string array truncated")
        out.append(raw[pos : pos + n])
        pos += n
    return out


def parse(buf: bytes) -> Gguf:
    r = Reader(buf)
    if r.take(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF (bad magic)")
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    kvs = []
    for _ in range(n_kv):
        key = r.string()
        vtype = r.u32()
        raw, value = _read_value(r, vtype)
        kvs.append(Kv(key, vtype, raw, value))
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.u64() for _ in range(r.u32())]
        tensors.append(Tensor(name, dims, r.u32(), r.u64()))
    align = 32
    for kv in kvs:
        if kv.key == b"general.alignment" and kv.vtype == GGUF_TYPE_UINT32:
            align = int(kv.value) or 32
    r.pos += (align - (r.pos % align)) % align
    return Gguf(version, tensors, kvs, buf[r.pos :])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path, nargs="?", help="omit to only print")
    ap.add_argument("--print", action="append", default=[], metavar="KEY")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=STRING",
        help="set a string key (repeatable; creates the key if absent)",
    )
    ap.add_argument(
        "--rename-token",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "rename one entry of the tokenizer.ggml.tokens array, in place "
            "(repeatable). The token count never changes, so every parallel "
            "array (scores, token_type) stays aligned and no id moves; only the "
            "KV block grows or shrinks, which moves the tensor-data base and "
            "therefore nothing else."
        ),
    )
    args = ap.parse_args()

    gguf = parse(args.src.read_bytes())

    for key in args.print:
        print(f"{key}={gguf.get(key)!r}")

    if args.rename_token:
        tokens_kv = next(
            (kv for kv in gguf.kvs if kv.key == b"tokenizer.ggml.tokens"), None
        )
        if tokens_kv is None:
            raise SystemExit("--rename-token needs a tokenizer.ggml.tokens array")
        tokens = _decode_string_array(tokens_kv.raw)
        by_text: dict[bytes, list[int]] = {}
        for i, t in enumerate(tokens):
            by_text.setdefault(t, []).append(i)
        for spec in args.rename_token:
            old, sep, new = spec.partition("=")
            if not sep:
                raise SystemExit(f"--rename-token wants OLD=NEW, got {spec!r}")
            hit = by_text.get(old.encode("utf-8"))
            if not hit:
                raise SystemExit(f"--rename-token: no token named {old!r}")
            if len(hit) > 1:
                raise SystemExit(
                    f"--rename-token: {old!r} is ambiguous at ids {hit}; a GGUF "
                    "with duplicate token text has no single right answer here"
                )
            i = hit[0]
            nb = new.encode("utf-8")
            if nb in by_text:
                raise SystemExit(
                    f"--rename-token: {new!r} already exists at id {by_text[nb][0]}, "
                    "and token_to_id would silently keep the old id"
                )
            print(f"note: renamed token {i} {old!r} -> {new!r}", file=sys.stderr)
            tokens[i] = nb
        tokens_kv.set_string_array(tokens)

    if args.dst is None:
        if not args.print and not args.rename_token:
            ap.error("nothing to do: pass --print and/or --set with a destination")
        return 0

    for spec in args.set:
        key, sep, value = spec.partition("=")
        if not sep:
            raise SystemExit(f"--set wants KEY=STRING, got {spec!r}")
        kb = key.encode()
        for kv in gguf.kvs:
            if kv.key == kb:
                kv.set_string(value)
                break
        else:
            gguf.kvs.append(Kv(kb, GGUF_TYPE_STRING, b"", None))
            gguf.kvs[-1].set_string(value)
            print(f"note: {key!r} was absent; appended", file=sys.stderr)

    args.dst.write_bytes(gguf.encode())
    print(
        f"wrote {args.dst}: {len(gguf.kvs)} keys, {len(gguf.tensors)} tensors, "
        f"alignment {gguf.alignment}, {len(gguf.data)} bytes of tensor data",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())