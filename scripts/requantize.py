#!/usr/bin/env python3
"""Rewrite a GGUF with every tensor moved to another ggml type, verbatim otherwise.

Why this exists
---------------
The parity suite has been able to prove architecture semantics (tensor naming,
norm placement, RoPE pairing, GQA grouping, MLA absorption) since tick 14, but
only on the *synthetic F32 fixtures*. Every real model on disk is Q4_K_M or
Q6_K, and on those the engine and llama.cpp disagree by cos 0.996-0.9993 --
large enough that "quantization noise" and "a real semantic bug" are not
distinguishable by inspection. Ticks 16 and 18 established the methodology:
when a two-way diff cannot say which side is wrong, build a third reference.

This script is the missing third lever. `--to f32` removes quantization from
the question entirely without changing a single architecture byte:

    real Q4_K_M model  ->  [this script, --to f32]  ->  metadata-identical F32 model

Measured on Tiny-Moe (tick 19), that one substitution settles the question:

| comparison                                   | worst rel L2 | cos      |
|----------------------------------------------|--------------|----------|
| engine F32 vs llama.cpp F32                  | 4.1e-6       | 1.000000 |
| engine Q4_K_M vs engine F32                  | **0.0**      | 1.000000 |
| engine Q4_K_M vs llama.cpp Q4_K_M            | 1.2e-1       | 0.9972   |
| llama.cpp F32 vs llama.cpp Q4_K_M            | 1.2e-1       | 0.9972   |

The last two agree to four significant figures at every position, and the
middle row is bit-identical, so `llama(Q4) - engine == llama(Q4) - exact`:
the entire gap is *llama.cpp's own sensitivity to 4-bit weights*, and the
engine computes the dequantized-weight forward pass exactly. No semantic bug
was hiding in the real-model gap; it was the definition of what a 4-bit model
computes.

The other direction, `--to q4_k_m`, quantizes an F32 fixture. That makes the
finding durable and cheap: the engine dequantizes a Q4_K tensor to exactly the
f32 values ggml does (`tests/quant_golden.rs`, bit for bit) and then runs
identical f32 arithmetic, so *engine(Q4_K_M) must be bit-identical to
engine(F32)*. `engine/tests/quant_gap.rs` asserts exactly that on a 70 KB
checked-in fixture, so a future kernel change that breaks the invariant fails
the suite instead of quietly moving real-model logits.

What is and is not preserved
----------------------------
Preserved verbatim: every metadata key, with its exact GGUF value type and
array-vs-scalar shape. llama.cpp's arch-specific defaults (`f_attention_scale`,
SWA period, `rope_freq_base_swa`, ...) are decided from metadata, so a
lossy metadata copy would silently change the arithmetic being measured.
`general.name` and `general.file_type` are rewritten -- they are descriptive
only, and leaving `file_type` at 15 would make both llama.cpp and our own
`--dtype` report an F32 file as quantized (or vice versa).

Changed: every tensor's type. F32 output is produced with gguf-py's
`dequantize()`, which is ggml's own reference `dequantize_row_*` implementation
-- the same authority `engine/tests/quant_golden.rs` pins the engine's kernels
against. So widening is exact, not lossy: the F32 file contains precisely the
f32 values both engines compute *before* their matmul arithmetic differs.
Quantized output uses gguf-py's `quantize()`, i.e. ggml's `from_float_ref`.

Usage:
    python3 scripts/requantize.py --to f32 \
        --in models/Tiny-Moe.Q4_K_M.gguf --out /tmp/tinymoe-f32.gguf
    python3 scripts/requantize.py --to q4_k_m \
        --in engine/tests/data/fixtures/synthetic-llama.gguf \
        --out engine/tests/data/fixtures/synthetic-llama-q4km.gguf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# `general.file_type` describes the file, not the model. Rewritten so an F32
# output is not reported as Q4_K_M by llama.cpp's banner or by our `--dtype`.
FILE_TYPE_F32 = 0


def import_gguf():
    try:
        import gguf  # type: ignore
    except ModuleNotFoundError:
        candidate = Path(
            __import__("os").environ.get("LLAMA_CPP_GGUF_PY", Path.home() / "llama.cpp" / "gguf-py")
        )
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
        try:
            import gguf  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "gguf-py is required. Set LLAMA_CPP_GGUF_PY or PYTHONPATH=~/llama.cpp/gguf-py."
            ) from exc
    return gguf


def arch_of(reader) -> str:
    for key in ("general.architecture", "general.type"):
        kv = reader.fields.get(key)
        if kv is not None:
            v = kv.contents()
            if isinstance(v, str):
                return v
    raise SystemExit("input has no general.architecture key")


def nbytes_hint(path: Path) -> bool:
    """True when the file is big enough that holding the output in RAM is unwise."""
    try:
        return path.stat().st_size > 512 * 1024 * 1024
    except OSError:
        return False


def pin_spill_dir(out: Path) -> None:
    """Make gguf-py's spill file land next to the output, not in $TMPDIR.

    `GGUFWriter(use_temp_file=True)` holds tensor payloads in a
    `SpooledTemporaryFile` that rolls over to a real file on disk after
    256 MB, and `tempfile` picks that file's directory from `$TMPDIR`. On a
    machine where `/tmp` is a tmpfs -- which is the normal case on a container
    or a workstation with `/tmp` mounted `tmpfs` -- the spill is RAM again, and
    a widening whose output is larger than `/tmp` dies part-way through with

        OSError: 524288 requested and 105472 written

    or `Not enough free space to write ... bytes`, having already produced
    nothing. That reads like a memory shortage, and it is not one: on the 3B
    Q4_K_M model the output is 12.4 GB while the largest single tensor
    dequantizes to 1.2 GB, so the job needs ~1.5 GB of RAM and 12.4 GB of
    *disk*. This is what made "the F32 identity needs 12.4 GB" look like a
    blocked experiment rather than a misdirected one.

    Pointing `tempfile` at the output's own directory makes the spill land on
    the filesystem that has to hold the result anyway.
    """
    import tempfile

    spill = out.parent
    try:
        spill.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create spill directory {spill}: {exc}")
    tempfile.tempdir = str(spill)


def report_headroom(out: Path, reader) -> None:
    """Print the output size and the free space before writing a byte.

    The failure above is an OSError from `ndarray.tofile` part-way through,
    which reports neither the size that was needed nor where it was trying to
    write. Both are knowable up front, and saying them out loud turns a
    five-minute mystery into a one-line answer.
    """
    try:
        free = os.statvfs(out.parent).f_bavail * os.statvfs(out.parent).f_frsize
    except OSError:
        return
    n_elem = sum(int(t.n_elements) for t in reader.tensors)
    need = n_elem * 4  # the f32 upper bound; quantized targets are smaller
    print(
        f"# widening {n_elem/1e9:.2f}e9 elements -> up to {need/1e9:.2f} GB in "
        f"{out.parent} ({free/1e9:.2f} GB free)"
    )
    if free < need:
        print(
            f"# WARNING: short by ~{(need - free)/1e9:.2f} GB; the write will fail "
            f"part-way with an opaque OSError"
        )


def copy_metadata(reader, writer, name: str, file_type: int) -> int:
    """Re-emit every KV with its original value type. Returns keys rewritten.

    The value *type* matters as much as the value: llama.cpp dispatches
    `get_key<u32>` on the stored GGUF type, so a `uint32` re-emitted as
    `uint64` is a different key as far as the arch-specific loaders are
    concerned and would silently fall back to a default.
    """
    rewritten = 0
    for key, field in reader.fields.items():
        # GGUFWriter.__init__ already emitted general.architecture.
        if key == "general.architecture":
            continue
        # `GGUFReader` *synthesises* three fields from the header before it
        # parses anything: GGUF.version, GGUF.tensor_count, GGUF.kv_count. They
        # are not in the file, so copying them back emits each one twice and the
        # result cannot be read at all ("Duplicate GGUF.version already in list").
        if key.startswith("GGUF."):
            continue
        val = field.contents()
        vtype = field.types[0]
        sub_type = field.types[-1] if len(field.types) > 1 else None

        if key == "general.name":
            val = name
            rewritten += 1
        elif key == "general.file_type":
            val = file_type
            rewritten += 1
        writer.add_key_value(key, val, vtype, sub_type)
    return rewritten


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument(
        "--to",
        default="f32",
        help="target ggml type for every tensor, e.g. f32, f16, q4_k_m, q6_k",
    )
    ap.add_argument("--name", default=None, help="override general.name in the output")
    ap.add_argument(
        "--keep-fp32",
        nargs="*",
        default=None,
        help="tensors whose name starts with one of these prefixes stay F32 "
        "(1-D norms and router rows are conventionally left that way anyway).",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="only convert tensors whose name starts with one of these prefixes "
        "(everything else is copied through byte for byte). For bisecting which "
        "tensor's quantization accounts for a divergence.",
    )
    ap.add_argument(
        "--type-map",
        nargs="*",
        default=None,
        help="per-tensor overrides, 'glob=GGML_TYPE' (e.g. '*.attn_v.weight=Q6_K'). "
        "Lets a file carry a mixture of types the way a real Q4_K_M does.",
    )
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--quantize-1d",
        action="store_true",
        help="also quantize 1-D tensors (norm weights, biases). Off by default "
        "because llama.cpp's own quantizer never does (llama-quant.cpp:294) and "
        "the CPU backend cannot run a mixed f32/q4_0 ggml_mul.",
    )
    ap.add_argument(
        "--quantize-token-embd",
        action="store_true",
        help="also quantize token_embd.weight. Off by default because llama.cpp "
        "cannot run a sub-8-bit embedding table (no mixed f32/q4_0 add).",
    )
    ap.add_argument(
        "--spill",
        action="store_true",
        help="always spool tensor payloads to a temp file instead of RAM "
        "(implied for inputs over 512 MB)",
    )
    args = ap.parse_args()

    gguf = import_gguf()
    from gguf.constants import GGMLQuantizationType  # type: ignore

    target_name = args.to.upper()
    target = getattr(GGMLQuantizationType, target_name, None)
    if target is None:
        names = ", ".join(
            n for n, v in vars(GGMLQuantizationType).items() if isinstance(v, GGMLQuantizationType)
        )
        raise SystemExit(f"unknown ggml type {args.to!r}. available: {names}")

    reader = gguf.GGUFReader(args.src, "r")
    arch = arch_of(reader)
    name = args.name or f"{args.src.stem}-{target_name.lower()}"

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    if args.dst.exists():
        args.dst.unlink()

    # use_temp_file spools tensor payloads to disk past 256 MB. Without it the
    # writer keeps every dequantized tensor alive until write_tensors_to_file(),
    # so a 3B-parameter model needs 12 GB of RAM before it writes a single byte.
    # A quantization bisect is exactly the job where the model is large and the
    # machine is already busy with something else.
    #
    # Spooling only helps if the spill directory is a real filesystem, so pin it
    # to the output's own directory before the writer can create a spill file.
    # See pin_spill_dir() for what happens without this.
    if args.spill or nbytes_hint(args.src):
        pin_spill_dir(args.dst)
    report_headroom(args.dst, reader)
    writer = gguf.GGUFWriter(args.dst, arch, use_temp_file=args.spill or nbytes_hint(args.src))
    n_rewritten = copy_metadata(
        reader, writer, name, FILE_TYPE_F32 if target == GGMLQuantizationType.F32 else int(target)
    )

    convert = (lambda tname: True) if args.only is None else (
        lambda tname: any(tname.startswith(p) for p in args.only)
    )
    # llama.cpp's own quantizer never quantizes a 1-D tensor
    # (`if (ggml_n_dims(tensor) < 2) return false;`, llama-quant.cpp:294), and it
    # has to: `build_norm` ends in `ggml_mul(cur, weight)` and the CPU
    # `binary_op` dispatcher accepts only homogeneous or bf16/f16/f32 operand
    # pairs, so a Q4_0 `output_norm.weight` aborts inside graph_compute with
    #   binary_op: unsupported types: dst: f32, src0: f32, src1: q4_0
    # rather than producing a wrong number. Same reason token_embd.weight is
    # held back: its lookup is `ggml_add(zero_f32, get_rows(tok_embd))`, and real
    # quantizers floor that category at Q5_K for Q4_K_M anyway. Both rules live
    # in `wants()` and apply only when the *target* is a quantized type, so
    # `--to f32` still widens every tensor.
    keep_fp32 = tuple(args.keep_fp32 or ())

    # `Q4_K_M` is not a ggml type, it is a *mixture*: llama.cpp picks a type per
    # tensor from an importance heuristic (`llama-quant.cpp`, tensor_category
    # plus `use_more_bits`), so it cannot be reproduced by naming one type. The
    # mixture rule the released files actually follow is close enough to
    # `--type-map '*.attn_v.weight=Q6_K' '*.ffn_down.weight=Q6_K'` to be worth
    # having: one file, two types, which is also what exercises the engine's
    # per-tensor type dispatch rather than a single kernel.
    type_map = []
    for spec in args.type_map or []:
        pat, _, tname = spec.partition("=")
        tval = getattr(GGMLQuantizationType, tname.upper(), None)
        if tval is None:
            raise SystemExit(f"--type-map: unknown ggml type {tname!r}")
        import fnmatch

        type_map.append((pat, tval))

    def type_for(tname: str):
        for pat, tval in type_map:
            if fnmatch.fnmatch(tname, pat):
                return tval
        return target

    def wants(tname: str, ndims: int):
        """Returns (convert?, target type or reason) — or (False, reason)."""
        if not convert(tname):
            return False, "copied (--only)"
        want = type_for(tname)
        # GGML_QUANT_SIZES is (block_size, type_size) — 1 element per block for the
        # scalar types. Unpacking it the other way round makes F32 look quantized
        # (`_, type_size` is 4, and 4 > 1), which then fires the two keep-rules
        # below on an *F32* target and silently leaves the offending tensor
        # un-widened.
        block, _ = gguf.quants.GGML_QUANT_SIZES[want]
        # The two keep-rules below exist because *quantizing* those tensors makes
        # llama.cpp abort, not because widening them is wrong. `--to f32` must
        # convert everything, or the "F32 widening" still carries a 4-bit
        # embedding table and the whole Q-versus-F32 comparison is not the one it
        # claims to be.
        quantized_target = block > 1
        if quantized_target and ndims < 2 and not args.quantize_1d:
            return False, "F32 kept (llama-quant.cpp:294 never quantizes 1-D)"
        if quantized_target and tname == "token_embd.weight" and not args.quantize_token_embd:
            return False, "F32 kept (llama.cpp has no mixed f32/q4_0 add)"
        if any(tname.startswith(p) for p in keep_fp32):
            return False, "kept (--keep-fp32)"
        if quantized_target:
            if int(t.shape[0]) % block != 0:
                # A quantized row must be a whole number of blocks. The tiny
                # synthetic fixtures deliberately use non-power-of-two widths
                # (64, 96) precisely to expose stride bugs, which is the same
                # reason they cannot be Q4_K-quantized: 96 is not a multiple of
                # Q4_K's 256-element block. Say so rather than reporting these
                # as ordinary passthrough.
                return False, f"kept (row {int(t.shape[0])} not a multiple of {want.name}'s {block})"
        return True, want

    n_conv = n_copied = n_elems = 0
    by_type: dict[str, int] = {}
    for t in reader.tensors:
        name_ = t.name
        # gguf-py's writer wants a uint8 payload described by the tensor's *byte*
        # shape when `raw_dtype` is set, so the passthrough branch has to convert.
        # Three traps, all of which corrupt the output *silently* unless caught:
        #   * `add_tensor_info`/`write_ti_data_to_file` store the shape in numpy
        #     order and write it out reversed, while `ReaderTensor.shape` is GGUF
        #     ne order. Passing `t.shape` verbatim emits a transposed tensor --
        #     which is why the first run produced a token_embd of 4096x768 for a
        #     model whose embedding table is 768 wide.
        #   * `quant_shape_to_byte_shape` indexes the last element, so it wants
        #     numpy order too.
        #   * Feeding a uint8 array the *logical* shape makes `add_tensor_info`
        #     convert the shape back to bytes, so a 768-float norm weight would
        #     be written as 6144 elements.
        np_shape = [int(d) for d in t.shape][::-1]
        convert_it, want = wants(name_, len(t.shape))
        if convert_it:
            if want == GGMLQuantizationType.F32:
                arr = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type)
                arr = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
                writer.add_tensor(name_, arr, raw_shape=np_shape)
            else:
                f32 = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type)
                f32 = np.ascontiguousarray(f32, dtype=np.float32).reshape(-1)
                q = gguf.quants.quantize(f32, want)
                writer.add_tensor(
                    name_,
                    np.ascontiguousarray(q).view(np.uint8),
                    raw_shape=list(gguf.quants.quant_shape_to_byte_shape(np_shape, want)),
                    raw_dtype=want,
                )
            n_conv += 1
            n_elems += int(t.n_elements)
            label = want.name
        else:
            # Copied through byte for byte, shape and all.
            byte_shape = list(gguf.quants.quant_shape_to_byte_shape(np_shape, t.tensor_type))
            writer.add_tensor(
                name_,
                np.ascontiguousarray(np.asarray(t.data).view(np.uint8)),
                raw_shape=byte_shape,
                raw_dtype=t.tensor_type,
            )
            n_copied += 1
            label = want if str(want).startswith(str(t.tensor_type.name)) else f"{t.tensor_type.name} ({want})"
        by_type[label] = by_type.get(label, 0) + 1
        if not args.quiet and (n_conv + n_copied) % 20 == 0:
            print(f"  {n_conv + n_copied} tensors...", flush=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    mixture = ", ".join(f"{n}x {k}" for k, n in sorted(by_type.items()))
    print(
        f"wrote {args.dst} ({args.dst.stat().st_size} bytes, arch={arch}, "
        f"default type {target_name})\n"
        f"  {n_conv} tensors converted ({n_elems} elements), {n_copied} copied through\n"
        f"  types: {mixture}\n"
        f"  {n_rewritten} metadata keys rewritten (general.name, general.file_type)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())