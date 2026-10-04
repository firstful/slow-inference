#!/bin/sh
# Regenerate tests/data/norm_golden.bin from llama.cpp's own LayerNorm kernel.
#
#   sh engine/scripts/gen_norm_golden.sh
#
# Needs a built llama.cpp ($LLAMA_CPP/build/bin/libggml*.so), the same one
# scripts/gen_dot_golden.sh needs.
#
# -ffp-contract=off is load-bearing for the same reason it is in
# gen_dot_golden.sh. The golden records the *kernel's* bits, and the tool's own
# `seq_f32` column is a stand-in for the engine's `sum::<f32>()` lines, which do
# not contract. Build with contraction on and that column would quietly become an
# fma chain and stop being evidence about the engine. Nothing in the LayerNorm
# reduction uses an FMA, so there is no explicit fmaf() here to preserve.
#
# The tool refuses to write the file unless its decomposition check passes: given
# the kernel's own mean and its own variance, (x-mean)*scale*w+b has to
# reproduce the graph's output bit for bit. If it does not, the two reduction
# columns are a plausible story about the kernel rather than the kernel's values,
# and a golden built on them would be worse than no golden.
#
# The tool calls the reference's own exported ggml_vec_cvar_f32 for the variance
# rather than reimplementing it, so the variance column of the table is
# libggml-cpu.so's output and not the tool's arithmetic. `nm -D
# build/bin/libggml-cpu.so | grep cvar` shows `T ggml_vec_cvar_f32`.
#
# The model argument is only used to open a GGUF context; --emit-golden
# allocates its own tensors and reads no weights, so any file will do and a
# missing one is not a reason to fail here.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
LLAMA_CPP=${LLAMA_CPP:-$HOME/llama.cpp}
out=$here/../tests/data/norm_golden.bin

if [ ! -e "$LLAMA_CPP/build/bin/libggml-cpu.so" ]; then
    echo "no $LLAMA_CPP/build/bin/libggml-cpu.so -- build llama.cpp, or set LLAMA_CPP" >&2
    exit 1
fi

tmp=${TMPDIR:-/tmp}/probe_norm.$$
cc -O2 -ffp-contract=off -Wall -Wextra -o "$tmp" \
   -I"$LLAMA_CPP/ggml/include" -I"$LLAMA_CPP/include" \
   "$here/../tools/probe_norm.c" \
   -L"$LLAMA_CPP/build/bin" -lggml -lggml-base -lggml-cpu -lm \
   -Wl,-rpath,"$LLAMA_CPP/build/bin"

model=${1:-$(ls "$here/../../models"/*.gguf 2>/dev/null | head -1 || echo /dev/null)}

mkdir -p "$(dirname "$out")"
"$tmp" "$model" --emit-golden "$out"
rm -f "$tmp"

echo "wrote $out ($(wc -c < "$out") bytes)"
echo "llama.cpp: $(git -C "$LLAMA_CPP" rev-parse --short HEAD)"