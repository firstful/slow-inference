#!/bin/sh
# Regenerate tests/data/dot_golden.bin from llama.cpp's own kernel.
#
#   sh engine/scripts/gen_dot_golden.sh
#
# Needs a built llama.cpp ($LLAMA_CPP/build/bin/libggml*.so), the same one
# scripts/gen_golden_vectors.sh and scripts/explain_quant_gap.py use.
#
# -ffp-contract=off is load-bearing and not a style choice. The golden records
# the *kernel's* bits, and the tool's own `seq_f32` column is a stand-in for the
# engine's old `sum::<f32>()` line, which does not contract. If the tool were
# built with contraction on, that column would quietly become an fma chain and
# would stop being evidence about the engine. The fma the kernel genuinely uses
# is written as an explicit fmaf() in the tool and survives either way.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
LLAMA_CPP=${LLAMA_CPP:-$HOME/llama.cpp}
out=$here/../tests/data/dot_golden.bin

if [ ! -e "$LLAMA_CPP/build/bin/libggml-cpu.so" ]; then
    echo "no $LLAMA_CPP/build/bin/libggml-cpu.so -- build llama.cpp, or set LLAMA_CPP" >&2
    exit 1
fi

tmp=${TMPDIR:-/tmp}/probe_f32_dot.$$
cc -O2 -ffp-contract=off -Wall -Wextra -o "$tmp" \
   -I"$LLAMA_CPP/ggml/include" -I"$LLAMA_CPP/include" \
   "$here/../tools/probe_f32_dot.c" \
   -L"$LLAMA_CPP/build/bin" -lggml -lggml-base -lggml-cpu -lm \
   -Wl,-rpath,"$LLAMA_CPP/build/bin"

# The model argument is only used to open a GGUF context; --emit-golden
# allocates its own tensors and reads no weights, so any file will do and a
# missing one is not a reason to fail here.
model=${1:-$(ls "$here/../../models"/*.gguf 2>/dev/null | head -1 || echo /dev/null)}

mkdir -p "$(dirname "$out")"
"$tmp" "$model" --emit-golden "$out"
rm -f "$tmp"

echo "wrote $out ($(wc -c < "$out") bytes)"
echo "llama.cpp: $(git -C "$LLAMA_CPP" rev-parse --short HEAD)"
