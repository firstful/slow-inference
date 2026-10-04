#!/usr/bin/env bash
# Rebuild tests/data/quant_golden.bin from the authoritative llama.cpp ggml
# dequantization kernels. Requires a built llama.cpp at $LLAMA_CPP (default
# ~/llama.cpp) with libggml-base.so present.
set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$LLAMA_CPP/build/bin/libggml-base.so" ]; then
    echo "no libggml-base.so under $LLAMA_CPP/build/bin — build llama.cpp ggml first" >&2
    exit 1
fi

BIN="$(mktemp -d)/gen_golden_vectors"
gcc -O2 -std=c11 -o "$BIN" \
    -I"$LLAMA_CPP/ggml/include" \
    "$HERE/tools/gen_golden_vectors.c" \
    -L"$LLAMA_CPP/build/bin" -lggml-base -lm \
    -Wl,-rpath,"$LLAMA_CPP/build/bin"

"$BIN" "$HERE/tests/data/quant_golden.bin"
ls -l "$HERE/tests/data/quant_golden.bin"
