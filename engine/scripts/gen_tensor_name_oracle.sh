#!/usr/bin/env bash
# Rebuild tests/data/tensor_name_oracle.json from a llama.cpp checkout.
#
# Unlike gen_golden_vectors.sh this needs no build: the oracle is extracted from
# llama.cpp's own sources (llama-arch.cpp's name table plus each model's
# load_arch_tensors), so a plain source tree is enough.
set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$LLAMA_CPP/src/llama-arch.cpp" ]; then
    echo "no src/llama-arch.cpp under $LLAMA_CPP — set LLAMA_CPP" >&2
    exit 1
fi

exec python3 "$HERE/scripts/gen_tensor_name_oracle.py" --llama-cpp "$LLAMA_CPP" "$@"