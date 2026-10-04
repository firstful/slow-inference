#!/usr/bin/env bash
# Build the *quantized* parity fixture pair and its two llama.cpp references.
#
# `tests/ref_parity.rs` runs on F32 fixtures on purpose: quantization noise is
# four orders of magnitude larger than the architecture semantics those tests
# gate on, so `tests/quant_gap.rs` exists to gate on the opposite statement --
# that quantization changes *nothing* observable, because the engine dequantizes
# a Q4_K row to exactly the f32 values ggml does (tests/quant_golden.rs, bit for
# bit) and then runs identical f32 arithmetic.
#
# Two files, same numbers, different storage:
#
#   synthetic-llama-q4_0.gguf      the fixture quantized to Q4_0 (68 KB)
#   synthetic-llama-q4_0-f32.gguf  the same fixture widened back to F32 (260 KB)
#
# Their architecture bytes are identical and so are their logits *in our
# engine*, bit for bit. llama.cpp moves by cos ~0.9999 between them, which is
# the whole finding: it quantizes activations to Q8_K inside every 4-bit
# matmul and accumulates in int16, we dequantize the weight and stay in f32.
#
# Requires: numpy + gguf-py (for scripts/requantize.py) and, for the .ref
# files, a built llama.cpp. Once the four files below are checked in, `cargo
# test` runs the comparison with neither.
set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
FIX="$HERE/tests/data/fixtures"
PROMPT="3,17,42,99,7,61,127,24"   # must match PROMPT in tests/quant_gap.rs

BASE="$FIX/synthetic-llama.gguf"
Q="$FIX/synthetic-llama-q4_0.gguf"
QF32="$FIX/synthetic-llama-q4_0-f32.gguf"

if [ ! -f "$BASE" ]; then
    echo "missing $BASE — run:" >&2
    echo "  python3 scripts/generate_synthetic_fixtures.py --arch llama --out $BASE" >&2
    exit 1
fi

echo "quantizing $BASE -> $Q"
PYTHONPATH="${LLAMA_CPP}/gguf-py:${PYTHONPATH:-}" python3 \
    "$REPO/scripts/requantize.py" --to q4_0 --in "$BASE" --out "$Q" --quiet

echo "widening $Q -> $QF32"
PYTHONPATH="${LLAMA_CPP}/gguf-py:${PYTHONPATH:-}" python3 \
    "$REPO/scripts/requantize.py" --to f32 --in "$Q" --out "$QF32" --quiet

if [ ! -f "$LLAMA_CPP/build/bin/libllama.so" ]; then
    echo "no libllama.so under $LLAMA_CPP/build/bin — stopping after the fixtures." >&2
    echo "the checked-in .ref files are still used by cargo test." >&2
    exit 0
fi

BIN="$(mktemp -d)/gen_ref_logits"
trap 'rm -rf "$(dirname "$BIN")"' EXIT
gcc -O2 -std=c11 -o "$BIN" \
    -I"$LLAMA_CPP/include" \
    -I"$LLAMA_CPP/ggml/include" \
    "$HERE/tools/gen_ref_logits.c" \
    -L"$LLAMA_CPP/build/bin" -lllama -lggml -lggml-base -lm \
    -Wl,-rpath,"$LLAMA_CPP/build/bin"

for pair in "$Q:q4_0" "$QF32:q4_0_f32"; do
    gguf="${pair%%:*}"
    stem="${pair##*:}"
    "$BIN" "$gguf" "$FIX/llama-$stem.ref" "$PROMPT" 2>/dev/null
    echo "wrote $FIX/llama-$stem.ref"
done

echo
echo "regenerate and gate with:  cargo test --release --test quant_gap"
