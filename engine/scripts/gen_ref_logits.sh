#!/usr/bin/env bash
# Rebuild tests/data/fixtures/{llama,qwen2,gpt2}.ref from llama.cpp's own
# forward pass.
#
# Each .ref is the exact f32 logits llama.cpp produces for the prompt baked
# into tests/ref_parity.rs (tokens 3,17,42,99,7,61,127,24). The Rust test
# replays the same tokens through our forward pass and gates on cosine /
# argmax / top-k agreement.
#
# Requires a built llama.cpp with libllama.so at $LLAMA_CPP (default
# ~/llama.cpp). This is the only step that needs llama.cpp — once the .ref
# files exist, `cargo test` runs the whole parity suite with no C toolchain
# and no llama.cpp checkout.
set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

PROMPT="3,17,42,99,7,61,127,24"

# `case:gguf_arch`. The second field differs from the first for three entries: the
# MoE fixture is generated as `--arch llama-moe` but its `general.architecture` is
# `llama`, the fused expert fixture is generated as `--arch deepseek2-fused` but
# its arch is `deepseek2`, and `llama-moe-wscale` is generated as its own case but
# carries `llama` too. The first two are deliberately the *same* architecture as a
# checked-in case, because a routing or tensor-container difference is not a new
# architecture. `llama-moe-wscale` is the same architecture a third time: it is
# `llama-moe` plus a `{arch}.expert_weights_scale` the `llama` arch must ignore,
# so both of its two `.ref` files come out byte-identical to `llama-moe`'s.
# That byte-identity is the oracle's own statement of the finding, and
# `tests/ref_parity.rs` reproduces it on every run.
# `--arch` below selects the float64 forward pass, which is keyed on the *file's*
# arch, so it takes the second field.
CASES=(
    "llama:llama"
    "qwen2:qwen2"
    "gpt2:gpt2"
    "deepseek2:deepseek2"
    "deepseek2-fused:deepseek2"
    "deepseek2-normw:deepseek2"
    "gemma3:gemma3"
    "llama-moe:llama"
    "llama-moe-wscale:llama"
)

if [ ! -f "$LLAMA_CPP/build/bin/libllama.so" ]; then
    echo "no libllama.so under $LLAMA_CPP/build/bin — build llama.cpp first" >&2
    exit 1
fi

BIN="$(mktemp -d)/gen_ref_logits"
trap 'rm -rf "$(dirname "$BIN")"' EXIT
gcc -O2 -std=c11 -o "$BIN" \
    -I"$LLAMA_CPP/include" \
    -I"$LLAMA_CPP/ggml/include" \
    "$HERE/tools/gen_ref_logits.c" \
    -L"$LLAMA_CPP/build/bin" -lllama -lggml -lggml-base -lm \
    -Wl,-rpath,"$LLAMA_CPP/build/bin"

for entry in "${CASES[@]}"; do
    case_name="${entry%%:*}"
    gguf_arch="${entry##*:}"
    gguf="$HERE/tests/data/fixtures/synthetic-$case_name.gguf"
    if [ ! -f "$gguf" ]; then
        echo "missing fixture $gguf — run scripts/generate_synthetic_fixtures.py --arch $case_name" >&2
        exit 1
    fi
    "$BIN" "$gguf" "$HERE/tests/data/fixtures/$case_name.ref" "$PROMPT" 2>/dev/null
    echo "wrote $HERE/tests/data/fixtures/$case_name.ref"
done

# The second, tight oracle: an independent float64 forward pass. `.ref` alone
# cannot catch a subtly wrong activation constant (see the module docstring in
# tests/ref_parity.rs), so every architecture needs both files.
#
# The float64 rows do not depend on `.ref` — only on the fixture and the token
# list, which `load_ref` cross-checks. `.ref` is passed as both --engine and
# --llama purely so that list is guaranteed identical and the emitted file can
# never silently drift onto a different prompt. Do NOT pass --f16-gelu here:
# the point of this file is to model exact math, not ggml's GELU table.
REPO_SCRIPTS="$REPO/scripts"
echo
echo "regenerating float64 references (needs numpy + gguf-py)"
for entry in "${CASES[@]}"; do
    case_name="${entry%%:*}"
    gguf_arch="${entry##*:}"
    PYTHONPATH="${LLAMA_CPP}/gguf-py:${PYTHONPATH:-}" python3 \
        "$REPO_SCRIPTS/arbitrate_logits.py" \
        --arch "$gguf_arch" \
        --gguf "$HERE/tests/data/fixtures/synthetic-$case_name.gguf" \
        --engine "$HERE/tests/data/fixtures/$case_name.ref" \
        --llama "$HERE/tests/data/fixtures/$case_name.ref" \
        --emit-ref "$HERE/tests/data/fixtures/$case_name.f64.ref" >/dev/null
    echo "wrote $HERE/tests/data/fixtures/$case_name.f64.ref"
done
