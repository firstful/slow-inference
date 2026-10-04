#!/usr/bin/env bash
# Rebuild engine/tests/ref_attrs/*.attrs — llama.cpp's own token attributes for
# the LSTRIP/RSTRIP rules, plus the record of the one rule that cannot run.
#
# This is the tokenizer counterpart of engine/scripts/gen_ref_tokens.sh and
# engine/scripts/gen_ref_logits.sh: it is the only step that needs a built
# llama.cpp, and it exists because those two oracles are both blind here.
#
# Why a third oracle
# ------------------
# `tokenizer_st_partition`'s strip flags change what a *neighbouring* raw
# fragment encodes to, so the natural oracle is ids from `llama_tokenize` — and
# that is what does not work. Reaching LSTRIP needs a vocabulary carrying
# `<mask>` (jina-v2-* / nomic-bert-moe / jina-bert-v3) or `[MASK]`
# (modern-bert), and no GGUF on disk has either. A synthetic vocabulary that
# does is only half the answer: with a trivial merge table the surrounding text
# encodes to nothing, so stripping whitespace out of it changes no ids and the
# oracle reports *equality* for a partition that is provably wrong. Measured:
# `a <mask> b` and `a<mask>b` produce byte-identical id lists on
# models/synthetic-gpt2-strip.gguf with and without the jina-v2-code label.
#
# The flags are attributes, and `llama_vocab_get_attr` is public API
# (llama.h:1085), so the honest oracle is the attribute itself. One line per id,
# the full mask, no corpus and no working merge table needed — and a difference
# names the rule that is wrong instead of the ids that moved.
#
# What each label proves
# ----------------------
#   baseline     no label: the flags are absent, so a rule that fires
#                unconditionally shows up here.
#   jina         tokenizer.ggml.pre = jina-v2-code  -> LSTRIP on <mask>
#   modernbert   general.name = modern-bert-...   -> LSTRIP on [MASK]
#   phi3         general.name = phi-3-...         -> does not load, see below
#
# The `phi3` case is the one worth reading. Upstream's phi-3 branch ends with
# `_set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, false)` over
# `{"<unk>", "<s>", "<empty>"}`, and `_set_token_attr` is
# `token_to_id.at(token)` — a throw, not a miss. `<empty>` is not a key in any
# vocabulary: a GGUF's empty token is rewritten to `[EMPTY_<i>]` when the token
# list is read (llama-vocab.cpp:2438), and no real tokenizer has a token with
# those eight literal characters. So the `.at` throws `std::out_of_range`,
# `load_vocab` reports "error loading model vocabulary: unordered_map::at", and
# a Phi-3 GGUF does not load at all. The three labelled variants below all carry
# `</s>`, `<unk>` and `<s>`, so the failure is reached *past* those and lands on
# `<empty>` — which is the unconditional part.
#
# Usage:
#   scripts/gen_strip_refs.sh
set -euo pipefail

LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$REPO/engine"
FIXTURE="$REPO/models/synthetic-gpt2-strip.gguf"
OUT="$HERE/tests/ref_attrs"
PATCH="$REPO/scripts/gguf_patch_kv.py"

if [ ! -f "$LLAMA_CPP/build/bin/libllama.so" ]; then
    echo "no libllama.so under $LLAMA_CPP/build/bin — build llama.cpp first" >&2
    exit 1
fi

if [ ! -f "$FIXTURE" ]; then
    PYTHONPATH="${LLAMA_CPP}/gguf-py:${PYTHONPATH:-}" python3 \
        "$REPO/scripts/generate_synthetic_fixtures.py" --arch gpt2-strip --out "$FIXTURE"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
BIN="$WORK/gen_ref_attrs"
gcc -O2 -std=c11 -o "$BIN" \
    -I"$LLAMA_CPP/include" \
    -I"$LLAMA_CPP/ggml/include" \
    "$HERE/tools/gen_ref_attrs.c" \
    -L"$LLAMA_CPP/build/bin" -lllama -lggml -lggml-base -lm \
    -Wl,-rpath,"$LLAMA_CPP/build/bin"

mkdir -p "$OUT"

# label <name> [key=value ...] -> path to the labelled copy on stdout
label() {
    local name="$1"; shift
    local gguf="$WORK/$name.gguf"
    if [ $# -eq 0 ]; then
        cp "$FIXTURE" "$gguf"
    else
        local args=()
        for kv in "$@"; do args+=(--set "$kv"); done
        python3 "$PATCH" "$FIXTURE" "$gguf" "${args[@]}"
    fi
    echo "$gguf"
}

# The strip rules are keyed on metadata, so the only thing that varies between
# these four files is one or two strings. Everything else — every tensor byte,
# every token text, every token_type — is `synthetic-gpt2-strip.gguf` verbatim.
run_one() {
    local name="$1"; shift
    local gguf rest kvs
    gguf="$(label "$name" "$@")"
    if "$BIN" "$gguf" "$OUT/$name.attrs" 2>"$WORK/$name.log"; then
        # Say which ids the label moved, so a reader does not have to diff four
        # files by hand to see which rule fired.
        python3 - "$OUT/$name.attrs" "$OUT/baseline.attrs" <<'PY' || true
import sys
def load(p):
    return {int(l.split()[1]): int(l.split()[2]) for l in open(p) if l.startswith("ATTR ")}
mine, base = load(sys.argv[1]), load(sys.argv[2])
moved = {i: (base[i], mine[i]) for i in mine if base.get(i) != mine[i]}
print("  ids moved by the label:", moved if moved else "none")
PY
        echo "wrote $OUT/$name.attrs"
    else
        echo "!! $name did not load — llama.cpp says:" >&2
        grep -E "error loading model vocabulary|error loading model:" "$WORK/$name.log" >&2 || true
        {
            echo "# $name.gguf does not load. Recorded because a rule that cannot"
            echo "# execute is a different thing from a rule that has not been"
            echo "# implemented, and only this file says which one it is."
            echo "LOAD_FAILED"
            grep -E "error loading model vocabulary|error loading model:" "$WORK/$name.log" || true
        } > "$OUT/$name.attrs"
        echo "wrote $OUT/$name.attrs (load failure)"
    fi
}

run_one baseline
run_one jina tokenizer.ggml.pre=jina-v2-code
run_one modernbert general.name=modern-bert-synthetic
run_one phi3 general.name=phi-3-synthetic