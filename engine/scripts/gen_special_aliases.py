#!/usr/bin/env python3
"""Generate engine/src/special_aliases.rs from llama.cpp's src/llama-vocab.cpp.

llama.cpp does not trust `tokenizer.ggml.token_type`. After reading it, it runs
several "auto-detect special tokens by text" blocks that OR LLAMA_TOKEN_ATTR_
CONTROL onto any token whose *spelling* is one of a hand-maintained alias list
-- end-of-text, end-of-message, the seven fill-in-the-middle roles, and the
end-of-generation set. There is also one block that does the opposite and
forces four gpt-oss tokens to USER_DEFINED.

Every alias is a display string, and several of them contain characters that do
not survive a copy-paste: a zero-width space (Granite's `<ZWSP>fim_prefix>`), a
SentencePiece marker (U+2581 in `▁<PRE>`), and full-width pipes and markers
(DeepSeek's `<｜end▁of▁sentence｜>`). Hand-transcribing 69 of those is a bug
waiting to happen, so they are extracted from the C++ instead -- the same
reason engine/src/unicode_tables.rs exists.

Usage: scripts/gen_special_aliases.py [--llama-cpp DIR] > ../src/special_aliases.rs
"""

import argparse
import hashlib
import re
import sys

# Each auto-detect block is guarded by `if (special_<slot>_id == LLAMA_TOKEN_NULL)`,
# i.e. it only runs when the GGUF did not declare the corresponding
# `tokenizer.ggml.<key>_token_id`. The name in the guard is the slot.
GUARDED_BLOCKS = [
    ("eot", "special_eot_id"),
    ("eom", "special_eom_id"),
    ("fim_pre", "special_fim_pre_id"),
    ("fim_suf", "special_fim_suf_id"),
    ("fim_mid", "special_fim_mid_id"),
    ("fim_pad", "special_fim_pad_id"),
    ("fim_rep", "special_fim_rep_id"),
    ("fim_sep", "special_fim_sep_id"),
]

# The end-of-generation set is not guarded -- it always runs, because it is a
# *list* rather than a single id. It also feeds `special_eog_ids`, which this
# engine does not model; only the CONTROL promotion is observable here.
EOG_START = "special_eog_ids.clear();"
EOG_END = "} else {"

GPT_OSS_START = "// @ngxson : quick hack for gpt-oss"

ALIAS_RE = re.compile(r't\.first == "((?:[^"\\]|\\.)*)"')


def aliases_in(text):
    """Every `t.first == "..."` spelling in `text`, in source order."""
    return [m.group(1) for m in ALIAS_RE.finditer(text)]


def rust_str(s):
    """A Rust string literal for `s`, escaped but otherwise verbatim.

    Non-ASCII is left alone: it is UTF-8 in the source file either way, and
    keeping it readable is the point -- a reviewer should be able to see that
    the Granite alias really does carry a zero-width space.
    """
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    return '"%s"' % out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", default="/home/reeds/llama.cpp")
    ap.add_argument("--min-aliases", type=int, default=69)
    args = ap.parse_args()

    path = "%s/src/llama-vocab.cpp" % args.llama_cpp
    with open(path, "rb") as f:
        raw = f.read()
    src = raw.decode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()

    # ---- the eight guarded blocks -------------------------------------------
    guarded = []
    for slot, guard in GUARDED_BLOCKS:
        header = "if (%s == LLAMA_TOKEN_NULL) {" % guard
        start = src.find(header)
        if start < 0:
            sys.exit("gen_special_aliases: no %s in %s" % (header, path))
        # The block ends at the `}` that closes it, found by brace counting from
        # the header. Anything after is the next block.
        depth = 0
        end = None
        for i in range(start, len(src)):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            sys.exit("gen_special_aliases: unterminated block for %s" % guard)
        got = aliases_in(src[start:end])
        if not got:
            sys.exit("gen_special_aliases: %s matched no aliases" % guard)
        guarded.append((slot, got))

    # ---- the end-of-generation set ------------------------------------------
    eog_start = src.find(EOG_START)
    if eog_start < 0:
        sys.exit("gen_special_aliases: no %r in %s" % (EOG_START, path))
    eog_end = src.find(EOG_END, eog_start)
    if eog_end < 0:
        sys.exit("gen_special_aliases: no %r after the eog block" % EOG_END)
    eog = aliases_in(src[eog_start:eog_end])

    # ---- the gpt-oss override ----------------------------------------------
    gpt_start = src.find(GPT_OSS_START)
    if gpt_start < 0:
        sys.exit("gen_special_aliases: no %r in %s" % (GPT_OSS_START, path))
    gpt_oss = aliases_in(src[gpt_start:src.find("\n", src.find("}", gpt_start))])

    total = sum(len(a) for _, a in guarded) + len(eog) + len(gpt_oss)
    if total < args.min_aliases:
        sys.exit(
            "gen_special_aliases: found %d aliases, expected at least %d -- "
            "llama.cpp moved the blocks or renamed them" % (total, args.min_aliases)
        )

    out = []
    w = out.append
    w("//! Token-text aliases llama.cpp promotes at load time.")
    w("//!")
    w("//! GENERATED by engine/scripts/gen_special_aliases.py -- do not edit by hand.")
    w("//!")
    w("//! Source: `src/llama-vocab.cpp` of the llama.cpp checkout at")
    w("//! `/home/reeds/llama.cpp`, sha256 `%s`." % digest)
    w("//!")
    w("//! llama.cpp reads `tokenizer.ggml.token_type` and then *overrides* it: any")
    w("//! token spelled like one of these is forced to CONTROL (or, for the gpt-oss")
    w("//! quartet, to USER_DEFINED). Qwen2.5-3B ships `</s>` as `token_type = 1`")
    w("//! NORMAL, and llama.cpp logs `control-looking token: 128247 '</s>' was not")
    w("//! control-type` while loading it.")
    w("//!")
    w("//! The override is invisible with `parse_special = false` -- CONTROL tokens")
    w("//! are skipped by `tokenizer_st_partition` anyway -- which is why it can go")
    w("//! unnoticed for as long as nobody tokenizes a prompt with")
    w("//! `parse_special = true`. See `build_specials` and `SpecialPromotion`.")
    w("")
    w("/// One auto-detect block: the GGUF key that suppresses it, and the spellings")
    w("/// that satisfy it when the key is absent.")
    w("pub struct AliasGroup {")
    w("    /// The `tokenizer.ggml.<slot>_token_id` key whose absence lets this block run.")
    w("    pub slot: &'static str,")
    w("    /// Aliases in source order; the first match wins, as upstream.")
    w("    pub aliases: &'static [&'static str],")
    w("}")
    w("")
    w("/// The eight blocks guarded by `special_*_id == LLAMA_TOKEN_NULL`.")
    w("/// `rustfmt::skip` so that the generator's output is the file: rustfmt")
    w("/// would otherwise reflow the one-alias groups and regeneration would show")
    w("/// up as a diff.")
    w("#[rustfmt::skip]")
    w("pub const CONTROL_ALIAS_GROUPS: &[AliasGroup] = &[")
    for slot, aliases in guarded:
        w("    AliasGroup {")
        w("        slot: %s," % rust_str(slot))
        w("        aliases: &[")
        for a in aliases:
            w("            %s," % rust_str(a))
        w("        ],")
        w("    },")
    w("];")
    w("")
    w("/// The end-of-generation set (llama-vocab.cpp's `special_eog_ids` block).")
    w("/// Unguarded: it always runs. This engine models only its CONTROL side;")
    w("/// the id set itself drives generation stopping, which it does not do.")
    w("pub const EOG_ALIASES: &[&str] = &[")
    for a in eog:
        w("    %s," % rust_str(a))
    w("];")
    w("")
    w("/// Forced to USER_DEFINED rather than CONTROL, which is the opposite effect")
    w("/// and is observable even with `parse_special = false`.")
    w("#[rustfmt::skip]")
    w("pub const USER_DEFINED_ALIASES: &[&str] = &[")
    for a in gpt_oss:
        w("    %s," % rust_str(a))
    w("];")
    w("")
    w("/// sha256 of the `src/llama-vocab.cpp` this was generated from. A different")
    w("/// value means upstream moved the blocks and this file is stale.")
    w("pub const SOURCE_SHA256: &str = %s;" % rust_str(digest))
    w("")
    w("/// Total aliases, as a cheap drift guard that needs no checkout.")
    w("pub const N_ALIASES: usize = %d;" % total)
    w("")

    sys.stdout.write("\n".join(out))
    sys.stderr.write(
        "gen_special_aliases: %d aliases (%d guarded blocks, %d eog, %d gpt-oss)\n"
        % (total, len(guarded), len(eog), len(gpt_oss))
    )


if __name__ == "__main__":
    main()