#!/usr/bin/env python3
"""Emit src/unicode_tables.rs from llama.cpp's src/unicode-data.cpp.

llama.cpp's pre-tokenizer regexes ('s|'t|'re|...| ?\\p{L}+| ?\\p{N}+|...) are
Unicode *property* patterns, so matching them exactly needs llama.cpp's own
codepoint classification table (`unicode_ranges_flags`, a run-length encoded
list of (start_cpt, flags) covering the whole 0..=0x10FFFF range). Deriving
the three properties the BPE regexes actually use from Rust's `char` methods
is close but not equal -- `char::is_alphabetic()` is Alphabetic (= L + Nl +
Other_Alphabetic), not L -- so the classification table is generated verbatim
instead of approximated.

Same reasoning as gen_quant_tables.py: the table is `const` in C and cannot be
linked against, so we parse the source. Regenerate with:

    python3 engine/scripts/gen_unicode_tables.py

Set $LLAMA_CPP to point at a llama.cpp checkout other than ~/llama.cpp.
"""
import os
import re
import sys

LLAMA_CPP = os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp"))
SOURCE = os.path.join(LLAMA_CPP, "src", "unicode-data.cpp")
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "unicode_tables.rs",
)


def parse_pairs(text, decl):
    """Extract the initializer_list<pair<uint32_t, uintN_t>> body of `decl`."""
    start = text.index(decl)
    open_brace = text.index("{", start)
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : i]
    raise SystemExit("unterminated initializer list for %s" % decl)


def pairs_of(body):
    out = []
    for m in re.finditer(r"\{\s*(0[xX][0-9A-Fa-f]+|\d+)\s*,\s*(0[xX][0-9A-Fa-f]+|\d+)\s*\}", body):
        out.append((int(m.group(1), 0), int(m.group(2), 0)))
    return out


def emit_pairs(lines, name, rtype, pairs, doc):
    # rustfmt would explode one tuple per line; the whole point of emitting
    # run-length tables is that they stay greppable as rows.
    lines.append("#[rustfmt::skip]")
    lines.append(doc)
    lines.append("pub static %s: [(u32, %s); %d] = [" % (name, rtype, len(pairs)))
    cur = "   "
    for a, b in pairs:
        piece = " (0x%X, 0x%X)," % (a, b)
        if len(cur) + len(piece) > 96:
            lines.append(cur.rstrip())
            cur = "   "
        cur += piece
    lines.append(cur.rstrip().rstrip(","))
    lines.append("];")
    lines.append("")


def parse_scalars(text, decl):
    """Extract the hex/decimal scalar list of an `= { ... };` initializer."""
    start = text.index(decl)
    open_brace = text.index("{", start)
    end = text.index("}", open_brace)
    body = text[open_brace + 1 : end]
    return [int(tok, 0) for tok in re.findall(r"0[xX][0-9A-Fa-f]+|\b\d+\b", body)]


def emit_scalars(lines, name, values, doc):
    lines.append("#[rustfmt::skip]")
    lines.append(doc)
    lines.append("pub static %s: [u32; %d] = [" % (name, len(values)))
    cur = "   "
    for v in values:
        piece = " 0x%X," % v
        if len(cur) + len(piece) > 96:
            lines.append(cur.rstrip())
            cur = "   "
        cur += piece
    lines.append(cur.rstrip().rstrip(","))
    lines.append("];")
    lines.append("")


def main():
    if not os.path.exists(SOURCE):
        sys.exit("no unicode-data.cpp under %s — set $LLAMA_CPP" % LLAMA_CPP)
    with open(SOURCE, encoding="utf-8") as f:
        text = f.read()

    flags = pairs_of(parse_pairs(text, "unicode_ranges_flags"))
    lower = pairs_of(parse_pairs(text, "unicode_map_lowercase"))
    # `unicode_ranges_flags` carries only the general categories; the WHITESPACE
    # helper flag is OR-ed in afterwards from a separate set. Both halves are
    # needed, and the pre-tokenizer state machines branch on both.
    ws = parse_scalars(text, "unicode_set_whitespace")

    # The flags table is run-length encoded: entry i covers
    # [start_i, start_{i+1}) and must be sorted and non-overlapping. The table
    # starts at codepoint 0 and covers the whole range, so no codepoint is left
    # with a zero flag word; codepoints past the last entry are UNDEFINED.
    if not flags:
        sys.exit("no unicode_ranges_flags entries parsed from %s" % SOURCE)
    if ws != sorted(set(ws)):
        sys.exit("unicode_set_whitespace is not strictly ascending")
    # unicode.cpp asserts the range table both starts at 0 and ends at
    # MAX_CODEPOINTS; the trailing sentinel is what makes the last run bounded.
    if flags[0][0] != 0 or flags[-1][0] != 0x110000:
        sys.exit(
            "unicode_ranges_flags must span 0..MAX_CODEPOINTS, got 0x%X..0x%X"
            % (flags[0][0], flags[-1][0])
        )
    for (a0, _), (a1, _) in zip(flags, flags[1:]):
        if a1 <= a0:
            sys.exit("unicode_ranges_flags is not strictly increasing at 0x%X" % a0)

    lines = [
        "//! Codepoint classification for the pre-tokenizer, generated from llama.cpp.",
        "//!",
        "//! GENERATED by engine/scripts/gen_unicode_tables.py from",
        "//! %s — do not edit by hand." % SOURCE,
        "//!",
        "//! `unicode_ranges_flags` is run-length encoded: entry `i` holds the flag",
        "//! word for every codepoint in `[start_i, start_{i+1})`, starting at",
        "//! codepoint 0 and covering the whole range. Codepoints past the last",
        "//! entry are reported as UNDEFINED, matching the `undef` arm of",
        "//! `unicode_cpt_flags_from_cpt`.",
        "",
        "#![allow(clippy::unreadable_literal)]",
        "",
        "/// Max number of codepoints (from unicode-data.h's MAX_CODEPOINTS).",
        "pub const MAX_CODEPOINTS: u32 = 0x110000;",
        "",
    ]

    emit_pairs(
        lines,
        "UNICODE_RANGES_FLAGS",
        "u16",
        flags,
        "/// (first codepoint of the run, flag word) — see unicode_cpt_flags.",
    )
    emit_pairs(
        lines,
        "UNICODE_MAP_LOWERCASE",
        "u32",
        lower,
        "/// (uppercase codepoint, lowercase codepoint), sorted, for unicode_tolower.",
    )
    emit_scalars(
        lines,
        "UNICODE_SET_WHITESPACE",
        ws,
        "/// Codepoints carrying the WHITESPACE helper flag, ascending.",
    )

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(
        "wrote %s (%d flag runs, %d lowercase pairs, %d whitespace codepoints)"
        % (OUT, len(flags), len(lower), len(ws))
    )


if __name__ == "__main__":
    main()
