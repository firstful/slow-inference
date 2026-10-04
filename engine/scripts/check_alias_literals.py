#!/usr/bin/env python3
"""Check hand-written alias literals against the generated alias table.

`src/special_aliases.rs` is generated from llama.cpp by
`scripts/gen_special_aliases.py`, and its entries carry characters that no
terminal, editor or agent harness round-trips reliably: U+FF5C (a full-width
pipe), U+2581 (a lower one-eighth block) and U+200B (zero-width space) all
occur, and the gemma4 tool-response marker is spelled *without* its closing
pipe upstream. That last one is the trap: a human or a model copying it into a
test "corrects" it into a 17-character string that matches nothing, the
promotion silently stops happening, and the test that was written to prove the
promotion works instead proves that the code agrees with itself.

So: anything in the Rust sources that *looks* like a special-token alias must be
byte-identical to an entry in the generated table. The table is the only place
the spellings live. Literals in the corpus and `.ref` data files are inputs and
are deliberately not checked -- llama.cpp tokenizes whatever they say.

Reports lengths and hex, never the literal: a rendered `<|...|>` is not
evidence of what is in the file, which is the whole reason this script reads
bytes.

Usage: scripts/check_alias_literals.py [paths ...]
  With no paths, checks src/*.rs and tests/*.rs.
Exit status 0 if every alias-shaped literal is in the table.
"""

import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLE = os.path.join(ROOT, "src", "special_aliases.rs")

# The character that makes a literal "alias-shaped", plus the three that make it
# dangerous: a full-width pipe (gemma4 / deepseek-ocr), a lower one-eighth block
# (SentencePiece space marker in the deepseek-ocr alias) and a zero-width space
# (the granite spellings). A literal containing any of these is a transcription
# risk whatever else it looks like.
LT, PIPE, GT = b"<", b"|", b">"
SPECIALS = {
    0xFF5C: "full-width pipe",
    0x2581: "lower one-eighth block",
    0x200B: "zero-width space",
}

# Literals that look like aliases but are not: test vocabularies of our own.
# Built from character codes so this file cannot itself be mangled into a
# different spelling than it claims to allow.
ALLOW = {
    LT + PIPE + b"ud" + PIPE + GT: "spm.rs's synthetic vocabulary token, not an alias",
}

STRING = re.compile(rb'"([^"\n]*)"')
TOKEN_RE = re.compile(rb"[A-Za-z0-9_]+")


def table_entries():
    with open(TABLE, "rb") as f:
        data = f.read()
    return {m.group(1) for m in STRING.finditer(data)}


def looks_like_alias(lit: bytes) -> bool:
    if lit.startswith(LT) and PIPE in lit:
        return True
    return any(b in SPECIALS for b in lit)


def describe(lit: bytes) -> str:
    return f"len {len(lit)}  " + " ".join(f"{b:02x}" for b in lit)


def main(argv):
    paths = argv or (
        sorted(glob.glob(os.path.join(ROOT, "src", "*.rs")))
        + sorted(glob.glob(os.path.join(ROOT, "tests", "*.rs")))
    )
    known = table_entries()
    # Fail loudly rather than silently allowing everything if the table moved.
    if not known:
        print(f"no aliases parsed out of {TABLE}", file=sys.stderr)
        return 2

    bad = 0
    seen = 0
    for path in paths:
        with open(path, "rb") as f:
            data = f.read()
        for lineno, line in enumerate(data.split(b"\n"), 1):
            for m in STRING.finditer(line):
                lit = m.group(1)
                if not looks_like_alias(lit):
                    continue
                seen += 1
                if lit in known:
                    continue
                if lit in ALLOW:
                    continue
                bad += 1
                rel = os.path.relpath(path, ROOT)
                print(f"{rel}:{lineno}: not in {os.path.relpath(TABLE, ROOT)}")
                print(f"    {describe(lit)}")
                print(f"    source: {TOKEN_RE.findall(lit)}")
    print(f"check_alias_literals: {seen} alias-shaped literal(s), {bad} not in the table")
    if bad:
        print(
            "  copy the spelling from src/special_aliases.rs by index, or add it to\n"
            "  ALLOW in scripts/check_alias_literals.py with a reason. Do not retype it:\n"
            "  upstream's gemma4 marker has no closing pipe, and 'fixing' it breaks\n"
            "  the match silently."
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
