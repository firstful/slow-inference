#!/usr/bin/env python3
"""Generate the tensor-name oracle from a llama.cpp checkout.

Architecture-side counterpart of `tools/gen_golden_vectors.c`: instead of comparing
dequantization kernels bit-for-bit, it compares *tensor names*.

For every architecture llama.cpp can load, ask three questions straight out of its
source:

  1. what GGUF arch string does `LLM_ARCH_NAMES` give the arch enum?  (llama-arch.cpp)
  2. which model class does `llama_model_mapping()` dispatch it to?    (llama-model.cpp)
  3. which `create_tensor(tn(LLM_TENSOR_*, ...))` calls does that class make?
                                                                    (models/*.cpp)

then evaluate `LLM_TN_IMPL::str()` -- `format(pattern, bid, xid) + "." + suffix` --
to get the concrete GGUF tensor names llama.cpp looks up on load.

Output is a checked-in JSON fixture that `tests/tensor_names.rs` replays against
`model::tensor_name()`. A missing tensor becomes a test failure instead of a runtime
"missing tensor 'blk.3.attn_norm_2.weight'" abort.

Usage:
    scripts/gen_tensor_name_oracle.py [--llama-cpp DIR] [--out FILE] [--arch NAME ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

# Probe indices used when expanding a name pattern. llama.cpp's own helper
# substitutes -1 when the caller passes no index; a real GGUF always has a
# block 0, and per-expert tensors always have an expert 0.
PROBE_BID = 0
PROBE_XID = 0

LAMBDA = "\x00"
PLACEHOLDER = "\x01%d\x02"


def strip_comments(src: str) -> str:
    """Replace // and /* */ comments with blanks, preserving offsets and literals."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '"' or c == "'":
            q = c
            i += 1
            while i < n and src[i] != q:
                i += 2 if src[i] == "\\" else 1
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def match_bracket(src: str, open_idx: int, o: str, c: str) -> int:
    """Index of the closer matching the opener at *open_idx*."""
    depth = 0
    i, n = open_idx, len(src)
    while i < n:
        ch = src[i]
        if ch in '"\'':
            i += 1
            while i < n and src[i] != ch:
                i += 2 if src[i] == "\\" else 1
        elif ch == o:
            depth += 1
        elif ch == c:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced {o}{c}")


def split_args(text: str) -> list[str]:
    """Split an argument list on top-level commas."""
    args, cur, depth = [], [], 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in '"\'':
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            cur.append(text[i:j + 1])
            i = j + 1
            continue
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return args


def call_args(src: str, after: int) -> tuple[str, int] | None:
    """Given *after* is the index just past an identifier, return its argument
    list and the index after the closing paren -- or None if it is not a call."""
    i = after
    while i < len(src) and src[i].isspace():
        i += 1
    if i >= len(src) or src[i] != "(":
        return None
    close = match_bracket(src, i, "(", ")")
    return src[i + 1:close], close + 1


# ---------------------------------------------------------------------------
# llama-arch.cpp / llama-model.cpp tables
# ---------------------------------------------------------------------------

def grab_table(src: str, header: str) -> str:
    at = src.index(header)
    brace = src.index("{", at)
    return src[brace:match_bracket(src, brace, "{", "}") + 1]


def parse_llama_arch(path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (arch_enum -> gguf_arch_string, tensor_enum -> name_pattern)."""
    src = strip_comments(open(path, encoding="utf-8").read())
    pair = r"\{\s*(%s)\s*,\s*\"([^\"]*)\"\s*\}"
    arch_names = {
        e: s for e, s in re.findall(pair % r"LLM_ARCH_\w+", grab_table(src, "LLM_ARCH_NAMES"))
    }
    tensor_names = {
        e: s for e, s in re.findall(pair % r"LLM_TENSOR_\w+", grab_table(src, "LLM_TENSOR_NAMES"))
    }
    return arch_names, tensor_names


def parse_arch_mapping(path: str) -> dict[str, str]:
    """Return llm_arch enum -> C++ model class, from llama_model_mapping()."""
    src = strip_comments(open(path, encoding="utf-8").read())
    at = src.index("llama_model_mapping")
    fn = src[at:match_bracket(src, src.index("{", at), "{", "}") + 1]
    return dict(
        re.findall(r"case\s+(LLM_ARCH_\w+):\s*return new (\w+)\s*\(", fn)
    )


def expand_name(pattern: str, suffix: str | None, bid: int, xid: int) -> str:
    """Reimplementation of LLM_TN_IMPL::str()."""
    args = [bid, xid]
    it = iter(args)

    def sub(_m: re.Match) -> str:
        return str(next(it, -1))

    name = re.sub(r"%d", sub, pattern)
    return f"{name}.{suffix}" if suffix else name


# ---------------------------------------------------------------------------
# models/*.cpp — walk load_arch_tensors
# ---------------------------------------------------------------------------

# llama_model_base helpers that expand into several create_tensor() calls.
# Flags mirror llama-model.cpp verbatim.
HELPER_EXPANSION: dict[str, list[tuple[str, str, str]]] = {
    # helper: [(tensor_enum, suffix, flags)]
    "create_tensor_qkv": [
        ("LLM_TENSOR_ATTN_QKV", "weight", "TENSOR_NOT_REQUIRED"),
        ("LLM_TENSOR_ATTN_QKV", "bias", "TENSOR_NOT_REQUIRED"),
        ("LLM_TENSOR_ATTN_Q", "weight", "<flags>"),
        ("LLM_TENSOR_ATTN_K", "weight", "<flags>"),
        ("LLM_TENSOR_ATTN_V", "weight", "<flags>"),
        ("LLM_TENSOR_ATTN_Q", "bias", "TENSOR_NOT_REQUIRED"),
        ("LLM_TENSOR_ATTN_K", "bias", "TENSOR_NOT_REQUIRED"),
        ("LLM_TENSOR_ATTN_V", "bias", "TENSOR_NOT_REQUIRED"),
    ],
    "create_tensor_gate_up_exps": [
        ("LLM_TENSOR_FFN_GATE_UP_EXPS", "weight", "TENSOR_NOT_REQUIRED"),
        ("LLM_TENSOR_FFN_GATE_EXPS", "weight", "<flags>"),
        ("LLM_TENSOR_FFN_UP_EXPS", "weight", "<flags>"),
    ],
}

BLOCK_KEYWORDS = ("if", "else", "for", "while", "switch")


def required_from_flags(flags: str) -> bool | None:
    """None when the flags expression is not a literal (we cannot tell)."""
    f = flags.strip()
    if "TENSOR_NOT_REQUIRED" in f:
        return False
    if re.fullmatch(r"(0|TENSOR_SKIP_IF_VIRTUAL)", f):
        return True
    if "<flags>" in f:
        return None
    return None


def scan_body(body: str, tensor_patterns: dict[str, str]) -> list[dict]:
    """Extract create_tensor() calls from a load_arch_tensors body.

    Records the enclosing condition (if any) so the caller can tell an
    unconditional tensor from a config-dependent one. Loop headers are not
    conditions: a tensor created inside `for (i = 0; ... i < n_layer; ++i)` is
    present in every model of the architecture.
    """
    out: list[dict] = []
    stack: list[str | None] = []   # enclosing conditions, innermost last
    pending: str | None = None
    aliases: dict[str, str] = {}    # local name -> "tn(...)" source text
    i, n = 0, len(body)

    def add(enum: str, suffix: str | None, bid, xid, dims: str, flags: str,
            cond: str | None, via: str) -> None:
        pat = tensor_patterns.get(enum)
        if pat is None:
            out.append({"name": None, "tensor": enum, "error": "no LLM_TENSOR_NAMES entry",
                        "source": via})
            return
        conds = [c for c in stack if c is not None and not c.startswith("for(")]
        out.append({
            "name": expand_name(pat, suffix, bid, xid),
            "tensor": enum,
            "suffix": suffix,
            "dims": " ".join(dims.split()),
            "flags": " ".join(flags.split()),
            "required": required_from_flags(flags),
            "cond": conds[-1] if conds else None,
            "via": via,
        })

    def parse_tn(tn_text: str, dims: str, flags: str, cond, via: str) -> None:
        args = split_args(tn_text)
        enum = args[0].strip()
        rest = args[1:]
        suffix = None
        idx: list[str] = []
        for a in rest:
            a = a.strip()
            if a.startswith('"'):
                suffix = a.strip('"')
            elif re.fullmatch(r"-?\w+", a):
                idx.append(a)
        bid = PROBE_BID if idx else -1
        xid = PROBE_XID if len(idx) > 1 else -1
        add(enum, suffix, bid, xid, dims, flags, cond, via)

    def tn_source(expr: str) -> str | None:
        """Resolve an expression used as create_tensor()'s first argument to the
        `tn(...)` text it stands for.

        Usually inline, but a few models hoist it into a local first:
        `const auto tn_ffn_up_weight = tn(LLM_TENSOR_FFN_UP, "weight", i);`
        """
        e = expr.strip()
        e = aliases.get(e, e)
        return e if e.startswith("tn(") else None

    while i < n:
        ch = body[i]
        if ch in '"\'':
            q = ch
            i += 1
            while i < n and body[i] != q:
                i += 2 if body[i] == "\\" else 1
            i += 1
            continue
        if ch == "{":
            stack.append(pending)
            pending = None
            i += 1
            continue
        if ch == "}":
            if stack:
                stack.pop()
            pending = None
            i += 1
            continue
        if ch == ";":
            pending = None
            i += 1
            continue
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", body[i:])
        if not m:
            i += 1
            continue
        word = m.group(0)
        after = i + m.end()
        if word in BLOCK_KEYWORDS:
            got = call_args(body, after)
            if got is not None:
                cond_text, end = got
                pending = "else" if word == "else" else f"{word}({' '.join(cond_text.split())})"
                i = end
                continue
            pending = "else" if word == "else" else f"{word}(…)"
            i = after
            continue
        # `const auto <name> = tn(...);` -- remember the alias, then rewind so the
        # statement's own scan continues normally.
        decl = re.compile(r"\s*(?:const\s+)?auto\s+(\w+)\s*=\s*tn\(").match(body[after:])
        if decl:
            got = call_args(body, after + decl.end() - 1)
            if got is not None:
                aliases[decl.group(1)] = f"tn({got[0]})"
                i = got[1]
                continue
        if word == "create_tensor":
            got = call_args(body, after)
            if got is not None:
                args = split_args(got[0])
                cond = next((c for c in reversed(stack) if c), None)
                tn_src = tn_source(args[0]) if args else None
                if len(args) >= 3 and tn_src is not None:
                    parse_tn(tn_src[3:tn_src.rindex(")")], args[1], args[2], cond, "create_tensor")
                elif len(args) >= 3:
                    out.append({"name": None, "error": "create_tensor without tn()",
                                "call": " ".join(got[0].split())[:200], "via": "create_tensor"})
                i = got[1]
                continue
        if word in HELPER_EXPANSION:
            got = call_args(body, after)
            if got is not None:
                args = split_args(got[0])
                cond = next((c for c in reversed(stack) if c), None)
                flags = args[-1] if args else "0"
                bid = PROBE_BID if len(args) > 1 else -1
                for enum, suffix, fl in HELPER_EXPANSION[word]:
                    add(enum, suffix, bid, -1, "(helper)", flags if fl == "<flags>" else fl,
                        cond, word)
                i = got[1]
                continue
        i = after

    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def parse_bases(models_h: str) -> dict[str, str]:
    """struct llama_model_x : public llama_model_y -> {x: y}."""
    src = strip_comments(open(models_h, encoding="utf-8").read())
    return dict(re.findall(
        r"struct\s+(\w+)\s*:\s*public\s+(\w+)", src))


def find_load_tensors(models_dir: str, klass: str,
                      bases: dict[str, str] | None = None) -> tuple[str, str, str] | None:
    """Return (source path, function body, defining class) for load_arch_tensors.

    Some architectures are pure subclasses (`struct llama_model_mistral4 : public
    llama_model_deepseek2`) and never define their own loader; follow the chain.
    """
    bases = bases or {}
    seen: set[str] = set()
    cls = klass
    while cls and cls not in seen:
        seen.add(cls)
        for fname in sorted(os.listdir(models_dir)):
            if not fname.endswith(".cpp"):
                continue
            path = os.path.join(models_dir, fname)
            src = strip_comments(open(path, encoding="utf-8").read())
            at = src.find(f"void {cls}::load_arch_tensors")
            if at < 0:
                continue
            brace = src.index("{", at)
            return (os.path.join("src/models", fname),
                    src[brace:match_bracket(src, brace, "{", "}") + 1],
                    cls)
        cls = bases.get(cls)
    return None


def llama_cpp_commit(root: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp")))
    ap.add_argument("--out", default=os.path.join(here, "..", "tests", "data", "tensor_name_oracle.json"))
    ap.add_argument("--arch", action="append", default=None,
                    help="GGUF architecture string; repeatable. Default: all.")
    args = ap.parse_args()

    root = os.path.abspath(args.llama_cpp)
    src = os.path.join(root, "src")
    if not os.path.isfile(os.path.join(src, "llama-arch.cpp")):
        print(f"error: {src}/llama-arch.cpp not found; pass --llama-cpp", file=sys.stderr)
        return 1

    arch_names, tensor_patterns = parse_llama_arch(os.path.join(src, "llama-arch.cpp"))
    mapping = parse_arch_mapping(os.path.join(src, "llama-model.cpp"))
    bases = parse_bases(os.path.join(src, "models", "models.h"))
    gguf_to_enum = {v: k for k, v in arch_names.items()}

    wanted = args.arch or sorted(gguf_to_enum)
    archs_out = []
    unresolved = []
    for gguf in wanted:
        enum = gguf_to_enum.get(gguf)
        klass = mapping.get(enum) if enum else None
        found = find_load_tensors(os.path.join(src, "models"), klass, bases) if klass else None
        if not found:
            unresolved.append((gguf, enum, klass))
            continue
        rel, body, defined_by = found
        tensors = scan_body(body, tensor_patterns)
        archs_out.append({
            "gguf_arch": gguf,
            "llm_arch": enum,
            "model_class": klass,
            "loader_in": defined_by if defined_by != klass else None,
            "source": rel,
            "tensors": tensors,
        })

    if unresolved:
        print(f"warning: {len(unresolved)} architecture(s) skipped:", file=sys.stderr)
        for gguf, enum, klass in unresolved:
            print(f"  - {gguf} ({enum} -> {klass})", file=sys.stderr)

    doc = {
        "_generated_by": "engine/scripts/gen_tensor_name_oracle.py",
        "_source_repo": root,
        "_source_commit": llama_cpp_commit(root),
        "probe_block_index": PROBE_BID,
        "probe_expert_index": PROBE_XID,
        "architectures": archs_out,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")

    n_t = sum(len(a["tensors"]) for a in archs_out)
    print(f"wrote {out}: {len(archs_out)} architectures, {n_t} tensor declarations")
    return 0


if __name__ == "__main__":

    sys.exit(main())
