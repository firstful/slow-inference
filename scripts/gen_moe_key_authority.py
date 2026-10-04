#!/usr/bin/env python3
"""What does each llama.cpp architecture actually pass to `build_moe_ffn`?

`build_moe_ffn` takes `norm_w`, `w_scale` and `gating_op` as plain arguments
(llama-graph.h:1075-1077), so what a `{arch}.expert_*` GGUF key is *worth* is
decided per architecture, in two separate places that are easy to conflate:

1. **whether the architecture reads the key at all.** `expert_weights_scale`
   initialises to `0.0f` and `build_moe_ffn` reads `0.0` as "unset" rather than
   "scale by zero"; `expert_weights_norm` initialises to `false`. Roughly forty
   architectures pass `hparams.expert_weights_scale` straight through and only
   twenty-two ever call `get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, ...)`.
2. **what it passes when the value is a literal.** `llama` passes a literal
   `true` for `norm_w` and `hparams.expert_weights_norm` is *never* consulted;
   `qwen2moe` passes a literal `false`. Reading the key for either is wrong.

So neither question is answerable from the key's presence in a GGUF, and the
engine's `moe_norm_w_for` / `moe_scale_for` tables are transcriptions of
`src/models/*.cpp`. This script asks the source the same question mechanically
and writes the answer to `engine/tests/data/moe_key_authority.json`, which
`engine/tests/moe_keys.rs` checks those tables against — so a transcription
cannot rot when upstream moves an architecture between groups.

Three resolutions happen here, all of which a grep gets wrong:

* **class hierarchy for `load_arch_hparams`.** `llama_model_base::load_hparams`
  is shared; the per-architecture key reads live in the pure-virtual
  `load_arch_hparams`. A subclass that defines none inherits its parent's
  wholesale -- `llama_model_mistral4` says so in a comment and has three lines of
  `build_arch_graph` -- so it reads everything the parent reads.
* **class hierarchy for the graph.** Same rule, different method: a subclass
  with no `build_arch_graph` inherits its parent's MoE call sites, so `mistral4`'s
  `norm_w` is whatever `deepseek2.cpp` passes.
* **argument position.** `build_moe_ffn` is overloaded (a 7-tensor and a
  13-tensor form) but `norm_w`, `w_scale` and `gating_op` are at indices 9, 10
  and 11 in both, so one index works for every call site.

No llama.cpp build required -- only a checkout. Regenerate with:

    python3 scripts/gen_moe_key_authority.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

KEYS = {
    # key name -> the hparam field it would land in
    "expert_weights_scale": ("LLM_KV_EXPERT_WEIGHTS_SCALE", "expert_weights_scale"),
    "expert_weights_norm": ("LLM_KV_EXPERT_WEIGHTS_NORM", "expert_weights_norm"),
    "expert_gating_func": ("LLM_KV_EXPERT_GATING_FUNC", "expert_gating_func"),
    "expert_groups": ("LLM_KV_EXPERT_GROUPS", "n_expert_group"),
    "expert_shared_feed_forward_length": (
        "LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH",
        "n_ff_shexp",
    ),
}

# `build_moe_ffn` is overloaded (llama-graph.h:1065 and :1086): a 7-tensor form
# and a 13-tensor form that adds a `_b` tensor beside every expert tensor. The
# parameter positions are the same *relative to the end* but not the same
# absolute -- `norm_w` is argument 9 in one and 13 in the other -- so the overload
# has to be identified per call site. `gpt-oss` is the only architecture using
# the wide form today, and reading its `norm_w` at index 9 yields `nullptr`,
# which is what caught this.
OVERLOAD_NARROW = (9, 10, 11)
OVERLOAD_WIDE = (13, 14, 15)
_WIDE_MARKERS = re.compile(r"\b(ffn_gate_inp_b|up_exps_b|gate_exps_b|down_exps_b|gate_up_exps_b)\b")


def llama_cpp_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def arch_enum_to_name(arch_cpp: Path) -> dict[str, str]:
    """`LLM_ARCH_LLAMA4` -> `"llama4"`, from `LLM_ARCH_NAMES` in llama-arch.cpp."""
    out: dict[str, str] = {}
    for enum, name in re.findall(r"\{\s*(LLM_ARCH_[A-Z0-9_]+)\s*,\s*\"([a-z0-9_.\-]+)\"\s*\}", arch_cpp.read_text()):
        out[enum] = name
    return out


def arch_to_class(model_cpp: Path) -> dict[str, str]:
    """`LLM_ARCH_LLAMA4` -> `"llama_model_llama4"`, from `llama_model_mapping()`."""
    out: dict[str, str] = {}
    for enum, cls in re.findall(
        r"case\s+(LLM_ARCH_[A-Z0-9_]+)\s*:\s*return new (llama_model_[a-z0-9_]+)\(params\)", model_cpp.read_text()
    ):
        out[enum] = cls
    return out


def class_chunks(models_h: Path) -> dict[str, tuple[str | None, str | None]]:
    """class -> (base class, graph-alias class), one entry per `struct`.

    Split on `struct llama_model_` rather than matching across the whole file:
    a struct body can contain `{}` (an inline constructor is usually
    `llama_model_x(params) {}`) and a body-limited pattern like `[^}]*` then
    cannot reach the `using graph = ...` line inside it. Chunks keep each
    declaration's two facts together and stop a match leaking into the next
    struct.
    """
    text = models_h.read_text()
    out: dict[str, tuple[str | None, str | None]] = {}
    for part in re.split(r"\nstruct\s+", text)[1:]:
        cls = re.match(r"(llama_model_[a-z0-9_]+)", part)
        if not cls:
            continue
        base = re.search(r":\s*public\s+(llama_model_[a-z0-9_]+)", part)
        alias = re.search(r"using graph = (llama_model_[a-z0-9_]+)::graph", part)
        out[cls.group(1)] = (base.group(1) if base else None, alias.group(1) if alias else None)
    return out


def class_parents(chunks: dict[str, tuple[str | None, str | None]]) -> dict[str, str | None]:
    """`llama_model_mistral4` -> `"llama_model_deepseek2"` (absent = direct base)."""
    return {cls: base for cls, (base, _) in chunks.items()}


def class_graph_alias(chunks: dict[str, tuple[str | None, str | None]]) -> dict[str, str]:
    """`llama_model_mistral4` -> `"llama_model_deepseek2"`, from `using graph = ...`.

    A subclass can reuse its parent's whole graph type, in which case the MoE
    call sites live in the *parent's* .cpp even though the subclass overrides
    `build_arch_graph`. `mistral4` and `deepseek2-ocr` both do this, so their
    routing is deepseek2's routing; resolving by base class alone would miss it,
    because `deepseek2-ocr`'s base is `llama_model_base`.
    """
    return {cls: alias for cls, (_, alias) in chunks.items() if alias}


def class_sources(models_dir: Path) -> dict[str, Path]:
    """class name -> the .cpp that defines it, searched rather than guessed.

    The file is usually `<suffix>.cpp` but not always (`llama_model_jina_bert_v2`
    lives in `jina-bert-v2.cpp`), so index every file by the classes it *defines
    a member of*.
    """
    out: dict[str, Path] = {}
    for path in sorted(models_dir.glob("*.cpp")):
        text = path.read_text()
        for cls in set(re.findall(r"\b(llama_model_[a-z0-9_]+)\b", text)):
            if re.search(rf"{cls}::\w+\s*\(", text):
                out.setdefault(cls, path)
    return out


def reads_key(path: Path, enum: str) -> bool:
    return re.search(rf"get_key\(\s*{enum}\b", path.read_text()) is not None


def defines(path: Path, method: str) -> bool:
    return re.search(rf"\b{method}\s*\(", path.read_text()) is not None


def split_args(arglist: str) -> list[str]:
    """Split a C++ argument list on top-level commas."""
    args, depth, cur = [], 0, []
    for ch in arglist:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return args


def moe_call_sites(path: Path) -> list[list[str]]:
    """Every `build_moe_ffn(...)` argument list in a file, comments stripped.

    Comments matter: `hparams.expert_weights_scale, // w_scale (route_scale=2.826)`
    is a different string from a bare argument, and a naive split on the raw text
    produces `hparams.expert_weights_scale, // ...` as one argument.
    """
    text = re.sub(r"//[^\n]*", "", path.read_text())
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    out: list[list[str]] = []
    # The lookbehind skips the *definition*
    # (`llama_model_x::graph::build_moe_ffn(...) const {`), whose parameter list
    # is a type signature, not a set of arguments -- and whose 10th parameter is
    # `llama_expert_gating_func_type gating_op`, which would otherwise be
    # recorded as deepseek2's routing decision.
    for m in re.finditer(r"(?<![:\w])build_moe_ffn\s*\(", text):
        i = m.end()
        depth = 1
        while i < len(text) and depth:
            if text[i] in "([{":
                depth += 1
            elif text[i] in ")]}":
                depth -= 1
            i += 1
        out.append(split_args(text[m.end() : i - 1]))
    return out


def arg_positions(args: list[str]) -> tuple[int, int, int]:
    """Indices of `norm_w`, `w_scale`, `gating_op` for one call site."""
    return OVERLOAD_WIDE if _WIDE_MARKERS.search(" ".join(args)) else OVERLOAD_NARROW


def normalise(arg: str) -> str:
    """Reduce an argument to something a table can be keyed on."""
    a = " ".join(arg.split())
    # deepseek2 passes its gating function as a C-style cast of the hparam:
    # `(llama_expert_gating_func_type) hparams.expert_gating_func`.
    a = re.sub(r"^\(\s*[\w:]+\s*\)\s*", "", a)
    if a in ("true", "false"):
        return f"literal_{a}"
    m = re.fullmatch(r"hparams\.(expert_\w+)", a)
    if m:
        return f"key:{m.group(1)}"
    m = re.fullmatch(r"LLAMA_EXPERT_GATING_FUNC_TYPE_(\w+)", a)
    if m:
        return f"gating:{m.group(1)}"
    return f"expr:{a[:40]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", type=Path, default=Path.home() / "llama.cpp")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "engine/tests/data/moe_key_authority.json",
    )
    args = ap.parse_args()
    root: Path = args.llama_cpp
    src = root / "src"

    names = arch_enum_to_name(src / "llama-arch.cpp")
    classes = arch_to_class(src / "llama-model.cpp")
    chunks = class_chunks(src / "models/models.h")
    parents = class_parents(chunks)
    graph_alias = class_graph_alias(chunks)
    sources = class_sources(src / "models")

    def hparams_of(cls: str, seen: frozenset[str]) -> tuple[Path | None, list[str]]:
        """(file whose `load_arch_hparams` the class uses, inheritance chain)."""
        if cls in seen:
            return None, []
        seen = seen | {cls}
        path = sources.get(cls)
        if path is None:
            return None, [cls]
        chain = [cls]
        if defines(path, "load_arch_hparams"):
            return path, chain
        if parent := parents.get(cls):
            up, upchain = hparams_of(parent, seen)
            return up, chain + upchain
        return None, chain

    def graph_of(cls: str, seen: frozenset[str]) -> tuple[Path | None, list[str]]:
        """(file defining the `build_moe_ffn` call sites this arch routes through).

        Follows `using graph = <parent>::graph` first: a subclass that reuses its
        parent's graph type routes exactly as the parent does, no matter which
        file its own `build_arch_graph` is in.
        """
        if cls in seen:
            return None, []
        seen = seen | {cls}
        chain = [cls]
        if alias := graph_alias.get(cls):
            up, upchain = graph_of(alias, seen)
            return up, chain + upchain
        path = sources.get(cls)
        if path is None:
            return None, chain
        if defines(path, "build_arch_graph"):
            if moe_call_sites(path):
                return path, chain
            # An override with no MoE call of its own still delegates when it
            # reuses the parent graph; otherwise there is nothing to record.
            if parent := parents.get(cls):
                up, upchain = graph_of(parent, seen)
                return up, chain + upchain
            return None, chain
        if parent := parents.get(cls):
            up, upchain = graph_of(parent, seen)
            return up, chain + upchain
        return None, chain

    rows: dict[str, dict] = {}
    for enum, name in sorted(names.items(), key=lambda kv: kv[1]):
        cls = classes.get(enum)
        if cls is None:
            continue
        entry: dict = {"class": cls}
        if parent := parents.get(cls):
            entry["inherits"] = parent

        for key, (kenum, _field) in KEYS.items():
            hpath, chain = hparams_of(cls, frozenset())
            ok = bool(hpath) and reads_key(hpath, kenum)
            entry[key] = ok
            if ok and len(chain) > 1:
                entry.setdefault("key_chain", {})[key] = " <- ".join(chain)

        gpath, gchain = graph_of(cls, frozenset())
        sites = moe_call_sites(gpath) if gpath else []
        # Self-check: a mis-identified overload shows up as `norm_w` being
        # something no architecture ever passes. Better to fail here than to
        # publish an oracle whose routing columns are shifted.
        for s in sites:
            i_nw, i_ws, i_g = arg_positions(s)
            if len(s) <= i_g:
                raise SystemExit(
                    f"{gpath.name}: build_moe_ffn call with {len(s)} arguments is too "
                    f"short for the identified overload ({i_g=}); the parser is wrong"
                )
            nw = normalise(s[i_nw])
            if nw not in ("literal_true", "literal_false", "key:expert_weights_norm"):
                raise SystemExit(
                    f"{gpath.name}: norm_w argument {i_nw} normalised to {nw!r}, which "
                    f"no architecture passes -- overload identified as "
                    f"{'wide' if i_nw == 13 else 'narrow'} but the arguments are "
                    f"{s!r}"
                )
        entry["builds_moe"] = bool(sites)
        if sites:
            entry["graph_in"] = gpath.name
            if len(gchain) > 1:
                entry["graph_chain"] = " <- ".join(gchain)
            entry["norm_w"] = sorted({normalise(s[arg_positions(s)[0]]) for s in sites})
            entry["w_scale"] = sorted({normalise(s[arg_positions(s)[1]]) for s in sites})
            entry["gating_op"] = sorted({normalise(s[arg_positions(s)[2]]) for s in sites})
            # A call site that forwards the key and an arch that does not read it
            # is the exact shape of the bug this oracle exists for.
            for field, keyname in (
                ("norm_w", "expert_weights_norm"),
                ("w_scale", "expert_weights_scale"),
            ):
                forwarded = any(v == f"key:{keyname}" for v in entry[field])
                entry[f"{field}_key_is_live"] = forwarded and entry[keyname]

        entry["any_key_read"] = any(entry[k] for k in KEYS)
        rows[name] = entry

    doc = {
        "llama_cpp_commit": llama_cpp_commit(root),
        "generator": "scripts/gen_moe_key_authority.py",
        "question": (
            "Per GGUF architecture string: (a) does the model class's load_hparams "
            "-- its own, or an inherited one when the subclass does not override it "
            "-- call ml.get_key(LLM_KV_*, hparams.<field>, ...)? (b) what does "
            "build_moe_ffn receive for norm_w (arg 9), w_scale (arg 10) and "
            "gating_op (arg 11) at that architecture's call sites?"
        ),
        "not_read_means": {
            "expert_weights_scale": "0.0f = 'unset' (llama-hparams.h:100); build_moe_ffn skips 0 and 1",
            "expert_weights_norm": "false (llama-hparams.h:101)",
            "expert_gating_func": "NONE (0), which build_moe_ffn aborts on -- each arch resolves it first",
        },
        "archs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")

    moe = [a for a, e in rows.items() if e["builds_moe"]]
    reading = [a for a, e in rows.items() if e["expert_weights_scale"]]
    # The precise statement: forwards the key AND does not read it. Using
    # `not w_scale_key_is_live` instead would also count gemma4, which forwards
    # nothing at all -- it passes a literal `1.0f`.
    dead = sorted(
        a for a in moe
        if rows[a]["w_scale"] == ["key:expert_weights_scale"]
        and not rows[a]["expert_weights_scale"]
    )
    print(
        f"wrote {args.out} ({args.out.stat().st_size} bytes): {len(rows)} architectures, "
        f"{len(moe)} call build_moe_ffn, {len(reading)} read expert_weights_scale"
    )
    print(f"  forward w_scale but do NOT read the key ({len(dead)}): {' '.join(dead)}")
    for arch in ("llama", "llama4", "olmoe", "deepseek2", "qwen2moe", "mistral4", "gpt2"):
        if arch in rows:
            e = rows[arch]
            print(
                f"  {arch:14} norm_w={e.get('norm_w')} w_scale={e.get('w_scale')} "
                f"gating={e.get('gating_op')}"
            )


if __name__ == "__main__":
    main()