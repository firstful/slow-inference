#!/usr/bin/env python3
"""Does each of the seven GQA head-pairing sites carry weight *on its own*?

`scripts/mutation_check.py` carries one mutation for the tick-28 defect
(`gqa_kv_head_paired_by_modulo`) and applies it with `all_sites=True`, because at
the time all seven sites really were wrong together. That answers "would the
suite notice if the pairing were wrong?" It does not answer the question that
matters for a *fix*: **was every one of the seven sites part of the bug, and is
every one of them pinned by something?**

Those are two different claims and they can diverge in both directions. A site
that is wrong but uncovered means the fix was incomplete and the suite cannot
prove it was not. A site that is covered by nothing reachable -- because the
architecture that reaches it has no fixture, or its fixture is MHA so the branch
is short-circuited -- is dead weight that looks like a safety net.

So this script breaks the single all-sites mutation into seven one-site
mutations and runs each against the real suite, baseline-aware. It also reports,
for each site, which forward pass contains it and which GGUF files on disk reach
that forward pass with a grouped-query geometry, because a missed site is much
easier to explain than to merely report.

Method, in one line: find the seven `let kv_h = h / kv_group;` sites in
`engine/src/model.rs`, mutate exactly one of them per run, and record which
tests go red that were not already red.

Usage:
    python3 scripts/arb_gqa_site_power.py
    python3 scripts/arb_gqa_site_power.py --tree /tmp/opencode/snap
    python3 scripts/arb_gqa_site_power.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET_REL = "engine/src/model.rs"
SITE_LINE = "let kv_h = h / kv_group;"
#: The wrong pairing, spelled exactly as `mutation_check.py` spells it, so the two
#: harnesses are measuring the same defect.
WRONG = "let kv_h = h % n_kv_heads;"
#: Largest context window, in lines, used to make a site's find string unique.
MAX_WINDOW = 60


@dataclass
class Site:
    """One `h / kv_group` occurrence, with a find string unique to it."""

    line: int  # 1-based, in engine/src/model.rs
    fn: str  # enclosing function in the engine
    find: str
    lines_used: tuple[int, int] = (0, 0)  # 1-based inclusive line span
    #: Filled in by the measurement pass.
    verdict: str = "?"
    new_failures: list[str] = field(default_factory=list)
    #: Worst `rel L2 vs float64` the ref_parity gate reported, so a detected site
    #: carries a magnitude and not only a name. "Detected" and "detected by
    #: 1e-1" are different claims about how much weight a site carries.
    worst_rel_l2: float | None = None
    architectures: list[str] = field(default_factory=list)


REL_L2_RE = re.compile(r"(\w[\w.-]*) pos (\d+): rel L2 vs float64 ([0-9.eE+-]+)")


def parse_rel_l2(out: str) -> tuple[float | None, list[str]]:
    """Worst per-position `rel L2 vs float64` in a failing gate, and its arches.

    `ref_parity::forward_matches_llama_cpp_logits` puts the distance in the panic
    message on purpose, so the next question -- which of the three
    implementations is wrong -- is answerable from the number. Reading it back
    out of the failure is how a detection gets a magnitude attached.
    """
    worst, arches = None, set()
    for arch, _pos, val in REL_L2_RE.findall(out):
        arches.add(arch)
        v = float(val)
        worst = v if worst is None else max(worst, v)
    return worst, sorted(arches)


def enclosing_fn(lines: list[str], idx: int) -> str:
    """Name of the `fn` whose body contains line `idx` (0-based)."""
    for j in range(idx, -1, -1):
        m = re.match(r"    (?:pub )?fn (\w+)\(", lines[j])
        if m:
            return m.group(1)
    return "<module>"


def site_indices(lines: list[str]) -> list[int]:
    return [i for i, ln in enumerate(lines) if ln.strip() == SITE_LINE]


def unique_find(lines: list[str], others: list[int], idx: int) -> tuple[str, int, int]:
    """Smallest window around `idx` whose text occurs exactly once in the file.

    Windows that would swallow a *different* site are rejected outright: a
    one-site mutation that silently edits two sites is the exact thing this
    script exists to rule out, and the whole point would be lost if the harness
    itself could not tell.
    """
    text = "\n".join(lines)
    for width in range(1, MAX_WINDOW + 1):
        for before in range(0, width + 1):
            after = width - before
            lo, hi = idx - before, idx + after
            if lo < 0 or hi >= len(lines):
                continue
            if any(o >= lo and o <= hi for o in others):
                continue
            cand = "\n".join(lines[lo : hi + 1])
            if text.count(cand) == 1:
                return cand, lo + 1, hi + 1
    raise SystemExit(
        f"no unique window around line {idx + 1} within {MAX_WINDOW} lines; "
        "widen MAX_WINDOW or disambiguate the source"
    )


def parse_sites(src: str) -> list[Site]:
    lines = src.splitlines()
    idxs = site_indices(lines)
    if not idxs:
        raise SystemExit(f"no `{SITE_LINE}` found in {TARGET_REL}")
    sites = []
    for i in idxs:
        find, lo, hi = unique_find(lines, [o for o in idxs if o != i], i)
        if SITE_LINE not in find or WRONG in find:
            raise SystemExit(f"line {i + 1}: window does not contain the site")
        sites.append(
            Site(line=i + 1, fn=enclosing_fn(lines, i), find=find, lines_used=(lo, hi))
        )
    return sites


def dispatch_table(src: str) -> dict[str, str]:
    """`general.architecture` string -> engine forward pass, read from the source.

    Two tables have to agree for this to be right: the `arch_from_string` match
    arms and the `if self.arch_variant == ... { return self.forward_X(...) }`
    dispatch inside `forward`. Both are parsed rather than transcribed, because a
    transcription is exactly the kind of thing that goes stale silently.
    """
    arms: dict[str, str] = {}
    arm_re = re.compile(r'^\s*((?:"\w+"\s*\|\s*)*"\w+")\s*=>\s*ArchVariant::(\w+),')
    for ln in src.splitlines():
        m = arm_re.match(ln)
        if m:
            for name in re.findall(r'"(\w+)"', m.group(1)):
                arms[name] = m.group(2)
    variant_to_fn: dict[str, str] = {}
    for m in re.finditer(
        r"if self\.arch_variant == ArchVariant::(\w+)\s*\{\s*return self\.(forward_\w+)\(",
        src,
    ):
        variant_to_fn[m.group(1)] = m.group(2)
    return {arch: variant_to_fn.get(v, "forward") for arch, v in arms.items()}


def import_gguf():
    import os

    try:
        import gguf  # type: ignore

        return gguf
    except ImportError:
        pass
    for cand in (
        os.environ.get("LLAMA_CPP_GGUF_PY"),
        Path.home() / "llama.cpp" / "gguf-py",
    ):
        if cand and Path(cand).is_dir():
            sys.path.insert(0, str(cand))
            import gguf  # type: ignore

            return gguf
    return None


def gguf_geometry(path: Path, arch_to_fn: dict[str, str]) -> dict | None:
    """`{arch, n_head, n_head_kv, forward}` read out of the file, or None."""
    gguf = import_gguf()
    if gguf is None:
        return None
    kv: dict[str, object] = {}
    try:
        reader = gguf.GGUFReader(str(path), "r")
    except Exception as exc:  # a file that will not open is itself a finding
        return {"error": f"{type(exc).__name__}: {exc}"}
    for key, field in reader.fields.items():
        try:
            kv[key] = field.contents()
        except Exception:
            kv[key] = None
    arch = kv.get("general.architecture")
    if not isinstance(arch, str):
        arch = None
    n_head = kv.get(f"{arch}.attention.head_count")
    n_kv = kv.get(f"{arch}.attention.head_count_kv")
    if n_kv is None:
        # Absent upstream means "same as n_head" (llama.cpp defaults
        # n_head_kv to n_head), which is MHA, not GQA.
        n_kv = n_head
    n_head = int(n_head) if n_head is not None else None
    n_kv = int(n_kv) if n_kv is not None else None
    gqa = None
    if n_head and n_kv:
        # llama.cpp's rule: h / n_rep, and the wrong rule is h % n_kv. They
        # differ exactly when the group is neither 1 nor the whole head count.
        gqa = n_head != n_kv and n_kv > 1
    return {
        "file": path.name,
        "arch": arch,
        "n_head": n_head,
        "n_head_kv": n_kv,
        "n_rep": (n_head // n_kv) if (n_head and n_kv) else None,
        "gqa": gqa,
        "forward": arch_to_fn.get(arch or "", "?"),
    }


def _failed_test_names(out: str) -> set[str]:
    """Failing test names, excluding the `test result:` summary line.

    Borrowed from `mutation_check.py` for the same reason: `test result: FAILED`
    starts with "test " and contains "FAILED", so without the exclusion every
    failure set contains a constant named `result:` and hides the fact that no
    test was ever named.
    """
    return {
        ln.split(" ")[1]
        for ln in out.splitlines()
        if ln.startswith("test ") and " FAILED" in ln and not ln.startswith("test result:")
    }


def run_suite(engine: Path) -> tuple[set[str], str]:
    proc = subprocess.run(
        ["cargo", "test", "--release", "--no-fail-fast"],
        cwd=engine,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    if "error[E" in out or "error: could not compile" in out:
        raise SystemExit("the suite does not compile in the snapshot; aborting")
    return _failed_test_names(out), out


def snapshot(src: Path, dst: Path) -> None:
    """Copy the repo minus `target/` and `models/`, then link `models/` back.

    `models_dir()` is `CARGO_MANIFEST_DIR/../models`, i.e. *outside* the crate, so
    a symlink at `engine/models` is not seen -- it has to be at the snapshot root.
    Getting this wrong turns six tests red with `NotFound`, which looks exactly
    like a regression in whatever was being measured.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src, dst, symlinks=True, ignore=shutil.ignore_patterns("target", "models", "__pycache__")
    )
    (dst / "models").symlink_to((src / "models").resolve())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=None, help="snapshot dir (default: a fresh one)")
    ap.add_argument("--keep", action="store_true", help="do not delete the snapshot")
    ap.add_argument("--out", default=str(REPO / "results" / "gqa_site_power.json"))
    ap.add_argument("--list", action="store_true", help="list the sites and exit")
    args = ap.parse_args()

    target = REPO / TARGET_REL
    live_src = target.read_text()
    sites = parse_sites(live_src)
    arch_to_fn = dispatch_table(live_src)

    if args.list:
        for s in sites:
            print(f"{s.line:5d}  {s.fn:20s} lines {s.lines_used[0]}-{s.lines_used[1]}")
        return 0

    tree = Path(args.tree) if args.tree else Path("/tmp/opencode/gqa_site_power")
    # Keyed on the file being measured, not on the directory: an existing but
    # empty `--tree` used to skip the snapshot and then fail three lines later
    # with a FileNotFoundError on `src/model.rs`.
    if not args.tree or not (tree / TARGET_REL).exists():
        snapshot(REPO, tree)
    engine = tree / "engine"
    measured = tree / TARGET_REL

    # Parse the sites from the file that is about to be *measured*, not from the
    # live tree. A concurrent agent editing `src/model.rs` between the snapshot
    # and the run would otherwise leave find strings derived from one revision
    # applied to another, and the report would carry the live tree's md5 next to
    # numbers the live tree never produced. Same failure family as the
    # snapshot/models symlink trap, and it is silent.
    src = measured.read_text()
    if src != live_src:
        sites = parse_sites(src)
        arch_to_fn = dispatch_table(src)
        print("note: src/model.rs changed under the snapshot; measuring the snapshot's copy")
    live_md5 = hashlib.md5(live_src.encode()).hexdigest()
    measured_md5 = hashlib.md5(src.encode()).hexdigest()

    geometries = []
    for gg in sorted((REPO / "models").glob("*.gguf")):
        g = gguf_geometry(gg, arch_to_fn)
        if g and "error" not in g:
            geometries.append(g)
    fixtures_dir = REPO / "engine" / "tests" / "data" / "fixtures"
    if fixtures_dir.is_dir():
        for gg in sorted(fixtures_dir.glob("*.gguf")):
            g = gguf_geometry(gg, arch_to_fn)
            if g and "error" not in g:
                geometries.append(g)

    print(f"snapshot: {tree}")
    print(f"model.rs md5 (measured): {measured_md5}")
    print(f"model.rs md5 (live tree): {live_md5}")
    baseline, _ = run_suite(engine)
    total_tests = None
    print(f"baseline: {len(baseline)} already-failing test(s)")
    for name in sorted(baseline):
        print(f"  - {name}")
    print()

    pristine = measured.read_text()
    try:
        for s in sites:
            mutated = pristine.replace(s.find, s.find.replace(SITE_LINE, WRONG), 1)
            if mutated == pristine:
                s.verdict = "NOT APPLIED"
                continue
            if pristine.count(SITE_LINE) != 7:
                raise SystemExit("the site count changed mid-run; aborting")
            measured.write_text(mutated)
            try:
                after, out = run_suite(engine)
            finally:
                measured.write_text(pristine)
            new = sorted(after - baseline)
            s.new_failures = new
            s.worst_rel_l2, s.architectures = parse_rel_l2(out)
            s.verdict = "detected" if new else "MISSED"
            mag = f"{s.worst_rel_l2:.3e}" if s.worst_rel_l2 is not None else "-"
            print(
                f"{s.line:5d}  {s.fn:20s}  {s.verdict:9s} {mag:>9s}  "
                f"{', '.join(s.architectures) or '-'}  {', '.join(new[:3]) or '-'}"
            )
    finally:
        measured.write_text(pristine)

    uncovered = [s for s in sites if s.verdict != "detected"]
    reachable: dict[str, list] = {}
    for g in geometries:
        if g.get("gqa") and g.get("forward"):
            reachable.setdefault(g["forward"], []).append(g)

    report = {
        "question": "is each of the seven GQA head-pairing sites individually load-bearing?",
        "defect": f"`{SITE_LINE}` -> `{WRONG}`, one site at a time",
        "model_rs_md5_measured": measured_md5,
        "model_rs_md5_live_tree": live_md5,
        "site_count": len(sites),
        "baseline_failures": sorted(baseline),
        "sites": [
            {
                "line": s.line,
                "forward_pass": s.fn,
                "find_lines": list(s.lines_used),
                "find": s.find,
                "verdict": s.verdict,
                "worst_rel_l2_vs_float64": s.worst_rel_l2,
                "gated_architectures": s.architectures,
                "new_failures": s.new_failures,
            }
            for s in sites
        ],
        "uncovered_sites": [s.line for s in uncovered],
        "gguf_geometries": geometries,
        "gqa_ggufs_by_forward": reachable,
        "note": (
            "A site is only observable if a GGUF that reaches its forward pass has a "
            "grouped-query geometry: for MHA (n_head_kv == n_head) the expansion "
            "branches are short-circuited and `h / 1 == h % n_head == h`, so the "
            "mutation is a no-op by construction. `n_head_kv == 1` is likewise a "
            "coincidence, not coverage."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print()
    print(f"GGUF geometries reaching each forward pass:")
    for fn in sorted({s.fn for s in sites}):
        rows = [g for g in geometries if g.get("forward") == fn]
        gqa = [g for g in rows if g.get("gqa")]
        print(f"  {fn:20s} {len(gqa)}/{len(rows)} GQA: " + ", ".join(
            f"{g['file']}({g['n_head']}/{g['n_head_kv']})" for g in rows
        ))
    print()
    if uncovered:
        print(f"{len(uncovered)} of {len(sites)} site(s) NOT individually detected: "
              f"{[s.line for s in uncovered]}")
    else:
        print(f"all {len(sites)} sites individually detected")
    print(f"wrote {out}")
    if not args.keep and not args.tree:
        shutil.rmtree(tree, ignore_errors=True)
    return 1 if uncovered else 0


if __name__ == "__main__":
    raise SystemExit(main())
