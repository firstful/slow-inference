#!/usr/bin/env python3
"""
Continuous slow-inference research agent (self-hosted).

Runs one work tick per invocation against the llm-gateway `free` pseudo-model,
appends a tick report, commits, and pushes to Gitea. A separate daily cron reads
the accumulated reports and produces the summary.

This replaces the `opencode run` loop that used to drive this project: the work
is identical (read ROADMAP/STATUS, pick the next item, implement, test, record),
only the driver changed — and the driver now costs nothing, because llm-gateway
`free` never spends OpenCode Go quota.

Usage:
  ./agent_tick.py                      # one tick, model=free
  ./agent_tick.py --model auto         # quality path (spends OCG quota)
  ./agent_tick.py --dry-run            # no writes, print the diff
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
GATEWAY = os.environ.get("LLM_GATEWAY", "http://llm.homelab.internal")
MODEL = os.environ.get("AGENT_MODEL", "free")
REPORTS = REPO / "agent-ticks"
STATE_FILE = REPO / "agent-ticks" / "state.json"
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "2500"))
TIMEOUT = int(os.environ.get("AGENT_TIMEOUT", "900"))

TOOLS = """You have these tools. Call one by emitting EXACTLY:
<functioncall>{"name": "tool", "arguments": {...}}</functioncall>

Tools:
- bash    {"command": "shell command"}          run a command, get stdout/stderr
- read    {"path": "...", "offset": 1, "limit": 200}
- write   {"path": "...", "content": "..."}
- grep    {"pattern": "regex", "path": "..."}   search the repo

Reply with plain text (no tool call) when the tick is done. Be terse.

CRITICAL RULES:
- Keep EVERY tool call under 300 characters. Short commands only, one at a time.
- No pipes into long output chains; use `ls`, `head -40`, `grep -c` instead.
- Never echo large files; use the read tool with a small limit.
- Emit exactly ONE <functioncall> per reply, then wait for the result."""


def do_tool(name, args):
    """Execute one tool call, return a bounded result string."""
    args = args or {}
    try:
        if name == "bash":
            r = run(args.get("command", ""), timeout=300)
            out = (r.stdout + r.stderr).strip()
            return out[:4000] or f"(exit {r.returncode}, no output)"
        if name == "read":
            path = REPO / args["path"].lstrip("/")
            if not path.exists():
                return f"ERROR: no such file: {args['path']}"
            off = int(args.get("offset", 1))
            lim = int(args.get("limit", 200))
            lines = path.read_text(errors="replace").splitlines()[off - 1:off - 1 + lim]
            return "\n".join(f"{off+i}|{l}" for i, l in enumerate(lines))
        if name == "write":
            path = REPO / args["path"].lstrip("/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args.get("content", ""))
            return f"wrote {len(args.get('content',''))} bytes to {args['path']}"
        if name == "grep":
            r = run(f"grep -rn --include=*.rs --include=*.py --include=*.md "
                    f"{json.dumps(args.get('pattern',''))} {json.dumps(args.get('path','.'))} | head -60")
            return (r.stdout + r.stderr).strip()[:3000] or "(no matches)"
    except Exception as e:
        return f"ERROR: tool {name} failed: {e}"
    return f"ERROR: unknown tool {name}"


# Free models sometimes emit zero-width chars inside the tags
# (observed: "<\u200bfunctioncall>"). Strip them before matching.
TOOLCALL_RE = re.compile(r"<\s*functioncall\s*>\s*(\{.*?\})\s*<\s*/\s*functioncall\s*>", re.S)


def strip_invisible(s: str) -> str:
    return re.sub(r"[\u200b\u200c\u200d\ufeff]", "", s)


def run_agent_loop(prompt, model, max_iters=12):
    """Multi-turn tool loop: keep feeding tool results back until the model
    replies with plain text (no tool call). Returns (text, served, n_calls)."""
    served, calls = "?", 0
    for _ in range(max_iters):
        text, served = ask(prompt, model=model)
        matches = salvage_calls(text)
        if not matches:
            return text, served, calls
        results = []
        for m in matches:
            try:
                tname, targs = _parse_call(m)
            except Exception as e:
                results.append(f"$ {m}\nERROR: bad tool call JSON: {e}")
                continue
            results.append(f"$ {tname} {json.dumps(targs)[:200]}\n{do_tool(tname, targs)}")
        prompt = ("Result of your tool call(s):\n\n" + "\n\n".join(results)
                  + "\n\nContinue. Use another tool call or reply with plain text when done.")
        calls += len(matches)
    return text, served, calls


def salvage_calls(text):
    """Match well-formed calls, plus one truncated tail: an opening tag whose
    JSON was cut off by max_tokens still tells us the tool and most args."""
    clean = strip_invisible(text)
    good = TOOLCALL_RE.findall(clean)
    consumed = sum(len(m) for m in good)
    if "<functioncall>" in clean[consumed:] and "</functioncall>" not in clean[consumed:]:
        blob = clean[clean.index("<functioncall>", consumed):]
        blob = blob[blob.index("{") + 1:]
        depth, out, in_str, esc = 1, [], False, False
        for ch in blob:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str and ch == "{":
                depth += 1
            elif not in_str and ch == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append(ch)
        good.append("{" + "".join(out))
    return good


def _parse_call(blob):
    d = json.loads(blob)
    return d.get("name", "bash"), d.get("arguments", {})


def run(cmd, timeout=120, cwd=REPO):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                          text=True, timeout=timeout)


def ask(prompt, model=MODEL, max_tokens=MAX_TOKENS):
    """One completion through the gateway. Returns text."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    import urllib.request
    req = urllib.request.Request(
        f"{GATEWAY}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "slow-inference-agent/1.0"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"], data.get("model", "?")


def repo_brief(maxchars=6000):
    """Compact orientation: roadmap tail, status, last tick's outcome."""
    parts = []
    for name, limit in (("ROADMAP.md", 2500), ("STATUS.json", 2500)):
        p = REPO / name
        if not p.exists():
            continue
        txt = p.read_text(errors="replace")
        parts.append(f"--- {name} (tail) ---\n{txt[-limit:]}")
    prior = sorted(REPORTS.glob("tick-*.md"))
    if prior:
        parts.append(f"--- previous tick ({prior[-1].name}) ---\n{prior[-1].read_text()[:1800]}")
    return "\n\n".join(parts)[:maxchars]


def tick_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"n": 0, "last_model": None, "last_at": None}


def commit_and_push(message, dry=False):
    if dry:
        return "dry-run: not committed"
    run("git add -A")
    has_changes = run("git diff --cached --quiet").returncode != 0
    if not has_changes:
        return "no changes"
    run(f'git -c user.name=agent -c user.email=agent@ffilt.us commit -q -m {json.dumps(message)}')
    push = run("git push -q origin HEAD", timeout=180)
    return "pushed" if push.returncode == 0 else f"push FAILED: {push.stderr.strip()[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--goal", default=None, help="override the tick objective")
    args = ap.parse_args()

    st = tick_state()
    n = st["n"] + 1
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    goal = args.goal or (
        "Pick the single highest-value unfinished item from ROADMAP.md that does not "
        "need a real MoE+GQA model on disk. Implement it in the Rust engine under engine/src, "
        "verify with the existing tests or a new example, and record what you measured. "
        "Prefer a measurement the project has NOT made before over more polish on what exists."
    )

    prompt = f"""You are the continuous research agent for the `slow-inference` project
(disk-streamed LLM inference in Rust, zero external crates; goal is to map the RAM floor,
accepting seconds-per-token).

Tick #{n} — {now}. Objective: {goal}

{TOOLS}

Here is the current state of the repo:

{repo_brief()}

Work the objective. Use tools. When you are done, end your reply with a line of the form:
SUMMARY: <one sentence: what you did and what you measured>
BLOCKED: <reason, or omit>

Keep it to one focused change per tick.
"""

    try:
        text, served, ncalls = run_agent_loop(prompt, model=args.model)
    except Exception as e:
        print(f"ERROR: gateway call failed: {e}", file=sys.stderr)
        return 1

    REPORTS.mkdir(exist_ok=True)
    report = REPORTS / f"tick-{n:04d}.md"
    report.write_text(
        f"# Tick {n} — {now}\n\n"
        f"- model requested: `{args.model}`\n"
        f"- model served: `{served}` ({ncalls} tool calls)\n"
        f"- objective: {goal}\n\n"
        f"## Agent output\n\n```\n{text.strip()}\n```\n"
    )

    summary = re.search(r"SUMMARY:\s*(.+)", text)
    blocked = re.search(r"BLOCKED:\s*(.+)", text)
    msg = f"agent tick {n} ({ncalls} tool calls): {(summary.group(1).strip() if summary else 'no SUMMARY line')[:160]}"
    if blocked:
        msg += f" [BLOCKED: {blocked.group(1).strip()[:100]}]"

    result = commit_and_push(msg, args.dry_run)
    STATE_FILE.write_text(json.dumps(
        {"n": n, "last_model": served, "last_at": now, "last_summary": summary.group(1).strip() if summary else None},
        indent=2))

    print(f"tick {n} | served={served} | {result}\n{msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())