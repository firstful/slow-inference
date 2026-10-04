#!/usr/bin/env python3
"""
Slow-Inference Agent Harness — Pi-style tool-calling loop for the Rust disk-streamed engine.

Architecture:
  - Prompt is written to a temp file, engine spawned in background
  - Engine output redirected to a result file
  - Harness polls result file for completion marker (output text:)
  - No blocking subprocess — turns can take hours at 10s+/tok
  - Tool calls parsed from <functioncall> XML tags in generated text

Usage:
  ./slow_agent.py --model path/to/model.gguf "Your prompt here"
  ./slow_agent.py --model path/to/model.gguf --max-turns 5 "Write a Python script"
  ./slow_agent.py --model ... --json-output "..."   # structured JSON on stdout
  ./slow_agent.py --verbose                         # debug info to stderr
"""

import subprocess
import time
import os
import sys
import json
import re
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# ── Config ──────────────────────────────────────────────────────────────────

ENGINE_PATH = os.path.expanduser("~/slow-inference/engine/target/release/slow-inference")
MODELS_DIR = os.path.expanduser("~/slow-inference/models")
WORK_DIR = os.path.expanduser("~/slow-inference")
AGENT_DIR = os.path.join(WORK_DIR, ".agent")  # prompt/result files live here


def build_system_prompt() -> str:
    """Build the system prompt with tool definitions."""
    tools_desc = []
    tools_desc.append("""You have access to these tools. To call one, output:

<functioncall>{"name": "tool_name", "arguments": {"arg1": "val1"}}</functioncall>

Available tools:""")

    TOOLS = [
        ("bash", "Run a shell command. Returns stdout/stderr.", {"command": "string (required) — the command to execute"}),
        ("read", "Read a file with line numbers. Use instead of cat.", {"path": "string (required)", "offset": "integer (default 1)", "limit": "integer (default 200)"}),
        ("write", "Write content to a file (overwrites). Creates parent dirs.", {"path": "string (required)", "content": "string (required)"}),
        ("edit", "Find-and-replace in a file. Uses fuzzy matching.", {"path": "string (required)", "old_string": "string (required)", "new_string": "string (required)"}),
        ("search", "Search file contents (regex) or find files by name.", {"path": "string (default .)", "pattern": "string (required)", "target": "string: 'content' or 'files'"}),
        ("list", "List directory contents.", {"path": "string (default .)", "pattern": "string (optional glob)"}),
    ]

    for name, desc, args in TOOLS:
        args_str = ", ".join(f"{k}" for k, v in args.items())
        tools_desc.append(f"\n  {name}: {desc}  Args: {args_str}")

    tools_desc.append("""
After each tool call, the result will be shown. You can call multiple tools in sequence.
When the task is complete, output a normal response without <functioncall> tags.
Think step-by-step, then call the appropriate tool.""")
    
    return "".join(tools_desc)


# ── File-based engine adapter (async/poll) ───────────────────────────────────

def spawn_engine(
    model_path: str,
    prompt: str,
    max_tokens: int = 512,
    engine_path: str = ENGINE_PATH,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Write prompt to a file, spawn engine in background with output redirected.
    
    Returns session info dict with paths needed to poll for results.
    The caller calls poll_engine_result() periodically to check completion.
    """
    os.makedirs(AGENT_DIR, exist_ok=True)
    
    if session_id is None:
        session_id = uuid.uuid4().hex[:12]
    
    prompt_file = os.path.join(AGENT_DIR, f"{session_id}_prompt.txt")
    result_file = os.path.join(AGENT_DIR, f"{session_id}_result.txt")
    done_marker = os.path.join(AGENT_DIR, f"{session_id}_done")

    engine_path = os.path.abspath(engine_path)
    model_path = os.path.abspath(model_path)

    # Write prompt to file
    with open(prompt_file, "w") as f:
        f.write(prompt)

    # Generate a shell wrapper that reads the prompt and calls the engine
    # (avoids shell escaping issues with multiline/long prompts)
    wrapper = os.path.join(AGENT_DIR, f"{session_id}_run.sh")
    with open(wrapper, "w") as f:
        f.write(
            f'#!/bin/bash\n'
            f'PROMPT=$(cat "{prompt_file}")\n'
            f'"{engine_path}" --model "{model_path}" --prompt "$PROMPT" '
            f'--tokens {max_tokens} --json > "{result_file}" 2>&1\n'
            f'echo "DONE:$?" > "{done_marker}"\n'
            f'rm -f "{wrapper}"\n'
        )
    os.chmod(wrapper, 0o755)

    proc = subprocess.Popen(
        ["/bin/bash", wrapper],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=WORK_DIR,
    )

    return {
        "session_id": session_id,
        "pid": proc.pid,
        "prompt_file": prompt_file,
        "result_file": result_file,
        "done_marker": done_marker,
        "max_tokens": max_tokens,
        "engine_path": engine_path,
        "model_path": model_path,
    }


def poll_engine_result(
    session: Dict[str, Any],
    poll_interval: float = 5.0,
) -> Dict[str, Any]:
    """
    Poll for engine completion by checking the done marker file.
    
    Returns the parsed result dict once complete.
    Blocks until done.
    """
    done_marker = session["done_marker"]
    result_file = session["result_file"]
    
    t0 = time.time()
    
    while True:
        if os.path.exists(done_marker):
            with open(done_marker) as f:
                content = f.read().strip()
            exit_code = int(content.replace("DONE:", "")) if "DONE:" in content else -1
            
            # Parse the result file
            os.makedirs(os.path.dirname(result_file) or ".", exist_ok=True)
            if os.path.exists(result_file):
                with open(result_file) as f:
                    output = f.read()
            else:
                output = ""
            
            elapsed = time.time() - t0
            
            # Parse output text from result
            output_text = ""
            for line in output.splitlines():
                if line.startswith("output text:"):
                    raw = line[len("output text:"):].strip()
                    if raw.startswith('"') and raw.endswith('"'):
                        output_text = raw[1:-1]
                    else:
                        output_text = raw
                    break

            # Parse ENGINEJSON
            json_payload = None
            for line in output.splitlines():
                if line.startswith("ENGINEJSON "):
                    try:
                        json_payload = json.loads(line[len("ENGINEJSON "):])
                    except json.JSONDecodeError:
                        pass
                    break

            # Count generated tokens
            gen_tokens = 0
            if json_payload and "gen_seq" in json_payload and "prompt_ids" in json_payload:
                gen_tokens = len(json_payload["gen_seq"]) - len(json_payload["prompt_ids"])

            # Clean up marker file
            try:
                os.remove(done_marker)
            except OSError:
                pass

            return {
                "stdout": output,
                "returncode": exit_code,
                "elapsed": elapsed,
                "json": json_payload,
                "output_text": output_text,
                "gen_tokens": gen_tokens,
                "session_id": session["session_id"],
            }
        
        time.sleep(poll_interval)


def poll_engine_result_nb(
    session: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Non-blocking poll — returns None if still running, result dict if done.
    """
    done_marker = session["done_marker"]
    if not os.path.exists(done_marker):
        return None
    return poll_engine_result(session, poll_interval=0)


def cleanup_session(session: Dict[str, Any]):
    """Remove temp files for a session."""
    for key in ["prompt_file", "result_file", "done_marker"]:
        f = session.get(key)
        if f and os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass


# ── Tool parser ─────────────────────────────────────────────────────────────

TOOL_CALL_RE = re.compile(
    r'<functioncall>\s*(\{.*?\})\s*</functioncall>', re.DOTALL
)


def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any], str]]:
    """Parse <functioncall>...</functioncall> tags from generated text."""
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        raw = match.group(1)
        try:
            parsed = json.loads(raw)
            name = parsed.get("name", "")
            args = parsed.get("arguments", {})
            if name and isinstance(args, dict):
                calls.append((name, args, raw))
            else:
                calls.append(("__malformed__", {"raw": raw}, raw))
        except json.JSONDecodeError:
            calls.append(("__parse_error__", {"raw": raw}, raw))
    return calls


def execute_tool(name: str, args: Dict[str, Any], workdir: str = ".") -> str:
    """Execute a tool and return its output."""
    try:
        if name == "bash":
            command = args.get("command", "")
            if not command:
                return "Error: no command provided"
            proc = subprocess.run(
                ["/bin/bash", "-c", command],
                capture_output=True, text=True, timeout=120,
                cwd=workdir,
            )
            out = proc.stdout
            if proc.stderr:
                out += "\n--- stderr ---\n" + proc.stderr[:2000]
            if proc.returncode != 0:
                out += f"\n--- exit code: {proc.returncode} ---"
            return out.strip() or "(no output)"

        elif name == "read":
            path = args.get("path", "")
            if not path:
                return "Error: no path provided"
            offset = int(args.get("offset", 1))
            limit = int(args.get("limit", 200))
            path = os.path.expanduser(path)
            if not os.path.exists(path):
                return f"Error: file not found: {path}"
            with open(path) as f:
                lines = f.readlines()
            start = max(0, offset - 1)
            end = min(len(lines), start + min(limit, 500))
            result = []
            for i in range(start, end):
                result.append(f"{i + 1}|{lines[i].rstrip()}")
            if start > 0:
                result.insert(0, f"... (skipped {start} lines)")
            if end < len(lines):
                result.append(f"... ({len(lines) - end} more lines)")
            return "\n".join(result)

        elif name == "write":
            path = args.get("path", "")
            content = args.get("content", "")
            if not path:
                return "Error: no path provided"
            path = os.path.expanduser(path)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"Written {len(content)} bytes to {path}"

        elif name == "edit":
            path = args.get("path", "")
            old = args.get("old_string", "")
            new = args.get("new_string", "")
            if not path or not old:
                return "Error: path and old_string required"
            path = os.path.expanduser(path)
            if not os.path.exists(path):
                return f"Error: file not found: {path}"
            with open(path) as f:
                content = f.read()
            if old not in content:
                return f"Error: old_string not found in {path}"
            new_content = content.replace(old, new, 1)
            replaced = content.count(old)
            with open(path, "w") as f:
                f.write(new_content)
            return f"Replaced 1 occurrence in {path} ({replaced - 1} remaining)"

        elif name == "search":
            path = args.get("path", ".")
            pattern = args.get("pattern", "")
            target = args.get("target", "content")
            path = os.path.expanduser(path)
            if target == "files":
                cmd = f'find "{path}" -name "{pattern}" -type f 2>/dev/null | head -50'
            else:
                cmd = f'grep -rn "{pattern}" "{path}" 2>/dev/null | head -50'
            proc = subprocess.run(
                ["/bin/bash", "-c", cmd],
                capture_output=True, text=True, timeout=30,
            )
            return proc.stdout.strip() or "(no matches)"

        elif name == "list":
            path = args.get("path", ".")
            pattern = args.get("pattern", "")
            path = os.path.expanduser(path)
            if pattern:
                cmd = f'ls -la "{path}/{pattern}" 2>/dev/null | head -50'
            else:
                cmd = f'ls -la "{path}" 2>/dev/null | head -50'
            proc = subprocess.run(
                ["/bin/bash", "-c", cmd],
                capture_output=True, text=True, timeout=10,
            )
            return proc.stdout.strip() or "(empty or not found)"

        else:
            return f"Error: unknown tool '{name}'"

    except subprocess.TimeoutExpired:
        return f"Error: tool '{name}' timed out (120s)"
    except Exception as e:
        return f"Error executing {name}: {e}"


# ── Agent loop ──────────────────────────────────────────────────────────────

def agent_loop(
    model_path: str,
    initial_prompt: str,
    max_turns: int = 10,
    max_tokens_per_turn: int = 1024,
    engine_path: str = ENGINE_PATH,
    verbose: bool = False,
    poll_seconds: float = 5.0,
) -> Tuple[str, List[Dict]]:
    """
    Pi-style agent loop: generate -> parse tool calls -> execute -> loop.
    
    Each generation turn spawns the engine in background and polls a result
    file until completion. No blocking subprocess with long timeout.
    
    Returns (final_text, message_history).
    """
    system = build_system_prompt()
    history = []

    context = f"{system}\n\n{initial_prompt}"

    def log(*a, **kw):
        if verbose:
            print(*a, **kw, file=sys.stderr)

    log(f"Agent loop: {max_turns} turns, {max_tokens_per_turn} tok/turn, {os.path.basename(model_path)}")
    
    session_results_dir = os.path.join(AGENT_DIR, f"run_{uuid.uuid4().hex[:8]}")
    os.makedirs(session_results_dir, exist_ok=True)

    for turn in range(max_turns):
        log(f"── Turn {turn + 1}/{max_turns} ──")

        # Build the full prompt from history
        if turn == 0:
            prompt_text = context
        else:
            prompt_text = context + "\n\n" + "\n\n".join(
                f"## {m['role'].upper()}\n{m['content']}"
                for m in history
            )
        prompt_text += "\n\n## ASSISTANT\n"

        log(f"  Prompt: {len(prompt_text)} chars")

        # Spawn engine in background
        session = spawn_engine(
            model_path=model_path,
            prompt=prompt_text,
            max_tokens=max_tokens_per_turn,
            engine_path=engine_path,
            session_id=f"t{turn}_{uuid.uuid4().hex[:8]}",
        )
        
        expected_tok = session["max_tokens"]
        estimated = expected_tok * 12 + 30  # ~12s/tok + model load
        log(f"  Spawned PID {session['pid']}, ~{estimated:.0f}s expected ({expected_tok} tok @~12s/tok)")
        log(f"  Polling {session['result_file']}")
        log(f"  [poll_interval={poll_seconds}s]")

        # Poll until done (blocking)
        result = poll_engine_result(session, poll_interval=poll_seconds)

        elapsed = result.get("elapsed", 0)
        gen_tokens = result.get("gen_tokens", 0)
        rate = f"{elapsed/gen_tokens:.1f}" if gen_tokens > 0 else "?"
        log(f"  Engine: {elapsed:.1f}s, {gen_tokens} tok ({rate} s/tok)")

        output_text = result.get("output_text", "")

        if verbose and output_text:
            log(f"  Generated text ({len(output_text)} chars):")
            log(f"    {output_text[:1000]}")
            if len(output_text) > 1000:
                log(f"    ... ({len(output_text) - 1000} more chars)")

        # Parse tool calls
        tool_calls = parse_tool_calls(output_text)

        if not tool_calls:
            log(f"  No tool calls — done.")
            history.append({"role": "assistant", "content": output_text})
            cleanup_session(session)
            return output_text, history

        if verbose:
            log(f"  Tool calls: {len(tool_calls)}")
            for name, args, _ in tool_calls:
                if name.startswith("__"):
                    log(f"    ⚠️  {name}: {args.get('raw', '')[:200]}")
                else:
                    log(f"    → {name}({json.dumps({k: str(v)[:60] for k, v in args.items()})})")

        # Add assistant output to history
        history.append({"role": "assistant", "content": output_text})

        # Execute tools and collect results
        results_block_parts = []
        for name, args, raw in tool_calls:
            if name.startswith("__"):
                results_block_parts.append(f"Error: could not parse tool call: {raw[:200]}")
                continue
            log(f"  Executing {name}...", end="", flush=True)
            t1 = time.time()
            res = execute_tool(name, args)
            tool_elapsed = time.time() - t1
            log(f" done ({tool_elapsed:.1f}s)")
            results_block_parts.append(f"**Tool: {name}**\n```\n{res[:3000]}\n```")

        results_block = "\n\n".join(results_block_parts)
        history.append({"role": "tool_results", "content": results_block})
        cleanup_session(session)

    log(f"Max turns ({max_turns}) reached.")
    return "", history


# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Slow-Inference Agent Harness — Pi-style tool-calling loop"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to GGUF model file")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max tool-calling turns (default: 10)")
    parser.add_argument("--tokens", type=int, default=1024,
                        help="Max tokens per generation (default: 1024)")
    parser.add_argument("--poll", type=float, default=5.0,
                        help="Poll interval seconds (default: 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show turn-by-turn debug output to stderr")
    parser.add_argument("--json-output", action="store_true",
                        help="Emit structured JSON (answer, metadata, tool calls)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models in MODELS_DIR")
    parser.add_argument("--spawn-only", action="store_true",
                        help="Spawn first turn and print session ID only (for external polling)")
    parser.add_argument("prompt", type=str, nargs="?",
                        help="Initial prompt (or read from stdin if omitted)")
    args = parser.parse_args()

    if args.list_models:
        if os.path.isdir(MODELS_DIR):
            print("Available models:")
            for f in sorted(Path(MODELS_DIR).iterdir()):
                if f.is_file():
                    size = f.stat().st_size
                    print(f"  {f.name} ({size/1e9:.2f} GB)")
        else:
            print(f"No models directory at {MODELS_DIR}")
        return

    # Model selection
    model_path = args.model
    if not model_path:
        models = sorted(f for f in Path(MODELS_DIR).iterdir() if f.is_file())
        if models:
            model_path = str(models[0])
            if args.verbose:
                print(f"Auto-selected model: {Path(model_path).name}", file=sys.stderr)
        else:
            print("No model found. Use --model or place a GGUF in ~/slow-inference/models/", file=sys.stderr)
            sys.exit(1)

    # Prompt
    prompt = args.prompt
    if not prompt:
        if args.verbose:
            print("Enter your prompt (Ctrl+D to submit, Ctrl+C to quit):", file=sys.stderr)
        try:
            prompt = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print("\nAborted.", file=sys.stderr)
            sys.exit(1)
        if not prompt:
            print("No prompt provided.", file=sys.stderr)
            sys.exit(1)

    # Spawn-only mode: emit session JSON for external polling
    if args.spawn_only:
        system = build_system_prompt()
        prompt_text = f"{system}\n\n{prompt}\n\n## ASSISTANT\n"
        session = spawn_engine(
            model_path=model_path,
            prompt=prompt_text,
            max_tokens=args.tokens,
            session_id=f"ext_{uuid.uuid4().hex[:8]}",
        )
        # Print session info as JSON for external tools
        print(json.dumps({k: v for k, v in session.items() if k != "engine_path"}))
        return

    # Full agent loop
    output_text, history = agent_loop(
        model_path=model_path,
        initial_prompt=prompt,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.tokens,
        verbose=args.verbose,
        poll_seconds=args.poll,
    )

    clean = TOOL_CALL_RE.sub("", output_text).strip()

    if args.json_output:
        result = {
            "answer": clean,
            "model": Path(model_path).name,
            "turns_used": len([m for m in history if m["role"] == "assistant"]),
            "history": history,
        }
        print(json.dumps(result))
    else:
        # Plain text: just the answer, no banners
        print(clean)


if __name__ == "__main__":
    main()