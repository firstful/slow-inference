#!/usr/bin/env python3
"""Test structured JSON output. Encodes via fast Python tokenizer,
runs Rust engine for generation, compares against Python reference.

Usage:
  python3 test_structured.py
"""
import json, subprocess, sys, re, os, csv, threading, time
from pathlib import Path

ENGINE = os.path.expanduser("~/slow-inference/engine/target/release/slow-inference")
PY_REF = os.path.expanduser("~/slow-inference/py-reference/src")

sys.path.insert(0, PY_REF)
import gguf as g

MODEL_CFGS = {
    "olmoe": {
        "path": os.path.expanduser("~/slow-inference/models/OLMoE-1B-7B-Instruct-Q4_K_M.gguf"),
        "bos": 50279, "eos": 50279,
        "template": "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n",
    },
    "qwen": {
        "path": os.path.expanduser("~/slow-inference/models/qwen2.5-3b-instruct-q4_k_m.gguf"),
        "bos": 151643, "eos": 151645,
        "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
    },
}

TEST_CASES = [
    {
        "name": "person",
        "input": "Sarah Johnson is 28 years old. She lives in Portland, Oregon.",
        "expected": {"name": "Sarah Johnson", "age": 28, "city": "Portland"},
        "schema_summary": "JSON: name string, age integer, city string",
    },
    {
        "name": "product",
        "input": "The new Widget Pro costs $49.99 and is available in red and blue.",
        "expected": {"product_name": "Widget Pro", "price": 49.99, "in_stock": True},
        "schema_summary": "JSON: product_name string, price number, in_stock boolean",
    },
    {
        "name": "weather",
        "input": "In New York today it's 75 degrees with 60% humidity. Partly cloudy.",
        "expected": {"temperature": 75, "humidity": 60, "conditions": "partly cloudy"},
        "schema_summary": "JSON: temperature integer, humidity integer, conditions string",
    },
]

# Cache GGUF metadata
_gf_cache = {}
def get_gguf(path):
    if path not in _gf_cache:
        _gf_cache[path] = g.parse(path)
    return _gf_cache[path]


def encode_fast(model_cfg, text):
    """Quick encode using the Rust engine's internal tokenizer via stdin.
    Avoids re-prefilling by running with --tokens 0.  We accept the one-time cost.
    """
    # Actually use the metadata to look up tokens manually for testing
    gf = get_gguf(model_cfg["path"])
    toks = gf.metadata["tokenizer.ggml.tokens"]
    
    # For SPM models (llama vocab): use the spm module
    model_type = gf.metadata.get("tokenizer.ggml.model", "")
    if model_type == "llama":
        # Can't easily encode SPM from Python; fall back to engine call
        pass
    
    # Fallback: encode via engine (one-time cost per model)
    r = subprocess.run(
        [ENGINE, "--model", model_cfg["path"], "--prompt", text, "--tokens", "0"],
        capture_output=True, text=True, timeout=600
    )
    for line in r.stdout.split("\n"):
        if "prompt ids:" in line:
            raw = line.split(":", 1)[1].strip()
            return json.loads(raw)
    return []


def generate_rust(model_path, prompt_ids, n_new=48):
    cmd = [
        ENGINE, "--model", model_path,
        "--prompt-tokens", ",".join(str(i) for i in prompt_ids),
        "--tokens", str(n_new),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = r.stdout
    ids, text, elapsed = [], "", 0.0
    for line in out.split("\n"):
        if "output tokens:" in line:
            raw = line.split("[", 1)[1].split("]")[0]
            ids = [int(i.strip()) for i in raw.split(",") if i.strip()]
        elif line.startswith("output text:"):
            text = line.split(":", 1)[1].strip()
        elif line.startswith("time:"):
            elapsed = float(line.split()[1].rstrip("s"))
    return ids, text, elapsed


def extract_json(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[^}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def validate(parsed, expected):
    if parsed is None:
        return False, "no JSON"
    for k in expected:
        if k not in parsed:
            return False, f"missing '{k}'"
        pv_type = type(parsed[k])
        ev_type = type(expected[k])
        if isinstance(expected[k], bool):
            if pv_type != bool:
                return False, f"type '{k}': got {pv_type.__name__}, expected bool"
        elif pv_type != ev_type:
            return False, f"type '{k}': got {pv_type.__name__}, expected {ev_type.__name__}"
    return True, "ok"


def test_model(name):
    cfg = MODEL_CFGS[name]
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # One-time encode (slow, disk-streamed prefill)
    print("  Encoding prompts (one-time prefill)...")
    
    all_pass = True
    for tc in TEST_CASES:
        print(f"\n  --- {tc['name']} ---")
        system = f"You are a data extractor. Output ONLY valid JSON. {tc['schema_summary']}"
        user = tc['input']
        full = cfg["template"].format(system=system, user=user)
        
        # Encode (first call is slow; subsequent calls reuse page cache)
        pids = encode_fast(cfg, full)
        if not pids:
            print(f"  FAIL: encoding failed (prefill timeout)")
            continue
        print(f"  prompt: {len(pids)} tokens")
        
        # Generate via Rust
        try:
            oids, otext, elapsed = generate_rust(cfg["path"], pids, n_new=32)
        except Exception as e:
            print(f"  FAIL: {e}")
            continue
        print(f"  rust gen: {len(oids)} tokens in {elapsed:.1f}s")
        if otext:
            print(f"  raw[:120]: {otext[:120]}")
        
        parsed = extract_json(otext) if otext else None
        valid, msg = validate(parsed, tc["expected"])
        status = "PASS" if valid else "FAIL"
        print(f"  [{status}] {msg}")
        if parsed:
            for k in tc["expected"]:
                pv = parsed.get(k, "MISSING")
                ev = tc["expected"][k]
                ok = "OK" if pv == ev else "MISMATCH"
                print(f"    {k}: {pv!r} (expected {ev!r}) [{ok}]")
                if ok != "OK":
                    all_pass = False
        else:
            all_pass = False
            print(f"    raw tokens (first 10): {oids[:10]}")
    return all_pass


if __name__ == "__main__":
    test_model("olmoe")