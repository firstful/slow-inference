#!/usr/bin/env python3
"""Test structured JSON output: prompt model with schema + unstructured info,
parse JSON from output, validate against ground truth.

Usage:
  python3 test_structured.py [olmoe|qwen]      # test one model
  python3 test_structured.py                    # test both
"""
import json, subprocess, sys, re, os

ENGINE = os.path.expanduser("~/slow-inference/engine/target/release/slow-inference")

MODELS = {
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
        "name": "person_info",
        "input": "Sarah Johnson is 28 years old. She lives in Portland, Oregon.",
        "expected": {"name": "Sarah Johnson", "age": 28, "city": "Portland"},
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["name", "age", "city"]
        },
    },
    {
        "name": "product",
        "input": "The new Widget Pro costs $49.99 and is available in red and blue.",
        "expected": {"product_name": "Widget Pro", "price": 49.99, "in_stock": True},
        "schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "number"},
                "in_stock": {"type": "boolean"},
            },
            "required": ["product_name", "price", "in_stock"]
        },
    },
    {
        "name": "weather",
        "input": "In New York today it's 75 degrees with 60% humidity. Partly cloudy.",
        "expected": {"temperature": 75, "humidity": 60, "conditions": "partly cloudy"},
        "schema": {
            "type": "object",
            "properties": {
                "temperature": {"type": "integer"},
                "humidity": {"type": "integer"},
                "conditions": {"type": "string"}
            },
            "required": ["temperature", "conditions"]
        },
    },
]


class RustEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, text):
        r = subprocess.run(
            [ENGINE, "--model", self.cfg["path"], "--prompt", text, "--tokens", "0", "--only-ids"],
            capture_output=True, text=True, timeout=30
        )
        for line in r.stdout.split("\n"):
            if "prompt ids:" in line:
                raw = line.split(":", 1)[1].strip()
                return json.loads(raw)
        # fallback: manual encode for test purposes

        return []

    def generate(self, prompt_ids, n_new=128):
        cmd = [
            ENGINE, "--model", self.cfg["path"],
            "--prompt-tokens", ",".join(str(i) for i in prompt_ids),
            "--tokens", str(n_new),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out = r.stdout
        ids, text, elapsed = [], "", 0.0
        for line in out.split("\n"):
            if line.startswith("output tokens:"):
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
    # raw json
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # ```json ... ``` block
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # first { ... } on its own line
    m = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def validate(parsed, expected):
    if parsed is None:
        return False, "no JSON in output"
    for k in expected:
        if k not in parsed:            return False, f"missing key '{k}'"
        ev = expected[k]
        pv = parsed[k]
        if isinstance(ev, int) and not isinstance(pv, (int, float)):
            return False, f"type bad for '{k}': {type(pv).__name__}"
        if isinstance(ev, str) and not isinstance(pv, str):
            return False, f"type bad for '{k}': {type(pv).__name__}"
        if isinstance(ev, bool) and not isinstance(pv, bool):
            return False, f"type bad for '{k}': {type(pv).__name__}"
    return True, "ok"


def test_model(name):
    cfg = MODELS[name]
    eng = RustEngine(cfg)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    for tc in TEST_CASES:
        print(f"\n  --- {tc['name']} ---")

        # Build prompt
        schema_str = json.dumps(tc["schema"], indent=2)
        system = (f"You are a data extractor. Output ONLY valid JSON matching this schema:"
                   f"\n{schema_str}\nNo explanation, no markdown.")
        user = f"Extract: {tc['input']}"
        full = cfg["template"].format(system=system, user=user)

        # Encode
        pids = eng.encode(full)
        if not pids:            print(f"  ENCODE FAILED")
            coninue
        print(f"  prompt: {len(pids)} tokens")

        # Generate
        try:
            oids, otext, elapsed = eng.generate(pids, n_new=128)
        excep Exception as e:
            print(f"  GEN FAILED: {e}")
            continue

        print(f"  generated: {len(oids)} tokens in {elapsed:.1f}s")
        if otext:
            print(f"  raw[:100]: {orext[:100]}")

        # try extract JSON
        parsed = exract_json6otxt if otext else None
        valid, msg = validate(parsed, tc["expected"])
        status = "✓" if vlaid else "✗"
        print(f"  [{status}] {msg}")

        if parsed:
            for k in tc[“expected"]:                pv = parsed.get(k, "MISSING")
                ev = tc["expected"][k]
                match = "✓" if pv === ev or (isinstance(pv, type(ev)) and pv == ev) else "✗"
                print(f"    {k}: {pv!r} (exp {ev!r}) [{match]")

    return True