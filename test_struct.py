#!/usr/bin/env python3
"""Structured output test harness. Encodes prompt in Python (fast via
tiktoken/transformers), saves prompt ids, runs Rust engine via subprocess
with file output, parses and validates results.

Usage: python3 test_struct.py [qwen|olmoe|all]
"""
import json, subprocess, sys, re, os, time, tempfile

sys.path.insert(0, os.path.expanduser("~/slow-inference/py-reference/src"))
import gguf as g

ENGINE = os.path.expanduser("~/slow-inference/engine/target/release/slow-inference")

def make_encoder(path):
    """Return a function that encodes text -> list of token ids for this model."""
    gf = g.parse(path)
    model_type = gf.metadata.get("tokenizer.ggml.model", "")
    # Use transformers for exact encoding match
    if "qwen" in path.lower():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
        return tok.encode
    elif "olmoe" in path.lower() or "llama" in path.lower():
        from transformers import AutoTokenizer
        tokenizer_name = "allenai/OLMoE-1B-7B-0924-Instruct"
        try:
            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            return tok.encode
        except Exception:
            pass
    # Fallback: use the engine itself (slow but works)
    def engine_encode(text):
        r = subprocess.run(
            [ENGINE, "--model", path, "--prompt", text, "--tokens", "0"],
            capture_output=True, text=True, timeout=900
        )
        for line in r.stdout.split("\n"):
            if "prompt ids:" in line:
                return json.loads(line.split(":", 1)[1].strip())
        return []
    return engine_encode


MODELS = {
    "qwen": {
        "path": os.path.expanduser("~/slow-inference/models/qwen2.5-3b-instruct-q4_k_m.gguf"),
        "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
    },
    "olmoe": {
        "path": os.path.expanduser("~/slow-inference/models/OLMoE-1B-7B-Instruct-Q4_K_M.gguf"),
        "template": "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n",
    },
}

CASES = [
    {
        "name": "person",
        "system": "Output JSON with: name (string), age (integer), city (string). No explanation.",
        "user": "Sarah Johnson is 28 years old. She lives in Portland, Oregon.",
        "expected": {"name": "Sarah Johnson", "age": 28, "city": "Portland"},
    },
    {
        "name": "product",
        "system": "Output JSON with: product_name (string), price (number), in_stock (boolean). No explanation.",
        "user": "The Widget Pro costs $49.99 and is in stock.",
        "expected": {"product_name": "Widget Pro", "price": 49.99, "in_stock": True},
    },
    {
        "name": "weather",
        "system": "Output JSON with: temperature (integer), humidity (integer), conditions (string). No explanation.",
        "user": "In New York it's 75F with 60% humidity, partly cloudy.",
        "expected": {"temperature": 75, "humidity": 60, "conditions": "partly cloudy"},
    },
]


def extract_json(text):
    if not text: return None
    text = text.strip()
    if text.startswith("{"):
        try: return json.loads(text)
        except: pass
    m = re.search(r"\{[^}]*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    return None


def run(name, path, template, encode, system, user, n_new=48):
    """Run one test. Returns (passed, detail_dict)."""
    prompt = template.replace("{system}", system).replace("{user}", user)
    pids = encode(prompt)
    if not pids:
        return False, {"error": "encode_failed", "prompt": prompt[:100]}

    # Save to temp file for engine
    outfile = f"/tmp/struct_{name}_{int(time.time())}.txt"
    pids_str = ",".join(str(i) for i in pids)
    cmd = [ENGINE, "--model", path, "--prompt-tokens", pids_str,
           "--tokens", str(n_new)]

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    elapsed = time.time() - t0

    # Parse output
    ids, text = [], ""
    for line in r.stdout.split("\n"):
        if line.startswith("output tokens:"):
            raw = line.split("[", 1)[1].split("]")[0]
            ids = [int(i.strip()) for i in raw.split(",") if i.strip()]
        elif line.startswith("output text:"):
            text = line.split(":", 1)[1].strip()

    parsed = extract_json(text) if text else None

    # Validate
    issues = []
    if parsed is None:
        issues.append("no_json")
    else:
        for k, ev in expected.items():
            if k not in parsed:
                issues.append(f"missing_{k}")
            elif type(ev) == bool and type(parsed[k]) != bool:
                issues.append(f"type_{k}")
            elif type(parsed[k]) != type(ev) and not (isinstance(ev, (int, float)) and isinstance(parsed[k], (int, float))):
                issues.append(f"type_{k}_{type(parsed[k]).__name__}")
            elif parsed[k] != ev:
                issues.append(f"val_{k}:{parsed[k]!r}")

    passed = len(issues) == 0
    return passed, {
        "prefill_tok": len(pids),
        "gen_tok": len(ids),
        "elapsed_s": round(elapsed, 1),
        "issues": ";".join(issues) if issues else "ok",
        "parsed": parsed,
        "raw_preview": (text[:150] if text else "(empty)"),
    }


def main():
    targets = [x for x in sys.argv[1:] if x != "all"] or list(MODELS.keys())
    results = []
    for model_name in targets:
        if model_name not in MODELS:
            print(f"Unknown: {model_name}")
            continue
        cfg = MODELS[model_name]
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        print("  Creating encoder...", end=" ", flush=True)
        encode = make_encoder(cfg["path"])
        print("done")

        for case in CASES:
            name = case["name"]
            print(f"  [{name}] running...", end=" ", flush=True)
            ok, info = run(model_name, cfg["path"], cfg["template"],
                          encode, case["system"], case["user"], n_new=48)
            status = "OK" if ok else "FAIL"
            print(f"{status}  ({info['prefill_tok']}+{info['gen_tok']}t, {info['elapsed_s']}s)")
            if info.get("parsed"):
                print(f"    => {json.dumps(info['parsed'], ensure_ascii=False)}")
            elif info.get("issues") != "ok":
                print(f"    [{info['issues']}] raw={info['raw_preview'][:100]}")
            results.append((model_name, name, status, info))

    # Table
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"{'Model':<8} {'Case':<10} {'Status':<6} {'Pre':<5} {'Gen':<4} {'Time':<8} Detail")
    n = len(results) + 1
    print("-" * 50)
    for r in results:
        i = r[3]
        print(f"{r[0]:<8} {r[1]:<10} {r[2]:<6} {i['prefill_tok']:<5} {i['gen_tok']:<4} {i['elapsed_s']:<8} {i['issues']}")


if __name__ == "__main__":
    main()