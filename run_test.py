#!/usr/bin/env python3
"""Run structured output tests with file-based output (no pipe buffering).
Usage: python3 run_test.py [olmoe|qwen|all]

Encodes via Python (fast), runs engine to a temp file, then parses.
"""
import json, subprocess, sys, re, os, time, tempfile, glob

os.environ["PYTHONUNBUFFERED"] = "1"
ENGINE = os.path.expanduser("~/slow-inference/engine/target/release/slow-inference")
MODEL_DIR = os.path.expanduser("~/slow-inference/models")
sys.path.insert(0, os.path.expanduser("~/slow-inference/py-reference/src"))

MODELS = {}
DEFAULT_NEW = 80

# --- lazy encoders ---
def make_qwen_encoder():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
    return lambda s: tok.encode(s)

def make_spm_encoder(name):
    """For SPM models: fallback to engine encode (slow but correct)."""
    path = os.path.join(MODEL_DIR, name)
    def engine_encode(text):
        out = tempfile.mktemp(suffix=".txt")
        r = subprocess.run(
            [ENGINE, "--model", path, "--prompt", text, "--tokens", "0"],
            capture_output=True, text=True, timeout=900
        )
        for line in r.stdout.split("\n"):
            if "prompt ids:" in line:
                return json.loads(line.split(":", 1)[1].strip())
        raise RuntimeError("encode failed")
    return engine_encode

for f in os.listdir(MODEL_DIR):
    if "olmoe" in f.lower():
        MODELS["olmoe"] = {
            "path": os.path.join(MODEL_DIR, f),
            "template": "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n",
            "encode": make_spm_encoder(f),
        }
    elif "qwen2.5" in f.lower():
        MODELS["qwen"] = {
            "path": os.path.join(MODEL_DIR, f),
            "template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
            "encode": make_qwen_encoder(),
        }

CASES = [
    ("person", "Output JSON with: name (string), age (integer), city (string). No explanation.",
     "Sarah Johnson is 28. Lives in Portland.",
     {"name": "Sarah Johnson", "age": 28, "city": "Portland"}),
    ("product", "Output JSON with: product_name (string), price (number), in_stock (boolean). No explanation.",
     "Widget Pro costs 49.99, in stock.",
     {"product_name": "Widget Pro", "price": 49.99, "in_stock": True}),
    ("weather", "Output JSON with: temperature (integer), humidity (integer), conditions (string). No explanation.",
     "New York: 75F, 60% humidity, partly cloudy.",
     {"temperature": 75, "humidity": 60, "conditions": "partly cloudy"}),
]

def extract_json(text):
    if not text: return None
    # Handle literal \n escape sequences (from engine output)
    text = text.replace("\\n", "\n").replace("\\\"", "\"")
    text = text.strip()
    if text.startswith("{"):
        try: return json.loads(text)
        except: pass
    for p in [r"```json\s*(.*?)\s*```", r"\{[^}]*\}"]:
        m = re.search(p, text, re.DOTALL)
        if m:
            try: return json.loads(m.group(1 if "json" in p else 0))
            except: pass
    return None

def run_one(model_cfg, system, user, expected, n_new=None):
    if n_new is None: n_new = DEFAULT_NEW
    prompt = model_cfg["template"].replace("{system}", system).replace("{user}", user)

    # Encode (Python fast path or engine fallback)
    t0 = time.time()
    pids = model_cfg["encode"](prompt)
    if not pids:
        return {"ok": False, "error": "encode_failed"}
    enc_time = time.time() - t0

    # Write prompt ids to a temp file for the engine
    pids_str = ",".join(str(i) for i in pids)

    # Run engine
    t0 = time.time()
    cmd = f'"{ENGINE}" --model "{model_cfg["path"]}" --prompt-tokens "{pids_str}" --tokens {n_new}'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    out_text = r.stdout

    # Parse output
    ids, text = [], ""
    for line in out_text.split("\n"):
        if line.startswith("output tokens:"):
            raw = line.split("[", 1)[1].split("]")[0]
            ids = [int(i.strip()) for i in raw.split(",") if i.strip()]
        elif line.startswith("output text:"):
            text = line.split(":", 1)[1].strip()

    parsed = extract_json(text) if text else None
    per_token = elapsed / max(len(ids), 1)

    # Validate
    issues = []
    if parsed is None:
        issues.append("no_json")
    else:
        for k, ev in expected.items():
            if k not in parsed:
                issues.append(f"missing_{k}")
            else:
                pv = parsed[k]
                if isinstance(ev, bool) and type(pv) != bool:
                    issues.append(f"type_{k}={type(pv).__name__}")
                elif isinstance(ev, str) and type(pv) != str:
                    issues.append(f"type_{k}={type(pv).__name__}")
                elif not isinstance(pv, type(ev)):
                    if isinstance(ev, (int, float)) and isinstance(pv, (int, float)):
                        pass  # int/number compatible
                    else:
                        issues.append(f"type_{k}={type(pv).__name__}")
                elif pv != ev:
                    issues.append(f"val_{k}={pv!r}")

    return {
        "ok": len(issues) == 0,
        "issues": ";".join(issues) if issues else "ok",
        "prefill_tok": len(pids),
        "gen_tok": len(ids),
        "elapsed_s": round(elapsed, 1),
        "per_token_s": round(per_token, 1),
        "encode_s": round(enc_time, 1),
        "parsed": parsed,
        "raw_preview": text[:150] if text else "(empty)",
        "top1_id": ids[0] if ids else None,
    }


def main():
    targets = [x for x in sys.argv[1:] if x != "all"] or list(MODELS.keys())
    results = []
    for model_name in targets:
        if model_name not in MODELS:
            print(f"Unknown: {model_name}. Available: {list(MODELS.keys())}")
            continue
        cfg = MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"  {model_name} ({os.path.basename(cfg['path'])})")
        print(f"{'='*60}")
        for name, system, user, expected in CASES:
            print(f"\n  [{name}] ", end="", flush=True)
            info = run_one(cfg, system, user, expected)
            status = "OK" if info["ok"] else "FAIL"
            pfx = info.get("encode_s", 0)
            print(f"{status} — {info['prefill_tok']}+{info['gen_tok']} tok, "
                  f"{info['elapsed_s']}s gen ({info.get('per_token_s','?')}s/tok), "
                  f"encode {info.get('encode_s','?')}s")
            if info.get("parsed"):
                print(f"    => {json.dumps(info['parsed'], ensure_ascii=False)}")
            elif info.get("issues") != "ok":
                print(f"    [{info['issues']}]")
                if info["raw_preview"]:
                    print(f"    raw: {info['raw_preview'][:200]}")
            results.append((model_name, name, status, info))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(f"{'Model':<8} {'Case':<10} {'S':<4} {'Pre':<5} {'Gen':<5} {'Time':<8} {'Tok/s':<6} {'Issues'}")
    print("-"*65)
    for r in results:
        i = r[3]
        tokps = f"{i['per_token_s']}s" if i.get('per_token_s') else "?"
        print(f"{r[0]:<8} {r[1]:<10} {r[2]:<4} {i['prefill_tok']:<5} {i['gen_tok']:<5} "
              f"{i['elapsed_s']:<8} {tokps:<6} {i['issues']}")


if __name__ == "__main__":
    main()