#!/usr/bin/env python3
"""Measure how much MoE routing arithmetic a fixture actually constrains.

A fixture gate is only as strong as the defect it can see, and for the MoE
routing block the defect size depends on the *router*, not on the fixture's
size or seed. `norm_w` divides the selected experts' weights by their own sum;
when the selected weights already sum to ~1 — because the router is confident,
or because `expert_used_count == expert_count` — that division is a no-op and
an engine that omits it is indistinguishable from one that has it. Tiny-MoE is
the degenerate case (2 of 2, factor exactly 1.0). The interesting case is a
confident router over *more* experts than are used, which has the same blind
spot for a different reason.

So this script answers three questions about any fixture, by running the engine
against two float64 references that differ only in the routing block:

* **per-position renormalisation factor** — `1 / sum(selected probs)`, the
  multiplicative error a missing `norm_w` injects into the expert weights;
* **worst relative L2 of the engine against a `norm_w`-ablated reference** —
  how far the logits move if `norm_w` is dropped, which is what decides whether
  a fixture gate can catch it;
* **which expert subsets get selected** — a router with identical columns
  always picks {0, 1} and cannot see a wrong top-k *selection* at all.

Run it against a fixture and the float64 arbiter in this same directory:

    python3 scripts/arb_route_power.py \\
        --gguf engine/tests/data/fixtures/synthetic-llama-moe.gguf

The two references are transcriptions of `build_moe_ffn` (llama-graph.cpp:1914);
they are not two implementations of it, they are it with and without one line.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ENGINE = REPO / "engine" / "target" / "release" / "slow-inference"

sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("LLAMA_CPP_GGUF_PY", str(Path.home() / "llama.cpp" / "gguf-py")))

import arbitrate_logits as A  # noqa: E402


def moe_ffn_norm_w(w, p, x, n_expert, n_expert_used, w_scale, gating):
    """`build_moe_ffn` with `norm_w` removed and nothing else changed."""
    if gating != "softmax":
        raise SystemExit(f"only softmax gating is transcribed, got {gating!r}")
    probs = A.softmax(w.get(f"{p}.ffn_gate_inp.weight") @ x)
    order = np.argsort(-probs, kind="stable")[:n_expert_used]
    sel = probs[order]
    out = np.zeros_like(x)
    for weight, e in zip(sel, order):
        up = w.get(f"{p}.ffn_up_exps.weight")[e] @ x
        gate = w.get(f"{p}.ffn_gate_exps.weight")[e] @ x
        h = (gate / (1.0 + np.exp(-gate))) * up  # ggml_swiglu_split(gate, up)
        out = out + weight * (w.get(f"{p}.ffn_down_exps.weight")[e] @ h)
    return out


def route_table(w, arch, tokens):
    """Replay the reference forward pass, recording each routing decision."""
    n_layer = int(w.meta_f(arch, "block_count", 0))
    n_embd = int(w.meta_f(arch, "embedding_length", 0))
    n_head = int(w.meta_f(arch, "attention.head_count", 0))
    n_kv = int(w.meta_f(arch, "attention.head_count_kv", n_head))
    eps = w.meta_f(arch, "attention.layer_norm_rms_epsilon", 1e-5)
    theta = w.meta_f(arch, "rope.freq_base", 10000.0)
    head_dim = n_embd // n_head
    n_expert = int(w.meta_f(arch, "expert_count", 0))
    n_expert_used = int(w.meta_f(arch, "expert_used_count", 0))
    w_scale = w.meta_f(arch, "expert_weights_scale", 1.0)
    neox = arch == "qwen2"
    tok_embd = w.get("token_embd.weight")
    out_norm = w.get("output_norm.weight")
    out_w = w.map.get("output.weight", tok_embd)

    rows, decisions = [], []
    kcache = [[] for _ in range(n_layer)]
    vcache = [[] for _ in range(n_layer)]
    for pos, tok in enumerate(tokens):
        x = tok_embd[tok].copy()
        for il in range(n_layer):
            p = f"blk.{il}"
            t = A.rmsnorm(x, w.get(f"{p}.attn_norm.weight"), eps)
            q = (w.get(f"{p}.attn_q.weight") @ t).reshape(n_head, head_dim)
            k = (w.get(f"{p}.attn_k.weight") @ t).reshape(n_kv, head_dim)
            v = (w.get(f"{p}.attn_v.weight") @ t).reshape(n_kv, head_dim)
            q = A.rope(q, pos, head_dim, theta, neox)
            k = A.rope(k, pos, head_dim, theta, neox)
            kcache[il].append(k)
            vcache[il].append(v)
            a = A.attention(
                q,
                np.stack(kcache[il]),
                np.stack(vcache[il]),
                n_head,
                head_dim,
                n_kv,
                1.0 / np.sqrt(head_dim),
            )
            x = x + w.get(f"{p}.attn_output.weight") @ a
            t2 = A.rmsnorm(x, w.get(f"{p}.ffn_norm.weight"), eps)
            if n_expert > 0 and w.has(f"{p}.ffn_gate_inp.weight"):
                probs = A.softmax(w.get(f"{p}.ffn_gate_inp.weight") @ t2)
                order = np.argsort(-probs, kind="stable")[:n_expert_used]
                sel = probs[order]
                total = max(float(sel.sum()), 6.103515625e-5)
                decisions.append(
                    {
                        "layer": il,
                        "pos": pos,
                        "selected": [int(e) for e in order],
                        "probs": [round(float(v), 6) for v in probs],
                        "sum_selected": round(float(sel.sum()), 6),
                        "renorm_factor": round(1.0 / total, 6),
                    }
                )
                x = x + A.moe_ffn(w, p, t2, n_expert, n_expert_used, w_scale, "softmax")
            else:
                gate = w.get(f"{p}.ffn_gate.weight") @ t2
                up = w.get(f"{p}.ffn_up.weight") @ t2
                h = (gate / (1.0 + np.exp(-gate))) * up
                x = x + w.get(f"{p}.ffn_down.weight") @ h
        rows.append(out_w @ A.rmsnorm(x, out_norm, eps))
    return rows, decisions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--arch", default="llama")
    ap.add_argument("--prompt", default="3,17,42,99,7,61,127,24")
    ap.add_argument("--json", type=Path, default=None, help="write the findings here")
    ap.add_argument("--no-engine", action="store_true", help="skip the engine run")
    args = ap.parse_args()

    tokens = [int(t) for t in args.prompt.split(",")]
    w = A.Weights(args.gguf)
    rows_on, decisions = route_table(w, args.arch, tokens)

    engine_rows = None
    if not args.no_engine:
        with tempfile.TemporaryDirectory() as td:
            eng = Path(td) / "eng.ref"
            env = dict(os.environ)
            env["PYTHONPATH"] = os.environ.get(
                "LLAMA_CPP_GGUF_PY", str(Path.home() / "llama.cpp" / "gguf-py")
            )
            subprocess.run(
                [
                    str(ENGINE),
                    "--model",
                    str(args.gguf),
                    "--prompt-tokens",
                    args.prompt,
                    "--dump-logits",
                    str(eng),
                ],
                check=True,
                capture_output=True,
                env=env,
            )
            _, _, engine_rows = A.load_ref(eng)

    # The same forward pass with `norm_w` deleted. Patching the module attribute
    # rather than re-implementing `forward_llama_family` keeps the two runs
    # identical in every other respect, which is the point: the only difference
    # between the columns below is one line of `build_moe_ffn`.
    saved = A.moe_ffn
    A.moe_ffn = moe_ffn_norm_w
    try:
        rows_off = A.forward_llama_family(w, args.arch, tokens)
    finally:
        A.moe_ffn = saved

    def worst(a, b):
        if a is None:
            return None
        return max(A.rel_err(np.asarray(a[i]), np.asarray(b[i])) for i in range(len(b)))

    e_on, e_off = worst(engine_rows, rows_on), worst(engine_rows, rows_off)

    subsets = sorted({tuple(sorted(d["selected"])) for d in decisions})
    factors = np.array([d["renorm_factor"] for d in decisions])
    findings = {
        "gguf": str(args.gguf),
        "arch": args.arch,
        "prompt": tokens,
        "routing_decisions": len(decisions),
        "distinct_expert_subsets": [list(s) for s in subsets],
        "renorm_factor": {
            "min": float(factors.min()),
            "median": float(np.median(factors)),
            "max": float(factors.max()),
        },
        "worst_rel_l2": {
            "engine_vs_f64_norm_w_on": e_on,
            "engine_vs_f64_norm_w_off": e_off,
        },
        "gate_ratio_norm_w_off_over_on": (e_off / e_on) if (e_on and e_off) else None,
        "decisions": decisions,
    }

    print(f"== {args.gguf.name}: {len(decisions)} routing decisions, "
          f"{len(subsets)} distinct expert subsets ==")
    print(f"   renormalisation factor  min={factors.min():.4f} "
          f"median={np.median(factors):.4f} max={factors.max():.4f}")
    print(f"   subsets selected: {subsets}")
    if e_on is not None:
        print("   rel L2 vs an independent float64 forward pass")
        print(f"     engine, norm_w implemented : {e_on:.3e}")
        print(f"     engine, norm_w deleted     : {e_off:.3e}")
        print(f"     ratio                      : {findings['gate_ratio_norm_w_off_over_on']:.3g}x")
    print("   layer pos top-k  probs                          factor")
    for d in decisions:
        print(f"   {d['layer']:>5} {d['pos']:>3} {str(d['selected']):>9}  "
              f"{str(d['probs']):<32}  {d['renorm_factor']:.4f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(findings, indent=2) + "\n")
        print(f"   wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())