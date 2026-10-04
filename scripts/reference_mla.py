#!/usr/bin/env python3
"""Independent reference forward pass for the synthetic DeepSeek-V2 MLA fixture."""

from __future__ import annotations

import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO / "models" / "synthetic-deepseek2-mla.gguf"
TOL = 1e-4


def import_gguf() -> object:
    try:
        import gguf  # type: ignore
    except ModuleNotFoundError:
        candidate = Path(
            os.environ.get(
                "LLAMA_CPP_GGUF_PY",
                Path.home() / "llama.cpp" / "gguf-py",
            )
        )
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
        try:
            import gguf  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "gguf-py is required. Set LLAMA_CPP_GGUF_PY or PYTHONPATH=~/llama.cpp/gguf-py."
            ) from exc
    return gguf


def f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def f32_dot(a, b) -> float:
    acc = 0.0
    for x, y in zip(a, b):
        acc = f32(acc + f32(f32(x) * f32(y)))
    return acc


def rmsnorm(x, w, eps: float) -> np.ndarray:
    ss = 0.0
    for v in x:
        ss = f32(ss + f32(f32(v) * f32(v)))
    ss = f32(ss / len(x))
    inv = f32(1.0 / math.sqrt(ss + eps))
    out = np.empty(len(x), dtype=np.float64)
    for i, (v, wi) in enumerate(zip(x, w)):
        out[i] = f32(f32(f32(v) * inv) * f32(wi))
    return out


def rope_inplace(x: np.ndarray, pos: int, theta: float) -> None:
    d = len(x) // 2
    for i in range(d):
        freq = theta ** (-2.0 * i / (2.0 * d))
        ang = float(pos) * freq
        c = f32(math.cos(ang))
        s = f32(math.sin(ang))
        a = f32(x[2 * i])
        b = f32(x[2 * i + 1])
        x[2 * i] = f32(f32(a * c) - f32(b * s))
        x[2 * i + 1] = f32(f32(a * s) + f32(b * c))


def softmax_f32(scores: np.ndarray) -> np.ndarray:
    m = float(max(scores))
    exps = np.empty(len(scores), dtype=np.float64)
    total = 0.0
    for i, v in enumerate(scores):
        e = f32(math.exp(f32(f32(v) - m)))
        exps[i] = e
        total = f32(total + e)
    out = np.empty(len(scores), dtype=np.float64)
    for i in range(len(scores)):
        out[i] = f32(f32(exps[i]) / total)
    return out


def argmax_first(v: np.ndarray) -> int:
    best = 0
    for i in range(len(v)):
        if float(v[i]) > float(v[best]):
            best = i
    return best


class RefMla:
    def __init__(self, gguf_mod, path: Path):
        reader = gguf_mod.GGUFReader(str(path))
        md = {}
        for k, f in reader.fields.items():
            if hasattr(f, 'contents'):
                md[k] = f.contents()
        arch = md.get("general.architecture")
        if arch != "deepseek2":
            raise SystemExit(f"expected deepseek2 arch, got {arch!r}")

        def get_u(k):
            v = md.get(k)
            return int(v) if v is not None else None

        def get_f(k, default=None):
            v = md.get(k)
            return float(v) if v is not None else default

        self.meta = {
            "n_layers": get_u("deepseek2.block_count"),
            "n_embd": get_u("deepseek2.embedding_length"),
            "n_ff": get_u("deepseek2.feed_forward_length"),
            "n_heads": get_u("deepseek2.attention.head_count"),
            "n_lora_kv": get_u("deepseek2.attention.kv_lora_rank"),
            "n_lora_q": get_u("deepseek2.attention.q_lora_rank"),
            "key_length_mla": get_u("deepseek2.attention.key_length_mla"),
            "value_length_mla": get_u("deepseek2.attention.value_length_mla"),
            "rope_dim": get_u("deepseek2.rope.dimension_count"),
            "rope_theta": get_f("deepseek2.rope.freq_base", 10000.0),
            "eps": get_f("deepseek2.attention.layer_norm_rms_epsilon", 1e-5),
            "n_experts": get_u("deepseek2.expert_count") or 0,
            "leading_dense": get_u("deepseek2.leading_dense_block_count") or 0,
            "rope_scaling_factor": get_f("deepseek2.rope.scaling.factor", 1.0),
            "rope_attn_factor": get_f("deepseek2.rope.scaling.attn_factor", 1.0),
            "yarn_log_mul": get_f("deepseek2.rope.scaling.yarn_log_multiplier", 0.0),
        }

        def tensor(name: str) -> np.ndarray:
            for t in reader.tensors:
                if t.name == name:
                    return np.asarray(t.data, dtype=np.float64)
            raise KeyError(name)

        def row(name: str) -> np.ndarray:
            return tensor(name).ravel()

        # weights
        self.token_embd = tensor("token_embd.weight")
        self.output_w = tensor("output.weight")
        self.output_norm = row("output_norm.weight")
        self.attn_norm = row("blk.0.attn_norm.weight")
        self.q_a_norm = row("blk.0.attn_q_a_norm.weight")
        self.q_a = tensor("blk.0.attn_q_a.weight")
        self.q_b = tensor("blk.0.attn_q_b.weight")
        self.kv_a_mqa = tensor("blk.0.attn_kv_a_mqa.weight")
        self.kv_a_norm = row("blk.0.attn_kv_a_norm.weight")
        self.k_b = tensor("blk.0.attn_k_b.weight")
        self.v_b = tensor("blk.0.attn_v_b.weight")
        self.attn_output = tensor("blk.0.attn_output.weight")
        self.ffn_norm = row("blk.0.ffn_norm.weight")
        self.ffn_up = tensor("blk.0.ffn_up.weight")
        self.ffn_gate = tensor("blk.0.ffn_gate.weight")
        self.ffn_down = tensor("blk.0.ffn_down.weight")

        m = self.meta
        assert m["n_embd"] == 8, m
        assert m["n_heads"] == 2, m
        assert m["n_lora_q"] == 4, m
        assert m["n_lora_kv"] == 2, m
        assert m["key_length_mla"] == 4, m
        assert m["value_length_mla"] == 4, m
        assert m["rope_dim"] == 2, m
        assert m["leading_dense"] == 1, m
        assert m["n_experts"] == 0, m

    def kq_scale(self) -> float:
        m = self.meta
        log_inv = math.log(1.0 / m["rope_scaling_factor"])
        attn_factor_org = m["rope_attn_factor"] * (1.0 + 0.1 * log_inv)
        mscale = attn_factor_org * (1.0 + 0.1 * m["yarn_log_mul"] * log_inv)
        return f32(f32(mscale * mscale) / math.sqrt(m["key_length_mla"]))

    def forward(self, token: int, pos: int, cache: np.ndarray):
        m = self.meta
        n_embd = m["n_embd"]
        n_heads = m["n_heads"]
        n_lora_kv = m["n_lora_kv"]
        rope_dim = m["rope_dim"]
        key_mla = m["key_length_mla"]
        val_mla = m["value_length_mla"]
        novec = key_mla - rope_dim
        stride = n_lora_kv + rope_dim
        eps = m["eps"]

        x = self.token_embd[token, :n_embd].astype(np.float64).copy()

        t = rmsnorm(x, self.attn_norm, eps)

        # Q: non-lite (n_lora_q > 0): q_a -> rmsnorm -> q_b
        n_lora_q = m["n_lora_q"]
        q_a = np.array([f32_dot(self.q_a[r, :], t) for r in range(n_lora_q)])
        q_a = rmsnorm(q_a, self.q_a_norm, eps)
        q = np.array([f32_dot(self.q_b[r, :], q_a) for r in range(n_heads * key_mla)])

        q_nope = np.zeros(n_heads * novec, dtype=np.float64)
        q_pe = np.zeros(n_heads * rope_dim, dtype=np.float64)
        for h in range(n_heads):
            off = h * key_mla
            for i in range(novec):
                q_nope[h * novec + i] = q[off + i]
            for i in range(rope_dim):
                q_pe[h * rope_dim + i] = q[off + novec + i]

        # KV compression
        kv_cmpr_pe = np.array([f32_dot(self.kv_a_mqa[r, :], t) for r in range(stride)])
        kv_cmpr = kv_cmpr_pe[:n_lora_kv].copy()
        k_pe = kv_cmpr_pe[n_lora_kv:].copy()
        kv_cmpr = rmsnorm(kv_cmpr, self.kv_a_norm, eps)

        for h in range(n_heads):
            rope_inplace(q_pe[h * rope_dim:(h + 1) * rope_dim], pos, m["rope_theta"])
        rope_inplace(k_pe, pos, m["rope_theta"])

        # Absorption: absorbed_q[h, k] = sum_i k_b[i, k, h] * q_nope[h, i]
        # On disk k_b shape [n_heads*n_lora_kv, novec] (rows= (n_heads*n_lora_kv), cols=novec)
        # Rows grouped by head then kv: head h, kv k -> row h*n_lora_kv + k
        absorbed_q = np.zeros(n_heads * n_lora_kv, dtype=np.float64)
        for h in range(n_heads):
            for k in range(n_lora_kv):
                row = self.k_b[h * n_lora_kv + k, :novec]
                absorbed_q[h * n_lora_kv + k] = f32_dot(row, q_nope[h * novec:(h + 1) * novec])

        # Cache write
        for i in range(n_lora_kv):
            cache[pos * stride + i] = kv_cmpr[i]
        for i in range(rope_dim):
            cache[pos * stride + n_lora_kv + i] = k_pe[i]

        kq_scale = self.kq_scale()
        att_out = np.zeros(n_heads * val_mla, dtype=np.float64)
        for h in range(n_heads):
            scores = np.zeros(pos + 1, dtype=np.float64)
            for p in range(pos + 1):
                k_off = p * stride
                s_lat = f32_dot(absorbed_q[h * n_lora_kv:(h + 1) * n_lora_kv], cache[k_off:k_off + n_lora_kv])
                s_rope = f32_dot(q_pe[h * rope_dim:(h + 1) * rope_dim], cache[k_off + n_lora_kv:k_off + stride])
                scores[p] = f32(f32(s_lat + s_rope) * kq_scale)
            probs = softmax_f32(scores)

            cmpr_weighted = np.zeros(n_lora_kv, dtype=np.float64)
            for p in range(pos + 1):
                k_off = p * stride
                w = probs[p]
                for i in range(n_lora_kv):
                    cmpr_weighted[i] = f32(cmpr_weighted[i] + f32(f32(w) * f32(cache[k_off + i])))

            # V expansion: v_b shape [n_heads*val_mla, n_lora_kv]; head h, value v -> row h*val_mla+v
            for v in range(val_mla):
                row = self.v_b[h * val_mla + v, :n_lora_kv]
                att_out[h * val_mla + v] = f32_dot(row, cmpr_weighted)

        # Output projection
        wo_out = np.array([f32_dot(self.attn_output[r, :], att_out) for r in range(n_embd)])
        for i in range(n_embd):
            x[i] = f32(f32(x[i]) + wo_out[i])

        # Dense FFN (fixture FFN weights are all zeros -> contributes nothing)
        t_ffn = rmsnorm(x, self.ffn_norm, eps)
        up = np.array([f32_dot(self.ffn_up[r, :], t_ffn) for r in range(m["n_ff"])])
        gate = np.array([f32_dot(self.ffn_gate[r, :], t_ffn) for r in range(m["n_ff"])])
        hvec = np.empty(m["n_ff"], dtype=np.float64)
        for i in range(m["n_ff"]):
            g = f32(gate[i])
            sig = f32(1.0 / f32(1.0 + math.exp(-g)))
            hvec[i] = f32(f32(sig) * f32(up[i]))
        ff_out = np.array([f32_dot(self.ffn_down[r, :], hvec) for r in range(n_embd)])
        for i in range(n_embd):
            x[i] = f32(f32(x[i]) + ff_out[i])

        # Final norm + output
        x = rmsnorm(x, self.output_norm, eps)
        logits = np.array([f32_dot(self.output_w[r, :], x) for r in range(self.token_embd.shape[0])])
        return logits

    def generate(self, tokens, n_new):
        m = self.meta
        stride = m["n_lora_kv"] + m["rope_dim"]
        max_seq = len(tokens) + n_new
        cache = np.zeros(max_seq * stride, dtype=np.float64)
        logits = None
        seq = list(tokens)
        for pos, tok in enumerate(tokens):
            logits = self.forward(tok, pos, cache)
        for _ in range(max(n_new - 1, 0)):
            nxt = argmax_first(logits)
            seq.append(nxt)
            logits = self.forward(nxt, len(seq) - 1, cache)
        seq.append(argmax_first(logits))
        order = sorted(range(len(logits)), key=lambda i: -float(logits[i]))
        top5 = [(i, f32(logits[i])) for i in order[:5]]
        return seq, top5


def main():
    gguf = import_gguf()
    fixture = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FIXTURE
    print(f"reference forward: {fixture}")
    ref = RefMla(gguf, fixture)

    # 1) single-token deterministic forward
    stride = ref.meta["n_lora_kv"] + ref.meta["rope_dim"]
    logits0 = ref.forward(0, 0, np.zeros(stride, dtype=np.float64))
    print("logits(token 0, pos 0):")
    print("  " + " ".join(f"{v:+.7f}" for v in logits0))

    # 2) two-token causal cache test
    seq, top5 = ref.generate([0, 1, 2], 3)
    print(f"generate([0,1,2], n_new=3): seq={seq} top5={[(i, round(v, 7)) for i, v in top5]}")

    # Dump machine-readable state for the Rust comparison harness
    import json
    out = {
        "fixture": str(fixture),
        "logits_token0_pos0": [round(v, 9) for v in logits0],
        "gen_seq": seq,
        "gen_top5": [[i, round(v, 9)] for i, v in top5],
        "prompt_ids": [0, 1, 2],
    }
    print("REFJSON " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())