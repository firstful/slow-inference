"""Disk-streaming layer-by-layer forward pass for llama-arch MoE GGUF models.

DISCIPLINE: no tensor is ever fully resident. Every weight row is
seek() + read() + dequant() at use time. Only these live in RAM:
  - GGUF header/tensor dir (KBs)
  - KV cache (layers x seq x n_embd x f32)
  - activation vectors for the current token
  - the single weight row currently being multiplied

For MoE: expert tensors are stacked [n_expert, n_out, n_in]; a chosen
expert's rows are contiguous, so we stream ONLY the selected experts' rows.
"""
import numpy as np

from .gguf import parse
from .quant import dequant_row


class StreamedTensor:
    """A tensor that lives on disk. Rows are read and dequantized one at a time."""

    def __init__(self, f, gf, tinfo):
        self.f = f
        self.base = gf.data_start + tinfo.offset
        self.dims = tinfo.dims          # ne0 = row width (input dim)
        self.dtype = tinfo.dtype
        self.nbytes = tinfo.nbytes
        if len(tinfo.dims) >= 2:
            self.row_elems = tinfo.dims[0]
            self.n_rows = int(np.prod(tinfo.dims[1:]))
            self.row_bytes = tinfo.nbytes // self.n_rows
        else:
            self.row_elems = tinfo.dims[0]
            self.n_rows = 1
            self.row_bytes = tinfo.nbytes

    def row(self, i: int) -> np.ndarray:
        """Read + dequantize flat row i (i = expert*n_rows_per_expert + r for 3D)."""
        self.f.seek(self.base + i * self.row_bytes)
        raw = self.f.read(self.row_bytes)
        return dequant_row(raw, self.dtype, self.row_elems)


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return (x / np.sqrt(np.mean(x * x) + eps)) * w


def rope(x: np.ndarray, pos: int, theta: float) -> None:
    """Apply RoPE in place to one head's vector of even length."""
    d = len(x) // 2
    freqs = theta ** (-2.0 * np.arange(d) / d)
    ang = pos * freqs
    cos, sin = np.cos(ang), np.sin(ang)
    a = x[0::2].copy()
    b = x[1::2].copy()
    x[0::2] = a * cos - b * sin
    x[1::2] = a * sin + b * cos


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Model:
    def __init__(self, path: str, verbose: bool = True):
        self.verbose = verbose
        self.gf = parse(path)
        self.f = open(path, "rb")
        md = self.gf.metadata
        self.n_layers = md["llama.block_count"]
        self.n_embd = md["llama.embedding_length"]
        self.n_ff = md["llama.feed_forward_length"]
        self.n_heads = md["llama.attention.head_count"]
        self.n_kv_heads = md["llama.attention.head_count_kv"]
        self.head_dim = self.n_embd // self.n_heads
        self.rope_theta = md.get("llama.rope.freq_base", 10000.0)
        self.eps = md.get("llama.attention.layer_norm_rms_epsilon", 1e-5)
        self.n_experts = md.get("llama.expert_count", 0)
        self.n_experts_used = md.get("llama.expert_used_count", 1)
        if self.n_kv_heads != self.n_heads:
            raise NotImplementedError("GQA not implemented yet; use an MHA model")
        self.path = path

    def close(self):
        self.f.close()

    def _t(self, name: str) -> StreamedTensor:
        return StreamedTensor(self.f, self.gf, self.gf.tensors[name])

    # ---- streamed matvec: y[r] = W_row_r . x ; W never fully resident ----
    def matvec(self, W: StreamedTensor, x: np.ndarray, row_ids=None) -> np.ndarray:
        ids = range(W.n_rows) if row_ids is None else row_ids
        y = np.empty(len(ids), dtype=np.float32)
        for j, r in enumerate(ids):
            y[j] = np.dot(W.row(r), x)
        return y

    def expert(self, layer: int, e: int, x: np.ndarray) -> np.ndarray:
        """One expert's FFN: down_e( silu(gate_e x) * up_e x )."""
        up_t = self._t(f"blk.{layer}.ffn_up_exps.weight")
        gate_t = self._t(f"blk.{layer}.ffn_gate_exps.weight")
        down_t = self._t(f"blk.{layer}.ffn_down_exps.weight")
        # stacked dims: up/gate (n_ff, n_embd, E) -> rows_per_expert = n_ff
        # down (n_embd, n_ff, E) -> rows_per_expert = n_embd
        rpe_up = up_t.n_rows // self.n_experts
        rpe_down = down_t.n_rows // self.n_experts
        ids_up = range(e * rpe_up, (e + 1) * rpe_up)
        ids_down = range(e * rpe_down, (e + 1) * rpe_down)
        up = self.matvec(up_t, x, ids_up)
        gate = self.matvec(gate_t, x, ids_up)
        h = silu(gate) * up
        return self.matvec(down_t, h, ids_down)

    def forward(self, token: int, pos: int, k_cache, v_cache) -> np.ndarray:
        n_embd, n_heads, head_dim = self.n_embd, self.n_heads, self.head_dim
        eps = self.eps

        x = self._t("token_embd.weight").row(token).copy()

        for layer in range(self.n_layers):
            # --- attention ---
            attn_norm_w = self._t(f"blk.{layer}.attn_norm.weight").row(0)
            t = rmsnorm(x, attn_norm_w, eps)

            q = self.matvec(self._t(f"blk.{layer}.attn_q.weight"), t)
            k = self.matvec(self._t(f"blk.{layer}.attn_k.weight"), t)
            v = self.matvec(self._t(f"blk.{layer}.attn_v.weight"), t)

            q = q.reshape(n_heads, head_dim)
            k = k.reshape(n_heads, head_dim)
            for h in range(n_heads):
                rope(q[h], pos, self.rope_theta)
                rope(k[h], pos, self.rope_theta)

            k_cache[layer][pos] = k.ravel()
            v_cache[layer][pos] = v

            scale = 1.0 / np.sqrt(head_dim)
            out = np.empty(n_embd, dtype=np.float32)
            for h in range(n_heads):
                keys = k_cache[layer][:pos + 1, h * head_dim:(h + 1) * head_dim]
                vals = v_cache[layer][:pos + 1, h * head_dim:(h + 1) * head_dim]
                scores = keys @ q[h] * scale
                probs = softmax(scores)
                out[h * head_dim:(h + 1) * head_dim] = probs @ vals

            x = x + self.matvec(self._t(f"blk.{layer}.attn_output.weight"), out)

            # --- MoE FFN ---
            ffn_norm_w = self._t(f"blk.{layer}.ffn_norm.weight").row(0)
            t = rmsnorm(x, ffn_norm_w, eps)

            router_logits = self.matvec(
                self._t(f"blk.{layer}.ffn_gate_inp.weight"), t)
            top = np.argsort(router_logits)[-self.n_experts_used:]

            moe_out = np.zeros(n_embd, dtype=np.float32)
            for e in top:
                moe_out += self.expert(layer, int(e), t)
            x = x + moe_out

        # --- final norm + logits ---
        out_norm_w = self._t("output_norm.weight").row(0)
        x = rmsnorm(x, out_norm_w, eps)
        return self.matvec(self._t("output.weight"), x)

    def generate(self, tokens: list, n_new: int) -> list:
        """Greedy decode. tokens: prompt token ids. Returns full sequence."""
        seq = list(tokens)
        n_layers = self.n_layers
        max_seq = len(tokens) + n_new
        k_cache = np.zeros((n_layers, max_seq, self.n_embd), dtype=np.float32)
        v_cache = np.zeros((n_layers, max_seq, self.n_embd), dtype=np.float32)

        # prefill prompt one token at a time (positions 0..len-2 just fill KV)
        logits = None
        for pos, tok in enumerate(seq):
            logits = self.forward(tok, pos, k_cache, v_cache)
        for step in range(n_new - 1):
            nxt = int(np.argmax(logits))
            seq.append(nxt)
            logits = self.forward(nxt, len(seq) - 1, k_cache, v_cache)
        seq.append(int(np.argmax(logits)))
        return seq
