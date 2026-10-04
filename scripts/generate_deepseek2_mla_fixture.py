#!/usr/bin/env python3
"""Generate a tiny deterministic DeepSeek-V2 MLA GGUF fixture.

The fixture is intentionally dense-only (one layer, no MoE) so it can validate
the MLA projection, latent KV cache, causal prefill, and output path without a
multi-gigabyte download. It uses llama.cpp's gguf-py writer rather than a
hand-written container.

Usage:
  PYTHONPATH=/path/to/llama.cpp/gguf-py python3 scripts/generate_deepseek2_mla_fixture.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

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
                "gguf-py is required. Set LLAMA_CPP_GGUF_PY or \"PYTHONPATH to llama.cpp/gguf-py.\""
            ) from exc
    return gguf


def add_tensor_data(writer: object, gguf: object, name: str, shape: tuple[int, ...], values: np.ndarray) -> None:
    """Register one tensor using either modern or legacy gguf-py APIs."""
    data = np.ascontiguousarray(values, dtype=np.float32).reshape(shape)

    # Modern gguf-py stores in-memory tensors until write_tensors_to_file().
    # Legacy writers require add_tensor_info() before write_tensor_data().
    if hasattr(writer, "add_tensor"):
        writer.add_tensor(name, data)
    elif hasattr(writer, "add_tensor_info") and hasattr(writer, "write_tensor_data"):
        writer.add_tensor_info(name, list(shape), data.dtype, data.nbytes)
        writer.write_tensor_data(data)
    else:
        raise RuntimeError("unsupported gguf-py writer API")


def eye_padded(rows: int, cols: int) -> np.ndarray:
    data = np.zeros((rows, cols), dtype=np.float32)
    for i in range(min(rows, cols)):
        data[i, i] = 1.0
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "slow-inference" / "models" / "synthetic-deepseek2-mla.gguf",
    )
    args = parser.parse_args()

    gguf = import_gguf()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_embd = 8
    n_heads = 2
    n_ff = 8
    n_vocab = 32
    n_lora_q = 4
    n_lora_kv = 2
    key_length_mla = 4
    value_length_mla = 4
    rope_dim = 2

    writer = gguf.GGUFWriter(args.output, "deepseek2")
    writer.add_string("general.name", "synthetic-deepseek2-mla")
    writer.add_uint32("general.file_type", 0)
    writer.add_uint32("general.alignment", 32)
    writer.add_uint32("deepseek2.block_count", 1)
    writer.add_uint32("deepseek2.embedding_length", n_embd)
    writer.add_uint32("deepseek2.feed_forward_length", n_ff)
    writer.add_uint32("deepseek2.vocab_size", n_vocab)
    writer.add_uint32("deepseek2.attention.head_count", n_heads)
    writer.add_uint32("deepseek2.attention.head_count_kv", n_lora_kv)
    writer.add_uint32("deepseek2.attention.key_length", key_length_mla)
    writer.add_uint32("deepseek2.attention.q_lora_rank", n_lora_q)
    writer.add_uint32("deepseek2.attention.kv_lora_rank", n_lora_kv)
    writer.add_uint32("deepseek2.attention.key_length_mla", key_length_mla)
    writer.add_uint32("deepseek2.attention.value_length_mla", value_length_mla)
    writer.add_uint32("deepseek2.expert_count", 0)
    writer.add_uint32("deepseek2.expert_used_count", 1)
    writer.add_uint32("deepseek2.expert_feed_forward_length", 0)
    writer.add_uint32("deepseek2.expert_shared_count", 0)
    writer.add_uint32("deepseek2.leading_dense_block_count", 1)
    writer.add_uint32("deepseek2.expert_gating_func", 0)
    writer.add_bool("deepseek2.expert_weights_norm", False)
    writer.add_float32("deepseek2.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_float32("deepseek2.rope.freq_base", 10_000.0)
    writer.add_uint32("deepseek2.rope.dimension_count", rope_dim)
    writer.add_float32("deepseek2.rope.scaling.factor", 1.0)
    writer.add_float32("deepseek2.rope.scaling.attn_factor", 1.0)
    writer.add_float32("deepseek2.rope.scaling.yarn_log_multiplier", 0.0)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(n_vocab)])
    writer.add_array("tokenizer.ggml.merges", [])

    # Token embeddings
    token_embd = np.zeros((n_vocab, n_embd), dtype=np.float32)
    token_embd[0] = np.arange(1, n_embd + 1, dtype=np.float32) / n_embd
    add_tensor_data(writer, gguf, "token_embd.weight", token_embd.shape, token_embd)

    # Output weights
    output = np.zeros((n_vocab, n_embd), dtype=np.float32)
    output[:n_embd] = np.eye(n_embd, dtype=np.float32)
    add_tensor_data(writer, gguf, "output.weight", output.shape, output)
    add_tensor_data(writer, gguf, "output_norm.weight", (n_embd,), np.ones(n_embd, dtype=np.float32))

    # Layer 0
    layer = 0
    prefix = f"blk.{layer}"
    # Attention norm
    add_tensor_data(writer, gguf, f"{prefix}.attn_norm.weight", (n_embd,), np.ones(n_embd, dtype=np.float32))
    # Q projection (split into QA and QB)
    add_tensor_data(writer, gguf, f"{prefix}.attn_q_a.weight", (n_lora_q, n_embd), eye_padded(n_lora_q, n_embd))
    add_tensor_data(writer, gguf, f"{prefix}.attn_q_a_norm.weight", (n_lora_q,), np.ones(n_lora_q, dtype=np.float32))
    add_tensor_data(writer, gguf, f"{prefix}.attn_q_b.weight", (n_heads * key_length_mla, n_lora_q), eye_padded(n_heads * key_length_mla, n_lora_q))
    # KV projection (combined)
    add_tensor_data(writer, gguf, f"{prefix}.attn_kv_a_mqa.weight", (n_lora_kv + rope_dim, n_embd), eye_padded(n_lora_kv + rope_dim, n_embd))
    add_tensor_data(writer, gguf, f"{prefix}.attn_kv_a_norm.weight", (n_lora_kv,), np.ones(n_lora_kv, dtype=np.float32))
    add_tensor_data(writer, gguf, f"{prefix}.attn_k_b.weight", (n_heads * n_lora_kv, key_length_mla - rope_dim), eye_padded(n_heads * n_lora_kv, key_length_mla - rope_dim))
    add_tensor_data(writer, gguf, f"{prefix}.attn_v_b.weight", (n_heads * value_length_mla, n_lora_kv), eye_padded(n_heads * value_length_mla, n_lora_kv))
    # Output projection
    add_tensor_data(writer, gguf, f"{prefix}.attn_output.weight", (n_embd, n_heads * value_length_mla), np.eye(n_embd, dtype=np.float32))

    # FFN norm
    add_tensor_data(writer, gguf, f"{prefix}.ffn_norm.weight", (n_embd,), np.ones(n_embd, dtype=np.float32))
    # FFN up
    add_tensor_data(writer, gguf, f"{prefix}.ffn_up.weight", (n_ff, n_embd), np.zeros((n_ff, n_embd), dtype=np.float32))
    # FFN gate
    add_tensor_data(writer, gguf, f"{prefix}.ffn_gate.weight", (n_ff, n_embd), np.zeros((n_ff, n_embd), dtype=np.float32))
    # FFN down
    add_tensor_data(writer, gguf, f"{prefix}.ffn_down.weight", (n_embd, n_ff), np.zeros((n_embd, n_ff), dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()