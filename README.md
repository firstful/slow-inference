# Slow Inference

**Disk-streamed LLM inference. Weights live on disk; computation pulls them in
layer-by-layer, row-by-row. Goal: minimize peak RAM, accept seconds-or-worse
per token.**

Engine language: **Rust** (zero-external-crate). The original Python prototype
that was used to validate the dequant kernels bit-exactly against `gguf-py`
lives in `py-reference/` and is kept as the reference implementation.

## What it is

A from-scratch inference engine for llama-arch Mixture-of-Experts GGUF models
with an explicit and measured peak-RAM discipline:

- **No mmap of the data section.** Only the GGUF header + tensor directory
  (a few KB) is parsed into memory. Weights stay on disk.
- **Every weight row is seek()+read()+dequant() at use time.** Row-chunked
  reads (<=256 KiB per chunk) amortize syscalls without unbounded buffering.
- **KV cache** is the only O(context) structure (layers x seq x embd, f32).
  Everything else is O(1) activation vectors.
- **MoE is the point**: expert tensors are stacked [n_expert, n_out, n_in],
  so a chosen expert's rows are contiguous; for each token we stream only the
  top-k experts' rows and skip the rest. E.g. OLMoE-1B-7B reads 8/64 experts
  => ~12% of the FFN weight bytes per token from disk.

The accepted trade-off: with nothing cached, each generated token re-reads its
active weights from disk, so throughput is bounded by disk bandwidth. We are
explicitly OK with seconds-per-token; the goal is to map the RAM floor, not to
chase tokens/sec.
## The idea

Conventional inference engines (llama.cpp, vLLM) mmap or load the whole model
into RAM up front. This project inverts that: the model file on disk is the
source of truth, and weights are read **one layer at a time** with a sliding
window of buffers. At any instant we hold only:

- the current layer's weights being multiplied (streamed in row-blocks),
- the KV cache,
- activations for the current token(s),
- a small prefetch buffer for the next tensor.

Everything else stays on disk. We never need the whole model in RAM, ever.

## Design goals (in priority order)

1. **Peak RSS is the score.** Target: peak RSS ≤ 5% of model file size.
   If we can go lower, go lower. Measure it honestly (VmHWM, not "should be").
2. **Correctness.** A token is only a token if the math is right. Cross-check
   logits/tokens against llama.cpp on the same GGUF whenever possible.
3. **Read efficiency.** Sequential per-tensor reads, aligned block streaming,
   optional readahead. O_DIRECT is on the table later; first get mechanics right.
4. **Speed is explicitly not a goal.** We expect seconds/minutes per token.
   This exists for autonomous batch jobs, not interactive use.

## Why MoE

Mixture-of-Experts models have huge total parameter counts but small *active*
sets per token. That makes them the perfect subject: a 30B-total model might
only need ~2-3B params of weights per token, so streaming disk→compute has a
chance of being tractable while still exercising the full layer-by-layer
disk-read machinery on a very large file.

## Model ladder

| Stage | Model | File size | Purpose |
|-------|-------|-----------|---------|
| 0 | Tiny-Moe (Q4_K_M, ~90MB) | 0.09 GB | mechanics smoke test |
| 1 | OLMoE-1B-7B-Instruct (Q4_K_M) | ~4.3 GB | first real MoE validation, ≤5% RSS |
| 2 | Qwen2.5-MOE-2X1.5B (~2.5GB Q4_K_M) | 2.5 GB | cross-arch check (Qwen2MoE arch) |
| 3 | Llama-3.2-4X3B-MOE (Q4_K_M, ~6GB) | 6 GB | bigger MoE |
| 4 | 30B-class MoE (Qwen3-30B-A3B etc.) | 10-30 GB | the headline experiment |

## Architecture

```
src/
  gguf.py       — minimal GGUF container parser (mmap-free, header + tensor dir only)
  quant.py      — dequant kernels for Q4_K_M / Q8_0 / F16 (numpy)
  runner.py     — layer-by-layer forward loop with strict buffer discipline
  rss.py        — peak-RSS instrumentation (VmHWM from /proc)
  bench.py      — end-to-end benchmark harness, writes results/
```

## Non-goals (for now)

- KV-cache paging to disk (maybe later — KV is small relative to weights)
- GPU acceleration (defeats the purpose)
- Batch >1 tokens per step (single-token streaming keeps buffers minimal)
- Training / fine-tuning

## Honest expectations

Sequential disk read of a 30GB file at ~500MB/s is ~60s *per token* at best.
On spinning rust it would be minutes. That is accepted and is the point.

## Status

**Tower #1 (OLMoE-1B-7B-Instruct Q4_K_M, 3.98 GB): running and numerically
validated.** Peak RSS **26.8 MiB (0.67% of the model file)**, ~4.8 s/token.
The forward pass matches the Python reference to < 1e-3 at every layer on
multi-position tests, and greedy output is coherent. See
[docs/results.md](docs/results.md) for measured numbers, the bug list
(f16-subnormal decode, RoPE divisor, QK-norm eps), and the validation method.

## Layout

```
slow-inference/
├── engine/            Rust crate (the real engine)
│   └── src/
│       ├── main.rs    CLI: --model --prompt --tokens, prints RSS + timing
│       ├── gguf.rs    minimal GGUF parser (header + tensor dir only)
│       ├── quant.rs   dequant kernels: F32/F16/Q8_0/Q4_K/Q6_K (bit-exact vs gguf-py)
│       ├── model.rs   llama-arch MoE forward pass, streamed matvec
│       └── rss.rs     VmHWM peak-RSS instrumentation
├── py-reference/      Python prototype (kernel validation vs gguf-py, runner logic)
├── models/            downloaded GGUFs
├── docs/              design notes
└── results/           measured runs (RSS, s/token)
```

## Usage

```
cargo run --release -- --model models/Tiny-Moe.Q4_K_M.gguf \
    --prompt "The capital of France is" --tokens 8
```
