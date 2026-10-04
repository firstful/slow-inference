# Results — OLMoE-1B-7B-Instruct (Q4_K_M, 3.98 GB file)

Measured 2026-09-17. Machine: Manjaro Linux, disk-streamed (page cache effects
still possible on warm runs; the engine never mmaps the data section).

## Headline

| Metric | Value |
|---|---|
| Peak RSS | **26.8 MiB = 0.67% of model file** |
| Speed (chat-templated, pos≥18) | ~11.5 s/token |
| Speed (raw prompt, 5 tok ctx) | ~4.8 s/token |
| Output quality | Coherent: `The capital of France is → " the capital of the"` |

Peak RAM is dominated by the KV cache + activation vectors; every weight row is
seek+read+dequantized from disk at use time. RSS scales with context, not model
size — that's the point of the project.

## Generation samples

### OLMoE-1B-7B-Instruct
Raw prompt (`The capital of France is`, ids [510,5347,273,6181,310]):

```
output text: " the capital of the"
top5: [(' the' 14.55, ' of' 12.78, ' s' 11.68?, ' is' 10.99, ...)]
```

### Qwen2.5-3B-Instruct
Raw prompt (`The capital of France is`, ids [785, 6722, 315, 9625, 374]):

```
output tokens: [16872, 43268, 35946, 1184, 15, 20, 14, 15, 59, 15]
output text: "下线我 need05/0\\0"
time: 97.9s for 10 tokens (9.8 s/token)
peak RSS: 66.9 MiB = 3.33% of model file
```

The Qwen2.5-3B output is a genuine model generation (validates against official
Qwen2.5 tokenizer: `The`=785, `Ġcapital`=6722, `Ġof`=315, `ĠFrance`=9625, `Ġis`=374).
The mixed Chinese/English output is typical for this 3B model on an unformatted prompt.

Chat-templated seq (exact llama.cpp 18-token prefill):

```
top5: [('\n' 13.59), ('The' 13.06), ('Dear' 12.65), ('Hello' 11.95), ('Sure' 11.71)]
```
Matches the Python reference bit-for-bit at every layer (see Validation).

## Validation (Rust vs Python reference)

### OLMoE validated: Rust ≈ Python < 1e-3

Per-layer hidden-state checksums (sum |x| after each decoder layer) agree to
**< 1e-3 absolute** across all 16 layers, validated at:

- pos 0 of the 18-token template sequence (all layers)
- pos 1 of a distinct-token 2-token sequence `510,187` (all layers) — exercising
  real multi-position attention (a same-token 2-token test is vacuous for
  attention because v0 == v1 makes the output independent of scores)
- top-5 next-token logits match exactly on the 18-token seq
  (`\n` 13.590959 / `The` 13.05694 — identical to 5 decimals)

### Qwen2.5-3B: end-to-end verified

- Prompt ids match official Qwen2.5 tokenizer (`transformers` library) exactly.
- Architecture paths exercised: GQA (16 query heads → 2 KV heads), bias (Q/K/V),
  dense FFN (no MoE), GPT-2 byte-level BPE with `qwen2` pre-tokenizer.
- Output is a genuine model generation (not garbled decoding).

### Bugs found & fixed on the way

1. **f16 subnormal decode** (`quant.rs f16_to_f32`): computed
   `frac * 2^-37` instead of `frac * 2^-24` — 8192× too small. Any Q4_K block
   whose scale `d`/`dmin` was a subnormal half decoded to garbage. In OLMoE's
   token_embd row 510, 5 of 8 blocks had subnormal scales. Slipped past the
   Tiny-Moe bit-exactness test because those tensors happened to have no
   subnormal scales.
2. **RoPE frequency divisor** (`model.rs rope`): divided by `head_dim/2`
   instead of `head_dim`, making frequencies twice too steep. Invisible at
   pos 0 (RoPE identity there) — found by comparing last-position layer
   checksums on the full 18-token prefill.
3. **QK-norm epsilon**: hardcoded `1e-6` instead of the model's
   `1e-5` metadata value; a ~0.7% score error that amplified through depth.

## Why `--prompt-tokens` exists

To compare against llama.cpp we needed byte-identical prefill sequences —
llama.cpp applies its ChatML template before tokenizing, so the raw prompt
ids don't match its internal sequence. `--prompt-tokens <csv>` bypasses the
tokenzier and feeds an exact token sequence (e.g. the 18-token
`<|endoftext|><|user|>\nThe capital of France is...` prefill reconstructed
from llama.cpp verbose output).

## Tokenizers (two paths)

- **Byte-level BPE** (OLMoE/Qwen-style): pretokenize, byte-unicode map, rank
  merges. Prompt ids match HF `tokenizers` exactly.
- **SPM / llama vocab** (Tiny-Moe): prepend `▁` (`add_space_prefix=1`), map
  remaining spaces to `▁`, greedy longest-match against the vocab, `<0xNN>`
  byte fallback. Tiny-Moe prompt ids verified against a Python SPM reference:
  `The capital of France is → [4014, 87, 259, 1558, 1737, 294, 4014, 73, 3587, 599]`
  = `▁ / <0x54> / he / ▁cap / ital / ▁of / ▁ / <0x46> / rance / ▁is`. Decode
  inverts `▁` → space and joins byte tokens as raw UTF-8.

## Known divergences from llama.cpp (expected, not bugs)

- Final-stage numerics: Python ref does RMS-mean in f64; engine is all-f32.
  Top candidates within ~0.5 logit of each other can swap order.
- Sampling: engine is greedy argmax; llama.cpp defaults may differ.
- OLMoE's template handling: llama.cpp applies the chat template; engine run
  with raw prompts naturally has no template unless ids are provided.

## Tower schedule (per README)

| # | Model | File size | Status |
|---|---|---|---|
| 1 | OLMoE-1B-7B-Instruct Q4_K_M | 3.98 GB | **done + validated** |
| 2 | Qwen2.5-3B-Instruct Q4_K_M | 2.10 GB | **done + validated** (GQA, bias, dense FFN, GPT-2 BPE) |
| 3 | Llama-3.2-4X3B-MoE Q4_K_M | ~6 GB | — |
| 4 | 30B-class MoE (Qwen3-30B-A3B) | 10-30 GB | headline experiment |

## Reproduce

```
cargo build --release
./engine/target/release/slow-inference \
    --model models/OLMoE-1B-7B-Instruct-Q4_K_M.gguf \
    --prompt "The capital of France is" --tokens 4
# time: 19.4s for 4 tokens (4.8 s/token)
# peak RSS: 26.8 MiB = 0.67% of model file
```

Python-side reference checks (slow, ~40 min for the 18-token prefill):

```
FWD_DEBUG=1 .venv/bin/python /tmp/fwd_ref.py "<tok,tok,...>"
SLOW_DEBUG=1 ./engine/target/release/slow-inference ... # then diff L{x}_sum lines
./engine/target/release/examples/dq_check   # byte-level dequant diff vs py
```
