# slow-inference: llama.cpp parity roadmap

**Goal:** Extend slow-inference to support every GGUF architecture that llama.cpp can load, using llama.cpp's model definitions (in `src/models/`) as the authoritative reference for tensor naming, forward-pass logic, and tokenizer conventions.

## Architecture assets

| Asset | Path |
|---|---|
| llama.cpp architecture enum | `~/llama.cpp/src/llama-arch.h` (153 architectures + UNKNOWN) |
| model `.cpp` files | `~/llama.cpp/src/models/*.cpp` (150+ files, one per architecture) |
| tensor name constants + GGUF key constants | `~/llama.cpp/src/llama-arch.h`, `gguf-py/gguf/constants.py` |
| slow-inference current engine | `~/slow-inference/engine/src/` (model.rs, gguf.rs, quant.rs, main.rs) |
| slow-inference tests | only 1 test in gguf.rs; no e2e, no forward-test, no regression |

## Current capability (baseline)

| Feature | Status |
|---|---|
| Supported GGUF types | F32, F16, Q4_0, Q8_0, Q4_K, Q6_K (Q8_K mapping is wrong: `(4,8)`) |
| Architectures | Single hard-coded llama-arch tensor naming |
| Tokenizer | Hand-rolled GPT-2 BPE + llama SPM (no proper pretokenizer for most models) |
| Attention | MHA/GQA only, per-head softmax, full KV cache |
| Tensor naming | Flat string lookup by hard-coded format strings |
| Metadata keys | `{arch}.block_count`, `{arch}.embedding_length`, etc. |
| Testing | 1 unit test (parse gguf header) |
| Weight type | Only one lookup per tensor; no per-layer dtype overrides |
| MoE | Basic top-k expert routing, no shared experts, no expert weights norm/scale |
| QK-norm | Per-head RMSNorm, via heuristic detection of `attn_q_norm.weight` existence |
| Bias | Binary flag — all layers or none |

## Tiered backlog

Each tier is ordered from simple to complex within the tier. Models listed are *representative* — the architecture class covers all models using it.

### Tier 0 — Existing works, just clean up

| Item | Effort | Detail |
|---|---|---|
| 0.1 Fix Q8_K type layout | minutes | `type_layout(16)` returns `(4, 8)` — should be `(4, 256)` per ggml. Fix: `GGML_Q8_K => Some((4, 256))`, write dequant kernel. |
| 0.2 Add Q2_K, Q3_K, Q5_K dequant | days | Port from ggml-quants.c: Q2_K (84,256), Q3_K (110,256), Q5_K (176,256). These are the most common quant types (Q2_K is rare but Q3_K and Q5_K are everywhere). |
| 0.3 Proper per-layer tensor detection | hours | Instead of binary `has_bias` and `qk_norm` flags from layer 0 only, check per-layer as llama.cpp does: `TENSOR_NOT_REQUIRED`. |
| 0.4 Add `output.weight` fallback to token_embd | minutes | ✅ **tick 30 — DONE, and as one helper (`Model::output_t`) rather than the four inline copies it used to be.** The copies are how it was lost: a rewrite at 11:00 on 2026-09-30 replaced two of the four with hard errors and left the comment above a third still describing the fallback it no longer performed, and `quant_gap` + `ref_parity` were red with "tied embeddings unavailable" until someone read that comment. `weight_tying_disabled` and the new `output_tie_bypassed_at_one_site` mutations both anchor on the one body |
| 0.5 Report arch and ftype on load | minutes | Print the architecture string and file type kv in the startup banner, not just computed params. |

### Tier 1 — Architecture name mapping (same tensor layout, different arch key)

The tensor layout is identical to llama: `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, `blk.N.attn_v.weight`, etc. These models just use a different value in `general.architecture`.

| # | Architecture | `general.architecture` value | Popular models (Q4_K_M < 100 GB) |
|---|---|---|---|
| 1.1 | Qwen2 | `qwen2` | Qwen2.5-7B (4.7 GB), Qwen2.5-14B (9.4 GB), Qwen2.5-32B (18.7 GB), Qwen2.5-Coder-7B, Qwen2.5-72B (42 GB Q4) |
| 1.2 | Gemma 2 | `gemma2` | Gemma-2-2B (1.6 GB), Gemma-2-9B (6.7 GB), Gemma-2-27B (19 GB) |
| 1.3 | CodeShell | `codeshell` | CodeShell-7B (4.6 GB) |
| 1.4 | Orion | `orion` | Orion-14B (9.0 GB) |
| 1.5 | InternLM2 | `internlm2` | InternLM2-7B, InternLM2-20B |
| 1.6 | StableLM | `stablelm` | StableLM-2-1.6B, 3B, 12B |
| 1.7 | Xverse | `xverse` | XVERSE-7B, 13B |
| 1.8 | PLamo | `plamo`, `plamo2` | PLamo-1B, PLamo-13B |
| 1.9 | Baichuan | `baichuan` | Baichuan-7B, 13B |
| 1.10 | Olmo / Olmo2 | `olmo`, `olmo2` | OLMo-1B, OLMo-7B |
| 1.11 | Smollm3 | `smollm3` | SmolLM3-135M, 500M, 1.7B |
| 1.12 | OpenELM | `openelm` | OpenELM-270M, 450M, 1.1B, 3B |
| 1.13 | Nemotron | `nemotron`, `nemotron_h` | Nemotron-H-4B, 8B, 21B |

**Implementation:** Add a `HashMap<&str, ArchVariant>` mapping architecture string → a trait/dispatch. Current llama code becomes `ArchVariant::Llama`. Each new variant maps arch string → same forward-pass code, potentially with minor parameter differences.

**Milestone checkpoint:** When all Tier 1 models parse, load metadata, and generate plausible tokens (test against a single prompt), the engine is no longer llama-only — any standard GQA decoder works.

**Estimate:** 2–3 days (mapping struct, 15 entries, validation against 3 model quants).

### Tier 2 — Tensor layout variants (same forward pass, different tensor names)

The forward pass structure (Q/K/V projection → RoPE → dot-product attention → weighted sum → O projection → FFN) is the same. The tensor naming differs or tensors are fused.

| # | Architecture | Key differences | Popular models (Q4_K_M < 100 GB) |
|---|---|---|---|
| 2.1 | Phi-3 / Phi-3.5 | `phi3`. Uses FUSED QKV: `blk.N.attn_qkv.weight` [3×n_embd, n_embd] instead of separate q/k/v. Also uses `blk.N.attn_output.bias`. | Phi-3-mini-4k (2.3 GB Q4), Phi-3-7B (4.5 GB), Phi-3.5-MoE (5.1 GB) |
| 2.2 | Falcon / Falcon-H1 | `falcon`. Uses parallel attention+FFN: single attn_norm, then QKV + FFN up/gate in parallel. Tensor names: `transformer.h.N.attn.query_key_value.weight`, single `output_norm`. No ffn_norm. | Falcon-7B, Falcon-40B, Falcon-H1-Next-7B, Falcon-H1-Next-40B |
| 2.3 | GPT-2 | `gpt2`. No per-layer norms — layer norm before AND after attention+FFN. Different tensor naming: `model.h.N.attn.c_attn.weight`, `ln_1.weight`, `ln_2.weight`, etc. Causal masking only. GELU activation. | GPT-2 (124M), GPT-2-large (774M), GPT-2-XL (1.5B) |
| 2.4 | GPT-NeoX / GPT-J | `gptneox`, `gptj`. Parallel attention+FFN, parallel_residual. Different tensor layout. | GPT-NeoX-20B, GPT-J-6B, Pythia 70M-12B |
| 2.5 | BLOOM | `bloom`. ALiBi instead of RoPE, fused QKV. | BLOOM-560M, BLOOM-3B, BLOOM-7B, BLOOM-176B |
| 2.6 | MPT | `mpt`. ALiBi bias, no QK-norm, uses QKV fused + ffn_up/gate/down fused. | MPT-7B, MPT-30B |

**Implementation:** For each variant, add a forward-pass function that:
1. Reads the variant's tensor names (could be HashMap or match)
2. Executes the same math but with possibly fused tensors (split QKV fused output)
3. Applies variant-specific activation (GELU, silu, etc.)

**Milestone checkpoint:** GPT-2 support unlocks the whole small-model space. Phi-3 support is the most requested compact model.

**Estimate:** 3–5 days per variant.

### Tier 3 — Attention mechanism changes

The forward pass structure changes beyond tensor names.

| # | Architecture | Key differences | Popular models | Effort |
|---|---|---|---|---|
| 3.1 | **Qwen2-VL** | Multi-modal: vision encoder → cross-attention layers. Also uses M-RoPE (multi-dimensional RoPE). | Qwen2-VL-7B, Qwen2-VL-72B | weeks |
| 3.2 | **DeepSeek-V2/V3** (`deepseek2`) | **Multi-head Latent Attention (MLA)**. Instead of computing K,V directly, project → latent → reconstruct with KV-norm. Completely different KV cache format (latent + per-token rope). Uses MoE with shared experts. | DeepSeek-V2-Lite (16B), V3-671B (too large), DeepSeek-Coder-33B | **major** (2–4 weeks) |
| 3.3 | **Gemma 3** (`gemma3`) | Sliding window attention + global attention alternating layers. Custom attention implementation. | Gemma-3-1B, 4B, 12B, 27B (all <100 GB) | 1–2 weeks |
| 3.4 | **Gemma 4** (`gemma4`) | Multi-head attention with expanded-head + post-attention gate. | Gemma-4-4B, 9B, 16B, 27B | 1 week |
| 3.5 | **GLM-4** (`glm4`, `glm4_moe`) | M-RoPE, QK-norm, custom attention mask. | GLM-4-9B, CodeGeeX-4 | 1 week |
| 3.6 | **Step-35** (`step35`) | Step-aware attention (third-order Taylor expansion of attention logits). Custom forward pass. | Step-35-30B, Step-35-Mini | 1 week |
| 3.7 | **Mistral-4** (`mistral4`) | MOE variant of DeepSeekV2 MLA + MoE with shared experts | Mistral-4-7B, 24B | depends on MLA |
| 3.8 | **DFlash / DeepSeek-V4** (`dflash`, `deepseek4`) | Advanced recurrent + attention hybrid, multi-layer KV caching | DFlash-7B, DeepSeek-V4 | 2–4 weeks |
| 3.9 | **K2-Horizon** `k2_horizon` | Attention gate (softplus×log2), grouped RMSNorm, optional MoVA | K2-Horizon-7B, 36B-MoVA | 1 week |
| 3.10 | **Mistral-3** | `mistral3`. Text-only but uses 3B-Teacher-Student layer arrangement. Interleaved soft-sliding window. | Mistral-3-2B, 7B, 24B | 1 week |

### Tier 4 — Non-transformer / Structured State-Space

Entirely different forward-pass mechanism. Cannot reuse the current attention+FFN architecture.

| # | Architecture | Mechanism | Popular models | Effort |
|---|---|---|---|---|
| 4.1 | **Mamba / Mamba2** | SSD/SSM: no attention, no KV cache. Discrete scan over sequence. State dimension ~16× hidden. | Mamba-2.8B, Mamba2-1.3B, Mamba-3B-slim, xLSTM (modular) | 1 week per variant |
| 4.2 | **RWKV-6 / 7** | Time-mix + channel-mix RNN. Linear attention + recurrence. Different tensor layout. | RWKV-3B, RWKV-7-3B/7B, Eagle-7B (RWKV) | 1 week |
| 4.3 | **Jamba** | Hybrid: Mamba layers + attention layers, alternating. Needs both SSM and attention. | Jamba-1.6B, Jamba-12B, Samba | 2 weeks |
| 4.4 | **BitNet** | 1.58-bit ternary weights. BitNet b1.58. All weights are {-1, 0, +1}, no multiplies. Different GGUF dtypes. | BitNet-b1.58-3B, BitNet-b1.58-7B | 5 days |
| 4.5 | **PLM / xLSTM** | Extended LSTM with higher-order recurrence. | PLM-1B | 1 week |

### Tier 5 — Multi-modal / Vision + Language

Requires a vision encoder (ViT/SigLIP) and a cross-attention projection layer.

| # | Architecture | Vision encoder | Popular models | Effort |
|---|---|---|---|---|
| 5.1 | **LLaVA / LLaVA-NeXT** | CLIP ViT (Sigmoid) + MLP connector | LLaVA-1.6-7B/13B/34B, LLaVA-Qwen, LLaVA-Llama3 | 2 weeks |
| 5.2 | **Qwen2-VL** | Vision transformer + spatial merging | Qwen2-VL-2B, 7B | 2–3 weeks |
| 5.3 | **Gemma 3 (multimodal)** | SigLIP + attention pooling | Gemma-3-4B-VL, Gemma-3-12B-VL | 1 week |
| 5.4 | **CogVLM** | Cross-attention from vision to text | CogVLM-17B | 2 weeks |
| 5.5 | **Florence-2** | Dual encoder | Florence-2-232M, 0.7B, 2.3B | 2 weeks |
| 5.6 | **MiniCPM-V** | SigLIP + resampler | MiniCPM-V-2B, MiniCPM-Llama3-V-2.5 | 2 weeks |

### Tier 6 — Embedding / Reranker / Classification

Non-generative models that produce vector embeddings.

| # | Architecture | Differences | Popular models | Effort |
|---|---|---|---|---|
| 6.1 | **BERT / ModernBERT** | Bidirectional encoder, no causal mask, CLS+pooling | BERT-base, ModernBERT-base, jina-bert-v2/v3 | 3 days |
| 6.2 | **Nomic-BERT** | BERT + Matryoshka representation | Nomic-Embed-Text-v1 (137M), v2 | 2 days |
| 6.3 | **Jina-BERT** | ALiBi + Bidirectional + Matryoshka | Jina-Embeddings-v2/v3 | 2 days |
| 6.4 | **Gemma-Embedding** | Decoder-style embedding model | Gemma-Embed-7B | 2 days |
| 6.5 | **Pangu-Embed** | Pangu-Embed | Pangu-Embed-1.2B | 2 days |
| 6.6 | **Llama-Embed, Minicpm3 (embedding task)** | Special-purpose | Various | 2 days |

### Tier 7 — Quantization coverage

Add all missing GGML types. Order matters because lower tiers need higher tiers for model support.

| # | Type | Block (bytes, elems) | Status | Verified |
|---|---|---|---|---|
| 7.1 | Q2_K | (84,256) | ✅ done | bit-exact (tick 13; layout bug fixed) |
| 7.2 | Q3_K | (110,256) | ✅ done | bit-exact (tick 13; layout bug fixed) |
| 7.3 | Q5_K | (176,256) | ✅ done | bit-exact (tick 13) |
| 7.4 | Q8_K (fix) | (292,256) | ✅ done | bit-exact (tick 13; type id 15, was 16) |
| 7.5 | IQ1_S / IQ1_M | (50,256) / (56,256) | ✅ done | bit-exact (tick 13) |
| 7.6 | IQ2_XXS / IQ2_XS / IQ2_S | (66,256) / (74,256) / (82,256) | ✅ done | bit-exact (tick 13) |
| 7.7 | IQ3_XXS / IQ3_S | (98,256) / (110,256) | ✅ done | bit-exact (tick 13) |
| 7.8 | IQ4_NL / IQ4_XS | (18,32) / (136,256) | ✅ done | bit-exact (tick 13) |
| 7.9 | TQ1_0 / TQ2_0 | (54,256) / (66,256) | ✅ done | bit-exact (tick 13) |
| 7.10 | BF16 | (2,1) | ✅ done | bit-exact (tick 13) |
| 7.11 | Q4_0 | (18,32) | ✅ done | bit-exact (tick 13; kernel was missing entirely) |

**Tier 7 status: complete.** All 21 types are proven bit-exact against llama.cpp's own
`dequantize_row_*` kernels by `engine/tests/quant_golden.rs`. Not implemented: MXFP4,
NVFP4, Q1_0, Q2_0 — no GGUF of these is in circulation and ggml exposes no reachable
reference kernel for them.

### Tier 8 — Tokenizer support

slow-inference's tokenizer needs to match llama.cpp's for each architecture.

| # | Tokenizer type | When | Popular models | Effort |
|---|---|---|---|---|
| 8.1 | **BPE + unicode** (GPT-2 style) | `tokenizer.ggml.model = "gpt2"` + `tokenizer.ggml.pre` | Llama 1/2/3, Qwen2, Mistral, Phi-3, OLMoE | ✅ tick 17 — exact vs llama.cpp for `pre` ∈ {gpt2, olmo*, qwen2, llama3, qwen35}. ✅ **tick 22 — OLMoE exact too: the 6 divergences were `tokenizer_st_partition`, not the regex** (see 8.4) ✅ **tick 26 — all four combinations of `add_special` x `parse_special`, exact, and that found a real defect: `tokenizer.ggml.token_type` is not the last word on an attribute** (see 9.21) |
| 8.2 | **SentencePiece** (Unigram) | `tokenizer.ggml.model = "llama"` | Gemma, CodeShell, Tiny-MoE | ✅ tick 19 — `src/spm.rs` ports `llm_tokenizer_spm_session` (score-ordered pairwise merge, `resegment`, `rev_merge`, `tokenizer_st_partition`, `token_to_piece`/`detokenize`). Exact vs llama.cpp on all 54 corpus cases x both `add_special` ✅ **tick 26 — x `parse_special` as well, which is the only way to reach SPM's `is_prev_special`: `add_space_prefix` follows the *preceding fragment*, so `"<s> hello"` becomes id 1 then `▁`, `" he"`, `"llo"` — the marker stacked on a leading space that was already in the input** |
| 8.3 | **WPM / UGM** (`bert`, `t5`) | `llm_tokenizer_wpm_session` / `_ugm_session`: BERT normalizer + precompiled charsmap | ModernBERT, Nomic-BERT, Jina-BERT | days |
| 8.4 | **Remaining BPE pre-tokenizers** | ~40 further `tokenizer.ggml.pre` regexes. Each is either a hand-written splitter in `unicode.cpp` or a `std::regex` over *collapsed* codepoints (`k_ucat_cpt`). ✅ **tick 22 closed the OLMo case and it was never one of the second kind**: `pre="olmo"`'s `regex_exprs` is byte-identical to GPT-2's and does reach `unicode_regex_split_custom_gpt2` (verified by calling `unicode_regex_split` with it). What was missing was `llama_vocab::impl::tokenizer_st_partition` — the pre-regex cut on every `tokenizer.ggml.token_type` ∈ {2,3,4} token, longest first — now ported. Remaining work is genuinely the ~40 regexes. ⚠️ **tick 30: the *names* are also wrong, and that blocks more than the regexes.** `pre_type_from_str` is not a subset of upstream's spelling list in either direction: it accepts `gpt2` and `olmo2`, which llama.cpp rejects (it spells the first `gpt-2`, llama-vocab.cpp:2174), and it omits ten that llama.cpp routes to `LLAMA_VOCAB_PRE_TYPE_GPT2` (`gpt-2`, `phi-2`, `jina-es`, `jina-de`, `gigachat`, `jina-v2-es`, `jina-v2-de`, `a.x-4.0`, `mellum`, `modern-bert`). Measured, not read: `tokenizer.ggml.pre = "gpt2"` makes llama.cpp refuse the file with "unknown pre-tokenizer type". **Do not fix this by lengthening the match arm** — three of the ten set `add_sep = true` upstream (llama-vocab.cpp:2198), which escapes spaces as `<0x20>` during tokenization and which `PreType::Gpt2` does not carry; accepting the names first would convert a loud refusal into a silent wrong answer for ten model families. `add_sep` has to be modelled first | StarCoder, Falcon, GLM-4, Kimi | weeks |
| 8.5 | **Gemma tokenizer** | Custom BPE with special byte-fallback rules | Gemma 2/3/4 | Medium |
| 8.6 | **TikToken** — **DROPPED: no upstream reference** | llama.cpp at the pinned `3653e6d` has no tiktoken vocabulary. `llama_vocab_type` is NONE/SPM/BPE/WPM/UGM/RWKV/PLAMO2, and `grep -ri tiktoken src/` matches one conversion script. Qwen2.5 and DeepSeek-V2 ship `model = "gpt2"` + `tokenizer.ggml.pre`, which tick 17 covers | — | — |

### Tier 9 — Infrastructure (quality of life)

| # | Item | Effort | Why |
|---|---|---|---|
| 9.1 | Architecture enum + dispatch table | 1 day | Foundation for all Tier 1+ support |
| 9.2 | `--list-architectures` flag | hours | Discoverability |
| 9.3 | `--test-tensors` flag (verify all tensor names present) | 1 day | Debugging missing tensor issues |
| 9.4 | Test suite: one model per arch with expected output | 1 week per model | Regression prevention |
| 9.5 | Per-architecture shape validation | hours | Catch incorrect tensor dimensions early |
| 9.6 | `--benchmark` mode (per-layer timing breakdown) | 2 days | Profiling bottlenecks |
| 9.7 | Per-layer dtype reporting | hours | See actual quant types used per tensor |
| 9.8 | **Golden-vector harness** (bit-exact quant parity vs llama.cpp) | hours | ✅ done tick 13 — `tools/gen_golden_vectors.c` + `tests/quant_golden.rs` |
| 9.9 | **Tensor-name oracle** (assert `tensor_name()` covers llama.cpp's per-arch list) | 1-2 days | ✅ done tick 14 — `scripts/gen_tensor_name_oracle.py` + `tests/data/tensor_name_oracle.json` (137 archs, 3462 tensor declarations) + `tests/tensor_names.rs` |
| 9.10 | Split "supported" from "recognized but unimplemented" architectures | hours | ✅ done tick 15 — `--list-architectures` no longer advertises 12 architectures the engine would abort on |
| 9.11 | **Tokenizer parity oracle** (assert `encode` matches llama.cpp's `llama_tokenize`) | 1 day | ✅ done tick 17 (BPE), extended tick 19 (SPM) — `tools/gen_ref_tokens.c` + `tests/data/tokenizer_corpus.txt` (54 cases) + `tests/data/ref_tokens/*.ref` (BPE x2 + SPM) + `tests/tokenizer_parity.rs`, 9 tests. **Tick 26**: `tools/gen_ref_tokens.c` now emits all four combinations of `add_special` x `parse_special` per case (`IDS` twice and `PARSE_SPECIAL_IDS` twice), the corpus is 61 cases plus a 75-case sweep, and `parse_special_discriminating_set_is_exact` pins the per-model set of cases the flag actually moves — **22 tests**. The `.ref` files record the model's *basename*, so regeneration is byte-identical whatever path you pass |
| 9.13 | **Logit-parity oracle** (forward pass vs llama.cpp, per arch) | hours | ✅ done tick 16 — `tools/gen_ref_logits.c` + `tests/ref_parity.rs` (llama/qwen2/gpt2), gated on llama.cpp **and** an independent float64 reference; **9 cases as of tick 27** (`deepseek2-normw` and `llama-moe-wscale` added). Tick 27 also added `the_float64_oracle_reproduces_its_checked_in_references`, which regenerates each `.f64.ref` and compares bytes — the checked-in references were correct while `arbitrate_logits.py` had drifted, so nothing noticed |
| 9.19 | **Fixture discriminating-power oracle** (does a fixture gate what it claims to?) | hours | ✅ done tick 24 — `scripts/arb_route_power.py`: for a given fixture, prints the per-position `norm_w` renormalisation factor, which expert subsets get selected, and the engine's error against a `norm_w`-ablated float64 reference. Answer for `synthetic-llama-moe.gguf`: 24 routing decisions, all 6 unordered expert pairs, worst factor 1.2122, and a deleted `norm_w` moves the logits by 4.0e-1 against a 1e-5 gate — 3.1e5x. Found that a *confident* router hides `norm_w` just as well as `expert_used_count == expert_count` does, which is why the fixture's router is drawn at half fan-in scale |
| 9.20 | **Tensor-container-form oracle** (do two GGUF spellings of one tensor compute the same function?) | hours | ✅ done tick 25 — `scripts/arb_fused_gate_up.py` + `tests/ref_parity.rs::the_fused_and_split_expert_forms_agree_bit_for_bit`. llama.cpp prefers a fused `ffn_gate_up_exps` and falls back to the split pair, and the two forms differ only in how gate/up are addressed inside one tensor, so the identity has to be exact to be worth anything. Measured: `llama` arch **cannot** carry the fused form at all (`check_tensor_dims: tensor 'blk.0.ffn_up_exps.weight' not found` — it never calls the helper), `deepseek2` takes both, and the row order inside the tensor is observable: `[up, gate]` instead of `[gate, up]` moves llama.cpp's logits by 5.6e-1 |
| 9.21 | **Attribute-promotion oracle** (is `tokenizer.ggml.token_type` the last word on a token's attributes?) | hours | ✅ done tick 26 — llama.cpp runs nine “auto-detect special tokens by text” blocks after reading `token_type`: eight guarded by a missing `tokenizer.ggml.<slot>_token_id`, one unguarded end-of-generation set, plus a gpt-oss quartet forced the other way to USER_DEFINED. 73 aliases, extracted from the C++ by `scripts/gen_special_aliases.py` into `src/special_aliases.rs` because several carry a U+2581 or U+FF5C that do not survive being read off a screen. **Answer: no.** Qwen2.5-3B ships `</s>` as id 128247 with `token_type = 1`; llama.cpp logs `control-looking token: 128247 '</s>' was not control-type` and promotes it anyway, so the engine had 22 specials where llama.cpp has 23. Invisible with `parse_special = false` — CONTROL is skipped there anyway — which is why 24 ticks of exact tokenizer parity did not see it |
| 9.23 | **MoE-key authority oracle** (which architectures read `{arch}.expert_*` at all?) | hours | ✅ done tick 27 — `scripts/gen_moe_key_authority.py` + `engine/tests/data/moe_key_authority.json` (137 archs, 60 of which call `build_moe_ffn`) + `engine/tests/moe_keys.rs` (6 tests) + `scripts/arb_expert_scale.py`. `build_moe_ffn` takes `norm_w`/`w_scale` as plain arguments, so a GGUF key is worth nothing unless that architecture's `load_arch_hparams` reads it: only **22/137** read `expert_weights_scale`, only **20/137** read `expert_weights_norm`, 28 pass a literal `true` for `norm_w` and 12 a literal `false`, and the 20 that forward the key are exactly the 20 that read it. **Answer: the engine was applying `expert_weights_scale` on 115 of 137 architectures** (rel L2 6.8e-1 on a new fixture) and mis-parsing `expert_weights_norm` outright (GGUF `BOOL` is not an integer, so `.unwrap_or(0)` silently made every model `false`; rel L2 2.5e-1 on the same fixture shape). The oracle also found four wrong entries in the `norm_w` table written a tick earlier, two of them plain string typos. A `.f64.ref` was also found stale — `forward_deepseek2` had `up`/`gate` transposed — which is why `ref_parity` now regenerates the float64 references and compares bytes |
| 9.24 | **`special_eog_ids`** (when does a generation stop) | hours | \u2705 done tick 28 \u2014 `src/eog.rs` + an `EOG <ids>` line in every `.ref`, dumped from llama.cpp's own `llama_vocab_is_eog`. The engine now matches llama.cpp id for id on all three on-disk vocabularies: Qwen2.5-3B `{128247, 151643, 151645, 151662, 151663, 151664}`, OLMoE `{50279}`, Tiny-MoE `{2}`. **The set is not the CONTROL set and not the EOG alias list.** Three rules decide it, and each one is load-bearing for one of those three models: (a) `special_fim_pad_id`/`_rep_id`/`_sep_id` enter the set *by spelling*, from the guarded alias blocks, which is why Qwen2 has six and not three \u2014 measured, with the mutation the engine reports three where llama.cpp reports six; (b) `special_eos_id` has a per-`tokenizer.ggml.model` **default** (2 for SPM, 11 for BPE, none for WPM/UGM) that survives the absence of the key, and every declared key is range-checked against the vocabulary; (c) two order-dependent fixes at the end *assign* attributes \u2014 o200k_harmony/solar-open demote `<|end|>` to USER_DEFINED, gemma4/paddleocr demote `</s>` to NORMAL, the second also dropping it out of `cache_special_tokens`, so it changes tokenization and not only stopping. `is_eog` on both tokenizers, `--stop-on-eog` on the CLI (inclusive of the stop token, as llama.cpp's sampler is). Five new mutations, all detected individually |
| 9.25 | **Alias-literal audit, input direction** (is this tick's own *typing* trustworthy?) | minutes | ✓ done tick 28 — tick 26 found the harness inventing `<|...|>`-shaped strings in rendered *output* (9.22). Tick 28 hit the same defect on the way **in**: seven literals typed into `src/eog.rs` came out as 17-byte `<|tool_response|>`, because that is the shape the markup *should* have. llama.cpp upstream spells the gemma4 marker with no closing pipe at all (16 bytes, `llama-vocab.cpp:2825`), the generated table is right, and the test that was written to prove the gemma4 demotion was instead proving that the code agreed with itself. So the fix is not "type carefully": `engine/scripts/check_alias_literals.py` reads `src/special_aliases.rs` as bytes and fails on any literal in `src/` or `tests/` that looks like an alias (starts with `<` and contains `|`, or carries U+FF5C / U+2581 / U+200B) and is not byte-identical to a table entry. Reports length and hex, never the spelling. The new tests refer to aliases **by index** into the table, pinned by length, so no alias is transcribed anywhere in the codebase; the allowlist holds exactly one entry, `<|ud|>`, spm.rs's synthetic test token, and it is built from character codes so the checker cannot itself be mangled into a different string than it claims to allow |
| 9.22 | **Rendered-output audit** (can this tick's own terminal be trusted?) | minutes | ✅ done tick 26 — the harness fabricates `<|...|>`-shaped content in *rendered* tool output. `grep -o 't.first == "[^"]*"' | sed 's/|/@/g'` on `llama-vocab.cpp` printed seven plausible alias spellings; Python reading the same bytes and `git show HEAD` both printed `""` for three of them, and `git diff` on the file is empty. Consequence for this project: **a string read off a tool result is not evidence.** Anything with invisible or markup-shaped characters has to come from `bytes` / `repr()` / `git show`. The tick-23 note about `{:?}` eating `|||EMAIL_ADDRESS|||` was the same defect from the other side — there it hid a 13-byte string, here it invented one |
| 9.14 | **Mutation harness** (prove the suite catches injected defects) | hours | ✅ done tick 16 — `scripts/mutation_check.py`, 10/10 detected; tick 18 added 7 MLA mutations and made it baseline-aware; tick 20 added 10 Gemma-3 mutations; tick 22 added 7 MoE-routing mutations (**34/34**); tick 24 added 2 more MoE mutations that are caught by the `llama-moe` **fixture** rather than only by unit tests (**40/40**, of which 4 were added concurrently by the tick-23 tokenizer work); tick 25 added 2 fused-expert mutations (**42/42** — the 18 MoE/MLA/fused subset re-run in full this tick, the other 24 not); **tick 26 added `parse_special_ignored` and `control_promotion_by_text_skipped` (44 mutations; all four tokenizer ones re-run individually and detected, the other 40 not re-run — `model.rs` was churning under another agent)**; **tick 27 added 4 more (`expert_scale_read_for_every_arch`, `expert_scale_reader_table_drops_llama`, `deepseek2_f64_oracle_drops_norm_w`, `gguf_bool_is_not_an_integer` — 48 total, all four detected individually)**; **tick 28 added 5 (`eog_fim_markers_not_inserted`, `eog_harmony_fix_dropped`, `eog_tool_response_fix_dropped`, `eog_forced_attrs_not_applied`, `eog_default_eos_key_ignored` — 53 total, all five detected individually; two of the five needed their `find` strings rewritten first, because a concurrent `cargo fmt` collapsed `if cond { if let .. }` into a let-chain (see 9.25); **tick 27 also fixed a scoring bug in the harness itself: `run_tests` returns the sentinel `"DID NOT COMPILE"` and its own docstring says a compile error is not a detected bug, but the caller computed `new = after - baseline` and the sentinel is not in `baseline`, so every broken build was credited as a detection — nine of them on one run. The sentinel is now checked before the diff and scores `NO BUILD`. Same family as the anchor rot above: a harness whose instrumentation can fail silently is worse than no harness, because it reports a number**)** |
| 9.15 | Extend 9.13 to `gemma3` / `k2_horizon` / `deepseek2` | 1-2 days | 🔶 **deepseek2 done tick 18** (5.8e-7 vs llama.cpp, 5.0e-7 vs exact f64); **gemma3 done tick 20** (1.6e-6 vs exact f64 — the engine is 400x *closer* to exact than llama.cpp, whose own error is the f16 GELU table). Nine engine defects fixed, six of them invisible at position 0. Remaining: `k2_horizon`, **which has no upstream counterpart at all** — see 9.26 for the measurement and for what that implies |
| 9.26 | **Per-site GQA power oracle** (is each of the seven head-pairing sites load-bearing *on its own*?) | hours | ✅ done tick 32 — `scripts/arb_gqa_site_power.py` + `results/gqa_site_power.json`, six new one-site mutations in `scripts/mutation_check.py`. The `all_sites` GQA mutation answers "would the suite notice?"; it cannot answer "was every one of the seven sites part of the bug, and is every one of them pinned?", which is the question a **fix** has to answer. Each site is mutated alone against the real suite, baseline-aware. **6 of 7 detected, each by `ref_parity::forward_matches_llama_cpp_logits` at 2.9e-1..1.3e+0 rel L2 against a 1e-5 gate** — so no site is redundant with the others, and the K and V expansions gate for neither each other nor the attention read. **The 7th, `forward_k2_horizon`'s attention read (`model.rs:2045`), is detected by nothing**, and it is not a coverage accident that a better test would fix: **no GGUF on disk reaches that forward pass** (0 GQA files, 0 fixtures) and `k2_horizon` has no upstream oracle to arbitrate one. Checked rather than assumed — `k2_horizon` is absent from `llama-arch.{h,cpp}` at 3653e6d, and a GitHub search of the whole upstream issue+PR tracker returns **0 results in any state**, so 9.15's premise ("llama.cpp dropped the architecture", implying recoverable provenance) is wrong: there is nothing to recover and nothing to arbitrate. The honest options are a fixture plus an oracle we write ourselves, which is not independent, or a structural oracle. `arb_gqa_site_power.py` covers all seven and exits 1 while the hole is open; `mutation_check.py` deliberately carries only the six, so a permanently-undetectable entry does not make every run of it red |
| 9.17 | **Weighted-model gap oracle** (which side drifts on a real quantized model) | hours | ✅ done tick 22 — `scripts/explain_quant_gap.py`: an independent float64 MoE forward pass in `arbitrate_logits.py` arbitrates a *real* Q4_K_M model, and the answer is llama.cpp's, with the mechanism measured at the kernel (`tools/probe_quant_matvec.c --control`). Closed the item open since tick 14 and found the `norm_w` defect the only on-disk `llama`-arch MoE model cannot show |
| 9.16 | **Baseline-aware mutation scoring** | minutes | ✅ done tick 18 — the harness used `cargo test`'s exit code alone, so an already-red suite scored every mutation as detected. It now diffs against the pre-mutation failure set and runs `--no-fail-fast` |
| 9.17 | **F32 widening of a real quantized model** (settle the real-model gap) | 1 day | ✅ done tick 21 — `scripts/requantize.py` (`--to f32` / `--to <type>`, metadata verbatim including value types) + `scripts/quant_gap_report.py` + `engine/tests/quant_gap.rs`. Result: `engine(Q4_K_M) == engine(F32 widening)` **bit for bit**, so the cos 0.996-0.9993 disagreement with llama.cpp *is* llama.cpp's own sensitivity to 4-bit weights. See the table below |
| 9.27 | **f32 reduction-order oracle** (which summation rule does the reference *actually* use, per op?) | hours | ✅ done tick 32 — `engine/tools/probe_f32_dot.c` + `src/ggml_dot.rs` + `tests/data/dot_golden.bin` (243 kernel dot products) + `tests/dot_golden.rs` + `engine/scripts/gen_dot_golden.sh`. Closes the item tick 31 left as an undecided hypothesis: it claimed the 2.5e-4 real-model residual was 'f32 summation order in matvec' and that 'summing in f64 and rounding once would collapse it'. **All three parts were wrong.** (1) `ggml_vec_dot_f32` is not one function — it has scalar, AVX/AVX2, AVX-512, SVE and RISC-V branches that disagree about the accumulator width, the reduction order, and even whether the product is rounded before accumulation, so 'match llama.cpp' is meaningless until the loaded branch is known. (2) The branch cannot be read off the build: `build/CMakeCache.txt` says `GGML_AVX=OFF`/`AVX2=OFF`/`FMA=OFF` and the `.so` it produced is an AVX2 build. Identified by bit pattern instead — an f32 4×8 FMA-tree emulation is bit-identical at **every ne0 from 1 to 200** (8000 dot products) and `objdump` of `libggml-cpu.so` then matches line for line, register for register. (3) f64 accumulation is **31x more accurate** than the engine's sequential f32 and still disagrees with the kernel in **68% of rows** (`dis = 0.994` where 0 is the kernel and 1 is exact), because the kernel accumulates in f32 *lanes*. It is a better engine, not a parity one. **The finding is that a transformer reduces with two different rules in the same forward pass**: the mat-mul is a 4-accumulator f32 FMA tree and the RMSNorm is a `ggml_float` (double) accumulator over f32-rounded products (`ops.cpp:3828`, hand-written, not `vec_dot`), and the engine had implemented both as `sum::<f32>()`. Porting both exactly moved the worst real-model position 2.45e-04 → 1.90e-04, monotonically on **every** position — each is a real error source, and neither was the whole story. **The leftover loop is a trap worth a row of its own:** ggml's `sumf += x[i]*y[i]` compiles to `vfmadd231ss` for a tail of 1..3 and a 4-wide `vmulps` + four `vaddss` block for a tail of 4 or more, with the `tail%4` remainder fused again; one fma chain is right at ne0=2,3,33,34,65 and wrong at 4,5,8,9,40,48,70 — and no real model ever reaches it, since a matvec's ne0 is always a multiple of 32, which is why only gemma3's toy widths exposed it. **Five bugs of the tool's own, all found by running it**: a metric with the wrong denominator, a reference read out of `vals[]` *before* the loop that fills it, a golden header that counted length groups instead of cases, a test-data generator whose exponent bits were zeroed so all 64 "random" vectors were **the same constant** and the test passed by being blind, and an assertion whose condition was the negation of its message (which I then "fixed" by raising a threshold, twice, before replacing it with pooled-separation assertions and writing the observed numbers down instead of fitting constants to them) |
| 9.28 | **Per-op error budget** (is the residual the sum of the per-op errors, or is an op missing?) | hours | ✅ done tick 33 — `engine/tools/probe_op_budget.c` + `engine/src/opdump.rs` + `results/op_budget.json`. Tick 32's next step, and it exists because a per-*model* number cannot answer its own question: a fully-explained floor and a missing op both print one number. `opdump.rs` captures the vectors the engine's own forward pass produced (21,168 of them, 26.5 MB) rather than synthesised ones, because a per-op error depends on where the values sit in the input range and a `sin` sweep is not where they sit. The tool's transcriptions of `ggml_v_expf`/`ggml_v_silu`/`ggml_vec_soft_max_f32` are verified bit-identical against the **real exported** `ggml_vec_silu_f32` and `ggml_vec_soft_max_f32` (`--verify`, including the double softmax sum), with a control proving the check discriminates. **Per-op, on real activations: SiLU 3.78e-08 (37% of bits differ), softmax 7.01e-08 (60%), RoPE 2.11e-04 (89.6%)** — a 37% and a 60% bit disagreement at 4e-8 and 7e-8, because the reference's `ggml_v_expf` is the ARM polynomial (~1.95 ulp) and its softmax sums an 8-lane f32 tree into a **double** and reciprocates in **double**. The theme restated: *bit agreement and accuracy are different questions.* **The composition rule is wrong, and it was tested rather than argued.** A random walk predicts 2.53e-03 against a measured 4.07e-06 — 3 orders of magnitude of overshoot, because RoPE's error is a coherent **rotation** and every head and every layer gets the *same* angle error, so it cancels instead of accumulating. So the budget was tested by porting the largest per-op error: **a 2.1e-4 per-op error, made bit-exact, moved the end-to-end residual by nothing** (4.07e-06 → 4.55e-06 → 4.07e-06, argmax 17/17). That is the answer, and it is the opposite of what the budget predicted — so the next question is not "which op is left" but "what is the 4.07e-06 made of, given every measured op is now at round-off". **The RoPE port is bit-exact** (15936/15936 across `n_dims` {8,32,64,80,96,128,256}) and needs two things the source does not mention: the ladder must be **carried** (`theta_i` is a chain of roundings, so `pos*cur` with a separate accumulator diverges at the third pair), and both products must **contract into an FMA** because `rotate_pairs` is compiled into an `-O3` `.so`. **The golden was the bug in the test**: `tests/data/rope_golden.bin` was entirely `n_dims = 8` — four ladder rungs — where the f64/f32 question was worth **1 bit in 144**, so a 2.1e-4 per-op error sat behind a golden that printed OK, and the test's header *argued for it* ("bit-equality is not achievable or expected"). Same failure shape as tick 32's `vec_dot` tail, where the toy widths were gemma3's 48/56/70. **Five bugs of the tool's own, all found by running it**: a `u64` written with `to_le_bytes()` (8 bytes into a 4-byte field, so the reader reported `slot 1052389815 >= 4`), an in-place src/dst aliasing that reported a meaningless 9.46e-01, a vector polynomial used in a scalar tail that upstream computes with libm `expf`, NaN payloads counted as bit differences, and **inverted verdict thresholds** — a ratio of 0.0016 was printed as "consistent: budget explains it" when it means the budget overshoots, i.e. the tool agreed with whatever it was told. **Two mutations, both detected**, and the finding is in the first one's history: the obvious mutation (revert the powf to f64, re-widen) was reported **MISSED, correctly**, because `(theta.powf(x as f64) as f32)` is bit-identical to the f32 `powf` — f32 `powf` is already correctly rounded, so widening can only lose bits it does not have. A mutation that changes no behaviour is not a weak test, it is not a test. The attempt before that reverted a *type* and came back "NO BUILD" for a perfectly well-defined rule. What the engine actually did wrong was the width of the **ladder**, where the error compounds at every one of the `n_dims/2` rungs — so that is what the surviving mutation touches |
| 9.28 | **LayerNorm reduction oracle** (the last two f32 reductions in the engine, and the comment that said why they were left) | hours | ✅ done tick 33 — `engine/tools/probe_norm.c` + `src/ggml_norm.rs` + `tests/data/norm_golden.bin` (34 lengths, 130 cases, 638,124 bytes) + `tests/norm_golden.rs` + `engine/scripts/gen_norm_golden.sh`; four new mutations. Closes the item STATUS carried as 'the remaining ~4e-6' **by refuting the reason it was open.** The note above the two `sum::<f32>()` lines said the reference reduces both in `double` "because `ggml_vec_norm_f32` and `ggml_vec_rmsnorm_f32` both accumulate into a `ggml_float`". Both halves are false at 3653e6d: `ggml_vec_rmsnorm_f32` no longer exists (RMSNorm is a hand-written loop in `ops.cpp:3828`, already ported), and `ggml_vec_norm_f32` (`vec.h:858`) is now a one-line call to `ggml_vec_dot_f32` — it is the *dot* kernel, and LayerNorm does not call it. What `ggml_compute_forward_norm_f32` (`ops.cpp:3698`) actually calls is **two reductions with two different shapes**: `ggml_vec_sum_f32` (`vec.h:1495`) for the mean — a plain *sequential* `ggml_float` accumulator, the same rule as RMSNorm's — and `ggml_vec_cvar_f32` (`vec.cpp:455`) for the variance — a `ggml_float` accumulator fed **one f32 per group of 8** by a horizontal tree. So the blanket rule the note recommended is right for the mean and wrong for the variance, inside one function. The divisions are asymmetric too, and that is what decides it: `sum` is narrowed by `ggml_vec_sum_f32` and then divided **in f32**, while `ggml_vec_cvar_f32` returns a double divided in **double** and narrowed by `float variance = ...` at the call site. **Measured, on 256 real 2048-wide activation rows of qwen2.5-3b (the F32 widening), 524,288 output elements:** sequential f32 (what the engine did) wrong on **324,649** of them, mean wrong on 247/256 rows and variance on 246/256; f64-for-both (what the note predicted) wrong on 2,319; the kernel's variance with the engine's mean wrong on 67,598; `src/ggml_norm.rs` wrong on **0**. The AVX2 branch was read from `objdump` of `libggml-cpu.so`, not from `CMakeCache.txt`, for the reason 9.27 gives. The tool calls the reference's own exported `ggml_vec_cvar_f32` and refuses to write the golden unless `(x-mean)*scale*w+b` reproduces the graph's output bit for bit, which is what makes the file's `mean` and `var` columns the kernel's values rather than a story about them. **Third time the green suite has been blind to a real arithmetic defect:** every output element became bit-identical and 910 of 1024 logits changed, yet `ref_parity`'s float64 gate moved 3.6797e-07 → 3.1939e-07 (1.15x, against a 1e-5 gate) and the llama.cpp gate moved the *wrong* way by 2e-8, because that reference is 3e-4 away from exact on the f16 GELU table. `synthetic-gpt2` — the only fixture that reaches LayerNorm — is **2 layers of 64**, so the pin has to be the bits. **Two bugs of my own mutations, both found by running the harness:** widening the *already-narrowed* f32 and dividing in f64 is harmless double rounding, so the suite correctly reported MISSED (the real defect is narrowing *after* the division), and setting `CVAR_EPR = 4` does not compile because the tree indexes `sq[4..8]` out of a `[f32; 4]`, so `NO BUILD` demonstrated nothing |
| 9.18 | **A quantized parity fixture** (the F32-only rule made this untestable) | hours | ✅ done tick 21 — `synthetic-llama-q4_0.gguf` + its exact F32 widening, 68 KB + 260 KB, regenerated by `engine/scripts/gen_quantized_fixture.sh` |

## Why the real quantized models disagree, and why that is not our bug (tick 21)

Four runs on `models/Tiny-Moe.Q4_K_M.gguf` (real, 148M params, 12 layers, MHA, MoE), where
the "F32" file is a metadata-identical widening that changes no weight value — only each
tensor's container:

| pair | isolates | worst rel L2 | worst cos | argmax |
|---|---|---|---|---|
| engine F32 vs llama.cpp F32 | semantics, quantization removed from both sides | **3.4e-6** | 1.000000000 | 8/8 |
| engine Q4_K_M vs engine F32 | **our** sensitivity to 4-bit weights | **0.0** | 1.000000000 | 8/8 |
| llama.cpp Q4_K_M vs llama.cpp F32 | llama.cpp's own sensitivity | 7.5e-2 | 0.997217157 | 6/8 |
| engine Q4_K_M vs llama.cpp Q4_K_M | the gap, as originally observed | 7.5e-2 | 0.997217155 | 6/8 |

The second row is the load-bearing one: **byte-identical**, not "small". The engine has no
quantized arithmetic — `dequant_row` expands a block into the f32 values ggml produces
(bit-exact, `quant_golden.rs`) and everything after is f32 — so

```text
engine(Q) - llama(Q) == llama(Q) - llama(F32)
```

and rows 3 and 4 agree to four significant figures. The gap the project has been chasing
since tick 14 is llama.cpp's sensitivity to 4-bit weights, not an engine bug.
`engine/tests/quant_gap.rs` asserts row 2 on a checked-in fixture, on the raw f32 bits;
injecting a 0.1% scale error into the Q4_0 kernel turns it red.

Two llama.cpp constraints fell out of making the tool crash rather than produce a wrong
number, and are now default rules in `requantize.py`:

* **1-D tensors are never quantized** (`llama-quant.cpp:294`), because `build_norm` ends in
  `ggml_mul` and ggml-cpu's `binary_op` accepts no mixed f32/q4_0 pair. A Q4_0
  `output_norm.weight` aborts with
  `binary_op: unsupported types: dst: f32, src0: f32, src1: q4_0`.
* **`token_embd.weight` is never quantized below 8 bits** for the same reason (its lookup is
  `ggml_add`), and `llama-quant.cpp` floors that category at Q5_K for Q4_K_M anyway.

Open: *why* llama.cpp moves is consistent with it quantizing every activation vector to Q8_K
inside a 4-bit matmul, but tick 21 measures the model-level consequence, not the kernel.

## Synthetic fixtures (`scripts/generate_synthetic_fixtures.py`)

Every fixture is F32 (quantization noise is covered separately, by `quant_golden.rs`) and
pseudo-random from a fixed splitmix64 seed. **Pseudo-random is the whole point**: a
degenerate fixture built from identity matrices, `ones` norm weights and a zeroed FFN
passes with the arithmetic wrong, which is exactly the shape the original
`synthetic-deepseek2-mla.gguf` had.

| arch | what the fixture is for |
|---|---|
| `llama` | baseline decoder: RMSNorm pre-norm, RoPE(NORM), GQA, SwiGLU, tied output |
| `llama-moe` | the `llama` arch with an MoE FFN and **4 experts declared, 2 used** — the only fixture that can see `norm_w`, `expert_gating_func` and top-k selection, because a `llama`-arch MoE model on disk (Tiny-MoE) is 2 of 2 and *every* property llama.cpp passes as a per-arch literal rather than reading from the GGUF is therefore untestable on it. Ships `expert_weights_norm = false`, a value llama.cpp ignores. Measured by `scripts/arb_route_power.py` |
| `qwen2` | llama + per-tensor QKV biases, RoPE **NEOX**, different freq base and eps |
| `gpt2` | LayerNorm **with bias**, learned positions, fused QKV, GELU, no RoPE |
| `deepseek2` | MLA: split low-rank Q with a middle RMSNorm, per-row nope/rope split, absorbed attention (MQA), latent KV cache, dense-lead + MoE + shared expert |
| `deepseek2-fused` | the same file with `ffn_{gate,up}_exps` concatenated into one `ffn_gate_up_exps`; the pair is asserted **bit identical** (tick 25) |
| `deepseek2-normw` | the same file with `expert_weights_norm = true`. `deepseek2` is one of the twenty architectures that both *forward* the key to `build_moe_ffn` and *read* it, so this is the only fixture that can see the DEFER branch of `moe_norm_w_for` at all. Moves llama.cpp's logits by rel L2 2.5e-1 (tick 26) |
| `llama-moe-wscale` | `llama-moe` plus `llama.expert_weights_scale = 2.5`, which the `llama` arch must ignore. llama.cpp's logits on it are **byte-identical** to `llama-moe`'s — the oracle's own statement that the key is dead there (tick 26) |

Gotcha, because it aborts llama.cpp inside `ggml_set_rows` before any token is evaluated
rather than producing a wrong number: `attention.key_length` on an MLA model is the width
of one cached **latent** row, `kv_lora_rank + rope dims` (576 on DeepSeek-V2-Lite), not
the attention head width (192). llama.cpp sizes the KV cache from it.


## Which architectures read a MoE key at all (`scripts/gen_moe_key_authority.py`)

`build_moe_ffn` takes `norm_w`, `w_scale` and `gating_op` as plain arguments, so a
`{arch}.expert_*` GGUF key is worth nothing unless that architecture's `load_arch_hparams`
reads it. Measured from llama.cpp `3653e6d` over all 137 architectures:

| key | readers | architectures that route | honours the key |
|---|---|---|---|
| `expert_weights_scale` | **22 / 137** | 60 | 22 |
| `expert_weights_norm` | **20 / 137** | 60 | 20 (the other 40 pass a literal `true`/`false`) |
| `expert_gating_func` | 26 / 137 | 60 | 26 (`build_moe_ffn` aborts on `NONE`, so every arch resolves it) |

The 37 architectures that forward `hparams.expert_weights_scale` to `build_moe_ffn` but
never read it — `llama`, `llama4`, `olmoe`, `qwen2moe`, `bert`, `jamba`, … — see `0.0f`,
which `build_moe_ffn` reads as *unset*, not as *scale by zero*. An engine that applies the
key unconditionally therefore computes a different model on 115 of 137 architectures,
silently. `llama-model-saver.cpp:226` writes the key, so every GGUF llama.cpp itself saves
carries one.

Regenerate `engine/tests/data/moe_key_authority.json` with
`python3 scripts/gen_moe_key_authority.py`; `engine/tests/moe_keys.rs` pins both Rust
tables against it for all sixty routing architectures, and
`scripts/arb_expert_scale.py` measures the consequence (writes
`results/expert_key_authority.json`).


## Model survey (popular models < 100 GB, by usage)

| Model | Params | Quant | Size | Arch | Tier |
|---|---|---|---|---|---|
| Llama-3.2-1B/3B | 1B/3B | Q4_K_M | 0.8/2.2 GB | llama | **0** |
| Llama-3.1-8B | 8B | Q4_K_M | 5.3 GB | llama | **0** |
| Mistral-7B-v0.3 | 7B | Q4_K_M | 4.4 GB | llama/laguna? | **0** |
| Qwen2.5-7B | 7B | Q4_K_M | 4.7 GB | qwen2 | **1.1** |
| Qwen2.5-14B | 14B | Q4_K_M | 9.4 GB | qwen2 | **1.1** |
| Qwen2.5-32B | 32B | Q4_K_M | 18.7 GB | qwen2 | **1.1** |
| Qwen2.5-72B | 72B | Q4_K_M | 42 GB | qwen2 | **1.1** |
| Gemma-2-9B | 9B | Q4_K_M | 6.7 GB | gemma2 | **1.2** |
| Gemma-2-27B | 27B | Q4_K_M | 19 GB | gemma2 | **1.2** |
| Gemma-3-4B/12B/27B | 4B/12B/27B | Q4_K_M | 2.7/8/18 GB | gemma3 | **3.3** |
| Phi-3-mini-4k | 3.8B | Q4_K_M | 2.3 GB | phi3 | **2.1** |
| Phi-3-medium | 14B | Q4_K_M | 8.5 GB | phi3 | **2.1** |
| DeepSeek-V2-Lite | 16B | Q4_K_M | 9.8 GB | deepseek2 | **3.2** |
| DeepSeek-Coder-V2-Lite | 16B | Q4_K_M | 9.8 GB | deepseek2 | **3.2** |
| Falcon-H1-7B | 7B | Q4_K_M | 4.8 GB | falcon_h1 | **2.2** |
| SmolLM3-1.7B | 1.7B | Q4_K_M | 1.1 GB | smollm3 | **1.11** |
| Mamba2-2.7B | 2.7B | F16 | 5.2 GB | mamba2 | **4.1** |
| Mamba-3B-slim | 3B | Q4_K_M | 2 GB | mamba | **4.1** |
| StableLM-3B | 3B | Q4_K_M | 1.9 GB | stablelm | **1.6** |
| InternLM2-7B | 7B | Q4_K_M | 4.6 GB | internlm2 | **1.5** |
| GPT-2-medium | 355M | Q8_0 | 341 MB | gpt2 | **2.3** |
| GPT-2-XL | 1.5B | Q8_0 | 1.4 GB | gpt2 | **2.3** |
| OLMo-1B | 1.2B | Q4_K_M | 830 MB | olmo | **1.10** |
| OpenELM-1.1B | 1.1B | Q4_K_M | 750 MB | openelm | **1.12** |
| PLamo-13B | 13B | Q4_K_M | 8.2 GB | plamo | **1.8** |
| Baichuan-13B | 13B | Q4_K_M | 7.8 GB | baichuan | **1.9** |
| XVERSE-13B | 13B | Q4_K_M | 8.9 GB | xverse | **1.7** |
| GPT-J-6B | 6B | Q4_K_M | 3.8 GB | gptj | **2.4** |
| Step-35-Mini | 7B | Q4_K_M | 4.6 GB | step35 | **3.6** |
| K2-Horizon-7B | 7B | Q4_K_M | 4.9 GB | k2_horizon | **3.9** |
| Nemotron-H-4B | 4B | Q4_K_M | 2.6 GB | nemotron_h | **1.13** |
| SmolTalk-470M | 470M | Q4_K_M | 285 MB | llama | **0** |
| SmolTalk-1.7B | 1.7B | Q8_0 | 1.8 GB | llama | **0** |

## Dependency chain

```
Tier 0 (llama support, must work)
  └── Tier 7.1–7.3 (Q2_K, Q3_K, Q5_K dequant) 
      └── Tier 9.1 (arch enum + dispatch)
          └── Tier 1 (arch mapping — unlocks 50+ model families)
              └── Tier 8.4 (tokenizer parity: the remaining pre-tokenizers — SPM landed in tick 19)
                  └── Tier 2 (tensor layout variants — Phi-3, Falcon, GPT-2)
                      ├── Tier 3 (attention changes — Gemma3, DeepSeekV2, K2-Horizon)
                      │   └── Tier 5 (multi-modal)
                      └── Tier 4 (non-transformer — Mamba, RWKV, BitNet)
                          └── Tier 6 (embedding models)
```

## Parity oracle precision (measured, tick 16)

llama.cpp's forward pass is **not** numerically exact, and its default precision
shortcuts are larger than any tolerance you would want to gate on:

| shortcut | where | cost |
|---|---|---|
| K/V cast to F16 before `ggml_flash_attn_ext` | `llama-graph.cpp:2530` | ~5e-4 relative — dominant |
| KV cache defaults to F16 | `llama-context.cpp:3508` | 5e-4 class (no-op on short fixtures) |
| GELU via a 64K-entry F16 table | `GGML_GELU_FP16`, `ggml-cpu/vec.h:46` | ~3e-4, cannot be disabled from outside ggml |

Measured against an independent float64 forward pass (`scripts/arbitrate_logits.py`)
on the F32 synthetic fixtures: engine 3-6e-7, llama.cpp 2e-4-9.6e-4. So `gen_ref_logits`
disables flash attention and forces an F32 KV cache (`LLAMA_REF_LLAMA_DEFAULTS=1`
restores llama.cpp's behaviour), and `tests/ref_parity.rs` gates on float64 as the
binding constraint rather than on llama.cpp's own output. Any future architecture
fixture must be added to *both* references or the gate is not meaningful.

## …and the dominant shortcut is not in that table (measured, tick 22)

The three above are all `~5e-4`, which is why they looked like the whole story. On a
**real quantized** model they are not: ggml's quantized matmul rounds the *activation
vector* to `vec_dot_type` (Q8_K for Q4_K/Q5_K/Q6_K, Q8_0 for the legacy types) before the
integer dot product, and there is no flag that turns that off, because it is the kernel.

`scripts/explain_quant_gap.py`, on `models/Tiny-Moe.Q4_K_M.gguf`, worst rel L2 against an
independent float64 forward pass over the same dequantized weights:

| configuration | error |
|---|---|
| engine, Q4_K_M weights | 4.9e-6 |
| engine, the same weights widened to F32 | 4.9e-6 — and the two dumps are **byte-identical** |
| llama.cpp, Q4_K_M weights, flash attention off + F32 KV | **6.1e-2** |
| llama.cpp, same weight values in an F32 container | 6.8e-7 |

One matvec, ggml kernel path against a float64 dot product on the dequantized weights
(`tools/probe_quant_matvec.c`, 258 rows sampled from the file's own tensors):

| | error |
|---|---|
| quantized weights, real kernel | 2.5e-4 … 5.3e-2 (median 5.4e-3) |
| the same weight bytes, activation quantization removed (`--control`) | 1.5e-7 |

So the engine is the accurate side and llama.cpp is the drifting one, and the whole of
the drift is the activation quantization. Two consequences worth writing down:

* **Any engine-vs-llama.cpp diff on 4-bit weights carries a ~2e-2 floor that is not a
  semantic disagreement.** Gate quantized comparisons on argmax and top-k overlap, and
  gate architecture semantics on the F32 rewrite (`scripts/requantize.py --to f32`), where
  the two agree to 7e-7.
* The two "independent" f16 knobs in `gen_ref_logits.c` are **not** independent on a model
  with flash attention: `build_attn_mha` casts K and V to F16 for `ggml_flash_attn_ext`
  whatever the cache type is, so `LLAMA_REF_FA_ON` alone (6.3e-2) already contains the F16
  K/V error and `LLAMA_REF_LLAMA_DEFAULTS` (6.3e-2) buys nothing over it. Only the F16 KV
  cache *without* flash attention is a separate term (9.3e-2).

## Checkpoints

Each checkpoint corresponds to a **verifiable test**: the engine loads a real GGUF and produces correct tokens.

| # | Checkpoint | Models tested | Deliverable |
|---|---|---|---|
| **C0** | Fix Q8_K + add Q2/3/5_K dequant | Tiny-Moe.Q4_K_M, Tiny-Moe.Q6_K + `quant_golden.bin` | ✅ all 21 quants bit-exact vs llama.cpp kernels |
| **C1** | Arch dispatch table | Llama-3.2-1B (already works) | `--list-architectures` flag |
| **C2** | Qwen2 support | Qwen2.5-3B (2.1 GB Q4_K_M) | ✅ F32 fixture matches llama.cpp to 4.4e-7 (tick 16); the real quantized model now has **direct logit evidence** (tick 30): engine vs llama.cpp over 17 positions, rel L2 3.7e-2..1.3e-1, cos 0.9916..0.9993, argmax 16/17, next token " Paris"/" moon" on both sides, on a geometry (16 heads / 2 KV heads) where the tick-28 GQA bug is visible and the pre-fix build is 16-24x worse. The *mechanism* of the residual gap was settled earlier (tick 21, on Tiny-MoE): `engine(Q4_K_M)` is byte-identical to `engine` on the same weights widened to F32, so the gap is llama.cpp's Q8_K activation quantization and not a bug — and its floor is now measured on this model's own weights too (7.0e-3 per matvec, 762 probes). The identity itself cannot be re-run at 3B scale here: the widening is 12.4 GB |
| **C3** | Gemma2 support | Gemma-2-2B (1.6 GB Q4) | Correct tokens vs llama.cpp |
| **C4** | Phi-3 support | Phi-3-mini-4k (2.3 GB Q4) | Fused QKV, bias |
| **C5** | GPT-2 support | GPT-2 (124M Q8) | ✅ synthetic F32 fixture matches llama.cpp (tick 16); real 124M Q8 still unvalidated |
| **C6** | Falcon support | Falcon-7B (4.7 GB Q4) | Parallel attention+FFN |
| **C7** | DeepSeek-V2 support | DeepSeek-V2-Lite (9.8 GB Q4) | ✅ fixture parity tick 18: pseudo-random F32 fixture with MoE + shared expert matches llama.cpp to 5.8e-7 and exact float64 to 5.0e-7. `ffn_gate_up_exps` (the fused MoE form) closed tick 25 — `TensorKind::FfnGateUpExps`, and the split/fused pair is asserted **bit identical** rather than merely close |
| **C8** | Gemma3 support | Gemma-3-4B (2.7 GB Q4) | ✅ done tick 20 — converter-faithful SWA fixture + f64 reference + 10 mutations. Still unverified against a real Gemma-3 GGUF |
| **C9** | K2-Horizon support | K2-Horizon-7B (4.9 GB Q4) | Attention gate + group norm — **no oracle exists**: llama.cpp has dropped the architecture |
| **C10** | Mamba2 support | Mamba2-2.7B (F16) | SSM scan |
| **C11** | Tokenizer parity | One per tokenizer type | ✅ BPE exact vs llama.cpp on Qwen2.5-3B (tick 17) **and OLMoE (40/40, tick 22 — the last 6 cases were the missing `tokenizer_st_partition`, not the `olmo` regex)**, plus a 75-case generated partition sweep; ✅ SPM exact vs llama.cpp on Tiny-MoE, 54/54 cases x both `add_special` (tick 19) |
| **C12** | Multi-modal (LLaVA) | LLaVA-1.6-7B | Vision + text generation |

## Recommended build order (first 10 PRs)

1. **PR-1** — Fix Q8_K dequant + add Q2_K, Q3_K, Q5_K dequant. Add BF16. (`quant.rs`)
2. **PR-2** — Architecture dispatch: enum + string → variant mapping, clean `Model::open()` arch string → trait. (`model.rs`)
3. **PR-3** — Qwen2 arch: add `arch("qwen2")` into existing llama forward pass. Test with Qwen2.5-0.5B-GGUF.
4. **PR-4** — Gemma2 arch: same pattern as Qwen2 but with QK-norm always on, different eps default.
5. **PR-5** — Phi-3 arch: fused QKV parsing, output bias, correct tensor name mapping.
6. **PR-6** — GPT-2 arch: GELU activation, no per-layer RMSNorm, no RoPE (no positional encoding).
7. **PR-7** — Falcon arch: parallel attention+FFN, single norm.
8. **PR-8** — DeepSeek-V2 arch: MLA, MoE with shared experts, tiktoken tokenizer.
9. **PR-9** — Gemma3 arch: sliding window + global alternating attention.
10. **PR-10** — K2-Horizon arch: attention gate, grouped RMSNorm, MoVA optional.