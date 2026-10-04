//! Disk-streaming layer-by-layer forward pass for llama-arch GGUF models.
//! Supports MHA, GQA, MoE, bias, and dense FFN paths.
//! DISCIPLINE: no tensor is ever fully resident. Weight rows are seek()+read()
//! +dequant() at use time in <=256 KiB chunks. Only these live in RAM:
//!   - GGUF header/tensor dir (KBs)
//!   - KV cache (layers x seq x embd f32)
//!   - activation vectors for the current token
//!   - the current row-chunk being multiplied (<=256 KiB)
use crate::ggml_dot::{ggml_dot_f32, ggml_rms_norm_scale};
use crate::ggml_norm::ggml_layernorm;
use crate::gguf::{GGUFFile, TensorInfo, parse};
use crate::quant::{read_rows_raw, read_rows_streamed};
use std::fs::File;
use std::io::{self};
// ---- Architecture dispatch ----
/// Supported architecture variants. Each wraps the architecture's tensor
/// naming convention and parameter defaults.
///
/// Tier 1 archs (Qwen2, Gemma2, etc.) share the llama tensor layout exactly
/// — only the `general.architecture` key differs. They use the same
/// `tensor_name()` patterns as `Llama`.
///
/// `from_string()` maps a GGUF architecture string to the variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchVariant {
    Llama,
    // Tier 1 — same tensor layout as Llama, different arch key
    Qwen2,
    Gemma2,
    Gemma, // gemma (original)
    CodeShell,
    Orion,
    InternLM2,
    StableLM,
    Xverse,
    PLamo,
    Baichuan,
    Olmo,
    Smollm3,
    OpenELM,
    Nemotron,
    // Tier 2 — tensor layout variants
    Phi3,
    GPT2,
    Falcon,
    // Tier 3 — attention changes
    DeepSeek2,
    K2Horizon,
    Gemma3,
}
/// Lookup from GGUF `general.architecture` string → ArchVariant.
pub fn arch_from_string(s: &str) -> ArchVariant {
    match s {
        "llama" | "llama4" | "laguna" | "deci" | "grok" => ArchVariant::Llama,
        "qwen2" | "qwen2moe" => ArchVariant::Qwen2,
        "gemma2" => ArchVariant::Gemma2,
        "gemma" => ArchVariant::Gemma,
        "codeshell" => ArchVariant::CodeShell,
        "orion" => ArchVariant::Orion,
        "internlm2" => ArchVariant::InternLM2,
        "stablelm" => ArchVariant::StableLM,
        "xverse" => ArchVariant::Xverse,
        "plamo" | "plamo2" => ArchVariant::PLamo,
        "baichuan" => ArchVariant::Baichuan,
        "olmo" | "olmo2" => ArchVariant::Olmo,
        "smollm3" => ArchVariant::Smollm3,
        "openelm" => ArchVariant::OpenELM,
        "nemotron" | "nemotron_h" | "nemotron_h_moe" => ArchVariant::Nemotron,
        "phi3" | "phimoe" => ArchVariant::Phi3,
        "gpt2" => ArchVariant::GPT2,
        "falcon" | "falcon-h1" => ArchVariant::Falcon,
        "deepseek2" => ArchVariant::DeepSeek2,
        "gemma3" | "gemma3n" => ArchVariant::Gemma3,
        "k2_horizon" | "k2-horizon" => ArchVariant::K2Horizon,
        _ => {
            eprintln!("[model] unknown architecture \"{s}\", treating as llama");
            ArchVariant::Llama
        }
    }
}
/// GGUF architecture strings the engine has a real tensor layout and forward pass
/// for. Everything here is covered by `tests/tensor_names.rs`, which demands that
/// `tensor_name()` can name everything llama.cpp's own `load_arch_tensors` asks for.
pub fn supported_architectures() -> Vec<&'static str> {
    vec![
        "llama",
        "qwen2",
        "qwen2moe",
        "gemma",
        "gemma2",
        "codeshell",
        "orion",
        "internlm2",
        "stablelm",
        "xverse",
        "plamo",
        "baichuan",
        "olmo",
        "olmo2",
        "smollm3",
        "openelm",
        "nemotron",
        "phi3",
        "gpt2",
        "falcon",
        "deepseek2",
        "gemma3",
    ]
}

/// Architecture strings `arch_from_string` recognizes but has no implementation
/// for. They load through the nearest implemented layout (usually `Llama`) and
/// will be missing tensors the forward pass needs; they are listed so that
/// `--list-architectures` does not advertise them as working.
///
/// `tests/tensor_names.rs` reads this and asserts the set stays in step with the
/// oracle: dropping an architecture here fails the test when llama.cpp adds it
/// back, and promoting one requires the tensor names to be covered first.
pub fn unimplemented_architectures() -> Vec<(&'static str, &'static str)> {
    vec![
        ("llama4", "expansion/attention-scale variants of llama"),
        ("laguna", "attention gate (attn_gate)"),
        ("deci", "decoder-decoder layers + LongRoPE factor tensors"),
        ("grok", "attention output norm + post_ffw_norm fallback"),
        ("plamo2", "Mamba/attention hybrid layers"),
        ("phimoe", "split QKV projections + MoE expert routing"),
        ("nemotron_h", "hybrid SSM/attention layers"),
        (
            "nemotron_h_moe",
            "hybrid SSM/attention layers + ffn_latent_{up,down}",
        ),
        ("falcon-h1", "hybrid SSM/attention layers"),
        ("gemma3n", "altup + laurel recurrent connections"),
        (
            "k2_horizon",
            "no upstream reference: llama.cpp dropped this arch",
        ),
        (
            "k2-horizon",
            "no upstream reference: llama.cpp dropped this arch",
        ),
    ]
}
// ---- Tensor name dispatch ----
/// Tensor kinds used by the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorKind {
    TokenEmbedd,
    Output,
    AttnNorm,
    AttnQ,
    AttnK,
    AttnV,
    AttnQKNorm, // qk-norm weight name (same for Q and K)
    AttnKNorm,  // attn_k_norm.weight — llama.cpp keeps Q and K norms separate
    AttnQBias,
    AttnKBias,
    AttnVBias,
    AttnQKVBias, // fused QKV bias (GPT-2 c_attn.bias)
    AttnOutput,
    AttnOutputBias,
    AttnNorm2,     // attn_norm_2.weight (Falcon post-attention norm)
    AttnNorm2Bias, // attn_norm_2.bias
    FfnNorm,
    FfnUp,
    FfnGate,
    FfnDown,
    FfnGateInp,  // MoE router
    FfnUpExps,   // MoE expert up
    FfnGateExps, // MoE expert gate
    /// MoE expert gate+up in one tensor, `{n_embd, n_ff*2, n_expert}`.
    ///
    /// llama.cpp's `create_tensor_gate_up_exps` (llama-model.cpp:2881) prefers
    /// this and only creates the split pair when it is absent, so a GGUF
    /// converted with `convert_hf_to_gguf.py --fuse-gate-up-exps` has *only*
    /// this tensor and an engine that reads split-only cannot load it at all.
    FfnGateUpExps,
    FfnDownExps, // MoE expert down
    OutputNorm,
    PositionEmbedd, // learned positional embeddings (GPT-2, etc.)
    AttnNormBias,   // attention norm bias
    FfnNormBias,    // FFN norm bias
    FfnUpBias,      // FFN up bias
    FfnDownBias,    // FFN down bias
    OutputNormBias, // final norm bias
    AttnPostNorm,   // attn_post_norm.weight
    FfnPostNorm,    // ffn_post_norm.weight
    Dense2Out,      // dense_2_out.weight
    Dense3Out,      // dense_3_out.weight
    // DeepSeek-V2 MLA specific
    AttnQANorm,      // attn_q_a_norm.weight
    AttnQA,          // attn_q_a.weight
    AttnQB,          // attn_q_b.weight
    AttnKVAMQA,      // attn_kv_a_mqa.weight
    AttnKVANorm,     // attn_kv_a_norm.weight
    AttnKB,          // attn_k_b.weight
    AttnVB,          // attn_v_b.weight
    FfnGateShexp,    // ffn_gate_shexp.weight
    FfnGateInpShexp, // ffn_gate_inp_shexp.weight (Qwen2-MoE shared-expert sigmoid gate)
    FfnUpShexp,      // ffn_up_shexp.weight
    FfnDownShexp,    // ffn_down_shexp.weight
    FfnExpProbsB,    // exp_probs_b.bias (optional)
    // K2-Horizon specific (attention gate, MoVA value experts)
    AttnGate,      // attn_gate.weight (softplus gate)
    AttnVGate,     // attn_v_gate.weight (value expert gate)
    AttnVGateBias, // attn_v_gate.bias (value expert gate bias)
    AttnVExps,     // attn_v_exps.weight (value experts)
}

/// Every tensor kind, in declaration order.
///
/// `tests/tensor_names.rs` enumerates this to build the set of GGUF tensor names
/// `tensor_name()` can produce for an architecture, and compares that set against
/// the oracle extracted from llama.cpp's own `load_arch_tensors`.
pub const ALL_TENSOR_KINDS: &[TensorKind] = &[
    TensorKind::TokenEmbedd,
    TensorKind::Output,
    TensorKind::AttnNorm,
    TensorKind::AttnQ,
    TensorKind::AttnK,
    TensorKind::AttnV,
    TensorKind::AttnQKNorm,
    TensorKind::AttnKNorm,
    TensorKind::AttnQBias,
    TensorKind::AttnKBias,
    TensorKind::AttnVBias,
    TensorKind::AttnQKVBias,
    TensorKind::AttnOutput,
    TensorKind::AttnOutputBias,
    TensorKind::AttnNorm2,
    TensorKind::AttnNorm2Bias,
    TensorKind::FfnNorm,
    TensorKind::FfnUp,
    TensorKind::FfnGate,
    TensorKind::FfnDown,
    TensorKind::FfnGateInp,
    TensorKind::FfnUpExps,
    TensorKind::FfnGateExps,
    TensorKind::FfnGateUpExps,
    TensorKind::FfnDownExps,
    TensorKind::OutputNorm,
    TensorKind::PositionEmbedd,
    TensorKind::AttnNormBias,
    TensorKind::FfnNormBias,
    TensorKind::FfnUpBias,
    TensorKind::FfnDownBias,
    TensorKind::OutputNormBias,
    TensorKind::AttnPostNorm,
    TensorKind::FfnPostNorm,
    TensorKind::Dense2Out,
    TensorKind::Dense3Out,
    TensorKind::AttnQANorm,
    TensorKind::AttnQA,
    TensorKind::AttnQB,
    TensorKind::AttnKVAMQA,
    TensorKind::AttnKVANorm,
    TensorKind::AttnKB,
    TensorKind::AttnVB,
    TensorKind::FfnGateShexp,
    TensorKind::FfnGateInpShexp,
    TensorKind::FfnUpShexp,
    TensorKind::FfnDownShexp,
    TensorKind::FfnExpProbsB,
    TensorKind::AttnGate,
    TensorKind::AttnVGate,
    TensorKind::AttnVGateBias,
    TensorKind::AttnVExps,
];
/// Which element-pairing convention a model uses for RoPE.
///
/// These are not interchangeable and getting them wrong is silent: the model
/// still produces plausible-looking text, just wrong. Mirrors
/// `llama_model_rope_type()` in llama.cpp `src/llama-model.cpp`, which is
/// itself a per-architecture table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// `GGML_ROPE_TYPE_NORMAL` (llama.cpp value 0). Rotate *adjacent* pairs
    /// (x0,x1), (x2,x3), ... This is what original LLaMA/HF-llama uses.
    Norm,
    /// `GGML_ROPE_TYPE_NEOX` (llama.cpp value 2). Rotate pairs offset by half
    /// the head dimension: (x0, x_{n/2}), (x1, x_{n/2+1}), ... This is HF's
    /// `rotate_half`, used by Qwen2/Qwen3, Gemma, Phi, GPT-NeoX, Falcon, ...
    NeoX,
}

/// Per-architecture RoPE convention, mirroring `llama_model_rope_type()`.
///
/// Keyed on the raw `general.architecture` string rather than `ArchVariant`,
/// because the convention is not constant within a variant — `olmo` is
/// GGML_ROPE_TYPE_NORMAL while `olmo2`/`olmoe` are NEOX, and both map to
/// `ArchVariant::Olmo`.
pub fn rope_type_for_arch(arch: &str) -> RopeType {
    // ggml rotates pairs offset by n_rot/2 for these.
    const NEOX: &[&str] = &[
        // "the pairs of head values are offset by n_rot/2"
        "falcon",
        "falcon-h1",
        "grok",
        "dbrx",
        "bert",
        "jina-bert-v3",
        "modern-bert",
        "nomic-bert",
        "nomic-bert-moe",
        "eurobert",
        "stablelm",
        "bitnet",
        "qwen",
        "qwen2",
        "qwen2moe",
        "qwen3",
        "qwen3moe",
        "dream",
        "llada-moe",
        "rnd1",
        "olmo2",
        "olmoe",
        "phi2",
        "phi3",
        "phimoe",
        "plamo",
        "plamo2",
        "plamo3",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3n",
        "gemma4",
        "gemma-embedding",
        "starcoder2",
        "openelm",
        "gptneox",
        "codeshell",
        "orion",
        "nemotron",
        "exaone",
        "exaone4",
        "exaone-moe",
        "minicpm3",
        "dots1",
        "hunyuan-moe",
        "hunyuan-dense",
        "hy-v3",
        "lfm2",
        "lfm2moe",
        "smallthinker",
        "seed-oss",
        "grovemoe",
        "apertus",
        "minimax-m2",
        "cogvlm",
        "pangu-embedding",
        "jais2",
        "openai-moe",
        "bailingmoe2",
    ];
    if NEOX.contains(&arch) {
        RopeType::NeoX
    } else {
        // GGML_ROPE_TYPE_NORMAL is llama.cpp's default for everything not
        // explicitly listed, including architectures with no RoPE at all.
        RopeType::Norm
    }
}

/// Internal alias for [`rope_apply`] so the many call sites stay short.
fn rope(x: &mut [f32], pos: usize, theta: f64, kind: RopeType) {
    rope_apply(x, pos, theta, kind)
}

/// Format a tensor name for the given architecture and layer/bias.
/// For Tier 1 architectures, tensor names match llama exactly.
pub fn tensor_name(arch: ArchVariant, kind: TensorKind, layer: usize) -> String {
    match arch {
        // Tier 0/1 — llama tensor names (also used by Qwen2, Gemma2, etc.)
        ArchVariant::Llama
        | ArchVariant::Qwen2
        | ArchVariant::Gemma2
        | ArchVariant::Gemma
        | ArchVariant::CodeShell
        | ArchVariant::Orion
        | ArchVariant::InternLM2
        | ArchVariant::StableLM
        | ArchVariant::Xverse
        | ArchVariant::PLamo
        | ArchVariant::Baichuan
        | ArchVariant::Olmo
        | ArchVariant::Smollm3
        | ArchVariant::Nemotron => llama_tensor_name(kind, layer),
        // Tier 2 — tensor name variants
        ArchVariant::Phi3 => phi3_tensor_name(kind, layer),
        ArchVariant::GPT2 => gpt2_tensor_name(kind, layer),
        ArchVariant::Falcon => falcon_tensor_name(kind, layer),
        // OpenELM fuses QKV like Phi-3 but keeps per-head q_norm/k_norm, which
        // come from the llama-layout fallback.
        ArchVariant::OpenELM => phi3_tensor_name(kind, layer),
        // Tier 3 — attention changes
        ArchVariant::DeepSeek2 => deepseek2_tensor_name(kind, layer),
        // Tier 3 — Gemma3: same llama tensor layout + post-norms
        ArchVariant::Gemma3 => gemma3_tensor_name(kind, layer),
        // Tier 3 — K2-Horizon: attention gate, grouped RMSNorm, MoVA value experts
        ArchVariant::K2Horizon => k2_horizon_tensor_name(kind, layer),
    }
}
/// DeepSeek-V2 MLA tensor names
fn deepseek2_tensor_name(kind: TensorKind, layer: usize) -> String {
    match kind {
        TensorKind::TokenEmbedd => "token_embd.weight".to_string(),
        TensorKind::Output => "output.weight".to_string(),
        TensorKind::OutputNorm => "output_norm.weight".to_string(),
        TensorKind::AttnNorm => format!("blk.{layer}.attn_norm.weight"),
        TensorKind::AttnOutput => format!("blk.{layer}.attn_output.weight"),
        TensorKind::FfnNorm => format!("blk.{layer}.ffn_norm.weight"),
        // MLA attention tensors
        TensorKind::AttnQANorm => format!("blk.{layer}.attn_q_a_norm.weight"),
        TensorKind::AttnQA => format!("blk.{layer}.attn_q_a.weight"),
        TensorKind::AttnQB => format!("blk.{layer}.attn_q_b.weight"),
        TensorKind::AttnKVAMQA => format!("blk.{layer}.attn_kv_a_mqa.weight"),
        TensorKind::AttnKVANorm => format!("blk.{layer}.attn_kv_a_norm.weight"),
        TensorKind::AttnKB => format!("blk.{layer}.attn_k_b.weight"),
        TensorKind::AttnVB => format!("blk.{layer}.attn_v_b.weight"),
        // MoE / shared expert tensors
        TensorKind::FfnGateInp => format!("blk.{layer}.ffn_gate_inp.weight"),
        TensorKind::FfnUpExps => format!("blk.{layer}.ffn_up_exps.weight"),
        TensorKind::FfnGateExps => format!("blk.{layer}.ffn_gate_exps.weight"),
        // deepseek2 is one of the eight architectures that call
        // `create_tensor_gate_up_exps`, so this name exists in the wild for it.
        // The other seven (cohere2moe, deepseek2-ocr, gemma4, hy_v3, mistral4,
        // qwen35moe, qwen3next) have no forward pass in the engine yet.
        TensorKind::FfnGateUpExps => format!("blk.{layer}.ffn_gate_up_exps.weight"),
        TensorKind::FfnDownExps => format!("blk.{layer}.ffn_down_exps.weight"),
        TensorKind::FfnGateShexp => format!("blk.{layer}.ffn_gate_shexp.weight"),
        TensorKind::FfnUpShexp => format!("blk.{layer}.ffn_up_shexp.weight"),
        TensorKind::FfnDownShexp => format!("blk.{layer}.ffn_down_shexp.weight"),
        TensorKind::FfnExpProbsB => format!("blk.{layer}.exp_probs_b.bias"),
        // Not used by DeepSeek-V2, fall back to llama
        _ => llama_tensor_name(kind, layer),
    }
}

/// Gemma3 tensor names: standard blk.N.* layout with sliding window attention post-norms
fn gemma3_tensor_name(kind: TensorKind, layer: usize) -> String {
    match kind {
        TensorKind::TokenEmbedd => "token_embd.weight".to_string(),
        TensorKind::Output => "output.weight".to_string(),
        TensorKind::OutputNorm => "output_norm.weight".to_string(),
        TensorKind::AttnNorm => format!("blk.{layer}.attn_norm.weight"),
        TensorKind::AttnOutput => format!("blk.{layer}.attn_output.weight"),
        TensorKind::FfnNorm => format!("blk.{layer}.ffn_norm.weight"),
        TensorKind::AttnPostNorm => format!("blk.{layer}.post_attention_norm.weight"),
        TensorKind::FfnPostNorm => format!("blk.{layer}.post_ffw_norm.weight"),
        TensorKind::AttnQ => format!("blk.{layer}.attn_q.weight"),
        TensorKind::AttnK => format!("blk.{layer}.attn_k.weight"),
        TensorKind::AttnV => format!("blk.{layer}.attn_v.weight"),
        TensorKind::AttnQKNorm => format!("blk.{layer}.attn_q_norm.weight"),
        TensorKind::AttnKNorm => format!("blk.{layer}.attn_k_norm.weight"),
        TensorKind::AttnQBias => format!("blk.{layer}.attn_q.bias"),
        TensorKind::AttnKBias => format!("blk.{layer}.attn_k.bias"),
        TensorKind::AttnVBias => format!("blk.{layer}.attn_v.bias"),
        TensorKind::AttnQKVBias => format!("blk.{layer}.attn_qkv.bias"),
        TensorKind::AttnOutputBias => format!("blk.{layer}.attn_output.bias"),
        TensorKind::FfnUp => format!("blk.{layer}.ffn_up.weight"),
        TensorKind::FfnGate => format!("blk.{layer}.ffn_gate.weight"),
        TensorKind::FfnDown => format!("blk.{layer}.ffn_down.weight"),
        TensorKind::FfnGateInp => format!("blk.{layer}.ffn_gate_inp.weight"),
        TensorKind::Dense2Out => "dense_2.weight".to_string(),
        TensorKind::Dense3Out => "dense_3.weight".to_string(),
        TensorKind::AttnNormBias => format!("blk.{layer}.attn_norm.bias"),
        TensorKind::FfnNormBias => format!("blk.{layer}.ffn_norm.bias"),
        TensorKind::FfnUpBias => format!("blk.{layer}.ffn_up.bias"),
        TensorKind::FfnDownBias => format!("blk.{layer}.ffn_down.bias"),
        TensorKind::OutputNormBias => "output_norm.bias".to_string(),
        // DeepSeek-V2 MLA — not valid in Gemma3 layout; fall back to llama name
        _ => llama_tensor_name(kind, layer),
    }
}
/// Llama-style tensor names: `blk.N.attn_q.weight`, etc.
fn llama_tensor_name(kind: TensorKind, layer: usize) -> String {
    match kind {
        TensorKind::TokenEmbedd => "token_embd.weight".to_string(),
        TensorKind::Output => "output.weight".to_string(),
        TensorKind::OutputNorm => "output_norm.weight".to_string(),
        TensorKind::AttnNorm => format!("blk.{layer}.attn_norm.weight"),
        TensorKind::AttnQ => format!("blk.{layer}.attn_q.weight"),
        TensorKind::AttnK => format!("blk.{layer}.attn_k.weight"),
        TensorKind::AttnV => format!("blk.{layer}.attn_v.weight"),
        TensorKind::AttnQKNorm => format!("blk.{layer}.attn_q_norm.weight"),
        TensorKind::AttnKNorm => format!("blk.{layer}.attn_k_norm.weight"),
        TensorKind::AttnQBias => format!("blk.{layer}.attn_q.bias"),
        TensorKind::AttnKBias => format!("blk.{layer}.attn_k.bias"),
        TensorKind::AttnVBias => format!("blk.{layer}.attn_v.bias"),
        TensorKind::AttnQKVBias => format!("blk.{layer}.attn_qkv.bias"),
        TensorKind::AttnOutput => format!("blk.{layer}.attn_output.weight"),
        TensorKind::AttnOutputBias => format!("blk.{layer}.attn_output.bias"),
        TensorKind::AttnNorm2 => format!("blk.{layer}.attn_norm_2.weight"),
        TensorKind::AttnNorm2Bias => format!("blk.{layer}.attn_norm_2.bias"),
        TensorKind::AttnGate => format!("blk.{layer}.attn_gate.weight"),
        TensorKind::AttnPostNorm => format!("blk.{layer}.post_attention_norm.weight"),
        TensorKind::FfnPostNorm => format!("blk.{layer}.post_ffw_norm.weight"),
        TensorKind::FfnNorm => format!("blk.{layer}.ffn_norm.weight"),
        TensorKind::FfnUp => format!("blk.{layer}.ffn_up.weight"),
        TensorKind::FfnGate => format!("blk.{layer}.ffn_gate.weight"),
        TensorKind::FfnDown => format!("blk.{layer}.ffn_down.weight"),
        TensorKind::FfnGateInp => format!("blk.{layer}.ffn_gate_inp.weight"),
        TensorKind::FfnUpExps => format!("blk.{layer}.ffn_up_exps.weight"),
        TensorKind::FfnGateExps => format!("blk.{layer}.ffn_gate_exps.weight"),
        TensorKind::FfnDownExps => format!("blk.{layer}.ffn_down_exps.weight"),
        // LLM_TENSOR_POS_EMBD is "position_embd" with no block index: it is a
        // single table shared by every layer (GPT-2 learned absolute positions).
        TensorKind::PositionEmbedd => "position_embd.weight".to_string(),
        TensorKind::AttnNormBias => format!("blk.{layer}.attn_norm.bias"),
        TensorKind::FfnNormBias => format!("blk.{layer}.ffn_norm.bias"),
        TensorKind::FfnUpBias => format!("blk.{layer}.ffn_up.bias"),
        TensorKind::FfnDownBias => format!("blk.{layer}.ffn_down.bias"),
        // Also not per-layer: LLM_TENSOR_OUTPUT_NORM is "output_norm".
        TensorKind::OutputNormBias => "output_norm.bias".to_string(),
        // DeepSeek-V2 MLA — not valid in llama layout; fall back to llama name
        TensorKind::AttnQANorm => format!("blk.{layer}.attn_q_a_norm.weight"),
        TensorKind::AttnQA => format!("blk.{layer}.attn_q_a.weight"),
        TensorKind::AttnQB => format!("blk.{layer}.attn_q_b.weight"),
        TensorKind::AttnKVAMQA => format!("blk.{layer}.attn_kv_a_mqa.weight"),
        TensorKind::AttnKVANorm => format!("blk.{layer}.attn_kv_a_norm.weight"),
        TensorKind::AttnKB => format!("blk.{layer}.attn_k_b.weight"),
        TensorKind::AttnVB => format!("blk.{layer}.attn_v_b.weight"),
        TensorKind::FfnGateShexp => format!("blk.{layer}.ffn_gate_shexp.weight"),
        TensorKind::FfnGateInpShexp => format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
        TensorKind::FfnUpShexp => format!("blk.{layer}.ffn_up_shexp.weight"),
        TensorKind::FfnDownShexp => format!("blk.{layer}.ffn_down_shexp.weight"),
        TensorKind::FfnExpProbsB => format!("blk.{layer}.exp_probs_b.bias"),
        // Gemma3 post-norms — not valid in llama layout; fall back
        _ => format!("blk.{layer}.unknown.weight"),
    }
}
/// Phi-3 tensor names: fused QKV, bias on output.
fn phi3_tensor_name(kind: TensorKind, layer: usize) -> String {
    match kind {
        TensorKind::AttnQ => format!("blk.{layer}.attn_qkv.weight"),
        TensorKind::AttnK => format!("blk.{layer}.attn_qkv.weight"),
        TensorKind::AttnV => format!("blk.{layer}.attn_qkv.weight"),
        _ => llama_tensor_name(kind, layer),
    }
}
/// GPT-2 tensor names: use standard GGUF names (same as llama for most, with fused QKV + biases)
fn gpt2_tensor_name(kind: TensorKind, layer: usize) -> String {
    match kind {
        TensorKind::TokenEmbedd => "token_embd.weight".to_string(),
        TensorKind::Output => "output.weight".to_string(), // tied embeddings handled separately
        TensorKind::PositionEmbedd => "position_embd.weight".to_string(),
        TensorKind::OutputNorm => "output_norm.weight".to_string(),
        TensorKind::OutputNormBias => "output_norm.bias".to_string(),
        TensorKind::AttnNorm => format!("blk.{layer}.attn_norm.weight"),
        TensorKind::AttnNormBias => format!("blk.{layer}.attn_norm.bias"),
        TensorKind::AttnQ => format!("blk.{layer}.attn_qkv.weight"),
        TensorKind::AttnK => format!("blk.{layer}.attn_qkv.weight"),
        TensorKind::AttnV => format!("blk.{layer}.attn_qkv.weight"),
        TensorKind::AttnQKVBias => format!("blk.{layer}.attn_qkv.bias"),
        TensorKind::AttnOutput => format!("blk.{layer}.attn_output.weight"),
        TensorKind::AttnOutputBias => format!("blk.{layer}.attn_output.bias"),
        TensorKind::FfnNorm => format!("blk.{layer}.ffn_norm.weight"),
        TensorKind::FfnNormBias => format!("blk.{layer}.ffn_norm.bias"),
        TensorKind::FfnUp => format!("blk.{layer}.ffn_up.weight"),
        TensorKind::FfnUpBias => format!("blk.{layer}.ffn_up.bias"),
        TensorKind::FfnDown => format!("blk.{layer}.ffn_down.weight"),
        TensorKind::FfnDownBias => format!("blk.{layer}.ffn_down.bias"),
        _ => llama_tensor_name(kind, layer),
    }
}
/// Falcon tensor names: canonical GGUF `blk.N.` prefix with fused QKV + LayerNorm bias.
/// Falcon uses a single attn_norm (with bias) for both attention and FFN (parallel path).
fn falcon_tensor_name(kind: TensorKind, layer: usize) -> String {
    match kind {
        TensorKind::TokenEmbedd => "token_embd.weight".to_string(),
        TensorKind::Output => "output.weight".to_string(),
        TensorKind::OutputNorm => "output_norm.weight".to_string(),
        TensorKind::OutputNormBias => "output_norm.bias".to_string(),
        TensorKind::AttnNorm => format!("blk.{layer}.attn_norm.weight"),
        TensorKind::AttnNormBias => format!("blk.{layer}.attn_norm.bias"),
        // Fused QKV: one tensor, split into Q/K/V by row-index ranges
        TensorKind::AttnQ => format!("blk.{layer}.attn_qkv.weight"),
        TensorKind::AttnK => format!("blk.{layer}.attn_qkv.weight"),
        TensorKind::AttnV => format!("blk.{layer}.attn_qkv.weight"),
        TensorKind::AttnOutput => format!("blk.{layer}.attn_output.weight"),
        TensorKind::FfnUp => format!("blk.{layer}.ffn_up.weight"),
        TensorKind::FfnDown => format!("blk.{layer}.ffn_down.weight"),
        // Falcon has no separate ffn_norm — single attn_norm for both branches
        // Falcon has no FfnGate — uses GELU activation, not SwiGLU
        _ => llama_tensor_name(kind, layer),
    }
}
/// K2-Horizon tensor names: standard blk.N.* layout with attention gate, MoVA, grouped norms.
fn k2_horizon_tensor_name(kind: TensorKind, layer: usize) -> String {
    match kind {
        TensorKind::TokenEmbedd => "token_embd.weight".to_string(),
        TensorKind::Output => "output.weight".to_string(),
        TensorKind::OutputNorm => "output_norm.weight".to_string(),
        TensorKind::AttnNorm => format!("blk.{layer}.attn_norm.weight"),
        TensorKind::AttnOutput => format!("blk.{layer}.attn_output.weight"),
        TensorKind::FfnNorm => format!("blk.{layer}.ffn_norm.weight"),
        TensorKind::AttnQ => format!("blk.{layer}.attn_q.weight"),
        TensorKind::AttnK => format!("blk.{layer}.attn_k.weight"),
        TensorKind::AttnV => format!("blk.{layer}.attn_v.weight"),
        TensorKind::AttnQKNorm => format!("blk.{layer}.attn_q_norm.weight"),
        TensorKind::FfnUp => format!("blk.{layer}.ffn_up.weight"),
        TensorKind::FfnGate => format!("blk.{layer}.ffn_gate.weight"),
        TensorKind::FfnDown => format!("blk.{layer}.ffn_down.weight"),
        TensorKind::FfnGateInp => format!("blk.{layer}.ffn_gate_inp.weight"),
        TensorKind::FfnUpExps => format!("blk.{layer}.ffn_up_exps.weight"),
        TensorKind::FfnGateExps => format!("blk.{layer}.ffn_gate_exps.weight"),
        TensorKind::FfnDownExps => format!("blk.{layer}.ffn_down_exps.weight"),
        // K2-Horizon specific: attention gate
        TensorKind::AttnGate => format!("blk.{layer}.attn_gate.weight"),
        // MoVA value experts
        TensorKind::AttnVGate => format!("blk.{layer}.attn_v_gate.weight"),
        TensorKind::AttnVGateBias => format!("blk.{layer}.attn_v_gate.bias"),
        TensorKind::AttnVExps => format!("blk.{layer}.attn_v_exps.weight"),
        // Shared experts (same pattern as DeepSeek-V2)
        TensorKind::FfnGateShexp => format!("blk.{layer}.ffn_gate_shexp.weight"),
        TensorKind::FfnUpShexp => format!("blk.{layer}.ffn_up_shexp.weight"),
        TensorKind::FfnDownShexp => format!("blk.{layer}.ffn_down_shexp.weight"),
        TensorKind::FfnExpProbsB => format!("blk.{layer}.exp_probs_b.bias"),
        // Positional embeddings
        TensorKind::PositionEmbedd => "position_embd.weight".to_string(),
        _ => llama_tensor_name(kind, layer),
    }
}
// ---- StreamedTensor ----
#[derive(Clone)]
pub struct StreamedTensor<'a> {
    file: &'a std::cell::RefCell<File>,
    base: u64,
    pub dims: &'a [u64],
    pub dtype: u32,
    /// row width in elements (dims[0])
    pub row_elems: usize,
    /// number of flat rows (product of dims[1..])
    pub n_rows: usize,
    pub row_bytes: usize,
    pub nbytes: u64,
}
impl<'a> StreamedTensor<'a> {
    pub fn new(file: &'a std::cell::RefCell<File>, gf: &'a GGUFFile, t: &'a TensorInfo) -> Self {
        let base = gf.data_start + t.offset;
        let n_rows: usize = if t.dims.len() >= 2 {
            t.dims[1..].iter().product::<u64>() as usize
        } else {
            1
        };
        let row_bytes = (t.nbytes as usize) / n_rows;
        StreamedTensor {
            file,
            base,
            dims: &t.dims,
            dtype: t.dtype,
            row_elems: t.dims[0] as usize,
            n_rows,
            row_bytes,
            nbytes: t.nbytes,
        }
    }
    /// Stream rows [start, start+n) and dequantize to f32 (n x row_elems).
    pub fn rows(&self, start: usize, n: usize, buf: &mut Vec<u8>) -> io::Result<Vec<f32>> {
        read_rows_streamed(
            &mut self.file.borrow_mut(),
            self.base,
            self.dtype,
            self.row_elems,
            start,
            n,
            self.row_bytes,
            buf,
        )
    }
    /// Stream raw packed bytes without dequant (for GPU upload).
    pub fn raw_rows(&self, start: usize, n: usize, buf: &mut Vec<u8>) -> io::Result<Vec<u8>> {
        read_rows_raw(
            &mut self.file.borrow_mut(),
            self.base,
            start,
            n,
            self.row_bytes,
            buf,
        )
    }
}
// ---- Model ----
pub struct Model {
    pub gf: GGUFFile,
    pub path: String,
    pub arch: String,
    pub arch_variant: ArchVariant,
    tensor_file: std::cell::RefCell<File>,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    /// RoPE pairing convention (see `rope_type_for_arch`)
    pub rope_type: RopeType,
    pub eps: f64,
    pub n_experts: usize,
    pub n_experts_used: usize,
    // DeepSeek-V2 MLA metadata
    pub n_lora_q: usize,
    pub n_lora_kv: usize,
    pub key_length_mla: usize,
    pub value_length_mla: usize,
    pub rope_dim: usize,
    pub rope_yarn_log_mul: f32,
    pub rope_scaling_factor: f32,
    pub rope_attn_factor: f32,
    pub n_ff_exp: usize,
    pub n_expert_shared: usize,
    pub leading_dense_layers: usize,
    pub expert_weights_scale: f64,
    pub expert_weights_norm: bool,
    pub expert_gating_func: u32,
    // K2-Horizon metadata
    pub n_norm_groups: usize,
    pub n_value_expert: usize,
    pub n_value_expert_used: usize,
    // Gemma3 SWA metadata
    pub n_swa: usize,
    pub swa_pattern: usize,
    pub rope_freq_base_swa: f64,
    pub f_attention_scale: f32,
    pub f_final_logit_softcapping: f32,
    pub qk_norm: bool,
    pub has_bias: bool,
    pub debug_checksums: bool,
    pub vocab_size: usize,
    pub vk: std::cell::RefCell<Option<crate::vk::VulkanBackend>>,
}

/// Is layer `layer` a sliding-window layer? (gemma3 / gemma2 / gemma4, and
/// every other arch that calls `llama_hparams::set_swa_pattern(n, false)`).
///
/// Upstream: `is_swa_impl[il] = n_pattern == 0 || (il % n_pattern < n_pattern - 1)`
/// (llama-hparams.cpp:8). The `n_pattern == 0` arm means *every* layer is SWA —
/// the opposite of "unset" — and treating a zero pattern as "no SWA anywhere"
/// inverts the model. A period of 1 is the degenerate other direction: `il % 1`
/// is always 0 and `0 < 0` is false, so nothing is SWA.
pub fn gemma3_is_swa(layer: usize, n_swa: usize, swa_period: usize) -> bool {
    if n_swa == 0 {
        return false;
    }
    swa_period == 0 || (layer % swa_period) < swa_period - 1
}

/// Gemma3's attention logit scale, which llama.cpp *computes* (gemma3.cpp:76)
/// rather than reading from metadata:
///
/// ```text
/// f_attention_scale = n_layer == 62 ? 1/sqrt(n_embd/n_head) : 1/sqrt(n_embd_head_k)
/// ```
///
/// 62 layers identifies the 27B checkpoint, and for it `n_embd/n_head` is 168
/// while `key_length` is 128 — so the branch is a 1.15x difference in every
/// attention logit, not a cosmetic one. Note this is `n_embd_head_k(0)`, i.e.
/// the *SWA* head width when layer 0 is an SWA layer, which it is under the
/// default pattern of 6.
///
/// `llama_model_gemma3::load_arch_hparams` never reads a
/// `gemma3.attention.scale` key, so an engine that trusts one will disagree with
/// llama.cpp on any file that carries it.
pub fn gemma3_attn_scale(n_layers: usize, n_embd: usize, n_heads: usize, head_dim: usize) -> f32 {
    if n_layers == 62 {
        1.0 / ((n_embd as f32) / (n_heads.max(1) as f32)).sqrt()
    } else {
        1.0 / (head_dim as f32).sqrt()
    }
}

impl Model {
    pub fn open(path: &str) -> io::Result<Self> {
        let gf = parse(path)?;
        let md = &gf.metadata;
        let get_u = |k: &str| -> io::Result<usize> {
            md.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("missing metadata {k}"))
                })
        };
        let arch_str = md
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();
        let arch_variant = arch_from_string(&arch_str);
        let n_layers = get_u(&format!("{arch_str}.block_count"))?;
        let n_embd = get_u(&format!("{arch_str}.embedding_length"))?;
        let n_ff = get_u(&format!("{arch_str}.feed_forward_length"))?;
        let n_heads = get_u(&format!("{arch_str}.attention.head_count"))?;
        let n_kv_heads = md
            .get(&format!("{arch_str}.attention.head_count_kv"))
            .and_then(|v| v.as_u64())
            .unwrap_or(n_heads as u64) as usize;
        let rope_theta = md
            .get(&format!("{arch_str}.rope.freq_base"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0);
        let rope_type = rope_type_for_arch(&arch_str);
        let eps = md
            .get(&format!("{arch_str}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f64())
            .or_else(|| {
                // GPT-2 (and other non-RMS archs) use the plain LayerNorm epsilon key
                md.get(&format!("{arch_str}.attention.layer_norm_epsilon"))
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1e-5);
        let n_experts = md
            .get(&format!("{arch_str}.expert_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let n_experts_used = md
            .get(&format!("{arch_str}.expert_used_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        // DeepSeek-V2 MLA metadata (canonical GGUF keys, see llama.cpp/src/models/deepseek2.cpp)
        let n_lora_q = md
            .get(&format!("{arch_str}.attention.q_lora_rank"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let n_lora_kv = md
            .get(&format!("{arch_str}.attention.kv_lora_rank"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let key_length_mla = md
            .get(&format!("{arch_str}.attention.key_length_mla"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let value_length_mla = md
            .get(&format!("{arch_str}.attention.value_length_mla"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let rope_dim = md
            .get(&format!("{arch_str}.rope.dimension_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let rope_yarn_log_mul = md
            .get(&format!("{arch_str}.rope.scaling.yarn_log_multiplier"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        // Cancel the factor from the convert script (see [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX] in llama.cpp)
        let rope_yarn_log_mul = if rope_yarn_log_mul != 0.0 {
            rope_yarn_log_mul / 0.1
        } else {
            0.0
        };
        let rope_scaling_factor = md
            .get(&format!("{arch_str}.rope.scaling.factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let rope_scaling_factor = if rope_scaling_factor == 0.0 {
            1.0
        } else {
            rope_scaling_factor
        };
        let rope_attn_factor = md
            .get(&format!("{arch_str}.rope.scaling.attn_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let n_ff_exp = md
            .get(&format!("{arch_str}.expert_feed_forward_length"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let n_expert_shared = md
            .get(&format!("{arch_str}.expert_shared_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let leading_dense_layers = md
            .get(&format!("{arch_str}.leading_dense_block_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let expert_weights_scale = md
            .get(&format!("{arch_str}.expert_weights_scale"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let expert_weights_norm = md
            .get(&format!("{arch_str}.expert_weights_norm"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            != 0;
        let expert_gating_func = md
            .get(&format!("{arch_str}.expert_gating_func"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        // vocab_size from canonical GGUF vocabulary metadata (LLM_KV_VOCAB_SIZE)
        let vocab_size = md
            .get(&format!("{arch_str}.vocab_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| {
                md.get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_vec_str())
                    .map(|v| v.len())
            })
            .unwrap_or(0);
        // K2-Horizon metadata (optional, defaults harmless for other archs)
        let n_norm_groups = md
            .get(&format!("{arch_str}.attention.group_norm_groups"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let n_value_expert = md
            .get(&format!("{arch_str}.attention.value_expert_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let n_value_expert_used = md
            .get(&format!("{arch_str}.attention.value_expert_used_count"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        // Gemma3 SWA metadata (canonical GGUF keys, see llama.cpp/src/models/gemma3.cpp)
        let n_swa = md
            .get(&format!("{arch_str}.attention.sliding_window"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let swa_pattern = md
            .get(&format!("{arch_str}.attention.sliding_window_pattern"))
            .and_then(|v| v.as_u64())
            .unwrap_or(6) as usize;
        // llama.cpp's default is 10000.0 (llama-hparams.h:127), *not* zero and
        // *not* "same as rope.freq_base". This matters more than it looks:
        // convert_hf_to_gguf.py's `Gemma3Model.set_gguf_parameters` calls
        // `add_rope_freq_base` for global layers but never `add_rope_freq_base_swa`,
        // even though gguf_writer.py implements it — so every real Gemma-3 GGUF
        // in the wild omits this key and leans on the default. Defaulting it to 0
        // here made SWA layers silently rotate at the global base (1e6) instead of
        // 1e4, a 100x error in every frequency of the SWA half of the network.
        let rope_freq_base_swa = md
            .get(&format!("{arch_str}.rope.freq_base_swa"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10_000.0);
        let f_attention_scale = md
            .get(&format!("{arch_str}.attention.scale"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let f_final_logit_softcapping = md
            .get(&format!("{arch_str}.final_logit_softcapping"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        // DeepSeek-V2 MLA shape validation
        if arch_variant == ArchVariant::DeepSeek2 {
            if n_lora_q > 0 && n_lora_kv == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "DeepSeek-V2 MLA: n_lora_kv must be > 0 when n_lora_q > 0",
                ));
            }
            if n_lora_kv > 0 && n_lora_q == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "DeepSeek-V2 MLA: n_lora_q must be > 0 when n_lora_kv > 0 (non-lite variant)",
                ));
            }
            if key_length_mla > 0 && rope_dim > 0 && rope_dim >= key_length_mla {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "DeepSeek-V2 MLA: rope_dim ({}) must be < key_length_mla ({})",
                        rope_dim, key_length_mla
                    ),
                ));
            }
            if n_ff_exp > 0 && n_expert_shared > 0 && n_experts == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "DeepSeek-V2 MoE: n_experts must be > 0 when shared experts are present",
                ));
            }
            if leading_dense_layers > n_layers {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "DeepSeek-V2: leading_dense_layers ({}) > n_layers ({})",
                        leading_dense_layers, n_layers
                    ),
                ));
            }
            if vocab_size == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "DeepSeek-V2: vocab_size is zero — cannot decode logits",
                ));
            }
        }
        // Detect qk_norm and has_bias from first layer — uses arch-aware names.
        let qk_norm =
            gf.tensors
                .contains_key(&tensor_name(arch_variant, TensorKind::AttnQKNorm, 0));
        // GPT-2 uses fused attn_qkv.bias, not separate attn_q.bias / attn_k.bias / attn_v.bias
        let has_bias = if arch_variant == ArchVariant::GPT2 {
            gf.tensors
                .contains_key(&tensor_name(arch_variant, TensorKind::AttnQKVBias, 0))
        } else {
            gf.tensors
                .contains_key(&tensor_name(arch_variant, TensorKind::AttnQBias, 0))
        };
        let tensor_file = std::cell::RefCell::new(File::open(path)?);
        let filetype = md
            .get("general.file_type")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let head_dim_raw = md
            .get(&format!("{arch_str}.attention.key_length"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_heads);
        // Reported only — no forward pass reads `head_dim` for an MLA architecture
        // (`forward_deepseek2` uses `key_length_mla`/`value_length_mla` and its own
        // cache layout). For those models `attention.key_length` is the width of one
        // cached *latent* row, kv_lora_rank + rope dims, so reporting it as
        // `head_dim` overstates the attention head width by a factor of ~3 on
        // DeepSeek-V2-Lite (576 vs 192). Print the real per-head width instead of
        // a number that looks plausible and is not.
        let head_dim = if arch_variant == ArchVariant::DeepSeek2 {
            md.get(&format!("{arch_str}.attention.key_length_mla"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(head_dim_raw)
        } else {
            head_dim_raw
        };
        eprintln!(
            "[model] arch={} variant={:?} layers={} embd={} heads={} kv={} ff={} head_dim={} experts={}x{} qk_norm={} bias={} file_type={}",
            arch_str,
            arch_variant,
            n_layers,
            n_embd,
            n_heads,
            n_kv_heads,
            n_ff,
            head_dim,
            n_experts,
            n_experts_used,
            qk_norm,
            has_bias,
            filetype
        );
        Ok(Model {
            gf,
            path: path.to_string(),
            arch: arch_str,
            arch_variant,
            tensor_file,
            n_layers,
            n_embd,
            n_ff,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta,
            rope_type,
            eps,
            n_experts,
            n_experts_used,
            // DeepSeek-V2 MLA metadata
            n_lora_q,
            n_lora_kv,
            key_length_mla,
            value_length_mla,
            rope_dim,
            rope_yarn_log_mul: rope_yarn_log_mul as f32,
            rope_scaling_factor: rope_scaling_factor as f32,
            rope_attn_factor: rope_attn_factor as f32,
            n_ff_exp,
            n_expert_shared,
            leading_dense_layers,
            expert_weights_scale,
            expert_weights_norm,
            expert_gating_func,
            // K2-Horizon metadata
            n_norm_groups,
            n_value_expert,
            n_value_expert_used,
            // Gemma3 SWA metadata
            n_swa,
            swa_pattern,
            rope_freq_base_swa,
            f_attention_scale,
            f_final_logit_softcapping,
            qk_norm,
            has_bias,
            debug_checksums: std::env::var("SLOW_DEBUG").is_ok(),
            vocab_size,
            vk: std::cell::RefCell::new(None),
        })
    }
    fn t<'a>(&'a self, kind: TensorKind, layer: usize) -> io::Result<StreamedTensor<'a>> {
        let name = tensor_name(self.arch_variant, kind, layer);
        let info = self
            .gf
            .tensors
            .get(&name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("tensor {name}")))?;
        Ok(StreamedTensor::new(&self.tensor_file, &self.gf, info))
    }
    /// The final `output.weight` projection, falling back to the tied token
    /// embedding matrix when the file carries no `output.weight`.
    ///
    /// llama.cpp does this for EVERY architecture, not just the ones that
    /// historically needed it: `output` is created `TENSOR_NOT_REQUIRED` and
    /// rebound to the token embedding tensor when that comes back NULL
    /// (`src/models/llama.cpp`, and every other model file since). Restricting
    /// the fallback to GPT-2/Falcon, as this engine used to, breaks any tied
    /// llama/qwen2 model — and every fixture in `tests/data/fixtures` is tied,
    /// so that mistake is not caught by "the real models still run".
    ///
    /// This is one helper rather than four copies of a `match`, because four
    /// copies is how it was lost once. Each architecture's forward pass held its
    /// own, a rewrite of two of them turned the other two into hard errors, and
    /// `quant_gap` + `ref_parity` went red with "tied embeddings unavailable"
    /// while the comment above one of the four still described the fallback it
    /// no longer performed. `weight_tying_disabled` in
    /// `scripts/mutation_check.py` anchors on this body, so removing the tying
    /// is one detected mutation rather than a silent per-architecture drift.
    fn output_t<'a>(&'a self) -> io::Result<StreamedTensor<'a>> {
        match self.t(TensorKind::Output, 0) {
            Ok(t) => Ok(t),
            Err(_) => self.t(TensorKind::TokenEmbedd, 0),
        }
    }
    /// Lookup a tensor by exact name string (for architecture-specific bridging).
    fn t_by_name<'a>(&'a self, name: &str) -> io::Result<StreamedTensor<'a>> {
        let info = self
            .gf
            .tensors
            .get(name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("tensor {name}")))?;
        Ok(StreamedTensor::new(&self.tensor_file, &self.gf, info))
    }
    /// y = W @ x, streaming W row-blocks from disk or GPU.
    /// Falls back to CPU if Vulkan is not available or weight format not supported.
    fn matvec<'a>(
        &'a self,
        w: &StreamedTensor<'a>,
        x: &[f32],
        buf: &mut Vec<u8>,
    ) -> io::Result<Vec<f32>> {
        let n = w.n_rows;
        let mut y = vec![0f32; n];

        // Try GPU path — chunked per matvec_gpu internal dispatch
        if let Some(ref mut vk) = *self.vk.borrow_mut()
            && let Some(gpu_y) = vk.matvec_gpu(w.dtype, w.row_elems, n, w, x)
        {
            return Ok(gpu_y);
        }

        // CPU fallback
        let chunk_rows = (262_144 / w.row_bytes).max(1);
        let mut start = 0;
        while start < n {
            let nr = chunk_rows.min(n - start);
            let rows = w.rows(start, nr, buf)?;
            for r in 0..nr {
                let row = &rows[r * w.row_elems..(r + 1) * w.row_elems];
                // ggml_vec_dot_f32's reduction, not a sequential f32 sum. The
                // difference is 2.1e-5..2.5e-4 on qwen2.5-3b and it was the
                // floor of every engine-vs-llama.cpp number in the project.
                // See src/ggml_dot.rs and tests/dot_golden.rs.
                y[start + r] = ggml_dot_f32(row, x);
            }
            start += nr;
        }
        Ok(y)
    }
    /// Helper: load bias tensor if present, return vec![0; n] otherwise.
    fn bias_vec(
        &self,
        layer: usize,
        kind: TensorKind,
        n: usize,
        buf: &mut Vec<u8>,
    ) -> io::Result<Vec<f32>> {
        let name = tensor_name(self.arch_variant, kind, layer);
        match self.t_by_name(&name) {
            Ok(t) => {
                let v = t.rows(0, 1, buf)?;
                Ok(v.iter().take(n).copied().collect())
            }
            Err(_) => Ok(vec![0f32; n]),
        }
    }
    /// LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    ///
    /// Both reductions are the reference's, in `src/ggml_norm.rs`, which is where
    /// the details live. What matters here is that this function is *not* the
    /// "accumulate in f64" shape it used to be written as, and that the reason
    /// the old `sum::<f32>()` lines were left alone was wrong in two ways:
    ///
    /// * `ggml_vec_rmsnorm_f32` no longer exists -- RMSNorm is a hand-written
    ///   loop in `ggml_compute_forward_rms_norm_f32` (ops.cpp:3828) that
    ///   `ggml_dot.rs::ggml_rms_norm_scale` already reproduces.
    /// * `ggml_vec_norm_f32` (vec.h:858) is now one line calling
    ///   `ggml_vec_dot_f32`, so it is the *dot* kernel, and LayerNorm does not
    ///   call it at all. `ggml_compute_forward_norm_f32` (ops.cpp:3698) calls
    ///   `ggml_vec_sum_f32` for the mean and `ggml_vec_cvar_f32` for the
    ///   variance, which have *different* shapes: a sequential f64 accumulator
    ///   for one, and a f64 accumulator fed one f32 per group of 8 by a
    ///   horizontal tree for the other.
    ///
    /// So this is the one reduction in the engine where "use f64" is right for
    /// the first half and wrong for the second. Measured against the kernel on
    /// 256 real 2048-wide activation rows of qwen2.5-3b, the old lines put
    /// 61.9% of output elements one or more ulps away; this puts 0% away.
    /// The crate cannot see the difference end to end -- synthetic-gpt2 is the
    /// only fixture that reaches LayerNorm at all -- which is why the pin is
    /// `tests/norm_golden.rs`, a bit-exactness table, and not a tolerance.
    pub fn layernorm(x: &[f32], w: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
        ggml_layernorm(x, w, bias, eps)
    }
    /// GELU activation (approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))))
    pub fn gelu(x: f32) -> f32 {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        0.5f32 * x * (1.0f32 + (c * (x + 0.044715f32 * x * x * x)).tanh())
    }
    /// Dense FFN: silu(gate(x)) * up(x) -> down, or GELU for GPT-2
    fn dense_ffn(&self, layer: usize, x: &[f32], buf: &mut Vec<u8>) -> io::Result<Vec<f32>> {
        self.dense_ffn_act(layer, x, buf, false)
    }
    fn dense_ffn_act(
        &self,
        layer: usize,
        x: &[f32],
        buf: &mut Vec<u8>,
        use_gelu: bool,
    ) -> io::Result<Vec<f32>> {
        let up_t = self.t(TensorKind::FfnUp, layer)?;
        let down_t = self.t(TensorKind::FfnDown, layer)?;
        let n_ff = self.n_ff;
        let mut up = vec![0f32; n_ff];
        let chunk = (262_144 / up_t.row_bytes).max(1);
        let mut s = 0usize;
        while s < n_ff {
            let n = chunk.min(n_ff - s);
            let u = up_t.rows(s, n, buf)?;
            for r in 0..n {
                let ur = &u[r * up_t.row_elems..(r + 1) * up_t.row_elems];
                up[s + r] = ggml_dot_f32(ur, x);
            }
            s += n;
        }
        // GPT-2: GELU(up(x) + b_up) — single dense layer, no gate
        let mut h = vec![0f32; n_ff];
        if use_gelu {
            let up_b = self.bias_vec(layer, TensorKind::FfnUpBias, n_ff, buf)?;
            for i in 0..n_ff {
                h[i] = Self::gelu(up[i] + up_b[i]);
            }
        } else {
            // Llama-style: silu(gate(x)) * up(x)
            let gate_t = self.t(TensorKind::FfnGate, layer)?;
            let mut gate = vec![0f32; n_ff];
            let chunk = (262_144 / gate_t.row_bytes).max(1);
            let mut s = 0usize;
            while s < n_ff {
                let n = chunk.min(n_ff - s);
                let g = gate_t.rows(s, n, buf)?;
                for r in 0..n {
                    let gr = &g[r * gate_t.row_elems..(r + 1) * gate_t.row_elems];
                    gate[s + r] = ggml_dot_f32(gr, x);
                }
                s += n;
            }
            for i in 0..n_ff {
                h[i] = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
            }
            crate::opdump::record("silu_gate", &gate);
        }
        let n_embd = self.n_embd;
        let mut out = vec![0f32; n_embd];
        let down_b = self.bias_vec(layer, TensorKind::FfnDownBias, n_embd, buf)?;
        let chunk = (262_144 / down_t.row_bytes).max(1);
        let mut s = 0usize;
        while s < n_embd {
            let n = chunk.min(n_embd - s);
            let d = down_t.rows(s, n, buf)?;
            for r in 0..n {
                let dr = &d[r * down_t.row_elems..(r + 1) * down_t.row_elems];
                out[s + r] = ggml_dot_f32(dr, &h) + down_b[s + r];
            }
            s += n;
        }
        Ok(out)
    }

    /// Gated GELU FFN: gelu(gate(x)) * up(x) -> down (Gemma3 style)
    fn dense_ffn_gated_gelu(
        &self,
        layer: usize,
        x: &[f32],
        buf: &mut Vec<u8>,
    ) -> io::Result<Vec<f32>> {
        let n_ff = self.n_ff;
        let up_t = self.t(TensorKind::FfnUp, layer)?;
        let gate_t = self.t(TensorKind::FfnGate, layer)?;
        let down_t = self.t(TensorKind::FfnDown, layer)?;
        let mut up = vec![0f32; n_ff];
        let mut gate = vec![0f32; n_ff];
        let chunk = (262_144 / up_t.row_bytes).max(1);
        let mut s = 0usize;
        while s < n_ff {
            let n = chunk.min(n_ff - s);
            let u = up_t.rows(s, n, buf)?;
            let g = gate_t.rows(s, n, buf)?;
            for r in 0..n {
                let ur = &u[r * up_t.row_elems..(r + 1) * up_t.row_elems];
                let gr = &g[r * up_t.row_elems..(r + 1) * up_t.row_elems];
                up[s + r] = ggml_dot_f32(ur, x);
                gate[s + r] = ggml_dot_f32(gr, x);
            }
            s += n;
        }
        let mut h = vec![0f32; n_ff];
        for i in 0..n_ff {
            h[i] = Self::gelu(gate[i]) * up[i];
        }
        let n_embd = self.n_embd;
        let mut out = vec![0f32; n_embd];
        let chunk = (262_144 / down_t.row_bytes).max(1);
        let mut s = 0usize;
        while s < n_embd {
            let n = chunk.min(n_embd - s);
            let d = down_t.rows(s, n, buf)?;
            for r in 0..n {
                let dr = &d[r * down_t.row_elems..(r + 1) * down_t.row_elems];
                out[s + r] = ggml_dot_f32(dr, &h);
            }
            s += n;
        }
        Ok(out)
    }

    /// One expert's FFN: down_e( silu(gate_e x) * (up_e x) ).
    ///
    /// The gate and up projections may arrive either as two tensors or as one
    /// fused `ffn_gate_up_exps` of shape `{n_embd, n_ff*2, n_expert}`.
    /// llama.cpp's `create_tensor_gate_up_exps` (llama-model.cpp:2881) tries the
    /// fused tensor first and creates the split pair only when it is absent, so a
    /// GGUF converted with `convert_hf_to_gguf.py --fuse-gate-up-exps` carries the
    /// fused tensor *instead of* the pair, and `build_moe_ffn` then takes two
    /// `ggml_view_3d`s of one matmul result (llama-graph.cpp:2100-2107):
    ///
    /// ```text
    /// cur = view(gate_up, n_ff, ..., offset 0)          // gate
    /// up  = view(gate_up, n_ff, ..., offset n_ff*nb[0]) // up
    /// ```
    ///
    /// i.e. per expert, rows `[2*e*n_ff_exp, 2*e*n_ff_exp + 2*n_ff_exp)` with
    /// **gate rows first, then up rows** — matching the converter's
    /// `torch.cat([gate, up], dim=1)`. Reading them the other way round is not a
    /// crash and not a small error; it is a different model. The order is
    /// measured: a `deepseek2` fixture built with the concatenation reversed
    /// changes llama.cpp's own logits.
    fn expert(&self, layer: usize, e: usize, x: &[f32], buf: &mut Vec<u8>) -> io::Result<Vec<f32>> {
        // Resolve the row layout first so the arithmetic below is identical for
        // both container forms -- that is what makes the split/fused identity
        // exact rather than merely close. `up_t`/`gate_t` are the same tensor in
        // the fused case and two different ones in the split case.
        let (up_t, gate_t, up_base, gate_base, rpe_up) =
            match self.t(TensorKind::FfnGateUpExps, layer) {
                Ok(t) => {
                    let rpe = t.n_rows / (2 * self.n_experts);
                    let base = e * 2 * rpe;
                    (t.clone(), t, base + rpe, base, rpe)
                }
                Err(_) => {
                    let up = self.t(TensorKind::FfnUpExps, layer)?;
                    let gate = self.t(TensorKind::FfnGateExps, layer)?;
                    let rpe = up.n_rows / self.n_experts;
                    (up, gate, e * rpe, e * rpe, rpe)
                }
            };
        let down_t = self.t(TensorKind::FfnDownExps, layer)?;
        let rpe_down = down_t.n_rows / self.n_experts;
        let mut up = vec![0f32; rpe_up];
        let mut gate = vec![0f32; rpe_up];
        let chunk = (262_144 / up_t.row_bytes).max(1);
        let mut s = 0usize;
        while s < rpe_up {
            let n = chunk.min(rpe_up - s);
            let u = up_t.rows(up_base + s, n, buf)?;
            let g = gate_t.rows(gate_base + s, n, buf)?;
            for r in 0..n {
                let ur = &u[r * up_t.row_elems..(r + 1) * up_t.row_elems];
                let gr = &g[r * gate_t.row_elems..(r + 1) * gate_t.row_elems];
                up[s + r] = ggml_dot_f32(ur, x);
                gate[s + r] = ggml_dot_f32(gr, x);
            }
            s += n;
        }
        let mut h = vec![0f32; rpe_up];
        for i in 0..rpe_up {
            h[i] = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
        }
        let mut out = vec![0f32; rpe_down];
        let chunk = (262_144 / down_t.row_bytes).max(1);
        let mut s = 0usize;
        while s < rpe_down {
            let n = chunk.min(rpe_down - s);
            let d = down_t.rows(e * rpe_down + s, n, buf)?;
            for r in 0..n {
                let dr = &d[r * down_t.row_elems..(r + 1) * down_t.row_elems];
                out[s + r] = ggml_dot_f32(dr, &h);
            }
            s += n;
        }
        Ok(out)
    }
    /// DeepSeek-V2 MLA forward pass (single token, streaming).
    /// Uses absorption-optimized MLA: Q = concat(wk_b_absorbed, q_pe),
    /// K = concat(kv_cmpr, k_pe) stored in latent cache, V expanded from latent via wv_b.
    /// After absorption, attention is MQA (1 KV group).
    fn forward_deepseek2(
        &self,
        token: usize,
        pos: usize,
        mla_k_cache: &mut [Vec<f32>],
        buf: &mut Vec<u8>,
    ) -> io::Result<Vec<f32>> {
        let n_embd = self.n_embd;
        let n_heads = self.n_heads;
        let n_lora_kv = self.n_lora_kv;
        let rope_dim = self.rope_dim;
        let key_length_mla = self.key_length_mla;
        let value_length_mla = self.value_length_mla;
        let n_ff_exp = self.n_ff_exp;
        let n_expert_shared = self.n_expert_shared;
        let n_embd_head_qk_nope = key_length_mla - rope_dim;
        let dk = n_lora_kv + rope_dim; // latent K dimension per position
        let eps = self.eps as f32;
        debug_assert!(
            key_length_mla > rope_dim,
            "key_length_mla > rope_dim required for MLA"
        );

        // token embedding
        let tok_t = self.t(TensorKind::TokenEmbedd, 0)?;
        let mut x = tok_t.rows(token, 1, buf)?;
        x.truncate(n_embd);

        #[allow(clippy::needless_range_loop)]
        for layer in 0..self.n_layers {
            // ---- attention ----
            let norm_t = self.t(TensorKind::AttnNorm, layer)?;
            let w = norm_t.rows(0, 1, buf)?;
            let t = rmsnorm(&x, &w, eps);

            // Q projection: lite vs non-lite
            let is_lite = self.n_lora_q == 0;
            let q = if is_lite {
                let wq_t = self.t(TensorKind::AttnQA, layer)?;
                self.matvec(&wq_t, &t, buf)?
            } else {
                let wq_a_t = self.t(TensorKind::AttnQA, layer)?;
                let q_a = self.matvec(&wq_a_t, &t, buf)?;
                let norm_q_t = self.t(TensorKind::AttnQANorm, layer)?;
                let q_norm_w = norm_q_t.rows(0, 1, buf)?;
                let q_a = rmsnorm(&q_a, &q_norm_w, eps);
                let wq_b_t = self.t(TensorKind::AttnQB, layer)?;
                self.matvec(&wq_b_t, &q_a, buf)?
            };

            // Split Q into per-head q_nope + q_pe
            let mut q_nope = vec![0f32; n_heads * n_embd_head_qk_nope];
            let mut q_pe = vec![0f32; n_heads * rope_dim];
            for h in 0..n_heads {
                let h_off = h * key_length_mla;
                for i in 0..n_embd_head_qk_nope {
                    q_nope[h * n_embd_head_qk_nope + i] = q[h_off + i];
                }
                for i in 0..rope_dim {
                    q_pe[h * rope_dim + i] = q[h_off + n_embd_head_qk_nope + i];
                }
            }

            // KV compression: x @ wkv_a_mqa → [kv_lora_rank + rope_dim]
            let wkv_a_t = self.t(TensorKind::AttnKVAMQA, layer)?;
            let kv_cmpr_pe = self.matvec(&wkv_a_t, &t, buf)?;
            let mut kv_cmpr: Vec<f32> = kv_cmpr_pe[..n_lora_kv].to_vec();
            let mut k_pe: Vec<f32> = kv_cmpr_pe[n_lora_kv..n_lora_kv + rope_dim].to_vec();

            // RMSNorm KV compressed latent
            let kv_norm_t = self.t(TensorKind::AttnKVANorm, layer)?;
            let kv_norm_w = kv_norm_t.rows(0, 1, buf)?;
            kv_cmpr = rmsnorm(&kv_cmpr, &kv_norm_w, eps);

            // RoPE on both q_pe (per head) and k_pe (shared)
            for h in 0..n_heads {
                rope(
                    &mut q_pe[h * rope_dim..(h + 1) * rope_dim],
                    pos,
                    self.rope_theta,
                    self.rope_type,
                );
            }
            rope(&mut k_pe, pos, self.rope_theta, self.rope_type);

            // Absorption: wk_b[n_embd_head_qk_nope, n_lora_kv, n_head] @ q_nope
            // absorbed_q[h, k] = sum_i wk_b[i, k, h] * q_nope[h, i]
            let wk_b_t = self.t(TensorKind::AttnKB, layer)?;
            let wk_b_rpe = wk_b_t.row_elems; // n_embd_head_qk_nope
            let rows_per_head = n_lora_kv;
            let mut absorbed_q = vec![0f32; n_heads * n_lora_kv];
            let chunk = (262_144 / wk_b_t.row_bytes).max(1);
            for h in 0..n_heads {
                let q_nope_h = &q_nope[h * n_embd_head_qk_nope..(h + 1) * n_embd_head_qk_nope];
                let mut s = 0usize;
                while s < n_lora_kv {
                    let n = chunk.min(n_lora_kv - s);
                    let rows = wk_b_t.rows(h * rows_per_head + s, n, buf)?;
                    for r in 0..n {
                        let row = &rows[r * wk_b_rpe..(r + 1) * wk_b_rpe];
                        absorbed_q[h * n_lora_kv + s + r] = ggml_dot_f32(row, q_nope_h);
                    }
                    s += n;
                }
            }

            // Store latent K = [kv_cmpr, k_pe] in MLA cache (MQA: 1 KV group)
            let mla_stride = dk;
            let kc = &mut mla_k_cache[layer];
            for i in 0..n_lora_kv {
                kc[pos * mla_stride + i] = kv_cmpr[i];
            }
            for i in 0..rope_dim {
                kc[pos * mla_stride + n_lora_kv + i] = k_pe[i];
            }

            // MQA attention: n_heads share one KV, V expanded from latent via wv_b
            let kq_scale = self.deepseek2_kq_scale();

            let mut att_out = vec![0f32; n_heads * value_length_mla];
            let wv_b_t = self.t(TensorKind::AttnVB, layer)?;
            let wv_b_rpe = wv_b_t.row_elems; // n_lora_kv
            let wv_b_rph = value_length_mla;

            for h in 0..n_heads {
                let a_off = h * n_lora_kv;
                let p_off = h * rope_dim;

                // scores over all cached positions
                let mut scores = vec![0f32; pos + 1];
                for (p, score) in scores.iter_mut().enumerate().take(pos + 1) {
                    let k_off = p * mla_stride;
                    let mut s = 0.0f32;
                    for i in 0..n_lora_kv {
                        s += absorbed_q[a_off + i] * kc[k_off + i];
                    }
                    for i in 0..rope_dim {
                        s += q_pe[p_off + i] * kc[k_off + n_lora_kv + i];
                    }
                    *score = s * kq_scale;
                }
                let mut probs = softmax(&mut scores);

                // weighted sum of compressed KV across positions
                let mut cmpr_weighted = vec![0f32; n_lora_kv];
                for (p, _prob) in probs.iter_mut().enumerate().take(pos + 1) {
                    let k_off = p * mla_stride;
                    let w = _prob;
                    for i in 0..n_lora_kv {
                        cmpr_weighted[i] += *w * kc[k_off + i];
                    }
                }

                // V expansion: wv_b[:,:,h]^T @ cmpr_weighted → [value_length_mla]
                let v_off = h * value_length_mla;
                let chunk = (262_144 / wv_b_t.row_bytes).max(1);
                let mut s = 0usize;
                while s < value_length_mla {
                    let n = chunk.min(value_length_mla - s);
                    let rows = wv_b_t.rows(h * wv_b_rph + s, n, buf)?;
                    for r in 0..n {
                        let row = &rows[r * wv_b_rpe..(r + 1) * wv_b_rpe];
                        att_out[v_off + s + r] = ggml_dot_f32(row, &cmpr_weighted);
                    }
                    s += n;
                }
            }

            // Output projection
            let wo_t = self.t(TensorKind::AttnOutput, layer)?;
            let wo_out = self.matvec(&wo_t, &att_out, buf)?;
            for i in 0..n_embd {
                x[i] += wo_out[i];
            }

            // ---- FFN ----
            let fnorm_t = self.t(TensorKind::FfnNorm, layer)?;
            let w_ffn = fnorm_t.rows(0, 1, buf)?;
            let t_ffn = rmsnorm(&x, &w_ffn, eps);

            if layer < self.leading_dense_layers {
                // Dense-lead FFN: SiLU gate + up + down
                let n_ff = self.n_ff;
                let up_t = self.t(TensorKind::FfnUp, layer)?;
                let gate_t = self.t(TensorKind::FfnGate, layer)?;
                let down_t = self.t(TensorKind::FfnDown, layer)?;
                let mut up = vec![0f32; n_ff];
                let mut gate = vec![0f32; n_ff];
                let chunk = (262_144 / up_t.row_bytes).max(1);
                let mut s = 0usize;
                while s < n_ff {
                    let n = chunk.min(n_ff - s);
                    let u = up_t.rows(s, n, buf)?;
                    let g = gate_t.rows(s, n, buf)?;
                    for r in 0..n {
                        let ur = &u[r * up_t.row_elems..(r + 1) * up_t.row_elems];
                        let gr = &g[r * up_t.row_elems..(r + 1) * up_t.row_elems];
                        up[s + r] = ggml_dot_f32(ur, &t_ffn);
                        gate[s + r] = ggml_dot_f32(gr, &t_ffn);
                    }
                    s += n;
                }
                let mut h = vec![0f32; n_ff];
                for i in 0..n_ff {
                    h[i] = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
                }
                let mut ff_out = vec![0f32; n_embd];
                let chunk = (262_144 / down_t.row_bytes).max(1);
                let mut s = 0usize;
                while s < n_embd {
                    let n = chunk.min(n_embd - s);
                    let d = down_t.rows(s, n, buf)?;
                    for r in 0..n {
                        let dr = &d[r * down_t.row_elems..(r + 1) * down_t.row_elems];
                        ff_out[s + r] = ggml_dot_f32(dr, &h);
                    }
                    s += n;
                }
                for i in 0..n_embd {
                    x[i] += ff_out[i];
                }
            } else {
                // MoE with shared experts
                let n_used = self.n_experts_used;
                let router_t = self.t(TensorKind::FfnGateInp, layer)?;
                let router_logits = self.matvec(&router_t, &t_ffn, buf)?;
                let top = moe_route_weights(
                    &router_logits,
                    n_used,
                    self.moe_norm_w(),
                    self.moe_scale(),
                    self.moe_gating(),
                );
                let mut moe_out = vec![0f32; n_embd];
                for (e, wgt) in top {
                    let e_out = self.expert(layer, e, &t_ffn, buf)?;
                    for i in 0..n_embd {
                        moe_out[i] += wgt * e_out[i];
                    }
                }
                // Shared experts
                let n_sh_dim = n_ff_exp * n_expert_shared;
                let up_sh_t = self.t(TensorKind::FfnUpShexp, layer)?;
                let gate_sh_t = self.t(TensorKind::FfnGateShexp, layer)?;
                let down_sh_t = self.t(TensorKind::FfnDownShexp, layer)?;
                let mut sh_up = vec![0f32; n_sh_dim];
                let mut sh_gate = vec![0f32; n_sh_dim];
                let chunk = (262_144 / up_sh_t.row_bytes).max(1);
                let mut s = 0usize;
                while s < n_sh_dim {
                    let n = chunk.min(n_sh_dim - s);
                    let u = up_sh_t.rows(s, n, buf)?;
                    let g = gate_sh_t.rows(s, n, buf)?;
                    for r in 0..n {
                        let ur = &u[r * up_sh_t.row_elems..(r + 1) * up_sh_t.row_elems];
                        let gr = &g[r * gate_sh_t.row_elems..(r + 1) * gate_sh_t.row_elems];
                        sh_up[s + r] = ggml_dot_f32(ur, &t_ffn);
                        sh_gate[s + r] = ggml_dot_f32(gr, &t_ffn);
                    }
                    s += n;
                }
                let mut sh_h = vec![0f32; n_sh_dim];
                for i in 0..n_sh_dim {
                    sh_h[i] = sh_gate[i] / (1.0 + (-sh_gate[i]).exp()) * sh_up[i];
                }
                let mut sh_out = vec![0f32; n_embd];
                let chunk = (262_144 / down_sh_t.row_bytes).max(1);
                let mut s = 0usize;
                while s < n_embd {
                    let n = chunk.min(n_embd - s);
                    let d = down_sh_t.rows(s, n, buf)?;
                    for r in 0..n {
                        let dr = &d[r * down_sh_t.row_elems..(r + 1) * down_sh_t.row_elems];
                        sh_out[s + r] = ggml_dot_f32(dr, &sh_h);
                    }
                    s += n;
                }
                for i in 0..n_embd {
                    x[i] += moe_out[i] + sh_out[i];
                }
            }
        }
        // final norm + logits
        let onorm_t = self.t(TensorKind::OutputNorm, 0)?;
        let w = onorm_t.rows(0, 1, buf)?;
        let x = rmsnorm(&x, &w, eps);
        let out_t = self.output_t()?;
        self.matvec(&out_t, &x, buf)
    }

    /// Gemma3 forward pass: sliding window + global alternating attention
    fn forward_gemma3(
        &self,
        token: usize,
        pos: usize,
        k_cache: &mut [Vec<f32>],
        v_cache: &mut [Vec<f32>],
        buf: &mut Vec<u8>,
    ) -> io::Result<Vec<f32>> {
        let n_embd = self.n_embd;
        let head_dim = self.head_dim;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let kv_group = n_heads / n_kv_heads;
        let kv_stride = n_kv_heads * head_dim;
        let eps = self.eps as f32;
        let n_swa = self.n_swa;
        let swa_period = self.swa_pattern;
        // llama.cpp *computes* f_attention_scale from the hparams
        // (gemma3.cpp:76) and never reads a `gemma3.attention.scale` key, so a
        // file carrying one is telling us nothing. Recomputing it here keeps the
        // two in agreement on a file where they differ.
        let f_scale = gemma3_attn_scale(self.n_layers, self.n_embd, self.n_heads, head_dim);
        // Gemma scales its embeddings by sqrt(n_embd) before the first norm
        // (gemma3.cpp:93). RMSNorm is scale-invariant so this cannot be absorbed
        // anywhere: it is the difference between reading the embedding table and
        // reading it times 6.93 on Gemma-3-1B.
        let embd_scale = (n_embd as f32).sqrt();
        // token embedding
        let tok_t = self.t(TensorKind::TokenEmbedd, 0)?;
        let mut x = tok_t.rows(token, 1, buf)?;
        x.truncate(n_embd);
        for v in x.iter_mut() {
            *v *= embd_scale;
        }
        for layer in 0..self.n_layers {
            let is_swa = gemma3_is_swa(layer, n_swa, swa_period);
            // ---- attention ----
            let norm_t = self.t(TensorKind::AttnNorm, layer)?;
            let w = norm_t.rows(0, 1, buf)?;
            let t = rmsnorm(&x, &w, eps);
            // Separate Q, K, V projections (not fused)
            let wq = self.t(TensorKind::AttnQ, layer)?;
            let wk = self.t(TensorKind::AttnK, layer)?;
            let wv = self.t(TensorKind::AttnV, layer)?;
            let q = self.matvec(&wq, &t, buf)?;
            let k = self.matvec(&wk, &t, buf)?;
            let v = self.matvec(&wv, &t, buf)?;
            // Per-head QK-norm (mandatory for Gemma3)
            let qw_t = self.t(TensorKind::AttnQKNorm, layer)?;
            let qw_v = qw_t.rows(0, 1, buf)?;
            let kw_t = self.t(TensorKind::AttnKNorm, layer)?;
            let kw_v = kw_t.rows(0, 1, buf)?;
            let mut q = qk_norm_row(&q, &qw_v, head_dim, eps);
            let mut k = qk_norm_row(&k, &kw_v, head_dim, eps);
            // RoPE: SWA layers use the SWA base, global layers the global base.
            // Both defaults come from llama.cpp rather than from the file —
            // `rope_freq_base_swa` is 10000.0 upstream and `rope.freq_base` is
            // what the converter wrote — and a converted Gemma-3 GGUF only ever
            // carries the latter. See the `rope_freq_base_swa` read in
            // `Model::from_file`.
            let rope_theta_local = if is_swa {
                self.rope_freq_base_swa
            } else {
                self.rope_theta
            };
            for h in 0..n_heads {
                let s = h * head_dim;
                rope(
                    &mut q[s..s + head_dim],
                    pos,
                    rope_theta_local,
                    self.rope_type,
                );
            }
            // RoPE on K (n_kv_heads format before GQA expansion)
            for h in 0..n_kv_heads {
                let s = h * head_dim;
                rope(
                    &mut k[s..s + head_dim],
                    pos,
                    rope_theta_local,
                    self.rope_type,
                );
            }
            // Q carries `f_attention_scale` and nothing else does. llama.cpp
            // applies it after RoPE and after the QK-norm (gemma3.cpp:154).
            for v in q.iter_mut() {
                *v *= f_scale;
            }
            // The attention output is `n_head * key_length` wide, which is NOT
            // n_embd. Gemma-3 sizes `attn_output.weight` as
            // `{n_embd_head_k * n_head, n_embd}` precisely because of this, and
            // on every released checkpoint (1B: 4 x 256 = 1024 vs n_embd 1152)
            // the two differ. `k_full`/`v_full` therefore have to be sized by
            // the head geometry too, not by n_embd.
            let qkv_width = n_heads * head_dim;
            // GQA broadcast: query head `h` reads KV head `h / n_rep`, where
            // `n_rep = n_heads / n_kv_heads` — llama.cpp's `kv_head = h / n_rep`
            // (llama-context.cpp, `llama_build_attn_mha`) and HF's `repeat_kv`,
            // which expands `[b, n_kv, 1, s, d]` to `[b, n_kv, n_rep, s, d]`.
            //
            // It is `h / n_rep`, NOT `h % n_kv_heads`, and the boundary of that
            // claim is not where intuition puts it. The two pairings agree only
            // for MHA (`n_rep == 1`) and for a single KV head (`n_kv_heads == 1`,
            // where both send every query head to head 0). `n_kv_heads == 2` is
            // NOT one of the coincidence cases: 4 query heads over 2 KV heads
            // group as [0,0,1,1] and interleave as [0,1,0,1]. An earlier version
            // of this comment claimed the opposite, which is the same mistake as
            // the code — it would have licensed exactly the fixture set that
            // cannot see this bug. `tests/gqa_mapping.rs` states the coincidence
            // set as an executable rule, over 45 shapes, and states the real
            // models' geometries from `models/*.gguf` beside it.
            //
            // Getting it wrong is not a small error: it permutes which query head
            // sees which cache row, every layer, and the logits come out
            // uncorrelated with the reference (rel L2 ~1.2, cos ~0.35) rather
            // than slightly off. Measured both ways on the real
            // qwen2.5-3b-instruct-q4_k_m.gguf — 16 query heads over 2 KV heads,
            // `n_rep = 8`, so the two pairings differ on 8 of 16 heads — by
            // `scripts/real_model_parity.py`:
            //
            //   h / n_rep      rel L2 3.7e-2 .. 1.3e-1   cos 0.9916 .. 0.9993
            //                 argmax 16/17 positions, top-10 sets 8/10 shared,
            //                 next token " Paris" (llama.cpp agrees)
            //   h % n_kv_heads rel L2 1.0 .. 2.1           cos -0.30 .. 0.53
            //                 argmax 1/17, top-10 sets ~0 shared, next token
            //                 " 门 treaty31pokite"
            //
            // The 3.7e-2..1.3e-1 band is llama.cpp's own Q4_K_M sensitivity, not
            // a disagreement: ggml rounds activations to Q8_K before every
            // quantized dot product, which measures 7.0e-3 per matvec on this
            // model's own weights (`engine/tools/probe_quant_matvec.c`) and
            // compounds over the ~250 matvecs of a token.
            let k_full = if n_kv_heads == n_heads {
                k
            } else {
                let mut k2 = vec![0f32; qkv_width];
                for h in 0..n_heads {
                    let kv_h = h / kv_group;
                    let src = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
                    let dst = &mut k2[h * head_dim..(h + 1) * head_dim];
                    dst.copy_from_slice(src);
                }
                k2
            };
            let v_full = if n_kv_heads == n_heads {
                v
            } else {
                let mut v2 = vec![0f32; qkv_width];
                for h in 0..n_heads {
                    let kv_h = h / kv_group;
                    let src = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
                    let dst = &mut v2[h * head_dim..(h + 1) * head_dim];
                    dst.copy_from_slice(src);
                }
                v2
            };
            // Write KV cache (standard MHA/GQA layout)
            for kv_h in 0..n_kv_heads {
                let ks = kv_h * kv_group * head_dim;
                for p in 0..head_dim {
                    k_cache[layer][pos * kv_stride + kv_h * head_dim + p] = k_full[ks + p];
                    v_cache[layer][pos * kv_stride + kv_h * head_dim + p] = v_full[ks + p];
                }
            }
            // Attention over relevant positions
            let (attn_start, attn_range) = if is_swa {
                let range = n_swa.min(pos + 1);
                let start = pos.saturating_add(1).saturating_sub(range);
                (start, range)
            } else {
                (0, pos + 1)
            };
            // Q already carries `f_attention_scale` (applied after RoPE above, as
            // llama.cpp does), so the score here must NOT scale a second time.
            // Doing both squares the factor: on a 14-wide head that is 1/14
            // instead of 1/sqrt(14), a 3.7x error on every attention logit. It is
            // invisible for the first token of any prompt — one key means softmax
            // returns 1 whatever the logits are — so it survives any single-token
            // smoke test and shows up as a ~0.5 relative L2 from position 1 on.
            // The same score line appears in the other forward passes below; they
            // scale there and nowhere here, which is why this one is easy to get
            // wrong when copied.
            let mut att_out = vec![0f32; n_heads * head_dim];
            for h in 0..n_heads {
                let kv_h = h / kv_group;
                let mut scores = vec![0f32; attn_range];
                for (oi, p) in (attn_start..=pos).enumerate() {
                    let kr = &k_cache[layer]
                        [p * kv_stride + kv_h * head_dim..p * kv_stride + (kv_h + 1) * head_dim];
                    let qh = &q[h * head_dim..(h + 1) * head_dim];
                    scores[oi] = ggml_dot_f32(kr, qh);
                }
                let probs = softmax(&mut scores);
                for (oi, p) in (attn_start..=pos).enumerate() {
                    let vr = &v_cache[layer]
                        [p * kv_stride + kv_h * head_dim..p * kv_stride + (kv_h + 1) * head_dim];
                    let pr = probs[oi];
                    for (o, vv) in att_out[h * head_dim..(h + 1) * head_dim].iter_mut().zip(vr) {
                        *o += pr * vv;
                    }
                }
            }
            // Output projection
            let wo_t = self.t(TensorKind::AttnOutput, layer)?;
            let wo_out = self.matvec(&wo_t, &att_out, buf)?;
            // Post-attention norm (before residual add)
            let post_attn_t = self.t(TensorKind::AttnPostNorm, layer)?;
            let post_attn_w = post_attn_t.rows(0, 1, buf)?;
            let wo_normed = rmsnorm(&wo_out, &post_attn_w, eps);
            for i in 0..n_embd {
                x[i] += wo_normed[i];
            }
            // ---- FFN (gated GELU) ----
            let fnorm_t = self.t(TensorKind::FfnNorm, layer)?;
            let w_ffn = fnorm_t.rows(0, 1, buf)?;
            let t_ffn = rmsnorm(&x, &w_ffn, eps);
            let ff_out = self.dense_ffn_gated_gelu(layer, &t_ffn, buf)?;
            // Post-FFN norm (before residual add)
            let post_ffn_t = self.t(TensorKind::FfnPostNorm, layer)?;
            let post_ffn_w = post_ffn_t.rows(0, 1, buf)?;
            let ff_normed = rmsnorm(&ff_out, &post_ffn_w, eps);
            for i in 0..n_embd {
                x[i] += ff_normed[i];
            }
        }
        // Final norm + logits
        let onorm_t = self.t(TensorKind::OutputNorm, 0)?;
        let w = onorm_t.rows(0, 1, buf)?;
        let x = rmsnorm(&x, &w, eps);
        let out_t = self.output_t()?;
        let mut logits = self.matvec(&out_t, &x, buf)?;
        // Final logit softcapping: tanh(logits / c) * c
        if self.f_final_logit_softcapping > 0.0 {
            let c = self.f_final_logit_softcapping;
            for v in &mut logits {
                *v = ((*v / c).tanh()) * c;
            }
        }
        Ok(logits)
    }

    fn forward_k2_horizon(
        &self,
        token: usize,
        pos: usize,
        k_cache: &mut [Vec<f32>],
        v_cache: &mut [Vec<f32>],
        buf: &mut Vec<u8>,
    ) -> io::Result<Vec<f32>> {
        let n_embd = self.n_embd;
        let head_dim = self.head_dim;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let kv_group = n_heads / n_kv_heads;
        let kv_stride = n_kv_heads * head_dim;
        let eps = self.eps as f32;
        let n_norm_groups = self.n_norm_groups.max(1);
        // token embedding
        let tok_t = self.t(TensorKind::TokenEmbedd, 0)?;
        let mut x = tok_t.rows(token, 1, buf)?;
        x.truncate(n_embd);
        // Static gate detection: check for attn_gate tensor at layer 0
        let attn_gate_name = tensor_name(self.arch_variant, TensorKind::AttnGate, 0);
        let has_attn_gate = self.gf.tensors.contains_key(&attn_gate_name);
        // Check for MoVA value experts
        let _has_mova = self.n_value_expert > 0;
        for layer in 0..self.n_layers {
            // ---- attention ----
            let norm_t = self.t(TensorKind::AttnNorm, layer)?;
            let w = norm_t.rows(0, 1, buf)?;
            // Use grouped RMSNorm if n_norm_groups > 1, else standard RMSNorm
            let t = if n_norm_groups > 1 {
                grouped_rmsnorm(&x, &w, n_norm_groups, eps)
            } else {
                rmsnorm(&x, &w, eps)
            };

            // Optionally apply attention gate before QKV projection
            let t_attn = if has_attn_gate {
                let gate_t = self.t(TensorKind::AttnGate, layer)?;
                let mut gate = self.matvec(&gate_t, &t, buf)?;
                let ln2 = std::f32::consts::LN_2;
                let inv_ln2 = 1.0 / ln2;
                for g in &mut gate {
                    let xv = *g * ln2;
                    *g = if xv > 30.0 { xv } else { xv.exp().ln_1p() };
                    *g *= inv_ln2;
                }
                // softplus(gate * ln2) * (1/ln2) applied per-element to normalized input
                t.iter().zip(gate).map(|(tv, gv)| tv * gv).collect()
            } else {
                t.clone()
            };

            // Q, K, V projections
            let wq = self.t(TensorKind::AttnQ, layer)?;
            let wk = self.t(TensorKind::AttnK, layer)?;
            let wv = self.t(TensorKind::AttnV, layer)?;
            let mut q = self.matvec(&wq, &t_attn, buf)?;
            let mut k = self.matvec(&wk, &t, buf)?;
            let v = self.matvec(&wv, &t, buf)?;

            // Optional per-head QK-norm (grouped RMSNorm with head_dim groups)
            if self.qk_norm {
                let qw_t = self.t(TensorKind::AttnQKNorm, layer)?;
                let qw_v = qw_t.rows(0, 1, buf)?;
                let kw_name = tensor_name(self.arch_variant, TensorKind::AttnQKNorm, layer)
                    .replace("attn_q_norm", "attn_k_norm");
                if self.gf.tensors.contains_key(&kw_name) {
                    let kw_t = self.t_by_name(&kw_name)?;
                    let kw_v = kw_t.rows(0, 1, buf)?;
                    q = qk_norm_row(&q, &qw_v, head_dim, eps);
                    k = qk_norm_row(&k, &kw_v, head_dim, eps);
                }
            }

            // RoPE on Q (n_heads) and K (n_kv_heads, before GQA expansion)
            for h in 0..n_heads {
                let s = h * head_dim;
                rope(
                    &mut q[s..s + head_dim],
                    pos,
                    self.rope_theta,
                    self.rope_type,
                );
            }
            for h in 0..n_kv_heads {
                let s = h * head_dim;
                rope(
                    &mut k[s..s + head_dim],
                    pos,
                    self.rope_theta,
                    self.rope_type,
                );
            }

            // Write KV cache BEFORE GQA expansion (n_kv_heads format)
            for kv_h in 0..n_kv_heads {
                for p in 0..head_dim {
                    k_cache[layer][pos * kv_stride + kv_h * head_dim + p] = k[kv_h * head_dim + p];
                    v_cache[layer][pos * kv_stride + kv_h * head_dim + p] = v[kv_h * head_dim + p];
                }
            }

            // Attention (reads from cache in n_kv_heads format, uses GQA group indexing)
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut att_out = vec![0f32; n_embd];
            for h in 0..n_heads {
                let kv_h = h / kv_group;
                let mut scores = vec![0f32; pos + 1];
                for p in 0..=pos {
                    let kr = &k_cache[layer]
                        [p * kv_stride + kv_h * head_dim..p * kv_stride + (kv_h + 1) * head_dim];
                    let qh = &q[h * head_dim..(h + 1) * head_dim];
                    scores[p] = ggml_dot_f32(kr, qh) * scale;
                }
                let probs = softmax(&mut scores);
                for p in 0..=pos {
                    let vr = &v_cache[layer]
                        [p * kv_stride + kv_h * head_dim..p * kv_stride + (kv_h + 1) * head_dim];
                    let pr = probs[p];
                    for (o, vv) in att_out[h * head_dim..(h + 1) * head_dim].iter_mut().zip(vr) {
                        *o += pr * vv;
                    }
                }
            }

            // Output projection
            let wo_t = self.t(TensorKind::AttnOutput, layer)?;
            let wo_out = self.matvec(&wo_t, &att_out, buf)?;
            for i in 0..n_embd {
                x[i] += wo_out[i];
            }

            // ---- FFN ----
            let fnorm_t = self.t(TensorKind::FfnNorm, layer)?;
            let w_ffn = fnorm_t.rows(0, 1, buf)?;
            let t_ffn = if n_norm_groups > 1 {
                grouped_rmsnorm(&x, &w_ffn, n_norm_groups, eps)
            } else {
                rmsnorm(&x, &w_ffn, eps)
            };

            if layer < self.leading_dense_layers {
                // Dense-lead FFN: SiLU gate + up + down
                let ff_out = self.dense_ffn(layer, &t_ffn, buf)?;
                for i in 0..n_embd {
                    x[i] += ff_out[i];
                }
            } else {
                // MoE FFN
                let has_router = self.gf.tensors.contains_key(&tensor_name(
                    self.arch_variant,
                    TensorKind::FfnGateInp,
                    layer,
                ));
                if self.n_experts > 0 && has_router {
                    let router_t = self.t(TensorKind::FfnGateInp, layer)?;
                    let router_logits = self.matvec(&router_t, &t_ffn, buf)?;
                    let top = moe_route_weights(
                        &router_logits,
                        self.n_experts_used,
                        self.moe_norm_w(),
                        self.moe_scale(),
                        self.moe_gating(),
                    );
                    let mut moe_out = vec![0f32; n_embd];
                    for (e, wgt) in top {
                        let e_out = self.expert(layer, e, &t_ffn, buf)?;
                        for i in 0..n_embd {
                            moe_out[i] += wgt * e_out[i];
                        }
                    }
                    // Shared experts (if present)
                    let has_shexp = self.n_expert_shared > 0
                        && self.gf.tensors.contains_key(&tensor_name(
                            self.arch_variant,
                            TensorKind::FfnGateShexp,
                            layer,
                        ));
                    if has_shexp {
                        let shexp_gate_t = self.t(TensorKind::FfnGateShexp, layer)?;
                        let shexp_up_t = self.t(TensorKind::FfnUpShexp, layer)?;
                        let shexp_down_t = self.t(TensorKind::FfnDownShexp, layer)?;
                        let shexp_n_ff = self.n_ff_exp;
                        let mut shexp_gate = vec![0f32; shexp_n_ff];
                        let mut shexp_up = vec![0f32; shexp_n_ff];
                        let chunk = (262_144 / shexp_up_t.row_bytes).max(1);
                        let mut s = 0usize;
                        while s < shexp_n_ff {
                            let n = chunk.min(shexp_n_ff - s);
                            let sg = shexp_gate_t.rows(s, n, buf)?;
                            let su = shexp_up_t.rows(s, n, buf)?;
                            for r in 0..n {
                                let sgr =
                                    &sg[r * shexp_up_t.row_elems..(r + 1) * shexp_up_t.row_elems];
                                let sur =
                                    &su[r * shexp_up_t.row_elems..(r + 1) * shexp_up_t.row_elems];
                                shexp_gate[s + r] = ggml_dot_f32(sgr, &t_ffn);
                                shexp_up[s + r] = ggml_dot_f32(sur, &t_ffn);
                            }
                            s += n;
                        }
                        let mut shexp_h = vec![0f32; shexp_n_ff];
                        for i in 0..shexp_n_ff {
                            shexp_h[i] =
                                shexp_gate[i] / (1.0 + (-shexp_gate[i]).exp()) * shexp_up[i];
                        }
                        let mut shexp_out = vec![0f32; n_embd];
                        let chunk = (262_144 / shexp_down_t.row_bytes).max(1);
                        let mut s = 0usize;
                        while s < n_embd {
                            let n = chunk.min(n_embd - s);
                            let sd = shexp_down_t.rows(s, n, buf)?;
                            for r in 0..n {
                                let sdr = &sd
                                    [r * shexp_down_t.row_elems..(r + 1) * shexp_down_t.row_elems];
                                shexp_out[s + r] = ggml_dot_f32(sdr, &shexp_h);
                            }
                            s += n;
                        }
                        for i in 0..n_embd {
                            moe_out[i] += shexp_out[i];
                        }
                    }
                    for i in 0..n_embd {
                        x[i] += moe_out[i];
                    }
                } else {
                    let ff_out = self.dense_ffn(layer, &t_ffn, buf)?;
                    for i in 0..n_embd {
                        x[i] += ff_out[i];
                    }
                }
            }

            if self.debug_checksums {
                let s: f32 = x.iter().map(|v| v.abs()).sum();
                eprintln!("L{layer} x_sum={s:.4}");
            }
        }
        // Final norm + logits
        let onorm_t = self.t(TensorKind::OutputNorm, 0)?;
        let w = onorm_t.rows(0, 1, buf)?;
        let x = if n_norm_groups > 1 {
            grouped_rmsnorm(&x, &w, n_norm_groups, eps)
        } else {
            rmsnorm(&x, &w, eps)
        };
        let out_t = self.output_t()?;
        self.matvec(&out_t, &x, buf)
    }

    /// Run a whole prompt through `forward`, returning every position's logits.
    ///
    /// This mirrors exactly what `tools/gen_ref_logits.c` asks llama.cpp to
    /// produce: every position emits logits, not just the last one, so the two
    /// can be diffed position by position. It is the entry point behind
    /// `--dump-logits` and behind `tests/ref_parity.rs`.
    pub fn prefill_logits(&self, tokens: &[usize]) -> io::Result<Vec<Vec<f32>>> {
        let mut rows = Vec::with_capacity(tokens.len());
        let mut buf = Vec::with_capacity(262_144);
        if self.arch_variant == ArchVariant::DeepSeek2 {
            let mla_stride = self.n_lora_kv + self.rope_dim;
            let mut k_cache = vec![vec![0.0f32; mla_stride * tokens.len().max(1)]; self.n_layers];
            let mut v_cache: Vec<Vec<f32>> = vec![vec![]; self.n_layers];
            for (pos, &tok) in tokens.iter().enumerate() {
                rows.push(self.forward(tok, pos, &mut k_cache, &mut v_cache, &mut buf)?);
            }
        } else {
            let kv_stride = self.n_kv_heads * self.head_dim;
            // forward() indexes the cache as pos * kv_stride, so the whole
            // prompt must fit — same sizing rule as Model::generate.
            let n_pos = tokens.len().max(1);
            let mut k_cache = vec![vec![0.0f32; kv_stride * n_pos]; self.n_layers];
            let mut v_cache = vec![vec![0.0f32; kv_stride * n_pos]; self.n_layers];
            for (pos, &tok) in tokens.iter().enumerate() {
                rows.push(self.forward(tok, pos, &mut k_cache, &mut v_cache, &mut buf)?);
            }
        }
        Ok(rows)
    }

    pub fn forward(
        &self,
        token: usize,
        pos: usize,
        k_cache: &mut [Vec<f32>],
        v_cache: &mut [Vec<f32>],
        buf: &mut Vec<u8>,
    ) -> io::Result<Vec<f32>> {
        // Dispatch to DeepSeek-V2 MLA forward pass
        if self.arch_variant == ArchVariant::DeepSeek2 {
            return self.forward_deepseek2(token, pos, k_cache, buf);
        }
        // Dispatch to Gemma3 forward pass (SWA + global attention, QK-norm, post-norms)
        if self.arch_variant == ArchVariant::Gemma3 {
            return self.forward_gemma3(token, pos, k_cache, v_cache, buf);
        }
        // Dispatch to K2-Horizon forward pass (attention gate, grouped RMSNorm, MoVA)
        if self.arch_variant == ArchVariant::K2Horizon {
            return self.forward_k2_horizon(token, pos, k_cache, v_cache, buf);
        }
        let n_embd = self.n_embd;
        let head_dim = self.head_dim;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let kv_group = n_heads / n_kv_heads;
        let kv_stride = n_kv_heads * head_dim;
        // token embedding row
        let tok_t = self.t(TensorKind::TokenEmbedd, 0)?;
        let mut x = tok_t.rows(token, 1, buf)?;
        x.truncate(n_embd);
        // GPT-2: add learned absolute positional embeddings (no RoPE)
        if self.arch_variant == ArchVariant::GPT2 {
            let pos_t = self.t(TensorKind::PositionEmbedd, 0)?;
            let pos_emb = pos_t.rows(pos, 1, buf)?;
            for i in 0..n_embd {
                x[i] += pos_emb[i];
            }
        }
        if self.debug_checksums {
            let s: f32 = x.iter().map(|v| v.abs()).sum();
            eprintln!("embd x_sum={s:.6} first3={:?}", &x[..3]);
        }
        crate::trace::record(crate::trace::SLOT_INP_EMBD, -1, pos, &x);
        for layer in 0..self.n_layers {
            // ---- attention ----
            let norm_t = self.t(TensorKind::AttnNorm, layer)?;
            let w = norm_t.rows(0, 1, buf)?;
            // GPT-2 and Falcon use LayerNorm (mean-centering + variance) with bias, not RMSNorm
            let t = if self.arch_variant == ArchVariant::GPT2
                || self.arch_variant == ArchVariant::Falcon
            {
                let b = self.bias_vec(layer, TensorKind::AttnNormBias, n_embd, buf)?;
                Self::layernorm(&x, &w, &b, self.eps as f32)
            } else {
                rmsnorm(&x, &w, self.eps as f32)
            };
            crate::trace::record(crate::trace::SLOT_ATTN_NORM, layer as i32, pos, &t);
            let (mut q, mut k, mut v) = if self.arch_variant == ArchVariant::Phi3
                || self.arch_variant == ArchVariant::Falcon
                || self.arch_variant == ArchVariant::OpenELM
            {
                // Fused attn_qkv.weight: [n_embd + 2*n_kv_heads*head_dim, n_embd]
                // One matvec, then split the result into Q/K/V slices.
                let qkv_t = self.t(TensorKind::AttnQ, layer)?;
                let qkv = self.matvec(&qkv_t, &t, buf)?;
                let q_end = n_embd;
                let k_end = n_embd + n_kv_heads * head_dim;
                (
                    qkv[..q_end].to_vec(),
                    qkv[q_end..k_end].to_vec(),
                    qkv[k_end..].to_vec(),
                )
            } else if self.arch_variant == ArchVariant::GPT2 {
                // GPT-2 has fused QKV: model.h.{layer}.attn.c_attn.weight [3*n_embd, n_embd]
                // One matvec, then split into Q/K/V (MHA: n_kv_heads = n_heads)
                let wqkv_t = self.t(TensorKind::AttnQ, layer)?;
                let qkv = self.matvec(&wqkv_t, &t, buf)?;
                let q_end = n_embd;
                let k_end = n_embd + n_embd; // n_kv_heads == n_heads for GPT-2 MHA
                (
                    qkv[..q_end].to_vec(),
                    qkv[q_end..k_end].to_vec(),
                    qkv[k_end..].to_vec(),
                )
            } else {
                let wq = self.t(TensorKind::AttnQ, layer)?;
                let wk = self.t(TensorKind::AttnK, layer)?;
                let wv = self.t(TensorKind::AttnV, layer)?;
                (
                    self.matvec(&wq, &t, buf)?,
                    self.matvec(&wk, &t, buf)?,
                    self.matvec(&wv, &t, buf)?,
                )
            };
            // Add bias if present
            if self.has_bias {
                if self.arch_variant == ArchVariant::GPT2 {
                    // GPT-2 has fused attn_qkv.bias [3*n_embd]; split into Q/K/V
                    let qkv_bias =
                        self.bias_vec(layer, TensorKind::AttnQKVBias, 3 * n_embd, buf)?;
                    for i in 0..n_embd {
                        q[i] += qkv_bias[i];
                    }
                    for i in 0..n_embd {
                        k[i] += qkv_bias[n_embd + i];
                    }
                    for i in 0..n_embd {
                        v[i] += qkv_bias[2 * n_embd + i];
                    }
                } else {
                    let q_bias = self.bias_vec(layer, TensorKind::AttnQBias, n_embd, buf)?;
                    let k_bias =
                        self.bias_vec(layer, TensorKind::AttnKBias, n_kv_heads * head_dim, buf)?;
                    let v_bias =
                        self.bias_vec(layer, TensorKind::AttnVBias, n_kv_heads * head_dim, buf)?;
                    for i in 0..n_embd {
                        q[i] += q_bias[i];
                    }
                    for i in 0..n_kv_heads * head_dim {
                        k[i] += k_bias[i];
                    }
                    for i in 0..n_kv_heads * head_dim {
                        v[i] += v_bias[i];
                    }
                }
            }
            // Expand K/V from n_kv_heads to n_heads (GQA repeat). Query head `h`
            // reads KV head `h / kv_group`, matching llama.cpp's
            // `kv_head = h / n_rep` — see the identical block in the Gemma-3
            // pass for why the modulo is wrong.
            let k_full = if n_kv_heads == n_heads {
                k
            } else {
                let mut k2 = vec![0f32; n_embd];
                for h in 0..n_heads {
                    let kv_h = h / kv_group;
                    let src = &k[kv_h * head_dim..(kv_h + 1) * head_dim];
                    let dst = &mut k2[h * head_dim..(h + 1) * head_dim];
                    dst.copy_from_slice(src);
                }
                k2
            };
            let v_full = if n_kv_heads == n_heads {
                v
            } else {
                let mut v2 = vec![0f32; n_embd];
                for h in 0..n_heads {
                    let kv_h = h / kv_group;
                    let src = &v[kv_h * head_dim..(kv_h + 1) * head_dim];
                    let dst = &mut v2[h * head_dim..(h + 1) * head_dim];
                    dst.copy_from_slice(src);
                }
                v2
            };
            let k = k_full;
            let v = v_full;
            // optional QK-norm
            let (mut q_norm, mut k_norm) = if self.qk_norm {
                let qw = self.t(TensorKind::AttnQKNorm, layer)?;
                let kw_name = tensor_name(self.arch_variant, TensorKind::AttnQKNorm, layer)
                    .replace("attn_q_norm", "attn_k_norm");
                let kw = self.t_by_name(&kw_name)?;
                let qw = qw.rows(0, 1, buf)?;
                let kw = kw.rows(0, 1, buf)?;
                (
                    qk_norm_row(&q, &qw, head_dim, self.eps as f32),
                    qk_norm_row(&k, &kw, head_dim, self.eps as f32),
                )
            } else {
                (q, k)
            };
            // RoPE (skip for GPT-2: uses learned positional embeddings instead)
            if self.arch_variant != ArchVariant::GPT2 {
                for h in 0..n_heads {
                    let s = h * head_dim;
                    crate::opdump::record("rope_q", &q_norm[s..s + head_dim]);
                    rope(
                        &mut q_norm[s..s + head_dim],
                        pos,
                        self.rope_theta,
                        self.rope_type,
                    );
                    crate::opdump::record("rope_q_out", &q_norm[s..s + head_dim]);
                }
                for h in 0..n_heads {
                    let s = h * head_dim;
                    rope(
                        &mut k_norm[s..s + head_dim],
                        pos,
                        self.rope_theta,
                        self.rope_type,
                    );
                }
            }
            // write KV cache
            let k_src = &k_norm;
            let v_src = &v;
            for kv_h in 0..n_kv_heads {
                let ks = kv_h * kv_group * head_dim;
                for p in 0..head_dim {
                    k_cache[layer][pos * kv_stride + kv_h * head_dim + p] = k_src[ks + p];
                    v_cache[layer][pos * kv_stride + kv_h * head_dim + p] = v_src[ks + p];
                }
            }
            // attention over cached keys
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut att_out = vec![0f32; n_embd];
            for h in 0..n_heads {
                let kv_h = h / kv_group;
                let mut scores = vec![0f32; pos + 1];
                for p in 0..=pos {
                    let kr = &k_cache[layer]
                        [p * kv_stride + kv_h * head_dim..p * kv_stride + (kv_h + 1) * head_dim];
                    let qh = &q_norm[h * head_dim..(h + 1) * head_dim];
                    scores[p] = ggml_dot_f32(kr, qh) * scale;
                }
                crate::opdump::record("softmax_scores", &scores);
                let probs = softmax(&mut scores);
                for p in 0..=pos {
                    let vr = &v_cache[layer]
                        [p * kv_stride + kv_h * head_dim..p * kv_stride + (kv_h + 1) * head_dim];
                    let pr = probs[p];
                    for (o, vv) in att_out[h * head_dim..(h + 1) * head_dim].iter_mut().zip(vr) {
                        *o += pr * vv;
                    }
                }
            }
            let wo = self.t(TensorKind::AttnOutput, layer)?;
            let wo_out = self.matvec(&wo, &att_out, buf)?;
            // GPT-2 has attn_output.bias; add it to the residual
            if self.arch_variant == ArchVariant::GPT2 && self.has_bias {
                let wo_b = self.bias_vec(layer, TensorKind::AttnOutputBias, n_embd, buf)?;
                for i in 0..n_embd {
                    x[i] += wo_out[i] + wo_b[i];
                }
            } else {
                for i in 0..n_embd {
                    x[i] += wo_out[i];
                }
            }
            // llama.cpp names this tensor "ffn_inp" (src/models/qwen2.cpp,
            // `ffn_inp = ggml_add(ctx0, cur, inpSA)`): the residual stream as
            // the FFN sees it, i.e. the attention branch's contribution
            // already added.
            crate::trace::record(crate::trace::SLOT_FFN_INP, layer as i32, pos, &x);
            // ---- FFN ----
            if self.arch_variant == ArchVariant::Falcon {
                // Falcon: parallel attention+FFN — FFN uses the SAME normalized
                // input as attention (`t`), not a separate norm of the residual.
                // Falcon uses GELU activation (no gate tensor).
                let ff_out = self.dense_ffn_act(layer, &t, buf, true)?;
                for i in 0..n_embd {
                    x[i] += ff_out[i];
                }
            } else {
                let fnorm_t = self.t(TensorKind::FfnNorm, layer)?;
                let w = fnorm_t.rows(0, 1, buf)?;
                // GPT-2 uses LayerNorm with bias for ffn_norm (ln_2)
                let t_ffn = if self.arch_variant == ArchVariant::GPT2 {
                    let b = self.bias_vec(layer, TensorKind::FfnNormBias, n_embd, buf)?;
                    Self::layernorm(&x, &w, &b, self.eps as f32)
                } else {
                    rmsnorm(&x, &w, self.eps as f32)
                };
                crate::trace::record(crate::trace::SLOT_FFN_NORM, layer as i32, pos, &t_ffn);
                let has_router = self.gf.tensors.contains_key(&tensor_name(
                    self.arch_variant,
                    TensorKind::FfnGateInp,
                    layer,
                ));
                if self.n_experts > 0 && has_router {
                    // MoE
                    let router_t = self.t(TensorKind::FfnGateInp, layer)?;
                    let router_logits = self.matvec(&router_t, &t_ffn, buf)?;
                    let top = moe_route_weights(
                        &router_logits,
                        self.n_experts_used,
                        self.moe_norm_w(),
                        self.moe_scale(),
                        self.moe_gating(),
                    );
                    let mut moe_out = vec![0f32; n_embd];
                    for (e, wgt) in top {
                        let e_out = self.expert(layer, e, &t_ffn, buf)?;
                        for i in 0..n_embd {
                            moe_out[i] += wgt * e_out[i];
                        }
                    }
                    for i in 0..n_embd {
                        x[i] += moe_out[i];
                    }
                } else {
                    let ff_out = if self.arch_variant == ArchVariant::GPT2 {
                        self.dense_ffn_act(layer, &t_ffn, buf, true)?
                    } else {
                        self.dense_ffn(layer, &t_ffn, buf)?
                    };
                    crate::trace::record(crate::trace::SLOT_FFN_OUT, layer as i32, pos, &ff_out);
                    for i in 0..n_embd {
                        x[i] += ff_out[i];
                    }
                }
            }
            crate::trace::record(crate::trace::SLOT_L_OUT, layer as i32, pos, &x);
            if self.debug_checksums {
                let s: f32 = x.iter().map(|v| v.abs()).sum();
                eprintln!("L{layer} x_sum={s:.4}");
            }
        }
        // final norm + logits
        let onorm_t = self.t(TensorKind::OutputNorm, 0)?;
        let w = onorm_t.rows(0, 1, buf)?;
        // GPT-2 and Falcon use LayerNorm with bias for output_norm
        let x =
            if self.arch_variant == ArchVariant::GPT2 || self.arch_variant == ArchVariant::Falcon {
                let b = self.bias_vec(0, TensorKind::OutputNormBias, n_embd, buf)?;
                Self::layernorm(&x, &w, &b, self.eps as f32)
            } else {
                rmsnorm(&x, &w, self.eps as f32)
            };
        // llama.cpp names this tensor "result_norm" (src/models/qwen2.cpp,
        // `cb(cur, "result_norm", -1)`). It is the last whole-model node before
        // the output projection, so it is the only place the trace can see the
        // final segment: layer 35's residual -> final norm -> output matmul ->
        // logits. Without it the trace would end at the logits with the last two
        // ops unobserved, and an error injected there would look like an error
        // that arrived from layer 35.
        crate::trace::record(crate::trace::SLOT_RESULT_NORM, -1, pos, &x);
        // Weight tying: when `output.weight` is absent, llama.cpp binds the
        // final projection to the token embedding matrix for EVERY
        // architecture (see llama.cpp src/models/llama.cpp: `output =
        // create_tensor(..., TENSOR_NOT_REQUIRED)` followed by `if (output ==
        // NULL) output = create_tensor(TOKEN_EMBD, ...)`). Restricting this to
        // GPT-2/Falcon, as we used to, breaks any tied llama/qwen2 model.
        // `output_t` is that binding; it is documented in one place because it
        // used to be spelled out here and in three other forward passes.
        let out_t = self.output_t()?;
        self.matvec(&out_t, &x, buf)
    }
    /// `norm_w` in `build_moe_ffn`. Upstream this is a **per-architecture
    /// constant passed as a literal at the call site**, not a metadata flag, and
    /// getting it wrong is invisible on any model that routes to every expert.
    ///
    /// The table is `src/models/*.cpp` read mechanically: 23 architectures pass
    /// `true`, 8 pass `false`, and the remaining ~24 pass
    /// `hparams.expert_weights_norm`, whose default is `false`
    /// (llama-hparams.h:101). `llama` is in the first group
    /// (llama.cpp:209), which is why Tiny-MoE — the only `llama`-arch MoE model
    /// on disk, and one with `expert_used_count == expert_count` — could never
    /// tell the difference between the two.
    pub fn moe_norm_w(&self) -> bool {
        moe_norm_w_for(&self.arch, self.expert_weights_norm)
    }

    /// See [`moe_scale_for`].
    pub fn moe_scale(&self) -> f32 {
        moe_scale_for(&self.arch, self.expert_weights_scale)
    }

    /// See [`moe_gating_for`].
    pub fn moe_gating(&self) -> u32 {
        moe_gating_for(
            &self.arch,
            self.expert_gating_func,
            self.arch_variant,
            self.n_layers,
            self.vocab_size,
        )
    }

    /// Print per-layer data type for each major tensor kind.
    /// Useful for debugging mixed-quant or unexpected quantization layouts.
    pub fn report_dtypes(&self) {
        use crate::gguf::dtype_name;
        let key_tensors = &[
            ("attn_q", TensorKind::AttnQ),
            ("attn_k", TensorKind::AttnK),
            ("attn_v", TensorKind::AttnV),
            ("attn_qkv", TensorKind::AttnQKVBias), // fused QKV weight shares with bias kind — just a probe
            ("attn_output", TensorKind::AttnOutput),
            ("ffn_gate", TensorKind::FfnGate),
            ("ffn_up", TensorKind::FfnUp),
            ("ffn_down", TensorKind::FfnDown),
        ];
        // Probe first two layers and last layer for summary (deduplicated)
        let last = self.n_layers.saturating_sub(1);
        let mut probe_layers: Vec<usize> = vec![0];
        if self.n_layers > 1 {
            probe_layers.push(1);
            if last > 1 {
                probe_layers.push(last);
            }
        }
        eprintln!(
            "--- per-layer dtypes (probe layers {}) ---",
            probe_layers
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        for &layer in &probe_layers {
            eprintln!("  layer {layer}:");
            for &(name, kind) in key_tensors {
                let tname = tensor_name(self.arch_variant, kind, layer);
                if let Some(ti) = self.gf.tensors.get(&tname) {
                    eprintln!("    {name:15} -> {} ({})", dtype_name(ti.dtype), tname);
                }
            }
        }
        // Non-layer tensors
        let global_tensors = &[
            ("token_embd", TensorKind::TokenEmbedd),
            ("output", TensorKind::Output),
            ("output_norm", TensorKind::OutputNorm),
            ("attn_norm", TensorKind::AttnNorm),
            ("ffn_norm", TensorKind::FfnNorm),
        ];
        eprintln!("  global:");
        for &(name, kind) in global_tensors {
            let tname = tensor_name(self.arch_variant, kind, 0);
            if let Some(ti) = self.gf.tensors.get(&tname) {
                eprintln!("    {name:15} -> {} ({})", dtype_name(ti.dtype), tname);
            }
        }
        eprintln!("---");
    }

    /// Dump every tensor in the GGUF directory grouped by layer. This is a
    /// structural probe: it never loads tensor data and is useful for checking
    /// names, dimensions, dtypes, and mixed-quant layouts after conversion.
    pub fn test_tensors(&self) -> io::Result<()> {
        use crate::gguf::dtype_name;
        let mut global: Vec<(String, String, String)> = Vec::new();
        let mut by_layer: Vec<(usize, String, String, String)> = Vec::new();
        for (name, ti) in &self.gf.tensors {
            let dims = ti
                .dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            if let Some(rest) = name.strip_prefix("blk.")
                && let Some(dot) = rest.find('.')
                && let Ok(l) = rest[..dot].parse::<usize>()
            {
                by_layer.push((
                    l,
                    rest[dot + 1..].to_string(),
                    dtype_name(ti.dtype).to_string(),
                    dims,
                ));
                continue;
            }
            global.push((name.clone(), dtype_name(ti.dtype).to_string(), dims));
        }
        global.sort_by(|a, b| a.0.cmp(&b.0));
        by_layer.sort_by_key(|(l, n, _, _)| (*l, n.clone()));
        eprintln!(
            "--- tensor directory: {} tensors ---",
            self.gf.tensors.len()
        );
        eprintln!("  global ({}):", global.len());
        for (name, dt, dims) in &global {
            eprintln!("    {name:42} {dt:6} [{dims}]");
        }
        let mut cur: Option<usize> = None;
        for (l, name, dt, dims) in &by_layer {
            if cur != Some(*l) {
                eprintln!("  blk.{l}:");
                cur = Some(*l);
            }
            eprintln!("    {name:42} {dt:6} [{dims}]");
        }
        eprintln!("--- tensor listing complete ---");
        Ok(())
    }
    /// Generate `n_new` new tokens auto-regressively after the given prompt.
    #[allow(clippy::type_complexity)]
    pub fn generate(
        &self,
        tokens: &[usize],
        n_new: usize,
    ) -> io::Result<(Vec<usize>, f64, Vec<(usize, f32)>)> {
        let t0 = std::time::Instant::now();
        let mut seq = tokens.to_vec();
        let max_seq = if seq.is_empty() {
            n_new
        } else {
            tokens.len() + n_new
        };
        let mut buf: Vec<u8> = Vec::with_capacity(262_144);
        let mut logits = None;
        // MLA uses latent KV cache sized by (kv_lora_rank + rope_dim) per position, MQA (1 KV group)
        // Standard archs use MHA/GQA cache sized by n_kv_heads * head_dim
        if self.arch_variant == ArchVariant::DeepSeek2 {
            let mla_stride = self.n_lora_kv + self.rope_dim;
            let mut mla_k_cache = vec![vec![0f32; max_seq * mla_stride]; self.n_layers];
            for (pos, tok) in tokens.iter().enumerate() {
                logits = Some(self.forward(*tok, pos, &mut mla_k_cache, &mut [], &mut buf)?);
            }
            if logits.is_none() {
                // Empty prompt: seed with token 0 at position 0
                seq.push(0);
                logits = Some(self.forward(0, 0, &mut mla_k_cache, &mut [], &mut buf)?);
            }
            for _ in 0..n_new.saturating_sub(1) {
                let nxt = argmax(logits.as_ref().unwrap());
                seq.push(nxt);
                logits =
                    Some(self.forward(nxt, seq.len() - 1, &mut mla_k_cache, &mut [], &mut buf)?);
            }
            seq.push(argmax(logits.as_ref().unwrap()));
        } else {
            let kv_stride = self.n_kv_heads * self.head_dim;
            let mut k_cache = vec![vec![0f32; max_seq * kv_stride]; self.n_layers];
            let mut v_cache = vec![vec![0f32; max_seq * kv_stride]; self.n_layers];
            for (pos, tok) in tokens.iter().enumerate() {
                logits = Some(self.forward(*tok, pos, &mut k_cache, &mut v_cache, &mut buf)?);
            }
            if logits.is_none() {
                // Empty prompt: seed with token 0 at position 0
                seq.push(0);
                logits = Some(self.forward(0, 0, &mut k_cache, &mut v_cache, &mut buf)?);
            }
            for _ in 0..n_new.saturating_sub(1) {
                let nxt = argmax(logits.as_ref().unwrap());
                seq.push(nxt);
                logits =
                    Some(self.forward(nxt, seq.len() - 1, &mut k_cache, &mut v_cache, &mut buf)?);
            }
            seq.push(argmax(logits.as_ref().unwrap()));
        }
        let dt = t0.elapsed().as_secs_f64();
        let lg = logits.as_ref().unwrap();
        let mut idx: Vec<usize> = (0..lg.len()).collect();
        idx.sort_by(|a, b| lg[*b].partial_cmp(&lg[*a]).unwrap());
        let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, lg[i])).collect();
        Ok((seq, dt, top5))
    }

    fn deepseek2_kq_scale_value(
        rope_scaling_factor: f32,
        rope_yarn_log_mul: f32,
        rope_attn_factor: f32,
        key_length_mla: f32,
    ) -> f32 {
        let log_inv_scale = (1.0f32 / rope_scaling_factor).ln();
        let attn_factor_org = rope_attn_factor * (1.0f32 + 0.1f32 * log_inv_scale);
        let mscale = attn_factor_org * (1.0f32 + 0.1f32 * rope_yarn_log_mul * log_inv_scale);
        mscale * mscale / key_length_mla.sqrt()
    }

    fn deepseek2_kq_scale(&self) -> f32 {
        Self::deepseek2_kq_scale_value(
            self.rope_scaling_factor,
            self.rope_yarn_log_mul,
            self.rope_attn_factor,
            self.key_length_mla as f32,
        )
    }
}

// ---- math helpers ----
/// Per-head RMSNorm for QK-norm
/// Per-head RMSNorm with a **shared** weight vector: `w` is one head's worth of
/// weights (`attn_q_norm.weight` / `attn_k_norm.weight` are `{n_embd_head_k}`
/// long in every architecture that has them) and it is applied to each head of
/// `v` independently.
///
/// ggml normalises the whole `{n_embd_head_k * n_head, n_embd_head_k}` view in a
/// single `ggml_rms_norm`, which is the same thing because RMSNorm reduces along
/// the contiguous `ne[0]` axis only.
///
/// The weight is *not* tiled per head. Slicing `w` head by head — which is what
/// this did — both panics (`w[h*head_dim..]` runs off the end for h > 0) and, for
/// the architectures where the weight *is* long enough to slice, silently applies
/// the wrong weights to every head after the first.
fn qk_norm_row(v: &[f32], w: &[f32], head_dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; v.len()];
    for (h, chunk) in v.chunks(head_dim).enumerate() {
        let normed = rmsnorm(chunk, w, eps);
        let start = h * head_dim;
        out[start..start + chunk.len()].copy_from_slice(&normed);
    }
    out
}
fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    // ggml accumulates the sum of squares in a double over f32-rounded products
    // (ggml/src/ggml-cpu/ops.cpp:3828), not in f32. This is a different
    // reduction rule from the mat-mul's, on purpose -- see src/ggml_dot.rs.
    let inv = ggml_rms_norm_scale(x, eps);
    x.iter().zip(w).map(|(v, wi)| v * inv * wi).collect()
}
/// Rotary position embedding, matching ggml's `rotate_pairs()` +
/// `ggml_rope_cache_init()` (ggml/src/ggml-cpu/ops.cpp).
///
/// The frequency ladder is `freq_base^(-2i/n_dims)` seeded at `freq_scale`
/// (which is 1.0 unless RoPE scaling is active), so `theta_0 = 1.0` and
/// `theta_i = freq_base^(-2i/n_dims)` — note the ladder does *not* start at
/// `freq_base` itself, which is an easy and silent way to get this wrong.
///
/// Only the element pairing differs between the two conventions:
/// * `Norm`   — ggml calls `rotate_pairs(n, n_offset=1, ..., scale=1)`:
///   pair `(i, i+1)` for i = 0, 2, 4, ... (adjacent pairs).
/// * `NeoX`   — ggml calls `rotate_pairs(n, n_offset=n/2, ..., scale=2)`:
///   pair `(i, i + n/2)` for i = 0, 1, 2, ... (halves offset).
///
/// `x` is one head's worth of rotary dims. ggml evaluates the angle in f32;
/// we use f64 for the transcendental and narrow on the way out, which tracks
/// it to ~3e-7 — verified by `tests/rope_parity.rs` against ggml itself.
pub fn rope_apply(x: &mut [f32], pos: usize, theta: f64, kind: RopeType) {
    let n_dims = x.len();
    if n_dims < 2 {
        return;
    }
    let theta_scale = (theta as f32).powf(-2.0 / n_dims as f32);
    let mut theta = pos as f32;
    let half = n_dims / 2;
    // step is how far apart the two elements of a pair are;
    // span is how far the pair index advances between pairs.
    let (n_offset, span) = match kind {
        RopeType::Norm => (1usize, 2usize),
        RopeType::NeoX => (half, 1usize),
    };
    let mut ic = 0usize;
    while ic + n_offset < n_dims {
        let (s, c) = (theta.sin(), theta.cos());
        theta *= theta_scale;
        let (a, b) = (x[ic], x[ic + n_offset]);
        // `mul_add`, not `a * c - b * s`. rotate_pairs (ops.cpp:5947) is
        // compiled into a `-O3` libggml-cpu.so, which contracts both products
        // into the FMA, so the reference rounds the product NOT at all while a
        // plain `a * c - b * s` rounds twice. This is the last 9 bits of 144 on
        // tests/data/rope_golden.bin: without it the port is 93.75%
        // bit-identical and looks almost right, which is the most expensive
        // kind of wrong. Same lesson as `ggml_dot.rs`, one level up — there the
        // fma was in the intrinsics, here it is only in what the compiler did
        // to the reference, and nothing in the source says so.
        x[ic] = a.mul_add(c, -(b * s));
        x[ic + n_offset] = a.mul_add(s, b * c);
        ic += span;
    }
}
fn softmax(x: &mut [f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|v| (v - max).exp()).sum();
    x.iter().map(|v| (v - max).exp() / sum).collect()
}

/// `{arch}.expert_gating_func` (`src/llama-hparams.h:13`). `NONE` is not a
/// usable value — `build_moe_ffn` aborts on it — it is the "key absent" marker
/// that a few architectures resolve from other hparams before calling in.
pub const GATING_NONE: u32 = 0;
pub const GATING_SOFTMAX: u32 = 1;
pub const GATING_SIGMOID: u32 = 2;
/// Applied to the *router weights* instead of the logits: the top-k slice is
/// re-softmaxed rather than taken from a softmax over all experts.
pub const GATING_SOFTMAX_WEIGHT: u32 = 3;
pub const GATING_SQRT_SOFTPLUS: u32 = 4;

/// `ggml_clamp(weights_sum, 6.103515625e-5, INFINITY)` in `build_moe_ffn`
/// (llama-graph.cpp:2055): the smallest positive normal float16, used as a
/// divide-by-zero guard.
pub const MOE_WEIGHT_SUM_FLOOR: f32 = 1.0 / 16384.0;

/// `llm_graph_context::build_moe_ffn`'s weight block, transcribed
/// (llama-graph.cpp:2035-2075), so that the routing arithmetic lives in one
/// place the forward passes call and the unit tests can reach.
///
/// Returns `(expert index, weight)` pairs, highest probability first. The three
/// things that are easy to get wrong, all of which were wrong here before:
///
/// * **The softmax is over *all* experts**, and `weights` starts life as
///   `get_rows(probs, selected_experts)` — the selected experts' probabilities
///   from that full softmax, *not* a softmax over the selected subset. Only
///   `GATING_SOFTMAX_WEIGHT` re-softmaxes the subset, and that is a different
///   gating function, not a different spelling of the same one.
/// * **`norm_w` divides the selected weights by their own sum**, clamped at
///   `MOE_WEIGHT_SUM_FLOOR`. It is `true` for most architectures, but it is a
///   per-architecture *constant* in llama.cpp rather than a metadata flag — see
///   `Model::moe_norm_w`. When `n_expert_used == n_expert` the division is a
///   no-op up to one ulp, because the softmax already sums to 1, which is
///   exactly why a model that uses every expert cannot detect its absence.
/// * **`w_scale` is applied last, and only when it is neither 0 nor 1**.
///   `hparams.expert_weights_scale` defaults to `0.0f` (llama-hparams.h:99) and
///   0 means "unset", not "multiply by zero".
fn moe_route_weights(
    router_logits: &[f32],
    n_used: usize,
    norm_w: bool,
    w_scale: f32,
    gating: u32,
) -> Vec<(usize, f32)> {
    let probs = match gating {
        GATING_NONE | GATING_SOFTMAX => softmax(&mut router_logits.to_vec()),
        GATING_SIGMOID => router_logits
            .iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect(),
        GATING_SQRT_SOFTPLUS => router_logits
            .iter()
            .map(|v| {
                // ggml_softplus has no overflow branch: for x > 20 `exp(-x)` is
                // ~0 and the value is just x.
                let sp = if *v > 20.0 {
                    *v
                } else {
                    (1.0 + (-v).exp()).ln()
                };
                sp.max(0.0).sqrt()
            })
            .collect(),
        GATING_SOFTMAX_WEIGHT => router_logits.to_vec(),
        other => unreachable!("expert_gating_func {other} has no forward pass"),
    };
    // ggml_argsort_top_k: descending, ties towards the lower index. `sort_by` is
    // stable and softmax is monotonic, so sorting on `probs` gives the same
    // permutation as sorting the logits.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap());
    order.truncate(n_used.min(order.len()));
    let mut sel: Vec<f32> = order.iter().map(|&e| probs[e]).collect();
    if gating == GATING_SOFTMAX_WEIGHT {
        sel = softmax(&mut sel);
    }
    if norm_w {
        let mut sum: f32 = sel.iter().sum();
        if sum < MOE_WEIGHT_SUM_FLOOR {
            sum = MOE_WEIGHT_SUM_FLOOR;
        }
        for v in &mut sel {
            *v /= sum;
        }
    }
    if w_scale != 0.0 && w_scale != 1.0 {
        for v in &mut sel {
            *v *= w_scale;
        }
    }
    order.into_iter().zip(sel).collect()
}

/// `norm_w` in `build_moe_ffn`. Upstream this is a **per-architecture constant
/// passed as a literal at the call site**, not a metadata flag, and getting it
/// wrong is invisible on any model that routes to every expert.
///
/// The table is `src/models/*.cpp` read mechanically: 23 architectures pass
/// `true`, 8 pass `false`, and the remaining ~24 pass
/// `hparams.expert_weights_norm`, whose default is `false`
/// (llama-hparams.h:101). `llama` is in the first group (llama.cpp:209), which
/// is why Tiny-MoE — the only `llama`-arch MoE model on disk, and one with
/// `expert_used_count == expert_count` — could never tell the difference
/// between the two.
pub fn moe_norm_w_for(arch: &str, expert_weights_norm: bool) -> bool {
    const ALWAYS: &[&str] = &[
        "arctic",
        "dbrx",
        "ernie4_5-moe",
        "gemma4",
        "granite",
        "granitehybrid",
        "granitemoe",
        "grok",
        "grovemoe",
        "hunyuan-moe",
        "kimi-linear",
        "lfm2",
        "lfm2moe",
        "llama",
        "llama-embed",
        "mellum",
        "mimo2",
        "minicpm",
        "minimax-m2",
        "mistral3",
        "phi3",
        "phimoe",
        "qwen35moe",
        "qwen3moe",
        "qwen3next",
        "qwen3vlmoe",
        "rnd1",
        "smallthinker",
    ];
    const NEVER: &[&str] = &[
        // All of these pass `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` or
        // `SIGMOID` with a renormalisation upstream, and would be broken by an
        // extra one, which is the whole point of the split.
        "bert",
        "deepseek",
        "gpt-oss",
        "jamba",
        "jina-bert-v2",
        "jina-bert-v3",
        "llada-moe",
        "llama4",
        "nomic-bert",
        "nomic-bert-moe",
        "olmoe",
        "qwen2moe",
    ];
    if ALWAYS.contains(&arch) {
        true
    } else if NEVER.contains(&arch) {
        false
    } else {
        // The only architectures that reach this arm are the twenty that both
        // forward `hparams.expert_weights_norm` to `build_moe_ffn` *and* read
        // the key, so deferring is exactly upstream's behaviour. The two sets
        // coincide in llama.cpp 3653e6d; `tests/moe_keys.rs` asserts it for all
        // sixty architectures that route, so this arm cannot become a place
        // where metadata is consulted for an architecture that never looks.
        expert_weights_norm
    }
}

/// Architectures whose `load_hparams` calls
/// `ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale, ...)`.
///
/// This is a **different question** from `moe_norm_w_for`'s: that table records
/// what each architecture *passes* for `norm_w`, this one records which
/// architectures ever *read* the key at all. Roughly forty architectures pass
/// `hparams.expert_weights_scale` to `build_moe_ffn`; only the twenty-two below
/// can have a non-default value in it, and for the rest the field keeps its
/// `0.0f` initialiser (llama-hparams.h:100) whatever the GGUF says — and
/// `build_moe_ffn` reads `0.0` as "no scale", not "scale by zero".
///
/// The consequence is that a file carrying the key is a *different model* to
/// llama.cpp and to us unless we know which architectures honour it. That is
/// not hypothetical: `llama-model-saver.cpp:226` writes the key unconditionally,
/// so every GGUF llama.cpp itself saves carries it.
///
/// Transcribed from `src/models/*.cpp` by `scripts/gen_moe_key_authority.py`,
/// which resolves the class hierarchy: a subclass without its own
/// `load_hparams` inherits its parent's key reads — `mistral4` is a `deepseek2`
/// with three lines of `build_arch_graph`, so it reads the key because
/// deepseek2 does. `tests/moe_keys.rs` pins this list against the generated
/// oracle so it cannot rot silently.
pub const MOE_EXPERT_WEIGHTS_SCALE_READERS: &[&str] = &[
    "afmoe",
    "bailingmoe",
    "bailingmoe2",
    "cohere2moe",
    "deepseek",
    "deepseek2",
    "deepseek2-ocr",
    "deepseek32",
    "deepseek4",
    "dflash",
    "dots1",
    "exaone-moe",
    "glm-dsa",
    "glm4moe",
    "hy_v3",
    "kimi-linear",
    "laguna",
    "minimax-m3",
    "mistral4",
    "nemotron_h",
    "nemotron_h_moe",
    "step35",
];

/// `w_scale` in `build_moe_ffn`: the routed weights' final multiplier, resolved
/// the way llama.cpp resolves it.
///
/// An architecture that does not read the key yields `0.0`, which
/// `moe_route_weights` treats as "unset" — the same "`0.0` means unset"
/// reading `moe_gating_for` needs for an absent `expert_gating_func`.
pub fn moe_scale_for(arch: &str, expert_weights_scale: f64) -> f32 {
    if MOE_EXPERT_WEIGHTS_SCALE_READERS.contains(&arch) {
        expert_weights_scale as f32
    } else {
        0.0
    }
}

/// `gating_op` in `build_moe_ffn`, resolved the way llama.cpp resolves it.
///
/// `build_moe_ffn` aborts on `NONE`, so `NONE` never reaches it: each
/// architecture either hard-codes its gating function (as `llama` does, at
/// llama.cpp:211) or infers one from other hparams first. `deepseek2` is the
/// only implemented architecture that takes the inference route
/// (deepseek2.cpp:23-31): absent the key it picks `SIGMOID` for the GLM-4.7 Lite
/// shape and `SOFTMAX` for everything else.
pub fn moe_gating_for(
    arch: &str,
    gating: u32,
    variant: ArchVariant,
    n_layers: usize,
    n_vocab: usize,
) -> u32 {
    if gating != GATING_NONE {
        return gating;
    }
    if variant == ArchVariant::DeepSeek2 || arch == "deepseek2" {
        return if (n_layers == 47 || n_layers == 48) && n_vocab == 154_880 {
            GATING_SIGMOID
        } else {
            GATING_SOFTMAX
        };
    }
    // Every other implemented architecture that can route passes its gating
    // function as a literal; `llama` is the representative case.
    GATING_SOFTMAX
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}
/// Grouped RMSNorm: split vector into `groups` equal chunks and apply independent RMSNorm per chunk.
fn grouped_rmsnorm(x: &[f32], w: &[f32], groups: usize, eps: f32) -> Vec<f32> {
    let group_size = x.len() / groups;
    let mut out = vec![0f32; x.len()];
    for g in 0..groups {
        let start = g * group_size;
        let normed = rmsnorm(
            &x[start..start + group_size],
            &w[start..start + group_size],
            eps,
        );
        out[start..start + group_size].copy_from_slice(&normed);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MoE routing -------------------------------------------------------
    //
    // These are the numbers `build_moe_ffn` produces, read off the graph rather
    // than off a run, because the only `llama`-arch MoE model on disk
    // (Tiny-MoE) routes to *every* expert and therefore cannot distinguish
    // `norm_w` from its absence. `scripts/arbitrate_logits.py::moe_ffn` is the
    // float64 transcription of the same block and the two are checked against
    // each other in `moe_routing_matches_the_float64_reference`.

    fn wsum(sel: &[(usize, f32)]) -> f32 {
        sel.iter().map(|(_, w)| *w).sum()
    }

    #[test]
    fn moe_norm_w_follows_the_per_architecture_constant_not_the_metadata() {
        // `llama` passes a literal `true` (llama.cpp:209) whatever the GGUF says,
        // and `qwen2moe` a literal `false` (qwen2moe.cpp:139) whatever the GGUF
        // says. Reading `{arch}.expert_weights_norm` for either is wrong, and
        // the two errors are indistinguishable in output magnitude — one inflates
        // the routed FFN, the other deflates it.
        assert!(moe_norm_w_for("llama", false));
        assert!(moe_norm_w_for("llama", true));
        assert!(!moe_norm_w_for("qwen2moe", true));
        assert!(!moe_norm_w_for("olmoe", true));
        assert!(!moe_norm_w_for("llama4", false));
        // Everything else defers to the metadata key, which defaults to false.
        for arch in [
            "deepseek2",
            "glm4-moe",
            "exaone-moe",
            "step35",
            "bailingmoe",
        ] {
            assert!(!moe_norm_w_for(arch, false), "{arch}");
            assert!(moe_norm_w_for(arch, true), "{arch}");
        }
        // `phi3` is the arch string phimoe files carry, and it hardcodes true.
        assert!(moe_norm_w_for("phi3", false));
    }

    #[test]
    fn moe_gating_prefers_the_key_then_infers_deepseek2() {
        // An explicit key always wins, whatever the shape says.
        assert_eq!(
            moe_gating_for(
                "deepseek2",
                GATING_SIGMOID,
                ArchVariant::DeepSeek2,
                61,
                129280
            ),
            GATING_SIGMOID
        );
        // Absent the key: the GLM-4.7 Lite shape is the only SIGMOID inference.
        assert_eq!(
            moe_gating_for("deepseek2", GATING_NONE, ArchVariant::DeepSeek2, 47, 154880),
            GATING_SIGMOID
        );
        assert_eq!(
            moe_gating_for("deepseek2", GATING_NONE, ArchVariant::DeepSeek2, 48, 154880),
            GATING_SIGMOID
        );
        // ... and it needs *both* halves of that shape, not either.
        assert_eq!(
            moe_gating_for("deepseek2", GATING_NONE, ArchVariant::DeepSeek2, 47, 129280),
            GATING_SOFTMAX
        );
        assert_eq!(
            moe_gating_for("deepseek2", GATING_NONE, ArchVariant::DeepSeek2, 61, 154880),
            GATING_SOFTMAX
        );
        assert_eq!(
            moe_gating_for("deepseek2", GATING_NONE, ArchVariant::DeepSeek2, 28, 129280),
            GATING_SOFTMAX
        );
        // llama passes the gating function as a literal, so an absent key is SOFTMAX
        // and never NONE — build_moe_ffn aborts on NONE.
        assert_eq!(
            moe_gating_for("llama", GATING_NONE, ArchVariant::Llama, 12, 4096),
            GATING_SOFTMAX
        );
    }

    #[test]
    fn moe_softmax_is_over_all_experts_not_the_selected_subset() {
        // One dominant expert and seven identical runners-up: a softmax over the
        // selected pair would give the leader 0.731, a softmax over all eight
        // gives it 0.269. The engine used to do the former, which inflates the
        // routed FFN by up to ~3x whenever the router is not near-uniform.
        let logits = [3.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let sel = moe_route_weights(&logits, 2, false, 0.0, GATING_SOFTMAX);
        assert_eq!(
            sel.iter().map(|(e, _)| *e).collect::<Vec<_>>(),
            vec![0, 1],
            "leader first, ties towards the lower index"
        );
        let expect_leader = 1.0 / (1.0 + 7.0 * (-1.0f32).exp());
        assert!(
            (sel[0].1 - expect_leader).abs() < 1e-6,
            "{} vs {expect_leader}",
            sel[0].1
        );
        assert!(
            sel[0].1 < 0.3,
            "a softmax over the selected subset would give ~0.73; got {}",
            sel[0].1
        );
        // and the pair really does not sum to 1, which is what norm_w is for
        assert!(wsum(&sel) < 0.45, "{}", wsum(&sel));
    }

    #[test]
    fn moe_norm_w_renormalises_the_selected_weights() {
        let logits = [4.0, 3.0, 2.0, -30.0, -30.0, -30.0, -30.0, -30.0];
        let unnorm = moe_route_weights(&logits, 2, false, 0.0, GATING_SOFTMAX);
        let normed = moe_route_weights(&logits, 2, true, 0.0, GATING_SOFTMAX);
        assert!(
            wsum(&unnorm) < 0.95,
            "the selected probabilities do not sum to 1 here, or this test proves nothing"
        );
        assert!((wsum(&normed) - 1.0).abs() < 1e-6, "{}", wsum(&normed));
        // renormalisation must be a pure rescale, not a re-selection
        let ratio = normed[0].1 / unnorm[0].1;
        assert!((ratio - normed[1].1 / unnorm[1].1).abs() < 1e-6);
    }

    #[test]
    fn moe_norm_w_is_a_no_op_when_every_expert_is_used() {
        // This is the whole reason the omission survived: Tiny-MoE has
        // expert_count == expert_used_count == 2, so the full softmax already
        // sums to 1 and the division cannot change anything.
        let logits = [1.0, -0.5, 2.0, 0.25];
        let a = moe_route_weights(&logits, 4, false, 0.0, GATING_SOFTMAX);
        let b = moe_route_weights(&logits, 4, true, 0.0, GATING_SOFTMAX);
        for ((ea, wa), (eb, wb)) in a.iter().zip(b.iter()) {
            assert_eq!(ea, eb);
            assert!(
                (wa - wb).abs() <= 2.0 * f32::EPSILON * wa.abs(),
                "{wa} {wb}"
            );
        }
    }

    #[test]
    fn moe_weight_scale_is_applied_last_and_zero_means_unset() {
        // hparams.expert_weights_scale defaults to 0.0f (llama-hparams.h:99) and
        // build_moe_ffn skips both 0 and 1. Reading 0 as "multiply by zero" would
        // silently delete the FFN of every model that omits the key.
        let logits = [2.0, 1.0, 0.5];
        let base = moe_route_weights(&logits, 3, true, 0.0, GATING_SOFTMAX);
        for unset in [0.0f32, 1.0] {
            let got = moe_route_weights(&logits, 3, true, unset, GATING_SOFTMAX);
            for ((_, a), (_, b)) in base.iter().zip(got.iter()) {
                assert_eq!(a, b);
            }
        }
        let scaled = moe_route_weights(&logits, 3, true, 0.5, GATING_SOFTMAX);
        assert!((wsum(&scaled) - 0.5).abs() < 1e-6);
        // and it lands *after* the renormalisation, so the sum is scale, not 1
        let combined = moe_route_weights(&logits, 2, true, 0.25, GATING_SOFTMAX);
        assert!((wsum(&combined) - 0.25).abs() < 1e-6, "{}", wsum(&combined));
    }

    #[test]
    fn moe_weight_scale_follows_which_architectures_read_the_key() {
        // `llama` passes `hparams.expert_weights_scale` to build_moe_ffn
        // (llama.cpp:211) but never reads the key, so the field is always 0.0
        // ("unset") for this architecture — a GGUF carrying
        // `llama.expert_weights_scale = 2.5` is scaled by llama.cpp's *readers*
        // and not by llama.cpp itself. Measured: llama.cpp's logits on
        // `synthetic-llama-moe-wscale.gguf` are byte-identical to its logits on
        // `synthetic-llama-moe.gguf`, while the engine's differ by rel L2 0.68.
        assert_eq!(moe_scale_for("llama", 2.5), 0.0);
        // Architectures that do read it get the value.
        assert_eq!(moe_scale_for("deepseek2", 2.5), 2.5);
        assert_eq!(moe_scale_for("cohere2moe", 0.0), 0.0);
        assert_eq!(moe_scale_for("olmoe", 2.5), 0.0);
        assert_eq!(moe_scale_for("llama4", 2.5), 0.0);
        assert_eq!(moe_scale_for("qwen2moe", 2.5), 0.0);
        // A key that is present and honoured multiplies the routed weights; one
        // that is ignored leaves the sum at 1 after norm_w. The two differ by
        // exactly the key's value, which is what makes the fixture's gate sharp.
        let logits = [2.0f32, 1.0, 0.5];
        let ignored = moe_route_weights(
            &logits,
            3,
            true,
            moe_scale_for("llama", 2.5),
            GATING_SOFTMAX,
        );
        let honoured = moe_route_weights(
            &logits,
            3,
            true,
            moe_scale_for("deepseek2", 2.5),
            GATING_SOFTMAX,
        );
        assert!((wsum(&ignored) - 1.0).abs() < 1e-6, "{}", wsum(&ignored));
        assert!((wsum(&honoured) - 2.5).abs() < 1e-6, "{}", wsum(&honoured));
        assert_eq!(moe_scale_for("llama", 0.0), 0.0);
    }

    #[test]
    fn moe_sum_clamp_is_the_smallest_positive_float16() {
        // ggml clamps the weight sum at 6.103515625e-5 before dividing. With
        // `n_used == 1` and one dominant expert the sum is ~1, so this only bites
        // when every probability underflows; the test pins the constant and the
        // branch, since the alternative is a division by zero.
        assert_eq!(MOE_WEIGHT_SUM_FLOOR, 1.0 / 16384.0);
        let logits = [-30.0f32, -40.0, -50.0];
        let sel = moe_route_weights(&logits, 2, true, 0.0, GATING_SOFTMAX);
        assert!(sel.iter().all(|(_, w)| w.is_finite() && *w >= 0.0));
        assert!(wsum(&sel) > 0.0);
    }

    #[test]
    fn moe_sigmoid_gating_is_not_a_softmax() {
        // GATING_SIGMOID (llama.cpp: LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID) never
        // sums to 1, which is what makes `norm_w` matter for it.
        let logits = [2.0, 1.0, 0.0, -1.0];
        let sel = moe_route_weights(&logits, 2, false, 0.0, GATING_SIGMOID);
        let expect = 1.0 / (1.0 + (-2.0f32).exp());
        assert!((sel[0].1 - expect).abs() < 1e-6);
        let sel = moe_route_weights(&logits, 2, true, 0.0, GATING_SIGMOID);
        assert!((wsum(&sel) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn moe_softmax_weight_renormalises_the_slice_instead_of_the_whole() {
        // SOFTMAX_WEIGHT is the one gating function that re-softmaxes the
        // selected slice; the result depends on n_used, which is how you tell it
        // apart from a plain softmax plus norm_w even though both sum to 1.
        let logits = [2.0, 1.0, 0.0, -1.0];
        let k2 = moe_route_weights(&logits, 2, false, 0.0, GATING_SOFTMAX_WEIGHT);
        let k3 = moe_route_weights(&logits, 3, false, 0.0, GATING_SOFTMAX_WEIGHT);
        let ratio = k2[0].1 / k3[0].1;
        // adding a third expert to the slice changes the first one's weight only
        // through the normaliser, so k2's is the larger of the two
        assert!(ratio > 1.0, "{ratio}");
        assert!((wsum(&k3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn moe_selection_breaks_ties_towards_the_lower_index() {
        // ggml_argsort_top_k is a stable descending sort; Rust's sort_by is
        // stable too, but only because it is — an unstable sort would reorder
        // exact ties and change the expert order for no visible reason.
        let logits = [1.0, 1.0, 1.0, 0.0];
        let sel = moe_route_weights(&logits, 2, false, 0.0, GATING_SOFTMAX);
        assert_eq!(sel.iter().map(|(e, _)| *e).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn moe_routing_matches_the_float64_reference() {
        // The values below are what `scripts/arbitrate_logits.py::moe_ffn`
        // computes in float64 for the same logits, which is a transcription of
        // the same llama.cpp block. If the two implementations ever disagree
        // about the routing, one of them is wrong and this is where it shows.
        type Case = (&'static [f32], usize, bool, f32, &'static [f32]);
        let cases: [Case; 4] = [
            (
                &[2.0, 1.0, 0.0, -1.0],
                2,
                true,
                0.0,
                &[0.7310586, 0.2689414],
            ),
            (
                &[2.0, 1.0, 0.0, -1.0],
                2,
                false,
                0.0,
                &[0.6439143, 0.2368828],
            ),
            (
                &[0.5, 0.4, 0.3, 0.2, 0.1],
                3,
                true,
                2.0,
                &[0.7343308, 0.66445, 0.6012192],
            ),
            (&[3.0, -1.0, -2.0, -3.0], 1, true, 0.0, &[1.0]),
        ];
        for (logits, n_used, norm_w, w_scale, expect) in cases {
            let sel = moe_route_weights(logits, n_used, norm_w, w_scale, GATING_SOFTMAX);
            assert_eq!(sel.len(), expect.len());
            for ((e, got), want) in sel.iter().zip(expect.iter()) {
                assert!((got - want).abs() < 1e-5, "expert {e}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn test_gpt2_tensor_names() {
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::TokenEmbedd, 0),
            "token_embd.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::PositionEmbedd, 0),
            "position_embd.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::AttnQ, 0),
            "blk.0.attn_qkv.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::AttnK, 0),
            "blk.0.attn_qkv.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::AttnV, 0),
            "blk.0.attn_qkv.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::AttnQKVBias, 0),
            "blk.0.attn_qkv.bias"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::AttnOutput, 0),
            "blk.0.attn_output.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::AttnOutputBias, 0),
            "blk.0.attn_output.bias"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::FfnUp, 0),
            "blk.0.ffn_up.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::FfnUpBias, 0),
            "blk.0.ffn_up.bias"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::FfnDown, 0),
            "blk.0.ffn_down.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::FfnDownBias, 0),
            "blk.0.ffn_down.bias"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::OutputNorm, 0),
            "output_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::GPT2, TensorKind::OutputNormBias, 0),
            "output_norm.bias"
        );
    }

    #[test]
    fn test_gelu() {
        assert!((Model::gelu(0.0) - 0.0).abs() < 1e-6);
        assert!((Model::gelu(1.0) - 0.841_192).abs() < 1e-6);
        assert!((Model::gelu(-1.0) + 0.158_808).abs() < 1e-6);
        assert!((Model::gelu(2.0) - 1.954_598).abs() < 1e-6);
    }

    #[test]
    fn test_layernorm() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        let b = [0.0, 0.0, 0.0, 0.0];
        let eps = 1e-5;
        let y = Model::layernorm(&x, &w, &b, eps);
        assert!((y[0] + 1.3416).abs() < 1e-4);
        assert!((y[1] + 0.4472).abs() < 1e-4);
        assert!((y[2] - 0.4472).abs() < 1e-4);
        assert!((y[3] - 1.3416).abs() < 1e-4);
    }

    #[test]
    fn test_arch_from_string_gpt2() {
        assert_eq!(arch_from_string("gpt2"), ArchVariant::GPT2);
    }

    #[test]
    fn test_arch_from_string_falcon() {
        assert_eq!(arch_from_string("falcon"), ArchVariant::Falcon);
        assert_eq!(arch_from_string("falcon-h1"), ArchVariant::Falcon);
    }

    #[test]
    fn test_arch_from_string_deepseek2() {
        assert_eq!(arch_from_string("deepseek2"), ArchVariant::DeepSeek2);
    }

    #[test]
    fn test_deepseek2_yarn_kq_scale() {
        let base = Model::deepseek2_kq_scale_value(1.0, 0.0, 1.0, 64.0);
        assert!((base - 0.125).abs() < 1e-7);

        let yarn = Model::deepseek2_kq_scale_value(1.25, 0.05, 1.1, 57.0);
        assert!((yarn - 0.15285398).abs() < 1e-6);
    }

    #[test]
    fn test_deepseek2_tensor_names() {
        let layer = 7;
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::TokenEmbedd, layer),
            "token_embd.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::Output, layer),
            "output.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::OutputNorm, layer),
            "output_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::AttnNorm, layer),
            "blk.7.attn_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::AttnQANorm, layer),
            "blk.7.attn_q_a_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::AttnQA, layer),
            "blk.7.attn_q_a.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::AttnQB, layer),
            "blk.7.attn_q_b.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::AttnKVAMQA, layer),
            "blk.7.attn_kv_a_mqa.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::AttnKVANorm, layer),
            "blk.7.attn_kv_a_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::AttnKB, layer),
            "blk.7.attn_k_b.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::AttnVB, layer),
            "blk.7.attn_v_b.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::AttnOutput, layer),
            "blk.7.attn_output.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::FfnNorm, layer),
            "blk.7.ffn_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::FfnGateInp, layer),
            "blk.7.ffn_gate_inp.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::FfnUpExps, layer),
            "blk.7.ffn_up_exps.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::FfnGateExps, layer),
            "blk.7.ffn_gate_exps.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::FfnDownExps, layer),
            "blk.7.ffn_down_exps.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::FfnGateShexp, layer),
            "blk.7.ffn_gate_shexp.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::FfnUpShexp, layer),
            "blk.7.ffn_up_shexp.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::FfnDownShexp, layer),
            "blk.7.ffn_down_shexp.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::FfnExpProbsB, layer),
            "blk.7.exp_probs_b.bias"
        );
        assert_eq!(
            tensor_name(ArchVariant::DeepSeek2, TensorKind::AttnQ, layer),
            "blk.7.attn_q.weight"
        );
    }

    #[test]
    fn test_gemma3_tensor_names() {
        let l = 3;
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::TokenEmbedd, l),
            "token_embd.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::Output, l),
            "output.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::OutputNorm, l),
            "output_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::AttnNorm, l),
            "blk.3.attn_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::AttnQ, l),
            "blk.3.attn_q.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::AttnK, l),
            "blk.3.attn_k.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::AttnV, l),
            "blk.3.attn_v.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::AttnQKNorm, l),
            "blk.3.attn_q_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::AttnOutput, l),
            "blk.3.attn_output.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::AttnPostNorm, l),
            "blk.3.post_attention_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::FfnNorm, l),
            "blk.3.ffn_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::FfnUp, l),
            "blk.3.ffn_up.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::FfnGate, l),
            "blk.3.ffn_gate.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::FfnDown, l),
            "blk.3.ffn_down.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::FfnPostNorm, l),
            "blk.3.post_ffw_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::Dense2Out, l),
            "dense_2.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Gemma3, TensorKind::Dense3Out, l),
            "dense_3.weight"
        );
    }

    #[test]
    fn test_arch_from_string_gemma3() {
        assert_eq!(arch_from_string("gemma3"), ArchVariant::Gemma3);
        assert_eq!(arch_from_string("gemma3n"), ArchVariant::Gemma3);
    }

    #[test]
    fn test_gemma3_swa_classification() {
        // Period 6 with dense_first=false: layers 0-4 SWA, layer 5 global (repeating).
        // This is the shape a converted Gemma-3 GGUF gets for free, because
        // convert_hf_to_gguf.py never writes attention.sliding_window_pattern.
        let per_layer: Vec<bool> = (0..12).map(|l| gemma3_is_swa(l, 4096, 6)).collect();
        assert_eq!(
            per_layer,
            vec![
                true, true, true, true, true, false, true, true, true, true, true, false
            ]
        );
        // n_swa == 0 means there is no sliding window at all — every layer global.
        for layer in 0..6 {
            assert!(
                !gemma3_is_swa(layer, 0, 6),
                "layer {layer} should not be SWA when n_swa=0"
            );
        }
        // A pattern of 1 makes `il % 1 < 0` false, so nothing is SWA either.
        for layer in 0..6 {
            assert!(
                !gemma3_is_swa(layer, 4096, 1),
                "layer {layer} should not be SWA with pattern=1"
            );
        }
        // A pattern of 0 is NOT "unset": llama-hparams.cpp:8 makes every layer SWA.
        for layer in 0..6 {
            assert!(
                gemma3_is_swa(layer, 4096, 0),
                "layer {layer} should be SWA with pattern=0 (llama.cpp: il % 0 < -1 is \
                 short-circuited to true by the n_pattern == 0 arm)"
            );
        }
    }

    #[test]
    fn test_gemma3_softcapping() {
        // Hard to test the actual method without a model, so test the formula
        let c = 30.0f32;
        // value within range should be mostly preserved
        let v = 5.0f32;
        let capped = ((v / c).tanh()) * c;
        assert!((capped - 4.96).abs() < 0.05, "softcapped={capped}");
        // large value should be clamped to ~c
        let v_big = 100.0f32;
        let capped_big = ((v_big / c).tanh()) * c;
        assert!((capped_big - c).abs() < 0.1, "softcapped_big={capped_big}");
        // zero capping -> no-op
        let c0 = 0.0f32;
        let v0 = 10.0f32;
        let capped0 = if c0 > 0.0 {
            ((v0 / c0).tanh()) * c0
        } else {
            v0
        };
        assert!((capped0 - v0).abs() < 1e-6);
    }

    #[test]
    fn test_gemma3_gelu_ffn_activation() {
        // Gated GELU: gelu(gate) * up — test the activation pattern
        // For gate=1.0, up=2.0: gelu(1.0) ≈ 0.8412, result ≈ 1.6824
        let gate = 1.0f32;
        let up = 2.0f32;
        let result = Model::gelu(gate) * up;
        assert!((result - 1.6824).abs() < 0.001, "gated_gelu={result}");
        // For gate=-0.5, up=1.5: gelu(-0.5) ≈ -0.154, result ≈ -0.231
        let gate = -0.5f32;
        let up = 1.5f32;
        let result = Model::gelu(gate) * up;
        assert!((result + 0.231).abs() < 0.01, "gated_gelu_neg={result}");
    }

    #[test]
    fn test_gemma3_attention_scale() {
        // Ordinary checkpoints: 1/sqrt(n_embd_head_k). Gemma-3-1B is
        // n_embd=1152, 4 heads, key_length=256 -> 1/16 = 0.0625.
        let s1b = gemma3_attn_scale(26, 1152, 4, 256);
        assert!((s1b - 0.0625).abs() < 1e-6, "1B scale={s1b}");
        // 27B is the one branch that divides by n_embd/n_head instead, and the
        // two differ there: n_embd=5376 over 32 heads is 168 while key_length is
        // 128, so 1/sqrt(168) != 1/sqrt(128).
        let s27b = gemma3_attn_scale(62, 5376, 32, 128);
        assert!(
            (s27b - (1.0f32 / 168.0f32.sqrt())).abs() < 1e-6,
            "27B scale={s27b}"
        );
        assert!(
            (s27b - (1.0f32 / 128.0f32.sqrt())).abs() > 0.01,
            "the 27B branch must differ from 1/sqrt(key_length), else this test \
             cannot tell a correct implementation from the generic one"
        );
        // A 62-layer model with a different geometry still takes the 27B branch —
        // the discriminator is the layer count alone, which is all upstream has.
        assert!((gemma3_attn_scale(62, 1152, 4, 256) - (1.0f32 / 288.0f32.sqrt())).abs() < 1e-6);
    }

    #[test]
    fn test_gemma3_rope_base_selection() {
        // SWA layer uses rope_freq_base_swa if set
        let rope_theta = 10000.0f64;
        let rope_swa = 10000.0f64;
        let is_swa = true;
        let base = if is_swa && rope_swa != 0.0 {
            rope_swa
        } else {
            rope_theta
        };
        assert!((base - 10000.0).abs() < 1e-10);
        // When rope_swa is 0, global base is used even for SWA
        let rope_swa_zero = 0.0f64;
        let base2 = if is_swa && rope_swa_zero != 0.0 {
            rope_swa_zero
        } else {
            rope_theta
        };
        assert!((base2 - 10000.0).abs() < 1e-10);
        // Global layer always uses rope_theta
        let is_global = false;
        let base3 = if is_global && rope_swa != 0.0 {
            rope_swa
        } else {
            rope_theta
        };
        assert!((base3 - 10000.0).abs() < 1e-10);
    }

    #[test]
    fn test_falcon_tensor_names() {
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::TokenEmbedd, 0),
            "token_embd.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::Output, 0),
            "output.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::OutputNorm, 0),
            "output_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::OutputNormBias, 0),
            "output_norm.bias"
        );
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::AttnNorm, 0),
            "blk.0.attn_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::AttnNormBias, 0),
            "blk.0.attn_norm.bias"
        );
        // Fused QKV
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::AttnQ, 0),
            "blk.0.attn_qkv.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::AttnK, 0),
            "blk.0.attn_qkv.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::AttnV, 0),
            "blk.0.attn_qkv.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::AttnOutput, 0),
            "blk.0.attn_output.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::FfnUp, 0),
            "blk.0.ffn_up.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::FfnDown, 0),
            "blk.0.ffn_down.weight"
        );
        // Falcon has no separate ffn_norm — falls through to llama default
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::FfnNorm, 0),
            "blk.0.ffn_norm.weight"
        );
        // Falcon has no FfnGate — falls through to llama default
        assert_eq!(
            tensor_name(ArchVariant::Falcon, TensorKind::FfnGate, 0),
            "blk.0.ffn_gate.weight"
        );
    }

    #[test]
    fn test_k2_horizon_tensor_names() {
        let l = 3;
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::TokenEmbedd, l),
            "token_embd.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::Output, l),
            "output.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::OutputNorm, l),
            "output_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::AttnNorm, l),
            "blk.3.attn_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::AttnQ, l),
            "blk.3.attn_q.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::AttnK, l),
            "blk.3.attn_k.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::AttnV, l),
            "blk.3.attn_v.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::AttnOutput, l),
            "blk.3.attn_output.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::FfnNorm, l),
            "blk.3.ffn_norm.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::FfnGate, l),
            "blk.3.ffn_gate.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::FfnUp, l),
            "blk.3.ffn_up.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::FfnDown, l),
            "blk.3.ffn_down.weight"
        );
        // K2-Horizon specific
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::AttnGate, l),
            "blk.3.attn_gate.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::AttnVGate, l),
            "blk.3.attn_v_gate.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::AttnVGateBias, l),
            "blk.3.attn_v_gate.bias"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::AttnVExps, l),
            "blk.3.attn_v_exps.weight"
        );
        // MoE + shared experts
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::FfnGateInp, l),
            "blk.3.ffn_gate_inp.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::FfnUpExps, l),
            "blk.3.ffn_up_exps.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::FfnGateExps, l),
            "blk.3.ffn_gate_exps.weight"
        );
        assert_eq!(
            tensor_name(ArchVariant::K2Horizon, TensorKind::FfnDownExps, l),
            "blk.3.ffn_down_exps.weight"
        );
    }

    #[test]
    fn test_k2_horizon_arch_from_string() {
        assert_eq!(arch_from_string("k2_horizon"), ArchVariant::K2Horizon);
        assert_eq!(arch_from_string("k2-horizon"), ArchVariant::K2Horizon);
    }

    #[test]
    fn test_grouped_rmsnorm() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        // 2 groups of 3 elements
        let out = grouped_rmsnorm(&x, &w, 2, 1e-6);
        // Group 0: [1,2,3] -> mean_sq = (1+4+9)/3 = 4.666... -> rms = 2.160..., norm = [0.4629, 0.9258, 1.3887]
        // Group 1: [4,5,6] -> mean_sq = (16+25+36)/3 = 25.666... -> rms = 5.066..., norm = [0.7895, 0.9868, 1.1842]
        assert_eq!(out.len(), 6);
        assert!((out[0] - 0.4629).abs() < 0.001);
        assert!((out[1] - 0.9258).abs() < 0.001);
        assert!((out[2] - 1.3887).abs() < 0.001);
        assert!((out[3] - 0.7895).abs() < 0.001);
        assert!((out[4] - 0.9868).abs() < 0.001);
        assert!((out[5] - 1.1842).abs() < 0.001);
    }
}
