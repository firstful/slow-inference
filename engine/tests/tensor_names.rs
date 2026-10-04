//! Tensor-name oracle: `model::tensor_name()` vs llama.cpp's own `load_arch_tensors`.
//!
//! Companion to `tests/quant_golden.rs`, on the architecture side. That test proves
//! the dequantization kernels are bit-exact by linking against llama.cpp's C
//! kernels; this one proves the engine asks for the *same tensors llama.cpp asks
//! for*, by comparing against names extracted mechanically from
//! `src/models/*.cpp` by `scripts/gen_tensor_name_oracle.py`.
//!
//! Why it matters: the `block_q2_K` field-order bug (tick 13) was invisible until
//! a golden-vector harness existed. Tensor names have the same failure mode --
//! `pos_embd.weight` instead of `position_embd.weight` still compiles, still
//! emits plausible tokens for the layers that do load, and only aborts on the
//! architectures that need it. Here that is a failing test instead.
//!
//! Layer index 0 and expert index 0 are used throughout: llama.cpp substitutes
//! `blk.%d` with the block index, so block 0 names stand in for all of them.

use serde_json::Value;
use slow_inference::model::{
    ALL_TENSOR_KINDS, arch_from_string, supported_architectures, tensor_name,
    unimplemented_architectures,
};
use std::collections::BTreeSet;

const ORACLE: &str = include_str!("data/tensor_name_oracle.json");

/// Oracle tensor names the engine is knowingly missing, as `(gguf_arch, name)`.
///
/// Each entry is a decision, not an accident -- the engine cannot bind a tensor
/// it has no forward-pass code for. Removing an entry fails the test, which is
/// the point: closing a gap is a deliberate edit here.
const KNOWN_GAPS: &[(&str, &str)] = &[
    // Fused QKV. llama.cpp's create_tensor_qkv() prefers attn_qkv.weight and
    // only falls back to split q/k/v. Every GGUF in the supported set ships the
    // split form, so the engine reads split only. Architectures that *require*
    // the fused form (phi3, gpt2, falcon, openelm) are named explicitly.
    ("baichuan", "blk.0.attn_qkv.weight"),
    ("codeshell", "blk.0.attn_qkv.weight"),
    ("gemma", "blk.0.attn_qkv.weight"),
    ("gemma2", "blk.0.attn_qkv.weight"),
    ("gemma3", "blk.0.attn_qkv.weight"),
    ("internlm2", "blk.0.attn_qkv.weight"),
    ("llama", "blk.0.attn_qkv.weight"),
    ("nemotron", "blk.0.attn_qkv.weight"),
    ("olmo", "blk.0.attn_qkv.weight"),
    ("olmo2", "blk.0.attn_qkv.weight"),
    ("orion", "blk.0.attn_qkv.weight"),
    ("plamo", "blk.0.attn_qkv.weight"),
    ("qwen2", "blk.0.attn_qkv.weight"),
    ("qwen2moe", "blk.0.attn_qkv.weight"),
    ("smollm3", "blk.0.attn_qkv.weight"),
    ("stablelm", "blk.0.attn_qkv.weight"),
    ("xverse", "blk.0.attn_qkv.weight"),
    // ...and the mirror image: phi3 GGUFs are fused-only, engine reads fused.
    ("phi3", "blk.0.attn_q.weight"),
    ("phi3", "blk.0.attn_k.weight"),
    ("phi3", "blk.0.attn_v.weight"),
    // RoPE tables. The engine derives RoPE from metadata (rope_freq_base,
    // rope_scaling.*) instead of loading precomputed inv_freq tables. Only
    // LongRoPE-style checkpoints ship these; none in the supported set do.
    ("llama", "rope_freqs.weight"),
    ("llama", "rope_factors_long.weight"),
    ("llama", "rope_factors_short.weight"),
    ("phi3", "rope_factors_long.weight"),
    ("phi3", "rope_factors_short.weight"),
    // Optional biases the engine has no tensor kind for.
    ("llama", "blk.0.ffn_gate.bias"),
    ("qwen2", "output.bias"),
    // DeepSeek-V2: the engine implements both expert container forms -- the split
    // `ffn_{gate,up,down}_exps` pair and the fused `ffn_gate_up_exps` that
    // llama.cpp's `create_tensor_gate_up_exps` prefers -- plus the attn_kv_b MLA
    // branch. MTP (nextn.*) speculative-decode layers are not implemented.
    ("deepseek2", "blk.0.attn_kv_b.weight"),
    ("deepseek2", "blk.0.nextn.eh_proj.weight"),
    ("deepseek2", "blk.0.nextn.enorm.weight"),
    ("deepseek2", "blk.0.nextn.hnorm.weight"),
    ("deepseek2", "blk.0.nextn.embed_tokens.weight"),
    ("deepseek2", "blk.0.nextn.shared_head_head.weight"),
    ("deepseek2", "blk.0.nextn.shared_head_norm.weight"),
];

struct ArchOracle {
    gguf_arch: String,
    llm_arch: String,
    source: String,
    /// (name, required, condition)
    tensors: Vec<(String, bool, String)>,
}

fn load_oracle() -> Vec<ArchOracle> {
    let v: Value = serde_json::from_str(ORACLE).expect("oracle json parses");
    v["architectures"]
        .as_array()
        .expect("architectures is an array")
        .iter()
        .map(|a| ArchOracle {
            gguf_arch: a["gguf_arch"].as_str().unwrap().to_string(),
            llm_arch: a["llm_arch"].as_str().unwrap().to_string(),
            source: a["source"].as_str().unwrap().to_string(),
            tensors: a["tensors"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|t| {
                    let name = t.get("name")?.as_str()?;
                    // required == null means the flags expression was not a
                    // literal; treat as optional.
                    let required = t["required"].as_bool().unwrap_or(false);
                    let cond = match t["cond"].as_str() {
                        Some(c) => c.to_string(),
                        None => String::new(),
                    };
                    Some((name.to_string(), required, cond))
                })
                .collect(),
        })
        .collect()
}

/// Every GGUF tensor name `tensor_name()` can produce for `arch` at block 0.
fn engine_names(gguf_arch: &str, block: usize) -> BTreeSet<String> {
    let arch = arch_from_string(gguf_arch);
    ALL_TENSOR_KINDS
        .iter()
        .map(|k| tensor_name(arch, *k, block))
        .collect()
}

#[test]
fn supported_architectures_are_all_present_upstream() {
    let oracle = load_oracle();
    let upstream: BTreeSet<&str> = oracle.iter().map(|a| a.gguf_arch.as_str()).collect();
    let missing: Vec<&str> = supported_architectures()
        .iter()
        .copied()
        .filter(|a| !upstream.contains(a))
        .collect();
    assert!(
        missing.is_empty(),
        "engine claims to support architecture(s) absent from the pinned llama.cpp \
         ({oracle_commit}): {missing:?}. Either llama.cpp dropped the arch (move it to \
         unimplemented_architectures()) or the spelling is wrong.",
        oracle_commit = crate_commit()
    );
    // Guard against a silently-empty or half-parsed fixture.
    assert!(
        oracle.len() > 100,
        "oracle only has {} architectures",
        oracle.len()
    );
    assert!(
        oracle.iter().all(|a| a.tensors.len() > 3),
        "some architecture has an implausibly small tensor list; the extractor broke"
    );
}

#[test]
fn architecture_claims_are_disjoint_and_justified() {
    // `supported_architectures()` is a promise: every architecture in it has full
    // required-tensor coverage (see the test above) and exists upstream. Anything
    // the engine recognizes but cannot honour goes in
    // `unimplemented_architectures()` with a reason, so `--list-architectures`
    // cannot advertise something that will abort on load.
    let supported: BTreeSet<&str> = supported_architectures().into_iter().collect();
    let deferred: BTreeSet<&str> = unimplemented_architectures()
        .iter()
        .map(|(a, _)| *a)
        .collect();

    let overlap: Vec<&str> = supported.intersection(&deferred).copied().collect();
    assert!(
        overlap.is_empty(),
        "architecture(s) listed as both supported and unimplemented: {overlap:?}"
    );

    for (arch, why) in unimplemented_architectures() {
        assert!(
            !why.trim().is_empty(),
            "{arch} is deferred with no reason given"
        );
    }
}

fn crate_commit() -> &'static str {
    // inlined at compile time so the failure message names the pinned revision
    ORACLE
        .split("\"_source_commit\": \"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .unwrap_or("?")
}

#[test]
fn engine_covers_every_required_tensor_llama_cpp_asks_for() {
    let oracle = load_oracle();
    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for a in &oracle {
        if !supported_architectures().contains(&a.gguf_arch.as_str()) {
            continue;
        }
        let names = engine_names(&a.gguf_arch, 0);
        for (name, required, cond) in &a.tensors {
            if !*required {
                continue;
            }
            checked += 1;
            if !names.contains(name) {
                let where_ = if cond.is_empty() {
                    "unconditional".to_string()
                } else {
                    format!("under `{cond}`")
                };
                failures.push(format!(
                    "{} [{}] from {} {where_}: engine cannot name `{name}`",
                    a.gguf_arch, a.llm_arch, a.source
                ));
            }
        }
    }

    assert!(
        checked > 200,
        "only {checked} required tensors examined; extractor suspect"
    );
    assert!(
        failures.is_empty(),
        "{} required tensor(s) the engine cannot name:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn optional_tensor_gaps_are_deliberate() {
    let oracle = load_oracle();
    let mut failures: Vec<String> = Vec::new();

    for a in &oracle {
        if !supported_architectures().contains(&a.gguf_arch.as_str()) {
            continue;
        }
        let names = engine_names(&a.gguf_arch, 0);
        for (name, required, cond) in &a.tensors {
            if *required || names.contains(name) {
                continue;
            }
            if KNOWN_GAPS
                .iter()
                .any(|(g, n)| *g == a.gguf_arch && *n == name)
            {
                continue;
            }
            let where_ = if cond.is_empty() {
                "always".to_string()
            } else {
                format!("under `{cond}`")
            };
            failures.push(format!(
                "{}: `{name}` ({where_}, {}/{})",
                a.gguf_arch, a.llm_arch, a.source
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} optional tensor(s) not named by the engine. Add the name to model.rs, or \
         record it in KNOWN_GAPS if the engine genuinely has no code for it:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn known_gaps_are_still_gaps() {
    // A gap that has been fixed but left in the allowlist is rot: it hides the
    // next regression behind a stale exemption.
    let oracle = load_oracle();
    let mut stale: Vec<String> = Vec::new();
    for (arch, name) in KNOWN_GAPS {
        let a = match oracle.iter().find(|a| a.gguf_arch == *arch) {
            Some(a) => a,
            None => {
                stale.push(format!(
                    "{arch}/{name}: architecture no longer in the oracle"
                ));
                continue;
            }
        };
        if !a.tensors.iter().any(|(n, _, _)| n == name) {
            stale.push(format!(
                "{arch}/{name}: name no longer requested by llama.cpp"
            ));
        } else if engine_names(arch, 0).contains(*name) {
            stale.push(format!(
                "{arch}/{name}: engine now names it -- drop the exemption"
            ));
        }
    }
    assert!(
        stale.is_empty(),
        "stale KNOWN_GAPS entries:\n{}",
        stale.join("\n")
    );
}
