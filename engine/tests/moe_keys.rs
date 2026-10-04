//! MoE hparam-key authority oracle: what each llama.cpp architecture passes to
//! `build_moe_ffn`, and which `{arch}.expert_*` GGUF keys it can see.
//!
//! `build_moe_ffn` takes `norm_w`, `w_scale` and `gating_op` as plain arguments
//! (llama-graph.h:1075-1077), so what a GGUF key is *worth* is decided per
//! architecture, in two separate places that are easy to conflate:
//!
//! 1. **whether the architecture reads the key at all.** ~50 architectures pass
//!    `hparams.expert_weights_scale` straight through and only 22 call
//!    `get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, ...)`. For the rest the field keeps
//!    its `0.0f` initialiser whatever the file says — and `build_moe_ffn` reads
//!    `0.0` as "no scale", not "scale by zero". Same shape for
//!    `expert_weights_norm`, whose default is `false`.
//! 2. **what it passes when the value is a literal.** `llama` passes a literal
//!    `true` for `norm_w` and never consults `hparams.expert_weights_norm`;
//!    `qwen2moe` passes a literal `false`. Reading the key for either is wrong.
//!
//! So neither question is answerable from a key's presence in a GGUF, and
//! `moe_norm_w_for` / `moe_scale_for` in `src/model.rs` are transcriptions of
//! `src/models/*.cpp`. `scripts/gen_moe_key_authority.py` derives the answer
//! mechanically — resolving the class hierarchy for both `load_arch_hparams` and
//! `using graph = <parent>::graph`, and both `build_moe_ffn` overloads — and this
//! file checks all sixty routing architectures against it, so the transcription
//! cannot rot when upstream moves one between groups.
//!
//! It has already earned its keep: the first oracle run found four entries wrong
//! in a table written a tick earlier. Two were string typos (`ernie4-5-moe` vs
//! `ernie4_5-moe`, `granite-hybrid` vs `granitehybrid`) that silently fell
//! through to the metadata default, and two were architectures missing entirely
//! (`gpt-oss`, `llama-embed`).

use serde_json::Value;
use slow_inference::model::{
    MOE_EXPERT_WEIGHTS_SCALE_READERS, moe_norm_w_for, moe_scale_for, supported_architectures,
    unimplemented_architectures,
};
use std::collections::BTreeSet;

const ORACLE: &str = include_str!("data/moe_key_authority.json");

/// Architectures llama.cpp has *dropped*, so they cannot appear in an oracle
/// generated from its source.
///
/// `k2_horizon` is here because the pinned checkout has no
/// `LLM_ARCH_K2_HORIZON`, no `models/k2-horizon.cpp`, and no class of that name;
/// the engine still names it in `unimplemented_architectures()` because it has a
/// forward pass. An exemption list rather than a filter, so adding an entry is
/// a decision someone has to write down.
const UPSTREAM_DROPPED: &[&str] = &["k2_horizon", "k2-horizon"];

fn oracle() -> Value {
    serde_json::from_str(ORACLE).expect("moe_key_authority.json is not valid JSON")
}

fn archs() -> serde_json::Map<String, Value> {
    match oracle()["archs"].as_object() {
        Some(m) => m.clone(),
        None => panic!("moe_key_authority.json has no `archs` object"),
    }
}

/// `(arch, norm_w, w_scale, gating_op)` as llama.cpp passes them, for every
/// architecture that calls `build_moe_ffn` at all.
type Routing = Vec<(String, Vec<String>, Vec<String>, Vec<String>)>;

fn routing() -> Routing {
    let mut out = Vec::new();
    for (arch, e) in archs() {
        if e["builds_moe"].as_bool() != Some(true) {
            continue;
        }
        let field = |k: &str| -> Vec<String> {
            e[k].as_array()
                .unwrap_or_else(|| panic!("{arch}: `{k}` is not an array"))
                .iter()
                .map(|v| v.as_str().expect("string").to_string())
                .collect()
        };
        out.push((
            arch.clone(),
            field("norm_w"),
            field("w_scale"),
            field("gating_op"),
        ));
    }
    out
}

#[test]
fn the_expert_weights_scale_reader_table_matches_llama_cpp() {
    let doc = oracle();
    let a = archs();
    let upstream: BTreeSet<String> = a
        .iter()
        .filter(|(_, e)| e["expert_weights_scale"] == Value::Bool(true))
        .map(|(a, _)| a.clone())
        .collect();
    let ours: BTreeSet<String> = MOE_EXPERT_WEIGHTS_SCALE_READERS
        .iter()
        .map(|s| s.to_string())
        .collect();

    let missing: Vec<&String> = upstream.difference(&ours).collect();
    let extra: Vec<&String> = ours.difference(&upstream).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "MOE_EXPERT_WEIGHTS_SCALE_READERS disagrees with llama.cpp {}:\n  \
         readable upstream but absent here: {missing:?}\n  \
         listed here but not read upstream: {extra:?}\n\
         regenerate with scripts/gen_moe_key_authority.py",
        doc["llama_cpp_commit"]
    );
    assert_eq!(
        ours.len(),
        22,
        "llama.cpp at {} reads expert_weights_scale in {} architectures. If that \
         count changed, this file was generated against a different checkout -- \
         regenerate it rather than editing the number.",
        doc["llama_cpp_commit"],
        ours.len()
    );
}

#[test]
fn the_norm_w_table_matches_every_routing_architecture_in_llama_cpp() {
    let rows = routing();
    assert!(
        rows.len() >= 55,
        "only {} architectures route; llama.cpp 3653e6d has 60. A parse failure \
         silently drops them, which is how the previous version of this script \
         lost 9 before anyone noticed.",
        rows.len()
    );
    let mut wrong = Vec::new();
    for (arch, norm_w, _, _) in &rows {
        // What upstream resolves to for an absent key, and for a present one.
        let expected = match norm_w.as_slice() {
            [v] if v == "literal_true" => (true, true),
            [v] if v == "literal_false" => (false, false),
            [v] if v == "key:expert_weights_norm" => (false, true),
            other => panic!("{arch}: norm_w {other:?} is not a shape this test knows"),
        };
        let got = (moe_norm_w_for(arch, false), moe_norm_w_for(arch, true));
        if got != expected {
            wrong.push(format!(
                "{arch}: upstream {norm_w:?} -> {expected:?}, engine {got:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn deferring_to_the_metadata_is_only_correct_where_the_key_is_read() {
    // The `else` arm of `moe_norm_w_for` returns the GGUF's
    // `expert_weights_norm`. That is upstream's behaviour *only* for
    // architectures that both forward the hparam and read it. If the two sets
    // ever stop coinciding, one of them is a bug: an architecture that forwards
    // an unread field sees `false` upstream and the file's value here.
    let rows = routing();
    let deferring: Vec<String> = rows
        .iter()
        .filter(|(_, nw, _, _)| nw.as_slice() == ["key:expert_weights_norm"])
        .map(|(a, _, _, _)| a.clone())
        .collect();
    let a = archs();
    let reading: Vec<String> = a
        .keys()
        .filter(|k| a[*k]["expert_weights_norm"] == Value::Bool(true))
        .cloned()
        .collect();
    assert_eq!(
        deferring, reading,
        "the architectures that forward expert_weights_norm and the ones that \
         read it are no longer the same set; one of the two is now wrong"
    );
    assert_eq!(
        deferring.len(),
        20,
        "20 architectures defer at llama.cpp 3653e6d; {} suggests the oracle is stale",
        deferring.len()
    );
}

#[test]
fn the_scale_is_applied_only_where_the_key_is_live() {
    // `w_scale_key_is_live` is the oracle's own statement of the bug this tick
    // fixed: the call site forwards `hparams.expert_weights_scale` and the loader
    // never fills it in, so any engine that reads the key directly scales the
    // routed weights by a factor the file's author chose for a different arch.
    let rows = routing();
    let forward_key: Vec<String> = rows
        .iter()
        .filter(|(_, _, ws, _)| ws.as_slice() == ["key:expert_weights_scale"])
        .map(|(a, _, _, _)| a.clone())
        .collect();
    assert_eq!(
        forward_key.len(),
        rows.len() - 1,
        "every routing architecture forwards w_scale except one (gemma4 passes a \
         literal 1.0f); {} of {} do",
        forward_key.len(),
        rows.len()
    );

    let mut wrong = Vec::new();
    for (arch, _, w_scale, _) in &rows {
        let reads = archs()[arch]["expert_weights_scale"] == Value::Bool(true);
        let forwarded_key = w_scale.as_slice() == ["key:expert_weights_scale"];
        // Upstream: the key's value iff it is both forwarded and read.
        let upstream_applies = reads && forwarded_key;
        let ours = moe_scale_for(arch, 2.5);
        let ours_applies = ours != 0.0;
        if ours_applies != upstream_applies {
            wrong.push(format!(
                "{arch}: upstream applies={upstream_applies} engine applies={ours_applies}"
            ));
        }
        if !upstream_applies {
            assert_eq!(ours, 0.0, "{arch}");
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

#[test]
fn an_architecture_that_does_not_read_the_key_gets_no_scale() {
    // The two facts that make the table load-bearing, stated directly so a future
    // edit to `moe_scale_for` cannot quietly invert them.
    assert_eq!(moe_scale_for("llama", 2.5), 0.0);
    assert_eq!(moe_scale_for("deepseek2", 2.5), 2.5);
    for arch in MOE_EXPERT_WEIGHTS_SCALE_READERS {
        assert_ne!(
            moe_scale_for(arch, 3.0),
            0.0,
            "{arch} reads the key but moe_scale_for ignores it"
        );
    }
}

#[test]
fn every_architecture_the_engine_names_is_in_the_oracle() {
    let archs = archs();
    let named: Vec<&str> = supported_architectures()
        .into_iter()
        .chain(unimplemented_architectures().iter().map(|(a, _)| *a))
        .collect();
    assert!(named.len() >= 30, "the architecture list shrank: {named:?}");
    let missing: Vec<&&str> = named
        .iter()
        .filter(|a| !UPSTREAM_DROPPED.contains(*a) && !archs.contains_key(**a))
        .collect();
    assert!(
        missing.is_empty(),
        "{missing:?} are named by the engine but absent from the oracle generated \
         against llama.cpp {} (or need adding to UPSTREAM_DROPPED with a reason)",
        oracle()["llama_cpp_commit"]
    );
}
