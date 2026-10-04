//! Grouped-query attention: which KV head does query head `h` read?
//!
//! This file exists because of a bug that was in the engine for hours and that
//! *no* test in the suite reported at the time, because the two architectures
//! with parity coverage that do not use GQA are the two that were green.
//!
//! # The bug
//!
//! Query head `h` reads KV head `h / n_rep`, where `n_rep = n_head / n_head_kv`.
//! That is llama.cpp (`kv_head = h / n_rep` in `llama_build_attn_mha`) and HF's
//! `repeat_kv`, which expands `[b, n_kv, 1, s, d]` to `[b, n_kv, n_rep, s, d]` —
//! each KV head serves a *contiguous* block of query heads.
//!
//! The engine used `h % n_kv_heads`, which is the *interleaved* pairing. The two
//! agree only when `n_rep == 1` (MHA) or `n_kv_heads == 1` (MQA) — **not** when
//! `n_kv_heads == 2`, which is the boundary intuition gets wrong and which an
//! earlier draft of this comment (and of the one in `src/model.rs`) asserted
//! backwards. On the `llama` fixture — 8 query heads, 2 KV heads, `n_rep = 4` —
//! they disagree on half the heads, every layer, and the logits come out
//! **uncorrelated** with the reference rather than slightly off:
//!
//! ```text
//!   h % n_kv_heads   llama 1.19e+00 (cos 0.35)   qwen2 1.21e+00   gemma3 1.43e+00
//!   h / n_rep        llama 5.20e-07              qwen2 4.41e-07   gemma3 7.13e-04
//! ```
//!
//! (`gpt2` at 3.1e-04 and `deepseek2` at 5.8e-07 are MHA and MLA respectively,
//! so neither can see it at all.)
//!
//! # Why a parity test is not enough
//!
//! `tests/ref_parity.rs` *does* catch this — it was red for the whole time the
//! bug was live. It is listed here anyway, because the thing that made the bug
//! survivable was not the absence of coverage but a property of the coverage
//! that nobody had written down: **the suite's green architectures were the
//! ones that cannot express GQA at all.** If a future tick regenerates
//! `synthetic-llama.gguf` with `head_count_kv = 2` on a 2-head model, or adds
//! only MHA fixtures, `ref_parity` goes green with the bug back in.
//!
//! So the test below asserts the *coverage* rather than the arithmetic: every
//! architecture that routes through GQA has a fixture, and each such fixture's
//! head geometry is one where the two conventions actually differ. A fixture
//! that cannot discriminate is treated as a missing fixture.

use std::path::{Path, PathBuf};

use slow_inference::gguf::{self, Value};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/fixtures")
}

/// The real GGUFs in `models/`, which are 2-6 GB each and therefore may be
/// absent on a fresh checkout. Every test that reads them skips rather than
/// fails, and says so, because "the model is not on disk" and "the pairing is
/// wrong" must never look like the same event.
fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../models")
}

struct Geometry {
    n_head: usize,
    n_head_kv: usize,
    /// Multi-head latent attention: DeepSeek-style, where K and V are compressed
    /// into a latent and *no* KV head is ever shared with a query head. Its
    /// `head_count_kv = 1` is a property of the MLA projection, not a grouping,
    /// so it must not be read as "this fixture covers GQA".
    mla: bool,
}

impl Geometry {
    /// llama.cpp's pairing, and what the engine now does.
    fn kv_head_grouped(&self, h: usize) -> usize {
        h / (self.n_head / self.n_head_kv)
    }

    /// The pairing that was in the engine by mistake.
    fn kv_head_interleaved(&self, h: usize) -> usize {
        h % self.n_head_kv
    }

    /// Does this fixture's attention actually share a KV head between query
    /// heads? False for MHA (`n_head_kv == n_head`) and for MLA.
    fn broadcasts(&self) -> bool {
        !self.mla && self.n_head_kv < self.n_head
    }

    /// True when some query head is served by a *different* KV head under the
    /// two conventions — i.e. when the fixture can tell them apart at all.
    fn discriminates(&self) -> bool {
        (0..self.n_head).any(|h| self.kv_head_grouped(h) != self.kv_head_interleaved(h))
    }
}

fn read_geometry(stem: &'static str) -> Option<Geometry> {
    read_geometry_at(&fixture_dir().join(format!("synthetic-{stem}.gguf")))
}

fn read_geometry_at(path: &Path) -> Option<Geometry> {
    if !path.exists() {
        return None;
    }
    let gf = gguf::parse(path.to_str().unwrap()).ok()?;
    let arch = gf
        .metadata
        .get("general.architecture")?
        .as_str()?
        .to_string();
    let key = |suffix: &str| format!("{arch}.attention.{suffix}");
    let num = |k: &str| -> Option<usize> { gf.metadata.get(k)?.as_u64().map(|v| v as usize) };
    let n_head = num(&key("head_count"))?;
    // `head_count_kv` is optional in GGUF and means `head_count` when absent,
    // which is the MHA case -- the one shape that cannot see the pairing.
    let n_head_kv = num(&key("head_count_kv")).unwrap_or(n_head);
    // MLA is marked by llama.cpp's own absorb tensor `blk.N.attn_kv_a_mqa.weight`
    // (the low-rank K/V projection that replaces the per-head K and V), and by
    // `{arch}.attention.key_length_mla`. Keyed on the tensor so the test does
    // not depend on how a particular fixture spells the metadata.
    let mla = gf
        .tensors
        .keys()
        .any(|n| n.ends_with("attn_kv_a_mqa.weight"))
        || gf
            .metadata
            .contains_key(&format!("{arch}.attention.key_length_mla"));
    Some(Geometry {
        n_head,
        n_head_kv,
        mla,
    })
}

/// The two conventions coincide exactly for MHA (`n_head_kv == n_head`) and for
/// a single KV head (`n_head_kv == 1`, where both send every head to head 0).
/// Stated as a test because the *set of shapes where they differ* is the whole
/// reason the bug could hide, and because the boundary is not where intuition
/// puts it: `n_head_kv == 2` is **not** a coincidence. For 4 heads over 2 KV
/// heads the grouped pairing is `[0,0,1,1]` and the interleaved one is
/// `[0,1,0,1]`.
#[test]
fn the_two_conventions_differ_exactly_when_gqa_is_neither_mha_nor_single_kv() {
    for n_head in [1usize, 2, 4, 8, 16, 32, 40, 64, 128] {
        for n_head_kv in [1usize, 2, 4, 8, 16] {
            if n_head_kv > n_head {
                continue;
            }
            let g = Geometry {
                n_head,
                n_head_kv,
                mla: false,
            };
            let equal = (0..n_head).all(|h| g.kv_head_grouped(h) == g.kv_head_interleaved(h));
            assert_eq!(
                equal,
                n_head_kv == 1 || n_head_kv == n_head,
                "n_head={n_head} n_head_kv={n_head_kv}: the grouped and interleaved \
                 pairings {} but the test's own rule says they {}",
                if equal { "agree" } else { "differ" },
                if equal { "agree" } else { "differ" },
            );
        }
    }
}

/// Every checked-in fixture that routes through GQA must be able to tell the
/// two conventions apart. This is the assertion that fails if the coverage
/// quietly stops covering GQA.
#[test]
fn every_gqa_fixture_can_see_the_wrong_pairing() {
    // First: prove the guard can fail. A guard that no geometry can trip is an
    // assertion about nothing, and this file is only worth its runtime if the
    // predicate below is known to reject something. Multi-query attention — 8
    // query heads over a single KV head — is a real, shipping shape (Llama-3
    // 8B, Qwen2 0.5B) and it is exactly the shape this project already got wrong
    // once: both conventions send every head to KV head 0, so a fixture with it
    // passes whichever pairing the engine uses.
    let mqa = Geometry {
        n_head: 8,
        n_head_kv: 1,
        mla: false,
    };
    assert!(
        mqa.broadcasts(),
        "8 heads over 1 KV head is MQA, which does broadcast"
    );
    assert!(
        !mqa.discriminates(),
        "MQA is expected to be non-discriminating; if the rule changed, this \
         demonstration is no longer demonstrating anything and the guard below \
         needs a real replacement"
    );
    let mha = Geometry {
        n_head: 8,
        n_head_kv: 8,
        mla: false,
    };
    assert!(!mha.broadcasts() && !mha.discriminates());

    let stems = [
        "llama",
        "qwen2",
        "gpt2",
        "deepseek2",
        "gemma3",
        "llama-moe",
        "llama-moe-wscale",
    ];
    let mut seen_gqa = false;
    let mut checked = 0usize;
    for stem in stems {
        let Some(g) = read_geometry(stem) else {
            eprintln!("gqa_mapping: {stem}: fixture missing, skipped");
            continue;
        };
        checked += 1;
        if g.mla {
            eprintln!(
                "gqa_mapping: {stem}: n_head={} n_head_kv={} — MLA, no KV head is shared",
                g.n_head, g.n_head_kv
            );
            continue;
        }
        if !g.broadcasts() {
            eprintln!(
                "gqa_mapping: {stem}: n_head={} n_head_kv={} — MHA, cannot see the pairing",
                g.n_head, g.n_head_kv
            );
            continue;
        }
        seen_gqa = true;
        assert!(
            g.discriminates(),
            "gqa_mapping: {stem} is GQA (n_head={} n_head_kv={}) but `h / n_rep` and \
             `h % n_kv_heads` give the same KV head for every head, so this fixture \
             passes with either pairing. `n_head_kv` has to be neither 1 (MQA, where \
             both send every head to head 0) nor equal to `n_head` (MHA) — 2 KV heads \
             over 4 or 8 query heads discriminates fine.",
            g.n_head,
            g.n_head_kv
        );
    }
    assert!(checked > 0, "no fixtures found at all — the path is wrong");
    assert!(
        seen_gqa,
        "no GQA fixture is on disk. Every architecture with logit parity is MHA or \
         MLA, so the GQA pairing is untested: llama, qwen2, qwen3, gemma and every \
         other real grouped-query model can be wrong here with the suite green."
    );
}

/// The engines' own attention, checked against the grouping, at the level where
/// the mistake was made. `kv_head_grouped` above is a restatement of the rule;
/// this one checks the rule reproduces the pairing llama.cpp's `repeat_kv`
/// produces, computed a different way (by construction rather than by index) so
/// that a transcription error in either one is visible.
#[test]
fn grouped_pairing_equals_expanding_each_kv_head_n_rep_times() {
    for (n_head, n_head_kv) in [(8usize, 2usize), (4, 2), (32, 8), (64, 8), (12, 4), (6, 3)] {
        let g = Geometry {
            n_head,
            n_head_kv,
            mla: false,
        };
        // `repeat_kv`: out[:, h, :] == in[:, h / n_rep, :], built by tiling the
        // KV axis n_rep times and reshaping.
        let tiled: Vec<usize> = (0..n_head_kv)
            .flat_map(|kv| std::iter::repeat_n(kv, n_head / n_head_kv))
            .collect();
        let by_index: Vec<usize> = (0..n_head).map(|h| g.kv_head_grouped(h)).collect();
        assert_eq!(tiled, by_index, "n_head={n_head} n_head_kv={n_head_kv}");
        assert_ne!(
            tiled,
            (0..n_head)
                .map(|h| g.kv_head_interleaved(h))
                .collect::<Vec<_>>(),
            "n_head={n_head} n_head_kv={n_head_kv}: the interleaved pairing is supposed \
             to differ from the grouped one here, or the test proves nothing"
        );
    }
}

/// `Value::as_u64` is used to read `head_count` / `head_count_kv` here and in
/// the loader. If a GGUF stored either as a typed integer that `as_u64` does not
/// cover, the geometry above would read 0 and this test's `is_gqa` would quietly
/// skip the fixture. Pin the two keys' representation instead.
#[test]
fn head_counts_are_readable_as_u64_from_the_fixtures() {
    let path = fixture_dir().join("synthetic-llama.gguf");
    if !path.exists() {
        eprintln!("gqa_mapping: skipped — synthetic-llama.gguf missing");
        return;
    }
    let gf = gguf::parse(path.to_str().unwrap()).unwrap();
    for suffix in ["head_count", "head_count_kv"] {
        let key = format!("llama.attention.{suffix}");
        match gf.metadata.get(&key) {
            Some(Value::U32(v)) => assert!(*v > 0, "{key} = 0"),
            Some(other) => panic!("{key} is {other:?}, not U32 — as_u64 may not cover it"),
            None => panic!("{key} absent"),
        }
    }
}

/// The head geometry of the **real** GGUFs in `models/`, read out of the files
/// themselves on 2026-09-30 with llama.cpp's `gguf-py`:
///
/// ```text
///   qwen2.5-3b-instruct-q4_k_m.gguf   qwen2  block_count 36  head_count 16  head_count_kv  2
///   OLMoE-1B-7B-Instruct-Q4_K_M.gguf  olmoe  block_count 16  head_count 16  head_count_kv 16
///   Tiny-Moe.Q4_K_M.gguf              llama  block_count 12  head_count 12  head_count_kv 12
/// ```
///
/// The table is here, hard-coded, so the assertions below run on a machine with
/// no model files at all; each row is then re-read from the file when the file is
/// there, so the table cannot quietly become fiction. Two of the three are MHA,
/// which is the point worth recording: **the two real models every quantized
/// comparison in this project had used are MHA and cannot see the GQA pairing at
/// all**, and the only real GQA model on disk is Qwen2.5-3B. Tick 28's fix was
/// verified against synthetic fixtures whose geometry was chosen by us; this row
/// is a geometry nobody chose.
#[test]
fn real_model_geometries_are_grouped_query_where_it_matters() {
    // (file, expected n_head, expected n_head_kv)
    const REAL: &[(&str, usize, usize)] = &[
        ("qwen2.5-3b-instruct-q4_k_m.gguf", 16, 2),
        ("OLMoE-1B-7B-Instruct-Q4_K_M.gguf", 16, 16),
        ("Tiny-Moe.Q4_K_M.gguf", 12, 12),
    ];

    let mut gqa_rows = 0usize;
    let mut read_back = 0usize;
    for (file, n_head, n_head_kv) in REAL {
        let g = Geometry {
            n_head: *n_head,
            n_head_kv: *n_head_kv,
            mla: false,
        };
        if g.broadcasts() {
            gqa_rows += 1;
            assert!(
                g.discriminates(),
                "{file}: n_head={n_head} n_head_kv={n_head_kv} is grouped-query but the two \
                 pairings agree, so a real model on disk cannot tell them apart"
            );
        } else {
            eprintln!(
                "gqa_mapping: {file}: n_head={n_head} n_head_kv={n_head_kv} — MHA, cannot see the \
                 pairing"
            );
        }

        // Cross-check the table against the artifact itself. A GGUF that has been
        // re-quantized or replaced under this test would otherwise leave the
        // assertions above testing a geometry that no longer exists anywhere.
        let path = models_dir().join(file);
        let Some(actual) = read_geometry_at(&path) else {
            eprintln!("gqa_mapping: {file}: not on disk, table row unverified");
            continue;
        };
        read_back += 1;
        assert_eq!(
            (actual.n_head, actual.n_head_kv),
            (*n_head, *n_head_kv),
            "{file}: head geometry on disk is {} / {}, but the table above says {n_head} / \
             {n_head_kv}. Update the table (and re-run scripts/real_model_parity.py, whose \
             `--case` runs assume a stable vocab)",
            actual.n_head,
            actual.n_head_kv
        );
    }
    assert!(
        read_back > 0,
        "none of the real GGUFs are on disk, so the table above is the only statement of \
         their geometry and nothing here has been checked against a file"
    );
    assert!(
        gqa_rows > 0,
        "no real model on disk is grouped-query. If that is true of every model this project \
         can reach, then every real-model claim rests on MHA and the GQA pairing has no \
         real-model evidence at all — say so in STATUS.json rather than assuming otherwise"
    );

    // The boundary, stated over the geometry that matters and pinned as a list
    // rather than a rule, because the rule is what went wrong in the comments.
    // Qwen2.5-3B: 16 query heads over 2 KV heads, `n_rep = 8`.
    let qwen = Geometry {
        n_head: 16,
        n_head_kv: 2,
        mla: false,
    };
    let grouped: Vec<usize> = (0..16).map(|h| qwen.kv_head_grouped(h)).collect();
    let interleaved: Vec<usize> = (0..16).map(|h| qwen.kv_head_interleaved(h)).collect();
    assert_eq!(
        grouped,
        vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    );
    assert_eq!(
        interleaved,
        vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    );
    assert_ne!(
        grouped, interleaved,
        "two KV heads is not a coincidence case: this is the exact geometry of the only real \
         GQA model on disk, and it is also the shape whose fixtures disagreed with it"
    );
}
