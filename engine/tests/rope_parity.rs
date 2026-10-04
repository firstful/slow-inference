//! RoPE parity test: our `rope()` against ggml's own op.
//!
//! `tests/data/rope_golden.bin` is produced by `tools/dump_rope.c`, which runs
//! ggml's `ggml_rope_ext()` over a known input in both GGML_ROPE_TYPE_NORMAL
//! and GGML_ROPE_TYPE_NEOX and records the result.
//!
//! The first version of this file asserted the engine to ~1e-6 absolute and
//! explained the slack in its header: "the two compute the angle in different
//! precision (we use f64 for the transcendental, ggml uses f32), so
//! bit-equality is not achievable or expected". **That was wrong.** ggml
//! evaluates the angle in f32 (`ggml_compute_forward_rope_flt`, ops.cpp:5953)
//! and so can the engine; the two tests below now hold it to bit equality
//! against the same golden. The 1e-6 assertion is kept as the original
//! regression it was written to be, but it is no longer the interesting gate —
//! `rope_is_bit_identical_to_ggml_not_merely_close` is, because a 1e-6 gate
//! passes a port that is 93.75% bit-identical, which is what the first f32
//! port actually was.
//!
//! This test exists because the NeoX/Normal distinction is silent when wrong:
//! both produce plausible-looking output, neither crashes. It was caught by
//! the end-to-end logit harness (scripts/compare_logits.py), not by review.
//!
//! Rebuild with: gcc -O2 -std=c11 -o /tmp/dump_rope tools/dump_rope.c \
//!   -I$HOME/llama.cpp/ggml/include -L$HOME/llama.cpp/build/bin \
//!   -lggml -lggml-base -lggml-cpu -lm && /tmp/dump_rope tests/data/rope_golden.bin

use std::path::Path;

use slow_inference::model::{RopeType, rope_apply};

struct Case {
    mode: i32,
    n_dims: usize,
    freq_base: f32,
    pos: i32,
    input: Vec<f32>,
    want: Vec<f32>,
}

fn parse(buf: &[u8]) -> Vec<Case> {
    assert_eq!(&buf[0..8], b"SLDROPE\0");
    let mut off = 8;
    let u32at = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let i32at = |b: &[u8], o: usize| i32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let f32at = |b: &[u8], o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let version = u32at(buf, off);
    off += 4;
    assert_eq!(version, 1, "rope golden format version mismatch");
    // cases are fixed-count; loop until the buffer is consumed
    let mut cases = Vec::new();
    while off < buf.len() {
        let mode = i32at(buf, off);
        let n_dims = u32at(buf, off + 4) as usize;
        let freq_base = f32at(buf, off + 8);
        let pos = i32at(buf, off + 12);
        off += 16;
        let take = n_dims * 4;
        let input: Vec<f32> = buf[off..off + take]
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        off += take;
        let want: Vec<f32> = buf[off..off + take]
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        off += take;
        cases.push(Case {
            mode,
            n_dims,
            freq_base,
            pos,
            input,
            want,
        });
    }
    cases
}

#[test]
fn rope_matches_ggml() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/rope_golden.bin");
    let buf = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing {}: {e}\nregenerate with engine/tools/dump_rope.c",
            path.display()
        )
    });
    let cases = parse(&buf);
    assert!(!cases.is_empty(), "no rope cases parsed");

    // ggml.h: GGML_ROPE_TYPE_NORMAL = 0, GGML_ROPE_TYPE_NEOX = 2
    const GGML_NORMAL: i32 = 0;
    const GGML_NEOX: i32 = 2;

    let mut checked_normal = 0;
    let mut checked_neox = 0;
    for c in &cases {
        let kind = match c.mode {
            GGML_NORMAL => {
                checked_normal += 1;
                RopeType::Norm
            }
            GGML_NEOX => {
                checked_neox += 1;
                RopeType::NeoX
            }
            other => panic!("unexpected ggml rope mode {other} in golden file"),
        };
        let mut got = c.input.clone();
        rope_apply(&mut got, c.pos as usize, c.freq_base as f64, kind);
        for (i, (g, w)) in got.iter().zip(c.want.iter()).take(c.n_dims).enumerate() {
            let d = (g - w).abs();
            assert!(
                d < 1e-6,
                "rope mode={} n_dims={} base={} pos={} elem {i}: got {:e} want {:e} (diff {d:e})",
                if kind == RopeType::Norm {
                    "NORMAL"
                } else {
                    "NEOX"
                },
                c.n_dims,
                c.freq_base,
                c.pos,
                g,
                w
            );
        }
    }
    assert!(
        checked_normal > 0 && checked_neox > 0,
        "both modes must be exercised"
    );
    eprintln!("[rope] {checked_normal} NORMAL + {checked_neox} NEOX cases vs ggml: OK");
}

/// The claim this file used to make in its header — that the engine "computes
/// the angle in different precision (we use f64 for the transcendental, ggml
/// uses f32), so bit-equality is not achievable or expected" — was false, and
/// the f64 rule was the reason the assertion had to be a tolerance at all.
///
/// `ggml_compute_forward_rope_flt` (ops.cpp:5953) evaluates the angle in f32
/// from end to end: `theta_scale = powf(...)` is an f32 pow, `ggml_rope_cache_init`
/// (ops.cpp:5846) multiplies the ladder in f32, and `rope_yarn` (ops.cpp:5829)
/// calls `sinf`/`cosf` on an f32. So the engine can be bit-identical, and now
/// is — and the 1e-6 tolerance above is a weaker gate than the implementation
/// deserves, which is worth knowing before anyone raises a real regression past
/// it without noticing.
///
/// Measured by `tools/probe_op_budget.c` on 4,755,456 values from the engine's
/// own forward pass: the f64 ladder is 2.11e-4 relative L2 from the reference
/// and differs in 89.6% of bits; the f32 ladder is bit-identical. f64 was
/// closer to the mathematics and further from llama.cpp.
///
/// The end-to-end consequence is the surprising half and is recorded in
/// `results/op_budget.json`: porting this moved the real-model residual by
/// nothing (4.0668e-6 -> 4.5492e-6 worst position on moon, i.e. noise). A 2e-4
/// per-op error that reaches neither the logits nor the next op's input is not
/// contributing to the residual, whatever a random-walk budget predicts. The
/// per-op number and the end-to-end number are different questions, and this is
/// the case where they come apart.
#[test]
fn rope_is_bit_identical_to_ggml_not_merely_close() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/rope_golden.bin");
    let buf = std::fs::read(&path).expect("rope golden");
    let cases = parse(&buf);
    let mut bits_differ = 0usize;
    let mut total = 0usize;
    for c in &cases {
        let kind = if c.mode == 0 {
            RopeType::Norm
        } else {
            RopeType::NeoX
        };
        let mut got = c.input.clone();
        rope_apply(&mut got, c.pos as usize, c.freq_base as f64, kind);
        for (g, w) in got.iter().zip(c.want.iter()).take(c.n_dims) {
            total += 1;
            if g.to_bits() != w.to_bits() {
                bits_differ += 1;
            }
        }
    }
    assert!(
        bits_differ == 0,
        "rope should be BIT-identical to ggml now, not close: {bits_differ}/{total} bits differ"
    );
    eprintln!("[rope] {total} values, {bits_differ} bits differ from ggml");
}
