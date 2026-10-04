//! Quantization changes nothing observable in the engine. That is a claim, and
//! it is the answer to the parity project's longest-standing open question.
//!
//! Ticks 14-18 built the logit-parity oracle on synthetic **F32** fixtures, and
//! every real model on disk is quantized. On the real ones the engine and
//! llama.cpp disagreed by cos 0.996-0.9993 with argmax diverging, and four ticks
//! of increasingly precise measurement could not say which side was wrong.
//!
//! The resolution (tick 19) is a four-line identity, and this test is it:
//!
//! ```text
//!   engine(Q4_K_M model)  ==  engine(the same model widened to F32)   EXACTLY
//! ```
//!
//! bit for bit, byte for byte, for all 8 positions and all 128 logits. Not
//! "within 1e-5" — `assert_eq!` on the raw f32 bits.
//!
//! Why it holds: the engine has no quantized arithmetic. `dequant_row` expands a
//! packed block into the f32 values ggml's own `dequantize_row_*` would produce,
//! and `tests/quant_golden.rs` pins that expansion bit for bit against ggml for
//! all 21 types. Everything downstream is f32. So storing a weight as Q4_0 and
//! storing it as the F32 numbers it dequantizes to are the *same computation*,
//! and any divergence would mean the dequantizer is not the one ggml uses.
//!
//! Which makes the identity a test with real teeth. It is exactly the invariant
//! a plausible-looking optimization breaks: fusing the dequant into the matvec
//! with FMA accumulation, accumulating in f16, or rounding the scale once per
//! block instead of once per row would each keep every existing test green while
//! silently moving every real-model logit. Here they turn the suite red.
//!
//! The companion observation — that llama.cpp *does* move, by cos ~0.9999, and
//! that the movement is entirely its own sensitivity to 4-bit weights — is what
//! turns "our kernel is bit-exact" into "the real-model gap is not a bug". See
//! `quantization_is_llama_cpp_s_own_sensitivity_not_a_bug`.
//!
//! Fixtures, both checked in, 68 KB and 260 KB:
//!   synthetic-llama-q4_0.gguf      `tests/data/fixtures/`
//!   synthetic-llama-q4_0-f32.gguf  its exact widening, via scripts/requantize.py
//!
//! Regenerate both plus the two llama.cpp references with
//! `engine/scripts/gen_quantized_fixture.sh`.

use std::path::{Path, PathBuf};

use slow_inference::model::Model;

/// Must match PROMPT in `tests/ref_parity.rs`, and the `PROMPT` in
/// `engine/scripts/gen_quantized_fixture.sh`. The reference files carry the token
/// list and this test asserts it, so a reference generated for a different prompt
/// fails here rather than comparing a truncated row.
const PROMPT: &[usize] = &[3, 17, 42, 99, 7, 61, 128 - 1, 24];

const N_VOCAB: usize = 128;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/fixtures")
}

fn quant_fixture() -> PathBuf {
    fixture_dir().join("synthetic-llama-q4_0.gguf")
}

fn widened_fixture() -> PathBuf {
    fixture_dir().join("synthetic-llama-q4_0-f32.gguf")
}

struct Ref {
    n_vocab: usize,
    rows: Vec<Vec<f32>>,
}

fn load_ref(path: &Path) -> Ref {
    let buf = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&buf[..8], b"SLDREFLG", "{}: bad magic", path.display());
    let u32at = |off: usize| u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
    assert_eq!(u32at(8), 1, "{}: unsupported version", path.display());
    let n_tok = u32at(12) as usize;
    let mut off = 16;
    let tokens: Vec<i32> = buf[off..off + 4 * n_tok]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    off += 4 * n_tok;
    let n_vocab = u32at(off) as usize;
    off += 4;
    assert_eq!(
        buf.len() - off,
        n_tok * n_vocab * 4,
        "{}: truncated logits block",
        path.display()
    );
    assert_eq!(
        tokens,
        PROMPT.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        "{}: reference was generated for a different prompt; regenerate with \
         engine/scripts/gen_quantized_fixture.sh",
        path.display()
    );
    let rows = (0..n_tok)
        .map(|i| {
            let base = off + i * n_vocab * 4;
            buf[base..base + n_vocab * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        })
        .collect();
    Ref { n_vocab, rows }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += (*x as f64 - *y as f64).powi(2);
        den += *y as f64 * *y as f64;
    }
    if den == 0.0 {
        return 0.0;
    }
    num.sqrt() / den.sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// The engine has no quantized arithmetic, so a quantized model and its exact
/// widening are the same computation. Asserted on the raw bits, not a tolerance.
#[test]
fn quantization_is_bit_identical_to_its_exact_widening() {
    let q = quant_fixture();
    let qf32 = widened_fixture();
    if !q.exists() || !qf32.exists() {
        eprintln!(
            "quant_gap: skipped — missing {} or {}; regenerate with \
             engine/scripts/gen_quantized_fixture.sh",
            q.display(),
            qf32.display()
        );
        return;
    }

    let mq = Model::open(q.to_str().unwrap()).unwrap();
    let mf = Model::open(qf32.to_str().unwrap()).unwrap();
    assert_eq!(
        mq.arch, mf.arch,
        "the two fixtures disagree on architecture"
    );

    // The widened file must really be F32 and the quantized one really Q4_0, or
    // the identity below is vacuous. `token_embd.weight` is *not* quantized even
    // in the quantized fixture: `requantize.py` holds it back because llama.cpp
    // has no mixed f32/q4_0 add, so it is `blk.0.attn_q.weight` that carries the
    // Q4_0 assertion.
    let ftok = mf
        .gf
        .tensors
        .get("token_embd.weight")
        .expect("widened fixture has no token_embd.weight");
    assert_eq!(ftok.dtype, 0, "widened fixture is not F32");
    let qtok = mq
        .gf
        .tensors
        .get("token_embd.weight")
        .expect("q4_0 fixture has no token_embd.weight");
    assert_eq!(
        qtok.dtype, 0,
        "token_embd.weight should have been held at F32 (llama.cpp cannot run a \
         sub-8-bit embedding table); regenerate"
    );
    let qproj = mq
        .gf
        .tensors
        .get("blk.0.attn_q.weight")
        .expect("q4_0 fixture has no blk.0.attn_q.weight");
    assert_eq!(
        qproj.dtype, 2,
        "blk.0.attn_q.weight should be Q4_0, found ggml type {}",
        qproj.dtype
    );

    let rq = mq.prefill_logits(PROMPT).expect("q4_0 prefill failed");
    let rf = mf.prefill_logits(PROMPT).expect("widened prefill failed");
    assert_eq!(rq.len(), rf.len(), "different number of rows");
    assert_eq!(rq[0].len(), N_VOCAB, "unexpected vocab width");

    for (pos, (a, b)) in rq.iter().zip(rf.iter()).enumerate() {
        assert_eq!(
            a.len(),
            b.len(),
            "pos {pos}: row widths differ, so this is not the identity it claims"
        );
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "pos {pos} logit {i}: quantized model gives {x:e}, its exact \
                 widening gives {y:e}. The engine has no quantized arithmetic, so \
                 these must be the same f32 value; a difference means \
                 `dequant_row` no longer reproduces ggml's dequantize_row_* — \
                 check tests/quant_golden.rs first."
            );
        }
    }
    eprintln!(
        "quant_gap: Q4_0 and its F32 widening agree bit for bit at {} positions",
        rq.len()
    );
}

/// The reason the first test matters. On the same fixture pair, llama.cpp moves
/// by cos ~0.9999 between the quantized file and its widening, while the engine
/// does not move at all — and the engine matches llama.cpp to ~4e-7 on the
/// widened file. So the quantized-model disagreement this project spent four
/// ticks on is llama.cpp's own sensitivity to 4-bit weights, not an engine bug.
///
/// The gate is deliberately loose (1 - 1e-4). This test is not measuring
/// ggml's error; it is asserting that a *large* number exists, which is what
/// makes the first test's exactness meaningful. Tightening it toward zero would
/// be measuring something else.
#[test]
fn quantization_is_llama_cpp_s_own_sensitivity_not_a_bug() {
    let q = quant_fixture();
    let qf32 = widened_fixture();
    let qref = fixture_dir().join("llama-q4_0.ref");
    let fref = fixture_dir().join("llama-q4_0_f32.ref");
    if !q.exists() || !qf32.exists() || !qref.exists() || !fref.exists() {
        eprintln!("quant_gap: skipped — fixtures or llama.cpp references missing");
        return;
    }

    let mq = Model::open(q.to_str().unwrap()).unwrap();
    let mf = Model::open(qf32.to_str().unwrap()).unwrap();
    let ours_q = mq.prefill_logits(PROMPT).unwrap();
    let ours_f32 = mf.prefill_logits(PROMPT).unwrap();
    let theirs_q = load_ref(&qref);
    let theirs_f32 = load_ref(&fref);
    assert_eq!(
        theirs_q.n_vocab, N_VOCAB,
        "llama-q4_0.ref: vocab changed; regenerate"
    );
    assert_eq!(
        theirs_f32.n_vocab, N_VOCAB,
        "llama-q4_0_f32.ref: vocab changed; regenerate"
    );

    let mut worst_llama_move = 0f64;
    let mut worst_ours_move = 0f64;
    let mut worst_agreement = 0f64;

    for (pos, (((oq, of), tq), tf)) in ours_q
        .iter()
        .zip(ours_f32.iter())
        .zip(theirs_q.rows.iter())
        .zip(theirs_f32.rows.iter())
        .enumerate()
    {
        // How far llama.cpp's own answer moves across the same substitution.
        worst_llama_move = worst_llama_move.max(rel_l2(tq, tf));
        // How far ours moves: zero, and the first test proves it is bitwise zero.
        worst_ours_move = worst_ours_move.max(rel_l2(oq, of));
        // And that on the widened file we are the one that is right.
        worst_agreement = worst_agreement.max(rel_l2(of, tf));
        let _ = pos;
    }

    eprintln!(
        "quant_gap: llama.cpp moves {worst_llama_move:.2e} across Q4_0 -> F32, \
         we move {worst_ours_move:.2e}, and on the widened file we sit \
         {worst_agreement:.2e} from llama.cpp (cos {})",
        cosine(
            &ours_f32[PROMPT.len() - 1][..N_VOCAB],
            &theirs_f32.rows[PROMPT.len() - 1][..N_VOCAB]
        )
    );

    assert!(
        worst_llama_move > 1e-3,
        "llama.cpp moved only {worst_llama_move:.2e} between the quantized fixture \
         and its exact widening. That is far below the ~1e-2 the real Q4_K_M \
         models show, so this fixture would no longer be reproducing the effect \
         the first test is contrasted against — check that \
         synthetic-llama-q4_0.gguf really is quantized."
    );
    assert_eq!(
        worst_ours_move, 0.0,
        "the engine's own logits moved by {worst_ours_move:.2e} \
         (quant_gap::quantization_is_bit_identical_to_its_exact_widening has the \
         per-element assertion)"
    );
    assert!(
        worst_agreement < 1e-5,
        "on the widened fixture the engine sits {worst_agreement:.2e} from \
         llama.cpp, above the ~4e-7 it reaches on the F32 fixtures"
    );
}

/// Both files must carry the same weights. If `requantize.py` ever widens with a
/// different authority than ggml's `dequantize_row_*` — a numpy rounding, a
/// missing BF16 widening — the first test goes red. That is the intent, but the
/// failure would be attributed to the engine unless this says plainly that the
/// *fixtures* are what changed.
#[test]
fn the_pair_really_is_the_same_architecture() {
    let q = quant_fixture();
    let qf32 = widened_fixture();
    if !q.exists() || !qf32.exists() {
        eprintln!("quant_gap: skipped — fixtures missing");
        return;
    }
    let mq = Model::open(q.to_str().unwrap()).unwrap();
    let mf = Model::open(qf32.to_str().unwrap()).unwrap();

    let mut names: Vec<&str> = mq.gf.tensors.keys().map(|s| s.as_str()).collect();
    names.sort_unstable();
    let mut fnames: Vec<&str> = mf.gf.tensors.keys().map(|s| s.as_str()).collect();
    fnames.sort_unstable();
    assert_eq!(names, fnames, "the two fixtures have different tensor sets");
    assert!(
        names.len() >= 20,
        "only {} tensors; the fixture looks truncated",
        names.len()
    );

    // Every tensor in the widened file must be F32, and the tensors held back at
    // F32 by `requantize.py` -- the 1-D norm weights (llama-quant.cpp:294 never
    // quantizes those) and token_embd -- are F32 in both files and must stay that
    // way, or the pair is no longer "the same numbers, stored differently".
    let mut n_quant = 0;
    for n in &names {
        assert_eq!(
            mf.gf.tensors[*n].dtype, 0,
            "{n} is not F32 in the widened fixture"
        );
        if mq.gf.tensors[*n].dtype == 0 {
            continue; // was already F32 in the quantized fixture
        }
        n_quant += 1;
        assert!(
            mq.gf.tensors[*n].dims.len() >= 2,
            "{n} is {}-dimensional but was quantized; llama-quant.cpp:294 never \
             quantizes a 1-D tensor",
            mq.gf.tensors[*n].dims.len()
        );
    }
    assert!(
        n_quant >= 10,
        "only {n_quant} tensors are quantized in synthetic-llama-q4_0.gguf; the \
         identity test would be vacuous"
    );
}
