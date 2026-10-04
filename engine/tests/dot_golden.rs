//! `ggml_dot_f32` must reproduce llama.cpp's kernel bit for bit.
//!
//! `tests/data/dot_golden.bin` is produced by `tools/probe_f32_dot.c
//! --emit-golden`, which links the same `libggml-cpu.so` the oracle uses and
//! calls the real `ggml_mul_mat` on the CPU backend. Every entry is one dot
//! product the reference actually computed: the operands travel with the
//! result, so this test cannot accidentally agree with the generator about a
//! PRNG it never sees.
//!
//! This is the check that makes the f32 residual a fixed thing rather than a
//! tolerance. The engine used to accumulate f32 dot products sequentially,
//! which put it 2.1e-5..2.5e-4 away from the reference on qwen2.5-3b -- the
//! floor of every comparison in the project. Accumulating in f64 does not fix
//! that (it is still wrong in 68% of rows, because the reference accumulates in
//! f32 lanes), and reproducing the reference's actual reduction does.
//!
//! Regenerate with `engine/scripts/gen_dot_golden.sh`.

use std::path::Path;
use std::process::Command;

use slow_inference::ggml_dot::ggml_dot_f32;

struct Case {
    ne0: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    expected: u32,
}

fn load(path: &Path) -> Vec<Case> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}\nregenerate with engine/scripts/gen_dot_golden.sh",
            path.display()
        )
    });
    assert_eq!(&bytes[0..8], b"SLDOTGLD", "golden file magic mismatch");
    let u32at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    assert_eq!(u32at(8), 1, "golden file version mismatch");
    let declared = u32at(12) as usize;
    let mut out = Vec::with_capacity(declared);
    let mut o = 16usize;
    let read = |o: &mut usize| -> &[u8] {
        let s = &bytes[*o..*o + 4];
        *o += 4;
        s
    };
    for _ in 0..declared {
        let ne0 = u32::from_le_bytes(read(&mut o).try_into().unwrap()) as usize;
        let mut a = vec![0f32; ne0];
        let mut b = vec![0f32; ne0];
        for v in a.iter_mut() {
            *v = f32::from_le_bytes(read(&mut o).try_into().unwrap());
        }
        for v in b.iter_mut() {
            *v = f32::from_le_bytes(read(&mut o).try_into().unwrap());
        }
        let expected = u32::from_le_bytes(read(&mut o).try_into().unwrap());
        out.push(Case {
            ne0,
            a,
            b,
            expected,
        });
    }
    assert_eq!(o, bytes.len(), "trailing bytes in golden file");
    out
}

#[test]
fn ggml_dot_f32_is_bit_identical_to_the_reference_kernel() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/dot_golden.bin");
    let cases = load(&path);
    assert!(
        cases.len() >= 100,
        "suspiciously few golden cases: {}",
        cases.len()
    );

    let mut by_len: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for c in &cases {
        *by_len.entry(c.ne0).or_default() += 1;
    }

    let mut bad = Vec::new();
    for c in &cases {
        let got = ggml_dot_f32(&c.a, &c.b);
        if got.to_bits() != c.expected {
            bad.push(format!(
                "ne0={} expected 0x{:08x} got 0x{:08x} ({got} vs {})",
                c.ne0,
                c.expected,
                got.to_bits(),
                f32::from_bits(c.expected)
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} dot products differ from the kernel:\n  {}",
        bad.len(),
        cases.len(),
        bad.iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );

    // The table is only a pin if it reaches past the main loop into both of the
    // leftover shapes. tail <= 3 takes the fused path, tail >= 4 takes the
    // 4-wide block path, and modelling the tail as a single fma chain gets the
    // first right and the second wrong.
    let has_fused_tail = by_len.keys().any(|&n| n % 32 > 0 && n % 32 <= 3);
    let has_block_tail = by_len.keys().any(|&n| n % 32 >= 4);
    let has_no_tail = by_len.keys().any(|&n| n % 32 == 0);
    assert!(
        has_fused_tail,
        "golden table has no length with a 1..3 element tail"
    );
    assert!(
        has_block_tail,
        "golden table has no length with a tail of 4 or more"
    );
    assert!(
        has_no_tail,
        "golden table has no length with no tail at all"
    );
}

#[test]
fn the_sequential_sum_the_engine_used_is_not_the_kernel() {
    // Without this, "bit-identical" could be satisfied by an implementation
    // that happens to coincide on 243 gaussian dot products, and the change
    // would be indistinguishable from leaving the old line in place.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/dot_golden.bin");
    let cases = load(&path);
    let tail_free: Vec<&Case> = cases
        .iter()
        .filter(|c| c.ne0 % 32 == 0 && c.ne0 >= 32)
        .collect();
    assert!(
        !tail_free.is_empty(),
        "the golden table has no tail-free case of length >= 32, so it cannot tell a \
         4-accumulator tree from a sequential sum"
    );
    let old_matches = tail_free
        .iter()
        .filter(|c| {
            let old: f32 = c.a.iter().zip(&c.b).map(|(x, y)| x * y).sum();
            old.to_bits() == c.expected
        })
        .count();
    // Not zero, and an earlier draft of this asserted zero: 7 of 67 is the
    // ordinary coincidence rate for two roundings of the same products on
    // well-conditioned random data. What would invalidate the module is the
    // opposite -- the old line matching nearly everywhere, which is what an
    // implementation that had silently collapsed to a sequential sum would do.
    assert!(
        old_matches * 4 < tail_free.len(),
        "the old sequential f32 sum already matched the kernel on {old_matches} of {} \
         tail-free cases; either the table is not exercising the reduction or the kernel \
         is not what the module in src/ggml_dot.rs says it is",
        tail_free.len()
    );
    eprintln!(
        "[dot_golden] {} tail-free cases; the old sequential f32 sum coincided on {}",
        tail_free.len(),
        old_matches
    );
}

#[test]
fn the_golden_is_the_real_kernel_and_not_a_reimplementation() {
    // probe_f32_dot links libggml and runs a real ggml graph. If the build
    // script ever stops passing -lggml, the table would still exist and the
    // other two tests would still pass -- against a file nobody regenerated.
    let sh = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/gen_dot_golden.sh");
    let text = std::fs::read_to_string(&sh)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", sh.display()));
    for needle in [
        "-lggml",
        "probe_f32_dot.c",
        "--emit-golden",
        "-ffp-contract=off",
    ] {
        assert!(
            text.contains(needle),
            "gen_dot_golden.sh no longer mentions {needle}"
        );
    }
    let out = Command::new("sh")
        .arg("-c")
        .arg("grep -n 'ggml_vec_dot_f32' \"$LLAMA_CPP/ggml/src/ggml-cpu/vec.cpp\" | head -1")
        .env(
            "LLAMA_CPP",
            std::env::var("LLAMA_CPP").unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                format!("{home}/llama.cpp")
            }),
        )
        .output();
    // A missing llama.cpp checkout is not this test's business: the golden file
    // is the evidence and it is already checked above. This is a warning that
    // the port may be pinned to a function that has moved.
    if let Ok(o) = out
        && o.status.success()
    {
        let s = String::from_utf8_lossy(&o.stdout);
        assert!(
            s.contains("ggml_vec_dot_f32"),
            "the reference vec.cpp no longer defines ggml_vec_dot_f32; the port in \
             src/ggml_dot.rs is pinned to a function that has moved"
        );
    }
}

/// The function in `src/ggml_dot.rs` is bit-exact, and every dot product in the
/// engine must be *that* function.
///
/// Those are different claims and the suite can only see the first one. A
/// change that left 23 of 24 call sites on a sequential `sum::<f32>()` was
/// invisible here: all 163 tests passed, because the synthetic fixtures are 12
/// layers of 768 and their tolerances are loose. The real 3B GQA model is where
/// it showed, and the number is worth recording because it is the reason this
/// test exists:
///
/// ```text
///                     capital      moon      vs llama.cpp(F32)
///   matvec only     2.0705e-05  1.8984e-04     the whole residual
///   all 24 sites    3.5300e-06  4.0668e-06     5.9x / 46.7x better
/// ```
///
/// So this test reads the source and requires that no dot product in the
/// forward passes is written as a sequential f32 sum. It is a source scan and
/// that is a real cost, paid on purpose: a runtime test cannot tell which of two
/// correctly-implemented reductions the engine chose, because the fixtures are
/// too small for the choice to matter.
///
/// It used to allow two survivors -- the mean and the variance in `layernorm`,
/// which are reductions of one vector rather than dot products -- and it failed
/// when they were ported, on purpose: the note above them claimed the reference
/// reduces both in `double` "because `ggml_vec_norm_f32` accumulates into a
/// `ggml_float`", and that note was wrong. The variance is a horizontal tree
/// over groups of 8 feeding a `ggml_float`, not a sequential double sum, so a
/// blanket f64 is right for the mean and wrong for the variance. They are now
/// ported in `src/ggml_norm.rs` and pinned by `tests/norm_golden.rs`, and this
/// test asserts there are no survivors left rather than naming which two to
/// expect.
#[test]
fn every_dot_product_in_the_forward_passes_uses_the_reference_reduction() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/model.rs");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let lines: Vec<&str> = src.lines().collect();

    // A dot product written inline. `ggml_dot_f32(a, b)` is the only spelling
    // allowed; anything that zips two sequences of f32 and sums the products in
    // f32 is a sequential accumulation again.
    let is_inline_dot = |l: &str| {
        l.contains(".iter()")
            && l.contains(".zip(")
            && l.contains("a * b")
            && l.contains("sum::<f32>()")
    };

    let mut offenders = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if is_inline_dot(l) {
            offenders.push(format!("  src/model.rs:{}: {}", i + 1, l.trim()));
        }
    }
    assert!(
        offenders.is_empty(),
        "{} dot product(s) in the forward passes are still a sequential f32 sum, \
         which on qwen2.5-3b costs 5.9x (capital) to 46.7x (moon) of agreement \
         with llama.cpp and is invisible to every other test in this crate:\n{}",
        offenders.len(),
        offenders.join("\n")
    );

    // No survivors. The old test required exactly two of them and named them,
    // which is a comment that goes stale silently; requiring zero is both
    // shorter and the stronger statement. The port that closed them is in
    // src/ggml_norm.rs and is pinned by tests/norm_golden.rs, which also covers
    // the reductions this scan does not look at -- they take one argument, so
    // they cannot match `is_inline_dot`.
    let norm_lines: Vec<&str> = lines
        .iter()
        .copied()
        .filter(|l| l.contains("x.iter().sum::<f32>()") || l.contains("(v - mean) * (v - mean)"))
        .collect();
    assert!(
        norm_lines.is_empty(),
        "the `layernorm` reductions are back inline in src/model.rs, which is where they \
         were for eight ticks while the note above them claimed the reference used \
         double for both:\n  {}",
        norm_lines.join("\n  ")
    );
    assert_eq!(
        src.matches("ggml_layernorm(").count(),
        1,
        "expected exactly one call of the LayerNorm port in src/model.rs -- the body of \
         Model::layernorm -- so a copy has not been reintroduced alongside it"
    );

    // And the other half of the claim: the sites must actually be calling the
    // port, not a local re-implementation of it.
    let calls = src.matches("ggml_dot_f32(").count();
    assert!(
        calls >= 24,
        "only {calls} call sites of ggml_dot_f32 remain in src/model.rs; there were 24 \
         after the conversion, and a forward pass has been deleted or bypassed"
    );
}
