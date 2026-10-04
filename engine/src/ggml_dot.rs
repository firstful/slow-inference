//! `ggml_vec_dot_f32`, reproduced.
//!
//! The engine's f32 mat-mul used to be
//!
//! ```text
//! row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()
//! ```
//!
//! which is a *sequential f32* accumulation. That was a choice nobody had
//! checked against the reference, and it is not what llama.cpp does. On an F32
//! model the gap between the two was 2.1e-5..2.5e-4 on qwen2.5-3b — the floor
//! of every engine-vs-llama.cpp comparison in the project, and the reason
//! nothing quantized on this architecture could be resolved below ~2.5e-4.
//!
//! `ggml_vec_dot_f32` (`ggml/src/ggml-cpu/vec.cpp:11`) is not one algorithm. It
//! has a scalar branch, an AVX/AVX2 branch, an AVX-512 branch, an SVE branch
//! and a RISC-V branch, and they do not agree with each other on anything — so
//! "match llama.cpp" has no meaning until the loaded library's branch is known.
//! It was: the AVX branch, four `__m256` accumulators of 8 lanes, FMA, a
//! three-level pairwise f32 tree, then the `hadd` horizontal, then a leftover
//! loop that the compiler rewrote into two different shapes.
//!
//! Everything below is measured, not read off a build flag, by
//! `engine/tools/probe_f32_dot.c`, which runs the real `ggml_mul_mat` kernel
//! and reports both the relative error and the *bit pattern* of every output
//! row. It is bit-identical to the kernel at every length from 1 to 200, on
//! four real weight files. `engine/tests/dot_golden.rs` pins it against a
//! checked-in table of the kernel's own bits.
//!
//! Two consequences worth stating, because both were believed and are false:
//!
//! * **Accumulating in f64 is not the same as matching llama.cpp.** It is 30x
//!   more accurate (2.6e-8 against 9.4e-7 relative to exact) and it still
//!   disagrees with the kernel in 68% of rows, because the kernel is not
//!   accumulating in f64 — it is accumulating in f32 lanes. f64 is a better
//!   engine, not a parity one.
//! * **`CMakeCache.txt` cannot be used to pick the branch.** The cache for the
//!   reference build says `GGML_AVX=OFF`, `GGML_AVX2=OFF`, `GGML_FMA=OFF`, and
//!   the library it produced is an AVX2 build whose `ggml_vec_dot_f32` is
//!   bit-identical to the four-accumulator tree below. The cache is stale
//!   relative to the `.so`; the bit patterns are not.

/// Elements per vector register on the reference build (`GGML_F32_EPR`, 8 for
/// `__m256`).
const EPR: usize = 8;
/// Independent accumulators (`GGML_F32_ARR`, 4).
const ARR: usize = 4;
/// Elements consumed per step of the main loop (`GGML_F32_STEP`, 32).
const STEP: usize = EPR * ARR;

/// `ggml_vec_dot_f32` on the reference CPU build, bit for bit.
///
/// The shape, from `libggml-cpu.so`'s `ggml_vec_dot_f32`:
///
/// ```text
///   sum[0..4] = zero                       # ymm0..ymm3
///   for i in (0..n).step_by(32):
///       sum[j] = fma(x[i+8j .. i+8j+8], y[i+8j .. i+8j+8], sum[j])   # vfmadd231ps
///   sum[0] += sum[2];  sum[1] += sum[3];  sum[0] += sum[1]
///   t[k] = sum[0][k] + sum[0][k+4]         # vextractf128 + vaddps
///   tail  = (t[0] + t[1]) + (t[2] + t[3]) # vhaddps, vhaddps
///   ... leftovers ...
/// ```
///
/// `f32::mul_add` is the one-rounding form the intrinsics have; every `+` below
/// is a separate f32 rounding, in the order written. Do not "simplify" the
/// reduction into a `sum()` — that is the bug this module exists to remove.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
// The lane loops below are indexed rather than zipped, and clippy is right that
// indexing is usually worse -- except that here the index *is* the specification.
// `acc[j][l]` is register `j`, lane `l` of `_mm256_fmadd_ps`, and the reduction
// is the literal transcription of `GGML_F32x8_REDUCE`. Rewriting it as
// `for (j, acc) in acc.iter_mut().enumerate()` would keep the behaviour and lose
// the one property that makes this file checkable against a disassembly.
#[allow(clippy::needless_range_loop, clippy::assign_op_pattern)]
pub fn ggml_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot product operands must agree in length"
    );
    let n = a.len();
    let np = n & !(STEP - 1);

    let mut acc = [[0f32; EPR]; ARR];
    let mut i = 0;
    while i < np {
        for j in 0..ARR {
            let base = i + EPR * j;
            for l in 0..EPR {
                acc[j][l] = a[base + l].mul_add(b[base + l], acc[j][l]);
            }
        }
        i += STEP;
    }

    // GGML_F32x8_REDUCE: offset 2 folds four registers into two, offset 1 folds
    // two into one. The pair order is load-bearing: (s0+s2) + (s1+s3).
    for l in 0..EPR {
        acc[0][l] += acc[2][l];
    }
    for l in 0..EPR {
        acc[1][l] += acc[3][l];
    }
    for l in 0..EPR {
        acc[0][l] += acc[1][l];
    }
    let t = [
        acc[0][0] + acc[0][4],
        acc[0][1] + acc[0][5],
        acc[0][2] + acc[0][6],
        acc[0][3] + acc[0][7],
    ];
    let mut tail = (t[0] + t[1]) + (t[2] + t[3]);

    // The leftovers are the part that reads like a detail and is not. The
    // compiler gave the loop two shapes: up to three remaining terms go through
    // `vfmadd231ss` (one rounding), while four or more run as a 4-wide
    // `vmulps` + four `vaddss` block (a rounding per product *and* per add) and
    // only the final `tail % 4` terms are fused. Modelling the tail as one
    // fma chain is right at ne0 = 2, 3, 33, 34, 65 and wrong at 4, 5, 8, 9, 40,
    // 48, 70 — which is every real length with a tail, since a real model's
    // ne0 is a multiple of 32 and so has no tail at all.
    let mut k = np;
    if n - np <= 3 {
        while k < n {
            tail = a[k].mul_add(b[k], tail);
            k += 1;
        }
    } else {
        while k + 4 <= n {
            let p0 = a[k] * b[k];
            let p1 = a[k + 1] * b[k + 1];
            let p2 = a[k + 2] * b[k + 2];
            let p3 = a[k + 3] * b[k + 3];
            tail = tail + p0;
            tail = tail + p1;
            tail = tail + p2;
            tail = tail + p3;
            k += 4;
        }
        while k < n {
            tail = a[k].mul_add(b[k], tail);
            k += 1;
        }
    }
    tail
}

/// `ggml_compute_forward_rms_norm_f32`'s scale, bit for bit.
///
/// This one is *not* `ggml_vec_dot_f32` and getting it wrong is the reason a
/// blanket "accumulate in f64" or a blanket "use the SIMD tree" both fail as
/// advice. The RMSNorm reduction at `ggml/src/ggml-cpu/ops.cpp:3828` is a
/// hand-written loop:
///
/// ```text
///   ggml_float sum = 0.0;                        // a DOUBLE, not f32
///   for (i00 = 0; i00 < ne00; i00++)
///       sum += (ggml_float)(x[i00] * x[i00]);     // product rounded to f32 first
///   const float mean  = sum/ne00;                 // f64 divide, narrowed to f32
///   const float scale = 1.0f/sqrtf(mean + eps);
/// ```
///
/// So the two reductions in a transformer disagree with each other: the mat-mul
/// accumulates in f32 lanes and the norm accumulates in f64. An engine that
/// uses one rule for both is wrong in one of them, and which one it got wrong
/// is not visible in any accuracy metric -- both are "close".
#[must_use]
pub fn ggml_rms_norm_scale(x: &[f32], eps: f32) -> f32 {
    let mut sum = 0f64;
    for v in x {
        // `*v * *v` first, *then* widen: the f32 product rounding is part of
        // the kernel's answer and widening both operands to f64 would drop it.
        sum += (*v * *v) as f64;
    }
    let mean = (sum / x.len() as f64) as f32;
    1.0f32 / (mean + eps).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's previous line, kept so the tests can assert the port is
    /// actually different and not a relabelling of what it replaced.
    fn seq_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }

    /// A tiny LCG producing well-scaled, cancelling data. Two mistakes here
    /// cost this file two rounds of test-fixing, and both are silent:
    ///
    /// * `f32::from_bits((s >> 40) as u32)` has its exponent bits zeroed, so it
    ///   is a denormal; subtracting a constant from it and scaling back gives
    ///   a value that rounds to *the same float for every draw*. All 64
    ///   "random" vectors were constant, every product was equal, and any
    ///   summation order returned the same bits -- which made the sequential
    ///   sum and the tree indistinguishable.
    /// * A single vector is one sample. A dot product's rounding is a coin
    ///   flip, so every structural claim here is made over several seeds and
    ///   asserted as a fraction, never once.
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s
        };
        (0..n)
            .map(|_| {
                let m = (next() >> 41) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0;
                // 16x of spread, so the products cancel the way a residual
                // stream's do instead of all landing in the same binade.
                let e = ((next() >> 58) % 9) as i32 - 4;
                m * 2f32.powi(e)
            })
            .collect()
    }

    /// Sixteen seeds. Six was not enough: at n=4 the two leftover shapes come
    /// out 3-for-3 on a coin flip, and a threshold that tight either fails on a
    /// fair result or has to be loosened until it proves nothing.
    const SEEDS: [u64; 16] = [
        1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597,
    ];

    /// How many of `SEEDS` give a different bit pattern under `f` than the
    /// value it is compared against. Sampled, not reasoned about.
    fn differs_over_seeds(mut f: impl FnMut(&[f32], &[f32]) -> u32) -> usize {
        let mut n = 0;
        for s in SEEDS {
            let a = noise(512, s * 0x9e37_79b9);
            let b = noise(512, s * 0x51ed_270b);
            if f(&a, &b) != 0 {
                n += 1;
            }
        }
        n
    }

    /// The reduction tree with no tail, so a test can perturb one part of the
    /// reduction without the leftover loop getting in the way. Indexed for the
    /// same reason the port is: `acc[j][l]` is one lane of one register, and
    /// that correspondence is what the tests are checking.
    #[allow(clippy::needless_range_loop, clippy::assign_op_pattern)]
    fn tree_only(a: &[f32], b: &[f32]) -> f32 {
        let np = a.len() & !(STEP - 1);
        let mut acc = [[0f32; EPR]; ARR];
        let mut i = 0;
        while i < np {
            for j in 0..ARR {
                for l in 0..EPR {
                    let k = i + EPR * j + l;
                    acc[j][l] = a[k].mul_add(b[k], acc[j][l]);
                }
            }
            i += STEP;
        }
        for l in 0..EPR {
            acc[0][l] += acc[2][l];
        }
        for l in 0..EPR {
            acc[1][l] += acc[3][l];
        }
        for l in 0..EPR {
            acc[0][l] += acc[1][l];
        }
        let t = [
            acc[0][0] + acc[0][4],
            acc[0][1] + acc[0][5],
            acc[0][2] + acc[0][6],
            acc[0][3] + acc[0][7],
        ];
        (t[0] + t[1]) + (t[2] + t[3])
    }

    /// Same as the port, but with a pure-fma leftover: the naive reading of
    /// ggml's `sumf += x[i]*y[i]`.
    fn fma_tail(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut acc = tree_only(a, b);
        for k in (n & !(STEP - 1))..n {
            acc = a[k].mul_add(b[k], acc);
        }
        acc
    }

    #[test]
    fn every_length_runs() {
        for n in 0..=2048usize {
            let a = noise(n, 0x51ed_270b);
            let b = noise(n, 0x9e37_79b9);
            assert!(ggml_dot_f32(&a, &b).is_finite(), "n={n}");
        }
    }

    #[test]
    fn the_tree_differs_from_the_sequential_sum_the_engine_used_to_do() {
        let (mut same, mut total) = (0, 0);
        for n in (STEP..=2048usize).step_by(STEP) {
            let a = noise(n, 0x51ed_270b);
            let b = noise(n, 0x9e37_79b9);
            total += 1;
            if ggml_dot_f32(&a, &b).to_bits() == seq_f32(&a, &b).to_bits() {
                same += 1;
            }
        }
        assert_eq!(total, 64);
        // and over independent seeds at a single length, for the same reason
        // Not "every length": two orderings of the same products do coincide
        // on the occasional well-conditioned input, and a test that insists
        // they never do is a test that gets "fixed" by weakening the
        // implementation. 90% is far above the coincidence rate and far below
        // anything a real match could reach.
        assert!(
            same * 10 < total,
            "{same} of {total} tail-free lengths returned the same bits as a \
             sequential f32 sum, which is not what a 4-accumulator tree does"
        );
        let over_seeds = differs_over_seeds(|a, b| {
            let n = a.len();
            u32::from(
                ggml_dot_f32(&a[..n], &b[..n]).to_bits() != seq_f32(&a[..n], &b[..n]).to_bits(),
            )
        });
        assert_eq!(
            over_seeds,
            SEEDS.len(),
            "only {over_seeds} of {} seeds at n=512 differ from a sequential f32 sum",
            SEEDS.len()
        );
    }

    /// The lane assignment and the intra-lane order are both load-bearing. Note
    /// what is *not* asserted: rotating all four lanes is a symmetry of this
    /// reduction -- (s0+s2)+(s1+s3) is invariant under swapping 0<->3 and
    //  / 1<->2 -- so "rotate the input" would pass on a port that had already
    //  collapsed to a single accumulator. Moving one product across a lane
    ///  boundary is the perturbation that cannot be absorbed.
    #[test]
    fn the_tree_shape_is_load_bearing() {
        let across = differs_over_seeds(|a, b| {
            let mut m = a.to_vec();
            m.swap(0, 8);
            u32::from(tree_only(&m, b).to_bits() != tree_only(a, b).to_bits())
        });
        assert_eq!(
            across,
            SEEDS.len(),
            "moving one product from accumulator 0 to accumulator 1 only mattered in \
             {across} of {} seeds, so the lane assignment is not load-bearing",
            SEEDS.len()
        );

        let within = differs_over_seeds(|a, b| {
            let mut m = a.to_vec();
            m.swap(0, 1);
            u32::from(tree_only(&m, b).to_bits() != tree_only(a, b).to_bits())
        });
        assert_eq!(
            within,
            SEEDS.len(),
            "the FMA order inside an accumulator only mattered in {within} of {} seeds",
            SEEDS.len()
        );
    }

    /// The leftover loop has two shapes and the port needs both. A pure fma
    /// chain is right for a tail of 1..3 and wrong for 4 or more, so it looks
    /// correct in exactly the lengths nobody thinks to test.
    ///
    /// The wrong shape is not *always* wrong. Measured over 16 seeds at 15
    /// lengths with a tail of 4 or more, a fma chain and the 4-wide block
    /// disagree on about a third of inputs and agree on the rest: when most of
    /// the tail is fused the two orders round the same way. So the assertion
    /// is a pooled separation, not a per-length identity. Identical functions
    /// score 0 here, which is the whole point.
    #[test]
    fn the_leftover_loop_has_two_shapes_and_needs_both() {
        const TAIL4PLUS: [usize; 15] = [4, 5, 7, 8, 9, 16, 20, 36, 40, 48, 56, 63, 70, 80, 100];
        let (mut differ, mut total) = (0, 0);
        let mut table = String::new();
        for &n in &TAIL4PLUS {
            let mut at_n = 0;
            for s in SEEDS {
                let a = noise(n, s * 0x9e37_79b9 + 3);
                let b = noise(n, s * 0x51ed_270b + 4);
                total += 1;
                if ggml_dot_f32(&a, &b).to_bits() != fma_tail(&a, &b).to_bits() {
                    differ += 1;
                    at_n += 1;
                }
            }
            table.push_str(&format!(" n{at_n}"));
        }
        eprintln!(
            "[leftover] seeds differing from a fused tail, of {}:{table}",
            SEEDS.len()
        );
        assert_eq!(total, TAIL4PLUS.len() * SEEDS.len());
        // A quarter. Identical functions score 0, so this discriminates, and it
        // is far below any plausible rate for genuinely different functions, so
        // it is not a constant fitted to make the current numbers pass. There
        // is deliberately no upper bound: whether two different roundings happen
        // to agree on a given input is a property of the data, not a fact to
        // pin, and an earlier draft asserted one and was wrong twice.
        assert!(
            differ * 4 >= total,
            "the 4-wide block tail and a fused tail disagreed on only {differ} of \
             {total} samples, so this test cannot see the difference between them"
        );

        // A tail of 1..3: the kernel fuses, so here the fma chain is correct
        // and the block path would be the bug. This one *is* an identity -- the
        // two are the same function there -- so every seed must match.
        for n in [33usize, 34, 35, 65, 97] {
            for s in SEEDS {
                let a = noise(n, s * 0x9e37_79b9 + 5);
                let b = noise(n, s * 0x51ed_270b + 6);
                assert_eq!(
                    ggml_dot_f32(&a, &b).to_bits(),
                    fma_tail(&a, &b).to_bits(),
                    "n={n} leaves a tail of {}, where the kernel fuses and so must the port",
                    n % STEP
                );
            }
        }
    }

    /// The engine's previous RMSNorm, for the same reason `seq_f32` is kept.
    fn rmsnorm_old(x: &[f32], eps: f32) -> f32 {
        let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        1.0 / (ss + eps).sqrt()
    }

    #[test]
    fn the_rms_norm_reduction_is_a_double_accumulator_over_f32_products() {
        // Both rules are in the same file on purpose: the mat-mul uses the f32
        // tree and the norm uses f64, and a port that unifies them is wrong in
        // one of the two places.
        let (mut diff, mut total) = (0, 0);
        for n in [128usize, 512, 2048, 4096] {
            for s in SEEDS {
                let x = noise(n, s * 0x9e37_79b9 + 11);
                total += 1;
                if ggml_rms_norm_scale(&x, 1e-6).to_bits() != rmsnorm_old(&x, 1e-6).to_bits() {
                    diff += 1;
                }
            }
        }
        assert!(
            diff * 4 >= total,
            "the f64 and f32 RMSNorm reductions agreed on {} of {total} samples, so this \
             test cannot see the accumulator it is here to pin",
            total - diff
        );
    }

    #[test]
    fn a_mismatched_operand_length_is_refused_not_truncated() {
        assert!(std::panic::catch_unwind(|| ggml_dot_f32(&[1.0, 2.0], &[1.0])).is_err());
    }
}
