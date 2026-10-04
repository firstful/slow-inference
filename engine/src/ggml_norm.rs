//! `ggml_compute_forward_norm_f32`, reproduced.
//!
//! This is the last f32 reduction in the engine that was not the reference's,
//! and getting it right required correcting the reason the other 23 were left
//! alone. `Model::layernorm` used to carry this note:
//!
//! > the reference reduces them in `double` (`ggml_vec_norm_f32` and
//! > `ggml_vec_rmsnorm_f32` both accumulate into a `ggml_float`)
//!
//! Both halves of that are wrong at llama.cpp 3653e6d.
//!
//! * `ggml_vec_rmsnorm_f32` no longer exists. RMSNorm is a hand-written loop
//!   in `ggml_compute_forward_rms_norm_f32` (ops.cpp:3828) with a `ggml_float`
//!   accumulator over f32-rounded products, and that is what `ggml_dot.rs`'s
//!   `ggml_rms_norm_scale` already reproduces.
//! * `ggml_vec_norm_f32` (vec.h:858) is now a single line --
//!   `ggml_vec_dot_f32(n, s, 0, x, 0, x, 0, 1); *s = sqrtf(*s);` -- so it is
//!   the *dot* kernel, and LayerNorm does not call it.
//!
//! What LayerNorm actually calls, in `ggml_compute_forward_norm_f32`
//! (ops.cpp:3698), is **two reductions with two different shapes**:
//!
//! ```text
//!   float sum = 0.0;
//!   ggml_vec_sum_f32(ne00, &sum, xf);          // vec.h:1495
//!   float mean = sum / ne00;
//!   variance = ggml_vec_cvar_f32(ne00, yf, xf, mean);   // vec.cpp:455
//!   const float scale = 1.0f / sqrtf(variance + eps);
//!   ggml_vec_scale_f32(ne00, yf, scale);
//! ```
//!
//! * **the mean** is `ggml_vec_sum_f32`: a plain *sequential* `ggml_float`
//!   accumulator, narrowed to `float` on the way out, and then divided by
//!   `ne00` **in f32**. No tree anywhere. This is the same rule as RMSNorm's,
//!   which is why a blanket "accumulate in f64" was half right.
//! * **the variance** is `ggml_vec_cvar_f32`: on the reference build (AVX2+FMA,
//!   confirmed from the disassembly of `libggml-cpu.so`, not from
//!   `CMakeCache.txt`) a `ggml_float` accumulator fed **one f32 per group of 8**
//!   by a horizontal tree, then a scalar tail. So the variance is *not* a
//!   sequential f64 sum, and using one leaves 2-13% of rows wrong depending on
//!   the data.
//!
//! And the divisions are not symmetric, which is the detail that decides
//! everything: `variance` is declared `float`, so `ggml_vec_cvar_f32`'s double
//! return value is narrowed *before* `variance + eps` and `1.0f/sqrtf` see it,
//! while `sum` is narrowed inside `ggml_vec_sum_f32` and then divided in f32.
//! Doing either division in f64 changes the answer by an ulp on a real fraction
//! of rows.
//!
//! Measured by `engine/tools/probe_norm.c`, which runs a real
//! `ggml_norm -> ggml_mul -> ggml_add` graph on the CPU backend -- the exact op
//! sequence `llm_graph_context::build_norm` builds for GPT-2 and Falcon -- and
//! calls the reference's own exported `ggml_vec_cvar_f32`. Against 256 real
//! 2048-wide activation rows of qwen2.5-3b (the F32 widening in
//! `/home/reeds/wide/`), 524,288 output elements:
//!
//! ```text
//!                     mean wrong   var wrong   output bits differ
//!   sequential f32      247/256     246/256     324649/524288  (61.9%)
//!   f64 for both          0/256       6/256       2319/524288  ( 0.44%)
//!   cvar + f32 mean     247/256       6/256      67598/524288  (12.9%)
//!   this module           0/256       0/256          0/524288  ( 0.00%)
//! ```
//!
//! The middle row is the important one: it is what the old comment predicted,
//! and it is still wrong, because the two reductions inside one norm disagree
//! with each other. `engine/tests/norm_golden.rs` pins the whole function
//! against a checked-in table of the kernel's own bits, with the operands.
//!
//! One thing this is *not* claimed to be: portable across llama.cpp builds.
//! `ggml_vec_cvar_f32` has six branches (AVX-512 16-wide, AVX2+FMA 8-wide,
//! SSE2 4-wide, NEON, VXE, RISC-V) and they do not agree. This module
//! reproduces the one the loaded library compiled, exactly as `ggml_dot.rs`
//! does for the dot product, and the golden table is what would catch it if
//! that changed.

/// Elements per vector register in `ggml_vec_cvar_f32`'s AVX2 branch.
const CVAR_EPR: usize = 8;

/// `ggml_vec_sum_f32` (vec.h:1495): the reference's mean reduction.
///
/// ```text
///   ggml_float sum = 0.0;
///   for (int i = 0; i < n; ++i) sum += (ggml_float) x[i];
///   *s = (float) sum;
/// ```
///
/// A sequential double accumulator narrowed once on the way out. It is *not* a
/// tree and it is *not* `ggml_vec_dot_f32`, which is the natural thing to reach
/// for given that this is a sum of a vector with itself.
pub fn ggml_vec_sum_f32(x: &[f32]) -> f32 {
    let mut sum = 0f64;
    for v in x {
        sum += *v as f64;
    }
    sum as f32
}

/// `ggml_compute_forward_norm_f32`'s `mean`: `ggml_vec_sum_f32` then an f32
/// division.
///
/// The division is in f32 because `ggml_vec_sum_f32` already narrowed `sum` to
/// a `float` and ops.cpp:3728 writes `float mean = sum/ne00`. Narrowing after a
/// double division instead disagrees with the kernel on a real fraction of
/// rows, which is the entire difference between `f64_seq` and this module in
/// `probe_norm`'s table.
pub fn ggml_norm_mean_f32(x: &[f32]) -> f32 {
    ggml_vec_sum_f32(x) / x.len() as f32
}

/// `ggml_vec_cvar_f32` (vec.cpp:455) on the AVX2+FMA build, which also writes
/// `centered = x - mean` because the kernel writes `y` in the same loop it
/// reduces.
///
/// ```text
///   for (; i + 7 < n; i += 8) {
///       val = x[i..i+8] - mean;            // vsubps
///       y[i..i+8] = val;                   // vmovups
///       val = val * val;                   // vmulps
///       t[k] = val[k] + val[k+4];          // vextractf128 + vaddps
///       u[k] = t[k] + t[k+2];              // vmovhlps + vaddps
///       sum += (double)(u[0] + u[1]);       // vmovshdup + vaddss, vcvtss2sd
///   }
///   for (; i < n; ++i) { val = x[i] - mean; y[i] = val; sum += (double)(val*val); }
///   return sum / n;                        // in double, narrowed by the caller
/// ```
///
/// So the accumulator is a double, but it receives one f32 per eight elements
/// rather than one per element: the f32 rounding inside each group survives,
/// and only the eight partial sums are added exactly. A sequential f64
/// accumulation of the squares is a different number, not a more accurate one.
pub fn ggml_vec_cvar_f32(x: &[f32], mean: f32, centered: &mut [f32]) -> f32 {
    let n = x.len();
    debug_assert_eq!(centered.len(), n);
    let mut sum = 0f64;
    let mut i = 0;
    while i + CVAR_EPR <= n {
        // The 8 f32 squares, then the horizontal tree. Written out rather than
        // looped so the tree shape is visible: the reduction is (0+4,1+5,2+6,
        // 3+7) then (0+2, 1+3) then (0+1), and every one of those adds is f32.
        let mut sq = [0f32; CVAR_EPR];
        for k in 0..CVAR_EPR {
            let v = x[i + k] - mean;
            centered[i + k] = v;
            sq[k] = v * v;
        }
        let t0 = sq[0] + sq[4];
        let t1 = sq[1] + sq[5];
        let t2 = sq[2] + sq[6];
        let t3 = sq[3] + sq[7];
        sum += ((t0 + t2) + (t1 + t3)) as f64;
        i += CVAR_EPR;
    }
    while i < n {
        let v = x[i] - mean;
        centered[i] = v;
        sum += (v * v) as f64;
        i += 1;
    }
    // `float variance = ggml_vec_cvar_f32(...)` -- the division is in double
    // and the caller narrows, so the narrowing happens here.
    (sum / n as f64) as f32
}

/// `ggml_compute_forward_norm_f32` (ops.cpp:3698) with the weight and bias
/// that `llm_graph_context::build_norm` applies afterwards as `ggml_mul` and
/// `ggml_add` (llama-graph.cpp:1555).
///
/// The epilogue is `(x[i]-mean) * scale * w[i] + b[i]` in that order and no
/// fused multiply-add, because the reference runs three separate elementwise
/// ops over the row: the norm leaves `(x-mean)*scale` in `y`, `ggml_mul`
/// multiplies by the weight, `ggml_add` adds the bias.
pub fn ggml_layernorm(x: &[f32], w: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean = ggml_norm_mean_f32(x);
    // cvar writes the centered row, which is the norm's own `y` before scaling,
    // so it is computed once and scaled in place rather than re-subtracted.
    let mut centered = vec![0f32; n];
    let var = ggml_vec_cvar_f32(x, mean, &mut centered);
    let scale = 1.0f32 / (var + eps).sqrt();
    centered
        .iter_mut()
        .zip(w)
        .enumerate()
        .map(|(i, (v, wi))| *v * scale * wi + bias.get(i).copied().unwrap_or(0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cvar_writes_the_centered_row() {
        // The kernel writes y in the same loop it reduces, so a caller that
        // forgets to use it re-derives (x - mean) -- which is exact, so the
        // test that would catch a mistake here is the golden, not this one.
        // What this does pin is that the two halves agree on `mean`.
        let x = [1.0f32, -2.0, 3.5, 0.25, -7.0, 11.0, 0.5, -0.5, 2.0];
        let mean = ggml_norm_mean_f32(&x);
        let mut y = vec![0f32; x.len()];
        ggml_vec_cvar_f32(&x, mean, &mut y);
        for i in 0..x.len() {
            assert_eq!(y[i], x[i] - mean, "element {i}");
        }
    }

    #[test]
    fn the_mean_divides_in_f32_not_in_f64() {
        // `ggml_vec_sum_f32` narrows to a float and ops.cpp:3728 then divides in
        // f32, so `(f64_acc as f32) / (n as f32)` is the kernel and
        // `(f64_acc / n as f64) as f32` is not. Both halves are asserted: that
        // the module takes the first, and that on a length of 17 the two really
        // are different numbers.
        //
        // 17 is not a guess -- searching n = 1..4096 for the first disagreement
        // finds it there, and the bit pattern is pinned below so a future edit
        // to the data pattern cannot quietly turn this into a tautology.
        let x: Vec<f32> = (0..17)
            .map(|i| ((i * 7919 % 1000) as f32) * 0.0031 - 1.5)
            .collect();
        let acc: f64 = x.iter().map(|v| *v as f64).sum();
        let got = ggml_norm_mean_f32(&x);
        assert_eq!(got.to_bits(), (acc as f32 / 17.0f32).to_bits());
        assert_eq!(
            got.to_bits(),
            0x3e0d93bb,
            "the witness input's f32-division mean moved"
        );
        assert_ne!(
            got.to_bits(),
            ((acc / 17.0f64) as f32).to_bits(),
            "the f32 and f64 divisions of the mean agree on this input, so nothing here is \
             pinning which one the module uses"
        );
    }

    #[test]
    fn the_variance_divides_in_f64_and_trees_above_seven() {
        // Two claims that pull in opposite directions and both have to hold:
        //
        //  * at n <= 7 the kernel takes the scalar tail for every element, so
        //    its rule IS a sequential f64 accumulation of the squares -- and
        //    the division is in double, narrowed by the caller.
        //  * at n >= 8 the AVX2 branch takes over and the rule is *not* that:
        //    one f32 per group of 8 through a horizontal tree. Asserting the
        //    two differ is what makes "the variance uses the kernel's shape"
        //    a statement about the code rather than an aspiration.
        let x: Vec<f32> = (0..1024)
            .map(|i| ((i * 31 % 97) as f32) * 0.017 - 0.8)
            .collect();

        let seq_f64 = |v: &[f32]| -> f32 {
            let m = ggml_norm_mean_f32(v);
            let mut s = 0f64;
            for x in v {
                let d = *x - m;
                s += (d * d) as f64;
            }
            (s / v.len() as f64) as f32
        };

        let short = &x[..5];
        let m5 = ggml_norm_mean_f32(short);
        let mut y5 = vec![0f32; short.len()];
        assert_eq!(
            ggml_vec_cvar_f32(short, m5, &mut y5).to_bits(),
            seq_f64(short).to_bits(),
            "n=5 is all scalar tail, so the two must agree exactly"
        );

        let eight = &x[..8];
        let m8 = ggml_norm_mean_f32(eight);
        let mut y8 = vec![0f32; eight.len()];
        let got8 = ggml_vec_cvar_f32(eight, m8, &mut y8);
        assert_ne!(
            got8.to_bits(),
            seq_f64(eight).to_bits(),
            "n=8 is one full group, so the kernel's tree and a sequential f64 sum must \
             differ -- if they do not, the 8-wide branch is not being reproduced"
        );
    }
}
