// probe_f32_dot.c — whose f32 dot product is it?
//
// Every parity number this project reports on an F32 model bottoms out in one
// function: `ggml_vec_dot_f32`. The engine does not have one. It writes
//
//     row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>()
//
// which is a *sequential f32* accumulation, and that is a choice, not a
// necessity. The oracle rounds once. The two differ, and the difference is
// currently the floor of every engine-vs-llama.cpp comparison on a real model:
// engine(F32) vs llama.cpp(F32) is 2.1e-5..2.5e-4 on qwen2.5-3b, and
// tick 31 attributed it to "f32 summation order" without saying which order
// llama.cpp actually uses. Reading `ggml/src/ggml-cpu/vec.cpp:11` says the
// answer is not a single thing: the function has a scalar branch, an AVX/AVX2
// branch, an AVX-512 branch, an SVE branch and a RISC-V branch, and they do not
// agree with each other on anything. So "match llama.cpp" has no meaning until
// the *loaded* library's branch is known.
//
// This tool settles it by measurement rather than by reading a build flag, and
// it measures five accumulation schemes against the real kernel on real weight
// rows read out of a real GGUF:
//
//   ggml    : a real ggml_mul_mat graph on the CPU backend. This is
//             ggml_vec_dot_f32 verbatim -- the exact kernel path a decoder
//             runs, with the branch the loaded library actually compiled.
//   seq_f32 : `acc += a[i]*b[i]` in f32. What the engine does.
//   simd_tree: the AVX shape -- 4 accumulators of 8 lanes, FMA, a 3-level
//             pairwise tree, then the hadd horizontal -- emulated in f32.
//   scal_f64: the no-SIMD shape -- product rounded to f32, accumulated in
//             `double`, rounded once on the way out. Note the product is
//             still f32: `sumf += (ggml_float)(x[i]*y[i])` casts *after* the
//             multiply, so the f64 accumulator holds f32 products. An engine
//             that accumulated f64 *products* would be more accurate than
//             llama.cpp and still not equal to it.
//   scal_f32: product in f32, accumulator in f32, sequential -- the same as
//             seq_f32, kept as a separate name so a mismatch between the two
//             is visible rather than silent.
//
// `exact` (product and accumulator both f64) is the reference the other four
// are scored against, and is not claimed to be reachable.
//
// The bitwise column is the point of the tool. A scheme that is *closer* to
// the kernel is a better engine; a scheme that is *bit-identical* to the
// kernel is a different and stronger thing, because it means the residual is
// not a tolerance to be argued down but a difference that can be deleted. So
// the tool reports, per scheme, both rel L2 against exact and the number of
// output rows whose f32 bit pattern differs from the kernel's.
//
// What to expect, and why it is not obvious: an f64 accumulator reduces the
// *accumulation* error to nothing, but the kernel's own error is not zero
// either -- if the loaded library has AVX2+FMA, it accumulates in f32 lanes
// and no amount of f64 in the engine can reproduce it bit-for-bit, only get
// closer. The tool does not assume either answer; it prints the runtime
// feature set and lets the bitwise counts decide.
//
// Activation vector: relative error here is linear in x, but it is *not*
// scale-invariant in its data-dependence -- it scales with sum|a_i b_i| over
// |sum a_i b_i|, i.e. with cancellation. Real model activations cancel
// differently from gaussian noise, so `--act` picks the source and every line
// names it. `emb:<id>` is a real row of token_embd.weight: the actual input to
// block 0. It is only usable when ne0 <= n_embd, and the tool says so rather
// than padding.
//
// Usage:
//   probe_f32_dot <model.gguf> [--rows R] [--seed S] [--act SRC] [--max-ne0 N]
//                 [--max-tensors T] [--control]
//
//   SRC = emb:<id>     real row of token_embd.weight (default 0)
//       = gauss        N(0,1)
//       = heavy        a few enormous entries, as a real residual stream has
//       = uniform      U(-1,1)
//
//   --control    the whole apparatus on synthetic gaussian f32 data with no
//                model at all: a real ggml graph, a real f64 reference, the
//                same metric. If the control is not ~1e-7 the model numbers
//                are meaningless.

#define _POSIX_C_SOURCE 200809L

#include <ggml.h>
#include <gguf.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------ rng
//
// splitmix64, matching engine/tools/probe_quant_matvec.c and
// scripts/generate_synthetic_fixtures.py, so all three share one convention.

static uint64_t rng_state;

static uint64_t rng_next(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    return (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
}

static double rng_uniform(void) {
    return (double) (rng_next() >> 11) / (double) (1ULL << 53);
}

static double rng_normal(void) {
    double u1 = rng_uniform();
    if (u1 < 1e-300) {
        u1 = 1e-300;
    }
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * rng_uniform());
}

static double rng_heavy(void) {
    const double u = rng_uniform();
    const double mag = u < 1e-6 ? 400.0 : (u < 1e-3 ? 20.0 : 1.0);
    return mag * (2.0 * rng_uniform() - 1.0);
}

static void fill_activations(float * x, int64_t n, int dist) {
    for (int64_t i = 0; i < n; ++i) {
        const double v = dist == 0 ? rng_normal()
                      : dist == 1 ? 2.0 * rng_uniform() - 1.0
                                  : rng_heavy();
        x[i] = (float) v;
    }
}

static void die(const char * msg) {
    fprintf(stderr, "probe_f32_dot: %s\n", msg);
    exit(1);
}

// ------------------------------------------------------------------ schemes
//
// Each returns a float, as the kernel's `*s` does. The compiler flag matters
// and is set by the build: -ffp-contract=off, because a bare `acc + a*b` in
// f32 must stay two roundings. Rust at -O does the same thing (LLVM does not
// set the `contract` fast-math flag by default), which is why seq_f32 here is
// a faithful stand-in for the engine's line and not an approximation of it.
// fmaf() is an explicit libm call and is not contractible, so simd_tree gets
// the single-rounding FMA ggml's _mm256_fmadd_ps performs.

static float dot_seq_f32(const float * a, const float * b, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

// product f32 (rounded), accumulator f64, one rounding on the way out.
// Exactly `ggml_vec_dot_f32`'s no-SIMD branch.
static float dot_scal_f64(const float * a, const float * b, int64_t n) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        acc += (double) (a[i] * b[i]);
    }
    return (float) acc;
}

// The x86 SIMD branch, emulated lane by lane in f32.
//   GGML_F32_ARR = 4, GGML_F32_EPR = 8, GGML_F32_STEP = 32
//   sum[j] holds a 256-bit register; 4 of them accumulate 32 elements a step
//   GGML_F32x8_REDUCE folds 4 -> 2 -> 1 with f32 adds, then
//   t0 = lo(x0) + hi(x0);  res = (t0[0]+t0[1]) + (t0[2]+t0[3])
static float dot_simd_tree(const float * a, const float * b, int64_t n) {
    float sum[4][8];
    memset(sum, 0, sizeof(sum));

    const int64_t np = n & ~(int64_t) 31;
    for (int64_t i = 0; i < np; i += 32) {
        for (int j = 0; j < 4; ++j) {
            for (int l = 0; l < 8; ++l) {
                const int64_t k = i + 8 * j + l;
                sum[j][l] = fmaf(a[k], b[k], sum[j][l]);
            }
        }
    }
    // GGML_F32x8_REDUCE, verbatim: offset 2 folds 4 registers into 2, offset 1
    // folds 2 into 1, and the pair order matters -- x0 = (s0+s2) + (s1+s3).
    float tail = 0.0f;
    {
        for (int l = 0; l < 8; ++l) {
            sum[0][l] = sum[0][l] + sum[2][l];
        }
        for (int l = 0; l < 8; ++l) {
            sum[1][l] = sum[1][l] + sum[3][l];
        }
        for (int l = 0; l < 8; ++l) {
            sum[0][l] = sum[0][l] + sum[1][l];
        }
        const float t0[4] = { sum[0][0] + sum[0][4], sum[0][1] + sum[0][5],
                              sum[0][2] + sum[0][6], sum[0][3] + sum[0][7] };
        tail = (t0[0] + t0[1]) + (t0[2] + t0[3]);
    }
    // The leftover loop, which the loaded library's compiler rewrote into two
    // shapes (libggml-cpu.so, `ggml_vec_dot_f32` +0x140 and +0x17d):
    //
    //   tail <= 3          straight-line vfmadd231ss -- FMA, one term at a time
    //   tail >= 4          a 4-wide loop of vmulps + four vaddss, then the
    //                      remaining tail%4 terms as vfmadd231ss again
    //
    // So the tail is *not* a single rule, and a port that uses one gets the
    // boundary right and the interior wrong. This was measured rather than
    // read: the naive "fma chain" reproduces the kernel at ne0 = 2, 3, 33, 34,
    // 65 and misses at 4, 5, 8, 9, 40, 48, 70.
    if (n - np <= 3) {
        for (int64_t i = np; i < n; ++i) {
            tail = fmaf(a[i], b[i], tail);
        }
    } else {
        int64_t i = np;
        for (; i + 4 <= n; i += 4) {
            const float p0 = a[i + 0] * b[i + 0];
            const float p1 = a[i + 1] * b[i + 1];
            const float p2 = a[i + 2] * b[i + 2];
            const float p3 = a[i + 3] * b[i + 3];
            tail = tail + p0;
            tail = tail + p1;
            tail = tail + p2;
            tail = tail + p3;
        }
        for (; i < n; ++i) {
            tail = fmaf(a[i], b[i], tail);
        }
    }
    return tail;
}

// product and accumulator both f64. The reference; deliberately not reachable
// by llama.cpp and so not a parity target.
static float dot_exact(const float * a, const float * b, int64_t n) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        acc += (double) a[i] * (double) b[i];
    }
    return (float) acc;
}

// The cancellation ratio, which is what sets the size of every number here:
// relative error is ~ (accumulation error) * sum|a_i b_i| / |sum a_i b_i|.
// A dot product that nearly cancels has an enormous ratio and therefore a
// huge relative error while being perfectly accurate in absolute terms. It is
// reported so the rel L2 columns can be read rather than merely compared.
static double cancellation(const float * a, const float * b, int64_t n, double abs_sum) {
    double ref = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        ref += (double) a[i] * (double) b[i];
    }
    return ref != 0.0 ? fabs(abs_sum / ref) : INFINITY;
}

// ------------------------------------------------------------------ one probe

struct scheme {
    const char * name;
    float (*fn)(const float *, const float *, int64_t);
};

static const struct scheme SCHEMES[] = {
    { "seq_f32",   dot_seq_f32   },
    { "scal_f64",  dot_scal_f64  },
    { "simd_tree", dot_simd_tree },
    { "exact",     dot_exact     },
};
#define N_SCHEMES ((int) (sizeof(SCHEMES) / sizeof(SCHEMES[0])))

struct probe_result {
    double l2_rel[N_SCHEMES];      // vs `exact` (f64 product, f64 accumulator)
    int    bitdiff[N_SCHEMES];     // rows whose f32 bits differ from ggml's
    double cross_l2;               // seq_f32 vs scal_f64, i.e. the engine's move
    double vs_ggml[N_SCHEMES];     // ||scheme - ggml|| / ||ggml - exact||
    double cancel;                 // mean cancellation ratio
};

// Run one (weights, shape, activation) triple through a real ggml graph and
// through every scheme, and report.
static struct probe_result probe(ggml_backend_t backend, const float * wdata,
                                 int64_t ne0, int64_t nrows, const float * x) {
    struct ggml_init_params p = { /*mem_size =*/ 128u * 1024u * 1024u,
                                  /*mem_buffer =*/ NULL, /*no_alloc =*/ true };
    struct ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        die("ggml_init failed");
    }

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, nrows);
    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 1);
    struct ggml_tensor * out = ggml_mul_mat(ctx, a, t);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == NULL) {
        die("alloc failed");
    }
    ggml_backend_tensor_set(a, wdata, 0, (size_t) ne0 * (size_t) nrows * sizeof(float));
    ggml_backend_tensor_set(t, x, 0, (size_t) ne0 * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        die("ggml_backend_graph_compute failed");
    }
    float * got = (float *) malloc((size_t) nrows * sizeof(float));
    if (!got) {
        die("oom");
    }
    ggml_backend_tensor_get(out, got, 0, (size_t) nrows * sizeof(float));

    float * vals[N_SCHEMES];
    for (int s = 0; s < N_SCHEMES; ++s) {
        vals[s] = (float *) malloc((size_t) nrows * sizeof(float));
        if (!vals[s]) {
            die("oom");
        }
    }

    double ss_err[N_SCHEMES], ss_ref = 0.0, ss_ggml_ref = 0.0, sum_cancel = 0.0;
    for (int s = 0; s < N_SCHEMES; ++s) {
        ss_err[s] = 0.0;
    }
    for (int64_t r = 0; r < nrows; ++r) {
        const float * row = wdata + r * ne0;
        double abs_row = 0.0;
        for (int64_t i = 0; i < ne0; ++i) {
            abs_row += fabs((double) row[i] * (double) x[i]);
        }
        sum_cancel += cancellation(row, x, ne0, abs_row);
        // The reference has to be computed before the loop that scores against
        // it. Reading vals[N_SCHEMES-1][r] *inside* that loop reads
        // uninitialized memory on row 0 and the previous row's value on every
        // row after it, which makes every err_vs_exact print as 1.0 and every
        // dis_vs_ggml print as ~1e-25: the denominators are garbage, not the
        // schemes. It is the same class of bug as the one control_f32() in
        // probe_quant_matvec.c already documents, and it is invisible unless
        // the identity scheme is scored against itself.
        const double exact = (double) dot_exact(row, x, ne0);
        for (int s = 0; s < N_SCHEMES; ++s) {
            const float v = SCHEMES[s].fn(row, x, ne0);
            vals[s][r] = v;
            const double e = (double) v - exact;
            ss_err[s] += e * e;
        }
        ss_ref += exact * exact;
        const double e = (double) got[r] - exact;
        ss_ggml_ref += e * e;
    }

    // ||ggml - exact|| is the kernel's own distance from exact arithmetic. It
    // is the denominator for "how close is scheme s to the kernel" and it is
    // the value that makes the other column legible: a scheme at 1.0 is as far
    // from the kernel as the kernel is from exact.
    const double den = sqrt(ss_ggml_ref);

    struct probe_result res;
    for (int s = 0; s < N_SCHEMES; ++s) {
        res.l2_rel[s] = ss_ref > 0.0 ? sqrt(ss_err[s]) / sqrt(ss_ref) : 0.0;
        int n_diff = 0;
        for (int64_t r = 0; r < nrows; ++r) {
            if (memcmp(&vals[s][r], &got[r], sizeof(float)) != 0) {
                n_diff++;
            }
        }
        res.bitdiff[s] = n_diff;
        double ss = 0.0;
        for (int64_t r = 0; r < nrows; ++r) {
            const double e = (double) vals[s][r] - (double) got[r];
            ss += e * e;
        }
        res.vs_ggml[s] = den > 0.0 ? sqrt(ss) / den : 0.0;
    }
    double ss_cross = 0.0, ss_n = 0.0;
    for (int64_t r = 0; r < nrows; ++r) {
        const double e = (double) vals[0][r] - (double) vals[1][r];
        ss_cross += e * e;
        ss_n += (double) vals[1][r] * (double) vals[1][r];
    }
    res.cross_l2 = ss_n > 0.0 ? sqrt(ss_cross) / sqrt(ss_n) : 0.0;
    res.cancel = sum_cancel / (double) nrows;

    for (int s = 0; s < N_SCHEMES; ++s) {
        free(vals[s]);
    }
    free(got);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// ------------------------------------------------------------------ control
//
// A number measured against a reference is only as good as the reference. This
// runs the identical graph, the identical schemes and the identical metric on
// synthetic gaussian f32 vectors of the same length, with no model involved at
// all. What remains is ggml's own accumulation, the harness and the metric --
// so this is the floor of the apparatus, and every rel L2 printed above it is
// that floor plus the property being studied.
static double control(ggml_backend_t backend, int64_t ne0, int64_t nrows, int dist) {
    float * w = (float *) malloc((size_t) ne0 * (size_t) nrows * sizeof(float));
    float * x = (float *) malloc((size_t) ne0 * sizeof(float));
    if (!w || !x) {
        die("oom");
    }
    fill_activations(w, ne0 * nrows, dist);
    fill_activations(x, ne0, dist);
    struct probe_result r = probe(backend, w, ne0, nrows, x);
    free(w);
    free(x);
    if (r.l2_rel[N_SCHEMES - 1] != 0.0) {
        die("the `exact` scheme does not score 0 against itself: the reference "
            "is wrong, so nothing else printed is meaningful");
    }
    return r.l2_rel[0]; // seq_f32 vs exact, which is the claim under test
}

// ------------------------------------------------------------------ gguf I/O

// Read a contiguous block of `count` rows of F32 values out of a tensor,
// dequantizing through ggml's own converter when the tensor is not already
// F32. Only the sampled rows are read, which is what makes this usable on the
// 12.7 GB F32 widening of a 3B model.
static float * read_rows(FILE * f, struct gguf_context * gctx, int64_t ti,
                         int64_t row0, int64_t count) {
    const enum ggml_type type = gguf_get_tensor_type(gctx, ti);
    const int64_t ne0 = gguf_get_tensor_ne(gctx, ti)[0];
    const size_t packed_row = ggml_row_size(type, ne0);
    if (packed_row == 0) {
        die("zero row size");
    }
    const size_t off = gguf_get_data_offset(gctx)
                     + gguf_get_tensor_offset(gctx, ti) + (size_t) row0 * packed_row;
    uint8_t * raw = (uint8_t *) malloc(packed_row * (size_t) count);
    float * out = (float *) malloc((size_t) ne0 * (size_t) count * sizeof(float));
    if (!raw || !out) {
        die("oom");
    }
    if (fseek(f, (long) off, SEEK_SET) != 0
        || fread(raw, packed_row, (size_t) count, f) != (size_t) count) {
        die("short tensor read");
    }
    if (type == GGML_TYPE_F32) {
        memcpy(out, raw, (size_t) ne0 * (size_t) count * sizeof(float));
    } else {
        const struct ggml_type_traits * tt = ggml_get_type_traits(type);
        if (tt == NULL || tt->to_float == NULL) {
            die("tensor type has no dequantizer");
        }
        for (int64_t r = 0; r < count; ++r) {
            tt->to_float(raw + (size_t) r * packed_row, out + (size_t) r * ne0, ne0);
        }
    }
    free(raw);
    return out;
}

static int64_t find_tensor(struct gguf_context * gctx, const char * name) {
    const int64_t n = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n; ++i) {
        if (strcmp(gguf_get_tensor_name(gctx, i), name) == 0) {
            return i;
        }
    }
    return -1;
}

// ------------------------------------------------------------------ shape scan
//
// `simd_tree` is bit-identical to the kernel at ne0 = 96, 768, 2048 and 11008.
// It is *not* bit-identical on every row of gemma3, whose shapes are 48, 56 and
// 70 -- all of them have a tail of 16, 24 or 6 elements past the last full
// 32-element step. So the question is not "does the tree work" but "what is the
// tail supposed to do", and the only way to answer that is to walk ne0 one at a
// time and print where it breaks. 40 rows per length, 16 lengths per line.
static void shape_scan(ggml_backend_t backend, int64_t max_ne0, int64_t rows) {
    printf("\n# shape scan: simd_tree bitwise vs the kernel, ne0 = 1..%lld\n",
           (long long) max_ne0);
    printf("# %-5s %-7s %-9s %-9s %s\n", "ne0", "np", "tree_bad", "seq_bad", "f64_bad");
    for (int64_t ne0 = 1; ne0 <= max_ne0; ++ne0) {
        float * w = (float *) malloc((size_t) ne0 * (size_t) rows * sizeof(float));
        float * x = (float *) malloc((size_t) ne0 * sizeof(float));
        if (!w || !x) {
            die("oom");
        }
        fill_activations(w, ne0 * rows, 0);
        fill_activations(x, ne0, 0);
        struct probe_result r = probe(backend, w, ne0, rows, x);
        free(w);
        free(x);
        if (r.bitdiff[2] != 0 || ne0 <= 4 || ne0 % 8 == 1) {
            printf("  %-5lld np=%-5lld %-9d %-9d %d\n", (long long) ne0,
                   (long long) (ne0 & ~(int64_t) 31), r.bitdiff[2], r.bitdiff[0],
                   r.bitdiff[1]);
        }
    }
}

// ------------------------------------------------------------------ golden
//
// Self-consistency is not parity. This emits, for a set of lengths spanning
// both leftover shapes, the real kernel's f32 bit pattern for a fixed set of
// inputs, so `engine/tests/dot_golden.rs` can demand that the engine's
// `ggml_dot_f32` reproduces the reference bit for bit. The inputs travel with
// the results, so the test does not have to re-derive a PRNG and cannot
// silently agree with the generator about one.
//
// Format (little-endian), mirroring tests/data/quant_golden.bin:
//   "SLDOTGLD" u32 version=1  u32 ncases
//   per case: u32 ne0, f32 a[ne0], f32 b[ne0], f32 result
static void emit_golden(ggml_backend_t backend, int64_t max_ne0, uint32_t seed,
                        const char * path) {
    // (ne0, rows) pairs. Both leftover shapes are covered (tail 1..3 by the
    // fma path, tail >= 4 by the block path) and every length a real model
    // actually uses is in here: 768 and 2048 for qwen2.5-3b, 4096 and 11008
    // for a 7B-class ffn. Row counts fall with length so the checked-in table
    // stays the same size as tests/data/quant_golden.bin rather than growing
    // to megabytes for a few extra samples of a 11008-wide matvec.
    static const struct { int32_t ne0; uint32_t rows; } cases[] = {
        { 1, 8 }, { 2, 8 }, { 3, 8 }, { 4, 8 }, { 5, 8 }, { 7, 8 },
        { 8, 8 }, { 9, 8 }, { 15, 8 }, { 16, 8 }, { 17, 8 }, { 31, 8 },
        { 32, 8 }, { 33, 8 }, { 34, 8 }, { 40, 8 }, { 48, 8 }, { 56, 8 },
        { 63, 8 }, { 64, 8 }, { 65, 8 }, { 70, 8 }, { 96, 8 }, { 100, 8 },
        { 128, 8 }, { 160, 8 }, { 200, 8 }, { 256, 8 }, { 384, 3 },
        { 512, 3 }, { 768, 3 }, { 1024, 3 }, { 2048, 3 }, { 4096, 2 },
        { 11008, 2 },
    };
    const size_t n_cases = sizeof(cases) / sizeof(cases[0]);

    FILE * out = fopen(path, "wb");
    if (!out) {
        die("cannot open golden for writing");
    }
    // The header carries the number of *dot products*, not the number of
    // length groups. An earlier version wrote the group count and the Rust
    // reader stopped after that many cases, leaving ~90% of the file unread
    // and reporting it as "trailing bytes" -- which is at least a loud failure,
    // but the reader now walks to the end of the buffer instead.
    uint32_t total_cases = 0;
    for (size_t c = 0; c < n_cases; ++c) {
        total_cases += cases[c].rows;
    }
    fwrite("SLDOTGLD", 1, 8, out);
    const uint32_t version = 1;
    fwrite(&version, 4, 1, out);
    fwrite(&total_cases, 4, 1, out);

    // A separate stream per case, so adding a length to the table above does
    // not perturb the vectors of the cases before it.
    for (size_t c = 0; c < n_cases; ++c) {
        const int64_t ne0 = cases[c].ne0 < max_ne0 ? cases[c].ne0 : max_ne0;
        const uint32_t rows = cases[c].rows;
        rng_state = seed + 0x9E3779B9u * (uint32_t) c;

        float * w = (float *) malloc((size_t) ne0 * rows * sizeof(float));
        float * x = (float *) malloc((size_t) ne0 * sizeof(float));
        if (!w || !x) {
            die("oom");
        }
        fill_activations(w, ne0 * rows, 0);
        fill_activations(x, ne0, 0);

        struct ggml_init_params p = { /*mem_size =*/ 256u * 1024u * 1024u,
                                      /*mem_buffer =*/ NULL, /*no_alloc =*/ true };
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, rows);
        struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 1);
        struct ggml_tensor * O = ggml_mul_mat(ctx, A, X);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, O);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            die("golden alloc failed");
        }
        ggml_backend_tensor_set(A, w, 0, (size_t) ne0 * rows * sizeof(float));
        ggml_backend_tensor_set(X, x, 0, (size_t) ne0 * sizeof(float));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            die("golden compute failed");
        }
        float * got = (float *) malloc((size_t) rows * sizeof(float));
        if (!got) {
            die("oom");
        }
        ggml_backend_tensor_get(O, got, 0, (size_t) rows * sizeof(float));

        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t len = (uint32_t) ne0;
            fwrite(&len, 4, 1, out);
            fwrite(w + (size_t) r * ne0, sizeof(float), (size_t) ne0, out);
            fwrite(x, sizeof(float), (size_t) ne0, out);
            fwrite(&got[r], sizeof(float), 1, out);
        }

        free(got);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        free(w);
        free(x);
    }
    fclose(out);
    uint32_t printed = 0;
    for (size_t c = 0; c < n_cases; ++c) {
        printed += cases[c].rows;
    }
    printf("# wrote %s: %zu lengths, %u dot products, seed=%u\n",
           path, n_cases, printed, seed);
}

// ------------------------------------------------------------------ main

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [--rows R] [--seed S] [--act SRC] "
                "[--max-ne0 N] [--max-tensors T] [--control] [--shape-scan N]\n"
                "       [--emit-golden PATH [--golden-seed S]]\n", argv[0]);
        return 2;
    }
    int64_t want_rows = 64;
    uint64_t seed = 42;
    int64_t max_ne0 = 1 << 16;
    int64_t max_tensors = 12;
    bool want_control = false;
    int64_t scan_max = 0;
    const char * golden_path = NULL;
    uint32_t golden_seed = 20260930u;
    const char * act_src = "emb:0";
    int act_dist = -1;
    long act_id = 0;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            want_rows = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--act") == 0 && i + 1 < argc) {
            act_src = argv[++i];
        } else if (strcmp(argv[i], "--max-ne0") == 0 && i + 1 < argc) {
            max_ne0 = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--max-tensors") == 0 && i + 1 < argc) {
            max_tensors = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--control") == 0) {
            want_control = true;
        } else if (strcmp(argv[i], "--shape-scan") == 0 && i + 1 < argc) {
            scan_max = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--emit-golden") == 0 && i + 1 < argc) {
            golden_path = argv[++i];
        } else if (strcmp(argv[i], "--golden-seed") == 0 && i + 1 < argc) {
            golden_seed = (uint32_t) strtoul(argv[++i], NULL, 10);
        } else {
            fprintf(stderr, "unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    rng_state = seed;

    if (strncmp(act_src, "emb:", 4) == 0) {
        act_id = strtol(act_src + 4, NULL, 10);
    } else if (strcmp(act_src, "gauss") == 0) {
        act_dist = 0;
    } else if (strcmp(act_src, "uniform") == 0) {
        act_dist = 1;
    } else if (strcmp(act_src, "heavy") == 0) {
        act_dist = 2;
    } else {
        fprintf(stderr, "--act must be emb:<id>, gauss, uniform or heavy\n");
        return 2;
    }

    FILE * f = fopen(argv[1], "rb");
    if (!f) {
        die("cannot open gguf");
    }
    struct gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ NULL };
    struct gguf_context * gctx = gguf_init_from_file(argv[1], gp);
    if (!gctx) {
        die("gguf_init_from_file failed");
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        die("ggml_backend_cpu_init failed");
    }

    printf("# probe_f32_dot %s\n", argv[1]);
    printf("# seed=%llu rows_per_tensor=%lld act=%s\n",
           (unsigned long long) seed, (long long) want_rows, act_src);
    printf("# runtime CPU features as the LOADED ggml sees them:\n");
    printf("#   avx=%d avx2=%d avx512=%d fma=%d f16c=%d\n",
           ggml_cpu_has_avx(), ggml_cpu_has_avx2(), ggml_cpu_has_avx512(),
           ggml_cpu_has_fma(), ggml_cpu_has_f16c());
    // The branch of ggml_vec_dot_f32 the loaded library compiled is chosen by
    // GGML_SIMD, a *compile-time* macro, not by the runtime features above.
    // So the features only predict it; the bitwise column below decides it.
    printf("#   => ggml_vec_dot_f32 %s\n",
           ggml_cpu_has_avx512() ? "predicted: AVX-512 branch (f32 lanes, FMA)"
           : ggml_cpu_has_avx2() || ggml_cpu_has_avx() ? "predicted: AVX/AVX2 branch (f32 lanes, FMA)"
           : "predicted: scalar branch (f32 product, double accumulator)");

    if (golden_path) {
        emit_golden(backend, 1 << 20, golden_seed, golden_path);
        ggml_backend_free(backend);
        gguf_free(gctx);
        fclose(f);
        return 0;
    }

    if (scan_max > 0) {
        shape_scan(backend, scan_max, 40);
        ggml_backend_free(backend);
        gguf_free(gctx);
        fclose(f);
        return 0;
    }

    if (want_control) {
        printf("# control (synthetic gaussian, ne0=2048, 256 rows, no model): "
               "seq_f32 err_vs_exact=%.3e\n", control(backend, 2048, 256, 0));
    }

    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    const int64_t emb = find_tensor(gctx, "token_embd.weight");
    const int64_t n_embd = emb >= 0 ? gguf_get_tensor_ne(gctx, emb)[0] : 0;

    int64_t n_shown = 0;
    double agg_l2[N_SCHEMES] = { 0 };
    double agg_vg[N_SCHEMES] = { 0 };
    int64_t agg_rows = 0;
    int64_t agg_bit[N_SCHEMES] = { 0 };
    long agg_n = 0;
    double agg_cross = 0.0, agg_cross_max = 0.0;
    double agg_cancel = 0.0;

    printf("\n%-22s %-5s %-5s %-6s %-6s ", "tensor", "ne0", "rows", "type", "act");
    for (int s = 0; s < N_SCHEMES; ++s) {
        printf("%-19s ", SCHEMES[s].name);
    }
    printf("%-6s\n", "cancel");
    printf("# per scheme, two numbers that must not be confused:\n");
    printf("#   err = ||scheme - exact|| / ||exact||        how accurate the scheme is\n");
    printf("#   dis = ||scheme - ggml|| / ||ggml - exact||  how far it is from the KERNEL\n");
    printf("#         dis = 0.000 and 0/N means bit-identical to ggml, which is the only\n");
    printf("#         thing here worth acting on\n");

    for (int64_t ti = 0; ti < n_tensors && n_shown < max_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(gctx, ti);
        const int64_t ne0 = gguf_get_tensor_ne(gctx, ti)[0];
        if (ne0 <= 0 || ne0 > max_ne0) {
            continue;
        }
        const enum ggml_type type = gguf_get_tensor_type(gctx, ti);
        if (type != GGML_TYPE_F32 && !ggml_is_quantized(type)) {
            continue;
        }
        const size_t tsize = gguf_get_tensor_size(gctx, ti);
        const size_t packed_row = ggml_row_size(type, ne0);
        if (packed_row == 0 || tsize % packed_row != 0) {
            continue;
        }
        const int64_t nrows = (int64_t) (tsize / packed_row);
        if (nrows < 2) {
            continue;
        }
        const int64_t rows = want_rows < nrows ? want_rows : nrows;

        // activation
        float * x = (float *) malloc((size_t) ne0 * sizeof(float));
        if (!x) {
            die("oom");
        }
        const char * act_label;
        if (act_dist >= 0) {
            fill_activations(x, ne0, act_dist);
            act_label = act_src;
        } else if (emb >= 0 && ne0 <= n_embd) {
            float * e = read_rows(f, gctx, emb, act_id, 1);
            if (ne0 == n_embd) {
                memcpy(x, e, (size_t) ne0 * sizeof(float));
            } else {
                // A prefix of a real embedding row. Not the activation this
                // matrix actually sees, and the label says so.
                memcpy(x, e, (size_t) ne0 * sizeof(float));
            }
            free(e);
            static char lbl[64];
            snprintf(lbl, sizeof(lbl), "emb:%ld%s", act_id, ne0 == n_embd ? "" : "/pre");
            act_label = lbl;
        } else {
            fill_activations(x, ne0, 0);
            act_label = "gauss*";
        }

        float * w = read_rows(f, gctx, ti, 0, rows);
        struct probe_result r = probe(backend, w, ne0, rows, x);
        free(w);
        free(x);

        printf("%-22s %-5lld %-5lld %-6s %-6s ", name, (long long) ne0,
               (long long) rows, ggml_type_name(type), act_label);
        for (int s = 0; s < N_SCHEMES; ++s) {
            printf("%.2e/%-2.0e %2d/%-3lld ", r.l2_rel[s], r.vs_ggml[s],
                   r.bitdiff[s], (long long) rows);
        }
        printf("%.0f\n", r.cancel);

        for (int s = 0; s < N_SCHEMES; ++s) {
            agg_l2[s] += r.l2_rel[s];
            agg_vg[s] += r.vs_ggml[s];
            agg_bit[s] += r.bitdiff[s];
        }
        agg_rows += rows;
        agg_n++;
        if (r.cross_l2 > agg_cross_max) {
            agg_cross_max = r.cross_l2;
        }
        agg_cross += r.cross_l2;
        agg_cancel += r.cancel;
        n_shown++;
    }

    if (agg_n > 0) {
        printf("\nSUMMARY tensors=%ld rows=%lld mean_cancel=%.1f\n",
               agg_n, (long long) agg_rows, agg_cancel / (double) agg_n);
        for (int s = 0; s < N_SCHEMES; ++s) {
            printf("  %-10s err_vs_exact=%.3e  dis_vs_ggml=%.3e  bitdiff=%lld/%lld%s\n",
                   SCHEMES[s].name, agg_l2[s] / (double) agg_n,
                   agg_vg[s] / (double) agg_n,
                   (long long) agg_bit[s], (long long) agg_rows,
                   agg_bit[s] == 0 ? "   <-- BIT-IDENTICAL TO THE KERNEL" : "");
        }
        printf("  engine's move seq_f32 -> scal_f64: mean rel L2 %.3e, worst %.3e\n",
               agg_cross / (double) agg_n, agg_cross_max);
        printf("  self-check: `exact` scored against itself = %.3e (must be 0)\n",
               agg_l2[N_SCHEMES - 1] / (double) agg_n);
        if (agg_l2[N_SCHEMES - 1] != 0.0) {
            fprintf(stderr, "probe_f32_dot: SELF-CHECK FAILED -- the reference is "
                            "wrong and nothing above is meaningful\n");
            return 1;
        }
    }

    ggml_backend_free(backend);
    gguf_free(gctx);
    fclose(f);
    return 0;
}
