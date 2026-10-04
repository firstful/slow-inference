// probe_norm.c -- whose LayerNorm reduction is it?
//
// Every other f32 reduction in this project has been settled by measurement
// rather than by reading a build flag: `probe_f32_dot.c` did it for
// `ggml_vec_dot_f32` (24 call sites, 46.7x on a real 36-layer model) and the
// same file settled `ggml_compute_forward_rms_norm_f32`. The two survivors were
// the mean and the variance in `Model::layernorm`, on GPT-2 and Falcon, and the
// note above them in `src/model.rs` claimed that the reference reduces them in
// `double` "because `ggml_vec_norm_f32` accumulates into a `ggml_float`". That
// claim is false at llama.cpp 3653e6d, in two ways:
//
//   * `ggml_vec_norm_f32` (ggml/src/ggml-cpu/vec.h:858) is now one line --
//     `ggml_vec_dot_f32(n, s, 0, x, 0, x, 0, 1); *s = sqrtf(*s);` -- so it is
//     the *dot* kernel, not a double accumulator. It is also not what LayerNorm
//     calls.
//   * `ggml_compute_forward_norm_f32` (ggml/src/ggml-cpu/ops.cpp:3698), the op
//     a decoder actually runs, calls two different functions with two
//     different shapes: `ggml_vec_sum_f32` for the mean (vec.h:1495 -- a plain
//     sequential `ggml_float` accumulator, no tree) and `ggml_vec_cvar_f32`
//     for the variance (vec.cpp:455 -- a `ggml_float` accumulator fed ONE f32
//     per group of 8 by a horizontal tree, i.e. the opposite of the mean's
//     rule).
//
// So the two reductions disagree with each other inside one function, and a
// blanket "accumulate in f64" is right for the mean and wrong for the
// variance. Which shape the variance takes is a compile-time choice among
// AVX-512 / AVX2+FMA / SSE2 / NEON / VXE / RISC-V, and -- as with the dot
// product -- `CMakeCache.txt` cannot be used to pick it. So this tool measures.
//
// What it measures, per input vector:
//
//   ggml      : a real `ggml_norm -> ggml_mul -> ggml_add` graph on the CPU
//               backend, which is `llm_graph_context::build_norm` verbatim
//               (src/llama-graph.cpp:1555) -- the exact op sequence GPT-2 and
//               Falcon run.
//   sum       : `ggml_sum` in the same graph, which for a single row is
//               `ggml_vec_sum_f32` (both are "accumulate in double, narrow
//               once") without needing a header-private symbol.
//   var       : `ggml_vec_cvar_f32` called directly. It is an exported symbol
//               in `libggml-cpu.so` (`nm -D` shows `T ggml_vec_cvar_f32`), so
//               this is the reference's own code and not a re-implementation.
//   decomp_ok : the decomposition check. Given the kernel's own `sum` and its
//               own `var`, `(x[i]-mean)*scale*w[i]+b[i]` must reproduce the
//               graph's output bit for bit. If it does, the two numbers this
//               tool writes into the golden file are demonstrably the kernel's
//               intermediate values and not a plausible-looking story about
//               them. `--emit-golden` refuses to write if it fails.
//
// And then the four candidates the engine might have used, scored bitwise:
//
//   seq_f32   : mean and variance both accumulated sequentially in f32. This is
//               what `Model::layernorm` does today.
//   f64_seq   : both accumulated sequentially in double. The obvious reading of
//               the old comment. Right for the mean, wrong for the variance.
//   cvar_f32  : the mean sequential in f32, the variance in the kernel's
//               shape (f64 accumulator, one f32 per group of 8 from a
//               horizontal tree, scalar tail).
//   cvar_f64  : mean sequential in double, variance in the kernel's shape.
//               The candidate that should be bit-exact.
//
// The bitwise column is the point, for the same reason it was for the dot
// product: a scheme that is *closer* to the kernel is a better engine, a
// scheme that is *bit-identical* is a difference that can be deleted. A
// relative-error column cannot tell those apart, and on these vectors the two
// are within an ulp of each other.
//
// Usage:
//   probe_norm <model.gguf> [--act SRC] [--seed S] [--eps E] [--max-ne0 N]
//             [--shape-scan N] [--emit-golden PATH [--golden-seed S]]
//
//   SRC = emb:<id>   real row of token_embd.weight (default: gauss; `heavy` is
//                    the closest synthetic stand-in for a residual stream)
//       = gauss      N(0,1)
//       = uniform    U(-1,1)
//       = heavy      a few enormous entries
//
// -ffp-contract=off is load-bearing and is set by the build script: the golden
// records the *kernel's* bits, and the tool's own `seq_f32` column is a
// stand-in for the engine's `sum::<f32>()`, which does not contract. The
// kernel's cvar has no FMA, so nothing here needs fmaf().

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

// The reference's own variance reduction, called rather than reimplemented.
// Declared in ggml/src/ggml-cpu/vec.h:76 and exported from libggml-cpu.so.
// ggml_float is double on every platform this builds on.
extern double ggml_vec_cvar_f32(const int n, float * y, const float * x, const float mean);

// ------------------------------------------------------------------ rng
//
// splitmix64, matching engine/tools/probe_f32_dot.c,
// engine/tools/probe_quant_matvec.c and scripts/generate_synthetic_fixtures.py,
// so all four share one convention.

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
    fprintf(stderr, "probe_norm: %s\n", msg);
    exit(1);
}

// ------------------------------------------------------------------ schemes
//
// Each of the four fills `mean` and `var` the way the engine's `layernorm`
// would, then produces the epilogue `(x[i]-mean) * scale * w[i] + b[i]` in the
// order `build_norm` does it: `ggml_norm` (which ends in
// `ggml_vec_scale_f32`), then `ggml_mul`, then `ggml_add`.

typedef void (*scheme_fn)(const float *, int64_t, float, const float *, const float *,
                         float *, float *, float *);

// mean: sequential f32. var: sequential f32. What src/model.rs does today.
static void norm_seq_f32(const float * x, int64_t n, float eps,
                         const float * w, const float * b, float * y,
                         float * mean_out, float * var_out) {
    float mean = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        mean += x[i];
    }
    mean /= (float) n;
    float var = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i] - mean;
        var += v * v;
    }
    var /= (float) n;
    const float scale = 1.0f / sqrtf(var + eps);
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (x[i] - mean) * scale * w[i] + b[i];
    }
    *mean_out = mean;
    *var_out = var;
}

// The variance in `ggml_vec_cvar_f32`'s shape on an AVX2+FMA build: for each
// group of 8, `val = x[i..i+8] - mean` in f32, `val *= val` in f32, then a
// horizontal tree of f32 adds (lo+hi, then 0+2, then 1+3), then ONE f32 widened
// into a double accumulator. Scalar tail, also into the double accumulator.
// The division is in double and narrowed once, because the caller holds the
// result in a `float` (`float variance = ggml_vec_cvar_f32(...)`, ops.cpp:3730).
static float var_cvar_shape(const float * x, int64_t n, float mean) {
    double sum = 0.0;
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        float sq[8];
        for (int k = 0; k < 8; ++k) {
            const float v = x[i + k] - mean;
            sq[k] = v * v;
        }
        // vextractf128 + vaddps, then vmovhlps + vaddps, then vmovshdup + vaddss
        const float t0 = sq[0] + sq[4];
        const float t1 = sq[1] + sq[5];
        const float t2 = sq[2] + sq[6];
        const float t3 = sq[3] + sq[7];
        const float u0 = t0 + t2;
        const float u1 = t1 + t3;
        sum += (double) (u0 + u1);
    }
    for (; i < n; ++i) {
        const float v = x[i] - mean;
        sum += (double) (v * v);
    }
    return (float) (sum / (double) n);
}

// mean: sequential f64 (that IS `ggml_vec_sum_f32`). var: sequential f64.
static void norm_f64_seq(const float * x, int64_t n, float eps,
                         const float * w, const float * b, float * y,
                         float * mean_out, float * var_out) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += (double) x[i];
    }
    // `ggml_vec_sum_f32` narrows to float on the way out (`*s = (float)sum`)
    // and ops.cpp:3728 then divides in FLOAT (`float mean = sum/ne00`). Doing
    // the division in double and narrowing once is a different answer, and it
    // is the entire difference between the two f64 rows in the table above.
    const float mean = (float) sum / (float) n;
    double vsum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i] - mean;
        vsum += (double) (v * v);
    }
    const float var = (float) (vsum / (double) n);
    const float scale = 1.0f / sqrtf(var + eps);
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (x[i] - mean) * scale * w[i] + b[i];
    }
    *mean_out = mean;
    *var_out = var;
}

// mean sequential in f32, variance in the kernel's shape.
static void norm_cvar_f32(const float * x, int64_t n, float eps,
                          const float * w, const float * b, float * y,
                          float * mean_out, float * var_out) {
    float mean = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        mean += x[i];
    }
    mean /= (float) n;
    const float var = var_cvar_shape(x, n, mean);
    const float scale = 1.0f / sqrtf(var + eps);
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (x[i] - mean) * scale * w[i] + b[i];
    }
    *mean_out = mean;
    *var_out = var;
}

// mean sequential in f64, variance in the kernel's shape. The candidate.
static void norm_cvar_f64(const float * x, int64_t n, float eps,
                          const float * w, const float * b, float * y,
                          float * mean_out, float * var_out) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += (double) x[i];
    }
    // `ggml_vec_sum_f32` narrows to float on the way out (`*s = (float)sum`)
    // and ops.cpp:3728 then divides in FLOAT (`float mean = sum/ne00`). Doing
    // the division in double and narrowing once is a different answer, and it
    // is the entire difference between the two f64 rows in the table above.
    const float mean = (float) sum / (float) n;
    const float var = var_cvar_shape(x, n, mean);
    const float scale = 1.0f / sqrtf(var + eps);
    for (int64_t i = 0; i < n; ++i) {
        y[i] = (x[i] - mean) * scale * w[i] + b[i];
    }
    *mean_out = mean;
    *var_out = var;
}

struct scheme {
    const char * name;
    scheme_fn fn;
};

static const struct scheme SCHEMES[] = {
    { "seq_f32",  norm_seq_f32  },  // what the engine does today
    { "f64_seq",  norm_f64_seq  },  // the obvious reading of the old comment
    { "cvar_f32", norm_cvar_f32 },
    { "cvar_f64", norm_cvar_f64 },  // the candidate
};
// A preprocessor constant rather than a `static const size_t`, because C makes
// a file-scope sizeof-derived length a VLA in an automatic array, which cannot
// carry an initializer.
#define N_SCHEMES 4
_Static_assert(sizeof(SCHEMES) / sizeof(SCHEMES[0]) == N_SCHEMES,
               "N_SCHEMES must match the SCHEMES table");

// ------------------------------------------------------------------ kernel

typedef struct { int32_t ne0; uint32_t rows; } len_spec_t;

// One length's worth of real kernel output.
struct kernel_case {
    int64_t ne0;
    uint32_t rows;
    float * x;   // ne0 * rows
    float * w;   // ne0
    float * b;   // ne0
    float * y;   // ne0 * rows  the graph's output
    float * sum; // rows        ggml_sum of each row
    float * var; // rows        ggml_vec_cvar_f32 at the kernel's mean
    int64_t decomp_bad; // (x[i]-mean)*scale*w[i]+b[i] != y[i], summed over rows
};

// A real graph per length: ggml_norm -> ggml_mul -> ggml_add, plus ggml_sum on
// the input. eps lives in the NORM op_params, so it is fixed per graph and is
// threaded through the schemes rather than the other way round.
static struct kernel_case * run_kernel(ggml_backend_t backend, const len_spec_t * spec,
                                       size_t n_spec, int64_t max_ne0, uint32_t seed,
                                       int dist, float eps, int64_t * total_bad) {
    struct kernel_case * cases = calloc(n_spec, sizeof(struct kernel_case));
    if (!cases) {
        die("oom");
    }
    int64_t bad_total = 0;
    for (size_t c = 0; c < n_spec; ++c) {
        const int64_t ne0 = spec[c].ne0 < max_ne0 ? spec[c].ne0 : max_ne0;
        const uint32_t rows = spec[c].rows;
        rng_state = seed + 0x9E3779B9u * (uint32_t) c;

        float * x = (float *) malloc((size_t) ne0 * rows * sizeof(float));
        float * w = (float *) malloc((size_t) ne0 * sizeof(float));
        float * b = (float *) malloc((size_t) ne0 * sizeof(float));
        if (!x || !w || !b) {
            die("oom");
        }
        fill_activations(x, ne0 * rows, dist);
        fill_activations(w, ne0, dist);
        fill_activations(b, ne0, dist);

        struct ggml_init_params p = { /*mem_size =*/ 256u * 1024u * 1024u,
                                      /*mem_buffer =*/ NULL, /*no_alloc =*/ true };
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, rows);
        struct ggml_tensor * W = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        struct ggml_tensor * B = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        struct ggml_tensor * N = ggml_norm(ctx, X, eps);
        struct ggml_tensor * M = ggml_mul(ctx, N, W);
        struct ggml_tensor * Y = ggml_add(ctx, M, B);
        // One ggml_sum per row, over a 1-D view. `ggml_sum` over the whole 2-D
        // tensor would give a single scalar for all rows -- it writes
        // ((float *) dst->data)[0] = sum and nothing else, ops.cpp:1309 -- and
        // reading `rows` floats out of a scalar is what an earlier draft of
        // this tool did, until ggml asserted on the out-of-bounds read. The
        // reduction a view of one row sees is `ggml_vec_sum_f32` over ne00
        // elements, which is exactly what the norm kernel computes for that
        // row, so this is the kernel's mean and not a second opinion on it.
        struct ggml_tensor ** S = (struct ggml_tensor **) malloc(rows * sizeof(*S));
        if (!S) {
            die("oom");
        }
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, Y);
        for (uint32_t r = 0; r < rows; ++r) {
            S[r] = ggml_sum(ctx, ggml_view_1d(ctx, X, ne0, (size_t) r * (size_t) ne0 * sizeof(float)));
            ggml_build_forward_expand(gf, S[r]);
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            die("alloc failed");
        }
        ggml_backend_tensor_set(X, x, 0, (size_t) ne0 * rows * sizeof(float));
        ggml_backend_tensor_set(W, w, 0, (size_t) ne0 * sizeof(float));
        ggml_backend_tensor_set(B, b, 0, (size_t) ne0 * sizeof(float));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            die("graph compute failed");
        }

        float * y = (float *) malloc((size_t) ne0 * rows * sizeof(float));
        float * sum = (float *) malloc((size_t) rows * sizeof(float));
        float * var = (float *) malloc((size_t) rows * sizeof(float));
        float * scratch = (float *) malloc((size_t) ne0 * sizeof(float));
        if (!y || !sum || !var || !scratch) {
            die("oom");
        }
        ggml_backend_tensor_get(Y, y, 0, (size_t) ne0 * rows * sizeof(float));
        for (uint32_t r = 0; r < rows; ++r) {
            ggml_backend_tensor_get(S[r], &sum[r], 0, sizeof(float));
        }

        int64_t case_bad = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            const float * xr = x + (size_t) r * ne0;
            const float * yr = y + (size_t) r * ne0;
            const float mean = sum[r] / (float) ne0;
            const float v = (float) ggml_vec_cvar_f32((int) ne0, scratch, xr, mean);
            var[r] = v;
            const float scale = 1.0f / sqrtf(v + eps);
            for (int64_t i = 0; i < ne0; ++i) {
                const float got = (xr[i] - mean) * scale * w[i] + b[i];
                if (memcmp(&got, &yr[i], sizeof(float)) != 0) {
                    case_bad++;
                }
            }
        }
        bad_total += case_bad;

        cases[c].ne0 = ne0;
        cases[c].rows = rows;
        cases[c].x = x;
        cases[c].w = w;
        cases[c].b = b;
        cases[c].y = y;
        cases[c].sum = sum;
        cases[c].var = var;
        cases[c].decomp_bad = case_bad;

        free(scratch);
        free(S);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    *total_bad = bad_total;
    return cases;
}

static void free_kernel(struct kernel_case * cases, size_t n) {
    for (size_t c = 0; c < n; ++c) {
        free(cases[c].x);
        free(cases[c].w);
        free(cases[c].b);
        free(cases[c].y);
        free(cases[c].sum);
        free(cases[c].var);
    }
    free(cases);
}

// ------------------------------------------------------------------ specs
//
// Lengths chosen for the same reason the dot table's were: cover every tail
// shape of the kernel's 8-wide loop (ne0 mod 8 in 0..7 and a few of each), then
// every n_embd a real LayerNorm model uses. GPT-2 family: 768, 1024, 1600.
// Falcon: 4544 and 3072. BERT-base: 768. n_embd_gqa is irrelevant here --
// LayerNorm reduces the full width, so it sees n_embd.

static const len_spec_t SPEC[] = {
    { 1, 4 },  { 2, 4 },  { 3, 4 },  { 4, 4 },   { 5, 4 },   { 6, 4 },
    { 7, 4 },  { 8, 4 },  { 9, 4 },  { 15, 4 },  { 16, 4 },  { 17, 4 },
    { 23, 4 }, { 24, 4 }, { 31, 4 }, { 32, 4 },  { 33, 4 },  { 40, 4 },
    { 48, 4 }, { 63, 4 }, { 64, 4 }, { 65, 4 },  { 70, 4 },  { 96, 4 },
    { 128, 4 },{ 160, 4 },{ 256, 4 },{ 384, 4 }, { 768, 4 }, { 1024, 4 },
    { 1600, 3 }, { 2048, 3 }, { 4544, 2 }, { 3072, 2 },
};
static const size_t N_SPEC = sizeof(SPEC) / sizeof(SPEC[0]);

// ------------------------------------------------------------------ report

static void report(struct kernel_case * cases, size_t n, int64_t total_bad, float eps) {
    uint64_t y_bits[N_SCHEMES] = { 0 };
    uint64_t mean_bits[N_SCHEMES] = { 0 };
    uint64_t var_bits[N_SCHEMES] = { 0 };
    uint64_t y_elem = 0;
    double y_maxabs[N_SCHEMES] = { 0 };
    double y_sumabs = 0;
    uint64_t rows = 0;

    float * y = (float *) malloc((size_t) 1 << 22);
    if (!y) {
        die("oom");
    }
    for (size_t c = 0; c < n; ++c) {
        const int64_t ne0 = cases[c].ne0;
        for (uint32_t r = 0; r < cases[c].rows; ++r) {
            const float * xr = cases[c].x + (size_t) r * ne0;
            const float * yr = cases[c].y + (size_t) r * ne0;
            const float kmean = cases[c].sum[r] / (float) ne0;
            const float kvar = cases[c].var[r];
            for (size_t s = 0; s < N_SCHEMES; ++s) {
                float m, v;
                SCHEMES[s].fn(xr, ne0, eps, cases[c].w, cases[c].b, y, &m, &v);
                if (memcmp(&m, &kmean, sizeof(float)) != 0) {
                    mean_bits[s]++;
                }
                if (memcmp(&v, &kvar, sizeof(float)) != 0) {
                    var_bits[s]++;
                }
                uint64_t bad = 0;
                double maxabs = 0;
                for (int64_t i = 0; i < ne0; ++i) {
                    if (memcmp(&y[i], &yr[i], sizeof(float)) != 0) {
                        bad++;
                        const double d = fabs((double) y[i] - (double) yr[i]);
                        if (d > maxabs) {
                            maxabs = d;
                        }
                    }
                    y_sumabs += fabs((double) yr[i]);
                }
                y_bits[s] += bad;
                y_maxabs[s] = fmax(y_maxabs[s], maxabs);
            }
            y_elem += (uint64_t) ne0;
            rows++;
        }
    }
    free(y);

    printf("\n%zu lengths, %llu rows, eps=%g\n", n, (unsigned long long) rows, (double) eps);
    printf("decomposition check  (x-mean)*scale*w+b  ==  graph output:  %lld element(s) differ\n",
           (long long) total_bad);
    if (total_bad != 0) {
        printf("  ^ this must be 0, or the mean and variance below are a story about the\n"
               "    kernel rather than the kernel's own values\n");
    }
    printf("\n%-9s %12s %12s %16s %14s %14s\n",
           "scheme", "mean wrong", "var wrong", "y bits differ", "y max|err|", "y rel L2");
    for (size_t s = 0; s < N_SCHEMES; ++s) {
        const double rel = y_sumabs > 0
            ? y_maxabs[s] * sqrt((double) y_elem) / y_sumabs
            : 0.0;
        printf("%-9s %6llu/%-5llu %6llu/%-5llu %9llu/%-6llu %14.3e %14.3e\n",
               SCHEMES[s].name,
               (unsigned long long) mean_bits[s], (unsigned long long) rows,
               (unsigned long long) var_bits[s],  (unsigned long long) rows,
               (unsigned long long) y_bits[s],    (unsigned long long) y_elem,
               y_maxabs[s], rel);
    }
    printf("\ncounts are rows out of %llu, and output elements out of %llu;\n"
           " y max|err| is absolute and the outputs are O(1) after weight and bias,\n"
           " so one f32 ulp here is about 1.2e-7 and no row is a tolerance question.\n",
           (unsigned long long) rows, (unsigned long long) y_elem);
}

// Per-length detail, so a table that is bit-exact in aggregate cannot be
// bit-exact by luck at one length and wrong at another.
static void shape_scan(ggml_backend_t backend, int64_t max_ne0, uint32_t seed,
                       int64_t scan_max, float eps) {
    len_spec_t spec[512];
    size_t n = 0;
    for (int64_t k = 1; k <= scan_max && n < 512; ++k) {
        spec[n].ne0 = (int32_t) k;
        spec[n].rows = 1;
        n++;
    }
    int64_t bad = 0;
    struct kernel_case * cases = run_kernel(backend, spec, n, max_ne0, seed, 0, eps, &bad);
    printf("# shape scan ne0 = 1..%lld, one row each, eps=%g\n", (long long) scan_max, (double) eps);
    printf("# %-6s %-22s", "ne0", "mean/var (s=seq_f32 c=cvar_f64)");
    for (size_t s = 0; s < N_SCHEMES; ++s) {
        printf(" %9s", SCHEMES[s].name);
    }
    printf("   [y bits differing]\n");
    uint64_t tot[N_SCHEMES] = { 0 };
    for (size_t c = 0; c < n; ++c) {
        const int64_t ne0 = cases[c].ne0;
        const float * xr = cases[c].x;
        const float kmean = cases[c].sum[0] / (float) ne0;
        const float kvar = cases[c].var[0];
        float * y = (float *) malloc((size_t) ne0 * sizeof(float));
        if (!y) {
            die("oom");
        }
        printf("%-6lld", (long long) ne0);
        for (size_t s = 0; s < N_SCHEMES; ++s) {
            float m, v;
            SCHEMES[s].fn(xr, ne0, eps, cases[c].w, cases[c].b, y, &m, &v);
            const int mbad = memcmp(&m, &kmean, sizeof(float)) != 0;
            const int vbad = memcmp(&v, &kvar, sizeof(float)) != 0;
            if (s == 0 || s == 3) {
                printf("  %c%c", mbad ? 'M' : '.', vbad ? 'V' : '.');
            }
            uint64_t bad_y = 0;
            for (int64_t i = 0; i < ne0; ++i) {
                if (memcmp(&y[i], &cases[c].y[i], sizeof(float)) != 0) {
                    bad_y++;
                }
            }
            tot[s] += bad_y;
            printf(" %9llu", (unsigned long long) bad_y);
        }
        printf("\n");
        free(y);
    }
    printf("# total y bits differing over ne0=1..%lld:\n", (long long) scan_max);
    for (size_t s = 0; s < N_SCHEMES; ++s) {
        printf("#   %-9s %llu\n", SCHEMES[s].name, (unsigned long long) tot[s]);
    }
    free_kernel(cases, n);
    (void) bad;
}

// ------------------------------------------------------------------ golden
//
//   "SLNRMGLD" u32 version=1  u32 ncases
//   per case: u32 ne0, f32 x[ne0], f32 w[ne0], f32 b[ne0],
//             f32 mean, f32 var, f32 y[ne0]
//
// x, w, b and y travel with the two reduction results, so the Rust reader never
// has to re-derive the PRNG and the test cannot silently agree with the
// generator about one. eps is fixed per file and written in the header, since
// the NORM op carries it in op_params and a reader that guessed it wrong would
// see every output differ by a scale factor.
static void emit_golden(ggml_backend_t backend, int64_t max_ne0, uint32_t seed, float eps,
                        const char * path) {
    int64_t bad = 0;
    struct kernel_case * cases = run_kernel(backend, SPEC, N_SPEC, max_ne0, seed, 0, eps, &bad);
    if (bad != 0) {
        die("decomposition check failed -- refusing to write a golden whose mean/variance \
             are not demonstrably the kernel's");
    }

    FILE * out = fopen(path, "wb");
    if (!out) {
        die("cannot open golden for writing");
    }
    uint32_t total_cases = 0;
    for (size_t c = 0; c < N_SPEC; ++c) {
        total_cases += cases[c].rows;
    }
    fwrite("SLNRMGLD", 1, 8, out);
    const uint32_t version = 1;
    fwrite(&version, 4, 1, out);
    fwrite(&total_cases, 4, 1, out);
    fwrite(&eps, sizeof(float), 1, out);

    uint64_t bytes = 20;
    for (size_t c = 0; c < N_SPEC; ++c) {
        const int64_t ne0 = cases[c].ne0;
        for (uint32_t r = 0; r < cases[c].rows; ++r) {
            const uint32_t len = (uint32_t) ne0;
            fwrite(&len, 4, 1, out);
            fwrite(cases[c].x + (size_t) r * ne0, sizeof(float), (size_t) ne0, out);
            fwrite(cases[c].w, sizeof(float), (size_t) ne0, out);
            fwrite(cases[c].b, sizeof(float), (size_t) ne0, out);
            const float mean = cases[c].sum[r] / (float) ne0;
            fwrite(&mean, sizeof(float), 1, out);
            fwrite(&cases[c].var[r], sizeof(float), 1, out);
            fwrite(cases[c].y + (size_t) r * ne0, sizeof(float), (size_t) ne0, out);
            // x, w, b and y are four f32 arrays of ne0; len, mean and var are three u32.
            bytes += 4 + (uint64_t) ne0 * 4 * 4 + 8;
        }
    }
    fclose(out);

    printf("# wrote %s: %zu lengths, %u cases, %llu bytes, seed=%u, eps=%g\n",
           path, N_SPEC, total_cases, (unsigned long long) bytes, seed, (double) eps);
    printf("# decomposition check: 0 element(s) differ, so mean and var in the file are\n"
           "#   the kernel's own values and the y column is its own output.\n");
    free_kernel(cases, N_SPEC);
}

// ------------------------------------------------------------------ activations

// A real row of token_embd.weight, which is the input to block 0 and therefore
// to the first LayerNorm in a real forward pass. `emb:<id>` is the only source
// here that is a real activation rather than a plausible synthetic one. On a
// quantized file the row is dequantized through ggml's own `to_float`, the same
// converter `probe_quant_matvec.c` uses, and the tool says which happened --
// because dequantized Q4_K_M weights are real numbers from a real model but are
// not what a residual stream looks like, and claiming otherwise would make this
// a worse experiment than it is.
static bool load_emb_row(FILE * f, struct gguf_context * gctx, long id, int64_t max_ne0,
                         float ** out, int64_t * out_ne0, int64_t rows_out) {
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    for (int64_t t = 0; t < n_tensors; ++t) {
        const char * name = gguf_get_tensor_name(gctx, t);
        if (!name || strcmp(name, "token_embd.weight") != 0) {
            continue;
        }
        const enum ggml_type type = gguf_get_tensor_type(gctx, t);
        const int64_t ne0 = gguf_get_tensor_ne(gctx, t)[0];
        const int64_t nelem = ne0 * gguf_get_tensor_ne(gctx, t)[1];
        if (ne0 > max_ne0) {
            printf("# token_embd.weight has ne0=%lld > --max-ne0 %lld\n",
                   (long long) ne0, (long long) max_ne0);
            return false;
        }
        if (id >= nelem) {
            printf("# token id %ld is past the %lld rows of token_embd\n", id, (long long) nelem);
            return false;
        }
        // Read `rows_out` consecutive rows so the mean and the variance see a
        // real distribution of real rows rather than one row repeated.
        int64_t rows = rows_out;
        if (rows > nelem - id) {
            rows = nelem - id;
        }
        const size_t packed_row = ggml_row_size(type, ne0);
        if (packed_row == 0) {
            die("zero row size");
        }
        uint8_t * raw = (uint8_t *) malloc(packed_row * (size_t) rows);
        float * out_rows = (float *) malloc((size_t) ne0 * (size_t) rows * sizeof(float));
        if (!raw || !out_rows) {
            die("oom");
        }
        const size_t off = gguf_get_data_offset(gctx)
                         + gguf_get_tensor_offset(gctx, t) + (size_t) id * packed_row;
        if (fseek(f, (long) off, SEEK_SET) != 0
            || fread(raw, packed_row, (size_t) rows, f) != (size_t) rows) {
            die("short tensor read");
        }
        if (type == GGML_TYPE_F32) {
            memcpy(out_rows, raw, (size_t) ne0 * (size_t) rows * sizeof(float));
        } else {
            const struct ggml_type_traits * tt = ggml_get_type_traits(type);
            if (tt == NULL || tt->to_float == NULL) {
                die("tensor type has no dequantizer");
            }
            for (int64_t r = 0; r < rows; ++r) {
                tt->to_float(raw + (size_t) r * packed_row,
                             out_rows + (size_t) r * ne0, ne0);
            }
            printf("# token_embd.weight is %s: rows dequantized through ggml's to_float.\n"
                   "#   These are real numbers from a real model but they are embedding\n"
                   "#   WEIGHTS, not a residual stream; the heavy-tail synthetic below is\n"
                   "#   the closer stand-in for what a LayerNorm actually sees.\n",
                   ggml_type_name(type));
        }
        free(raw);
        *out = out_rows;
        *out_ne0 = ne0;
        return true;
    }
    printf("# no token_embd.weight in this file\n");
    return false;
}

// ------------------------------------------------------------------ main

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [--act SRC] [--rows R] [--seed S] [--eps E]\n"
                "       [--max-ne0 N] [--shape-scan N]\n"
                "       [--emit-golden PATH [--golden-seed S]]\n", argv[0]);
        return 2;
    }
    uint64_t seed = 42;
    int64_t max_ne0 = 1 << 16;
    float eps = 1e-5f;
    int64_t scan_max = 0;
    const char * golden_path = NULL;
    uint32_t golden_seed = 20260930u;
    const char * act_src = "heavy";
    int64_t emb_rows = 64;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--eps") == 0 && i + 1 < argc) {
            eps = (float) atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-ne0") == 0 && i + 1 < argc) {
            max_ne0 = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--shape-scan") == 0 && i + 1 < argc) {
            scan_max = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--emit-golden") == 0 && i + 1 < argc) {
            golden_path = argv[++i];
        } else if (strcmp(argv[i], "--golden-seed") == 0 && i + 1 < argc) {
            golden_seed = (uint32_t) strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--act") == 0 && i + 1 < argc) {
            act_src = argv[++i];
        } else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            emb_rows = atoll(argv[++i]);
        } else {
            fprintf(stderr, "unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    if (strcmp(act_src, "gauss") != 0 && strcmp(act_src, "uniform") != 0 &&
        strcmp(act_src, "heavy") != 0 && strncmp(act_src, "emb:", 4) != 0) {
        fprintf(stderr, "--act must be emb:<id>, gauss, uniform or heavy\n");
        return 2;
    }
    rng_state = seed;

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

    printf("# probe_norm %s\n", argv[1]);
    printf("# seed=%llu act=%s eps=%g\n", (unsigned long long) seed, act_src, (double) eps);
    printf("# llama.cpp 3653e6d, LayerNorm = ggml_compute_forward_norm_f32 (ops.cpp:3698)\n");
    printf("#   mean = ggml_vec_sum_f32  (vec.h:1495) -- sequential ggml_float, no tree\n");
    printf("#   var  = ggml_vec_cvar_f32 (vec.cpp:455) -- ggml_float accumulator fed ONE f32\n");
    printf("#                                       per group of 8 by a horizontal tree\n");
    printf("# runtime CPU features as the LOADED ggml sees them:\n");
    printf("#   avx=%d avx2=%d avx512=%d fma=%d\n",
           ggml_cpu_has_avx(), ggml_cpu_has_avx2(), ggml_cpu_has_avx512(),
           ggml_cpu_has_fma());
    printf("#   => cvar branch predicted: %s\n",
           ggml_cpu_has_avx512() ? "AVX-512 (16-wide)"
           : ggml_cpu_has_avx2() ? "AVX2+FMA (8-wide)"
           : ggml_cpu_has_avx()   ? "SSE2/AVX (4-wide)"
                                  : "scalar tail only");
    printf("#   that only predicts it. The bitwise column decides.\n");

    if (golden_path) {
        emit_golden(backend, max_ne0, golden_seed, eps, golden_path);
        ggml_backend_free(backend);
        gguf_free(gctx);
        fclose(f);
        return 0;
    }
    if (scan_max > 0) {
        shape_scan(backend, max_ne0, (uint32_t) seed, scan_max, eps);
        ggml_backend_free(backend);
        gguf_free(gctx);
        fclose(f);
        return 0;
    }

    float * emb = NULL;
    int64_t emb_ne0 = 0;
    if (strncmp(act_src, "emb:", 4) == 0 &&
        !load_emb_row(f, gctx, strtol(act_src + 4, NULL, 10), max_ne0, &emb, &emb_ne0,
                      emb_rows)) {
        printf("# falling back to the heavy-tail synthetic distribution\n");
    }

    if (emb) {
        // Only the length this model actually has: padding a shorter vector out
        // to ne0 would change the reduction, and truncating a longer one would
        // change which rows the kernel reads. So one length, 64 real rows.
        len_spec_t one[1] = { { .ne0 = (int32_t) emb_ne0, .rows = (uint32_t) emb_rows } };
        int64_t bad = 0;
        struct kernel_case * cases =
            run_kernel(backend, one, 1, max_ne0, (uint32_t) seed, 0, eps, &bad);
        printf("\n## real activation: %lld rows of token_embd.weight (ne0=%lld) -- the input to\n"
               "## block 0's LayerNorm. w and b are synthetic, since what is being measured\n"
               "## is the reduction and not the weights.\n",
               (long long) emb_rows, (long long) emb_ne0);
        report(cases, 1, bad, eps);
        free_kernel(cases, 1);
        free(emb);
    } else {
        // All three distributions, because the shape of the answer depends on
        // the data: a residual stream is not gaussian, and a reduction that
        // agrees on gaussian noise can disagree where the entries are large.
        const char * names[3] = { "gauss", "uniform", "heavy" };
        const int dists[3] = { 0, 1, 2 };
        for (int di = 0; di < 3; ++di) {
            if (strcmp(act_src, names[di]) != 0) {
                continue;
            }
            int64_t bad = 0;
            struct kernel_case * cases =
                run_kernel(backend, SPEC, N_SPEC, max_ne0, (uint32_t) seed, dists[di], eps, &bad);
            printf("\n## synthetic activation: %s\n", names[di]);
            report(cases, N_SPEC, bad, eps);
            free_kernel(cases, N_SPEC);
        }
    }

    ggml_backend_free(backend);
    gguf_free(gctx);
    fclose(f);
    return 0;
}