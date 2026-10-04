// probe_quant_matvec.c — how wrong is ggml's own quantized mat-mul?
//
// Every parity claim about a quantized model reduces to one question this
// project never measured: when ggml multiplies a quantized weight matrix by an
// f32 activation vector, how far is the answer from exact arithmetic?
//
// The answer is not "it accumulates in f16" — the hypothesis ticks 15-18
// carried, and it is wrong. ggml's integer dot products accumulate in int32 and
// are exact. The loss comes from somewhere else: **the activations are
// themselves quantized**. Before the dot product, ggml rounds the activation
// vector to `vec_dot_type` (Q8_K for Q4_K/Q5_K/Q6_K, Q8_0 for Q4_0, ...) with
// a per-block scale. A dot product against the *rounded* activation is a
// different number from one against the exact activation, and no order of
// accumulation recovers the difference.
//
// This tool measures that error with ggml as both sides of the comparison:
//
//   ggml  : a real `ggml_mul_mat` graph node on the CPU backend, with the
//           weight bytes read verbatim out of a GGUF. This is the exact kernel
//           path a decoder runs — activation quantization, SIMD kernels,
//           reduction order and all. No internals are reimplemented.
//   exact : type_traits->to_float (the dequantizer, already proven bit-exact
//           against itself by engine/tests/quant_golden.rs), then accumulate the
//           dot product in double. The only approximation on this side is f32
//           rounding in the weights, ~6e-8.
//
// The gap between the two is the noise floor of *any* engine-vs-llama.cpp diff
// on a quantized model, and it does not depend on the Rust engine being right
// about anything.
//
// Activation distribution matters: the rounding error of the activation
// quantizer is set by the dynamic range inside each block, so a gaussian vector
// and a heavy-tailed one with a few 400-sigma outliers are different
// experiments. `--dists` runs all three.
//
// Usage:
//   probe_quant_matvec <model.gguf> [--rows R] [--seed S] [--dists] [--max-ne0 N]
//   probe_quant_matvec --dtype     # only report the model's type histogram

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
// splitmix64, matching scripts/generate_synthetic_fixtures.py, so a fixture and
// this probe share one seed convention.

static uint64_t rng_state;

static uint64_t rng_next(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
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
    // Most entries small, a few enormous: the shape of a residual stream with
    // outlier features, and the shape that punishes a per-block activation
    // quantizer hardest.
    const double u = rng_uniform();
    const double mag = u < 1e-6 ? 400.0 : (u < 1e-3 ? 20.0 : 1.0);
    return mag * (2.0 * rng_uniform() - 1.0);
}

static void fill_activations(float * x, int64_t n, int dist) {
    // The relative error below is scale-invariant — both sides are linear in x
    // — so unit scale loses nothing.
    for (int64_t i = 0; i < n; ++i) {
        const double v = dist == 0 ? rng_normal()
                      : dist == 1 ? 2.0 * rng_uniform() - 1.0
                                  : rng_heavy();
        x[i] = (float) v;
    }
}

static const char * dist_name(int d) {
    switch (d) {
        case 0:  return "gaussian";
        case 1:  return "uniform";
        case 2:  return "heavy_tail";
        default: return "?";
    }
}

static void die(const char * msg) {
    fprintf(stderr, "probe_quant_matvec: %s\n", msg);
    exit(1);
}

// ------------------------------------------------------------------ one probe

struct probe_result {
    double l2_rel;     // ||ggml - exact||_2 / ||exact||_2 over the output vector
    double max_rel;    // max |err| / rms(exact)
    double rms_ref;    // rms of the exact output, for scale
};

// Run one (weights, shape, activation distribution) triple through ggml and
// through exact f64 math, and report the disagreement.
static struct probe_result probe(ggml_backend_t backend, enum ggml_type type,
                                 const uint8_t * wdata, int64_t ne0, int64_t nrows,
                                 int dist) {
    struct ggml_init_params p = { /*mem_size =*/ 128u * 1024u * 1024u,
                                  /*mem_buffer =*/ NULL, /*no_alloc =*/ true };
    struct ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        die("ggml_init failed");
    }

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, type, ne0, nrows);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 1);
    struct ggml_tensor * out = ggml_mul_mat(ctx, a, x);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == NULL) {
        fprintf(stderr, "alloc failed for %s ne0=%lld nrows=%lld\n",
                ggml_type_name(type), (long long) ne0, (long long) nrows);
        exit(1);
    }
    ggml_backend_tensor_set(a, wdata, 0, ggml_nbytes(a));

    float * xf = (float *) malloc((size_t) ne0 * sizeof(float));
    if (!xf) {
        die("oom");
    }
    fill_activations(xf, ne0, dist);
    ggml_backend_tensor_set(x, xf, 0, (size_t) ne0 * sizeof(float));

    const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        die("ggml_backend_graph_compute failed");
    }
    float * got = (float *) malloc((size_t) nrows * sizeof(float));
    if (!got) {
        die("oom");
    }
    ggml_backend_tensor_get(out, got, 0, (size_t) nrows * sizeof(float));

    // exact: ggml's own dequantizer, accumulated in double
    const struct ggml_type_traits * tt = ggml_get_type_traits(type);
    const size_t packed_row = ggml_row_size(type, ne0);
    float * wf = (float *) malloc((size_t) ne0 * sizeof(float));
    if (!wf || tt == NULL || tt->to_float == NULL) {
        die("type has no dequantizer");
    }
    double * ref = (double *) malloc(sizeof(double) * (size_t) nrows);
    if (!ref) {
        die("oom");
    }
    for (int64_t r = 0; r < nrows; ++r) {
        tt->to_float(wdata + (size_t) r * packed_row, wf, ne0);
        double acc = 0.0;
        for (int64_t i = 0; i < ne0; ++i) {
            acc += (double) wf[i] * (double) xf[i];
        }
        ref[r] = acc;
    }

    // Metrics, and why these and not per-row relative error.
    //
    // A per-row |err| / |ref| is the obvious thing to print and it is
    // useless: a row whose dot product nearly cancels has ref ~ 0 and its
    // relative error is unbounded while being irrelevant. A 27% "error" on
    // such a row says nothing about the mat-vec, and it is the first thing this
    // tool printed before the metrics were fixed.
    //
    // What matters downstream is the error of the output *vector*, because that
    // is what gets added into a residual stream and softmaxed. So:
    //
    //   l2_rel  = ||ggml - exact||_2 / ||exact||_2     the number to propagate
    //   max_rel = max|err| / rms(exact)                worst single output element
    double ss_err = 0.0, ss_ref = 0.0;
    double worst = 0.0;
    for (int64_t r = 0; r < nrows; ++r) {
        const double e = (double) got[r] - ref[r];
        ss_err += e * e;
        ss_ref += ref[r] * ref[r];
    }
    const double rms_ref = sqrt(ss_ref / (double) nrows);
    for (int64_t r = 0; r < nrows; ++r) {
        const double e = fabs((double) got[r] - ref[r]);
        if (rms_ref > 1e-30 && e / rms_ref > worst) {
            worst = e / rms_ref;
        }
    }
    struct probe_result res = {
        sqrt(ss_err) / (ss_ref > 0.0 ? sqrt(ss_ref) : 1.0),   // l2_rel
        worst,                                                 // max_rel
        rms_ref,
    };
    free(ref);

    free(wf);
    free(got);
    free(xf);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// ------------------------------------------------------------------ control
//
// A measurement that reports "ggml's quantized mat-mul is off by 1e-2" is only
// worth anything if the harness itself is exact to 1e-7. This runs the identical
// graph, with the identical activation vector, on an F32 copy of the *same*
// dequantized weights: there is no activation quantization left, so this is the
// floor of the whole apparatus — the ggml graph executor, the buffer transfers,
// the f64 reference, the metric. If the control is not ~1e-6, every number this
// tool prints is meaningless.
static double control_f32(ggml_backend_t backend, enum ggml_type type,
                          const uint8_t * wdata, int64_t ne0, int64_t nrows, int dist) {
    const struct ggml_type_traits * tt = ggml_get_type_traits(type);
    const size_t packed_row = ggml_row_size(type, ne0);

    struct ggml_init_params p = { /*mem_size =*/ 256u * 1024u * 1024u,
                                  /*mem_buffer =*/ NULL, /*no_alloc =*/ true };
    struct ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        die("ggml_init failed (control)");
    }
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, nrows);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, 1);
    struct ggml_tensor * out = ggml_mul_mat(ctx, a, x);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == NULL) {
        die("control alloc failed");
    }

    float * wrow = (float *) malloc((size_t) ne0 * sizeof(float));
    float * xf   = (float *) malloc((size_t) ne0 * sizeof(float));
    if (!wrow || !xf) {
        die("oom");
    }
    fill_activations(xf, ne0, dist);
    // The reference has to be built in the same pass that fills the tensor:
    // `wrow` is a single-row buffer, so accumulating afterwards would silently
    // compare every output element against the *last* row's dot product. That
    // bug shipped in the first version of this control and showed up as
    // l2_rel = 8.6, which is exactly the kind of absurd number that gets filed
    // under "the quantized path is terrible" instead of "the tool is wrong".
    double * ref = (double *) malloc(sizeof(double) * (size_t) nrows);
    if (!ref) {
        die("oom");
    }
    for (int64_t r = 0; r < nrows; ++r) {
        tt->to_float(wdata + (size_t) r * packed_row, wrow, ne0);
        double acc = 0.0;
        for (int64_t i = 0; i < ne0; ++i) {
            acc += (double) wrow[i] * (double) xf[i];
        }
        ref[r] = acc;
        ggml_backend_tensor_set(a, wrow, (size_t) r * ne0 * sizeof(float),
                                (size_t) ne0 * sizeof(float));
    }
    ggml_backend_tensor_set(x, xf, 0, (size_t) ne0 * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        die("control compute failed");
    }
    float * got = (float *) malloc((size_t) nrows * sizeof(float));
    if (!got) {
        die("oom");
    }
    ggml_backend_tensor_get(out, got, 0, (size_t) nrows * sizeof(float));

    double ss_err = 0.0, ss_ref = 0.0;
    for (int64_t r = 0; r < nrows; ++r) {
        const double e = (double) got[r] - ref[r];
        ss_err += e * e;
        ss_ref += ref[r] * ref[r];
    }
    free(ref); free(wrow); free(xf); free(got);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ss_ref > 0.0 ? sqrt(ss_err) / sqrt(ss_ref) : 0.0;
}

// ------------------------------------------------------------------ main

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--rows R] [--seed S] [--dists] [--max-ne0 N]\n",
                argv[0]);
        return 2;
    }
    int64_t want_rows = 32;
    uint64_t seed = 42;
    bool all_dists = false;
    int64_t max_ne0 = 1 << 20;
    bool want_control = false;
    int controls_run = 0;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            want_rows = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--dists") == 0) {
            all_dists = true;
        } else if (strcmp(argv[i], "--max-ne0") == 0 && i + 1 < argc) {
            max_ne0 = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--control") == 0) {
            want_control = true;
        } else {
            fprintf(stderr, "unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    rng_state = seed;

    FILE * f = fopen(argv[1], "rb");
    if (!f) {
        die("cannot open gguf");
    }
    struct gguf_init_params gp = { /*no_alloc =*/ false, /*ctx =*/ NULL };
    struct gguf_context * gctx = gguf_init_from_file(argv[1], gp);
    if (!gctx) {
        die("gguf_init_from_file failed");
    }
    const size_t data_off = gguf_get_data_offset(gctx);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        die("ggml_backend_cpu_init failed");
    }

    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    printf("# probe_quant_matvec %s\n", argv[1]);
    printf("# seed=%llu rows_per_tensor=%lld dists=%s  (rel error of ggml's own\n",
           (unsigned long long) seed, (long long) want_rows, all_dists ? "all" : "gaussian");
    printf("#  quantized mul_mat against exact f64 math on the same weights)\n");

    // Aggregate over every tensor sharing (type, ne0, nrows): that is one
    // "mat-mul shape", and the summary should read as a table of shapes.
    struct shape_agg {
        enum ggml_type type;
        int64_t ne0;
        int64_t nrows;
        double worst;
        double sum_rms;
        double max_rms;
        int dist;
        long n;
    } * aggs = NULL;
    int n_aggs = 0;

    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * name = gguf_get_tensor_name(gctx, ti);
        const enum ggml_type type = gguf_get_tensor_type(gctx, ti);
        if (!ggml_is_quantized(type)) {
            continue;
        }
        const struct ggml_type_traits * tt = ggml_get_type_traits(type);
        if (tt == NULL || tt->to_float == NULL) {
            continue;
        }
        const int64_t ne0 = gguf_get_tensor_ne(gctx, ti)[0];
        if (ne0 <= 0 || ne0 > max_ne0) {
            continue;
        }
        const size_t tsize = gguf_get_tensor_size(gctx, ti);
        const size_t packed_row = ggml_row_size(type, ne0);
        if (packed_row == 0 || tsize % packed_row != 0) {
            continue;
        }
        const int64_t nrows = (int64_t) (tsize / packed_row);

        uint8_t * data = (uint8_t *) malloc(tsize);
        if (!data) {
            die("oom");
        }
        if (fseek(f, (long) (data_off + gguf_get_tensor_offset(gctx, ti)), SEEK_SET) != 0
            || fread(data, 1, tsize, f) != tsize) {
            die("short tensor read");
        }

        const int n_dists = all_dists ? 3 : 1;
        for (int d = 0; d < n_dists; ++d) {
            const int64_t rows = want_rows < nrows ? want_rows : nrows;
            const struct probe_result r =
                probe(backend, type, data, ne0, rows, d);
            printf("%-22s %-8s ne0=%-6lld nrows=%-6lld %-11s rows=%-4lld "
                   "l2_rel=%.3e max_rel=%.3e rms_out=%.3e\n",
                   name, ggml_type_name(type), (long long) ne0, (long long) nrows,
                   dist_name(d), (long long) rows, r.l2_rel, r.max_rel, r.rms_ref);

            if (want_control && controls_run < 3) {
                const double c = control_f32(backend, type, data, ne0, rows, d);
                printf("%-22s CONTROL (same weights as f32, no activation quant): "
                       "l2_rel=%.3e\n", name, c);
                controls_run++;
            }

            int found = -1;
            for (int i = 0; i < n_aggs; ++i) {
                if (aggs[i].type == type && aggs[i].ne0 == ne0
                    && aggs[i].nrows == nrows && aggs[i].dist == d) {
                    found = i;
                    break;
                }
            }
            if (found < 0) {
                aggs = (struct shape_agg *) realloc(aggs, sizeof(*aggs) * (size_t) (n_aggs + 1));
                if (!aggs) {
                    die("oom");
                }
                aggs[n_aggs].type  = type;
                aggs[n_aggs].ne0   = ne0;
                aggs[n_aggs].nrows = nrows;
                aggs[n_aggs].worst = 0.0;
                aggs[n_aggs].sum_rms = 0.0;
                aggs[n_aggs].max_rms = 0.0;
                aggs[n_aggs].dist  = d;
                aggs[n_aggs].n     = 0;
                found = n_aggs++;
            }
            if (r.l2_rel > aggs[found].worst) {
                aggs[found].worst = r.l2_rel;
            }
            aggs[found].sum_rms += r.l2_rel;
            if (r.l2_rel > aggs[found].max_rms) {
                aggs[found].max_rms = r.l2_rel;
            }
            aggs[found].n++;
        }
        free(data);
    }

    printf("\n# aggregate per (type, ne0, nrows, dist): rms_rel is the per-matvec\n");
    printf("# error; a sum of k independent such rows has ~sqrt(k) times this\n");
    for (int i = 0; i < n_aggs; ++i) {
        printf("SUMMARY type=%-8s ne0=%-6lld nrows=%-6lld %-11s tensors=%-4ld "
               "mean_rms=%.3e worst_rms=%.3e worst_rel=%.3e\n",
               ggml_type_name(aggs[i].type), (long long) aggs[i].ne0,
               (long long) aggs[i].nrows, dist_name(aggs[i].dist), aggs[i].n,
               aggs[i].sum_rms / (double) aggs[i].n, aggs[i].max_rms, aggs[i].worst);
    }
    free(aggs);
    ggml_backend_free(backend);
    gguf_free(gctx);
    fclose(f);
    return 0;
}
