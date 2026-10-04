// probe_op_budget.c — a PER-OP error budget, which is a different question
// from the per-model one the project has been measuring.
//
// `probe_f32_dot.c` answers "does this one reduction match the kernel". The
// open question is different: the engine(F32) vs llama.cpp(F32) residual on
// qwen2.5-3b is now 4.07e-6 after every reduction was matched exactly, and
// nothing says whether that 4.07e-6 is the *composed* consequence of the ops
// we have already measured, or an op nobody has looked at. A per-model number
// cannot tell those apart: a floor that is fully explained and a floor that
// happens to be the right size both print one number.
//
// This tool measures, on the activations the engine's own forward pass
// actually produced (captured by src/opdump.rs, not synthesised), for each
// op that does not reduce but does round:
//
//   op        reference side                        engine side
//   --------  ------------------------------------  ------------------------------
//   silu      ggml_vec_silu_f32 -> ggml_v_expf       x / (1 + expf(-x))   (libm)
//   softmax   ggml_vec_soft_max_f32                  (v - max).exp(), summed
//             expf is ggml_v_expf; sum is f32        in SEQUENTIAL f32, then
//             8-blocks into a double; scale is       divided by that f32
//             1.0/sum in double then scale_f32
//   rope      ggml_compute_forward_rope_flt          f64 ladder, f64 sin/cos,
//             f32 powf/sinf/cosf, f32 ladder         narrowed to f32 at the end
//
// and prints, for each, the relative L2 and the BIT AGREEMENT between the
// reference kernel and the engine's formulation. Those are different questions
// and the project has conflated them before: f64 accumulation is 31x more
// accurate than llama.cpp's kernel and still disagrees with it in 68% of rows.
//
// The composition at the bottom is the point. Each op's error is measured on
// real inputs; the op's contribution to the *logit* residual is bounded by
// (relative error of the op) x (number of times it runs per token), combined
// in quadrature as a random walk. That last step is an assumption and it is
// labelled as one: it is checked afterwards by the end-to-end number, and the
// check is the only thing that makes the budget worth anything.

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// The reference's own exp. Transcribed from ggml/src/ggml-cpu/vec.h:1215
// (ggml_v_expf, __m256 branch) — "adapted from arm limited optimized routine,
// the maximum error is 1.45358 plus 0.5 ulps". This is NOT libm expf and the
// difference is the whole reason silu and softmax are in this file.
//
// `--verify` links the REAL libggml-cpu.so and calls ggml_vec_silu_f32 and
// ggml_vec_soft_max_f32 directly, then checks the transcriptions below against
// them bit for bit. A transcription of a SIMD kernel that is merely plausible
// is worth nothing, and the project's own history is the argument: a hand
// transcription of a GGUF key table shipped with four wrong entries.
// ---------------------------------------------------------------------------
static inline __m256 v_expf(__m256 x) {
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b =
        _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                         _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k =
        _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    const __m256i c = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                        _mm256_set1_ps(0x1.573e2ep-5f)),
                        u,
                        _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                        _mm256_set1_ps(0x1.fffdb6p-2f))),
        u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
        return _mm256_fmadd_ps(j, k, k);
    const __m256i g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x82000000u));
    const __m256 s1 =
        _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(
        _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                      _mm256_set1_ps(192), _CMP_GT_OQ));
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(_mm256_castsi256_ps(d),
                         _mm256_or_ps(
                             _mm256_and_ps(_mm256_castsi256_ps(c),
                                           _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                             _mm256_andnot_ps(_mm256_castsi256_ps(c),
                                              _mm256_fmadd_ps(k, j, k)))));
}

// ggml_silu_f32, vec.h:1046. Note the tail of ggml_vec_silu_f32 is THIS, not
// the vector path: the vector polynomial is only used for the 8-wide blocks.
static inline float ggml_silu_f32(float x) { return x / (1.0f + expf(-x)); }

// ggml_v_silu, vec.h:1255
static inline __m256 v_silu(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    return _mm256_div_ps(x, _mm256_add_ps(one, v_expf(_mm256_sub_ps(zero, x))));
}

// ggml_vec_silu_f32, vec.cpp:413 — 8-wide blocks with a scalar tail.
static void ref_silu(int n, const float *x, float *y) {
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(y + i, v_silu(_mm256_loadu_ps(x + i)));
    for (; i < n; ++i)
        y[i] = ggml_silu_f32(x[i]); /* vec.h:1046 -- libm expf, NOT the polynomial */
}
// The engine's SiLU, verbatim: libm expf, f32 throughout.
static void eng_silu(int n, const float *x, float *y) {
    for (int i = 0; i < n; ++i)
        y[i] = x[i] / (1.0f + expf(-x[i]));
}

// ---------------------------------------------------------------------------
// Softmax. ggml_vec_soft_max_f32 (vec.cpp:531), AVX2 branch, transcribed.
// The three things that differ from the engine, all of them real:
//   1. exp is ggml_v_expf, not libm expf
//   2. the sum is eight f32 lanes reduced as a TREE, widened to double and
//      accumulated in double — not a sequential f32 sum
//   3. ops.cpp:5549 does `sum = 1.0/sum` in DOUBLE and then scales by it,
//      so the reciprocal is a double reciprocal narrowed to f32, whereas the
//      engine divides each element by the f32 sum
// ---------------------------------------------------------------------------
static double ref_softmax(int n, float *y, const float *x, float max) {
    int i = 0;
    double sum = 0.0;
    for (; i + 7 < n; i += 8) {
        __m256 val = v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1), _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (double)_mm_cvtss_f32(val2);
    }
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (double)val;
        y[i] = val;
    }
    return sum;
}

static void ref_softmax_full(int n, const float *x, float *y) {
    float max = -INFINITY;
    for (int i = 0; i < n; ++i)
        if (x[i] > max) max = x[i];
    double sum = ref_softmax(n, y, x, max);
    float recip = (float)(1.0 / sum);
    for (int i = 0; i < n; ++i)
        y[i] *= recip;
}

static void eng_softmax_full(int n, const float *x, float *y) {
    float max = -INFINITY;
    for (int i = 0; i < n; ++i)
        if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += expf(x[i] - max);
    for (int i = 0; i < n; ++i) y[i] = expf(x[i] - max) / sum;
}

// ---------------------------------------------------------------------------
// RoPE. ggml_compute_forward_rope_flt (ops.cpp:5953) with
// ggml_rope_cache_init (ops.cpp:5846) and rotate_pairs (ops.cpp:5934),
// NEOX, no yarn (ext_factor 0, freq_factors NULL, mscale 1), f32 throughout.
// The engine's rope_apply uses an f64 ladder and f64 sin/cos narrowed once.
// ---------------------------------------------------------------------------
static void ref_rope(int n_dims, int pos, float theta_base, const float *src, float *dst) {
    const float theta_scale = powf(theta_base, -2.0f / n_dims);
    float theta = (float)pos;
    const int half = n_dims / 2;
    for (int i0 = 0; i0 < n_dims; i0 += 2) {
        const float c = cosf(theta), s = sinf(theta);
        const int ic = i0 / 2;
        const float x0 = src[ic], x1 = src[ic + half];
        dst[ic] = x0 * c - x1 * s;
        dst[ic + half] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
}

static void eng_rope(int n_dims, int pos, double theta_base, const float *src, float *dst) {
    const double theta_scale = pow(theta_base, -2.0 / (double)n_dims);
    double cur = 1.0;
    const int half = n_dims / 2;
    for (int i = 0; i < half; ++i) {
        const double ang = (double)pos * cur;
        const float s = (float)sin(ang), c = (float)cos(ang);
        cur *= theta_scale;
        const float a = src[i], b = src[i + half];
        dst[i] = a * c - b * s;
        dst[i + half] = a * s + b * c;
    }
}

// Two NaNs with different payloads are not a kernel difference; the polynomial
// and libm both saturate to a NaN and nothing downstream can tell them apart.
static int bits_equal(const float *a, const float *b) {
    if (memcmp(a, b, 4) == 0) return 1;
    return isnan(*a) && isnan(*b);
}

typedef struct { uint32_t slot, len, n; float *v; } rec;
static rec *recs;
static long n_recs;
static void load_acts(const char *path);
static void port_rope_f32(int n_dims, int pos, float theta_base,
                          const float *src, float *dst);

// ---------------------------------------------------------------------------
// --verify: the transcriptions above, against the real exported kernels.
// ---------------------------------------------------------------------------
#ifdef HAVE_GGML
extern void ggml_vec_silu_f32(int n, float *y, const float *x);
extern double ggml_vec_soft_max_f32(int n, float *y, const float *x, float max);
extern float ggml_vec_dot_f32(int n, float *s, const float *x, const float *y);

static int verify(const char *path) {
    static float x[16384], mine[16384], real[16384], ctrl[16384];
    int bad = 0;
    uint64_t s = 0x243F6A8885A308D3ULL;
    for (int i = 0; i < 16384; ++i) {           // full-range f32, incl. denormals
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        uint32_t bits = (uint32_t)(s >> 20);
        if (i % 97 == 0) bits = (uint32_t)(s >> 40) | 0x7f7f0000u; /* near +-1 */
        memcpy(&x[i], &bits, 4);
    }
    printf("verifying transcriptions against libggml-cpu.so\n");

    // silu: ggml_vec_silu_f32
    for (int n = 1; n <= 4096; n = n < 64 ? n + 1 : n * 2) {
        ref_silu(n, x, mine);
        ggml_vec_silu_f32(n, real, x);
        for (int i = 0; i < n; ++i) {
            if (!bits_equal(&mine[i], &real[i])) {
                printf("  silu n=%-5d MISMATCH at %d: mine %.9g  real %.9g\n", n, i, mine[i], real[i]);
                bad = 1;
                break;
            }
        }
    }
    // and the identity control: my own engine-side silu must NOT match, or
    // the check above is not discriminating.
    eng_silu(4096, x, ctrl);
    long d = 0; for (int i = 0; i < 4096; ++i) if (!bits_equal(&ctrl[i], &real[i])) d++;
    printf("  silu transcription vs real kernel : %s\n", bad ? "MISMATCH" : "bit-identical, n=1..4096");
    printf("  control, engine silu vs real kernel: %ld/4096 bits differ (%s)\n",
           d, d > 0 ? "discriminating" : "*** NOT DISCRIMINATING ***");
    if (d == 0) bad = 1;

    // softmax: ggml_vec_soft_max_f32, called exactly as ops.cpp:5546 does
    int bad2 = 0;
    for (int n = 1; n <= 4096; n = n < 64 ? n + 1 : n * 2) {
        float max = -INFINITY;
        for (int i = 0; i < n; ++i) if (x[i] > max) max = x[i];
        double got = ref_softmax(n, mine, x, max);
        double want = ggml_vec_soft_max_f32(n, real, x, max);
        int smis = 0;
        for (int i = 0; i < n; ++i) if (!bits_equal(&mine[i], &real[i])) { smis = 1; break; }
        if (smis || memcmp(&got, &want, 8)) {
            printf("  softmax n=%-5d MISMATCH (sum %.17g vs %.17g)\n", n, got, want);
            bad2 = 1;
        }
    }
    printf("  softmax transcription vs real     : %s\n", bad2 ? "MISMATCH" : "bit-identical incl. the double sum");
    bad |= bad2;

    if (path) {
        // If an activation file is given, verify on THOSE inputs too — the
        // f32 bit patterns a real forward pass produces are not the same
        // distribution as random bits, and a polynomial's edge cases live in
        // its range ends.
        load_acts(path);
        long vs = 0, ns = 0, vo = 0, no = 0;
        for (long r = 0; r < n_recs; ++r) {
            const int n = (int)recs[r].len;
            if (n > 16384) continue;
            if (recs[r].slot == 0) {
                ref_silu(n, recs[r].v, mine);
                ggml_vec_silu_f32(n, real, recs[r].v);
                for (int i = 0; i < n; ++i) { ns++; if (!bits_equal(&mine[i], &real[i])) vs++; }
            } else if (recs[r].slot == 1) {
                float max = -INFINITY;
                for (int i = 0; i < n; ++i) if (recs[r].v[i] > max) max = recs[r].v[i];
                ref_softmax(n, mine, recs[r].v, max);
                ggml_vec_soft_max_f32(n, real, recs[r].v, max);
                for (int i = 0; i < n; ++i) { no++; if (!bits_equal(&mine[i], &real[i])) vo++; }
            }
        }
        printf("  on the real activations: silu %ld/%ld bits differ, softmax %ld/%ld\n", vs, ns, vo, no);
        if (vs || vo) { printf("  *** MISMATCH on real inputs ***\n"); bad = 1; }
        else printf("  real activations: both bit-identical\n");
    }
    {
        // the f32 port, against the f32 reference, on synthetic AND real inputs
        int bad3 = 0;
        for (int n = 2; n <= 512 && !bad3; n += 2) {
            for (int p = 0; p < 64; ++p) {
                for (int i = 0; i < n; ++i) x[i] = sinf((float)(i * 0.11 + p)) * 3.0f;
                ref_rope(n, p, 1000000.0f, x, mine);
                port_rope_f32(n, p, 1000000.0f, x, ctrl);
                if (memcmp(mine, ctrl, (size_t)n * 4)) {
                    printf("  f32 rope port MISMATCH n=%d pos=%d\n", n, p);
                    bad3 = 1; break;
                }
            }
        }
        if (n_recs && !bad3) {
            long vp = 0, np = 0;
            for (long r = 0; r < n_recs; ++r) {
                if (recs[r].slot != 0) continue;
                const int n = (int)recs[r].len; if (n > 16384) continue;
                ref_rope(n, (int)recs[r].n, 1000000.0f, recs[r].v, mine);
                port_rope_f32(n, (int)recs[r].n, 1000000.0f, recs[r].v, ctrl);
                for (int i = 0; i < n; ++i) { np++; if (memcmp(&mine[i], &ctrl[i], 4)) vp++; }
            }
            printf("  f32 rope port on the real vectors: %ld/%ld bits differ\n", vp, np);
            if (vp) bad3 = 1;
        }
        printf("  f32 rope port vs the f32 reference   : %s\n",
               bad3 ? "MISMATCH" : "bit-identical");
        bad |= bad3;
    }
    (void)ggml_vec_dot_f32;
    return bad;
}
#endif

// The port the engine is about to be given: the reference's OWN rule, which is
// f32 from end to end (theta_scale = powf, the ladder multiplies in f32, and
// the angle is sinf/cosf of an f32). It must be bit-identical to ref_rope --
// if it is not, porting it would move the model without matching anything.
static void port_rope_f32(int n_dims, int pos, float theta_base,
                          const float *src, float *dst) {
    const float theta_scale = powf(theta_base, -2.0f / n_dims);
    float theta = (float)pos;
    const int half = n_dims / 2;
    for (int i = 0; i < half; ++i) {
        const float c = cosf(theta), s = sinf(theta);
        theta *= theta_scale;
        const float a = src[i], b = src[i + half];
        dst[i] = a * c - b * s;
        dst[i + half] = a * s + b * c;
    }
}

// ---------------------------------------------------------------------------
// activation file
// ---------------------------------------------------------------------------
static void load_acts(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    char magic[8];
    uint32_t ver, n_slots;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "SOPBUF\0\0", 8)) {
        fprintf(stderr, "not a SOPBUF file\n"); exit(1);
    }
    if (fread(&ver, 4, 1, f) != 1 || fread(&n_slots, 4, 1, f) != 1) exit(1);
    if (ver != 1) { fprintf(stderr, "version %u\n", ver); exit(1); }
    long cap = 1024;
    recs = malloc(cap * sizeof(rec));
    for (;;) {
        uint32_t slot, len, n;
        if (fread(&slot, 4, 1, f) != 1) break;
        if (fread(&len, 4, 1, f) != 1) break;
        if (fread(&n, 4, 1, f) != 1) break;
        if (slot >= n_slots) { fprintf(stderr, "slot %u >= %u\n", slot, n_slots); exit(1); }
        if ((long)n_recs == cap) { cap *= 2; recs = realloc(recs, cap * sizeof(rec)); }
        recs[n_recs].slot = slot; recs[n_recs].len = len; recs[n_recs].n = n;
        recs[n_recs].v = malloc((size_t)len * sizeof(float));
        if (fread(recs[n_recs].v, sizeof(float), len, f) != len) { fprintf(stderr, "short read\n"); exit(1); }
        n_recs++;
    }
    fclose(f);
}

typedef struct { double num, den; long n, bitdiff; } acc;

static void acc_add(acc *a, const float *got, const float *want, int n) {
    for (int i = 0; i < n; ++i) {
        double d = (double)got[i] - (double)want[i];
        a->num += d * d;
        a->den += (double)want[i] * (double)want[i];
        if (!bits_equal(&got[i], &want[i])) a->bitdiff++;
    }
    a->n += n;
}

static void report(const char *name, acc *a) {
    const double rel = a->den > 0 ? sqrt(a->num / a->den) : 0.0;
    printf("  %-22s rel L2 %10.4e   bits differing %8ld / %8ld  (%.2f%%)\n",
           name, rel, a->bitdiff, a->n, a->n ? 100.0 * (double)a->bitdiff / (double)a->n : 0.0);
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--verify") == 0) {
#ifdef HAVE_GGML
        const char *acts = argc > 2 ? argv[2] : NULL;
        return verify(acts) ? 1 : 0;
#else
        fprintf(stderr, "built without -DHAVE_GGML; cannot verify against the real kernels\n");
        return 2;
#endif
    }
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <ops.bin> <rope_theta> <n_layers> [--control]\n"
                "       %s --verify [<ops.bin>]\n"
                "  --control runs on synthetic vectors with no model, so the\n"
                "  reference-vs-engine machinery can be shown able to fail.\n"
                "  --verify links libggml-cpu.so and checks the transcriptions\n"
                "  against the real exported kernels, bit for bit.\n",
                argv[0], argv[0]);
        return 2;
    }
    const int control = (argc > 4 && strcmp(argv[4], "--control") == 0);
    const float rope_theta = (float)atof(argv[2]);
    const int n_layers = atoi(argv[3]);

    if (control) {
        // No model: 64 vectors of the right shapes, deterministic.
        static float x[11008], a[11008], b[11008];
        acc s_silu = {0}, s_soft = {0}, s_rope = {0};
        for (int t = 0; t < 64; ++t) {
            for (int i = 0; i < 11008; ++i)
                x[i] = sinf((float)(i * 0.017 + t)) * (1.0f + (float)(t % 7));
            ref_silu(11008, x, a); eng_silu(11008, x, b); acc_add(&s_silu, b, a, 11008);
            for (int i = 0; i < 11008; ++i) x[i] = cosf((float)(i * 0.031 + t)) * 8.0f;
            ref_softmax_full(11008, x, a); eng_softmax_full(11008, x, b);
            acc_add(&s_soft, b, a, 11008);
            for (int i = 0; i < 128; ++i) x[i] = sinf((float)(i * 0.07 + t));
            ref_rope(128, t, 1000000.0f, x, a); eng_rope(128, t, 1000000.0, x, b);
            acc_add(&s_rope, b, a, 128);
        }
        puts("control (synthetic, no model):");
        report("silu", &s_silu);
        report("softmax", &s_soft);
        report("rope", &s_rope);
        return 0;
    }

    load_acts(argv[1]);
    long n_silu = 0, n_soft = 0, n_rope = 0;
    acc s_silu = {0}, s_soft = {0}, s_rope = {0}, s_rope_out = {0};
    long n_rope_dim = 0, n_soft_len_sum = 0, n_soft_len_max = 0;

    float *a = malloc(1 << 22), *b = malloc(1 << 22);
    float *slot2_in = NULL; int slot2_len = 0, slot2_pos = 0;
    for (long r = 0; r < n_recs; ++r) {
        const int n = (int)recs[r].len;
        if (n > (1 << 21)) { fprintf(stderr, "vector too long: %d\n", n); return 1; }
        switch (recs[r].slot) {
        case 0: /* silu input: the FFN gate pre-activation */
            ref_silu(n, recs[r].v, a); eng_silu(n, recs[r].v, b);
            acc_add(&s_silu, b, a, n); n_silu++;
            break;
        case 1: /* softmax input: attention scores */
            ref_softmax_full(n, recs[r].v, a); eng_softmax_full(n, recs[r].v, b);
            acc_add(&s_soft, b, a, n); n_soft++;
            n_soft_len_sum += n;
            if (n > n_soft_len_max) n_soft_len_max = n;
            break;
        case 2: /* rope input */
            ref_rope(n, (int)recs[r].n, rope_theta, recs[r].v, a);
            eng_rope(n, (int)recs[r].n, (double)rope_theta, recs[r].v, b);
            acc_add(&s_rope, b, a, n); n_rope++; n_rope_dim = n;
            free(slot2_in);
            slot2_in = malloc((size_t)n * sizeof(float));
            memcpy(slot2_in, recs[r].v, (size_t)n * sizeof(float));
            slot2_len = n; slot2_pos = (int)recs[r].n;
            break;
        case 3: /* rope output as the engine produced it: replay the engine's
                 * own rule on the slot-2 input and compare against what it
                 * actually emitted. src and dst must be separate buffers --
                 * eng_rope writes dst[i] before reading src[i + half], so
                 * passing one buffer for both silently rotates half-rotated
                 * data and reports rel L2 ~0.95 that means nothing. */
            if (slot2_in && slot2_len == n && slot2_pos == (int)recs[r].n) {
                eng_rope(n, (int)recs[r].n, (double)rope_theta, slot2_in, b);
                acc_add(&s_rope_out, b, recs[r].v, n);
            }
            break;
        }
    }

    printf("activations: %ld vectors from %s\n", n_recs, argv[1]);
    printf("  silu %ld vectors, softmax %ld vectors (mean len %.1f, max %ld), "
           "rope %ld vectors of dim %ld\n\n",
           n_silu, n_soft, n_soft ? (double)n_soft_len_sum / (double)n_soft : 0.0,
           n_soft_len_max, n_rope, n_rope_dim);
    puts("per-op relative L2 of the engine's formulation against the reference kernel,");
    puts("on the vectors the engine's own forward pass produced:");
    report("silu", &s_silu);
    report("softmax", &s_soft);
    report("rope", &s_rope);
    report("rope (vs engine out)", &s_rope_out);
    (void)0;

    // The composition. Random-walk assumption, stated as one.
    const double e_silu = s_silu.den > 0 ? sqrt(s_silu.num / s_silu.den) : 0;
    const double e_soft = s_soft.den > 0 ? sqrt(s_soft.num / s_soft.den) : 0;
    const double e_rope = s_rope.den > 0 ? sqrt(s_rope.num / s_rope.den) : 0;
    const double silu_calls = (double)n_layers;
    const double rope_calls = (double)n_layers * 2.0 * 2.0; /* 2 tensors, q and k */
    const double soft_calls = (double)n_layers;              /* 16 heads share it */
    const double pred =
        sqrt(e_silu * e_silu * silu_calls + e_rope * e_rope * rope_calls +
             e_soft * e_soft * soft_calls);
    printf("\ncomposition (ASSUMES a random walk over independent roundings -- an\n"
           "assumption, not a result; the end-to-end number is what tests it):\n"
           "  silu     %10.4e x %5.0f calls\n"
           "  rope     %10.4e x %5.0f calls\n"
           "  softmax  %10.4e x %5.0f calls\n"
           "  predicted composed rel L2  %10.4e\n",
           e_silu, silu_calls, e_rope, rope_calls, e_soft, soft_calls, pred);
    const double measured = 4.0668e-6;
    printf("\nmeasured end-to-end engine(F32) vs llama.cpp(F32), moon worst "
           "position: %.4e\n", measured);
    const double ratio = measured / pred;
    printf("ratio measured/predicted = %.4fx\n", ratio);
    if (ratio > 0.3 && ratio < 3.0)
        puts("  -> the random walk adds up: this budget explains the residual");
    else if (ratio <= 0.3)
        puts("  -> the budget OVERSHOOTS by orders of magnitude, so the random-walk\n"
             "     composition is WRONG, not the ops. An op whose error is a coherent\n"
             "     ROTATION (rope) or whose error is correlated across its inputs does\n"
             "     not accumulate like independent roundings. Read the per-op numbers,\n"
             "     do not read the sum.");
    else
        puts("  -> the budget UNDERSHOOTS: some op is still unaccounted for");
    return 0;
}
