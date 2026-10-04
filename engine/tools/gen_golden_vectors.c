// gen_golden_vectors.c — produce bit-exact dequantization golden vectors.
//
// Links against llama.cpp's ggml (libggml-base.so) and calls the *authoritative*
// `dequantize_row_*` kernels from ggml/src/ggml-quants.c via
// `ggml_get_type_traits()->to_float`. For every ggml type we emit a handful of
// packed rows plus the f32 row ggml produces from them.
//
// The Rust test engine/tests/quant_golden.rs replays these vectors and demands
// a bit-identical result. This turns "the kernel looks right" into "the kernel
// is right", which is the whole point of the parity project.
//
// Usage: gen_golden_vectors <out.bin>
// Rebuild: scripts/gen_golden_vectors.sh

#include <ggml.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAGIC "SLDQGOLD"
#define VERSION 1u

static const double k_pi = 3.14159265358979323846;

// pattern ids (mirrored in the Rust test)
enum {
    PAT_RAND  = 0,  // deterministic pseudo-random packed bytes
    PAT_ZERO  = 1,  // all 0x00
    PAT_ONES  = 2,  // all 0xFF
    PAT_QUANT = 3,  // produced by ggml's from_float_ref over a deterministic signal
    PAT_COUNT = 4,
};

// ggml's type_traits table leaves .to_float NULL for Q8_K even though
// dequantize_row_q8_K is exported from libggml-base. Bind it explicitly.
void dequantize_row_q8_K(const void * x, float * y, int64_t k);

static uint64_t rng_state;

static uint64_t rng_next(void) {
    // splitmix64: small, deterministic, good avalanche for our purposes
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_seed(uint64_t seed) {
    rng_state = seed;
}

static void fill_random(uint8_t * buf, size_t n) {
    size_t i = 0;
    while (i + 8 <= n) {
        uint64_t v = rng_next();
        memcpy(buf + i, &v, 8);
        i += 8;
    }
    if (i < n) {
        uint64_t v = rng_next();
        memcpy(buf + i, &v, n - i);
    }
}

static void wr_u32(FILE * f, uint32_t v) {
    fwrite(&v, 4, 1, f);
}

// deterministic, well-conditioned f32 signal: a ramp through [-1,1] with a
// couple of octaves of sine and a little noise, so quantizers see a realistic
// mix of large/small magnitudes.
static void signal_row(float * x, int n, int variant) {
    for (int i = 0; i < n; ++i) {
        double t = (double) i / (double) n;
        double v = 2.0 * t - 1.0;
        v += 0.35 * sin(2.0 * k_pi * (double) (variant + 1) * t);
        v += 0.12 * sin(2.0 * k_pi * 17.0 * t + 0.7);
        v += 0.03 * (double) (rng_next() % 1000) / 1000.0;
        x[i] = (float) v;
    }
}

static const enum ggml_type k_types[] = {
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_K,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_IQ1_M,
    GGML_TYPE_TQ1_0,
    GGML_TYPE_TQ2_0,
};

// cases per pattern, for patterns that scale (RAND, QUANT)
#define N_RAND  6
#define N_QUANT 3

#define NE0 256

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <out.bin>\n", argv[0]);
        return 2;
    }

    FILE * f = fopen(argv[1], "wb");
    if (!f) {
        perror("fopen");
        return 1;
    }

    fwrite(MAGIC, 1, 8, f);
    wr_u32(f, VERSION);
    wr_u32(f, (uint32_t) (sizeof(k_types) / sizeof(k_types[0])));

    for (size_t ti = 0; ti < sizeof(k_types) / sizeof(k_types[0]); ++ti) {
        const enum ggml_type type = k_types[ti];
        const struct ggml_type_traits * tt = ggml_get_type_traits(type);
        const char * name = ggml_type_name(type);

        ggml_to_float_t to_float = tt ? tt->to_float : NULL;
        ggml_from_float_t from_float_ref = tt ? tt->from_float_ref : NULL;

        if (type == GGML_TYPE_Q8_K) {
            to_float = (ggml_to_float_t) dequantize_row_q8_K;
        }
        if (!to_float) {
            fprintf(stderr, "skipping %s: no to_float\n", name);
            continue;
        }
        // some reference quantizers need ggml's shared codebook tables
        if (from_float_ref) {
            ggml_quantize_init(type);
        }

        const int blck   = (int) tt->blck_size;
        const size_t bsz = tt->type_size;

        // count cases first so we can write n_cases up front
        uint32_t n_cases = 0;
        n_cases += N_RAND;                 // PAT_RAND
        n_cases += 1;                      // PAT_ZERO
        n_cases += 1;                      // PAT_ONES
        if (from_float_ref) {
            n_cases += N_QUANT;            // PAT_QUANT
        }

        wr_u32(f, (uint32_t) type);
        char namebuf[16];
        memset(namebuf, 0, sizeof(namebuf));
        snprintf(namebuf, sizeof(namebuf), "%s", name);
        fwrite(namebuf, 1, 16, f);
        wr_u32(f, (uint32_t) bsz);
        wr_u32(f, (uint32_t) blck);
        wr_u32(f, n_cases);

        const size_t row_bytes = (size_t) (NE0 / blck) * bsz;
        uint8_t * raw  = (uint8_t *) malloc(row_bytes);
        float  * want = (float *) malloc(sizeof(float) * NE0);
        float  * x    = (float *) malloc(sizeof(float) * NE0);
        if (!raw || !want || !x) {
            fprintf(stderr, "oom\n");
            return 1;
        }

        for (int pat = 0; pat < PAT_COUNT; ++pat) {
            int reps = 0;
            switch (pat) {
                case PAT_RAND:  reps = N_RAND;  break;
                case PAT_ZERO:  reps = 1;       break;
                case PAT_ONES:  reps = 1;       break;
                case PAT_QUANT: reps = from_float_ref ? N_QUANT : 0; break;
                default: break;
            }

            for (int r = 0; r < reps; ++r) {
                switch (pat) {
                    case PAT_RAND:
                        rng_seed(0x51D0C0DEULL + (uint64_t) type * 1000003ULL + (uint64_t) r * 7919ULL);
                        fill_random(raw, row_bytes);
                        break;
                    case PAT_ZERO:
                        memset(raw, 0, row_bytes);
                        break;
                    case PAT_ONES:
                        memset(raw, 0xFF, row_bytes);
                        break;
                    case PAT_QUANT:
                        rng_seed(0xA5A5F00DULL + (uint64_t) type * 31ULL + (uint64_t) r * 104729ULL);
                        signal_row(x, NE0, r);
                        from_float_ref(x, raw, NE0);
                        break;
                    default:
                        break;
                }

                to_float(raw, want, NE0);

                wr_u32(f, (uint32_t) pat);
                wr_u32(f, (uint32_t) NE0);
                wr_u32(f, (uint32_t) row_bytes);
                fwrite(raw, 1, row_bytes, f);
                fwrite(want, sizeof(float), NE0, f);
            }
        }

        free(raw);
        free(want);
        free(x);
    }

    fclose(f);
    return 0;
}
