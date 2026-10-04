// dump_rope.c — ground-truth RoPE outputs straight from ggml.
//
// The engine's `rope()` claims to implement both GGML_ROPE_TYPE_NORMAL and
// GGML_ROPE_TYPE_NEOX. Reading ggml's CPU kernel is not enough to trust that
// transcription — the element pairing is easy to get subtly wrong and the
// failure is silent. This tool runs ggml's own op over a known input and dumps
// the result, so `engine/tests/rope_ground_truth.rs` can assert against it.
//
// Usage: dump_rope <out.bin>

#include <ggml.h>
#include <ggml-cpu.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "SLDROPE"
#define VERSION 1u

// The ladder is walked once per PAIR, so its error compounds with n_dims/2.
// This file used to hard-code N_DIMS 8 -- four rungs -- and the f64/f32
// question was then worth ONE bit in 144, which is why a 2.1e-4 per-op
// difference could sit in the engine behind a golden that said "OK". Real head
// dims are 64..256 (32..128 rungs), so the widths below are the ones that
// decide anything. 8 is kept as the toy case, not as the only case.
#define MAX_DIMS 256

// A distinctive, non-symmetric input so a wrong pairing cannot accidentally
// produce the right answer.
static void fill_input(float * x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f + 0.25f * (float) i;
    }
}

static void run_mode(FILE * f, int mode, float freq_base, int pos,
                     const float * in, float * out, int nd) {
    struct ggml_init_params ip = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nd);
    struct ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    struct ggml_tensor * r = ggml_rope_ext(
            ctx, a, p, NULL, nd, mode,
            512,          // n_ctx_orig
            freq_base,
            1.0f,         // freq_scale
            0.0f,         // ext_factor  (disables YaRN)
            1.0f,         // attn_factor
            0.0f, 0.0f);  // beta_fast, beta_slow

    memcpy(a->data, in, sizeof(float) * nd);
    ((int32_t *) p->data)[0] = pos;

    // `r` IS the rope result — read it back, not a separate dst tensor.
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    memcpy(out, r->data, sizeof(float) * nd);

    // Record: mode, n_dims, freq_base, pos, input[nd], output[nd].
    // (The file-level magic/version are written once by main, not per case.)
    int32_t m = mode;
    uint32_t ndu = (uint32_t) nd;
    int32_t pp = pos;
    fwrite(&m, 4, 1, f);
    fwrite(&ndu, 4, 1, f);
    fwrite(&freq_base, 4, 1, f);
    fwrite(&pp, 4, 1, f);
    fwrite(in, sizeof(float), (size_t) nd, f);
    fwrite(out, sizeof(float), (size_t) nd, f);

    ggml_free(ctx);
}

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

    static float in[MAX_DIMS];
    static float out[MAX_DIMS];

    // file-level header, written exactly once
    fwrite(MAGIC, 1, 8, f);
    uint32_t v = VERSION;
    fwrite(&v, 4, 1, f);

    const float bases[3] = { 10000.0f, 1000000.0f, 500.0f };
    const int   poses[4] = { 0, 1, 7, 137 };
    // Real head dims, plus the toy width it used to be the only case at. 8 is
    // kept so the file still covers the shape the old golden covered.
    const int   dims[7]  = { 8, 32, 64, 80, 96, 128, 256 };
    for (int di = 0; di < 7; ++di) {
        for (int bi = 0; bi < 3; ++bi) {
            for (int pi = 0; pi < 4; ++pi) {
                const int nd = dims[di];
                fill_input(in, nd);
                run_mode(f, GGML_ROPE_TYPE_NORMAL, bases[bi], poses[pi], in, out, nd);
                fill_input(in, nd);
                run_mode(f, GGML_ROPE_TYPE_NEOX,   bases[bi], poses[pi], in, out, nd);
            }
        }
    }
    fclose(f);
    fprintf(stderr, "wrote %s\n", argv[1]);
    return 0;
}
