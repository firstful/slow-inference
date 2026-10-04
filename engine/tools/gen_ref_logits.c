// gen_ref_logits.c — emit reference logits from llama.cpp's own forward pass.
//
// The whole parity project stalled on "we have no GGUF for architecture X".
// This tool removes that blocker: link it against a built llama.cpp, point it
// at ANY gguf (including tiny synthetic fixtures), and it writes the exact
// f32 logits llama.cpp produces for a fixed token sequence. The Rust harness
// (tests/ref_parity.rs) replays the same tokens through our own forward pass
// and diffs, which turns "structurally implemented" into "numerically verified".
//
// Output format (little-endian), matching engine/tests/data/*.ref:
//   magic  : "SLDREFLG" (8 bytes)
//   version: u32 = 1
//   n_tok  : u32            number of prompt tokens
//   tokens: n_tok * i32
//   n_vocab: u32
//   logits : n_tok * n_vocab * f32   (row i = logits after consuming tokens[0..=i])
//
// Every position emits logits (batch.logits[i] = 1) so we diff the whole
// prefill, not just the final token.
//
// Usage: gen_ref_logits <model.gguf> <out.ref> [tok0,tok1,...]
//
// Environment (all optional; the default is "as precise as llama.cpp allows",
// which is what tests/ref_parity.rs gates on):
//
//   LLAMA_REF_LLAMA_DEFAULTS=1   shipping llama.cpp: flash attention on, f16 KV
//   LLAMA_REF_FA_ON               keep flash attention on while still using an
//                                 f32 KV cache (isolates the two shortcuts)
//   LLAMA_REF_KV_F16              keep the f16 KV cache while still disabling
//                                 flash attention
//   LLAMA_REF_THREADS=N           number of threads; ggml's reduction order
//                                 depends on it, so this is a precision knob
//                                 and not just a speed one

#define _POSIX_C_SOURCE 200809L   // setenv()

#include <llama.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "SLDREFLG"
#define VERSION 1u

static void die(const char * msg) {
    fprintf(stderr, "gen_ref_logits: %s\n", msg);
    exit(1);
}

static void wr_u32(FILE * f, uint32_t v) {
    fwrite(&v, 4, 1, f);
}

static void wr_i32(FILE * f, int32_t v) {
    fwrite(&v, 4, 1, f);
}

// Parse "1,2,3" into an int32 array. Tokens default to 0,1,2,... if not given.
static int32_t * parse_tokens(const char * s, uint32_t * out_n) {
    if (!s || !*s) {
        return NULL;
    }
    // count commas to size the array
    uint32_t n = 1;
    for (const char * p = s; *p; ++p) {
        if (*p == ',') {
            n++;
        }
    }
    int32_t * toks = (int32_t *) malloc(sizeof(int32_t) * n);
    if (!toks) {
        die("oom");
    }
    uint32_t i = 0;
    const char * p = s;
    while (*p && i < n) {
        char * end = NULL;
        long v = strtol(p, &end, 10);
        toks[i++] = (int32_t) v;
        if (!end || *end != ',') {
            break;
        }
        p = end + 1;
    }
    *out_n = i;
    return toks;
}

// Hide every GPU backend unless the caller already asked for one by name.
// ggml reads these at backend-load time; -1 is ggml's "no devices" sentinel.
static void force_cpu_backends(void) {
    static const char * const vars[] = {
        "GGML_VK_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "GGML_CUDA_VISIBLE_DEVICES",
        "GGML_HIP_VISIBLE_DEVICES",
        "GGML_METAL_VISIBLE_DEVICES",
        "GGML_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
        "GGML_SYCL_VISIBLE_DEVICES",
        "GGML_RPC_VISIBLE_DEVICES",
        "GGML_OPENCL_VISIBLE_DEVICES",
        "GGML_CANN_VISIBLE_DEVICES",
    };
    for (size_t i = 0; i < sizeof(vars) / sizeof(vars[0]); ++i) {
        if (getenv(vars[i]) == NULL) {
            setenv(vars[i], "-1", 0);   // 0 = don't overwrite an existing value
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <model.gguf> <out.ref> [tok0,tok1,...]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];

    // Force the CPU backend before any backend is loaded. A GPU (Vulkan here)
    // produces bit-different reductions, and the whole point of this tool is a
    // stable oracle to diff against. llama.cpp has no in-process API for
    // "CPU only" that survives model loading here, but every backend honours
    // the GGML_<X>_VISIBLE_DEVICES convention, so we set them defensively
    // before llama_backend_init(). A caller-supplied value always wins.
    force_cpu_backends();

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;   // CPU only — determinism over speed

    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        die("llama_model_load_from_file failed");
    }

    // Determine prompt tokens.
    uint32_t n_tok = 0;
    int32_t * tokens = parse_tokens(argc == 4 ? argv[3] : NULL, &n_tok);
    if (!tokens || n_tok == 0) {
        // Default: first 8 valid token ids, so the tool is useful with no args.
        n_tok = 8;
        tokens = (int32_t *) malloc(sizeof(int32_t) * n_tok);
        if (!tokens) {
            die("oom");
        }
        const struct llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        for (uint32_t i = 0; i < n_tok; ++i) {
            tokens[i] = (int32_t) (i % (uint32_t) n_vocab);
        }
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = (uint32_t) n_tok;
    cparams.n_batch    = (uint32_t) n_tok;
    cparams.n_threads  = 1;
    cparams.n_threads_batch = 1;
    cparams.no_perf     = true;

    // This tool exists to be a *numerical ground truth* for architecture
    // semantics, so it must not carry llama.cpp's inference-time precision
    // shortcuts. Two of them dominate, and both are enabled by default:
    //
    // 1. Flash attention. `build_attn_mha` (llama-graph.cpp:2530) casts K and V
    //    to GGML_TYPE_F16 before `ggml_flash_attn_ext`. That costs ~5e-4
    //    relative error in the attention output.
    // 2. The f16 KV cache (llama-context.cpp:3508, type_k/type_v = F16).
    //
    // Measured against an independent float64 forward pass
    // (scripts/arbitrate_logits.py) on the synthetic F32 fixtures, with both
    // defaults in place llama.cpp lands 2e-4..9.6e-4 away from exact while the
    // engine lands 3e-7..6e-7. In other words llama.cpp's *own* forward pass
    // is the less accurate of the two by three orders of magnitude, and using
    // it unmodified would put a 1e-4 noise floor under the comparison — which
    // is exactly the band a subtly wrong activation constant or a transposed
    // slice would hide in. Turning both off drops llama.cpp into the engine's
    // error band, which is what makes the tolerance gate in tests/ref_parity.rs
    // able to mean something.
    //
    // Set LLAMA_REF_LLAMA_DEFAULTS=1 to reproduce llama.cpp's shipping
    // behaviour instead (useful for diffing real quantized models, where this
    // precision loss is swamped by quantization error anyway).
    if (!getenv("LLAMA_REF_LLAMA_DEFAULTS")) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.type_k = GGML_TYPE_F32;
        cparams.type_v = GGML_TYPE_F32;
    }

    // The two shortcuts above are independent knobs, and on a real quantized
    // model they are worth very different amounts, so allow them to be toggled
    // separately. Measured on qwen2.5-3b Q4_K_M they are not a rounding detail:
    // llama.cpp disagrees with *itself* by cos 0.9967-0.9995 between its
    // shipping configuration and the one above, which is the whole width of the
    // "engine gap" this project spent three ticks attributing to quantization.
    // Default (neither set) is unchanged: both shortcuts off unless
    // LLAMA_REF_LLAMA_DEFAULTS is set.
    if (getenv("LLAMA_REF_FA_ON") && !getenv("LLAMA_REF_LLAMA_DEFAULTS")) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
    if (getenv("LLAMA_REF_KV_F16") && !getenv("LLAMA_REF_LLAMA_DEFAULTS")) {
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
    }

    // ggml's reduction order depends on the thread count, so the number of
    // threads is itself a precision knob. Held at 1 by default for determinism;
    // overridable so "how stable is the oracle" can be measured directly.
    const char * nthreads = getenv("LLAMA_REF_THREADS");
    const int nt = nthreads ? atoi(nthreads) : 1;
    if (nt > 0) {
        cparams.n_threads      = (int32_t) nt;
        cparams.n_threads_batch = (int32_t) nt;
    }

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        die("llama_init_from_model failed");
    }
    llama_set_n_threads(ctx, nt, nt);

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // Build a batch where every position outputs logits.
    struct llama_batch batch = llama_batch_init((int32_t) n_tok, 0, 1);
    batch.n_tokens = (int32_t) n_tok;   // llama_batch_init leaves this unset
    for (uint32_t i = 0; i < n_tok; ++i) {
        batch.token[i]    = (llama_token) tokens[i];
        batch.pos[i]      = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = 1;   // output logits for EVERY position
    }

    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        die("llama_decode returned nonzero");
    }
    llama_synchronize(ctx);

    FILE * f = fopen(out_path, "wb");
    if (!f) {
        perror("fopen");
        return 1;
    }
    fwrite(MAGIC, 1, 8, f);
    wr_u32(f, VERSION);
    wr_u32(f, n_tok);
    for (uint32_t i = 0; i < n_tok; ++i) {
        wr_i32(f, tokens[i]);
    }
    wr_u32(f, (uint32_t) n_vocab);
    for (uint32_t i = 0; i < n_tok; ++i) {
        const float * row = llama_get_logits_ith(ctx, (int32_t) i);
        if (!row) {
            die("llama_get_logits_ith returned NULL");
        }
        fwrite(row, sizeof(float), (size_t) n_vocab, f);
    }
    fclose(f);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr,
            "gen_ref_logits: wrote %u tokens x %d vocab logits to %s\n",
            n_tok, n_vocab, out_path);
    free(tokens);
    return 0;
}
