// dump_layer_trace.c — llama.cpp's OWN residual stream, layer by layer.
//
// Every number this project has for the real model is one rel L2 at the LOGITS,
// which is 36 layers of composition with no visibility inside. A per-op error
// budget does not close that gap (see tools/probe_op_budget.c): its per-op
// numbers overshoot the measured end-to-end residual by three orders of
// magnitude, because a per-op number does not compose into a per-model number.
// The per-layer trace is the instrument that does answer the composition
// question: it says whether the engine's error is INJECTED evenly (f32
// rounding, irreducible) or AMPLIFIED at one layer (a per-call error small
// enough to hide in every other measurement the project owns).
//
// How the two sides are made comparable. llama.cpp names the interesting nodes
// itself — `cb(cur, "l_out", il)` in src/models/qwen2.cpp reaches the graph
// callback in llama_context::graph_get_cb (src/llama-context.cpp:2485), which
// calls `ggml_format_name(cur, "%s-%d", name, il)`. So the reference side needs
// no arch-specific table of ours: the slot numbers below are read out of the
// names llama.cpp already gives its own nodes, and src/trace.rs says the same
// thing about the engine side. Renaming a node upstream renumbers our slots,
// which is the correct failure direction.
//
//   node name      slot  what it is
//   -------------  ----  -------------------------------------------------
//   embd           0     token embedding (GET_ROWS out of token_embd.weight)
//   attn_norm-<il> 1     RMSNorm(LayerNorm) that feeds Q/K/V
//   ffn_inp-<il>   2     residual after the attention branch is added
//   ffn_norm-<il>  3     RMSNorm(LayerNorm) that feeds the FFN
//   ffn_out-<il>   4     the FFN's own output, BEFORE it is added
//   l_out-<il>     5     residual after layer <il> (the per-layer row)
//   result_norm    6     the final norm, before the output projection
//
// Slots 2 and 4 exist so the trace can attribute a divergence to a BRANCH
// (attention vs FFN) rather than only to a layer boundary, and slot 6 exists
// so the last segment — final norm then output matmul, the two ops the trace
// otherwise cannot see — is not invisible. The engine records the same seven
// at the same points (src/model.rs, the crate::trace::record call sites).
//
// Output format (little-endian), byte-identical to src/trace.rs so the two
// files can be diffed without a converter:
//
//   magic  : "SLTRACE\0" (8 bytes)
//   version: u32 = 1
//   n_slots: u32 = 7
//   record*: slot i32 | layer i32 | pos u32 | len u32 | len * f32
//
// Controls this tool has, because a dump tool that silently dumps nothing looks
// exactly like a model that agrees everywhere:
//
//   * every node name it sees is counted and printed at the end, and any name
//     it does NOT recognise is printed — an upstream rename shows up as an
//     unexpected name rather than as a suspiciously perfect trace.
//   * a named node that is not f32, or whose first dimension is not the
//     embedding width, is a hard error rather than a skipped record.
//   * --manifest writes the node inventory without running a forward pass, so
//     the slot table can be checked against the build in seconds.
//   * the exact record count is derived and checked, so a partial dump is an
//     error rather than a small divergence.
//
// Usage: dump_layer_trace <model.gguf> <out.trace|-> [tok0,tok1,...] [--manifest] [--one-per-decode]
//
// The context parameters are gen_ref_logits.c's, deliberately, INCLUDING THE
// DECODE SHAPE — one `llama_decode` carrying all n_tok positions rather than
// one call per token. A trace taken under a different configuration would be a
// trace of a different forward pass: ggml reduces ne1 = 1 (gemv) in a different
// f32 order than ne1 = n_tok (gemm), which at n_threads = 1 is worth an amount
// comparable to the residual being traced, and the per-op budget already showed
// that llama.cpp's inference-time precision shortcuts are worth ~5e-4 — three
// orders of magnitude above it. `--one-per-decode` runs the other shape so that
// term is measured rather than argued about.

#define _POSIX_C_SOURCE 200809L // setenv()

#include <ggml.h>
#include <llama.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "SLTRACE\0"
#define VERSION 1u
#define N_SLOTS 7u

static const char * const SLOT_NAMES[N_SLOTS] = {
    // Slot 0 is NOT named "inp_embd", which is the name llama.cpp DOES give this
    // tensor -- in llama-graph.cpp:2280 -- but that tensor is a graph *input*
    // (`ggml_set_input`), so it never appears in the graph's node list and an
    // eval callback cannot see it. The node that actually carries the value is
    // the GET_ROWS that reads token_embd.weight, which ggml names "embd". A
    // dump tool that silently looked for the upstream name would write nothing,
    // which looks exactly like a model that agrees everywhere.
    "embd", "attn_norm", "ffn_inp", "ffn_norm", "ffn_out", "l_out", "result_norm",
};

typedef struct {
    FILE * f;
    int    n_written;
    int    n_unknown;      // named nodes this tool does not map
    int    n_seen[N_SLOTS];
    int    n_embd;
    int    pos;            // current decode position
    bool   to_stdout;
} sink;

static void die(const char * msg) {
    fprintf(stderr, "dump_layer_trace: %s\n", msg);
    exit(1);
}

static void wr_u32(FILE * f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void wr_i32(FILE * f, int32_t v) { fwrite(&v, 4, 1, f); }

// Map a node name onto a slot, or -1. The name is llama.cpp's own; for il >= 0
// it is "<name>-<il>", for il < 0 it is the bare name.
static int slot_of(const char * name, int * out_layer) {
    *out_layer = -1;
    for (uint32_t s = 0; s < N_SLOTS; s++) {
        const size_t n = strlen(SLOT_NAMES[s]);
        if (strncmp(name, SLOT_NAMES[s], n) != 0) {
            continue;
        }
        if (name[n] == '\0') {
            return (int) s;
        }
        if (name[n] == '-') {
            char * end = NULL;
            long v = strtol(name + n + 1, &end, 10);
            if (end && *end == '\0') {
                *out_layer = (int) v;
                return (int) s;
            }
        }
    }
    return -1;
}

// The ggml eval callback protocol has TWO phases and both of them are load
// bearing; getting either wrong produces a file that is merely short, with no
// error anywhere (ggml/src/ggml-backend.cpp:1736-1764):
//
//   ask=true   FILTER. Return true for a node whose data we want. ggml then
//              runs the subgraph ending at that node and calls back again.
//              Returning false here for everything means the data phase is
//              never entered at all.
//   ask=false  DATA. t->data is valid. Returning FALSE here ABORTS the rest of
//              the graph (the loop's `if (need && !cb(t,false)) break;`), so a
//              dump that returns false after the first record measures exactly
//              one layer and calls it a trace.
//
// The node inventory is taken in the ask phase because that is the only phase
// every node passes through.
static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    sink * s = (sink *) user_data;
    const char * name = t->name[0] ? t->name : NULL;
    int layer = -1;
    const int slot = name ? slot_of(name, &layer) : -1;
    if (ask) {
        if (slot < 0) {
            // Only names llama.cpp itself gave a node (i.e. one its own cb()
            // call touched). Everything else in the graph is unnamed here.
            if (s->n_unknown < 24) {
                fprintf(stderr, "unmapped node name: %s\n", name);
            }
            s->n_unknown++;
            return false;
        }
        // Ask for the node whatever its type: a wanted name carrying something
        // other than f32 has to reach the data phase and be a hard error there,
        // not be skipped by the filter.
        return true;
    }
    if (slot < 0) {
        return true;
    }
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "node %s is not f32 (type %d) — refusing to write a "
                        "half-understood record\n", name, (int) t->type);
        exit(1);
    }
    if (getenv("DLT_DEBUG_NE")) {
        fprintf(stderr, "DBG node %-14s ne=[%lld %lld %lld %lld] nb=[%lld %lld]\n", name,
                (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                (long long) t->nb[1], (long long) t->nb[2]);
    }
    if (t->ne[0] != s->n_embd) {
        fprintf(stderr, "node %s has ne[0] = %lld, expected n_embd = %d\n", name,
                (long long) t->ne[0], s->n_embd);
        exit(1);
    }
    // ne[1] is the token count inside this ubatch; with one decode per call it
    // is 1, but writing it as a loop keeps the file format identical if the
    // caller ever batches (positions continue from s->pos).
    const int64_t n_tok = t->ne[1] * t->ne[2] * t->ne[3];
    for (int64_t j = 0; j < n_tok; j++) {
        const float * row = (const float *) ((const char *) t->data + j * t->nb[1]);
        wr_i32(s->f, slot);
        wr_i32(s->f, layer);
        wr_u32(s->f, (uint32_t) (s->pos + j));
        wr_u32(s->f, (uint32_t) t->ne[0]);
        if (s->to_stdout) {
            for (int64_t i = 0; i < t->ne[0]; i++) {
                fwrite(&row[i], 4, 1, stdout);
            }
        } else {
            fwrite(row, 4, (size_t) t->ne[0], s->f);
        }
        s->n_written++;
        s->n_seen[slot]++;
    }
    return true;   // false here would abort the rest of the graph
}

static int32_t * parse_tokens(const char * str, uint32_t * out_n) {
    uint32_t n = 1;
    for (const char * p = str; *p; ++p) {
        if (*p == ',') {
            n++;
        }
    }
    int32_t * toks = (int32_t *) malloc(sizeof(int32_t) * n);
    if (!toks) {
        die("oom");
    }
    uint32_t i = 0;
    const char * p = str;
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

static void force_cpu_backends(void) {
    static const char * const vars[] = {
        "GGML_VK_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
        "GGML_CUDA_VISIBLE_DEVICES", "GGML_HIP_VISIBLE_DEVICES", "GGML_METAL_VISIBLE_DEVICES",
        "GGML_VISIBLE_DEVICES", "ONEAPI_DEVICE_SELECTOR", "GGML_SYCL_VISIBLE_DEVICES",
        "GGML_RPC_VISIBLE_DEVICES", "GGML_OPENCL_VISIBLE_DEVICES", "GGML_CANN_VISIBLE_DEVICES",
    };
    for (size_t i = 0; i < sizeof(vars) / sizeof(vars[0]); ++i) {
        if (getenv(vars[i]) == NULL) {
            setenv(vars[i], "-1", 0);
        }
    }
}

int main(int argc, char ** argv) {
    bool manifest = false;
    bool one_per_decode = false;
    bool all_output = false;
    const char * model_path = NULL;
    const char * out_path   = NULL;
    const char * tok_arg    = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--manifest") == 0) {
            manifest = true;
        } else if (strcmp(argv[i], "--one-per-decode") == 0) {
            one_per_decode = true;
        } else if (strcmp(argv[i], "--all-output") == 0) {
            all_output = true;
        } else if (!model_path) {
            model_path = argv[i];
        } else if (!out_path) {
            out_path = argv[i];
        } else {
            tok_arg = argv[i];
        }
    }
    if (!model_path) {
        fprintf(stderr, "usage: %s <model.gguf> <out.trace|-> [tok0,tok1,...] "
                        "[--manifest] [--one-per-decode]\n", argv[0]);
        return 2;
    }

    force_cpu_backends();
    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        die("llama_model_load_from_file failed");
    }
    const int n_embd = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);

    uint32_t n_tok = 0;
    int32_t * tokens = parse_tokens(tok_arg ? tok_arg : "785,6722,315,9625,374", &n_tok);

    sink s;
    memset(&s, 0, sizeof(s));
    s.n_embd = n_embd;
    s.pos    = 0;
    if (out_path && strcmp(out_path, "-") == 0) {
        s.f = stdout;
        s.to_stdout = true;
    } else if (out_path) {
        s.f = fopen(out_path, "wb");
        if (!s.f) {
            die("cannot open output");
        }
    } else {
        die("no output path given (use '-' for stdout)");
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) n_tok;
    cparams.n_batch         = (uint32_t) n_tok;
    cparams.n_threads       = 1;
    cparams.n_threads_batch = 1;
    cparams.no_perf         = true;
    cparams.cb_eval         = eval_cb;
    cparams.cb_eval_user_data = &s;
    // Same precision configuration gen_ref_logits.c uses, and the reason is
    // written there: llama.cpp's shipping defaults (flash attention, f16 KV)
    // are worth ~5e-4, which is a hundred times the residual being traced.
    if (!getenv("LLAMA_REF_LLAMA_DEFAULTS")) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.type_k = GGML_TYPE_F32;
        cparams.type_v = GGML_TYPE_F32;
    }

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        die("llama_init_from_model failed");
    }

    if (s.f && !manifest) {
        fwrite(MAGIC, 1, 8, s.f);
        wr_u32(s.f, VERSION);
        wr_u32(s.f, N_SLOTS);
    }

    fprintf(stderr, "dump_layer_trace: %s  n_embd=%d n_layer=%d n_tok=%u threads=1\n",
            model_path, n_embd, n_layer, n_tok);

    // The decode shape must be gen_ref_logits.c's, or this is a trace of a
    // different forward pass than the logits numbers it is supposed to
    // explain. That means ONE `llama_decode` carrying all n_tok positions, not
    // one call per token: ggml picks a different matmul kernel and therefore a
    // different f32 reduction order for ne1 = 1 (gemv) than for ne1 = n_tok
    // (gemm), and with n_threads = 1 that order is the whole answer at the 4e-6
    // level this trace exists to resolve. `--one-per-decode` runs the other
    // shape so the size of that term can be MEASURED rather than assumed away.
    if (one_per_decode) {
        for (uint32_t i = 0; i < n_tok; i++) {
            s.pos = (int) i;
            llama_batch b = llama_batch_get_one(&tokens[i], 1);
            if (llama_decode(ctx, b) != 0) {
                die("llama_decode failed");
            }
        }
    } else {
        llama_batch batch = llama_batch_init((int32_t) n_tok, 0, 1);
        batch.n_tokens = (int32_t) n_tok;   // llama_batch_init leaves this unset
        if (all_output) {
            batch.logits = (int8_t *) calloc(n_tok, 1);
            for (uint32_t i = 0; i < n_tok; ++i) {
                batch.logits[i] = 1;
            }
        }
        for (uint32_t i = 0; i < n_tok; ++i) {
            batch.token[i]    = (llama_token) tokens[i];
            batch.pos[i]      = (llama_pos) i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
        }
        s.pos = 0;
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            die("llama_decode failed");
        }
    }
    llama_synchronize(ctx);

    fprintf(stderr, "records written: %d\n", s.n_written);
    for (uint32_t k = 0; k < N_SLOTS; k++) {
        fprintf(stderr, "  slot %u %-12s %d records\n", k, SLOT_NAMES[k], s.n_seen[k]);
    }
    fprintf(stderr, "unmapped node names: %d\n", s.n_unknown);

    // The expected count is exactly derivable, so a partial dump cannot be
    // mistaken for a small divergence. Every named slot is one node per layer
    // (or one whole-model node), and every such node carries all n_tok
    // positions, so the total is n_tok x (2 + n_layer x 5) under BOTH decode
    // shapes -- which is the point: the count cannot distinguish a batched
    // trace from a per-token trace, so the record stream has to be compared on
    // content (scripts/layer_trace.py checks it) rather than on its length.
    const long long expect = (long long) n_tok * (2 + (long long) n_layer * 5);
    fprintf(stderr, "expected records: %lld (%u positions x (2 + %d layers x 5))%s\n",
            expect, n_tok, n_layer, one_per_decode ? "  [one-per-decode CONTROL shape]" : "");
    int rc = 0;
    if (!manifest && s.n_written != (int) expect) {
        fprintf(stderr, "RECORD COUNT MISMATCH: wrote %d, expected %lld — some named "
                        "node was not a graph node this run\n", s.n_written, expect);
        rc = 1;
    }

    if (s.f) {
        fflush(s.f);
        if (!s.to_stdout) {
            fclose(s.f);
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(tokens);
    return rc;
}