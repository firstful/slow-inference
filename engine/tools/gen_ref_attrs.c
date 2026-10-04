// gen_ref_attrs.c — dump every token's `llama_token_attr` mask from llama.cpp.
//
// Why a second tool, when gen_ref_tokens.c already exists
// -------------------------------------------------------
// `gen_ref_tokens` compares *token ids*: what `llama_tokenize` produces for a
// corpus. That is the right oracle for the pre-tokenizer and for the special
// partition, and it is blind to LLAMA_TOKEN_ATTR_LSTRIP / RSTRIP in a way that
// matters here, because reaching either flag needs a token that no vocabulary
// in this project has — `<mask>` for the jina-v2 rule, `[MASK]` for the
// modern-bert rule. The obvious fixture (a synthetic vocabulary carrying those
// tokens) then fails for an unrelated reason: with a trivial merge table the
// surrounding text encodes to *nothing*, so stripping whitespace out of it
// changes no ids and the oracle reports equality for a partition that is
// provably wrong.
//
// The flags are attributes, and `llama_vocab_get_attr` is public API
// (llama.h:1085). So the honest oracle is the attribute itself: one line per
// id, the full mask, from llama.cpp. It needs no corpus, no working merge
// table, and no text encoding — only a GGUF that loads — and a difference in
// the mask names the rule that is wrong instead of the ids that moved.
//
// Usage: gen_ref_attrs <model.gguf> <out.attrs>
//
// Format, one line per id, in id order:
//
//   ATTRS <n_vocab>
//   ATTR <id> <mask>
//
// `mask` is decimal, and is the full `llama_token_attr` OR — not a projection.
// A projection would let a port drop SINGLE_WORD or BYTE unnoticed, which is
// the same mistake as testing the `tokenizer.ggml.token_type` key instead of
// the derived attribute.

#define _POSIX_C_SOURCE 200809L   // setenv()

#include <llama.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char * msg) {
    fprintf(stderr, "gen_ref_attrs: %s\n", msg);
    exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <out.attrs>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    // Vocab-only load. llama.cpp has no such flag, and there need not be one:
    // this tool is only ever pointed at the synthetic fixtures, which are a few
    // hundred KB. Pointing it at a real model "works" and then takes minutes.
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == NULL) {
        die("llama_model_load_from_file failed");
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab == NULL) {
        die("llama_model_get_vocab returned NULL");
    }

    FILE * f = fopen(out_path, "w");
    if (f == NULL) {
        die("cannot open output");
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    fprintf(f, "ATTRS %d\n", n_vocab);
    for (llama_token id = 0; id < n_vocab; id++) {
        fprintf(f, "ATTR %d %d\n", id, (int) llama_vocab_get_attr(vocab, id));
    }
    fclose(f);
    fprintf(stderr, "gen_ref_attrs: %d ids -> %s\n", n_vocab, out_path);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}