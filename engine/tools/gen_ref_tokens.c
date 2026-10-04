// gen_ref_tokens.c — emit reference token ids from llama.cpp's own tokenizer.
//
// Ticks 13, 14 and 15 each built an oracle for a component the engine claimed
// to implement but had never checked against upstream: the quant kernels, the
// forward pass, the tensor names. The tokenizer was the last major component
// with no oracle at all — and it is the one whose mistakes are invisible,
// because a wrong pre-tokenizer regex still produces plausible tokens.
//
// This tool closes that gap. It loads any GGUF, tokenizes every line of a
// corpus with `llama_tokenize`, and writes the ids (plus the piece strings, so
// a divergence is readable without a debugger) in a line-oriented format that
// `engine/tests/tokenizer_parity.rs` replays against our own tokenizer.
//
// The corpus file is `name<TAB>escaped-text`, one case per line, `#` comments.
// Escapes are `\\`, `\t`, `\n`, `\r` and `\xNN`; everything else is literal
// UTF-8. Corpus and reference ids live in separate files on purpose — the
// corpus is input, the ids are the thing under test, and the test reads both
// so that editing the corpus invalidates the reference loudly.
//
// `llama_tokenize` takes four booleans' worth of behaviour in two: `add_special`
// and `parse_special`. All four combinations are emitted, because the
// `parse_special` axis is the one that decides whether CONTROL and UNKNOWN
// tokens are cut out of the raw text by `tokenizer_st_partition` at all, and a
// reference file that only carries `parse_special = false` cannot tell an
// engine that implements the flag from one that hard-codes it off.
//
// Usage: gen_ref_tokens <model.gguf> <corpus.txt> <out.ref>

#define _POSIX_C_SOURCE 200809L   // setenv()

#include <llama.h>

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char * msg) {
    fprintf(stderr, "gen_ref_tokens: %s\n", msg);
    exit(1);
}

// One `llama_tokenize` call, with the sign convention unwrapped.
static int tokenize(const struct llama_vocab * vocab, const char * text, int32_t text_len,
                    llama_token * toks, int max_tok, bool add_special, bool parse_special) {
    int n = llama_tokenize(vocab, text, text_len, toks, max_tok, add_special, parse_special);
    if (n < 0) {
        n = -n;
    }
    if (n > max_tok) {
        die("token buffer too small");
    }
    return n;
}

static void print_ids(FILE * f, const char * tag, const llama_token * toks, int n) {
    fputs(tag, f);
    for (int i = 0; i < n; ++i) {
        fprintf(f, " %d", (int) toks[i]);
    }
    fputc('\n', f);
}

// Append one byte to a growable buffer.
static void push(char ** buf, size_t * len, size_t * cap, char c) {
    if (*len + 1 > *cap) {
        *cap = *cap ? *cap * 2 : 64;
        *buf = realloc(*buf, *cap);
        if (!*buf) {
            die("oom");
        }
    }
    (*buf)[(*len)++] = c;
}

// Decode one escape starting at *i, advancing *i past whatever it consumed.
// Every path advances by at least one byte, so a bare `while (s[i])` loop
// cannot stall.
static int unescape(const char * s, size_t * i) {
    unsigned char c = (unsigned char) s[*i];
    if (c != '\\' || s[*i + 1] == '\0') {
        (*i)++;
        return c;
    }
    switch (s[*i + 1]) {
        case 'n': *i += 2; return '\n';
        case 't': *i += 2; return '\t';
        case 'r': *i += 2; return '\r';
        case '\\': *i += 2; return '\\';
        case 'x': {
            int hi = s[*i + 2];
            int lo = s[*i + 3];
            if (!isxdigit(hi) || !isxdigit(lo)) {
                (*i)++;  // not a valid escape: the backslash is literal
                return '\\';
            }
            char hex[3] = { (char) hi, (char) lo, 0 };
            int v = (int) strtol(hex, NULL, 16);
            *i += 4;
            return v;
        }
        default: (*i)++; return '\\';
    }
}

// Render `s` back into the corpus escaping, so the reference file is
// self-describing and can be eyeballed without the corpus.
//
// `utf8_ok` passes printable bytes through; set it to 0 for token pieces, which
// are byte-decoded and can split a multi-byte codepoint across two tokens — a
// ZWJ emoji sequence does exactly that — and would otherwise make the file
// invalid UTF-8.
static void escape_into(FILE * f, const char * s, int len, int utf8_ok) {
    for (int i = 0; i < len; ++i) {
        unsigned char c = (unsigned char) s[i];
        switch (c) {
            case '\\': fputs("\\\\", f); break;
            case '\n': fputs("\\n", f); break;
            case '\t': fputs("\\t", f); break;
            case '\r': fputs("\\r", f); break;
            default:
                if (c < 0x20 || c == 0x7F || (c >= 0x80 && !utf8_ok)) {
                    fprintf(f, "\\x%02X", c);
                } else {
                    fputc(c, f);
                }
        }
    }
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <model.gguf> <corpus.txt> <out.ref>\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * corpus     = argv[2];
    const char * out_path   = argv[3];

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;   // tokenizer is backend-independent, but load stays CPU-only
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        die("llama_model_load_from_file failed");
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenizing twice per case (add_special on and off) is cheap; sizing the
    // buffer from the first call is not reliable because BOS/EOS can add tokens.
    const int max_tok = n_vocab + 64;
    llama_token * toks = malloc(sizeof(llama_token) * (size_t) max_tok);
    if (!toks) {
        die("oom");
    }

    FILE * in = fopen(corpus, "rb");
    if (!in) {
        perror("fopen corpus");
        return 1;
    }
    FILE * out = fopen(out_path, "wb");
    if (!out) {
        perror("fopen out");
        return 1;
    }

    fprintf(out, "# tokenizer parity oracle — generated by engine/tools/gen_ref_tokens.c\n");
    // The basename, not the path: the path is whatever the caller typed, so
    // recording it would make the file's bytes depend on the working directory
    // and a regeneration would show up as a diff when nothing changed.
    {
        const char * slash = strrchr(model_path, '/');
        fprintf(out, "# model %s (vocab %d)\n", slash ? slash + 1 : model_path, n_vocab);
    }
    fprintf(out, "# per case: CASE <name> | four token counts | TEXT <escaped>\n");
    fprintf(out, "#   IDS <ids>               parse_special=false add_special=false\n");
    fprintf(out, "#   IDS <ids>               parse_special=false add_special=true\n");
    fprintf(out, "#   PARSE_SPECIAL_IDS <ids> parse_special=true  add_special=false\n");
    fprintf(out, "#   PARSE_SPECIAL_IDS <ids> parse_special=true  add_special=true\n");
    fprintf(out, "#   then one PIECE line per parse_special=false add_special=false id\n");
    fprintf(out, "# the EOG line precedes the cases: EOG <ids> from llama_vocab_is_eog\n");
    // The EOG set, from llama.cpp's own `llama_vocab_is_eog` over every id.
    // This is llama.cpp's answer to "when do we stop", and it is derived from
    // token *text* (llama-vocab.cpp:2785-2946) rather than from any GGUF key, so
    // nothing in the metadata says what it will be. It also assigns attributes
    // on the way through, which is why it belongs in the tokenizer oracle rather
    // than in the forward-pass one. Ascending, because upstream's is a std::set.
    {
        int n_eog = 0;
        fputs("EOG", out);
        for (llama_token id = 0; id < n_vocab; ++id) {
            if (llama_vocab_is_eog(vocab, id)) {
                fprintf(out, " %d", (int) id);
                n_eog++;
            }
        }
        fputc('\n', out);
        fprintf(stderr, "gen_ref_tokens: %d eog ids\n", n_eog);
    }

    char line[1 << 20];
    // Pieces are at most a few bytes, but the API takes the buffer length, so
    // size it generously once rather than tracking capacity.
    char  piece[512];
    char *text = NULL;
    size_t tcap = 0;
    int n_cases = 0;

    while (fgets(line, sizeof(line), in)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
            line[--n] = '\0';
        }
        if (n == 0 || line[0] == '#') {
            continue;
        }
        char * tab = strchr(line, '\t');
        if (!tab) {
            die("corpus line without a TAB separator");
        }
        *tab = '\0';
        const char * name = line;
        const char * esc  = tab + 1;

        // Decode the escaped text.
        size_t tlen = 0;
        for (size_t i = 0; esc[i] != '\0'; ) {
            push(&text, &tlen, &tcap, (char) unescape(esc, &i));
        }
        push(&text, &tlen, &tcap, '\0');   // NUL terminates; tlen excludes it
        const int32_t text_len = (int32_t) (tlen - 1);

        // Four calls, one per (add_special, parse_special) combination.
        int counts[2][2];
        for (int ps = 0; ps < 2; ++ps) {
            for (int as = 0; as < 2; ++as) {
                counts[ps][as] = tokenize(vocab, text, text_len, toks, max_tok,
                                         as != 0, ps != 0);
            }
        }

        fprintf(out, "CASE %s | NOSPECIAL %d | SPECIAL %d | PARSE_SPECIAL_NOSPECIAL %d | PARSE_SPECIAL_SPECIAL %d\n",
                name, counts[0][0], counts[0][1], counts[1][0], counts[1][1]);
        fprintf(out, "TEXT ");
        escape_into(out, text, (int) text_len, 1);
        fputc('\n', out);

        // parse_special=false first, then parse_special=true; within each, the
        // add_special=false call precedes the add_special=true one.
        for (int ps = 0; ps < 2; ++ps) {
            for (int as = 0; as < 2; ++as) {
                tokenize(vocab, text, text_len, toks, max_tok, as != 0, ps != 0);
                print_ids(out, ps ? "PARSE_SPECIAL_IDS" : "IDS", toks, counts[ps][as]);
            }
        }

        // One PIECE line per id, so an empty or `|`-bearing piece cannot be
        // confused with a separator and the count is exact.
        tokenize(vocab, text, text_len, toks, max_tok, false, false);
        for (int i = 0; i < counts[0][0]; ++i) {
            int plen = llama_token_to_piece(vocab, toks[i], piece, (int32_t) sizeof(piece), 0, false);
            if (plen < 0) {
                die("llama_token_to_piece returned negative");
            }
            if (plen >= (int) sizeof(piece)) {
                die("piece does not fit the buffer");
            }
            fputs("PIECE ", out);
            escape_into(out, piece, plen, 0);
            fputc('\n', out);
        }
        n_cases++;
    }

    fclose(in);
    fclose(out);
    free(toks);
    free(text);

    fprintf(stderr, "gen_ref_tokens: %d cases -> %s\n", n_cases, out_path);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
