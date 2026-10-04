//! Tokenizer parity against llama.cpp, over `tests/data/tokenizer_corpus.txt`.
//!
//! This is the oracle tick 13 built for the quant kernels and tick 15 built for
//! the tensor names, applied to the last big component that had none. Every id
//! here is llama.cpp's own output: `engine/tools/gen_ref_tokens.c` calls
//! `llama_tokenize` on the same corpus and `engine/scripts/gen_ref_tokens.sh`
//! regenerates the `.ref` files.
//!
//! Why it matters: a wrong pre-tokenizer regex still *works*. It emits plausible
//! token ids, the forward pass accepts them, and generation looks like language
//! with slightly wrong grammar. Nothing else in the engine would catch it. And
//! the previous tokenizer ignored `tokenizer.ggml.pre` entirely while using a
//! hand-written approximation of the GPT-2 regex — so it was tokenizing Qwen2
//! (`pre = "qwen2"`) and OLMoE (`pre = "olmo"`) models with the wrong regex.
//! These tests fail on that by construction: the two pre-tokenizers disagree on
//! digit runs (`\p{N}` vs `\p{N}+`), on `!S`-style contractions, and on where a
//! leading space attaches.

use slow_inference::gguf::Value;
use slow_inference::spm::Spm;
use slow_inference::tokenizer::{Bpe, PreType};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const CORPUS: &str = "tests/data/tokenizer_corpus.txt";
const SWEEP_CORPUS: &str = "tests/data/tokenizer_sweep.txt";
const REF_DIR: &str = "tests/data/ref_tokens";

/// One corpus case: the name and the decoded text.
#[derive(Debug)]
struct Case {
    name: String,
    text: String,
}

/// What llama.cpp produced for one case. `pieces` stays in the corpus's escape
/// form: llama.cpp byte-decodes token pieces, so one piece can be a *fragment*
/// of a multi-byte codepoint (a ZWJ emoji sequence splits across four tokens)
/// and is not valid UTF-8 on its own.
#[derive(Debug)]
struct RefCase {
    text: String,
    /// `[add_special][parse_special]` -> the ids. Both axes matter:
    /// `add_special` decides whether BOS/EOS are prepended and appended, and
    /// `parse_special` decides whether the vocabulary's CONTROL and UNKNOWN
    /// tokens are cut out of the raw text at all.
    ids: [[Vec<i32>; 2]; 2],
    pieces: Vec<String>,
    /// Whether the TEXT line was seen. The empty-prompt case has an empty
    /// string there, so presence cannot be inferred from the value.
    text_seen: bool,
    /// How many id lines were seen; must be 4. The empty prompt makes
    /// `ids[0][0]` empty, so presence cannot be inferred from emptiness.
    ids_lines: usize,
    /// The token counts `gen_ref_tokens.c` recorded on the CASE line, checked
    /// against the id lines so a truncated or duplicated record cannot pass.
    counts: [usize; 4],
}

/// Decode the corpus file's escapes (`\\`, `\t`, `\n`, `\r`, `\xNN`). Anything
/// else after a backslash is a literal backslash, matching the C unescaper in
/// `tools/gen_ref_tokens.c`.
fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'\\' || i + 1 >= b.len() {
            out.push(b[i]);
            i += 1;
            continue;
        }
        match b[i + 1] {
            b'n' => {
                out.push(b'\n');
                i += 2;
            }
            b't' => {
                out.push(b'\t');
                i += 2;
            }
            b'r' => {
                out.push(b'\r');
                i += 2;
            }
            b'\\' => {
                out.push(b'\\');
                i += 2;
            }
            b'x' if i + 3 < b.len()
                && b[i + 2].is_ascii_hexdigit()
                && b[i + 3].is_ascii_hexdigit() =>
            {
                let v = u8::from_str_radix(&s[i + 2..i + 4], 16).expect("checked hexdigits");
                out.push(v);
                i += 4;
            }
            _ => {
                out.push(b[i]);
                i += 1;
            }
        }
    }
    String::from_utf8(out).expect("escaped text is valid UTF-8")
}

fn read_corpus() -> Vec<Case> {
    read_corpus_at(
        Path::new(env!("CARGO_MANIFEST_DIR")).join(CORPUS).as_path(),
        30,
    )
}

fn read_corpus_at(path: &Path, min_cases: usize) -> Vec<Case> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let mut cases = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, escaped)) = line.split_once('\t') else {
            panic!("corpus line {} has no TAB separator", lineno + 1);
        };
        cases.push(Case {
            name: name.to_string(),
            text: unescape(escaped),
        });
    }
    assert!(
        cases.len() >= min_cases,
        "corpus shrank to {} cases",
        cases.len()
    );
    cases
}

/// Parse one `.ref` file. The format is written by `gen_ref_tokens.c`:
///
/// ```text
/// CASE <name> | NOSPECIAL <n> | SPECIAL <n> | PARSE_SPECIAL_NOSPECIAL <n> | PARSE_SPECIAL_SPECIAL <n>
/// TEXT <escaped>
/// IDS <id>...               parse_special=false add_special=false
/// IDS <id>...               parse_special=false add_special=true
/// PARSE_SPECIAL_IDS <id>... parse_special=true  add_special=false
/// PARSE_SPECIAL_IDS <id>... parse_special=true  add_special=true
/// PIECE <p>                 one line per parse_special=false add_special=false id
/// ```
fn read_ref(path: &Path) -> HashMap<String, RefCase> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let mut out: HashMap<String, RefCase> = HashMap::new();
    let mut current: Option<String> = None;
    // Which of the four id lines is being read, as `parse_special * 2 +
    // add_special`. Counting rather than testing emptiness, because the
    // degenerate prompt legitimately produces zero tokens for
    // `add_special = false` and that case is in the corpus.
    let mut ids_line = 0usize;

    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        // The `EOG` header line is read by `read_eog`, not here; skipping it by
        // name keeps this parser's "data before any CASE" guard honest about the
        // *cases* rather than about the file's preamble.
        if line.is_empty() || line.starts_with('#') || line == "EOG" || line.starts_with("EOG ") {
            continue;
        }
        if let Some(rest) = line.strip_prefix("CASE ") {
            let mut fields = rest.split('|').map(str::trim);
            let name = fields.next().unwrap().to_string();
            let mut counts = [0usize; 4];
            for (i, key) in [
                "NOSPECIAL",
                "SPECIAL",
                "PARSE_SPECIAL_NOSPECIAL",
                "PARSE_SPECIAL_SPECIAL",
            ]
            .iter()
            .enumerate()
            {
                let f = fields
                    .next()
                    .unwrap_or_else(|| panic!("{path:?}: CASE {name} is missing {key}"));
                let (k, v) = f
                    .split_once(' ')
                    .unwrap_or_else(|| panic!("{path:?}: CASE {name} field {f:?} has no count"));
                assert_eq!(k, *key, "{path:?}: CASE {name} field {k:?} out of order");
                counts[i] = v
                    .parse()
                    .unwrap_or_else(|e| panic!("{path:?}: CASE {name} bad count {v:?}: {e}"));
            }
            assert!(!out.contains_key(&name), "{path:?}: duplicate case {name}");
            out.insert(
                name.clone(),
                RefCase {
                    text: String::new(),
                    text_seen: false,
                    ids_lines: 0,
                    counts,
                    ids: Default::default(),
                    pieces: Vec::new(),
                },
            );
            current = Some(name);
            ids_line = 0;
            continue;
        }
        let Some(name) = current.as_ref() else {
            panic!("{path:?} line {}: data before any CASE", lineno + 1);
        };
        let entry = out.get_mut(name).expect("current case exists");
        if let Some(rest) = line.strip_prefix("TEXT ") {
            entry.text = unescape(rest);
            entry.text_seen = true;
        } else if let Some((tag, rest)) = line
            .strip_prefix("PARSE_SPECIAL_IDS")
            .map(|rest| (1usize, rest))
            .or_else(|| line.strip_prefix("IDS").map(|rest| (0usize, rest)))
        {
            let ids: Vec<i32> = rest
                .split_whitespace()
                .map(|t| {
                    t.parse::<i32>()
                        .unwrap_or_else(|e| panic!("bad id {t:?}: {e}"))
                })
                .collect();
            assert!(
                ids_line < 4,
                "{path:?}: case {name} has more than four id lines"
            );
            entry.ids[tag][ids_line % 2] = ids;
            assert_eq!(
                entry.counts[ids_line],
                entry.ids[tag][ids_line % 2].len(),
                "{path:?}: case {name} recorded {} tokens on the CASE line but its id line {ids_line} has {}",
                entry.counts[ids_line],
                entry.ids[tag][ids_line % 2].len()
            );
            ids_line += 1;
            entry.ids_lines = ids_line;
        } else if let Some(rest) = line.strip_prefix("PIECE ") {
            entry.pieces.push(rest.to_string());
        } else {
            panic!("{path:?} line {}: unrecognised record", lineno + 1);
        }
    }

    for (name, r) in &out {
        assert!(r.text_seen, "{path:?}: case {name} has no TEXT line");
        assert_eq!(
            r.ids_lines, 4,
            "{path:?}: case {name} has {} id lines, expected 4",
            r.ids_lines
        );
        assert_eq!(
            r.pieces.len(),
            r.ids[0][0].len(),
            "{path:?}: case {name} PIECES/IDS length mismatch"
        );
    }
    out
}

/// Load a tokenizer from a real GGUF, or `None` if it is not on disk. The
/// `.ref` files are checked in, but the vocabularies they came from are
/// multi-GB, so a checkout without `models/` skips rather than fails.
fn load_or_skip(model: &str) -> Option<Bpe> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("models")
        .join(model);
    if !path.exists() {
        eprintln!("skipping {model}: not in models/");
        return None;
    }
    let gf = slow_inference::gguf::parse(&path.to_string_lossy())
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    Some(Bpe::from_metadata(&gf.metadata).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

/// The two real BPE GGUFs with a checked-in `.ref`. `pre` is declared per model
/// and differs — this is the field the old tokenizer discarded.
/// (gguf, .ref, declared pre-tokenizer, min vocab, min merges)
const BPE_FIXTURES: &[(&str, &str, PreType, usize, usize)] = &[
    (
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "qwen2.5-3b-instruct-q4_k_m.ref",
        PreType::Qwen2,
        150_000,
        150_000,
    ),
    (
        "OLMoE-1B-7B-Instruct-Q4_K_M.gguf",
        "OLMoE-1B-7B-Instruct-Q4_K_M.ref",
        PreType::Gpt2,
        50_000,
        50_000,
    ),
];

fn ref_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(REF_DIR)
        .join(name)
}

/// The one corpus case that legitimately produces zero tokens for
/// `add_special = false`: llama.cpp's SPM path returns nothing for an empty
/// string, and the BPE paths do the same.
const EMPTY_PROMPT_CASE: &str = "empty_text";

/// The corpus the reference files were generated from must still be the corpus
/// in the tree: a reworded case silently invalidates every id below it.
#[test]
fn reference_files_match_the_current_corpus() {
    let cases = read_corpus();
    assert!(
        cases.iter().any(|c| c.name == EMPTY_PROMPT_CASE),
        "the {EMPTY_PROMPT_CASE} case is missing from the corpus"
    );
    for (model, ref_file) in all_ref_fixtures() {
        let refs = read_ref(&ref_path(ref_file));
        assert_eq!(
            refs.len(),
            cases.len(),
            "{ref_file}: reference has {} cases, corpus has {} — regenerate with scripts/gen_ref_tokens.sh",
            refs.len(),
            cases.len()
        );
        for c in &cases {
            let r = refs
                .get(&c.name)
                .unwrap_or_else(|| panic!("{ref_file}: no reference for case {}", c.name));
            assert_eq!(r.text, c.text, "{ref_file}: case {} text drifted", c.name);
            assert!(
                c.name == EMPTY_PROMPT_CASE || !r.ids[0][0].is_empty(),
                "{ref_file}: case {} produced no tokens; the corpus or the GGUF is broken",
                c.name
            );
        }
        eprintln!("{model}: {} cases match the reference corpus", cases.len());
    }
}

/// Every case, all four combinations of `add_special` and `parse_special`,
/// ids must match llama.cpp exactly.
///
/// Qwen2 (`pre = "qwen2"`) has a hand-written pre-tokenizer state machine in
/// llama.cpp that this engine ports verbatim, so it must match on every case.
///
/// OLMoE (`pre = "olmo"`) also must, and for a while it did not: six cases
/// involving runs of two or more spaces diverged. Two ticks of notes blamed
/// llama.cpp's generic `std::regex` path over collapsed codepoints — the claim
/// was that OLMo's `regex_exprs` did not equal the literal
/// `unicode_regex_split_custom` key and so fell through. It does equal it, byte
/// for byte (`llama-vocab.cpp:367` vs `unicode.cpp:1053`), and calling
/// `unicode_regex_split` directly with OLMo's expression list returns exactly
/// the GPT-2 state machine's words.
///
/// The real cause is one step earlier. `llama_vocab::impl::tokenize` runs
/// `tokenizer_st_partition` before any regex: it cuts the raw text on every
/// vocabulary token whose `tokenizer.ggml.token_type` is UNKNOWN, CONTROL or
/// USER_DEFINED, longest first. OLMoE marks ids 50254–50276 — every run of 2
/// to 24 literal spaces — USER_DEFINED, so they are cut on even with
/// `parse_special = false` (which skips CONTROL and UNKNOWN, i.e. all of
/// Qwen2's specials). A run of two or more spaces therefore becomes one
/// whitespace token and the following word loses its leading space.
///
/// The divergence list this test used to carry is gone rather than shrunk:
/// `the_partition_sweep_matches_llama_cpp` below is its replacement, and it
/// fails if the partition regresses to nothing.
///
/// The `parse_special` half of the matrix is what stops the flag from being
/// decorative. `parse_special_special_discriminating_set_is_exact` pins which
/// corpus cases llama.cpp actually tokenizes differently under it, per model,
/// so this test cannot pass for an engine that hard-codes it off.
#[test]
fn token_ids_match_llama_cpp() {
    let cases = read_corpus();
    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for (model, ref_file, _, _, _) in BPE_FIXTURES {
        let Some(tok) = load_or_skip(model) else {
            continue;
        };
        let refs = read_ref(&ref_path(ref_file));
        for c in &cases {
            let r = &refs[&c.name];
            for parse_special in [false, true] {
                for add_special in [false, true] {
                    let expect = &r.ids[usize::from(parse_special)][usize::from(add_special)];
                    let got = tok.encode(&c.text, add_special, parse_special);
                    if &got == expect {
                        continue;
                    }
                    failures.push(format!(
                        "{model} case {} (add_special={add_special} parse_special={parse_special}):\n  input   {:?}\n  words   {:?}\n  want    {:?}\n  got     {:?}\n  pieces  {:?}",
                        c.name,
                        c.text,
                        tok.pretokenize(&c.text),
                        expect,
                        got,
                        r.pieces,
                    ));
                }
            }
            checked += 1;
        }
    }
    assert!(
        failures.is_empty(),
        "{} BPE case(s) diverge from llama.cpp:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
    assert!(
        checked > 0,
        "no BPE GGUF available — parity was not actually exercised"
    );
}

// ---------------------------------------------------------------------------
// special-token partition (llama.cpp tokenizer_st_partition)
// ---------------------------------------------------------------------------

/// `llama_token_type`, the subset the partition cares about.
const TYPE_NORMAL: i64 = 1;
const TYPE_UNKNOWN: i64 = 2;
const TYPE_CONTROL: i64 = 3;
const TYPE_USER_DEFINED: i64 = 4;

/// A synthetic BPE vocabulary. `types` is `tokenizer.ggml.token_type`; pass an
/// empty slice to omit the key, which llama.cpp treats as "every token NORMAL".
fn synth_bpe(tokens: &[&str], types: &[i64]) -> Bpe {
    synth_bpe_with_extra(tokens, types, &[])
}

/// [`synth_bpe`] plus arbitrary metadata keys, so a fixture can *declare* the
/// thing whose absence is supposed to enable a behaviour.
fn synth_bpe_with_extra(tokens: &[&str], types: &[i64], extra: &[(&str, Value)]) -> Bpe {
    let mut md = HashMap::new();
    md.insert("tokenizer.ggml.model".into(), Value::Str("gpt2".into()));
    md.insert("tokenizer.ggml.pre".into(), Value::Str("gpt2".into()));
    md.insert(
        "tokenizer.ggml.tokens".into(),
        Value::Array(tokens.iter().map(|s| Value::Str((*s).into())).collect()),
    );
    if !types.is_empty() {
        md.insert(
            "tokenizer.ggml.token_type".into(),
            Value::Array(types.iter().map(|&t| Value::I32(t as i32)).collect()),
        );
    }
    for (k, v) in extra {
        md.insert((*k).to_string(), v.clone());
    }
    Bpe::from_metadata(&md).expect("synthetic metadata is valid")
}

/// A miniature of the OLMoE vocabulary, where ids 3/4/7 are whitespace runs
/// marked USER_DEFINED and ids 5/8 are CONTROL. `0`/`1`/`2`/`6` are ordinary
/// byte-level BPE pieces; there are no merges, so every pre-token word maps
/// straight to an id.
fn whitespace_run_vocab_tokens() -> [&'static str; 9] {
    [
        "a", "\u{120}a", "\u{120}", "  ", "    ", "<s>", "b", "   ", "",
    ]
}

fn whitespace_run_vocab_types() -> [i64; 9] {
    [
        TYPE_NORMAL,
        TYPE_NORMAL,
        TYPE_NORMAL,
        TYPE_USER_DEFINED,
        TYPE_USER_DEFINED,
        TYPE_CONTROL,
        TYPE_NORMAL,
        TYPE_USER_DEFINED,
        TYPE_CONTROL,
    ]
}

fn whitespace_run_vocab() -> Bpe {
    synth_bpe(
        &whitespace_run_vocab_tokens(),
        &whitespace_run_vocab_types(),
    )
}

/// `tokenizer_st_partition` runs before the pre-tokenizer, so a USER_DEFINED
/// token is emitted as itself and never reaches the regex or the merge table.
///
/// This is the whole reason OLMoE used to diverge. `"a  b"` pre-tokenizes to
/// `["a", " ", " b"]` and BPEs to three ids; llama.cpp emits `["a"]`, the
/// two-space token, `["b"]`.
#[test]
fn user_defined_tokens_are_cut_out_before_pre_tokenization() {
    let tok = whitespace_run_vocab();
    assert_eq!(
        tok.pretokenize("a  b"),
        vec!["a", "\u{120}", "\u{120}b"],
        "without the partition a two-space run emits one space and attaches \
         the other to the next word"
    );
    assert_eq!(tok.encode("a  b", false, false), vec![0, 3, 6]);
    // A single space is not a special token, so it keeps the GPT-2 attachment
    // rule: the space joins the following word. `Ġb` is not in this miniature
    // vocabulary, so it byte-falls-back to `Ġ` + `b`.
    assert_eq!(tok.encode("a b", false, false), vec![0, 2, 6]);
}

/// Under `parse_special = false` llama.cpp cuts on USER_DEFINED tokens but not
/// on CONTROL or UNKNOWN ones, which is the setting a bare prompt uses. Qwen2's
/// `<|im_*|>` family is CONTROL and so invisible here; OLMoE's whitespace runs
/// are USER_DEFINED and are not.
#[test]
fn control_tokens_are_not_cut_on_when_parse_special_is_false() {
    let tok = whitespace_run_vocab();
    let ids = tok.encode("a <s> b", false, false);
    assert!(
        !ids.contains(&5),
        "<s> is CONTROL and must be pre-tokenized, not emitted: {ids:?}"
    );
    // The same text with <s> re-marked USER_DEFINED *is* cut on. Without this
    // half the test above would also pass for an engine that never partitions.
    let mut md_types = whitespace_run_vocab_types();
    md_types[5] = TYPE_USER_DEFINED;
    let cut = synth_bpe(&whitespace_run_vocab_tokens(), &md_types);
    assert_eq!(cut.encode("a<s>b", false, false), vec![0, 5, 6]);
}

/// The same text with `parse_special = true`, which is what
/// `llama-server`'s `tokenize_mixed(vocab, text, true, true)` and the
/// `--parse-special` flag select.
///
/// llama.cpp's condition is one line
/// (`llama-vocab.cpp:3174`):
///
/// ```text
/// if (!parse_special && (data.attr & (CONTROL | UNKNOWN))) continue;
/// ```
///
/// so UNKNOWN and CONTROL are the *same* branch and USER_DEFINED is the
/// fallthrough. Both halves matter. The UNKNOWN half has no coverage from any
/// vocabulary on disk — Qwen2, OLMoE and Tiny-MoE declare zero UNKNOWN tokens —
/// so this is the only test in the suite that reaches it, which is why the
/// synthetic vocabulary is worth its own assertions rather than being folded
/// into the CONTROL test above.
#[test]
fn control_and_unknown_tokens_are_cut_only_when_parse_special_is_set() {
    for &type_of in &[TYPE_CONTROL, TYPE_UNKNOWN] {
        let mut md_types = whitespace_run_vocab_types();
        md_types[5] = type_of;
        let tok = synth_bpe(&whitespace_run_vocab_tokens(), &md_types);
        let kind = if type_of == TYPE_CONTROL {
            "CONTROL"
        } else {
            "UNKNOWN"
        };

        // parse_special = false: id 5 is spelled out, never emitted.
        let off = tok.encode("a<s>b", false, false);
        assert!(
            !off.contains(&5),
            "{kind} id 5 must not be emitted with parse_special=false: {off:?}"
        );

        // parse_special = true: the same three characters become three ids,
        // because the partition replaces "a<s>b" with ["a", <s>, "b"] and
        // "a"/"b" never reach the pre-tokenizer as one word.
        assert_eq!(
            tok.encode("a<s>b", false, true),
            vec![0, 5, 6],
            "{kind} id 5 must be emitted with parse_special=true"
        );

        // The whole prompt is the special: llama.cpp erases the raw fragment
        // rather than splitting it, so nothing else is emitted.
        assert_eq!(tok.encode("<s>", false, true), vec![5], "{kind} alone");

        // Two in a row: the second match sits at the base offset, so there is
        // no left reminder and no empty raw fragment in between.
        assert_eq!(
            tok.encode("<s><s>", false, true),
            vec![5, 5],
            "{kind} twice"
        );

        // A prefix of the special is not a special. This is what separates
        // "matches the token's whole text" from "matches a prefix", and it is
        // the same check as `special_partial` in the corpus, at the unit level.
        assert_eq!(
            tok.encode("<s", false, true),
            tok.encode("<s", false, false),
            "{kind} prefix must not be cut on"
        );

        // USER_DEFINED is the other branch and partitions either way, so the
        // flag must not move it. The two-space run is id 3.
        assert_eq!(
            tok.encode("a  b", false, true),
            vec![0, 3, 6],
            "{kind}: USER_DEFINED partitions regardless of parse_special"
        );
    }
}

/// llama.cpp overrides `tokenizer.ggml.token_type` by token *text*: a token
/// spelled like an end-of-generation marker is forced to CONTROL whatever the
/// file says. Qwen2.5-3B is the real instance — it ships `</s>` as id 128247
/// with `token_type = 1` (NORMAL) and llama.cpp logs
/// `control-looking token: 128247 '</s>' was not control-type` while loading
/// it. Trusting the file gives 22 specials where llama.cpp has 23.
///
/// This is a whole class of bug that only exists on one axis: with
/// `parse_special = false` a CONTROL token is skipped by the partition anyway,
/// so every assertion in this file passed for twenty-four ticks with the
/// promotion missing.
#[test]
fn a_control_token_promoted_by_text_can_override_the_declared_token_type() {
    // `</s>` and `<|fim_prefix|>` both marked NORMAL, which is what Qwen2.5-3B
    // actually ships for `</s>`.
    let tokens = ["a", "</s>", "<|fim_prefix|>"];
    let normal = [TYPE_NORMAL, TYPE_NORMAL, TYPE_NORMAL];
    let tok = synth_bpe(&tokens, &normal);
    assert_eq!(
        tok.n_special_tokens(),
        2,
        "`</s>` is an end-of-generation alias and `<|fim_prefix|>` a fill-in-the-middle \
         one; both are promoted even though the file calls them NORMAL"
    );
    // parse_special = false: the promotion is invisible, because CONTROL is
    // skipped. This half is why the bug survived.
    assert!(
        !tok.encode("a</s>", false, false).contains(&1),
        "a CONTROL token must not be emitted with parse_special=false"
    );
    // parse_special = true: it is the file's type that would have hidden it.
    assert_eq!(
        tok.encode("a</s>", false, true),
        vec![0, 1],
        "</s> must be emitted: llama.cpp promotes it to CONTROL by spelling"
    );
    assert_eq!(
        tok.encode("</s>", false, true),
        vec![1],
        "and it participates in the partition like any other special"
    );
}

/// The promotion is suppressed when the GGUF declares the id it is looking for,
/// *including the CONTROL promotion*. A converter that writes
/// `tokenizer.ggml.fim_pre_token_id` is saying "I already know", and llama.cpp
/// then runs no alias block at all for that slot. This guard is easy to get
/// wrong in the direction of being more permissive, and being more permissive
/// changes token ids.
///
/// `<|fim_prefix|>` is the right token to test it on: every end-of-text alias is
/// *also* in the end-of-generation set, which is unguarded, so testing the
/// EOT guard with one of those would pass whether or not the guard exists.
#[test]
fn a_declared_token_id_suppresses_that_blocks_promotion() {
    let tokens = ["a", "<|fim_prefix|>", "</s>"];
    let normal = [TYPE_NORMAL, TYPE_NORMAL, TYPE_NORMAL];
    let undeclared = synth_bpe(&tokens, &normal);
    assert_eq!(undeclared.n_special_tokens(), 2);
    let declared = synth_bpe_with_extra(
        &tokens,
        &normal,
        &[("tokenizer.ggml.fim_pre_token_id", Value::I32(1))],
    );
    assert_eq!(
        declared.n_special_tokens(),
        1,
        "`<|fim_prefix|>` is a FIM_PRE alias and the GGUF declares a FIM_PRE id, \
         so that block does not run; `</s>` is only an *end-of-generation* alias, \
         and that block is unguarded, so it still promotes"
    );
    assert!(
        !declared.encode("a<|fim_prefix|>", false, true).contains(&1),
        "the suppressed block must leave the token as NORMAL"
    );
}

/// The regression, on the real vocabulary: 23 specials, and id 128247 among
/// them. 22 is what the engine produced while it trusted `token_type`.
#[test]
fn qwen2_promotes_its_eos_to_control_the_way_llama_cpp_does() {
    const QWEN: &str = "qwen2.5-3b-instruct-q4_k_m.gguf";
    let Some(tok) = load_or_skip(QWEN) else {
        return;
    };
    assert_eq!(
        tok.token_text(128247),
        Some("</s>"),
        "the token under test is the one llama.cpp named in its warning"
    );
    // Measured against llama.cpp's own `llama_vocab_get_attr`: 21 CONTROL and 2
    // USER_DEFINED. The GGUF itself declares only 20 CONTROL.
    assert_eq!(
        tok.n_special_tokens(),
        23,
        "{QWEN}: llama.cpp has 23 specials, the file declares 22"
    );
    assert_eq!(
        tok.encode("a</s>", false, true),
        vec![a_prefixed(&tok), 128247],
        "{QWEN}: </s> is emitted as its own id under parse_special"
    );
}

/// The id that `a` becomes after the GPT-2 pre-tokenizer has had it, so the
/// assertion above can name one token rather than hard-coding the merge.
fn a_prefixed(tok: &Bpe) -> i32 {
    tok.encode("a", false, false)[0]
}

/// The zero-length token's `[EMPTY_8]` rewrite has to keep the sentinel out of
/// the output under `parse_special = true` too. With the flag on, an empty
/// CONTROL token *is* a partition needle, and `[EMPTY_8]` matches nothing —
/// where without the rewrite a zero-length needle would match at every
/// position and `encode` could not make progress.
#[test]
fn the_empty_token_sentinel_stays_unreachable_with_parse_special_set() {
    let tok = whitespace_run_vocab();
    for add_special in [false, true] {
        for parse_special in [false, true] {
            let ids = tok.encode("a <s> b", add_special, parse_special);
            assert!(
                !ids.contains(&8),
                "id 8 is the rewritten empty CONTROL token and must never be \
                 emitted (add_special={add_special} parse_special={parse_special}): {ids:?}"
            );
        }
    }
    assert_eq!(
        tok.token_text(8),
        Some("[EMPTY_8]"),
        "the rewrite itself is pinned by an_empty_token_becomes_an_unreachable_sentinel"
    );
}

/// The partition consumes the longest matching special first, so a run of six
/// spaces is `["    ", "  "]` and not `["  ", "  ", "  "]`. Five spaces leaves
/// a single space behind, which is no longer a special token and goes to the
/// pre-tokenizer as the byte-encoded word `Ġ` (id 2 here).
#[test]
fn the_longest_special_token_is_consumed_first() {
    let tok = whitespace_run_vocab();
    assert_eq!(tok.encode("      ", false, false), vec![4, 3]);
    assert_eq!(tok.encode("  ", false, false), vec![3]);
    assert_eq!(tok.encode("   ", false, false), vec![7]);
    assert_eq!(tok.encode("     ", false, false), vec![4, 2]);
}

/// llama.cpp replaces a zero-length token with a printable sentinel at load
/// time. Left alone, an empty special token would match at every position of
/// the partition and the text could not be tokenized at all.
#[test]
fn an_empty_token_becomes_an_unreachable_sentinel() {
    let tok = whitespace_run_vocab();
    assert_eq!(
        tok.n_special_tokens(),
        5,
        "3 whitespace runs + <s> + the empty CONTROL token"
    );
    // The rewrite itself. llama.cpp logs `empty token at index 8` and stores
    // `[EMPTY_8]`; keeping the empty string instead would put a zero-length
    // needle in the partition, which matches at every position and cannot make
    // progress.
    assert_eq!(tok.token_text(8), Some("[EMPTY_8]"));
    assert_eq!(tok.token_text(0), Some("a"));
    assert_eq!(tok.token_text(-1), None);
    assert_eq!(tok.token_text(99), None);
    assert_eq!(tok.encode("", false, false), Vec::<i32>::new());
    assert_eq!(tok.encode("a", false, false), vec![0]);
    assert_eq!(tok.encode("a  b", false, false), vec![0, 3, 6]);
}

/// A vocabulary with no `tokenizer.ggml.token_type` has nothing to cut on, so
/// it tokenizes exactly as it did before the partition existed. This is the
/// shape of every hand-written tokenizer fixture in this file, and of any GGUF
/// converted before the key was written.
#[test]
fn a_vocabulary_without_token_type_is_not_partitioned() {
    let tok = synth_bpe(&["a", "\u{120}a", "\u{120}", "  ", "b"], &[]);
    assert_eq!(tok.n_special_tokens(), 0);
    // "  " is in the vocabulary but is a NORMAL token, so the pre-tokenizer
    // still splits the run: `a`, then one space as `Ġ`, then `Ġb` — which is
    // absent here and so byte-falls-back to `Ġ` + `b`.
    assert_eq!(tok.encode("a  b", false, false), vec![0, 2, 2, 4]);
    assert_eq!(tok.encode("a b", false, false), vec![0, 2, 4]);
}

/// Cross-check the special-token set against llama.cpp's own count. Loading
/// OLMoE logs `special tokens cache size = 29`, which is
/// `{UNKNOWN, CONTROL, USER_DEFINED}` over `tokenizer.ggml.token_type` — 23
/// whitespace runs, `|||IP_ADDRESS|||`, `|||EMAIL_ADDRESS|||`,
/// `|||PHONE_NUMBER|||`, `<|padding|>`, `<pad>`, the empty EOG marker, and
/// `</tool_call>`-style PII sentinels.
///
/// Every expected id below was measured by calling `llama_tokenize` on
/// OLMoE with `parse_special = false` — not derived from the implementation.
#[test]
fn olmoe_special_token_count_matches_llama_cpp() {
    let Some(tok) = load_or_skip("OLMoE-1B-7B-Instruct-Q4_K_M.gguf") else {
        eprintln!("skipping: OLMoE GGUF not in models/");
        return;
    };
    assert_eq!(tok.n_special_tokens(), 29);
    // llama.cpp logs `special tokens cache size = 29` for this model and no
    // `empty token at index N` line, so its 50279 is the ordinary 13-byte
    // `<|endoftext|>` and the zero-length rewrite must not fire here.
    assert_eq!(tok.token_text(50279).map(str::len), Some(13));
    let spaces = |n: usize| " ".repeat(n);
    // Ids 50254..50276 are runs of 24 down to 2 literal spaces.
    assert_eq!(tok.encode(&spaces(2), false, false), vec![50276]);
    assert_eq!(tok.encode(&spaces(3), false, false), vec![50275]);
    assert_eq!(tok.encode(&spaces(24), false, false), vec![50254]);
    // A run longer than the longest token is consumed greedily, longest first,
    // and the leftover is matched against the remaining specials: 25 leaves a
    // single space (no such token, so it is pre-tokenized to `Ġ` = 209), 26
    // leaves two (id 50276), 30 leaves six (id 50272).
    assert_eq!(tok.encode(&spaces(25), false, false), vec![50254, 209]);
    assert_eq!(tok.encode(&spaces(26), false, false), vec![50254, 50276]);
    assert_eq!(tok.encode(&spaces(30), false, false), vec![50254, 50272]);
    assert_eq!(tok.encode(&spaces(48), false, false), vec![50254, 50254]);
    assert_eq!(
        tok.encode(&spaces(49), false, false),
        vec![50254, 50254, 209]
    );
    // A run with a word after it: the whitespace token is emitted on its own
    // and the word keeps no leading space.
    assert_eq!(tok.encode("  x", false, false), vec![50276, 89]);
    assert_eq!(tok.encode("   x", false, false), vec![50275, 89]);
    assert_eq!(
        tok.encode("a  b   c", false, false),
        vec![66, 50276, 67, 50275, 68]
    );
    assert_eq!(
        tok.encode("x  y  z", false, false),
        vec![89, 50276, 90, 50276, 91]
    );
    // The PII sentinels are USER_DEFINED too, so a literal occurrence is the
    // whole prompt and becomes token 0 un-tokenized.
    assert_eq!(tok.encode("|||IP_ADDRESS|||", false, false), vec![0]);
    // ...while `<s>` is CONTROL: `parse_special = false` means llama.cpp
    // pre-tokenizes it into `Ġ<`, `s`, `>` instead of emitting id 29.
    assert_eq!(
        tok.encode("  <s>  ", false, false),
        vec![50276, 29, 84, 31, 50276]
    );
}

/// The main corpus has six whitespace-run cases because that is how the
/// divergence was first noticed; this sweep is 75 cases aimed at the same
/// component, replayed against llama.cpp's own ids.
///
/// It is a separate file because the main corpus's exact-set allowlists make a
/// new case there a decision rather than a keystroke, and because this one
/// should be *generated* rather than curated. Regenerate with
/// `scripts/gen_tokenizer_sweep.py > tests/data/tokenizer_sweep.txt` and
/// `scripts/gen_ref_tokens.sh <olmoe.gguf>`.
#[test]
fn the_partition_sweep_matches_llama_cpp() {
    let Some(tok) = load_or_skip("OLMoE-1B-7B-Instruct-Q4_K_M.gguf") else {
        eprintln!("skipping: OLMoE GGUF not in models/");
        return;
    };
    let cases = read_corpus_at(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join(SWEEP_CORPUS),
        60,
    );
    let refs = read_ref(&ref_path("OLMoE-1B-7B-Instruct-Q4_K_M.sweep.ref"));
    assert_eq!(
        refs.len(),
        cases.len(),
        "the sweep reference is out of date — rerun scripts/gen_ref_tokens.sh"
    );

    let mut failures: Vec<String> = Vec::new();
    let mut whitespace_tokens_seen = 0usize;
    for c in &cases {
        let r = &refs[&c.name];
        // Counted once per case, not once per combination: the guard is about
        // the corpus reaching the partition at all, and OLMoE's whitespace runs
        // are USER_DEFINED so all four combinations emit them equally.
        let mut reached_partition = false;
        for parse_special in [false, true] {
            for add_special in [false, true] {
                let expect = &r.ids[usize::from(parse_special)][usize::from(add_special)];
                let got = tok.encode(&c.text, add_special, parse_special);
                if &got != expect {
                    failures.push(format!(
                        "case {} (add_special={add_special} parse_special={parse_special}):\n  input   {:?}\n  want    {:?}\n  got     {:?}\n  pieces  {:?}",
                        c.name, c.text, expect, got, r.pieces,
                    ));
                }
                reached_partition |= got.iter().any(|&i| (50254..=50276).contains(&i));
            }
        }
        whitespace_tokens_seen += usize::from(reached_partition);
    }
    assert!(
        failures.is_empty(),
        "{} sweep case(s) diverge from llama.cpp:\n{}",
        failures.len(),
        failures.join("\n")
    );
    // The sweep has to actually reach the whitespace specials, or it would be
    // green for an engine that partitions nothing.
    assert!(
        whitespace_tokens_seen >= 40,
        "only {whitespace_tokens_seen} of the sweep's cases emitted a whitespace-run \
         token; the corpus is not exercising the partition"
    );
}

/// The pre-tokenizer is selected by `tokenizer.ggml.pre`. Pinning it keeps a
/// future refactor from collapsing both models back onto one regex.
#[test]
fn each_model_selects_the_pre_tokenizer_it_declares() {
    for (model, _, want, min_vocab, min_merges) in BPE_FIXTURES {
        let Some(tok) = load_or_skip(model) else {
            continue;
        };
        assert_eq!(tok.pre_type(), *want, "{model}");
        assert!(
            tok.n_vocab() >= *min_vocab,
            "{model}: vocab {} below the expected {min_vocab}",
            tok.n_vocab()
        );
        assert!(
            tok.n_merges() >= *min_merges,
            "{model}: {} merges below the expected {min_merges}",
            tok.n_merges()
        );
    }
}

/// Pre-tokenize `text` with a synthetic tokenizer for the given pre-tokenizer.
/// An empty vocabulary is fine — `pretokenize` never consults it.
fn words_with(text: &str, pre: PreType) -> Vec<String> {
    let name = match pre {
        PreType::Gpt2 => "gpt2",
        PreType::Qwen2 => "qwen2",
        PreType::Qwen35 => "qwen35",
        PreType::Llama3 => "llama3",
    };
    assert_eq!(words_for(text, name).2, pre);
    words_for(text, name).0
}

/// `(pre-token words, vocabulary size, resolved PreType)` for a named pre.
fn words_for(text: &str, pre_name: &str) -> (Vec<String>, usize, PreType) {
    let mut md = HashMap::new();
    md.insert("tokenizer.ggml.model".into(), Value::Str("gpt2".into()));
    md.insert("tokenizer.ggml.pre".into(), Value::Str(pre_name.into()));
    md.insert("tokenizer.ggml.tokens".into(), Value::Array(Vec::new()));
    let tok = Bpe::from_metadata(&md).expect("synthetic metadata is valid");
    let n = tok.n_vocab();
    (tok.pretokenize(text), n, tok.pre_type())
}

/// The Qwen2 and GPT-2/OLMo splitters must actually disagree, or the corpus is
/// not testing what it claims to. These are the sharpest differences: digit-run
/// grouping and the case-insensitive contraction alternative.
#[test]
fn the_two_pretokenizers_disagree_where_the_regexes_do() {
    // `\p{N}` vs `\p{N}+`: Qwen2 emits one word per digit, GPT-2 one word per run.
    assert_eq!(words_with("12345", PreType::Gpt2), vec!["12345"]);
    assert_eq!(
        words_with("12345", PreType::Qwen2),
        vec!["1", "2", "3", "4", "5"]
    );

    // GPT-2's contraction alternative is lower-case only, so `IT'S` does not
    // match it: the apostrophe and the S become separate words. Qwen2 spells
    // the alternative `'[sS]`, so it matches.
    assert_eq!(words_with("IT'S", PreType::Gpt2), vec!["IT", "'", "S"]);
    assert_eq!(words_with("IT'S", PreType::Qwen2), vec!["IT", "'S"]);

    // Lower case contracts in both, so the divergence above is about case only.
    assert_eq!(words_with("it's", PreType::Gpt2), vec!["it", "'s"]);
    assert_eq!(words_with("it's", PreType::Qwen2), vec!["it", "'s"]);

    // A comma-split number differs in both letter-run grouping and digit runs.
    let gpt2 = words_with("1,234,567", PreType::Gpt2);
    let qwen2 = words_with("1,234,567", PreType::Qwen2);
    assert_eq!(gpt2, vec!["1", ",", "234", ",", "567"]);
    assert_eq!(qwen2, vec!["1", ",", "2", "3", "4", ",", "5", "6", "7"]);

    // pre="olmo" shares the GPT-2 regex upstream, so it must too here.
    assert_eq!(words_for("12345", "olmo").0, words_for("12345", "gpt2").0);
}

/// Pre-tokenization is byte-level: every pre-token word must consist only of
/// chars in the GPT-2 bytes-to-unicode alphabet, and the concatenation of all
/// of them must be the byte-for-byte encoding of the input.
#[test]
fn pretokenizer_is_lossless_and_byte_encoded() {
    let table = slow_inference::tokenizer::byte_to_unicode_table();
    for text in [
        "Hello, world!",
        "  leading",
        "tab\there",
        "你好 \u{1F600}",
        "12345 !!! ???",
        "line\n\nbreak\r\n",
    ] {
        for pre in [
            PreType::Gpt2,
            PreType::Qwen2,
            PreType::Qwen35,
            PreType::Llama3,
        ] {
            let words = words_with(text, pre);
            let rejoined: String = words.concat();
            let expected: String = text.bytes().map(|b| table[b as usize]).collect();
            assert_eq!(
                rejoined, expected,
                "{pre:?} on {text:?}: pre-token words do not re-encode to the input"
            );
            for w in &words {
                assert!(
                    w.chars().all(|c| table.contains(&c)),
                    "{pre:?} on {text:?}: word {w:?} contains a char outside the byte alphabet"
                );
            }
            assert!(!words.is_empty(), "{pre:?} on {text:?}: no pre-token words");
        }
    }
}

/// Unsupported pre-tokenizers must be refused by name, not silently
/// tokenized with the wrong regex — that failure mode is what this whole
/// file exists to prevent.
#[test]
fn unsupported_pre_and_model_are_refused_by_name() {
    let base = |model: &str, pre: &str| {
        let mut md = HashMap::new();
        md.insert("tokenizer.ggml.model".into(), Value::Str(model.into()));
        md.insert("tokenizer.ggml.pre".into(), Value::Str(pre.into()));
        md.insert("tokenizer.ggml.tokens".into(), Value::Array(Vec::new()));
        Bpe::from_metadata(&md)
    };

    let err = |model: &str, pre: &str| base(model, pre).err();
    // llama3/qwen35 have ports, so these must load.
    assert!(err("gpt2", "llama3").is_none());
    assert!(err("gpt2", "qwen35").is_none());
    // starcoder, bloom and friends have no port yet; refusing them by name is
    // the whole point.
    assert_eq!(
        err("gpt2", "starcoder"),
        Some(slow_inference::tokenizer::Error::UnsupportedPre(
            "starcoder".into()
        ))
    );
    assert_eq!(
        err("gpt2", "bloom"),
        Some(slow_inference::tokenizer::Error::UnsupportedPre(
            "bloom".into()
        ))
    );
    // "default" is llama.cpp's own fallback, and it is a 4-expression regex we
    // have not ported — refusing is correct, guessing is not.
    assert_eq!(
        err("gpt2", "default"),
        Some(slow_inference::tokenizer::Error::UnsupportedPre(
            "default".into()
        ))
    );
    assert_eq!(
        err("gpt2", ""),
        Some(slow_inference::tokenizer::Error::UnsupportedPre(
            String::new()
        ))
    );
    assert_eq!(
        err("llama", "default"),
        Some(slow_inference::tokenizer::Error::UnsupportedModel(
            "llama".into()
        ))
    );
    assert!(base("gpt2", "qwen2").is_ok());
    assert!(base("gpt2", "olmo").is_ok());
}

// ---------------------------------------------------------------------------
// SPM (`tokenizer.ggml.model = "llama"`)
// ---------------------------------------------------------------------------
//
// The BPE section above covers `tokenizer.ggml.model = "gpt2"`. This covers the
// other vocabulary type actually present in `models/`: SentencePiece. It has no
// pre-tokenizer at all — the whole string is one segmentation problem — so the
// parity question is not "which regex" but "which pairs merge, in what order".
//
// What the engine had before this tick was a greedy longest-match loop in
// `main.rs`. That is a different algorithm, not a paraphrase, so the tests below
// do two jobs: match llama.cpp exactly, and prove that the corpus would have
// caught the old one.

/// The real SPM GGUF with a checked-in `.ref`: a 4096-token llama-2 style
/// vocabulary. 4096 is small enough that it has no multi-digit tokens and no
/// long words, which is exactly what makes byte fallback and stranded markers
/// observable.
const SPM_FIXTURES: &[(&str, &str, usize)] =
    &[("Tiny-Moe.Q4_K_M.gguf", "Tiny-Moe.Q4_K_M.ref", 4096)];

fn all_ref_fixtures() -> Vec<(&'static str, &'static str)> {
    BPE_FIXTURES
        .iter()
        .map(|(m, r, _, _, _)| (*m, *r))
        .chain(SPM_FIXTURES.iter().map(|(m, r, _)| (*m, *r)))
        .collect()
}

/// Corpus cases where `parse_special = true` changes llama.cpp's ids, per
/// reference file, measured from llama.cpp's own output.
///
/// This is the guard that stops `parse_special` from being decorative. A
/// `tokenize_st_partition(text, false)` that ignores its argument passes every
/// `parse_special = false` assertion in this file, so without a positive
/// requirement the new matrix would be green for an engine that never
/// implemented the flag. Checked as an exact set for the same reason
/// [`SPM_MERGE_ORDER_SENSITIVE`] is: a case appearing or disappearing means the
/// corpus or the vocabulary changed, and either way somebody should look.
///
/// The set is small and that is not a defect. Which cases discriminate is a
/// property of *which tokens the vocabulary marks CONTROL*, not of the shapes:
/// * Qwen2 marks 21 CONTROL tokens including the `<|im_*|>` family, and its
///   `add_bos_token` is false, so `add_special` is a no-op there and only the
///   `parse_special` axis can move.
/// * OLMoE has three CONTROL tokens (`<|padding|>`, `<|endoftext|>`, `<pad>`)
///   and only `special_in_word` spells one. Its 26 USER_DEFINED tokens — the
///   whitespace runs and PII sentinels — partition under both settings, which
///   is the invariance the sweep checks 75 times over.
/// * Tiny-MoE has three CONTROL tokens and no USER_DEFINED ones at all.
const PARSE_SPECIAL_DISCRIMINATING: &[(&str, &[&str])] = &[
    ("OLMoE-1B-7B-Instruct-Q4_K_M.ref", &["special_in_word"]),
    ("OLMoE-1B-7B-Instruct-Q4_K_M.sweep.ref", &["control_pad"]),
    (
        "qwen2.5-3b-instruct-q4_k_m.ref",
        &[
            "spm_control_as_text",
            "special_alone",
            "special_edges",
            "special_in_word",
            "special_tight",
            "special_twice",
            "symbols",
        ],
    ),
    ("qwen2.5-3b-instruct-q4_k_m.sweep.ref", &["control_fim"]),
    (
        "Tiny-Moe.Q4_K_M.ref",
        &["spm_control_as_text", "spm_special_space_prefix"],
    ),
];

/// Every reference file must make `parse_special` observable, and the engine
/// must make it observable on exactly the same cases.
///
/// Runs without any GGUF present: it only reads the checked-in `.ref` files,
/// which are llama.cpp's own output. The engine half is skipped when the
/// vocabulary is not on disk.
#[test]
fn parse_special_discriminating_set_is_exact() {
    let mut saw = 0usize;
    for (ref_file, want) in PARSE_SPECIAL_DISCRIMINATING {
        let refs = read_ref(&ref_path(ref_file));
        let mut got: Vec<&str> = refs
            .iter()
            .filter(|(_, r)| r.ids[1][0] != r.ids[0][0] || r.ids[1][1] != r.ids[0][1])
            .map(|(name, _)| name.as_str())
            .collect();
        got.sort_unstable();
        let mut want_sorted = want.to_vec();
        want_sorted.sort_unstable();
        assert_eq!(
            got, want_sorted,
            "{ref_file}: the set of corpus cases parse_special changes changed. \
             Newly discriminating cases are worth recording; newly silent ones \
             mean the flag is no longer exercised on this vocabulary."
        );
        assert!(!got.is_empty(), "{ref_file}: parse_special changes nothing");
        saw += 1;
    }
    assert_eq!(
        saw,
        PARSE_SPECIAL_DISCRIMINATING.len(),
        "no reference file was checked"
    );

    // The engine half: same cases, same direction, using the corpus *text*
    // rather than the case name. `load_or_skip` / `load_spm_or_skip` return
    // None without the GGUF, which is not a failure — the reference half above
    // already ran.
    let cases = read_corpus();
    for (model, ref_file, _, _, _) in BPE_FIXTURES {
        let Some(tok) = load_or_skip(model) else {
            continue;
        };
        let refs = read_ref(&ref_path(ref_file));
        for c in &cases {
            let r = &refs[&c.name];
            assert_eq!(
                tok.encode(&c.text, false, true) != tok.encode(&c.text, false, false),
                r.ids[1][0] != r.ids[0][0],
                "{model} case {}: the engine and llama.cpp disagree about whether \
                 parse_special changes this case",
                c.name
            );
        }
    }
    for (model, ref_file, _) in SPM_FIXTURES {
        let Some(tok) = load_spm_or_skip(model) else {
            continue;
        };
        let refs = read_ref(&ref_path(ref_file));
        for c in &cases {
            let r = &refs[&c.name];
            assert_eq!(
                tok.encode(&c.text, false, true) != tok.encode(&c.text, false, false),
                r.ids[1][0] != r.ids[0][0],
                "{model} case {}: the engine and llama.cpp disagree about whether \
                 parse_special changes this case",
                c.name
            );
        }
    }
}

fn load_spm_or_skip(model: &str) -> Option<Spm> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("models")
        .join(model);
    if !path.exists() {
        eprintln!("skipping {model}: not in models/");
        return None;
    }
    let gf = slow_inference::gguf::parse(&path.to_string_lossy())
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    Some(Spm::from_metadata(&gf.metadata).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

/// Every corpus case, all four combinations of `add_special` and
/// `parse_special`, ids must match llama.cpp.
///
/// Unlike the OLMoE BPE case this is expected to be exact everywhere. The SPM
/// algorithm has no pre-tokenizer ambiguity and no regex to fall through to, so
/// every divergence that exists here is a bug in the port.
///
/// `parse_special` is the interesting axis for SPM. The partition is shared
/// with BPE, but SPM then does something on top of it: `add_space_prefix`
/// applies to a raw fragment whenever the *preceding fragment* was a token, so
/// `"<s> hello"` becomes id 1 followed by the tokenization of `"▁ hello"` — the
/// marker stacked on a leading space that was already in the input. That is
/// `spm_special_space_prefix` in the corpus, and it is the only thing in the
/// suite that can see the interaction.
#[test]
fn spm_token_ids_match_llama_cpp() {
    let cases = read_corpus();
    let mut checked = 0usize;
    for (model, ref_file, min_vocab) in SPM_FIXTURES {
        let Some(tok) = load_spm_or_skip(model) else {
            continue;
        };
        assert!(
            tok.n_vocab() >= *min_vocab,
            "{model}: vocab {} below the expected {min_vocab}",
            tok.n_vocab()
        );
        let refs = read_ref(&ref_path(ref_file));
        for c in &cases {
            let r = &refs[&c.name];
            for parse_special in [false, true] {
                for add_special in [false, true] {
                    let expect = &r.ids[usize::from(parse_special)][usize::from(add_special)];
                    let got = tok.encode(&c.text, add_special, parse_special);
                    assert_eq!(
                        &got,
                        expect,
                        "{model} case {} (add_special={add_special} parse_special={parse_special}):\n  input   {:?}\n  want    {:?}\n  got     {:?}\n  pieces  {:?}\n  as ids  {:?}",
                        c.name,
                        c.text,
                        expect,
                        got,
                        r.pieces,
                        got.iter()
                            .map(|&i| tok.piece(i).unwrap_or("<oob>"))
                            .collect::<Vec<_>>(),
                    );
                }
            }
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "no SPM GGUF available — parity was not actually exercised"
    );
}

/// The corpus must be able to tell the SPM port from the greedy longest-match it
/// replaced, or the parity test above is green for the wrong reason.
///
/// `scripts/find_spm_discriminators.py` searched this vocabulary for texts where
/// the two disagree and the `spm_*` corpus cases are its output; this test
/// re-implements the old algorithm and requires the corpus to still catch it.
/// It is deliberately the *old* algorithm that lives here: a second hand-written
/// tokenizer to compare against is worth more than the certainty that removing
/// it changed nothing.
#[test]
fn spm_corpus_catches_the_greedy_longest_match_it_replaced() {
    let mut diverged = Vec::new();
    for (model, _, _) in SPM_FIXTURES {
        let Some(tok) = load_spm_or_skip(model) else {
            continue;
        };
        for c in read_corpus() {
            let greedy = greedy_longest_match(&tok, &c.text);
            let port = tok.encode(&c.text, false, false);
            if greedy != port {
                diverged.push(c.name);
            }
        }
    }
    diverged.sort();
    eprintln!(
        "greedy longest-match diverges on {} cases: {diverged:?}",
        diverged.len()
    );
    assert_eq!(
        diverged, SPM_MERGE_ORDER_SENSITIVE,
        "the set of cases where greedy longest-match disagrees with llama.cpp's \
         SPM merge changed. Cases that newly diverge are worth recording: they \
         mean the corpus is exercising a shape the port handles differently than \
         the algorithm it replaced, and that difference is invisible without this \
         check."
    );
}

/// `engine/src/main.rs`'s old `spm::encode`, transcribed. Greedy: at each
/// position take the longest vocabulary entry that starts here (up to a 24
/// character cap), otherwise emit the character's bytes as `<0xNN>`.
fn greedy_longest_match(tok: &Spm, text: &str) -> Vec<i32> {
    const CAP: usize = 24;
    let mut escaped = String::from("\u{2581}");
    escaped.push_str(&text.replace(' ', "\u{2581}"));
    let chars: Vec<char> = escaped.chars().collect();
    let mut ids = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let mut hit = None;
        for l in (1..=CAP.min(chars.len() - i)).rev() {
            let cand: String = chars[i..i + l].iter().collect();
            if tok.piece_id(&cand).is_some() {
                hit = Some(cand);
                break;
            }
        }
        match hit {
            Some(t) => {
                ids.push(tok.piece_id(&t).unwrap());
                i += t.chars().count();
            }
            None => {
                for byte in chars[i].encode_utf8(&mut [0u8; 4]).as_bytes() {
                    ids.push(tok.byte_token(*byte).unwrap_or(0));
                }
                i += 1;
            }
        }
    }
    ids
}

/// A non-BPE vocabulary must be refused by name rather than silently run through
/// the BPE merge table, which is the same failure mode
/// `unsupported_pre_and_model_are_refused_by_name` covers on the other side.
#[test]
fn spm_refuses_non_spm_vocabularies() {
    let md = |model: &str| {
        let mut md = HashMap::new();
        md.insert("tokenizer.ggml.model".into(), Value::Str(model.into()));
        md.insert("tokenizer.ggml.tokens".into(), Value::Array(Vec::new()));
        md
    };
    assert_eq!(
        Spm::from_metadata(&md("gpt2")).err(),
        Some(slow_inference::spm::Error::NotSpm("gpt2".into()))
    );
    assert_eq!(
        Spm::from_metadata(&md("")).err(),
        Some(slow_inference::spm::Error::NotSpm(String::new()))
    );
    // An SPM vocabulary with no token list at all is malformed; an *empty* one
    // is merely useless, and is not the same error.
    assert_eq!(
        Spm::from_metadata(&md("llama")).err(),
        None,
        "an empty token list loads; it is a valid but useless vocabulary"
    );
    let mut no_tokens = HashMap::new();
    no_tokens.insert("tokenizer.ggml.model".into(), Value::Str("llama".into()));
    assert_eq!(
        Spm::from_metadata(&no_tokens).err(),
        Some(slow_inference::spm::Error::Malformed(
            "no tokenizer.ggml.tokens"
        ))
    );
}

/// Corpus cases where greedy longest-match and llama.cpp's SPM merge disagree,
/// measured with [`greedy_longest_match`] against this port.
///
/// Checked as an exact set, like [`OLMO_KNOWN_DIVERGENCES`]: a case appearing or
/// disappearing means either the port changed or the corpus changed, and both
/// deserve a look. It is not only the `spm_*` cases — several older BPE-motivated
/// cases trip it too, and one day they might stop.
///
/// What each group shows:
/// * `spm_merge_order_*` were *chosen* for this, by
///   `scripts/find_spm_discriminators.py` sweeping the vocabulary for texts where
///   the two algorithms differ. Readable words, so the corpus does not look like
///   gibberish.
/// * `double_space`, `dash`, `url`, `newline_between`, `symbols`, `accented` are
///   older cases that trip it incidentally; the disagreement is the same
///   phenomenon (a marker or punctuation run stranded mid-word).
/// * `special_in_word` (`mail<|endoftext|>box`) is the same phenomenon with a
///   sharper difference than it looks. Greedy emits
///   `… end of te xt …` in four tokens; the port emits `… end o ft ex t …` in
///   five. Both tokenize the same characters, greedy merges *more*, and the only
///   way to know which is right is to have an oracle. This case entered the set
///   with the `parse_special` corpus additions.
/// * `spm_control_as_text` and `spm_special_space_prefix` are not really about
///   merge order: greedy has no notion of special tokens and so happily emits
///   id 1 for a literal `<s>`, which llama.cpp refuses to do with
///   `parse_special = false`. For `"<s> hello"` greedy produces
///   `▁ <s> ▁he llo` and the port produces `▁ < s > ▁he llo` — the same text, one
///   token shorter, and a CONTROL token the engine is not allowed to emit.
/// * `empty_text` is a degenerate-input case: the old greedy code prefixed the
///   space marker unconditionally, so it produced one token for an empty prompt.
///   llama.cpp returns none, and the engine now exits instead of running a
///   forward pass over a token it did not intend.
const SPM_MERGE_ORDER_SENSITIVE: &[&str] = &[
    "accented",
    "dash",
    "double_space",
    "empty_text",
    "newline_between",
    "special_in_word",
    "spm_control_as_text",
    "spm_merge_order_development",
    "spm_merge_order_especially",
    "spm_merge_order_sentence",
    "spm_merge_order_yesterday",
    "spm_special_space_prefix",
    "symbols",
    "url",
];

// ---------------------------------------------------------------------------
// special_eog_ids: when does a generation stop
// ---------------------------------------------------------------------------

/// The EOG ids recorded in a `.ref` file, or `None` when the file predates the
/// `EOG` line.
///
/// `None` is a hard error rather than a skip: the ids come from llama.cpp's
/// `llama_vocab_is_eog`, so there is nothing to compare against without them, and
/// silently passing would make this test decorative in exactly the way
/// `parse_special` was before tick 26.
fn read_eog(ref_file: &str) -> Vec<i32> {
    let path = ref_path(ref_file);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut found = None;
    for (lineno, line) in text.lines().enumerate() {
        let Some(rest) = line.strip_prefix("EOG") else {
            continue;
        };
        assert!(
            found.is_none(),
            "{}: two EOG lines (line {})",
            path.display(),
            lineno + 1
        );
        found = Some(
            rest.split_whitespace()
                .map(|t| {
                    t.parse::<i32>()
                        .unwrap_or_else(|e| panic!("{}: bad eog id {t:?}: {e}", path.display()))
                })
                .collect::<Vec<i32>>(),
        );
    }
    found.unwrap_or_else(|| {
        panic!(
            "{}: no EOG line — regenerate with scripts/gen_ref_tokens.sh \
             (llama.cpp's EOG set comes from llama_vocab_is_eog, so the engine \
             has nothing to compare against without it)",
            path.display()
        )
    })
}

/// Every checked-in `.ref` carries llama.cpp's EOG set. Cheap, and it runs in a
/// checkout with no `models/` at all.
#[test]
fn every_reference_file_records_llama_cpp_s_eog_set() {
    for (model, ref_file) in all_ref_fixtures() {
        let mut ids = read_eog(ref_file);
        assert!(
            !ids.is_empty(),
            "{ref_file} ({model}): llama.cpp found no end-of-generation token"
        );
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            ids, sorted,
            "{ref_file}: EOG ids must ascend, with no repeats"
        );
        ids.clear();
    }
}

/// The engine's EOG set is llama.cpp's, id for id, for both vocabulary types.
///
/// This is the check that the derivation in `src/eog.rs` is not an
/// approximation. Three rules in it are load-bearing for exactly these three
/// vocabularies, and each one shows up as a *different number* on the right:
///
/// * Qwen2.5-3B declares no `fim_*_token_id`, so its `<|fim_pad|>`, `<|file_sep|>`
///   and `<|fim_repo|>` enter the set by spelling, from the guarded alias blocks
///   rather than from the EOG list. A port that read only the EOG list would
///   report three ids where llama.cpp reports six. The sixth is id 128247,
///   `</s>`, which the file calls NORMAL — the promotion tick 26 found.
/// * OLMoE has exactly one, `<|endoftext|>`. Its BPE default EOS is 11, so if
///   the GGUF declares `eos_token_id` and the port ignored the key, or if the
///   port applied the default when the key is present, this would read
///   `{11, 50279}` and fail.
/// * Tiny-MoE is SPM, where the default EOS is 2 *and* `</s>` is id 2, so the
///   key and the default agree. It is in the suite because a port that only ever
///   read the EOG list would get that one right by accident.
#[test]
fn eog_set_matches_llama_cpp() {
    let mut checked = 0usize;
    for (model, ref_file, _, _, _) in BPE_FIXTURES {
        let Some(tok) = load_or_skip(model) else {
            continue;
        };
        assert_eq!(
            tok.eog_ids(),
            read_eog(ref_file).as_slice(),
            "{model}: engine EOG set differs from llama_vocab_is_eog"
        );
        // `is_eog` is the accessor the CLI and any sampler use; check it against
        // the set on both sides of every boundary rather than trusting it to
        // delegate.
        for id in -1..(tok.n_vocab() as i32 + 1) {
            assert_eq!(
                tok.is_eog(id),
                tok.eog_ids().contains(&id),
                "{model}: is_eog({id}) disagrees with the set"
            );
        }
        checked += 1;
    }
    for (model, ref_file, _) in SPM_FIXTURES {
        let Some(tok) = load_spm_or_skip(model) else {
            continue;
        };
        assert_eq!(
            tok.eog_ids(),
            read_eog(ref_file).as_slice(),
            "{model}: engine EOG set differs from llama_vocab_is_eog"
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no GGUF on disk: the EOG comparison did not run"
    );
}

/// The EOG set is not the CONTROL set, and the difference is the interesting
/// part: Qwen2.5's `</s>` is CONTROL because of a spelling, and its three
/// FIM ids are CONTROL for the same reason, but none of them is a special token
/// the file declares. Anyone "simplifying" the derivation to
/// `is_control(id) || id == eos_key` fails this first.
#[test]
fn the_eog_set_is_not_just_the_control_tokens() {
    let Some(tok) = load_or_skip("qwen2.5-3b-instruct-q4_k_m.gguf") else {
        return;
    };
    let eog = tok.eog_ids().to_vec();
    assert!(
        eog.len() >= 4,
        "qwen2.5-3b: EOG set has {} ids, expected the FIM trio and more",
        eog.len()
    );
    // Each id's own text is what put it there. Printed as ids and lengths only:
    // these spellings contain `|`, and a rendered `<|...|>` is not evidence of
    // what is in the file.
    for &id in &eog {
        let text = tok.token_text(id).expect("id in range");
        assert!(
            text.starts_with('<') || text.starts_with("</"),
            "qwen2.5-3b: EOG id {id} has text len {} that is not an end marker",
            text.len()
        );
    }
}

/// The stop token is part of the output, so `--stop-on-eog` truncates *after*
/// it. Pure function, no GGUF: the interesting assertion is the boundary.
#[test]
fn truncation_keeps_the_stop_token() {
    let eog = slow_inference::eog::Eog {
        ids: vec![4],
        forced: Vec::new(),
    };
    assert_eq!(slow_inference::eog::stop_at_eog(&[1, 4, 5], &eog), Some(1));
    assert_eq!(slow_inference::eog::stop_at_eog(&[1, 2, 3], &eog), None);
}

// ---------------------------------------------------------------------------
// the two attribute assignments at the end of llama.cpp's EOG block
// ---------------------------------------------------------------------------
//
// No GGUF on disk has the tokens these need: `<|end|>` next to `<|call|>` and
// `<|return|>` is o200k_harmony, and the tool-response marker next to `</s>` is
// gemma4 / paddleocr. So they are synthetic, and they are tested as
// *tokenization* rather than as a set, because the assignment changes both what
// the engine stops on and which strings the partition cuts.
//
// Alias spellings come from the generated table by index. A hand-typed copy of
// one of them is exactly the bug `scripts/check_alias_literals.py` exists for:
// upstream spells the gemma4 marker without its closing pipe.

const A_END: usize = 2;
const A_RETURN: usize = 3;
const A_CALL: usize = 4;
const A_S: usize = 9;
const A_TOOL_RESPONSE: usize = 19;

fn alias(i: usize) -> &'static str {
    slow_inference::special_aliases::EOG_ALIASES[i]
}

/// The o200k_harmony fix: a vocabulary with the turn markers assigns `<|end|>`
/// USER_DEFINED, so the partition cuts on it even with `parse_special = false`.
///
/// The observable is deliberately the *cut*, not the attribute. An engine that
/// only erased the id from its EOG set would keep cutting on `<|end|>` as
/// CONTROL-with-the-flag-on and pass a test that looks at the set — and would
/// differ from llama.cpp on every harmony prompt.
#[test]
fn harmony_turns_end_into_a_user_defined_cut() {
    let tokens = ["a", "b", alias(A_END), alias(A_RETURN), alias(A_CALL)];
    let types = [
        TYPE_NORMAL,
        TYPE_NORMAL,
        TYPE_CONTROL,
        TYPE_CONTROL,
        TYPE_CONTROL,
    ];
    let with_markers = synth_bpe(&tokens, &types);
    let end_id = 2;

    assert!(
        !with_markers.is_eog(end_id),
        "the turn markers must take the end token out of the EOG set"
    );
    let prompt = format!("a{}b", alias(A_END));
    assert!(
        with_markers.encode(&prompt, false, false).contains(&end_id),
        "USER_DEFINED is cut even with parse_special = false"
    );

    // Without `<|call|>` the token stays CONTROL: still not EOG (it is in the
    // alias list), still cut with the flag on, and *not* cut with it off.
    let plain = synth_bpe(&tokens[..3], &types[..3]);
    assert!(plain.is_eog(end_id));
    assert!(!plain.encode(&prompt, false, false).contains(&end_id));
    assert!(plain.encode(&prompt, false, true).contains(&end_id));
}

/// The gemma4 / paddleocr fix: the tool-response marker assigns `</s>` NORMAL, so
/// it leaves the EOG set *and* leaves `cache_special_tokens`. With
/// `parse_special = true` — which is what a chat template needs — `</s>` is no
/// longer lifted out of the prompt, which is the visible consequence.
#[test]
fn the_tool_response_marker_takes_s_out_of_the_cut_set() {
    let tokens = ["a", "b", alias(A_S), alias(A_TOOL_RESPONSE)];
    let types = [TYPE_NORMAL, TYPE_NORMAL, TYPE_CONTROL, TYPE_CONTROL];
    let fixed = synth_bpe(&tokens, &types);
    let s_id = 2;

    assert!(
        !fixed.is_eog(s_id),
        "the tool-response marker must take the end token out of the EOG set"
    );
    let prompt = format!("a{}b", alias(A_S));
    assert!(
        !fixed.encode(&prompt, false, true).contains(&s_id),
        "a NORMAL token is not cut out of the raw text"
    );

    // Without the marker `</s>` is CONTROL and is both EOG and cut.
    let plain = synth_bpe(&tokens[..3], &types[..3]);
    assert!(plain.is_eog(s_id));
    assert!(plain.encode(&prompt, false, true).contains(&s_id));
}

/// The default EOS id is not decoration: a BPE vocabulary that declares no
/// `eos_token_id` still stops on id 11, and an SPM one on id 2. Read only from
/// the key, the engine would stop on nothing at all for half the vocabularies in
/// the wild.
#[test]
fn the_default_eos_id_is_in_the_set_without_a_key() {
    let filler: Vec<String> = (0..12).map(|i| format!("f{i}")).collect();
    let refs: Vec<&str> = filler.iter().map(String::as_str).collect();
    let types = vec![TYPE_NORMAL; refs.len()];
    assert_eq!(synth_bpe(&refs, &types).eog_ids(), &[11]);
    // A vocabulary too small for 11 has it disabled instead, which is the other
    // half of the range check.
    let tiny = synth_bpe(&["a"], &[TYPE_NORMAL]);
    assert!(tiny.eog_ids().is_empty());
}
