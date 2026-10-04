//! GGUF tokenizer, ported from llama.cpp's `llama-vocab.cpp` + `unicode.cpp`.
//!
//! The previous tokenizer lived in `main.rs` and hand-rolled a "simplified"
//! GPT-2 pre-tokenizer regex. That is not a paraphrase of the real thing, and
//! a pre-tokenizer is exactly the component where a paraphrase changes token
//! ids: it decides which byte runs are visible to the merge table. Worse, the
//! whole `tokenizer.ggml.pre` metadata key was ignored, even though every real
//! BPE GGUF sets it and it selects the regex.
//!
//! What is on disk right now (`models/`, dumped from the GGUF headers):
//!
//! | model                | `tokenizer.ggml.model` | `tokenizer.ggml.pre` |
//! |----------------------|------------------------|-----------------------|
//! | Qwen2.5-3B-instruct  | `gpt2`                 | `qwen2`               |
//! | OLMoE-1B-7B-Instruct | `gpt2`                 | `olmo`                |
//! | Tiny-MoE             | `llama`                | `default`             |
//!
//! `qwen2` and `olmo` select *different* pre-tokenizer regexes, so the old
//! engine tokenized every one of them with GPT-2's.
//!
//! The two regexes that cover those models are hand-written state machines in
//! llama.cpp (`unicode_regex_split_custom_gpt2` / `..._qwen2`) rather than
//! engine-driven regexes, so they port exactly — no regex engine, no Unicode
//! property guesses. Codepoint classification comes from
//! `src/unicode_tables.rs`, generated from llama.cpp's `unicode-data.cpp`.
//! `tests/tokenizer_parity.rs` diffs the result against llama.cpp's own
//! tokenizer over a checked-in corpus.
//!
//! ## The step before the regex: `tokenizer_st_partition`
//!
//! The pre-tokenizer is not the first thing that sees the prompt.
//! `llama_vocab::impl::tokenize` first splits the raw text on every *special*
//! token in the vocabulary — `tokenizer.ggml.token_type` 2 (UNKNOWN), 3
//! (CONTROL) or 4 (USER_DEFINED) — and only pre-tokenizes the text fragments
//! that fall between them, longest special token first. A special fragment is
//! emitted as its own id and never reaches the regex or the merge table.
//!
//! That is invisible on a vocabulary whose special tokens are all control
//! markers, because `llama_tokenize(..., parse_special = false)` skips CONTROL
//! and UNKNOWN. It is *not* invisible on OLMoE: ids 50254–50276 are runs of
//! 2 to 24 literal spaces marked USER_DEFINED, so they partition. A run of two
//! or more spaces is emitted as a single whitespace token and the word after it
//! keeps no leading space — which is what this engine used to get wrong on six
//! corpus cases, having concluded for two ticks that llama.cpp's OLMo path fell
//! through to a `std::regex` over collapsed codepoints. It does not: OLMo's
//! `regex_exprs` is byte-identical to GPT-2's and does reach
//! `unicode_regex_split_custom_gpt2`. The divergence was this partition.
//!
//! `parse_special` is the other half of the same switch and it is a parameter
//! rather than a constant: with it true the partition cuts on CONTROL and
//! UNKNOWN as well, and a matched special is emitted as its own id.
//! `llama-server` tokenizes that way (`tokenize_mixed(vocab, text, true,
//! true)`) and a bare `--prompt` arguably should not, so [`Bpe::encode`] takes
//! the flag and `main.rs` exposes it as `--parse-special`. All four
//! combinations of the two booleans are checked against llama.cpp for three real
//! vocabularies by `tests/tokenizer_parity.rs`.
//!
//! ## Scope
//!
//! BPE (`tokenizer.ggml.model = "gpt2"`) with the `gpt2`, `olmo`, `qwen2`,
//! `stablelm2`, `gpt-4o` and `llama3` pre-tokenizers. Every other
//! `tokenizer.ggml.pre` — and SPM/WPM/UGM/RWKV/PLAMO2 vocabularies — returns
//! [`Error::UnsupportedPre`] / [`Error::UnsupportedModel`] rather than
//! silently tokenizing them wrongly. The SPM path (`model = "llama"`) is still
//! the old greedy longest-match in `main.rs` and is *known* not to match
//! llama.cpp's bigram-Viterbi SPM.

use crate::gguf;
use crate::unicode_tables::{
    self, UNICODE_MAP_LOWERCASE, UNICODE_RANGES_FLAGS, UNICODE_SET_WHITESPACE,
};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;

// ---------------------------------------------------------------------------
// unicode_cpt_flags (llama.cpp src/unicode.h)
// ---------------------------------------------------------------------------

/// Flag bits from `unicode_cpt_flags` in llama.cpp's `src/unicode.h`. Only the
/// four the pre-tokenizer state machines consult are named.
const FLAG_NUMBER: u16 = 0x0002;
const FLAG_LETTER: u16 = 0x0004;
const FLAG_WHITESPACE: u16 = 0x0100;

/// `unicode_cpt_flags_from_cpt`: binary search the run-length-encoded table,
/// then OR in the WHITESPACE helper flag.
///
/// The two halves come from different tables in llama.cpp: `unicode.cpp`'s
/// `unicode_cpt_flags_array()` copies `unicode_ranges_flags` (general
/// categories only) and *then* sets `is_whitespace` for every member of
/// `unicode_set_whitespace`. Reading only the ranges table makes U+0020 a
/// separator rather than whitespace, which silently breaks every pre-tokenizer
/// boundary rule that tests `\s` — the GPT-2 splitter then swallows the space
/// after a comma into the punctuation word.
///
/// Codepoints past `MAX_CODEPOINTS` are UNDEFINED, i.e. all flags zero — the
/// pre-tokenizer state machines rely on that to stop scanning.
pub fn flags_from_cpt(cpt: u32) -> u16 {
    if cpt >= unicode_tables::MAX_CODEPOINTS {
        return 0;
    }
    let base = match UNICODE_RANGES_FLAGS.binary_search_by(|&(start, _)| start.cmp(&cpt)) {
        Ok(i) => UNICODE_RANGES_FLAGS[i].1,
        // `i` is the insertion point: the run starting at i-1 covers `cpt`.
        Err(0) => 0,
        Err(i) => UNICODE_RANGES_FLAGS[i - 1].1,
    };
    if UNICODE_SET_WHITESPACE.binary_search(&cpt).is_ok() {
        base | FLAG_WHITESPACE
    } else {
        base
    }
}

/// `unicode_tolower`: lower-case map lookup, identity when absent.
pub fn tolower(cpt: u32) -> u32 {
    match UNICODE_MAP_LOWERCASE.binary_search_by(|&(from, _)| from.cmp(&cpt)) {
        Ok(i) => UNICODE_MAP_LOWERCASE[i].1,
        Err(_) => cpt,
    }
}

/// `unicode_byte_to_utf8_map`: the GPT-2 bytes-to-unicode table, built the
/// same way llama.cpp builds it — the printable ASCII/Latin-1 bytes keep
/// themselves, and the remaining bytes are handed out codepoints 256, 257, ...
/// in increasing byte order.
pub fn byte_to_unicode_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut taken = [false; 256];
    for ch in 0x21u32..=0x7E {
        table[ch as usize] = char::from_u32(ch).unwrap();
        taken[ch as usize] = true;
    }
    for ch in 0xA1u32..=0xAC {
        table[ch as usize] = char::from_u32(ch).unwrap();
        taken[ch as usize] = true;
    }
    for ch in 0xAEu32..=0xFF {
        table[ch as usize] = char::from_u32(ch).unwrap();
        taken[ch as usize] = true;
    }
    let mut n = 0u32;
    for ch in 0..256usize {
        if !taken[ch] {
            table[ch] = char::from_u32(256 + n).expect("byte-unicode table stays in range");
            n += 1;
        }
    }
    table
}

/// Map every byte of a word through [`byte_to_unicode_table`] — llama.cpp's
/// `unicode_byte_encoding_process`, which byte-encodes *after* splitting.
fn byte_encode_word(word: &str, table: &[char; 256]) -> String {
    word.bytes().map(|b| table[b as usize]).collect::<String>()
}

// ---------------------------------------------------------------------------
// pre-tokenizer regexes (llama.cpp src/unicode.cpp)
// ---------------------------------------------------------------------------

/// Which pre-tokenizer regex `tokenizer.ggml.pre` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreType {
    /// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
    Gpt2,
    /// `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
    Qwen2,
    /// Same as Qwen2 but letter runs also absorb `\p{M}` combining marks.
    Qwen35,
    /// The Llama-3 regex; differs from Qwen2 only in `\p{N}{1,3}`.
    Llama3,
}

/// Map `tokenizer.ggml.pre` to a regex, mirroring the `pre_type` switch in
/// `llama_vocab::impl::load`. Returns the pre-tokenizer string on failure so
/// the error can name the exact unsupported value.
pub fn pre_type_from_str(pre: &str) -> Result<PreType, String> {
    Ok(match pre {
        // llama.cpp warns and uses LLAMA_VOCAB_PRE_TYPE_DEFAULT when the key is
        // absent; "default" itself is that same 4-expression regex.
        "" | "default" => return Err(pre.to_string()),
        // `olmo` really is in this group: llama.cpp's pre_type switch gives
        // OLMo, GPT-2, MPT, Jais, Trillion, Granite-Docling and the three
        // jina/roberta BPEs one regex_exprs entry.
        "gpt2" | "mpt" | "olmo" | "olmo2" | "jais" | "trillion" | "granite-docling"
        | "jina-v1-en" | "jina-v2-code" | "roberta-bpe" => PreType::Gpt2,
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "stablelm2" | "hunyuan"
        | "solar-open" => PreType::Qwen2,
        "qwen35" => PreType::Qwen35,
        "llama3" | "dbrx" | "smaug-bpe" | "chatglm4" | "granite-embed-multi-311m" => {
            PreType::Llama3
        }
        _ => return Err(pre.to_string()),
    })
}

const OUT_OF_RANGE: u32 = 0xFFFF_FFFF;

// ---------------------------------------------------------------------------
// special-token partition (llama.cpp src/llama-vocab.cpp)
// ---------------------------------------------------------------------------

/// `llama_token_type` values that put a token in `cache_special_tokens`, i.e.
/// make `tokenizer_st_partition` cut the raw text on it.
const TOKEN_TYPE_UNKNOWN: u64 = 2;
const TOKEN_TYPE_CONTROL: u64 = 3;
const TOKEN_TYPE_USER_DEFINED: u64 = 4;

/// One entry of llama.cpp's `cache_special_tokens`.
struct Special {
    text: String,
    id: i32,
    /// `true` for USER_DEFINED, `false` for UNKNOWN / CONTROL.
    ///
    /// `tokenizer_st_partition` skips UNKNOWN and CONTROL when
    /// `parse_special == false` and cuts on USER_DEFINED either way — see the
    /// note above about OLMoE's whitespace runs.
    user_defined: bool,
    /// `LLAMA_TOKEN_ATTR_LSTRIP` / `LLAMA_TOKEN_ATTR_RSTRIP`.
    ///
    /// Both trim whitespace off the *raw text fragment* next to this token
    /// before pre-tokenization, so unlike every other attribute here they
    /// change what a neighbouring fragment encodes to rather than whether the
    /// token is a cut point. Set only by [`derive_strips`].
    lstrip: bool,
    rstrip: bool,
}

/// A slice of the prompt: either text to pre-tokenize or a special token to
/// emit verbatim. `llama_vocab::impl::fragment_buffer_variant` holds the same
/// two cases.
///
/// Public because the strip rules are only observable here. `encode` collapses
/// fragments to ids, and on a vocabulary whose merge table cannot encode the
/// surrounding text — every synthetic fixture, and the strip fixture included —
/// the ids are identical whether or not a flag was honoured. The fragment list
/// is the thing under test, so it is the thing a test must be able to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fragment {
    Raw(String),
    Special(i32),
}

/// Split `text` into pre-token words (byte spans of the codepoint list),
/// following `unicode_regex_split_custom_gpt2`.
fn split_custom_gpt2(cpts: &[u32]) -> Vec<(usize, usize)> {
    let n = cpts.len();
    let flags_at = |i: usize| -> u16 { if i < n { flags_from_cpt(cpts[i]) } else { 0 } };
    let cpt_at = |i: usize| -> u32 { if i < n { cpts[i] } else { OUT_OF_RANGE } };
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut prev_end = 0usize;
    let mut pos = 0usize;

    while pos < n {
        let cpt = cpt_at(pos);
        let flags = flags_at(pos);

        // regex: 's|'t|'re|'ve|'m|'ll|'d
        if cpt == b'\'' as u32 && pos + 1 < n {
            let next = cpt_at(pos + 1);
            if next == b's' as u32
                || next == b't' as u32
                || next == b'm' as u32
                || next == b'd' as u32
            {
                pos += emit(&mut out, &mut prev_end, pos + 2);
                continue;
            }
            if pos + 2 < n {
                let next2 = cpt_at(pos + 2);
                if (next == b'r' as u32 && next2 == b'e' as u32)
                    || (next == b'v' as u32 && next2 == b'e' as u32)
                    || (next == b'l' as u32 && next2 == b'l' as u32)
                {
                    pos += emit(&mut out, &mut prev_end, pos + 3);
                    continue;
                }
            }
        }

        let mut flags2 = if cpt == b' ' as u32 {
            flags_at(pos + 1)
        } else {
            flags
        };

        // regex: <space>?\p{L}+
        if flags2 & FLAG_LETTER != 0 {
            pos += usize::from(cpt == b' ' as u32);
            while flags2 & FLAG_LETTER != 0 {
                pos += 1;
                flags2 = flags_at(pos);
            }
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: <space>?\p{N}+
        if flags2 & FLAG_NUMBER != 0 {
            pos += usize::from(cpt == b' ' as u32);
            while flags2 & FLAG_NUMBER != 0 {
                pos += 1;
                flags2 = flags_at(pos);
            }
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: <space>?[^\s\p{L}\p{N}]+
        if flags2 & (FLAG_WHITESPACE | FLAG_LETTER | FLAG_NUMBER) == 0 && flags2 != 0 {
            pos += usize::from(cpt == b' ' as u32);
            while flags2 & (FLAG_WHITESPACE | FLAG_LETTER | FLAG_NUMBER) == 0 && flags2 != 0 {
                pos += 1;
                flags2 = flags_at(pos);
            }
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        let mut num_ws = 0usize;
        while flags_at(pos + num_ws) & FLAG_WHITESPACE != 0 {
            num_ws += 1;
        }

        // regex: \s+(?!\S)
        if num_ws > 1 && pos + num_ws < n {
            pos += num_ws - 1;
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: \s+
        if num_ws > 0 {
            pos += num_ws;
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        // no matches
        pos += 1;
        emit(&mut out, &mut prev_end, pos);
    }

    out
}

/// Split `text` into pre-token words, following
/// `unicode_regex_split_custom_qwen2`.
fn split_custom_qwen2(cpts: &[u32]) -> Vec<(usize, usize)> {
    let n = cpts.len();
    let flags_at = |i: usize| -> u16 { if i < n { flags_from_cpt(cpts[i]) } else { 0 } };
    let cpt_at = |i: usize| -> u32 { if i < n { cpts[i] } else { OUT_OF_RANGE } };
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut prev_end = 0usize;
    let mut pos = 0usize;

    while pos < n {
        let cpt = cpt_at(pos);
        let flags = flags_at(pos);

        // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)  — the only case-insensitive
        // alternative in this regex, which is why Qwen2 needs its own
        // splitter instead of reusing the GPT-2 one.
        if cpt == b'\'' as u32 && pos + 1 < n {
            let next = tolower(cpt_at(pos + 1));
            if next == b's' as u32
                || next == b't' as u32
                || next == b'm' as u32
                || next == b'd' as u32
            {
                pos += emit(&mut out, &mut prev_end, pos + 2);
                continue;
            }
            if pos + 2 < n {
                let next2 = tolower(cpt_at(pos + 2));
                if (next == b'r' as u32 && next2 == b'e' as u32)
                    || (next == b'v' as u32 && next2 == b'e' as u32)
                    || (next == b'l' as u32 && next2 == b'l' as u32)
                {
                    pos += emit(&mut out, &mut prev_end, pos + 3);
                    continue;
                }
            }
        }

        // regex: [^\r\n\p{L}\p{N}]?\p{L}+
        if !(cpt == b'\r' as u32 || cpt == b'\n' as u32 || flags & FLAG_NUMBER != 0)
            && (flags & FLAG_LETTER != 0 || flags_at(pos + 1) & FLAG_LETTER != 0)
        {
            pos += 1;
            while flags_at(pos) & FLAG_LETTER != 0 {
                pos += 1;
            }
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: \p{N}
        if flags & FLAG_NUMBER != 0 {
            pos += 1;
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
        let mut flags2 = if cpt == b' ' as u32 {
            flags_at(pos + 1)
        } else {
            flags
        };
        if flags2 & (FLAG_WHITESPACE | FLAG_LETTER | FLAG_NUMBER) == 0 && flags2 != 0 {
            pos += usize::from(cpt == b' ' as u32);
            while flags2 & (FLAG_WHITESPACE | FLAG_LETTER | FLAG_NUMBER) == 0 && flags2 != 0 {
                pos += 1;
                flags2 = flags_at(pos);
            }
            let mut cpt2 = cpt_at(pos);
            while cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                pos += 1;
                cpt2 = cpt_at(pos);
            }
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        let mut num_ws = 0usize;
        let mut last_end_r_or_n = 0usize;
        while flags_at(pos + num_ws) & FLAG_WHITESPACE != 0 {
            let cpt2 = cpt_at(pos + num_ws);
            if cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                last_end_r_or_n = pos + num_ws + 1;
            }
            num_ws += 1;
        }

        // regex: \s*[\r\n]+
        if last_end_r_or_n > 0 {
            pos = last_end_r_or_n;
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: \s+(?!\S)
        if num_ws > 1 && pos + num_ws < n {
            pos += num_ws - 1;
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: \s+
        if num_ws > 0 {
            pos += num_ws;
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        // no matches
        pos += 1;
        emit(&mut out, &mut prev_end, pos);
    }

    out
}

/// llama.cpp's `_add_token` helper: close the span `[prev_end, end)` if it is
/// non-empty, advance the cursor, and return how many codepoints it consumed.
fn emit(out: &mut Vec<(usize, usize)>, prev_end: &mut usize, end: usize) -> usize {
    let len = end - *prev_end;
    if len > 0 {
        out.push((*prev_end, end));
    }
    *prev_end = end;
    len
}

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// `tokenizer.ggml.pre` names a pre-tokenizer this engine does not
    /// implement. Carries the offending value.
    UnsupportedPre(String),
    /// `tokenizer.ggml.model` is not `gpt2`. Carries the offending value.
    UnsupportedModel(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnsupportedPre(p) => write!(
                f,
                "pre-tokenizer '{p}' is not implemented (see src/tokenizer.rs)"
            ),
            Error::UnsupportedModel(m) => {
                write!(
                    f,
                    "tokenizer model '{m}' is not implemented (only gpt2/BPE)"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

// ---------------------------------------------------------------------------
// the tokenizer
// ---------------------------------------------------------------------------

/// A byte-level BPE tokenizer loaded from a GGUF's metadata.
///
/// The pre-tokenizer regex and the merge table are the whole of the algorithm;
/// everything else is bookkeeping llama.cpp also does (BOS/EOS insertion,
/// byte-fallback for tokens missing from the vocabulary).
pub struct Bpe {
    pre: PreType,
    /// `tokenizer.ggml.tokens` -> id
    token_to_id: HashMap<String, i32>,
    /// `tokenizer.ggml.tokens` by id, after the zero-length rewrite.
    texts: Vec<String>,
    /// `(left, right)` -> merge rank, where rank 0 merges first.
    ranks: HashMap<(String, String), u32>,
    byte_table: [char; 256],
    bos: Option<i32>,
    add_bos: bool,
    ignore_merges: bool,
    /// `cache_special_tokens`, longest text first.
    specials: Vec<Special>,
    /// `special_eog_ids`, derived from token text rather than from any key.
    eog: crate::eog::Eog,
}

/// `isspace` in the "C" locale, which is what llama.cpp calls: the six ASCII
/// whitespace bytes and nothing else. Not `char::is_whitespace`, which also
/// accepts U+00A0, U+2028 and the Unicode space separators — so a port that used
/// it would strip a non-breaking space that upstream leaves in the fragment.
///
/// Working in bytes rather than `char`s is what makes the loop safe: an
/// arbitrary byte is never whitespace, so `left_end` and `right_start` can only
/// ever land on a character boundary.
fn is_llama_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

impl Bpe {
    /// Load a BPE tokenizer from GGUF metadata.
    ///
    /// `md` is the metadata map of a `gguf::GGUFFile`.
    pub fn from_metadata(md: &HashMap<String, gguf::Value>) -> Result<Bpe, Error> {
        let model = md
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if model != "gpt2" {
            return Err(Error::UnsupportedModel(model.to_string()));
        }
        let pre_str = md
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let pre = pre_type_from_str(pre_str).map_err(Error::UnsupportedPre)?;

        let tokens = md
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_vec_str())
            .ok_or_else(|| Error::UnsupportedPre("no tokenizer.ggml.tokens".into()))?;
        // llama.cpp rewrites a zero-length token at load time
        // (`llama-vocab.cpp`: `if (word.empty()) word = "[EMPTY_" + i + "]"`).
        // Without it an empty special token matches at every position of the
        // partition loop; the sentinel is unreachable in real text, so the
        // loop terminates and no empty token can ever be emitted.
        let token_text: Vec<String> = tokens
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if s.is_empty() {
                    format!("[EMPTY_{i}]")
                } else {
                    (*s).to_string()
                }
            })
            .collect();
        let token_to_id: HashMap<String, i32> = token_text
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as i32))
            .collect();

        let merges = md
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_vec_str())
            .unwrap_or_default();
        let mut ranks: HashMap<(String, String), u32> = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            // llama.cpp splits at the first space *after* index 0, so a leading
            // space in either half is preserved. Both merge halves may be
            // multi-byte (the byte-unicode alphabet has 2-byte chars), so scan
            // bytes rather than chars: 0x20 never occurs inside a UTF-8
            // sequence, which makes both cut points char boundaries.
            let Some(pos) = m.as_bytes()[1..]
                .iter()
                .position(|&b| b == b' ')
                .map(|i| i + 1)
            else {
                continue;
            };
            ranks.insert(
                (m[..pos].to_string(), m[pos + 1..].to_string()),
                rank as u32,
            );
        }

        let bos = md
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as i32);
        // llama.cpp's default for BPE is `add_bos = false`; only an explicit
        // `tokenizer.ggml.add_bos_token` key turns it on.
        let add_bos = md
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| match v {
                gguf::Value::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);
        let ignore_merges = md
            .get("tokenizer.ggml.ignore_merges")
            .and_then(|v| match v {
                gguf::Value::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(false);

        // `special_eog_ids` first, because the two workaround blocks in it
        // assign attributes that `build_specials` has to honour, and it needs
        // the same rewritten texts and promotion pass to do that.
        let promotions = special_text_promotions(&token_text, md);
        let eog = crate::eog::derive(&token_text, md, &promotions);
        // Then `cache_special_tokens`, and only then the strip flags — upstream's
        // order, and load-bearing: the phi-3 branch iterates the built set.
        let mut specials = build_specials(md, &eog);
        derive_strips(md, &mut specials);

        Ok(Bpe {
            pre,
            token_to_id,
            texts: token_text,
            ranks,
            byte_table: byte_to_unicode_table(),
            bos,
            add_bos,
            ignore_merges,
            specials,
            eog,
        })
    }

    /// The ids carrying `LLAMA_TOKEN_ATTR_LSTRIP` / `RSTRIP`, ascending.
    ///
    /// This is a projection of the attribute, and deliberately so. The full
    /// `llama_token_attr` mask is the better oracle and `engine/tools/
    /// gen_ref_attrs.c` dumps it, but the engine does not model SINGLE_WORD or
    /// BYTE, so comparing the whole mask would fail on bits outside this
    /// component and say nothing about the strips. What is compared here is the
    /// part that is implemented.
    pub fn strip_ids(&self) -> (Vec<i32>, Vec<i32>) {
        let mut lstrip: Vec<i32> = self
            .specials
            .iter()
            .filter(|s| s.lstrip)
            .map(|s| s.id)
            .collect();
        let mut rstrip: Vec<i32> = self
            .specials
            .iter()
            .filter(|s| s.rstrip)
            .map(|s| s.id)
            .collect();
        lstrip.sort_unstable();
        rstrip.sort_unstable();
        (lstrip, rstrip)
    }

    /// The whole set, for callers that need `stop_at_eog`.
    pub fn eog(&self) -> &crate::eog::Eog {
        &self.eog
    }

    /// `llama_vocab::impl::is_eog`: does this token end a generation?
    pub fn is_eog(&self, id: i32) -> bool {
        self.eog.is_eog(id)
    }

    /// The ids that end a generation, ascending.
    pub fn eog_ids(&self) -> &[i32] {
        &self.eog.ids
    }

    /// Number of entries in the special-token partition, for tests.
    pub fn n_special_tokens(&self) -> usize {
        self.specials.len()
    }

    /// The vocabulary text of `id`, after the zero-length rewrite.
    ///
    /// `ids_for` maps strings to ids, not the other way round, so this is also
    /// the only way to see what the rewrite did — llama.cpp logged the same
    /// string as `empty token at index N` at load time.
    pub fn token_text(&self, id: i32) -> Option<&str> {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.texts.get(i))
            .map(String::as_str)
    }

    /// The selected pre-tokenizer, for reporting and tests.
    pub fn pre_type(&self) -> PreType {
        self.pre
    }

    /// Number of entries in the merge table.
    pub fn n_merges(&self) -> usize {
        self.ranks.len()
    }

    /// Size of the vocabulary.
    pub fn n_vocab(&self) -> usize {
        self.token_to_id.len()
    }

    /// The pre-token words of `text`, byte-encoded — the unit the merge table
    /// sees. Exposed because it is the piece most worth diffing by eye.
    pub fn pretokenize(&self, text: &str) -> Vec<String> {
        let cpts: Vec<u32> = text.chars().map(u32::from).collect();
        let spans = match self.pre {
            PreType::Gpt2 => split_custom_gpt2(&cpts),
            PreType::Qwen2 | PreType::Qwen35 => split_custom_qwen2(&cpts),
            PreType::Llama3 => split_custom_llama3(&cpts),
        };
        spans
            .into_iter()
            .map(|(a, b)| {
                let word: String = cpts[a..b]
                    .iter()
                    .map(|&c| char::from_u32(c).unwrap())
                    .collect();
                byte_encode_word(&word, &self.byte_table)
            })
            .collect()
    }

    /// `llama_vocab::impl::tokenizer_st_partition`: cut `text` on the
    /// vocabulary's special tokens, longest text first, before any
    /// pre-tokenization happens.
    ///
    /// `parse_special` is llama.cpp's fourth argument to `llama_tokenize`.
    /// When it is `false` — which is what a bare prompt is, and what this
    /// engine's CLI passes unless `--parse-special` is given — llama.cpp skips
    /// UNKNOWN and CONTROL tokens and cuts only on USER_DEFINED ones. The
    /// distinction is load-bearing: OLMoE marks 23 whitespace runs USER_DEFINED,
    /// so they are cut on, while Qwen2's `<|im_start|>` family is CONTROL and is
    /// not. When it is `true` llama.cpp cuts on all three and emits the matched
    /// token as its own id, which is what `llama-server` does.
    pub fn partition(&self, text: &str, parse_special: bool) -> Vec<Fragment> {
        let mut frags = vec![Fragment::Raw(text.to_string())];
        for sp in &self.specials {
            if !parse_special && !sp.user_defined {
                continue;
            }
            let needle = sp.text.as_bytes();
            let mut i = 0usize;
            while i < frags.len() {
                let Fragment::Raw(raw) = frags[i].clone() else {
                    i += 1;
                    continue;
                };
                // llama.cpp keeps scanning the remainder to the right of each
                // match within the same fragment, so this is one loop over the
                // fragment rather than one `find`.
                let mut out: Vec<Fragment> = Vec::new();
                let bytes = raw.as_bytes();
                let mut base = 0usize;
                while base < bytes.len() {
                    let Some(at) = raw[base..].find(sp.text.as_str()) else {
                        break;
                    };
                    let start = base + at;
                    let end = start + needle.len();
                    // LSTRIP: pull the left reminder's end back over trailing
                    // whitespace, and emit it only if what is left is non-empty
                    // (llama-vocab.cpp:3214). The post-strip emptiness test is
                    // the load-bearing half — a run of spaces *only* before the
                    // token must vanish, not survive as an empty fragment.
                    let mut left_end = start;
                    if sp.lstrip {
                        while left_end > base && is_llama_space(bytes[left_end - 1]) {
                            left_end -= 1;
                        }
                    }
                    if left_end > base {
                        out.push(Fragment::Raw(raw[base..left_end].to_string()));
                    }
                    out.push(Fragment::Special(sp.id));
                    // RSTRIP: advance the right reminder's start over
                    // whitespace. No rule can currently set it (see
                    // `derive_strips`), but the branch belongs here or the flag
                    // would be accepted and silently ignored — which is how a
                    // port ends up claiming support for something it does not
                    // do.
                    let mut right_start = end;
                    if sp.rstrip {
                        while right_start < bytes.len() && is_llama_space(bytes[right_start]) {
                            right_start += 1;
                        }
                    }
                    // Upstream does NOT emit the right reminder here. It moves
                    // its search window to the right reminder's offset and keeps
                    // looking *inside* that same text, erasing what it emitted
                    // for the previous match if it finds something
                    // (llama-vocab.cpp:3239, then the `buffer.erase_after` at
                    // 3253). So `<|im_end|><|im_end|>` is two special ids and no
                    // text between them — emitting the right reminder eagerly and
                    // re-searching it duplicates it and feeds the second copy to
                    // the pre-tokenizer.
                    base = right_start;
                }
                if base < raw.len() {
                    out.push(Fragment::Raw(raw[base..].to_string()));
                }
                if out.is_empty() {
                    i += 1;
                    continue;
                }
                let added = out.len();
                frags.splice(i..=i, out);
                i += added;
            }
        }
        frags
    }

    /// Encode `text` to token ids, matching
    /// `llama_tokenize(vocab, text, add_special, parse_special)`.
    ///
    /// `add_special` prepends BOS when the GGUF asks for it. `parse_special`
    /// selects the vocabulary's CONTROL and UNKNOWN tokens as partition
    /// boundaries — see [`Bpe::partition`].
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<i32> {
        let mut out: Vec<i32> = Vec::new();
        if add_special
            && self.add_bos
            && let Some(bos) = self.bos
        {
            out.push(bos);
        }
        for frag in self.partition(text, parse_special) {
            match frag {
                Fragment::Special(id) => out.push(id),
                Fragment::Raw(raw) => {
                    for word in self.pretokenize(&raw) {
                        out.extend(self.bpe_word(&word));
                    }
                }
            }
        }
        out
    }

    /// `llm_tokenizer_bpe_session::tokenize`'s per-word half: merge the word's
    /// symbols lowest-rank-first, then map each surviving symbol to an id.
    fn bpe_word(&self, word: &str) -> Vec<i32> {
        // Symbol `i` is the byte range `starts[i] .. starts[i] + len[i]`, so a
        // merge is just `len[left] += len[right]` — the strings are only
        // materialised when a merge rank has to be looked up.
        let starts: Vec<usize> = word.char_indices().map(|(i, _)| i).collect();
        let mut len: Vec<usize> = word.chars().map(char::len_utf8).collect();
        if starts.is_empty() {
            return Vec::new();
        }
        if self.ignore_merges && self.token_to_id.contains_key(word) {
            return self.ids_for(word);
        }

        // prev/next links over the symbols; -1 terminates.
        let mut prev: Vec<i32> = (0..starts.len() as i32).map(|i| i - 1).collect();
        let mut next: Vec<i32> = (1..starts.len() as i32).collect();
        next.push(-1);

        let mut queue: BinaryHeap<Reverse<Bigram>> = BinaryHeap::new();
        for i in 0..starts.len() - 1 {
            push_bigram(self, &mut queue, word, &starts, &len, i, i + 1);
        }

        while let Some(Reverse(bg)) = queue.pop() {
            let (l, r) = (bg.left as usize, bg.right as usize);
            // Stale-entry guard. llama.cpp compares `left + right` against the
            // text captured at push time and skips the pair if they differ; a
            // symbol's text is fully determined by its start offset and length,
            // so comparing the two lengths is the same test. Adjacency is
            // checked too: a pair can stay contiguous-looking in the byte range
            // while no longer being neighbours in the symbol list.
            if len[l] != bg.left_len || len[r] != bg.right_len || next[l] != bg.right {
                continue;
            }
            if std::env::var_os("SLOW_TOKENIZER_TRACE").is_some() {
                eprintln!(
                    "    merge {:?} + {:?} -> {:?} (rank {})",
                    &word[starts[l]..starts[l] + len[l]],
                    &word[starts[r]..starts[r] + len[r]],
                    &word[starts[l]..starts[l] + len[l] + len[r]],
                    bg.rank
                );
            }
            let after = next[r];
            len[l] += len[r];
            len[r] = 0;
            next[l] = after;
            if after >= 0 {
                prev[after as usize] = l as i32;
            }
            let before = prev[l];
            if before >= 0 {
                push_bigram(self, &mut queue, word, &starts, &len, before as usize, l);
            }
            if after >= 0 {
                push_bigram(self, &mut queue, word, &starts, &len, l, after as usize);
            }
        }

        let mut out = Vec::new();
        let mut i = 0i32;
        while i != -1 {
            let iu = i as usize;
            if len[iu] != 0 {
                out.extend(self.ids_for(&word[starts[iu]..starts[iu] + len[iu]]));
            }
            i = next[iu];
        }
        out
    }

    /// `vocab.text_to_token`, with llama.cpp's per-byte fallback for symbols
    /// the merge table produced but the vocabulary does not contain.
    fn ids_for(&self, s: &str) -> Vec<i32> {
        if let Some(&id) = self.token_to_id.get(s) {
            return vec![id];
        }
        let mut out = Vec::new();
        for b in s.bytes() {
            let piece = self.byte_table[b as usize].to_string();
            // llama.cpp pushes the byte token only if it exists, so a vocab
            // missing part of the byte alphabet silently drops characters.
            if let Some(&id) = self.token_to_id.get(&piece) {
                out.push(id);
            }
        }
        out
    }
}

/// Ids whose token *spelling* makes llama.cpp override `tokenizer.ggml.token_type`
/// at load time, and by how much.
///
/// This is the part of the attribute pipeline that is not in the file. Upstream
/// reads `token_type`, then runs a set of "auto-detect special tokens by text"
/// blocks over `token_to_id` and ORs CONTROL onto any token spelled like an
/// end-of-text, end-of-message, fill-in-the-middle or end-of-generation marker;
/// a final gpt-oss block does the opposite and forces four tokens to
/// USER_DEFINED. The alias lists are in [`crate::special_aliases`], extracted
/// from the C++ by `scripts/gen_special_aliases.py` because several of them
/// carry characters that do not survive being read off a screen.
///
/// Qwen2.5-3B is the case that matters: it ships `</s>` as id 128247 with
/// `token_type = 1` (NORMAL), and `</s>` is in the end-of-generation alias list.
/// llama.cpp logs
///
/// ```text
/// load: control-looking token: 128247 '</s>' was not control-type;
///       this is probably a bug in the model. its type will be overridden
/// ```
///
/// and treats it as CONTROL. Trusting the file instead makes it 22 specials
/// where llama.cpp has 23.
///
/// The override is *invisible* with `parse_special = false`, because that is
/// precisely the setting in which CONTROL tokens are skipped, so it can sit in
/// the port for as long as nobody tokenizes with `parse_special = true`. It is
/// still not cosmetic: CONTROL also decides whether `llama_token_to_piece`
/// renders a token, and USER_DEFINED decides whether the partition cuts on it
/// with the flag off.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SpecialPromotions {
    /// Forced to CONTROL by one of the alias blocks.
    pub control: Vec<usize>,
    /// Forced to USER_DEFINED by the gpt-oss override, which *replaces* the
    /// attribute rather than adding to it.
    pub user_defined: Vec<usize>,
    /// The id each alias block settled on, as `(ggml slot, id)` — the
    /// `special_eot_id` / `special_fim_pad_id` / … locals upstream. Only blocks
    /// that ran are recorded, i.e. only slots whose `tokenizer.ggml.<slot>_token_id`
    /// key is absent, which is exactly upstream's `== LLAMA_TOKEN_NULL` guard.
    ///
    /// These ids outlive the promotion: `special_eog_ids` inserts `fim_pad`,
    /// `fim_rep` and `fim_sep` by id, and the sanity pass inserts `eot` and
    /// `eom`, so the promotion pass is where they have to be remembered. See
    /// [`crate::eog`].
    pub slot_ids: Vec<(&'static str, usize)>,
}

impl SpecialPromotions {
    fn promote_control(&mut self, id: usize) {
        if !self.control.contains(&id) {
            self.control.push(id);
        }
    }
    fn promote_user_defined(&mut self, id: usize) {
        if !self.user_defined.contains(&id) {
            self.user_defined.push(id);
        }
    }

    /// The id of the first token whose text is in `aliases`, or `None`.
    ///
    /// llama.cpp walks an `std::unordered_map`, so when a vocabulary contains
    /// *two* spellings from the same block the winner is whichever the hash
    /// table happened to yield first — undefined, and not reproducible. This
    /// takes the lowest id, which is deterministic; no vocabulary on disk
    /// contains two spellings from one block, so the difference has never been
    /// observable and is recorded rather than emulated.
    fn first_match(tokens: &[String], aliases: &[&str]) -> Option<usize> {
        tokens.iter().position(|t| aliases.contains(&t.as_str()))
    }
}

/// llama.cpp's "auto-detect special tokens by text" pass, over the *rewritten*
/// token texts (the zero-length rewrite happens first upstream too, so an
/// empty token is `[EMPTY_n]` and cannot match an alias).
///
/// `tokens` must be in id order and the same length as the vocabulary.
pub fn special_text_promotions(
    tokens: &[String],
    md: &HashMap<String, gguf::Value>,
) -> SpecialPromotions {
    let mut out = SpecialPromotions::default();

    // Eight blocks, each suppressed by the GGUF declaring the id it is looking
    // for. A declared id means "the converter already told us", and the block
    // does not run — including its CONTROL promotion, which is the subtle part:
    // a model that declares `tokenizer.ggml.eot_token_id` does *not* get its
    // EOT token promoted by spelling.
    for group in crate::special_aliases::CONTROL_ALIAS_GROUPS {
        let key = format!("tokenizer.ggml.{}_token_id", group.slot);
        if md.contains_key(&key) {
            continue;
        }
        if let Some(id) = SpecialPromotions::first_match(tokens, group.aliases) {
            out.promote_control(id);
            out.slot_ids.push((group.slot, id));
        }
    }

    // Unguarded: the end-of-generation set always runs, and it promotes every
    // alias it finds rather than stopping at the first.
    for (id, text) in tokens.iter().enumerate() {
        if crate::special_aliases::EOG_ALIASES.contains(&text.as_str()) {
            out.promote_control(id);
        }
    }

    // Runs last and assigns rather than ORs, so it wins over the blocks above.
    for (id, text) in tokens.iter().enumerate() {
        if crate::special_aliases::USER_DEFINED_ALIASES.contains(&text.as_str()) {
            out.promote_user_defined(id);
        }
    }

    out
}

/// llama.cpp's `cache_special_tokens`: every token whose
/// `tokenizer.ggml.token_type` is UNKNOWN, CONTROL or USER_DEFINED — *plus*
/// whatever [`special_text_promotions`] promotes — sorted by descending text
/// length.
///
/// Three details are load-bearing and none is obvious from the field names:
///
/// * **Order is longest-first**, so a run of 30 spaces is cut as one 24-space
///   token and then three 2-space tokens rather than fifteen 2-space ones.
/// * llama.cpp's `std::sort` comparator looks only at the length, so the order
///   of equal-length specials is unspecified. This port breaks ties by
///   ascending id, which is deterministic and matches upstream whenever the
///   lengths differ (every vocabulary with ties found so far has none).
/// * the promoted ids from [`special_text_promotions`] join the list even
///   though the file says they are NORMAL. This is the `</s>` bug.
/// * and the two attribute *assignments* in [`crate::eog::derive`] come last.
///   They assign rather than OR, so they win over both the file and the
///   promotion, and upstream runs them before it builds this cache —
///   which is why `</s>` forced NORMAL leaves this list altogether.
fn build_specials(md: &HashMap<String, gguf::Value>, eog: &crate::eog::Eog) -> Vec<Special> {
    let tokens = md
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_vec_str())
        .unwrap_or_default();
    let types = md
        .get("tokenizer.ggml.token_type")
        .and_then(|v| v.as_vec_u64())
        .unwrap_or_default();
    // Upstream keys its alias tables on the rewritten text, and the rewrite
    // happens while the token list is being read.
    let rewritten: Vec<String> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if t.is_empty() {
                format!("[EMPTY_{i}]")
            } else {
                (*t).to_string()
            }
        })
        .collect();
    let promotions = special_text_promotions(&rewritten, md);
    let mut specials: Vec<Special> = rewritten
        .iter()
        .enumerate()
        .filter_map(|(i, text)| {
            let t = types.get(i).copied().unwrap_or(0);
            // `attr = LLAMA_TOKEN_ATTR_NORMAL` — not "clear the bits", assign.
            // A token that arrives here as USER_DEFINED stops being a partition
            // boundary in *both* parse_special settings.
            if eog.forced_attr(i as i32) == Some(crate::eog::Forced::Normal) {
                return None;
            }
            let user_defined = if eog.forced_attr(i as i32) == Some(crate::eog::Forced::UserDefined)
            {
                true
            } else if promotions.user_defined.contains(&i) {
                // The gpt-oss override assigns the attribute outright, so a
                // promoted token is USER_DEFINED even if the file says CONTROL.
                true
            } else if promotions.control.contains(&i) {
                false
            } else {
                match t {
                    TOKEN_TYPE_USER_DEFINED => true,
                    TOKEN_TYPE_CONTROL | TOKEN_TYPE_UNKNOWN => false,
                    // A vocabulary with no `tokenizer.ggml.token_type` makes
                    // every token NORMAL upstream, so there is nothing to cut on.
                    _ => return None,
                }
            };
            Some(Special {
                text: text.clone(),
                id: i as i32,
                user_defined,
                lstrip: false,
                rstrip: false,
            })
        })
        .collect();
    specials.sort_by(|a, b| b.text.len().cmp(&a.text.len()).then(a.id.cmp(&b.id)));
    specials
}

/// llama.cpp's last block in `llama_vocab::impl::load()`: the only place
/// `LLAMA_TOKEN_ATTR_LSTRIP` / `LLAMA_TOKEN_ATTR_RSTRIP` are ever set, and
/// keyed on metadata that has nothing to do with the vocabulary's contents.
///
/// Ordering is not a detail. Upstream runs this **after** `cache_special_tokens`
/// is built and sorted (llama-vocab.cpp:2953 then 3016), which is why the phi-3
/// branch can iterate `cache_special_tokens` and why a token demoted to NORMAL
/// by `special_eog_ids` has already left the set. Applying it inside
/// [`build_specials`] would set flags on ids the partition never visits.
///
/// Three rules, and one of them cannot run:
///
/// 1. `tokenizer.ggml.pre` contains `jina-v2-de` / `jina-v2-es` /
///    `jina-v2-code`, or `general.architecture` contains `nomic-bert-moe` /
///    `jina-bert-v3`: sets `LSTRIP` on `<mask>`, and only if the vocabulary has
///    one — upstream warns and continues when it does not.
/// 2. `general.name` lowercased contains `modern-bert`: sets `LSTRIP` on
///    `[MASK]`, same presence check.
/// 3. `general.name` lowercased contains `phi-3` or `phi3`: sets `RSTRIP` on
///    every `cache_special_tokens` id, then on `</s>`, then *clears* `RSTRIP` on
///    `<unk>`, `<s>` and `<empty>`.
///
/// Rule 3 is upstream's only use of `<empty>`, and it is what
/// `token_to_id.at("<empty>")` looks up. A GGUF's empty token is rewritten to
/// `[EMPTY_<i>]` when the token list is read (llama-vocab.cpp:2438), so
/// `<empty>` is not a key in `token_to_id` for any vocabulary lacking those
/// eight literal characters — and no real tokenizer has such a token. The `.at`
/// throws `std::out_of_range`, `load_vocab` reports "error loading model
/// vocabulary: unordered_map::at", and the model does not load at all. Measured
/// on `models/synthetic-gpt2-strip.gguf`, which *does* carry `</s>`, `<unk>`
/// and `<s>` and still fails — see `scripts/gen_strip_refs.sh`.
///
/// So the flags rule 3 would set are unobservable in llama.cpp, and a port that
/// reproduces them reproduces code that has never run. This implementation does
/// not set them, and `phi3_rstrip_is_unreachable_upstream` in
/// `tests/strip_rules.rs` asserts the absence and names the reason. Writing the
/// loop out anyway would look like parity while being unverifiable, which is
/// worse than a documented omission.
///
/// Upstream's `.at` also throws for `</s>`, `<unk>` and `<s>` when absent, so on
/// a vocabulary missing those the failure is reached slightly earlier. The
/// `<empty>` case is the one that matters because it is unconditional.
fn derive_strips(md: &HashMap<String, gguf::Value>, specials: &mut [Special]) {
    let text_of = |key: &str| {
        md.get(key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase()
    };
    let contains_any = |hay: &str, needles: &[&str]| needles.iter().any(|n| hay.contains(n));

    let set_on = |specials: &mut [Special], id: i32, lstrip: bool, rstrip: bool| {
        if let Some(sp) = specials.iter_mut().find(|sp| sp.id == id) {
            sp.lstrip |= lstrip;
            sp.rstrip |= rstrip;
        }
    };
    // Upstream's `token_to_id.at(text)`, which throws rather than missing. Every
    // rule that reaches it makes the model unloadable (see above), so a missing
    // token is reported and skipped here instead of being propagated.
    let id_of = |text: &str| specials.iter().find(|sp| sp.text == text).map(|sp| sp.id);

    let pre = text_of("tokenizer.ggml.pre");
    let arch = text_of("general.architecture");
    let name = text_of("general.name");

    if contains_any(&pre, &["jina-v2-de", "jina-v2-es", "jina-v2-code"])
        || contains_any(&arch, &["nomic-bert-moe", "jina-bert-v3"])
    {
        match id_of("<mask>") {
            Some(id) => set_on(specials, id, true, false),
            None => eprintln!("strip rule: no <mask> in the vocabulary; skipping LSTRIP"),
        }
    } else if contains_any(&name, &["modern-bert"]) {
        match id_of("[MASK]") {
            Some(id) => set_on(specials, id, true, false),
            None => eprintln!("strip rule: no [MASK] in the vocabulary; skipping LSTRIP"),
        }
    }
}

#[derive(PartialEq, Eq)]
struct Bigram {
    rank: u32,
    left: i32,
    right: i32,
    /// Byte lengths of the two symbols when this pair was queued. A merge that
    /// happened since then makes the entry stale.
    left_len: usize,
    right_len: usize,
}

/// Queue the (left, right) pair if the merge table knows it.
fn push_bigram(
    tok: &Bpe,
    queue: &mut BinaryHeap<Reverse<Bigram>>,
    word: &str,
    starts: &[usize],
    len: &[usize],
    left: usize,
    right: usize,
) {
    let text = |i: usize| &word[starts[i]..starts[i] + len[i]];
    let key = (text(left).to_string(), text(right).to_string());
    if let Some(&rank) = tok.ranks.get(&key) {
        queue.push(Reverse(Bigram {
            rank,
            left: left as i32,
            right: right as i32,
            left_len: len[left],
            right_len: len[right],
        }));
    }
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Natural order on (rank, left): llama.cpp's `llm_bigram_bpe::comparator`
        // is a min-heap, and the queue wraps entries in `Reverse` so that
        // `BinaryHeap`'s max-heap pops the lowest (rank, left) first.
        self.rank.cmp(&other.rank).then(self.left.cmp(&other.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Llama-3 pre-tokenizer
// ---------------------------------------------------------------------------

/// `unicode_regex_split_custom_llama3`. Identical to Qwen2 except that digit
/// runs are grouped as `\p{N}{1,3}` and the trailing-newline group is
/// ` ?[^\s\p{L}\p{N}]+[\r\n/]*` (note the `/`).
fn split_custom_llama3(cpts: &[u32]) -> Vec<(usize, usize)> {
    let n = cpts.len();
    let flags_at = |i: usize| -> u16 { if i < n { flags_from_cpt(cpts[i]) } else { 0 } };
    let cpt_at = |i: usize| -> u32 { if i < n { cpts[i] } else { OUT_OF_RANGE } };
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut prev_end = 0usize;
    let mut pos = 0usize;

    while pos < n {
        let cpt = cpt_at(pos);
        let flags = flags_at(pos);

        // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == b'\'' as u32 && pos + 1 < n {
            let next = cpt_at(pos + 1);
            if next == b's' as u32
                || next == b't' as u32
                || next == b'm' as u32
                || next == b'd' as u32
            {
                pos += emit(&mut out, &mut prev_end, pos + 2);
                continue;
            }
            if pos + 2 < n {
                let next2 = cpt_at(pos + 2);
                if (next == b'r' as u32 && next2 == b'e' as u32)
                    || (next == b'v' as u32 && next2 == b'e' as u32)
                    || (next == b'l' as u32 && next2 == b'l' as u32)
                {
                    pos += emit(&mut out, &mut prev_end, pos + 3);
                    continue;
                }
            }
        }

        // regex: [^\r\n\p{L}\p{N}]?\p{L}+
        if !(cpt == b'\r' as u32 || cpt == b'\n' as u32 || flags & FLAG_NUMBER != 0)
            && (flags & FLAG_LETTER != 0 || flags_at(pos + 1) & FLAG_LETTER != 0)
        {
            pos += 1;
            while flags_at(pos) & FLAG_LETTER != 0 {
                pos += 1;
            }
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: \p{N}{1,3}
        if flags & FLAG_NUMBER != 0 {
            pos += 1;
            while pos - prev_end < 3 && flags_at(pos) & FLAG_NUMBER != 0 {
                pos += 1;
            }
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: <space>?[^\s\p{L}\p{N}]+[\r\n/]*
        let mut flags2 = if cpt == b' ' as u32 {
            flags_at(pos + 1)
        } else {
            flags
        };
        if flags2 & (FLAG_WHITESPACE | FLAG_LETTER | FLAG_NUMBER) == 0 && flags2 != 0 {
            pos += usize::from(cpt == b' ' as u32);
            while flags2 & (FLAG_WHITESPACE | FLAG_LETTER | FLAG_NUMBER) == 0 && flags2 != 0 {
                pos += 1;
                flags2 = flags_at(pos);
            }
            let mut cpt2 = cpt_at(pos);
            while cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 || cpt2 == b'/' as u32 {
                pos += 1;
                cpt2 = cpt_at(pos);
            }
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        let mut num_ws = 0usize;
        let mut last_end_r_or_n = 0usize;
        while flags_at(pos + num_ws) & FLAG_WHITESPACE != 0 {
            let cpt2 = cpt_at(pos + num_ws);
            if cpt2 == b'\r' as u32 || cpt2 == b'\n' as u32 {
                last_end_r_or_n = pos + num_ws + 1;
            }
            num_ws += 1;
        }

        if last_end_r_or_n > 0 {
            pos = last_end_r_or_n;
            emit(&mut out, &mut prev_end, pos);
            continue;
        }
        if num_ws > 1 && pos + num_ws < n {
            pos += num_ws - 1;
            emit(&mut out, &mut prev_end, pos);
            continue;
        }
        if num_ws > 0 {
            pos += num_ws;
            emit(&mut out, &mut prev_end, pos);
            continue;
        }

        pos += 1;
        emit(&mut out, &mut prev_end, pos);
    }

    out
}
