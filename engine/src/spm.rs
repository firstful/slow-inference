//! GGUF SentencePiece tokenizer (`tokenizer.ggml.model = "llama"`), ported from
//! llama.cpp's `llm_tokenizer_spm_session`.
//!
//! What this replaces: a greedy longest-match loop that lived in `main.rs`. It
//! was not a paraphrase of the real algorithm, it was a different algorithm,
//! and on a real vocabulary it produces different ids. On
//! `models/Tiny-Moe.Q4_K_M.gguf`:
//!
//! ```text
//! "yesterday"   llama.cpp:  ▁ | yes | ter | day
//!               greedy:     ▁ye | ster | day
//! ```
//!
//! llama.cpp does not segment first and match second. It starts from single
//! characters, proposes every adjacent pair that happens to be in the
//! vocabulary, and merges the highest-*scoring* pair first, repeatedly. Which
//! pairs are still eligible depends on what has already merged, so the result is
//! decided by the scores rather than by position: in "yesterday" the pair
//! `(s,t)` outscores `(y,e)`, so "yes" forms and the leading `▁` is stranded. A
//! longest-match scan can never strand anything — it walks left to right and
//! takes the longest thing it sees — and here it also loses a token, because
//! `▁ye` is a vocabulary entry that ends one character short of a real one.
//!
//! ## Three things that are easy to get wrong
//!
//! 1. **Scores are neither a length preference nor cumulative.** Every
//!    candidate pair carries the score of the *whole* merged token, so
//!    [`Spm::try_add_bigram`] re-reads it from the vocabulary each time it
//!    considers a pair. Summing sub-scores, or using `-log p`, gives a
//!    different answer.
//! 2. **`rev_merge` is keyed by text alone, not by position.** When two
//!    different positions spell the same pair, the last one queued wins and
//!    `resegment` recurses into *those* symbol indices. The recursion therefore
//!    reads symbols that may already have been merged away, in which case their
//!    length is 0, the empty string is not in the vocabulary, and the segment
//!    expands to nothing. That is what upstream does and it is load-bearing for
//!    matching it, so [`Spm::tokenize`] reproduces it rather than fixing it.
//! 3. **Whitespace is escaped before segmentation, not after.**
//!    `llama_escape_whitespace` rewrites every `' '` to U+2581 `▁`, which is an
//!    ordinary *normal* token in the vocabulary. So a literal `▁` in the input
//!    is indistinguishable from a space: `"a▁b"` encodes to the same ids as
//!    `"a b"`. Nothing normalizes it away, and there is a corpus case for it.
//!
//! There is no pre-tokenizer here at all: SPM vocabularies declare
//! `tokenizer.ggml.pre = "default"` and llama.cpp never consults it on this
//! path. The whole string is one segmentation problem.
//!
//! Parity: `engine/tests/tokenizer_parity.rs` diffs this against
//! `tests/data/ref_tokens/Tiny-Moe.Q4_K_M.ref`, which is llama.cpp's own output
//! for the shared corpus.

use crate::gguf;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's space marker.
pub const SPACE_MARKER: char = '\u{2581}';
const SPACE_MARKER_STR: &str = "\u{2581}";

// `llama_token_type` (include/llama.h).
const TOKEN_TYPE_UNKNOWN: i64 = 2;
const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;
const TOKEN_TYPE_BYTE: i64 = 6;

/// `llama_token_attr` bits (include/llama.h:101). The literal values matter: they
/// are what `token_to_piece` dispatches on, and NORMAL versus BYTE is a
/// behavioural difference — a NORMAL token is unescaped and lstripped, a BYTE
/// token is emitted raw.
const ATTR_UNKNOWN: u16 = 1 << 0;
const ATTR_NORMAL: u16 = 1 << 2;
const ATTR_CONTROL: u16 = 1 << 3;
const ATTR_USER_DEFINED: u16 = 1 << 4;
const ATTR_BYTE: u16 = 1 << 5;
const ATTR_LSTRIP: u16 = 1 << 7;
const ATTR_RSTRIP: u16 = 1 << 8;

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// `tokenizer.ggml.model` is not `llama`. Carries the offending value.
    NotSpm(String),
    /// The metadata is not shaped like an SPM vocabulary.
    Malformed(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NotSpm(m) => write!(
                f,
                "tokenizer model '{m}' is not an SPM vocabulary (want \"llama\")"
            ),
            Error::Malformed(what) => write!(f, "malformed SPM vocabulary: {what}"),
        }
    }
}

impl std::error::Error for Error {}

/// `_contains_any` on a lower-cased GGUF string.
fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

// ---------------------------------------------------------------------------
// the tokenizer
// ---------------------------------------------------------------------------

/// A SentencePiece (Unigram) vocabulary loaded from GGUF metadata.
pub struct Spm {
    /// `tokenizer.ggml.tokens`, in id order.
    tokens: Vec<String>,
    /// `tokenizer.ggml.scores`, parallel to `tokens`. An absent array means
    /// every score is 0.0, which is llama.cpp's `scores ? scores[i] : 0.0f`.
    scores: Vec<f32>,
    /// token text -> id. Upstream asserts this and `tokens` have equal length,
    /// so a duplicate token text means a corrupt file rather than a shadowed id.
    id_of: HashMap<String, i32>,
    /// per-token attribute bits, after the model-name heuristics
    attr: Vec<u16>,
    /// `cache_special_tokens`: CONTROL | USER_DEFINED | UNKNOWN, longest text
    /// first.
    special: Vec<usize>,
    bos: Option<i32>,
    eos: Option<i32>,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
    /// `special_eog_ids`: the tokens that end a generation, derived from token
    /// text rather than from any key. See [`crate::eog`].
    eog: crate::eog::Eog,
}

impl Spm {
    /// Load an SPM tokenizer from GGUF metadata, or [`Error::NotSpm`].
    pub fn from_metadata(md: &HashMap<String, gguf::Value>) -> Result<Spm, Error> {
        let model = md
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if model != "llama" {
            return Err(Error::NotSpm(model.to_string()));
        }

        let raw: Vec<&str> = md
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_vec_str())
            .ok_or(Error::Malformed("no tokenizer.ggml.tokens"))?;
        // llama.cpp replaces an empty token with a placeholder and warns. An
        // empty token would otherwise be reachable through `text_to_token` and
        // expand to nothing at resegment time.
        let tokens: Vec<String> = raw
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

        let scores: Vec<f32> = match md.get("tokenizer.ggml.scores") {
            Some(v) => v
                .as_vec_f32()
                .ok_or(Error::Malformed("tokenizer.ggml.scores is not an array"))?,
            None => vec![0.0; tokens.len()],
        };
        if scores.len() != tokens.len() {
            return Err(Error::Malformed("scores and tokens differ in length"));
        }

        let toktypes: Vec<i64> = md
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_vec_u64())
            .map(|v| v.iter().map(|&x| x as i64).collect())
            .unwrap_or_default();
        let mut attr: Vec<u16> = (0..tokens.len())
            .map(|i| match toktypes.get(i) {
                // UNDEFINED, and an absent `token_type` array, both land on
                // NORMAL: upstream initialises `attr` to NORMAL and only the
                // listed cases overwrite it.
                None | Some(1) => ATTR_NORMAL,
                Some(&TOKEN_TYPE_UNKNOWN) => ATTR_UNKNOWN,
                Some(&TOKEN_TYPE_CONTROL) => ATTR_CONTROL,
                Some(&TOKEN_TYPE_USER_DEFINED) => ATTR_USER_DEFINED,
                Some(&TOKEN_TYPE_BYTE) => ATTR_BYTE,
                _ => 0,
            })
            .collect();

        // llama.cpp does not stop at `tokenizer.ggml.token_type`. It then walks
        // `token_to_id` and promotes any token *spelled* like an end-of-text,
        // end-of-message, fill-in-the-middle or end-of-generation marker to
        // CONTROL, and forces four gpt-oss tokens to USER_DEFINED. Qwen2.5-3B
        // ships `</s>` with `token_type = 1` and gets exactly that promotion —
        // llama.cpp logs `control-looking token: 128247 '</s>' was not
        // control-type` while loading it. Tiny-MoE already declares `<s>`/`</s>`
        // CONTROL, so nothing on disk exercises this from the SPM side; it is
        // here because `attr` feeds `partition`, `is_special` and `piece`, and
        // all three are wrong for a promoted token that was left NORMAL.
        let eog = {
            let promo = crate::tokenizer::special_text_promotions(&tokens, md);
            for &id in &promo.control {
                attr[id] |= ATTR_CONTROL;
            }
            // The gpt-oss block assigns rather than ORs, so it clears whatever
            // the block above (or the file) put there.
            for &id in &promo.user_defined {
                attr[id] = ATTR_USER_DEFINED;
            }

            // And then the two workarounds at the end of llama.cpp's EOG block,
            // which assign as well and therefore win over the promotion above.
            // They belong here rather than in `eog` because they change how the
            // text is cut, not just when generation stops: `</s>` forced NORMAL
            // leaves `special` entirely, so `parse_special = true` stops cutting
            // on it. Upstream builds `cache_special_tokens` after both blocks.
            let eog = crate::eog::derive(&tokens, md, &promo);
            for (id, forced) in &eog.forced {
                let Ok(i) = usize::try_from(*id) else {
                    continue;
                };
                if i >= attr.len() {
                    continue;
                }
                attr[i] = match forced {
                    crate::eog::Forced::UserDefined => ATTR_USER_DEFINED,
                    crate::eog::Forced::Normal => ATTR_NORMAL,
                };
            }
            eog
        };

        let mut id_of: HashMap<String, i32> = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            if id_of.insert(t.clone(), i as i32).is_some() {
                return Err(Error::Malformed("duplicate token text"));
            }
        }

        // The `general.name` / `tokenizer.ggml.pre` / `general.architecture`
        // heuristics (llama-vocab.cpp:3010). They are the only source of
        // LSTRIP/RSTRIP, and both flags change how the input is cut around a
        // special token, so a model in one of these families does not tokenize
        // like one outside them.
        let lower = |k: &str| {
            md.get(k)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_lowercase()
        };
        let (name, pre, arch) = (
            lower("general.name"),
            lower("tokenizer.ggml.pre"),
            lower("general.architecture"),
        );
        {
            let set_attr = |attr: &mut Vec<u16>, text: &str, bit: u16, value: bool| {
                if let Some(&id) = id_of.get(text) {
                    let id = id as usize;
                    if value {
                        attr[id] |= bit;
                    } else {
                        attr[id] &= !bit;
                    }
                }
            };
            if contains_any(&pre, &["jina-v2-de", "jina-v2-es", "jina-v2-code"])
                || contains_any(&arch, &["nomic-bert-moe", "jina-bert-v3"])
            {
                set_attr(&mut attr, "<mask>", ATTR_LSTRIP, true);
            } else if contains_any(&name, &["phi-3", "phi3"]) {
                for id in special_ids(&attr) {
                    attr[id] |= ATTR_RSTRIP;
                }
                set_attr(&mut attr, "</s>", ATTR_RSTRIP, true);
                for t in ["<unk>", "<s>", "<|endoftext|>"] {
                    set_attr(&mut attr, t, ATTR_RSTRIP, false);
                }
            } else if contains_any(&name, &["modern-bert"]) {
                set_attr(&mut attr, "[MASK]", ATTR_LSTRIP, true);
            }
        }

        let mut special = special_ids(&attr);
        // Longest text first, matching `cache_special_tokens`'s sort. Upstream
        // uses the unstable `std::sort`, so equal-length texts have no defined
        // order there; this stable sort keeps them in id order, which produces
        // the same fragments for any two tokens that cannot overlap.
        special.sort_by(|&a, &b| tokens[b].len().cmp(&tokens[a].len()));

        let int_key = |k: &str| md.get(k).and_then(|v| v.as_u64()).map(|v| v as i32);
        let bool_key = |k: &str| {
            md.get(k).and_then(|v| match v {
                gguf::Value::Bool(b) => Some(*b),
                _ => None,
            })
        };

        Ok(Spm {
            // llama.cpp's SPM defaults, each overridable by the GGUF.
            add_space_prefix: bool_key("tokenizer.ggml.add_space_prefix").unwrap_or(true),
            add_bos: bool_key("tokenizer.ggml.add_bos_token").unwrap_or(true),
            add_eos: bool_key("tokenizer.ggml.add_eos_token").unwrap_or(false),
            tokens,
            scores,
            id_of,
            attr,
            special,
            bos: int_key("tokenizer.ggml.bos_token_id"),
            eos: int_key("tokenizer.ggml.eos_token_id"),
            eog,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
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

    /// The declared space-marker prefix, for reporting and tests.
    pub fn add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }

    /// How many tokens `tokenizer_st_partition` may cut the input around.
    pub fn n_special(&self) -> usize {
        self.special.len()
    }

    /// Whether `id` carries a CONTROL or UNKNOWN attribute, i.e. whether
    /// llama.cpp would classify it as special in the `parse_special = false`
    /// sense.
    pub fn is_special(&self, id: i32) -> bool {
        self.attr_of(id) & (ATTR_CONTROL | ATTR_UNKNOWN) != 0
    }

    /// The token text for `id`, or `None` if out of range.
    pub fn piece(&self, id: i32) -> Option<&str> {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.tokens.get(i))
            .map(String::as_str)
    }

    /// `text_to_token`: the id of a token spelled exactly `text`.
    pub fn piece_id(&self, text: &str) -> Option<i32> {
        self.id_of.get(text).copied()
    }

    /// `llama_vocab::byte_to_token` for SPM: the `<0xNN>` token for a byte, or
    /// the bare byte as a one-character string when that spelling is absent.
    /// `None` if the vocabulary has neither, which upstream aborts on.
    pub fn byte_token(&self, byte: u8) -> Option<i32> {
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        let name = [
            b'<',
            b'0',
            b'x',
            HEX[(byte >> 4) as usize],
            HEX[(byte & 15) as usize],
            b'>',
        ];
        self.id_of
            .get(std::str::from_utf8(&name).expect("ascii hex name"))
            .or_else(|| {
                std::str::from_utf8(std::slice::from_ref(&byte))
                    .ok()
                    .and_then(|s| self.id_of.get(s))
            })
            .copied()
    }

    /// Encode `text`, matching
    /// `llama_tokenize(vocab, text, add_special, parse_special)`. `add_special`
    /// prepends BOS and appends EOS when the GGUF asks for them.
    ///
    /// `parse_special` reaches [`Spm::partition`]. Under it SPM does something
    /// BPE does not: `add_space_prefix` is applied to a raw fragment whenever
    /// the *preceding fragment* was a special token, not whenever the text
    /// happens to start with a space. So `"<s> hello"` under `parse_special`
    /// emits id 1 and then tokenizes `"▁ hello"` — the marker is added on top of
    /// a leading space that was already there, and the result is `▁`, `" he"`,
    /// `"llo"`.
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<i32> {
        let mut out: Vec<i32> = Vec::new();

        // `is_prev_special` starts true, so the first text fragment always gets
        // the space prefix; an inline special token resets it, so the text after
        // one gets a prefix too.
        let mut is_prev_special = true;
        if add_special && self.add_bos {
            // GGML_ASSERT(special_bos_id != LLAMA_TOKEN_NULL) upstream: an SPM
            // vocabulary with add_bos and no bos id is a broken file. Emitting
            // nothing rather than panicking keeps a bad GGUF from killing the
            // process, and the attribute is observable in `encode`'s output.
            let Some(bos) = self.bos else { return out };
            out.push(bos);
            is_prev_special = true;
        }

        for frag in self.partition(text, parse_special) {
            match frag {
                Fragment::Raw(s) => {
                    let mut s = escape_whitespace(s);
                    if self.add_space_prefix && is_prev_special {
                        s.insert(0, SPACE_MARKER);
                    }
                    self.tokenize(&s, &mut out);
                    is_prev_special = false;
                }
                Fragment::Token(id) => {
                    out.push(id as i32);
                    is_prev_special = true;
                }
            }
        }

        if add_special
            && self.add_eos
            && let Some(eos) = self.eos
        {
            out.push(eos);
        }
        out
    }

    /// `llm_tokenizer_spm_session::tokenize` — the score-ordered pairwise
    /// merge, followed by `resegment`.
    ///
    /// `text` arrives already whitespace-escaped. Upstream creates a fresh
    /// session per fragment, and this call is that lifetime.
    fn tokenize(&self, text: &str, output: &mut Vec<i32>) {
        let mut session = Session::new(text);
        session.run(self, output);
    }

    /// llama.cpp's byte fallback: one `<0xNN>` token per byte. `byte_to_token`
    /// throws upstream when a vocabulary has neither spelling; skipping is the
    /// closest non-aborting equivalent, and it keeps one missing entry from
    /// taking down the whole encode.
    fn emit_bytes(&self, raw: &[u8], output: &mut Vec<i32>) {
        for &b in raw {
            if let Some(id) = self.byte_token(b) {
                output.push(id);
            }
        }
    }

    /// `llama_vocab::impl::tokenizer_st_partition` with `parse_special` as
    /// given: cut the input around every special token that is not excluded.
    ///
    /// With `parse_special = false`, CONTROL and UNKNOWN tokens are skipped and
    /// only USER_DEFINED ones participate — which is why `"<s>hello</s>"` comes
    /// out spelled out character by character rather than as ids 1 and 2.
    fn partition<'a>(&self, text: &'a str, parse_special: bool) -> Vec<Fragment<'a>> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut buffer = vec![Fragment::Raw(text)];

        for &sid in &self.special {
            let a = self.attr[sid];
            if !parse_special && a & (ATTR_CONTROL | ATTR_UNKNOWN) != 0 {
                continue;
            }
            let needle = self.tokens[sid].as_str();
            let (lstrip, rstrip) = (a & ATTR_LSTRIP != 0, a & ATTR_RSTRIP != 0);

            let mut out: Vec<Fragment<'_>> = Vec::new();
            for frag in buffer {
                let s = match frag {
                    Fragment::Token(id) => {
                        out.push(Fragment::Token(id));
                        continue;
                    }
                    Fragment::Raw(s) => s,
                };
                // The unsearched remainder of `s`, as a half-open byte range.
                let mut start = 0usize;
                while let Some(at) = s[start..].find(needle) {
                    let at = start + at;
                    let mut left_end = at;
                    if lstrip {
                        while left_end > start && s.as_bytes()[left_end - 1].is_ascii_whitespace() {
                            left_end -= 1;
                        }
                    }
                    if left_end > start {
                        out.push(Fragment::Raw(&s[start..left_end]));
                    }
                    out.push(Fragment::Token(sid));
                    start = at + needle.len();
                    if rstrip {
                        while start < s.len() && s.as_bytes()[start].is_ascii_whitespace() {
                            start += 1;
                        }
                    }
                    if start >= s.len() {
                        break;
                    }
                }
                if start < s.len() {
                    out.push(Fragment::Raw(&s[start..]));
                }
            }
            buffer = out;
        }
        buffer
    }

    /// `llama_vocab::impl::token_to_piece` for SPM: the text one token
    /// contributes to a detokenized string, with up to `lstrip` leading spaces
    /// removed.
    ///
    /// The dispatch is on the token *attribute*, not on the token's spelling,
    /// which is why `attr` keeps NORMAL and BYTE apart:
    ///
    /// * UNKNOWN / CONTROL → nothing. `special` is false on every path this
    ///   engine uses, so these never render.
    /// * USER_DEFINED → the raw token text, marker and angle brackets intact.
    /// * NORMAL → the token text with `▁` un-escaped to a space, then lstripped.
    /// * BYTE → the single raw byte, which is `<0xNN>`'s payload.
    /// * anything else → nothing.
    pub fn piece_bytes(&self, id: i32, lstrip: usize) -> Vec<u8> {
        let Some(text) = self.piece(id) else {
            return Vec::new();
        };
        let attr = self.attr_of(id);
        if attr & (ATTR_UNKNOWN | ATTR_CONTROL) != 0 {
            return Vec::new();
        }
        if attr & ATTR_USER_DEFINED != 0 {
            return text.as_bytes().to_vec();
        }
        if attr & ATTR_NORMAL != 0 {
            // `_try_copy` strips up to `lstrip` leading spaces *from the
            // unescaped text*, so a token beginning with two markers loses
            // exactly one and keeps the other.
            let unescaped = unescape_whitespace(text);
            let bytes = unescaped.as_bytes();
            let mut start = 0;
            while start < lstrip && start < bytes.len() && bytes[start] == b' ' {
                start += 1;
            }
            return bytes[start..].to_vec();
        }
        if attr & ATTR_BYTE != 0 {
            // `<0xNN>` -> N. Upstream reads the byte out of the token text; if the
            // spelling is not that shape it emits nothing.
            return byte_token_value(text).map(|b| vec![b]).unwrap_or_default();
        }
        Vec::new()
    }

    /// `llama_vocab::impl::detokenize`: ids to text.
    ///
    /// `remove_special` skips a leading BOS and a trailing EOS, which is also
    /// what disables the leading-space strip: llama.cpp sets `remove_space =
    /// add_space_prefix` and then clears it when it drops the BOS, because a
    /// prompt that already begins with `<s>` has no marker to strip.
    pub fn detokenize(&self, ids: &[i32], remove_special: bool) -> String {
        let mut ids = ids;
        // A BOOLEAN, not "the first token": only a BOS disables the strip.
        let mut lstrip = usize::from(self.add_space_prefix);
        if remove_special && self.add_bos && ids.first().copied() == self.bos {
            lstrip = 0;
            ids = &ids[1..];
        }
        if remove_special && self.add_eos && ids.last().copied() == self.eos && !ids.is_empty() {
            ids = &ids[..ids.len() - 1];
        }
        let mut out: Vec<u8> = Vec::new();
        for &id in ids {
            out.extend(self.piece_bytes(id, lstrip));
            lstrip = 0;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn attr_of(&self, id: i32) -> u16 {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.attr.get(i))
            .copied()
            .unwrap_or(0)
    }
}

/// `llama_escape_whitespace`: `replace_all(text, " ", "▁")`.
///
/// Applied to a whole fragment *before* segmentation, not after. U+2581 is an
/// ordinary normal token in an SPM vocabulary, so this is the only thing
/// distinguishing a space from the 260 other characters, and a literal `▁` in the
/// input therefore encodes identically to a space.
fn escape_whitespace(text: &str) -> String {
    if text.contains(' ') {
        text.replace(' ', SPACE_MARKER_STR)
    } else {
        text.to_string()
    }
}

/// `llama_unescape_whitespace`: `replace_all(word, "▁", " ")`.
fn unescape_whitespace(text: &str) -> String {
    if text.contains(SPACE_MARKER) {
        text.replace(SPACE_MARKER, " ")
    } else {
        text.to_string()
    }
}

/// `<0xNN>` -> the byte, if that is what this token is.
fn byte_token_value(t: &str) -> Option<u8> {
    let b = t.as_bytes();
    if b.len() != 6 || &b[..3] != b"<0x" || b[5] != b'>' {
        return None;
    }
    u8::from_str_radix(std::str::from_utf8(&b[3..5]).ok()?, 16).ok()
}

fn special_ids(attr: &[u16]) -> Vec<usize> {
    attr.iter()
        .enumerate()
        .filter(|e| e.1 & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN) != 0)
        .map(|(i, _)| i)
        .collect()
}

// ---------------------------------------------------------------------------
// work queue
// ---------------------------------------------------------------------------

/// `llm_tokenizer_spm_session`: the symbols, the linked list they hang off, the
/// work queue, and the reverse merge table. One per fragment, matching upstream's
/// lifetime.
struct Session<'a> {
    /// The escaped fragment, as bytes. Merged spans are byte ranges into this.
    bytes: &'a [u8],
    /// Byte offset of each symbol's first byte, plus a trailing end offset.
    offs: Vec<usize>,
    /// Current byte length of each symbol; 0 once absorbed into its left neighbour.
    size: Vec<usize>,
    prev: Vec<i32>,
    next: Vec<i32>,
    queue: BinaryHeap<Bigram>,
    /// merged text -> the (left, right) symbol pair that produced it.
    rev_merge: HashMap<String, (usize, usize)>,
}

impl<'a> Session<'a> {
    fn new(text: &'a str) -> Session<'a> {
        let mut offs: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        offs.push(text.len());
        let n = offs.len() - 1;
        Session {
            bytes: text.as_bytes(),
            size: (0..n).map(|k| offs[k + 1] - offs[k]).collect(),
            prev: (0..n).map(|k| k as i32 - 1).collect(),
            next: (0..n)
                .map(|k| if k + 1 < n { k as i32 + 1 } else { -1 })
                .collect(),
            offs,
            queue: BinaryHeap::new(),
            rev_merge: HashMap::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.size.is_empty()
    }

    /// Byte span of symbol `k`.
    fn span(&self, k: usize) -> &'a [u8] {
        &self.bytes[self.offs[k]..self.offs[k] + self.size[k]]
    }

    /// Merge every pair that can be merged, highest score first.
    fn run(&mut self, vocab: &Spm, output: &mut Vec<i32>) {
        if self.is_empty() {
            return;
        }
        // Seed the queue with every pair that is already a token.
        for k in 1..self.size.len() {
            self.try_add_bigram(vocab, k as i32 - 1, k as i32);
        }

        while let Some(bg) = self.queue.pop() {
            let (l, r) = (bg.left as usize, bg.right as usize);
            // Stale entry: either symbol was merged away, or its extent changed
            // since this pair was queued. Comparing summed lengths is llama.cpp's
            // own test, expressed without the captured text.
            if self.size[l] == 0 || self.size[r] == 0 || self.size[l] + self.size[r] != bg.size {
                continue;
            }
            if std::env::var_os("SLOW_TOKENIZER_TRACE").is_some() {
                eprintln!(
                    "    spm merge {:?} + {:?} -> {:?} (score {})",
                    String::from_utf8_lossy(self.span(l)),
                    String::from_utf8_lossy(self.span(r)),
                    String::from_utf8_lossy(&[self.span(l), self.span(r)].concat()),
                    bg.score
                );
            }
            self.size[l] += self.size[r];
            self.size[r] = 0;
            self.next[l] = self.next[r];
            if self.next[r] >= 0 {
                self.prev[self.next[r] as usize] = l as i32;
            }
            self.try_add_bigram(vocab, self.prev[l], l as i32);
            self.try_add_bigram(vocab, l as i32, self.next[l]);
        }

        let mut i: usize = 0;
        while i != usize::MAX {
            self.resegment(vocab, i, output, 0);
            i = if self.next[i] < 0 {
                usize::MAX
            } else {
                self.next[i] as usize
            };
        }
    }

    /// `try_add_bigram`: queue `(left, right)` if their concatenation is a
    /// token, recording the pair in `rev_merge` for `resegment`.
    fn try_add_bigram(&mut self, vocab: &Spm, left: i32, right: i32) {
        if left < 0 || right < 0 {
            return;
        }
        let (l, r) = (left as usize, right as usize);
        let joined = [self.span(l), self.span(r)].concat();
        let Ok(key) = std::str::from_utf8(&joined) else {
            return;
        };
        let Some(&token) = vocab.id_of.get(key) else {
            return;
        };
        // Dead upstream (`n_tokens()` is the vocabulary size and every id comes
        // from it) but cheap, and it keeps a truncated id from indexing `scores`.
        if token as u32 >= vocab.tokens.len() as u32 {
            return;
        }
        self.queue.push(Bigram {
            left,
            right,
            score: vocab.scores[token as usize],
            size: joined.len(),
        });
        // Keyed by text, so a later position spelling the same pair overwrites an
        // earlier one. See the module docs.
        self.rev_merge.insert(key.to_string(), (l, r));
    }

    /// `resegment`: turn a merged symbol into ids, recursing through
    /// `rev_merge` when the whole span is not a token, and falling back to bytes
    /// when it is not a pair either.
    ///
    /// `depth` bounds the recursion. Upstream has no bound, but the depth here
    /// is the number of merges that produced the span, which is at most the
    /// number of characters, so the cap is unreachable for any input llama.cpp
    /// accepts. It exists so a cycle in `rev_merge` — reachable in principle,
    /// since the map is keyed by text — surfaces as a test failure rather than a
    /// stack overflow.
    fn resegment(&self, vocab: &Spm, sym: usize, output: &mut Vec<i32>, depth: usize) {
        const MAX_DEPTH: usize = 4096;
        if depth >= MAX_DEPTH {
            debug_assert!(false, "spm resegment recursion exceeded {MAX_DEPTH}");
            return;
        }
        let raw = self.span(sym);
        let Ok(text) = std::str::from_utf8(raw) else {
            // Not valid UTF-8 on its own, which for an SPM vocabulary means a
            // multi-byte codepoint that no token spans. Fall through to bytes.
            vocab.emit_bytes(raw, output);
            return;
        };
        if let Some(&id) = vocab.id_of.get(text) {
            output.push(id);
            return;
        }
        if let Some(&(l, r)) = self.rev_merge.get(text) {
            self.resegment(vocab, l, output, depth + 1);
            self.resegment(vocab, r, output, depth + 1);
            return;
        }
        vocab.emit_bytes(raw, output);
    }
}

// ---------------------------------------------------------------------------
// work queue
// ---------------------------------------------------------------------------

#[derive(PartialEq)]
struct Bigram {
    score: f32,
    left: i32,
    right: i32,
    /// Combined byte length when queued; a merge since then makes it stale.
    size: usize,
}

// `Ord` on `score` routes NaN to `Equal`, which is not `Eq`-consistent for
// f32, so `Eq` is deliberately not implemented.
impl Eq for Bigram {}

impl Ord for Bigram {
    /// Ascending on score, so the max-heap pops the highest score first.
    ///
    /// `llm_bigram_spm::comparator` is handed to a `std::priority_queue`, which
    /// pops its maximum under that comparator — the best pair, meaning highest
    /// score and, on ties, lowest left index. That is *natural* ascending order
    /// on `(score, left)` under a plain max-heap, so no `Reverse` is involved.
    ///
    /// The tempting mistake is to copy the BPE queue in `tokenizer.rs`, which is
    /// ascending on merge *rank* and therefore needs `Reverse` to be a min-heap.
    /// SPM's ranking is already "lower is better" (scores are negative log
    /// counts) and it wants the maximum, so `Reverse` here inverts the merge
    /// order. Nothing crashes: the engine merges the rarest pairs first and emits
    /// more, shorter tokens, which is precisely the defect only a reference
    /// oracle catches. On `models/Tiny-Moe.Q4_K_M.gguf` it turned "hello, world!"
    /// into 8 tokens instead of 5.
    ///
    /// NaN scores cannot occur (they come from a GGUF array); treating one as
    /// Equal keeps `Ord` total. `right` is not a tie-break upstream, but two
    /// entries that agree on everything an outcome depends on are
    /// interchangeable, so ordering them cannot change the result.
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then(self.left.cmp(&other.left))
            .then(self.right.cmp(&other.right))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

enum Fragment<'a> {
    Raw(&'a str),
    Token(usize),
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------
//
// Synthetic vocabularies rather than the real 4096-token one. A real vocabulary
// exercises the algorithm but cannot *isolate* a decision: making two scores tie,
// making a longer token lose, or making a pair reachable from two positions all
// take a contrived vocabulary to arrange. Every expectation below is derived by
// hand from the merge order, not recorded from a run — a test that asserts what
// the code printed only pins the code.
//
// The shared base is what llama.cpp itself puts at the front of an SPM
// vocabulary: three CONTROL tokens, then the 256 `<0xNN>` byte tokens.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Value;

    const N_BYTE_TOKENS: usize = 256;
    /// Id of `<unk>`, which the base always puts first. It carries the UNKNOWN
    /// attribute, so `decode` drops it.
    const UNK: i32 = 0;

    #[derive(Clone)]
    struct Builder {
        toks: Vec<String>,
        scores: Vec<f32>,
        types: Vec<i64>,
        extra: Vec<(String, Value)>,
        /// Metadata keys deliberately left out, so "the GGUF does not say this"
        /// is expressible in a fixture.
        omit: Vec<String>,
    }

    impl Builder {
        fn new() -> Builder {
            let mut b = Builder {
                toks: Vec::new(),
                scores: Vec::new(),
                types: Vec::new(),
                extra: Vec::new(),
                omit: Vec::new(),
            };
            for t in ["<unk>", "<s>", "</s>"] {
                b = b.raw(t, -1000.0, TOKEN_TYPE_CONTROL);
            }
            for byte in 0..N_BYTE_TOKENS {
                b = b.raw(&format!("<0x{byte:02X}>"), 0.0, TOKEN_TYPE_BYTE);
            }
            b
        }

        fn raw(mut self, text: &str, score: f32, ty: i64) -> Builder {
            self.toks.push(text.to_string());
            self.scores.push(score);
            self.types.push(ty);
            self
        }

        /// A NORMAL token.
        fn tok(self, text: &str, score: f32) -> Builder {
            self.raw(text, score, 1)
        }

        fn key(mut self, k: &str, v: Value) -> Builder {
            self.extra.push((k.to_string(), v));
            self
        }

        fn omit(mut self, k: &str) -> Builder {
            self.omit.push(k.to_string());
            self
        }

        /// Drop the space-marker prefix, so a fixture reads as written.
        fn no_prefix(self) -> Builder {
            self.key("tokenizer.ggml.add_space_prefix", Value::Bool(false))
        }

        fn id_of(&self, text: &str) -> i32 {
            self.toks
                .iter()
                .position(|t| t == text)
                .unwrap_or_else(|| panic!("no token {text:?} in the fixture")) as i32
        }

        fn build(&self) -> Spm {
            let mut md = HashMap::new();
            md.insert("tokenizer.ggml.model".into(), Value::Str("llama".into()));
            md.insert("tokenizer.ggml.pre".into(), Value::Str("default".into()));
            md.insert(
                "tokenizer.ggml.tokens".into(),
                Value::Array(self.toks.iter().cloned().map(Value::Str).collect()),
            );
            md.insert(
                "tokenizer.ggml.scores".into(),
                Value::Array(self.scores.iter().copied().map(Value::F32).collect()),
            );
            md.insert(
                "tokenizer.ggml.token_type".into(),
                Value::Array(self.types.iter().copied().map(Value::I64).collect()),
            );
            if !self.omit.iter().any(|k| k == "tokenizer.ggml.bos_token_id") {
                md.insert(
                    "tokenizer.ggml.bos_token_id".into(),
                    Value::U64(self.id_of("<s>") as u64),
                );
            }
            for k in &self.omit {
                md.remove(k);
            }
            for (k, v) in &self.extra {
                md.insert(k.clone(), v.clone());
            }
            Spm::from_metadata(&md).expect("synthetic SPM metadata is valid")
        }
    }

    fn spm(b: &Builder) -> Spm {
        b.build()
    }

    fn ids(t: &Spm, text: &str) -> Vec<i32> {
        t.encode(text, false, false)
    }

    fn names<'a>(t: &'a Spm, ids: &[i32]) -> Vec<&'a str> {
        ids.iter().map(|&i| t.piece(i).unwrap_or("<oob>")).collect()
    }

    /// Scores beat length, not the other way round.
    ///
    /// `"abc"` scores -2 and `"▁ab"` scores -3, so the merge takes `"ab"` then
    /// `"abc"` and strands the leading marker — even though `"▁ab"` is a longer
    /// vocabulary entry and a longest-match scan would grab it. `▁ab` is queued
    /// at size 5 and by the time it is popped symbol 1 has grown to 3 bytes, so
    /// it is dropped as stale.
    #[test]
    fn a_lower_scoring_longer_token_loses() {
        let t = spm(&Builder::new()
            .tok("▁", -100.0)
            .tok("a", -900.0)
            .tok("b", -900.0)
            .tok("c", -900.0)
            .tok("ab", -1.0)
            .tok("abc", -2.0)
            .tok("▁ab", -3.0));
        let got = ids(&t, "abc");
        assert_eq!(names(&t, &got), vec!["▁", "abc"]);
        // And the greedy answer is different, which is the point: this is the
        // shape that separated the two algorithms on the real vocabulary.
        assert_ne!(got.first().copied(), t.piece_id("▁ab"));
    }

    /// Equal scores are broken by position, not by length.
    ///
    /// `"aaaa"` has three `"aa"` candidates at left indices 0, 1 and 2. Index 0
    /// merges; 1 is stale because its symbol was absorbed; 2 is still valid and
    /// merges next. `"aaa"` is only ever queued at size 3, and by the time it
    /// could pop the symbol is 2+2 bytes, so it never forms. Greedy would give
    /// `aaa | a`.
    #[test]
    fn ties_break_toward_the_lowest_left_index() {
        let t = spm(&Builder::new()
            .no_prefix()
            .tok("a", -900.0)
            .tok("aa", -1.0)
            .tok("aaa", -2.0));
        let got = ids(&t, "aaaa");
        assert_eq!(names(&t, &got), vec!["aa", "aa"]);
    }

    /// An absent `tokenizer.ggml.scores` means every pair ties, so the merge is
    /// decided entirely by position. `abc` forms only because `ab` merged first
    /// and left `c` adjacent to it; `bc` was queued before `abc` but its left
    /// symbol is gone by then.
    #[test]
    fn absent_scores_make_the_whole_merge_position_decided() {
        let b = Builder::new()
            .no_prefix()
            .tok("a", 0.0)
            .tok("b", 0.0)
            .tok("c", 0.0)
            .tok("ab", 0.0)
            .tok("bc", 0.0)
            .tok("abc", 0.0);
        // Drop the scores array entirely, which is what upstream defaults to 0.0.
        let mut no_scores = HashMap::new();
        no_scores.insert("tokenizer.ggml.model".into(), Value::Str("llama".into()));
        no_scores.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(b.toks.iter().cloned().map(Value::Str).collect()),
        );
        no_scores.insert(
            "tokenizer.ggml.token_type".into(),
            Value::Array(b.types.iter().copied().map(Value::I64).collect()),
        );
        no_scores.insert("tokenizer.ggml.add_space_prefix".into(), Value::Bool(false));
        let t = Spm::from_metadata(&no_scores).unwrap();
        // The same vocabulary with explicit all-zero scores must agree, since
        // upstream's `scores ? scores[i] : 0.0f` makes the two the same thing.
        let with_scores = spm(&b);
        // "ab" wins the tie at left index 0 and grows to "abc", absorbing the
        // trailing "c"; "bc" was queued earlier but its left symbol is gone.
        let got = ids(&t, "abc");
        assert_eq!(names(&t, &got), vec!["abc"]);
        assert_eq!(
            names(&with_scores, &ids(&with_scores, "abc")),
            names(&t, &got)
        );
    }

    /// Byte fallback is per *byte*, not per codepoint: a 3-byte CJK character no
    /// token spans becomes three `<0xNN>` tokens.
    #[test]
    fn byte_fallback_is_one_token_per_byte_not_per_codepoint() {
        let t = spm(&Builder::new().tok("▁", -100.0));
        let got = ids(&t, "\u{4f60}");
        assert_eq!(names(&t, &got), vec!["▁", "<0xE4>", "<0xBD>", "<0xA0>"]);
        // And they detokenize back to the text that was encoded: `▁` renders as
        // the space it stands for, and because this fixture has the space prefix
        // on, the leading-space strip eats exactly that one space.
        assert!(t.add_space_prefix());
        assert_eq!(t.detokenize(&got, false), "\u{4f60}");
        // A bare byte token renders as its single byte, not as `<0xNN>`.
        assert_eq!(t.detokenize(&[t.byte_token(b'\n').unwrap()], false), "\n");
        assert_eq!(
            t.piece_bytes(t.byte_token(b'\n').unwrap(), 0),
            b"\n".to_vec()
        );
        // A vocabulary with no token list at all is refused, not invented.
        let mut md = HashMap::new();
        md.insert("tokenizer.ggml.model".into(), Value::Str("llama".into()));
        assert_eq!(
            Spm::from_metadata(&md).err(),
            Some(Error::Malformed("no tokenizer.ggml.tokens"))
        );
    }

    #[test]
    fn the_space_marker_prefix_follows_the_gguf() {
        let t = spm(&Builder::new()
            .tok("▁", -100.0)
            .tok("h", -900.0)
            .tok("e", -900.0)
            .tok("he", -2.0)
            .tok("▁he", -1.0));
        // Default for SPM is on.
        assert!(t.add_space_prefix());
        assert_eq!(names(&t, &ids(&t, "he")), vec!["▁he"]);
        assert_eq!(names(&t, &ids(&t, " he")), vec!["▁", "▁he"]);

        let t = spm(&Builder::new()
            .no_prefix()
            .tok("▁", -100.0)
            .tok("h", -900.0)
            .tok("e", -900.0)
            .tok("he", -2.0)
            .tok("▁he", -1.0));
        assert!(!t.add_space_prefix());
        assert_eq!(names(&t, &ids(&t, "he")), vec!["he"]);
        // With the prefix off, a leading space is escaped instead, which is the
        // same marker the prefix would have added.
        assert_eq!(names(&t, &ids(&t, " he")), vec!["▁he"]);

        // "▁he" is only reachable because "he" exists: the merge only ever joins
        // *adjacent current* symbols, so a vocabulary entry that no chain of
        // merges can build is unreachable however well it scores.
        let t = spm(&Builder::new()
            .tok("h", -900.0)
            .tok("e", -900.0)
            .tok("▁he", -1.0));
        assert_eq!(
            names(&t, &ids(&t, "he")),
            vec!["<0xE2>", "<0x96>", "<0x81>", "h", "e"]
        );
    }

    /// The degenerate prompt. llama.cpp's own comment on the SPM path:
    /// `encode('', add_special_tokens=True)` returns `[BOS]` and
    /// `encode('', add_special_tokens=False)` returns `[]`. The engine used to
    /// emit one spurious space marker, because the old greedy code prefixed it
    /// unconditionally.
    #[test]
    fn the_empty_prompt_is_nothing_and_bos_is_one_token() {
        let t = spm(&Builder::new().tok("▁", -100.0));
        assert!(ids(&t, "").is_empty());
        assert_eq!(t.encode("", true, false), vec![1]);
        // add_eos is off by default for SPM, so nothing is appended.
        let t_eos = spm(&Builder::new()
            .tok("▁", -100.0)
            .key("tokenizer.ggml.add_eos_token", Value::Bool(true))
            .key("tokenizer.ggml.eos_token_id", Value::U64(2)));
        assert_eq!(t_eos.encode("", true, false), vec![1, 2]);
    }

    /// The gemma4 / paddleocr fix, on the SPM side: a vocabulary carrying the
    /// tool-response marker assigns `</s>` NORMAL, so it leaves both the EOG set
    /// and `cache_special_tokens` -- and therefore stops being cut out of the
    /// prompt even with `parse_special = true`.
    ///
    /// The mutation this guards against is silent in both directions: without the
    /// assignment `</s>` still stops generation and still cuts, so a test that
    /// only looked at the EOG set would pass an engine that tokenizes wrongly.
    #[test]
    fn the_tool_response_marker_takes_s_out_of_the_cut_set() {
        let marker = crate::special_aliases::EOG_ALIASES[19];
        let base = Builder::new()
            .tok("a", -1.0)
            .tok("b", -2.0)
            .tok("\u{2581}a", -0.5);

        let plain = spm(&base);
        assert!(plain.is_eog(2), "</s> is id 2 and an alias match");
        assert_eq!(plain.encode(plain.piece(2).unwrap(), false, true), vec![2]);

        let fixed = spm(&base.clone().raw(marker, -0.1, TOKEN_TYPE_CONTROL));
        assert!(!fixed.is_eog(2));
        assert!(
            !fixed.is_special(2),
            "NORMAL is neither CONTROL nor UNKNOWN"
        );
        assert_ne!(
            fixed.encode(fixed.piece(2).unwrap(), false, true),
            vec![2],
            "a NORMAL token is spelled out, not lifted out of the text"
        );
    }

    /// U+2581 is a NORMAL token, so a literal one in the input is the same string
    /// as an escaped space and encodes identically. Nothing normalizes it.
    #[test]
    fn a_literal_space_marker_is_indistinguishable_from_a_space() {
        let t = spm(&Builder::new()
            .tok("a", -1.0)
            .tok("b", -2.0)
            .tok("▁a", -0.5)
            .tok("▁b", -0.4));
        assert_eq!(ids(&t, "a\u{2581}b"), ids(&t, "a b"));
        assert_eq!(names(&t, &ids(&t, "a\u{2581}b")), vec!["▁a", "▁b"]);
    }

    /// CONTROL and UNKNOWN tokens are *not* split out when `parse_special` is
    /// false, which is the default for a bare prompt. So `"<s>hello</s>"` is
    /// spelled out, not turned into ids.
    #[test]
    fn control_tokens_stay_text_when_specials_are_not_parsed() {
        let b =
            Builder::new()
                .tok("a", -900.0)
                .tok("b", -900.0)
                .raw("<ctl>", -0.5, TOKEN_TYPE_CONTROL);
        let t = spm(&b);
        let got = t.encode("<ctl>", false, false);
        assert_ne!(
            got,
            vec![b.id_of("<ctl>")],
            "CONTROL must not be emitted as an id"
        );
        assert!(got.len() > 1);
        assert!(t.is_special(b.id_of("<ctl>")));
        // `<s>` from the base vocabulary is CONTROL too and is treated the same.
        assert!(t.is_special(b.id_of("<s>")));

        // The same text with the flag on is exactly the one id, and `encode`
        // threads the parameter through to the partition rather than dropping it.
        assert_eq!(t.encode("<ctl>", false, true), vec![b.id_of("<ctl>")]);

        // The partition function splits on it when asked, which is what makes the
        // skip above a decision rather than an accident.
        let show = |t: &Spm, text: &str, parse: bool| -> Vec<String> {
            t.partition(text, parse)
                .iter()
                .map(|f| match f {
                    Fragment::Raw(s) => format!("raw({s})"),
                    Fragment::Token(id) => format!("tok({})", t.piece(*id as i32).unwrap_or("?")),
                })
                .collect()
        };
        assert_eq!(show(&t, "<ctl>", true), vec!["tok(<ctl>)"]);
        assert_eq!(show(&t, "<ctl>", false), vec!["raw(<ctl>)"]);
        assert_eq!(
            show(&t, "a<ctl>b", true),
            vec!["raw(a)", "tok(<ctl>)", "raw(b)"]
        );
        // The three from the base vocabulary plus "<ctl>".
        assert_eq!(t.n_special(), 4);
    }

    /// USER_DEFINED tokens are split out even without `parse_special` — that is
    /// why llama.cpp's skip is narrowed to CONTROL and UNKNOWN. The split resets
    /// `is_prev_special`, so the text after the marker gets its own space prefix.
    #[test]
    fn user_defined_tokens_are_split_out_even_without_parsing_specials() {
        let b = Builder::new()
            .raw("<u>", -0.5, TOKEN_TYPE_USER_DEFINED)
            .tok("hi", -1.0)
            .tok("▁hi", -0.9);
        let t = spm(&b);
        assert!(!t.is_special(b.id_of("<u>")));
        assert_eq!(ids(&t, "<u>hi"), vec![b.id_of("<u>"), b.id_of("▁hi")]);
        // Text before the marker keeps the prefix it would otherwise have had.
        assert_eq!(names(&t, &ids(&t, "hi")), vec!["▁hi"]);
    }

    /// `rev_merge` is keyed by text alone, so when two positions spell the same
    /// pair the *last* one queued wins and `resegment` recurses into those stored
    /// indices — which may already have been merged away, in which case the
    /// symbol's length is 0 and the segment expands to nothing.
    ///
    /// Reproduced here because it is upstream behaviour the port deliberately
    /// keeps. "Fixing" it — recursing into the symbol's own neighbours instead —
    /// would stop matching llama.cpp, so the shape is pinned rather than left
    /// implicit.
    #[test]
    fn rev_merge_is_keyed_by_text_not_position() {
        // `"ab"` is queued from position (1,2) and `"ab"` is also spelled by the
        // merged span, so `rev_merge["ab"]` ends up pointing at (1,2) even when the
        // segment being resegmented is at position 0. Symbol 1 is dead by then.
        let t = spm(&Builder::new()
            .no_prefix()
            .tok("a", -900.0)
            .tok("b", -900.0)
            .tok("ab", -1.0));
        // No "aa", so (0,1) cannot form: the surviving segment is exactly the
        // recorded pair and nothing pathological happens.
        assert_eq!(names(&t, &ids(&t, "ab")), vec!["ab"]);

        // With "aa" available, "aa" gets merged and can leave `rev_merge` entries
        // pointing at dead symbols. The result is still well-formed: every input
        // byte is accounted for by exactly one emitted id.
        let t = spm(&Builder::new()
            .no_prefix()
            .tok("a", -900.0)
            .tok("b", -900.0)
            .tok("aa", -0.5)
            .tok("ab", -1.0));
        for text in ["aab", "abaa", "aaaab", "aabaab"] {
            let got = ids(&t, text);
            let rendered: String = got
                .iter()
                .filter_map(|&i| t.piece(i))
                .filter(|p| !p.starts_with("<0x"))
                .collect();
            assert_eq!(rendered, text, "{text:?} did not round-trip: {got:?}");
        }
    }

    /// `llama_detokenize`: the leading-space strip, the BOS/EOS skip, and the
    /// attribute dispatch all matter here, and none of them are visible in
    /// `encode`.
    #[test]
    fn detokenize_strips_one_leading_space_and_only_from_the_first_token() {
        // Prefix off: nothing is stripped and markers render as spaces.
        let b = Builder::new()
            .no_prefix()
            .tok("▁", -100.0)
            .tok("▁he", -1.0)
            .tok("▁▁x", -1.0)
            .tok("hi", -1.0);
        let t = spm(&b);
        assert!(!t.add_space_prefix());
        assert_eq!(t.detokenize(&[b.id_of("▁he")], false), " he");
        assert_eq!(t.detokenize(&[b.id_of("▁▁x")], false), "  x");
        assert_eq!(t.detokenize(&[], true), "");
        // An id outside the vocabulary is skipped rather than panicking.
        assert_eq!(t.detokenize(&[i32::MAX, b.id_of("▁he")], false), " he");

        // Prefix on: the first token loses exactly one leading space, and only
        // one — `_try_copy` strips `lstrip` characters, not all of them.
        let b2 = Builder::new()
            .tok("▁", -100.0)
            .tok("▁he", -1.0)
            .tok("▁▁x", -1.0)
            .tok("hi", -1.0);
        let t2 = spm(&b2);
        assert!(t2.add_space_prefix());
        assert_eq!(t2.detokenize(&[b2.id_of("▁he")], false), "he");
        assert_eq!(t2.detokenize(&[b2.id_of("▁▁x")], false), " x");
        assert_eq!(
            t2.detokenize(&[b2.id_of("▁he"), b2.id_of("hi")], false),
            "hehi"
        );
    }

    #[test]
    fn detokenize_skips_bos_and_eos_and_renders_specials_as_nothing() {
        // `<eot>` is the first token after the 256 byte tokens and the three
        // control ones, so it lands at 259 + 2 here.
        let eos = 261i32;
        let b = Builder::new()
            .tok("▁", -100.0)
            .tok("▁he", -1.0)
            .raw("<eot>", 0.0, TOKEN_TYPE_CONTROL)
            .key("tokenizer.ggml.add_eos_token", Value::Bool(true))
            .key("tokenizer.ggml.eos_token_id", Value::U64(eos as u64));
        let t = spm(&b);

        // A leading BOS is skipped, and being a BOS it disables the leading-space
        // strip: the prompt already carries its own boundary.
        assert_eq!(t.detokenize(&[1, b.id_of("▁he")], true), " he");
        // With remove_special off the BOS is *not* skipped, so the one-shot strip
        // is spent on a CONTROL token that renders as nothing and the first real
        // token keeps its space. Upstream behaves the same way; it is why
        // `common_detokenize` passes remove_special = true.
        assert_eq!(t.detokenize(&[1, b.id_of("▁he")], false), " he");
        assert_eq!(t.detokenize(&[b.id_of("▁he")], false), "he");
        // CONTROL and UNKNOWN render as nothing whatever `remove_special` says,
        // because `unparse_special` is false on every path here.
        assert_eq!(t.detokenize(&[b.id_of("<eot>")], true), "");
        assert_eq!(t.detokenize(&[UNK], false), "");
        // A trailing EOS is dropped by remove_special, and renders as nothing
        // either way.
        assert_eq!(t.detokenize(&[b.id_of("▁he"), eos], true), "he");
        assert_eq!(t.detokenize(&[eos], true), "");
        assert_eq!(t.detokenize(&[eos], false), "");
    }

    #[test]
    fn a_user_defined_token_renders_verbatim_and_a_byte_token_as_its_byte() {
        let b = Builder::new()
            .raw("<|ud|>", 0.0, TOKEN_TYPE_USER_DEFINED)
            .tok("▁a", -1.0);
        let t = spm(&b);
        let a = t.byte_token(0xFE).unwrap();
        // The base vocabulary already declares all 256 `<0xNN>` entries as BYTE.
        assert_eq!(t.attr_of(a) & ATTR_BYTE, ATTR_BYTE);
        // USER_DEFINED keeps its markers and angle brackets; NORMAL un-escapes;
        // BYTE emits the payload.
        assert_eq!(t.detokenize(&[b.id_of("<|ud|>")], false), "<|ud|>");
        // The payload is the raw byte, which is not valid UTF-8 on its own; the
        // lossy conversion happens once, at the text boundary.
        assert_eq!(t.piece_bytes(a, 0), vec![0xFEu8]);
        assert_eq!(t.detokenize(&[a], false), "\u{fffd}");
        assert_eq!(t.detokenize(&[b.id_of("▁a")], false), "a");
        // lstrip only applies to NORMAL text; a BYTE payload and a USER_DEFINED
        // spelling are copied verbatim.
        assert_eq!(t.piece_bytes(a, 1), vec![0xFEu8]);
        assert_eq!(t.piece_bytes(b.id_of("<|ud|>"), 1), b"<|ud|>".to_vec());
        assert_eq!(t.piece_bytes(UNK, 0), Vec::<u8>::new());
    }

    /// `detokenize(encode(x))` is *not* the identity for an input containing a
    /// literal U+2581: encoding cannot tell one from a space, so it comes back as
    /// one. That asymmetry is upstream's, not this port's, and it is the same
    /// reason the empty prompt has to be special-cased.
    #[test]
    fn encode_decode_round_trips_plain_text_but_not_a_literal_marker() {
        let b = Builder::new()
            .no_prefix()
            .tok("▁", -100.0)
            .tok("▁he", -1.0)
            .tok("he", -1.0)
            .tok("hi", -1.0);
        let t = spm(&b);
        for text in ["hello", "he", "", "x y"] {
            assert_eq!(
                t.detokenize(&ids(&t, text), false),
                text,
                "round trip failed for {text:?}"
            );
        }
        // A literal marker goes in as a space and comes out as one.
        assert_eq!(t.detokenize(&ids(&t, "a\u{2581}b"), false), "a b");
    }

    #[test]
    fn an_empty_token_text_is_replaced_and_a_duplicate_is_refused() {
        // llama.cpp warns and substitutes a placeholder; an empty token would
        // otherwise be reachable through `text_to_token` and expand to nothing.
        let t = spm(&Builder::new().raw("", -1.0, 1).tok("a", -2.0));
        assert_eq!(t.piece(259), Some("[EMPTY_259]"));
        assert_eq!(t.piece_id(""), None);

        // A duplicate is the one shape that makes upstream's
        // `id_to_token.size() == token_to_id.size()` assert fire, so it is an
        // error here rather than a silent shadowing.
        let mut dup = HashMap::new();
        dup.insert("tokenizer.ggml.model".into(), Value::Str("llama".into()));
        dup.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(vec![Value::Str("a".into()), Value::Str("a".into())]),
        );
        assert_eq!(
            Spm::from_metadata(&dup).err(),
            Some(Error::Malformed("duplicate token text"))
        );
    }

    #[test]
    fn a_scores_array_of_the_wrong_length_is_refused() {
        let mut md = HashMap::new();
        md.insert("tokenizer.ggml.model".into(), Value::Str("llama".into()));
        md.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(vec![Value::Str("a".into()), Value::Str("b".into())]),
        );
        md.insert(
            "tokenizer.ggml.scores".into(),
            Value::Array(vec![Value::F32(-1.0), Value::F32(-2.0), Value::F32(-3.0)]),
        );
        assert_eq!(
            Spm::from_metadata(&md).err(),
            Some(Error::Malformed("scores and tokens differ in length"))
        );
    }

    #[test]
    fn a_missing_bos_id_with_add_bos_set_ends_the_encode() {
        // GGML_ASSERT(special_bos_id != LLAMA_TOKEN_NULL) upstream. Returning
        // nothing keeps a broken GGUF from panicking mid-prompt; the engine's
        // main() then exits on the empty id list.
        let t = spm(&Builder::new()
            .omit("tokenizer.ggml.bos_token_id")
            .tok("▁", -100.0));
        assert!(t.encode("hello", true, false).is_empty());
        assert!(!t.encode("hello", false, false).is_empty());
    }
}
