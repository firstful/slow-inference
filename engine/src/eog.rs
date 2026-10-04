//! llama.cpp's `special_eog_ids`: the set of token ids that end a generation.
//!
//! Upstream's answer to "when do we stop" is derived from *token text*, not from
//! the GGUF, and the derivation is 160 lines of `llama-vocab.cpp` with two
//! order-dependent fixes at the end. It was missing here even though the engine
//! had ported the neighbouring block: tick 26 implemented the CONTROL promotion
//! that the EOG block performs on its way through, and the comment it left said
//! so — "This engine models only its CONTROL side; the id set itself drives
//! generation stopping, which it does not do." This module is the other half.
//!
//! The rules, in the order `llama_vocab::impl::load` applies them (line numbers
//! are the checkout recorded in [`crate::special_aliases::SOURCE_SHA256`]):
//!
//! 1. `special_fim_pad_id`, `special_fim_rep_id`, `special_fim_sep_id` are
//!    inserted, each already resolved by the key reader or by the text
//!    auto-detect (2788-2800). **Auto-detected**, so a Qwen2 vocabulary's
//!    `<|fim_pad|>` and `<|file_sep|>` reach the set by spelling alone even
//!    though neither is in the EOG alias list.
//! 2. Every token whose text is in [`crate::special_aliases::EOG_ALIASES`] is
//!    inserted, and promoted to CONTROL if it was not (2802-2834).
//! 3. `special_eos_id`, `special_eot_id`, `special_eom_id` are inserted if they
//!    are not already (2859-2872). `eos` has a per-tokenizer-model *default*,
//!    which is why this is not "the key says so": a GGUF with no
//!    `tokenizer.ggml.eos_token_id` still stops on id 2 for an SPM vocabulary.
//! 4. o200k_harmony / solar-open: a vocabulary with tool-call markers erases
//!    `<|end|>` from the set and *assigns* it USER_DEFINED (2877-2916).
//! 5. gemma4 / paddleocr: a vocabulary with `<|tool_response>` erases `</s>`
//!    and *assigns* it NORMAL (2918-2946).
//!
//! Steps 4 and 5 are why this is a struct and not a `Vec<i32>`: the erasures
//! come with attribute changes that outlive them. Assigning `</s>` NORMAL takes
//! it out of `cache_special_tokens` entirely, so `parse_special = true` stops
//! cutting on it — a tokenization change, not just a stopping change. Upstream
//! builds that cache at 2951, after all of the above.
//!
//! No vocabulary on disk reaches steps 4 or 5: that is what the synthetic
//! tests in this file are for, and it is why they exist here rather than in
//! `tests/tokenizer_parity.rs`.

use crate::gguf;
use crate::tokenizer::SpecialPromotions;
use std::collections::{BTreeSet, HashMap};

/// An attribute llama.cpp *assigns* (not ORs) in one of the two workaround
/// blocks. Assignment matters: the EOG block ORs CONTROL onto a token, and the
/// fixes below replace the whole attribute, clearing the CONTROL bit again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Forced {
    /// `attr = LLAMA_TOKEN_ATTR_USER_DEFINED` — `<|end|>`, o200k_harmony.
    UserDefined,
    /// `attr = LLAMA_TOKEN_ATTR_NORMAL` — `</s>`, gemma4/paddleocr.
    Normal,
}

/// The derived set, plus the attributes that go with it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Eog {
    /// Ascending ids. Upstream's is a `std::set<llama_token>`, so this order is
    /// the upstream order and the only one a comparison can use.
    pub ids: Vec<i32>,
    /// `(id, attribute)` pairs to assign after the promotion pass, in upstream
    /// order. At most two entries, one per workaround that fires.
    pub forced: Vec<(i32, Forced)>,
}

impl Eog {
    /// `llama_vocab::impl::is_eog` (3101).
    pub fn is_eog(&self, id: i32) -> bool {
        self.ids.binary_search(&id).is_ok()
    }

    /// What `forced` says about `id`, if anything.
    pub fn forced_attr(&self, id: i32) -> Option<Forced> {
        self.forced.iter().find(|(i, _)| *i == id).map(|(_, a)| *a)
    }
}

/// `special_eos_id`'s default, keyed on `tokenizer.ggml.model` (1955-2110).
///
/// The declaration default is 2 and every recognised model overwrites it, so the
/// table is not redundant: WPM and T5 have no default at all, and BPE defaults
/// to 11 rather than 2. A model whose tokenizer this engine cannot load is not
/// representable here — upstream throws on an unknown `tokenizer.ggml.model` —
/// so the fallback is the declaration default, which is what upstream would
/// have used had the switch not run.
pub fn default_eos_id(model: &str) -> Option<i32> {
    match model {
        "no_vocab" | "none" | "bert" | "rwkv" | "gemma4" => None,
        "gpt2" | "hybriddna" | "whitespace" => Some(11),
        "t5" => Some(1),
        "llama" | "plamo2" => Some(2),
        _ => Some(2),
    }
}

/// Read `tokenizer.ggml.<slot>_token_id`, with upstream's range check: a
/// declared id that is not a valid token is *ignored*, leaving whatever default
/// was in place (2537-2552). `n_tokens` is the vocabulary size.
fn key_id(md: &HashMap<String, gguf::Value>, slot: &str, n_tokens: usize) -> Option<i32> {
    let key = format!("tokenizer.ggml.{slot}_token_id");
    let v = md.get(&key)?.as_u64()?;
    let id = u32::try_from(v).ok()? as i32;
    (id >= 0 && (id as usize) < n_tokens).then_some(id)
}

/// `special_<slot>_id` after both the key reader and the text auto-detect.
///
/// The key wins when it is present *and* in range. A key that is present and out
/// of range leaves the default in place, which for these slots is `NULL` — and a
/// `NULL` is exactly the state the text auto-detect's `== LLAMA_TOKEN_NULL` guard
/// tests, so the spelling still wins. Reading the key twice to find that out
/// would be wrong; here it falls through on purpose.
fn slot_id(
    md: &HashMap<String, gguf::Value>,
    promo: &SpecialPromotions,
    slot: &str,
    n_tokens: usize,
) -> Option<i32> {
    if let Some(id) = key_id(md, slot, n_tokens) {
        return Some(id);
    }
    promo
        .slot_ids
        .iter()
        .find(|(s, _)| *s == slot)
        .map(|(_, id)| *id as i32)
}

/// The text of `id` in the EOG set, `None` if it is not in the set. Upstream
/// reads `id_to_token[tid].text` and skips ids outside the vocabulary (2887,
/// 2927) — the same check, kept so a declared-but-absent id cannot panic here.
fn eog_text<'a>(tokens: &'a [String], set: &BTreeSet<i32>, id: i32) -> Option<&'a str> {
    if !set.contains(&id) {
        return None;
    }
    usize::try_from(id)
        .ok()
        .and_then(|i| tokens.get(i))
        .map(String::as_str)
}

/// Derive `special_eog_ids` and the attributes that go with it.
///
/// `tokens` must be the vocabulary in id order *after* the zero-length rewrite
/// (`"" -> "[EMPTY_n]"`), because upstream keys every block on the rewritten
/// text. `promo` is [`special_text_promotions`]'s result for the same list, and
/// is read here for the ids its alias blocks chose.
pub fn derive(
    tokens: &[String],
    md: &HashMap<String, gguf::Value>,
    promo: &SpecialPromotions,
) -> Eog {
    let n = tokens.len();
    let mut set: BTreeSet<i32> = BTreeSet::new();

    // (1) fill-in-the-middle markers that pad a response.
    for slot in ["fim_pad", "fim_rep", "fim_sep"] {
        if let Some(id) = slot_id(md, promo, slot, n) {
            set.insert(id);
        }
    }

    // (2) end-of-generation spellings. Every match, not the first: this is the
    // one unguarded loop, and a vocabulary may spell several.
    for (id, text) in tokens.iter().enumerate() {
        if crate::special_aliases::EOG_ALIASES.contains(&text.as_str()) {
            set.insert(id as i32);
        }
    }

    // (3) the three ids the GGUF (or the model defaults) name. Each is inserted
    // only if absent, and llama.cpp warns when that happens — a vocabulary whose
    // declared EOS is spelled like nothing in the alias list still stops there.
    let model = md
        .get("tokenizer.ggml.model")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let eos = key_id(md, "eos", n).or_else(|| {
        // The declaration default is only dropped when it is not a real token.
        default_eos_id(model).filter(|&d| (d as usize) < n)
    });
    for id in [
        eos,
        slot_id(md, promo, "eot", n),
        slot_id(md, promo, "eom", n),
    ]
    .into_iter()
    .flatten()
    {
        set.insert(id);
    }

    // (4) o200k_harmony and solar-open both use `<|end|>` for a *turn* that is
    // not a generation end, so they erase it and let it render as text.
    let mut forced: Vec<(i32, Forced)> = Vec::new();
    let (mut has_return, mut has_call, mut has_end, mut has_flush) = (false, false, false, false);
    let mut end_id = None;
    for id in set.iter().copied() {
        match eog_text(tokens, &set, id) {
            Some("<|return|>") => has_return = true,
            Some("<|call|>") | Some("<|calls|>") => has_call = true,
            Some("<|flush|>") => has_flush = true,
            Some("<|end|>") => {
                has_end = true;
                end_id = Some(id);
            }
            _ => {}
        }
    }
    if ((has_return && has_call && has_end) || (has_call && has_flush && has_end))
        && let Some(id) = end_id
    {
        set.remove(&id);
        forced.push((id, Forced::UserDefined));
    }

    // (5) gemma4 and paddleocr put `<|tool_response>` in the set and use `</s>`
    // for something that is not the end of the answer.
    let mut has_tool_response = false;
    let mut s_id = None;
    for id in set.iter().copied() {
        match eog_text(tokens, &set, id) {
            Some("<|tool_response>") => has_tool_response = true,
            Some("</s>") => s_id = Some(id),
            _ => {}
        }
    }
    if has_tool_response && let Some(id) = s_id {
        set.remove(&id);
        forced.push((id, Forced::Normal));
    }

    Eog {
        ids: set.into_iter().collect(),
        forced,
    }
}

/// The index of the first token in `seq` that ends a generation, inclusive.
///
/// llama.cpp's samplers return `true` from `llama_sampler_apply` on an EOG token
/// *after* handing the token back, and `common/sampling.cpp` breaks with the
/// token already in the result — so the stop token is part of the output. With no
/// EOG token in `seq` this is `None`, i.e. "generated all of it".
pub fn stop_at_eog(seq: &[i32], eog: &Eog) -> Option<usize> {
    seq.iter().position(|&id| eog.is_eog(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Value;
    use crate::special_aliases::{CONTROL_ALIAS_GROUPS, EOG_ALIASES};
    use crate::tokenizer::special_text_promotions;

    // ------------------------------------------------------------------
    // how this module refers to token spellings
    //
    // By index into the generated alias tables, never by typing them. A typed
    // copy of a generated table is what drifts: llama.cpp's own spelling of the
    // gemma4 tool-response marker has no closing `|` before the `>`, and the
    // first version of this file "corrected" it while being written — which
    // silently stopped matching, and cost an afternoon. So the spellings live in
    // one generated place (`special_aliases.rs`, checked against upstream by
    // `scripts/check_alias_literals.py`), the indices are pinned below by
    // length, and no alias is transcribed anywhere.
    // ------------------------------------------------------------------

    const EOT_ID: usize = 0;
    const END: usize = 2;
    const RETURN: usize = 3;
    const CALL: usize = 4;
    const FLUSH: usize = 5;
    const CALLS: usize = 6;
    const END_OF_TEXT: usize = 8;
    const S: usize = 9;
    const TOOL_RESPONSE: usize = 19;

    /// The first alias of a guarded block, by slot name — so the FIM tests never
    /// spell `<|fim_pad|>` either.
    fn group(slot: &str) -> &'static [&'static str] {
        CONTROL_ALIAS_GROUPS
            .iter()
            .find(|g| g.slot == slot)
            .map(|g| g.aliases)
            .unwrap_or_else(|| panic!("no alias group for slot {slot}"))
    }

    /// The indices above mean what this module says they mean. Lengths are the
    /// only thing here that can be typed safely.
    #[test]
    fn the_alias_indices_are_the_ones_this_module_assumes() {
        for (i, len) in [
            (EOT_ID, 10),
            (END, 7),
            (RETURN, 10),
            (CALL, 8),
            (FLUSH, 9),
            (CALLS, 9),
            (END_OF_TEXT, 13),
            (S, 4),
            (TOOL_RESPONSE, 16),
        ] {
            assert_eq!(
                EOG_ALIASES[i].len(),
                len,
                "EOG_ALIASES[{i}] is not the spelling these tests mean"
            );
        }
        assert!(EOG_ALIASES.contains(&EOG_ALIASES[S]));
    }

    /// A vocabulary of `fillers` filler tokens plus the given `(text, token_type)`
    /// pairs, as a GGUF metadata map. Type 1 is NORMAL, 3 CONTROL, 4 USER_DEFINED.
    fn vocab(model: &str, fillers: usize, extra: &[(&str, u64)]) -> HashMap<String, gguf::Value> {
        let mut tokens: Vec<Value> = (0..fillers).map(|i| Value::Str(format!("f{i}"))).collect();
        let mut types: Vec<Value> = vec![Value::U64(1); fillers];
        for (t, ty) in extra {
            tokens.push(Value::Str((*t).to_string()));
            types.push(Value::U64(*ty));
        }
        let mut md = HashMap::new();
        md.insert(
            "tokenizer.ggml.model".to_string(),
            Value::Str(model.to_string()),
        );
        md.insert("tokenizer.ggml.tokens".to_string(), Value::Array(tokens));
        md.insert("tokenizer.ggml.token_type".to_string(), Value::Array(types));
        md
    }

    /// The id of `text`, so no test counts filler tokens by hand.
    fn id_of(md: &HashMap<String, gguf::Value>, text: &str) -> i32 {
        md.get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_vec_str())
            .expect("tokens")
            .iter()
            .position(|t| *t == text)
            .unwrap_or_else(|| panic!("{text:?} is not in the vocabulary")) as i32
    }

    fn derive_md(md: &HashMap<String, gguf::Value>) -> Eog {
        let tokens: Vec<String> = md
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_vec_str())
            .expect("tokens")
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let promo = special_text_promotions(&tokens, md);
        derive(&tokens, md, &promo)
    }

    /// `special_eog_ids` for a vocabulary that declares nothing: the
    /// end-of-generation spellings plus the SPM default EOS. This is the shape
    /// of every real llama-2-style vocabulary, and it is why the set cannot be
    /// approximated as "the CONTROL tokens".
    #[test]
    fn default_eos_id_joins_the_text_matches_for_spm() {
        let md = vocab("llama", 4, &[(EOG_ALIASES[S], 3)]);
        let eog = derive_md(&md);
        let s_id = id_of(&md, EOG_ALIASES[S]);
        assert!(eog.is_eog(s_id));
        assert!(eog.is_eog(2), "the SPM default EOS id is not in the set");
        assert_eq!(eog.ids, vec![2, s_id]);
    }

    /// The same declaration under `tokenizer.ggml.model = "gpt2"`: the default
    /// moves from 2 to 11, and 11 is not a token in a three-token vocabulary, so
    /// upstream disables it (2537). A port that reads only the key would get this
    /// wrong in both directions.
    #[test]
    fn an_out_of_range_default_eos_is_disabled() {
        let md = vocab("gpt2", 2, &[(EOG_ALIASES[END_OF_TEXT], 3)]);
        assert_eq!(derive_md(&md).ids, vec![2]);
        // With enough tokens for it, 11 comes back.
        let big = vocab("gpt2", 20, &[(EOG_ALIASES[END_OF_TEXT], 3)]);
        assert_eq!(derive_md(&big).ids, vec![11, 20]);
    }

    /// A declared `tokenizer.ggml.eos_token_id` overrides the default, and one
    /// that is out of range is ignored rather than trusted (2547).
    #[test]
    fn a_declared_eos_key_wins_and_is_range_checked() {
        let mut md = vocab("llama", 2, &[(EOG_ALIASES[S], 3)]);
        md.insert("tokenizer.ggml.eos_token_id".to_string(), Value::U32(0));
        assert_eq!(derive_md(&md).ids, vec![0, 2]);

        md.insert("tokenizer.ggml.eos_token_id".to_string(), Value::U32(99));
        assert_eq!(derive_md(&md).ids, vec![2]);
    }

    /// The FIM trio enters the EOG set by *auto-detection*, not by being in the
    /// EOG alias list: `<|fim_pad|>`, `<|file_sep|>` and `<|fim_repo|>` are in the
    /// fim_pad / fim_sep / fim_rep alias groups, and those groups set the ids
    /// that step (1) inserts. A Qwen2 vocabulary therefore stops on its pad,
    /// repository and file-separator tokens, which is not what anyone reading the
    /// EOG list would expect.
    #[test]
    fn fim_pad_rep_and_sep_enter_by_spelling() {
        let (pad, rep, sep) = (
            group("fim_pad")[0],
            group("fim_rep")[0],
            group("fim_sep")[0],
        );
        let md = vocab("gpt2", 20, &[(pad, 3), (rep, 3), (sep, 3)]);
        let (pad_id, rep_id, sep_id) = (id_of(&md, pad), id_of(&md, rep), id_of(&md, sep));
        assert_eq!(derive_md(&md).ids, vec![11, pad_id, rep_id, sep_id]);

        // Declaring the key points the id somewhere else, and *that* is what is
        // inserted: the auto-detect is suppressed, not merely outranked.
        let mut declared = md.clone();
        declared.insert("tokenizer.ggml.fim_pad_token_id".to_string(), Value::U32(0));
        assert_eq!(derive_md(&declared).ids, vec![0, 11, rep_id, sep_id]);
    }

    /// Step (4): o200k_harmony has `<|return|>`, `<|call|>` and `<|end|>`, so
    /// `<|end|>` stops being an end and becomes USER_DEFINED — which the caller
    /// has to apply, because USER_DEFINED is cut even with `parse_special =
    /// false`. Without the fix, harmony would stop mid-turn at every message
    /// boundary.
    #[test]
    fn harmony_erases_end_and_marks_it_user_defined() {
        let md = vocab(
            "gpt2",
            20,
            &[
                (EOG_ALIASES[RETURN], 3),
                (EOG_ALIASES[CALL], 3),
                (EOG_ALIASES[END], 3),
            ],
        );
        let end_id = id_of(&md, EOG_ALIASES[END]);
        let eog = derive_md(&md);
        assert_eq!(
            eog.ids,
            vec![
                11,
                id_of(&md, EOG_ALIASES[RETURN]),
                id_of(&md, EOG_ALIASES[CALL])
            ]
        );
        assert_eq!(eog.forced, vec![(end_id, Forced::UserDefined)]);
        assert!(!eog.is_eog(end_id));
        assert_eq!(eog.forced_attr(end_id), Some(Forced::UserDefined));
    }

    /// solar-open spells the same two markers `<|calls|>` and `<|flush|>`, and
    /// the fix has to fire on that combination too.
    #[test]
    fn solar_open_takes_the_other_branch() {
        let md = vocab(
            "gpt2",
            20,
            &[
                (EOG_ALIASES[CALLS], 3),
                (EOG_ALIASES[FLUSH], 3),
                (EOG_ALIASES[END], 3),
            ],
        );
        let end_id = id_of(&md, EOG_ALIASES[END]);
        let eog = derive_md(&md);
        assert_eq!(
            eog.ids,
            vec![
                11,
                id_of(&md, EOG_ALIASES[CALLS]),
                id_of(&md, EOG_ALIASES[FLUSH])
            ]
        );
        assert_eq!(eog.forced, vec![(end_id, Forced::UserDefined)]);
    }

    /// Both fixes can fire on one vocabulary: a model with the turn markers, the
    /// tool-response marker and `</s>` loses `<|end|>` *and* `</s>`, in upstream
    /// order — which matters, because the second one's assignment is what makes
    /// the first one's token stop being CONTROL.
    #[test]
    fn both_workarounds_can_fire_on_one_vocabulary() {
        let md = vocab(
            "gpt2",
            20,
            &[
                (EOG_ALIASES[RETURN], 3),
                (EOG_ALIASES[CALL], 3),
                (EOG_ALIASES[END], 3),
                (EOG_ALIASES[TOOL_RESPONSE], 3),
                (EOG_ALIASES[S], 3),
            ],
        );
        let eog = derive_md(&md);
        assert_eq!(
            eog.ids,
            vec![
                11,
                id_of(&md, EOG_ALIASES[RETURN]),
                id_of(&md, EOG_ALIASES[CALL]),
                id_of(&md, EOG_ALIASES[TOOL_RESPONSE]),
            ]
        );
        assert_eq!(
            eog.forced,
            vec![
                (id_of(&md, EOG_ALIASES[END]), Forced::UserDefined),
                (id_of(&md, EOG_ALIASES[S]), Forced::Normal),
            ]
        );
        assert!(!eog.is_eog(id_of(&md, EOG_ALIASES[S])));
    }

    /// Step (5) alone: the tool-response marker without the turn markers.
    /// `</s>` leaves the set *and* is assigned NORMAL, which is the stronger half
    /// — NORMAL also takes it out of `cache_special_tokens`, so `parse_special =
    /// true` stops cutting on it. gemma4 is the model this exists for.
    #[test]
    fn tool_response_alone_erases_and_normalizes_s() {
        let md = vocab(
            "gpt2",
            20,
            &[(EOG_ALIASES[S], 3), (EOG_ALIASES[TOOL_RESPONSE], 3)],
        );
        let s_id = id_of(&md, EOG_ALIASES[S]);
        let eog = derive_md(&md);
        assert_eq!(eog.ids, vec![11, id_of(&md, EOG_ALIASES[TOOL_RESPONSE])]);
        assert_eq!(eog.forced, vec![(s_id, Forced::Normal)]);
        assert!(!eog.is_eog(s_id));
    }

    /// Neither marker: `</s>` stays an end. This is the ordinary case, and the
    /// one the engine was accidentally right about before.
    #[test]
    fn without_the_markers_s_is_untouched() {
        let md = vocab("llama", 2, &[(EOG_ALIASES[S], 3)]);
        let eog = derive_md(&md);
        assert!(eog.is_eog(id_of(&md, EOG_ALIASES[S])));
        assert!(eog.forced.is_empty());
    }

    /// The `<|end|>` fix needs *two* markers. One of the two is an ordinary
    /// vocabulary, and a port that fired on a single marker would erase
    /// `<|end|>` — and demote it out of CONTROL — for every model that has it.
    #[test]
    fn one_marker_is_not_enough_to_erase_end() {
        let md = vocab(
            "gpt2",
            20,
            &[(EOG_ALIASES[RETURN], 3), (EOG_ALIASES[END], 3)],
        );
        let eog = derive_md(&md);
        assert!(eog.is_eog(id_of(&md, EOG_ALIASES[END])));
        assert!(eog.forced.is_empty());
    }

    /// `special_eot_id` is inserted by name whatever its text — which is the whole
    /// point of reading the key rather than the spelling (2864). The key also
    /// *suppresses* the text auto-detect, so the id it names and the id the
    /// spelling would have chosen can be different tokens.
    #[test]
    fn a_declared_eot_id_stops_generation_whatever_its_text() {
        let mut md = vocab(
            "llama",
            2,
            &[(EOG_ALIASES[S], 3), (EOG_ALIASES[EOT_ID], 3), ("STOP", 3)],
        );
        md.insert("tokenizer.ggml.eot_token_id".to_string(), Value::U32(4));
        let eog = derive_md(&md);
        assert!(eog.is_eog(4), "the declared eot id is not in the set");
        assert!(eog.is_eog(2));
        // The *other* spelling is in the set too, and not because of the key:
        // step (2) is a plain alias lookup, so suppressing the auto-detect does
        // not remove it. Only `special_eot_id` itself is key-driven.
        assert!(eog.is_eog(id_of(&md, EOG_ALIASES[EOT_ID])));
    }

    /// The stop token is part of the output: llama.cpp's sampler hands the token
    /// back and *then* reports the stop, so truncating must keep it.
    #[test]
    fn the_stop_token_is_inclusive() {
        let eog = Eog {
            ids: vec![7],
            forced: Vec::new(),
        };
        assert_eq!(stop_at_eog(&[1, 2, 7, 3], &eog), Some(2));
        assert_eq!(stop_at_eog(&[1, 2, 3], &eog), None);
        assert_eq!(stop_at_eog(&[], &eog), None);
    }
}
