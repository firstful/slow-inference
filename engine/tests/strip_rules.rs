// strip_rules.rs — the LSTRIP/RSTRIP attribute rules in `tokenizer_st_partition`.
//
// What this file is for
// ---------------------
// `tests/tokenizer_parity.rs` compares token *ids*. That cannot see these
// rules, for a reason worth stating because it is not obvious: LSTRIP trims
// whitespace off a raw fragment next to a special token, and every synthetic
// fixture in this repo has a trivial merge table, so the surrounding text
// encodes to nothing and trimming it changes no ids. Measured on
// models/synthetic-gpt2-strip.gguf, `a <mask> b` and `a<mask>b` produce
// byte-identical id lists with and without the jina-v2-code label that sets
// LSTRIP. An id oracle would call that a pass.
//
// So the oracle here is the attribute itself: `engine/tools/gen_ref_attrs.c`
// dumps `llama_vocab_get_attr` for every id of four GGUFs that differ only in
// one or two metadata strings, and this test replays the derivation against
// them. Two of the four rules are reachable and checked for exact agreement; the
// third is unreachable and checked for its absence, with the reason.
//
// Vocabulary, not ids
// -------------------
// `models/synthetic-gpt2-strip.gguf` is `synthetic-gpt2.gguf` with one list
// changed — `tokenizer.ggml.tokens` — to carry the five tokens the rules name.
// The label under test (`general.name`, `tokenizer.ggml.pre`) is a *string*
// metadata key, so the test rewrites it in the parsed metadata map rather than
// shipping four near-identical binaries. `scripts/gguf_patch_kv.py` is the same
// edit on the file, which is how the `.attrs` references were produced; it
// round-trips 7 of the repo's 8 GGUFs byte-identically, so the map edit and the
// file edit are the same edit.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use slow_inference::gguf;
use slow_inference::tokenizer::{Bpe, Fragment};

/// `models/synthetic-gpt2-strip.gguf`'s strip-rule tokens, by index. Never by
/// text: tick 29 lost four literals to a harness that renders `<|...|>` as
/// markup, and a misspelled token is a test that silently stops testing.
const ID_MASK: i32 = 0;
const ID_BRACKET_MASK: i32 = 1;
const ID_S: i32 = 2;
const ID_UNK: i32 = 3;
const ID_BOS: i32 = 4;
const N_SPECIALS: usize = 5;

const LLAMA_TOKEN_ATTR_LSTRIP: i32 = 1 << 7;
const LLAMA_TOKEN_ATTR_RSTRIP: i32 = 1 << 8;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("models")
        .join("synthetic-gpt2-strip.gguf")
}

fn metadata() -> HashMap<String, gguf::Value> {
    let path = fixture();
    assert!(
        path.exists(),
        "missing {} — run scripts/generate_synthetic_fixtures.py --arch gpt2-strip",
        path.display()
    );
    gguf::parse(&path.to_string_lossy())
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
        .metadata
}

/// `Bpe::from_metadata` under one label. `overrides` are the metadata strings
/// the four reference GGUFs differ by.
fn labelled(overrides: &[(&str, &str)]) -> Bpe {
    let mut md = metadata();
    for (k, v) in overrides {
        md.insert((*k).to_string(), gguf::Value::Str((*v).to_string()));
    }
    Bpe::from_metadata(&md).expect("strip fixture is a BPE vocabulary")
}

/// The ids whose `llama_token_attr` carries `flag`, ascending, from the named
/// `.attrs` reference file.
fn ids_with(ref_file: &str, flag: i32) -> Vec<i32> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("ref_attrs")
        .join(ref_file);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut out: Vec<i32> = text
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            if it.next()? != "ATTR" {
                return None;
            }
            let id: i32 = it.next()?.parse().ok()?;
            let mask: i32 = it.next()?.parse().ok()?;
            (mask & flag != 0).then_some(id)
        })
        .collect();
    out.sort_unstable();
    out
}

fn fragments(tok: &Bpe, text: &str, parse_special: bool) -> Vec<Fragment> {
    tok.partition(text, parse_special)
}

fn raw(s: &str) -> Fragment {
    Fragment::Raw(s.to_string())
}
fn sp(id: i32) -> Fragment {
    Fragment::Special(id)
}

/// The fixture carries exactly the five strip-rule tokens and nothing else that
/// is special, so any id other than these five in a strip set means a rule
/// matched something it should not have.
fn assert_only_strip_rule_ids(got: &[i32]) {
    for id in got {
        assert!(
            (0..N_SPECIALS as i32).contains(id),
            "strip flag set on id {id}, which is not a strip-rule token"
        );
    }
}

/// The unlabelled fixture must set no strip flags at all. Without this, a
/// derivation that fired unconditionally would still pass the two labelled
/// tests — it would just also set the flag on the wrong id in the baseline.
#[test]
fn the_unlabelled_fixture_strips_nothing() {
    let tok = labelled(&[]);
    let (lstrip, rstrip) = tok.strip_ids();
    assert_eq!(
        lstrip,
        Vec::<i32>::new(),
        "baseline sets LSTRIP with no label"
    );
    assert_eq!(
        rstrip,
        Vec::<i32>::new(),
        "baseline sets RSTRIP with no label"
    );
}

/// Rule 1: `tokenizer.ggml.pre` containing a jina-v2-* string sets LSTRIP on
/// `<mask>` and on nothing else. Pinned against llama.cpp's own attributes.
#[test]
fn jina_v2_pre_sets_lstrip_on_the_angle_bracket_mask_only() {
    let (lstrip, rstrip) = labelled(&[("tokenizer.ggml.pre", "jina-v2-code")]).strip_ids();
    assert_eq!(lstrip, ids_with("jina.attrs", LLAMA_TOKEN_ATTR_LSTRIP));
    assert_eq!(lstrip, vec![ID_MASK]);
    assert_eq!(
        rstrip,
        Vec::<i32>::new(),
        "no rule can set RSTRIP — see phi3"
    );
    assert_only_strip_rule_ids(&lstrip);
}

/// Upstream matches a three-entry substring list, not an equality. Two of the
/// three cannot be *reached* from the engine today, for a reason that has
/// nothing to do with the strips and is recorded here rather than papered over:
/// `pre_type_from_str` is missing ten spellings that llama.cpp groups under
/// `LLAMA_VOCAB_PRE_TYPE_GPT2` (llama-vocab.cpp:2175) — `gpt-2`, `phi-2`,
/// `jina-es`, `jina-de`, `gigachat`, `jina-v2-es`, `jina-v2-de`, `a.x-4.0`,
/// `mellum`, `modern-bert` — so a GGUF declaring any of them is refused before
/// a single attribute is derived.
///
/// The same gap runs the other way, and it cost this file a fixture: the engine
/// accepts `gpt2`, which upstream does not — llama.cpp spells it `gpt-2`
/// (llama-vocab.cpp:2174) and answers "unknown pre-tokenizer type" to `gpt2`.
/// Measured, by trying it. So the engine's match arm and upstream's are not a
/// subset relation in either direction, and neither can be extended by copying
/// the other. `mpt` is the value the strip fixture uses, chosen because it is in
/// both and triggers no rule.
///
/// They are deliberately still absent from the engine. Three of the spellings
/// the engine *does* accept in that group (`jina-v1-en`, `jina-v2-code`,
/// `roberta-bpe`) also set `add_sep = true` upstream, which escapes spaces as
/// `<0x20>` during tokenization, and this engine's `PreType::Gpt2` carries no
/// such flag. Adding the names would turn a loud "I cannot load this" into a
/// silent wrong answer for ten more families. That is a ROADMAP 8.4 item and it
/// needs `add_sep` modelled first, not just a longer match arm.
///
/// This test pins the one spelling that is reachable, and pins that the other
/// two are refused — so the day the pre-group is completed, this fails and says
/// what to finish.
#[test]
fn only_the_reachable_jina_v2_spelling_loads_and_the_other_two_are_refused() {
    let (lstrip, _) = labelled(&[("tokenizer.ggml.pre", "jina-v2-code")]).strip_ids();
    assert_eq!(lstrip, vec![ID_MASK]);

    let mut md = metadata();
    for pre in ["jina-v2-de", "jina-v2-es"] {
        md.insert("tokenizer.ggml.pre".into(), gguf::Value::Str(pre.into()));
        let err = match Bpe::from_metadata(&md) {
            Err(e) => e,
            Ok(_) => panic!("pre_type_from_str accepted pre={pre}"),
        };
        assert!(
            matches!(err, slow_inference::tokenizer::Error::UnsupportedPre(ref v) if v == pre),
            "pre={pre}: {err:?}"
        );
    }
}

/// The architecture half of rule 1: `general.architecture` naming nomic-bert-moe
/// or jina-bert-v3 sets the same flag with no `pre` label at all.
#[test]
fn the_architecture_half_of_the_rule_needs_no_pre_label() {
    for arch in ["nomic-bert-moe", "jina-bert-v3"] {
        let (lstrip, _) = labelled(&[("general.architecture", arch)]).strip_ids();
        assert_eq!(lstrip, vec![ID_MASK], "arch={arch}");
    }
    // ...and it must not fire for an architecture that merely looks similar.
    let (lstrip, _) = labelled(&[("general.architecture", "nomic-bert")]).strip_ids();
    assert!(
        lstrip.is_empty(),
        "'nomic-bert' is not a substring rule match"
    );
}

/// Rule 2: `general.name` lowercased containing `modern-bert` sets LSTRIP on
/// `[MASK]`. Pinned against llama.cpp's attributes, and checked to be a
/// *different* id from rule 1 — two rules that both landed on id 0 would pass
/// either test alone.
#[test]
fn modern_bert_name_sets_lstrip_on_the_bracketed_mask_only() {
    let (lstrip, rstrip) = labelled(&[("general.name", "modern-bert-synthetic")]).strip_ids();
    assert_eq!(
        lstrip,
        ids_with("modernbert.attrs", LLAMA_TOKEN_ATTR_LSTRIP)
    );
    assert_eq!(lstrip, vec![ID_BRACKET_MASK]);
    assert_ne!(
        ID_BRACKET_MASK, ID_MASK,
        "the fixture cannot tell the rules apart"
    );
    assert_eq!(rstrip, Vec::<i32>::new());
}

/// `general.name` is lowercased before the substring test, so case must not
/// decide the rule. `MODEL_NAME` would not match upstream.
#[test]
fn the_name_rule_is_case_insensitive_like_upstreams_tolower() {
    let (lstrip, _) = labelled(&[("general.name", "Modern-BERT-Synthetic")]).strip_ids();
    assert_eq!(lstrip, vec![ID_BRACKET_MASK]);
}

/// Rule 3 is unreachable in llama.cpp, so the engine sets nothing and says so.
/// Asserting the *absence* is the only honest form: a port that implemented the
/// loop would look like parity and could never be checked against anything,
/// because no file makes upstream execute it.
#[test]
fn phi3_rstrip_is_unreachable_upstream_so_the_engine_sets_no_flag() {
    let (lstrip, rstrip) = labelled(&[("general.name", "phi-3-synthetic")]).strip_ids();
    assert_eq!(
        (lstrip, rstrip),
        (Vec::<i32>::new(), Vec::<i32>::new()),
        "the phi-3 branch cannot run upstream — see phi3.attrs"
    );
    // The reference file has to agree, or this test is asserting the engine's
    // choice rather than upstream's behaviour.
    let ref_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("ref_attrs")
        .join("phi3.attrs");
    let text = std::fs::read_to_string(&ref_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", ref_path.display()));
    assert!(
        text.contains("LOAD_FAILED") && text.contains("unordered_map::at"),
        "{} should record that llama.cpp refuses to load a phi-3 GGUF",
        ref_path.display()
    );
}

// --- behaviour: what the flag does to the fragments -----------------------
//
// These are the assertions that would catch a partition that accepts the flags
// and ignores them. Ids cannot: see the file header.

#[test]
fn lstrip_removes_the_whitespace_to_the_left_of_the_token() {
    let tok = labelled(&[("tokenizer.ggml.pre", "jina-v2-code")]);
    assert_eq!(
        fragments(&tok, "a <mask> b", true),
        vec![raw("a"), sp(ID_MASK), raw(" b")]
    );
    // Same input, flag off: the space survives as its own fragment boundary.
    let plain = labelled(&[]);
    assert_eq!(
        fragments(&plain, "a <mask> b", true),
        vec![raw("a "), sp(ID_MASK), raw(" b")]
    );
}

#[test]
fn a_run_of_spaces_that_is_only_before_the_token_disappears_entirely() {
    let tok = labelled(&[("tokenizer.ggml.pre", "jina-v2-code")]);
    // The post-strip emptiness check is the half that is easy to omit: emitting
    // an empty Raw instead of nothing is invisible in the ids and visible here.
    assert_eq!(
        fragments(&tok, "    <mask> b", true),
        vec![sp(ID_MASK), raw(" b")]
    );
    assert_eq!(fragments(&tok, "    <mask>", true), vec![sp(ID_MASK)]);
}

#[test]
fn lstrip_stops_at_the_first_non_space_and_leaves_the_rest() {
    let tok = labelled(&[("tokenizer.ggml.pre", "jina-v2-code")]);
    assert_eq!(
        fragments(&tok, "a    <mask> b", true),
        vec![raw("a"), sp(ID_MASK), raw(" b")]
    );
    // Only *trailing* whitespace of the left reminder goes; leading and interior
    // whitespace is untouched, so `"  a  <mask>"` keeps its leading run.
    assert_eq!(
        fragments(&tok, "  a  <mask>", true),
        vec![raw("  a"), sp(ID_MASK)]
    );
}

/// `isspace` in the C locale: six ASCII bytes, not Unicode whitespace. A port
/// using `char::is_whitespace` would eat the non-breaking space here, and that is
/// a real behavioural difference on any European-language prompt.
#[test]
fn the_stripped_set_is_the_six_ascii_bytes_and_nothing_else() {
    let tok = labelled(&[("tokenizer.ggml.pre", "jina-v2-code")]);
    for (ws, code) in [
        (" ", "0x20"),
        ("\t", "0x09"),
        ("\n", "0x0A"),
        ("\u{b}", "0x0B"),
        ("\u{c}", "0x0C"),
        ("\r", "0x0D"),
    ] {
        assert_eq!(
            fragments(&tok, &format!("a{ws}<mask> b"), true),
            vec![raw("a"), sp(ID_MASK), raw(" b")],
            "{code} is C-locale whitespace"
        );
    }
    for (ws, code) in [
        ("\u{a0}", "U+00A0"),
        ("\u{2028}", "U+2028"),
        ("\u{2003}", "U+2003"),
        ("\u{85}", "U+0085"),
    ] {
        // The left reminder keeps it *attached*, because LSTRIP only walks back
        // over whitespace at its end: with a non-space at that position there is
        // nothing to strip and the whole remainder stays as one fragment.
        assert_eq!(
            fragments(&tok, &format!("a{ws}<mask> b"), true),
            vec![raw(&format!("a{ws}")), sp(ID_MASK), raw(" b")],
            "{code} is whitespace to Rust but not to llama.cpp"
        );
    }
}

/// A CONTROL token is only a cut point under `parse_special = true`, so the flag
/// must not make it one. Without this, a partition that consulted LSTRIP before
/// the `parse_special` gate would silently start honouring `<mask>` in bare
/// prompts — a behaviour change in the *default* path.
#[test]
fn lstrip_does_not_make_a_control_token_a_cut_point_on_its_own() {
    let tok = labelled(&[("tokenizer.ggml.pre", "jina-v2-code")]);
    let got = fragments(&tok, "a <mask> b", false);
    assert_eq!(
        got,
        vec![raw("a <mask> b")],
        "parse_special = false must not cut"
    );
}

/// Two occurrences in one fragment must both be cut, with no text between them.
/// This is the case that caught the eager right-reminder: emitting the right
/// reminder and then searching inside it duplicates the second copy and hands
/// it to the pre-tokenizer.
#[test]
fn two_occurrences_in_one_fragment_produce_no_text_between_them() {
    let tok = labelled(&[("tokenizer.ggml.pre", "jina-v2-code")]);
    assert_eq!(
        fragments(&tok, "a <mask><mask> b", true),
        vec![raw("a"), sp(ID_MASK), sp(ID_MASK), raw(" b")]
    );
    // And the same for the unflagged token, which is a regression guard for the
    // partition rather than for the strips.
    assert_eq!(
        fragments(&tok, "a <s><s> b", true),
        vec![raw("a "), sp(ID_BOS), sp(ID_BOS), raw(" b")]
    );
}

/// The longest-match order must survive: `<mask>` is 6 bytes and `<s>` 3, so
/// after the jina label the LSTRIP token is still matched in preference to a
/// shorter one that could match inside it. There is no proper prefix pair in
/// this vocabulary, so this pins the ordering rather than the discrimination.
#[test]
fn the_flag_does_not_disturb_longest_first_matching() {
    let tok = labelled(&[("tokenizer.ggml.pre", "jina-v2-code")]);
    assert_eq!(
        fragments(&tok, "<mask> <s>", true),
        vec![sp(ID_MASK), raw(" "), sp(ID_BOS)]
    );
}

/// The vocabulary carries no `<mask>` in a model that lacks it — upstream warns
/// and continues, so the engine must too rather than treating the absence as
/// "strip everything".
/// RSTRIP appears in no reachable reference, and the two ids the phi-3 branch
/// would have touched carry no flag either. Pinning the absence matters because
/// these constants otherwise have nothing to do: a derivation that set RSTRIP on
/// `</s>` would be invisible to the three tests above, all of which assert on
/// `lstrip` alone.
#[test]
fn no_reachable_reference_sets_rstrip_and_the_phi3_ids_stay_bare() {
    for file in ["baseline.attrs", "jina.attrs", "modernbert.attrs"] {
        assert!(
            ids_with(file, LLAMA_TOKEN_ATTR_RSTRIP).is_empty(),
            "{file} sets RSTRIP, which no reachable rule can do"
        );
    }
    for file in ["baseline.attrs", "jina.attrs", "modernbert.attrs"] {
        let lstrip = ids_with(file, LLAMA_TOKEN_ATTR_LSTRIP);
        for id in [ID_S, ID_UNK, ID_BOS] {
            assert!(
                !lstrip.contains(&id),
                "{file} sets LSTRIP on id {id}, which no rule names"
            );
        }
    }
}

#[test]
fn a_missing_strip_token_is_reported_not_fatal() {
    let mut md = metadata();
    let types = md
        .get("tokenizer.ggml.token_type")
        .and_then(|v| v.as_vec_u64())
        .expect("types");
    let tokens: Vec<gguf::Value> = md
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_vec_str())
        .expect("tokens")
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != ID_MASK as usize)
        .map(|(_, t)| gguf::Value::Str((*t).to_string()))
        .collect();
    let types: Vec<gguf::Value> = types
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != ID_MASK as usize)
        .map(|(_, t)| gguf::Value::U64(*t))
        .collect();
    md.insert("tokenizer.ggml.tokens".into(), gguf::Value::Array(tokens));
    md.insert(
        "tokenizer.ggml.token_type".into(),
        gguf::Value::Array(types),
    );
    md.insert(
        "tokenizer.ggml.pre".into(),
        gguf::Value::Str("jina-v2-code".into()),
    );
    let tok = Bpe::from_metadata(&md).expect("still a BPE vocabulary");
    assert_eq!(tok.strip_ids(), (Vec::<i32>::new(), Vec::<i32>::new()));
}
