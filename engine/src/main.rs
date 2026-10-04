use slow_inference::gguf;
use slow_inference::model;
use slow_inference::rss;
use slow_inference::spm;
use slow_inference::tokenizer;

use std::collections::HashMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::new();
    let mut prompt = String::from("The capital of France is");
    let mut n_tokens = 8usize;
    let mut raw_ids: Option<Vec<usize>> = None;
    let mut no_special = false;
    let mut parse_special = false;
    let mut stop_on_eog = false;
    let mut json_output = false;
    let mut test_tensors = false;
    let mut dtype_report = false;
    let mut use_vulkan = false;
    let mut logits_out: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = args[i].clone();
            }
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            "--tokens" => {
                i += 1;
                n_tokens = args[i].parse().unwrap();
            }
            "--prompt-tokens" => {
                i += 1;
                raw_ids = Some(
                    args[i]
                        .split(',')
                        .map(|s| s.trim().parse().unwrap())
                        .collect(),
                );
            }
            "--json" => {
                json_output = true;
            }
            "--no-special" => {
                // Do not prepend BOS even when the GGUF asks for it. Matches
                // `llama_tokenize(..., add_special = false, ...)`.
                no_special = true;
            }
            "--parse-special" => {
                // The other half of the same call: `llama_tokenize(...,
                // parse_special = true)` cuts the prompt on the vocabulary's
                // CONTROL and UNKNOWN tokens and emits them as their own ids
                // instead of spelling them out. `llama-server`'s
                // `tokenize_mixed(vocab, prompt, true, true)` sets it, so a
                // prompt pasted through a chat template needs it; a bare
                // `--prompt` usually does not, and the default stays off.
                parse_special = true;
            }
            "--stop-on-eog" => {
                // llama-server's sampler always stops on `llama_vocab_is_eog`,
                // and this is the same predicate: the engine's own derivation of
                // it, from token text rather than from a key, in `src/eog.rs`.
                // Off by default for the same reason `--parse-special` is: this
                // binary's `--tokens N` is a debugging rollout of exactly N
                // argmax steps, and truncating it hides the logits of the steps
                // after the stop, which is the thing being debugged.
                stop_on_eog = true;
            }
            "--test-tensors" => {
                test_tensors = true;
            }
            "--list-architectures" => {
                let supported = model::supported_architectures();
                let unimplemented = model::unimplemented_architectures();
                eprintln!("Supported GGUF architectures ({}):", supported.len());
                for arch in &supported {
                    eprintln!("  {arch}");
                }
                eprintln!(
                    "\nRecognized but not implemented ({}): loads through the nearest\n\
                     implemented layout and will be missing tensors.",
                    unimplemented.len()
                );
                for (arch, why) in &unimplemented {
                    eprintln!("  {arch:<14} {why}");
                }
                return;
            }
            "--dtype" => {
                dtype_report = true;
            }
            "--vulkan" => {
                use_vulkan = true;
            }
            "--dump-logits" => {
                i += 1;
                logits_out = Some(args[i].clone());
            }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }
    if model_path.is_empty() {
        eprintln!(
            "usage: slow-inference --model <gguf> [--prompt \"...\"] [--tokens N] [--json] [--no-special]\n           [--parse-special] [--stop-on-eog] [--list-architectures] [--dtype]\n           [--test-tensors] [--dump-logits <file>]"
        );
        std::process::exit(1);
    }

    let file_size = std::fs::metadata(&model_path).unwrap().len();
    println!("model: {model_path} ({:.2} GB)", file_size as f64 / 1e9);

    let m = model::Model::open(&model_path).unwrap();

    // Vulkan backend (optional — requires --vulkan flag and compatible hardware)
    if use_vulkan {
        match slow_inference::vk::VulkanBackend::new() {
            Ok(mut vk_backend) => {
                slow_inference::vk::load_all_pipelines(&mut vk_backend);
                *m.vk.borrow_mut() = Some(vk_backend);
                eprintln!("Vulkan backend initialized");
            }
            Err(e) => eprintln!("Vulkan backend unavailable: {e}"),
        }
    }

    if dtype_report {
        m.report_dtypes();
        return;
    }
    if test_tensors {
        m.test_tensors().unwrap();
        return;
    }
    println!(
        "arch={} layers={} embd={} heads={} kv={} ff={} experts={}x{} qk_norm={} bias={}",
        m.arch,
        m.n_layers,
        m.n_embd,
        m.n_heads,
        m.n_kv_heads,
        m.n_ff,
        m.n_experts,
        m.n_experts_used,
        m.qk_norm,
        m.has_bias
    );

    // tokenizer: BPE via src/tokenizer.rs for `tokenizer.ggml.model = "gpt2"`,
    // SentencePiece via src/spm.rs for `model = "llama"`. Both are ported from
    // llama.cpp rather than paraphrased and both are proven against it by
    // tests/tokenizer_parity.rs.
    let bpe = tokenizer::Bpe::from_metadata(&m.gf.metadata).ok();
    let spm = match &bpe {
        Some(_) => None,
        None => spm::Spm::from_metadata(&m.gf.metadata).ok(),
    };
    // The EOG count is in the banner because the set is derived from token text
    // and nothing in the GGUF records it: it is the one part of the vocabulary
    // that cannot be read off the file, so it is worth being able to see at a
    // glance whether the derivation found anything. `--stop-on-eog` then uses it.
    if let Some(bpe) = &bpe {
        println!(
            "tokenizer: BPE pre={:?} vocab={} merges={} eog={}",
            bpe.pre_type(),
            bpe.n_vocab(),
            bpe.n_merges(),
            bpe.eog_ids().len()
        );
    }
    if let Some(spm) = &spm {
        println!(
            "tokenizer: SPM vocab={} special={} add_space_prefix={} eog={}",
            spm.n_vocab(),
            spm.n_special(),
            spm.add_space_prefix(),
            spm.eog_ids().len()
        );
    }
    // `add_special` follows llama.cpp's own default for a prompt (the server
    // path's `tokenize_mixed(vocab, prompt, true, true)`), which prepends BOS
    // when the GGUF says `add_bos_token`. Qwen2.5 sets that key to false so its
    // prompt ids are unaffected; Tiny-Moe sets it true, and without the BOS the
    // engine and llama.cpp disagree on the very first forward pass.
    //
    // `parse_special` is the *other* half of that same call and defaults to
    // false here even though the server sets it true, because a bare `--prompt`
    // is not a chat template. `--parse-special` turns it on; tests/tokenizer_parity.rs
    // checks all four combinations against llama.cpp for three vocabularies.
    let add_special = !no_special;
    let prompt_ids = match raw_ids {
        Some(ids) => ids,
        None => match (&bpe, &spm) {
            (Some(bpe), _) => bpe
                .encode(&prompt, add_special, parse_special)
                .iter()
                .map(|&v| v as usize)
                .collect(),
            (None, Some(spm)) => spm
                .encode(&prompt, add_special, parse_special)
                .iter()
                .map(|&v| v as usize)
                .collect(),
            (None, None) => {
                eprintln!(
                    "no supported tokenizer in this GGUF: tokenizer.ggml.model is {:?}",
                    m.gf.metadata
                        .get("tokenizer.ggml.model")
                        .and_then(|v| v.as_str())
                );
                std::process::exit(1);
            }
        },
    };
    if prompt_ids.is_empty() {
        eprintln!("the prompt produced zero tokens; nothing to run");
        std::process::exit(1);
    }

    slow_inference::opdump::open();
    slow_inference::trace::open();
    // --dump-logits: emit per-position logits in the same binary format as
    // tools/gen_ref_logits.c, so the two can be diffed directly. This is the
    // parity oracle for architectures we have no real GGUF for: point both
    // sides at the same tiny synthetic fixture.
    if let Some(path) = &logits_out {
        let rows = m
            .prefill_logits(&prompt_ids)
            .expect("prefill for --dump-logits");
        dump_logits(path, &prompt_ids, m.vocab_size, &rows);
        slow_inference::opdump::close();
        // flush the per-layer residual stream so it is complete on disk even
        // though this path returns early (the trace is what makes --dump-logits
        // attributable to a layer rather than only to the logits)
        slow_inference::trace::close();
        println!(
            "wrote {} logits rows x {} vocab to {path}",
            rows.len(),
            m.vocab_size
        );
        return;
    }

    println!("prompt ids: {prompt_ids:?}");

    let mut parity_logits = None;
    if json_output && !prompt_ids.is_empty() {
        let mut k_cache: Vec<Vec<f32>>;
        let mut v_cache: Vec<Vec<f32>>;
        if m.arch_variant == model::ArchVariant::DeepSeek2 {
            // MLA cache: each layer stores [kv_cmpr | k_pe] per position
            let mla_stride = m.n_lora_kv + m.rope_dim;
            k_cache = vec![vec![0.0; mla_stride]; m.n_layers];
            v_cache = vec![vec![]; m.n_layers];
        } else {
            let stride = m.n_kv_heads * m.head_dim;
            k_cache = vec![vec![0.0; stride]; m.n_layers];
            v_cache = vec![vec![0.0; stride]; m.n_layers];
        }
        let mut buf = Vec::with_capacity(262_144);
        parity_logits = Some(
            m.forward(prompt_ids[0], 0, &mut k_cache, &mut v_cache, &mut buf)
                .expect("forward for parity logits"),
        );
    }

    let (seq, dt, top5) = m.generate(&prompt_ids, n_tokens).unwrap();

    let new_ids: Vec<i32> = seq[prompt_ids.len()..].iter().map(|&v| v as i32).collect();
    // Truncate at the first end-of-generation token, *including* it: llama.cpp's
    // sampler returns the token and then reports the stop, so the stop token is in
    // the output. Cutting the rollout short here rather than inside `generate`
    // leaves the forward pass untouched: greedy argmax depends only on the
    // tokens before it, so the ids up to the stop are the same either way.
    let eog: Option<&slow_inference::eog::Eog> = match (&bpe, &spm) {
        (Some(b), _) => Some(b.eog()),
        (None, Some(s)) => Some(s.eog()),
        (None, None) => None,
    };
    let kept = match (stop_on_eog, eog) {
        (true, Some(eog)) => slow_inference::eog::stop_at_eog(&new_ids, eog)
            .map(|i| i + 1)
            .unwrap_or(new_ids.len()),
        _ => new_ids.len(),
    };
    if kept < new_ids.len() {
        println!(
            "stopped after {kept} of {} tokens: id {} ends the generation",
            new_ids.len(),
            new_ids[kept - 1]
        );
    }
    let out = match &spm {
        Some(spm) => spm.detokenize(&new_ids[..kept], true),
        None => {
            let table = byte_unicode();
            let ids: Vec<usize> = new_ids[..kept].iter().map(|&v| v as usize).collect();
            decode_ids(&ids, &load_vocab(&m.gf.metadata), &table)
        }
    };
    println!("output tokens: {:?}", &new_ids[..kept]);
    println!("output text: {out:?}");
    if !top5.is_empty() {
        println!("top5: {top5:?}");
    }
    println!(
        "time: {dt:.1}s for {n_tokens} tokens ({:.1} s/token)",
        dt / n_tokens as f64
    );

    let peak = rss::peak_rss_kib();
    let cur = rss::current_rss_kib();
    println!(
        "peak RSS: {} = {:.2}% of model file",
        rss::fmt(peak),
        100.0 * (peak * 1024) as f64 / file_size as f64
    );
    println!("current RSS: {}", rss::fmt(cur));
    if json_output {
        let mut payload = serde_json::Map::new();
        payload.insert(
            "fixture".to_string(),
            serde_json::Value::String(model_path.clone()),
        );
        payload.insert(
            "prompt_ids".to_string(),
            serde_json::Value::Array(
                prompt_ids
                    .iter()
                    .map(|&v| serde_json::Value::Number(serde_json::Number::from(v)))
                    .collect(),
            ),
        );
        payload.insert(
            "gen_seq".to_string(),
            serde_json::Value::Array(
                seq.iter()
                    .map(|&v| serde_json::Value::Number(serde_json::Number::from(v)))
                    .collect(),
            ),
        );
        payload.insert(
            "gen_top5".to_string(),
            serde_json::Value::Array(
                top5.iter()
                    .map(|(id, v)| {
                        serde_json::Value::Array(vec![
                            serde_json::Value::Number(serde_json::Number::from(*id)),
                            serde_json::Number::from_f64(*v as f64)
                                .map_or(serde_json::Value::Null, serde_json::Value::Number),
                        ])
                    })
                    .collect(),
            ),
        );
        if let Some(ref logits) = parity_logits {
            payload.insert(
                "logits_token0_pos0".to_string(),
                serde_json::Value::Array(
                    logits
                        .iter()
                        .map(|&v| {
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(v as f64).unwrap(),
                            )
                        })
                        .collect(),
                ),
            );
        }
        println!("ENGINEJSON {}", serde_json::to_string(&payload).unwrap());
    }
}

/// Run the prompt through the forward pass, returning one logits row per
/// consumed token (row `i` = logits after consuming `tokens[..=i]`).
///
/// This mirrors exactly what tools/gen_ref_logits.c asks llama.cpp to produce:
/// every position outputs logits, not just the final one.
/// Serialize logits in the SLDREFLG v1 format understood by
/// engine/tests/ref_parity.rs and produced by tools/gen_ref_logits.c.
fn dump_logits(path: &str, tokens: &[usize], n_vocab: usize, rows: &[Vec<f32>]) {
    let mut out: Vec<u8> = Vec::with_capacity(32 + rows.len() * n_vocab * 4);
    out.extend_from_slice(b"SLDREFLG");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for &t in tokens {
        out.extend_from_slice(&(t as i32).to_le_bytes());
    }
    out.extend_from_slice(&(n_vocab as u32).to_le_bytes());
    for row in rows {
        // The engine's final projection can be wider than the reference vocab
        // (weight-tying / padded output rows); truncate to the agreed width.
        let chunk: Vec<[u8; 4]> = row.iter().take(n_vocab).map(|v| v.to_le_bytes()).collect();
        for c in chunk {
            out.extend_from_slice(&c);
        }
    }
    std::fs::write(path, &out).expect("write logits dump");
}

/// (token string, id) pairs — ids are positions in tokenizer.ggml.tokens
type Vocab = Vec<(String, usize)>;

fn load_vocab(md: &HashMap<String, gguf::Value>) -> Vocab {
    let tokens = md
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_vec_str())
        .expect("no tokenizer.ggml.tokens");
    tokens
        .into_iter()
        .enumerate()
        .map(|(i, s)| (s.to_string(), i))
        .collect()
}

/// GPT-2 byte-level unicode mapping: each byte 0..256 -> a printable char.
/// (The classic bytes_to_unicode table, computed.)
fn byte_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    let printable =
        |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    for b in 0..256u32 {
        if printable(b) {
            table[b as usize] = char::from_u32(b).unwrap();
        }
    }
    for b in 0..256u32 {
        if !printable(b) {
            table[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

fn decode_ids(ids: &[usize], vocab: &Vocab, table: &[char; 256]) -> String {
    let rev: HashMap<char, u8> = table
        .iter()
        .enumerate()
        .map(|(b, c)| (*c, b as u8))
        .collect();
    let mut bytes: Vec<u8> = Vec::new();
    for &id in ids {
        if let Some((tok, _)) = vocab.iter().find(|&&(_, v)| v == id) {
            // Each token is either byte-unicode encoded (all chars in rev) or
            // a native Unicode string (contains chars NOT in rev, e.g. CJK).
            // Map byte-encoded chars back; pass native Unicode through as UTF-8.
            if tok.chars().all(|c| rev.contains_key(&c)) {
                // Pure byte-encoded: reverse map to bytes
                for c in tok.chars() {
                    bytes.push(rev[&c]);
                }
            } else {
                // Contains native Unicode: pass through as UTF-8
                bytes.extend(tok.as_bytes());
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
