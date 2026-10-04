#!/usr/bin/env python3
"""Does the test suite actually *catch* bugs? Prove it, don't assume it.

A green test suite is only worth something if it would go red on a real defect.
The parity work already made this concrete once: a `cos >= 1 - 1e-6` gate over
llama.cpp's default output passed a GELU coefficient that was wrong in the
fourth significant figure, because llama.cpp's own forward pass was 2e-4 away
from exact and swamped it. That failure was invisible until the gate was
measured against an independent float64 reference.

This script makes the property mechanical instead of anecdotal. Each entry below
is a single, small, plausible defect injected into one engine source file. It
runs the real `cargo test --release` for each one and requires that **every**
mutation turns the suite red. A mutation that survives is either a hole in the
tests or a threshold set too loose, and either way it is a finding.

Every mutation is applied by exact string replacement, so a mutation that fails
to apply is itself reported rather than silently skipped — a typo in a "find"
string must never read as a passing test.

Usage:
    python3 scripts/mutation_check.py            # all mutations
    python3 scripts/mutation_check.py --only gpt2_qkv_split
    python3 scripts/mutation_check.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "engine"
TARGET = ENGINE / "src" / "model.rs"
#: Default source file for a mutation; `Mutation.target` overrides it.
DEFAULT_TARGET = "src/model.rs"


@dataclass(frozen=True)
class Mutation:
    name: str
    find: str
    replace: str
    #: What the defect models — shown in the report so the coverage reads as a
    #: list of failure modes rather than a list of string edits.
    models: str
    #: Architectures the mutation is expected to break. Used only for the
    #: report; the pass/fail criterion is always "cargo test went red".
    note: str = ""
    all_sites: bool = False
    #: Path under engine/ of the file to mutate.
    target: str = DEFAULT_TARGET


MUTATIONS: list[Mutation] = [
    Mutation(
        name="gelu_coef_4th_digit",
        find="0.044715f32 * x * x * x",
        replace="0.04471f32 * x * x * x",
        models="GELU constant wrong in the 4th significant figure (GPT-2, Falcon, Gemma)",
        note="subtle: lands at ~2e-6 rel L2, just above the f32 rounding floor",
    ),
    Mutation(
        name="rmsnorm_eps_dropped",
        find="""                rmsnorm(&x, &w, self.eps as f32)
            };
            let (mut q, mut k, mut v) = if self.arch_variant == ArchVariant::Phi3""",
        replace="""                rmsnorm(&x, &w, 0.0)
            };
            let (mut q, mut k, mut v) = if self.arch_variant == ArchVariant::Phi3""",
        models="RMSNorm epsilon ignored in the shared decoder forward path",
        all_sites=False,
    ),
    Mutation(
        name="attn_scale_no_sqrt",
        find="""            // attention over cached keys
            let scale = 1.0 / (head_dim as f32).sqrt();""",
        replace="""            // attention over cached keys
            let scale = 1.0 / (head_dim as f32);""",
        models="attention scale 1/d instead of 1/sqrt(d)",
    ),
    Mutation(
        name="silu_sign_error",
        find="gate[i] / (1.0 + (-gate[i]).exp())",
        replace="gate[i] / (1.0 + (gate[i]).exp())",
        models="SwiGLU sign error in the gate sigmoid",
        all_sites=True,
    ),
    Mutation(
        name="gpt2_pos_embd_dropped",
        find="""            for i in 0..n_embd {
                x[i] += pos_emb[i];
            }""",
        replace="""            for i in 0..n_embd {
                x[i] += 0.0 * pos_emb[i];
            }""",
        models="GPT-2 learned positional embeddings not added",
    ),
    Mutation(
        name="gpt2_layernorm_to_rms",
        find="""                let b = self.bias_vec(layer, TensorKind::AttnNormBias, n_embd, buf)?;
                Self::layernorm(&x, &w, &b, self.eps as f32)
            } else {
                rmsnorm(&x, &w, self.eps as f32)
            };""",
        replace="""                let b = self.bias_vec(layer, TensorKind::AttnNormBias, n_embd, buf)?;
                let _ = &b;
                rmsnorm(&x, &w, self.eps as f32)
            } else {
                rmsnorm(&x, &w, self.eps as f32)
            };""",
        models="GPT-2 LayerNorm (mean-centring) replaced by RMSNorm",
    ),
    Mutation(
        name="gpt2_qkv_q_slice_to_kv",
        find="""                let k_end = n_embd + n_embd; // n_kv_heads == n_heads for GPT-2 MHA
                (
                    qkv[..q_end].to_vec(),
                    qkv[q_end..k_end].to_vec(),""",
        replace="""                let k_end = n_embd + n_embd; // n_kv_heads == n_heads for GPT-2 MHA
                (
                    qkv[q_end..k_end].to_vec(),
                    qkv[..q_end].to_vec(),""",
        models="GPT-2 fused-QKV Q and K slices swapped",
    ),
    Mutation(
        name="attn_residual_before_output_proj",
        find="""                for i in 0..n_embd {
                    x[i] += wo_out[i] + wo_b[i];
                }""",
        replace="""                for i in 0..n_embd {
                    x[i] = wo_out[i] + wo_b[i];
                }""",
        models="attention residual add replaced by overwrite (residual lost)",
    ),
    Mutation(
        name="weight_tying_disabled",
        find="""        match self.t(TensorKind::Output, 0) {
            Ok(t) => Ok(t),
            Err(_) => self.t(TensorKind::TokenEmbedd, 0),
        }""",
        replace="        self.t(TensorKind::Output, 0)",
        models="output.weight fallback to token_embd removed (ROADMAP 0.4 undone)",
        note="one site: the tying fallback now lives in Model::output_t, which every "
        "architecture's forward pass calls. This used to be four copies of a match "
        "and was lost by a rewrite that took out two of them while the comment "
        "above a third still described it. Every fixture is tied, so this must break "
        "llama, qwen2, gpt2, deepseek2 and gemma3 at once",
        all_sites=False,
    ),
    Mutation(
        name="output_tie_bypassed_at_one_site",
        find="let out_t = self.output_t()?;",
        replace="let out_t = self.t(TensorKind::Output, 0)?;",
        models="one architecture's forward pass reads output.weight directly, "
        "re-introducing a per-architecture drift in the tying rule",
        note="models the failure the helper exists to make unrepresentable: it is "
        "detected as a red suite rather than as a code smell, and the shape it "
        "re-creates is exactly what was undone at 11:00 on 2026-09-30",
        all_sites=True,
    ),
    Mutation(
        name="lstrip_rule_aims_at_the_other_token",
        find='Some(id) => set_on(specials, id, true, false),\n            None => eprintln!("strip rule: no <mask>',
        replace='Some(id) => set_on(specials, id + 1, true, false),\n            None => eprintln!("strip rule: no <mask>',
        models="the jina-v2 LSTRIP rule setting the flag on the wrong id, so the two "
        "reachable rules collide on one token",
        note="jina and modern-bert name different tokens on purpose: a port that "
        "aimed both at the same one would pass either test alone",
        target="src/tokenizer.rs",
    ),
    Mutation(
        name="lstrip_strip_removed_from_the_partition",
        find="""                    let mut left_end = start;
                    if sp.lstrip {
                        while left_end > base && is_llama_space(bytes[left_end - 1]) {
                            left_end -= 1;
                        }
                    }""",
        replace="""                    let left_end = start;""",
        models="tokenizer_st_partition accepting LLAMA_TOKEN_ATTR_LSTRIP and ignoring it",
        note="the flags are derived correctly and then do nothing — invisible to an "
        "id oracle, because the synthetic merge tables cannot encode the "
        "surrounding text either way",
        target="src/tokenizer.rs",
    ),
    Mutation(
        name="lstrip_emits_the_empty_left_reminder",
        find="""                    if left_end > base {
                        out.push(Fragment::Raw(raw[base..left_end].to_string()));
                    }""",
        replace="""                    if left_end >= base {
                        out.push(Fragment::Raw(raw[base..left_end].to_string()));
                    }""",
        models="the post-strip emptiness test dropped, so a run of spaces that is "
        "only before the token survives as an empty raw fragment",
        note="upstream pushes the left reminder only if it is non-empty AFTER the "
        "strip (llama-vocab.cpp:3214); the ids are identical either way",
        target="src/tokenizer.rs",
    ),
    Mutation(
        name="c_locale_space_set_loses_the_vertical_tab",
        find="matches!(b, b' ' | b'\\t' | b'\\n' | 0x0b | 0x0c | b'\\r')",
        replace="b.is_ascii_whitespace()",
        models="llama.cpp's `isspace` replaced by Rust's `is_ascii_whitespace`, which "
        "has five bytes to C's six",
        note="0x0B (vertical tab) is the difference: C's isspace accepts it, Rust's "
        "is_ascii_whitespace does not",
        target="src/tokenizer.rs",
    ),
    Mutation(
        name="phi3_rstrip_branch_implemented_despite_being_unreachable",
        find="""    if contains_any(&pre, &["jina-v2-de", "jina-v2-es", "jina-v2-code"])""",
        replace="""    if contains_any(&name, &["phi-3", "phi3"]) {
        for sp in specials.iter_mut() {
            sp.rstrip = true;
        }
    } else if contains_any(&pre, &["jina-v2-de", "jina-v2-es", "jina-v2-code"])""",
        models="the phi-3 RSTRIP branch ported even though llama.cpp cannot execute "
        "it, so the engine reports a rule upstream has never run",
        note="this is the failure mode of reading llama-vocab.cpp without measuring "
        "it: the loop is right there and looks portable. models/ has no vocabulary "
        "whose ids could be checked against upstream for it",
        target="src/tokenizer.rs",
    ),
    Mutation(
        name="gqa_repeat_wrong_group",
        find="let kv_h = h / kv_group;",
        replace="let kv_h = h % n_kv_heads;",
        models="GQA head->KV-head mapping changed from h/kv_group to h%n_kv_heads",
        note="applied at every site. Identical when kv_group==1, so the MHA "
        "architectures (llama, gpt2) are unaffected by construction — qwen2 is "
        "the only fixture that can catch it, which is exactly the point",
        all_sites=True,
    ),
    # ---- DeepSeek-V2 MLA ----
    # The deepseek2 fixture is the only one that can catch these: nothing else
    # has a latent KV cache, a split nope/rope head, or a MoE block. If the
    # engine's MLA regressed, llama/qwen2/gpt2 would stay green.
    Mutation(
        name="ds2_kq_scale_uses_nope_width",
        find="            self.key_length_mla as f32,",
        replace="            (self.key_length_mla - self.rope_dim) as f32,",
        models="MLA kq_scale divides by the nope width instead of the full head width",
        note="subtle: llama.cpp pre-scales kq_scale by 1/sqrt(key_length_mla), "
        "where key_length_mla counts the rope dims too. DeepSeek-V2-Lite uses "
        "192/128, so the two differ by a factor of 1.22",
    ),
    Mutation(
        name="ds2_nope_rope_split_swapped",
        find="""                for i in 0..n_embd_head_qk_nope {
                    q_nope[h * n_embd_head_qk_nope + i] = q[h_off + i];
                }
                for i in 0..rope_dim {
                    q_pe[h * rope_dim + i] = q[h_off + n_embd_head_qk_nope + i];
                }""",
        replace="""                for i in 0..rope_dim {
                    q_pe[h * rope_dim + i] = q[h_off + i];
                }
                for i in 0..n_embd_head_qk_nope {
                    q_nope[h * n_embd_head_qk_nope + i] = q[h_off + rope_dim + i];
                }""",
        models="MLA per-head nope/rope halves swapped",
        note="only detectable when key_length_mla != 2*rope_dim; the fixture uses "
        "6 vs 4 for exactly that reason",
    ),
    Mutation(
        name="ds2_q_a_norm_dropped",
        find="""                let q_a = rmsnorm(&q_a, &q_norm_w, eps);
                let wq_b_t = self.t(TensorKind::AttnQB, layer)?;""",
        replace="""                let _ = &q_norm_w;
                let wq_b_t = self.t(TensorKind::AttnQB, layer)?;""",
        models="the middle RMSNorm between attn_q_a and attn_q_b is skipped",
        note="this norm has no analogue in the llama path, so no other fixture "
        "can see it disappear",
    ),
    Mutation(
        name="ds2_kv_norm_covers_rope_too",
        find="            let mut kv_cmpr: Vec<f32> = kv_cmpr_pe[..n_lora_kv].to_vec();",
        replace="            let mut kv_cmpr: Vec<f32> = kv_cmpr_pe.to_vec();",
        models="the latent RMSNorm is applied to the rope part of the fused KV projection",
        note="slices the other way: kv_cmpr_pe[..n_lora_kv] is the latent; taking "
        "the whole row and then reading rope_dim of it as k_pe shifts everything",
    ),
    Mutation(
        name="ds2_absorb_head_stride",
        find="                    let rows = wk_b_t.rows(h * rows_per_head + s, n, buf)?;",
        replace="                    let rows = wk_b_t.rows(s * self.n_heads + h, n, buf)?;",
        models="absorbed-K row indexing treats wk_b as (head, kv) instead of (kv, head)",
        note="attn_k_b is {qk_nope, kv_lora, head} in GGUF, so the flat row order is "
        "head-major. Transposing it still 'works' and gives plausible numbers",
    ),
    Mutation(
        name="ds2_k_pe_not_rotated",
        find="            rope(&mut k_pe, pos, self.rope_theta, self.rope_type);",
        replace="            let _ = (&mut k_pe, pos);",
        models="the shared RoPE half of the latent key is not rotated",
        note="Q is still rotated, so this is not obviously broken — but the "
        "latent K cache is what positions the query against",
    ),
    Mutation(
        name="ds2_shared_expert_dropped",
        find="""                for i in 0..n_embd {
                    x[i] += moe_out[i] + sh_out[i];
                }""",
        replace="""                for i in 0..n_embd {
                    x[i] += moe_out[i];
                }
                let _ = &sh_out;""",
        models="the DeepSeek-V2 shared expert is computed but not added to the residual",
    ),
    Mutation(
        name="fused_expert_gate_up_views_swapped",
        find="""                    (t.clone(), t, base + rpe, base, rpe)""",
        replace="""                    (t.clone(), t, base, base + rpe, rpe)""",
        models="ffn_gate_up_exps read with the up rows first and the gate rows "
               "second, instead of llama.cpp's gate-then-up order",
        note="the two orders are not interchangeable by accident: generating the "
              "fused fixture with np.concatenate([up, gate]) instead of "
              "[gate, up] changes llama.cpp's own logits, which is what makes this "
              "a defect rather than a harmless rename. Caught twice over -- by "
              "ref_parity's tolerance gate against the two oracles, and by "
              "the_fused_and_split_expert_forms_agree_bit_for_bit, which is a "
              "bit-for-bit comparison and so has no slack at all",
    ),
    Mutation(
        name="fused_expert_never_read",
        find="""            match self.t(TensorKind::FfnGateUpExps, layer) {""",
        replace="""            match Err::<crate::model::StreamedTensor, _>(
                std::io::Error::from(std::io::ErrorKind::NotFound),
            ) {""",
        models="the fused ffn_gate_up_exps tensor is bound but never consulted, so "
               "a GGUF converted with --fuse-gate-up-exps falls through to a "
               "split-only lookup",
        note="the shape of the gap this tick closed. Before it, the engine had no "
              "FfnGateUpExps kind at all and aborted with `tensor "
              "blk.1.ffn_up_exps.weight not found`; the mutation reinstates the "
              "read-only-the-split behaviour so the fixture keeps loading and the "
              "identity test is what notices the expert weights are wrong",
    ),
    # ---------------------------------------------------------------- gemma3
    #
    # Nine defects in the Gemma-3 path. Six of them are invisible at position 0
    # of any prompt, which is what makes the architecture a good adversary: with a
    # single token in the KV cache the softmax is 1 whatever the logits are, and
    # RoPE is the identity at pos 0, so every one of them passes a
    # "does it load and emit 128 numbers" smoke test and none of them pass an
    # 8-token one.
    Mutation(
        name="g3_embd_scale_dropped",
        find="""        let embd_scale = (n_embd as f32).sqrt();""",
        replace="""        let embd_scale = 1.0f32;""",
        models="Gemma's sqrt(n_embd) embedding scale omitted (gemma3.cpp:93)",
        note="the largest gemma-specific constant, and the one a llama-shaped reading misses first",
    ),
    Mutation(
        name="g3_attn_out_width_uses_n_embd",
        find="""            let mut att_out = vec![0f32; n_heads * head_dim];
            for h in 0..n_heads {
                let kv_h = h / kv_group;
                let mut scores = vec![0f32; attn_range];""",
        replace="""            let mut att_out = vec![0f32; n_embd];
            for h in 0..n_heads {
                let kv_h = h / kv_group;
                let mut scores = vec![0f32; attn_range];""",
        models="attention output treated as embedding_length wide instead of  n_head * key_length (Gemma-3-1B: 1024 vs 1152)",
        note="the tensor shape itself is right; only the destination width is not",
    ),
    Mutation(
        name="g3_attn_scale_applied_twice",
        find="""                    scores[oi] = kr.iter().zip(qh).map(|(a, b)| a * b).sum::<f32>();
                }
                let probs = softmax(&mut scores);""",
        replace="""                    scores[oi] = kr.iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() * f_scale;
                }
                let probs = softmax(&mut scores);""",
        models="f_attention_scale applied to Q and again to the score, squaring it",
        note="invisible at position 0: one key in the cache means softmax == 1",
    ),
    Mutation(
        name="g3_k_norm_reads_q_norm",
        find="""            let kw_t = self.t(TensorKind::AttnKNorm, layer)?;""",
        replace="""            let kw_t = self.t(TensorKind::AttnQKNorm, layer)?;""",
        models="the K-norm weight vector is the Q-norm one (both are length  n_embd_head_k, so a name mix-up is not a shape error)",
    ),
    Mutation(
        name="g3_qk_norm_weight_only_first_head",
        find="""    for (h, chunk) in v.chunks(head_dim).enumerate() {
        let normed = rmsnorm(chunk, w, eps);""",
        replace="""    for (h, chunk) in v.chunks(head_dim).enumerate() {
        let normed = rmsnorm(chunk, if h == 0 { w } else { &w[..0] }, eps);""",
        models="the shared QK-norm weight vector applied to the first head only  instead of to every head",
        note="the pre-fix code sliced `w` per head, which panicked; this is the \
              non-panicking version of the same mistake",
    ),
    Mutation(
        name="g3_swa_layers_use_global_rope_base",
        find="""            let rope_theta_local = if is_swa {
                self.rope_freq_base_swa
            } else {
                self.rope_theta
            };""",
        replace="""            let rope_theta_local = self.rope_theta;""",
        models="SWA layers rotated at the global base (1e6) instead of the SWA  base (10000)",
        note="invisible at position 0: RoPE there is the identity",
    ),
    Mutation(
        name="g3_swa_freq_base_default_zero",
        find="""            .get(&format!("{arch_str}.rope.freq_base_swa"))
            .and_then(|v| v.as_f64())
            .unwrap_or(10_000.0);""",
        replace="""            .get(&format!("{arch_str}.rope.freq_base_swa"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);""",
        models="rope.freq_base_swa defaults to 0 / 'unset' instead of llama.cpp's  10000, which is the value a real converted Gemma-3 GGUF relies on because the converter never writes the key",
    ),
    Mutation(
        name="g3_swa_pattern_zero_means_no_swa",
        find="""    swa_period == 0 || (layer % swa_period) < swa_period - 1
}""",
        replace="""    swa_period > 0 && (layer % swa_period) < swa_period - 1
}""",
        models="a sliding_window_pattern of 0 read as 'no SWA' when llama.cpp reads  it as 'every layer is SWA'",
        note="not visible in the 8-layer fixture (pattern 6); the unit test on \
              gemma3_is_swa is what has to catch this one",
    ),
    Mutation(
        name="g3_attn_post_norm_after_residual",
        find="""            // Post-attention norm (before residual add)
            let post_attn_t = self.t(TensorKind::AttnPostNorm, layer)?;
            let post_attn_w = post_attn_t.rows(0, 1, buf)?;
            let wo_normed = rmsnorm(&wo_out, &post_attn_w, eps);
            for i in 0..n_embd {
                x[i] += wo_normed[i];
            }""",
        replace="""            // Post-attention norm (before residual add)
            let post_attn_t = self.t(TensorKind::AttnPostNorm, layer)?;
            let post_attn_w = post_attn_t.rows(0, 1, buf)?;
            let mut branch = wo_out.clone();
            for i in 0..n_embd {
                branch[i] += x[i];
            }
            let wo_normed = rmsnorm(&branch, &post_attn_w, eps);
            for i in 0..n_embd {
                x[i] += wo_normed[i];
            }""",
        models="post-attention norm moved from the branch output to the residual sum, i.e. norm(x + attn) instead of norm(attn) + x",
    ),
    Mutation(
        name="g3_rope_pairing_norm_instead_of_neox",
        find="""            let rope_theta_local = if is_swa {
                self.rope_freq_base_swa
            } else {
                self.rope_theta
            };
            for h in 0..n_heads {
                let s = h * head_dim;
                rope(
                    &mut q[s..s + head_dim],
                    pos,
                    rope_theta_local,
                    self.rope_type,
                );
            }""",
        replace="""            let rope_theta_local = if is_swa {
                self.rope_freq_base_swa
            } else {
                self.rope_theta
            };
            let rope_kind = RopeType::Norm;
            for h in 0..n_heads {
                let s = h * head_dim;
                rope(
                    &mut q[s..s + head_dim],
                    pos,
                    rope_theta_local,
                    rope_kind,
                );
            }""",
        models="RoPE pairing read as NORM (pairs 2i, 2i+1) when llama.cpp uses  NEOX for gemma3 (pairs offset by n_rot/2)",
        note="invisible at position 0; this is the bug that cost the most time to \
              find in this tick, and it only shows from position 1",
    ),
    Mutation(
        name="moe_norm_w_dropped",
        find="""    if norm_w {
        let mut sum: f32 = sel.iter().sum();""",
        replace="""    if false && norm_w {
        let mut sum: f32 = sel.iter().sum();""",
        models="MoE routing weights left un-normalised: build_moe_ffn divides the "
               "selected weights by their own sum whenever norm_w is set",
        note="the defect this tick actually fixed. Previously only the unit test on "
              "moe_route_weights could catch it, because the one llama-arch MoE file "
              "on disk (Tiny-MoE, 2 of 2 experts) makes the sum exactly 1; the "
              "synthetic-llama-moe fixture (4 experts, 2 used) now catches it too, "
              "and does so through a numeric gate rather than an assertion",
    ),
    Mutation(
        name="llama_norm_w_reads_metadata",
        find="""        "llama",
        "llama-embed",""",
        replace="""        "llama-embed",""",
        models="`llama` dropped from the norm_w ALWAYS table, so the per-arch "
               "constant is replaced by reading {arch}.expert_weights_norm",
        note="the realistic shape of the bug, and the one the synthetic-llama-moe "
              "fixture was built for: that fixture deliberately ships "
              "expert_weights_norm = false, the value llama.cpp ignores because it "
              "passes a literal `true` at llama.cpp:209. Catches on the fixture "
              "alone -- the deepseek2 fixture sets the key to false too, but "
              "deepseek2 genuinely wants false, so it agrees either way",
    ),
    Mutation(
        name="moe_gating_default_sigmoid",
        find="""    // Every other implemented architecture that can route passes its gating
    // function as a literal; `llama` is the representative case.
    GATING_SOFTMAX""",
        replace="""    // Every other implemented architecture that can route passes its gating
    // function as a literal; `llama` is the representative case.
    GATING_SIGMOID""",
        models="absent {arch}.expert_gating_func resolved to SIGMOID for every "
               "architecture instead of SOFTMAX",
        note="only synthetic-llama-moe can see this: gen_deepseek2 writes "
              "expert_gating_func = 1 explicitly, so it never takes the absent-key "
              "path, and a real llama-arch MoE file carries only expert_count and "
              "expert_used_count. Same blind spot as norm_w -- a fixture that states "
              "the answer in metadata cannot test the code that derives it",
    ),
    Mutation(
        name="moe_norm_w_always_on",
        find="""    if ALWAYS.contains(&arch) {
        true
    } else if NEVER.contains(&arch) {""",
        replace="""    if ALWAYS.contains(&arch) || true {
        true
    } else if NEVER.contains(&arch) {""",
        models="norm_w applied to every architecture, including the eight whose "
               "llama.cpp model file passes a literal `false` (qwen2moe, olmoe, ...)",
        note="inflates the routed FFN by sum(probs)/sum(top-k probs); qwen2moe on "
              "disk is 64 experts routed to 8, so it would be a large error",
    ),
    Mutation(
        name="moe_norm_w_from_metadata_only",
        find="""    if ALWAYS.contains(&arch) {
        true
    } else if NEVER.contains(&arch) {""",
        replace="""    if false {
        true
    } else if NEVER.contains(&arch) {""",
        models="norm_w read only from {arch}.expert_weights_norm, ignoring the "
               "per-architecture constant llama.cpp passes at the call site",
        note="the inverse of moe_norm_w_always_on: llama.cpp hardcodes `true` for "
              "llama and never looks at the metadata key at all",
    ),
    Mutation(
        name="moe_weight_scale_ignored",
        find="""    if w_scale != 0.0 && w_scale != 1.0 {
        for v in &mut sel {
            *v *= w_scale;
        }
    }""",
        replace="""    if false && w_scale != 0.0 && w_scale != 1.0 {
        for v in &mut sel {
            *v *= w_scale;
        }
    }""",
        models="expert_weights_scale parsed from metadata and then never applied",
        note="no model on disk sets it, so only a unit test can see this",
    ),
    Mutation(
        name="moe_weight_scale_always_applied",
        find="""    if w_scale != 0.0 && w_scale != 1.0 {
        for v in &mut sel {
            *v *= w_scale;
        }
    }""",
        replace="""    for v in &mut sel {
        *v *= w_scale;
    }""",
        models="expert_weights_scale applied unconditionally, so llama.cpp's 0.0f "
               "default for an absent key deletes the routed FFN instead of "
               "meaning 'unset'",
        note="no model on disk sets it, so only a unit test can see this",
    ),
    Mutation(
        name="moe_softmax_weight_softmaxed_over_all_experts",
        find="""    if gating == GATING_SOFTMAX_WEIGHT {
        sel = softmax(&mut sel);
    }""",
        replace="""    if false && gating == GATING_SOFTMAX_WEIGHT {
        sel = softmax(&mut sel);
    }""",
        models="SOFTMAX_WEIGHT gating not re-softmaxing the selected slice, which "
               "is the only thing that distinguishes it from SOFTMAX",
    ),
    Mutation(
        name="moe_deepseek2_gating_inference_dropped",
        find="""    if variant == ArchVariant::DeepSeek2 || arch == "deepseek2" {
        return if (n_layers == 47 || n_layers == 48) && n_vocab == 154_880 {
            GATING_SIGMOID
        } else {
            GATING_SOFTMAX
        };
    }""",
        replace="""    if variant == ArchVariant::DeepSeek2 || arch == "deepseek2" {
        return GATING_SOFTMAX;
    }""",
        models="deepseek2 absent expert_gating_func: the GLM-4.7 Lite shape "
               "(47/48 layers, 154880 vocab) is routed with SIGMOID upstream",
        note="deepseek2.cpp:23-31",
    ),
    # --- tokenizer.rs: the special-token partition -------------------------
    # These three model the tick that found the partition: an engine that
    # pre-tokenizes correctly but never partitions still produces plausible
    # tokens, and diverged from llama.cpp on every run of two or more spaces
    # for two ticks before anyone looked upstream.
    Mutation(
        name="partition_never_runs",
        find="""        let mut frags = vec![Fragment::Raw(text.to_string())];
        for sp in &self.specials {""",
        replace="""        let mut frags = vec![Fragment::Raw(text.to_string())];
        for sp in self.specials.iter().take(0) {""",
        models="tokenizer_st_partition dropped entirely: USER_DEFINED tokens in "
               "the vocabulary are pre-tokenized like ordinary text",
        note="the state the engine was actually in for two ticks",
        target="src/tokenizer.rs",
    ),
    Mutation(
        name="partition_shortest_first",
        find="specials.sort_by(|a, b| b.text.len().cmp(&a.text.len()).then(a.id.cmp(&b.id)));",
        replace="specials.sort_by(|a, b| a.text.len().cmp(&b.text.len()).then(a.id.cmp(&b.id)));",
        models="cache_special_tokens ordered shortest-first instead of "
               "longest-first, so a 26-space run becomes 13 two-space tokens",
        target="src/tokenizer.rs",
    ),
    Mutation(
        name="partition_ignores_user_defined",
        find="""            if !parse_special && !sp.user_defined {
                continue;
            }""",
        replace="""            if true {
                continue;
            }""",
        models="USER_DEFINED special tokens skipped under parse_special = false, "
               "i.e. the CONTROL/UNKNOWN filter applied to all three types",
        target="src/tokenizer.rs",
    ),
    Mutation(
        name="parse_special_ignored",
        find="""            if !parse_special && !sp.user_defined {
                continue;
            }""",
        replace="""            if !sp.user_defined {
                continue;
            }""",
        models="parse_special dropped from tokenizer_st_partition, so CONTROL "
               "and UNKNOWN tokens are never cut out and `parse_special = true` "
               "is exactly `parse_special = false`",
        note="the state the engine was in before tick 25, when `encode` took two "
             "booleans. Nothing in the suite could see it: every assertion with "
             "the flag off is unaffected, and the corpus had no CONTROL token in "
             "it until this tick added one.",
        target="src/tokenizer.rs",
    ),
    Mutation(
        name="control_promotion_by_text_skipped",
        find="""    let promotions = special_text_promotions(&rewritten, md);""",
        replace="""    let promotions = SpecialPromotions::default();""",
        models="the load-time 'auto-detect special tokens by text' pass dropped, "
               "so a token the GGUF marks NORMAL stays NORMAL even when its "
               "spelling is an end-of-generation alias",
        note="the real bug this tick found: Qwen2.5-3B ships </s> (id 128247) as "
             "token_type = 1 and llama.cpp promotes it to CONTROL anyway, so the "
             "engine had 22 specials where llama.cpp has 23. Invisible with "
             "parse_special = false, which is exactly why 24 ticks of parity "
             "testing did not see it.",
        target="src/tokenizer.rs",
    ),
    Mutation(
        name="empty_token_not_rewritten",
        find="""                if s.is_empty() {
                    format!("[EMPTY_{i}]")
                } else {
                    (*s).to_string()
                }""",
        replace="""                (*s).to_string()""",
        models="a zero-length token kept as zero-length, so it matches at every "
               "position of the partition and text can never be tokenized",
        target="src/tokenizer.rs",
    ),
    Mutation(
        name="eog_fim_markers_not_inserted",
        find='''    for slot in ["fim_pad", "fim_rep", "fim_sep"] {''',
        replace='''    for slot in ["fim_pad", "fim_rep", "fim_sep", "fim_pre"] {''',
        models="the three fill-in-the-middle ids that enter special_eog_ids by "
               "auto-detection stay out of it, so a Qwen2 vocabulary stops on "
               "three tokens where llama.cpp stops on six",
        note="measured, not assumed: with this mutation the engine reports "
             "[128247, 151643, 151645] where llama_vocab_is_eog reports those "
             "three plus 151662/151663/151664. The <|fim_pad|>, <|file_sep|> "
             "and <|fim_repo|> spellings are in the *guarded* alias groups, not "
             "in the EOG list, which is why the alias table alone is not the "
             "answer.",
        target="src/eog.rs",
    ),
    Mutation(
        name="eog_harmony_fix_dropped",
        find="""if ((has_return && has_call && has_end) || (has_call && has_flush && has_end))""",
        replace="""if false && ((has_return && has_call && has_end) || (has_call && has_flush && has_end))""",
        models="o200k_harmony and solar-open keep <|end|> as an end-of-generation "
               "token, so a generation stops at every message boundary instead of "
               "every turn",
        target="src/eog.rs",
    ),
    Mutation(
        name="eog_tool_response_fix_dropped",
        find="""    if has_tool_response
        && let Some(id) = s_id""",
        replace="""    if has_tool_response && has_return
        && let Some(id) = s_id""",
        models="gemma4 and paddleocr keep </s> as an end token and as a CONTROL "
               "special, so the prompt's </s> is lifted out as an id when a chat "
               "template asks for parse_special",
        target="src/eog.rs",
    ),
    Mutation(
        name="eog_forced_attrs_not_applied",
        find="""            for (id, forced) in &eog.forced {""",
        replace="""            for (id, forced) in &[] as &[(i32, crate::eog::Forced)] {""",
        models="the EOG set is right but the attribute assignments that go with it "
               "are dropped, so the ids leave the set without leaving "
               "cache_special_tokens",
        note="the two halves have to be tested separately: an engine that only "
             "erased the ids would pass every assertion about the set and still "
             "tokenize a harmony prompt wrongly.",
        target="src/spm.rs",
    ),
    Mutation(
        name="eog_default_eos_key_ignored",
        find="""    let eos = key_id(md, "eos", n).or_else(|| {""",
        replace="""    let eos = Some(2).filter(|_| true).or_else(|| {""",
        models="special_eos_id read from its per-tokenizer-model default even when "
               "the GGUF declares one, so a model that moved its end token is "
               "stopped at the old id",
        target="src/eog.rs",
    ),
    Mutation(
        name="expert_scale_read_for_every_arch",
        find="""pub fn moe_scale_for(arch: &str, expert_weights_scale: f64) -> f32 {
    if MOE_EXPERT_WEIGHTS_SCALE_READERS.contains(&arch) {
        expert_weights_scale as f32
    } else {
        0.0
    }
}""",
        replace="""pub fn moe_scale_for(arch: &str, expert_weights_scale: f64) -> f32 {
    let _ = arch;
    expert_weights_scale as f32
}""",
        models="`{arch}.expert_weights_scale` applied for every architecture, "
               "including the 115 of 137 whose load_hparams never reads the key",
        note="the tick-26 defect. llama.cpp's `llama` passes "
              "`hparams.expert_weights_scale` to build_moe_ffn but its loader never "
              "calls get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, ...), so the field is 0.0 "
              "'unset' whatever the GGUF says. Caught twice over: ref_parity's "
              "llama-moe-wscale case (rel L2 6.8e-1 vs llama.cpp) and "
              "tests/moe_keys.rs's reader-table comparison",
    ),
    Mutation(
        name="expert_scale_reader_table_drops_llama",
        find="""    "hy_v3",
    "kimi-linear",""",
        replace="""    "hy_v3",
    "kimi-linear",
    "llama",""",
        models="an architecture added to the expert_weights_scale reader table that "
               "does not read the key upstream",
        note="the realistic direction of the rot this tick guarded against: llama.cpp "
              "adds a new model file, it forwards the hparam, and the table starts "
              "claiming a reader that does not exist. Only the oracle comparison in "
              "tests/moe_keys.rs can see this, because the resulting numbers are the "
              "same as the defect above",
    ),
    Mutation(
        name="deepseek2_f64_oracle_drops_norm_w",
        find="""                if norm_w:
                    total = float(np.sum(sel))""",
        replace="""                if False:
                    total = float(np.sum(sel))""",
        models="the float64 arbiter's deepseek2 MoE branch skipping `norm_w`, so "
               "the tight oracle agrees with an engine that also skips it",
        note="guards the *oracle*, not the engine: deepseek2 forwards and reads "
              "`expert_weights_norm`, so a real DeepSeek-V2 GGUF's value matters and "
              "the plain fixture happens to set it false. Only the "
              "deepseek2-normw fixture can see this",
        target="../scripts/arbitrate_logits.py",
    ),
    Mutation(
        name="gguf_bool_is_not_an_integer",
        find="""            Value::Bool(v) => Some(*v as u64),""",
        replace="""            Value::Bool(_) => None,""",
        models="GGUF BOOL metadata read as \"absent\", so `unwrap_or(0)` turns a "
               "true flag into false with no error anywhere",
        note="the tick-26 parse bug. `{arch}.expert_weights_norm` is written with "
              "`add_bool`, i.e. GGUF type BOOL, which is a distinct type from U8. Every "
              "fixture before deepseek2-normw wrote it as false, so the parse bug and "
              "the value agreed by coincidence and nothing could see it. With the flag "
              "set true, llama.cpp renormalises the routed weights and the engine does "
              "not: rel L2 2.5e-1, 25000x the gate",
        target="src/gguf.rs",
    ),
    Mutation(
        name="gqa_kv_head_paired_by_modulo",
        find="let kv_h = h / kv_group;",
        replace="let kv_h = h % n_kv_heads;",
        models="GQA pairing query head `h` with KV head `h % n_kv_heads` instead of "
               "`h / n_rep` — the interleaved pairing rather than llama.cpp's "
               "`kv_head = h / n_rep`",
        note="the exact state of `src/model.rs` for several hours on 2026-09-30. All "
              "seven sites (Gemma-3's expansion and attention, K2-Horizon's attention, "
              "and the generic forward's expansion and attention) were wrong together. "
              "Not a subtle error: rel L2 1.19 against llama.cpp on the llama fixture, "
              "cos 0.35 — uncorrelated, not drifting. Two fixtures with logit parity "
              "cannot see it at all (gpt2 is MHA, deepseek2 is MLA), which is why the "
              "suite can be green with this wrong for every real GQA model. The pairing "
              "coincides with the correct one only for MHA and for a single KV head; "
              "`n_head_kv == 2` is *not* a coincidence. tests/gqa_mapping.rs pins the "
              "coverage property that made this survivable",
        all_sites=True,
    ),
    # The six sites of the mutation above, one at a time. `all_sites` answers
    # "would the suite notice?"; it cannot answer "was every one of the seven
    # sites part of the bug, and is every one of them pinned?", which is the
    # question a *fix* has to answer. Measured by scripts/arb_gqa_site_power.py,
    # one site per run against the real suite, baseline-aware:
    #
    #   line 1834 forward_gemma3 k expansion      rel L2 5.79e-1   detected
    #   line 1846 forward_gemma3 v expansion      rel L2 1.33e+0   detected
    #   line 1881 forward_gemma3 attention read   rel L2 9.68e-1   detected
    #   line 2045 forward_k2_horizon attn read    --              MISSED
    #   line 2356 forward       k expansion       rel L2 2.87e-1   detected
    #   line 2368 forward       v expansion       rel L2 1.03e+0   detected
    #   line 2427 forward       attention read    rel L2 9.94e-1   detected
    #
    # (gate: tests/ref_parity.rs F64_REL_L2 = 1e-5, so 4-5 orders of magnitude.)
    # Each site is load-bearing on its own; none is redundant with the others.
    # The seventh is NOT here, and that is deliberate: k2_horizon has no fixture
    # and no GGUF on disk reaches that forward pass, so a mutation of it is
    # undetectable and adding it would make every run of this harness exit 1 for
    # a hole nobody has closed. `arb_gqa_site_power.py` covers all seven and does
    # exit 1 while the hole is open. ROADMAP 9.15.
    Mutation(
        name="gqa_gemma3_k_expansion_paired_by_modulo",
        find="""                let mut k2 = vec![0f32; qkv_width];
                for h in 0..n_heads {
                    let kv_h = h / kv_group;""",
        replace="""                let mut k2 = vec![0f32; qkv_width];
                for h in 0..n_heads {
                    let kv_h = h % n_kv_heads;""",
        models="Gemma-3's K expansion paired by modulo -- one site of the seven",
        note="rel L2 5.79e-1 against the float64 reference on synthetic-gemma3 "
              "(4 query heads over 2 KV heads), 5.8e4x the gate. The weakest of "
              "the six because the cache write reads only the first head of each "
              "group out of the expansion, so a wrong expansion still produces a "
              "structured answer rather than noise.",
    ),
    Mutation(
        name="gqa_gemma3_v_expansion_paired_by_modulo",
        find="""                let mut v2 = vec![0f32; qkv_width];
                for h in 0..n_heads {
                    let kv_h = h / kv_group;""",
        replace="""                let mut v2 = vec![0f32; qkv_width];
                for h in 0..n_heads {
                    let kv_h = h % n_kv_heads;""",
        models="Gemma-3's V expansion paired by modulo -- one site of the seven",
        note="rel L2 1.33e+0, the largest single-site error measured. K and V are "
              "separate sites with separate weights, so neither one gates for the "
              "other.",
    ),
    Mutation(
        name="gqa_gemma3_attention_read_paired_by_modulo",
        find="""                let kv_h = h / kv_group;
                let mut scores = vec![0f32; attn_range];""",
        replace="""                let kv_h = h % n_kv_heads;
                let mut scores = vec![0f32; attn_range];""",
        models="Gemma-3's attention reading the wrong KV head -- one site of the seven",
        note="rel L2 9.68e-1. Distinct from the two expansion sites: the expansion "
              "fills k_full/v_full and the cache write then samples one head per "
              "group out of it, so this site can be wrong while both expansions are "
              "right. Unique in the file by `attn_range` (SWA window) on the next "
              "line.",
    ),
    Mutation(
        name="gqa_generic_k_expansion_paired_by_modulo",
        find="""                let mut k2 = vec![0f32; n_embd];
                for h in 0..n_heads {
                    let kv_h = h / kv_group;""",
        replace="""                let mut k2 = vec![0f32; n_embd];
                for h in 0..n_heads {
                    let kv_h = h % n_kv_heads;""",
        models="the shared forward pass's K expansion paired by modulo -- one site of the seven",
        note="rel L2 2.87e-1 on synthetic-llama (8 query heads over 2 KV heads), and "
              "it also reddens quant_gap's real-model case. This is the site that "
              "carries the qwen2.5-3b and llama-arch MoE geometry, so it is the one "
              "the tick-30 real-model check leans on.",
    ),
    Mutation(
        name="gqa_generic_v_expansion_paired_by_modulo",
        find="""                let mut v2 = vec![0f32; n_embd];
                for h in 0..n_heads {
                    let kv_h = h / kv_group;""",
        replace="""                let mut v2 = vec![0f32; n_embd];
                for h in 0..n_heads {
                    let kv_h = h % n_kv_heads;""",
        models="the shared forward pass's V expansion paired by modulo -- one site of the seven",
        note="rel L2 1.03e+0 on synthetic-llama.",
    ),
    Mutation(
        name="gqa_generic_attention_read_paired_by_modulo",
        find="""            // attention over cached keys
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut att_out = vec![0f32; n_embd];
            for h in 0..n_heads {
                let kv_h = h / kv_group;""",
        replace="""            // attention over cached keys
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut att_out = vec![0f32; n_embd];
            for h in 0..n_heads {
                let kv_h = h % n_kv_heads;""",
        models="the shared forward pass's attention reading the wrong KV head -- one site of the seven",
        note="rel L2 9.94e-1 on synthetic-llama. The five-line window is needed "
              "because k2_horizon's attention site is textually identical; the "
              "`attention over cached keys` comment is the only thing that tells "
              "them apart, which is exactly the fragility the shared-helper "
              "refactor in STATUS.next_steps would remove.",
    ),
    Mutation(
        name="one_dot_product_left_sequential_f32",
        find="""                    let qh = &q_norm[h * head_dim..(h + 1) * head_dim];
                    scores[p] = ggml_dot_f32(kr, qh) * scale;""",
        replace="""                    let qh = &q_norm[h * head_dim..(h + 1) * head_dim];
                    scores[p] = kr.iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() * scale;""",
        models="one of the 24 dot products in the forward passes reverted to a "
               "sequential f32 sum, i.e. the state the engine was in for every dot "
               "product except the one in matvec() until 2026-09-30",
        note="The whole crate is blind to this: reverting this single site leaves "
              "the other 163 tests green, because the synthetic fixtures are 12 "
              "layers of 768 and the choice of f32 reduction does not move them. On "
              "qwen2.5-3b the same reversion across all 24 sites moved rel L2 from "
              "3.53e-06 to 2.07e-05 (capital) and 4.07e-06 to 1.90e-04 (moon). "
              "Detected by tests/dot_golden.rs::"
              "every_dot_product_in_the_forward_passes_uses_the_reference_reduction, "
              "which scans the source; that is the point of it.",
    ),
    Mutation(
        name="layernorm_mean_back_to_sequential_f32",
        find="""    let mut sum = 0f64;
    for v in x {
        sum += *v as f64;
    }
    sum as f32""",
        replace="""    let mut sum = 0f32;
    for v in x {
        sum += *v;
    }
    sum""",
        models="the LayerNorm mean reverted to a sequential f32 accumulation -- the "
               "state src/model.rs was in until 2026-09-30, where it was wrong on "
               "247 of 256 real activation rows",
        note="This is the 'plain f32' defect that a parity reader would assume was "
              "already covered by the dot-product work. It is not: ggml_vec_sum_f32 "
              "accumulates in double, and the fixture gate cannot see the difference.",
        target="src/ggml_norm.rs",
    ),
    Mutation(
        name="layernorm_mean_narrowed_late",
        find="""pub fn ggml_norm_mean_f32(x: &[f32]) -> f32 {
    ggml_vec_sum_f32(x) / x.len() as f32
}""",
        replace="""pub fn ggml_norm_mean_f32(x: &[f32]) -> f32 {
    let mut sum = 0f64;
    for v in x {
        sum += *v as f64;
    }
    (sum / x.len() as f64) as f32
}""",
        models="the mean narrowed after the division instead of before it",
        note="ops.cpp:3728 is `float mean = sum/ne00` on a sum that ggml_vec_sum_f32 "
              "has already narrowed, so the kernel rounds twice: once on the way out "
              "of the accumulator, once after the division. This mutation rounds "
              "once. Note what it is NOT: widening the narrowed float and dividing in "
              "f64 is harmless, because that is a double rounding of the same f32 "
              "quotient and almost always lands on the same bits -- an earlier draft "
              "of this entry modelled that instead and the suite was right to miss "
              "it, which is the harness doing its job on the harness.",
        target="src/ggml_norm.rs",
    ),
    Mutation(
        name="layernorm_variance_fell_back_to_the_scalar_tail",
        find="    while i + CVAR_EPR <= n {",
        replace="    while i + CVAR_EPR <= n && false {",
        models="ggml_vec_cvar_f32's 8-wide group loop not being taken, i.e. the "
               "variance reduced as a sequential f64 accumulation of the squares",
        note="The nearest miss: this is what 'just accumulate in f64' looks like, and "
              "it agrees with the kernel on 98.6% of output elements -- which is why "
              "a relative-error metric would have called the old code done.",
        target="src/ggml_norm.rs",
    ),
    Mutation(
        name="layernorm_variance_transcribed_from_the_sse2_branch",
        find="        sum += ((t0 + t2) + (t1 + t3)) as f64;",
        replace="        sum += (((sq[0] + sq[2]) + (sq[1] + sq[3]))\n            + ((sq[4] + sq[6]) + (sq[5] + sq[7]))) as f64;",
        models="the horizontal tree transcribed from ggml_vec_cvar_f32's SSE2 branch "
               "(4-wide, vec.cpp:480) instead of the AVX2 one (8-wide) that the "
               "reference build actually compiled",
        note="The portability hazard in src/ggml_norm.rs's header, made executable: "
              "the reference function has six branches and they do not agree, so a "
              "port transcribed on a machine whose build differs from the oracle's "
              "would land here and be silently wrong. Not the same as the constant: "
              "changing CVAR_EPR to 4 does not compile, because the tree indexes "
              "sq[4..8] out of a [f32; 4], and a mutation that breaks the build "
              "demonstrates nothing.",
        target="src/ggml_norm.rs",
    ),
    Mutation(
        name="rope_ladder_extra_rung",
        find="        let (s, c) = (theta.sin(), theta.cos());\n        theta *= theta_scale;",
        replace="        let (s, c) = (theta.sin(), theta.cos());\n        theta *= theta_scale;\n        theta = (theta as f64 * theta_scale as f64) as f32;",
        models="the frequency ladder advancing TWICE per rung. The engine's ladder used "
               "to be carried in f64 while the reference carries it in f32, and the "
               "whole cost of that was the error COMPOUNDING at each of the n_dims/2 "
               "rungs -- 2.11e-4 relative L2 from the reference and 89.6% of bits "
               "differing (results/op_budget.json, on 21168 real activation vectors). "
               "A one-rung-too-far ladder is the same defect from the outside: the same "
               "compounding, reachable while still compiling in f32.",
        note="The first two attempts at this entry are the finding, and both were run. "
             "(1) Revert the powf to f64 and re-widen: REPORTED MISSED, correctly -- "
             "`(theta.powf(-2.0/n as f64) as f32)` is bit-identical to the f32 powf for "
             "every freq_base in the golden, because f32 powf is already correctly "
             "rounded and widening can only lose bits it does not have. A mutation that "
             "changes no behaviour is not a weak test, it is not a test. (2) Revert the "
             "powf AND the ladder variable's type: does not compile, and the harness "
             "reported 'NO BUILD' for a rule that is perfectly well defined -- the same "
             "trap this file has already recorded for CVAR_EPR. What the engine actually "
             "did wrong was never the width of a single powf; it was the WIDTH OF THE "
             "LADDER, so the mutation has to touch the ladder. Detected by 4 tests, "
             "including forward_matches_llama_cpp_logits end to end, so this is not a "
             "golden-only gate.",
        target="src/model.rs",
    ),
    Mutation(
        name="rope_rotation_without_fma",
        find="        x[ic] = a.mul_add(c, -(b * s));",
        replace="        x[ic] = a * c - b * s;",
        models="the rotation as two roundings instead of one. rotate_pairs (ops.cpp:5947) "
               "is compiled into an -O3 libggml-cpu.so, which contracts both products "
               "into an FMA, so the reference does not round the product at all",
        note="Nothing in the reference's SOURCE says the fma is there; it is only in what "
             "the compiler did. On the old all-n_dims=8 golden this was 1 bit in 144 and "
             "the 1e-6 tolerance test passed it -- a port that is 93.75% bit-identical "
             "looks almost right, which is the most expensive kind of wrong. On the "
             "regenerated golden (7 widths, real head dims) it is 1822/15936.",
        target="src/model.rs",
    ),
]


def apply_mutation(src: str, m: Mutation) -> str | None:
    """Return mutated source, or None if the find string is not present.

    When `all_sites` is set every occurrence is replaced — that models a defect
    in a shared helper (the SwiGLU sigmoid, say), which really would be wrong
    everywhere. Otherwise the find string must be unique, so an edit aimed at
    one architecture cannot silently spread to the others.
    """
    n = src.count(m.find)
    if n == 0:
        return None
    if n > 1 and not m.all_sites:
        return None
    return src.replace(m.find, m.replace) if m.all_sites else src.replace(m.find, m.replace, 1)


def _failed_test_names(out: str) -> set[str]:
    """Test names from cargo's per-test result lines.

    `test result: FAILED. ...` also starts with "test " and also contains "FAILED",
    so it has to be excluded explicitly — otherwise the set of "failing tests"
    contains a constant `result:` that looks like a real name and hides the fact
    that no test was named at all.
    """
    return {
        ln.split(" ")[1]
        for ln in out.splitlines()
        if ln.startswith("test ")
        and " FAILED" in ln
        and not ln.startswith("test result:")
    }


def run_tests() -> tuple[bool, str]:
    """Run the whole suite and decide whether *this* run failed.

    Returns (green, detail) where `detail` is the sorted list of failing tests.

    `--no-fail-fast` matters: without it `cargo test` stops after the first failing
    binary, so a suite with several test targets only ever reports the first one and
    the per-mutation detail is misleading.

    The *caller* is responsible for comparing against the pre-mutation baseline.
    A suite that is already red for unrelated reasons would otherwise make every
    mutation look detected, which is the exact failure mode this script exists to
    prevent, one level up.
    """
    proc = subprocess.run(
        ["cargo", "test", "--release", "--no-fail-fast"],
        cwd=ENGINE,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    # A compile error is not a detected bug — the mutation did not produce a
    # running program, so we learned nothing. Distinguish it explicitly.
    if "error[E" in out or "error: could not compile" in out:
        return False, "DID NOT COMPILE"
    failed = _failed_test_names(out)
    detail = ", ".join(sorted(failed)[:5]) if failed else "-"
    return proc.returncode == 0, detail


def failing_tests() -> set[str]:
    proc = subprocess.run(
        ["cargo", "test", "--release", "--no-fail-fast"],
        cwd=ENGINE,
        capture_output=True,
        text=True,
    )
    out = proc.stdout + proc.stderr
    if "error[E" in out or "error: could not compile" in out:
        raise SystemExit(
            "the suite does not compile, so there is no baseline to compare "
            "against — fix that first, otherwise every mutation would score as "
            "detected for free"
        )
    return _failed_test_names(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None, help="run only these mutations")
    ap.add_argument("--list", action="store_true", help="list mutation names and exit")
    args = ap.parse_args()

    if args.list:
        for m in MUTATIONS:
            print(f"{m.name:32s} {m.models}")
        return 0

    chosen = [m for m in MUTATIONS if not args.only or m.name in args.only]
    if not chosen:
        raise SystemExit(f"no mutation matched {args.only}")

    # Baseline first. A suite that is already red for unrelated reasons (another
    # agent mid-edit, a known open bug) would otherwise make every mutation below
    # report "detected" without having done anything. Detection is therefore
    # *new* failures relative to this snapshot, not simply a red exit code.
    baseline = failing_tests()
    if baseline:
        print(f"note: {len(baseline)} test(s) already failing before any mutation:")
        for name in sorted(baseline):
            print(f"  - {name}")
        print("      a mutation counts as detected only if it adds a NEW failure.")
        print()

    results = []
    # Cache the pristine text of every file a mutation touches, so the loop can
    # restore each one without depending on the order the mutations run in.
    originals: dict[Path, str] = {}

    def pristine(path: Path) -> str:
        if path not in originals:
            originals[path] = path.read_text()
        return originals[path]

    try:
        for m in chosen:
            path = ENGINE / m.target
            mutated = apply_mutation(pristine(path), m)
            if mutated is None:
                results.append((m, "NOT APPLIED", "find string missing or ambiguous"))
                continue
            path.write_text(mutated)
            green, detail = run_tests()
            after = {d.strip() for d in detail.split(",") if d.strip() != "-"}
            new = after - baseline
            # A mutation that stops the crate compiling produced no running
            # program, so it demonstrated nothing. `run_tests` says so in its own
            # docstring and returns the sentinel string "DID NOT COMPILE" -- but
            # the sentinel is not in `baseline`, so the old scoring fell through
            # to the `new` branch and credited it as a detection. It scored 9 of
            # 48 mutations "detected" on the run where a concurrent agent's
            # `src/spm.rs` edit broke the build mid-loop. The sentinel has to be
            # checked *before* the diff, not after.
            if detail == "DID NOT COMPILE":
                results.append((m, "NO BUILD", "mutation (or another agent) broke the build"))
            elif not new:
                results.append((m, "MISSED", "suite no worse than baseline"))
            else:
                results.append((m, "detected", "new failure: " + ", ".join(sorted(new)[:4])))
            path.write_text(pristine(path))
    finally:
        for path, text in originals.items():
            path.write_text(text)

    print()
    print(f"{'mutation':32s} {'verdict':10s} detail")
    print("-" * 100)
    width = max(len(m.name) for m in MUTATIONS)
    for m, verdict, detail in results:
        print(f"{m.name:{width}s} {verdict:10s} {detail}")

    print()
    for m, verdict, _ in results:
        print(f"  {m.name:32s} models: {m.models}")

    missed = [m.name for m, v, _ in results if v != "detected"]
    if missed:
        print(f"\n{len(missed)} of {len(results)} mutation(s) not detected: {missed}")
        return 1
    print(f"\nall {len(results)} mutation(s) detected — the suite discriminates")
    return 0


if __name__ == "__main__":
    sys.exit(main())