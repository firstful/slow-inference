//! `ggml_layernorm` must reproduce llama.cpp's kernel bit for bit.
//!
//! `tests/data/norm_golden.bin` is produced by `tools/probe_norm.c
//! --emit-golden`, which links the same `libggml-cpu.so` the oracle uses, runs a
//! real `ggml_norm -> ggml_mul -> ggml_add` graph on the CPU backend, and calls
//! the reference's own exported `ggml_vec_cvar_f32` for the variance. Every
//! entry is one norm the reference actually computed: `x`, `w`, `b`, the two
//! reduction results and `y` all travel with it, so this test cannot
//! accidentally agree with the generator about a PRNG it never sees.
//!
//! Why a bit-exactness table and not a tolerance. The reduction this replaces
//! was `sum::<f32>()` twice, and it was not "close but not close enough" -- it
//! was wrong on 61.9% of output elements of a real 2048-wide activation, by up
//! to 48 f32 ulps, on GPT-2 and Falcon. Every other test in this crate is
//! blind to that: synthetic-gpt2 is the only fixture that reaches LayerNorm at
//! all, it is 12 layers of 64, and both the old and the new implementation are
//! argmax-perfect on it. So the pin has to be the bits.
//!
//! Regenerate with `engine/scripts/gen_norm_golden.sh`.

use std::path::Path;

use slow_inference::ggml_norm::{ggml_layernorm, ggml_norm_mean_f32, ggml_vec_cvar_f32};

const EPS: f32 = 1.0e-5;

struct Case {
    ne0: usize,
    x: Vec<f32>,
    w: Vec<f32>,
    b: Vec<f32>,
    mean: f32,
    var: f32,
    y: Vec<f32>,
}

fn load(path: &Path) -> Vec<Case> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}\nregenerate with engine/scripts/gen_norm_golden.sh",
            path.display()
        )
    });
    assert_eq!(&bytes[0..8], b"SLNRMGLD", "golden file magic mismatch");
    let u32at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    assert_eq!(u32at(8), 1, "golden file version mismatch");
    let declared = u32at(12) as usize;
    let eps = f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    assert_eq!(
        eps.to_bits(),
        EPS.to_bits(),
        "golden was generated at eps={eps}, this test assumes {EPS}"
    );
    let mut out = Vec::with_capacity(declared);
    let mut o = 20usize;
    let read = |o: &mut usize| -> [u8; 4] {
        let s = [bytes[*o], bytes[*o + 1], bytes[*o + 2], bytes[*o + 3]];
        *o += 4;
        s
    };
    let readf = |o: &mut usize| -> f32 { f32::from_le_bytes(read(o)) };
    for _ in 0..declared {
        let ne0 = u32::from_le_bytes(read(&mut o)) as usize;
        let mut vecs = [
            Vec::with_capacity(ne0),
            Vec::with_capacity(ne0),
            Vec::with_capacity(ne0),
        ];
        for v in vecs.iter_mut() {
            for _ in 0..ne0 {
                v.push(readf(&mut o));
            }
        }
        let mean = readf(&mut o);
        let var = readf(&mut o);
        let mut y = Vec::with_capacity(ne0);
        for _ in 0..ne0 {
            y.push(readf(&mut o));
        }
        out.push(Case {
            ne0,
            x: vecs[0].clone(),
            w: vecs[1].clone(),
            b: vecs[2].clone(),
            mean,
            var,
            y,
        });
    }
    assert_eq!(o, bytes.len(), "trailing bytes in golden file");
    out
}

fn golden_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/norm_golden.bin")
}

#[test]
fn ggml_layernorm_is_bit_identical_to_the_reference_kernel() {
    let cases = load(&golden_path());
    assert!(
        cases.len() >= 100,
        "suspiciously few golden cases: {}",
        cases.len()
    );

    let mut bad_y = Vec::new();
    let mut bad_mean = Vec::new();
    let mut bad_var = Vec::new();
    for c in &cases {
        if ggml_norm_mean_f32(&c.x).to_bits() != c.mean.to_bits() {
            bad_mean.push(format!(
                "ne0={} expected 0x{:08x} got 0x{:08x}",
                c.ne0,
                c.mean.to_bits(),
                ggml_norm_mean_f32(&c.x).to_bits()
            ));
        }
        let mut centered = vec![0f32; c.ne0];
        let got = ggml_vec_cvar_f32(&c.x, c.mean, &mut centered);
        if got.to_bits() != c.var.to_bits() {
            bad_var.push(format!(
                "ne0={} expected 0x{:08x} got 0x{:08x}",
                c.ne0,
                c.var.to_bits(),
                got.to_bits()
            ));
        }
        let got = ggml_layernorm(&c.x, &c.w, &c.b, EPS);
        assert_eq!(got.len(), c.ne0, "the port returned the wrong length");
        for (i, (g, want)) in got.iter().zip(&c.y).enumerate() {
            if g.to_bits() != want.to_bits() {
                bad_y.push(format!(
                    "ne0={} i={} expected 0x{:08x} ({want}) got 0x{:08x} ({g})",
                    c.ne0,
                    i,
                    want.to_bits(),
                    g.to_bits()
                ));
                break;
            }
        }
    }
    assert!(
        bad_mean.is_empty(),
        "{} of {} means differ from the kernel:\n  {}",
        bad_mean.len(),
        cases.len(),
        bad_mean
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
    assert!(
        bad_var.is_empty(),
        "{} of {} variances differ from the kernel:\n  {}",
        bad_var.len(),
        cases.len(),
        bad_var
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
    assert!(
        bad_y.is_empty(),
        "{} of {} LayerNorm outputs differ from the kernel:\n  {}",
        bad_y.len(),
        cases.len(),
        bad_y
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

#[test]
fn the_table_covers_every_tail_shape_of_the_eight_wide_loop() {
    // `ggml_vec_cvar_f32` has a scalar tail and an 8-wide group loop. A table
    // of only multiples of 8 could not tell a port that keeps the tree from one
    // that fell back to a sequential f64 sum on lengths under 8, which is the
    // single most likely way to get this function wrong and the one least
    // likely to show up on a real model, since a real n_embd is a multiple of
    // 32 or 64.
    let cases = load(&golden_path());
    let mut by_tail = [0usize; 8];
    let mut rows = 0usize;
    for c in &cases {
        by_tail[c.ne0 % 8] += 1;
        rows += 1;
        assert!(c.ne0 >= 1, "a zero-length case would divide by zero");
    }
    assert_eq!(rows, cases.len());
    for (t, n) in by_tail.iter().enumerate() {
        assert!(
            *n > 0,
            "golden table has no length whose ne0 mod 8 is {t}; the {t}-element tail of \
             the kernel's group loop is untested, so a port that dropped it would pass"
        );
    }
    // And the lengths a real LayerNorm actually reduces: GPT-2 768/1024/1600,
    // Falcon 3072/4544.
    let lens: std::collections::BTreeSet<usize> = cases.iter().map(|c| c.ne0).collect();
    for want in [768usize, 1024, 1600, 3072, 4544] {
        assert!(
            lens.contains(&want),
            "golden table has no n_embd={want}, so no real LayerNorm width is covered"
        );
    }
}

/// The old implementation is not the kernel, and neither is the plausible fix.
///
/// Three claims, in increasing order of how easy they are to get wrong:
///
/// 1. `sum::<f32>()` twice -- what `Model::layernorm` did -- disagrees with the
///    kernel on most rows.
/// 2. Replacing both with a sequential f64 accumulation, which is what the note
///    above the old code said the reference does, is *better* and still wrong:
///    the variance is a tree over groups of 8, not a sequential sum, so it
///    leaves a real fraction of rows a ulp off. Reporting only (2) would make
///    this look like a rounding detail.
/// 3. Porting only the variance and leaving the mean in f32 is the worst of the
///    three, and is the shape a partial port takes.
///
/// This is the test that makes "bit-identical" mean something. Without it an
/// implementation that coincided with the kernel on 130 gaussian vectors would
/// be indistinguishable from leaving the old lines in place.
#[test]
fn the_old_reduction_and_the_obvious_fix_are_both_not_the_kernel() {
    let cases = load(&golden_path());

    let old = |c: &Case| -> Vec<f32> {
        let n = c.ne0;
        let mean = c.x.iter().sum::<f32>() / n as f32;
        let var = c.x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        c.x.iter()
            .zip(&c.w)
            .enumerate()
            .map(|(i, (v, wi))| (v - mean) * inv * wi + c.b[i])
            .collect()
    };
    let f64_both = |c: &Case| -> Vec<f32> {
        let n = c.ne0;
        let sum: f64 = c.x.iter().map(|v| *v as f64).sum();
        let mean = (sum as f32) / n as f32;
        let vsum: f64 = c.x.iter().map(|v| ((*v - mean) as f64).powi(2)).sum();
        let var = (vsum / n as f64) as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        c.x.iter()
            .zip(&c.w)
            .enumerate()
            .map(|(i, (v, wi))| (v - mean) * inv * wi + c.b[i])
            .collect()
    };
    // The partial port: the kernel's variance, the engine's mean.
    let var_only = |c: &Case| -> Vec<f32> {
        let n = c.ne0;
        let mean = c.x.iter().sum::<f32>() / n as f32;
        let mut centered = vec![0f32; n];
        let var = ggml_vec_cvar_f32(&c.x, mean, &mut centered);
        let inv = 1.0 / (var + EPS).sqrt();
        c.x.iter()
            .zip(&c.w)
            .enumerate()
            .map(|(i, (v, wi))| (v - mean) * inv * wi + c.b[i])
            .collect()
    };

    let mut disagree = [0usize; 3];
    let mut elems = [0usize; 3];
    for c in &cases {
        let got = ggml_layernorm(&c.x, &c.w, &c.b, EPS);
        for (k, cand) in [old(c), f64_both(c), var_only(c)].iter().enumerate() {
            for i in 0..c.ne0 {
                elems[k] += 1;
                if cand[i].to_bits() != got[i].to_bits() {
                    disagree[k] += 1;
                }
            }
        }
    }
    let names = [
        "sequential f32 (the old Model::layernorm)",
        "sequential f64 for both (what the old comment predicted)",
        "the kernel's variance with the engine's mean (a partial port)",
    ];
    for k in 0..3 {
        eprintln!(
            "[norm_golden] {}: {}/{} elements differ from the kernel",
            names[k], disagree[k], elems[k]
        );
    }
    let frac = |k: usize| disagree[k] as f64 / elems[k] as f64;

    // The old code was not close: it is wrong on most elements.
    assert!(
        frac(0) > 0.5,
        "the old sequential-f32 layernorm disagrees with the kernel on only {:.1}% of \
         elements, so the table is not exercising what it claims to",
        100.0 * frac(0)
    );

    // f64-for-both is the *near miss*, and it is the reason this module is not
    // four lines. It agrees with the kernel on ~98.6% of elements -- closer
    // than any of the other candidates by an order of magnitude -- and is still
    // not the kernel, because `ggml_vec_cvar_f32` reduces groups of 8 through a
    // horizontal tree instead of accumulating one square at a time. A relative
    // error metric would have called this done.
    assert!(
        disagree[1] > 0,
        "f64-for-both is bit-identical to the kernel on this table, which would mean the \
         kernel's variance really is a sequential f64 sum and src/ggml_norm.rs's tree is \
         decoration"
    );
    assert!(
        frac(1) < 0.05,
        "f64-for-both disagrees on {:.1}% of elements, which is not the near miss the \
         probe measured (1.4%); either the golden moved or the reference's variance rule \
         changed",
        100.0 * frac(1)
    );

    // And the partial port -- the kernel's variance with the engine's mean --
    // is markedly worse than the near miss, which is why "port the variance and
    // leave the mean" is not a shortcut past this.
    assert!(
        frac(2) > 4.0 * frac(1),
        "porting only the variance ({:.1}%) is not much worse than porting neither \
         ({:.1}%); the mean is not a free parameter",
        100.0 * frac(2),
        100.0 * frac(1)
    );
}

/// The pin has to be about the reference, not about a file nobody regenerated.
#[test]
fn the_golden_is_the_real_kernel_and_not_a_reimplementation() {
    // probe_norm links libggml and runs a real ggml graph, and it calls
    // `ggml_vec_cvar_f32` out of libggml-cpu.so rather than reimplementing it.
    // If the build script ever stops passing -lggml, the table would still exist
    // and the other two tests would still pass -- against a file nobody
    // regenerated.
    let sh = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/gen_norm_golden.sh");
    let text = std::fs::read_to_string(&sh)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", sh.display()));
    for needle in [
        "-lggml",
        "probe_norm.c",
        "--emit-golden",
        "-ffp-contract=off",
        "ggml_vec_cvar_f32",
    ] {
        assert!(
            text.contains(needle),
            "gen_norm_golden.sh no longer mentions {needle}"
        );
    }
    // And the function the port is pinned to has to still be where it was.
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/probe_norm.c");
    let probe = std::fs::read_to_string(&src)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", src.display()));
    assert!(
        probe.contains("extern double ggml_vec_cvar_f32"),
        "probe_norm.c no longer declares the reference's own ggml_vec_cvar_f32, so the \
         golden's variance column may have become a re-implementation"
    );
    // A missing llama.cpp checkout is not this test's business -- the golden file
    // is the evidence and it is checked above. This is a warning that the port
    // may be pinned to a function that has moved.
    let home = std::env::var("HOME").unwrap_or_default();
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg("grep -n 'ggml_vec_cvar_f32' \"$LLAMA_CPP/ggml/src/ggml-cpu/vec.cpp\" | head -1")
        .env(
            "LLAMA_CPP",
            std::env::var("LLAMA_CPP").unwrap_or_else(|_| format!("{home}/llama.cpp")),
        )
        .output();
    if let Ok(o) = out
        && o.status.success()
    {
        let s = String::from_utf8_lossy(&o.stdout);
        assert!(
            s.contains("ggml_vec_cvar_f32"),
            "the reference vec.cpp no longer defines ggml_vec_cvar_f32; the port in \
             src/ggml_norm.rs is pinned to a function that has moved"
        );
    }
}

/// The engine must actually call the port, not a local copy of it.
///
/// `Model::layernorm` used to hold the two reductions inline, and a copy is how
/// the previous twenty-three-site sweep lost its coverage: the fix existed and
/// twenty-three of twenty-four call sites were untouched. This is the same pin
/// for the one function, and it is deliberately a source scan -- a runtime test
/// cannot tell which of two correctly-implemented reductions the engine chose,
/// because the only fixture that reaches LayerNorm is 12 layers of 64.
#[test]
fn model_layernorm_delegates_to_the_reference_reduction() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/model.rs");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    // No sequential f32 accumulation of a vector anywhere in the file any more.
    let offenders: Vec<String> = src
        .lines()
        .enumerate()
        .filter(|(_, l)| {
            (l.contains("sum::<f32>()") || l.contains("sum::<f64>()"))
                && l.contains(".iter()")
                && !l.contains("ggml_dot")
        })
        .map(|(i, l)| format!("  src/model.rs:{}: {}", i + 1, l.trim()))
        .collect();
    assert!(
        offenders.is_empty(),
        "{} sequential accumulation(s) remain in src/model.rs; the f32 reductions were \
         all replaced by ports in src/ggml_dot.rs and src/ggml_norm.rs:\n{}",
        offenders.len(),
        offenders.join("\n")
    );

    // And the three call sites of Model::layernorm go through the module rather
    // than re-deriving the mean or the variance inline.
    let fn_body = src
        .split("pub fn layernorm(")
        .nth(1)
        .and_then(|rest| rest.split("/// GELU activation").next())
        .unwrap_or_else(|| panic!("cannot find Model::layernorm in {}", path.display()));
    assert!(
        fn_body.contains("ggml_layernorm("),
        "Model::layernorm no longer calls ggml_layernorm; body was:\n{fn_body}"
    );
    assert!(
        !fn_body.contains("sum::<"),
        "Model::layernorm computes a reduction inline again; body was:\n{fn_body}"
    );

    let calls = src.matches("Self::layernorm(").count();
    assert_eq!(
        calls, 3,
        "expected 3 call sites of Model::layernorm (GPT-2 attn_norm, GPT-2 ffn_norm, \
         GPT-2/Falcon output_norm), found {calls}"
    );
}
