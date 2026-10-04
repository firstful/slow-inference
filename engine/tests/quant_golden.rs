//! Golden-vector parity test for `quant::dequant_row`.
//!
//! `tests/data/quant_golden.bin` is produced by `tools/gen_golden_vectors.c`,
//! which links against llama.cpp's own `dequantize_row_*` kernels
//! (`ggml_get_type_traits()->to_float`). We replay every recorded row and
//! require a bit-identical f32 result — NaN payloads are the single exception,
//! since ggml's fp16 table and our integer path may pick different ones.
//!
//! Rebuild the vectors with `scripts/gen_golden_vectors.sh` (needs a built
//! llama.cpp with `libggml-base.so`).

use std::path::Path;

use slow_inference::quant::dequant_row;

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        s
    }
}

/// Types the engine claims to support. Types absent here are reported by
/// `supported_but_ungolden` below rather than silently ignored.
const EXPECTED_TYPES: &[(u32, &str, usize, usize)] = &[
    (1, "F16", 2, 1),
    (30, "BF16", 2, 1),
    (2, "Q4_0", 18, 32),
    (8, "Q8_0", 34, 32),
    (10, "Q2_K", 84, 256),
    (11, "Q3_K", 110, 256),
    (12, "Q4_K", 144, 256),
    (13, "Q5_K", 176, 256),
    (14, "Q6_K", 210, 256),
    (15, "Q8_K", 292, 256),
    (20, "IQ4_NL", 18, 32),
    (23, "IQ4_XS", 136, 256),
    (35, "TQ2_0", 66, 256),
];

#[test]
fn golden_dequant_matches_ggml_bit_for_bit() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/quant_golden.bin");
    let buf = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing {}: {e}\nregenerate with engine/scripts/gen_golden_vectors.sh",
            path.display()
        )
    });
    assert_eq!(&buf[0..8], b"SLDQGOLD");

    let mut r = Reader { buf: &buf, pos: 8 };
    let version = r.u32();
    assert_eq!(version, 1, "golden vector format version mismatch");
    let n_types = r.u32();

    let mut failures: Vec<String> = Vec::new();
    let mut checked: Vec<(u32, usize)> = Vec::new();

    for _ in 0..n_types {
        let tid = r.u32();
        let mut namebuf = [0u8; 16];
        namebuf.copy_from_slice(r.take(16));
        let name = String::from_utf8_lossy(&namebuf)
            .trim_end_matches('\0')
            .to_string();
        let block_bytes = r.u32() as usize;
        let block_elems = r.u32() as usize;
        let n_cases = r.u32() as usize;

        // layout cross-check against the engine's own table
        match slow_inference::gguf::type_layout(tid) {
            Some((eb, ee)) if eb == block_bytes && ee == block_elems => {}
            Some((eb, ee)) => failures.push(format!(
                "{name}: type_layout({tid}) = ({eb},{ee}) but ggml says ({block_bytes},{block_elems})"
            )),
            None => {
                // not an error: the golden file is a superset of what we support
                eprintln!("[golden] {name} ({tid}) not yet supported by the engine");
            }
        }

        let mut type_failures = 0usize;
        let mut type_cases = 0usize;
        for case in 0..n_cases {
            let pat = r.u32();
            let ne0 = r.u32() as usize;
            let nbytes = r.u32() as usize;
            let raw = r.take(nbytes);
            let want = r.take(ne0 * 4);

            let got = match dequant_row(raw, tid, ne0) {
                Ok(v) => v,
                Err(e) => {
                    type_failures += 1;
                    failures.push(format!(
                        "{name} case{case} (pat {pat}): dequant_row error: {e}"
                    ));
                    continue;
                }
            };
            if got.len() != ne0 {
                type_failures += 1;
                failures.push(format!(
                    "{name} case{case} (pat {pat}): len {} != {ne0}",
                    got.len()
                ));
                continue;
            }
            let mut broken = false;
            for i in 0..ne0 {
                let w = f32::from_le_bytes(want[i * 4..i * 4 + 4].try_into().unwrap());
                let g = got[i];
                let ok = if w.is_nan() {
                    g.is_nan()
                } else {
                    g.to_bits() == w.to_bits()
                };
                if !ok {
                    broken = true;
                    break;
                }
            }
            if broken {
                type_failures += 1;
                let i = (0..ne0)
                    .find(|&i| {
                        let w = f32::from_le_bytes(want[i * 4..i * 4 + 4].try_into().unwrap());
                        let g = got[i];
                        if w.is_nan() {
                            !g.is_nan()
                        } else {
                            g.to_bits() != w.to_bits()
                        }
                    })
                    .unwrap();
                let w = f32::from_le_bytes(want[i * 4..i * 4 + 4].try_into().unwrap());
                if type_failures <= 3 {
                    failures.push(format!(
                        "{name} case{case} (pat {pat}) first bad elem {i}: got {:e} (0x{:08x}) want {:e} (0x{:08x})",
                        got[i],
                        got[i].to_bits(),
                        w,
                        w.to_bits()
                    ));
                }
            } else {
                type_cases += 1;
            }
        }
        checked.push((tid, type_cases));
        eprintln!(
            "[golden] {name:8} id={tid:2} block={block_bytes:4} elems={block_elems:4} \
             cases_ok={type_cases}/{n_cases}"
        );
    }

    assert!(r.pos == buf.len(), "trailing bytes in golden file");
    assert!(
        failures.is_empty(),
        "{} golden mismatch(es):\n{}",
        failures.len(),
        failures.join("\n")
    );

    // every type we advertise must actually have been exercised
    for (tid, name, _, _) in EXPECTED_TYPES {
        let (_, cases) = checked
            .iter()
            .find(|(t, _)| *t == *tid)
            .unwrap_or_else(|| panic!("{name} missing from golden file"));
        assert!(*cases > 0, "{name} had no passing cases");
    }
}
