//! Dequantization kernels for GGML quant formats -> f32.
//!
//! Ported from the Python prototype in py-reference/src/quant.py, which was
//! validated BIT-EXACT against gguf-py's dequantize on real model tensors
//! (Tiny-Moe Q4_K + Q6_K rows, max abs err 0.00e+00).
//! Layouts follow ggml/src/ggml-quants.c dequantize_row_* exactly.
//!
//! PR-1: Added Q2_K, Q3_K, Q5_K, Q8_K dequant + BF16 passthrough.
//! Fixed Q8_K type_layout from (4,8) to (292,256) in gguf.rs.

use std::io::{self, Read, Seek, SeekFrom};

use crate::gguf::{
    GGML_BF16, GGML_F16, GGML_F32, GGML_IQ1_M, GGML_IQ1_S, GGML_IQ2_S, GGML_IQ2_XS, GGML_IQ2_XXS,
    GGML_IQ3_S, GGML_IQ3_XXS, GGML_IQ4_NL, GGML_IQ4_XS, GGML_Q2_K, GGML_Q3_K, GGML_Q4_0, GGML_Q4_K,
    GGML_Q5_K, GGML_Q6_K, GGML_Q8_0, GGML_Q8_K, GGML_TQ1_0, GGML_TQ2_0, type_layout,
};
use crate::quant_tables::{
    IQ1S_DELTA, IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL, QK_K, QK4_NL,
};

fn f16_to_f32(bits: u16) -> f32 {
    // IEEE half -> single, bit-exact arithmetic port
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits & 0x7C00) >> 10) as u32;
    let frac = (bits & 0x03FF) as u32;
    let out = if exp == 0 {
        // subnormal half -> subnormal single: value = frac * 2^-24
        (frac as f32 * 2f32.powi(-24)).to_bits()
    } else if exp == 31 {
        if frac == 0 {
            0x7F80_0000 // inf
        } else {
            0x7FC0_0000 | frac // NaN
        }
    } else {
        ((exp - 15 + 127) << 23) | (frac << 13)
    };
    f32::from_bits(sign | out)
}

/// Dequantize ONE row of `ne0` elements. `raw` holds exactly the row's bytes.
/// Returns a fresh Vec<f32> of length ne0.
pub fn dequant_row(raw: &[u8], dtype: u32, ne0: usize) -> io::Result<Vec<f32>> {
    match dtype {
        GGML_F32 => Ok(raw[..ne0 * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect()),
        GGML_F16 => Ok(raw[..ne0 * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| f16_to_f32(u16::from_le_bytes(c)))
            .collect()),
        GGML_BF16 => {
            // BF16 -> F32: left-shift by 16 bits (same as F32 but low 16 bits are zero)
            Ok(raw[..ne0 * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| {
                    let bits = u16::from_le_bytes(c) as u32;
                    f32::from_bits(bits << 16)
                })
                .collect())
        }
        GGML_Q4_0 => q4_0(raw, ne0),
        GGML_Q8_0 => q8_0(raw, ne0),
        GGML_Q8_K => q8_k(raw, ne0),
        GGML_Q2_K => q2_k(raw, ne0),
        GGML_Q3_K => q3_k(raw, ne0),
        GGML_Q4_K => q4_k(raw, ne0),
        GGML_Q5_K => q5_k(raw, ne0),
        GGML_Q6_K => q6_k(raw, ne0),
        GGML_IQ2_XXS => iq2_xxs(raw, ne0),
        GGML_IQ2_XS => iq2_xs(raw, ne0),
        GGML_IQ3_XXS => iq3_xxs(raw, ne0),
        GGML_IQ1_S => iq1_s(raw, ne0),
        GGML_IQ4_NL => iq4_nl(raw, ne0),
        GGML_IQ3_S => iq3_s(raw, ne0),
        GGML_IQ2_S => iq2_s(raw, ne0),
        GGML_IQ4_XS => iq4_xs(raw, ne0),
        GGML_IQ1_M => iq1_m(raw, ne0),
        GGML_TQ1_0 => tq1_0(raw, ne0),
        GGML_TQ2_0 => tq2_0(raw, ne0),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("dequant for ggml type {dtype} not implemented"),
        )),
    }
}

/// Read a whole row out of `raw`, checking the packed length against
/// `type_layout` first. Every kernel below goes through this so a short or
/// misaligned slice yields an error instead of a panic.
fn row_bytes(dtype: u32, ne0: usize) -> io::Result<usize> {
    let (blk_size, blk_elems) = type_layout(dtype).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("ggml type {dtype} has no known block layout"),
        )
    })?;
    if !ne0.is_multiple_of(blk_elems) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("row length {ne0} is not a multiple of {blk_elems} for ggml type {dtype}"),
        ));
    }
    Ok((ne0 / blk_elems) * blk_size)
}

fn checked(raw: &[u8], need: usize, dtype: u32) -> io::Result<&[u8]> {
    if raw.len() < need {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "row for ggml type {dtype} needs {need} bytes, got {}",
                raw.len()
            ),
        ));
    }
    Ok(&raw[..need])
}

// ---- Q4_0 (legacy, 32-elem block, 18 bytes: f16 d, u8[16] nibbles) ----------

fn q4_0(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    // ggml-quants.c dequantize_row_q4_0: nibbles are offset by 8, and the
    // low nibbles fill the first half of the block while the high nibbles
    // fill the second half.
    let need = row_bytes(GGML_Q4_0, ne0)?;
    let blk = checked(raw, need, GGML_Q4_0)?;
    let nblk = ne0 / 32;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 18;
        let d = f16_to_f32(u16::from_le_bytes(blk[off..off + 2].try_into().unwrap()));
        for j in 0..16 {
            let byte = blk[off + 2 + j];
            out[b * 32 + j] = ((byte & 0x0F) as i32 - 8) as f32 * d;
            out[b * 32 + 16 + j] = ((byte >> 4) as i32 - 8) as f32 * d;
        }
    }
    Ok(out)
}

// ---- Q8_0 (legacy, 32-elem block, 34 bytes) --------------------------------

fn q8_0(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_Q8_0, ne0)?;
    let blk = checked(raw, need, GGML_Q8_0)?;
    let nblk = ne0 / 32;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 34;
        let d = f16_to_f32(u16::from_le_bytes(blk[off..off + 2].try_into().unwrap()));
        for i in 0..32 {
            let q = blk[off + 2 + i] as i8 as f32;
            out[b * 32 + i] = q * d;
        }
    }
    Ok(out)
}

// ---- Q8_K (256-elem block, 292 bytes: f32 d, i8[256] qs, i16[16] bsums) ----

fn q8_k(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    // block_q8_K: float d, int8_t qs[QK_K], int16_t bsums[QK_K/16]
    // Verified against ggml-common.h static_assert: sizeof = 4 + 256 + 32 = 292
    let need = row_bytes(GGML_Q8_K, ne0)?;
    let blk = checked(raw, need, GGML_Q8_K)?;
    let nblk = ne0 / 256;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 292;
        // d is f32 (not f16!)
        let d = f32::from_le_bytes(blk[off..off + 4].try_into().unwrap());
        let qs_start = off + 4;
        for j in 0..256 {
            let q = blk[qs_start + j] as i8 as f32;
            out[b * 256 + j] = q * d;
        }
    }
    Ok(out)
}

// ---- Q2_K (256-elem block, 84 bytes) --------------------------------------

fn q2_k(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_Q2_K, ne0)?;
    let raw = checked(raw, need, GGML_Q2_K)?;
    // block_q2_K (ggml-common.h) puts the scales FIRST:
    //   [0..16)  scales  QK_K/16 bytes (4-bit scale | 4-bit min per byte)
    //   [16..80) qs      QK_K/4 bytes, 2 bits per element
    //   [80..82) d       f16 super-block scale for the scales
    //   [82..84) dmin    f16 super-block scale for the mins
    //
    // ggml walks two 128-elem halves; within a half it emits FOUR groups of
    // (16 elems from qs[0..16], then 16 from qs[16..32]) and consumes TWO scale
    // bytes per group. The scale index therefore advances twice as fast as the
    // group index: is = 2*j, 2*j+1 within a half, offset by 8 per half.
    let nblk = ne0 / 256;
    let mut out = vec![0f32; ne0];
    let mut pos = 0usize;
    for b in 0..nblk {
        let off = b * 84;
        let sc = &raw[off..off + 16];
        let qs = &raw[off + 16..off + 80];
        let d = f16_to_f32(u16::from_le_bytes(
            raw[off + 80..off + 82].try_into().unwrap(),
        ));
        let dmin = f16_to_f32(u16::from_le_bytes(
            raw[off + 82..off + 84].try_into().unwrap(),
        ));

        // Two 128-elem halves
        for half in 0..2 {
            let q_half = &qs[half * 32..(half + 1) * 32];
            for j in 0..4 {
                let shift = (j * 2) as u8;
                let is = half * 8 + j * 2;

                let sc_byte = sc[is];
                let dl = d * (sc_byte & 0xF) as f32;
                let ml = dmin * ((sc_byte >> 4) as f32);
                for &byte in q_half[0..16].iter() {
                    let q = ((byte >> shift) & 3) as i32;
                    out[pos] = dl * q as f32 - ml;
                    pos += 1;
                }

                let sc_byte = sc[is + 1];
                let dl = d * (sc_byte & 0xF) as f32;
                let ml = dmin * ((sc_byte >> 4) as f32);
                for &byte in q_half[16..32].iter() {
                    let q = ((byte >> shift) & 3) as i32;
                    out[pos] = dl * q as f32 - ml;
                    pos += 1;
                }
            }
        }
    }
    Ok(out)
}

// ---- Q3_K (256-elem block, 110 bytes) --------------------------------------

fn q3_k(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_Q3_K, ne0)?;
    let raw = checked(raw, need, GGML_Q3_K)?;
    // block_q3_K (ggml-common.h) starts with the high-bit mask:
    //   [0..32)   hmask  QK_K/8 bytes, one high-bit mask per element
    //   [32..96)  qs     QK_K/4 bytes, 2 bits per element
    //   [96..108) scales 12 bytes, 16 six-bit scales
    //   [108..110) d     f16 super-block scale
    let nblk = ne0 / 256;
    let mut out = vec![0f32; ne0];
    let mut pos = 0usize;
    for b in 0..nblk {
        let off = b * 110;
        let hm = &raw[off..off + 32]; // 32 bytes, reused by both halves
        let qs = &raw[off + 32..off + 96];
        let scale_raw = &raw[off + 96..off + 108];
        let d_all = f16_to_f32(u16::from_le_bytes(
            raw[off + 108..off + 110].try_into().unwrap(),
        ));

        // Decode 12 bytes into 16 signed int8 scales: match ggml-quants.c logic
        let kmask1: u32 = 0x03030303;
        let kmask2: u32 = 0x0f0f0f0f;

        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes(scale_raw[0..4].try_into().unwrap());
        aux[1] = u32::from_le_bytes(scale_raw[4..8].try_into().unwrap());
        aux[2] = u32::from_le_bytes(scale_raw[8..12].try_into().unwrap());

        let tmp = aux[2];
        let new_aux0 = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
        let new_aux1 = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let new_aux2 = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        let new_aux3 = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);

        // C reads the unpacked scales through an `int8_t *`, so each byte is
        // sign-extended and the `- 32` happens in int, not int8. Keep i32
        // arithmetic so arbitrary bytes cannot wrap.
        let mut s = [0i32; 16];
        for i in 0..4 {
            s[i] = new_aux0.to_le_bytes()[i] as i8 as i32 - 32;
            s[4 + i] = new_aux1.to_le_bytes()[i] as i8 as i32 - 32;
            s[8 + i] = new_aux2.to_le_bytes()[i] as i8 as i32 - 32;
            s[12 + i] = new_aux3.to_le_bytes()[i] as i8 as i32 - 32;
        }

        // Two 128-elem halves. hm is the full 32 bytes shared across both halves.
        // m advances 1,2,4,8 per group within a half, then 16,32,64,128 for the
        // next half — matching ggml-quants.c where m starts at 1 and increments
        // 8 times across all 8 groups (4 per half).
        let mut m = 1u8;
        for half in 0..2 {
            let q_half = &qs[half * 32..(half + 1) * 32];
            for j in 0..4 {
                let shift = (j * 2) as u8;
                let sc_idx = half * 8 + j * 2;

                let dl = d_all * s[sc_idx] as f32;
                for (&q_byte, &h_byte) in q_half[0..16].iter().zip(hm[0..16].iter()) {
                    let q_val = ((q_byte >> shift) & 3) as i32;
                    let sign = if (h_byte & m) != 0 { 0 } else { 4 };
                    out[pos] = dl * (q_val - sign) as f32;
                    pos += 1;
                }

                let dl = d_all * s[sc_idx + 1] as f32;
                for (&q_byte, &h_byte) in q_half[16..32].iter().zip(hm[16..32].iter()) {
                    let q_val = ((q_byte >> shift) & 3) as i32;
                    let sign = if (h_byte & m) != 0 { 0 } else { 4 };
                    out[pos] = dl * (q_val - sign) as f32;
                    pos += 1;
                }

                m <<= 1;
            }
        }
    }
    Ok(out)
}

// ---- Q4_K helper: decode scale/min from the 12-byte pack -------------------

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

// ---- Q4_K (256-elem block, 144 bytes) --------------------------------------

fn q4_k(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_Q4_K, ne0)?;
    let raw = checked(raw, need, GGML_Q4_K)?;
    // 256-elem blocks, 144 B: 2B d, 2B dmin, 12B scales, 128B nibbles.
    // Per 64-elem group g: 32 bytes; low nibbles -> first 32 outputs (scale 2g),
    // high nibbles -> next 32 outputs (scale 2g+1).
    let nblk = ne0 / 256;
    let mut out = vec![0f32; ne0];
    let mut pos = 0usize;
    for b in 0..nblk {
        let off = b * 144;
        let d = f16_to_f32(u16::from_le_bytes(raw[off..off + 2].try_into().unwrap()));
        let dmin = f16_to_f32(u16::from_le_bytes(
            raw[off + 2..off + 4].try_into().unwrap(),
        ));
        let sc = &raw[off + 4..off + 16];
        let qs = &raw[off + 16..off + 144];
        for g in 0..4 {
            let (s0, m0) = get_scale_min_k4(2 * g, sc);
            let (s1, m1) = get_scale_min_k4(2 * g + 1, sc);
            let (d1, m1f) = (d * s0 as f32, dmin * m0 as f32);
            let (d2, m2f) = (d * s1 as f32, dmin * m1 as f32);
            let q = &qs[g * 32..(g + 1) * 32];
            for i in 0..32 {
                out[pos + i] = d1 * (q[i] & 0xF) as f32 - m1f;
                out[pos + 32 + i] = d2 * (q[i] >> 4) as f32 - m2f;
            }
            pos += 64;
        }
    }
    Ok(out)
}

// ---- Q5_K (256-elem block, 176 bytes) --------------------------------------

fn q5_k(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_Q5_K, ne0)?;
    let raw = checked(raw, need, GGML_Q5_K)?;
    // block_q5_K layout:
    //   [0..2)    d       f16 super-block scale
    //   [2..4)    dmin    f16 super-block scale for mins
    //   [4..16)   scales  12B (same packing as Q4_K)
    //   [16..18)  (padding / alignment - skipped, qh starts at 16)
    //   [16..48)  qh      32B = QK_K/8 (high bit for each quant)
    //   [48..176) qs      128B = QK_K/2 (low 4 bits)
    //
    // Wait — check size: 2+2+12+32+128 = 176. Correct.
    // Each 64-elem group: 32 low-nibbles -> first 32, 32 high-nibbles next 32.
    //   qh: 4 planes of 1 bit each. 4 planes * 64 = 256.
    //   plane 0: bit 0 of qh[0..31] -> first 32 of group (u1)
    //   plane 1: bit 1 of qh[0..31] -> last 32 of group (u2)
    //   plane 2: bit 2 of qh[0..31] -> next group first
    //   plane 3: bit 3 of qh[0..31] -> next group last
    let nblk = ne0 / 256;
    let mut out = vec![0f32; ne0];
    let mut pos = 0usize;
    for b in 0..nblk {
        let off = b * 176;
        let d = f16_to_f32(u16::from_le_bytes(raw[off..off + 2].try_into().unwrap()));
        let dmin = f16_to_f32(u16::from_le_bytes(
            raw[off + 2..off + 4].try_into().unwrap(),
        ));
        let sc = &raw[off + 4..off + 16];
        let qh_bytes = &raw[off + 16..off + 48];
        let ql = &raw[off + 48..off + 176];

        for g in 0..4 {
            let is0 = 2 * g;
            let is1 = 2 * g + 1;
            let (sc0, m0) = get_scale_min_k4(is0, sc);
            let (sc1, m1) = get_scale_min_k4(is1, sc);
            let d1 = d * sc0 as f32;
            let m1f = dmin * m0 as f32;
            let d2 = d * sc1 as f32;
            let m2f = dmin * m1 as f32;

            // qh is the same 32 bytes for all groups — only the bit masks change.
            //   u1 = 1 << (g*2)   (bits 0,2,4,6 of each qh byte)
            //   u2 = 2 << (g*2)   (bits 1,3,5,7 of each qh byte)
            let u1 = 1u8 << (g * 2);
            let u2 = 2u8 << (g * 2);

            let qg = &ql[g * 32..(g + 1) * 32];
            // first 32 elements: use ql low nibble + qh bit plane u1 (always qh[0..32))
            for l in 0..32 {
                let low = qg[l] & 0xF;
                let high_bit = if (qh_bytes[l] & u1) != 0 { 16 } else { 0 };
                out[pos] = d1 * (low + high_bit) as f32 - m1f;
                pos += 1;
            }
            // second 32 elements: use ql high nibble + qh bit plane u2 (always qh[0..32))
            for l in 0..32 {
                let low = qg[l] >> 4;
                let high_bit = if (qh_bytes[l] & u2) != 0 { 16 } else { 0 };
                out[pos] = d2 * (low + high_bit) as f32 - m2f;
                pos += 1;
            }
        }
    }
    Ok(out)
}

// ---- Q6_K (256-elem block, 210 bytes) --------------------------------------

fn q6_k(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_Q6_K, ne0)?;
    let raw = checked(raw, need, GGML_Q6_K)?;
    // 256-elem blocks, 210 B = 2 superblocks of 128 elements.
    // Layout: 128B ql | 64B qh | 16B scales (i8) | 2B d (f16).
    // Per superblock s: ql[s*64..], qh[s*32..], scales[s*8..].
    // ql nibble order: lo bytes 0..64, hi bytes 0..64; qh gives 4 2-bit
    // planes (b&3, >>2&3, >>4&3, >>6&3). q = ql|(qh<<4) - 32, 8 groups of 16.
    let nblk = ne0 / 256;
    let mut out = vec![0f32; ne0];
    let mut pos = 0usize;
    for b in 0..nblk {
        let off = b * 210;
        let blk = &raw[off..off + 210];
        let d = f16_to_f32(u16::from_le_bytes(blk[208..210].try_into().unwrap()));
        for sb in 0..2 {
            let ql = &blk[sb * 64..sb * 64 + 64];
            let qh = &blk[128 + sb * 32..128 + sb * 32 + 32];
            let mut ql_nib = [0u8; 128];
            let mut qh_bits = [0u8; 128];
            for i in 0..64 {
                ql_nib[i] = ql[i] & 0xF;
                ql_nib[64 + i] = ql[i] >> 4;
            }
            for i in 0..32 {
                qh_bits[i] = qh[i] & 3;
                qh_bits[32 + i] = (qh[i] >> 2) & 3;
                qh_bits[64 + i] = (qh[i] >> 4) & 3;
                qh_bits[96 + i] = (qh[i] >> 6) & 3;
            }
            for g in 0..8 {
                let sc = blk[192 + sb * 8 + g] as i8 as f32;
                let base = g * 16;
                for i in 0..16 {
                    let q =
                        ((ql_nib[base + i] | (qh_bits[base + i] << 4)) as i8 as i32 - 32) as f32;
                    out[pos + sb * 128 + base + i] = d * sc * q;
                }
            }
        }
        pos += 256;
    }
    Ok(out)
}

// ---- i-quants (codebook-indexed) --------------------------------------------
//
// All of these walk the block in QK_K/32-element groups, where a group is 8
// output values per 4 sub-groups of 8. `x[i].qs` etc. are byte arrays even
// where the layout is conceptually u32/u16, so the index arithmetic below is
// plain byte slicing, matching ggml-quants.c op for op.

/// Sign mask byte for 8 consecutive grid elements (ggml's kmask_iq2xs).
#[inline]
fn sign_of(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Byte `i` of a codebook grid packed into a little-endian integer word.
/// The grids are `uint64_t`/`uint32_t` in C, indexed by the packed index;
/// element `j` of grid `g` is byte `j` of `g`'s little-endian encoding.
#[inline]
fn grid_byte(word: u64, j: usize) -> u8 {
    (word >> (8 * j)) as u8
}

// ---- IQ4_NL (32-elem block, 18 bytes: f16 d, u8[16]) ------------------------

fn iq4_nl(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_IQ4_NL, ne0)?;
    let blk = checked(raw, need, GGML_IQ4_NL)?;
    let nblk = ne0 / QK4_NL;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 18;
        let d = f16_to_f32(u16::from_le_bytes(blk[off..off + 2].try_into().unwrap()));
        let qs = &blk[off + 2..off + 18];
        let base = b * QK4_NL;
        for j in 0..QK4_NL / 2 {
            let byte = qs[j];
            out[base + j] = d * KVALUES_IQ4NL[(byte & 0xf) as usize] as f32;
            out[base + QK4_NL / 2 + j] = d * KVALUES_IQ4NL[(byte >> 4) as usize] as f32;
        }
    }
    Ok(out)
}

// ---- IQ4_XS (256-elem block, 136 bytes: f16 d, u16 scales_h, u8[4] scales_l,
//      u8[128] qs) -----------------------------------------------------------

fn iq4_xs(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_IQ4_XS, ne0)?;
    let blk = checked(raw, need, GGML_IQ4_XS)?;
    let nblk = ne0 / QK_K;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 136;
        let d = f16_to_f32(u16::from_le_bytes(blk[off..off + 2].try_into().unwrap()));
        let scales_h = u16::from_le_bytes(blk[off + 2..off + 4].try_into().unwrap());
        let scales_l = &blk[off + 4..off + 8];
        let qs = &blk[off + 8..off + 136];
        let base = b * QK_K;
        for ib in 0..(QK_K / 32) {
            let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                | ((((scales_h >> (2 * ib)) & 3) as u8) << 4);
            let dl = d * (ls as i32 - 32) as f32;
            let q = &qs[ib * 16..ib * 16 + 16];
            for j in 0..16 {
                out[base + ib * 32 + j] = dl * KVALUES_IQ4NL[(q[j] & 0xf) as usize] as f32;
                out[base + ib * 32 + 16 + j] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
            }
        }
    }
    Ok(out)
}

// ---- IQ2_XXS (256-elem block, 66 bytes: f16 d, u8[64] qs) -------------------

fn iq2_xxs(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_IQ2_XXS, ne0)?;
    let blk = checked(raw, need, GGML_IQ2_XXS)?;
    let nblk = ne0 / QK_K;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 66;
        let d = f16_to_f32(u16::from_le_bytes(blk[off..off + 2].try_into().unwrap()));
        let qs = &blk[off + 2..off + 66];
        let base = b * QK_K;
        for ib32 in 0..(QK_K / 32) {
            // ggml memcpys 8 bytes into two u32: aux32[1]'s top nibble is the
            // 3-bit scale, its bytes 1..8 hold 4 x 7-bit codebook indices.
            let aux0 = u32::from_le_bytes(qs[ib32 * 8..ib32 * 8 + 4].try_into().unwrap());
            let aux1 = u32::from_le_bytes(qs[ib32 * 8 + 4..ib32 * 8 + 8].try_into().unwrap());
            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[(aux0 >> (8 * l)) as usize & 0xff];
                let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    out[base + ib32 * 32 + l * 8 + j] =
                        db * grid_byte(grid, j) as f32 * sign_of(signs, j);
                }
            }
        }
    }
    Ok(out)
}

// ---- IQ2_XS (256-elem block, 74 bytes: f16 d, u8[64] qs, u8[8] scales) ------

fn iq2_xs(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_IQ2_XS, ne0)?;
    let blk = checked(raw, need, GGML_IQ2_XS)?;
    let nblk = ne0 / QK_K;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 74;
        let d = f16_to_f32(u16::from_le_bytes(blk[off..off + 2].try_into().unwrap()));
        let qs = &blk[off + 2..off + 66];
        let scales = &blk[off + 66..off + 74];
        let base = b * QK_K;
        for ib32 in 0..(QK_K / 32) {
            let sc = scales[ib32];
            let db0 = d * (0.5 + (sc & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + (sc >> 4) as f32) * 0.25;
            for l in 0..4 {
                // qs is uint16_t[QK_K/8] in ggml-common.h, so C's qs[4*ib32 + l]
                // is at byte 2*(4*ib32 + l) here.
                let k = 4 * ib32 + l;
                let packed = u16::from_le_bytes(qs[k * 2..k * 2 + 2].try_into().unwrap());
                let grid = IQ2XS_GRID[(packed & 511) as usize];
                let signs = KSIGNS_IQ2XS[(packed >> 9) as usize];
                let db = if l / 2 == 0 { db0 } else { db1 };
                for j in 0..8 {
                    out[base + ib32 * 32 + l * 8 + j] =
                        db * grid_byte(grid, j) as f32 * sign_of(signs, j);
                }
            }
        }
    }
    Ok(out)
}

// ---- IQ2_S (256-elem block, 82 bytes: f16 d, u8[64] qs, u8[8] qh,
//      u8[8] scales) ---------------------------------------------------------

fn iq2_s(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    // layout: d | qs[64] | qh[8] | scales[8]; the sign bits live in the
    // upper half of `qs` (qs + QK_K/8).
    let need = row_bytes(GGML_IQ2_S, ne0)?;
    let blk = checked(raw, need, GGML_IQ2_S)?;
    let nblk = ne0 / QK_K;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 82;
        let d = f16_to_f32(u16::from_le_bytes(blk[off..off + 2].try_into().unwrap()));
        let qs = &blk[off + 2..off + 66];
        let qh = &blk[off + 66..off + 74];
        let scales = &blk[off + 74..off + 82];
        let signs = &qs[32..64];
        let base = b * QK_K;
        for ib32 in 0..(QK_K / 32) {
            let sc = scales[ib32];
            let db0 = d * (0.5 + (sc & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + (sc >> 4) as f32) * 0.25;
            for l in 0..4 {
                let dl = if l / 2 == 0 { db0 } else { db1 };
                let idx =
                    (qs[ib32 * 4 + l] as usize) | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[idx];
                let sign = signs[ib32 * 4 + l];
                for j in 0..8 {
                    out[base + ib32 * 32 + l * 8 + j] =
                        dl * grid_byte(grid, j) as f32 * sign_of(sign, j);
                }
            }
        }
    }
    Ok(out)
}

// ---- IQ3_XXS (256-elem block, 98 bytes: f16 d, u8[96] qs) ------------------

fn iq3_xxs(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    // layout: d | qs[96]; qs[0..64] are 2 x 256 4-bit grid indices, the
    // remaining 32 bytes hold 8 x (3-bit scale + 4 x 7-bit sign patterns).
    let need = row_bytes(GGML_IQ3_XXS, ne0)?;
    let blk = checked(raw, need, GGML_IQ3_XXS)?;
    let nblk = ne0 / QK_K;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 98;
        let d = f16_to_f32(u16::from_le_bytes(blk[off..off + 2].try_into().unwrap()));
        let qs = &blk[off + 2..off + 98];
        let base = b * QK_K;
        for ib32 in 0..(QK_K / 32) {
            let aux = u32::from_le_bytes(qs[64 + ib32 * 4..64 + ib32 * 4 + 4].try_into().unwrap());
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l] as usize];
                let g2 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l + 1] as usize];
                for j in 0..4 {
                    out[base + ib32 * 32 + l * 8 + j] =
                        db * grid_byte(g1 as u64, j) as f32 * sign_of(signs, j);
                    out[base + ib32 * 32 + l * 8 + 4 + j] =
                        db * grid_byte(g2 as u64, j) as f32 * sign_of(signs, j + 4);
                }
            }
        }
    }
    Ok(out)
}

// ---- IQ3_S (256-elem block, 110 bytes: f16 d, u8[64] qs, u8[8] qh,
//      u8[32] signs, u8[4] scales) ------------------------------------------

fn iq3_s(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_IQ3_S, ne0)?;
    let blk = checked(raw, need, GGML_IQ3_S)?;
    let nblk = ne0 / QK_K;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 110;
        let d = f16_to_f32(u16::from_le_bytes(blk[off..off + 2].try_into().unwrap()));
        let qs = &blk[off + 2..off + 66];
        let qh = &blk[off + 66..off + 74];
        let signs = &blk[off + 74..off + 106];
        let scales = &blk[off + 106..off + 110];
        let base = b * QK_K;
        let mut qs_off = 0usize;
        let mut sign_base = 0usize;
        let mut qh_off = 0usize;
        for (pair, &sc) in scales.iter().enumerate() {
            let db1 = d * (1.0 + 2.0 * ((sc & 0xf) as f32));
            let db2 = d * (1.0 + 2.0 * ((sc >> 4) as f32));
            // ggml consumes 8 sign bytes per 64-element group: the first
            // 32-element half reads signs[l], then `signs += 4`, then the
            // second half reads signs[l] again.
            for half in 0..2 {
                let (db, qb, so) = if half == 0 {
                    (db1, qh[qh_off], sign_base)
                } else {
                    (db2, qh[qh_off + 1], sign_base + 4)
                };
                for l in 0..4 {
                    let g1 = IQ3S_GRID
                        [(qs[qs_off + 2 * l] as usize) | (((qb as usize) << (8 - 2 * l)) & 256)];
                    let g2 = IQ3S_GRID[(qs[qs_off + 2 * l + 1] as usize)
                        | (((qb as usize) << (7 - 2 * l)) & 256)];
                    let sign = signs[so + l];
                    let obase = base + (pair * 64) + half * 32 + l * 8;
                    for j in 0..4 {
                        out[obase + j] = db * grid_byte(g1 as u64, j) as f32 * sign_of(sign, j);
                        out[obase + 4 + j] =
                            db * grid_byte(g2 as u64, j) as f32 * sign_of(sign, j + 4);
                    }
                }
                qs_off += 8;
            }
            sign_base += 8;
            qh_off += 2;
        }
    }
    Ok(out)
}

// ---- IQ1_S (256-elem block, 50 bytes: f16 d, u8[32] qs, u16[8] qh) ---------

fn iq1_s(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_IQ1_S, ne0)?;
    let blk = checked(raw, need, GGML_IQ1_S)?;
    let nblk = ne0 / QK_K;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 50;
        let d = f16_to_f32(u16::from_le_bytes(blk[off..off + 2].try_into().unwrap()));
        let qs = &blk[off + 2..off + 34];
        let qh = &blk[off + 34..off + 50];
        let base = b * QK_K;
        for ib in 0..(QK_K / 32) {
            let hv = u16::from_le_bytes(qh[ib * 2..ib * 2 + 2].try_into().unwrap());
            let dl = d * (2.0 * (((hv >> 12) & 7) as f32) + 1.0);
            let delta = if hv & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for l in 0..4 {
                let idx = (qs[ib * 4 + l] as usize) | ((((hv >> (3 * l)) & 7) as usize) << 8);
                let grid = IQ1S_GRID[idx];
                for j in 0..8 {
                    let g = grid_byte(grid, j) as i8;
                    out[base + ib * 32 + l * 8 + j] = dl * (g as f32 + delta);
                }
            }
        }
    }
    Ok(out)
}

// ---- IQ1_M (256-elem block, 56 bytes: u8[32] qs, u8[16] qh, u8[8] scales) --

fn iq1_m(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    // layout: qs[32] | qh[16] | scales[8]; the 4-bit super-block scale is
    // packed across the top nibbles of all four u16 scale words.
    let need = row_bytes(GGML_IQ1_M, ne0)?;
    let blk = checked(raw, need, GGML_IQ1_M)?;
    let nblk = ne0 / QK_K;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 56;
        let qs = &blk[off..off + 32];
        let qh = &blk[off + 32..off + 48];
        let scales = &blk[off + 48..off + 56];
        let sc: [u16; 4] = std::array::from_fn(|i| {
            u16::from_le_bytes(scales[i * 2..i * 2 + 2].try_into().unwrap())
        });
        let scale_bits =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = f16_to_f32(scale_bits);

        let base = b * QK_K;
        for ib in 0..(QK_K / 32) {
            let dl1 = d * (2.0 * (((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) as f32) + 1.0);
            let dl2 = d * (2.0 * (((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as f32) + 1.0);
            let h0 = qh[ib * 2];
            let h1 = qh[ib * 2 + 1];
            let idx = [
                qs[ib * 4] as usize | (((h0 as usize) << 8) & 0x700),
                qs[ib * 4 + 1] as usize | (((h0 as usize) << 4) & 0x700),
                qs[ib * 4 + 2] as usize | (((h1 as usize) << 8) & 0x700),
                qs[ib * 4 + 3] as usize | (((h1 as usize) << 4) & 0x700),
            ];
            let delta = [
                if h0 & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if h0 & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if h1 & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if h1 & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
            ];
            for l in 0..4 {
                let dl = if l < 2 { dl1 } else { dl2 };
                let grid = IQ1S_GRID[idx[l]];
                let obase = base + ib * 32 + l * 8;
                for j in 0..8 {
                    let g = grid_byte(grid, j) as i8;
                    out[obase + j] = dl * (g as f32 + delta[l]);
                }
            }
        }
    }
    Ok(out)
}

// ---- TQ2_0 (256-elem block, 66 bytes: u8[64] qs, f16 d) --------------------

fn tq2_0(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    let need = row_bytes(GGML_TQ2_0, ne0)?;
    let blk = checked(raw, need, GGML_TQ2_0)?;
    let nblk = ne0 / QK_K;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 66;
        let qs = &blk[off..off + 64];
        let d = f16_to_f32(u16::from_le_bytes(
            blk[off + 64..off + 66].try_into().unwrap(),
        ));
        let base = b * QK_K;
        let mut pos = 0usize;
        // 4 interleaved 2-bit planes per 32-byte chunk
        for j in (0..64).step_by(32) {
            for l in 0..4 {
                for m in 0..32 {
                    let q = ((qs[j + m] >> (l * 2)) & 3) as i32 - 1;
                    out[base + pos] = q as f32 * d;
                    pos += 1;
                }
            }
        }
    }
    Ok(out)
}

// ---- TQ1_0 (256-elem block, 54 bytes: u8[48] qs, u8[4] qh, f16 d) ---------

fn tq1_0(raw: &[u8], ne0: usize) -> io::Result<Vec<f32>> {
    // Ternary quants. Each byte of qs holds 5 trits as base-3 digits; ggml
    // multiplies by 3^n and rescales with ((q*3)>>8) - 1. The 4 qh bytes hold
    // a 5th trit group with 8 trits each, so the block splits into a 240-trit
    // qs region and a 16-trit qh region.
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    const QS_LEN: usize = 48;
    const QH_LEN: usize = 4;

    let need = row_bytes(GGML_TQ1_0, ne0)?;
    let blk = checked(raw, need, GGML_TQ1_0)?;
    let nblk = ne0 / QK_K;
    let mut out = vec![0f32; ne0];
    for b in 0..nblk {
        let off = b * 54;
        let qs = &blk[off..off + QS_LEN];
        let qh = &blk[off + QS_LEN..off + QS_LEN + QH_LEN];
        let d = f16_to_f32(u16::from_le_bytes(
            blk[off + QS_LEN + QH_LEN..off + 54].try_into().unwrap(),
        ));
        let base = b * QK_K;
        let mut pos = 0usize;

        // ggml emits 32-byte windows of qs first (the largest multiple of 32
        // that fits), then the 16-byte tail, then 4 planes over qh.
        let emit = |out: &mut [f32], pos: &mut usize, src: &[u8], planes: &[u8]| {
            for &p3 in planes {
                for &byte in src {
                    // C stores the trit product in a uint8_t, so it wraps
                    let q = (byte as u16 * p3 as u16) & 0xff;
                    let xi = ((q * 3) >> 8) as i32;
                    out[*pos] = (xi - 1) as f32 * d;
                    *pos += 1;
                }
            }
        };
        let main_len = QS_LEN - QS_LEN % 32; // 32 bytes -> 160 trits
        for chunk in qs[..main_len].chunks(32) {
            emit(&mut out[base..], &mut pos, chunk, &POW3[..5]);
        }
        for chunk in qs[main_len..].chunks(16) {
            emit(&mut out[base..], &mut pos, chunk, &POW3[..5]);
        }
        for &p3 in &POW3[..4] {
            emit(&mut out[base..], &mut pos, qh, &[p3]);
        }
        debug_assert_eq!(pos, QK_K);
    }
    Ok(out)
}

/// Read + dequantize `n_rows` consecutive rows starting at flat row `start`.
/// `row_bytes` is the packed row size on disk. Reads at most ~one row-block
/// at a time into a reusable buffer (RSS discipline: no full-tensor buffer).
#[allow(clippy::too_many_arguments)]
pub fn read_rows_streamed(
    f: &mut std::fs::File,
    base: u64,
    dtype: u32,
    ne0: usize,
    start: usize,
    n_rows: usize,
    row_bytes: usize,
    buf: &mut Vec<u8>,
) -> io::Result<Vec<f32>> {
    let (blk_size, blk_elems) =
        type_layout(dtype).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad type"))?;
    debug_assert_eq!(row_bytes, (ne0 / blk_elems) * blk_size);
    let mut out = vec![0f32; ne0 * n_rows];

    // stream in chunks of up to 256 KiB of packed rows
    let chunk_rows = (262_144 / row_bytes).max(1);
    let mut done = 0usize;
    while done < n_rows {
        let n = chunk_rows.min(n_rows - done);
        let need = n * row_bytes;
        buf.clear();
        buf.resize(need, 0);
        f.seek(SeekFrom::Start(
            base + (start + done) as u64 * row_bytes as u64,
        ))?;
        f.read_exact(&mut buf[..need])?;
        for r in 0..n {
            let row = dequant_row(&buf[r * row_bytes..(r + 1) * row_bytes], dtype, ne0)?;
            out[(done + r) * ne0..(done + r + 1) * ne0].copy_from_slice(&row);
        }
        done += n;
    }
    Ok(out)
}

/// Read raw packed bytes without dequant (for GPU upload).
/// Same streaming pattern as read_rows_streamed but returns raw bytes.
#[allow(clippy::too_many_arguments)]
pub fn read_rows_raw(
    f: &mut std::fs::File,
    base: u64,
    start: usize,
    n_rows: usize,
    row_bytes: usize,
    buf: &mut Vec<u8>,
) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(n_rows * row_bytes);
    let chunk_rows = (262_144 / row_bytes).max(1);
    let mut done = 0usize;
    while done < n_rows {
        let n = chunk_rows.min(n_rows - done);
        let need = n * row_bytes;
        buf.clear();
        buf.resize(need, 0);
        f.seek(SeekFrom::Start(
            base + (start + done) as u64 * row_bytes as u64,
        ))?;
        f.read_exact(&mut buf[..need])?;
        out.extend_from_slice(&buf[..need]);
        done += n;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Little-endian ggml_half bytes for `v`, so tests can poke scales directly.
    /// Handles the normal f32 range, which is all the tests need.
    fn half(v: f32) -> [u8; 2] {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = (((bits >> 23) & 0xff) as i32 - 127 + 15) as u16;
        let frac = ((bits >> 13) & 0x3ff) as u16;
        (sign | (exp << 10) | frac).to_le_bytes()
    }

    // Smoke test: dequant a single Q8_K block and verify shape + non-NaN
    #[test]
    fn dequant_q8_k_smoke() {
        // One block: 292 bytes, 256 elems
        let raw = vec![0u8; 292];
        // d = f32(0), so all outputs should be 0
        let out = q8_k(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn dequant_q2_k_smoke() {
        let raw = vec![0u8; 84];
        let out = q2_k(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        // d=0, dmin=0 -> all zeros
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn dequant_q3_k_smoke() {
        let raw = vec![0u8; 110];
        let out = q3_k(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn dequant_q5_k_smoke() {
        let raw = vec![0u8; 176];
        let out = q5_k(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn bf16_passthrough() {
        // BF16 1.0 = bits 0x3F80 in lower 16, zero in upper 16
        let raw = vec![0x80u8, 0x3F]; // BF16 of 1.0 as little-endian u16
        let out = dequant_row(&raw, GGML_BF16, 1).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dequant_row_dispatch_q8_k() {
        let raw = vec![0u8; 292];
        let out = dequant_row(&raw, GGML_Q8_K, 256).unwrap();
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn dequant_row_dispatch_q2_k() {
        let raw = vec![0u8; 84];
        let out = dequant_row(&raw, GGML_Q2_K, 256).unwrap();
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn dequant_row_dispatch_q3_k() {
        let raw = vec![0u8; 110];
        let out = dequant_row(&raw, GGML_Q3_K, 256).unwrap();
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn dequant_row_dispatch_q5_k() {
        let raw = vec![0u8; 176];
        let out = dequant_row(&raw, GGML_Q5_K, 256).unwrap();
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn dequant_row_bf16() {
        let raw = vec![0x80u8, 0x3F, 0x00u8, 0x40]; // 1.0, 2.0
        let out = dequant_row(&raw, GGML_BF16, 2).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn dequant_q8_k_nonzero_d() {
        // One block with d=1.0 (f32) and some qs values
        let mut raw = vec![0u8; 292];
        // Set d = 1.0f32 at offset 0
        raw[0..4].copy_from_slice(&f32::to_le_bytes(1.0));
        // Set qs[0] = 42, qs[1] = -3
        raw[4] = 42u8;
        raw[5] = (-3i8) as u8;
        let out = q8_k(&raw, 256).unwrap();
        assert_eq!(out[0], 42.0);
        assert_eq!(out[1], -3.0);
        assert_eq!(out[255], 0.0);
    }

    // ---- regression guards for bugs the ggml golden-vector test found ----
    //
    // tests/quant_golden.rs replays llama.cpp's own kernels and is the real
    // check; these keep the individual layouts pinned even without the binary
    // fixture present.

    /// block_q2_K puts scales[16] and qs[64] FIRST, then d/dmin. Getting this
    /// order wrong silently produced garbage for every Q2_K tensor.
    #[test]
    fn q2_k_field_order_is_scales_qs_d_dmin() {
        let mut raw = vec![0u8; 84];
        raw[0] = 0x21; // scales[0]: scale nibble = 1, min nibble = 2
        raw[16] = 0b00_01_10_11; // qs[0]: 4 quants, 2 bits each
        raw[80..82].copy_from_slice(&half(1.0));
        raw[82..84].copy_from_slice(&half(0.5));
        let out = q2_k(&raw, 256).unwrap();
        // scales[0] = 0x21 -> dl = d*1 = 1, ml = dmin*2 = 1. Group 0 (shift 0)
        // takes one 2-bit quant per byte of qs[0..16], so only out[0] sees the
        // 0b11 we planted; the remaining 15 bytes are zero -> 0*1 - 1 = -1.
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], -1.0);
        assert_eq!(out[15], -1.0);
        // outputs 16..32 read qs[16..32] against scales[1] = 0 -> all zero
        assert_eq!(out[16], 0.0);
        assert_eq!(out[31], 0.0);
    }

    /// block_q3_K starts with hmask[32], then qs[64], scales[12], d.
    #[test]
    fn q3_k_field_order_is_hmask_qs_scales_d() {
        let mut raw = vec![0u8; 110];
        raw[0] = 0b1010_1010; // hmask[0]
        raw[32] = 0b00_01_10_11; // qs[0]
        // scales[0..12] all zero -> every unpacked scale is -32, and the high
        // bit of each packed scale lives in the top nibble, which is zero.
        raw[108..110].copy_from_slice(&half(1.0));
        let out = q3_k(&raw, 256).unwrap();
        // all-zero scale bytes unpack to -32, so every dl is 1.0 * -32 = -32.
        // Each group emits 32 outputs: 16 from qs[0..16] then 16 from qs[16..32],
        // with the 2-bit plane advancing per group.
        //  group 0 (plane 0): out[0] reads qs[0] & 3 = 3, hmask bit 0 clear
        //                      -> subtract 4 -> -32 * -1
        //  group 0 cont.    : out[1] reads qs[1] = 0 -> -32 * -4
        //  group 1 (plane 1): out[32] reads (qs[0] >> 2) & 3 = 2, hmask bit 1
        //                      set -> no adjust -> -32 * 2
        assert_eq!(out[0], -32.0 * (3 - 4) as f32);
        assert_eq!(out[1], -32.0 * (0 - 4) as f32);
        assert_eq!(out[15], -32.0 * (0 - 4) as f32);
        assert_eq!(out[32], -32.0 * 2.0);
    }

    /// GGML_TYPE_Q8_K is 15 in ggml.h. It used to be defined as 16, which is
    /// IQ2_XXS, so every Q8_K tensor was mis-dispatched.
    #[test]
    fn q8_k_type_id_matches_ggml() {
        assert_eq!(crate::gguf::GGML_Q8_K, 15);
        assert_eq!(crate::gguf::GGML_IQ2_XXS, 16);
        let raw = vec![0u8; 292];
        assert!(dequant_row(&raw, 15, 256).is_ok());
        // 66 bytes is an IQ2_XXS row, not a Q8_K row: must not be read as one
        assert!(dequant_row(&[0u8; 66], 16, 256).is_ok());
        assert!(dequant_row(&[0u8; 66], 15, 256).is_err());
    }

    /// Short or non-aligned rows must be errors, never panics.
    #[test]
    fn short_rows_error_instead_of_panicking() {
        assert!(dequant_row(&[0u8; 10], GGML_Q4_0, 32).is_err());
        assert!(dequant_row(&[0u8; 291], GGML_Q8_K, 256).is_err());
        assert!(dequant_row(&[0u8; 84], GGML_Q2_K, 300).is_err());
        assert!(dequant_row(&[0u8; 17], GGML_IQ4_NL, 32).is_err());
        assert!(dequant_row(&[0u8; 135], GGML_IQ4_XS, 256).is_err());
    }

    #[test]
    fn iq4_nl_matches_its_lookup_table() {
        let mut raw = vec![0u8; 18];
        raw[0..2].copy_from_slice(&half(2.0));
        raw[2] = 0x37; // qs[0] = 7 | (14 << 4)
        let out = iq4_nl(&raw, 32).unwrap();
        assert_eq!(out[0], 2.0 * KVALUES_IQ4NL[7] as f32);
        assert_eq!(out[16], 2.0 * KVALUES_IQ4NL[3] as f32);
    }

    #[test]
    fn tq1_0_trit_product_wraps_at_256() {
        // qs[j] * pow3 is computed in a uint8_t in ggml, so 200 * 243 = 48000
        // wraps to 56 -> xi = (56*3)>>8 = 0 -> value -1.
        let mut raw = vec![0u8; 54];
        raw[0] = 200;
        raw[52..54].copy_from_slice(&half(1.0)); // block_tq1_0 puts d last
        let out = tq1_0(&raw, 256).unwrap();
        // position 0 is trit group 0 (pow3 = 1): (200*3)>>8 = 2 -> value +1
        assert_eq!(out[0], 1.0);
        // position 32 is trit group 1 (pow3 = 3): 200*3 = 600 wraps to 88,
        // (88*3)>>8 = 1 -> value 0
        assert_eq!(out[32], 0.0);
        // position 64 is trit group 2 (pow3 = 9): 1800 wraps to 8,
        // (8*3)>>8 = 0 -> value -1
        assert_eq!(out[64], -1.0);
    }

    #[test]
    fn tq2_0_shifts_are_interleaved_planes() {
        let mut raw = vec![0b11_10_01_00u8; 66]; // four quants per byte, 2 bits each
        raw[64..66].copy_from_slice(&half(1.0));
        let out = tq2_0(&raw, 256).unwrap();
        // The four 2-bit planes are read as full sweeps over each 32-byte
        // window, so output 0..32 is plane 0 (bits 0..1 = 0), 32..64 is plane 1
        // (bits 2..3 = 1), 64..96 plane 2 (= 2) and 96..128 plane 3 (= 3).
        const PLANES: [(usize, usize, f32); 4] =
            [(0, 32, -1.0), (32, 64, 0.0), (64, 96, 1.0), (96, 128, 2.0)];
        for (plane, &(lo, hi, want)) in PLANES.iter().enumerate() {
            for (i, &v) in out[lo..hi].iter().enumerate() {
                assert_eq!(v, want, "plane {plane} at {}", lo + i);
            }
        }
    }
}
