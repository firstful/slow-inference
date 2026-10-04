use slow_inference::gguf;
use slow_inference::quant;

use std::io::{Read, Seek};

fn main() {
    let path = "../models/OLMoE-1B-7B-Instruct-Q4_K_M.gguf";
    let gf = gguf::parse(path).unwrap();
    let t = gf.tensors.get("token_embd.weight").unwrap();
    let ne0 = t.dims[0] as usize;
    let n_rows = t.dims[1] as usize;
    let row_bytes = (t.nbytes as usize) / n_rows;
    println!("ne0={} row_bytes={} dtype={}", ne0, row_bytes, t.dtype);

    let mut f = std::fs::File::open(path).unwrap();
    let start = 510usize;
    let mut raw = vec![0u8; row_bytes];
    f.seek(std::io::SeekFrom::Start(
        gf.data_start + t.offset + (start as u64) * row_bytes as u64,
    ))
    .unwrap();
    f.read_exact(&mut raw).unwrap();

    let row = quant::dequant_row(&raw, t.dtype, ne0).unwrap();
    let sum: f32 = row.iter().map(|v| v.abs()).sum();
    println!("rust row510 abs_sum={:.6} first3={:?}", sum, &row[..3]);
    for b in 0..8 {
        let s: f32 = row[b * 256..(b + 1) * 256].iter().map(|v| v.abs()).sum();
        println!("block {}: abs_sum={:.4}", b, s);
    }
}
