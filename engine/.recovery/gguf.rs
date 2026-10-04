//! Minimal GGUF container parser.
//!
//! Reads ONLY the header + tensor directory into memory (a few KB). Tensor
//! data stays on disk; callers get (offset, size) and read what they need
//! with seek()/read(). No mmap of the data section, ever.
//!
//! GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};

pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q4_0: u32 = 2;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_Q2_K: u32 = 10;
pub const GGML_Q3_K: u32 = 11;
pub const GGML_Q4_K: u32 = 12;
pub const GGML_Q5_K: u32 = 13;
pub const GGML_Q6_K: u32 = 14;
pub const GGML_Q8_K: u32 = 16;
pub const GGML_BF16: u32 = 30;

/// (bytes per block, elements per block) — from ggml.h ggml_blck_size/type_size
/// Verified against type_traits[] in ggml.c:
///   Q2_K: sizeof(block_q2_K)=84,  blck=QK_K=256
///   Q3_K: sizeof(block_q3_K)=110, blck=QK_K=256
///   Q4_K: sizeof(block_q4_K)=144, blck=QK_K=256
///   Q5_K: sizeof(block_q5_K)=176, blck=QK_K=256
///   Q6_K: sizeof(block_q6_K)=210, blck=QK_K=256
///   Q8_K: sizeof(block_q8_K)=292, blck=QK_K=256
/// Human-readable name for a GGML data type value.
pub fn dtype_name(etype: u32) -> &'static str {
    match etype {
        GGML_F32 => "F32",
        GGML_F16 => "F16",
        GGML_Q4_0 => "Q4_0",
        GGML_Q8_0 => "Q8_0",
        GGML_Q2_K => "Q2_K",
        GGML_Q3_K => "Q3_K",
        GGML_Q4_K => "Q4_K",
        GGML_Q5_K => "Q5_K",
        GGML_Q6_K => "Q6_K",
        GGML_Q8_K => "Q8_K",
        GGML_BF16 => "BF16",
        _ => "UNKNOWN",
    }
}

pub fn type_layout(etype: u32) -> Option<(usize, usize)> {
    match etype {
        GGML_F32 => Some((4, 1)),
        GGML_F16 => Some((2, 1)),
        GGML_BF16 => Some((2, 1)),
        GGML_Q4_0 => Some((18, 32)),
        GGML_Q8_0 => Some((34, 32)),
        GGML_Q2_K => Some((84, 256)),
        GGML_Q3_K => Some((110, 256)),
        GGML_Q4_K => Some((144, 256)),
        GGML_Q5_K => Some((176, 256)),
        GGML_Q6_K => Some((210, 256)),
        GGML_Q8_K => Some((292, 256)),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ne[0] is fastest-varying (row width), as stored in the file
    pub dims: Vec<u64>,
    pub dtype: u32,
    /// offset relative to the start of the data section
    pub offset: u64,
    pub nbytes: u64,
}

#[derive(Debug)]
pub struct GGUFFile {
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    /// absolute file offset where tensor data begins
    pub data_start: u64,
    pub file_size: u64,
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U8(v) => Some(*v as u64),
            Value::U16(v) => Some(*v as u64),
            Value::U32(v) => Some(*v as u64),
            Value::U64(v) => Some(*v),
            Value::I8(v) => Some(*v as u64),
            Value::I16(v) => Some(*v as u64),
            Value::I32(v) => Some(*v as u64),
            Value::I64(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F32(v) => Some(*v as f64),
            Value::F64(v) => Some(*v),
            Value::U8(v) => Some(*v as f64),
            Value::U16(v) => Some(*v as f64),
            Value::U32(v) => Some(*v as f64),
            Value::U64(v) => Some(*v as f64),
            Value::I8(v) => Some(*v as f64),
            Value::I16(v) => Some(*v as f64),
            Value::I32(v) => Some(*v as f64),
            Value::I64(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_vec_str(&self) -> Option<Vec<&str>> {
        match self {
            Value::Array(a) => a.iter().map(|v| v.as_str()).collect(),
            _ => None,
        }
    }
    pub fn as_vec_u64(&self) -> Option<Vec<u64>> {
        match self {
            Value::Array(a) => a.iter().map(|v| v.as_u64()).collect(),
            _ => None,
        }
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated header",
            ));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str_(&mut self) -> io::Result<String> {
        let n = self.u64()? as usize;
        if n > 10_000_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("insane string len {n}"),
            ));
        }
        let b = self.take(n)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }
    fn value(&mut self, vtype: u32) -> io::Result<Value> {
        Ok(match vtype {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(self.f32()?),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.str_()?),
            9 => {
                let etype = self.u32()?;
                let count = self.u64()? as usize;
                let mut v = Vec::with_capacity(count.min(1_000_000));
                for _ in 0..count {
                    v.push(self.value(etype)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.i64()?),
            12 => Value::F64(self.f64()?),
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "bad value type")),
        })
    }
}

/// Parse GGUF header + tensor directory. Reads only the header region:
/// starts with a 1 MiB window, grows if the vocab arrays don't fit.
pub fn parse(path: &str) -> io::Result<GGUFFile> {
    let mut f = std::fs::File::open(path)?;
    let file_size = f.seek(SeekFrom::End(0))?;
    f.seek(SeekFrom::Start(0))?;

    let mut head = [0u8; 24];
    f.read_exact(&mut head)?;
    let magic = u32::from_le_bytes(head[0..4].try_into().unwrap());
    if magic != 0x4655_4747 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a GGUF file",
        ));
    }
    let version = u32::from_le_bytes(head[4..8].try_into().unwrap());
    let tensor_count = u64::from_le_bytes(head[8..16].try_into().unwrap()) as usize;
    let kv_count = u64::from_le_bytes(head[16..24].try_into().unwrap()) as usize;

    let mut region_size: usize = 1 << 20;
    loop {
        f.seek(SeekFrom::Start(0))?;
        let mut buf = vec![0u8; region_size.min(file_size as usize)];
        f.read_exact(&mut buf)?;
        let mut r = Cursor { buf: &buf, pos: 24 };

        let parsed =
            (|| -> io::Result<(HashMap<String, Value>, HashMap<String, TensorInfo>, usize)> {
                let mut metadata = HashMap::new();
                for _ in 0..kv_count {
                    let key = r.str_()?;
                    let vtype = r.u32()?;
                    let val = r.value(vtype)?;
                    metadata.insert(key, val);
                }
                let mut tensors = HashMap::new();
                for _ in 0..tensor_count {
                    let name = r.str_()?;
                    let n_dims = r.u32()? as usize;
                    let mut dims = Vec::with_capacity(n_dims);
                    for _ in 0..n_dims {
                        dims.push(r.u64()?);
                    }
                    let dtype = r.u32()?;
                    let offset = r.u64()?;
                    let (blk_size, blk_elems) = type_layout(dtype).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unsupported ggml type {dtype}"),
                        )
                    })?;
                    let numel: u64 = dims.iter().product();
                    let nbytes = (numel / blk_elems as u64) * blk_size as u64;
                    tensors.insert(
                        name.clone(),
                        TensorInfo {
                            name,
                            dims,
                            dtype,
                            offset,
                            nbytes,
                        },
                    );
                }
                Ok((metadata, tensors, r.pos))
            })();

        match parsed {
            Ok((metadata, tensors, pos)) => {
                let data_start = (pos as u64 + 31) & !31;
                return Ok(GGUFFile {
                    version,
                    metadata,
                    tensors,
                    data_start,
                    file_size,
                });
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                eprintln!(
                    "[gguf] header region {} bytes hit EOF: {}; growing",
                    region_size, e
                );
                if region_size >= (1 << 30) || region_size >= file_size as usize {
                    return Err(e);
                }
                region_size *= 4;
            }
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tiny_moe() {
        let gf = parse("../models/Tiny-Moe.Q4_K_M.gguf").unwrap();
        assert_eq!(gf.tensors.len(), 123);
        assert!(gf.tensors.contains_key("token_embd.weight"));
        let t = &gf.tensors["token_embd.weight"];
        println!(
            "token_embd dims={:?} dtype={} nbytes={}",
            t.dims, t.dtype, t.nbytes
        );
    }

    #[test]
    fn type_layout_k_quants() {
        // Verify all K-quant block sizes match ggml.c type_traits
        assert_eq!(type_layout(GGML_Q2_K), Some((84, 256)));
        assert_eq!(type_layout(GGML_Q3_K), Some((110, 256)));
        assert_eq!(type_layout(GGML_Q4_K), Some((144, 256)));
        assert_eq!(type_layout(GGML_Q5_K), Some((176, 256)));
        assert_eq!(type_layout(GGML_Q6_K), Some((210, 256)));
        assert_eq!(type_layout(GGML_Q8_K), Some((292, 256)));
        assert_eq!(type_layout(GGML_BF16), Some((2, 1)));
    }
}
