//! Per-layer divergence trace: capture the residual stream (and the branch
//! deltas feeding it) at named points inside the forward pass. SNAPSHOT-ONLY
//! scaffolding, in the same spirit as `opdump`: inert unless `SLOW_TRACE`
//! names an output file.
//!
//! Why this exists. Every number this project has for the real model is a
//! single rel L2 at the LOGITS, which is 36 layers of composition with no
//! visibility inside. A per-op error budget does not close that gap -- it was
//! measured in tick 33 and it overshoots the end-to-end residual by three
//! orders of magnitude, because a per-op number does not compose into a
//! per-model number. The trace is the instrument that does: it says whether an
//! error is injected evenly (f32 rounding, irreducible) or amplified at one
//! layer (an op that is small per call and compounds).
//!
//! The slot numbers are llama.cpp's OWN graph node names, so the two sides are
//! compared node for node rather than slot for slot. See
//! `llama_model_qwen2::graph::build_arch_graph` (src/models/qwen2.cpp), where
//! `cb(cur, "attn_norm", il)`, `cb(ffn_inp, "ffn_inp", il)`,
//! `cb(cur, "ffn_norm", il)`, `cb(cur, "ffn_out", il)`, `cb(cur, "l_out", il)`
//! and `cb(cur, "result_norm", -1)` name exactly these tensors.
use std::cell::RefCell;
use std::io::Write;

pub const SLOT_INP_EMBD: i32 = 0;
pub const SLOT_ATTN_NORM: i32 = 1;
pub const SLOT_FFN_INP: i32 = 2;
pub const SLOT_FFN_NORM: i32 = 3;
pub const SLOT_FFN_OUT: i32 = 4;
pub const SLOT_L_OUT: i32 = 5;
pub const SLOT_RESULT_NORM: i32 = 6;

thread_local! {
    static SINK: RefCell<Option<std::io::BufWriter<std::fs::File>>> = const { RefCell::new(None) };
}

pub fn open() {
    if let Ok(path) = std::env::var("SLOW_TRACE") {
        let f = std::fs::File::create(&path).expect("SLOW_TRACE");
        let mut f = std::io::BufWriter::new(f);
        // magic | version | n_slots  (8 | u32 | u32)
        f.write_all(b"SLTRACE\0").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&7u32.to_le_bytes()).unwrap();
        f.flush().unwrap();
        SINK.with(|s| *s.borrow_mut() = Some(f));
    }
}

/// Record one vector. Header per record is `slot | layer | pos | len`
/// (i32 | i32 | u32 | u32) followed by `len` f32, matching
/// `tools/dump_layer_trace.c` byte for byte.
pub fn record(slot: i32, layer: i32, pos: usize, v: &[f32]) {
    SINK.with(|s| {
        let mut b = s.borrow_mut();
        let Some(f) = b.as_mut() else { return };
        f.write_all(&slot.to_le_bytes()).unwrap();
        f.write_all(&layer.to_le_bytes()).unwrap();
        f.write_all(&(pos as u32).to_le_bytes()).unwrap();
        f.write_all(&(v.len() as u32).to_le_bytes()).unwrap();
        for x in v {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    });
}

pub fn close() {
    SINK.with(|s| {
        if let Some(mut f) = s.borrow_mut().take() {
            f.flush().unwrap();
        }
    });
}