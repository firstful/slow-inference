//! Activation capture for the per-op error budget. SNAPSHOT-ONLY scaffolding,
//! not part of the engine: it exists so `probe_op_budget.c` can be run on the
//! vectors the forward pass *actually* produced rather than on synthetic ones.
//! Inert unless `SLOW_OP_DUMP` names an output file.
use std::cell::RefCell;
use std::io::Write;

thread_local! {
    static SINK: RefCell<Option<std::io::BufWriter<std::fs::File>>> = const { RefCell::new(None) };
    static N: RefCell<u64> = const { RefCell::new(0) };
}

pub fn open() {
    if let Ok(path) = std::env::var("SLOW_OP_DUMP") {
        let f = std::fs::File::create(&path).expect("SLOW_OP_DUMP");
        let mut f = std::io::BufWriter::new(f);
        // magic | version | n_slots  (u32 | u32 | u32)
        f.write_all(b"SOPBUF\0\0").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&4u32.to_le_bytes()).unwrap();
        f.flush().unwrap();
        SINK.with(|s| *s.borrow_mut() = Some(f));
    }
}

/// Record one vector. `name` must be one of the four fixed slots; the file
/// stores them in a fixed order with an index per record so the C side can
/// pair them up without parsing strings.
pub fn record(name: &str, v: &[f32]) {
    let slot: u32 = match name {
        "silu_gate" => 0,
        "softmax_scores" => 1,
        "rope_q" => 2,
        "rope_q_out" => 3,
        _ => return,
    };
    SINK.with(|s| {
        let mut b = s.borrow_mut();
        let Some(f) = b.as_mut() else { return };
        let n = N.with(|c| {
            let mut c = c.borrow_mut();
            let v = *c;
            *c = v + 1;
            v
        });
        f.write_all(&slot.to_le_bytes()).unwrap();
        f.write_all(&(v.len() as u32).to_le_bytes()).unwrap();
        // (n as u32) -- n is a u64 counter and `n.to_le_bytes()` would write
        // EIGHT bytes here, silently skewing every record after the first by
        // four. The reader would then be reading f32 payload bytes as a slot
        // number, which is how this was found: "slot 1052389815 >= 4".
        f.write_all(&(n as u32).to_le_bytes()).unwrap();
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
