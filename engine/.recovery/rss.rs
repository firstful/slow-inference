//! Peak-RSS instrumentation via /proc/self/status VmHWM (kernel high-water mark).

use std::fs;

fn read_status_field(name: &str) -> u64 {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(name) {
            let rest = rest.trim_start_matches(|c: char| c == ':' || c.is_whitespace());
            let kb: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            return kb.parse().unwrap_or(0);
        }
    }
    0
}

pub fn current_rss_kib() -> u64 {
    read_status_field("VmRSS")
}

/// VmHWM: peak resident set size since process start. Cannot be reset.
pub fn peak_rss_kib() -> u64 {
    read_status_field("VmHWM")
}

pub fn fmt(kib: u64) -> String {
    if kib >= (1 << 20) {
        format!("{:.2} GiB", kib as f64 / (1 << 20) as f64)
    } else if kib >= 1024 {
        format!("{:.1} MiB", kib as f64 / 1024.0)
    } else {
        format!("{kib} KiB")
    }
}
