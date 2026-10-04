"""Peak-RSS instrumentation via /proc/self/status VmHWM (kernel-truth high-water mark)."""
import os


def current_rss_kib() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def peak_rss_kib() -> int:
    """VmHWM: peak resident set size since process start. Cannot be reset."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def fmt_kib(kib: int) -> str:
    if kib >= (1 << 20):
        return f"{kib / (1 << 20):.2f} GiB"
    if kib >= 1024:
        return f"{kib / 1024:.1f} MiB"
    return f"{kib} KiB"


class RSSGuard:
    """Context manager that records peak RSS delta around a phase."""

    def __init__(self, label: str):
        self.label = label
        self.start_peak = 0

    def __enter__(self):
        self.start_peak = peak_rss_kib()
        return self

    def __exit__(self, *exc):
        end_peak = peak_rss_kib()
        self.delta = end_peak - self.start_peak
        print(f"[rss] {self.label}: peak delta {fmt_kib(self.delta)}, "
              f"process peak now {fmt_kib(end_peak)}")
