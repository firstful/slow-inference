# slow-inference

Disk-streamed LLM inference engine (Rust, zero-external-crate). Weights stay on disk; layer-by-layer row streaming. Goal: minimize peak RAM, accept seconds-per-token.