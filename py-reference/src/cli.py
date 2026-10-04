"""CLI: tokenize, run disk-streamed inference, report peak RSS vs model size.

Usage: .venv/bin/python -m src.cli --model models/Tiny-Moe.Q4_K_M.gguf \
           --prompt "The capital of France is" --tokens 8
"""
import argparse
import os
import sys
import time

import numpy as np


def load_vocab_and_bpe(gf, path):
    """Extract llama-tokenizer vocab + merges from GGUF metadata."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    tokens = []
    field = r.get_field("tokenizer.ggml.tokens")
    parts = field.parts
    # parts: [0]=name len?, structure per reader: data parts alternate len/bytes
    n = int(parts[4][0]) if len(parts) > 4 else None
    # generic: iterate parts in pairs starting at index 4 (after array header)
    i = 4
    while i + 1 < len(parts):
        ln = int(parts[i][0])
        s = bytes(parts[i + 1][:ln]).decode("utf-8", errors="replace")
        tokens.append(s)
        i += 2
    merges = []
    mfield = r.get_field("tokenizer.ggml.merges")
    if mfield is not None:
        i = 4
        while i + 1 < len(mfield.parts):
            ln = int(mfield.parts[i][0])
            s = bytes(mfield.parts[i + 1][:ln]).decode("utf-8", errors="replace")
            merges.append(tuple(s.split(" ")))
            i += 2
    return tokens, merges


def bpe_encode(text: str, vocab: list, merges: list):
    """Tiny GPT-2-style BPE over the llama vocab (with ▁ = space marker)."""
    from collections import Counter
    # simple byte-level fallback + merge application
    words = re_split(text)
    ids = []
    ranks = {m: r for r, m in enumerate(merges)}
    for w in words:
        if not w:
            continue
        symbols = list("▁" + w if not w.startswith("▁") else w)
        symbols = [c if c in vocab else "?" for c in symbols]
        # merge pairs by rank
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(ranks.get(p, 1e12), i) for i, p in enumerate(pairs)]
            if not ranked or min(ranked)[0] == 1e12:
                break
            _, i = min(ranked)
            merged = symbols[i] + symbols[i + 1]
            symbols = symbols[:i] + [merged] + symbols[i + 2:]
        for s in symbols:
            if s in vocab:
                ids.append(vocab.index(s))
            else:
                # byte fallback
                for b in s.encode("utf-8"):
                    tok = f"<0x{b:02X}>"
                    ids.append(vocab.index(tok) if tok in vocab else 0)
    return ids


def re_split(text: str):
    import re
    return re.findall(r"\s?\S+", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=8)
    args = ap.parse_args()

    from .runner import Model
    from .rss import peak_rss_kib, current_rss_kib, fmt_kib

    model_file_size = os.path.getsize(args.model)
    print(f"model: {args.model} ({model_file_size / 1e9:.2f} GB)")

    m = Model(args.model)
    print(f"arch=llama layers={m.n_layers} embd={m.n_embd} heads={m.n_heads} "
          f"ff={m.n_ff} experts={m.n_experts}x{m.n_experts_used}")

    vocab, merges = load_vocab_and_bpe(m.gf, args.model)
    prompt_ids = bpe_encode(args.prompt, vocab, merges)
    print(f"prompt ids: {prompt_ids}")

    t0 = time.perf_counter()
    seq = m.generate(prompt_ids, args.tokens)
    dt = time.perf_counter() - t0

    out_text = "".join(vocab[i] for i in seq[len(prompt_ids):]).replace("▁", " ")
    print(f"output tokens: {seq[len(prompt_ids):]}")
    print(f"output text: {out_text!r}")
    print(f"time: {dt:.1f}s for {args.tokens} tokens ({dt / args.tokens:.1f} s/token)")

    peak = peak_rss_kib() * 1024
    rss = current_rss_kib() * 1024
    print(f"peak RSS: {fmt_kib(peak)} = {100 * peak / model_file_size:.2f}% of model file")
    print(f"current RSS: {fmt_kib(rss)}")
    m.close()


if __name__ == "__main__":
    main()
