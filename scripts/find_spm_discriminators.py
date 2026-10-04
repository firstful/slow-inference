#!/usr/bin/env python3
"""Find texts where llama.cpp's SPM tokenizer and the engine's old greedy
longest-match disagree, on a real SPM vocabulary.

Two reasons this exists rather than just trusting the parity test to fail:

  * a parity test that passes on *every* case because both implementations are
    trivially right is not evidence, and
  * the greedy replacement is not a paraphrase of the Viterbi one, so I want the
    corpus to contain the specific shape of text where they differ -- otherwise
    the green test proves nothing about the swap.

Both implementations below are transcribed from source:
  * llm_tokenizer_spm_session::tokenize  (src/llama-vocab.cpp:117)
  * engine/src/main.rs spm::encode       (greedy longest-match, 24-char cap)

Usage: scripts/find_spm_discriminators.py <gguf> [--dump]
"""

import argparse
import heapq
import os
import sys

_LLAMA_CPP = os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp"))
sys.path.insert(0, os.path.join(_LLAMA_CPP, "gguf-py"))

from gguf import GGUFReader  # noqa: E402

SPACE = "▁"


class Spm:
    """llm_tokenizer_spm_session, transcribed. Validated against llama.cpp's own
    llama_tokenize on the 40-case corpus before being used to pick anything."""

    def __init__(self, toks, scores):
        self.id_of = {t: i for i, t in enumerate(toks)}
        self.score_of = scores
        self.n = len(toks)

    def text_to_token(self, text):
        return self.id_of.get(text, -1)

    def tokenize(self, text):
        """llm_tokenizer_spm_session::tokenize."""
        # split into utf8 chars -> symbols over byte spans of `text`
        offs = []
        i = 0
        for ch in text:
            offs.append(i)
            i += len(ch.encode())
        n = len(offs)
        base = offs[0] if offs else 0
        # symbol k covers text_bytes[offs[k] .. offs[k+1]) where offs[n] = len
        ends = offs[1:] + [i]
        size = [ends[k] - offs[k] for k in range(n)]

        prev = [k - 1 for k in range(n)]
        nxt = [k + 1 if k + 1 < n else -1 for k in range(n)]

        btext = text.encode()

        def sym_text(k):
            return btext[offs[k] : offs[k] + size[k]].decode("utf-8", "surrogateescape")

        queue = []
        rev_merge = {}

        def try_add_bigram(left, right):
            if left == -1 or right == -1:
                return
            s = sym_text(left) + sym_text(right)
            tok = self.text_to_token(s)
            if tok == -1 or tok >= self.n:
                return
            # comparator: higher score first, ties -> lower left first
            # std::priority_queue pops the max under `cmp`; emulate with a
            # negated key through heapq's min-heap.
            heapq.heappush(queue, (-self.score_of[tok], left, right, len(s.encode()), tok, s))
            rev_merge[s] = (left, right)

        for k in range(1, n):
            try_add_bigram(k - 1, k)

        while queue:
            _, left, right, bsize, _, _ = heapq.heappop(queue)
            if size[left] == 0 or size[right] == 0 or size[left] + size[right] != bsize:
                continue
            size[left] += size[right]
            size[right] = 0
            nxt[left] = nxt[right]
            if nxt[right] >= 0:
                prev[nxt[right]] = left
            try_add_bigram(prev[left], left)
            try_add_bigram(left, nxt[left])

        out = []

        def resegment(k):
            s = sym_text(k)
            tok = self.text_to_token(s)
            if tok != -1:
                out.append(tok)
                return
            p = rev_merge.get(s)
            if p is None:
                for b in btext[offs[k] : offs[k] + size[k]]:
                    out.append(self.text_to_token("<0x%02X>" % b))
                return
            resegment(p[0])
            resegment(p[1])

        k = 0
        while k != -1:
            resegment(k)
            k = nxt[k]
        return out

    def encode(self, text, add_space_prefix=True):
        # NOTE: no `if not text` guard, deliberately. The engine's old greedy code
        # had none either, so for the empty prompt it emitted a spurious space
        # marker where llama.cpp emits nothing; that is a real divergence this
        # script exists to surface, not a bug in the transcription.
        t = (" " if add_space_prefix else "") + text
        t = t.replace(" ", SPACE)
        return self.tokenize(t)


def greedy_encode(text, spm, add_space_prefix=True, cap=24):
    """engine/src/main.rs spm::encode, transcribed."""
    if not text:
        return []
    s = SPACE + text.replace(" ", SPACE)
    chars = list(s)
    ids = []
    i = 0
    while i < len(chars):
        matched = None
        for l in range(min(cap, len(chars) - i), 0, -1):
            cand = "".join(chars[i : i + l])
            if cand in spm.id_of:
                matched = cand
                break
        if matched:
            ids.append(spm.id_of[matched])
            i += len(matched)
        else:
            for byte in chars[i].encode():
                ids.append(spm.id_of.get("<0x%02X>" % byte, 0))
            i += 1
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--max-n", type=int, default=6, help="max chars per candidate")
    args = ap.parse_args()

    r = GGUFReader(args.gguf)
    toks = r.fields["tokenizer.ggml.tokens"].contents()
    scores = r.fields["tokenizer.ggml.scores"].contents()
    spm = Spm(toks, scores)
    normal = [t for t, ty in zip(toks, r.fields["tokenizer.ggml.token_type"].contents()) if ty == 1]
    print("vocab %d, normal %d" % (len(toks), len(normal)), file=sys.stderr)

    # Candidates: every 2-symbol merge the Viterbi can make (i.e. every vocab
    # token that is a concatenation of two vocab-or-byte pieces) is too many;
    # instead sweep pairs of short tokens and short byte runs.
    alphabet = [t for t in normal if len(t) <= 3]
    by_first = {}
    for t in alphabet:
        by_first.setdefault(t[0], []).append(t)

    found = []
    seen = set()
    # 1. every short normal token, alone and doubled
    cands = list(alphabet)
    # 2. pairs of short tokens
    import itertools

    pairs = list(itertools.product(alphabet, repeat=2))
    print("candidates: %d tokens + %d pairs" % (len(cands), len(pairs)), file=sys.stderr)

    def check(text):
        if text in seen:
            return
        seen.add(text)
        a = spm.encode(text.replace(SPACE, " "))
        b = greedy_encode(text.replace(SPACE, " "), spm)
        if a != b:
            found.append((text, a, b))
            return True
        return False

    for t in cands[:4000]:
        check(t)
    n = 0
    for x, y in pairs:
        if check(x + y):
            n += 1
            if n >= 25:
                break

    print("divergences: %d" % len(found))
    for text, a, b in found[:25]:
        print(repr(text))
        print("  llama.cpp:", a, [toks[i] for i in a])
        print("  greedy   :", b, [toks[i] for i in b])
        print()


if __name__ == "__main__":
    main()
