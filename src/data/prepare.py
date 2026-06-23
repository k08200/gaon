"""Download, tokenize, and pack a pretraining corpus into flat .bin token files.

Streams shards from HuggingFace (default: FineWeb-Edu sample), tokenizes with the
Qwen3 tokenizer, and writes a contiguous uint32 token stream that train.py mmaps.

Usage:
    python -m src.data.prepare --dataset fineweb-edu --target-tokens 100_000_000 \
        --out data/fineweb_edu --split train

Scale up by raising --target-tokens and adding more sources (see DATASETS).
Mix multilingual / Korean / code by running multiple times into separate dirs and
listing them all in the training config's `data_dirs`.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")  # multi-core batch encoding

import numpy as np

# Curated source registry. (hf_repo, name, text_field)
DATASETS = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "fineweb-2-kor": ("HuggingFaceFW/fineweb-2", "kor_Hang", "text"),
    "the-stack": ("bigcode/the-stack-v2-dedup", None, "content"),
}

DTYPE = np.uint32  # Qwen3 vocab (151936) fits in uint32


def build_tokenizer(model_id: str = "Qwen/Qwen3-0.6B"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token; cannot delimit documents")
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="fineweb-edu")
    ap.add_argument("--out", required=True, help="output directory for shards")
    ap.add_argument("--split", default="train")
    ap.add_argument("--target-tokens", type=int, default=100_000_000)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000,
                    help="tokens per .bin shard")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    repo, name, field = DATASETS[args.dataset]
    tok = build_tokenizer(args.tokenizer)
    eos = tok.eos_token_id

    ds = load_dataset(repo, name=name, split=args.split, streaming=True)

    buf: list[int] = []
    written = 0
    shard_idx = 0
    batch: list[str] = []

    def flush(tokens: list[int]) -> None:
        nonlocal shard_idx
        path = os.path.join(args.out, f"shard_{shard_idx:05d}.bin")
        np.array(tokens, dtype=DTYPE).tofile(path)
        print(f"  wrote {path}  ({len(tokens):,} tokens)", flush=True)
        shard_idx += 1

    def drain_shards() -> None:
        nonlocal written, buf
        while len(buf) >= args.shard_tokens:
            flush(buf[: args.shard_tokens])
            written += args.shard_tokens
            buf = buf[args.shard_tokens:]
            print(f"progress: {written:,} / {args.target_tokens:,} tokens", flush=True)

    def encode_batch() -> None:
        # Batched call -> the Rust fast tokenizer parallelizes across cores.
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            ids.append(eos)
            buf.extend(ids)
        batch.clear()

    for ex in ds:
        text = ex.get(field)
        if not text:
            continue
        batch.append(text)
        if len(batch) >= 1000:
            encode_batch()
            drain_shards()
            if written >= args.target_tokens:
                break

    if batch and written < args.target_tokens:
        encode_batch()
        drain_shards()
    if buf and written < args.target_tokens:
        flush(buf)
        written += len(buf)

    print(f"done. total ~{written:,} tokens in {shard_idx} shard(s) at {args.out}")


if __name__ == "__main__":
    import sys

    main()
    # The HF fast tokenizer's Rust threads can segfault during interpreter
    # finalization (PyGILState_Release), returning a non-zero code even though
    # all shards are already on disk. Skip finalization with a clean exit so
    # downstream chaining (&&, auto_run) sees success.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
