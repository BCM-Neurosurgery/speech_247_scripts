"""
GPT-2-large word-level embedding worker.

Processes all words for one interval using KV-cached autoregressive forward
passes.  Each new word is forwarded through the model with the KV cache from
all preceding words already accumulated — O(1) GPU work per new token instead
of O(context_len²) as in the original per-word-from-scratch approach.

Usage:
    python gpt2_embedding_worker.py <transcripts_csv> <interval_id> <out_dir>

Output (written to <out_dir>/):
    {interval_id}_gpt2_embeddings.npy   float16, shape (n_words, 49, 1280)
    {interval_id}_meta.json             n_words, csv_row_indices
    {interval_id}_SUCCESS
    {interval_id}_error.txt             written only on failure
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Model, GPT2Tokenizer


MAX_KV_TOKENS = 1020   # stay below GPT-2's 1024-token limit


def build_model(device):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2Model.from_pretrained("gpt2-large", dtype=torch.float16)
    model = model.to(device).eval()
    return tokenizer, model


def embed_interval(words: list[str], tokenizer, model, device) -> np.ndarray:
    """
    Returns float16 array of shape (n_words, n_layers, hidden_dim).

    Uses GPT-2's KV cache so only the new word tokens are forwarded each step;
    attention over the growing context is handled by the cached key/values.
    When the cache would exceed MAX_KV_TOKENS the oldest tokens are evicted.
    """
    past_key_values = None
    embeddings = []

    for i, word in enumerate(words):
        # GPT-2 BPE: words after the first get a leading space so the tokenizer
        # produces the same token IDs as when the word appears mid-sentence.
        text_chunk = word if i == 0 else " " + word
        tok = tokenizer(
            text_chunk, return_tensors="pt", add_special_tokens=False
        ).to(device)
        n_new = tok.input_ids.shape[1]

        # Evict oldest KV entries if we'd exceed the position-embedding limit.
        if past_key_values is not None:
            kv_len = past_key_values[0][0].shape[2]
            excess = kv_len + n_new - MAX_KV_TOKENS
            if excess > 0:
                past_key_values = tuple(
                    (k[:, :, excess:, :], v[:, :, excess:, :])
                    for k, v in past_key_values
                )

        with torch.no_grad():
            out = model(
                **tok,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )

        past_key_values = out.past_key_values

        # out.hidden_states: tuple of (n_layers+1) tensors, each (1, n_new, H)
        # Stack → (n_layers, n_new, H), average over word tokens → (n_layers, H)
        hidden = torch.stack([h.squeeze(0) for h in out.hidden_states])  # (L, n_new, H)
        word_emb = hidden.mean(dim=1).cpu().to(torch.float16).numpy()     # (L, H)
        embeddings.append(word_emb)

    return np.stack(embeddings)  # (n_words, n_layers, hidden_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transcripts_csv", type=Path)
    parser.add_argument("interval_id", type=str)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    success_path = out_dir / f"{args.interval_id}_SUCCESS"
    error_path = out_dir / f"{args.interval_id}_error.txt"

    if success_path.exists():
        print(f"already done: {args.interval_id}", flush=True)
        sys.exit(0)

    try:
        print(f"loading transcripts for {args.interval_id}", flush=True)
        df = pd.read_csv(args.transcripts_csv)
        interval_df = df[df["interval_id"] == args.interval_id].copy()

        if interval_df.empty:
            raise ValueError(f"No rows found for interval_id={args.interval_id}")

        words = interval_df["word"].astype(str).tolist()
        row_indices = interval_df.index.tolist()
        print(f"  {len(words)} words", flush=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  device: {device}", flush=True)

        print("  loading gpt2-large", flush=True)
        tokenizer, model = build_model(device)

        print("  running forward passes", flush=True)
        embeddings = embed_interval(words, tokenizer, model, device)
        print(f"  embeddings shape: {embeddings.shape}", flush=True)

        emb_path = out_dir / f"{args.interval_id}_gpt2_embeddings.npy"
        np.save(str(emb_path), embeddings)

        meta = {
            "interval_id": args.interval_id,
            "n_words": len(words),
            "csv_row_indices": row_indices,
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
        }
        with open(out_dir / f"{args.interval_id}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        success_path.write_text("ok\n")
        print("done", flush=True)

    except Exception:
        tb = traceback.format_exc()
        error_path.write_text(tb)
        print(f"FAILED:\n{tb}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
