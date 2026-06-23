"""
SBERT cosine-similarity worker for convo accuracy evaluation.

Reads a matches CSV (output of the matching step), encodes ref_text and hyp_text
for matched rows using a SentenceTransformer, and writes cosine_sim back to an
output CSV.

Usage:
    python sbert_worker.py --input_csv /path/to/matches.csv \
                           --output_csv /path/to/matches_sbert.csv

Optional:
    --model_name   SentenceTransformer model (default: all-mpnet-base-v2)
    --hf_home      HuggingFace cache dir (default: /scratch/tahaismail424/hf)
    --batch_size   Encoding batch size (default: 64)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--model_name", default="all-mpnet-base-v2")
    p.add_argument("--hf_home", default="/scratch/tahaismail424/hf")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["HF_HOME"] = args.hf_home
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    model = SentenceTransformer(args.model_name).to(device)

    matches = pd.read_csv(args.input_csv, index_col=0)
    matched_mask = matches["matched"].astype(bool)
    matched = matches[matched_mask].copy()

    matches["cosine_sim"] = np.nan

    if len(matched) == 0:
        print("no matched rows — writing output with all NaN cosine_sim")
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        matches.to_csv(args.output_csv)
        return

    print(f"encoding {len(matched)} matched sentence pairs …", flush=True)

    ref_embeddings = model.encode(
        matched["ref_text"].fillna("").tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device,
    )
    hyp_embeddings = model.encode(
        matched["hyp_text"].fillna("").tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device,
    )

    sims = (ref_embeddings * hyp_embeddings).sum(axis=1)
    matches.loc[matched.index, "cosine_sim"] = sims

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output_csv)
    print(f"saved {len(matches)} rows → {args.output_csv}")
    print(f"cosine_sim: mean={sims.mean():.3f}  median={np.median(sims):.3f}")


if __name__ == "__main__":
    main()
