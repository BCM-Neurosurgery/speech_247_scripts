"""
Build per-word fastText embeddings and semantic cluster predictions for one patient.

Outputs under vad_new/{patient}:
  embeddings/fasttext_word_embeddings.npy
  embeddings/semantic_cluster_predictions.npy
  embeddings/semantic_cluster_manifest.json
  decoding_inputs/all_words_filtered.csv
"""

from __future__ import annotations

import argparse
import json
import string
import traceback
from pathlib import Path

import dill as pickle
import fasttext
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


TEXT_COL = "word"
PRED_COL = "convo_cluster_pred"
EMBED_NAME = "fasttext_word_embeddings.npy"
PRED_NAME = "semantic_cluster_predictions.npy"
MANIFEST_NAME = "semantic_cluster_manifest.json"
SAFE_DF_NAME = "all_words_filtered.csv"

# Keep the convenience dataframe row-aligned but avoid persisting text-bearing fields.
SAFE_COLUMNS = [
    "patient",
    "interval_id",
    "toc",
    "word_start_s",
    "word_end_s",
    "word_score",
    "segment_idx",
    "segment_start_s",
    "segment_end_s",
    "speaker",
    "interval_utc_start",
    "interval_utc_end",
    "interval_dur_s",
    "br_ts",
    "br_te",
    "utc_word_start",
    "utc_word_end",
    "br_word_start",
    "br_word_end",
    "word_dur_s",
    "word_start_bin",
    "word_end_bin",
    "neural_row_idx",
    "has_neural_features",
    "word_counts_path",
    "word_frs_path",
    "word_durs_path",
]


def sanitize_word(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    return text.rstrip(string.punctuation)


def batched_ranges(n_rows: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, n_rows)) for start in range(0, n_rows, batch_size)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True)
    parser.add_argument("--transcripts-csv", type=Path, required=True)
    parser.add_argument("--fasttext-model", type=Path, required=True)
    parser.add_argument("--classifier-pkl", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--decoding-inputs-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    embeddings_dir = args.embeddings_dir.resolve()
    decoding_inputs_dir = args.decoding_inputs_dir.resolve()
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    decoding_inputs_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = embeddings_dir / EMBED_NAME
    pred_path = embeddings_dir / PRED_NAME
    manifest_path = embeddings_dir / MANIFEST_NAME
    safe_df_path = decoding_inputs_dir / SAFE_DF_NAME
    success_path = embeddings_dir / "_SUCCESS"
    error_path = embeddings_dir / "semantic_cluster_error.txt"

    if success_path.exists() and embedding_path.exists() and pred_path.exists() and manifest_path.exists() and safe_df_path.exists() and not args.force:
        print(f"already done: {args.patient}", flush=True)
        return 0

    if success_path.exists():
        success_path.unlink()
    if error_path.exists():
        error_path.unlink()

    try:
        header_df = pd.read_csv(args.transcripts_csv, nrows=0)
        columns = list(header_df.columns)
        if TEXT_COL not in columns:
            raise ValueError(f"{args.transcripts_csv} is missing required column {TEXT_COL!r}")

        safe_usecols = [c for c in SAFE_COLUMNS if c in columns]
        read_usecols = [TEXT_COL] + safe_usecols

        counts_df = pd.read_csv(args.transcripts_csv, usecols=[TEXT_COL])
        n_rows = len(counts_df)
        del counts_df

        print(f"loading fastText model: {args.fasttext_model}", flush=True)
        ft = fasttext.load_model(str(args.fasttext_model))
        print(f"loading semantic classifier: {args.classifier_pkl}", flush=True)
        with open(args.classifier_pkl, "rb") as f:
            classifier = pickle.load(f)

        sample_embedding = np.asarray(ft.get_word_vector("test"), dtype=np.float32)
        embed_dim = int(sample_embedding.shape[0])
        embedding_mm = open_memmap(embedding_path, mode="w+", dtype=np.float32, shape=(n_rows, embed_dim))
        pred_mm = open_memmap(pred_path, mode="w+", dtype=np.int32, shape=(n_rows,))

        if safe_df_path.exists():
            safe_df_path.unlink()

        total_valid = 0
        total_missing = 0
        batch_ranges = batched_ranges(n_rows, args.batch_size)
        reader = pd.read_csv(args.transcripts_csv, usecols=read_usecols, chunksize=args.batch_size)

        for batch_idx, batch_df in enumerate(reader):
            start = batch_idx * args.batch_size
            stop = start + len(batch_df)
            words = batch_df[TEXT_COL].map(sanitize_word)
            embeddings = np.empty((len(batch_df), embed_dim), dtype=np.float32)
            valid_mask = np.ones(len(batch_df), dtype=bool)

            for i, word in enumerate(words):
                if word:
                    embeddings[i] = np.asarray(ft.get_word_vector(word), dtype=np.float32)
                else:
                    embeddings[i] = np.nan
                    valid_mask[i] = False

            preds = np.full(len(batch_df), -1, dtype=np.int32)
            if valid_mask.any():
                preds[valid_mask] = np.asarray(classifier.predict(embeddings[valid_mask]), dtype=np.int32)

            embedding_mm[start:stop] = embeddings
            pred_mm[start:stop] = preds

            safe_df = batch_df[safe_usecols].copy() if safe_usecols else pd.DataFrame(index=batch_df.index)
            safe_df[PRED_COL] = preds
            safe_df.to_csv(safe_df_path, mode="a", index=False, header=(batch_idx == 0))

            total_valid += int(valid_mask.sum())
            total_missing += int((~valid_mask).sum())
            print(
                f"[{args.patient}] batch {batch_idx + 1}/{len(batch_ranges)} rows {start}:{stop} "
                f"valid={int(valid_mask.sum())} missing={int((~valid_mask).sum())}",
                flush=True,
            )

        embedding_mm.flush()
        pred_mm.flush()

        manifest = {
            "patient": args.patient,
            "transcripts_csv": str(args.transcripts_csv),
            "fasttext_model": str(args.fasttext_model),
            "classifier_pkl": str(args.classifier_pkl),
            "embedding_path": str(embedding_path),
            "prediction_path": str(pred_path),
            "safe_df_path": str(safe_df_path),
            "n_rows": n_rows,
            "embedding_dim": embed_dim,
            "n_valid_words": total_valid,
            "n_missing_words": total_missing,
            "text_column_read": TEXT_COL,
            "text_columns_saved": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        success_path.write_text("ok\n")
        print(f"done: {args.patient}", flush=True)
        return 0

    except Exception:
        tb = traceback.format_exc()
        error_path.write_text(tb)
        print(tb, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
