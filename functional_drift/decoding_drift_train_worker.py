"""
Decoding drift Phase-1 worker — fits a full-day XGBoost semantic-category classifier
using the same class-balanced resample (same seed) as the existing per-day results,
but trained on ALL class-balanced words (no train/test holdout) with fixed
hyperparameters from the existing best_params JSON.

Usage:
    python decoding_drift_train_worker.py
        --patient PATIENT
        --vad-root PATH
        --out-dir PATH
        --train-date DATE
        --resample-idx INT
        --best-params-json PATH    best_params_resample_{r}.json from per-day run
        --word-idx-path PATH       global word indices for this training day
        --cluster-preds-path PATH
        --frs-path PATH
        [--seed-stride INT]        default 42 (seed = resample_idx * seed_stride)
        [--n-jobs INT]             XGBoost n_jobs (default 4)

Outputs in <out_dir>/:
    {patient}_r{r}_fullday_model.pkl      (model, sampled_idx_final, y_train, mu, sd)
    {patient}_r{r}_meta.json
    r{r}_TRAIN_SUCCESS  /  r{r}_error.txt
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import dill as pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


# ── Utilities (mirrors scat_classifier_worker) ────────────────────────────────

def class_balance_words(categories: pd.Series, rng: np.random.Generator):
    cat = categories[(categories >= 0) & (categories != 10)]
    counts = cat.value_counts()
    median_count = int(np.round(np.median(counts.values)))
    to_keep = []
    for cluster_id, _ in counts.items():
        cluster_idx = cat[cat == cluster_id].index.to_numpy()
        if len(cluster_idx) > median_count:
            picked = rng.choice(cluster_idx, size=median_count, replace=False)
            to_keep.extend(picked.tolist())
        else:
            to_keep.extend(cluster_idx.tolist())
    to_keep = np.asarray(to_keep, dtype=int)
    return cat.loc[to_keep].to_numpy(), to_keep


def impute_X_all(X: np.ndarray) -> np.ndarray:
    X_imp = X.copy()
    channel_means = np.nanmean(X_imp, axis=0)
    channel_means = np.where(np.isnan(channel_means), 0.0, channel_means)
    inds = np.where(np.isnan(X_imp))
    X_imp[inds] = np.take(channel_means, inds[1])
    return X_imp


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def standardize_apply(X: np.ndarray, mu, sd) -> np.ndarray:
    return (X - mu) / sd


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', required=True)
    parser.add_argument('--vad-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--train-date', required=True)
    parser.add_argument('--resample-idx', type=int, required=True)
    parser.add_argument('--best-params-json', type=Path, required=True)
    parser.add_argument('--word-idx-path', type=Path, required=True)
    parser.add_argument('--cluster-preds-path', type=Path, required=True)
    parser.add_argument('--frs-path', type=Path, required=True)
    parser.add_argument('--seed-stride', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=4)
    args = parser.parse_args()

    patient    = args.patient
    r          = args.resample_idx
    out_dir    = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    success_path = out_dir / f'r{r}_TRAIN_SUCCESS'
    error_path   = out_dir / f'r{r}_error.txt'

    if success_path.exists():
        print(f'already done: {patient} r={r} {args.train_date}', flush=True)
        sys.exit(0)

    try:
        seed = r * args.seed_stride
        rng  = np.random.default_rng(seed)

        # ── Load best hyperparameters from existing per-day result ────────────
        best_params = json.loads(args.best_params_json.read_text())
        print(f'loaded params from {args.best_params_json.name}', flush=True)

        # ── Load data ─────────────────────────────────────────────────────────
        word_idx = np.load(args.word_idx_path).astype(int)
        frs      = np.load(args.frs_path, mmap_mode='r')
        categories_all = np.load(args.cluster_preds_path, mmap_mode='r')

        frs_day  = np.asarray(frs[word_idx])
        cats_day = pd.Series(np.asarray(categories_all[word_idx]).astype(int))
        print(f'  day words: {len(word_idx)}, valid cats: {int((cats_day[(cats_day >= 0) & (cats_day != 10)]).count())}',
              flush=True)

        # ── Class-balanced sampling (same seed as per-day analysis) ───────────
        y_sampled, sampled_idx_raw = class_balance_words(cats_day, rng)

        frs_keep        = frs_day[sampled_idx_raw]
        mask_keep       = ~np.isnan(frs_keep).all(axis=1)
        X_all           = frs_keep[mask_keep]
        y_all           = y_sampled[mask_keep]
        sampled_idx_final = sampled_idx_raw[mask_keep]
        X_all           = impute_X_all(X_all)

        # Sanity checks
        unique_classes, class_counts = np.unique(y_all, return_counts=True)
        n_classes = len(unique_classes)
        if n_classes < 2:
            raise ValueError(f'Only {n_classes} class after balancing — skipping.')
        print(f'  balanced: {len(X_all)} words, {n_classes} classes', flush=True)

        # ── Standardize on ALL training words (no holdout) ───────────────────
        mu, sd = standardize_fit(X_all)
        X_std  = standardize_apply(X_all, mu, sd)

        # ── Fit XGBoost with fixed params ─────────────────────────────────────
        fit_params = dict(best_params)
        fit_params['device']    = 'cuda' if __import__('subprocess').run(
            ['nvidia-smi'], capture_output=True).returncode == 0 else 'cpu'
        fit_params['tree_method'] = 'hist'
        fit_params['n_jobs']    = args.n_jobs
        fit_params['random_state'] = seed
        fit_params.pop('predictor', None)
        fit_params.pop('use_label_encoder', None)

        model = XGBClassifier(**fit_params)
        model.fit(X_std, y_all)
        print(f'  model trained', flush=True)

        # ── Save outputs ──────────────────────────────────────────────────────
        payload = dict(
            model=model,
            sampled_idx_final=sampled_idx_final,   # indices into day-subsetted frs
            y_train=y_all,                          # class labels for those words
            mu=mu,                                  # standardization params
            sd=sd,
            seed=seed,
            best_params=best_params,
            classes=unique_classes,
        )
        with open(out_dir / f'{patient}_r{r}_fullday_model.pkl', 'wb') as f:
            pickle.dump(payload, f)

        meta = dict(
            patient=patient,
            train_date=args.train_date,
            resample_idx=r,
            seed=seed,
            n_words_balanced=int(len(X_all)),
            n_classes=n_classes,
            frs_path=str(args.frs_path),
            cluster_preds_path=str(args.cluster_preds_path),
            word_idx_path=str(args.word_idx_path),
        )
        with open(out_dir / f'{patient}_r{r}_meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=json_default)

        success_path.write_text('ok\n')
        print('done', flush=True)

    except Exception:
        tb = traceback.format_exc()
        error_path.write_text(tb)
        print(f'FAILED:\n{tb}', file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
