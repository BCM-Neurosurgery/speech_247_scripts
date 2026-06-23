"""
Decoding drift Phase-2 worker — cross-day inference and permutation testing.

For each test date > train_date, applies the Phase-1 full-day XGBoost model to
test-day words (standardized with training mu/sd), computes actual metrics, then
runs N permutation tests (shuffle training X, recompute mu/sd, retrain, evaluate)
parallelized via joblib within the job.

Usage:
    python decoding_drift_test_worker.py
        --patient PATIENT
        --vad-root PATH
        --train-date DATE
        --resample-idx INT
        --train-out-dir PATH      Phase-1 output directory for this (train_date, resample_idx)
        --source-run NAME         e.g. scat_xgboost_sampled_norm_filtered_per_day
        --word-idx-source-run NAME  encoding source run used to locate test-day word_idx files
                                    (e.g. word_level_duration_cv_filtered_speech_per_day)
        [--out-dir PATH]          default: train-out-dir
        [--n-permutations INT]    default 200
        [--n-jobs INT]            joblib workers (default 4)
        [--valid-cat-min INT]     default 0
        [--valid-cat-max INT]     default 9
        [--min-test-words INT]    default 20
        --cluster-preds-path PATH
        --frs-path PATH
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
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier


# ── Utilities ────────────────────────────────────────────────────────────────

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


def compute_metrics(y_true, y_pred, y_proba, classes):
    acc   = float(accuracy_score(y_true, y_pred))
    bacc  = float(balanced_accuracy_score(y_true, y_pred))
    f1m   = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    try:
        y_bin = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            y_bin = np.hstack([1 - y_bin, y_bin])
        auc = float(roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro'))
    except Exception:
        auc = float('nan')
    return dict(acc=acc, balanced_accuracy=bacc, f1_macro=f1m, auc_macro_ovr=auc)


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


# ── Top-level permutation worker (picklable by joblib) ────────────────────────

def _run_decoding_perm(
    perm_seed: int,
    X_train: np.ndarray,    # (n_train, n_feats) imputed, NaN-free
    y_train: np.ndarray,    # (n_train,) integer labels
    X_test: np.ndarray,     # (n_test, n_feats) imputed, NaN-free
    y_test: np.ndarray,     # (n_test,) integer labels
    best_params: dict,
    classes: np.ndarray,
) -> dict:
    """Shuffle training X, refit standardization + XGBoost, evaluate on test set."""
    rng = np.random.default_rng(perm_seed)
    perm = rng.permutation(len(X_train))
    X_train_perm = X_train[perm]

    mu_perm, sd_perm = standardize_fit(X_train_perm)
    X_tr_std = standardize_apply(X_train_perm, mu_perm, sd_perm)
    X_te_std = standardize_apply(X_test, mu_perm, sd_perm)

    fit_params = dict(best_params)
    fit_params['device']    = 'cpu'
    fit_params['tree_method'] = 'hist'
    fit_params['n_jobs']    = 1  # avoid thread oversubscription inside joblib
    fit_params['random_state'] = int(perm_seed)
    fit_params.pop('predictor', None)
    fit_params.pop('use_label_encoder', None)

    m = XGBClassifier(**fit_params)
    m.fit(X_tr_std, y_train)

    y_pred  = m.predict(X_te_std)
    y_proba = m.predict_proba(X_te_std)

    return compute_metrics(y_test, y_pred, y_proba, classes)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', required=True)
    parser.add_argument('--vad-root', type=Path, required=True)
    parser.add_argument('--train-date', required=True)
    parser.add_argument('--resample-idx', type=int, required=True)
    parser.add_argument('--train-out-dir', type=Path, required=True)
    parser.add_argument('--source-run', required=True,
                        help='Decoding source run for locating per-day results')
    parser.add_argument('--word-idx-source-run', required=True,
                        help='Encoding source run used to locate test-day word_idx .npy files')
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--n-permutations', type=int, default=200)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--valid-cat-min', type=int, default=0)
    parser.add_argument('--valid-cat-max', type=int, default=9)
    parser.add_argument('--min-test-words', type=int, default=20)
    parser.add_argument('--cluster-preds-path', type=Path, required=True)
    parser.add_argument('--frs-path', type=Path, required=True)
    args = parser.parse_args()

    patient   = args.patient
    r         = args.resample_idx
    train_out = args.train_out_dir
    out_dir   = args.out_dir if args.out_dir else train_out
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Load Phase-1 payload ──────────────────────────────────────────────
        model_pkl = train_out / f'{patient}_r{r}_fullday_model.pkl'
        if not model_pkl.exists():
            raise FileNotFoundError(f'Phase-1 model missing: {model_pkl}')

        with open(model_pkl, 'rb') as f:
            payload = pickle.load(f)

        model             = payload['model']         # trained XGBClassifier
        sampled_idx_final = payload['sampled_idx_final']  # indices into day-subsetted frs
        y_train           = payload['y_train']        # labels for training words
        mu_train          = payload['mu']             # standardization params
        sd_train          = payload['sd']
        best_params       = payload['best_params']
        train_classes     = payload['classes']        # unique class labels

        # ── Load raw data ─────────────────────────────────────────────────────
        frs_all  = np.load(args.frs_path, mmap_mode='r')
        cats_all = np.load(args.cluster_preds_path, mmap_mode='r')

        # ── Locate training day word_idx (to resolve frs day-subset indices) ──
        enc_base     = args.vad_root / patient / 'encoding' / args.word_idx_source_run
        train_day_dir = enc_base / args.train_date
        train_idx_files = list(train_day_dir.glob(f'{patient}_*_word_idx.npy')) if train_day_dir.exists() else []
        if not train_idx_files:
            raise FileNotFoundError(f'Training day word_idx not found in {train_day_dir}')
        train_word_idx = np.load(train_idx_files[0]).astype(int)

        # Reconstruct training features for permutation testing
        frs_train_day   = np.asarray(frs_all[train_word_idx])
        X_train_raw     = frs_train_day[sampled_idx_final]
        X_train_raw     = impute_X_all(X_train_raw)

        print(f'training data: {len(X_train_raw)} words, {X_train_raw.shape[1]} neurons', flush=True)

        # ── Discover test dates ────────────────────────────────────────────────
        test_dates = []
        if enc_base.exists():
            for d in sorted(enc_base.iterdir()):
                if d.is_dir() and d.name > args.train_date:
                    idx_files = list(d.glob(f'{patient}_*_word_idx.npy'))
                    if idx_files:
                        test_dates.append((d.name, idx_files[0]))

        print(f'test dates found: {len(test_dates)}', flush=True)

        # ── Iterate over test dates ───────────────────────────────────────────
        for test_date, test_idx_path in test_dates:
            success_path = out_dir / f'r{r}_{test_date}_DRIFT_SUCCESS'
            error_path   = out_dir / f'r{r}_{test_date}_error.txt'
            result_path  = out_dir / f'{patient}_{args.train_date}_{test_date}_r{r}_drift.pkl'

            if success_path.exists():
                print(f'  [r={r} {test_date}] already done, skipping', flush=True)
                continue

            try:
                test_word_idx = np.load(test_idx_path).astype(int)
                frs_test      = np.asarray(frs_all[test_word_idx])
                cats_test     = np.asarray(cats_all[test_word_idx]).astype(int)

                # Filter to valid categories that overlap with training classes
                valid_mask = (
                    (cats_test >= args.valid_cat_min) &
                    (cats_test <= args.valid_cat_max) &
                    np.isin(cats_test, train_classes)
                )
                X_test_raw = frs_test[valid_mask]
                y_test     = cats_test[valid_mask]

                if len(y_test) < args.min_test_words:
                    print(f'  [r={r} {test_date}] only {len(y_test)} valid test words, skipping',
                          flush=True)
                    continue

                X_test_raw = impute_X_all(X_test_raw)

                # Classes available in test set (subset of train_classes)
                test_classes = np.unique(y_test)

                # ── Actual model predictions ──────────────────────────────────
                X_test_std = standardize_apply(X_test_raw, mu_train, sd_train)
                y_pred     = model.predict(X_test_std)
                y_proba    = model.predict_proba(X_test_std)
                model_classes = model.classes_

                actual_metrics = compute_metrics(y_test, y_pred, y_proba, model_classes)

                day_offset = (
                    __import__('datetime').datetime.strptime(test_date, '%Y-%m-%d') -
                    __import__('datetime').datetime.strptime(args.train_date, '%Y-%m-%d')
                ).days

                # ── Permutation test ──────────────────────────────────────────
                print(f'  [r={r} {test_date}] {args.n_permutations} permutations '
                      f'(n_jobs={args.n_jobs})', flush=True)

                perm_seeds = np.arange(args.n_permutations) + 20000 + r * 1000
                perm_results = Parallel(n_jobs=args.n_jobs, backend='threading', verbose=0)(
                    delayed(_run_decoding_perm)(
                        int(seed),
                        X_train_raw, y_train,
                        X_test_raw, y_test,
                        best_params, model_classes,
                    )
                    for seed in perm_seeds
                )

                # Aggregate perm distributions
                perm_acc  = np.array([p['acc'] for p in perm_results])
                perm_bacc = np.array([p['balanced_accuracy'] for p in perm_results])
                perm_f1   = np.array([p['f1_macro'] for p in perm_results])
                perm_auc  = np.array([p['auc_macro_ovr'] for p in perm_results])

                n_perm = args.n_permutations
                result = dict(
                    patient=patient,
                    train_date=args.train_date,
                    test_date=test_date,
                    resample_idx=r,
                    day_offset=day_offset,
                    n_train_words=int(len(X_train_raw)),
                    n_test_words=int(len(y_test)),
                    n_classes_train=int(len(train_classes)),
                    n_classes_test=int(len(test_classes)),
                    n_permutations=n_perm,
                    # Actual metrics
                    acc=actual_metrics['acc'],
                    balanced_accuracy=actual_metrics['balanced_accuracy'],
                    f1_macro=actual_metrics['f1_macro'],
                    auc_macro_ovr=actual_metrics['auc_macro_ovr'],
                    # Perm distributions
                    perm_acc=perm_acc,
                    perm_balanced_accuracy=perm_bacc,
                    perm_f1_macro=perm_f1,
                    perm_auc=perm_auc,
                    # p-values (one-sided: fraction of perms >= actual)
                    p_val_acc=(np.sum(perm_acc  >= actual_metrics['acc'])  + 1) / (n_perm + 1),
                    p_val_bacc=(np.sum(perm_bacc >= actual_metrics['balanced_accuracy']) + 1) / (n_perm + 1),
                    p_val_f1=(np.sum(perm_f1  >= actual_metrics['f1_macro']) + 1) / (n_perm + 1),
                    p_val_auc=(np.sum(perm_auc[~np.isnan(perm_auc)] >= actual_metrics['auc_macro_ovr'])
                               + 1) / (n_perm + 1) if not np.isnan(actual_metrics['auc_macro_ovr']) else float('nan'),
                )

                with open(result_path, 'wb') as f:
                    pickle.dump(result, f)
                success_path.write_text('ok\n')
                print(f'  [r={r} {test_date}] done — {result_path.name}', flush=True)

            except Exception:
                tb = traceback.format_exc()
                error_path.write_text(tb)
                print(f'  [r={r} {test_date}] ERROR:\n{tb}', file=sys.stderr, flush=True)

        print('all test dates processed', flush=True)

    except Exception:
        tb = traceback.format_exc()
        print(f'FATAL:\n{tb}', file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
