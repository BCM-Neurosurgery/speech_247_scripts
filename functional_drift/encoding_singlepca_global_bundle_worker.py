"""
Single-PCA encoding drift Phase-0 worker — fits ONE global preprocessing bundle
(standardize → PCA → standardize) on ALL words from ALL recording days for a
patient, then saves it for use in Phase-1 and Phase-2.

Having a shared bundle means per-day GLM weight vectors (W) live in the same
feature space across all days, making cosine-distance comparisons interpretable.

Usage:
    python encoding_singlepca_global_bundle_worker.py <patient> <vad_root> <out_dir>
        --source-run NAME          e.g. word_level_duration_cv_filtered_speech_per_day
        [--n-pca INT]              default 100
        [--gpt2-layer INT]         default -1
        [--spike-offset-idx INT]   default 0
        [--embeddings-path PATH]
        [--counts-path PATH]
        [--durations-path PATH]

Outputs in <out_dir>/:
    {patient}_global_bundle.pkl          preprocessing bundle dict
    {patient}_global_bundle_meta.json
    {patient}_GLOBAL_BUNDLE_SUCCESS  /  {patient}_global_bundle_error.txt
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import dill as pickle
import numpy as np
from sklearn.decomposition import PCA


# ── Utilities ─────────────────────────────────────────────────────────────────

def first_existing(paths):
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def resolve_inputs(patient, vad_root, emb=None, cnt=None, dur=None):
    r = Path(vad_root) / patient
    emb = first_existing([emb, r / 'embeddings' / f'{patient}_gpt2_embeddings.npy',
                           r / 'all_convo_recording' / 'all_words_filtered_all_layers_gpt2.npy'])
    cnt = first_existing([cnt, r / 'neural_embeddings' / 'word_spike_counts_offsets_all.npy',
                           r / 'all_convo_recording' / 'word_spike_counts_offsets_all.npy'])
    dur = first_existing([dur, r / 'neural_embeddings' / 'word_durs.npy',
                           r / 'all_convo_recording' / 'word_durs.npy'])
    return emb, cnt, dur


def nan_clean_mask(X, Y):
    return ~np.isnan(X).all(axis=1) & ~np.isnan(Y).all(axis=1)


def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def fit_preprocessing_bundle(X_raw, n_pca=100):
    mu_raw, sd_raw = standardize_fit(X_raw)
    X_std = standardize_apply(X_raw, mu_raw, sd_raw)
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X_std)
    mu_pca, sd_pca = standardize_fit(X_pca)
    return dict(mu_raw=mu_raw, sd_raw=sd_raw, pca=pca, mu_pca=mu_pca, sd_pca=sd_pca)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('patient')
    parser.add_argument('vad_root', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--n-pca', type=int, default=100)
    parser.add_argument('--gpt2-layer', type=int, default=-1)
    parser.add_argument('--spike-offset-idx', type=int, default=0)
    parser.add_argument('--embeddings-path', type=Path, default=None)
    parser.add_argument('--counts-path', type=Path, default=None)
    parser.add_argument('--durations-path', type=Path, default=None)
    args = parser.parse_args()

    patient = args.patient
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    success_path = out_dir / f'{patient}_GLOBAL_BUNDLE_SUCCESS'
    error_path   = out_dir / f'{patient}_global_bundle_error.txt'

    if success_path.exists():
        print(f'already done: {patient} global bundle', flush=True)
        sys.exit(0)

    try:
        emb_path, cnt_path, dur_path = resolve_inputs(
            patient, args.vad_root,
            args.embeddings_path, args.counts_path, args.durations_path,
        )
        if any(p is None for p in [emb_path, cnt_path, dur_path]):
            raise FileNotFoundError(f'Missing data inputs for {patient}')

        # ── Discover ALL word-index files from source run ──────────────────────
        enc_base = Path(args.vad_root) / patient / 'encoding' / args.source_run
        if not enc_base.exists():
            raise FileNotFoundError(f'No encoding dir: {enc_base}')

        day_idx_pairs = []
        for d in sorted(enc_base.iterdir()):
            if not d.is_dir():
                continue
            idx_files = list(d.glob(f'{patient}_*_word_idx.npy'))
            if idx_files:
                day_idx_pairs.append((d.name, idx_files[0]))

        if not day_idx_pairs:
            raise ValueError(f'No word-index files found under {enc_base}')

        print(f'Found {len(day_idx_pairs)} days: {[d for d,_ in day_idx_pairs]}', flush=True)

        emb_all = np.load(emb_path, mmap_mode='r')
        cnt_all = np.load(cnt_path, mmap_mode='r')

        # ── Load and concatenate embeddings from all days ─────────────────────
        X_chunks   = []
        n_total    = 0
        days_used  = []

        for date_str, idx_path in day_idx_pairs:
            word_idx = np.load(idx_path).astype(int)
            X_raw    = emb_all[:, args.gpt2_layer].copy().astype(np.float32)[word_idx]
            Y_raw    = (cnt_all[args.spike_offset_idx] if cnt_all.ndim == 3
                        else cnt_all).astype(np.float32)[word_idx]
            mask     = nan_clean_mask(X_raw, Y_raw)
            X_clean  = X_raw[mask]
            if len(X_clean) > 0:
                X_chunks.append(X_clean)
                n_total   += len(X_clean)
                days_used.append(date_str)
                print(f'  [{date_str}] {len(X_clean)} clean words', flush=True)
            else:
                print(f'  [{date_str}] all NaN — skipped', flush=True)

        if not X_chunks:
            raise ValueError(f'No valid words found for {patient}')

        X_all = np.concatenate(X_chunks, axis=0)
        print(f'Total: {n_total} words, shape {X_all.shape}', flush=True)

        # ── Fit global preprocessing bundle ───────────────────────────────────
        print(f'Fitting PCA({args.n_pca}) on all days...', flush=True)
        bundle = fit_preprocessing_bundle(X_all, n_pca=args.n_pca)
        explained = bundle['pca'].explained_variance_ratio_.sum()
        print(f'PCA done — explained variance: {explained:.3f}', flush=True)

        # ── Save ──────────────────────────────────────────────────────────────
        bundle_path = out_dir / f'{patient}_global_bundle.pkl'
        with open(bundle_path, 'wb') as f:
            pickle.dump(bundle, f)

        meta = dict(
            patient=patient,
            source_run=args.source_run,
            n_days=len(days_used),
            date_dirs=days_used,
            n_total_words=int(n_total),
            n_pca=args.n_pca,
            gpt2_layer=args.gpt2_layer,
            spike_offset_idx=args.spike_offset_idx,
            explained_variance=float(explained),
            embeddings_path=str(emb_path),
            counts_path=str(cnt_path),
        )
        with open(out_dir / f'{patient}_global_bundle_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        success_path.write_text('ok\n')
        print(f'done — bundle saved: {bundle_path}', flush=True)

    except Exception:
        tb = traceback.format_exc()
        error_path.write_text(tb)
        print(f'FAILED:\n{tb}', file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
