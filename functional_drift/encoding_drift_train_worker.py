"""
Encoding drift Phase-1 worker — fits a full-day Poisson GLM using per-neuron
alpha hyperparameters extracted from existing nested-CV per-day results.

Usage:
    python encoding_drift_train_worker.py <patient> <vad_root> <out_dir>
        --train-date DATE
        --cv-results-pkl PATH       {patient}_encoding_results_cv.pkl from per-day run
        --word-idx-path PATH        .npy of global word indices for this training day
        [--spike-offset-idx INT]    default 0
        [--gpt2-layer INT]          default -1
        [--n-pca INT]               default 100
        [--max-iter INT]            default 500
        [--embeddings-path PATH]
        [--counts-path PATH]
        [--durations-path PATH]

Outputs in <out_dir>/:
    {patient}_fullday_model.tar       {'W': Tensor(n_pca,K), 'b': Tensor(K,), 'alpha': Tensor(K,)}
    {patient}_fullday_bundle.pkl      preprocessing bundle (mu_raw, sd_raw, pca, mu_pca, sd_pca)
    {patient}_fullday_train_idx.npy   global word indices used (post NaN-cleaning)
    {patient}_fullday_meta.json
    {patient}_TRAIN_SUCCESS  /  {patient}_error.txt
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import dill as pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.optim import LBFGS


# ── Utilities (self-contained) ────────────────────────────────────────────────

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
    """Return boolean mask of rows where not ALL X or ALL Y values are NaN."""
    return ~np.isnan(X).all(axis=1) & ~np.isnan(Y).all(axis=1)


def impute_Y_col_means(Y):
    Y_imp = Y.copy()
    col_means = np.nanmean(Y_imp, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    col_means = np.round(col_means).astype(int)
    rows, cols = np.where(np.isnan(Y_imp))
    Y_imp[rows, cols] = col_means[cols]
    return Y_imp


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


class PoissonRidgeBatched(nn.Module):
    def __init__(self, d, K, alpha):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(d, K))
        self.b = nn.Parameter(torch.zeros(K))
        alpha_t = torch.as_tensor(np.atleast_1d(np.asarray(alpha, np.float32)))
        if alpha_t.ndim == 0:
            alpha_t = alpha_t.expand(K)
        self.register_buffer('alpha', alpha_t)

    def forward(self, X, offset=None):
        eta = X @ self.W + self.b
        if offset is not None:
            eta = eta + offset[:, None]
        return torch.exp(eta.clamp(-20., 20.))

    def loss(self, X, y, offset=None):
        mu = self.forward(X, offset)
        nll = torch.sum(mu - y * torch.log(mu.clamp_min(1e-10)))
        reg = 0.5 * torch.sum(self.alpha * (self.W ** 2).sum(dim=0))
        return nll + reg


def fit_glm(X_t, Y_t, alpha, offset_t=None, max_iter=500, tol=1e-6):
    n, d = X_t.shape
    K = Y_t.shape[1]
    model = PoissonRidgeBatched(d, K, alpha).to(X_t.device)
    opt = LBFGS(model.parameters(), lr=1.0, max_iter=max_iter,
                tolerance_grad=tol, tolerance_change=tol,
                history_size=10, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = model.loss(X_t, Y_t, offset_t)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        W = model.W.detach().clone()
        b = model.b.detach().clone()
    return W, b


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('patient')
    parser.add_argument('vad_root', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--train-date', required=True)
    parser.add_argument('--cv-results-pkl', type=Path, required=True)
    parser.add_argument('--word-idx-path', type=Path, required=True)
    parser.add_argument('--spike-offset-idx', type=int, default=0)
    parser.add_argument('--gpt2-layer', type=int, default=-1)
    parser.add_argument('--n-pca', type=int, default=100)
    parser.add_argument('--max-iter', type=int, default=500)
    parser.add_argument('--embeddings-path', type=Path, default=None)
    parser.add_argument('--counts-path', type=Path, default=None)
    parser.add_argument('--durations-path', type=Path, default=None)
    args = parser.parse_args()

    patient = args.patient
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    success_path = out_dir / f'{patient}_TRAIN_SUCCESS'
    error_path   = out_dir / f'{patient}_error.txt'

    if success_path.exists():
        print(f'already done: {patient} {args.train_date}', flush=True)
        sys.exit(0)

    try:
        # ── Extract best alpha per neuron from existing CV results ────────────
        print(f'loading CV results: {args.cv_results_pkl}', flush=True)
        with open(args.cv_results_pkl, 'rb') as f:
            results_df = pickle.load(f)
        summary = results_df[results_df['is_summary'] == True].sort_values('neuron_idx')
        best_alpha = summary['best_alpha_mean'].values.astype(np.float32)
        print(f'  n_neurons={len(best_alpha)}, alpha=[{best_alpha.min():.3g}, {best_alpha.max():.3g}]',
              flush=True)

        # ── Resolve data paths ────────────────────────────────────────────────
        emb_path, cnt_path, dur_path = resolve_inputs(
            patient, args.vad_root,
            args.embeddings_path, args.counts_path, args.durations_path,
        )
        if any(p is None for p in [emb_path, cnt_path, dur_path]):
            raise FileNotFoundError(f'Missing data inputs for {patient}')

        # ── Load and subset to training day ───────────────────────────────────
        word_idx = np.load(args.word_idx_path).astype(int)
        print(f'  day word_idx: {len(word_idx)} words', flush=True)

        emb_all  = np.load(emb_path, mmap_mode='r')
        cnt_all  = np.load(cnt_path, mmap_mode='r')
        durs_all = np.load(dur_path, mmap_mode='r')

        X_raw = emb_all[:, args.gpt2_layer].copy().astype(np.float32)[word_idx]
        Y_raw = (cnt_all[args.spike_offset_idx] if cnt_all.ndim == 3
                 else cnt_all).astype(np.float32)[word_idx]
        durs  = np.asarray(durs_all)[word_idx]

        # ── NaN cleaning ──────────────────────────────────────────────────────
        mask = nan_clean_mask(X_raw, Y_raw)
        X_raw = X_raw[mask]
        Y_raw = Y_raw[mask]
        durs  = durs[mask]
        Y_raw = impute_Y_col_means(Y_raw)
        log_durs = np.log(np.maximum(durs, 1e-6)).astype(np.float32)

        # Save global indices of words actually used (for Phase-2 to reload training features)
        train_global_idx = word_idx[mask]
        print(f'  after NaN clean: {len(X_raw)} words, {Y_raw.shape[1]} neurons', flush=True)

        # ── Preprocessing bundle (fit on ALL training words) ──────────────────
        bundle = fit_preprocessing_bundle(X_raw, n_pca=args.n_pca)
        X_proc = standardize_apply(
            bundle['pca'].transform(standardize_apply(X_raw, bundle['mu_raw'], bundle['sd_raw'])),
            bundle['mu_pca'], bundle['sd_pca'],
        ).astype(np.float32)

        # ── Fit full-day GLM with fixed alpha ─────────────────────────────────
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'  device: {device}', flush=True)
        X_t   = torch.as_tensor(X_proc, device=device)
        Y_t   = torch.as_tensor(Y_raw.astype(np.float32), device=device)
        dur_t = torch.as_tensor(log_durs, device=device)

        W, b = fit_glm(X_t, Y_t, best_alpha, offset_t=dur_t, max_iter=args.max_iter)

        # ── Save outputs ──────────────────────────────────────────────────────
        torch.save(
            {'W': W.cpu(), 'b': b.cpu(), 'alpha': torch.as_tensor(best_alpha)},
            out_dir / f'{patient}_fullday_model.tar',
        )
        with open(out_dir / f'{patient}_fullday_bundle.pkl', 'wb') as f:
            pickle.dump(bundle, f)
        np.save(out_dir / f'{patient}_fullday_train_idx.npy', train_global_idx)

        meta = dict(
            patient=patient,
            train_date=args.train_date,
            n_words=int(len(X_raw)),
            n_neurons=int(Y_raw.shape[1]),
            n_pca=args.n_pca,
            spike_offset_idx=args.spike_offset_idx,
            gpt2_layer=args.gpt2_layer,
            embeddings_path=str(emb_path),
            counts_path=str(cnt_path),
            durations_path=str(dur_path),
            word_idx_path=str(args.word_idx_path),
        )
        with open(out_dir / f'{patient}_fullday_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        success_path.write_text('ok\n')
        print('done', flush=True)

    except Exception:
        tb = traceback.format_exc()
        error_path.write_text(tb)
        print(f'FAILED:\n{tb}', file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
