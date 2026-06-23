"""
Single-PCA encoding drift Phase-2 worker — cross-day inference and permutation
testing using the shared global PCA bundle.

Key difference from encoding_drift_test_worker.py:
  - The global bundle is FIXED across all permutations (it was fit on all data,
    not just the training day), so permutations only shuffle training X and
    re-fit the GLM — the feature space never changes between permutations or
    between actual and null models.

For each test date > train_date:
  1. Apply global bundle to test-day features → get predictions
  2. Run N permutations: shuffle training X → apply global bundle → refit GLM
     → evaluate on test set (same global bundle)
  3. Save per-neuron metrics + permutation p-values

Usage:
    python encoding_singlepca_test_worker.py <patient> <vad_root>
        --train-date DATE
        --train-out-dir DIR        Phase-1 output directory
        --global-bundle-pkl PATH   Phase-0 output bundle
        --source-run NAME
        [--out-dir DIR]            default: train-out-dir
        [--n-permutations INT]     default 200
        [--n-jobs INT]             joblib workers (default 4)
        [--spike-offset-idx INT]   default 0
        [--gpt2-layer INT]         default -1
        [--embeddings-path PATH]
        [--counts-path PATH]
        [--durations-path PATH]

Outputs per test date in <out_dir>/:
    {patient}_{train_date}_{test_date}_drift_results.pkl
    {test_date}_DRIFT_SUCCESS  /  {test_date}_error.txt
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
from joblib import Parallel, delayed
from scipy.special import gammaln
from scipy.stats import pearsonr, spearmanr
from torch.optim import LBFGS


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


def impute_Y_col_means(Y):
    Y_imp = Y.copy()
    col_means = np.nanmean(Y_imp, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    col_means = np.round(col_means).astype(int)
    rows, cols = np.where(np.isnan(Y_imp))
    Y_imp[rows, cols] = col_means[cols]
    return Y_imp


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def apply_bundle(X_raw, bundle):
    X_std = standardize_apply(X_raw, bundle['mu_raw'], bundle['sd_raw'])
    X_pca = bundle['pca'].transform(X_std)
    return standardize_apply(X_pca, bundle['mu_pca'], bundle['sd_pca'])


def poisson_ll(y_true, mu_pred):
    mu = np.clip(mu_pred, 1e-10, None)
    return (y_true * np.log(mu) - mu - gammaln(y_true + 1)).sum(axis=0)  # (K,)


def pseudo_r2(ll_model, ll_null):
    return 1.0 - ll_model / ll_null


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


def fit_glm_cpu(X_proc, Y_np, alpha, log_durs_np, max_iter=200, tol=1e-6):
    """CPU-only GLM fit for use inside joblib workers."""
    X_t   = torch.as_tensor(X_proc.astype(np.float32))
    Y_t   = torch.as_tensor(Y_np.astype(np.float32))
    dur_t = torch.as_tensor(log_durs_np.astype(np.float32))
    K     = Y_np.shape[1]
    model = PoissonRidgeBatched(X_t.shape[1], K, alpha)
    opt   = LBFGS(model.parameters(), lr=1.0, max_iter=max_iter,
                  tolerance_grad=tol, tolerance_change=tol,
                  history_size=10, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(set_to_none=True)
        l = model.loss(X_t, Y_t, dur_t)
        l.backward()
        return l

    opt.step(closure)
    with torch.no_grad():
        W = model.W.detach().numpy()
        b = model.b.detach().numpy()
    return W, b


# ── Permutation worker ─────────────────────────────────────────────────────────

def _run_encoding_perm(
    perm_seed,
    X_train_raw,   # raw (pre-bundle) training features
    Y_train,
    log_durs_train,
    X_test_proc,   # already bundled test features (fixed — bundle is global)
    Y_test,
    log_durs_test,
    alpha,
    bundle,
    max_iter,
):
    """
    Permutation: shuffle training X in raw space, apply fixed global bundle,
    refit GLM only. Test set is already processed with the global bundle.
    The feature space never changes — only the GLM weights are randomised.
    """
    rng = np.random.RandomState(perm_seed)
    perm = rng.permutation(len(X_train_raw))
    X_train_shuf = X_train_raw[perm]

    X_tr_proc = apply_bundle(X_train_shuf, bundle).astype(np.float32)

    W_perm, b_perm = fit_glm_cpu(X_tr_proc, Y_train, alpha, log_durs_train,
                                  max_iter=max_iter)

    eta    = X_test_proc @ W_perm + b_perm + log_durs_test[:, None]
    mu_perm = np.exp(np.clip(eta, -30, 30)).astype(np.float64)
    return poisson_ll(Y_test.astype(np.float64), mu_perm)  # (K,)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('patient')
    parser.add_argument('vad_root', type=Path)
    parser.add_argument('--train-date', required=True)
    parser.add_argument('--train-out-dir', type=Path, required=True)
    parser.add_argument('--global-bundle-pkl', type=Path, required=True)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--n-permutations', type=int, default=200)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--spike-offset-idx', type=int, default=0)
    parser.add_argument('--gpt2-layer', type=int, default=-1)
    parser.add_argument('--embeddings-path', type=Path, default=None)
    parser.add_argument('--counts-path', type=Path, default=None)
    parser.add_argument('--durations-path', type=Path, default=None)
    args = parser.parse_args()

    patient   = args.patient
    train_out = args.train_out_dir
    out_dir   = args.out_dir if args.out_dir else train_out
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Load Phase-0 global bundle ────────────────────────────────────────
        print(f'loading global bundle: {args.global_bundle_pkl}', flush=True)
        with open(args.global_bundle_pkl, 'rb') as f:
            bundle = pickle.load(f)
        n_pca = bundle['pca'].n_components_

        # ── Load Phase-1 outputs ──────────────────────────────────────────────
        model_tar      = train_out / f'{patient}_fullday_model.tar'
        train_idx_npy  = train_out / f'{patient}_fullday_train_idx.npy'
        meta_json      = train_out / f'{patient}_fullday_meta.json'

        for fp in [model_tar, train_idx_npy, meta_json]:
            if not fp.exists():
                raise FileNotFoundError(f'Phase-1 output missing: {fp}')

        state  = torch.load(model_tar, map_location='cpu', weights_only=False)
        W_cpu  = state['W'].numpy()      # (n_pca, K)
        b_cpu  = state['b'].numpy()      # (K,)
        alpha  = state['alpha'].numpy()  # (K,)

        train_global_idx = np.load(train_idx_npy).astype(int)

        with open(meta_json) as f:
            meta = json.load(f)

        # ── Resolve data paths ────────────────────────────────────────────────
        emb_path, cnt_path, dur_path = resolve_inputs(
            patient, args.vad_root,
            args.embeddings_path, args.counts_path, args.durations_path,
        )
        if any(p is None for p in [emb_path, cnt_path, dur_path]):
            raise FileNotFoundError(f'Missing data inputs for {patient}')

        emb_all  = np.load(emb_path, mmap_mode='r')
        cnt_all  = np.load(cnt_path, mmap_mode='r')
        durs_all = np.load(dur_path, mmap_mode='r')

        # ── Load training features (raw, for permutation shuffling) ───────────
        X_train_raw  = emb_all[:, args.gpt2_layer].copy().astype(np.float32)[train_global_idx]
        Y_train      = (cnt_all[args.spike_offset_idx] if cnt_all.ndim == 3
                        else cnt_all).astype(np.float32)[train_global_idx]
        Y_train      = impute_Y_col_means(Y_train)
        durs_train   = np.asarray(durs_all)[train_global_idx]
        log_durs_train = np.log(np.maximum(durs_train, 1e-6)).astype(np.float32)

        K = Y_train.shape[1]
        print(f'loaded training data: {len(X_train_raw)} words, {K} neurons', flush=True)

        # ── Discover test dates ───────────────────────────────────────────────
        enc_base   = Path(args.vad_root) / patient / 'encoding' / args.source_run
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
            success_path = out_dir / f'{test_date}_DRIFT_SUCCESS'
            error_path   = out_dir / f'{test_date}_error.txt'
            result_path  = out_dir / f'{patient}_{args.train_date}_{test_date}_drift_results.pkl'

            if success_path.exists():
                print(f'  [{test_date}] already done, skipping', flush=True)
                continue

            try:
                test_idx    = np.load(test_idx_path).astype(int)
                X_test_raw  = emb_all[:, args.gpt2_layer].copy().astype(np.float32)[test_idx]
                Y_test      = (cnt_all[args.spike_offset_idx] if cnt_all.ndim == 3
                               else cnt_all).astype(np.float32)[test_idx]
                durs_test   = np.asarray(durs_all)[test_idx]

                mask_te      = nan_clean_mask(X_test_raw, Y_test)
                X_test_raw   = X_test_raw[mask_te]
                Y_test       = Y_test[mask_te]
                durs_test    = durs_test[mask_te]
                Y_test       = impute_Y_col_means(Y_test)
                log_durs_test = np.log(np.maximum(durs_test, 1e-6)).astype(np.float32)

                if len(X_test_raw) == 0:
                    print(f'  [{test_date}] all NaN after cleaning, skipping', flush=True)
                    continue

                # ── Actual model predictions using global bundle ───────────────
                X_test_proc = apply_bundle(X_test_raw, bundle).astype(np.float32)
                eta     = X_test_proc @ W_cpu + b_cpu + log_durs_test[:, None]
                mu_hat  = np.exp(np.clip(eta, -30, 30)).astype(np.float64)

                Y_test_d = Y_test.astype(np.float64)
                ll_real  = poisson_ll(Y_test_d, mu_hat)  # (K,)

                avg_rate = Y_train.sum(axis=0) / np.exp(log_durs_train).sum()
                mu_null  = (avg_rate * np.exp(log_durs_test)[:, None]).astype(np.float64)
                ll_null  = poisson_ll(Y_test_d, mu_null)  # (K,)

                pr2  = pseudo_r2(ll_real, ll_null)
                pear = np.array([
                    pearsonr(Y_test[:, k], mu_hat[:, k])[0] if np.std(mu_hat[:, k]) > 0 else np.nan
                    for k in range(K)
                ])
                spear = np.array([
                    spearmanr(Y_test[:, k], mu_hat[:, k])[0] if np.std(mu_hat[:, k]) > 0 else np.nan
                    for k in range(K)
                ])

                day_offset = (
                    __import__('datetime').datetime.strptime(test_date, '%Y-%m-%d') -
                    __import__('datetime').datetime.strptime(args.train_date, '%Y-%m-%d')
                ).days

                # ── Permutation test (global bundle is fixed, only GLM shuffled) ─
                print(f'  [{test_date}] running {args.n_permutations} permutations '
                      f'(n_jobs={args.n_jobs})', flush=True)

                perm_seeds = np.arange(args.n_permutations) + 10000
                ll_perms = Parallel(n_jobs=args.n_jobs, backend='threading', verbose=0)(
                    delayed(_run_encoding_perm)(
                        int(seed),
                        X_train_raw, Y_train, log_durs_train,
                        X_test_proc,   # already bundled — never changes between perms
                        Y_test, log_durs_test,
                        alpha, bundle, 200,
                    )
                    for seed in perm_seeds
                )
                ll_perms = np.array(ll_perms)  # (n_perm, K)

                p_vals = (np.sum(ll_perms >= ll_real, axis=0) + 1) / (args.n_permutations + 1)

                result = dict(
                    patient=patient,
                    train_date=args.train_date,
                    test_date=test_date,
                    day_offset=day_offset,
                    n_train_words=int(len(X_train_raw)),
                    n_test_words=int(len(X_test_raw)),
                    n_neurons=K,
                    n_permutations=args.n_permutations,
                    ll_real=ll_real,
                    ll_null=ll_null,
                    pseudo_r2=pr2,
                    pearson_corr=pear,
                    spearman_corr=spear,
                    ll_perms=ll_perms,
                    p_val_ll=p_vals,
                    ll_perm_mean=ll_perms.mean(axis=0),
                )

                with open(result_path, 'wb') as f:
                    pickle.dump(result, f)
                success_path.write_text('ok\n')
                print(f'  [{test_date}] done — saved {result_path.name}', flush=True)

            except Exception:
                tb = traceback.format_exc()
                error_path.write_text(tb)
                print(f'  [{test_date}] ERROR:\n{tb}', file=sys.stderr, flush=True)

        print('all test dates processed', flush=True)

    except Exception:
        tb = traceback.format_exc()
        print(f'FATAL:\n{tb}', file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
