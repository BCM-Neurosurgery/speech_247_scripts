"""
Poisson GLM encoding worker — one patient per SLURM job.

Usage:
    python poisson_glm_worker.py <patient> <vad_root> <out_dir>
        [--spike-offset-idx IDX]   index into word_spike_counts_offsets_all.npy (default 8)
        [--gpt2-layer IDX]         GPT-2 layer to use, -1 = last (default -1)
        [--n-pca N]                PCA components (default 100)
        [--outer-splits K]         outer CV folds (default 5)
        [--inner-splits K]         inner CV folds for alpha tuning (default 5)
        [--n-alphas N]             number of alpha candidates (default 30)
        [--alpha-low F]            log10 lower bound for alpha grid (default -3)
        [--alpha-high F]           log10 upper bound for alpha grid (default 3)
        [--n-shuffles N]           permutation baseline shuffles (default 0)
        [--embeddings-path PATH]   optional explicit embeddings path override
        [--counts-path PATH]       optional explicit spike-count path override
        [--durations-path PATH]    optional explicit duration path override

The preferred layout is:
    vad_new/{patient}/embeddings/{patient}_gpt2_embeddings.npy
    vad_new/{patient}/neural_embeddings/word_spike_counts_offsets_all.npy
    vad_new/{patient}/neural_embeddings/word_durs.npy

Legacy fallback is still supported for older outputs under:
    vad_new/{patient}/all_convo_recording/

Outputs (written to <out_dir>/):
    {patient}_encoding_results_cv.pkl
    {patient}_encoding_models_cv.tar
    {patient}_meta.json
    {patient}_SUCCESS
    {patient}_error.txt
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import LBFGS, Adam
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy.special import gammaln
from scipy.stats import pearsonr, spearmanr
import dill as pickle


def first_existing(paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


def resolve_patient_inputs(patient, vad_root, embeddings_path=None, counts_path=None, durations_path=None):
    patient_root = Path(vad_root) / patient

    embeddings_path = first_existing([
        embeddings_path,
        patient_root / "embeddings" / f"{patient}_gpt2_embeddings.npy",
        patient_root / "all_convo_recording" / "all_words_filtered_all_layers_gpt2.npy",
    ])
    counts_path = first_existing([
        counts_path,
        patient_root / "neural_embeddings" / "word_spike_counts_offsets_all.npy",
        patient_root / "all_convo_recording" / "word_spike_counts_offsets_all.npy",
    ])
    durations_path = first_existing([
        durations_path,
        patient_root / "neural_embeddings" / "word_durs.npy",
        patient_root / "all_convo_recording" / "word_durs.npy",
    ])

    missing = []
    if embeddings_path is None:
        missing.append("embeddings")
    if counts_path is None:
        missing.append("spike_counts")
    if durations_path is None:
        missing.append("durations")
    if missing:
        raise FileNotFoundError(
            f"Missing inputs for {patient}: {', '.join(missing)}. "
            "Expected them under vad_new/{patient}/embeddings and vad_new/{patient}/neural_embeddings "
            "or passed explicitly via --embeddings-path/--counts-path/--durations-path."
        )

    return {
        "patient_root": patient_root,
        "embeddings_path": embeddings_path,
        "counts_path": counts_path,
        "durations_path": durations_path,
    }


# ── Data utilities ────────────────────────────────────────────────────────────

def nan_clean_XY(X, Y, durations=None):
    mask = ~np.isnan(X).all(axis=1) & ~np.isnan(Y).all(axis=1)
    if durations is not None:
        return X[mask], Y[mask], durations[mask]
    return X[mask], Y[mask]


def impute_Y_col_means(Y):
    """Replace NaNs in Y with per-channel nanmean (rounded to int for spike counts)."""
    Y_imp = Y.copy()
    col_means = np.nanmean(Y_imp, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    col_means = np.round(col_means).astype(int)
    rows, cols = np.where(np.isnan(Y_imp))
    Y_imp[rows, cols] = col_means[cols]
    return Y_imp


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return mu, sd


def _standardize_apply(X, mu, sd):
    return (X - mu) / sd


def _prep_X_with_pca(X_tr_raw, X_te_raw, n_components=100):
    """Fit standardize → PCA → standardize on X_tr only, apply to both."""
    mu_raw, sd_raw = _standardize_fit(X_tr_raw)
    Xtr_std = _standardize_apply(X_tr_raw, mu_raw, sd_raw)
    Xte_std = _standardize_apply(X_te_raw, mu_raw, sd_raw)

    pca = PCA(n_components=n_components)
    Xtr_pca = pca.fit_transform(Xtr_std)
    Xte_pca = pca.transform(Xte_std)

    mu_pca, sd_pca = _standardize_fit(Xtr_pca)
    Xtr_s = _standardize_apply(Xtr_pca, mu_pca, sd_pca)
    Xte_s = _standardize_apply(Xte_pca, mu_pca, sd_pca)

    bundle = dict(mu_raw=mu_raw, sd_raw=sd_raw, pca=pca, mu_pca=mu_pca, sd_pca=sd_pca)
    return Xtr_s, Xte_s, bundle


def backtransform_coef(w_std, b_std, mu_pca, sd_pca):
    w_pca = w_std / sd_pca[:, np.newaxis]
    intercept_pca = b_std - np.dot(mu_pca, w_pca)
    return w_pca, intercept_pca


# ── Metrics ───────────────────────────────────────────────────────────────────

def poisson_ll_per_neuron(y_true, mu_pred):
    mu = np.clip(mu_pred, 1e-10, None)
    return (y_true * np.log(mu) - mu - gammaln(y_true + 1)).sum(axis=0)  # (K,)


def pseudo_r2(ll_model, ll_null):
    return 1.0 - ll_model / ll_null


def pearson_per_neuron(y, yhat):
    return np.array([
        pearsonr(y[:, k], yhat[:, k])[0] if np.std(yhat[:, k]) > 0 else np.nan
        for k in range(y.shape[1])
    ])


def spearman_per_neuron(y, yhat):
    return np.array([
        spearmanr(y[:, k], yhat[:, k])[0] if np.std(yhat[:, k]) > 0 else np.nan
        for k in range(y.shape[1])
    ])


# ── Model ─────────────────────────────────────────────────────────────────────

def _to_device(x, device):
    if x is None:
        return None
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


class PoissonRidgeBatched(nn.Module):
    """Log-linear Poisson model with per-neuron ridge penalty, fit over all K neurons at once."""

    def __init__(self, d, K, alpha=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(d, K))
        self.b = nn.Parameter(torch.zeros(K))
        alpha_t = torch.as_tensor(alpha, dtype=torch.float32)
        if alpha_t.ndim == 0:
            alpha_t = alpha_t.expand(K)
        self.register_buffer("alpha", alpha_t)

    def set_params(self, w0=None, b0=None):
        with torch.no_grad():
            if w0 is not None:
                self.W.copy_(w0)
            if b0 is not None:
                self.b.copy_(b0)

    def forward(self, X, offset=None):
        eta = X @ self.W + self.b
        if offset is not None:
            eta = eta + offset[:, None]
        # clamp eta before exp to prevent float32 overflow → nan/inf gradient → LBFGS divergence.
        # exp(20) ≈ 5e8 spikes, well above any physiological rate, so this never clips real data.
        return torch.exp(eta.clamp(-20.0, 20.0))

    def loss(self, X, y, offset=None):
        mu = self.forward(X, offset)
        nll = torch.sum(mu - y * torch.log(mu.clamp_min(1e-10)))
        reg = 0.5 * torch.sum(self.alpha * (self.W ** 2).sum(dim=0))
        return nll + reg


def fit_poisson_ridge_lbfgs(
    X_t, y_t, alpha, *,
    offset_t=None,
    init_w=None, init_b=None,
    max_iter=200, tol=1e-6,
    use_full_batch=True, batch_size=65536, adam_steps=None,
):
    n, d = X_t.shape
    K = y_t.shape[1]
    model = PoissonRidgeBatched(d, K, alpha=alpha).to(X_t.device)
    if init_w is not None or init_b is not None:
        model.set_params(init_w, init_b)

    if use_full_batch:
        optimizer = LBFGS(
            model.parameters(), lr=1.0, max_iter=max_iter,
            tolerance_grad=tol, tolerance_change=tol,
            history_size=10, line_search_fn="strong_wolfe",
        )
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(X_t, y_t, offset_t)
            loss.backward()
            return loss
        optimizer.step(closure)
    else:
        opt = Adam(model.parameters(), lr=1e-2)
        if adam_steps is None:
            adam_steps = min(2000, max(400, 4 * (n // batch_size + 1)))
        for _ in range(adam_steps):
            idx = torch.randint(0, n, (min(batch_size, n),), device=X_t.device)
            off_b = None if offset_t is None else offset_t[idx]
            opt.zero_grad(set_to_none=True)
            model.loss(X_t[idx], y_t[idx], off_b).backward()
            opt.step()
        optimizer = LBFGS(
            model.parameters(), lr=1.0, max_iter=max_iter // 2,
            tolerance_grad=tol, tolerance_change=tol,
            history_size=10, line_search_fn="strong_wolfe",
        )
        def closure2():
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(X_t, y_t, offset_t)
            loss.backward()
            return loss
        optimizer.step(closure2)

    with torch.no_grad():
        w = model.W.detach().clone()
        b = model.b.detach().clone()
    return model, w, b


def approx_edf_batched(Xs_t, mu_t, alpha, n_probe=64):
    """Hutchinson trace estimate: edf_k = trace(B_k (B_k + α_k I)^{-1})."""
    _, d = Xs_t.shape
    K = mu_t.shape[1]
    device = Xs_t.device

    if not torch.is_tensor(alpha):
        alpha = torch.tensor(np.atleast_1d(np.asarray(alpha, dtype=np.float32)), device=device)
    else:
        alpha = alpha.float().to(device)
    if alpha.ndim == 0:
        alpha = alpha.expand(K)

    XT_W = Xs_t.T[None, :, :] * mu_t.T[:, None, :]   # (K, d, n)
    B = XT_W @ Xs_t                                    # (K, d, d)
    A = B + alpha[:, None, None] * torch.eye(d, device=device)
    z = torch.randn(K, d, n_probe, device=device)
    v = torch.linalg.solve(A, z)
    Bv = B @ v
    return (z * Bv).sum(dim=1).mean(dim=1)             # (K,)


# ── Inner CV for alpha tuning ─────────────────────────────────────────────────

def _tune_alpha_inner_cv(
    X_tr_raw, Y_tr_np, off_tr_np, *,
    alphas, inner_splits=5, seed=42, device="cuda",
    n_pca_components=100, use_full_batch=True, batch_size=65536,
    max_iter=200, tol=1e-6,
):
    """Return per-neuron best alpha by inner-CV Poisson log-likelihood."""
    kf = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    K = Y_tr_np.shape[1]
    alphas_desc = np.array(sorted(alphas, reverse=True), dtype=float)
    splits = list(kf.split(X_tr_raw))

    fold_data = []
    for tr_f, va_f in splits:
        Xtr_s, Xva_s, _ = _prep_X_with_pca(X_tr_raw[tr_f], X_tr_raw[va_f], n_components=n_pca_components)
        off_tr_f = off_tr_np[tr_f] if off_tr_np is not None else None
        off_va_f = off_tr_np[va_f] if off_tr_np is not None else None
        fold_data.append({
            "Xtr_t": _to_device(Xtr_s, device),
            "Ytr_t": _to_device(Y_tr_np[tr_f], device),
            "Xva_t": _to_device(Xva_s, device),
            "Yva_np": Y_tr_np[va_f].astype(np.float64),
            "off_tr_t": _to_device(off_tr_f, device),
            "off_va_t": _to_device(off_va_f, device),
        })

    fold_cache = {i: (None, None) for i in range(len(fold_data))}
    scores = np.empty((len(alphas_desc), len(fold_data), K), dtype=np.float64)

    for a_idx, a in enumerate(alphas_desc):
        for fi, fd in enumerate(fold_data):
            model, W, b = fit_poisson_ridge_lbfgs(
                fd["Xtr_t"], fd["Ytr_t"], alpha=float(a),
                offset_t=fd["off_tr_t"],
                init_w=fold_cache[fi][0], init_b=fold_cache[fi][1],
                max_iter=max_iter, tol=tol,
                use_full_batch=use_full_batch, batch_size=batch_size,
            )
            fold_cache[fi] = (W, b)
            with torch.no_grad():
                mu_va = model(fd["Xva_t"], fd["off_va_t"]).clamp_min(1e-10).cpu().numpy()
            scores[a_idx, fi] = poisson_ll_per_neuron(fd["Yva_np"], mu_va)

    mean_ll = scores.mean(axis=1)   # (A, K)
    best_a_idx = np.argmax(mean_ll, axis=0)
    best_alpha_vec = alphas_desc[best_a_idx]
    alpha_ll_mean = mean_ll[best_a_idx, np.arange(K)]
    alpha_ll_std = scores.std(axis=1)[best_a_idx, np.arange(K)]
    warm_start = next(iter(fold_cache.values()))
    return best_alpha_vec, alpha_ll_mean, alpha_ll_std, warm_start


# ── Outer nested CV ───────────────────────────────────────────────────────────

def run_nested_cv(
    X_full, Y_full, *,
    offset_full=None,
    seed=42,
    outer_splits=5,
    inner_splits=5,
    alphas,
    n_shuffles=0,
    device="cuda",
    use_full_batch=True,
    batch_size=65536,
    n_pca_components=100,
    n_probe_edf=64,
    verbose=True,
):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    kf_outer = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold_i, (tr_idx, te_idx) in enumerate(kf_outer.split(X_full)):
        if verbose:
            print(f"  fold {fold_i + 1}/{outer_splits}", flush=True)

        X_tr_raw, X_te_raw = X_full[tr_idx], X_full[te_idx]
        Y_tr_np, Y_te_np = Y_full[tr_idx], Y_full[te_idx]
        off_tr = offset_full[tr_idx] if offset_full is not None else None
        off_te = offset_full[te_idx] if offset_full is not None else None

        if verbose:
            print("    tuning alpha", flush=True)
        best_alpha, alpha_ll_mean, alpha_ll_std, (init_w, init_b) = _tune_alpha_inner_cv(
            X_tr_raw, Y_tr_np, off_tr,
            alphas=alphas, inner_splits=inner_splits,
            seed=seed + 1000 * fold_i, device=device,
            n_pca_components=n_pca_components,
            use_full_batch=use_full_batch, batch_size=batch_size,
        )

        if verbose:
            print("    fitting outer model", flush=True)
        Xtr_s, Xte_s, bundle = _prep_X_with_pca(X_tr_raw, X_te_raw, n_components=n_pca_components)
        X_tr_t = _to_device(Xtr_s, device)
        Y_tr_t = _to_device(Y_tr_np, device)
        X_te_t = _to_device(Xte_s, device)
        off_tr_t = _to_device(off_tr, device)
        off_te_t = _to_device(off_te, device)

        model, w_std_t, b_std_t = fit_poisson_ridge_lbfgs(
            X_tr_t, Y_tr_t, alpha=best_alpha,
            offset_t=off_tr_t, init_w=init_w, init_b=init_b,
            max_iter=300, tol=1e-6,
            use_full_batch=use_full_batch, batch_size=batch_size,
        )

        with torch.no_grad():
            mu_te = model(X_te_t, off_te_t).clamp_min(1e-10).cpu().numpy()
            mu_tr_fit = model(X_tr_t, off_tr_t)

        ll_real = poisson_ll_per_neuron(Y_te_np, mu_te)

        if off_tr is not None:
            avg_rate = Y_tr_np.sum(axis=0) / np.exp(off_tr).sum()
            mu_null = avg_rate * np.exp(off_te)[:, None]
        else:
            avg_rate = Y_tr_np.mean(axis=0)
            mu_null = np.broadcast_to(avg_rate, Y_te_np.shape).copy()
        ll_null = poisson_ll_per_neuron(Y_te_np, mu_null)

        pr2 = pseudo_r2(ll_real, ll_null)
        pear = pearson_per_neuron(Y_te_np, mu_te)
        spear = spearman_per_neuron(Y_te_np, mu_te)

        edf = approx_edf_batched(X_tr_t, mu_tr_fit, best_alpha, n_probe=n_probe_edf).cpu().numpy()
        aic = 2 * edf - 2 * ll_real
        bic = np.log(Y_te_np.shape[0]) * edf - 2 * ll_real

        w_pca, _ = backtransform_coef(
            w_std_t.detach().cpu().numpy(),
            b_std_t.detach().cpu().numpy(),
            bundle["mu_pca"], bundle["sd_pca"],
        )

        ll_shufs = ll_xshuf_mean = ll_diff = p_val_ll_xshuf = None
        if n_shuffles > 0:
            if verbose:
                print(f"    {n_shuffles} permutations", flush=True)
            ll_shufs_list = []
            for _ in range(n_shuffles):
                perm = rng.permutation(len(tr_idx))
                Xs_tr, Xs_te, _ = _prep_X_with_pca(X_tr_raw[perm], X_te_raw, n_components=n_pca_components)
                m_shuf, _, _ = fit_poisson_ridge_lbfgs(
                    _to_device(Xs_tr, device), Y_tr_t, alpha=best_alpha,
                    offset_t=off_tr_t, max_iter=200, tol=1e-6,
                    use_full_batch=use_full_batch, batch_size=batch_size,
                )
                with torch.no_grad():
                    mu_s = m_shuf(_to_device(Xs_te, device), off_te_t).clamp_min(1e-10).cpu().numpy()
                ll_shufs_list.append(poisson_ll_per_neuron(Y_te_np, mu_s))
            ll_shufs = np.array(ll_shufs_list)          # (n_shuffles, K)
            ll_xshuf_mean = ll_shufs.mean(axis=0)
            ll_diff = ll_real - ll_xshuf_mean
            p_val_ll_xshuf = (np.sum(ll_shufs >= ll_real, axis=0) + 1) / (n_shuffles + 1)

        fold_metrics.append(dict(
            fold=fold_i,
            state_dict=model.state_dict(),
            best_alpha=best_alpha,
            alpha_ll_mean=alpha_ll_mean,
            alpha_ll_std=alpha_ll_std,
            ll_real=ll_real,
            ll_null=ll_null,
            pseudo_r2=pr2,
            pearson_corr=pear,
            spearman_corr=spear,
            edf=edf,
            aic=aic,
            bic=bic,
            ll_shufs=ll_shufs,
            ll_xshuf_mean=ll_xshuf_mean,
            ll_diff=ll_diff,
            p_val_ll_xshuf=p_val_ll_xshuf,
            coef_pca_space=w_pca.astype(float),
        ))

    def _agg(key):
        vals = np.array([fm[key] for fm in fold_metrics], dtype=float)
        return {"mean": np.nanmean(vals, axis=0), "std": np.nanstd(vals, axis=0)}

    p_val_agg = ll_xshuf_mean_agg = None
    if n_shuffles > 0:
        T_obs = np.mean([fm["ll_real"] for fm in fold_metrics], axis=0)
        T_perm = np.stack([fm["ll_shufs"] for fm in fold_metrics], axis=0).mean(axis=0)  # (n_shuffles, K)
        p_val_agg = (np.sum(T_perm >= T_obs, axis=0) + 1) / (n_shuffles + 1)
        ll_xshuf_mean_agg = T_perm.mean(axis=0)

    summary = dict(
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        best_alpha=_agg("best_alpha"),
        ll_real=_agg("ll_real"),
        pseudo_r2=_agg("pseudo_r2"),
        pearson_corr=_agg("pearson_corr"),
        spearman_corr=_agg("spearman_corr"),
        edf=_agg("edf"),
        aic=_agg("aic"),
        bic=_agg("bic"),
        p_val_ll_xshuf=p_val_agg,
        ll_xshuf_mean=ll_xshuf_mean_agg,
    )
    return fold_metrics, summary


# ── Results serialisation ─────────────────────────────────────────────────────

def _get(x, n):
    return x[n] if x is not None else np.nan


def build_results_df(fold_metrics, summary):
    K = len(fold_metrics[0]["ll_real"])
    rows = []

    for fm in fold_metrics:
        fi = fm["fold"]
        for n in range(K):
            rows.append(dict(
                neuron_idx=n, fold_id=fi, is_summary=False,
                outer_splits=summary["outer_splits"],
                inner_splits=summary["inner_splits"],
                best_alpha=fm["best_alpha"][n],
                alpha_ll_mean=fm["alpha_ll_mean"][n],
                alpha_ll_std=fm["alpha_ll_std"][n],
                ll_real=fm["ll_real"][n],
                ll_null=fm["ll_null"][n],
                pseudo_r2=fm["pseudo_r2"][n],
                pearson_corr=fm["pearson_corr"][n],
                spearman_corr=fm["spearman_corr"][n],
                edf=fm["edf"][n],
                aic=fm["aic"][n],
                bic=fm["bic"][n],
                ll_xshuf_mean=_get(fm["ll_xshuf_mean"], n),
                ll_diff=_get(fm["ll_diff"], n),
                p_val_ll_xshuf=_get(fm["p_val_ll_xshuf"], n),
                ll_shufs=fm["ll_shufs"][:, n] if fm["ll_shufs"] is not None else np.nan,
                coef_pca_space=fm["coef_pca_space"][n],
            ))

    for n in range(K):
        rows.append(dict(
            neuron_idx=n, fold_id=np.nan, is_summary=True,
            outer_splits=summary["outer_splits"],
            inner_splits=summary["inner_splits"],
            best_alpha_mean=summary["best_alpha"]["mean"][n],
            best_alpha_std=summary["best_alpha"]["std"][n],
            ll_real_mean=summary["ll_real"]["mean"][n],
            ll_real_std=summary["ll_real"]["std"][n],
            pseudo_r2_mean=summary["pseudo_r2"]["mean"][n],
            pseudo_r2_std=summary["pseudo_r2"]["std"][n],
            pearson_corr_mean=summary["pearson_corr"]["mean"][n],
            pearson_corr_std=summary["pearson_corr"]["std"][n],
            spearman_corr_mean=summary["spearman_corr"]["mean"][n],
            spearman_corr_std=summary["spearman_corr"]["std"][n],
            edf_mean=summary["edf"]["mean"][n],
            edf_std=summary["edf"]["std"][n],
            aic_mean=summary["aic"]["mean"][n],
            aic_std=summary["aic"]["std"][n],
            bic_mean=summary["bic"]["mean"][n],
            bic_std=summary["bic"]["std"][n],
            p_val_ll_xshuf=_get(summary["p_val_ll_xshuf"], n),
            ll_xshuf_mean=_get(summary["ll_xshuf_mean"], n),
        ))

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patient",       type=str)
    parser.add_argument("vad_root",      type=Path)
    parser.add_argument("out_dir",       type=Path)
    parser.add_argument("--spike-offset-idx", type=int,   default=8)
    parser.add_argument("--gpt2-layer",       type=int,   default=-1)
    parser.add_argument("--n-pca",            type=int,   default=100)
    parser.add_argument("--outer-splits",     type=int,   default=5)
    parser.add_argument("--inner-splits",     type=int,   default=5)
    parser.add_argument("--n-alphas",         type=int,   default=30)
    parser.add_argument("--alpha-low",        type=float, default=-3.0)
    parser.add_argument("--alpha-high",       type=float, default=3.0)
    parser.add_argument("--n-shuffles",       type=int,   default=0)
    parser.add_argument("--embeddings-path",  type=Path,  default=None)
    parser.add_argument("--counts-path",      type=Path,  default=None)
    parser.add_argument("--durations-path",   type=Path,  default=None)
    parser.add_argument("--word-idx-path",    type=Path,  default=None,
                        help=".npy file of integer indices into the full word arrays "
                             "(used to subset to e.g. patient-speech-only words)")
    args = parser.parse_args()

    patient = args.patient
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    success_path = out_dir / f"{patient}_SUCCESS"
    error_path   = out_dir / f"{patient}_error.txt"

    if success_path.exists():
        print(f"already done: {patient}", flush=True)
        sys.exit(0)

    try:
        resolved = resolve_patient_inputs(
            patient,
            args.vad_root,
            embeddings_path=args.embeddings_path,
            counts_path=args.counts_path,
            durations_path=args.durations_path,
        )
        print(f"loading data for {patient}", flush=True)
        print(f"  embeddings: {resolved['embeddings_path']}", flush=True)
        print(f"  counts:     {resolved['counts_path']}", flush=True)
        print(f"  durations:  {resolved['durations_path']}", flush=True)

        gpt_raw = np.load(resolved["embeddings_path"], mmap_mode="r")
        X = gpt_raw[:, args.gpt2_layer].copy().astype(np.float32)

        counts_raw = np.load(resolved["counts_path"])
        Y = counts_raw[args.spike_offset_idx].astype(np.float32)

        durs_raw = np.load(resolved["durations_path"])
        log_durs = np.log(np.maximum(durs_raw, 1e-6)).astype(np.float32)

        if args.word_idx_path is not None:
            word_idx = np.load(args.word_idx_path)
            word_idx = word_idx.astype(int)
            X        = X[word_idx]
            Y        = Y[word_idx]
            log_durs = log_durs[word_idx]
            print(f"  word subset: {len(word_idx)} words (from {args.word_idx_path})", flush=True)

        print(f"  raw  X:{X.shape}  Y:{Y.shape}  durs:{log_durs.shape}", flush=True)
        X, Y, log_durs = nan_clean_XY(X, Y, durations=log_durs)
        Y = impute_Y_col_means(Y)
        print(f"  clean X:{X.shape}  Y:{Y.shape}", flush=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  device: {device}", flush=True)

        alphas = np.logspace(args.alpha_low, args.alpha_high, args.n_alphas)

        fold_metrics, summary = run_nested_cv(
            X, Y,
            offset_full=log_durs,
            seed=42,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            alphas=alphas,
            n_shuffles=args.n_shuffles,
            device=device,
            n_pca_components=args.n_pca,
            verbose=True,
        )

        results_df = build_results_df(fold_metrics, summary)
        models_dict = {f"fold_{fm['fold']}": fm["state_dict"] for fm in fold_metrics}

        with open(out_dir / f"{patient}_encoding_results_cv.pkl", "wb") as f:
            pickle.dump(results_df, f)
        torch.save(models_dict, out_dir / f"{patient}_encoding_models_cv.tar")

        meta = dict(
            patient=patient,
            patient_root=str(resolved["patient_root"]),
            embeddings_path=str(resolved["embeddings_path"]),
            counts_path=str(resolved["counts_path"]),
            durations_path=str(resolved["durations_path"]),
            word_idx_path=str(args.word_idx_path) if args.word_idx_path else None,
            n_words=int(X.shape[0]),
            n_neurons=int(Y.shape[1]),
            n_pca=args.n_pca,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            n_shuffles=args.n_shuffles,
            spike_offset_idx=args.spike_offset_idx,
            gpt2_layer=args.gpt2_layer,
        )
        with open(out_dir / f"{patient}_meta.json", "w") as f:
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
