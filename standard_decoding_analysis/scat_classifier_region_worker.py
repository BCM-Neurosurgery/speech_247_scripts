"""
Per-brain-resample region worker for sampled scatter-cluster decoding.

For a given (patient, day, region, brain_resample_idx) it:
  1. Loads micro_info.csv to find the unit column mask for `region`
  2. Subsamples N_UNITS=8 units from that region (seeded by brain_resample_idx)
     — if the region has exactly N_UNITS, uses all of them (only one brain resample)
  3. Runs N_CAT_RESAMPLES category-balance resamples, each as a full XGB classify+perm job
  4. Writes outputs per cat resample and a single SUCCESS flag after all complete

Output files (in --out-dir):
  scat_region_{region}_brain{b}_cat{c}.pkl
  summary_{region}_brain{b}_cat{c}.json
  best_params_{region}_brain{b}.json   (from cat resample 0, or the first to succeed)
  region_{region}_brain{b}_SUCCESS
  region_{region}_brain{b}_error.txt   (on failure)
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import dill as pickle
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from xgboost import XGBClassifier


# ── Helpers (shared with main worker) ─────────────────────────────────────────

def class_balance_words(categories: pd.Series, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cat = categories[(categories >= 0) & (categories != 10)]
    counts = cat.value_counts()
    median_count = int(np.round(np.median(counts.values)))
    to_keep: list[int] = []
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


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def make_xgb_base_params(num_class: int, random_state: int, n_jobs: int) -> dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": num_class,
        "tree_method": "hist",
        "device": "cpu",
        "eval_metric": "mlogloss",
        "random_state": random_state,
        "n_estimators": 1200,
        "n_jobs": n_jobs,
        "verbosity": 1,
    }


def choose_consensus_params(param_options: list[dict[str, Any]]) -> dict[str, Any]:
    params = dict(param_options[0])
    params["max_depth"] = int(np.min([p["max_depth"] for p in param_options]))
    params["min_child_weight"] = int(np.max([p["min_child_weight"] for p in param_options]))
    params["gamma"] = float(np.max([p["gamma"] for p in param_options]))
    params["reg_lambda"] = float(np.max([p["reg_lambda"] for p in param_options]))
    params["reg_alpha"] = float(np.max([p["reg_alpha"] for p in param_options]))
    params["subsample"] = float(np.median([p["subsample"] for p in param_options]))
    params["colsample_bytree"] = float(np.median([p["colsample_bytree"] for p in param_options]))
    params["learning_rate"] = float(np.median([p["learning_rate"] for p in param_options]))
    params["n_estimators"] = int(np.median([p["n_estimators"] for p in param_options]))
    params["tree_method"] = "hist"
    params["device"] = "cuda"
    params.pop("predictor", None)
    params.pop("use_label_encoder", None)
    return params


def score_metric(y_true: np.ndarray, y_pred: np.ndarray, which: str) -> float:
    if which == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if which == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro"))
    if which == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError("perm_metric must be one of: accuracy, f1_macro, balanced_accuracy")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── Core classify+permute logic ────────────────────────────────────────────────

def run_one_resample(
    X: np.ndarray,
    y: np.ndarray,
    *,
    params: dict[str, Any] | None,
    test_size: float,
    random_state: int,
    n_iter: int,
    cv_splits: int,
    scoring: str,
    perm_metric: str,
    n_shuffles: int,
    n_jobs: int,
) -> tuple[XGBClassifier, dict[str, Any]]:
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y))
    K = int(np.unique(y_enc).size)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y_enc, test_size=test_size, stratify=y_enc, random_state=random_state
    )

    mu, sd = standardize_fit(Xtr)
    Xtr_std = standardize_apply(Xtr, mu, sd)
    Xte_std = standardize_apply(Xte, mu, sd)

    base_params = make_xgb_base_params(K, random_state, n_jobs)
    base = XGBClassifier(**base_params)

    if params is None:
        param_dist = {
            "max_depth": [4, 6, 8, 10],
            "min_child_weight": [1, 2, 5, 10],
            "gamma": [0.0, 0.5, 1.0, 2.0],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_lambda": [0.0, 1.0, 5.0, 10.0],
            "reg_alpha": [0.0, 0.5, 1.0],
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [400, 800, 1200, 1600],
        }
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            verbose=1,
            n_jobs=1,
            refit=True,
            random_state=random_state,
        )
        search.fit(Xtr_std, ytr)
        best_model = search.best_estimator_
        best_cv_score = float(search.best_score_)
    else:
        fit_params = dict(base_params)
        fit_params.update(params)
        fit_params["device"] = "cuda"
        fit_params["tree_method"] = "hist"
        fit_params["n_jobs"] = n_jobs
        fit_params.pop("predictor", None)
        fit_params.pop("use_label_encoder", None)
        best_model = XGBClassifier(**fit_params)
        best_model.fit(Xtr_std, ytr)
        best_cv_score = np.nan

    yhat = best_model.predict(Xte_std)
    yproba = best_model.predict_proba(Xte_std)

    acc = float(accuracy_score(yte, yhat))
    f1_mac = float(f1_score(yte, yhat, average="macro"))
    f1_mic = float(f1_score(yte, yhat, average="micro"))
    bal_acc = float(balanced_accuracy_score(yte, yhat))
    cm = confusion_matrix(yte, yhat)
    report = classification_report(yte, yhat)

    auc_macro = None
    try:
        y_bin = label_binarize(yte, classes=np.arange(K))
        auc_macro = float(roc_auc_score(y_bin, yproba, multi_class="ovr", average="macro"))
    except Exception:
        pass

    counts = np.bincount(yte, minlength=K).astype(float)
    p = counts / counts.sum()
    p0_prop = float((p**2).sum())
    p0_major = float(p.max())
    k_corr = int((yhat == yte).sum())
    n = int(yte.size)
    binom_p = float(binomtest(k_corr, n, p0_prop, alternative="greater").pvalue)

    obs_score = score_metric(yte, yhat, perm_metric)
    rng_perm = np.random.default_rng(random_state)
    perm_scores = np.empty(n_shuffles, dtype=float) if n_shuffles > 0 else None
    perm_pval = None

    for s in range(n_shuffles):
        perm = rng_perm.permutation(len(Xtr_std))
        Xtr_perm = Xtr_std[perm]
        perm_model = XGBClassifier(**best_model.get_params())
        perm_model.fit(Xtr_perm, ytr)
        yhat_perm = perm_model.predict(Xte_std)
        perm_scores[s] = score_metric(yte, yhat_perm, perm_metric)

    if perm_scores is not None:
        perm_pval = float((np.sum(perm_scores >= obs_score) + 1) / (n_shuffles + 1))

    results = {
        "y_true": yte,
        "y_pred": yhat,
        "y_proba": yproba,
        "acc": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_mac,
        "f1_micro": f1_mic,
        "cm": cm,
        "class_report": report,
        "auc_macro_ovr": auc_macro,
        "best_params": best_model.get_params(),
        "cv_best_score": best_cv_score,
        "classes": le.classes_,
        "baseline_proportional_acc": p0_prop,
        "baseline_majority_acc": p0_major,
        "binomial_p_vs_proportional": binom_p,
        "perm_metric": perm_metric,
        "perm_scores": perm_scores,
        "perm_pvalue": perm_pval,
        "observed_perm_metric": obs_score,
        "test_size": test_size,
        "random_state": random_state,
        "n_shuffles": n_shuffles,
        "perm_null": "shuffle_training_X_only",
    }
    return best_model, results


def build_summary_row(
    patient: str,
    region: str,
    brain_resample_idx: int,
    cat_resample_idx: int,
    seed: int,
    n_region_units: int,
    unit_col_indices: np.ndarray,
    sampled_idx_raw: np.ndarray,
    sampled_idx_final: np.ndarray,
    results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "patient": patient,
        "region": region,
        "brain_resample_idx": brain_resample_idx,
        "cat_resample_idx": cat_resample_idx,
        "seed": seed,
        "n_region_units": n_region_units,
        "n_units_used": int(len(unit_col_indices)),
        "n_sampled_raw": int(len(sampled_idx_raw)),
        "n_sampled_final": int(len(sampled_idx_final)),
        "n_test": int(len(results["y_true"])),
        "acc": float(results["acc"]),
        "balanced_accuracy": float(results["balanced_accuracy"]),
        "f1_macro": float(results["f1_macro"]),
        "f1_micro": float(results["f1_micro"]),
        "auc_macro_ovr": None if results["auc_macro_ovr"] is None else float(results["auc_macro_ovr"]),
        "baseline_proportional_acc": float(results["baseline_proportional_acc"]),
        "baseline_majority_acc": float(results["baseline_majority_acc"]),
        "binomial_p_vs_proportional": float(results["binomial_p_vs_proportional"]),
        "perm_metric": results["perm_metric"],
        "observed_perm_metric": float(results["observed_perm_metric"]),
        "perm_pvalue": None if results["perm_pvalue"] is None else float(results["perm_pvalue"]),
        "cv_best_score": None if np.isnan(results["cv_best_score"]) else float(results["cv_best_score"]),
        "perm_null": results["perm_null"],
    }


# ── Pre-flight checks ──────────────────────────────────────────────────────────

def check_data_feasibility(
    y: np.ndarray,
    test_size: float,
    cv_splits: int,
    do_hyperparam_search: bool,
) -> None:
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)
    if n_classes < 2:
        raise ValueError(f"Only {n_classes} semantic class after balancing — need ≥ 2 to classify.")

    min_class_n = int(class_counts.min())
    min_for_split = max(
        int(np.ceil(1.0 / test_size)),
        int(np.ceil(1.0 / (1.0 - test_size))),
    )
    if min_class_n < min_for_split:
        raise ValueError(
            f"Smallest class has {min_class_n} samples; stratified split "
            f"(test_size={test_size}) needs ≥ {min_for_split} per class."
        )

    if do_hyperparam_search:
        approx_train_per_class = int(min_class_n * (1.0 - test_size))
        if approx_train_per_class < cv_splits:
            raise ValueError(
                f"Smallest class would have ~{approx_train_per_class} training samples; "
                f"StratifiedKFold needs ≥ {cv_splits} per class."
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--brain-resample-idx", type=int, required=True)
    parser.add_argument("--micro-info-path", type=Path, required=True)
    parser.add_argument("--frs-path", type=Path, required=True)
    parser.add_argument("--cluster-preds-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--n-units", type=int, default=8)
    parser.add_argument("--n-cat-resamples", type=int, default=10)
    parser.add_argument("--seed-stride", type=int, default=42)
    parser.add_argument("--word-idx-path", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--n-shuffles", type=int, default=50)
    parser.add_argument("--scoring", type=str, default="f1_macro")
    parser.add_argument("--perm-metric", type=str, default="f1_macro")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    b = args.brain_resample_idx
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    success_path = out_dir / f"region_{args.region}_brain{b}_SUCCESS"
    error_path   = out_dir / f"region_{args.region}_brain{b}_error.txt"

    # Check if all cat resamples already done
    all_done = all(
        (out_dir / f"scat_region_{args.region}_brain{b}_cat{c}.pkl").exists()
        and (out_dir / f"summary_{args.region}_brain{b}_cat{c}.json").exists()
        for c in range(args.n_cat_resamples)
    )
    if success_path.exists() and all_done:
        print(f"already done: {args.patient} region={args.region} brain={b}", flush=True)
        return 0

    if success_path.exists():
        success_path.unlink()
    if error_path.exists():
        error_path.unlink()

    try:
        # ── Load micro_info and find region unit mask ──────────────────────────
        micro_df = pd.read_csv(args.micro_info_path, index_col=0)
        region_mask = micro_df["region_symbol"] == args.region
        n_region_units = int(region_mask.sum())
        if n_region_units == 0:
            raise ValueError(f"Region '{args.region}' not found in {args.micro_info_path}")

        region_col_indices = np.where(region_mask.values)[0]

        # ── Subsample units ────────────────────────────────────────────────────
        brain_seed = b * args.seed_stride
        brain_rng = np.random.default_rng(brain_seed)

        if n_region_units > args.n_units:
            unit_col_indices = brain_rng.choice(region_col_indices, size=args.n_units, replace=False)
        else:
            unit_col_indices = region_col_indices

        # ── Load neural data ───────────────────────────────────────────────────
        frs_full = np.load(args.frs_path, mmap_mode="r")
        categories = np.load(args.cluster_preds_path, mmap_mode="r")
        categories = pd.Series(np.asarray(categories).astype(int))

        if len(categories) != len(frs_full):
            raise ValueError(
                f"Length mismatch: cluster preds has {len(categories)} rows "
                f"but firing rates has {len(frs_full)} rows"
            )

        # Subset to word-idx window if provided (per-day filtering)
        if args.word_idx_path is not None:
            word_idx = np.load(args.word_idx_path).astype(int)
            frs_full = np.asarray(frs_full[word_idx])
            categories = categories.iloc[word_idx].reset_index(drop=True)
        else:
            frs_full = np.asarray(frs_full)

        # Select region units
        frs_region = frs_full[:, unit_col_indices]

        valid_cats = categories[(categories >= 0) & (categories != 10)]
        if len(valid_cats) == 0:
            raise ValueError("No words with valid semantic cluster labels after subsetting.")

        # ── Load consensus params ──────────────────────────────────────────────
        params = None
        if args.params_json is not None:
            params = json.loads(args.params_json.read_text())

        best_params_written = False
        n_failed = 0

        # ── Cat resample loop ──────────────────────────────────────────────────
        for c in range(args.n_cat_resamples):
            pkl_path     = out_dir / f"scat_region_{args.region}_brain{b}_cat{c}.pkl"
            summary_path = out_dir / f"summary_{args.region}_brain{b}_cat{c}.json"
            bp_path      = out_dir / f"best_params_{args.region}_brain{b}.json"

            if pkl_path.exists() and summary_path.exists():
                print(f"  cat {c}: already done — skipping", flush=True)
                if not best_params_written and bp_path.exists():
                    best_params_written = True
                continue

            try:
                cat_seed = (b * args.n_cat_resamples + c) * args.seed_stride
                cat_rng  = np.random.default_rng(cat_seed)

                y_sampled, sampled_idx_raw = class_balance_words(categories, cat_rng)
                frs_keep = frs_region[sampled_idx_raw]

                mask_keep_x = ~np.isnan(frs_keep).all(axis=1)
                X = frs_keep[mask_keep_x]
                y = y_sampled[mask_keep_x]
                sampled_idx_final = sampled_idx_raw[mask_keep_x]
                X = impute_X_all(X)

                check_data_feasibility(
                    y, args.test_size, args.cv_splits, do_hyperparam_search=(params is None)
                )

                print(
                    f"  cat {c}: X={X.shape}, n_classes={len(np.unique(y))}, "
                    f"seed={cat_seed}",
                    flush=True,
                )

                model, results = run_one_resample(
                    X, y,
                    params=params,
                    test_size=args.test_size,
                    random_state=cat_seed,
                    n_iter=args.n_iter,
                    cv_splits=args.cv_splits,
                    scoring=args.scoring,
                    perm_metric=args.perm_metric,
                    n_shuffles=args.n_shuffles,
                    n_jobs=args.n_jobs,
                )

                results["seed"] = cat_seed
                results["brain_resample_idx"] = b
                results["cat_resample_idx"] = c
                results["region"] = args.region
                results["unit_col_indices"] = unit_col_indices
                results["sampled_idx_raw"] = sampled_idx_raw
                results["sampled_idx_final"] = sampled_idx_final

                summary = build_summary_row(
                    args.patient, args.region, b, c,
                    cat_seed, n_region_units, unit_col_indices,
                    sampled_idx_raw, sampled_idx_final, results,
                )

                with open(pkl_path, "wb") as f:
                    pickle.dump((model, results), f)
                summary_path.write_text(json.dumps(summary, indent=2, default=json_default))

                if not best_params_written:
                    bp_path.write_text(
                        json.dumps(results["best_params"], indent=2, default=json_default)
                    )
                    best_params_written = True

                print(
                    f"  cat {c}: done — f1_macro={results['f1_macro']:.4f} "
                    f"perm_p={results['perm_pvalue']:.3e}",
                    flush=True,
                )

            except Exception:
                tb = traceback.format_exc()
                cat_err_path = out_dir / f"region_{args.region}_brain{b}_cat{c}_error.txt"
                cat_err_path.write_text(tb)
                print(f"  cat {c}: FAILED\n{tb}", flush=True)
                n_failed += 1

        if n_failed == args.n_cat_resamples:
            raise RuntimeError(f"All {args.n_cat_resamples} cat resamples failed.")

        success_path.write_text("ok\n")
        print(
            f"done: patient={args.patient} region={args.region} brain={b} "
            f"({args.n_cat_resamples - n_failed}/{args.n_cat_resamples} cat resamples ok)",
            flush=True,
        )
        return 0

    except Exception:
        tb = traceback.format_exc()
        error_path.write_text(tb)
        print(tb, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
