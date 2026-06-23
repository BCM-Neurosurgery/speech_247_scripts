"""
Plot summary statistics from SCAT XGBoost decoding results.

Usage:
    python plot_decoding_results.py <aggregate_pkl> [--out-dir OUT_DIR] [--patient PATIENT]

Outputs SVG figures to <out_dir>/ (same dir as pkl by default).

Figures:
    1. metrics_across_resamples.svg  — box/strip of f1_macro, bal_acc, f1_micro, AUC across 20 resamples
    2. permutation_test.svg          — observed score vs permutation null distribution (pooled across resamples)
    3. confusion_matrix.svg          — average confusion matrix (normalised by true label) across resamples
    4. per_class_f1.svg              — per-class F1 scores across resamples (box + strip)
    5. summary_panel.svg             — compact 4-panel overview
    6. perm_pvalues.svg              — distribution of per-resample permutation p-values
"""

import argparse
import json
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    classification_report,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def savefig(fig, path):
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def load_results(pkl_path):
    with open(pkl_path, "rb") as f:
        agg = pickle.load(f)
    # agg is dict: resample_0 ... resample_N
    resamples = []
    for key in sorted(agg.keys(), key=lambda k: int(k.split("_")[1])):
        entry = agg[key]
        r = entry[1] if isinstance(entry, (tuple, list)) else entry
        r = dict(r)  # copy so we can add keys
        r["resample_key"] = key
        resamples.append(r)
    return resamples


def load_summary_csv(pkl_path):
    csv_path = pkl_path.with_name(pkl_path.stem.replace("_all_resamples", "_all_resamples") + ".csv")
    csv_path = pkl_path.parent / "scat_sampled_all_resamples.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def get_class_labels(resamples):
    """Return sorted unique class IDs across all resamples (integers → strings)."""
    all_classes = set()
    for r in resamples:
        all_classes.update(r["classes"].tolist())
    return sorted(all_classes)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_metrics_across_resamples(resamples, out_dir, patient):
    metrics = {
        "f1_macro":          [r["f1_macro"] for r in resamples],
        "balanced_accuracy": [r["balanced_accuracy"] for r in resamples],
        "f1_micro":          [r["f1_micro"] for r in resamples],
        "AUC (macro OvR)":   [r["auc_macro_ovr"] if r["auc_macro_ovr"] is not None else np.nan
                              for r in resamples],
    }
    baselines = {
        "f1_macro":          [r.get("baseline_proportional_acc", np.nan) for r in resamples],
        "balanced_accuracy": [1.0 / len(r["classes"]) for r in resamples],
        "f1_micro":          [r.get("baseline_proportional_acc", np.nan) for r in resamples],
        "AUC (macro OvR)":   [0.5] * len(resamples),
    }

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    fig.suptitle(f"{patient} — Decoding metrics across {len(resamples)} resamples", fontsize=13)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for ax, (metric, vals), color, (_, base) in zip(
        axes, metrics.items(), colors, baselines.items()
    ):
        vals_arr = np.array(vals, dtype=float)
        x = np.zeros(len(vals_arr))
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(vals_arr))
        ax.boxplot(vals_arr[~np.isnan(vals_arr)], positions=[0], widths=0.4,
                   patch_artist=True, boxprops=dict(facecolor=color, alpha=0.4),
                   medianprops=dict(color="black", linewidth=2),
                   whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                   flierprops=dict(marker="o", markersize=3, alpha=0.5))
        ax.scatter(x + jitter, vals_arr, s=25, color=color, alpha=0.7, zorder=3)
        base_arr = np.array(base, dtype=float)
        ax.axhline(np.nanmean(base_arr), color="tomato", lw=1.5, linestyle="--",
                   label=f"chance={np.nanmean(base_arr):.3f}")
        med = np.nanmedian(vals_arr)
        ax.set_title(f"{metric}\nmedian={med:.3f}", fontsize=9)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_metrics_across_resamples.svg")


def plot_permutation_test(resamples, out_dir, patient):
    obs_scores = np.array([r["observed_perm_metric"] for r in resamples])
    perm_metric_name = resamples[0].get("perm_metric", "f1_macro")

    # Pool all permutation scores
    perm_all = []
    for r in resamples:
        ps = r.get("perm_scores")
        if ps is not None:
            perm_all.append(np.array(ps, dtype=float))
    perm_pooled = np.concatenate(perm_all) if perm_all else np.array([])

    pvalues = np.array([
        r["perm_pvalue"] if r.get("perm_pvalue") is not None else np.nan
        for r in resamples
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{patient} — Permutation test ({perm_metric_name})", fontsize=13)

    # Left: observed vs null distribution
    ax = axes[0]
    if len(perm_pooled) > 0:
        ax.hist(perm_pooled, bins=40, color="#AAAAAA", edgecolor="white", linewidth=0.4,
                alpha=0.7, density=True, label=f"null (n={len(perm_pooled)})")
    for obs in obs_scores:
        ax.axvline(obs, color="#4C72B0", lw=0.8, alpha=0.5)
    ax.axvline(np.nanmean(obs_scores), color="#C44E52", lw=2.5, linestyle="-",
               label=f"mean observed={np.nanmean(obs_scores):.3f}")
    ax.set_xlabel(perm_metric_name)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.set_title("Observed (blue lines) vs null")

    # Right: per-resample p-values
    ax = axes[1]
    ax.scatter(range(len(pvalues)), pvalues, s=40, color="#4C72B0", zorder=3)
    ax.axhline(0.05, color="tomato", lw=1.5, linestyle="--", label="p=0.05")
    n_sig = int(np.sum(pvalues < 0.05))
    ax.set_xlabel("Resample index")
    ax.set_ylabel("Permutation p-value")
    ax.set_title(f"Per-resample p-values ({n_sig}/{len(pvalues)} p<0.05)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_permutation_test.svg")


def plot_confusion_matrix(resamples, out_dir, patient):
    classes = get_class_labels(resamples)
    K = len(classes)
    class_strs = [str(c) for c in classes]

    # Build label-aligned confusion matrices and average
    cms_norm = []
    for r in resamples:
        y_t = r["y_true"]
        y_p = r["y_pred"]
        r_classes = sorted(r["classes"].tolist())
        cm_r = confusion_matrix(y_t, y_p, labels=list(range(len(r_classes))))
        # normalise by row (true label)
        row_sums = cm_r.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cms_norm.append(cm_r.astype(float) / row_sums)

    cm_mean = np.mean(cms_norm, axis=0)
    cm_std = np.std(cms_norm, axis=0)

    fig, ax = plt.subplots(figsize=(max(7, K * 0.8), max(6, K * 0.7)))

    # Custom colormap: white → blue
    cmap = LinearSegmentedColormap.from_list("wb", ["white", "#4C72B0"])
    im = ax.imshow(cm_mean, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Proportion (row-normalised)")

    thresh = cm_mean.max() / 2.0
    for i in range(K):
        for j in range(K):
            val = cm_mean[i, j]
            err = cm_std[i, j]
            text = f"{val:.2f}\n±{err:.2f}"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"pred {c}" for c in class_strs], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"true {c}" for c in class_strs], fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{patient} — Avg confusion matrix (±std, {len(resamples)} resamples)\nrow-normalised by true label")

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_confusion_matrix.svg")


def plot_per_class_f1(resamples, out_dir, patient):
    classes = get_class_labels(resamples)
    K = len(classes)
    class_strs = [str(c) for c in classes]

    # Per-class F1 per resample
    per_class_f1 = np.full((len(resamples), K), np.nan)
    for ri, r in enumerate(resamples):
        y_t = r["y_true"]
        y_p = r["y_pred"]
        r_classes = sorted(r["classes"].tolist())
        f1s = f1_score(y_t, y_p, average=None, labels=list(range(len(r_classes))), zero_division=0)
        for ki, cls in enumerate(r_classes):
            if cls in classes:
                gi = classes.index(cls)
                per_class_f1[ri, gi] = f1s[ki]

    fig, ax = plt.subplots(figsize=(max(8, K * 0.9), 5))
    chance = 1.0 / K

    means = np.nanmean(per_class_f1, axis=0)
    stds = np.nanstd(per_class_f1, axis=0)
    x = np.arange(K)

    # strip plot
    rng = np.random.default_rng(1)
    for ri in range(len(resamples)):
        jitter = rng.uniform(-0.2, 0.2, K)
        ax.scatter(x + jitter, per_class_f1[ri], s=15, alpha=0.3, color="#4C72B0")
    ax.bar(x, means, width=0.5, alpha=0.4, color="#4C72B0", label="mean F1")
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
    ax.axhline(chance, color="tomato", lw=1.5, linestyle="--", label=f"chance=1/{K}={chance:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in class_strs], rotation=45, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title(f"{patient} — Per-class F1 across {len(resamples)} resamples (mean ± std)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_per_class_f1.svg")


def plot_perm_pvalues(resamples, out_dir, patient):
    pvalues = np.array([
        r["perm_pvalue"] if r.get("perm_pvalue") is not None else np.nan
        for r in resamples
    ])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pvalues, bins=20, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(0.05, color="tomato", lw=1.5, linestyle="--", label="p=0.05")
    n_sig = int(np.sum(pvalues < 0.05))
    ax.set_xlabel("Permutation p-value")
    ax.set_ylabel("Count")
    ax.set_title(f"{patient} — Permutation p-values ({n_sig}/{len(pvalues)} significant, p<0.05)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_perm_pvalues.svg")


def plot_summary_panel(resamples, out_dir, patient):
    perm_metric_name = resamples[0].get("perm_metric", "f1_macro")
    f1_vals = np.array([r["f1_macro"] for r in resamples])
    bal_vals = np.array([r["balanced_accuracy"] for r in resamples])
    auc_vals = np.array([r["auc_macro_ovr"] if r["auc_macro_ovr"] is not None else np.nan
                         for r in resamples])
    pvalues = np.array([r["perm_pvalue"] if r.get("perm_pvalue") is not None else np.nan
                        for r in resamples])

    perm_all = []
    for r in resamples:
        ps = r.get("perm_scores")
        if ps is not None:
            perm_all.append(np.array(ps, dtype=float))
    perm_pooled = np.concatenate(perm_all) if perm_all else np.array([])

    classes = get_class_labels(resamples)
    K = len(classes)
    chance_f1 = 1.0 / K

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"{patient} — Semantic category decoding (XGBoost, {len(resamples)} resamples, {K} classes)",
        fontsize=14, y=1.01
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    # 1. F1 macro across resamples (sorted bar)
    ax1 = fig.add_subplot(gs[0, :2])
    order = np.argsort(f1_vals)[::-1]
    colors_bar = ["#C44E52" if pvalues[i] < 0.05 else "#4C72B0" for i in order]
    ax1.bar(range(len(f1_vals)), f1_vals[order], color=colors_bar, width=0.8, linewidth=0)
    ax1.axhline(chance_f1, color="tomato", lw=1.5, linestyle="--",
                label=f"chance=1/{K}={chance_f1:.3f}")
    ax1.axhline(np.nanmean(f1_vals), color="black", lw=1.5, linestyle=":",
                label=f"mean={np.nanmean(f1_vals):.3f}")
    ax1.set_xlabel("Resample rank")
    ax1.set_ylabel("F1 macro")
    n_sig = int(np.sum(pvalues < 0.05))
    ax1.set_title(f"F1 macro sorted ({n_sig}/{len(resamples)} p<0.05 in red)")
    ax1.legend(fontsize=8)

    # 2. F1 macro histogram
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(f1_vals, bins=12, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax2.axvline(np.nanmedian(f1_vals), color="black", lw=1.5, linestyle="--",
                label=f"med={np.nanmedian(f1_vals):.3f}")
    ax2.axvline(chance_f1, color="tomato", lw=1.5, linestyle="--", label=f"chance={chance_f1:.3f}")
    ax2.set_xlabel("F1 macro")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=7)
    ax2.set_title("F1 macro dist.")

    # 3. Observed vs null
    ax3 = fig.add_subplot(gs[0, 3])
    if len(perm_pooled) > 0:
        ax3.hist(perm_pooled, bins=30, color="#AAAAAA", edgecolor="white", linewidth=0.3,
                 alpha=0.8, density=True, label="null")
    ax3.axvline(np.nanmean(f1_vals), color="#C44E52", lw=2.0, linestyle="-",
                label=f"obs={np.nanmean(f1_vals):.3f}")
    ax3.set_xlabel(perm_metric_name)
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=7)
    ax3.set_title("Obs. vs null")

    # 4. Balanced accuracy
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(bal_vals, bins=12, color="#55A868", edgecolor="white", linewidth=0.5)
    ax4.axvline(np.nanmedian(bal_vals), color="black", lw=1.5, linestyle="--",
                label=f"med={np.nanmedian(bal_vals):.3f}")
    ax4.axvline(1.0 / K, color="tomato", lw=1.5, linestyle="--", label=f"chance=1/{K}")
    ax4.set_xlabel("Balanced accuracy")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=7)
    ax4.set_title("Balanced accuracy")

    # 5. AUC
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(auc_vals[~np.isnan(auc_vals)], bins=12, color="#8172B2", edgecolor="white", linewidth=0.5)
    ax5.axvline(np.nanmedian(auc_vals), color="black", lw=1.5, linestyle="--",
                label=f"med={np.nanmedian(auc_vals):.3f}")
    ax5.axvline(0.5, color="tomato", lw=1.5, linestyle="--", label="chance=0.5")
    ax5.set_xlabel("AUC macro OvR")
    ax5.set_ylabel("Count")
    ax5.legend(fontsize=7)
    ax5.set_title("AUC macro OvR")

    # 6. Per-resample p-value scatter
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(range(len(pvalues)), pvalues, s=35, color="#4C72B0", zorder=3, alpha=0.8)
    ax6.axhline(0.05, color="tomato", lw=1.5, linestyle="--", label="p=0.05")
    ax6.set_xlabel("Resample")
    ax6.set_ylabel("Perm. p-value")
    ax6.set_title(f"Per-resample p-values")
    ax6.legend(fontsize=7)

    # 7. F1 macro vs balanced accuracy scatter
    ax7 = fig.add_subplot(gs[1, 3])
    sig_mask = pvalues < 0.05
    ax7.scatter(bal_vals[~sig_mask], f1_vals[~sig_mask], s=25, alpha=0.5, color="gray", label="n.s.")
    ax7.scatter(bal_vals[sig_mask], f1_vals[sig_mask], s=35, alpha=0.8, color="#C44E52", label="p<0.05")
    ax7.set_xlabel("Balanced accuracy")
    ax7.set_ylabel("F1 macro")
    ax7.set_title("F1 macro vs bal. acc.")
    ax7.legend(fontsize=7)

    savefig(fig, out_dir / f"{patient}_summary_panel.svg")


def print_text_summary(resamples, patient):
    f1_vals = np.array([r["f1_macro"] for r in resamples])
    bal_vals = np.array([r["balanced_accuracy"] for r in resamples])
    auc_vals = np.array([r["auc_macro_ovr"] if r["auc_macro_ovr"] is not None else np.nan
                         for r in resamples])
    pvalues = np.array([r["perm_pvalue"] if r.get("perm_pvalue") is not None else np.nan
                        for r in resamples])
    classes = get_class_labels(resamples)
    K = len(classes)

    print(f"\n=== {patient} Decoding Summary ===")
    print(f"  classes: {K}   resamples: {len(resamples)}")
    print(f"  F1 macro      — mean={np.nanmean(f1_vals):.4f}  median={np.nanmedian(f1_vals):.4f}  std={np.nanstd(f1_vals):.4f}  chance=1/{K}={1/K:.4f}")
    print(f"  Balanced acc  — mean={np.nanmean(bal_vals):.4f}  median={np.nanmedian(bal_vals):.4f}")
    print(f"  AUC macro OvR — mean={np.nanmean(auc_vals):.4f}  median={np.nanmedian(auc_vals):.4f}")
    print(f"  Perm p<0.05   — {int(np.sum(pvalues < 0.05))}/{len(pvalues)} resamples")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate_pkl", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--patient", type=str, default=None)
    args = parser.parse_args()

    pkl_path = args.aggregate_pkl
    out_dir = args.out_dir if args.out_dir is not None else pkl_path.parent
    patient = args.patient if args.patient is not None else pkl_path.stem.split("_scat")[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"patient: {patient}")
    print(f"pkl:     {pkl_path}")
    print(f"out_dir: {out_dir}")

    resamples = load_results(pkl_path)
    print(f"resamples loaded: {len(resamples)}")

    print_text_summary(resamples, patient)

    plot_summary_panel(resamples, out_dir, patient)
    plot_metrics_across_resamples(resamples, out_dir, patient)
    plot_permutation_test(resamples, out_dir, patient)
    plot_confusion_matrix(resamples, out_dir, patient)
    plot_per_class_f1(resamples, out_dir, patient)
    plot_perm_pvalues(resamples, out_dir, patient)

    print("done")


if __name__ == "__main__":
    main()
