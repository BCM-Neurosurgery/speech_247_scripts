"""
Plot summary statistics from Poisson GLM encoding results.

Usage:
    python plot_encoding_results.py <pkl_path> [--out-dir OUT_DIR] [--patient PATIENT]

Outputs SVG figures to <out_dir>/ (default: same directory as pkl_path).

Figures produced:
    1. pseudo_r2_distribution.svg     — histogram + sorted bar of pseudo-R² across neurons
    2. correlations_distribution.svg  — Pearson / Spearman correlation distributions
    3. pseudo_r2_vs_pearson.svg       — scatter: pseudo-R² mean vs Pearson r
    4. significance.svg               — pseudo-R² coloured by shuffle p-value (if shuffles ran)
    5. best_alpha_distribution.svg    — distribution of tuned ridge alphas
    6. edf_vs_pseudo_r2.svg           — effective degrees of freedom vs pseudo-R²
"""

import argparse
from pathlib import Path

import dill as pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── helpers ───────────────────────────────────────────────────────────────────

def load_summary(pkl_path):
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    summary = df[df["is_summary"] == True].copy().reset_index(drop=True)
    summary = summary.sort_values("neuron_idx").reset_index(drop=True)
    return df, summary


def savefig(fig, path):
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ── individual plots ──────────────────────────────────────────────────────────

def plot_pseudo_r2_distribution(summary, out_dir, patient):
    pr2 = summary["pseudo_r2_mean"].values
    pr2_std = summary["pseudo_r2_std"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{patient} — Pseudo-R² across neurons (5-fold CV mean)", fontsize=13)

    # histogram
    ax = axes[0]
    ax.hist(pr2, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(np.nanmedian(pr2), color="tomato", lw=1.5, linestyle="--",
               label=f"median = {np.nanmedian(pr2):.3f}")
    ax.axvline(0, color="black", lw=0.8, linestyle=":")
    ax.set_xlabel("Pseudo-R²")
    ax.set_ylabel("Neuron count")
    ax.legend(fontsize=9)
    ax.set_title("Distribution")

    # sorted bar
    ax = axes[1]
    order = np.argsort(pr2)[::-1]
    x = np.arange(len(pr2))
    ax.bar(x, pr2[order], color="#4C72B0", width=1.0, linewidth=0)
    ax.errorbar(x, pr2[order], yerr=pr2_std[order],
                fmt="none", ecolor="black", elinewidth=0.4, capsize=0, alpha=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Neuron rank")
    ax.set_ylabel("Pseudo-R²")
    ax.set_title("Sorted neurons (mean ± std)")

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_pseudo_r2_distribution.svg")


def plot_correlations_distribution(summary, out_dir, patient):
    pear = summary["pearson_corr_mean"].values
    spear = summary["spearman_corr_mean"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{patient} — Correlation distributions (5-fold CV mean)", fontsize=13)

    for ax, vals, label, color in [
        (axes[0], pear,  "Pearson r",   "#55A868"),
        (axes[1], spear, "Spearman ρ",  "#C44E52"),
    ]:
        ax.hist(vals, bins=30, color=color, edgecolor="white", linewidth=0.5)
        med = np.nanmedian(vals)
        ax.axvline(med, color="black", lw=1.5, linestyle="--", label=f"median = {med:.3f}")
        ax.axvline(0, color="gray", lw=0.8, linestyle=":")
        ax.set_xlabel(label)
        ax.set_ylabel("Neuron count")
        ax.legend(fontsize=9)
        ax.set_title(label)

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_correlations_distribution.svg")


def plot_pseudo_r2_vs_pearson(summary, out_dir, patient):
    pr2 = summary["pseudo_r2_mean"].values
    pear = summary["pearson_corr_mean"].values

    has_pval = "p_val_ll_xshuf" in summary.columns and not summary["p_val_ll_xshuf"].isna().all()

    fig, ax = plt.subplots(figsize=(6, 5))

    if has_pval:
        pval = summary["p_val_ll_xshuf"].values
        sig = pval < 0.05
        ax.scatter(pear[~sig], pr2[~sig], s=25, alpha=0.5, color="gray", label="n.s.")
        sc = ax.scatter(pear[sig], pr2[sig], s=35, alpha=0.8,
                        c=np.log10(pval[sig] + 1e-6), cmap="plasma_r", label="p<0.05")
        fig.colorbar(sc, ax=ax, label="log₁₀(shuffle p-value)")
    else:
        ax.scatter(pear, pr2, s=25, alpha=0.6, color="#4C72B0")

    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.axvline(0, color="black", lw=0.8, linestyle=":")
    ax.set_xlabel("Pearson r (mean CV)")
    ax.set_ylabel("Pseudo-R² (mean CV)")
    ax.set_title(f"{patient} — Pseudo-R² vs Pearson r")
    if has_pval:
        ax.legend(fontsize=9)

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_pseudo_r2_vs_pearson.svg")


def plot_significance(summary, out_dir, patient):
    if "p_val_ll_xshuf" not in summary.columns or summary["p_val_ll_xshuf"].isna().all():
        print("  no shuffle p-values — skipping significance plot")
        return

    pr2 = summary["pseudo_r2_mean"].values
    pval = summary["p_val_ll_xshuf"].values
    order = np.argsort(pr2)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{patient} — Shuffle significance (n=50 permutations)", fontsize=13)

    # sorted pseudo-R² coloured by significance
    ax = axes[0]
    x = np.arange(len(pr2))
    colors = ["#C44E52" if p < 0.05 else "#AAAAAA" for p in pval[order]]
    ax.bar(x, pr2[order], color=colors, width=1.0, linewidth=0)
    ax.axhline(0, color="black", lw=0.8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#C44E52", label="p<0.05"),
                        Patch(color="#AAAAAA", label="n.s.")], fontsize=9)
    ax.set_xlabel("Neuron rank")
    ax.set_ylabel("Pseudo-R²")
    ax.set_title(f"Sorted neurons — {(pval < 0.05).sum()}/{len(pval)} significant")

    # p-value histogram
    ax = axes[1]
    ax.hist(pval, bins=20, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(0.05, color="tomato", lw=1.5, linestyle="--", label="p=0.05")
    ax.set_xlabel("Shuffle p-value")
    ax.set_ylabel("Neuron count")
    ax.legend(fontsize=9)
    ax.set_title("Distribution of p-values")

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_significance.svg")


def plot_best_alpha_distribution(summary, out_dir, patient):
    if "best_alpha_mean" not in summary.columns:
        return
    alpha = summary["best_alpha_mean"].values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(alpha + 1e-10), bins=20, color="#8172B2", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("log₁₀(best α)")
    ax.set_ylabel("Neuron count")
    ax.set_title(f"{patient} — Tuned ridge penalty distribution")
    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_best_alpha_distribution.svg")


def plot_edf_vs_pseudo_r2(summary, out_dir, patient):
    if "edf_mean" not in summary.columns:
        return
    edf = summary["edf_mean"].values
    pr2 = summary["pseudo_r2_mean"].values

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(edf, pr2, s=25, alpha=0.6, color="#4C72B0")
    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.set_xlabel("Effective degrees of freedom (EDF)")
    ax.set_ylabel("Pseudo-R² (mean CV)")
    ax.set_title(f"{patient} — EDF vs Pseudo-R²")
    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_edf_vs_pseudo_r2.svg")


def plot_summary_panel(summary, out_dir, patient):
    """Single-figure summary panel: PR2 sorted bar, PR2 histogram, Pearson hist, scatter."""
    pr2 = summary["pseudo_r2_mean"].values
    pr2_std = summary["pseudo_r2_std"].values
    pear = summary["pearson_corr_mean"].values
    spear = summary["spearman_corr_mean"].values

    has_pval = "p_val_ll_xshuf" in summary.columns and not summary["p_val_ll_xshuf"].isna().all()
    pval = summary["p_val_ll_xshuf"].values if has_pval else None

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{patient} — GPT-2 → Firing Rate (Poisson GLM, 5-fold nested CV)", fontsize=14, y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Sorted pseudo-R² bar
    ax1 = fig.add_subplot(gs[0, :2])
    order = np.argsort(pr2)[::-1]
    x = np.arange(len(pr2))
    if has_pval:
        colors = ["#C44E52" if pval[order[i]] < 0.05 else "#AAAAAA" for i in range(len(pr2))]
    else:
        colors = "#4C72B0"
    ax1.bar(x, pr2[order], color=colors, width=1.0, linewidth=0)
    ax1.errorbar(x, pr2[order], yerr=pr2_std[order],
                 fmt="none", ecolor="black", elinewidth=0.4, capsize=0, alpha=0.4)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xlabel("Neuron rank")
    ax1.set_ylabel("Pseudo-R²")
    title = f"Sorted neurons (n={len(pr2)})"
    if has_pval:
        title += f" — {(pval < 0.05).sum()} significant (p<0.05)"
    ax1.set_title(title)

    # 2. Pseudo-R² histogram
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(pr2, bins=25, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax2.axvline(np.nanmedian(pr2), color="tomato", lw=1.5, linestyle="--",
                label=f"med={np.nanmedian(pr2):.3f}")
    ax2.axvline(0, color="black", lw=0.8, linestyle=":")
    ax2.set_xlabel("Pseudo-R²")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)
    ax2.set_title("Pseudo-R² dist.")

    # 3. Pearson histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(pear, bins=25, color="#55A868", edgecolor="white", linewidth=0.5)
    ax3.axvline(np.nanmedian(pear), color="black", lw=1.5, linestyle="--",
                label=f"med={np.nanmedian(pear):.3f}")
    ax3.axvline(0, color="gray", lw=0.8, linestyle=":")
    ax3.set_xlabel("Pearson r")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=8)
    ax3.set_title("Pearson corr. dist.")

    # 4. Spearman histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(spear, bins=25, color="#C44E52", edgecolor="white", linewidth=0.5)
    ax4.axvline(np.nanmedian(spear), color="black", lw=1.5, linestyle="--",
                label=f"med={np.nanmedian(spear):.3f}")
    ax4.axvline(0, color="gray", lw=0.8, linestyle=":")
    ax4.set_xlabel("Spearman ρ")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=8)
    ax4.set_title("Spearman corr. dist.")

    # 5. Pseudo-R² vs Pearson scatter
    ax5 = fig.add_subplot(gs[1, 2])
    if has_pval:
        sig = pval < 0.05
        ax5.scatter(pear[~sig], pr2[~sig], s=20, alpha=0.4, color="gray")
        ax5.scatter(pear[sig], pr2[sig], s=30, alpha=0.8, color="#C44E52")
    else:
        ax5.scatter(pear, pr2, s=20, alpha=0.6, color="#4C72B0")
    ax5.axhline(0, color="black", lw=0.8, linestyle=":")
    ax5.axvline(0, color="black", lw=0.8, linestyle=":")
    ax5.set_xlabel("Pearson r")
    ax5.set_ylabel("Pseudo-R²")
    ax5.set_title("Pseudo-R² vs Pearson")

    savefig(fig, out_dir / f"{patient}_summary_panel.svg")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--patient", type=str, default=None)
    args = parser.parse_args()

    pkl_path = args.pkl_path
    out_dir = args.out_dir if args.out_dir is not None else pkl_path.parent
    patient = args.patient if args.patient is not None else pkl_path.stem.split("_encoding")[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"patient: {patient}")
    print(f"pkl:     {pkl_path}")
    print(f"out_dir: {out_dir}")

    _, summary = load_summary(pkl_path)
    print(f"neurons: {len(summary)}")
    print(f"pseudo-R² — median={np.nanmedian(summary['pseudo_r2_mean']):.4f}  "
          f"mean={np.nanmean(summary['pseudo_r2_mean']):.4f}")

    plot_summary_panel(summary, out_dir, patient)
    plot_pseudo_r2_distribution(summary, out_dir, patient)
    plot_correlations_distribution(summary, out_dir, patient)
    plot_pseudo_r2_vs_pearson(summary, out_dir, patient)
    plot_significance(summary, out_dir, patient)
    plot_best_alpha_distribution(summary, out_dir, patient)
    plot_edf_vs_pseudo_r2(summary, out_dir, patient)

    print("done")


if __name__ == "__main__":
    main()
