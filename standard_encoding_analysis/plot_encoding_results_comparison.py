"""
Compare Poisson GLM encoding results: all-speech vs patient-speech-only.

Usage:
    python plot_encoding_results_comparison.py <all_speech_pkl> <patient_speech_pkl>
        [--out-dir OUT_DIR]
        [--patient PATIENT]

Figures produced (SVG):
    1. pseudo_r2_comparison_panel.svg  — histograms + sorted bars side-by-side
    2. pseudo_r2_scatter.svg           — per-neuron scatter: all-speech vs patient-speech PR²
    3. pearson_comparison_panel.svg    — same for Pearson r
    4. delta_pseudo_r2.svg             — Δ(patient − all) sorted bar; highlights gainers/losers
    5. summary_stats.svg               — table of population-level statistics
"""

import argparse
from pathlib import Path

import dill as pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def load_summary(pkl_path: Path) -> pd.DataFrame:
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    summary = df[df["is_summary"] == True].copy().reset_index(drop=True)
    return summary.sort_values("neuron_idx").reset_index(drop=True)


def savefig(fig, path: Path):
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def align_summaries(s_all: pd.DataFrame, s_ps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only neurons present in both summaries, aligned by neuron_idx."""
    common = sorted(set(s_all["neuron_idx"]) & set(s_ps["neuron_idx"]))
    s_all = s_all[s_all["neuron_idx"].isin(common)].sort_values("neuron_idx").reset_index(drop=True)
    s_ps  = s_ps[s_ps["neuron_idx"].isin(common)].sort_values("neuron_idx").reset_index(drop=True)
    return s_all, s_ps


# ── individual plots ──────────────────────────────────────────────────────────

def plot_pseudo_r2_comparison_panel(s_all, s_ps, out_dir, patient):
    pr2_all = s_all["pseudo_r2_mean"].values
    pr2_ps  = s_ps["pseudo_r2_mean"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{patient} — Pseudo-R²: All Speech vs Patient Speech Only", fontsize=14)

    COLORS = {"all": "#4C72B0", "patient": "#C44E52"}
    LABELS = {"all": "All speech", "patient": "Patient speech only"}

    # Row 0: histograms
    for ax, vals, key in [(axes[0, 0], pr2_all, "all"), (axes[0, 1], pr2_ps, "patient")]:
        ax.hist(vals, bins=30, color=COLORS[key], edgecolor="white", linewidth=0.5)
        med = np.nanmedian(vals)
        ax.axvline(med, color="black", lw=1.5, linestyle="--",
                   label=f"median={med:.3f}")
        ax.axvline(0, color="gray", lw=0.8, linestyle=":")
        ax.set_xlabel("Pseudo-R²")
        ax.set_ylabel("Neuron count")
        ax.set_title(LABELS[key])
        ax.legend(fontsize=9)

    # Row 1: sorted bars
    for ax, vals, key in [(axes[1, 0], pr2_all, "all"), (axes[1, 1], pr2_ps, "patient")]:
        order = np.argsort(vals)[::-1]
        x = np.arange(len(vals))
        std_col = "pseudo_r2_std"
        errs = s_all[std_col].values[order] if key == "all" else s_ps[std_col].values[order]
        ax.bar(x, vals[order], color=COLORS[key], width=1.0, linewidth=0)
        ax.errorbar(x, vals[order], yerr=errs,
                    fmt="none", ecolor="black", elinewidth=0.4, capsize=0, alpha=0.4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Neuron rank")
        ax.set_ylabel("Pseudo-R²")
        ax.set_title(f"{LABELS[key]} — sorted")

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_pseudo_r2_comparison_panel.svg")


def plot_pseudo_r2_scatter(s_all, s_ps, out_dir, patient):
    pr2_all = s_all["pseudo_r2_mean"].values
    pr2_ps  = s_ps["pseudo_r2_mean"].values

    has_pval_all = ("p_val_ll_xshuf" in s_all.columns
                    and not s_all["p_val_ll_xshuf"].isna().all())
    has_pval_ps  = ("p_val_ll_xshuf" in s_ps.columns
                    and not s_ps["p_val_ll_xshuf"].isna().all())

    fig, ax = plt.subplots(figsize=(6, 6))

    if has_pval_all or has_pval_ps:
        sig_all = s_all["p_val_ll_xshuf"].values < 0.05 if has_pval_all else np.zeros(len(pr2_all), bool)
        sig_ps  = s_ps["p_val_ll_xshuf"].values < 0.05  if has_pval_ps  else np.zeros(len(pr2_ps),  bool)
        sig_both = sig_all & sig_ps
        ax.scatter(pr2_all[~sig_both], pr2_ps[~sig_both],
                   s=25, alpha=0.4, color="gray", label="n.s. in both")
        ax.scatter(pr2_all[sig_both], pr2_ps[sig_both],
                   s=40, alpha=0.8, color="#C44E52", label="sig. in both (p<0.05)")
        ax.legend(fontsize=9)
    else:
        ax.scatter(pr2_all, pr2_ps, s=25, alpha=0.6, color="#4C72B0")

    lims = [
        min(np.nanmin(pr2_all), np.nanmin(pr2_ps)) - 0.02,
        max(np.nanmax(pr2_all), np.nanmax(pr2_ps)) + 0.02,
    ]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Pseudo-R² — all speech")
    ax.set_ylabel("Pseudo-R² — patient speech only")
    ax.set_title(f"{patient} — Per-neuron Pseudo-R² comparison\n"
                 f"(n={len(pr2_all)} neurons, diagonal = equal performance)")

    # annotate with correlation
    valid = ~(np.isnan(pr2_all) | np.isnan(pr2_ps))
    if valid.sum() > 2:
        from scipy.stats import pearsonr
        r, _ = pearsonr(pr2_all[valid], pr2_ps[valid])
        ax.text(0.05, 0.93, f"Pearson r = {r:.3f}", transform=ax.transAxes, fontsize=10)

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_pseudo_r2_scatter.svg")


def plot_pearson_comparison_panel(s_all, s_ps, out_dir, patient):
    pear_all = s_all["pearson_corr_mean"].values
    pear_ps  = s_ps["pearson_corr_mean"].values

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"{patient} — Pearson r: All Speech vs Patient Speech Only", fontsize=14)

    COLORS = {"all": "#4C72B0", "patient": "#C44E52"}

    for ax, vals, key in [(axes[0], pear_all, "all"), (axes[1], pear_ps, "patient")]:
        ax.hist(vals, bins=30, color=COLORS[key], edgecolor="white", linewidth=0.5)
        med = np.nanmedian(vals)
        ax.axvline(med, color="black", lw=1.5, linestyle="--", label=f"med={med:.3f}")
        ax.axvline(0, color="gray", lw=0.8, linestyle=":")
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("Neuron count")
        ax.set_title("All speech" if key == "all" else "Patient speech only")
        ax.legend(fontsize=9)

    # Scatter
    ax = axes[2]
    ax.scatter(pear_all, pear_ps, s=20, alpha=0.6, color="#4C72B0")
    lims = [
        min(np.nanmin(pear_all), np.nanmin(pear_ps)) - 0.02,
        max(np.nanmax(pear_all), np.nanmax(pear_ps)) + 0.02,
    ]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("All speech")
    ax.set_ylabel("Patient speech only")
    ax.set_title("Per-neuron scatter")

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_pearson_comparison_panel.svg")


def plot_delta_pseudo_r2(s_all, s_ps, out_dir, patient):
    pr2_all = s_all["pseudo_r2_mean"].values
    pr2_ps  = s_ps["pseudo_r2_mean"].values
    delta   = pr2_ps - pr2_all

    order = np.argsort(delta)[::-1]
    x     = np.arange(len(delta))
    colors = ["#C44E52" if d > 0 else "#4C72B0" for d in delta[order]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{patient} — Δ Pseudo-R² (patient speech − all speech)", fontsize=14)

    ax = axes[0]
    ax.bar(x, delta[order], color=colors, width=1.0, linewidth=0)
    ax.axhline(0, color="black", lw=0.8)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#C44E52", label="patient > all (gain)"),
        Patch(color="#4C72B0", label="patient < all (loss)"),
    ], fontsize=9)
    ax.set_xlabel("Neuron rank (by Δ)")
    ax.set_ylabel("Δ Pseudo-R²")
    n_gain = (delta > 0).sum()
    ax.set_title(f"Sorted Δ — {n_gain}/{len(delta)} neurons gain with patient-only subset")

    ax = axes[1]
    ax.hist(delta, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    med = np.nanmedian(delta)
    ax.axvline(med, color="tomato", lw=1.5, linestyle="--", label=f"median Δ={med:.4f}")
    ax.axvline(0, color="black", lw=0.8, linestyle=":")
    ax.set_xlabel("Δ Pseudo-R²")
    ax.set_ylabel("Neuron count")
    ax.legend(fontsize=9)
    ax.set_title("Distribution of Δ")

    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_delta_pseudo_r2.svg")


def plot_summary_stats(s_all, s_ps, out_dir, patient):
    from scipy.stats import wilcoxon

    metrics = [
        ("pseudo_r2_mean",       "Pseudo-R²"),
        ("pearson_corr_mean",    "Pearson r"),
        ("spearman_corr_mean",   "Spearman ρ"),
    ]

    rows = []
    for col, label in metrics:
        if col not in s_all.columns or col not in s_ps.columns:
            continue
        a = s_all[col].dropna().values
        p = s_ps[col].dropna().values
        # paired Wilcoxon on common neurons (already aligned)
        n = min(len(a), len(p))
        a, p = a[:n], p[:n]
        try:
            stat, pval = wilcoxon(a, p)
        except Exception:
            pval = np.nan
        rows.append({
            "Metric":         label,
            "All — median":   f"{np.nanmedian(a):.4f}",
            "All — mean":     f"{np.nanmean(a):.4f}",
            "Patient — median": f"{np.nanmedian(p):.4f}",
            "Patient — mean":   f"{np.nanmean(p):.4f}",
            "Δ median":       f"{np.nanmedian(p) - np.nanmedian(a):+.4f}",
            "Wilcoxon p":     f"{pval:.3e}" if not np.isnan(pval) else "n/a",
        })

    df_stats = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 1.5 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_stats.values,
        colLabels=df_stats.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    ax.set_title(f"{patient} — Population summary (n={len(s_all)} neurons)", fontsize=12, pad=10)
    fig.tight_layout()
    savefig(fig, out_dir / f"{patient}_summary_stats.svg")

    print(df_stats.to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("all_speech_pkl",     type=Path,
                        help="Result PKL from the all-speech encoding run")
    parser.add_argument("patient_speech_pkl", type=Path,
                        help="Result PKL from the patient-speech-only encoding run")
    parser.add_argument("--out-dir",  type=Path, default=None)
    parser.add_argument("--patient",  type=str,  default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.patient_speech_pkl.parent
    patient = args.patient or args.patient_speech_pkl.stem.split("_encoding")[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"patient:           {patient}")
    print(f"all-speech pkl:    {args.all_speech_pkl}")
    print(f"patient-speech pkl:{args.patient_speech_pkl}")
    print(f"out_dir:           {out_dir}")

    s_all = load_summary(args.all_speech_pkl)
    s_ps  = load_summary(args.patient_speech_pkl)
    print(f"neurons — all: {len(s_all)}  patient: {len(s_ps)}")

    s_all, s_ps = align_summaries(s_all, s_ps)
    print(f"aligned neurons: {len(s_all)}")

    plot_pseudo_r2_comparison_panel(s_all, s_ps, out_dir, patient)
    plot_pseudo_r2_scatter(s_all, s_ps, out_dir, patient)
    plot_pearson_comparison_panel(s_all, s_ps, out_dir, patient)
    plot_delta_pseudo_r2(s_all, s_ps, out_dir, patient)
    plot_summary_stats(s_all, s_ps, out_dir, patient)

    print("done")


if __name__ == "__main__":
    main()
