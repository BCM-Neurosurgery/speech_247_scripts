import json, copy

with open('figure_4.ipynb') as f:
    nb_old = json.load(f)
old_cells = nb_old['cells']

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": [src]}

# ── helpers to fetch existing cells ──────────────────────────────────────────
def old(i): return ''.join(old_cells[i]['source'])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 0: Title
# ─────────────────────────────────────────────────────────────────────────────
c00 = md_cell(
"""# Figure 4 — Decoding Performance

**Main panels:**
- **A** — per-patient F1-micro (whole-stay, filtered speech)
- **B** — paired line: filtered vs patient-speech decoding
- **D+E** — 24/7 vs conversation & podcast comparisons (combined)
- **G** — whole-stay vs per-day paired line (filtered speech)
- **C** — decoder F1-micro vs population size (per-day × resample dots, WLS)
- **E_epoch** — pooled epoch scatter: F1-micro vs patient-speech fraction (WLS)
- **F** — pie charts: fraction of patients with positive / sig. positive WLS trend
- **CV** — encoding vs decoding dispersion comparison (CV, Feltz–Miller test)

**Supplementary:**  S1 per-patient epoch WLS scatter | S2–S5 heatmaps & epoch line plots"""
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: Imports md (keep)
# ─────────────────────────────────────────────────────────────────────────────
c01 = md_cell(old(1))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: Imports + updated COL dict + WLS utilities
# ─────────────────────────────────────────────────────────────────────────────
c02 = code_cell(
"""import json
import warnings
from pathlib import Path

import dill as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from scipy.stats import chi2

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'svg.fonttype':        'none',
    'font.family':         'sans-serif',
    'font.sans-serif':     ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.linewidth':      0.8,
    'xtick.major.width':   0.8,
    'ytick.major.width':   0.8,
    'font.size':           9,
    'axes.labelsize':      9,
    'axes.titlesize':      10,
    'xtick.labelsize':     8,
    'ytick.labelsize':     8,
    'legend.fontsize':     8,
    'figure.dpi':          150,
})

COL = {
    # primary condition colours
    'filtered':     '#41b373',   # bright green  — all filtered speech
    'patient':      '#ee2e7b',   # bright pink   — patient speech only
    'patient_mid':  '#f887b1',   # medium pink
    'unfiltered':   '#3e71b8',   # cerulean
    # bar fill colours (3-method comparison)
    'bar0':         '#3e71b8',   # cerulean
    'bar0_light':   '#e0effa',   # sky blue
    'bar1':         '#8a5ba6',   # purple
    'bar1_light':   '#eae0ef',   # light lavender
    'bar2':         '#f18032',   # orange
    'bar2_light':   '#ffebd9',   # baby orange
    'bar3':         '#ee2e7b',   # bright pink
    # per-day lighter variants
    'filtered_day': '#daedde',   # mint green    — per-day filtered
    'patient_day':  '#fbb0cb',   # baby pink     — per-day patient
    # connector / dot colours
    'line_connect': '#cccccc',   # light gray    — paired lines
    'dot_line':     '#8a5ba6',   # purple        — dots in paired line plots
    'dot_bar':      '#f887b1',   # medium pink   — dots within comparison bars
    'dot_scatter':  '#3e71b8',   # cerulean      — epoch/population scatter dots
}
BAR_COLORS = [COL['bar0'], COL['bar1'], COL['bar2'], COL['bar3']]


def savefig(fig, name: str, fig_dir: Path):
    fig.savefig(fig_dir / f'{name}.svg', format='svg', bbox_inches='tight')
    fig.savefig(fig_dir / f'{name}.png', format='png', dpi=120, bbox_inches='tight')


# ── WLS utilities (mirrors Figure 3) ─────────────────────────────────────────
def compute_density_weights(x):
    x = np.asarray(x, float)
    edges = np.histogram_bin_edges(x, bins='fd')
    if len(edges) > 21:
        edges = np.histogram_bin_edges(x, bins=20)
    counts, edges = np.histogram(x, bins=edges)
    w = np.empty(len(x))
    for j in range(len(x)):
        idx = min(np.searchsorted(edges[1:], x[j]), len(counts) - 1)
        w[j] = 1.0 / max(counts[idx], 1)
    return w / w.sum()


def fit_wls(x, y, w):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    slope, intercept = np.polyfit(x, y, 1, w=w)
    xbar = np.average(x, weights=w)
    ybar = np.average(y, weights=w)
    cov  = np.average((x - xbar) * (y - ybar), weights=w)
    vx   = np.average((x - xbar) ** 2, weights=w)
    vy   = np.average((y - ybar) ** 2, weights=w)
    r = cov / np.sqrt(vx * vy) if (vx > 0 and vy > 0) else 0.0
    return float(slope), float(intercept), float(r)


def permutation_pval_wls(x, y, w, n_perm=500, seed=0):
    rng = np.random.default_rng(seed)
    obs_slope, _, _ = fit_wls(x, y, w)
    perm_slopes = np.array([
        fit_wls(rng.permutation(x), y, w)[0]
        for _ in range(n_perm)
    ])
    p = (np.sum(np.abs(perm_slopes) >= abs(obs_slope)) + 1) / (n_perm + 1)
    return float(p), float(obs_slope)


# ── Feltz & Miller (1996) asymptotic test for equality of two CVs ─────────────
def feltz_miller_test(vals1, vals2):
    \"\"\"
    Test H0: CV(vals1) == CV(vals2).
    Returns (chi2_stat, p_value) with 1 degree of freedom.
    Both arrays must have positive mean.
    \"\"\"
    v1 = np.asarray(vals1, float)
    v2 = np.asarray(vals2, float)
    v1 = v1[np.isfinite(v1)]
    v2 = v2[np.isfinite(v2)]
    n1, n2 = len(v1), len(v2)
    cv1 = np.std(v1, ddof=1) / np.mean(v1)
    cv2 = np.std(v2, ddof=1) / np.mean(v2)
    # pooled CV estimate
    cv_pool = ((n1 - 1) * cv1 + (n2 - 1) * cv2) / (n1 + n2 - 2)
    denom = 0.5 + cv_pool ** 2
    sum_nw_logcv  = (n1 - 1) * np.log(cv1) ** 2 + (n2 - 1) * np.log(cv2) ** 2
    sum_nw_logcv1 = (n1 - 1) * np.log(cv1)     + (n2 - 1) * np.log(cv2)
    T = (sum_nw_logcv - sum_nw_logcv1 ** 2 / (n1 + n2 - 2)) / denom
    p = chi2.sf(T, df=1)
    return float(T), float(p)


print('imports + WLS ok')
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: Config md (keep)
# ─────────────────────────────────────────────────────────────────────────────
c03 = md_cell(old(3))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: Config code (add ENC_PE_RUN)
# ─────────────────────────────────────────────────────────────────────────────
src04 = old(4)
# Insert ENC_PE_RUN after PE_PS_RUN line
src04 = src04.replace(
    "PE_PS_RUN = 'scat_xgboost_sampled_norm_patient_per_epoch'",
    "PE_PS_RUN = 'scat_xgboost_sampled_norm_patient_per_epoch'\n\n"
    "# ── Encoding run names (for CV comparison panel) ─────────────────────────────\n"
    "ENC_PE_RUN = 'word_level_duration_cv_filtered_speech_per_epoch'"
)
c04 = code_cell(src04)

# ─────────────────────────────────────────────────────────────────────────────
# CELLS 5–6: Patients (keep, but update PATIENT_COLORS note)
# ─────────────────────────────────────────────────────────────────────────────
c05 = md_cell(old(5))
c06 = code_cell(old(6))  # keeps PATIENT_COLORS for fallback

# ─────────────────────────────────────────────────────────────────────────────
# CELLS 7–8: Load filtered/patient results
# ─────────────────────────────────────────────────────────────────────────────
c07 = md_cell(old(7))
c08 = code_cell(old(8))

# ─────────────────────────────────────────────────────────────────────────────
# CELLS 9–10: Panel A (keep)
# ─────────────────────────────────────────────────────────────────────────────
c09 = md_cell(old(9))
c10 = code_cell(old(10))

# ─────────────────────────────────────────────────────────────────────────────
# CELLS 11–12: Panel B — update draw_paired_lineplot to use COL['dot_line']
# ─────────────────────────────────────────────────────────────────────────────
c11 = md_cell(old(11))
src12 = old(12)
# Replace patient-colored scatter with monotone dot_line
src12 = src12.replace(
    "        pc = PATIENT_COLORS.get(pt, 'gray')\n"
    "        if np.isfinite(lv):\n"
    "            ax.scatter(x_left,  lv, color=pc, s=55, zorder=4,\n"
    "                       edgecolors='white', linewidths=0.8)\n"
    "        if np.isfinite(rv):\n"
    "            ax.scatter(x_right, rv, color=pc, s=55, zorder=4,\n"
    "                       edgecolors='white', linewidths=0.8)",
    "        if np.isfinite(lv):\n"
    "            ax.scatter(x_left,  lv, color=COL['dot_line'], s=55, zorder=4,\n"
    "                       edgecolors='white', linewidths=0.8)\n"
    "        if np.isfinite(rv):\n"
    "            ax.scatter(x_right, rv, color=COL['dot_line'], s=55, zorder=4,\n"
    "                       edgecolors='white', linewidths=0.8)"
)
# Remove patient legend from Panel B
src12 = src12.replace(
    "handles_b = [plt.Line2D([0], [0], marker='o', linestyle='none',\n"
    "                         color=PATIENT_COLORS.get(pt, 'gray'),\n"
    "                         markersize=6, label=pt)\n"
    "             for pt in common_pts_b]\n"
    "ax.legend(handles=handles_b, title='Patient', fontsize=6,\n"
    "          title_fontsize=6.5, frameon=False, loc='lower right', ncol=2)\n",
    ""
)
c12 = code_cell(src12)

# ─────────────────────────────────────────────────────────────────────────────
# CELLS 13–14: Load convo/podcast (keep)
# ─────────────────────────────────────────────────────────────────────────────
c13 = md_cell(old(15))
c14 = code_cell(old(16))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 15: Panel D+E combined md
# ─────────────────────────────────────────────────────────────────────────────
c15 = md_cell(
"""## 7. Panel D+E — 24/7 vs Conversation & Podcast Comparison (Combined)

Mirrors Figure 3 Panel C: 24/7 bar + 3 paired method groups (convo/podcast with same
color per method, different hatch). Dots use `COL['dot_bar']`.
Dotted line = 24/7 group mean baseline."""
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 16: Panel D+E combined code
# ─────────────────────────────────────────────────────────────────────────────
c16 = code_cell(
"""def compute_pt_means_f1(res_dict):
    \"\"\"Per-patient mean F1-micro across resamples. Returns {pt: mean}.\"\"\"
    return {
        pt: float(np.mean(f1s))
        for pt, f1s in res_dict.items()
        if len(f1s) > 0
    }


def f1_mean_sem(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    n = len(a)
    m = float(np.mean(a)) if n > 0 else np.nan
    s = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, s, n


# ── compute per-patient means ─────────────────────────────────────────────────
pt_means_247         = compute_pt_means_f1(res_filtered)
pt_means_convo_m     = compute_pt_means_f1(res_convo_manual)
pt_means_convo_a     = compute_pt_means_f1(res_convo_auto)
pt_means_convo_wx    = compute_pt_means_f1(res_convo_whisperx)
pt_means_podcast_m   = compute_pt_means_f1(res_podcast_manual)
pt_means_podcast_a   = compute_pt_means_f1(res_podcast_auto)
pt_means_podcast_wx  = compute_pt_means_f1(res_podcast_whisperx)

all_comparison_pts = (
    set(pt_means_convo_m) | set(pt_means_convo_a) | set(pt_means_convo_wx) |
    set(pt_means_podcast_m) | set(pt_means_podcast_a) | set(pt_means_podcast_wx)
)
pt_247_subset = {pt: v for pt, v in pt_means_247.items() if pt in all_comparison_pts}

# ── layout constants ──────────────────────────────────────────────────────────
BAR_W         = 0.40
DOT_C         = COL['dot_bar']
x_247         = 0.0
pairs_x       = [(1.6, 2.1), (3.6, 4.1), (5.6, 6.1)]
pair_colors   = [COL['bar0'], COL['bar1'], COL['bar2']]
HATCH_CONVO   = '//'
HATCH_PODC    = '\\\\\\\\'
method_labels = ['Manual\\nPRAAT', 'PRAAT+\\nauto-thresh', 'WhisperX+\\nauto-thresh']
convo_data    = [pt_means_convo_m,  pt_means_convo_a,  pt_means_convo_wx]
podcast_data  = [pt_means_podcast_m, pt_means_podcast_a, pt_means_podcast_wx]

rng_de = np.random.default_rng(42)
fig_de, ax = plt.subplots(figsize=(11, 4.8))


def _draw_bar_f1(ax, xpos, data_dict, color, hatch, rng, bar_w=BAR_W, dot_c=DOT_C):
    vals = np.array([v for v in data_dict.values() if np.isfinite(v)])
    if len(vals) == 0:
        return
    mean = float(np.mean(vals))
    sem  = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    ax.bar(xpos, mean, bar_w, yerr=sem, capsize=4,
           color=color, alpha=0.72, hatch=hatch,
           edgecolor=color, linewidth=0.8, zorder=2,
           error_kw={'ecolor': 'dimgray', 'linewidth': 1.1})
    for v in vals:
        jitter = rng.uniform(-bar_w * 0.32, bar_w * 0.32)
        ax.scatter(xpos + jitter, v, color=dot_c, s=26, zorder=4,
                   alpha=0.85, edgecolors='white', linewidths=0.4)
    ax.text(xpos, mean + sem + 0.001, f'n={len(vals)}',
            ha='center', va='bottom', fontsize=6, color='dimgray')


# 24/7 bar
_draw_bar_f1(ax, x_247, pt_247_subset, COL['filtered'], '//', rng_de)

# paired method bars
for gi, (x_con, x_pod) in enumerate(pairs_x):
    _draw_bar_f1(ax, x_con, convo_data[gi],   pair_colors[gi], HATCH_CONVO, rng_de)
    _draw_bar_f1(ax, x_pod, podcast_data[gi], pair_colors[gi], HATCH_PODC,  rng_de)

# 24/7 baseline dotted line
vals_247_arr = np.array([v for v in pt_247_subset.values() if np.isfinite(v)])
if len(vals_247_arr):
    mean_247_baseline = float(np.mean(vals_247_arr))
    ax.axhline(mean_247_baseline, color=COL['filtered'],
               linestyle=':', linewidth=1.6, zorder=1, alpha=0.8,
               label=f'24/7 mean = {mean_247_baseline:.3f}')

# x-ticks
xtick_pos    = [x_247]
xtick_labels = ['24/7\\n(filtered)']
for x_con, x_pod in pairs_x:
    xtick_pos   += [x_con, x_pod]
    xtick_labels += ['Convo', 'Podcast']
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_labels, ha='center', fontsize=8)

# method group labels
y_top_de = ax.get_ylim()[1]
for gi, (x_con, x_pod) in enumerate(pairs_x):
    ax.text((x_con + x_pod) / 2, y_top_de * 0.98, method_labels[gi],
            ha='center', va='top', fontsize=8.5, fontweight='bold',
            color=pair_colors[gi])

legend_els = [
    Patch(facecolor='white', edgecolor='dimgray', hatch='//',     label='Convo recording'),
    Patch(facecolor='white', edgecolor='dimgray', hatch='\\\\\\\\', label='Podcast recording'),
    plt.Line2D([0], [0], color=COL['filtered'], linestyle=':', lw=1.6, label='24/7 mean'),
]
ax.legend(handles=legend_els, fontsize=8, frameon=False, loc='upper right')
ax.set_ylabel('F1-micro (mean ± SEM across patients)')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_title(
    'D+E  —  24/7 vs conversation & podcast recordings\\n'
    '(dot = patient mean; bar = mean ± SEM; same color = same transcription method)'
)
plt.tight_layout()
savefig(fig_de, 'DE_combined_comparison', FIG_DIR)
plt.show()
print('saved D+E')
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELLS 17–18: Per-day helpers (keep)
# ─────────────────────────────────────────────────────────────────────────────
c17 = md_cell(old(21))
c18 = code_cell(old(22))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 19: Panel G md
# ─────────────────────────────────────────────────────────────────────────────
c19 = md_cell(
"""## 9. Panel G — Whole-Stay vs Per-Day Filtered Speech (Paired Line Plot)

Patient mean per condition; connecting lines show within-patient change;
diamonds = group mean ± SEM. Welch's t-test on all resamples."""
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 20: Panel G code — remove patient legend, keep draw_paired_lineplot
# ─────────────────────────────────────────────────────────────────────────────
c20 = code_cell(
"""# ── load per-day filtered data ────────────────────────────────────────────────
perday_rs_fs = load_perday_resamples(PD_RUN)
print(f'Filtered-speech per-day resamples: {len(perday_rs_fs)} rows  |  '
      f'{perday_rs_fs["patient"].nunique()} patients')

if not perday_rs_fs.empty:
    pts_g = sorted([pt for pt in perday_rs_fs['patient'].unique() if pt in res_filtered],
                   key=lambda pt: float(np.mean(res_filtered[pt])), reverse=True)

    left_g  = {pt: float(np.mean(res_filtered[pt])) for pt in pts_g}
    right_g = {pt: float(perday_rs_fs[perday_rs_fs['patient'] == pt]['f1_micro'].mean())
               for pt in pts_g}

    _rs_left_g  = np.concatenate([res_filtered[pt] for pt in pts_g])
    _rs_right_g = perday_rs_fs[perday_rs_fs['patient'].isin(pts_g)]['f1_micro'].values

    fig_g, ax = plt.subplots(figsize=(3.8, 5.2))
    draw_paired_lineplot(
        ax,
        left_vals=left_g, right_vals=right_g, patient_order=pts_g,
        left_label='Whole-stay\\n(filtered)', right_label='Per-day\\n(pooled resamples)',
        left_color=COL['filtered'], right_color=COL['filtered_day'],
        title=(
            'G  \\u2014  Whole-stay vs per-day (filtered speech)\\n'
            '(dot = patient mean F1; diamond = group mean \\u00b1 SEM;\\n'
            "Welch's t-test on all resamples)"
        ),
        all_left_rs=_rs_left_g, all_right_rs=_rs_right_g,
    )
    plt.tight_layout()
    savefig(fig_g, 'G_paired_wholestay_vs_perday_filtered', FIG_DIR)
    plt.show()
    print('saved G')
else:
    print('Skipping G — no per-day filtered data.')
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 21: Load n_units md
# ─────────────────────────────────────────────────────────────────────────────
c21 = md_cell(
"""## 10. Load Population Size (n_units) and Per-Day Scatter Data

Used by Panel C (decoder F1 vs population size, per-day × resample dots)."""
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 22: Load n_units
# ─────────────────────────────────────────────────────────────────────────────
c22 = code_cell(
"""UNITS_ENC_RUN = 'word_level_duration_cv_all_n'

n_units_map = {}
for pt in ALL_PATIENTS:
    p = VAD_ROOT / pt / 'encoding' / UNITS_ENC_RUN / f'{pt}_encoding_results_cv.pkl'
    if p.exists():
        try:
            df_enc = pickle.load(open(p, 'rb'))
            n_units_map[pt] = int(df_enc['neuron_idx'].nunique())
        except Exception:
            pass

print(f'n_units loaded for {len(n_units_map)} patients: {n_units_map}')
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 23: Panel C per-day scatter md
# ─────────────────────────────────────────────────────────────────────────────
c23 = md_cell(
"""## 11. Panel C — Decoder F1-micro vs Population Size (Per-Day × Resample)

Each dot = one patient × day × resample (up to 20 resamples per day per patient).
No patient coloring. WLS regression with inverse bin-density weights on n_units."""
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 24: Panel C per-day scatter code
# ─────────────────────────────────────────────────────────────────────────────
c24 = code_cell(
"""# build per-day per-resample scatter (patient x day x resample)
scat_c_rows = []
for _, row in perday_rs_fs.iterrows():
    pt = row['patient']
    if pt in n_units_map:
        scat_c_rows.append({'n_units': float(n_units_map[pt]),
                            'f1_micro': float(row['f1_micro']),
                            'patient': pt})
scat_c_df = pd.DataFrame(scat_c_rows)
print(f'Panel C scatter: {len(scat_c_df)} rows, {scat_c_df["patient"].nunique()} patients')

if scat_c_df.empty:
    print('Skipping Panel C — no per-day scatter data.')
else:
    x_c = scat_c_df['n_units'].values
    y_c = scat_c_df['f1_micro'].values

    w_c = compute_density_weights(x_c)
    slope_c, intercept_c, r_c = fit_wls(x_c, y_c, w_c)
    p_c, _ = permutation_pval_wls(x_c, y_c, w_c, n_perm=500, seed=42)

    fig_c, ax_c = plt.subplots(figsize=(5.5, 4.2))
    rng_c = np.random.default_rng(99)
    jitter_c = rng_c.uniform(-0.4, 0.4, size=len(x_c))
    ax_c.scatter(x_c + jitter_c, y_c,
                 color=COL['dot_scatter'], s=10, alpha=0.35,
                 edgecolors='none', zorder=2)

    x_line_c = np.linspace(x_c.min(), x_c.max(), 200)
    ax_c.plot(x_line_c, slope_c * x_line_c + intercept_c,
              color=COL['filtered'], linewidth=2.0, zorder=5, label='WLS fit')

    p_str_c = f'p={p_c:.4f}' if p_c >= 1e-4 else 'p<0.0001'
    ax_c.text(0.03, 0.97,
              f'WLS  r={r_c:.3f}, {p_str_c}\\n'
              f'n={len(x_c)} (patient×day×resample, {scat_c_df["patient"].nunique()} patients)\\n'
              '(weights \\u221d 1/bin density of n_units)',
              ha='left', va='top', transform=ax_c.transAxes,
              fontsize=7.5, color='dimgray',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.88, ec='none'))

    ax_c.set_xlabel('Population size (n units, encoding array)')
    ax_c.set_ylabel('F1-micro (per resample)')
    ax_c.set_title(
        'C  \\u2014  Decoder performance vs population size\\n'
        '(per-day \\u00d7 resample dots; WLS regression)'
    )
    ax_c.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax_c.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    savefig(fig_c, 'C_decoder_vs_n_units_perday', FIG_DIR)
    plt.show()
    print('saved C')
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 25: Per-epoch helpers md
# ─────────────────────────────────────────────────────────────────────────────
c25 = md_cell(
"""## 12. Per-Epoch Helpers — Loaders for Decoding & Encoding Epoch Data

`load_perepoch` — per-epoch decoding (F1-micro mean/SEM across 20 resamples).
`load_perepoch_resamples` — per-resample f1_micro (for scatter & CV).
`load_enc_perepoch` — per-epoch encoding pseudo-R² (for CV comparison panel).
`load_enc_r2_vals` — all individual sig-unit pseudo-R² values (for CV computation).
`load_transcript_fractions` — patient-speech fraction per epoch window."""
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 26: Per-epoch helpers code
# ─────────────────────────────────────────────────────────────────────────────
# Keep existing load_perepoch + load_perepoch_resamples + load_transcript_fractions
# Add load_enc_perepoch and load_enc_r2_vals
existing_epoch_helpers = old(32)  # has load_perepoch + load_perepoch_resamples + load_transcript_fractions

enc_helpers = """

# ── Encoding per-epoch helpers (for CV comparison) ────────────────────────────
def _load_enc_pkl(pkl_path):
    try:
        df = pickle.load(open(pkl_path, 'rb'))
        return df
    except Exception:
        return pd.DataFrame()


def _enc_sig_filter(df):
    if df.empty:
        return df
    if 'is_summary' in df.columns:
        return df[(df['is_summary'] == True) &
                  (df['p_val_ll_xshuf'] < 0.05) &
                  (df['pseudo_r2_mean'] > 0)]
    return pd.DataFrame()


def load_enc_perepoch(run_name):
    \"\"\"Per-epoch encoding: mean pseudo-R² across sig units per epoch.\"\"\"
    records = []
    for pt in ALL_PATIENTS:
        base = VAD_ROOT / pt / 'encoding' / run_name
        if not base.exists():
            continue
        for date_dir in sorted(base.iterdir()):
            if not date_dir.is_dir():
                continue
            for epoch_dir in sorted(date_dir.iterdir()):
                if not epoch_dir.is_dir():
                    continue
                pkl = epoch_dir / f'{pt}_encoding_results_cv.pkl'
                if not pkl.exists():
                    continue
                df = _load_enc_pkl(pkl)
                sig = _enc_sig_filter(df)
                r2s = sig['pseudo_r2_mean'].values if len(sig) > 0 else np.array([])
                r2_mean = float(np.mean(r2s)) if len(r2s) > 0 else np.nan
                r2_sem  = float(np.std(r2s, ddof=1) / np.sqrt(len(r2s))) if len(r2s) > 1 else 0.0
                epoch_str     = epoch_dir.name
                epoch_start_h = int(epoch_str.split('-')[0])
                records.append({
                    'patient': pt, 'date': date_dir.name,
                    'epoch': epoch_str, 'epoch_start_h': epoch_start_h,
                    'r2_mean': r2_mean, 'r2_sem': r2_sem, 'n_units': len(r2s),
                })
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out['date'] = pd.to_datetime(out['date'])
    def _add_hours_enc(g):
        g = g.sort_values(['date', 'epoch_start_h'])
        first_dt = g['date'].min()
        g['hours_from_start'] = (
            (g['date'] - first_dt).dt.total_seconds() / 3600 + g['epoch_start_h']
        ).round(1)
        return g
    return out.groupby('patient', group_keys=False).apply(_add_hours_enc).reset_index(drop=True)


def load_enc_r2_vals(run_name):
    \"\"\"Collect all individual sig-unit pseudo-R² values (for CV computation).\"\"\"
    r2_all = []
    for pt in ALL_PATIENTS:
        base = VAD_ROOT / pt / 'encoding' / run_name
        if not base.exists():
            continue
        for date_dir in sorted(base.iterdir()):
            if not date_dir.is_dir():
                continue
            for epoch_dir in sorted(date_dir.iterdir()):
                if not epoch_dir.is_dir():
                    continue
                pkl = epoch_dir / f'{pt}_encoding_results_cv.pkl'
                if not pkl.exists():
                    continue
                df = _load_enc_pkl(pkl)
                sig = _enc_sig_filter(df)
                if len(sig) > 0:
                    r2_all.extend(sig['pseudo_r2_mean'].values.tolist())
    return np.array(r2_all, float)
"""

c26 = code_cell(existing_epoch_helpers + enc_helpers)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 27: Load epoch data md
# ─────────────────────────────────────────────────────────────────────────────
c27 = md_cell("## 13. Load Per-Epoch Decoding Data (for Supp S1 scatter & Panels E, F)")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 28: Load epoch data code (keep from cell 34)
# ─────────────────────────────────────────────────────────────────────────────
c28 = code_cell(old(34))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 29: Supp header
# ─────────────────────────────────────────────────────────────────────────────
c29 = md_cell(
"""---
# Supplementary Panels

| Panel | Description |
|-------|-------------|
| S1    | Per-patient epoch scatter: F1-micro vs patient-speech fraction (WLS) |
| S2    | Per-day heatmap (filtered speech) |
| S3    | Per-epoch line plot (filtered speech) |
| S4    | Per-day heatmap (patient speech only) |
| S5    | Per-epoch line plot (patient speech only) |"""
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 30: Supp S1 (old J with WLS) md
# ─────────────────────────────────────────────────────────────────────────────
c30 = md_cell(
"""## Supplementary S1 — Per-Patient Epoch Scatter: F1-micro vs Patient-Speech Fraction (WLS)

One subplot per patient. Dots = per-resample-epoch scores.
WLS fit with inverse bin-density weights. Populates `patient_epoch_stats` for Panel F pie charts."""
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 31: Supp S1 code (old J rewritten with WLS)
# ─────────────────────────────────────────────────────────────────────────────
c31 = code_cell(
"""patient_epoch_stats = {}  # filled here, used by Panel F pie charts

if scatter_df.empty:
    print('Skipping S1 — no scatter data.')
else:
    patients_s1 = sorted(scatter_df['patient'].unique())
    N_s1   = len(patients_s1)
    n_cols = min(3, N_s1)
    n_rows = int(np.ceil(N_s1 / n_cols))

    fig_s1, axes_s1 = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4.5, n_rows * 3.8),
        squeeze=False,
    )
    axes_s1_flat = axes_s1.flatten()
    for ax in axes_s1_flat:
        ax.axis('off')

    for i, pt in enumerate(patients_s1):
        ax = axes_s1_flat[i]
        ax.axis('on')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        pt_data  = scatter_df[scatter_df['patient'] == pt]
        x_vals   = pt_data['patient_frac'].values
        y_vals   = pt_data['f1_micro'].values
        n_epochs = pt_data.groupby(['date', 'epoch']).ngroups

        ax.scatter(x_vals, y_vals, color=COL['dot_scatter'], alpha=0.35, s=14,
                   edgecolors='none', zorder=3)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.6, zorder=1)

        stats_pt = {'n': len(x_vals), 'n_epochs': n_epochs,
                    'r_wls': np.nan, 'p_wls': np.nan}

        if len(x_vals) >= 4:
            w_pt = compute_density_weights(x_vals)
            slope_pt, intercept_pt, r_pt = fit_wls(x_vals, y_vals, w_pt)
            p_pt, _ = permutation_pval_wls(x_vals, y_vals, w_pt, n_perm=500, seed=i)
            stats_pt.update({'r_wls': r_pt, 'p_wls': p_pt})

            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_line, slope_pt * x_line + intercept_pt,
                    color=COL['filtered'], linewidth=1.8, zorder=4)

            p_str = f'p={p_pt:.3f}' if p_pt >= 0.001 else 'p<0.001'
            ax.text(0.05, 0.97,
                    f'WLS r={r_pt:.3f}\\n{p_str}\\nn={len(x_vals)} ({n_epochs} epochs)',
                    ha='left', va='top', transform=ax.transAxes,
                    fontsize=7, color='dimgray',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, ec='none'))

        patient_epoch_stats[pt] = stats_pt

        # smart y-axis
        y_finite_pt = y_vals[np.isfinite(y_vals)]
        if len(y_finite_pt) > 3:
            q75_pt, q25_pt = np.percentile(y_finite_pt, [75, 25])
            y_top_pt = max(q75_pt + 3.0 * (q75_pt - q25_pt), 0.20)
        else:
            y_top_pt = 1.0
        ax.set_ylim(bottom=-0.25, top=y_top_pt)

        ax.set_xlabel('Patient-speech fraction', fontsize=7.5)
        ax.set_ylabel('F1-micro (per resample)', fontsize=7.5)
        ax.set_title(pt, fontsize=8.5, fontweight='bold')

    for ax in axes_s1_flat[N_s1:]:
        ax.axis('off')

    fig_s1.suptitle(
        'S1  —  Per-patient epoch F1-micro vs patient-speech fraction (WLS)',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    savefig(fig_s1, 'S1_per_patient_epoch_wls_scatter', FIG_DIR)
    plt.show()
    print(f'saved S1 ({N_s1} patients)')
    print('patient_epoch_stats:', {pt: {k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in s.items()}
                                    for pt, s in patient_epoch_stats.items()})
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 32: Panel E pooled scatter md
# ─────────────────────────────────────────────────────────────────────────────
c32 = md_cell(
"""## 14. Panel E — Pooled Epoch Scatter: F1-micro vs Patient-Speech Fraction (WLS)

All patients' per-resample-epoch dots pooled. Single WLS fit.
Y-axis: smart IQR-based top limit, bottom = −0.25."""
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 33: Panel E pooled scatter code
# ─────────────────────────────────────────────────────────────────────────────
c33 = code_cell(
"""if scatter_df.empty:
    print('Skipping E — no scatter data.')
    fig_e_pool = None
else:
    x_all = scatter_df['patient_frac'].values
    y_all = scatter_df['f1_micro'].values

    w_all = compute_density_weights(x_all)
    slope_e, intercept_e, r_e = fit_wls(x_all, y_all, w_all)
    p_e, _ = permutation_pval_wls(x_all, y_all, w_all, n_perm=500, seed=999)

    fig_e_pool, ax_e = plt.subplots(figsize=(6.5, 5.2))
    ax_e.scatter(x_all, y_all, color=COL['dot_scatter'],
                 alpha=0.30, s=10, edgecolors='none', zorder=2)
    ax_e.axhline(0, color='gray', linestyle='--', linewidth=0.7, zorder=1)

    x_line_e = np.linspace(x_all.min(), x_all.max(), 200)
    ax_e.plot(x_line_e, slope_e * x_line_e + intercept_e,
              color=COL['filtered'], linewidth=2.2, zorder=5, label='WLS fit')

    p_str_e = f'p={p_e:.4f}' if p_e >= 1e-4 else 'p<0.0001'
    n_pts_e = len(x_all)
    ax_e.text(0.03, 0.97,
              f'WLS  r={r_e:.3f}, {p_str_e}\\n'
              f'n={n_pts_e} resample-epochs ({scatter_df["patient"].nunique()} patients)\\n'
              '(weights \\u221d 1/bin density of speech fraction)',
              ha='left', va='top', transform=ax_e.transAxes,
              fontsize=8, color='dimgray',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.88, ec='none'))

    y_finite_e = y_all[np.isfinite(y_all)]
    if len(y_finite_e) > 3:
        q75_e, q25_e = np.percentile(y_finite_e, [75, 25])
        y_top_e = max(q75_e + 3.0 * (q75_e - q25_e), 0.30)
    else:
        y_top_e = 1.0
    ax_e.set_ylim(bottom=-0.25, top=y_top_e)

    ax_e.set_xlabel('Patient-speech fraction')
    ax_e.set_ylabel('F1-micro (per resample)')
    ax_e.set_title(
        'E  \\u2014  Epoch decoding vs patient-speech fraction (all patients pooled)\\n'
        '(WLS; weights \\u221d 1/bin density)'
    )
    ax_e.legend(fontsize=8, frameon=False, loc='lower right')
    plt.tight_layout()
    savefig(fig_e_pool, 'E_pooled_epoch_wls_scatter', FIG_DIR)
    plt.show()
    print('saved E')
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 34: Panel F pie charts md
# ─────────────────────────────────────────────────────────────────────────────
c34 = md_cell(
"""## 15. Panel F — Per-Patient Epoch WLS Summary (Pie Charts)

Left pie: % patients with significant positive WLS trend (r > 0 AND p < 0.05).
Right pie: % patients with any positive WLS trend (r > 0)."""
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 35: Panel F pie charts code
# ─────────────────────────────────────────────────────────────────────────────
c35 = code_cell(
"""pts_stat = [pt for pt, s in patient_epoch_stats.items()
             if not np.isnan(s.get('r_wls', np.nan))]
n_total = len(pts_stat)

if n_total == 0:
    print('No patients with WLS fits — skipping Panel F.')
else:
    n_pos_sig = sum(
        patient_epoch_stats[pt]['r_wls'] > 0 and
        patient_epoch_stats[pt]['p_wls'] < 0.05
        for pt in pts_stat
    )
    n_pos = sum(patient_epoch_stats[pt]['r_wls'] > 0 for pt in pts_stat)

    fig_f, axes_f = plt.subplots(1, 2, figsize=(7, 4))

    for ax_pi, n_yes, title_pi in [
        (axes_f[0], n_pos_sig,
         f'Sig. positive WLS trend\\n(r>0 & p<0.05)\\n{n_pos_sig}/{n_total} patients'),
        (axes_f[1], n_pos,
         f'Any positive WLS trend\\n(r>0)\\n{n_pos}/{n_total} patients'),
    ]:
        sizes = [n_yes, n_total - n_yes]
        colors_pi = [COL['filtered'], COL['bar2_light']]
        wedges, texts, autotexts = ax_pi.pie(
            sizes,
            labels=['Yes', 'No'],
            colors=colors_pi,
            autopct='%1.0f%%',
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=1.5),
        )
        for at in autotexts:
            at.set_fontsize(10)
        ax_pi.set_title(title_pi, fontsize=8.5, pad=6)

    fig_f.suptitle(
        'F  \\u2014  Per-patient epoch decoding trend summary',
        fontsize=10, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    savefig(fig_f, 'F_epoch_wls_pie', FIG_DIR)
    plt.show()
    print(f'saved F  |  sig-pos: {n_pos_sig}/{n_total}, any-pos: {n_pos}/{n_total}')
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 36: CV comparison md
# ─────────────────────────────────────────────────────────────────────────────
c36 = md_cell(
"""## 16. Panel CV — Encoding vs Decoding Epoch Dispersion Comparison

**CV computation:**
- Encoding CV = coefficient of variation of all individual sig-unit pseudo-R² values
  across patients / epochs (pooled).
- Decoding CV = coefficient of variation of all individual resample F1-micro values
  across patients / epochs (pooled).

**Feltz & Miller (1996) asymptotic test** for equality of two CVs (χ², df=1).

**Line plots:**
- Top axis: mean pseudo-R² per `hours_from_start` bin ± SEM, pooled across patients.
- Bottom axis: mean F1-micro per `hours_from_start` bin ± SEM, pooled across patients.
- Truncated at bins where fewer than N_patients / 2 patients have data."""
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 37: CV comparison code
# ─────────────────────────────────────────────────────────────────────────────
c37 = code_cell(
"""# ── load encoding per-epoch data ──────────────────────────────────────────────
print('Loading encoding per-epoch data for CV panel...')
perepoch_enc = load_enc_perepoch(ENC_PE_RUN)
print(f'Encoding per-epoch: {len(perepoch_enc)} records  |  '
      f'{perepoch_enc["patient"].nunique() if not perepoch_enc.empty else 0} patients')

enc_r2_vals = load_enc_r2_vals(ENC_PE_RUN)
print(f'Encoding individual R² values: {len(enc_r2_vals)}')

# decoding individual resample values — use perepoch_rs_fs (no transcript filter needed)
dec_f1_vals = perepoch_rs_fs['f1_micro'].values if not perepoch_rs_fs.empty else np.array([])
print(f'Decoding individual F1 values: {len(dec_f1_vals)}')

# ── CVs ───────────────────────────────────────────────────────────────────────
enc_r2_pos = enc_r2_vals[enc_r2_vals > 0]   # CV meaningful only for positive values
dec_f1_pos = dec_f1_vals[dec_f1_vals > 0]

cv_enc = float(np.std(enc_r2_pos, ddof=1) / np.mean(enc_r2_pos)) if len(enc_r2_pos) > 1 else np.nan
cv_dec = float(np.std(dec_f1_pos, ddof=1) / np.mean(dec_f1_pos)) if len(dec_f1_pos) > 1 else np.nan
print(f'CV encoding (sig units, r²>0): {cv_enc:.3f}  (n={len(enc_r2_pos)})')
print(f'CV decoding (resamples, F1>0): {cv_dec:.3f}  (n={len(dec_f1_pos)})')

# ── Feltz–Miller test ─────────────────────────────────────────────────────────
if len(enc_r2_pos) > 1 and len(dec_f1_pos) > 1:
    fm_chi2, fm_p = feltz_miller_test(enc_r2_pos, dec_f1_pos)
    fm_str = f'Feltz–Miller: \\u03c7²={fm_chi2:.2f}, p={fm_p:.4f}' if fm_p >= 1e-4 else \
             f'Feltz–Miller: \\u03c7²={fm_chi2:.2f}, p<0.0001'
    print(fm_str)
else:
    fm_chi2, fm_p, fm_str = np.nan, np.nan, 'Feltz–Miller: insufficient data'

# ── pooled epoch line plot helpers ────────────────────────────────────────────
def make_epoch_pool_series(df, y_col, bin_col='hours_from_start'):
    \"\"\"
    Pool across patients at each time bin.
    Returns DataFrame: bin, mean, sem, n_patients.
    Truncates at bins with < half patients.
    \"\"\"
    n_pts_total = df['patient'].nunique()
    rows = []
    for b, grp in df.groupby(bin_col):
        vals = grp[y_col].dropna().values
        n_pts_here = grp['patient'].nunique()
        if n_pts_here < 1:
            continue
        rows.append({
            'bin': float(b),
            'mean': float(np.mean(vals)) if len(vals) > 0 else np.nan,
            'sem':  float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            'n_patients': n_pts_here,
        })
    out = pd.DataFrame(rows).sort_values('bin')
    # truncate at first bin with < half patients (keep only leading contiguous valid bins)
    half = n_pts_total / 2
    mask = out['n_patients'] >= half
    # find last valid contiguous bin
    valid_idx = out.index[mask]
    if len(valid_idx) == 0:
        return out.iloc[:0]
    last_valid = valid_idx[-1]
    return out.loc[:last_valid]

if not perepoch_enc.empty and not perepoch_fs.empty:
    enc_pool = make_epoch_pool_series(perepoch_enc, 'r2_mean')
    dec_pool = make_epoch_pool_series(perepoch_fs,  'f1_mean')
    print(f'Encoding pool: {len(enc_pool)} time bins (up to {enc_pool["bin"].max():.0f}h)')
    print(f'Decoding pool: {len(dec_pool)} time bins (up to {dec_pool["bin"].max():.0f}h)')

    fig_cv, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=False)

    # Top: encoding pseudo-R²
    if not enc_pool.empty:
        ax_top.plot(enc_pool['bin'], enc_pool['mean'],
                    color=COL['filtered'], linewidth=1.8, zorder=3)
        ax_top.fill_between(enc_pool['bin'],
                             enc_pool['mean'] - enc_pool['sem'],
                             enc_pool['mean'] + enc_pool['sem'],
                             color=COL['filtered'], alpha=0.20, zorder=2)
    ax_top.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax_top.set_ylabel('Mean pseudo-R²')
    ax_top.set_title('Encoding: per-epoch mean pseudo-R² (sig units, filtered speech)', fontsize=9)
    cv_enc_str = f'CV = {cv_enc:.3f}' if not np.isnan(cv_enc) else 'CV = n/a'
    ax_top.text(0.98, 0.97, cv_enc_str, ha='right', va='top',
                transform=ax_top.transAxes, fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='none'))

    # Bottom: decoding F1-micro
    if not dec_pool.empty:
        ax_bot.plot(dec_pool['bin'], dec_pool['mean'],
                    color=COL['patient'], linewidth=1.8, zorder=3)
        ax_bot.fill_between(dec_pool['bin'],
                             dec_pool['mean'] - dec_pool['sem'],
                             dec_pool['mean'] + dec_pool['sem'],
                             color=COL['patient'], alpha=0.20, zorder=2)
    ax_bot.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax_bot.set_xlabel('Hours from EMU admission')
    ax_bot.set_ylabel('Mean F1-micro')
    ax_bot.set_title('Decoding: per-epoch mean F1-micro (filtered speech)', fontsize=9)
    cv_dec_str = f'CV = {cv_dec:.3f}' if not np.isnan(cv_dec) else 'CV = n/a'
    ax_bot.text(0.98, 0.97, cv_dec_str, ha='right', va='top',
                transform=ax_bot.transAxes, fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, ec='none'))

    # Feltz–Miller annotation on whole figure
    fig_cv.text(0.5, 1.002, fm_str,
                ha='center', va='bottom', fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9, ec='goldenrod'))

    fig_cv.suptitle(
        'CV  \\u2014  Encoding vs decoding epoch performance dispersion\\n'
        '(pooled across patients; truncated at < 50% patients per bin)',
        fontsize=10, fontweight='bold', y=1.04
    )
    plt.tight_layout()
    savefig(fig_cv, 'CV_encoding_vs_decoding_dispersion', FIG_DIR)
    plt.show()
    print('saved CV comparison panel')
else:
    print('Skipping CV panel — missing encoding or decoding per-epoch data.')
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 38: Mockup md
# ─────────────────────────────────────────────────────────────────────────────
c38 = md_cell("## 17. Combined Mock-up (Main Panels A–CV)")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 39: Mockup code
# ─────────────────────────────────────────────────────────────────────────────
c39 = code_cell(
"""panel_pngs = [
    ('A — per-patient barplot',            FIG_DIR / 'A_decoding_per_patient.png'),
    ('B — filtered vs patient (paired)',   FIG_DIR / 'B_filtered_vs_patient_paired.png'),
    ('D+E — 24/7 vs convo & podcast',      FIG_DIR / 'DE_combined_comparison.png'),
    ('G — whole-stay vs per-day (paired)', FIG_DIR / 'G_paired_wholestay_vs_perday_filtered.png'),
    ('C — decoder vs n_units (per-day)',   FIG_DIR / 'C_decoder_vs_n_units_perday.png'),
    ('E — pooled epoch scatter (WLS)',     FIG_DIR / 'E_pooled_epoch_wls_scatter.png'),
    ('F — pie charts (WLS summary)',       FIG_DIR / 'F_epoch_wls_pie.png'),
    ('CV — encoding vs decoding dispersion', FIG_DIR / 'CV_encoding_vs_decoding_dispersion.png'),
]

N_COLS = 4
N_ROWS = int(np.ceil(len(panel_pngs) / N_COLS))

fig_mock, axes_mock = plt.subplots(
    N_ROWS, N_COLS,
    figsize=(N_COLS * 6.0, N_ROWS * 5.5),
    facecolor='white',
)
axes_flat = axes_mock.flatten()
for ax in axes_flat:
    ax.axis('off')

for i, (label, png_path) in enumerate(panel_pngs):
    if png_path.exists():
        axes_flat[i].imshow(imread(str(png_path)))
    else:
        axes_flat[i].text(
            0.5, 0.5,
            f'{label}\\n(not yet generated)',
            ha='center', va='center',
            transform=axes_flat[i].transAxes,
            fontsize=9, color='#888',
        )
    axes_flat[i].set_title(label, fontsize=8, fontweight='bold', pad=3)
    axes_flat[i].axis('off')

fig_mock.suptitle('Figure 4 — main panels mock-up', fontsize=14, fontweight='bold', y=1.002)
plt.tight_layout()
mockup_path = FIG_DIR / 'figure_4_main_mockup.png'
fig_mock.savefig(mockup_path, dpi=130, bbox_inches='tight', facecolor='white')
plt.show()
print(f'Main mock-up saved → {mockup_path}')
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELLS 40–49: Supplementary panels (keep from old cells 39–51, renumber)
# old 40 → supp header md (skip, we have c29 for the supp section header above)
# old 40 = supp shared helpers md
# old 41 = supp shared helpers code
# old 42 = S1 heatmap filtered md → now S2
# old 43 = S1 heatmap filtered code → now S2
# old 44 = S2 epoch line filtered md → now S3
# old 45 = S2 epoch line filtered code → now S3
# old 46 = S3 heatmap patient md → now S4
# old 47 = S3 heatmap patient code → now S4
# old 48 = S4 epoch line patient md → now S5
# old 49 = S4 epoch line patient code → now S5
# old 50 = supp mockup md
# old 51 = supp mockup code
# ─────────────────────────────────────────────────────────────────────────────
c40 = md_cell(old(40))  # Shared Helpers
c41 = code_cell(old(41))

def rename_supp(src, old_label, new_label):
    return src.replace(old_label, new_label)

# S2 heatmap filtered (old S1)
s2_hm_md_src = old(42).replace('S1  ', 'S2  ').replace('## Supplementary S1', '## Supplementary S2')
c42 = md_cell(s2_hm_md_src)
s2_hm_code_src = old(43).replace("'S1_per_day_heatmap_filtered'", "'S2_per_day_heatmap_filtered'") \
                         .replace("S1  —", "S2  —") \
                         .replace("'S1'", "'S2'") \
                         .replace("'saved S1'", "'saved S2'") \
                         .replace("print('saved S1')", "print('saved S2')")
c43 = code_cell(s2_hm_code_src)

# S3 epoch line filtered (old S2)
s3_ep_md_src = old(44).replace('S2  ', 'S3  ').replace('## Supplementary S2', '## Supplementary S3')
c44 = md_cell(s3_ep_md_src)
s3_ep_code_src = old(45).replace("'S2_per_epoch_filtered'", "'S3_per_epoch_filtered'") \
                         .replace("S2  —", "S3  —") \
                         .replace("print('saved S2')", "print('saved S3')")
c45 = code_cell(s3_ep_code_src)

# S4 heatmap patient (old S3)
s4_hm_md_src = old(46).replace('S3  ', 'S4  ').replace('## Supplementary S3', '## Supplementary S4')
c46 = md_cell(s4_hm_md_src)
s4_hm_code_src = old(47).replace("'S3_per_day_heatmap_patient_speech'", "'S4_per_day_heatmap_patient_speech'") \
                         .replace("S3  —", "S4  —") \
                         .replace("print('saved S3')", "print('saved S4')")
c47 = code_cell(s4_hm_code_src)

# S5 epoch line patient (old S4)
s5_ep_md_src = old(48).replace('S4  ', 'S5  ').replace('## Supplementary S4', '## Supplementary S5')
c48 = md_cell(s5_ep_md_src)
s5_ep_code_src = old(49).replace("'S4_per_epoch_patient_speech'", "'S5_per_epoch_patient_speech'") \
                         .replace("S4  —", "S5  —") \
                         .replace("print('saved S4')", "print('saved S5')")
c49 = code_cell(s5_ep_code_src)

# Supp mockup
c50 = md_cell(old(50))
supp_mock_src = old(51)
supp_mock_src = (
    supp_mock_src
    .replace(
        "supp_pngs = [\n"
        "    ('S1', FIG_DIR / 'S1_per_day_heatmap_filtered.png'),\n"
        "    ('S2', FIG_DIR / 'S2_per_epoch_filtered.png'),\n"
        "    ('S3', FIG_DIR / 'S3_per_day_heatmap_patient_speech.png'),\n"
        "    ('S4', FIG_DIR / 'S4_per_epoch_patient_speech.png'),\n"
        "]",
        "supp_pngs = [\n"
        "    ('S1 — epoch WLS scatter',   FIG_DIR / 'S1_per_patient_epoch_wls_scatter.png'),\n"
        "    ('S2 — per-day heatmap filt', FIG_DIR / 'S2_per_day_heatmap_filtered.png'),\n"
        "    ('S3 — epoch line filtered', FIG_DIR / 'S3_per_epoch_filtered.png'),\n"
        "    ('S4 — per-day heatmap pt',  FIG_DIR / 'S4_per_day_heatmap_patient_speech.png'),\n"
        "    ('S5 — epoch line patient',  FIG_DIR / 'S5_per_epoch_patient_speech.png'),\n"
        "]"
    )
    .replace(
        "fig_smock, axes_smock = plt.subplots(2, 2, figsize=(13, 10), facecolor='white')",
        "fig_smock, axes_smock = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')"
    )
    .replace(
        "Figure 4 — Supplementary panels mock-up",
        "Figure 4 — Supplementary panels mock-up (S1–S5)"
    )
)
c51 = code_cell(supp_mock_src)

# ─────────────────────────────────────────────────────────────────────────────
# Assemble new notebook
# ─────────────────────────────────────────────────────────────────────────────
new_cells = [
    c00, c01, c02, c03, c04, c05, c06, c07, c08, c09,  # 0-9
    c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,  # 10-19
    c20, c21, c22, c23, c24, c25, c26, c27, c28, c29,  # 20-29
    c30, c31, c32, c33, c34, c35, c36, c37, c38, c39,  # 30-39
    c40, c41, c42, c43, c44, c45, c46, c47, c48, c49,  # 40-49
    c50, c51,                                            # 50-51
]

nb_new = dict(nb_old)
nb_new['cells'] = new_cells

with open('figure_4.ipynb', 'w') as f:
    json.dump(nb_new, f, indent=1)

print(f'figure_4.ipynb saved with {len(new_cells)} cells')
