import json, copy

# ── shared helper strings ─────────────────────────────────────────────────────

REGION_LOAD_HELPERS = '''\
# ── Region comparison helpers ─────────────────────────────────────────────────
REGION_DEC_RUN = 'scat_xgboost_region_per_day'
ENC_REGION_RUN = 'word_level_duration_cv_filtered_speech_per_day'
N_CAT_RESAMPLES_R = 10   # cat resamples per brain resample

REGION_LABELS = {
    'ACC': 'ACC', 'HPC': 'HPC', 'OFC': 'OFC',
    'PCC': 'PCC', 'Th': 'Thal.', 'A': 'Amyg.',
}

def load_region_dec_data():
    """f1_micro per region pooled across patient × day × brain_resample × cat_resample."""
    region_vals = {}
    for pt in ALL_PATIENTS:
        base = VAD_ROOT / pt / 'standard_decoding_analysis' / REGION_DEC_RUN
        if not base.exists():
            continue
        for date_dir in sorted(base.iterdir()):
            if not date_dir.is_dir():
                continue
            for region_dir in sorted(date_dir.iterdir()):
                if not region_dir.is_dir():
                    continue
                region = region_dir.name
                for brain_dir in sorted(region_dir.iterdir()):
                    if not brain_dir.is_dir():
                        continue
                    b_lbl = brain_dir.name   # 'brain0', 'brain1', ...
                    for c in range(N_CAT_RESAMPLES_R):
                        sp = brain_dir / f'summary_{region}_{b_lbl}_cat{c}.json'
                        if sp.exists():
                            try:
                                d = json.load(open(sp))
                                region_vals.setdefault(region, []).append(float(d['f1_micro']))
                            except Exception:
                                pass
    return {r: np.array(v) for r, v in region_vals.items()}


def load_region_enc_data():
    """pseudo_r2_mean per region pooled across patient × day × unit (all units, no sig filter)."""
    region_vals = {}
    for pt in ALL_PATIENTS:
        micro_p = VAD_ROOT / pt / f'{pt}_micro_info.csv'
        if not micro_p.exists():
            continue
        df_micro = pd.read_csv(micro_p, index_col=0).reset_index(drop=True)
        base = VAD_ROOT / pt / 'encoding' / ENC_REGION_RUN
        if not base.exists():
            continue
        for date_dir in sorted(base.iterdir()):
            if not date_dir.is_dir():
                continue
            pkl = date_dir / f'{pt}_encoding_results_cv.pkl'
            if not pkl.exists():
                continue
            try:
                df = pickle.load(open(pkl, 'rb'))
            except Exception:
                continue
            if 'is_summary' not in df.columns:
                continue
            for _, row in df[df['is_summary'] == True].iterrows():
                nidx = int(row['neuron_idx'])
                if 0 <= nidx < len(df_micro):
                    region = df_micro.iloc[nidx]['region_symbol']
                    r2 = float(row['pseudo_r2_mean'])
                    if np.isfinite(r2):
                        region_vals.setdefault(region, []).append(r2)
    return {r: np.array(v) for r, v in region_vals.items()}


def draw_region_boxplot(ax, region_data, metric_label, panel_letter,
                        bar_color_key='bar0', showfliers=True):
    """Boxplot one box per region; Kruskal–Wallis ANOVA; rank list text."""
    from scipy.stats import kruskal, ranksums

    if not region_data:
        ax.text(0.5, 0.5, 'No region data loaded.',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        return

    # order by median descending
    regions = sorted(region_data.keys(),
                     key=lambda r: float(np.median(region_data[r])), reverse=True)
    vals_list = [region_data[r] for r in regions]

    # Kruskal–Wallis
    if len(vals_list) >= 2:
        kw_H, kw_p = kruskal(*vals_list)
        kw_str = f'Kruskal–Wallis: H={kw_H:.2f}, p={kw_p:.4f}' \
                 if kw_p >= 1e-4 else f'Kruskal–Wallis: H={kw_H:.2f}, p<0.0001'
    else:
        kw_str = ''

    # pairwise Wilcoxon between adjacent ranks
    wilcox_sigs = []
    for i in range(len(regions) - 1):
        _, p_wx = ranksums(vals_list[i], vals_list[i + 1])
        wilcox_sigs.append(p_wx)

    # colours per box (cycle through COL palette)
    box_cols = [COL['bar0'], COL['bar1'], COL['bar2'],
                COL['bar3'], COL['filtered'], COL['patient']]

    bp = ax.boxplot(
        vals_list,
        patch_artist=True,
        widths=0.55,
        showfliers=showfliers,
        medianprops=dict(color='white', linewidth=2.0, solid_capstyle='round'),
        whiskerprops=dict(linewidth=0.8, color='dimgray'),
        capprops=dict(linewidth=0.8, color='dimgray'),
        flierprops=dict(marker='.', markersize=2.5, alpha=0.25,
                        markeredgecolor='none'),
        boxprops=dict(linewidth=0.8),
    )
    for patch, col in zip(bp['boxes'], box_cols[:len(regions)]):
        patch.set_facecolor(col)
        patch.set_alpha(0.78)

    ax.set_xticks(range(1, len(regions) + 1))
    ax.set_xticklabels([REGION_LABELS.get(r, r) for r in regions], fontsize=9)
    ax.set_ylabel(metric_label)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)

    # n= labels above boxes
    y_vals_all = np.concatenate(vals_list)
    ymax_data  = float(np.percentile(y_vals_all[np.isfinite(y_vals_all)], 99)) if len(y_vals_all) else 1.0
    y_label_pos = ymax_data * 1.03
    for i, (r, vals) in enumerate(zip(regions, vals_list)):
        ax.text(i + 1, y_label_pos, f'n={len(vals):,}',
                ha='center', va='bottom', fontsize=6.5, color='dimgray')

    # significance stars between adjacent pairs
    for i, p_wx in enumerate(wilcox_sigs):
        stars = '***' if p_wx < 0.001 else ('**' if p_wx < 0.01 else ('*' if p_wx < 0.05 else 'ns'))
        mid_x = i + 1.5
        ax.text(mid_x, y_label_pos * 1.04, stars,
                ha='center', va='bottom', fontsize=6.5, color='dimgray')

    # rank + Kruskal text box
    rank_lines = [f'{i+1}.  {REGION_LABELS.get(r,r):<8s}  med={np.median(region_data[r]):.4f}'
                  for i, r in enumerate(regions)]
    box_text = 'Rank (by median):\n' + '\\n'.join(rank_lines) + '\\n\\n' + kw_str
    ax.text(0.985, 0.975, box_text,
            ha='right', va='top', transform=ax.transAxes,
            fontsize=6.5, color='dimgray', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.92, ec='none'))

    ax.set_title(panel_letter, fontsize=10, fontweight='bold', loc='left')
    ax.set_xlim(0.35, len(regions) + 0.65)
'''

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — add encoding region panel G
# ─────────────────────────────────────────────────────────────────────────────
with open('figure_3.ipynb') as f:
    nb3 = json.load(f)
cells3 = nb3['cells']

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

def code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": [src]}

# Where to insert: before cell 39 (mockup md), i.e. at position 39
INSERT_POS_3 = 39

NEW_G_MD3 = md_cell(
"""## 18. Panel G — Encoding Performance by Brain Region

Box per region; distribution = all units × days × patients.
Kruskal–Wallis one-way ANOVA + pairwise Wilcoxon rank-sum between adjacent-rank regions.
Rank list by median pseudo-R² displayed in panel."""
)

NEW_G_CODE3 = code_cell(
REGION_LOAD_HELPERS + '''
# ── load encoding region data ──────────────────────────────────────────────────
print('Loading encoding region data...')
region_enc = load_region_enc_data()
for r, v in sorted(region_enc.items()):
    print(f'  {r}: n={len(v):,}  med={np.median(v):.4f}  mean={np.mean(v):.4f}')

if not region_enc:
    print('No encoding region data found.')
    fig_g3 = None
else:
    fig_g3, ax_g3 = plt.subplots(figsize=(max(5, len(region_enc) * 1.3 + 1.5), 5.2))
    draw_region_boxplot(
        ax_g3, region_enc,
        metric_label='Pseudo R\\u00b2 (all units, per-day)',
        panel_letter='G  \\u2014  Encoding performance by brain region',
    )
    plt.tight_layout()
    savefig(fig_g3, 'G_encoding_by_region', FIG_DIR)
    plt.show()
    print('saved G (encoding region boxplot)')
'''
)

# Update mockup in cell 39+2=41 to include G
# (after insertion, old cell 39 becomes 41)
# Actually: insert two cells before current cell 39 → new cells at 39,40
# Then current cell 39 (mockup md) becomes 41, cell 40 (mockup code) becomes 42

cells3_new = cells3[:INSERT_POS_3] + [NEW_G_MD3, NEW_G_CODE3] + cells3[INSERT_POS_3:]

# Find and update mockup code cell (now at position 41 = INSERT_POS_3+2)
MOCK_CODE_IDX3 = INSERT_POS_3 + 2  # was 40, now 42 after insert of 2 cells
# Actually originally mockup was at cells3[39] and [40], so after insert of 2 cells
# at position 39, mockup md is at 41 and mockup code is at 42
MOCK_CODE_IDX3 = INSERT_POS_3 + 2 + 1  # mockup code was at 40, now at 42

# Update mockup code to include G
src_mock3 = ''.join(cells3_new[42]['source'])
if 'panel_pngs' in src_mock3:
    # Add panel G
    old_mock_end3 = (
        "    ('F — pie charts (WLS summary)',       FIG_DIR / 'F_epoch_wls_pie.png'),\n"
        "]"
    )
    new_mock_end3 = (
        "    ('F — pie charts (WLS summary)',       FIG_DIR / 'F_epoch_wls_pie.png'),\n"
        "    ('G — encoding by brain region',       FIG_DIR / 'G_encoding_by_region.png'),\n"
        "]"
    )
    src_mock3 = src_mock3.replace(old_mock_end3, new_mock_end3)
    # Also update N_COLS and grid
    src_mock3 = src_mock3.replace('N_COLS = 4', 'N_COLS = 4')  # stays 4 (9 panels = 3 rows)
    cells3_new[42]['source'] = [src_mock3]
    print(f'Figure 3 mockup updated (cell 42)')
else:
    print(f'WARNING: mockup not found at cell 42')
    print(f'  cell 42 starts with: {repr("".join(cells3_new[42]["source"])[:80])}')

# Also update the title MD (cell 0)
src_title3 = ''.join(cells3_new[0]['source'])
if '**F**' in src_title3:
    cells3_new[0]['source'] = [src_title3.replace(
        '- **F** — pie charts', 
        '- **F** — pie charts\n- **G** — encoding performance by brain region'
    )]

nb3['cells'] = cells3_new
with open('figure_3.ipynb', 'w') as f:
    json.dump(nb3, f, indent=1)
print(f'figure_3.ipynb: {len(cells3_new)} cells saved.')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — rename CV→G, add decoding region panel H
# ─────────────────────────────────────────────────────────────────────────────
with open('figure_4.ipynb') as f:
    nb4 = json.load(f)
cells4 = nb4['cells']

# Step 1: rename CV → G in cells 36 and 37
for ci in [36, 37]:
    src = ''.join(cells4[ci]['source'])
    src = (src
        .replace('## 16. Panel CV —', '## 16. Panel G —')
        .replace("'CV  \\u2014", "'G  \\u2014")
        .replace("## CV comparison", "## G comparison")
        .replace("'CV_encoding_vs_decoding_dispersion'", "'G_encoding_vs_decoding_dispersion'")
        .replace("'CV comparison'", "'G comparison'")
        .replace("fig_cv", "fig_g4cv")
        .replace("ax_top", "ax_g4_top")
        .replace("ax_bot", "ax_g4_bot")
    )
    cells4[ci]['source'] = [src]
print('Figure 4: CV panels renamed to G')

# Also fix the cell 36 md title
src36_4 = ''.join(cells4[36]['source'])
if 'Panel CV' in src36_4 or 'Panel G —' not in src36_4:
    cells4[36]['source'] = [src36_4.replace(
        '## 16. Panel CV',
        '## 16. Panel G'
    )]

# Step 2: insert region comparison panel H before mockup (currently at cells 38,39)
INSERT_POS_4 = 38

NEW_H_MD4 = md_cell(
"""## 17. Panel H — Decoding Performance by Brain Region

Box per region; distribution = all resamples × days × patients × brain-resamples.
Kruskal–Wallis one-way ANOVA + pairwise Wilcoxon rank-sum between adjacent-rank regions.
Rank list by median F1-micro displayed in panel."""
)

NEW_H_CODE4 = code_cell(
REGION_LOAD_HELPERS + '''
# ── load decoding region data ──────────────────────────────────────────────────
print('Loading decoding region data...')
region_dec = load_region_dec_data()
for r, v in sorted(region_dec.items()):
    print(f'  {r}: n={len(v):,}  med={np.median(v):.4f}  mean={np.mean(v):.4f}')

if not region_dec:
    print('No decoding region data found.')
    fig_h4 = None
else:
    fig_h4, ax_h4 = plt.subplots(figsize=(max(5, len(region_dec) * 1.3 + 1.5), 5.2))
    draw_region_boxplot(
        ax_h4, region_dec,
        metric_label='F1-micro (all resamples, per-day)',
        panel_letter='H  \\u2014  Decoding performance by brain region',
    )
    plt.tight_layout()
    savefig(fig_h4, 'H_decoding_by_region', FIG_DIR)
    plt.show()
    print('saved H (decoding region boxplot)')
'''
)

cells4_new = cells4[:INSERT_POS_4] + [NEW_H_MD4, NEW_H_CODE4] + cells4[INSERT_POS_4:]

# Update mockup code (was at 39, now at 41)
MOCK_CODE_IDX4 = INSERT_POS_4 + 2 + 1  # mockup code was at 39, now 41
src_mock4 = ''.join(cells4_new[41]['source'])
if 'panel_pngs' in src_mock4:
    old_mock_end4 = (
        "    ('CV — encoding vs decoding dispersion', FIG_DIR / 'CV_encoding_vs_decoding_dispersion.png'),\n"
        "]"
    )
    new_mock_end4 = (
        "    ('G — encoding vs decoding dispersion', FIG_DIR / 'G_encoding_vs_decoding_dispersion.png'),\n"
        "    ('H — decoding by brain region',         FIG_DIR / 'H_decoding_by_region.png'),\n"
        "]"
    )
    src_mock4 = src_mock4.replace(old_mock_end4, new_mock_end4)
    cells4_new[41]['source'] = [src_mock4]
    print(f'Figure 4 mockup updated (cell 41)')
else:
    print(f'WARNING: fig4 mockup not found at cell 41')
    print(f'  cell 41 starts with: {repr("".join(cells4_new[41]["source"])[:80])}')

# Update title md (cell 0)
src_title4 = ''.join(cells4_new[0]['source'])
cells4_new[0]['source'] = [src_title4.replace(
    '- **CV** — encoding vs decoding dispersion comparison',
    '- **G** — encoding vs decoding dispersion comparison\n- **H** — decoding performance by brain region'
).replace(
    '- **E_epoch**',
    '- **E**'
)]

nb4['cells'] = cells4_new
with open('figure_4.ipynb', 'w') as f:
    json.dump(nb4, f, indent=1)
print(f'figure_4.ipynb: {len(cells4_new)} cells saved.')
