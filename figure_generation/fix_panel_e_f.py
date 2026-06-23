import json, textwrap

# ── shared color for negative trend ──────────────────────────────────────────
NEG_COLOR = "'#d46f4d'"   # warm orange-red, contrasts with COL['filtered']

# ══════════════════════════════════════════════════════════════════════════════
# NEW Panel E overlay code (figure 3) — replaces cell 34
# ══════════════════════════════════════════════════════════════════════════════
NEW_E_FIG3 = '''if scatter_df.empty or not patient_epoch_stats:
    print('Skipping E — no scatter data or no patient stats (run S1 first).')
else:
    # ── tier classification ────────────────────────────────────────────────
    def _tier(pt):
        s = patient_epoch_stats.get(pt, {})
        r, p = s.get('r_wls', np.nan), s.get('p_wls', 1.0)
        if np.isnan(r): return 'other'
        if r > 0 and p < 0.05: return 'pos_sig'
        if r > 0:               return 'pos_ns'
        return 'other'

    TIER_ORDER  = ['other', 'pos_ns', 'pos_sig']   # draw back→front
    DOT_ALPHA   = {'pos_sig': 0.40, 'pos_ns': 0.15, 'other': 0.07}
    LINE_ALPHA  = {'pos_sig': 1.00, 'pos_ns': 0.50, 'other': 0.22}
    LINE_LW     = {'pos_sig': 1.8,  'pos_ns': 1.2,  'other': 0.9}
    LINE_COL    = {'pos_sig': COL['filtered'], 'pos_ns': COL['filtered'], 'other': '#888888'}
    DOT_COL     = {'pos_sig': COL['dot_scatter'], 'pos_ns': COL['dot_scatter'], 'other': '#aaaaaa'}
    ZORDER      = {'other': 1, 'pos_ns': 2, 'pos_sig': 3}

    patients_e = sorted(scatter_df['patient'].unique())

    fig_e, ax_e = plt.subplots(figsize=(7.0, 5.5))
    ax_e.axhline(0, color='gray', linestyle='--', linewidth=0.7, zorder=0)

    for tier in TIER_ORDER:
        for pt in patients_e:
            if _tier(pt) != tier: continue
            pt_data = scatter_df[scatter_df['patient'] == pt]
            x_vals  = pt_data['patient_frac'].values
            y_vals  = pt_data['pseudo_r2_mean'].values
            s = patient_epoch_stats.get(pt, {})
            ax_e.scatter(x_vals, y_vals,
                         color=DOT_COL[tier], alpha=DOT_ALPHA[tier],
                         s=8, edgecolors='none', zorder=ZORDER[tier])
            if 'slope' in s and len(x_vals) >= 4:
                x_l = np.linspace(x_vals.min(), x_vals.max(), 200)
                y_l = s['slope'] * x_l + s['intercept']
                ax_e.plot(x_l, y_l, color=LINE_COL[tier],
                          alpha=LINE_ALPHA[tier], linewidth=LINE_LW[tier],
                          zorder=ZORDER[tier] + 0.5)

    # ── legend proxies ────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color=COL['filtered'], lw=1.8, alpha=1.0,
               label='Positive & significant (r>0, p<0.05)'),
        Line2D([0],[0], color=COL['filtered'], lw=1.2, alpha=0.50,
               label='Positive, not significant (r>0)'),
        Line2D([0],[0], color='#888888',      lw=0.9, alpha=0.22,
               label='Non-positive trend'),
    ]
    ax_e.legend(handles=legend_elems, fontsize=7.5, frameon=False,
                loc='upper left', handlelength=1.8)

    n_pos_sig_e = sum(1 for pt in patients_e if _tier(pt) == 'pos_sig')
    n_pos_ns_e  = sum(1 for pt in patients_e if _tier(pt) == 'pos_ns')
    n_other_e   = sum(1 for pt in patients_e if _tier(pt) == 'other')
    ax_e.text(0.97, 0.03,
              f'{n_pos_sig_e} sig-pos  ·  {n_pos_ns_e} pos  ·  {n_other_e} other'
              f'  (N={len(patients_e)} patients)',
              ha='right', va='bottom', transform=ax_e.transAxes,
              fontsize=7, color='dimgray',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.88, ec='none'))

    ax_e.set_xlabel('Patient speech fraction per epoch (words)', fontsize=10)
    ax_e.set_ylabel('Pseudo R\\u00b2 (fitted unit)', fontsize=10)
    ax_e.set_title(
        'E  \\u2014  Per-patient epoch encoding vs speech fraction\\n'
        '(opacity encodes WLS significance tier; lines = per-patient WLS fits)',
        fontsize=10,
    )
    plt.tight_layout()
    savefig(fig_e, 'E_epoch_overlay', FIG_DIR)
    plt.show()
    print('saved E (overlay)')
'''

# ══════════════════════════════════════════════════════════════════════════════
# NEW Panel F overlay code (figure 4) — replaces cell 33
# ══════════════════════════════════════════════════════════════════════════════
NEW_F_FIG4 = '''if scatter_df.empty or not patient_epoch_stats:
    print('Skipping F — no scatter data or no patient stats (run S1 first).')
    fig_e_pool = None
else:
    # ── tier classification ────────────────────────────────────────────────
    def _tier(pt):
        s = patient_epoch_stats.get(pt, {})
        r, p = s.get('r_wls', np.nan), s.get('p_wls', 1.0)
        if np.isnan(r): return 'other'
        if r > 0 and p < 0.05: return 'pos_sig'
        if r > 0:               return 'pos_ns'
        return 'other'

    TIER_ORDER  = ['other', 'pos_ns', 'pos_sig']
    DOT_ALPHA   = {'pos_sig': 0.40, 'pos_ns': 0.15, 'other': 0.07}
    LINE_ALPHA  = {'pos_sig': 1.00, 'pos_ns': 0.50, 'other': 0.22}
    LINE_LW     = {'pos_sig': 1.8,  'pos_ns': 1.2,  'other': 0.9}
    LINE_COL    = {'pos_sig': COL['filtered'], 'pos_ns': COL['filtered'], 'other': '#888888'}
    DOT_COL     = {'pos_sig': COL['dot_scatter'], 'pos_ns': COL['dot_scatter'], 'other': '#aaaaaa'}
    ZORDER      = {'other': 1, 'pos_ns': 2, 'pos_sig': 3}

    patients_e = sorted(scatter_df['patient'].unique())

    fig_e_pool, ax_e = plt.subplots(figsize=(7.0, 5.5))
    ax_e.axhline(0, color='gray', linestyle='--', linewidth=0.7, zorder=0)

    for tier in TIER_ORDER:
        for pt in patients_e:
            if _tier(pt) != tier: continue
            pt_data = scatter_df[scatter_df['patient'] == pt]
            x_vals  = pt_data['patient_frac'].values
            y_vals  = pt_data['f1_micro'].values
            s = patient_epoch_stats.get(pt, {})
            ax_e.scatter(x_vals, y_vals,
                         color=DOT_COL[tier], alpha=DOT_ALPHA[tier],
                         s=8, edgecolors='none', zorder=ZORDER[tier])
            if 'slope' in s and len(x_vals) >= 4:
                x_l = np.linspace(x_vals.min(), x_vals.max(), 200)
                y_l = s['slope'] * x_l + s['intercept']
                ax_e.plot(x_l, y_l, color=LINE_COL[tier],
                          alpha=LINE_ALPHA[tier], linewidth=LINE_LW[tier],
                          zorder=ZORDER[tier] + 0.5)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], color=COL['filtered'], lw=1.8, alpha=1.0,
               label='Positive & significant (r>0, p<0.05)'),
        Line2D([0],[0], color=COL['filtered'], lw=1.2, alpha=0.50,
               label='Positive, not significant (r>0)'),
        Line2D([0],[0], color='#888888',      lw=0.9, alpha=0.22,
               label='Non-positive trend'),
    ]
    ax_e.legend(handles=legend_elems, fontsize=7.5, frameon=False,
                loc='upper left', handlelength=1.8)

    n_pos_sig_e = sum(1 for pt in patients_e if _tier(pt) == 'pos_sig')
    n_pos_ns_e  = sum(1 for pt in patients_e if _tier(pt) == 'pos_ns')
    n_other_e   = sum(1 for pt in patients_e if _tier(pt) == 'other')
    ax_e.text(0.97, 0.03,
              f'{n_pos_sig_e} sig-pos  ·  {n_pos_ns_e} pos  ·  {n_other_e} other'
              f'  (N={len(patients_e)} patients)',
              ha='right', va='bottom', transform=ax_e.transAxes,
              fontsize=7, color='dimgray',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.88, ec='none'))

    ax_e.set_xlabel('Patient speech fraction per epoch (words)', fontsize=10)
    ax_e.set_ylabel('F1-micro', fontsize=10)
    ax_e.set_title(
        'F  \\u2014  Per-patient epoch decoding vs speech fraction\\n'
        '(opacity encodes WLS significance tier; lines = per-patient WLS fits)',
        fontsize=10,
    )
    plt.tight_layout()
    savefig(fig_e_pool, 'F_epoch_overlay', FIG_DIR)
    plt.show()
    print('saved F (overlay)')
'''

# ══════════════════════════════════════════════════════════════════════════════
# NEW pie chart code — figure 3 Panel F (cell 38)
# ══════════════════════════════════════════════════════════════════════════════
NEW_PIE_FIG3 = '''pts_stat = [pt for pt, s in patient_epoch_stats.items()
            if not np.isnan(s.get('r_wls', np.nan))]
N_stat = len(pts_stat)

if N_stat == 0:
    print('No patient_epoch_stats available — run S1 scatter cell first.')
else:
    def _r(pt): return patient_epoch_stats[pt].get('r_wls', 0)
    def _p(pt): return patient_epoch_stats[pt].get('p_wls', 1)

    # counts for each pie
    # Pie 1: significance-based
    n_ps  = sum(1 for pt in pts_stat if _r(pt) > 0 and _p(pt) < 0.05)   # sig positive
    n_ns_ = sum(1 for pt in pts_stat if _r(pt) < 0 and _p(pt) < 0.05)   # sig negative
    n_nt  = N_stat - n_ps - n_ns_                                          # not sig
    # Pie 2: direction-based
    n_ap  = sum(1 for pt in pts_stat if _r(pt) > 0)                       # any positive
    n_an  = sum(1 for pt in pts_stat if _r(pt) < 0)                       # any negative
    n_az  = N_stat - n_ap - n_an                                           # r = 0

    _COL_NEG  = '#d46f4d'
    _COL_NEUT = '#e0e0e0'

    pie_defs = [
        dict(counts=[n_ps, n_ns_, n_nt],
             colors=[COL['filtered'], _COL_NEG, _COL_NEUT],
             labels=[f'Sig. positive\\n{n_ps}/{N_stat}',
                     f'Sig. negative\\n{n_ns_}/{N_stat}',
                     f'Not significant\\n{n_nt}/{N_stat}'],
             center_pct=100.0*n_ps/N_stat,
             title='Significant WLS trend\\n(p < 0.05)'),
        dict(counts=[n_ap, n_an, n_az],
             colors=[COL['filtered'], _COL_NEG, _COL_NEUT],
             labels=[f'Any positive (r>0)\\n{n_ap}/{N_stat}',
                     f'Any negative (r<0)\\n{n_an}/{N_stat}',
                     f'No trend (r\\u22480)\\n{n_az}/{N_stat}'],
             center_pct=100.0*n_ap/N_stat,
             title='Any positive WLS trend'),
    ]

    fig_f_pie, axes_f = plt.subplots(1, 2, figsize=(9.0, 4.8))

    for ax, pd_ in zip(axes_f, pie_defs):
        wedges, _ = ax.pie(
            pd_['counts'],
            colors=pd_['colors'],
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.8},
        )
        # bold positive % in centre
        ax.text(0, 0, f"{pd_['center_pct']:.0f}%",
                ha='center', va='center', fontsize=22, fontweight='bold',
                color=COL['filtered'])
        ax.set_title(pd_['title'], fontsize=9.5, pad=8)
        ax.legend(wedges, pd_['labels'], fontsize=7.5,
                  loc='lower center', bbox_to_anchor=(0.5, -0.22),
                  ncol=1, frameon=False)

    fig_f_pie.suptitle(
        f'F  \\u2014  Per-patient epoch encoding trend summary  (N={N_stat} patients)\\n'
        '(WLS per patient on filtered-speech epochs; p via permutation n=500)',
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    savefig(fig_f_pie, 'F_pie_summary', FIG_DIR)
    plt.show()
    print(f'saved F pie  |  sig-pos={n_ps}, sig-neg={n_ns_}, not-sig={n_nt}  '
          f'|  any-pos={n_ap}, any-neg={n_an}')
'''

# ══════════════════════════════════════════════════════════════════════════════
# NEW pie chart code — figure 4 Panel G (cell 35)
# ══════════════════════════════════════════════════════════════════════════════
NEW_PIE_FIG4 = '''pts_stat = [pt for pt, s in patient_epoch_stats.items()
             if not np.isnan(s.get('r_wls', np.nan))]
n_total = len(pts_stat)

if n_total == 0:
    print('No patients with WLS fits — skipping Panel G.')
else:
    def _r(pt): return patient_epoch_stats[pt].get('r_wls', 0)
    def _p(pt): return patient_epoch_stats[pt].get('p_wls', 1)

    n_ps  = sum(1 for pt in pts_stat if _r(pt) > 0 and _p(pt) < 0.05)
    n_ns_ = sum(1 for pt in pts_stat if _r(pt) < 0 and _p(pt) < 0.05)
    n_nt  = n_total - n_ps - n_ns_
    n_ap  = sum(1 for pt in pts_stat if _r(pt) > 0)
    n_an  = sum(1 for pt in pts_stat if _r(pt) < 0)
    n_az  = n_total - n_ap - n_an

    _COL_NEG  = '#d46f4d'
    _COL_NEUT = '#e0e0e0'

    pie_defs = [
        dict(counts=[n_ps, n_ns_, n_nt],
             colors=[COL['filtered'], _COL_NEG, _COL_NEUT],
             labels=[f'Sig. positive\\n{n_ps}/{n_total}',
                     f'Sig. negative\\n{n_ns_}/{n_total}',
                     f'Not significant\\n{n_nt}/{n_total}'],
             center_pct=100.0*n_ps/n_total,
             title='Significant WLS trend\\n(p < 0.05)'),
        dict(counts=[n_ap, n_an, n_az],
             colors=[COL['filtered'], _COL_NEG, _COL_NEUT],
             labels=[f'Any positive (r>0)\\n{n_ap}/{n_total}',
                     f'Any negative (r<0)\\n{n_an}/{n_total}',
                     f'No trend (r\\u22480)\\n{n_az}/{n_total}'],
             center_pct=100.0*n_ap/n_total,
             title='Any positive WLS trend'),
    ]

    fig_g, axes_g = plt.subplots(1, 2, figsize=(9.0, 4.8))

    for ax, pd_ in zip(axes_g, pie_defs):
        wedges, _ = ax.pie(
            pd_['counts'],
            colors=pd_['colors'],
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.8},
        )
        ax.text(0, 0, f"{pd_['center_pct']:.0f}%",
                ha='center', va='center', fontsize=22, fontweight='bold',
                color=COL['filtered'])
        ax.set_title(pd_['title'], fontsize=9.5, pad=8)
        ax.legend(wedges, pd_['labels'], fontsize=7.5,
                  loc='lower center', bbox_to_anchor=(0.5, -0.22),
                  ncol=1, frameon=False)

    fig_g.suptitle(
        f'G  \\u2014  Per-patient epoch decoding trend summary  (N={n_total} patients)\\n'
        '(WLS per patient on filtered-speech epochs; p via permutation n=500)',
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    savefig(fig_g, 'G_epoch_wls_pie', FIG_DIR)
    plt.show()
    print(f'saved G pie  |  sig-pos={n_ps}, sig-neg={n_ns_}, not-sig={n_nt}  '
          f'|  any-pos={n_ap}, any-neg={n_an}')
'''

# ══════════════════════════════════════════════════════════════════════════════
# PATCH HELPER
# ══════════════════════════════════════════════════════════════════════════════
def set_cell(nb, idx, new_src):
    lines = new_src.split('\n')
    nb['cells'][idx]['source'] = [l+'\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def add_slope_to_stats(src):
    """Add slope/intercept storage to patient_epoch_stats in S1 cell."""
    OLD = "            stats_pt.update({'r_wls': r_pt, 'p_wls': p_pt})"
    NEW = "            stats_pt.update({'r_wls': r_pt, 'p_wls': p_pt, 'slope': slope_pt, 'intercept': intercept_pt})"
    assert OLD in src, f"Could not find stats_pt.update line in:\n{src[src.find('stats_pt'):][:200]}"
    return src.replace(OLD, NEW)

# ══════════════════════════════════════════════════════════════════════════════
# APPLY TO FIGURE 3
# ══════════════════════════════════════════════════════════════════════════════
with open('figure_3.ipynb') as f:
    nb3 = json.load(f)

# 1. S1 cell (36): add slope/intercept to stats
src36 = ''.join(nb3['cells'][36]['source'])
src36 = add_slope_to_stats(src36)
lines = src36.split('\n')
nb3['cells'][36]['source'] = [l+'\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
print('fig3 cell 36: slope/intercept stored')

# 2. Panel E cell (34): replace with overlay
set_cell(nb3, 34, NEW_E_FIG3)
print('fig3 cell 34: replaced with overlay (Panel E)')

# 3. Panel F pie cell (38): replace with 3-slice
set_cell(nb3, 38, NEW_PIE_FIG3)
print('fig3 cell 38: replaced with 3-slice pie (Panel F)')

# 4. Update mockup (cell 42): panel E filename
src42 = ''.join(nb3['cells'][42]['source'])
src42 = src42.replace("'E_cumulative_epoch_scatter.png'", "'E_epoch_overlay.png'")
src42 = src42.replace("'E — cumulative epoch scatter (WLS)'", "'E — per-patient epoch overlay'")
lines = src42.split('\n')
nb3['cells'][42]['source'] = [l+'\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
print('fig3 cell 42: mockup filename updated')

with open('figure_3.ipynb', 'w') as f:
    json.dump(nb3, f, indent=1)
print('figure_3.ipynb saved\n')

# ══════════════════════════════════════════════════════════════════════════════
# APPLY TO FIGURE 4
# ══════════════════════════════════════════════════════════════════════════════
with open('figure_4.ipynb') as f:
    nb4 = json.load(f)

# 1. S1 cell (31): add slope/intercept to stats
src31 = ''.join(nb4['cells'][31]['source'])
src31 = add_slope_to_stats(src31)
lines = src31.split('\n')
nb4['cells'][31]['source'] = [l+'\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
print('fig4 cell 31: slope/intercept stored')

# 2. Panel F scatter cell (33): replace with overlay
set_cell(nb4, 33, NEW_F_FIG4)
print('fig4 cell 33: replaced with overlay (Panel F)')

# 3. Panel G pie cell (35): replace with 3-slice
set_cell(nb4, 35, NEW_PIE_FIG4)
print('fig4 cell 35: replaced with 3-slice pie (Panel G)')

# 4. Update mockup (cell 41): panel F filename
src41 = ''.join(nb4['cells'][41]['source'])
src41 = src41.replace("'F_pooled_epoch_wls_scatter.png'", "'F_epoch_overlay.png'")
src41 = src41.replace("'F — pooled epoch scatter (WLS)'", "'F — per-patient epoch overlay'")
lines = src41.split('\n')
nb4['cells'][41]['source'] = [l+'\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
print('fig4 cell 41: mockup filename updated')

with open('figure_4.ipynb', 'w') as f:
    json.dump(nb4, f, indent=1)
print('figure_4.ipynb saved')
