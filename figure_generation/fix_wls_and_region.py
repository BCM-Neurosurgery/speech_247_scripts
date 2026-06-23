import json

# ───────────────────────────────────────────────────────────────────────────────
# FIX 1 — Figure 3: remove vis mask from WLS line drawing
# ───────────────────────────────────────────────────────────────────────────────
with open('figure_3.ipynb') as f:
    nb3 = json.load(f)

# ── Cell 34 (Panel E) ─────────────────────────────────────────────────────────
src34 = ''.join(nb3['cells'][34]['source'])

OLD34 = """    # WLS line — only draw the portion within the visible y range
    x_line_e = np.linspace(x_all.min(), x_all.max(), 500)
    y_line_e = slope_e * x_line_e + intercept_e
    vis_e = (y_line_e >= y_bot_e) & (y_line_e <= y_top_e)
    if vis_e.any():
        ax_e.plot(x_line_e[vis_e], y_line_e[vis_e],
                  color=COL['filtered'], linewidth=2.2, zorder=5, label='WLS fit')
    else:
        ax_e.plot(x_line_e, y_line_e,
                  color=COL['filtered'], linewidth=2.2, zorder=5, label='WLS fit')"""

NEW34 = """    # WLS line — plot full range, matplotlib clips to ylim naturally
    x_line_e = np.linspace(x_all.min(), x_all.max(), 500)
    y_line_e = slope_e * x_line_e + intercept_e
    ax_e.plot(x_line_e, y_line_e,
              color=COL['filtered'], linewidth=2.2, zorder=5, label='WLS fit')"""

assert OLD34 in src34, "OLD34 not found in cell 34"
src34 = src34.replace(OLD34, NEW34)
lines = src34.split('\n')
nb3['cells'][34]['source'] = [l+'\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
print('Cell 34: vis mask removed')

# ── Cell 36 (S1 per-patient) ──────────────────────────────────────────────────
src36 = ''.join(nb3['cells'][36]['source'])

OLD36 = """            x_line = np.linspace(x_vals.min(), x_vals.max(), 300)
            y_line_pt = slope_pt * x_line + intercept_pt
            vis_pt = (y_line_pt >= -0.25) & (y_line_pt <= y_top_pt)
            if vis_pt.any():
                ax.plot(x_line[vis_pt], y_line_pt[vis_pt],
                        color=COL['filtered'], linewidth=1.8, zorder=4)
            else:
                ax.plot(x_line, y_line_pt,
                        color=COL['filtered'], linewidth=1.8, zorder=4)"""

NEW36 = """            x_line = np.linspace(x_vals.min(), x_vals.max(), 300)
            y_line_pt = slope_pt * x_line + intercept_pt
            ax.plot(x_line, y_line_pt,
                    color=COL['filtered'], linewidth=1.8, zorder=4)"""

assert OLD36 in src36, "OLD36 not found in cell 36"
src36 = src36.replace(OLD36, NEW36)
lines = src36.split('\n')
nb3['cells'][36]['source'] = [l+'\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
print('Cell 36: vis mask removed')

# ───────────────────────────────────────────────────────────────────────────────
# FIX 2 — Figure 3 cell 40: remove custom ylim, let matplotlib auto-set
#           Also fix n= label positioning to use whisker cap data coords
# ───────────────────────────────────────────────────────────────────────────────
src40 = ''.join(nb3['cells'][40]['source'])

# Remove the entire ylim block + reposition block
OLD_YLIM = """    ax.set_xlim(0.35, len(regions) + 0.65)
    # y-axis: clip to data percentiles (no outliers)
    y_all_fin = np.concatenate([v for v in vals_list if len(v) > 0])
    y_all_fin = y_all_fin[np.isfinite(y_all_fin)]
    if len(y_all_fin) > 3:
        y_p1  = float(np.percentile(y_all_fin, 1))
        y_p99 = float(np.percentile(y_all_fin, 99))
        buf   = max((y_p99 - y_p1) * 0.12, 0.02)
        y_bot_r = y_p1  - buf
        y_top_r = y_p99 + buf * 2.5   # extra headroom for n= labels
        ax.set_ylim(bottom=y_bot_r, top=y_top_r)
    # reposition n= labels and significance stars within the new ylim
    y_top_lim = ax.get_ylim()[1]
    y_lbl_pos = y_top_lim * 0.92
    for txt in ax.texts:
        if txt.get_text().startswith('n='):
            txt.set_y(y_lbl_pos)"""

NEW_XLIM = "    ax.set_xlim(0.35, len(regions) + 0.65)"

assert OLD_YLIM in src40, "OLD_YLIM block not found in cell 40"
src40 = src40.replace(OLD_YLIM, NEW_XLIM)

# Also fix n= label y position: use whisker cap coords after boxplot is drawn
# Current code sets n= labels at ymax_data * 1.03 before ylim is known.
# Replace that with: use transform-based (axes fraction) positioning so labels
# are always inside the plot area regardless of auto ylim.
OLD_NLABELS = """    # n= labels above boxes
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
                ha='center', va='bottom', fontsize=6.5, color='dimgray')"""

NEW_NLABELS = """    # n= labels and significance stars — use axes-fraction y so they are always visible
    for i, (r, vals) in enumerate(zip(regions, vals_list)):
        ax.text(i + 1, 1.01, f'n={len(vals):,}',
                ha='center', va='bottom', fontsize=6.5, color='dimgray',
                transform=ax.get_xaxis_transform())

    for i, p_wx in enumerate(wilcox_sigs):
        stars = '***' if p_wx < 0.001 else ('**' if p_wx < 0.01 else ('*' if p_wx < 0.05 else 'ns'))
        mid_x = i + 1.5
        ax.text(mid_x, 1.055, stars,
                ha='center', va='bottom', fontsize=6.5, color='dimgray',
                transform=ax.get_xaxis_transform())"""

assert OLD_NLABELS in src40, "OLD_NLABELS not found in cell 40"
src40 = src40.replace(OLD_NLABELS, NEW_NLABELS)

lines = src40.split('\n')
nb3['cells'][40]['source'] = [l+'\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
print('Cell 40: custom ylim removed, n= labels use axes-fraction transform')

with open('figure_3.ipynb', 'w') as f:
    json.dump(nb3, f, indent=1)
print('figure_3.ipynb saved')


# ───────────────────────────────────────────────────────────────────────────────
# FIX 3 — Figure 4 cell 39: same region boxplot ylim fix
# ───────────────────────────────────────────────────────────────────────────────
with open('figure_4.ipynb') as f:
    nb4 = json.load(f)

src39 = ''.join(nb4['cells'][39]['source'])

# Check if it has the same ylim block
has_ylim_block = 'y_all_fin' in src39

if has_ylim_block:
    src39 = src39.replace(OLD_YLIM, NEW_XLIM)
    src39 = src39.replace(OLD_NLABELS, NEW_NLABELS)
    lines = src39.split('\n')
    nb4['cells'][39]['source'] = [l+'\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    print('Cell 39 (fig4): custom ylim removed')
else:
    # Check what n= label positioning looks like
    if 'y_label_pos' in src39:
        src39 = src39.replace(OLD_NLABELS, NEW_NLABELS)
        lines = src39.split('\n')
        nb4['cells'][39]['source'] = [l+'\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        print('Cell 39 (fig4): n= label positioning updated')
    else:
        print('Cell 39 (fig4): no ylim block found, checking structure...')
        for j, line in enumerate(src39.split('\n')):
            if any(k in line for k in ['ylim', 'y_all', 'y_label', 'n=', 'ymax']):
                print(f'  L{j}: {line}')

with open('figure_4.ipynb', 'w') as f:
    json.dump(nb4, f, indent=1)
print('figure_4.ipynb saved')
