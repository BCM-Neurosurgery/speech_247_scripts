import json, re

with open('figure_3.ipynb') as f:
    nb = json.load(f)
cells = nb['cells']

def set_src(cell, src): cell['source'] = [src]

# ── Cell 34: Panel E — add smart top ylim ─────────────────────────────────────
src34 = ''.join(cells[34]['source'])
old_e = '    ax_e.set_ylim(bottom=-0.25)'
new_e = (
    '    y_finite_e = y_all[np.isfinite(y_all)]\n'
    '    if len(y_finite_e) > 3:\n'
    '        q75_e, q25_e = np.percentile(y_finite_e, [75, 25])\n'
    '        y_top_e = q75_e + 3.0 * (q75_e - q25_e)\n'
    '        y_top_e = max(y_top_e, 0.30)\n'
    '    else:\n'
    '        y_top_e = 1.0\n'
    '    ax_e.set_ylim(bottom=-0.25, top=y_top_e)'
)
if old_e in src34:
    src34 = src34.replace(old_e, new_e)
    set_src(cells[34], src34)
    print('Cell 34: Panel E ylim fixed')
else:
    print('Cell 34: NOT FOUND old string', repr(old_e[:60]))

# ── Cell 36: S1 per-patient scatter — add smart top ylim per patient ──────────
src36 = ''.join(cells[36]['source'])
old_s1 = '        ax.set_ylim(bottom=-0.25)'
new_s1 = (
    '        y_finite_pt = y_vals[np.isfinite(y_vals)]\n'
    '        if len(y_finite_pt) > 3:\n'
    '            q75_pt, q25_pt = np.percentile(y_finite_pt, [75, 25])\n'
    '            y_top_pt = q75_pt + 3.0 * (q75_pt - q25_pt)\n'
    '            y_top_pt = max(y_top_pt, 0.20)\n'
    '        else:\n'
    '            y_top_pt = 1.0\n'
    '        ax.set_ylim(bottom=-0.25, top=y_top_pt)'
)
if old_s1 in src36:
    src36 = src36.replace(old_s1, new_s1)
    set_src(cells[36], src36)
    print('Cell 36: S1 ylim fixed')
else:
    print('Cell 36: NOT FOUND old string', repr(old_s1[:60]))

with open('figure_3.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print('figure_3.ipynb saved')
