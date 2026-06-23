import json, copy, re

# ── 1. Figure 4: rename panels ────────────────────────────────────────────────
with open('figure_4.ipynb') as f:
    nb4 = json.load(f)

# replacement map: (old_text, new_text) pairs applied to ALL cells
# Order matters — do longer/more-specific strings first
REPLACEMENTS = [
    # PNG filenames
    ('DE_combined_comparison.png',              'C_combined_comparison.png'),
    ('G_paired_wholestay_vs_perday_filtered.png', 'D_paired_wholestay_vs_perday_filtered.png'),
    ('C_decoder_vs_n_units_perday.png',         'E_decoder_vs_n_units_perday.png'),
    ('E_pooled_epoch_wls_scatter.png',          'F_pooled_epoch_wls_scatter.png'),
    ('F_epoch_wls_pie.png',                     'G_epoch_wls_pie.png'),
    ('G_encoding_vs_decoding_dispersion.png',   'H_encoding_vs_decoding_dispersion.png'),
    ('H_decoding_by_region.png',                'I_decoding_by_region.png'),
    # Markdown headers  (match "Panel X" labels)
    ('Panel D+E —',   'Panel C —'),
    ('Panel D+E:',    'Panel C:'),
    # whole-stay panel: two occurrences of "Panel G" — only the first one (whole-stay vs per-day)
    # Use sentinel to avoid double-replacing  
    # We'll handle these carefully below
]

def apply_simple(src):
    for old, new in REPLACEMENTS:
        src = src.replace(old, new)
    return src

# For "Panel G" we have two different panels; we need positional / context-aware replacement
# Cell 19-20: whole-stay vs per-day → Panel D
# Cell 36-37: CV comparison        → Panel H
PANEL_G_SUBS = {
    19: [('Panel G —', 'Panel D —'), ('Panel G:', 'Panel D:')],
    20: [('G_paired', 'D_paired')],
    36: [('Panel G —', 'Panel H —'), ('Panel G:', 'Panel H:'),
         ('## 16. Panel G', '## 16. Panel H')],
    37: [('G_encoding_vs_decoding', 'H_encoding_vs_decoding')],  # handled by REPLACEMENTS too but fine
    38: [('Panel H —', 'Panel I —'), ('Panel H:', 'Panel I:'),
         ('## 17. Panel H', '## 17. Panel I')],
    39: [('H_decoding_by_region', 'I_decoding_by_region')],
    40: [('A–CV', 'A–I')],
    41: [
        ("'D+E — 24/7 vs convo & podcast'",         "'C — 24/7 vs convo & podcast'"),
        ("'G — whole-stay vs per-day (paired)'",     "'D — whole-stay vs per-day (paired)'"),
        ("'C — decoder vs n_units (per-day)'",       "'E — decoder vs n_units (per-day)'"),
        ("'E — pooled epoch scatter (WLS)'",         "'F — pooled epoch scatter (WLS)'"),
        ("'F — pie charts (WLS summary)'",           "'G — pie charts (WLS summary)'"),
        ("'G — encoding vs decoding dispersion'",    "'H — encoding vs decoding dispersion'"),
        ("'H — decoding by brain region'",           "'I — decoding by brain region'"),
    ],
}

# Also: cell 15 markdown header
CELL_PATCHES = {
    15: [('## 7. Panel D+E —', '## 7. Panel C —'),
         ('Panel D+E:', 'Panel C:'),
         ('D+E', 'C')],
    19: [('## 9. Panel G —', '## 9. Panel D —'),
         ('Whole-Stay', 'Whole-Stay'),
         ('Panel G —', 'Panel D —')],
    21: [('Panel C (', 'Panel E (')],
    23: [('## 11. Panel C —', '## 11. Panel E —'),
         ('Panel C:', 'Panel E:')],
    32: [('## 14. Panel E —', '## 14. Panel F —'),
         ('Panel E:', 'Panel F:'), ('Panel E —', 'Panel F —')],
    34: [('## 15. Panel F —', '## 15. Panel G —'),
         ('Panel F:', 'Panel G:'), ('Panel F —', 'Panel G —')],
    36: [('## 16. Panel G —', '## 16. Panel H —'),
         ('Panel G:', 'Panel H:'), ('Panel G —', 'Panel H —')],
    38: [('## 17. Panel H —', '## 17. Panel I —'),
         ('Panel H:', 'Panel I:'), ('Panel H —', 'Panel I —')],
    40: [('A–CV', 'A–I')],
}

for i, cell in enumerate(nb4['cells']):
    src = ''.join(cell['source'])
    src = apply_simple(src)

    patches = CELL_PATCHES.get(i, [])
    for old, new in patches:
        src = src.replace(old, new)

    # Write back as single string list
    if isinstance(cell['source'], list):
        lines = src.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        cell['source'] = src

# Also fix title cell (cell 0) markdown
title_src = ''.join(nb4['cells'][0]['source'])
title_src = title_src.replace('- **D+E**', '- **C**')
title_src = title_src.replace('- **G** — whole-stay vs per-day', '- **D** — whole-stay vs per-day')
title_src = title_src.replace('- **C** — decoder', '- **E** — decoder')
title_src = title_src.replace('- **E** — pooled', '- **F** — pooled')
title_src = title_src.replace('- **F** — pie', '- **G** — pie')
title_src = title_src.replace('- **G** — encoding vs decoding', '- **H** — encoding vs decoding')
title_src = title_src.replace('- **H** — decoding by brain', '- **I** — decoding by brain')
lines = title_src.split('\n')
nb4['cells'][0]['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open('figure_4.ipynb', 'w') as f:
    json.dump(nb4, f, indent=1)
print('figure_4.ipynb panel names updated')


# ── 2. Figure 3: fix region boxplot y-axis ───────────────────────────────────
with open('figure_3.ipynb') as f:
    nb3 = json.load(f)

# Find the region cell (cell 40)
for i, cell in enumerate(nb3['cells']):
    src = ''.join(cell['source'])
    if 'draw_region_boxplot' in src and 'def draw_region_boxplot' in src:
        print(f'Fixing figure_3 region cell: {i}')
        
        # Fix 1: showfliers=False
        src = src.replace(
            'showfliers=showfliers,',
            'showfliers=False,',
        )
        
        # Fix 2: add explicit ylim after ax.set_xlim line, using percentile-based limits
        # Find the closing line and insert ylim
        OLD_XLIM = "    ax.set_xlim(0.35, len(regions) + 0.65)"
        NEW_XLIM = (
            "    ax.set_xlim(0.35, len(regions) + 0.65)\n"
            "    # y-axis: clip to data percentiles (no outliers)\n"
            "    y_all_fin = np.concatenate([v for v in vals_list if len(v) > 0])\n"
            "    y_all_fin = y_all_fin[np.isfinite(y_all_fin)]\n"
            "    if len(y_all_fin) > 3:\n"
            "        y_p1  = float(np.percentile(y_all_fin, 1))\n"
            "        y_p99 = float(np.percentile(y_all_fin, 99))\n"
            "        buf   = max((y_p99 - y_p1) * 0.12, 0.02)\n"
            "        y_bot_r = y_p1  - buf\n"
            "        y_top_r = y_p99 + buf * 2.5   # extra headroom for n= labels\n"
            "        ax.set_ylim(bottom=y_bot_r, top=y_top_r)\n"
            "    # reposition n= labels and significance stars within the new ylim\n"
            "    y_top_lim = ax.get_ylim()[1]\n"
            "    y_lbl_pos = y_top_lim * 0.92\n"
            "    for txt in ax.texts:\n"
            "        if txt.get_text().startswith('n='):\n"
            "            txt.set_y(y_lbl_pos)"
        )
        src = src.replace(OLD_XLIM, NEW_XLIM)
        
        # Write back
        lines = src.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        print('  Applied showfliers=False + ylim fix')
        break

with open('figure_3.ipynb', 'w') as f:
    json.dump(nb3, f, indent=1)
print('figure_3.ipynb region y-axis fixed')
