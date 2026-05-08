#!/usr/bin/env python3
"""
Beta Diversity Analysis — Bray-Curtis PCoA + PERMANOVA (Species Level)
ALL DATASETS COMBINED — Batch vs. Condition Variance Partitioning
-----------------------------------------------------------------------
Control vs CRC only.

Batch vs. Condition Analysis:
  1. Batch-coloured PCoA panel alongside the condition-coloured panel.
  2. PERMANOVA on dataset_name alone (batch effect).
  3. Partial PERMANOVA — condition effect after controlling for dataset.
  4. Variance partitioning (db-RDA style, sequential):
       [Batch alone]     = ω²(dataset)
       [Condition alone] = ω²(condition | dataset controlled)
       [Shared]          = ω²(condition) − ω²(condition | dataset)
       [Residual]        = 1 − batch − condition_partial − shared

Permutations: 9999
"""

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
from skbio.stats.distance import permanova, permdisp, DistanceMatrix

warnings.filterwarnings("ignore", category=RuntimeWarning)

# CONFIGURATION
FILE_PATH    = './data/filtered_nine_crc_final_clean_train.tsv'
PERMUTATIONS = 9999
OUTPUT_FIG   = 'beta_diversity_batch_vs_condition.png'
OUTPUT_CSV   = './results/beta_diversity_batch_vs_condition.csv'

CONDITION_MAP = {
    'control': 'control', 'Control': 'control', 'CONTROL': 'control',
    'adenoma': None,       'Adenoma': None,       'ADENOMA': None, 
    'crc':     'CRC',      'Crc':     'CRC',      'CRC':     'CRC',
    'IBD':     None,       'ibd':     None,
}

METADATA_COLS = ['dataset_name', 'study_condition', 'sampleID',
                 'body_site', 'disease', 'age', 'gender', 'BMI',
                 'country', 'fobt', 'sequencing_instrument']

COLOR_DICT = {
    'control' : '#E41A1C',
    'CRC'     : '#4D4D4D',
}

BATCH_PALETTE = [
    '#E69F00', '#56B4E9', '#009E73', '#F0E442',
    '#0072B2', '#D55E00', '#CC79A7', '#999999', '#000000',
]

DATASET_LABEL_MAP = {
    'FengQ_2015'      : 'Feng et al. (2015)',
    'HanniganGD_2017' : 'Hannigan et al. (2017)',
    'ThomasAM_2019_a' : 'Thomas et al. (2019) (a)',
    'ThomasAM_2019_b' : 'Thomas et al. (2019) (b)',
    'ThomasAM_2019_c' : 'Thomas et al. (2019) (c)',
    'VogtmannE_2016'  : 'Vogtmann et al. (2016)',
    'WirbelJ_2018'    : 'Wirbel et al. (2018)',
    'YuJ_2015'        : 'Yu et al. (2015)',
    'ZellerG_2014'    : 'Zeller et al. (2014)',
}

# HELPER FUNCTIONS
def omega_squared(dist_matrix, grouping):
    d_mat  = dist_matrix.data
    n      = d_mat.shape[0]
    groups = np.unique(grouping)
    k      = len(groups)

    ss_total  = np.sum(d_mat ** 2) / (2 * n)
    ss_within = 0.0
    for g in groups:
        idx = np.where(grouping == g)[0]
        if len(idx) > 1:
            g_dists    = d_mat[np.ix_(idx, idx)]
            ss_within += np.sum(g_dists ** 2) / (2 * len(idx))

    ss_between  = ss_total - ss_within
    df_between  = k - 1
    df_within   = n - k
    ms_within   = ss_within / df_within if df_within > 0 else 0.0
    numerator   = ss_between - df_between * ms_within
    denominator = ss_total   + ms_within

    return float(numerator / denominator) if denominator > 0 else np.nan


def residual_distance_matrix(dm, grouping):
    from scipy.spatial.distance import cdist
    n_dims   = min(dm.data.shape[0] - 1, 50)
    pcoa_res = pcoa(dm, number_of_dimensions=n_dims)
    coords   = pcoa_res.samples.values
    ids      = list(pcoa_res.samples.index)

    groups    = np.array(grouping)
    residuals = coords.copy()
    for g in np.unique(groups):
        mask     = groups == g
        centroid = coords[mask].mean(axis=0)
        residuals[mask] -= centroid

    resid_dm = cdist(residuals, residuals, metric='euclidean')
    return DistanceMatrix(resid_dm, ids=ids)


def get_permanova_pvalue(result):
    if 'p-value' in result:
        return result['p-value']
    if 'p_value' in result:
        return result['p_value']
    return np.nan


def sig_stars(p):
    if pd.isna(p): return 'n/a'
    if p < 0.001:  return '***'
    if p < 0.01:   return '**'
    if p < 0.05:   return '*'
    return 'ns'


def fmt(val, decimals=4):
    return f"{val:.{decimals}f}" if not pd.isna(val) else "N/A"


def _omega_label(o):
    if pd.isna(o): return 'n/a'
    if o < 0.01:   return 'negligible'
    if o < 0.06:   return 'small'
    if o < 0.14:   return 'medium'
    return 'large'


# 1. LOADING DATA
print("Loading data...")
df = pd.read_csv(FILE_PATH, sep='\t', index_col=0)

metadata = df[['dataset_name', 'study_condition', 'sampleID']].copy()
metadata['study_condition'] = (
    metadata['study_condition'].str.strip().map(CONDITION_MAP)
)

unmapped = metadata['study_condition'].isna()
if unmapped.sum() > 0:
    raw_vals = df.loc[unmapped, 'study_condition'].unique()
    print(f"  Excluding {unmapped.sum()} sample(s) with "
          f"unmapped/excluded condition(s): {list(raw_vals)}")
    metadata = metadata[~unmapped].copy()

# 2. SPECIES DATA
species_cols = [c for c in df.columns if c not in METADATA_COLS]
species_df   = ( df[species_cols].loc[metadata.index].apply(pd.to_numeric, errors='coerce').fillna(0))

# 3. ZERO-SUM GUARD
row_sums      = species_df.sum(axis=1)
zero_sum_mask = row_sums == 0
if zero_sum_mask.sum() > 0:
    print(f"  Excluding {zero_sum_mask.sum()} zero-sum sample(s).")
    metadata   = metadata[~zero_sum_mask].copy()
    species_df = species_df[~zero_sum_mask]
    row_sums   = row_sums[~zero_sum_mask]


# 4. TSS NORMALISATION 
bad_sum_mask = ~np.isclose(row_sums, 100.0, atol=0.1)
if bad_sum_mask.sum() > 0:
    print(f"  Excluding {bad_sum_mask.sum()} sample(s) with out-of-range row sums.")
    metadata   = metadata[~bad_sum_mask].copy()
    species_df = species_df[~bad_sum_mask]
    row_sums   = row_sums[~bad_sum_mask]
else:
    print(f"  TSS check passed: all {len(row_sums)} samples sum to 100 ± 0.1.")

species_rel = species_df.div(row_sums, axis=0)

prop_sums = species_rel.sum(axis=1)
bad_prop  = ~np.isclose(prop_sums, 1.0, atol=1e-4)
if bad_prop.sum() > 0:
    print(f"  {bad_prop.sum()} sample(s) failed post-TSS check — excluding.")
    metadata    = metadata[~bad_prop].copy()
    species_rel = species_rel[~bad_prop]
else:
    print(f"  Normalisation passed. "
          f"Matrix: {species_rel.shape[0]} samples × {species_rel.shape[1]} species.")

print(f"\n  Total samples after QC: {len(metadata)}")
for cond in ['control', 'CRC']:
    n = (metadata['study_condition'] == cond).sum()
    print(f"    {cond:<10}: {n}")

datasets = sorted(metadata['dataset_name'].unique())
DATASET_COLOR = {d: BATCH_PALETTE[i % len(BATCH_PALETTE)]
                 for i, d in enumerate(datasets)}
print(f"\n  Datasets ({len(datasets)}): {datasets}")

# 5. BUILD SUBSET  of Control vs CRC 
idx_cc     = metadata[metadata['study_condition'].isin(['control', 'CRC'])].index
meta_cc    = metadata.loc[idx_cc].copy()
species_cc = species_rel.loc[idx_cc]

print(f"\n  CC — control: {(meta_cc['study_condition']=='control').sum()} | "
      f"CRC: {(meta_cc['study_condition']=='CRC').sum()}")

# 6. DISTANCE MATRIX + PCoA
print("\nComputing Bray-Curtis distance matrix and PCoA...")
print("  (This may take several minutes for large pooled cohorts)")

dm_cc   = beta_diversity("braycurtis", species_cc.values, ids=species_cc.index)
pcoa_cc = pcoa(dm_cc, number_of_dimensions=2)
pc1_cc  = pcoa_cc.proportion_explained.iloc[0] * 100
pc2_cc  = pcoa_cc.proportion_explained.iloc[1] * 100
ev_cc   = pc1_cc + pc2_cc
print(f"  CC distance matrix done  — EV%: {ev_cc:.1f}%")

# 7. CONDITION PERMANOVA + PERMDISP
print(f"\nRunning condition PERMANOVA + PERMDISP (permutations={PERMUTATIONS})...")

perm_cc   = permanova(dm_cc, meta_cc, column='study_condition',
                      permutations=PERMUTATIONS)
p_cc      = get_permanova_pvalue(perm_cc)
omega2_cc = omega_squared(dm_cc, meta_cc['study_condition'].values)
print(f"  CC condition PERMANOVA — p={p_cc:.6f}  {sig_stars(p_cc)}")

try:
    disp_cc   = permdisp(dm_cc, meta_cc['study_condition'],
                         permutations=PERMUTATIONS)
    p_disp_cc = get_permanova_pvalue(disp_cc)
except Exception as e:
    print(f"  CC PERMDISP failed: {e}")
    p_disp_cc = np.nan


# 8. BATCH EFFECT ANALYSIS
print(f"\n{'='*70}")
print("  BATCH vs. CONDITION VARIANCE PARTITIONING")
print(f"{'='*70}")

# 8a. PERMANOVA on dataset_name
print("\n  8a. PERMANOVA on dataset_name (batch effect)...")

perm_batch_cc   = permanova(dm_cc, meta_cc, column='dataset_name',
                             permutations=PERMUTATIONS)
p_batch_cc      = get_permanova_pvalue(perm_batch_cc)
omega2_batch_cc = omega_squared(dm_cc, meta_cc['dataset_name'].values)
print(f"    CC batch PERMANOVA — p={p_batch_cc:.6f}  {sig_stars(p_batch_cc)}  "
      f"ω²={omega2_batch_cc:.4f} ({_omega_label(omega2_batch_cc)})")

try:
    disp_batch_cc   = permdisp(dm_cc, meta_cc['dataset_name'],
                               permutations=PERMUTATIONS)
    p_disp_batch_cc = get_permanova_pvalue(disp_batch_cc)
    print(f"    CC batch PERMDISP  — p={fmt(p_disp_batch_cc)}  {sig_stars(p_disp_batch_cc)}")
except Exception as e:
    print(f"  CC batch PERMDISP failed: {e}")
    p_disp_batch_cc = np.nan

# 8b. Partial PERMANOVA
print("\n  8b. Partial PERMANOVA — condition | dataset (residual matrix)...")

print("    Computing residual distance matrix for CC...")
dm_resid_cc       = residual_distance_matrix(dm_cc, meta_cc['dataset_name'].values)
perm_partial_cc   = permanova(dm_resid_cc, meta_cc, column='study_condition',
                               permutations=PERMUTATIONS)
p_partial_cc      = get_permanova_pvalue(perm_partial_cc)
omega2_partial_cc = omega_squared(dm_resid_cc, meta_cc['study_condition'].values)
print(f"    CC partial PERMANOVA — p={p_partial_cc:.6f}  {sig_stars(p_partial_cc)}  "
      f"ω²={omega2_partial_cc:.4f} ({_omega_label(omega2_partial_cc)})")

# 8c. Variance partitioning
print("\n  8c. Variance partitioning...")

def vp(omega2_batch, omega2_cond_marginal, omega2_cond_partial):
    batch    = max(omega2_batch, 0)
    cond     = max(omega2_cond_partial, 0)
    shared   = max(omega2_cond_marginal - omega2_cond_partial, 0)
    residual = max(1 - batch - cond - shared, 0)
    return dict(batch=batch, condition=cond, shared=shared,
                residual=residual,
                total_explained=batch + cond + shared)

vp_cc = vp(omega2_batch_cc, omega2_cc, omega2_partial_cc)

print(f"\n    CC variance partitioning:")
print(f"      Batch alone     : {vp_cc['batch']:.4f}  ({vp_cc['batch']*100:.1f}%)")
print(f"      Condition alone : {vp_cc['condition']:.4f}  ({vp_cc['condition']*100:.1f}%)")
print(f"      Shared          : {vp_cc['shared']:.4f}  ({vp_cc['shared']*100:.1f}%)")
print(f"      Residual        : {vp_cc['residual']:.4f}  ({vp_cc['residual']*100:.1f}%)")
print(f"      Total explained : {vp_cc['total_explained']:.4f}  ({vp_cc['total_explained']*100:.1f}%)")

# 9. PRINT FULL RESULTS SUMMARY
print("\n" + "="*75)
print("  FULL STATISTICAL RESULTS  |  All datasets combined  |  Species level")
print("  ω² interpretation: <0.01 negligible | 0.01–0.06 small | "
      "0.06–0.14 medium | >0.14 large")
print("="*75)

print(f"\n  Control vs CRC")
print(f"    ── Condition (marginal) ──")
print(f"    PERMANOVA  p = {p_cc:.6f}  {sig_stars(p_cc)}")
print(f"    PERMDISP   p = {fmt(p_disp_cc)}  {sig_stars(p_disp_cc)}")
print(f"    Omega²       = {fmt(omega2_cc, 4)}  ({_omega_label(omega2_cc)})")
print(f"    EV (PC1+PC2) = {ev_cc:.1f}%")
print(f"    ── Batch (dataset_name) ──")
print(f"    PERMANOVA  p = {p_batch_cc:.6f}  {sig_stars(p_batch_cc)}")
print(f"    PERMDISP   p = {fmt(p_disp_batch_cc)}  {sig_stars(p_disp_batch_cc)}")
print(f"    Omega²       = {fmt(omega2_batch_cc, 4)}  ({_omega_label(omega2_batch_cc)})")
print(f"    ── Condition | Batch controlled (partial) ──")
print(f"    PERMANOVA  p = {p_partial_cc:.6f}  {sig_stars(p_partial_cc)}")
print(f"    Omega²       = {fmt(omega2_partial_cc, 4)}  ({_omega_label(omega2_partial_cc)})")
print(f"    ── Variance partitioning ──")
print(f"    Batch alone     {vp_cc['batch']*100:.1f}%  |  "
      f"Condition alone {vp_cc['condition']*100:.1f}%  |  "
      f"Shared {vp_cc['shared']*100:.1f}%  |  "
      f"Residual {vp_cc['residual']*100:.1f}%")

# 10. SAVE CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

results_df = pd.DataFrame([{
    'comparison'                     : 'control vs CRC',
    'n_control'                      : (meta_cc['study_condition'] == 'control').sum(),
    'n_comparison'                   : (meta_cc['study_condition'] == 'CRC').sum(),
    'PERMANOVA_p'                    : round(p_cc, 6),
    'PERMANOVA_sig'                  : sig_stars(p_cc),
    'PERMDISP_p'                     : round(p_disp_cc, 6) if not pd.isna(p_disp_cc) else np.nan,
    'PERMDISP_sig'                   : sig_stars(p_disp_cc),
    'omega2_condition'               : max(round(omega2_cc, 4), 0),
    'omega2_condition_interp'        : _omega_label(omega2_cc),
    'EV_pct'                         : round(ev_cc, 2),
    'PERMANOVA_batch_p'              : round(p_batch_cc, 6),
    'PERMANOVA_batch_sig'            : sig_stars(p_batch_cc),
    'PERMDISP_batch_p'               : round(p_disp_batch_cc, 6) if not pd.isna(p_disp_batch_cc) else np.nan,
    'PERMDISP_batch_sig'             : sig_stars(p_disp_batch_cc),
    'omega2_batch'                   : max(round(omega2_batch_cc, 4), 0),
    'omega2_batch_interp'            : _omega_label(omega2_batch_cc),
    'PERMANOVA_partial_p'            : round(p_partial_cc, 6),
    'PERMANOVA_partial_sig'          : sig_stars(p_partial_cc),
    'omega2_condition_partial'       : max(round(omega2_partial_cc, 4), 0),
    'omega2_condition_partial_interp': _omega_label(omega2_partial_cc),
    'VP_batch_pct'                   : round(vp_cc['batch'] * 100, 2),
    'VP_condition_pct'               : round(vp_cc['condition'] * 100, 2),
    'VP_shared_pct'                  : round(vp_cc['shared'] * 100, 2),
    'VP_residual_pct'                : round(vp_cc['residual'] * 100, 2),
    'VP_total_explained_pct'         : round(vp_cc['total_explained'] * 100, 2),
}])

results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n  Stats saved → {OUTPUT_CSV}")

# 11. PLOT — 1 row, 2 panels:
#     [batch PCoA]  |  [condition PCoA]
print("\nRendering figure...")


def confidence_ellipse(ax, x, y, color, n_std=2.448,
                       fill_alpha=0.10, line_alpha=0.75, lw=1.5):
    if len(x) < 3:
        return
    cov        = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order      = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta      = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w          = 2 * n_std * np.sqrt(vals[0])
    h          = 2 * n_std * np.sqrt(vals[1])
    cx, cy     = np.mean(x), np.mean(y)
    ax.add_patch(Ellipse(xy=(cx, cy), width=w, height=h, angle=theta,
                         facecolor=color, edgecolor='none',
                         alpha=fill_alpha, zorder=1))
    ax.add_patch(Ellipse(xy=(cx, cy), width=w, height=h, angle=theta,
                         facecolor='none', edgecolor=color,
                         alpha=line_alpha, lw=lw, linestyle='--', zorder=2))


def draw_pcoa_batch(ax, pcoa_result, meta_df, pc1, pc2, title):
    coords = pcoa_result.samples[['PC1', 'PC2']].copy()
    coords = coords.join(meta_df[['dataset_name']])

    for ds in sorted(meta_df['dataset_name'].unique()):
        sub       = coords[coords['dataset_name'] == ds]
        label_str = DATASET_LABEL_MAP.get(ds, ds)
        ax.scatter(sub['PC1'], sub['PC2'],
                   color=DATASET_COLOR[ds],
                   label=label_str,
                   alpha=0.55, s=25,
                   edgecolors='white', linewidth=0.2)
        confidence_ellipse(ax, sub['PC1'].values, sub['PC2'].values,
                           color=DATASET_COLOR[ds])

    ax.set_xlabel(f"PCo1 ({pc1:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PCo2 ({pc2:.1f}%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.legend(loc='lower right', fontsize=6, framealpha=0.85,
              edgecolor='#cccccc', markerscale=0.8,
              title='Dataset', title_fontsize=6)
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.set_facecolor('#fafafa')


def draw_pcoa_condition(ax, pcoa_result, meta_df, group_col,
                        group_order, pc1, pc2, ev, title,
                        p_val, p_disp, omega2,
                        p_batch, p_disp_batch, o2_batch,
                        p_part, o2_part):
    coords = pcoa_result.samples[['PC1', 'PC2']].copy()
    coords = coords.join(meta_df[[group_col]])

    for grp in group_order:
        sub = coords[coords[group_col] == grp]
        if sub.empty:
            continue
        ax.scatter(sub['PC1'], sub['PC2'],
                   color=COLOR_DICT.get(grp, '#999999'),
                   label=grp, alpha=0.65, s=30,
                   edgecolors='white', linewidth=0.3)
        confidence_ellipse(ax, sub['PC1'].values, sub['PC2'].values,
                           color=COLOR_DICT.get(grp, '#999999'))

    ax.set_xlabel(f"PCo1 ({pc1:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PCo2 ({pc2:.1f}%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)

    stats_text = (
        f"PERMANOVA: p={p_val:.4f} {sig_stars(p_val)}  |  PERMDISP: p={fmt(p_disp)} {sig_stars(p_disp)}\n"
        f"Condition ω² = {max(omega2, 0):.4f}\n"
        f"────────────────────────────────────────────────\n"
        f"Batch PERMANOVA: p={p_batch:.4f} {sig_stars(p_batch)}  |  PERMDISP: p={fmt(p_disp_batch)} {sig_stars(p_disp_batch)}\n"
        f"Batch ω² = {max(o2_batch, 0):.4f}\n"
        f"Partial ω² (cond|batch) = {max(o2_part, 0):.4f}\n"
        f"EV = {ev:.1f}%"
    )
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            va='top', ha='left', fontsize=7.5,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor='#cccccc', alpha=0.9))

    handles = [mpatches.Patch(color=COLOR_DICT.get(g, '#999999'), label=g)
               for g in group_order]
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              framealpha=0.9, edgecolor='#cccccc')
    ax.axhline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.axvline(0, color='grey', lw=0.5, ls='--', alpha=0.4)
    ax.set_facecolor('#fafafa')


# ── Single row: 2 panels ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    'Beta Diversity — Species Level  |  All Datasets Combined\n'
    'Bray-Curtis PCoA  ·  Batch vs. Condition  |  Control vs CRC',
    fontsize=13, fontweight='bold', y=1.02
)

draw_pcoa_batch(
    ax=axes[0],
    pcoa_result=pcoa_cc,
    meta_df=meta_cc,
    pc1=pc1_cc, pc2=pc2_cc,
    title='Coloured by dataset (batch)',
)

draw_pcoa_condition(
    ax=axes[1],
    pcoa_result=pcoa_cc,
    meta_df=meta_cc,
    group_col='study_condition',
    group_order=['control', 'CRC'],
    pc1=pc1_cc, pc2=pc2_cc, ev=ev_cc,
    title='Control vs CRC (condition)',
    p_val=p_cc, p_disp=p_disp_cc, omega2=omega2_cc,
    p_batch=p_batch_cc, p_disp_batch=p_disp_batch_cc,
    o2_batch=omega2_batch_cc,
    p_part=p_partial_cc, o2_part=omega2_partial_cc,
)

sns.despine(fig=fig)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight')
plt.show()
print(f"Figure saved → {OUTPUT_FIG}")
print(f"Stats  saved → {OUTPUT_CSV}")