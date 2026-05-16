#!/usr/bin/env python3
"""
Beta Diversity Analysis — Bray-Curtis PCoA + PERMANOVA (Genus Level)
ALL DATASETS COMBINED  |  SEEN DATASET
Batch vs. Condition Variance Partitioning  |  Control vs CRC only
-----------------------------------------------------------------------
Loads pre-split genus-level data from ./data/prepared/genus/.
No aggregation, no splitting — all preprocessing already done.

Batch vs. Condition Analysis:
  1. Batch-coloured PCoA panel alongside the condition-coloured panel.
  2. PERMANOVA + PERMDISP on dataset_name alone (batch effect).
  3. Partial PERMANOVA — condition effect after controlling for dataset.
  4. Variance partitioning (db-RDA style, sequential).

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
from pd_utils import relative_abundance

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIGURATION
# ============================================================
PREPARED_DIR = "./data/prepared/genus/"
PERMUTATIONS = 9999
OUTPUT_FIG   = "beta_diversity_batch_vs_condition_genus.png"
OUTPUT_CSV   = "./results/beta_diversity_batch_vs_condition_genus.csv"

METADATA_COLS = [
    "dataset_name", "sampleID", "subjectID",
    "study_condition", "disease",
    "age", "gender", "country", "BMI", "fobt",
]

COLOR_DICT = {
    "control": "#E41A1C",
    "CRC":     "#4D4D4D",
}

BATCH_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999", "#000000",
]

DATASET_LABEL_MAP = {
    "FengQ_2015":        "Feng et al. (2015)",
    "HanniganGD_2017":   "Hannigan et al. (2017)",
    "ThomasAM_2019_a":   "Thomas et al. (2019) (a)",
    "ThomasAM_2019_b":   "Thomas et al. (2019) (b)",
    "ThomasAM_2019_c":   "Thomas et al. (2019) (c)",
    "VogtmannE_2016":    "Vogtmann et al. (2016)",
    "WirbelJ_2018":      "Wirbel et al. (2018)",
    "YuJ_2015":          "Yu et al. (2015)",
    "ZellerG_2014":      "Zeller et al. (2014)",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def omega_squared(dist_matrix, grouping):
    d_mat  = dist_matrix.data
    n      = d_mat.shape[0]
    groups = np.unique(grouping)
    k      = len(groups)

    ss_total  = np.sum(d_mat**2) / (2 * n)
    ss_within = 0.0
    for g in groups:
        idx = np.where(grouping == g)[0]
        if len(idx) > 1:
            g_dists    = d_mat[np.ix_(idx, idx)]
            ss_within += np.sum(g_dists**2) / (2 * len(idx))

    ss_between  = ss_total - ss_within
    df_between  = k - 1
    df_within   = n - k
    ms_within   = ss_within / df_within if df_within > 0 else 0.0
    numerator   = ss_between - df_between * ms_within
    denominator = ss_total + ms_within

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

    resid_dm = cdist(residuals, residuals, metric="euclidean")
    return DistanceMatrix(resid_dm, ids=ids)


def get_permanova_pvalue(result):
    if "p-value" in result: return result["p-value"]
    if "p_value" in result: return result["p_value"]
    return np.nan


def sig_stars(p):
    if pd.isna(p):  return "n/a"
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return "ns"


def fmt(val, decimals=4):
    return f"{val:.{decimals}f}" if not pd.isna(val) else "N/A"


def _omega_label(o):
    if pd.isna(o):  return "n/a"
    if o < 0.01:    return "negligible"
    if o < 0.06:    return "small"
    if o < 0.14:    return "medium"
    return "large"


# ============================================================
# 1. LOAD PREPARED DATA
# ============================================================
print("Loading prepared data...")

X_train_df     = pd.read_csv(os.path.join(PREPARED_DIR, "X_train_full.csv"))
y_train        = pd.read_csv(os.path.join(PREPARED_DIR, "y_train_full.csv")).squeeze()
surviving_taxa = (
    pd.read_csv(os.path.join(PREPARED_DIR, "all_taxa_full.csv"), header=None)
    .squeeze()
    .tolist()
)

print(f"  X_train : {X_train_df.shape}")
print(f"  y_train : {len(y_train)}  "
      f"(CRC: {int(y_train.sum())}, Control: {int((y_train == 0).sum())})")
print(f"  Taxa    : {len(surviving_taxa)} genera")

# ── build metadata and genus matrix ──────────────────────────
meta_cols_present = [c for c in METADATA_COLS if c in X_train_df.columns]

if "study_condition" in X_train_df.columns:
    metadata = X_train_df[meta_cols_present].copy().reset_index(drop=True)
else:
    metadata = pd.DataFrame(index=X_train_df.index)
    if "dataset_name" in X_train_df.columns:
        metadata["dataset_name"] = X_train_df["dataset_name"].values
    if "sampleID" in X_train_df.columns:
        metadata["sampleID"] = X_train_df["sampleID"].values
    metadata["study_condition"] = y_train.map({1: "CRC", 0: "control"}).values

metadata = metadata.reset_index(drop=True)

available_taxa = [c for c in surviving_taxa if c in X_train_df.columns]
genus_df = (
    X_train_df[available_taxa]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .reset_index(drop=True)
)
feature_cols = available_taxa

print(
    f"\n  Pooled train: {len(metadata)} samples  "
    f"(control={(metadata['study_condition'] == 'control').sum()}, "
    f"CRC={(metadata['study_condition'] == 'CRC').sum()})"
)
print(f"  Genera: {len(feature_cols)}")


# ============================================================
# 2. ZERO-SUM GUARD
# ============================================================
row_sums      = genus_df[feature_cols].sum(axis=1)
zero_sum_mask = row_sums == 0
if zero_sum_mask.sum() > 0:
    print(f"  Excluding {zero_sum_mask.sum()} zero-sum sample(s).")
    keep     = ~zero_sum_mask
    genus_df = genus_df[keep].reset_index(drop=True)
    metadata = metadata[keep].reset_index(drop=True)
    row_sums = genus_df[feature_cols].sum(axis=1)

# ============================================================
# 3. ROW SUM INFO — no exclusion, data is verified correct
# ============================================================
print(
    f"  Row sum range: {row_sums.min():.4f} – {row_sums.max():.4f}  "
    f"({len(row_sums)} samples retained)"
)

genus_norm        = relative_abundance(genus_df, feature_cols=feature_cols)
genus_proportions = genus_norm[feature_cols]

# Post-normalisation sanity check
prop_sums = genus_proportions.sum(axis=1)
bad_prop  = ~np.isclose(prop_sums, 1.0, atol=1e-4)
if bad_prop.sum() > 0:
    print(f"  {bad_prop.sum()} sample(s) failed post-normalisation check — excluding.")
    keep              = ~bad_prop
    genus_df          = genus_df[keep].reset_index(drop=True)
    metadata          = metadata[keep].reset_index(drop=True)
    genus_proportions = genus_proportions[keep].reset_index(drop=True)
else:
    print(
        f"  Normalisation passed. "
        f"Matrix: {genus_proportions.shape[0]} samples × {genus_proportions.shape[1]} genera."
    )

print(f"\n  Total train samples after QC: {len(metadata)}")
for cond in ["control", "CRC"]:
    print(f"    {cond:<10}: {(metadata['study_condition'] == cond).sum()}")

datasets = sorted(metadata["dataset_name"].unique())
DATASET_COLOR = {
    d: BATCH_PALETTE[i % len(BATCH_PALETTE)] for i, d in enumerate(datasets)
}
print(f"\n  Datasets ({len(datasets)}): {datasets}")


# ============================================================
# 4. BUILD SUBSET — Control vs CRC
# ============================================================
cc_mask  = metadata["study_condition"].isin(["control", "CRC"])
meta_cc  = metadata[cc_mask].reset_index(drop=True)
genus_cc = genus_proportions[cc_mask].reset_index(drop=True)

print(
    f"\n  CC — control: {(meta_cc['study_condition']=='control').sum()} | "
    f"CRC: {(meta_cc['study_condition']=='CRC').sum()}"
)


# ============================================================
# 5. DISTANCE MATRIX + PCoA
# ============================================================
print("\nComputing Bray-Curtis distance matrix and PCoA...")
print("  (This may take several minutes for large pooled cohorts)")

dm_cc   = beta_diversity("braycurtis", genus_cc.values, ids=genus_cc.index)
pcoa_cc = pcoa(dm_cc, number_of_dimensions=2)
pc1_cc  = pcoa_cc.proportion_explained.iloc[0] * 100
pc2_cc  = pcoa_cc.proportion_explained.iloc[1] * 100
ev_cc   = pc1_cc + pc2_cc
print(f"  CC distance matrix done  — EV%: {ev_cc:.1f}%")


# ============================================================
# 6. CONDITION PERMANOVA + PERMDISP
# ============================================================
print(f"\nRunning PERMANOVA + PERMDISP (condition, permutations={PERMUTATIONS})...")

perm_cc   = permanova(dm_cc, meta_cc, column="study_condition", permutations=PERMUTATIONS)
p_cc      = get_permanova_pvalue(perm_cc)
omega2_cc = omega_squared(dm_cc, meta_cc["study_condition"].values)
print(f"  CC condition PERMANOVA — p={p_cc:.6f}  {sig_stars(p_cc)}")

try:
    disp_cc   = permdisp(dm_cc, meta_cc["study_condition"], permutations=PERMUTATIONS)
    p_disp_cc = get_permanova_pvalue(disp_cc)
    print(f"  CC condition PERMDISP  — p={fmt(p_disp_cc)}  {sig_stars(p_disp_cc)}")
except Exception as e:
    print(f"  CC PERMDISP failed: {e}")
    p_disp_cc = np.nan


# ============================================================
# 7. BATCH EFFECT ANALYSIS
# ============================================================
print(f"\n{'='*70}")
print("  BATCH vs. CONDITION VARIANCE PARTITIONING")
print(f"{'='*70}")

print("\n  7a. PERMANOVA + PERMDISP on dataset_name (batch effect)...")
perm_batch_cc   = permanova(dm_cc, meta_cc, column="dataset_name", permutations=PERMUTATIONS)
p_batch_cc      = get_permanova_pvalue(perm_batch_cc)
omega2_batch_cc = omega_squared(dm_cc, meta_cc["dataset_name"].values)
print(
    f"    CC batch PERMANOVA — p={p_batch_cc:.6f}  {sig_stars(p_batch_cc)}  "
    f"ω²={omega2_batch_cc:.4f} ({_omega_label(omega2_batch_cc)})"
)

try:
    disp_batch_cc   = permdisp(dm_cc, meta_cc["dataset_name"], permutations=PERMUTATIONS)
    p_disp_batch_cc = get_permanova_pvalue(disp_batch_cc)
    print(f"    CC batch PERMDISP  — p={fmt(p_disp_batch_cc)}  {sig_stars(p_disp_batch_cc)}")
except Exception as e:
    print(f"  CC batch PERMDISP failed: {e}")
    p_disp_batch_cc = np.nan

print("\n  7b. Partial PERMANOVA — condition | dataset (residual matrix)...")
print("    Computing residual distance matrix for CC...")
dm_resid_cc     = residual_distance_matrix(dm_cc, meta_cc["dataset_name"].values)
perm_partial_cc = permanova(dm_resid_cc, meta_cc, column="study_condition", permutations=PERMUTATIONS)
p_partial_cc    = get_permanova_pvalue(perm_partial_cc)
omega2_partial_cc = omega_squared(dm_resid_cc, meta_cc["study_condition"].values)
print(
    f"    CC partial PERMANOVA — p={p_partial_cc:.6f}  {sig_stars(p_partial_cc)}  "
    f"ω²={omega2_partial_cc:.4f} ({_omega_label(omega2_partial_cc)})"
)

print("\n  7c. Variance partitioning...")


def vp(omega2_batch, omega2_cond_marginal, omega2_cond_partial):
    batch    = max(omega2_batch, 0)
    cond     = max(omega2_cond_partial, 0)
    shared   = max(omega2_cond_marginal - omega2_cond_partial, 0)
    residual = max(1 - batch - cond - shared, 0)
    return dict(
        batch=batch, condition=cond, shared=shared, residual=residual,
        total_explained=batch + cond + shared,
    )


vp_cc = vp(omega2_batch_cc, omega2_cc, omega2_partial_cc)

print(f"\n    CC variance partitioning:")
print(f"      Batch alone     : {vp_cc['batch']:.4f}  ({vp_cc['batch']*100:.1f}%)")
print(f"      Condition alone : {vp_cc['condition']:.4f}  ({vp_cc['condition']*100:.1f}%)")
print(f"      Shared          : {vp_cc['shared']:.4f}  ({vp_cc['shared']*100:.1f}%)")
print(f"      Residual        : {vp_cc['residual']:.4f}  ({vp_cc['residual']*100:.1f}%)")
print(f"      Total explained : {vp_cc['total_explained']:.4f}  ({vp_cc['total_explained']*100:.1f}%)")


# ============================================================
# 8. PRINT FULL RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 75)
print("  FULL STATISTICAL RESULTS  |  Train set (80% per dataset)  |  Genus level")
print("  ω² interpretation: <0.01 negligible | 0.01–0.06 small | 0.06–0.14 medium | >0.14 large")
print("=" * 75)

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
print(
    f"    Batch alone     {vp_cc['batch']*100:.1f}%  |  "
    f"Condition alone {vp_cc['condition']*100:.1f}%  |  "
    f"Shared {vp_cc['shared']*100:.1f}%  |  "
    f"Residual {vp_cc['residual']*100:.1f}%"
)


# ============================================================
# 9. SAVE CSV
# ============================================================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

results_df = pd.DataFrame([{
    "comparison":                       "control vs CRC",
    "n_control":                        (meta_cc["study_condition"] == "control").sum(),
    "n_comparison":                     (meta_cc["study_condition"] == "CRC").sum(),
    "PERMANOVA_p":                      round(p_cc, 6),
    "PERMANOVA_sig":                    sig_stars(p_cc),
    "PERMDISP_p":                       round(p_disp_cc, 6) if not pd.isna(p_disp_cc) else np.nan,
    "PERMDISP_sig":                     sig_stars(p_disp_cc),
    "omega2_condition":                 max(round(omega2_cc, 4), 0),
    "omega2_condition_interp":          _omega_label(omega2_cc),
    "EV_pct":                           round(ev_cc, 2),
    "PERMANOVA_batch_p":                round(p_batch_cc, 6),
    "PERMANOVA_batch_sig":              sig_stars(p_batch_cc),
    "PERMDISP_batch_p":                 round(p_disp_batch_cc, 6) if not pd.isna(p_disp_batch_cc) else np.nan,
    "PERMDISP_batch_sig":               sig_stars(p_disp_batch_cc),
    "omega2_batch":                     max(round(omega2_batch_cc, 4), 0),
    "omega2_batch_interp":              _omega_label(omega2_batch_cc),
    "PERMANOVA_partial_p":              round(p_partial_cc, 6),
    "PERMANOVA_partial_sig":            sig_stars(p_partial_cc),
    "omega2_condition_partial":         max(round(omega2_partial_cc, 4), 0),
    "omega2_condition_partial_interp":  _omega_label(omega2_partial_cc),
    "VP_batch_pct":                     round(vp_cc["batch"] * 100, 2),
    "VP_condition_pct":                 round(vp_cc["condition"] * 100, 2),
    "VP_shared_pct":                    round(vp_cc["shared"] * 100, 2),
    "VP_residual_pct":                  round(vp_cc["residual"] * 100, 2),
    "VP_total_explained_pct":           round(vp_cc["total_explained"] * 100, 2),
}])

results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n  Stats saved → {OUTPUT_CSV}")


# ============================================================
# 10. PLOT — 1 row, 2 panels
# ============================================================
print("\nRendering figure...")


def confidence_ellipse(ax, x, y, color, n_std=2.448,
                       fill_alpha=0.10, line_alpha=0.75, lw=1.5):
    if len(x) < 3:
        return
    cov  = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w  = 2 * n_std * np.sqrt(vals[0])
    h  = 2 * n_std * np.sqrt(vals[1])
    cx, cy = np.mean(x), np.mean(y)
    ax.add_patch(Ellipse(
        xy=(cx, cy), width=w, height=h, angle=theta,
        facecolor=color, edgecolor="none", alpha=fill_alpha, zorder=1,
    ))
    ax.add_patch(Ellipse(
        xy=(cx, cy), width=w, height=h, angle=theta,
        facecolor="none", edgecolor=color, alpha=line_alpha,
        lw=lw, linestyle="--", zorder=2,
    ))


def draw_pcoa_batch(ax, pcoa_result, meta_df, pc1, pc2, title):
    coords = pcoa_result.samples[["PC1", "PC2"]].copy()
    coords = coords.join(meta_df[["dataset_name"]])

    for ds in sorted(meta_df["dataset_name"].unique()):
        sub       = coords[coords["dataset_name"] == ds]
        label_str = DATASET_LABEL_MAP.get(ds, ds)
        ax.scatter(
            sub["PC1"], sub["PC2"],
            color=DATASET_COLOR[ds], label=label_str,
            alpha=0.55, s=25, edgecolors="white", linewidth=0.2,
        )
        confidence_ellipse(ax, sub["PC1"].values, sub["PC2"].values,
                            color=DATASET_COLOR[ds])

    ax.set_xlabel(f"PCo1 ({pc1:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PCo2 ({pc2:.1f}%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.legend(loc="lower right", fontsize=6, framealpha=0.85,
              edgecolor="#cccccc", markerscale=0.8,
              title="Dataset", title_fontsize=6)
    ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
    ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
    ax.set_facecolor("#fafafa")


def draw_pcoa_condition(ax, pcoa_result, meta_df, group_col, group_order,
                        pc1, pc2, ev, title, p_val, p_disp, omega2,
                        p_batch, p_disp_batch, o2_batch, p_part, o2_part):
    coords = pcoa_result.samples[["PC1", "PC2"]].copy()
    coords = coords.join(meta_df[[group_col]])

    for grp in group_order:
        sub = coords[coords[group_col] == grp]
        if sub.empty:
            continue
        ax.scatter(
            sub["PC1"], sub["PC2"],
            color=COLOR_DICT.get(grp, "#999999"), label=grp,
            alpha=0.65, s=30, edgecolors="white", linewidth=0.3,
        )
        confidence_ellipse(ax, sub["PC1"].values, sub["PC2"].values,
                            color=COLOR_DICT.get(grp, "#999999"))

    ax.set_xlabel(f"PCo1 ({pc1:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PCo2 ({pc2:.1f}%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    stats_text = (
        f"PERMANOVA: p={p_val:.4f} {sig_stars(p_val)}  |  PERMDISP: p={fmt(p_disp)} {sig_stars(p_disp)}\n"
        f"Condition ω² = {max(omega2, 0):.4f}\n"
        f"────────────────────────────────────────────────\n"
        f"Batch PERMANOVA: p={p_batch:.4f} {sig_stars(p_batch)}  |  PERMDISP: p={fmt(p_disp_batch)} {sig_stars(p_disp_batch)}\n"
        f"Batch ω² = {max(o2_batch, 0):.4f}\n"
        f"Partial ω² (cond|batch) = {max(o2_part, 0):.4f}\n"
        f"EV = {ev:.1f}%"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes, va="top", ha="left",
        fontsize=7.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    handles = [
        mpatches.Patch(color=COLOR_DICT.get(g, "#999999"), label=g)
        for g in group_order
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="#cccccc")
    ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
    ax.axvline(0, color="grey", lw=0.5, ls="--", alpha=0.4)
    ax.set_facecolor("#fafafa")


fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.subplots_adjust(top=0.88)
fig.suptitle(
    "Beta Diversity — Genus Level\n"
    "Bray-Curtis PCoA  ·  Batch vs. Condition  |  Control vs CRC",
    fontsize=13, fontweight="bold",
)

draw_pcoa_batch(
    ax=axes[0], pcoa_result=pcoa_cc, meta_df=meta_cc,
    pc1=pc1_cc, pc2=pc2_cc,
    title="Coloured by dataset (batch)",
)

draw_pcoa_condition(
    ax=axes[1], pcoa_result=pcoa_cc, meta_df=meta_cc,
    group_col="study_condition", group_order=["control", "CRC"],
    pc1=pc1_cc, pc2=pc2_cc, ev=ev_cc,
    title="Control vs CRC (condition)",
    p_val=p_cc, p_disp=p_disp_cc, omega2=omega2_cc,
    p_batch=p_batch_cc, p_disp_batch=p_disp_batch_cc,
    o2_batch=omega2_batch_cc, p_part=p_partial_cc, o2_part=omega2_partial_cc,
)

sns.despine(fig=fig)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=450, bbox_inches="tight")
plt.show()
print(f"Figure saved → {OUTPUT_FIG}")
print(f"Stats  saved → {OUTPUT_CSV}")