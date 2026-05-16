#!/usr/bin/env python3
"""
Alpha Diversity Analysis — Shannon Index
ALL DATASETS COMBINED  |  GENUS LEVEL  |  Control vs CRC only
--------------------------------------------------------------------
Loads pre-split genus-level data from ./data/prepared/genus/.
No aggregation, no splitting — all preprocessing already done.

One statistical comparison across the pooled cohort:
  control vs CRC  → Mann-Whitney U + Cliff's delta

No BH correction needed (single test).
Cliff's delta 95% CI via bootstrapping (n=5000 iterations).

Cliff's delta sign convention:
  δ > 0 → control HIGHER Shannon diversity than CRC
  δ < 0 → control LOWER Shannon diversity than CRC

Shannon H' computed in BITS (log2), consistent with QIIME2 / phyloseq.
------------------------------------------------------------------------
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
from pd_utils import relative_abundance

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
PREPARED_DIR    = "./data/prepared/genus/"
BOOTSTRAP_ITERS = 5000
RANDOM_SEED     = 42
OUTPUT_FIG      = "alpha_diversity_combined_genus_shannon_train.png"
OUTPUT_STATS    = "alpha_diversity_combined_genus_stats_train.csv"
OUTPUT_DESC     = "alpha_diversity_combined_genus_descriptive_train.csv"

MEM_GUARD_ELEMENTS = 5_000_000

METADATA_COLS = [
    "dataset_name", "sampleID", "subjectID",
    "study_condition", "disease",
    "age", "gender", "country", "BMI", "fobt",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def sig_stars(p):
    if pd.isna(p):   return "n/a"
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return "ns"


def cliffs_delta_label(d):
    if pd.isna(d):   return "n/a"
    abs_d = abs(d)
    if abs_d < 0.147: return "negligible"
    if abs_d < 0.330: return "small"
    if abs_d < 0.474: return "medium"
    return "large"


def cliffs_delta(x, y):
    x = np.array(x, dtype=float); x = x[~np.isnan(x)]
    y = np.array(y, dtype=float); y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return (more - less) / (len(x) * len(y))


def cliffs_delta_ci(x, y, n_boot=BOOTSTRAP_ITERS, seed=RANDOM_SEED):
    x = np.array(x, dtype=float); x = x[~np.isnan(x)]
    y = np.array(y, dtype=float); y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    total_elements = n_boot * len(x) * len(y)
    if total_elements <= MEM_GUARD_ELEMENTS:
        xb   = rng.choice(x, size=(n_boot, len(x)), replace=True)
        yb   = rng.choice(y, size=(n_boot, len(y)), replace=True)
        more = np.sum(xb[:, :, None] > yb[:, None, :], axis=(1, 2))
        less = np.sum(xb[:, :, None] < yb[:, None, :], axis=(1, 2))
        boot_deltas = (more - less) / (len(x) * len(y))
    else:
        batch_size  = max(1, int(MEM_GUARD_ELEMENTS / (len(x) * len(y))))
        boot_deltas = np.empty(n_boot, dtype=float)
        n_done = 0
        while n_done < n_boot:
            bs = min(batch_size, n_boot - n_done)
            xb = rng.choice(x, size=(bs, len(x)), replace=True)
            yb = rng.choice(y, size=(bs, len(y)), replace=True)
            m  = np.sum(xb[:, :, None] > yb[:, None, :], axis=(1, 2))
            l  = np.sum(xb[:, :, None] < yb[:, None, :], axis=(1, 2))
            boot_deltas[n_done : n_done + bs] = (m - l) / (len(x) * len(y))
            n_done += bs
    ci_lower, ci_upper = np.percentile(boot_deltas, [2.5, 97.5])
    return round(float(ci_lower), 4), round(float(ci_upper), 4)


def shannon_entropy(proportions):
    p = np.array(proportions, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    return -np.sum(p * np.log2(p))


def boxplot_stats(values):
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {k: np.nan for k in
                ["n", "median", "Q1", "Q3", "IQR", "whisker_min", "whisker_max"]}
    q1, median, q3 = np.percentile(v, [25, 50, 75])
    iqr         = q3 - q1
    whisker_min = v[v >= q1 - 1.5 * iqr].min()
    whisker_max = v[v <= q3 + 1.5 * iqr].max()
    return {
        "n":           len(v),
        "median":      round(median,      4),
        "Q1":          round(q1,          4),
        "Q3":          round(q3,          4),
        "IQR":         round(iqr,         4),
        "whisker_min": round(whisker_min, 4),
        "whisker_max": round(whisker_max, 4),
    }


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

# ── build metadata and species matrices ──────────────────────
meta_cols_present = [c for c in METADATA_COLS if c in X_train_df.columns]
available_taxa    = [c for c in surviving_taxa if c in X_train_df.columns]

if "study_condition" in X_train_df.columns:
    metadata = X_train_df[meta_cols_present].copy().reset_index(drop=True)
else:
    metadata = pd.DataFrame(index=X_train_df.index)
    if "dataset_name" in X_train_df.columns:
        metadata["dataset_name"] = X_train_df["dataset_name"].values
    metadata["study_condition"] = y_train.map({1: "CRC", 0: "control"}).values

metadata   = metadata.reset_index(drop=True)
species_df = (
    X_train_df[available_taxa]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .reset_index(drop=True)
)

print(
    f"\n  Pooled train : {len(metadata)} samples  "
    f"(control={(metadata['study_condition'] == 'control').sum()}, "
    f"CRC={(metadata['study_condition'] == 'CRC').sum()})"
)


# ============================================================
# 2. ZERO-SUM GUARD
# ============================================================
row_sums      = species_df.sum(axis=1)
zero_sum_mask = row_sums == 0
if zero_sum_mask.sum() > 0:
    print(f"  Excluding {zero_sum_mask.sum()} zero-sum sample(s).")
    keep       = ~zero_sum_mask
    metadata   = metadata[keep].reset_index(drop=True)
    species_df = species_df[keep].reset_index(drop=True)
    row_sums   = row_sums[keep].reset_index(drop=True)

print(
    f"  Row sum range: {row_sums.min():.4f} – {row_sums.max():.4f}  "
    f"({len(row_sums)} samples retained)"
)


# ============================================================
# 3. RELATIVE ABUNDANCE + SHANNON
# ============================================================
genus_norm        = relative_abundance(species_df)
genus_proportions = genus_norm[available_taxa]

print("Calculating genus-level Shannon H' (bits, log2)...")
metadata = metadata.copy()
metadata["shannon"] = genus_proportions.apply(
    lambda x: shannon_entropy(x.values), axis=1
).values

nan_shannon = metadata["shannon"].isna()
if nan_shannon.sum() > 0:
    print(f"  Excluding {nan_shannon.sum()} undefined Shannon sample(s).")
    metadata = metadata[~nan_shannon].reset_index(drop=True)

print(f"\n  Total samples after QC: {len(metadata)}")
for cond in ["control", "CRC"]:
    print(f"    {cond:<10}: {(metadata['study_condition'] == cond).sum()} samples")


# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 75)
print("  DESCRIPTIVE STATISTICS  |  Genus level  |  Train set (80% per dataset)")
print("=" * 75)

desc_rows = []
print(
    f"\n  {'Condition':<12} {'n':>5} {'Median':>8} {'Q1':>8} {'Q3':>8} "
    f"{'IQR':>8} {'Wsk_min':>8} {'Wsk_max':>8}"
)
print(f"  {'-'*67}")
for cond in ["control", "CRC"]:
    vals = metadata[metadata["study_condition"] == cond]["shannon"]
    s    = boxplot_stats(vals)
    desc_rows.append({"condition": cond, **s})
    n_str    = str(int(s["n"])) if not pd.isna(s["n"]) else "n/a"
    med_str  = f"{s['median']:>8.4f}"      if not pd.isna(s["median"])      else f"{'n/a':>8}"
    q1_str   = f"{s['Q1']:>8.4f}"          if not pd.isna(s["Q1"])          else f"{'n/a':>8}"
    q3_str   = f"{s['Q3']:>8.4f}"          if not pd.isna(s["Q3"])          else f"{'n/a':>8}"
    iqr_str  = f"{s['IQR']:>8.4f}"         if not pd.isna(s["IQR"])         else f"{'n/a':>8}"
    wmin_str = f"{s['whisker_min']:>8.4f}"  if not pd.isna(s["whisker_min"]) else f"{'n/a':>8}"
    wmax_str = f"{s['whisker_max']:>8.4f}"  if not pd.isna(s["whisker_max"]) else f"{'n/a':>8}"
    print(
        f"  {cond:<12} {n_str:>5} {med_str} {q1_str} {q3_str} "
        f"{iqr_str} {wmin_str} {wmax_str}"
    )

pd.DataFrame(desc_rows).to_csv(OUTPUT_DESC, index=False)
print(f"\n  Descriptive stats saved -> {OUTPUT_DESC}")


# ============================================================
# 5. STATISTICAL TESTING
# ============================================================
print(
    f"\nRunning Mann-Whitney U + Cliff's delta "
    f"(bootstrap n={BOOTSTRAP_ITERS}, seed={RANDOM_SEED})..."
)

ctrl_vals = metadata[metadata["study_condition"] == "control"]["shannon"]
crc_vals  = metadata[metadata["study_condition"] == "CRC"]["shannon"]

_, p_b           = mannwhitneyu(ctrl_vals, crc_vals, alternative="two-sided")
d_b              = cliffs_delta(ctrl_vals, crc_vals)
ci_lo_b, ci_hi_b = cliffs_delta_ci(ctrl_vals, crc_vals)

results = {
    "ctrl_vs_crc": {
        "comparison":   "control vs CRC",
        "n_control":    len(ctrl_vals),
        "n_comparison": len(crc_vals),
        "p_value":      round(p_b, 6),
        "significance": sig_stars(p_b),
        "cliffs_delta": round(d_b, 4),
        "ci_lower":     ci_lo_b,
        "ci_upper":     ci_hi_b,
        "effect_label": cliffs_delta_label(d_b),
    },
}

print("\n" + "=" * 85)
print("  STATISTICAL RESULTS  |  Genus level  |  Train set (80% per dataset)")
print("  Sign: d > 0 = control HIGHER diversity | d < 0 = control LOWER")
print("=" * 85)
for r in results.values():
    print(f"\n  {r['comparison']}")
    print(f"    n_control={r['n_control']}, n_comparison={r['n_comparison']}")
    print(f"    Mann-Whitney p = {r['p_value']:.6f}  {r['significance']}")
    print(
        f"    Cliff's d     = {r['cliffs_delta']:+.4f}  "
        f"95% CI [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]  "
        f"-> {r['effect_label']}"
    )

pd.DataFrame(results).T.to_csv(OUTPUT_STATS, index=False)
print(f"\n  Stats saved -> {OUTPUT_STATS}")


# ============================================================
# 6. PLOT — publication quality
# ============================================================
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.linewidth":     1.2,
    "axes.edgecolor":     "#2C2C2C",
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    "xtick.major.size":   5,
    "ytick.major.size":   5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "pdf.fonttype":       42,
    "svg.fonttype":       "none",
})

COLOR      = {"control": "#C0392B", "CRC": "#2C3E50"}
EDGE_COLOR = {"control": "#922B21", "CRC": "#17202A"}
condition_order  = ["control", "CRC"]
condition_labels = {"control": "Control\n(Healthy)", "CRC": "CRC\n(Cancer)"}

fig, ax = plt.subplots(figsize=(4.5, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9FA")

for i, cond in enumerate(condition_order):
    vals = metadata[metadata["study_condition"] == cond]["shannon"].dropna().values
    if len(vals) < 2:
        continue
    vparts = ax.violinplot(
        vals, positions=[i],
        showmeans=False, showmedians=False, showextrema=False, widths=0.7,
    )
    for body in vparts["bodies"]:
        body.set_facecolor(COLOR[cond])
        body.set_edgecolor(EDGE_COLOR[cond])
        body.set_linewidth(1.3)
        body.set_alpha(0.72)

for i, cond in enumerate(condition_order):
    vals   = metadata[metadata["study_condition"] == cond]["shannon"].dropna().values
    median = np.median(vals)
    ax.hlines(median, i - 0.09, i + 0.09, colors="black", linewidth=2.3, zorder=6)

for i, cond in enumerate(condition_order):
    vals   = metadata[metadata["study_condition"] == cond]["shannon"].dropna().values
    q1, q3 = np.percentile(vals, [25, 75])
    ax.vlines(i, q1, q3, colors="white",   linewidth=5,   zorder=4, alpha=0.65)
    ax.vlines(i, q1, q3, colors="#333333", linewidth=1.6, zorder=5, alpha=0.9)

rng_plot = np.random.default_rng(42)
for i, cond in enumerate(condition_order):
    vals = metadata[metadata["study_condition"] == cond]["shannon"].dropna().values
    scatter_vals = (
        vals if len(vals) <= 600
        else vals[rng_plot.choice(len(vals), size=600, replace=False)]
    )
    jitter = rng_plot.uniform(-0.18, 0.18, size=len(scatter_vals))
    ax.scatter(
        i + jitter, scatter_vals,
        color=EDGE_COLOR[cond], edgecolors="none",
        alpha=0.22, s=6, zorder=3,
    )

ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#CCCCCC", alpha=0.9, zorder=0)
ax.set_axisbelow(True)

all_vals = metadata["shannon"].dropna().values
y_pad    = (all_vals.max() - all_vals.min()) * 0.05
ax.set_ylim(all_vals.min() - y_pad, all_vals.max() + y_pad)

ax.set_xticks([0, 1])
ax.set_xticklabels(["", ""])
ax.tick_params(axis="x", which="both", length=0, pad=4)
ax.tick_params(axis="y", labelsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

for i, cond in enumerate(condition_order):
    n     = (metadata["study_condition"] == cond).sum()
    label = f"{condition_labels[cond]}\nn = {n:,}"
    ax.text(
        i, -0.04, label,
        transform=ax.get_xaxis_transform(),
        ha="center", va="top",
        fontsize=10.5, color="#1A1A1A", linespacing=1.5,
    )

ax.set_ylabel("Shannon Index H' (bits)", fontsize=12, fontweight="bold", labelpad=10)
ax.set_xlabel("")
ax.set_title(
    "Alpha Diversity - Genus Level\n",
    fontsize=13, fontweight="bold", pad=14, color="#1A1A1A",
)

legend_elements = [
    Patch(
        facecolor=COLOR[c], edgecolor=EDGE_COLOR[c],
        alpha=0.80, linewidth=1.2, label=lbl,
    )
    for c, lbl in [("control", "Control"), ("CRC", "CRC")]
]
leg = ax.legend(
    handles=legend_elements,
    loc="upper center", bbox_to_anchor=(0.5, -0.2),
    ncol=2, frameon=True, fontsize=10,
    title="Study Groups", title_fontsize=10,
    framealpha=0.95, edgecolor="#CCCCCC", borderpad=0.8,
)
leg.get_frame().set_linewidth(0.8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)
ax.spines["left"].set_color("#2C2C2C")
ax.spines["bottom"].set_color("#2C2C2C")

plt.tight_layout(pad=1.8)
plt.subplots_adjust(bottom=0.22)
plt.savefig(
    OUTPUT_FIG, dpi=300, bbox_inches="tight",
    facecolor="white", edgecolor="none",
)
plt.show()
print(f"\nFigure saved -> {OUTPUT_FIG}")