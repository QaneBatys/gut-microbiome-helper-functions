#!/usr/bin/env python3
"""
Alpha Diversity Analysis — Shannon Index
ALL DATASETS COMBINED  |  GENUS LEVEL  |  Control vs CRC only
--------------------------------------------------------------------
Species-level columns are collapsed to genus level via aggregate_taxa()
before TSS normalisation via relative_abundance() and Shannon H' calculation.

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

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
from pd_utils import (
    aggregate_taxa,
    relative_abundance,
    _all_meta,
    prepare_target_class,
    split_dataset,
)

warnings.filterwarnings("ignore")

# CONFIGURATION
FILE_PATH = "./data/filtered_nine_crc_final_clean.tsv"
BOOTSTRAP_ITERS = 5000
RANDOM_SEED = 42
OUTPUT_FIG = "alpha_diversity_combined_genus_shannon_train.png"
OUTPUT_STATS = "alpha_diversity_combined_genus_stats_train.csv"
OUTPUT_DESC = "alpha_diversity_combined_genus_descriptive_train.csv"

MEM_GUARD_ELEMENTS = 5_000_000

CONDITION_MAP = {
    "control": "control",
    "Control": "control",
    "CONTROL": "control",
    "adenoma": None,
    "Adenoma": None,
    "ADENOMA": None,
    "crc": "CRC",
    "Crc": "CRC",
    "CRC": "CRC",
    "IBD": None,
    "ibd": None,
}


# HELPER FUNCTIONS
def sig_stars(p):
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def cliffs_delta_label(d):
    if pd.isna(d):
        return "n/a"
    abs_d = abs(d)
    if abs_d < 0.147:
        return "negligible"
    if abs_d < 0.330:
        return "small"
    if abs_d < 0.474:
        return "medium"
    return "large"


def cliffs_delta(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return (more - less) / (len(x) * len(y))


def cliffs_delta_ci(x, y, n_boot=BOOTSTRAP_ITERS, seed=RANDOM_SEED):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    total_elements = n_boot * len(x) * len(y)
    if total_elements <= MEM_GUARD_ELEMENTS:
        xb = rng.choice(x, size=(n_boot, len(x)), replace=True)
        yb = rng.choice(y, size=(n_boot, len(y)), replace=True)
        more = np.sum(xb[:, :, None] > yb[:, None, :], axis=(1, 2))
        less = np.sum(xb[:, :, None] < yb[:, None, :], axis=(1, 2))
        boot_deltas = (more - less) / (len(x) * len(y))
    else:
        batch_size = max(1, int(MEM_GUARD_ELEMENTS / (len(x) * len(y))))
        boot_deltas = np.empty(n_boot, dtype=float)
        n_done = 0
        while n_done < n_boot:
            bs = min(batch_size, n_boot - n_done)
            xb = rng.choice(x, size=(bs, len(x)), replace=True)
            yb = rng.choice(y, size=(bs, len(y)), replace=True)
            m = np.sum(xb[:, :, None] > yb[:, None, :], axis=(1, 2))
            l = np.sum(xb[:, :, None] < yb[:, None, :], axis=(1, 2))
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
        return {
            k: np.nan
            for k in ["n", "median", "Q1", "Q3", "IQR", "whisker_min", "whisker_max"]
        }
    q1, median, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    whisker_min = v[v >= q1 - 1.5 * iqr].min()
    whisker_max = v[v <= q3 + 1.5 * iqr].max()
    return {
        "n": len(v),
        "median": round(median, 4),
        "Q1": round(q1, 4),
        "Q3": round(q3, 4),
        "IQR": round(iqr, 4),
        "whisker_min": round(whisker_min, 4),
        "whisker_max": round(whisker_max, 4),
    }


# 1. LOADING DATA
print("Loading data...")
df = pd.read_csv(FILE_PATH, sep="\t", index_col=0)

METADATA_COLS = [
    "dataset_name",
    "study_condition",
    "sampleID",
    "body_site",
    "disease",
    "age",
    "gender",
    "BMI",
    "country",
    "fobt",
    "sequencing_instrument",
]

metadata = df[["dataset_name", "study_condition", "sampleID"]].copy()
metadata["study_condition"] = metadata["study_condition"].str.strip().map(CONDITION_MAP)

unmapped = metadata["study_condition"].isna()
if unmapped.sum() > 0:
    raw_vals = df.loc[unmapped, "study_condition"].unique()
    print(f"  Excluding {unmapped.sum()} sample(s): {list(raw_vals)}")
    metadata = metadata[~unmapped].copy()
    df = df.loc[metadata.index].copy()

# Propagate mapped condition back so aggregate_taxa / relative_abundance see it
df["study_condition"] = metadata["study_condition"]


print("\nSplitting data (stratified 80/20 per dataset)...")
# Adding 'crc_label' column
df_labeled = prepare_target_class(df, "study_condition", "CRC", "crc_label")

splits = split_dataset(df_labeled, "crc_label", strategy="individual")

# Report per-dataset breakdown
print(f"\n  {'Dataset':<35} {'Total':>6} {'Train':>6} {'Test':>5}")
print(f"  {'-'*55}")
for ds, (X_tr, X_te, y_tr, y_te) in splits.items():
    total = len(X_tr) + len(X_te)
    print(f"  {ds:<35} {total:>6} {len(X_tr):>6} {len(X_te):>5}")

# Assemble pooled train set; crc_label was excluded by split_dataset
train_df = pd.concat(
    [X_tr for X_tr, _, _, _ in splits.values()],
    ignore_index=True,
)
n_train_ctrl = (train_df["study_condition"] == "control").sum()
n_train_crc = (train_df["study_condition"] == "CRC").sum()
print(
    f"\n  Pooled train: {len(train_df)} samples  "
    f"(control={n_train_ctrl}, CRC={n_train_crc})"
)


# 2. AGGREGATE TO GENUS LEVEL
print("Collapsing species to genus level...")

n_species_before = len([c for c in train_df.columns if c not in METADATA_COLS])

genus_df = aggregate_taxa(train_df, "genus")

n_genera = len([c for c in genus_df.columns if c not in set(_all_meta())])
print(f"  Species: {n_species_before}  ->  Genera: {n_genera}")

# Re-align metadata index after aggregate_taxa resets the index
metadata = genus_df[["dataset_name", "study_condition", "sampleID"]].copy()
feature_cols = [c for c in genus_df.columns if c not in set(_all_meta())]


# 3. ZERO-SUM GUARD
row_sums = genus_df[feature_cols].sum(axis=1)
zero_sum_mask = row_sums == 0
if zero_sum_mask.sum() > 0:
    print(f"  Excluding {zero_sum_mask.sum()} zero-sum sample(s).")
    genus_df = genus_df[~zero_sum_mask].reset_index(drop=True)
    metadata = genus_df[["dataset_name", "study_condition", "sampleID"]].copy()
    row_sums = genus_df[feature_cols].sum(axis=1)


# 4. TSS CHECK + RELATIVE ABUNDANCE
bad_sum_mask = ~np.isclose(row_sums, 100.0, atol=0.1)
if bad_sum_mask.sum() > 0:
    print(f"  Excluding {bad_sum_mask.sum()} out-of-range sample(s).")
    genus_df = genus_df[~bad_sum_mask].reset_index(drop=True)
    metadata = genus_df[["dataset_name", "study_condition", "sampleID"]].copy()
    row_sums = genus_df[feature_cols].sum(axis=1)
else:
    print(
        f"  TSS check passed: {len(row_sums)} samples "
        f"(range: {row_sums.min():.4f}-{row_sums.max():.4f})."
    )

genus_norm = relative_abundance(
    genus_df, feature_cols=feature_cols
)
genus_proportions = genus_norm[feature_cols]


# 5. SHANNON INDEX (BITS)
print("Calculating genus-level Shannon H' (bits, log2)...")
metadata = metadata.copy()
metadata["shannon"] = genus_proportions.apply(
    lambda x: shannon_entropy(x.values), axis=1
).values

nan_shannon = metadata["shannon"].isna()
if nan_shannon.sum() > 0:
    print(f"  Excluding {nan_shannon.sum()} undefined Shannon sample(s).")
    metadata = metadata[~nan_shannon].copy()

print(f"\n  Total samples after QC: {len(metadata)}")
for cond in ["control", "CRC"]:
    print(f"    {cond:<10}: {(metadata['study_condition']==cond).sum()} samples")


# 6. DESCRIPTIVE STATISTICS
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
    s = boxplot_stats(vals)
    desc_rows.append({"condition": cond, **s})
    n_str = str(int(s["n"])) if not pd.isna(s["n"]) else "n/a"
    med_str = f"{s['median']:>8.4f}" if not pd.isna(s["median"]) else f"{'n/a':>8}"
    q1_str = f"{s['Q1']:>8.4f}" if not pd.isna(s["Q1"]) else f"{'n/a':>8}"
    q3_str = f"{s['Q3']:>8.4f}" if not pd.isna(s["Q3"]) else f"{'n/a':>8}"
    iqr_str = f"{s['IQR']:>8.4f}" if not pd.isna(s["IQR"]) else f"{'n/a':>8}"
    wmin_str = (
        f"{s['whisker_min']:>8.4f}" if not pd.isna(s["whisker_min"]) else f"{'n/a':>8}"
    )
    wmax_str = (
        f"{s['whisker_max']:>8.4f}" if not pd.isna(s["whisker_max"]) else f"{'n/a':>8}"
    )
    print(
        f"  {cond:<12} {n_str:>5} {med_str} {q1_str} {q3_str} "
        f"{iqr_str} {wmin_str} {wmax_str}"
    )

pd.DataFrame(desc_rows).to_csv(OUTPUT_DESC, index=False)
print(f"\n  Descriptive stats saved -> {OUTPUT_DESC}")


# 7. STATISTICAL TESTING
print(
    f"\nRunning Mann-Whitney U + Cliff's delta "
    f"(bootstrap n={BOOTSTRAP_ITERS}, seed={RANDOM_SEED})..."
)

ctrl_vals = metadata[metadata["study_condition"] == "control"]["shannon"]
crc_vals = metadata[metadata["study_condition"] == "CRC"]["shannon"]

_, p_b = mannwhitneyu(ctrl_vals, crc_vals, alternative="two-sided")
d_b = cliffs_delta(ctrl_vals, crc_vals)
ci_lo_b, ci_hi_b = cliffs_delta_ci(ctrl_vals, crc_vals)

results = {
    "ctrl_vs_crc": {
        "comparison": "control vs CRC",
        "n_control": len(ctrl_vals),
        "n_comparison": len(crc_vals),
        "p_value": round(p_b, 6),
        "significance": sig_stars(p_b),
        "cliffs_delta": round(d_b, 4),
        "ci_lower": ci_lo_b,
        "ci_upper": ci_hi_b,
        "effect_label": cliffs_delta_label(d_b),
    },
}


# 8. PRINT RESULTS
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


# ── 9. PLOT — publication quality ─────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#2C2C2C",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

COLOR = {"control": "#C0392B", "CRC": "#2C3E50"}
EDGE_COLOR = {"control": "#922B21", "CRC": "#17202A"}
condition_order = ["control", "CRC"]
condition_labels = {
    "control": "Control\n(Healthy)",
    "CRC": "CRC\n(Cancer)",
}

fig, ax = plt.subplots(figsize=(4.5, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9FA")

# Violin plot
for i, cond in enumerate(condition_order):
    vals = metadata[metadata["study_condition"] == cond]["shannon"].dropna().values
    if len(vals) < 2:
        continue
    vparts = ax.violinplot(
        vals,
        positions=[i],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.7,
    )
    for body in vparts["bodies"]:
        body.set_facecolor(COLOR[cond])
        body.set_edgecolor(EDGE_COLOR[cond])
        body.set_linewidth(1.3)
        body.set_alpha(0.72)

# Median line
for i, cond in enumerate(condition_order):
    vals = metadata[metadata["study_condition"] == cond]["shannon"].dropna().values
    median = np.median(vals)
    ax.hlines(median, i - 0.09, i + 0.09, colors="black", linewidth=2.3, zorder=6)

# IQR bar
for i, cond in enumerate(condition_order):
    vals = metadata[metadata["study_condition"] == cond]["shannon"].dropna().values
    q1, q3 = np.percentile(vals, [25, 75])
    ax.vlines(i, q1, q3, colors="white", linewidth=5, zorder=4, alpha=0.65)
    ax.vlines(i, q1, q3, colors="#333333", linewidth=1.6, zorder=5, alpha=0.9)

# Jittered scatter
rng_plot = np.random.default_rng(42)
for i, cond in enumerate(condition_order):
    vals = metadata[metadata["study_condition"] == cond]["shannon"].dropna().values
    scatter_vals = (
        vals
        if len(vals) <= 600
        else vals[rng_plot.choice(len(vals), size=600, replace=False)]
    )
    jitter = rng_plot.uniform(-0.18, 0.18, size=len(scatter_vals))
    ax.scatter(
        i + jitter,
        scatter_vals,
        color=EDGE_COLOR[cond],
        edgecolors="none",
        alpha=0.22,
        s=6,
        zorder=3,
    )

# Horizontal grid
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#CCCCCC", alpha=0.9, zorder=0)
ax.set_axisbelow(True)

# y-axis range
all_vals = metadata["shannon"].dropna().values
y_pad = (all_vals.max() - all_vals.min()) * 0.05
ax.set_ylim(all_vals.min() - y_pad, all_vals.max() + y_pad)

# x-axis labels
ax.set_xticks([0, 1])
ax.set_xticklabels(["", ""])
ax.tick_params(axis="x", which="both", length=0, pad=4)
ax.tick_params(axis="y", labelsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

for i, cond in enumerate(condition_order):
    n = (metadata["study_condition"] == cond).sum()
    label = f"{condition_labels[cond]}\nn = {n:,}"
    ax.text(
        i,
        -0.04,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10.5,
        color="#1A1A1A",
        linespacing=1.5,
    )

# Axis labels & title
ax.set_ylabel("Shannon Index H' (bits)", fontsize=12, fontweight="bold", labelpad=10)
ax.set_xlabel("")
ax.set_title(
    "Alpha Diversity - Genus Level\n",
    fontsize=13,
    fontweight="bold",
    pad=14,
    color="#1A1A1A",
)

# Legend
legend_elements = [
    Patch(
        facecolor=COLOR[c],
        edgecolor=EDGE_COLOR[c],
        alpha=0.80,
        linewidth=1.2,
        label=lbl,
    )
    for c, lbl in [("control", "Control"), ("CRC", "CRC")]
]
leg = ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    ncol=2,
    frameon=True,
    fontsize=10,
    title="Study Groups",
    title_fontsize=10,
    framealpha=0.95,
    edgecolor="#CCCCCC",
    borderpad=0.8,
)
leg.get_frame().set_linewidth(0.8)

# Spine cleanup
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)
ax.spines["left"].set_color("#2C2C2C")
ax.spines["bottom"].set_color("#2C2C2C")

plt.tight_layout(pad=1.8)
plt.subplots_adjust(bottom=0.22)
plt.savefig(
    OUTPUT_FIG, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
)
plt.show()
print(f"\nFigure saved -> {OUTPUT_FIG}")
