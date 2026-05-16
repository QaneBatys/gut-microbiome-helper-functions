#!/usr/bin/env python3
"""
Alpha Diversity Analysis — Shannon Index
PER DATASET  |  GENUS LEVEL  |  Control vs CRC only  |  SEEN DATASET
-------------------------------------------------------------------------------------
Loads pre-split genus-level data from ./data/prepared/genus/.
No aggregation, no splitting — all preprocessing already done.

One statistical comparison per dataset:
  control vs CRC  → Mann-Whitney U + Cliff's delta

All p-values BH-corrected across datasets.
Cliff's delta 95% CI via bootstrapping (n=5000 iterations).
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
from statsmodels.stats.multitest import multipletests
from pd_utils import relative_abundance

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
PREPARED_DIR    = "./data/prepared/genus/"
MIN_SAMPLES     = 10
BOOTSTRAP_ITERS = 5000
RANDOM_SEED     = 42
OUTPUT_FIG      = "alpha_diversity_genus_per_dataset_train.png"
OUTPUT_STATS    = "alpha_diversity_genus_per_dataset_stats_train.csv"
OUTPUT_DESC     = "alpha_diversity_genus_per_dataset_descriptive_train.csv"

MEM_GUARD_ELEMENTS = 5_000_000

DATASET_LABEL_MAP = {
    "FengQ_2015":        "Feng et al. (2015)",
    "HanniganGD_2017":   "Hannigan et al. (2018)",
    "ThomasAM_2019_a":   "Thomas et al. (2019)(a)",
    "ThomasAM_2019_b":   "Thomas et al. (2019)(b)",
    "ThomasAM_2019_c":   "Thomas et al. (2019)(c)",
    "VogtmannE_2016":    "Vogtmann et al. (2016)",
    "WirbelJ_2018":      "Wirbel et al. (2019)",
    "YuJ_2015":          "Yu et al. (2017)",
    "ZellerG_2014":      "Zeller et al. (2014)",
}

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
    if pd.isna(d):    return "n/a"
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


def bh_correct(df, col):
    padj_col = "padj_" + col
    testable = df[col].notna()
    if testable.sum() > 0:
        _, p_adj, _, _ = multipletests(df.loc[testable, col], method="fdr_bh")
        df.loc[testable, padj_col] = p_adj
    else:
        df[padj_col] = np.nan


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


def fmt_p(p):
    if pd.isna(p):  return "n/a"
    if p < 0.0001:  return "<0.0001"
    return f"{p:.4f}"


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
    keep      = ~zero_sum_mask
    genus_df  = genus_df[keep].reset_index(drop=True)
    metadata  = metadata[keep].reset_index(drop=True)
    row_sums  = genus_df[feature_cols].sum(axis=1)

# ============================================================
# 3. ROW SUM INFO — no exclusion, data is verified correct
# ============================================================
print(
    f"  Row sum range: {row_sums.min():.4f} – {row_sums.max():.4f}  "
    f"({len(row_sums)} samples retained)"
)

genus_norm        = relative_abundance(genus_df, feature_cols=feature_cols)
genus_proportions = genus_norm[feature_cols]


# ============================================================
# 4. SHANNON INDEX (BITS)
# ============================================================
print("Calculating genus-level Shannon H' (bits, log2)...")
metadata = metadata.copy()
metadata["shannon"] = genus_proportions.apply(
    lambda x: shannon_entropy(x.values), axis=1
).values

nan_shannon = metadata["shannon"].isna()
if nan_shannon.sum() > 0:
    print(f"  Excluding {nan_shannon.sum()} undefined Shannon sample(s).")
    metadata = metadata[~nan_shannon].reset_index(drop=True)

print(f"\n  Total train samples after QC: {len(metadata)}")
for cond in ["control", "CRC"]:
    print(f"    {cond:<10}: {(metadata['study_condition'] == cond).sum()} samples")


# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 75)
print("  DESCRIPTIVE STATISTICS — GENUS LEVEL  |  Train set (80% per dataset)")
print("=" * 75)

desc_rows = []
for ds in sorted(metadata["dataset_name"].unique()):
    for cond in ["control", "CRC"]:
        vals = metadata[
            (metadata["dataset_name"] == ds) &
            (metadata["study_condition"] == cond)
        ]["shannon"]
        s = boxplot_stats(vals)
        desc_rows.append({"level": "per_dataset", "dataset": ds, "condition": cond, **s})

print(
    f"\n  {'Condition':<12} {'n':>5} {'Median':>8} {'Q1':>8} {'Q3':>8} "
    f"{'IQR':>8} {'Wsk_min':>8} {'Wsk_max':>8}"
)
print(f"  {'-'*67}")
for cond in ["control", "CRC"]:
    vals = metadata[metadata["study_condition"] == cond]["shannon"]
    s    = boxplot_stats(vals)
    desc_rows.append({"level": "all_datasets", "dataset": "ALL", "condition": cond, **s})
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
# 6. STATISTICAL TESTING
# ============================================================
print(
    f"\nRunning Mann-Whitney U + Cliff's delta "
    f"(bootstrap n={BOOTSTRAP_ITERS}, seed={RANDOM_SEED})..."
)

all_results = []

for ds in sorted(metadata["dataset_name"].unique()):
    subset    = metadata[metadata["dataset_name"] == ds]
    row       = {"dataset": ds}
    ctrl_vals = subset[subset["study_condition"] == "control"]["shannon"]
    crc_vals  = subset[subset["study_condition"] == "CRC"]["shannon"]

    row["n_ctrl"] = len(ctrl_vals)
    row["n_crc"]  = len(crc_vals)

    if len(ctrl_vals) >= MIN_SAMPLES and len(crc_vals) >= MIN_SAMPLES:
        _, p         = mannwhitneyu(ctrl_vals, crc_vals, alternative="two-sided")
        d            = cliffs_delta(ctrl_vals, crc_vals)
        ci_lo, ci_hi = cliffs_delta_ci(ctrl_vals, crc_vals)
        row.update({
            "p_ctrl_vs_crc":           p,
            "delta_ctrl_vs_crc":       round(d, 4),
            "delta_ctrl_vs_crc_ci_lo": ci_lo,
            "delta_ctrl_vs_crc_ci_hi": ci_hi,
            "delta_ctrl_vs_crc_label": cliffs_delta_label(d),
        })
    else:
        row.update({
            "p_ctrl_vs_crc":           np.nan,
            "delta_ctrl_vs_crc":       np.nan,
            "delta_ctrl_vs_crc_ci_lo": np.nan,
            "delta_ctrl_vs_crc_ci_hi": np.nan,
            "delta_ctrl_vs_crc_label": "n/a (insufficient n)",
        })

    all_results.append(row)

stats_df = pd.DataFrame(all_results)
bh_correct(stats_df, "p_ctrl_vs_crc")


# ============================================================
# 7. PRINT STATISTICAL RESULTS
# ============================================================
print("\n" + "=" * 100)
print(
    "  MANN-WHITNEY U  |  BH-corrected FDR  |  CLIFF'S DELTA (95% Bootstrap CI) — GENUS LEVEL"
)
print(
    "  Train set (80% per dataset)  |  Sign: d > 0 = control HIGHER | d < 0 = control LOWER"
)
print("=" * 100)
print(
    f"\n  {'Dataset':<25} {'p_raw':>8} {'p_adj':>8} {'sig':>4} "
    f"{'delta':>7} {'95% CI':>18} {'effect':>10}"
)
print(f"  {'-'*90}")

for _, row in stats_df.iterrows():
    p_r   = row.get("p_ctrl_vs_crc",           np.nan)
    p_a   = row.get("padj_p_ctrl_vs_crc",      np.nan)
    d     = row.get("delta_ctrl_vs_crc",        np.nan)
    ci_lo = row.get("delta_ctrl_vs_crc_ci_lo",  np.nan)
    ci_hi = row.get("delta_ctrl_vs_crc_ci_hi",  np.nan)
    lbl   = row.get("delta_ctrl_vs_crc_label",  "n/a")
    p_str = (
        f"{p_r:>8.4f} {p_a:>8.4f} {sig_stars(p_a):>4}"
        if not pd.isna(p_r) else f"{'skipped':>21}"
    )
    d_str = (
        f"{d:>+7.3f} [{ci_lo:>+.3f}, {ci_hi:>+.3f}] {lbl:>10}"
        if not pd.isna(d) else f"{'n/a (insufficient n)':>36}"
    )
    print(f"  {row['dataset']:<25} {p_str}   {d_str}")

stats_df.to_csv(OUTPUT_STATS, index=False)
print(f"\n  Stats saved -> {OUTPUT_STATS}")


# ============================================================
# 8. PLOT — publication quality
# ============================================================
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,
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
condition_labels = {"control": "Ctrl", "CRC": "CRC"}

datasets = sorted(metadata["dataset_name"].unique())
ncols = 3
nrows = int(np.ceil(len(datasets) / ncols))

fig, axes = plt.subplots(
    nrows, ncols, figsize=(5.0 * ncols, 6.5 * nrows),
    constrained_layout=False,
)
fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.93, bottom=0.10)
axes = axes.flatten()

rng_plot = np.random.default_rng(42)

for i, ds in enumerate(datasets):
    ax  = axes[i]
    sub = metadata[metadata["dataset_name"] == ds]

    ax.set_facecolor("#F8F9FA")
    present = [c for c in condition_order if c in sub["study_condition"].values]

    for j, cond in enumerate(present):
        vals = sub[sub["study_condition"] == cond]["shannon"].dropna().values
        if len(vals) < 2:
            continue
        vparts = ax.violinplot(
            vals, positions=[j],
            showmeans=False, showmedians=False, showextrema=False, widths=0.7,
        )
        for body in vparts["bodies"]:
            body.set_facecolor(COLOR[cond])
            body.set_edgecolor(EDGE_COLOR[cond])
            body.set_linewidth(1.3)
            body.set_alpha(0.72)

    for j, cond in enumerate(present):
        vals   = sub[sub["study_condition"] == cond]["shannon"].dropna().values
        median = np.median(vals)
        ax.hlines(median, j - 0.09, j + 0.09, colors="black", linewidth=2.1, zorder=6)

    for j, cond in enumerate(present):
        vals   = sub[sub["study_condition"] == cond]["shannon"].dropna().values
        q1, q3 = np.percentile(vals, [25, 75])
        ax.vlines(j, q1, q3, colors="white",   linewidth=4.5, zorder=4, alpha=0.65)
        ax.vlines(j, q1, q3, colors="#333333", linewidth=1.5, zorder=5, alpha=0.9)

    for j, cond in enumerate(present):
        vals = sub[sub["study_condition"] == cond]["shannon"].dropna().values
        scatter_vals = (
            vals if len(vals) <= 300
            else vals[rng_plot.choice(len(vals), 300, replace=False)]
        )
        jitter = rng_plot.uniform(-0.16, 0.16, size=len(scatter_vals))
        ax.scatter(
            j + jitter, scatter_vals,
            color=EDGE_COLOR[cond], edgecolors="none",
            alpha=0.22, s=6, zorder=3,
        )

    ax.yaxis.grid(True, linestyle="--", linewidth=0.55,
                  color="#CCCCCC", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)

    panel_vals = sub["shannon"].dropna().values
    if len(panel_vals) > 0:
        y_pad = (panel_vals.max() - panel_vals.min()) * 0.08
        ax.set_ylim(panel_vals.min() - y_pad, panel_vals.max() + y_pad)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([""] * len(present))
    ax.tick_params(axis="x", which="both", length=0, pad=4)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    for j, cond in enumerate(present):
        n = (sub["study_condition"] == cond).sum()
        ax.text(
            j, -0.06,
            f"{condition_labels[cond]}\nn = {n:,}",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top",
            fontsize=9, color="#1A1A1A", linespacing=1.45,
        )

    ax.set_title(DATASET_LABEL_MAP.get(ds, ds), weight="bold", fontsize=10, pad=14)
    ax.set_ylabel("Shannon H' (bits)", fontsize=9, labelpad=6)
    ax.set_xlabel("")

    stat_row = stats_df[stats_df["dataset"] == ds]
    if not stat_row.empty:
        r    = stat_row.iloc[0]
        p_cc = r.get("padj_p_ctrl_vs_crc", np.nan)
        d_cc = r.get("delta_ctrl_vs_crc",  np.nan)
        cc_str = (
            f"FDR={fmt_p(p_cc)} {sig_stars(p_cc)}, d={d_cc:+.3f}"
            if not pd.isna(p_cc) else "skipped"
        )
        ax.text(
            0.03, 0.03,
            f"Ctrl vs CRC:\n{cc_str}",
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=7, family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white", edgecolor="#CCCCCC", alpha=0.90,
            ),
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#2C2C2C")
    ax.spines["bottom"].set_color("#2C2C2C")

for j in range(len(datasets), len(axes)):
    axes[j].axis("off")

legend_elements = [
    Patch(
        facecolor=COLOR[c], edgecolor=EDGE_COLOR[c],
        alpha=0.80, linewidth=1.2, label=lbl,
    )
    for c, lbl in [("control", "Control (Healthy)"), ("CRC", "CRC (Cancer)")]
]

fig.legend(
    handles=legend_elements,
    loc="lower center", ncol=2,
    frameon=True, fontsize=10,
    title="Study Groups", title_fontsize=10,
    framealpha=0.92, edgecolor="#CCCCCC",
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "Alpha Diversity \u2014 Genus Level (Shannon H\u2019) Per Dataset",
    fontsize=14, fontweight="bold", y=0.97,
)

plt.savefig(
    OUTPUT_FIG, dpi=300, bbox_inches="tight",
    facecolor="white", edgecolor="none",
)
plt.show()
print(f"\nFigure saved -> {OUTPUT_FIG}")