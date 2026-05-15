"""
da_pooled.py
============
Analysis B — Pooled blocked-permutation DA.
Analysis C — Per-dataset validation grid.

Run independently:
    python da_pooled.py

Outputs:
    results/da_pooled_<level>.csv              pooled DA results
    results/da_intersection_<level>.csv        pooled AND per-cohort significant taxa
    results/da_validation_grid_<level>.csv     per-dataset validation grid
    results/da_volcano_<level>.png
    results/da_validation_grid_<level>.png
    results/da_data_quality_report_<level>.csv

Note: the intersection table requires da_agreement_<level>.csv produced by
da_percohort.py.  If that file is missing the intersection step is skipped.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from da_shared import (
    # config
    TAXON_LEVEL, DATASET_LABEL_MAP,
    Q_THRESHOLD_POOLED, Q_THRESHOLD_PERCOHORT,
    MIN_GROUP_SIZE, MIN_EFFECT_CLR, GRID_P_THRESHOLD,
    N_PERMUTATIONS, RANDOM_SEED,
    # output paths
    OUT_POOLED, OUT_INTERSECTION, OUT_GRID,
    OUT_HM_POOLED, OUT_VOLCANO, OUT_AGREEMENT,
    # data + helpers
    load_data, get_prevalent_taxa,
    clr_features, renormalize_features,
    compute_auroc, generalized_fold_change_quantiles,
    blocked_wilcoxon_permutation_p, _short,
)


# ===========================================================
# ANALYSIS B — POOLED DA
# ===========================================================

def run_pooled_da(df_feat, df_meta):
    """
    Blocked permutation Wilcoxon for significance.
    OLS with dataset dummies for effect size and direction.
    """
    kept_taxa = get_prevalent_taxa(df_feat, df_meta)
    if not kept_taxa:
        return pd.DataFrame()

    df_feat = df_feat.reset_index(drop=True)
    df_meta = df_meta.reset_index(drop=True)

    df_sub = df_feat[kept_taxa].copy()
    df_sub = renormalize_features(df_sub)

    invalid = ~df_sub.notna().all(axis=1)
    if invalid.sum() > 0:
        print(f"  Dropping {int(invalid.sum())} samples with invalid relative abundance.")
        df_sub  = df_sub[~invalid].reset_index(drop=True)
        df_meta = df_meta[~invalid].reset_index(drop=True)

    clr      = clr_features(df_sub)
    nan_rows = clr.isna().any(axis=1)
    if nan_rows.sum() > 0:
        print(f"  Dropping {int(nan_rows.sum())} samples with NaN CLR.")
        keep    = ~nan_rows
        clr     = clr[keep].reset_index(drop=True)
        df_sub  = df_sub[keep].reset_index(drop=True)
        df_meta = df_meta[keep].reset_index(drop=True)

    print(f"  CLR matrix: {clr.shape[0]} x {clr.shape[1]} "
          f"(NaN: {int(clr.isna().sum().sum())})")

    labels        = df_meta["condition"].values
    blocks        = df_meta["dataset_name"].values
    condition_arr = (labels == "CRC").astype(float)
    n_ctrl        = int((labels == "control").sum())
    n_crc         = int((labels == "CRC").sum())

    print(f"\nPooled blocked permutation DA: n_ctrl={n_ctrl}, n_CRC={n_crc}")
    print(f"Testing {len(kept_taxa)} taxa | permutations={N_PERMUTATIONS}")

    ds_dummies = pd.get_dummies(df_meta["dataset_name"], drop_first=True).astype(float)
    X_ols      = sm.add_constant(pd.concat(
        [pd.Series(condition_arr, name="condition_CRC"), ds_dummies], axis=1
    ))

    records = []
    for i, taxon in enumerate(kept_taxa, 1):
        if taxon not in clr.columns:
            continue
        x = clr[taxon].values
        if np.isnan(x).any():
            continue

        try:
            model = sm.OLS(x, X_ols).fit()
            coef  = float(model.params["condition_CRC"])
            se    = float(model.bse["condition_CRC"])
            t_st  = float(model.tvalues["condition_CRC"])
            p_ols = float(model.pvalues["condition_CRC"])
        except Exception:
            coef = se = t_st = p_ols = np.nan

        p_blocked = blocked_wilcoxon_permutation_p(
            x, labels, blocks=blocks,
            group_a="control", group_b="CRC",
            n_perm=N_PERMUTATIONS, random_state=RANDOM_SEED,
        )
        auroc = compute_auroc(x, labels, case_label="CRC")
        gfc   = generalized_fold_change_quantiles(df_sub[taxon].values, labels)

        records.append({
            "taxon"     : taxon,
            "lfc"       : round(coef, 4) if not np.isnan(coef) else np.nan,
            "se"        : round(se,   4) if not np.isnan(se)   else np.nan,
            "t_stat"    : round(t_st, 4) if not np.isnan(t_st) else np.nan,
            "p_ols"     : p_ols,
            "p_val"     : p_blocked,
            "wilcoxon_p": p_blocked,
            "auroc"     : auroc,
            "gfc"       : round(gfc, 4) if not np.isnan(gfc) else np.nan,
            "n_ctrl"    : n_ctrl,
            "n_crc"     : n_crc,
        })

        if i % 50 == 0:
            print(f"  {i}/{len(kept_taxa)} taxa tested...")

    if not records:
        print("  ERROR: No valid DA results.")
        return pd.DataFrame()

    df_res = pd.DataFrame(records)
    _, q, _, _ = multipletests(df_res["p_val"].fillna(1), method="fdr_bh")
    df_res["q_val"]       = np.round(q, 6)
    df_res["significant"] = df_res["q_val"] < Q_THRESHOLD_POOLED
    df_res["direction"]   = np.where(df_res["lfc"] > 0, "enriched_CRC", "depleted_CRC")
    df_res = df_res.sort_values("q_val").reset_index(drop=True)

    n_sig = int(df_res["significant"].sum())
    print(f"\nPooled DA: {len(df_res)} tested, {n_sig} significant (q<{Q_THRESHOLD_POOLED})")
    print(f"  Enriched: {int((df_res['significant'] & (df_res['direction']=='enriched_CRC')).sum())}")
    print(f"  Depleted: {int((df_res['significant'] & (df_res['direction']=='depleted_CRC')).sum())}")
    print(f"  q_val: min={df_res['q_val'].min():.4f}  median={df_res['q_val'].median():.4f}")
    print(f"  p_val: min={df_res['p_val'].min():.6f}  median={df_res['p_val'].median():.4f}")
    return df_res


# ===========================================================
# INTERSECTION TABLE
# ===========================================================

def build_intersection_table(pooled_df, agree=None):
    """
    Taxa significant in pooled DA AND in >=2 per-cohort datasets.
    Loads da_agreement CSV from disk if agree is not passed directly
    (requires da_percohort.py to have been run first).
    """
    if pooled_df.empty:
        return pd.DataFrame()

    if agree is None:
        try:
            agree = pd.read_csv(OUT_AGREEMENT)
            print(f"  Loaded agreement stats from: {OUT_AGREEMENT}")
        except FileNotFoundError:
            print(f"  WARNING: {OUT_AGREEMENT} not found. "
                  "Run da_percohort.py first to get the intersection table.")
            return pd.DataFrame()

    pooled_sig    = pooled_df[pooled_df["significant"]].copy()
    percohort_sig = agree[agree["n_significant"] >= 2].copy()

    if pooled_sig.empty or percohort_sig.empty:
        return pd.DataFrame()

    keep_agree = [c for c in [
        "taxon", "n_tested", "n_significant", "pct_significant",
        "direction_vote", "direction_consistency", "agreement_tier",
        "mean_lfc", "sd_lfc",
    ] if c in percohort_sig.columns]

    intersection = pooled_sig.merge(
        percohort_sig[keep_agree], on="taxon", how="inner",
        suffixes=("_pooled", "_percohort"),
    )
    sort_cols = [c for c in ["q_val", "n_significant", "direction_consistency"]
                 if c in intersection.columns]
    if sort_cols:
        intersection = intersection.sort_values(
            sort_cols, ascending=[True, False, False][:len(sort_cols)]
        )
    return intersection.reset_index(drop=True)


# ===========================================================
# ANALYSIS C — VALIDATION GRID
# ===========================================================

def build_validation_grid(df_feat, df_meta, pooled_df, top_n=40):
    """
    For each pooled-significant taxon x each dataset compute the median CLR
    difference (CRC − control). Raw diff is stored for gradient colouring;
    p-values are kept separately for significance stars.
    """
    if pooled_df.empty or "significant" not in pooled_df.columns:
        print("  Pooled results empty — validation grid empty.")
        return pd.DataFrame(), pd.DataFrame()

    sig_taxa = pooled_df[pooled_df["significant"]].head(top_n)["taxon"].tolist()
    if not sig_taxa:
        print("  No pooled-significant taxa — validation grid empty.")
        return pd.DataFrame(), pd.DataFrame()

    kept   = get_prevalent_taxa(df_feat, df_meta)
    df_sub = df_feat[kept].copy()
    df_sub = renormalize_features(df_sub)

    valid  = df_sub.notna().all(axis=1)
    df_sub = df_sub.loc[valid].reset_index(drop=True)
    df_mv  = df_meta.loc[valid].copy().reset_index(drop=True)
    clr    = clr_features(df_sub)

    sig_taxa = [t for t in sig_taxa if t in clr.columns]
    datasets = sorted(df_mv["dataset_name"].unique())

    grid     = pd.DataFrame(np.nan, index=sig_taxa, columns=datasets)
    pval_mat = pd.DataFrame(np.nan, index=sig_taxa, columns=datasets)

    print(f"\nValidation grid: {len(sig_taxa)} taxa x {len(datasets)} datasets")
    print("  Cell colour = median ΔCLR (CRC − control)")
    print(f"  Significance star = Wilcoxon p < {GRID_P_THRESHOLD}")

    for ds in datasets:
        mask_ctrl = (df_mv["dataset_name"] == ds) & (df_mv["condition"] == "control")
        mask_crc  = (df_mv["dataset_name"] == ds) & (df_mv["condition"] == "CRC")
        if mask_ctrl.sum() < MIN_GROUP_SIZE or mask_crc.sum() < MIN_GROUP_SIZE:
            continue

        for taxon in sig_taxa:
            ctrl_vals = clr.loc[mask_ctrl, taxon].values
            crc_vals  = clr.loc[mask_crc,  taxon].values
            grid.loc[taxon, ds] = np.median(crc_vals) - np.median(ctrl_vals)

            try:
                _, p = mannwhitneyu(ctrl_vals, crc_vals, alternative="two-sided")
            except Exception:
                p = np.nan
            pval_mat.loc[taxon, ds] = p

    return grid, pval_mat


# ===========================================================
# PLOTS
# ===========================================================

def plot_volcano(pooled_df, save_path=None):
    if pooled_df.empty:
        print("  Pooled results empty — volcano skipped.")
        return

    df = pooled_df.copy()
    df["neg_log10_q"] = -np.log10(df["q_val"].clip(lower=1e-10))
    colours = np.where(df["significant"] & (df["lfc"] > 0), "#F0997B",
              np.where(df["significant"] & (df["lfc"] < 0), "#5DCAA5", "#CCCCCC"))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")
    ax.scatter(df["lfc"], df["neg_log10_q"], c=colours, s=22,
               alpha=0.75, edgecolors="none", zorder=3)
    ax.axhline(-np.log10(Q_THRESHOLD_POOLED), color="#888", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(0, color="#888", ls="--", lw=0.8, alpha=0.7)

    for _, row in df[df["significant"]].nlargest(15, "neg_log10_q").iterrows():
        ax.annotate(_short(row["taxon"], 28),
                    xy=(row["lfc"], row["neg_log10_q"]),
                    xytext=(6, 2), textcoords="offset points", fontsize=6.5,
                    arrowprops=dict(arrowstyle="-", color="#AAA", lw=0.5))

    ax.set_xlabel("LFC / OLS coefficient on CLR (CRC vs control)", fontsize=11)
    ax.set_ylabel("-log10(q)", fontsize=11)
    ax.set_title(
        f"Volcano — Pooled DA ({TAXON_LEVEL})\n"
        f"{int(df['significant'].sum())} significant taxa (q<{Q_THRESHOLD_POOLED})",
        fontsize=12,
    )
    ax.legend(handles=[
        Patch(facecolor="#F0997B", label="Enriched in CRC"),
        Patch(facecolor="#5DCAA5", label="Depleted in CRC"),
        Patch(facecolor="#CCC",    label="Not significant"),
    ], fontsize=9, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#CCC", alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Volcano saved -> {save_path}")
    plt.show()


def plot_validation_grid(grid_df, pval_mat, pooled_df, df_meta, save_path=None):
    if grid_df.empty:
        print("  Validation grid empty — skipping.")
        return

    # ── order rows by pooled LFC (high → low) ─────────────────────────────
    lfc_lookup = pooled_df.set_index("taxon")["lfc"]
    taxa_order = (grid_df.index.to_series()
                  .map(lfc_lookup).sort_values(ascending=False).index.tolist())
    mat          = grid_df.loc[taxa_order]
    pval_aligned = pval_mat.loc[taxa_order]

    # ── bin raw diff into integer codes ────────────────────────────────────
    #  0      = not tested (NaN)
    #  1 – 6  = depleted   (diff < 0), bin 1=smallest … 6=largest
    #  7 – 12 = enriched   (diff > 0), bin 7=smallest … 12=largest
    EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf])

    def _code(v):
        if np.isnan(v):
            return 0
        b = min(int(np.searchsorted(EDGES[1:], abs(v), side="right")) + 1, 6)
        return b + 6 if v > 0 else b

    int_mat = mat.applymap(_code).values

    # ── colourmap ──────────────────────────────────────────────────────────
    dep_colours = ["#C8EDE3", "#89D9BC", "#5DCAA5", "#2DA87E", "#1A7A5A", "#0D5240"]
    enr_colours = ["#FDEAE4", "#F5C4B0", "#F0997B", "#E86C46", "#C0392B", "#8B1A0D"]
    cmap = ListedColormap(["#D3D1C7"] + dep_colours + enr_colours)
    norm = BoundaryNorm(np.arange(-0.5, 13.5, 1), cmap.N)

    n_taxa, n_coh = mat.shape
    fig, ax = plt.subplots(figsize=(max(10, n_coh * 1.3), max(8, n_taxa * 0.32)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F8F9FA")
    ax.imshow(int_mat, aspect="auto", cmap=cmap, norm=norm, interpolation="none")

    # ── significance stars ─────────────────────────────────────────────────
    for row_i, taxon in enumerate(taxa_order):
        for col_j, ds in enumerate(mat.columns):
            p = pval_aligned.loc[taxon, ds]
            if not np.isnan(p) and p < GRID_P_THRESHOLD:
                ax.text(col_j, row_i, "★", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

    # ── x-axis: dataset labels + sample counts ────────────────────────────
    ax.set_xticks(range(n_coh))
    ax.set_xticklabels([DATASET_LABEL_MAP.get(ds, ds) for ds in mat.columns],
                       rotation=35, ha="right", fontsize=9)
    for col_idx, ds in enumerate(mat.columns):
        n_c = int(((df_meta["dataset_name"] == ds) & (df_meta["condition"] == "control")).sum())
        n_r = int(((df_meta["dataset_name"] == ds) & (df_meta["condition"] == "CRC")).sum())
        ax.text(col_idx, -0.7, f"ctrl={n_c}\nCRC={n_r}",
                ha="center", va="bottom", fontsize=6, color="#555",
                transform=ax.get_xaxis_transform())

    # ── y-axis: taxon labels coloured by pooled direction ─────────────────
    dir_lookup = pooled_df.set_index("taxon")["direction"]
    ax.set_yticks(range(n_taxa))
    for i, taxon in enumerate(taxa_order):
        colour = "#C0392B" if dir_lookup.get(taxon, "") == "enriched_CRC" else "#1A6B55"
        ax.text(-0.05, i, _short(taxon), ha="right", va="center", fontsize=7.5,
                color=colour, transform=ax.get_yaxis_transform())
    ax.set_yticklabels([])

    ax.set_title(
        f"Per-dataset validation of pooled-significant taxa — {TAXON_LEVEL} level\n"
        f"Colour = median ΔCLR magnitude  |  ★ Wilcoxon p < {GRID_P_THRESHOLD}  |  "
        f"Pooled q < {Q_THRESHOLD_POOLED}",
        fontsize=11, pad=14,
    )

    # ── legend ────────────────────────────────────────────────────────────
    bin_labels = ["0 – 0.2", "0.2 – 0.4", "0.4 – 0.6",
                  "0.6 – 0.8", "0.8 – 1.0", "≥ 1.0"]
    legend_items = (
        [Patch(facecolor="#D3D1C7", label="Not tested / too few samples")]
        + [Patch(facecolor=c, label=f"Depleted  |ΔCLR| {lbl}")
           for c, lbl in zip(dep_colours, bin_labels)]
        + [Patch(facecolor=c, label=f"Enriched  |ΔCLR| {lbl}")
           for c, lbl in zip(enr_colours, bin_labels)]
        + [Patch(facecolor="grey", label=f"★ = Wilcoxon p < {GRID_P_THRESHOLD}")]
    )
    plt.subplots_adjust(bottom=0.18, right=0.70)
    ax.legend(handles=legend_items,
              bbox_to_anchor=(1.03, 1.0), loc="upper left",
              bbox_transform=ax.transAxes,
              fontsize=7.5, framealpha=0.95, edgecolor="#CCC",
              title="Cell colour  (median ΔCLR)", title_fontsize=8,
              borderaxespad=0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Validation grid saved -> {save_path}")
    plt.show()


# ===========================================================
# MAIN
# ===========================================================

def main():
    df_feat, df_meta = load_data()

    # ── Analysis B: Pooled ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ANALYSIS B — POOLED DA (blocked permutation)")
    print(f"{'='*60}")

    pooled_df = run_pooled_da(df_feat, df_meta)
    pooled_df.to_csv(OUT_POOLED, index=False)
    print(f"\n  Saved: {OUT_POOLED}")

    # Intersection with per-cohort results (loads agreement CSV if available)
    intersection_df = build_intersection_table(pooled_df)
    if not intersection_df.empty:
        intersection_df.to_csv(OUT_INTERSECTION, index=False)
        print(f"  Saved: {OUT_INTERSECTION}")
        print(f"  Intersection (pooled AND >=2 cohorts): {len(intersection_df)} taxa")
        print(intersection_df[
            ["taxon", "q_val", "n_significant", "agreement_tier", "direction"]
        ].head(10))

    # ── Analysis C: Validation grid ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ANALYSIS C — PER-DATASET VALIDATION GRID")
    print(f"{'='*60}")

    grid_df, pval_mat = build_validation_grid(df_feat, df_meta, pooled_df)
    if not grid_df.empty:
        grid_df.to_csv(OUT_GRID, index=True)
        print(f"  Saved: {OUT_GRID}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PLOTS")
    print(f"{'='*60}")

    plot_volcano(pooled_df, save_path=OUT_VOLCANO)
    if not grid_df.empty:
         plot_validation_grid(grid_df, pval_mat, pooled_df, df_meta, save_path=OUT_HM_POOLED)

    return {
        "pooled_df"      : pooled_df,
        "intersection_df": intersection_df,
        "grid_df"        : grid_df,
        "pval_mat"       : pval_mat,
    }


if __name__ == "__main__":
    out = main()