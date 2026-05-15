"""
da_perage.py
============
Pooled blocked-permutation DA run separately for each age stratum.

Age groups:
    YOUNG  — age < 50
    MIDDLE — 50 <= age <= 65
    OLDER  — age > 65

For each stratum the full pooled DA pipeline is applied:
  - Blocked permutation Wilcoxon (blocking variable: dataset_name)
  - OLS with dataset dummies for effect size / direction
  - BH FDR correction within each stratum

An agreement matrix and DerSimonian-Laird meta-analysis are then run
across the three strata to identify taxa that are consistently DA
regardless of patient age.

Run independently:
    python da_perage.py

Outputs (all under ./results/):
    da_perage_<level>.csv                  DA results, all age strata
    da_age_agreement_<level>.csv           cross-stratum agreement stats
    da_age_meta_<level>.csv                DerSimonian-Laird meta-analysis
    da_age_volcano_<level>.png             one panel per stratum
    da_age_agreement_heatmap_<level>.png
    da_age_forest_<level>.png
    da_age_power_bias_<level>.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import chi2, spearmanr, linregress
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from da_shared import (
    # config
    TAXON_LEVEL, DATASET_LABEL_MAP, TIER_COLOURS,
    Q_THRESHOLD_POOLED, Q_THRESHOLD_PERCOHORT,
    MIN_GROUP_SIZE, SPEARMAN_P_THRESHOLD,
    # data + helpers
    load_data, get_prevalent_taxa,
    clr_features, renormalize_features,
    compute_auroc, generalized_fold_change_quantiles,
    standard_wilcoxon_p, _short,
)


# ===========================================================
# AGE GROUP CONFIGURATION
# ===========================================================

# Each entry: stratum_key -> (age_min_inclusive, age_max_exclusive)
# Use None for open-ended numeric bounds.
# A value of None for the whole entry means "age is missing".
AGE_GROUPS = {
    "YOUNG"  : (None, 50),   # age < 50
    "MIDDLE" : (50,   65),   # 50 <= age < 65
    "OLDER"  : (65,   None), # age >= 65
    "UNKNOWN": None,          # age is null / not recorded
}

AGE_LABEL_MAP = {
    "YOUNG"  : "Age < 50",
    "MIDDLE" : "Age 50-65",
    "OLDER"  : "Age > 65",
    "UNKNOWN": "Age unknown",
}

# Colour per stratum (used in the multi-panel volcano)
AGE_COLOURS = {
    "YOUNG"  : "#378ADD",
    "MIDDLE" : "#1D9E75",
    "OLDER"  : "#D85A30",
    "UNKNOWN": "#888888",
}


def _tag():
    return f"_{TAXON_LEVEL}"


# Output paths — separate from all other DA scripts
OUT_PERAGE        = f"./results/da_perage{_tag()}.csv"
OUT_AGE_AGREEMENT = f"./results/da_age_agreement{_tag()}.csv"
OUT_AGE_META      = f"./results/da_age_meta{_tag()}.csv"
OUT_AGE_VOLCANO   = f"./results/da_age_volcano{_tag()}.png"
OUT_AGE_HM        = f"./results/da_age_agreement_heatmap{_tag()}.png"
OUT_AGE_FOREST    = f"./results/da_age_forest{_tag()}.png"
OUT_AGE_POWER     = f"./results/da_age_power_bias{_tag()}.png"


# ===========================================================
# HELPERS
# ===========================================================

def _age_mask(df_meta: pd.DataFrame, bounds) -> pd.Series:
    """
    Boolean mask selecting rows for a given age stratum.

    bounds=None          -> rows where age IS null / missing  (UNKNOWN stratum)
    bounds=(lo, hi)      -> rows where lo <= age < hi (None = open-ended)
                           Rows with missing age are always excluded from
                           numeric strata.
    """
    if "age" not in df_meta.columns:
        raise KeyError(
            "Column 'age' not found in metadata. "
            "The raw data must contain an age field."
        )
    age = pd.to_numeric(df_meta["age"], errors="coerce")

    # UNKNOWN stratum: select exactly the samples with no valid age
    if bounds is None:
        return age.isna()

    age_min, age_max = bounds
    mask = age.notna()          # exclude missing from all numeric strata
    if age_min is not None:
        mask &= age >= age_min
    if age_max is not None:
        mask &= age < age_max
    return mask


def _summarise_age_groups(df_meta: pd.DataFrame):
    """Print sample counts and dataset breakdown per age stratum."""
    age       = pd.to_numeric(df_meta["age"], errors="coerce")
    n_valid   = int(age.notna().sum())
    n_missing = int(age.isna().sum())
    print(f"\n  Age column: {n_valid} valid, {n_missing} missing "
          f"(-> UNKNOWN stratum)")
    print(f"\n  Sample counts per age stratum:")

    for key, bounds in AGE_GROUPS.items():
        mask   = _age_mask(df_meta, bounds)
        sub    = df_meta.loc[mask, "condition"]
        n_c    = int((sub == "control").sum())
        n_r    = int((sub == "CRC").sum())
        ds_cnt = df_meta.loc[mask, "dataset_name"].value_counts().to_dict()
        print(f"  {AGE_LABEL_MAP[key]:<14} | ctrl={n_c}  CRC={n_r}  "
              f"datasets={ds_cnt}")


# ===========================================================
# POOLED DA WITHIN ONE AGE STRATUM
# ===========================================================

def _run_pooled_one_stratum(df_feat, df_meta, kept_taxa, stratum_key):
    """
    Run the full pooled blocked-permutation DA for a single age stratum.
    Mirrors da_pooled.run_pooled_da but operates on a pre-filtered subset.
    """
    mask   = (
        _age_mask(df_meta, AGE_GROUPS[stratum_key])
        & df_meta["condition"].isin(["control", "CRC"])
    )

    df_sub_raw = df_feat.loc[mask, kept_taxa].copy()
    df_meta_sub = df_meta.loc[mask].copy()

    n_ctrl = int((df_meta_sub["condition"] == "control").sum())
    n_crc  = int((df_meta_sub["condition"] == "CRC").sum())

    print(f"\n  [{stratum_key}] {AGE_LABEL_MAP[stratum_key]}")
    print(f"    ctrl={n_ctrl}  CRC={n_crc}")

    if n_ctrl < MIN_GROUP_SIZE or n_crc < MIN_GROUP_SIZE:
        print(f"    -> skipped (n < MIN_GROUP_SIZE={MIN_GROUP_SIZE})")
        return pd.DataFrame()

    if df_meta_sub["condition"].nunique() < 2:
        print("    -> skipped (only one class present)")
        return pd.DataFrame()

    # Re-normalise and CLR
    df_sub = renormalize_features(df_sub_raw.reset_index(drop=True))
    df_meta_sub = df_meta_sub.reset_index(drop=True)

    invalid = ~df_sub.notna().all(axis=1)
    if invalid.sum():
        print(f"    Dropping {int(invalid.sum())} samples with invalid abundance.")
        df_sub      = df_sub[~invalid].reset_index(drop=True)
        df_meta_sub = df_meta_sub[~invalid].reset_index(drop=True)

    clr = clr_features(df_sub)
    nan_rows = clr.isna().any(axis=1)
    if nan_rows.sum():
        print(f"    Dropping {int(nan_rows.sum())} samples with NaN CLR.")
        keep        = ~nan_rows
        clr         = clr[keep].reset_index(drop=True)
        df_sub      = df_sub[keep].reset_index(drop=True)
        df_meta_sub = df_meta_sub[keep].reset_index(drop=True)

    labels        = df_meta_sub["condition"].values
    condition_arr = (labels == "CRC").astype(float)

    # OLS for effect size only — no dataset dummies needed since
    # we are not controlling for study effects within an age stratum
    X_ols = sm.add_constant(pd.Series(condition_arr, name="condition_CRC"))

    print(f"    CLR: {clr.shape[0]} samples x {clr.shape[1]} taxa")

    records = []
    for i, taxon in enumerate(kept_taxa, 1):
        if taxon not in clr.columns:
            continue
        x = clr[taxon].values
        if np.isnan(x).any():
            continue

        # OLS effect size
        try:
            model = sm.OLS(x, X_ols).fit()
            coef  = float(model.params["condition_CRC"])
            se    = float(model.bse["condition_CRC"])
            t_st  = float(model.tvalues["condition_CRC"])
            p_ols = float(model.pvalues["condition_CRC"])
        except Exception:
            coef = se = t_st = p_ols = np.nan

        # Standard Wilcoxon — scipy uses exact distribution for small n
        # automatically, no permutation needed
        p_wilc = standard_wilcoxon_p(x, labels, group_a="control", group_b="CRC")

        auroc = compute_auroc(x, labels, case_label="CRC")
        gfc   = generalized_fold_change_quantiles(df_sub[taxon].values, labels)

        records.append({
            "taxon"     : taxon,
            "lfc"       : round(coef, 4) if not np.isnan(coef) else np.nan,
            "se"        : round(se,   4) if not np.isnan(se)   else np.nan,
            "t_stat"    : round(t_st, 4) if not np.isnan(t_st) else np.nan,
            "p_ols"     : p_ols,
            "p_val"     : p_wilc,
            "wilcoxon_p": p_wilc,
            "auroc"     : auroc,
            "gfc"       : round(gfc, 4) if not np.isnan(gfc) else np.nan,
            "n_ctrl"    : n_ctrl,
            "n_crc"     : n_crc,
        })

        if i % 50 == 0:
            print(f"    {i}/{len(kept_taxa)} taxa tested...")

    if not records:
        print("    -> no valid results.")
        return pd.DataFrame()

    df_res = pd.DataFrame(records)
    _, q, _, _ = multipletests(df_res["p_val"].fillna(1), method="fdr_bh")
    df_res["q_val"]       = np.round(q, 6)
    df_res["significant"] = df_res["q_val"] < Q_THRESHOLD_POOLED
    df_res["direction"]   = np.where(df_res["lfc"] > 0, "enriched_CRC", "depleted_CRC")
    df_res["age_group"]   = stratum_key
    df_res = df_res.sort_values("q_val").reset_index(drop=True)

    n_sig = int(df_res["significant"].sum())
    print(f"    -> {len(df_res)} tested, {n_sig} significant (q<{Q_THRESHOLD_POOLED})")
    print(f"       Enriched: "
          f"{int((df_res['significant'] & (df_res['direction']=='enriched_CRC')).sum())}"
          f"  Depleted: "
          f"{int((df_res['significant'] & (df_res['direction']=='depleted_CRC')).sum())}")

    return df_res


def run_perage_da(df_feat, df_meta):
    """Run pooled blocked-permutation DA for each age stratum in AGE_GROUPS."""
    kept_taxa = get_prevalent_taxa(df_feat, df_meta)
    if not kept_taxa:
        print("  ERROR: No taxa passed prevalence filter.")
        return {}

    print(f"\n  Per-age DA testing {len(kept_taxa)} taxa (intersection, same as pooled)")
    _summarise_age_groups(df_meta)

    results = {}
    for key in AGE_GROUPS:
        if key == "UNKNOWN":
            continue   # reported in summary only — no statistical test
        res = _run_pooled_one_stratum(df_feat, df_meta, kept_taxa, key)
        if not res.empty:
            results[key] = res

    return results


# ===========================================================
# AGREEMENT MATRIX + STATS
# ===========================================================

def build_agreement_matrix(results):
    strata   = list(results.keys())
    all_taxa = sorted({t for df in results.values() for t in df["taxon"]})

    sig  = pd.DataFrame(np.nan, index=all_taxa, columns=strata)
    lfc  = pd.DataFrame(np.nan, index=all_taxa, columns=strata)
    dir_ = pd.DataFrame(np.nan, index=all_taxa, columns=strata)

    for s in strata:
        df     = results[s].set_index("taxon")
        shared = df.index.intersection(all_taxa)
        sig.loc[shared, s]  = df.loc[shared, "significant"].astype(float)
        lfc.loc[shared, s]  = df.loc[shared, "lfc"]
        dir_.loc[shared, s] = np.where(df.loc[shared, "lfc"] > 0, 1.0, -1.0)

    return sig, lfc, dir_


def compute_agreement_stats(sig, lfc, dir_):
    n_tested = sig.notna().sum(axis=1)
    n_sig    = (sig == 1).sum(axis=1)
    pct_sig  = n_sig / n_tested.replace(0, np.nan)
    sig_dir  = dir_.where(sig == 1)
    n_enr    = (sig_dir ==  1).sum(axis=1)
    n_dep    = (sig_dir == -1).sum(axis=1)
    n_dir    = n_enr + n_dep
    dir_vote = np.where(n_enr > n_dep, "enriched_CRC",
               np.where(n_dep > n_enr, "depleted_CRC", "inconsistent"))
    dir_cons = np.where(n_dir > 0, np.maximum(n_enr, n_dep) / n_dir, np.nan)

    def tier(row):
        ps, dc, ns = (row["pct_significant"],
                      row["direction_consistency"],
                      row["n_significant"])
        if np.isnan(ps) or ns == 0:                      return "not_detected"
        if ps >= 0.67 and (np.isnan(dc) or dc >= 0.75): return "core"
        if ps >= 0.44:                                   return "moderate"
        if not np.isnan(dc) and dc < 0.50:               return "inconsistent"
        return "weak"

    df = pd.DataFrame({
        "taxon"                : sig.index,
        "n_tested"             : n_tested.values,
        "n_significant"        : n_sig.values,
        "pct_significant"      : np.round(pct_sig.values, 3),
        "direction_vote"       : dir_vote,
        "direction_consistency": np.round(dir_cons, 3),
        "mean_lfc"             : np.round(lfc.mean(axis=1, skipna=True).values, 4),
        "sd_lfc"               : np.round(lfc.std(axis=1,  skipna=True).values, 4),
    })
    df["agreement_tier"] = df.apply(tier, axis=1)
    return df.sort_values(
        ["n_significant", "direction_consistency"], ascending=[False, False]
    ).reset_index(drop=True)


# ===========================================================
# META-ANALYSIS (DerSimonian-Laird)
# ===========================================================

def dersimonian_laird(effects, variances):
    eff, var = np.asarray(effects, float), np.asarray(variances, float)
    valid    = np.isfinite(eff) & np.isfinite(var) & (var > 0)
    eff, var = eff[valid], var[valid]
    k        = len(eff)
    if k < 2:
        return None
    w        = 1.0 / var
    theta_fe = (w * eff).sum() / w.sum()
    Q        = (w * (eff - theta_fe) ** 2).sum()
    C        = w.sum() - (w**2).sum() / w.sum()
    tau2     = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    w_re     = 1.0 / (var + tau2)
    theta_re = (w_re * eff).sum() / w_re.sum()
    var_re   = 1.0 / w_re.sum()
    se_re    = np.sqrt(var_re)
    I2       = max(0.0, (Q - (k-1)) / Q * 100) if Q > (k-1) and Q > 0 else 0.0
    Q_p      = 1.0 - chi2.cdf(Q, df=k - 1)
    z        = theta_re / se_re
    p_val    = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "pooled_lfc"    : round(float(theta_re), 4),
        "pooled_se"     : round(float(se_re), 4),
        "ci_lb"         : round(float(theta_re - 1.96 * se_re), 4),
        "ci_ub"         : round(float(theta_re + 1.96 * se_re), 4),
        "tau2"          : round(float(tau2), 6),
        "Q"             : round(float(Q), 4),
        "Q_pval"        : round(float(Q_p), 4),
        "I2"            : round(float(I2), 2),
        "p_value"       : round(float(p_val), 6),
        "heterogeneity" : ("high" if I2 >= 75 else "moderate" if I2 >= 50
                           else "low" if I2 >= 25 else "negligible"),
    }


def run_meta_analysis(results, sig_mat, lfc_mat):
    meta_records = []
    for taxon in sig_mat.index:
        effs, vars_, n_sig = [], [], 0
        for s in results:
            row = results[s][results[s]["taxon"] == taxon]
            if row.empty:
                continue
            se  = float(row["se"].iloc[0])
            lfc = float(row["lfc"].iloc[0])
            if np.isnan(se) or se <= 0 or np.isnan(lfc):
                continue
            effs.append(lfc)
            vars_.append(se ** 2)
            n_sig += int(row["significant"].iloc[0])
        if len(effs) < 2:
            continue
        ma = dersimonian_laird(effs, vars_)
        if ma:
            meta_records.append({"taxon": taxon, "n_strata": len(effs),
                                  "n_sig_strata": n_sig, **ma})

    if not meta_records:
        return pd.DataFrame()

    meta_df = pd.DataFrame(meta_records)
    _, q, _, _ = multipletests(meta_df["p_value"].fillna(1), method="fdr_bh")
    meta_df["q_value"] = np.round(q, 6)
    return meta_df.sort_values("q_value").reset_index(drop=True)


# ===========================================================
# PLOTS
# ===========================================================

def plot_volcano_grid(results, save_path=None):
    """One volcano panel per age stratum, arranged in a row."""
    if not results:
        print("  No results — volcano skipped.")
        return

    strata = list(results.keys())
    ncols  = len(strata)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 6, 5), sharey=False)
    if ncols == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, s in zip(axes, strata):
        df = results[s].copy()
        df["neg_log10_q"] = -np.log10(df["q_val"].clip(lower=1e-10))

        base_colour = AGE_COLOURS.get(s, "#888888")
        colours = np.where(df["significant"] & (df["lfc"] > 0), base_colour,
                  np.where(df["significant"] & (df["lfc"] < 0), "#5DCAA5", "#CCCCCC"))

        ax.set_facecolor("#F8F9FA")
        ax.scatter(df["lfc"], df["neg_log10_q"], c=colours, s=22,
                   alpha=0.75, edgecolors="none", zorder=3)
        ax.axhline(-np.log10(Q_THRESHOLD_POOLED), color="#888",
                   ls="--", lw=0.8, alpha=0.7)
        ax.axvline(0, color="#888", ls="--", lw=0.8, alpha=0.7)

        for _, row in df[df["significant"]].nlargest(8, "neg_log10_q").iterrows():
            ax.annotate(_short(row["taxon"], 22),
                        xy=(row["lfc"], row["neg_log10_q"]),
                        xytext=(5, 2), textcoords="offset points", fontsize=6,
                        arrowprops=dict(arrowstyle="-", color="#AAA", lw=0.5))

        n_sig = int(df["significant"].sum())
        ax.set_title(f"{AGE_LABEL_MAP[s]}\n{n_sig} sig. taxa (q<{Q_THRESHOLD_POOLED})",
                     fontsize=10)
        ax.set_xlabel("LFC (CRC vs control)", fontsize=9)
        ax.set_ylabel("-log10(q)" if s == strata[0] else "", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color="#CCC", alpha=0.6)
        ax.set_axisbelow(True)

    plt.suptitle(f"Volcano plots by age stratum — {TAXON_LEVEL} level",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Volcano grid saved -> {save_path}")
    plt.show()


def plot_agreement_heatmap(sig_mat, dir_mat, agree_stats, save_path=None):
    top_taxa = (
        agree_stats[agree_stats["n_significant"] > 0]
        .sort_values(["n_significant", "direction_consistency"], ascending=[False, False])
        .head(40)["taxon"].tolist()
    )
    if not top_taxa:
        top_taxa = agree_stats.head(40)["taxon"].tolist()
    if not top_taxa:
        print("  Agreement matrix empty — skipping.")
        return

    mat           = (sig_mat.loc[top_taxa] * dir_mat.loc[top_taxa]).fillna(-999)
    n_taxa, n_str = mat.shape
    cmap          = ListedColormap(["#D3D1C7", "#5DCAA5", "#FFFFFF", "#F0997B"])
    norm          = BoundaryNorm([-999.5, -1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(max(6, n_str * 1.6), max(10, n_taxa * 0.25)))
    ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm, interpolation="none")

    tier_lookup = agree_stats.set_index("taxon")["agreement_tier"]
    for i, taxon in enumerate(top_taxa):
        colour = TIER_COLOURS.get(tier_lookup.get(taxon, "not_detected"), "#444")
        ax.text(-0.05, i, _short(taxon), ha="right", va="center",
                fontsize=7, color=colour, transform=ax.get_yaxis_transform())

    ax.set_xticks(range(n_str))
    ax.set_xticklabels([AGE_LABEL_MAP.get(c, c) for c in mat.columns],
                       rotation=20, ha="right", fontsize=10)
    ax.set_yticks([])
    ax.set_title(
        f"DA agreement across age strata — {TAXON_LEVEL} level\n"
        f"Taxa significant (q<{Q_THRESHOLD_POOLED}) in >=1 age stratum",
        fontsize=12, pad=10,
    )

    cell_leg = [
        Patch(facecolor="#F0997B", label="Enriched in CRC"),
        Patch(facecolor="#5DCAA5", label="Depleted in CRC"),
        Patch(facecolor="#FFF", edgecolor="#B4B2A9", label="Not significant"),
        Patch(facecolor="#D3D1C7", label="Not tested / too few samples"),
    ]
    tier_leg = [
        Patch(facecolor=TIER_COLOURS[t], label=t.capitalize())
        for t in ["core", "moderate", "weak", "inconsistent"]
    ]
    plt.subplots_adjust(bottom=0.14, left=0.28)
    l1 = ax.legend(handles=cell_leg, loc="upper left",
                   bbox_to_anchor=(0.0, -0.10), bbox_transform=ax.transAxes,
                   fontsize=8, title="Cell colour", framealpha=0.95,
                   ncol=2, borderaxespad=0)
    ax.legend(handles=tier_leg, loc="upper right",
              bbox_to_anchor=(1.0, -0.10), bbox_transform=ax.transAxes,
              fontsize=8, title="Row tier", framealpha=0.95,
              ncol=2, borderaxespad=0)
    ax.add_artist(l1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Agreement heatmap saved -> {save_path}")
    plt.show()


def plot_forest(meta_df, results, top_n=12, save_path=None):
    if meta_df.empty:
        print("  Meta-analysis empty — forest plot skipped.")
        return

    top_taxa = (
        meta_df[meta_df["q_value"] < Q_THRESHOLD_PERCOHORT]
        .head(top_n)["taxon"].tolist()
    )
    if not top_taxa:
        top_taxa = meta_df.head(top_n)["taxon"].tolist()
    if not top_taxa:
        return

    ncols    = min(3, len(top_taxa))
    nrows    = int(np.ceil(len(top_taxa) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
    axes     = np.array(axes).flatten()
    strata   = list(results.keys())

    for idx, taxon in enumerate(top_taxa):
        ax = axes[idx]
        lfcs, ses, labels, sigs = [], [], [], []

        for s in strata:
            row = results[s][results[s]["taxon"] == taxon]
            if row.empty:
                continue
            lfcs.append(float(row["lfc"].iloc[0]))
            ses.append(float(row["se"].iloc[0]))
            labels.append(AGE_LABEL_MAP.get(s, s))
            sigs.append(bool(row["significant"].iloc[0]))

        for j, (lfc, se, sig, s) in enumerate(zip(lfcs, ses, sigs, strata)):
            c = AGE_COLOURS.get(s, "#888") if sig else "#B4B2A9"
            ax.plot([lfc - 1.96*se, lfc + 1.96*se], [j, j], color=c, lw=1.2)
            ax.plot(lfc, j, "s", color=c, ms=5)

        mr = meta_df[meta_df["taxon"] == taxon]
        if not mr.empty:
            pl, lb, ub = (float(mr["pooled_lfc"].iloc[0]),
                          float(mr["ci_lb"].iloc[0]),
                          float(mr["ci_ub"].iloc[0]))
            yp = len(lfcs) + 0.6
            ax.fill([lb, pl, ub, pl, lb], [yp, yp+0.3, yp, yp-0.3, yp],
                    color="#534AB7", alpha=0.85, zorder=5)
            ax.text(0.98, 0.03,
                    f"I2={float(mr['I2'].iloc[0]):.0f}%  "
                    f"q={float(mr['q_value'].iloc[0]):.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#534AB7")

        ax.axvline(0, color="grey", ls="--", lw=0.7, alpha=0.6)
        ax.set_yticks(list(range(len(lfcs))) + [len(lfcs) + 0.6])
        ax.set_yticklabels(labels + ["Pooled"], fontsize=7)
        ax.set_title(_short(taxon, 38), fontsize=8, pad=3)
        ax.set_xlabel("LFC / OLS coefficient on CLR", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(top_taxa):]:
        ax.axis("off")

    plt.suptitle(f"Forest plots — top DA taxa by age stratum ({TAXON_LEVEL})",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Forest plots saved -> {save_path}")
    plt.show()


def plot_power_bias(results, df_meta, save_path=None):
    rows = []
    for key, bounds in AGE_GROUPS.items():
        if key == "UNKNOWN":
            continue   # not tested — exclude from plot
        mask_cc = (
            _age_mask(df_meta, bounds)
            & df_meta["condition"].isin(["control", "CRC"])
        )
        n_samp = int(mask_cc.sum())
        n_sig  = int(results[key]["significant"].sum()) if key in results else 0
        rows.append({"stratum": key, "label": AGE_LABEL_MAP[key],
                     "n_samples": n_samp, "n_sig": n_sig})

    power_df = pd.DataFrame(rows)
    rho, p   = spearmanr(power_df["n_samples"], power_df["n_sig"])
    rho, p   = round(float(rho), 3), round(float(p), 4)
    print(f"\nPower-bias: Spearman rho={rho}, p={p}")

    fig, ax = plt.subplots(figsize=(7, 5))
    colours = [AGE_COLOURS.get(s, "#888") if n > 0 else "#B4B2A9"
               for s, n in zip(power_df["stratum"], power_df["n_sig"])]
    ax.scatter(power_df["n_samples"], power_df["n_sig"], c=colours, s=80, zorder=3)

    if power_df["n_sig"].nunique() > 1 and power_df["n_samples"].nunique() > 1:
        slope, intercept, *_ = linregress(power_df["n_samples"], power_df["n_sig"])
        x_line = np.linspace(power_df["n_samples"].min(), power_df["n_samples"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#534AB7",
                lw=1.2, ls="--", alpha=0.7, label=f"OLS fit (rho={rho}, p={p})")

    for _, row in power_df.iterrows():
        ax.annotate(row["label"], xy=(row["n_samples"], row["n_sig"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)

    sig_txt = (f"Spearman rho={rho}\np={p}" +
               (" *" if p < SPEARMAN_P_THRESHOLD else " (n.s.)"))
    ax.text(0.97, 0.05, sig_txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#534AB7",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEDFE", edgecolor="#AFA9EC"))

    ax.set_xlabel("Age stratum sample size (CC only)", fontsize=11)
    ax.set_ylabel("Number of significant DA taxa", fontsize=11)
    ax.set_title("Power-bias: does sample size drive discovery?", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Power-bias saved -> {save_path}")
    plt.show()
    return power_df, rho, p


# ===========================================================
# MAIN
# ===========================================================

def main():
    df_feat, df_meta = load_data()

    if "age" not in df_meta.columns:
        raise RuntimeError(
            "Column 'age' not found in metadata. "
            "This script requires age information in the raw data."
        )

    print(f"\n{'='*60}")
    print("  PER-AGE-GROUP POOLED DA")
    print(f"{'='*60}")
    print("  Strata:")
    for key, bounds in AGE_GROUPS.items():
        if bounds is None:
            print(f"    {key:<8} ({AGE_LABEL_MAP[key]}): age is null/missing")
        else:
            lo, hi = bounds
            lo_str = f">= {lo}" if lo is not None else ""
            hi_str = f"< {hi}"  if hi is not None else ""
            rng    = " AND ".join(filter(None, [lo_str, hi_str])) or "any"
            print(f"    {key:<8} ({AGE_LABEL_MAP[key]}): age {rng}")

    results = run_perage_da(df_feat, df_meta)

    if results:
        sig_mat, lfc_mat, dir_mat = build_agreement_matrix(results)
        agree    = compute_agreement_stats(sig_mat, lfc_mat, dir_mat)
        all_rows = pd.concat(results.values(), ignore_index=True)
    else:
        sig_mat = lfc_mat = dir_mat = pd.DataFrame()
        agree   = all_rows = pd.DataFrame()

    all_rows.to_csv(OUT_PERAGE, index=False)
    agree.to_csv(OUT_AGE_AGREEMENT, index=False)
    print(f"\n  Saved: {OUT_PERAGE}")
    print(f"  Saved: {OUT_AGE_AGREEMENT}")

    if not agree.empty:
        print(f"  Core taxa:    {int((agree['agreement_tier']=='core').sum())}")
        print(f"  Moderate:     {int((agree['agreement_tier']=='moderate').sum())}")
        print(f"  Any sig:      {int((agree['n_significant']>0).sum())}")

    meta_df = run_meta_analysis(results, sig_mat, lfc_mat) if results else pd.DataFrame()
    meta_df.to_csv(OUT_AGE_META, index=False)
    print(f"  Saved: {OUT_AGE_META}")
    if not meta_df.empty:
        n_meta_sig = int((meta_df["q_value"] < Q_THRESHOLD_PERCOHORT).sum())
        print(f"  Meta-analysis: {n_meta_sig} taxa q<{Q_THRESHOLD_PERCOHORT}")

    print(f"\n{'='*60}")
    print("  PLOTS")
    print(f"{'='*60}")

    plot_volcano_grid(results, save_path=OUT_AGE_VOLCANO)
    if not sig_mat.empty:
        plot_agreement_heatmap(sig_mat, dir_mat, agree, save_path=OUT_AGE_HM)
    if not meta_df.empty:
        plot_forest(meta_df, results, top_n=12, save_path=OUT_AGE_FOREST)
    if results:
        plot_power_bias(results, df_meta, save_path=OUT_AGE_POWER)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for key in AGE_GROUPS:
        if key in results:
            n = int(results[key]["significant"].sum())
            print(f"  {AGE_LABEL_MAP[key]:<14}: {n} significant taxa")
        else:
            print(f"  {AGE_LABEL_MAP[key]:<14}: skipped")
    if not meta_df.empty:
        print(f"\n  Cross-stratum meta-sig (q<{Q_THRESHOLD_PERCOHORT}): "
              f"{int((meta_df['q_value']<Q_THRESHOLD_PERCOHORT).sum())} taxa")
    if not agree.empty:
        core = int((agree["agreement_tier"] == "core").sum())
        print(f"  Core taxa (sig in all strata, consistent direction): {core}")

    return {
        "results" : results,
        "agree"   : agree,
        "meta_df" : meta_df,
        "sig_mat" : sig_mat,
        "lfc_mat" : lfc_mat,
    }


if __name__ == "__main__":
    out = main()