"""
da_percountry.py
================
Per-country-group differential abundance analysis (CC: control vs CRC).

Structure mirrors da_percohort.py but the unit of iteration is a
country group rather than a dataset.  All shared config, wrappers,
and statistical helpers come from da_shared.py.

Run independently:
    python da_percountry.py

Outputs (all under ./results/):
    da_percountry_<level>.csv           DA results, all groups
    da_country_agreement_<level>.csv    cross-group agreement stats
    da_country_meta_<level>.csv         DerSimonian-Laird meta-analysis
    da_country_agreement_heatmap_<level>.png
    da_country_forest_<level>.png
    da_country_power_bias_<level>.png
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
    TAXON_LEVEL, TIER_COLOURS,
    Q_THRESHOLD_PERCOHORT, MIN_GROUP_SIZE, SPEARMAN_P_THRESHOLD,
    # data + helpers
    load_data, get_prevalent_taxa,
    clr_features, renormalize_features,
    compute_auroc, generalized_fold_change_quantiles,
    standard_wilcoxon_p, _short,
)


# ===========================================================
# COUNTRY GROUP CONFIGURATION
# ===========================================================

# Keys become the stratum labels used in all outputs.
# 'WEST' pools several countries; the others are single-country strata.
COUNTRY_GROUPS = {
    "USA_CAN": ["USA", "CAN"],
    "FRA"    : ["FRA"],
    "ITA"    : ["ITA"],
    "JPN"    : ["JPN"],
    "AUT"    : ["AUT"],
    "CHN"    : ["CHN"],
    "DEU"    : ["DEU"],
}

GROUP_LABEL_MAP = {
    "USA_CAN": "USA / Canada",
    "FRA"    : "France (FRA)",
    "ITA"    : "Italy (ITA)",
    "JPN"    : "Japan (JPN)",
    "AUT"    : "Austria (AUT)",
    "CHN"    : "China (CHN)",
    "DEU"    : "Germany (DEU)",
}

def _tag():
    return f"_{TAXON_LEVEL}"

# Output paths — separate from da_percohort outputs so nothing is overwritten
OUT_PERCOUNTRY    = f"./results/da_percountry{_tag()}.csv"
OUT_CTR_AGREEMENT = f"./results/da_country_agreement{_tag()}.csv"
OUT_CTR_META      = f"./results/da_country_meta{_tag()}.csv"
OUT_CTR_HM        = f"./results/da_country_agreement_heatmap{_tag()}.png"
OUT_CTR_FOREST    = f"./results/da_country_forest{_tag()}.png"
OUT_CTR_POWER     = f"./results/da_country_power_bias{_tag()}.png"


# ===========================================================
# HELPERS
# ===========================================================

def _get_group_mask(df_meta, countries: list[str]) -> pd.Series:
    """Boolean mask selecting rows whose country is in *countries*."""
    if "country" not in df_meta.columns:
        raise KeyError(
            "Column 'country' not found in df_meta. "
            "Ensure the raw data contains a country field."
        )
    return df_meta["country"].str.upper().isin([c.upper() for c in countries])


def _summarise_groups(df_meta):
    """Print sample counts per country group for a quick sanity check."""
    print("\nSample counts per country group:")
    for gname, countries in COUNTRY_GROUPS.items():
        mask  = _get_group_mask(df_meta, countries)
        sub   = df_meta.loc[mask, "condition"]
        n_c   = int((sub == "control").sum())
        n_r   = int((sub == "CRC").sum())
        found = sorted(df_meta.loc[mask, "country"].str.upper().unique())
        print(f"  {GROUP_LABEL_MAP[gname]:<30} | ctrl={n_c}  CRC={n_r}  "
              f"countries found={found}")


# ===========================================================
# PER-COUNTRY-GROUP DA
# ===========================================================

def _da_one_group(clr_sub, orig_sub, labels, group_a="control", group_b="CRC"):
    """Run DA for a single country group — identical logic to da_percohort."""
    labels  = labels.values if isinstance(labels, pd.Series) else np.asarray(labels)
    mask_a  = labels == group_a
    mask_b  = labels == group_b
    if mask_a.sum() < MIN_GROUP_SIZE or mask_b.sum() < MIN_GROUP_SIZE:
        return pd.DataFrame()

    y_binary = (labels == group_b).astype(float)
    records  = []

    for taxon in clr_sub.columns:
        x = clr_sub[taxon].values
        try:
            model  = sm.OLS(x, sm.add_constant(y_binary)).fit()
            coef   = float(model.params[1])
            se     = float(model.bse[1])
            t_stat = float(model.tvalues[1])
            p_ols  = float(model.pvalues[1])
        except Exception:
            coef = se = t_stat = p_ols = np.nan

        p_wilc = standard_wilcoxon_p(x, labels, group_a, group_b)
        auroc  = compute_auroc(x, labels, case_label=group_b)
        gfc    = generalized_fold_change_quantiles(
            orig_sub[taxon].values, labels, group_a, group_b
        )
        records.append({
            "taxon"     : taxon,
            "lfc"       : coef,   "se"      : se,
            "t_stat"    : t_stat, "p_ols"   : p_ols,
            "p_val"     : p_wilc, "wilcoxon_p": p_wilc,
            "auroc"     : auroc,  "gfc"     : gfc,
            "n_ctrl"    : int(mask_a.sum()),
            "n_crc"     : int(mask_b.sum()),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    _, q, _, _ = multipletests(df["p_val"].fillna(1), method="fdr_bh")
    df["q_val"]       = q
    df["significant"] = df["q_val"] < Q_THRESHOLD_PERCOHORT
    df["direction"]   = np.where(df["lfc"] > 0, "enriched_CRC", "depleted_CRC")
    return df.sort_values("q_val").reset_index(drop=True)


def run_percountry_da(df_feat, df_meta):
    """
    Iterate over COUNTRY_GROUPS; for each group subset the samples,
    re-normalise, CLR-transform, and run Wilcoxon + OLS DA.

    Uses the same intersection taxa as da_percohort.py (via get_prevalent_taxa)
    so results are directly comparable across the two analyses.
    """
    intersection_taxa = get_prevalent_taxa(df_feat, df_meta)
    if not intersection_taxa:
        print("  ERROR: No taxa passed intersection prevalence filter.")
        return {}

    print(f"\n  Per-country DA testing {len(intersection_taxa)} taxa "
          f"(intersection, same as per-cohort analysis)")

    _summarise_groups(df_meta)

    results = {}

    for gname, countries in COUNTRY_GROUPS.items():
        mask_grp = _get_group_mask(df_meta, countries)
        cond_grp = df_meta.loc[mask_grp, "condition"]

        # Keep only CC samples
        mask_cc   = mask_grp & df_meta["condition"].isin(["control", "CRC"])
        labels_cc = df_meta.loc[mask_cc, "condition"]

        n_ctrl = int((labels_cc == "control").sum())
        n_crc  = int((labels_cc == "CRC").sum())

        print(f"\n  [{gname}] {GROUP_LABEL_MAP[gname]}")
        print(f"    ctrl={n_ctrl}  CRC={n_crc}")

        if labels_cc.nunique() < 2:
            print("    -> skipped (only one class present)")
            continue
        if n_ctrl < MIN_GROUP_SIZE or n_crc < MIN_GROUP_SIZE:
            print(f"    -> skipped (n < MIN_GROUP_SIZE={MIN_GROUP_SIZE})")
            continue

        orig_cc   = df_feat.loc[mask_cc, intersection_taxa].copy()
        orig_cc   = renormalize_features(orig_cc)
        valid_rows = orig_cc.notna().all(axis=1)
        orig_cc   = orig_cc.loc[valid_rows]
        labels_cc = labels_cc.loc[valid_rows]

        if orig_cc.empty:
            print("    -> empty after re-normalisation")
            continue

        clr_cc = clr_features(orig_cc)
        res    = _da_one_group(clr_cc, orig_cc, labels_cc)

        if not res.empty:
            res["country_group"] = gname
            res["countries"]     = ", ".join(countries)
            results[gname]       = res
            print(f"    -> {len(res)} tested, "
                  f"{int(res['significant'].sum())} significant "
                  f"(q<{Q_THRESHOLD_PERCOHORT})")

    return results


# ===========================================================
# AGREEMENT MATRIX + STATS
# ===========================================================

def build_agreement_matrix(results):
    groups   = list(results.keys())
    all_taxa = sorted({t for df in results.values() for t in df["taxon"]})

    sig  = pd.DataFrame(np.nan, index=all_taxa, columns=groups)
    lfc  = pd.DataFrame(np.nan, index=all_taxa, columns=groups)
    dir_ = pd.DataFrame(np.nan, index=all_taxa, columns=groups)

    for g in groups:
        df     = results[g].set_index("taxon")
        shared = df.index.intersection(all_taxa)
        sig.loc[shared, g]  = df.loc[shared, "significant"].astype(float)
        lfc.loc[shared, g]  = df.loc[shared, "lfc"]
        dir_.loc[shared, g] = np.where(df.loc[shared, "lfc"] > 0, 1.0, -1.0)

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
        for g in results:
            row = results[g][results[g]["taxon"] == taxon]
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
            meta_records.append({"taxon": taxon, "n_groups": len(effs),
                                  "n_sig_groups": n_sig, **ma})

    if not meta_records:
        return pd.DataFrame()

    meta_df = pd.DataFrame(meta_records)
    _, q, _, _ = multipletests(meta_df["p_value"].fillna(1), method="fdr_bh")
    meta_df["q_value"] = np.round(q, 6)
    return meta_df.sort_values("q_value").reset_index(drop=True)


# ===========================================================
# PLOTS
# ===========================================================

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
    n_taxa, n_grp = mat.shape
    cmap          = ListedColormap(["#D3D1C7", "#5DCAA5", "#FFFFFF", "#F0997B"])
    norm          = BoundaryNorm([-999.5, -1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(max(7, n_grp * 1.4), max(10, n_taxa * 0.25)))
    ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm, interpolation="none")

    tier_lookup = agree_stats.set_index("taxon")["agreement_tier"]
    for i, taxon in enumerate(top_taxa):
        colour = TIER_COLOURS.get(tier_lookup.get(taxon, "not_detected"), "#444")
        ax.text(-0.05, i, _short(taxon), ha="right", va="center",
                fontsize=7, color=colour, transform=ax.get_yaxis_transform())

    # x-axis: country group labels
    ax.set_xticks(range(n_grp))
    ax.set_xticklabels(
        [GROUP_LABEL_MAP.get(c, c) for c in mat.columns],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_yticks([])
    ax.set_title(
        f"Per-country-group DA agreement — {TAXON_LEVEL} level\n"
        f"Taxa with >=1 significant group call (q<{Q_THRESHOLD_PERCOHORT})",
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
    groups   = list(results.keys())

    for idx, taxon in enumerate(top_taxa):
        ax = axes[idx]
        lfcs, ses, labels, sigs = [], [], [], []

        for g in groups:
            row = results[g][results[g]["taxon"] == taxon]
            if row.empty:
                continue
            lfcs.append(float(row["lfc"].iloc[0]))
            ses.append(float(row["se"].iloc[0]))
            labels.append(GROUP_LABEL_MAP.get(g, g))
            sigs.append(bool(row["significant"].iloc[0]))

        for j, (lfc, se, sig) in enumerate(zip(lfcs, ses, sigs)):
            c = "#1D9E75" if sig else "#B4B2A9"
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

    plt.suptitle(f"Forest plots — top DA taxa by country group ({TAXON_LEVEL})",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Forest plots saved -> {save_path}")
    plt.show()


def plot_power_bias(results, df_meta, save_path=None):
    rows = []
    for g, countries in COUNTRY_GROUPS.items():
        if g not in results:
            n_sig = 0
        else:
            n_sig = int(results[g]["significant"].sum())

        mask_cc = (
            _get_group_mask(df_meta, countries)
            & df_meta["condition"].isin(["control", "CRC"])
        )
        n_samp = int(mask_cc.sum())
        rows.append({"group": g, "label": GROUP_LABEL_MAP[g],
                     "n_samples": n_samp, "n_sig": n_sig})

    power_df = pd.DataFrame(rows)
    rho, p   = spearmanr(power_df["n_samples"], power_df["n_sig"])
    rho, p   = round(float(rho), 3), round(float(p), 4)
    print(f"\nPower-bias: Spearman rho={rho}, p={p}")

    fig, ax = plt.subplots(figsize=(7, 5))
    colours = ["#1D9E75" if n > 0 else "#B4B2A9" for n in power_df["n_sig"]]
    ax.scatter(power_df["n_samples"], power_df["n_sig"], c=colours, s=70, zorder=3)

    if power_df["n_sig"].nunique() > 1 and power_df["n_samples"].nunique() > 1:
        slope, intercept, *_ = linregress(power_df["n_samples"], power_df["n_sig"])
        x_line = np.linspace(power_df["n_samples"].min(), power_df["n_samples"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#534AB7",
                lw=1.2, ls="--", alpha=0.7, label=f"OLS fit (rho={rho}, p={p})")

    for _, row in power_df.iterrows():
        ax.annotate(row["label"], xy=(row["n_samples"], row["n_sig"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=7)

    sig_txt = (f"Spearman rho={rho}\np={p}" +
               (" *" if p < SPEARMAN_P_THRESHOLD else " (n.s.)"))
    ax.text(0.97, 0.05, sig_txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#534AB7",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEDFE", edgecolor="#AFA9EC"))

    ax.set_xlabel("Country group sample size (CC only)", fontsize=11)
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

    # Validate that 'country' column is present
    if "country" not in df_meta.columns:
        raise RuntimeError(
            "Column 'country' not found in metadata. "
            "This script requires country information in the raw data."
        )

    print(f"\n{'='*60}")
    print("  PER-COUNTRY-GROUP DA")
    print(f"{'='*60}")
    print(f"  Groups defined:")
    for g, countries in COUNTRY_GROUPS.items():
        print(f"    {g:<6} -> {', '.join(countries)}")

    results = run_percountry_da(df_feat, df_meta)

    if results:
        sig_mat, lfc_mat, dir_mat = build_agreement_matrix(results)
        agree    = compute_agreement_stats(sig_mat, lfc_mat, dir_mat)
        all_rows = pd.concat(results.values(), ignore_index=True)
    else:
        sig_mat = lfc_mat = dir_mat = pd.DataFrame()
        agree   = all_rows = pd.DataFrame()

    all_rows.to_csv(OUT_PERCOUNTRY, index=False)
    agree.to_csv(OUT_CTR_AGREEMENT, index=False)
    print(f"\n  Saved: {OUT_PERCOUNTRY}")
    print(f"  Saved: {OUT_CTR_AGREEMENT}")

    if not agree.empty:
        print(f"\n  Core taxa:    {int((agree['agreement_tier']=='core').sum())}")
        print(f"  Moderate:     {int((agree['agreement_tier']=='moderate').sum())}")
        print(f"  Any sig:      {int((agree['n_significant']>0).sum())}")

    meta_df = run_meta_analysis(results, sig_mat, lfc_mat) if results else pd.DataFrame()
    meta_df.to_csv(OUT_CTR_META, index=False)
    print(f"  Saved: {OUT_CTR_META}")
    if not meta_df.empty:
        n_meta_sig = int((meta_df["q_value"] < Q_THRESHOLD_PERCOHORT).sum())
        print(f"  Meta-analysis: {n_meta_sig} taxa q<{Q_THRESHOLD_PERCOHORT}")

    print(f"\n{'='*60}")
    print("  PLOTS")
    print(f"{'='*60}")

    if not sig_mat.empty:
        plot_agreement_heatmap(sig_mat, dir_mat, agree, save_path=OUT_CTR_HM)
    if not meta_df.empty:
        plot_forest(meta_df, results, top_n=12, save_path=OUT_CTR_FOREST)
    if results:
        plot_power_bias(results, df_meta, save_path=OUT_CTR_POWER)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Groups analysed:      {len(results)}/{len(COUNTRY_GROUPS)}")
    if not agree.empty:
        print(f"  Any sig taxa:         {int((agree['n_significant']>0).sum())}")
        print(f"  Core taxa (>=67% groups, consistent dir): "
              f"{int((agree['agreement_tier']=='core').sum())}")
    if not meta_df.empty:
        print(f"  Meta-sig taxa (q<{Q_THRESHOLD_PERCOHORT}): "
              f"{int((meta_df['q_value']<Q_THRESHOLD_PERCOHORT).sum())}")

    return {
        "results" : results,
        "agree"   : agree,
        "meta_df" : meta_df,
        "sig_mat" : sig_mat,
        "lfc_mat" : lfc_mat,
    }


if __name__ == "__main__":
    out = main()