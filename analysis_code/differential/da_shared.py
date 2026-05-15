"""
da_shared.py
============
Shared configuration, library wrappers, and statistical helpers used by
both da_percohort.py and da_pooled.py.

Library functions used (no local re-implementations):
    pd_utils   — aggregate_taxa, relative_abundance, prevalence_filtering,
                 _all_meta, register_metadata_col
    pd_transform — clr_transform, check_data_quality
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(str(Path(__file__).parent / ".."))

from pd_utils import (
    relative_abundance,
    prevalence_filtering,
    _all_meta,
    register_metadata_col,
)
from pd_transform import clr_transform as _library_clr, check_data_quality


# ===========================================================
# CONFIGURATION
# ===========================================================

TAXON_LEVEL = "genus"   # "species" or "genus"


def _tag():
    return f"_{TAXON_LEVEL}"


def _prepared_dir():
    """Return the prepared-data directory for the active taxon level."""
    return f"./data/prepared/{TAXON_LEVEL}/"


os.makedirs("./results", exist_ok=True)

# Output paths — referenced by both scripts
OUT_POOLED       = f"./results/da_pooled{_tag()}.csv"
OUT_PERCOHORT    = f"./results/da_percohort{_tag()}.csv"
OUT_AGREEMENT    = f"./results/da_agreement{_tag()}.csv"
OUT_META         = f"./results/da_meta{_tag()}.csv"
OUT_INTERSECTION = f"./results/da_intersection{_tag()}.csv"
OUT_GRID         = f"./results/da_validation_grid{_tag()}.csv"
OUT_HM_POOLED    = f"./results/da_validation_grid{_tag()}.png"
OUT_HM_PERCOHORT = f"./results/da_agreement_heatmap{_tag()}.png"
OUT_VOLCANO      = f"./results/da_volcano{_tag()}.png"
OUT_FOREST       = f"./results/da_forest{_tag()}.png"
OUT_POWER        = f"./results/da_power_bias{_tag()}.png"
OUT_QUALITY      = f"./results/da_data_quality_report{_tag()}.csv"

METADATA_COLS = [
    "dataset_name",
    "sampleID",
    "subjectID",
    "study_condition",
    "disease",
    "age",
    "gender",
    "country",
    "BMI",
    "fobt",
]

DATASET_LABEL_MAP = {
    "FengQ_2015"      : "Feng et al. (2015)",
    "HanniganGD_2017" : "Hannigan et al. (2018)",
    "ThomasAM_2019_a" : "Thomas et al. (2019)(a)",
    "ThomasAM_2019_b" : "Thomas et al. (2019)(b)",
    "ThomasAM_2019_c" : "Thomas et al. (2019)(c)",
    "VogtmannE_2016"  : "Vogtmann et al. (2016)",
    "WirbelJ_2018"    : "Wirbel et al. (2019)",
    "YuJ_2015"        : "Yu et al. (2017)",
    "ZellerG_2014"    : "Zeller et al. (2014)",
}

Q_THRESHOLD_POOLED    = 0.05
Q_THRESHOLD_PERCOHORT = 0.25
MIN_EFFECT_CLR        = 0.10
GRID_P_THRESHOLD      = 0.05
MIN_PREVALENCE        = 0.10
MIN_GROUP_SIZE        = 5
SPEARMAN_P_THRESHOLD  = 0.05
PREVALENCE_MODE       = "intersection"   # "union" or "intersection"
N_PERMUTATIONS        = 999
RANDOM_SEED           = 42

TIER_COLOURS = {
    "core"        : "#1D9E75",
    "moderate"    : "#378ADD",
    "weak"        : "#EF9F27",
    "inconsistent": "#D85A30",
    "not_detected": "#B4B2A9",
}


# ===========================================================
# DATA LOADING
# ===========================================================

def load_data():
    """
    Load, quality-check, filter, and split into (df_feat, df_meta) on the
    80 % train set — identical preparation to the ML pipeline.
    """
    prepared_dir = _prepared_dir()
    print("Loading prepared data...")
    X_train_df     = pd.read_csv(os.path.join(prepared_dir, "X_train_full.csv"))
    y_train        = pd.read_csv(os.path.join(prepared_dir, "y_train_full.csv")).squeeze()
    surviving_taxa = (
        pd.read_csv(os.path.join(prepared_dir, "all_taxa_full.csv"), header=None)
        .squeeze()
        .tolist()
    )

    print(f"  X_train : {X_train_df.shape}")
    print(f"  y_train : {len(y_train)}  "
          f"(CRC: {int(y_train.sum())}, Control: {int((y_train == 0).sum())})")
    print(f"  Taxa    : {len(surviving_taxa)} {TAXON_LEVEL}s")

    # Reconstruct metadata from prepared file
    meta_cols_present = [c for c in METADATA_COLS if c in X_train_df.columns]

    # If the prepared file still carries metadata columns, pull them; otherwise
    # rebuild study_condition from the binary label vector.
    if "study_condition" in X_train_df.columns:
        df_meta = X_train_df[meta_cols_present].copy().reset_index(drop=True)
    else:
        df_meta = pd.DataFrame(index=X_train_df.index)
        if "dataset_name" in X_train_df.columns:
            df_meta["dataset_name"] = X_train_df["dataset_name"].values
        if "sampleID" in X_train_df.columns:
            df_meta["sampleID"] = X_train_df["sampleID"].values
        # Map binary label → condition string expected by downstream code
        df_meta["study_condition"] = y_train.map({1: "CRC", 0: "control"}).values

    df_meta = df_meta.reset_index(drop=True)

    # Use only the surviving taxa so the feature set matches the ML pipeline
    available_taxa = [c for c in surviving_taxa if c in X_train_df.columns]
    df_feat = (
        X_train_df[available_taxa]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .reset_index(drop=True)
    )

    print(f"  After condition filter: {len(df_meta)} samples")
    print(f"  CRC: {int(y_train.sum())}  Control: {int((y_train == 0).sum())}")

    # Map study_condition → condition label expected by downstream code
    df_meta["condition"] = df_meta["study_condition"].map({
        "CRC": "CRC", "crc": "CRC", "Crc": "CRC",
        "control": "control", "Control": "control", "CONTROL": "control",
    })
    register_metadata_col("condition")

    df_meta = df_meta[df_meta["condition"].notna()].reset_index(drop=True)
    df_feat = df_feat.loc[df_meta.index].reset_index(drop=True)

    print(f"\n  X_train: {df_feat.shape[0]} samples x {df_feat.shape[1]} features")
    print(f"  Conditions: {df_meta['condition'].value_counts().to_dict()}")

    # Data quality report
    combined       = make_combined_df(df_feat, df_meta)
    quality_report = check_data_quality(combined)
    quality_report.to_csv(OUT_QUALITY, index=False)
    print(f"\nData quality report:\n{quality_report}")
    print(f"  Saved -> {OUT_QUALITY}")

    return df_feat, df_meta


# ===========================================================
# LIBRARY WRAPPERS
# ===========================================================

def make_combined_df(df_feat, df_meta):
    """Combine metadata + features so library functions can use _all_meta()."""
    register_metadata_col("condition")
    return pd.concat(
        [df_meta.reset_index(drop=True), df_feat.reset_index(drop=True)],
        axis=1,
    )


def clr_features(df_feat):
    """Apply library CLR transform to a features-only DataFrame."""
    feature_cols = list(df_feat.columns)
    out = _library_clr(df_feat.copy(), feature_cols=feature_cols, pseudo_count=1e-9)
    return out[feature_cols]


def renormalize_features(df_feat):
    """
    Normalise rows to relative abundance (sum=1) via library relative_abundance(),
    then scale back to percentage (sum=100) to match the rest of the pipeline.
    """
    feature_cols = list(df_feat.columns)
    out = relative_abundance(df_feat.copy(), feature_cols=feature_cols)
    out[feature_cols] = out[feature_cols] * 100.0
    return out


# ===========================================================
# PREVALENCE FILTER
# ===========================================================

def get_prevalent_taxa(df_feat, df_meta):
    """
    Per-dataset prevalence filter via library prevalence_filtering().
    Returns a sorted list of taxon names that pass the threshold.
    """
    register_metadata_col("condition")
    combined = make_combined_df(df_feat, df_meta)
    how      = "intersect" if PREVALENCE_MODE == "intersection" else "union"

    print(f"\nPer-dataset prevalence (threshold={MIN_PREVALENCE}, mode={PREVALENCE_MODE}):")
    feature_cols = [c for c in combined.columns if c not in _all_meta()]
    for ds in sorted(combined["dataset_name"].unique()):
        sub    = combined[combined["dataset_name"] == ds]
        prev   = (sub[feature_cols] > 0).mean(axis=0)
        n_pass = int((prev >= MIN_PREVALENCE).sum())
        print(f"  {DATASET_LABEL_MAP.get(ds, ds):<18}: {n_pass} taxa")

    filtered = prevalence_filtering(combined, threshold=MIN_PREVALENCE, how=how)
    kept     = [c for c in filtered.columns if c not in _all_meta()]
    print(f"  After {PREVALENCE_MODE}: {len(kept)} taxa retained")
    return sorted(kept)


# ===========================================================
# EFFECT SIZE HELPERS
# ===========================================================

def compute_auroc(values, labels, case_label="CRC"):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    y_true = (labels == case_label).astype(int)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, values))
    except Exception:
        return np.nan


def generalized_fold_change_quantiles(
    values, labels, group_a="control", group_b="CRC",
    pseudo_count=1e-9, quantiles=np.arange(0.1, 1.0, 0.1),
):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    ctrl   = values[labels == group_a] / 100.0 + pseudo_count
    case   = values[labels == group_b] / 100.0 + pseudo_count
    if len(ctrl) == 0 or len(case) == 0:
        return np.nan
    return float(np.mean(
        np.quantile(np.log2(case), quantiles) - np.quantile(np.log2(ctrl), quantiles)
    ))


def standard_wilcoxon_p(values, labels, group_a="control", group_b="CRC"):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    ctrl   = values[labels == group_a]
    case   = values[labels == group_b]
    if len(ctrl) == 0 or len(case) == 0:
        return np.nan
    try:
        _, p = mannwhitneyu(ctrl, case, alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan


# ===========================================================
# BLOCKED PERMUTATION TEST
# ===========================================================

def _mwu_centered_stat(values, is_case):
    case_vals = values[is_case]
    ctrl_vals = values[~is_case]
    if len(case_vals) == 0 or len(ctrl_vals) == 0:
        return np.nan
    u = mannwhitneyu(case_vals, ctrl_vals, alternative="two-sided").statistic
    return float(u - len(case_vals) * len(ctrl_vals) / 2.0)


def blocked_wilcoxon_permutation_p(
    values, labels, blocks=None,
    group_a="control", group_b="CRC",
    n_perm=N_PERMUTATIONS, random_state=RANDOM_SEED,
):
    rng    = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    valid  = np.isfinite(values) & np.isin(labels, [group_a, group_b])
    values, labels = values[valid], labels[valid]
    blocks = (np.repeat("single_block", len(values))
              if blocks is None else np.asarray(blocks)[valid])
    is_case = labels == group_b
    if is_case.sum() == 0 or (~is_case).sum() == 0:
        return np.nan
    observed = _mwu_centered_stat(values, is_case)
    if np.isnan(observed):
        return np.nan
    perm_stats = []
    for _ in range(n_perm):
        perm_case = is_case.copy()
        for block in np.unique(blocks):
            idx = blocks == block
            perm_case[idx] = rng.permutation(perm_case[idx])
        s = _mwu_centered_stat(values, perm_case)
        if not np.isnan(s):
            perm_stats.append(s)
    if not perm_stats:
        return np.nan
    perm_stats = np.asarray(perm_stats)
    return float((np.sum(np.abs(perm_stats) >= abs(observed)) + 1) /
                 (len(perm_stats) + 1))


# ===========================================================
# MISC
# ===========================================================

def _short(taxon, n=45):
    return (
        str(taxon).split("|")[-1]
        .replace("s__", "").replace("g__", "").replace("_", " ")[:n]
    )