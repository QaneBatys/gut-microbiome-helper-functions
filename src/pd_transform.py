"""
Utility functions for gut microbiome data preprocessing: transformations, batch correction, and data quality checks.
Main Functions:
    log_transform( df: pd.DataFrame, pseudocount:float = 1e-9, feature_cols: list | None = None) -> pd.DataFrame: return a copy of pd where log10 transformation is applied to feature columns.
    clr_transform( df: pd.DataFrame, feature_cols: list | None = None, pseudo_count: float = 1e-9 ) -> pd.DataFrame: return a copy of pd where central log transformation is applied to feature columns.
    remove_batch_effects( df: pd.DataFrame, batch_col: str, feature_cols: list | None = None, method: str = "combat", ) -> pd.DataFrame: apply inmoose batch effect removal to feature and return copy of pd.
    check_data_quality(df: pd.DataFrame) -> pd.DataFrame: check if the dataset has quality data.
"""

import warnings
import pandas as pd
import numpy as np
from inmoose.pycombat import pycombat_norm

from pd_utils import _all_meta, register_metadata_col


# 3.1
def log_transform(
    df: pd.DataFrame, pseudo_count: float = 1e-9, feature_cols: list | None = None
) -> pd.DataFrame:
    df_out = df.copy()
    cols = (
        feature_cols
        if feature_cols is not None
        else [c for c in df.columns if c not in _all_meta()]
    )
    df_out[cols] = np.log10(df_out[cols].astype(float) + pseudo_count)  # ← was df[cols]
    return df_out


# 3.2
def clr_transform(
    df: pd.DataFrame, feature_cols: list | None = None, pseudo_count: float = 1e-9
) -> pd.DataFrame:
    """
    Apply CLR (Centered Log-Ratio) transformation to feature columns.

    Each value becomes:
        log(x / geometric_mean(sample))

    Parameters:
    df : pd.DataFrame
        Input dataframe (samples x features + metadata)
    feature_cols : list | None
        Columns to transform. If None, inferred automatically.
    pseudo_count : float
        Small constant added to avoid log(0)

    Returns:
    pd.DataFrame
        Copy of df with CLR-transformed feature columns
    """

    df_out = df.copy()

    # infer feature columns using global metadata config
    if feature_cols is None:
        meta_cols = _all_meta()
        feature_cols = [col for col in df.columns if col not in meta_cols]

    # convert to numeric
    # handle zeros
    X = df_out[feature_cols].astype(float) + pseudo_count

    # compute geometric mean per sample (row-wise)
    geometric_mean = np.exp(np.mean(np.log(X), axis=1))

    # apply CLR
    X_clr = np.log(X.div(geometric_mean, axis=0))

    # assign back
    df_out[feature_cols] = X_clr

    return df_out


# 3.3
def remove_batch_effects(
    df: pd.DataFrame,
    batch_col: str,
    feature_cols: list | None = None,
    method: str = "combat",
) -> pd.DataFrame:
    """
    Correct batch effects in feature columns.
    """

    if batch_col not in df.columns:
        raise ValueError(f"Missing batch column: {batch_col}")

    if feature_cols is None:
        meta_cols = _all_meta() | {batch_col}
        feature_cols = [col for col in df.columns if col not in meta_cols]

    method = method.lower()
    if method == "limma":
        raise NotImplementedError(
            "method='limma' is not implemented here. "
            "This function currently supports only method='combat'."
        )
    if method != "combat":
        raise ValueError("method must be 'combat' or 'limma'")

    df_out = df.copy()

    # features → (features x samples)
    X = df_out[feature_cols].astype(float).T

    # batch labels (per sample)
    batches = df_out[batch_col].astype(str).str.strip().tolist()

    # apply ComBat
    X_corrected = pycombat_norm(X, batches)

    # ensure DataFrame
    if not isinstance(X_corrected, pd.DataFrame):
        X_corrected = pd.DataFrame(
            X_corrected,
            index=X.index,
            columns=X.columns,
        )

    # back to original shape
    df_out.loc[:, feature_cols] = X_corrected.T.values

    return df_out


# 3.4
def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit the input dataframe and summarize common data quality issues.

    Returns
    -------
    pd.DataFrame
        Report with columns: ['check', 'status', 'affected', 'details']
    """
    rows = []

    feature_cols = [col for col in df.columns if col not in _all_meta()]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    missing_per_col = df.isna().sum()
    missing_cols = missing_per_col[missing_per_col > 0]

    rows.append(
        {
            "check": "missing_values",
            "status": "fail" if len(missing_cols) > 0 else "pass",
            "affected": int(len(missing_cols)),
            "details": (
                missing_cols.to_dict() if len(missing_cols) > 0 else "no missing values"
            ),
        }
    )

    duplicate_mask = df.index.duplicated(keep=False)
    duplicate_ids = df.index[duplicate_mask]

    rows.append(
        {
            "check": "duplicate_sample_identifiers",
            "status": "fail" if len(duplicate_ids) > 0 else "pass",
            "affected": int(duplicate_mask.sum()),
            "details": (
                duplicate_ids.tolist()
                if len(duplicate_ids) > 0
                else "no duplicate sample identifiers"
            ),
        }
    )

    zero_count_mask = X.sum(axis=1) == 0

    rows.append(
        {
            "check": "zero_count_samples",
            "status": "fail" if zero_count_mask.any() else "pass",
            "affected": int(zero_count_mask.sum()),
            "details": (
                df.index[zero_count_mask].tolist()
                if zero_count_mask.any()
                else "no zero-count samples"
            ),
        }
    )

    negative_mask = X < 0
    negative_col_names = X.columns[negative_mask.any(axis=0)]

    rows.append(
        {
            "check": "negative_values",
            "status": "fail" if negative_mask.any().any() else "pass",
            "affected": int(negative_mask.sum().sum()),
            "details": (
                negative_col_names.tolist()
                if len(negative_col_names) > 0
                else "no negative values"
            ),
        }
    )

    zero_var_mask = X.nunique(dropna=True) <= 1
    zero_var_cols = X.columns[zero_var_mask]

    rows.append(
        {
            "check": "zero_variance_columns",
            "status": "fail" if len(zero_var_cols) > 0 else "pass",
            "affected": int(len(zero_var_cols)),
            "details": (
                zero_var_cols.tolist()
                if len(zero_var_cols) > 0
                else "no zero-variance columns"
            ),
        }
    )

    return pd.DataFrame(rows, columns=["check", "status", "affected", "details"])
