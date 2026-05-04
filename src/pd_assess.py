"""
Model assessment utilities for gut microbiome classification.
Helper functions:
    _check_binary_target(y: pd.Series) -> None: check if there are other target values;
    _check_feature_alignment(X: pd.DataFrame, model) -> None: Verify that *X* columns match the feature names the model was trained on.
    _summary_table(metrics: dict) -> str: creates a table for the logger

Main Functinos:
    assess_model(X, y, model):Evaluate a fitted estimator on (X, y) and return a comprehensive performance dictionary including confusion matrix and all standard binary classification metrics.
    plot_auroc(X, y, model, label): Plot ROC curve(s) with shaded AUC region and legend-embedded AUROC values. Supports multi-curve overlays (pass lists). When a single triple is supplied, 
        95 % CI bands are added via bootstrapping.
    plot_precision_recall(X, y, model, label): Plot Precision-Recall curve(s) annotated with AUPRC. Includes a no-skill baseline at class prevalence and 95 % CI bands via bootstrapping for single-model calls.
    stratify_by_age(df, n_groups, age_col): Divide df into n_groups quantile-based age strata and append 'age_group' (int label) and 'age_group_range' (str interval) columns.

Decision threshold: The active threshold is recorded in every returned dict under the key ``'threshold'``.
"""

import logging
import warnings


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from pd_utils import register_metadata_col


# MODULE LOGGER
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# CONSTANT THRESHOLD
DECISION_THRESHOLD: float = 0.5


# HELPER FUNCTIONS
def _check_binary_target(y: pd.Series) -> None:
    """Raise ValueError if *y* contains values outside {0, 1}."""
    unique_vals = set(y.dropna().unique())
    if not unique_vals.issubset({0, 1}):
        invalid = sorted(unique_vals - {0, 1})
        raise ValueError(
            f"y must contain only binary labels {{0, 1}}; "
            f"found unexpected values: {invalid}"
        )


def _check_feature_alignment(X: pd.DataFrame, model) -> None:
    """
    Verify that *X* columns match the feature names the model was trained on.

    Checks ``model.feature_names_in_`` (set by sklearn ≥ 1.0 when the
    estimator is fitted on a DataFrame).  If the attribute is absent the
    check is skipped with a debug-level note.

    Raises
    ------
    ValueError
        If the column sets differ (reports missing and extra columns).
    """
    if not hasattr(model, "feature_names_in_"):
        logger.debug(
            "Model does not expose 'feature_names_in_'; "
            "feature alignment check skipped."
        )
        return

    expected = set(model.feature_names_in_)
    provided = set(X.columns)

    missing = expected - provided
    extra   = provided - expected

    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing from X: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected in X: {sorted(extra)}")
        raise ValueError(
            "Feature mismatch between X and model training features — "
            + "; ".join(parts)
        )


def _summary_table(metrics: dict) -> str:
    """Build a compact human-readable summary string for the logger."""
    threshold = metrics.get("threshold", DECISION_THRESHOLD)
    lines = [
        "",
        f"  === assess_model  (threshold={threshold}) ===",
        f"  {'Metric':<22} {'Value':>10}",
        "  " + "-" * 35,
    ]
    display_keys = [
        "auroc", "auprc", "accuracy",
        "sensitivity", "specificity", "f1",
    ]
    for key in display_keys:
        val = metrics.get(key, float("nan"))
        lines.append(f"  {key:<22} {val:>10.4f}")

    cm = metrics.get("confusion_matrix")
    if cm is not None:
        lines += [
            "",
            "  Confusion matrix (rows=true, cols=pred):",
            f"    TN={cm[0,0]:>5}  FP={cm[0,1]:>5}",
            f"    FN={cm[1,0]:>5}  TP={cm[1,1]:>5}",
        ]
    lines.append("")
    return "\n".join(lines)



# 5.1 
def assess_model(
    X: pd.DataFrame,
    y: pd.Series,
    model,
) -> dict:
    """
    Evaluate a previously fitted model on (X, y).

    Parameters:
    X : pd.DataFrame: Feature matrix for evaluation.  Must match the feature space used during training (verified via ``model.feature_names_in_`` when available).
    y : pd.Series: True binary target labels (values must be in {0, 1}).
    model : fitted sklearn estimator: A model already fitted via ``train_model`` or equivalent.  Must implement ``predict_proba``.

    Returns:
    dict:
        ``'auroc'``            float — Area Under the ROC Curve
        ``'auprc'``            float — Area Under the Precision-Recall Curve
        ``'accuracy'``         float — fraction of correct predictions
        ``'sensitivity'``      float — recall / true positive rate
                                       (TP / (TP + FN))
        ``'specificity'``      float — true negative rate
                                       (TN / (TN + FP))
        ``'f1'``               float — harmonic mean of precision & recall
        ``'confusion_matrix'`` np.ndarray (2, 2) — [[TN, FP], [FN, TP]]
        ``'threshold'``        float — decision threshold used (mirrors
                                       ``DECISION_THRESHOLD``)
        ``'y_prob'``           np.ndarray — predicted probabilities for
                                       the positive class
        ``'y_pred'``           np.ndarray — hard labels at *threshold*

    Raises:
    ValueError
        If X columns do not match the model's training features.
    ValueError
        If y contains values outside {0, 1}.
    TypeError
        If *model* does not implement ``predict_proba``.
    """
    # Validation                                                           
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"model of type '{type(model).__name__}' does not implement "
            "predict_proba, which is required for probability-based metrics."
        )

    _check_binary_target(y)
    _check_feature_alignment(X, model)

    # Predictions
    threshold = DECISION_THRESHOLD
    y_prob    = model.predict_proba(X)[:, 1]
    y_pred    = (y_prob >= threshold).astype(int)

    # Metrics                                                              
    cm = confusion_matrix(y, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity  = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    else:
        # Degenerate case — only one class present in y_pred
        sensitivity  = float("nan")
        specificity  = float("nan")

    metrics = {
        "auroc":            roc_auc_score(y, y_prob),
        "auprc":            average_precision_score(y, y_prob),
        "accuracy":         accuracy_score(y, y_pred),
        "sensitivity":      sensitivity,
        "specificity":      specificity,
        "f1":               f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "threshold":        threshold,
        "y_prob":           y_prob,
        "y_pred":           y_pred,
    }

    logger.info(_summary_table(metrics))
    return metrics


# ---------------------------------------------------------------------------
# Plot styling constants  (tweak here to restyle all plots)
# ---------------------------------------------------------------------------
_PALETTE = [
    "#2563EB",  # blue
    "#DC2626",  # red
    "#16A34A",  # green
    "#D97706",  # amber
    "#7C3AED",  # violet
    "#0891B2",  # cyan
    "#DB2777",  # pink
    "#65A30D",  # lime
]
_BOOTSTRAP_N_REPLICATES: int = 1000
_BOOTSTRAP_CI_ALPHA:     float = 0.95
_FIG_SIZE = (7, 6)


# ---------------------------------------------------------------------------
# Internal bootstrap helper
# ---------------------------------------------------------------------------

def _bootstrap_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_replicates: int = _BOOTSTRAP_N_REPLICATES,
    ci_alpha: float   = _BOOTSTRAP_CI_ALPHA,
    random_state: int = 42,
) -> tuple:
    """
    Estimate a confidence band for the ROC curve via non-parametric
    bootstrapping.

    Parameters
    ----------
    y_true       : 1-D array of true binary labels.
    y_prob       : 1-D array of predicted positive-class probabilities.
    n_replicates : number of bootstrap samples.
    ci_alpha     : confidence level (e.g. 0.95 → 2.5 % / 97.5 % quantiles).
    random_state : seed for reproducibility.

    Returns
    -------
    (mean_fpr, tpr_lower, tpr_upper)
        All three are 1-D arrays of length 100, evaluated on a common
        linspace(0, 1, 100) FPR grid.
    """
    rng       = np.random.default_rng(random_state)
    n         = len(y_true)
    fpr_grid  = np.linspace(0, 1, 100)
    tpr_boot  = np.zeros((n_replicates, len(fpr_grid)))

    for i in range(n_replicates):
        idx    = rng.integers(0, n, size=n)
        yb, pb = y_true[idx], y_prob[idx]

        # Skip replicates with only one class (AUROC undefined)
        if len(np.unique(yb)) < 2:
            tpr_boot[i] = np.nan
            continue

        fpr_i, tpr_i, _ = roc_curve(yb, pb)
        tpr_boot[i]      = np.interp(fpr_grid, fpr_i, tpr_i)

    lo  = (1 - ci_alpha) / 2 * 100
    hi  = (1 - (1 - ci_alpha) / 2) * 100
    tpr_lower = np.nanpercentile(tpr_boot, lo,  axis=0)
    tpr_upper = np.nanpercentile(tpr_boot, hi, axis=0)

    return fpr_grid, tpr_lower, tpr_upper


# 5.2  Public API
def plot_auroc(
    X,
    y,
    model,
    label=None,
) -> plt.Figure:
    """
    Plot ROC curve(s) annotated with AUROC values.

    Supports single-model and multi-model / multi-dataset overlay.
    When exactly one (X, y, model) triple is provided, a 95 % confidence
    band is drawn via 1 000-replicate bootstrapping.

    Parameters
    X : pd.DataFrame | list[pd.DataFrame]: Feature matrix or list of matrices for multi-curve plots.
    y : pd.Series | list[pd.Series]: Target vector or list of vectors aligned with *X*.
    model : fitted estimator | list: Single model or list of fitted models; must align with *X* / *y*.
    label : str | list[str] | None: Curve label(s) for the legend.  Defaults to ``'Model'`` (single) or ``'Model 1'``, ``'Model 2'``, … (multi).

    Returns:
    matplotlib.figure.Figure
        ROC figure with diagonal chance line, shaded AUC region (single
        model) or solid curves (multi-model), and an AUROC-annotated legend.

    Raises:
    ValueError
        If list arguments have unequal lengths.
    TypeError
        If any model does not implement ``predict_proba``.
    """
    # Normalise inputs to lists                                            
    def _to_list(val):
        return val if isinstance(val, list) else [val]

    X_list     = _to_list(X)
    y_list     = _to_list(y)
    model_list = _to_list(model)
    label_list = _to_list(label) if label is not None else [None] * len(X_list)

    # Length consistency
    lengths = {len(X_list), len(y_list), len(model_list), len(label_list)}
    if len(lengths) != 1:
        raise ValueError(
            f"X, y, model, and label must all have the same length when "
            f"supplied as lists. Got lengths: X={len(X_list)}, "
            f"y={len(y_list)}, model={len(model_list)}, "
            f"label={len(label_list)}."
        )

    n_curves   = len(X_list)
    single     = (n_curves == 1)

    # Default labels
    if single:
        label_list = [label_list[0] or "Model"]
    else:
        label_list = [
            lbl or f"Model {i+1}"
            for i, lbl in enumerate(label_list)
        ]

    # Validate models                                                      
    for i, m in enumerate(model_list):
        if not hasattr(m, "predict_proba"):
            raise TypeError(
                f"model[{i}] (type '{type(m).__name__}') does not implement "
                "predict_proba."
            )

    # Figure setup                                                         
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    # Diagonal chance line
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", linewidth=1.2,
        color="#9CA3AF", label="Random chance (AUROC = 0.50)",
        zorder=1,
    )

    # Draw curves                                                          
    for i, (Xi, yi, mi, lbl) in enumerate(
        zip(X_list, y_list, model_list, label_list)
    ):
        y_arr    = np.asarray(yi)
        y_prob   = mi.predict_proba(Xi)[:, 1]
        auroc    = roc_auc_score(y_arr, y_prob)
        fpr, tpr, _ = roc_curve(y_arr, y_prob)

        color     = _PALETTE[i % len(_PALETTE)]
        curve_lbl = f"{lbl}  (AUROC = {auroc:.3f})"

        ax.plot(fpr, tpr, linewidth=2.2, color=color, label=curve_lbl, zorder=3)

        if single:
            # Shaded area under the curve
            ax.fill_between(fpr, tpr, alpha=0.12, color=color, zorder=2)

            # 95 % CI band via bootstrapping
            fpr_grid, tpr_lo, tpr_hi = _bootstrap_roc(y_arr, y_prob)
            ci_pct = int(_BOOTSTRAP_CI_ALPHA * 100)
            ax.fill_between(
                fpr_grid, tpr_lo, tpr_hi,
                alpha=0.20, color=color,
                label=f"{ci_pct} % CI (bootstrap n={_BOOTSTRAP_N_REPLICATES})",
                zorder=2,
            )

        logger.debug(f"  plot_auroc | {lbl} | AUROC={auroc:.4f}")

    # Axes decoration                                                      
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=12)
    ax.set_title("Receiver Operating Characteristic", fontsize=13, fontweight="bold")

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(labelsize=10)

    ax.grid(True, linestyle=":", linewidth=0.7, color="#E5E7EB", zorder=0)
    ax.set_axisbelow(True)

    legend = ax.legend(
        loc="lower right",
        fontsize=10,
        framealpha=0.9,
        edgecolor="#D1D5DB",
    )

    fig.tight_layout()
    return fig


# Internal bootstrap helper for PR curve
def _bootstrap_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_replicates: int = _BOOTSTRAP_N_REPLICATES,
    ci_alpha: float   = _BOOTSTRAP_CI_ALPHA,
    random_state: int = 42,
) -> tuple:
    """
    Estimate a confidence band for the PR curve via non-parametric
    bootstrapping.

    The PR curve is evaluated on a shared recall grid (linspace(0, 1, 100))
    using right-side interpolation so that precision is conservative between
    observed recall points.

    Parameters
    ----------
    y_true       : 1-D array of true binary labels.
    y_prob       : 1-D array of predicted positive-class probabilities.
    n_replicates : number of bootstrap samples.
    ci_alpha     : confidence level.
    random_state : seed for reproducibility.

    Returns
    -------
    (recall_grid, precision_lower, precision_upper)
        All three are 1-D arrays of length 100.
    """
    rng          = np.random.default_rng(random_state)
    n            = len(y_true)
    recall_grid  = np.linspace(0, 1, 100)
    prec_boot    = np.zeros((n_replicates, len(recall_grid)))

    for i in range(n_replicates):
        idx    = rng.integers(0, n, size=n)
        yb, pb = y_true[idx], y_prob[idx]

        if len(np.unique(yb)) < 2:
            prec_boot[i] = np.nan
            continue

        prec_i, rec_i, _ = precision_recall_curve(yb, pb)
        # sklearn returns in decreasing-recall order; flip for interpolation
        prec_i = prec_i[::-1]
        rec_i  = rec_i[::-1]
        prec_boot[i] = np.interp(recall_grid, rec_i, prec_i)

    lo  = (1 - ci_alpha) / 2 * 100
    hi  = (1 - (1 - ci_alpha) / 2) * 100
    prec_lower = np.nanpercentile(prec_boot, lo,  axis=0)
    prec_upper = np.nanpercentile(prec_boot, hi, axis=0)

    return recall_grid, prec_lower, prec_upper


# 5.3  
def plot_precision_recall(
    X,
    y,
    model,
    label=None,
) -> plt.Figure:
    """
    Plot Precision-Recall curve(s) annotated with AUPRC values.

    Particularly informative for class-imbalanced settings common in
    microbiome cancer cohorts.  Supports single-model and multi-model /
    multi-dataset overlay (pass lists).  When exactly one (X, y, model)
    triple is provided, a 95 % confidence band is drawn via bootstrapping
    and a no-skill baseline is added at the positive-class prevalence.

    Parameters
    ----------
    X : pd.DataFrame | list[pd.DataFrame]
        Feature matrix or list of matrices for multi-curve plots.
    y : pd.Series | list[pd.Series]
        Binary target vector or list of vectors aligned with *X*.
    model : fitted estimator | list
        Single fitted model or list; must align with *X* / *y*.
    label : str | list[str] | None
        Curve label(s) for the legend.  Defaults to ``'Model'`` (single)
        or ``'Model 1'``, ``'Model 2'``, … (multi).

    Returns
    -------
    matplotlib.figure.Figure
        PR figure with no-skill baseline (single model), shaded AUPRC
        region (single model), optional 95 % CI band, and an
        AUPRC-annotated legend.

    Raises
    ------
    ValueError
        If list arguments have unequal lengths.
    TypeError
        If any model does not implement ``predict_proba``.
    """
    # ------------------------------------------------------------------ #
    # Normalise inputs to lists (mirrors plot_auroc)                      #
    # ------------------------------------------------------------------ #
    def _to_list(val):
        return val if isinstance(val, list) else [val]

    X_list     = _to_list(X)
    y_list     = _to_list(y)
    model_list = _to_list(model)
    label_list = _to_list(label) if label is not None else [None] * len(X_list)

    lengths = {len(X_list), len(y_list), len(model_list), len(label_list)}
    if len(lengths) != 1:
        raise ValueError(
            f"X, y, model, and label must all have the same length when "
            f"supplied as lists. Got lengths: X={len(X_list)}, "
            f"y={len(y_list)}, model={len(model_list)}, "
            f"label={len(label_list)}."
        )

    n_curves = len(X_list)
    single   = (n_curves == 1)

    if single:
        label_list = [label_list[0] or "Model"]
    else:
        label_list = [
            lbl or f"Model {i+1}"
            for i, lbl in enumerate(label_list)
        ]

    # ------------------------------------------------------------------ #
    # Validate models                                                      #
    # ------------------------------------------------------------------ #
    for i, m in enumerate(model_list):
        if not hasattr(m, "predict_proba"):
            raise TypeError(
                f"model[{i}] (type '{type(m).__name__}') does not implement "
                "predict_proba."
            )

    # ------------------------------------------------------------------ #
    # Figure setup                                                         #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    # No-skill baseline — drawn only for single model using its prevalence
    if single:
        prevalence = float(np.mean(np.asarray(y_list[0])))
        ax.axhline(
            y=prevalence,
            linestyle="--", linewidth=1.2,
            color="#9CA3AF",
            label=f"No skill  (prevalence = {prevalence:.2f})",
            zorder=1,
        )

    # ------------------------------------------------------------------ #
    # Draw curves                                                          #
    # ------------------------------------------------------------------ #
    for i, (Xi, yi, mi, lbl) in enumerate(
        zip(X_list, y_list, model_list, label_list)
    ):
        y_arr  = np.asarray(yi)
        y_prob = mi.predict_proba(Xi)[:, 1]
        auprc  = average_precision_score(y_arr, y_prob)

        # sklearn returns precision/recall in decreasing-recall order
        prec, rec, _ = precision_recall_curve(y_arr, y_prob)

        color     = _PALETTE[i % len(_PALETTE)]
        curve_lbl = f"{lbl}  (AUPRC = {auprc:.3f})"

        ax.plot(rec, prec, linewidth=2.2, color=color, label=curve_lbl, zorder=3)

        if single:
            # Shade area under the PR curve
            ax.fill_between(rec, prec, alpha=0.12, color=color, zorder=2)

            # 95 % CI band via bootstrapping
            rec_grid, prec_lo, prec_hi = _bootstrap_pr(y_arr, y_prob)
            ci_pct = int(_BOOTSTRAP_CI_ALPHA * 100)
            ax.fill_between(
                rec_grid, prec_lo, prec_hi,
                alpha=0.20, color=color,
                label=f"{ci_pct} % CI (bootstrap n={_BOOTSTRAP_N_REPLICATES})",
                zorder=2,
            )

        logger.debug(f"  plot_precision_recall | {lbl} | AUPRC={auprc:.4f}")

    # ------------------------------------------------------------------ #
    # Axes decoration                                                      #
    # ------------------------------------------------------------------ #
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Recall  (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision  (Positive Predictive Value)", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(labelsize=10)

    ax.grid(True, linestyle=":", linewidth=0.7, color="#E5E7EB", zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
        edgecolor="#D1D5DB",
    )

    fig.tight_layout()
    return fig


# 5.4
def stratify_by_age(
    df: pd.DataFrame,
    n_groups: int,
    age_col: str = "age",
) -> pd.DataFrame:
    """
    Divide *df* into *n_groups* quantile-based age strata.

    Binning is quantile-based (equal-frequency) so each stratum contains
    approximately the same number of records regardless of the age
    distribution.  Records with missing age are retained with ``NaN`` in
    both new columns.

    Parametersж
    df : pd.DataFrame: Input DataFrame containing an age column.
    n_groups : int: Number of age strata. ≥ 2 and ≤ the number of unique non-null age values.
    age_col : str, default ``'age'`` Name of the column containing age values.

    Returns:
    pd.DataFrame
        Copy of *df* with two additional columns appended:
        ``'age_group'``
            int (0-indexed) — stratum label, ``NaN`` for missing ages.
        ``'age_group_range'``
            str — human-readable interval, e.g. ``'[45, 60)'``,
            ``NaN`` for missing ages.

    Raises:
    ValueError
        If *n_groups* < 2.
    ValueError
        If *age_col* is not in *df*.
    ValueError
        If *n_groups* exceeds the number of unique non-null age values.
    """
    # Validation                                                           
    if age_col not in df.columns:
        raise ValueError(
            f"age_col='{age_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if n_groups < 2:
        raise ValueError(
            f"n_groups must be >= 2, got {n_groups}."
        )

    n_unique = df[age_col].dropna().nunique()
    if n_groups > n_unique:
        raise ValueError(
            f"n_groups ({n_groups}) exceeds the number of unique non-null "
            f"age values ({n_unique})."
        )

    # Warn about missing ages
    n_missing = df[age_col].isna().sum()
    if n_missing > 0:
        warnings.warn(
            f"{n_missing} record(s) have missing values in '{age_col}'. "
            "They will be retained with NaN in 'age_group' and "
            "'age_group_range'.",
            UserWarning,
            stacklevel=2,
        )


    # Quantile-based binning                                               
    result = df.copy()

    # Handles ties at boundaries and returns bin edges for labeling.
    age_cut, bins = pd.qcut(
        result[age_col],
        q=n_groups,
        labels=False,          # returns integer codes 0 … n_groups-1
        retbins=True,
        duplicates="drop",     # merges bins whose edges collapse due to ties
    )

    # Actual number of bins may be fewer than n_groups when ties forced a merge
    actual_groups = len(bins) - 1
    if actual_groups < n_groups:
        warnings.warn(
            f"Ties at quantile boundaries reduced the number of strata from "
            f"{n_groups} to {actual_groups}.",
            UserWarning,
            stacklevel=2,
        )

    result["age_group"] = age_cut.astype("Int64")   # nullable int → NaN safe

    # Build readable interval strings, e.g. '[45, 60)'
    # Last bin uses ']' (closed on right): '[75, 90]'
    interval_labels: list = []
    for i in range(actual_groups):
        lo = bins[i]
        hi = bins[i + 1]
        bracket_close = "]" if i == actual_groups - 1 else ")"
        interval_labels.append(f"[{lo:.4g}, {hi:.4g}{bracket_close}")

    # Map integer code → interval string; NaN codes remain NaN
    code_to_label = {i: interval_labels[i] for i in range(actual_groups)}
    result["age_group_range"] = result["age_group"].map(code_to_label)
    register_metadata_col("age_group", "age_group_range")

    # Summary log                                                          
    lines = [
        "",
        f"  === stratify_by_age  (n_groups={actual_groups}) ===",
        f"  {'Group':<6} {'Range':<20} {'Count':>7}  {'%':>6}",
        "  " + "-" * 44,
    ]
    total_valid = int(result["age_group"].notna().sum())
    for code, interval in code_to_label.items():
        count = int((result["age_group"] == code).sum())
        pct   = 100 * count / total_valid if total_valid > 0 else 0.0
        lines.append(f"  {code:<6} {interval:<20} {count:>7}  {pct:>5.1f}%")
    if n_missing:
        lines.append(f"  {'NaN':<6} {'(missing)':<20} {int(n_missing):>7}")
    lines.append("")
    logger.info("\n".join(lines))

    return result.reset_index(drop=True)

# 5.5
def plot_confusion_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    title: str     = "Confusion Matrix",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot and optionally save a confusion matrix heatmap.

    Parameters:
    X         : pd.DataFrame  Feature matrix.
    y         : pd.Series     True binary labels.
    model     : fitted sklearn estimator
    title     : str           Plot title.
    save_path : str | None    If provided, saves the figure to this path.

    Returns:
    matplotlib.figure.Figure
    """
    import seaborn as sns

    _check_binary_target(y)
    _check_feature_alignment(X, model)

    y_pred = model.predict(X)
    cm     = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot        = True,
        fmt          = "d",
        cmap         = "Blues",
        xticklabels  = ["Control", "CRC"],
        yticklabels  = ["Control", "CRC"],
        ax           = ax,
    )
    ax.set_xlabel("Predicted",  fontsize=12)
    ax.set_ylabel("True",       fontsize=12)
    ax.set_title(title,         fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Confusion matrix saved to: {save_path}")

    return fig