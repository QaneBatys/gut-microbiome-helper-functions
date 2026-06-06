"""
Machine learning utilities for gut microbiome classification.

Main Functions:
  train_model(X, y, model, strategy, params)
      Train an estimator with a chosen validation strategy and return a
      full metrics dict (+ optional CSV report).

  tune_hyperparameters(X, y, model, param_grid, strategy, params)
      Optimise hyperparameters via GridSearchCV or RandomizedSearchCV
      (auto-selected by grid size) and return (best_estimator, results_df).

Wraps sklearn estimators with a unified interface that supports multiple
validation strategies (train-test, k-fold, stratified-k-fold,
leave-one-dataset-out) and returns a consistent results dictionary with a
full metrics suite.

Metrics computed per evaluation:
  accuracy          — fraction of correct predictions
  balanced_accuracy — macro-averaged recall (robust to class imbalance)
  mcc               — Matthews Correlation Coefficient  [-1, 1]
  auroc             — Area Under the ROC Curve
  auprc             — Area Under the Precision-Recall Curve
  precision         — positive predictive value
  recall            — sensitivity / true positive rate
  specificity       — true negative rate
  f1                — harmonic mean of precision and recall

For CV strategies every metric is reported as mean ± std across folds.
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone

from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# MODULE LOGGER
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# CONSTANTS
_VALID_STRATEGIES = {"train-test", "k-fold", "stratified-k-fold", "lodo", "all"}
_RANDOM_STATE     = 42

# ORDERED LIST FOR SUMMARY TABLE AND CSV
_METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "auroc",
    "auprc",
    "precision",
    "recall",
    "specificity",
    "f1",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _check_predict_proba(model) -> None:
    """Raise TypeError if *model* does not implement predict_proba."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"model of type '{type(model).__name__}' does not implement "
            "predict_proba, which is required for probability-based metrics."
        )


def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Compute the full metrics suite for one evaluation split.

    Parameters:
    y_true : array-like   Ground-truth binary labels.
    y_pred : array-like   Hard predicted labels.
    y_prob : array-like   Predicted probabilities for the positive class.

    Returns:
    dict  Keys match ``_METRIC_NAMES``.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Specificity = TN / (TN + FP) — safe against zero-division
    if cm.shape == (2, 2):
        tn, fp      = cm[0, 0], cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    else:
        specificity = float("nan")

    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc":               matthews_corrcoef(y_true, y_pred),
        "auroc":             roc_auc_score(y_true, y_prob),
        "auprc":             average_precision_score(y_true, y_prob),
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "recall":            recall_score(y_true, y_pred, zero_division=0),
        "specificity":       specificity,
        "f1":                f1_score(y_true, y_pred, zero_division=0),
    }


def _aggregate_fold_metrics(fold_metrics: list) -> dict:
    """
    Compute mean and std for every metric across CV folds.

    Parameters:
    fold_metrics : list[dict]  One dict per fold, each from ``_compute_metrics``.

    Returns:
    dict
        Keys: ``<metric>`` (mean), ``<metric>_std``, and ``cv_fold_metrics``
        (the raw per-fold list).
    """
    aggregated = {}
    for metric in _METRIC_NAMES:
        values = [fm[metric] for fm in fold_metrics]
        aggregated[metric]          = float(np.nanmean(values))
        aggregated[f"{metric}_std"] = float(np.nanstd(values))
    aggregated["cv_fold_metrics"] = fold_metrics
    return aggregated


def _summary_table(strategy: str, metrics: dict, fold_labels: list = None) -> str:
    """Build a compact human-readable summary string for the logger."""
    lines = [
        "",
        f"  Strategy : {strategy}",
        f"  {'Metric':<22} {'Mean':>8}  {'Std':>8}",
        "  " + "-" * 42,
    ]
    for m in _METRIC_NAMES:
        mean    = metrics.get(m, float("nan"))
        std     = metrics.get(f"{m}_std", None)
        std_str = f"{std:8.4f}" if std is not None else "        "
        lines.append(f"  {m:<22} {mean:8.4f}  {std_str}")

    if fold_labels:
        cv_fold_metrics = metrics.get("cv_fold_metrics", [])
        if cv_fold_metrics:
            lines.append("\n  Per-fold AUROC:")
            for lbl, fm in zip(fold_labels, cv_fold_metrics):
                lines.append(f"    {lbl}: {fm['auroc']:.4f}")
    lines.append("")
    return "\n".join(lines)


def _write_validation_csv(
    strategy: str,
    metrics: dict,
    output_path: str,
    fold_labels: list = None,
) -> None:
    """
    Write a dedicated validation CSV with per-fold metrics + overall summary
    for the best model.

    File is always OVERWRITTEN (not appended) so it contains only the
    best model results.

    Row layout:
    * One row per fold with raw metric values.
    * One final OVERALL row with mean ± std across folds.

    Parameters:
    strategy    : str   Written to the ``strategy`` column.
    metrics     : dict  From ``_aggregate_fold_metrics`` or ``_compute_metrics``.
    output_path : str   Destination CSV path (overwritten each call).
    fold_labels : list  Per-fold identifiers (default: ``fold_1``, ``fold_2``, …).
    """
    rows = []

    cv_fold_metrics = metrics.get("cv_fold_metrics", [])
    labels = fold_labels or [f"fold_{i+1}" for i in range(len(cv_fold_metrics))]

    # per-fold rows
    for lbl, fm in zip(labels, cv_fold_metrics):
        row = {"strategy": strategy, "split": lbl}
        for m in _METRIC_NAMES:
            row[m] = fm.get(m, float("nan"))
        rows.append(row)

    # overall summary row — mean ± std
    overall = {"strategy": strategy, "split": "overall"}
    for m in _METRIC_NAMES:
        mean = metrics.get(m, float("nan"))
        std  = metrics.get(f"{m}_std", float("nan"))
        overall[m]          = mean
        overall[f"{m}_std"] = std
    rows.append(overall)

    df        = pd.DataFrame(rows)
    std_cols  = [f"{m}_std" for m in _METRIC_NAMES]
    col_order = ["strategy", "split"] + _METRIC_NAMES + std_cols
    df        = df.reindex(columns=col_order)

    df.to_csv(output_path, index=False)
    logger.info(f"  Validation metrics written to: {output_path}")


def _write_test_csv(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str,
    dataset_col: str = None,
) -> None:
    """
    Write a dedicated test CSV with per-dataset accuracy + overall metrics.

    Row layout:
    * One row per dataset (if ``dataset_col`` is provided in X_test).
    * One final OVERALL row with all metrics on the full test set.

    Parameters:
    model       : fitted estimator
    X_test      : pd.DataFrame  Test feature matrix.
    y_test      : pd.Series     True test labels.
    output_path : str           Destination CSV path (overwritten each call).
    dataset_col : str | None    Column in X_test containing dataset names.
    """
    rows = []

    feature_cols = (
        [c for c in X_test.columns if c != dataset_col]
        if dataset_col and dataset_col in X_test.columns
        else list(X_test.columns)
    )

    # per-dataset rows
    if dataset_col and dataset_col in X_test.columns:
        for ds in sorted(X_test[dataset_col].unique()):
            mask   = X_test[dataset_col] == ds
            X_ds   = X_test.loc[mask, feature_cols]
            y_ds   = y_test.loc[mask]

            if len(y_ds.unique()) < 2:
                logger.warning(
                    f"  Test dataset '{ds}' has only one class — "
                    "metrics may be undefined."
                )

            y_pred = model.predict(X_ds)
            y_prob = model.predict_proba(X_ds)[:, 1]
            m      = _compute_metrics(y_ds, y_pred, y_prob)

            row = {
                "split":   ds,
                "n":       int(len(y_ds)),
                "n_pos":   int(y_ds.sum()),
                "n_neg":   int((y_ds == 0).sum()),
            }
            row.update(m)
            rows.append(row)

    # overall row — full test set
    y_pred_all = model.predict(X_test[feature_cols])
    y_prob_all = model.predict_proba(X_test[feature_cols])[:, 1]
    m_all      = _compute_metrics(y_test, y_pred_all, y_prob_all)

    overall = {
        "split": "overall",
        "n":     int(len(y_test)),
        "n_pos": int(y_test.sum()),
        "n_neg": int((y_test == 0).sum()),
    }
    overall.update(m_all)
    rows.append(overall)

    col_order = ["split", "n", "n_pos", "n_neg"] + _METRIC_NAMES
    df        = pd.DataFrame(rows).reindex(columns=col_order)

    df.to_csv(output_path, index=False)
    logger.info(f"  Test metrics written to: {output_path}")


def _metrics_to_csv(
    strategy: str,
    metrics: dict,
    output_path: str,
    fold_labels: list = None,
) -> None:
    """
    Append strategy metrics to a CSV report file.

    The file is created with a header on the first call; subsequent calls
    append rows so that one file accumulates results from multiple
    train_model() calls (e.g. different strategies or model types).

    Row layout:
    * One **summary** row  — mean values across folds (or the single split).
    * One row **per fold** — raw metric values, ``_std`` columns left blank.

    Parameters:
    strategy    : str   Written to the ``strategy`` column.
    metrics     : dict  From ``_aggregate_fold_metrics`` or ``_compute_metrics``.
    output_path : str   Destination CSV path.
    fold_labels : list  Per-fold identifiers (default: ``fold_1``, ``fold_2``, …).
    """
    rows = []

    # Summary row
    summary_row = {"strategy": strategy, "split": "summary"}
    for m in _METRIC_NAMES:
        summary_row[m]          = metrics.get(m, float("nan"))
        summary_row[f"{m}_std"] = metrics.get(f"{m}_std", None)
    rows.append(summary_row)

    # Per-fold rows (present only for CV strategies)
    cv_fold_metrics = metrics.get("cv_fold_metrics", [])
    labels = fold_labels or [f"fold_{i+1}" for i in range(len(cv_fold_metrics))]
    for lbl, fm in zip(labels, cv_fold_metrics):
        fold_row = {"strategy": strategy, "split": lbl}
        for m in _METRIC_NAMES:
            fold_row[m]          = fm.get(m, float("nan"))
            fold_row[f"{m}_std"] = None
        rows.append(fold_row)

    new_df = pd.DataFrame(rows)

    std_cols  = [f"{m}_std" for m in _METRIC_NAMES]
    col_order = ["strategy", "split"] + _METRIC_NAMES + std_cols
    new_df    = new_df.reindex(columns=col_order)

    file_exists = os.path.isfile(output_path)
    new_df.to_csv(output_path, mode="a", header=not file_exists, index=False)
    verb = "appended to" if file_exists else "written to"
    logger.info(f"  Metrics {verb}: {output_path}")


# ============================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================

def _train_test(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    output_path: str = None,
) -> dict:
    """Single 80/20 stratified train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=_RANDOM_STATE, stratify=y
    )
    fitted = clone(model)
    fitted.fit(X_train, y_train)

    y_pred   = fitted.predict(X_test)
    y_prob   = fitted.predict_proba(X_test)[:, 1]
    metrics  = _compute_metrics(y_test, y_pred, y_prob)

    # Add _std keys as None for a consistent dict shape
    metrics_full = {**metrics, **{f"{m}_std": None for m in _METRIC_NAMES}}

    logger.info(_summary_table("train-test", metrics_full))

    if output_path:
        _metrics_to_csv("train-test", metrics_full, output_path)

    return {
        "strategy":  "train-test",
        "model":     fitted,
        "metrics":   metrics_full,
        "auroc":     metrics["auroc"],
        "auroc_std": None,
        "cv_scores": [],
    }


def _kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    k: int = 5,
    stratified: bool = False,
    n_jobs: int = -1,
    strategy_name: str = None,
    output_path: str = None,
) -> dict:
    """K-fold (or stratified-k-fold) cross-validation."""
    strategy_name = strategy_name or ("stratified-k-fold" if stratified else "k-fold")

    cv = (
        StratifiedKFold(n_splits=k, shuffle=True, random_state=_RANDOM_STATE)
        if stratified
        else KFold(n_splits=k, shuffle=True, random_state=_RANDOM_STATE)
    )

    fold_metrics_list: list = []
    fitted_models: list     = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)

        y_pred = fold_model.predict(X_te)
        y_prob = fold_model.predict_proba(X_te)[:, 1]
        fm     = _compute_metrics(y_te, y_pred, y_prob)

        fold_metrics_list.append(fm)
        fitted_models.append(fold_model)
        logger.debug(
            f"  {strategy_name} | fold {fold_idx+1}/{k} "
            f"| AUROC={fm['auroc']:.4f}  MCC={fm['mcc']:.4f}"
        )

    agg         = _aggregate_fold_metrics(fold_metrics_list)
    fold_labels = [f"fold_{i+1}" for i in range(k)]

    logger.info(_summary_table(strategy_name, agg, fold_labels))

    if output_path:
        _metrics_to_csv(strategy_name, agg, output_path, fold_labels)

    return {
        "strategy":  strategy_name,
        "model":     fitted_models,
        "metrics":   agg,
        "auroc":     agg["auroc"],
        "auroc_std": agg["auroc_std"],
        "cv_scores": [fm["auroc"] for fm in fold_metrics_list],
    }


def _lodo(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    dataset_col: str,
    n_jobs: int = -1,
    output_path: str = None,
) -> dict:
    """Leave-One-Dataset-Out cross-validation."""
    if dataset_col not in X.columns:
        raise ValueError(
            f"dataset_col='{dataset_col}' not found in X. "
            f"Available columns: {list(X.columns)}"
        )

    datasets     = X[dataset_col].unique()
    feature_cols = [c for c in X.columns if c != dataset_col]

    fold_metrics_list: list = []
    fitted_models: list     = []
    fold_labels: list       = []

    for ds in datasets:
        test_mask  = X[dataset_col] == ds
        train_mask = ~test_mask

        X_tr = X.loc[train_mask, feature_cols]
        y_tr = y.loc[train_mask]
        X_te = X.loc[test_mask,  feature_cols]
        y_te = y.loc[test_mask]

        if len(y_te.unique()) < 2:
            warnings.warn(
                f"LODO fold for dataset '{ds}' has only one class in the "
                "test set — fold skipped.",
                UserWarning,
                stacklevel=3,
            )
            continue

        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)

        y_pred = fold_model.predict(X_te)
        y_prob = fold_model.predict_proba(X_te)[:, 1]
        fm     = _compute_metrics(y_te, y_pred, y_prob)

        fold_metrics_list.append(fm)
        fitted_models.append(fold_model)
        fold_labels.append(str(ds))
        logger.debug(
            f"  lodo | left-out='{ds}' "
            f"| AUROC={fm['auroc']:.4f}  MCC={fm['mcc']:.4f}"
        )

    if fold_metrics_list:
        agg = _aggregate_fold_metrics(fold_metrics_list)
    else:
        agg = {m: float("nan") for m in _METRIC_NAMES}
        agg.update({f"{m}_std": float("nan") for m in _METRIC_NAMES})
        agg["cv_fold_metrics"] = []

    logger.info(_summary_table("lodo", agg, fold_labels))

    if output_path:
        _metrics_to_csv("lodo", agg, output_path, fold_labels)

    return {
        "strategy":    "lodo",
        "model":       fitted_models,
        "metrics":     agg,
        "fold_labels": fold_labels,
        "auroc":       agg["auroc"],
        "auroc_std":   agg["auroc_std"],
        "cv_scores":   [fm["auroc"] for fm in fold_metrics_list],
    }


# ============================================================
# MAIN FUNCTIONS
# ============================================================

# 4.1
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    strategy: str = "train-test",
    params: dict  = None,
) -> dict:
    """
    Train a scikit-learn estimator using the specified validation strategy.

    Parameters:
    X : pd.DataFrame: Feature matrix (samples × features). Must not contain the target column.
    y : pd.Series: Binary target vector aligned with *X*.
    model : sklearn estimator
        An **unfitted** scikit-learn-compatible classifier that implements
        ``fit`` and ``predict_proba``.
    strategy : str, default ``'train-test'``
        Validation strategy. One of:
        ``'train-test'``        — single 80/20 stratified split.
        ``'k-fold'``            — k-fold CV (k from params, default 5).
        ``'stratified-k-fold'`` — stratified k-fold CV.
        ``'lodo'``              — leave-one-dataset-out CV.
        ``'all'``               — runs all four strategies sequentially.
    params: dict | None, default ``None``
        Optional configuration. Supported keys:
        ``'k'``                  int — number of CV folds (default 5).
        ``'dataset_col'``        str — column in *X* with dataset IDs (LODO only).
        ``'n_jobs'``             int — parallelism hint (default -1).
        ``'output_csv'``         str — path for the general metrics CSV report.
        ``'validation_csv'``     str — path for dedicated validation CSV
                                       (per-fold + overall for best model).
        ``'test_csv'``           str — path for dedicated test CSV
                                       (per-dataset + overall).
        ``'X_test'``             pd.DataFrame — held-out test features for
                                       test CSV (required for test_csv).
        ``'y_test'``             pd.Series    — held-out test labels for
                                       test CSV (required for test_csv).

    Returns:
    dict
        ``'strategy'``            str | list[str]
        ``'model'``               fitted estimator or list of estimators
        ``'metrics'``             dict — mean values for all metrics; CV
                                  strategies also include ``<metric>_std``
                                  and ``cv_fold_metrics`` (raw per-fold dicts)
        ``'auroc'``               float | list  — convenience alias
        ``'auroc_std'``           float | None  — convenience alias
        ``'cv_scores'``           list[float]   — per-fold AUROCs
        ``'results'``             dict (strategy='all' only)

        Full metrics suite in ``'metrics'``:
        accuracy, balanced_accuracy, mcc, auroc, auprc,
        precision, recall, specificity, f1

    Raises:
    ValueError  If *strategy* is not recognised.
    ValueError  If ``strategy='lodo'`` and ``params['dataset_col']`` is absent.
    TypeError   If *model* does not implement ``predict_proba``.
    """
    # Input validation
    strategy = strategy.lower()
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {sorted(_VALID_STRATEGIES)}, "
            f"got '{strategy}'."
        )

    _check_predict_proba(model)

    params          = params or {}
    k               = int(params.get("k", 5))
    dataset_col     = params.get("dataset_col", None)
    n_jobs          = int(params.get("n_jobs", -1))
    output_path     = params.get("output_csv",     None)
    validation_path = params.get("validation_csv", None)
    test_path       = params.get("test_csv",       None)
    X_test          = params.get("X_test",         None)
    y_test          = params.get("y_test",         None)

    if strategy == "lodo" and dataset_col is None:
        raise ValueError(
            "strategy='lodo' requires params['dataset_col'] to be set."
        )

    # Dispatch
    if strategy == "train-test":
        result = _train_test(X, y, model, output_path=output_path)
    elif strategy == "k-fold":
        result = _kfold_cv(
            X, y, model, k=k, stratified=False,
            n_jobs=n_jobs, strategy_name="k-fold",
            output_path=output_path,
        )
    elif strategy == "stratified-k-fold":
        result = _kfold_cv(
            X, y, model, k=k, stratified=True,
            n_jobs=n_jobs, strategy_name="stratified-k-fold",
            output_path=output_path,
        )
    elif strategy == "lodo":
        result = _lodo(
            X, y, model, dataset_col=dataset_col,
            n_jobs=n_jobs, output_path=output_path,
        )
    else:
        # strategy == 'all'
        all_results: dict = {}
        for strat in ("train-test", "k-fold", "stratified-k-fold", "lodo"):
            if strat == "lodo" and dataset_col is None:
                warnings.warn(
                    "strategy='all' skipped 'lodo' because "
                    "params['dataset_col'] was not provided.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            all_results[strat] = train_model(
                X, y, model, strategy=strat, params=params
            )

        strategies_run = list(all_results.keys())

        # Aggregate summary table logged at INFO
        header = f"\n  {'Strategy':<24} {'AUROC':>8}  {'±':>6}  {'MCC':>8}  {'F1':>8}"
        lines  = ["\n  === strategy='all' aggregate ===", header, "  " + "-" * 60]
        for s in strategies_run:
            m   = all_results[s]["metrics"]
            std = m.get("auroc_std") or 0.0
            lines.append(
                f"  {s:<24} {m['auroc']:8.4f}  ±{std:5.4f}  "
                f"{m['mcc']:8.4f}  {m['f1']:8.4f}"
            )
        lines.append("")
        logger.info("\n".join(lines))

        return {
            "strategy":  strategies_run,
            "model":     [all_results[s]["model"]     for s in strategies_run],
            "metrics":   [all_results[s]["metrics"]   for s in strategies_run],
            "auroc":     [all_results[s]["auroc"]     for s in strategies_run],
            "auroc_std": [all_results[s]["auroc_std"] for s in strategies_run],
            "cv_scores": [all_results[s]["cv_scores"] for s in strategies_run],
            "results":   all_results,
        }

    # ── write validation CSV (per-fold + overall) ────────────
    if validation_path:
        fold_labels = result.get("fold_labels", None)
        _write_validation_csv(
            strategy    = result["strategy"],
            metrics     = result["metrics"],
            output_path = validation_path,
            fold_labels = fold_labels,
        )


    # ── write test CSV (per-dataset + overall) ───────────────
    if test_path:
        if X_test is None or y_test is None:
            warnings.warn(
                "test_csv was specified but X_test / y_test were not "
                "provided in params — test CSV skipped.",
                UserWarning,
                stacklevel=2,
            )
        else:
            final_fitted = clone(model)
            X_for_fit = (
                X.drop(columns=[dataset_col])
                if dataset_col and dataset_col in X.columns
                else X
            )
            final_fitted.fit(X_for_fit, y)

            _write_test_csv(
                model       = final_fitted,
                X_test      = X_test,
                y_test      = y_test,
                output_path = test_path,
                dataset_col = dataset_col,
            )

    return result


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _build_cv_splitter(strategy: str, k: int, dataset_col: str = None, X: pd.DataFrame = None):
    """
    Return a cross-validation splitter compatible with GridSearchCV.

    For ``'lodo'`` a pre-split list of (train_idx, test_idx) tuples is
    returned instead of a splitter object — sklearn's search estimators
    accept both.

    Parameters
    strategy    : str           One of the four non-'all' strategy names.
    k           : int           Number of folds (ignored for lodo).
    dataset_col : str | None    Required for lodo.
    X           : pd.DataFrame  Required for lodo (to build the index lists).

    Returns
    -------
    sklearn CV splitter or list of (train_idx, test_idx) tuples.
    """
    if strategy == "train-test":
        # Single 80/20 split expressed as a one-fold pre-split list
        indices = np.arange(len(X))
        tr, te  = train_test_split(
            indices, test_size=0.2, random_state=_RANDOM_STATE
        )
        return [(tr, te)]

    if strategy == "k-fold":
        return KFold(n_splits=k, shuffle=True, random_state=_RANDOM_STATE)

    if strategy == "stratified-k-fold":
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=_RANDOM_STATE)

    if strategy == "lodo":
        splits = []
        for ds in X[dataset_col].unique():
            te_mask = (X[dataset_col] == ds).values
            tr_idx  = np.where(~te_mask)[0]
            te_idx  = np.where( te_mask)[0]
            splits.append((tr_idx, te_idx))
        return splits

    raise ValueError(
        f"_build_cv_splitter: unrecognised strategy '{strategy}'."
    )


def _count_grid_combinations(param_grid: dict) -> int:
    """Return the total number of hyperparameter combinations in *param_grid*."""
    total = 1
    for values in param_grid.values():
        try:
            total *= len(values)
        except TypeError:
            # scipy distribution or other samplable — treat as large
            return int(1e9)
    return total


# 4.2
def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    param_grid: dict,
    strategy: str = "stratified-k-fold",
    params: dict  = None,
) -> tuple:
    """
    Hyperparameter optimisation via GridSearchCV or RandomizedSearchCV.

    The search method is chosen automatically based on the number of grid
    combinations:

    * **≤ threshold** combinations  →  ``GridSearchCV``   (exhaustive)
    * **>  threshold** combinations  →  ``RandomizedSearchCV`` (sampled)

    The threshold defaults to 50 and can be overridden via
    ``params['grid_threshold']``.

    Parameters
    X : pd.DataFrame
        Feature matrix (samples × features).
    y : pd.Series
        Binary target vector aligned with *X*.
    model : sklearn estimator
        Unfitted base estimator (must implement ``predict_proba``).
    param_grid : dict
        Parameter grid passed directly to the search estimator.
        Values can be lists (Grid) or scipy distributions (Randomized).
    strategy : str, default ``'stratified-k-fold'``
        Inner CV strategy — same options as ``train_model`` except
        ``'all'`` is not supported here.
    params : dict | None, default ``None``
        Optional configuration. Supported keys:
        ``'k'``              int   — number of CV folds (default 5).
        ``'dataset_col'``    str   — column in *X* with dataset IDs (LODO only).
        ``'n_jobs'``         int   — parallelism (default -1).
        ``'n_iter'``         int   — fixed iterations for RandomizedSearchCV.
                                     Ignored if ``search_ratio`` is set.
                                     Default 50.
        ``'search_ratio'``   float — fraction of total combinations to sample,
                                     e.g. 0.80 samples 80% of the grid.
                                     Overrides ``n_iter`` when provided.
                                     Must be in (0, 1].
        ``'grid_threshold'`` int   — max combinations before switching to
                                     Randomized (default 50).
        ``'output_csv'``     str   — path for a CSV of all evaluated
                                     combinations (mean AUROC + std).
        ``'best_params_csv'`` str  — path for a dedicated CSV containing
                                     only the best model hyperparameters.
        ``'random_state'``   int   — seed for RandomizedSearchCV
                                     (default module ``_RANDOM_STATE``).

    Returns:
        tuple (best_estimator, results_df)

        ``best_estimator``
            Fitted sklearn estimator re-trained on the **full** *X* / *y*
            using the best hyperparameters found.

        ``results_df``
            pd.DataFrame with one row per evaluated combination, columns:
            ``rank``, ``params``, ``mean_auroc``, ``std_auroc``,
            plus one ``param_<name>`` column per hyperparameter for easy
            filtering.  Sorted by ``mean_auroc`` descending.

    Raises:
    ValueError
        If *strategy* is ``'all'`` (not supported for inner CV loop).
    ValueError
        If *strategy* is ``'lodo'`` and ``params['dataset_col']`` is absent.
    ValueError
        If ``search_ratio`` is not in (0, 1].
    TypeError
        If *model* does not implement ``predict_proba``.
    """
    # Validation
    strategy = strategy.lower()

    if strategy == "all":
        raise ValueError(
            "strategy='all' is not supported for tune_hyperparameters. "
            "Choose a single strategy: 'train-test', 'k-fold', "
            "'stratified-k-fold', or 'lodo'."
        )
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {sorted(_VALID_STRATEGIES - {'all'})}, "
            f"got '{strategy}'."
        )

    _check_predict_proba(model)

    params           = params or {}
    k                = int(params.get("k", 5))
    dataset_col      = params.get("dataset_col",     None)
    n_jobs           = int(params.get("n_jobs",      -1))
    n_iter           = int(params.get("n_iter",      50))
    search_ratio     = params.get("search_ratio",    None)
    grid_threshold   = int(params.get("grid_threshold", 50))
    output_path      = params.get("output_csv",      None)
    best_params_path = params.get("best_params_csv", None)
    random_state     = int(params.get("random_state", _RANDOM_STATE))

    # ── validate search_ratio ─────────────────────────────────
    if search_ratio is not None:
        if not (0 < search_ratio <= 1):
            raise ValueError(
                f"search_ratio must be in (0, 1], got {search_ratio}."
            )

    if strategy == "lodo" and dataset_col is None:
        raise ValueError(
            "strategy='lodo' requires params['dataset_col'] to be set."
        )

    # For LODO the dataset column must be excluded from the feature matrix
    # that the search estimator sees; we handle it by passing a stripped X.
    X_search = X.drop(columns=[dataset_col]) if strategy == "lodo" else X

    # Build CV splitter
    cv = _build_cv_splitter(strategy, k, dataset_col=dataset_col, X=X)

    # Choose search method
    n_combinations = _count_grid_combinations(param_grid)
    use_random     = n_combinations > grid_threshold

    # ── compute n_iter from ratio if provided ─────────────────
    if use_random and search_ratio is not None:
        n_iter = max(1, int(n_combinations * search_ratio))
        logger.info(
            f"  tune_hyperparameters: search_ratio={search_ratio} → "
            f"n_iter={n_iter} / {n_combinations} "
            f"({search_ratio*100:.1f}% of grid)"
        )
    elif use_random:
        logger.info(
            f"  tune_hyperparameters: {n_combinations} combinations > "
            f"threshold {grid_threshold} → RandomizedSearchCV "
            f"(n_iter={n_iter}, "
            f"{100*n_iter/n_combinations:.1f}% of grid)"
        )
    else:
        logger.info(
            f"  tune_hyperparameters: {n_combinations} combinations ≤ "
            f"threshold {grid_threshold} → GridSearchCV (exhaustive)"
        )

    # Shared kwargs for both search estimators
    common_kwargs = dict(
        estimator   = clone(model),
        scoring     = "roc_auc",
        cv          = cv,
        refit       = True,
        n_jobs      = n_jobs,
        error_score = "raise",
    )

    if use_random:
        search = RandomizedSearchCV(
            **common_kwargs,
            param_distributions = param_grid,
            n_iter              = n_iter,
            random_state        = random_state,
        )
    else:
        search = GridSearchCV(
            **common_kwargs,
            param_grid = param_grid,
        )

    # Run search
    search.fit(X_search, y)

    best_params    = search.best_params_
    best_estimator = search.best_estimator_   # already refit on full data

    logger.info(
        f"  Best AUROC : {search.best_score_:.4f}\n"
        f"  Best params: {best_params}"
    )

    # Build results DataFrame
    cv_results = pd.DataFrame(search.cv_results_)

    results_df = pd.DataFrame({
        "mean_auroc": cv_results["mean_test_score"],
        "std_auroc":  cv_results["std_test_score"],
        "params":     cv_results["params"],
    })

    # Explode param dict into individual columns for easy filtering
    param_cols = pd.json_normalize(cv_results["params"]).add_prefix("param_")
    results_df = pd.concat(
        [results_df.reset_index(drop=True), param_cols.reset_index(drop=True)],
        axis=1,
    )

    results_df = (
        results_df
        .sort_values("mean_auroc", ascending=False)
        .reset_index(drop=True)
    )
    results_df.insert(0, "rank", results_df.index + 1)

    # ── write all combinations CSV ────────────────────────────
    if output_path:
        file_exists = os.path.isfile(output_path)
        results_df.to_csv(output_path, mode="a", header=not file_exists, index=False)
        verb = "appended to" if file_exists else "written to"
        logger.info(f"  Tuning results {verb}: {output_path}")

    # ── write best params CSV ─────────────────────────────────
    if best_params_path:
        best_row = pd.DataFrame([{
            "best_auroc":    search.best_score_,
            "strategy":      strategy,
            "search_method": "RandomizedSearchCV" if use_random else "GridSearchCV",
            "n_iter":        n_iter if use_random else n_combinations,
            "n_combinations": n_combinations,
            "search_ratio":  round(n_iter / n_combinations, 4) if use_random else 1.0,
            **{f"param_{k}": v for k, v in best_params.items()},
        }])
        best_row.to_csv(best_params_path, index=False)
        logger.info(f"  Best params written to: {best_params_path}")

    # Summary log
    top_n = min(5, len(results_df))
    lines = [
        "",
        f"  === tune_hyperparameters ({strategy}) ===",
        f"  Search method  : {'RandomizedSearchCV' if use_random else 'GridSearchCV'}",
        f"  Total combos   : {n_combinations}",
        f"  Sampled        : {n_iter if use_random else n_combinations} "
        f"({100 * (n_iter if use_random else n_combinations) / n_combinations:.1f}%)",
        f"  {'Rank':<6} {'Mean AUROC':>10}  {'Std':>8}  Params",
        "  " + "-" * 70,
    ]
    for _, row in results_df.head(top_n).iterrows():
        lines.append(
            f"  {int(row['rank']):<6} {row['mean_auroc']:>10.4f}  "
            f"{row['std_auroc']:>8.4f}  {row['params']}"
        )
    lines.append("")
    logger.info("\n".join(lines))

    return best_estimator, results_df