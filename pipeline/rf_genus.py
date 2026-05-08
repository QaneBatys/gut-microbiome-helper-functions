#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from pd_utils     import prepare_target_class, split_dataset, prevalence_filtering, aggregate_taxa, _all_meta
from pd_transform import clr_transform
from pd_ml        import tune_hyperparameters, train_model
from pd_assess    import plot_auroc, plot_precision_recall, plot_confusion_matrix

# 1. CONFIGURATION
DATA_PATH = "./data/filtered_nine_crc_final_clean.tsv"
OUT_DIR   = "./results/rf_train_val/"
N_JOBS    = -1
os.makedirs(OUT_DIR, exist_ok=True)

# 2. LOAD
print("Loading data...")
df = pd.read_csv(DATA_PATH, sep="\t", index_col=0)
print(f"  Loaded : {df.shape[0]} samples × {df.shape[1]} columns")

# 3. PREPARE TARGET
#    CRC = 1, control = 0, adenoma excluded
df = prepare_target_class(
    df,
    input_attr         = "study_condition",
    target_class_label = "CRC",
    output_attr        = "target",
    exclude_labels     = ["adenoma"],
)
print(f"  After condition filter : {len(df)} samples")
print(f"  CRC     : {int(df['target'].sum())}")
print(f"  Control : {int((df['target'] == 0).sum())}")
print(f"  Missing age : {df['age'].isna().sum()} — retained")

# 4. GENUS-LEVEL AGGREGATION (Applied before split — deterministic sum)
n_before = len([c for c in df.columns if c not in _all_meta()])
df = aggregate_taxa(df, taxa_level="genus")
n_after  = len([c for c in df.columns if c not in _all_meta()])
print(f"\n  Genus aggregation : {n_before} species → {n_after} genera")

# 5. SPLIT — stratified per dataset → train | test
#    Per-dataset split ensures every study is in both sets.
#    X_train / X_test include all metadata columns + features.
splits = split_dataset(
    df,
    target_class_attr = "target",
    strategy          = "individual",
)

print("\n" + "=" * 65)
print("  PER-DATASET SPLIT SUMMARY (train / test)")
print("=" * 65)
print(f"  {'Dataset':<30} {'Train':>7}  {'Test':>6}  "
      f"{'Train CRC%':>10}  {'Test CRC%':>10}")
print(f"  {'─'*62}")

X_train_list, X_test_list = [], []
y_train_list, y_test_list = [], []

for ds, (X_tr, X_te, y_tr, y_te) in splits.items():
    print(f"  {ds:<30} {len(X_tr):>7}  {len(X_te):>6}  "
          f"{100*y_tr.mean():>9.1f}%  {100*y_te.mean():>9.1f}%")
    X_train_list.append(X_tr)
    X_test_list.append(X_te)
    y_train_list.append(y_tr)
    y_test_list.append(y_te)

X_train = pd.concat(X_train_list, axis=0).reset_index(drop=True)
X_test  = pd.concat(X_test_list,  axis=0).reset_index(drop=True)
y_train = pd.concat(y_train_list, axis=0).reset_index(drop=True)
y_test  = pd.concat(y_test_list,  axis=0).reset_index(drop=True)

print(f"\n  {'─'*62}")
print(f"  {'TOTAL':<30} {len(X_train):>7}  {len(X_test):>6}  "
      f"{100*y_train.mean():>9.1f}%  {100*y_test.mean():>9.1f}%")

# 6. PREVALENCE FILTERING — fitted on X_train only
#    Filters per dataset using intersect — a taxon must pass
#    10% prevalence threshold in EVERY dataset to survive.
#    Same surviving taxa list applied to X_test.
X_train_filtered = prevalence_filtering(
    X_train,
    threshold = 0.10,
    how       = "intersect",
)

meta_cols      = [c for c in X_train_filtered.columns if c in _all_meta()]
surviving_taxa = [c for c in X_train_filtered.columns if c not in _all_meta()]

print(f"\n  Raw genera              : "
      f"{len([c for c in X_train.columns if c not in _all_meta()])}")
print(f"  After prevalence filter : {len(surviving_taxa)} genera")

# apply same surviving taxa to test — no fitting on test
X_test_filtered = X_test[meta_cols + surviving_taxa].copy()

# 7. CLR TRANSFORM — applied AFTER filtering, per sample
#    Removes compositional constraint of microbiome data.
X_train_clr = clr_transform(X_train_filtered)
X_test_clr  = clr_transform(X_test_filtered)

# 8. NO SCALING FOR RANDOM FOREST
#    Tree-based models are invariant to feature scaling.
#    CLR transform already handles compositionality.
X_train_df = pd.DataFrame(
    X_train_clr[surviving_taxa].values,
    columns = surviving_taxa
)
X_test_df = pd.DataFrame(
    X_test_clr[surviving_taxa].values,
    columns = surviving_taxa
)

print(f"\n  After CLR (no scaling for RF) :")
print(f"  X_train_df : {X_train_df.shape}")
print(f"  X_test_df  : {X_test_df.shape}")

X_test_df_with_ds = X_test_df.copy()
X_test_df_with_ds["dataset_name"] = X_test["dataset_name"].values

# 9. HYPERPARAMETER TUNING — LODO
#    RandomForest n_jobs=1 — grid search handles parallelism.
param_grid = {
    "n_estimators":      [500, 1000, 2000],
    "max_depth":         [5, 10, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
    "max_features":      ["sqrt", "log2", 0.01, 0.05, 0.1, 0.2, 0.3],
    "ccp_alpha":         [0.0, 1e-4, 1e-3],
    "class_weight":      ["balanced", "balanced_subsample"],
    "max_samples":       [0.5, 0.7, 0.9, None],
    "criterion":         ["gini", "entropy"],
}

base_model = RandomForestClassifier(
    random_state = 42,
    n_jobs       = 1,
)

X_train_df_lodo = X_train_df.copy()
X_train_df_lodo["dataset_name"] = X_train["dataset_name"].values

best_model, tuning_df = tune_hyperparameters(
    X          = X_train_df_lodo,
    y          = y_train,
    model      = base_model,
    param_grid = param_grid,
    strategy   = "lodo",
    params     = {
        "dataset_col":    "dataset_name",
        "search_ratio":   0.80,
        "grid_threshold": 0,
        "n_jobs":         N_JOBS,
        "output_csv":      os.path.join(OUT_DIR, "tuning_all_combinations.csv"),
        "best_params_csv": os.path.join(OUT_DIR, "tuning_best_params.csv"),
    },
)

# 10. TRAIN FINAL MODEL — LODO validation
result = train_model(
    X        = X_train_df_lodo,
    y        = y_train,
    model    = best_model,
    strategy = "lodo",
    params   = {
        "dataset_col":    "dataset_name",
        "n_jobs":         1,
        "validation_csv": os.path.join(OUT_DIR, "validation_metrics.csv"),
        "test_csv":       os.path.join(OUT_DIR, "test_metrics.csv"),
        "X_test":         X_test_df_with_ds,
        "y_test":         y_test,
    },
)

# 11. FINAL RESULTS — overall + per dataset
val_metrics = result["metrics"]
test_prob   = best_model.predict_proba(X_test_df)[:, 1]
test_auroc  = roc_auc_score(y_test, test_prob)
test_acc    = accuracy_score(y_test, best_model.predict(X_test_df))

print("\n" + "=" * 55)
print("  FINAL RESULTS — OVERALL")
print("=" * 55)
print(f"  Val  AUROC : {val_metrics['auroc']:.4f}")
print(f"  Test AUROC : {test_auroc:.4f}")
print(f"  Test Acc   : {test_acc:.4f}")

print("\n" + "=" * 70)
print("  TEST RESULTS — PER DATASET")
print("=" * 70)
print(f"  {'Dataset':<30} {'n':>5}  {'Control':>9}  {'CRC':>6}  "
      f"{'AUROC':>8}  {'Accuracy':>10}")
print(f"  {'─'*67}")

for ds in sorted(X_test["dataset_name"].unique()):
    mask   = X_test["dataset_name"].values == ds
    X_ds   = X_test_df[mask]
    y_ds   = y_test[mask]

    if len(y_ds.unique()) < 2:
        print(f"  {ds:<30} — skipped (one class only)")
        continue

    prob_ds = best_model.predict_proba(X_ds)[:, 1]
    pred_ds = best_model.predict(X_ds)
    auc_ds  = roc_auc_score(y_ds, prob_ds)
    acc_ds  = accuracy_score(y_ds, pred_ds)
    n_ctrl  = int((y_ds == 0).sum())
    n_crc   = int((y_ds == 1).sum())

    print(f"  {ds:<30} {len(y_ds):>5}  {n_ctrl:>9}  {n_crc:>6}  "
          f"{auc_ds:>8.4f}  {acc_ds:>10.4f}")

print(f"  {'─'*67}")
print(f"  {'OVERALL':<30} {len(y_test):>5}  "
      f"{int((y_test==0).sum()):>9}  {int(y_test.sum()):>6}  "
      f"{test_auroc:>8.4f}  {test_acc:>10.4f}")

# 12. PLOTS
fig_roc = plot_auroc(
    X_test_df, y_test, best_model,
    label="Random Forest (Train/Val)"
)
fig_roc.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=300, bbox_inches="tight")

fig_pr = plot_precision_recall(
    X_test_df, y_test, best_model,
    label="Random Forest (Train/Val)"
)
fig_pr.savefig(os.path.join(OUT_DIR, "pr_curve.png"), dpi=300, bbox_inches="tight")

fig_cm = plot_confusion_matrix(
    X_test_df, y_test, best_model,
    title     = "Confusion Matrix — Test Set (Random Forest)",
    save_path = os.path.join(OUT_DIR, "confusion_matrix.png"),
)

# 13. OUTPUT FILES
outputs = {
    "tuning_all_combinations.csv": "all hyperparameter combinations + AUROC",
    "tuning_best_params.csv":      "best hyperparameters + best AUROC",
    "validation_metrics.csv":      "validation set metrics",
    "test_metrics.csv":            "per-dataset + overall test metrics",
    "roc_curve.png":               "ROC curve with 95% CI",
    "pr_curve.png":                "Precision-Recall curve",
    "confusion_matrix.png":        "confusion matrix on test set",
}
print("\n" + "=" * 60)
print("  OUTPUT FILES")
print("=" * 60)
for fname, desc in outputs.items():
    fpath  = os.path.join(OUT_DIR, fname)
    exists = "✅" if os.path.isfile(fpath) else "❌ not created"
    print(f"  {exists}  {fname:<38} {desc}")

print(f"\n✅ Pipeline complete. All outputs saved to: {OUT_DIR}")