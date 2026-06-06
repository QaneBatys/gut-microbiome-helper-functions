### GUT MICROBIOME DATA HELPER FUNCTIONS ###
-----------------------------------------------------------------------------------------

The purpose was to create helper functions and organize them into libraries.

- pd_utils.py: contains utility functions for gut microbiome data processing. (register_metadata_col, _all_meta, filter_datasets, prepare_target_class, split_dataset, relative_abundance, aggregate_taxa, prevalence_filtering)
- pd_transform.py: contains 9tility functions for gut microbiome data preprocessing: transformations, batch correction, and data quality checks. (log_transform, clr_transform, remove_batch_effects, check_data_quality)
- pd_ml.py: contains machine learning utilities for gut microbiome classification. (train_model, tune_hyperparameters)
- pd_assess.py: contains model assessment utilities for gut microbiome classification. (_check_binary_target, _check_feature_alignment, _summary_table, assess_model, plot_auroc, plot_precision_recall, stratify_by_age)


