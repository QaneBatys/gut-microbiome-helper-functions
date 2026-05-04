"""
Utility functions for gut microbiome data processing.
Helpers functions: 
    >register_metadata_col(*col_names: str) -> None: to add newly created columns on the metadata.
    >_all_meta() -> set[str]: to return all metadata columns
Main Functions:
    filter_datasets( df: pd.DataFrame, dataset_id: list = [], country_list: list = [], ) -> pd.DataFrame: return copy pd that filters out by dataset name, then country 
    prepare_target_class( df: pd.DataFrame, input_attr: str, target_class_label: str, output_attr: str, exclude_labels: list = [], ) -> pd.DataFrame: return a copy pd with additional column of target dataset
    split_dataset( df: pd.DataFrame, target_class_attr: str, strategy: str = "all", ) -> tuple | dict: helps to split dataset either randomly or in the stratified way. 
    relative_abundance( df: pd.DataFrame, feature_cols: list = None, ) -> pd.DataFrame: return the copy of pd that normalizes each samples' sum to 1.
    aggregate_taxa(df: pd.DataFrame, taxa_level: str) -> pd.DataFrame: return a copy of pd where the features are joined by their taxonimical unit
    prevalence_filtering( df: pd.DataFrame, threshold: float, how: str = "union", ) -> pd.DataFrame: return a copy of pd after prevalence filtering.
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections.abc import Iterable

# CONFIGURATION CONSTANTs
DATASET_ID_COL = "dataset_name"
COUNTRY_COL    = "country"
RANDOM_STATE   = 42

# NON FEATURE COLUMNS
METADATA_COLS  = [
    "dataset_name", "sampleID", "subjectID",
    "study_condition", "disease",
    "age", "gender", "country", "BMI", "fobt",
]

# Runtime-registered non-feature columns (e.g. binary targets added by
# prepare_target_class). Populated automatically
_EXTRA_METADATA_COLS: set[str] = set()

def register_metadata_col(*col_names: str) -> None:
    """Register columns as non-feature so all functions ignore them."""
    _EXTRA_METADATA_COLS.update(col_names)

def _all_meta() -> set[str]:
    """Union of fixed METADATA_COLS and any runtime-registered columns."""
    return set(METADATA_COLS) | _EXTRA_METADATA_COLS

# 2.1 
def filter_datasets(
    df: pd.DataFrame,
    dataset_id: list = [],
    country_list: list = [],
) -> pd.DataFrame:
    """
    Return a filtered copy of *df* keeping only rows that match the
    specified datasets or countries.

    Priority: 
    1. Filter by dataset_id
    2. Else Filter by country (case-insensitive).
    3. If both are empty, UserWarning and return full DataFrame.

    Parameters:
    df : pd.DataFrame
    dataset_id : list[str]
    country_list : list[str] (case-insensitive match).

    Returns:
    pd.DataFrame
        Filtered copy with reset index. All original columns and dtypes are preserved.

    Raises: 
    KeyError
        If the expected identifier or country column is absent from *df*.
    TypeError
        If dataset_id or country_list is not list-like.
    """
    # Check the arguments' tyoe 
    if not isinstance(dataset_id, Iterable) or isinstance(dataset_id, str):
        raise TypeError(
            f"dataset_id must be a list-like object, got {type(dataset_id).__name__}."
        )

    if not isinstance(country_list, Iterable) or isinstance(country_list, str):
        raise TypeError(
            f"country_list must be a list-like object, got {type(country_list).__name__}."
        )

    # Check if the dataset is correct 
    if DATASET_ID_COL not in df.columns:
        raise KeyError(
            f"Expected dataset identifier column '{DATASET_ID_COL}' not found in DataFrame."
        )
    if COUNTRY_COL not in df.columns:
        raise KeyError(
            f"Expected country column '{COUNTRY_COL}' not found in DataFrame."
        )

    # Filtering Logic
    if dataset_id:                          
        mask = df[DATASET_ID_COL].isin(dataset_id)
        return df[mask].reset_index(drop=True)

    if country_list:                       
        country_list_lower = [c.lower() for c in country_list]
        mask = df[COUNTRY_COL].str.lower().isin(country_list_lower)
        return df[mask].reset_index(drop=True)

    warnings.warn(
        "Both dataset_id and country_list are empty. "
        "Returning the original DataFrame unchanged.",
        UserWarning,
        stacklevel=2, # caller's error
    )
    return df.copy().reset_index(drop=True)

# 2.2
def prepare_target_class(
    df: pd.DataFrame,
    input_attr: str,
    target_class_label: str,
    output_attr: str,
    exclude_labels: list = [],
) -> pd.DataFrame:
    """
    Construct a binary target column and append it to a copy of *df*.

    Parameters: 
    df : pd.DataFrame: Must contain the column *input_attr*.
    input_attr : str: column name
    target_class_label : str: value to change
    output_attr : str: new column for binary value
    exclude_labels : list[str]: not needed values for binarization. 

    Returns:
    pd.DataFrame
        Copy of *df* (post-exclusion) with an additional integer column
        *output_attr* (values ``0`` / ``1``) and a reset index.

    Raises:
    ValueError
        If *input_attr* is not a column in *df*.
    ValueError
        If *target_class_label* is absent after applying *exclude_labels*.
    """
    # Check if the input attribute exist. 
    if input_attr not in df.columns:
        raise ValueError(
            f"Column '{input_attr}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    result = df.copy()

    # Excluding labels that are not needed 
    if exclude_labels:
        result = result[~result[input_attr].isin(exclude_labels)]

    # Check if the target label in the column
    if target_class_label not in result[input_attr].values:
        raise ValueError(
            f"target_class_label '{target_class_label}' not found in column "
            f"'{input_attr}' after applying exclude_labels={exclude_labels}."
        )

    # Warn if it is overwriting existing columns
    if output_attr in result.columns:
        warnings.warn(
            f"Column '{output_attr}' already exists and will be overwritten.",
            UserWarning,
            stacklevel=2,
        )

    # Binarization 
    result[output_attr] = (result[input_attr] == target_class_label).astype(int)

    # Auto-register so downstream functions never treat this as a feature
    register_metadata_col(output_attr)

    return result.reset_index(drop=True)

# 2.3  
def split_dataset(
    df: pd.DataFrame,
    target_class_attr: str,
    strategy: str = "all",
) -> tuple | dict:
    """
    Split *df* into train/test feature matrices and target vectors.

    Stratified 80/20 sampling is always applied to preserve class
    proportions.  
    Feature columns are inferred by excluding ``METADATA_COLS`` and 
    *target_class_attr* from the DataFrame.

    Parameters:
    df : pd.DataFrame
    target_class_attr : str: name of binary target column 
    strategy : {'all', 'individual'}
        ``'all'``        — single 80/20 split over the whole DataFrame;
                           returns one ``(X_train, X_test, y_train, y_test)``.
        ``'individual'`` — independent 80/20 split per dataset
                           (identified by ``DATASET_ID_COL``); returns a
                           ``dict[str, tuple]`` mapping each dataset id to
                           its own ``(X_train, X_test, y_train, y_test)``.

    Returns:
    tuple
        ``(X_train, X_test, y_train, y_test)`` when ``strategy='all'``.
        X_train and X_test include all metadata columns + feature columns.
    dict[str, tuple]
        Mapping of dataset id → ``(X_train, X_test, y_train, y_test)``
        when ``strategy='individual'``.
        X_train and X_test include all metadata columns + feature columns.

    Raises:
    ValueError
        If *strategy* is not ``'all'`` or ``'individual'``.
    ValueError
        If *target_class_attr* is not present in *df*.
    """
    _VALID_STRATEGIES = {"all", "individual"}

    # Check if the strategy is valid
    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {_VALID_STRATEGIES}, got '{strategy}'."
        )

    # Check if target class is inside pandas
    if target_class_attr not in df.columns:
        raise ValueError(
            f"target_class_attr '{target_class_attr}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # All columns except target — metadata + features both included in X
    X_cols = [c for c in df.columns if c != target_class_attr]

    def _single_split(data: pd.DataFrame) -> tuple:
        """80/20 stratified split for a given sub-DataFrame."""
        X = data[X_cols]
        y = data[target_class_attr]
        return train_test_split(
            X, y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    if strategy == "all":
        return _single_split(df)

    if DATASET_ID_COL not in df.columns:
        raise KeyError(
            f"DATASET_ID_COL '{DATASET_ID_COL}' not found in DataFrame; "
            "required for strategy='individual'."
        )

    results: dict[str, tuple] = {}
    for dataset_id, group in df.groupby(DATASET_ID_COL):
        results[dataset_id] = _single_split(group.reset_index(drop=True))

    return results

# 2.4 
def relative_abundance(
    df: pd.DataFrame,
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    Convert raw taxon count columns to relative abundance (row-wise).

    Each sample's feature values are divided by the sample's total count
    sum so that every row sums to ``1.0`` over *feature_cols*.
    Non-feature metadata columns are left unchanged.

    Parameters:
    df : pd.DataFrame
    feature_cols : list[str] | None
        Explicit list of taxon column names to transform.  If ``None``,
        columns are inferred by excluding ``METADATA_COLS`` from *df*.

    Returns:
    pd.DataFrame
        Copy of *df* with taxon columns with float relative abundances.

    Raises:
    ValueError
        If any name in *feature_cols* is not a column in *df*.
    ValueError
        If any count value in the feature columns is negative.
    """
    result = df.copy()

    # Extracting feature columns 
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in _all_meta()]
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"The following feature_cols were not found in DataFrame: {missing}"
            )

    feature_data = result[feature_cols]

    # Checking the negative value
    if (feature_data < 0).any(axis=None):
        neg_cols = feature_data.columns[(feature_data < 0).any()].tolist()
        raise ValueError(
            f"Negative values detected in feature columns: {neg_cols}. "
            "Count data must be non-negative."
        )

    # Row wise normalization
    row_sums = feature_data.sum(axis=1)

    # Warnning about zero-sum samples (would produce NaN row)
    zero_sum_idx = row_sums[row_sums == 0].index.tolist()
    if zero_sum_idx:
        warnings.warn(
            f"{len(zero_sum_idx)} sample(s) have a total count of zero and will "
            f"produce NaN relative abundances. Affected indices: {zero_sum_idx}",
            UserWarning,
            stacklevel=2,
        )

    result[feature_cols] = feature_data.div(row_sums, axis=0)

    return result


# 2.5 
# Maps accepted rank names to their 0-based position in a semicolon-delimited
# lineage string (k__...; p__...; c__...; o__...; f__...; g__...; s__...)
_TAXA_LEVEL_INDEX: dict[str, int] = {
    "kingdom": 0,
    "phylum":  1,
    "class":   2,
    "order":   3,
    "family":  4,
    "genus":   5,
    "species": 6,
}

def aggregate_taxa(df: pd.DataFrame, taxa_level: str) -> pd.DataFrame:
    """
    Aggregate (sum) abundance columns to the specified taxonomic rank.

    Taxon column names must follow the semicolon-delimited lineage
    format, e.g.::
        k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;...

    The function parses each feature column name, extracts the token at
    the requested rank position, strips the two-character rank prefix
    (``k__``, ``p__``, etc.), and sums all columns that map to the same
    label.

    Parameters:
    df : pd.DataFrame
    taxa_level : str
        Taxonomic rank at which to aggregate.  One of:
        ``'kingdom'``, ``'phylum'``, ``'class'``, ``'order'``,
        ``'family'``, ``'genus'``, ``'species'``.

    Returns:
    pd.DataFrame
        Metadata columns preserved; one aggregated column per unique
        taxon label at *taxa_level* 

    Raises:
    ValueError
        If *taxa_level* is not one of the accepted rank strings.
    ValueError
        If no feature columns can be parsed to the specified rank.
    """
    # Check if the taxa level is valid
    taxa_level = taxa_level.lower()
    if taxa_level not in _TAXA_LEVEL_INDEX:
        raise ValueError(
            f"taxa_level must be one of {list(_TAXA_LEVEL_INDEX)}, "
            f"got '{taxa_level}'."
        )

    level_idx = _TAXA_LEVEL_INDEX[taxa_level]

    # Extracting feature columns
    meta = _all_meta()
    meta_cols    = [c for c in df.columns if c in meta]
    feature_cols = [c for c in df.columns if c not in meta]

    # Parse each feature column name to its rank label
    col_to_label: dict[str, str] = {}
    unparseable: list[str] = []

    for col in feature_cols:
        parts = col.split("|")
        if len(parts) > level_idx:
            token = parts[level_idx].strip()
            label = token[3:] if token[1:3] == "__" else token
            col_to_label[col] = label
        else:
            unparseable.append(col)

    if unparseable:
        warnings.warn(
            f"{len(unparseable)} feature column(s) could not be parsed to rank "
            f"'{taxa_level}' and were excluded: {unparseable[:10]}"
            f"{'...' if len(unparseable) > 10 else ''}",
            UserWarning,
            stacklevel=2,
        )

    if not col_to_label:
        raise ValueError(
            f"No feature columns could be parsed to rank '{taxa_level}'. "
            "Check that column names follow the semicolon-delimited lineage format."
        )

    # Sum columns with same name
    meta_df      = df[meta_cols].copy()
    feature_df   = df[list(col_to_label.keys())].copy()
    feature_df   = feature_df.rename(columns=col_to_label)

    # Groupby on the transposed frame sums columns with identical names
    aggregated   = feature_df.T.groupby(level=0).sum().T

    return pd.concat([meta_df, aggregated], axis=1).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2.6  prevalence_filtering
# ---------------------------------------------------------------------------
def prevalence_filtering(
    df: pd.DataFrame,
    threshold: float,
    how: str = "union",
) -> pd.DataFrame:
    """
    Remove low-prevalence taxa using per-dataset prevalence estimates.

    For each dataset (identified by ``DATASET_ID_COL``) the prevalence of
    a taxon is defined as the fraction of samples in which its abundance
    is strictly greater than zero.  The *how* parameter controls how
    per-dataset decisions are combined across datasets.

    Parameters:
    df : pd.DataFrame
    threshold : float, (0, 1]
        Minimum prevalence fraction to retain a taxon.
    how : {'union', 'intersect'}
        ``'union'``     — retain a taxon if it passes in **at least one**
                          dataset.
        ``'intersect'`` — retain a taxon only if it passes in **every**
                          dataset.

    Returns:
    pd.DataFrame
        Copy of *df* with only surviving taxa columns plus all metadata
        columns.  Index is reset.

    Raises:
    ValueError
        If *threshold* is not in (0, 1].
    ValueError
        If *how* is not ``'union'`` or ``'intersect'``.
    KeyError
        If ``DATASET_ID_COL`` is absent from *df*.
    """
    # Threshold validation
    if not (0 < threshold <= 1):
        raise ValueError(
            f"threshold must be in (0, 1], got {threshold}."
        )
    # Strategy name validation
    if how not in {"union", "intersect"}:
        raise ValueError(
            f"how must be 'union' or 'intersect', got '{how}'."
        )
    # Dataset name validation
    if DATASET_ID_COL not in df.columns:
        raise KeyError(
            f"DATASET_ID_COL '{DATASET_ID_COL}' not found in DataFrame."
        )

    # Extracting feature.
    feature_cols = [c for c in df.columns if c not in _all_meta()]

    # Per-dataset prevalence 
    # Builds a DataFrame: rows = datasets, columns = taxa
    # Each cell = fraction of samples with abundance > 0
    prev_df = df.groupby(DATASET_ID_COL)[feature_cols].apply(lambda g: (g > 0).mean())
    passes  = prev_df >= threshold                          # bool: does each taxon pass?

    # Aggreagation type
    if how == "union":
        keep_mask = passes.any(axis=0)          # True if passes in ≥1 dataset
    else:  
        keep_mask = passes.all(axis=0)          # True only if passes in all datasets

    surviving_taxa = keep_mask[keep_mask].index.tolist()

    # Rebuilding
    meta_cols    = [c for c in df.columns if c in _all_meta()]
    kept_cols    = meta_cols + surviving_taxa

    return df[kept_cols].copy().reset_index(drop=True)