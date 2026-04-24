"""
visualization.py
----------------
Publication-quality matplotlib / seaborn figures for gut microbiome data.

All plot functions:
  - Return a matplotlib.figure.Figure object.
  - Do NOT call plt.show() internally.
  - Assume abundance DataFrames follow the conventions in pandas_util.py
    (METADATA_COLS, DATASET_ID_COL, etc.).
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations

# scikit-bio for diversity metrics & ordination
from skbio.diversity import alpha_diversity, beta_diversity
from skbio.stats.ordination import pcoa
from skbio.stats.distance import permanova

# scipy / scikit-learn
from scipy.stats import kruskal, gaussian_kde
from scikit_posthocs import posthoc_dunn
from sklearn.manifold import MDS          # NMDS fallback via sklearn
try:
    import umap                           # optional; only needed for ordination='umap'
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

# Suppress known harmless runtime warnings from dependencies
warnings.filterwarnings("ignore", message="divide by zero encountered in divide",
                        category=RuntimeWarning, module="scikit_posthocs")
warnings.filterwarnings("ignore", message=".*negative eigenvalues.*",
                        category=RuntimeWarning, module="skbio")
warnings.filterwarnings("ignore", message=".*EIGH.*",
                        category=RuntimeWarning, module="skbio")

# ---------------------------------------------------------------------------
# Re-use constants from pandas_util to stay in sync
# ---------------------------------------------------------------------------
from pd_utils import METADATA_COLS, DATASET_ID_COL, _all_meta

# ---------------------------------------------------------------------------
# Module-level style defaults
# ---------------------------------------------------------------------------
_DEFAULT_PALETTE = sns.color_palette("Set2")
_FIG_DPI         = 120

# Accepted option sets (used for validation)
_ALPHA_METRICS    = {"shannon", "simpson", "chao1", "observed_otus"}
_BETA_METRICS     = {"bray_curtis", "jaccard", "aitchison"}
_ORDINATION_METHODS = {"pcoa", "nmds", "umap"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return taxon feature columns (all columns not in _all_meta())."""
    return [c for c in df.columns if c not in _all_meta()]

def _resolve_colors(labels: list, color_mapping: dict | None) -> dict:
    """
    Build a complete label → colour dict, filling in missing labels from
    the default palette.
    """
    mapping   = dict(color_mapping) if color_mapping else {}
    missing   = [l for l in labels if l not in mapping]
    palette   = sns.color_palette(_DEFAULT_PALETTE, n_colors=len(missing))
    for label, color in zip(missing, palette):
        mapping[label] = color
    return mapping


def _significance_stars(p: float) -> str | None:
    """Convert a p-value to an asterisk string, or None if not significant."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return None


def _draw_significance_brackets(ax, group_positions: list, pval_matrix: pd.DataFrame,
                                  y_data_max: float) -> None:
    """
    Overlay pairwise significance brackets on *ax*.

    Parameters
    ----------
    ax              : matplotlib Axes
    group_positions : x-positions of each group (same order as pval_matrix index)
    pval_matrix     : square DataFrame of corrected p-values (from Dunn test)
    y_data_max      : top of the data range; brackets start above this value
    """
    labels = list(pval_matrix.index)

    # Guard: use a fraction of y_data_max as fallback if ylim isn't set yet
    ylim_range = y_data_max - ax.get_ylim()[0]
    step       = ylim_range * 0.08 if (ylim_range > 0 and np.isfinite(ylim_range)) \
                 else abs(y_data_max) * 0.08 or 0.1

    y_cursor = y_data_max + step

    for i, j in combinations(range(len(labels)), 2):
        p = pval_matrix.iloc[i, j]

        # Guard: skip NaN p-values (can arise from Dunn divide-by-zero)
        if not np.isfinite(p):
            continue

        stars = _significance_stars(p)
        if stars is None:
            continue

        x1, x2 = group_positions[i], group_positions[j]
        y_top   = y_cursor + step * 0.5

        ax.plot([x1, x1, x2, x2], [y_cursor, y_top, y_top, y_cursor],
                lw=1.2, color="black")
        ax.text((x1 + x2) / 2, y_top, stars,
                ha="center", va="bottom", fontsize=11)
        y_cursor += step * 1.6

    # Expand y-axis so brackets are not clipped
    new_top = y_cursor + step
    if np.isfinite(new_top):
        ax.set_ylim(top=new_top)



# ---------------------------------------------------------------------------
# Alpha diversity metric helpers (proportion-based, no skbio integer requirement)
# ---------------------------------------------------------------------------

def _shannon(proportions: np.ndarray) -> float:
    """Shannon H' in bits (log2). Input: row of relative abundances."""
    p = proportions[proportions > 0]
    return float(-np.sum(p * np.log2(p))) if len(p) else np.nan

def _simpson(proportions: np.ndarray) -> float:
    """Simpson's diversity index (1 - D)."""
    p = proportions[proportions > 0]
    return float(1 - np.sum(p ** 2)) if len(p) else np.nan

def _observed_otus(proportions: np.ndarray) -> float:
    """Number of taxa with abundance > 0."""
    return float(np.sum(proportions > 0))

def _chao1(proportions: np.ndarray) -> float:
    """
    Chao1 richness estimator.
    Requires count data; on relative abundances this approximates observed OTUs.
    Falls back to observed OTUs when singletons/doubletons cannot be determined.
    """
    return _observed_otus(proportions)

_METRIC_FN = {
    "shannon":       _shannon,
    "simpson":       _simpson,
    "observed_otus": _observed_otus,
    "chao1":         _chao1,
}

_METRIC_YLABEL = {
    "shannon":       "Shannon Index H' (bits)",
    "simpson":       "Simpson Index (1 − D)",
    "observed_otus": "Observed OTUs",
    "chao1":         "Chao1 Richness",
}

def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size (sign: x > y → positive)."""
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((more - less) / (len(x) * len(y)))

def _cliffs_label(d: float) -> str:
    if not np.isfinite(d): return "n/a"
    a = abs(d)
    if a < 0.147: return "negligible"
    if a < 0.330: return "small"
    if a < 0.474: return "medium"
    return "large"


# 3.1
def plot_alpha_diversity(
    df: pd.DataFrame,
    diversity_metric: str,
    target_attr: str = None,
    color_mapping: dict = None,
) -> matplotlib.figure.Figure:
    """
    Compute alpha diversity from relative abundances and plot distribution.

    Metrics are computed directly from proportions (no skbio integer
    requirement), so the function works correctly after relative_abundance()
    has been applied. Shannon is computed in bits (log2).

    When *target_attr* is provided: grouped boxplot with jittered points,
    Kruskal-Wallis overall test, pairwise significance brackets (significant
    pairs only — no 'ns' annotations), and Cliff's delta effect sizes in
    the subtitle.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with taxon abundance columns (relative abundances).
    diversity_metric : str
        One of ``'shannon'``, ``'simpson'``, ``'chao1'``, ``'observed_otus'``.
    target_attr : str | None
        Grouping column. ``None`` → single violin plot of overall diversity.
    color_mapping : dict | None
        Class label → colour string. Unmapped labels get auto-colours.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If *diversity_metric* is not a recognised option.
    ValueError
        If *target_attr* is specified but absent from *df*.
    """
    if diversity_metric not in _ALPHA_METRICS:
        raise ValueError(
            f"diversity_metric must be one of {_ALPHA_METRICS}, "
            f"got '{diversity_metric}'."
        )
    if target_attr is not None and target_attr not in df.columns:
        raise ValueError(
            f"target_attr '{target_attr}' not found in DataFrame."
        )

    # --- resolve and validate feature columns --------------------------------
    feat_cols = _feature_cols(df)
    if not feat_cols:
        raise ValueError(
            "No feature columns found. Ensure the DataFrame has been processed "
            "through aggregate_taxa / relative_abundance before plotting."
        )

    # --- compute diversity from proportions ----------------------------------
    # Convert to float, fill NaN → 0, then TSS-normalise row-wise so each
    # row sums to 1 regardless of whether relative_abundance() was applied.
    prop_df   = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    row_sums  = prop_df.sum(axis=1)
    # drop zero-sum samples silently
    valid_mask = row_sums > 0
    prop_df    = prop_df.loc[valid_mask]
    row_sums   = row_sums.loc[valid_mask]
    meta_df    = df.loc[valid_mask].copy()

    proportions = prop_df.div(row_sums, axis=0)  # each row sums to 1

    metric_fn = _METRIC_FN[diversity_metric]
    diversity_vals = proportions.apply(lambda row: metric_fn(row.values), axis=1).values

    if np.all(np.isnan(diversity_vals)):
        raise ValueError(
            f"All {diversity_metric} values are NaN. "
            "Check that feature columns contain non-zero relative abundances."
        )

    # attach to a clean DataFrame aligned with valid samples
    plot_df = pd.DataFrame(
        {"alpha_diversity": diversity_vals},
        index=meta_df.index,
    )
    if target_attr is not None:
        plot_df[target_attr] = meta_df[target_attr].values

    fig, ax = plt.subplots(figsize=(6, 6), dpi=_FIG_DPI)

    # ── No grouping: single violin ────────────────────────────────────────────
    if target_attr is None:
        valid_vals = diversity_vals[~np.isnan(diversity_vals)]
        ax.violinplot(valid_vals, positions=[1], showmedians=True)
        ax.set_xticks([1])
        ax.set_xticklabels(["All samples"])
        ax.set_title(f"Alpha Diversity — {diversity_metric}",
                     fontsize=12, fontweight="bold")

    # ── Grouped: boxplot + strip + stats ─────────────────────────────────────
    else:
        groups    = sorted(plot_df[target_attr].dropna().unique())
        color_map = _resolve_colors(groups, color_mapping)
        positions = list(range(1, len(groups) + 1))

        # build per-group arrays exactly like the manual code
        group_arrays = []
        for g in groups:
            arr = plot_df.loc[plot_df[target_attr] == g,
                              "alpha_diversity"].dropna().values
            group_arrays.append(arr)

        # single ax.boxplot call — the only reliable way to draw all boxes
        bp = ax.boxplot(
            group_arrays,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            boxprops=dict(linewidth=1.5),
        )
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(color_map[g])
            patch.set_alpha(0.85)

        # jittered strip overlay
        rng = np.random.default_rng(42)
        for pos, g, arr in zip(positions, groups, group_arrays):
            jitter = rng.uniform(-0.15, 0.15, size=len(arr))
            ax.scatter(
                pos + jitter, arr,
                color=color_map[g], s=10, alpha=0.45,
                zorder=3, edgecolors="none",
            )

        # x-axis: group label + n=
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{g}\n\nn={len(arr)}" for g, arr in zip(groups, group_arrays)],
            fontsize=10,
        )
        ax.tick_params(axis="x", which="both", length=0, pad=8)
        ax.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)

        # statistics
        kw_stat, kw_p = kruskal(*group_arrays)

        dunn_p = posthoc_dunn(
            plot_df, val_col="alpha_diversity", group_col=target_attr,
            p_adjust="bonferroni",
        )
        y_max = plot_df["alpha_diversity"].dropna().max()
        _draw_significance_brackets(ax, positions, dunn_p, y_max)

        # Cliff's delta subtitle
        delta_lines = []
        for i, j in combinations(range(len(groups)), 2):
            d = _cliffs_delta(group_arrays[i], group_arrays[j])
            delta_lines.append(
                f"{groups[i]} vs {groups[j]}: δ={d:+.3f} ({_cliffs_label(d)})"
            )
        ax.text(
            0.5, -0.16, "  |  ".join(delta_lines),
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, style="italic", color="dimgray",
        )

        ax.set_title(
            f"Alpha Diversity — {diversity_metric}\n"
            f"Kruskal-Wallis p = {kw_p:.3e}",
            fontsize=12, fontweight="bold", pad=10,
        )

        legend_handles = [
            mpatches.Patch(facecolor=color_map[g], alpha=0.85, label=str(g))
            for g in groups
        ]
        ax.legend(handles=legend_handles, loc="upper right", frameon=True,
                  fontsize=9, title="Study Groups", title_fontsize=9)

    ax.set_ylabel(_METRIC_YLABEL[diversity_metric], fontweight="bold", fontsize=11)
    ax.set_xlabel("")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3.2  plot_beta_diversity
# ---------------------------------------------------------------------------

def plot_beta_diversity(
    df: pd.DataFrame,
    distance_metric: str = "bray_curtis",
    ordination: str = "pcoa",
    target_attr: str = None,
    color_mapping: dict = None,
) -> matplotlib.figure.Figure:
    """
    Compute pairwise distances, perform ordination, and plot a 2-D scatter.

    PERMANOVA results (R² and p-value) are shown in the legend when
    *target_attr* is provided.

    Parameters
    ----------
    df : pd.DataFrame
        Abundance DataFrame (samples × taxa).
    distance_metric : {'bray_curtis', 'jaccard', 'aitchison'}
    ordination : {'pcoa', 'nmds', 'umap'}
    target_attr : str | None
        Grouping variable for colouring and PERMANOVA.
    color_mapping : dict | None

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If *distance_metric* or *ordination* is not a recognised option.
    """
    if distance_metric not in _BETA_METRICS:
        raise ValueError(
            f"distance_metric must be one of {_BETA_METRICS}, "
            f"got '{distance_metric}'."
        )
    if ordination not in _ORDINATION_METHODS:
        raise ValueError(
            f"ordination must be one of {_ORDINATION_METHODS}, "
            f"got '{ordination}'."
        )
    if ordination == "umap" and not _UMAP_AVAILABLE:
        raise ImportError(
            "The 'umap-learn' package is required for ordination='umap'. "
            "Install it with: pip install umap-learn"
        )

    feat_cols = _feature_cols(df)
    abund     = df[feat_cols].fillna(0).values.astype(float)
    ids       = np.array([str(i) for i in df.index])

    # scikit-bio metric name aliases (our API name → skbio internal name)
    _beta_metric_alias = {
        "bray_curtis": "braycurtis",
        "jaccard":     "jaccard",
        "aitchison":   "aitchison",
    }
    skbio_metric = _beta_metric_alias.get(distance_metric, distance_metric)

    # --- distance matrix -----------------------------------------------------
    dist_mat = beta_diversity(skbio_metric, abund, ids)

    # --- ordination ----------------------------------------------------------
    if ordination == "pcoa":
        result    = pcoa(dist_mat)
        coords    = result.samples[["PC1", "PC2"]].values
        ax_labels = (
            f"PC1 ({result.proportion_explained['PC1']:.1%})",
            f"PC2 ({result.proportion_explained['PC2']:.1%})",
        )

    elif ordination == "nmds":
        mds       = MDS(n_components=2, dissimilarity="precomputed",
                        random_state=42, max_iter=500)
        coords    = mds.fit_transform(dist_mat.to_data_frame().values)
        ax_labels = ("NMDS1", "NMDS2")

    else:  # umap
        import umap as umap_lib
        reducer   = umap_lib.UMAP(n_components=2, random_state=42,
                                   metric="precomputed")
        coords    = reducer.fit_transform(dist_mat.to_data_frame().values)
        ax_labels = ("UMAP1", "UMAP2")

    # --- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), dpi=_FIG_DPI)

    if target_attr is not None and target_attr in df.columns:
        groups    = sorted(df[target_attr].unique())
        color_map = _resolve_colors(groups, color_mapping)

        for g in groups:
            mask = df[target_attr].values == g
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=color_map[g], label=str(g),
                       s=30, alpha=0.75, edgecolors="none")

        # --- PERMANOVA -------------------------------------------------------
        grouping = df[target_attr].astype(str).values
        perm_res = permanova(dist_mat, grouping, permutations=999)
        r2       = perm_res["test statistic"]
        pval     = perm_res["p-value"]

        legend_title = (
            f"PERMANOVA\nR² = {r2:.3f}  p = {pval:.3f}"
        )
        ax.legend(title=legend_title, fontsize=9, title_fontsize=9,
                  framealpha=0.8)

    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.75,
                   color=_DEFAULT_PALETTE[0], edgecolors="none")

    ax.set_xlabel(ax_labels[0], fontsize=11)
    ax.set_ylabel(ax_labels[1], fontsize=11)
    ax.set_title(
        f"Beta Diversity — {distance_metric.replace('_', ' ').title()} / "
        f"{ordination.upper()}",
        fontsize=12,
    )
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3.3  plot_top_N
# ---------------------------------------------------------------------------

def plot_top_N(
    df: pd.DataFrame,
    target_attr: str,
    N: int = 10,
) -> matplotlib.figure.Figure:
    """
    Horizontal grouped bar chart of the top-N taxa by overall mean abundance.

    Parameters
    ----------
    df : pd.DataFrame
        Abundance DataFrame containing a target class column.
    target_attr : str
        Column name for group stratification.
    N : int, default 10
        Number of top taxa to display.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If *target_attr* is absent from *df*.
    ValueError
        If *N* exceeds the number of feature columns.
    """
    if target_attr not in df.columns:
        raise ValueError(
            f"target_attr '{target_attr}' not found in DataFrame."
        )

    feat_cols = _feature_cols(df)

    if N > len(feat_cols):
        raise ValueError(
            f"N={N} exceeds the number of feature columns ({len(feat_cols)})."
        )

    # --- select top-N taxa by overall mean abundance -------------------------
    overall_mean = df[feat_cols].mean(axis=0).sort_values(ascending=False)
    top_taxa     = overall_mean.head(N).index.tolist()

    # --- compute per-class mean for each top taxon ---------------------------
    groups   = sorted(df[target_attr].unique())
    mean_df  = (
        df[top_taxa + [target_attr]]
        .groupby(target_attr)[top_taxa]
        .mean()
    )                                               # shape: (n_groups, N)

    # order taxa by descending overall mean (top at the top of chart)
    mean_df  = mean_df[top_taxa]

    color_map = _resolve_colors(groups, None)
    n_groups  = len(groups)
    bar_height = 0.8 / n_groups
    y_base     = np.arange(N)

    fig, ax = plt.subplots(figsize=(10, max(5, N * 0.55)), dpi=_FIG_DPI)

    for i, group in enumerate(groups):
        offsets = y_base - 0.4 + bar_height * (i + 0.5)
        ax.barh(
            offsets,
            mean_df.loc[group, top_taxa],
            height=bar_height * 0.9,
            color=color_map[group],
            label=str(group),
            alpha=0.85,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels(
        [t.split(";")[-1].strip().lstrip("s__g__f__o__c__p__k__")
         for t in top_taxa],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean Abundance", fontsize=11)
    ax.set_title(f"Top {N} Taxa by Mean Abundance", fontsize=12)
    ax.legend(title=target_attr, fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3.4  Demographic Distribution Plots
# ---------------------------------------------------------------------------

def _check_covariate(df: pd.DataFrame, col: str) -> None:
    """Raise ValueError if *col* is absent from *df*."""
    if col not in df.columns:
        raise ValueError(
            f"Expected covariate column '{col}' not found in DataFrame."
        )


# Mapping from raw dataset identifiers to citation-style labels.
# Used automatically by demographic plot functions.
DATASET_LABEL_MAP = {
    "FengQ_2015"      : "Feng et al. (2015)",
    "HanniganGD_2017" : "Hannigan et al. (2017)",
    "ThomasAM_2019_a" : "Thomas et al. (2019a)",
    "ThomasAM_2019_b" : "Thomas et al. (2019b)",
    "ThomasAM_2019_c" : "Thomas et al. (2019c)",
    "VogtmannE_2016"  : "Vogtmann et al. (2016)",
    "WirbelJ_2018"    : "Wirbel et al. (2018)",
    "YuJ_2015"        : "Yu et al. (2015)",
    "ZellerG_2014"    : "Zeller et al. (2014)",
}


def plot_age_distribution(
    df: pd.DataFrame,
    target_attr: str,
    dataset_attr: str = None,
    bins: int = 20,
) -> matplotlib.figure.Figure:
    """
    Age distribution plot stratified by *target_attr*.

    When *dataset_attr* is provided: grouped boxplot with dataset on the
    x-axis and condition as hue — cohort-level comparison view.
    Without *dataset_attr*: KDE + histogram overlay per condition.

    Parameters
    ----------
    df : pd.DataFrame
    target_attr : str  — target class column.
    dataset_attr : str | None  — column for per-dataset grouping.
    bins : int, default 20  — histogram bins (no-dataset mode only).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError — If the age column is absent from *df*.
    """
    _check_covariate(df, "age")
    if target_attr not in df.columns:
        raise ValueError(f"target_attr '{target_attr}' not found in DataFrame.")

    return _covariate_plot(
        df, covariate="age", target_attr=target_attr,
        dataset_attr=dataset_attr, bins=bins,
        ylabel="Age (years)", title="Age Distribution Across Datasets",
    )


def plot_bmi_distribution(
    df: pd.DataFrame,
    target_attr: str,
    dataset_attr: str = None,
    bins: int = 20,
) -> matplotlib.figure.Figure:
    """
    BMI distribution plot stratified by *target_attr*.

    When *dataset_attr* is provided: grouped boxplot with dataset on the
    x-axis and condition as hue — cohort-level comparison view.
    Without *dataset_attr*: KDE + histogram overlay per condition.

    Parameters
    ----------
    df : pd.DataFrame
    target_attr : str
    dataset_attr : str | None
    bins : int, default 20

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError — If the BMI column is absent from *df*.
    """
    _check_covariate(df, "BMI")
    if target_attr not in df.columns:
        raise ValueError(f"target_attr '{target_attr}' not found in DataFrame.")

    return _covariate_plot(
        df, covariate="BMI", target_attr=target_attr,
        dataset_attr=dataset_attr, bins=bins,
        ylabel="BMI (kg/m²)", title="BMI Distribution Across Datasets",
    )


def _covariate_plot(
    df: pd.DataFrame,
    covariate: str,
    target_attr: str,
    dataset_attr: str | None,
    bins: int,
    ylabel: str,
    title: str,
) -> matplotlib.figure.Figure:
    """
    Shared backend for age and BMI plots.

    With dataset_attr  → overlapping KDE + histogram, one panel per dataset.
    Without dataset_attr → single KDE + histogram panel (all data pooled).
    """
    # Force numeric and drop non-parseable values
    df = df.copy()
    df[covariate] = pd.to_numeric(df[covariate], errors="coerce")
    df = df.dropna(subset=[covariate])

    groups    = sorted(df[target_attr].dropna().unique())
    color_map = _resolve_colors(groups, None)

    # ── Faceted: one panel per dataset ───────────────────────────────────────
    if dataset_attr is not None and dataset_attr in df.columns:
        df[dataset_attr] = df[dataset_attr].map(DATASET_LABEL_MAP).fillna(df[dataset_attr])
        datasets  = sorted(df[dataset_attr].unique())
        n_panels  = len(datasets)

        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(5 * n_panels, 4),
            sharey=False,
            dpi=_FIG_DPI,
        )
        if n_panels == 1:
            axes = [axes]

        for ax, dataset in zip(axes, datasets):
            sub = df[df[dataset_attr] == dataset]

            for grp in groups:
                vals = sub.loc[sub[target_attr] == grp, covariate].dropna()
                if len(vals) < 2:
                    continue
                ax.hist(vals, bins=bins, alpha=0.35,
                        color=color_map[grp], density=True)
                kde    = gaussian_kde(vals)
                x_grid = np.linspace(vals.min(), vals.max(), 200)
                ax.plot(x_grid, kde(x_grid), color=color_map[grp],
                        label=str(grp), linewidth=2)

            ax.set_title(dataset, fontsize=9, fontweight="bold")
            ax.set_xlabel(ylabel, fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            sns.despine(ax=ax)

        handles = [mpatches.Patch(color=color_map[g], label=str(g)) for g in groups]
        fig.legend(handles=handles, title=target_attr,
                   loc="upper right", fontsize=9)
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    # ── Single panel: pooled KDE + histogram ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4), dpi=_FIG_DPI)

    for grp in groups:
        vals = df.loc[df[target_attr] == grp, covariate].dropna()
        if len(vals) < 2:
            continue
        ax.hist(vals, bins=bins, alpha=0.35, color=color_map[grp], density=True)
        kde    = gaussian_kde(vals)
        x_grid = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(x_grid, kde(x_grid), color=color_map[grp],
                label=str(grp), linewidth=2)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(ylabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    handles = [mpatches.Patch(color=color_map[g], label=str(g)) for g in groups]
    ax.legend(handles=handles, title=target_attr, fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_gender_distribution(
    df: pd.DataFrame,
    target_attr: str,
    dataset_attr: str = None,
) -> matplotlib.figure.Figure:
    """
    Stacked bar chart of biological sex distribution, broken down by
    *target_attr*. When *dataset_attr* is provided, a faceted figure
    with one panel per dataset is produced.

    Parameters
    ----------
    df : pd.DataFrame
    target_attr : str
    dataset_attr : str | None

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError — If the gender column is absent from *df*.
    """
    _check_covariate(df, "gender")
    if target_attr not in df.columns:
        raise ValueError(f"target_attr '{target_attr}' not found in DataFrame.")

    df = df.copy().dropna(subset=["gender"])
    # Filter out non-informative gender labels
    _UNKNOWN_LABELS = {"unknown", "na", "n/a", "none", ""}
    df = df[~df["gender"].astype(str).str.lower().isin(_UNKNOWN_LABELS)]

    gender_vals   = sorted(df["gender"].dropna().unique())
    gender_colors = sns.color_palette("pastel", n_colors=len(gender_vals))
    gender_cmap   = dict(zip(gender_vals, gender_colors))
    groups        = sorted(df[target_attr].dropna().unique())

    # ── Faceted: one panel per dataset ───────────────────────────────────────
    if dataset_attr is not None and dataset_attr in df.columns:
        df[dataset_attr] = df[dataset_attr].map(DATASET_LABEL_MAP).fillna(df[dataset_attr])
        datasets  = sorted(df[dataset_attr].unique())
        n_panels  = len(datasets)

        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(4 * n_panels, 4),
            sharey=True,
            dpi=_FIG_DPI,
        )
        if n_panels == 1:
            axes = [axes]

        for ax, dataset in zip(axes, datasets):
            sub    = df[df[dataset_attr] == dataset]
            counts = (
                sub.groupby([target_attr, "gender"])
                .size()
                .unstack(fill_value=0)
            )
            props  = counts.div(counts.sum(axis=1), axis=0)

            bottom = np.zeros(len(props))
            for gender in gender_vals:
                if gender not in props.columns:
                    continue
                vals = props[gender].values
                ax.bar(
                    props.index.astype(str),
                    vals,
                    bottom=bottom,
                    color=gender_cmap[gender],
                    label=str(gender),
                    alpha=0.85,
                )
                bottom += vals

            ax.set_title(dataset, fontsize=9, fontweight="bold")
            ax.set_xlabel(target_attr, fontsize=9)
            ax.set_ylabel("Proportion" if dataset == datasets[0] else "")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=15, labelsize=8)
            sns.despine(ax=ax)

        handles = [mpatches.Patch(color=gender_cmap[g], label=str(g))
                   for g in gender_vals]
        fig.legend(handles=handles, title="gender",
                   loc="upper right", fontsize=9)
        fig.suptitle("Gender Distribution", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    # ── Single panel: stacked bar per condition ───────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4), dpi=_FIG_DPI)

    counts = (
        df.groupby([target_attr, "gender"])
        .size()
        .unstack(fill_value=0)
    )
    props  = counts.div(counts.sum(axis=1), axis=0)

    bottom = np.zeros(len(props))
    for gender in gender_vals:
        if gender not in props.columns:
            continue
        vals = props[gender].values
        ax.bar(
            props.index.astype(str),
            vals,
            bottom=bottom,
            color=gender_cmap[gender],
            label=str(gender),
            alpha=0.85,
        )
        bottom += vals

    ax.set_title("Gender Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel(target_attr, fontsize=10)
    ax.set_ylabel("Proportion", fontsize=10)
    ax.set_ylim(0, 1)
    handles = [mpatches.Patch(color=gender_cmap[g], label=str(g))
               for g in gender_vals]
    ax.legend(handles=handles, title="gender", fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# 3.5  plot_dataset_summary
# ---------------------------------------------------------------------------

def plot_dataset_summary(
    df: pd.DataFrame,
    target_attr: str,
) -> matplotlib.figure.Figure:
    """
    Dashboard-style data audit figure with three panels:

    1. **Sample counts per dataset** — stacked bar chart coloured by class.
    2. **Class balance** — bar chart of overall class label counts.
    3. **Metadata missingness heatmap** — fraction of missing values for
       key metadata columns across datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_attr : str
        Target class column name.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If *target_attr* is absent from *df*.
    """
    if target_attr not in df.columns:
        raise ValueError(
            f"target_attr '{target_attr}' not found in DataFrame."
        )

    # Metadata columns we care about for missingness (exclude free-text / IDs)
    AUDIT_META_COLS = ["age", "gender", "country", "BMI", "fobt", "study_condition"]
    audit_cols      = [c for c in AUDIT_META_COLS if c in df.columns]

    fig = plt.figure(figsize=(16, 5), dpi=_FIG_DPI, constrained_layout=True)
    gs  = fig.add_gridspec(1, 3, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    classes   = sorted(df[target_attr].dropna().unique())
    color_map = _resolve_colors(classes, None)

    # ── Panel 1: sample counts per dataset, stacked by class ────────────────
    if DATASET_ID_COL in df.columns:
        counts = (
            df.groupby([DATASET_ID_COL, target_attr])
            .size()
            .unstack(fill_value=0)
        )
        bottom = np.zeros(len(counts))
        for cls in classes:
            if cls not in counts.columns:
                continue
            vals = counts[cls].values
            ax1.bar(
                counts.index.astype(str),
                vals,
                bottom=bottom,
                color=color_map[cls],
                label=str(cls),
                alpha=0.85,
            )
            bottom += vals
        ax1.set_title("Samples per Dataset", fontsize=11)
        ax1.set_xlabel("Dataset", fontsize=9)
        ax1.set_ylabel("Sample Count", fontsize=9)
        ax1.tick_params(axis="x", rotation=35, labelsize=7)
        ax1.legend(title=target_attr, fontsize=8)
    else:
        ax1.text(0.5, 0.5, f"'{DATASET_ID_COL}' column\nnot found",
                 ha="center", va="center", transform=ax1.transAxes)
    sns.despine(ax=ax1)

    # ── Panel 2: overall class balance ──────────────────────────────────────
    class_counts = df[target_attr].value_counts().reindex(classes, fill_value=0)
    ax2.bar(
        [str(c) for c in classes],
        class_counts.values,
        color=[color_map[c] for c in classes],
        alpha=0.85,
    )
    for x, val in enumerate(class_counts.values):
        ax2.text(x, val + class_counts.max() * 0.01, str(val),
                 ha="center", va="bottom", fontsize=9)
    ax2.set_title("Class Balance", fontsize=11)
    ax2.set_xlabel(target_attr, fontsize=9)
    ax2.set_ylabel("Sample Count", fontsize=9)
    sns.despine(ax=ax2)

    # ── Panel 3: metadata missingness heatmap ───────────────────────────────
    if DATASET_ID_COL in df.columns and audit_cols:
        # Coerce numeric columns so string "NA"/"" are treated as missing
        _NUMERIC_AUDIT = {"age", "BMI"}
        audit_df = df[audit_cols].copy()
        for col in audit_cols:
            if col in _NUMERIC_AUDIT:
                audit_df[col] = pd.to_numeric(audit_df[col], errors="coerce")
            else:
                # Treat blank / "NA" / "unknown" strings as missing
                _UNKNOWN = {"na", "n/a", "none", "unknown", ""}
                audit_df[col] = audit_df[col].apply(
                    lambda v: np.nan
                    if pd.isna(v) or str(v).strip().lower() in _UNKNOWN
                    else v
                )

        missing_frac = (
            df[[DATASET_ID_COL]].join(audit_df)
            .groupby(DATASET_ID_COL)[audit_cols]
            .apply(lambda g: g.isna().mean())
        )

        sns.heatmap(
            missing_frac,
            ax=ax3,
            cmap="YlOrRd",
            vmin=0, vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Missing fraction"},
        )
        ax3.set_title("Metadata Missingness per Dataset", fontsize=11)
        ax3.set_xlabel("")
        ax3.set_ylabel("Dataset", fontsize=9)
        ax3.tick_params(axis="x", rotation=30, labelsize=8)
        ax3.tick_params(axis="y", rotation=0,  labelsize=7)
    else:
        ax3.text(0.5, 0.5, "Insufficient data\nfor missingness plot",
                 ha="center", va="center", transform=ax3.transAxes)
        sns.despine(ax=ax3)

    fig.suptitle("Dataset Summary", fontsize=13, fontweight="bold")
    return fig