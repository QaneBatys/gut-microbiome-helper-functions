import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 1. LOADING AND TRANSPOSING
df = pd.read_csv(
    "./data/filtered_nine_crc_final.tsv", sep="\t", header=None, index_col=0
)
df_t = df.T
df_t.columns = df_t.columns.astype(str).str.strip()

# 2. CONDITION AND AGE
target_conditions = ["control", "adenoma", "CRC"]
df_filtered = df_t[df_t["study_condition"].isin(target_conditions)].copy()
df_filtered["age"] = pd.to_numeric(df_filtered["age"], errors="coerce")
df_age = df_filtered.dropna(subset=["age"]).copy()

# 3. STYLING
sns.set_style("ticks")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.2

# DATASET NAME MAPPING
DATASET_LABEL_MAP = {
    "FengQ_2015": "Feng et al. (2015)",
    "HanniganGD_2017": "Hannigan et al. (2017)",
    "ThomasAM_2019_a": "Thomas et al. (2019) (a)",
    "ThomasAM_2019_b": "Thomas et al. (2019) (b)",
    "ThomasAM_2019_c": "Thomas et al. (2019) (c)",
    "VogtmannE_2016": "Vogtmann et al. (2016)",
    "WirbelJ_2018": "Wirbel et al. (2018)",
    "YuJ_2015": "Yu et al. (2015)",
    "ZellerG_2014": "Zeller et al. (2014)",
}
df_age["dataset_name"] = (
    df_age["dataset_name"].map(DATASET_LABEL_MAP).fillna(df_age["dataset_name"])
)

#  Control=Red, Adenoma=Blue, CRC=Grey
color_dict = {"control": "#E41A1C", "adenoma": "#377EB8", "CRC": "#4D4D4D"}

# 4. VISUALIZATION
plt.figure(figsize=(14, 7))

ax = sns.boxplot(
    data=df_age,
    x="dataset_name",
    y="age",
    hue="study_condition",
    hue_order=["control", "adenoma", "CRC"],
    palette=color_dict,
    linewidth=1.2,
    fliersize=3,
    width=0.7,
)

datasets = df_age["dataset_name"].unique()
for i in range(len(datasets) - 1):
    plt.axvline(i + 0.5, color="black", lw=0.8, ls=":", alpha=0.3)

# Formatting
plt.title(
    "Age Distribution Across Global CRC Datasets", fontsize=16, weight="bold", pad=25
)
plt.ylabel("Age (Years)", fontsize=12, weight="bold")
plt.xlabel("")
plt.xticks(rotation=0, ha="center", fontsize=9, style="italic")
plt.yticks(fontsize=10)

# Clean up axes
sns.despine()

# Legend
plt.legend(
    title="Study Condition",
    title_fontsize="11",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=True,
    edgecolor="black",
)

plt.tight_layout()
plt.savefig("scientific_age_distribution.png", dpi=300)
print("Scientific graph saved as 'scientific_age_distribution.png'")
plt.show()

# 6. CSV EXPORT
age_summary = (
    df_age.groupby(["dataset_name", "study_condition"])["age"]
    .agg(["mean", "std", "count"])
    .round(2)
)
age_summary.to_csv("./data/age_distribution_summary.csv")
print("Summary table exported to './data/age_distribution_summary.csv'")
