import geopandas as gpd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

#paths
data = '../../03_PROCESSED/PLATFORM_READY/CLUMP_PARAMETERS_RESULTS.gpkg'

if os.path.exists(data):
   df = gpd.read_file(data)
   print(f'Read successful,{len(df)} rows')
else:
   raise FileExistsError

print(df.columns)


# --------------------------------------------------
# PRESENTATION STYLE
# --------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Aptos", "Arial", "DejaVu Sans"],
    "font.size": 12,

    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.labelweight": "medium",

    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    "figure.facecolor": "#FAFBF9",
    "axes.facecolor": "#FAFBF9",

    "text.color": "#26352F",
    "axes.labelcolor": "#52645A",
    "xtick.color": "#64736B",
    "ytick.color": "#64736B",

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": True,

    "axes.edgecolor": "#DCE4DE",
    "axes.linewidth": 0.8,

    "grid.color": "#E5EBE6",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.8,

    "savefig.facecolor": "#FAFBF9",
    "savefig.dpi": 300
})

# Colour palette
FOREST = "#245B45"
DARK = "#26352F"
SAGE = "#86A995"
TEAL = "#4C9A91"
GOLD = "#D5A64A"
CORAL = "#C77963"
LIGHT = "#FAFBF9"
MUTED = "#77867E"

PSP_PALETTE = [
    "#245B45",
    "#4C9A91",
    "#86A995",
    "#D5A64A",
    "#C77963",
    "#647B9A",
    "#9B83A8"
]

# Consistent PSP colours across charts
psps = sorted(df["psp"].dropna().unique(), key=str)

PSP_COLORS = {
    psp: PSP_PALETTE[i % len(PSP_PALETTE)]
    for i, psp in enumerate(psps)
}

# Clean numeric fields
numeric_cols = [
    "total_culms",
    "avg_culm_diameter_cm",
    "canopy_volume_m3",
    "canopy_area",
    "hmean",
    "aboveground_biomass_kg"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Helper function for chart formatting
def finish_chart(fig, ax, filename):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", alpha=0.8)
    ax.grid(axis="x", visible=False)

    fig.tight_layout(pad=2.5)

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        facecolor=LIGHT
    )

    plt.show()




# --------------------------------------------------
# 1. BIOMASS DISTRIBUTION
# --------------------------------------------------

data = df["aboveground_biomass_kg"].dropna()

fig, ax = plt.subplots(figsize=(13.33, 7.5))

# Histogram
ax.hist(
    data,
    bins="auto",
    color=FOREST,
    alpha=0.88,
    edgecolor=LIGHT,
    linewidth=1.2
)

# Median and mean
median = data.median()
mean = data.mean()

ax.axvline(
    median,
    color=GOLD,
    linewidth=2.5,
    linestyle="--",
    label=f"Median: {median:,.1f} kg"
)

ax.axvline(
    mean,
    color=TEAL,
    linewidth=2,
    linestyle=":",
    label=f"Mean: {mean:,.1f} kg"
)

# Titles
ax.set_title(
    "Distribution of Estimated Clump Biomass",
    loc="left",
    pad=25
)

ax.text(
    0, 1.02,
    "Final modelled aboveground biomass across individual bamboo clumps",
    transform=ax.transAxes,
    fontsize=12,
    color=MUTED
)

ax.set_xlabel("Aboveground biomass per clump (kg)", labelpad=12)
ax.set_ylabel("Number of clumps", labelpad=12)

ax.legend(
    frameon=False,
    loc="upper right",
    fontsize=11
)

# Summary annotation
ax.text(
    0.98, 0.78,
    f"CLUMPS  {len(data)}\n"
    f"MEAN  {mean:,.1f} kg\n"
    f"MEDIAN  {median:,.1f} kg",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=12,
    color=DARK,
    linespacing=1.8,
    bbox=dict(
        boxstyle="round,pad=0.8",
        facecolor="#EDF2ED",
        edgecolor="none"
    )
)

finish_chart(fig, ax, "01_biomass_distribution.png")



# --------------------------------------------------
# 2. MEAN HEIGHT VS BIOMASS
# --------------------------------------------------

plot_df = df.dropna(
    subset=["hmean", "aboveground_biomass_kg", "psp"]
).copy()

fig, ax = plt.subplots(figsize=(13.33, 7.5))

# Scatter points by PSP
for psp in psps:
    subset = plot_df[plot_df["psp"] == psp]

    ax.scatter(
        subset["hmean"],
        subset["aboveground_biomass_kg"],
        s=85,
        color=PSP_COLORS[psp],
        alpha=0.78,
        edgecolors="white",
        linewidth=0.8,
        label=str(psp),
        zorder=3
    )

# Overall trend line
x = plot_df["hmean"].to_numpy()
y = plot_df["aboveground_biomass_kg"].to_numpy()

if len(x) >= 2 and np.ptp(x) > 0:
    slope, intercept = np.polyfit(x, y, 1)

    x_line = np.linspace(x.min(), x.max(), 100)

    ax.plot(
        x_line,
        slope * x_line + intercept,
        color=DARK,
        linewidth=2.2,
        linestyle="--",
        alpha=0.8,
        label="Overall trend"
    )

ax.set_title(
    "Clump Height and Estimated Biomass",
    loc="left",
    pad=25
)

ax.text(
    0, 1.02,
    "Relationship between LiDAR-derived mean height and modelled biomass",
    transform=ax.transAxes,
    fontsize=12,
    color=MUTED
)

ax.set_xlabel("Mean clump height (m)", labelpad=12)
ax.set_ylabel("Aboveground biomass (kg)", labelpad=12)

ax.legend(
    title="PSP",
    frameon=False,
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    fontsize=10
)

ax.margins(x=0.08, y=0.12)

finish_chart(fig, ax, "02_height_vs_biomass.png")



# --------------------------------------------------
# 3. TOTAL CULMS VS BIOMASS
# --------------------------------------------------

plot_df = df.dropna(
    subset=["total_culms", "aboveground_biomass_kg", "psp"]
).copy()

fig, ax = plt.subplots(figsize=(13.33, 7.5))

for psp in psps:
    subset = plot_df[plot_df["psp"] == psp]

    ax.scatter(
        subset["total_culms"],
        subset["aboveground_biomass_kg"],
        s=85,
        color=PSP_COLORS[psp],
        alpha=0.78,
        edgecolors="white",
        linewidth=0.8,
        label=str(psp),
        zorder=3
    )

# Overall trend
x = plot_df["total_culms"].to_numpy()
y = plot_df["aboveground_biomass_kg"].to_numpy()

if len(x) >= 2 and np.ptp(x) > 0:
    slope, intercept = np.polyfit(x, y, 1)

    x_line = np.linspace(x.min(), x.max(), 100)

    ax.plot(
        x_line,
        slope * x_line + intercept,
        color=DARK,
        linewidth=2.2,
        linestyle="--",
        alpha=0.8,
        label="Overall trend"
    )

ax.set_title(
    "Culm Count and Estimated Biomass",
    loc="left",
    pad=25
)

ax.text(
    0, 1.02,
    "Relationship between the number of culms and modelled clump biomass",
    transform=ax.transAxes,
    fontsize=12,
    color=MUTED
)

ax.set_xlabel("Total number of culms per clump", labelpad=12)
ax.set_ylabel("Aboveground biomass (kg)", labelpad=12)

ax.legend(
    title="PSP",
    frameon=False,
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    fontsize=10
)

ax.margins(x=0.08, y=0.12)

finish_chart(fig, ax, "03_culms_vs_biomass.png")



# --------------------------------------------------
# 4. MEAN BIOMASS BY PSP
# --------------------------------------------------

plot_df = df.dropna(
    subset=["psp", "aboveground_biomass_kg"]
).copy()

summary = (
    plot_df.groupby("psp")["aboveground_biomass_kg"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .sort_values("mean", ascending=True)
)

# Replace undefined SD for groups with one observation
summary["std"] = summary["std"].fillna(0)

fig, ax = plt.subplots(figsize=(13.33, 7.5))

y_pos = np.arange(len(summary))

colors = [
    PSP_COLORS[p] for p in summary["psp"]
]

ax.barh(
    y_pos,
    summary["mean"],
    color=colors,
    height=0.58,
    alpha=0.92,
    edgecolor="none"
)

ax.set_yticks(y_pos)
ax.set_yticklabels(summary["psp"].astype(str))

# Add mean and sample size labels
for i, row in summary.reset_index(drop=True).iterrows():

    ax.text(
        row["mean"] + summary["mean"].max() * 0.025,
        i,
        f'{row["mean"]:,.1f} kg  |  n={int(row["count"])}',
        va="center",
        fontsize=11,
        color=DARK
    )

ax.set_title(
    "Biomass Estimates Across Sampling Plots",
    loc="left",
    pad=25
)

ax.text(
    0, 1.02,
    "Mean modelled aboveground biomass per clump, grouped by PSP",
    transform=ax.transAxes,
    fontsize=12,
    color=MUTED
)

ax.set_xlabel("Mean aboveground biomass per clump (kg)", labelpad=12)
ax.set_ylabel("Sampling plot (PSP)", labelpad=12)

ax.set_xlim(
    0,
    (summary["mean"] + summary["std"]).max() * 1.4
)

finish_chart(fig, ax, "04_biomass_by_psp.png")



from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --------------------------------------------------
# OBSERVED VS MODELLED BIOMASS
# --------------------------------------------------
observed = '../../02_INTERMEDIATE/merged_acre_ecoplanet_parameters.gpkg'
observed_df = gpd.read_file(observed)
modelled_df = df.copy()
# Check join key uniqueness first
keys = ["psp", "clump_id", "date_of_collection"]

print("Modelled duplicate keys:",
      modelled_df.duplicated(keys).sum())

print("Observed duplicate keys:",
      observed_df.duplicated(keys).sum())

# Join observed and modelled biomass
comparison_df = modelled_df.merge(
    observed_df[keys + ["total_agb"]],
    on=keys,
    how="left",
    suffixes=("_modelled", "_observed"),
    indicator=True,
    validate="one_to_one"
)

print(comparison_df.columns)

# Rename for clarity
comparison_df = comparison_df.rename(columns={
    "aboveground_biomass_kg": "modelled_biomass_kg",
    "total_agb": "observed_biomass_kg"
})


print(comparison_df["_merge"].value_counts())
print("Matched records:", comparison_df["observed_biomass_kg"].notna().sum())


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --------------------------------------------------
# OBSERVED VS MODELLED BIOMASS
# --------------------------------------------------

plot_df = comparison_df.dropna(
    subset=["observed_biomass_kg", "modelled_biomass_kg", "psp"]
).copy()

observed = plot_df["observed_biomass_kg"]
modelled = plot_df["modelled_biomass_kg"]

# Metrics for the displayed records
r2 = r2_score(observed, modelled)
mae = mean_absolute_error(observed, modelled)
rmse = np.sqrt(mean_squared_error(observed, modelled))
bias = np.mean(modelled - observed)

fig, ax = plt.subplots(figsize=(13.33, 7.5))

# Points by PSP
for psp in sorted(plot_df["psp"].unique(), key=str):

    subset = plot_df[plot_df["psp"] == psp]

    ax.scatter(
        subset["observed_biomass_kg"],
        subset["modelled_biomass_kg"],
        s=90,
        color=PSP_COLORS.get(psp, FOREST),
        alpha=0.82,
        edgecolors="white",
        linewidth=0.9,
        label=str(psp),
        zorder=3
    )

# 1:1 reference line
lower = min(observed.min(), modelled.min())
upper = max(observed.max(), modelled.max())

padding = (upper - lower) * 0.06

lower -= padding
upper += padding

ax.plot(
    [lower, upper],
    [lower, upper],
    color=DARK,
    linestyle="--",
    linewidth=2,
    label="1:1 reference",
    zorder=2
)

# Metrics annotation
metrics_text = (
    f"R²     {r2:.2f}\n"
    f"MAE    {mae:.1f} kg\n"
    f"RMSE   {rmse:.1f} kg\n"
    f"Bias   {bias:+.1f} kg\n"
    f"n      {len(plot_df)}"
)

ax.text(
    0.97, 0.05,
    metrics_text,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=12,
    color=DARK,
    linespacing=1.8,
    bbox=dict(
        boxstyle="round,pad=0.8",
        facecolor="#EDF2ED",
        edgecolor="none"
    )
)

ax.set_title(
    "Observed vs Modelled Aboveground Biomass",
    loc="left",
    pad=25
)

ax.text(
    0, 1.02,
    "Comparison of field-observed and model-estimated biomass at clump level",
    transform=ax.transAxes,
    fontsize=12,
    color=MUTED
)

ax.set_xlabel("Observed aboveground biomass (kg)", labelpad=12)
ax.set_ylabel("Modelled aboveground biomass (kg)", labelpad=12)

ax.set_xlim(lower, upper)
ax.set_ylim(lower, upper)

# Equal scaling is essential for interpreting the 1:1 line
ax.set_aspect("equal", adjustable="box")

ax.legend(
    title="PSP",
    frameon=False,
    bbox_to_anchor=(1.01, 1),
    loc="upper left"
)

finish_chart(
    fig,
    ax,
    "06_observed_vs_modelled_biomass.png"
)



# --------------------------------------------------
# RESIDUAL PLOT
# --------------------------------------------------

plot_df = plot_df.copy()

plot_df["residual"] = (
    plot_df["modelled_biomass_kg"]
    - plot_df["observed_biomass_kg"]
)

fig, ax = plt.subplots(figsize=(13.33, 7.5))

for psp in sorted(plot_df["psp"].unique(), key=str):

    subset = plot_df[plot_df["psp"] == psp]

    ax.scatter(
        subset["observed_biomass_kg"],
        subset["residual"],
        s=85,
        color=PSP_COLORS.get(psp, FOREST),
        alpha=0.8,
        edgecolors="white",
        linewidth=0.8,
        label=str(psp)
    )

# Zero-error reference
ax.axhline(
    0,
    color=DARK,
    linestyle="--",
    linewidth=2
)

ax.set_title(
    "Biomass Prediction Residuals",
    loc="left",
    pad=25
)

ax.text(
    0, 1.02,
    "Difference between modelled and observed biomass across clumps",
    transform=ax.transAxes,
    fontsize=12,
    color=MUTED
)

ax.set_xlabel("Observed aboveground biomass (kg)", labelpad=12)
ax.set_ylabel("Residual (modelled − observed) (kg)", labelpad=12)

ax.legend(
    title="PSP",
    frameon=False,
    bbox_to_anchor=(1.01, 1),
    loc="upper left"
)

ax.margins(x=0.08, y=0.15)

finish_chart(
    fig,
    ax,
    "07_biomass_residuals.png"
)
