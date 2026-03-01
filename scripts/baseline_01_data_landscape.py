"""
Baseline 01: Data Landscape — What data do we have?

Usage: python scripts/baseline_01_data_landscape.py
Outputs: figures/baseline_01_*.png, stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.loaders.import_gss import import_dataset
from src.loaders.clean_raw_data import clean_datasets, DataConfig
from src.analyzers.overlap_analyzer import calculate_overlap_matrix, plot_overlap_matrix

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df, _ = import_dataset()
    cleaned_df = clean_datasets()

    exclude = DataConfig.EXCLUDE_COLS
    belief_vars = [c for c in cleaned_df.columns if c not in exclude]

    # ── 1. Dimensions ────────────────────────────────────────
    print("=== RAW DATASET ===")
    print(f"Respondents (rows): {raw_df.shape[0]:,}")
    print(f"Variables (columns): {raw_df.shape[1]}")
    print(f"Years covered: {int(raw_df['YEAR'].min())} - {int(raw_df['YEAR'].max())}")
    print(f"Unique years: {raw_df['YEAR'].nunique()}")
    print()
    print("=== CLEANED DATASET ===")
    print(f"Respondents (rows): {cleaned_df.shape[0]:,}")
    print(f"Variables (columns): {cleaned_df.shape[1]}")
    print(f"Belief variables (excl. YEAR/BALLOT/ID): {len(belief_vars)}")

    # ── 2. Respondents per year ──────────────────────────────
    year_counts = cleaned_df["YEAR"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(year_counts.index, year_counts.values, color="steelblue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Respondents")
    ax.set_title("Number of Respondents per GSS Year")
    ax.set_xticks(year_counts.index[::2])
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_01_respondents_per_year.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nMin respondents/year: {year_counts.min():,} ({year_counts.idxmin()})")
    print(f"Max respondents/year: {year_counts.max():,} ({year_counts.idxmax()})")
    print(f"Mean respondents/year: {year_counts.mean():,.0f}")

    # ── 3. Variable completeness ─────────────────────────────
    missing_pct = cleaned_df[belief_vars].isnull().mean().sort_values(ascending=False) * 100

    print(f"\nVariables with <25% missing: {(missing_pct < 25).sum()}")
    print(f"Variables with 25-50% missing: {((missing_pct >= 25) & (missing_pct < 50)).sum()}")
    print(f"Variables with 50-75% missing: {((missing_pct >= 50) & (missing_pct < 75)).sum()}")
    print(f"Variables with >75% missing: {(missing_pct >= 75).sum()}")
    print()
    print("Top 20 most-missing variables:")
    print(missing_pct.head(20).to_string())

    # Year-by-variable availability heatmap
    years = sorted(cleaned_df["YEAR"].unique())
    availability = pd.DataFrame(index=years, columns=belief_vars, dtype=float)
    for year in years:
        year_data = cleaned_df[cleaned_df["YEAR"] == year][belief_vars]
        availability.loc[year] = year_data.notna().mean()

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(
        availability.astype(float).T, cmap="YlOrRd", ax=ax,
        xticklabels=2, yticklabels=True,
        cbar_kws={"label": "Fraction non-missing"},
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Variable")
    ax.set_title("Variable Availability by Year")
    ax.tick_params(axis="y", labelsize=5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_01_availability_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4. Pairwise overlap ──────────────────────────────────
    overlap_matrix = calculate_overlap_matrix(cleaned_df[belief_vars])

    mask = np.triu(np.ones(overlap_matrix.shape, dtype=bool), k=1)
    upper_vals = overlap_matrix.values[mask]

    print(f"\nPairwise overlap statistics (% of all respondents):")
    print(f"  Min:    {np.nanmin(upper_vals):.1f}%")
    print(f"  Median: {np.nanmedian(upper_vals):.1f}%")
    print(f"  Mean:   {np.nanmean(upper_vals):.1f}%")
    print(f"  Max:    {np.nanmax(upper_vals):.1f}%")
    print(f"  Pairs with <10% overlap: {(upper_vals < 10).sum()}")
    print(f"  Pairs with <5% overlap:  {(upper_vals < 5).sum()}")

    plot_overlap_matrix(overlap_matrix, show=False)
    plt.savefig(FIGURES_DIR / "baseline_01_overlap_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 5. Response distributions ────────────────────────────
    representative_vars = [
        "POLVIEWS", "PARTYID", "EQWLTH", "HELPPOOR",
        "SPKHOMO", "COLRAC", "LIBMSLM",
        "ABANY", "ABDEFECT", "ABRAPE",
        "CONFINAN", "CONSCI", "CONPRESS", "CONLEGIS",
        "NATHEALY", "NATARMSY", "NATEDUCY", "NATENVIY",
        "PREMARSX", "HOMOSEX", "CAPPUN",
    ]
    representative_vars = [v for v in representative_vars if v in cleaned_df.columns]

    n_vars = len(representative_vars)
    n_cols = 5
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
    axes = axes.flatten()

    for i, var in enumerate(representative_vars):
        ax = axes[i]
        data = cleaned_df[var].dropna()
        ax.hist(data, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.axvline(x=data.mean(), color="orange", linestyle="-", alpha=0.7, label=f"mean={data.mean():.2f}")
        ax.set_title(var, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlim(-1.1, 1.1)

    for j in range(n_vars, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Response Distributions (cleaned, [-1, 1] scale)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_01_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary stats
    dist_stats = pd.DataFrame({
        "mean": cleaned_df[belief_vars].mean(),
        "std": cleaned_df[belief_vars].std(),
        "skew": cleaned_df[belief_vars].skew(),
        "n_responses": cleaned_df[belief_vars].notna().sum(),
    }).sort_values("skew")

    print("\nMost left-skewed variables (negative skew = pile-up at high end):")
    print(dist_stats.head(10)[["mean", "skew", "n_responses"]].to_string())
    print()
    print("Most right-skewed variables (positive skew = pile-up at low end):")
    print(dist_stats.tail(10)[["mean", "skew", "n_responses"]].to_string())
    print()
    print(f"Variables with |skew| < 0.5 (roughly symmetric): {(dist_stats['skew'].abs() < 0.5).sum()} of {len(dist_stats)}")

    print("\nDone. Figures saved to figures/baseline_01_*.png")


if __name__ == "__main__":
    main()
