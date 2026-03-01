"""
Baseline 03: Temporal & Subgroup Comparisons.

Usage: python scripts/baseline_03_comparisons.py
Outputs: figures/baseline_03_*.png, stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import (
    calculate_correlation_matrix, CorrelationMethod, EdgeSuppressionMethod,
)
from src.generators.corr_make_conditioned_network import calculate_conditioned_correlation_matrix
from src.analyzers.matrix_compare import compare_matrices, find_differential_edges
from src.analyzers.graph_similarity import graph_similarity
from src.analyzers.triad_analyzer import count_triads

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

METHOD = CorrelationMethod.PEARSON
EDGE_SUPP = EdgeSuppressionMethod.REGULARIZATION
SUPP_PARAMS = {"regularization": 0.2}


def quick_centrality(corr_matrix, top_n=10):
    mat = corr_matrix.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat.abs())
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
    bc = nx.betweenness_centrality(G, weight="weight")
    strength = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
    bc_df = pd.DataFrame.from_dict(bc, orient="index", columns=["betweenness"]).sort_values("betweenness", ascending=False)
    str_df = pd.DataFrame.from_dict(strength, orient="index", columns=["strength"]).sort_values("strength", ascending=False)
    return bc_df.head(top_n), str_df.head(top_n)


def balance_summary(triads, label):
    pos = triads["positive_triads"]
    neg = triads["negative_triads"]
    total = pos + neg
    print(f"{label}: {pos} balanced / {neg} unbalanced = {100 * pos / total:.1f}% balanced (total: {total})")
    if neg > 0:
        print(f"  Unbalanced triads: {triads['negative_triad_nodes']}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df = clean_datasets()

    # ── Part A: Temporal Comparison ──────────────────────────
    early_years = list(range(1975, 1986))
    late_years = list(range(2010, 2021, 2))
    print(f"Early period years: {early_years}")
    print(f"Late period years: {late_years}")

    corr_early = calculate_correlation_matrix(
        cleaned_df, years_of_interest=early_years,
        method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS, verbose=True,
    )
    corr_late = calculate_correlation_matrix(
        cleaned_df, years_of_interest=late_years,
        method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS, verbose=True,
    )

    print(f"\nEarly network: {corr_early.shape[0]} variables")
    print(f"Late network: {corr_late.shape[0]} variables")
    common_vars = corr_early.columns.intersection(corr_late.columns)
    print(f"Common variables: {len(common_vars)}")

    # 3.2 Structural comparison
    temporal_comp = compare_matrices(corr_early, corr_late)
    print("\n=== Temporal Comparison: 1975-1985 vs 2010-2020 ===")
    for key, val in temporal_comp.items():
        if isinstance(val, dict):
            v1, v2 = val["matrix1"], val["matrix2"]
            if isinstance(v1, float):
                print(f"  {key}: {v1:.4f} -> {v2:.4f}")
            else:
                print(f"  {key}: {v1} -> {v2}")
        elif isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    # 3.3 Differential edges
    stronger_early, stronger_late = find_differential_edges(corr_early, corr_late, top_n=15)
    print("\n=== Top 15 Edges Stronger in EARLY Period ===")
    for var1, var2, diff in stronger_early:
        print(f"  {var1:25s} -- {var2:25s}  diff={diff:+.4f}")
    print("\n=== Top 15 Edges Stronger in LATE Period ===")
    for var1, var2, diff in stronger_late:
        print(f"  {var1:25s} -- {var2:25s}  diff={diff:+.4f}")

    # 3.4 Similarity
    ged_result = graph_similarity(corr_early, corr_late, similarity_method="graph_edit_distance", edge_threshold=0.0)
    print(f"\nGraph Edit Distance: {ged_result}")

    # 3.5 Centrality shift
    bc_df_e, str_df_e = quick_centrality(corr_early)
    bc_df_l, str_df_l = quick_centrality(corr_late)
    print("\n=== Top 10 by Betweenness: EARLY ===")
    print(bc_df_e.to_string())
    print("\n=== Top 10 by Betweenness: LATE ===")
    print(bc_df_l.to_string())
    print("\n=== Top 10 by Strength: EARLY ===")
    print(str_df_e.to_string())
    print("\n=== Top 10 by Strength: LATE ===")
    print(str_df_l.to_string())

    # 3.6 Balance
    triads_early = count_triads(corr_early, return_names=True)
    triads_late = count_triads(corr_late, return_names=True)
    balance_summary(triads_early, "Early (1975-1985)")
    balance_summary(triads_late, "Late (2010-2020)")

    # ── Part B: Subgroup Comparison ──────────────────────────
    REF_YEARS = list(range(2000, 2011, 2))

    corr_liberal = calculate_conditioned_correlation_matrix(
        cleaned_df, years_of_interest=REF_YEARS,
        method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS,
        variable_to_condition="POLVIEWS", condition="less_than_zero", verbose=True,
    )
    corr_conservative = calculate_conditioned_correlation_matrix(
        cleaned_df, years_of_interest=REF_YEARS,
        method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS,
        variable_to_condition="POLVIEWS", condition="greater_than_zero", verbose=True,
    )

    print(f"\nLiberal network: {corr_liberal.shape[0]} variables")
    print(f"Conservative network: {corr_conservative.shape[0]} variables")

    # 3.8 Structural comparison
    subgroup_comp = compare_matrices(corr_liberal, corr_conservative)
    print("\n=== Subgroup Comparison: Liberal vs Conservative (2000-2010) ===")
    for key, val in subgroup_comp.items():
        if isinstance(val, dict):
            v1, v2 = val["matrix1"], val["matrix2"]
            if isinstance(v1, float):
                print(f"  {key}: Liberal={v1:.4f}, Conservative={v2:.4f}")
            else:
                print(f"  {key}: Liberal={v1}, Conservative={v2}")
        elif isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    # Differential edges
    stronger_lib, stronger_con = find_differential_edges(corr_liberal, corr_conservative, top_n=15)
    print("\n=== Top 15 Edges Stronger in LIBERAL Network ===")
    for var1, var2, diff in stronger_lib:
        print(f"  {var1:25s} -- {var2:25s}  diff={diff:+.4f}")
    print("\n=== Top 15 Edges Stronger in CONSERVATIVE Network ===")
    for var1, var2, diff in stronger_con:
        print(f"  {var1:25s} -- {var2:25s}  diff={diff:+.4f}")

    # Balance comparison
    triads_lib = count_triads(corr_liberal, return_names=True)
    triads_con = count_triads(corr_conservative, return_names=True)
    balance_summary(triads_lib, "Liberal")
    balance_summary(triads_con, "Conservative")

    # ── Comparison figure ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Temporal: edge counts
    labels = ["Early\n(1975-85)", "Late\n(2010-20)"]
    early_edges = temporal_comp["num_edges"]["matrix1"]
    late_edges = temporal_comp["num_edges"]["matrix2"]
    axes[0].bar(labels, [early_edges, late_edges], color=["steelblue", "coral"])
    axes[0].set_ylabel("Number of Edges")
    axes[0].set_title("Temporal: Edge Count")

    # Subgroup: edge counts
    labels = ["Liberal", "Conservative"]
    lib_edges = subgroup_comp["num_edges"]["matrix1"]
    con_edges = subgroup_comp["num_edges"]["matrix2"]
    axes[1].bar(labels, [lib_edges, con_edges], color=["blue", "red"], alpha=0.7)
    axes[1].set_ylabel("Number of Edges")
    axes[1].set_title("Subgroup: Edge Count (2000-2010)")

    plt.suptitle("Baseline Comparisons", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_03_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nDone. Figures saved to figures/baseline_03_*.png")


if __name__ == "__main__":
    main()
