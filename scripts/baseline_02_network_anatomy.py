"""
Baseline 02: Network Anatomy — What does one belief network look like?

Usage: python scripts/baseline_02_network_anatomy.py
Outputs: figures/baseline_02_*.png, stdout
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
from src.analyzers.matrix_compare import compare_matrices
from src.analyzers.triad_analyzer import count_triads
from src.analyzers.frustration_analyzer import calculate_frustration
from src.visualizers.network_visualizer import generate_html_visualization, calculate_network_stats

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df = clean_datasets()
    YEARS = list(range(2000, 2011, 2))
    print(f"Reference period years: {YEARS}")

    # ── 2.1 Method comparison ────────────────────────────────
    corr_simple = calculate_correlation_matrix(
        cleaned_df, years_of_interest=YEARS,
        method=CorrelationMethod.PEARSON, partial=False,
        edge_suppression=EdgeSuppressionMethod.NONE, verbose=False,
    )

    corr_partial = calculate_correlation_matrix(
        cleaned_df, years_of_interest=YEARS,
        method=CorrelationMethod.PEARSON, partial=True,
        edge_suppression=EdgeSuppressionMethod.NONE, verbose=False,
    )

    corr_reg = calculate_correlation_matrix(
        cleaned_df, years_of_interest=YEARS,
        method=CorrelationMethod.PEARSON, partial=True,
        edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
        suppression_params={"regularization": 0.2}, verbose=False,
    )

    print(f"Simple Pearson:      {corr_simple.shape[0]} vars, {(np.abs(corr_simple.values) > 0).sum() // 2} non-zero edges")
    if corr_partial is not None:
        print(f"Partial Pearson:     {corr_partial.shape[0]} vars, {(np.abs(corr_partial.values) > 0).sum() // 2} non-zero edges")
    else:
        print("Partial Pearson:     FAILED — correlation matrix is singular")
        print("  This is expected and is exactly why regularization is needed.")
    print(f"Regularized partial: {corr_reg.shape[0]} vars, {(np.abs(corr_reg.values) > 0).sum() // 2} non-zero edges")

    comparisons = {}
    comparisons["Simple vs Regularized"] = compare_matrices(corr_simple, corr_reg)
    if corr_partial is not None:
        comparisons["Simple vs Partial"] = compare_matrices(corr_simple, corr_partial)
        comparisons["Partial vs Regularized"] = compare_matrices(corr_partial, corr_reg)

    rows = []
    for name, comp in comparisons.items():
        rows.append({
            "Comparison": name,
            "Edges (A)": comp["num_edges"]["matrix1"],
            "Edges (B)": comp["num_edges"]["matrix2"],
            "Pearson r": f"{comp['pearson_correlation']:.3f}",
            "Spearman r": f"{comp['spearman_correlation']:.3f}",
            "Euclidean dist": f"{comp['euclidean_distance']:.3f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 2.2 Network stats ────────────────────────────────────
    corr_abs = corr_reg.copy()
    np.fill_diagonal(corr_abs.values, 0)
    G_ref = nx.from_pandas_adjacency(corr_abs.abs())
    G_ref.remove_edges_from([(u, v) for u, v, d in G_ref.edges(data=True) if d["weight"] == 0])

    stats = calculate_network_stats(G_ref)
    print("\n=== Reference Network Stats (2000-2010, Regularized Partial Pearson) ===")
    for key, val in stats.items():
        if key != "degree_distribution":
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")

    # ── 2.3 Degree distribution ──────────────────────────────
    degrees = (np.abs(corr_reg.values) > 0).sum(axis=1) - 1
    degree_series = pd.Series(degrees, index=corr_reg.columns).sort_values(ascending=False)

    print("\nTop 15 most connected variables:")
    print(degree_series.head(15).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(degree_series.values, bins=range(0, degree_series.max() + 2), alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Degree Distribution")

    top15 = degree_series.head(15)
    axes[1].barh(range(15), top15.values)
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels(top15.index, fontsize=8)
    axes[1].set_xlabel("Degree")
    axes[1].set_title("Top 15 by Degree")
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_02_degree_dist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2.4 Centrality ───────────────────────────────────────
    corr_abs_c = corr_reg.copy()
    np.fill_diagonal(corr_abs_c.values, 0)
    G_cent = nx.from_pandas_adjacency(corr_abs_c.abs())
    G_cent.remove_edges_from([(u, v) for u, v, d in G_cent.edges(data=True) if d["weight"] == 0])

    bc_dict = nx.betweenness_centrality(G_cent, weight="weight")
    bc_df = pd.DataFrame.from_dict(bc_dict, orient="index", columns=["betweenness"]).sort_values("betweenness", ascending=False)
    degree_dict = dict(G_cent.degree())
    deg_df = pd.DataFrame.from_dict(degree_dict, orient="index", columns=["degree"]).sort_values("degree", ascending=False)
    strength_dict = {n: sum(d["weight"] for _, _, d in G_cent.edges(n, data=True)) for n in G_cent.nodes()}
    str_df = pd.DataFrame.from_dict(strength_dict, orient="index", columns=["strength"]).sort_values("strength", ascending=False)

    print("\n=== Top 15 by Betweenness Centrality ===")
    print(bc_df.head(15).to_string())
    print("\n=== Top 15 by Degree ===")
    print(deg_df.head(15).to_string())
    print("\n=== Top 15 by Strength (sum of edge weights) ===")
    print(str_df.head(15).to_string())

    # ── 2.5 Structural balance ───────────────────────────────
    triad_result = count_triads(corr_reg, return_names=True, return_sums=True)
    pos = triad_result["positive_triads"]
    neg = triad_result["negative_triads"]
    total = pos + neg
    print(f"\nTotal triads: {total}")
    print(f"Balanced (positive product): {pos} ({100 * pos / total:.1f}%)")
    print(f"Unbalanced (negative product): {neg} ({100 * neg / total:.1f}%)")
    if neg > 0:
        print(f"\nUnbalanced triads:")
        for nodes in triad_result["negative_triad_nodes"]:
            print(f"  {nodes}")

    # ── 2.6 Frustration ──────────────────────────────────────
    frust_matrix = calculate_frustration(corr_reg)
    frust_vals = frust_matrix.values.copy()
    np.fill_diagonal(frust_vals, 0)
    mask = np.triu(np.ones_like(frust_vals, dtype=bool), k=1)
    rows_idx, cols_idx = np.where(mask & (frust_vals > 0))

    frustrated_edges = []
    for r, c in zip(rows_idx, cols_idx):
        frustrated_edges.append({
            "var1": frust_matrix.index[r],
            "var2": frust_matrix.columns[c],
            "frustration_pct": frust_vals[r, c],
            "edge_weight": corr_reg.iloc[r, c],
        })
    frust_df = pd.DataFrame(frustrated_edges).sort_values("frustration_pct", ascending=False)

    print(f"\nTotal edges with any frustration: {len(frust_df)}")
    print(f"Edges with >25% frustration: {(frust_df['frustration_pct'] > 25).sum()}")
    print("\nTop 20 most frustrated edges:")
    print(frust_df.head(20).to_string(index=False))

    # ── 2.7 Community detection + visualization ──────────────
    communities = nx.community.louvain_communities(G_ref, weight="weight", seed=42)
    print(f"\nNumber of communities: {len(communities)}")
    for i, comm in enumerate(sorted(communities, key=len, reverse=True)):
        print(f"Community {i + 1} ({len(comm)} members): {sorted(comm)}")

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G_ref, k=2, iterations=100, seed=42, weight="weight")
    sorted_comms = sorted(communities, key=len, reverse=True)
    cmap = plt.cm.get_cmap("tab20", len(sorted_comms))
    node_colors = []
    for node in G_ref.nodes():
        for i, comm in enumerate(sorted_comms):
            if node in comm:
                node_colors.append(cmap(i))
                break

    nx.draw_networkx_edges(G_ref, pos, alpha=0.15, ax=ax)
    nx.draw_networkx_nodes(G_ref, pos, node_color=node_colors, node_size=80, ax=ax)
    nx.draw_networkx_labels(G_ref, pos, font_size=6, alpha=0.8, ax=ax)
    ax.set_title("Belief Network Colored by Louvain Community")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "baseline_02_communities.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nDone. Figures saved to figures/baseline_02_*.png")


if __name__ == "__main__":
    main()
