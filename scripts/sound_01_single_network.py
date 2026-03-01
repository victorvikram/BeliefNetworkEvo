"""
Sound 01: Single Network Analysis — dimensionality, communities, centrality, balance.

Usage: python scripts/sound_01_single_network.py
Outputs: figures/sound_01_*.png, stdout
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
from src.analyzers.triad_analyzer import count_triads
from src.visualizers.network_visualizer import calculate_network_stats

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df = clean_datasets()
    YEARS = list(range(2000, 2011, 2))
    print(f"Reference period: {YEARS}")

    corr_reg = calculate_correlation_matrix(
        cleaned_df, years_of_interest=YEARS,
        method=CorrelationMethod.PEARSON, partial=True,
        edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
        suppression_params={"regularization": 0.2}, verbose=True,
    )
    print(f"\nReference network: {corr_reg.shape[0]} variables, "
          f"{(np.abs(corr_reg.values) > 0).sum() // 2} non-zero edges")

    # ── 1.1 PCA via eigendecomposition ───────────────────────
    network_vars = list(corr_reg.columns)

    corr_simple = calculate_correlation_matrix(
        cleaned_df, years_of_interest=YEARS,
        variables_of_interest=network_vars,
        method=CorrelationMethod.PEARSON, partial=False,
        edge_suppression=EdgeSuppressionMethod.NONE, verbose=False,
    )

    pca_vars = [v for v in network_vars if v in corr_simple.columns]
    corr_pca = corr_simple.loc[pca_vars, pca_vars].copy()
    np.fill_diagonal(corr_pca.values, 1.0)
    corr_pca = corr_pca.fillna(0)
    print(f"PCA input: {len(pca_vars)} variables (eigendecomposition of pairwise Pearson correlation matrix)")

    eigenvalues, eigenvectors = np.linalg.eigh(corr_pca.values)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = np.maximum(eigenvalues, 0)

    total_var = eigenvalues.sum()
    explained = eigenvalues / total_var
    cumulative = np.cumsum(explained)

    print("\nVariance explained by top components:")
    for i in range(5):
        print(f"  PC{i + 1}: {explained[i]:.3f} ({explained[i] * 100:.1f}%)")
    print(f"\n  Cumulative PC1-3: {cumulative[2] * 100:.1f}%")
    print(f"  Components needed for 50% variance: {np.searchsorted(cumulative, 0.5) + 1}")
    print(f"  Components needed for 80% variance: {np.searchsorted(cumulative, 0.8) + 1}")

    if explained[0] > 0.40 and explained[1] < 0.10:
        print("\n** VERDICT: Network is approximately ONE-DIMENSIONAL. **")
    elif explained[0] > 0.25:
        print("\n** VERDICT: Dominant first dimension, but meaningful higher dimensions exist. **")
    else:
        print("\n** VERDICT: Network is genuinely multi-dimensional. **")

    # Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_show = min(20, len(explained))
    axes[0].bar(range(1, n_show + 1), explained[:n_show], alpha=0.7, label="Individual")
    axes[0].plot(range(1, n_show + 1), cumulative[:n_show], "ro-", label="Cumulative")
    axes[0].axhline(y=1 / len(pca_vars), color="gray", linestyle="--", alpha=0.5,
                    label=f"Uniform ({1 / len(pca_vars):.3f})")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance Explained")
    axes[0].set_title("Scree Plot")
    axes[0].legend()

    axes[1].plot(range(1, len(cumulative) + 1), cumulative, "b-")
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    axes[1].axhline(y=0.8, color="gray", linestyle=":", alpha=0.5, label="80%")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance Explained")
    axes[1].set_title("Cumulative Variance")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_01_scree.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Loadings
    loadings = pd.DataFrame(eigenvectors[:, :2], index=pca_vars, columns=["PC1", "PC2"])
    loadings["PC1_abs"] = loadings["PC1"].abs()
    loadings["PC2_abs"] = loadings["PC2"].abs()

    print("\n=== Top 15 variables by |PC1 loading| ===")
    print(loadings.sort_values("PC1_abs", ascending=False)[["PC1"]].head(15).to_string())
    print("\n=== Top 15 variables by |PC2 loading| ===")
    print(loadings.sort_values("PC2_abs", ascending=False)[["PC2"]].head(15).to_string())

    # PC1 vs PC2 scatter
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(loadings["PC1"], loadings["PC2"], alpha=0.6, s=30)
    top_pc1 = loadings.nlargest(8, "PC1_abs").index
    top_pc2 = loadings.nlargest(8, "PC2_abs").index
    to_label = set(top_pc1) | set(top_pc2)
    for var in to_label:
        ax.annotate(var, (loadings.loc[var, "PC1"], loadings.loc[var, "PC2"]), fontsize=7, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"PC1 loading ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 loading ({explained[1] * 100:.1f}% var)")
    ax.set_title("Variable Loadings: PC1 vs PC2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_01_loadings_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 1.2 Network stats ────────────────────────────────────
    corr_abs = corr_reg.copy()
    np.fill_diagonal(corr_abs.values, 0)
    G_ref = nx.from_pandas_adjacency(corr_abs.abs())
    G_ref.remove_edges_from([(u, v) for u, v, d in G_ref.edges(data=True) if d["weight"] == 0])

    stats = calculate_network_stats(G_ref)
    print("\n=== Reference Network Stats (2000-2010) ===")
    print(f"  Nodes:            {stats['num_nodes']}")
    print(f"  Edges:            {stats['num_edges']}")
    print(f"  Density:          {stats['density']:.4f}")
    print(f"  Average degree:   {stats['avg_degree']:.2f}")
    print(f"  Clustering coeff: {stats['clustering_coefficient']:.4f}")
    print(f"  Transitivity:     {stats['global_clustering_coefficient']:.4f}")

    # ── 1.3 Community detection ──────────────────────────────
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
    plt.savefig(FIGURES_DIR / "sound_01_communities.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 1.4 Centrality with dimensionality context ───────────
    bc_dict = nx.betweenness_centrality(G_ref, weight="weight")
    degree_dict = dict(G_ref.degree())
    strength_dict = {n: sum(d["weight"] for _, _, d in G_ref.edges(n, data=True)) for n in G_ref.nodes()}

    centrality_df = pd.DataFrame({"degree": degree_dict, "betweenness": bc_dict, "strength": strength_dict})
    centrality_df = centrality_df.join(loadings[["PC1", "PC1_abs", "PC2", "PC2_abs"]], how="inner")
    centrality_df.rename(columns={"PC1": "PC1_loading", "PC2": "PC2_loading"}, inplace=True)

    print("\n=== Top 15 by Betweenness ===")
    print(centrality_df.sort_values("betweenness", ascending=False)[["betweenness", "degree", "PC1_abs"]].head(15).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    r_deg = centrality_df[["PC1_abs", "degree"]].corr().iloc[0, 1]
    axes[0].scatter(centrality_df["PC1_abs"], centrality_df["degree"], alpha=0.6, s=30)
    axes[0].set_xlabel("|PC1 Loading|")
    axes[0].set_ylabel("Degree")
    axes[0].set_title(f"|PC1 Loading| vs Degree (r={r_deg:.3f})")
    for _, row in centrality_df.iterrows():
        if row["degree"] > centrality_df["degree"].quantile(0.9) and row["PC1_abs"] < centrality_df["PC1_abs"].median():
            axes[0].annotate(row.name, (row["PC1_abs"], row["degree"]), fontsize=7, color="red")
        elif row["degree"] > centrality_df["degree"].quantile(0.95):
            axes[0].annotate(row.name, (row["PC1_abs"], row["degree"]), fontsize=7, alpha=0.6)

    r_btw = centrality_df[["PC1_abs", "betweenness"]].corr().iloc[0, 1]
    axes[1].scatter(centrality_df["PC1_abs"], centrality_df["betweenness"], alpha=0.6, s=30)
    axes[1].set_xlabel("|PC1 Loading|")
    axes[1].set_ylabel("Betweenness Centrality")
    axes[1].set_title(f"|PC1 Loading| vs Betweenness (r={r_btw:.3f})")
    for _, row in centrality_df.iterrows():
        if row["betweenness"] > centrality_df["betweenness"].quantile(0.9) and row["PC1_abs"] < centrality_df["PC1_abs"].median():
            axes[1].annotate(row.name, (row["PC1_abs"], row["betweenness"]), fontsize=7, color="red")
        elif row["betweenness"] > centrality_df["betweenness"].quantile(0.95):
            axes[1].annotate(row.name, (row["PC1_abs"], row["betweenness"]), fontsize=7, alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_01_centrality_vs_pc1.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nCorrelation between |PC1 loading| and degree: r = {r_deg:.3f}")
    print(f"Correlation between |PC1 loading| and betweenness: r = {r_btw:.3f}")

    # Bridge variables
    median_pc1 = centrality_df["PC1_abs"].median()
    high_cent = centrality_df["betweenness"] > centrality_df["betweenness"].quantile(0.75)
    low_pc1 = centrality_df["PC1_abs"] < median_pc1
    bridges = centrality_df[high_cent & low_pc1].sort_values("betweenness", ascending=False)
    if len(bridges) > 0:
        print("\n=== Bridge Variables (high centrality, low PC1 loading) ===")
        print(bridges[["betweenness", "degree", "PC1_abs", "PC1_loading"]].to_string())

    # ── 1.5 Balance with null model ──────────────────────────
    triad_result = count_triads(corr_reg, return_names=True)
    obs_pos = triad_result["positive_triads"]
    obs_neg = triad_result["negative_triads"]
    obs_total = obs_pos + obs_neg
    obs_balance = obs_pos / obs_total if obs_total > 0 else np.nan
    print(f"\nObserved: {obs_pos} balanced / {obs_neg} unbalanced = {obs_balance * 100:.1f}% balanced")

    # Sign-shuffle null model
    corr_vals = corr_reg.values.copy()
    np.fill_diagonal(corr_vals, 0)
    triu_mask = np.triu(np.ones_like(corr_vals, dtype=bool), k=1)
    edge_weights = corr_vals[triu_mask]
    nonzero_mask = edge_weights != 0
    nonzero_weights = edge_weights[nonzero_mask]
    n_positive = (nonzero_weights > 0).sum()
    n_negative = (nonzero_weights < 0).sum()
    n_edges = len(nonzero_weights)
    frac_positive = n_positive / n_edges

    print(f"Edge signs: {n_positive} positive, {n_negative} negative ({frac_positive:.3f} frac positive)")
    print("Running 1000 sign-shuffle permutations...")

    rng = np.random.default_rng(42)
    null_balances = []
    for _ in range(1000):
        shuffled = np.zeros_like(corr_vals)
        abs_weights = np.abs(nonzero_weights)
        signs = np.ones(n_edges)
        neg_indices = rng.choice(n_edges, size=n_negative, replace=False)
        signs[neg_indices] = -1
        signed_weights = abs_weights * signs
        full_upper = np.zeros(triu_mask.sum())
        full_upper[nonzero_mask] = signed_weights
        shuffled[triu_mask] = full_upper
        shuffled = shuffled + shuffled.T
        shuffled_df = pd.DataFrame(shuffled, index=corr_reg.index, columns=corr_reg.columns)
        result = count_triads(shuffled_df)
        null_pos = result["positive_triads"]
        null_total = null_pos + result["negative_triads"]
        if null_total > 0:
            null_balances.append(null_pos / null_total)

    null_balances = np.array(null_balances)
    p_value = (null_balances >= obs_balance).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(null_balances, bins=50, alpha=0.7, edgecolor="black", label="Null (sign-shuffled)")
    ax.axvline(obs_balance, color="red", linewidth=2, linestyle="--",
               label=f"Observed ({obs_balance * 100:.1f}%)")
    ax.set_xlabel("Balance Ratio (fraction of positive triads)")
    ax.set_ylabel("Count")
    ax.set_title("Structural Balance: Observed vs Null Model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_01_balance_null.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nNull mean: {null_balances.mean() * 100:.1f}%, Null std: {null_balances.std() * 100:.1f}%")
    print(f"p-value (one-sided): {p_value:.4f}")
    if p_value < 0.05:
        print("=> Balance is SIGNIFICANTLY higher than expected by chance.")
    else:
        print("=> Balance is NOT significantly higher than expected.")

    print("\nDone. Figures saved to figures/sound_01_*.png")


if __name__ == "__main__":
    main()
