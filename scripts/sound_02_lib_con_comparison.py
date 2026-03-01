"""
Sound 02: Liberal vs Conservative Comparison — matched samples, permutation test, bootstrap CIs.

Usage: python scripts/sound_02_lib_con_comparison.py
Outputs: figures/sound_02_*.png, stdout
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
from src.analyzers.matrix_compare import compare_matrices
from src.visualizers.network_visualizer import calculate_network_stats

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

METHOD = CorrelationMethod.PEARSON
EDGE_SUPP = EdgeSuppressionMethod.REGULARIZATION
SUPP_PARAMS = {"regularization": 0.2}


def get_communities(corr_matrix):
    mat = corr_matrix.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat.abs())
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
    comms = nx.community.louvain_communities(G, weight="weight", seed=42)
    node_comm = {}
    for i, c in enumerate(comms):
        for node in c:
            node_comm[node] = i
    return comms, node_comm, G


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df = clean_datasets()
    YEARS = list(range(2000, 2011, 2))
    print(f"Reference period: {YEARS}")

    # ── 2.1 Build matched networks ───────────────────────────
    df_period = cleaned_df[cleaned_df["YEAR"].isin(YEARS)].copy()
    df_polviews = df_period[df_period["POLVIEWS"].notna()].copy()

    df_liberal = df_polviews[df_polviews["POLVIEWS"] < 0]
    df_conservative = df_polviews[df_polviews["POLVIEWS"] > 0]

    N_lib = len(df_liberal)
    N_con = len(df_conservative)
    print(f"Liberal: {N_lib}, Conservative: {N_con}")
    print(f"Downsampling conservatives from {N_con} to {N_lib}")

    corr_liberal, _ = calculate_conditioned_correlation_matrix(
        cleaned_df, years_of_interest=YEARS,
        method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS,
        variable_to_condition="POLVIEWS", condition="less_than_zero",
        return_df=True, verbose=True,
    )
    print(f"\nLiberal network: {corr_liberal.shape[0]} vars, "
          f"{(np.abs(corr_liberal.values) > 0).sum() // 2} edges")

    df_con_matched = df_conservative.sample(n=N_lib, random_state=42)
    corr_con_matched = calculate_correlation_matrix(
        df_con_matched, method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS, verbose=True,
    )
    print(f"\nMatched conservative network: {corr_con_matched.shape[0]} vars, "
          f"{(np.abs(corr_con_matched.values) > 0).sum() // 2} edges")

    corr_con_full = calculate_conditioned_correlation_matrix(
        cleaned_df, years_of_interest=YEARS,
        method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS,
        variable_to_condition="POLVIEWS", condition="greater_than_zero", verbose=True,
    )

    # ── 2.2 Permutation test ────────────────────────────────
    common_vars = sorted(set(corr_liberal.columns) & set(corr_con_matched.columns))
    lib_aligned = corr_liberal.loc[common_vars, common_vars]
    con_aligned = corr_con_matched.loc[common_vars, common_vars]

    triu_mask = np.triu(np.ones(len(common_vars), dtype=bool), k=1)
    lib_upper = lib_aligned.values[triu_mask]
    con_upper = con_aligned.values[triu_mask]
    observed_distance = np.sqrt(np.sum((lib_upper - con_upper) ** 2))
    print(f"\nObserved Euclidean distance: {observed_distance:.4f}")

    N_PERMS = 200
    df_pool = pd.concat([df_liberal, df_conservative]).copy()
    n_total = len(df_pool)
    print(f"Pooled sample: {n_total}, running {N_PERMS} permutations...")

    rng = np.random.default_rng(42)
    null_distances = []
    for i in range(N_PERMS):
        shuffled_idx = rng.permutation(n_total)
        group_a_idx = shuffled_idx[:N_lib]
        group_b_idx = shuffled_idx[N_lib:N_lib + N_lib]
        df_a = df_pool.iloc[group_a_idx]
        df_b = df_pool.iloc[group_b_idx]
        try:
            corr_a = calculate_correlation_matrix(df_a, method=METHOD, partial=True,
                                                   edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS, verbose=False)
            corr_b = calculate_correlation_matrix(df_b, method=METHOD, partial=True,
                                                   edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS, verbose=False)
            common = sorted(set(corr_a.columns) & set(corr_b.columns))
            if len(common) < 10:
                continue
            mask = np.triu(np.ones(len(common), dtype=bool), k=1)
            dist = np.sqrt(np.sum((corr_a.loc[common, common].values[mask] - corr_b.loc[common, common].values[mask]) ** 2))
            null_distances.append(dist)
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  Permutation {i + 1}/{N_PERMS}")

    null_distances = np.array(null_distances)
    p_value = (null_distances >= observed_distance).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(null_distances, bins=30, alpha=0.7, edgecolor="black",
            label=f"Null (random splits, N={len(null_distances)})")
    ax.axvline(observed_distance, color="red", linewidth=2, linestyle="--",
               label=f"Observed ({observed_distance:.3f})")
    ax.set_xlabel("Euclidean Distance Between Correlation Matrices")
    ax.set_ylabel("Count")
    ax.set_title("Permutation Test: Is Lib/Con Difference Real?")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_02_permutation.png", dpi=150, bbox_inches="tight")
    plt.close()

    z_score = (observed_distance - null_distances.mean()) / null_distances.std()
    print(f"\nNull mean: {null_distances.mean():.4f}, Null std: {null_distances.std():.4f}")
    print(f"Z-score: {z_score:.2f}, p-value: {p_value:.4f}")

    # ── 2.3 Bootstrap CIs ───────────────────────────────────
    N_BOOT = 200
    boot_common_vars = sorted(set(corr_liberal.columns) & set(corr_con_matched.columns))
    n_vars = len(boot_common_vars)
    n_edges = n_vars * (n_vars - 1) // 2
    boot_diffs = np.full((N_BOOT, n_edges), np.nan)

    rng_boot = np.random.default_rng(123)
    print(f"\nBootstrap: {N_BOOT} iterations...")
    for i in range(N_BOOT):
        lib_sample = df_liberal.sample(n=len(df_liberal), replace=True, random_state=rng_boot.integers(1e9))
        con_sample = df_conservative.sample(n=len(df_liberal), replace=True, random_state=rng_boot.integers(1e9))
        try:
            corr_lib_b = calculate_correlation_matrix(lib_sample, method=METHOD, partial=True,
                                                       edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS, verbose=False)
            corr_con_b = calculate_correlation_matrix(con_sample, method=METHOD, partial=True,
                                                       edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS, verbose=False)
            lib_vals = corr_lib_b.reindex(index=boot_common_vars, columns=boot_common_vars, fill_value=0).values
            con_vals = corr_con_b.reindex(index=boot_common_vars, columns=boot_common_vars, fill_value=0).values
            mask = np.triu(np.ones(n_vars, dtype=bool), k=1)
            boot_diffs[i] = lib_vals[mask] - con_vals[mask]
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  Bootstrap {i + 1}/{N_BOOT}")

    valid_mask = ~np.all(np.isnan(boot_diffs), axis=1)
    boot_diffs_valid = boot_diffs[valid_mask]
    print(f"Completed {valid_mask.sum()} / {N_BOOT} bootstrap iterations")

    mean_diff = np.nanmean(boot_diffs_valid, axis=0)
    ci_lower = np.nanpercentile(boot_diffs_valid, 2.5, axis=0)
    ci_upper = np.nanpercentile(boot_diffs_valid, 97.5, axis=0)

    edge_labels = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            edge_labels.append((boot_common_vars[i], boot_common_vars[j]))

    sig_stronger_lib = []
    sig_stronger_con = []
    for idx_e, (v1, v2) in enumerate(edge_labels):
        if ci_lower[idx_e] > 0:
            sig_stronger_lib.append((v1, v2, mean_diff[idx_e], ci_lower[idx_e], ci_upper[idx_e]))
        elif ci_upper[idx_e] < 0:
            sig_stronger_con.append((v1, v2, mean_diff[idx_e], ci_lower[idx_e], ci_upper[idx_e]))

    sig_stronger_lib.sort(key=lambda x: abs(x[2]), reverse=True)
    sig_stronger_con.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\nEdges significantly stronger in LIBERAL: {len(sig_stronger_lib)}")
    print(f"Edges significantly stronger in CONSERVATIVE: {len(sig_stronger_con)}")

    print("\n=== Top 15 Edges Significantly Stronger in LIBERAL Network ===")
    for v1, v2, md, cl, cu in sig_stronger_lib[:15]:
        print(f"  {v1:25s} {v2:25s} {md:+.4f} [{cl:+.4f}, {cu:+.4f}]")
    print("\n=== Top 15 Edges Significantly Stronger in CONSERVATIVE Network ===")
    for v1, v2, md, cl, cu in sig_stronger_con[:15]:
        print(f"  {v1:25s} {v2:25s} {md:+.4f} [{cl:+.4f}, {cu:+.4f}]")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(mean_diff, bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Mean Edge Weight Difference (Liberal - Conservative)")
    ax.set_ylabel("Count (edges)")
    ax.set_title(f"Distribution of Edge Differences\n"
                 f"({len(sig_stronger_lib)} sig. stronger in liberal, "
                 f"{len(sig_stronger_con)} sig. stronger in conservative)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_02_edge_diffs.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2.4 Density comparison ───────────────────────────────
    comp_matched = compare_matrices(corr_liberal, corr_con_matched)
    comp_full = compare_matrices(corr_liberal, corr_con_full)

    metrics = ["num_edges", "density", "avg_degree", "avg_weight_sum",
               "clustering_coefficient", "calc_num_triangles"]
    print(f"\n{'Metric':30s} {'Liberal':>10s} {'Con(matched)':>12s} {'Con(full)':>12s}")
    print("-" * 66)
    for m in metrics:
        lib_val = comp_matched[m]["matrix1"]
        con_m_val = comp_matched[m]["matrix2"]
        con_f_val = comp_full[m]["matrix2"]
        if isinstance(lib_val, float):
            print(f"{m:30s} {lib_val:10.4f} {con_m_val:12.4f} {con_f_val:12.4f}")
        else:
            print(f"{m:30s} {lib_val:10d} {con_m_val:12d} {con_f_val:12d}")

    # ── 2.5 Community comparison ─────────────────────────────
    lib_comms, lib_node_comm, G_lib = get_communities(corr_liberal)
    con_comms, con_node_comm, G_con = get_communities(corr_con_matched)

    print(f"\nLiberal communities: {len(lib_comms)}")
    for i, c in enumerate(sorted(lib_comms, key=len, reverse=True)):
        print(f"  L{i + 1} ({len(c)}): {sorted(c)}")
    print(f"\nConservative communities: {len(con_comms)}")
    for i, c in enumerate(sorted(con_comms, key=len, reverse=True)):
        print(f"  C{i + 1} ({len(c)}): {sorted(c)}")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for ax, G, comms, title in [
        (axes[0], G_lib, lib_comms, f"Liberal Network (N={N_lib})"),
        (axes[1], G_con, con_comms, f"Conservative Network (N={N_lib}, matched)"),
    ]:
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42, weight="weight")
        sorted_comms = sorted(comms, key=len, reverse=True)
        n_comms = len(sorted_comms)
        cmap = plt.cm.get_cmap("tab20", max(n_comms, 1))
        colors = []
        for node in G.nodes():
            for i, c in enumerate(sorted_comms):
                if node in c:
                    colors.append(cmap(i))
                    break
        nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=60, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=5, alpha=0.7, ax=ax)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_02_communities.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nDone. Figures saved to figures/sound_02_*.png")


if __name__ == "__main__":
    main()
