"""
Quick patch: recompute observed distances using relaxed LASSO to match bootstrap,
then regenerate the sound_09 figure.

Usage: PYTHONIOENCODING=utf-8 python scripts/sound_09c_patch_observed.py
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import sys
import warnings
warnings.filterwarnings("ignore")
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import (
    CorrelationMethod, EdgeSuppressionMethod, calculate_correlation_matrix,
)
from src.analyzers.temporal import build_rolling_windows

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
CACHE_DIR = PROJECT_DIR / "data" / "cache"

METHOD = CorrelationMethod.PEARSON
EDGE_SUPP = EdgeSuppressionMethod.REGULARIZATION
SUPP_PARAMS = {"regularization": 0.2}
RELAXED_PARAMS = {"regularization": 0.2, "tol": 1e-3, "max_iter": 50}
EXCLUDE_VARS = {"POLVIEWS", "PARTYID"}


def build_graph(corr_matrix):
    mat = corr_matrix.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat.abs())
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
    return G


def compute_observed(corr, fixed_vars):
    corr = corr.reindex(index=fixed_vars, columns=fixed_vars, fill_value=0)
    G = build_graph(corr)
    upper = corr.values[np.triu_indices(len(fixed_vars), k=1)]
    return {
        "edge_weights": upper,
        "density": nx.density(G),
        "clustering": nx.average_clustering(G, weight="weight"),
        "degree_centrality": nx.degree_centrality(G),
        "n_edges": G.number_of_edges(),
    }


def euclidean_distance_upper(corr_a, corr_b, fixed_vars):
    a = corr_a.reindex(index=fixed_vars, columns=fixed_vars, fill_value=0).values
    b = corr_b.reindex(index=fixed_vars, columns=fixed_vars, fill_value=0).values
    mask = np.triu(np.ones(len(fixed_vars), dtype=bool), k=1)
    return np.sqrt(np.sum((a[mask] - b[mask]) ** 2))


def main():
    cache_path = CACHE_DIR / "sound_09_results.pkl"
    with open(cache_path, "rb") as f:
        all_results = pickle.load(f)
    print(f"Loaded {len(all_results)} windows from cache")

    print("Loading data...")
    cleaned_df = clean_datasets()

    # Rebuild windows with RELAXED params
    print("Rebuilding observed windows with relaxed LASSO...")
    windows = build_rolling_windows(
        cleaned_df,
        window_size=4, step_size=2, min_years_per_window=3, min_n_per_group=100,
        method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=RELAXED_PARAMS,
        group_col="POLVIEWS",
        group_conditions={"lib": "< 0", "con": "> 0"},
        match_samples=True, random_state=42, verbose=False,
    )
    print(f"Built {len(windows)} windows")

    all_var_sets = [set(w.common_vars) for w in windows]
    fixed_vars = sorted(set.intersection(*all_var_sets) - EXCLUDE_VARS)
    print(f"Fixed vars: {len(fixed_vars)}")

    # Patch each cached result with relaxed-LASSO observed values
    for w in windows:
        mid_year = w.mid_year
        if mid_year not in all_results:
            continue

        obs_lib = compute_observed(w.networks["lib"], fixed_vars)
        obs_con = compute_observed(w.networks["con"], fixed_vars)
        obs_distance = euclidean_distance_upper(w.networks["lib"], w.networks["con"], fixed_vars)

        all_results[mid_year]["observed_lib"] = obs_lib
        all_results[mid_year]["observed_con"] = obs_con
        all_results[mid_year]["observed_distance"] = obs_distance

    # Save patched cache
    with open(cache_path, "wb") as f:
        pickle.dump(all_results, f)
    print("Patched cache saved")

    # ── Regenerate figure ────────────────────────────────────────────
    N_PERMS = 1000
    N_BOOT = 1000
    sorted_years = sorted(all_results.keys())
    mid_years = np.array(sorted_years)
    obs_dists = np.array([all_results[y]["observed_distance"] for y in sorted_years])

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    # Top-left: Euclidean distance with null band only
    ax = axes[0, 0]
    null_ci_lower, null_ci_upper = [], []
    for y in sorted_years:
        nd = all_results[y]["null_distances"]
        if len(nd) > 1:
            null_ci_lower.append(np.percentile(nd, 2.5))
            null_ci_upper.append(np.percentile(nd, 97.5))
        else:
            null_ci_lower.append(np.nan)
            null_ci_upper.append(np.nan)

    ax.fill_between(mid_years, null_ci_lower, null_ci_upper,
                     alpha=0.25, color="gray", label="Null 95% range (permutation)")
    ax.plot(mid_years, obs_dists, "ko-", linewidth=2, markersize=5, label="Observed")
    ax.set_ylabel("Euclidean Distance (Lib-Con)")
    ax.set_title("Lib-Con Distance vs Null")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: p-values
    ax = axes[0, 1]
    p_vals = np.array([all_results[y]["p_value"] for y in sorted_years])
    colors = ["green" if p < 0.001 else "blue" if p < 0.05 else "red" for p in p_vals]
    ax.bar(mid_years, p_vals, width=1.5, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(0.05, color="orange", linewidth=1.5, linestyle="--", label="p=0.05")
    ax.axhline(0.001, color="red", linewidth=1.5, linestyle="--", label="p=0.001")
    ax.set_ylabel("p-value")
    ax.set_title("Per-Window Permutation p-values")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Mid-left: Density with bootstrap CIs
    ax = axes[1, 0]
    lib_dens = np.array([all_results[y]["observed_lib"]["density"] for y in sorted_years])
    con_dens = np.array([all_results[y]["observed_con"]["density"] for y in sorted_years])
    lib_d_lo = np.array([all_results[y]["boot_agg_lib"]["density_ci"][0]
                         if all_results[y]["boot_agg_lib"] else np.nan for y in sorted_years])
    lib_d_hi = np.array([all_results[y]["boot_agg_lib"]["density_ci"][1]
                         if all_results[y]["boot_agg_lib"] else np.nan for y in sorted_years])
    con_d_lo = np.array([all_results[y]["boot_agg_con"]["density_ci"][0]
                         if all_results[y]["boot_agg_con"] else np.nan for y in sorted_years])
    con_d_hi = np.array([all_results[y]["boot_agg_con"]["density_ci"][1]
                         if all_results[y]["boot_agg_con"] else np.nan for y in sorted_years])

    ax.fill_between(mid_years, lib_d_lo, lib_d_hi, alpha=0.2, color="blue")
    ax.fill_between(mid_years, con_d_lo, con_d_hi, alpha=0.2, color="red")
    ax.plot(mid_years, lib_dens, "b.-", linewidth=1.5, label="Liberal")
    ax.plot(mid_years, con_dens, "r.-", linewidth=1.5, label="Conservative")
    ax.set_ylabel("Density")
    ax.set_title("Network Density with Bootstrap 95% CIs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mid-right: Clustering with bootstrap CIs
    ax = axes[1, 1]
    lib_clust = np.array([all_results[y]["observed_lib"]["clustering"] for y in sorted_years])
    con_clust = np.array([all_results[y]["observed_con"]["clustering"] for y in sorted_years])
    lib_c_lo = np.array([all_results[y]["boot_agg_lib"]["clustering_ci"][0]
                         if all_results[y]["boot_agg_lib"] else np.nan for y in sorted_years])
    lib_c_hi = np.array([all_results[y]["boot_agg_lib"]["clustering_ci"][1]
                         if all_results[y]["boot_agg_lib"] else np.nan for y in sorted_years])
    con_c_lo = np.array([all_results[y]["boot_agg_con"]["clustering_ci"][0]
                         if all_results[y]["boot_agg_con"] else np.nan for y in sorted_years])
    con_c_hi = np.array([all_results[y]["boot_agg_con"]["clustering_ci"][1]
                         if all_results[y]["boot_agg_con"] else np.nan for y in sorted_years])

    ax.fill_between(mid_years, lib_c_lo, lib_c_hi, alpha=0.2, color="blue")
    ax.fill_between(mid_years, con_c_lo, con_c_hi, alpha=0.2, color="red")
    ax.plot(mid_years, lib_clust, "b.-", linewidth=1.5, label="Liberal")
    ax.plot(mid_years, con_clust, "r.-", linewidth=1.5, label="Conservative")
    ax.set_ylabel("Clustering Coefficient")
    ax.set_title("Clustering with Bootstrap 95% CIs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Top-10 centrality
    ax = axes[2, 0]
    repr_year = sorted_years[len(sorted_years) // 2]
    repr_res = all_results[repr_year]
    if repr_res["boot_agg_lib"]:
        obs_cent = repr_res["observed_lib"]["degree_centrality"]
        boot_ci = repr_res["boot_agg_lib"]["centrality_ci"]
        sorted_v = sorted(obs_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        var_names = [v for v, _ in sorted_v]
        var_cents = [c for _, c in sorted_v]
        ci_lowers = [boot_ci[v][0] for v in var_names]
        ci_uppers = [boot_ci[v][1] for v in var_names]
        y_pos = np.arange(len(var_names))
        ax.barh(y_pos, var_cents, xerr=[np.array(var_cents) - np.array(ci_lowers),
                                         np.array(ci_uppers) - np.array(var_cents)],
                capsize=3, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(var_names, fontsize=8)
        ax.set_xlabel("Degree Centrality")
        ax.set_title(f"Top-10 Liberal Centrality (window ~{repr_year:.0f})")
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Null distribution
    ax = axes[2, 1]
    nd = repr_res["null_distances"]
    obs_d = repr_res["observed_distance"]
    if len(nd) > 0:
        ax.hist(nd, bins=min(30, len(nd)), alpha=0.7, edgecolor="black", color="gray",
                label=f"Null (N={len(nd)})")
        ax.axvline(obs_d, color="red", linewidth=2, linestyle="--",
                   label=f"Observed ({obs_d:.3f})")
        ax.set_xlabel("Euclidean Distance")
        ax.set_ylabel("Count")
        ax.set_title(f"Null Distribution (window ~{repr_year:.0f})")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for row_axes in axes:
        for a in row_axes:
            if a.get_xlim()[1] > 1900:
                a.set_xlim(mid_years.min() - 1, mid_years.max() + 1)

    plt.suptitle("Per-Window Bootstrap & Permutation Tests\n"
                 f"({N_PERMS} permutations, {N_BOOT} bootstraps, {len(fixed_vars)} fixed vars, "
                 f"excl. POLVIEWS/PARTYID)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_09_bootstrap_windows.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'sound_09_bootstrap_windows.png'}")


if __name__ == "__main__":
    main()
