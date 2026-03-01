"""
Sound 03: Liberal vs Conservative — Temporal Trajectory (1974-2022).

Usage: python scripts/sound_03_lib_con_temporal.py
Outputs: figures/sound_03_*.png, stdout
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
from scipy import stats as sp_stats

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import CorrelationMethod, EdgeSuppressionMethod
from src.analyzers.matrix_compare import compare_matrices
from src.analyzers.temporal import build_rolling_windows
from src.visualizers.network_visualizer import calculate_network_stats

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def count_communities(corr_matrix):
    mat = corr_matrix.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat.abs())
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
    if G.number_of_nodes() == 0:
        return 0
    comms = nx.community.louvain_communities(G, weight="weight", seed=42)
    return sum(1 for c in comms if len(c) >= 3)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df = clean_datasets()

    # Build rolling windows using temporal.py
    print("Building rolling windows...")
    windows = build_rolling_windows(
        cleaned_df,
        window_size=4, step_size=2, min_years_per_window=3,
        min_n_per_group=100,
        method=CorrelationMethod.PEARSON, partial=True,
        edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
        suppression_params={"regularization": 0.2},
        group_col="POLVIEWS",
        group_conditions={"lib": "< 0", "con": "> 0"},
        match_samples=True, random_state=42,
        verbose=True,
    )
    print(f"Completed {len(windows)} windows")

    # ── Compute metrics per window ───────────────────────────
    results = []
    for w in windows:
        corr_lib = w.networks["lib"]
        corr_con = w.networks["con"]
        comp = compare_matrices(corr_lib, corr_con)

        results.append({
            "window_start": w.start_year,
            "window_end": w.end_year,
            "window_mid": w.mid_year,
            "n_matched": w.matched_n,
            "common_vars": len(w.common_vars),
            "lib_edges": comp["num_edges"]["matrix1"],
            "con_edges": comp["num_edges"]["matrix2"],
            "lib_density": comp["density"]["matrix1"],
            "con_density": comp["density"]["matrix2"],
            "lib_avg_degree": comp["avg_degree"]["matrix1"],
            "con_avg_degree": comp["avg_degree"]["matrix2"],
            "lib_clustering": comp["clustering_coefficient"]["matrix1"],
            "con_clustering": comp["clustering_coefficient"]["matrix2"],
            "lib_triangles": comp["calc_num_triangles"]["matrix1"],
            "con_triangles": comp["calc_num_triangles"]["matrix2"],
            "euclidean_distance": comp["euclidean_distance"],
            "pearson_r": comp["pearson_correlation"],
            "lib_communities": count_communities(corr_lib),
            "con_communities": count_communities(corr_con),
        })

    df_results = pd.DataFrame(results)
    print(df_results[["window_start", "window_end", "n_matched", "lib_edges", "con_edges",
                       "euclidean_distance"]].to_string(index=False))

    # ── 3.2 Trajectory plots ────────────────────────────────
    mid = df_results["window_mid"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    ax = axes[0, 0]
    ax.plot(mid, df_results["euclidean_distance"], "ko-", linewidth=2)
    ax.set_ylabel("Euclidean Distance")
    ax.set_title("Global Difference (Lib vs Con)")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(mid, df_results["lib_edges"], "b.-", label="Liberal", linewidth=1.5)
    ax.plot(mid, df_results["con_edges"], "r.-", label="Conservative", linewidth=1.5)
    ax.set_ylabel("Number of Edges")
    ax.set_title("Edge Count (sample-matched)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(mid, df_results["lib_density"], "b.-", label="Liberal", linewidth=1.5)
    ax.plot(mid, df_results["con_density"], "r.-", label="Conservative", linewidth=1.5)
    ax.set_ylabel("Density")
    ax.set_title("Network Density (sample-matched)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(mid, df_results["lib_clustering"], "b.-", label="Liberal", linewidth=1.5)
    ax.plot(mid, df_results["con_clustering"], "r.-", label="Conservative", linewidth=1.5)
    ax.set_ylabel("Clustering Coefficient")
    ax.set_title("Clustering (sample-matched)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(mid, df_results["pearson_r"], "ko-", linewidth=2)
    ax.set_ylabel("Pearson r (edge weights)")
    ax.set_title("Lib-Con Network Similarity")
    ax.set_xlabel("Window Midpoint (year)")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(mid, df_results["n_matched"], "g.-", linewidth=1.5)
    ax.set_ylabel("Matched N per group")
    ax.set_title("Sample Size (matched)")
    ax.set_xlabel("Window Midpoint (year)")
    ax.grid(True, alpha=0.3)

    for row_axes in axes:
        for ax in row_axes:
            ax.set_xlim(mid.min() - 1, mid.max() + 1)

    plt.suptitle("Liberal vs Conservative Belief Networks: 1974-2022\n"
                 "(4-year rolling windows, sample-size matched)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_03_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Derived metrics (differences) ────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for ax, diff_col, title in [
        (axes[0, 0], df_results["lib_edges"] - df_results["con_edges"], "Edge Count Difference"),
        (axes[0, 1], df_results["lib_density"] - df_results["con_density"], "Density Difference"),
        (axes[1, 0], df_results["lib_clustering"] - df_results["con_clustering"], "Clustering Difference"),
        (axes[1, 1], df_results["lib_triangles"] - df_results["con_triangles"], "Triangle Count Difference"),
    ]:
        ax.bar(mid, diff_col, width=1.5,
               color=["blue" if d > 0 else "red" for d in diff_col], alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Liberal - Conservative")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Window Midpoint (year)")
    axes[1, 1].set_xlabel("Window Midpoint (year)")

    plt.suptitle("Lib minus Con: Positive = Liberal higher (blue), Negative = Conservative higher (red)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_03_differences.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 3.3 Summary ─────────────────────────────────────────
    display_cols = [
        "window_start", "window_end", "n_matched",
        "lib_edges", "con_edges", "lib_density", "con_density",
        "lib_clustering", "con_clustering", "euclidean_distance", "pearson_r",
    ]
    print("\n=== Full Results Table ===")
    print(df_results[display_cols].to_string(index=False, float_format="%.4f"))

    n_windows = len(df_results)
    first_third = df_results.head(n_windows // 3)
    last_third = df_results.tail(n_windows // 3)

    print(f"\nEarly windows (mid ~{first_third['window_mid'].mean():.0f}):")
    print(f"  Euclidean distance: {first_third['euclidean_distance'].mean():.3f}")
    print(f"  Lib edges: {first_third['lib_edges'].mean():.0f}, Con edges: {first_third['con_edges'].mean():.0f}")

    print(f"\nLate windows (mid ~{last_third['window_mid'].mean():.0f}):")
    print(f"  Euclidean distance: {last_third['euclidean_distance'].mean():.3f}")
    print(f"  Lib edges: {last_third['lib_edges'].mean():.0f}, Con edges: {last_third['con_edges'].mean():.0f}")

    slope, intercept, r_val, p_val, std_err = sp_stats.linregress(
        df_results["window_mid"], df_results["euclidean_distance"])
    print(f"\nEuclidean distance trend: slope={slope:.4f}/year, r={r_val:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("=> Lib/con networks are DIVERGING over time." if slope > 0
              else "=> Lib/con networks are CONVERGING over time.")
    else:
        print("=> No significant trend in lib/con divergence.")

    print("\nDone. Figures saved to figures/sound_03_*.png")


if __name__ == "__main__":
    main()
