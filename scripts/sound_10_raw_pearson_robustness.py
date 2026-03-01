"""
Sound 10: Raw Pearson Correlation Robustness Check.

Tests whether the lib/con network divergence survives without regularization.
Uses raw pairwise Pearson correlations (df.corr()) instead of graphical LASSO
partial correlations. If the divergence appears in raw correlations too, it
is not an artifact of the LASSO sparsification procedure.

Usage: PYTHONIOENCODING=utf-8 python scripts/sound_10_raw_pearson_robustness.py
Outputs: figures/sound_10_raw_pearson.png, stdout
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import sys
import pickle
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import linregress
from joblib import Parallel, delayed

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import CorrelationMethod, EdgeSuppressionMethod
from src.analyzers.temporal import build_rolling_windows

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
CACHE_DIR = PROJECT_DIR / "data" / "cache"

EXCLUDE_VARS = {"POLVIEWS", "PARTYID"}
N_PERMS = 1000
N_JOBS = 8


def euclidean_distance_upper(corr_a, corr_b, fixed_vars):
    """Euclidean distance between upper triangles of two aligned matrices."""
    a = corr_a.reindex(index=fixed_vars, columns=fixed_vars, fill_value=0).values
    b = corr_b.reindex(index=fixed_vars, columns=fixed_vars, fill_value=0).values
    mask = np.triu(np.ones(len(fixed_vars), dtype=bool), k=1)
    return np.sqrt(np.sum((a[mask] - b[mask]) ** 2))


def raw_corr(df, fixed_vars):
    """Compute raw pairwise Pearson correlation matrix on fixed_vars."""
    available = [v for v in fixed_vars if v in df.columns]
    return df[available].corr().reindex(index=fixed_vars, columns=fixed_vars, fill_value=0)


def single_permutation_raw(pool_df, n_group, fixed_vars, seed):
    """Shuffle labels, compute raw Pearson correlation, return Euclidean distance."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool_df))
    df_a = pool_df.iloc[idx[:n_group]]
    df_b = pool_df.iloc[idx[n_group:2 * n_group]]
    corr_a = raw_corr(df_a, fixed_vars)
    corr_b = raw_corr(df_b, fixed_vars)
    return euclidean_distance_upper(corr_a, corr_b, fixed_vars)


def build_graph(corr_matrix, threshold=0.0):
    """Build a NetworkX graph from a correlation matrix."""
    mat = corr_matrix.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat.abs())
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True)
                          if d["weight"] <= threshold])
    return G


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cleaned_df = clean_datasets()

    # Build windows using LASSO just to get the same time slices and sample matching
    print("Building rolling windows (for time slices and sample matching)...")
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
    print(f"Built {len(windows)} windows")

    # Fixed vars = intersection across all windows, minus POLVIEWS/PARTYID
    all_var_sets = [set(w.common_vars) for w in windows]
    fixed_vars = sorted(set.intersection(*all_var_sets) - EXCLUDE_VARS)
    print(f"Fixed variables: {len(fixed_vars)}")

    # Load LASSO results for comparison
    lasso_cache = CACHE_DIR / "sound_09_results.pkl"
    lasso_results = None
    if lasso_cache.exists():
        with open(lasso_cache, "rb") as f:
            lasso_results = pickle.load(f)
        print(f"Loaded LASSO results ({len(lasso_results)} windows) for comparison")

    # ── Per-window raw Pearson analysis ──────────────────────────────
    results = []
    t_start = time.time()

    for w_idx, w in enumerate(windows):
        print(f"\n[Window {w_idx+1}/{len(windows)}] {w.start_year}-{w.end_year} "
              f"(mid={w.mid_year}), N_matched={w.matched_n}")

        # Get matched data for this window
        df_window = cleaned_df[cleaned_df["YEAR"].isin(w.years_in_window)].copy()
        df_pv = df_window[df_window["POLVIEWS"].notna()]
        df_lib = df_pv[df_pv["POLVIEWS"] < 0]
        df_con = df_pv[df_pv["POLVIEWS"] > 0]

        rng = np.random.default_rng(42)
        if w.matched_n is not None:
            if len(df_lib) > w.matched_n:
                df_lib = df_lib.sample(n=w.matched_n,
                                       random_state=int(rng.integers(1e9)))
            if len(df_con) > w.matched_n:
                df_con = df_con.sample(n=w.matched_n,
                                       random_state=int(rng.integers(1e9)))

        n_group = min(len(df_lib), len(df_con))
        pool_df = pd.concat([df_lib.head(n_group), df_con.head(n_group)]).copy()

        # Raw Pearson correlation matrices
        corr_lib = raw_corr(df_lib, fixed_vars)
        corr_con = raw_corr(df_con, fixed_vars)

        # Observed distance
        obs_distance = euclidean_distance_upper(corr_lib, corr_con, fixed_vars)

        # Network properties (using raw correlations with no threshold)
        G_lib = build_graph(corr_lib)
        G_con = build_graph(corr_con)
        lib_density = nx.density(G_lib)
        con_density = nx.density(G_con)
        lib_clustering = nx.average_clustering(G_lib, weight="weight")
        con_clustering = nx.average_clustering(G_con, weight="weight")

        # Permutation test
        perm_rng = np.random.default_rng(99 + w_idx * 2000)
        perm_seeds = perm_rng.integers(0, 2**31, size=N_PERMS)

        t0 = time.time()
        null_distances = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(single_permutation_raw)(pool_df, n_group, fixed_vars, int(s))
            for s in perm_seeds
        )
        t_perm = time.time() - t0

        null_distances = np.array([d for d in null_distances if d is not None])
        p_value = (null_distances >= obs_distance).mean()
        z_score = ((obs_distance - null_distances.mean()) / null_distances.std()
                   if null_distances.std() > 0 else np.nan)

        # LASSO comparison
        lasso_dist = np.nan
        lasso_p = np.nan
        lasso_z = np.nan
        if lasso_results and w.mid_year in lasso_results:
            lr = lasso_results[w.mid_year]
            lasso_dist = lr["observed_distance"]
            lasso_p = lr["p_value"]
            lasso_z = lr["z_score"]

        print(f"  Raw Pearson: dist={obs_distance:.4f}, p={p_value:.4f}, z={z_score:.2f} "
              f"({len(null_distances)}/{N_PERMS} valid, {t_perm:.1f}s)")
        print(f"  LASSO:       dist={lasso_dist:.4f}, p={lasso_p:.4f}, z={lasso_z:.2f}")

        results.append({
            "start_year": w.start_year,
            "end_year": w.end_year,
            "mid_year": w.mid_year,
            "matched_n": w.matched_n,
            "raw_distance": obs_distance,
            "raw_p": p_value,
            "raw_z": z_score,
            "raw_null_mean": null_distances.mean(),
            "raw_null_std": null_distances.std(),
            "null_distances": null_distances,
            "lasso_distance": lasso_dist,
            "lasso_p": lasso_p,
            "lasso_z": lasso_z,
            "lib_density": lib_density,
            "con_density": con_density,
            "lib_clustering": lib_clustering,
            "con_clustering": con_clustering,
        })

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")

    df_res = pd.DataFrame(results)

    # ── Trend analysis ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TREND ANALYSIS")
    print(f"{'='*60}")

    # Raw Pearson distance trend
    sl_raw, _, r_raw, p_raw, _ = linregress(df_res["mid_year"], df_res["raw_distance"])
    print(f"\nRaw Pearson distance trend:")
    print(f"  slope={sl_raw:.5f}/yr, r={r_raw:.3f}, p={p_raw:.4f}")

    # LASSO distance trend (from sound_09)
    lasso_mask = df_res["lasso_distance"].notna()
    if lasso_mask.sum() >= 3:
        sl_lasso, _, r_lasso, p_lasso, _ = linregress(
            df_res.loc[lasso_mask, "mid_year"],
            df_res.loc[lasso_mask, "lasso_distance"])
        print(f"\nLASSO distance trend (from sound_09):")
        print(f"  slope={sl_lasso:.5f}/yr, r={r_lasso:.3f}, p={p_lasso:.4f}")
    else:
        sl_lasso, r_lasso, p_lasso = np.nan, np.nan, np.nan

    # Significance summary
    n_sig_raw = (df_res["raw_p"] < 0.05).sum()
    n_sig_lasso = (df_res["lasso_p"] < 0.05).sum() if lasso_mask.any() else 0
    print(f"\nSignificant windows (p<0.05):")
    print(f"  Raw Pearson: {n_sig_raw}/{len(df_res)}")
    print(f"  LASSO:       {n_sig_lasso}/{lasso_mask.sum()}")

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PER-WINDOW COMPARISON")
    print(f"{'='*60}")
    display = df_res[["start_year", "end_year", "matched_n",
                       "raw_distance", "raw_p", "raw_z",
                       "lasso_distance", "lasso_p", "lasso_z"]].copy()
    display.columns = ["Start", "End", "N", "Raw Dist", "Raw p", "Raw z",
                        "LASSO Dist", "LASSO p", "LASSO z"]
    print(display.to_string(index=False, float_format="%.4f"))

    # ── Figure ──────────────────────────────────────────────────────
    print("\nGenerating figure...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    mid = df_res["mid_year"].values

    # Panel A: Raw Pearson distance with null band (shows significance alone)
    ax = axes[0, 0]
    null_lo = np.array([np.percentile(r["null_distances"], 2.5) for r in results])
    null_hi = np.array([np.percentile(r["null_distances"], 97.5) for r in results])
    ax.fill_between(mid, null_lo, null_hi, alpha=0.25, color="gray",
                     label="Null 95% range (permutation)")
    ax.plot(mid, df_res["raw_distance"], "ko-", linewidth=2, markersize=5,
            label="Observed")
    x = mid.astype(float)
    i_raw = linregress(x, df_res["raw_distance"].values).intercept
    ax.plot(x, i_raw + sl_raw * x, "r--", alpha=0.7,
            label=f"Trend: slope={sl_raw:.4f}/yr, p={p_raw:.3f}")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Euclidean Distance (Lib-Con)")
    ax.set_title("A. Raw Pearson: Distance vs Null", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Raw p-values per window
    ax = axes[0, 1]
    p_vals = df_res["raw_p"].values
    colors = ["green" if p < 0.001 else "blue" if p < 0.05 else "red"
              for p in p_vals]
    ax.bar(mid, p_vals, width=1.5, color=colors, alpha=0.7, edgecolor="black")
    ax.axhline(0.05, color="orange", linewidth=1.5, linestyle="--", label="p=0.05")
    ax.axhline(0.001, color="red", linewidth=1.5, linestyle="--", label="p=0.001")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("p-value")
    ax.set_title("B. Raw Pearson Permutation p-values", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel C: Raw vs LASSO distance comparison
    ax = axes[1, 0]
    ax.plot(mid, df_res["raw_distance"], "bs-", linewidth=2, markersize=5,
            label="Raw Pearson")
    if lasso_mask.any():
        ax.plot(df_res.loc[lasso_mask, "mid_year"],
                df_res.loc[lasso_mask, "lasso_distance"],
                "ko-", linewidth=2, markersize=5, label="LASSO partial")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Euclidean Distance (Lib-Con)")
    ax.set_title("C. Raw vs LASSO Distance (both diverge)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Note different y-axis scales
    ax.text(0.02, 0.98, f"Raw trend: slope={sl_raw:.4f}/yr, r={r_raw:.3f}\n"
            f"LASSO trend: slope={sl_lasso:.4f}/yr, r={r_lasso:.3f}" if not np.isnan(sl_lasso) else "",
            transform=ax.transAxes, fontsize=7, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel D: Example null distribution (middle window)
    ax = axes[1, 1]
    repr_idx = len(results) // 2
    repr_res = results[repr_idx]
    nd = repr_res["null_distances"]
    obs_d = repr_res["raw_distance"]
    ax.hist(nd, bins=30, alpha=0.7, edgecolor="black", color="steelblue",
            label=f"Null (N={len(nd)})")
    ax.axvline(obs_d, color="red", linewidth=2, linestyle="--",
               label=f"Observed ({obs_d:.3f})")
    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Count")
    ax.set_title(f"D. Null Distribution (~{repr_res['mid_year']:.0f}): "
                 f"z={repr_res['raw_z']:.1f}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for row_axes in axes:
        for a in row_axes:
            xlim = a.get_xlim()
            if xlim[1] > 1900:
                a.set_xlim(mid.min() - 1, mid.max() + 1)

    plt.suptitle("Robustness Check: Raw Pearson Correlations (no LASSO)\n"
                 f"({N_PERMS} permutations, {len(fixed_vars)} fixed vars, "
                 f"excl. POLVIEWS/PARTYID)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_10_raw_pearson.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'sound_10_raw_pearson.png'}")

    # ── Verdict ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    raw_diverging = sl_raw > 0 and p_raw < 0.05
    if raw_diverging:
        print(f"PASS: Lib/con divergence appears in raw Pearson correlations")
        print(f"  (slope={sl_raw:.5f}/yr, p={p_raw:.4f})")
        print(f"  The LASSO is NOT creating a spurious difference.")
    elif sl_raw > 0:
        print(f"MARGINAL: Raw Pearson shows positive slope but p={p_raw:.4f}")
        print(f"  Divergence is present but weaker without regularization.")
    else:
        print(f"FAIL: Raw Pearson does NOT show divergence (slope={sl_raw:.5f})")
        print(f"  The divergence may be a LASSO artifact.")

    n_both_sig = sum(1 for _, r in df_res.iterrows()
                     if r["raw_p"] < 0.05 and r["lasso_p"] < 0.05)
    print(f"\nWindows significant in BOTH methods: {n_both_sig}/{len(df_res)}")
    print(f"Windows significant in raw only: "
          f"{n_sig_raw - n_both_sig}/{len(df_res)}")

    # Correlation between raw and LASSO distances
    if lasso_mask.sum() >= 3:
        corr_dist = np.corrcoef(df_res.loc[lasso_mask, "raw_distance"],
                                 df_res.loc[lasso_mask, "lasso_distance"])[0, 1]
        print(f"\nCorrelation between raw and LASSO distances: r={corr_dist:.3f}")
        if corr_dist > 0.8:
            print("  Strong agreement — both methods capture the same signal.")
        elif corr_dist > 0.5:
            print("  Moderate agreement — methods capture overlapping signals.")
        else:
            print("  Weak agreement — methods may capture different aspects.")


if __name__ == "__main__":
    main()
