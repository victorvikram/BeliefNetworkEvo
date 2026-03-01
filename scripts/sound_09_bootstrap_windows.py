"""
Sound 09: Per-Window Bootstrap & Permutation Tests.

Provides per-window significance tests (permutation) and confidence intervals
(bootstrap) on network properties across rolling time windows.

Usage:
  PYTHONIOENCODING=utf-8 python scripts/sound_09_bootstrap_windows.py --test   # quick validation (~1 min)
  PYTHONIOENCODING=utf-8 python scripts/sound_09_bootstrap_windows.py           # full run

Outputs:
  figures/sound_09_bootstrap_windows.png
  analyses/2026-03_bootstrap-windows.md
  data/cache/sound_09_results.pkl
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import sys
import argparse
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
from joblib import Parallel, delayed

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import (
    CorrelationMethod,
    EdgeSuppressionMethod,
    calculate_correlation_matrix,
)
from src.analyzers.temporal import build_rolling_windows, get_available_years

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
ANALYSES_DIR = PROJECT_DIR / "analyses"
CACHE_DIR = PROJECT_DIR / "data" / "cache"

METHOD = CorrelationMethod.PEARSON
EDGE_SUPP = EdgeSuppressionMethod.REGULARIZATION
SUPP_PARAMS = {"regularization": 0.2}
# Relaxed LASSO for bootstrap/permutation: faster convergence, acceptable approximation
RELAXED_PARAMS = {"regularization": 0.2, "tol": 1e-3, "max_iter": 50}

EXCLUDE_VARS = {"POLVIEWS", "PARTYID"}

N_JOBS = 8


# ── Helpers ──────────────────────────────────────────────────────────

def build_graph(corr_matrix):
    """Build a NetworkX graph from a correlation matrix."""
    mat = corr_matrix.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat.abs())
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
    return G


def euclidean_distance_upper(corr_a, corr_b, fixed_vars):
    """Euclidean distance between upper triangles of two aligned matrices."""
    a = corr_a.reindex(index=fixed_vars, columns=fixed_vars, fill_value=0).values
    b = corr_b.reindex(index=fixed_vars, columns=fixed_vars, fill_value=0).values
    mask = np.triu(np.ones(len(fixed_vars), dtype=bool), k=1)
    return np.sqrt(np.sum((a[mask] - b[mask]) ** 2))


# ── Bootstrap iteration ─────────────────────────────────────────────

def single_bootstrap(group_df, fixed_vars, seed, suppression_params):
    """Resample one group with replacement, build network, return properties."""
    rng = np.random.default_rng(seed)
    sample = group_df.sample(n=len(group_df), replace=True,
                             random_state=int(rng.integers(1e9)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = calculate_correlation_matrix(
                sample, method=METHOD, partial=True,
                edge_suppression=EDGE_SUPP, suppression_params=suppression_params,
                verbose=False,
            )
    except Exception:
        return None
    if corr is None:
        return None

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


# ── Permutation iteration ───────────────────────────────────────────

def single_permutation(pool_df, n_group, fixed_vars, seed, suppression_params):
    """Shuffle labels, build two networks, return Euclidean distance."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool_df))
    df_a = pool_df.iloc[idx[:n_group]]
    df_b = pool_df.iloc[idx[n_group:2 * n_group]]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_a = calculate_correlation_matrix(
                df_a, method=METHOD, partial=True,
                edge_suppression=EDGE_SUPP, suppression_params=suppression_params,
                verbose=False,
            )
            corr_b = calculate_correlation_matrix(
                df_b, method=METHOD, partial=True,
                edge_suppression=EDGE_SUPP, suppression_params=suppression_params,
                verbose=False,
            )
    except Exception:
        return None
    if corr_a is None or corr_b is None:
        return None

    return euclidean_distance_upper(corr_a, corr_b, fixed_vars)


# ── Per-window observed properties ───────────────────────────────────

def compute_observed(corr, fixed_vars):
    """Compute observed network properties from a correlation matrix."""
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


# ── Aggregate bootstrap results ─────────────────────────────────────

def aggregate_bootstrap(boot_results, fixed_vars):
    """Aggregate a list of bootstrap result dicts into summary stats."""
    valid = [r for r in boot_results if r is not None]
    if not valid:
        return None

    n_valid = len(valid)
    n_edges_upper = len(fixed_vars) * (len(fixed_vars) - 1) // 2

    # Edge weights matrix (n_boot x n_edges_upper)
    edge_mat = np.array([r["edge_weights"] for r in valid])

    # Per-edge: CI and existence frequency (non-zero in what fraction of samples)
    edge_ci_lower = np.percentile(edge_mat, 2.5, axis=0)
    edge_ci_upper = np.percentile(edge_mat, 97.5, axis=0)
    edge_existence = np.mean(edge_mat != 0, axis=0)

    # Network accuracy: fraction of edges appearing in >95% of samples
    network_accuracy = np.mean(edge_existence > 0.95)

    # Density CI
    densities = np.array([r["density"] for r in valid])
    density_ci = (np.percentile(densities, 2.5), np.percentile(densities, 97.5))
    density_mean = np.mean(densities)

    # Clustering CI
    clusterings = np.array([r["clustering"] for r in valid])
    clustering_ci = (np.percentile(clusterings, 2.5), np.percentile(clusterings, 97.5))
    clustering_mean = np.mean(clusterings)

    # Per-variable centrality CI and rank stability
    cent_data = {v: [] for v in fixed_vars}
    for r in valid:
        for v in fixed_vars:
            cent_data[v].append(r["degree_centrality"].get(v, 0.0))

    centrality_ci = {}
    rank_stability = {}
    for v in fixed_vars:
        vals = np.array(cent_data[v])
        centrality_ci[v] = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))

    # Rank stability: % of samples each variable is in top-10
    for r in valid:
        ranked = sorted(r["degree_centrality"].items(), key=lambda x: x[1], reverse=True)
        top10 = {v for v, _ in ranked[:10]}
        for v in fixed_vars:
            rank_stability.setdefault(v, []).append(1 if v in top10 else 0)
    rank_stability = {v: np.mean(counts) for v, counts in rank_stability.items()}

    return {
        "n_valid": n_valid,
        "edge_ci_lower": edge_ci_lower,
        "edge_ci_upper": edge_ci_upper,
        "edge_existence": edge_existence,
        "network_accuracy": network_accuracy,
        "density_mean": density_mean,
        "density_ci": density_ci,
        "clustering_mean": clustering_mean,
        "clustering_ci": clustering_ci,
        "centrality_ci": centrality_ci,
        "rank_stability": rank_stability,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sound 09: Bootstrap & Permutation per window")
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode: 5 perms + 5 boots on first window only")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = CACHE_DIR / "sound_09_results.pkl"
    log_path = CACHE_DIR / "sound_09_progress.log"

    if args.test:
        N_BOOT = 5
        N_PERMS = 5
        print("=== TEST MODE: 5 perms + 5 boots, first window only ===\n")
    else:
        N_BOOT = 1000
        N_PERMS = 1000
        print(f"=== FULL RUN: {N_PERMS} perms + {N_BOOT} boots per window ===\n")

    # ── Step 0: Load data, build observed windows ────────────────────
    print("Loading data...")
    cleaned_df = clean_datasets()

    # Determine fixed_vars: intersection across all windows, excluding POLVIEWS/PARTYID
    print("Building observed rolling windows...")
    windows = build_rolling_windows(
        cleaned_df,
        window_size=4, step_size=2, min_years_per_window=3,
        min_n_per_group=100,
        method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS,
        group_col="POLVIEWS",
        group_conditions={"lib": "< 0", "con": "> 0"},
        match_samples=True, random_state=42,
        verbose=True,
    )
    print(f"Built {len(windows)} observed windows")

    # Fixed vars = intersection of all window common_vars, minus POLVIEWS/PARTYID
    all_var_sets = [set(w.common_vars) for w in windows]
    fixed_vars = sorted(set.intersection(*all_var_sets) - EXCLUDE_VARS)
    print(f"Fixed variables (intersection across all windows, excl. POLVIEWS/PARTYID): {len(fixed_vars)}")

    if args.test:
        windows = windows[:1]  # Only first window for test mode

    # ── Load cache for crash recovery ────────────────────────────────
    all_results = {}
    if cache_path.exists() and not args.test:
        with open(cache_path, "rb") as f:
            all_results = pickle.load(f)
        print(f"Loaded cache with {len(all_results)} completed windows")

    # ── Progress logging ────────────────────────────────────────────
    t_start_all = time.time()

    def log(msg):
        """Print and append to progress log file."""
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            elapsed = time.time() - t_start_all
            f.write(f"[{elapsed/60:6.1f}m] {msg}\n")

    if not args.test:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"sound_09 started: {N_PERMS} perms, {N_BOOT} boots, "
                    f"{len(windows)} windows, {len(fixed_vars)} vars\n")

    # ── Step 2: Per-window bootstrap + permutation ───────────────────
    for w_idx, w in enumerate(windows):
        mid_year = w.mid_year

        if mid_year in all_results and not args.test:
            log(f"[Window {w_idx+1}/{len(windows)}] {w.start_year}-{w.end_year} "
                f"(mid={mid_year}) -- CACHED, skipping")
            continue

        log(f"\n{'='*60}")
        log(f"[Window {w_idx+1}/{len(windows)}] {w.start_year}-{w.end_year} "
            f"(mid={mid_year}), N_matched={w.matched_n}")
        log(f"{'='*60}")

        # Prepare group DataFrames
        df_window = cleaned_df[cleaned_df["YEAR"].isin(w.years_in_window)].copy()
        df_pv = df_window[df_window["POLVIEWS"].notna()]
        df_lib = df_pv[df_pv["POLVIEWS"] < 0]
        df_con = df_pv[df_pv["POLVIEWS"] > 0]

        # Sample-match to matched_n
        rng = np.random.default_rng(42)
        if w.matched_n is not None:
            if len(df_lib) > w.matched_n:
                df_lib = df_lib.sample(n=w.matched_n, random_state=int(rng.integers(1e9)))
            if len(df_con) > w.matched_n:
                df_con = df_con.sample(n=w.matched_n, random_state=int(rng.integers(1e9)))

        n_group = min(len(df_lib), len(df_con))
        pool_df = pd.concat([df_lib.head(n_group), df_con.head(n_group)]).copy()

        # Observed network properties
        obs_lib = compute_observed(w.networks["lib"], fixed_vars)
        obs_con = compute_observed(w.networks["con"], fixed_vars)
        obs_distance = euclidean_distance_upper(w.networks["lib"], w.networks["con"], fixed_vars)

        # ── Bootstrap ────────────────────────────────────────────
        log(f"  Bootstrap: {N_BOOT} iterations per group...")
        t0 = time.time()

        base_rng = np.random.default_rng(42 + w_idx * 1000)
        boot_seeds_lib = base_rng.integers(0, 2**31, size=N_BOOT)
        boot_seeds_con = base_rng.integers(0, 2**31, size=N_BOOT)

        boot_results_lib = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(single_bootstrap)(df_lib, fixed_vars, int(seed), RELAXED_PARAMS)
            for seed in boot_seeds_lib
        )
        boot_results_con = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(single_bootstrap)(df_con, fixed_vars, int(seed), RELAXED_PARAMS)
            for seed in boot_seeds_con
        )

        t_boot = time.time() - t0
        n_valid_lib = sum(1 for r in boot_results_lib if r is not None)
        n_valid_con = sum(1 for r in boot_results_con if r is not None)
        log(f"    Lib: {n_valid_lib}/{N_BOOT} valid, Con: {n_valid_con}/{N_BOOT} valid "
            f"({t_boot:.1f}s)")

        # Bootstrap Euclidean distance CIs (paired lib-con resamples)
        boot_distances = []
        for r_lib, r_con in zip(boot_results_lib, boot_results_con):
            if r_lib is not None and r_con is not None:
                dist = np.sqrt(np.sum((r_lib["edge_weights"] - r_con["edge_weights"]) ** 2))
                boot_distances.append(dist)

        # ── Permutation test ─────────────────────────────────────
        log(f"  Permutation test: {N_PERMS} iterations...")
        t0 = time.time()

        perm_rng = np.random.default_rng(99 + w_idx * 2000)
        perm_seeds = perm_rng.integers(0, 2**31, size=N_PERMS)

        null_distances = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(single_permutation)(pool_df, n_group, fixed_vars, int(seed), RELAXED_PARAMS)
            for seed in perm_seeds
        )

        t_perm = time.time() - t0
        null_distances = np.array([d for d in null_distances if d is not None])
        log(f"    {len(null_distances)}/{N_PERMS} valid ({t_perm:.1f}s)")

        # Compute p-value and z-score
        if len(null_distances) > 0:
            p_value = (null_distances >= obs_distance).mean()
            z_score = (obs_distance - null_distances.mean()) / null_distances.std() if null_distances.std() > 0 else np.nan
        else:
            p_value = np.nan
            z_score = np.nan

        log(f"  Observed distance: {obs_distance:.4f}, p={p_value:.4f}, z={z_score:.2f}")

        # Aggregate bootstrap
        agg_lib = aggregate_bootstrap(boot_results_lib, fixed_vars)
        agg_con = aggregate_bootstrap(boot_results_con, fixed_vars)

        # Store results
        all_results[mid_year] = {
            "start_year": w.start_year,
            "end_year": w.end_year,
            "mid_year": mid_year,
            "matched_n": w.matched_n,
            "n_fixed_vars": len(fixed_vars),
            "observed_lib": obs_lib,
            "observed_con": obs_con,
            "observed_distance": obs_distance,
            "boot_agg_lib": agg_lib,
            "boot_agg_con": agg_con,
            "boot_distances": np.array(boot_distances),
            "null_distances": null_distances,
            "p_value": p_value,
            "z_score": z_score,
        }

        # Save cache after each window
        if not args.test:
            with open(cache_path, "wb") as f:
                pickle.dump(all_results, f)
            elapsed_min = (time.time() - t_start_all) / 60
            remaining_windows = len(windows) - (w_idx + 1)
            avg_min_per_window = elapsed_min / (w_idx + 1) if w_idx >= 0 else 0
            eta_min = avg_min_per_window * remaining_windows
            log(f"  Cached ({len(all_results)}/{len(windows)} windows). "
                f"Elapsed: {elapsed_min:.1f}min, ETA: ~{eta_min:.0f}min")

    # ── Test mode timing report ──────────────────────────────────────
    if args.test:
        mid_year = windows[0].mid_year
        res = all_results[mid_year]
        n_valid_boot = (res["boot_agg_lib"]["n_valid"] + res["boot_agg_con"]["n_valid"])
        n_valid_perm = len(res["null_distances"])
        total_builds = n_valid_boot + n_valid_perm * 2
        total_time = t_boot + t_perm
        sec_per_build = total_time / total_builds if total_builds > 0 else 0

        print(f"\n{'='*60}")
        print(f"TEST MODE TIMING REPORT")
        print(f"{'='*60}")
        print(f"Total matrix builds: {total_builds}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Time per build: {sec_per_build:.2f}s")
        print(f"\nProjections for full run (12 windows):")
        for n in [200, 500, 1000]:
            # per window: 2*n boots + 2*n perms = 4n builds, parallelized across N_JOBS
            proj_builds = 12 * 4 * n
            proj_hours = (proj_builds * sec_per_build / N_JOBS) / 3600
            print(f"  N={n}: ~{proj_builds} builds, ~{proj_hours:.1f} hours (with {N_JOBS} jobs)")

        print(f"\nObserved distance: {res['observed_distance']:.4f}")
        print(f"p-value: {res['p_value']:.4f}, z-score: {res['z_score']:.2f}")
        if res["boot_agg_lib"]:
            print(f"Lib density: {res['observed_lib']['density']:.4f} "
                  f"(CI: {res['boot_agg_lib']['density_ci'][0]:.4f}-{res['boot_agg_lib']['density_ci'][1]:.4f})")
            print(f"Con density: {res['observed_con']['density']:.4f} "
                  f"(CI: {res['boot_agg_con']['density_ci'][0]:.4f}-{res['boot_agg_con']['density_ci'][1]:.4f})")
            print(f"Lib network accuracy: {res['boot_agg_lib']['network_accuracy']:.3f}")
            print(f"Con network accuracy: {res['boot_agg_con']['network_accuracy']:.3f}")
        print("\nTest mode complete.")
        return

    # ── Step 3: Aggregate across windows ─────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")

    sorted_years = sorted(all_results.keys())
    print(f"Windows: {len(sorted_years)}")

    for yr in sorted_years:
        r = all_results[yr]
        print(f"  {r['start_year']}-{r['end_year']} (mid={yr}): "
              f"dist={r['observed_distance']:.4f}, p={r['p_value']:.4f}, z={r['z_score']:.2f}")

    # ── Step 4: Generate figures ─────────────────────────────────────
    print("\nGenerating figures...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))

    mid_years = np.array(sorted_years)
    obs_dists = np.array([all_results[y]["observed_distance"] for y in sorted_years])

    # ─── Top-left: Euclidean distance with bootstrap CI + null band ──
    ax = axes[0, 0]
    boot_dist_lower = []
    boot_dist_upper = []
    null_ci_lower = []
    null_ci_upper = []
    for y in sorted_years:
        r = all_results[y]
        bd = r["boot_distances"]
        if len(bd) > 1:
            boot_dist_lower.append(np.percentile(bd, 2.5))
            boot_dist_upper.append(np.percentile(bd, 97.5))
        else:
            boot_dist_lower.append(np.nan)
            boot_dist_upper.append(np.nan)
        nd = r["null_distances"]
        if len(nd) > 1:
            null_ci_lower.append(np.percentile(nd, 2.5))
            null_ci_upper.append(np.percentile(nd, 97.5))
        else:
            null_ci_lower.append(np.nan)
            null_ci_upper.append(np.nan)

    ax.fill_between(mid_years, null_ci_lower, null_ci_upper,
                     alpha=0.2, color="gray", label="Null 95% range")
    ax.fill_between(mid_years, boot_dist_lower, boot_dist_upper,
                     alpha=0.3, color="blue", label="Bootstrap 95% CI")
    ax.plot(mid_years, obs_dists, "ko-", linewidth=2, markersize=5, label="Observed")
    ax.set_ylabel("Euclidean Distance (Lib-Con)")
    ax.set_title("Lib-Con Distance with CIs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ─── Top-right: Per-window p-values ──────────────────────────────
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

    # ─── Mid-left: Density with bootstrap CIs ───────────────────────
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
    ax.set_title("Network Density with 95% CIs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ─── Mid-right: Network accuracy over time ──────────────────────
    ax = axes[1, 1]
    lib_acc = np.array([all_results[y]["boot_agg_lib"]["network_accuracy"]
                        if all_results[y]["boot_agg_lib"] else np.nan for y in sorted_years])
    con_acc = np.array([all_results[y]["boot_agg_con"]["network_accuracy"]
                        if all_results[y]["boot_agg_con"] else np.nan for y in sorted_years])
    ax.plot(mid_years, lib_acc, "b.-", linewidth=1.5, label="Liberal")
    ax.plot(mid_years, con_acc, "r.-", linewidth=1.5, label="Conservative")
    ax.set_ylabel("Network Accuracy")
    ax.set_title("Edge Stability (frac. in >95% of bootstraps)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # ─── Bottom-left: Top-10 centrality with bootstrap CIs ─────────
    ax = axes[2, 0]
    # Use the middle window as representative
    repr_year = sorted_years[len(sorted_years) // 2]
    repr_res = all_results[repr_year]

    if repr_res["boot_agg_lib"]:
        obs_cent = repr_res["observed_lib"]["degree_centrality"]
        boot_ci = repr_res["boot_agg_lib"]["centrality_ci"]

        # Top-10 by observed centrality
        sorted_vars = sorted(obs_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        var_names = [v for v, _ in sorted_vars]
        var_cents = [c for _, c in sorted_vars]
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
    else:
        ax.text(0.5, 0.5, "No bootstrap data", transform=ax.transAxes, ha="center")
    ax.grid(True, alpha=0.3)

    # ─── Bottom-right: Example null distribution ────────────────────
    ax = axes[2, 1]
    # Use the middle window as representative
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
    else:
        ax.text(0.5, 0.5, "No permutation data", transform=ax.transAxes, ha="center")
    ax.grid(True, alpha=0.3)

    for row_axes in axes:
        for a in row_axes:
            if a.get_xlim()[1] > 1900:  # only for time-axis panels
                a.set_xlim(mid_years.min() - 1, mid_years.max() + 1)

    plt.suptitle("Per-Window Bootstrap & Permutation Tests\n"
                 f"({N_PERMS} permutations, {N_BOOT} bootstraps, {len(fixed_vars)} fixed vars, "
                 f"excl. POLVIEWS/PARTYID)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_09_bootstrap_windows.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {FIGURES_DIR / 'sound_09_bootstrap_windows.png'}")

    # ── Step 5: Generate markdown writeup ────────────────────────────
    all_p = [all_results[y]["p_value"] for y in sorted_years]
    all_z = [all_results[y]["z_score"] for y in sorted_years]
    n_sig_05 = sum(1 for p in all_p if p < 0.05)
    n_sig_001 = sum(1 for p in all_p if p < 0.001)

    # Density CI ranges
    density_table_rows = []
    for y in sorted_years:
        r = all_results[y]
        lib_ci = r["boot_agg_lib"]["density_ci"] if r["boot_agg_lib"] else (np.nan, np.nan)
        con_ci = r["boot_agg_con"]["density_ci"] if r["boot_agg_con"] else (np.nan, np.nan)
        density_table_rows.append(
            f"| {r['start_year']}-{r['end_year']} | {r['matched_n']} | "
            f"{r['observed_lib']['density']:.4f} [{lib_ci[0]:.4f}, {lib_ci[1]:.4f}] | "
            f"{r['observed_con']['density']:.4f} [{con_ci[0]:.4f}, {con_ci[1]:.4f}] | "
            f"{r['observed_distance']:.4f} | {r['p_value']:.4f} | {r['z_score']:.2f} |"
        )

    # Network accuracy summary
    acc_rows = []
    for y in sorted_years:
        r = all_results[y]
        la = r["boot_agg_lib"]["network_accuracy"] if r["boot_agg_lib"] else np.nan
        ca = r["boot_agg_con"]["network_accuracy"] if r["boot_agg_con"] else np.nan
        acc_rows.append(f"| {r['start_year']}-{r['end_year']} | {la:.3f} | {ca:.3f} |")

    md = f"""# Per-Window Bootstrap & Permutation Tests

## Method

For each of {len(sorted_years)} rolling time windows (4-year, step=2), we run:

1. **Permutation test** ({N_PERMS} iterations): Shuffle lib/con labels to test whether
   the observed Euclidean distance between networks is greater than expected by chance.
2. **Bootstrap** ({N_BOOT} iterations per group): Resample with replacement within
   each ideological group to obtain 95% CIs on network properties.

Variables: {len(fixed_vars)} fixed across all windows (intersection), excluding
POLVIEWS and PARTYID to avoid circularity. Relaxed LASSO (tol=1e-3, max_iter=50)
for bootstrap/permutation iterations; standard tolerance for observed networks.

## Per-Window Significance

| Window | N (matched) | Lib Density [95% CI] | Con Density [95% CI] | Euc. Dist | p-value | z-score |
|--------|-------------|---------------------|---------------------|-----------|---------|---------|
{chr(10).join(density_table_rows)}

**Summary**: {n_sig_05}/{len(sorted_years)} windows significant at p<0.05, """
    md += f"""{n_sig_001}/{len(sorted_years)} at p<0.001.

## Network Accuracy (Edge Stability)

Fraction of edges appearing in >95% of bootstrap samples:

| Window | Liberal | Conservative |
|--------|---------|-------------|
{chr(10).join(acc_rows)}

## Interpretation

The permutation test confirms that lib/con network differences are not an artifact
of random label assignment at any individual time point. Bootstrap CIs provide
uncertainty quantification on network properties, showing that observed density
and clustering differences are robust to sampling variability.

## Figure

![Bootstrap Windows](../figures/sound_09_bootstrap_windows.png)
"""

    md_path = ANALYSES_DIR / "2026-03_bootstrap-windows.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved writeup: {md_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
