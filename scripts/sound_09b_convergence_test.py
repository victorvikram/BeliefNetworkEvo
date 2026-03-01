"""
Sound 09b: Convergence test — do 1000 permutations/bootstraps suffice?

Runs 10,000 permutations and 10,000 bootstraps on one early window,
tracking cumulative statistics every 250 iterations.

Usage: PYTHONIOENCODING=utf-8 python scripts/sound_09b_convergence_test.py
Outputs: figures/sound_09b_convergence.png
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import sys
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
from joblib import Parallel, delayed

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import (
    CorrelationMethod, EdgeSuppressionMethod, calculate_correlation_matrix,
)
from src.analyzers.temporal import build_rolling_windows

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"

METHOD = CorrelationMethod.PEARSON
EDGE_SUPP = EdgeSuppressionMethod.REGULARIZATION
SUPP_PARAMS = {"regularization": 0.2}
RELAXED_PARAMS = {"regularization": 0.2, "tol": 1e-3, "max_iter": 50}
EXCLUDE_VARS = {"POLVIEWS", "PARTYID"}

N_TOTAL = 20000
CHECKPOINT = 250
N_JOBS = 8


def euclidean_distance_upper(corr_a, corr_b, fixed_vars):
    a = corr_a.reindex(index=fixed_vars, columns=fixed_vars, fill_value=0).values
    b = corr_b.reindex(index=fixed_vars, columns=fixed_vars, fill_value=0).values
    mask = np.triu(np.ones(len(fixed_vars), dtype=bool), k=1)
    return np.sqrt(np.sum((a[mask] - b[mask]) ** 2))


def single_permutation(pool_df, n_group, fixed_vars, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool_df))
    df_a = pool_df.iloc[idx[:n_group]]
    df_b = pool_df.iloc[idx[n_group:2 * n_group]]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_a = calculate_correlation_matrix(
                df_a, method=METHOD, partial=True,
                edge_suppression=EDGE_SUPP, suppression_params=RELAXED_PARAMS, verbose=False)
            corr_b = calculate_correlation_matrix(
                df_b, method=METHOD, partial=True,
                edge_suppression=EDGE_SUPP, suppression_params=RELAXED_PARAMS, verbose=False)
    except Exception:
        return None
    if corr_a is None or corr_b is None:
        return None
    return euclidean_distance_upper(corr_a, corr_b, fixed_vars)


def single_bootstrap(group_df, fixed_vars, seed):
    import networkx as nx
    rng = np.random.default_rng(seed)
    sample = group_df.sample(n=len(group_df), replace=True,
                             random_state=int(rng.integers(1e9)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = calculate_correlation_matrix(
                sample, method=METHOD, partial=True,
                edge_suppression=EDGE_SUPP, suppression_params=RELAXED_PARAMS, verbose=False)
    except Exception:
        return None
    if corr is None:
        return None
    corr = corr.reindex(index=fixed_vars, columns=fixed_vars, fill_value=0)
    mat = corr.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat.abs())
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
    return nx.density(G)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cleaned_df = clean_datasets()

    print("Building observed windows...")
    windows = build_rolling_windows(
        cleaned_df,
        window_size=4, step_size=2, min_years_per_window=3, min_n_per_group=100,
        method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS,
        group_col="POLVIEWS",
        group_conditions={"lib": "< 0", "con": "> 0"},
        match_samples=True, random_state=42, verbose=False,
    )

    # Pick first window (early, 1974-1978)
    w = windows[0]
    print(f"Test window: {w.start_year}-{w.end_year} (mid={w.mid_year}), N_matched={w.matched_n}")

    # Fixed vars
    all_var_sets = [set(ww.common_vars) for ww in windows]
    fixed_vars = sorted(set.intersection(*all_var_sets) - EXCLUDE_VARS)
    print(f"Fixed vars: {len(fixed_vars)}")

    # Observed distance
    obs_distance = euclidean_distance_upper(w.networks["lib"], w.networks["con"], fixed_vars)
    print(f"Observed distance: {obs_distance:.4f}")

    # Prepare data
    df_window = cleaned_df[cleaned_df["YEAR"].isin(w.years_in_window)].copy()
    df_pv = df_window[df_window["POLVIEWS"].notna()]
    df_lib = df_pv[df_pv["POLVIEWS"] < 0]
    df_con = df_pv[df_pv["POLVIEWS"] > 0]
    rng = np.random.default_rng(42)
    if w.matched_n is not None:
        if len(df_lib) > w.matched_n:
            df_lib = df_lib.sample(n=w.matched_n, random_state=int(rng.integers(1e9)))
        if len(df_con) > w.matched_n:
            df_con = df_con.sample(n=w.matched_n, random_state=int(rng.integers(1e9)))
    n_group = min(len(df_lib), len(df_con))
    pool_df = pd.concat([df_lib.head(n_group), df_con.head(n_group)]).copy()

    # ── Run all 10,000 permutations ──────────────────────────────────
    print(f"\nRunning {N_TOTAL} permutations...")
    perm_rng = np.random.default_rng(99)
    perm_seeds = perm_rng.integers(0, 2**31, size=N_TOTAL)

    t0 = time.time()
    all_null = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(single_permutation)(pool_df, n_group, fixed_vars, int(s))
        for s in perm_seeds
    )
    t_perm = time.time() - t0
    all_null = [d for d in all_null if d is not None]
    print(f"  {len(all_null)}/{N_TOTAL} valid ({t_perm:.0f}s)")

    # Track cumulative stats at each checkpoint
    checkpoints = list(range(CHECKPOINT, len(all_null) + 1, CHECKPOINT))
    perm_n = []
    perm_p = []
    perm_z = []
    perm_null_mean = []
    perm_null_std = []
    for cp in checkpoints:
        subset = np.array(all_null[:cp])
        p = (subset >= obs_distance).mean()
        z = (obs_distance - subset.mean()) / subset.std() if subset.std() > 0 else np.nan
        perm_n.append(cp)
        perm_p.append(p)
        perm_z.append(z)
        perm_null_mean.append(subset.mean())
        perm_null_std.append(subset.std())

    # ── Run all 10,000 bootstraps (liberal density) ──────────────────
    print(f"\nRunning {N_TOTAL} bootstraps (liberal group)...")
    boot_rng = np.random.default_rng(42)
    boot_seeds = boot_rng.integers(0, 2**31, size=N_TOTAL)

    t0 = time.time()
    all_boot = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(single_bootstrap)(df_lib, fixed_vars, int(s))
        for s in boot_seeds
    )
    t_boot = time.time() - t0
    all_boot = [d for d in all_boot if d is not None]
    print(f"  {len(all_boot)}/{N_TOTAL} valid ({t_boot:.0f}s)")

    # Track cumulative CI width at each checkpoint
    boot_checkpoints = list(range(CHECKPOINT, len(all_boot) + 1, CHECKPOINT))
    boot_n = []
    boot_ci_lo = []
    boot_ci_hi = []
    boot_ci_width = []
    boot_mean = []
    for cp in boot_checkpoints:
        subset = np.array(all_boot[:cp])
        lo = np.percentile(subset, 2.5)
        hi = np.percentile(subset, 97.5)
        boot_n.append(cp)
        boot_ci_lo.append(lo)
        boot_ci_hi.append(hi)
        boot_ci_width.append(hi - lo)
        boot_mean.append(np.mean(subset))

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: z-score convergence
    ax = axes[0, 0]
    ax.plot(perm_n, perm_z, "k.-", linewidth=1.5)
    ax.axhline(perm_z[-1], color="red", linewidth=1, linestyle="--", alpha=0.5,
               label=f"Final z={perm_z[-1]:.2f}")
    ax.axvline(1000, color="blue", linewidth=1, linestyle=":", alpha=0.7, label="N=1000")
    ax.set_xlabel("Number of permutations")
    ax.set_ylabel("z-score")
    ax.set_title("Permutation z-score convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: p-value convergence
    ax = axes[0, 1]
    ax.plot(perm_n, perm_p, "k.-", linewidth=1.5)
    ax.axhline(0.05, color="orange", linewidth=1, linestyle="--", label="p=0.05")
    ax.axvline(1000, color="blue", linewidth=1, linestyle=":", alpha=0.7, label="N=1000")
    ax.set_xlabel("Number of permutations")
    ax.set_ylabel("p-value")
    ax.set_title("Permutation p-value convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: null distribution mean & std convergence
    ax = axes[1, 0]
    ax.plot(perm_n, perm_null_mean, "b.-", linewidth=1.5, label="Null mean")
    ax.fill_between(perm_n,
                     np.array(perm_null_mean) - np.array(perm_null_std),
                     np.array(perm_null_mean) + np.array(perm_null_std),
                     alpha=0.2, color="blue", label="Null mean +/- 1 SD")
    ax.axhline(obs_distance, color="red", linewidth=1.5, linestyle="--",
               label=f"Observed ({obs_distance:.3f})")
    ax.axvline(1000, color="blue", linewidth=1, linestyle=":", alpha=0.7, label="N=1000")
    ax.set_xlabel("Number of permutations")
    ax.set_ylabel("Euclidean distance")
    ax.set_title("Null distribution convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: bootstrap CI convergence
    ax = axes[1, 1]
    ax.fill_between(boot_n, boot_ci_lo, boot_ci_hi, alpha=0.3, color="blue",
                     label="95% CI")
    ax.plot(boot_n, boot_mean, "b.-", linewidth=1.5, label="Mean density")
    ax.axvline(1000, color="blue", linewidth=1, linestyle=":", alpha=0.7, label="N=1000")
    ax.set_xlabel("Number of bootstraps")
    ax.set_ylabel("Liberal network density")
    ax.set_title("Bootstrap CI convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Convergence Test: Window {w.start_year}-{w.end_year}\n"
                 f"({N_TOTAL} iterations, checkpoints every {CHECKPOINT})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_09b_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print summary
    print(f"\n{'='*60}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*60}")
    # Print at key checkpoints (index = N/250 - 1)
    key_ns = [250, 1000, 5000, 10000, 20000]
    print(f"\nPermutation z-score:")
    for n in key_ns:
        idx = n // CHECKPOINT - 1
        if idx < len(perm_z):
            print(f"  At N={n:>5d}:  {perm_z[idx]:.2f}")
    print(f"\nBootstrap CI width (density):")
    for n in key_ns:
        idx = n // CHECKPOINT - 1
        if idx < len(boot_ci_width):
            print(f"  At N={n:>5d}:  {boot_ci_width[idx]:.5f}")

    print(f"\nSaved: {FIGURES_DIR / 'sound_09b_convergence.png'}")


if __name__ == "__main__":
    main()
