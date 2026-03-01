"""
Sound 07: Robustness Checks — four methodological challenges to core findings.

Check 1: Fixed Variables — Does the divergence trend survive when variables are held constant?
Check 2: Alpha Sensitivity — Do results hold across regularization levels?
Check 3: Full Matrix Test — Does centrality divergence require sparsification?
Check 4: POLVIEWS/PARTYID Exclusion — Is the centrality finding circular?

Usage: PYTHONIOENCODING=utf-8 python scripts/sound_07_robustness.py
Outputs: figures/sound_07_robustness.png, analyses/2026-03_sound-robustness.md, stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import linregress, spearmanr

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import CorrelationMethod, EdgeSuppressionMethod
from src.analyzers.matrix_compare import compare_matrices
from src.analyzers.temporal import build_rolling_windows, get_available_years

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
ANALYSES_DIR = Path(__file__).resolve().parent.parent / "analyses"


# ── Helpers ─────────────────────────────────────────────────────────
def build_graph(corr_matrix):
    """Build a NetworkX graph from a correlation matrix."""
    mat = corr_matrix.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat.abs())
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
    return G


def compute_euclidean_trend(windows):
    """Compute Euclidean distance between lib/con per window, return (mid_years, distances, slope, r, p)."""
    rows = []
    for w in windows:
        comp = compare_matrices(w.networks["lib"], w.networks["con"])
        rows.append({"mid_year": w.mid_year, "euc_dist": comp["euclidean_distance"]})
    df = pd.DataFrame(rows)
    if len(df) < 3:
        return df["mid_year"].values, df["euc_dist"].values, np.nan, np.nan, np.nan
    slope, _, r, p, _ = linregress(df["mid_year"], df["euc_dist"])
    return df["mid_year"].values, df["euc_dist"].values, slope, r, p


def compute_centrality_rho_trend(windows):
    """Compute Spearman rho between lib/con degree centrality per window, return (mid_years, rhos, slope, r, p)."""
    rows = []
    for w in windows:
        G_lib = build_graph(w.networks["lib"])
        G_con = build_graph(w.networks["con"])
        deg_lib = nx.degree_centrality(G_lib)
        deg_con = nx.degree_centrality(G_con)
        common = sorted(set(deg_lib.keys()) & set(deg_con.keys()))
        if len(common) < 10:
            continue
        rho, _ = spearmanr([deg_lib[n] for n in common], [deg_con[n] for n in common])
        rows.append({"mid_year": w.mid_year, "rho": rho})
    df = pd.DataFrame(rows)
    if len(df) < 3:
        return df["mid_year"].values, df["rho"].values, np.nan, np.nan, np.nan
    slope, _, r, p, _ = linregress(df["mid_year"], df["rho"])
    return df["mid_year"].values, df["rho"].values, slope, r, p


def compute_centrality_slopes(windows):
    """Compute per-variable degree centrality slopes over time. Returns DataFrame."""
    cent_data = {}  # var -> list of (mid_year, deg_lib, deg_con)
    for w in windows:
        G_lib = build_graph(w.networks["lib"])
        G_con = build_graph(w.networks["con"])
        deg_lib = nx.degree_centrality(G_lib)
        deg_con = nx.degree_centrality(G_con)
        common = sorted(set(deg_lib.keys()) & set(deg_con.keys()))
        lib_ranked = pd.Series({n: deg_lib[n] for n in common}).rank(ascending=False)
        con_ranked = pd.Series({n: deg_con[n] for n in common}).rank(ascending=False)
        for v in common:
            cent_data.setdefault(v, []).append((w.mid_year, lib_ranked[v] - con_ranked[v]))

    rows = []
    for v, ts in cent_data.items():
        if len(ts) < 5:
            continue
        years = np.array([t[0] for t in ts])
        diffs = np.array([t[1] for t in ts])
        slope, _, r, p, _ = linregress(years, np.abs(diffs))
        rows.append({"variable": v, "slope": slope, "r": r, "p": p, "mean_abs_rank_diff": np.mean(np.abs(diffs))})
    return pd.DataFrame(rows).sort_values("slope", ascending=False)


# ════════════════════════════════════════════════════════════════════
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cleaned_df = clean_datasets()

    # ── Step 0: Determine fixed_vars ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 0: DETERMINE FIXED VARIABLES")
    print("=" * 60)

    # First pass: build windows without fixed_vars to find the intersection
    windows_unfixed = build_rolling_windows(
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
    fixed_vars = sorted(set.intersection(*[set(w.common_vars) for w in windows_unfixed]))
    print(f"Unfixed windows: {len(windows_unfixed)}")
    print(f"Variable counts per window: {[len(w.common_vars) for w in windows_unfixed]}")
    print(f"Fixed variables (intersection): {len(fixed_vars)}")

    # ── Check 1: Fixed Variables ────────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 1: FIXED VARIABLES — DOES DIVERGENCE SURVIVE?")
    print("=" * 60)

    # Original (unfixed) trends
    mid_orig, euc_orig, slope_euc_orig, r_euc_orig, p_euc_orig = compute_euclidean_trend(windows_unfixed)
    mid_rho_orig, rho_orig, slope_rho_orig, r_rho_orig, p_rho_orig = compute_centrality_rho_trend(windows_unfixed)

    print(f"\nOriginal (unfixed) Euclidean distance trend:")
    print(f"  slope={slope_euc_orig:.5f}/yr, r={r_euc_orig:.3f}, p={p_euc_orig:.4f}")
    print(f"Original (unfixed) centrality rho trend:")
    print(f"  slope={slope_rho_orig:.5f}/yr, r={r_rho_orig:.3f}, p={p_rho_orig:.4f}")

    # Rebuild with fixed vars
    windows_fixed = build_rolling_windows(
        cleaned_df,
        window_size=4, step_size=2, min_years_per_window=3,
        min_n_per_group=100,
        method=CorrelationMethod.PEARSON, partial=True,
        edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
        suppression_params={"regularization": 0.2},
        group_col="POLVIEWS",
        group_conditions={"lib": "< 0", "con": "> 0"},
        match_samples=True, random_state=42,
        fixed_vars=fixed_vars,
        verbose=True,
    )
    print(f"Fixed-variable windows: {len(windows_fixed)}")

    mid_fix, euc_fix, slope_euc_fix, r_euc_fix, p_euc_fix = compute_euclidean_trend(windows_fixed)
    mid_rho_fix, rho_fix, slope_rho_fix, r_rho_fix, p_rho_fix = compute_centrality_rho_trend(windows_fixed)

    print(f"\nFixed-variable Euclidean distance trend:")
    print(f"  slope={slope_euc_fix:.5f}/yr, r={r_euc_fix:.3f}, p={p_euc_fix:.4f}")
    print(f"Fixed-variable centrality rho trend:")
    print(f"  slope={slope_rho_fix:.5f}/yr, r={r_rho_fix:.3f}, p={p_rho_fix:.4f}")

    euc_retained = slope_euc_fix / slope_euc_orig * 100 if slope_euc_orig != 0 else float("nan")
    rho_retained = slope_rho_fix / slope_rho_orig * 100 if slope_rho_orig != 0 else float("nan")
    print(f"\nSlope retained: Euclidean={euc_retained:.0f}%, Centrality rho={rho_retained:.0f}%")

    check1_pass = (p_euc_fix < 0.05 and slope_euc_fix > 0)
    print(f"CHECK 1 VERDICT: {'PASS' if check1_pass else 'FAIL'} — divergence trend "
          f"{'survives' if check1_pass else 'does NOT survive'} fixed-variable control")

    # ── Check 2: Alpha Sensitivity ──────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 2: ALPHA SENSITIVITY")
    print("=" * 60)

    alphas = [0.1, 0.15, 0.2, 0.25, 0.3]
    alpha_results = []

    for alpha in alphas:
        print(f"\n  Alpha = {alpha}...")
        w_alpha = build_rolling_windows(
            cleaned_df,
            window_size=4, step_size=2, min_years_per_window=3,
            min_n_per_group=100,
            method=CorrelationMethod.PEARSON, partial=True,
            edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
            suppression_params={"regularization": alpha},
            group_col="POLVIEWS",
            group_conditions={"lib": "< 0", "con": "> 0"},
            match_samples=True, random_state=42,
            fixed_vars=fixed_vars,
            verbose=False,
        )
        _, _, sl_euc, r_euc, p_euc = compute_euclidean_trend(w_alpha)
        _, _, sl_rho, r_rho, p_rho = compute_centrality_rho_trend(w_alpha)
        alpha_results.append({
            "alpha": alpha, "n_windows": len(w_alpha),
            "euc_slope": sl_euc, "euc_r": r_euc, "euc_p": p_euc,
            "rho_slope": sl_rho, "rho_r": r_rho, "rho_p": p_rho,
        })
        print(f"    {len(w_alpha)} windows, euc_slope={sl_euc:.5f} (p={p_euc:.4f}), "
              f"rho_slope={sl_rho:.5f} (p={p_rho:.4f})")

    df_alpha = pd.DataFrame(alpha_results)
    print(f"\n=== Alpha Sensitivity Summary ===")
    print(df_alpha.to_string(index=False, float_format="%.4f"))

    n_euc_sig = (df_alpha["euc_p"] < 0.05).sum()
    n_rho_sig = (df_alpha["rho_p"] < 0.05).sum()
    all_euc_positive = (df_alpha["euc_slope"] > 0).all()
    check2_pass = n_euc_sig >= 4 and all_euc_positive
    print(f"\nEuclidean slope positive for all alphas: {all_euc_positive}")
    print(f"Euclidean slope significant (p<0.05): {n_euc_sig}/{len(alphas)}")
    print(f"Centrality rho slope significant (p<0.05): {n_rho_sig}/{len(alphas)}")
    print(f"CHECK 2 VERDICT: {'PASS' if check2_pass else 'FAIL'} — divergence "
          f"{'robust' if check2_pass else 'NOT robust'} across alpha values")

    # ── Check 3: Full Matrix Test ───────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 3: FULL MATRIX — DOES CENTRALITY DIVERGENCE NEED SPARSIFICATION?")
    print("=" * 60)

    # Use the same year ranges as the fixed-variable windows
    available_years = get_available_years(cleaned_df, require_col="POLVIEWS")
    rng = np.random.default_rng(42)

    full_matrix_rows = []
    sparse_rho_rows = []

    for w in windows_fixed:
        # Get data for this window
        df_window = cleaned_df[cleaned_df["YEAR"].isin(w.years_in_window)].copy()
        df_pv = df_window[df_window["POLVIEWS"].notna()]
        df_lib = df_pv[df_pv["POLVIEWS"] < 0]
        df_con = df_pv[df_pv["POLVIEWS"] > 0]

        # Sample-match to the same N as the sparse windows used
        matched_n = min(len(df_lib), len(df_con))
        if len(df_lib) > matched_n:
            df_lib = df_lib.sample(n=matched_n, random_state=rng.integers(1_000_000_000))
        if len(df_con) > matched_n:
            df_con = df_con.sample(n=matched_n, random_state=rng.integers(1_000_000_000))

        # Full pairwise Pearson correlation (no partial, no regularization)
        corr_lib = df_lib[fixed_vars].corr()
        corr_con = df_con[fixed_vars].corr()

        # Weighted degree: sum(|r|) for each variable (excluding self-correlation)
        wdeg_lib = corr_lib.abs().sum() - 1
        wdeg_con = corr_con.abs().sum() - 1

        rho_full, _ = spearmanr(wdeg_lib.values, wdeg_con.values)
        full_matrix_rows.append({"mid_year": w.mid_year, "rho_full": rho_full})

        # Sparse graph centrality rho (from the pre-built sparse windows)
        G_lib = build_graph(w.networks["lib"])
        G_con = build_graph(w.networks["con"])
        deg_lib = nx.degree_centrality(G_lib)
        deg_con = nx.degree_centrality(G_con)
        common = sorted(set(deg_lib.keys()) & set(deg_con.keys()))
        if len(common) >= 10:
            rho_sparse, _ = spearmanr([deg_lib[n] for n in common], [deg_con[n] for n in common])
            sparse_rho_rows.append({"mid_year": w.mid_year, "rho_sparse": rho_sparse})

    df_full = pd.DataFrame(full_matrix_rows)
    df_sparse = pd.DataFrame(sparse_rho_rows)

    if len(df_full) >= 3:
        sl_full, _, r_full, p_full, _ = linregress(df_full["mid_year"], df_full["rho_full"])
    else:
        sl_full, r_full, p_full = np.nan, np.nan, np.nan

    if len(df_sparse) >= 3:
        sl_sparse, _, r_sparse, p_sparse, _ = linregress(df_sparse["mid_year"], df_sparse["rho_sparse"])
    else:
        sl_sparse, r_sparse, p_sparse = np.nan, np.nan, np.nan

    print(f"\nFull-matrix weighted-degree rho trend:")
    print(f"  slope={sl_full:.5f}/yr, r={r_full:.3f}, p={p_full:.4f}")
    print(f"Sparse-graph degree-centrality rho trend:")
    print(f"  slope={sl_sparse:.5f}/yr, r={r_sparse:.3f}, p={p_sparse:.4f}")

    # Interpretation
    full_declining = sl_full < 0 and p_full < 0.10
    sparse_declining = sl_sparse < 0 and p_sparse < 0.10
    if full_declining and sparse_declining:
        check3_verdict = "BOTH decline — sparsification NOT required for this finding"
        check3_pass = True
    elif sparse_declining and not full_declining:
        check3_verdict = "Only sparse declines — sparsification reveals hidden structure"
        check3_pass = True
    elif not sparse_declining:
        check3_verdict = "Sparse does NOT decline — centrality divergence claim weakened"
        check3_pass = False
    else:
        check3_verdict = "Ambiguous — both trends are weak"
        check3_pass = False

    print(f"\nCHECK 3 VERDICT: {check3_verdict}")

    # ── Check 4: POLVIEWS/PARTYID Exclusion ─────────────────────
    print("\n" + "=" * 60)
    print("CHECK 4: POLVIEWS/PARTYID EXCLUSION — IS CENTRALITY FINDING CIRCULAR?")
    print("=" * 60)

    vars_no_pol = [v for v in fixed_vars if v not in ("POLVIEWS", "PARTYID")]
    print(f"Variables after excluding POLVIEWS and PARTYID: {len(vars_no_pol)} (was {len(fixed_vars)})")

    windows_no_pol = build_rolling_windows(
        cleaned_df,
        window_size=4, step_size=2, min_years_per_window=3,
        min_n_per_group=100,
        method=CorrelationMethod.PEARSON, partial=True,
        edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
        suppression_params={"regularization": 0.2},
        group_col="POLVIEWS",
        group_conditions={"lib": "< 0", "con": "> 0"},
        match_samples=True, random_state=42,
        fixed_vars=vars_no_pol,
        verbose=True,
    )

    mid_nopol, rho_nopol, slope_nopol, r_nopol, p_nopol = compute_centrality_rho_trend(windows_no_pol)
    _, euc_nopol, slope_euc_nopol, r_euc_nopol, p_euc_nopol = compute_euclidean_trend(windows_no_pol)

    print(f"\nCentrality rho trend (excl. POLVIEWS/PARTYID):")
    print(f"  slope={slope_nopol:.5f}/yr, r={r_nopol:.3f}, p={p_nopol:.4f}")
    print(f"Euclidean distance trend (excl. POLVIEWS/PARTYID):")
    print(f"  slope={slope_euc_nopol:.5f}/yr, r={r_euc_nopol:.3f}, p={p_euc_nopol:.4f}")

    # Compare to fixed-variable baseline (with POLVIEWS/PARTYID)
    print(f"\nComparison to fixed-variable baseline:")
    print(f"  Centrality rho slope: {slope_rho_fix:.5f} (with) vs {slope_nopol:.5f} (without)")
    print(f"  Euclidean dist slope: {slope_euc_fix:.5f} (with) vs {slope_euc_nopol:.5f} (without)")

    # Top centrality movers without POLVIEWS/PARTYID
    df_cent_nopol = compute_centrality_slopes(windows_no_pol)
    if len(df_cent_nopol) > 0:
        print(f"\n=== Top 10 Fastest-Growing Centrality Rank Disagreement (excl. POLVIEWS/PARTYID) ===")
        print(df_cent_nopol.head(10).to_string(index=False))

    check4_pass = (slope_nopol < 0 and p_nopol < 0.10)
    print(f"\nCHECK 4 VERDICT: {'PASS' if check4_pass else 'FAIL'} — centrality divergence "
          f"{'survives' if check4_pass else 'does NOT survive'} POLVIEWS/PARTYID exclusion")

    # ════════════════════════════════════════════════════════════
    # SUMMARY FIGURE (4 panels)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY FIGURE")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Check 1 — Fixed vs unfixed Euclidean distance
    ax = axes[0, 0]
    ax.plot(mid_orig, euc_orig, "s--", color="gray", alpha=0.6, label="Original (variable N vars)", markersize=4)
    ax.plot(mid_fix, euc_fix, "ko-", linewidth=2, label=f"Fixed ({len(fixed_vars)} vars)", markersize=5)
    # Trend lines
    x_orig = np.array(mid_orig, dtype=float)
    x_fix = np.array(mid_fix, dtype=float)
    i_orig = linregress(x_orig, euc_orig).intercept
    i_fix = linregress(x_fix, euc_fix).intercept
    ax.plot(x_orig, i_orig + slope_euc_orig * x_orig, "--", color="gray", alpha=0.4)
    ax.plot(x_fix, i_fix + slope_euc_fix * x_fix, "r--", alpha=0.7,
            label=f"Fixed trend: slope={slope_euc_fix:.4f}/yr, p={p_euc_fix:.3f}")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Euclidean Distance (lib vs con)")
    ax.set_title("A. Check 1: Fixed Variables", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Check 2 — Alpha sensitivity
    ax = axes[0, 1]
    ax.plot(df_alpha["alpha"], df_alpha["euc_slope"], "ko-", linewidth=2, label="Euc. dist. slope")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Regularization alpha")
    ax.set_ylabel("Euclidean distance trend slope (/yr)")
    ax2 = ax.twinx()
    ax2.plot(df_alpha["alpha"], df_alpha["rho_slope"], "rs--", linewidth=1.5, label="Centrality rho slope")
    ax2.set_ylabel("Centrality rho trend slope (/yr)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    # Mark significance with filled markers
    for _, row in df_alpha.iterrows():
        if row["euc_p"] < 0.05:
            ax.plot(row["alpha"], row["euc_slope"], "go", markersize=10, alpha=0.3, zorder=0)
    ax.set_title("B. Check 2: Alpha Sensitivity", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel C: Check 3 — Full matrix vs sparse graph
    ax = axes[1, 0]
    ax.plot(df_full["mid_year"], df_full["rho_full"], "b^-", linewidth=1.5, label="Full matrix (weighted degree)")
    ax.plot(df_sparse["mid_year"], df_sparse["rho_sparse"], "ko-", linewidth=2, label="Sparse graph (degree centrality)")
    # Trend lines
    if len(df_full) >= 3:
        x_f = df_full["mid_year"].values.astype(float)
        i_f = linregress(x_f, df_full["rho_full"].values).intercept
        ax.plot(x_f, i_f + sl_full * x_f, "b--", alpha=0.5,
                label=f"Full trend: slope={sl_full:.4f}, p={p_full:.3f}")
    if len(df_sparse) >= 3:
        x_s = df_sparse["mid_year"].values.astype(float)
        i_s = linregress(x_s, df_sparse["rho_sparse"].values).intercept
        ax.plot(x_s, i_s + sl_sparse * x_s, "k--", alpha=0.5,
                label=f"Sparse trend: slope={sl_sparse:.4f}, p={p_sparse:.3f}")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Spearman rho (lib vs con centrality)")
    ax.set_title("C. Check 3: Full Matrix vs Sparse", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel D: Check 4 — With vs without POLVIEWS/PARTYID
    ax = axes[1, 1]
    ax.plot(mid_rho_fix, rho_fix, "ko-", linewidth=2, label="With POLVIEWS/PARTYID", markersize=5)
    ax.plot(mid_nopol, rho_nopol, "g^-", linewidth=1.5, label="Without POLVIEWS/PARTYID", markersize=5)
    # Trend lines
    if len(mid_rho_fix) >= 3:
        x_f2 = np.array(mid_rho_fix, dtype=float)
        i_f2 = linregress(x_f2, rho_fix).intercept
        ax.plot(x_f2, i_f2 + slope_rho_fix * x_f2, "k--", alpha=0.5)
    if len(mid_nopol) >= 3:
        x_np = np.array(mid_nopol, dtype=float)
        i_np = linregress(x_np, rho_nopol).intercept
        ax.plot(x_np, i_np + slope_nopol * x_np, "g--", alpha=0.5,
                label=f"Excl. trend: slope={slope_nopol:.4f}, p={p_nopol:.3f}")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Spearman rho (lib vs con centrality)")
    ax.set_title("D. Check 4: Exclude POLVIEWS/PARTYID", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Robustness Checks: Four Methodological Challenges",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_07_robustness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/sound_07_robustness.png")

    # ════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary_rows = [
        {"Check": "1. Fixed Variables (Euc. dist.)",
         "Slope": f"{slope_euc_fix:.5f}", "r": f"{r_euc_fix:.3f}", "p": f"{p_euc_fix:.4f}",
         "Verdict": "PASS" if check1_pass else "FAIL"},
        {"Check": "1. Fixed Variables (Cent. rho)",
         "Slope": f"{slope_rho_fix:.5f}", "r": f"{r_rho_fix:.3f}", "p": f"{p_rho_fix:.4f}",
         "Verdict": "-"},
        {"Check": "2. Alpha Sensitivity (range)",
         "Slope": f"[{df_alpha['euc_slope'].min():.4f}, {df_alpha['euc_slope'].max():.4f}]",
         "r": f"[{df_alpha['euc_r'].min():.2f}, {df_alpha['euc_r'].max():.2f}]",
         "p": f"{n_euc_sig}/{len(alphas)} sig.",
         "Verdict": "PASS" if check2_pass else "FAIL"},
        {"Check": "3. Full Matrix (weighted deg.)",
         "Slope": f"{sl_full:.5f}", "r": f"{r_full:.3f}", "p": f"{p_full:.4f}",
         "Verdict": "INFO"},
        {"Check": "3. Sparse Graph (deg. cent.)",
         "Slope": f"{sl_sparse:.5f}", "r": f"{r_sparse:.3f}", "p": f"{p_sparse:.4f}",
         "Verdict": check3_verdict.split(" — ")[0] if " — " in check3_verdict else check3_verdict},
        {"Check": "4. Excl. POLVIEWS/PARTYID (rho)",
         "Slope": f"{slope_nopol:.5f}", "r": f"{r_nopol:.3f}", "p": f"{p_nopol:.4f}",
         "Verdict": "PASS" if check4_pass else "FAIL"},
    ]
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    # ════════════════════════════════════════════════════════════
    # ANALYSIS WRITEUP
    # ════════════════════════════════════════════════════════════
    # Collect top movers for Check 4 writeup
    top_movers_str = ""
    if len(df_cent_nopol) > 0:
        for _, row in df_cent_nopol.head(5).iterrows():
            top_movers_str += f"  - {row['variable']}: slope={row['slope']:.4f}, r={row['r']:.3f}\n"

    writeup = f"""# Sound Analysis: Robustness Checks

## Overview

Four robustness checks were applied to the core findings from the temporal
belief-network analysis. These address: (1) variable-count confounding,
(2) regularization sensitivity, (3) necessity of sparsification, and
(4) circularity from including POLVIEWS/PARTYID as network nodes.

All checks use fixed variables (N={len(fixed_vars)}, the intersection of
variables available across all rolling windows) unless otherwise noted.

---

## Check 1: Fixed Variables

**Question:** Does the lib/con divergence trend survive when the same variables
are used in every window?

**Problem:** The original analysis uses per-window variables (~80 early, ~118 late).
Adding variables mechanically inflates Euclidean distance.

**Results:**
- Original (unfixed): slope={slope_euc_orig:.5f}/yr, r={r_euc_orig:.3f}, p={p_euc_orig:.4f}
- Fixed ({len(fixed_vars)} vars): slope={slope_euc_fix:.5f}/yr, r={r_euc_fix:.3f}, p={p_euc_fix:.4f}
- Slope retained: {euc_retained:.0f}%

**Centrality rho:**
- Original: slope={slope_rho_orig:.5f}/yr, r={r_rho_orig:.3f}, p={p_rho_orig:.4f}
- Fixed: slope={slope_rho_fix:.5f}/yr, r={r_rho_fix:.3f}, p={p_rho_fix:.4f}

**Verdict:** {"PASS" if check1_pass else "FAIL"} — The divergence trend
{"survives" if check1_pass else "does NOT survive"} the fixed-variable control.
{"The effect is not an artifact of changing variable counts." if check1_pass else "The original finding may be an artifact of changing variable counts."}

---

## Check 2: Alpha Sensitivity

**Question:** Are the results specific to alpha=0.2, or do they hold across
regularization levels?

**Results:**

| Alpha | Euc. Slope | Euc. r | Euc. p | Rho Slope | Rho p |
|-------|-----------|--------|--------|-----------|-------|
"""

    for _, row in df_alpha.iterrows():
        writeup += f"| {row['alpha']:.2f}  | {row['euc_slope']:.5f}  | {row['euc_r']:.3f} | {row['euc_p']:.4f} | {row['rho_slope']:.5f}  | {row['rho_p']:.4f} |\n"

    writeup += f"""
- Euclidean slope positive for all alphas: {all_euc_positive}
- Euclidean slope significant (p<0.05): {n_euc_sig}/{len(alphas)}

**Verdict:** {"PASS" if check2_pass else "FAIL"} — Divergence is
{"robust" if check2_pass else "NOT robust"} across regularization levels.

---

## Check 3: Full Matrix vs Sparse Graph

**Question:** Does the centrality divergence require sparsification (graphical
LASSO), or does it appear in raw pairwise correlations too?

**Method:** For each window, compute raw pairwise Pearson correlations (no
partial, no regularization). Measure weighted degree as sum(|r|) per variable.
Compare lib/con weighted-degree Spearman rho trend to the sparse-graph
degree-centrality rho trend.

**Results:**
- Full-matrix weighted-degree rho trend: slope={sl_full:.5f}/yr, r={r_full:.3f}, p={p_full:.4f}
- Sparse-graph degree-centrality rho trend: slope={sl_sparse:.5f}/yr, r={r_sparse:.3f}, p={p_sparse:.4f}

**Verdict:** {check3_verdict}

---

## Check 4: POLVIEWS/PARTYID Exclusion

**Question:** POLVIEWS is used to split lib/con groups AND appears as a network
node. Within the liberal group, POLVIEWS has restricted range (only values < 0),
mechanically affecting its correlations. Is the centrality divergence an artifact?

**Method:** Exclude POLVIEWS and PARTYID from the variable list, rebuild
networks, recompute centrality rho trend.

**Results:**
- With POLVIEWS/PARTYID: slope={slope_rho_fix:.5f}/yr, r={r_rho_fix:.3f}, p={p_rho_fix:.4f}
- Without POLVIEWS/PARTYID: slope={slope_nopol:.5f}/yr, r={r_nopol:.3f}, p={p_nopol:.4f}
- Euclidean distance (without): slope={slope_euc_nopol:.5f}/yr, r={r_euc_nopol:.3f}, p={p_euc_nopol:.4f}

**Top centrality movers (excluding POLVIEWS/PARTYID):**
{top_movers_str if top_movers_str else "  (insufficient data)"}

**Verdict:** {"PASS" if check4_pass else "FAIL"} — Centrality divergence
{"survives" if check4_pass else "does NOT survive"} the exclusion of
POLVIEWS/PARTYID.

---

## Summary Table

| Check | Metric | Slope | r | p | Verdict |
|-------|--------|-------|---|---|---------|
| 1. Fixed Vars | Euc. dist. | {slope_euc_fix:.5f} | {r_euc_fix:.3f} | {p_euc_fix:.4f} | {"PASS" if check1_pass else "FAIL"} |
| 2. Alpha Sensitivity | Euc. dist. range | [{df_alpha['euc_slope'].min():.4f}, {df_alpha['euc_slope'].max():.4f}] | - | {n_euc_sig}/{len(alphas)} sig. | {"PASS" if check2_pass else "FAIL"} |
| 3. Full vs Sparse | Full weighted-deg | {sl_full:.5f} | {r_full:.3f} | {p_full:.4f} | INFO |
| 3. Full vs Sparse | Sparse deg-cent | {sl_sparse:.5f} | {r_sparse:.3f} | {p_sparse:.4f} | INFO |
| 4. Excl. POLVIEWS | Cent. rho | {slope_nopol:.5f} | {r_nopol:.3f} | {p_nopol:.4f} | {"PASS" if check4_pass else "FAIL"} |

## Implications for the Paper

"""
    # Generate implications dynamically
    passes = sum([check1_pass, check2_pass, check4_pass])
    total = 3  # checks 1, 2, 4 have pass/fail; check 3 is informational

    if passes == total:
        writeup += """All core findings survive the robustness checks. The divergence trend is not
an artifact of changing variable counts, is stable across regularization levels,
and is not driven by the circularity of including POLVIEWS/PARTYID as nodes.
The paper's temporal claims can be stated with confidence.
"""
    elif passes >= 2:
        writeup += f"""{passes}/{total} checks passed. Most core findings survive, but the
failed check(s) require qualification in the paper. Specific claims affected by
the failed check should be weakened or presented with appropriate caveats.
"""
    else:
        writeup += f"""Only {passes}/{total} checks passed. The core findings have significant
methodological vulnerabilities. The paper needs substantial revision of its
temporal claims. Consider whether the surviving findings are sufficient for the
paper's argument.
"""

    writeup_path = ANALYSES_DIR / "2026-03_sound-robustness.md"
    with open(writeup_path, "w", encoding="utf-8") as f:
        f.write(writeup)
    print(f"\nSaved: {writeup_path}")

    print("\nDone. All robustness checks complete.")


if __name__ == "__main__":
    main()
