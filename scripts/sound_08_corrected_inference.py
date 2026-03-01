"""
Sound 08: Corrected Statistical Inference — HAC standard errors, non-overlapping
validation, FDR correction, and structural break tests.

Addresses reviewer critique that 22 overlapping windows (step=2, width=4) share 50%
of data between adjacent windows, inflating OLS standard errors. All regressions use
fixed variables excluding POLVIEWS/PARTYID (addressing circularity simultaneously).

Usage: PYTHONIOENCODING=utf-8 python scripts/sound_08_corrected_inference.py
Outputs: figures/sound_08_corrected_inference.png,
         analyses/2026-03_corrected-inference.md, stdout
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
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.multitest import multipletests
from scipy.stats import linregress, spearmanr

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import CorrelationMethod, EdgeSuppressionMethod
from src.analyzers.matrix_compare import compare_matrices
from src.analyzers.temporal import build_rolling_windows

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
ANALYSES_DIR = Path(__file__).resolve().parent.parent / "analyses"


# ── Helper functions ────────────────────────────────────────────────


def hac_regression(x, y, maxlags=1):
    """OLS regression with HAC (Newey-West) standard errors.

    Returns dict with slope, intercept, p_value, ci_lower, ci_upper,
    r_squared, durbin_watson, n_obs, std_err.
    """
    X = sm.add_constant(np.array(x, dtype=float))
    Y = np.array(y, dtype=float)
    model = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    dw = durbin_watson(model.resid)
    ci = model.conf_int(alpha=0.05)
    return {
        "slope": model.params[1],
        "intercept": model.params[0],
        "p_value": model.pvalues[1],
        "ci_lower": ci[1, 0],
        "ci_upper": ci[1, 1],
        "r_squared": model.rsquared,
        "durbin_watson": dw,
        "n_obs": len(Y),
        "std_err": model.bse[1],
    }


def ols_regression(x, y):
    """Plain OLS via linregress for comparison. Returns dict matching hac_regression keys."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    slope, intercept, r, p, se = linregress(x, y)
    return {
        "slope": slope,
        "intercept": intercept,
        "p_value": p,
        "r_squared": r ** 2,
        "std_err": se,
        "r": r,
        "n_obs": len(y),
    }


def build_graph(corr_matrix):
    """Build a NetworkX graph from a correlation matrix."""
    mat = corr_matrix.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat.abs())
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
    return G


def compute_euclidean_distances(windows):
    """Return (mid_years, distances) arrays from lib/con comparison."""
    mid_years, dists = [], []
    for w in windows:
        comp = compare_matrices(w.networks["lib"], w.networks["con"])
        mid_years.append(w.mid_year)
        dists.append(comp["euclidean_distance"])
    return np.array(mid_years), np.array(dists)


def compute_centrality_rho_series(windows):
    """Return (mid_years, rhos) arrays of Spearman rho between lib/con degree centrality."""
    mid_years, rhos = [], []
    for w in windows:
        G_lib = build_graph(w.networks["lib"])
        G_con = build_graph(w.networks["con"])
        deg_lib = nx.degree_centrality(G_lib)
        deg_con = nx.degree_centrality(G_con)
        common = sorted(set(deg_lib.keys()) & set(deg_con.keys()))
        if len(common) < 10:
            continue
        rho, _ = spearmanr([deg_lib[n] for n in common], [deg_con[n] for n in common])
        mid_years.append(w.mid_year)
        rhos.append(rho)
    return np.array(mid_years), np.array(rhos)


def compute_sign_disagreement_series(windows):
    """Return (mid_years, fracs) of sign disagreement fraction per window."""
    mid_years, fracs = [], []
    for w in windows:
        lib_mat = w.networks["lib"].values
        con_mat = w.networks["con"].values
        n_vars = len(w.common_vars)
        triu = np.triu_indices(n_vars, k=1)
        lib_upper = lib_mat[triu]
        con_upper = con_mat[triu]
        both_nonzero = (lib_upper != 0) & (con_upper != 0)
        n_both = both_nonzero.sum()
        n_disagree = (both_nonzero & (np.sign(lib_upper) != np.sign(con_upper))).sum()
        mid_years.append(w.mid_year)
        fracs.append(n_disagree / n_both if n_both > 0 else 0)
    return np.array(mid_years), np.array(fracs)


def compute_density_series(windows, group):
    """Return (mid_years, densities) for a given group ('lib' or 'con')."""
    mid_years, densities = [], []
    for w in windows:
        G = build_graph(w.networks[group])
        mid_years.append(w.mid_year)
        densities.append(nx.density(G))
    return np.array(mid_years), np.array(densities)


def compute_clustering_series(windows, group):
    """Return (mid_years, clusterings) for a given group."""
    mid_years, clusterings = [], []
    for w in windows:
        G = build_graph(w.networks[group])
        mid_years.append(w.mid_year)
        clusterings.append(nx.average_clustering(G, weight="weight"))
    return np.array(mid_years), np.array(clusterings)


def compute_per_variable_centrality_slopes(windows, hac_maxlags=1):
    """Compute per-variable degree centrality rank-difference slopes with HAC.

    Returns DataFrame with variable, slope, p_value, p_ols columns.
    """
    cent_data = {}  # var -> list of (mid_year, |rank_diff|)
    for w in windows:
        G_lib = build_graph(w.networks["lib"])
        G_con = build_graph(w.networks["con"])
        deg_lib = nx.degree_centrality(G_lib)
        deg_con = nx.degree_centrality(G_con)
        common = sorted(set(deg_lib.keys()) & set(deg_con.keys()))
        lib_ranked = pd.Series({n: deg_lib[n] for n in common}).rank(ascending=False)
        con_ranked = pd.Series({n: deg_con[n] for n in common}).rank(ascending=False)
        for v in common:
            cent_data.setdefault(v, []).append((w.mid_year, abs(lib_ranked[v] - con_ranked[v])))

    rows = []
    for v, ts in cent_data.items():
        if len(ts) < 5:
            continue
        years = np.array([t[0] for t in ts])
        vals = np.array([t[1] for t in ts])
        hac = hac_regression(years, vals, maxlags=hac_maxlags)
        ols = ols_regression(years, vals)
        rows.append({
            "variable": v,
            "slope": hac["slope"],
            "p_value": hac["p_value"],
            "p_ols": ols["p_value"],
            "r_squared": hac["r_squared"],
            "mean_abs_rank_diff": np.mean(vals),
        })
    return pd.DataFrame(rows).sort_values("slope", ascending=False)


def compute_per_variable_degree_slopes(windows, group, hac_maxlags=1):
    """Compute per-variable degree centrality slopes for a single group with HAC.

    Returns DataFrame with variable, slope, p_value, p_ols columns.
    """
    deg_data = {}  # var -> list of (mid_year, degree_centrality)
    for w in windows:
        G = build_graph(w.networks[group])
        deg = nx.degree_centrality(G)
        for v, d in deg.items():
            deg_data.setdefault(v, []).append((w.mid_year, d))

    rows = []
    for v, ts in deg_data.items():
        if len(ts) < 5:
            continue
        years = np.array([t[0] for t in ts])
        vals = np.array([t[1] for t in ts])
        hac = hac_regression(years, vals, maxlags=hac_maxlags)
        ols = ols_regression(years, vals)
        rows.append({
            "variable": v,
            "slope": hac["slope"],
            "p_value": hac["p_value"],
            "p_ols": ols["p_value"],
            "r_squared": hac["r_squared"],
        })
    return pd.DataFrame(rows).sort_values("slope", ascending=False)


def segmented_regression(x, y):
    """Grid-search breakpoints for piecewise linear model, compare to linear via AIC.

    Returns dict with best_breakpoint, aic_linear, aic_segmented, is_better,
    slope_before, slope_after, p_break.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    # Linear model
    X_lin = sm.add_constant(x)
    model_lin = sm.OLS(y, X_lin).fit()
    aic_lin = model_lin.aic

    # Grid search breakpoints (exclude first/last 3 points)
    best_aic = np.inf
    best_bp = None
    best_model = None
    candidates = x[3:-3] if n > 8 else x[2:-2]

    for bp in candidates:
        x_before = np.maximum(0, bp - x)  # 0 after breakpoint
        x_after = np.maximum(0, x - bp)    # 0 before breakpoint
        X_seg = np.column_stack([np.ones(n), x_before, x_after])
        try:
            model_seg = sm.OLS(y, X_seg).fit()
            if model_seg.aic < best_aic:
                best_aic = model_seg.aic
                best_bp = bp
                best_model = model_seg
        except Exception:
            continue

    if best_model is None:
        return {
            "best_breakpoint": None,
            "aic_linear": aic_lin,
            "aic_segmented": np.nan,
            "is_better": False,
            "slope_before": model_lin.params[1],
            "slope_after": model_lin.params[1],
            "p_break": np.nan,
        }

    # Slopes: before breakpoint = -coef on x_before, after = +coef on x_after
    slope_before = -best_model.params[1]
    slope_after = best_model.params[2]

    # F-test comparing nested models
    ssr_full = best_model.ssr
    ssr_reduced = model_lin.ssr
    df_extra = best_model.df_model - model_lin.df_model
    df_resid = best_model.df_resid
    if df_extra > 0 and df_resid > 0 and ssr_full > 0:
        f_stat = ((ssr_reduced - ssr_full) / df_extra) / (ssr_full / df_resid)
        from scipy.stats import f as f_dist
        p_break = 1 - f_dist.cdf(f_stat, df_extra, df_resid)
    else:
        p_break = np.nan

    return {
        "best_breakpoint": best_bp,
        "aic_linear": aic_lin,
        "aic_segmented": best_aic,
        "is_better": best_aic < aic_lin - 2,  # AIC improvement > 2
        "slope_before": slope_before,
        "slope_after": slope_after,
        "p_break": p_break,
    }


def fmt_p(p):
    """Format p-value for display."""
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ════════════════════════════════════════════════════════════════════


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cleaned_df = clean_datasets()

    # ── Step 0: Determine fixed variables (excl. POLVIEWS/PARTYID) ──
    print("\n" + "=" * 70)
    print("STEP 0: DETERMINE FIXED VARIABLES (excl. POLVIEWS/PARTYID)")
    print("=" * 70)

    # First pass: unfixed windows to find intersection
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
    all_fixed = sorted(set.intersection(*[set(w.common_vars) for w in windows_unfixed]))
    vars_no_pol = [v for v in all_fixed if v not in ("POLVIEWS", "PARTYID")]
    print(f"Fixed variables (intersection): {len(all_fixed)}")
    print(f"After excluding POLVIEWS/PARTYID: {len(vars_no_pol)}")

    # ── Build windows ───────────────────────────────────────────────
    print("\nBuilding overlapping windows (step=2)...")
    windows_overlap = build_rolling_windows(
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
    print(f"Overlapping windows: {len(windows_overlap)}")

    print("\nBuilding non-overlapping windows (step=4)...")
    windows_nonoverlap = build_rolling_windows(
        cleaned_df,
        window_size=4, step_size=4, min_years_per_window=3,
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
    print(f"Non-overlapping windows: {len(windows_nonoverlap)}")

    # ════════════════════════════════════════════════════════════════
    # PART A: HAC-corrected trend regressions (overlapping windows)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART A: HAC-CORRECTED TREND REGRESSIONS (overlapping, step=2)")
    print("=" * 70)

    results_a = {}

    # A1: Euclidean distance trend
    mid_o, euc_o = compute_euclidean_distances(windows_overlap)
    hac_euc = hac_regression(mid_o, euc_o, maxlags=1)
    ols_euc = ols_regression(mid_o, euc_o)
    results_a["euc_dist"] = {"hac": hac_euc, "ols": ols_euc, "x": mid_o, "y": euc_o}
    print(f"\nA1. Euclidean distance trend:")
    print(f"  OLS:  slope={ols_euc['slope']:.5f}, p={fmt_p(ols_euc['p_value'])}, r={ols_euc['r']:.3f}")
    print(f"  HAC:  slope={hac_euc['slope']:.5f}, p={fmt_p(hac_euc['p_value'])}, "
          f"CI=[{hac_euc['ci_lower']:.5f}, {hac_euc['ci_upper']:.5f}], DW={hac_euc['durbin_watson']:.3f}")

    # A2: Centrality rho trend
    mid_rho_o, rho_o = compute_centrality_rho_series(windows_overlap)
    hac_rho = hac_regression(mid_rho_o, rho_o, maxlags=1)
    ols_rho = ols_regression(mid_rho_o, rho_o)
    results_a["cent_rho"] = {"hac": hac_rho, "ols": ols_rho, "x": mid_rho_o, "y": rho_o}
    print(f"\nA2. Centrality rho trend:")
    print(f"  OLS:  slope={ols_rho['slope']:.5f}, p={fmt_p(ols_rho['p_value'])}, r={ols_rho['r']:.3f}")
    print(f"  HAC:  slope={hac_rho['slope']:.5f}, p={fmt_p(hac_rho['p_value'])}, "
          f"CI=[{hac_rho['ci_lower']:.5f}, {hac_rho['ci_upper']:.5f}], DW={hac_rho['durbin_watson']:.3f}")

    # A3: Sign disagreement trend
    mid_sign_o, sign_o = compute_sign_disagreement_series(windows_overlap)
    # Sign disagreement may be constant (all zeros) if LASSO edges never disagree in sign
    if np.std(sign_o) == 0:
        print(f"\nA3. Sign disagreement: CONSTANT at {sign_o[0]:.6f} across all windows (no trend to test)")
        hac_sign = {"slope": 0, "intercept": sign_o[0], "p_value": 1.0,
                     "ci_lower": 0, "ci_upper": 0, "r_squared": 0, "durbin_watson": np.nan,
                     "n_obs": len(sign_o), "std_err": 0}
        ols_sign = {"slope": 0, "intercept": sign_o[0], "p_value": 1.0,
                     "r_squared": 0, "std_err": 0, "r": 0, "n_obs": len(sign_o)}
    else:
        hac_sign = hac_regression(mid_sign_o, sign_o, maxlags=1)
        ols_sign = ols_regression(mid_sign_o, sign_o)
        print(f"\nA3. Sign disagreement trend:")
        print(f"  OLS:  slope={ols_sign['slope']:.6f}, p={fmt_p(ols_sign['p_value'])}, r={ols_sign['r']:.3f}")
        print(f"  HAC:  slope={hac_sign['slope']:.6f}, p={fmt_p(hac_sign['p_value'])}, "
              f"CI=[{hac_sign['ci_lower']:.6f}, {hac_sign['ci_upper']:.6f}], DW={hac_sign['durbin_watson']:.3f}")
    results_a["sign_disagree"] = {"hac": hac_sign, "ols": ols_sign, "x": mid_sign_o, "y": sign_o}

    # A4: Density trends (lib, con)
    for group in ["lib", "con"]:
        mid_d, dens = compute_density_series(windows_overlap, group)
        hac_d = hac_regression(mid_d, dens, maxlags=1)
        ols_d = ols_regression(mid_d, dens)
        results_a[f"density_{group}"] = {"hac": hac_d, "ols": ols_d, "x": mid_d, "y": dens}
        print(f"\nA4. Density trend ({group}):")
        print(f"  OLS:  slope={ols_d['slope']:.6f}, p={fmt_p(ols_d['p_value'])}")
        print(f"  HAC:  slope={hac_d['slope']:.6f}, p={fmt_p(hac_d['p_value'])}, DW={hac_d['durbin_watson']:.3f}")

    # A5: Clustering trends (lib, con)
    for group in ["lib", "con"]:
        mid_c, clust = compute_clustering_series(windows_overlap, group)
        hac_c = hac_regression(mid_c, clust, maxlags=1)
        ols_c = ols_regression(mid_c, clust)
        results_a[f"clustering_{group}"] = {"hac": hac_c, "ols": ols_c, "x": mid_c, "y": clust}
        print(f"\nA5. Clustering trend ({group}):")
        print(f"  OLS:  slope={ols_c['slope']:.6f}, p={fmt_p(ols_c['p_value'])}")
        print(f"  HAC:  slope={hac_c['slope']:.6f}, p={fmt_p(hac_c['p_value'])}, DW={hac_c['durbin_watson']:.3f}")

    # ════════════════════════════════════════════════════════════════
    # PART B: Non-overlapping window validation (step=4)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART B: NON-OVERLAPPING WINDOW VALIDATION (step=4)")
    print("=" * 70)

    results_b = {}

    # B1: Euclidean distance
    mid_no, euc_no = compute_euclidean_distances(windows_nonoverlap)
    ols_euc_no = ols_regression(mid_no, euc_no)
    results_b["euc_dist"] = {"ols": ols_euc_no, "x": mid_no, "y": euc_no}
    print(f"\nB1. Euclidean distance (non-overlapping, n={len(mid_no)}):")
    print(f"  OLS:  slope={ols_euc_no['slope']:.5f}, p={fmt_p(ols_euc_no['p_value'])}, r={ols_euc_no['r']:.3f}")

    # B2: Centrality rho
    mid_rho_no, rho_no = compute_centrality_rho_series(windows_nonoverlap)
    ols_rho_no = ols_regression(mid_rho_no, rho_no)
    results_b["cent_rho"] = {"ols": ols_rho_no, "x": mid_rho_no, "y": rho_no}
    print(f"\nB2. Centrality rho (non-overlapping, n={len(mid_rho_no)}):")
    print(f"  OLS:  slope={ols_rho_no['slope']:.5f}, p={fmt_p(ols_rho_no['p_value'])}, r={ols_rho_no['r']:.3f}")

    # B3: Sign disagreement
    mid_sign_no, sign_no = compute_sign_disagreement_series(windows_nonoverlap)
    if np.std(sign_no) == 0:
        print(f"\nB3. Sign disagreement (non-overlapping): CONSTANT at {sign_no[0]:.6f}")
        ols_sign_no = {"slope": 0, "intercept": sign_no[0], "p_value": 1.0,
                        "r_squared": 0, "std_err": 0, "r": 0, "n_obs": len(sign_no)}
    else:
        ols_sign_no = ols_regression(mid_sign_no, sign_no)
        print(f"\nB3. Sign disagreement (non-overlapping, n={len(mid_sign_no)}):")
        print(f"  OLS:  slope={ols_sign_no['slope']:.6f}, p={fmt_p(ols_sign_no['p_value'])}, r={ols_sign_no['r']:.3f}")
    results_b["sign_disagree"] = {"ols": ols_sign_no, "x": mid_sign_no, "y": sign_no}

    # B4: Density
    for group in ["lib", "con"]:
        mid_d_no, dens_no = compute_density_series(windows_nonoverlap, group)
        ols_d_no = ols_regression(mid_d_no, dens_no)
        results_b[f"density_{group}"] = {"ols": ols_d_no, "x": mid_d_no, "y": dens_no}
        print(f"\nB4. Density ({group}, non-overlapping, n={len(mid_d_no)}):")
        print(f"  OLS:  slope={ols_d_no['slope']:.6f}, p={fmt_p(ols_d_no['p_value'])}")

    # B5: Clustering
    for group in ["lib", "con"]:
        mid_c_no, clust_no = compute_clustering_series(windows_nonoverlap, group)
        ols_c_no = ols_regression(mid_c_no, clust_no)
        results_b[f"clustering_{group}"] = {"ols": ols_c_no, "x": mid_c_no, "y": clust_no}
        print(f"\nB5. Clustering ({group}, non-overlapping, n={len(mid_c_no)}):")
        print(f"  OLS:  slope={ols_c_no['slope']:.6f}, p={fmt_p(ols_c_no['p_value'])}")

    # ════════════════════════════════════════════════════════════════
    # PART C: FDR correction for variable-level tests
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART C: FDR CORRECTION FOR VARIABLE-LEVEL TESTS")
    print("=" * 70)

    # C1: Per-variable centrality rank-difference slopes
    print("\nC1. Per-variable centrality rank-difference slopes...")
    df_cent_slopes = compute_per_variable_centrality_slopes(windows_overlap, hac_maxlags=1)
    n_cent_vars = len(df_cent_slopes)

    if n_cent_vars > 0:
        rejected_cent, p_corrected_cent, _, _ = multipletests(
            df_cent_slopes["p_value"].values, method="fdr_bh"
        )
        df_cent_slopes["p_fdr"] = p_corrected_cent
        df_cent_slopes["sig_fdr"] = rejected_cent

        n_sig_ols = (df_cent_slopes["p_ols"] < 0.05).sum()
        n_sig_hac = (df_cent_slopes["p_value"] < 0.05).sum()
        n_sig_fdr = rejected_cent.sum()

        print(f"  Variables tested: {n_cent_vars}")
        print(f"  Significant (OLS p<0.05): {n_sig_ols}")
        print(f"  Significant (HAC p<0.05): {n_sig_hac}")
        print(f"  Significant (FDR q<0.05): {n_sig_fdr}")

        if n_sig_fdr > 0:
            print(f"\n  Top FDR-surviving variables (centrality rank-diff):")
            surviving = df_cent_slopes[df_cent_slopes["sig_fdr"]].sort_values("slope", ascending=False)
            for _, row in surviving.head(10).iterrows():
                print(f"    {row['variable']:30s} slope={row['slope']:.4f}, "
                      f"p_HAC={fmt_p(row['p_value'])}, q_FDR={fmt_p(row['p_fdr'])}")

    # C2: Per-variable degree centrality slopes (total network via lib+con combined isn't available,
    # so test lib and con separately)
    for group in ["lib", "con"]:
        print(f"\nC2. Per-variable degree slopes ({group})...")
        df_deg_slopes = compute_per_variable_degree_slopes(windows_overlap, group, hac_maxlags=1)
        n_deg_vars = len(df_deg_slopes)

        if n_deg_vars > 0:
            rejected_deg, p_corrected_deg, _, _ = multipletests(
                df_deg_slopes["p_value"].values, method="fdr_bh"
            )
            df_deg_slopes["p_fdr"] = p_corrected_deg
            df_deg_slopes["sig_fdr"] = rejected_deg

            n_sig_ols_d = (df_deg_slopes["p_ols"] < 0.05).sum()
            n_sig_hac_d = (df_deg_slopes["p_value"] < 0.05).sum()
            n_sig_fdr_d = rejected_deg.sum()

            print(f"  Variables tested: {n_deg_vars}")
            print(f"  Significant (OLS p<0.05): {n_sig_ols_d}")
            print(f"  Significant (HAC p<0.05): {n_sig_hac_d}")
            print(f"  Significant (FDR q<0.05): {n_sig_fdr_d}")

    # ════════════════════════════════════════════════════════════════
    # PART D: Structural break tests
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART D: STRUCTURAL BREAK TESTS")
    print("=" * 70)

    # D1: Euclidean distance breakpoint
    seg_euc = segmented_regression(mid_o, euc_o)
    print(f"\nD1. Euclidean distance — structural break test:")
    print(f"  AIC (linear):    {seg_euc['aic_linear']:.2f}")
    print(f"  AIC (segmented): {seg_euc['aic_segmented']:.2f}")
    print(f"  Best breakpoint: {seg_euc['best_breakpoint']}")
    print(f"  Segmented better (delta AIC > 2): {seg_euc['is_better']}")
    print(f"  Slope before break: {seg_euc['slope_before']:.5f}")
    print(f"  Slope after break:  {seg_euc['slope_after']:.5f}")
    print(f"  F-test p-value:     {fmt_p(seg_euc['p_break']) if not np.isnan(seg_euc['p_break']) else 'N/A'}")

    # D2: Centrality rho breakpoint
    seg_rho = segmented_regression(mid_rho_o, rho_o)
    print(f"\nD2. Centrality rho — structural break test:")
    print(f"  AIC (linear):    {seg_rho['aic_linear']:.2f}")
    print(f"  AIC (segmented): {seg_rho['aic_segmented']:.2f}")
    print(f"  Best breakpoint: {seg_rho['best_breakpoint']}")
    print(f"  Segmented better (delta AIC > 2): {seg_rho['is_better']}")
    print(f"  Slope before break: {seg_rho['slope_before']:.5f}")
    print(f"  Slope after break:  {seg_rho['slope_after']:.5f}")
    print(f"  F-test p-value:     {fmt_p(seg_rho['p_break']) if not np.isnan(seg_rho['p_break']) else 'N/A'}")

    # ════════════════════════════════════════════════════════════════
    # PART E: Summary figure + comparison table + writeup
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART E: SUMMARY")
    print("=" * 70)

    # ── Summary table ───────────────────────────────────────────
    print("\n=== OLS vs HAC vs Non-Overlapping Comparison ===")
    header = f"{'Metric':<25s} {'OLS slope':>10s} {'OLS p':>10s} {'HAC slope':>10s} {'HAC p':>10s} {'DW':>6s} {'NoOv slope':>11s} {'NoOv p':>10s}"
    print(header)
    print("-" * len(header))

    comparison_rows = []
    for key, label in [
        ("euc_dist", "Euclidean distance"),
        ("cent_rho", "Centrality rho"),
        ("sign_disagree", "Sign disagreement"),
        ("density_lib", "Density (lib)"),
        ("density_con", "Density (con)"),
        ("clustering_lib", "Clustering (lib)"),
        ("clustering_con", "Clustering (con)"),
    ]:
        ra = results_a[key]
        rb = results_b.get(key, {})
        ols_p = ra["ols"]["p_value"]
        hac_p = ra["hac"]["p_value"]
        noov_p = rb["ols"]["p_value"] if "ols" in rb else np.nan
        noov_slope = rb["ols"]["slope"] if "ols" in rb else np.nan

        row = {
            "metric": label,
            "ols_slope": ra["ols"]["slope"],
            "ols_p": ols_p,
            "hac_slope": ra["hac"]["slope"],
            "hac_p": hac_p,
            "dw": ra["hac"]["durbin_watson"],
            "noov_slope": noov_slope,
            "noov_p": noov_p,
        }
        comparison_rows.append(row)

        dw_val = ra["hac"]["durbin_watson"]
        dw_str = f"{dw_val:>6.3f}" if not np.isnan(dw_val) else "   N/A"
        print(f"{label:<25s} {ra['ols']['slope']:>10.5f} {fmt_p(ols_p):>10s} "
              f"{ra['hac']['slope']:>10.5f} {fmt_p(hac_p):>10s} "
              f"{dw_str} "
              f"{noov_slope:>11.5f} {fmt_p(noov_p):>10s}")

    # ── 4-panel figure ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Helper: compute CI band pivoting through (mean_x, mean_y)
    def ci_band(x_arr, result_hac):
        """Compute 95% CI band for the fitted line using slope CI bounds."""
        mean_x = np.mean(x_arr)
        mean_y = result_hac["intercept"] + result_hac["slope"] * mean_x
        lo = mean_y + result_hac["ci_lower"] * (x_arr - mean_x)
        hi = mean_y + result_hac["ci_upper"] * (x_arr - mean_x)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    # Panel A: Euclidean distance with HAC CI + non-overlapping overlay
    ax = axes[0, 0]
    ax.plot(mid_o, euc_o, "ko-", markersize=4, linewidth=1.5, label="Overlapping (step=2)")
    # HAC trend line
    x_trend = np.linspace(mid_o.min(), mid_o.max(), 100)
    y_trend = hac_euc["intercept"] + hac_euc["slope"] * x_trend
    ax.plot(x_trend, y_trend, "r-", linewidth=2,
            label=f"HAC: slope={hac_euc['slope']:.4f}, p={fmt_p(hac_euc['p_value'])}")
    # HAC 95% CI band
    ci_lo, ci_hi = ci_band(x_trend, hac_euc)
    ax.fill_between(x_trend, ci_lo, ci_hi, alpha=0.15, color="red", label="95% CI (slope)")
    # Non-overlapping
    ax.plot(mid_no, euc_no, "b^", markersize=8, alpha=0.7, zorder=5,
            label=f"Non-overlap (step=4): slope={ols_euc_no['slope']:.4f}, p={fmt_p(ols_euc_no['p_value'])}")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Euclidean Distance (lib vs con)")
    ax.set_title("A. Euclidean Distance: HAC-Corrected Trend", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel B: Centrality rho with HAC CI + non-overlapping overlay
    ax = axes[0, 1]
    ax.plot(mid_rho_o, rho_o, "ko-", markersize=4, linewidth=1.5, label="Overlapping (step=2)")
    x_trend_r = np.linspace(mid_rho_o.min(), mid_rho_o.max(), 100)
    y_trend_r = hac_rho["intercept"] + hac_rho["slope"] * x_trend_r
    ax.plot(x_trend_r, y_trend_r, "r-", linewidth=2,
            label=f"HAC: slope={hac_rho['slope']:.4f}, p={fmt_p(hac_rho['p_value'])}")
    ci_lo_r, ci_hi_r = ci_band(x_trend_r, hac_rho)
    ax.fill_between(x_trend_r, ci_lo_r, ci_hi_r, alpha=0.15, color="red", label="95% CI (slope)")
    ax.plot(mid_rho_no, rho_no, "b^", markersize=8, alpha=0.7, zorder=5,
            label=f"Non-overlap: slope={ols_rho_no['slope']:.4f}, p={fmt_p(ols_rho_no['p_value'])}")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Spearman rho (lib vs con centrality)")
    ax.set_title("B. Centrality Rho: HAC-Corrected Trend", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel C: P-value comparison bar chart
    ax = axes[1, 0]
    metrics = ["Euc. dist.", "Cent. rho", "Sign disagr.", "Dens. (lib)", "Dens. (con)",
               "Clust. (lib)", "Clust. (con)"]
    keys = ["euc_dist", "cent_rho", "sign_disagree", "density_lib", "density_con",
            "clustering_lib", "clustering_con"]
    ols_ps = [results_a[k]["ols"]["p_value"] for k in keys]
    hac_ps = [results_a[k]["hac"]["p_value"] for k in keys]
    noov_ps = [results_b[k]["ols"]["p_value"] for k in keys]

    x_pos = np.arange(len(metrics))
    width = 0.25
    # Use -log10(p) for visual comparison, cap at 6 for readability
    def neg_log_p(p):
        return min(-np.log10(max(p, 1e-10)), 6)

    ax.bar(x_pos - width, [neg_log_p(p) for p in ols_ps], width, label="OLS", color="gray", alpha=0.7)
    ax.bar(x_pos, [neg_log_p(p) for p in hac_ps], width, label="HAC", color="steelblue", alpha=0.7)
    ax.bar(x_pos + width, [neg_log_p(p) for p in noov_ps], width, label="Non-overlap", color="seagreen", alpha=0.7)
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p=0.05")
    ax.axhline(-np.log10(0.01), color="orange", linestyle=":", linewidth=1, label="p=0.01")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("C. P-Value Comparison: OLS vs HAC vs Non-Overlap", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel D: Structural break test — Euclidean distance with piecewise fit
    ax = axes[1, 1]
    ax.plot(mid_o, euc_o, "ko-", markersize=4, linewidth=1.5, label="Euclidean distance")
    # Linear fit
    y_lin = hac_euc["intercept"] + hac_euc["slope"] * mid_o
    ax.plot(mid_o, y_lin, "b--", linewidth=1.5, alpha=0.6, label="Linear fit")
    # Segmented fit (if better)
    if seg_euc["is_better"] and seg_euc["best_breakpoint"] is not None:
        bp = seg_euc["best_breakpoint"]
        x_before = np.maximum(0, bp - mid_o)
        x_after = np.maximum(0, mid_o - bp)
        X_seg = np.column_stack([np.ones(len(mid_o)), x_before, x_after])
        model_seg = sm.OLS(euc_o, X_seg).fit()
        y_seg = model_seg.predict(X_seg)
        ax.plot(mid_o, y_seg, "r-", linewidth=2,
                label=f"Segmented (break={bp:.0f}, dAIC={seg_euc['aic_linear'] - seg_euc['aic_segmented']:.1f})")
        ax.axvline(bp, color="red", linestyle=":", alpha=0.5)
    else:
        ax.text(0.5, 0.95, "No significant structural break\n(segmented model not preferred by AIC)",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Euclidean Distance")
    ax.set_title("D. Structural Break Test: Euclidean Distance", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Corrected Statistical Inference ({len(vars_no_pol)} fixed vars, excl. POLVIEWS/PARTYID)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "sound_08_corrected_inference.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_path}")

    # ── Analysis writeup ────────────────────────────────────────
    # Build FDR summary strings
    fdr_cent_str = ""
    if n_cent_vars > 0:
        fdr_cent_str = (
            f"- Variables tested: {n_cent_vars}\n"
            f"- Significant (OLS p<0.05): {n_sig_ols}\n"
            f"- Significant (HAC p<0.05): {n_sig_hac}\n"
            f"- Significant (FDR q<0.05): {n_sig_fdr}\n"
        )
        if n_sig_fdr > 0:
            fdr_cent_str += "\nFDR-surviving variables (centrality rank-difference slopes):\n"
            surviving = df_cent_slopes[df_cent_slopes["sig_fdr"]].sort_values("slope", ascending=False)
            for _, row in surviving.head(10).iterrows():
                fdr_cent_str += f"  - {row['variable']}: slope={row['slope']:.4f}, q={fmt_p(row['p_fdr'])}\n"

    # Build comparison table for writeup
    comp_table = "| Metric | OLS slope | OLS p | HAC slope | HAC p | DW | Non-overlap slope | Non-overlap p |\n"
    comp_table += "|--------|-----------|-------|-----------|-------|----|-------------------|---------------|\n"
    for row in comparison_rows:
        dw_md = f"{row['dw']:.3f}" if not np.isnan(row['dw']) else "N/A"
        comp_table += (
            f"| {row['metric']} | {row['ols_slope']:.5f} | {fmt_p(row['ols_p'])} | "
            f"{row['hac_slope']:.5f} | {fmt_p(row['hac_p'])} | {dw_md} | "
            f"{row['noov_slope']:.5f} | {fmt_p(row['noov_p'])} |\n"
        )

    # Assess headline survival
    euc_survives_hac = hac_euc["p_value"] < 0.05
    rho_survives_hac = hac_rho["p_value"] < 0.05
    euc_survives_noov = ols_euc_no["p_value"] < 0.05
    rho_survives_noov = ols_rho_no["p_value"] < 0.05

    euc_verdict = "SURVIVES" if (euc_survives_hac and euc_survives_noov) else (
        "WEAKENED" if (euc_survives_hac or euc_survives_noov) else "FAILS")
    rho_verdict = "SURVIVES" if (rho_survives_hac and rho_survives_noov) else (
        "WEAKENED" if (rho_survives_hac or rho_survives_noov) else "FAILS")

    euc_break_str = "No" if not seg_euc["is_better"] else (
        f"Yes (breakpoint ~{seg_euc['best_breakpoint']:.0f}, "
        f"slope before={seg_euc['slope_before']:.5f}, after={seg_euc['slope_after']:.5f})"
    )
    rho_break_str = "No" if not seg_rho["is_better"] else (
        f"Yes (breakpoint ~{seg_rho['best_breakpoint']:.0f}, "
        f"slope before={seg_rho['slope_before']:.5f}, after={seg_rho['slope_after']:.5f})"
    )

    writeup = f"""# Corrected Statistical Inference

## Problem

All trend p-values in previous analyses are potentially invalid because:

1. **Overlapping windows**: 22 windows (step=2, width=4) share 50% of data between
   adjacent windows, violating the independence assumption and inflating OLS standard
   errors. As the statistician reviewer noted: "p < 0.0001 could become p ~ 0.02-0.08."

2. **Multiple comparisons**: ~4,200 implicit edge-level and variable-level tests have
   zero correction, inflating family-wise error rate.

3. **Acceleration claims**: No formal structural break test supports claims of
   "accelerating" divergence.

## Methods

### HAC Correction (Part A)

Newey-West HAC standard errors with maxlags=1 on all headline trend regressions
using overlapping windows (step=2, n={len(windows_overlap)}). This accounts for the
serial autocorrelation induced by window overlap. Durbin-Watson statistics are
reported to verify autocorrelation structure.

### Non-Overlapping Validation (Part B)

Independent replication using step=4 windows (n={len(windows_nonoverlap)}). With no
window overlap, observations are independent and plain OLS is valid. Lower power due
to fewer observations, but any surviving trend is free from autocorrelation bias.

### FDR Correction (Part C)

Benjamini-Hochberg false discovery rate correction applied to all per-variable tests:
- Centrality rank-difference slopes ({n_cent_vars} tests)
- Per-variable degree centrality slopes (lib and con, ~{n_cent_vars} each)

### Structural Break Tests (Part D)

Grid-search segmented regression compared to linear fit via AIC. Tests whether
the Euclidean distance and centrality rho trends show a structural break (acceleration
or deceleration) rather than a simple linear trend.

### Configuration

All analyses use:
- Fixed variables: {len(vars_no_pol)} (intersection across all windows, excluding POLVIEWS and PARTYID)
- Regularized partial Pearson correlation (alpha=0.2)
- Sample-matched liberal vs conservative groups
- Random state: 42

---

## Results

### Part A: HAC-Corrected Trends

{comp_table}

**Key findings:**
- Euclidean distance: OLS p={fmt_p(ols_euc['p_value'])} -> HAC p={fmt_p(hac_euc['p_value'])} (DW={hac_euc['durbin_watson']:.3f})
- Centrality rho: OLS p={fmt_p(ols_rho['p_value'])} -> HAC p={fmt_p(hac_rho['p_value'])} (DW={hac_rho['durbin_watson']:.3f})
- Sign disagreement: OLS p={fmt_p(ols_sign['p_value'])} -> HAC p={fmt_p(hac_sign['p_value'])} (DW={hac_sign['durbin_watson']:.3f})

Durbin-Watson interpretation: DW << 2 confirms positive autocorrelation from window
overlap, justifying the HAC correction. Values closer to 2 for non-overlapping windows
would further confirm this.

### Part B: Non-Overlapping Validation

- Euclidean distance: slope={ols_euc_no['slope']:.5f}, p={fmt_p(ols_euc_no['p_value'])}, r={ols_euc_no['r']:.3f}
- Centrality rho: slope={ols_rho_no['slope']:.5f}, p={fmt_p(ols_rho_no['p_value'])}, r={ols_rho_no['r']:.3f}
- Sign disagreement: slope={ols_sign_no['slope']:.6f}, p={fmt_p(ols_sign_no['p_value'])}, r={ols_sign_no['r']:.3f}

Directional agreement: slopes from non-overlapping windows {"agree" if (np.sign(ols_euc_no['slope']) == np.sign(hac_euc['slope'])) else "DISAGREE"} with overlapping-window slopes for Euclidean distance.

### Part C: FDR-Corrected Variable-Level Tests

**Centrality rank-difference slopes:**
{fdr_cent_str}

### Part D: Structural Break Tests

- Euclidean distance: acceleration detected = {euc_break_str}
- Centrality rho: acceleration detected = {rho_break_str}

---

## Verdict

| Claim | HAC (p<0.05) | Non-overlap (p<0.05) | Overall |
|-------|:---:|:---:|:---:|
| Lib/con diverging (Euclidean) | {"Yes" if euc_survives_hac else "No"} | {"Yes" if euc_survives_noov else "No"} | **{euc_verdict}** |
| Centrality misalignment | {"Yes" if rho_survives_hac else "No"} | {"Yes" if rho_survives_noov else "No"} | **{rho_verdict}** |

### Implications for the Paper

"""

    # Dynamic verdict
    if euc_verdict == "SURVIVES" and rho_verdict == "SURVIVES":
        writeup += """Both headline findings survive HAC correction and non-overlapping validation.
The original p-values were inflated by serial autocorrelation, but the corrected
p-values remain significant. The paper can report HAC-corrected statistics with
confidence. Report non-overlapping validation as supplementary evidence.
"""
    elif euc_verdict == "SURVIVES":
        writeup += """The Euclidean distance (divergence) trend survives both corrections. The centrality
rho trend is weakened. The paper should lead with Euclidean distance as the primary
metric and present centrality results with appropriate caveats about reduced
significance under HAC correction.
"""
    elif rho_verdict == "SURVIVES":
        writeup += """The centrality rho trend survives but the Euclidean distance trend is weakened.
The paper should reframe around structural misalignment (centrality divergence)
rather than overall divergence magnitude.
"""
    else:
        writeup += """Neither headline finding clearly survives both HAC correction and non-overlapping
validation. The paper needs substantial revision. Consider: (1) presenting results
with honest uncertainty, (2) reframing as suggestive evidence rather than strong
claims, (3) using the non-overlapping analysis as the primary specification.
"""

    writeup += f"""
### Recommended Reporting

For the paper, we recommend:
1. **Primary specification**: Non-overlapping windows (step=4) as the cleanest test
2. **Secondary specification**: Overlapping windows with HAC correction (step=2, maxlags=1)
3. **Report Durbin-Watson**: Shows the autocorrelation structure explicitly
4. **FDR correction**: All per-variable results should use FDR-corrected q-values
5. **Structural breaks**: {"Report segmented regression results" if seg_euc['is_better'] or seg_rho['is_better'] else "Acceleration claims should be dropped (linear model preferred by AIC)"}
"""

    writeup_path = ANALYSES_DIR / "2026-03_corrected-inference.md"
    with open(writeup_path, "w", encoding="utf-8") as f:
        f.write(writeup)
    print(f"Saved: {writeup_path}")

    print("\nDone. Corrected inference analysis complete.")


if __name__ == "__main__":
    main()
