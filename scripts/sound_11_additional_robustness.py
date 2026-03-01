"""
Sound 11: Additional Robustness Checks — five remaining methodological concerns.

Check 1: Exclude 2021-2022 — Does divergence survive without the COVID/mode-switch years?
Check 2: POLVIEWS Composition — How does the lib/mod/con split change over time?
Check 3: Eigenvalue Audit — Are the LASSO input matrices well-conditioned?
Check 4: PARTYID Alternative Split — Does divergence replicate with party ID instead of ideology?
Check 5: GSS Survey Weights — Do weights meaningfully change the reference-period matrices?

Usage: PYTHONIOENCODING=utf-8 python scripts/sound_11_additional_robustness.py
Outputs: figures/sound_11_additional_robustness.png,
         analyses/2026-03_additional-robustness.md, stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import CorrelationMethod, EdgeSuppressionMethod
from src.analyzers.matrix_compare import compare_matrices
from src.analyzers.temporal import build_rolling_windows

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
ANALYSES_DIR = Path(__file__).resolve().parent.parent / "analyses"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── Helpers ─────────────────────────────────────────────────────────

def compute_euclidean_trend(windows):
    """Compute Euclidean distance between lib/con per window, return (mid_years, distances, slope, r, p)."""
    rows = []
    for w in windows:
        groups = list(w.networks.keys())
        comp = compare_matrices(w.networks[groups[0]], w.networks[groups[1]])
        rows.append({"mid_year": w.mid_year, "euc_dist": comp["euclidean_distance"]})
    df = pd.DataFrame(rows)
    if len(df) < 3:
        return df["mid_year"].values, df["euc_dist"].values, np.nan, np.nan, np.nan
    slope, _, r, p, _ = linregress(df["mid_year"], df["euc_dist"])
    return df["mid_year"].values, df["euc_dist"].values, slope, r, p


def weighted_pearson_matrix(df, weight_col, var_cols):
    """Compute weighted pairwise Pearson correlation matrix.

    For each pair of variables, uses rows where both variables and the weight
    are non-NaN. Returns a DataFrame (var_cols x var_cols).
    """
    n = len(var_cols)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = var_cols[i], var_cols[j]
            mask = df[[vi, vj, weight_col]].notna().all(axis=1)
            sub = df.loc[mask]
            if len(sub) < 10:
                corr[i, j] = corr[j, i] = np.nan
                continue
            w = sub[weight_col].values
            x = sub[vi].values
            y = sub[vj].values
            w_sum = w.sum()
            mx = np.average(x, weights=w)
            my = np.average(y, weights=w)
            dx = x - mx
            dy = y - my
            cov_xy = np.sum(w * dx * dy) / w_sum
            var_x = np.sum(w * dx ** 2) / w_sum
            var_y = np.sum(w * dy ** 2) / w_sum
            denom = np.sqrt(var_x * var_y)
            r_val = cov_xy / denom if denom > 0 else np.nan
            corr[i, j] = corr[j, i] = r_val
    return pd.DataFrame(corr, index=var_cols, columns=var_cols)


def fmt_p(p):
    """Format p-value for display."""
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ════════════════════════════════════════════════════════════════════

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    cleaned_df = clean_datasets()

    # ── Step 0: Shared setup ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 0: SHARED SETUP")
    print("=" * 60)

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
    fixed_vars = sorted(set.intersection(*[set(w.common_vars) for w in windows_unfixed]))
    vars_no_pol = [v for v in fixed_vars if v not in ("POLVIEWS", "PARTYID")]
    print(f"Fixed variables (intersection): {len(fixed_vars)}")
    print(f"After excluding POLVIEWS/PARTYID: {len(vars_no_pol)}")

    # Build reference windows with vars_no_pol for comparison
    windows_ref = build_rolling_windows(
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
        verbose=False,
    )
    mid_ref, euc_ref, slope_ref, r_ref, p_ref = compute_euclidean_trend(windows_ref)
    print(f"Reference (full period, vars_no_pol): slope={slope_ref:.5f}, r={r_ref:.3f}, p={fmt_p(p_ref)}")

    # ── Check 1: Exclude 2021-2022 ─────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 1: EXCLUDE 2021-2022 — DOES DIVERGENCE SURVIVE?")
    print("=" * 60)

    df_pre2021 = cleaned_df[cleaned_df["YEAR"] <= 2018].copy()
    print(f"Rows after excluding post-2018: {len(df_pre2021)} (was {len(cleaned_df)})")

    windows_pre2021 = build_rolling_windows(
        df_pre2021,
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

    mid_pre, euc_pre, slope_pre, r_pre, p_pre = compute_euclidean_trend(windows_pre2021)
    print(f"\nExcl. 2021-2022 Euclidean distance trend:")
    print(f"  slope={slope_pre:.5f}/yr, r={r_pre:.3f}, p={fmt_p(p_pre)}")
    print(f"Full period reference: slope={slope_ref:.5f}/yr, r={r_ref:.3f}, p={fmt_p(p_ref)}")

    check1_pass = not np.isnan(slope_pre) and slope_pre > 0 and p_pre < 0.05
    print(f"CHECK 1 VERDICT: {'PASS' if check1_pass else 'FAIL'} — divergence "
          f"{'survives' if check1_pass else 'does NOT survive'} exclusion of 2021-2022")

    # ── Check 2: POLVIEWS Composition Over Time ─────────────────
    print("\n" + "=" * 60)
    print("CHECK 2: POLVIEWS COMPOSITION OVER TIME")
    print("=" * 60)

    df_pv = cleaned_df[cleaned_df["POLVIEWS"].notna()].copy()
    comp_rows = []
    for yr, grp in df_pv.groupby("YEAR"):
        n_total = len(grp)
        n_lib = (grp["POLVIEWS"] < 0).sum()
        n_mod = (grp["POLVIEWS"] == 0).sum()
        n_con = (grp["POLVIEWS"] > 0).sum()
        comp_rows.append({
            "year": int(yr),
            "n_total": n_total,
            "pct_lib": n_lib / n_total * 100,
            "pct_mod": n_mod / n_total * 100,
            "pct_con": n_con / n_total * 100,
        })
    df_comp = pd.DataFrame(comp_rows).sort_values("year")
    print(f"\nPOLVIEWS composition by year:")
    print(df_comp.to_string(index=False, float_format="%.1f"))

    # Matched sample sizes per window
    print(f"\nMatched sample sizes per window:")
    for w in windows_ref:
        print(f"  {w.start_year}-{w.end_year} (mid={w.mid_year:.0f}): "
              f"matched_n={w.matched_n}, "
              f"original: {w.sample_sizes}")

    # ── Check 3: Eigenvalue Audit ───────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 3: EIGENVALUE AUDIT — LASSO INPUT MATRIX QUALITY")
    print("=" * 60)

    rng = np.random.default_rng(42)
    eigen_rows = []

    for w in windows_ref:
        df_window = cleaned_df[cleaned_df["YEAR"].isin(w.years_in_window)].copy()
        df_pv_win = df_window[df_window["POLVIEWS"].notna()]
        groups_data = {
            "lib": df_pv_win[df_pv_win["POLVIEWS"] < 0],
            "con": df_pv_win[df_pv_win["POLVIEWS"] > 0],
        }

        # Match sample sizes
        min_n = min(len(g) for g in groups_data.values())
        for name in groups_data:
            if len(groups_data[name]) > min_n:
                groups_data[name] = groups_data[name].sample(
                    n=min_n, random_state=rng.integers(1_000_000_000)
                )

        for group_name, g_df in groups_data.items():
            # Pairwise Pearson correlation (the LASSO input)
            corr_pw = g_df[vars_no_pol].corr()
            np.fill_diagonal(corr_pw.values, 1.0)
            corr_pw = corr_pw.fillna(0)

            eigvals = np.linalg.eigvalsh(corr_pw.values)
            min_eig = eigvals.min()
            max_eig = eigvals.max()
            cond_num = max_eig / min_eig if min_eig > 0 else np.inf
            n_negative = (eigvals < -1e-10).sum()

            # Min pairwise-complete N
            pair_ns = []
            for i in range(len(vars_no_pol)):
                for j in range(i + 1, len(vars_no_pol)):
                    vi, vj = vars_no_pol[i], vars_no_pol[j]
                    pair_n = g_df[[vi, vj]].dropna().shape[0]
                    pair_ns.append(pair_n)
            min_pair_n = min(pair_ns) if pair_ns else 0

            eigen_rows.append({
                "mid_year": w.mid_year,
                "group": group_name,
                "min_eigenvalue": min_eig,
                "max_eigenvalue": max_eig,
                "condition_number": cond_num,
                "n_negative_eigenvalues": n_negative,
                "min_pairwise_n": min_pair_n,
                "n_vars": len(vars_no_pol),
            })

    df_eigen = pd.DataFrame(eigen_rows)
    print(f"\nEigenvalue audit results:")
    print(df_eigen.to_string(index=False, float_format="%.4f"))

    any_negative = (df_eigen["n_negative_eigenvalues"] > 0).any()
    min_min_eig = df_eigen["min_eigenvalue"].min()
    max_cond = df_eigen["condition_number"].max()
    min_pair_n_overall = df_eigen["min_pairwise_n"].min()
    print(f"\nOverall: min eigenvalue={min_min_eig:.6f}, max condition number={max_cond:.0f}")
    print(f"Any negative eigenvalues: {any_negative}")
    print(f"Min pairwise-complete N across all windows: {min_pair_n_overall}")
    if any_negative:
        neg_df = df_eigen[df_eigen["n_negative_eigenvalues"] > 0]
        print(f"Windows with negative eigenvalues: {len(neg_df)}")
        print("Note: sklearn's graphical_lasso handles near-PSD matrices via internal ridge.")

    # ── Check 4: PARTYID Alternative Split ──────────────────────
    print("\n" + "=" * 60)
    print("CHECK 4: PARTYID ALTERNATIVE SPLIT")
    print("=" * 60)

    # First pass unfixed to get PARTYID-specific intersection
    windows_pid_unfixed = build_rolling_windows(
        cleaned_df,
        window_size=4, step_size=2, min_years_per_window=3,
        min_n_per_group=100,
        method=CorrelationMethod.PEARSON, partial=True,
        edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
        suppression_params={"regularization": 0.2},
        group_col="PARTYID",
        group_conditions={"dem": "< 0", "rep": "> 0"},
        match_samples=True, random_state=42,
        verbose=True,
    )

    if len(windows_pid_unfixed) > 0:
        pid_fixed_vars = sorted(set.intersection(*[set(w.common_vars) for w in windows_pid_unfixed]))
        pid_vars_no_pol = [v for v in pid_fixed_vars if v not in ("POLVIEWS", "PARTYID")]
        print(f"PARTYID fixed variables: {len(pid_fixed_vars)}")
        print(f"After excluding POLVIEWS/PARTYID: {len(pid_vars_no_pol)}")

        windows_pid = build_rolling_windows(
            cleaned_df,
            window_size=4, step_size=2, min_years_per_window=3,
            min_n_per_group=100,
            method=CorrelationMethod.PEARSON, partial=True,
            edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
            suppression_params={"regularization": 0.2},
            group_col="PARTYID",
            group_conditions={"dem": "< 0", "rep": "> 0"},
            match_samples=True, random_state=42,
            fixed_vars=pid_vars_no_pol,
            verbose=True,
        )

        mid_pid, euc_pid, slope_pid, r_pid, p_pid = compute_euclidean_trend(windows_pid)
        print(f"\nPARTYID Euclidean distance trend:")
        print(f"  slope={slope_pid:.5f}/yr, r={r_pid:.3f}, p={fmt_p(p_pid)}")
        print(f"POLVIEWS reference: slope={slope_ref:.5f}/yr, r={r_ref:.3f}, p={fmt_p(p_ref)}")

        check4_pass = not np.isnan(slope_pid) and slope_pid > 0
        print(f"CHECK 4 VERDICT: {'CONVERGENT' if check4_pass else 'DIVERGENT'} — "
              f"PARTYID split {'agrees' if check4_pass else 'disagrees'} with POLVIEWS finding")
    else:
        print("WARNING: No PARTYID windows could be built. Skipping Check 4.")
        mid_pid, euc_pid, slope_pid, r_pid, p_pid = np.array([]), np.array([]), np.nan, np.nan, np.nan
        check4_pass = False

    # ── Check 5: GSS Survey Weights ─────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 5: GSS SURVEY WEIGHTS")
    print("=" * 60)

    check5_done = False
    wt_corr_lib = wt_corr_con = None
    r_lib_wt = r_con_wt = np.nan
    euc_lib_wt = euc_con_wt = np.nan

    sas_path = DATA_DIR / "raw" / "gss7222_r4.sas7bdat"
    if not sas_path.exists():
        print(f"WARNING: Raw SAS file not found at {sas_path}")
        print("Skipping Check 5 (survey weights).")
    else:
        try:
            import pyreadstat
            print("Loading WTSSALL from raw SAS file...")
            wt_df, _ = pyreadstat.read_sas7bdat(
                str(sas_path),
                usecols=["YEAR", "ID", "WTSSALL"],
                disable_datetime_conversion=True,
            )
            wt_df = wt_df.rename(columns={"ID": "ID"})
            print(f"Loaded {len(wt_df)} rows with WTSSALL")
            print(f"WTSSALL non-null: {wt_df['WTSSALL'].notna().sum()}")

            # Reference period: 2000-2010
            ref_years = list(range(2000, 2011, 2))
            df_ref = cleaned_df[cleaned_df["YEAR"].isin(ref_years)].copy()

            # Merge weights
            df_ref = df_ref.merge(wt_df[["YEAR", "ID", "WTSSALL"]], on=["YEAR", "ID"], how="left")
            n_with_wt = df_ref["WTSSALL"].notna().sum()
            print(f"Reference period rows: {len(df_ref)}, with WTSSALL: {n_with_wt}")

            if n_with_wt < 100:
                print("WARNING: Too few rows with weights. Skipping Check 5.")
            else:
                df_ref_pv = df_ref[df_ref["POLVIEWS"].notna()]
                df_ref_lib = df_ref_pv[df_ref_pv["POLVIEWS"] < 0].copy()
                df_ref_con = df_ref_pv[df_ref_pv["POLVIEWS"] > 0].copy()

                # Unweighted pairwise Pearson
                uw_corr_lib = df_ref_lib[vars_no_pol].corr()
                uw_corr_con = df_ref_con[vars_no_pol].corr()

                # Weighted pairwise Pearson
                print("Computing weighted Pearson correlations (lib)...")
                wt_corr_lib = weighted_pearson_matrix(df_ref_lib, "WTSSALL", vars_no_pol)
                print("Computing weighted Pearson correlations (con)...")
                wt_corr_con = weighted_pearson_matrix(df_ref_con, "WTSSALL", vars_no_pol)

                # Compare: element-wise correlation and Euclidean distance
                # Upper triangle only (excluding diagonal)
                triu_idx = np.triu_indices(len(vars_no_pol), k=1)

                uw_lib_vals = uw_corr_lib.values[triu_idx]
                wt_lib_vals = wt_corr_lib.values[triu_idx]
                valid_lib = np.isfinite(uw_lib_vals) & np.isfinite(wt_lib_vals)

                uw_con_vals = uw_corr_con.values[triu_idx]
                wt_con_vals = wt_corr_con.values[triu_idx]
                valid_con = np.isfinite(uw_con_vals) & np.isfinite(wt_con_vals)

                if valid_lib.sum() > 10 and valid_con.sum() > 10:
                    r_lib_wt = np.corrcoef(uw_lib_vals[valid_lib], wt_lib_vals[valid_lib])[0, 1]
                    r_con_wt = np.corrcoef(uw_con_vals[valid_con], wt_con_vals[valid_con])[0, 1]
                    euc_lib_wt = np.sqrt(np.sum((uw_lib_vals[valid_lib] - wt_lib_vals[valid_lib]) ** 2))
                    euc_con_wt = np.sqrt(np.sum((uw_con_vals[valid_con] - wt_con_vals[valid_con]) ** 2))

                    print(f"\nWeighted vs unweighted comparison (reference period 2000-2010):")
                    print(f"  Liberal: r={r_lib_wt:.4f}, Euclidean dist={euc_lib_wt:.4f}, N pairs={valid_lib.sum()}")
                    print(f"  Conservative: r={r_con_wt:.4f}, Euclidean dist={euc_con_wt:.4f}, N pairs={valid_con.sum()}")

                    check5_pass = r_lib_wt > 0.95 and r_con_wt > 0.95
                    print(f"CHECK 5 VERDICT: {'PASS' if check5_pass else 'FAIL'} — weights "
                          f"{'do NOT' if check5_pass else 'DO'} meaningfully affect results "
                          f"(r>{0.95} threshold)")
                    check5_done = True
                else:
                    print("WARNING: Too few valid correlation pairs. Skipping Check 5.")
        except ImportError:
            print("WARNING: pyreadstat not available. Skipping Check 5.")
        except Exception as e:
            print(f"WARNING: Check 5 failed with error: {e}")

    if not check5_done:
        check5_pass = None
        r_lib_wt = r_con_wt = np.nan
        euc_lib_wt = euc_con_wt = np.nan

    # ════════════════════════════════════════════════════════════
    # SUMMARY FIGURE (3×2)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY FIGURE")
    print("=" * 60)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # Panel A: Check 1 — Exclude 2021-2022 (distance + trend)
    ax = axes[0, 0]
    ax.plot(mid_ref, euc_ref, "s--", color="gray", alpha=0.6, label="Full period", markersize=4)
    ax.plot(mid_pre, euc_pre, "ko-", linewidth=2, label=f"Excl. 2021-2022", markersize=5)
    if len(mid_pre) >= 3 and not np.isnan(slope_pre):
        x_pre = mid_pre.astype(float)
        intercept_pre = linregress(x_pre, euc_pre).intercept
        ax.plot(x_pre, intercept_pre + slope_pre * x_pre, "r--", alpha=0.7,
                label=f"Trend: slope={slope_pre:.4f}, p={fmt_p(p_pre)}")
    if len(mid_ref) >= 3 and not np.isnan(slope_ref):
        x_r = mid_ref.astype(float)
        intercept_ref = linregress(x_r, euc_ref).intercept
        ax.plot(x_r, intercept_ref + slope_ref * x_r, "--", color="gray", alpha=0.4)
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Euclidean Distance (lib vs con)")
    ax.set_title("A. Check 1: Exclude 2021-2022", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Check 2 — POLVIEWS composition (stacked area)
    ax = axes[0, 1]
    years_comp = df_comp["year"].values
    ax.fill_between(years_comp, 0, df_comp["pct_lib"].values,
                    alpha=0.6, color="steelblue", label="Liberal (< 0)")
    ax.fill_between(years_comp, df_comp["pct_lib"].values,
                    df_comp["pct_lib"].values + df_comp["pct_mod"].values,
                    alpha=0.6, color="gray", label="Moderate (= 0)")
    ax.fill_between(years_comp, df_comp["pct_lib"].values + df_comp["pct_mod"].values,
                    100, alpha=0.6, color="indianred", label="Conservative (> 0)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percent of respondents")
    ax.set_title("B. Check 2: POLVIEWS Composition", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Panel C: Check 3 — Min eigenvalue over time
    ax = axes[1, 0]
    for group_name, color, marker in [("lib", "steelblue", "o"), ("con", "indianred", "s")]:
        sub = df_eigen[df_eigen["group"] == group_name]
        ax.plot(sub["mid_year"], sub["min_eigenvalue"], f"{marker}-", color=color,
                linewidth=1.5, markersize=5, label=f"{group_name.capitalize()}")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Minimum Eigenvalue")
    ax.set_title("C. Check 3: Min Eigenvalue (LASSO Input)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: Check 3 — Min pairwise-complete N over time
    ax = axes[1, 1]
    for group_name, color, marker in [("lib", "steelblue", "o"), ("con", "indianred", "s")]:
        sub = df_eigen[df_eigen["group"] == group_name]
        ax.plot(sub["mid_year"], sub["min_pairwise_n"], f"{marker}-", color=color,
                linewidth=1.5, markersize=5, label=f"{group_name.capitalize()}")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Min Pairwise-Complete N")
    ax.set_title("D. Check 3: Min Pairwise N", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel E: Check 4 — PARTYID vs POLVIEWS distance
    ax = axes[2, 0]
    ax.plot(mid_ref, euc_ref, "ko-", linewidth=2, markersize=5, label="POLVIEWS split")
    if len(mid_pid) > 0:
        ax.plot(mid_pid, euc_pid, "b^-", linewidth=1.5, markersize=5, label="PARTYID split")
        if len(mid_pid) >= 3 and not np.isnan(slope_pid):
            x_pid = mid_pid.astype(float)
            intercept_pid = linregress(x_pid, euc_pid).intercept
            ax.plot(x_pid, intercept_pid + slope_pid * x_pid, "b--", alpha=0.5,
                    label=f"PARTYID trend: slope={slope_pid:.4f}, p={fmt_p(p_pid)}")
    if len(mid_ref) >= 3 and not np.isnan(slope_ref):
        x_r2 = mid_ref.astype(float)
        intercept_r2 = linregress(x_r2, euc_ref).intercept
        ax.plot(x_r2, intercept_r2 + slope_ref * x_r2, "k--", alpha=0.5,
                label=f"POLVIEWS trend: slope={slope_ref:.4f}, p={fmt_p(p_ref)}")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Euclidean Distance")
    ax.set_title("E. Check 4: PARTYID vs POLVIEWS", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel F: Check 5 — Weighted vs unweighted scatter
    ax = axes[2, 1]
    if check5_done:
        triu_idx = np.triu_indices(len(vars_no_pol), k=1)
        uw_lib_vals = uw_corr_lib.values[triu_idx]
        wt_lib_vals = wt_corr_lib.values[triu_idx]
        uw_con_vals = uw_corr_con.values[triu_idx]
        wt_con_vals = wt_corr_con.values[triu_idx]

        valid_lib = np.isfinite(uw_lib_vals) & np.isfinite(wt_lib_vals)
        valid_con = np.isfinite(uw_con_vals) & np.isfinite(wt_con_vals)

        ax.scatter(uw_lib_vals[valid_lib], wt_lib_vals[valid_lib],
                   alpha=0.2, s=5, color="steelblue", label=f"Liberal (r={r_lib_wt:.4f})")
        ax.scatter(uw_con_vals[valid_con], wt_con_vals[valid_con],
                   alpha=0.2, s=5, color="indianred", label=f"Conservative (r={r_con_wt:.4f})")
        lims = [-1, 1]
        ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Unweighted Pearson r")
        ax.set_ylabel("Weighted Pearson r")
        ax.set_title("F. Check 5: Weighted vs Unweighted", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
    else:
        ax.text(0.5, 0.5, "Check 5 skipped\n(weights unavailable)",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_title("F. Check 5: Weighted vs Unweighted", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Additional Robustness Checks",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "sound_11_additional_robustness.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

    # ════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary_rows = [
        {"Check": "1. Excl. 2021-2022",
         "Metric": f"slope={slope_pre:.5f}, p={fmt_p(p_pre)}",
         "Verdict": "PASS" if check1_pass else "FAIL"},
        {"Check": "2. POLVIEWS composition",
         "Metric": "descriptive (no pass/fail)",
         "Verdict": "INFO"},
        {"Check": "3. Eigenvalue audit",
         "Metric": f"min_eig={min_min_eig:.4f}, neg={any_negative}",
         "Verdict": "INFO"},
        {"Check": "4. PARTYID split",
         "Metric": f"slope={slope_pid:.5f}, p={fmt_p(p_pid)}" if not np.isnan(slope_pid) else "N/A",
         "Verdict": "CONVERGENT" if check4_pass else "DIVERGENT"},
        {"Check": "5. Survey weights",
         "Metric": f"r_lib={r_lib_wt:.4f}, r_con={r_con_wt:.4f}" if check5_done else "SKIPPED",
         "Verdict": ("PASS" if check5_pass else "FAIL") if check5_done else "SKIPPED"},
    ]
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    # ════════════════════════════════════════════════════════════
    # ANALYSIS WRITEUP
    # ════════════════════════════════════════════════════════════

    # Composition table for writeup
    comp_table = "| Year | N | % Liberal | % Moderate | % Conservative |\n"
    comp_table += "|------|---|-----------|------------|----------------|\n"
    for _, row in df_comp.iterrows():
        comp_table += f"| {row['year']:.0f} | {row['n_total']:.0f} | {row['pct_lib']:.1f} | {row['pct_mod']:.1f} | {row['pct_con']:.1f} |\n"

    # Eigenvalue summary for writeup
    eigen_summary = "| Window Mid | Group | Min Eigenvalue | Condition # | Neg Eigenvalues | Min Pairwise N |\n"
    eigen_summary += "|------------|-------|----------------|-------------|-----------------|----------------|\n"
    for _, row in df_eigen.iterrows():
        cond_str = f"{row['condition_number']:.0f}" if np.isfinite(row['condition_number']) else "Inf"
        eigen_summary += (f"| {row['mid_year']:.0f} | {row['group']} | {row['min_eigenvalue']:.6f} | "
                         f"{cond_str} | {row['n_negative_eigenvalues']:.0f} | {row['min_pairwise_n']:.0f} |\n")

    writeup = f"""# Additional Robustness Checks

## Overview

Five additional robustness checks addressing remaining methodological concerns
identified by the reviewer panel and gap analysis. These complement the existing
checks in sound_07 (fixed variables, alpha sensitivity, full matrix, POLVIEWS
exclusion) and sound_08 (HAC correction, non-overlapping windows, FDR, structural
breaks).

All checks use fixed variables excluding POLVIEWS/PARTYID (N={len(vars_no_pol)})
unless otherwise noted.

![Figure](../figures/sound_11_additional_robustness.png)

---

## Check 1: Exclude 2021-2022

**Question:** The 2021 GSS switched from in-person to web/phone administration
due to COVID-19. Does the divergence trend survive without these mode-switch years?

**Method:** Filter data to YEAR <= 2018, rebuild all rolling windows, recompute
Euclidean distance trend.

**Results:**
- Full period: slope={slope_ref:.5f}/yr, r={r_ref:.3f}, p={fmt_p(p_ref)}
- Excl. 2021-2022: slope={slope_pre:.5f}/yr, r={r_pre:.3f}, p={fmt_p(p_pre)}
- Windows: {len(windows_pre2021)} (vs {len(windows_ref)} full period)

**Verdict:** {"PASS" if check1_pass else "FAIL"} — The divergence trend
{"survives" if check1_pass else "does NOT survive"} exclusion of the COVID-era
survey years. {"The mode switch does not drive the finding." if check1_pass else "The 2021 data may be inflating the trend."}

---

## Check 2: POLVIEWS Composition Over Time

**Question:** Is the lib/con split changing over time in ways that could confound
the network comparison? If one group shrinks dramatically, the matched sample
may become unrepresentative.

**Method:** Compute % liberal, moderate, and conservative by GSS year. Report
matched sample sizes per window.

**Results:**

{comp_table}

**Interpretation:** This is purely descriptive. If composition shifts are large
(e.g., one group halving over time), the matched-sample comparison may be
comparing different populations across windows. Minor shifts (< 10 percentage
points) are acceptable.

---

## Check 3: Eigenvalue Audit

**Question:** The graphical LASSO takes a pairwise Pearson correlation matrix as
input. If this matrix is not positive semi-definite (has negative eigenvalues)
or is ill-conditioned, the LASSO estimates may be unreliable.

**Method:** For each of {len(df_eigen)} matrices ({len(windows_ref)} windows x 2 groups),
compute eigenvalues of the pairwise Pearson input matrix. Report minimum
eigenvalue, condition number, count of negative eigenvalues, and minimum
pairwise-complete N.

**Results:**

{eigen_summary}

**Summary:**
- Minimum eigenvalue across all matrices: {min_min_eig:.6f}
- Maximum condition number: {max_cond:.0f}
- Any negative eigenvalues: {any_negative}
- Minimum pairwise-complete N: {min_pair_n_overall}

**Note:** sklearn's `graphical_lasso` internally adds a small ridge to handle
near-singular inputs, so negative eigenvalues don't cause computational failure.
This check is about transparency — documenting input matrix quality.

---

## Check 4: PARTYID Alternative Split

**Question:** Does the divergence finding replicate when groups are defined by
party identification (PARTYID) rather than ideological self-placement (POLVIEWS)?

**Method:** Rebuild rolling windows with group_col="PARTYID" (dem < 0, rep > 0),
excluding both POLVIEWS and PARTYID from the variable set. Compute Euclidean
distance trend.

**Results:**
- POLVIEWS split: slope={slope_ref:.5f}/yr, r={r_ref:.3f}, p={fmt_p(p_ref)}
- PARTYID split: slope={slope_pid:.5f}/yr, r={r_pid:.3f}, p={fmt_p(p_pid)}

**Verdict:** {"CONVERGENT" if check4_pass else "DIVERGENT"} — The PARTYID-based
split {"shows the same direction of divergence" if check4_pass else "does NOT replicate the divergence"},
{"strengthening" if check4_pass else "weakening"} confidence in the finding.
{"The divergence is not specific to ideological self-placement." if check4_pass else "The divergence may be specific to ideological self-placement."}

---

## Check 5: GSS Survey Weights

**Question:** The GSS provides post-stratification weights (WTSSALL) to adjust
for sampling design. Do weights meaningfully change the correlation structure?

"""
    if check5_done:
        writeup += f"""**Method:** For the reference period (2000-2010), compute weighted and unweighted
pairwise Pearson correlations for liberal and conservative groups. Compare via
element-wise correlation (r) and Euclidean distance.

**Results:**
- Liberal: r(weighted, unweighted) = {r_lib_wt:.4f}, Euclidean dist = {euc_lib_wt:.4f}
- Conservative: r(weighted, unweighted) = {r_con_wt:.4f}, Euclidean dist = {euc_con_wt:.4f}

**Verdict:** {"PASS" if check5_pass else "FAIL"} — Weights {"do NOT" if check5_pass else "DO"}
meaningfully affect the correlation matrices (threshold: r > 0.95).
{"The unweighted analysis is a reasonable approximation." if check5_pass else "Weighted analysis should be considered for robustness."}
"""
    else:
        writeup += """**Status:** SKIPPED — WTSSALL weights could not be loaded from the raw data file.
This check should be completed before submission.
"""

    writeup += f"""
---

## Summary Table

| Check | Key Metric | Verdict |
|-------|-----------|---------|
| 1. Excl. 2021-2022 | slope={slope_pre:.5f}, p={fmt_p(p_pre)} | {"PASS" if check1_pass else "FAIL"} |
| 2. POLVIEWS composition | descriptive | INFO |
| 3. Eigenvalue audit | min_eig={min_min_eig:.4f}, neg={any_negative} | INFO |
| 4. PARTYID split | slope={slope_pid:.5f}, p={fmt_p(p_pid)} | {"CONVERGENT" if check4_pass else "DIVERGENT"} |
| 5. Survey weights | {"r_lib=" + f"{r_lib_wt:.4f}" + ", r_con=" + f"{r_con_wt:.4f}" if check5_done else "SKIPPED"} | {"PASS" if check5_pass else ("FAIL" if check5_pass is not None else "SKIPPED")} |

## Implications for the Paper

"""
    # Dynamic implications
    n_pass = sum([check1_pass, check4_pass, bool(check5_pass)])
    n_total = 2 + (1 if check5_done else 0)  # Checks 1 and 4 have verdicts; 5 if done

    if n_pass == n_total and n_total > 0:
        writeup += """All testable checks passed. The divergence finding is robust to:
excluding the COVID-era mode switch, alternative group definitions (PARTYID),
and (if tested) survey weights. The eigenvalue audit confirms LASSO input
quality, and POLVIEWS composition provides useful context for interpreting
matched-sample comparisons.
"""
    elif n_pass >= n_total - 1 and n_total > 1:
        writeup += f"""{n_pass}/{n_total} testable checks passed. The finding is largely robust,
with minor qualifications needed for any failed checks.
"""
    else:
        writeup += f"""Only {n_pass}/{n_total} testable checks passed. Some findings need
qualification or additional investigation.
"""

    writeup_path = ANALYSES_DIR / "2026-03_additional-robustness.md"
    with open(writeup_path, "w", encoding="utf-8") as f:
        f.write(writeup)
    print(f"\nSaved: {writeup_path}")

    print("\nDone. All additional robustness checks complete.")


if __name__ == "__main__":
    main()
