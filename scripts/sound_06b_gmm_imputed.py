"""
Sound 06b: GMM clustering on raw belief variables via Probabilistic PCA.

Motivation:
  Previous GMM analyses (sound_06, sound_15) used domain-averaged scores,
  which (a) collapsed ~130 variables into 10 summary scores, losing all
  within-domain structure, and (b) required listwise deletion, dropping
  ~80% of respondents. This script works on the raw belief variables —
  the same space the belief network is defined over.

Key methodological choice — Probabilistic PCA (PPCA):
  GSS uses ballot splits: each respondent answers a random subset of
  questions, giving ~43% missing data. Nobody answers everything.
  Standard PCA requires complete data. PPCA handles this natively via
  an EM algorithm that treats missing values as latent variables,
  estimating the principal subspace using only observed entries.

  The generative model is:
    x_observed = W @ z + mu + epsilon
  where z ~ N(0, I) are latent scores, W is the loading matrix, and
  epsilon ~ N(0, sigma^2 * I) is isotropic noise. EM alternates between:
    E-step: infer z for each person using their observed variables
    M-step: update W, mu, sigma^2 using all persons' inferred z's

  This avoids both (a) imputation artifacts and (b) listwise deletion.

Pipeline:
  1. Load 2004-2008 window, split by POLVIEWS into liberal/conservative
  2. Identify belief variables (>=30 obs per group, <=80% missing)
  3. Drop respondents with <50% of variables observed
  4. Fit PPCA via EM (handles missing data internally)
  5. Extract latent scores for all respondents
  6. GMM on the latent scores
  7. Stability check: vary number of PCs retained

Usage: python scripts/sound_06b_gmm_imputed.py
Outputs: figures/sound_06b_*.png, stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from src.loaders.clean_raw_data import clean_datasets

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# ── Configuration ──────────────────────────────────────────────
WINDOW_START = 2004
WINDOW_END = 2008
MIN_OBS_PER_VAR = 30       # minimum non-missing per group to keep a variable
MAX_MISSING_VAR = 0.80     # drop variables with >80% missing
MIN_COMPLETE_PERSON = 0.50 # drop persons with <50% of variables observed
PCA_VARIANCE_TARGET = 0.80 # keep PCs explaining this much cumulative variance
PPCA_MAX_ITER = 200        # EM iterations for PPCA
PPCA_TOL = 1e-6            # convergence tolerance for PPCA
GMM_MAX_K = 10
GMM_N_INIT = 200           # random starts per GMM fit
GMM_COV_TYPE = "full"

META_COLS = {
    "YEAR", "ID_", "BALLOT", "SAMPLE", "OVERSAMP", "FORMWT",
    "WTSSALL", "VSTRAT", "VPSU", "WTSSPS", "WTSSNRPS",
}


# ══════════════════════════════════════════════════════════════
# PROBABILISTIC PCA WITH MISSING DATA (EM implementation)
# ══════════════════════════════════════════════════════════════

def ppca_em(X, n_components, max_iter=PPCA_MAX_ITER, tol=PPCA_TOL, verbose=True):
    """Probabilistic PCA via EM, handling missing data.

    Implements the EM algorithm from Tipping & Bishop (1999) extended
    for missing data following Ilin & Raiko (2010).

    Parameters:
      X:            (n, d) array with np.nan for missing entries
      n_components: number of latent dimensions (q)
      max_iter:     maximum EM iterations
      tol:          convergence threshold on log-likelihood change
      verbose:      print progress

    Returns:
      Z:        (n, q) latent scores for each observation
      W:        (d, q) loading matrix
      mu:       (d,) estimated mean of each variable
      sigma2:   scalar noise variance
      logliks:  list of log-likelihoods per iteration

    The model is: x_i = W @ z_i + mu + epsilon_i
    where z_i ~ N(0, I), epsilon_i ~ N(0, sigma2 * I)
    Missing entries in x_i are simply excluded from the likelihood.
    """
    n, d = X.shape
    q = n_components

    # Observed masks: True where data exists
    O = ~np.isnan(X)

    # Initialize mu from column means (using observed data only)
    mu = np.nanmean(X, axis=0)

    # Center data (keeping NaN as NaN)
    X_centered = X - mu

    # Initialize W randomly, sigma2 from data variance
    rng = np.random.RandomState(42)
    W = rng.randn(d, q) * 0.1
    sigma2 = 1.0

    logliks = []

    # Group observations by missingness pattern for batch processing.
    # GSS ballot splits produce a small number of distinct patterns,
    # so this turns O(n) inversions into O(#patterns) per E-step.
    pattern_map = {}  # tuple(obs_mask) -> list of observation indices
    for i in range(n):
        key = tuple(O[i])
        if key not in pattern_map:
            pattern_map[key] = []
        pattern_map[key].append(i)
    n_patterns = len(pattern_map)
    if verbose:
        print(f"    {n_patterns} distinct missingness patterns "
              f"(from {n} observations)", flush=True)

    for iteration in range(max_iter):
        # ── E-step ────────────────────────────────────────────
        # For each observation i, compute:
        #   E[z_i | x_i_obs] and Cov[z_i | x_i_obs]
        # Only using the observed dimensions of x_i.
        #
        # Key optimization: observations sharing the same missingness
        # pattern have the same M_inv, so we compute it once per pattern.
        # We also cache ZZt_i and Cov_i for the M-step and sigma2 update.

        Z = np.zeros((n, q))             # E[z_i]
        ZZt_all = np.zeros((n, q, q))    # E[z_i z_i^T] per observation
        Cov_all = np.zeros((n, q, q))    # Cov[z_i | x_i_obs] per observation
        total_loglik = 0.0

        for pattern_key, indices in pattern_map.items():
            obs_mask = np.array(pattern_key)
            d_obs = obs_mask.sum()
            if d_obs == 0:
                continue

            W_obs = W[obs_mask]           # (d_obs, q)

            # Shared posterior precision for this pattern
            M = sigma2 * np.eye(q) + W_obs.T @ W_obs     # (q, q)
            M_inv = np.linalg.solve(M, np.eye(q))         # (q, q)
            Cov_z = sigma2 * M_inv                         # (q, q)

            # For log-likelihood: marginal covariance of observed x
            C_obs = W_obs @ W_obs.T + sigma2 * np.eye(d_obs)
            try:
                L_obs = np.linalg.cholesky(C_obs)
                logdet = 2.0 * np.sum(np.log(np.diag(L_obs)))
                C_inv = np.linalg.solve(C_obs, np.eye(d_obs))
                ll_const = -0.5 * (d_obs * np.log(2 * np.pi) + logdet)
                can_compute_ll = True
            except np.linalg.LinAlgError:
                can_compute_ll = False

            # Precompute M_inv @ W_obs^T for posterior mean
            MiWt = M_inv @ W_obs.T   # (q, d_obs)

            for i in indices:
                x_obs = X_centered[i, obs_mask]    # (d_obs,)
                z_i = MiWt @ x_obs                 # (q,)
                Z[i] = z_i
                ZZt_all[i] = Cov_z + np.outer(z_i, z_i)
                Cov_all[i] = Cov_z

                if can_compute_ll:
                    total_loglik += ll_const - 0.5 * (x_obs @ C_inv @ x_obs)

        logliks.append(total_loglik)

        # Check convergence
        if iteration > 0:
            delta = abs(logliks[-1] - logliks[-2])
            if verbose and iteration % 20 == 0:
                print(f"    iter {iteration}: loglik={total_loglik:.1f}, "
                      f"delta={delta:.4f}", flush=True)
            if delta < tol:
                if verbose:
                    print(f"    Converged at iteration {iteration} "
                          f"(delta={delta:.6f} < {tol})", flush=True)
                break
        elif verbose:
            print(f"    iter {iteration}: loglik={total_loglik:.1f}", flush=True)

        # ── M-step ────────────────────────────────────────────
        # Update W using cached ZZt_all from E-step.
        # For each variable j:
        #   W_new[j] = (sum_i:j_obs x_ij E[z_i]^T) @ (sum_i:j_obs E[z_i z_i^T])^-1

        W_new = np.zeros((d, q))
        for j in range(d):
            obs_j = O[:, j]  # which observations have variable j
            n_obs_j = obs_j.sum()
            if n_obs_j == 0:
                continue
            # Numerator: sum over obs with j present of x_ij * E[z_i]^T
            x_j = X_centered[obs_j, j]   # (n_obs_j,)
            Z_j = Z[obs_j]               # (n_obs_j, q)
            numerator = x_j @ Z_j        # (q,)

            # Denominator: sum of cached E[z_i z_i^T] for obs with j
            denom = ZZt_all[obs_j].sum(axis=0)  # (q, q) — vectorized!

            try:
                W_new[j] = np.linalg.solve(denom, numerator)
            except np.linalg.LinAlgError:
                W_new[j] = W[j]

        # Update sigma2 using cached Cov_all
        sigma2_new = 0.0
        n_total_obs = 0
        for i in range(n):
            obs_mask = O[i]
            d_obs = obs_mask.sum()
            if d_obs == 0:
                continue
            n_total_obs += d_obs
            x_obs = X_centered[i, obs_mask]
            W_obs_new = W_new[obs_mask]
            z_i = Z[i]

            # E[||x_obs - W_obs z_i||^2]
            resid = x_obs - W_obs_new @ z_i
            sigma2_new += np.dot(resid, resid)

            # Trace term: Tr(W_obs_new^T W_obs_new Cov[z_i])
            sigma2_new += np.trace(W_obs_new.T @ W_obs_new @ Cov_all[i])

        sigma2 = max(sigma2_new / n_total_obs, 1e-10)
        W = W_new

        # Update mu from residuals
        mu_new = np.zeros(d)
        for j in range(d):
            obs_j = O[:, j]
            if obs_j.sum() > 0:
                mu_new[j] = np.mean(X[obs_j, j] - W_new[j] @ Z[obs_j].T)
        mu = mu_new
        X_centered = X - mu

    else:
        if verbose:
            print(f"    Did not converge after {max_iter} iterations", flush=True)

    return Z, W, mu, sigma2, logliks


def select_variables(df_lib, df_con, all_cols):
    """Select belief variables meeting observation thresholds."""
    candidates = []
    for c in all_cols:
        if df_lib[c].notna().sum() >= MIN_OBS_PER_VAR and df_con[c].notna().sum() >= MIN_OBS_PER_VAR:
            candidates.append(c)

    selected, dropped = [], []
    for c in candidates:
        lib_miss, con_miss = df_lib[c].isna().mean(), df_con[c].isna().mean()
        if max(lib_miss, con_miss) <= MAX_MISSING_VAR:
            selected.append(c)
        else:
            dropped.append((c, lib_miss, con_miss))

    return selected, dropped


def filter_respondents(data, min_frac=MIN_COMPLETE_PERSON):
    """Drop respondents with too few observed variables."""
    obs_frac = data.notna().mean(axis=1)
    mask = obs_frac >= min_frac
    return data[mask].copy(), (~mask).sum()


def fit_gmm_range(X, max_k=GMM_MAX_K, n_init=GMM_N_INIT):
    """Fit GMMs for k=1..max_k on complete data matrix X."""
    n = len(X)
    results = {}
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type=GMM_COV_TYPE,
            n_init=n_init, max_iter=500, random_state=42,
        )
        gmm.fit(X)
        ll = gmm.score(X) * n
        results[k] = {
            "bic": gmm.bic(X),
            "aic": -2 * ll + 2 * gmm._n_parameters(),
            "loglik": ll,
            "model": gmm,
            "n_params": gmm._n_parameters(),
        }
    return results


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # STEP 1: LOAD AND SPLIT DATA
    # ══════════════════════════════════════════════════════════════
    print("=" * 70, flush=True)
    print("STEP 1: Load data and split by ideology", flush=True)
    print("=" * 70, flush=True)

    df = clean_datasets()
    window_df = df[(df["YEAR"] >= WINDOW_START) & (df["YEAR"] <= WINDOW_END)].copy()
    print(f"Window {WINDOW_START}-{WINDOW_END}: {len(window_df)} total respondents", flush=True)

    lib_df = window_df[window_df["POLVIEWS"] < 0]
    con_df = window_df[window_df["POLVIEWS"] > 0]
    print(f"  Liberals:      {len(lib_df)}", flush=True)
    print(f"  Conservatives: {len(con_df)}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STEP 2: SELECT VARIABLES
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}", flush=True)
    print("STEP 2: Select belief variables", flush=True)
    print("=" * 70, flush=True)

    all_belief_cols = [c for c in window_df.columns if c not in META_COLS]
    print(f"Total columns (excluding metadata): {len(all_belief_cols)}", flush=True)

    variables, dropped = select_variables(lib_df, con_df, all_belief_cols)
    print(f"Variables passing filters: {len(variables)}", flush=True)
    if dropped:
        print(f"Dropped {len(dropped)} variables (>{MAX_MISSING_VAR:.0%} missing):", flush=True)
        for var, lm, cm in dropped:
            print(f"    {var:35s}  lib={lm:.1%}, con={cm:.1%}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STEP 3: FILTER RESPONDENTS
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}", flush=True)
    print("STEP 3: Filter respondents by completeness", flush=True)
    print("=" * 70, flush=True)

    lib_data = lib_df[variables]
    con_data = con_df[variables]
    lib_filtered, lib_dropped = filter_respondents(lib_data)
    con_filtered, con_dropped = filter_respondents(con_data)

    print(f"Minimum completeness: {MIN_COMPLETE_PERSON:.0%}", flush=True)
    print(f"  Liberals:      {len(lib_filtered)} kept, {lib_dropped} dropped "
          f"({lib_dropped/len(lib_data):.1%})", flush=True)
    print(f"  Conservatives: {len(con_filtered)} kept, {con_dropped} dropped "
          f"({con_dropped/len(con_data):.1%})", flush=True)

    for label, data in [("Liberal", lib_filtered), ("Conservative", con_filtered)]:
        miss = data.isna().mean()
        person_miss = data.isna().mean(axis=1)
        print(f"\n  {label} remaining missingness:", flush=True)
        print(f"    Per-variable: median={miss.median():.1%}, max={miss.max():.1%}", flush=True)
        print(f"    Per-person:   median={person_miss.median():.1%}, max={person_miss.max():.1%}", flush=True)
        print(f"    Overall:      {data.isna().sum().sum() / data.size:.1%}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STEP 4: PROBABILISTIC PCA
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}", flush=True)
    print("STEP 4: Probabilistic PCA (EM with missing data)", flush=True)
    print("=" * 70, flush=True)

    # First, determine number of components by fitting with a generous q
    # and examining the implied variance explained.
    # We'll try q = 30 (well above what we expect to need) then trim.

    ppca_results = {}

    for group_label, data in [("Liberal", lib_filtered), ("Conservative", con_filtered)]:
        print(f"\n  {group_label} (n={len(data)}, d={len(variables)}):", flush=True)
        X = data.values.copy()  # (n, d) with NaN

        # Standardize using observed means and stds per variable
        col_means = np.nanmean(X, axis=0)
        col_stds = np.nanstd(X, axis=0)
        col_stds[col_stds == 0] = 1.0
        X_std = (X - col_means) / col_stds

        # Fit PPCA with q=30
        q_initial = 30
        print(f"  Fitting PPCA with q={q_initial}...", flush=True)
        Z, W, mu, sigma2, logliks = ppca_em(X_std, n_components=q_initial)

        # Compute variance explained by each component
        # In PPCA, the columns of W approximate the scaled eigenvectors.
        # Variance explained by component j ~ ||W[:,j]||^2
        component_var = np.sum(W**2, axis=0)  # (q,)
        total_var = np.sum(component_var) + sigma2 * len(variables)
        var_explained = component_var / total_var
        # Sort by variance explained (descending)
        sort_idx = np.argsort(-var_explained)
        var_explained_sorted = var_explained[sort_idx]
        cumvar = np.cumsum(var_explained_sorted)

        # Determine number of components for target variance
        n_pcs = int(np.searchsorted(cumvar, PCA_VARIANCE_TARGET) + 1)
        n_pcs = min(n_pcs, q_initial)

        print(f"  sigma2 (noise variance): {sigma2:.4f}", flush=True)
        print(f"  Components for {PCA_VARIANCE_TARGET:.0%} variance: {n_pcs} "
              f"(actual: {cumvar[n_pcs-1]:.1%})", flush=True)
        print(f"  EM iterations: {len(logliks)}", flush=True)
        print(f"  Variance explained by first 10 components:", flush=True)
        for i in range(min(10, q_initial)):
            print(f"    PC{i+1}: {var_explained_sorted[i]:.1%} "
                  f"(cumulative: {cumvar[i]:.1%})", flush=True)

        # Reorder Z columns by variance explained and trim
        Z_sorted = Z[:, sort_idx][:, :n_pcs]

        ppca_results[group_label] = {
            "Z": Z_sorted, "W": W, "mu": mu, "sigma2": sigma2,
            "var_explained": var_explained_sorted, "cumvar": cumvar,
            "n_pcs": n_pcs, "logliks": logliks,
            "col_means": col_means, "col_stds": col_stds,
        }

    # ══════════════════════════════════════════════════════════════
    # STEP 5: GMM ON PPCA LATENT SCORES
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}", flush=True)
    print(f"STEP 5: GMM on PPCA scores (k=1..{GMM_MAX_K}, n_init={GMM_N_INIT})", flush=True)
    print("=" * 70, flush=True)

    all_gmm_results = {}

    for group_label in ["Liberal", "Conservative"]:
        Z = ppca_results[group_label]["Z"]
        n, d = Z.shape
        print(f"\n  {group_label.upper()} (n={n}, d={d} latent dims)", flush=True)

        results = fit_gmm_range(Z)
        all_gmm_results[group_label] = results

        # Model fit table
        print(f"\n  {'k':>3s}  {'#par':>5s}  {'LogLik':>10s}  {'BIC':>10s}  {'AIC':>10s}  "
              f"{'par/obs':>7s}", flush=True)
        print(f"  {'-' * 50}", flush=True)
        for k in range(1, GMM_MAX_K + 1):
            r = results[k]
            ratio = r["n_params"] / n
            print(f"  {k:3d}  {r['n_params']:5d}  {r['loglik']:10.1f}  {r['bic']:10.1f}  "
                  f"{r['aic']:10.1f}  {ratio:7.3f}", flush=True)

        # BIC deltas
        print(f"\n  BIC deltas (positive = k preferred over k-1):", flush=True)
        for k in range(2, GMM_MAX_K + 1):
            delta = results[k-1]["bic"] - results[k]["bic"]
            interp = ("Very strong" if delta > 10 else "Strong" if delta > 6
                      else "Positive" if delta > 2 else "Weak" if delta > 0 else "FAVORS k-1")
            print(f"    k={k}: dBIC = {delta:+.1f}  ({interp})", flush=True)

        best_k = min(range(1, GMM_MAX_K + 1), key=lambda k: results[k]["bic"])
        print(f"\n  BIC-selected k = {best_k}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # STEP 6: SENSITIVITY TO NUMBER OF PCs
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}", flush=True)
    print("STEP 6: Sensitivity — does BIC-selected k change with # PCs?", flush=True)
    print("=" * 70, flush=True)

    for group_label in ["Liberal", "Conservative"]:
        pr = ppca_results[group_label]
        Z_full = pr["Z"]  # already trimmed to n_pcs
        max_pcs = Z_full.shape[1]

        print(f"\n  {group_label.upper()}:", flush=True)
        print(f"  {'#PCs':>5s}  {'CumVar':>7s}  {'BIC k=1':>10s}  {'BIC k=2':>10s}  "
              f"{'BIC k=3':>10s}  {'BIC k=4':>10s}  {'Best k':>6s}", flush=True)
        print(f"  {'-' * 60}", flush=True)

        for n_test in [5, 10, 15, 20, max_pcs]:
            if n_test > max_pcs:
                continue
            Z_test = Z_full[:, :n_test]
            cumvar_at = pr["cumvar"][n_test - 1]

            bics = []
            for k in range(1, 5):
                gmm = GaussianMixture(
                    n_components=k, covariance_type=GMM_COV_TYPE,
                    n_init=GMM_N_INIT, max_iter=500, random_state=42,
                )
                gmm.fit(Z_test)
                bics.append(gmm.bic(Z_test))

            best = np.argmin(bics) + 1
            print(f"  {n_test:5d}  {cumvar_at:6.1%}  {bics[0]:10.1f}  {bics[1]:10.1f}  "
                  f"{bics[2]:10.1f}  {bics[3]:10.1f}  k={best}", flush=True)

    # ══════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}", flush=True)
    print("Generating figures...", flush=True)
    print("=" * 70, flush=True)

    # ── Figure 1: PPCA convergence + variance explained ────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (group, color) in enumerate([("Liberal", "blue"), ("Conservative", "red")]):
        pr = ppca_results[group]

        # Top: EM convergence
        ax = axes[0, col]
        ax.plot(pr["logliks"], color=color, linewidth=1.5)
        ax.set_xlabel("EM iteration")
        ax.set_ylabel("Log-likelihood")
        ax.set_title(f"{group} — PPCA convergence")
        ax.grid(True, alpha=0.3)

        # Bottom: variance explained (scree)
        ax = axes[1, col]
        n_show = min(30, len(pr["var_explained"]))
        pcs = range(1, n_show + 1)
        ax.bar(pcs, pr["var_explained"][:n_show], color=color, alpha=0.5, label="Individual")
        ax.plot(pcs, pr["cumvar"][:n_show], "o-", color=color, markersize=4, label="Cumulative")
        ax.axhline(PCA_VARIANCE_TARGET, color="gray", linestyle="--", alpha=0.5,
                    label=f"{PCA_VARIANCE_TARGET:.0%} target")
        ax.axvline(pr["n_pcs"], color="gray", linestyle=":", alpha=0.5,
                    label=f"{pr['n_pcs']} PCs")
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance explained")
        ax.set_title(f"{group} — {pr['n_pcs']} PCs for {PCA_VARIANCE_TARGET:.0%}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Probabilistic PCA: Convergence and Variance (2004-2008)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06b_ppca.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 2: BIC curves ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ks = range(1, GMM_MAX_K + 1)

    for ax, (group, color) in zip(axes, [("Liberal", "blue"), ("Conservative", "red")]):
        results = all_gmm_results[group]
        bics = [results[k]["bic"] for k in ks]
        aics = [results[k]["aic"] for k in ks]

        ax.plot(list(ks), bics, "o-", color=color, markersize=8, linewidth=2, label="BIC")
        ax.plot(list(ks), aics, "s--", color=color, markersize=5, alpha=0.5, label="AIC")

        best_k = np.argmin(bics) + 1
        ax.axvline(best_k, color="gray", linestyle=":", alpha=0.5, label=f"BIC min (k={best_k})")

        n_pcs = ppca_results[group]["n_pcs"]
        n_obs = ppca_results[group]["Z"].shape[0]
        ax.set_xlabel("Number of components (k)")
        ax.set_ylabel("Information criterion")
        ax.set_title(f"{group} (n={n_obs}, {n_pcs} PCs) — BIC min at k={best_k}")
        ax.set_xticks(list(ks))
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"GMM on PPCA Latent Scores (n_init={GMM_N_INIT}, 2004-2008)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06b_bic_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 3: BIC deltas ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (group, color) in zip(axes, [("Liberal", "blue"), ("Conservative", "red")]):
        results = all_gmm_results[group]
        deltas = [results[k-1]["bic"] - results[k]["bic"] for k in range(2, GMM_MAX_K + 1)]

        ax.bar(range(2, GMM_MAX_K + 1), deltas, color=color, alpha=0.7, edgecolor="white")
        ax.axhline(10, color="black", linestyle="--", alpha=0.5, label="Strong evidence (10)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("k")
        ax.set_ylabel("BIC(k-1) - BIC(k)")
        ax.set_title(f"{group}")
        ax.set_xticks(range(2, GMM_MAX_K + 1))
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("BIC Improvement Per Added Component", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06b_bic_deltas.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 4: PC1 vs PC2 scatter colored by cluster ──────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (group, base_color) in zip(axes, [("Liberal", "blue"), ("Conservative", "red")]):
        results = all_gmm_results[group]
        Z = ppca_results[group]["Z"]
        best_k = min(range(1, GMM_MAX_K + 1), key=lambda k: results[k]["bic"])
        model = results[best_k]["model"]
        labels = model.predict(Z)

        colors = plt.cm.tab10(np.linspace(0, 1, best_k))
        for c in range(best_k):
            mask = labels == c
            ax.scatter(Z[mask, 0], Z[mask, 1], alpha=0.3, s=8, color=colors[c],
                       label=f"C{c+1} ({model.weights_[c]:.1%})")

        ax.set_xlabel("PPCA Component 1")
        ax.set_ylabel("PPCA Component 2")
        ax.set_title(f"{group} k={best_k}")
        ax.legend(fontsize=8, markerscale=3)
        ax.grid(True, alpha=0.3)

    plt.suptitle("GMM Clusters in PPCA Space (2004-2008)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06b_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFigures saved to figures/sound_06b_*.png", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
