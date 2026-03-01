"""
Sound 06: Coalition vs Independence — What drives conservative modularity?

Previous analyses found that conservative belief networks have higher modularity
(~0.72) than liberal networks (~0.65). Two competing hypotheses:

1. Independence: Individual conservatives are more heterogeneous — they hold
   belief positions more independently of one another.
2. Coalition: Conservatives form distinct subgroups (religious right, libertarians,
   hawks) that are internally coherent but different from each other.

This script tests these hypotheses using individual-level GSS data.

Usage: python scripts/sound_06_coalition_vs_independence.py
Outputs: figures/sound_06_*.png, stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.loaders.clean_raw_data import clean_datasets

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# Domain labels from sound_04 (community detection results)
DOMAIN_LABELS = {
    "Political": [
        "POLVIEWS", "PARTYID", "PRESLAST_DEMREP", "WOULDVOTELAST_DEMREP",
        "PRESLAST_NONCONFORM", "WOULDVOTELAST_NONCONFORM",
        "EQWLTH", "HELPBLK", "HELPNOT", "HELPPOOR",
        "AFFRMACT", "WRKWAYUP", "RACDIF1", "RACDIF2", "RACDIF3", "RACDIF4",
        "CAPPUN", "COURTS", "GETAHEAD", "DIDVOTELAST", "GUNLAW",
        "NATFARE", "NATFAREY", "NATRACE", "NATRACEY",
    ],
    "Civil liberties": [
        "SPKATH", "SPKCOM", "SPKHOMO", "SPKMIL", "SPKMSLM", "SPKRAC",
        "COLATH", "COLCOM", "COLHOMO", "COLMIL", "COLMSLM", "COLRAC",
        "LIBATH", "LIBCOM", "LIBHOMO", "LIBMIL", "LIBMSLM", "LIBRAC",
    ],
    "Morality/family": [
        "HOMOSEX", "PREMARSX", "TEENSEX", "XMARSEX", "GRASS", "PORNLAW",
        "PRAYER", "SPANKING", "FEFAM", "FEPOL", "FEPRESCH", "FECHLD",
        "DIVLAW", "SEXEDUC", "MARHOMO",
    ],
    "Institutions": [
        "CONARMY", "CONBUS", "CONEDUC", "CONFED", "CONFINAN",
        "CONJUDGE", "CONLABOR", "CONLEGIS", "CONMEDIC", "CONPRESS",
        "CONSCI", "CONTV",
    ],
    "Spending": [
        "NATEDUC", "NATEDUCY", "NATENRGY", "NATENVIR", "NATENVIY",
        "NATHEAL", "NATHEALY", "NATSCI", "NATPARK", "NATMASS",
        "NATROAD", "NATSPAC", "NATSPACY", "NATSOC",
        "NATAID", "NATAIDY", "NATARMS", "NATARMSY",
        "NATCITY", "NATCITYY", "NATCRIME", "NATCRIMY",
        "NATDRUG", "NATDRUGY", "NATCHLD",
    ],
    "Abortion": [
        "ABANY", "ABDEFECT", "ABHLTH", "ABNOMORE", "ABPOOR", "ABRAPE", "ABSINGLE",
        "LETDIE1", "SUICIDE1", "SUICIDE2",
    ],
    "Child-rearing": ["OBEY", "THNKSELF", "WORKHARD", "HELPOTH", "POPULAR"],
    "Police": ["POLABUSE", "POLATTAK", "POLESCAP", "POLHITOK", "POLMURDR"],
    "Religion": [
        "CONCLERG", "POSTLIFE",
        "RELIG_Protestant", "RELIG_Catholic", "RELIG_None",
        "RELIG_Jewish", "RELIG_Other", "RELIG_Buddhism", "RELIG_Hinduism",
        "RELIG_Other_eastern_religions", "RELIG_Muslim",
        "RELIG_Orthodox_christian", "RELIG_Christian",
        "RELIG_Inter_nondenominational",
    ],
    "Social trust": ["FAIR", "HELPFUL", "TRUST"],
}

# Variables excluded from Political domain scores (they define the lib/con split)
POLVIEWS_VARS = {"POLVIEWS", "PARTYID"}


def compute_domain_scores(df, domain_labels, min_items=3):
    """Compute mean domain score per respondent, requiring min_items non-missing.

    For the Political domain, POLVIEWS and PARTYID are excluded (they define
    the lib/con split).

    Returns a DataFrame with one column per domain, indexed like df.
    """
    scores = {}
    for domain, items in domain_labels.items():
        if domain == "Political":
            items = [v for v in items if v not in POLVIEWS_VARS]
        available = [v for v in items if v in df.columns]
        if len(available) < min_items:
            continue
        sub = df[available]
        valid_count = sub.notna().sum(axis=1)
        mean_score = sub.mean(axis=1)
        mean_score[valid_count < min_items] = np.nan
        scores[domain] = mean_score
    return pd.DataFrame(scores, index=df.index)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load and filter data ──────────────────────────────────
    print("Loading data...")
    df = clean_datasets()
    df = df[(df["YEAR"] >= 2000) & (df["YEAR"] <= 2020)].copy()
    print(f"Respondents 2000-2020: {len(df)}")

    # Split lib / con (exclude moderates)
    lib_mask = df["POLVIEWS"] < 0
    con_mask = df["POLVIEWS"] > 0
    df_lib = df[lib_mask].copy()
    df_con = df[con_mask].copy()
    print(f"Liberals: {len(df_lib)}, Conservatives: {len(df_con)}")

    # Compute domain scores
    scores_lib = compute_domain_scores(df_lib, DOMAIN_LABELS)
    scores_con = compute_domain_scores(df_con, DOMAIN_LABELS)
    domains = sorted(scores_lib.columns)
    print(f"\nDomains computed: {len(domains)}")
    for d in domains:
        n_lib = scores_lib[d].notna().sum()
        n_con = scores_con[d].notna().sum()
        print(f"  {d:20s}  lib={n_lib:5d} ({n_lib/len(df_lib)*100:4.1f}%)  "
              f"con={n_con:5d} ({n_con/len(df_con)*100:4.1f}%)")

    # ══════════════════════════════════════════════════════════
    # Section 1: Cross-Domain Correlation Structure
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("SECTION 1: Cross-Domain Correlation Structure")
    print("="*60)

    corr_lib = scores_lib[domains].corr()
    corr_con = scores_con[domains].corr()

    # Mean absolute cross-domain correlation (off-diagonal only)
    n_d = len(domains)
    mask_offdiag = ~np.eye(n_d, dtype=bool)
    mean_abs_lib = np.nanmean(np.abs(corr_lib.values[mask_offdiag]))
    mean_abs_con = np.nanmean(np.abs(corr_con.values[mask_offdiag]))
    print(f"\nMean |cross-domain correlation|:")
    print(f"  Liberal:      {mean_abs_lib:.4f}")
    print(f"  Conservative: {mean_abs_con:.4f}")
    print(f"  Ratio (con/lib): {mean_abs_con/mean_abs_lib:.3f}")

    if mean_abs_con < mean_abs_lib:
        print("  -> Consistent with INDEPENDENCE hypothesis (weaker cross-domain coupling)")
    else:
        print("  -> NOT consistent with independence (con correlations are not weaker)")

    # Side-by-side heatmaps
    short_names = [d[:12] for d in domains]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    vmin, vmax = -0.5, 0.5
    im1 = axes[0].imshow(corr_lib.values, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_xticks(range(n_d)); axes[0].set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(n_d)); axes[0].set_yticklabels(short_names, fontsize=8)
    axes[0].set_title(f"Liberal (mean |r| = {mean_abs_lib:.3f})")
    for i in range(n_d):
        for j in range(n_d):
            v = corr_lib.values[i, j]
            axes[0].text(j, i, f"{v:.2f}" if np.isfinite(v) else "", ha="center", va="center", fontsize=6)

    im2 = axes[1].imshow(corr_con.values, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_xticks(range(n_d)); axes[1].set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(n_d)); axes[1].set_yticklabels(short_names, fontsize=8)
    axes[1].set_title(f"Conservative (mean |r| = {mean_abs_con:.3f})")
    for i in range(n_d):
        for j in range(n_d):
            v = corr_con.values[i, j]
            axes[1].text(j, i, f"{v:.2f}" if np.isfinite(v) else "", ha="center", va="center", fontsize=6)

    fig.colorbar(im2, ax=axes, shrink=0.8, label="Pearson r")
    plt.suptitle("Cross-Domain Correlation Structure: Liberal vs Conservative", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06_domain_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: sound_06_domain_correlations.png")

    # ══════════════════════════════════════════════════════════
    # Section 2: Belief Constraint (Individual-Level)
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("SECTION 2: Belief Constraint (Individual-Level R-squared)")
    print("="*60)

    # PCA on full sample (both lib + con) for a common basis
    scores_all = compute_domain_scores(df, DOMAIN_LABELS)[domains]
    valid_mask_all = scores_all.notna().all(axis=1)
    scores_complete = scores_all[valid_mask_all]
    print(f"Respondents with all {len(domains)} domain scores: {len(scores_complete)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(scores_complete.values)
    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    print(f"PC1 variance explained (full sample): {pca.explained_variance_ratio_[0]:.4f}")

    # Compute R^2 for each person: how much of their profile is captured by PC1
    def compute_r2(scores_df, scaler, pca):
        """For each row, compute R^2 = 1 - SS_res/SS_tot from PC1 projection."""
        valid = scores_df.dropna()
        if len(valid) == 0:
            return pd.Series(dtype=float)
        X = scaler.transform(valid.values)
        proj = pca.transform(X) @ pca.components_  # reconstruct from PC1
        ss_tot = np.sum(X**2, axis=1)
        ss_res = np.sum((X - proj)**2, axis=1)
        r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)
        return pd.Series(r2, index=valid.index)

    r2_lib = compute_r2(scores_lib[domains], scaler, pca)
    r2_con = compute_r2(scores_con[domains], scaler, pca)
    print(f"\nR-squared (PC1 constraint):")
    print(f"  Liberal:      mean={r2_lib.mean():.4f}, median={r2_lib.median():.4f}, N={len(r2_lib)}")
    print(f"  Conservative: mean={r2_con.mean():.4f}, median={r2_con.median():.4f}, N={len(r2_con)}")

    stat, p_val = mannwhitneyu(r2_lib.dropna(), r2_con.dropna(), alternative="two-sided")
    print(f"  Mann-Whitney U = {stat:.0f}, p = {p_val:.2e}")
    if r2_con.mean() < r2_lib.mean():
        print("  -> Conservatives LESS constrained (consistent with INDEPENDENCE)")
    else:
        print("  -> Conservatives MORE constrained (consistent with COALITION)")

    # KDE plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 50)
    ax.hist(r2_lib.dropna(), bins=bins, density=True, alpha=0.4, color="blue", label="Liberal")
    ax.hist(r2_con.dropna(), bins=bins, density=True, alpha=0.4, color="red", label="Conservative")
    # KDE overlay
    from scipy.stats import gaussian_kde
    for data, color, label in [(r2_lib.dropna(), "blue", "Liberal KDE"),
                                (r2_con.dropna(), "red", "Conservative KDE")]:
        kde = gaussian_kde(data, bw_method=0.1)
        x = np.linspace(0, 1, 200)
        ax.plot(x, kde(x), color=color, linewidth=2, label=label)
    ax.axvline(r2_lib.mean(), color="blue", linestyle="--", alpha=0.7, label=f"Lib mean={r2_lib.mean():.3f}")
    ax.axvline(r2_con.mean(), color="red", linestyle="--", alpha=0.7, label=f"Con mean={r2_con.mean():.3f}")
    ax.set_xlabel("R-squared (variance explained by PC1)")
    ax.set_ylabel("Density")
    ax.set_title(f"Belief Constraint: Liberal vs Conservative\nMann-Whitney p = {p_val:.2e}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06_constraint.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: sound_06_constraint.png")

    # ══════════════════════════════════════════════════════════
    # Section 3: Respondent Clustering (GMM)
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("SECTION 3: Respondent Clustering (GMM)")
    print("="*60)

    # Prepare complete-case data for each group
    scores_lib_complete = scores_lib[domains].dropna()
    scores_con_complete = scores_con[domains].dropna()
    print(f"Complete cases — Lib: {len(scores_lib_complete)}, Con: {len(scores_con_complete)}")

    X_lib = StandardScaler().fit_transform(scores_lib_complete.values)
    X_con = StandardScaler().fit_transform(scores_con_complete.values)

    k_range = range(1, 6)
    bic_lib, bic_con = [], []
    aic_lib, aic_con = [], []

    for k in k_range:
        gmm_l = GaussianMixture(n_components=k, random_state=42, n_init=5, covariance_type="full")
        gmm_c = GaussianMixture(n_components=k, random_state=42, n_init=5, covariance_type="full")
        gmm_l.fit(X_lib)
        gmm_c.fit(X_con)
        bic_lib.append(gmm_l.bic(X_lib))
        bic_con.append(gmm_c.bic(X_con))
        aic_lib.append(gmm_l.aic(X_lib))
        aic_con.append(gmm_c.aic(X_con))

    best_k_lib = list(k_range)[np.argmin(bic_lib)]
    best_k_con = list(k_range)[np.argmin(bic_con)]
    print(f"\nBest k (BIC): Liberal = {best_k_lib}, Conservative = {best_k_con}")

    print("\nBIC values:")
    print(f"  {'k':>3s}  {'BIC(lib)':>12s}  {'BIC(con)':>12s}")
    for i, k in enumerate(k_range):
        print(f"  {k:3d}  {bic_lib[i]:12.0f}  {bic_con[i]:12.0f}")

    # BIC curves plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(list(k_range), bic_lib, "bo-", label="Liberal")
    axes[0].plot(list(k_range), bic_con, "ro-", label="Conservative")
    axes[0].set_xlabel("Number of Components (k)")
    axes[0].set_ylabel("BIC")
    axes[0].set_title("BIC (lower is better)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), aic_lib, "bo-", label="Liberal")
    axes[1].plot(list(k_range), aic_con, "ro-", label="Conservative")
    axes[1].set_xlabel("Number of Components (k)")
    axes[1].set_ylabel("AIC")
    axes[1].set_title("AIC (lower is better)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"GMM Model Selection: Best k (BIC) = Lib:{best_k_lib}, Con:{best_k_con}",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06_clustering_bic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: sound_06_clustering_bic.png")

    # Fit best models and extract cluster profiles
    best_k = max(best_k_lib, best_k_con, 2)  # at least 2 for interesting comparison
    print(f"\nFitting GMM with k={best_k} for cluster profile comparison...")

    scaler_lib = StandardScaler().fit(scores_lib_complete.values)
    scaler_con = StandardScaler().fit(scores_con_complete.values)

    gmm_lib_best = GaussianMixture(n_components=best_k, random_state=42, n_init=10, covariance_type="full")
    gmm_con_best = GaussianMixture(n_components=best_k, random_state=42, n_init=10, covariance_type="full")
    gmm_lib_best.fit(scaler_lib.transform(scores_lib_complete.values))
    gmm_con_best.fit(scaler_con.transform(scores_con_complete.values))

    # Centroids in original (unscaled) domain-score space
    centroids_lib = scaler_lib.inverse_transform(gmm_lib_best.means_)
    centroids_con = scaler_con.inverse_transform(gmm_con_best.means_)

    # Cluster sizes
    labels_lib = gmm_lib_best.predict(scaler_lib.transform(scores_lib_complete.values))
    labels_con = gmm_con_best.predict(scaler_con.transform(scores_con_complete.values))
    for i in range(best_k):
        n_l = (labels_lib == i).sum()
        n_c = (labels_con == i).sum()
        print(f"  Cluster {i}: Lib={n_l} ({n_l/len(labels_lib)*100:.1f}%), "
              f"Con={n_c} ({n_c/len(labels_con)*100:.1f}%)")

    # Cluster profile heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 5 + 0.3 * best_k))
    vmin_c = min(centroids_lib.min(), centroids_con.min())
    vmax_c = max(centroids_lib.max(), centroids_con.max())
    abs_max = max(abs(vmin_c), abs(vmax_c))

    im1 = axes[0].imshow(centroids_lib, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
    axes[0].set_xticks(range(len(domains)))
    axes[0].set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(best_k))
    sizes_lib = [f"n={(labels_lib==i).sum()}" for i in range(best_k)]
    axes[0].set_yticklabels([f"Cluster {i} ({sizes_lib[i]})" for i in range(best_k)], fontsize=9)
    axes[0].set_title("Liberal Cluster Centroids")
    for i in range(best_k):
        for j in range(len(domains)):
            axes[0].text(j, i, f"{centroids_lib[i,j]:.2f}", ha="center", va="center", fontsize=6)

    im2 = axes[1].imshow(centroids_con, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
    axes[1].set_xticks(range(len(domains)))
    axes[1].set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(best_k))
    sizes_con = [f"n={(labels_con==i).sum()}" for i in range(best_k)]
    axes[1].set_yticklabels([f"Cluster {i} ({sizes_con[i]})" for i in range(best_k)], fontsize=9)
    axes[1].set_title("Conservative Cluster Centroids")
    for i in range(best_k):
        for j in range(len(domains)):
            axes[1].text(j, i, f"{centroids_con[i,j]:.2f}", ha="center", va="center", fontsize=6)

    fig.colorbar(im2, ax=axes, shrink=0.8, label="Mean Domain Score")
    plt.suptitle(f"Cluster Profiles (k={best_k}): Liberal vs Conservative", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06_cluster_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: sound_06_cluster_profiles.png")

    # ══════════════════════════════════════════════════════════
    # Section 4: Distribution Shape
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("SECTION 4: Distribution Shape")
    print("="*60)

    from scipy.stats import kurtosis, gaussian_kde

    # Grid of KDE plots
    n_domains_plot = len(domains)
    ncols = 4
    nrows = int(np.ceil(n_domains_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten()

    variance_ratios = {}
    kurtosis_data = {}

    for idx, domain in enumerate(domains):
        ax = axes_flat[idx]
        lib_vals = scores_lib[domain].dropna()
        con_vals = scores_con[domain].dropna()

        var_lib = lib_vals.var()
        var_con = con_vals.var()

        if len(lib_vals) < 10 or len(con_vals) < 10 or var_lib == 0 or var_con == 0:
            variance_ratios[domain] = np.nan
            kurtosis_data[domain] = {"lib": np.nan, "con": np.nan}
            reason = "zero variance (ranking items)" if var_lib == 0 or var_con == 0 else "insufficient data"
            ax.set_title(f"{domain}\n({reason})", fontsize=9)
            ax.text(0.5, 0.5, reason, transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            continue

        variance_ratios[domain] = var_con / var_lib

        kurt_lib = kurtosis(lib_vals, fisher=True)
        kurt_con = kurtosis(con_vals, fisher=True)
        kurtosis_data[domain] = {"lib": kurt_lib, "con": kurt_con}

        # KDE
        x_range = np.linspace(
            min(lib_vals.min(), con_vals.min()) - 0.1,
            max(lib_vals.max(), con_vals.max()) + 0.1,
            200
        )
        try:
            kde_l = gaussian_kde(lib_vals, bw_method=0.2)
            kde_c = gaussian_kde(con_vals, bw_method=0.2)
            ax.plot(x_range, kde_l(x_range), color="blue", linewidth=1.5, label="Liberal")
            ax.plot(x_range, kde_c(x_range), color="red", linewidth=1.5, label="Conservative")
            ax.fill_between(x_range, kde_l(x_range), alpha=0.15, color="blue")
            ax.fill_between(x_range, kde_c(x_range), alpha=0.15, color="red")
        except Exception:
            ax.hist(lib_vals, bins=20, density=True, alpha=0.4, color="blue", label="Liberal")
            ax.hist(con_vals, bins=20, density=True, alpha=0.4, color="red", label="Conservative")

        ax.set_title(domain, fontsize=9, fontweight="bold")
        ax.set_xlabel("Domain Score", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n_domains_plot, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("Domain Score Distributions: Liberal vs Conservative", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: sound_06_distributions.png")

    # Variance ratios bar chart
    vr_df = pd.Series(variance_ratios).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["red" if v > 1 else "blue" for v in vr_df.values]
    bars = ax.bar(range(len(vr_df)), vr_df.values, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Equal variance")
    ax.set_xticks(range(len(vr_df)))
    ax.set_xticklabels([d[:14] for d in vr_df.index], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Variance Ratio (Conservative / Liberal)")
    ax.set_title("Domain Score Variance Ratios")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for i, (d, v) in enumerate(vr_df.items()):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06_variance_ratios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: sound_06_variance_ratios.png")

    # Print variance ratios and kurtosis
    print(f"\n{'Domain':>20s}  {'Var(con)/Var(lib)':>17s}  {'Kurt(lib)':>10s}  {'Kurt(con)':>10s}")
    for d in vr_df.index:
        print(f"{d:>20s}  {variance_ratios[d]:17.3f}  "
              f"{kurtosis_data[d]['lib']:10.3f}  {kurtosis_data[d]['con']:10.3f}")

    n_higher_var = sum(1 for v in variance_ratios.values() if v > 1)
    mean_vr = np.nanmean(list(variance_ratios.values()))
    print(f"\nDomains where con has higher variance: {n_higher_var}/{len(variance_ratios)}")
    print(f"Mean variance ratio: {mean_vr:.3f}")

    # Kurtosis comparison
    n_more_platykurtic = sum(1 for d in domains
                             if kurtosis_data[d]["con"] < kurtosis_data[d]["lib"])
    print(f"Domains where con is more platykurtic (flatter): {n_more_platykurtic}/{len(domains)}")

    # ══════════════════════════════════════════════════════════
    # Narrative Summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("NARRATIVE SUMMARY")
    print("="*60)

    print(f"""
Cross-domain correlations:
  Mean |r|: Lib={mean_abs_lib:.3f}, Con={mean_abs_con:.3f}
  {'Weaker conservative coupling -> INDEPENDENCE' if mean_abs_con < mean_abs_lib
   else 'Stronger conservative coupling -> COALITION'}

Belief constraint (R-squared from PC1):
  Mean R2: Lib={r2_lib.mean():.3f}, Con={r2_con.mean():.3f}
  {'Conservatives less constrained -> INDEPENDENCE' if r2_con.mean() < r2_lib.mean()
   else 'Conservatives more constrained -> COALITION'}
  Mann-Whitney p = {p_val:.2e}

Clustering (GMM):
  Best k (BIC): Lib={best_k_lib}, Con={best_k_con}
  {'Conservatives prefer more clusters -> COALITION' if best_k_con > best_k_lib
   else 'Same or fewer clusters -> INDEPENDENCE' if best_k_con <= best_k_lib else ''}

Distribution shape:
  Domains where con has higher variance: {n_higher_var}/{len(variance_ratios)}
  Mean variance ratio (con/lib): {mean_vr:.3f}
  {'Higher conservative variance -> INDEPENDENCE' if mean_vr > 1.05
   else 'Similar variance -> ambiguous' if mean_vr > 0.95
   else 'Lower conservative variance -> unexpected'}
""")

    print("Done. Figures saved to figures/sound_06_*.png")


if __name__ == "__main__":
    main()
