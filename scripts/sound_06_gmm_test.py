"""
GMM sensitivity test: k=1..10, n_init=20/50/100, on 2004-2008 window.

Reports BIC and best log-likelihood for each (k, n_init) combination
to see whether results stabilize with more random starts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

from src.loaders.clean_raw_data import clean_datasets

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

POLVIEWS_VARS = {"POLVIEWS", "PARTYID"}


def compute_domain_scores(df, min_items=3):
    scores = {}
    for domain, items in DOMAIN_LABELS.items():
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
    import pandas as pd
    return pd.DataFrame(scores, index=df.index)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = clean_datasets()

    # 2004-2008 window
    window_df = df[(df["YEAR"] >= 2004) & (df["YEAR"] <= 2008)].copy()
    print(f"Window 2004-2008: {len(window_df)} respondents")

    lib_df = window_df[window_df["POLVIEWS"] < 0]
    con_df = window_df[window_df["POLVIEWS"] > 0]

    domains = sorted(DOMAIN_LABELS.keys())
    lib_scores = compute_domain_scores(lib_df)[domains].dropna()
    con_scores = compute_domain_scores(con_df)[domains].dropna()

    X_lib = StandardScaler().fit_transform(lib_scores.values)
    X_con = StandardScaler().fit_transform(con_scores.values)

    print(f"Complete cases — Lib: {len(X_lib)}, Con: {len(X_con)}")
    print(f"Domains: {len(domains)}")

    max_k = 10
    n_inits = [5, 10, 20, 50, 100, 200, 500]

    # Store results: {group: {n_init: {k: {"bic":, "ll":}}}}
    all_results = {}

    for group_label, X in [("LIBERAL", X_lib), ("CONSERVATIVE", X_con)]:
        n = len(X)
        all_results[group_label] = {}

        print(f"\n{'=' * 80}")
        print(f"  {group_label} (n={n})")
        print(f"{'=' * 80}")

        # Header
        header = f"  {'k':>2s}"
        for ni in n_inits:
            header += f"  {'BIC(ni='+str(ni)+')':>14s} {'LogLik':>10s}"
        print(header)
        print(f"  {'-' * (2 + len(n_inits) * 26)}")

        for ni in n_inits:
            all_results[group_label][ni] = {}

        for k in range(1, max_k + 1):
            row = f"  {k:2d}"
            for ni in n_inits:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    n_init=ni,
                    max_iter=500,
                    random_state=42,
                )
                gmm.fit(X)
                ll = gmm.score(X) * n
                bic = gmm.bic(X)
                all_results[group_label][ni][k] = {"bic": bic, "ll": ll}
                row += f"  {bic:14.1f} {ll:10.1f}"
            print(row, flush=True)

        # Show BIC-selected k for each n_init
        print(f"\n  BIC-selected k:")
        for ni in n_inits:
            bics = [all_results[group_label][ni][k]["bic"] for k in range(1, max_k + 1)]
            best_k = np.argmin(bics) + 1
            print(f"    n_init={ni:2d}: k={best_k} (BIC={bics[best_k-1]:.1f})")

    # ── Figure 1: BIC curves by n_init ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ks = range(1, max_k + 1)
    styles = {
        5:   ("o:", 0.25),
        10:  ("v:", 0.35),
        20:  ("s--", 0.45),
        50:  ("^-.", 0.6),
        100: ("D-.", 0.75),
        200: ("p-", 0.9),
        500: ("*-", 1.0),
    }

    for ax, (group, color) in zip(axes, [("LIBERAL", "blue"), ("CONSERVATIVE", "red")]):
        for ni in n_inits:
            bics = [all_results[group][ni][k]["bic"] for k in ks]
            marker, alpha = styles[ni]
            ax.plot(ks, bics, marker, color=color, alpha=alpha, markersize=6,
                    label=f"n_init={ni}", linewidth=1.5)

        ax.set_xlabel("Number of components (k)")
        ax.set_ylabel("BIC (lower = better)")
        ax.set_title(f"{group}")
        ax.set_xticks(list(ks))
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("BIC Sensitivity to Random Starts (2004-2008 window)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06_gmm_bic_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 2: Log-likelihood curves by n_init ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (group, color) in zip(axes, [("LIBERAL", "blue"), ("CONSERVATIVE", "red")]):
        for ni in n_inits:
            lls = [all_results[group][ni][k]["ll"] for k in ks]
            marker, alpha = styles[ni]
            ax.plot(ks, lls, marker, color=color, alpha=alpha, markersize=6,
                    label=f"n_init={ni}", linewidth=1.5)

        ax.set_xlabel("Number of components (k)")
        ax.set_ylabel("Log-likelihood (higher = better fit)")
        ax.set_title(f"{group}")
        ax.set_xticks(list(ks))
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Log-Likelihood Sensitivity to Random Starts (2004-2008)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06_gmm_ll_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 3: BIC gap relative to n_init=500 ──────────────
    compare_inits = [5, 10, 20, 50, 100, 200]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bar_width = 0.12

    for ax, (group, color) in zip(axes, [("LIBERAL", "blue"), ("CONSERVATIVE", "red")]):
        for i, ni in enumerate(compare_inits):
            gaps = []
            for k in ks:
                bic_ni = all_results[group][ni][k]["bic"]
                bic_500 = all_results[group][500][k]["bic"]
                gaps.append(bic_ni - bic_500)
            positions = [k + (i - len(compare_inits)/2) * bar_width for k in ks]
            alpha = 0.3 + 0.7 * (i / (len(compare_inits) - 1))
            ax.bar(positions, gaps, width=bar_width, alpha=alpha, color=color,
                   label=f"n_init={ni}", edgecolor="white", linewidth=0.3)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Number of components (k)")
        ax.set_ylabel("BIC(n_init=X) - BIC(n_init=500)")
        ax.set_title(f"{group}")
        ax.set_xticks(list(ks))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("BIC Instability Relative to 500 Starts\n(positive = fewer starts missed a better solution)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06_gmm_bic_gap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFigures saved to figures/sound_06_gmm_*.png")
    print("Done.")


if __name__ == "__main__":
    main()
