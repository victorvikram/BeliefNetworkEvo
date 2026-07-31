"""
GMM n_init=1000 on 2004-2008 window. Simple sequential approach.
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

from src.loaders.clean_raw_data import clean_datasets

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
N_STARTS = 1000
MAX_K = 10

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
    import pandas as pd
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
    return pd.DataFrame(scores, index=df.index)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    df = clean_datasets()

    window_df = df[(df["YEAR"] >= 2004) & (df["YEAR"] <= 2008)].copy()
    print(f"Window 2004-2008: {len(window_df)} respondents", flush=True)

    lib_df = window_df[window_df["POLVIEWS"] < 0]
    con_df = window_df[window_df["POLVIEWS"] > 0]

    domains = sorted(DOMAIN_LABELS.keys())
    lib_scores = compute_domain_scores(lib_df)[domains].dropna()
    con_scores = compute_domain_scores(con_df)[domains].dropna()

    X_lib = StandardScaler().fit_transform(lib_scores.values)
    X_con = StandardScaler().fit_transform(con_scores.values)

    print(f"Complete cases — Lib: {len(X_lib)}, Con: {len(X_con)}", flush=True)
    print(f"n_init={N_STARTS}, k=1..{MAX_K}", flush=True)

    all_results = {}

    for group_label, X in [("LIBERAL", X_lib), ("CONSERVATIVE", X_con)]:
        n = len(X)
        print(f"\n{'=' * 80}", flush=True)
        print(f"  {group_label} (n={n})", flush=True)
        print(f"{'=' * 80}", flush=True)

        results = {}
        for k in range(1, MAX_K + 1):
            print(f"  k={k}...", end=" ", flush=True)
            gmm = GaussianMixture(
                n_components=k, covariance_type="full",
                n_init=N_STARTS, max_iter=500, random_state=42,
            )
            gmm.fit(X)
            ll = gmm.score(X) * n
            bic = gmm.bic(X)
            aic = -2 * ll + 2 * gmm._n_parameters()
            results[k] = {"bic": bic, "aic": aic, "loglik": ll, "model": gmm}
            print(f"BIC={bic:.1f}, LL={ll:.1f}", flush=True)

        all_results[group_label] = results

        # Table
        print(f"\n  {'k':>2s}  {'LogLik':>10s}  {'BIC':>10s}  {'AIC':>10s}", flush=True)
        print(f"  {'-' * 38}", flush=True)
        for k in range(1, MAX_K + 1):
            r = results[k]
            print(f"  {k:2d}  {r['loglik']:10.1f}  {r['bic']:10.1f}  {r['aic']:10.1f}", flush=True)

        # BIC deltas
        print(f"\n  BIC deltas:", flush=True)
        for k in range(2, MAX_K + 1):
            delta = results[k-1]["bic"] - results[k]["bic"]
            print(f"    k={k}: dBIC = {delta:+.1f}", flush=True)

        best_k = min(range(1, MAX_K + 1), key=lambda k: results[k]["bic"])
        print(f"\n  BIC-selected k = {best_k}", flush=True)

    # ── Figure ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ks = range(1, MAX_K + 1)

    for ax, (group, color) in zip(axes, [("LIBERAL", "blue"), ("CONSERVATIVE", "red")]):
        results = all_results[group]
        bics = [results[k]["bic"] for k in ks]
        aics = [results[k]["aic"] for k in ks]

        ax.plot(list(ks), bics, "o-", color=color, markersize=8, linewidth=2, label="BIC")
        ax.plot(list(ks), aics, "s--", color=color, markersize=5, linewidth=1.5, alpha=0.5, label="AIC")
        best_k = np.argmin(bics) + 1
        ax.axvline(best_k, color="gray", linestyle=":", alpha=0.5, label=f"BIC min (k={best_k})")
        ax.set_xlabel("Number of components (k)")
        ax.set_ylabel("Information criterion")
        ax.set_title(f"{group} — BIC min at k={best_k}")
        ax.set_xticks(list(ks))
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"GMM Model Selection with {N_STARTS} Random Starts (2004-2008)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_06_gmm_1000.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFigure saved: figures/sound_06_gmm_1000.png", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
