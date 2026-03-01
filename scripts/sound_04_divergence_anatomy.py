"""
Sound 04: Divergence Anatomy — what drives the lib/con divergence?

Usage: python scripts/sound_04_divergence_anatomy.py
Outputs: figures/sound_04_*.png, stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import linregress, spearmanr

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import CorrelationMethod, EdgeSuppressionMethod
from src.analyzers.temporal import build_rolling_windows

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# Fixed domain labels from sound_01 community detection
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


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df = clean_datasets()

    # Build rolling windows
    print("Building rolling windows...")
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
    print(f"Collected {len(windows)} windows")
    for w in windows:
        print(f"  {w.start_year}-{w.end_year}: {len(w.common_vars)} vars, N={w.matched_n}")

    # ── 4.1 Sign disagreements ───────────────────────────────
    sign_data = []
    for w in windows:
        lib_mat = w.networks["lib"].values
        con_mat = w.networks["con"].values
        n_vars = len(w.common_vars)
        triu = np.triu_indices(n_vars, k=1)
        lib_upper = lib_mat[triu]
        con_upper = con_mat[triu]
        both_nonzero = (lib_upper != 0) & (con_upper != 0)
        n_both = both_nonzero.sum()
        sign_disagree = both_nonzero & (np.sign(lib_upper) != np.sign(con_upper))
        n_disagree = sign_disagree.sum()
        sign_data.append({
            "mid_year": w.mid_year,
            "n_both_nonzero": int(n_both),
            "n_sign_disagree": int(n_disagree),
            "frac_sign_disagree": n_disagree / n_both if n_both > 0 else 0,
        })

    df_signs = pd.DataFrame(sign_data)
    print("\n=== Sign Disagreements ===")
    print(df_signs.to_string(index=False))
    slope_s, intercept_s, r_s, p_s, _ = linregress(df_signs["mid_year"], df_signs["frac_sign_disagree"])
    print(f"Trend: slope={slope_s:.6f}/yr, r={r_s:.3f}, p={p_s:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_signs["mid_year"], df_signs["n_sign_disagree"], "ko-")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Number of Edge Sign Disagreements")
    axes[0].set_xlabel("Window Midpoint (year)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_signs["mid_year"], df_signs["frac_sign_disagree"], "ko-")
    axes[1].set_ylabel("Fraction")
    axes[1].set_title("Fraction of Edges with Sign Disagreement")
    axes[1].set_xlabel("Window Midpoint (year)")
    axes[1].grid(True, alpha=0.3)
    x = df_signs["mid_year"].values
    axes[1].plot(x, intercept_s + slope_s * x, "r--", alpha=0.7,
                 label=f"slope={slope_s:.5f}/yr, r={r_s:.3f}, p={p_s:.4f}")
    axes[1].legend(fontsize=9)
    plt.suptitle("Edge Sign Disagreements Between Liberal and Conservative Networks", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_04_sign_disagree.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4.2 Dimensionality divergence ────────────────────────
    dim_data = []
    for w in windows:
        for label in ["lib", "con"]:
            mat = w.networks[label].values.copy()
            np.fill_diagonal(mat, 1.0)
            eigenvalues = np.linalg.eigvalsh(mat)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[eigenvalues > 0]
            total_var = eigenvalues.sum()
            pc1_var = eigenvalues[0] / total_var if total_var > 0 else 0
            pr = (eigenvalues.sum()) ** 2 / (eigenvalues ** 2).sum()
            dim_data.append({
                "mid_year": w.mid_year, "group": label,
                "pc1_var": pc1_var, "participation_ratio": pr,
            })

    df_dim = pd.DataFrame(dim_data)
    df_dim_lib = df_dim[df_dim["group"] == "lib"].reset_index(drop=True)
    df_dim_con = df_dim[df_dim["group"] == "con"].reset_index(drop=True)

    print("\n=== Dimensionality ===")
    print(f"{'Year':>6s} {'PC1(lib)':>9s} {'PC1(con)':>9s} {'PR(lib)':>8s} {'PR(con)':>8s}")
    for _, rl in df_dim_lib.iterrows():
        rc = df_dim_con[df_dim_con["mid_year"] == rl["mid_year"]].iloc[0]
        print(f"{rl['mid_year']:6.0f} {rl['pc1_var']:9.4f} {rc['pc1_var']:9.4f} "
              f"{rl['participation_ratio']:8.2f} {rc['participation_ratio']:8.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_dim_lib["mid_year"], df_dim_lib["pc1_var"], "b.-", label="Liberal")
    axes[0].plot(df_dim_con["mid_year"], df_dim_con["pc1_var"], "r.-", label="Conservative")
    axes[0].set_ylabel("Variance Explained")
    axes[0].set_title("PC1 Variance Explained")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df_dim_lib["mid_year"], df_dim_lib["participation_ratio"], "b.-", label="Liberal")
    axes[1].plot(df_dim_con["mid_year"], df_dim_con["participation_ratio"], "r.-", label="Conservative")
    axes[1].set_ylabel("Participation Ratio")
    axes[1].set_title("Effective Dimensionality")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("Dimensionality Divergence", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_04_dimensionality.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== Dimensionality Trends ===")
    for metric in ["pc1_var", "participation_ratio"]:
        for group, df_g in [("lib", df_dim_lib), ("con", df_dim_con)]:
            s, _, r, p, _ = linregress(df_g["mid_year"], df_g[metric])
            print(f"{metric} ({group}): slope={s:.6f}/yr, r={r:.3f}, p={p:.4f}")

    # ── 4.3 Edge-level divergence drivers ────────────────────
    edge_timeseries = {}
    for w in windows:
        vars_w = w.common_vars
        lib_mat = w.networks["lib"]
        con_mat = w.networks["con"]
        for i_idx, v1 in enumerate(vars_w):
            for j_idx in range(i_idx + 1, len(vars_w)):
                v2 = vars_w[j_idx]
                edge = (v1, v2)
                if edge not in edge_timeseries:
                    edge_timeseries[edge] = {}
                diff = abs(lib_mat.iloc[i_idx, j_idx] - con_mat.iloc[i_idx, j_idx])
                edge_timeseries[edge][w.mid_year] = diff

    edge_slopes = []
    for edge, ts in edge_timeseries.items():
        if len(ts) < 5:
            continue
        years = np.array(sorted(ts.keys()))
        diffs = np.array([ts[y] for y in years])
        slope, _, r, p, _ = linregress(years, diffs)
        edge_slopes.append({
            "var1": edge[0], "var2": edge[1],
            "slope": slope, "r": r, "p": p, "mean_diff": diffs.mean(),
        })

    df_edges = pd.DataFrame(edge_slopes).sort_values("slope", ascending=False)
    print(f"\nTotal edges tracked: {len(df_edges)}")
    print("\n=== Top 20 Fastest-DIVERGING Edges ===")
    print(df_edges.head(20)[["var1", "var2", "slope", "r", "p", "mean_diff"]].to_string(index=False))
    print("\n=== Top 20 Fastest-CONVERGING Edges ===")
    print(df_edges.tail(20).sort_values("slope")[["var1", "var2", "slope", "r", "p", "mean_diff"]].to_string(index=False))

    # Top 10 diverging timeseries
    top_10 = df_edges.head(10)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes_flat = axes.flatten()
    for idx, (_, row) in enumerate(top_10.iterrows()):
        edge = (row["var1"], row["var2"])
        ts = edge_timeseries[edge]
        years = np.array(sorted(ts.keys()))
        diffs = np.array([ts[y] for y in years])
        ax = axes_flat[idx]
        ax.plot(years, diffs, "ko-", markersize=3)
        s, i_val, _, _, _ = linregress(years, diffs)
        ax.plot(years, i_val + s * years, "r--", alpha=0.7)
        ax.set_title(f"{row['var1']}\nvs {row['var2']}", fontsize=8)
        ax.set_ylabel("|Lib - Con|", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Top 10 Fastest-Diverging Edges", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_04_top_edges.png", dpi=150, bbox_inches="tight")
    plt.close()

    n_diverging = (df_edges["slope"] > 0).sum()
    n_converging = (df_edges["slope"] < 0).sum()
    n_sig_div = ((df_edges["slope"] > 0) & (df_edges["p"] < 0.05)).sum()
    n_sig_conv = ((df_edges["slope"] < 0) & (df_edges["p"] < 0.05)).sum()
    print(f"\nEdges diverging: {n_diverging}, converging: {n_converging}")
    print(f"Significantly diverging (p<0.05): {n_sig_div}, converging: {n_sig_conv}")

    # ── 4.4 Domain decomposition ─────────────────────────────
    var_to_domain = {}
    for domain, vars_list in DOMAIN_LABELS.items():
        for v in vars_list:
            var_to_domain[v] = domain

    domain_names = sorted(DOMAIN_LABELS.keys())
    n_domains = len(domain_names)
    domain_pair_keys = []
    for i in range(n_domains):
        for j in range(i, n_domains):
            domain_pair_keys.append((domain_names[i], domain_names[j]))

    decomp_data = []
    for w in windows:
        vars_w = w.common_vars
        lib_mat = w.networks["lib"].values
        con_mat = w.networks["con"].values
        n = len(vars_w)
        var_domains = [var_to_domain.get(v, None) for v in vars_w]

        pair_contributions = {k: 0.0 for k in domain_pair_keys}
        total_d2 = 0.0
        unassigned_d2 = 0.0

        for i_idx in range(n):
            for j_idx in range(i_idx + 1, n):
                d2_ij = (lib_mat[i_idx, j_idx] - con_mat[i_idx, j_idx]) ** 2
                total_d2 += d2_ij
                d1 = var_domains[i_idx]
                d2 = var_domains[j_idx]
                if d1 is None or d2 is None:
                    unassigned_d2 += d2_ij
                    continue
                pair = tuple(sorted([d1, d2]))
                if pair in pair_contributions:
                    pair_contributions[pair] += d2_ij

        row = {"mid_year": w.mid_year, "total_d2": total_d2, "unassigned_d2": unassigned_d2}
        for pair, val in pair_contributions.items():
            label = f"{pair[0]} x {pair[1]}" if pair[0] != pair[1] else f"{pair[0]} (within)"
            row[label] = val
        decomp_data.append(row)

    df_decomp = pd.DataFrame(decomp_data)
    contrib_cols = [c for c in df_decomp.columns if c not in ["mid_year", "total_d2"]]
    analysis_cols = [c for c in contrib_cols if c != "unassigned_d2"]
    mean_contribs = df_decomp[analysis_cols].mean().sort_values(ascending=False)
    total_mean = df_decomp["total_d2"].mean()

    print(f"\n=== Average d² Contribution by Domain Pair ===")
    for label, val in mean_contribs.head(20).items():
        print(f"  {label:45s} {val:.4f}  ({val / total_mean * 100:5.1f}%)")

    within_cols = [c for c in analysis_cols if "(within)" in c]
    between_cols = [c for c in analysis_cols if "(within)" not in c]
    within_mean = df_decomp[within_cols].sum(axis=1).mean()
    between_mean = df_decomp[between_cols].sum(axis=1).mean()
    print(f"\nWithin-domain d²:  {within_mean:.4f} ({within_mean / total_mean * 100:.1f}%)")
    print(f"Between-domain d²: {between_mean:.4f} ({between_mean / total_mean * 100:.1f}%)")

    # Stacked area chart
    top_n_pairs = 10
    top_labels = mean_contribs.head(top_n_pairs).index.tolist()
    other_analysis = [c for c in analysis_cols if c not in top_labels]

    plot_df = df_decomp[["mid_year"] + top_labels].copy()
    plot_df["Other domains"] = df_decomp[other_analysis + ["unassigned_d2"]].sum(axis=1)

    fig, ax = plt.subplots(figsize=(14, 7))
    cols_to_stack = top_labels + ["Other domains"]
    ax.stackplot(plot_df["mid_year"], *[plot_df[c] for c in cols_to_stack],
                 labels=cols_to_stack, alpha=0.8)
    ax.plot(df_decomp["mid_year"], df_decomp["total_d2"], "k-", linewidth=2, label="Total d²")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Squared Euclidean Distance (d²)")
    ax.set_title("Domain Decomposition of Lib/Con Divergence")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_04_domain_decomp.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Within vs between
    within_ts = df_decomp[within_cols].sum(axis=1)
    between_ts = df_decomp[between_cols].sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_decomp["mid_year"], within_ts, "b.-", label="Within-domain", linewidth=2)
    ax.plot(df_decomp["mid_year"], between_ts, "r.-", label="Between-domain", linewidth=2)
    ax.plot(df_decomp["mid_year"], df_decomp["total_d2"], "k.--", label="Total", alpha=0.5)
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("d²")
    ax.set_title("Within-Domain vs Between-Domain Divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_04_domain_within_between.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== Domain Trends ===")
    for series, label in [(within_ts, "Within-domain"), (between_ts, "Between-domain"),
                           (df_decomp["total_d2"], "Total")]:
        s, _, r, p, _ = linregress(df_decomp["mid_year"], series)
        print(f"{label} d² trend: slope={s:.5f}/yr, r={r:.3f}, p={p:.4f}")

    within_share = within_ts / df_decomp["total_d2"]
    s_share, _, r_share, p_share, _ = linregress(df_decomp["mid_year"], within_share)
    print(f"Within-domain share trend: slope={s_share:.6f}/yr, r={r_share:.3f}, p={p_share:.4f}")

    # ── 4.5 Modularity divergence ────────────────────────────
    mod_data = []
    for w in windows:
        for label in ["lib", "con"]:
            mat = w.networks[label].copy()
            np.fill_diagonal(mat.values, 0)
            G = nx.from_pandas_adjacency(mat.abs())
            G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
            if G.number_of_edges() == 0:
                continue
            comms = nx.community.louvain_communities(G, weight="weight", seed=42)
            Q = nx.community.modularity(G, comms, weight="weight")
            n_comms = sum(1 for c in comms if len(c) >= 3)
            mod_data.append({
                "mid_year": w.mid_year, "group": label,
                "modularity": Q, "n_communities": n_comms,
            })

    df_mod = pd.DataFrame(mod_data)
    df_mod_lib = df_mod[df_mod["group"] == "lib"].reset_index(drop=True)
    df_mod_con = df_mod[df_mod["group"] == "con"].reset_index(drop=True)

    print("\n=== Modularity Over Time ===")
    print(f"{'Year':>6s} {'Q(lib)':>8s} {'Q(con)':>8s} {'#C(lib)':>8s} {'#C(con)':>8s}")
    for _, rl in df_mod_lib.iterrows():
        rc = df_mod_con[df_mod_con["mid_year"] == rl["mid_year"]].iloc[0]
        print(f"{rl['mid_year']:6.0f} {rl['modularity']:8.4f} {rc['modularity']:8.4f} "
              f"{rl['n_communities']:8.0f} {rc['n_communities']:8.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df_mod_lib["mid_year"], df_mod_lib["modularity"], "b.-", label="Liberal")
    axes[0].plot(df_mod_con["mid_year"], df_mod_con["modularity"], "r.-", label="Conservative")
    axes[0].set_ylabel("Modularity (Q)")
    axes[0].set_title("Network Modularity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df_mod_lib["mid_year"], df_mod_lib["n_communities"], "b.-", label="Liberal")
    axes[1].plot(df_mod_con["mid_year"], df_mod_con["n_communities"], "r.-", label="Conservative")
    axes[1].set_ylabel("Number of Communities (size >= 3)")
    axes[1].set_title("Number of Communities")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("Modularity Divergence", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_04_modularity.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== Modularity Trends ===")
    for metric in ["modularity", "n_communities"]:
        for group, df_g in [("lib", df_mod_lib), ("con", df_mod_con)]:
            s, _, r, p, _ = linregress(df_g["mid_year"], df_g[metric])
            print(f"{metric} ({group}): slope={s:.5f}/yr, r={r:.3f}, p={p:.4f}")

    # ── 4.6 Centrality divergence ────────────────────────────
    cent_rank_corr = []
    cent_changes = {}

    for w in windows:
        lib_mat = w.networks["lib"].copy()
        con_mat = w.networks["con"].copy()
        np.fill_diagonal(lib_mat.values, 0)
        np.fill_diagonal(con_mat.values, 0)

        G_lib = nx.from_pandas_adjacency(lib_mat.abs())
        G_con = nx.from_pandas_adjacency(con_mat.abs())
        G_lib.remove_edges_from([(u, v) for u, v, d in G_lib.edges(data=True) if d["weight"] == 0])
        G_con.remove_edges_from([(u, v) for u, v, d in G_con.edges(data=True) if d["weight"] == 0])

        deg_lib = nx.degree_centrality(G_lib)
        deg_con = nx.degree_centrality(G_con)

        common_nodes = sorted(set(deg_lib.keys()) & set(deg_con.keys()))
        lib_cents = [deg_lib[n] for n in common_nodes]
        con_cents = [deg_con[n] for n in common_nodes]

        rho, p_val = spearmanr(lib_cents, con_cents)
        cent_rank_corr.append({"mid_year": w.mid_year, "spearman_rho": rho, "p_value": p_val})

        lib_ranked = pd.Series(lib_cents, index=common_nodes).rank(ascending=False)
        con_ranked = pd.Series(con_cents, index=common_nodes).rank(ascending=False)
        for v in common_nodes:
            if v not in cent_changes:
                cent_changes[v] = {}
            cent_changes[v][w.mid_year] = lib_ranked[v] - con_ranked[v]

    df_cent = pd.DataFrame(cent_rank_corr)
    print("\n=== Centrality Rank Correlation ===")
    print(df_cent.to_string(index=False))

    s_cent, i_cent, r_cent, p_cent, _ = linregress(df_cent["mid_year"], df_cent["spearman_rho"])
    print(f"Trend: slope={s_cent:.5f}/yr, r={r_cent:.3f}, p={p_cent:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_cent["mid_year"], df_cent["spearman_rho"], "ko-", linewidth=2)
    x = df_cent["mid_year"].values
    ax.plot(x, i_cent + s_cent * x, "r--", alpha=0.7,
            label=f"slope={s_cent:.5f}/yr, r={r_cent:.3f}, p={p_cent:.4f}")
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Centrality Rank Correlation (Liberal vs Conservative)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_04_centrality_rank.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Variables with fastest-growing centrality rank disagreement
    cent_slopes = []
    for v, ts in cent_changes.items():
        if len(ts) < 5:
            continue
        years = np.array(sorted(ts.keys()))
        diffs = np.array([ts[y] for y in years])
        slope_v, _, r_v, p_v, _ = linregress(years, np.abs(diffs))
        cent_slopes.append({
            "variable": v, "slope": slope_v, "r": r_v, "p": p_v,
            "mean_rank_diff": np.mean(diffs), "last_rank_diff": diffs[-1],
        })
    df_cent_slopes = pd.DataFrame(cent_slopes).sort_values("slope", ascending=False)
    print("\n=== Variables with Fastest-Growing Centrality Rank Disagreement ===")
    print(df_cent_slopes.head(15).to_string(index=False))

    print("\n=== Variables with Largest Final Centrality Rank Difference ===")
    df_final_diff = df_cent_slopes.sort_values("last_rank_diff", key=abs, ascending=False)
    print(df_final_diff.head(15)[["variable", "mean_rank_diff", "last_rank_diff"]].to_string(index=False))

    print("\nDone. Figures saved to figures/sound_04_*.png")


if __name__ == "__main__":
    main()
