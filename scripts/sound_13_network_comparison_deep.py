"""
Sound 13: Deep Structural Comparison — how liberal and conservative belief
networks differ, and what those differences imply.

Analyses:
  1. Hub comparison: top hubs per group, group-specific hubs
  2. Edge classification: within-domain vs between-domain for differential edges
  3. Community switch mapping: where do switching variables go?
  4. Top differential edges with substantive interpretation context
  5. Bridge variable comparison (betweenness centrality)

Uses 2000-2010 reference period, same parameters as sound_02.

Usage: python scripts/sound_13_network_comparison_deep.py
Outputs: figures/sound_13_*.png, stdout
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

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import (
    calculate_correlation_matrix, CorrelationMethod, EdgeSuppressionMethod,
)
from src.generators.corr_make_conditioned_network import calculate_conditioned_correlation_matrix

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

METHOD = CorrelationMethod.PEARSON
EDGE_SUPP = EdgeSuppressionMethod.REGULARIZATION
SUPP_PARAMS = {"regularization": 0.2}

# Domain assignments from sound_01 community detection
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


def build_var_to_domain(variables):
    """Map each variable to its domain."""
    var_domain = {}
    for domain, vars_list in DOMAIN_LABELS.items():
        for v in vars_list:
            if v in variables:
                var_domain[v] = domain
    # Assign 'Unknown' to any variable not in the mapping
    for v in variables:
        if v not in var_domain:
            var_domain[v] = "Unknown"
    return var_domain


def build_graph(corr_matrix):
    """Build a NetworkX graph from a correlation matrix."""
    mat = corr_matrix.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat.abs())
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
    return G


def get_communities(G):
    """Detect Louvain communities."""
    if G.number_of_edges() == 0:
        return [], {}
    comms = nx.community.louvain_communities(G, weight="weight", seed=42)
    node_comm = {}
    for i, c in enumerate(comms):
        for node in c:
            node_comm[node] = i
    return comms, node_comm


def edge_domain_type(v1, v2, var_domain):
    """Classify edge as within-domain or between-domain."""
    d1 = var_domain.get(v1, "Unknown")
    d2 = var_domain.get(v2, "Unknown")
    if d1 == d2:
        return "within", d1
    else:
        return "between", f"{min(d1,d2)}-{max(d1,d2)}"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df = clean_datasets()
    YEARS = list(range(2000, 2011, 2))
    print(f"Reference period: {YEARS}")

    # ── Build matched networks (same as sound_02) ─────────────
    df_period = cleaned_df[cleaned_df["YEAR"].isin(YEARS)].copy()
    df_polviews = df_period[df_period["POLVIEWS"].notna()].copy()

    df_liberal = df_polviews[df_polviews["POLVIEWS"] < 0]
    df_conservative = df_polviews[df_polviews["POLVIEWS"] > 0]

    N_lib = len(df_liberal)
    N_con = len(df_conservative)
    print(f"Liberal: {N_lib}, Conservative: {N_con}")

    corr_liberal, _ = calculate_conditioned_correlation_matrix(
        cleaned_df, years_of_interest=YEARS,
        method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS,
        variable_to_condition="POLVIEWS", condition="less_than_zero",
        return_df=True, verbose=True,
    )

    df_con_matched = df_conservative.sample(n=N_lib, random_state=42)
    corr_conservative = calculate_correlation_matrix(
        df_con_matched, method=METHOD, partial=True,
        edge_suppression=EDGE_SUPP, suppression_params=SUPP_PARAMS, verbose=True,
    )

    # Align to common variables
    common_vars = sorted(set(corr_liberal.columns) & set(corr_conservative.columns))
    lib_mat = corr_liberal.loc[common_vars, common_vars]
    con_mat = corr_conservative.loc[common_vars, common_vars]
    print(f"\nCommon variables: {len(common_vars)}")

    # Build graphs
    G_lib = build_graph(lib_mat)
    G_con = build_graph(con_mat)
    print(f"Liberal edges: {G_lib.number_of_edges()}, Conservative edges: {G_con.number_of_edges()}")

    # Domain mapping
    var_domain = build_var_to_domain(common_vars)

    # ══════════════════════════════════════════════════════════
    # ANALYSIS 1: Hub Comparison
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS 1: HUB COMPARISON")
    print("=" * 70)

    deg_lib = nx.degree_centrality(G_lib)
    deg_con = nx.degree_centrality(G_con)
    btw_lib = nx.betweenness_centrality(G_lib, weight="weight")
    btw_con = nx.betweenness_centrality(G_con, weight="weight")

    # Top 15 hubs by degree centrality
    top_lib = sorted(deg_lib.keys(), key=lambda v: deg_lib[v], reverse=True)[:15]
    top_con = sorted(deg_con.keys(), key=lambda v: deg_con[v], reverse=True)[:15]

    print("\n=== Top 15 Hubs by Degree Centrality ===")
    print(f"{'Rank':>4s}  {'Liberal Hub':25s} {'Deg':>6s} {'Domain':20s}  {'Conservative Hub':25s} {'Deg':>6s} {'Domain':20s}")
    print("-" * 120)
    for i in range(15):
        vl = top_lib[i]
        vc = top_con[i]
        print(f"{i+1:4d}  {vl:25s} {deg_lib[vl]:6.3f} {var_domain.get(vl,''):20s}  {vc:25s} {deg_con[vc]:6.3f} {var_domain.get(vc,''):20s}")

    # Variables that are hubs for one group but not the other
    top20_lib_set = set(sorted(deg_lib.keys(), key=lambda v: deg_lib[v], reverse=True)[:20])
    top20_con_set = set(sorted(deg_con.keys(), key=lambda v: deg_con[v], reverse=True)[:20])

    lib_only_hubs = top20_lib_set - top20_con_set
    con_only_hubs = top20_con_set - top20_lib_set

    print(f"\n=== Group-Specific Hubs (top-20 for one group, not the other) ===")
    print(f"\nLiberal-only hubs ({len(lib_only_hubs)}):")
    for v in sorted(lib_only_hubs, key=lambda v: deg_lib[v], reverse=True):
        con_rank = sorted(deg_con.keys(), key=lambda x: deg_con[x], reverse=True).index(v) + 1
        print(f"  {v:25s}  lib_deg={deg_lib[v]:.3f} (rank {top_lib.index(v)+1 if v in top_lib else '>15'})  con_deg={deg_con[v]:.3f} (rank {con_rank})")

    print(f"\nConservative-only hubs ({len(con_only_hubs)}):")
    for v in sorted(con_only_hubs, key=lambda v: deg_con[v], reverse=True):
        lib_rank = sorted(deg_lib.keys(), key=lambda x: deg_lib[x], reverse=True).index(v) + 1
        print(f"  {v:25s}  con_deg={deg_con[v]:.3f} (rank {top_con.index(v)+1 if v in top_con else '>15'})  lib_deg={deg_lib[v]:.3f} (rank {lib_rank})")

    # Biggest centrality rank differences
    lib_rank_dict = {v: i+1 for i, v in enumerate(sorted(deg_lib.keys(), key=lambda v: deg_lib[v], reverse=True))}
    con_rank_dict = {v: i+1 for i, v in enumerate(sorted(deg_con.keys(), key=lambda v: deg_con[v], reverse=True))}

    rank_diffs = []
    for v in common_vars:
        lr = lib_rank_dict.get(v, len(common_vars))
        cr = con_rank_dict.get(v, len(common_vars))
        rank_diffs.append({"variable": v, "lib_rank": lr, "con_rank": cr,
                           "rank_diff": lr - cr, "abs_rank_diff": abs(lr - cr),
                           "domain": var_domain.get(v, "Unknown")})
    df_ranks = pd.DataFrame(rank_diffs).sort_values("abs_rank_diff", ascending=False)

    print(f"\n=== Top 20 Largest Centrality Rank Differences ===")
    print(f"{'Variable':25s} {'Lib Rank':>9s} {'Con Rank':>9s} {'Diff':>6s} {'Domain':20s} {'More central for':20s}")
    print("-" * 100)
    for _, row in df_ranks.head(20).iterrows():
        more_central = "Liberal" if row["rank_diff"] > 0 else "Conservative"
        print(f"{row['variable']:25s} {row['lib_rank']:9d} {row['con_rank']:9d} {row['rank_diff']:+6d} {row['domain']:20s} {more_central:20s}")

    # Figure 1: Hub comparison scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    lib_ranks = [lib_rank_dict[v] for v in common_vars]
    con_ranks = [con_rank_dict[v] for v in common_vars]
    ax.scatter(lib_ranks, con_ranks, alpha=0.4, s=20, color="gray")

    # Highlight top rank-difference variables
    for _, row in df_ranks.head(15).iterrows():
        v = row["variable"]
        ax.annotate(v, (lib_rank_dict[v], con_rank_dict[v]),
                    fontsize=7, fontweight="bold",
                    color="blue" if row["rank_diff"] > 0 else "red")
        ax.scatter([lib_rank_dict[v]], [con_rank_dict[v]],
                   color="blue" if row["rank_diff"] > 0 else "red", s=50, zorder=5)

    ax.plot([0, len(common_vars)], [0, len(common_vars)], "k--", alpha=0.3)
    ax.set_xlabel("Liberal Degree Centrality Rank (1 = most central)")
    ax.set_ylabel("Conservative Degree Centrality Rank (1 = most central)")
    ax.set_title("Hub Hierarchy Comparison: Liberal vs Conservative\n(Blue = more central for liberals, Red = more central for conservatives)")
    ax.set_xlim(0, len(common_vars) + 1)
    ax.set_ylim(0, len(common_vars) + 1)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_13_hub_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ══════════════════════════════════════════════════════════
    # ANALYSIS 2: Edge Classification (within vs between domain)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS 2: DIFFERENTIAL EDGES — WITHIN VS BETWEEN DOMAIN")
    print("=" * 70)

    # Get all edges and their weights
    triu_mask = np.triu(np.ones(len(common_vars), dtype=bool), k=1)
    lib_vals = lib_mat.values.copy()
    con_vals = con_mat.values.copy()

    edge_data = []
    for i in range(len(common_vars)):
        for j in range(i + 1, len(common_vars)):
            v1, v2 = common_vars[i], common_vars[j]
            lw = lib_vals[i, j]
            cw = con_vals[i, j]
            diff = lw - cw
            etype, epair = edge_domain_type(v1, v2, var_domain)
            edge_data.append({
                "v1": v1, "v2": v2,
                "lib_weight": lw, "con_weight": cw, "diff": diff, "abs_diff": abs(diff),
                "lib_present": abs(lw) > 0, "con_present": abs(cw) > 0,
                "edge_type": etype, "domain_pair": epair,
            })
    df_edges = pd.DataFrame(edge_data)

    # Edges present in one network but not the other
    lib_only_edges = df_edges[(df_edges["lib_present"]) & (~df_edges["con_present"])].copy()
    con_only_edges = df_edges[(~df_edges["lib_present"]) & (df_edges["con_present"])].copy()
    shared_edges = df_edges[(df_edges["lib_present"]) & (df_edges["con_present"])].copy()

    print(f"\nEdge counts:")
    print(f"  Liberal-only edges:      {len(lib_only_edges)}")
    print(f"  Conservative-only edges: {len(con_only_edges)}")
    print(f"  Shared edges:            {len(shared_edges)}")

    # Classify lib-only and con-only by within/between domain
    def classify_edge_set(df_subset, label):
        within = df_subset[df_subset["edge_type"] == "within"]
        between = df_subset[df_subset["edge_type"] == "between"]
        total = len(df_subset)
        if total == 0:
            return
        print(f"\n  {label}: {len(within)} within-domain ({100*len(within)/total:.1f}%), "
              f"{len(between)} between-domain ({100*len(between)/total:.1f}%)")
        # Break down by domain pair
        if len(between) > 0:
            pair_counts = between["domain_pair"].value_counts().head(10)
            print(f"    Top between-domain pairs:")
            for pair, count in pair_counts.items():
                print(f"      {pair}: {count}")

    classify_edge_set(lib_only_edges, "Liberal-only edges")
    classify_edge_set(con_only_edges, "Conservative-only edges")

    # Top differential edges (largest absolute difference, both present)
    top_diff = df_edges[df_edges["lib_present"] | df_edges["con_present"]].sort_values("abs_diff", ascending=False).head(30)

    print(f"\n=== Top 30 Differential Edges (by |lib_weight - con_weight|) ===")
    print(f"{'V1':25s} {'V2':25s} {'Lib':>7s} {'Con':>7s} {'Diff':>7s} {'Type':8s} {'Domain/Pair':30s} {'Stronger for':15s}")
    print("-" * 130)
    for _, row in top_diff.iterrows():
        stronger = "Liberal" if row["diff"] > 0 else "Conservative"
        print(f"{row['v1']:25s} {row['v2']:25s} {row['lib_weight']:+7.3f} {row['con_weight']:+7.3f} "
              f"{row['diff']:+7.3f} {row['edge_type']:8s} {row['domain_pair']:30s} {stronger:15s}")

    # Aggregate: for edges stronger for libs vs cons, what fraction are within/between?
    lib_stronger = df_edges[(df_edges["diff"] > 0) & (df_edges["lib_present"] | df_edges["con_present"]) & (df_edges["abs_diff"] > 0.01)]
    con_stronger = df_edges[(df_edges["diff"] < 0) & (df_edges["lib_present"] | df_edges["con_present"]) & (df_edges["abs_diff"] > 0.01)]

    print(f"\n=== Within vs Between Domain for Differential Edges (|diff| > 0.01) ===")
    classify_edge_set(lib_stronger, "Lib-stronger edges")
    classify_edge_set(con_stronger, "Con-stronger edges")

    # Figure 2: Within vs between domain bar chart
    categories = ["Lib-only", "Con-only", "Lib-stronger\n(shared)", "Con-stronger\n(shared)"]
    shared_lib_stronger = shared_edges[shared_edges["diff"] > 0.01]
    shared_con_stronger = shared_edges[shared_edges["diff"] < -0.01]

    within_counts = [
        (lib_only_edges["edge_type"] == "within").sum(),
        (con_only_edges["edge_type"] == "within").sum(),
        (shared_lib_stronger["edge_type"] == "within").sum(),
        (shared_con_stronger["edge_type"] == "within").sum(),
    ]
    between_counts = [
        (lib_only_edges["edge_type"] == "between").sum(),
        (con_only_edges["edge_type"] == "between").sum(),
        (shared_lib_stronger["edge_type"] == "between").sum(),
        (shared_con_stronger["edge_type"] == "between").sum(),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(categories))
    width = 0.35
    bars1 = ax.bar([xi - width/2 for xi in x], within_counts, width, label="Within-domain", color="steelblue")
    bars2 = ax.bar([xi + width/2 for xi in x], between_counts, width, label="Between-domain", color="coral")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Number of Edges")
    ax.set_title("Edge Differences by Domain Type:\nWithin-Domain (consolidation) vs Between-Domain (bridging)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    # Add count labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)), ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, str(int(h)), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_13_edge_domain_type.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ══════════════════════════════════════════════════════════
    # ANALYSIS 3: Community Switch Mapping
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS 3: COMMUNITY SWITCH MAPPING")
    print("=" * 70)

    lib_comms, lib_node_comm = get_communities(G_lib)
    con_comms, con_node_comm = get_communities(G_con)

    # Label communities by their dominant domain
    def label_community(comm_members, var_domain):
        """Label a community by its most common domain."""
        domains = [var_domain.get(v, "Unknown") for v in comm_members]
        from collections import Counter
        counts = Counter(domains)
        top_domain, top_count = counts.most_common(1)[0]
        return f"{top_domain} ({top_count}/{len(comm_members)})"

    lib_comm_labels = {}
    print(f"\nLiberal communities ({len(lib_comms)}):")
    for i, c in enumerate(sorted(lib_comms, key=len, reverse=True)):
        label = label_community(c, var_domain)
        lib_comm_labels[i] = label
        # Map from louvain index to sorted index
        for node in c:
            lib_node_comm[node] = i
        print(f"  L{i}: {label} — {sorted(c)}")

    # Redo to ensure consistent labeling
    lib_comms_sorted = sorted(lib_comms, key=len, reverse=True)
    lib_node_comm = {}
    for i, c in enumerate(lib_comms_sorted):
        for node in c:
            lib_node_comm[node] = i

    con_comms_sorted = sorted(con_comms, key=len, reverse=True)
    con_node_comm = {}
    for i, c in enumerate(con_comms_sorted):
        for node in c:
            con_node_comm[node] = i

    print(f"\nConservative communities ({len(con_comms_sorted)}):")
    for i, c in enumerate(con_comms_sorted):
        label = label_community(c, var_domain)
        print(f"  C{i}: {label} — {sorted(c)}")

    # Find variables that switch communities
    # Use domain-based alignment: match communities by dominant domain overlap
    print(f"\n=== Variables That Switch Communities ===")
    print(f"{'Variable':25s} {'Domain':20s} {'Liberal Community':35s} {'Conservative Community':35s}")
    print("-" * 120)

    switchers = []
    for v in common_vars:
        if v not in lib_node_comm or v not in con_node_comm:
            continue
        li = lib_node_comm[v]
        ci = con_node_comm[v]
        lib_members = set(lib_comms_sorted[li])
        con_members = set(con_comms_sorted[ci])
        # Check if the community composition is substantially different
        overlap = len(lib_members & con_members) / max(len(lib_members | con_members), 1)
        if overlap < 0.6:  # communities are substantially different
            lib_label = label_community(lib_comms_sorted[li], var_domain)
            con_label = label_community(con_comms_sorted[ci], var_domain)
            switchers.append({
                "variable": v,
                "domain": var_domain.get(v, "Unknown"),
                "lib_comm": f"L{li}: {lib_label}",
                "con_comm": f"C{ci}: {con_label}",
                "overlap": overlap,
            })

    switchers.sort(key=lambda x: x["overlap"])
    for s in switchers:
        print(f"{s['variable']:25s} {s['domain']:20s} {s['lib_comm']:35s} {s['con_comm']:35s}")

    print(f"\nTotal switchers (overlap < 60%): {len(switchers)}")

    # ══════════════════════════════════════════════════════════
    # ANALYSIS 4: Bridge Variable Comparison
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS 4: BRIDGE VARIABLE COMPARISON (BETWEENNESS CENTRALITY)")
    print("=" * 70)

    top_btw_lib = sorted(btw_lib.keys(), key=lambda v: btw_lib[v], reverse=True)[:15]
    top_btw_con = sorted(btw_con.keys(), key=lambda v: btw_con[v], reverse=True)[:15]

    print(f"\n=== Top 15 Bridge Variables (Betweenness Centrality) ===")
    print(f"{'Rank':>4s}  {'Liberal Bridge':25s} {'Btw':>7s} {'Domain':20s}  {'Conservative Bridge':25s} {'Btw':>7s} {'Domain':20s}")
    print("-" * 120)
    for i in range(15):
        vl = top_btw_lib[i]
        vc = top_btw_con[i]
        print(f"{i+1:4d}  {vl:25s} {btw_lib[vl]:7.4f} {var_domain.get(vl,''):20s}  {vc:25s} {btw_con[vc]:7.4f} {var_domain.get(vc,''):20s}")

    # Betweenness rank differences
    btw_lib_rank = {v: i+1 for i, v in enumerate(sorted(btw_lib.keys(), key=lambda v: btw_lib[v], reverse=True))}
    btw_con_rank = {v: i+1 for i, v in enumerate(sorted(btw_con.keys(), key=lambda v: btw_con[v], reverse=True))}

    btw_rank_diffs = []
    for v in common_vars:
        lr = btw_lib_rank.get(v, len(common_vars))
        cr = btw_con_rank.get(v, len(common_vars))
        btw_rank_diffs.append({"variable": v, "lib_rank": lr, "con_rank": cr,
                                "rank_diff": lr - cr, "abs_rank_diff": abs(lr - cr),
                                "domain": var_domain.get(v, "Unknown")})
    df_btw_ranks = pd.DataFrame(btw_rank_diffs).sort_values("abs_rank_diff", ascending=False)

    print(f"\n=== Top 15 Largest Betweenness Rank Differences ===")
    print(f"{'Variable':25s} {'Lib Rank':>9s} {'Con Rank':>9s} {'Diff':>6s} {'Domain':20s} {'Bridges more for':20s}")
    print("-" * 100)
    for _, row in df_btw_ranks.head(15).iterrows():
        bridges_for = "Liberal" if row["rank_diff"] > 0 else "Conservative"
        print(f"{row['variable']:25s} {row['lib_rank']:9d} {row['con_rank']:9d} {row['rank_diff']:+6d} {row['domain']:20s} {bridges_for:20s}")

    # Figure 3: Betweenness comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Top bridges side by side
    n_show = 12
    ax = axes[0]
    y = range(n_show)
    lib_btw_vals = [btw_lib[v] for v in top_btw_lib[:n_show]]
    con_btw_vals = [btw_con[v] for v in top_btw_con[:n_show]]
    ax.barh([yi - 0.2 for yi in y], lib_btw_vals, 0.35, label="Liberal", color="blue", alpha=0.7)
    ax.barh([yi + 0.2 for yi in y], con_btw_vals, 0.35, label="Conservative", color="red", alpha=0.7)
    ax.set_yticks(list(y))
    lib_labels = [f"{v} [{var_domain.get(v,'')[:8]}]" for v in top_btw_lib[:n_show]]
    con_labels = [f"{v} [{var_domain.get(v,'')[:8]}]" for v in top_btw_con[:n_show]]
    # Use liberal labels on left
    ax.set_yticklabels(lib_labels, fontsize=7)
    ax.set_xlabel("Betweenness Centrality")
    ax.set_title("Top Bridges: Liberal Network")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")

    # Panel B: Conservative bridges
    ax = axes[1]
    ax.barh(list(y), con_btw_vals, 0.7, color="red", alpha=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(con_labels, fontsize=7)
    ax.set_xlabel("Betweenness Centrality")
    ax.set_title("Top Bridges: Conservative Network")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_13_bridge_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ══════════════════════════════════════════════════════════
    # ANALYSIS 5: Domain-Pair Connectivity Matrix
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ANALYSIS 5: DOMAIN-PAIR CONNECTIVITY")
    print("=" * 70)

    domains_present = sorted(set(var_domain.values()) - {"Unknown"})

    # Count edges and mean weight between each domain pair for each group
    def domain_connectivity(corr_mat, var_domain, domains, common_vars):
        """Compute edge count and mean |weight| between each domain pair."""
        n_domains = len(domains)
        edge_counts = np.zeros((n_domains, n_domains))
        weight_sums = np.zeros((n_domains, n_domains))

        domain_idx = {d: i for i, d in enumerate(domains)}
        var_list = list(corr_mat.columns)

        for i in range(len(var_list)):
            for j in range(i + 1, len(var_list)):
                v1, v2 = var_list[i], var_list[j]
                d1 = var_domain.get(v1, "Unknown")
                d2 = var_domain.get(v2, "Unknown")
                if d1 == "Unknown" or d2 == "Unknown":
                    continue
                w = abs(corr_mat.iloc[i, j])
                if w > 0:
                    di, dj = domain_idx[d1], domain_idx[d2]
                    edge_counts[di, dj] += 1
                    edge_counts[dj, di] += 1
                    weight_sums[di, dj] += w
                    weight_sums[dj, di] += w

        # Normalize by number of possible edges in each domain pair
        domain_sizes = {}
        for d in domains:
            domain_sizes[d] = sum(1 for v in common_vars if var_domain.get(v) == d)

        density = np.zeros((n_domains, n_domains))
        for i in range(n_domains):
            for j in range(n_domains):
                ni = domain_sizes[domains[i]]
                nj = domain_sizes[domains[j]]
                if i == j:
                    possible = ni * (ni - 1) / 2
                else:
                    possible = ni * nj
                if possible > 0:
                    density[i, j] = edge_counts[i, j] / possible

        return edge_counts, weight_sums, density

    lib_counts, lib_weights, lib_density = domain_connectivity(lib_mat, var_domain, domains_present, common_vars)
    con_counts, con_weights, con_density = domain_connectivity(con_mat, var_domain, domains_present, common_vars)

    # Difference matrix
    density_diff = lib_density - con_density

    print(f"\n=== Domain-Pair Edge Density (Liberal - Conservative) ===")
    print(f"Positive = denser for liberals, Negative = denser for conservatives\n")
    print(f"{'':20s}", end="")
    for d in domains_present:
        print(f" {d[:8]:>8s}", end="")
    print()
    for i, d1 in enumerate(domains_present):
        print(f"{d1:20s}", end="")
        for j, d2 in enumerate(domains_present):
            val = density_diff[i, j]
            if i <= j:
                print(f" {val:+8.3f}", end="")
            else:
                print(f" {'':8s}", end="")
        print()

    # Biggest domain-pair differences
    pair_diffs = []
    for i in range(len(domains_present)):
        for j in range(i, len(domains_present)):
            pair_diffs.append({
                "domain_pair": f"{domains_present[i]} x {domains_present[j]}" if i != j else f"{domains_present[i]} (within)",
                "lib_density": lib_density[i, j],
                "con_density": con_density[i, j],
                "diff": density_diff[i, j],
                "abs_diff": abs(density_diff[i, j]),
                "type": "within" if i == j else "between",
            })
    df_pair_diffs = pd.DataFrame(pair_diffs).sort_values("abs_diff", ascending=False)

    print(f"\n=== Top 15 Domain-Pair Density Differences ===")
    print(f"{'Domain Pair':40s} {'Lib Dens':>9s} {'Con Dens':>9s} {'Diff':>8s} {'Type':8s} {'Denser for':15s}")
    print("-" * 95)
    for _, row in df_pair_diffs.head(15).iterrows():
        denser = "Liberal" if row["diff"] > 0 else "Conservative"
        print(f"{row['domain_pair']:40s} {row['lib_density']:9.4f} {row['con_density']:9.4f} "
              f"{row['diff']:+8.4f} {row['type']:8s} {denser:15s}")

    # Figure 4: Domain connectivity heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    short_labels = [d[:8] for d in domains_present]

    ax = axes[0]
    im = ax.imshow(lib_density, cmap="Blues", vmin=0, vmax=max(lib_density.max(), con_density.max()))
    ax.set_xticks(range(len(domains_present)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(domains_present)))
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_title("Liberal Edge Density")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    im = ax.imshow(con_density, cmap="Reds", vmin=0, vmax=max(lib_density.max(), con_density.max()))
    ax.set_xticks(range(len(domains_present)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(domains_present)))
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_title("Conservative Edge Density")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    max_abs = np.abs(density_diff).max()
    im = ax.imshow(density_diff, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(range(len(domains_present)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(domains_present)))
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_title("Difference (Liberal - Conservative)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Domain-Pair Edge Density: Liberal vs Conservative Networks", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_13_domain_connectivity.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nNetwork basics:")
    print(f"  Liberal:      {G_lib.number_of_edges()} edges, clustering={nx.average_clustering(G_lib, weight='weight'):.3f}")
    print(f"  Conservative: {G_con.number_of_edges()} edges, clustering={nx.average_clustering(G_con, weight='weight'):.3f}")

    print(f"\nEdge composition:")
    print(f"  Liberal-only edges:      {len(lib_only_edges)} ({(lib_only_edges['edge_type']=='between').mean()*100:.0f}% between-domain)")
    print(f"  Conservative-only edges: {len(con_only_edges)} ({(con_only_edges['edge_type']=='between').mean()*100:.0f}% between-domain)")

    print(f"\nHub hierarchy (Spearman rho of degree centrality ranks): ", end="")
    from scipy.stats import spearmanr
    common_degs_lib = [deg_lib[v] for v in common_vars]
    common_degs_con = [deg_con[v] for v in common_vars]
    rho, p = spearmanr(common_degs_lib, common_degs_con)
    print(f"rho={rho:.3f}, p={p:.4f}")

    print(f"\nCommunity switchers: {len(switchers)} variables in substantially different communities")

    print(f"\nDomain connectivity: top 5 lib-denser pairs:")
    for _, row in df_pair_diffs[df_pair_diffs["diff"] > 0].head(5).iterrows():
        print(f"  {row['domain_pair']:40s} diff={row['diff']:+.4f}")
    print(f"Top 5 con-denser pairs:")
    for _, row in df_pair_diffs[df_pair_diffs["diff"] < 0].sort_values("diff").head(5).iterrows():
        print(f"  {row['domain_pair']:40s} diff={row['diff']:+.4f}")

    print(f"\nDone. Figures saved to figures/sound_13_*.png")


if __name__ == "__main__":
    main()
