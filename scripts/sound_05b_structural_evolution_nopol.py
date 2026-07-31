"""
Sound 05b: Structural Evolution (POLVIEWS/PARTYID excluded) — community stability and hub migration.

Same analysis as sound_05, but with POLVIEWS and PARTYID removed from the
fixed variable set. This is the primary analysis for the revised paper;
sound_05 results become supplementary.

Usage: python scripts/sound_05b_structural_evolution_nopol.py
Outputs: figures/sound_05b_*.png, stdout
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
from sklearn.metrics import normalized_mutual_info_score

from src.loaders.clean_raw_data import clean_datasets
from src.generators.corr_make_network import CorrelationMethod, EdgeSuppressionMethod
from src.analyzers.temporal import build_rolling_windows

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def build_graph(corr_matrix):
    """Build a NetworkX graph from a correlation matrix."""
    mat = corr_matrix.copy()
    np.fill_diagonal(mat.values, 0)
    G = nx.from_pandas_adjacency(mat.abs())
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 0])
    return G


def detect_communities(G):
    """Detect Louvain communities and return (communities_list, node_to_label_dict)."""
    if G.number_of_edges() == 0:
        return [], {}
    comms = nx.community.louvain_communities(G, weight="weight", seed=42)
    node_label = {}
    for i, c in enumerate(comms):
        for node in c:
            node_label[node] = i
    return comms, node_label


def compute_nmi(labels_a, labels_b, common_nodes):
    """Compute NMI between two community label dicts over common nodes."""
    a = [labels_a.get(n, -1) for n in common_nodes]
    b = [labels_b.get(n, -1) for n in common_nodes]
    return normalized_mutual_info_score(a, b)


def align_labels_greedy(prev_labels, curr_labels, common_nodes):
    """Align community labels between consecutive windows via greedy max-overlap matching."""
    prev_comms = {}
    curr_comms = {}
    for n in common_nodes:
        pl = prev_labels.get(n)
        cl = curr_labels.get(n)
        if pl is not None:
            prev_comms.setdefault(pl, set()).add(n)
        if cl is not None:
            curr_comms.setdefault(cl, set()).add(n)

    # Greedy mapping: for each curr community, find best-matching prev community
    mapping = {}
    used_prev = set()
    # Sort by size descending for stable matching
    for cl in sorted(curr_comms.keys(), key=lambda k: len(curr_comms[k]), reverse=True):
        best_pl = None
        best_overlap = 0
        for pl in prev_comms.keys():
            if pl in used_prev:
                continue
            overlap = len(curr_comms[cl] & prev_comms[pl])
            if overlap > best_overlap:
                best_overlap = overlap
                best_pl = pl
        if best_pl is not None:
            mapping[cl] = best_pl
            used_prev.add(best_pl)
        else:
            mapping[cl] = cl + 1000  # unmapped

    # Remap curr labels
    remapped = {}
    for n in common_nodes:
        cl = curr_labels.get(n)
        if cl is not None and cl in mapping:
            remapped[n] = mapping[cl]
        elif cl is not None:
            remapped[n] = cl
    return remapped


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_df = clean_datasets()

    # ── Build windows ────────────────────────────────────────
    print("Building total network windows...")
    total_windows = build_rolling_windows(
        cleaned_df,
        window_size=4, step_size=2, min_years_per_window=3,
        min_n_per_group=100,
        method=CorrelationMethod.PEARSON, partial=True,
        edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
        suppression_params={"regularization": 0.2},
        group_col=None,
        verbose=True,
    )
    print(f"Total windows: {len(total_windows)}")

    print("\nBuilding lib/con windows...")
    group_windows = build_rolling_windows(
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
    print(f"Group windows: {len(group_windows)}")

    # Compute fixed_vars = intersection of common_vars across all windows
    all_var_sets = [set(w.common_vars) for w in total_windows] + [set(w.common_vars) for w in group_windows]
    fixed_vars = sorted(set.intersection(*all_var_sets))
    print(f"\nFixed variables (intersection across all windows): {len(fixed_vars)}")

    # ── Exclude POLVIEWS and PARTYID ─────────────────────────
    vars_no_pol = [v for v in fixed_vars if v not in ("POLVIEWS", "PARTYID")]
    print(f"After excluding POLVIEWS/PARTYID: {len(vars_no_pol)} variables")

    # Rebuild with vars_no_pol for cross-window comparability
    print("\nRebuilding with fixed variables (POLVIEWS/PARTYID excluded)...")
    total_windows = build_rolling_windows(
        cleaned_df,
        window_size=4, step_size=2, min_years_per_window=3,
        min_n_per_group=100,
        method=CorrelationMethod.PEARSON, partial=True,
        edge_suppression=EdgeSuppressionMethod.REGULARIZATION,
        suppression_params={"regularization": 0.2},
        group_col=None,
        fixed_vars=vars_no_pol,
        verbose=True,
    )

    group_windows = build_rolling_windows(
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

    print(f"Total windows (fixed, no POLVIEWS/PARTYID): {len(total_windows)}")
    print(f"Group windows (fixed, no POLVIEWS/PARTYID): {len(group_windows)}")

    # ══════════════════════════════════════════════════════════
    # Section A: Community Stability
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECTION A: COMMUNITY STABILITY (POLVIEWS/PARTYID EXCLUDED)")
    print("=" * 60)

    # Detect communities at each window for total, lib, con
    def get_all_community_labels(windows_list, network_key):
        """Returns list of (mid_year, node_label_dict) for a given network key."""
        results = []
        for w in windows_list:
            G = build_graph(w.networks[network_key])
            _, node_labels = detect_communities(G)
            results.append((w.mid_year, node_labels, w.common_vars))
        return results

    total_comm_data = get_all_community_labels(total_windows, "total")
    lib_comm_data = get_all_community_labels(group_windows, "lib")
    con_comm_data = get_all_community_labels(group_windows, "con")

    # NMI between consecutive windows
    def compute_consecutive_nmi(comm_data):
        nmis = []
        for i in range(1, len(comm_data)):
            mid_year = comm_data[i][0]
            common = sorted(set(comm_data[i - 1][2]) & set(comm_data[i][2]))
            nmi = compute_nmi(comm_data[i - 1][1], comm_data[i][1], common)
            nmis.append({"mid_year": mid_year, "nmi": nmi})
        return pd.DataFrame(nmis)

    nmi_total = compute_consecutive_nmi(total_comm_data)
    nmi_lib = compute_consecutive_nmi(lib_comm_data)
    nmi_con = compute_consecutive_nmi(con_comm_data)

    print("\n=== NMI Between Consecutive Windows (no POLVIEWS/PARTYID) ===")
    print(f"{'Year':>6s} {'Total':>8s} {'Lib':>8s} {'Con':>8s}")
    for i in range(len(nmi_total)):
        t_row = nmi_total.iloc[i]
        # Find matching lib/con by closest year
        l_row = nmi_lib.iloc[min(i, len(nmi_lib) - 1)]
        c_row = nmi_con.iloc[min(i, len(nmi_con) - 1)]
        print(f"{t_row['mid_year']:6.0f} {t_row['nmi']:8.3f} {l_row['nmi']:8.3f} {c_row['nmi']:8.3f}")

    # Figure 1: NMI trajectory
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(nmi_total["mid_year"], nmi_total["nmi"], "ko-", label="Total", linewidth=2)
    ax.plot(nmi_lib["mid_year"], nmi_lib["nmi"], "b.-", label="Liberal", linewidth=1.5)
    ax.plot(nmi_con["mid_year"], nmi_con["nmi"], "r.-", label="Conservative", linewidth=1.5)
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("NMI (consecutive windows)")
    ax.set_title("Community Stability: NMI Between Consecutive Windows\n(POLVIEWS/PARTYID excluded)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_05b_nmi_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Community switchers: align labels and count switches
    def count_switches(comm_data):
        """Count how many times each variable switches communities."""
        switch_counts = {}
        for i in range(1, len(comm_data)):
            common = sorted(set(comm_data[i - 1][2]) & set(comm_data[i][2]))
            prev_labels = comm_data[i - 1][1]
            curr_labels = comm_data[i][1]
            aligned = align_labels_greedy(prev_labels, curr_labels, common)
            for n in common:
                if n not in switch_counts:
                    switch_counts[n] = 0
                pl = prev_labels.get(n)
                cl = aligned.get(n)
                if pl is not None and cl is not None and pl != cl:
                    switch_counts[n] += 1
        return pd.Series(switch_counts).sort_values(ascending=False)

    switches_total = count_switches(total_comm_data)
    switches_lib = count_switches(lib_comm_data)
    switches_con = count_switches(con_comm_data)

    # Figure 2: Top-20 community switchers
    top_switchers = switches_total.head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(top_switchers))
    width = 0.3
    ax.bar([xi - width for xi in x], [switches_total.get(v, 0) for v in top_switchers.index],
           width, label="Total", color="gray")
    ax.bar(list(x), [switches_lib.get(v, 0) for v in top_switchers.index],
           width, label="Liberal", color="blue", alpha=0.7)
    ax.bar([xi + width for xi in x], [switches_con.get(v, 0) for v in top_switchers.index],
           width, label="Conservative", color="red", alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(top_switchers.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Community Switches")
    ax.set_title("Top-20 Most Community-Switching Variables\n(POLVIEWS/PARTYID excluded)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_05b_community_switchers.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n=== Top 20 Community Switchers (total network, no POLVIEWS/PARTYID) ===")
    print(top_switchers.to_string())

    # Stable cores: variables in same community >= 80% of windows
    def find_stable_cores(comm_data, threshold=0.8):
        """Find variables that stay in the same community >= threshold fraction of windows."""
        all_vars = set()
        for _, labels, vars_w in comm_data:
            all_vars.update(vars_w)

        # For each variable, track its community label across windows (after alignment)
        var_labels = {v: [] for v in all_vars}
        prev_labels = None
        for i, (_, labels, vars_w) in enumerate(comm_data):
            if i > 0 and prev_labels is not None:
                common = sorted(set(comm_data[i - 1][2]) & set(vars_w))
                aligned = align_labels_greedy(prev_labels, labels, common)
                current = aligned
            else:
                current = labels

            for v in vars_w:
                var_labels[v].append(current.get(v, -1))
            prev_labels = current

        # Find variables where the most common label appears >= threshold
        stable = {}
        for v, lbls in var_labels.items():
            if len(lbls) < 3:
                continue
            from collections import Counter
            counts = Counter(lbls)
            most_common_label, most_common_count = counts.most_common(1)[0]
            frac = most_common_count / len(lbls)
            if frac >= threshold:
                stable[v] = {"label": most_common_label, "frac": frac, "n_windows": len(lbls)}
        return stable

    stable_cores = find_stable_cores(total_comm_data, threshold=0.8)
    print(f"\n=== Stable Community Cores (>= 80% consistency, no POLVIEWS/PARTYID) ===")
    print(f"Variables in stable cores: {len(stable_cores)} / {len(vars_no_pol)}")

    # Group by community label
    core_by_comm = {}
    for v, info in stable_cores.items():
        lbl = info["label"]
        core_by_comm.setdefault(lbl, []).append((v, info["frac"]))
    for lbl in sorted(core_by_comm.keys()):
        members = sorted(core_by_comm[lbl], key=lambda x: -x[1])
        var_names = [m[0] for m in members]
        print(f"  Core {lbl} ({len(members)} members): {var_names}")

    # Figure 3: Stable cores heatmap — variable x window community membership
    # Show only vars_no_pol, color by community label
    n_windows_total = len(total_comm_data)
    heatmap_vars = sorted(vars_no_pol)
    heatmap_data = np.full((len(heatmap_vars), n_windows_total), np.nan)
    mid_years = []

    prev_labels = None
    for wi, (mid_year, labels, vars_w) in enumerate(total_comm_data):
        mid_years.append(mid_year)
        if wi > 0 and prev_labels is not None:
            common = sorted(set(total_comm_data[wi - 1][2]) & set(vars_w))
            aligned = align_labels_greedy(prev_labels, labels, common)
            current = aligned
        else:
            current = labels

        for vi, v in enumerate(heatmap_vars):
            if v in current:
                heatmap_data[vi, wi] = current[v]
        prev_labels = current

    fig, ax = plt.subplots(figsize=(16, max(8, len(heatmap_vars) * 0.15)))
    cmap = plt.cm.get_cmap("tab20", int(np.nanmax(heatmap_data)) + 1)
    im = ax.imshow(heatmap_data, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(mid_years)))
    ax.set_xticklabels([f"{y:.0f}" for y in mid_years], rotation=45, fontsize=7)
    ax.set_yticks(range(len(heatmap_vars)))
    ax.set_yticklabels(heatmap_vars, fontsize=5)
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Variable")
    ax.set_title("Community Membership Over Time (Total Network, POLVIEWS/PARTYID excluded)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_05b_community_cores.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ══════════════════════════════════════════════════════════
    # Section B: Hub Migration
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SECTION B: HUB MIGRATION (POLVIEWS/PARTYID EXCLUDED)")
    print("=" * 60)

    # Compute degree + betweenness centrality at each window
    def compute_centralities(windows_list, network_key):
        """Returns list of (mid_year, degree_dict, betweenness_dict)."""
        results = []
        for w in windows_list:
            G = build_graph(w.networks[network_key])
            deg = nx.degree_centrality(G)
            btw = nx.betweenness_centrality(G, weight="weight")
            results.append((w.mid_year, deg, btw))
        return results

    cent_total = compute_centralities(total_windows, "total")
    cent_lib = compute_centralities(group_windows, "lib")
    cent_con = compute_centralities(group_windows, "con")

    # Centrality rank correlation between consecutive windows
    def consecutive_rank_corr(cent_data):
        results = []
        for i in range(1, len(cent_data)):
            mid_year = cent_data[i][0]
            prev_deg = cent_data[i - 1][1]
            curr_deg = cent_data[i][1]
            common = sorted(set(prev_deg.keys()) & set(curr_deg.keys()))
            if len(common) < 10:
                continue
            prev_vals = [prev_deg[n] for n in common]
            curr_vals = [curr_deg[n] for n in common]
            rho, p = spearmanr(prev_vals, curr_vals)
            results.append({"mid_year": mid_year, "spearman_rho": rho, "p_value": p})
        return pd.DataFrame(results)

    rank_total = consecutive_rank_corr(cent_total)
    rank_lib = consecutive_rank_corr(cent_lib)
    rank_con = consecutive_rank_corr(cent_con)

    print("\n=== Centrality Rank Stability (no POLVIEWS/PARTYID) ===")
    print(f"Total: mean rho = {rank_total['spearman_rho'].mean():.3f}")
    print(f"Lib:   mean rho = {rank_lib['spearman_rho'].mean():.3f}")
    print(f"Con:   mean rho = {rank_con['spearman_rho'].mean():.3f}")

    # Figure 5: Centrality rank stability
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rank_total["mid_year"], rank_total["spearman_rho"], "ko-", label="Total", linewidth=2)
    ax.plot(rank_lib["mid_year"], rank_lib["spearman_rho"], "b.-", label="Liberal", linewidth=1.5)
    ax.plot(rank_con["mid_year"], rank_con["spearman_rho"], "r.-", label="Conservative", linewidth=1.5)
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Spearman rho (degree centrality)")
    ax.set_title("Centrality Rank Stability Between Consecutive Windows\n(POLVIEWS/PARTYID excluded)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_05b_centrality_rank_corr.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Top-10 hub trajectories (total network degree centrality over time)
    # Find variables with highest average degree centrality
    avg_degree = {}
    for mid_year, deg, _ in cent_total:
        for v, d in deg.items():
            if v not in avg_degree:
                avg_degree[v] = []
            avg_degree[v].append(d)
    avg_degree_mean = {v: np.mean(vals) for v, vals in avg_degree.items() if len(vals) >= 5}
    top_hubs = sorted(avg_degree_mean.keys(), key=lambda v: avg_degree_mean[v], reverse=True)[:10]

    # Figure 4: Top-10 degree centrality trajectories
    fig, ax = plt.subplots(figsize=(14, 6))
    for v in top_hubs:
        years_v = []
        deg_v = []
        for mid_year, deg, _ in cent_total:
            if v in deg:
                years_v.append(mid_year)
                deg_v.append(deg[v])
        ax.plot(years_v, deg_v, ".-", label=v, linewidth=1.5)
    ax.set_xlabel("Window Midpoint (year)")
    ax.set_ylabel("Degree Centrality")
    ax.set_title("Top-10 Hub Variable Degree Centrality Over Time\n(Total Network, POLVIEWS/PARTYID excluded)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_05b_hub_degree_top10.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n=== Top-10 Hub Variables (no POLVIEWS/PARTYID) ===")
    for v in top_hubs:
        print(f"  {v:30s} mean_deg_cent={avg_degree_mean[v]:.4f}")

    # Biggest gainers/losers: regress degree centrality on time
    cent_slopes = []
    for v in avg_degree.keys():
        ts = []
        for mid_year, deg, _ in cent_total:
            if v in deg:
                ts.append((mid_year, deg[v]))
        if len(ts) < 5:
            continue
        years_arr = np.array([t[0] for t in ts])
        deg_arr = np.array([t[1] for t in ts])
        slope, intercept, r, p, _ = linregress(years_arr, deg_arr)
        cent_slopes.append({
            "variable": v, "slope": slope, "r": r, "p": p,
            "mean_deg": np.mean(deg_arr),
            "first_deg": deg_arr[0], "last_deg": deg_arr[-1],
        })

    df_slopes = pd.DataFrame(cent_slopes).sort_values("slope", ascending=False)

    print("\n=== Top 10 Centrality GAINERS (no POLVIEWS/PARTYID) ===")
    print(df_slopes.head(10)[["variable", "slope", "r", "p", "mean_deg"]].to_string(index=False))
    print("\n=== Top 10 Centrality LOSERS (no POLVIEWS/PARTYID) ===")
    print(df_slopes.tail(10).sort_values("slope")[["variable", "slope", "r", "p", "mean_deg"]].to_string(index=False))

    # Figure 6: Gainers/losers bar chart
    top_gainers = df_slopes.head(10)
    top_losers = df_slopes.tail(10).sort_values("slope")
    combined = pd.concat([top_gainers, top_losers])

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["green" if s > 0 else "red" for s in combined["slope"]]
    ax.barh(range(len(combined)), combined["slope"].values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined["variable"].values, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Degree Centrality Slope (per year)")
    ax.set_title("Biggest Centrality Gainers (green) and Losers (red)\n(POLVIEWS/PARTYID excluded)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sound_05b_centrality_gainers_losers.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary stats
    print("\n=== Summary (POLVIEWS/PARTYID excluded) ===")
    print(f"Fixed variables: {len(vars_no_pol)} (from {len(fixed_vars)} minus POLVIEWS, PARTYID)")
    print(f"NMI (total, mean): {nmi_total['nmi'].mean():.3f}")
    print(f"NMI (lib, mean):   {nmi_lib['nmi'].mean():.3f}")
    print(f"NMI (con, mean):   {nmi_con['nmi'].mean():.3f}")
    print(f"Stable core variables (>= 80%): {len(stable_cores)}/{len(vars_no_pol)}")
    print(f"Rank stability (total, mean rho): {rank_total['spearman_rho'].mean():.3f}")
    print(f"Rank stability (lib, mean rho):   {rank_lib['spearman_rho'].mean():.3f}")
    print(f"Rank stability (con, mean rho):   {rank_con['spearman_rho'].mean():.3f}")

    print("\nDone. Figures saved to figures/sound_05b_*.png")


if __name__ == "__main__":
    main()
