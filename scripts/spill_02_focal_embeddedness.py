"""
Spill 02: Does spillover strength track how embedded the focal item is? (exploratory)

HOMOSEX passed every spill_01 check; GRASS passed the go criterion but failed the
specificity checks. GRASS also has almost no partial-correlation edges, and spillover
needs edges to travel along. If that is the reason, then across every item used as focal
in spill_01's placebo, r_own should rise with the item's embeddedness.

Two embeddedness measures per focal item k, from the baseline networks of the two groups:
  strength     mean over groups of sum_j |w_g,kj|      (how connected k is)
  difference   sum_j |w_g1,kj - w_g2,kj|               (how differently the groups wire k;
                                                        the DiD identifies off this)

Reads the GRASS run's results, where no item has same-object exclusions, so every focal
item is treated alike. Exploratory: specified after seeing the GRASS result, and reported
as such.

Usage: PYTHONIOENCODING=utf-8 python scripts/spill_02_focal_embeddedness.py
       (run spill_01 with --focal GRASS first)
Outputs: stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
NOT_OUTCOMES = {"POLVIEWS", "PARTYID"}


def embeddedness(nets, groups, estimator, k):
    g1, g2 = groups
    W1, W2 = nets[g1, estimator], nets[g2, estimator]
    common = [j for j in W1.index.intersection(W2.index) if j != k and j not in NOT_OUTCOMES]
    w1, w2 = W1.loc[k, common].values, W2.loc[k, common].values
    return 0.5 * (np.abs(w1).sum() + np.abs(w2).sum()), np.abs(w1 - w2).sum()


def main():
    with open(CACHE_DIR / "spill_01_grass_results.pkl", "rb") as f:
        results = pickle.load(f)

    print("=" * 70)
    print("SPILL 02: SPILLOVER STRENGTH VS FOCAL EMBEDDEDNESS (exploratory)")
    print("=" * 70)
    for condition in ("B", "D"):
        res = results["conditions"][condition]
        for est in ("partial", "raw"):
            fp = res[f"focal_placebo_{est}"]
            rows = []
            for k, r in fp.items():
                strength, diff = embeddedness(res["nets"], res["groups"], est, k)
                rows.append({"item": k, "r_own": r, "strength": strength, "difference": diff})
            df = pd.DataFrame(rows).set_index("item")
            rho_s, p_s = spearmanr(df["strength"], df["r_own"])
            rho_d, p_d = spearmanr(df["difference"], df["r_own"])
            print(f"\nCondition {condition}, {est}: {len(df)} focal items")
            print(f"  r_own vs strength:   Spearman {rho_s:+.3f} (p={p_s:.3f})")
            print(f"  r_own vs difference: Spearman {rho_d:+.3f} (p={p_d:.3f})")
            for item in ("HOMOSEX", "GRASS"):
                if item in df.index:
                    row = df.loc[item]
                    s_rank = int((df["strength"] >= row["strength"]).sum())
                    d_rank = int((df["difference"] >= row["difference"]).sum())
                    print(f"  {item:8s} r_own={row['r_own']:+.3f}  strength rank {s_rank}/{len(df)}  "
                          f"difference rank {d_rank}/{len(df)}")
            top = df.sort_values("r_own", ascending=False).head(5)
            print("  top r_own: " + ", ".join(f"{i} {v:+.2f}" for i, v in top["r_own"].items()))


if __name__ == "__main__":
    main()
