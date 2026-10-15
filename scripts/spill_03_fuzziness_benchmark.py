"""
Spill 03: Is condition F weak because of sorting, or because its groups are fuzzy?
(exploratory)

Condition F of spill_01 removes ideological sorting by assigning groups from traits fixed
before adulthood. But those predicted camps are only mildly ideological (background model
AUC about 0.66), so their networks differ less and the test has less power. A weak F result
is therefore ambiguous.

This benchmark builds groups that are exactly as fuzzy as F's but are made of
self-identified liberals, moderates and conservatives, mixed in F's baseline proportions —
so they are as ideological as F's groups yet still exposed to re-sorting. Everything else
matches F: the same eligible respondents (US-born, born 1925-1962, complete background),
the same group sizes per window, the same reweighting, the same estimators.

  F about equal to G   F's weakness is fuzziness; sorting is not shown to drive the effect
  F well below G       removing sorting removes part of the effect

G is drawn at random, so it is repeated over draws and summarised by its mean and 95%
range. Designed after the spill_01 smoke run showed F's point estimates were small, and
before the full F results; reported as exploratory.

Usage: PYTHONIOENCODING=utf-8 python scripts/spill_03_fuzziness_benchmark.py [--focal ITEM]
Outputs: stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import contextlib

import numpy as np
import pandas as pd

import spill_01_homosex_pilot as s1

N_DRAWS = 30
SEED = 20261016


def self_id(polviews):
    return np.where(polviews < 0, "lib", np.where(polviews > 0, "con", "mod"))


def benchmark_sample(eligible, f_sample, shares, rng):
    """Self-identified respondents mixed to F's baseline proportions, at F's group sizes."""
    g1, g2 = s1.GROUPS["F"]
    parts = []
    for window in ("base", "late"):
        pool = eligible[eligible["window"] == window].copy()
        sizes = f_sample[f_sample["window"] == window]["group"].value_counts()
        taken = set()
        for group in (g1, g2):
            for camp in ("lib", "mod", "con"):
                want = int(round(sizes[group] * shares[group][camp]))
                avail = pool[(pool["camp"] == camp) & ~pool.index.isin(taken)]
                pick = avail.sample(n=min(want, len(avail)), random_state=rng)
                taken.update(pick.index)
                parts.append(pick.assign(group=group))
    return pd.concat(parts).reset_index(drop=True)


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        df, belief_vars = s1.load_sample()
        f_sample, diag = s1.predicted_camps(df)
    g1, g2 = s1.GROUPS["F"]

    eligible = df[(df["BORN"] == 1) & df["COHORT"].between(*s1.COHORT_RANGE["F"])]
    eligible = eligible[eligible[s1.BACKGROUND_COLS + s1.REWEIGHT_COLS["F"]].notna().all(axis=1)]
    eligible = eligible[eligible["POLVIEWS"].notna()].copy()
    eligible["camp"] = self_id(eligible["POLVIEWS"])

    base_f = f_sample[(f_sample["window"] == "base") & f_sample["POLVIEWS"].notna()]
    shares = {}
    for group in (g1, g2):
        camps = pd.Series(self_id(base_f.loc[base_f["group"] == group, "POLVIEWS"]))
        shares[group] = camps.value_counts(normalize=True).reindex(["lib", "mod", "con"]).fillna(0).to_dict()

    print("=" * 70)
    print(f"SPILL 03: FUZZINESS BENCHMARK FOR CONDITION F ({s1.FOCAL}, exploratory)")
    print("=" * 70)
    for group in (g1, g2):
        print(f"F baseline mix, {group}: " + ", ".join(f"{k} {v:.0%}" for k, v in shares[group].items()))

    with contextlib.redirect_stdout(io.StringIO()):
        nets, shifts = s1.networks_and_shifts(f_sample, belief_vars, "F")
    f_obs = {est: s1.spillover(nets, shifts, (g1, g2), est) for est in s1.ESTIMATORS}

    rng = np.random.RandomState(SEED)
    draws = {est: [] for est in s1.ESTIMATORS}
    for _ in range(N_DRAWS):
        sample = benchmark_sample(eligible, f_sample, shares, rng)
        with contextlib.redirect_stdout(io.StringIO()):
            nets, shifts = s1.networks_and_shifts(sample, belief_vars, "F")
        for est in s1.ESTIMATORS:
            draws[est].append(s1.spillover(nets, shifts, (g1, g2), est))

    for est in s1.ESTIMATORS:
        g = pd.DataFrame(draws[est])
        print(f"\n[{est}]")
        for stat in ("beta", "r_own", "r_swap"):
            lo, hi = np.nanpercentile(g[stat], [2.5, 97.5])
            print(f"  {stat:7s} F = {f_obs[est][stat]:+.3f}   G mean = {g[stat].mean():+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}] over {N_DRAWS} draws")
        below = (g["r_own"] > f_obs[est]["r_own"]).mean()
        print(f"  share of G draws with r_own above F: {below:.0%}")


if __name__ == "__main__":
    main()
