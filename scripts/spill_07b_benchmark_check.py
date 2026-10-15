"""
Spill 07b: Is spill_07's observed benchmark informative? (exploratory)

spill_07 compared sorting-driven camp shifts with the observed cross-sectional camp change
over the panel windows, and found no correlation — not even for the *actual* sorting. Two
checks, both specified after seeing that result:

  1  Reliability. The camp-differenced change y is computed separately for 2006->10 and
     2008->12, which share no survey year and so no respondents. If those two y vectors do
     not correlate, the four-year benchmark is mostly sampling noise and the comparison
     could not succeed.
  2  A benchmark with more signal: the 27-year within-cohort camp change from spill_01
     condition B (1987-91 -> 2014-18). Sorting measured over four years is compared with
     change over three decades, so this asks only whether the *direction* of sorting
     matches the direction the camps actually travelled.

  3  Whether the arrangement-driven shift points anywhere the position-driven shift does
     not: its partial correlation with the 27-year change, holding the position shift fixed
     (run with --distinct).

Usage: PYTHONIOENCODING=utf-8 python scripts/spill_07b_benchmark_check.py [--distinct]
       (needs data/cache/spill_01_homosex_results.pkl from spill_01)
Outputs: stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import contextlib
import pickle

import numpy as np
import pandas as pd

import spill_06_structured_sorting as s6
import spill_07_sorting_simulation as s7
from src.loaders.clean_raw_data import clean_datasets, DataConfig

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def window_y(df, camp, items, sd, t):
    d = {}
    for c in ("lib", "con"):
        d[c] = (df[(df["YEAR"] == t + 4) & (camp == c)][items].mean()
                - df[(df["YEAR"] == t) & (camp == c)][items].mean()) / sd
    return d["lib"] - d["con"]


def corr_on(a, b):
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def main():
    items = sorted(set(DataConfig().all_questions) - s6.NOT_IN_FIT)
    panel = s6.load_panel_wave1(items, keep_moderates=True)
    exclude = pd.MultiIndex.from_frame(panel[["YEAR_1", "ID_1"]].astype(float))
    xs, camp_xs = s6.load_cross_section(items, exclude)
    items = [i for i in items if panel[i].notna().sum() >= s6.MIN_PAIR_N
             and xs[i].notna().sum() >= s6.MIN_PAIR_N]
    xs = xs[items]
    sd = xs.std()

    with contextlib.redirect_stdout(io.StringIO()):
        cleaned = clean_datasets()
    keys = pd.MultiIndex.from_frame(cleaned[["YEAR", "ID"]].astype(float))
    df = cleaned[~keys.isin(exclude) & cleaned["POLVIEWS"].notna() & (cleaned["POLVIEWS"] != 0)]
    camp = np.where(df["POLVIEWS"] < 0, "lib", "con")

    print("=" * 70)
    print("SPILL 07b: CHECKING THE OBSERVED BENCHMARK (exploratory)")
    print("=" * 70)
    y06, y08, y10 = (window_y(df, camp, items, sd, t) for t in (2006, 2008, 2010))
    r, n = corr_on(y06, y08)
    print(f"1. Reliability: y(2006->10) vs y(2008->12), no shared respondents: r = {r:+.3f} over {n} items")
    r, n = corr_on(y06, y10)
    print(f"   y(2006->10) vs y(2010->14), sharing the 2010 sample:         r = {r:+.3f} over {n} items")

    with open(CACHE_DIR / "spill_01_homosex_results.pkl", "rb") as fh:
        b = pickle.load(fh)["conditions"]["B"]["shifts"]
    y_long = (b["lib"] - b["con"]).reindex(items)

    with contextlib.redirect_stdout(io.StringIO()):
        nets = s6.camp_networks(xs, camp_xs, items)
    camp_rows = panel[panel["camp"].isin(["lib", "con"])]
    pred = s6.person_predictors(camp_rows, xs, camp_xs, items, nets).reindex(panel.index)
    fits = s7.person_fits(panel, xs, camp_xs, items, nets)

    print("\n2. Sorting scenarios against the 27-year within-cohort camp change (spill_01 B):")
    for est in s6.ESTIMATORS:
        sh = s7.scenario_shifts(panel, pred, fits, items, sd, est)
        parts = []
        for scen in s7.SCENARIOS:
            y = (sh[scen]["lib"] - sh[scen]["con"]).reindex(items)
            r, n = corr_on(y, y_long)
            parts.append(f"{scen} {r:+.3f}")
        print(f"   [{est}] " + "; ".join(parts) + f"  (n = {n} items)")


if __name__ == "__main__" and "--distinct" not in sys.argv:
    main()


def distinct_direction():
    """Check 3: does the arrangement shift point anywhere position sorting does not?

    Partial correlation of the arrangement-driven y with the 27-year y, holding the
    position-driven y fixed, with a person bootstrap.
    """
    items = sorted(set(DataConfig().all_questions) - s6.NOT_IN_FIT)
    panel = s6.load_panel_wave1(items, keep_moderates=True)
    exclude = pd.MultiIndex.from_frame(panel[["YEAR_1", "ID_1"]].astype(float))
    xs, camp_xs = s6.load_cross_section(items, exclude)
    items = [i for i in items if panel[i].notna().sum() >= s6.MIN_PAIR_N
             and xs[i].notna().sum() >= s6.MIN_PAIR_N]
    xs = xs[items]
    sd = xs.std()
    with contextlib.redirect_stdout(io.StringIO()):
        nets = s6.camp_networks(xs, camp_xs, items)
    camp_rows = panel[panel["camp"].isin(["lib", "con"])]
    pred = s6.person_predictors(camp_rows, xs, camp_xs, items, nets).reindex(panel.index)
    fits = s7.person_fits(panel, xs, camp_xs, items, nets)
    with open(CACHE_DIR / "spill_01_homosex_results.pkl", "rb") as fh:
        b = pickle.load(fh)["conditions"]["B"]["shifts"]
    y_long = (b["lib"] - b["con"]).reindex(items)

    def stats(p, pr, ft, est):
        sh = s7.scenario_shifts(p, pr, ft, items, sd, est)
        ya = (sh["arrangement"]["lib"] - sh["arrangement"]["con"]).reindex(items)
        yp = (sh["position"]["lib"] - sh["position"]["con"]).reindex(items)
        ok = ya.notna() & yp.notna() & y_long.notna()
        ya, yp, yl = ya[ok].values, yp[ok].values, y_long[ok].values
        resid = lambda v: v - np.polyval(np.polyfit(yp, v, 1), yp)
        return np.corrcoef(ya, yp)[0, 1], np.corrcoef(resid(ya), resid(yl))[0, 1]

    print("\n3. Arrangement vs position direction, and partial correlation with the 27-year change:")
    rng = np.random.default_rng(20261020)
    for est in s6.ESTIMATORS:
        r_ap, pr_obs = stats(panel, pred, fits, est)
        boot = []
        for _ in range(200):
            idx = rng.integers(0, len(panel), len(panel))
            try:
                boot.append(stats(panel.iloc[idx].reset_index(drop=True), pred.iloc[idx].reset_index(drop=True),
                                  fits.iloc[idx].reset_index(drop=True), est)[1])
            except Exception:
                continue
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        print(f"   [{est}] corr(arrangement y, position y) = {r_ap:+.3f}; partial corr with 27-year y "
              f"holding position fixed = {pr_obs:+.3f} [{lo:+.3f}, {hi:+.3f}] ({len(boot)} bootstraps)")


if __name__ == "__main__" and "--distinct" in sys.argv:
    distinct_direction()
