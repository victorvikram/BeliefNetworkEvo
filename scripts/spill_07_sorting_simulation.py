"""
Spill 07: Does sorting by arrangement move camps along their own structure? (simulation)

spill_06 found that people leave a camp when their belief pattern fits the other camp's
network better than their own, beyond their positions. The spillover tests (spill_01) found
that camps' beliefs move together along each camp's own structure, more than individuals'
changes can explain. This connects the two: if camps changed *only* through who joins and
leaves, with every person's beliefs frozen at wave 1, would they move the way the
cross-sections show — and how much of that comes from sorting by arrangement rather than by
position?

Fixed before the first run (tracker, 2026-10-15). 2006-14 panels.
  Models   exit: spill_06's full model for wave-1 liberals and conservatives.
           entry: for wave-1 moderates, two logits (joined liberal; joined conservative) on
           relative fit towards liberal (fit to liberal minus conservative network), position
           towards liberal, mean fit, age, degree, sex and panel.
  Scenarios, each a change in camp means computed from wave-1 beliefs only:
           actual      real wave-3 camps (who actually joined and left, crossings included)
           full        model probabilities of staying, leaving and joining
           position    the same with the relative-fit coefficients set to zero
           arrangement full minus position: the shift owed to sorting by arrangement
  Observed: the cross-sectional camp change pooled over 2006->10, 2008->12, 2010->14, with
           the panel's own wave-1 respondents removed.
  Measures on y = D_lib - D_con, the camp-differenced change across items:
           correlation with the observed y; network lag — y against W_lib D_lib - W_con D_con
           (own networks) and against W_con D_lib - W_lib D_con (swapped)
  Verdict (raw networks primary, partial secondary): the arrangement contribution correlates
           positively with the observed y, and its own-network lag exceeds its swapped lag,
           both with 95% person-bootstrap intervals excluding zero.
Simplification: direct crossings from one camp to the other are exits, not entries.

Usage: PYTHONIOENCODING=utf-8 python scripts/spill_07_sorting_simulation.py [--quick]
Outputs: data/cache/spill_07_results.pkl, stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import io
import contextlib
import pickle
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

import spill_06_structured_sorting as s6
from src.loaders.clean_raw_data import clean_datasets, DataConfig

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
WINDOWS = (2006, 2008, 2010)
ESTIMATORS = s6.ESTIMATORS
SCENARIOS = ("actual", "full", "position", "arrangement")
N_BOOT = 20 if "--quick" in sys.argv else 300
SEED = 20261019


# ── Inputs ─────────────────────────────────────────────────────────


def person_fits(panel, xs, camp_xs, items, nets):
    """Fit to each camp's network and position towards liberal, for everyone incl. moderates."""
    mu, sd = xs.mean(), xs.std()
    Z = (panel[items] - mu) / sd
    M = Z.notna().astype(float).values
    Z0 = Z.fillna(0).values
    lib_z = ((xs[camp_xs.values == "lib"].mean() - mu) / sd).values
    con_z = ((xs[camp_xs.values == "con"].mean() - mu) / sd).values
    d = np.nan_to_num(lib_z - con_z)
    out = pd.DataFrame(index=panel.index)
    out["pos_lib"] = (Z0 @ d) / np.maximum(M @ np.abs(d), 1e-9)
    for est in ESTIMATORS:
        out[f"fit_lib_{est}"] = s6.fit_scores(Z0, M, nets["lib", est])
        out[f"fit_con_{est}"] = s6.fit_scores(Z0, M, nets["con", est])
    return out


def observed_change(items, sd, exclude):
    """Cross-sectional camp change, pooled over the three panel windows."""
    with contextlib.redirect_stdout(io.StringIO()):
        cleaned = clean_datasets()
    keys = pd.MultiIndex.from_frame(cleaned[["YEAR", "ID"]].astype(float))
    df = cleaned[~keys.isin(exclude) & cleaned["POLVIEWS"].notna() & (cleaned["POLVIEWS"] != 0)]
    camp = np.where(df["POLVIEWS"] < 0, "lib", "con")
    out = {}
    for c in ("lib", "con"):
        diffs = [df[(df["YEAR"] == t + 4) & (camp == c)][items].mean()
                 - df[(df["YEAR"] == t) & (camp == c)][items].mean() for t in WINDOWS]
        out[c] = pd.concat(diffs, axis=1).mean(axis=1) / sd
    return out


# ── Models ─────────────────────────────────────────────────────────


def logistic(eta):
    return 1 / (1 + np.exp(-eta))


def exit_probs(members, pred, est):
    """Probability of leaving the camp, with and without the relative-fit term."""
    X = s6.design(members, pred, est)
    ok = X.notna().all(axis=1)
    params = sm.Logit(members.loc[ok, "switch"], X[ok]).fit(disp=0).params
    no_rel = params.copy()
    no_rel["rel_fit"] = 0.0
    full = pd.Series(np.nan, index=members.index)
    pos = pd.Series(np.nan, index=members.index)
    full[ok] = logistic(X[ok] @ params)
    pos[ok] = logistic(X[ok] @ no_rel)
    return full, pos


def entry_design(mods, fits, est):
    z = s6.zscore
    X = pd.DataFrame(index=mods.index)
    X["rel_lib"] = z(fits[f"fit_lib_{est}"] - fits[f"fit_con_{est}"])
    X["pos_lib"] = z(fits["pos_lib"])
    X["mean_fit"] = z((fits[f"fit_lib_{est}"] + fits[f"fit_con_{est}"]) / 2)
    X["age"] = z(mods["AGE_1"])
    X["degree"] = mods["DEGREE_1"]
    X["female"] = (mods["SEX_1"] == 2).astype(float)
    for pv in sorted(mods["PANEL"].unique())[1:]:
        X[f"p{pv}"] = (mods["PANEL"] == pv).astype(float)
    return sm.add_constant(X)


def entry_probs(mods, fits, est):
    """Probability that a wave-1 moderate joins each camp, with and without relative fit."""
    X = entry_design(mods, fits, est)
    ok = X.notna().all(axis=1)
    out = {}
    for c, joined in (("lib", mods["POL3"] < 0), ("con", mods["POL3"] > 0)):
        params = sm.Logit(joined[ok].astype(int), X[ok]).fit(disp=0).params
        no_rel = params.copy()
        no_rel["rel_lib"] = 0.0
        full = pd.Series(0.0, index=mods.index)
        pos = pd.Series(0.0, index=mods.index)
        full[ok] = logistic(X[ok] @ params)
        pos[ok] = logistic(X[ok] @ no_rel)
        out[c] = (full, pos)
    return out


# ── Scenarios and measures ─────────────────────────────────────────


def weighted_mean(X, w):
    present = X.notna()
    return X.fillna(0).mul(w, axis=0).sum() / present.mul(w, axis=0).sum()


def scenario_shifts(panel, pred, fits, items, sd, est):
    """Camp shifts under each scenario, from wave-1 beliefs only."""
    X = panel[items]
    members = {c: panel["camp"] == c for c in ("lib", "con")}
    camp_rows = panel[panel["camp"].isin(["lib", "con"])]
    p_exit_full, p_exit_pos = exit_probs(camp_rows, pred.loc[camp_rows.index], est)
    mods = panel[panel["camp"] == "mod"]
    p_join = entry_probs(mods, fits.loc[mods.index], est)

    shifts = {s: {} for s in SCENARIOS}
    for c in ("lib", "con"):
        base = X[members[c]].mean()
        in_c3 = (panel["POL3"] < 0) if c == "lib" else (panel["POL3"] > 0)
        shifts["actual"][c] = (X[in_c3].mean() - base) / sd
        for scen, p_exit, k in (("full", p_exit_full, 0), ("position", p_exit_pos, 1)):
            w = pd.Series(0.0, index=panel.index)
            w[camp_rows.index[camp_rows["camp"] == c]] = 1 - p_exit[camp_rows["camp"] == c].fillna(0)
            w[mods.index] = p_join[c][k]
            shifts[scen][c] = (weighted_mean(X, w) - base) / sd
        shifts["arrangement"][c] = shifts["full"][c] - shifts["position"][c]
    return shifts


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def measures(shift, observed, nets, est, items):
    d_lib, d_con = shift["lib"].reindex(items), shift["con"].reindex(items)
    ok = d_lib.notna() & d_con.notna() & observed["lib"].reindex(items).notna() \
        & observed["con"].reindex(items).notna()
    keep = [i for i in items if ok[i]]
    W_lib = nets["lib", est].loc[keep, keep].values
    W_con = nets["con", est].loc[keep, keep].values
    a, b = d_lib[keep].values, d_con[keep].values
    y = a - b
    y_obs = (observed["lib"] - observed["con"])[keep].values
    return {"obs_corr": corr(y, y_obs),
            "lag_own": corr(y, W_lib @ a - W_con @ b),
            "lag_swap": corr(y, W_con @ a - W_lib @ b),
            "size": float(np.linalg.norm(y)), "n_items": len(keep)}


def all_measures(panel, pred, fits, items, sd, observed, nets):
    out = {}
    for est in ESTIMATORS:
        sh = scenario_shifts(panel, pred, fits, items, sd, est)
        out[est] = {s: measures(sh[s], observed, nets, est, items) for s in SCENARIOS}
    return out


# ── Main ───────────────────────────────────────────────────────────


def main():
    t0 = time.time()
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
    fits = person_fits(panel, xs, camp_xs, items, nets)
    observed = observed_change(items, sd, exclude)

    print("=" * 74)
    print("SPILL 07: SORTING SIMULATION — beliefs frozen at wave 1 (GSS 2006-14 panels)")
    print("=" * 74)
    print(pd.crosstab(panel["camp"], np.sign(panel["POL3"]).map({-1: "lib", 0: "mod", 1: "con"}),
                      rownames=["wave 1"], colnames=["wave 3"]).to_string())

    obs = all_measures(panel, pred, fits, items, sd, observed, nets)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(panel), len(panel))
        bp = panel.iloc[idx].reset_index(drop=True)
        try:
            boots.append(all_measures(bp, pred.iloc[idx].reset_index(drop=True),
                                      fits.iloc[idx].reset_index(drop=True),
                                      items, sd, observed, nets))
        except Exception:
            continue

    verdict = []
    for est in ESTIMATORS:
        print(f"\n[{est} networks]   {len(boots)} bootstraps; 95% intervals in brackets")
        print(f"  {'scenario':12s} {'corr with observed y':>24s} {'lag own':>8s} {'lag swap':>9s} "
              f"{'own - swap':>20s} {'size':>6s}")
        for scen in SCENARIOS:
            m = obs[est][scen]
            oc = np.nanpercentile([b[est][scen]["obs_corr"] for b in boots], [2.5, 97.5])
            gap = np.array([b[est][scen]["lag_own"] - b[est][scen]["lag_swap"] for b in boots])
            gi = np.nanpercentile(gap, [2.5, 97.5])
            print(f"  {scen:12s} {m['obs_corr']:+.3f} [{oc[0]:+.3f},{oc[1]:+.3f}]   "
                  f"{m['lag_own']:+.3f} {m['lag_swap']:+9.3f} "
                  f"{m['lag_own'] - m['lag_swap']:+.3f} [{gi[0]:+.3f},{gi[1]:+.3f}] {m['size']:6.2f}")
            if scen == "arrangement" and oc[0] > 0 and gi[0] > 0:
                verdict.append(est)
        print(f"  arrangement share of the simulated shift (size ratio): "
              f"{obs[est]['arrangement']['size'] / obs[est]['full']['size']:.2f}; "
              f"simulated full / actual sorting size: {obs[est]['full']['size'] / obs[est]['actual']['size']:.2f}")

    print("\n" + "=" * 74)
    print(f"VERDICT: {'SUPPORTED via ' + ', '.join(verdict) if verdict else 'NOT SUPPORTED'}")
    print("=" * 74)
    with open(CACHE_DIR / "spill_07_results.pkl", "wb") as fh:
        pickle.dump({"observed": obs, "boots": boots, "verdict": verdict}, fh)
    print(f"Saved data/cache/spill_07_results.pkl; elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
