"""
Spill 04: Panel spillover test — the same people, camps fixed at their first interview.

spill_01's lib/con result could be produced by re-sorting: self-identified liberals in a
later period are not the people who called themselves liberal earlier. Condition F removed
sorting with predicted camps, but those groups were too fuzzy to give a clear answer. The
GSS panels follow the same respondents over three waves, so fixing each person's camp at
wave 1 removes sorting entirely while keeping the camps sharply ideological.

Design, fixed before the first run (tracker, 2026-10-15):
  - the 2006, 2008 and 2010 three-wave panels, pooled; respondents interviewed at waves 1
    and 3 (2006->10, 2008->12, 2010->14)
  - camp from wave-1 POLVIEWS (liberal < 0 < conservative); a person keeps it in both waves
  - networks from wave-1 answers, per camp: regularized partial correlations (graphical
    LASSO, alpha = 0.2, nearest-positive-definite fallback) and raw Pearson correlations
  - change in item j for camp g: mean over the camp's respondents of (wave 3 - wave 1),
    using respondents answering at both waves, in pooled wave-1 SDs
  - camps differenced as in spill_01:
        D_lib,j - D_con,j = beta (w_lib,j D_lib,H - w_con,j D_con,H) + a + e_j
  - permutation null shuffles camp labels per person; bootstrap resamples persons within camp
  - survival rule, the same as spill_01 condition F: beta > 0, permutation p < 0.05 and
    r_own > r_swap, for at least one estimator

Regression to the mean, found in the smoke run: choosing camps on a noisy wave-1 POLVIEWS
pulls wave-3 answers towards the centre on every item correlated with ideology, so
liberals drift right and conservatives left with no real change (the smoke run showed
conservatives liberalising on HOMOSEX far more than liberals). `--camp-from mean13` labels
each person by the mean of their wave-1 and wave-3 POLVIEWS instead. The label is still
fixed per person, so there is still no re-sorting, and measurement error now enters both
waves symmetrically, so regression to the mean cancels in the wave-3 minus wave-1 change.
The wave-1 rule is the pre-specified design; mean13 is a post-smoke-run correction and is
reported as such.

HOMOSEX is the test; GRASS and CAPPUN run alongside as secondary. The belief set is
DataConfig.all_questions, the directly mapped items, without the derived vote and religion
dummies that spill_01's cleaned frame also carries. Unweighted, like spill_01.

Usage: PYTHONIOENCODING=utf-8 python scripts/spill_04_panel_test.py [--camp-from mean13] [--quick]
Outputs: figures/spill_04_panel_test<_mean13>.png, data/cache/spill_04_results<_mean13>.pkl, stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import contextlib
import pickle
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyreadstat as prs
from joblib import Parallel, delayed

from src.loaders.clean_raw_data import DataConfig, transform_column
from src.generators.corr_make_network import (
    calculate_correlation_matrix, calculate_regularized_partial_correlations, filter_nans,
)

ROOT = Path(__file__).resolve().parent.parent
PANEL_DIR = ROOT / "data" / "raw" / "gss_panels"
FIGURES_DIR = ROOT / "figures"
CACHE_DIR = ROOT / "data" / "cache"

PANEL_FILES = {
    2006: "GSS_2006_Panel_Stata/GSS_panel06w123_R6a - Stata.dta",
    2008: "GSS_2008_Panel_stata/GSS_panel08w123_R6 - stata.dta",
    2010: "GSS_2010_Panel_stata/GSS_panel2010w123_R6 - stata.dta",
}
FOCALS = ("HOMOSEX", "GRASS", "CAPPUN")
SAME_OBJECT = {"HOMOSEX": {"MARHOMO", "SPKHOMO", "COLHOMO", "LIBHOMO"},
               "GRASS": set(), "CAPPUN": set()}
NOT_OUTCOMES = {"POLVIEWS", "PARTYID"}
CAMPS = ("lib", "con")
ESTIMATORS = ("partial", "raw")
ALPHA = 0.2
MIN_N = 100
MIN_FOCAL_SHIFT = 0.10
N_PERM = 12 if "--quick" in sys.argv else 500
N_BOOT = 12 if "--quick" in sys.argv else 300
N_JOBS = 12
SEED = 20261017
CAMP_RULE = sys.argv[sys.argv.index("--camp-from") + 1] if "--camp-from" in sys.argv else "wave1"
TAG = "" if CAMP_RULE == "wave1" else f"_{CAMP_RULE}"


# ── Data ───────────────────────────────────────────────────────────


def recode(series, mapping):
    """Panel missing codes arrive lower-case ('d', 'i', 'n'); DataConfig expects upper-case."""
    series = series.map(lambda v: v.upper() if isinstance(v, str) else v)
    return transform_column(series, mapping)


def load_panels():
    """Wave-1 and wave-3 answers for every wave-3 completer, one row per person."""
    config = DataConfig()
    mappings = config.all_mappings
    items = sorted(config.all_questions)
    base_parts, late_parts, panel_ids = [], [], []
    for panel, rel in PANEL_FILES.items():
        path = PANEL_DIR / rel
        _, meta = prs.read_dta(str(path), metadataonly=True)
        lower = {c.lower(): c for c in meta.column_names}
        present = [i for i in items if f"{i.lower()}_1" in lower and f"{i.lower()}_3" in lower]
        cols = [lower[f"{i.lower()}_{w}"] for i in present for w in (1, 3)] + [lower["year_3"]]
        cols = list(dict.fromkeys(cols))
        raw, _ = prs.read_dta(str(path), usecols=cols, user_missing=True)
        raw.columns = [c.lower() for c in raw.columns]
        done = pd.to_numeric(raw["year_3"], errors="coerce").notna()
        raw = raw[done].reset_index(drop=True)
        base = pd.DataFrame({i: recode(raw[f"{i.lower()}_1"], mappings[i]) for i in present})
        late = pd.DataFrame({i: recode(raw[f"{i.lower()}_3"], mappings[i]) for i in present})
        base_parts.append(base)
        late_parts.append(late)
        panel_ids.append(pd.Series(panel, index=base.index))
    base = pd.concat(base_parts, ignore_index=True).astype(float)
    late = pd.concat(late_parts, ignore_index=True).astype(float)
    panel = pd.concat(panel_ids, ignore_index=True)
    ideology = base["POLVIEWS"] if CAMP_RULE == "wave1" else (base["POLVIEWS"] + late["POLVIEWS"]) / 2
    camp = pd.Series(np.where(ideology < 0, "lib", np.where(ideology > 0, "con", "mod")),
                     index=base.index)
    camp[ideology.isna()] = "mod"
    keep = camp.isin(CAMPS)
    return (base[keep].reset_index(drop=True), late[keep].reset_index(drop=True),
            camp[keep].reset_index(drop=True), panel[keep].reset_index(drop=True))


# ── Statistics ─────────────────────────────────────────────────────


def partial_network(frame):
    """Regularized partial correlations; nearest-positive-definite refit if ill-conditioned."""
    net = calculate_correlation_matrix(
        frame, method="pearson", partial=True,
        edge_suppression="regularization", suppression_params={"regularization": ALPHA},
    )
    if net is not None:
        return net
    corr = frame.corr()
    clean, removed = filter_nans(corr.values)
    cols = [c for i, c in enumerate(corr.columns) if i not in removed]
    vals, vecs = np.linalg.eigh(clean)
    psd = (vecs * np.clip(vals, 1e-4, None)) @ vecs.T
    d = np.sqrt(np.diag(psd))
    net = pd.DataFrame(calculate_regularized_partial_correlations(psd / np.outer(d, d), alpha=ALPHA),
                       index=cols, columns=cols)
    np.fill_diagonal(net.values, 0)
    return net


def networks_and_shifts(base, late, camp):
    nets = {}
    for g in CAMPS:
        b = base[camp == g]
        nets[g, "partial"] = partial_network(b)
        raw = b.corr(min_periods=MIN_N).fillna(0)
        np.fill_diagonal(raw.values, 0)
        nets[g, "raw"] = raw
    sd = base.std()
    change = late - base                       # NaN unless answered at both waves
    shifts = {}
    for g in CAMPS:
        c = change[camp == g]
        d = c.mean() / sd
        d[(c.notna().sum() < MIN_N) | (sd == 0)] = np.nan
        shifts[g] = d
    return nets, shifts


def slope(x, y):
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    return np.nan if denom == 0 else float((xc * (y - y.mean())).sum() / denom)


def corr(x, y):
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spillover(nets, shifts, estimator, focal, exclude):
    W1, W2 = nets["lib", estimator], nets["con", estimator]
    out = {"beta": np.nan, "r_own": np.nan, "r_swap": np.nan, "n_items": 0}
    common = W1.index.intersection(W2.index)
    h1, h2 = shifts["lib"].get(focal, np.nan), shifts["con"].get(focal, np.nan)
    if focal not in common or not (np.isfinite(h1) and np.isfinite(h2)):
        return out
    items = [j for j in common if j != focal and j not in NOT_OUTCOMES | exclude
             and np.isfinite(shifts["lib"].get(j, np.nan)) and np.isfinite(shifts["con"].get(j, np.nan))]
    out["n_items"] = len(items)
    if len(items) < 10:
        return out
    w1, w2 = W1.loc[focal, items].values, W2.loc[focal, items].values
    y = shifts["lib"][items].values - shifts["con"][items].values
    x_own, x_swap = w1 * h1 - w2 * h2, w2 * h1 - w1 * h2
    out.update(beta=slope(x_own, y), r_own=corr(x_own, y), r_swap=corr(x_swap, y))
    return out


def all_stats(base, late, camp):
    nets, shifts = networks_and_shifts(base, late, camp)
    return {(f, est): spillover(nets, shifts, est, f, SAME_OBJECT[f])
            for f in FOCALS for est in ESTIMATORS}


def run_chunk(kind, base, late, camp, seeds):
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if kind == "perm":
            b, l = base, late
            c = pd.Series(rng.permutation(camp.values), index=camp.index)
        else:
            idx = np.concatenate([rng.choice(np.flatnonzero(camp.values == g), (camp == g).sum())
                                  for g in CAMPS])
            b = base.iloc[idx].reset_index(drop=True)
            l = late.iloc[idx].reset_index(drop=True)
            c = camp.iloc[idx].reset_index(drop=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out.append(all_stats(b, l, c))
        except Exception:
            out.append(None)
    return out


def run_replicates(kind, base, late, camp, seeds):
    chunks = np.array_split(np.asarray(seeds), N_JOBS * 2)
    batches = Parallel(n_jobs=N_JOBS)(
        delayed(run_chunk)(kind, base, late, camp, [int(s) for s in ch]) for ch in chunks if len(ch))
    return [r for batch in batches for r in batch if r is not None]


def p_upper(null, value):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    return (1 + (null >= value).sum()) / (1 + len(null))


def focal_rank(nets, shifts, estimator, focal):
    """Rank of the focal item's r_own among items that shifted in both camps."""
    cands = [k for k in nets["lib", estimator].index.intersection(nets["con", estimator].index)
             if k not in NOT_OUTCOMES
             and abs(shifts["lib"].get(k, 0)) >= MIN_FOCAL_SHIFT
             and abs(shifts["con"].get(k, 0)) >= MIN_FOCAL_SHIFT]
    if focal not in cands:
        cands.append(focal)
    r = pd.Series({k: spillover(nets, shifts, estimator, k,
                                SAME_OBJECT.get(k, set()) if k == focal else set())["r_own"]
                   for k in cands}).dropna()
    return int((r >= r.get(focal, np.nan)).sum()), len(r), float(r.median())


# ── Main ───────────────────────────────────────────────────────────


def main():
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        base, late, camp, panel = load_panels()

    print("=" * 70)
    print(f"SPILL 04: PANEL SPILLOVER TEST (camp rule: {CAMP_RULE})")
    print("=" * 70)
    print(f"{N_PERM} permutations, {N_BOOT} bootstraps; items in both waves: {base.shape[1]}")
    print(pd.crosstab(panel, camp).to_string())

    with contextlib.redirect_stdout(io.StringIO()):
        nets, shifts = networks_and_shifts(base, late, camp)
    observed = {(f, est): spillover(nets, shifts, est, f, SAME_OBJECT[f])
                for f in FOCALS for est in ESTIMATORS}

    seeds = np.random.default_rng(SEED).integers(0, 2**31, N_PERM + N_BOOT)
    perms = run_replicates("perm", base, late, camp, seeds[:N_PERM])
    boots = run_replicates("boot", base, late, camp, seeds[N_PERM:])

    summary = {}
    for f in FOCALS:
        print(f"\n── {f} ── shift (pooled wave-1 SDs, within person): "
              f"lib {shifts['lib'].get(f, np.nan):+.3f}, con {shifts['con'].get(f, np.nan):+.3f}")
        for est in ESTIMATORS:
            o = observed[f, est]
            gap = o["r_own"] - o["r_swap"]
            s = {**o, "gap": gap,
                 "p_beta": p_upper([p[f, est]["beta"] for p in perms], o["beta"]),
                 "p_gap": p_upper([p[f, est]["r_own"] - p[f, est]["r_swap"] for p in perms], gap),
                 "beta_ci": np.nanpercentile([b[f, est]["beta"] for b in boots], [2.5, 97.5]),
                 "rank": focal_rank(nets, shifts, est, f)}
            summary[f, est] = s
            rank, n, med = s["rank"]
            print(f"  [{est:7s}] items={s['n_items']:3d}  beta={s['beta']:+.3f} "
                  f"bootCI[{s['beta_ci'][0]:+.3f},{s['beta_ci'][1]:+.3f}] p_perm={s['p_beta']:.3f}  "
                  f"r_own={s['r_own']:+.3f} r_swap={s['r_swap']:+.3f} gap p={s['p_gap']:.3f}  "
                  f"focal rank {rank}/{n} (median r_own {med:+.3f})")

    print("\nPer panel, HOMOSEX point estimates:")
    for pyear in PANEL_FILES:
        m = panel == pyear
        with contextlib.redirect_stdout(io.StringIO()):
            n_p, s_p = networks_and_shifts(base[m], late[m], camp[m])
        parts = []
        for est in ESTIMATORS:
            o = spillover(n_p, s_p, est, "HOMOSEX", SAME_OBJECT["HOMOSEX"])
            parts.append(f"{est} r_own {o['r_own']:+.3f} / r_swap {o['r_swap']:+.3f}")
        print(f"  {pyear}: shift lib {s_p['lib'].get('HOMOSEX', np.nan):+.3f} "
              f"con {s_p['con'].get('HOMOSEX', np.nan):+.3f}; " + "; ".join(parts))

    survives = [est for est in ESTIMATORS
                if summary["HOMOSEX", est]["beta"] > 0 and summary["HOMOSEX", est]["p_beta"] < 0.05
                and summary["HOMOSEX", est]["r_own"] > summary["HOMOSEX", est]["r_swap"]]
    print("\n" + "=" * 70)
    print(f"PANEL VERDICT ({CAMP_RULE}, HOMOSEX): {'SURVIVES via ' + ', '.join(survives) if survives else 'DOES NOT SURVIVE'}")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"{f}\n{est}" for f in FOCALS for est in ESTIMATORS]
    pos = np.arange(len(labels))
    ax.bar(pos - 0.2, [summary[f, e]["r_own"] for f in FOCALS for e in ESTIMATORS], 0.4,
           color="#333333", label="r_own (own camp's wave-1 network)")
    ax.bar(pos + 0.2, [summary[f, e]["r_swap"] for f in FOCALS for e in ESTIMATORS], 0.4,
           color="#bbbbbb", label="r_swap (other camp's network)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("correlation with camp-differenced change")
    ax.set_title(f"Panel test: same people, fixed camps ({CAMP_RULE}), 2006-14 panels", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"spill_04_panel_test{TAG}.png", dpi=150)

    with open(CACHE_DIR / f"spill_04_results{TAG}.pkl", "wb") as fh:
        pickle.dump({"summary": summary, "nets": nets, "shifts": shifts, "survives": survives}, fh)
    print(f"Saved figures/spill_04_panel_test{TAG}.png and data/cache/spill_04_results{TAG}.pkl; "
          f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
