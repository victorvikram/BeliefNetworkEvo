"""
Spill 01: HOMOSEX spillover pilot — does a group's own belief network predict how a
shift in one belief spreads to its other beliefs?

Between 1987-91 and 2014-18 liberals and conservatives both liberalised sharply on
HOMOSEX. If population belief structure shapes belief change, each camp's other beliefs
should move in proportion to their baseline edge with HOMOSEX in *that camp's* network.

Design — the two groups differenced, so item-level drift common to both cancels:

    D_gj = beta * w_gj * D_gH + a_g + c_j + e_gj,          g in {g1, g2}
    D_g1,j - D_g2,j = beta * (w_g1,j D_g1,H - w_g2,j D_g2,H) + a + e_j

  D_gj   change in group g's mean of item j, in pooled baseline SDs of j
  w_gj   baseline edge between HOMOSEX and j in group g's network
  beta > 0 means movement follows each group's own arrangement, beyond what they share.

Estimators: regularized partial correlations (graphical LASSO, alpha=0.2, the project's
primary network) and raw Pearson correlations on pairs with at least MIN_N joint
observations. POLVIEWS and PARTYID stay in the networks but are never outcomes, and
neither are the other items about homosexuality (MARHOMO, SPKHOMO, COLHOMO, LIBHOMO),
since spillover means movement in *other* beliefs.

Conditions:
  A  liberals vs conservatives, all respondents
  B  liberals vs conservatives, within-cohort: born 1925-1969, present in both windows,
     so cohort replacement cannot move the means
  C  as B, with each camp's later window reweighted to its baseline composition on cohort
     band, degree, race, sex and region (propensity weights)
  D  college degree vs no degree, US-born, born 1925-1965, reweighted as C without
     degree; US-born so that immigration cannot change who is in a cohort. People
     cannot re-sort into these groups on their attitudes, so D is immune to the ideological
     sorting that A-C cannot rule out (self-identified liberals in 2014-18 are not the
     kind of people who called themselves liberal in 1987-91)
  E  liberals vs conservatives, born 1925-1969, each camp's late window reweighted to its
     baseline mix of party (Dem/Ind/Rep, leaners as partisans) and denomination
     fundamentalism as well as C's demographics
  F  predicted camps, the decisive sorting control. A baseline-window logistic model
     predicts liberal vs conservative from traits fixed before adulthood (region, place
     type, religion and fundamentalism at 16, parents' degrees, own degree, cohort band,
     race, sex). The same model assigns every respondent in both windows; top third vs
     bottom third of predicted probability (cut-points from the baseline window), US-born,
     born 1925-1962, reweighted on cohort band, race, sex and region. Membership never
     depends on self-identification, so re-sorting cannot move it.

Inference: permutation null (group labels shuffled within each window, networks
re-estimated) and respondent bootstrap CIs (resampled within group x window). The
bootstrap adds noise to the estimated edges and so biases beta towards zero; the
permutation test is the primary inference.

Placebos:
  swap      each group's movement predicted from the *other* group's edges (r_swap)
  focal     every other item that shifted in both groups used as the focal item, to ask
            whether HOMOSEX is special or any item's edges predict movement equally well

Graphical LASSO occasionally fails on the pairwise-complete correlation matrix, which need
not be positive definite (it happened for the college group in D). `partial_network()` then
refits on the nearest positive definite correlation matrix, and the run reports when it did.

Robustness: Spearman rank version of r_own (guards against a few high-leverage items),
and r_own with the homosexuality items left in.

Exploratory: network lag — every item's movement against the network-weighted movement
of its neighbours, differenced across groups. Same permutation null.

Sorting verdict, fixed before E and F were run: the result survives sorting if F gives
beta > 0, permutation p < 0.05 and r_own > r_swap for at least one estimator.

Strict go criterion, fixed after the GRASS replication showed the original too lenient, for
any further focal item (all three for the same estimator): the original condition-B rule;
focal rank in the top quartile in B; and condition D beta > 0 with permutation p < 0.05.

Go criterion, fixed before the first run: in condition B, beta > 0 with permutation
p < 0.05 for at least one estimator, and r_own > r_swap for that estimator. Anything else
is a no-go or a redesign, reported as such. The homosexuality-item exclusion, condition D
(US-born only) and the joint-observation floor for raw correlations were added after a 12-replicate
smoke run, to close threats it exposed; the criterion itself was not changed.

Replication: `--focal GRASS` reruns the identical design with marijuana legalisation as the
focal item, under the same go criterion. GRASS has no same-object exclusions, since the item
set holds no other marijuana items.

CAPPUN surfaced in the GRASS run's focal placebo on the 1987-91/2014-18 windows, so its
independent test uses held-out windows: `--focal CAPPUN --base 1993-1996 --late 2006-2010`.

Usage: PYTHONIOENCODING=utf-8 python scripts/spill_01_homosex_pilot.py [--focal ITEM]
           [--base YYYY-YYYY --late YYYY-YYYY] [--quick]
       --focal defaults to HOMOSEX; GRASS is the pre-specified replication
       --base/--late default to 1987-1991 and 2014-2018; other windows add a suffix to outputs
       --quick runs 12 permutations and 12 bootstraps, for checking the pipeline only
Outputs: figures/spill_01_<focal>_pilot.png, data/cache/spill_01_<focal>_results.pkl, stdout
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
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.loaders.clean_raw_data import clean_datasets, DataConfig
from src.loaders.demographics import load_demographics, attach_demographics
from src.generators.corr_make_network import (
    calculate_correlation_matrix, calculate_regularized_partial_correlations, filter_nans,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

FOCAL_SETTINGS = {
    "HOMOSEX": {"same_object": {"MARHOMO", "SPKHOMO", "COLHOMO", "LIBHOMO"},
                "direction": "higher = more disapproving"},
    "GRASS": {"same_object": set(), "direction": "higher = favours legalisation"},
    "CAPPUN": {"same_object": set(), "direction": "higher = favours the death penalty"},
}
FOCAL = sys.argv[sys.argv.index("--focal") + 1] if "--focal" in sys.argv else "HOMOSEX"
SAME_OBJECT = FOCAL_SETTINGS[FOCAL]["same_object"]
def _years(flag, default):
    """Parse --base/--late YYYY-YYYY into a list of years."""
    if flag not in sys.argv:
        return default
    start, end = sys.argv[sys.argv.index(flag) + 1].split("-")
    return list(range(int(start), int(end) + 1))


BASE_YEARS = _years("--base", list(range(1987, 1992)))    # before the shift
LATE_YEARS = _years("--late", list(range(2014, 2019)))    # before the 2021 mode switch
WINDOW_TAG = ("" if (BASE_YEARS, LATE_YEARS) == (list(range(1987, 1992)), list(range(2014, 2019)))
              else f"_{BASE_YEARS[0]}-{BASE_YEARS[-1]}_{LATE_YEARS[0]}-{LATE_YEARS[-1]}")
ESTIMATORS = ("partial", "raw")
CONDITIONS = ("A", "B", "C", "D", "E", "F")
GROUPS = {"A": ("lib", "con"), "B": ("lib", "con"), "C": ("lib", "con"),
          "D": ("college", "nocollege"), "E": ("lib", "con"), "F": ("predlib", "predcon")}
NOT_OUTCOMES = {"POLVIEWS", "PARTYID"}
ALPHA = 0.2
MIN_N = 100                 # minimum non-missing per group x window, and joint obs for raw r
COHORT_RANGE = {"B": (1925, 1969), "C": (1925, 1969), "D": (1925, 1965),
                "E": (1925, 1969), "F": (1925, 1962)}
REWEIGHT_COLS = {"C": ["COHORT_BAND", "DEGREE", "RACE", "SEX", "REGION"],
                 "D": ["COHORT_BAND", "RACE", "SEX", "REGION"],
                 "E": ["COHORT_BAND", "DEGREE", "RACE", "SEX", "REGION", "FUND", "PARTY3"],
                 "F": ["COHORT_BAND", "RACE", "SEX", "REGION"]}
BACKGROUND_COLS = ["COHORT_BAND", "RACE", "SEX", "REG16", "RES16", "RELIG16_GROUP", "FUND16",
                   "PADEG_CAT", "MADEG_CAT", "DEGREE"]
MIN_FOCAL_SHIFT = 0.10      # focal placebo: |shift| in pooled SDs, both groups
R_STRONG = 0.05             # strong form: |raw r| with the focal item in both groups
N_PERM = 12 if "--quick" in sys.argv else 300
N_BOOT = 12 if "--quick" in sys.argv else 300
N_JOBS = 12
SEED = 20261015


# ── Sample construction ─────────────────────────────────────────────


def load_sample():
    """Every respondent in the two windows, with demographics attached."""
    with contextlib.redirect_stdout(io.StringIO()):
        cleaned = clean_datasets()
    belief_vars = [c for c in cleaned.columns if c not in DataConfig.EXCLUDE_COLS]

    df = cleaned[cleaned["YEAR"].isin(BASE_YEARS + LATE_YEARS)]
    df = attach_demographics(
        df, ["COHORT", "DEGREE", "RACE", "SEX", "REGION", "BORN", "FUND",
             "REG16", "RES16", "RELIG16", "FUND16", "PADEG", "MADEG"],
        load_demographics())
    df["window"] = np.where(df["YEAR"].isin(BASE_YEARS), "base", "late")
    df["COHORT_BAND"] = (df["COHORT"] // 5) * 5
    df["PARTY3"] = np.sign(df["PARTYID"])                    # Dem -1, Ind 0, Rep +1
    df["RELIG16_GROUP"] = df["RELIG16"].clip(upper=5)         # Prot, Cath, Jewish, None, Other
    df["PADEG_CAT"] = df["PADEG"].fillna(9)                   # missing parent degree as a category
    df["MADEG_CAT"] = df["MADEG"].fillna(9)
    return df.reset_index(drop=True), belief_vars


def condition_sample(df, condition):
    """Rows and group labels for one condition."""
    g1, g2 = GROUPS[condition]
    if condition == "F":
        return predicted_camps(df)[0]
    if condition == "D":
        df = df[df["DEGREE"].notna() & (df["BORN"] == 1)].copy()
        df["group"] = np.where(df["DEGREE"] >= 3, g1, g2)
    else:
        df = df[df["POLVIEWS"].notna() & (df["POLVIEWS"] != 0)].copy()
        df["group"] = np.where(df["POLVIEWS"] < 0, g1, g2)
    if condition in COHORT_RANGE:
        df = df[df["COHORT"].between(*COHORT_RANGE[condition])]
    if condition in REWEIGHT_COLS:
        df = df[df[REWEIGHT_COLS[condition]].notna().all(axis=1)]
    return df.reset_index(drop=True)


def predicted_camps(df):
    """Condition F: groups assigned from background traits fixed before adulthood.

    Returns (sample, diagnostics). The model is fitted on baseline respondents who call
    themselves liberal or conservative, then applied to everyone in both windows.
    """
    g1, g2 = GROUPS["F"]
    df = df[(df["BORN"] == 1) & df["COHORT"].between(*COHORT_RANGE["F"])]
    df = df[df[BACKGROUND_COLS + REWEIGHT_COLS["F"]].notna().all(axis=1)].copy()
    X = pd.get_dummies(df[BACKGROUND_COLS].astype(int).astype(str), drop_first=True)
    train = (df["window"] == "base") & df["POLVIEWS"].notna() & (df["POLVIEWS"] != 0)
    y = (df.loc[train, "POLVIEWS"] < 0).astype(int)
    model = LogisticRegression(C=1.0, max_iter=5000).fit(X[train], y)
    df["p_lib"] = model.predict_proba(X)[:, 1]
    base_p = df.loc[df["window"] == "base", "p_lib"]
    lo, hi = base_p.quantile([1 / 3, 2 / 3])
    df = df[(df["p_lib"] <= lo) | (df["p_lib"] >= hi)].copy()
    df["group"] = np.where(df["p_lib"] >= hi, g1, g2)

    diag = {"auc": roc_auc_score(y, model.predict_proba(X[train])[:, 1])}
    for window in ("base", "late"):
        for group in (g1, g2):
            rows = df[(df["window"] == window) & (df["group"] == group) & df["POLVIEWS"].notna()]
            diag[window, group] = {
                "share_lib": float((rows["POLVIEWS"] < 0).mean()),
                "share_con": float((rows["POLVIEWS"] > 0).mean()),
            }
    return df.reset_index(drop=True), diag


def propensity_weights(sample, condition):
    """Weights that give each group's later window its baseline composition.

    Baseline rows get weight 1. Later rows get (1 - p) / p, normalised to mean 1 within
    group, where p is the probability of being in the later window given demographics.
    """
    weights = pd.Series(1.0, index=sample.index)
    if condition not in REWEIGHT_COLS:
        return weights
    cols = REWEIGHT_COLS[condition]
    for group in GROUPS[condition]:
        rows = sample[sample["group"] == group]
        X = pd.get_dummies(rows[cols].astype(int).astype(str), drop_first=True)
        y = (rows["window"] == "late").astype(int)
        model = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
        p = np.clip(model.predict_proba(X)[:, 1], 0.02, 0.98)
        w = (1 - p) / p
        late = (y == 1).values
        weights.loc[rows.index[late]] = w[late] / w[late].mean()
    return weights


# ── Core statistics ────────────────────────────────────────────────


def partial_network(base):
    """Regularized partial correlations, with a fallback for ill-conditioned input.

    Pairwise-complete correlation matrices need not be positive definite, and graphical
    LASSO then fails as ill-conditioned. On failure, the correlation matrix is projected to
    the nearest positive definite correlation matrix (eigenvalues floored at 1e-4, unit
    diagonal restored) and refitted. Returns (network, used_fallback).
    """
    net = calculate_correlation_matrix(
        base, method="pearson", partial=True,
        edge_suppression="regularization", suppression_params={"regularization": ALPHA},
    )
    if net is not None:
        return net, False
    corr = base.corr()
    clean, removed = filter_nans(corr.values)
    cols = [c for i, c in enumerate(corr.columns) if i not in removed]
    vals, vecs = np.linalg.eigh(clean)
    psd = (vecs * np.clip(vals, 1e-4, None)) @ vecs.T
    d = np.sqrt(np.diag(psd))
    psd = psd / np.outer(d, d)
    pcor = calculate_regularized_partial_correlations(psd, alpha=ALPHA)
    net = pd.DataFrame(pcor, index=cols, columns=cols)
    np.fill_diagonal(net.values, 0)
    return net, True



def weighted_means(frame, cols, w):
    x = frame[cols]
    present = x.notna()
    num = (x.fillna(0).mul(w, axis=0)).sum()
    den = present.mul(w, axis=0).sum()
    return num / den, present.sum()


def networks_and_shifts(sample, belief_vars, condition):
    """Baseline networks per group, and each group's standardised mean shift per item."""
    groups = GROUPS[condition]
    weights = propensity_weights(sample, condition)

    nets = {}
    for group in groups:
        base = sample[(sample["group"] == group) & (sample["window"] == "base")][belief_vars]
        nets[group, "partial"], nets[group, "fallback"] = partial_network(base)
        raw = base.corr(min_periods=MIN_N).fillna(0)
        np.fill_diagonal(raw.values, 0)
        nets[group, "raw"] = raw

    sd = sample[sample["window"] == "base"][belief_vars].std()
    shifts = {}
    for group in groups:
        rows = sample["group"] == group
        base = sample[rows & (sample["window"] == "base")]
        late = sample[rows & (sample["window"] == "late")]
        m_base, n_base = weighted_means(base, belief_vars, weights[base.index])
        m_late, n_late = weighted_means(late, belief_vars, weights[late.index])
        d = (m_late - m_base) / sd
        d[(n_base < MIN_N) | (n_late < MIN_N) | (sd == 0)] = np.nan
        shifts[group] = d
    return nets, shifts


def slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    return np.nan if denom == 0 else (xc * (y - y.mean())).sum() / denom


def corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def outcome_items(nets, shifts, groups, estimator, focal, exclude=None):
    g1, g2 = groups
    W1, W2 = nets[g1, estimator], nets[g2, estimator]
    if W1 is None or W2 is None:
        return []
    common = W1.index.intersection(W2.index)
    if focal not in common:
        return []
    exclude = NOT_OUTCOMES | (exclude or set())
    return [j for j in common
            if j != focal and j not in exclude
            and np.isfinite(shifts[g1].get(j, np.nan))
            and np.isfinite(shifts[g2].get(j, np.nan))]


def spillover(nets, shifts, groups, estimator, focal=FOCAL, exclude=None):
    """Own-network DiD slope and correlations for one focal item."""
    if exclude is None:
        exclude = SAME_OBJECT if focal == FOCAL else set()
    items = outcome_items(nets, shifts, groups, estimator, focal, exclude)
    out = {"beta": np.nan, "beta_swap": np.nan, "r_own": np.nan, "r_swap": np.nan,
           "rho_own": np.nan, "n_items": len(items)}
    if len(items) < 10 or not np.isfinite(shifts[groups[0]].get(focal, np.nan)) \
            or not np.isfinite(shifts[groups[1]].get(focal, np.nan)):
        return out
    g1, g2 = groups
    h1, h2 = shifts[g1][focal], shifts[g2][focal]
    w1 = nets[g1, estimator].loc[focal, items].values
    w2 = nets[g2, estimator].loc[focal, items].values
    y = shifts[g1][items].values - shifts[g2][items].values
    x_own = w1 * h1 - w2 * h2
    x_swap = w2 * h1 - w1 * h2
    out.update(beta=slope(x_own, y), beta_swap=slope(x_swap, y),
               r_own=corr(x_own, y), r_swap=corr(x_swap, y),
               rho_own=float(spearmanr(x_own, y)[0]))
    return out


def network_lag(nets, shifts, groups, estimator):
    """Exploratory: each item's group-differenced movement against its neighbours'."""
    g1, g2 = groups
    W1, W2 = nets[g1, estimator], nets[g2, estimator]
    if W1 is None or W2 is None:
        return np.nan
    items = [j for j in W1.index.intersection(W2.index)
             if j not in NOT_OUTCOMES
             and np.isfinite(shifts[g1].get(j, np.nan))
             and np.isfinite(shifts[g2].get(j, np.nan))]
    d1, d2 = shifts[g1][items], shifts[g2][items]
    x = W1.loc[items, items].values @ d1.values - W2.loc[items, items].values @ d2.values
    y = d1.values - d2.values
    return corr(x, y)


def replicate_stats(sample, belief_vars, condition):
    groups = GROUPS[condition]
    nets, shifts = networks_and_shifts(sample, belief_vars, condition)
    stats = {}
    for est in ESTIMATORS:
        s = spillover(nets, shifts, groups, est)
        s["lag_r"] = network_lag(nets, shifts, groups, est)
        stats[est] = s
    return stats


# ── Resampling ─────────────────────────────────────────────────────


def permuted(sample, seed):
    """Shuffle group labels within each window, keeping group sizes."""
    rng = np.random.default_rng(seed)
    out = sample.copy()
    for window in ("base", "late"):
        idx = out.index[out["window"] == window]
        out.loc[idx, "group"] = rng.permutation(out.loc[idx, "group"].values)
    return out


def bootstrapped(sample, seed):
    """Resample respondents with replacement within group x window."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in sample.groupby(["group", "window"]):
        parts.append(grp.iloc[rng.integers(0, len(grp), len(grp))])
    return pd.concat(parts).reset_index(drop=True)


def run_chunk(kind, sample, belief_vars, condition, seeds):
    """A batch of replicates, so each worker receives the sample once per batch."""
    out = []
    for seed in seeds:
        data = permuted(sample, seed) if kind == "perm" else bootstrapped(sample, seed)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out.append(replicate_stats(data, belief_vars, condition))
        except Exception:
            out.append(None)
    return out


def run_replicates(kind, sample, belief_vars, condition, seeds):
    chunks = np.array_split(np.asarray(seeds), N_JOBS * 2)
    batches = Parallel(n_jobs=N_JOBS)(
        delayed(run_chunk)(kind, sample, belief_vars, condition, [int(s) for s in chunk])
        for chunk in chunks if len(chunk))
    return [r for batch in batches for r in batch if r is not None]


def p_upper(null, value):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    return (1 + (null >= value).sum()) / (1 + len(null))


# ── Descriptives ───────────────────────────────────────────────────


def focal_neighbours(nets, groups, estimator, k=8):
    rows = []
    for group in groups:
        w = nets[group, estimator].loc[FOCAL]
        w = w.drop([j for j in NOT_OUTCOMES if j in w.index])
        top = w.reindex(w.abs().sort_values(ascending=False).index)[:k]
        rows.append(f"  {group}: " + ", ".join(f"{j} {v:+.2f}" for j, v in top.items()))
    return "\n".join(rows)


def strong_form(nets, shifts, groups):
    """Items whose raw correlation with the focal item has opposite signs in the two groups."""
    g1, g2 = groups
    items = outcome_items(nets, shifts, groups, "raw", FOCAL, SAME_OBJECT)
    r1 = nets[g1, "raw"].loc[FOCAL, items]
    r2 = nets[g2, "raw"].loc[FOCAL, items]
    rows = []
    for j in items:
        if abs(r1[j]) >= R_STRONG and abs(r2[j]) >= R_STRONG and np.sign(r1[j]) != np.sign(r2[j]):
            rows.append({
                "item": j, f"r_{g1}": r1[j], f"r_{g2}": r2[j],
                f"d_{g1}": shifts[g1][j], f"d_{g2}": shifts[g2][j],
                f"hit_{g1}": np.sign(shifts[g1][j]) == np.sign(r1[j] * shifts[g1][FOCAL]),
                f"hit_{g2}": np.sign(shifts[g2][j]) == np.sign(r2[j] * shifts[g2][FOCAL]),
            })
    return pd.DataFrame(rows)


def focal_placebo(nets, shifts, groups, estimator):
    """r_own for every item that shifted in both groups, used as the focal item."""
    g1, g2 = groups
    candidates = [k for k in outcome_items(nets, shifts, groups, estimator, FOCAL) + [FOCAL]
                  if abs(shifts[g1][k]) >= MIN_FOCAL_SHIFT and abs(shifts[g2][k]) >= MIN_FOCAL_SHIFT]
    return pd.Series({k: spillover(nets, shifts, groups, estimator, focal=k)["r_own"]
                      for k in candidates}).dropna()


# ── Main ───────────────────────────────────────────────────────────


def main():
    t0 = time.time()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df, belief_vars = load_sample()

    print("=" * 70)
    print(f"SPILL 01: {FOCAL} SPILLOVER PILOT")
    print("=" * 70)
    print(f"Windows: base {BASE_YEARS[0]}-{BASE_YEARS[-1]}, late {LATE_YEARS[0]}-{LATE_YEARS[-1]}; "
          f"{N_PERM} permutations, {N_BOOT} bootstraps per condition")

    results = {"conditions": {}}
    for condition in CONDITIONS:
        groups = GROUPS[condition]
        sample = condition_sample(df, condition)
        counts = sample.groupby(["group", "window"]).size().unstack()
        print(f"\n── Condition {condition}: {groups[0]} vs {groups[1]} ──")
        print(counts.to_string())
        if condition in REWEIGHT_COLS:
            w = propensity_weights(sample, condition)
            ess = []
            for g in groups:
                wl = w[(sample["group"] == g) & (sample["window"] == "late")]
                ess.append(f"{g} {wl.sum() ** 2 / (wl ** 2).sum():.0f} of {len(wl)}")
            print("Late-window effective sample size after reweighting: " + "; ".join(ess))
        if condition == "F":
            diag = predicted_camps(df)[1]
            print(f"Background model AUC (baseline, lib vs con): {diag['auc']:.3f}")
            for window in ("base", "late"):
                shares = "; ".join(
                    f"{g} {diag[window, g]['share_lib']:.0%} lib / {diag[window, g]['share_con']:.0%} con"
                    for g in groups)
                print(f"  {window}: {shares}")

        with contextlib.redirect_stdout(io.StringIO()):
            nets, shifts = networks_and_shifts(sample, belief_vars, condition)
        observed = {}
        for est in ESTIMATORS:
            observed[est] = spillover(nets, shifts, groups, est)
            observed[est]["lag_r"] = network_lag(nets, shifts, groups, est)
            observed[est]["r_own_incl_same_object"] = spillover(
                nets, shifts, groups, est, exclude=set())["r_own"]
        print(f"{FOCAL} shift (pooled SDs): {groups[0]} {shifts[groups[0]][FOCAL]:+.3f}, "
              f"{groups[1]} {shifts[groups[1]][FOCAL]:+.3f}")
        fell_back = [g for g in groups if nets[g, "fallback"]]
        if fell_back:
            print(f"Partial network needed the positive-definite fallback for: {', '.join(fell_back)}")

        seeds = np.random.default_rng(SEED + ord(condition)).integers(0, 2**31, N_PERM + N_BOOT)
        perms = run_replicates("perm", sample, belief_vars, condition, seeds[:N_PERM])
        boots = run_replicates("boot", sample, belief_vars, condition, seeds[N_PERM:])

        summary = {}
        for est in ESTIMATORS:
            obs = observed[est]
            gap = obs["r_own"] - obs["r_swap"]
            null_beta = [p[est]["beta"] for p in perms]
            summary[est] = {
                **obs,
                "gap": gap,
                "p_beta": p_upper(null_beta, obs["beta"]),
                "p_gap": p_upper([p[est]["r_own"] - p[est]["r_swap"] for p in perms], gap),
                "p_lag": p_upper([p[est]["lag_r"] for p in perms], obs["lag_r"]),
                "beta_ci": np.nanpercentile([b[est]["beta"] for b in boots], [2.5, 97.5]),
                "gap_ci": np.nanpercentile([b[est]["r_own"] - b[est]["r_swap"] for b in boots],
                                           [2.5, 97.5]),
                "null_beta": np.asarray(null_beta, float),
                "n_perm": len(perms), "n_boot": len(boots),
            }
            s = summary[est]
            print(f"  [{est:7s}] items={s['n_items']:3d}  beta={s['beta']:+.3f} "
                  f"bootCI[{s['beta_ci'][0]:+.3f},{s['beta_ci'][1]:+.3f}] p_perm={s['p_beta']:.3f}")
            print(f"            r_own={s['r_own']:+.3f} (spearman {s['rho_own']:+.3f}; "
                  f"with same-object items {s['r_own_incl_same_object']:+.3f})  "
                  f"r_swap={s['r_swap']:+.3f}  gap={gap:+.3f} "
                  f"bootCI[{s['gap_ci'][0]:+.3f},{s['gap_ci'][1]:+.3f}] p_perm={s['p_gap']:.3f}")
            print(f"            network lag r={s['lag_r']:+.3f} p_perm={s['p_lag']:.3f}")

        results["conditions"][condition] = {
            "summary": summary, "counts": counts, "shifts": shifts, "nets": nets,
            "groups": groups,
        }

        if condition in ("B", "D", "F"):
            print(f"{FOCAL} neighbours, baseline ({condition}, partial):")
            print(focal_neighbours(nets, groups, "partial"))
            print(f"{FOCAL} neighbours, baseline ({condition}, raw):")
            print(focal_neighbours(nets, groups, "raw"))
            sf = strong_form(nets, shifts, groups)
            results["conditions"][condition]["strong_form"] = sf
            print(f"Strong form ({condition}): {len(sf)} items with opposite-sign raw edges "
                  f"(|r| >= {R_STRONG} in both groups)")
            if len(sf):
                print(sf.round(3).to_string(index=False))
            for est in ESTIMATORS:
                fp = focal_placebo(nets, shifts, groups, est)
                rank = int((fp >= fp.get(FOCAL, np.nan)).sum())
                results["conditions"][condition][f"focal_placebo_{est}"] = fp
                print(f"Focal placebo ({est}): {FOCAL} r_own={fp.get(FOCAL, np.nan):+.3f}, "
                      f"rank {rank} of {len(fp)} items shifting >= {MIN_FOCAL_SHIFT} SD in both "
                      f"groups; median r_own={fp.median():+.3f}")

    # ── Go / no-go, as fixed in the docstring ──
    b = results["conditions"]["B"]["summary"]
    go = [est for est in ESTIMATORS
          if b[est]["beta"] > 0 and b[est]["p_beta"] < 0.05 and b[est]["r_own"] > b[est]["r_swap"]]
    print("\n" + "=" * 70)
    print(f"GO CRITERION (condition B): {'GO via ' + ', '.join(go) if go else 'NO-GO'}")
    print("=" * 70)
    results["go"] = go

    strict = []
    for est in go:
        fp = results["conditions"]["B"][f"focal_placebo_{est}"]
        rank = int((fp >= fp.get(FOCAL, np.nan)).sum())
        d = results["conditions"]["D"]["summary"][est]
        if rank <= np.ceil(len(fp) / 4) and d["beta"] > 0 and d["p_beta"] < 0.05:
            strict.append(est)
    f = results["conditions"]["F"]["summary"]
    survives = [est for est in ESTIMATORS
                if f[est]["beta"] > 0 and f[est]["p_beta"] < 0.05 and f[est]["r_own"] > f[est]["r_swap"]]
    strict_text = "GO via " + ", ".join(strict) if strict else "NO-GO"
    sorting_text = "SURVIVES via " + ", ".join(survives) if survives else "DOES NOT SURVIVE"
    print(f"STRICT CRITERION (B + top-quartile focal rank + D): {strict_text}")
    print(f"SORTING VERDICT (condition F): {sorting_text}")
    print("=" * 70)
    results["strict_go"] = strict
    results["survives_sorting"] = survives

    make_figure(results)
    out_pkl = CACHE_DIR / f"spill_01_{FOCAL.lower()}{WINDOW_TAG}_results.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {out_pkl}")
    print(f"Elapsed {time.time() - t0:.0f}s")


def make_figure(results):
    with contextlib.redirect_stdout(io.StringIO()):
        cleaned = clean_datasets()
    full = cleaned[cleaned["POLVIEWS"].notna() & (cleaned["POLVIEWS"] != 0) & cleaned[FOCAL].notna()]
    full = full.assign(camp=np.where(full["POLVIEWS"] < 0, "lib", "con"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    for camp, colour in (("lib", "#3b6fb6"), ("con", "#c0392b")):
        series = full[full["camp"] == camp].groupby("YEAR")[FOCAL].mean()
        ax.plot(series.index, series.values, marker="o", ms=3, color=colour, label=camp)
    for years in (BASE_YEARS, LATE_YEARS):
        ax.axvspan(years[0] - 0.5, years[-1] + 0.5, color="grey", alpha=0.15)
    ax.set_title(f"A. {FOCAL} by camp ({FOCAL_SETTINGS[FOCAL]['direction']})", fontweight="bold")
    ax.set_xlabel("Year")
    ax.legend()

    ax = axes[0, 1]
    res = results["conditions"]["B"]
    nets, shifts, (g1, g2) = res["nets"], res["shifts"], res["groups"]
    items = outcome_items(nets, shifts, (g1, g2), "raw", FOCAL, SAME_OBJECT)
    x = (nets[g1, "raw"].loc[FOCAL, items].values * shifts[g1][FOCAL]
         - nets[g2, "raw"].loc[FOCAL, items].values * shifts[g2][FOCAL])
    y = shifts[g1][items].values - shifts[g2][items].values
    ax.scatter(x, y, s=14, alpha=0.7, color="#555555")
    k = slope(x, y)
    xs = np.linspace(x.min(), x.max(), 10)
    ax.plot(xs, y.mean() + k * (xs - x.mean()), color="black")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("Predicted: w_lib·ΔH_lib − w_con·ΔH_con (raw r)")
    ax.set_ylabel("Observed: Δ_lib − Δ_con (pooled SDs)")
    ax.set_title("B. Camp-differenced movement vs own-network prediction (B)", fontweight="bold")

    ax = axes[1, 0]
    labels, r_own, r_swap, null95 = [], [], [], []
    for condition in CONDITIONS:
        for est in ESTIMATORS:
            s = results["conditions"][condition]["summary"][est]
            labels.append(f"{condition}\n{est}")
            r_own.append(s["r_own"])
            r_swap.append(s["r_swap"])
    pos = np.arange(len(labels))
    ax.bar(pos - 0.2, r_own, width=0.4, color="#333333", label="r_own")
    ax.bar(pos + 0.2, r_swap, width=0.4, color="#bbbbbb", label="r_swap (other group's edges)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("correlation with group-differenced movement")
    ax.set_title("C. Own vs other group's network, by condition", fontweight="bold")
    ax.legend()

    ax = axes[1, 1]
    fp = results["conditions"]["B"]["focal_placebo_partial"]
    ax.hist(fp.values, bins=25, color="#999999")
    if FOCAL in fp:
        ax.axvline(fp[FOCAL], color="#c0392b", lw=2, label=FOCAL)
    ax.set_xlabel("r_own with item as focal (partial, B)")
    ax.set_title(f"D. Focal placebo: is {FOCAL} special?", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    out_fig = FIGURES_DIR / f"spill_01_{FOCAL.lower()}{WINDOW_TAG}_pilot.png"
    plt.savefig(out_fig, dpi=150)
    print(f"\nSaved {out_fig}")


if __name__ == "__main__":
    main()
