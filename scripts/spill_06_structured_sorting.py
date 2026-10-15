"""
Spill 06: Structure-guided sorting — do people leave a camp when their belief pattern fits
the other camp's network?

The spillover work (spill_01-05) found that camps change along their own belief structure,
and that for liberals a large share of that change is re-sorting rather than individuals
changing their minds. This tests the sorting mechanism directly in the GSS 2006-14 panels.

Fixed before the first run (tracker, 2026-10-15):
  - people: wave-1 liberals and conservatives (POLVIEWS != 0) in the 2006, 2008 and 2010
    panels, with POLVIEWS also observed at wave 3
  - outcome: left their wave-1 camp by wave 3 (liberal -> moderate or conservative, or the
    reverse)
  - camp networks: from the GSS cross-sections 2004-2014 with the panel's own wave-1
    respondents removed, so nobody is scored against a network their answers shaped;
    POLVIEWS and PARTYID excluded from the networks and from the fit
  - fit of a person to a network W:  sum_{j<k} w_jk z_j z_k / sum_{j<k} |w_jk|  over the pairs
    they answered, z standardised on the cross-section. It is unchanged if every answer flips
    sign, so it scores which beliefs go together (arrangement), not where the person stands
    (position)
  - predictor: relative fit = fit to the other camp's network minus fit to their own
  - controls: mean fit to the two networks; issue position towards the other camp;
    cross-pressure (share of answered items on the other camp's side of the midpoint, among
    items where the camps differ by at least 0.1 SD); ideological strength |POLVIEWS_1|;
    camp; panel; age; degree; sex. Logit with HC1 errors
  - verdict: relative fit > 0 with p < 0.05 in the full model, for at least one estimator
    (regularized partial correlations primary, raw correlations secondary)

Replication: `--panel 2020` runs the same test on the 2016-20 panel — 2016 and 2018
respondents re-interviewed in 2020 — with networks from the 2012-18 cross-sections (these
respondents removed; 2020 left out because of its mode change) and two extra controls,
origin year and web vs phone in 2020, since the 2020 wave had no in-person interviews.
Specified before running (tracker, 2026-10-15).

Supplementary: a permutation null that shuffles camp labels in the cross-section before the
networks are estimated, so relative fit is computed against networks that differ only by
noise.

Usage: PYTHONIOENCODING=utf-8 python scripts/spill_06_structured_sorting.py [--panel 2020] [--quick]
Outputs: data/cache/spill_06_results<_2020>.pkl, stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import contextlib
import pickle
import time

import numpy as np
import pandas as pd
import pyreadstat as prs
import statsmodels.api as sm
from joblib import Parallel, delayed

from src.loaders.clean_raw_data import clean_datasets, DataConfig, transform_column
from src.generators.corr_make_network import (
    calculate_correlation_matrix, calculate_regularized_partial_correlations, filter_nans,
)

ROOT = Path(__file__).resolve().parent.parent
PANEL_DIR = ROOT / "data" / "raw" / "gss_panels"
CACHE_DIR = ROOT / "data" / "cache"
PANEL_FILES = {
    2006: "GSS_2006_Panel_Stata/GSS_panel06w123_R6a - Stata.dta",
    2008: "GSS_2008_Panel_stata/GSS_panel08w123_R6 - stata.dta",
    2010: "GSS_2010_Panel_stata/GSS_panel2010w123_R6 - stata.dta",
}
PANEL_2020 = "GSS_2020_panel_stata_1a/gss2020panel_r1a.dta"
PANEL_SET = sys.argv[sys.argv.index("--panel") + 1] if "--panel" in sys.argv else "2006-14"
XS_YEARS = list(range(2012, 2019)) if PANEL_SET == "2020" else list(range(2004, 2015))
TAG = "_2020" if PANEL_SET == "2020" else ""
NOT_IN_FIT = {"POLVIEWS", "PARTYID"}
ESTIMATORS = ("partial", "raw")
ALPHA = 0.2
MIN_PAIR_N = 100
MIN_SIDE_GAP = 0.10
N_PERM = 12 if "--quick" in sys.argv else 200
N_JOBS = 12
SEED = 20261018


# ── Data ───────────────────────────────────────────────────────────


def upper(series):
    return series.map(lambda v: v.upper() if isinstance(v, str) else v)


def load_panel_2020(items):
    """2016 and 2018 respondents re-interviewed in 2020, in the same layout as load_panel_wave1."""
    mappings = DataConfig().all_mappings
    path = PANEL_DIR / PANEL_2020
    _, meta = prs.read_dta(str(path), metadataonly=True, encoding="latin1")
    lower = {c.lower(): c for c in meta.column_names}
    present = [i for i in items if f"{i.lower()}_1a" in lower and f"{i.lower()}_1b" in lower]
    wave1 = [i.lower() for i in present] + ["id", "year", "age", "sex", "degree", "polviews"]
    cols = [lower[f"{v}_{w}"] for v in wave1 for w in ("1a", "1b")]
    cols += [lower["polviews_2"], lower["mode_2"], lower["year_2"], lower["samptype"]]
    raw, _ = prs.read_dta(str(path), usecols=cols, user_missing=True, encoding="latin1")
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw[pd.to_numeric(raw["year_2"], errors="coerce").notna()].reset_index(drop=True)
    from_2016 = raw["samptype"] == 2016

    def first_wave(v):
        return raw[f"{v}_1a"].where(from_2016, raw[f"{v}_1b"])

    out = pd.DataFrame({i: transform_column(upper(first_wave(i.lower())), mappings[i]) for i in present})
    out["POL1"] = transform_column(upper(first_wave("polviews")), mappings["POLVIEWS"])
    out["POL3"] = transform_column(upper(raw["polviews_2"]), mappings["POLVIEWS"])
    for v in ("id", "year", "age", "sex", "degree"):
        out[f"{v.upper()}_1"] = pd.to_numeric(first_wave(v), errors="coerce")
    out["PANEL"] = raw["samptype"].astype(int)
    out["MODE_WEB"] = (pd.to_numeric(raw["mode_2"], errors="coerce") == 4).astype(float)
    return out


def load_panel_wave1(items):
    if PANEL_SET == "2020":
        return finish_panel(load_panel_2020(items), items)
    mappings = DataConfig().all_mappings
    parts = []
    for panel, rel in PANEL_FILES.items():
        path = PANEL_DIR / rel
        _, meta = prs.read_dta(str(path), metadataonly=True)
        lower = {c.lower(): c for c in meta.column_names}
        present = [i for i in items if f"{i.lower()}_1" in lower]
        extra = ["id_1", "year_1", "age_1", "sex_1", "degree_1", "polviews_1", "polviews_3"]
        raw, _ = prs.read_dta(str(path), usecols=[lower[f"{i.lower()}_1"] for i in present]
                              + [lower[e] for e in extra], user_missing=True)
        raw.columns = [c.lower() for c in raw.columns]
        out = pd.DataFrame({i: transform_column(upper(raw[f"{i.lower()}_1"]), mappings[i]) for i in present})
        out["POL1"] = transform_column(upper(raw["polviews_1"]), mappings["POLVIEWS"])
        out["POL3"] = transform_column(upper(raw["polviews_3"]), mappings["POLVIEWS"])
        for e in ("id_1", "year_1", "age_1", "sex_1", "degree_1"):
            out[e.upper()] = pd.to_numeric(raw[e], errors="coerce")
        out["PANEL"] = panel
        parts.append(out)
    return finish_panel(pd.concat(parts, ignore_index=True), items)


def finish_panel(df, items):
    """Shared tail: complete the item columns, define camp and the switching outcome."""
    for i in items:
        if i not in df:
            df[i] = np.nan
    df[items] = df[items].astype(float)
    df = df[df["POL1"].notna() & (df["POL1"] != 0) & df["POL3"].notna()].reset_index(drop=True)
    df["camp"] = np.where(df["POL1"] < 0, "lib", "con")
    df["switch"] = np.where(df["camp"] == "lib", df["POL3"] >= 0, df["POL3"] <= 0).astype(int)
    return df


def load_cross_section(items, exclude):
    with contextlib.redirect_stdout(io.StringIO()):
        cleaned = clean_datasets()
    xs = cleaned[cleaned["YEAR"].isin(XS_YEARS)]
    keys = pd.MultiIndex.from_frame(xs[["YEAR", "ID"]].astype(float))
    xs = xs[~keys.isin(exclude)]
    xs = xs[xs["POLVIEWS"].notna() & (xs["POLVIEWS"] != 0)]
    camp = pd.Series(np.where(xs["POLVIEWS"] < 0, "lib", "con"), index=xs.index)
    return xs[items].astype(float), camp


# ── Networks and person scores ─────────────────────────────────────


def partial_network(frame):
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


def camp_networks(xs, camp, items):
    nets = {}
    for g in ("lib", "con"):
        frame = xs[camp.values == g]
        nets[g, "partial"] = partial_network(frame).reindex(index=items, columns=items).fillna(0)
        raw = frame.corr(min_periods=MIN_PAIR_N).reindex(index=items, columns=items).fillna(0)
        np.fill_diagonal(raw.values, 0)
        nets[g, "raw"] = raw
    return nets


def fit_scores(Z0, M, W):
    """Fit of each person to network W over the pairs they answered."""
    Wv = W.values
    num = 0.5 * ((Z0 @ Wv) * Z0).sum(axis=1)
    den = 0.5 * ((M @ np.abs(Wv)) * M).sum(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def person_predictors(panel, xs, camp, items, nets):
    mu = xs.mean()
    sd = xs.std()
    Z = (panel[items] - mu) / sd
    M = Z.notna().astype(float).values
    Z0 = Z.fillna(0).values

    lib_z = ((xs[camp.values == "lib"].mean() - mu) / sd).values
    con_z = ((xs[camp.values == "con"].mean() - mu) / sd).values
    d = np.nan_to_num(lib_z - con_z)                       # liberal minus conservative
    position_lib = (Z0 @ d) / np.maximum(M @ np.abs(d), 1e-9)
    mid = np.nan_to_num((lib_z + con_z) / 2)
    differs = np.abs(d) >= MIN_SIDE_GAP
    on_lib_side = ((Z0 - mid) * d > 0) & (M > 0) & differs
    on_con_side = ((Z0 - mid) * d < 0) & (M > 0) & differs
    answered = ((M > 0) & differs).sum(axis=1)
    is_lib = (panel["camp"] == "lib").values

    out = pd.DataFrame(index=panel.index)
    out["pos_other"] = np.where(is_lib, -position_lib, position_lib)
    out["cross"] = np.where(is_lib, on_con_side.sum(axis=1), on_lib_side.sum(axis=1)) / np.maximum(answered, 1)
    for est in ESTIMATORS:
        f_lib = fit_scores(Z0, M, nets["lib", est])
        f_con = fit_scores(Z0, M, nets["con", est])
        own = np.where(is_lib, f_lib, f_con)
        other = np.where(is_lib, f_con, f_lib)
        out[f"rel_fit_{est}"] = other - own
        out[f"mean_fit_{est}"] = (other + own) / 2
    return out


# ── Models ─────────────────────────────────────────────────────────


def zscore(x):
    return (x - x.mean()) / x.std()


def design(panel, pred, est, controls=True):
    X = pd.DataFrame({"rel_fit": zscore(pred[f"rel_fit_{est}"])}, index=panel.index)
    X["con"] = (panel["camp"] == "con").astype(float)
    if controls:
        X["mean_fit"] = zscore(pred[f"mean_fit_{est}"])
        X["pos_other"] = zscore(pred["pos_other"])
        X["cross"] = zscore(pred["cross"])
        X["strength"] = zscore(panel["POL1"].abs())
        X["age"] = zscore(panel["AGE_1"])
        X["degree"] = panel["DEGREE_1"]
        X["female"] = (panel["SEX_1"] == 2).astype(float)
        for pv in sorted(panel["PANEL"].unique())[1:]:
            X[f"p{pv}"] = (panel["PANEL"] == pv).astype(float)
        if "MODE_WEB" in panel:
            X["web"] = panel["MODE_WEB"]
    return sm.add_constant(X)


def fit_logit(panel, pred, est, controls=True):
    X = design(panel, pred, est, controls)
    ok = X.notna().all(axis=1)
    return sm.Logit(panel.loc[ok, "switch"], X[ok]).fit(disp=0, cov_type="HC1")


def permuted_coef(panel, xs, camp, items, seed):
    """Relative-fit coefficient against networks estimated on shuffled camp labels.

    Only the networks are permuted; position and cross-pressure stay defined against the
    real camps, since they describe the person rather than the networks.
    """
    rng = np.random.default_rng(seed)
    fake = pd.Series(rng.permutation(camp.values), index=camp.index)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            nets = camp_networks(xs, fake, items)
            pred = person_predictors(panel, xs, camp, items, nets)
            return {est: float(fit_logit(panel, pred, est).params["rel_fit"]) for est in ESTIMATORS}
    except Exception:
        return None


# ── Main ───────────────────────────────────────────────────────────


def main():
    t0 = time.time()
    items = sorted(set(DataConfig().all_questions) - NOT_IN_FIT)
    panel = load_panel_wave1(items)
    exclude = pd.MultiIndex.from_frame(panel[["YEAR_1", "ID_1"]].astype(float))
    xs, camp = load_cross_section(items, exclude)
    items = [i for i in items if panel[i].notna().sum() >= MIN_PAIR_N and xs[i].notna().sum() >= MIN_PAIR_N]
    xs = xs[items]

    print("=" * 70)
    print(f"SPILL 06: STRUCTURE-GUIDED SORTING (GSS {PANEL_SET} panel{'s' if PANEL_SET != '2020' else ''})")
    print("=" * 70)
    print(f"Cross-section networks: {XS_YEARS[0]}-{XS_YEARS[-1]}, {len(xs)} respondents "
          f"(panel wave-1 respondents removed), {len(items)} items")
    print(pd.crosstab(panel["camp"], panel["switch"], margins=True)
          .rename(columns={0: "stayed", 1: "left camp"}).to_string())

    with contextlib.redirect_stdout(io.StringIO()):
        nets = camp_networks(xs, camp, items)
    pred = person_predictors(panel, xs, camp, items, nets)

    print("\nCorrelations of relative fit with the position-based predictors:")
    for est in ESTIMATORS:
        r = pred[[f"rel_fit_{est}", "pos_other", "cross"]].corr().iloc[0, 1:]
        print(f"  {est}: with position towards other camp {r['pos_other']:+.2f}, "
              f"with cross-pressure {r['cross']:+.2f}")

    results = {}
    for est in ESTIMATORS:
        simple = fit_logit(panel, pred, est, controls=False)
        full = fit_logit(panel, pred, est, controls=True)
        results[est] = {"simple": simple, "full": full}
        print(f"\n[{est}]  odds ratio per SD of relative fit (p)")
        print(f"  camp only:     OR {np.exp(simple.params['rel_fit']):.3f} (p={simple.pvalues['rel_fit']:.4f}), n={int(simple.nobs)}")
        print(f"  full controls: OR {np.exp(full.params['rel_fit']):.3f} (p={full.pvalues['rel_fit']:.4f}), n={int(full.nobs)}")
        for v in ("mean_fit", "pos_other", "cross", "strength"):
            print(f"      {v:9s} OR {np.exp(full.params[v]):.3f} (p={full.pvalues[v]:.4f})")

    seeds = np.random.default_rng(SEED).integers(0, 2**31, N_PERM)
    perms = Parallel(n_jobs=N_JOBS)(
        delayed(permuted_coef)(panel, xs, camp, items, int(s)) for s in seeds)
    perms = [p for p in perms if p is not None]
    print(f"\nPermutation null (camp labels shuffled before networks), {len(perms)} draws:")
    for est in ESTIMATORS:
        obs = results[est]["full"].params["rel_fit"]
        null = np.array([p[est] for p in perms])
        p_perm = (1 + (null >= obs).sum()) / (1 + len(null))
        results[est]["p_perm"] = p_perm
        print(f"  {est}: observed coef {obs:+.3f}; null mean {null.mean():+.3f}, "
              f"95th pct {np.percentile(null, 95):+.3f}; p_perm={p_perm:.3f}")

    supported = [est for est in ESTIMATORS
                 if results[est]["full"].params["rel_fit"] > 0 and results[est]["full"].pvalues["rel_fit"] < 0.05]
    print("\n" + "=" * 70)
    print(f"VERDICT: {'SUPPORTED via ' + ', '.join(supported) if supported else 'NOT SUPPORTED'}")
    print("=" * 70)

    with open(CACHE_DIR / f"spill_06_results{TAG}.pkl", "wb") as fh:
        pickle.dump({"pred": pred, "panel": panel[["camp", "switch", "POL1", "POL3", "PANEL"]],
                     "coefs": {e: results[e]["full"].params.to_dict() for e in ESTIMATORS},
                     "pvalues": {e: results[e]["full"].pvalues.to_dict() for e in ESTIMATORS},
                     "p_perm": {e: results[e]["p_perm"] for e in ESTIMATORS},
                     "supported": supported}, fh)
    print(f"Saved data/cache/spill_06_results{TAG}.pkl; elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
