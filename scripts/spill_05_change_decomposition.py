"""
Spill 05: What moves a camp — individuals changing, or who is in it? (exploratory)

The panel test (spill_04) found within-person change on HOMOSEX tiny over four years, while
the cross-sectional camps moved much more over the same years. This splits each camp's
four-year change in a focal item, for the three GSS panel windows, into:

  total          cross-sectional camp mean, year t+4 minus year t (cumulative file)
  within-cohort  the same, restricted to cohorts adult in year t, so cohort replacement
                 is removed; what remains is individual change plus re-sorting between
                 camps (and differential mortality)
  individual     the panel's own respondents, camp fixed by the mean of their wave-1 and
                 wave-3 POLVIEWS (spill_04's mean13 rule), wave 3 minus wave 1

so that  replacement = total - within-cohort  and  re-sorting ~ within-cohort - individual.
Raw [-1, 1] units, unweighted. Exploratory, specified after the panel results.

Usage: PYTHONIOENCODING=utf-8 python scripts/spill_05_change_decomposition.py
Outputs: stdout
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import contextlib

import numpy as np
import pandas as pd
import pyreadstat as prs

from src.loaders.clean_raw_data import clean_datasets, DataConfig, transform_column
from src.loaders.demographics import load_demographics, attach_demographics

PANEL_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "gss_panels"
PANELS = {
    2006: "GSS_2006_Panel_Stata/GSS_panel06w123_R6a - Stata.dta",
    2008: "GSS_2008_Panel_stata/GSS_panel08w123_R6 - stata.dta",
    2010: "GSS_2010_Panel_stata/GSS_panel2010w123_R6 - stata.dta",
}
FOCALS = ("HOMOSEX", "GRASS", "CAPPUN")


def camp_of(polviews):
    return np.where(polviews < 0, "lib", np.where(polviews > 0, "con", "mod"))


def panel_change(year, focal, mappings):
    path = PANEL_DIR / PANELS[year]
    cols = [f"{focal.lower()}_1", f"{focal.lower()}_3", "polviews_1", "polviews_3", "year_3"]
    raw, _ = prs.read_dta(str(path), usecols=cols, user_missing=True)
    up = lambda s: s.map(lambda v: v.upper() if isinstance(v, str) else v)
    raw = raw[pd.to_numeric(raw["year_3"], errors="coerce").notna()]
    f1 = transform_column(up(raw[cols[0]]), mappings[focal]).astype(float)
    f3 = transform_column(up(raw[cols[1]]), mappings[focal]).astype(float)
    p1 = transform_column(up(raw["polviews_1"]), mappings["POLVIEWS"]).astype(float)
    p3 = transform_column(up(raw["polviews_3"]), mappings["POLVIEWS"]).astype(float)
    camp = pd.Series(camp_of((p1 + p3) / 2), index=raw.index)
    change = f3 - f1
    return {c: change[camp == c].mean() for c in ("lib", "con")}


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        cleaned = clean_datasets()
    df = attach_demographics(cleaned, ["COHORT"], load_demographics())
    df = df[df["POLVIEWS"].notna()].copy()
    df["camp"] = camp_of(df["POLVIEWS"])
    mappings = DataConfig().all_mappings

    print("=" * 78)
    print("SPILL 05: DECOMPOSING FOUR-YEAR CAMP CHANGE (exploratory, raw [-1, 1] units)")
    print("=" * 78)
    for focal in FOCALS:
        print(f"\n── {focal} ──")
        print(f"{'window':11s} {'camp':4s} {'total':>8s} {'within-cohort':>14s} {'individual':>11s} "
              f"{'replacement':>12s} {'re-sorting':>11s}")
        for year in PANELS:
            indiv = panel_change(year, focal, mappings)
            for c in ("lib", "con"):
                t0 = df[(df["YEAR"] == year) & (df["camp"] == c)]
                t1 = df[(df["YEAR"] == year + 4) & (df["camp"] == c)]
                total = t1[focal].mean() - t0[focal].mean()
                adult_by = year - 18
                t1c = t1[t1["COHORT"] <= adult_by]
                t0c = t0[t0["COHORT"] <= adult_by]
                within = t1c[focal].mean() - t0c[focal].mean()
                print(f"{year}-{year + 4:<6d} {c:4s} {total:+8.3f} {within:+14.3f} {indiv[c]:+11.3f} "
                      f"{total - within:+12.3f} {within - indiv[c]:+11.3f}")


if __name__ == "__main__":
    main()
