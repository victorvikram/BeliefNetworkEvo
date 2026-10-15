"""
GSS Demographics, Weights and Design Variables
----------------------------------------------

Loads respondent demographics, survey weights and sampling-design variables from the
raw GSS file, keyed on YEAR and ID.

These are kept out of `clean_datasets()` on purpose. Scripts treat every column of the
cleaned frame outside `DataConfig.EXCLUDE_COLS` as a belief variable, so adding
demographics there would put AGE, INCOME and the weights into every network. Attach
them explicitly with `attach_demographics()` instead.

Values are the raw GSS codes, coerced to numeric. GSS special missing codes (the
letters pyreadstat returns for .d, .i, .n and so on) become NaN, as do the numeric
missing codes listed in `NUMERIC_MISSING`. Nothing is rescaled.
"""

import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyreadstat as prs

KEY_COLS = ["YEAR", "ID"]

DEMOGRAPHIC_COLS = [
    "AGE",       # topcoded at 89
    "COHORT",    # year of birth
    "SEX",
    "RACE",
    "HISPANIC",
    "BORN",      # born in this country
    "EDUC",      # years of schooling
    "DEGREE",
    "MARITAL",
    "REGION",
    "SRCBELT",   # urbanicity
    "XNORCSIZ",
    "REALINC",   # family income, constant dollars
    "CONINC",
    "FUND",      # religious fundamentalism of denomination
]

DESIGN_COLS = [
    "WTSSALL",   # standard weight, 1972-2018
    "WTSSPS",    # post-stratification weight, all years
    "WTSSNRPS",  # post-stratification weight, nonrespondents adjusted
    "FORMWT",
    "OVERSAMP",
    "PHASE",
    "SAMPLE",
    "SAMPCODE",
    "VSTRAT",    # variance stratum
    "VPSU",      # variance primary sampling unit
    "MODE",      # in-person or phone
    "SPANENG",   # interview language
]

ALL_COLS = DEMOGRAPHIC_COLS + DESIGN_COLS

# Missing codes stored as ordinary numbers rather than SAS special missings
NUMERIC_MISSING = {"COHORT": [9999]}

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_RAW_FILE = _DATA_DIR / "raw" / "gss7222_r4.sas7bdat"
_CACHE_FILE = _DATA_DIR / "cache" / "demographics.pkl"


def load_demographics() -> pd.DataFrame:
    """
    Load demographics, weights and design variables for every respondent.

    Reads from `data/cache/demographics.pkl` when it holds every column in
    `ALL_COLS`, otherwise re-reads the raw SAS file and rebuilds the cache.

    Returns:
        DataFrame with YEAR, ID and the columns in ALL_COLS, one row per respondent.
    """
    if _CACHE_FILE.exists():
        with open(_CACHE_FILE, "rb") as f:
            df = pickle.load(f)
        if set(KEY_COLS + ALL_COLS) <= set(df.columns):
            return df[KEY_COLS + ALL_COLS]

    if not _RAW_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found at: {_RAW_FILE}\n"
            f"Please download the GSS data file (gss7222_r4.sas7bdat) and place it in: {_RAW_FILE.parent}"
        )

    print("Loading demographics from raw SAS file (slow, cached afterwards)...")
    df, _ = prs.read_sas7bdat(
        str(_RAW_FILE),
        usecols=KEY_COLS + ALL_COLS,
        user_missing=True,
        disable_datetime_conversion=True,
        formats_as_category=False,
    )
    for col in ALL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, codes in NUMERIC_MISSING.items():
        df[col] = df[col].mask(df[col].isin(codes))

    if df.duplicated(subset=KEY_COLS).any():
        raise ValueError("YEAR and ID do not uniquely identify respondents in the raw file.")

    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "wb") as f:
        pickle.dump(df, f)
    print(f"Cached demographics to: {_CACHE_FILE}")

    return df[KEY_COLS + ALL_COLS]


def attach_demographics(df: pd.DataFrame, columns: Optional[List[str]] = None,
                        demographics: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Left-join demographic columns onto a frame that has YEAR and ID.

    The input's index and row order are preserved. Callers must drop the attached
    columns again before building a belief network from the result.

    Args:
        df: Frame with YEAR and ID columns, e.g. the output of `clean_datasets()`.
        columns: Which columns of ALL_COLS to attach. Defaults to all of them.
        demographics: Pre-loaded output of `load_demographics()`. Loaded if None.

    Returns:
        Copy of `df` with the requested columns added.
    """
    columns = list(ALL_COLS) if columns is None else list(columns)
    unknown = set(columns) - set(ALL_COLS)
    if unknown:
        raise ValueError(f"Not demographic or design columns: {sorted(unknown)}")
    clash = set(columns) & set(df.columns)
    if clash:
        raise ValueError(f"Columns already present in df: {sorted(clash)}")

    if demographics is None:
        demographics = load_demographics()
    lookup = demographics.set_index(KEY_COLS)[columns]
    if not lookup.index.is_unique:
        raise ValueError("Demographics are not unique on YEAR and ID.")

    return df.join(lookup, on=KEY_COLS)
