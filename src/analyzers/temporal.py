"""
Rolling-window network builder for temporal analysis.

Provides reusable infrastructure for building belief networks across time windows,
with optional group conditioning and sample-size matching.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.generators.corr_make_network import (
    CorrelationMethod,
    EdgeSuppressionMethod,
    calculate_correlation_matrix,
)
from src.loaders.clean_raw_data import DataConfig


@dataclass
class WindowResult:
    """Result from a single rolling window."""
    start_year: int
    end_year: int
    mid_year: float
    years_in_window: List[int]
    networks: Dict[str, pd.DataFrame]   # group_name -> corr matrix (aligned to common_vars)
    common_vars: List[str]
    sample_sizes: Dict[str, int]        # group_name -> original N
    matched_n: Optional[int]


def get_available_years(df: pd.DataFrame, require_col: Optional[str] = None) -> List[int]:
    """Get sorted list of unique years in the dataset.

    Parameters
    ----------
    df : DataFrame with a YEAR column.
    require_col : If given, only include years where this column has non-null values.
    """
    if require_col is not None:
        subset = df[df[require_col].notna()]
    else:
        subset = df
    years = sorted(subset["YEAR"].dropna().unique().astype(int).tolist())
    return years


def build_rolling_windows(
    df: pd.DataFrame,
    window_size: int = 4,
    step_size: int = 2,
    min_years_per_window: int = 3,
    min_n_per_group: int = 100,
    method: CorrelationMethod = CorrelationMethod.PEARSON,
    partial: bool = True,
    edge_suppression: EdgeSuppressionMethod = EdgeSuppressionMethod.REGULARIZATION,
    suppression_params: Optional[dict] = None,
    group_col: Optional[str] = None,
    group_conditions: Optional[Dict[str, str]] = None,
    match_samples: bool = True,
    random_state: int = 42,
    fixed_vars: Optional[List[str]] = None,
    verbose: bool = False,
) -> List[WindowResult]:
    """Build correlation networks across rolling time windows.

    Parameters
    ----------
    df : Cleaned GSS DataFrame (with YEAR column).
    window_size : Width of each window in years.
    step_size : Step between window start years.
    min_years_per_window : Skip window if fewer GSS years fall in it.
    min_n_per_group : Skip window if any group has fewer respondents.
    method, partial, edge_suppression, suppression_params : Correlation params.
    group_col : Column to split on (e.g. 'POLVIEWS'). None = total network.
    group_conditions : Dict mapping group names to conditions.
        Values: '< 0', '> 0', '== 0'. E.g. {'lib': '< 0', 'con': '> 0'}.
    match_samples : If True, downsample all groups to the smallest group's N.
    random_state : Seed for reproducibility.
    fixed_vars : If provided, restrict all matrices to these variables.
    verbose : Print progress information.

    Returns
    -------
    List of WindowResult, one per successful window.
    """
    if suppression_params is None:
        suppression_params = {"regularization": 0.2}

    # Determine available years
    require_col = group_col if group_col else None
    available_years = get_available_years(df, require_col=require_col)

    first_year = available_years[0]
    last_year = available_years[-1]
    window_starts = list(range(first_year, last_year - window_size + 2, step_size))

    if verbose:
        print(f"Available years: {first_year}-{last_year}")
        print(f"Rolling windows: {len(window_starts)} windows")

    rng = np.random.default_rng(random_state)
    results: List[WindowResult] = []

    for start in window_starts:
        end = start + window_size
        window_years = [y for y in available_years if start <= y <= end]

        if len(window_years) < min_years_per_window:
            continue

        df_window = df[df["YEAR"].isin(window_years)].copy()

        # --- No grouping: build a single total network ---
        if group_col is None:
            try:
                corr = calculate_correlation_matrix(
                    df_window,
                    method=method,
                    partial=partial,
                    edge_suppression=edge_suppression,
                    suppression_params=suppression_params,
                    verbose=False,
                )
            except Exception as e:
                if verbose:
                    print(f"  Window {start}-{end}: FAILED ({e})")
                continue

            if corr is None:
                if verbose:
                    print(f"  Window {start}-{end}: correlation matrix returned None")
                continue

            net_vars = sorted(corr.columns.tolist())
            if fixed_vars is not None:
                net_vars = sorted(set(net_vars) & set(fixed_vars))
                if len(net_vars) < 10:
                    continue
                corr = corr.loc[net_vars, net_vars]

            results.append(WindowResult(
                start_year=start,
                end_year=end,
                mid_year=start + window_size / 2,
                years_in_window=window_years,
                networks={"total": corr},
                common_vars=net_vars,
                sample_sizes={"total": len(df_window)},
                matched_n=None,
            ))
            continue

        # --- Grouped: split, optionally match, build networks ---
        if group_conditions is None:
            raise ValueError("group_conditions required when group_col is set")

        df_pv = df_window[df_window[group_col].notna()]
        groups: Dict[str, pd.DataFrame] = {}

        for name, cond in group_conditions.items():
            if cond == "< 0":
                groups[name] = df_pv[df_pv[group_col] < 0]
            elif cond == "> 0":
                groups[name] = df_pv[df_pv[group_col] > 0]
            elif cond == "== 0":
                groups[name] = df_pv[df_pv[group_col] == 0]
            else:
                raise ValueError(f"Unknown condition: {cond}")

        # Check minimum sizes
        sample_sizes = {name: len(g) for name, g in groups.items()}
        if any(n < min_n_per_group for n in sample_sizes.values()):
            continue

        # Sample matching
        matched_n = None
        if match_samples:
            matched_n = min(sample_sizes.values())
            matched_groups: Dict[str, pd.DataFrame] = {}
            for name, g in groups.items():
                if len(g) > matched_n:
                    matched_groups[name] = g.sample(
                        n=matched_n, random_state=rng.integers(1e9)
                    )
                else:
                    matched_groups[name] = g
            groups = matched_groups

        # Build correlation matrices
        networks: Dict[str, pd.DataFrame] = {}
        failed = False
        for name, g in groups.items():
            try:
                corr = calculate_correlation_matrix(
                    g,
                    method=method,
                    partial=partial,
                    edge_suppression=edge_suppression,
                    suppression_params=suppression_params,
                    verbose=False,
                )
                if corr is None:
                    raise ValueError("correlation matrix returned None")
                networks[name] = corr
            except Exception as e:
                if verbose:
                    print(f"  Window {start}-{end} ({name}): FAILED ({e})")
                failed = True
                break

        if failed:
            continue

        # Align to common variables
        all_var_sets = [set(net.columns) for net in networks.values()]
        common = sorted(set.intersection(*all_var_sets))

        if fixed_vars is not None:
            common = sorted(set(common) & set(fixed_vars))

        if len(common) < 10:
            continue

        aligned = {name: net.loc[common, common] for name, net in networks.items()}

        results.append(WindowResult(
            start_year=start,
            end_year=end,
            mid_year=start + window_size / 2,
            years_in_window=window_years,
            networks=aligned,
            common_vars=common,
            sample_sizes=sample_sizes,
            matched_n=matched_n,
        ))

    if verbose:
        print(f"Completed {len(results)} windows")

    return results
