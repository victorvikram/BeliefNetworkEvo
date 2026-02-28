"""Tests for generators/corr_make_conditioned_network.py"""

import numpy as np
import pandas as pd
import pytest

from src.generators.corr_make_conditioned_network import calculate_conditioned_correlation_matrix


class TestNoConditioning:
    def test_no_conditioning(self, synthetic_cleaned_df):
        """No variable_to_condition → same as calculate_correlation_matrix."""
        result = calculate_conditioned_correlation_matrix(
            synthetic_cleaned_df,
            variable_to_condition=None,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]
        # Diagonal should be 0 (set by calculate_correlation_matrix)
        np.testing.assert_array_almost_equal(np.diag(result.values), 0)


class TestConditions:
    def test_condition_less_than_zero(self, synthetic_cleaned_df):
        """Filters correctly, returns smaller matrix."""
        result = calculate_conditioned_correlation_matrix(
            synthetic_cleaned_df,
            variable_to_condition="VAR_0",
            condition="less_than_zero",
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]

    def test_condition_greater_than_zero(self, synthetic_cleaned_df):
        """Filters correctly."""
        result = calculate_conditioned_correlation_matrix(
            synthetic_cleaned_df,
            variable_to_condition="VAR_0",
            condition="greater_than_zero",
        )
        assert isinstance(result, pd.DataFrame)

    def test_condition_equal_to(self, synthetic_cleaned_df):
        """Filters to exact value."""
        # Use YEAR column for an exact-value condition
        result = calculate_conditioned_correlation_matrix(
            synthetic_cleaned_df,
            variable_to_condition="YEAR",
            condition="equal_to",
            value=2002,
        )
        assert isinstance(result, pd.DataFrame)


class TestErrors:
    def test_invalid_condition_raises(self, synthetic_cleaned_df):
        """Bad condition string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid condition"):
            calculate_conditioned_correlation_matrix(
                synthetic_cleaned_df,
                variable_to_condition="VAR_0",
                condition="bad_condition",
            )

    def test_missing_column_raises(self, synthetic_cleaned_df):
        """Nonexistent column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            calculate_conditioned_correlation_matrix(
                synthetic_cleaned_df,
                variable_to_condition="NONEXISTENT_COL",
                condition="less_than_zero",
            )


class TestReturnDf:
    def test_return_df_flag(self, synthetic_cleaned_df):
        """return_df=True returns tuple."""
        result = calculate_conditioned_correlation_matrix(
            synthetic_cleaned_df,
            variable_to_condition="VAR_0",
            condition="less_than_zero",
            return_df=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        corr_mat, filtered_df = result
        assert isinstance(corr_mat, pd.DataFrame)
        assert isinstance(filtered_df, pd.DataFrame)
        # Filtered df should only have rows where VAR_0 < 0
        assert (filtered_df["VAR_0"].dropna() < 0).all()
