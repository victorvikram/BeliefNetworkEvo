"""Tests for analyzers/belief_change.py"""

import numpy as np
import pandas as pd
import pytest

from source_code.analyzers.belief_change import align_dfs, align_dfs_multiple, subtract_dataframes


class TestAlignDfs:
    def test_align_dfs_union(self, small_corr_matrix, small_corr_matrix_alt):
        """Union of two DataFrames with overlapping + unique vars, missing filled with 0."""
        df1_aligned, df2_aligned = align_dfs(small_corr_matrix, small_corr_matrix_alt, method="union")

        expected_vars = sorted(set(small_corr_matrix.index) | set(small_corr_matrix_alt.index))
        assert list(df1_aligned.index) == expected_vars
        assert list(df1_aligned.columns) == expected_vars
        assert list(df2_aligned.index) == expected_vars
        assert list(df2_aligned.columns) == expected_vars

        # Variables unique to df2 should be 0 in df1_aligned
        for var in set(small_corr_matrix_alt.index) - set(small_corr_matrix.index):
            assert (df1_aligned.loc[var, :] == 0).all()
            assert (df1_aligned.loc[:, var] == 0).all()

    def test_align_dfs_intersection(self, small_corr_matrix, small_corr_matrix_alt):
        """Intersection keeps only shared vars."""
        df1_aligned, df2_aligned = align_dfs(small_corr_matrix, small_corr_matrix_alt, method="intersection")

        common_vars = sorted(set(small_corr_matrix.index) & set(small_corr_matrix_alt.index))
        assert list(df1_aligned.index) == common_vars
        assert list(df1_aligned.columns) == common_vars
        assert list(df2_aligned.index) == common_vars
        assert list(df2_aligned.columns) == common_vars

    def test_align_dfs_identical(self, small_corr_matrix):
        """Same vars → no change."""
        df1_aligned, df2_aligned = align_dfs(small_corr_matrix, small_corr_matrix.copy(), method="union")
        pd.testing.assert_frame_equal(df1_aligned, small_corr_matrix)
        pd.testing.assert_frame_equal(df2_aligned, small_corr_matrix)

    def test_align_dfs_multiple(self, small_corr_matrix, small_corr_matrix_alt, identity_corr_matrix):
        """Aligning 3+ DataFrames."""
        aligned = align_dfs_multiple([small_corr_matrix, small_corr_matrix_alt, identity_corr_matrix])
        assert len(aligned) == 3
        all_vars = sorted(
            set(small_corr_matrix.index)
            | set(small_corr_matrix_alt.index)
            | set(identity_corr_matrix.index)
        )
        for df in aligned:
            assert list(df.index) == all_vars
            assert list(df.columns) == all_vars


class TestSubtractDataframes:
    def test_subtract_dataframes(self, small_corr_matrix):
        """df1 - df2 with matching vars."""
        result = subtract_dataframes(small_corr_matrix, small_corr_matrix)
        assert (result.values == 0).all()

    def test_subtract_dataframes_mismatch(self, small_corr_matrix, small_corr_matrix_alt):
        """Different var sets, intersection mode."""
        result = subtract_dataframes(small_corr_matrix, small_corr_matrix_alt, mismatch_method="intersection")
        common_vars = sorted(set(small_corr_matrix.index) & set(small_corr_matrix_alt.index))
        assert list(result.index) == common_vars
        assert list(result.columns) == common_vars
