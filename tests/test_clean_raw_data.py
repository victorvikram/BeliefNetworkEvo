"""Tests for loaders/clean_raw_data.py (unit-testable parts only, no data loading)."""

import numpy as np
import pandas as pd
import pytest

from src.loaders.clean_raw_data import transform_column, DataConfig


class TestTransformColumn:
    def test_transform_column_basic_mapping(self):
        """Known series + mapping → expected output."""
        series = pd.Series([1, 2, 3])
        mapping = {1: -1, 2: 0, 3: 1}
        result = transform_column(series, mapping)
        expected = pd.Series([-1, 0, 1])
        pd.testing.assert_series_equal(result, expected, check_dtype=False)

    def test_transform_column_nan_handling(self):
        """'I', 'N', 'Y' values → NaN."""
        series = pd.Series([1, "I", "N", "Y", 2])
        mapping = {1: -1, 2: 1}
        result = transform_column(series, mapping)
        assert result.iloc[0] == -1
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])
        assert result.iloc[4] == 1

    def test_transform_column_unmapped_values(self):
        """Values not in mapping → NaN."""
        series = pd.Series([1, 2, 99])
        mapping = {1: -1, 2: 1}
        result = transform_column(series, mapping)
        assert result.iloc[0] == -1
        assert result.iloc[1] == 1
        assert pd.isna(result.iloc[2])


class TestDataConfig:
    def test_data_config_all_mappings(self):
        """DataConfig().all_mappings returns dict covering all VARS_*."""
        config = DataConfig()
        mappings = config.all_mappings
        assert isinstance(mappings, dict)
        assert len(mappings) > 0
        # Every mapped variable should appear in at least one VARS_* list
        all_vars = (
            config.VARS_B + config.VARS_C + config.VARS_D + config.VARS_E
            + config.VARS_F + config.VARS_G + config.VARS_H + config.VARS_I
            + config.VARS_J
        )
        for var in all_vars:
            assert var in mappings, f"{var} missing from all_mappings"

    def test_data_config_all_questions(self):
        """DataConfig().all_questions returns non-empty list with no duplicates."""
        config = DataConfig()
        questions = config.all_questions
        assert len(questions) > 0
        assert len(questions) == len(set(questions))
