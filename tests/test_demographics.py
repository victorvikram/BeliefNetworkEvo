"""Tests for loaders/demographics.py (join logic only, no data loading)."""

import numpy as np
import pandas as pd
import pytest

from src.loaders.demographics import attach_demographics


@pytest.fixture
def demographics():
    return pd.DataFrame({
        "YEAR": [2000.0, 2000.0, 2002.0],
        "ID": [1.0, 2.0, 1.0],
        "AGE": [30.0, 45.0, 60.0],
        "WTSSPS": [1.1, 0.9, 1.0],
    })


@pytest.fixture
def cleaned():
    # Shuffled order and a non-default index, as after filtering a cleaned frame
    return pd.DataFrame({
        "YEAR": [2002.0, 2000.0, 2004.0],
        "ID": [1.0, 2.0, 7.0],
        "HOMOSEX": [0.5, -1.0, 1.0],
    }, index=[10, 3, 99])


class TestAttachDemographics:
    def test_values_match_on_year_and_id(self, cleaned, demographics):
        """ID repeats across years, so the join must use YEAR as well."""
        out = attach_demographics(cleaned, ["AGE"], demographics)
        assert out.loc[10, "AGE"] == 60.0
        assert out.loc[3, "AGE"] == 45.0

    def test_preserves_index_and_order(self, cleaned, demographics):
        out = attach_demographics(cleaned, ["AGE", "WTSSPS"], demographics)
        assert list(out.index) == [10, 3, 99]
        pd.testing.assert_frame_equal(out[cleaned.columns], cleaned)

    def test_unmatched_rows_get_nan(self, cleaned, demographics):
        out = attach_demographics(cleaned, ["AGE"], demographics)
        assert np.isnan(out.loc[99, "AGE"])

    def test_does_not_modify_input(self, cleaned, demographics):
        before = cleaned.copy()
        attach_demographics(cleaned, ["AGE"], demographics)
        pd.testing.assert_frame_equal(cleaned, before)

    def test_rejects_unknown_column(self, cleaned, demographics):
        with pytest.raises(ValueError, match="Not demographic"):
            attach_demographics(cleaned, ["HOMOSEX"], demographics)

    def test_rejects_existing_column(self, cleaned, demographics):
        with_age = cleaned.assign(AGE=1.0)
        with pytest.raises(ValueError, match="already present"):
            attach_demographics(with_age, ["AGE"], demographics)
