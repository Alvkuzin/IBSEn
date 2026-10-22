"""Regression checks of summed lab/comoving electron spectra and populations."""

import numpy as np
import pytest

from .electrons_cases import (
    ELECTRON_CASES, REFERENCE_DIR, calculate_electron_tables,
    electron_table_specs, load_electron_tables,
)


RTOL = 1e-5
# Preserve sensitivity in the tiny high-energy tail; no absolute spectrum floor.
ATOL_BY_UNIT = {"eV": 0.0, "1/eV": 0.0, "1": 0.0, "cm": 1.0}


@pytest.mark.parametrize("case_name", ELECTRON_CASES)
def test_electron_tables(case_name):
    actual = calculate_electron_tables(case_name)
    expected = load_electron_tables(REFERENCE_DIR / f"{case_name}.txt")

    for name, spec in electron_table_specs(case_name).items():
        assert actual[name].shape == expected[name].shape == spec["shape"]
        assert np.isfinite(actual[name]).all(), f"{case_name}/{name}: non-finite calculation"
        assert np.isfinite(expected[name]).all(), f"{case_name}/{name}: non-finite reference"
        for index, (column, unit) in enumerate(zip(spec["columns"], spec["units"])):
            np.testing.assert_allclose(
                actual[name][:, index], expected[name][:, index],
                rtol=RTOL, atol=ATOL_BY_UNIT[unit], equal_nan=False,
                err_msg=f"{case_name}/{name}: {column} differs from the saved reference",
            )
