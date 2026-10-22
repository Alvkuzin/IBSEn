"""SED and total-transmission regression tests for all three spectrum methods."""

import numpy as np
import pytest

from .spectrum_cases import COLUMNS, ENERGIES, REFERENCE_DIR, SPECTRUM_CASES, calculate_spectrum_table


@pytest.mark.parametrize("case_name", SPECTRUM_CASES)
def test_spectrum_table(case_name):
    actual = calculate_spectrum_table(case_name)
    expected = np.loadtxt(REFERENCE_DIR / f"{case_name}.txt")
    assert actual.shape == expected.shape == (len(ENERGIES), len(COLUMNS))
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    assert np.all(actual[:, 1] >= 0)
    assert np.all((actual[:, 2] >= 0) & (actual[:, 2] <= 1))
    for i, column in enumerate(COLUMNS):
        np.testing.assert_allclose(actual[:, i], expected[:, i], rtol=1e-5, atol=0,
                                   equal_nan=False, err_msg=f"{case_name}: {column}")
