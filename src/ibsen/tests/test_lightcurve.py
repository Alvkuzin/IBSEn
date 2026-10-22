"""Fluxes, photon indexes, and IBS diagnostics over a PSRB orbit."""

import numpy as np
import pytest

from .lightcurve_cases import (
    COLUMNS, UNITS, LIGHTCURVE_CASES, N_TIMES, REFERENCE_DIR, calculate_lightcurve_table,
)


@pytest.mark.parametrize("case_name", LIGHTCURVE_CASES)
def test_lightcurve_table(case_name):
    actual = calculate_lightcurve_table(case_name)
    expected = np.loadtxt(REFERENCE_DIR / f"{case_name}.txt")
    assert actual.shape == expected.shape == (N_TIMES, len(COLUMNS))
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    for i, (column, unit) in enumerate(zip(COLUMNS, UNITS)):
        atol = 1.0 if unit == "cm" else 1e-6 if unit == "s" else 0.0
        np.testing.assert_allclose(actual[:, i], expected[:, i], rtol=1e-5, atol=atol,
                                   equal_nan=False, err_msg=f"{case_name}: {column}")
