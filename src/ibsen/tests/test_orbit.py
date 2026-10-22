"""Compare freshly calculated orbit tables with committed reference values."""

import numpy as np
import pytest

from .orbit_cases import COLUMNS, ORBIT_CASES, REFERENCE_DIR, calculate_orbit_table


# Relative tolerance allows small floating-point differences between platforms.
# Absolute tolerances cover values near zero, in each column's physical units.
RTOL = 1e-5
ATOL = {"ttab": 1e-6, "xtab_p": 1.0, "ytab_p": 1.0,
        "rtab": 1.0, "nu_truetab": 1e-12}


@pytest.mark.parametrize("case_name", ORBIT_CASES)
def test_orbit_table(case_name, monkeypatch):
    def reject_system_defaults(*args, **kwargs):
        pytest.fail("Orbit regression cases must not load system defaults")

    monkeypatch.setattr("ibsen.orbit.get_parameters", reject_system_defaults)
    actual = calculate_orbit_table(case_name)
    expected = np.loadtxt(REFERENCE_DIR / f"{case_name}.txt")

    assert actual.shape == expected.shape == (50, len(COLUMNS))
    assert np.isfinite(actual).all(), f"{case_name}: non-finite calculated values"
    assert np.isfinite(expected).all(), f"{case_name}: non-finite reference values"
    for index, column in enumerate(COLUMNS):
        np.testing.assert_allclose(
            actual[:, index], expected[:, index],
            rtol=RTOL, atol=ATOL[column], equal_nan=False,
            err_msg=f"{case_name}: {column} differs from the saved reference",
        )
