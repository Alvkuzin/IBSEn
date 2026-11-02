"""Fluxes, photon indexes, and IBS diagnostics over a PSRB orbit."""

import numpy as np
import pytest

from ibsen import ALLOWED_FOR_MULTIPLICATION, LightCurve

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


def _lightcurve_for_multiplier_tests():
    """Create a minimal instance without constructing the physical pipeline."""
    lc = LightCurve.__new__(LightCurve)
    lc.t = np.array([-2.0, 0.0, 3.0])
    lc.norm_e = 10.0
    lc.ecut = 4.0
    return lc


def test_time_multipliers_are_precalculated_and_composed():
    lc = _lightcurve_for_multiplier_tests()

    lc._precalculate_effective_values([
        {"parameter": "norm_e", "func": lambda t: t + 3.0},
        {"parameter": "norm_e", "func": lambda t: 2.0},
    ])

    np.testing.assert_allclose(lc._effective_values["norm_e"], [20.0, 60.0, 120.0])
    assert lc._effective("norm_e", 1) == 60.0
    assert lc._effective("ecut", 1) == lc.ecut
    assert not hasattr(lc, "time_multipliers")


def test_unknown_time_multiplier_warns_and_is_ignored():
    lc = _lightcurve_for_multiplier_tests()

    with pytest.warns(UserWarning, match="ALLOWED_FOR_MULTIPLICATION"):
        lc._precalculate_effective_values([
            {"parameter": "not_a_lightcurve_parameter", "func": lambda t: 2.0},
        ])

    assert lc._effective_values == {}


def test_time_multiplier_must_return_a_finite_scalar():
    lc = _lightcurve_for_multiplier_tests()

    with pytest.raises(ValueError, match="non-scalar"):
        lc._precalculate_effective_values([
            {"parameter": "norm_e", "func": lambda t: [1.0, 2.0]},
        ])

    with pytest.raises(ValueError, match="non-finite"):
        lc._precalculate_effective_values([
            {"parameter": "norm_e", "func": lambda t: np.nan},
        ])


def test_allowed_for_multiplication_covers_each_pipeline_stage():
    assert {
        "gamma_max",       # IBS
        "norm_e",          # ElectronsOnIBS
        "distance",        # SpectrumIBS
    } <= set(ALLOWED_FOR_MULTIPLICATION)
