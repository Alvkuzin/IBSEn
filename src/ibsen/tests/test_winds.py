"""Compare PSRB wind calculations with saved tables for each parameter case."""

import numpy as np
import pytest

from .winds_cases import COLUMNS, REFERENCE_DIR, TIMES, WIND_CASES, calculate_winds_table


RTOL = 1e-5
ATOL = {
    "t": 1e-6, "r_pe": 1.0, "b_apex": 1e-12, "u_apex": 1e-12,
    "vec_x": 1.0, "vec_y": 1.0, "vec_z": 1.0,
}


@pytest.mark.parametrize("case_name", WIND_CASES)
def test_winds_table(case_name):
    actual = calculate_winds_table(case_name)
    expected = np.loadtxt(REFERENCE_DIR / f"{case_name}.txt")

    assert actual.shape == expected.shape == (len(TIMES), len(COLUMNS))
    assert np.isfinite(actual).all(), f"{case_name}: non-finite calculated values"
    assert np.isfinite(expected).all(), f"{case_name}: non-finite reference values"
    for index, column in enumerate(COLUMNS):
        np.testing.assert_allclose(
            actual[:, index], expected[:, index],
            rtol=RTOL, atol=ATOL[column], equal_nan=False,
            err_msg=f"{case_name}: {column} differs from the saved reference",
        )
        
@pytest.mark.parametrize("f_p", [1e-3, 1e-2, 0.99])
def test_winds_analytical(f_p):
    from ibsen import Orbit, Pulsar, OpticalStar, Winds
    # 1. Initialize objects with your chosen parameters.
    orbit = Orbit('psrb')
    star = OpticalStar('psrb', f_d=0.0)
    pulsar = Pulsar(f_p=f_p, r_p_ref=star.R_s)
    winds = Winds(orbit=orbit, star=star, pulsar=pulsar)

    # 2. Calculate the numerical and analytical results.
    times = orbit.timegrid_lin_in_anomaly(n_samples=101)
    actual = winds.dist_pe(times)
    expected = orbit.r(times) * f_p**0.5 / (1. + f_p**0.5)

    # 3. Compare them.
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=1.0,  # Absolute tolerance in the quantity's units.
        equal_nan=False,
    )
