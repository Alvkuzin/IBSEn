"""Tabulated absorption regressions and optional direct-integration diagnostics."""

import numpy as np
import pytest

from .absorption_cases import (
    ABSORPTION_CASES, GG_CASES, REFERENCE_DIR, absorption_table_specs,
    calculate_absorption_tables, direct_gg_tau,
)


@pytest.mark.parametrize("case_name", ABSORPTION_CASES)
def test_absorption_table(case_name):
    actual = calculate_absorption_tables(case_name)["values"]
    expected = np.loadtxt(REFERENCE_DIR / f"{case_name}.txt")
    spec = absorption_table_specs(case_name)["values"]
    assert actual.shape == expected.shape == spec["shape"]
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    for i, (column, unit) in enumerate(zip(spec["columns"], spec["units"])):
        if column.startswith("tau"):
            assert np.all(actual[:, i] >= 0)
        np.testing.assert_allclose(actual[:, i], expected[:, i], rtol=1e-5,
                                   atol=1.0 if unit == "cm" else 0.0,
                                   equal_nan=False, err_msg=f"{case_name}: {column}")


@pytest.mark.gg_direct
@pytest.mark.parametrize("case_name", GG_CASES)
def test_gg_direct_discrepancy(case_name, request, capsys):
    if not request.config.getoption("--run-gg-direct"):
        pytest.skip("Use --run-gg-direct to compare tables with precise gamma-gamma integration")
    reference = np.loadtxt(REFERENCE_DIR / f"{case_name}.txt")
    expected = reference[:, -1]
    actual = direct_gg_tau(case_name, reference)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    assert np.all(actual >= 0) and np.all(expected >= 0)
    difference = np.abs(actual - expected)
    # Relative errors at zero/tiny tabulated optical depth are uninformative.
    mask = expected > 1e-4
    with capsys.disabled():
        print(f"\n{case_name}: direct tau_gg_iso_2d(fast=False) versus saved gg_tab_d_mu")
        print(f"  Absolute |tau_direct - tau_tab|: max={difference.max():.8e}, mean={difference.mean():.8e}")
        if np.any(mask):
            relative = difference[mask] / expected[mask]
            print(f"  Relative |difference|/tau_tab (tau_tab > 1e-4; {mask.sum()}/{mask.size} points): "
                  f"max={relative.max():.8e}, mean={relative.mean():.8e}")
        else:
            print("  Relative discrepancy: unavailable (all tabulated tau <= 1e-4)")
    # This is a diagnostic comparison, intentionally without a closeness threshold.
