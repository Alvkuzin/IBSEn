"""Regression checks of PSRB IBS quantities on the point grid."""

import numpy as np
import pytest

from .ibs_cases import (
    IBS_CASES, REFERENCE_DIR, calculate_ibs_tables, ibs_table_specs, load_ibs_tables,
)


RTOL = 1e-5
ATOL_BY_UNIT = {"cm": 1.0, "rad": 1e-12, "1": 1e-12,
                "erg/cm^3": 1e-12, "G": 1e-12, "K": 1e-6}


@pytest.mark.parametrize("case_name", IBS_CASES)
def test_ibs_tables(case_name):
    actual = calculate_ibs_tables(case_name)
    expected = load_ibs_tables(REFERENCE_DIR / f"{case_name}.txt")

    for name, spec in ibs_table_specs(case_name).items():
        assert actual[name].shape == expected[name].shape == spec["shape"]
        assert np.isfinite(actual[name]).all(), f"{case_name}/{name}: non-finite calculation"
        assert np.isfinite(expected[name]).all(), f"{case_name}/{name}: non-finite reference"
        for index, (column, unit) in enumerate(zip(spec["columns"], spec["units"])):
            np.testing.assert_allclose(
                actual[name][:, index], expected[name][:, index],
                rtol=RTOL, atol=ATOL_BY_UNIT[unit], equal_nan=False,
                err_msg=f"{case_name}/{name}: {column} differs from the saved reference",
            )
