"""Explicit orbit inputs shared by regression tests and reference generation."""

from pathlib import Path
import numpy as np
from ibsen.orbit import Orbit


DAY = 86400.
SOLAR_MASS = 1.99e33 

# Approximate system parameters, deliberately independent of get_parameters().
# Periods are in seconds, masses in grams, and angles in radians.
ORBIT_CASES = {
    "psrb": {
        "T": 1236.72 * DAY,
        "e": 0.87,
        "M_s": 24.0 * SOLAR_MASS,
        "M_p": 1.4 * SOLAR_MASS,
        "nu_los": np.deg2rad(131.3),
        "incl_los": np.deg2rad(155.5),
        "n": 50,
    },
    "ls5039": {
        "T": 3.9 * DAY,
        "e": 0.35,
        "M_s": 23.0 * SOLAR_MASS,
        "M_p": 1.4 * SOLAR_MASS,
        "nu_los": np.deg2rad(224.2),
        "incl_los": np.deg2rad(60.0),
        "n": 50,
    },
}

COLUMNS = ("ttab", "xtab_p", "ytab_p", "rtab", "nu_truetab")
UNITS = ("s", "cm", "cm", "cm", "rad")
REFERENCE_DIR = Path(__file__).parent / "reference" / "orbit"


def calculate_orbit_table(case_name):
    """Return the orbit's own n=50 tabulation, without resampling it."""
    orbit = Orbit(sys_name=None, **ORBIT_CASES[case_name])
    return np.column_stack([getattr(orbit, column) for column in COLUMNS])
