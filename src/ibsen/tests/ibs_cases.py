"""PSRB IBS reference calculations on the point grid."""

from pathlib import Path

import numpy as np

from ibsen import IBS, IBS3D, Orbit, OpticalStar, Pulsar, Winds
from .winds_cases import DAY, winds_case_parameters


N_IBS = 15
N_PHI = 13
ENERGY_EV = 1e12
REFERENCE_DIR = Path(__file__).parent / "reference" / "ibs"

IBS_CASES = {
    f"psrb_{ndim}d_{orientation or 'none'}_{label}": {
        "ndim": ndim, "orientation": orientation, "time_days": time_days,
    }
    for ndim, orientation in ((2, None), (3, None), (3, "flow_p"))
    for label, time_days in (("m50d", -50.0), ("m15d", -15.0), ("p20d", 20.0))
}

POINT_COLUMNS = (
    "s", "x", "y", "theta", "theta1", "r", "r1", "g", "dopl_star", "dopl",
    "gg_abs_1TeV", "ug", "ug_comov_iso", "ug_comov_ani", "b", "b_comov",
    "T_s_eff", "T_s_eff_comov_iso", "T_s_eff_comov_ani",
    "scattering_angle", "scattering_angle_comoving",
)
POINT_UNITS = (
    "cm", "cm", "cm", "rad", "rad", "cm", "cm", "1", "1", "1",
    "1", "erg/cm^3", "erg/cm^3", "erg/cm^3", "G", "G", "K", "K", "K", "rad", "rad",
)


def ibs_table_specs(case_name):
    """Describe the point grid shared by every saved quantity."""
    n_points = 2 * N_IBS if IBS_CASES[case_name]["ndim"] == 2 else N_IBS
    return {
        "points": {
            "shape": (n_points, len(POINT_COLUMNS)),
            "columns": POINT_COLUMNS, "units": POINT_UNITS,
        },
    }


def ibs_case_parameters(case_name):
    case = IBS_CASES[case_name]
    base_winds = winds_case_parameters("psrb_base")
    environment = {key: base_winds[key] for key in ("orbit", "star", "pulsar", "winds")}
    settings = {
        "t_to_calculate_beta_eff": case["time_days"] * DAY,
        "n": N_IBS, "s_max": 1.0, "gamma_max": 3.0, "s_max_g": 4.0,
        "coef_quench": 0.0, "abs_gg_filename": "psrb",
    }
    if case["ndim"] == 3:
        settings.update(n_phi=N_PHI, orientation=case["orientation"], shield_star=0.0)
    else:
        settings["include_incl_in_los"] = False
    return {
        "environment": environment, "ibs": settings, "ndim": case["ndim"],
        "orientation": case["orientation"], "gg_energy_eV": ENERGY_EV,
        "gg_analyt": False, "gg_what_return": "abs",
    }


def build_ibs(case_name):
    params = ibs_case_parameters(case_name)
    env = params["environment"]
    winds = Winds(
        orbit=Orbit(**env["orbit"]), star=OpticalStar(**env["star"]),
        pulsar=Pulsar(**env["pulsar"]), **env["winds"],
    )
    cls = IBS if params["ndim"] == 2 else IBS3D
    return cls(winds=winds, **params["ibs"])


def ibs_point_values(ibs):
    """Read the actual attributes, retaining their full azimuthal grids."""
    values = {}
    for column in POINT_COLUMNS:
        if column == "gg_abs_1TeV":
            absorption = np.asarray(ibs.gg_abs(
                np.array([ENERGY_EV]), analyt=False, what_return="abs",
            ))
            if absorption.shape != np.shape(ibs.s) + (1,):
                raise ValueError(f"Unexpected gg_abs shape: {absorption.shape}")
            values[column] = absorption[..., 0]
        elif column == "scattering_angle_comoving":
            values[column] = ibs.scattering_angle_comov
        elif column == "theta1" and ibs.ndim == 3:
            # IBS3D exposes this azimuth-independent angle on its one-horn grid.
            values[column] = ibs.theta1_1horn
        else:
            values[column] = getattr(ibs, column)
    return values


def _spatial_column(values, ibs, size, column):
    array = np.asarray(values)
    if not np.isfinite(array).all():
        raise ValueError(f"{column}: non-finite IBS values")
    if ibs.ndim == 3 and array.shape == (N_PHI, size):
        return np.mean(array, axis=0)
    # theta1_1horn is already one-dimensional; 2D-IBS arrays are also 1D.
    if array.shape == (size,) and (ibs.ndim == 2 or column == "theta1"):
        return array
    raise ValueError(f"{column}: unexpected IBS array shape {array.shape}")


def calculate_ibs_tables(case_name):
    ibs = build_ibs(case_name)
    specs = ibs_table_specs(case_name)
    n_points = specs["points"]["shape"][0]
    expected_grid = (n_points,) if ibs.ndim == 2 else (N_PHI, n_points)
    if np.shape(ibs.s) != expected_grid:
        raise ValueError(f"Unexpected IBS grid: {np.shape(ibs.s)}")
    points = ibs_point_values(ibs)
    return {
        "points": np.column_stack([
            _spatial_column(points[column], ibs, n_points, column)
            for column in POINT_COLUMNS
        ]),
    }


def load_ibs_tables(path):
    """Read the single point table in an IBS reference file."""
    return {"points": np.loadtxt(path, ndmin=2)}
