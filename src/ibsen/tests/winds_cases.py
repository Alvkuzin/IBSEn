"""PSRB wind cases shared by regression tests and reference generation."""

from pathlib import Path

import numpy as np

from ibsen import Orbit, OpticalStar, Pulsar, Winds
from ibsen.get_obs_data import get_parameters


DAY = 86400.0
TIMES = np.linspace(-100.0, 100.0, 101) * DAY
PRECALCULATE_TIMES = np.linspace(-200.0, 100.0, 601) * DAY

BASE_PARAMETERS = {
    "f_d": 1e3,
    "delta": 0.02,
    "np_disk": 3.0,
    "height_exp": 0.5,
    "f_p": 0.1,
    "orientation": None,
    "hyst": False,
    "alpha_interaction": 0.0,
}

# Each case starts from the baseline, not the preceding case. Hysteresis is
# the deliberate exception: enabling it also sets alpha_interaction to 1.
WIND_CASES = {
    "psrb_base": dict(BASE_PARAMETERS),
    "psrb_fd_100": {**BASE_PARAMETERS, "f_d": 100.0},
    "psrb_delta_004": {**BASE_PARAMETERS, "delta": 0.04},
    "psrb_np_disk_35": {**BASE_PARAMETERS, "np_disk": 3.5},
    "psrb_height_exp_025": {**BASE_PARAMETERS, "height_exp": 0.25},
    "psrb_fp_02": {**BASE_PARAMETERS, "f_p": 0.2},
    "psrb_flow": {**BASE_PARAMETERS, "orientation": "flow"},
    "psrb_flow_p": {**BASE_PARAMETERS, "orientation": "flow_p"},
    "psrb_alpha_1": {**BASE_PARAMETERS, "alpha_interaction": 1.0},
    "psrb_hyst": {**BASE_PARAMETERS, "hyst": True, "alpha_interaction": 1.0},
}

COLUMNS = ("t", "r_pe", "b_apex", "u_apex", "vec_x", "vec_y", "vec_z")
UNITS = ("s", "cm", "G", "erg/cm^3", "cm", "cm", "cm")
REFERENCE_DIR = Path(__file__).parent / "reference" / "winds"


def winds_case_parameters(case_name):
    """Resolve and record every constructor setting, including PSRB defaults."""
    case = WIND_CASES[case_name]
    system = get_parameters("psrb")
    orbit = {key: system[key] for key in ("T", "e", "M_s", "M_p", "nu_los", "incl_los")}
    orbit.update(sys_name="psrb", sys_params=None, n=None, allow_missing=False)
    star = {
        "sys_name": "psrb", "sys_params": None, "allow_missing": False,
        "R_s": system["R_s"], "T_s": system["T_s"], "M_s": system["M_s"],
        "f_w": 1.0, "alpha_disk_deg": 0.0, "incl_disk_deg": 30.0,
        "f_d": case["f_d"], "np_disk": case["np_disk"],
        "delta": case["delta"], "height_exp": case["height_exp"],
        "rad_prof": "pl", "r_trunk": None, "vert_prof": "gauss",
        "np_disk_in": None, "v_polar_wind": 3e8,
        "p_enh": [1.0], "p_enh_true_an": [0.0],
        "h_enh": [1.0], "h_enh_true_an": [0.0],
        "b_model": "linear", "b_ref": 1.0, "r_b_ref": system["R_s"],
    }
    pulsar = {
        "f_p": case["f_p"], "r_p_ref": system["R_s"],
        "b_model": "linear", "b_ref": 1.0, "r_b_ref": 1e13,
        "L_spindown": None, "sigma_magn": None,
    }
    winds = {
        "hyst": case["hyst"], "alpha_interaction": case["alpha_interaction"],
        "k_time": 1.0, "incl_puls_vel": False,
        "t_precalculate": PRECALCULATE_TIMES.tolist(),
    }
    return {
        "orbit": orbit, "star": star, "pulsar": pulsar, "winds": winds,
        "orientation": case["orientation"], "times": TIMES.tolist(),
    }


def calculate_winds_table(case_name):
    """Calculate physical wind outputs independently at each saved epoch."""
    params = winds_case_parameters(case_name)
    orbit = Orbit(**params["orbit"])
    star = OpticalStar(**params["star"])
    pulsar = Pulsar(**params["pulsar"])
    winds = Winds(orbit=orbit, star=star, pulsar=pulsar, **params["winds"])
    orientation = params["orientation"]
    rows = []
    for t in params["times"]:
        r_pe = winds.dist_pe(t, orientation=orientation)
        b_pulsar, b_star = winds.magn_fields_apex(t, orientation=orientation)
        u_apex = winds.u_g_density_apex(t, orientation=orientation)
        if orientation is None:
            vector = orbit.vector_sp(t)
        else:
            vector = winds.vec_pe_3d(t, orientation=orientation)
        rows.append((t, r_pe, b_pulsar + b_star, u_apex, *vector))
    return np.asarray(rows)
