"""A 71-epoch PSRB light curve and its physical diagnostics."""

from pathlib import Path

import numpy as np

from ibsen import LightCurve, Orbit
from .electrons_cases import electron_case_parameters
from .spectrum_cases import spectrum_settings


REFERENCE_DIR = Path(__file__).parent / "reference" / "lightcurve"
LIGHTCURVE_CASES = {"psrb_lc_simple_stat_mimic": {}}
N_TIMES = 71
BANDS = [[1e3, 1e4], [4e11, 1e13]]
COLUMNS = (
    "t", "flux_1_10keV", "flux_0p4_10TeV", "index_1_10keV", "index_0p4_10TeV",
    "r_sp", "r_pe", "ntot_lab", "ntot_comov", "b_mean", "b_comov_mean",
    "ug_mean", "ug_comov_iso_mean", "ug_comov_ani_mean", "dopl_mean",
    "scattering_angle_mean", "scattering_angle_comov_mean",
)
UNITS = ("s", "erg/s/cm^2", "1/s/cm^2", "1", "1", "cm", "cm", "1", "1",
         "G", "G", "erg/cm^3", "erg/cm^3", "erg/cm^3", "1", "rad", "rad")
MEAN_ATTRIBUTES = ("b", "b_comov", "ug", "ug_comov_iso", "ug_comov_ani",
                   "dopl", "scattering_angle", "scattering_angle_comov")


def lightcurve_case_parameters(case_name):
    setup = electron_case_parameters("psrb_electrons_p20d_stat_mimic")
    env = setup["environment"]
    star, pulsar = env["star"], env["pulsar"]
    orbit = Orbit(**env["orbit"])
    params = {
        **{key: env["orbit"][key] for key in ("T", "e", "M_s", "M_p", "nu_los", "incl_los")},
        **{key: star[key] for key in ("R_s", "T_s", "alpha_disk_deg", "incl_disk_deg", "f_d",
                                    "np_disk", "delta", "height_exp", "rad_prof", "r_trunk",
                                    "np_disk_in", "vert_prof", "p_enh", "h_enh")},
        **{key: env["winds"][key] for key in ("hyst", "k_time", "alpha_interaction", "incl_puls_vel")},
        **setup["electrons"],
        **{key: setup["ibs"][key] for key in ("s_max", "gamma_max", "s_max_g", "n_phi",
                                              "orientation", "coef_quench", "shield_star", "abs_gg_filename")},
        **{key: value for key, value in spectrum_settings("simple").items() if key != "ic_approx"},
        "f_p": pulsar["f_p"], "p_enh_times": [0.0], "h_enh_times": [0.0],
        "puls_b_model": pulsar["b_model"], "puls_b_ref": pulsar["b_ref"],
        "puls_r_ref": pulsar["r_b_ref"], "puls_L_spindown": pulsar["L_spindown"],
        "puls_sigma_magn": pulsar["sigma_magn"],
        "s_b_model": star["b_model"], "s_b_ref": star["b_ref"], "s_r_ref": star["r_b_ref"],
        "ibs_ndim": 3, "n_ibs": setup["ibs"]["n"],
        "times": orbit.timegrid_lin_in_anomaly(n_samples=N_TIMES).tolist(),
        "bands": BANDS, "bands_ind": BANDS, "epows": [1.0, 0.0],
        "full_spec": False, "to_parall": False, "n_cores": 1, "allow_missing": False,
    }
    return params


def calculate_lightcurve_table(case_name):
    params = lightcurve_case_parameters(case_name)
    lc = LightCurve(**{**params, "times": np.asarray(params["times"])})
    lc.calculate()
    rows = []
    for i, (ibs, electrons) in enumerate(zip(lc.ibs_classes, lc.els_classes)):
        means = []
        for attribute in MEAN_ATTRIBUTES:
            values = np.asarray(getattr(ibs, attribute))
            if values.shape != (params["n_phi"], params["n_ibs"]) or not np.isfinite(values).all():
                raise ValueError(f"epoch {i}: invalid {attribute} grid")
            means.append(np.mean(values))
        rows.append((lc.t[i], *lc.fluxes[i], *lc.indexes[i], lc.r_sps[i], lc.r_pes[i],
                     electrons.ntot, electrons.ntot_comov, *means))
    return np.asarray(rows)
