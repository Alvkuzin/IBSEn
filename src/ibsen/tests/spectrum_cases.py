"""PSRB SED and mean transmission references at +20 days."""

from pathlib import Path

import numpy as np

from ibsen import IBS3D, ElectronsOnIBS, Orbit, OpticalStar, Pulsar, SpectrumIBS, Winds
from ibsen.get_obs_data import get_parameters
from ibsen.utils import loggrid
from .electrons_cases import electron_case_parameters


REFERENCE_DIR = Path(__file__).parent / "reference" / "spectrum"
SPECTRUM_CASES = {f"psrb_sed_p20d_{method}": {"method": method}
                  for method in ("apex", "simple", "full")}
ENERGIES = np.concatenate((loggrid(3e2 / 1.2, 1e4 * 1.2, 37),
                           loggrid(4e11 / 1.2, 1e13 * 1.2, 37)))
COLUMNS = ("e_ph", "sed_total", "abs_tot_mean")
UNITS = ("eV", "erg/s/cm^2", "1")


def spectrum_settings(method):
    return {
        "sys_name": "psrb", "sys_params": None, "distance": get_parameters("psrb")["D"],
        "mechanisms": ["syn", "ic"], "method": method, "ic_ani": True,
        "delta_power": 4.0, "lorentz_boost": True, "ani_lorentz_boost": True,
        "abs_photoel": True, "abs_gg": True, "nh_tbabs": 0.8,
        "mode": "int", "ne_mult": 0.7, "ic_approx": "KN", "nEed_syn": 51, "nEed_ic": 171,
    }


def spectrum_case_parameters(case_name):
    return {
        "electron_setup": electron_case_parameters("psrb_electrons_p20d_stat_ibs"),
        "spectrum": spectrum_settings(SPECTRUM_CASES[case_name]["method"]),
        "e_ph": ENERGIES.tolist(),
    }


def calculate_spectrum_table(case_name):
    params = spectrum_case_parameters(case_name)
    setup = params["electron_setup"]
    env = setup["environment"]
    winds = Winds(orbit=Orbit(**env["orbit"]), star=OpticalStar(**env["star"]),
                  pulsar=Pulsar(**env["pulsar"]), **env["winds"])
    ibs = IBS3D(winds=winds, **setup["ibs"])
    electrons = ElectronsOnIBS(ibs=ibs, **setup["electrons"])
    electrons.calculate(**setup["calculate"])
    spec = SpectrumIBS(els=electrons, **params["spectrum"])
    spec.calculate(e_ph=np.asarray(params["e_ph"]))
    # if spec.abs_tot.shape != ibs.s_mid.shape + (len(ENERGIES),):
    #     raise ValueError(f"{case_name}: unexpected absorption grid {spec.abs_tot.shape}")
    if not np.isfinite(spec.abs_tot).all():
        raise ValueError(f"{case_name}: non-finite absorption values")
    return np.column_stack((spec.e_ph, spec.sed, np.mean(spec.abs_tot, axis=tuple(range(spec.abs_tot.ndim - 1)))
                            ))
