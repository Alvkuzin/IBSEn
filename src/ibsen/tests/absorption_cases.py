"""Photoelectric and tabulated gamma-gamma optical depths for PSRB."""

from pathlib import Path

import numpy as np

from ibsen import Orbit
from ibsen.absorption import absorption
from ibsen.get_obs_data import get_parameters


REFERENCE_DIR = Path(__file__).parent / "reference" / "absorption"
N_SAMPLES = 50
PHOTO_ENERGIES = np.geomspace(3e2, 1e5, N_SAMPLES)
GG_ENERGIES = np.geomspace(1e10, 1e14, N_SAMPLES)
GG_ORBIT_ENERGY = 1e12
ABSORPTION_CASES = {
    "psrb_photoelectric": {}, "psrb_gg_periastron": {}, "psrb_gg_orbit": {},
}
GG_CASES = ("psrb_gg_periastron", "psrb_gg_orbit")


def absorption_table_specs(case_name):
    if case_name == "psrb_photoelectric":
        columns, units = ("E_eV", "tau_Nh_0p1", "tau_Nh_1"), ("eV", "1", "1")
    elif case_name == "psrb_gg_periastron":
        columns, units = ("E_eV", "tau"), ("eV", "1")
    elif case_name == "psrb_gg_orbit":
        columns, units = ("x", "y", "tau"), ("cm", "cm", "1")
    else:
        raise ValueError(case_name)
    return {"values": {"shape": (N_SAMPLES, len(columns)), "columns": columns, "units": units}}


def absorption_case_parameters(case_name):
    system = get_parameters("psrb")
    orbit = {key: system[key] for key in ("T", "e", "M_s", "M_p", "nu_los", "incl_los")}
    orbit.update(sys_name="psrb", n=None)
    params = {"orbit": orbit, "R_s": system["R_s"], "T_s": system["T_s"]}
    if case_name == "psrb_photoelectric":
        params.update(E_eV=PHOTO_ENERGIES.tolist(), Nh=[0.1, 1.0], abund="wilm")
    elif case_name == "psrb_gg_periastron":
        params.update(E_eV=GG_ENERGIES.tolist(), x=float(Orbit(**orbit).r_periastr), y=0.0, z=0.0,
                      filename="gg_abs_psrb_d_mu", what_return="tau")
    elif case_name == "psrb_gg_orbit":
        params.update(E_eV=GG_ORBIT_ENERGY, z=0.0, filename="gg_abs_psrb_d_mu", what_return="tau",
                      times=Orbit(**orbit).timegrid_lin_in_anomaly(n_samples=N_SAMPLES).tolist())
    else:
        raise ValueError(case_name)
    return params


def calculate_absorption_tables(case_name):
    params = absorption_case_parameters(case_name)
    orbit = Orbit(**params["orbit"])
    if case_name == "psrb_photoelectric":
        energy = np.asarray(params["E_eV"])
        transmissions = [absorption.abs_photoel(energy, Nh=nh, abund=params["abund"])
                         for nh in params["Nh"]]
        if any(np.any((values <= 0) | (values > 1)) for values in transmissions):
            raise ValueError("Photoelectric transmission must be positive to obtain finite tau")
        table = np.column_stack((energy, *[-np.log(values) for values in transmissions]))
    elif case_name == "psrb_gg_periastron":
        energy = np.asarray(params["E_eV"])
        position = np.array([params["x"], params["y"], params["z"]])
        tau = absorption.gg_tab_d_mu(energy, position, orbit.unit_los,
                                     filename=params["filename"], what_return="tau")
        table = np.column_stack((energy, tau))
    else:
        times = np.asarray(params["times"])
        x, y = orbit.x(times), orbit.y(times)
        positions = np.column_stack((x, y, np.full_like(x, params["z"])))
        tau = absorption.gg_tab_d_mu(params["E_eV"], positions, orbit.unit_los,
                                     filename=params["filename"], what_return="tau")
        # A scalar energy still produces a trailing, one-element energy axis.
        tau = tau[..., 0]
        table = np.column_stack((x, y, tau))
    return {"values": table}


def direct_gg_tau(case_name, reference):
    """Integrate at the saved energies/positions, with the precise solver mode."""
    params = absorption_case_parameters(case_name)
    if case_name == "psrb_gg_periastron":
        samples = ((energy, params["x"], params["y"]) for energy in reference[:, 0])
    elif case_name == "psrb_gg_orbit":
        samples = ((params["E_eV"], x, y) for x, y in reference[:, :2])
    else:
        raise ValueError("Direct integration applies only to gamma-gamma cases")
    return np.array([
        absorption.tau_gg_iso_2d(
            eg=energy / 5.11e5, x=float(x), y=float(y), z=0.0,
            R_s=params["R_s"], T_s=params["T_s"],
            incl_los=params["orbit"]["incl_los"], nu_los=params["orbit"]["nu_los"], fast=False,
        ) for energy, x, y in samples
    ])
