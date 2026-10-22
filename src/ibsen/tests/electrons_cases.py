"""Spatially summed ElectronsOnIBS references for the baseline PSRB winds."""

from io import StringIO
from pathlib import Path

import numpy as np

from ibsen import IBS3D, ElectronsOnIBS, Orbit, OpticalStar, Pulsar, Winds
from .ibs_cases import N_IBS, N_PHI
from .winds_cases import DAY, winds_case_parameters


REFERENCE_DIR = Path(__file__).parent / "reference" / "electrons"
ELECTRON_CASES = {
    f"psrb_electrons_p20d_{cooling}": {"time_days": 20.0, "cooling": cooling}
    for cooling in ("no", "stat_apex", "stat_ibs", "stat_mimic", "adv")
}

# Expected lengths of the native energy grids for these fixed settings.
# The comoving grid extends beyond the lab grid and has its own row count.
ENERGY_GRID_SIZES = {
    "no": (2019, 2264),
    "stat_apex": (678, 760),
    "stat_ibs": (678, 760),
    "stat_mimic": (678, 760),
    "adv": (826, 926),
}


def electron_table_specs(case_name):
    n_lab, n_comov = ENERGY_GRID_SIZES[ELECTRON_CASES[case_name]["cooling"]]
    return {
        "spectrum_lab": {
            "shape": (n_lab, 2),
            "columns": ("e_vals", "dNe_de_mid_sum"),
            "units": ("eV", "1/eV"),
        },
        "spectrum_comov": {
            "shape": (n_comov, 2),
            "columns": ("e_vals_comov", "dNe_de_mid_comov_sum"),
            "units": ("eV", "1/eV"),
        },
        "spatial_population": {
            "shape": (N_IBS - 1, 3),
            "columns": ("s_mid", "n_i_mid_sum_phi", "n_i_mid_comov_sum_phi"),
            "units": ("cm", "1", "1"),
        },
    }


def electron_case_parameters(case_name):
    base = winds_case_parameters("psrb_base")
    environment = {key: base[key] for key in ("orbit", "star", "pulsar", "winds")}
    return {
        "environment": environment,
        "ibs": {
            "t_to_calculate_beta_eff": ELECTRON_CASES[case_name]["time_days"] * DAY,
            "n": N_IBS, "n_phi": N_PHI, "orientation": None,
            "s_max": 1.0, "gamma_max": 3.0, "s_max_g": 4.0,
            "coef_quench": 0.0, "shield_star": 0.0, "abs_gg_filename": "psrb",
        },
        "electrons": {
            "cooling": ELECTRON_CASES[case_name]["cooling"],
            "to_inject_e": "ecpl", "to_inject_theta": "3d",
            "ecut": 1e12, "p_e": 2.0, "norm_e": 1e35 / 1.6e-12, "epow_norm_e": 1.0,
            "beta_e": 1.0, "ebr": 1e12, "delta_p_e": 0.5, "n_e_cut": 0.0,
            "eta_a": 1.0, "eta_syn": 1.0, "eta_ic": 1.0,
            "emin": 1e9, "emax": 5.1e14, "emin_grid": 1e8, "emax_grid": 5.1e14,
            "to_cut_e": True, "to_cut_theta": False,
            "where_cut_theta": np.pi / 2, "use_mimic_eta": False,
        },
        "calculate": {"require_lorentz": True, "to_return": False},
    }


def calculate_electron_tables(case_name):
    params = electron_case_parameters(case_name)
    env = params["environment"]
    winds = Winds(
        orbit=Orbit(**env["orbit"]), star=OpticalStar(**env["star"]),
        pulsar=Pulsar(**env["pulsar"]), **env["winds"],
    )
    ibs = IBS3D(winds=winds, **params["ibs"])
    electrons = ElectronsOnIBS(ibs=ibs, **params["electrons"])
    electrons.calculate(**params["calculate"])

    spatial_shape = (N_PHI, N_IBS - 1)
    specs = electron_table_specs(case_name)
    n_lab = specs["spectrum_lab"]["shape"][0]
    n_comov = specs["spectrum_comov"]["shape"][0]
    expected_shapes = {
        "e_vals": (n_lab,), "e_vals_comov": (n_comov,),
        "dNe_de_mid": spatial_shape + (n_lab,),
        "dNe_de_mid_comov": spatial_shape + (n_comov,),
        "n_i_mid": spatial_shape, "n_i_mid_comov": spatial_shape,
    }
    for name, shape in expected_shapes.items():
        values = np.asarray(getattr(electrons, name))
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{case_name}: invalid {name}, expected finite array of shape {shape}")
        if np.any(values < 0):
            raise ValueError(f"{case_name}: negative values in {name}")
    for name in ("e_vals", "e_vals_comov"):
        if not np.all(np.diff(getattr(electrons, name)) > 0):
            raise ValueError(f"{case_name}: {name} must be strictly increasing")

    # The arclength grid is shared across phi; populations belong to segments.
    if ibs.s_mid.shape != spatial_shape or not np.isfinite(ibs.s_mid).all():
        raise ValueError(f"{case_name}: invalid midpoint arclength grid")
    s_mid = ibs.s_mid[0, :]
    np.testing.assert_allclose(
        ibs.s_mid, np.broadcast_to(s_mid, spatial_shape),
        rtol=1e-5, atol=1.0, equal_nan=False,
        err_msg="All phi slices must share the saved arclength grid",
    )
    # dNe_de_mid already contains segment lengths: sum, without extra ds or
    # phi weights. Each spectrum retains its own native energy grid.
    return {
        "spectrum_lab": np.column_stack((
            electrons.e_vals, np.sum(electrons.dNe_de_mid, axis=(0, 1)),
        )),
        "spectrum_comov": np.column_stack((
            electrons.e_vals_comov, np.sum(electrons.dNe_de_mid_comov, axis=(0, 1)),
        )),
        "spatial_population": np.column_stack((
            s_mid, np.sum(electrons.n_i_mid, axis=0),
            np.sum(electrons.n_i_mid_comov, axis=0),
        )),
    }


def load_electron_tables(path):
    """Read the three native-grid tables from an electron reference file."""
    blocks = {}
    current = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("# Table: "):
            current = line.removeprefix("# Table: ")
            if current in blocks:
                raise ValueError(f"Duplicate electron table: {current}")
            blocks[current] = []
        elif line.strip() and not line.startswith("#"):
            if current is None:
                raise ValueError("Electron reference data must belong to a table section")
            blocks[current].append(line)
    if set(blocks) != {"spectrum_lab", "spectrum_comov", "spatial_population"}:
        raise ValueError(f"Unexpected electron table sections: {tuple(blocks)}")
    return {name: np.loadtxt(StringIO("\n".join(lines)), ndmin=2)
            for name, lines in blocks.items()}
