"""Deliberately regenerate reference tables, or write candidate tables."""

import argparse
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path

import numpy as np

from ibsen.tests import (absorption_cases, electrons_cases, ibs_cases, lightcurve_cases,
                         orbit_cases, spectrum_cases, winds_cases)


SUITES = {
    "orbit": {
        "module": orbit_cases,
        "cases": orbit_cases.ORBIT_CASES,
        "calculate": orbit_cases.calculate_orbit_table,
        "parameters": lambda name: orbit_cases.ORBIT_CASES[name],
        "shape": (50, len(orbit_cases.COLUMNS)),
        "description": lambda name: "Coordinates: pulsar barycentric x/y; r is star-to-pulsar separation.",
    },
    "winds": {
        "module": winds_cases,
        "cases": winds_cases.WIND_CASES,
        "calculate": winds_cases.calculate_winds_table,
        "parameters": winds_cases.winds_case_parameters,
        "shape": (len(winds_cases.TIMES), len(winds_cases.COLUMNS)),
        "description": lambda name: (
            "b_apex = B_pulsar + B_star [G]; u_apex = stellar photon energy density [erg/cm^3]. "
            "Vector: " + ("star-to-pulsar (orbit.vector_sp)." if
                winds_cases.WIND_CASES[name]["orientation"] is None else
                "pulsar-to-apex (winds.vec_pe_3d).")
        ),
    },
    "ibs": {
        "module": ibs_cases,
        "cases": ibs_cases.IBS_CASES,
        "calculate": ibs_cases.calculate_ibs_tables,
        "parameters": ibs_cases.ibs_case_parameters,
        "table_specs": ibs_cases.ibs_table_specs,
        "description": lambda name: (
            "All quantities on the point grid; x/y are barycentric; r/r1 are pulsar/star distances. "
            "gg_abs_1TeV is exp(-tau) at 1e12 eV; scattering_angle_comoving uses scattering_angle_comov. "
            + ("3D arrays: arithmetic mean over phi (axis=0); theta1 uses theta1_1horn."
               if ibs_cases.IBS_CASES[name]["ndim"] == 3 else "2D arrays: both horns in native order.")
            + (" Current flow_p behavior: the IBS axis still uses the default flow vector."
               if ibs_cases.IBS_CASES[name]["orientation"] == "flow_p" else "")
        ),
    },
    "electrons": {
        "module": electrons_cases,
        "cases": electrons_cases.ELECTRON_CASES,
        "calculate": electrons_cases.calculate_electron_tables,
        "parameters": electrons_cases.electron_case_parameters,
        "table_specs": electrons_cases.electron_table_specs,
        "description": lambda name: (
            "Spectra: sum dNe_de_mid and dNe_de_mid_comov over phi and spatial segments "
            "(axes 0,1), on e_vals and e_vals_comov respectively. "
            "Populations: sum n_i_mid and n_i_mid_comov over phi (axis 0), versus s_mid. "
            "Unweighted sums; no additional ds factors or interpolation."
        ),
    },
    "spectrum": {
        "module": spectrum_cases,
        "cases": spectrum_cases.SPECTRUM_CASES,
        "calculate": spectrum_cases.calculate_spectrum_table,
        "parameters": spectrum_cases.spectrum_case_parameters,
        "shape": (len(spectrum_cases.ENERGIES), len(spectrum_cases.COLUMNS)),
        "description": lambda name: "Total absorbed SED; abs_tot_mean is mean transmission exp(-tau) over both IBS midpoint axes.",
    },
    "lightcurve": {
        "module": lightcurve_cases,
        "cases": lightcurve_cases.LIGHTCURVE_CASES,
        "calculate": lightcurve_cases.calculate_lightcurve_table,
        "parameters": lightcurve_cases.lightcurve_case_parameters,
        "shape": (lightcurve_cases.N_TIMES, len(lightcurve_cases.COLUMNS)),
        "description": lambda name: (
            "71 equal-true-anomaly epochs. X-ray energy flux; TeV photon flux; indexes fit dN/dE. "
            "Fields, Doppler factor, and scattering angles are arithmetic means over all IBS points. "
            "ntot_lab/comov are total particle populations."
        ),
    },
    "absorption": {
        "module": absorption_cases,
        "cases": absorption_cases.ABSORPTION_CASES,
        "calculate": absorption_cases.calculate_absorption_tables,
        "parameters": absorption_cases.absorption_case_parameters,
        "table_specs": absorption_cases.absorption_table_specs,
        "description": lambda name: (
            "Photoelectric tau=-log(abs_photoel); Nh=0.1 and 1.0 in the function's 1e22 units; abund=wilm."
            if name == "psrb_photoelectric" else
            "Gamma-gamma optical depth from gg_tab_d_mu(what_return='tau'); coordinates are star-relative, z=0; n_los=orbit.unit_los."
        ),
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module", choices=tuple(SUITES),
        help="Select a calculation module (default: all, or the suites containing --case).",
    )
    parser.add_argument(
        "--case", choices=tuple(name for suite in SUITES.values() for name in suite["cases"]), action="append",
        help="Regenerate only this case; repeat to select several (default: all).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Candidate directory; multiple suites use per-module subdirectories. Default: replace references.",
    )
    args = parser.parse_args()
    selected = {name: suite for name, suite in SUITES.items() if
                (args.module is None or name == args.module) and
                (args.case is None or any(case in suite["cases"] for case in args.case))}
    if args.case and any(not any(case in suite["cases"] for suite in selected.values()) for case in args.case):
        parser.error("A selected case does not belong to the selected module")

    # Finish and validate every calculation before replacing any reference.
    tables = []
    for module_name, suite in selected.items():
        for name in suite["cases"]:
            if args.case is not None and name not in args.case:
                continue
            calculated = suite["calculate"](name)
            if "table_specs" in suite:
                specs = suite["table_specs"](name)
                sections = calculated
            else:
                specs = {"": {"shape": suite["shape"], "columns": suite["module"].COLUMNS,
                              "units": suite["module"].UNITS}}
                sections = {"": calculated}
            if sections.keys() != specs.keys():
                raise ValueError(f"{name}: unexpected table sections")
            for section, table in sections.items():
                if table.shape != specs[section]["shape"] or not np.isfinite(table).all():
                    raise ValueError(f"{name}/{section}: refusing to save an invalid table")
            tables.append((module_name, name, sections, specs))

    dependencies = {name: version(name) for name in ("numpy", "scipy", "astropy")}
    for module_name, name, sections, specs in tables:
        suite = selected[module_name]
        module = suite["module"]
        output_dir = module.REFERENCE_DIR
        if args.output_dir is not None:
            output_dir = args.output_dir / module_name if len(selected) > 1 else args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        header = "\n".join([
            f"{module_name.capitalize()} regression reference: {name}",
            "Parameters (cgs, angles in rad except *_deg): " + json.dumps(suite["parameters"](name), sort_keys=True),
            "Dependencies: " + json.dumps(dependencies, sort_keys=True),
            suite["description"](name),
        ])
        destination = output_dir / f"{name}.txt"
        output = StringIO()
        output.write("".join(f"# {line}\n" for line in header.splitlines()))
        for section, table in sections.items():
            spec = specs[section]
            if section:
                output.write(f"# Table: {section}\n")
            columns = " ".join(f"{column}[{unit}]" for column, unit in zip(spec["columns"], spec["units"]))
            np.savetxt(output, table, fmt="%.8e", header=columns)
        destination.write_text(output.getvalue())
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
