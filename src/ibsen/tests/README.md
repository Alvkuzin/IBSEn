# Numerical regression tests

Install the package and test dependencies in your Python environment, from the
repository root:

```sh
python -m pip install -e '.[test]'
python -m pytest
```

The tests live in `src/ibsen/tests/` and are installed as `ibsen.tests`.
Paths such as `reference/orbit/` below are relative to that directory. The case
modules locate their references relative to their own files, so they also work
when run outside the repository.

The wheel includes the test scripts, this README, and all reference tables.
After installing `ibsen[test]`, run the installed suite from any directory with:

```sh
python -m pytest --pyargs ibsen.tests
python -m pytest -p ibsen.tests.pytest_plugin --pyargs ibsen.tests.test_absorption --run-gg-direct -v
```

For the optional diagnostics, `-p` loads the custom command-line option before
pytest parses it. Repository file-path commands discover this option automatically.

The file-path commands below assume you are in the repository root.
`renew_tests.py` remains a repository-root script; use it with the editable
installation above to update the references in `src/ibsen/tests/reference/`.

## Orbit

Pytest runs two orbit cases, PSRB and LS5039. The explicit inputs are in
`orbit_cases.py`. Both
initialize `Orbit` with `n=50` and compare the resulting arrays with the saved
50-row text tables in `reference/orbit/`.

Each table has these columns:

| Column | Meaning | Unit |
| --- | --- | --- |
| `ttab` | Time relative to periastron | s |
| `xtab_p` | Pulsar barycentric X coordinate | cm |
| `ytab_p` | Pulsar barycentric Y coordinate | cm |
| `rtab` | Star-to-pulsar separation | cm |
| `nu_truetab` | True anomaly | rad |

The tables retain the initialization grid exactly. Currently this
uses eccentric anomalies from -2.5 pi to +2.5 pi, covering more than one orbit.
The line-of-sight parameters are explicit, but these five arrays do not test
line-of-sight calculations.

Comparisons use relative tolerance `1e-5`, plus absolute tolerances of `1e-6`
seconds, 1 cm, or `1e-12` radians as appropriate. Shapes must match and all
values must be finite. These are regression tolerances.
Saved results record current behavior; they are not
an independent proof that the physics is correct.

Run only one system with:

```sh
python -m pytest src/ibsen/tests/test_orbit.py -k ls5039
```

## Winds

`winds_cases.py` defines ten PSRB cases. Orbital and stellar system values come
from `get_parameters("psrb")`; all remaining `Pulsar`, `OpticalStar`, and `Winds`
constructor settings are explicit. Each table contains 101 evenly spaced times
from -100 to +100 days relative to periastron, saved in seconds.

The baseline is `f_d=1000`, `delta=0.02`, `np_disk=3`, `height_exp=0.5`,
`f_p=0.1`, `orientation=None`, `hyst=False`, and `alpha_interaction=0`.
Each variant starts from that baseline:

| Case | Change from baseline |
| --- | --- |
| `psrb_base` | None |
| `psrb_fd_100` | `f_d=100` |
| `psrb_delta_004` | `delta=0.04` |
| `psrb_np_disk_35` | `np_disk=3.5` |
| `psrb_height_exp_025` | `height_exp=0.25` |
| `psrb_fp_02` | `f_p=0.2` |
| `psrb_flow` | `orientation="flow"` |
| `psrb_flow_p` | `orientation="flow_p"` |
| `psrb_alpha_1` | `alpha_interaction=1` (hysteresis remains off) |
| `psrb_hyst` | `hyst=True` and `alpha_interaction=1` |

The `psrb_alpha_1` result should equal the baseline: interaction strength is
inactive while hysteresis is off. The hysteresis case uses 601 precalculation
times from -200 to +100 days, with `k_time=1` and `incl_puls_vel=False`.
That fixed earlier history is part of the reference calculation.

Tables in `reference/winds/` have these columns:

| Column | Meaning | Unit |
| --- | --- | --- |
| `t` | Time relative to periastron | s |
| `r_pe` | Pulsar-to-apex distance from `Winds.dist_pe` | cm |
| `b_apex` | Sum of pulsar and stellar magnetic fields at the apex | G |
| `u_apex` | Stellar photon energy density at the apex | erg/cm^3 |
| `vec_x`, `vec_y`, `vec_z` | Components of the selected displacement vector | cm |

For `orientation=None`, the vector is `Orbit.vector_sp(t)` (star to pulsar).
For `flow` or `flow_p`, it is `Winds.vec_pe_3d(t, orientation=...)` (pulsar to
apex). These are different displacement conventions, recorded in each header.
The orientation is also passed to the distance, magnetic-field, and photon-field
methods. Epochs are evaluated individually using their scalar-time interfaces.

Both magnetic fields use the linear model: the pulsar has `b_ref=1 G` at
`r_b_ref=1e13 cm`; the star has `b_ref=1 G` at its surface. This makes both
contributions to `b_apex` nonzero. Wind comparisons use relative tolerance
`1e-5`, with absolute tolerances of `1e-6 s`, `1 cm`, `1e-12 G`, and
`1e-12 erg/cm^3`. All shapes must match and all numbers must be finite.

Run just the wind tests with:

```sh
python -m pytest src/ibsen/tests/test_winds.py -v
```

## IBS

`ibs_cases.py` defines nine PSRB cases using the base wind configuration from
`winds_cases.py`: at each of -50, -15, and +20 days, it constructs a 2D IBS
with no orientation option, a 3D IBS with `orientation=None`, and a 3D IBS
with `orientation="flow_p"`.

Both classes receive `n=15` (the constructor's name for `n_ibs`). The 2D class
reflects its one-horn grid to produce 30 points. The 3D class receives
`n_phi=13` and produces arrays of shape `(13, 15)`. Other explicit settings
are `s_max=1`, `gamma_max=3`, `s_max_g=4`, and `coef_quench=0`.
The 2D line-of-sight setting remains `include_incl_in_los=False`; the 3D
shielding setting is `shield_star=0`.

Each case has one file in `reference/ibs/`, containing one numerical table
on the normal point grid:

| Columns | Units |
| --- | --- |
| `s`, `x`, `y`, `r`, `r1` | cm |
| `theta`, `theta1`, `scattering_angle`, `scattering_angle_comoving` | rad |
| `g`, `dopl_star`, `dopl`, `gg_abs_1TeV` | dimensionless |
| `ug`, `ug_comov_iso`, `ug_comov_ani` | erg/cm^3 |
| `b`, `b_comov` | G |
| `T_s_eff`, `T_s_eff_comov_iso`, `T_s_eff_comov_ani` | K |

Each table has 21 columns and 30 rows for 2D or 15 rows for 3D.
Temperatures use the point-based attributes directly. For 3D,
quantities are calculated on the full grid and then averaged over phi with
the arithmetic mean along axis 0. Comoving transformations happen before
averaging. `theta1` uses the phi-independent `theta1_1horn` array because
the 3D class has no `theta1` attribute. `scattering_angle_comoving` is the
table name for the existing `scattering_angle_comov` attribute.

The coordinates `x` and `y` are barycentric; `r` and `r1` are distances from
the pulsar and star. Absorption is the transmission `exp(-tau)`, evaluated
by `gg_abs` at `1e12 eV` (1 TeV) using the PSRB opacity table and
`analyt=False`. This records the existing 3D absorption approximation.

**Recorded implementation limitation:** although `orientation="flow_p"`
is passed to `IBS3D`, its current axis construction calls `vec_pe_3d` without
forwarding that orientation, so the axis uses the default `flow` vector.
The regression references deliberately record this current behavior;
it is likely a bug, I will fix it soon.

Comparisons use relative tolerance `1e-5`, plus absolute tolerances of
1 cm, `1e-6 K`, or `1e-12` in the other columns' units. All values must
be finite and each table must have the expected shape. Use `np.loadtxt(path)`
to read a reference file, or `load_ibs_tables(path)["points"]`.

Run the IBS cases, generate candidate tables, or replace only their references:

```sh
python -m pytest src/ibsen/tests/test_ibs.py -v
python renew_tests.py --module ibs --output-dir /tmp/ibsen-ibs-candidates
python renew_tests.py --module ibs
```

For example, `--case psrb_3d_flow_p_m15d` selects just the -15-day 3D flow-p case.

## ElectronsOnIBS

`electrons_cases.py` defines five PSRB cases, all at +20 days, using cooling
laws `no`, `stat_apex`, `stat_ibs`, `stat_mimic`, and `adv`.
Each constructs `IBS3D` with the baseline wind parameters, `orientation=None`,
`n=15`, and `n_phi=13`. Other geometry settings match the IBS regression cases.

Only the cooling law changes between cases. All use exponential-cutoff power-law
injection (`p_e=2`, `ecut=1e12 eV`), `eta_a=eta_syn=eta_ic=1`, and
`norm_e=1e35/1.6e-12` with `epow_norm_e=1`, matching the normalization and
adiabatic setting in the original smoke test. All electron constructor settings
are explicit in the case module and recorded in the file headers.
`calculate(require_lorentz=True)` computes both lab and comoving distributions.

Each case has a file in `reference/electrons/` with three `# Table: ...` sections:

| Section | Columns | Rows |
| --- | --- | --- |
| `spectrum_lab` | `e_vals` [eV], `dNe_de_mid_sum` [1/eV] | Depends on cooling law |
| `spectrum_comov` | `e_vals_comov` [eV], `dNe_de_mid_comov_sum` [1/eV] | Depends on cooling law |
| `spatial_population` | `s_mid` [cm], `n_i_mid_sum_phi`, `n_i_mid_comov_sum_phi` [counts] | 14 |

The native lab/comoving energy-grid lengths are 2019/2264 for `no`,
678/760 for each stationary law, and 826/926 for `adv`. Each case retains
those native grids without resampling.

The midpoint distributions have shape `(13, 14, number_of_energies)`.
The spectra are summed over both spatial axes `(0, 1)` and saved on their own
native energy grids. The lab grid spans `1e8` to `5.1e14 eV`; the comoving
grid is extended by the Lorentz transformation. The population arrays have
shape `(13, 14)` and are summed only over phi (axis 0). Their spatial coordinate
is `ibs.s_mid[0, :]`, after checking that every phi slice shares that grid.
These are sums, not phi averages. No extra segment lengths or angular weights
are applied: `dNe_de_mid` already includes segment lengths.

The test checks all three tables with relative tolerance `1e-5`. The spectra
and populations have zero absolute tolerance so tiny high-energy values are
still compared; arclength has absolute tolerance 1 cm. Shapes and finite values
are checked, and calculated spectra and populations must be nonnegative.
Use `load_electron_tables(path)` to read the three tables as a dictionary.

```sh
python -m pytest src/ibsen/tests/test_electrons.py -v
python renew_tests.py --module electrons --output-dir /tmp/ibsen-electron-candidates
python renew_tests.py --module electrons
```

For one cooling law, use e.g. `--case psrb_electrons_p20d_adv` or
`--case psrb_electrons_p20d_stat_ibs`.

## Spectrum

`spectrum_cases.py` defines three SED cases at +20 days with `method="apex"`,
`"simple"`, and `"full"`. They use the baseline PSRB outflows, 3D IBS
(`n=15`, `n_phi=13`, `orientation=None`), and the electron setup with
`cooling="stat_ibs"`. Synchrotron and inverse Compton emission are enabled,
along with Doppler boosting, photoelectric absorption (`nh_tbabs=0.8`), and
gamma-gamma absorption from the PSRB table.

The energy grid follows the original smoke test: two logarithmic ranges,
`300/1.2` to `1e4*1.2 eV` and `4e11/1.2` to `1e13*1.2 eV`, each using
37 nodes per decade. Each file in `reference/spectrum/` contains:

| Column | Meaning | Unit |
| --- | --- | --- |
| `e_ph` | Photon energy | eV |
| `sed_total` | Total absorbed, boosted synchrotron + IC SED | erg/s/cm^2 |
| `abs_tot_mean` | Arithmetic mean of `SpectrumIBS.abs_tot` over both spatial axes | dimensionless |

`abs_tot_mean` is the mean transmission `exp(-tau)`, including both absorption
mechanisms. It is evaluated on the IBS midpoint grid even for the apex SED
method; it is not an optical depth or an emission-weighted mean.

```sh
python -m pytest src/ibsen/tests/test_spectrum.py -v
python renew_tests.py --module spectrum
```

## Light curve

`lightcurve_cases.py` defines one PSRB light curve with `method="simple"`,
3D IBS (`n_ibs=15`, `n_phi=13`, `orientation=None`), and
`cooling="stat_mimic"`. It uses the same baseline winds, electron injection,
magnetic fields, boosting, and absorption settings as the other pipeline tests.
It calculates 71 times with `orbit.timegrid_lin_in_anomaly(n_samples=71)`;
the API parameter is named `n_samples`, not `n`. The default time bounds span
almost one full orbit around periastron, with equal steps in true anomaly.

The reference in `reference/lightcurve/` contains 17 columns:

- Time in seconds.
- Flux and photon index in 1–10 keV and 0.4–10 TeV. As in the smoke test,
  the X-ray flux is an energy flux (`epow=1`, erg/s/cm^2), and the TeV flux
  is a photon flux (`epow=0`, 1/s/cm^2). Both index bands match the flux bands.
- `r_sp` and `r_pe` in cm.
- Total electron populations `ntot_lab` and `ntot_comov`.
- Mean `b` and `b_comov` in G.
- Mean `ug`, `ug_comov_iso`, and `ug_comov_ani` in erg/cm^3.
- Mean `dopl`.
- Mean `scattering_angle` and `scattering_angle_comov` in radians.

All field, Doppler-factor, and angle means are unweighted arithmetic means over
both axes of the IBS **point** arrays. Electron totals come from the lab and
comoving segment populations. Calculation is serial for reproducibility.

```sh
python -m pytest src/ibsen/tests/test_lightcurve.py -v
python renew_tests.py --module lightcurve
```

## Absorption

`absorption_cases.py` follows the interfaces used in `tutorials/absorption.ipynb`,
with PSRB system parameters. The three reference files use 50 samples each:

| Case | Independent coordinates | Saved optical depths |
| --- | --- | --- |
| `psrb_photoelectric` | `E_eV`, 300 eV–100 keV, logarithmic | `tau_Nh_0p1`, `tau_Nh_1` |
| `psrb_gg_periastron` | `E_eV`, 10 GeV–100 TeV, logarithmic | `tau` at `(r_periastr, 0, 0)` |
| `psrb_gg_orbit` | `x`, `y`, from 50 equal-true-anomaly orbital samples | `tau` at 1 TeV, `z=0` |

Photoelectric absorption uses `abs_photoel` with `abund="wilm"` and `Nh=0.1`
or `1.0` in its usual column-density units. That function returns transmission,
so the saved optical depth is `-log(transmission)`. Gamma-gamma references use
`gg_tab_d_mu(..., filename="gg_abs_psrb_d_mu", what_return="tau")`, with
`orbit.unit_los` as the observer direction. Positions are star-relative
`Orbit.x/y` coordinates with `z=0`.

```sh
python -m pytest src/ibsen/tests/test_absorption.py -v
python renew_tests.py --module absorption
```

Two additional diagnostic tests are skipped by default. Enable them with:

```sh
python -m pytest src/ibsen/tests/test_absorption.py --run-gg-direct -v
```

They evaluate `tau_gg_iso_2d(fast=False)` at the saved periastron energies and
the saved orbital positions, using `eg=E_eV/5.11e5` as in the notebook. Each
comparison prints the maximum and mean absolute discrepancy
`abs(tau_direct - tau_tab)`, plus maximum and mean relative discrepancy
`abs(tau_direct - tau_tab)/tau_tab` for points where saved `tau_tab > 1e-4`.
The number of points used for relative statistics is printed. Absolute
statistics include all 50 points. Reports are printed even with pytest's
normal output capture enabled.

These optional tests check finite, nonnegative values but deliberately impose
no closeness threshold. The ordinary regression tests still compare freshly
interpolated values with the saved tables at relative tolerance `1e-5`.

## Regenerating references

All reference tables are saved with `%.8e` formatting (eight digits after the
decimal point). Regression and analytical tests use relative tolerance `1e-5`.

Normal test runs never write reference tables. To generate candidate orbit results
in a separate directory for inspection:

```sh
python renew_tests.py --module orbit --output-dir /tmp/ibsen-orbit-candidates
```

After reviewing an intentional change, replace the references with:

```sh
python renew_tests.py --module orbit
```

The same goes for all other modules. 


With no arguments, `python renew_tests.py` regenerates all seven suites. When
multiple suites are selected with `--output-dir`, it creates a subdirectory
for each selected module (`orbit/`, `winds/`, `ibs/`, `electrons/`, `spectrum/`,
`lightcurve/`, or `absorption/`). With just one suite
selected, tables go directly into
the specified directory.

Use `--case psrb`, `--case ls5039`, or a wind case such as `--case psrb_hyst`
to update only that case. Review and commit
updated tables together with the code change; Git retains the old versions.
The table headers record the input parameters, units, and numerical dependency
versions. The comparison reads numerical columns, not header text.
