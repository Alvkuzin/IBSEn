# IBSEn: calculates the IBS Emissivity. Now, only for PSRB.

A collection of Python scripts for processing, modeling, and visualizing observational data of the gamma-ray binary pulsar PSR B1259-63. This project includes tools to handle observational data, compute intrabinary shock (IBS) emissivity, and fit theoretical models to light curves and spectra.

---

## Project Structure

Project/
├── ObsData/ # Observed light curves and spectral data
├── TabData/ # Pre-tabulated numerical data
├── Outputs/ # Output of model fits and results
├── GetObsData.py # Collects and organizes observational data
├── Orbit.py # Orbital characteristics and solutions
├── TransportShock.py # Electron transport equation solvers
├── Absorbtion.py # Photoelectric and γ-γ absorption scripts
├── ElEv.py # Electron evolution on the IBS
├── SpecIBS.py # IBS spectrum calculation (Synchrotron + IC)
├── LC.py # PSRB light curve model
├── DopplBoost_fit_plots.py # Fits model to observed light curves
├── Show_ObsLightCurves.py # Visualizes observed light curves
├── Show_precalc_LC_fits.py # Displays model fits to data
└── Show_spectra.py # Compares observed and theoretical spectra

---

## Usage

To collect and clean the raw observational data:

```bash
python GetObsData.py

To calculate and fit light curves:


python DopplBoost_fit_plots.py

To visualize observed and theoretical light curves:

python Show_ObsLightCurves.py
python Show_precalc_LC_fits.py

To display spectra:

python Show_spectra.py
```

## Requirements 

This project requires Python 3 and some standard scientific libraries. You can install them (if needed) with:

```bash
pip install numpy scipy matplotlib
```


Additional dependencies may apply depending on plotting or fitting routines used.

## Notes

This repository is currently not organized as a Python package.

Data and outputs are stored locally in the provided directories.

The codebase is tailored to observational analysis of PSR B1259-63 but can be adapted to other similar systems.

## License 

This project is licensed under the MIT License. See the LICENSE file for details.
---


