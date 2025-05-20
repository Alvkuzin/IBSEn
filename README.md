# IBSEn: calculates the IBS Emissivity for PSR B1259-63.

IBSEn: IntraBinary Shock Emission n-not-meaning-anything is collection of Python scripts for processing, modeling, and visualizing observational data of the gamma-ray binary pulsar PSR B1259-63. This project includes tools to handle observational data, compute intrabinary shock (IBS) emissivity: spectra and light curves, and fit theoretical models to light curves and spectra.

---

## Project Structure

The project contains several folders:

1. ObsData/ Observed light curves and spectral data
2. TabData/ Pre-tabulated numerical data
3. Outputs/ Output of model fits and results

and a bunch of Python sctipts:

1. For reading the observational data:
   - `GetObsData.py`  Collects and organizes observational data
2. Some general scripts:
   - `Orbit.py`  Orbital characteristics and solutions
   - `TransportShock.py`  Electron transport equation solvers
   - `Absorbtion.py`  Photoelectric and γ-γ absorption scripts
3. Directly related to the emission calculation:
   - `ElEv.py`  Electron evolution on the IBS
   - `SpecIBS.py`  IBS spectrum calculation (Synchrotron + IC)
   - `LC.py`  PSRB light curve model
   - `DopplBoost\_fit\_plots.py` # Fits model to observed light curves
4. For displaying results or data:
   - `Show_ObsLightCurves.py`  Visualizes observed light curves 
   - `Show\_precalc\_LC_fits.py`  Displays model fits to data
   - `Show_spectra.py`  Compares observed and theoretical spectra


---

## Usage

To collect and clean the raw observational data:

```bash
python GetObsData.py
```

To calculate and fit light curves:

```bash
python DopplBoost_fit_plots.py
```

To visualize observed and theoretical light curves:

```bash
python Show_ObsLightCurves.py
python Show_precalc_LC_fits.py
```

To display spectra:

```bash
python Show_spectra.py
```

## Requirements 

This project requires Python 3, [Naima](https://naima.readthedocs.io/en/latest/index.html), and some standard scientific libraries. You can install them (if needed, Naima excluded) with:

```bash
pip install -r requirements.txt
```


Additional dependencies may apply depending on plotting or fitting routines used.

## Notes

This repository is currently not organized as a Python package.

Data and outputs are stored locally in the provided directories.

The codebase is tailored to observational analysis of PSR B1259-63 but can be adapted to other similar systems.

## License 

This project is licensed under the MIT License. See the LICENSE file for details.
---


