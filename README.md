# IBSEn: calculates the IBS Emissivity for PSR B1259-63.

IBSEn: **I**ntra**B**inary **S**hock **E**mission **n**-**n**ot-mea**n**i**n**g-a**n**ythi**n**g is a collection of Python scripts for processing, modeling, and visualizing observational data of the gamma-ray binariy pulsars, especially PSR B1259-63. This project includes tools to handle observational data, compute intrabinary shock (IBS) emissivity: spectra and light curves, and (somewhere in future) fit theoretical models to light curves and spectra.


## Installation & Requirements 
### Installation

Clone and install manually. Normal installation:

```bash
git clone https://github.com/Alvkuzin/IBSEn
cd IBSEn
pip install .
```

For the installation and contributing, replace the last line with 

```bash
pip install -e .
```

After the installation, install the required packages with
```bash
pip install -r requirements.txt
```


### Requirements
This project requires:
 - Python 3 (please install yourself),
 - [Naima](https://naima.readthedocs.io/en/latest/installation.html),
 - And some standard scientific libraries. Install them, including Naima, with:

```bash
pip install -r requirements.txt
```

**Note on Naima.** If you have both Naima and numpy installed and they are not in conflict, you are good to go. 
For me though it was a problem that Naima, and if I understand correctly, requires Numpy v. < 2. But I had Numpy of the version 2...! So you can either downgrade Numpy to <2:
```bash
pip install "numpy<2.0"
```

or create an environment with Numpy < 2 and Naima installed (and some other packages which do not seem to be in conflict with Numpy or Naima):

```bash
conda env create -f environment.yml
conda activate ibsen_env
git clone https://github.com/Alvkuzin/IBSEn
cd IBSEn
pip install .
```

**But if you just want to use IBSEn for Orbit / Winds & Stars / IBS / electron evolution and NOT for specta / light curves, you're good to go without installing Naima at all.** 

Scripts now are not suited for running from the command line... To find out if the installation works, try running a very basic python script that simply initializes a lot of classes with more or less default parameters:
```bash
python test_ibsen.py
```
You should get the output of

```bash
PSR B1259-63 periastron dist [au] =  0.8351387469104009
----- at t=20 days after periastron -----
effective beta =  0.04164299814865532
IBS opening angle =  2.4497037807657196
tot number of e on IBS =  7.309978160194949e+36
from spec, flux 0.3-10 keV =  4.360820860647597e-15
from LC, flux 0.3-10 keV =  [4.36492365e-15]
```


## Usage
The package consists of six main classes, you can find their description and usage examples in ``tutorials``:
 
 1. ``Orbit`` (self-explanatory);
 2. ``Winds`` in which currently all information about NS, optical star, and their winds is stored. Here you can calculate the magnetic/photon fields in the random point from stars, winds pressure, or the position of the equilibrium between pulsar and optical stars winds as a function of time;
 3. ``IBS``: Intrabinary shock - an object with stuff about IBS geometry and the bulk motion along it;
 4. ``ElectronsOnIBS`` describes the population of ultra-relativistic electrons on the IBS, allows to calculate stationary e-spectra in each point of IBS;
 5. ``SpectrumIBS`` computes the non-thermal spectrum of the ultra-relativistic electrons;
 6. ``LightCurve`` performs the spectrum calculation for a number of moments of time calculating the light curve.

See tutorials in `tutorials` folder.

## Notes

The codebase is tailored to observational analysis of PSR B1259-63 but can be adapted to other similar systems.

## License 

This project is licensed under the MIT License. See the LICENSE file for details.
---


