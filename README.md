# IBSEn: calculates the IBS Emissivity for PSR B1259-63.

IBSEn: **I**ntra**B**inary **S**hock **E**mission **n**-**n**ot-mea**n**i**n**g-a**n**ythi**n**g is a Python package for modeling and visualizing observational data of the gamma-ray binariy pulsars, especially PSR B1259-63. This project includes tools to compute intrabinary shock (IBS) emissivity: spectra and light curves, and (somewhere in future) fit theoretical models to light curves and spectra.


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

**Note on Naima.** If you have both Naima and numpy installed and they are not in conflict, you are good to go. It is a good idea to create a separate environment for IBSEn where the numpy<2 and  Naima will be installed.


Scripts now are mainly not suited for running from the command line... And I don't have proper tests yet, so to find out if the installation works, try running a very basic python script that simply initializes a lot of classes with more or less default parameters:
```bash
python test_ibsen.py
```
You should get the output of

```bash
PSR B1259-63 periastron dist [au] =  0.8623138966114536
----- at t=20 days after periastron -----
effective beta =  0.035321971059500246
IBS opening angle =  2.4833832541305156
----- using the cooling law: stat_ibs -----
tot number of e on IBS =  7.08705617073463e+39
from spec, flux 0.3-10 keV =  7.0616353227403065e-12
from spec, flux 0.4-10 TeV =  1.3470105074688486e-12
from LC, flux 0.3-10 keV =  7.0616353227403065e-12
from LC, flux 0.4-10 TeV =  1.3470105074688486e-12

```

## Usage
There is a poor attempt at the graphical interface: run it with
```bash
ibsen
```
You can explore how the IBS, SEDs, and light curves change for different parameters there, as well as display some data together with a model, and perform a simple normalization fitting to SEDs / LCs. 

Mainly, though, the package is meant to be ran as a part of a python script.

The package consists of six main classes, you can find their description and usage examples in ``tutorials``. The general idea is this: you define
an orbit of the binary, then the outflows ("winds") properties, then the intrabinary shock (IBS), then compute the electrons evolution/transport over IBS,
then calculate the photon spectrum. There is also a possibility of doing all these steps at once by calculating a light curve.
 
 1. ``Orbit`` (self-explanatory);
Let's initiate an Orbit object and plot an orbit: 
```python
from ibsen import Orbit
import matplotlib.pyplot as plt   
import numpy as np
DAY = 86400
orb = Orbit(T=25*DAY, e=0.7, M=30*2e33,
                 nu_los=90*np.pi/180, incl_los=20*np.pi/180)
t = np.linspace(-20*DAY, 70*DAY, 1000)
plt.plot(orb.x(t), orb.y(t))
```
 2. ``Winds`` in which currently all information about NS, optical star, and their winds are stored. Here you can calculate the magnetic/photon fields in any point, winds pressure, the position of the equilibrium between pulsar and optical stars winds as a function of time, etc.
Initiate `Winds` with a decretion disk pressure 100 times stronger than the polar wind (see tutorials for further info) and plot the star-to-emission zone distance VS time.
```python
from ibsen import Winds
winds = Winds(orbit=orb, f_d=100, Ropt=7e11, Mopt=28*2e33, Topt=4e4, ns_b_apex=10)
t1 = np.linspace(-3*DAY, 3*DAY, 1000)
plt.plot(t1/DAY, winds.dist_se_1d(t1))
td1, td2 = winds.times_of_disk_passage
plt.axvline(td1/DAY)
plt.axvline(td2/DAY)
```
 3. ``IBS``: Intrabinary shock - an object with stuff about IBS geometry and the bulk motion along it;
Initiate IBS at a time of 2 days after the periastron and take a look at it, colorcoding it with the doppler factor of matter bulk motion.
```python
from ibsen import IBS # or from ibsen.ibs import IBS3D 
ibs = IBS(winds=winds, t_to_calculate_beta_eff=2*DAY)
ibs.peek(showtime=(-3*DAY, 3*DAY), show_winds=True, ibs_color='doppler')
```
 4. ``ElectronsOnIBS`` describes the population of ultra-relativistic electrons on the IBS, allows to calculate stationary e-spectra in each point of IBS;
Take a look at the electron spectrum over IBS if the apex magnetic field is 10 [G] (was specified in `Winds`):
```python
from ibsen import ElectronsOnIBS
elev = ElectronsOnIBS(ibs=ibs, cooling='stat_mimic')
elev.calculate()
elev.peek()
```
 5. ``SpectrumIBS`` computes the non-thermal photon spectrum emitted by the ultra-relativistic electrons;
Calculate the synchrotron + inverse Compton spectrum from the population of electrons we just found:
```python
from ibsen.spec import SpectrumIBS
spec = SpectrumIBS(els=elev, method='simple', mechanisms=['syn', 'ic',], distance=3e3*3e18)
spec.calculate(e_ph = np.geomspace(3e2, 1e13, 1001))
spec.peek()
```
 6. ``LightCurve`` performs the spectrum calculation for a number of moments of time, returning the light curve and all intermediate calculation results.
```python
from ibsen import LightCurve
t_lc = np.linspace(-3*DAY, 3*DAY, 100)
lc = LightCurve(times = t_lc, 
                to_parall=True, n_cores=4,
                T=25*DAY, e=0.7, M=30*2e33,
                 nu_los=90*np.pi/180, incl_los=20*np.pi/180, 
                 Ropt=7e11, Mopt=28*2e33, Topt=4e4,
                distance=3e3*3e18,
                bands = ([300, 1e4], ), cooling='stat_mimic',
                f_d=100, 
                ns_b_ref=10, ns_r_ref=ibs.x_apex, # so that the field in the apex (at t = 2 days) = 1
                method='simple', mechanisms=['syn', 'ic'])
lc.calculate()
lc.peek()
```
See tutorials in `tutorials` folder for the complete description of these models.


### TODO
 1. When calculating IC, should we account for the seed photons coming from the disk?
 2. How to visualize winds in 3D?
 3. Write fitting utils for LC/SEDs. It should take several datasets and fit to the theoretical model using the same sets of parameters except normalizations.
 4. If one is bored so much that one feels the need to make the graphical interface better, it'd be cool to add all gamma-ray binaies systems in the drop-down windows, as well as allow for the basic system parameters variation (such as Torb, Mopt, e...) instead of keeping them strictly fixed. Also, the default values and ranges of sliders should be system-dependent.


## License 

This project is licensed under the MIT License. See the LICENSE file for details.
---


