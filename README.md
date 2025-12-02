# IBSEn: calculates the IBS Emissivity for PSR B1259-63.

IBSEn: **I**ntra**B**inary **S**hock **E**mission **n**-**n**ot-mea**n**i**n**g-a**n**ythi**n**g is a collection of Python scripts for processing, modeling, and visualizing observational data of the gamma-ray binariy pulsars, especially PSR B1259-63. This project includes tools to compute intrabinary shock (IBS) emissivity: spectra and light curves, and (somewhere in future) fit theoretical models to light curves and spectra.


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

For me though there was a problem that Naima, if I understand correctly, requires Numpy v. < 2. But I had Numpy of the version 2...! So you can either downgrade Numpy to <2:
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

Scripts now are not suited for running from the command line... And I don't have proper tests yet, so to find out if the installation works, try running a very basic python script that simply initializes a lot of classes with more or less default parameters:
```bash
python test_ibsen.py
```
You should get the output of

```bash
PSR B1259-63 periastron dist [au] =  0.9057236987554738
----- at t=20 days after periastron -----
effective beta =  0.04263905625747101
IBS opening angle =  2.444830788798784
tot number of e on IBS =  1.1671010229969223e+42
from spec, flux 0.3-10 keV =  1.3935430802714186e-10
from LC, flux 0.3-10 keV =  [1.39641903e-10]
```


## Usage
The package consists of six main classes, you can find their description and usage examples in ``tutorials``:
 
 1. ``Orbit`` (self-explanatory);
Let's initiate an Orbit object and plot an orbit: 
```python
from ibsen.orbit import Orbit
import matplotlib.pyplot as plt   
import numpy as np
DAY = 86400
orb = Orbit(T=25*DAY, e=0.7, M=30*2e33,
                 nu_los=90*np.pi/180, incl_los=20*np.pi/180)
t = np.linspace(-20*DAY, 70*DAY, 1000)
plt.plot(orb.x(t), orb.y(t))
```
 2. ``Winds`` in which currently all information about NS, optical star, and their winds is stored. Here you can calculate the magnetic/photon fields in the random point from stars, winds pressure, or the position of the equilibrium between pulsar and optical stars winds as a function of time;
Initiate Winds with a decretion disk pressure 100 times stronger than the polar wind (in which sense - see tutorials) and plot the star-emission zone separation VS time.
```python
from ibsen.winds import Winds
winds = Winds(orbit=orb, f_d=100, Ropt=7e11, Mopt=28*2e33, Topt=4e4, ns_b_apex=10)
t1 = np.linspace(-3*DAY, 3*DAY, 1000)
plt.plot(t1/DAY, winds.dist_se_1d(t1))
td1, td2 = winds.times_of_disk_passage
plt.axvline(td1/DAY)
plt.axvline(td2/DAY)
```
 3. ``IBS``: Intrabinary shock - an object with stuff about IBS geometry and the bulk motion along it;
Initiate IBS at a time of 2 days after the periastron and take a look at it, colorcoding it with the doppler factor.
```python
from ibsen.ibs import IBS
ibs = IBS(winds=winds, t_to_calculate_beta_eff=2*DAY)
ibs.peek(showtime=(-3*DAY, 3*DAY), show_winds=True, ibs_color='doppler')
```
 4. ``ElectronsOnIBS`` describes the population of ultra-relativistic electrons on the IBS, allows to calculate stationary e-spectra in each point of IBS;
Take a look at the electron spectrum over IBS if the apex magnetic field is 1G:
```python
from ibsen.el_ev import ElectronsOnIBS
elev = ElectronsOnIBS(ibs=ibs, cooling='stat_mimic')
elev.calculate()
elev.peek()
```
 5. ``SpectrumIBS`` computes the non-thermal photon spectrum emitted by the ultra-relativistic electrons;
Calculate the synchrotron + inverse Compton spectrum from the population of electrons we obtained above:
```python
from ibsen.spec import SpectrumIBS
spec = SpectrumIBS(els=elev, method='simple', mechanisms=['syn', 'ic',], distance=3e3*3e18)
spec.calculate(e_ph = np.geomspace(3e2, 1e13, 1001))
spec.peek()
```
 6. ``LightCurve`` performs the spectrum calculation for a number of moments of time calculating the light curve.
```python
from ibsen.lc import LightCurve
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

## Notes

The codebase is tailored to observational analysis of PSR B1259-63 but can be adapted to other similar systems.

### TODO

 1. Occasional NaNs in IC spectrum. Should we ignore them (e.g. by interpolating)? Are they there because of something internally Naimian or is it my fault?
 2. It seems rather simple to make a 3d interactable visualisation of the IBS. The only thing that stops me is the visualisation of winds in 3d. 
For now, the only more or less visually pleasant 3d representation of winds in Python is plot a loooot of points with densities \propto density/pressure of winds,
but it is very time-consuming. Maybe we shoud just ignore winds in 3d visualisation.



## License 

This project is licensed under the MIT License. See the LICENSE file for details.
---


