# IBSEn: calculates the IBS Emissivity for PSR B1259-63.

IBSEn: **I**ntra**B**inary **S**hock **E**mission **n**-**n**ot-mea**n**i**n**g-a**n**ythi**n**g is a collection of Python scripts for processing, modeling, and visualizing observational data of the gamma-ray binariy pulsars, especially PSR B1259-63. This project includes tools to handle observational data, compute intrabinary shock (IBS) emissivity: spectra and light curves, and (somewhere in future) fit theoretical models to light curves and spectra.


---


## Installation & Requirements 
### Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/your-username/your-repo-name.git
```

Or clone and install manually. Normal installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install .
```

For the installation and contributing, replace the last line with 

```bash
pip install -e .
```

### Requirements
This project requires:
 1. Python 3 (please install yourself),
 2. [Naima](https://naima.readthedocs.io/en/latest/index.html) (please also install yourself), 
 3. And some standard scientific libraries. If needed, you can install them, Naima excluded, with:

```bash
pip install -r requirements.txt
pip install naima
```

**Note on Naima.** Naima, and if I understand correctly, requires Numpy v. < 2. But I had Numpy of the version 2...! So you can either downgrade Numpy to <2:
```bash
pip install "numpy<2.0"


```

or (what my best friend ChatGPT told me to do) create an environment with Numpy < 2 and Naima installed (and some other packages which do not seem to be in conflict with Numpy or Naima):

```bash
conda create -n myenv numpy<2.0 naima
conda activate myenv

```

If you follow the second way, ensure that you install IBSEn in myenv. 

**But if you just want to use IBSEn for Orbit / Winds & Stars / IBS / electron evolution and NOT for specta / light curves, you're good to go installing Naima at all. **


## Usage

Scripts now are not suited for running from the command line... Try doing this:
```python
import ibsen
from ibsen.orbit import Orbit

orb = Orbit('psrb')
print(orb.T/86400, orb.e, orb.M/2e33)
```

See tutorials in `tutorials` folder.

## Notes

The codebase is tailored to observational analysis of PSR B1259-63 but can be adapted to other similar systems.

## License 

This project is licensed under the MIT License. See the LICENSE file for details.
---


