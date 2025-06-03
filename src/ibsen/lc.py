# pulsar/lightcurve.py
from ibsen.spec import SpectrumIBS
import numpy as np
# from naima.models import ExponentialCutoffPowerLaw, Synchrotron, InverseCompton
import astropy.units as u
# import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.optimize import brentq#, root, fsolve, least_squares, minimize
from scipy.optimize import curve_fit
# from pathlib import Path
from numpy import pi, sin, cos
from joblib import Parallel, delayed
import multiprocessing
# import time
# import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, interp1d

class LightCurve:
    def __init__(self, spec, times, bands = ( [3e2, 1e4], ), to_parall=False):
        self.spec = spec
        self.times = times
        self.bands = bands
        self.to_parall = to_parall
        

    # @property
    def calculate(self):
        
        