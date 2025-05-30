# pulsar/spectrum.py
import numpy as np

class Spectrum:
    def __init__(self, orbit, ibs):
        self.orbit = orbit
        self.ibs = ibs

    def emissivity(self, t):
        shape = self.ibs.shape(t)
        spec = shape #* np.sin(t/self.orbit.P)
        return spec
        # return f"Emissivity at t={t} based on {shape}"
