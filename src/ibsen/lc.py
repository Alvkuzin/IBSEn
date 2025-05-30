# pulsar/lightcurve.py

class LightCurve:
    def __init__(self, spectrum):
        self.spectrum = spectrum

    def compute(self, times):
        return [self.spectrum.emissivity(t) for t in times]
