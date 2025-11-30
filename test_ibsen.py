"""
    Very stupid very dumb very primitive very basic very shallow test of the
    code. Just testing that everything can imported OK and simple calculations
    are performed withour errors.
"""
import ibsen
import numpy as np

DAY = 86400
t = 20 * DAY

from ibsen.orbit import Orbit

orbit = Orbit('psrb')
print('PSR B1259-63 periastron dist [au] = ', orbit.r_periastr/1.496e13)

from ibsen.winds import Winds
print('----- at t=20 days after periastron -----')

winds = Winds(orbit=orbit, sys_name='psrb', f_d=100, ns_b_apex=1)
print('effective beta = ', winds.beta_eff(t=t))

from ibsen.ibs import IBS

ibs = IBS(winds=winds, t_to_calculate_beta_eff=t)
print('IBS opening angle = ', ibs.thetainf)

from ibsen.el_ev import ElectronsOnIBS

elev = ElectronsOnIBS(ibs=ibs, cooling='no')
elev.calculate()
print('tot number of e on IBS = ', elev.ntot)

from ibsen.spec import SpectrumIBS

spec = SpectrumIBS(sys_name='psrb', 
                   els=elev, method='simple', mechanisms=['syn',])
spec.calculate(e_ph = np.logspace(2., 4.3, 101))
print('from spec, flux 0.3-10 keV = ', spec.flux(300, 1e4))

from ibsen.lc import LightCurve

lc = LightCurve(times = np.array([t,]), sys_name='psrb',
                bands = ([300, 1e4],), cooling='no',
                f_d=100, 
                ns_b_ref=1, ns_r_ref=ibs.x_apex, # so that the field in the apex = 1
                method='simple', mechanisms=['syn',])
lc.calculate()
print('from LC, flux 0.3-10 keV = ', lc.fluxes[0])

