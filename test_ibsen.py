"""
    Very stupid very dumb very primitive very basic very shallow test of the
    code. Just testing that everything can imported OK and simple calculations
    are performed withour errors.
"""
import ibsen
from ibsen.utils import loggrid
import numpy as np
import argparse

DAY = 86400
t = 20 * DAY


def test_func(method='simple', ibs_ndim=2):
    from ibsen.orbit import Orbit
    
    orbit = Orbit('psrb')
    print('PSR B1259-63 periastron dist [au] = ', orbit.r_periastr/1.496e13)
    
    from ibsen.winds import Winds
    print('----- at t=20 days after periastron -----')
    
    winds = Winds(orbit=orbit, sys_name='psrb', f_d=100, ns_b_ref=1, ns_r_ref=1e13)
    print('effective beta = ', winds.beta_eff(t=t))
    
    from ibsen.ibs import IBS, IBS3D
    
    if ibs_ndim==2:
        ibs = IBS(winds=winds, t_to_calculate_beta_eff=t)
    if ibs_ndim==3:
        ibs = IBS3D(winds=winds, t_to_calculate_beta_eff=t)
    
    print('IBS opening angle = ', ibs.thetainf)
    
    from ibsen.el_ev import ElectronsOnIBS
    
    elev = ElectronsOnIBS(ibs=ibs, cooling='stat_ibs', eta_a=1)
    elev.calculate()
    print('tot number of e on IBS = ', elev.ntot)
    
    from ibsen.spec import SpectrumIBS
    
    spec = SpectrumIBS(sys_name='psrb', abs_photoel=True, abs_gg=True,
                       els=elev, method=method, mechanisms=['syn', 'ic'])
    e_calc = np.concatenate(((loggrid(3e2/1.2, 1e4*1.2, 37)), loggrid(4e11/1.2, 1e13*1.2, 37)))
    spec.calculate(e_ph = e_calc)
    print('from spec, flux 0.3-10 keV = ', spec.flux(300, 1e4, epow=1))
    print('from spec, flux 0.4-10 TeV = ', spec.flux(4e11, 1e13, epow=1))
    
    
    from ibsen.lc import LightCurve
    
    lc = LightCurve(times = np.array([t]), sys_name='psrb',
                    bands = ([300, 1e4], [4e11, 1e13]),
                    epows=(1, 1),
                    cooling='stat_ibs',
                    f_d=100,  eta_a=1, 
                    abs_photoel=True,
                    ns_b_ref=1, ns_r_ref=1e13, abs_gg=True,
                    ibs_ndim=ibs_ndim,
                    method=method, mechanisms=['syn', 'ic'])
    lc.calculate()
    print('from LC, flux 0.3-10 keV = ', lc.fluxes[0, 0])
    print('from LC, flux 0.4-10 TeV = ', lc.fluxes[0, 1])

def main():
    parser = argparse.ArgumentParser(
        description="""
        Simple IBSEn test.
        """
    )

    parser.add_argument(
        "--method",
        type=str,
        default='simple',        
        help="method: 'simple', 'apex', or 'full', optional, default 'simple'."
    )
    parser.add_argument(
        "--ndim",
        type=int,
        default=2,        
        help="IBS dimensions: 2 or 3, optional, default 2."
    )
    args = parser.parse_args()
    test_func(method = str(args.method), ibs_ndim = int(args.ndim))

if __name__ == "__main__":
    main()
