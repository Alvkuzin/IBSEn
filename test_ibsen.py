"""
    Very stupid very dumb very primitive very basic very shallow test of the
    code. Just testing that everything can imported OK and simple calculations
    are performed withour errors.
"""
import ibsen
from ibsen.utils import loggrid
import numpy as np
import argparse

from ibsen import Orbit, Winds, OpticalStar, Pulsar, IBS, IBS3D, ElectronsOnIBS, SpectrumIBS, LightCurve

DAY = 86400
t = 20 * DAY
f_d = 100
puls_b_ref = 1.0
abs_gg = True


def test_func(method='simple', ibs_ndim=2, coolings=('stat_ibs',)):
        
    orbit = Orbit('psrb')
    print('PSR B1259-63 periastron dist [au] = ', orbit.r_periastr/1.496e13)
    
    print('----- at t=20 days after periastron -----')
    
    star = OpticalStar(sys_name='psrb', f_d=f_d)
    
    pulsar = Pulsar(b_ref=puls_b_ref, r_b_ref=1e13, r_p_ref=star.Ropt)
    
    winds = Winds(orbit=orbit, star=star, pulsar=pulsar)
    print('effective beta = ', winds.beta_eff(t=t))
    
    
    if ibs_ndim==2:
        ibs = IBS(winds=winds, t_to_calculate_beta_eff=t)
    if ibs_ndim==3:
        ibs = IBS3D(winds=winds, t_to_calculate_beta_eff=t)
    
    print('IBS opening angle = ', ibs.thetainf)
    for cooling in coolings:
        print(f"----- using the cooling law: {cooling} -----")
        
        elev = ElectronsOnIBS(ibs=ibs, cooling=cooling, eta_a=1)
        elev.calculate()
        print('tot number of e on IBS = ', elev.ntot)
            
        spec = SpectrumIBS(sys_name='psrb', abs_photoel=True, abs_gg=abs_gg,
                           els=elev, method=method, mechanisms=['syn', 'ic'])
        e_calc = np.concatenate(((loggrid(3e2/1.2, 1e4*1.2, 37)), loggrid(4e11/1.2, 1e13*1.2, 37)))
        spec.calculate(e_ph = e_calc)
        print('from spec, flux 0.3-10 keV = ', spec.flux(300, 1e4, epow=1))
        print('from spec, flux 0.4-10 TeV = ', spec.flux(4e11, 1e13, epow=1))
        
            
        lc = LightCurve(times = np.array([t]), sys_name='psrb',
                        bands = ([300, 1e4], [4e11, 1e13]),
                        epows=(1, 1),
                        cooling=cooling,
                        f_d=f_d,  eta_a=1, 
                        abs_photoel=True,
                        puls_b_ref=puls_b_ref, puls_r_ref=1e13, abs_gg=abs_gg,
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
    parser.add_argument(
        "--testall",
        type=bool,
        default=False,        
        help="Whether to loop over all available cooling laws."
    )
    
    args = parser.parse_args()
    if bool(args.testall):
        coolings = ('no', 'stat_apex', 'stat_ibs', 'stat_mimic', 
                    'leak_ibs', 'leak_mimic', 'adv')
    else:
        coolings = ('stat_ibs',)
    test_func(method = str(args.method), ibs_ndim = int(args.ndim), coolings=coolings)

if __name__ == "__main__":
    main()
