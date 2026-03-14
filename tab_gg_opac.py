import argparse
from astropy import constants as const
import numpy as np

from ibsen.absorbtion.absorbtion import tabulate_absgg
from ibsen.orbit import Orbit

def main():
    DAY = 86400.
    R_SOLAR = float(const.R_sun.cgs.value)
    M_SOLAR = float(const.M_sun.cgs.value)
    # RAD_IN_DEG = np.pi / 180.0
    
    parser = argparse.ArgumentParser(
        description="""
        Tabulate gamma-gamma absorbtion from the command line and put to a
        ibsen-readable file.
        """
    )
    parser.add_argument("--T", type=float, required=True,
                        help="T: Orbital system [days]")
    parser.add_argument("--e", type=float, required=True,
                        help="e: Orbit eccentricity")
    parser.add_argument("--M", type=float, required=True,
                        help="M: Total system mass [Msol]")
    parser.add_argument("--nu_los", type=float, required=True,
                        help="""
                        nu_los: angle between direction to periastron 
                        and line of sight [deg]
                        """)
    parser.add_argument("--incl_los", type=float, required=True,
                        help="""
                        incl_los: Orbit inclination [deg] (angle between 
                        orbital angular velocity and lne of sight)
                        """)
    parser.add_argument("--Topt", type=float, required=True,
                        help="Topt: Optical star effective temperature [K]")
    parser.add_argument("--Ropt", type=float, required=True,
                        help="Ropt: Optical star radius [Rsol]")
    parser.add_argument("--fast", type=bool, required=True,
                        help="fast: Whether to calculate faster (True) or correct (False)")
    parser.add_argument("--filename", type=str, required=True,
                        help="""filename: opacitiy tables will be saved
                        as ibsen/absorbtion/absorb_tab/<filename>.nc""")
    parser.add_argument(
        "--nrho",
        type=int,
        default=21,        
        help="nrho: number of nods on r-axis, optional, default 21"
    )
    parser.add_argument(
        "--nphi",
        type=int,
        default=25,        
        help="nphi: number of nods on phi-axis, optional, default 25"
    )
    parser.add_argument(
        "--ne",
        type=int,
        default=28,       
        help="ne: number of nods on e-axis, optional, default 28"
    )
                        

    args = parser.parse_args()
    orb = Orbit(T = args.T * DAY,
                e = args.e,
                M = args.M * M_SOLAR,
                nu_los = np.deg2rad(args.nu_los),
                incl_los = np.deg2rad(args.incl_los),
                )
    tabulate_absgg(orb = orb,
                   nrho = args.nrho,
                   nphi = args.nphi,
                   ne = args.ne,
                   Topt = args.Topt,
                   Ropt = args.Ropt * R_SOLAR,
                   to_return=False,
                   fast = args.fast,
                   filename=args.filename)


if __name__ == "__main__":
    main()
