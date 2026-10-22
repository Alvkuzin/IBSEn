import argparse
from astropy import constants as const
import numpy as np

from ibsen.absorption.absorption import tabulate_absgg_d_mu
from ibsen.orbit import Orbit

def main():
    DAY = 86400.
    R_SOLAR = float(const.R_sun.cgs.value)
    M_SOLAR = float(const.M_sun.cgs.value)
    # RAD_IN_DEG = np.pi / 180.0
    
    parser = argparse.ArgumentParser(
        description="""
        Tabulate gamma-gamma absorption from the command line and put to a
        ibsen-readable file.
        """
    )
    # parser.add_argument("--T", type=float, required=True,
    #                     help="T: Orbital system [days]")
    # parser.add_argument("--e", type=float, required=True,
    #                     help="e: Orbit eccentricity")
    # parser.add_argument("--M_s", type=float, required=True,
    #                     help="M_s: optical-star mass [Msol]")
    # parser.add_argument("--M_p", type=float, required=True,
    #                     help="M_p: pulsar mass [Msol]")
    # parser.add_argument("--nu_los", type=float, required=True,
    #                     help="""
    #                     nu_los: angle between direction to periastron 
    #                     and line of sight [deg]
    #                     """)
    # parser.add_argument("--incl_los", type=float, required=True,
    #                     help="""
    #                     incl_los: Orbit inclination [deg] (angle between 
    #                     orbital angular velocity and lne of sight)
    #                     """)
    parser.add_argument("--Ts", type=float, required=True,
                        help="Ts: optical-star effective temperature [K]")
    
    parser.add_argument("--Rs", type=float, required=True,
                        help="R_s: optical-star radius [Rsol]")
    parser.add_argument("--fast", type=bool, required=True,
                        help="fast: Whether to calculate faster (True) or correct (False)")
    parser.add_argument("--filename", type=str, required=True,
                        help="""filename: opacitiy tables will be saved
                        as ibsen/absorption/absorp_tab/<filename>.nc""")
    parser.add_argument("--rmin", type=float, required=False, default=1.0,
                        help="rmin: min distance from the star to tabulate [in R_s]. Default 1")
    parser.add_argument("--rmax", type=float, required=False,  default=1000.0,
                        help="rmax: max distance from the star to tabulate [in R_s]. Default 1000")
    parser.add_argument(
        "--nr",
        type=int,
        default=21,        
        help="nr: number of nods on r-axis, optional, default 21"
    )
    parser.add_argument(
        "--nmu",
        type=int,
        default=25,        
        help="nmu: number of nods on mu = cos phi axis, optional, default 25"
    )
    parser.add_argument(
        "--ne",
        type=int,
        default=28,       
        help="ne: number of nods on e-axis, optional, default 28"
    )
                        

    args = parser.parse_args()
    # orb = Orbit(T = args.T * DAY,
    #             e = args.e,
    #             M_s=args.M_s * M_SOLAR,
    #             M_p=args.M_p * M_SOLAR,
    #             nu_los = np.deg2rad(args.nu_los),
    #             incl_los = np.deg2rad(args.incl_los),
    #             )
    tabulate_absgg_d_mu(
                   nr = args.nr,
                   nmu = args.nmu,
                   ne = args.ne,
                   T_s=args.Ts,
                   R_s=args.Rs * R_SOLAR,
                   rmin=args.rmin,
                   rmax=args.rmax,
                   to_return=False,
                   fast = args.fast,
                   filename=args.filename)


if __name__ == "__main__":
    main()
