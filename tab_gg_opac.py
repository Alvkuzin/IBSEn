import argparse
from astropy import constants as const
import numpy as np

from ibsen.absorption.absorption import tabulate_absgg_d_mu
from ibsen.orbit import Orbit

def main():
    R_SOLAR = float(const.R_sun.cgs.value)
    
    parser = argparse.ArgumentParser(
        description="""
        Tabulate gamma-gamma absorption from the command line and put to a
        ibsen-readable file.
        """
    )

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
