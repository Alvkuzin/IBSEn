import numpy as np
from astropy import constants as const

G = float(const.G.cgs.value)

R_SOLAR = float(const.R_sun.cgs.value)
M_SOLAR = float(const.M_sun.cgs.value)
PARSEC = float(const.pc.cgs.value)
DAY = 86400

known_names = ('psrb', 'rb', 'bw', 'ls5039', 'psrj2032', 'ls61', 'test')

def get_parameters(sys_name):
    """
    Quickly access some PSRB orbital parameters: 
    orbital period T [s], eccentricity e, total mass M [g],  
    distance to the system D [cm], star radius Ropt [cm].

    Returns : dictionary
    -------
    'e', 'M' [g], 'D' [cm], 'Ropt' [cm], 'T' [s]

    """

    if sys_name == 'psrb': # Negueruela et al 2011
        Torb_here = 1236.724526*DAY; e_here = 0.874; Topt_here = 3.3e4
        Mopt = 31 * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 2.4e3 * PARSEC; Ropt_here = 9.2 * R_SOLAR
        nu_los = 2.3; incl_los= 22 / 180 * np.pi
        
    elif sys_name == 'rb':
        Torb_here = 0.5*DAY; e_here = 0; Topt_here = 3e3 
        Mopt = 0.5  * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 1e3 * PARSEC; Ropt_here = 0.3 * R_SOLAR
        nu_los = 0; incl_los=np.pi/4
        
    elif sys_name == 'bw':
        Torb_here = 0.1*DAY; e_here = 0; Topt_here = 1e3 
        Mopt = 0.1  * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 1e3 * PARSEC; Ropt_here = 0.01 * R_SOLAR
        nu_los = 0.;  incl_los=np.pi/4
        
    elif sys_name == 'test':
        Torb_here = 100.0*DAY; e_here = 0.5; Topt_here = 3.0e4
        Mopt = 30 * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 2.4e3 * PARSEC; Ropt_here = 10.0 * R_SOLAR
        nu_los = 135/180*np.pi; incl_los= 45 / 180 * np.pi
        
    elif sys_name == 'ls5039': # Casares et al  2005 
        Torb_here = 3.906*DAY; e_here = 0.35; Topt_here = 3.9e4
        Mopt = 22.9 * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 1.89e3 * PARSEC; Ropt_here = 9.3 * R_SOLAR
        nu_los = (270-45.8)/180*np.pi; incl_los= 60 / 180 * np.pi
        
    elif sys_name == 'psrj2032': # Ho et al 2017, Lyne et al 2015
        Torb_here = 16500*DAY; e_here = 0.96; Topt_here = 2e4
        Mopt = 15 * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 1.68e3 * PARSEC; Ropt_here = 10 * R_SOLAR
        nu_los = (270-40)/180*np.pi; incl_los= 30 / 180 * np.pi
        
    elif sys_name == 'ls61': # Chernyakova et al 2020, Dubus 2013 
        Torb_here = 26.5*DAY; e_here = 0.537; Topt_here = 2.25e4
        Mopt = 12 * M_SOLAR; M_ns = 1.4  * M_SOLAR
        M_here = Mopt + M_ns; D_here = 2.49e3 * PARSEC; Ropt_here = 10 * R_SOLAR
        nu_los = (270+141)/180*np.pi; incl_los= 30 / 180 * np.pi
        
        
    else:
        raise ValueError(f'Unknown name: {sys_name}')
    

    res = { 'e': e_here, 'M': M_here, 'D': D_here, 'Ropt': Ropt_here, 
           'T': Torb_here, 'Topt': Topt_here, 'Mopt': Mopt, 'M_ns': M_ns,
           'nu_los': nu_los, 'incl_los': incl_los}
    return res

