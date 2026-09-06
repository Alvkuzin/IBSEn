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
    Return the tabulated orbital and stellar parameters for a known system.

    Returns
    -------
    dict
        Keys are ``T`` [s], ``e``, ``M`` [g], ``M_s`` [g], ``M_p`` [g],
        ``D`` [cm], ``R_s`` [cm], ``T_s`` [K], ``nu_los`` [rad], and
        ``incl_los`` [rad]. ``M`` is the derived total mass
        ``M_s + M_p``.

    """

    if sys_name == 'psrb': # Negueruela et al 2011
        Torb_here = 1236.724526*DAY; e_here = 0.8699; T_s_here = 3.3e4
        M_s = 24. * M_SOLAR; M_p = 1.4  * M_SOLAR
        M_here = M_s + M_p; D_here = 2.4e3 * PARSEC; R_s_here = 9.2 * R_SOLAR
        nu_los = np.deg2rad(270. - 138.665); incl_los= np.deg2rad(155.5) 
        
    elif sys_name == 'rb':
        Torb_here = 0.5*DAY; e_here = 0; T_s_here = 3.e3 
        M_s = 0.5  * M_SOLAR; M_p = 1.4  * M_SOLAR
        M_here = M_s + M_p; D_here = 1.e3 * PARSEC; R_s_here = 0.3 * R_SOLAR
        nu_los = 0.; incl_los=np.pi/4.
        
    elif sys_name == 'bw':
        Torb_here = 0.1*DAY; e_here = 0.; T_s_here = 1.e3 
        M_s = 0.1  * M_SOLAR; M_p = 1.4  * M_SOLAR
        M_here = M_s + M_p; D_here = 1.e3 * PARSEC; R_s_here = 0.01 * R_SOLAR
        nu_los = 0.;  incl_los=np.pi/4.
        
    elif sys_name == 'test':
        Torb_here = 100.0*DAY; e_here = 0.5; T_s_here = 3.0e4
        M_s = 30. * M_SOLAR; M_p = 1.4  * M_SOLAR
        M_here = M_s + M_p; D_here = 2.4e3 * PARSEC; R_s_here = 10.0 * R_SOLAR
        nu_los = np.deg2rad(45.); incl_los= np.deg2rad(45.)
        
    elif sys_name == 'ls5039': # Casares et al  2005 
        Torb_here = 3.906*DAY; e_here = 0.35; T_s_here = 3.9e4
        M_s = 22.9 * M_SOLAR; M_p = 1.4  * M_SOLAR
        M_here = M_s + M_p; D_here = 1.89e3 * PARSEC; R_s_here = 9.3 * R_SOLAR
        nu_los = np.deg2rad(270-45.8); incl_los= np.deg2rad(60.)
        
    elif sys_name == 'psrj2032': # Ho et al 2017, Lyne et al 2015
        Torb_here = 16500.*DAY; e_here = 0.96; T_s_here = 2.e4
        M_s = 15. * M_SOLAR; M_p = 1.4  * M_SOLAR
        M_here = M_s + M_p; D_here = 1.68e3 * PARSEC; R_s_here = 10. * R_SOLAR
        nu_los = np.deg2rad(270.-40.); incl_los= np.deg2rad(30.)
        
    elif sys_name == 'ls61': # Chernyakova et al 2020, Dubus 2013 
        Torb_here = 26.5*DAY; e_here = 0.537; T_s_here = 2.25e4
        M_s = 12. * M_SOLAR; M_p = 1.4  * M_SOLAR
        M_here = M_s + M_p; D_here = 2.49e3 * PARSEC; R_s_here = 10. * R_SOLAR
        nu_los = np.deg2rad(270.+141.); incl_los= np.deg2rad(30.)
        
        
    else:
        raise ValueError(f'Unknown name: {sys_name}')
    

    res = { 'e': e_here, 'M': M_here, 'D': D_here, 'R_s': R_s_here, 
           'T': Torb_here, 'T_s': T_s_here, 'M_s': M_s, 'M_p': M_p,
           'nu_los': nu_los, 'incl_los': incl_los}
    return res
