import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import Planck13, FlatLambdaCDM
from scipy.integrate import quad

def luminosity_distance_eagle(z_eval):
    """
    Calculate the luminosity distance for EAGLE galaxies at a given redshift z_eval using the Planck2013 cosmology
    """
    return Planck13.luminosity_distance(z_eval).to(u.parsec).value

def luminosity_distance_sdss(z_eval):
    """
    Calculate the luminosity distance for SDSS galaxies at a given redshift z_eval using a Flat Lambda CDM cosmology
    with Omega_matter = 0.3, omega_lambda = 0.7 and H0 = 70 km/s/Mpc
    """
    return FlatLambdaCDM(H0=70, Om0=0.3).luminosity_distance(z_eval).to(u.parsec).value
    
def app_to_abs_mag(m, d_L):
    """
    Convert apparent magnitude m to absolute magnitude M using luminosity distance d_L (pc)
    """
    return m - 5*np.log10(d_L/10)

def abs_to_app_mag(M, d_L):
    """
    Convert absolute magnitude M to apparent magnitude m using luminosity distance d_L (pc)
    """
    return M + 5*np.log10(d_L/10)
    
def flux_to_magAB(flux):
    """
    Convert flux (Jy) to AB apparent magnitude m_AB
    """
    return -2.5*np.log10(np.true_divide(flux,3631))

def magAB_to_flux(m_AB):
    """
    Convert AB apparent magnitude to flux (Jy)
    """
    return 3631*10**(-np.true_divide(m_AB,2.5))
