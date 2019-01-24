import numpy as np
from scipy.integrate import quad

def luminosity_distance(z_eval, H_0 = 67.77, omega_m = 0.307, omega_lambda = 0.693):
    """
    Calculate luminosity distance in Lambda CDM universe (omega_k = 0) at a given redshift z_eval
    using   d_L(z) = (1+z)*d_C(z)                   luminosity distance
            d_C(z) = c*integral dz'/H from 0 to z   comoving distance
    Default parameters: Planck 2013
    Arguments
        H_0             - Hubble constant today = 67.77 (km/s/Mpc)
        omega_m         - matter density = 0.307
        omega_lambda    - dark energy density = 0.693
    Returns:
        d_L             - luminosity distance at z_eval (pc)
    """    
    
    c = 299792458 #m/s
    
    def integrand(z, omega_m, omega_lambda):
        return 1/np.sqrt(omega_m*((1+z)**3.) + omega_lambda)
    
    I, I_err = quad(integrand, 0, z_eval, args=(omega_m, omega_lambda))
    print("Integrand: ", I, " error: ", I_err)
    
    # Comoving distance (pc)
    d_C = c/H_0*(10**3) * I 
    
    # Luminosity distance (pc)
    d_L = (1+z_eval)*d_C
    return d_L

def app_to_abs_mag(m, d_L):
    """
    Convert apparent magnitude to absolute magnitude
    Arguments:
        m           - apparent magnitude
        d_L         - luminosity distance (pc)
    Returns:
        M           - absolute magnitude
    """
    return m - 5*np.log10(d_L/10)

def abs_to_app_mag(M, d_L):
    """
    Convert absolute magnitude to apparent magnitude
    Arguments:
        M           - absolute magnitude
        d_L         - luminosity distance (pc)
    Returns:
        m           - apparent magnitude
    """
    return M + 5*np.log10(d_L/10)
    

def flux_to_magAB(flux):
    """
    Convert flux to AB magnitude using: m_AB = -2.5 log10(flux/3631 Jy)
    Arguments:
        flux        - (Jy)
    Returns
        m_AB        - AB apparent magnitude
    """
    return -2.5*np.log10(flux/3631)


def magAB_to_flux(m_AB):
    """
    Convert AB magnitude to flux using: flux = 3631 Jy * 10**(-m_AB/2.5)
    Arguments:
        m_AB        - AB apparent magnitude
    Returns:
        flux        - (Jy)
    """
    return 3631*10**(-m_AB/2.5)

