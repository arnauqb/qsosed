"""
This module computes the AGN Spectral Energy Density in the UV/X-Ray energy range, following Kubota & Done (2018). This is equivalent to AGNSED in XSPEC.
"""
import numpy as np
import constants as const
from scipy import integrate, optimize
from astropy import units as u

def convert_units(old, new_unit):
    """
    Convert *value* in *current_unit* units, to new_unit. Uses spectral equivalencies of astropy.
    """
    new = old.to( new_unit, equivalencies = u.spectral() )
    return new.value

class SED:
    """
    Class to handle the AGN SED calculation functions. Implements Kubota & Done (2018) paper.
    """

    energy_min = 1e-4 # keV
    energy_max = 200. # keV
    freq_min = convert_units(energy_min * u.keV, u.Hz)
    freq_max = convert_units(energy_max * u.keV, u.Hz )

    r_max = 1400.
    def __init__(self, M = 1e8, mdot = 0.5, astar = 0, sign = 1, hard_xray_fraction = 0.02):

        # read parameters #
        self.M = M # black hole mass in solar masses
        self.mdot = mdot # mdot = Mdot / Mdot_Edd
        self.astar = astar # dimensionless black hole spin
        self.spin_sign = sign # +1 for prograde rotation, -1 for retrograde

        # useful quantities #
        self.Rg = const.G * M * const.Ms / const.c ** 2 # gravitational radius
        self.isco = self.compute_isco() # Innermost Stable Circular Orbit
        self.eta = self.compute_efficiency() # Accretion efficiency defined as L = \eta \dot M c^2
        self.hard_xray_fraction = hard_xray_fraction

    def compute_isco(self):
        """
        Computes the Innermost Stable Circular Orbit. Depends only on astar.
        """
        z1 = 1 + (1 - self.astar**2)**(1 / 3) * ((1 + self.astar)**(1 / 3) + (1 - self.astar)**(1 / 3))
        z2 = np.sqrt(3 * self.astar**2 + z1**2)
        rms = 3 + z2 - self.spin_sign * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
        return rms
    
    def compute_efficiency(self):
        """ 
        Accretion Efficiency

        Parameters
        ----------
        isco :  float
                Innermost stable circular orbit
        """ 
        eta = 1 - np.sqrt( 1 - 2 / (3 * self.isco) )
        return eta

    def nt_rel_factors(self, r=6):
        """
        Relatistic A,B,C factors of the Novikov-Thorne model.
        
        Parameters
            Black Hole Mass in solar Masses
        -----------
        r : float
            Disc radial distance.
        """

        yms = np.sqrt(self.isco)
        y1 = 2 * np.cos((np.arccos(self.astar) - np.pi) / 3)
        y2 = 2 * np.cos((np.arccos(self.astar) + np.pi) / 3)
        y3 = -2 * np.cos(np.arccos(self.astar) / 3)
        y = np.sqrt(r)
        C = 1 - 3 / r + 2 * self.astar / r**(3 / 2)
        B = 3 * (y1 - self.astar)**2 * np.log(
            (y - y1) / (yms - y1)) / (y * y1 * (y1 - y2) * (y1 - y3))
        B += 3 * (y2 - self.astar)**2 * np.log(
            (y - y2) / (yms - y2)) / (y * y2 * (y2 - y1) * (y2 - y3))
        B += 3 * (y3 - self.astar)**2 * np.log(
            (y - y3) / (yms - y3)) / (y * y3 * (y3 - y1) * (y3 - y2))
        A = 1 - yms / y - 3 * self.astar * np.log(y / yms) / (2 * y)
        factor = (A-B)/C
        return factor

    def eddington_luminosity(self):
        """
        Eddington Luminosity. Reads from constants module.
        emmisivity_constant = 4 * pi * mp * c^3 / sigma_t

        Parameters
        ----------
        M : float
            Black Hole Mass in solar Masses
        """
        
        Ledd = const.emissivity_constant * self.Rg
        return Ledd

    def mass_accretion_rate(self):
        """
        Mass Accretion Rate in units of g/s.
        """

        Mdot = self.mdot * self.eddington_luminosity() / ( self.eta * const.c**2)
        return Mdot

    def disc_nt_temperature4(self, r):
        """
        Computes Novikov-Thorne temperature in Kelvin (to the power of 4) of accretion disc annulus at radius r.
        Parameters
        ----------
        r : float
            Disc radius in Rg. 
        """

        nt_constant = 3 * const.m_p * const.c**5 / ( 2 * const.sigma_sb * const.sigma_t * const.G * const.Ms)
        rel_factor = self.nt_rel_factors( r )
        aux = self.mdot / ( self.M * self.eta * r**3 )
        t4 = nt_constant * rel_factor * aux
        return t4

    def disc_spectral_radiance(self, nu, r):
        """
        Disc spectral radiance in units of erg / cm^2 / sr, assuming black-body radiation.

        Parameters
        ----------
        nu : float
             Frequency in Hz.
        r :  float
             Disc radius in Rg.
        """

        bb_constant = 2 * const.h / (const.c ** 2)
        temperature = self.disc_nt_temperature4(r) ** (1./4.)
        planck_spectrum_exp = np.exp( const.h * nu / ( const.k_B *  temperature ))
        planck_spectrum = bb_constant * nu**3 * 1./ ( planck_spectrum_exp - 1)
        return planck_spectrum

    def disc_radiance(self, r):
        """
        Disc radiance in units of erg / cm^2 / s / sr, assuming black-body radiation.

        Parameters
        ----------
        r : float
            Disc radius in Rg.
        """

        radiance = const.sigma_sb * disc_nt_temperature4(r)
        return radiance

    def disc_spectral_luminosity(self, nu):
        """
        Disc spectral luminosity in units of erg.
        Parameters
        ----------
        nu : float
             Frequency in Hz.
        """
        radial_integral = 2 * self.Rg**2 * integrate.quad( lambda r: r * self.disc_spectral_radiance(nu,r), self.isco, self.r_max)[0]
        angular_integral = 2 * np.pi**2
        spectral_lumin = radial_integral * angular_integral 
        return spectral_lumin

    def disc_luminosity(self):
        """
        Disc Luminosity in units of erg / s.
        """
        constant =  const.sigma_sb * 4 * np.pi * self.Rg**2
        lumin = constant * integrate.quad(lambda r: r*self.disc_nt_temperature4(r), self.isco, self.r_max)[0]
        return lumin









    





