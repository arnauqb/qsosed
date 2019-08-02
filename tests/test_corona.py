"""
Tests involving the disk component.
"""
import pyagn.sed as sed
from pyagn import constants
from scipy import integrate
import numpy as np
from numpy import testing
from astropy import units as u

sed_test = sed.SED(M=1e8, mdot = 0.5, reprocessing = False)

    
def test_conservation_energy():
    distance = 1e20
    bol_lumin = sed_test.bolometric_luminosity
    total_spectral_flux = sed_test.total_flux(1e20)
    total_flux = integrate.trapz(x = sed_test.ENERGY_RANGE_KEV, y = total_spectral_flux)
    total_flux_erg = sed.convert_units(total_flux * u.keV, u.erg)
    total_lumin = total_flux_erg * 4 * np.pi * distance**2
    testing.assert_approx_equal(total_lumin, bol_lumin)

