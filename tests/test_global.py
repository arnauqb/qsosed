import pyagn.sed as sed
from pyagn import constants
from scipy import integrate
import numpy as np
from numpy import testing
from astropy import units as u

M_range = np.geomspace(1e6,1e10,5)
mdot_range = np.geomspace(0.1,1,5)

def test_conservation_energy(M,mdot):
    for M in M_range:
        for mdot in mdot_range:
            sed_test = sed.SED(M = M, mdot = mdot)
            distance = 1e20
            bol_lumin = sed_test.bolometric_luminosity
            total_spectral_flux = sed_test.total_flux(distance)
            total_flux = integrate.trapz(x = sed_test.ENERGY_RANGE_KEV, y = total_spectral_flux / sed_test.ENERGY_RANGE_KEV)
            total_flux_erg = sed.convert_units(total_flux * u.keV, u.erg)
            total_lumin = total_flux_erg * 4 * np.pi * distance**2
            testing.assert_approx_equal(total_lumin, bol_lumin, significant = 1)
