import pytest
import pyagn.sed as sed
from pyagn import constants
from scipy import integrate
import numpy as np
from numpy import testing
from astropy import units as u

M_range = [1e8]#np.geomspace(1e6,1e10,5)
mdot_range = [0.5] #np.geomspace(0.1,1,5)

def test_conservation_energy():
    for M in M_range:
        for mdot in mdot_range:
            print("M = %e \t mdot = %.2f"%(M, mdot))
            sed_test = sed.SED(M = M, mdot = mdot, reprocessing = True)
            distance = 1e20
            bol_lumin = sed_test.bolometric_luminosity
            total_spectral_flux = sed_test.total_flux(distance)
            total_flux = integrate.trapz(x = sed_test.ENERGY_RANGE_KEV, y = total_spectral_flux)
            total_flux_erg = sed.convert_units(total_flux * u.keV, u.erg)
            total_lumin = total_flux_erg * 4 * np.pi * distance**2
            print(total_lumin, bol_lumin)
            testing.assert_approx_equal(total_lumin, bol_lumin, significant = 1)

test_conservation_energy()
