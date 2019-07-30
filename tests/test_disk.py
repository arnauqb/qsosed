"""
Tests involving the disk component.
"""
import pyagn.sed as sed
from scipy import integrate
import numpy as np
from numpy import testing
from astropy import units as u

sed_test = sed.SED()

def test_disk_radiance():
    """
    Checks that int_isco^rout radiance = bol. luminosity
    """
    const = (sed_test.Rg)**2 * 4 * np.pi
    integral_radiance = integrate.quad(lambda r: r * sed_test.disk_radiance(r), sed_test.isco, sed_test.gravity_radius)[0]
    integral_radiance *= const
    testing.assert_almost_equal(integral_radiance / sed_test.bolumetric_luminosity,1., decimal = 2)

def test_efficiency():
    testing.assert_almost_equal(sed_test.efficiency, 0.05719, decimal = 4)

def test_isco():
    testing.assert_almost_equal(sed_test.isco, 6.)
    sed_test_2 = sed.SED(astar = 0.998)
    testing.assert_almost_equal(sed_test_2.isco, 1.23, decimal = 2)

def test_nt_rel_factors():
    testing.assert_equal(sed_test._nt_rel_factors(sed_test.isco),0)

