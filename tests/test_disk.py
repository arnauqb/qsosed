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

def test_schwarzschild_radius():
    testing.assert_approx_equal(2 * sed_test.Rg, 2.95337e13, significant = 4)

def test_efficiency():
    testing.assert_approx_equal(sed_test.efficiency, 0.05719, significant = 4)

def test_isco():
    testing.assert_approx_equal(sed_test.isco, 6.)
    sed_test_2 = sed.SED(astar = 0.998)
    testing.assert_approx_equal(sed_test_2.isco, 1.23, significant = 2)

def test_gravity_radius():
    XSPEC_GR = 1581.6308089974727
    XSPEC_T_AT_GR = 4047.1865836535567
    testing.assert_approx_equal(sed_test.gravity_radius, XSPEC_GR)
    t_at_gr = sed_test.disk_nt_temperature4(sed_test.gravity_radius)**(1./4.)
    testing.assert_approx_equal(t_at_gr, XSPEC_T_AT_GR)

def test_nt_rel_factors():
    testing.assert_equal(sed_test._nt_rel_factors(sed_test.isco),0)

def test_eddington_luminosity():
    testing.assert_approx_equal(sed_test.eddington_luminosity, 1.257e46, significant = 5)

def test_accretion_rate():
    EDD_ACC_RATE = sed_test.eddington_luminosity / ( 0.1 * constants.c**2)
    testing.assert_approx_equal(EDD_ACC_RATE, 1.39916e26, significant = 2)

def test_gravity_radius():
    testing.assert_approx_equal(sed_test.gravity_radius, 1579.9645, significant = 3)

def test_disk_spectral_radiance():
    r_range = [200,400,600] 
    for r in r_range:
        radiance = sed_test.disk_radiance(r)
        radiance_2 = np.pi * integrate.trapz( x= sed_test.ENERGY_RANGE_ERG, y = sed_test.disk_spectral_radiance(sed_test.ENERGY_RANGE_ERG, r))
        testing.assert_approx_equal(radiance, radiance_2, significant = 2)
    
def test_disk_radiance():
    """
    Checks that int_isco^rout radiance = bol. luminosity
    """
    const = (sed_test.Rg)**2 * 4 * np.pi
    integral_radiance = const * integrate.quad(lambda r: r * sed_test.disk_radiance(r), sed_test.isco, sed_test.gravity_radius)[0]
    testing.assert_approx_equal(integral_radiance, sed_test.bolometric_luminosity, significant = 2)

def test_disk_truncated_spectral_luminosity():
    sed_test.disk_rin = sed_test.isco
    total_lumin_erg = []
    for energy_erg in sed_test.ENERGY_RANGE_ERG:
        total_lumin_erg.append(sed_test.disk_spectral_luminosity(energy_erg))

    total_lumin = integrate.trapz(x = sed_test.ENERGY_RANGE_ERG, y = total_lumin_erg)
    total_lumin2 = integrate.quad(sed_test.disk_spectral_luminosity, sed_test.ENERGY_MIN_ERG, sed_test.ENERGY_MAX_ERG)[0]
    testing.assert_approx_equal(total_lumin, total_lumin2, significant = 1)
    testing.assert_approx_equal(total_lumin2, sed_test.bolometric_luminosity, significant = 2)




