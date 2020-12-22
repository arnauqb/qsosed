import pytest
import qsosed.sed as sed
from qsosed.sed import convert_units
from qsosed import constants
from scipy import integrate
import numpy as np
from numpy import testing
from astropy import units as u

sed_test = sed.SED(M=1e8, mdot=0.5, reprocessing=True)
distance = 100  # Mpc
distance_cm = convert_units(distance * u.Mpc, u.cm)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def test_setup():
    XSPEC_GAMMA_WARM = 2.5
    XSPEC_KT_WARM = 0.2
    XSPEC_ALBEDO = 0.3
    XSPEC_LDISS_HOT = 2.0183853632627917e-2  # Ledd

    testing.assert_equal(sed_test.warm_electron_energy, XSPEC_KT_WARM)
    testing.assert_equal(sed_test.warm_photon_index, XSPEC_GAMMA_WARM)
    testing.assert_equal(sed_test.reflection_albedo, XSPEC_ALBEDO)
    testing.assert_approx_equal(
        sed_test.hard_xray_fraction, XSPEC_LDISS_HOT, significant=2
    )


def test_corona():
    XSPEC_GAMMA_HOT = 2.2826826043284854
    XSPEC_RHOT = 8.9008961989970832
    XSPEC_T_AT_RHOT = 107102.98423666063
    XSPEC_T_SEED = 202220.98528710150
    XSPEC_L_DISS_HOT = 2.0183853632627917e-2
    XSPEC_LHOT = 3.6390444796974425e-002

    testing.assert_approx_equal(sed_test.corona_radius, XSPEC_RHOT, significant=3)
    t_at_rhot = sed_test.disk_nt_temperature4(sed_test.corona_radius) ** (1.0 / 4.0)
    testing.assert_approx_equal(t_at_rhot, XSPEC_T_AT_RHOT, significant=2)
    testing.assert_approx_equal(
        sed_test.corona_dissipated_luminosity / sed_test.eddington_luminosity,
        XSPEC_L_DISS_HOT,
        significant=2,
    )
    testing.assert_approx_equal(
        sed_test.corona_luminosity / sed_test.eddington_luminosity,
        XSPEC_LHOT,
        significant=1,
    )
    testing.assert_approx_equal(
        sed_test.corona_photon_index, XSPEC_GAMMA_HOT, significant=3
    )


def test_corona_luminosity():
    dist = 1e23
    corona_lumin = sed_test.corona_luminosity
    corona_spectral_flux = sed_test.corona_flux(dist)
    corona_flux = integrate.trapz(x=sed_test.ENERGY_RANGE_KEV, y=corona_spectral_flux)
    corona_lumin_calc = corona_flux * 4 * np.pi * dist ** 2
    corona_lumin_calc = convert_units(corona_lumin_calc * u.keV, u.erg)
    testing.assert_approx_equal(corona_lumin_calc, corona_lumin, significant=4)


# TODO
# def test_corona_flux():
#    XSPEC_PHOTON_ENERGY_ERG = 2.96335734e-11
#    XSPEC_FLUX = 1.08165039e-12
#    XSPEC_PHOTON_ENERGY_KEV = convert_units(XSPEC_PHOTON_ENERGY_ERG * u.erg, u.keV)
#
#    corona_flux = sed_test.corona_flux(distance_cm)
#    idx = find_nearest(sed_test.ENERGY_RANGE_KEV, XSPEC_PHOTON_ENERGY_KEV)
#    energy = sed_test.ENERGY_RANGE_KEV[idx]
#    flux = corona_flux[idx] * energy
#    flux_erg = convert_units(flux * u.keV, u.Hz)
#    testing.assert_approx_equal(flux_erg, XSPEC_FLUX)
