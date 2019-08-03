import pytest
import pyagn.sed as sed
from pyagn.sed import convert_units
from pyagn import constants
from scipy import integrate
import numpy as np
from numpy import testing
from astropy import units as u

sed_test = sed.SED(M=1e8, mdot = 0.5, reprocessing = True)
distance = 100 #Mpc
distance_cm = convert_units(distance * u.Mpc, u.cm) 

def test_setup():
    XSPEC_GAMMA_WARM = 2.5
    XSPEC_KT_WARM = 0.2
    XSPEC_ALBEDO = 0.3
    XSPEC_LDISS_HOT = 2.0183853632627917e-2 # Ledd

    testing.assert_equal(sed_test.warm_electron_energy, XSPEC_KT_WARM)
    testing.assert_equal(sed_test.warm_photon_index, XSPEC_GAMMA_WARM)
    testing.assert_equal(sed_test.reflection_albedo, XSPEC_ALBEDO)
    testing.assert_approx_equal(sed_test.hard_xray_fraction, XSPEC_LDISS_HOT, significant = 2)

