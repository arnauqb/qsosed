"""
Tests involving the disc component.
"""
import unittest
import agnsed.sed as sed
from scipy import integrate
import numpy as np
from astropy import units as u

sed_test = sed.SED()
energy_range_erg = sed.convert_units(sed_test.energy_range * u.keV, u.erg)

class test_disc_blackbody(unittest.TestCase):
    """
    Tests to assure we integrate from radiance to luminosity correctly.
    """
    def test_disc_radiance(self):
        """
        Checks that int_isco^rout radiance = bol. luminosity
        """
        const = (sed_test.Rg)**2 * 4 * np.pi
        integral_radiance = integrate.quad(lambda r: r * sed_test.disc_radiance(r), sed_test.isco, sed_test.r_max)[0]
        integral_radiance *= const
        self.assertAlmostEqual(integral_radiance / sed_test.bolumetric_luminosity,1., places = 2)
        
    def test_disc_spectral_luminosity(self):
        """
        Check that the energy integral of the spectral luminosity is the total luminosity.
        """
        lumin_e_list = []
        for energy in energy_range_erg:
            lumin_e = sed_test.disc_spectral_luminosity(energy)
            lumin_e_list.append(lumin_e)
        integ = integrate.simps(x = energy_range_erg, y = lumin_e_list)
        self.assertAlmostEqual( integ / sed_test.bolumetric_luminosity ,1., places = 2)
        
if __name__ == '__main__':
    
    unittest.main()
        
        