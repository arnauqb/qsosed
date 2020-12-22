"""
This module computes the AGN Spectral Energy Density in the UV/X-Ray energy range, following Kubota & Done (2018). This is equivalent to AGNSED in XSPEC.
Written by Arnau Quera-Bofarull (arnau.quera-bofarull@durham.ac.uk) in Durham, UK.
"""
import numpy as np
import qsosed.constants as const
from scipy import integrate, optimize
from astropy import units as u
from qsosed.xspec_routines import donthcomp
 
def convert_units(old, new_unit):
    """
    Convert units using astropy spectral equivalence.

    Parameters

    old : astropy units quantity
          value with units that needs conversion

    new_unit : astropy unit
          target unit
    """
    new = old.to( new_unit, equivalencies = u.spectral() )
    return new.value

def _nt_rel_factors(r, spin, isco):
        """
        Relatistic A,B,C factors of the Novikov-Thorne model.
        
        Parameters
            Black Hole Mass in solar Masses
        -----------
        r : float
            disk radial distance.
        """

        yms = np.sqrt(isco)
        y1 = 2 * np.cos((np.arccos(spin) - np.pi) / 3)
        y2 = 2 * np.cos((np.arccos(spin) + np.pi) / 3)
        y3 = -2 * np.cos(np.arccos(spin) / 3)
        y = np.sqrt(r)
        C = 1 - 3 / r + 2 * spin / r**(1.5)
        B = 3 * (y1 - spin)**2 * np.log(
            (y - y1) / (yms - y1)) / (y * y1 * (y1 - y2) * (y1 - y3))
        B += 3 * (y2 - spin)**2 * np.log(
            (y - y2) / (yms - y2)) / (y * y2 * (y2 - y1) * (y2 - y3))
        B += 3 * (y3 - spin)**2 * np.log(
            (y - y3) / (yms - y3)) / (y * y3 * (y3 - y1) * (y3 - y2))
        A = 1 - yms / y - 3 * spin * np.log(y / yms) / (2 * y)
        factor = (A-B)/C
        return factor

def blackbody_spectral_radiance(energy, T):
        """
        disk spectral radiance in units of  1 / cm^2 / s / sr, assuming black-body radiation.

        Parameters
        ----------
        energy : float
             Energy in erg.
        r :  float
             disk radius in Rg.
        """
        bb_constant = 2 / (const.h**3 * const.c ** 2)
        planck_spectrum_exp = np.exp( energy / ( const.k_B *  T))
        planck_spectrum = bb_constant * energy**3 * 1./ ( planck_spectrum_exp - 1)
        return planck_spectrum


class SED:
    """
    Class to handle the AGN SED calculation functions. Implements Kubota & Done (2018) paper.
    """
    ENERGY_MIN = 1e-4 # keV
    ENERGY_MIN_ERG = convert_units(ENERGY_MIN * u.keV, u.erg) 
    ENERGY_MAX = 200. # keV
    ENERGY_MAX_ERG = convert_units(ENERGY_MAX * u.keV, u.erg) 
    ENERGY_RANGE_NUM_BINS = 500
    ENERGY_BIN_EDGES_KEV = np.geomspace(ENERGY_MIN, ENERGY_MAX, ENERGY_RANGE_NUM_BINS + 1)
    ENERGY_BIN_WIDTHS_KEV = np.diff(ENERGY_BIN_EDGES_KEV)
    ENERGY_RANGE_KEV = ENERGY_BIN_EDGES_KEV[:-1] + (ENERGY_BIN_EDGES_KEV[1:] - ENERGY_BIN_EDGES_KEV[:-1]) / 2. 
    ENERGY_RANGE_ERG = convert_units(ENERGY_RANGE_KEV * u.keV, u.erg) #np.geomspace(ENERGY_MIN_ERG, ENERGY_MAX_ERG, ENERGY_RANGE_NUM_BINS)
    ELECTRON_REST_MASS = 511. #kev
    ENERGY_UV_LOW_CUT_KEV = 0.00387
    ENERGY_UV_HIGH_CUT_KEV = 0.06
    ENERGY_XRAY_LOW_CUT_KEV = 0.1
    UV_MASK = (ENERGY_RANGE_KEV > ENERGY_UV_LOW_CUT_KEV) & (ENERGY_RANGE_KEV < ENERGY_UV_HIGH_CUT_KEV)

    def __init__(self, 
            M=1e8,
            mdot=0.5,
            spin=0,
            spin_sign=1,
            mu=1,
            reprocessing=False,
            hard_xray_fraction=0.02,
            corona_electron_energy=100,
            warm_electron_energy=0.2,
            warm_photon_index=2.5,
            reflection_albedo=0.3,
            number_bins_fractions=3000,
            ):

        # read parameters #
        self.M = M # black hole mass in solar masses
        self.mdot = mdot # mdot = Mdot / Mdot_Eddington
        self.spin = spin # dimensionless black hole spin
        self.spin_sign = spin_sign # +1 for prograde rotation, -1 for retrograde
        self.mu = mu

        # useful quantities #
        self.Rg = const.G * M * const.Ms / const.c ** 2 # gravitational radius
        self.number_bins_fractions = number_bins_fractions
        
        # model parameters
        self.hard_xray_fraction = hard_xray_fraction # fraction of energy in Eddington units in the corona.
        self.corona_electron_energy = corona_electron_energy # temperature of the corona's electrons in keV.
        self.warm_electron_energy = warm_electron_energy # temperature of the soft region's electrons in keV.
        self.warm_photon_index = warm_photon_index # powerlaw index of the warm component. 
        self.reflection_albedo = reflection_albedo # reflection albedo for the reprocessed flux.
        self.corona_radius = self.corona_find_radius
        self.corona_height = min(100., self.corona_radius)
        self.warm_radius = 2 * self.corona_radius#0.87 * 2 * self.corona_radius
        self.disc_alpha = 0.1
        
        # set reprocessing to false to compute corona luminosity
        self.reprocessing = False
        self.corona_luminosity = self.corona_compute_luminosity
        self.reprocessing = reprocessing # set reprocessing to the desired value
        self.uv_fraction, self.xray_fraction = self.compute_uv_and_xray_fraction() 
        try:
            assert(reprocessing in [False,True])#
        except:
            print("Reprocessing has to be either False (no reprocessing) or True (include reprocessing).")

            
    @property
    def isco(self):
        """
        Computes the Innermost Stable Circular Orbit. Depends only on spin.
        """
        
        z1 = 1 + (1 - self.spin**2)**(1 / 3) * ((1 + self.spin)**(1 / 3) + (1 - self.spin)**(1 / 3))
        z2 = np.sqrt(3 * self.spin**2 + z1**2)
        rms = 3 + z2 - self.spin_sign * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
        return rms

        
    @property
    def efficiency(self):
        """ 
        Accretion Efficiency

        Parameters
        ----------
        isco :  float
                Innermost stable circular orbit
        """ 
        
        eta = 1 - np.sqrt( 1 - 2 / (3 * self.isco) )
        return eta

    @property
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

    @property
    def bolometric_luminosity(self):
        """
        Bolometric Luminosity given by L = mdot * Ledd.
        """
        return self.eddington_luminosity * self.mdot
    
    @property
    def mass_accretion_rate(self):
        """
        Mass Accretion Rate in units of g/s.
        """
        Mdot = self.mdot * self.eddington_luminosity / ( self.efficiency * const.c**2)
        return Mdot
    
    @property
    def gravity_radius(self):
        """
        Self-gravity radius as described by Laor & Netzer (1989).
        """
        mass = (self.M / 1e9)
        alpha = 0.1 # assumption
        r_sg = 2150 * mass**(-2./9.) * self.mdot**(4./9.) * alpha**(2./9.)
        return r_sg
        
    """
    disk functions.
    """
    def disk_nt_temperature4(self, r):
        """
        Computes Novikov-Thorne temperature in Kelvin (to the power of 4) of accretion disk annulus at radius r.
        Parameters
        ----------
        r : float
            disk radius in Rg. 
        """
        nt_constant = 3 * const.m_p * const.c**5 / ( 2 * const.sigma_sb * const.sigma_t * const.G * const.Ms)
        rel_factor = _nt_rel_factors(r, self.spin, self.isco)
        aux = self.mdot / (self.M * self.efficiency * r**3) 
        t4 = nt_constant * rel_factor * aux
        return t4

    def disk_core_temperature(self,r):
        """
        Returns temperature at the disc midplane, following "Accretion Power in Astrophysics, ed.3, p. 244".
        """
        constant = 1.4e6
        alpha = self.disc_alpha
        mdot = (self.mass_accretion_rate / 1e26)
        m = self.M / 1e8
        f = ( 1 - np.sqrt(self.isco/ r ) )
        r = (r * self.Rg / 1e14)
        T = constant * alpha**(-1./5.) * mdot**(3/10) * m**(1/4) * r**(-3./4.) * f**(6./5.)
        return T

    def disk_number_density(self, r):
        """
        Density at the surface of the accretion disc following the Shakura-Sunyaev model. Formula copied from N18 (scaling luminosity).
        """
        R = r * 2 * self.Rg
        cut = 18 * (self.M)**(2./21.) * \
            (self.mdot / self.efficiency)**(16./21.) * 2 * self.Rg
        if (R <= cut):
            rho = 5.24e-4 * (self.M)**(-1.) * \
                (self.mdot/self.efficiency)**(-2.) * (R/(2*self.Rg))**(3./2.)
        else:
            rho = 4.66 * (self.M)**(-7./10.) * (self.mdot /
                                                            self.efficiency)**(2./5.) * (R/(2*self.Rg))**(-33./20.)
        return rho / (const.m_p * self.mu)

    def disk_gas_pressure(self,r):
        """
        Gas pressure of the disk.
        """
        density = self.disk_number_density(r) * const.m_p * self.mu
        temperature = self.disk_nt_temperature4(r)**(1./4.)
        gas_pressure = density * const.k_B * temperature / (self.mu * const.m_p ) 
        return gas_pressure

    def disk_radiation_pressure(self,r):
        """
        Radiation pressure of the disc.
        """
        temperature4 = self.disk_nt_temperature4(r)
        rad_pressure = 4 * const.sigma_sb * temperature4 / ( 3 * const.c)
        return rad_pressure

    def disk_total_pressure(self,r):
        return self.disk_gas_pressure(r) + self.disk_radiation_pressure(r)

    def disk_scale_height(self, r):
        """
        Disk scale height H, obtained by solving the vertical hydrostatic equilibrium equation, P/H = rho(GMH/R^3), and ignoring the disc self-gravity.
        """
        pressure = self.disk_total_pressure(r)
        R = r * self.Rg
        density = self.disk_number_density(r) * const.m_p
        H = np.sqrt(pressure * R**3 / (const.G * self.M * const.Ms * density))
        h = H / self.Rg
        return h
    
    def reprocessed_flux(self, radius):
        """
        Reprocessed flux as given by eq. 5 of Kubota & Done (2018).
        """
        R = radius * self.Rg
        M = self.M * const.Ms
        Lhot = self.corona_luminosity
        H = self.corona_radius * self.Rg
        a = self.reflection_albedo
        aux = 3. * const.G * M / ( 8 * np.pi * R**3.)
        aux *= 2 * Lhot / (const.c**2)
        aux *= H / (6 * self.Rg) * (1.-a)
        aux *= (1. + (H/R)**2)**(-3./2.)
        return aux
    
    def disk_temperature4(self,r):
        """
        disk effective temperature. This takes into account reprocessing.
        """
        radiance = self.disk_nt_temperature4(r) 
        if( self.reprocessing ):
            radiance += self.reprocessed_flux(r) / const.sigma_sb
        teff4 = radiance 
        return teff4

    def disk_spectral_radiance(self, energy, r):
        T = self.disk_temperature4(r) ** (1./4.)
        ps = blackbody_spectral_radiance(energy, T)
        return ps
        
    def disk_spectral_radiance_kev(self, energy, r):
        """
        disk spectral radiance in units of  1 / cm^2 / s / sr, assuming black-body radiation.
        Parameters
        ----------
        energy : float
             Energy in keV.
        r :  float
             disk radius in Rg.
        """
        energy_erg = convert_units(energy * u.keV, u.erg)
        planck_spec = self.disk_spectral_radiance(energy_erg, r)
        return planck_spec
        

    def disk_radiance(self, r):
        """
        disk radiance in units of erg / cm^2 / s, assuming black-body radiation.

        Parameters
        ----------
        r : float
            disk radius in Rg.
        """
        radiance = const.sigma_sb * self.disk_temperature4(r) 
        return radiance

    def disk_radiance_kev(self, r):
        """
        disk radiance in units of kev / cm^2 / s / sr, assuming black-body radiation.

        Parameters
        ----------
        r : float
            disk radius in Rg.
        """
        radiance_erg = const.sigma_sb * self.disk_temperature4(r)
        radiance_kev = convert_units(radiance_erg * u.erg, u.keV)
        return radiance_kev

    def disk_spectral_luminosity(self, energy):
        """
        disk spectral luminosity in units of 1 / s.

        Parameters
        ----------
        energy : float
            Energy in erg.
        """
        radial_integral = 2 * np.pi**2 * self.Rg**2 * integrate.quad( lambda r: r * self.disk_spectral_radiance(energy, r), self.warm_radius, self.gravity_radius)[0] # units of 1 / s 
        spectral_lumin = 2 * radial_integral # 2 sides of the disk
        return spectral_lumin
    
    @property
    def disk_luminosity(self):
        """
        disk Luminosityin units of erg / s.
        """
        constant =  const.sigma_sb * 4 * np.pi * self.Rg**2
        lumin = constant * integrate.quad(lambda r: r*self.disk_temperature4(r), self.warm_radius, self.gravity_radius)[0]
        return lumin

    def disk_truncated_luminosity(self, r_min, r_max):
        """
        disk Luminosity in units of erg / s.

        Parameters
        ----------
        r_in : float
               Inner disk radius. Defaults to ISCO.
        r_max: float
                Outer disk radius. Defaults to 1400Rg.
        """
        if(r_min == None):
            r_min = self.isco
        if(r_max == None):
            r_max = self.gravity_radius
        constant =  const.sigma_sb * 4 * np.pi * self.Rg**2
        lumin = constant * integrate.quad(lambda r: r*self.disk_nt_temperature4(r), r_min, r_max)[0]
        return lumin

    @property
    def disk_sed(self):
        """
        disk SED in energy units.
        EL_E[ KeV KeV / s / KeV]
        """
        lumin = []
        for energy_erg in self.ENERGY_RANGE_ERG:
            lumin.append(self.disk_spectral_luminosity(energy_erg))
        sed = np.array(lumin) * self.ENERGY_RANGE_KEV
        return sed
    
    def disk_flux(self, distance):
        """
        Flux of the disk component in units of keV^2 ( Photons / cm^2 / s / keV).
        
        Parameters
        ----------
        distance: float
                  Distance to the source in cm.
        """
        return self.disk_sed / self.ENERGY_RANGE_KEV / (4*np.pi*distance**2) # keV / keV / s / cm^2
    
    def disk_flux_r(self, r, dr, distance, exclude_warm_region=True):
        """
        Spectral energy flux from the disc at a radius r.

        Args:
        r : radius in Rg.
        dr : annulus width in Rg.
        distance: distance from the source in cm.
        """
        if ( r <= self.warm_radius and exclude_warm_region):
            return np.zeros(len(self.ENERGY_RANGE_KEV))

        disk_energy_flux_bin = np.pi * self.disk_spectral_radiance(self.ENERGY_RANGE_ERG, r) # flux in mid bin units 1 / cm^2 / s
        disk_energy_flux = disk_energy_flux_bin / self.ENERGY_RANGE_KEV # 1 / cm^2 / s / keV
        disk_lumin = 4 * np.pi * (self.Rg)**2 * r * dr * disk_energy_flux  # 1 / s / keV
        disk_energy_flux = disk_lumin / ( 4 * np.pi * distance**2) * self.ENERGY_RANGE_KEV  # keV / cm^2 / s / keV
        return disk_energy_flux
    
    """
    Corona section. Hot compton thin region, responsible for hard X-Ray emission.
    """

    @property
    def corona_dissipated_luminosity(self):
        """
        Intrinsic luminosity from the Corona. This is assumed to be a constant fraction of the Eddington luminosity,
        regardless of actual accretion rate.
        """

        cor_dissip_lumin = self.hard_xray_fraction * self.eddington_luminosity
        return cor_dissip_lumin

    def _corona_compute_radius_kernel(self, r_cor):
        """
        Auxiliary function to compute corona radius.

        Parameters
        ----------
        r_cor : float
                Candidate corona radius.
        """

        truncated_disk_lumin = self.disk_truncated_luminosity(r_min = self.isco, r_max = r_cor)
        lumin_diff = truncated_disk_lumin - self.corona_dissipated_luminosity
        return lumin_diff

    @property
    def corona_find_radius(self):
        """
        Computes corona radius in Rg.
        """

        try:
            corona_radius = optimize.brentq(self._corona_compute_radius_kernel, self.isco, self.gravity_radius)
        except:
            print("Accretion rate is too low to power a corona. Radius is smaller than last circular stable orbit.")
            corona_radius = 0
        return corona_radius

    def _corona_covering_factor(self, r):
        """
        Corona covering factor as seen from the disk at radius r > r_cor.

        Parameters
        ----------
        r : float
            Observer disk radius.
        """

        if ( r < self.corona_radius):
            print("Radius smaller than corona radius!")
            return None
        theta_0 = np.arcsin( self.corona_height/ r)
        covering_factor = theta_0 - 0.5 * np.sin( 2 * theta_0 )
        return covering_factor

    @property
    def corona_seed_luminosity(self):
        """
        Seed photon luminosity intercepted from the warm region and the outer disk. 
        Calculated assuming a truncated disk and spherical hot flow geometry.
        """

        integral = integrate.quad( lambda r: r * self.disk_temperature4(r) * self._corona_covering_factor(r), self.corona_radius, self.gravity_radius)[0]
        constant = 4 * self.Rg **2 * const.sigma_sb 
        seed_lumin = constant * integral
        return seed_lumin
    
    @property
    def corona_compute_luminosity(self):
        """
        Total corona luminosity, given by the sum of the seed photons and the truncated disk flow.
        """
        corona_lum = self.corona_seed_luminosity + self.corona_dissipated_luminosity
        return corona_lum

    @property
    def corona_photon_index(self):
        """
        Photon index (Gamma) for the corona SED. The functional form is assumed to be
        L_nu = k nu ^(-alpha) = k nu^( 1 - gamma ), where alpha = gamma - 1
        Computed using equation 14 of Beloborodov (1999).
        """     
        #reproc = self.reprocessing
        #self.reprocessing = False
        gamma_cor = 7./3. * ( self.corona_dissipated_luminosity / self.corona_seed_luminosity )**(-0.1)
        #self.reprocessing = reproc
        return gamma_cor

    def compton_photon_flux(self, ear, params):
        photon_number_per_bin = donthcomp(ear = self.ENERGY_RANGE_KEV, param = params) # units of Photons / cm^2 / s 
        photon_number_flux = photon_number_per_bin / self.ENERGY_RANGE_KEV # units of Photons /cm^2 / s / keV
        return photon_number_flux

    def corona_photon_flux(self):
        """
        Corona flux computed using donthcomp from Xspec.
        """
        gamma = self.corona_photon_index
        kt_e = self.corona_electron_energy
        t_corona = self.disk_temperature4(self.corona_radius)**(1./4.) * const.k_B
        t_corona_kev = convert_units(t_corona * u.erg, u.keV)
        ywarm = (4./9. * self.warm_photon_index) ** (-4.5)
        params = [gamma, kt_e, t_corona_kev * np.exp(ywarm), 0, 0]
        #photon_number_flux = donthcomp(ear = self.ENERGY_RANGE_KEV, param = params) # units of Photons / cm^2 / s 
        photon_number_flux = self.compton_photon_flux(ear=self.ENERGY_RANGE_KEV, params=params)
        return photon_number_flux

    def corona_flux(self, distance):
        """
        Corona flux computed using donthcomp from Xspec.
        
        Parameters
        ----------
        
        distance : float
                   distance to source.
        """
   
        # We integrate the flux only where is non-zero.
        photon_number_flux = self.corona_photon_flux()
        mask = photon_number_flux > 0
        flux_array = np.zeros(len(self.ENERGY_RANGE_KEV))
        #flux = integrate.simps(x=self.ENERGY_RANGE_KEV, y=photon_number_flux) # photons / cm^2 / s
        flux = np.sum(self.ENERGY_BIN_WIDTHS_KEV * photon_number_flux * self.ENERGY_RANGE_KEV) # photons / cm^2 /s
        flux_erg = convert_units(flux * u.keV, u.erg)
        # We renormalize to the correct distance.
        ratio = (self.corona_luminosity / (4 * np.pi* distance**2)) / flux_erg
        flux = ratio * photon_number_flux[mask] * self.ENERGY_RANGE_KEV[mask] # units of keV / cm^2 / s / keV
        flux_array[mask] = flux
        return flux_array

    """
    Warm region section.
    """ 

    def warm_flux_r(self, r, dr, distance):
        """
        Energy flux of the warm compton region. Units of keV / cm^2 / s.
        
        Parameters
        ----------
        
        radius : float
                 disk radius.
        """
        
        ff = np.zeros(self.ENERGY_RANGE_NUM_BINS)
        if ( r > self.warm_radius):
            return ff 
        # xspec parameters #
        gamma = self.warm_photon_index
        kt_e = self.warm_electron_energy
        t_warm = self.disk_temperature4(r)**(1./4.) * const.k_B
        t_warm_kev = convert_units(t_warm * u.erg, u.keV)
        params = [gamma, kt_e, t_warm_kev, 0, 0]
        photon_flux_r = self.compton_photon_flux(ear=self.ENERGY_RANGE_KEV, params=params) # 1 / cm^2 / s / keV
        mask = photon_flux_r > 0
        if(len(photon_flux_r[mask]) == 0):
            return ff
        #energy_flux_r_integrated = integrate.simps(x= self.ENERGY_RANGE_ERG[mask], y = photon_flux_r[mask]) # energy flux in keV / cm2 / s
        energy_flux_r_integrated = np.sum(self.ENERGY_BIN_WIDTHS_KEV[mask] * photon_flux_r[mask] * self.ENERGY_RANGE_KEV[mask]) # keV / cm^2 / s
        energy_flux_r_integrated = convert_units(energy_flux_r_integrated * u.keV, u.erg) 
        # we then normalize the flux using the local disc energy flux.
        disk_lumin = 4 * np.pi * (self.Rg)**2. * r * dr * self.disk_radiance(r)
        disk_flux = disk_lumin / (4. * np.pi * distance**2)
        ratio = disk_flux / energy_flux_r_integrated
        energy_flux_r = ratio * photon_flux_r * self.ENERGY_RANGE_KEV
        ff[mask] = energy_flux_r[mask]
        return ff 

    #@property
    def warm_flux(self, distance):
        """
        warm SED in energy units, [ KeV / s / KeV].
        """
        r_range = np.linspace(self.corona_radius, self.warm_radius, 500) # the soft-compton region extends form Rcor to 2Rcor.
        grid = np.zeros((len(r_range), len(self.ENERGY_RANGE_KEV)))
        dr = r_range[1] - r_range[0]
        for i,r in enumerate(r_range):
            ff = self.warm_flux_r(r, dr, distance)
            grid[i] = ff
        # we now integrate over all radii.
        flux_array = []
        for row in np.transpose(grid):
            energy_flux = integrate.simps(x = r_range, y = row / dr)
            flux_array.append(energy_flux)
        flux_array = np.array(flux_array) 
        return flux_array
    
    def total_flux(self, distance):
        """
        Total flux at distance.
        
        Parameters
        ----------
        distance : float
                   distance to source in cm.
        """
        
        disk_flux = self.disk_flux(distance)
        warm_flux = self.warm_flux(distance)
        corona_flux = self.corona_flux(distance)
        total_flux = disk_flux + warm_flux + corona_flux
        return total_flux

    def compute_uv_and_xray_fraction(self):
        """
        Computes the UV to X-Ray ratio from the SED.
        We consider X-Ray all the ionizing radiation above 0.1 keV,
        and UV all radiation between 0.00387 keV and 0.06 keV.
        """

        sed_flux = self.total_flux(1e26)
        xray_mask = self.ENERGY_RANGE_KEV > self.ENERGY_XRAY_LOW_CUT_KEV
        uv_mask = self.UV_MASK
        xray_flux = sed_flux[xray_mask]
        uv_flux = sed_flux[uv_mask]
        xray_energy_range = self.ENERGY_RANGE_KEV[xray_mask]
        uv_energy_range = self.ENERGY_RANGE_KEV[uv_mask]
        xray_int_flux = integrate.simps(x = xray_energy_range, y = xray_flux)
        uv_int_flux = integrate.simps(x = uv_energy_range, y = uv_flux)
        total_flux = integrate.simps(x = self.ENERGY_RANGE_KEV, y = sed_flux)
        uv_fraction = uv_int_flux / total_flux
        xray_fraction = xray_int_flux / total_flux
        return uv_fraction, xray_fraction

    def compute_uv_fraction_radial(self, r, dr, distance):
        """
        Auxiliary function to compute the radial fraction of UV luminosity.
        """
        component_fractions = []
        fluxes = []
        if ( r == self.corona_radius):
            corona_flux = self.corona_flux(distance)
            fluxes.append(corona_flux)

        else:
            corona_flux = np.zeros(self.ENERGY_RANGE_NUM_BINS)
            fluxes.append(corona_flux)

        warm_flux = self.warm_flux_r(r, dr, distance)
        fluxes.append(warm_flux)
        disk_flux = self.disk_flux_r(r, dr, distance)
        fluxes.append(disk_flux)
        total_flux = np.sum(fluxes, axis = 0)
        assert(len(total_flux) == self.ENERGY_RANGE_NUM_BINS)
        mask = total_flux > 0
        uv_mask = mask & (self.UV_MASK)
        int_total_flux = integrate.simps(x = self.ENERGY_RANGE_KEV[mask], y = total_flux[mask])
        int_total_flux_uv = integrate.simps(x = self.ENERGY_RANGE_KEV[uv_mask], y = total_flux[uv_mask])
        fraction_total = int_total_flux_uv / int_total_flux
        int_uv_flux = 0
        for flux in fluxes:
            mask = flux > 0
            uv_mask = mask & (self.UV_MASK)
            flux_uv = flux[uv_mask]
            if len(flux_uv) == 0:
                component_fractions.append(0)
                continue
            energy_range_uv = self.ENERGY_RANGE_KEV[uv_mask]
            int_uv_flux = integrate.simps(x = energy_range_uv, y = flux_uv)
            fraction = int_uv_flux / int_total_flux
            component_fractions.append(fraction)

        return [fraction_total, int_total_flux_uv, int_total_flux, component_fractions]
    
    def compute_uv_fractions(self, inner_radius = "warm_radius", outer_radius = 1000, distance=1e20, log_spaced=False, include_corona = False, return_all = True):
        """
        Computes the fraction of UV luminosity to the total UV luminosity at each radii. Return the fraction list, and the UV and total flux (optional).

        Args:
        distance: distance from the object in cm.
        return_flux: whether to retun the uv/total flux or not.
        include_corona: whether to include the radiation from the corona in the calculation or not.
        """
        #if(include_corona):
        #    r_in = self.corona_radius
        #else:
        #    r_in = self.warm_radius
        #r_range = np.geomspace(r_in, self.gravity_radius, 1000)
        #d_log_r = np.log10(r_range[1]) - np.log10(r_range[0])
        #dr = r_range[1] - r_range[0]
        if inner_radius == "warm_radius":
            inner_radius = self.warm_radius
        if log_spaced:
            r_range = np.geomspace(inner_radius, outer_radius, self.number_bins_fractions)
            dr = np.log10(r_range[1]) - np.log10(r_range[0])
            border_value = r_range[-1] + 10**(np.log10(r_range[-1]) + dr) 
        else:
            r_range = np.linspace(inner_radius, outer_radius, self.number_bins_fractions)
            dr = r_range[1] - r_range[0]
            border_value = r_range[-1] + dr

        dr_range = np.diff(r_range, append=border_value) #self.uv_fraction_radius_range[1] - self.uv_fraction_radius_range[0]
        fraction_list = []
        total_uv_flux = 0
        total_flux = 0
        component_fractions_list = []
        for i, r in enumerate(r_range):
            if r <= self.corona_radius:
                fraction_list.append(0)
                component_fractions_list.append([0,0])
                continue
            dr = dr_range[i]
            uv_fraction, int_uv_flux, int_total_flux, component_fractions = self.compute_uv_fraction_radial(r, dr, distance)
            total_uv_flux += int_uv_flux
            total_flux += int_total_flux
            fraction_list.append(uv_fraction)
            component_fractions_list.append(component_fractions)
        if(return_all):
            return [fraction_list, total_uv_flux, total_flux, np.array(component_fractions_list)]
        else:
            return r_range, np.array(fraction_list)
