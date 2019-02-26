"""
This module computes the AGN Spectral Energy Density in the UV/X-Ray energy range, following Kubota & Done (2018). This is equivalent to AGNSED in XSPEC.
"""
import numpy as np
import agnsed.constants as const
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from astropy import units as u
from memoized_property import memoized_property as property
from agnsed.xspec_routines import donthcomp


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

class SED:
    """
    Class to handle the AGN SED calculation functions. Implements Kubota & Done (2018) paper.
    """

    energy_min = 1e-4 # keV
    energy_max = 200. # keV
    energy_range = np.geomspace(energy_min, energy_max, 100)
    energy_range_erg = convert_units(energy_range * u.keV, u.erg) 
    freq_min = convert_units(energy_min * u.keV, u.Hz)
    freq_max = convert_units(energy_max * u.keV, u.Hz )
    freq_range = np.geomspace(freq_min, freq_max, 100)
    r_max = 1400.

    def __init__(self, M = 1e8, mdot = 0.5, astar = 0, sign = 1, hard_xray_fraction = 0.02, corona_electron_energy = 100, warm_electron_energy = 0.2, warm_photon_index = 2.5,  reprocessing = False, reflection_albedo = 0.3):

        # read parameters #
        self.M = M # black hole mass in solar masses
        self.mdot = mdot # mdot = Mdot / Mdot_Edd
        self.astar = astar # dimensionless black hole spin
        self.spin_sign = sign # +1 for prograde rotation, -1 for retrograde

        # useful quantities #
        self.Rg = const.G * M * const.Ms / const.c ** 2 # gravitational radius
        #self.isco = self.compute_isco() # Innermost Stable Circular Orbit
        #self.eta = self.compute_efficiency() # Accretion efficiency defined as L = \eta \dot M c^2
        self.hard_xray_fraction = hard_xray_fraction # fraction of energy in Eddington units in the corona.
        self.corona_electron_energy = corona_electron_energy
        self.warm_electron_energy = warm_electron_energy
        self.warm_photon_index = warm_photon_index
        self.reflection_albedo = reflection_albedo
        self.reprocessing = reprocessing
        self.electron_rest_mass = 511. #kev
        self.corona_height = min(100., self.corona_radius)
        try:
            assert(reprocessing in [False,True])#
        except:
            print("Reprocessing has to be either False (no reprocessing) or True (include reprocessing)")
        # corona variables

    @property
    def isco(self):
        """
        Computes the Innermost Stable Circular Orbit. Depends only on astar.
        """
        z1 = 1 + (1 - self.astar**2)**(1 / 3) * ((1 + self.astar)**(1 / 3) + (1 - self.astar)**(1 / 3))
        z2 = np.sqrt(3 * self.astar**2 + z1**2)
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

    def _nt_rel_factors(self, r=6):
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
    def bolumetric_luminosity(self):
        """
        Bolumetric Luminosity given by L = mdot * Ledd.
        """
        return self.eddington_luminosity * self.mdot
    
    @property
    def mass_accretion_rate(self):
        """
        Mass Accretion Rate in units of g/s.
        """

        Mdot = self.mdot * self.eddington_luminosity() / ( self.efficiency * const.c**2)
        return Mdot

    """
    Disc functions.
    """

    def disc_nt_temperature4(self, r):
        """
        Computes Novikov-Thorne temperature in Kelvin (to the power of 4) of accretion disc annulus at radius r.
        Parameters
        ----------
        r : float
            Disc radius in Rg. 
        """

        nt_constant = 3 * const.m_p * const.c**5 / ( 2 * const.sigma_sb * const.sigma_t * const.G * const.Ms)
        rel_factor = self._nt_rel_factors( r )
        aux = self.mdot / ( self.M * self.efficiency * r**3 )
        t4 = nt_constant * rel_factor * aux
        return t4

    def disc_temperature4(self,r):
        """
        Disc effective temperature. This takes into account reprocessing.
        """
        radiance = self.disc_nt_temperature4(r) 
        if( self.reprocessing ):
            radiance += self.reprocessed_flux(r) / const.sigma_sb
        teff4 = radiance 
        return teff4

    def disc_spectral_radiance_freq(self, nu, r):
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
        temperature = self.disc_temperature4(r) ** (1./4.)
        planck_spectrum_exp = np.exp( const.h * nu / ( const.k_B *  temperature ))
        planck_spectrum = bb_constant * nu**3 * 1./ ( planck_spectrum_exp - 1)
        return planck_spectrum

    def disc_spectral_radiance(self, energy, r):
        """
        Disc spectral radiance in units of  1 / cm^2 / s / sr, assuming black-body radiation.

        Parameters
        ----------
        energy : float
             Energy in erg.
        r :  float
             Disc radius in Rg.
        """

        bb_constant = 2 / (const.h**3 * const.c ** 2)
        temperature = self.disc_temperature4(r) ** (1./4.)
        planck_spectrum_exp = np.exp( energy / ( const.k_B *  temperature ))
        planck_spectrum = bb_constant * energy**3 * 1./ ( planck_spectrum_exp - 1)
        return planck_spectrum

    def disc_radiance(self, r):
        """
        Disc radiance in units of erg / cm^2 / s / sr, assuming black-body radiation.

        Parameters
        ----------
        r : float
            Disc radius in Rg.
        """

        radiance = const.sigma_sb * self.disc_temperature4(r)
        return radiance

    def disc_spectral_luminosity(self, energy):
        """
        Disc spectral luminosity in units of 1 / s.

        Parameters
        ----------
        energy : float
            Energy in erg.
        """
        radial_integral = 2 * np.pi**2 * self.Rg**2 * integrate.quad( lambda r: r * self.disc_spectral_radiance(energy,r), self.isco, self.r_max)[0]
        spectral_lumin = 2 * radial_integral # 2 sides of the disc
        #spectral_lumin = np.clip(spectral_lumin, a_min = 1, a_max = None)
        return spectral_lumin
    
    @property
    def disc_luminosity(self):
        """
        Disc Luminosityin units of erg / s.

        Parameters
        ----------
        r_in : float
               Inner disc radius. Defaults to ISCO.
        r_max: float
                Outer disc radius. Defaults to 1400Rg.
        """

        constant =  const.sigma_sb * 4 * np.pi * self.Rg**2
        lumin = constant * integrate.quad(lambda r: r*self.disc_temperature4(r), self.isco, self.r_max)[0]
        return lumin

    def disc_truncated_luminosity(self, r_min, r_max):
        """
        Disc Luminosity in units of erg / s.

        Parameters
        ----------
        r_in : float
               Inner disc radius. Defaults to ISCO.
        r_max: float
                Outer disc radius. Defaults to 1400Rg.
        """

        if(r_min == None):
            r_min = self.isco
        if(r_max == None):
            r_max = self.r_max

        constant =  const.sigma_sb * 4 * np.pi * self.Rg**2
        lumin = constant * integrate.quad(lambda r: r*self.disc_nt_temperature4(r), r_min, r_max)[0]
        return lumin

    @property
    def disc_sed(self):
        """
        Disc SED in energy units. First row is xaxis, second is yaxis.
        yaxis: EL_E[ KeV KeV / s / KeV]
        xaxis: E [KeV]
        """
        energy_range_erg = convert_units(self.energy_range * u.keV, u.erg)
        lumin = []
        for energy_erg in energy_range_erg:
            lumin.append(self.disc_spectral_luminosity(energy_erg))
        sed = np.array(lumin) * self.energy_range
        return sed
    
    def disc_flux(self, distance):
        return self.disc_sed / (4*np.pi*distance**2)
        
    @property
    def disc_peak_energy(self):
        """
        Energy of maximum disc emission.
        """

        arg_max = np.argmax(self.disc_sed)
        energy_max = self.energy_range[arg_max]
        return energy_max

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

        truncated_disc_lumin = self.disc_truncated_luminosity(r_min = self.isco, r_max = r_cor)
        lumin_diff = truncated_disc_lumin - self.corona_dissipated_luminosity
        return lumin_diff

    @property
    def corona_radius(self):
        """
        Computes corona radius.
        """

        try:
            corona_radius = optimize.brentq(self._corona_compute_radius_kernel, self.isco, self.r_max)
        except:
            print("Accretion rate is too low to power a corona. Radius is smaller than last circular stable orbit.")
            corona_radius = 0
        return corona_radius

    def _corona_covering_factor(self, r):
        """
        Corona covering factor as seen from the disc at radius r > r_cor.

        Parameters
        ----------
        r : float
            Observer disc radius.
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
        Calculated assuming a truncated disc and spherical hot flow geometry.
        """

        integral = integrate.quad( lambda r: r * self.disc_temperature4(r) * self._corona_covering_factor(r), self.corona_radius, self.r_max)[0]
        constant = 4 * self.Rg **2 * const.sigma_sb 
        seed_lumin = constant * integral
        return seed_lumin

    @property
    def corona_luminosity(self):
        """
        Total corona luminosity, given by the sum of the seed photons and the truncated disc flow.
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

        gamma_cor = 7./3. * ( self.corona_dissipated_luminosity / self.corona_seed_luminosity )**(-0.1)
        return gamma_cor

    def corona_flux(self, distance):
        """
        Corona SED computed using donthcomp from Xspec.
        """

        gamma = self.corona_photon_index
        kt_e = self.corona_electron_energy
        t_corona = self.disc_temperature4(self.corona_radius)**(1./4.) * const.k_B
        t_corona_kev = convert_units(t_corona * u.erg, u.keV)
        ywarm = (4./9. * self.warm_photon_index) ** (-4.5)
        params = [gamma, kt_e, t_corona_kev * np.exp(ywarm), 0, 0]
        photon_number_flux = donthcomp(ear = self.energy_range, param = params)
        mask = photon_number_flux > 0
        flux_array = np.zeros(len(self.energy_range))
        #flux = integrate.simps(x=self.energy_range_erg, y=photon_number_flux)
        flux = integrate.trapz(x= self.energy_range_erg, y=photon_number_flux)
        ratio = (self.corona_luminosity / (4 * np.pi* distance**2)) / flux
        flux = ratio * photon_number_flux[mask] * self.energy_range[mask]
        flux_array[mask] = flux
        return flux_array

    """
    Warm region section.
    """ 

    def warm_flux_r(self, radius):
        gamma = self.warm_photon_index
        kt_e = self.warm_electron_energy
        t_warm = self.disc_temperature4(radius)**(1./4.) * const.k_B
        t_warm_kev = convert_units(t_warm * u.erg, u.keV)
        params = [gamma, kt_e, t_warm_kev, 0, 0]
        #print(params)
        photon_number_flux = donthcomp(ear = self.energy_range, param = params)
        #flux = integrate.trapz(x= self.energy_range_erg, y=photon_number_flux)
        return photon_number_flux

    #@property
    def warm_flux(self, distance):
        """
        warm SED in energy units, [ KeV KeV / s / KeV].
        """

        r_range = np.linspace(self.corona_radius, 2. * self.corona_radius,10)
        grid = np.zeros((len(r_range), len(self.energy_range)))
        #flux_r_list = []
        
        for i,r in enumerate(r_range):
            flux_r_E = self.warm_flux_r(r)
            mask = flux_r_E>0
            flux_r = integrate.trapz(x= self.energy_range_erg[mask], y=flux_r_E[mask])
            disc_lumin = 4 * np.pi * (self.Rg)**2. * r * self.disc_radiance(r)
            disc_flux = disc_lumin / (4. * np.pi * distance**2)
            warm_lumin = 4 * np.pi * flux_r * r * (self.Rg**2)
            ratio = disc_flux / flux_r
            flux_r = ratio * flux_r_E[mask] * self.energy_range[mask]
            grid[i,mask] = flux_r
            #flux_r_list.append(np.array(flux_r))
            
        fluxes = []
        for row in np.transpose(grid):
            fluxes.append(integrate.simps(x = r_range, y = row))
        #flux_r_list = flux_r_list
        return fluxes
    
    def total_spectral_luminosity(self, nu):
        """
        Total spectral luminosity in units of [ erg / s ].

        Parameters
        ----------
        nu : float
             Frequency in Hz.
        """

        lumin_corona = self.corona_spectral_luminosity(nu)
        lumin_disc = self.disc_spectral_luminosity(nu)
        lumin_warm = self.warm_spectral_luminosity(nu)
        total = lumin_corona + lumin_disc + lumin_warm
        return total

    def reprocessed_flux(self, radius):
        """
        Reprocessed flux as given by eq. 5 of Kubota & Done (2018).
        """

        R = radius * self.Rg
        M = self.M * const.Ms
        Lhot = self.corona_luminosity
        H = self.corona_radius
        a = self.reflection_albedo
        aux = 3. * const.G * M / ( 8 * np.pi * R**3.)
        aux *= 2 * Lhot / (const.c**2)
        aux *= H / (6 * self.Rg) * (1.-a)
        aux *= (1. + (H/R)**2)**(-3./2.)
        return aux


# plotting routines 
    def disc_plot_sed_freq(self):
        """
        Plot disc SED in frequency units.
        yaxis: nuLnu [ Hz erg / s / Hz]
        xaxis: nu [Hz]
        """

        sed_freq = self.disc_sed_freq
        fig, ax = plt.subplots()
        ax.loglog(self.freq_range, sed_freq)
        ax.set_ylim(np.max(sed_freq)/1000, 2*np.max(sed_freq))
        ax.set_xlabel(r"Frequency $\nu$ [ Hz ]")
        ax.set_ylabel(r"$\nu \, L_\nu$  [ Hz erg / s / Hz ]")
        return fig, ax

    def disc_plot_sed_energy(self):
        """
        Plot disc SED in energy units.
        yaxis: EL_E [ keV keV / s / keV]
        xaxis: E [keV]
        """

        sed_energy = self.disc_sed_energy
        fig, ax = plt.subplots()
        ax.loglog(self.energy_range, sed_energy)
        ax.set_ylim(np.max(sed_energy)/1000, 2*np.max(sed_energy))
        ax.set_xlabel(r"Energy $E$ [ keV ]")
        ax.set_ylabel(r"$E \, L_E$  [ keV keV / s / keV]")
        return fig, ax

    def corona_plot_sed_freq(self):
        """
        Plot corona SED in frequency unitssel.
        yaxis: nuLnu [ Hz erg / s / Hz]
        xaxis: nu [Hz]
        """

        sed_freq = self.corona_sed_freq
        fig, ax = plt.subplots()
        ax.loglog(self.freq_range, sed_freq)
        ax.set_ylim(np.max(sed_freq)/1000, 2*np.max(sed_freq))
        ax.set_xlabel(r"Frequency $\nu$ [ Hz ]")
        ax.set_ylabel(r"$\nu \, L_\nu$  [ Hz erg / s / Hz ]")
        return fig, ax

    def corona_plot_sed_energy(self):
        """
        Plot corona SED in energy units.
        yaxis: EL_E [ keV keV / s / keV]
        xaxis: E [keV]
        """

        sed_energy = self.corona_sed_energy
        fig, ax = plt.subplots()
        ax.loglog(self.energy_range, sed_energy)
        ax.set_ylim(np.max(sed_energy)/1000, 2*np.max(sed_energy))
        ax.set_xlabel(r"Energy $E$ [ keV ]")
        ax.set_ylabel(r"$E \, L_E$  [ keV keV / s / keV]")
        return fig, ax

    def total_plot_sed_freq(self):
        """
        Plot total SED in frequency units.
        yaxis: nuLnu [ Hz erg / s / Hz]
        xaxis: nu [Hz]
        """

        sed_freq = self.total_sed_freq
        fig, ax = plt.subplots()
        ax.loglog(self.freq_range, sed_freq)
        ax.set_ylim(np.max(sed_freq)/1000, 2*np.max(sed_freq))
        ax.set_xlabel(r"Frequency $\nu$ [ Hz ]")
        ax.set_ylabel(r"$\nu \, L_\nu$  [ Hz erg / s / Hz ]")
        return fig, ax

    def total_plot_sed_energy(self):
        """
        Plot disc SED in energy units.
        yaxis: EL_E [ keV keV / s / keV]
        xaxis: E [keV]
        """

        sed_energy = self.total_sed_energy
        fig, ax = plt.subplots()
        ax.loglog(self.energy_range, sed_energy)
        ax.set_ylim(np.max(sed_energy)/1000, 2*np.max(sed_energy))
        ax.set_xlabel(r"Energy $E$ [ keV ]")
        ax.set_ylabel(r"$E \, L_E$  [ keV keV / s / keV]")
        return fig, ax









    





    






    





