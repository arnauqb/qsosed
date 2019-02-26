"""
This module computes the AGN Spectral Energy Density in the UV/X-Ray energy range, following Kubota & Done (2018). This is equivalent to AGNSED in XSPEC.
Written by Arnau Quera-Bofarull in Durham, UK.
"""
import numpy as np
import pyagn.constants as const
import matplotlib.pyplot as plt
import matplotlib
plt.style.context('seaborn-talk')
matplotlib.rcParams.update({'font.size': 16})
from matplotlib import cm
from scipy import integrate, optimize
from astropy import units as u
from memoized_property import memoized_property as property
from pyagn.xspec_routines import donthcomp


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
    energy_range = np.geomspace(energy_min, energy_max, 100) # keV
    energy_range_erg = convert_units(energy_range * u.keV, u.erg) #erg

    def __init__(self, M = 1e8, mdot = 0.5, astar = 0, astar_sign = 1, reprocessing = False, hard_xray_fraction = 0.02, corona_electron_energy = 100, warm_electron_energy = 0.2, warm_photon_index = 2.5, reflection_albedo = 0.3):

        # read parameters #
        self.M = M # black hole mass in solar masses
        self.mdot = mdot # mdot = Mdot / Mdot_Eddington
        self.astar = astar # dimensionless black hole spin
        self.astar_sign = astar_sign # +1 for prograde rotation, -1 for retrograde

        # useful quantities #
        self.Rg = const.G * M * const.Ms / const.c ** 2 # gravitational radius
        
        # model parameters
        self.hard_xray_fraction = hard_xray_fraction # fraction of energy in Eddington units in the corona.
        self.corona_electron_energy = corona_electron_energy # temperature of the corona's electrons in keV.
        self.warm_electron_energy = warm_electron_energy # temperature of the soft region's electrons in keV.
        self.warm_photon_index = warm_photon_index # powerlaw index of the warm component. 
        self.reflection_albedo = reflection_albedo # reflection albedo for the reprocessed flux.
        self.electron_rest_mass = 511. #kev
        self.corona_height = min(100., self.corona_radius)
        
        # set reprocessing to false to compute corona luminosity
        #self.reprocessing = False
        #self.corona_luminosity_norepr = self.corona_luminosity
        self.reprocessing = reprocessing # set reprocessing to the desired value
        
        try:
            assert(reprocessing in [False,True])#
        except:
            print("Reprocessing has to be either False (no reprocessing) or True (include reprocessing).")

    @property
    def isco(self):
        """
        Computes the Innermost Stable Circular Orbit. Depends only on astar.
        """
        
        z1 = 1 + (1 - self.astar**2)**(1 / 3) * ((1 + self.astar)**(1 / 3) + (1 - self.astar)**(1 / 3))
        z2 = np.sqrt(3 * self.astar**2 + z1**2)
        rms = 3 + z2 - self.astar_sign * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))
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

    def _nt_rel_factors(self, r):
        """
        Relatistic A,B,C factors of the Novikov-Thorne model.
        
        Parameters
            Black Hole Mass in solar Masses
        -----------
        r : float
            disk radial distance.
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
        rel_factor = self._nt_rel_factors(r)
        aux = self.mdot / (self.M * self.efficiency * r**3) 
        t4 = nt_constant * rel_factor * aux
        return t4
    
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
        #print(aux)
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
        temperature = self.disk_temperature4(r) ** (1./4.)
        planck_spectrum_exp = np.exp( energy / ( const.k_B *  temperature ))
        planck_spectrum = bb_constant * energy**3 * 1./ ( planck_spectrum_exp - 1)
        return planck_spectrum

    def disk_radiance(self, r):
        """
        disk radiance in units of erg / cm^2 / s / sr, assuming black-body radiation.

        Parameters
        ----------
        r : float
            disk radius in Rg.
        """

        radiance = const.sigma_sb * self.disk_temperature4(r)
        return radiance

    def disk_spectral_luminosity(self, energy):
        """
        disk spectral luminosity in units of 1 / s.

        Parameters
        ----------
        energy : float
            Energy in erg.
        """
        radial_integral = 2 * np.pi**2 * self.Rg**2 * integrate.quad( lambda r: r * self.disk_spectral_radiance(energy,r), self.corona_radius * 2., self.gravity_radius)[0]
        spectral_lumin = 2 * radial_integral # 2 sides of the disk
        return spectral_lumin
    
    @property
    def disk_luminosity(self):
        """
        disk Luminosityin units of erg / s.
        """

        constant =  const.sigma_sb * 4 * np.pi * self.Rg**2
        lumin = constant * integrate.quad(lambda r: r*self.disk_temperature4(r), 2.*self.corona_radius, self.gravity_radius)[0]
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
        energy_range_erg = convert_units(self.energy_range * u.keV, u.erg)
        lumin = []
        for energy_erg in energy_range_erg:
            lumin.append(self.disk_spectral_luminosity(energy_erg))
        sed = np.array(lumin) * self.energy_range
        return sed
    
    def disk_flux(self, distance):
        """
        Flux of the disk component in units of keV^2 ( Photons / cm^2 / s / keV).
        
        Parameters
        ----------
        distance: float
                  Distance to the source in cm.
        """
        return self.disk_sed / (4*np.pi*distance**2)
        
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
    def corona_radius(self):
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
    def corona_luminosity(self):
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
        reproc = self.reprocessing
        self.reprocessing = False
        gamma_cor = 7./3. * ( self.corona_dissipated_luminosity / self.corona_seed_luminosity )**(-0.1)
        self.reprocessing = reproc
        return gamma_cor

    def corona_flux(self, distance):
        """
        Corona flux computed using donthcomp from Xspec.
        
        Parameters
        ----------
        
        distance : float
                   distance to source.
        """

        gamma = self.corona_photon_index
        kt_e = self.corona_electron_energy
        t_corona = self.disk_temperature4(self.corona_radius)**(1./4.) * const.k_B
        t_corona_kev = convert_units(t_corona * u.erg, u.keV)
        ywarm = (4./9. * self.warm_photon_index) ** (-4.5)
        params = [gamma, kt_e, t_corona_kev * np.exp(ywarm), 0, 0]
        photon_number_flux = donthcomp(ear = self.energy_range, param = params) # units of Photons / cm^2 / s / keV
        
        # We integrate the flux only where is non-zero.
        mask = photon_number_flux > 0
        flux_array = np.zeros(len(self.energy_range))
        flux = integrate.simps(x=self.energy_range_erg, y=photon_number_flux) # units of Photons / cm^2 / s
        
        # We renormalize to the correct distance.
        ratio = (self.corona_luminosity / (4 * np.pi* distance**2)) / flux
        flux = ratio * photon_number_flux[mask] * self.energy_range[mask] # units of keV / cm^2 / s
        flux_array[mask] = flux
        return flux_array

    """
    Warm region section.
    """ 

    def warm_flux_r(self, radius):
        """
        Photon flux per energy bin for a disk annulus at radius radius in the warm Compton region.
        
        Parameters
        ----------
        
        radius : float
                 disk radius.
        """
        
        gamma = self.warm_photon_index
        kt_e = self.warm_electron_energy
        t_warm = self.disk_temperature4(radius)**(1./4.) * const.k_B
        t_warm_kev = convert_units(t_warm * u.erg, u.keV)
        params = [gamma, kt_e, t_warm_kev, 0, 0]
        photon_number_flux = donthcomp(ear = self.energy_range, param = params)
        return photon_number_flux # units of Photons / cm^2 / s / keV

    #@property
    def warm_flux(self, distance):
        """
        warm SED in energy units, [ KeV KeV / s / KeV].
        """

        r_range = np.linspace(self.corona_radius, 2. * self.corona_radius,10) # the soft-compton region extends form Rcor to 2Rcor.
        grid = np.zeros((len(r_range), len(self.energy_range)))
        for i,r in enumerate(r_range):
            # we first integrate along the relevant energies
            flux_r_E = self.warm_flux_r(r)
            mask = flux_r_E>0
            flux_r = integrate.trapz(x= self.energy_range_erg[mask], y=flux_r_E[mask])
            # we then normalize the flux using the local disc flux.
            disk_lumin = 4 * np.pi * (self.Rg)**2. * r * self.disk_radiance(r)
            disk_flux = disk_lumin / (4. * np.pi * distance**2)
            warm_lumin = 4 * np.pi * flux_r * r * (self.Rg**2)
            ratio = disk_flux / flux_r
            flux_r = ratio * flux_r_E[mask] * self.energy_range[mask]
            grid[i,mask] = flux_r # units of keV / cm^2 / s / per annulus.
            
        # we now integrate over all radii.
        flux_array = []
        for row in np.transpose(grid):
            flux_array.append(integrate.simps(x = r_range, y = row))
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
        corona_flux = self.warm_flux(distance)
        total_flux = disk_flux + warm_flux + corona_flux
        return total_flux


    """
    Plotting Routines.
    """
    
    def plot_disk_flux(self, distance, color = 'b', ax = None, label = None):
        """
        Plot disk flux in energy units.
        yaxis: EL_E [ keV keV / s / keV]
        xaxis: E [keV]
        """

        flux = self.disk_flux(distance)
        
        if(ax is None):
            fig, ax = plt.subplots()
            
        ax.loglog(self.energy_range, flux, color = color, label=label, linewidth = 2)
        ax.set_ylim(np.max(flux) * 1e-2, 2*np.max(flux))
        ax.set_xlabel(r"Energy $E$ [ keV ]")
        ax.set_ylabel(r"$E \, F_E$  [ keV$^2$ (Photons / cm$^{2}$ / s / keV]")
        return ax

    def plot_corona_flux(self, distance, color = 'b', ax = None, label = None):
        """
        Plot corona flux in energy units.
        yaxis: EL_E [ keV keV / s / keV]
        xaxis: E [keV]
        """

        flux = self.corona_flux(distance)
        if(ax is None):
            fig, ax = plt.subplots()
        ax.loglog(self.energy_range, flux, color = color, label=label, linewidth = 2)
        ax.set_ylim(max(flux) * 1e-2, 2*max(flux))
        ax.set_xlabel(r"Energy $E$ [ keV ]")
        ax.set_ylabel(r"$E \, F_E$  [ keV$^2$ (Photons / cm$^{2}$ / s / keV]")
        return ax

    def plot_warm_flux(self, distance, color = 'b', ax = None, label = None):
        """
        Plot warm component flux in energy units.
        yaxis: EL_E [ keV keV / s / keV]
        xaxis: E [keV]
        """

        flux = self.warm_flux(distance)
        
        if(ax is None):
            fig, ax = plt.subplots()
        ax.loglog(self.energy_range, flux, color = color, label=label, linewidth = 2)
        ax.set_ylim(np.max(flux) * 1e-2, 2*np.max(flux))
        ax.set_xlabel(r"Energy $E$ [ keV ]")
        ax.set_ylabel(r"$E \, F_E$  [ keV$^2$ (Photons / cm$^{2}$ / s / keV]")
        return ax


    def plot_total_flux(self, distance, ax = None):
        """
        Plot total flux in energy units.
        yaxis: EL_E [ keV keV / s / keV]
        xaxis: E [keV]
        """
        colors =iter(cm.viridis(np.linspace(0,1,4)))
        if(ax is None):
            print("creating figure.")
            fig, ax = plt.subplots()
        flux = self.corona_flux(distance) + self.disk_flux(distance) + self.warm_flux(distance)
        ax.loglog(self.energy_range, flux, color = next(colors), linewidth=4, label = 'Total')
        ax = self.plot_disk_flux(distance, color = next(colors), ax=ax, label = 'Disk component')
        ax = self.plot_corona_flux(distance, color = next(colors), ax=ax, label = 'Corona component')
        ax = self.plot_warm_flux(distance, color = next(colors), ax=ax, label = 'Warm component')
        
        ax.set_ylim(np.max(flux)* 1e-2, 2*np.max(flux))
        ax.set_xlabel(r"Energy $E$ [ keV ]")
        ax.set_ylabel(r"$E \, F_E$  [ keV$^2$ (Photons / cm$^{2}$ / s / keV]")
        ax.legend()
        return ax

