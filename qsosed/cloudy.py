import os, sys, shutil
import numpy as np
from scipy import integrate
from memoized_property import memoized_property as property

import pyCloudy as pc
import pandas as pd

from qsosed.sed import SED


#pc.config.cloudy_exe = "/cosma/local/cloudy/c17.01/bin/cloudy.exe"
pc.config.cloudy_exe = "/home/arnau/cloudy/source/cloudy.exe"
pc.log_.level = 2



class Model(SED):
    """
    Model class to handle a Cloudy model. It inherits from the SED class from QSOSED, so it has access to all its functions.
    """
    def __init__(self,
                 M=1e8,
                 mdot=0.5,
                 spin=0,
                 spin_sign=1,
                 distance=10,
                 thickness=10,
                 density=1e10,
                 temperature=1e4,
                 ionization_parameter=1e5,
                 working_path="cloudy_model",
                 ):
        SED.__init__(self, M=M, mdot=mdot, number_bins_fractions=500)#, spin=spin, spin_sign=spin_sign, mu=1, reprocessing=True)
        self.distance = distance
        self.distance_cm = self.distance * self.Rg
        self.thickness = thickness
        self.density = density
        self.temperature = temperature
        self.ionization_parameter = ionization_parameter
        self.working_path = working_path
        try:
            os.mkdir(self.working_path)
        except:
            shutil.rmtree(self.working_path)
            os.mkdir(self.working_path)

    def _write_table_sed_file(self, flux):
        """
        Aux function to write custom sed file for Cloudy.
        """
        file_path = os.path.join(self.working_path, "spectrum.sed")
        with open(file_path, "w") as f:
            first_line = f"{self.ENERGY_RANGE_KEV[0]}\t{flux[0]}\tnuFnu units kev\n"
            f.writelines(first_line)
            for i in range(1, len(flux)):
                flux_value = flux[i]
                energy_value = self.ENERGY_RANGE_KEV[i]
                f.writelines(f"{energy_value}\t{flux_value}\n")


    def run_cloudy_model(self, flux=None):
        """
        Initialized and runs a cloudy model, with gas density and temperature for the corresponding spectrum.
        Spectrum is defined in the Qsosed energy range.
        """
        if flux is None:
            flux = self.total_flux(self.distance_cm) 
        self._write_table_sed_file(flux)
        intensity = integrate.trapz(x = self.ENERGY_RANGE_ERG, y = flux)# / 1000

        cloudy_input_fname = "model"
        cloudy_input_path = os.path.join(self.working_path, cloudy_input_fname)
        print(cloudy_input_path)
        self.cloudy_model = pc.CloudyInput(cloudy_input_path)
        self.cloudy_model.set_iterate()
        cloudy_input_parameters = (f"table sed \"spectrum.sed\"",
                                   #f"intensity total {np.log10(intensity)}",
                                   f"xi {np.log10(self.ionization_parameter)}",
                                   f"hden {np.log10(self.density)}",
                                   #f"constant temperature {temperature}",
                                   f"iterate until convergence",
                                   f"stop thickness {np.log10(self.thickness * self.Rg)}",
                                   f"save continuum last separate units kev \".cont_kev\"",
                                   f"save total opacity last separate units kev \".opa\"",
        )
        self.cloudy_model.set_other(cloudy_input_parameters)
        self.cloudy_model.print_input()
        self.cloudy_model.run_cloudy()
    
    @property
    def continuum_incident(self):
        # read incoming continuum #
        fname = self.cloudy_model.model_name + ".cont_kev"
        continuum = pd.read_csv(fname, sep="\s+", skiprows=1, usecols=(0,1), names=("energy", "continuum_incident"))
        return continuum

    @property
    def continuum_transmitted(self):
        # read incoming continuum #
        fname = self.cloudy_model.model_name + ".cont_kev"
        continuum = pd.read_csv(fname, sep="\s+", skiprows=1, usecols=(0,2), names=("energy", "continuum_transmitted"))
        return continuum

    @property
    def continuum_incident_uv(self):
        continuum = self.continuum_incident
        continuum_uv = continuum[(continuum.energy > self.ENERGY_UV_LOW_CUT_KEV) & (continuum.energy < self.ENERGY_UV_HIGH_CUT_KEV)]
        return continuum_uv

    @property
    def continuum_incident_xray(self):
        continuum = self.continuum_incident
        continuum_xray = continuum[(continuum.energy > self.ENERGY_XRAY_LOW_CUT_KEV) & (continuum.energy < self.ENERGY_RANGE_KEV[-1])]
        return continuum_xray

    @property
    def opacities(self):
        fname = self.cloudy_model.model_name + ".opa"
        opacities = pd.read_csv(fname, sep="\s+", skiprows=1, usecols=(0,1), names=("energy", "total_opacity"))
        return opacities

    @property
    def opacities_uv(self):
        opacities = self.opacities
        opacities_uv = opacities[(opacities.energy > self.ENERGY_UV_LOW_CUT_KEV) & (opacities.energy < self.ENERGY_UV_HIGH_CUT_KEV)]
        return opacities_uv

    @property
    def opacities_xray(self):
        opacities = self.opacities  
        opacities_xray = opacities[(opacities.energy > self.ENERGY_XRAY_LOW_CUT_KEV) & (opacities.energy < self.ENERGY_RANGE_KEV[-1])]
        return opacities_xray
    
    @property
    def opacity_uv_average(self):
        average_uv_opacity = np.average(self.opacities_uv["total_opacity"], weights=self.continuum_incident_uv["continuum_incident"])
        return average_uv_opacity

    @property
    def opacity_xray_average(self):
        average_xray_opacity = np.average(self.opacities_xray["total_opacity"], weights=self.continuum_incident_xray["continuum_incident"])
        return average_xray_opacity

    @property
    def optical_depth_uv_average(self):
        self.tau_uv = self.opacity_uv_average * (self.thickness * self.Rg)
        return self.tau_uv

    @property
    def optical_depth_xray_average(self):
        self.tau_uv = self.opacity_xray_average * (self.thickness * self.Rg)
        return self.tau_uv


    #def compute_opacities(self):

if __name__=="__main__":
    test = Model(1e8, 0.5)
    test.run_cloudy_model()
    test.compute_opacity_and_optical_depth()

        