import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
plt.style.context('seaborn-talk')
matplotlib.rcParams.update({'font.size': 15})
"""
Plotting Routines.
"""

def plot_disk_flux(sed, distance, color = 'b', ax = None, label = None):
    """
    Plot disk flux in energy units.
    yaxis: EL_E [ keV keV / s / keV]
    xaxis: E [keV]
    """
    flux = sed.disk_flux(distance)
    
    if(ax is None):
        fig, ax = plt.subplots()
        
    ax.loglog(sed.ENERGY_RANGE_KEV, flux, color = color, label=label, linewidth = 2)
    ax.set_ylim(np.max(flux) * 1e-2, 2*np.max(flux))
    ax.set_xlabel(r"Energy $E$ [ keV ]")
    ax.set_ylabel(r"$E \, F_E$  [ keV$^2$ (Photons / cm$^{2}$ / s / keV)]")
    return ax

def plot_corona_flux(sed, distance, color = 'b', ax = None, label = None):
    """
    Plot corona flux in energy units.
    yaxis: EL_E [ keV keV / s / keV]
    xaxis: E [keV]
    """

    flux = sed.corona_flux(distance)
    if(ax is None):
        fig, ax = plt.subplots()
    ax.loglog(sed.ENERGY_RANGE_KEV, flux, color = color, label=label, linewidth = 2)
    ax.set_ylim(max(flux) * 1e-2, 2*max(flux))
    ax.set_xlabel(r"Energy $E$ [ keV ]")
    ax.set_ylabel(r"$E \, F_E$  [ keV$^2$ (Photons / cm$^{2}$ / s / keV)]")
    return ax

def plot_warm_flux(sed, distance, color = 'b', ax = None, label = None):
    """
    Plot warm component flux in energy units.
    yaxis: EL_E [ keV keV / s / keV]
    xaxis: E [keV]
    """

    flux = sed.warm_flux(distance)
    
    if(ax is None):
        fig, ax = plt.subplots()
    ax.loglog(sed.ENERGY_RANGE_KEV, flux, color = color, label=label, linewidth = 2)
    ax.set_ylim(np.max(flux) * 1e-2, 2*np.max(flux))
    ax.set_xlabel(r"Energy $E$ [ keV ]")
    ax.set_ylabel(r"$E \, F_E$  [ keV$^2$ (Photons / cm$^{2}$ / s / keV)]")
    return ax


def plot_total_flux(sed, distance, ax = None):
    """
    Plot total flux in energy units.
    yaxis: EL_E [ keV keV / s / keV]
    xaxis: E [keV]
    """
    
    flux_disk = sed.disk_flux(distance)
    flux_warm = sed.warm_flux(distance)
    flux_corona = sed.corona_flux(distance)
    flux_total = flux_disk + flux_warm + flux_corona
    
    colors = [(114,36,108), (221,90,97), (249,139,86), (249,248,113)]#iter(cm.viridis(np.linspace(0,1,4)))
    colors_norm = []
    for color in colors:
        color_n = np.array(list(color)) / 255.
        colors_norm.append(color_n)
    colors = iter(colors_norm)
    
    if(ax is None):
        print("creating figure.")
        fig, ax = plt.subplots()
        
    linewidth = 3
    ax.loglog(sed.ENERGY_RANGE_KEV, flux_total, color = next(colors), linewidth=linewidth, label = 'Total')
    ax.loglog(sed.ENERGY_RANGE_KEV, flux_disk, color = next(colors), linewidth=linewidth, label = 'Disc component')
    ax.loglog(sed.ENERGY_RANGE_KEV, flux_warm, color = next(colors), linewidth=linewidth, label = 'Warm Component')
    ax.loglog(sed.ENERGY_RANGE_KEV, flux_corona, color = next(colors), linewidth=linewidth, label = 'Corona component')
    
    ax.set_ylim(np.max(flux_total)* 1e-2, 2*np.max(flux_total))
    ax.set_xlabel(r"Energy $E$ [ keV ]")
    ax.set_ylabel(r"$E \, F_E$  [ keV$^2$ (Photons / cm$^{2}$ / s / keV)]")
    #ax.legend()
    return ax

