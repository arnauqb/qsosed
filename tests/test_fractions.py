import qsosed.sed as sed
import numpy as np
from numpy import testing

M_range = np.geomspace(1e6, 1e10, 3)
mdot_range = np.geomspace(0.1, 1, 3)


def test_fraction():
    """
    Tests whether integrating all the fractions of UV times the local luminosity returns the global UV luminosity.
    """
    for M in M_range:
        for mdot in mdot_range:
            print(M, mdot)
            bh = sed.SED(M=M, mdot=mdot, reprocessing=False)
            _, total_uv_flux, total_flux, _ = bh.compute_uv_fractions(
                distance=1e26, return_all=True, include_corona=True
            )
            uvf = total_uv_flux / total_flux
            testing.assert_approx_equal(uvf, bh.uv_fraction, significant=2)
