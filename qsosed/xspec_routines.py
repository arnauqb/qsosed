"""
This part of the code was taken from https://github.com/ADThomas-astro/oxaf/blob/master/oxaf.py .
Credit to A.D. Thomas.
Code was adapted from Xspec for https://arxiv.org/pdf/1611.05165.pdf .
"""

import numpy as np 

def donthcomp(ear, param):
    """
    This function was adapted by ADT from the subroutine donthcomp in
    donthcomp.f, distributed with XSpec.
    Nthcomp documentation:
    https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelNthcomp.html
    Refs:
    Zdziarski, Johnson & Magdziarz 1996, MNRAS, 283, 193,
    as extended by Zycki, Done & Smith 1999, MNRAS 309, 561
    Note that the subroutine has been modified so that parameter 4
    is ignored, and the seed spectrum is always a blackbody.
    ear: Energy vector, listing "Energy At Right" of bins (keV)
    param: list of parameters; see the 5 parameters listed below.
    The original fortran documentation for this subroutine is included below:
    Driver for the Comptonization code solving Kompaneets equation
    seed photons  -  (disk) blackbody
    reflection + Fe line with smearing
    
    Model parameters:
    1: photon spectral index
    2: plasma temperature in keV
    3: (disk)blackbody temperature in keV
    4: type of seed spectrum (0 - blackbody, 1 - diskbb)
    5: redshift
    """
    param = np.array(param)
    param = np.insert(param,0,0)
    ne = ear.size  # Length of energy bin vector
    # Note that this model does not calculate errors.
    #c     xth is the energy array (units m_e c^2)
    #c     spnth is the nonthermal spectrum alone (E F_E)
    #c     sptot is the total spectrum array (E F_E), = spref if no reflection
    zfactor = 1.0 + param[5]
    #c  calculate internal source spectrum
    #                           blackbody temp,   plasma temp,      Gamma
    xth, nth, spt = _thcompton(param[3] / 511.0, param[2] / 511.0, param[1])
    # The temperatures are normalized by 511 keV, the electron rest energy
    # Calculate normfac:
    xninv = 511.0 / zfactor
    ih = 1
    xx = 1.0 / xninv
    while (ih < nth and xx > xth[ih]):
        ih = ih + 1
    il = ih - 1
    spp = spt[il] + (spt[ih] - spt[il]) * (xx - xth[il]) / (xth[ih] - xth[il])
    normfac = 1.0 / spp

    #c     zero arrays
    photar = np.zeros(ne)
    prim   = np.zeros(ne)
    #c     put primary into final array only if scale >= 0.
    j = 0
    for i in range(0, ne):
        while (j <= nth and 511.0 * xth[j] < ear[i] * zfactor):
            j = j + 1
        if (j <= nth):
            if (j > 0):
                jl = j - 1
                prim[i] = spt[jl] + ((ear[i] / 511.0 * zfactor - xth[jl]) * 
                                     (spt[jl + 1] - spt[jl]) / 
                                     (xth[jl + 1] - xth[jl])                 )
            else:
                prim[i] = spt[0]
    for i in range(1, ne):
        photar[i] = (0.5 * (prim[i] / ear[i]**2 + prim[i - 1] / ear[i - 1]**2) 
                         * (ear[i] - ear[i - 1]) * normfac                    )

    return photar

def _thcompton(tempbb, theta, gamma):
    """
    This function was adapted by ADT from the subroutine thcompton in
    donthcomp.f, distributed with XSpec.
    Nthcomp documentation:
    https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelNthcomp.html
    Refs:
    Zdziarski, Johnson & Magdziarz 1996, MNRAS, 283, 193,
    as extended by Zycki, Done & Smith 1999, MNRAS 309, 561
    The original fortran documentation for this subroutine is included below:
    Thermal Comptonization; solves Kompaneets eq. with some
    relativistic corrections. See Lightman \ Zdziarski (1987), ApJ
    The seed spectrum is a blackbody.
    version: January 96
    #c  input parameters:
    #real * 8 tempbb,theta,gamma
    """
    #c use internally Thomson optical depth
    tautom = np.sqrt(2.250 + 3.0 / (theta * ((gamma + .50)**2 - 2.250))) - 1.50

    # Initialise arrays
    dphdot = np.zeros(900); rel = np.zeros(900); c2 = np.zeros(900)
    sptot  = np.zeros(900); bet = np.zeros(900); x  = np.zeros(900)

    #c JMAX  -  # OF PHOTON ENERGIES
    #c delta is the 10 - log interval of the photon array.
    delta = 0.02
    deltal = delta * np.log(10.0)
    xmin = 1e-4 * tempbb
    xmax = 40.0 * theta
    jmax = min(899, int(np.log10(xmax / xmin) / delta) + 1)

    #c X  -  ARRAY FOR PHOTON ENERGIES
    # Energy array is normalized by 511 keV, the rest energy of an electron
    x[:(jmax + 1)] = xmin * 10.0**(np.arange(jmax + 1) * delta)

    #c compute c2(x), and rel(x) arrays
    #c c2(x) is the relativistic correction to Kompaneets equation
    #c rel(x) is the Klein - Nishina cross section divided by the
    #c Thomson crossection
    for j in range(0, jmax):
        w = x[j]
    #c c2 is the Cooper's coefficient calculated at w1
    #c w1 is x(j + 1 / 2) (x(i) defined up to jmax + 1)
        w1 = np.sqrt(x[j] * x[j + 1])
        c2[j] = (w1**4 / (1.0 + 4.60 * w1 + 1.1 * w1 * w1))
        if (w <= 0.05):
            #c use asymptotic limit for rel(x) for x less than 0.05
            rel[j] = (1.0 - 2.0 * w + 26.0 * w * w * 0.2)
        else:
            z1 = (1.0 + w) / w**3
            z2 = 1.0 + 2.0 * w
            z3 = np.log(z2)
            z4 = 2.0 * w * (1.0 + w) / z2
            z5 = z3 / 2.0 / w
            z6 = (1.0 + 3.0 * w) / z2 / z2
            rel[j] = (0.75 * (z1 * (z4 - z3) + z5 - z6))

    #c the thermal emission spectrum
    jmaxth = min(900, int(np.log10(50 * tempbb / xmin) / delta))
    if (jmaxth > jmax):
       jmaxth = jmax
    planck = 15.0 / (np.pi * tempbb)**4
    dphdot[:jmaxth] = planck * x[:jmaxth]**2 / (np.exp(x[:jmaxth] / tempbb)-1)

    #c compute beta array, the probability of escape per Thomson time.
    #c bet evaluated for spherical geometry and nearly uniform sources.
    #c Between x = 0.1 and 1.0, a function flz modifies beta to allow
    #c the increasingly large energy change per scattering to gradually
    #c eliminate spatial diffusion
    jnr  = int(np.log10(0.10 / xmin) / delta + 1)
    jnr  = min(jnr, jmax - 1)
    jrel = int(np.log10(1 / xmin) / delta + 1)
    jrel = min(jrel, jmax)
    xnr  = x[jnr - 1]
    xr   = x[jrel - 1]
    for j in range(0, jnr - 1):
        taukn = tautom * rel[j]
        bet[j] = 1.0 / tautom / (1.0 + taukn / 3.0)
    for j in range(jnr - 1, jrel):
        taukn = tautom * rel[j]
        arg = (x[j] - xnr) / (xr - xnr)
        flz = 1 - arg
        bet[j] = 1.0 / tautom / (1.0 + taukn / 3.0 * flz)
    for j in range(jrel, jmax):
        bet[j] = 1.0 / tautom

    dphesc = _thermlc(tautom, theta, deltal, x, jmax, dphdot, bet, c2)

    #c     the spectrum in E F_E
    for j in range(0, jmax - 1):
        sptot[j] = dphesc[j] * x[j]**2

    return x, jmax, sptot

def _thermlc(tautom, theta, deltal, x, jmax, dphdot, bet, c2):
    """
    This function was adapted by ADT from the subroutine thermlc in 
    donthcomp.f, distributed with XSpec.
    Nthcomp documentation:
    https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelNthcomp.html
    Refs:
    Zdziarski, Johnson & Magdziarz 1996, MNRAS, 283, 193,
    as extended by Zycki, Done & Smith 1999, MNRAS 309, 561
    The original fortran documentation for this subroutine is included below:
    This program computes the effects of Comptonization by
    nonrelativistic thermal electrons in a sphere including escape, and
    relativistic corrections up to photon energies of 1 MeV.
    the dimensionless photon energy is x = hv / (m * c * c)
    The input parameters and functions are:
    dphdot(x), the photon production rate
    tautom, the Thomson scattering depth
    theta, the temperature in units of m*c*c
    c2(x), and bet(x), the coefficients in the K - equation and the
      probability of photon escape per Thomson time, respectively,
      including Klein - Nishina corrections
    The output parameters and functions are:
    dphesc(x), the escaping photon density
    """
    dphesc = np.zeros(900)  # Initialise the output
    a = np.zeros(900); b   = np.zeros(900); c = np.zeros(900)
    d = np.zeros(900); alp = np.zeros(900); u = np.zeros(900)
    g = np.zeros(900); gam = np.zeros(900)

    #c u(x) is the dimensionless photon occupation number
    c20 = tautom / deltal

    #c determine u
    #c define coefficients going into equation
    #c a(j) * u(j + 1) + b(j) * u(j) + c(j) * u(j - 1) = d(j)
    for j in range(1, jmax - 1):
        w1 = np.sqrt( x[j] * x[j + 1] )
        w2 = np.sqrt( x[j - 1] * x[j] )
        #c  w1 is x(j + 1 / 2)
        #c  w2 is x(j - 1 / 2)
        a[j] =  -c20 * c2[j] * (theta / deltal / w1 + 0.5)
        t1 =  -c20 * c2[j] * (0.5 - theta / deltal / w1)
        t2 = c20 * c2[j - 1] * (theta / deltal / w2 + 0.5)
        t3 = x[j]**3 * (tautom * bet[j])
        b[j] = t1 + t2 + t3
        c[j] = c20 * c2[j - 1] * (0.5 - theta / deltal / w2)
        d[j] = x[j] * dphdot[j]

    #c define constants going into boundary terms
    #c u(1) = aa * u(2) (zero flux at lowest energy)
    #c u(jx2) given from region 2 above
    x32 = np.sqrt(x[0] * x[1])
    aa = (theta / deltal / x32 + 0.5) / (theta / deltal / x32 - 0.5)

    #c zero flux at the highest energy
    u[jmax - 1] = 0.0

    #c invert tridiagonal matrix
    alp[1] = b[1] + c[1] * aa
    gam[1] = a[1] / alp[1]
    for j in range(2, jmax - 1):
        alp[j] = b[j] - c[j] * gam[j - 1]
        gam[j] = a[j] / alp[j]
    g[1] = d[1] / alp[1]
    for j in range(2, jmax - 2):
        g[j] = (d[j] - c[j] * g[j - 1]) / alp[j]
    g[jmax - 2] = (d[jmax - 2] - a[jmax - 2] * u[jmax - 1] 
                               - c[jmax - 2] * g[jmax - 3]) / alp[jmax - 2]
    u[jmax - 2] = g[jmax - 2]
    for j in range(2, jmax + 1):
        jj = jmax - j
        u[jj] = g[jj] - gam[jj] * u[jj + 1]
    u[0] = aa * u[1]
    #c compute new value of dph(x) and new value of dphesc(x)
    dphesc[:jmax] = x[:jmax] * x[:jmax] * u[:jmax] * bet[:jmax] * tautom

    return dphesc
