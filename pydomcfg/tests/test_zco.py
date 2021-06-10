"""
Tests for zco
"""

import numpy as np
import xarray as xr

from pydomcfg.domzgr.zco import Zco
from pydomcfg.tests.bathymetry import Bathymetry


def test_zco_orca2():
    """
    The test consists in reproducing ORCA2 grid
    z3T/W and e3T/W as computed by NEMO v3.6
    (see pag 62 of v3.6 manual for the input parameters).
    This test validates stretched grids with analytical e3 and no
    double tanh.
    """

    # Results to replicate:
    # ORCA2 zco model levels depth and vertical
    # scale factors as computed by NEMO v3.6.
    # -------------------------------------------------------------
    #                      gdept_1d   gdepw_1d   e3t_1d     e3w_1d
    expected = np.array(
        [
            [4.9999, 0.0000, 10.0000, 9.9998],
            [15.0003, 10.0000, 10.0008, 10.0003],
            [25.0018, 20.0008, 10.0023, 10.0014],
            [35.0054, 30.0032, 10.0053, 10.0036],
            [45.0133, 40.0086, 10.0111, 10.0077],
            [55.0295, 50.0200, 10.0225, 10.0159],
            [65.0618, 60.0429, 10.0446, 10.0317],
            [75.1255, 70.0883, 10.0876, 10.0625],
            [85.2504, 80.1776, 10.1714, 10.1226],
            [95.4943, 90.3521, 10.3344, 10.2394],
            [105.9699, 100.6928, 10.6518, 10.4670],
            [116.8962, 111.3567, 11.2687, 10.9095],
            [128.6979, 122.6488, 12.4657, 11.7691],
            [142.1952, 135.1597, 14.7807, 13.4347],
            [158.9606, 150.0268, 19.2271, 16.6467],
            [181.9628, 169.4160, 27.6583, 22.7828],
            [216.6479, 197.3678, 43.2610, 34.2988],
            [272.4767, 241.1259, 70.8772, 55.2086],
            [364.3030, 312.7447, 116.1088, 90.9899],
            [511.5348, 429.7234, 181.5485, 146.4270],
            [732.2009, 611.8891, 261.0346, 220.3500],
            [1033.2173, 872.8738, 339.3937, 301.4219],
            [1405.6975, 1211.5880, 402.2568, 373.3136],
            [1830.8850, 1612.9757, 444.8663, 426.0031],
            [2289.7679, 2057.1314, 470.5516, 459.4697],
            [2768.2423, 2527.2169, 484.9545, 478.8342],
            [3257.4789, 3011.8994, 492.7049, 489.4391],
            [3752.4422, 3504.4551, 496.7832, 495.0725],
            [4250.4012, 4001.1590, 498.9040, 498.0165],
            [4749.9133, 4500.0215, 500.0000, 499.5419],
            [5250.2266, 5000.0000, 500.5646, 500.3288],
        ]
    )

    # Testing pydomcfg.domzgr.zco
    # -----------------------------------
    # ORCA2 input parameters
    ppdzmin = 10.0
    pphmax = 5000.0
    ppkth = 21.43336197938
    ppacr = 3
    ppsur = -4762.96143546300
    ppa0 = 255.58049070440
    ppa1 = 245.58132232490
    ldbletanh = False

    jpk = 31

    # Bathymetry dataset
    ds_bathy = Bathymetry(1000.0, 1200.0, 100, 200).flat(5000.0)

    # zco grid generator
    zco = Zco(ds_bathy, jpk)

    # zco mesh with analytical e3
    dsz_an = zco(
        ppdzmin, pphmax, ppkth, ppacr, ppsur, ppa0, ppa1, ldbletanh, ln_e3_dep=False
    )

    varname = ["z3T", "z3W", "e3T", "e3W"]
    i2 = 50
    j2 = 100
    # reference ocean.output values are
    # given with 4 digits precision
    eps = 1.0e-5
    for n, var in enumerate(varname):
        actual = dsz_an[var].values[:, j2, i2]
        np.testing.assert_allclose(expected[:, n], actual, rtol=eps, atol=0)


def test_zco_uniform():
    """
    The test consists in comparing z3T/W and e3T/W of
    a uniform grid with 51 levels computed with both
    anaytical and finite differences e3.
    """

    # Input parameters
    ppdzmin = 10.0
    pphmax = 5000.0
    ppkth = 0.0
    ppacr = 0.0

    jpk = 51

    # Bathymetry dataset
    ds_bathy = Bathymetry(1000.0, 1200.0, 100, 200).flat(5000.0)

    # zco grid generator
    zco = Zco(ds_bathy, jpk)

    # zco mesh with analytical e3
    dsz_an = zco(ppdzmin, pphmax, ppkth, ppacr, ln_e3_dep=False)

    # zco mesh with finite difference e3
    dsz_fd = zco(ppdzmin, pphmax, ppkth, ppacr, ln_e3_dep=True)

    # truncation errors
    eps = 1.0e-14
    xr.testing.assert_allclose(dsz_fd, dsz_an, rtol=eps, atol=0)
