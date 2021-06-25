"""
Tests for sco
"""

import numpy as np
import pytest
import xarray as xr

from pydomcfg.domzgr.sco import Sco
from pydomcfg.utils import _calc_rmax, _smooth_MB06

from .bathymetry import Bathymetry

ds_bathy = Bathymetry(1000, 1000, 200, 200).sea_mount(5000.0, 1)
ds_bathy["Bathymetry"] = ds_bathy["Bathymetry"].where(ds_bathy["Bathymetry"] > 550, 0.0)
sco = Sco(ds_bathy, jpk=51)

ocean = ds_bathy["Bathymetry"].where(ds_bathy["Bathymetry"] == 0, 1)

np.nanmax(_calc_rmax(ds_bathy["Bathymetry"]) * ocean)
ds_s = sco(min_dep=500.0, max_dep=3500.0, rmax=0.01)
np.nanmax(_calc_rmax(ds_s["hbatt"]) * ocean)

ds_s["hbatt"].isel({"y": 100}).plot()
ds_bathy["Bathymetry"].isel({"y": 100}).plot()
plt.gca().invert_yaxis()
plt.show()


# def test_zco_orca2():
#    """
#    The test consists in reproducing ORCA2 grid
#    z3T/W and e3T/W as computed by NEMO v3.6
#    (see pag 62 of v3.6 manual for the input parameters).
#    This test validates stretched grids with analytical e3 and no
#    double tanh.
#    """

#    # Bathymetry dataset
#    ds_bathy = Bathymetry(1.0e3, 1.2e3, 1, 1).flat(5.0e3)
#
#    # zco grid generator
#    zco = Zco(ds_bathy, jpk=31)

#    # zco mesh with analytical e3 using ORCA2 input parameters
#    # See pag 62 of v3.6 manual for the input parameters
#    dsz_an = zco(
#        ppdzmin=10.0,
#        pphmax=5000.0,
#        ppkth=21.43336197938,
#        ppacr=3,
#        ppsur=-4762.96143546300,
#        ppa0=255.58049070440,
#        ppa1=245.58132232490,
#        ldbletanh=False,
#        ln_e3_dep=False,
#    )

#    # reference ocean.output values are
#    # given with 4 digits precision
#    eps = 1.0e-5
#    for n, varname in enumerate(["z3T", "z3W", "e3T", "e3W"]):
#        expected = ORCA2_VGRID[:, n]
#        actual = dsz_an[varname].squeeze().values
#        np.testing.assert_allclose(expected, actual, rtol=eps, atol=0)
