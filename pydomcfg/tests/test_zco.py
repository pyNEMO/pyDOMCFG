"""
Tests for zco
"""

import numpy as np
import pytest
import xarray as xr

from pydomcfg.domzgr.zco import Zco

from .bathymetry import Bathymetry
from .data import ORCA2_VGRID


def test_zco_orca2():
    """
    The test consists in reproducing ORCA2 grid
    z3T/W and e3T/W as computed by NEMO v3.6
    (see pag 62 of v3.6 manual for the input parameters).
    This test validates stretched grids with analytical e3 and no
    double tanh.
    """

    # Bathymetry dataset
    ds_bathy = Bathymetry(1.0e3, 1.2e3, 1, 1).flat(5.0e3)

    # zco grid generator
    zco = Zco(ds_bathy, jpk=31)

    # zco mesh with analytical e3 using ORCA2 input parameters
    # See pag 62 of v3.6 manual for the input parameters
    dsz_an = zco(
        ppdzmin=10.0,
        pphmax=5000.0,
        ppkth=21.43336197938,
        ppacr=3,
        ppsur=-4762.96143546300,
        ppa0=255.58049070440,
        ppa1=245.58132232490,
        ldbletanh=False,
        ln_e3_dep=False,
    )

    # reference ocean.output values are
    # given with 4 digits precision
    eps = 1.0e-5
    for n, varname in enumerate(["z3T", "z3W", "e3T", "e3W"]):
        expected = ORCA2_VGRID[:, n]
        actual = dsz_an[varname].squeeze().values
        np.testing.assert_allclose(expected, actual, rtol=eps, atol=0)


def test_zco_uniform():
    """
    The test consists in comparing z3T/W and e3T/W of
    a uniform grid with 51 levels computed with both
    anaytical and finite differences e3.
    """

    # Input parameters
    kwargs = dict(
        ppdzmin=10,
        pphmax=5.0e3,
        ppkth=0,
        ppacr=0,
    )

    # Bathymetry dataset
    ds_bathy = Bathymetry(10.0e3, 1.2e3, 1, 1).flat(5.0e3)

    # zco grid generator
    zco = Zco(ds_bathy, jpk=51)

    # Compare zco mesh with analytical VS finite difference e3
    expected = zco(**kwargs, ln_e3_dep=True)
    actual = zco(**kwargs, ln_e3_dep=False)
    eps = 1.0e-14  # truncation errors
    xr.testing.assert_allclose(expected, actual, rtol=eps, atol=0)


def test_zco_x_y_invariant():
    """Make sure all vertical columns are identical"""

    # Generate 2x2 flat bathymetry dataset
    ds_bathy = Bathymetry(10.0e3, 1.2e3, 2, 2).flat(5.0e3)
    zco = Zco(ds_bathy, jpk=10)
    ds = zco(ppdzmin=10, pphmax=5.0e3)

    # Check z3 and e3
    for varname in ["z3T", "z3W", "e3T", "e3W"]:
        expected = ds[varname].isel(x=0, y=0)
        actual = ds[varname]
        assert (expected == actual).all()


def test_zco_errors():
    """Make sure we raise informative errors"""

    # Input parameters
    kwargs = dict(ppdzmin=10, pphmax=5.0e3, ppkth=1, ppacr=1)

    # Generate test data
    ds_bathy = Bathymetry(1.0e3, 1.2e3, 1, 1).flat(5.0e3)
    zco = Zco(ds_bathy, jpk=10)

    # Without ldbletanh flag, only allow all pps set or none of them
    with pytest.raises(
        ValueError, match="ppa2, ppkth2 and ppacr2 MUST be all None or all float"
    ):
        zco(**kwargs, ppa2=1, ppkth2=1, ppacr2=None)

    # When ldbletanh flag is True, all coefficients must be specified
    with pytest.raises(ValueError, match="ppa2, ppkth2 and ppacr2 MUST be all float"):
        zco(**kwargs, ldbletanh=True, ppa2=1, ppkth2=1, ppacr2=None)


def test_zco_warnings():
    """Make sure we warn when arguments are ignored"""

    # Initialize test class
    ds_bathy = Bathymetry(1.0e3, 1.2e3, 1, 1).flat(5.0e3)
    zco = Zco(ds_bathy, jpk=10)

    # Uniform: Ignore stretching
    kwargs = dict(ppdzmin=10, pphmax=5.0e3, ppkth=0, ppacr=0)
    expected = zco(**kwargs, ppsur=None, ppa0=999_999, ppa1=None)
    with pytest.warns(
        UserWarning, match="ppsur, ppa0 and ppa1 are ignored when ppacr == ppkth == 0"
    ):
        actual = zco(**kwargs, ppsur=2, ppa0=2, ppa1=2)
    xr.testing.assert_identical(expected, actual)

    # ldbletanh OFF: Ignore double tanh
    kwargs = dict(ppdzmin=10, pphmax=5.0e3, ldbletanh=False)
    expected = zco(**kwargs, ppa2=None, ppkth2=None, ppacr2=None)
    with pytest.warns(
        UserWarning, match="ppa2, ppkth2 and ppacr2 are ignored when ldbletanh is False"
    ):
        actual = zco(**kwargs, ppa2=2, ppkth2=2, ppacr2=2)
    xr.testing.assert_identical(expected, actual)

    # Uniform case: Ignore double tanh
    kwargs = dict(ppdzmin=10, pphmax=5.0e3, ppkth=0, ppacr=0)
    expected = zco(**kwargs, ppa2=None, ppkth2=None, ppacr2=None)
    with pytest.warns(
        UserWarning,
        match="ppa2, ppkth2 and ppacr2 are ignored when ppacr == ppkth == 0",
    ):
        actual = zco(**kwargs, ppa2=2, ppkth2=2, ppacr2=2)
    xr.testing.assert_identical(expected, actual)
