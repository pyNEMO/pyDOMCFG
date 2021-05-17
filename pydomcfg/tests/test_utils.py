"""
Tests for utils module
"""

import numpy as np
import xarray as xr

from pydomcfg.utils import generate_nemo_rectilinear_grid


def test_generate_nemo_rectilinear_grid():

    # x: irregular spacing
    glamf = [2, 10, 20]
    glamt = [0, 6, 15]  # In-between f, starting from 0
    e1t = [4, 8, 10]  # f spacing, left bound at -glamf[0]
    e1f = [6, 9, 10]  # t spacing, e1t[-1] == e1f[-1]

    # y: regular spacing
    e2t = e2f = np.ones(6)
    gphit = np.arange(6)
    gphif = 0.5 + np.arange(6)

    # Generate
    ds = generate_nemo_rectilinear_grid(e1t, e2t)

    # Test deltas
    exp_dict = {}
    exp_dict["e1t"], exp_dict["e2t"] = np.meshgrid(e1t, e2t)
    exp_dict["e1f"], exp_dict["e2f"] = np.meshgrid(e1f, e2f)
    exp_dict["e1u"], exp_dict["e2u"] = (exp_dict["e1f"], exp_dict["e2t"])
    exp_dict["e1v"], exp_dict["e2v"] = (exp_dict["e1t"], exp_dict["e2f"])
    for varname, expected in exp_dict.items():
        actual = ds[varname].values
        np.testing.assert_equal(expected, actual)

    # Test coords
    exp_dict = {}
    exp_dict["glamt"], exp_dict["gphit"] = np.meshgrid(glamt, gphit)
    exp_dict["glamf"], exp_dict["gphif"] = np.meshgrid(glamf, gphif)
    exp_dict["glamu"], exp_dict["gphiu"] = (exp_dict["glamf"], exp_dict["gphit"])
    exp_dict["glamv"], exp_dict["gphiv"] = (exp_dict["glamt"], exp_dict["gphif"])
    for varname, expected in exp_dict.items():
        actual = ds[varname].values
        np.testing.assert_equal(expected, actual)

    # nav_lon, nav_lat
    for expected, actual in zip(["glamt", "gphit"], ["nav_lon", "nav_lat"]):
        xr.testing.assert_equal(ds[expected], ds[actual])
