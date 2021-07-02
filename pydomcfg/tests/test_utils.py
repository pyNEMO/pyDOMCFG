"""
Tests for utils module
"""

import numpy as np
import pytest
import xarray as xr

from pydomcfg.utils import _check_namelist_entries, generate_cartesian_grid


def test_generate_cartesian_grid():

    # x: irregular spacing
    glamf = [2, 10, 20]
    glamt = [0, 6, 15]  # In-between f
    e1t = [4, 8, 10]  # f spacing
    e1f = [6, 9, 10]  # t spacing, e1t[-1] == e1f[-1]

    # y: regular spacing
    gphif = np.arange(6) + 10.5
    gphit = np.arange(6) + 10
    e2t = e2f = np.ones(6)

    # Generate
    ds = generate_cartesian_grid(ppe1_m=e1t, ppe2_m=1, jpjglo=6, ppgphi0=10)

    # Left f bounds has been removed
    assert ds.sizes == {"x": 3, "y": 6}

    # Test deltas
    exp_dict = {}
    exp_dict["e1t"], exp_dict["e2t"] = np.meshgrid(e1t, e2t)
    exp_dict["e1f"], exp_dict["e2f"] = np.meshgrid(e1f, e2f)
    exp_dict["e1u"], exp_dict["e2u"] = np.meshgrid(e1f, e2t)
    exp_dict["e1v"], exp_dict["e2v"] = np.meshgrid(e1t, e2f)
    for varname, expected in exp_dict.items():
        actual = ds[varname].values
        np.testing.assert_equal(expected, actual)

    # Test coords
    exp_dict = {}
    exp_dict["glamt"], exp_dict["gphit"] = np.meshgrid(glamt, gphit)
    exp_dict["glamf"], exp_dict["gphif"] = np.meshgrid(glamf, gphif)
    exp_dict["glamu"], exp_dict["gphiu"] = np.meshgrid(glamf, gphit)
    exp_dict["glamv"], exp_dict["gphiv"] = np.meshgrid(glamt, gphif)
    for varname, expected in exp_dict.items():
        actual = ds[varname].values
        np.testing.assert_equal(expected, actual)

    # nav_lon, nav_lat
    for expected, actual in zip(["glamt", "gphit"], ["nav_lon", "nav_lat"]):
        xr.testing.assert_equal(ds[expected], ds[actual])


def test_check_namelist_entries():

    error_match = (
        "Value does not match expected types for 'pptest'."
        "\nValue: 'wrong'\nType: 'str'"
        "\nExpected types: \\('NoneType', 'int', 'float'\\)"
    )
    with pytest.raises(TypeError, match=error_match):
        _check_namelist_entries({"pptest": "wrong"})

    error_match = (
        "Mismatch in number of values provided for 'sn_test'."
        "\nValues: \\[None\\]\nNumber of values: 1\nExpected number of values: 9"
    )
    with pytest.raises(ValueError, match=error_match):
        _check_namelist_entries({"sn_test": [None]})

    error_match = (
        "Values do not match expected types for 'sn_test'."
        "\nValues: \\[None, None, None, None, None, None, None, None, None\\]"
        "\nTypes: \\['NoneType', 'NoneType', 'NoneType', 'NoneType', 'NoneType',"
        " 'NoneType', 'NoneType', 'NoneType', 'NoneType'\\]"
        "\nExpected types:"
        " \\['str', 'int', 'str', 'bool', 'bool', 'str', 'str', 'str', 'str'\\]"
    )
    with pytest.raises(TypeError, match=error_match):
        _check_namelist_entries({"sn_test": [None] * 9})
