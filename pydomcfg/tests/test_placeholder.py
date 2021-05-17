"""
TODO: This is just a placeholder to showcase the use of pre-build datasets
"""

import pytest

from .datasets import ds_flat, ds_island

datasets = [ds_flat, ds_island]


@pytest.mark.parametrize("ds", datasets)
def test_variables_added(ds):
    assert {"Bathymetry", "mask"} <= set(ds.variables)
