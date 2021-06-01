"""
TODO: This is just a placeholder to showcase the use of pre-build datasets
"""

import pytest

from .bathymetry import Bathymetry

square = Bathymetry(range(10), range(10))
shallow = square.flat(10)
deep = square.flat(1.0e3)

datasets = [shallow, deep]


@pytest.mark.parametrize("ds", datasets)
def test_variables_added(ds):
    assert {"Bathymetry", "mask"} <= set(ds.variables)
