"""
Module to generate datasets for testing
"""

import xarray as xr
from xarray import Dataset

from pydomcfg.utils import generate_cartesian_grid


class Bathymetry:
    """
    Class to generate idealized test bathymetry datasets.
    """

    def __init__(self, x_f, y_f):
        """
        Initialize class generating NEMO Cartesian grid.

        Parameters
        ----------
        x_f, y_f: 1D array-like
            1D arrays defining the cell faces (units: m).
        """
        self._coords = generate_cartesian_grid(x_f, y_f)

    def flat(self, depth: float) -> Dataset:
        """
        Flat bottom case.

        Parameters
        ----------
        depth: float
            Bottom depth (units: m).

        Returns
        -------
        Dataset
        """
        ds = self._coords
        ds["Bathymetry"] = xr.full_like(ds["glamt"], depth)
        return _add_attributes(_add_mask(ds))


def _add_mask(ds: Dataset) -> Dataset:
    """
    Infer sea mask from bathymetry

    Parameters
    ----------
    ds: Dataset
        Dataset with Bathymetry variable

    Returns
    -------
    Dataset
    """
    ds["mask"] = xr.where(ds["Bathymetry"] > 0, 1, 0)
    return ds


def _add_attributes(ds: Dataset) -> Dataset:
    """
    Add CF attributes to bathymetry and mask variables

    Parameters
    ----------
    ds: Dataset
        Dataset with bathymetry and mask variables

    Returns
    -------
    Dataset
    """
    attrs_dict = {
        "Bathymetry": dict(standard_name="sea_floor_depth_below_geoid", units="m"),
        "mask": dict(standard_name="sea_binary_mask", units="1"),
    }
    for varname, attrs in attrs_dict.items():
        ds[varname].attrs = attrs
        ds[varname].attrs["coordinates"] = "glamt gphit"
    return ds
