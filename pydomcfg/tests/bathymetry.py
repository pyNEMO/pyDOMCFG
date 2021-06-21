"""
Module to generate datasets for testing
"""

import numpy as np
import xarray as xr
from xarray import Dataset

from pydomcfg.utils import generate_cartesian_grid


class Bathymetry:
    """
    Class to generate idealized test bathymetry datasets.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize class generating NEMO Cartesian grid.

        Parameters
        ----------
        *args
            Arguments passed on to :py:func:`pydomcfg.utils.generate_cartesian_grid`
        *kwargs
            Keyword arguments passed on to
            :py:func:`pydomcfg.utils.generate_cartesian_grid`.
        """
        self._coords = generate_cartesian_grid(*args, **kwargs)

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

    def sea_mount(self, depth: float, stiff: float = 1) -> Dataset:
        """
        Channel with seamount case.

        Produces bathymetry of a channel with a Gaussian seamount in order to
        simulate an idealised test case. Based on Marsaleix et al., 2009
        doi:10.1016/j.ocemod.2009.06.011 Eq. 15.

        Parameters
        ----------
        depth: float
            Bottom depth (units: m).
        stiff: float
            Scale factor for steepness of seamount (units: None)

        Returns
        -------
        Dataset
        """
        ds = self._coords

        # Find half way point for sea mount location
        half_way = {k: v // 2 for k, v in ds.sizes.items()}
        glamt_mid, gphit_mid = (g.isel(half_way) for g in (ds.glamt, ds.gphit))

        # Define sea mount bathymetry
        ds["Bathymetry"] = depth * (
            1.0
            - 0.9
            * np.exp(
                -(
                    stiff
                    / 40.0e3 ** 2
                    * ((ds.glamt - glamt_mid) ** 2 + (ds.gphit - gphit_mid) ** 2)
                )
            )
        )

        # Add rmax of Bathymetry
        # ds["rmax"] = DataArray(
        #    _calc_rmax(ds["Bathymetry"].to_masked_array()), dims=["y", "x"]
        # )
        ds["rmax"] = _calc_rmax(ds["Bathymetry"])

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
    ds["mask"] = xr.where(ds["Bathymetry"] > 0, 1, 0)  # TODO: should this be bool?
    return ds


def _add_attributes(ds: Dataset) -> Dataset:
    """
    Add CF attributes to bathymetry and mask variables

    Parameters
    ----------
    ds: Dataset
        Dataset with bathymetry and mask variables [and rmax]

    Returns
    -------
    Dataset
    """

    attrs_dict: dict = {
        "Bathymetry": dict(standard_name="sea_floor_depth_below_geoid", units="m"),
        "mask": dict(standard_name="sea_binary_mask", units="1"),
    }

    if "rmax" in ds:
        attrs_dict["rmax"] = dict(standard_name="rmax", units="1")

    for varname, attrs in attrs_dict.items():
        ds[varname].attrs = attrs
        ds[varname].attrs["coordinates"] = "glamt gphit"
    return ds


def _calc_rmax(depth):
    """
    Calculate rmax: measure of steepness

    This function returns the slope steepness criteria rmax, which is simply
    (H[0] - H[1]) / (H[0] + H[1])

    Parameters
    ----------
    depth: float
        Bottom depth (units: m).

    Returns
    -------
    rmax: float
        Slope steepness value (units: None)
    """

    both_rmax = []
    for dim in depth.dims:

        rolled = depth.rolling({dim: 2}).construct("tmp_dim")

        # |(H[0] - H[1])| / (H[0] + H[1])
        # First value is NaN
        diff = rolled.diff("tmp_dim").squeeze("tmp_dim")
        rmax = diff / rolled.sum("tmp_dim")

        # (rmax[0] + rmax[1]) / 2
        # First two values are NaN
        rmax = rmax.rolling({dim: 2}).mean()

        # First and last values are zero
        rmax = rmax.shift({dim: -1}).fillna(0)

        both_rmax.append(np.abs(rmax))

    return np.maximum(*both_rmax)
