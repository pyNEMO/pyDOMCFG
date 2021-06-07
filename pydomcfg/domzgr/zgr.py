"""
Base class to generate NEMO v4.0 vertical grids.
"""

from itertools import product

import numpy as np
from xarray import DataArray, Dataset


class Zgr:
    """
    Base class to generate NEMO vertical grids.
    """

    def __init__(self, ds_bathy: Dataset, jpk: int):
        """
        Initialize class.

        Parameters
        ----------
        ds_bathy: Dataset
             xarray dataset including grid coordinates and bathymetry
        jpk: int
             number of computational levels
        """

        self._bathy = ds_bathy
        self._jpk = jpk

    # -------------------------------------------------------------------------
    def init_ds(self) -> Dataset:
        ds = self._bathy.copy()
        jpi = ds.dims["x"]
        jpj = ds.dims["y"]
        jpk = self._jpk

        var = ["z3", "e3"]
        grd = ["T", "W"]
        crd = [("z", range(jpk)), ("y", range(jpj)), ("x", range(jpi))]

        da = DataArray(np.full((jpk, jpj, jpi), np.nan), coords=crd)

        # Initialise a dataset with z3 and e3 dataarray filled with nan
        da = DataArray(np.zeros(shape=(jpk, jpj, jpi)) * np.nan, coords=crd)
        for v, g in product(var, grd):
            ds[v + g] = da.copy()
            ds = ds.set_coords(v + g)
        return ds

    # -------------------------------------------------------------------------
    def sigma(self, k: int, grd: str) -> float:
        """
        Provide the analytical function for sigma-coordinate,
        a uniform non-dimensional vertical coordinate describing
        the non-dimensional position of model levels.

        Consider that 0. <= sigma <= -1, with

                 sigma =  0 at the shallower boundary
                 sigma = -1 at the deeper boundary

        Parameters
        ----------
        k: int
          Model level index. Note that
             *) T-points are at integer values (between 1 and jpk)
             *) W-points are at integer values - 1/2 (between 0.5 and jpk-0.5)
        grd: str
          If we are dealing with "T" or "W" model levels.

        Returns
        -------
        float
        """

        kindx = float(k + 1)  # to deal with python convention
        if grd == "W":
            kindx -= 0.5

        ps = -(kindx - 0.5) / float(self._jpk - 1)
        return ps

    # -------------------------------------------------------------------------
    @staticmethod
    def compute_z3(su, ss1, a1, a2, a3, ss2=0.0, a4=0.0):
        """
        Generalised function providing the analytical
        transformation from computational space to
        physical space. It takes advantage of the fact that
        z-coordinates can be considered a special case of
        s-coordinate.

        Note that z is downward positive.

        Parameters
        ----------
        su  : uniform non-dimensional vertical coordinate s,
              aka sigma-coordinates. 0 <= s <= 1
        ss1 : stretched non-dimensional vertical coordinate s,
              0 <= s <= 1
        ss2 : second stretched non-dimensional vertical coordinate s,
              0 <= s <= 1 (only used for zco with ldbletanh = True)
        a1  : parameter of the transformation
        a2  : parameter of the transformation
        a3  : parameter of the transformation
        a4  : parameter of the transformation (only used for zco with ldbletanh = True)
        """

        z = a1 + a2 * su + a3 * ss1 + a4 * ss2
        return z

    # -------------------------------------------------------------------------
    def compute_e3(self, ds: Dataset) -> Dataset:
        """
        Grid cell thickness computed as discrete derivative
        (central-difference) of levels' depth

        Parameters
        ----------
        ds: Dataset

        Returns
        -------
        Dataset
        """
        for k in range(self._jpk - 1):
            ds["e3T"][k, :, :] = ds["z3W"][k + 1, :, :] - ds["z3W"][k, :, :]
            ds["e3W"][k + 1, :, :] = ds["z3T"][k + 1, :, :] - ds["z3T"][k, :, :]
        # Bottom:
        k = -1
        ds["e3T"][k, :, :] = 2.0 * (ds["z3T"][k, :, :] - ds["z3W"][k, :, :])
        # Surface:
        k = 0
        ds["e3W"][k, :, :] = 2.0 * (ds["z3T"][k, :, :] - ds["z3W"][k, :, :])
        return ds
