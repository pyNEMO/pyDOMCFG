"""
Base class to generate NEMO v4.0 vertical grids.
"""


from typing import Tuple, Union

import xarray as xr
from xarray import DataArray, Dataset

from ..utils import VerticalGridTuple


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
             xarray dataset including grid coordinates and bathymetry.
             All bathymetry values MUST be >= 0, where 0 is land.
        jpk: int
             number of computational levels
        """

        # Input arguments
        self._bathy = ds_bathy.copy()
        self._jpk = jpk

        # Z dimension
        self._z = DataArray(range(jpk), dims="z")

    # -------------------------------------------------------------------------
    def _compute_sigma(self, kindx: DataArray) -> Tuple[DataArray, ...]:
        """
        Provide the analytical function for sigma-coordinate,
        a uniform non-dimensional vertical coordinate describing
        the non-dimensional position of model levels.

        Consider that -1 <= sigma <= 0, with:

            * sigma = 0 at the shallower boundary

            * sigma = -1 at the deeper boundary

        Parameters
        ----------
        kindx: DataArray
            z-axis indexes

        Returns
        -------
        tuple
            (sigmaT, sigmaW) of Datarrays of uniform non-dimensional
            sigma-coordinate (-1 <= sigma <= 0) for T and W grids
        """

        k_staggered = VerticalGridTuple(t=kindx + 0.5, w=kindx)

        return VerticalGridTuple(*(-k / (self._jpk - 1.0) for k in k_staggered))

    # -------------------------------------------------------------------------
    @staticmethod
    def _compute_z3(
        su: DataArray,
        ss1: DataArray,
        a1: Union[float, DataArray],
        a2: Union[float, DataArray],
        a3: Union[float, DataArray],
    ) -> DataArray:
        """
        Generalised function providing the analytical transformation from computational
        space to physical space. It takes advantage of the fact that z-coordinates can
        be considered a special case of s-coordinate.

        Parameters
        ----------
        su: DataArray
            uniform non-dimensional vertical coordinate s, aka sigma-coordinates.
            0 <= s <= 1
        ss1: DataArray
            stretched non-dimensional vertical coordinate s,
            0 <= s <= 1
        a1, a2, a3: float, DataArray
            parameters of the transformation

        Returns
        -------
        DataArray
            Depths of model levels

        Notes
        -----
        z is downward positive.
        """

        return a1 + a2 * su + a3 * ss1

    # -------------------------------------------------------------------------
    @staticmethod
    def _compute_e3(z3t: DataArray, z3w: DataArray) -> Tuple[DataArray, ...]:
        """
        Return grid cell thickness (e3{t,w}) computed as discrete derivative
        (central-difference) of levels' depth (z3{t,w}).
        """

        z3_staggered = VerticalGridTuple(t=z3t, w=z3w)
        k_to_fill_staggered = VerticalGridTuple(t=-1, w=0)
        e3_staggered = []
        for z3, k_to_fill in zip(z3_staggered, k_to_fill_staggered):
            diff = z3.diff("z")
            fill = 2.0 * (z3t[{"z": k_to_fill}] - z3w[{"z": k_to_fill}])
            e3_staggered.append(xr.concat([diff, fill], "z"))

        return VerticalGridTuple(*e3_staggered)

    # --------------------------------------------------------------------------
    def _merge_z3_and_e3(
        self, z3t: DataArray, z3w: DataArray, e3t: DataArray, e3w: DataArray
    ) -> Dataset:
        """
        Merge {z,e}3{t,w} with ds_bathy, broadcasting and adding CF attributes.
        {z,e}3{t,w} are coordinates of the returned dataset.
        """

        # Merge computed variables with bathymetry
        ds = xr.Dataset({"z3T": z3t, "z3W": z3w, "e3T": e3t, "e3W": e3w})
        ds = ds.broadcast_like(self._bathy["Bathymetry"])
        ds = ds.set_coords(ds.variables)
        ds = ds.merge(self._bathy)

        # Add CF attributes
        ds["z"] = ds["z"]
        ds["z"].attrs = dict(axis="Z", long_name="z-dimension index")
        for var in ["z3T", "z3W"]:
            ds[var].attrs = dict(
                standard_name="depth", long_name="Depth", units="m", positive="down"
            )
        for var in ["e3T", "e3W"]:
            ds[var].attrs = dict(
                standard_name="cell_thickness", long_name="Thickness", units="m"
            )

        return ds
