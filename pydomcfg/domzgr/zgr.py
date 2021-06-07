"""
Base class to generate NEMO v4.0 vertical grids.
"""

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
    def sigma(self, grd: str) -> DataArray:
        """
        Provide the analytical function for sigma-coordinate,
        a uniform non-dimensional vertical coordinate describing
        the non-dimensional position of model levels.

        Consider that 0. <= sigma <= -1, with

                 sigma =  0 at the shallower boundary
                 sigma = -1 at the deeper boundary

        Parameters
        ----------
        grd: str
            If we are dealing with "T" or "W" model levels.

        Returns
        -------
        DataArray
            Uniform non-dimensional sigma-coordinate (0. <= sigma <= -1)

        Notes
        -----
        *) T-points are at integer values (between 1 and jpk)
        *) W-points are at integer values - 1/2 (between 0.5 and jpk-0.5)
        """

        kindx = DataArray(
            range(1, self._jpk + 1), coords={"z": range(self._jpk)}, dims="z"
        ).astype(float)

        if grd == "W":
            kindx -= 0.5

        ps = -(kindx - 0.5) / (self._jpk - 1.0)
        return ps

    # -------------------------------------------------------------------------
    @staticmethod
    def compute_z3(
        su: DataArray,
        ss1: DataArray,
        a1: float,
        a2: float,
        a3: float,
        ss2: float = 0.0,
        a4: float = 0.0,
    ) -> DataArray:
        """
        Generalised function providing the analytical
        transformation from computational space to
        physical space. It takes advantage of the fact that
        z-coordinates can be considered a special case of
        s-coordinate.

        Note that z is downward positive.

        Parameters
        ----------
        su: DataArray
            uniform non-dimensional vertical coordinate s,
            aka sigma-coordinates. 0 <= s <= 1
        ss1: DataArray
            stretched non-dimensional vertical coordinate s,
            0 <= s <= 1
        a1: float
            parameter of the transformation
        a2: float
            parameter of the transformation
        a3: float
            parameter of the transformation
        ss2: float
            second stretched non-dimensional vertical coordinate s,
            0 <= s <= 1 (only used for zco with ldbletanh = True)
        a4: float
            parameter of the transformation (only used for zco with ldbletanh = True)

        Returns
        -------
        DataArray
            Depths of model levels
        """

        z = a1 + a2 * su + a3 * ss1 + a4 * ss2
        return z

    # -------------------------------------------------------------------------
    def compute_e3(self, ds: Dataset):
        """
        Grid cell thickness computed as discrete derivative
        (central-difference) of levels' depth

        Parameters
        ----------
        ds: Dataset
            xarray dataset with empty ``e3{T,W}`` and ``z3{T,W}``
            correctly computed
        Returns
        -------
        ds: Dataset
            xarray dataset with ``e3{T,W}`` correctly computed
        """
        ds["e3T"] = ds["z3W"].diff("z", label="lower")
        ds["e3W"] = ds["z3T"].diff("z", label="upper")

        # Bottom:
        for varname, k in zip(["e3T", "e3W"], [-1, 0]):
            ds[varname][{"z": k}] = 2.0 * (ds["z3T"][{"z": k}] - ds["z3W"][{"z": k}])

        return ds
