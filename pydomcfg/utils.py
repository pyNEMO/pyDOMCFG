"""
Utilities
"""

from typing import Optional

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset


def generate_cartesian_grid(
    ppe1_m,
    ppe2_m,
    jpiglo: Optional[int] = None,
    jpjglo: Optional[int] = None,
    ppglam0: float = 0,
    ppgphi0: float = 0,
) -> Dataset:
    """
    Generate coordinates and spacing of a NEMO Cartesian grid.

    Parameters
    ----------
    ppe1_m, ppe2_m: float, 1D array-like
        Grid spacing along x/y axis (units: m).
    jpiglo, jpjglo: int, optional
        Size of x/y dimension.
    ppglam0, ppgphi0: float
        x/y coordinate of first T-point (units: m).

    Returns
    -------
    Dataset
        Equivalent of NEMO coordinates file.

    Raises
    ------
    ValueError
        If ppe{1,2}_m is of type 1D array-like and jp{i,j}glo is not None,
        or ppe{1,2}_m is a multidimensional array.
    """

    ds = Dataset()
    for dim, ppe, jp, ppg in zip(
        ["x", "y"], [ppe1_m, ppe2_m], [jpiglo, jpjglo], [ppglam0, ppgphi0]
    ):

        ppe = np.asarray(ppe, dtype=float)
        if (ppe.shape and jp) or (not ppe.shape and not jp):
            raise ValueError("Do not specify jp{i,j}glo if ppe{1,2}_m is a vector")
        elif len(ppe.shape) > 1:
            raise ValueError("jp{i,j}glo must be a number or a vector.")

        # c: center f:face
        delta_c = DataArray(ppe if ppe.shape else np.full(jp, ppe), dims=dim)
        coord_f = delta_c.pad(
            {dim: (1, 0)}, constant_values=ppg - 0.5 * delta_c[0]
        ).cumsum(dim)
        coord_c = coord_f.rolling({dim: 2}).mean().dropna(dim)
        delta_f = coord_c.diff(dim).pad({dim: (0, 1)}, constant_values=delta_c[-1])

        # Remove coord_f left bound
        coord_f = coord_f.isel({dim: slice(1, None)})

        # Add attributes
        for da in [coord_c, coord_f]:
            da.attrs = dict(
                units="m", long_name=f"{dim}-coordinate in Cartesian system"
            )
        for da in [delta_c, delta_f]:
            da.attrs = dict(units="m", long_name=f"{dim}-axis spacing")

        # Fill dataset and add attributes
        eprefix = "e" + ("1" if dim == "x" else "2")
        gprefix = "g" + ("lam" if dim == "x" else "phi")
        nav_coord = "nav_" + ("lon" if dim == "x" else "lat")
        vel_c = "v" if dim == "x" else "u"
        vel_f = "v" if dim == "y" else "u"
        ds[nav_coord] = ds[gprefix + "t"] = ds[gprefix + vel_c] = coord_c
        ds[gprefix + "f"] = ds[gprefix + vel_f] = coord_f
        ds[eprefix + "t"] = ds[eprefix + vel_c] = delta_c
        ds[eprefix + "f"] = ds[eprefix + vel_f] = delta_f
        ds[dim] = DataArray(
            range(jp or len(ppe)), dims=dim, attrs=dict(axis=dim.upper())
        )

    # Generate 2D coordinates
    # Order dims (y, x) for convenience (e.g., for plotting)
    (ds,) = xr.broadcast(ds)
    ds = ds.transpose(*("y", "x"))

    return ds.set_coords(ds.variables)