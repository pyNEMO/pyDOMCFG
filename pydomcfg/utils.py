"""
Utilities
"""

from typing import Hashable, Iterable, Iterator, Optional

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

NEMO_NONE = 999_999


def _is_nemo_none(var: Hashable) -> bool:
    """Assess if a NEMO parameter is None"""
    return (var or NEMO_NONE) == NEMO_NONE


def _are_nemo_none(var: Iterable) -> Iterator[bool]:
    """Iterate over namelist parameters and assess if they are None"""
    for v in var:
        yield _is_nemo_none(v)


def _calc_rmax(depth: DataArray) -> DataArray:
    """
    Calculate rmax: measure of steepness
    This function returns the slope paramater field

    r = abs(Hb - Ha) / (Ha + Hb)

    where Ha and Hb are the depths of adjacent grid cells (Mellor et al 1998).

    Reference:
    *) Mellor, Oey & Ezer, J Atm. Oce. Tech. 15(5):1122-1131, 1998.

    Parameters
    ----------
    depth: DataArray
        Bottom depth (units: m).

    Returns
    -------
    DataArray
        2D slope parameter (units: None)

    Notes
    -----
    This function uses a "conservative approach" and rmax is overestimated.
    rmax at T points is the maximum rmax estimated at any adjacent U/V point.
    """
    # Mask land
    depth = depth.where(depth > 0)

    # Loop over x and y
    both_rmax = []
    for dim in depth.dims:

        # Compute rmax
        rolled = depth.rolling({dim: 2}).construct("window_dim")
        diff = rolled.diff("window_dim").squeeze("window_dim")
        rmax = np.abs(diff) / rolled.sum("window_dim")

        # Construct dimension with velocity points adjacent to any T point
        # We need to shift as we rolled twice
        rmax = rmax.rolling({dim: 2}).construct("vel_points")
        rmax = rmax.shift({dim: -1})

        both_rmax.append(rmax)

    # Find maximum rmax at adjacent U/V points
    rmax = xr.concat(both_rmax, "vel_points")
    rmax = rmax.max("vel_points", skipna=True)

    # Mask halo points
    for dim in rmax.dims:
        rmax[{dim: [0, -1]}] = 0

    return rmax.fillna(0)


def _smooth_MB06(
    depth: DataArray,
    rmax: float,
    tol: float = 1.0e-8,
    max_iter: int = 10_000,
) -> DataArray:
    """
    Direct iterative method of Martinho and Batteen (2006) consistent
    with NEMO implementation.

    The algorithm ensures that

                H_ij - H_n
                ---------- < rmax
                H_ij + H_n

    where H_ij is the depth at some point (i,j) and H_n is the
    neighbouring depth in the east, west, south or north direction.

    Reference:
    *) Martinho & Batteen, Oce. Mod. 13(2):166-175, 2006.

    Parameters
    ----------
    depth: DataArray
        Bottom depth.
    rmax: float
        Maximum slope parameter allowed
    tol: float, default = 1.0e-8
        Tolerance for the iterative method
    max_iter: int, default = 10000
        Maximum number of iterations

    Returns
    -------
    DataArray
        Smooth version of the bottom topography with
        a maximum slope parameter < rmax.
    """

    # Set scaling factor used for smoothing
    zrfact = (1.0 - rmax) / (1.0 + rmax)

    # Initialize envelope bathymetry
    zenv = depth

    for _ in range(max_iter):

        # Initialize lists of DataArrays to concatenate
        all_ztmp = []
        all_zr = []
        for dim in zenv.dims:

            # Shifted arrays
            zenv_m1 = zenv.shift({dim: -1})
            zenv_p1 = zenv.shift({dim: +1})

            # Compute zr
            zr = (zenv_m1 - zenv) / (zenv_m1 + zenv)
            zr = zr.where((zenv > 0) & (zenv_m1 > 0), 0)
            for dim_name in zenv.dims:
                zr[{dim_name: -1}] = 0
            all_zr += [zr]

            # Compute ztmp
            zr_p1 = zr.shift({dim: +1})
            all_ztmp += [zenv.where(zr <= rmax, zenv_m1 * zrfact)]
            all_ztmp += [zenv.where(zr_p1 >= -rmax, zenv_p1 * zrfact)]

        # Update envelope bathymetry
        zenv = xr.concat([zenv] + all_ztmp, "dummy_dim").max("dummy_dim")

        # Check target rmax
        zr = xr.concat(all_zr, "dummy_dim")
        if ((np.abs(zr) - rmax) <= tol).all():
            return zenv

    raise ValueError(
        "Iterative method did NOT converge."
        " You might want to increase the number of iterations and/or the tolerance."
    )


def generate_cartesian_grid(
    ppe1_m,
    ppe2_m,
    jpiglo: Optional[int] = None,
    jpjglo: Optional[int] = None,
    ppglam0: float = 0,
    ppgphi0: float = 0,
    chunks: Optional[dict] = None,
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
    chunks: dict, optional
         Chunk sizes along each dimension (e.g., ``{"x": 5, "y": 5}``).
         Requires ``dask`` installed.

    Returns
    -------
    Dataset
        Equivalent of NEMO coordinates file.

    Raises
    ------
    ValueError
        If ``ppe{1,2}_m`` is a vector and ``jp{i,j}glo`` is specified, or viceversa.

    Notes
    -----
    Vectors are loaded into memory. If ``chunks`` is specified, 2D arrays are coerced
    into dask arrays before broadcasting.
    """

    ds = Dataset()
    for dim, ppe, jp, ppg in zip(
        ["x", "y"], [ppe1_m, ppe2_m], [jpiglo, jpjglo], [ppglam0, ppgphi0]
    ):

        # Check and convert ppe to numpy array
        ppe = np.asarray(ppe, dtype=float)
        if (ppe.shape and jp) or (not ppe.shape and not jp):
            raise ValueError(
                "`jp{i,j}glo` must be specified"
                " if and only if `ppe{1,2}_m` is not a vector."
            )

        # c: center f:face
        delta_c = DataArray(ppe if ppe.shape else ppe.repeat(jp), dims=dim)
        coord_f = delta_c.cumsum(dim) + (ppg - 0.5 * delta_c[0])
        coord_c = coord_f.rolling({dim: 2}).mean().fillna(ppg)
        delta_f = coord_c.diff(dim).pad({dim: (0, 1)}, constant_values=delta_c[-1])

        # Add attributes
        for da in [coord_c, coord_f]:
            da.attrs = dict(
                units="m", long_name=f"{dim}-coordinate in Cartesian system"
            )
        for da in [delta_c, delta_f]:
            da.attrs = dict(units="m", long_name=f"{dim}-axis spacing")

        # Fill dataset
        eprefix = "e" + ("1" if dim == "x" else "2")
        gprefix = "g" + ("lam" if dim == "x" else "phi")
        nav_coord = "nav_" + ("lon" if dim == "x" else "lat")
        vel_c = "v" if dim == "x" else "u"
        vel_f = "v" if dim == "y" else "u"
        ds[nav_coord] = ds[gprefix + "t"] = ds[gprefix + vel_c] = coord_c
        ds[gprefix + "f"] = ds[gprefix + vel_f] = coord_f
        ds[eprefix + "t"] = ds[eprefix + vel_c] = delta_c
        ds[eprefix + "f"] = ds[eprefix + vel_f] = delta_f

        # Upgrade dimension to coordinate so we can add CF-attributes
        ds[dim] = ds[dim]
        ds[dim].attrs = dict(axis=dim.upper(), long_name=f"{dim}-dimension index")

    # Generate 2D coordinates (create dask arrays before broadcasting).
    # Order dims (y, x) for convenience (e.g., for plotting).
    (ds,) = xr.broadcast(ds if chunks is None else ds.chunk(chunks))
    ds = ds.transpose(*("y", "x"))

    return ds.set_coords(ds.variables)
