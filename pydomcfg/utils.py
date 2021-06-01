"""
Utilities
"""

import xarray as xr
from xarray import DataArray, Dataset


def generate_cartesian_grid(x_f, y_f) -> Dataset:
    """
    Generate coordinates and spacing of a NEMO Cartesian grid.

    Parameters
    ----------
    x_f, y_f: 1D array-like
        1D arrays defining the cell faces (units: m).

    Returns
    -------
    Dataset
        Equivalent of NEMO coordinates file.

    Note
    ----
    The first face is not included in the output dataset.
    """

    ds = Dataset()
    for coord, dim in zip([y_f, x_f], ["y", "x"]):

        # c: center f:face
        coord_f = DataArray(coord, dims=dim)
        coord_c = coord_f.rolling({dim: 2}).mean().dropna(dim)
        delta_c = coord_f.diff(dim)
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
            range(len(coord) - 1), dims=dim, attrs=dict(axis=dim.upper())
        )

    # Generate 2D coordinates
    # Order dims (y, x) for convenience (e.g., for plotting)
    (ds,) = xr.broadcast(ds)
    ds = ds.transpose(*("y", "x"))

    return ds.set_coords(ds.variables)
