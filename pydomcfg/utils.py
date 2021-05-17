"""
Utilities
"""

import xarray as xr
from xarray import DataArray, Dataset


def generate_nemo_rectilinear_grid(delta_x, delta_y) -> Dataset:
    """
    Generate coordinates and spacing of a NEMO rectilinear grid.

    Parameters
    ----------
    delta_x, delta_y: 1D array-like
        1D arrays corresponding to NEMO grid-spacing variables e1t and e2t.

    Returns
    -------
    Dataset
        Equivalent of NEMO coordinates file.
    """

    ds = Dataset()
    for delta, dim in zip([delta_x, delta_y], ["x", "y"]):

        # Initialize DataArray and check values
        # c: center f: face
        delta_c = DataArray(delta, dims=dim)
        last_delta_c = delta_c.isel({dim: -1})
        assert delta_c.all() > 0, "All deltas must be positive!"

        # Infer from delta_c
        # coord_f: cumulative sum of delta_c starting at zero
        # coord_c: rolling mean of coord_f
        # delta_f: diff of coord_c, adding last delta_f == last delta_c
        coord_f = delta_c.cumsum(dim).pad({dim: (1, 0)}, constant_values=0)
        coord_c = coord_f.rolling({dim: 2}).mean()
        delta_f = coord_c.diff(dim).shift({dim: -1}).fillna(last_delta_c)

        # Remove left bound
        coord_c = coord_c.isel({dim: slice(1, None)})
        coord_f = coord_f.isel({dim: slice(1, None)})

        # Start coord_c at zero
        first_coord_c = coord_c.isel({dim: 0})
        coord_c = coord_c - first_coord_c
        coord_f = coord_f - first_coord_c

        # Fill dataset
        eprefix = "e" + ("1" if dim == "x" else "2")
        gprefix = "g" + ("lam" if dim == "x" else "phi")
        nav_coord = "nav_" + ("lon" if dim == "x" else "lat")
        vel_c = "v" if dim == "x" else "u"
        vel_f = "v" if dim == "y" else "u"
        ds[eprefix + "t"] = ds[eprefix + vel_c] = delta_c
        ds[eprefix + "f"] = ds[eprefix + vel_f] = delta_f
        ds[gprefix + "t"] = ds[gprefix + vel_c] = ds[nav_coord] = coord_c
        ds[gprefix + "f"] = ds[gprefix + vel_f] = coord_f

    # Generate 2D coordinates
    # Order dims (y, x) for convenience (e.g., for plotting)
    (ds,) = xr.broadcast(ds)
    ds = ds.transpose(*("y", "x"))

    return ds.set_coords(ds.variables)
