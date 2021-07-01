"""
Utilities
"""

from typing import Any, Mapping, Optional

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


def _maybe_to_int(value: Any) -> Any:
    """Convert floats that are integers"""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _check_namelist_entries(entries_mapper: Mapping[str, Any]):

    # Rudimentary checks on namelist entries.
    # TODO:
    #   I haven't really tested this
    prefix_type_mapper = {
        "ln_": bool,
        "nn_": int,
        "rn_": (int, float),
        "cn_": str,
        "sn_": [str, int, str, bool, bool, str, str, str, str],
        "jp": int,
        "pp": (type(None), int, float),
        "cp": str,
    }

    for key, val in entries_mapper.items():

        # Find matching keys
        for prefix, maybe_key_type in prefix_type_mapper.items():
            if key.startswith(prefix):
                # Found a match!
                break
        else:
            # No match, skip key
            continue

        # Get expected types
        maybe_key_type = prefix_type_mapper[prefix]
        if isinstance(maybe_key_type, (type, tuple)):
            # Single type or tuple of types
            key_type = maybe_key_type
            val_types = None
        elif isinstance(maybe_key_type, list):
            # Lists like "sn_"
            key_type = type(maybe_key_type)
            val_types = maybe_key_type
        else:
            raise NotImplementedError

        # Check key type
        if not isinstance(_maybe_to_int(val), key_type):
            raise TypeError(
                f"Value does not match expected types for {key!r}."
                f"\nValue: {val!r}\nType: {type(val)}"
                f"\nExpected types: {key_type}"
            )

        # Check list of values
        if val_types:
            # Check length
            if len(val) != len(val_types):
                raise ValueError(
                    f"Mismatch in number of values provided for {key!r}."
                    f"\nValues: {val}\nLength: {len(val)}"
                    f"\nExpected length: {len(val_types)}"
                )

            # Check type of each element
            for v, v_type in zip(val, val_types):
                if not isinstance(_maybe_to_int(v), v_type):
                    raise TypeError(
                        f"Values do not match expected types for {key!r}."
                        f"\nValues: {val}\nTypes: {list(map(type, val))}"
                        f"\nExpected types: {val_types}"
                    )
