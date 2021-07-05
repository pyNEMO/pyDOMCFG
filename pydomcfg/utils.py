"""
Utilities
"""

import inspect
from collections import ChainMap
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Tuple, TypeVar, Union, cast

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

F = TypeVar("F", bound=Callable[..., Any])


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


def _maybe_float_to_int(value: Any) -> Any:
    """Convert floats that are integers"""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _check_namelist_entries(entries_mapper: Mapping[str, Any]):
    """
    Check whether namelist entries follow NEMO convention for names and types

    Parameters
    ----------
    entries_mapper: Mapping
        Object mapping entry names to their values.

    Notes
    -----
    This function does not accept nested objects.
    I.e., namelist blocks must be chained first.
    """

    prefix_mapper = {
        # Use tuple if multiple types are allowed.
        # Use lists to specify types allowed for each list item.
        "ln_": bool,
        "nn_": int,
        "rn_": (int, float),
        "cn_": str,
        "sn_": [str, int, str, bool, bool, str, str, str, str],
        "jp": int,
        "pp": (type(None), int, float),
        "cp": str,
    }

    def _type_names(
        type_or_tuple: Union[type, Tuple[type, ...]]
    ) -> Union[str, Tuple[str, ...]]:
        """Return type name(s) for a pretty print"""
        if isinstance(type_or_tuple, tuple):
            return tuple(type_class.__name__ for type_class in type_or_tuple)
        return type_or_tuple.__name__

    for key, val in entries_mapper.items():

        for prefix, val_type_or_list_of_types in prefix_mapper.items():
            if key.startswith(prefix):
                val_type = (
                    val_type_or_list_of_types
                    if isinstance(val_type_or_list_of_types, (type, tuple))
                    else list
                )
                list_of_types = (
                    val_type_or_list_of_types
                    if isinstance(val_type_or_list_of_types, list)
                    else []
                )
                break
        else:
            # No match found
            continue

        val_int = _maybe_float_to_int(val)
        if not isinstance(val_int, val_type):
            raise TypeError(
                f"Value does not match expected types for {key!r}."
                f"\nValue: {val!r}"
                f"\nType: {_type_names(type(val_int))!r}"
                f"\nExpected types: {_type_names(val_type)!r}"
            )

        if list_of_types:

            if len(val) != len(list_of_types):
                raise ValueError(
                    f"Mismatch in number of values provided for {key!r}."
                    f"\nValues: {val!r}\nNumber of values: {len(val)!r}"
                    f"\nExpected number of values: {len(list_of_types)!r}"
                )

            val_int = list(map(_maybe_float_to_int, val))
            if not all(map(isinstance, val_int, list_of_types)):
                raise TypeError(
                    f"Values do not match expected types for {key!r}."
                    f"\nValues: {val!r}"
                    f"\nTypes: {list(map(_type_names, map(type, val_int)))!r}"
                    f"\nExpected types: {list(map(_type_names, list_of_types))!r}"
                )


def _check_parameters(func: F) -> F:
    """
    Decorator to check whether parameter names & types follow NEMO conventions.
    To be used with class methods.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        argnames = list(inspect.signature(func).parameters)
        argnames.remove("self")
        args_dict = {name: arg for name, arg in zip(argnames, args)}
        args_and_kwargs = ChainMap(args_dict, kwargs)
        _check_namelist_entries(args_and_kwargs)
        return func(self, *args, **kwargs)

    return cast(F, wrapper)
