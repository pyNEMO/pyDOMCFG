import warnings
from pathlib import Path
from typing import IO, Any, Callable, TypeVar, Union, cast

import xarray as xr
from xarray import Dataset

from .domzgr.zco import Zco

try:
    import f90nml

    HAS_F90NML = True
except ImportError:
    HAS_F90NML = False

F = TypeVar("F", bound=Callable[..., Any])


def _jpk_check(func: F) -> F:
    """
    Decorator to raise an error if jpk was not set
    """

    def wrapper(self, *args, **kwargs):
        if not self.jpk:
            raise ValueError(
                f"Set `jpk` before calling `obj.domcfg.{func.__name__}`."
                " For example: obj.domcfg.jpk = 31"
            )
        return func(self, *args, **kwargs)

    return cast(F, wrapper)


@xr.register_dataset_accessor("domcfg")
class Accessor:
    def __init__(self, xarray_obj: Dataset):
        self._obj = xarray_obj
        self._jpk = 0
        self._nml_ref = None

    # Set attributes
    @property
    def jpk(self) -> int:
        """
        Number of computational levels

        Returns
        -------
        int
        """
        return self._jpk

    @jpk.setter
    def jpk(self, value: int):
        if value < 0:
            raise ValueError("`jpk` MUST be >= 0 (use 0 to unset jpk)")
        self._jpk = value

    @property
    def nml_ref_path(self) -> str:
        """
        Path to reference namelist.

        Returns
        -------
        str
        """
        return self.nml_ref_path

    @nml_ref_path.setter
    def nml_ref_path(self, value: str):
        self._nml_ref_path = value

    # domzgr methods
    @_jpk_check
    def zco(self, *args: Any, **kwargs: Any) -> Dataset:
        return Zco(self._obj, self._jpk)(*args, **kwargs)

    zco.__doc__ = Zco.__call__.__doc__

    # Emulate NEMO DOMAINcfg tools
    def from_namelist(self, nml_cfg_path_or_io: Union[str, Path, IO[str]]):
        """
        TODO
        """

        # Sanity checks
        self._namelist_checks

        # Read namelists: cfg overrides ref
        # TODO:
        #   If we specify the tird argument, the resulting namelist is written
        #   The advantage is that it is formatted as the reference namelist
        #   We could to it to a tmp file, then add the resulting string to the
        #   global attributes of the final dataset.
        nml = f90nml.patch(self.nml_ref_path, f90nml.read(nml_cfg_path_or_io))

    @property
    def _namelist_checks(self):

        if not HAS_F90NML:
            raise ImportError("`f90nml` MUST be installed to use `obj.from_namelist()`")

        if not self.nml_ref_path:
            raise ValueError(
                "Set `nml_ref_path` before calling `obj.from_namelist`"
                " For example: obj.domcfg.nml_ref_path = 'path/to/nml_ref'"
            )

        if self.jpk:
            warnings.warn(
                "`obj.domcfg.jpk` is ignored." " `jpk` is inferred from the namelists."
            )
