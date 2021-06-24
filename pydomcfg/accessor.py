import inspect
import warnings
from collections import ChainMap
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
        self._nml_ref_path = ""

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
        return self._nml_ref_path

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

        # Parse and return a ChainMap
        nml = self._namelist_parser(nml_cfg_path_or_io)

        # Pick zgr class
        if nml["ln_zco"]:
            zgr_class = Zco
        else:
            raise NotImplementedError("Only `Zco` has been implemented so far.")

        # Inspect the __call__ signatures and get the approriate variables
        # from the namelists.
        parameters = inspect.signature(zgr_class.__call__).parameters
        kwargs = {key: nml[key] for key in parameters if key != "self"}
        return Zco(self._obj, nml["jpkdta"])(**kwargs)

    def _namelist_parser(
        self, nml_cfg_path_or_io: Union[str, Path, IO[str]]
    ) -> ChainMap:

        if not HAS_F90NML:
            raise ImportError(
                "`f90nml` MUST be installed" " to use `obj.domcfg.from_namelist()`"
            )

        if not self.nml_ref_path:
            raise ValueError(
                "Set `nml_ref_path` before calling `obj.domcfg.from_namelist()`"
                " For example: obj.domcfg.nml_ref_path = 'path/to/nml_ref'"
            )

        if self.jpk:
            warnings.warn(
                "`obj.domcfg.jpk` is ignored." " `jpk` is inferred from the namelists."
            )

        # Read namelists: cfg overrides ref
        # TODO:
        #   If we specify the third argument, the resulting namelist is written
        #   The advantage is that it is formatted as the reference namelist
        #   We could do it to a tmp file, then add the resulting string to the
        #   global attributes of the final dataset.
        nml_cfg = f90nml.read(nml_cfg_path_or_io)
        nml = f90nml.patch(self.nml_ref_path, nml_cfg)

        # Chain all nam blocks
        # TODO:
        #   Using ChainMap we disregard nam blocks, so assume there aren't
        #   variables with the same name in different nam blocks. Is it OK?
        #   If NO, maybe it's OK if we just select relevant blocks?
        #   ChainMap would make life much easier...
        #   We just look at the zgr call signatures, and we extract the variables
        #   with the same name from any block.
        chained = ChainMap(*nml.todict().values())

        mutually_exclusive = ("ln_zco", "ln_zps", "ln_sco")
        if sum(chained.get(key) for key in mutually_exclusive) != 1:
            raise ValueError(
                "One and only one of the following variables MUST be set:"
                f" {mutually_exclusive}"
            )

        return chained
