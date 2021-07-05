import inspect
import warnings
from collections import ChainMap
from functools import wraps
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
ZGR_MAPPER = {
    "ln_zco": Zco,
    # "ln_zps": TODO,
    # "ln_sco": TODO
}


def _jpk_check(func: F) -> F:
    """
    Decorator to raise an error if jpk was not set
    """

    @wraps(func)
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
    # TODO:
    #   I think the process of creating the public API and doc
    #   can be further automatized, but let's not put too much effort into it
    #   until we settle on the back-end structure:
    #   See: https://github.com/pyNEMO/pyDOMCFG/issues/45
    @_jpk_check
    def zco(self, *args: Any, **kwargs: Any) -> Dataset:
        name = inspect.stack()[0][3]
        return ZGR_MAPPER["ln_" + name](self._obj, self._jpk)(*args, **kwargs)

    zco.__doc__ = Zco.__call__.__doc__

    # Emulate NEMO DOMAINcfg tools
    def from_namelist(self, nml_cfg_path_or_io: Union[str, Path, IO[str]]) -> Dataset:
        """
        Auto-populate pydomcfg parameters using NEMO DOMAINcfg namelists.

        Parameters
        ----------
        nml_cfg_path_or_io: str, Path, IO
            Path pointing to a namelist_cfg,
            or namelist_cfg previously opened with open()

        Returns
        -------
        Dataset
        """

        nml_chained = self._namelist_parser(nml_cfg_path_or_io)
        zgr_initialized, kwargs = self._get_zgr_initialized_and_kwargs(nml_chained)
        return zgr_initialized(**kwargs)

    def _namelist_parser(
        self, nml_cfg_path_or_io: Union[str, Path, IO[str]]
    ) -> ChainMap:
        """Parse namelists using f90nml, chaining all namblocks"""

        if not HAS_F90NML:
            raise ImportError(
                "`f90nml` MUST be installed to use `obj.domcfg.from_namelist()`"
            )

        if not self.nml_ref_path:
            raise ValueError(
                "Set `nml_ref_path` before calling `obj.domcfg.from_namelist()`"
                " For example: obj.domcfg.nml_ref_path = 'path/to/nml_ref'"
            )

        if self.jpk:
            warnings.warn(
                "`obj.domcfg.jpk` is ignored. `jpk` is inferred from the namelists."
            )

        # Read namelists: cfg overrides ref
        nml_cfg = f90nml.read(nml_cfg_path_or_io)
        nml = f90nml.patch(self.nml_ref_path, nml_cfg)

        return ChainMap(*nml.todict().values())

    def _get_zgr_initialized_and_kwargs(self, nml_chained: ChainMap):

        # TODO: Add return type hint when abstraction in base class is implemented

        # Pick the appropriate class
        zgr_classes = [
            value for key, value in ZGR_MAPPER.items() if nml_chained.get(key)
        ]
        if len(zgr_classes) != 1:
            raise ValueError(
                "One and only one of the following variables MUST be `.true.`:"
                f" {tuple(ZGR_MAPPER)}"
            )
        zgr_class = zgr_classes[0]

        # Compatibility with NEMO DOMAINcfg
        if nml_chained.get("ldbletanh") is False:
            for pp in ["ppa2", "ppkth2", "ppacr2"]:
                nml_chained[pp] = None

        # Get kwargs, converting 999_999 to None
        parameters = list(inspect.signature(zgr_class.__call__).parameters)
        parameters.remove("self")
        kwargs = {
            key: None if nml_chained[key] == 999_999 else nml_chained[key]
            for key in parameters
            if key in nml_chained
        }

        return zgr_class(self._obj, nml_chained["jpkdta"]), kwargs
