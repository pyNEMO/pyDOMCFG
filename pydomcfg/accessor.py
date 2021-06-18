from typing import Any, Callable, TypeVar, cast

import xarray as xr
from xarray import Dataset

from .domzgr.zco import Zco

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

    @property
    def jpk(self):
        return self._jpk

    @jpk.setter
    def jpk(self, value: int):
        if value <= 0:
            raise ValueError("`jpk` MUST be > 0")
        self._jpk = value

    @_jpk_check
    def zco(self, *args, **kwargs) -> Dataset:
        return Zco(self._obj, self._jpk)(*args, **kwargs)
    zco.__doc__ = Zco.__call__.__doc__