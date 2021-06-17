from typing import Callable

import xarray as xr
from xarray import Dataset

from .domzgr.zco import Zco


def _property_and_jpk_check(func: Callable) -> Callable:
    """
    1. Transform the function to property
    2. Raise an error if jpk was not set
    """

    @property  # type: ignore # TODO: "property" used with a non-method.
    def wrapper(self):
        if not self.jpk:
            raise ValueError(
                f"Set `jpk` before calling `obj.domcfg.{func.__name__}`."
                " For example: obj.domcfg.jpk = 31"
            )

        return func(self)

    return wrapper


@xr.register_dataset_accessor("domcfg")
class Accessor:
    def __init__(self, xarray_obj: Dataset):
        self._obj = xarray_obj
        self._jpk: int = 0

    @property
    def jpk(self):
        return self._jpk

    @jpk.setter
    def jpk(self, value: int):
        if value <= 0:
            raise ValueError("`jpk` MUST be > 1")
        self._jpk = value

    @_property_and_jpk_check
    def zco(self) -> Callable:
        return Zco(self._obj, self._jpk).__call__
