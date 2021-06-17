import xarray as xr

from .domzgr.zco import Zco


def _property_and_jpk_check(func):
    """
    1. Transform the function to property
    2. Raise an error if jpk was not set
    """

    @property
    def wrapper(self, *args, **kwargs):
        if self.jpk is None:
            raise ValueError(
                f"You must set `jpk` before calling `obj.domcfg.{func.__name__}`."
                " For example: obj.domcfg.jpk = 31"
            )

        return func(self, *args, **kwargs)

    return wrapper


@xr.register_dataset_accessor("domcfg")
class Accessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._jpk = None

    @property
    def jpk(self):
        return self._jpk

    @jpk.setter
    def jpk(self, value):
        self._jpk = value

    @_property_and_jpk_check
    def zco(self):
        return Zco(self._obj, self._jpk).__call__
