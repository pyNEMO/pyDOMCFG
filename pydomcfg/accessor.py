import xarray as xr

from .domzgr.zco import Zco


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

    @property
    def zco(self):
        return Zco(self._obj, self._jpk)
