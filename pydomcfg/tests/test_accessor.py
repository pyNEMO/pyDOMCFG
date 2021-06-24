import pytest

import pydomcfg  # noqa: F401

from .bathymetry import Bathymetry


def test_jpk():
    """Make sure jpk is set correctly"""

    # Initialize test class
    ds_bathy = Bathymetry(1.0e3, 1.2e3, 1, 1).flat(5.0e3)

    # Can't call methods without setting jpk first
    with pytest.raises(ValueError, match="Set `jpk` before calling `obj.domcfg.zco`"):
        ds_bathy.domcfg.zco()

    # jpk must be > 0
    with pytest.raises(ValueError, match="`jpk` MUST be >= 0"):
        ds_bathy.domcfg.jpk = -1

    # Has been set correctly.
    ds_bathy.domcfg.jpk = 1
    assert ds_bathy.domcfg.jpk == 1
