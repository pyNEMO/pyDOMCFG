from pkg_resources import DistributionNotFound, get_distribution

from . import utils
from .accessor import Accessor

try:
    __version__ = get_distribution("pydomcfg").version
except DistributionNotFound:
    # package is not installed
    __version__ = "unknown"

__all__ = (
    "Accessor",
    "utils",
)
