"""
Module to generate datasets for testing
"""

import numpy as np
import xarray as xr

from pydomcfg.utils import generate_nemo_rectilinear_grid

# Flat bottom
delta_x = np.full((5,), 1.0e3)
delta_y = np.full((10,), 1.0e3)
ds_flat = generate_nemo_rectilinear_grid(delta_x, delta_y)
ds_flat["Bathymetry"] = xr.full_like(ds_flat["glamt"], 1.0e3)
ds_flat["mask"] = xr.where(ds_flat["Bathymetry"] > 0, 1, 0)

# Island
delta_x = np.full((10,), 1.0e3)
delta_y = np.full((10,), 1.0e3)
ds_island = generate_nemo_rectilinear_grid(delta_x, delta_y)
ds_island["Bathymetry"] = np.hypot(
    *(ds_island[coord] - ds_island[coord].mean() for coord in ["glamt", "gphit"])
)
ds_island["Bathymetry"] -= ds_island["Bathymetry"].min()
ds_island["Bathymetry"] /= ds_island["Bathymetry"].max()
ds_island["Bathymetry"] *= 1.0e3
ds_island["mask"] = xr.where(ds_island["Bathymetry"] > 0, 1, 0)
