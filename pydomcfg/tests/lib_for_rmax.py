#!/usr/bin/env python

import netCDF4 as nc4
import numpy as np
import xarray as xr
from xarray import DataArray, Dataset


# =======================================================================================
def calc_rmax_np(depth):
    """
    Calculate rmax: measure of steepness

    This function returns the slope steepness criteria rmax, which is simply
    (H[0] - H[1]) / (H[0] + H[1])
    Parameters
    ----------
    depth: float
            Bottom depth (units: m).
    Returns
    -------
    rmax: float
            Slope steepness value (units: None)
    """
    rmax_x, rmax_y = np.zeros_like(depth), np.zeros_like(depth)

    rmax_x[:, 1:-1] = 0.5 * (
        np.diff(depth[:, :-1], axis=1) / (depth[:, :-2] + depth[:, 1:-1])
        + np.diff(depth[:, 1:], axis=1) / (depth[:, 1:-1] + depth[:, 2:])
    )
    rmax_y[1:-1, :] = 0.5 * (
        np.diff(depth[:-1, :], axis=0) / (depth[:-2, :] + depth[1:-1, :])
        + np.diff(depth[1:, :], axis=0) / (depth[1:-1, :] + depth[2:, :])
    )

    return np.maximum(np.abs(rmax_x), np.abs(rmax_y))


# =======================================================================================
def calc_rmax_xr1(depth):
    """
    Calculate rmax: measure of steepness
    This function returns the maximum slope paramater

    rmax = abs(Hb - Ha) / (Ha + Hb)

    where Ha and Hb are the depths of adjacent grid cells (Mellor et al 1998).

    Reference:
    *) Mellor, Oey & Ezer, J Atm. Oce. Tech. 15(5):1122-1131, 1998.

    Parameters
    ----------
    depth: DataArray
        Bottom depth (units: m).

    Returns
    -------
    DataArray
        2D maximum slope parameter (units: None)

    """
    both_rmax = []

    for dim in depth.dims:

        # |Ha - Hb| / (Ha + Hb)
        rolled = depth.rolling({dim: 2}).construct("window_dim")
        # Ha - Hb: diff is computed at U/V points
        diff = rolled.diff("window_dim").squeeze("window_dim")
        # rmax is computed at U/V points
        rmax = np.abs(diff) / rolled.sum("window_dim")

        # (rmax_a + rmax_b]) / 2 -> to compute rmax on T points
        rolled = rmax.rolling({dim: 2}).construct("window_dim")
        rmax = rolled.mean("window_dim", skipna=True)

        # 1. Place on the correct index (shift -1 as we rolled twice)
        # 2. Force first/last values = 0
        # 3. Replace land values with 0
        rmax = rmax.shift({dim: -1})
        rmax[{dim: [0, -1]}] = 0
        rmax = rmax.fillna(0)

        both_rmax.append(rmax)

    return np.maximum(*both_rmax)

# =======================================================================================
def calc_rmax_xr2(depth):
    """
    Calculate rmax: measure of steepness
    This function returns the slope steepness criteria rmax, which is simply
    |H[0] - H[1]| / (H[0] + H[1])
    Parameters
    ----------
    depth: float
        Bottom depth (units: m).
    Returns
    -------
    rmax: float
        Slope steepness value (units: None)
    Notes
    -----
    This function uses a "conservative approach" and rmax is overestimated.
    rmax at T points is the maximum rmax estimated at any adjacent U/V point.
    """

    # Loop over x and y
    both_rmax = []
    for dim in depth.dims:

        # Compute rmax
        rolled = depth.rolling({dim: 2}).construct("window_dim")
        diff = rolled.diff("window_dim").squeeze("window_dim")
        rmax = np.abs(diff) / rolled.sum("window_dim")

        # Construct dimension with velocity points adjacent to any T point
        # We need to shift as we rolled twice and to force boundaries = NaN
        rmax = rmax.rolling({dim: 2}).construct("vel_points")
        rmax = rmax.shift({dim: -1})
        rmax[{dim: [0, -1]}] = None

        both_rmax.append(rmax)

    rmax = xr.concat(both_rmax, "vel_points")
    rmax = rmax.max("vel_points", skipna=True)

    return rmax.fillna(0)

# =======================================================================================
def SlopeParam(raw_bathy, msk):

    # This code is slightly modified from
    # https://github.com/ESMG/pyroms/blob/master/bathy_smoother/bathy_smoother/bathy_tools.py
    #
    # This function computes the slope parameter defined as
    #
    #                Z_ij - Z_n
    #                ----------
    #                Z_ij + Z_n
    #
    # where Z_ij is the depth at some point i,j
    # and Z_n is the neighbouring depth in the
    # east,west,south or north sense.
    #
    # This code is adapted from the matlab code
    # "LP Bathymetry" by Mathieu Dutour Sikiric
    # http://drobilica.irb.hr/~mathieu/Bathymetry/index.html
    # For a description of the method, see
    # M. Dutour Sikiric, I. Janekovic, M. Kuzmic, A new approach to
    # bathymetry smoothing in sigma-coordinate ocean models, Ocean
    # Modelling 29 (2009) 128--136.

    """
    SloParMat = SlopeParam(raw_bathy)

    raw_bathy: raw bathymetry interpolated on the model grid.
               It must be a positive depths field.
    msk      : is the mask of the grid

    """

    bathy = np.copy(raw_bathy)
    # print bathy.shape
    nj, ni = bathy.shape

    # Masking land points: bathy is a positive depths field

    bathy[msk == 0.0] = np.nan

    nghb_pnts = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

    SloParMat = np.zeros(bathy.shape)

    for j in range(1, nj - 1):
        for i in range(1, ni - 1):
            if msk[j, i] == 1:
                slopar = 0.0

                for n in range(4):
                    j_nghb = j + nghb_pnts[n][0]
                    i_nghb = i + nghb_pnts[n][1]
                    if msk[j_nghb, i_nghb] == 1:
                        dep1 = bathy[j, i]
                        dep2 = bathy[j_nghb, i_nghb]
                        delta = abs((dep1 - dep2) / (dep1 + dep2))
                        slopar = np.maximum(slopar, delta)
                SloParMat[j, i] = slopar

    return SloParMat
