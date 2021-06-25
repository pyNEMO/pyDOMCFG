#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import netCDF4 as nc4
from lib_for_rmax import *
import xarray as xr

rn_sbot_max = 7000.
rn_sbot_min = 10. 
rn_rmax = 0.24

nc = "/home/h01/dbruciaf/mod_dev/jmmp_MEs/amm7_mes_vgrid/MEs_2env_0.24_0.07_opt_v2/bathymeter_mo_ps43.nc"
ds_bathy = xr.open_dataset(nc)
bathy = ds_bathy["Bathymetry"].squeeze()

# Computing LSM
ocean = xr.where(bathy > 0, 1, 0) 

# setting max depth
bathy = np.minimum(bathy, rn_sbot_max)

# setting min depth
bathy = np.maximum(bathy, rn_sbot_min)*ocean

da_env = bathy.copy()
zenv = da_env.data

nj = zenv.shape[0]
ni = zenv.shape[1]

# set first land point adjacent to a wet cell to
# min_dep as this needs to be included in smoothing
for j in range(nj - 1):
    for i in range(ni - 1):
          if not ocean[j, i]:
             ip1 = np.minimum(i + 1, ni)
             jp1 = np.minimum(j + 1, nj)
             im1 = np.maximum(i - 1, 0)
             jm1 = np.maximum(j - 1, 0)
             if (
                 bathy[jp1, im1]
                 + bathy[jp1, i]
                 + bathy[jp1, ip1]
                 + bathy[j, im1]
                 + bathy[j, ip1]
                 + bathy[jm1, im1]
                 + bathy[jm1, i]
                 + bathy[jm1, ip1]
                 ) > 0.0:
                 zenv[j, i] = rn_sbot_min

# set scaling factor used for smoothing
zrfact = (1.0 - rn_rmax) / (1.0 + rn_rmax)

# initialise temporary evelope depth arrays
ztmpi1 = zenv.copy()
ztmpi2 = zenv.copy()
ztmpj1 = zenv.copy()
ztmpj2 = zenv.copy()

# initial maximum slope parameter
zrmax = 1.
zri = np.ones(zenv.shape)
zrj = np.ones(zenv.shape)

tol = 1.0e-8
itr = 0
max_itr = 10000

while itr <= max_itr and (zrmax - rn_rmax) > tol:

      itr += 1
      zrmax = 0.0
      # we set zrmax from previous r-values (zri and zrj) first
      # if set after current r-value calculation (as previously)
      # we could exit DO WHILE prematurely before checking r-value
      # of current zenv
      #max_zri = np.nanmax(np.absolute(zri))
      #max_zrj = np.nanmax(np.absolute(zrj))
      #zrmax = np.nanmax([zrmax, max_zrj, max_zri])
      for j in range(nj):
          for i in range(ni):
              zrmax = np.amax([zrmax, np.absolute(zri[j,i]), np.absolute(zrj[j,i])])

      print("Iter:", itr, "rmax: ", zrmax)

      zri *= 0.0
      zrj *= 0.0

      for j in range(nj - 1):
          for i in range(ni - 1):
              ip1 = np.minimum(i + 1, ni)
              jp1 = np.minimum(j + 1, nj)
              if zenv[j, i] > 0.0 and zenv[j, ip1] > 0.0:
                 zri[j, i] = (zenv[j, ip1] - zenv[j, i]) / (zenv[j, ip1] + zenv[j, i]) 
              if zenv[j, i] > 0.0 and zenv[jp1, i] > 0.0:
                 zrj[j, i] = (zenv[jp1, i] - zenv[j, i]) / (zenv[jp1, i] + zenv[j, i])
              if zri[j, i] > rn_rmax:
                 ztmpi1[j, i] = zenv[j, ip1] * zrfact
              if zri[j, i] < -rn_rmax:
                 ztmpi2[j, ip1] = zenv[j, i] * zrfact
              if zrj[j, i] > rn_rmax:
                 ztmpj1[j, i] = zenv[jp1, i] * zrfact
              if zrj[j, i] < -rn_rmax:
                 ztmpj2[jp1, i] = zenv[j, i] * zrfact

      for j in range(nj):
          for i in range(ni):
               zenv[j,i] = np.amax([zenv[j,i], ztmpi1[j,i], ztmpi2[j,i], ztmpj1[j,i], ztmpj2[j,i]])

      #ztmpi = np.maximum(ztmpi1, ztmpi2)
      #ztmpj = np.maximum(ztmpj1, ztmpj2)
      #zenv = np.maximum(zenv, np.maximum(ztmpi, ztmpj))

# set all points to avoid undefined scale value warnings
zenv = np.maximum(zenv, rn_sbot_min)

da_env.data = zenv

# calc rmax according to:

# 1. calc_rmax_xr
rmax = np.amax(calc_rmax_xr(da_env)*ocean).values 
print("1. calc_rmax_xr: " + str(rmax))

# 2. calc_rmax_np
rmax = np.amax(calc_rmax_np(zenv)*ocean.values)
print("2. calc_rmax_np: " + str(rmax))

# 3. SlopeParam
rmax = np.amax(SlopeParam(zenv,ocean.values))
print("3. SlopeParam: " + str(rmax))

da_env.data = calc_rmax_np(zenv)*ocean.values - SlopeParam(zenv,ocean.values)
da_env.plot()
plt.show()
