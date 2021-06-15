#!/usr/bin/env python

"""
Class to generate NEMO v4.0 s-coordinates
"""

from typing import Optional  # , Tuple

# import numpy as np
from xarray import Dataset

from .zgr import Zgr

# from pydomcfg.utils import is_nemo_none


class Sco(Zgr):
    """
    Class to generate terrain-following coordinates dataset objects.
    Currently, four types of terrain-following grids can be genrated:
    *) uniform sigma-coordinates (Phillips 1957)
    *) stretched s-coordinates with Song & Haidvogel 1994 stretching
    *) stretched s-coordinates with Siddorn & Furner 2013 stretching
    *) stretched s-coordinates with Madec et al. 1996 stretching

    Method
    ------
    *) Model levels' depths depT/W are defined from analytical function.
    *) Model vertical scale factors e3 (i.e., grid cell thickness) can
       be computed as

             1) analytical derivative of depth function
                (ln_e3_dep=False); for backward compatibility with v3.6.
             2) discrete derivative (central-difference) of levels' depth
                (ln_e3_dep=True). The only possibility from v4.0.

    References:
       *) NEMO v4.0 domzgr/{zgr_sco,s_sh94,s_sf12,s_tanh} subroutine
       *) Phillips, J. Meteorol., 14, 184-185, 1957.
       *) Song & Haidvogel, J. Comp. Phy., 115, 228-244, 1994.
       *) Siddorn & Furner, Oce. Mod. 66:1-13, 2013.
       *) Madec, Delecluse, Crepon & Lott, JPO 26(8):1393-1408, 1996.
    """

    # --------------------------------------------------------------------------
    def __call__(
        self,
        bot_min: float,
        bot_max: float,
        hc: float = 0.0,
        rmax: Optional[float] = None,
        stretch: Optional[str] = None,
        psurf: Optional[float] = None,
        pbott: Optional[float] = None,
        alpha: Optional[float] = None,
        efold: Optional[float] = None,
        pbot2: Optional[float] = None,
        ln_e3_dep: bool = True,
    ) -> Dataset:
        """
        Generate NEMO terrain-following model levels.

        Parameters
        ----------
        bot_min: float
            Minimum depth of bottom topography surface (>0) (m)
        bot_max: float
            Maximum depth of bottom topography surface (>0) (m)
        hc: float
            critical depth for transition from uniform sigma
            to stretched s-coordinates (>0) (m)
        rmax: float, optional
            maximum slope parameter value allowed
        stretch: str, optional
            Type of stretching applied:
            *)  None  = no stretching, i.e. uniform sigma-coord.
            *) "sh94" = Song & Haidvogel 1994 stretching
            *) "sf12" = Siddorn & Furner 2013 stretching
            *) "md96" = Madec et al. 1996 stretching
        psurf: float, optional
            sh94: surface control parameter (0<= psurf <=20)
            md96: surface control parameter (0<= psurf <=20)
            sf12: thickness of first model layer (m)
        pbott: float, optional
            sh94: bottom control parameter (0<= pbott <=1)
            md96: bottom control parameter (0<= pbott <=1)
            sf12: scaling factor for computing thickness
            of bottom level Zb
        alpha: float, optional
            sf12: stretching parameter
        efold: float, optional
            sf12: efold length scale for transition from sigma
            to stretched coord
        pbot2: float, optional
            sf12 offset for calculating Zb = H*pbott + pbot2
        ln_e3_dep: bool
            Logical flag to comp. e3 as fin. diff. (True) or
            analyt. (False) (default = True)

        Returns
        -------
        Dataset
           Describing the 3D geometry of the model
        """

        self._bot_min = bot_min
        self._bot_max = bot_max
        self._hc = hc
        self.rmax = rmax
        self._stretch = stretch
        self._ln_e3_dep = ln_e3_dep

        # set stretching parameters after checking their consistency
        if self._stretch:
            self._set_stretch_par(psurf, pbott, alpha, efold, pbot2)

        ds = self._init_ds()

        # compute envelope bathymetry
        ds_env = self._compute_env(ds)

        # compute sigma-coordinates for z3 computation
        kindx = ds_env["z"]
        sigma = (self._compute_sigma(kk) for kk in kindx)
        self._sigT, self._sigW = sigma

        # compute z3 depths of zco vertical levels
        # dsz = self._sco_z3(ds_env)

        # compute e3 scale factors
        # dse = self._compute_e3(dsz) if self._ln_e3_dep else self._analyt_e3(dsz)

        # return dse

    # --------------------------------------------------------------------------
    def _set_stretch_par(self, psurf, pbott, alpha, efold, pbot2):
        """
        Set stretching parameters after checking
        consistency of input parameters
        """
        if not (psurf and pbott):
            if self._stretch == "sh94":
                srf = "rn_theta"
                bot = "rn_bb"
            elif self._stretch == "md96":
                srf = "rn_theta"
                bot = "rn_thetb"
            elif self._stretch == "sf12":
                srf = "rn_zs"
                bot = "rn_zb_a"
            msg = (
                srf
                + " and "
                + bot
                + "MUST be set when using "
                + self._stretch
                + " stretching."
            )
            raise ValueError(msg)

        if self._stretch == "sf12":
            if not (alpha and efold and pbot2):
                msg = "rn_alpha, rn_efold and rn_zb_b MUST be set when \
                       using sf12 stretching."
                raise ValueError(msg)

        # setting stretching parameters
        self._psurf = psurf if psurf else 0.0
        self._pbott = pbott if pbott else 0.0
        self._alpha = alpha if alpha else 0.0
        self._efold = efold if efold else 0.0
        self._pbot2 = pbot2 if pbot2 else 0.0

    # --------------------------------------------------------------------------
    def _compute_env(self, ds: Dataset) -> Dataset:
        """
        Compute the envelope bathymetry surface by applying the
        Martinho & Batteen (2006) smoothing algorithm to the
        actual topography to reduce the maximum value of the slope
        parameter
                      r = abs(Hb-Ha) / (Ha+Hb)

        where Ha and Hb are the depths of adjacent grid cells.
        The maximum slope parameter is reduced to be <= rmax.

        Reference:
          *) Martinho & Batteen, Oce. Mod. 13(2):166-175, 2006.

        Parameters
        ----------
        ds: Dataset
            xarray dataset with the 2D bottom topography DataArray
        Returns
        -------
        ds: Dataset
            xarray dataset with the 2D envelope bathymetry DataArray
        """

        return ds
