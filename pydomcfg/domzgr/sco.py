#!/usr/bin/env python

"""
Class to generate NEMO v4.0 s-coordinates
"""

from typing import Optional, Tuple

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from pydomcfg.utils import _smooth_MB06

from .zgr import Zgr


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

    def __call__(
        self,
        min_dep: float,
        max_dep: float,
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
        min_dep: float
            Minimum depth of bottom topography surface (>0) (m)
        max_dep: float
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

        self._min_dep = min_dep
        self._max_dep = max_dep
        self._hc = hc
        self._rmax = rmax
        self._stretch = stretch
        self._ln_e3_dep = ln_e3_dep

        # set stretching parameters after checking their consistency
        if self._stretch:
            self._check_stretch_par(psurf, pbott, alpha, efold, pbot2)
            self._psurf = psurf or 0.0
            self._pbott = pbott or 0.0
            self._alpha = alpha or 0.0
            self._efold = efold or 0.0
            self._pbot2 = pbot2 or 0.0

        bathy = self._bathy["Bathymetry"]

        # compute land-sea mask of the domain:
        # 0 = land
        # 1 = ocean
        self._lsm = xr.where(bathy > 0, 1, 0)

        # set maximum and minumum depths of model bathymetry
        bathy = np.minimum(bathy, self._max_dep)
        bathy = np.maximum(bathy, self._min_dep) * self._lsm

        # compute envelope bathymetry DataArray
        self._envlp = self._compute_env(bathy)

        # compute sigma-coordinates for z3 computation
        self._sigmas = self._compute_sigma(self._z)

        # compute z3 depths of zco vertical levels
        z3t, z3w = self._sco_z3

        # compute e3 scale factors
        e3t, e3w = self._compute_e3(z3t, z3w)

        # addind this only to not make darglint complying
        # ds = self._bathy.copy()
        # ds["hbatt"] = self._envlp
        return self._merge_z3_and_e3(z3t, z3w, e3t, e3w)

    def _check_stretch_par(self, psurf, pbott, alpha, efold, pbot2):
        """
        Check consistency of stretching parameters
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
            raise ValueError(
                f"{srf} and {bot} MUST be set when using {self._stretch} stretching."
            )

        if self._stretch == "sf12":
            if not (alpha and efold and pbot2):
                raise ValueError(
                    "rn_alpha, rn_efold and rn_zb_b MUST be set when using \
                     sf12 stretching."
                )

    def _compute_env(self, depth: DataArray) -> DataArray:
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
        depth: DataArray
            xarray DataArray of the 2D bottom topography
            it MUST have only two dimensions
        Returns
        -------
        DataArray
            xarray DataArray of the 2D envelope bathymetry
        """

        if self._rmax:

            lsm = self._lsm

            # set first land point adjacent to a wet cell to
            # min_dep as this needs to be included in smoothing
            cst_lsm = lsm.rolling({dim: 3 for dim in lsm.dims}, min_periods=2).sum()
            cst_lsm = cst_lsm.shift({dim: -1 for dim in lsm.dims})
            cst_lsm = (cst_lsm > 0) & (lsm == 0)
            zenv = depth.where(cst_lsm == 0, self._min_dep)

            zenv = _smooth_MB06(zenv, self._rmax)
            zenv = zenv.where(zenv > self._min_dep, self._min_dep)

        return zenv

    @property
    def _sco_z3(self) -> Tuple[DataArray, ...]:
        """Compute and return z3{t,w} for s-coordinates grids"""

        grids = ("T", "W")
        sigmas = self._sigmas
        scosrf = self._envlp * 0.0  # unperturbed free-surface

        both_z3 = []
        for grid, sigma in zip(grids, sigmas):

            if self._stretch:
                # Stretched sco grid
                su = -sigma
                ss = self._stretch_sco(-sigma)
                a1 = scosrf
                a2 = DataArray((0.0))  # TODO: Why can't use float here?
                a3 = self._envlp
                if self._stretch != "sf12":
                    a2 += self._hc
                    a3 -= self._hc
            else:
                # Uniform sco grid
                su = -sigma
                ss = DataArray((0.0))
                a1 = a3 = scosrf
                a2 = self._envlp

            z3 = self._compute_z3(su, ss, a1, a2, a3)

            both_z3 += [z3]

        return tuple(both_z3)

    def _stretch_sco(self, sigma: DataArray) -> DataArray:
        """
        Wrapping method for calling generalised analytical
        stretching function for terrain-following s-coordinates.

        Parameters
        ----------
        sigma: DataArray
            Uniform non-dimensional sigma-coordinate:
            MUST BE positive, i.e. 0 <= sigma <= 1

        Returns
        -------
        DataArray
            Stretched coordinate
        """
        if self._stretch == "sh94":
            ss = self._sh94(sigma)
        elif self._stretch == "md96":
            ss = self._md96(sigma)
        # elif self._stretch == "sf12":
        #    ss = self._sf12(sigma)

        return ss

    def _sh94(self, sigma: DataArray) -> DataArray:
        """
        Song and Haidvogel 1994 analytical stretching
        function for terrain-following s-coordinates.

        Reference:
            Song & Haidvogel, J. Comp. Phy., 115, 228-244, 1994.

        Parameters
        ----------
        sigma: DataArray
            Uniform non-dimensional sigma-coordinate:
            MUST BE positive, i.e. 0 <= sigma <= 1

        Returns
        -------
        DataArray
            Stretched coordinate
        """
        ca = self._psurf
        cb = self._pbott

        if ca == 0.0:
            ss = sigma
        else:
            ss = (1.0 - cb) * np.sinh(ca * sigma) / np.sinh(ca) + cb * (
                (np.tanh(ca * (sigma + 0.0)) - np.tanh(0.0 * ca))
                / (2.0 * np.tanh(0.0 * ca))
            )

        return ss

    def _md96(self, sigma: DataArray) -> DataArray:
        """
        Madec et al. 1996 analytical stretching
        function for terrain-following s-coordinates.

        Reference:
            pag 65 of NEMO Manual
            Madec, Lott, Delecluse and Crepon, 1996. JPO, 26, 1393-1408

        Parameters
        ----------
        sigma: DataArray
            Uniform non-dimensional sigma-coordinate:
            MUST BE positive, i.e. 0 <= sigma <= 1

        Returns
        -------
        DataArray
            Stretched coordinate
        """
        ca = self._psurf
        cb = self._pbott

        ss = (
            (np.tanh(ca * (sigma + cb)) - np.tanh(cb * ca))
            * (np.cosh(ca) + np.cosh(ca * (2.0e0 * cb - 1.0e0)))
            / (2.0 * np.sinh(ca))
        )

        return ss

    # def _sf12(self, sigma: DataArray) -> DataArray:
    #    """
    #    Siddorn and Furner 2012 analytical stretching
    #    function for terrain-following s-coordinates.
    #
    #    Reference:
    #        Siddorn & Furner, Oce. Mod. 66:1-13, 2013.

    #    Parameters
    #    ----------
    #    sigma: DataArray
    #        Uniform non-dimensional sigma-coordinate:
    #        MUST BE positive, i.e. 0 <= sigma <= 1
    #
    #    Returns
    #    -------
    #    DataArray
    #        Stretched coordinate
    #    """
