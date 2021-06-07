#!/usr/bin/env python

"""
Class to generate NEMO v4.0 standard geopotential z-coordinates
"""


import numpy as np
from xarray import Dataset

from .zgr import Zgr


class Zco(Zgr):

    # In future we may want to get rid of this
    # and use a more pythonic way. We could do
    # it when  we read the namelist.
    pp_to_be_computed = 999999.0

    """
    Class to generate geopotential z-coordinates dataset objects.

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
       *) NEMO v4.0 domzgr/zgr_z subroutine
       *) Marti, Madec & Delecluse, 1992, JGR, 97, No8, 12,763-12,766.
    """

    # --------------------------------------------------------------------------
    def __call__(
        self,
        ppdzmin,
        pphmax,
        ppkth: float = 0.0,
        ppacr: int = 0,
        ppsur: float = pp_to_be_computed,
        ppa0: float = pp_to_be_computed,
        ppa1: float = pp_to_be_computed,
        ldbletanh: bool = False,
        ppa2: float = 0.0,
        ppkth2: float = 0.0,
        ppacr2: float = 0.0,
    ) -> Dataset:
        """
        Generate NEMO geopotential z-coordinates model levels.

        Parameters
        ----------
        ppdzmin: float
            Minimum thickness for the top layer (units: m)
        pphmax: float
            Depth of the last W-level (total depth of the ocean basin) (units: m)
        ppacr: int
            Stretching factor, nondimensional > 0. The larger ppacr,
            the smaller the stretching. Values from 3 to 10 are usual.
            (default = 0., no stretching, uniform grid)
        ppkth: float
            Model level at which approximately max stretching occurs.
            Nondimensional > 0, usually of order 1/2 or 2/3 of jpk.
            (default = 0., i.e. no stretching, uniform grid)
        ppsur: float
            Coeff. controlling distibution of vertical levels
            (default = pp_to_be_computed, i.e. computed from
            ppdzmin, pphmax, ppkth and ppacr)
        ppa0: float
            Coeff. controlling distibution of vertical levels
            (default = pp_to_be_computed, i.e. computed from
            ppdzmin, pphmax, ppkth and ppacr)
        ppa1: float
            Coeff. controlling distibution of vertical levels
            (default = pp_to_be_computed, i.e. computed from
            ppdzmin, pphmax, ppkth and ppacr)
        ldbletanh: bool
            Logical flag to use or not double tanh stretching function (default = False)
        ppa2: float
            Double tanh stretching function parameter
        ppkth2: float
            Double tanh stretching function parameter
        ppacr2: float
            Double tanh stretching function parameter

        Returns
        -------
        Dataset
           Describing the 3D geometry of the model

        """
        self._ppdzmin = ppdzmin
        self._pphmax = pphmax
        self._ppkth = ppkth
        self._ppacr = ppacr
        self._ppsur = ppsur
        self._ppa0 = ppa0
        self._ppa1 = ppa1
        self._ldbletanh = ldbletanh
        self._ppa2 = ppa2
        self._ppkth2 = ppkth2
        self._ppacr2 = ppacr2

        # computing coeff. if needed
        if self._ppkth * self._ppacr > 0.0:
            self._compute_pp()

        # compute z3 depths of vertical levels
        if self._ppkth * self._ppacr == 0.0:
            # uniform zco grid
            suT = -self.sigma("T")
            suW = -self.sigma("W")
            s1T = s1W = s2T = s2W = 0.0
            a1 = a3 = a4 = 0.0
            a2 = self._pphmax
        else:
            # stretched zco grid
            suT = -self.sigma("T") * (self._jpk - 1) + 1
            suW = -self.sigma("W") * (self._jpk - 1) + 1
            s1T = self._stretch_zco1(suT)
            s1W = self._stretch_zco1(suW)
            a1 = self._ppsur
            a2 = self._ppa0
            a3 = self._ppa1 * self._ppacr
            if self._ldbletanh:
                s2T = self._stretch_zco2(suT)
                s2W = self._stretch_zco2(suW)
                a4 = self._ppa2 * self._ppacr2
            else:
                s2T = s2W = a4 = 0.0

        ds = Dataset()
        ds["z3T"] = self.compute_z3(suT, s1T, a1, a2, a3, s2T, a4)
        ds["z3W"] = self.compute_z3(suW, s1W, a1, a2, a3, s2W, a4)
        ds["z3W"] = ds["z3W"].where(ds["z"] > 0, 0)

        # compute e3 scale factors
        ds = self.compute_e3(ds).broadcast_like(self._bathy)
        ds = ds.set_coords(ds.variables)

        return self._bathy.merge(ds)

    # --------------------------------------------------------------------------
    def _compute_pp(self):
        """
        Compute the coefficients for zco grid if requested.

        Raises
        ------
        ValueError
            If ldbletanh = True and parametrs are equal to 0.

        """
        if (
            self._ppsur == self.pp_to_be_computed
            or self._ppa0 == self.pp_to_be_computed
            or self._ppa1 == self.pp_to_be_computed
        ):

            aa = self._ppdzmin - self._pphmax / float(self._jpk - 1)
            bb = np.tanh((1 - self._ppkth) / self._ppacr)
            cc = self._ppacr / float(self._jpk - 1)
            dd = np.log(np.cosh((self._jpk - self._ppkth) / self._ppacr)) - np.log(
                np.cosh((1 - self._ppkth) / self._ppacr)
            )

            self._ppa1 = aa / (bb - cc * dd)
            self._ppa0 = self._ppdzmin - self._ppa1 * bb
            self._ppsur = -self._ppa0 - self._ppa1 * self._ppacr * np.log(
                np.cosh((1 - self._ppkth) / self._ppacr)
            )

            if self._ldbletanh and self._ppa2 * self._ppkth2 * self._ppacr2 == 0.0:
                raise ValueError(
                    "ppa2, ppkth2 and ppacr2 must > 0. " "when ldbletanh = True"
                )

    # --------------------------------------------------------------------------
    def _stretch_zco1(self, k: float):
        """
        Provide the standard analytical stretching function for NEMO z-coordinates.

        Parameters
        ----------
        k: float
            Vertical index

        Returns
        -------
        ss: float
            Stretched coordinate
        """

        ss = np.log(np.cosh((k - self._ppkth) / self._ppacr))
        return ss

    # --------------------------------------------------------------------------
    def _stretch_zco2(self, k: float):
        """
        Provide the double tanh analytical stretching function for NEMO z-coordinates.

        Parameters
        ----------
        k: float
            Vertical index
        Returns
        -------
        ss: float
            Stretched coordinate
        """

        ss = np.log(np.cosh((k - self._ppkth2) / self._ppacr2))
        return ss
