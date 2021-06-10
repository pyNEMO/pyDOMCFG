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
        ln_e3_dep: bool = True,
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
            Logical flag to use or not double tanh stretching function
            (default = False)
        ppa2: float
            Double tanh stretching function parameter
        ppkth2: float
            Double tanh stretching function parameter
        ppacr2: float
            Double tanh stretching function parameter
        ln_e3_dep: bool
            Logical flag to comp. e3 as fin. diff. (True) or
            analyt. (False) (default = True)

        Returns
        -------
        Dataset
           Describing the 3D geometry of the model

        Raises
        -------
        ValueError
            If ldbletanh = True and parametrs are equal to 0.

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
        self._is_uniform = not ppkth or not ppacr
        self._ln_e3_dep = ln_e3_dep

        # Checking consistency of input parameters
        if self._ldbletanh:
            if self._ppa2 * self._ppkth2 * self._ppacr2 == 0.0:
                raise ValueError(
                    "ppa2, ppkth2 and ppacr2 MUST be > 0. " "when ldbletanh = True"
                )

        ds = self._init_ds()

        # computing coeff. if needed
        if not self._is_uniform:
            self._ppsur, self._ppa0, self._ppa1 = self._compute_pp()

        # compute sigma-coordinates for z3 computation
        kindx = ds["z"]
        sigma, sigma_p1 = (self._compute_sigma(kk) for kk in (kindx, kindx + 1.0))
        self._sigT, self._sigW = sigma
        self._sigTp1, self._sigWp1 = sigma_p1

        # compute z3 depths of zco vertical levels
        dsz = self._zco_z3(ds)

        # compute e3 scale factors
        dse = self._compute_e3(dsz) if self._ln_e3_dep else self._analyt_e3(dsz)

        return dse

    # --------------------------------------------------------------------------
    def _compute_pp(self) -> tuple:
        """
        Compute the coefficients for zco grid if needed.

        Returns
        -------
        tuple
            (ppsur, ppa0, ppa1)

        """
        pp_in = (self._ppsur, self._ppa0, self._ppa1)
        if not pp_in.count(self.pp_to_be_computed):
            return pp_in

        aa = self._ppdzmin - self._pphmax / float(self._jpk - 1)
        bb = np.tanh((1 - self._ppkth) / self._ppacr)
        cc = self._ppacr / float(self._jpk - 1)
        dd = np.log(np.cosh((self._jpk - self._ppkth) / self._ppacr))
        ee = np.log(np.cosh((1.0 - self._ppkth) / self._ppacr))

        ppa1 = aa / (bb - cc * (dd - ee))
        ppa0 = self._ppdzmin - self._ppa1 * bb
        ppsur = -(self._ppa0 + self._ppa1 * self._ppacr * ee)

        return (ppsur, ppa0, ppa1)

    # --------------------------------------------------------------------------
    def _stretch_zco(self, sigma: float, ldbletanh: bool = False):
        """
        Provide the generalised analytical stretching function for NEMO z-coordinates.

        Parameters
        ----------
        sigma: float
            Uniform non-dimensional sigma-coordinate:
            MUST BE positive, i.e. 0 <= sigma <= 1
        ldbletanh: bool
            True only if used to compute the double tanh stretching

        Returns
        -------
        ss: float
            Stretched coordinate
        """

        kk = sigma * (self._jpk - 1.0) + 1.0

        if not ldbletanh:
            kth = self._ppkth
            acr = float(self._ppacr)
        else:
            kth = self._ppkth2
            acr = float(self._ppacr2)

        ss = np.log(np.cosh((kk - kth) / acr))
        return ss

    # --------------------------------------------------------------------------
    def _zco_z3(self, ds: Dataset):
        """
        Compute z3T/W for z-coordinates grids

        Parameters
        ----------
        ds: Dataset
            xarray dataset with empty ``z3{T,W}``
        Returns
        -------
        ds: Dataset
            xarray dataset with ``z3{T,W}`` correctly computed
        """

        sigmas = [self._sigT, self._sigW]
        sigmas_p1 = [self._sigTp1, self._sigWp1]

        for varname, sig, sig_p1 in zip(["z3T", "z3W"], sigmas, sigmas_p1):
            for k in range(self._jpk):
                if self._is_uniform:
                    # uniform zco grid
                    su = -sig[{"z": k}]
                    s1 = s2 = 0.0
                    a1 = a3 = a4 = 0.0
                    a2 = self._pphmax
                else:
                    # stretched zco grid
                    su = -sig_p1[{"z": k}]
                    s1 = self._stretch_zco(-sig[{"z": k}])
                    a1 = self._ppsur
                    a2 = self._ppa0 * (self._jpk - 1)
                    a3 = self._ppa1 * self._ppacr
                    # double tahh
                    if self._ldbletanh:
                        s2 = self._stretch_zco(-sig[{"z": k}], self._ldbletanh)
                        a4 = self._ppa2 * self._ppacr2
                    else:
                        s2 = a4 = 0.0
                ds[varname][{"z": k}] = self._compute_z3(su, s1, a1, a2, a3, s2, a4)

        # force first w-level to be exactly at zero
        ds["z3W"][{"z": 0}] = 0.0

        return ds

    # --------------------------------------------------------------------------
    def _analyt_e3(self, ds: Dataset):
        """
        Provide e3T/W as analytical derivative of depth function
        for backward compatibility with v3.6.

        Parameters
        ----------
        ds: Dataset
            xarray dataset with empty ``e3{T,W}`` and ``z3{T,W}``
            correctly computed
        Returns
        -------
        ds: Dataset
            xarray dataset with ``e3{T,W}`` correctly computed
        """

        sigmas = [self._sigT, self._sigW]

        for varname, sig in zip(["e3T", "e3W"], sigmas):
            if self._is_uniform:
                # uniform zco grid
                ds[varname][{"z": slice(self._jpk)}] = self._pphmax / (self._jpk - 1.0)
            else:
                # stretched zco grid
                a0 = self._ppa0
                a1 = self._ppa1
                # now faster to use loop, to be optimised
                # using xarray features in the future
                for k in range(self._jpk):
                    kk = -sig[{"z": k}] * (self._jpk - 1.0) + 1.0
                    tanh1 = np.tanh((kk - self._ppkth) / self._ppacr)
                    if self._ldbletanh:
                        a2 = self._ppa2
                        tanh2 = np.tanh((kk - self._ppkth2) / self._ppacr2)
                    else:
                        a2 = tanh2 = 0.0

                    ds[varname][{"z": k}] = a0 + a1 * tanh1 + a2 * tanh2

        return ds
