"""
Class to generate NEMO v4.0 standard geopotential z-coordinates
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from xarray import DataArray, Dataset

from pydomcfg.utils import _are_nemo_none, _is_nemo_none

from .zgr import Zgr


class Zco(Zgr):
    """
    Class to generate geopotential z-coordinates dataset objects.
    """

    # --------------------------------------------------------------------------
    def __call__(
        self,
        ppdzmin: float,
        pphmax: float,
        ppkth: float = 0,
        ppacr: int = 0,
        ppsur: Optional[float] = None,
        ppa0: Optional[float] = None,
        ppa1: Optional[float] = None,
        ppa2: Optional[float] = None,
        ppkth2: Optional[float] = None,
        ppacr2: Optional[float] = None,
        ldbletanh: Optional[bool] = None,
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
        ppkth: float, default = 0
            Model level at which approximately max stretching occurs.
            Nondimensional > 0, usually of order 1/2 or 2/3 of ``jpk``.
            (0 means no stretching, uniform grid)
        ppacr: int, default = 0
            Stretching factor, nondimensional > 0. The larger ``ppacr``,
            the smaller the stretching. Values from 3 to 10 are usual.
            (0 means no stretching, uniform grid)
        ppsur, ppa0, ppa1: float, optional
            Coefficients controlling distibution of vertical levels
            (None: compute from ``ppdzmin``, ``pphmax``, ``ppkth`` and ``ppacr``)
        ppa2, ppkth2, ppacr2: float, optional
            Double tanh stretching function parameters.
            (None: Double tanh OFF)
        ldbletanh: bool, optional
            Logical flag to switch ON/OFF the double tanh stretching function.
            This flag is only needed for compatibility with NEMO DOMAINcfg tools.
            Just set ``ppa2``, ``ppkth2``, and ``ppacr2`` to switch ON
            the double tanh stretching function.
        ln_e3_dep: bool, default = True
            Logical flag to comp. e3 as fin. diff. (True) oranalyt. (False)
            (default = True)

        Returns
        -------
        Dataset
           Describing the 3D geometry of the model

        Notes
        -----
        * Model levels' depths ``z3{T,W}`` are defined from analytical function.

        * Model vertical scale factors ``e3{T,W}`` (i.e., grid cell thickness)
          can be computed as:

            - analytical derivative of depth function (``ln_e3_dep=False``);
              for backward compatibility with v3.6.

            - discrete derivative (central-difference) of levels' depth
              (``ln_e3_dep=True``). The only possibility from v4.0.

        * See [1]_ and equations E.1-4 in the NEMO4 v4 ocean engine manual [2]_.

        See Also
        --------
        NEMOv4.0: ``domzgr/zgr_z`` subroutine

        References
        ----------
        .. [1] Marti, O., Madec, G., and Delecluse, P. (1992), Comment [on “Net
           diffusivity in ocean general circulation models with nonuniform grids”
           by F. L. Yin and I. Y. Fung], J. Geophys. Res., 97( C8), 12763– 12766,
           http://doi.org/10.1029/92JC00306.

        .. [2] Madec Gurvan, Romain Bourdallé-Badie, Jérôme Chanut, Emanuela Clementi,
           Andrew Coward, Christian Ethé, … Guillaume Samson. (2019, October 24).
           NEMO ocean engine (Version v4.0).
           Notes Du Pôle De Modélisation De L'institut Pierre-simon Laplace (IPSL).
           Zenodo. http://doi.org/10.5281/zenodo.3878122
        """

        # Init
        self._ppdzmin = ppdzmin
        self._pphmax = pphmax
        self._ppkth = ppkth
        self._ppacr = ppacr
        self._ln_e3_dep = ln_e3_dep
        self._is_uniform = not (ppkth or ppacr)

        # Set double tanh flag and coefficients (dummy floats when double tanh is OFF)
        pp2_in = (ppa2, ppkth2, ppacr2)
        self._ldbletanh, pp2_out = self._get_ldbletanh_and_pp2(ldbletanh, pp2_in)
        self._ppa2, self._ppkth2, self._ppacr2 = pp2_out

        # Initialize stretching coefficients (dummy floats for uniform case)
        self._ppsur, self._ppa0, self._ppa1 = self._compute_pp(ppsur, ppa0, ppa1)

        # Compute and store sigma
        self._sigmas = self._compute_sigma(self._z)

        # Compute z3 depths of zco vertical levels
        z3t, z3w = self._zco_z3

        # Compute e3 scale factors (cell thicknesses)
        e3t, e3w = self._compute_e3(z3t, z3w) if self._ln_e3_dep else self._analyt_e3

        return self._merge_z3_and_e3(z3t, z3w, e3t, e3w)

    # --------------------------------------------------------------------------
    def _compute_pp(
        self,
        ppsur: Optional[float],
        ppa0: Optional[float],
        ppa1: Optional[float],
    ) -> Tuple[float, ...]:
        """
        Compute and return the coefficients for zco grid.
        Only None or 999999. are replaced.
        Return 0s for uniform case.
        """

        # Uniform grid, return dummy zeros
        if self._is_uniform:
            if not all(_are_nemo_none((ppsur, ppa0, ppa1))):
                warnings.warn(
                    "Uniform grid case (no stretching):"
                    " ppsur, ppa0 and ppa1 are ignored when ppacr == ppkth == 0"
                )

            return (0, 0, 0)

        # Strecthing parameters
        aa = self._ppdzmin - self._pphmax / (self._jpk - 1)
        bb = np.tanh((1 - self._ppkth) / self._ppacr)
        cc = self._ppacr / (self._jpk - 1)
        dd = np.log(np.cosh((self._jpk - self._ppkth) / self._ppacr))
        ee = np.log(np.cosh((1.0 - self._ppkth) / self._ppacr))

        # Substitute only if is None or 999999
        ppa1_out = (aa / (bb - cc * (dd - ee))) if _is_nemo_none(ppa1) else ppa1
        ppa0_out = (self._ppdzmin - ppa1_out * bb) if _is_nemo_none(ppa0) else ppa0
        ppsur_out = (
            -(ppa0_out + ppa1_out * self._ppacr * ee) if _is_nemo_none(ppsur) else ppsur
        )

        return (ppsur_out, ppa0_out, ppa1_out)

    # --------------------------------------------------------------------------
    def _stretch_zco(self, sigma: DataArray, ldbletanh: bool = False) -> DataArray:
        """
        Provide the generalised analytical stretching function for NEMO z-coordinates.

        Parameters
        ----------
        sigma: DataArray
            Uniform non-dimensional sigma-coordinate:
            MUST BE positive, i.e. 0 <= sigma <= 1
        ldbletanh: bool
            True only if used to compute the double tanh stretching

        Returns
        -------
        DataArray
            Stretched coordinate
        """

        kk = sigma * (self._jpk - 1.0) + 1.0

        if ldbletanh:
            # Double tanh
            kth = self._ppkth2
            acr = self._ppacr2
        else:
            # Single tanh
            kth = self._ppkth
            acr = self._ppacr

        return np.log(np.cosh((kk - kth) / acr))

    # --------------------------------------------------------------------------
    @property
    def _zco_z3(self) -> Tuple[DataArray, ...]:
        """Compute and return z3{t,w} for z-coordinates grids"""

        grids = ("T", "W")
        sigmas = self._sigmas
        sigmas_p1 = self._compute_sigma(self._z + 1)

        both_z3 = []
        for grid, sigma, sigma_p1 in zip(grids, sigmas, sigmas_p1):

            if self._is_uniform:
                # Uniform zco grid
                su = -sigma
                s1 = DataArray((0.0))
                a1 = a3 = 0.0
                a2 = self._pphmax
            else:
                # Stretched zco grid
                su = -sigma_p1
                s1 = self._stretch_zco(-sigma)
                a1 = self._ppsur
                a2 = self._ppa0 * (self._jpk - 1.0)
                a3 = self._ppa1 * self._ppacr
            z3 = self._compute_z3(su, s1, a1, a2, a3)

            if self._ldbletanh:
                # Add double tanh term
                ss2 = self._stretch_zco(-sigma, self._ldbletanh)
                a4 = self._ppa2 * self._ppacr2
                z3 += ss2 * a4

            if grid == "W":
                # Force first w-level to be exactly at zero
                z3[{"z": 0}] = 0.0

            both_z3 += [z3]

        return tuple(both_z3)

    # --------------------------------------------------------------------------
    @property
    def _analyt_e3(self) -> Tuple[DataArray, ...]:
        """
        Backward compatibility with v3.6:
        Return e3{t,w} as analytical derivative of depth function z3{t,w}.
        """

        if self._is_uniform:
            # Uniform: Return 0d DataArrays
            e3 = DataArray((self._pphmax / (self._jpk - 1.0)))
            return tuple([e3, e3])

        both_e3 = []
        for sigma in self._sigmas:
            # Stretched zco grid
            a0 = self._ppa0
            a1 = self._ppa1
            kk = -sigma * (self._jpk - 1.0) + 1.0
            tanh1 = np.tanh((kk - self._ppkth) / self._ppacr)
            e3 = a0 + a1 * tanh1

            if self._ldbletanh:
                # Add double tanh term
                a2 = self._ppa2
                tanh2 = np.tanh((kk - self._ppkth2) / self._ppacr2)
                e3 += a2 * tanh2

            both_e3 += [e3]

        return tuple(both_e3)

    def _get_ldbletanh_and_pp2(
        self, ldbletanh: Optional[bool], pp2: Tuple[Optional[float], ...]
    ) -> Tuple[bool, Tuple[float, ...]]:
        """
        If ldbletanh is None, its bool value is inferred from pp2.
        Return pp2=(0, 0, 0) when double tanh is switched off.
        """

        pp_are_none = tuple(pp is None for pp in pp2)
        prefix_msg = "ppa2, ppkth2 and ppacr2"
        ldbletanh_out = ldbletanh if (ldbletanh is not None) else not any(pp_are_none)

        # Warnings: Ignore double tanh coeffiecients
        if ldbletanh_out and self._is_uniform:
            # Uniform and double tanh
            warning_msg = (
                "Uniform grid case (no stretching):"
                f" {prefix_msg} are ignored when ppacr == ppkth == 0"
            )
        elif ldbletanh is False and not all(pp_are_none):
            # ldbletanh False and double tanh coefficients specified
            warning_msg = f"{prefix_msg} are ignored when ldbletanh is False"
        else:
            # All good
            warning_msg = ""

        if warning_msg:
            # Warn and return dummy values
            warnings.warn(warning_msg)
            return (False, (0, 0, 0))

        # Errors: pp have inconsistent types
        if ldbletanh is True and any(pp_are_none):
            raise ValueError(f"{prefix_msg} MUST be all float when ldbletanh is True")
        if ldbletanh is None and (any(pp_are_none) and not all(pp_are_none)):
            raise ValueError(f"{prefix_msg} MUST be all None or all float")

        return (ldbletanh_out, tuple(pp or 0 for pp in pp2))
