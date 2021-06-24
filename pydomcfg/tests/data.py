"""
Test data
"""
import numpy as np

# Results to replicate:
# ORCA2 zco model levels depth and vertical
# scale factors as computed by NEMO v3.6.
# -------------------------------------------------------------
# gdept_1d   gdepw_1d   e3t_1d     e3w_1d
ORCA2_VGRID = np.array(
    [
        [4.9999, 0.0000, 10.0000, 9.9998],
        [15.0003, 10.0000, 10.0008, 10.0003],
        [25.0018, 20.0008, 10.0023, 10.0014],
        [35.0054, 30.0032, 10.0053, 10.0036],
        [45.0133, 40.0086, 10.0111, 10.0077],
        [55.0295, 50.0200, 10.0225, 10.0159],
        [65.0618, 60.0429, 10.0446, 10.0317],
        [75.1255, 70.0883, 10.0876, 10.0625],
        [85.2504, 80.1776, 10.1714, 10.1226],
        [95.4943, 90.3521, 10.3344, 10.2394],
        [105.9699, 100.6928, 10.6518, 10.4670],
        [116.8962, 111.3567, 11.2687, 10.9095],
        [128.6979, 122.6488, 12.4657, 11.7691],
        [142.1952, 135.1597, 14.7807, 13.4347],
        [158.9606, 150.0268, 19.2271, 16.6467],
        [181.9628, 169.4160, 27.6583, 22.7828],
        [216.6479, 197.3678, 43.2610, 34.2988],
        [272.4767, 241.1259, 70.8772, 55.2086],
        [364.3030, 312.7447, 116.1088, 90.9899],
        [511.5348, 429.7234, 181.5485, 146.4270],
        [732.2009, 611.8891, 261.0346, 220.3500],
        [1033.2173, 872.8738, 339.3937, 301.4219],
        [1405.6975, 1211.5880, 402.2568, 373.3136],
        [1830.8850, 1612.9757, 444.8663, 426.0031],
        [2289.7679, 2057.1314, 470.5516, 459.4697],
        [2768.2423, 2527.2169, 484.9545, 478.8342],
        [3257.4789, 3011.8994, 492.7049, 489.4391],
        [3752.4422, 3504.4551, 496.7832, 495.0725],
        [4250.4012, 4001.1590, 498.9040, 498.0165],
        [4749.9133, 4500.0215, 500.0000, 499.5419],
        [5250.2266, 5000.0000, 500.5646, 500.3288],
    ]
)

# See pag 62 of v3.6 manual for the input parameters
ORCA2_NAMELIST = """
!-----------------------------------------------------------------------
&namcfg        !   parameters of the configuration
!-----------------------------------------------------------------------
   !
   ln_e3_dep   = .false.   ! =T : e3=dk[depth] in discret sens.
   !                       !      ===>>> will become the only possibility in v4.0
   !                       ! =F : e3 analytical derivative of depth function
   !                       !      only there for backward compatibility test with v3.6
   !                       !
   cp_cfg      =  "orca"   !  name of the configuration
   jp_cfg      =       2   !  resolution of the configuration
   jpidta      =     180   !  1st lateral dimension ( >= jpi )
   jpjdta      =     148   !  2nd    "         "    ( >= jpj )
   jpkdta      =      31   !  number of levels      ( >= jpk )
   Ni0glo      =     180   !  1st dimension of global domain --> i =jpidta
   Nj0glo      =     148   !  2nd    -                  -    --> j  =jpjdta
   jpkglo      =      31
   jperio      =       4   !  lateral cond. type (between 0 and 6)
   ln_use_jattr = .false.  !  use (T) the file attribute: open_ocean_jstart, if present
                           !  in netcdf input files, as the start j-row for reading
   ln_domclo = .false.     ! computation of closed sea masks (see namclo)
/
!-----------------------------------------------------------------------
&namdom        !
!-----------------------------------------------------------------------
   jphgr_msh   =       0               !  type of horizontal mesh
   ppglam0     =  999999.0             !  longitude of first raw and column T-point (jphgr_msh = 1)
   ppgphi0     =  999999.0             ! latitude  of first raw and column T-point (jphgr_msh = 1)
   ppe1_deg    =  999999.0             !  zonal      grid-spacing (degrees)
   ppe2_deg    =  999999.0             !  meridional grid-spacing (degrees)
   ppe1_m      =  999999.0             !  zonal      grid-spacing (degrees)
   ppe2_m      =  999999.0             !  meridional grid-spacing (degrees)
   ppsur       =   -4762.96143546300   !  ORCA r4, r2 and r05 coefficients
   ppa0        =     255.58049070440   ! (default coefficients)
   ppa1        =     245.58132232490   !
   ppkth       =      21.43336197938   !
   ppacr       =       3.0             !
   ppdzmin     =  999999.0             !  Minimum vertical spacing
   pphmax      =  999999.0             !  Maximum depth
   ldbletanh   =  .FALSE.              !  Use/do not use double tanf function for vertical coordinates
   ppa2        =  999999.0             !  Double tanh function parameters
   ppkth2      =  999999.0             !
   ppacr2      =  999999.0             !
/
!-----------------------------------------------------------------------
&namzgr        !   vertical coordinate                                  (default: NO selection)
!-----------------------------------------------------------------------
   ln_zco      = .true.    !  z-coordinate - full    steps
   ln_zps      = .false.   !  z-coordinate - partial steps
   ln_sco      = .false.   !  s- or hybrid z-s-coordinate
   ln_isfcav   = .false.   !  ice shelf cavity
   ln_linssh   = .false.   !  linear free surface
/
"""
