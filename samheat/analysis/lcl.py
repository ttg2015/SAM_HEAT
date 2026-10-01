# ---------------------------------------------------------------------------------------
# Original author: David M. Romps (v1.0 2017, v1.1 2021), with SAM-thermodynamics edits by Robert van der Drift (4/29/2023)
# Source repo: precip_extremes_RH_RCE (van der Drift & O'Gorman 2025, arXiv:2412.16306),
#   precip/scripts/lcl.py
# Vendored into SAM_HEAT on 2026-10-01. Changes: none (verbatim copy; Romps' citation header below is retained).
# ---------------------------------------------------------------------------------------
# Version 1.0 released by David Romps on September 12, 2017.
# Version 1.1 vectorized lcl.R, released on May 24, 2021.
# 
# When using this code, please cite:
# 
# @article{16lcl,
#   Title   = {Exact expression for the lifting condensation level},
#   Author  = {David M. Romps},
#   Journal = {Journal of the Atmospheric Sciences},
#   Year    = {2017},
#   Month   = dec,
#   Number  = {12},
#   Pages   = {3891--3900},
#   Volume  = {74}
# }
#
# This lcl function returns the height of the lifting condensation level
# (LCL) in meters.  The inputs are:
# - p in Pascals
# - T in Kelvins
# - Exactly one of rh, rhl, and rhs (dimensionless, from 0 to 1):
#    * The value of rh is interpreted to be the relative humidity with
#      respect to liquid water if T >= 273.15 K and with respect to ice if
#      T < 273.15 K. 
#    * The value of rhl is interpreted to be the relative humidity with
#      respect to liquid water
#    * The value of rhs is interpreted to be the relative humidity with
#      respect to ice
# - return_ldl is an optional logical flag.  If true, the lifting deposition
#   level (LDL) is returned instead of the LCL. 
# - return_min_lcl_ldl is an optional logical flag.  If true, the minimum of the
#   LCL and LDL is returned.

import math
import scipy.special
import numpy as np

def lcl(p,T,rh=None,rhl=None,rhs=None,return_ldl=False,return_min_lcl_ldl=False):

    # Parameters
    Ttrip = 273.16     # K
    ptrip = 611.65     # Pa
    E0v   = 2.3740e6   # J/kg
    E0s   = 0.3337e6   # J/kg
    ggr   = 9.81       # m/s^2
    rgasa = 287.04     # J/kg/K 
    rgasv = 461        # J/kg/K 
    cva   = 719        # J/kg/K
    cvv   = 1418       # J/kg/K 
    cvl   = 4119       # J/kg/K 
    cvs   = 1861       # J/kg/K 
    cpa   = cva + rgasa
    cpv   = cvv + rgasv

    # rvd mod (4/29/2023): replaced pvstar equations with SAM's thermodynamics
    # The saturation vapor pressure over liquid water
    def pvstarl(t):
        aw=[6.11239921, 0.443987641, 0.142986287*1e-1, 0.264847430*1e-3, 0.302950461*1e-5, 0.206739458*1e-7, 0.640689451*1e-10, -0.952447341*1e-13,-0.976195544*1e-15]
        aw0 = aw[0]
        aw1 = aw[1]
        aw2 = aw[2]
        aw3 = aw[3]
        aw4 = aw[4]
        aw5 = aw[5]
        aw6 = aw[6]
        aw7 = aw[7]
        aw8 = aw[8]

        dt = np.fmax(-80,t-273.16)

        return aw0 + dt*(aw1+dt*(aw2+dt*(aw3+dt*(aw4+dt*(aw5+dt*(aw6+dt*(aw7+aw8*dt))))))) 


    # The saturation vapor pressure over solid ice
    def pvstars(t):
        ai=[6.11147274, 0.503160820, 0.188439774*1e-1,0.420895665*1e-3, 0.615021634*1e-5,0.602588177*1e-7,0.385852041*1e-9, 0.146898966*1e-11, 0.252751365*1e-14]
        ai0 = ai[0]
        ai1 = ai[1]
        ai2 = ai[2]
        ai3 = ai[3]
        ai4 = ai[4]
        ai5 = ai[5]
        ai6 = ai[6]
        ai7 = ai[7]
        ai8 = ai[8]

        dt = (t>185)*(t-273.16) + (t<=185)*np.fmax(-100,t-273.16)

        return (t>185)*(ai0 + dt*(ai1+dt*(ai2+dt*(ai3+dt*(ai4+dt*(ai5+dt*(ai6+dt*(ai7+ai8*dt))))))))\
        + (t<=185)*(0.00763685 + dt*(0.000151069+dt*7.48215e-7))   

    # Calculate pv from rh, rhl, or rhs
    rh_counter = 0
    if rh is not None:
        rh_counter = rh_counter + 1
    if rhl is not None:
        rh_counter = rh_counter + 1
    if rhs is not None:
        rh_counter = rh_counter + 1
    if rh_counter != 1:
        print(rh_counter)
        # exit('Error in lcl: Exactly one of rh, rhl, and rhs must be specified')
    if rh is not None:
        # The variable rh is assumed to be 
        # with respect to liquid if T > Ttrip and 
        # with respect to solid if T < Ttrip
    
        # rvd mod (4/29/2023): always wrt liquid for my simulations.
        # if T > Ttrip:
        pv = rh * pvstarl(T)
        # else:
            # pv = rh * pvstars(T)
        rhl = pv / pvstarl(T)
        rhs = pv / pvstars(T)
    elif rhl is not None:
        pv = rhl * pvstarl(T)
        rhs = pv / pvstars(T)
        if T > Ttrip:
            rh = rhl
        else:
            rh = rhs
    elif rhs is not None:
        pv = rhs * pvstars(T)
        rhl = pv / pvstarl(T)
        if T > Ttrip:
            rh = rhl
        else:
            rh = rhs
    # if pv > p:
    #     return NA

    # Calculate lcl_liquid and lcl_solid
    qv = rgasa*pv / (rgasv*p + (rgasa-rgasv)*pv)
    rgasm = (1-qv)*rgasa + qv*rgasv
    cpm = (1-qv)*cpa + qv*cpv
    # rvd mod: never rh = 0.
    # if rh == 0:
    #     return cpm*T/ggr
    aL = -(cpv-cvl)/rgasv + cpm/rgasm
    bL = -(E0v-(cvv-cvl)*Ttrip)/(rgasv*T)
    cL = pv/pvstarl(T)*np.exp(-(E0v-(cvv-cvl)*Ttrip)/(rgasv*T))
    aS = -(cpv-cvs)/rgasv + cpm/rgasm
    bS = -(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T)
    cS = pv/pvstars(T)*np.exp(-(E0v+E0s-(cvv-cvs)*Ttrip)/(rgasv*T))
    lcl = cpm*T/ggr*( 1 - bL/(aL*scipy.special.lambertw(bL/aL*cL**(1/aL),-1).real) )
    ldl = cpm*T/ggr*( 1 - bS/(aS*scipy.special.lambertw(bS/aS*cS**(1/aS),-1).real) )

    # Return either lcl or ldl
    if return_ldl:
        return ldl
    elif return_min_lcl_ldl:
        return min(lcl,ldl)
    else:
        return lcl
