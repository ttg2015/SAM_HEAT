# ---------------------------------------------------------------------------------------
# Original author: Robert van der Drift (December 2024)
# Source repo: precip_extremes_RH_RCE (van der Drift & O'Gorman 2025, arXiv:2412.16306),
#   precip/scripts/thermo_SAM.py
# Vendored into SAM_HEAT on 2026-10-01. Changes: none (verbatim copy; note dtq_dz calls tMALR, which is not defined upstream and is unused here).
# ---------------------------------------------------------------------------------------
# Python versions of SAM's thermodynamic functions
# Additionally, some functions for calculating the
# (pseudoadiabatic) moist adiabatic lapse rate and
# related quantities are included, consistent with
# SAM's assumptions. The condensation integral uses
# the function "dq_dz(T,p)," for example.
#
# Robert van der Drift, December 12th 2024
#

import numpy as np

# Constants:
cpd   = 1004 # J/kg/K (heat capacity at const. p of dry air)
lcond = 2.5104*1e6 # J/kg (Latent Heat of Vaporization)
lfus  = 0.3336*1e6 # J/kg (Latent Heat of Fusion)
lsub  = 2.8440*1e6 # J/kg (Latent Heat of Sublimation)
g     = 9.81 # m/s^2 (gravity)
Rv    = 461 # J/kg/K (Water Vapor Gas Const)
Rd    = 287.05 # J/kg/K (Dry Gas Const)kappa = lambda ql,T: R/(cpd + ())
kappa = Rd/cpd
p0    = 1000
epsi  = 0.61

##############
# PARTITIONS #
##############

def omega_n(t):
    """
        Partition function for Liquid vs Ice Non-Precipitating Condensates
    """
    
    t = np.array(t)
    t00n = 253.16
    t0n = 273.16
    om = np.fmax(0,np.fmin(1,(t-t00n)/(t0n-t00n)))
    return om

def omega_p(t):
    """
        Partition function for Liquid vs Ice Precipitating condensates.
    """
    
    t = np.array(t)
    t00p = 268.16
    t0p = 283.16
    om = np.fmax(0,np.fmin(1,(t-t00p)/(t0p-t00p)))
    return om

def omega_g(t):
    """
        Partition function for Graupel vs Snow.
    """
    
    t = np.array(t)
    t00g = 223.16
    t0g = 283.16 
    om = np.fmax(0,np.fmin(1,(t-t00g)/(t0g-t00g)))
    return om


def dom_dT(t):
    """
        Derivative of the omega_n partition function.
    """
    t = np.array(t)
    t00n = 253.16
    t0n = 273.16
    dom = (1/(t0n-t00n))*(t < t0n)*(t > t00n)
    return dom


##############
# SATURATION #
##############

def esatw(t):
    """
        Saturation vapor pressure over water.
    """
    
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

def qsatw(t,p):
    """
        Saturation mixing ratio over water
    """
    
    esat = esatw(t)
    qs = 0.622 * esat/np.fmax(esat,p - esat)
    return qs*1e3

def esati(t):
    """
        Saturation vapor pressure over ice
    """
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
    
    return  (t>273.15)*esatw(t) + (t<=273.15)*(t>185)*(ai0 + dt*(ai1+dt*(ai2+dt*(ai3+dt*(ai4+dt*(ai5+dt*(ai6+dt*(ai7+ai8*dt))))))))\
    + (t<=185)*(0.00763685 + dt*(0.000151069+dt*7.48215e-7))   


def qsati(t,p): 
    """
        Saturation mixing ratio over ice.
    """
    esat = esati(t)
    qs = (t<=273.15)*(0.622 * esat/np.fmax(esat,p-esat))

    return qs*1e3


def qsat(t,p):
    """
        Saturation mixing ratio, as partitioned between solid and liquid by omega_n.
    """
    qsatws = qsatw(t,p)
    qsatis = qsati(t,p)
    
    om = omega_n(t)
    qs = qsatws*om + qsatis*(1-om)
    return qs

############
# SAT GRAD #
############

def dtesatw(t):
    aw=[0.443956472, 0.285976452e-1, 0.794747212e-3, 0.121167162e-4, 0.103167413e-6, 0.385208005e-9, -0.604119582e-12, -0.792933209e-14, -0.599634321e-17]
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

def dtqsatw(t,p):
    return 0.622*dtesatw(t)*1e3/p

def dtesati(t):
    ai=[0.503223089, 0.377174432e-1, 0.126710138e-2, 0.249065913e-4, 0.312668753e-6, 0.255653718e-8, 0.132073448e-10, 0.390204672e-13, 0.497275778e-16]
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
    
    return  (t>273.15)*dtesatw(t) + (t<=273.15)*(t>185)*(ai0 + dt*(ai1+dt*(ai2+dt*(ai3+dt*(ai4+dt*(ai5+dt*(ai6+dt*(ai7+ai8*dt))))))))\
    + (t<=185)*(0.00763685 + dt*(0.000151069+dt*7.48215e-7))

def dtqsati(t,p):
    return 0.622*dtesati(t)*1e3/p

def dtqsat(t,p):
    """
        Saturation mixing ratio, as partitioned between solid and liquid by omega_n.
    """
    dtqsatws = dtqsatw(t,p)
    dtqsatis = dtqsati(t,p)
    
    qsatws = qsatw(t,p)
    qsatis = qsati(t,p)
    
    om = omega_n(t)
    dom = dom_dT(t)
    dtqs = (dtqsatws-dtqsatis)*om + (qsatws-qsatis)*dom + dtqsatis
    return dtqs

def dpqsat(t,p):
    qs = qsat(t,p)
    return -qs/(Rd*t)

##############
# LAPSE RATE #
##############

# These functions do not appear in SAM, but they provide (moist adiabatic) lapse rates that
# use SAM's partition functions and other thermodynamic assumptions

def claus_clap(T,qs,l):
    return l*qs/(Rv*T*T)


def MALR(T,p):
    """
        Calculate the moist adiabatic lapse rate
    """

    qsw = qsatw(T,p)/1e3
    qsi = qsati(T,p)/1e3
    
    # Determine partition of non-precipitating phase:
    om  = omega_n(T)
    dom = dom_dT(T)

    leff = lcond*om + lsub*(1-om)

    # Weighted qsat
    qs = qsw*om + qsi*(1-om)
    
    # Technically dq/dp * dp/dz
    dq_dp = g*qs/(Rd*T)

    # Clausius Clapeyron, for ice and vapor independently, then weighted together. Plus omega contr.
    dq_dT = claus_clap(T,qsw,lcond*om) + claus_clap(T,qsi,lsub*(1-om)) + dom*(qsw - qsi)
    
    # Lapse rate
    numer = -(g + leff*dq_dp)
    denom = cpd + leff*dq_dT
    
    return numer/denom

def dtMALR(T,p):
    """
        Calculate the derivative (wrt temp) of the moist adiabatic lapse rate
    """

    qsw = qsatw(T,p)/1e3
    qsi = qsati(T,p)/1e3
    dtqsw = dtqsatw(T,p)/1e3
    dtqsi = dtqsati(T,p)/1e3

    # Determine partition of non-precipitating phase:
    om  = omega_n(T)
    dom = dom_dT(T)
    
    # in order to calculate weighted partition
    qs = qsw*om + qsi*(1-om)
    
    leff   = lcond*om + lsub*(1-om)
    lqeff  = lcond*qsw*om + lsub*qsi*(1-om)
    l2qeff = (lcond**2)*qsw*om + (lsub**2)*qsi*(1-om)
    dtqeff = dtqsw*om + dtqsi*(1-om)
    
    # differences 
    dqs  = qsw - qsi
    dlqs = lcond*qsw - lsub*qsi
    
    # dq/dp * dp/dz
    dq_dp = g*qs/(Rd*T)

    # dq/dT
    dq_dT = dom*dqs + lqeff/(Rv*T**2)

    # MALR numerator and denominator
    numer = -(g + leff*dq_dp)
    denom = cpd + leff*dq_dT

    # dtMALR
    dt_numer = -(g/(Rd*T))*(leff*(lqeff/(Rv*T**2) - qs/T) - dom*qs*lfus)
    dt_denom = (leff/(Rv*T**2))*(dom*(2*dlqs - lqeff*lfus/leff) - 2*lqeff/T + l2qeff/(Rv*T**2))

    # Return dtMALR
    return dt_numer/denom - dt_denom*numer/denom**2

def dq_dz(T,p):
    """
        Calculate dqsat/dz following a moist adiabat
    """
    
    # Saturation Mixing Ratios
    qsw = qsatw(T,p)/1e3
    qsi = qsati(T,p)/1e3
    
    # Determine partition of non-precipitating phase:
    om  = omega_n(T)
    dom = dom_dT(T)
    
    # in order to calculate weighted partition
    qs = qsw*om + qsi*(1-om) # Ignore differences in how denominator is calculated/assume they are negligible

    leff = lcond*om + lsub*(1-om)
    
    # Technically dq_dz|_T/g = rho*dq_dp
    dq_dp = qs/(Rd*T)
    # Clausius Clapeyron, for ice and vapor independently, then weighted together.
    dq_dT = claus_clap(T,qsw,lcond)*om + claus_clap(T,qsi,lsub)*(1-om) - dom*(qsw-qsi)
    
    # MALR
    Gamma = -g*(1 + leff*dq_dp)/(cpd + leff*dq_dT)
    
    return Gamma*dq_dT + g*dq_dp

def dtq_dz(T,p):
    """
        Calculate the hydrostatic gradient in dq/dT.
    """
    
    # Saturation Mixing Ratios
    dtqsw = dtqsatw(T,p)/1e3
    dtqsi = dtqsati(T,p)/1e3
    
    # Determine partition of non-precipitating phase:
    om  = omega_n(T)
    # in order to calculate weighted partition
    dtqs = dtqsw*om + dtqsi*(1-om) # Ignore differences in how denominator is calculated/assume they are negligible
    
    # Technically dq_dz|_T/g = rho*dq_dp
    dq_dp = dtqs/(Rd*T)
    # Clausius Clapeyron, for ice and vapor independently, then weighted together.
    dq_dT = claus_clap(T,dtqsw,lcond*om) + claus_clap(T,dtqsi,lsub*(1-om))
    
    # "Effective" specific enthalpy of precipitation 
    Leff = lcond*om + lsub*(1-om)

    # MALR
    tGamma = tMALR(T,p)
    
    return tGamma*dq_dT + g*dq_dp
