"""Local static stability of moist air after Emanuel (1994), *Atmospheric Convection*, ch. 6.

    unsaturated (eq. 6.1.7):  N^2   = g d ln(theta_v)/dz
    saturated   (eq. 6.2.10): N_m^2 = 1/(1 + r_T) * [ Gamma_m ds*/dz - (c_l Gamma_m ln T + g) dr_T/dz ]

with Emanuel's reversible moist entropy per unit mass of dry air (his eq. 4.5.9)
    s = (c_pd + r_T c_l) ln T - R_d ln p_d + L_v r / T - r R_v ln H,
evaluated at saturation (r = r*, H = 1) for s*, and Gamma_m his reversible moist-adiabatic lapse
rate (eq. 4.7.3; written as in Marquet & Geleyn, arXiv:1401.2379, eqs. 28-29). Liquid water only,
as in the book: no ice phase. The saturated form is the moist Brunt-Vaisala frequency of Lalas &
Einaudi (1974) / Durran & Klemp (1982) written with entropy.

Interpretation, level by level:
    N^2 > 0 and N_m^2 > 0  absolutely stable
    N^2 > 0 and N_m^2 < 0  conditionally unstable (stable unless saturated)
    N^2 < 0                absolutely unstable

Units: T [K], p [hPa], mixing ratios [kg/kg], z [m]; N^2 [s^-2]; Gamma_m [K/m]; s [J kg^-1 K^-1].
"""
import numpy as np
import xarray as xr

# Emanuel (1994) constants
CPD, CPV, CL = 1005.7, 1870., 4190.          # J/kg/K: dry air, water vapour, liquid water
RD, RV = 287.04, 461.50                      # J/kg/K
EPS = RD / RV
LV0, T0 = 2.501e6, 273.15                    # J/kg at 0 C
G = 9.81
P0 = 1000.                                   # hPa


def lv(T):
    """Latent heat of vaporization L_v(T) = L_v0 + (c_pv - c_l)(T - 273.15), consistent with the entropy."""
    return LV0 + (CPV - CL) * (T - T0)


def esat(T):
    """Saturation vapour pressure over liquid water [hPa], Bolton (1980) (Emanuel eq. 4.4.13)."""
    Tc = T - T0
    return 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))


def rsat(T, p):
    """Saturation mixing ratio over liquid [kg/kg]."""
    e = esat(T)
    return EPS * e / np.maximum(p - e, 1e-3)


def theta_v(T, p, rv):
    """Virtual potential temperature of unsaturated air (r_T = r_v), Emanuel's definition:
    theta (1 + r_v/eps) / (1 + r_v), with the moist Poisson exponent (R_d + r R_v)/(c_pd + r c_pv)."""
    chi = (RD + rv * RV) / (CPD + rv * CPV)
    return T * (P0 / p) ** chi * (1 + rv / EPS) / (1 + rv)


def n2_unsaturated(T, p, rv, z):
    """Eq. 6.1.7: N^2 = g d ln(theta_v)/dz along the last axis (z) [s^-2]."""
    return G * np.gradient(np.log(theta_v(T, p, rv)), z, axis=-1)


def gamma_m(T, p, rT):
    """Reversible moist-adiabatic lapse rate [K/m] of saturated air with total water r_T (>= r*).
    Gamma_m = g/c_p* (1 + r_T) (1 + L_v r*/(R_d T)) / (1 + (1 + r*/eps) L_v^2 r*/(c_p* R_v T^2) + c_l r_l/c_p*),
    c_p* = c_pd + c_pv r*, r_l = r_T - r* (Emanuel eq. 4.7.3; Marquet & Geleyn eqs. 28-29)."""
    rs = rsat(T, p)
    rl = np.maximum(rT - rs, 0.)
    L = lv(T)
    cps = CPD + CPV * rs
    num = 1 + L * rs / (RD * T)
    den = 1 + (1 + rs / EPS) * L ** 2 * rs / (cps * RV * T ** 2) + CL * rl / cps
    return G / cps * (1 + rT) * num / den


def s_star(T, p, rT):
    """Saturated reversible moist entropy per unit mass of dry air [J/kg/K] (Emanuel eq. 4.5.9 with
    r = r*, H = 1): (c_pd + r_T c_l) ln T - R_d ln(p - e*) + L_v r*/T. Pressures in Pa inside the log;
    the additive constant is irrelevant for gradients."""
    e = esat(T)
    return (CPD + rT * CL) * np.log(T) - RD * np.log((p - e) * 100.) + lv(T) * rsat(T, p) / T


def n2_saturated_terms(T, p, rT, z):
    """Eq. 6.2.10 split exactly into its two physical parts (derivatives along the last axis, z):
        entropy = Gamma_m [ds*/dz - c_l ln T dr_T/dz] / (1 + r_T) = Gamma_m (ds*/dz at fixed r_T)/(1 + r_T)
                  -> the classic conditional-instability test (lapse rate vs the moist adiabat)
        loading = -g dr_T/dz / (1 + r_T)
                  -> condensate loading: a lifted parcel keeps its total water (reversible framework)
    N_m^2 = entropy + loading."""
    gm = gamma_m(T, p, rT)
    ds = np.gradient(s_star(T, p, rT), z, axis=-1)
    drT = np.gradient(rT, z, axis=-1)
    entropy = gm * (ds - CL * np.log(T) * drT) / (1 + rT)
    loading = -G * drT / (1 + rT)
    return entropy, loading


def n2_saturated(T, p, rT, z):
    """Eq. 6.2.10: N_m^2 = [Gamma_m ds*/dz - (c_l Gamma_m ln T + g) dr_T/dz] / (1 + r_T) [s^-2]."""
    entropy, loading = n2_saturated_terms(T, p, rT, z)
    return entropy + loading


def stability_profiles(C, condensate=('QN',)):
    """N^2 (6.1.7) and N_m^2 (6.2.10) from composite profiles C (dims ..., z) with TABS [K], QV and
    condensate [g/kg] and coordinate p [hPa] (composites.composite_experiment attaches it).

    Unsaturated N^2 uses the actual vapour r_v = QV. The saturated N_m^2 asks "how stable would this
    layer be if it were saturated?": r_T = r*(T, p) + condensate (sum of `condensate` fields; by
    default the cloud condensate QN, add 'QP' to include falling rain). Returns a Dataset with
    N2, N2_m (= N2_m_entropy + N2_m_loading), GAMMA_M [K/km], S_STAR and CLASS (0 stable,
    1 conditionally unstable, 2 absolutely unstable; from N2 and N2_m), same dims as C."""
    T = C['TABS']
    p = C['p'].broadcast_like(T)
    z = C['z'].values
    rv = C['QV'] / 1000.
    cond = sum((C[q] for q in condensate), xr.zeros_like(T)) / 1000.
    rT = rsat(T, p) + cond
    zdim = T.dims.index('z')
    # move z last for np.gradient(axis=-1), then back
    order = [d for d in T.dims if d != 'z'] + ['z']
    Tt, pt, rvt, rTt = (a.transpose(*order).values for a in (T, p, rv, rT))
    out = xr.Dataset(coords={k: v for k, v in C.coords.items() if set(v.dims) <= set(order)})
    out['N2'] = (order, n2_unsaturated(Tt, pt, rvt, z))
    ent, load = n2_saturated_terms(Tt, pt, rTt, z)
    out['N2_m'] = (order, ent + load)
    out['N2_m_entropy'] = (order, ent)
    out['N2_m_loading'] = (order, load)
    out['GAMMA_M'] = (order, 1000. * gamma_m(Tt, pt, rTt))
    out['S_STAR'] = (order, s_star(Tt, pt, rTt))
    out['CLASS'] = xr.where(out.N2 < 0, 2, xr.where(out.N2_m < 0, 1, 0)).where(np.isfinite(out.N2))
    out['N2'].attrs.update(units='s-2', long_name='N^2 unsaturated, Emanuel (1994) eq. 6.1.7')
    out['N2_m'].attrs.update(units='s-2', long_name='N_m^2 saturated, Emanuel (1994) eq. 6.2.10')
    out['N2_m_entropy'].attrs.update(units='s-2', long_name='6.2.10 entropy term: Gamma_m (ds*/dz)_rT / (1+r_T)')
    out['N2_m_loading'].attrs.update(units='s-2', long_name='6.2.10 condensate-loading term: -g dr_T/dz / (1+r_T)')
    out['GAMMA_M'].attrs.update(units='K/km', long_name='reversible moist-adiabatic lapse rate')
    out['S_STAR'].attrs.update(units='J/kg/K', long_name='saturated moist entropy per kg dry air')
    out['CLASS'].attrs.update(long_name='0 stable, 1 conditionally unstable, 2 absolutely unstable')
    return out.transpose(*T.dims) if zdim != len(order) - 1 else out
