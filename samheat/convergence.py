"""The Stage-1 convergence measure (paper Eq. 2 and Sec. 2c): the pressure-weighted mean
temperature of the 600-400 hPa layer, compared with the reference within 0.1 K.

tabs_integral mirrors SAM_SRC/simple_ocean.f90 tabs_integral() (and iter_conv.py). It
accepts the two bounds in either order (the v2 sweep passed them descending and got 0).
"""
import numpy as np


def tabs_integral(p1, p2, p, tabs):
    """Mean of tabs between pressures p1 and p2 (hPa, any order); p decreases with index."""
    plow, pupp = min(p1, p2), max(p1, p2)
    p = np.asarray(p, float)
    tabs = np.asarray(tabs, float)
    nz = len(p)
    out = 0.0
    for k in range(1, nz - 1):
        kk = nz - k
        dp = min(p[kk - 1] - p[kk], p[kk - 1] - plow, pupp - p[kk], pupp - plow)
        dp = max(dp, 0.0)
        a = min(max((plow - p[kk]) / (p[kk - 1] - p[kk]), 0.0), 1.0)
        b = min(max((p[kk - 1] - pupp) / (p[kk - 1] - p[kk]), 0.0), 1.0)
        out += (dp / 2) * (tabs[kk] * (1 - a + b) + tabs[kk - 1] * (1 - b + a)) / (pupp - plow)
    return out


def layer_T(stat, t_start, t_end, p_layer=(400., 600.)):
    """Layer-mean T of the time-mean STAT profile between t_start and t_end (days)."""
    prof = stat['TABS'].sel(time=slice(t_start, t_end)).mean('time').values
    return tabs_integral(p_layer[0], p_layer[1], stat['p'].values, prof)


def initial_layer_T(stat, p_layer=(400., 600.)):
    """What SAM relaxes towards: tabs0_int, set in setdata.f90 from the initial profile
    (= the reference sounding the run was started from)."""
    return tabs_integral(p_layer[0], p_layer[1], stat['p'].values, stat['TABS'].isel(time=0).values)


def stage1_check(stat, window_days=10, tol_K=0.1, p_layer=(400., 600.), T_ref=None):
    """Paper criterion. Returns dict(converged, T_ref, T_last, diff, sst, day).

    T_ref defaults to the run's own initial layer mean (see initial_layer_T)."""
    t1 = float(stat['time'][-1])
    T0 = initial_layer_T(stat, p_layer) if T_ref is None else T_ref
    T1 = layer_T(stat, t1 - window_days, t1, p_layer)
    sst_var = 'SST' if 'SST' in stat else 'TABS_S'
    sst = float(stat[sst_var].sel(time=slice(t1 - window_days, t1)).mean())
    ok = bool(np.isfinite(T1) and np.isfinite(sst) and abs(T1 - T0) < tol_K)
    return dict(converged=ok, T_ref=float(T0), T_last=float(T1), diff=float(T1 - T0),
                sst=sst, day=t1)


def drift_check(stat, window_days=10, tol_K=0.1, p_layer=(400., 600.)):
    """Ocean-reference equilibrium: last window vs the one before."""
    t1 = float(stat['time'][-1])
    a = layer_T(stat, t1 - window_days, t1, p_layer)
    b = layer_T(stat, t1 - 2 * window_days, t1 - window_days, p_layer)
    return dict(converged=bool(abs(a - b) < tol_K), T_last=float(a), T_prev=float(b),
                drift=float(a - b), day=t1)
