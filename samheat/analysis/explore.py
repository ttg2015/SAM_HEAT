"""One-liners for exploring the runs in a notebook.

    from samheat.analysis.explore import load, stat_profiles, plot_profiles, plot_grid, masks

    C = load("vdd2025_vddgrid", [0, 2000], masks.cloud)            # composites: (rv, group, z)
    S = stat_profiles("vdd2025_vddgrid", [0, 2000])                 # SAM's own STAT profiles: (rv, z)
    plot_profiles(C, "MSE", group="cloud")                         # one line per r_v, coloured by RH
    plot_profiles(C, "MSE", by="group", rv=0)                      # one line per group
    plot_grid(C, ["W", "QN", "MSE", "THETA_E"], group="cloud")     # several panels

Everything returns plain xarray objects / matplotlib axes, so you can keep going with xarray
(.sel, .mean, arithmetic, .plot) whenever these helpers are not enough.
"""
import inspect
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import masks                                     # noqa: F401  (re-exported for notebooks)
from .composites import SAT_PAIRS, composite_experiment, fit_xlim, near_surface_rh
from ..paths import SITE

DEFAULT_XLIMS = {'MSE': (320, 350), 'MSE_frozen': (320, 350), 'THETA_E': (320, 350), 'MSE_SAM': (320, 350)}


def experiment_dir(experiment):
    """An experiment name (under SITE.work_root) or a path."""
    p = Path(experiment)
    return p if p.is_absolute() else Path(SITE.work_root) / p


def load(experiment, rv_list, mask, name=None, source='auto', tmin=30, tmax=60, include_all=True,
         force=False, **params):
    """Composites of every run (stage2_rv<rv>) of an experiment for one mask -> Dataset (rv, group, z).
    mask: a function from samheat.analysis.masks or your own; **params go to the mask.
    Cached per mask code + parameters + time window; force=True recomputes."""
    module = sys.modules.get(getattr(mask, '__module__', ''), None)
    extra = ''
    if module is not None and module.__name__.startswith('samheat.'):
        extra = inspect.getsource(module)                   # library edits also refresh the cache
    return composite_experiment(experiment_dir(experiment), rv_list, mask, name or mask.__name__, params,
                                source=source, tmin=tmin, tmax=tmax, include_all=include_all,
                                force=force, key_extra=extra)


def stat_profiles(experiment, rv_list, variables=None, tmin=30, tmax=60, run_name='stage2_rv{rv:g}'):
    """Time-mean (tmin..tmax days) STAT profiles of every run -> Dataset (rv, z), with RH(rv).
    variables: list of names, or None for every (time, z) variable. SAM writes MSE and the
    cloud-conditional MSECLD etc. in K (h/c_p): multiply by 1.004 for kJ/kg."""
    from ..io.stat import open_stat
    runs = []
    for rv in rv_list:
        rdir = experiment_dir(experiment) / run_name.format(rv=rv)
        st = open_stat(rdir)
        if st is None:
            raise FileNotFoundError(f'no STAT output in {rdir}')
        names = variables or [v for v in st.data_vars if st[v].dims == ('time', 'z')]
        m = st[names].sel(time=slice(tmin, tmax)).mean('time')
        m = m.assign_coords(RH=near_surface_rh(rdir, tmin, tmax))
        runs.append(m.expand_dims(rv=[float(rv)]))
    return xr.concat(runs, dim='rv', coords=['RH'])


def rh_colors(D, cmap='YlGnBu'):
    """{rv: colour}: moister runs darker along `cmap` (uses the RH coordinate; falls back to order)."""
    rvs = D['rv'].values
    rh = D['RH'].values.astype(float) if 'RH' in D.coords else np.full(len(rvs), np.nan)
    if not np.isfinite(rh).any():                        # no RH available: colour by order
        rh = np.arange(len(rvs), dtype=float)[::-1]
    lo, hi = np.nanmin(rh), np.nanmax(rh)
    n = (rh - lo) / (hi - lo) if hi > lo else np.ones_like(rh)
    cm = plt.get_cmap(cmap)
    return {rv: cm(0.3 + 0.7 * x) for rv, x in zip(rvs, n)}


def _label(D, rv=None, group=None):
    parts = []
    if rv is not None:
        parts.append(f'rv {rv:g}')
        if 'RH' in D.coords:
            parts.append(f'RH {float(D.RH.sel(rv=rv)):.0f}%')
    if group is not None:
        parts.append(str(group))
        if 'fraction' in D.coords and 'group' in D['fraction'].dims:
            fr = D['fraction'].sel(group=group)
            if rv is not None and 'rv' in fr.dims:
                fr = fr.sel(rv=rv)
            parts.append(f'{100 * float(fr):.1f}%')
    return ', '.join(parts)


def plot_profiles(D, var, by='rv', group=None, rv=None, ax=None, zmax=16000, xlim='auto', sat=None,
                  anomaly=False, legend=True, colors=None, **line_kw):
    """One panel: `var` against height.

    by='rv'    one line per r_v, coloured by RH (choose `group` if D has groups)
    by='group' one line per group (choose `rv` if D has several runs)
    sat        also draw the saturated counterpart dashed (default: yes when it exists)
    anomaly    subtract group 'all' of the same run
    xlim       'auto' (fit the data below zmax; fixed 320-350 for MSE / theta_e), None, or (lo, hi)
    Works for composites (rv, group, z), stability output, STAT profiles (rv, z) and single runs."""
    ax = ax or plt.subplots(figsize=(4.5, 5))[1]
    da = D[var]
    if 'group' in da.dims and by == 'rv':
        group = group if group is not None else [g for g in D.group.values if g != 'all'][0]
    if 'rv' in da.dims and by == 'group':
        rv = rv if rv is not None else float(D.rv.values[0])
    names = [var] + ([SAT_PAIRS[var]] if (sat is None or sat) and var in SAT_PAIRS and SAT_PAIRS[var] in D else [])
    lines = (D.rv.values if by == 'rv' and 'rv' in da.dims else
             [g for g in D.group.values if not (anomaly and g == 'all')] if by == 'group' else [None])
    colors = colors or (rh_colors(D) if by == 'rv' and 'rv' in da.dims else {})
    shown = []
    for i, key in enumerate(lines):
        sel = dict(rv=key if by == 'rv' else rv, group=group if by == 'rv' else key)
        for j, name in enumerate(names):
            prof = D[name].sel({k: v for k, v in sel.items() if k in D[name].dims and v is not None})
            if anomaly:
                prof = prof - D[name].sel({k: (v if k != 'group' else 'all') for k, v in sel.items()
                                           if k in D[name].dims and (v is not None or k == 'group')})
            c = colors.get(key, f'C{i}')
            ax.plot(prof, D.z / 1000, c=c, ls='--' if j else '-', lw=1 if j else 1.5,
                    label=None if j else _label(D, sel['rv'] if by == 'rv' else rv, sel['group']), **line_kw)
            shown.append(np.asarray(prof))
    if isinstance(xlim, tuple):
        ax.set_xlim(*xlim)
    elif xlim == 'auto':
        if var in DEFAULT_XLIMS and not anomaly:
            ax.set_xlim(*DEFAULT_XLIMS[var])
        else:
            fit_xlim(ax, shown, D.z, zmax)
    if anomaly:
        ax.axvline(0, c='.7', lw=.5)
    if zmax:
        ax.set_ylim(0, zmax / 1000)
    units = da.attrs.get('units', '')
    ax.set_xlabel(f'{var}{" - all" if anomaly else ""}' + (f' [{units}]' if units else ''))
    ax.set_ylabel('z (km)')
    title = f'group: {group}' if by == 'rv' and group is not None else (f'rv {rv:g}' if rv is not None else '')
    ax.set_title(title + (' (dashed: saturated)' if len(names) > 1 else ''), fontsize=9)
    if legend:
        ax.legend(fontsize=7)
    return ax


def plot_grid(D, variables, ncol=4, **kw):
    """Several plot_profiles panels sharing the height axis; kw as for plot_profiles."""
    nrow = -(-len(variables) // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4.3 * nrow), sharey=True, squeeze=False)
    for i, (ax, v) in enumerate(zip(axs.flat, variables)):
        plot_profiles(D, v, ax=ax, legend=(i == 0), **kw)
        if i % ncol:
            ax.set_ylabel('')
    for ax in axs.flat[len(variables):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_cloud_env(C, var='MSE', rv=None, cloud='cloud', env='clear', ax=None, zmax=16000, xlim=(320, 350)):
    """The classic RCE picture in one panel, for each run (colour = RH):
        solid   `var` of cloudy air            (group `cloud`)
        dashed  `var` of the whole domain      (group 'all')
        dotted  saturated `var` (h*) of the environment (group `env`, e.g. clear air)
    Needs a composite from a mask with those groups, e.g. load(..., masks.cloud)."""
    from matplotlib.lines import Line2D
    ax = ax or plt.subplots(figsize=(5.5, 6))[1]
    sat = SAT_PAIRS[var]
    rvs = C.rv.values if rv is None else np.atleast_1d(rv)
    col = rh_colors(C)
    for r in rvs:
        c = col.get(r, 'k')
        ax.plot(C[var].sel(rv=r, group=cloud), C.z / 1000, c=c, lw=1.8)
        ax.plot(C[var].sel(rv=r, group='all'), C.z / 1000, c=c, lw=1.2, ls='--')
        ax.plot(C[sat].sel(rv=r, group=env), C.z / 1000, c=c, lw=1.2, ls=':')
    handles = [Line2D([], [], c='k', lw=1.8, label=f'{var}, {cloud}'),
               Line2D([], [], c='k', ls='--', label=f'{var}, domain mean'),
               Line2D([], [], c='k', ls=':', label=f'{sat}, {env} (h*)')]
    handles += [Line2D([], [], c=col[r], lw=3, label=_label(C, rv=r)) for r in rvs]
    ax.legend(handles=handles, fontsize=7)
    if xlim:
        ax.set_xlim(*xlim)
    if zmax:
        ax.set_ylim(0, zmax / 1000)
    ax.set_xlabel(f'{var} [kJ/kg]'); ax.set_ylabel('z (km)')
    return ax


def compare_cloud_with_stat(C, S, zmax=10000., cloud='cloud'):
    """How well a 3D cloud composite (mask masks.cloud, SAM's criterion) reproduces SAM's own STAT
    cloud statistics, per run, below zmax:
        cloud fraction  fraction_z[cloud]  vs  STAT CLD
        cloud MSE       MSE_SAM[cloud]     vs  1.004 * STAT MSECLD   (SAM writes MSE in K = h/c_p)
    Returns a DataFrame of max / rms differences. Expect small, not zero: the composites use
    3-hourly snapshots, STAT averages every statistics sample in each hour."""
    import pandas as pd
    rows = []
    for r in C.rv.values:
        zs = C.z.values[C.z.values <= zmax]
        fr = C.fraction_z.sel(rv=r, group=cloud).sel(z=zs)
        st = S.sel(rv=r).interp(z=zs)
        dfrac = fr.values - st['CLD'].values
        h3d = C['MSE_SAM'].sel(rv=r, group=cloud).sel(z=zs).values
        hst = 1.004 * st['MSECLD'].values
        ok = (st['CLD'].values > 0) & np.isfinite(h3d)
        dh = h3d[ok] - hst[ok]
        rows.append(dict(rv=r, cloud_frac_mean_3D=float(np.nanmean(fr)), cloud_frac_mean_STAT=float(np.nanmean(st['CLD'])),
                         frac_max_abs_diff=float(np.nanmax(np.abs(dfrac))), frac_rms_diff=float(np.sqrt(np.nanmean(dfrac ** 2))),
                         mse_max_abs_diff_kJkg=float(np.max(np.abs(dh))) if dh.size else np.nan,
                         mse_rms_diff_kJkg=float(np.sqrt(np.mean(dh ** 2))) if dh.size else np.nan))
    return pd.DataFrame(rows).set_index('rv').round(4)
