"""Conditional (composite) vertical profiles of SAM 3D output.

Every snapshot, a user-defined *mask function* sorts the 128x128 columns into groups (e.g.
updraft / downdraft / intermediate). Every 3D field, plus MSE, frozen MSE, liquid/ice static
energy and theta_e, is then averaged over each group, level by level, over all snapshots.

    classify(f, z, **params) -> {group name: boolean (y, x) array}
        f : dict of float64 arrays (z, y, x) for ONE snapshot: the 3D fields (U V W PP QRAD
            TABS QV QN QP) and the derived ones (MSE MSE_frozen S_LI THETA_E), SAM units.
        z : (z,) heights [m].
        Groups must be the same every snapshot. They need not be exclusive or exhaustive.

composite(data, classify)            one run    -> Dataset (group, z), coords count, fraction
composite_experiment(exp, rvs, ...)  all r_v    -> Dataset (rv, group, z), coords RH(rv); cached
"""
import hashlib
import inspect
from pathlib import Path

import numpy as np
import xarray as xr

DERIVED_VERSION = 2      # bump when derived_fields changes, so cached composites are recomputed
# saturated counterparts drawn as dashed lines next to their parent quantity
SAT_PAIRS = {'MSE': 'MSE_SAT', 'MSE_frozen': 'MSE_frozen_SAT', 'THETA_E': 'THETA_ES'}

# SAM constants (SAM_SRC/params.f90) and the liquid/ice partition of MICRO_SAM1MOM/micro_params.f90
CP, G, LC, LF = 1004., 9.81, 2.5104e6, 0.3336e6
TBGMIN, TBGMAX, TPRMIN, TPRMAX = 253.16, 273.16, 268.16, 283.16


def derived_fields(f, p, z):
    """MSE variants and theta_e for one snapshot. f: 3D arrays (z, y, x) in SAM units
    (T K, q g/kg, PP Pa); p: (z,) hPa; z: (z,) m. Returns a dict of (z, y, x) arrays."""
    T, qv = f['TABS'], f['QV']
    qn = f.get('QN', np.zeros_like(T))
    qp = f.get('QP', np.zeros_like(T))
    zz = np.asarray(z, float)[:, None, None]
    ice_n = 1 - np.clip((T - TBGMIN) / (TBGMAX - TBGMIN), 0, 1)       # ice fraction of QN
    ice_p = 1 - np.clip((T - TPRMIN) / (TPRMAX - TPRMIN), 0, 1)       # ice fraction of QP
    q_ice = (ice_n * qn + ice_p * qp) / 1000.                          # kg/kg
    q_liq = ((1 - ice_n) * qn + (1 - ice_p) * qp) / 1000.
    out = {
        # simple MSE (kJ/kg), as defined by the user: 1.005 T + 2.5 qv + g z / 1000
        'MSE': 1.005 * T + 2.5 * qv + (9.8 / 1000) * zz,
        # frozen MSE (kJ/kg): conserved under condensation, evaporation and freezing
        'MSE_frozen': (CP * T + G * zz + LC * qv / 1000. - LF * q_ice) / 1000.,
        # SAM's prognostic liquid/ice static energy (kJ/kg)
        'S_LI': (CP * T + G * zz - LC * q_liq - (LC + LF) * q_ice) / 1000.,
    }
    # equivalent potential temperature (K), Bolton (1980) eqs. 22 and 39, at p + PP
    ptot = np.asarray(p, float)[:, None, None] + f.get('PP', 0.) / 100.
    e = ptot * (qv / 1000.) / (0.622 + qv / 1000.)
    TL = 2840. / (3.5 * np.log(T) - np.log(np.maximum(e, 1e-6)) - 4.805) + 55.
    out['THETA_E'] = _bolton_theta_e(T, ptot, qv, TL)
    # saturated counterparts: the same air if it were saturated (q_v -> q*, no condensate).
    # q* is SAM's saturation mixing ratio, liquid/ice mixed with the cloud partition (thermo_SAM.qsat).
    from .thermo_SAM import qsat
    qs = qsat(T, ptot)                                                  # g/kg
    out['MSE_SAT'] = 1.005 * T + 2.5 * qs + (9.8 / 1000) * zz
    out['MSE_frozen_SAT'] = (CP * T + G * zz + LC * qs / 1000.) / 1000.
    out['THETA_ES'] = _bolton_theta_e(T, ptot, qs, T)                   # saturated: T_L = T
    return out


def _bolton_theta_e(T, p, r, TL):
    """Bolton (1980) eq. 39; T, TL in K, p in hPa, r in g/kg."""
    return (T * (1000. / p) ** (0.2854 * (1 - 0.28e-3 * r))
            * np.exp((3.376 / TL - 0.00254) * r * (1 + 0.81e-3 * r)))


def fit_xlim(ax, profiles, z, zmax=None, pad=0.05):
    """Set ax's x-limits to the range of `profiles` (list of 1D arrays on z) below zmax."""
    z = np.asarray(z, float)
    sel = np.ones(z.size, bool) if zmax is None else z <= zmax
    vals = np.concatenate([np.asarray(v, float)[sel] for v in profiles])
    vals = vals[np.isfinite(vals)]
    if vals.size:
        lo, hi = vals.min(), vals.max()
        d = (hi - lo) * pad or abs(hi) * 1e-3 or 1e-3
        ax.set_xlim(lo - d, hi + d)


def _snapshot_p(s, nz):
    """Pressure profile (hPa) of one snapshot: a coordinate p(z) from the .bin3D reader or a
    data variable p(time, z) from netCDF files opened with open_mfdataset."""
    return np.asarray(s['p'].values, float).reshape(-1, nz)[0]


def composite(data, classify, params=None, include_all=True, progress=50):
    """Composite profiles of one run. data: Dataset (time, z, y, x) with p; classify(f, z, **params): see module.
    include_all adds group 'all' (domain mean), handy for anomalies (group - all)."""
    fields = [v for v in data.data_vars if data[v].dims == ('time', 'z', 'y', 'x')]
    z = data['z'].values.astype(float)
    nz, nt = len(z), data.sizes['time']
    sums, count, groups = {}, {}, None
    for t in range(nt):
        s = data.isel(time=t).load()
        f = {v: s[v].values.astype(np.float64) for v in fields}
        f.update(derived_fields(f, _snapshot_p(s, nz), z))
        masks = dict(classify(f, z, **(params or {})))
        if include_all:
            masks['all'] = np.ones(f['W'].shape[1:], bool)
        if groups is None:
            groups = list(masks)
        elif list(masks) != groups:
            raise ValueError(f'mask returned groups {list(masks)} at snapshot {t}, expected {groups}')
        for g, m in masks.items():
            m = np.asarray(m, bool)
            count[g] = count.get(g, 0) + int(m.sum())
            for v, a in f.items():
                sums[(g, v)] = sums.get((g, v), 0.) + a[:, m].sum(axis=1)
        if progress and (t % progress == 0 or t == nt - 1):
            print(f'    snapshot {t + 1}/{nt}')
    names = list(dict.fromkeys(v for _, v in sums))
    ncol = data.sizes['y'] * data.sizes['x'] * nt
    with np.errstate(invalid='ignore', divide='ignore'):
        out = xr.Dataset(
            {v: (('group', 'z'), np.array([sums[(g, v)] / count[g] if count[g] else np.full(nz, np.nan)
                                          for g in groups])) for v in names},
            coords=dict(group=groups, z=z, count=('group', [count[g] for g in groups]),
                        fraction=('group', [count[g] / ncol for g in groups])))
    for v in names:
        if v in data:
            out[v].attrs.update(data[v].attrs)
    out.attrs.update(n_snapshots=nt, t_start=float(data['time'][0]), t_end=float(data['time'][-1]))
    return out


def _step_of(path):
    """Model step from SAM's output name <case>_<caseid>_<nsub>_<step>.<ext> (None if not parseable)."""
    try:
        return int(Path(path).name.split('.')[0].rsplit('_', 1)[-1])
    except ValueError:
        return None


def resolve_source(run_dir, source='auto'):
    """'auto' -> 'nc' if every OUT_3D/*.bin3D has its converted .nc (or there are only .nc files),
    else 'bin'. 'bin' and 'nc' are returned unchanged. The two give bit-identical data."""
    if source != 'auto':
        return source
    out = Path(run_dir) / 'OUT_3D'
    bins = sorted(out.glob('*.bin3D'))
    if bins:
        return 'nc' if all(b.with_suffix('.nc').exists() for b in bins) else 'bin'
    return 'nc' if any(out.glob('*.nc')) else 'bin'


def open_3d(run_dir, source='auto', tmin=None, tmax=None, steps_per_day=8640):
    """3D output of one run as Dataset (time, z, y, x) with p.

    source: 'bin' reads OUT_3D/*.bin3D directly; 'nc' opens the converted .nc files (one per
    snapshot, from bin3D2nc); 'auto' picks 'nc' when every snapshot is converted. For 'nc', files
    outside [tmin, tmax] are skipped by their step number (steps_per_day = 86400/dt) and the rest
    are opened as plain consecutive snapshots, without cross-checking coordinates (much faster)."""
    run_dir = Path(run_dir)
    source = resolve_source(run_dir, source)
    if source == 'bin':
        from ..io.sam3d import open_run_3d
        return open_run_3d(run_dir, tmin=tmin, tmax=tmax)
    if source != 'nc':
        raise ValueError(f"source must be 'auto', 'bin' or 'nc', not {source!r}")
    files = sorted((run_dir / 'OUT_3D').glob('*.nc'), key=lambda f: (_step_of(f) is None, _step_of(f) or 0, f.name))
    if not files:
        raise FileNotFoundError(f'no .nc in {run_dir / "OUT_3D"}: convert first or use source="bin"')
    if all(_step_of(f) is not None for f in files):
        day = lambda f: _step_of(f) / steps_per_day
        files = [f for f in files if (tmin is None or day(f) >= tmin - 1e-6) and (tmax is None or day(f) <= tmax + 1e-6)]
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', data_vars='minimal',
                           coords='minimal', compat='override', join='override')
    return ds.sel(time=slice(tmin, tmax))


def mask_key(classify, params=None, **extra):
    """Short hash identifying a mask definition (its source code and parameters) for caching."""
    try:
        src = inspect.getsource(classify)
    except (OSError, TypeError):
        src = getattr(classify, '__name__', repr(classify))
    text = repr((src, sorted((params or {}).items()), sorted(extra.items())))
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def near_surface_rh(run_dir, tmin=None, tmax=None):
    """Mean lowest-level RELH (%) of a run's STAT file over [tmin, tmax] days (NaN if unavailable)."""
    try:
        from ..io.stat import open_stat
        st = open_stat(run_dir)
        return float(st['RELH'].isel(z=0).sel(time=slice(tmin, tmax)).mean())
    except Exception:
        return float('nan')


def stat_pressure(run_dir, z, tmin=None, tmax=None):
    """Mean pressure profile p(z) [hPa] from a run's STAT file over [tmin, tmax] (NaN if unavailable).
    Needed for stability diagnostics; the composites themselves only carry PP (perturbation)."""
    try:
        from ..io.stat import open_stat
        st = open_stat(run_dir)
        prof = st['p'].sel(time=slice(tmin, tmax)).mean('time') if 'time' in st['p'].dims else st['p']
        return np.interp(z, st['z'].values, prof.values)
    except Exception:
        return np.full(len(z), np.nan)


def composite_experiment(experiment, rv_list, classify, mask_name='mask', mask_params=None,
                         source='auto', tmin=None, tmax=None, include_all=True, cache=True,
                         force=False, run_name='stage2_rv{rv:g}', key_extra=''):
    """composite() for every r_v of an experiment -> Dataset (rv, group, z) with coords
    fraction(rv, group), count(rv, group) and RH(rv). Each run is cached in
    <experiment>/analysis_cache/composite_<mask_name>_<key>_rv<rv>.nc; the key changes when the
    mask's code or parameters, the time window or include_all change (not with the source:
    .bin3D and converted .nc give identical results)."""
    experiment = Path(experiment)
    # key_extra: e.g. the source of helper functions the mask calls, so editing them also refreshes the cache
    key = mask_key(classify, mask_params, tmin=tmin, tmax=tmax, include_all=include_all,
                   derived=DERIVED_VERSION, extra=key_extra)          # not the source: bin and nc are bit-identical
    cdir = experiment / 'analysis_cache'
    runs = []
    for rv in rv_list:
        rdir = experiment / run_name.format(rv=rv)
        f = cdir / f'composite_{mask_name}_{key}_rv{rv:g}.nc'
        if cache and f.exists() and not force:
            print(f'rv={rv:g}: from cache ({f.name})')
            c = xr.load_dataset(f)
        else:
            src = resolve_source(rdir, source)
            print(f'rv={rv:g}: computing from {rdir} ({src} files)')
            c = composite(open_3d(rdir, src, tmin, tmax), classify, mask_params, include_all)
            c = c.assign_coords(RH=near_surface_rh(rdir, c.attrs['t_start'], c.attrs['t_end']))
            if cache:
                cdir.mkdir(exist_ok=True)
                c.attrs.update(mask_name=mask_name, mask_key=key, mask_params=repr(mask_params or {}),
                               source=src)
                c.to_netcdf(f)
        if 'p' not in c.coords:
            c = c.assign_coords(p=('z', stat_pressure(rdir, c.z.values, c.attrs.get('t_start'), c.attrs.get('t_end')),
                                   dict(units='hPa', long_name='mean pressure (STAT)')))
        runs.append(c.expand_dims(rv=[float(rv)]))
    out = xr.concat(runs, dim='rv', coords=['count', 'fraction', 'RH', 'p'], combine_attrs='drop_conflicts')
    out.attrs.update(mask_name=mask_name, mask_key=key, mask_params=repr(mask_params or {}))
    return out
