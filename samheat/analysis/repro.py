"""Reproduction of van der Drift & O'Gorman (2025), "Dependence of convective precipitation
extremes on near-surface relative humidity" (arXiv:2412.16306), from SAM_HEAT runs.

Functions (paper reference in brackets):
    table1             horizontal- and time-mean Ts, T, RH, qv, P                      [Table 1]
    mean_profiles      mean RH(z), theta(z) and the Romps (2017) LCL                   [Fig. 1]
    extreme_mean       mean of all values at and above the q-th percentile             [P_e, C_e]
    block_bootstrap    8x8-gridpoint x 1-snapshot block bootstrap of an extreme mean   [Sec. 3c, Fig. 2]
    condensation_rate  column condensation C, LCL..z_t, moist-adiabatic dq*/dz, updrafts only  [Eq. 3]
    extreme_profiles   mean rho*w~ and -(dq*/dz)_ma over the top 0.1 % of C columns    [Eq. 5 inputs]
    bootstrap_extreme_profiles   the same, for block-bootstrap replicates
    decompose          fractional-change decomposition of P_e                          [Eq. 4-5, Fig. 3]
    analyze_run / load_summaries   glue: one run directory -> everything above (cached)

CONVENTIONS AND UNITS (read before comparing with the paper).
  * SAM 2D 'Prec' is in mm/day; the paper's P_e, C_e are in mm/hr. All P_e / C_e here are
    mm/hr (Prec / 24; C: 1 g m-2 s-1 = 3.6 mm/hr). Mean P in `table1` is mm/day, as in Table 1.
  * 3D fields as written by SAM: TABS K, QV g/kg, W m/s, p [mb] on the z coordinate.
    rho*w~ ('dy') is kg m-2 s-1, -(dq*/dz)_ma ('th') is g kg-1 m-1, so C = int dy*th dz is
    g m-2 s-1.
  * 'near-surface' is the lowest model level. In the paper it is 40 m; in SAM_HEAT's grid
    (FWRCE/grd, z(1) = 25 m) it is 25 m, so near-surface T, RH and q_v differ slightly from
    Table 1 by construction.
  * Scaling rates are returned per unit fractional change of RH: 1.1 means 1.1 % per %.
  * Percentile definition: P_e = mean(P[P >= percentile(P, q)]). The C_e / profile statistics
    use the equivalent top-k mean, k = ceil(N (1 - q/100)); the two differ by at most one
    sample in ~4e6.
"""
import math
import json
import pickle
from pathlib import Path

import numpy as np
import xarray as xr

from .lcl import lcl
from .thermo_SAM import dq_dz, qsat

RD = 287.05                # J/kg/K, as in vdD's load_3D
G_M2_S_TO_MM_HR = 3.6      # 1 g m-2 s-1 = 1e-3 mm / s = 3.6 mm/hr
ZT = 14e3                  # m, upper bound of the condensation integral (paper Eq. 3)


# ---------------------------------------------------------------------------- Table 1, Fig. 1
def _last_days(ds, last_days):
    t1 = float(ds['time'][-1])
    return ds.sel(time=slice(t1 - last_days, t1))


def table1(stat_ds, last_days=30, ts=None, prec2d=None):
    """Horizontal- and time-mean quantities of one run over the last `last_days` days.

    stat_ds : converted STAT dataset (time, z) -- e.g. samheat.io.stat.open_stat(run_dir).
    ts      : surface temperature override [K]; otherwise STAT 'SST' (or 'TABS_S') is used.
    prec2d  : optional 2D dataset with 'Prec' (mm/day) used for P if STAT has no 'PREC'.
    Returns dict: Ts [K], T_sfc [K] (lowest level), RH [%], qv [g/kg], P [mm/day]
    (NaN where the source variable is absent).
    """
    s = _last_days(stat_ds, last_days)
    lo = s.isel(z=0)
    if ts is None:
        ts = next((float(s[v].mean()) for v in ('SST', 'TABS_S') if v in s), np.nan)
    if 'PREC' in s:
        P = float(s['PREC'].mean())
    elif prec2d is not None:
        t0, t1 = float(s['time'][0]), float(s['time'][-1])
        P = float(prec2d['Prec'].sel(time=slice(t0, t1)).mean())
    elif 'PRECIP' in s:                  # profile of precipitation flux; lowest level
        P = float(lo['PRECIP'].mean())
    else:
        P = np.nan
    return dict(Ts=float(ts), T_sfc=float(lo['TABS'].mean()), RH=float(lo['RELH'].mean()),
                qv=float(lo['QV'].mean()), P=P)


def mean_profiles(stat_ds, last_days=30):
    """Time-mean RH(z) [%], theta(z) [K] and the Romps (2017) LCL (paper Fig. 1).

    The LCL is computed exactly like the paper: from the horizontal- and time-mean lowest-level
    p, T and RH. Returns an xarray.Dataset on z with RELH, THETA, TABS, p, and scalar `zlcl`
    [m above the surface = z(lowest level) + Romps' lifting distance of that parcel].
    """
    s = _last_days(stat_ds, last_days).mean('time')
    lo = s.isel(z=0)
    z0 = float(s['z'][0])
    p, T, rh = float(lo['p']), float(lo['TABS']), float(lo['RELH']) / 100.0
    zlcl = z0 + float(lcl(p * 1e2, T, rh=rh))
    keep = [v for v in ('RELH', 'THETA', 'TABS', 'p') if v in s]
    out = s[keep].copy()
    out['zlcl'] = zlcl
    out['zlcl'].attrs.update(units='m', long_name='Romps (2017) LCL from mean lowest-level T, p, RH')
    return out


# ---------------------------------------------------------------------------- extremes
def extreme_mean(field, q=99.9):
    """Mean of all values at and above the q-th percentile of `field` (paper's P_e / C_e)."""
    x = np.asarray(field, dtype=float).ravel()
    x = x[np.isfinite(x)]
    thr = np.percentile(x, q)
    return float(x[x >= thr].mean())


def _blocks(field, block):
    """(nt, ny, nx) -> (nt * ny/block * nx/block, block*block); ragged edges are dropped."""
    nt, ny, nx = field.shape
    by, bx = ny // block, nx // block
    f = np.asarray(field)[:, :by * block, :bx * block]
    return f.reshape(nt, by, block, bx, block).transpose(0, 1, 3, 2, 4).reshape(nt * by * bx, -1)


def block_draws(shape, block=8, n=100, seed=0):
    """Yield n arrays of block indices (with replacement), the same blocks for every call
    with equal (shape, block, n, seed). `shape` = (nt, ny, nx); a block is block x block
    gridpoints x one snapshot, indexed t*(ny//block)*(nx//block) + (y//block)*(nx//block) + x//block."""
    nt, ny, nx = shape
    nb = nt * (ny // block) * (nx // block)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.integers(0, nb, nb)


def block_bootstrap(field2d_time, block=8, n=100, q=99.9, seed=0):
    """Block-bootstrap the extreme mean of a (time, y, x) field (paper Sec. 3c).

    The field is cut into block x block (default 8 x 8 km at 1 km spacing) x 1-snapshot
    blocks; the set of blocks is resampled n times with replacement into datasets of the
    same size, and `extreme_mean(., q)` is computed for each. Deterministic given `seed`.
    Returns dict: dist (n,), p05, p95, mean, estimate (extreme mean of the original field).
    """
    f = np.asarray(field2d_time)
    if f.ndim != 3:
        raise ValueError('field2d_time must have shape (time, y, x)')
    blocks = _blocks(f, block)
    dist = np.array([extreme_mean(blocks[ids], q) for ids in block_draws(f.shape, block, n, seed)])
    return dict(dist=dist, p05=float(np.percentile(dist, 5)), p95=float(np.percentile(dist, 95)),
                mean=float(dist.mean()), estimate=extreme_mean(f, q))


# ---------------------------------------------------------------------------- condensation
def _rho_profile(ds3d, rho, z, p):
    if rho is None:
        if 'rho' in ds3d:
            rho = ds3d['rho']
        else:                                   # vdD's load_3D: p / (R * mean T)
            Tm = ds3d['TABS'].mean(('time', 'y', 'x')).values
            return p * 100.0 / (RD * Tm)
    if isinstance(rho, xr.DataArray):
        if 'z' in rho.dims and rho.sizes['z'] != len(z):
            rho = rho.interp(z=z)
        return np.asarray(rho.values, float)
    return np.asarray(rho, float)


def _terms(tabs, qv0, w, z, p, rho):
    """Integrands for numpy arrays tabs, w (nt, nz, ny, nx), qv0 (nt, ny, nx) [g/kg], on levels z.
    Returns dy = rho*max(w,0) (nt,nz,ny,nx), th = -(dq*/dz)_ma * mask(z >= z_LCL) [g/kg/m],
    zlcl (nt,ny,nx) [m]."""
    tabs = tabs.astype(np.float64)
    th = -dq_dz(tabs, p[None, :, None, None]) * 1e3
    rh0 = np.minimum(qv0 / qsat(tabs[:, 0], p[0]), 1.0)
    zlcl = z[0] + np.asarray(lcl(p[0] * 1e2, tabs[:, 0], rh=rh0), float)
    th = th * (zlcl[:, None] <= z[None, :, None, None])
    dy = rho[None, :, None, None] * np.fmax(w, 0.0)
    return dy, th, zlcl


def _levels(ds3d, zt):
    z = ds3d['z'].values.astype(float)
    nz = int(np.searchsorted(z, zt, side='right'))
    return z[:nz], ds3d['p'].values.astype(float)[:nz], nz


def condensation_rate(ds3d, zt=ZT, rho=None, chunk=8):
    """Column-integrated condensation rate C of paper Eq. (3), for every column and snapshot.

        C = int_{z_LCL}^{z_t} -(dq*/dz)_ma  rho  max(w, 0)  dz

    z_LCL: Romps (2017) LCL of the column's own lowest-level T, p and RH (via .lcl);
    (dq*/dz)_ma: SAM-consistent moist-adiabatic gradient (thermo_SAM.dq_dz, partitioned
    liquid/ice); rho: kg/m3 profile (default: ds3d['rho'] if present, else p/(R <T>) as in
    vdD; pass e.g. the STAT 'RHO' profile for SAM's own base state); updrafts only; trapezoid
    in z over levels <= zt (default 14 km).

    ds3d : Dataset (time, z, y, x) with TABS [K], QV [g/kg], W [m/s] and p(z) [mb]
           (as returned by samheat.io.sam3d.open_run_3d). Processed `chunk` snapshots at a time.
    Returns Dataset: C (time, y, x) [mm/hr] and zlcl (time, y, x) [m].
    """
    z, p, nz = _levels(ds3d, zt)
    rho = _rho_profile(ds3d, rho, ds3d['z'].values, ds3d['p'].values)[:nz]
    nt = ds3d.sizes['time']
    C = np.empty((nt, ds3d.sizes['y'], ds3d.sizes['x']), np.float32)
    ZL = np.empty_like(C)
    for a in range(0, nt, chunk):
        sl = slice(a, min(a + chunk, nt))
        tabs = ds3d['TABS'].isel(time=sl, z=slice(0, nz)).values
        w = ds3d['W'].isel(time=sl, z=slice(0, nz)).values
        qv0 = ds3d['QV'].isel(time=sl, z=0).values
        dy, th, zlcl = _terms(tabs, qv0, w, z, p, rho)
        C[sl] = np.trapezoid(dy * th, z, axis=1) * G_M2_S_TO_MM_HR
        ZL[sl] = zlcl
    coords = {k: ds3d[k] for k in ('time', 'y', 'x')}
    return xr.Dataset(dict(C=(('time', 'y', 'x'), C, dict(units='mm/hr', long_name='column condensation rate (Eq. 3)')),
                           zlcl=(('time', 'y', 'x'), ZL, dict(units='m'))), coords=coords)


def _topk(N, q):
    return int(math.ceil(N * (1.0 - q / 100.0)))


def extreme_profiles(ds3d, cond, q=99.9, zt=ZT, rho=None, cand_frac=0.01):
    """Profiles of rho*w~ ('dy') and -(dq*/dz)_ma ('th') averaged over the columns at and above
    the q-th percentile of C (the subscript e of paper Eq. 5), and C_e itself.

    cond : output of `condensation_rate` (same ds3d). The top `cand_frac` of columns are kept
    with their profiles ('cand', sorted by C) so that `bootstrap_extreme_profiles` needs no
    second pass over the 3D data.
    Returns dict: z (m, up to zt), Ce [mm/hr], dy (nz) [kg m-2 s-1], th (nz) [g kg-1 m-1],
    k (columns averaged), cand (dict).
    """
    C = cond['C'].values
    z, p, nz = _levels(ds3d, zt)
    rho = _rho_profile(ds3d, rho, ds3d['z'].values, ds3d['p'].values)[:nz]
    k = _topk(C.size, q)
    K = min(C.size, max(int(C.size * cand_frac), 2 * k))
    flat = np.argpartition(C.ravel(), -K)[-K:]
    flat = flat[np.argsort(-C.ravel()[flat], kind='stable')]
    ti, yi, xi = np.unravel_index(flat, C.shape)
    dy = np.empty((K, nz))
    th = np.empty((K, nz))
    for t in np.unique(ti):
        sel = np.nonzero(ti == t)[0]
        tabs = ds3d['TABS'].isel(time=[t], z=slice(0, nz)).values
        w = ds3d['W'].isel(time=[t], z=slice(0, nz)).values
        qv0 = ds3d['QV'].isel(time=[t], z=0).values
        d, h, _ = _terms(tabs, qv0, w, z, p, rho)
        dy[sel] = d[0][:, yi[sel], xi[sel]].T
        th[sel] = h[0][:, yi[sel], xi[sel]].T
    Cs = C.ravel()[flat].astype(float)
    cand = dict(C=Cs, dy=dy, th=th, t=ti, y=yi, x=xi, shape=C.shape, k=k)
    return dict(z=z, Ce=float(Cs[:k].mean()), dy=dy[:k].mean(0), th=th[:k].mean(0), k=k, cand=cand)


def bootstrap_extreme_profiles(cand, block=8, n=100, seed=0):
    """Block-bootstrap C_e, dy_e(z), th_e(z) from the candidate columns of `extreme_profiles`.

    Uses the same block draws as `block_bootstrap` for equal (block, n, seed), so P_e and C_e
    replicates are paired. Exact (not an approximation) provided the candidates contain the
    top-k of every replicate, which is checked (raise -> increase `cand_frac`).
    Returns dict Ce (n,), dy (n, nz), th (n, nz).
    """
    nt, ny, nx = cand['shape']
    by, bx = ny // block, nx // block
    nb = nt * by * bx
    valid = (cand['y'] < by * block) & (cand['x'] < bx * block)
    bid = np.where(valid, cand['t'] * by * bx + (cand['y'] // block) * bx + cand['x'] // block, 0)
    k = cand['k']
    Ce, dy, th = [], [], []
    for ids in block_draws(cand['shape'], block, n, seed):
        w = np.bincount(ids, minlength=nb)[bid].astype(float) * valid
        cw = np.cumsum(w)
        if cw[-1] < k:
            raise ValueError('candidate set too small for this bootstrap replicate; raise cand_frac')
        m = int(np.searchsorted(cw, k))
        w = w[:m + 1].copy()
        w[m] -= cw[m] - k
        Ce.append(w @ cand['C'][:m + 1] / k)
        dy.append(w @ cand['dy'][:m + 1] / k)
        th.append(w @ cand['th'][:m + 1] / k)
    return dict(Ce=np.array(Ce), dy=np.array(dy), th=np.array(th))


# ---------------------------------------------------------------------------- decomposition
def decompose(runs):
    """Fractional-change decomposition of P_e between adjacent r_v simulations (paper Eq. 4-5).

    runs : list of dicts, one per r_v (sorted internally by 'rv'), each with
        rv, RH (horizontal/time-mean near-surface RH; any unit), Pe [mm/hr], Ce [mm/hr],
        z [m], dy (nz) = (rho w~)_e [kg m-2 s-1], th (nz) = (-(dq*/dz)_ma mu)_e [g kg-1 m-1].
      Pe, Ce, dy, th may carry identical leading axes (e.g. bootstrap replicates); the result
      then has them too.

    For each adjacent pair (A -> B), with x_bar the mean of A and B:
        d ln P  ~ dP / P_bar                      (precipitation)
        efficiency: d(Pe/Ce) / mean(Pe/Ce)
        condensation: dCe / Ce_bar
        dynamic     = int d(dy) * (th_A + th_B) dz * 3.6 / (Ce_A + Ce_B)
        thermodyn.  = int (dy_A + dy_B) * d(th) dz * 3.6 / (Ce_A + Ce_B)
    each pair-difference being (B - A), where B has the higher r_v. Pair values divided by
    ln(RH_B/RH_A) give per-pair scaling rates; summing the pair changes and dividing by
    ln(RH_last/RH_first) gives the total scaling rate (the paper's "% per %", here per unit).
    Returns dict(rv, RH, pairs={name: (npair, ...)}, total={name: (...)}), names:
    P, eff, C, dynamic, thermo, residual (= C - dynamic - thermo; the nonlinear + sampling
    mismatch between the mean of products and the product of means).
    """
    runs = sorted(runs, key=lambda r: r['rv'])
    rv = np.array([r['rv'] for r in runs], float)
    RH = np.array([r['RH'] for r in runs], float)
    z = np.asarray(runs[0]['z'], float)
    Pe = np.stack([np.asarray(r['Pe'], float) for r in runs])
    Ce = np.stack([np.asarray(r['Ce'], float) for r in runs])
    dy = np.stack([np.asarray(r['dy'], float) for r in runs])
    th = np.stack([np.asarray(r['th'], float) for r in runs])
    eff = Pe / Ce
    lnrh = np.log(RH[1:] / RH[:-1]).reshape((-1,) + (1,) * (Pe.ndim - 1))

    def frac(x):
        return 2.0 * (x[1:] - x[:-1]) / (x[1:] + x[:-1])

    den = Ce[1:] + Ce[:-1]
    dyn = np.trapezoid((dy[1:] - dy[:-1]) * (th[1:] + th[:-1]), z, axis=-1) * G_M2_S_TO_MM_HR / den
    thm = np.trapezoid((dy[1:] + dy[:-1]) * (th[1:] - th[:-1]), z, axis=-1) * G_M2_S_TO_MM_HR / den
    changes = dict(P=frac(Pe), eff=frac(eff), C=frac(Ce), dynamic=dyn, thermo=thm)
    changes['residual'] = changes['C'] - dyn - thm
    pairs = {k: v / lnrh for k, v in changes.items()}
    tot = np.log(RH[-1] / RH[0])
    total = {k: v.sum(axis=0) / tot for k, v in changes.items()}
    return dict(rv=rv, RH=RH, pairs=pairs, total=total)


# ---------------------------------------------------------------------------- glue
def run_dir(experiment, rv):
    """<experiment>/stage2_rv<rv>, the naming used by samheat.stages."""
    return Path(experiment) / f'stage2_rv{rv:g}'


def analyze_run(rdir, rv, last_days=30, q=99.9, block=8, n_boot=100, seed=0, zt=ZT,
                use_stat_rho=True):
    """Everything the notebook needs from one stage-2 run directory.

    Reads OUT_STAT (converted .nc), OUT_2D (.2Dbin, 'Prec') and OUT_3D (.bin3D) over the last
    `last_days` days, and returns a plain dict (picklable): rv, table1, profiles, Pe, Pe_boot
    (dist/p05/p95), Ce, Ce_boot, z, dy, th, dy_boot, th_boot, RH (mean near-surface), zlcl.
    """
    from ..io.sam2d import open_run_2d, prec_mm_hr
    from ..io.sam3d import open_run_3d
    from ..io.stat import open_stat

    rdir = Path(rdir)
    stat = open_stat(rdir)          # converts .stat -> .nc with stat2nc if needed
    if stat is None:
        raise FileNotFoundError(f'no STAT .nc in {rdir / "OUT_STAT"}')
    t1 = float(stat['time'][-1])
    d2 = open_run_2d(rdir, ['Prec'], tmin=t1 - last_days)
    tab = table1(stat, last_days, prec2d=d2)
    prof = mean_profiles(stat, last_days)
    P = prec_mm_hr(d2).values
    Pb = block_bootstrap(P, block, n_boot, q, seed)

    d3 = open_run_3d(rdir, ['TABS', 'QV', 'W'], tmin=t1 - last_days, zmax=zt + 2e3)
    rho = None
    if use_stat_rho and 'RHO' in stat:
        rho = _last_days(stat, last_days)['RHO'].mean('time')
    cond = condensation_rate(d3, zt, rho)
    ex = extreme_profiles(d3, cond, q, zt, rho)
    cb = bootstrap_extreme_profiles(ex['cand'], block, n_boot, seed)
    return dict(rv=float(rv), table1=tab, RH=tab['RH'], zlcl=float(prof['zlcl']),
                profiles=dict(z=prof['z'].values, RELH=prof['RELH'].values,
                              THETA=prof['THETA'].values),
                Pe=Pb['estimate'], Pe_boot=Pb, Ce=ex['Ce'], Ce_boot=cb['Ce'], z=ex['z'],
                dy=ex['dy'], th=ex['th'], dy_boot=cb['dy'], th_boot=cb['th'],
                mean_C=float(cond['C'].mean()), n_snapshots=int(d2.sizes['time']))


def load_summaries(experiment, rv_list, cache=True, **kw):
    """analyze_run for every r_v of an experiment; results cached in <experiment>/analysis_cache/
    (delete the pickle to recompute). Returns a list of dicts sorted by rv."""
    experiment = Path(experiment)
    cdir = experiment / 'analysis_cache'
    ledger_file = experiment / 'ledger.json'
    if not experiment.exists():
        raise FileNotFoundError(f'{experiment} does not exist: check EXPERIMENT in the config cell')
    ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else {}
    out = []
    for rv in rv_list:
        stage = ledger.get(f'rv{rv:g}', {}).get('stage')
        if ledger and stage != 'done':
            # never analyse (or cache) a run whose Stage 2 has not finished
            print(f'rv={rv:g}: not ready (stage {stage!r}); skipped')
            continue
        f = cdir / f'vdD_rv{rv:g}.pkl'
        if cache and f.exists():
            out.append(pickle.loads(f.read_bytes()))
            continue
        s = analyze_run(run_dir(experiment, rv), rv, **kw)
        if cache:
            cdir.mkdir(exist_ok=True)
            f.write_bytes(pickle.dumps(s))
        out.append(s)
    return sorted(out, key=lambda s: s['rv'])


# ---------------------------------------------------------------------------- sub-period robustness
def subperiod_windows(t_end, last_days=30, n_sub=3):
    """n_sub equal consecutive windows [(t0, t1), ...] [day] covering (t_end - last_days, t_end].
    Each window is half-open (t0, t1] except the first, which includes t0 -- i.e. together
    they hold exactly the snapshots analyze_run uses (t >= t_end - last_days), none twice."""
    edges = t_end - last_days + last_days * np.arange(n_sub + 1) / n_sub
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]


def _in_window(t, i, w, n_sub):
    """Boolean mask of times belonging to window i = (t0, t1)."""
    t = np.asarray(t, float)
    lo = (t >= w[0] - 1e-6) if i == 0 else (t > w[0] + 1e-6)
    return lo & (t <= w[1] + 1e-6)


def _subperiod_run(rdir, rv, windows, q=99.9, zt=ZT, use_stat_rho=True):
    """P_e, C_e, extreme profiles and mean RH of one run for each (t0, t1) window.
    Returns a list of dicts (rv, RH, Pe, Ce, z, dy, th, n_snapshots), one per window."""
    from ..io.sam2d import open_run_2d, prec_mm_hr
    from ..io.sam3d import open_run_3d
    from ..io.stat import open_stat

    rdir = Path(rdir)
    stat = open_stat(rdir)
    if stat is None:
        raise FileNotFoundError(f'no STAT .nc in {rdir / "OUT_STAT"}')
    t_end = windows[-1][1]
    d2 = open_run_2d(rdir, ['Prec'], tmin=windows[0][0] - 1e-6, tmax=t_end + 1e-6)
    d3 = open_run_3d(rdir, ['TABS', 'QV', 'W'], tmin=windows[0][0] - 1e-6, tmax=t_end + 1e-6,
                     zmax=zt + 2e3)
    st = np.asarray(stat['time'].values, float)
    out = []
    n = len(windows)
    for i, w in enumerate(windows):
        m2 = _in_window(d2['time'].values, i, w, n)
        m3 = _in_window(d3['time'].values, i, w, n)
        ms = _in_window(st, i, w, n)
        if not m2.any() or not m3.any() or not ms.any():
            raise ValueError(f'{rdir}: no 2D/3D/STAT snapshots in window {w}')
        P = prec_mm_hr(d2.isel(time=np.nonzero(m2)[0])).values
        sub3 = d3.isel(time=np.nonzero(m3)[0])
        ss = stat.isel(time=np.nonzero(ms)[0])
        rho = ss['RHO'].mean('time') if (use_stat_rho and 'RHO' in stat) else None
        cond = condensation_rate(sub3, zt, rho)
        ex = extreme_profiles(sub3, cond, q, zt, rho)
        out.append(dict(rv=float(rv), RH=float(ss['RELH'].isel(z=0).mean()),
                        Pe=extreme_mean(P, q), Ce=ex['Ce'], z=ex['z'], dy=ex['dy'], th=ex['th'],
                        n_snapshots=int(m3.sum()), t0=w[0], t1=w[1]))
    return out


def decomposition_table(per_window):
    """Tidy DataFrame from {period_label: [run dicts]}: runs `decompose` per period.
    Columns: period, scope ('total' or 'rv_a->rv_b'), quantity (P, eff, C, dynamic, thermo,
    residual), value (scaling rate per unit fractional RH change)."""
    import pandas as pd
    rows = []
    for label, runs in per_window.items():
        r = decompose(runs)
        for k, v in r['total'].items():
            rows.append(dict(period=label, scope='total', quantity=k, value=float(v)))
        for k, v in r['pairs'].items():
            for j, val in enumerate(np.atleast_1d(v)):
                rows.append(dict(period=label, scope=f'{r["rv"][j]:g}->{r["rv"][j + 1]:g}',
                                 quantity=k, value=float(val)))
    return pd.DataFrame(rows)


def subperiod_summaries(experiment, rv_list, n_sub=3, last_days=30, q=99.9, zt=ZT,
                        use_stat_rho=True, cache=True):
    """CHECK A: is the dynamic term just sampling noise?

    Splits the last `last_days` days of every run into n_sub equal consecutive sub-periods
    (e.g. days 30-40, 40-50, 50-60), computes P_e, C_e and the extreme profiles in each, and
    runs `decompose` on every sub-period (across the r_v series). Per-run results are cached
    in <experiment>/analysis_cache/subperiods_rv*_n*_d*_q*.pkl (delete to recompute).

    Returns (df, per_run): `df` is the tidy table from `decomposition_table`
    (period 'sub1'.., scope, quantity, value); `per_run` has one row per (period, rv) with
    t0, t1, RH, Pe, Ce, eff = Pe/Ce, n_snapshots. Runs whose ledger stage is not 'done' are skipped.
    """
    import pandas as pd
    experiment = Path(experiment)
    if not experiment.exists():
        raise FileNotFoundError(f'{experiment} does not exist: check EXPERIMENT in the config cell')
    ledger_file = experiment / 'ledger.json'
    ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else {}
    cdir = experiment / 'analysis_cache'
    per_rv = {}
    for rv in rv_list:
        stage = ledger.get(f'rv{rv:g}', {}).get('stage')
        if ledger and stage != 'done':
            print(f'rv={rv:g}: not ready (stage {stage!r}); skipped')
            continue
        f = cdir / f'subperiods_rv{rv:g}_n{n_sub}_d{last_days:g}_q{q:g}.pkl'
        if cache and f.exists():
            per_rv[rv] = pickle.loads(f.read_bytes())
            continue
        rdir = run_dir(experiment, rv)
        from ..io.stat import open_stat
        stat = open_stat(rdir)
        if stat is None:
            raise FileNotFoundError(f'no STAT .nc in {rdir / "OUT_STAT"}')
        windows = subperiod_windows(float(stat['time'][-1]), last_days, n_sub)
        per_rv[rv] = _subperiod_run(rdir, rv, windows, q, zt, use_stat_rho)
        if cache:
            cdir.mkdir(exist_ok=True)
            f.write_bytes(pickle.dumps(per_rv[rv]))
    if len(per_rv) < 2:
        raise ValueError('need at least two finished runs for a decomposition')
    n = len(next(iter(per_rv.values())))
    per_window = {f'sub{i + 1}': [per_rv[rv][i] for rv in per_rv] for i in range(n)}
    df = decomposition_table(per_window)
    runs = pd.DataFrame([dict(period=lab, rv=r['rv'], t0=r['t0'], t1=r['t1'], RH=r['RH'],
                              Pe=r['Pe'], Ce=r['Ce'], eff=r['Pe'] / r['Ce'],
                              n_snapshots=r['n_snapshots'])
                         for lab, rs in per_window.items() for r in rs])
    return df, runs
