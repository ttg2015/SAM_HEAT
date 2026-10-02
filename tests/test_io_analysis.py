"""Fast tests for samheat.io.sam2d / sam3d and samheat.analysis.repro on tiny synthetic data.

The binary files are written here in the exact Fortran sequential-unformatted layout
(write_fields2D.f90 / write_fields3D.f90 + compress3D.f90 savebin branch), independently of
the readers. NOTE: the .bin3D layout has not been checked against a real SAM file.
"""
import struct

import numpy as np
import pytest

from samheat.analysis import repro
from samheat.analysis.lcl import lcl
from samheat.io import sam2d, sam3d


# ------------------------------------------------------------------ synthetic binary writers
def rec(f, payload):
    f.write(struct.pack('<i', len(payload)) + payload + struct.pack('<i', len(payload)))


def header(name, long_name, units):
    return f'{name:<8} {long_name:<80} {units:<10}'.encode()           # 8 + 1 + 80 + 1 + 10


def tiles(field, nsx, nsy):
    """Split a global (..., ny, nx) array into per-subdomain tiles in rank order."""
    ny, nx = field.shape[-2] // nsy, field.shape[-1] // nsx
    for n in range(nsx * nsy):
        j0, i0 = (n // nsx) * ny, (n % nsx) * nx
        yield field[..., j0:j0 + ny, i0:i0 + nx]


NSX, NSY, NXT, NYT = 2, 2, 4, 3          # 2 x 2 subdomains of 4 x 3 points -> global 8 x 6


def write_2dbin(path, snaps, dx=1000., dy=1000.):
    """snaps: list of (nstep, time, {name: (ny_g, nx_g) array})."""
    with open(path, 'wb') as f:
        for nstep, t, flds in snaps:
            rec(f, struct.pack('<i', nstep))
            rec(f, struct.pack('<7i', NXT, NYT, 1, NSX * NSY, NSX, NSY, len(flds)))
            rec(f, struct.pack('<f', dx))
            rec(f, struct.pack('<f', dy))
            rec(f, struct.pack('<f', t))
            for name, a in flds.items():
                rec(f, header(name, 'Long ' + name, 'mm/day'))
                for tile in tiles(a, NSX, NSY):
                    rec(f, np.ascontiguousarray(tile, '<f4').tobytes())


def write_bin3d(path, time, z, p, flds, dx=1000., dy=1000.):
    """flds: {name: (nz, ny_g, nx_g)}; tile payload is x fastest, then y, then z."""
    nz = len(z)
    with open(path, 'wb') as f:
        rec(f, struct.pack('<7i', NXT, NYT, nz, NSX * NSY, NSX, NSY, len(flds)))
        for v in z:
            rec(f, struct.pack('<f', v))
        for v in p:
            rec(f, struct.pack('<f', v))
        rec(f, struct.pack('<f', dx))
        rec(f, struct.pack('<f', dy))
        rec(f, struct.pack('<f', time))
        for name, a in flds.items():
            rec(f, header(name, 'Long ' + name, 'K'))
            for tile in tiles(a, NSX, NSY):
                rec(f, np.ascontiguousarray(tile, '<f4').tobytes())


# ------------------------------------------------------------------ io tests
def test_2dbin_roundtrip_and_units(tmp_path):
    rng = np.random.default_rng(1)
    snaps, truth = [], []
    for i in range(3):
        a = rng.random((NYT * NSY, NXT * NSX)).astype('f4') * 50
        b = rng.random((NYT * NSY, NXT * NSX)).astype('f4')
        snaps.append((100 * (i + 1), 30.0 + 0.125 * i, dict(Prec=a, SHF=b)))
        truth.append(a)
    d = tmp_path / 'run' / 'OUT_2D'
    d.mkdir(parents=True)
    write_2dbin(d / 'x.2Dbin', snaps)
    ds = sam2d.open_run_2d(tmp_path / 'run', ['Prec'])
    assert list(ds.data_vars) == ['Prec']
    assert ds['Prec'].shape == (3, 6, 8)
    np.testing.assert_array_equal(ds['Prec'].values, np.stack(truth))
    np.testing.assert_allclose(ds['time'], [30.0, 30.125, 30.25])
    assert ds['Prec'].attrs['units'] == 'mm/day'
    np.testing.assert_allclose(sam2d.prec_mm_hr(ds).values, np.stack(truth) / 24)
    assert sam2d.read_2dbin(d / 'x.2Dbin').data_vars.keys() == {'Prec', 'SHF'}
    assert float(ds['x'][1]) == 1000.


def test_bin3d_roundtrip_and_open_run(tmp_path):
    rng = np.random.default_rng(2)
    nz = 5
    z = np.array([25., 100., 250., 500., 900.])
    p = np.array([1000., 990., 975., 950., 910.])
    d = tmp_path / 'run' / 'OUT_3D'
    d.mkdir(parents=True)
    truth = {}
    for i, t in enumerate([30.25, 30.125, 30.375]):      # deliberately out of name order
        T = rng.random((nz, NYT * NSY, NXT * NSX)).astype('f4') + 280
        W = rng.standard_normal((nz, NYT * NSY, NXT * NSX)).astype('f4')
        write_bin3d(d / f'FWRCE_x_0004_{i:010d}.bin3D', t, z, p, dict(TABS=T, W=W))
        truth[t] = (T, W)
    h = sam3d.scan_bin3D(next(d.glob('*.bin3D')))
    assert (h['nx'], h['ny'], h['nz'], h['nsubs']) == (NXT, NYT, nz, 4)
    for lazy in (False, True):
        ds = sam3d.open_run_3d(tmp_path / 'run', lazy=lazy)
        assert ds['TABS'].dims == ('time', 'z', 'y', 'x') and ds['TABS'].shape == (3, nz, 6, 8)
        np.testing.assert_allclose(ds['time'], [30.125, 30.25, 30.375])
        np.testing.assert_allclose(ds['z'], z)
        np.testing.assert_allclose(ds['p'], p)
        for t, (T, W) in truth.items():
            s = ds.sel(time=t)
            np.testing.assert_array_equal(s['TABS'].values, T)
            np.testing.assert_array_equal(s['W'].values, W)
    sub = sam3d.open_run_3d(tmp_path / 'run', ['W'], tmin=30.2, zmax=300.)
    assert list(sub.data_vars) == ['W'] and sub['W'].shape == (2, 3, 6, 8)
    lev, meta = sam3d.read_bin3D_level(next(d.glob('*0000000001.bin3D')), ['TABS'], k=2)
    np.testing.assert_array_equal(lev['TABS'], truth[30.125][0][2])
    assert meta['z'] == 250.
    with pytest.raises(KeyError):
        sam3d.open_run_3d(tmp_path / 'run', ['NOPE'])


# ------------------------------------------------------------------ analysis tests
def test_extreme_mean_known_array():
    x = np.arange(1, 1001, dtype=float)              # 1..1000; 99th percentile = 990.01
    assert repro.extreme_mean(x, 99.0) == pytest.approx(np.mean(np.arange(991, 1001)))
    assert repro.extreme_mean(x, 0.0) == pytest.approx(x.mean())
    assert repro.extreme_mean(x.reshape(10, 10, 10), 100.0) == 1000.0


def test_block_bootstrap_shape_determinism_blocks():
    rng = np.random.default_rng(3)
    f = rng.exponential(1.0, (6, 16, 16))
    a = repro.block_bootstrap(f, block=8, n=20, q=95, seed=7)
    b = repro.block_bootstrap(f, block=8, n=20, q=95, seed=7)
    c = repro.block_bootstrap(f, block=8, n=20, q=95, seed=8)
    assert a['dist'].shape == (20,)
    np.testing.assert_array_equal(a['dist'], b['dist'])
    assert not np.array_equal(a['dist'], c['dist'])
    assert a['p05'] <= a['mean'] <= a['p95']
    assert a['estimate'] == repro.extreme_mean(f, 95)
    # blocks are 8x8 and 1 snapshot: every block is a contiguous 8x8 patch of one time
    blk = repro._blocks(f, 8)
    assert blk.shape == (6 * 2 * 2, 64)
    np.testing.assert_array_equal(np.sort(blk[0]), np.sort(f[0, :8, :8].ravel()))


def test_lcl_reference_values():
    # lcl.py has no worked example; check against physics. At rh=1 the LCL is at the parcel.
    assert abs(lcl(1e5, 300.0, rh=1.0)) < 1.0
    # Espy's rule: z_LCL ~ 125 m/K x (T - Td); T=300 K, RH=50 % -> Td = 288.7 K -> ~1.41 km.
    assert lcl(1e5, 300.0, rh=0.5) == pytest.approx(1419.4, abs=1.0)    # value of this code
    assert 1350 < lcl(1e5, 300.0, rh=0.5) < 1500
    assert lcl(1e5, 300.0, rh=0.4) > lcl(1e5, 300.0, rh=0.6)


def _fake_3d(nt=2, nz=30, ny=4, nx=4, w0=2.0, T0=300.0):
    import xarray as xr
    z = np.linspace(25, 14500, nz)
    p = 1000.0 * np.exp(-z / 8000.0)
    T = np.clip(T0 - 6.5e-3 * z, 200, None)
    tabs = np.broadcast_to(T[None, :, None, None], (nt, nz, ny, nx)).copy()
    qv = np.full((nt, nz, ny, nx), 5.0)
    qv[:, 0] = 0.7 * repro.qsat(T[0], p[0])        # RH = 70 % at the lowest level
    w = np.full((nt, nz, ny, nx), w0)
    w[:, :, 0, 0] = -1.0                           # a downdraft column: no condensation
    ds = xr.Dataset({k: (('time', 'z', 'y', 'x'), v) for k, v in
                     dict(TABS=tabs, QV=qv, W=w).items()},
                    coords=dict(time=np.arange(nt) * 0.125, z=z, p=('z', p),
                                y=np.arange(ny) * 1e3, x=np.arange(nx) * 1e3))
    return ds


def test_condensation_rate_matches_hand_integral():
    ds = _fake_3d()
    rho = np.full(ds.sizes['z'], 1.0)
    cond = repro.condensation_rate(ds, zt=14e3, rho=rho, chunk=1)
    z = ds['z'].values
    p = ds['p'].values
    T = ds['TABS'].values[0, :, 1, 1]
    zl = z[0] + lcl(p[0] * 1e2, T[0], rh=0.7)
    th = -repro.dq_dz(T, p) * 1e3
    k = (z <= 14e3)
    integrand = np.where(zl <= z, th, 0.0) * 2.0 * 1.0
    expect = np.trapezoid(integrand[k], z[k]) * 3.6
    assert cond['C'].shape == (2, 4, 4)
    np.testing.assert_allclose(cond['C'].values[:, 1, 1], expect, rtol=1e-5)
    np.testing.assert_allclose(cond['zlcl'].values[0, 1, 1], zl, rtol=1e-6)
    assert cond['C'].values[0, 0, 0] == 0.0 and expect > 0         # downdraft column
    ex = repro.extreme_profiles(ds, cond, q=90, zt=14e3, rho=rho, cand_frac=0.5)
    assert ex['Ce'] == pytest.approx(expect, rel=1e-5)
    assert ex['th'].shape == ex['z'].shape


def test_bootstrap_extreme_profiles_exact_vs_bruteforce():
    rng = np.random.default_rng(5)
    nt, ny, nx, nz, blk, q, n = 3, 8, 8, 4, 4, 90.0, 6
    C = rng.random((nt, ny, nx))
    dy = rng.random((nt, ny, nx, nz))
    th = rng.random((nt, ny, nx, nz))
    flat = np.argsort(-C.ravel())
    ti, yi, xi = np.unravel_index(flat, C.shape)
    k = repro._topk(C.size, q)
    cand = dict(C=C.ravel()[flat], dy=dy[ti, yi, xi], th=th[ti, yi, xi], t=ti, y=yi, x=xi,
                shape=C.shape, k=k)
    got = repro.bootstrap_extreme_profiles(cand, blk, n, seed=11)
    cb = repro._blocks(C, blk)
    dy_b = np.stack([repro._blocks(dy[..., j], blk) for j in range(nz)], -1)
    th_b = np.stack([repro._blocks(th[..., j], blk) for j in range(nz)], -1)
    for r, ids in enumerate(repro.block_draws(C.shape, blk, n, 11)):
        c = cb[ids].ravel()
        top = np.argsort(-c)[:k]
        assert got['Ce'][r] == pytest.approx(c[top].mean())
        np.testing.assert_allclose(got['dy'][r], dy_b[ids].reshape(-1, nz)[top].mean(0))
        np.testing.assert_allclose(got['th'][r], th_b[ids].reshape(-1, nz)[top].mean(0))


def _run(rv, RH, dyscale, thscale, eff):
    z = np.linspace(0, 14e3, 15)
    dy = dyscale * np.sin(np.pi * z / 14e3)
    th = thscale * np.exp(-z / 4000.0)
    Ce = np.trapezoid(dy * th, z) * 3.6
    return dict(rv=rv, RH=RH, z=z, dy=dy, th=th, Ce=Ce, Pe=eff * Ce)


def test_decompose_pure_dynamic_and_pure_thermo():
    # dynamic only: th fixed, dy shrinks, efficiency fixed -> dynamic == C == P (exactly)
    r = repro.decompose([_run(0, 75., 1.0, 1e-3, .2), _run(500, 60., .8, 1e-3, .2),
                         _run(2000, 45., .6, 1e-3, .2)])
    for k in ('P', 'C', 'dynamic'):
        np.testing.assert_allclose(r['total'][k], r['total']['C'])
    assert r['total']['thermo'] == pytest.approx(0, abs=1e-12)
    assert r['total']['eff'] == pytest.approx(0, abs=1e-12)
    assert abs(r['total']['residual']) < 1e-12
    assert r['total']['C'] > 0                                    # decreasing RH, decreasing C
    expect = (np.log(.8 / 1.0) + np.log(.6 / .8)) / np.log(45. / 75.)
    assert r['total']['C'] == pytest.approx(expect, rel=0.05)
    # thermodynamic only
    r = repro.decompose([_run(0, 75., 1.0, 1e-3, .2), _run(2000, 45., 1.0, .7e-3, .2)])
    assert r['total']['dynamic'] == pytest.approx(0, abs=1e-12)
    assert r['total']['thermo'] == pytest.approx(r['total']['C'])
    # efficiency only, with a leading bootstrap axis
    runs = [_run(0, 75., 1., 1e-3, .2), _run(2000, 45., 1., 1e-3, .1)]
    for x in runs:
        for k in ('Pe', 'Ce'):
            x[k] = np.array([x[k], x[k]])
        for k in ('dy', 'th'):
            x[k] = np.stack([x[k], x[k]])
    r = repro.decompose(runs)
    assert r['total']['eff'].shape == (2,) and r['pairs']['P'].shape == (1, 2)
    assert r['total']['C'] == pytest.approx([0, 0], abs=1e-12)
    assert r['total']['P'] == pytest.approx(r['total']['eff'])


def test_table1_and_mean_profiles_fake_stat():
    import xarray as xr
    t = np.arange(0.0, 60.0, 1.0) + 0.5
    z = np.array([25., 100., 1000.])
    stat = xr.Dataset(dict(
        TABS=(('time', 'z'), np.tile([297., 295., 288.], (len(t), 1))),
        QV=(('time', 'z'), np.tile([14., 12., 8.], (len(t), 1))),
        RELH=(('time', 'z'), np.tile([75., 70., 60.], (len(t), 1))),
        THETA=(('time', 'z'), np.tile([300., 300.5, 301.], (len(t), 1))),
        p=('z', [1000., 990., 900.]), SST=('time', np.full(len(t), 300.0)),
        PREC=('time', np.full(len(t), 2.8))), coords=dict(time=t, z=z))
    tb = repro.table1(stat, 30)
    assert tb == pytest.approx(dict(Ts=300.0, T_sfc=297.0, RH=75.0, qv=14.0, P=2.8))
    pr = repro.mean_profiles(stat, 30)
    assert float(pr['zlcl']) == pytest.approx(25 + lcl(1e5, 297.0, rh=0.75))
    assert pr['RELH'].shape == (3,)


# ------------------------------------------------------------------ sub-period check (A)
def _synthetic_run(root, rv, seed, times, nz=5):
    import xarray as xr
    rng = np.random.default_rng(seed)
    run = root / f'stage2_rv{rv:g}'
    for sub in ('OUT_2D', 'OUT_3D', 'OUT_STAT'):
        (run / sub).mkdir(parents=True)
    z = np.array([25., 100., 250., 500., 900.])
    p = np.array([1000., 990., 975., 950., 910.])
    ny, nx = NYT * NSY, NXT * NSX
    prec = {}
    snaps = []
    for i, t in enumerate(times):
        a = (rng.exponential(1.0, (ny, nx)) * 10).astype('f4')
        prec[t] = a
        snaps.append((i, t, dict(Prec=a)))
        T = (np.linspace(300, 285, nz)[:, None, None] + rng.random((nz, ny, nx))).astype('f4')
        QV = np.full((nz, ny, nx), 12., 'f4')
        QV[0] = 0.99 * repro.qsat(T[0], p[0])        # near-saturated: LCL below the top level
        W = (rng.standard_normal((nz, ny, nx)) + 1).astype('f4')
        write_bin3d(run / 'OUT_3D' / f'x_{i:010d}.bin3D', t, z, p, dict(TABS=T, QV=QV, W=W))
    write_2dbin(run / 'OUT_2D' / 'x.2Dbin', snaps)
    st = xr.Dataset(dict(RELH=(('time', 'z'), np.full((len(times), nz), 70. - rv / 100.))),
                    coords=dict(time=np.array(times), z=z))
    st.to_netcdf(run / 'OUT_STAT' / 'x.nc')
    return prec


def test_subperiod_windows_partition_snapshots():
    w = repro.subperiod_windows(60.0, 30, 3)
    assert w == [(30.0, 40.0), (40.0, 50.0), (50.0, 60.0)]
    t = np.arange(30.0, 60.0001, 0.125)                    # 3-hourly, includes both ends
    masks = [repro._in_window(t, i, x, 3) for i, x in enumerate(w)]
    assert np.sum(masks, axis=0).tolist() == [1] * len(t)  # every snapshot exactly once
    assert [int(m.sum()) for m in masks] == [81, 80, 80]


def test_subperiod_summaries_synthetic(tmp_path):
    times = [30.0 + 0.5 * i for i in range(13)]            # 30.0 .. 36.0
    precs = {rv: _synthetic_run(tmp_path, rv, 10 + int(rv), times) for rv in (0, 500)}
    df, runs = repro.subperiod_summaries(tmp_path, [0, 500], n_sub=3, last_days=6, q=90.0)
    assert sorted(runs.period.unique()) == ['sub1', 'sub2', 'sub3']
    assert runs[runs.rv == 0].n_snapshots.tolist() == [5, 4, 4]
    assert runs[runs.rv == 0].t0.tolist() == [30.0, 32.0, 34.0]
    # P_e of sub-period 2 = extreme mean of exactly the snapshots in (32, 34]
    sel = [t for t in times if 32.0 < t <= 34.0]
    expect = repro.extreme_mean(np.stack([precs[0][t] for t in sel]) / 24.0, 90.0)
    got = runs[(runs.rv == 0) & (runs.period == 'sub2')].Pe.iloc[0]
    assert got == pytest.approx(expect, rel=1e-5)
    tot = df[(df.scope == 'total') & (df.quantity == 'dynamic')]
    assert len(tot) == 3 and np.isfinite(tot.value).all()
    assert set(df.quantity) == {'P', 'eff', 'C', 'dynamic', 'thermo', 'residual'}
    assert set(df.scope) == {'total', '0->500'}
    # cached: second call must not touch the data
    assert len(list((tmp_path / 'analysis_cache').glob('subperiods_*.pkl'))) == 2
    for sub in ('OUT_2D', 'OUT_3D'):
        for f in (tmp_path / 'stage2_rv0' / sub).glob('*'):
            f.unlink()
    df2, _ = repro.subperiod_summaries(tmp_path, [0, 500], n_sub=3, last_days=6, q=90.0)
    np.testing.assert_allclose(df2.value, df.value)


# ------------------------------------------------------------------ bin3D vs converter (B)
def _mock_converter(perturb=0.0, shift_w=False):
    import xarray as xr

    def fake_sh(cmd, cwd=None, check=True, modules=(), quiet=True):
        assert 'bin3D2nc' in cmd
        f = next(__import__('pathlib').Path(cwd).glob('*.bin3D'))
        ds = sam3d.read_bin3D(f)
        if perturb:
            ds['TABS'] = ds['TABS'] * (1 + perturb)
        if shift_w:
            ds['W'] = ds['W'].isel(z=slice(1, None)).reindex(z=ds['z'])
        ds.to_netcdf(f.with_suffix('.nc'))
        return ''
    return fake_sh


def _one_bin3d(tmp_path):
    rng = np.random.default_rng(4)
    z = np.array([25., 100., 250.])
    p = np.array([1000., 990., 975.])
    f = tmp_path / 'a' / 'x.bin3D'
    f.parent.mkdir()
    shp = (3, NYT * NSY, NXT * NSX)
    write_bin3d(f, 30.5, z, p, dict(TABS=rng.random(shp).astype('f4') + 280,
                                    W=rng.standard_normal(shp).astype('f4')))
    return f


def test_compare_bin3d_with_converter_mocked(tmp_path, monkeypatch, capsys):
    from samheat.io import check3d
    f = _one_bin3d(tmp_path)
    monkeypatch.setattr('samheat.slurm.sh', _mock_converter())
    df = check3d.compare_bin3d_with_converter(f)
    assert df.attrs['passed'] and 'PASS' in capsys.readouterr().out
    assert set(df[df.kind == 'field'].variable) == {'TABS', 'W'}
    assert set(df[df.kind == 'coord'].variable) == {'x', 'y', 'z', 'p', 'time'}
    assert df.max_abs.dropna().max() == 0
    monkeypatch.setattr('samheat.slurm.sh', _mock_converter(perturb=1e-3))
    df = check3d.compare_bin3d_with_converter(f, verbose=False)
    assert not df.attrs['passed']
    assert not df.set_index('variable').loc['TABS', 'ok'] and df.set_index('variable').loc['W', 'ok']
    monkeypatch.setattr('samheat.slurm.sh', _mock_converter(shift_w=True))
    assert not check3d.compare_bin3d_with_converter(f, verbose=False).attrs['passed']
