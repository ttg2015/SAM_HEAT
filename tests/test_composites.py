"""Composite profiles (samheat.analysis.composites) on synthetic SAM output."""
import numpy as np
import pytest
import xarray as xr

from samheat.analysis import composites as C
from test_io_analysis import NSX, NSY, NXT, NYT, write_bin3d

Z = np.array([37.5, 121., 208., 300., 900.])
P = np.array([1000., 990., 980., 970., 910.])


def _fields(rng, nz=5, ny=NYT * NSY, nx=NXT * NSX):
    sh = (nz, ny, nx)
    return dict(W=rng.standard_normal(sh).astype('f4'), PP=rng.standard_normal(sh).astype('f4'),
                TABS=(np.linspace(300, 290, nz)[:, None, None] + rng.random(sh)).astype('f4'),
                QV=np.full(sh, 12., 'f4'), QN=np.zeros(sh, 'f4'), QP=np.zeros(sh, 'f4'))


def _dataset(nt=3, p_as_variable=False, seed=0):
    rng = np.random.default_rng(seed)
    snaps = [_fields(rng) for _ in range(nt)]
    dv = {k: (('time', 'z', 'y', 'x'), np.stack([s[k] for s in snaps])) for k in snaps[0]}
    ny, nx = snaps[0]['W'].shape[1:]
    coords = dict(time=30 + np.arange(nt) / 8., z=Z, y=np.arange(ny) * 1e3, x=np.arange(nx) * 1e3)
    if p_as_variable:                       # what open_mfdataset gives for bin3D2nc output
        dv['p'] = (('time', 'z'), np.tile(P, (nt, 1)))
    else:                                   # what samheat.io.sam3d gives
        coords['p'] = ('z', P)
    return xr.Dataset(dv, coords=coords)


def symmetric(f, z, k=1, X=0.5):
    w = f['W'][k]
    return dict(updraft=w > X, downdraft=w < -X, intermediate=np.abs(w) <= X)


@pytest.mark.parametrize('p_as_variable', [False, True])
def test_composite_matches_direct_means(p_as_variable):
    ds = _dataset(p_as_variable=p_as_variable)
    c = C.composite(ds, symmetric, progress=0)
    assert list(c.group.values) == ['updraft', 'downdraft', 'intermediate', 'all']
    assert float(c.fraction.sel(group=['updraft', 'downdraft', 'intermediate']).sum()) == pytest.approx(1.0)
    # 'all' is the plain domain mean, the others are conditional means
    assert np.allclose(c.TABS.sel(group='all'), ds.TABS.astype(float).mean(('time', 'y', 'x')))
    W1 = ds.W.isel(z=1)
    m = W1 > 0.5
    assert np.allclose(c.TABS.sel(group='updraft'), ds.TABS.where(m).mean(('time', 'y', 'x')))
    assert float(c.W.sel(group='updraft').isel(z=1)) > 0.5
    assert {'MSE', 'MSE_frozen', 'S_LI', 'THETA_E'} <= set(c.data_vars)


def test_derived_fields_reference_values():
    f = dict(TABS=np.full((1, 1, 1), 300.), QV=np.full((1, 1, 1), 15.), PP=np.zeros((1, 1, 1)))
    d = C.derived_fields(f, np.array([1000.]), np.array([0.]))
    assert d['MSE'].item() == pytest.approx(1.005 * 300 + 2.5 * 15)
    assert d['MSE_frozen'].item() == pytest.approx((1004 * 300 + 2.5104e6 * 0.015) / 1000)
    assert d['THETA_E'].item() == pytest.approx(344.1, abs=0.2)       # Bolton at 300 K, 15 g/kg, 1000 hPa


def test_group_names_must_be_constant():
    ds = _dataset()
    calls = iter(range(10))
    bad = lambda f, z: {'a': f['W'][1] > 0} if next(calls) == 0 else {'b': f['W'][1] > 0}
    with pytest.raises(ValueError, match='groups'):
        C.composite(ds, bad, progress=0)


def test_experiment_cache_and_key(tmp_path):
    rng = np.random.default_rng(3)
    for rv in (0, 500):
        out = tmp_path / f'stage2_rv{rv}' / 'OUT_3D'
        out.mkdir(parents=True)
        for i in range(2):
            write_bin3d(out / f'x_{i:010d}.bin3D', 30 + i / 8, Z, P, _fields(rng))
    r1 = C.composite_experiment(tmp_path, [0, 500], symmetric, 'sym', dict(k=1, X=0.5))
    assert r1.TABS.dims == ('rv', 'group', 'z') and list(r1.rv.values) == [0., 500.]
    files = sorted((tmp_path / 'analysis_cache').glob('composite_sym_*'))
    assert len(files) == 2
    r2 = C.composite_experiment(tmp_path, [0, 500], symmetric, 'sym', dict(k=1, X=0.5))
    xr.testing.assert_allclose(r1.TABS, r2.TABS)
    # a different threshold is a different cache entry
    C.composite_experiment(tmp_path, [0], symmetric, 'sym', dict(k=1, X=0.8))
    assert len(list((tmp_path / 'analysis_cache').glob('composite_sym_*'))) == 3


def test_saturated_fields_and_fit_xlim():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = np.full((1, 1, 1), 300.)
    f = dict(TABS=T, QV=np.full((1, 1, 1), 15.), PP=np.zeros((1, 1, 1)))
    d = C.derived_fields(f, np.array([1000.]), np.array([0.]))
    from samheat.analysis.thermo_SAM import qsat
    qs = qsat(300., 1000.)
    assert 22 < qs < 23                                                  # ~22.4 g/kg at 300 K, 1000 hPa
    assert d['MSE_SAT'].item() == pytest.approx(1.005 * 300 + 2.5 * qs)
    assert d['THETA_ES'].item() > d['THETA_E'].item()                    # subsaturated air: theta_e < theta_e*
    # saturated air: theta_e == theta_es
    f['QV'] = np.full((1, 1, 1), qs)
    d = C.derived_fields(f, np.array([1000.]), np.array([0.]))
    assert d['THETA_E'].item() == pytest.approx(d['THETA_ES'].item(), abs=0.3)
    fig, ax = plt.subplots()
    z = np.array([0., 1000., 20000.])
    C.fit_xlim(ax, [np.array([330., 320., 480.]), np.array([340., 325., 490.])], z, zmax=16000)
    lo, hi = ax.get_xlim()
    assert 318 < lo < 320 and 340 < hi < 342
    plt.close(fig)
