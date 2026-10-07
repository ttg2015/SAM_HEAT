"""Level-wise masks, the masks library and the explore one-liners on synthetic data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from samheat.analysis import composites as C, explore as X, masks as M
from test_composites import P, Z, _dataset, _fields
from test_io_analysis import write_bin3d


def _with_clouds(seed=0):
    ds = _dataset(seed=seed)
    rng = np.random.default_rng(seed)
    qn = np.where(rng.random(ds.QN.shape) < 0.2, 0.5, 0.0).astype('f4')   # 20 % cloudy points
    return ds.assign(QN=(ds.QN.dims, qn))


def test_level_mask_matches_direct_conditional_mean():
    ds = _with_clouds()
    c = C.composite(ds, M.cloud, dict(qn_min=0.01), progress=0)
    cloudy = ds.QN > 0.01
    expect = ds.TABS.astype(float).where(cloudy).mean(('time', 'y', 'x'))
    np.testing.assert_allclose(c.TABS.sel(group='cloud'), expect)
    np.testing.assert_allclose(c.fraction_z.sel(group='cloud'), cloudy.mean(('time', 'y', 'x')))
    assert c.fraction_z.dims == ('group', 'z')
    np.testing.assert_allclose(c.fraction_z.sel(group=['cloud', 'clear']).sum('group'), 1.0)


def test_column_masks_keep_constant_fraction_profile():
    ds = _dataset()
    c = C.composite(ds, M.symmetric, dict(k=1, X=0.5), progress=0)
    fz = c.fraction_z.sel(group='updraft').values
    assert np.allclose(fz, fz[0]) and np.isclose(float(c.fraction.sel(group='updraft')), fz[0])


def test_every_library_mask_runs():
    ds = _with_clouds()
    for name, fn in {**M.COLUMN_MASKS, **M.LEVEL_MASKS}.items():
        c = C.composite(ds.isel(time=[0]), fn, progress=0)
        assert 'all' in c.group.values, name


def test_sam_cloud_threshold_is_at_most_001():
    ds = _dataset()
    s = ds.isel(time=0)
    f = {v: s[v].values.astype(float) for v in ds.data_vars}
    f.update(C.derived_fields(f, P, Z))
    thr = M.sam_cloud_threshold(f, Z)
    assert thr.shape == (len(Z), 1, 1) and (thr <= 0.01).all() and (thr > 0).all()


def test_load_and_plots(tmp_path, monkeypatch):
    from samheat.paths import SITE
    monkeypatch.setattr(SITE, 'work_root', tmp_path)
    rng = np.random.default_rng(2)
    for rv in (0, 2000):
        out = tmp_path / 'exp' / f'stage2_rv{rv}' / 'OUT_3D'
        out.mkdir(parents=True)
        for i in range(2):
            f = _fields(rng)
            f['QN'] = np.where(rng.random(f['QN'].shape) < .2, .5, 0.).astype('f4')
            write_bin3d(out / f'x_{i:010d}.bin3D', 30 + i / 8, Z, P, f)
    D = X.load('exp', [0, 2000], M.cloud, tmin=None, tmax=None)
    assert D.TABS.dims == ('rv', 'group', 'z')
    ax = X.plot_profiles(D, 'MSE', group='cloud')
    assert len(ax.lines) == 4                                   # 2 runs x (MSE + dashed MSE_SAT)
    ax = X.plot_profiles(D, 'TABS', by='group', rv=0, anomaly=True)
    assert len([l for l in ax.lines if l.get_label() and not l.get_label().startswith('_')]) == 2   # cloud, clear (not 'all')
    fig = X.plot_grid(D, ['W', 'QN', 'MSE', 'THETA_E', 'TABS'], group='clear')
    assert sum(a.get_visible() for a in fig.axes) == 5
    plt.close('all')
