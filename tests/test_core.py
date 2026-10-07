"""Fast tests (about a second) of the namelist, convergence and the guarded state machine."""
import numpy as np
import pytest

from samheat import namelist, presets
from samheat.convergence import tabs_integral
from samheat.paths import Site
from samheat.presets import DAY, NEVER
from samheat.stages import Experiment
from fake_hpc import FakeSlurm, FakeStat, P, T_PROFILE


# ---------------------------------------------------------------- namelist -------------
def test_roundtrip(tmp_path):
    p = presets.paper_base()
    namelist.write_prm(tmp_path / 'prm', p)
    assert namelist.read_prm(tmp_path / 'prm') == p


def test_unknown_key_raises():
    with pytest.raises(namelist.NamelistError):
        namelist.merge(presets.paper_base(), {'doperpetaul': True})


def test_keys_are_case_insensitive():
    assert namelist.merge({}, {'ocean': False}) == {'OCEAN': False}


def test_perpetual_guard(tmp_path):
    p = presets.paper_base()
    p['doperpetual'] = False
    with pytest.raises(namelist.NamelistError, match='perpetual'):
        namelist.write_prm(tmp_path / 'prm', p)
    namelist.write_prm(tmp_path / 'prm', p, allow_diurnal=True)


# ---------------------------------------------------------------- convergence ----------
def test_tabs_integral_order_and_value():
    T = np.full(64, 250.0)
    assert tabs_integral(400, 600, P, T) == pytest.approx(250.0)
    assert tabs_integral(600, 400, P, T_PROFILE) == tabs_integral(400, 600, P, T_PROFILE)


# ---------------------------------------------------------------- state machine --------
def make(tmp_path, behaviour=None, offset=None, **cfg):
    site = Site(work_root=tmp_path)
    sched = FakeSlurm(behaviour)
    exp = Experiment(dict(rv_list=[0, 500], **cfg), site=site, scheduler=sched,
                     stat_loader=FakeStat(**(offset or {})), link_exe=False)
    return exp, sched


def stages(exp):
    return {k: e['stage'] for k, e in exp.ledger.load().items() if k != '_meta'}


def test_happy_path(tmp_path):
    exp, sched = make(tmp_path)
    exp.start()
    exp.advance()                                  # ocean done -> both Stage 1 submitted
    assert stages(exp) == {'ocean': 'done', 'rv0': 1, 'rv500': 1}
    exp.advance()                                  # converged -> Stage 2
    assert stages(exp) == {'ocean': 'done', 'rv0': 2, 'rv500': 2}
    exp.advance()
    assert set(stages(exp).values()) == {'done'}
    prm = namelist.read_prm(exp.root / 'stage2_rv500' / 'FWRCE' / 'prm')
    assert prm['doperpetual'] and prm['solar_constant'] == 565.0
    assert prm['nsave3Dstart'] == 30 * DAY and prm['nsave3Dend'] == NEVER and prm['nsave3D'] == 1080
    assert prm['dodynamicocean'] is False and prm['r_stomata'] == 500.0
    s1 = namelist.read_prm(exp.root / 'stage1_rv500' / 'FWRCE' / 'prm')
    assert s1['sst_ft'] and s1['dodynamicocean'] and s1['tau_sst'] == 311040.0
    assert s1['tabs_s'] == pytest.approx(303.9, abs=0.05)          # paper-table first guess
    # Stage 2 starts from the ocean-reference sounding, not the template one
    assert (exp.root / 'stage2_rv500' / 'FWRCE' / 'snd').read_text() == exp.snd_ref.read_text()
    assert len(sched.submissions) == 5


def test_never_converging_stops(tmp_path):
    exp, sched = make(tmp_path, offset=dict(offset={'stage1_rv500': 0.5}),
                      stage1=dict(max_days=70))
    exp.start()
    for _ in range(20):
        exp.advance()
    st = stages(exp)
    assert st['rv500'] == 'failed' and st['rv0'] == 'done'
    assert sched.submissions.count('stage1_rv500') == 4          # 40, 50, 60, 70 days


def test_walltime_resubmits_then_gives_up(tmp_path):
    exp, sched = make(tmp_path, behaviour={'ocean': 'walltime'}, max_submits=3)
    exp.start()
    for _ in range(10):
        exp.advance()
    assert stages(exp)['ocean'] == 'failed'
    assert sched.submissions.count('ocean') == 3


def test_stuck_fails_after_two(tmp_path):
    exp, sched = make(tmp_path, behaviour={'ocean': 'stuck'})
    exp.start()
    for _ in range(10):
        exp.advance()
    assert stages(exp)['ocean'] == 'failed'
    assert sched.submissions.count('ocean') == 2


def test_stop_file(tmp_path):
    exp, sched = make(tmp_path)
    exp.start()
    (exp.root / 'STOP').touch()
    exp.advance()
    assert stages(exp)['ocean'] == 'ocean'


def test_nan_is_corrupted(tmp_path):
    exp, _ = make(tmp_path, offset=dict(nan={'ocean'}))
    exp.start(); exp.advance()
    assert stages(exp)['ocean'] == 'corrupted'


def test_diurnal_output_is_caught(tmp_path):
    exp, _ = make(tmp_path, offset=dict(diurnal={'ocean'}))
    exp.start(); exp.advance()
    assert stages(exp)['ocean'] == 'failed'


def test_physics_change_refused(tmp_path):
    exp, _ = make(tmp_path)
    exp.start()
    exp2, _ = make(tmp_path, ts_ocean=301.0)
    with pytest.raises(RuntimeError, match='new EXPERIMENT name'):
        exp2.advance()
    exp3, _ = make(tmp_path, max_submits=9)       # non-physics settings may change
    exp3.advance()


def test_auto_advance_script(tmp_path):
    exp, _ = make(tmp_path, auto_advance=True)
    exp.start()
    script = (exp.root / 'ocean' / 'run.slurm').read_text()
    assert 'samheat.advance' in script and 'STOP' in script and '--from-job $SLURM_JOB_ID' in script


def test_cold_tropopause_is_accepted():
    """A ~194 K cold point (seen in the first perpetual-sun ocean run) is physical."""
    import xarray as xr
    from samheat.sounding import snd_from_stat
    T = T_PROFILE.copy(); T[40] = 193.9
    st = xr.Dataset(dict(TABS=(('time', 'z'), np.tile(T, (24, 1))), QV=(('time', 'z'), np.ones((24, 64))),
                         p=('z', P)), coords=dict(time=np.arange(24) / 2.4, z=np.arange(64.)))
    assert snd_from_stat(st, 5, 1006.91)['tp'].shape == (64,)


def test_grid_lever_and_old_experiments(tmp_path):
    exp, _ = make(tmp_path, grid='grd_vdd2025', name='vddgrid')
    exp.start()
    grd = (exp.root / 'ocean' / 'FWRCE' / 'grd').read_text().split()
    assert float(grd[0]) == 37.5
    # an experiment created before the GRID lever existed (no 'grid' in experiment.json) still opens
    import json
    old, _ = make(tmp_path, name='oldexp')
    old.start()
    cfg = json.loads((old.root / 'experiment.json').read_text()); cfg.pop('grid')
    (old.root / 'experiment.json').write_text(json.dumps(cfg))
    old.advance()
    with pytest.raises(FileNotFoundError):
        make(tmp_path, grid='no_such_grid')


def test_ts_guesses_and_rv_list_may_change_mid_experiment(tmp_path):
    exp, sched = make(tmp_path, name='grow')
    exp.start(); exp.advance(); exp.advance(); exp.advance()          # rv0, rv500 done
    more = dict(rv_list=[0, 500, 100000], name='grow',
                stage1=dict(ts_offset={0: 0.0, 500: 3.8, 100000: 14.7}))
    exp2 = Experiment(more, site=exp.site, scheduler=sched, stat_loader=FakeStat(), link_exe=False)
    exp2.advance()                                                    # adds rv100000 and launches it
    e = exp2.ledger.load()['rv100000']
    assert e['stage'] == 1 and e['ts'] == pytest.approx(314.7)
    with pytest.raises(RuntimeError, match='new EXPERIMENT name'):   # real physics still locked
        Experiment(dict(more, stage1=dict(tau_sst=1e5)), site=exp.site, scheduler=sched,
                   stat_loader=FakeStat(), link_exe=False).advance()


def test_wind_experiment_borrows_calm_reference(tmp_path):
    from samheat.sounding import read_snd
    base, sched = make(tmp_path, name='calm', grid='grd_vdd2025')
    base.start(); base.advance()                                      # calm ocean done -> T_ref
    T_ref = base.ledger.load()['_meta']['T_ref']
    n_before = len(sched.submissions)
    windy = Experiment(dict(name='calm_U5', rv_list=[0, 500], grid='grd_vdd2025', reference_from='calm',
                            wind=dict(profile='uniform', U=5.0, tauls=7200.)),
                       site=base.site, scheduler=sched, stat_loader=FakeStat(), link_exe=False)
    windy.start()
    st = windy.ledger.load()
    assert st['_meta']['T_ref'] == T_ref and st['ocean']['stage'] == 'done'
    assert 'ocean' not in sched.submissions[n_before:]                # no new ocean run
    assert sorted(sched.submissions[n_before:]) == ['stage1_rv0', 'stage1_rv500']
    prm = namelist.read_prm(windy.root / 'stage1_rv500' / 'FWRCE' / 'prm')
    assert prm['donudging_uv'] is True and prm['tauls'] == 7200.0
    assert prm['nudging_uv_z1'] == -1.0 and prm['doperpetual'] is True
    snd = read_snd(windy.root / 'stage1_rv500' / 'FWRCE' / 'snd')
    assert np.allclose(snd['u'], 5.0) and np.allclose(snd['v'], 0.0)
    # the same thermodynamic profile as the calm reference
    calm = read_snd(base.snd_ref)
    assert np.allclose(snd['tp'], calm['tp']) and np.allclose(snd['q'], calm['q'])
    assert float((windy.root / 'stage1_rv500' / 'FWRCE' / 'grd').read_text().split()[0]) == 37.5
    windy.advance(); windy.advance()                                  # Stage 1 -> 2 -> done
    assert {k: e['stage'] for k, e in windy.ledger.load().items() if k != '_meta'} == \
        {'ocean': 'done', 'rv0': 'done', 'rv500': 'done'}


def test_borrow_refuses_mismatched_reference(tmp_path):
    base, sched = make(tmp_path, name='calm2')
    base.start(); base.advance()
    other = Experiment(dict(name='bad', rv_list=[0], grid='grd_vdd2025', reference_from='calm2',
                            wind=dict(U=5.0)), site=base.site, scheduler=sched, stat_loader=FakeStat(), link_exe=False)
    with pytest.raises(RuntimeError, match='grid'):
        other.start()


def test_linear_wind_profile():
    from samheat.sounding import wind_profile
    u, v = wind_profile(np.array([0., 500., 1000., 3000.]), dict(profile='linear', U=10., z_top=1000.))
    assert np.allclose(u, [0, 5, 10, 10]) and np.allclose(v, 0)
    assert np.allclose(wind_profile(np.array([0., 1.]), None)[0], 0)
