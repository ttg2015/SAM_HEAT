"""The experiment: ocean reference -> Stage 1 (Ts relaxation) -> Stage 2 (fixed Ts).

    <work_root>/<name>/
        experiment.json        the configuration (fixed once the first job is submitted)
        ledger.json            one entry per run + 'ocean' + '_meta', with full history
        STOP                   create this file to halt ALL automatic steps
        reference/snd_ref      equilibrium sounding from the ocean run (last avg_days)
        ocean/                 r_v = 0, fixed Ts = ts_ocean, ocean.days
        stage1_rv<rv>/         dodynamicocean + sst_ft: Ts relaxes until 600-400 hPa T = T_ref
        stage2_rv<rv>/         fixed Ts, 2D/3D output, the analysis run

advance() is the only thing that moves runs forward. It is safe to call any time, by hand
or (AUTO_ADVANCE) by a job when it ends. Guards against runaway loops, all in this file:
    * max_submits jobs per run per stage, then 'failed'
    * Stage 1 never runs past stage1.max_days, then 'failed'
    * a resubmission that makes no progress twice in a row -> 'failed'
    * NaN in the STAT file -> 'corrupted' (never resubmitted)
    * a STOP file in the experiment directory -> advance() does nothing
"""
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import convergence, namelist, presets
from .checks import perpetual_from_stat
from .io.stat import nan_fields, open_stat
from .ledger import Ledger, note
from .logs import finished_cleanly, health, last_nstep
from .paths import SITE
from .presets import DAY, NEVER
from .slurm import Slurm, advance_command, batch_script
from .sounding import read_snd, snd_from_stat, with_wind, write_snd
from .staging import CASE_FILES, stage_run, update_prm

DEFAULTS = dict(
    name='vdd2025_repro',
    rv_list=[0, 200, 500, 1000, 2000],
    ts_ocean=300.0,
    grid='grd',           # file in case/FWRCE: 'grd' = stock RCE grid (25 m), 'grd_vdd2025' = van der Drift's (37.5 m)
    ocean=dict(days=50, avg_days=10, snd='snd_vdd300K'),
    stage1=dict(days=40, extend_days=10, window_days=10, tol_K=0.1, max_days=100,
                tau_sst=311040.0, p_low=400.0, p_upp=600.0,
                # first guess of Ts - ts_ocean (paper Table 1); other r_v interpolated
                ts_offset={0: 0.0, 200: 1.9, 500: 3.9, 1000: 6.0, 2000: 8.3}),
    stage2=dict(days=60, out_start_day=30, out_every_hr=3, save2D=True, save3D=True),
    overrides={},
    allow_diurnal=False,
    auto_advance=False,
    dry_run=False,
    max_submits=8,
    # mean wind: None = calm (no nudging). dict(profile='uniform', U=5., V=0., tauls=7200.) nudges the
    # domain-mean wind toward the target with time scale tauls [s] (Muller 2013: 2 h), whole column
    # unless z1/z2 [m] are given; 'linear' adds z_top/u_bottom (see sounding.wind_profile).
    wind=None,
    # name (or path) of a finished experiment whose ocean reference (T_ref, snd_ref) is reused
    # instead of running a new ocean run: the free troposphere is held at THAT reference.
    reference_from=None,
)
# Settings that may change after the experiment has started (they affect no physics).
MUTABLE = {'auto_advance', 'dry_run', 'max_submits', 'rv_list'}
TERMINAL = ('done', 'failed', 'corrupted')


def _without_guesses(key, value):
    """Stage-1 first guesses of Ts (stage1.ts_offset) change only where Stage 1 starts, not the
    converged state, so they may be edited mid-experiment (e.g. to add very dry r_v values)."""
    if key == 'stage1' and isinstance(value, dict):
        return {k: v for k, v in value.items() if k != 'ts_offset'}
    return value


def _deep_update(base, new):
    out = deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != 'overrides':
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def rv_key(rv):
    return f'rv{rv:g}'


class Experiment:
    def __init__(self, cfg=None, site=SITE, scheduler=None, stat_loader=None, link_exe=True):
        self.cfg = _deep_update(DEFAULTS, cfg)
        self.cfg['stage1']['ts_offset'] = {float(k): v for k, v in self.cfg['stage1']['ts_offset'].items()}
        c = self.cfg
        self.site = site
        self.sched = scheduler or Slurm()
        self.load_stat = stat_loader or (lambda d: open_stat(d, site))
        self.link_exe = link_exe
        self.root = Path(site.work_root) / (c['name'] + ('_dryrun' if c['dry_run'] else ''))
        self.ledger = Ledger(self.root / 'ledger.json')
        self.snd_ref = self.root / 'reference' / 'snd_ref'
        namelist.merge({}, c['overrides'])            # fail early on a typo in OVERRIDES
        if not (CASE_FILES / site.case / c['grid']).exists():
            raise FileNotFoundError(f"GRID '{c['grid']}' not found in {CASE_FILES / site.case}")

    # ------------------------------------------------------------------ configuration --
    @classmethod
    def from_dir(cls, exp_root, **kw):
        cfg = json.loads((Path(exp_root) / 'experiment.json').read_text())
        site = kw.pop('site', SITE)
        site.work_root = Path(exp_root).parent
        cfg['name'] = Path(exp_root).name.removesuffix('_dryrun')
        return cls(cfg, site=site, **kw)

    def _lock_config(self):
        f = self.root / 'experiment.json'
        cur = json.loads(json.dumps(self.cfg, default=str))
        if f.exists():
            old = json.loads(f.read_text())
            comparable = lambda d, k: _without_guesses(k, d.get(k, json.loads(json.dumps(DEFAULTS.get(k), default=str))))
            diff = sorted(k for k in set(old) | set(cur)
                          if k not in MUTABLE and k != 'name' and comparable(old, k) != comparable(cur, k))
            if diff:
                raise RuntimeError(
                    f'{self.root.name} already exists with different settings for {diff}.\n'
                    'Physics must not change mid-experiment: use a new EXPERIMENT name.')
        self.root.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(cur, indent=2))

    def stopped(self):
        return (self.root / 'STOP').exists()

    # ------------------------------------------------------------------ namelists ------
    def params(self, stage, rv=0.0, ts=None, name=''):
        """The complete prm for one run. Stage-specific keys are applied AFTER the user's
        OVERRIDES only where the stage depends on them (dynamicocean, output windows)."""
        c = self.cfg
        p = namelist.merge(presets.paper_base(), c['overrides'])
        p.update(caseid=name, nrestart=0, r_stomata=float(rv))
        if stage == 'ocean':
            p.update(tabs_s=float(c['ts_ocean']), dodynamicocean=False, sst_ft=False,
                     nstop=int(c['ocean']['days'] * DAY))
        elif stage == 1:
            s = c['stage1']
            p.update(tabs_s=float(ts), dodynamicocean=True, sst_ft=True, tau_sst=float(s['tau_sst']),
                     p_low=float(min(s['p_low'], s['p_upp'])), p_upp=float(max(s['p_low'], s['p_upp'])),
                     nstop=int(s['days'] * DAY))
        elif stage == 2:
            s = c['stage2']
            every = int(round(s['out_every_hr'] * 3600 / p['dt']))
            start = int(s['out_start_day'] * DAY)
            p.update(tabs_s=float(ts), dodynamicocean=False, sst_ft=False, nstop=int(s['days'] * DAY),
                     nsave2D=every, nsave3D=every,
                     nsave2Dstart=start if s['save2D'] else NEVER, nsave2Dend=NEVER,
                     nsave3Dstart=start if s['save3D'] else NEVER, nsave3Dend=NEVER,
                     save2Davg=False, save3Davg=False)
        elif stage == 'smoke':
            p.update(tabs_s=float(c['ts_ocean']), nstop=int(ts), nstat=60, nstatfrq=60,
                     nsave2D=360, nsave2Dstart=360, nsave2Dend=NEVER)
        else:
            raise ValueError(stage)
        w = c.get('wind')
        if w:
            p.update(donudging_uv=True, tauls=float(w.get('tauls', 7200.)),
                     nudging_uv_z1=float(w.get('z1', -1.)), nudging_uv_z2=float(w.get('z2', 1e6)))
        return p

    def ts_guess(self, rv):
        off = self.cfg['stage1']['ts_offset']
        xs = sorted(off)
        return self.cfg['ts_ocean'] + float(np.interp(rv, xs, [off[x] for x in xs]))

    # ------------------------------------------------------------------ submission -----
    def _write_and_submit(self, e, key, walltime):
        run_dir = Path(e['dir'])
        auto = advance_command(self.root, key) if self.cfg['auto_advance'] else None
        (run_dir / 'run.slurm').write_text(batch_script(run_dir, run_dir.name, walltime, auto, self.site))
        e['nstep_at_submit'] = last_nstep(run_dir / 'run.out') or 0
        e['job_id'] = 'DRYRUN' if self.cfg['dry_run'] else self.sched.submit(run_dir)
        e['n_submits'] = e.get('n_submits', 0) + 1

    def _windy(self, snd):
        """The sounding a run starts from: with WIND, a copy whose u, v columns are the target
        wind (SAM's nudging target and initial wind), written once next to snd_ref."""
        w = self.cfg.get('wind')
        if not w:
            return snd
        out = self.root / 'reference' / f'{Path(snd).name}_wind'
        if not out.exists():
            write_snd(out, with_wind(read_snd(snd), w))
        return out

    def _new_run(self, state, key, stage, rv, ts, snd, walltime):
        name = 'ocean' if stage == 'ocean' else f'stage{stage}_{key}'
        run_dir = self.root / name
        snd = self._windy(snd)
        p = self.params(stage, rv, ts, name)
        stage_run(run_dir, p, snd, allow_diurnal=self.cfg['allow_diurnal'], site=self.site,
                  link_exe=self.link_exe, grid=self.cfg['grid'], meta=dict(experiment=self.cfg['name'], stage=stage, rv=rv))
        e = state.setdefault(key, dict(key=key, rv=rv))
        e.update(stage=stage, dir=str(run_dir), nstop=p['nstop'], ts=ts, n_submits=0, no_progress=0)
        self._write_and_submit(e, key, walltime)
        note(e, f'{name} submitted (job {e["job_id"]}, Ts={ts}, nstop={p["nstop"]})')
        print(f'{key}: {name} submitted, job {e["job_id"]}')

    def _resubmit(self, e, changes, why):
        """Every resubmission goes through here, so the guard cannot be bypassed.
        Note (restart.f90 read_statement): with nrestart=1 SAM restores most physics switches,
        the insolation settings, dodynamicocean and tabs0_int from the restart file; nstop,
        r_stomata, sst_ft, tau_sst and the output windows are re-read from the prm."""
        if e['n_submits'] >= self.cfg['max_submits']:
            self._fail(e, f'reached max_submits={self.cfg["max_submits"]} ({why})')
            return
        if changes:
            update_prm(e['dir'], changes, allow_diurnal=self.cfg['allow_diurnal'], site=self.site)
        self._write_and_submit(e, e['key'], self.site.walltime[self._wt(e)])
        note(e, f'resubmitted: {why} (job {e["job_id"]}, submit {e["n_submits"]})')
        print(f'{e["key"]}: resubmitted ({why}), job {e["job_id"]}')

    @staticmethod
    def _wt(e):
        return {'ocean': 'ocean', 1: 'stage1', 2: 'stage2'}[e['stage']]

    def _fail(self, e, why, status='failed'):
        e['stage_before_fail'] = e['stage']
        e['stage'] = status
        note(e, f'{status.upper()}: {why}')
        print(f'{e["key"]}: {status.upper()} -- {why}')

    # ------------------------------------------------------------------ user actions ---
    def start(self):
        """Submit the ocean reference run and register every r_v as 'waiting'."""
        self._lock_config()
        with self.ledger.edit() as state:
            state.setdefault('_meta', {})
            for rv in self.cfg['rv_list']:
                state.setdefault(rv_key(rv), dict(key=rv_key(rv), rv=float(rv), stage='waiting',
                                                  history=[]))
            if 'ocean' in state:
                print(f'ocean: already tracked (stage {state["ocean"]["stage"]})')
                return
            if self.cfg.get('reference_from'):
                self._borrow_reference(state)
                self._launch_waiting(state)
                return
            snd = CASE_FILES / self.site.case / self.cfg['ocean']['snd']
            self._new_run(state, 'ocean', 'ocean', 0.0, float(self.cfg['ts_ocean']), snd,
                          self.site.walltime['ocean'])

    def _borrow_reference(self, state):
        """Reuse a finished experiment's ocean reference (T_ref and snd_ref). Refuses if anything
        that defines the reference differs: grid, ocean SST and settings, layer, OVERRIDES, sun."""
        base = Path(self.cfg['reference_from'])
        base = base if base.is_absolute() else Path(self.site.work_root) / base
        meta = Ledger(base / 'ledger.json').load()
        if meta.get('ocean', {}).get('stage') != 'done' or 'T_ref' not in meta.get('_meta', {}) \
                or not (base / 'reference' / 'snd_ref').exists():
            raise RuntimeError(f'{base} has no finished ocean reference (run its ocean stage first)')
        bcfg = json.loads((base / 'experiment.json').read_text())
        get = lambda d, k: d.get(k, json.loads(json.dumps(DEFAULTS.get(k), default=str)))
        mine = json.loads(json.dumps(self.cfg, default=str))
        bad = [k for k in ('grid', 'ts_ocean', 'ocean', 'overrides', 'allow_diurnal') if get(bcfg, k) != get(mine, k)]
        bad += [f'stage1.{k}' for k in ('p_low', 'p_upp') if get(bcfg, 'stage1').get(k) != mine['stage1'].get(k)]
        if bad:
            raise RuntimeError(f'cannot borrow the reference of {base.name}: different {bad}')
        self.snd_ref.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(base / 'reference' / 'snd_ref', self.snd_ref)
        T_ref = meta['_meta']['T_ref']
        state['_meta'].update(T_ref=T_ref, reference_from=str(base),
                              ocean_drift_K=meta['_meta'].get('ocean_drift_K'))
        state['ocean'] = dict(key='ocean', rv=0.0, stage='done', borrowed_from=str(base), history=[])
        note(state['ocean'], f'reference borrowed from {base} (T_ref={T_ref:.3f} K); no ocean run')
        print(f'ocean: reference borrowed from {base.name} (T_ref = {T_ref:.3f} K)')

    def smoke_test(self, hours=2):
        """A short perpetual-sun run (own directory) to check the build, the prm and SOLIN."""
        run_dir = self.root.parent / f'{self.cfg["name"]}_smoke'
        p = self.params('smoke', ts=int(hours * 360), name=run_dir.name)
        stage_run(run_dir, p, CASE_FILES / self.site.case / self.cfg['ocean']['snd'],
                  allow_diurnal=self.cfg['allow_diurnal'], site=self.site, link_exe=self.link_exe, grid=self.cfg['grid'])
        (run_dir / 'run.slurm').write_text(batch_script(run_dir, run_dir.name,
                                                        self.site.walltime['smoke'], site=self.site))
        job = self.sched.submit(run_dir)
        print(f'smoke test submitted: job {job} in {run_dir}')
        return run_dir

    def check_smoke(self, run_dir):
        out = Path(run_dir) / 'run.out'
        print('finished cleanly:', finished_cleanly(out), '| max CFL, NaN in log:', health(out))
        stat = self.load_stat(run_dir)
        if stat is None:
            print('no STAT output yet'); return None
        r = perpetual_from_stat(stat)
        print(r['message'])
        print('NaN fields:', nan_fields(stat) or 'none')
        return r

    def advance(self, run=None, from_job=None):
        """Take the next step for every run (or only `run`). Safe to call any time."""
        if self.stopped():
            print(f'STOP file present in {self.root}: nothing done.'); return
        self._lock_config()
        with self.ledger.edit() as state:
            for key, e in list(state.items()):
                if key == '_meta' or (run and key != run):
                    continue
                if e['stage'] in TERMINAL or e['stage'] == 'waiting':
                    continue
                if e.get('job_id') == 'DRYRUN':
                    print(f'{key}: dry-run entry, nothing to advance'); continue
                if e.get('job_id') != from_job and self.sched.in_queue(e.get('job_id')):
                    print(f'{key}: queued / running'); continue
                self._step(state, e)
            # new r_v added to RV_LIST after the start
            for rv in self.cfg['rv_list']:
                k = rv_key(rv)
                if k not in state:
                    state[k] = dict(key=k, rv=float(rv), stage='waiting', history=[])
            if state.get('ocean', {}).get('stage') == 'done':
                self._launch_waiting(state)

    def _step(self, state, e):
        run_dir = Path(e['dir'])
        out = run_dir / 'run.out'
        if not finished_cleanly(out, e['nstop']):
            n = last_nstep(out) or 0
            if n <= e.get('nstep_at_submit', 0):
                e['no_progress'] = e.get('no_progress', 0) + 1
                if e['no_progress'] >= 2:
                    self._fail(e, f'no progress in two submissions in a row (stuck at step {n}); see {out}')
                    return
            else:
                e['no_progress'] = 0
            self._resubmit(e, {'nrestart': 1}, f'stopped at step {n} of {e["nstop"]} (walltime?)')
            return
        e['no_progress'] = 0
        stat = self.load_stat(run_dir)
        if stat is None:
            self._fail(e, f'finished but no STAT output in {run_dir / "OUT_STAT"}'); return
        bad = nan_fields(stat)
        if bad:
            self._fail(e, f'NaN in {bad}; restart files are suspect, stage a fresh run', 'corrupted')
            return
        sun = perpetual_from_stat(stat)
        e['solin'] = sun.get('mean')
        if not sun['ok'] and not self.cfg['allow_diurnal']:
            self._fail(e, sun['message']); return
        if e['stage'] == 'ocean':
            self._finish_ocean(state, e, stat)
        elif e['stage'] == 1:
            self._step_stage1(state, e, stat)
        elif e['stage'] == 2:
            e['stage'] = 'done'
            note(e, 'Stage 2 complete; NaN and perpetual-sun checks passed')
            print(f'{e["key"]}: Stage 2 complete.')

    def _finish_ocean(self, state, e, stat):
        c, s1 = self.cfg, self.cfg['stage1']
        layer = (s1['p_low'], s1['p_upp'])
        avg = c['ocean']['avg_days']
        drift = convergence.drift_check(stat, avg, s1['tol_K'], layer)
        snd0 = read_snd(Path(e['dir']) / self.site.case / 'snd')
        write_snd(self.snd_ref, snd_from_stat(stat, avg, snd0['psfc']))
        t1 = float(stat['time'][-1])
        T_ref = convergence.layer_T(stat, t1 - avg, t1, layer)
        state['_meta'].update(T_ref=T_ref, snd_ref=str(self.snd_ref), ocean_drift_K=drift['drift'])
        e['stage'] = 'done'
        note(e, f'done: T_ref={T_ref:.3f} K (last {avg} d), drift vs previous {avg} d = {drift["drift"]:+.3f} K')
        print(f'ocean: done. T_ref = {T_ref:.3f} K; drift over the last two windows {drift["drift"]:+.3f} K'
              + ('' if drift['converged'] else '  (larger than tol_K: consider a longer ocean run)'))

    def _launch_waiting(self, state):
        if not self.snd_ref.exists():
            return
        for key, e in state.items():
            if key != '_meta' and e.get('stage') == 'waiting':
                self._new_run(state, key, 1, e['rv'], round(self.ts_guess(e['rv']), 2),
                              self.snd_ref, self.site.walltime['stage1'])

    def _step_stage1(self, state, e, stat):
        s1 = self.cfg['stage1']
        r = convergence.stage1_check(stat, s1['window_days'], s1['tol_K'],
                                     (s1['p_low'], s1['p_upp']), T_ref=state['_meta']['T_ref'])
        e.setdefault('checks', []).append(r)
        msg = (f'day {r["day"]:.1f}: layer T {r["T_last"]:.3f} vs T_ref {r["T_ref"]:.3f} '
               f'(diff {r["diff"]:+.3f} K), Ts {r["sst"]:.2f} K')
        print(f'{e["key"]}: {msg}')
        if r['converged']:
            note(e, f'Stage 1 converged: {msg}')
            e['stage1_dir'], e['converged_ts'] = e['dir'], r['sst']
            self._new_run(state, e['key'], 2, e['rv'], round(r['sst'], 4), self.snd_ref,
                          self.site.walltime['stage2'])
            return
        new = e['nstop'] + int(s1['extend_days'] * DAY)
        if new > s1['max_days'] * DAY:
            self._fail(e, f'Stage 1 not converged by max_days={s1["max_days"]} ({msg})'); return
        note(e, f'not converged: {msg}')
        self._resubmit(e, {'nrestart': 1, 'nstop': new}, f'extend to day {new / DAY:g}')
        if e['stage'] == 1:
            e['nstop'] = new

    # ------------------------------------------------------------------ views ----------
    def status(self):
        """One row per run: stage, job, progress, Ts, last convergence check."""
        rows = []
        for key, e in self.ledger.load().items():
            if key == '_meta':
                continue
            d = Path(e['dir']) if 'dir' in e else None
            n = last_nstep(d / 'run.out') if d else None
            cfl, bad = health(d / 'run.out') if d else (0.0, False)
            last = (e.get('checks') or [{}])[-1]
            rows.append(dict(run=key, rv=e.get('rv'), stage=e['stage'], job=e.get('job_id'),
                             day=None if n is None else round(n / DAY, 1),
                             target_day=None if 'nstop' not in e else e['nstop'] / DAY,
                             submits=e.get('n_submits'), Ts=e.get('ts'),
                             dT_K=None if not last else round(last['diff'], 3),
                             max_cfl=round(cfl, 2), nan_in_log=bad))
        return rows

    def history(self, key):
        return self.ledger.load().get(key, {}).get('history', [])
