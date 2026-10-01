"""A fake SLURM + fake SAM output, so the state machine can be tested in a second.

FakeSlurm.submit() 'runs' the job immediately by appending to run.out, according to the
behaviour assigned to that run directory's name:
    'ok'        reaches nstop and prints SAM's exit line
    'walltime'  advances some steps but does not finish
    'stuck'     makes no progress at all
FakeStat builds a STAT-like xarray Dataset whose 600-400 hPa layer temperature is
T_REF + offset(run name), with constant (or diurnal) SOLIN, and optional NaN.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from samheat import namelist
from samheat.checks import EXPECTED_SOLIN

P = np.linspace(1000., 50., 64)
Z = np.linspace(25., 27000., 64)
T_PROFILE = 300. - 0.0065 * Z + np.where(Z > 15000, 0.0065 * (Z - 15000), 0)


class FakeSlurm:
    def __init__(self, behaviour=None):
        self.behaviour = behaviour or {}          # run dir name -> 'ok' | 'walltime' | 'stuck'
        self.n = 0
        self.submissions = []

    def submit(self, run_dir):
        run_dir = Path(run_dir)
        self.n += 1
        job = str(1000 + self.n)
        self.submissions.append(run_dir.name)
        prm = namelist.read_prm(run_dir / 'FWRCE' / 'prm')
        out = run_dir / 'run.out'
        prev = 0
        if out.exists():
            for l in out.read_text().splitlines():
                if l.startswith('NSTEP'):
                    prev = int(l.split('=')[1].split('NCYCLE')[0])
        how = self.behaviour.get(run_dir.name, 'ok')
        with open(out, 'a') as f:
            if how == 'ok':
                f.write(f'NSTEP = {prm["nstop"]} NCYCLE=1\n Finished with SAM, exiting\n')
            elif how == 'walltime':
                f.write(f'NSTEP = {min(prev + 1000, prm["nstop"] - 1)} NCYCLE=1\n')
            elif how == 'stuck':
                f.write(f'NSTEP = {prev} NCYCLE=1\n')
        return job

    def in_queue(self, job_id):
        return False


class FakeStat:
    def __init__(self, offset=None, diurnal=(), nan=()):
        self.offset = offset or {}                # run dir name -> K added to TABS
        self.diurnal, self.nan = set(diurnal), set(nan)

    def __call__(self, run_dir):
        run_dir = Path(run_dir)
        name = run_dir.name
        prm = namelist.read_prm(run_dir / 'FWRCE' / 'prm')
        days = prm['nstop'] / 8640
        t = np.arange(1, int(days * 24) + 1) / 24.0
        T = np.tile(T_PROFILE + self.offset.get(name, 0.0), (t.size, 1))
        if name in self.nan:
            T[-1, 3] = np.nan
        sol = np.full(t.size, EXPECTED_SOLIN)
        if name in self.diurnal:
            sol = np.maximum(0, 1360 * np.cos(2 * np.pi * (t - 0.5)))
        return xr.Dataset(
            dict(TABS=(('time', 'z'), T), QV=(('time', 'z'), np.full_like(T, 5.0)),
                 U=(('time', 'z'), np.zeros_like(T)), SOLIN=('time', sol),
                 SST=('time', np.full(t.size, prm['tabs_s'] + 0.123)),
                 p=('z', P)),
            coords=dict(time=t, z=Z))
