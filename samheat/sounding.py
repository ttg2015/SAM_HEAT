"""SAM snd files: read, write, and build one from a STAT time mean (port of
SAM_OUTPUT_SCRIPTS/snd_from_sam.py with sanity checks)."""
import re
from pathlib import Path

import numpy as np

R_SND, CP_SND = 287.05, 1004.0      # constants used by snd_from_sam.py
PLAUSIBLE_T = (200.0, 320.0)


def read_snd(path):
    """First time block of a snd file -> dict(z, p, tp, q, u, v, psfc)."""
    lines = Path(path).read_text().splitlines()
    head = re.split(r'[,\s]+', lines[1].strip())
    nz, psfc = int(float(head[1])), float(head[2])
    data = np.array([[float(x) for x in l.split()] for l in lines[2:2 + nz]])
    z, p, tp, q, u, v = data.T
    return dict(z=z, p=p, tp=tp, q=q, u=u, v=v, psfc=psfc)


def write_snd(path, snd):
    """Two identical time blocks (day 0 and day 1): a time-invariant sounding."""
    nz = len(snd['z'])
    rows = ''.join('%.4f    %.4f    %.4f    %.4f    %.4f    %.4f\n'
                   % tuple(snd[c][i] for c in ('z', 'p', 'tp', 'q', 'u', 'v')) for i in range(nz))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('  z[m] p[mb] tp[K] q[g/kg] u[m/s] v[m/s]\n'
                    + f'0, {nz},{snd["psfc"]:f}\n' + rows + f'1, {nz},{snd["psfc"]:f}\n' + rows)
    return path


def snd_from_stat(stat, avg_days, psfc, u=None, v=None):
    """Equilibrium sounding from the last `avg_days` of a STAT dataset.

    psfc: surface pressure written in the header (take it from the sounding the run started
    from, as van der Drift did). u, v default to zero (no mean wind)."""
    t1 = float(stat['time'][-1])
    m = stat.sel(time=slice(t1 - avg_days, t1)).mean('time')
    z, p, T, q = (np.asarray(m[k].values, float) for k in ('z', 'p', 'TABS', 'QV'))
    if np.isnan(p).any() or np.isnan(T).any() or np.isnan(q).any():
        raise ValueError('NaN in the reference profile')
    if not (p[:-1] >= p[1:]).all():
        raise ValueError('pressure not decreasing with height')
    if not (PLAUSIBLE_T[0] < T.min() and T.max() < PLAUSIBLE_T[1]):
        raise ValueError(f'temperature out of range {T.min():.1f}-{T.max():.1f} K')
    zeros = np.zeros_like(z)
    return dict(z=z, p=p, tp=T * (1000. / p) ** (R_SND / CP_SND), q=q,
                u=zeros if u is None else u, v=zeros if v is None else v, psfc=psfc)
