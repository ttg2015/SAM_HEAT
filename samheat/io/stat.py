"""STAT output: convert .stat -> .nc with SAM's stat2nc when needed, then open with xarray."""
from pathlib import Path

import numpy as np
import xarray as xr

from ..paths import SITE
from ..slurm import sh


def open_stat(run_dir, site=SITE, convert=True):
    """Newest STAT netCDF of a run (converted first if the .stat is newer). None if none."""
    out = Path(run_dir) / 'OUT_STAT'
    if convert:
        for st in sorted(out.glob('*.stat')):
            nc = st.with_suffix('.nc')
            if not (nc.exists() and nc.stat().st_mtime >= st.stat().st_mtime):
                sh(f'{Path(site.util_dir) / "stat2nc"} {st.name}', cwd=out, modules=site.modules)
    ncs = sorted(out.glob('*.nc'))
    return xr.open_dataset(ncs[-1]) if ncs else None


def nan_fields(stat, fields=('TABS', 'QV', 'U', 'p')):
    """Names of fields that contain NaN (a run that 'finished' but blew up)."""
    return [f for f in fields if f in stat and np.isnan(stat[f].values).any()]
