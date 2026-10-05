"""Check our pure-Python .bin3D reader against SAM's own converter (UTIL/bin3D2nc). HPC only.

`compare_bin3d_with_converter(path)` copies ONE .bin3D into a temp dir, runs `bin3D2nc` on the
copy, opens the .nc with xarray and compares every common variable with `read_bin3D`.

What bin3D2nc does with W (not verified here -- the util source was not available when this was
written): some SAM versions write W on interface levels / shifted. The comparison therefore
reports, separately, whether the z (and p) coordinates agree and whether the array shapes agree;
if W is on a different grid the row for W shows shape_ok=False (or a large difference) and the
z-coordinate row shows the discrepancy.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..paths import SITE
from .. import slurm
from .sam3d import read_bin3D

RTOL = 1e-5


def _squeeze(a):
    a = np.asarray(a)
    while a.ndim > 3 and a.shape[0] == 1:
        a = a[0]
    return a


def _coord(ds, names):
    for n in names:
        if n in ds.variables:
            return np.asarray(ds[n].values, float).ravel()
    return None


def _coord_row(name, ours, theirs, tol, offset_ok=False):
    row = dict(variable=name, kind='coord', shape_ok=None, max_abs=np.nan, max_rel=np.nan,
               max_pointwise_rel=np.nan, ok=False, note='')
    if theirs is None:
        row['note'] = 'absent in converter output'
        return row
    if ours.shape != theirs.shape:
        row.update(shape_ok=False, note=f'length {ours.shape[0]} (ours) vs {theirs.shape[0]} (converter)')
        return row
    d = np.abs(ours - theirs)
    scale = max(np.abs(theirs).max(), 1e-30)
    row.update(shape_ok=True, max_abs=float(d.max()), max_rel=float(d.max() / scale))
    row['ok'] = bool(d.max() <= tol * scale + 1e-9)
    if not row['ok'] and offset_ok and ours.size > 1:
        # horizontal axes: same spacing, different origin (e.g. cell centres) is acceptable
        dd = np.abs(np.diff(ours) - np.diff(theirs)).max()
        if dd <= tol * scale:
            row['ok'] = True
            row['note'] = f'same spacing, origin offset {float(theirs[0] - ours[0]):g}'
    return row


def compare_bin3d_with_converter(bin3d_path, site=SITE, rtol=RTOL, keep=None, verbose=True):
    """Compare read_bin3D with SAM's bin3D2nc on one file. Returns a DataFrame.

    One row per common field (kind='field': max_abs, max_rel = max|a-b| / max|converter|,
    max_pointwise_rel over points with |converter| > 1e-6 max, shape_ok, ok) and per coordinate
    (x, y, z, p, time; kind='coord'). Prints PASS/FAIL (fields and z/p/time must agree to
    `rtol`; x/y only in spacing). `keep`: optional directory to leave the temp copy in.
    """
    src = Path(bin3d_path)
    tmp = Path(keep) if keep else Path(tempfile.mkdtemp(prefix='bin3d_check_'))
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        copy = tmp / src.name
        shutil.copy2(src, copy)
        exe = Path(site.util_dir) / 'bin3D2nc'
        slurm.sh(f'{exe} {copy.name}', cwd=tmp, modules=site.modules)
        ncs = sorted(tmp.glob('*.nc'))
        if not ncs:
            raise FileNotFoundError(f'bin3D2nc produced no .nc in {tmp}')
        with xr.open_dataset(ncs[0]) as nc:
            nc = nc.load()
        ours = read_bin3D(copy)
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)

    rows = []
    common = [v for v in ours.data_vars if v in nc.data_vars]
    for v in common:
        a, b = _squeeze(ours[v].values).astype(float), _squeeze(nc[v].values).astype(float)
        row = dict(variable=v, kind='field', shape_ok=a.shape == b.shape, max_abs=np.nan,
                   max_rel=np.nan, max_pointwise_rel=np.nan, ok=False,
                   note=f'ours {a.shape} vs converter {b.shape}')
        if row['shape_ok']:
            d = np.abs(a - b)
            scale = max(np.abs(b).max(), 1e-30)
            m = np.abs(b) > 1e-6 * scale
            row.update(max_abs=float(d.max()), max_rel=float(d.max() / scale),
                       max_pointwise_rel=float((d[m] / np.abs(b[m])).max()) if m.any() else 0.0,
                       note='')
            row['ok'] = bool(row['max_rel'] <= rtol)
        rows.append(row)
    for v in sorted(set(ours.data_vars) ^ set(nc.data_vars)):
        rows.append(dict(variable=v, kind='field', shape_ok=None, max_abs=np.nan, max_rel=np.nan,
                         max_pointwise_rel=np.nan, ok=None,
                         note='only in ' + ('ours' if v in ours.data_vars else 'converter')))
    cn = {'x': ['x'], 'y': ['y'], 'z': ['z'], 'p': ['p', 'pres', 'pressure'], 'time': ['time']}
    for name, alts in cn.items():
        rows.append(_coord_row(name, np.asarray(ours[name].values, float).ravel(),
                               _coord(nc, alts), rtol, offset_ok=name in ('x', 'y')))
    df = pd.DataFrame(rows)
    need = df[((df.kind == 'field') & df.ok.notna()) | ((df.kind == 'coord') & df.variable.isin(['z', 'p', 'time', 'x', 'y']))]
    passed = bool(len(common)) and bool(need.ok.astype(bool).all())
    if verbose:
        print(f'{"PASS" if passed else "FAIL"}: {src.name}: {len(common)} common fields, '
              f'tolerance {rtol:g} relative (max |a-b| / max |converter|)')
        if not passed:
            print(df[df.ok.eq(False)].to_string())
    df.attrs['passed'] = passed
    return df
