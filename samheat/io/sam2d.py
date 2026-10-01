"""Read SAM .2Dbin output directly in Python (no Fortran converter, no netCDF-Fortran).

Port of SAM_CODE/06_surface_2m/sam2d.py. Mirrors UTIL/SRC/2Dbin2nc.f (Khairoutdinov)
record by record. Verified against a real 32-task FWRCE .2Dbin (241 snapshots).

File layout, per snapshot (Fortran sequential unformatted, 4-byte record markers,
little-endian, as written by write_fields2D.f90):

    nstep
    nx, ny, nz, nsubs, nsubsx, nsubsy, nfields     (per-subdomain nx, ny)
    dx
    dy
    time [days]
    nfields x ( header 'name(8) ' ' long_name(80) ' ' units(10)',
                nsubs x (float32 tile of nx*ny, x fastest) )

Subdomain n sits at tile column n % nsubsx, tile row n // nsubsx.

UNITS. SAM's 2D 'Prec' is the instantaneous surface precipitation rate in **mm/day**
(units attribute 'mm/day'). van der Drift & O'Gorman (2025) report P_e in **mm/hr**:
divide by 24 (`prec_mm_hr`).
"""
from pathlib import Path

import numpy as np
import xarray as xr


class FortranRecords:
    """Sequential reader of Fortran unformatted records (4-byte markers, little-endian)."""

    def __init__(self, f):
        self.f = f

    def _marker(self):
        b = self.f.read(4)
        if len(b) < 4:
            return None
        return int(np.frombuffer(b, '<i4')[0])

    def read(self):
        """Next record's payload as bytes, or None at end of file."""
        n = self._marker()
        if n is None:
            return None
        body = self.f.read(n)
        tail = self._marker()
        if len(body) != n or tail != n:
            raise IOError('Corrupt record marker (file truncated?)')
        return body

    def skip(self):
        """Skip the next record without reading it. Returns its payload length."""
        n = self._marker()
        self.f.seek(n + 4, 1)
        return n


def _records(f):
    """Generator over record payloads (kept for backwards compatibility)."""
    rr = FortranRecords(f)
    while True:
        body = rr.read()
        if body is None:
            return
        yield body


def read_2dbin(path, fields=None):
    """Read one .2Dbin file into an xarray.Dataset with dims (time, y, x).

    Parameters
    ----------
    path : str or Path
    fields : iterable of str, optional
        2D field names to keep (e.g. ['Prec']); default all. Unwanted tiles are skipped
        without being copied.

    Returns
    -------
    xarray.Dataset
        Coordinates time [day], x, y [m], nstep. Each variable has `long_name`, `units`.
    """
    path = Path(path)
    fields = None if fields is None else set(fields)
    times, steps, data, meta = [], [], {}, {}
    grid = None
    with open(path, 'rb') as f:
        rr = FortranRecords(f)
        while True:
            rec = rr.read()
            if rec is None:
                break
            nstep = int(np.frombuffer(rec, '<i4')[0])
            nx, ny, nz, nsubs, nsx, nsy, nf = (int(v) for v in np.frombuffer(rr.read(), '<i4'))
            dx = float(np.frombuffer(rr.read(), '<f4')[0])
            dy = float(np.frombuffer(rr.read(), '<f4')[0])
            t = float(np.frombuffer(rr.read(), '<f4')[0])
            if grid is None:
                grid = (nx, ny, nsx, nsy, dx, dy)
            nxg, nyg = nx * nsx, ny * nsy
            for _ in range(nf):
                h = rr.read()
                name = h[:8].decode().strip()
                want = fields is None or name in fields
                if want:
                    meta[name] = dict(long_name=h[9:89].decode().strip(),
                                      units=h[90:100].decode().strip())
                    fld = np.empty((nyg, nxg), np.float32)
                for n in range(nsubs):
                    if not want:
                        rr.skip()
                        continue
                    tile = rr.read()
                    j0, i0 = (n // nsx) * ny, (n % nsx) * nx
                    fld[j0:j0 + ny, i0:i0 + nx] = np.frombuffer(tile, '<f4').reshape(ny, nx)
                if want:
                    data.setdefault(name, []).append(fld)
            times.append(t)
            steps.append(nstep)
    if grid is None:
        raise IOError(f'{path}: empty file')

    nx, ny, nsx, nsy, dx, dy = grid
    coords = dict(time=('time', np.array(times), dict(units='day')),
                  x=('x', dx * np.arange(nx * nsx), dict(units='m')),
                  y=('y', dy * np.arange(ny * nsy), dict(units='m')),
                  nstep=('time', np.array(steps)))
    ds = xr.Dataset({k: (('time', 'y', 'x'), np.stack(v), meta[k]) for k, v in data.items()},
                    coords=coords)
    ds.attrs['source'] = str(path)
    return ds


def find_2dbin(run_dir):
    """All .2Dbin files in <run_dir>/OUT_2D, sorted by name."""
    return sorted((Path(run_dir) / 'OUT_2D').glob('*.2Dbin'))


def open_run_2d(run_dir, fields=('Prec',), tmin=None, tmax=None):
    """Read and concatenate every OUT_2D/*.2Dbin of a run (restarts append or add files).

    Duplicate times (overlapping restarts) keep the last occurrence. `tmin`/`tmax` [day]
    subset the result. Raises FileNotFoundError if there is no .2Dbin.
    """
    files = find_2dbin(run_dir)
    if not files:
        raise FileNotFoundError(f'no .2Dbin in {Path(run_dir) / "OUT_2D"}')
    parts = [read_2dbin(f, fields) for f in files]
    ds = parts[0] if len(parts) == 1 else xr.concat(parts, 'time')
    _, last = np.unique(ds['time'].values[::-1], return_index=True)
    keep = np.sort(len(ds['time']) - 1 - last)
    ds = ds.isel(time=keep).sortby('time')
    if tmin is not None or tmax is not None:
        ds = ds.sel(time=slice(tmin, tmax))
    return ds


def prec_mm_hr(ds, name='Prec'):
    """Surface precipitation rate in mm/hr from SAM's 'Prec' (mm/day)."""
    units = ds[name].attrs.get('units', 'mm/day').strip()
    if units != 'mm/day':
        raise ValueError(f"expected Prec in mm/day, got '{units}'")
    out = ds[name] / 24.0
    out.attrs.update(units='mm/hr', long_name='Surface precipitation rate')
    return out
