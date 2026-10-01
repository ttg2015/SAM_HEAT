"""Pure-Python reader for SAM .bin3D files (save3Dbin = .true.), plus a UTIL fallback.

RECORD LAYOUT (one file = one output time; Fortran sequential unformatted, 4-byte record
markers, little-endian, float32). Derived by reading three sources side by side:
SAM_SRC/write_fields3D.f90 (header + field order), SAM_SRC/compress3D.f90 (the `savebin`
branch: raw float32, no compression) and UTIL/SRC/bin3D2nc.f (the reader):

    nx, ny, nz, nsubs, nsubsx, nsubsy, nfields      int32 x 7   (per-subdomain nx, ny)
    z(k)      k = 1..nz     one float32 record per level   [m]
    pres(k)   k = 1..nz     one float32 record per level   [mb]
    dx ; dy ; time [days]   one float32 record each
    nfields x ( header record  name(8) ' ' long_name(80) ' ' units(10)   (100 bytes)
                nsubs x ( one record: float32 tile, nx*ny*nz values, x fastest, then y, then z ) )

Subdomain n (the n-th tile record = MPI rank n) sits at tile column n % nsubsx and tile row
n // nsubsx (bin3D2nc: j0 = n/nsubsx, i0 = n - j0*nsubsx).
Field order written by write_fields3D: U V W PP [QRAD] [REL] [REI] TABS QV [QN] [QP] and then
the microphysics fields flagged in flag_micro3Dout; names are read from the file.
Units as written: U,V,W m/s; PP Pa; TABS K; QV, QN, QP g/kg; QRAD K/day.

STATUS / WHAT IS UNVERIFIED. This reader has been tested only on synthetic files written by
this package's own test-writer in the layout above; **no real .bin3D file was available**
when it was written (the layout is certain only as far as the three Fortran sources are
read correctly). It additionally assumes: little-endian ints/floats and 4-byte record
markers (gfortran/ifort defaults; if a file fails with 'Corrupt record marker' this is the
first suspect), `output_sep = .false.` (a single file per time; with output_sep SAM writes
one `.bin3D_<rank>` file per task and those are NOT read here), and that the file was not
gzipped (`dogzip3D`). The 2D reader (sam2d.py), which uses the same record scheme, was
verified on a real .2Dbin. If in doubt, run `convert_3d` (UTIL bin3D2nc) on one file and
compare with `read_bin3D`.
"""
from pathlib import Path

import numpy as np
import xarray as xr

from ..paths import SITE
from ..slurm import sh
from .sam2d import FortranRecords

try:                                    # lazy loading is optional
    import dask
    import dask.array as da
except ImportError:                     # pragma: no cover
    dask = None
    da = None


def scan_bin3D(path):
    """Read only the header of a .bin3D file: grid, time and the file offset of every field.

    Returns a dict with nx, ny, nz, nsubs, nsx, nsy, nf (per-subdomain nx, ny), dx, dy,
    time [day], z [m], p [mb] (numpy), and `fields`: {name: (offset, long_name, units)}
    where `offset` is the byte position of the first tile record of that field.
    """
    path = Path(path)
    with open(path, 'rb') as f:
        rr = FortranRecords(f)
        nx, ny, nz, nsubs, nsx, nsy, nf = (int(v) for v in np.frombuffer(rr.read(), '<i4'))
        z = np.array([np.frombuffer(rr.read(), '<f4')[0] for _ in range(nz)])
        p = np.array([np.frombuffer(rr.read(), '<f4')[0] for _ in range(nz)])
        dx = float(np.frombuffer(rr.read(), '<f4')[0])
        dy = float(np.frombuffer(rr.read(), '<f4')[0])
        time = float(np.frombuffer(rr.read(), '<f4')[0])
        fields = {}
        for _ in range(nf):
            h = rr.read()
            name = h[:8].decode().strip()
            fields[name] = (f.tell(), h[9:89].decode().strip(), h[90:100].decode().strip())
            for _ in range(nsubs):
                n = rr.skip()
                if n != 4 * nx * ny * nz:
                    raise IOError(f'{path}: tile record is {n} bytes, expected '
                                  f'{4 * nx * ny * nz} (layout mismatch?)')
    return dict(nx=nx, ny=ny, nz=nz, nsubs=nsubs, nsx=nsx, nsy=nsy, nf=nf, dx=dx, dy=dy,
                time=time, z=z, p=p, fields=fields)


def _read_field(path, offset, nx, ny, nz, nsubs, nsx, k0, k1):
    """One field, levels k0:k1, merged over subdomains -> float32 (k1-k0, ny*nsy, nx*nsx)."""
    nsy = nsubs // nsx
    out = np.empty((k1 - k0, ny * nsy, nx * nsx), np.float32)
    nlev = nx * ny
    with open(path, 'rb') as f:
        f.seek(offset)
        for n in range(nsubs):
            start = f.tell() + 4                     # after the leading marker
            f.seek(start + 4 * k0 * nlev)
            tile = np.frombuffer(f.read(4 * (k1 - k0) * nlev), '<f4')
            f.seek(start + 4 * nx * ny * nz + 4)     # past the trailing marker
            j0, i0 = (n // nsx) * ny, (n % nsx) * nx
            out[:, j0:j0 + ny, i0:i0 + nx] = tile.reshape(k1 - k0, ny, nx)
    return out


def _levels(z, zmax):
    return len(z) if zmax is None else int(np.searchsorted(z, zmax, side='right'))


def _build(paths, fields, zmax, lazy):
    """Dataset (time, z, y, x) from scanned files `paths` = [(path, header)] sorted by time."""
    h0 = paths[0][1]
    names = [n for n in h0['fields'] if all(n in h['fields'] for _, h in paths)]
    if fields is not None:
        missing = [n for n in fields if n not in names]
        if missing:
            raise KeyError(f'fields not in {paths[0][0]}: {missing}; available: {names}')
        names = list(fields)
    nz = _levels(h0['z'], zmax)
    nxg, nyg = h0['nx'] * h0['nsx'], h0['ny'] * h0['nsy']
    data = {}
    for name in names:
        arrs = []
        for path, h in paths:
            args = (str(path), h['fields'][name][0], h['nx'], h['ny'], h['nz'], h['nsubs'],
                    h['nsx'], 0, nz)
            if lazy and dask is not None:
                arrs.append(da.from_delayed(dask.delayed(_read_field)(*args),
                                            shape=(nz, nyg, nxg), dtype=np.float32))
            else:
                arrs.append(_read_field(*args))
        arr = da.stack(arrs) if (lazy and dask is not None) else np.stack(arrs)
        _, long_name, units = h0['fields'][name]
        data[name] = (('time', 'z', 'y', 'x'), arr, dict(long_name=long_name, units=units))
    coords = dict(time=('time', np.array([h['time'] for _, h in paths]), dict(units='day')),
                  z=('z', h0['z'][:nz], dict(units='m', long_name='height')),
                  p=('z', h0['p'][:nz], dict(units='mb', long_name='pressure')),
                  y=('y', h0['dy'] * np.arange(nyg), dict(units='m')),
                  x=('x', h0['dx'] * np.arange(nxg), dict(units='m')))
    ds = xr.Dataset(data, coords=coords)
    ds.attrs['source'] = str(paths[0][0]) if len(paths) == 1 else f'{len(paths)} .bin3D files'
    return ds


def read_bin3D(path, fields=None, zmax=None, lazy=False):
    """Read one .bin3D file -> xarray.Dataset with dims (time=1, z, y, x).

    fields : names to keep (default all in the file, e.g. U V W PP TABS QV QN QP QRAD).
    zmax   : keep only levels with z <= zmax [m] (reads proportionally fewer bytes).
    lazy   : return dask-backed arrays (reads on compute). Needs dask.
    Coordinates: time [day], z [m], p(z) [mb], x, y [m].  See the module docstring for the
    unverified assumptions.
    """
    return _build([(Path(path), scan_bin3D(path))], fields, zmax, lazy)


def read_bin3D_level(path, fields, k=0):
    """One model level k (0 = lowest) of selected fields -> ({name: (y, x) array}, meta).

    Convenience carried over from 06_surface_2m/sam2d.py. meta has time, z, pres, dx, units.
    """
    h = scan_bin3D(path)
    out, units = {}, {}
    for name in fields:
        if name not in h['fields']:
            raise KeyError(f'{path}: field {name} not found; available: {list(h["fields"])}')
        off, _, u = h['fields'][name]
        out[name] = _read_field(str(path), off, h['nx'], h['ny'], h['nz'], h['nsubs'],
                                h['nsx'], k, k + 1)[0]
        units[name] = u
    return out, dict(time=h['time'], z=float(h['z'][k]), pres=float(h['p'][k]), dx=h['dx'],
                     units=units)


def find_3d(run_dir):
    """All .bin3D files in <run_dir>/OUT_3D, sorted by name (= by time step)."""
    return sorted((Path(run_dir) / 'OUT_3D').glob('*.bin3D'))


def open_run_3d(run_dir, fields=None, tmin=None, tmax=None, zmax=None, lazy=True):
    """All OUT_3D/*.bin3D of a run, concatenated in time -> xarray.Dataset (time, z, y, x).

    Only file headers are read up front (cheap); with lazy=True (default, needs dask) each
    (file, field) is read on demand, so `ds['W']` costs nothing until computed.
    tmin/tmax [day] select snapshots; zmax [m] truncates the vertical grid (vdD: 16 km).
    Files are ordered by their stored time; duplicate times keep the last file.
    """
    files = find_3d(run_dir)
    if not files:
        raise FileNotFoundError(f'no .bin3D in {Path(run_dir) / "OUT_3D"}')
    scanned = {}
    for f in files:
        h = scan_bin3D(f)
        if (tmin is None or h['time'] >= tmin) and (tmax is None or h['time'] <= tmax):
            scanned[round(h['time'], 6)] = (f, h)
    if not scanned:
        raise ValueError(f'no snapshots between {tmin} and {tmax} in {run_dir}')
    paths = [scanned[t] for t in sorted(scanned)]
    return _build(paths, fields, zmax, lazy)


def convert_3d(run_dir, site=SITE, overwrite=False):
    """Fallback: convert every OUT_3D/*.bin3D to .nc with SAM's UTIL `bin3D2nc` (HPC only).

    Runs `<site.util_dir>/bin3D2nc <file>` inside OUT_3D after loading `site.modules`; skips
    files whose .nc is already newer unless overwrite=True. Returns the sorted list of .nc
    paths, which `xarray.open_mfdataset` can open (dims time, z, y, x; same as read_bin3D).
    """
    out = Path(run_dir) / 'OUT_3D'
    exe = Path(site.util_dir) / 'bin3D2nc'
    for b in sorted(out.glob('*.bin3D')):
        nc = b.with_suffix('.nc')
        if overwrite or not (nc.exists() and nc.stat().st_mtime >= b.stat().st_mtime):
            sh(f'{exe} {b.name}', cwd=out, modules=site.modules)
    return sorted(out.glob('*.nc'))
