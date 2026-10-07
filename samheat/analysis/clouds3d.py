"""3D cloud pictures and movies: cloudy grid cells drawn as white boxes, rain as blue boxes.

    from samheat.analysis.clouds3d import cloud_frame, cloud_movie
    cloud_frame(run_dir, time=45.0, out="clouds.png")          # one picture
    cloud_movie(run_dir, out="clouds.mp4", tmin=30, tmax=60)   # every 3-hourly snapshot

Two renderers:
    'pyvista'    (pretty, fast) VTK: cloudy cells on the true stretched grid, shaded, sky gradient,
                 ground plane, slowly rotating camera. pip install pyvista imageio imageio-ffmpeg.
                 Headless on a cluster it needs EGL or OSMesa (VTK >= 9.4 picks one automatically);
                 if it fails, use 'matplotlib'.
    'matplotlib' (fallback, no extra packages) mplot3d voxels, horizontally coarsened to keep the
                 number of boxes manageable; slower (seconds per frame) and flatter looking.
renderer='auto' tries pyvista first.
"""
import os
from pathlib import Path

import numpy as np

from .composites import open_3d

QN_CLOUD = 0.01     # g/kg: SAM's cloud threshold in the warm lower troposphere (smaller aloft)
QP_RAIN = 0.1       # g/kg: SAM's precipitation threshold


def _edges(c):
    """Cell edges from cell centres (half-way between centres; outer edges extrapolated; >= 0)."""
    c = np.asarray(c, float)
    mid = 0.5 * (c[1:] + c[:-1])
    e = np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])
    return np.maximum(e, 0.)


def _snapshot(ds, t, zmax):
    s = ds.sel(time=t, method='nearest') if not isinstance(t, (int, np.integer)) else ds.isel(time=t)
    s = s.sel(z=slice(None, zmax))
    return (s['QN'].values.astype(float), s['QP'].values.astype(float) if 'QP' in s else None,
            s.x.values.astype(float), s.y.values.astype(float), s.z.values.astype(float), float(s.time))


def _coarsen_max(a, n):
    """(z, y, x) -> max over n x n horizontal blocks."""
    if n <= 1:
        return a
    nz, ny, nx = a.shape
    a = a[:, :ny - ny % n, :nx - nx % n]
    return a.reshape(nz, ny // n, n, nx // n, n).max(axis=(2, 4))


# ------------------------------------------------------------------ pyvista ------------------------
def _pv_plotter(size):
    import pyvista as pv
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background('lightskyblue', top='white')
    return pv, pl


def _pv_draw(pv, pl, qn, qp, x, y, z, qn_min, qp_min, rain, title, zscale=3.):
    km = 1e-3
    grid = pv.RectilinearGrid(_edges(x) * km, _edges(y) * km, _edges(z) * km * zscale)
    grid.cell_data['qn'] = qn.transpose(2, 1, 0).ravel(order='F')       # VTK: x fastest
    pl.clear_actors()
    xe, ye = _edges(x) * km, _edges(y) * km
    ground = pv.Plane(center=(xe.mean(), ye.mean(), 0), i_size=np.ptp(xe), j_size=np.ptp(ye))
    pl.add_mesh(ground, color='#7a8b6f')
    cl = grid.threshold(qn_min, scalars='qn')
    if cl.n_cells:
        pl.add_mesh(cl, color='white', smooth_shading=False, ambient=0.35, diffuse=0.7, specular=0.0)
    if rain and qp is not None:
        grid.cell_data['qp'] = qp.transpose(2, 1, 0).ravel(order='F')
        rn = grid.threshold(qp_min, scalars='qp')
        if rn.n_cells:
            pl.add_mesh(rn, color='royalblue', opacity=0.35)
    pl.add_text(title + (f'   (heights x{zscale:g})' if zscale != 1 else ''), font_size=10, color='black', name='title')
    return xe, ye


def _pv_camera(pl, xe, ye, ztop_km, azimuth):
    L = max(np.ptp(xe), np.ptp(ye))
    cx, cy = xe.mean(), ye.mean()
    a = np.deg2rad(azimuth)
    pl.camera.position = (cx + 1.6 * L * np.cos(a), cy + 1.6 * L * np.sin(a), 0.9 * L)
    pl.camera.focal_point = (cx, cy, 0.25 * ztop_km)
    pl.camera.up = (0, 0, 1)


# ------------------------------------------------------------------ matplotlib ---------------------
def _mpl_draw(ax, qn, qp, x, y, z, qn_min, qp_min, rain, coarsen, title, azimuth):
    qn_c = _coarsen_max(qn, coarsen)
    xs, ys = x[::coarsen][:qn_c.shape[2]], y[::coarsen][:qn_c.shape[1]]
    dx, dy = (x[1] - x[0]) * coarsen, (y[1] - y[0]) * coarsen
    xe = np.append(xs, xs[-1] + dx) / 1e3
    ye = np.append(ys, ys[-1] + dy) / 1e3
    ze = _edges(z) / 1e3
    X, Y, Z = np.meshgrid(xe, ye, ze, indexing='ij')
    filled = (qn_c > qn_min).transpose(2, 1, 0)                           # (x, y, z)
    colors = np.empty(filled.shape, dtype=object)
    colors[filled] = '#ffffffee'
    if rain and qp is not None:
        wet = (_coarsen_max(qp, coarsen) > qp_min).transpose(2, 1, 0) & ~filled
        filled = filled | wet
        colors[wet] = '#4169e155'
    ax.cla()
    ax.set_facecolor('#bfe3ff')
    if filled.any():
        ax.voxels(X, Y, Z, filled, facecolors=colors, edgecolor='#d0d8e0', linewidth=0.2)
    ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[0], ye[-1]); ax.set_zlim(0, ze[-1])
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)'); ax.set_zlabel('z (km)')
    ax.set_box_aspect((1, 1, 0.45))
    ax.view_init(elev=22, azim=azimuth)
    ax.set_title(title, fontsize=9)


# ------------------------------------------------------------------ public -------------------------
def _choose(renderer):
    if renderer != 'auto':
        return renderer
    try:
        import pyvista  # noqa: F401
        return 'pyvista'
    except ImportError:
        return 'matplotlib'


def cloud_frame(run_dir, time=None, out='clouds.png', source='auto', zmax=16000., qn_min=QN_CLOUD,
                qp_min=QP_RAIN, rain=True, renderer='auto', coarsen=2, azimuth=-60., size=(1280, 720), zscale=3.):
    """One 3D picture of the clouds (QN > qn_min g/kg, white) and rain (QP > qp_min g/kg, blue) at
    `time` [day] (nearest snapshot; default: the last). zscale exaggerates heights (pyvista; the
    matplotlib renderer stretches its box instead). Returns the output path."""
    ds = open_3d(run_dir, source)
    ds = ds[[v for v in ('QN', 'QP') if v in ds]]
    t = float(ds.time[-1]) if time is None else time
    qn, qp, x, y, z, tt = _snapshot(ds, t, zmax)
    title = f'{Path(run_dir).name}   day {tt:.2f}'
    renderer = _choose(renderer)
    if renderer == 'pyvista':
        pv, pl = _pv_plotter(size)
        xe, ye = _pv_draw(pv, pl, qn, qp, x, y, z, qn_min, qp_min, rain, title, zscale)
        _pv_camera(pl, xe, ye, zscale * zmax / 1e3, azimuth)
        pl.screenshot(str(out))
        pl.close()
    else:
        import matplotlib
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        ax = fig.add_subplot(projection='3d')
        _mpl_draw(ax, qn, qp, x, y, z, qn_min, qp_min, rain, coarsen, title, azimuth)
        fig.savefig(out, dpi=100)
        plt.close(fig)
    return Path(out)


def cloud_movie(run_dir, out='clouds.mp4', tmin=None, tmax=None, source='auto', every=1, fps=12,
                zmax=16000., qn_min=QN_CLOUD, qp_min=QP_RAIN, rain=True, rotate=0.5, azimuth0=-60.,
                renderer='auto', coarsen=2, size=(1280, 720), progress=20, zscale=3.):
    """Movie of the 3D clouds through the snapshots between tmin and tmax [day] (every `every`-th).
    rotate: camera degrees per frame (0 = fixed). Writes mp4 (needs imageio-ffmpeg or system ffmpeg);
    if no ffmpeg is found, writes a GIF next to it instead. Returns the output path."""
    ds = open_3d(run_dir, source, tmin, tmax)
    ds = ds[[v for v in ('QN', 'QP') if v in ds]]
    idx = list(range(0, ds.sizes['time'], every))
    renderer = _choose(renderer)
    out = Path(out)
    name = Path(run_dir).name
    if renderer == 'pyvista':
        pv, pl = _pv_plotter(size)
        try:
            pl.open_movie(str(out), framerate=fps, quality=7)
        except Exception:
            out = out.with_suffix('.gif')
            pl.open_gif(str(out), fps=fps)
        for n, i in enumerate(idx):
            qn, qp, x, y, z, tt = _snapshot(ds, i, zmax)
            xe, ye = _pv_draw(pv, pl, qn, qp, x, y, z, qn_min, qp_min, rain, f'{name}   day {tt:.2f}', zscale)
            _pv_camera(pl, xe, ye, zscale * zmax / 1e3, azimuth0 + rotate * n)
            pl.write_frame()
            if progress and (n % progress == 0 or n == len(idx) - 1):
                print(f'  frame {n + 1}/{len(idx)}')
        pl.close()
        return out
    import matplotlib.pyplot as plt
    from matplotlib import animation
    fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax = fig.add_subplot(projection='3d')

    def draw(n):
        qn, qp, x, y, z, tt = _snapshot(ds, idx[n], zmax)
        _mpl_draw(ax, qn, qp, x, y, z, qn_min, qp_min, rain, coarsen, f'{name}   day {tt:.2f}',
                  azimuth0 + rotate * n)
        if progress and (n % progress == 0 or n == len(idx) - 1):
            print(f'  frame {n + 1}/{len(idx)}')
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(idx), blit=False)
    if not animation.writers.is_available('ffmpeg'):
        try:
            import imageio_ffmpeg
            plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if animation.writers.is_available('ffmpeg'):
        anim.save(out, writer=animation.FFMpegWriter(fps=fps), dpi=100)
    else:
        out = out.with_suffix('.gif')
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return out
