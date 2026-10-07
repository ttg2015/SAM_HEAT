"""Ready-made masks for composites, and the thermodynamic helpers they use.

A mask is a function  mask(f, z, **params) -> {group name: boolean array}
    f : dict of 3D float arrays (z, y, x) for ONE snapshot, SAM units: U V W PP QRAD TABS QV QN QP
        (T K, q g/kg, PP Pa) plus derived MSE, MSE_frozen, S_LI, THETA_E and their saturated
        versions MSE_SAT, MSE_frozen_SAT, THETA_ES (kJ/kg, K).
    z : (z,) heights [m].
The arrays may be
    (y, x)     COLUMN masks: the same columns at every level (e.g. near-surface W > X), or
    (z, y, x)  LEVEL masks: chosen at each height (e.g. cloudy here: QN > threshold).
Groups must be the same every snapshot; they may overlap or leave points out.

Note: SAM stores W(k) on the interface BELOW scalar level k; w_at_levels() averages it onto the
scalar levels where T, q and the MSE live.
"""
import numpy as np

G, CP, EPSV = 9.81, 1004., 0.61


# ------------------------------------------------------------------ helpers -------------------
def w_at_levels(f):
    """W interpolated to the scalar levels: 0.5 (W(k) + W(k+1)); top level keeps W(top)."""
    W = f['W']
    out = W.copy()
    out[:-1] = 0.5 * (W[:-1] + W[1:])
    return out


def buoyancy(f, include_qp=True):
    """SAM's buoyancy (van der Drift & O'Gorman 2025, eq. 7) on scalar levels [m/s^2]:
    B = g [(T - T')/T' (1 + eps_v qv' - qn' - qp') + eps_v (qv - qv') - (qn + qp - qn' - qp')],
    primes = horizontal mean of the snapshot. include_qp=False drops rain/snow loading."""
    T, qv, qn = f['TABS'], f['QV'] / 1e3, f['QN'] / 1e3
    qp = f['QP'] / 1e3 if include_qp else np.zeros_like(T)
    m = lambda a: a.mean(axis=(1, 2), keepdims=True)
    Tm, qvm, qnm, qpm = m(T), m(qv), m(qn), m(qp)
    return G * ((T - Tm) / Tm * (1 + EPSV * qvm - qnm - qpm) + EPSV * (qv - qvm) - (qn + qp - qnm - qpm))


def buoyancy_at_w(B, k):
    """Buoyancy at the W level k (the interface below scalar level k)."""
    return B[0] if k == 0 else 0.5 * (B[k - 1] + B[k])


def lcl_height(f, z, k_src=0, p_src=1000.):
    """Rough LCL [m] of the air at level k_src: Espy, 125 m per K of dew-point depression."""
    T, qv = f['TABS'][k_src], f['QV'][k_src] / 1e3
    e = np.maximum(p_src * qv / (0.622 + qv), 1e-3)                       # hPa
    Td = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112)) + 273.15
    return z[k_src] + 125. * np.maximum(T - Td, 0.)


def parcel_cape(f, z, k_src=0, zmax=16000.):
    """CAPE and CIN proxies [J/kg] per column (MSE approximation): lift the air at level k_src with
    its h and compare with the snapshot-mean h*(z): b = g (h_p - h*_env)/(c_p T_env), counted above
    the column's LCL only. CAPE = positive area below zmax; CIN = negative area from LCL to LFC."""
    hp = f['MSE'][k_src]
    hs = f['MSE_SAT'].mean(axis=(1, 2))
    Te = f['TABS'].mean(axis=(1, 2))
    b = G * 1000. * (hp[None] - hs[:, None, None]) / (CP * Te[:, None, None])
    dz = np.gradient(z)[:, None, None]
    zz = z[:, None, None]
    above = (zz > lcl_height(f, z, k_src)[None]) & (zz <= zmax)
    pos = (b > 0) & above
    cape = np.where(pos, b * dz, 0.).sum(axis=0)
    zlfc = np.where(pos, zz, np.inf).min(axis=0)
    cin = np.where(above & (b < 0) & (zz < zlfc[None]), b * dz, 0.).sum(axis=0)
    return cape, cin


# ------------------------------------------------------------------ column masks --------------
def symmetric(f, z, k=1, X=0.3):
    """|W| > X at W level k."""
    w = f['W'][k]
    return dict(updraft=w > X, downdraft=w < -X, intermediate=np.abs(w) <= X)


def sigma(f, z, k=1, n=2.0):
    """|W| > n standard deviations of W at level k in this snapshot."""
    w = f['W'][k]
    X = n * w.std()
    return dict(updraft=w > X, downdraft=w < -X, intermediate=np.abs(w) <= X)


def percentile(f, z, k=1, q=1.0):
    """Strongest q % of upward and of downward W at level k in this snapshot."""
    w = f['W'][k]
    hi, lo = np.percentile(w, [100 - q, q])
    return dict(updraft=w >= hi, downdraft=w <= lo, intermediate=(w > lo) & (w < hi))


def layer_mean(f, z, zbot=0., ztop=1000., X=0.3):
    """|W| > X for W averaged over the levels between zbot and ztop [m]."""
    sel = (z >= zbot) & (z <= ztop)
    w = f['W'][sel].mean(axis=0)
    return dict(updraft=w > X, downdraft=w < -X, intermediate=np.abs(w) <= X)


def buoyant(f, z, k=1, X=0.3, include_qp=True):
    """Updrafts at W level k split by their buoyancy there: buoyant (B > 0) or forced (B <= 0)."""
    w = f['W'][k]
    b = buoyancy_at_w(buoyancy(f, include_qp), k)
    return dict(buoyant_updraft=(w > X) & (b > 0), forced_updraft=(w > X) & (b <= 0),
                downdraft=w < -X, intermediate=np.abs(w) <= X)


def buoyancy_only(f, z, k=1, B0=0.01, include_qp=True):
    """Columns by buoyancy at W level k alone: positive (B > B0), negative (B < -B0), neutral."""
    b = buoyancy_at_w(buoyancy(f, include_qp), k)
    return dict(positive=b > B0, negative=b < -B0, neutral=np.abs(b) <= B0)


def parcel_h(f, z, k_src=0, cape_min=200., cin_max=20., zmax=16000.):
    """Columns by what their near-surface air could do: free / capped (CIN < -cin_max) / stable."""
    cape, cin = parcel_cape(f, z, k_src, zmax)
    ok = cape >= cape_min
    return dict(free=ok & (cin >= -cin_max), capped=ok & (cin < -cin_max), stable=~ok)


# ------------------------------------------------------------------ level masks ---------------
def sam_cloud_threshold(f, z):
    """SAM's own 'cloudy' threshold per level in g/kg, exactly as statistics.f90 (CLD conditional):
    QN > min(1e-5 kg/kg, 0.01 * qsatw(tabs0(k), pres(k))), with qsatw SAM's saturation mixing ratio
    over water (sat.f90; same polynomial in thermo_SAM.qsatw) at the level's horizontal-mean
    temperature and pressure. QN = cloud liquid + ice (SAM's qcl + qci)."""
    from .thermo_SAM import qsatw
    Tm = f['TABS'].mean(axis=(1, 2))
    pm = f['P'].mean(axis=(1, 2)) if 'P' in f else np.full(len(Tm), 1000.)
    return np.minimum(0.01, 0.01 * qsatw(Tm, pm))[:, None, None]          # qsatw returns g/kg


def cloud(f, z, qn_min=None):
    """At each level: cloudy vs clear. qn_min [g/kg]; None = SAM's own criterion (sam_cloud_threshold),
    so the 'cloud' composite of MSE matches the STAT file's MSECLD (which SAM always writes, in K)."""
    thr = sam_cloud_threshold(f, z) if qn_min is None else qn_min
    c = f['QN'] > thr
    return dict(cloud=c, clear=~c)


def cloud_updraft(f, z, qn_min=None, w_min=1.0):
    """At each level: cloudy updrafts (cloudy and w > w_min m/s), other cloud, clear air."""
    thr = sam_cloud_threshold(f, z) if qn_min is None else qn_min
    c = f['QN'] > thr
    up = w_at_levels(f) > w_min
    return dict(cloud_updraft=c & up, other_cloud=c & ~up, clear=~c)


def core(f, z, qn_min=None, include_qp=True):
    """At each level, cloud-core sampling (Siebesma & Cuijpers 1995): cloudy, rising (w > 0) and
    positively buoyant (B > 0); plus the rest of the cloud and clear air."""
    thr = sam_cloud_threshold(f, z) if qn_min is None else qn_min
    c = f['QN'] > thr
    is_core = c & (w_at_levels(f) > 0) & (buoyancy(f, include_qp) > 0)
    return dict(core=is_core, other_cloud=c & ~is_core, clear=~c)


def sam_cores(f, z, w_min=1.0, include_qp=True):
    """At each level, SAM's own (non-LES) core definitions (doSAMconditionals): updraft core =
    buoyant (B > 0) and w > w_min; downdraft core = negatively buoyant and w < -w_min."""
    w, B = w_at_levels(f), buoyancy(f, include_qp)
    up, dn = (B > 0) & (w > w_min), (B < 0) & (w < -w_min)
    return dict(updraft_core=up, downdraft_core=dn, other=~(up | dn))


# ------------------------------------------------------------------ precipitation -------------
QP_RAIN = 0.1      # g/kg = 1e-4 kg/kg: SAM's only precipitation threshold (statistics.f90, CDN / HYDRO)


def rain(f, z, qp_min=QP_RAIN):
    """At each level: precipitating hydrometeors present (QP > qp_min g/kg) vs not."""
    r = f['QP'] > qp_min
    return dict(rain=r, no_rain=~r)


def hydrometeors(f, z, qn_min=None, qp_min=QP_RAIN):
    """At each level, SAM's exclusive HYDRO classes: cloud (SAM's criterion), else rain only
    (QP > qp_min), else clear. 'rain_only' is e.g. rain falling below cloud base (re-evaporation)."""
    thr = sam_cloud_threshold(f, z) if qn_min is None else qn_min
    c = f['QN'] > thr
    r = (f['QP'] > qp_min) & ~c
    return dict(cloud=c, rain_only=r, clear=~(c | r))


def precipitating_cloud(f, z, qn_min=None, qp_min=QP_RAIN):
    """At each level: cloud with precipitation (cloudy and QP > qp_min), non-precipitating cloud, clear."""
    thr = sam_cloud_threshold(f, z) if qn_min is None else qn_min
    c = f['QN'] > thr
    r = f['QP'] > qp_min
    return dict(precipitating_cloud=c & r, non_precipitating_cloud=c & ~r, clear=~c)


def raining_column(f, z, qp_min=QP_RAIN, zmax=1000.):
    """Columns with rain near the surface: max QP below zmax [m] > qp_min. A proxy for surface rain
    from the 3D fields (the 2D 'Prec' is in separate files); 0.1 g/kg is roughly a few mm/h,
    so very light rain is missed."""
    q = f['QP'][np.asarray(z) <= zmax].max(axis=0)
    return dict(raining=q > qp_min, dry=q <= qp_min)


def heavy_rain_column(f, z, q=1.0, zmax=1000.):
    """Columns with the heaviest near-surface rain in this snapshot: top q % of max QP below zmax
    (a 3D proxy for the precipitation-extreme columns of van der Drift & O'Gorman)."""
    qp = f['QP'][np.asarray(z) <= zmax].max(axis=0)
    hi = np.percentile(qp, 100 - q)
    heavy = (qp >= hi) & (qp > 0)
    return dict(heavy_rain=heavy, other=~heavy)


def cloud_water_path(f, z):
    """Cloud (QN) water path [kg/m^2] per column: sum rho QN dz, rho = p/(R_d T_v) from P, TABS, QV."""
    rho = 100. * f['P'] / (287.04 * f['TABS'] * (1 + 0.608 * f['QV'] / 1e3))
    dz = np.gradient(np.asarray(z, float))[:, None, None]
    return (rho * f['QN'] / 1e3 * dz).sum(axis=0)


def cloudy_column(f, z, cwp_min=0.02):
    """Columns whose cloud water path exceeds cwp_min kg/m^2 (SAM's own column-cloud threshold,
    cwpmax = 0.02 in statistics.f90) vs clear-sky columns."""
    c = cloud_water_path(f, z) > cwp_min
    return dict(cloudy_column=c, clear_column=~c)


COLUMN_MASKS = dict(symmetric=symmetric, sigma=sigma, percentile=percentile, layer_mean=layer_mean,
                    buoyant=buoyant, buoyancy_only=buoyancy_only, parcel_h=parcel_h,
                    raining_column=raining_column, heavy_rain_column=heavy_rain_column,
                    cloudy_column=cloudy_column)
LEVEL_MASKS = dict(cloud=cloud, cloud_updraft=cloud_updraft, core=core, sam_cores=sam_cores,
                   rain=rain, hydrometeors=hydrometeors, precipitating_cloud=precipitating_cloud)
