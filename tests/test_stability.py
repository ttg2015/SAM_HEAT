"""Emanuel (1994) stability diagnostics: reference values and self-consistency."""
import numpy as np
import pytest
import xarray as xr

from samheat.analysis import stability as S


def test_gamma_m_reference_values():
    # textbook moist-adiabatic lapse rates near the surface (saturated, no condensate):
    # about 3.6-3.9 K/km at 300 K / 1000 hPa, about 6.5 K/km at 0 C / 700 hPa
    g300 = 1000 * S.gamma_m(300., 1000., S.rsat(300., 1000.))
    g273 = 1000 * S.gamma_m(273.15, 700., S.rsat(273.15, 700.))
    assert 3.4 < g300 < 4.0
    assert 5.4 < g273 < 6.2                         # 700 hPa: less than the ~6.5 K/km at 1000 hPa
    assert 1000 * S.gamma_m(200., 200., S.rsat(200., 200.)) == pytest.approx(9.8, abs=0.1)   # nearly dry


def _moist_adiabat(T0=300., p0=1000., rT=None, ztop=10000., dz=5.):
    """Integrate a reversible moist adiabat (dT/dz = -Gamma_m, r_T constant) with hydrostatic p."""
    rT = S.rsat(T0, p0) if rT is None else rT
    z = np.arange(0., ztop + dz, dz)
    T, p = np.empty_like(z), np.empty_like(z)
    T[0], p[0] = T0, p0
    for i in range(len(z) - 1):
        rs = S.rsat(T[i], p[i])
        Trho = T[i] * (1 + rs / S.EPS) / (1 + rT)
        T[i + 1] = T[i] - S.gamma_m(T[i], p[i], rT) * dz
        p[i + 1] = p[i] * np.exp(-S.G * dz / (S.RD * Trho))
    return z, T, p, np.full_like(z, rT)


def test_moist_adiabat_has_constant_s_star_and_zero_nm2():
    z, T, p, rT = _moist_adiabat()
    s = S.s_star(T, p, rT)
    assert np.ptp(s) < 0.5                         # J/kg/K over 10 km (vs ~ 50 for a dry adiabat)
    n2m = S.n2_saturated(T, p, rT, z)[5:-5]
    n2 = S.n2_unsaturated(T, p, S.rsat(T, p), z)[5:-5]
    assert np.abs(n2m).max() < 2e-6                # neutral to saturated displacements ...
    assert n2.min() > 3e-5                         # ... but stable to dry ones: conditionally neutral


def test_unsaturated_n2_dry_values():
    z = np.linspace(0, 5000, 501)
    p = 1000 * np.exp(-z / 8000)
    rv = np.zeros_like(z)
    T_dry = 300 * (p / 1000) ** (S.RD / S.CPD)      # constant theta -> N^2 = 0
    assert np.abs(S.n2_unsaturated(T_dry, p, rv, z)).max() < 1e-7
    T = 300 - 0.0065 * z                            # standard lapse rate
    n2 = S.n2_unsaturated(T, p, rv, z)
    assert n2[1:-1].mean() == pytest.approx(9.81 / 290 * (0.00976 - 0.0065), rel=0.15)


def test_stability_profiles_on_composite_layout():
    z, T, p, rT = _moist_adiabat(ztop=6000., dz=50.)
    nz = len(z)
    T2 = np.stack([T, T - 0.002 * z])               # second "group": steeper lapse rate
    C = xr.Dataset(dict(TABS=(('group', 'z'), T2), QV=(('group', 'z'), 1000 * S.rsat(T2, np.stack([p, p])) * 0.8),
                        QN=(('group', 'z'), np.zeros((2, nz)))),
                   coords=dict(group=['adiabat', 'steep'], z=z, p=('z', p)))
    out = S.stability_profiles(C)
    assert out.N2_m.dims == ('group', 'z')
    np.testing.assert_allclose(out.N2_m, out.N2_m_entropy + out.N2_m_loading)
    # on the moist adiabat the entropy (conditional-instability) term is ~0; r_T = r*(z) decreases
    # upward, so the loading term is stabilising (a lifted parcel keeps its extra water)
    # (small, < 5e-6, not 0: the profile was built for a parcel that keeps its condensate, which
    #  cools more slowly aloft than saturated air without condensate; dry N^2 here is ~1e-4)
    assert np.abs(out.N2_m_entropy.sel(group='adiabat').values[2:-2]).max() < 6e-6
    assert (out.N2_m_loading.sel(group='adiabat').values[2:-2] > 0).all()
    assert (out.N2_m_entropy.sel(group='steep').values[2:-2] < 0).all()   # steeper than the moist adiabat
    assert set(np.unique(out.CLASS.values[~np.isnan(out.CLASS.values)])) <= {0, 1, 2}
