"""Write and read SAM's prm namelist, strictly.

render(): a complete prm from a dict. Unknown keys raise (the old "silently skipped" bug).
parse():  read a prm back into a dict.
write_prm(): render + write + read back + compare + perpetual-sun guard.
"""
import math
import re
from pathlib import Path

from .presets import PERPETUAL_REQUIRED, SGS_TKE, SLM_NML

# Every key of the &PARAMETERS namelist in SAM_SRC/setparm.f90 (FWRCE build, incl. rvd mods).
# A key not in this set would make SAM abort with "bad specification in PARAMETERS namelist".
VALID_KEYS = frozenset("""
dodamping doupperbound docloud doprecip dolongwave doshortwave dosgs dz doconstdz docoriolis
docoriolisz dosurface dolargescale doradforcing fluxt0 fluxq0 tau0 tabs_s z0 nelapse nelapsemin
dt dx dy fcor ug vg nstop caseid case_restart caseid_restart nstat nstatfrq nprint nrestart
doradsimple nsave3D nsave3Dstart nsave3Dend dosfcforcing donudging_uv donudging_tq donudging_t
donudging_q tauls tautqls nudging_uv_z1 nudging_uv_z2 nudging_t_z1 nudging_t_z2 nudging_q_z1
nudging_q_z2 dofplane timelargescale longitude0 latitude0 day0 nrad OCEAN LAND SFC_FLX_FXD
SFC_TAU_FXD soil_wetness doensemble nensemble dowallx dowally nsave2D nsave2Dstart nsave2Dend
qnsave3D docolumn save2Dbin save2Davg save3Davg save3Dbin save2Dsep save3Dsep dogzip2D dogzip3D
restart_sep doseasons doperpetual doradhomo dosfchomo doisccp domodis domisr dodynamicocean
ocean_type delta_sst depth_slab_ocean Szero deltaS timesimpleocean dosolarconstant
solar_constant zenith_angle rundatadir dotracers output_sep perturb_type doSAMconditionals
dosatupdnconditionals doscamiopdata iopfile dozero_out_day0 nstatmom nstatmomstart nstatmomend
savemomsep savemombin nmovie nmoviestart nmovieend nrestart_skip bubble_x0 bubble_y0 bubble_z0
bubble_radius_hor bubble_radius_ver bubble_dtemp bubble_dq dosmoke dossthomo rad3Dout nxco2
dosimfilesout notracegases doradlat doradlon ncycle_max doseawater SLM LES_S r_stomata sst_ft
p_low p_upp tau_sst
""".split())
_CANON = {k.lower(): k for k in VALID_KEYS}   # Fortran namelists are case-insensitive


class NamelistError(ValueError):
    pass


def canonical(key):
    try:
        return _CANON[key.lower()]
    except KeyError:
        raise NamelistError(f'{key!r} is not a &PARAMETERS key in setparm.f90 (typo?)') from None


def merge(base, overrides):
    """base + overrides, with every override key checked against setparm.f90."""
    out = {canonical(k): v for k, v in base.items()}
    for k, v in (overrides or {}).items():
        out[canonical(k)] = v
    return out


def _fmt(v):
    if isinstance(v, bool):
        return '.true.' if v else '.false.'
    if isinstance(v, str):
        return "'" + v + "'"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, int):
        return str(v)
    raise NamelistError(f'cannot write value {v!r} of type {type(v).__name__}')


def _group(name, d):
    body = ''.join(f' {k} = {_fmt(v)},\n' for k, v in d.items())
    return f'&{name}\n{body}/\n'


def render(params, header=''):
    """Full prm text: &SGS_TKE, &SLM, &PARAMETERS (one key per line)."""
    for k in params:
        canonical(k)
    head = ''.join(f'! {l}\n' for l in header.splitlines()) if header else ''
    return head + _group('SGS_TKE', SGS_TKE) + _group('SLM', SLM_NML) + _group('PARAMETERS', params)


def _parse_value(s):
    s = s.strip()
    low = s.lower()
    if low in ('.true.', 't', 'true'):
        return True
    if low in ('.false.', 'f', 'false'):
        return False
    if s[:1] in "'\"":
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        return float(s)


def parse(text, group='PARAMETERS'):
    """Read one namelist group into {canonical key: value} (comments stripped)."""
    m = re.search(rf'&{group}\b(.*?)^\s*/', text, re.S | re.M | re.I)
    if not m:
        raise NamelistError(f'no &{group} group found')
    body = re.sub(r'!.*', '', m.group(1))
    out = {}
    for k, v in re.findall(r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|[^,\n]+)", body):
        out[canonical(k) if group == 'PARAMETERS' else k] = _parse_value(v)
    return out


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def check_perpetual(params):
    """Problems if the run is not perpetual, constant sun as in the paper."""
    return [f'{k} = {params.get(k)!r} (paper: {v!r})'
            for k, v in PERPETUAL_REQUIRED.items() if not _same(params.get(k), v)]


def write_prm(path, params, allow_diurnal=False, header=''):
    """Write the prm, read it back, check every value, enforce perpetual sun."""
    bad = check_perpetual(params)
    if bad and not allow_diurnal:
        raise NamelistError('Not perpetual sun like the paper:\n  ' + '\n  '.join(bad) +
                            '\nSet ALLOW_DIURNAL=True if you really want this.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(params, header))
    back = parse(path.read_text())
    diff = [k for k in params if k not in back or not _same(back[k], params[k])]
    if diff or set(back) != set(params):
        raise NamelistError(f'{path}: read-back mismatch for {diff or set(back) ^ set(params)}')
    return path


def read_prm(path):
    return parse(Path(path).read_text())
