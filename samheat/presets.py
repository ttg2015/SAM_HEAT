"""Complete namelists. Every run's prm is rendered from one of these plus overrides; no prm
file is ever edited by hand or inherited from a template.

PAPER_VDD2025 is van der Drift & O'Gorman (2025) as actually run: the top-level prm of their
precip_extremes_RH_RCE repo (copied to case/FWRCE/prm_upstream_reference) with the stage
changes from SAM_RUN_SCRIPTS/land_sims.sh. Differences from that file, all output-only:
  * save2Dbin / save3Dbin = .true. (uncompressed binaries; our Python readers expect them)
  * nsave2Dend / nsave3Dend stay 999999999 (the v4 bug was 432000 inherited from a template)

Python values: bool -> .true./.false., str -> quoted, numbers as written.
"""
from copy import deepcopy

DAY = 8640          # model steps per day at dt = 10 s
NEVER = 999999999

SGS_TKE = {'dosmagor': False}
SLM_NML = {'landtype0': 16, 'LAI0': 0}

PAPER_VDD2025 = {
    # --- run control ---
    'caseid': '', 'caseid_restart': '', 'case_restart': '', 'nrestart': 0,
    'nstop': 50 * DAY, 'nprint': 180, 'nelapsemin': 420,
    # --- surface: ocean with van der Drift's evaporative resistance (oceflx.f90:246) ---
    'OCEAN': True, 'LAND': False, 'SLM': False, 'dosfcforcing': False,
    'ocean_type': 0, 'tabs_s': 300.0, 'delta_sst': 0.0, 'nxco2': 1.0,
    'r_stomata': 0.0,
    # --- SST: fixed unless dodynamicocean; sst_ft = relax to free-trop T (simple_ocean.f90) ---
    'timesimpleocean': 0.0, 'dodynamicocean': False, 'dossthomo': True,
    'depth_slab_ocean': 0.5, 'Szero': -107.5,
    'sst_ft': False, 'p_low': 400.0, 'p_upp': 600.0, 'tau_sst': 311040.0,
    'LES_S': False,
    # --- physics switches ---
    'dosgs': True, 'dodamping': True, 'doupperbound': True, 'docloud': True, 'doprecip': True,
    'docoriolis': False, 'dolongwave': True, 'doshortwave': True, 'dosurface': True,
    'dolargescale': False, 'doradforcing': False, 'SFC_FLX_FXD': False, 'SFC_TAU_FXD': False,
    'donudging_uv': False, 'donudging_tq': False, 'tauls': 7200.0,
    # --- insolation: PERPETUAL, constant sun (paper Sec. 2a; rad_full.f90:691-695) ---
    'doperpetual': True, 'dosolarconstant': True,
    'solar_constant': 565.0, 'zenith_angle': 42.332,
    'longitude0': -90.0, 'latitude0': 0.0, 'day0': 0.0, 'doseasons': False,
    # --- grid / time ---
    'dx': 1000.0, 'dy': 1000.0, 'dt': 10.0, 'nrad': 30, 'doradlon': False, 'doradlat': False,
    # --- statistics ---
    'nstat': 360, 'nstatfrq': 60,
    'doSAMconditionals': False, 'dosatupdnconditionals': False,
    # --- 3D / 2D output (off by default; stages switch it on) ---
    'nsave3D': 1080, 'nsave3Dstart': NEVER, 'nsave3Dend': NEVER,
    'save3Dbin': True, 'rad3Dout': True, 'save3Davg': False,
    'nsave2D': 1080, 'nsave2Dstart': NEVER, 'nsave2Dend': NEVER,
    'save2Dbin': True, 'save2Davg': False,
}

# The values a run must have to count as "perpetual sun like the paper".
PERPETUAL_REQUIRED = {'doperpetual': True, 'dosolarconstant': True,
                      'solar_constant': 565.0, 'zenith_angle': 42.332}


def paper_base():
    """A fresh copy of the paper namelist (safe to modify)."""
    return deepcopy(PAPER_VDD2025)
