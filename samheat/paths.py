"""Where things live on Torch. The ONE place to edit paths, modules and SLURM settings.

Everything else in samheat imports SITE from here. Override any field from a notebook with
    from samheat.paths import SITE; SITE.work_root = Path('/scratch/.../my_test')
"""
from dataclasses import dataclass, field
from pathlib import Path

SCM = Path('/scratch/ttg2015/Torch/ExtremeHeat/SCM')


@dataclass
class Site:
    # --- model build ---
    build: Path = SCM / 'SAM_FWRCE_build'                          # holds the executable
    executable: str = 'SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM'
    case: str = 'FWRCE'
    rundata_source: Path = SCM / 'SAM_FWRCE_build'                 # RUNDATA/ + radiation *.nc live here
    util_dir: Path = SCM / 'SAM6.11.8' / 'UTIL'                    # stat2nc, bin3D2nc, 2Dbin2nc
    # --- where new experiments go ---
    work_root: Path = SCM / 'SAM_HEAT_runs'
    # --- environment ---
    modules: tuple = ('intel/2025.2', 'hdf5/intel/1.12.0', 'netcdf-c/intel/4.7.4',
                      'netcdf-fortran/intel/4.5.3', 'openmpi/intel/5.0.8')
    python_activate: str = ''          # e.g. 'source ~/miniconda3/bin/activate samheat' (used by AUTO_ADVANCE)
    # --- SLURM ---
    account: str = 'torch_pr_673_general'
    partition: str = ''                # '' = cluster default
    nodes: int = 1
    tasks_per_node: int = 32           # must equal nsubdomains_x * nsubdomains_y in domain.f90 (4 x 8)
    memory: str = '32GB'
    walltime: dict = field(default_factory=lambda: {
        'ocean': '10:00:00', 'stage1': '08:00:00', 'stage2': '12:00:00', 'smoke': '00:30:00'})


SITE = Site()
