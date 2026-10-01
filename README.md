# SAM_HEAT

Clean, reproducible launcher and analysis for SAM (FWRCE build) radiative-convective
equilibrium experiments over surfaces of varying evaporative resistance `r_v`.

1. **Reproduce** van der Drift & O'Gorman (2025), *Dependence of convective precipitation
   extremes on near-surface relative humidity*.
2. **Extend** the same top-down framework to heat extremes and their vertical structure.

Code is written on the Mac, pushed to GitHub and pulled on NYU Torch, where all runs and
notebooks execute. Model output stays on Torch.

## Quickstart (on Torch)
```bash
git clone <repo-url> SAM_HEAT && cd SAM_HEAT
git config core.hooksPath .githooks
```
No install needed: the first cell of every notebook adds this repo to Python's path. (Optional
alternative: `pip install -e .` into your Jupyter environment.) Updating is just `git pull`.
1. `notebooks/00_Build_and_Smoke.ipynb`: checks the paths, prints the prm, runs 2 hours, and checks that the sun is perpetual.
2. `notebooks/01_Launch_Control.ipynb`: edit the **LEVERS** cell, then run **Start** once and **Advance** whenever you want.
3. `notebooks/02_Repro_vdD2025.ipynb`: Table 1 and Figs. 1–3 compared with the paper.

Paths, modules and SLURM account are set in **`samheat/paths.py`**, and nowhere else.

## The protocol (paper Sec. 2)
| Stage | What | Key settings |
|---|---|---|
| ocean | r_v = 0, fixed Ts = 300 K, 50 days | `snd_ref` and `T_ref` come from the last 10 days |
| 1 | Ts relaxes so the 600–400 hPa mean T returns to `T_ref` | `dodynamicocean`, `sst_ft`, `tau_sst`=3.6 d; 40 days, then +10 d until within 0.1 K |
| 2 | fixed Ts (the mean of Stage 1's last 10 days), 60 days from rest | 2D and 3D snapshots every 3 h over days 30–60 |

Every run uses **perpetual sun**: `doperpetual`, `dosolarconstant`, `solar_constant=565`,
`zenith_angle=42.332`. That gives TOA insolation of 418 W/m², constant in time.

## Safety rails (why this repo exists)
- **The prm is generated, never edited.** It is rendered from `samheat/presets.py`, read back and verified. Unknown keys raise an error, which fixes the old "silently skipped" bug.
- **Perpetual-sun guard.** It applies twice: on the prm before submitting, and on the model output (`SOLIN` constant) before a run is accepted. The SAM_CODE runs had a diurnal cycle because `FWRCE/prm` never set `doperpetual`.
- **No inherited output windows.** The `nsave*End=999999999` setting is explicit; the v4 3D output had stopped at day 50.
- **Stage 2 starts from the ocean-reference sounding.** The v3 sweep used the template sounding.
- **`advance()` cannot loop forever.** A run stops at `max_submits` per stage, at Stage 1 `max_days`, or after two no-progress resubmissions. NaN marks a run `corrupted`. A `STOP` file freezes everything.
- **Physics settings are frozen per experiment.** To change them, use a new experiment name.
- **Every run folder has a `config.json`** recording every setting plus the samheat git hash. `ledger.json` records every action.

## Layout
```
samheat/   paths presets namelist staging slurm ledger stages advance logs convergence
           sounding checks  io/ (stat, sam2d, sam3d)  analysis/ (vdD scripts + repro)
case/FWRCE grd (64 levels), snd_vdd300K (van der Drift's sounding), prm_upstream_reference
notebooks/ 00 build+smoke, 01 launch control, 02 reproduction, 10 heat statistics
tests/     fast pytest suite with a fake SLURM:  python -m pytest -q
```

## Notes and caveats
- The vertical grid and sounding are van der Drift's (first level 25 m, dz = 50 m near the surface). The paper text mentions 37.5 m spacing and a 40 m lowest level; we follow his actual files.
- The upstream runs wrote compressed output (`save*bin=F`). We write `.2Dbin` and `.bin3D` (`save*bin=T`) for the Python readers. This does not change the physics.
- An explanation of the SAM source is in the Obsidian vault: `Cloud_Permitting_Model_Heat_Extremes/SAM Under the Hood/`.

## Provenance
SAM: Marat Khairoutdinov. FWRCE modifications (`r_stomata`, SST relaxation) and the analysis
scripts in `samheat/analysis/` (vendored): Robert van der Drift with Paul O'Gorman
(`precip_extremes_RH_RCE`). `lcl.py`: David Romps (2017).
