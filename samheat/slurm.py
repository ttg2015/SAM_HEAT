"""SLURM: the batch script, submission and queue queries.

Scheduler is a tiny interface so tests can swap in a fake one (tests/fake_slurm.py).
"""
import subprocess
import sys
from pathlib import Path

from .paths import SITE


def sh(cmd, cwd=None, check=True, modules=(), quiet=True):
    """Run a bash command (optionally after `module load`), return stdout."""
    if modules:
        cmd = 'module purge && ' + ' && '.join(f'module load {m}' for m in modules) + ' && ' + cmd
    r = subprocess.run(['bash', '-lc', cmd], cwd=str(cwd) if cwd else None,
                       capture_output=True, text=True)
    if not quiet:
        if r.stdout: print(r.stdout)
        if r.stderr: print('[stderr]', r.stderr)
    if check and r.returncode != 0:
        raise RuntimeError(f'Command failed (exit {r.returncode}): {cmd}\n{r.stderr}')
    return r.stdout


def batch_script(run_dir, job_name, walltime, auto_advance_cmd=None, site=SITE):
    """run.slurm text. With auto_advance_cmd, the job calls `advance` for itself after SAM
    exits normally. (A job killed at walltime never reaches that line: run advance()
    by hand; it will resubmit from the restart files.)"""
    s = site
    part = f'#SBATCH --partition={s.partition}\n' if s.partition else ''
    mods = '\n'.join(f'module load {m}' for m in s.modules)
    tail = ''
    if auto_advance_cmd:
        tail = f'''
# ---- AUTO_ADVANCE: take the next step (guarded; see samheat/stages.py) ----
if [ -f "{Path(run_dir).parent / 'STOP'}" ]; then echo "STOP file present: not advancing"; exit 0; fi
{auto_advance_cmd}
'''
    return f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={s.account}
{part}#SBATCH --nodes={s.nodes}
#SBATCH --ntasks-per-node={s.tasks_per_node}
#SBATCH --cpus-per-task=1
#SBATCH --time={walltime}
#SBATCH --mem={s.memory}
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --open-mode=append
#SBATCH --export=ALL

cd {run_dir}
module purge
unset LD_LIBRARY_PATH
{mods}
echo "===== SUBMISSION: $(date) (job $SLURM_JOB_ID) on $(hostname) ====="
mpirun -np $SLURM_NTASKS ./{s.executable}
status=$?
echo "===== END: $(date) (mpirun exit $status) ====="
[ $status -ne 0 ] && exit $status
{tail}'''


def advance_command(exp_root, run_key):
    """The command a job runs for AUTO_ADVANCE: this same Python, this same code."""
    repo = Path(__file__).resolve().parents[1]      # works without `pip install`
    return (f'PYTHONPATH={repo}:$PYTHONPATH {sys.executable} -m samheat.advance '
            f'--exp {exp_root} --run {run_key} --from-job $SLURM_JOB_ID')


class Slurm:
    """The real scheduler."""

    def submit(self, run_dir):
        return sh('sbatch run.slurm', cwd=run_dir).strip().split()[-1]

    def in_queue(self, job_id):
        if not job_id or job_id == 'DRYRUN':
            return False
        return bool(sh(f'squeue -h -j {job_id} 2>/dev/null', check=False).strip())

    def cancel(self, job_id):
        sh(f'scancel {job_id}', check=False)
