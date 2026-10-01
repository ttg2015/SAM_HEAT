"""Command-line entry used by jobs when AUTO_ADVANCE is on:
    python -m samheat.advance --exp <experiment dir> --run <key> --from-job $SLURM_JOB_ID
It runs exactly the same Experiment.advance() as the notebook."""
import argparse

from .stages import Experiment


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--run', default=None)
    ap.add_argument('--from-job', default=None)
    a = ap.parse_args(argv)
    Experiment.from_dir(a.exp).advance(run=a.run, from_job=a.from_job)


if __name__ == '__main__':
    main()
