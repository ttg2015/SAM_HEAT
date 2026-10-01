"""Reading SAM's stdout log (<run>.out, appended across resubmissions)."""
import re
from pathlib import Path


def _lines(out_file):
    p = Path(out_file)
    return p.read_text(errors='replace').splitlines() if p.exists() else []


def last_nstep(out_file):
    for l in reversed(_lines(out_file)):
        if l.strip().startswith('NSTEP'):
            try:
                return int(l.split('NSTEP')[1].split('NCYCLE')[0].strip('= '))
            except ValueError:
                return None
    return None


def finished_cleanly(out_file, nstop=None):
    """SAM printed its exit line in the last lines, and (if given) reached nstop.
    The nstop test matters because the log is appended: an old exit line can linger."""
    tail = _lines(out_file)[-20:]
    if not any('Finished with SAM, exiting' in l for l in tail):
        return False
    return nstop is None or (last_nstep(out_file) or 0) >= 0.99 * nstop


def health(out_file):
    """(max CFL seen, NaN/Inf seen) from the log."""
    max_cfl, bad = 0.0, False
    for l in _lines(out_file):
        if re.search(r'\bnan\b|\binfinity\b|\binf\b', l, re.I):
            bad = True
        if 'CFL' in l:
            for m in re.finditer(r'CFL\w*\s*=\s*([-+0-9.eE]+)', l):
                try:
                    max_cfl = max(max_cfl, float(m.group(1)))
                except ValueError:
                    pass
    return max_cfl, bad
