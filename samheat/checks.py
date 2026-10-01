"""Checks on model OUTPUT (the prm-side guard lives in namelist.check_perpetual)."""
import math

import numpy as np

from .presets import PERPETUAL_REQUIRED

EXPECTED_SOLIN = PERPETUAL_REQUIRED['solar_constant'] * math.cos(math.radians(PERPETUAL_REQUIRED['zenith_angle']))


def perpetual_from_stat(stat, rel_tol=0.02):
    """Top-of-atmosphere insolation must be constant at solar_constant*cos(zenith) (~418 W/m2).
    With a diurnal cycle it swings between 0 and ~1360 W/m2.
    Returns dict(ok, mean, min, max, expected, message)."""
    if 'SOLIN' not in stat:
        return dict(ok=False, message='no SOLIN in STAT file')
    s = np.asarray(stat['SOLIN'].values, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return dict(ok=False, message='SOLIN is all NaN')
    mean, lo, hi = float(s.mean()), float(s.min()), float(s.max())
    ok = (abs(mean - EXPECTED_SOLIN) < rel_tol * EXPECTED_SOLIN
          and (hi - lo) < rel_tol * EXPECTED_SOLIN)
    msg = (f'SOLIN mean {mean:.1f} (min {lo:.1f}, max {hi:.1f}) W/m2; '
           f'expected constant {EXPECTED_SOLIN:.1f} -> {"PERPETUAL OK" if ok else "NOT PERPETUAL"}')
    return dict(ok=ok, mean=mean, min=lo, max=hi, expected=EXPECTED_SOLIN, message=msg)
