"""How far the x = 1750 m section is from a Gaussian.

Cunningham et al. 2005 state that the plume cross sections, even time-averaged,
do not exhibit a self-similar Gaussian structure; they give no number. This
reports one: a single 2-D Gaussian (amplitude, centre, two widths, no tilt) is
least-squares fitted to the time-mean theta section, and the residual RMS is
given as a fraction of the section's peak. A sum of two such Gaussians is
fitted alongside, so the table also says how much of the misfit is the
bifurcation itself. No pass threshold is applied: the paper supplies none,
and the threshold is the author's to set.

    END=1600 WINDOW=1000 python3 analyse_slices.py RUN_DIR
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

from plot_fig6 import CASES, read_slices

end = float(os.environ.get('END', 1600)); window = float(os.environ.get('WINDOW', 1000))
run_dir = sys.argv[1]


def gauss(p, Y, Z):
    A, y0, z0, sy, sz = p
    return A * np.exp(-0.5 * (((Y - y0) / sy) ** 2 + ((Z - z0) / sz) ** 2))


def fit(th, Y, Z, two):
    peak = th.max(); iy, iz = np.unravel_index(np.argmax(th), th.shape)
    mask = th > 0.0
    if two:
        # seed the two lobes on either side of the column-maximum profile's midpoint
        P = th.max(axis=1); mid = len(P) // 2
        l = int(np.argmax(P[:mid])); r = int(mid + np.argmax(P[mid:]))
        p0 = [P[l], Y[l, 0], Z[l, np.argmax(th[l])], 80, 80, P[r], Y[r, 0], Z[r, np.argmax(th[r])], 80, 80]
        model = lambda p: gauss(p[:5], Y, Z) + gauss(p[5:], Y, Z)
        lo = [0, 0, 0, 10, 10] * 2; hi = [10 * peak, 1200, 1000, 600, 600] * 2
    else:
        p0 = [peak, Y[iy, iz], Z[iy, iz], 150, 100]
        model = lambda p: gauss(p, Y, Z)
        lo = [0, 0, 0, 10, 10]; hi = [10 * peak, 1200, 1000, 600, 600]
    r = least_squares(lambda p: (model(p) - th).ravel(), p0, bounds=(lo, hi))
    resid = model(r.x) - th
    return np.sqrt(np.mean(resid ** 2)) / peak, np.sqrt(np.mean(resid[mask] ** 2)) / peak, r.x


print(f"time-mean section over {end - window:.0f}-{end:.0f} s; residual RMS as a fraction of the peak anomaly")
print(f"{'case':<18}{'peak K':>8}{'1 Gaussian':>12}{'in plume':>10}{'2 Gaussians':>13}{'in plume':>10}{'ratio 1/2':>11}")
for letter, tag, z0, q0 in CASES:
    t, F, y, z, _ = read_slices(os.path.join(run_dir, tag + '.slice'))
    sel = (t >= end - window - 1e-6) & (t <= end + 1e-6)
    th = F[sel].mean(axis=0)
    Y, Z = np.meshgrid(y, z, indexing='ij')
    r1, r1p, _ = fit(th, Y, Z, False)
    r2, r2p, _ = fit(th, Y, Z, True)
    print(f"{tag:<18}{th.max():>8.2f}{r1:>12.3f}{r1p:>10.3f}{r2:>13.3f}{r2p:>10.3f}{r1 / r2:>11.2f}")
print("'in plume' restricts the RMS to cells where the mean anomaly is positive;"
      " 'ratio 1/2' is how many times worse the single Gaussian fits than the pair.")
