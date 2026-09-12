"""Count and place the transverse rollers on the plume's upper face.

From a centreline section (`--slice-xz`), for each x the strongest
clockwise omega_y (du/dz - dw/dx > 0, seen from +y) inside the plume
(theta anomaly above a threshold) is taken; the rollers are the local
maxima of that curve along x that exceed a fraction of the curve's global
maximum. Reported per time: the height of the plume's upper edge where
the first roller sits (the top of the laminar base), the roller
positions, their mean spacing, and the count.

    python3 xz_rollers.py RUN.slice [--times 600 900] [--theta 0.5] [--frac 0.4] [--xmin 700]
"""
import argparse

import numpy as np

from plot_xz import omega_y, read_xz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('slice')
    ap.add_argument('--times', type=float, nargs='+', default=[600.0, 900.0])
    ap.add_argument('--theta', type=float, default=0.5, help='theta anomaly (K) that bounds the plume')
    ap.add_argument('--frac', type=float, default=0.4, help='a roller must exceed this fraction of the strongest')
    ap.add_argument('--min', type=float, default=0.15,
                    help='and this absolute omega_y (1/s); the laminar shear layer at mu = 4 peaks below 0.1')
    ap.add_argument('--xmin', type=float, default=700.0, help='ignore the source region upstream of this x')
    ap.add_argument('--zmax', type=float, default=1300.0)
    a = ap.parse_args()
    t, th, u, w, x, z, dx, y_used = read_xz(a.slice)
    om = omega_y(u, w, dx)
    for tt in a.times:
        i = int(np.argmin(np.abs(t - tt)))
        inside = (th[i] >= a.theta) & (z[None, :] <= a.zmax)
        col = np.where(inside, om[i], 0.0).max(axis=1)   # strongest clockwise omega_y per x
        col[x < a.xmin] = 0.0
        peak = col.max()
        thr = max(a.frac * peak, a.min)
        # local maxima along x, at least 3 cells apart
        idx = [j for j in range(1, len(col) - 1) if col[j] >= thr and col[j] > col[j - 1] and col[j] >= col[j + 1]]
        keep = []
        for j in idx:
            if not keep or j - keep[-1] >= 3:
                keep.append(j)
        xs = x[keep]
        zs = []
        for j in keep:
            zs.append(z[np.argmax(np.where(inside[j], om[i][j], -1e9))])
        spacing = np.diff(xs).mean() if len(xs) > 1 else float('nan')
        # the plume's upper edge (highest theta >= thr) at the first roller's x
        if keep:
            j0 = keep[0]
            edge = z[inside[j0]].max()
        else:
            edge = float('nan')
        print(f't = {t[i]:.0f} s: rollers {len(keep)}, strongest omega_y {peak:.3f} 1/s, first at x = {xs[0] if keep else float("nan"):.0f} m '
              f'(z = {zs[0] if keep else float("nan"):.0f} m, plume upper edge there {edge:.0f} m), mean spacing {spacing:.0f} m')
        print('   x (m): ' + ' '.join(f'{v:.0f}' for v in xs))
        print('   z (m): ' + ' '.join(f'{v:.0f}' for v in zs))
        print('   omega_y: ' + ' '.join(f'{col[j]:.3f}' for j in keep))


if __name__ == '__main__':
    main()
