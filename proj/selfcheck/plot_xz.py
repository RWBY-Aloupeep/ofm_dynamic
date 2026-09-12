"""Draw the plume's centreline x-z section: theta contours over omega_y.

Reads the sections written by `selfcheck --test plume --slice-xz`, computes
the transverse vorticity omega_y = du/dz - dw/dx by central differences on
the cell-centred velocity, and draws it as a diverging colour field with
the theta anomaly contoured at 0.25, 0.5, 1, 2, 4, ... K on top, at the requested times. This
is where Cunningham et al. 2005 place the onset of transition: shear-layer
(Kelvin-Helmholtz) rollers on the plume's upstream face, at the top of a
laminar base, with the plume turbulent above them.

    python3 plot_xz.py RUN.slice out.png --times 300 600 900 [--xmax 1200] [--zmax 1000]
    python3 plot_xz.py A.slice B.slice out.png --times 600 --labels "mu = 4" "mu = 1"

(the output .png is the last positional argument.)
"""
import argparse
import struct

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_xz(path):
    with open(path, 'rb') as f:
        assert f.read(8)[:7] == b'OFMSLXZ', path
        nx, nz = struct.unpack('<ii', f.read(8))
        dx, x0, z0, y_used = struct.unpack('<ffff', f.read(16))
        n = nx * nz
        rec = 4 + 3 * 4 * n
        times, th, u, w = [], [], [], []
        while True:
            buf = f.read(rec)
            if len(buf) < rec:
                break
            times.append(struct.unpack('<f', buf[:4])[0])
            a = np.frombuffer(buf[4:], dtype='<f4')
            th.append(a[:n].reshape(nx, nz)); u.append(a[n:2 * n].reshape(nx, nz)); w.append(a[2 * n:].reshape(nx, nz))
    x = x0 + dx * np.arange(nx); z = z0 + dx * np.arange(nz)
    return np.array(times), np.array(th), np.array(u), np.array(w), x, z, dx, y_used


def omega_y(u, w, dx):
    dudz = np.gradient(u, dx, axis=1)
    dwdx = np.gradient(w, dx, axis=0)
    return dudz - dwdx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='section files (.slice), then the output .png last')
    ap.add_argument('--times', type=float, nargs='+', default=[600.0])
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--xmax', type=float, default=1800.0)
    ap.add_argument('--zmax', type=float, default=1200.0)
    ap.add_argument('--xmin-scale', type=float, default=700.0,
                    help='the colour limit is the 99.5th percentile of |omega_y| downstream of this x (the source region is far stronger)')
    a = ap.parse_args()
    files, out = a.files[:-1], a.files[-1]
    times = a.times
    xmax, zmax = a.xmax, a.zmax
    labels = a.labels if a.labels else files

    rows = len(files); cols = len(times)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), squeeze=False)
    for r, path in enumerate(files):
        t, th, u, w, x, z, dx, y_used = read_xz(path)
        om = omega_y(u, w, dx)
        sel = om[:, (x >= a.xmin_scale) & (x <= xmax), :][:, :, (z <= zmax)]
        lim = np.percentile(np.abs(sel), 99.5)
        for c, tt in enumerate(times):
            i = int(np.argmin(np.abs(t - tt)))
            ax = axes[r, c]
            X, Z = np.meshgrid(x, z, indexing='ij')
            ax.pcolormesh(X, Z, om[i], cmap='RdBu_r', vmin=-lim, vmax=lim, shading='nearest', rasterized=True)
            levels = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
            ax.contour(X, Z, th[i], levels=levels, colors='k', linewidths=0.4)
            ax.set_xlim(0, xmax); ax.set_ylim(0, zmax); ax.set_aspect('equal')
            ax.set_title(f'{labels[r]}   t = {t[i]:.0f} s   omega_y in [{-lim:.3f}, {lim:.3f}] 1/s', fontsize=8)
            ax.set_xlabel('x (m)', fontsize=8); ax.set_ylabel('z (m)', fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(f'Centreline section (y = {y_used:.0f} m): omega_y colour (red = clockwise seen from +y), theta anomaly contours at 0.25, 0.5, 1, 2, 4, ... K', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=170)
    print('wrote', out)


if __name__ == '__main__':
    main()
