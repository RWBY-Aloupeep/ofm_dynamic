"""Draw Cunningham et al. 2005 Fig. 6 from our runs, and set it beside the paper's.

Reads the theta sections written by `selfcheck --test plume --slice`, averages
them over the last WINDOW seconds of a run of END seconds (the same window the
orderings are read over), and contours the anomaly every 0.25 K starting at
0.25 K -- the figure's own contour spec -- on the paper's axes (y 0-1200 m,
z 0-1000 m), in the paper's panel order: left column Q0 = 1 kW/m^3, right
column 0.5, rows z0 = 50, 100, 150 m.

    END=1600 WINDOW=1000 python3 plot_fig6.py RUN_DIR out.png [--pdf cunningham2005.pdf]
    python3 plot_fig6.py RUN_DIR out.png --time 600      # one snapshot instead

With --pdf, a second figure (out_vs_paper.png) puts each of the paper's scanned
panels beside ours, row by row.
"""
import os
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CASES = [('a', 'shear_z50_q1000', 50, 1.0), ('b', 'shear_z50_q500', 50, 0.5),
         ('c', 'shear_z100_q1000', 100, 1.0), ('d', 'shear_z100_q500', 100, 0.5),
         ('e', 'shear_z150_q1000', 150, 1.0), ('f', 'shear_z150_q500', 150, 0.5)]
PDF_BOXES = {'a': (140, 115, 395, 325), 'b': (455, 115, 710, 325),
             'c': (140, 355, 395, 570), 'd': (455, 355, 710, 570),
             'e': (140, 600, 395, 810), 'f': (455, 600, 710, 810)}   # 110 dpi page pixels


def read_slices(path):
    with open(path, 'rb') as f:
        magic = f.read(8)
        assert magic == b'OFMSLICE', path
        ny, nz = struct.unpack('<ii', f.read(8))
        dx, y0, z0, plane_x = struct.unpack('<ffff', f.read(16))
        times, fields = [], []
        rec = 4 + 4 * ny * nz
        while True:
            buf = f.read(rec)
            if len(buf) < rec:
                break
            times.append(struct.unpack('<f', buf[:4])[0])
            fields.append(np.frombuffer(buf[4:], dtype='<f4').reshape(ny, nz))
    y = y0 + dx * np.arange(ny)
    z = z0 + dx * np.arange(nz)
    return np.array(times), np.array(fields), y, z, plane_x


def section(path, end, window, snapshot):
    t, F, y, z, plane_x = read_slices(path)
    if snapshot is not None:
        i = int(np.argmin(np.abs(t - snapshot)))
        return F[i], y, z, f't = {t[i]:.0f} s'
    sel = (t >= end - window - 1e-6) & (t <= end + 1e-6)
    return F[sel].mean(axis=0), y, z, f'mean {end - window:.0f}-{end:.0f} s ({sel.sum()} samples)'


def draw_panel(ax, th, y, z, letter, label):
    levels = np.arange(0.25, max(th.max(), 0.5) + 0.25, 0.25)
    ax.contour(y, z, th.T, levels=levels, colors='k', linewidths=0.6)
    ax.set_xlim(0, 1200); ax.set_ylim(0, 1000)
    ax.set_xticks([0, 1200]); ax.set_yticks([0, 1000])
    ax.set_xlabel('y (m)', fontsize=8); ax.set_ylabel('z (m)', fontsize=8)
    ax.tick_params(labelsize=7, direction='in', length=3)
    ax.text(0.04, 0.92, f'({letter})', transform=ax.transAxes, fontsize=9, style='italic')
    ax.text(0.97, 0.04, label, transform=ax.transAxes, fontsize=6, ha='right', color='0.35')
    ax.set_aspect('equal')


def main():
    args = sys.argv[1:]
    run_dir, out = args[0], args[1]
    pdf = args[args.index('--pdf') + 1] if '--pdf' in args else None
    snapshot = float(args[args.index('--time') + 1]) if '--time' in args else None
    end = float(os.environ.get('END', 1600)); window = float(os.environ.get('WINDOW', 1000))

    fig, axes = plt.subplots(3, 2, figsize=(6.4, 7.6))
    ours = {}
    for (letter, tag, z0, q0), ax in zip(CASES, axes.ravel()):
        th, y, z, label = section(os.path.join(run_dir, tag + '.slice'), end, window, snapshot)
        ours[letter] = (th, y, z)
        draw_panel(ax, th, y, z, letter, label)
    fig.suptitle('Potential temperature at x = 1750 m, contours every 0.25 K from 300.25 K\n'
                 'left Q0 = 1 kW/m3, right 0.5; rows z0 = 50, 100, 150 m', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200)
    print('wrote', out)

    if pdf:
        import pymupdf
        page = pymupdf.open(pdf)[7]
        pix = page.get_pixmap(dpi=220, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        s = 220 / 110
        fig, axes = plt.subplots(3, 4, figsize=(11, 7.4))
        for r, row in enumerate(axes):
            for c, letter in enumerate((CASES[2 * r][0], CASES[2 * r + 1][0])):
                x0, y0, x1, y1 = [int(v * s) for v in PDF_BOXES[letter]]
                axp, axo = row[2 * c], row[2 * c + 1]
                axp.imshow(img[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=255)
                axp.set_axis_off()
                axp.set_title(f'paper ({letter})', fontsize=8)
                th, y, z = ours[letter]
                draw_panel(axo, th, y, z, letter, '')
                axo.set_title('this run', fontsize=8)
        fig.suptitle('Cunningham et al. 2005 Fig. 6 (scan) beside the same six cases from the flow-map solver', fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out2 = os.path.splitext(out)[0] + '_vs_paper.png'
        fig.savefig(out2, dpi=200)
        print('wrote', out2)


if __name__ == '__main__':
    main()
