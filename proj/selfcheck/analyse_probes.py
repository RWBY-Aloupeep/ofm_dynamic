"""Shedding frequency from the wake probes written by `selfcheck --test plume --probes`.

Cunningham et al. 2005 report vortex shedding in the plume's wake at a
Strouhal number St = f D / U ~ 0.25 with D the source diameter (225 m)
and U the ambient wind (4.5 m/s), i.e. f = 0.005 Hz, a 200 s period.
This script takes each probe's lateral velocity v(t) over a window,
detrends it, averages Hann-windowed periodograms over 800 s segments
(Welch), and reports the frequency of the spectral peak within
[1/600, 1/20] Hz, its Strouhal number, and how far
the peak stands above the spectrum's median (a flat spectrum has ratio
~1; a clean line gives >> 10). The same is done for w and theta so a
signal that is a lateral flapping (v) can be told from a pulsing (w,
theta). A figure with the time series and the spectra is written when an
output name is given.

    python3 analyse_probes.py RUN.probes [--t0 1000] [--t1 3000] [--png out.png]
"""
import argparse
import re

import numpy as np

D_SOURCE = 225.0   # m, source diameter (Cunningham et al. 2005)
U_WIND = 4.5       # m/s, ambient wind
F_LO, F_HI = 1.0 / 600.0, 1.0 / 20.0


def read_probes(path):
    with open(path) as f:
        f.readline()
        pos_line = f.readline()
    pos = [tuple(float(v) for v in m.split()) for m in re.findall(r'\(([^)]*)\)', pos_line.split(':', 1)[1])]
    ncol = 1 + 4 * len(pos)
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            tok = line.split()
            if len(tok) == ncol:      # a run still writing leaves a partial last line
                rows.append([float(v) for v in tok])
    data = np.array(rows)
    t = data[:, 0]
    n = (data.shape[1] - 1) // 4
    u = data[:, 1::4][:, :n]; v = data[:, 2::4][:, :n]; w = data[:, 3::4][:, :n]; th = data[:, 4::4][:, :n]
    return t, pos, u, v, w, th


def spectrum(t, x, seg_s=800.0):
    """Welch estimate: linear detrend, Hann-windowed segments of seg_s
    seconds at 50% overlap, periodograms averaged. Resolution 1/seg_s Hz
    (St 0.0625 at 800 s), enough to tell St 0.25 from the 0.14 a 350 s
    window puts in its lowest bin."""
    dt = float(np.median(np.diff(t)))
    y = x - np.polyval(np.polyfit(t, x, 1), t)
    m = min(len(y), int(round(seg_s / dt)))
    win = np.hanning(m)
    acc = None; k = 0
    for s0 in range(0, len(y) - m + 1, max(1, m // 2)):
        seg = y[s0:s0 + m]
        seg = (seg - seg.mean()) * win
        pk = np.abs(np.fft.rfft(seg)) ** 2
        acc = pk if acc is None else acc + pk
        k += 1
    f = np.fft.rfftfreq(m, dt)
    return f, acc / k


def peak(f, p):
    band = (f >= F_LO) & (f <= F_HI)
    fb, pb = f[band], p[band]
    i = int(np.argmax(pb))
    ratio = pb[i] / max(np.median(pb), 1e-30)
    return fb[i], ratio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('probes')
    ap.add_argument('--t0', type=float, default=1000.0)
    ap.add_argument('--t1', type=float, default=1e9)
    ap.add_argument('--png', default=None)
    a = ap.parse_args()
    t, pos, u, v, w, th = read_probes(a.probes)
    sel = (t >= a.t0) & (t <= a.t1)
    t = t[sel]; u = u[sel]; v = v[sel]; w = w[sel]; th = th[sel]
    print(f'# window {t[0]:.0f}-{t[-1]:.0f} s, {len(t)} samples, dt {np.median(np.diff(t)):.3f} s')
    print(f'# St = f D / U with D = {D_SOURCE:.0f} m, U = {U_WIND:.1f} m/s; St 0.25 <-> f {0.25 * U_WIND / D_SOURCE:.4f} Hz, period {D_SOURCE / (0.25 * U_WIND):.0f} s')
    print('# probe  x y z | rms v  f_v Hz  period s  St_v  peak/median | rms w  f_w  St_w  peak/median | rms th  f_th  St_th  peak/median')
    rows = []
    for q in range(len(pos)):
        out = [f'{q:2d}  {pos[q][0]:5.0f} {pos[q][1]:4.0f} {pos[q][2]:4.0f} |']
        rec = {}
        for name, x in (('v', v[:, q]), ('w', w[:, q]), ('th', th[:, q])):
            f, p = spectrum(t, x)
            fp, ratio = peak(f, p)
            rec[name] = (x.std(), fp, ratio, f, p)
            if name == 'v':
                out.append(f'{x.std():6.3f} {fp:8.5f} {1 / fp:7.0f} {fp * D_SOURCE / U_WIND:6.3f} {ratio:8.1f} |')
            else:
                out.append(f'{x.std():6.3f} {fp:8.5f} {fp * D_SOURCE / U_WIND:6.3f} {ratio:8.1f} |')
        print(' '.join(out))
        rows.append(rec)

    if a.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = len(pos)
        fig, axes = plt.subplots(n, 2, figsize=(12, 1.6 * n), squeeze=False)
        for q in range(n):
            ax = axes[q, 0]
            ax.plot(t, v[:, q] - v[:, q].mean(), lw=0.6, label='v')
            ax.plot(t, w[:, q] - w[:, q].mean(), lw=0.6, label='w', alpha=0.7)
            ax.set_ylabel(f'p{q} ({pos[q][0]:.0f},{pos[q][1]:.0f},{pos[q][2]:.0f})', fontsize=7)
            ax.tick_params(labelsize=6)
            if q == 0:
                ax.legend(fontsize=6, loc='upper right')
            ax = axes[q, 1]
            for name, c in (('v', 'C0'), ('w', 'C1'), ('th', 'C2')):
                _, fp, ratio, f, p = rows[q][name]
                band = (f >= F_LO) & (f <= F_HI)
                ax.semilogy(f[band], p[band] / max(p[band].max(), 1e-30), c, lw=0.7, label=name)
            ax.axvline(0.25 * U_WIND / D_SOURCE, color='k', ls='--', lw=0.6)
            ax.set_xlim(F_LO, F_HI)
            ax.set_ylim(1e-4, 1.5)
            ax.tick_params(labelsize=6)
            if q == 0:
                ax.legend(fontsize=6, loc='upper right')
                ax.set_title('normalised power spectra; dashed = St 0.25', fontsize=7)
        axes[-1, 0].set_xlabel('t (s)', fontsize=7)
        axes[-1, 1].set_xlabel('f (Hz)', fontsize=7)
        fig.tight_layout()
        fig.savefig(a.png, dpi=140)
        print('wrote', a.png)


if __name__ == '__main__':
    main()
