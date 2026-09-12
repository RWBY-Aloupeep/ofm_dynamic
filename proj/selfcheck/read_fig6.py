"""Read the bifurcation off Cunningham et al. 2005 Fig. 6 by pixel.

The paper's two ordering statements -- wider bifurcation for the deeper shear
layer, and wider for the weaker source -- are read from this figure, and the
paper quotes only panels (e) and (f) for the second one. This script measures
each panel so both statements can be checked against the figure itself and
against our theta_split.

Input: the paper's PDF as attached in Zotero (proj:wildfire), which is a JBIG2
scan; Fig. 6 is on page index 7. Not committed. Usage:

    python3 read_fig6.py /path/to/cunningham2005.pdf

Per panel it reports
  * outer width: extent of the first plotted contour (300.25 K) along y;
  * peak split: distance between the two lobes' deepest-nested enclosed
    regions, i.e. the innermost closed contour of each lobe -- the figure's
    counterpart of theta_split;
  * gap: the y-range between the lobes with no ink at all.
Pixel-to-metre conversion uses the panel frame (0-1200 m). Expect a few tens
of metres of uncertainty from the scan and from contour crowding.

Needs pymupdf, numpy, scipy.
"""
import sys
from collections import deque

import numpy as np
import pymupdf
from scipy import ndimage

pdf = sys.argv[1] if len(sys.argv) > 1 else 'cunningham2005.pdf'
page = pymupdf.open(pdf)[7]
pix = page.get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
ink = a < 128

# Panel boxes in 110 dpi pixel coordinates of the rendered page, scaled.
s = 300 / 110
boxes = {'a': (140, 115, 395, 325), 'b': (455, 115, 710, 325),
         'c': (140, 355, 395, 570), 'd': (455, 355, 710, 570),
         'e': (140, 600, 395, 810), 'f': (455, 600, 710, 810)}
meta = {'a': (50, 1.0), 'b': (50, 0.5), 'c': (100, 1.0), 'd': (100, 0.5),
        'e': (150, 1.0), 'f': (150, 0.5)}

print(f"{'panel':<6}{'z0':>5}{'Q0':>5}{'outer width':>13}{'peak split':>12}{'gap':>7}")
for k, b in boxes.items():
    x0, y0, x1, y1 = [int(v * s) for v in b]
    sub = ink[y0:y1, x0:x1]
    rows = sub.mean(axis=1)
    cols = sub.mean(axis=0)
    r = np.where(rows > 0.3)[0]
    c = np.where(cols > 0.3)[0]
    top, bot = r.min(), r.max()
    left, right = c.min(), c.max()
    fx = right - left
    inner = sub[top + 30:bot - 30, left + 30:right - 30].copy()
    inner[:70, :120] = False            # the "(e)" panel letter
    ny, nx = inner.shape

    def ym(px):
        return (px + 30) / fx * 1200

    colink = inner.sum(axis=0)
    xs = np.where(colink > 0)[0]
    width = ym(xs.max()) - ym(xs.min())

    # Nesting depth of every enclosed white region: BFS over regions that
    # touch through ink, starting from the region touching the frame.
    ink2 = ndimage.binary_dilation(inner, iterations=1)
    lab, n = ndimage.label(~ink2)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    bg = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))) - {0}
    adj = {i: set() for i in range(1, n + 1)}
    for i in range(1, n + 1):
        if sizes[i - 1] < 15:
            continue
        m = ndimage.binary_dilation(lab == i, iterations=4)
        for j in np.unique(lab[m]):
            if j != 0 and j != i:
                adj[i].add(j)
                adj[j].add(i)
    depth = {i: 0 for i in bg}
    q = deque(bg)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    cents = {i: ndimage.center_of_mass(lab == i) for i in depth if sizes[i - 1] >= 15}
    L = [i for i in cents if cents[i][1] < nx / 2]
    R = [i for i in cents if cents[i][1] >= nx / 2]
    il = max(L, key=lambda i: depth[i])
    ir = max(R, key=lambda i: depth[i])
    split = ym(cents[ir][1]) - ym(cents[il][1])

    z = np.where(colink[xs.min():xs.max()] == 0)[0] + xs.min()
    runs = [g for g in np.split(z, np.where(np.diff(z) != 1)[0] + 1) if len(g) > 5] if len(z) else []
    gap = (ym(max(runs, key=len).max()) - ym(max(runs, key=len).min())) if runs else 0.0
    z0, q0 = meta[k]
    print(f"{k:<6}{z0:>5}{q0:>5}{width:>13.0f}{split:>12.0f}{gap:>7.0f}")
