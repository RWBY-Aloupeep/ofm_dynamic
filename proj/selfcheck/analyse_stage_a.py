import csv, os, math, statistics as st

WINDOW = 400.0  # last 200 s of a 600 s run

def stats(vals):
    if not vals:
        return (0.0, 0.0, 0.0, 0)
    m = st.fmean(vals)
    sd = st.pstdev(vals)
    return (m, sd, sd / math.sqrt(len(vals)), len(vals))

def load(root):
    out = {}
    for name in sorted(os.listdir(root)):
        if not name.endswith('.csv'):
            continue
        with open(os.path.join(root, name)) as f:
            rs = list(csv.DictReader(f))
        if not rs or abs(float(rs[-1]['time']) - 600.0) > 1e-6:
            continue
        late = [r for r in rs if float(r['time']) >= WINDOW]
        out[name[:-4]] = {
            'width': stats([float(r['theta_width']) for r in late]),
            'split': stats([float(r['theta_split']) for r in late if int(r['bifurcated'])]),
            'om': stats([float(r['best_omega']) for r in late]),
            'bif': st.fmean([int(r['bifurcated']) for r in late]),
            'dT': stats([float(r['max_theta']) for r in late]),
        }
    return out

def sep(a, b):
    """How many combined standard errors separate two (mean, sd, sem, n) tuples."""
    pooled = math.sqrt(a[2] ** 2 + b[2] ** 2)
    return (b[0] - a[0], (abs(b[0] - a[0]) / pooled if pooled else float('inf')))

ROOT = '/gscratch/amath/diwenxu/wildfire-sim-runs'
# Default: the second cut's pair. Any other sets are named on the command line
# as label=dir, e.g. analyse_stage_a.py open-xy=stage-a-3-open-xy-pr long=stage-a-3-long-closed-pr;
# a bare directory name is its own label. The pairwise table at the end compares
# every set against the first.
import sys
if len(sys.argv) > 1:
    sets = []
    for arg in sys.argv[1:]:
        label, _, d = arg.partition('=')
        d = d or label
        sets.append((label, d if os.path.isabs(d) else f'{ROOT}/{d}'))
else:
    sets = [('plain semi-Lagrangian', f'{ROOT}/stage-a-dense-plain'),
            ('BFECC + clamp',         f'{ROOT}/stage-a-dense-bfecc')]

tabs = {}
for label, root in sets:
    if not os.path.isdir(root):
        continue
    t = load(root)
    tabs[label] = t
    print(f"\n===== theta advection: {label} · mean over last 200 s =====")
    print(f"{'case':<20}{'width':>10}{'+-':>7}{'split':>10}{'+-':>7}"
          f"{'bif':>6}{'|om|':>9}{'maxdT':>8}")
    for k in sorted(t):
        v = t[k]
        print(f"{k:<20}{v['width'][0]:>10.1f}{v['width'][2]:>7.1f}"
              f"{v['split'][0]:>10.1f}{v['split'][2]:>7.1f}"
              f"{v['bif']:>6.2f}{v['om'][0]:>9.4f}{v['dT'][0]:>8.2f}")

for label, t in tabs.items():
    print(f"\n--- {label}: z0 ordering (deeper shear -> wider?) ---")
    for q in ('1000', '500'):
        for key in ('width', 'split'):
            row = [t.get(f'shear_z{z}_q{q}') for z in (50, 100, 150)]
            if any(r is None for r in row):
                continue
            vs = [r[key] for r in row]
            if all(v[3] == 0 for v in vs):
                continue
            mono = all(vs[i][0] < vs[i + 1][0] for i in range(2))
            d, k = sep(vs[0], vs[-1])
            print(f"  Q0={q:<5}{key:<6}: " + ", ".join(f"{v[0]:.1f}+-{v[2]:.1f}" for v in vs)
                  + f"  monotone={str(mono):<5} end-to-end {d:+.1f} m = {k:.1f}x sem"
                  + ("  SEPARABLE" if k > 2 else "  not separable"))

for label, t in tabs.items():
    print(f"\n--- {label}: Q0 effect (weaker source -> wider, per the paper) ---")
    for key in ('width', 'split'):
        for z in (50, 100, 150):
            a, b = t.get(f'shear_z{z}_q1000'), t.get(f'shear_z{z}_q500')
            if a is None or b is None:
                continue
            if a[key][3] == 0 or b[key][3] == 0:
                continue
            d, k = sep(a[key], b[key])
            verdict = 'paper' if d > 0 else 'OPPOSITE'
            print(f"  {key:<6} z0={z:<4}: strong {a[key][0]:7.1f}+-{a[key][2]:.1f}"
                  f"   weak {b[key][0]:7.1f}+-{b[key][2]:.1f}"
                  f"   weak-strong {d:+7.1f} m = {k:5.1f}x sem  -> {verdict}")

labels = [l for l, _ in sets if l in tabs]
if len(labels) >= 2:
    base = tabs[labels[0]]
    for other in labels[1:]:
        tb = tabs[other]
        print(f"\n--- {other} against {labels[0]}: split and width, with the separation in combined standard errors ---")
        print(f"{'case':<20}{'split base':>11}{'split':>9}{'diff':>8}{'x sem':>7}"
              f"{'width base':>12}{'width':>9}{'diff':>8}{'x sem':>7}{'maxdT b':>9}{'maxdT':>8}")
        for k in sorted(set(base) & set(tb)):
            a, b = base[k], tb[k]
            ds, ks = sep(a['split'], b['split']) if a['split'][3] and b['split'][3] else (float('nan'), float('nan'))
            dw, kw = sep(a['width'], b['width'])
            print(f"{k:<20}{a['split'][0]:>11.1f}{b['split'][0]:>9.1f}{ds:>+8.1f}{ks:>7.1f}"
                  f"{a['width'][0]:>12.1f}{b['width'][0]:>9.1f}{dw:>+8.1f}{kw:>7.1f}{a['dT'][0]:>9.2f}{b['dT'][0]:>8.2f}")
