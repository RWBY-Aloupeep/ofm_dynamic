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

if len(tabs) == 2:
    ta, tb = tabs['plain semi-Lagrangian'], tabs['BFECC + clamp']
    print("\n--- what the theta scheme changes ---")
    print(f"{'case':<20}{'width plain':>12}{'width bfecc':>12}{'d%':>8}"
          f"{'maxdT plain':>13}{'maxdT bfecc':>12}{'bif p':>7}{'bif b':>7}")
    for k in sorted(set(ta) & set(tb)):
        wp, wb = ta[k]['width'][0], tb[k]['width'][0]
        print(f"{k:<20}{wp:>12.1f}{wb:>12.1f}{(wb - wp) / wp * 100:>7.1f}%"
              f"{ta[k]['dT'][0]:>13.2f}{tb[k]['dT'][0]:>12.2f}"
              f"{ta[k]['bif']:>7.2f}{tb[k]['bif']:>7.2f}")
