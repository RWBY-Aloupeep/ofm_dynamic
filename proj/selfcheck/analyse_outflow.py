"""Reads the translating-vortex verification runs and puts the open boundary's
error next to the solver's own.

    python3 analyse_outflow.py [/gscratch/amath/diwenxu/wildfire-sim-runs/outflow-v2]

The long box is measured on exactly the nodes of the short one, so at every
sample the difference between the open box and the long box is what the open
face did: to the circulation still inside the measured rectangle, to the
vorticity field in the interior, and to what is left behind after the vortex
has gone. The closed box, the treatment being replaced, is reported the same
way. The proposed pass criteria are printed with the numbers; they are proposals
until the plan Artifact adopts them.
"""
import csv, os, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else '/gscratch/amath/diwenxu/wildfire-sim-runs/outflow-v2'
GAMMA = 0.25
X_OUT = 2.0
A = 0.05


def load(name):
    path = os.path.join(ROOT, name + '.csv')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


for n in (1, 5):
    runs = {m: load(f'vortex_{m}_n{n}') for m in ('open', 'conv', 'closed', 'long')}
    if runs['conv'] is None:
        del runs['conv']
    if any(r is None for r in runs.values()):
        print(f'n = {n}: missing runs', [m for m, r in runs.items() if r is None])
        continue
    tested = [m for m in ('open', 'conv', 'closed') if m in runs]
    print(f'\n===== n = {n} =====')
    print(f"{'t':>6}{'x_c':>7} | {'G_in exact':>10}{'long':>9}" + ''.join(f"{m:>9}" for m in tested) + " | "
          f"{'G_core ex':>9}{'long':>8}" + ''.join(f"{m:>8}" for m in tested) + f" | {'L2int long':>10}" + ''.join(f"{m:>8}" for m in tested)
          + f" | {'peak long':>9}" + ''.join(f"{m:>7}" for m in tested))
    s = {m: {'dG': 0.0, 'dL2': 0.0, 'dGex': 0.0, 'res': None, 'res_peak': None, 'peak_face': None,
             'min_w': 0.0} for m in tested}
    solver = {'gdev': 0.0, 'l2': 0.0, 'wall': None}
    for rows in zip(runs['long'], *[runs[m] for m in tested]):
        l, o = rows[0], rows[1]
        t, xc = o['time'], o['x_c_exact']
        print(f"{t:6.3f}{xc:7.3f} | {o['gamma_in_exact']:10.5f}{l['gamma_in']:9.5f}" + ''.join(f"{r['gamma_in']:9.5f}" for r in rows[1:]) + " | "
              f"{o['gamma_core_exact']:9.5f}{l['gamma_core']:8.5f}" + ''.join(f"{r['gamma_core']:8.5f}" for r in rows[1:]) + " | "
              f"{l['l2_interior']:10.4f}" + ''.join(f"{r['l2_interior']:8.4f}" for r in rows[1:]) + " | "
              f"{l['peak_omega'] / l['peak_omega_exact']:9.4f}" + ''.join(f"{r['peak_omega'] / r['peak_omega_exact']:7.4f}" for r in rows[1:]))
        in_transit = xc < X_OUT + 4 * A
        solver['gdev'] = max(solver['gdev'], abs(l['gamma_in'] - l['gamma_in_exact']) / GAMMA)
        if in_transit:
            solver['l2'] = max(solver['l2'], l['l2_interior'])
        if xc < X_OUT - 4 * A:
            solver['wall'] = (l['gamma_in'] - l['gamma_core']) / GAMMA
        for m, r in zip(tested, rows[1:]):
            d = s[m]
            d['dG'] = max(d['dG'], abs(r['gamma_in'] - l['gamma_in']) / GAMMA)
            d['dGex'] = max(d['dGex'], abs(r['gamma_in'] - r['gamma_in_exact']) / GAMMA)
            d['min_w'] = min(d['min_w'], r['min_omega'] / r['peak_omega_exact'])
            if in_transit:
                d['dL2'] = max(d['dL2'], r['l2_interior'] - l['l2_interior'])
            if d['peak_face'] is None and xc >= X_OUT:
                d['peak_face'] = r['peak_omega'] / l['peak_omega']
            if r['gamma_in_exact'] < 1e-3 * GAMMA:
                d['res'] = (r['gamma_in'] - l['gamma_in']) / GAMMA
                d['res_peak'] = r['peak_omega'] / r['peak_omega_exact']
    print(f"\nsolver alone (long box, same nodes): max |G_in - exact|/G = {solver['gdev']:.4f}, "
          f"max interior L2 error vs exact in transit = {solver['l2']:.4f}, "
          f"circulation outside the core box before the face = {solver['wall'] if solver['wall'] is not None else float('nan'):+.4f} G")
    for m in tested:
        d = s[m]
        print(f"{m:>7} box, against the long box on the same nodes:")
        print(f"         max |G_in - G_in(long)| / G                    = {d['dG']:.4f}   (< 0.02 ?)  {'PASS' if d['dG'] < 0.02 else 'FAIL'}")
        print(f"         max interior L2 error added while in transit   = {d['dL2']:+.4f}   (< 0.01 ?)  {'PASS' if d['dL2'] < 0.01 else 'FAIL'}")
        if d['res'] is not None:
            print(f"         circulation left behind after exit             = {d['res']:+.4f} G   (|.| < 0.01 ?)  {'PASS' if abs(d['res']) < 0.01 else 'FAIL'}")
            print(f"         peak vorticity left behind after exit           = {d['res_peak']:.4f} of the initial peak   (< 0.01 ?)  {'PASS' if d['res_peak'] < 0.01 else 'FAIL'}")
        print(f"         peak / long-box peak when the centre reaches the face = {d['peak_face'] if d['peak_face'] is not None else float('nan'):.4f};"
              f" most negative vorticity seen = {d['min_w']:+.4f} of the peak; max |G_in - exact|/G = {d['dGex']:.4f}")
