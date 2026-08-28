#!/usr/bin/env python3
"""Table the (n, dt) sweep written by reinit_dt_sweep.sbatch.

Reads the last diagnostic row of every run -- which the sweep arranges to land
exactly on t = 1 s -- and answers the three questions the sweep was run for:

  1. Does the error collapse onto a function of the cycle length n*dt alone?
     If it does, "reinit and dt" is a one-parameter relation and an optimal n
     follows from dt. If it does not, the two knobs are not interchangeable.
  2. Is there an interior minimum in n, or does the error fall monotonically
     until the run stops being stable?
  3. Does the knee sit at constant n*dt or at constant n*sigma? The gamma set
     doubles the velocity at fixed dt and dx to separate the two.

Standard library only; no plotting.
"""

import csv
import math
import os
import re
import sys

TAG = re.compile(r"^(floor|visclo|visc|gamma2|cost)_dt(\d+)_n(\d+)$")

# The viscosity each viscous set asked the solver for.
NU_OF = {"visc": 1e-3, "visclo": 3e-4}


FIELDS = ("time", "step", "nu_eff", "nu_rel_err", "max_vorticity")


def last_row(path):
    """Last complete diagnostic row of a run, or None if there is none.

    A run still in flight can leave a half-written final line, so rows are only
    accepted once every field this summary reads parses as a number.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in reversed(rows):
        try:
            for key in FIELDS:
                float(row[key])
        except (TypeError, ValueError, KeyError):
            continue
        return row
    return None


def load(out_dir):
    runs = {}
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".csv") or name == "timings.csv":
            continue
        m = TAG.match(name[:-4])
        if not m:
            continue
        kind, inv_dt, n = m.group(1), int(m.group(2)), int(m.group(3))
        row = last_row(os.path.join(out_dir, name))
        if row is None:
            runs[(kind, inv_dt, n)] = None
            continue
        runs[(kind, inv_dt, n)] = {
            "t": float(row["time"]),
            "step": int(row["step"]),
            "nu_eff": float(row["nu_eff"]),
            "rel_err": float(row["nu_rel_err"]),
            "max_w": float(row["max_vorticity"]),
        }
    return runs


def timings(out_dir):
    path = os.path.join(out_dir, "timings.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            m = TAG.match(row["tag"])
            if m:
                out[(m.group(1), int(m.group(2)), int(m.group(3)))] = (
                    float(row["wall_s"]),
                    int(row["rc"]),
                )
    return out


def surface(runs, kind, field, fmt, title, note=""):
    keys = [k for k in runs if k[0] == kind]
    if not keys:
        return
    inv_dts = sorted({k[1] for k in keys})
    ns = sorted({k[2] for k in keys})
    print()
    print(title)
    if note:
        print(note)
    header = f"{'n':>5}" + "".join(f"{'dt=1/' + str(d):>14}" for d in inv_dts)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for n in ns:
        cells = []
        for d in inv_dts:
            r = runs.get((kind, d, n))
            if r is None:
                cells.append(f"{'--':>14}" if (kind, d, n) in runs else f"{'':>14}")
            else:
                cells.append(f"{fmt % r[field]:>14}")
        print(f"{n:>5}" + "".join(cells))
    print("  (-- = the run produced no diagnostic: non-finite field, or it failed to start)")


def loglog_slope(xs, ys):
    """Least-squares slope of log y against log x."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return num / den if den else float("nan")


def exponents(runs, kind):
    """alpha in E ~ N_reinit^alpha, measured separately along each axis.

    N_reinit = T/(n dt). If the per-cycle cost depended only on the cycle, the
    two would agree and the surface would collapse onto n*dt.
    """
    keys = [k for k in runs if k[0] == kind and runs[k]]
    if not keys:
        return
    print()
    print(f"[{kind}] E ~ N_reinit^alpha, fitted along each axis separately")
    print(f"  {'held fixed':>18}  {'points':>6}  {'alpha':>7}")
    for inv_dt in sorted({k[1] for k in keys}):
        sub = sorted((k[2] for k in keys if k[1] == inv_dt))
        xs = [1.0 / (n / float(inv_dt)) for n in sub]  # N_reinit = T/(n dt), T = 1
        ys = [runs[(kind, inv_dt, n)]["nu_eff"] for n in sub]
        print(f"  {'dt = 1/' + str(inv_dt):>18}  {len(sub):>6}  {loglog_slope(xs, ys):>7.3f}   (varying n)")
    for n in sorted({k[2] for k in keys}):
        sub = sorted((k[1] for k in keys if k[2] == n))
        if len(sub) < 2:
            continue
        xs = [1.0 / (n / float(d)) for d in sub]
        ys = [runs[(kind, d, n)]["nu_eff"] for d in sub]
        print(f"  {'n = ' + str(n):>18}  {len(sub):>6}  {loglog_slope(xs, ys):>7.3f}   (varying dt)")
    print("  Equal alphas down the two blocks => the surface collapses onto n*dt.")
    print("  Unequal => n and dt are not interchangeable and there is no one-line rule.")


def collapse(runs, kind):
    """Every point as (cycle length n*dt, error), to see the collapse by eye."""
    keys = sorted((k for k in runs if k[0] == kind and runs[k]), key=lambda k: k[2] / float(k[1]))
    if not keys:
        return
    print()
    print(f"[{kind}] all points ordered by cycle length n*dt")
    print(f"  {'n':>5} {'1/dt':>6} {'n*dt':>10} {'N_reinit':>9} {'nu_eff':>12} {'nu_eff*n*dt':>13}")
    for k in keys:
        n, inv_dt = k[2], k[1]
        cyc = n / float(inv_dt)
        r = runs[k]
        print(f"  {n:>5} {inv_dt:>6} {cyc:>10.5f} {1.0 / cyc:>9.1f} "
              f"{r['nu_eff']:>12.4e} {r['nu_eff'] * cyc:>13.4e}")
    print("  A flat last column means the error is a fixed charge per reinitialization.")


def decompose(runs, kind):
    """Split the viscous error into the two terms that make it change sign.

    The floor adds to the recovered viscosity, so it pushes the error positive
    by floor/nu. Everything left over is the path integral failing to deliver
    the viscosity it was asked for:

        rel_err = floor/nu - deficit      =>   deficit = floor/nu - rel_err

    The floor falls as the cycle lengthens and the deficit grows, so the total
    passes through zero. That zero is a cancellation, not a minimum of either
    term, which is why it has to be checked at more than one nu.
    """
    nu = NU_OF.get(kind)
    keys = sorted((k for k in runs if k[0] == kind and runs[k]),
                  key=lambda k: (k[1], k[2]))
    if not nu or not keys:
        return
    print()
    print(f"[{kind}] nu = {nu:g}: where the error comes from")
    print(f"  {'n':>5} {'1/dt':>6} {'n*dt':>9} {'floor/nu':>10} {'rel_err':>9} {'deficit':>9}")
    for k in keys:
        n, inv_dt = k[2], k[1]
        fl = runs.get(("floor", inv_dt, n))
        if not fl:
            continue
        cyc = n / float(inv_dt)
        floor_frac = fl["nu_eff"] / nu
        rel = runs[k]["rel_err"]
        print(f"  {n:>5} {inv_dt:>6} {cyc:>9.5f} {100 * floor_frac:>9.2f}% "
              f"{100 * rel:>8.2f}% {100 * (floor_frac - rel):>8.2f}%")
    print("  deficit is the part of nu the path integral never delivered; it is")
    print("  the term that grows with cycle length and it does not depend on nu.")


def deficit_by_n(runs, kind):
    """The deficit against n, one column per dt.

    This is the table that separates the two error terms structurally. The floor
    is a function of the cycle duration n*dt and nothing else. If the deficit
    were too, the rows here would slope; if it is a per-sub-step loss, the rows
    are flat and the deficit depends on n alone. Read the rows, not the columns.
    """
    nu = NU_OF.get(kind)
    keys = [k for k in runs if k[0] == kind and runs[k]]
    if not nu or not keys:
        return
    inv_dts = sorted({k[1] for k in keys})
    print()
    print(f"[{kind}] nu = {nu:g}: deficit against n, one column per dt")
    header = f"{'n':>5}" + "".join(f"{'dt=1/' + str(d):>12}" for d in inv_dts) + f"{'spread':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for n in sorted({k[2] for k in keys}):
        vals, cells = [], []
        for d in inv_dts:
            r, fl = runs.get((kind, d, n)), runs.get(("floor", d, n))
            if r and fl:
                v = fl["nu_eff"] / nu - r["rel_err"]
                vals.append(v)
                cells.append(f"{100 * v:>11.2f}%")
            else:
                cells.append(f"{'':>12}")
        spread = (max(vals) / min(vals) - 1.0) if len(vals) > 1 and min(vals) > 0 else float("nan")
        print(f"{n:>5}" + "".join(cells) + f"{100 * spread:>8.0f}%")
    print("  A flat row means the deficit is charged per sub-step of the cycle, not")
    print("  per unit of time -- the opposite of how the dissipation floor is charged.")
    print("  The small-n rows are the unreliable ones: there floor/nu is large, and")
    print("  subtracting a floor measured at nu = 0 from a run at nu > 0 is only")
    print("  approximate because the two fields are not the same field.")


def crossings(runs, kind):
    """Cycle length at which the signed error passes through zero, per dt."""
    nu = NU_OF.get(kind)
    keys = [k for k in runs if k[0] == kind and runs[k]]
    if not nu or not keys:
        return
    print()
    print(f"[{kind}] nu = {nu:g}: zero of the signed error, interpolated in log n")
    print(f"  {'1/dt':>6} {'n*':>8} {'n* dt':>10}")
    for inv_dt in sorted({k[1] for k in keys}):
        ns = sorted(k[2] for k in keys if k[1] == inv_dt)
        hit = None
        for a, b in zip(ns, ns[1:]):
            ea = runs[(kind, inv_dt, a)]["rel_err"]
            eb = runs[(kind, inv_dt, b)]["rel_err"]
            if ea == 0.0:
                hit = float(a)
                break
            if ea * eb < 0.0:
                f = ea / (ea - eb)
                hit = math.exp(math.log(a) + f * (math.log(b) - math.log(a)))
                break
        if hit is None:
            side = "below n=1" if runs[(kind, inv_dt, ns[0])]["rel_err"] < 0 else "above the ladder"
            print(f"  {inv_dt:>6} {side:>8}")
        else:
            print(f"  {inv_dt:>6} {hit:>8.2f} {hit / inv_dt:>10.5f}")
    print("  A constant last column means the best interval is a fixed cycle *duration*,")
    print("  so n* scales as 1/dt. Compare the two nu: if the column moves when nu")
    print("  changes, the crossing is a cancellation and not a property of the scheme.")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = load(out_dir)
    if not runs:
        print(f"no runs found in {out_dir}")
        return 1

    surface(runs, "floor", "nu_eff", "%.4e",
            "nu = 0: the numerical dissipation floor at t = 1 s, as an equivalent viscosity",
            "  (lower is better; this is the error floor every attribution result stands on)")
    surface(runs, "visc", "rel_err", "%+.4f",
            "nu = 1e-3: relative error of the recovered viscosity at t = 1 s",
            "  (this is the one that can see the source-term path integral degrading with cycle length)")
    surface(runs, "visclo", "rel_err", "%+.4f",
            "nu = 3e-4: the same, at a viscosity the floor is a larger fraction of",
            "  (the control on whether the sign change above is an optimum or a cancellation)")
    surface(runs, "gamma2", "nu_eff", "%.4e",
            "nu = 0 at twice the circulation: the same ladder with the velocity doubled",
            "  (compare the knee against the floor table's dt = 1/480 column)")

    exponents(runs, "floor")
    collapse(runs, "floor")
    decompose(runs, "visc")
    deficit_by_n(runs, "visc")
    deficit_by_n(runs, "visclo")
    crossings(runs, "visc")
    crossings(runs, "visclo")

    tm = timings(out_dir)
    cost = sorted((k for k in tm if k[0] == "cost"), key=lambda k: k[2])
    if cost:
        print()
        print("cost: wall clock for 480 steps of simulated time, diagnostics off")
        print(f"  {'n':>5} {'wall_s':>8} {'vs n=1':>8} {'rc':>4}")
        base = tm[cost[0]][0] if cost else 1.0
        for k in cost:
            w, rc = tm[k]
            print(f"  {k[2]:>5} {w:>8.1f} {w / base:>8.3f} {rc:>4}")
        print("  A floor well above zero is the per-step work that raising n cannot remove:")
        print("  ReinitAsync marches O(n) sub-steps per cycle, so only the once-per-cycle")
        print("  Projection 2 / map reset / BFECC term decays as 1/n.")

    # Every set holds T = 1 s, so a last sample short of it means the run is
    # still in flight or stopped early, and its row is not comparable.
    short = [(k, v["t"]) for k, v in runs.items() if v and k[0] != "cost" and v["t"] < 0.999]
    if short:
        print()
        print("runs whose last sample is short of t = 1 (still running, or stopped early):")
        for k, t in sorted(short):
            print(f"  {k[0]}_dt{k[1]}_n{k[2]}  t = {t:.3f}")

    failed = [k for k, v in runs.items() if v is None]
    if failed:
        print()
        print("runs with no diagnostic (non-finite or failed to start):")
        for k in sorted(failed):
            print(f"  {k[0]}_dt{k[1]}_n{k[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
