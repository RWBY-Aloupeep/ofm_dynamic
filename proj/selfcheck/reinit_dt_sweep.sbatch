#!/bin/bash -l
# Is there an optimal reinitialization interval, and does it follow dt?
#
# D1 measured two points that both reduce the number of reinitializations per
# unit of simulated time, and they did not agree on what that buys:
#
#   coarsening dt at n = 1   (1/120 -> 1/480, 4x fewer reinits): floor 7.02e-5 -> 1.98e-5
#   raising n at fixed dt    (1 -> 10,        10x fewer reinits): loss 26.1% -> 10.2%
#
# Written as E ~ N_reinit^alpha with N_reinit = T/(n dt), the first gives
# alpha = 0.91 and the second alpha = 0.41. If the per-cycle error depended only
# on the cycle -- its duration, or the distance the map is marched across it --
# the two would give the same exponent and E would collapse onto a function of
# n*dt alone. They do not, so this sweep measures the two-dimensional surface
# instead of assuming the collapse. The two numbers above also come from
# different cases and different metrics, which is the other reason to measure
# one surface with one metric.
#
# The case is D1's columnar Burgers vortex, the only one here with a closed-form
# answer: b(t)^2 = b0^2 + 4 nu t, read back self-normalised as
# nu_eff = b0^2 (w0/w - 1) / (4t). At nu = 0 that is the numerical dissipation
# floor; at nu = 1e-3 it is the recovered viscosity, and its error is the metric.
#
# Four sets:
#   nu0      the floor as a function of (n, dt), at fixed simulated time T = 1 s
#   nuvisc   the same grid at nu = 1e-3, because the inviscid floor cannot see
#            the one error term that grows with cycle length: the source-term
#            path integral is accumulated through F across the cycle, so a longer
#            cycle contaminates it. That term is what could put an interior
#            minimum in n, and set nu0 alone would miss it.
#   gamma    the same n ladder at twice the circulation, hence twice the velocity
#            and twice the strain rate, at one dt. This is the discriminator
#            between two candidate rules that coincide at fixed dx and fixed
#            flow: n*dt = const (constant cycle duration -- which is what LFM's
#            own driver enforces, dt = 1/(frame_rate * reinit_every)) against
#            n*sigma = const (constant per-cycle displacement / map distortion,
#            sigma = u dt/dx). Under the first the ladder's knee stays at the
#            same n*dt when Gamma doubles; under the second it halves.
#   cost     wall clock with diagnostics switched off, to separate the part of
#            the cost that falls with n from the part that does not. ReinitAsync
#            marches O(n) sub-steps per cycle (ofm.cu:233,251), so the marching
#            work per unit simulated time is roughly independent of n; what
#            raising n actually saves is one Projection 2, one map reset and one
#            BFECC pass per cycle, which is a term that decays as 1/n.
#
# All runs hold the simulated time fixed at T = 1 s so that "per unit time" is
# the same axis for every configuration, and every n divides every step count so
# the last diagnostic lands exactly on t = 1 rather than at the first cycle
# boundary past it. (D1 §8's own two floor numbers were sampled at t = 1.01 and
# t = 0.75, which is why the 3.55x there is not a clean ratio.)
#
# In a fresh git worktree the submodules start out empty and the build dies on
# a missing amgpcg.h, so run `git submodule update --init --recursive` (from the
# login node, which is the one with a network) before the first submission.
#
#     sbatch reinit_dt_sweep.sbatch                  # everything, in this order
#     SET=nu0 sbatch reinit_dt_sweep.sbatch          # one set
#     SUMMARY_ONLY=1 bash reinit_dt_sweep.sbatch     # re-table what is on disk
#
# Pin every case of a sweep to one GPU generation; gpu-rtx6k is Turing, which is
# what the build's cubin targets. See stage_a_sweep.sbatch for why this matters.
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=reinit-dt
set -u

module load cuda/12.6.3 gcc/11.2.0

OUT=${OUT:-/gscratch/amath/diwenxu/wildfire-sim-runs/reinit-dt}
BIN=${BIN:-./build/selfcheck}
RES=${RES:-16}                 # tiles per side; 16 = 128^3, D1's floor grid
SET=${SET:-all}                # all | nu0 | nuvisc | gamma | cost
BUILD=${BUILD:-1}
mkdir -p "$OUT"

# T = 1 s in every case. Every n below divides 240, hence divides all four step
# counts, so the final sample is exactly at t = 1.
DTS=(0.00208333333 0.00416666667 0.00104166667 0.000520833333)
STEPS=(480 240 960 1920)
DTNAMES=(480 240 960 1920)     # named by 1/dt
NS=(1 2 4 8 16 48 120 240)     # 240 stores 240 velocity fields: ~6 GB on device

TIMINGS="$OUT/timings.csv"

# run_case <tag> <steps> <dt> <n> <nu> <diag> [extra...]
run_case () {
  tag=$1; steps=$2; dt=$3; n=$4; nu=$5; diag=$6; shift 6
  if [ -s "$OUT/$tag.csv" ] && [ -z "${FORCE:-}" ]; then
    echo "=== $tag (already on disk, skipping) ==="
    return 0
  fi
  echo "=== $tag ==="
  t0=$SECONDS
  "$BIN" --test burgers --res-tiles "$RES" --nu "$nu" --dt "$dt" \
         --steps "$steps" --diag-every "$diag" --reinit-every "$n" \
         "$@" --csv "$OUT/$tag.csv" 2>&1 | tee "$OUT/$tag.log"
  rc=${PIPESTATUS[0]}
  echo "$tag,$n,$dt,$nu,$steps,$((SECONDS - t0)),$rc" >> "$TIMINGS"
  # A configuration that fails is a data point, not a reason to stop the sweep.
  return 0
}

sweep_surface () { # sweep_surface <nu> <prefix>
  nu=$1; prefix=$2
  for i in 0 1 2 3; do
    for n in "${NS[@]}"; do
      # diag-every = n samples once per cycle, which is the only cadence that
      # lands a sample on every cycle boundary including the last one.
      run_case "${prefix}_dt${DTNAMES[$i]}_n${n}" "${STEPS[$i]}" "${DTS[$i]}" \
               "$n" "$nu" "$n"
    done
  done
}

if [ -z "${SUMMARY_ONLY:-}" ] && [ "$BUILD" = 1 ]; then
  echo "=== build ==="
  xmake -P . -y || exit 1
fi

if [ -z "${SUMMARY_ONLY:-}" ]; then
  [ -s "$TIMINGS" ] || echo "tag,n,dt,nu,steps,wall_s,rc" > "$TIMINGS"

  # nu = 0 first, and within it dt = 1/480 first: that row is the ladder that
  # locates the knee, so it is the row worth having if the job is cut short.
  if [ "$SET" = all ] || [ "$SET" = nu0 ];    then sweep_surface 0     floor; fi
  if [ "$SET" = all ] || [ "$SET" = nuvisc ]; then sweep_surface 1e-3  visc;  fi

  # A second viscosity, because the nu = 1e-3 surface changes sign in n and the
  # sign change is the thing to be careful about: the floor makes nu_eff too
  # large and the path-integral deficit makes it too small, so where the total
  # error vanishes is where the two cancel, not where either is small. If that
  # crossing is a real optimum it stays put when nu changes; if it is a
  # cancellation it moves, because floor/nu moves and the deficit does not.
  # 3e-4 is as low as this grid allows while keeping the explicit diffusion
  # number nu*2dt/dx^2 under its ~1/6 limit at the coarsest dt.
  if [ "$SET" = all ] || [ "$SET" = nuvisclo ]; then sweep_surface 3e-4 visclo; fi

  if [ "$SET" = all ] || [ "$SET" = gamma ]; then
    for n in "${NS[@]}"; do
      run_case "gamma2_dt480_n${n}" 480 "${DTS[0]}" "$n" 0 "$n" --circulation 0.10
    done
  fi

  if [ "$SET" = all ] || [ "$SET" = cost ]; then
    # diag-every larger than the run means one measurement at the first cycle
    # boundary and none after, so the wall clock is the solver's and not the
    # diagnostic's.
    for n in "${NS[@]}"; do
      run_case "cost_dt480_n${n}" 480 "${DTS[0]}" "$n" 0 100000
    done
  fi
fi

echo
echo "=== nu_eff at t = 1, by set ==="
# Slurm copies the script into its spool directory, so $0 is not where the
# summary script lives; the submission directory is.
python3 "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/reinit_dt_summary.py" "$OUT"
