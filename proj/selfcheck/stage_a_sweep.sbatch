#!/bin/bash -l
# Stage A sweep: buoyant plume in a sheared cross flow (Cunningham et al. 2005).
#
# Two sets, both from the paper:
#   1. Fig. 6 -- mu = 4 kg/(m s), z0 in {50, 100, 150} m, Q0 in {1000, 500} W/m^3.
#      The result to reproduce is an ordering, not a value: the bifurcation is
#      wider for the deeper shear layer, and wider for the weaker source. Fig. 6
#      reads that off the potential-temperature contours, so the column to look
#      at is theta_split, not the omega_z extrema separation next to it.
#   2. Figs 4-5 -- z0 = 100 m, Q0 = 1000 W/m^3, mu in {4, 1, 0.15, 0.0015}.
#      The large-mu run should be laminar and symmetric; the pair should survive
#      as mu falls while the plume turns turbulent and asymmetric. Below the
#      solver's own numerical dissipation floor this sweep measures nothing, so
#      it only says something once the grid is fine enough to put that floor
#      under the smaller mu values.
#
# Run the whole set in one job:
#     sbatch stage_a_sweep.sbatch
# or one case per job, which is what the 10 m grid needs:
#     sbatch --array=0-5 stage_a_sweep.sbatch     # the Fig. 6 six
#     sbatch --array=6-9 stage_a_sweep.sbatch     # the viscosity four
# then summarise with
#     SUMMARY_ONLY=1 OUT=... bash stage_a_sweep.sbatch
#
# Pin every case of a sweep to one GPU generation. gpu-rtx6k is Turing, and the
# ckpt partition has rtx6k nodes too (--constraint=rtx6k), which is the way in
# when the owned partition is at its GPU limit. An ordering result read off cases
# that ran on different hardware is not one.
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=stage-a
set -u

# A login shell, because "module" is a shell function the profile defines.
# gcc/11.2.0 is what supplies the GLIBCXX_3.4.29 the binary links against;
# the cuda module only puts its own lib64 on the path.
module load cuda/12.6.3 gcc/11.2.0

OUT=${OUT:-/gscratch/amath/diwenxu/wildfire-sim-runs/stage-a}
BIN=${BIN:-./build/selfcheck}
STEPS=${STEPS:-2400}       # 600 s at dt = 0.25 s, the paper's quasi-steady time
DIAG=${DIAG:-240}
DT=${DT:-0.25}
TILES=${TILES:-23 15 19}   # 184 x 120 x 152 cells at dx = 10 m
N=${N:-1}                  # reinitialization cycle length
SET=${SET:-all}            # all | fig6 | visc
EXTRA=${EXTRA:-}           # extra flags for every case, e.g. --theta-advection plain
mkdir -p "$OUT"

# Fixed order, so that a Slurm array index names a case: 0-5 are Fig. 6, 6-9 the
# viscosity series.
CASES=(
  "shear_z50_q1000   --z0 50  --q0 1000 --mu 4"
  "shear_z50_q500    --z0 50  --q0 500  --mu 4"
  "shear_z100_q1000  --z0 100 --q0 1000 --mu 4"
  "shear_z100_q500   --z0 100 --q0 500  --mu 4"
  "shear_z150_q1000  --z0 150 --q0 1000 --mu 4"
  "shear_z150_q500   --z0 150 --q0 500  --mu 4"
  "visc_mu4          --z0 100 --q0 1000 --mu 4"
  "visc_mu1          --z0 100 --q0 1000 --mu 1"
  "visc_mu0.15       --z0 100 --q0 1000 --mu 0.15"
  "visc_mu0.0015     --z0 100 --q0 1000 --mu 0.0015"
)

run () { # run <tag> <extra args...>
  tag=$1; shift
  echo "=== $tag ==="
  "$BIN" --test plume --tiles $TILES --dt "$DT" --steps "$STEPS" --diag-every "$DIAG" \
         --reinit-every "$N" $EXTRA --csv "$OUT/$tag.csv" "$@" 2>&1 | tee "$OUT/$tag.log"
}

summarise () {
  echo
  echo "last diagnostic row of each run in $OUT:"
  printf '%-20s %10s %10s %9s %12s %11s\n' \
         run 'th_split' 'th_width' 'best_x' 'best_th_spl' 'omega_split'
  for f in "$OUT"/*.csv; do
    [ -e "$f" ] || continue
    # columns: split_width=12, best_x=13, best_omega=14, best_split=15,
    #          theta_width=16, theta_split=17, theta_saddle=18, theta_peak=19,
    #          bifurcated=20, best_theta_width=21, best_theta_split=22,
    #          best_bifurcated=23
    tail -1 "$f" | awk -F, -v n="$(basename "$f" .csv)" \
      '{printf "%-20s %7.1f %-2s %10.1f %9.1f %9.1f %-2s %11.1f\n", \
               n, $17, ($20==1 ? "" : "!1"), $16, $13, $22, ($23==1 ? "" : "!1"), $12}'
  done
  echo
  echo "(!1 marks a plane whose theta profile is a single lobe, not a bifurcation)"
}

if [ -n "${SUMMARY_ONLY:-}" ]; then
  summarise
  exit 0
fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  run ${CASES[$SLURM_ARRAY_TASK_ID]}
  exit $?
fi

if [ "$SET" = all ] || [ "$SET" = fig6 ]; then
  for i in 0 1 2 3 4 5; do run ${CASES[$i]}; done
fi
if [ "$SET" = all ] || [ "$SET" = visc ]; then
  for i in 6 7 8 9; do run ${CASES[$i]}; done
fi

summarise
