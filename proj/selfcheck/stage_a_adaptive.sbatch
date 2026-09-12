#!/bin/bash -l
# Adaptive reinitialization, first pass, on the collapsing case
# (z0 = 150 m, Q0 = 0.5 kW/m^3), 1500 s.
#
# Calibration first: fixed cycles with --log-strain, which reports per
# diagnostic the largest per-cycle strain bound sum(max|grad u|) dt. The
# threshold eps has to sit between what the holding runs reach and what the
# collapsing ones reach.
#   0  n = 1,  dt = 0.25   (holds)          log strain
#   1  n = 5,  dt = 0.125  (holds, drifts)  log strain
#   2  n = 5,  dt = 0.25   (collapses)      log strain
# Then the controller itself, at dt = 0.25 with the history buffer sized for
# ten steps, at three guesses of eps:
#   3  --adaptive-reinit 0.05
#   4  --adaptive-reinit 0.10
#   5  --adaptive-reinit 0.20
#
#     sbatch --array=0-5 stage_a_adaptive.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --job-name=stage-a-adaptive
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v8/selfcheck}
OUT="$ROOT/stage-a-4-adaptive"; mkdir -p "$OUT"
VARIANTS=(
  "log-n1-dt0.25    --reinit-every 1  --dt 0.25  --steps 6000  --diag-every 120 --log-strain"
  "log-n5-dt0.125   --reinit-every 5  --dt 0.125 --steps 12000 --diag-every 240 --log-strain"
  "log-n5-dt0.25    --reinit-every 5  --dt 0.25  --steps 6000  --diag-every 120 --log-strain"
  "adapt-eps0.05    --reinit-every 10 --dt 0.25  --steps 6000  --diag-every 120 --adaptive-reinit 0.05"
  "adapt-eps0.10    --reinit-every 10 --dt 0.25  --steps 6000  --diag-every 120 --adaptive-reinit 0.10"
  "adapt-eps0.20    --reinit-every 10 --dt 0.25  --steps 6000  --diag-every 120 --adaptive-reinit 0.20"
)
read -r tag args <<< "${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ adaptive / $tag: z0 150, Q0 500 ################"
"$BIN" --test plume --tiles 23 15 19 --theta-advection bfecc-clamp --outflow cx \
       --z0 150 --q0 500 --mu 4 --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
