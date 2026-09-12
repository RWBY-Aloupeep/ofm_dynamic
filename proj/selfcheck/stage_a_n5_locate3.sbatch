#!/bin/bash -l
# Third localisation of the n = 5 collapse (z0 = 150 m, Q0 = 0.5 kW/m^3).
# Round two found the loss follows the cycle duration n*dt (n = 5, dt = 0.125
# holds; n = 2, dt = 0.625 collapses), not the clamp, and that a thicker
# heating layer holds -- but Q = Q0 exp(-z/h) integrates to Q0*h, so h = 100
# was also a 4x stronger source. This round deconfounds that and pins the
# duration rule from both ends.
#   0  n = 5,  h = 100 m, Q0 = 125   same total heat as h = 25, Q0 = 500
#   1  n = 5,  h = 50 m,  Q0 = 250   same total heat
#   2  n = 1,  h = 100 m, Q0 = 125   control for the source change
#   3  n = 10, dt = 0.0625 s         cycle 0.625 s with many sub-steps
#   4  n = 1,  dt = 1.25 s           cycle 1.25 s with one sub-step
#   5  n = 5,  Q0 = 750              where between 500 and 1000 the threshold sits
#
#     sbatch --array=0-5 stage_a_n5_locate3.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --job-name=stage-a-locate3
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v6/selfcheck}
OUT="$ROOT/stage-a-4-locate3"; mkdir -p "$OUT"
VARIANTS=(
  "n5-h100-q125  --reinit-every 5  --dt 0.25   --steps 6000  --diag-every 120 --h 100 --q0 125"
  "n5-h50-q250   --reinit-every 5  --dt 0.25   --steps 6000  --diag-every 120 --h 50  --q0 250"
  "n1-h100-q125  --reinit-every 1  --dt 0.25   --steps 6000  --diag-every 120 --h 100 --q0 125"
  "n10-dt0.0625  --reinit-every 10 --dt 0.0625 --steps 24000 --diag-every 480"
  "n1-dt1.25     --reinit-every 1  --dt 1.25   --steps 1200  --diag-every 24"
  "n5-q750       --reinit-every 5  --dt 0.25   --steps 6000  --diag-every 120 --q0 750"
)
read -r tag args <<< "${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ locate3 / $tag: z0 150 ################"
"$BIN" --test plume --tiles 23 15 19 --theta-advection bfecc-clamp --outflow cx \
       --z0 150 --q0 500 --mu 4 --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
