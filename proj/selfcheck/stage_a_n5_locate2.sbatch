#!/bin/bash -l
# Second localisation of the n = 5 collapse (z0 = 150 m, Q0 = 0.5 kW/m^3).
# The first round excluded the open face, the viscous source and the drag,
# and showed the loss grows with cycle length. This round asks what about
# the buoyancy source the cycle mishandles. One thing changed per run:
#   0  n = 5, h = 50 m      thicker heating layer (5 cells instead of 2.5)
#   1  n = 5, h = 100 m     10 cells
#   2  n = 1, h = 100 m     control for the h change itself
#   3  n = 5, velocity BFECC clamp off
#   4  n = 5, dt = 0.125 s  same n, half the cycle duration (12000 steps)
#   5  n = 2, dt = 0.625 s  same cycle duration as n = 5 at dt 0.25 (2400 steps)
# 4 and 5 separate "per sub-step" from "per cycle duration": if the loss
# follows n, 4 collapses and 5 does not; if it follows n*dt, the reverse.
#
#     sbatch --array=0-5 stage_a_n5_locate2.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --job-name=stage-a-locate2
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v6/selfcheck}
OUT="$ROOT/stage-a-4-locate2"; mkdir -p "$OUT"
VARIANTS=(
  "n5-h50        --reinit-every 5 --dt 0.25  --steps 6000  --diag-every 120 --h 50"
  "n5-h100       --reinit-every 5 --dt 0.25  --steps 6000  --diag-every 120 --h 100"
  "n1-h100       --reinit-every 1 --dt 0.25  --steps 6000  --diag-every 120 --h 100"
  "n5-noclamp    --reinit-every 5 --dt 0.25  --steps 6000  --diag-every 120 --no-velocity-clamp"
  "n5-dt0.125    --reinit-every 5 --dt 0.125 --steps 12000 --diag-every 240"
  "n2-dt0.625    --reinit-every 2 --dt 0.625 --steps 2400  --diag-every 48"
)
read -r tag args <<< "${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ locate2 / $tag: z0 150, Q0 500 ################"
"$BIN" --test plume --tiles 23 15 19 --theta-advection bfecc-clamp --outflow cx \
       --z0 150 --q0 500 --mu 4 --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
