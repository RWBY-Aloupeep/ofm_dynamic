#!/bin/bash -l
# With the projection converged (60 iterations), how long can the cycle
# get before the weak plume degrades? Weak case (z0 = 150 m, Q0 = 0.5),
# dt = 0.25 s, 1500 s, max |div u| reported.
#   0  n = 10  cycle 2.5 s
#   1  n = 20  cycle 5 s
#   2  n = 40  cycle 10 s
#   3  n = 10  cycle 2.5 s, 120 iterations (is 60 still enough at 2.5 s?)
#
#     sbatch --array=0-3 stage_a_cycle_cg60.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage-a-cycle
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v11/selfcheck}
OUT="$ROOT/stage-a-4-cycle-cg60"; mkdir -p "$OUT"
VARIANTS=(
  "c2.5-cg60    --reinit-every 10 --cg-iter 60  --diag-every 120"
  "c5-cg60      --reinit-every 20 --cg-iter 60  --diag-every 120"
  "c10-cg60     --reinit-every 40 --cg-iter 60  --diag-every 120"
  "c2.5-cg120   --reinit-every 10 --cg-iter 120 --diag-every 120"
)
read -r tag args <<< "${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ cycle / $tag: z0 150, Q0 500 ################"
"$BIN" --test plume --tiles 23 15 19 --dt 0.25 --steps 6000 --log-div \
       --theta-advection bfecc-clamp --outflow cx --z0 150 --q0 500 --mu 4 \
       --csv "$OUT/$tag.csv" --profile "$OUT/$tag.prof" $args 2>&1 | tee "$OUT/$tag.log"
