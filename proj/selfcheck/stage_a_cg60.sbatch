#!/bin/bash -l
# The six Fig. 6 cases with the projection converged (60 AMGPCG iterations
# instead of the fixed 15), at the criterion cycle (n = 1, dt = 0.25) and at
# the 1.25 s cycle (n = 5, dt = 0.25) that collapsed under 15 iterations.
# 1600 s, sampled every 30 s, averaged over the last 1000 s. If the 1.25 s
# cycle now reads the same orderings with its 75% higher peak vorticity,
# the floor lever is back, and the n = 1 rows say how much the unconverged
# projection had moved the criterion numbers themselves.
#
#     sbatch --array=0-11 stage_a_cg60.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --job-name=stage-a-cg60
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v13/selfcheck}
CG=${CG:-60}
CASES=(
  "shear_z50_q1000   --z0 50  --q0 1000 --mu 4"
  "shear_z50_q500    --z0 50  --q0 500  --mu 4"
  "shear_z100_q1000  --z0 100 --q0 1000 --mu 4"
  "shear_z100_q500   --z0 100 --q0 500  --mu 4"
  "shear_z150_q1000  --z0 150 --q0 1000 --mu 4"
  "shear_z150_q500   --z0 150 --q0 500  --mu 4"
)
i=${SLURM_ARRAY_TASK_ID:-0}
if [ "$i" -le 5 ]; then N=1; c=$i; set=fig6-n1-cg$CG; else N=5; c=$((i-6)); set=fig6-n5-cg$CG; fi
OUT="$ROOT/stage-a-4-$set"; mkdir -p "$OUT"
read -r tag args <<< "${CASES[$c]}"
echo "################ $set / $tag ################"
"$BIN" --test plume --tiles 23 15 19 --dt 0.25 --steps 6400 --diag-every 120 \
       --reinit-every "$N" --cg-iter "$CG" --log-div --theta-advection bfecc-clamp --outflow cx \
       --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
