#!/bin/bash -l
# The six Fig. 6 cases at n = 5, dt = 0.125 s: a 0.625 s reinitialisation
# cycle, which the localisation rounds found the weak plume tolerates, at
# 2.5x the cycle duration of n = 1, dt = 0.25 -- so 2.5x lower on the
# numerical floor if the orderings still read the same. 1600 s, sampled
# every 30 s, downstream convective face, Pr = 0.7.
#
#     sbatch --array=0-5 stage_a_n5_dt0125.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage-a-n5-dt0125
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v7/selfcheck}
OUT="$ROOT/stage-a-4-fig6-n5-dt0125"; mkdir -p "$OUT"
CASES=(
  "shear_z50_q1000   --z0 50  --q0 1000 --mu 4"
  "shear_z50_q500    --z0 50  --q0 500  --mu 4"
  "shear_z100_q1000  --z0 100 --q0 1000 --mu 4"
  "shear_z100_q500   --z0 100 --q0 500  --mu 4"
  "shear_z150_q1000  --z0 150 --q0 1000 --mu 4"
  "shear_z150_q500   --z0 150 --q0 500  --mu 4"
)
read -r tag args <<< "${CASES[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ fig6 n = 5, dt = 0.125 / $tag ################"
"$BIN" --test plume --tiles 23 15 19 --dt 0.125 --steps 12800 --diag-every 240 \
       --reinit-every 5 --theta-advection bfecc-clamp --outflow cx \
       --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
