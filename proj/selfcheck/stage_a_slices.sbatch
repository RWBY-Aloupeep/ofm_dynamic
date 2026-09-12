#!/bin/bash -l
# The six Fig. 6 cases with the x = 1750 m theta section written at every
# diagnostic, so plot_fig6.py can draw the paper's figure from the run.
# Default configuration of the fourth cut: n = 5, downstream convective
# outflow, Pr = 0.7, 1600 s, sampled every 30 s.
#
#     sbatch --array=0-5 stage_a_slices.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage-a-slices
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v4/selfcheck}
N=${N:-5}
STEPS=${STEPS:-6400}
DIAG=${DIAG:-120}
OUT="$ROOT/stage-a-4-slices-n$N"; mkdir -p "$OUT"
CASES=(
  "shear_z50_q1000   --z0 50  --q0 1000 --mu 4"
  "shear_z50_q500    --z0 50  --q0 500  --mu 4"
  "shear_z100_q1000  --z0 100 --q0 1000 --mu 4"
  "shear_z100_q500   --z0 100 --q0 500  --mu 4"
  "shear_z150_q1000  --z0 150 --q0 1000 --mu 4"
  "shear_z150_q500   --z0 150 --q0 500  --mu 4"
)
read -r tag args <<< "${CASES[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ slices / $tag: n = $N ################"
"$BIN" --test plume --tiles 23 15 19 --dt 0.25 --steps "$STEPS" --diag-every "$DIAG" \
       --reinit-every "$N" --theta-advection bfecc-clamp --outflow cx \
       --csv "$OUT/$tag.csv" --slice "$OUT/$tag.slice" $args 2>&1 | tee "$OUT/$tag.log"
