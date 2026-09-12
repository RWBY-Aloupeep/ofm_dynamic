#!/bin/bash -l
# Where the residual divergence lives: the solver's per-iteration residual
# norm (--cg-verbose, first 40 steps) against the projected field's
# divergence statistics (--log-div: max and its cell, rms, interior rms,
# l2, boundary-layer share). Weak case, 1.25 s cycle.
#   0  15 iterations,  40 steps  (stats on the last logged projection)
#   1  120 iterations, 40 steps
#   2  15 iterations,  2400 steps, stats every 240
#   3  120 iterations, 2400 steps, stats every 240
#   4  15 iterations, n = 1, 40 steps (no reinitialisation projection)
#
#     sbatch --array=0-4 stage_a_divstats.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=stage-a-div
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v17/selfcheck}
OUT="$ROOT/stage-a-4-divstats"; mkdir -p "$OUT"
COMMON="--test plume --tiles 23 15 19 --dt 0.25 --theta-advection bfecc-clamp --outflow cx --z0 150 --q0 500 --mu 4 --log-div"
# The per-iteration residual is only printed in tolerance mode; a tolerance
# of 1e-30 is never met (the floor is abs_tol_ = 1e-14 on rTr), so the count
# is the fixed one and the residual is printed.
VERB="--cg-tol 1e-30 --cg-verbose"
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) tag=cg15-40;    args="--reinit-every 5 --cg-iter 15  --steps 40   --diag-every 40  $VERB";;
  1) tag=cg120-40;   args="--reinit-every 5 --cg-iter 120 --steps 40   --diag-every 40  $VERB";;
  2) tag=cg15-2400;  args="--reinit-every 5 --cg-iter 15  --steps 2400 --diag-every 240";;
  3) tag=cg120-2400; args="--reinit-every 5 --cg-iter 120 --steps 2400 --diag-every 240";;
  4) tag=n1-cg15-40; args="--reinit-every 1 --cg-iter 15  --steps 40   --diag-every 40  $VERB";;
esac
echo "################ divstats / $tag ################"
"$BIN" $COMMON $args --csv "$OUT/$tag.csv" 2>&1 | tee "$OUT/$tag.log"
