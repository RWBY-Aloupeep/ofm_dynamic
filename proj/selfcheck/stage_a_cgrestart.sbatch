#!/bin/bash -l
# Restarted conjugate gradient (--cg-restart K) against the stall: on the
# plume's pure-Neumann problem the residual norm stops falling after ~35
# iterations and then rises. Weak case, 1.25 s cycle.
#   0  120 iterations, restart 30, 40 steps, residual printed (compare stage-a-4-divstats/cg120-40)
#   1  120 iterations, restart 30, 2400 steps, divergence statistics every 240
#   2   60 iterations, restart 30, 2400 steps
#   3  120 iterations, restart 15, 40 steps, residual printed
#
#     sbatch --array=0-3 stage_a_cgrestart.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=stage-a-cgr
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v19/selfcheck}
OUT="$ROOT/stage-a-4-cgrestart"; mkdir -p "$OUT"
COMMON="--test plume --tiles 23 15 19 --dt 0.25 --theta-advection bfecc-clamp --outflow cx --z0 150 --q0 500 --mu 4 --log-div --reinit-every 5"
VERB="--cg-tol 1e-30 --cg-verbose"
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) tag=cg120-r30-40;   args="--cg-iter 120 --cg-restart 30 --steps 40   --diag-every 40 $VERB";;
  1) tag=cg120-r30-2400; args="--cg-iter 120 --cg-restart 30 --steps 2400 --diag-every 240";;
  2) tag=cg60-r30-2400;  args="--cg-iter 60  --cg-restart 30 --steps 2400 --diag-every 240";;
  3) tag=cg120-r15-40;   args="--cg-iter 120 --cg-restart 15 --steps 40   --diag-every 40 $VERB";;
  # restart 1 is preconditioned steepest descent (p = z every iteration): if
  # it converges monotonically where CG does not, the CG recurrence is what
  # breaks (a preconditioner that is not symmetric), not the multigrid cycle.
  4) tag=cg120-r1-40;    args="--cg-iter 120 --cg-restart 1  --steps 40   --diag-every 40 $VERB";;
  5) tag=cg120-r5-40;    args="--cg-iter 120 --cg-restart 5  --steps 40   --diag-every 40 $VERB";;
esac
echo "################ cgrestart / $tag ################"
"$BIN" $COMMON $args --csv "$OUT/$tag.csv" 2>&1 | tee "$OUT/$tag.log"
