#!/bin/bash -l
# The coarsest multigrid level (4 levels on the plume grid: ~9000 cells) is
# "solved" by bottom_smoothing_ = 10 Gauss-Seidel sweeps. --amg-bottom N raises
# that. With flexible CG:
#   0  --test amg-sym, bottom 10 (reference), 100, 400: residual per iteration
#      on the random and the smooth right-hand side
#   1  weak case, 1.25 s cycle, 120 flexible, bottom 400, 40 steps, residual printed
#   2  weak case, 1.25 s cycle, 120 flexible, bottom 400, 2400 steps, statistics
#   3  criterion configuration z0 = 100, Q0 = 1000, 1600 s, 120 flexible, bottom 400
#
#     sbatch --array=0-3 stage_a_amgbottom.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage-a-bottom
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v23/selfcheck}
OUT="$ROOT/stage-a-4-amgbottom"; mkdir -p "$OUT"
COMMON="--test plume --tiles 23 15 19 --dt 0.25 --theta-advection bfecc-clamp --outflow cx --mu 4 --log-div --reinit-every 5 --cg-flexible --amg-bottom 400"
VERB="--cg-tol 1e-30 --cg-verbose"
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) for nb in 10 100 400; do
       tag="sym-cx-bottom$nb"
       echo "################ amg-sym / $tag ################"
       "$BIN" --test amg-sym --tiles 23 15 19 --outflow cx --cg-iter 120 --reinit-every 5 --amg-bottom $nb 2>&1 | tee "$OUT/$tag.log"
     done; exit 0;;
  1) tag=cg120-flex-b400-40;   args="--z0 150 --q0 500  --cg-iter 120 --steps 40   --diag-every 40 $VERB";;
  2) tag=cg120-flex-b400-2400; args="--z0 150 --q0 500  --cg-iter 120 --steps 2400 --diag-every 240";;
  3) tag=crit-cg120-flex-b400; args="--z0 100 --q0 1000 --cg-iter 120 --steps 6400 --diag-every 120";;
esac
echo "################ amgbottom / $tag ################"
"$BIN" $COMMON $args --csv "$OUT/$tag.csv" 2>&1 | tee "$OUT/$tag.log"
