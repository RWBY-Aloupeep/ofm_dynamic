#!/bin/bash -l
# Flexible conjugate gradient (--cg-flexible, Polak-Ribiere beta) against the
# non-symmetric multigrid preconditioner.
#   0  --test amg-sym: standard and flexible CG on a random and a smooth
#      right-hand side, residual per iteration
#   1  weak case, 1.25 s cycle, 120 iterations flexible, 40 steps, residual printed
#   2  same, 60 iterations flexible
#   3  weak case, 1.25 s cycle, 120 flexible, 2400 steps, divergence statistics
#   4  criterion configuration z0 = 100 m, Q0 = 1 kW/m^3, 1600 s, 120 flexible
#   5  criterion configuration, 60 flexible
#
#     sbatch --array=0-5 stage_a_cgflex.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage-a-flex
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v22/selfcheck}
OUT="$ROOT/stage-a-4-cgflex"; mkdir -p "$OUT"
COMMON="--test plume --tiles 23 15 19 --dt 0.25 --theta-advection bfecc-clamp --outflow cx --mu 4 --log-div --reinit-every 5 --cg-flexible"
VERB="--cg-tol 1e-30 --cg-verbose"
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) tag=sym-cx-flex; "$BIN" --test amg-sym --tiles 23 15 19 --outflow cx --cg-iter 120 --reinit-every 5 2>&1 | tee "$OUT/$tag.log"; exit 0;;
  1) tag=cg120-flex-40;   args="--z0 150 --q0 500  --cg-iter 120 --steps 40   --diag-every 40 $VERB";;
  2) tag=cg60-flex-40;    args="--z0 150 --q0 500  --cg-iter 60  --steps 40   --diag-every 40 $VERB";;
  3) tag=cg120-flex-2400; args="--z0 150 --q0 500  --cg-iter 120 --steps 2400 --diag-every 240";;
  4) tag=crit-cg120-flex; args="--z0 100 --q0 1000 --cg-iter 120 --steps 6400 --diag-every 120";;
  5) tag=crit-cg60-flex;  args="--z0 100 --q0 1000 --cg-iter 60  --steps 6400 --diag-every 120";;
esac
echo "################ cgflex / $tag ################"
"$BIN" $COMMON $args --csv "$OUT/$tag.csv" 2>&1 | tee "$OUT/$tag.log"
