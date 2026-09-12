#!/bin/bash -l
# The remaining items of the Stage A report's next-steps list, criterion
# configuration (1.25 s cycle, 120 iterations unless stated).
#   0  St shedding: mu = 1,    3000 s, wake probes every step (--probes)
#   1  St shedding: mu = 0.15, 3000 s, same
#   2  --cg-tol diagnostic: 60 steps with the per-iteration residual printed
#      for the first 40 projections (why the tolerance mode runs to its cap)
#   3  2.5 s cycle, strong source z0 = 50 that blew up at 570 s: 200
#      iterations, residual, profile and the centreline section every 30 s,
#      to see whether it is the residual or the flow
#   4  same as 3 with 120 iterations (the run that blew up), diagnosed
#
#     sbatch --array=0-4 stage_a_sec8.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage-a-sec8
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v16/selfcheck}
OUT="$ROOT/stage-a-4-sec8"; mkdir -p "$OUT"
COMMON="--test plume --tiles 23 15 19 --dt 0.25 --theta-advection bfecc-clamp --outflow cx"
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) tag=st-mu1;   args="--reinit-every 5 --cg-iter 120 --steps 12000 --diag-every 120 --z0 100 --q0 1000 --mu 1    --probes $OUT/$tag.probes";;
  1) tag=st-mu0.15; args="--reinit-every 5 --cg-iter 120 --steps 12000 --diag-every 120 --z0 100 --q0 1000 --mu 0.15 --probes $OUT/$tag.probes";;
  2) tag=cgtol-diag; args="--reinit-every 5 --cg-iter 200 --cg-tol 1e-3 --cg-verbose --steps 60 --diag-every 60 --z0 150 --q0 500 --mu 4";;
  3) tag=c2.5-z50-q1000-cg200; args="--reinit-every 10 --cg-iter 200 --steps 3600 --diag-every 120 --z0 50 --q0 1000 --mu 4 --log-div --profile $OUT/$tag.prof --slice-xz $OUT/$tag.slice";;
  4) tag=c2.5-z50-q1000-cg120; args="--reinit-every 10 --cg-iter 120 --steps 3600 --diag-every 120 --z0 50 --q0 1000 --mu 4 --log-div --profile $OUT/$tag.prof --slice-xz $OUT/$tag.slice";;
esac
echo "################ sec8 / $tag ################"
"$BIN" $COMMON $args --csv "$OUT/$tag.csv" 2>&1 | tee "$OUT/$tag.log"
