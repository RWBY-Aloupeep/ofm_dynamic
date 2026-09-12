#!/bin/bash -l
# Is the AMG preconditioner symmetric positive-definite on the plume's Poisson
# problem (what conjugate gradient assumes)? --test amg-sym compares
# r2 . M r1 with r1 . M r2 for random r, checks positivity, does the same for
# the Laplacian, and prints CG's residual on a random and a smooth right-hand
# side. Seconds of GPU time.
#
#     sbatch stage_a_amgsym.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --job-name=stage-a-sym
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v21/selfcheck}
OUT="$ROOT/stage-a-4-amgsym"; mkdir -p "$OUT"
for cfg in "cx 120" "closed 120"; do
  set -- $cfg
  tag="sym-$1"
  echo "################ amg-sym / $tag ################"
  "$BIN" --test amg-sym --tiles 23 15 19 --outflow $1 --cg-iter $2 --reinit-every 5 2>&1 | tee "$OUT/$tag.log"
done
