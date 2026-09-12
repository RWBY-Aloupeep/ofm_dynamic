#!/bin/bash -l
# The projection solved to a residual tolerance instead of a fixed count:
# --cg-tol REL with --cg-iter as the cap. On the weak case (z0 = 150 m,
# Q0 = 0.5), 900 s, max |div u| reported, to see what tolerance a converged
# plume needs and whether the short cycle then pays fewer iterations.
#   0  cycle 1.25 s, tol 1e-2, cap 200
#   1  cycle 1.25 s, tol 1e-3, cap 200
#   2  cycle 1.25 s, tol 1e-4, cap 400
#   3  cycle 0.25 s, tol 1e-3, cap 200
#
#     sbatch --array=0-3 stage_a_cgtol.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --job-name=stage-a-cgtol
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v12/selfcheck}
OUT="$ROOT/stage-a-4-cgtol"; mkdir -p "$OUT"
VARIANTS=(
  "c1.25-tol1e-2  --reinit-every 5 --cg-tol 1e-2 --cg-iter 200"
  "c1.25-tol1e-3  --reinit-every 5 --cg-tol 1e-3 --cg-iter 200"
  "c1.25-tol1e-4  --reinit-every 5 --cg-tol 1e-4 --cg-iter 400"
  "c0.25-tol1e-3  --reinit-every 1 --cg-tol 1e-3 --cg-iter 200"
)
read -r tag args <<< "${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ cgtol / $tag: z0 150, Q0 500 ################"
/usr/bin/time -f "wall %e s" "$BIN" --test plume --tiles 23 15 19 --dt 0.25 --steps 3600 --diag-every 120 \
       --theta-advection bfecc-clamp --outflow cx --z0 150 --q0 500 --mu 4 --log-div \
       --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
