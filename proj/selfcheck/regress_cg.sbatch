#!/bin/bash -l
# Do the Stage 0 numbers move when the projection is converged? D1 (viscous
# Burgers, 128^3, n = 1 and n = 5) and D2 (attribution, 128^3, n = 5) at the
# fixed 15 iterations they were recorded with and at 60. The recorded 128^3
# D1 number is +1.69% at n = 1 (RESULTS.md, D1 result table); D2's 128^3
# figures are in the Stage 0 report.
#
#     sbatch regress_cg.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=regress-cg
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
BIN=${BIN:-./build-v13/selfcheck}
OUT=${OUT:-/gscratch/amath/diwenxu/wildfire-sim-runs/stage-a-4-regress-cg}; mkdir -p "$OUT"
for CG in 15 60; do
  for N in 1 5; do
    echo "=== D1 burgers nu = 1e-3, 128^3, n = $N, cg $CG ==="
    "$BIN" --test burgers --res-tiles 16 --nu 1e-3 --dt 0.00208333333 --steps 480 --diag-every 480 \
           --reinit-every "$N" --cg-iter "$CG" --csv "$OUT/d1_n${N}_cg${CG}.csv" 2>&1 | tail -5
  done
  echo "=== D2 attribution, 128^3, n = 5, cg $CG ==="
  "$BIN" --test attribution --res-tiles 16 --nu 1e-3 --dt 0.00208333333 --steps 480 --diag-every 480 \
         --reinit-every 5 --cg-iter "$CG" --csv "$OUT/d2_n5_cg${CG}.csv" 2>&1 | tail -6
done
