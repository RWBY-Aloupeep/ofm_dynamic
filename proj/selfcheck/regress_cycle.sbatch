#!/bin/bash -l
# Regression for the cycle_len_ refactor of OFM: with fixed cycles the
# solver must reproduce the (n, dt) sweep's floor numbers to the digit.
#   n = 1, dt = 1/480  -> nu_eff = 7.0246e-5   (RESULTS.md, the (n, dt) surface)
#   n = 4, dt = 1/1920 -> nu_eff = 7.0412e-5
# Burgers vortex, nu = 0, 128^3, T = 1 s.
#
#     sbatch regress_cycle.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --job-name=regress-cycle
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
BIN=${BIN:-./build-v8/selfcheck}
OUT=${OUT:-/gscratch/amath/diwenxu/wildfire-sim-runs/stage-a-4-regress}; mkdir -p "$OUT"
echo "=== n = 1, dt = 1/480 (expect 7.0246e-5) ==="
"$BIN" --test burgers --res-tiles 16 --nu 0 --dt 0.00208333333 --steps 480 --diag-every 480 --reinit-every 1 --csv "$OUT/n1.csv" 2>&1 | tail -6
echo "=== n = 4, dt = 1/1920 (expect 7.0412e-5) ==="
"$BIN" --test burgers --res-tiles 16 --nu 0 --dt 0.000520833333 --steps 1920 --diag-every 1920 --reinit-every 4 --csv "$OUT/n4.csv" 2>&1 | tail -6
