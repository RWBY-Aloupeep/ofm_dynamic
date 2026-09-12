#!/bin/bash -l
# The reconstruction's loss under accumulated strain, measured where the
# answer is known: a Gaussian vortex column (core 0.05 = 6.4 cells, Gamma
# 0.25) in a planar shear u_x = S (y - 1/2), inviscid, 128 x 128 x 16. The
# circulation of the r = 0.3 circle about the core is exactly
# Gamma - S pi r^2 for all time; the map accumulates S n dt of strain per
# cycle. Sweep S and n at dt = 1/384 over T = 4 s. S = 0 is the plain
# dissipation control; the per-cycle strain S n dt runs from 0 to 1.33.
#
#     sbatch shear_sweep.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=shear-sweep
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
BIN=${BIN:-./build-v9/selfcheck}
OUT=${OUT:-/gscratch/amath/diwenxu/wildfire-sim-runs/shear-sweep}; mkdir -p "$OUT"
DT=0.00260416667   # 1/384
STEPS=1536         # 4 s
for S in 0 1 4; do
  for N in 1 4 16 64 128; do
    tag="S${S}_n${N}"
    echo "################ $tag ################"
    "$BIN" --test shear --res-tiles 16 --shear "$S" --core 0.05 --circulation 0.25 --loop-radius 0.3 \
           --dt "$DT" --steps "$STEPS" --diag-every 128 --reinit-every "$N" \
           --csv "$OUT/$tag.csv" 2>&1 | tail -16 | tee "$OUT/$tag.log"
  done
done
