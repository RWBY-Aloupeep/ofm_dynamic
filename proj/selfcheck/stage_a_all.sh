#!/bin/bash -l
# Stage A, second cut: the whole record in one allocation.
#
# Three sweeps, all on one GPU generation so the orderings are read off
# comparable runs:
#   coarse  -- 96 x 64 x 80, dx = 18.75 m, n = 1. The grid the first cut used,
#              re-run because the theta advection now takes the step's midpoint
#              velocity rather than the end-of-step one.
#   fine    -- 184 x 120 x 152, dx = 10 m, n = 1. The paper's own grid.
#   fine-n5 -- the same, at reinit_every = 5, which the buoyancy and theta
#              rewiring is what makes possible.
module load cuda/12.6.3 gcc/11.2.0
cd "$(dirname "$0")"

ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}

echo "################ coarse grid, dx = 18.75 m, n = 1 ################"
OUT="$ROOT/stage-a-coarse2" TILES="12 8 10" DT=0.5 STEPS=1200 DIAG=600 N=1 \
  bash stage_a_sweep.sbatch

echo
echo "################ fine grid, dx = 10 m, n = 1 ################"
OUT="$ROOT/stage-a-fine" TILES="23 15 19" DT=0.25 STEPS=2400 DIAG=600 N=1 \
  bash stage_a_sweep.sbatch

echo
echo "################ fine grid, dx = 10 m, n = 5 ################"
OUT="$ROOT/stage-a-fine-n5" TILES="23 15 19" DT=0.25 STEPS=2400 DIAG=600 N=5 \
  bash stage_a_sweep.sbatch
