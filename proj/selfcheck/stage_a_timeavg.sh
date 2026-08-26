#!/bin/bash -l
# The measurement the orderings actually need: the six Fig. 6 cases at the
# paper's grid, sampled every 30 s instead of every 150 s, so the width can be
# time-averaged over the last few hundred seconds rather than read off one
# instant of an unsteady plume.
#
# Run twice, differing only in how theta is advected, which makes the pair a
# controlled test of the scheme:
#   plain  -- the bare semi-Lagrangian step the first cut used
#   bfecc  -- the same BFECC-plus-clamp the solver applies to the velocity
module load cuda/12.6.3 gcc/11.2.0
cd "$(dirname "$0")"

ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}

echo "################ fine, dense sampling, theta plain semi-Lagrangian ################"
OUT="$ROOT/stage-a-dense-plain" TILES="23 15 19" DT=0.25 STEPS=2400 DIAG=120 N=1 \
  SET=fig6 EXTRA="--theta-advection plain" bash stage_a_sweep.sbatch

echo
echo "################ fine, dense sampling, theta BFECC + clamp ################"
OUT="$ROOT/stage-a-dense-bfecc" TILES="23 15 19" DT=0.25 STEPS=2400 DIAG=120 N=1 \
  SET=fig6 EXTRA="--theta-advection bfecc-clamp" bash stage_a_sweep.sbatch
