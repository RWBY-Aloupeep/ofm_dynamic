#!/bin/bash -l
# Is the plume-specific part of the long-cycle loss the bottom wall? The
# candidate: the backward map of the updraught's root cells reaches below
# the ground over a long cycle (w tau ~ 5-10 m at 1.25 s against dx = 10 m),
# is clamped to the wall row, and reconstructs w there from a row where w
# is ~0. Two probes on the weak case (z0 = 150 m, Q0 = 0.5 kW/m^3), 900 s:
#   0  ground source, cycle 0.25 s (n = 1)          holds     + profiles
#   1  ground source, cycle 1.25 s (n = 5)          collapses + profiles
#   2  source elevated to z = 100 m, cycle 0.25 s   control
#   3  source elevated to z = 100 m, cycle 1.25 s   if this holds, the wall is it
# --profile writes horizontal means of w and theta over the source column per
# z level at every diagnostic, so where the updraught goes first is visible.
#
#     sbatch --array=0-3 stage_a_wall.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=stage-a-wall
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v10/selfcheck}
OUT="$ROOT/stage-a-4-wall"; mkdir -p "$OUT"
VARIANTS=(
  "ground-c0.25    --reinit-every 1 --z-src 0"
  "ground-c1.25    --reinit-every 5 --z-src 0"
  "elevated-c0.25  --reinit-every 1 --z-src 100"
  "elevated-c1.25  --reinit-every 5 --z-src 100"
)
read -r tag args <<< "${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ wall / $tag: z0 150, Q0 500 ################"
"$BIN" --test plume --tiles 23 15 19 --dt 0.25 --steps 3600 --diag-every 120 \
       --theta-advection bfecc-clamp --outflow cx --z0 150 --q0 500 --mu 4 \
       --csv "$OUT/$tag.csv" --profile "$OUT/$tag.prof" $args 2>&1 | tee "$OUT/$tag.log"
