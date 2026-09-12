#!/bin/bash -l
# Fourth localisation of the collapse (z0 = 150 m, Q0 = 0.5 kW/m^3). Round
# three showed the loss follows the reinitialisation cycle's duration alone:
# n = 1 at dt = 1.25 s collapses, n = 10 at dt = 0.0625 s holds, and a thicker
# heating layer at the same total heat does not help. What remains that grows
# with the cycle's duration is the impulse-form accumulation of the body force
# itself. --direct-force bypasses it: dt*f is added to the velocity before
# each advance, velocity-form, and the viscous channel goes with it.
#   0  n = 1, dt = 1.25, direct force        if this holds, the path integral is the culprit
#   1  n = 1, dt = 0.25, direct force        control: the direct route at the usual step
#   2  n = 1, dt = 1.25, mu = 0              the impulse route without viscosity, for a like-for-like pair with 0
#
#     sbatch --array=0-2 stage_a_n5_locate4.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=stage-a-locate4
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v7/selfcheck}
OUT="$ROOT/stage-a-4-locate4"; mkdir -p "$OUT"
VARIANTS=(
  "n1-dt1.25-direct  --reinit-every 1 --dt 1.25 --steps 1200 --diag-every 24  --direct-force"
  "n1-dt0.25-direct  --reinit-every 1 --dt 0.25 --steps 6000 --diag-every 120 --direct-force"
  "n1-dt1.25-mu0     --reinit-every 1 --dt 1.25 --steps 1200 --diag-every 24  --mu 0"
)
read -r tag args <<< "${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ locate4 / $tag: z0 150, Q0 500 ################"
"$BIN" --test plume --tiles 23 15 19 --theta-advection bfecc-clamp --outflow cx \
       --z0 150 --q0 500 --mu 4 --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
