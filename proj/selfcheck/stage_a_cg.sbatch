#!/bin/bash -l
# Is the long-cycle loss the projection not finishing? The pressure solve
# runs a fixed 15 AMGPCG iterations, not to a tolerance. A longer cycle
# lets the impulse's gauge part grow, the residual divergence after 15
# iterations grows with it, and a residual divergence over the heated
# column is a field of spurious mass sinks -- the growing mean downdraught
# the profiles show. Weak case (z0 = 150 m, Q0 = 0.5 kW/m^3), 900 s, with
# max |div u| after projection reported at every diagnostic.
#   0  cycle 1.25 s (n = 5), 15 iterations   the collapsing baseline
#   1  cycle 1.25 s, 30
#   2  cycle 1.25 s, 60
#   3  cycle 1.25 s, 120
#   4  cycle 0.25 s (n = 1), 15              the holding baseline
#   5  cycle 0.25 s, 60                      does the short cycle change at all?
#
#     sbatch --array=0-5 stage_a_cg.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --job-name=stage-a-cg
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v11/selfcheck}
OUT="$ROOT/stage-a-4-cg"; mkdir -p "$OUT"
VARIANTS=(
  "c1.25-cg15   --reinit-every 5 --cg-iter 15"
  "c1.25-cg30   --reinit-every 5 --cg-iter 30"
  "c1.25-cg60   --reinit-every 5 --cg-iter 60"
  "c1.25-cg120  --reinit-every 5 --cg-iter 120"
  "c0.25-cg15   --reinit-every 1 --cg-iter 15"
  "c0.25-cg60   --reinit-every 1 --cg-iter 60"
)
read -r tag args <<< "${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ cg / $tag: z0 150, Q0 500 ################"
"$BIN" --test plume --tiles 23 15 19 --dt 0.25 --steps 3600 --diag-every 120 \
       --theta-advection bfecc-clamp --outflow cx --z0 150 --q0 500 --mu 4 --log-div \
       --csv "$OUT/$tag.csv" --profile "$OUT/$tag.prof" $args 2>&1 | tee "$OUT/$tag.log"
