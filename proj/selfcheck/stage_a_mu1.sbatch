#!/bin/bash -l
# The unsteady plume below mu = 4, looked at directly. Criterion configuration
# (1.25 s cycle, 120 iterations, z0 = 100 m, Q0 = 1 kW/m^3), 1600 s.
#   0  mu = 1, with --slice: the x = 1750 m sections, to measure the
#      left/right asymmetry the paper describes and we have not measured
#   1  mu = 1, with --sponge: does the unsteadiness survive damping under the
#      lid, or is it the lid (the plume top reaches 1341-1502 m of 1520)
#   2  mu = 0.0015 rerun (the ladder's failed member)
#
#     sbatch --array=0-2 stage_a_mu1.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage-a-mu1
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v14/selfcheck}
OUT="$ROOT/stage-a-4-mu1"; mkdir -p "$OUT"
VARIANTS=(
  "mu1-slice   --mu 1      --slice $OUT/mu1-slice.slice"
  "mu1-sponge  --mu 1      --sponge"
  "mu0.0015    --mu 0.0015"
)
read -r tag args <<< "${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ mu1 / $tag ################"
"$BIN" --test plume --tiles 23 15 19 --dt 0.25 --steps 6400 --diag-every 120 \
       --reinit-every 5 --cg-iter 120 --log-div --theta-advection bfecc-clamp --outflow cx \
       --z0 100 --q0 1000 --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
