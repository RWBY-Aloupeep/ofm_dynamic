#!/bin/bash -l
# Localise the n = 5 collapse of the weak-source plume. At n = 5 the z0 = 150 m,
# Q0 = 0.5 kW/m^3 case loses its plume after about 450 s (top 1275 m -> 95 m,
# w_max 7 -> 1.6 m/s with a 9 K anomaly still present), while n = 1 holds
# steady at 635 m. The same case, one thing changed at a time:
#   0  n = 2
#   1  n = 3
#   2  n = 5, drag off (--cd-a 0)
#   3  n = 5, mu = 0 (no viscous source in the cycle)
#   4  n = 5, drag off and mu = 0
#   5  n = 5, closed box (is it the open face?)
#
#     sbatch --array=0-5 stage_a_n5_locate.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=stage-a-locate
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v5/selfcheck}
OUT="$ROOT/stage-a-4-locate"; mkdir -p "$OUT"
VARIANTS=(
  "n2            --reinit-every 2 --outflow cx"
  "n3            --reinit-every 3 --outflow cx"
  "n5-nodrag     --reinit-every 5 --outflow cx --cd-a 0"
  "n5-mu0        --reinit-every 5 --outflow cx --mu 0"
  "n5-nodrag-mu0 --reinit-every 5 --outflow cx --cd-a 0 --mu 0"
  "n5-closed     --reinit-every 5 --outflow closed"
)
read -r tag args <<< "${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "################ locate / $tag: z0 150, Q0 500 ################"
"$BIN" --test plume --tiles 23 15 19 --dt 0.25 --steps 6000 --diag-every 120 \
       --theta-advection bfecc-clamp --z0 150 --q0 500 --mu 4 \
       --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
