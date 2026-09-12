#!/bin/bash -l
# Centreline x-z sections across the transition: mu = 4 (laminar), 1 and
# 0.15, at the criterion configuration (1.25 s cycle, 120 iterations,
# z0 = 100 m, Q0 = 1 kW/m^3), 900 s, sections every 30 s. plot_xz.py draws
# omega_y with theta contours, to set beside Cunningham's Fig. 4/5.
#
#     sbatch --array=0-2 stage_a_xz.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --job-name=stage-a-xz
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v15/selfcheck}
OUT="$ROOT/stage-a-4-xz"; mkdir -p "$OUT"
MUS=(4 1 0.15)
mu=${MUS[${SLURM_ARRAY_TASK_ID:-0}]}
tag="mu$mu"
echo "################ xz / $tag ################"
"$BIN" --test plume --tiles 23 15 19 --dt 0.25 --steps 3600 --diag-every 120 \
       --reinit-every 5 --cg-iter 120 --theta-advection bfecc-clamp --outflow cx \
       --z0 100 --q0 1000 --mu "$mu" --csv "$OUT/$tag.csv" --slice-xz "$OUT/$tag.slice" 2>&1 | tee "$OUT/$tag.log"
