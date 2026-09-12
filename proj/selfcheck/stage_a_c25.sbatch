#!/bin/bash -l
# The six Fig. 6 cases at the 2.5 s cycle (n = 10, dt = 0.25) with 120
# iterations (residual 0.055 on the weak case), and the viscosity ladder at
# the criterion configuration (1.25 s cycle, 120 iterations). 1600 s,
# sampled every 30 s, last 1000 s averaged.
#   0-5   Fig. 6 six at 2.5 s / 120
#   6-12  mu ladder (40, 12, 4, 1, 0.15, 0.0015, 0) at 1.25 s / 120
#
#     sbatch --array=0-12 stage_a_c25.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --job-name=stage-a-c25
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v13/selfcheck}
CASES=(
  "shear_z50_q1000   --z0 50  --q0 1000 --mu 4      --reinit-every 10"
  "shear_z50_q500    --z0 50  --q0 500  --mu 4      --reinit-every 10"
  "shear_z100_q1000  --z0 100 --q0 1000 --mu 4      --reinit-every 10"
  "shear_z100_q500   --z0 100 --q0 500  --mu 4      --reinit-every 10"
  "shear_z150_q1000  --z0 150 --q0 1000 --mu 4      --reinit-every 10"
  "shear_z150_q500   --z0 150 --q0 500  --mu 4      --reinit-every 10"
  "visc_mu40         --z0 100 --q0 1000 --mu 40     --reinit-every 5"
  "visc_mu12         --z0 100 --q0 1000 --mu 12     --reinit-every 5"
  "visc_mu4          --z0 100 --q0 1000 --mu 4      --reinit-every 5"
  "visc_mu1          --z0 100 --q0 1000 --mu 1      --reinit-every 5"
  "visc_mu0.15       --z0 100 --q0 1000 --mu 0.15   --reinit-every 5"
  "visc_mu0.0015     --z0 100 --q0 1000 --mu 0.0015 --reinit-every 5"
  "visc_mu0          --z0 100 --q0 1000 --mu 0      --reinit-every 5"
)
i=${SLURM_ARRAY_TASK_ID:-0}
if [ "$i" -le 5 ]; then set=fig6-c2.5-cg120; else set=visc-c1.25-cg120; fi
OUT="$ROOT/stage-a-4-$set"; mkdir -p "$OUT"
read -r tag args <<< "${CASES[$i]}"
echo "################ $set / $tag ################"
"$BIN" --test plume --tiles 23 15 19 --dt 0.25 --steps 6400 --diag-every 120 \
       --cg-iter 120 --log-div --theta-advection bfecc-clamp --outflow cx \
       --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
