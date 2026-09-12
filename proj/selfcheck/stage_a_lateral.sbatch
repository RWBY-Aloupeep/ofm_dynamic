#!/bin/bash -l
# The lateral-face tail. With lateral faces convective (cxy) the third cut
# found the split 10-25 m narrower than with lateral walls (cx), while
# moving the walls 600 m out changed nothing; the suspect was the net-flux
# correction being spread over the lateral faces. --outflow cxy-d keeps the
# lateral faces convective but puts the whole correction on the downstream
# face. Six cases at the criterion configuration (1.25 s cycle, 120
# iterations, 1600 s, last 1000 s), cxy and cxy-d; cx exists already
# (stage-a-4-fig6-n5-cg120).
#   0-5   cxy
#   6-11  cxy-d
#
#     sbatch --array=0-11 stage_a_lateral.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --job-name=stage-a-lateral
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v14/selfcheck}
CASES=(
  "shear_z50_q1000   --z0 50  --q0 1000 --mu 4"
  "shear_z50_q500    --z0 50  --q0 500  --mu 4"
  "shear_z100_q1000  --z0 100 --q0 1000 --mu 4"
  "shear_z100_q500   --z0 100 --q0 500  --mu 4"
  "shear_z150_q1000  --z0 150 --q0 1000 --mu 4"
  "shear_z150_q500   --z0 150 --q0 500  --mu 4"
)
i=${SLURM_ARRAY_TASK_ID:-0}
if [ "$i" -le 5 ]; then mode=cxy; c=$i; else mode=cxy-d; c=$((i-6)); fi
OUT="$ROOT/stage-a-4-fig6-$mode-cg120"; mkdir -p "$OUT"
read -r tag args <<< "${CASES[$c]}"
echo "################ $mode / $tag ################"
"$BIN" --test plume --tiles 23 15 19 --dt 0.25 --steps 6400 --diag-every 120 \
       --reinit-every 5 --cg-iter 120 --log-div --theta-advection bfecc-clamp --outflow "$mode" \
       --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
