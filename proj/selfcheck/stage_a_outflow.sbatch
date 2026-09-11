#!/bin/bash -l
# Stage A, third cut: the six Fig. 6 cases at the paper's 10 m grid under each
# boundary treatment, with the two references that say what "far from the
# boundary" gives, and the theta diffusion on or off. Each array index is one
# set of six; all on rtx6k so the sets are comparable. Sampled every 30 s and
# averaged over the last 200 s by analyse_stage_a.py.
#
#     sbatch --array=0-17 stage_a_outflow.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --job-name=stage-a-outflow
set -u
# Under sbatch $0 is the spool copy of the script; submit from proj/selfcheck.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}

# name | tiles | extra flags
SETS=(
  "closed-nopr    | 23 15 19 | --outflow closed --pr 0"
  "closed-pr      | 23 15 19 | --outflow closed"
  "open-x-pr      | 23 15 19 | --outflow x"
  "open-xy-pr     | 23 15 19 | --outflow xy"
  "open-xy-nopr   | 23 15 19 | --outflow xy --pr 0"
  "long-closed-pr | 46 15 19 | --outflow closed"
  "wide-closed-pr | 23 30 19 | --outflow closed --len-y 2400 --src-y 1200"
  "open-xy-sponge | 23 15 19 | --outflow xy --sponge"
  "open-xy-pr-n5  | 23 15 19 | --outflow xy"
  "conv-x-pr      | 23 15 19 | --outflow cx"
  "conv-xy-pr     | 23 15 19 | --outflow cxy"
  "conv-xy-nopr   | 23 15 19 | --outflow cxy --pr 0"
  "conv-xy-sponge | 23 15 19 | --outflow cxy --sponge"
  "conv-xy-pr-n5  | 23 15 19 | --outflow cxy"
  "wide-closed-l3 | 23 30 19 | --outflow closed --len-y 2400 --src-y 1200 --max-levels 3"
  "closed-pr-l3   | 23 15 19 | --outflow closed --max-levels 3"
  "wide-conv-l3   | 23 30 19 | --outflow cx --len-y 2400 --src-y 1200 --max-levels 3"
  "conv-x-pr-l3   | 23 15 19 | --outflow cx --max-levels 3"
)
IFS='|' read -r name tiles extra <<< "${SETS[${SLURM_ARRAY_TASK_ID:-0}]}"
name=$(echo $name); tiles=$(echo $tiles); extra=$(echo $extra)
N=1; case "$name" in *-n5) N=5;; esac
echo "################ $name: tiles $tiles, n = $N, $extra ################"
# stage_a_sweep.sbatch would read the array index as a case index; hide it.
env -u SLURM_ARRAY_TASK_ID OUT="$ROOT/stage-a-3-$name" TILES="$tiles" DT=0.25 STEPS=2400 DIAG=120 N=$N \
  SET=fig6 EXTRA="$extra" bash stage_a_sweep.sbatch
