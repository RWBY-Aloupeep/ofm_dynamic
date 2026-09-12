#!/bin/bash -l
# The convective downstream face fed from the last projected velocity at the
# reinitialization's projection (--face-projected), against the run that blew
# up at 570 s with the face fed from the impulse-reconstructed velocity.
#   0  2.5 s cycle, strong source z0 = 50 m, 120 iterations, 900 s, --face-projected
#   1  criterion configuration (1.25 s, 120 it), z0 = 100 m, Q0 = 1 kW/m^3,
#      1600 s, --face-projected: the regression against the criterion run
#   2  the same as 1 without the flag, rebuilt binary: bit-for-bit control
#
#     sbatch --array=0-2 stage_a_faceproj.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage-a-face
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build-v18/selfcheck}
OUT="$ROOT/stage-a-4-faceproj"; mkdir -p "$OUT"
COMMON="--test plume --tiles 23 15 19 --dt 0.25 --theta-advection bfecc-clamp --outflow cx --mu 4 --cg-iter 120"
case ${SLURM_ARRAY_TASK_ID:-0} in
  0) tag=c2.5-z50-q1000-fp; args="--reinit-every 10 --steps 3600 --diag-every 120 --z0 50 --q0 1000 --log-div --face-projected --slice-xz $OUT/$tag.slice";;
  1) tag=c1.25-z100-q1000-fp; args="--reinit-every 5 --steps 6400 --diag-every 120 --z0 100 --q0 1000 --log-div --face-projected";;
  2) tag=c1.25-z100-q1000-ctl; args="--reinit-every 5 --steps 6400 --diag-every 120 --z0 100 --q0 1000 --log-div";;
esac
echo "################ faceproj / $tag ################"
"$BIN" $COMMON $args --csv "$OUT/$tag.csv" 2>&1 | tee "$OUT/$tag.log"
