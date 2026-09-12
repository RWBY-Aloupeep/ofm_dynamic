#!/bin/bash -l
# Stage A, fourth cut: the same six Fig. 6 cases and the same viscosity four,
# at n = 1 and n = 5, each run to 1600 s and sampled every 30 s so the last
# 1000 s can be averaged. n = 5 dissipates about a third less vorticity than
# n = 1 (RESULTS.md, "n = 5 is markedly less dissipative") and is the default
# from here on; its livelier flow needs the longer window to bring the standard
# error back to what n = 1 had over 200 s. Downstream convective outflow and
# Pr = 0.7 theta diffusion throughout, the configuration the third cut settled on.
#
#     sbatch --array=0-25 stage_a_n5.sbatch
#
# Array index: 0-5 Fig. 6 at n = 1, 6-11 Fig. 6 at n = 5, 12-18 viscosity at
# n = 1, 19-25 viscosity at n = 5. All on rtx6k so the sets are comparable.
#
# The viscosity ladder is extended above the paper's mu = 4 (12, 40) and down
# to mu = 0, because the 10 m sweep's response to mu was linear in mu across
# 0.0015-4 rather than saturating: that is what a sweep entirely below the
# numerical floor looks like, and the bend, if there is one, must be found
# above 4. mu = 0 is the nu = 0 control D1 uses.
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=stage-a-n5
set -u
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load cuda/12.6.3 gcc/11.2.0
ROOT=${ROOT:-/gscratch/amath/diwenxu/wildfire-sim-runs}
BIN=${BIN:-./build/selfcheck}
STEPS=${STEPS:-6400}   # 1600 s at dt = 0.25 s
DIAG=${DIAG:-120}      # every 30 s
DT=0.25
TILES="23 15 19"

CASES=(
  "shear_z50_q1000   --z0 50  --q0 1000 --mu 4"
  "shear_z50_q500    --z0 50  --q0 500  --mu 4"
  "shear_z100_q1000  --z0 100 --q0 1000 --mu 4"
  "shear_z100_q500   --z0 100 --q0 500  --mu 4"
  "shear_z150_q1000  --z0 150 --q0 1000 --mu 4"
  "shear_z150_q500   --z0 150 --q0 500  --mu 4"
  "visc_mu40         --z0 100 --q0 1000 --mu 40"
  "visc_mu12         --z0 100 --q0 1000 --mu 12"
  "visc_mu4          --z0 100 --q0 1000 --mu 4"
  "visc_mu1          --z0 100 --q0 1000 --mu 1"
  "visc_mu0.15       --z0 100 --q0 1000 --mu 0.15"
  "visc_mu0.0015     --z0 100 --q0 1000 --mu 0.0015"
  "visc_mu0          --z0 100 --q0 1000 --mu 0"
)
i=${SLURM_ARRAY_TASK_ID:-0}
if   [ "$i" -le 5 ];  then N=1; c=$i;           set=fig6-n1
elif [ "$i" -le 11 ]; then N=5; c=$((i-6));     set=fig6-n5
elif [ "$i" -le 18 ]; then N=1; c=$((i-12+6));  set=visc-n1
else                       N=5; c=$((i-19+6));  set=visc-n5
fi
OUT="$ROOT/stage-a-4-$set"; mkdir -p "$OUT"
read -r tag args <<< "${CASES[$c]}"
echo "################ $set / $tag: n = $N, ${STEPS} steps, downstream convective, Pr 0.7 ################"
"$BIN" --test plume --tiles $TILES --dt "$DT" --steps "$STEPS" --diag-every "$DIAG" \
       --reinit-every "$N" --theta-advection bfecc-clamp --outflow cx \
       --csv "$OUT/$tag.csv" $args 2>&1 | tee "$OUT/$tag.log"
