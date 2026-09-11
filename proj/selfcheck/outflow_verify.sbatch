#!/bin/bash -l
# Verification of the open (pressure-outlet) boundary on the translating
# Gaussian vortex column: the pressure-outlet box, the convective box, the closed
# control, and the long box the vortex never leaves, each at n = 1 and n = 5. Builds first, on the compute
# node, so the login node never compiles.
#
#     sbatch outflow_verify.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=outflow-verify
# Load first: lmod's init reads LD_PRELOAD, which set -u would trip over.
module load cuda/12.6.3 gcc/11.2.0
set -u
# Under sbatch $0 is the spool copy of the script; submit from proj/selfcheck.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
OUT=${OUT:-/gscratch/amath/diwenxu/wildfire-sim-runs/outflow}
mkdir -p "$OUT"
if [ -z "${NO_BUILD:-}" ]; then
  set -o pipefail
  xmake -P . -y 2>&1 | tail -20 || { echo 'BUILD FAILED'; exit 1; }
fi
BIN=./build/selfcheck
for n in 1 5; do
  # Steps and sampling must be whole cycles: 672 / 16 at n = 1, 680 / 20 at n = 5.
  STEPS=672; DIAG=16; [ $n = 5 ] && STEPS=680 && DIAG=20
  for mode in open conv closed; do
    echo "=== $mode n=$n ==="
    $BIN --test outflow --outflow $mode --reinit-every $n --steps $STEPS --diag-every $DIAG \
         --csv "$OUT/vortex_${mode}_n$n.csv" 2>&1 | tee "$OUT/vortex_${mode}_n$n.log" || exit 1
  done
  echo "=== long n=$n ==="
  $BIN --test outflow --outflow open --long --reinit-every $n --steps $STEPS --diag-every $DIAG \
       --csv "$OUT/vortex_long_n$n.csv" 2>&1 | tee "$OUT/vortex_long_n$n.log" || exit 1
done
