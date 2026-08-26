#!/bin/bash -l
# Stage A sweep: buoyant plume in a sheared cross flow (Cunningham et al. 2005).
#
# Two sets, both from the paper:
#   1. Fig. 6 -- mu = 4 kg/(m s), z0 in {50, 100, 150} m, Q0 in {1000, 500} W/m^3.
#      The result to reproduce is an ordering, not a value: the bifurcation is
#      wider for the deeper shear layer, and wider for the weaker source.
#   2. Figs 4-5 -- z0 = 100 m, Q0 = 1000 W/m^3, mu in {4, 1, 0.15, 0.0015}.
#      The large-mu run should be laminar and symmetric; the pair should survive
#      as mu falls while the plume turns turbulent and asymmetric.
#
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=stage-a
set -u

# A login shell, because "module" is a shell function the profile defines.
# gcc/11.2.0 is what supplies the GLIBCXX_3.4.29 the binary links against;
# the cuda module only puts its own lib64 on the path.
module load cuda/12.6.3 gcc/11.2.0

OUT=${OUT:-/gscratch/amath/diwenxu/wildfire-sim-runs/stage-a}
BIN=${BIN:-./build/selfcheck}
STEPS=${STEPS:-2400}       # 600 s at dt = 0.25 s, the paper's quasi-steady time
DIAG=${DIAG:-240}
DT=${DT:-0.25}
TILES=${TILES:-23 15 19}   # 184 x 120 x 152 cells at dx = 10 m
mkdir -p "$OUT"

run () { # run <tag> <extra args...>
  tag=$1; shift
  echo "=== $tag ==="
  "$BIN" --test plume --tiles $TILES --dt "$DT" --steps "$STEPS" --diag-every "$DIAG" \
         --csv "$OUT/$tag.csv" "$@" 2>&1 | tee "$OUT/$tag.log"
}

for z0 in 50 100 150; do
  for q0 in 1000 500; do
    run "shear_z${z0}_q${q0}" --z0 "$z0" --q0 "$q0" --mu 4
  done
done

for mu in 4 1 0.15 0.0015; do
  run "visc_mu${mu}" --z0 100 --q0 1000 --mu "$mu"
done

echo
echo "last diagnostic row of each run:"
printf '%-24s %10s %10s %12s %12s\n' run 'split@plane' best_x best_omega best_split
for f in "$OUT"/*.csv; do
  # columns: ...,split_width=12,best_x=13,best_omega=14,best_split=15
  tail -1 "$f" | awk -F, -v n="$(basename "$f" .csv)" \
    '{printf "%-24s %10.1f %10.1f %12.4e %12.1f\n", n, $12, $13, $14, $15}'
done
