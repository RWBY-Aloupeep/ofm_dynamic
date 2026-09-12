#!/bin/bash -l
# Build the self-check harness on a compute node (never on the login node).
# The binary carries cubin for sm_75 and sm_89, so a build on gpu-l40s runs on
# gpu-rtx6k. SELFCHECK_TARGETDIR (default build) is where the binary lands;
# point it elsewhere while jobs are still executing the old one.
#
#     sbatch build.sbatch
#     SELFCHECK_TARGETDIR=build-v4 sbatch --export=ALL build.sbatch
#
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --job-name=selfcheck-build
# set -e only: lmod's init script trips over set -u (LD_PRELOAD unbound).
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# A fresh worktree starts with empty submodules; the build needs both.
[ -e ../../src/AMGPCG_Pybind_Torch/solver/amgpcg.h ] || git submodule update --init --recursive
module load cuda/12.6.3 gcc/11.2.0
export SELFCHECK_TARGETDIR=${SELFCHECK_TARGETDIR:-build}
xmake -P . -y
"./$SELFCHECK_TARGETDIR/selfcheck" --help | head -3 || true
ls -la "$SELFCHECK_TARGETDIR/selfcheck"
