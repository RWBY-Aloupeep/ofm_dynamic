#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
PROJ_DIR="${ROOT_DIR}/proj/dynamic_obstacle_headless"
BUILD_DIR="${PROJ_DIR}/build"
OUTPUT_DIR="${PROJ_DIR}/outputs/plume/res128"
LOG_DIR="${OUTPUT_DIR}/logs"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

exec > >(tee "${LOG_FILE}") 2>&1

cmake -S "${PROJ_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build "${BUILD_DIR}" -j > "${LOG_DIR}/build.log" 2>&1

echo "[INFO] Running simulation..."
"${BUILD_DIR}/dynamic_obstacle_headless" \
  --resolution 64,64,64 \
  --output_dir "${OUTPUT_DIR}" \
  > "${LOG_DIR}/sim.log" 2>&1

echo "[INFO] Checking npy files..."
python3 "${PROJ_DIR}/scripts/check_npy.py" "${OUTPUT_DIR}" \
  > "${LOG_DIR}/check.log" 2>&1

echo "[INFO] Visualizing..."
python3 "${PROJ_DIR}/scripts/visualize_vorticity.py" "${OUTPUT_DIR}" \
  --axis z \
  --index mid \
  --log \
  --gif \
  --fps 12 \
  --vtk \
  --iso \
  > "${LOG_DIR}/viz.log" 2>&1

echo "[DONE] All logs saved to ${LOG_DIR}"