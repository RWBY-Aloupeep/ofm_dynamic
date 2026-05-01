#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
PROJ_DIR="${ROOT_DIR}/proj/dynamic_obstacle_headless"
BUILD_DIR="${PROJ_DIR}/build"
OUTPUT_DIR="${PROJ_DIR}/outputs/vorticity/headless_64"

cmake -S "${PROJ_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j

"${BUILD_DIR}/dynamic_obstacle_headless" \
  --steps 100 \
  --save_interval 5 \
  --resolution 64,64,64 \
  --output_dir "${OUTPUT_DIR}"

python3 "${PROJ_DIR}/scripts/check_npy.py" "${OUTPUT_DIR}"
python3 "${PROJ_DIR}/scripts/visualize_vorticity.py" "${OUTPUT_DIR}" --axis z --index mid --log --gif
