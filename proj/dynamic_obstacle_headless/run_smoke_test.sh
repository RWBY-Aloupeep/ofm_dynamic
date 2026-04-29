#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
PROJ_DIR="${ROOT_DIR}/proj/dynamic_obstacle_headless"
BUILD_DIR="${PROJ_DIR}/build"
OUTPUT_DIR="${BUILD_DIR}/outputs/headless_smoke"

cmake -S "${PROJ_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j

"${BUILD_DIR}/dynamic_obstacle_headless" \
  --steps 20 \
  --save_interval 1 \
  --resolution 32,32,32 \
  --output_dir "${OUTPUT_DIR}"

python3 "${PROJ_DIR}/check_npy.py" "${OUTPUT_DIR}"
python3 "${PROJ_DIR}/visualize_vorticity.py" "${OUTPUT_DIR}" --axis z --index mid --log --gif
