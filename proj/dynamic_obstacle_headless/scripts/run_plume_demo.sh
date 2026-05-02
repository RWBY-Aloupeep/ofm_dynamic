#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJ_DIR}/../.." && pwd)"
BUILD_DIR="${PROJ_DIR}/build"

RESOLUTION="64,64,64"
CUDA_ARCH="75"
STEPS=1000
SAVE_INTERVAL=50
PLUME_STRENGTH=0.12
PLUME_RADIUS=0.06
SWIRL_STRENGTH=0.03
OUTPUT_ROOT="${PROJ_DIR}/outputs"

NX="${RESOLUTION%%,*}"
RES_ROOT="${OUTPUT_ROOT}/plume/res${NX}"

mkdir -p "${OUTPUT_ROOT}" "${BUILD_DIR}"

TMP_LOG_DIR="${OUTPUT_ROOT}/_tmp_logs"
mkdir -p "${TMP_LOG_DIR}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
BUILD_TMP_LOG="${TMP_LOG_DIR}/build_${TIMESTAMP}.log"
SIM_TMP_LOG="${TMP_LOG_DIR}/sim_${TIMESTAMP}.log"

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] PROJ_DIR=${PROJ_DIR}"
echo "[INFO] BUILD_DIR=${BUILD_DIR}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] RESOLUTION=${RESOLUTION}"

echo "[INFO] Configuring and building..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cmake -S "${PROJ_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  2>&1 | tee "${BUILD_TMP_LOG}"

cmake --build "${BUILD_DIR}" -j 2>&1 | tee -a "${BUILD_TMP_LOG}"

echo "[INFO] Running simulation..."
"${BUILD_DIR}/dynamic_obstacle_headless" \
  --resolution "${RESOLUTION}" \
  --steps "${STEPS}" \
  --save_interval "${SAVE_INTERVAL}" \
  --plume_strength "${PLUME_STRENGTH}" \
  --plume_radius "${PLUME_RADIUS}" \
  --swirl_strength "${SWIRL_STRENGTH}" \
  --output_dir "${OUTPUT_ROOT}" \
  2>&1 | tee "${SIM_TMP_LOG}"

if [[ ! -d "${RES_ROOT}" ]]; then
  echo "[ERROR] Expected resolution output root not found: ${RES_ROOT}" >&2
  exit 1
fi

RUN_DIR="$(find "${RES_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'ps*_pr*_sw*_*' -printf '%T@\t%p\n' | sort -nr | head -n1 | cut -f2-)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "[ERROR] No run directory found under ${RES_ROOT}" >&2
  exit 1
fi

VORT_DIR="${RUN_DIR}/vorticity"
LOG_DIR="${RUN_DIR}/logs"
PREVIEW_DIR="${RUN_DIR}/preview"
STATS_DIR="${RUN_DIR}/stats"

mkdir -p "${VORT_DIR}" "${LOG_DIR}" "${PREVIEW_DIR}" "${STATS_DIR}"

cp "${BUILD_TMP_LOG}" "${LOG_DIR}/build.log"
cp "${SIM_TMP_LOG}" "${LOG_DIR}/sim.log"

echo "[INFO] Checking npy files in ${VORT_DIR}..."
python3 "${PROJ_DIR}/scripts/check_npy.py" "${VORT_DIR}" \
  | tee "${LOG_DIR}/check.log"

echo "[INFO] Generating previews in ${PREVIEW_DIR}..."
python3 "${PROJ_DIR}/scripts/visualize_vorticity.py" "${VORT_DIR}" \
  --output_dir "${PREVIEW_DIR}" \
  --axis z \
  --index mid \
  --log \
  --crop-boundary 8 \
  --gif \
  --fps 12 \
  --vtk \
  --iso \
  | tee "${LOG_DIR}/viz.log"

echo "[DONE]"
echo "Run dir: ${RUN_DIR}"
echo "Vorticity dir: ${VORT_DIR}"
echo "Logs: ${LOG_DIR}"
echo "Preview: ${PREVIEW_DIR}"
