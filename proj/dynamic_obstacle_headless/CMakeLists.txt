cmake_minimum_required(VERSION 3.24)

project(dynamic_obstacle_headless LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures (e.g. 80;86;90)" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)

FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    spdlog
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.13.0.tar.gz)

FetchContent_MakeAvailable(spdlog)

set(OFM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
set(OFM_SRC_ROOT "${OFM_ROOT}/src")
set(OFM_SRC_DIR "${OFM_SRC_ROOT}/ofm")
set(AMGPCG_ROOT "${OFM_SRC_ROOT}/AMGPCG_Pybind_Torch")
set(AMGPCG_COMMON_DIR "${AMGPCG_ROOT}/common")
set(AMGPCG_SOLVER_DIR "${AMGPCG_ROOT}/solver")

if(NOT EXISTS "${AMGPCG_COMMON_DIR}" OR NOT EXISTS "${AMGPCG_SOLVER_DIR}")
    message(FATAL_ERROR
        "Missing AMGPCG sources at ${AMGPCG_ROOT}.\n"
        "This standalone headless build needs only the AMGPCG common/solver sources (no GUI/render deps).\n"
        "Initialize the submodule before building (if network policy allows):\n"
        "  git submodule update --init src/AMGPCG_Pybind_Torch")
endif()

file(GLOB AMGPCG_COMMON_SOURCES CONFIGURE_DEPENDS
    "${AMGPCG_COMMON_DIR}/*.cu"
    "${AMGPCG_COMMON_DIR}/*.cpp"
    "${AMGPCG_COMMON_DIR}/*.cc"
    "${AMGPCG_COMMON_DIR}/*.cxx")

file(GLOB AMGPCG_SOLVER_SOURCES CONFIGURE_DEPENDS
    "${AMGPCG_SOLVER_DIR}/*.cu"
    "${AMGPCG_SOLVER_DIR}/*.cpp"
    "${AMGPCG_SOLVER_DIR}/*.cc"
    "${AMGPCG_SOLVER_DIR}/*.cxx")

add_executable(dynamic_obstacle_headless
    main.cu
    "${OFM_SRC_DIR}/ofm.cu"
    "${OFM_SRC_DIR}/ofm_init.cu"
    "${OFM_SRC_DIR}/ofm_util.cu"
    ${AMGPCG_COMMON_SOURCES}
    ${AMGPCG_SOLVER_SOURCES})

set_target_properties(dynamic_obstacle_headless PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(dynamic_obstacle_headless PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${OFM_SRC_ROOT}"
    "${OFM_SRC_ROOT}/engine"
    "${OFM_SRC_DIR}"
    "${AMGPCG_ROOT}"
    "${AMGPCG_COMMON_DIR}"
    "${AMGPCG_SOLVER_DIR}")

target_compile_options(dynamic_obstacle_headless PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-include cstdint>
    $<$<COMPILE_LANGUAGE:CUDA>:-include cstdint>)
    
target_link_libraries(dynamic_obstacle_headless PRIVATE
    nlohmann_json::nlohmann_json
    spdlog::spdlog_header_only
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    CUDA::cusolver)
