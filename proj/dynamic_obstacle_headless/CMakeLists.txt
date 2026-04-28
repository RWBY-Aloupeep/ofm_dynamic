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

set(OFM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../src/ofm")
set(AMGPCG_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../src/AMGPCG_Pybind_Torch")
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
    "${OFM_ROOT}/ofm.cu"
    "${OFM_ROOT}/ofm_init.cu"
    "${OFM_ROOT}/ofm_util.cu"
    ${AMGPCG_COMMON_SOURCES}
    ${AMGPCG_SOLVER_SOURCES})

set_target_properties(dynamic_obstacle_headless PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(dynamic_obstacle_headless PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${OFM_ROOT}"
    "${AMGPCG_COMMON_DIR}"
    "${AMGPCG_SOLVER_DIR}")

target_link_libraries(dynamic_obstacle_headless PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    CUDA::cusolver)
