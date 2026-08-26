-- Standalone, headless build for the solver self-check harness.
--
-- This is deliberately NOT wired into proj/xmake.lua. The applications there link
-- the render engine (Vulkan + GLFW + Dear ImGui + VTK) and open a window, which is
-- the right thing for interactive use but cannot run on a headless compute node.
-- Building from this directory pulls in only CUDA and the numerical sources, so the
-- GUI build path is left exactly as it was.
--
--   module load cuda/12.6.3
--   cd proj/selfcheck && xmake && ./build/selfcheck --help

add_rules("mode.release", "mode.debug")
add_requires("cuda", { system = true })

target("selfcheck")
    set_kind("binary")
    set_targetdir("build")
    set_languages("cxx17")
    set_policy("build.intermediate_directory", false)

    -- compat/ must precede any engine path so that "core/tool/logger.h" resolves to
    -- the headless shim. The engine include directory is never added at all here.
    add_includedirs("compat")
    add_includedirs(".")
    add_includedirs("../../src/AMGPCG_Pybind_Torch/common")
    add_includedirs("../../src/AMGPCG_Pybind_Torch/solver")
    add_includedirs("../../src/ofm")

    -- Numerical stack only. src/ofm/ofm_init.cu is intentionally excluded: it is the
    -- one solver file that depends on the engine's JSON Configuration type, and the
    -- harness configures the solver programmatically instead.
    add_files("../../src/AMGPCG_Pybind_Torch/common/*.cc")
    add_files("../../src/AMGPCG_Pybind_Torch/common/*.cu")
    add_files("../../src/AMGPCG_Pybind_Torch/solver/*.cu")
    add_files("../../src/ofm/ofm.cu")
    add_files("../../src/ofm/ofm_util.cu")
    add_files("*.cu")

    -- gpu-rtx6k is Turing (sm_75); gpu-l40s and gpu-l40 are Ada (sm_89). The
    -- compute_75 PTX is a JIT fallback so that any device from Turing on -- the
    -- ckpt partition mixes several -- still runs instead of failing every launch.
    -- PTX only JITs forwards, so this covers sm_75 and newer and nothing older;
    -- CheckDeviceUsable catches the rest. Pin a sweep to one partition even so:
    -- an ordering result should not be read off cases that ran on different
    -- hardware.
    add_cugencodes("sm_75", "sm_89", "compute_75")
    add_cuflags("--std c++17", "-lineinfo", "--extended-lambda")
    add_packages("cuda")

    -- AMGPCG_Pybind_Torch/common/mem.cc names uint8_t without including <cstdint>.
    -- That compiles on the toolchain upstream uses but not with GCC here. Force the
    -- include from the build rather than patching the submodule, which is a separate
    -- repository this project only consumes.
    add_cxxflags("-include", "cstdint", { force = true })

    if is_mode("debug") then
        add_defines("DEBUG")
    end
    if is_mode("release") then
        add_defines("NDEBUG")
    end
