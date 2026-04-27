if type(add_rules) == "function" then
    add_rules("mode.release", "mode.debug")
end
local common_xmake = "../AMGPCG_Pybind_Torch/common/xmake.lua"
local solver_xmake = "../AMGPCG_Pybind_Torch/solver/xmake.lua"
local engine_xmake = "../engine/xmake.lua"
if os.isfile(common_xmake) then
    includes(common_xmake)
else
    cprint("${yellow}warning: missing %s (did you run submodule init?)", common_xmake)
end
if os.isfile(solver_xmake) then
    includes(solver_xmake)
else
    cprint("${yellow}warning: missing %s (did you run submodule init?)", solver_xmake)
end
if os.isfile(engine_xmake) then
    includes(engine_xmake)
else
    cprint("${yellow}warning: missing %s (did you run submodule init?)", engine_xmake)
end
target("ofm")
    add_rules("plugin.vsxmake.autoupdate")
    if type(set_policy) == "function" then
        set_policy("build.intermediate_directory", false)
    end
    set_kind("static")
    add_headerfiles("*.h")
    add_files("*.cu")
    add_includedirs(".",{public=true})
    add_cugencodes("compute_75")
    add_cuflags("--std c++17", "-lineinfo")
    add_deps("common")
    add_deps("solver")
    add_deps("engine")
