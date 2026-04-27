if type(add_rules) == "function" then
    add_rules("mode.release", "mode.debug")
end

local function _enabled(v)
    return v == true or v == "y" or v == "yes" or v == "true" or v == "1" or v == 1
end

local with_gui = false
local headless = false
if type(get_config) == "function" then
    with_gui = _enabled(get_config("with_gui"))
    headless = _enabled(get_config("headless"))
elseif type(has_config) == "function" then
    with_gui = has_config("with_gui")
    headless = has_config("headless")
end

local need_engine = with_gui and (not headless)
local use_embedded_amgpcg = headless

local common_xmake = "../AMGPCG_Pybind_Torch/common/xmake.lua"
local solver_xmake = "../AMGPCG_Pybind_Torch/solver/xmake.lua"

if not use_embedded_amgpcg then
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
end

if need_engine then
    local engine_xmake = "../engine/xmake.lua"
    if os.isfile(engine_xmake) then
        includes(engine_xmake)
    else
        cprint("${yellow}warning: missing %s (did you run submodule init?)", engine_xmake)
    end
end

local function _add_existing_files(pattern)
    local files = os.files(pattern)
    if files and #files > 0 then
        add_files(table.unpack(files))
    end
end

target("ofm")
    add_rules("plugin.vsxmake.autoupdate")
    if type(set_policy) == "function" then
        set_policy("build.intermediate_directory", false)
    end
    set_kind("static")
    add_headerfiles("*.h")
    add_files("*.cu")
    add_includedirs(".", {public=true})
    add_cugencodes("compute_75")
    add_cuflags("--std c++17", "-lineinfo")
    
    if use_embedded_amgpcg then
        local amgpcg_root = "../AMGPCG_Pybind_Torch"
        local common_dir = amgpcg_root .. "/common"
        local solver_dir = amgpcg_root .. "/solver"

        add_includedirs(common_dir, solver_dir, { public = true })

        _add_existing_files(common_dir .. "/*.cu")
        _add_existing_files(common_dir .. "/*.cpp")
        _add_existing_files(common_dir .. "/*.cc")
        _add_existing_files(common_dir .. "/*.cxx")

        _add_existing_files(solver_dir .. "/*.cu")
        _add_existing_files(solver_dir .. "/*.cpp")
        _add_existing_files(solver_dir .. "/*.cc")
        _add_existing_files(solver_dir .. "/*.cxx")
    else
        add_deps("common")
        add_deps("solver")
    end

    if need_engine then
        add_deps("engine")
    end
