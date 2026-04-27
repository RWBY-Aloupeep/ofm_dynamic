if type(add_rules) == "function" then
    add_rules("mode.release", "mode.debug")
end

add_requires("cuda", { system = true, configs = { utils = { "cublas", "cusparse", "cusolver" } } })

if type(option) == "function" then
    option("with_gui")
        set_default(false)
        set_showmenu(true)
        set_description("Enable GUI/renderer projects and their Vulkan/GLFW/ImGui/VTK dependencies")
    
    option("headless")
        set_default(true)
        set_showmenu(true)
        set_description("Configure/build only headless targets and skip GUI/renderer dependencies")
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

if headless then
    with_gui = false
end

local ofm_xmake = "./../src/ofm/xmake.lua"
if os.isfile(ofm_xmake) then
    includes(ofm_xmake)
else
    cprint("${yellow}warning: missing %s (did you run submodule init?)", ofm_xmake)
end

if with_gui then
    local engine_xmake = "./../src/engine/xmake.lua"
    if os.isfile(engine_xmake) then
        includes(engine_xmake)
    else
        cprint("${yellow}warning: missing %s (did you run submodule init?)", engine_xmake)
    end
    
    add_requires("vulkansdk", "glfw 3.4", "glm 1.0.1")
    add_requires("glslang 1.3", { configs = { binaryonly = true } })
    add_requires("imgui 1.91.1", { configs = { glfw_vulkan = true } })
    add_requires("vtk 9.3.1")
end

if type(set_policy) == "function" then
    set_policy("build.intermediate_directory", false)
end
if type(set_runtimes) == "function" then
    set_runtimes("MD")
end

includes("dynamic_obstacle_headless")
if with_gui then
    includes("voxelization", "dynamic_obstacle")
end
if type(add_options) == "function" then
    add_options("compile_commands")
end