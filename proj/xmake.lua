add_rules("mode.release", "mode.debug")

includes("./../src/ofm/xmake.lua")

add_requires("cuda", {system=true, configs={utils={"cublas","cusparse","cusolver"}}})

set_policy("build.intermediate_directory", false)
set_runtimes("MD")

includes("headless_ofm")
add_options("compile_commands")

option("all")
    set_default(true)
    set_showmenu(false)
    set_description("Build all examples")