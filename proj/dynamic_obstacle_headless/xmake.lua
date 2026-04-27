target("dynamic_obstacle_headless")
    set_targetdir("build")

    if is_plat("windows") then
        add_rules("plugin.vsxmake.autoupdate")
        add_cxxflags("/utf-8")
    end

    set_languages("cxx20")
    set_kind("binary")

    add_files("main.cu")
    add_includedirs(".", { public = true })

    add_cugencodes("compute_75")
    add_cuflags("--std c++20", "-lineinfo")

    add_deps("ofm")
    if type(has_target) == "function" and has_target("data_io") then
        add_deps("data_io")
    end

    add_packages("cuda")

    if is_mode("debug") then
        add_cxxflags("-DDEBUG")
    end
    if is_mode("release") then
        add_cxxflags("-DNDEBUG")
    end