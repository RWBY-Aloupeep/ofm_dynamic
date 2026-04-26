add_rules("mode.release", "mode.debug")

target("headless")
    set_targetdir("build")
    set_kind("binary")
    set_languages("cxx20")

    add_files("main.cpp")
    add_includedirs(".", {public = true})
    add_includedirs("../..", {public = true})
    add_includedirs("./../src/ofm", {public = true})
    add_includedirs("./../src/engine", {public = true})

    add_cugencodes("compute_75")
    add_cuflags("--std c++20", "-lineinfo")

    add_deps("ofm")
    add_packages("cuda")

    if is_mode("debug") then
        add_cxxflags("-DDEBUG")
    end
    if is_mode("release") then
        add_cxxflags("-DNDEBUG")
    end