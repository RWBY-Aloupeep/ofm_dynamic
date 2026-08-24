#pragma once

// Headless compatibility shim.
//
// AMGPCG_Pybind_Torch/common/timer.cu is the only file in the solver stack that
// includes "core/tool/logger.h", which normally comes from the render engine and
// pulls in spdlog, the Vulkan-based Configuration type, and a global Logger
// instance. The self-check harness is a console program with no renderer, so this
// header supplies the handful of logging macros timer.cu actually uses
// (INFO_ALL / ERROR_ALL) as plain stream output.
//
// Nothing outside proj/selfcheck sees this file: it is reachable only through this
// target's include path, so the GUI applications keep using the engine's logger.

#include <iostream>
#include <string>

namespace selfcheck_compat {

inline void LogEmit(std::ostream&)
{
}

template <typename T, typename... Rest>
inline void LogEmit(std::ostream& os, const T& first, const Rest&... rest)
{
    os << first;
    LogEmit(os, rest...);
}

template <typename... Args>
inline void LogLine(std::ostream& os, const char* level, const Args&... args)
{
    os << "[" << level << "] ";
    LogEmit(os, args...);
    os << std::endl;
}

} // namespace selfcheck_compat

#define SELFCHECK_LOG_OUT(level, ...) ::selfcheck_compat::LogLine(std::cout, level, __VA_ARGS__)
#define SELFCHECK_LOG_ERR(level, ...) ::selfcheck_compat::LogLine(std::cerr, level, __VA_ARGS__)

#define TRACE_ALL(...) SELFCHECK_LOG_OUT("trace", __VA_ARGS__)
#define DEBUG_ALL(...) SELFCHECK_LOG_OUT("debug", __VA_ARGS__)
#define INFO_ALL(...) SELFCHECK_LOG_OUT("info", __VA_ARGS__)
#define WARN_ALL(...) SELFCHECK_LOG_OUT("warn", __VA_ARGS__)
#define ERROR_ALL(...) SELFCHECK_LOG_ERR("error", __VA_ARGS__)
#define CRITICAL_ALL(...) SELFCHECK_LOG_ERR("critical", __VA_ARGS__)

#define TRACE_CONSOLE(...) TRACE_ALL(__VA_ARGS__)
#define DEBUG_CONSOLE(...) DEBUG_ALL(__VA_ARGS__)
#define INFO_CONSOLE(...) INFO_ALL(__VA_ARGS__)
#define WARN_CONSOLE(...) WARN_ALL(__VA_ARGS__)
#define ERROR_CONSOLE(...) ERROR_ALL(__VA_ARGS__)
#define CRITICAL_CONSOLE(...) CRITICAL_ALL(__VA_ARGS__)

#define TRACE_FILE(...) TRACE_ALL(__VA_ARGS__)
#define DEBUG_FILE(...) DEBUG_ALL(__VA_ARGS__)
#define INFO_FILE(...) INFO_ALL(__VA_ARGS__)
#define WARN_FILE(...) WARN_ALL(__VA_ARGS__)
#define ERROR_FILE(...) ERROR_ALL(__VA_ARGS__)
#define CRITICAL_FILE(...) CRITICAL_ALL(__VA_ARGS__)
