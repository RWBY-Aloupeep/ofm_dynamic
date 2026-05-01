#include "utils/timer.h"

GPUTimer::GPUTimer() = default;
GPUTimer::GPUTimer(bool) {}
GPUTimer::~GPUTimer() = default;

GPUTimer::Scope GPUTimer::profileScope(cudaStream_t, const std::string&) { return {}; }
GPUTimer::Scope GPUTimer::profileScope(const std::string&) { return {}; }

void GPUTimer::record(cudaStream_t, const std::string&) {}
void GPUTimer::record(const std::string&) {}
void GPUTimer::sync() {}
void GPUTimer::clear() {}
void GPUTimer::print() {}
void GPUTimer::printAndClear() {}
void GPUTimer::setEnable(bool) {}
bool GPUTimer::isEnable() const { return false; }
