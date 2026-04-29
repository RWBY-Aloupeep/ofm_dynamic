#pragma once

#include <cuda_runtime.h>
#include <string>

class GPUTimer {
public:
    class Scope {
    public:
        Scope() = default;
        ~Scope() = default;
        Scope(const Scope&) = default;
        Scope(Scope&&) noexcept = default;
        Scope& operator=(const Scope&) = default;
        Scope& operator=(Scope&&) noexcept = default;
    };

    GPUTimer();
    explicit GPUTimer(bool _enable);
    ~GPUTimer();

    Scope profileScope(cudaStream_t _stream, const std::string& _name);
    Scope profileScope(const std::string& _name);

    void record(cudaStream_t _stream, const std::string& _name);
    void record(const std::string& _name);
    void sync();
    void clear();
    void print();
    void printAndClear();
    void setEnable(bool _enable);
    bool isEnable() const;
};

#ifndef CUDA_PROFILE_SCOPE
#define CUDA_PROFILE_SCOPE(_profiler, _stream, _name) \
    [[maybe_unused]] auto _gpu_profile_scope_##__LINE__ = (_profiler).profileScope((_stream), (_name))
#endif
