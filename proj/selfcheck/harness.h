#pragma once

// Shared setup and measurement helpers for the D5 solver self-check cases.

#include "ofm.h"
#include "timer.h"

#include <vector>

namespace selfcheck {

// Solver configuration, mirroring the sequence ofm::InitOFMAsync performs but
// without the engine's JSON Configuration type.
struct SolverConfig {
    int3 tile_dim    = { 32, 16, 16 }; // 256 x 128 x 128 cells
    float len_y      = 1.0f;           // physical extent along y; dx follows from it
    float inlet_norm = 0.0f;
    float inlet_angle = 0.0f;
    int cg_iter      = 15;
    bool bfecc_clamp = true;
};

// A circular vortex filament with a regularized (Rosenhead-Moore) core.
struct RingSpec {
    float3 center;
    float3 axis; // need not be normalized
    float radius;
    float core;
    float circulation;
};

// Cell-centered scalar field pulled back to the host, in plain x-major order.
struct HostField {
    int3 dim;
    std::vector<float> data;

    float At(int i, int j, int k) const
    {
        return data[(static_cast<size_t>(i) * dim.y + j) * dim.z + k];
    }
};

struct FieldStats {
    double kinetic_energy;
    float max_speed;
    float max_vorticity;
    bool finite;
};

void SetupSolver(ofm::OFM& solver, const SolverConfig& config, GPUTimer& profiler, cudaStream_t stream);

// Superposes the Biot-Savart velocity of each ring onto the solver's current
// velocity state, then makes the result divergence free.
void AddVortexRingsAsync(ofm::OFM& solver, const std::vector<RingSpec>& rings, int num_segments, bool project, cudaStream_t stream);

void ProjectCurrentVelocityAsync(ofm::OFM& solver, cudaStream_t stream);

FieldStats ComputeFieldStats(ofm::OFM& solver, cudaStream_t stream);

// Cell-centered vorticity magnitude, de-tiled into x-major host order.
HostField DownloadVorticityNorm(ofm::OFM& solver, cudaStream_t stream);

} // namespace selfcheck
