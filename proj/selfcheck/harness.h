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
    // Steps per reinitialization cycle. 1 is the one-step (OFM) scheme; the LFM
    // paper runs its leapfrog vortex ring figure at 10.
    int reinit_every = 1;
    // Flow-map marching order: 2, 4, or anything else for TVD-RK3.
    int rk_order     = 3;
};

// A circular vortex filament with a regularized (Rosenhead-Moore) core.
struct RingSpec {
    float3 center;
    float3 axis; // need not be normalized
    float radius;
    float core;
    float circulation;
};

// A columnar vortex aligned with +z and uniform in z, carrying the Burgers
// profile U_theta(r) = (Gamma / 2 pi r)(1 - exp(-r^2 / b_w^2)) (Tohidi et al.
// 2018, Eq. 7 -- the model the fire whirl literature fits to measured cores).
struct ColumnVortexSpec {
    float centre_x;
    float centre_y;
    float core;        // b_w
    float circulation; // Gamma_inf
};

// Radial diagnostics of a columnar vortex, measured about its axis.
struct ColumnDiag {
    float max_vorticity;  // peak |omega|, at the axis
    float r_peak;         // radius where the azimuthal velocity is largest
    float u_theta_peak;
    bool valid;
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

// Superposes a columnar Burgers vortex onto the current velocity state.
void AddColumnVortexAsync(ofm::OFM& solver, const ColumnVortexSpec& spec, bool project, cudaStream_t stream);

// Peak vorticity and the radius of peak azimuthal velocity, by radial binning
// about the axis over the whole column.
ColumnDiag MeasureColumnVortex(ofm::OFM& solver, float centre_x, float centre_y, cudaStream_t stream);

} // namespace selfcheck
