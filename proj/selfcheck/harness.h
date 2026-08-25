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

// A divergence-free velocity field with closed-form vorticity and vertical-velocity
// gradient, used to calibrate the tilting/stretching operator:
//
//   u = ( -omega y - (g/2) x,  omega x - c z - (g/2) y,  w0 + s x + g z )
//
// giving vorticity (c, -s, 2 omega) and grad(w) = (s, 0, g), so that
//   tilting    = w_x d_x w + w_y d_y w = c s
//   stretching = w_z d_z w             = 2 omega g
// both uniform in space, which makes any error in the operator obvious.
struct ShearSpec {
    float omega; // solid-body rotation rate about z
    float c;     // vertical shear of the y velocity, sets w_x
    float s;     // horizontal gradient of the vertical velocity, sets w_y and d_x w
    float g;     // vertical stretching rate
    float w0;
};

// Split of the vertical vorticity source into the two terms the fire whirl
// literature argues about. Averages are over the interior, excluding the two
// outermost cell layers where the centred differences would be one-sided.
struct TiltStretch {
    double tilting_mean;
    double stretching_mean;
    double tilting_abs_mean;
    double stretching_abs_mean;
    double ratio; // |tilting| / (|tilting| + |stretching|)
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

// Writes the analytic shear field above into the solver's velocity state.
void SetShearFieldAsync(ofm::OFM& solver, const ShearSpec& spec, cudaStream_t stream);

// Reports tilting and stretching separately, by central differences on the
// cell-centred velocity.
TiltStretch MeasureTiltingStretching(ofm::OFM& solver, cudaStream_t stream);

// Line integral of a staggered vector field around a circle of the given radius,
// centred on (centre_x, centre_y) in the mid-z plane, traversed counter-clockwise.
//
// Applied to the velocity this is the circulation of that loop. Applied to a
// source channel's accumulator it is that channel's contribution to the loop's
// circulation over the cycle, because the accumulator holds the same pullback
// that the impulse itself is built from.
//
// Clobbers solver.u_, which the other diagnostics here also use as scratch.
double CirculationOnCircle(ofm::OFM& solver,
                           const ofm::DHMemory<float>& field_x, const ofm::DHMemory<float>& field_y, const ofm::DHMemory<float>& field_z,
                           float centre_x, float centre_y, float radius, int samples, cudaStream_t stream);

} // namespace selfcheck
