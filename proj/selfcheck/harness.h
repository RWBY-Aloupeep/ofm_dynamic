#pragma once

// Shared setup and measurement helpers for the D1 solver self-check cases.

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

// Same integral at many radii, but the field is pulled back to the host once
// instead of once per radius. A sweep fine enough to place an isocontour needs
// hundreds of loops, and at 256^3 one download per loop is minutes of transfer.
std::vector<double> CirculationRadialSweep(ofm::OFM& solver,
                                           const ofm::DHMemory<float>& field_x, const ofm::DHMemory<float>& field_y, const ofm::DHMemory<float>& field_z,
                                           float centre_x, float centre_y, const std::vector<double>& radii, int samples, cudaStream_t stream);

// ---------------------------------------------------------------------------
// D3: core radii
// ---------------------------------------------------------------------------

// An azimuthally and axially averaged radial profile, binned at the grid
// spacing. Bin b covers [b*dx, (b+1)*dx) and is reported at its centre.
struct RadialProfile {
    std::vector<double> r;
    std::vector<double> v;
    bool valid = true;
};

enum ProfileKind {
    kProfileAzimuthal = 0, // U_theta, the component b_w is fitted to
    kProfileAxial     = 1  // U_z, the component b_A is taken from
};

// A columnar axial jet, uniform in z, carrying a Gaussian radial profile
//   u_z(r) = w_peak * exp(-r^2 / scale^2).
// Uniform in z with no transverse component, so it is divergence free; it does
// violate the normal-velocity condition at the z walls, which is why the case
// that uses it takes no time steps.
struct AxialJetSpec {
    float centre_x;
    float centre_y;
    float scale;
    float w_peak;
};
void SetAxialJetAsync(ofm::OFM& solver, const AxialJetSpec& spec, cudaStream_t stream);

RadialProfile MeasureRadialProfile(ofm::OFM& solver, int kind, float centre_x, float centre_y, float r_max, cudaStream_t stream);

// The same binning applied to an analytic Gaussian evaluated at cell centres,
// so a profile the solver does not carry (the excess temperature, until the
// low-Mach extension provides one) reaches the estimators with the same
// discretisation error as the ones it does.
RadialProfile GaussianProfileOnGrid(const ofm::OFM& solver, float centre_x, float centre_y, float r_max,
                                    double amplitude, double scale);

// Least-squares fit of the Burgers profile (Tohidi et al. 2018, Eq. 7)
//   U_theta(r) = (Gamma_inf / 2 pi r) (1 - exp(-r^2 / b_w^2))
// to a measured profile. The model is linear in Gamma_inf, so the fit is a
// one-dimensional search over b_w with Gamma_inf eliminated in closed form.
struct BurgersFit {
    double gamma_inf;
    double b_w;
    double r_peak;   // peak of the measured profile, refined sub-bin
    double ratio;    // r_peak / b_w, which the model puts at 1.12091
    double rms_rel;  // RMS residual over the peak value
    int iters;
    bool ok;
};
BurgersFit FitBurgers(const RadialProfile& profile);

// Root of exp(-x)(2x + 1) = 1, whose square root is the constant Tohidi et al.
// quote as 1.12091. Computed rather than assumed, so the fit is checked against
// the model and not against a transcribed number.
double BurgersPeakConstant();

// Tohidi et al. 2018 Sec. 4.2: b_A and b_T are the radii at which the axial
// velocity and the excess temperature fall to a fraction of their maximum at
// that height -- 0.5 in the continuous flame region (Lei et al. 2015b).
double RadiusAtFraction(const RadialProfile& profile, double fraction);

// Tohidi et al. 2018, Eq. 9: b_A = Q_hat / sqrt(M_hat) from the specific mass
// and axial momentum fluxes. Unlike the fraction-of-maximum form this needs no
// top-hat assumption, and for a Gaussian profile of scale a it returns a.
double RadiusFromFluxes(const RadialProfile& axial);

// ---------------------------------------------------------------------------
// D4: vortex-flame Damkohler number
// ---------------------------------------------------------------------------

// Linan, Vera & Sanchez 2015, Sec. 7: the strain a vortex imposes on a flame is
// A_Gamma = Gamma / (2 r0^2), and the vortex Damkohler number Da_Gamma = A_e /
// A_Gamma compares the vortex turnover time with the chemical time, with local
// extinction expected for Da_Gamma <~ 1. Evaluated pointwise by taking Gamma and
// r0 at the same radius, which reproduces the paper's definition at r0.
struct DamkohlerCurve {
    std::vector<double> r;
    std::vector<double> gamma;
    std::vector<double> a_gamma;
    std::vector<double> da;
    double r_contour; // radius where Da_Gamma = 1; negative when there is none
    bool valid;
};
DamkohlerCurve MeasureDamkohler(ofm::OFM& solver, float centre_x, float centre_y,
                                double a_extinction, double r_min, double r_max, int radius_samples,
                                int loop_samples, cudaStream_t stream);

// Where Da_Gamma = 1 sits for an analytic Burgers vortex. With x = (r/b_w)^2 the
// condition reduces to (1 - exp(-x))/x = 2 A_e b_w^2 / Gamma_inf, whose left side
// decreases monotonically from 1 to 0, so there is one root when the right side
// is below 1 and none otherwise. Returns a negative value in the latter case.
double BurgersDamkohlerContour(double gamma_inf, double b_w, double a_extinction);

// The extinction strain rate that places the analytic contour at r = b_w*sqrt(x).
double ExtinctionRateForContour(double gamma_inf, double b_w, double x);

// ---------------------------------------------------------------------------
// Stage A: a buoyant plume from a surface heat source in a sheared cross flow,
// after Cunningham, Goodrick, Hussaini & Linn 2005 (Int. J. Wildland Fire 14,
// 61-75).
//
// Their model solves the compressible equations in a density-stratified
// atmosphere; this is the solver's incompressible Boussinesq reduction of it.
// Their base state is neutral -- ambient potential temperature uniform at
// 300 K -- which is what makes the reduction defensible at all. What it drops
// is the volume expansion of the heated air, and near the source that is not
// small: at Q0 = 1 kW/m^3 the parcel-following temperature anomaly reaches
// several tens of K, so dT/T0 is O(0.1) rather than O(0.01). Treat the plume
// structure as reproduced and the absolute widths as approximate until the
// low-Mach extension lands.
struct PlumeSpec {
    float u0     = 4.5f;    // m/s, cross-flow speed well above the shear layer
    float z0     = 100.0f;  // m, shear-layer depth in U(z) = u0*tanh(z/z0)
    float q0     = 1000.0f; // W/m^3, peak volumetric heating
    float src_x  = 450.0f;  // m, heat-source centre
    float src_y  = 600.0f;  // m
    float r1     = 75.0f;   // m, uniformly heated radius
    float r2     = 150.0f;  // m, outer radius of the heated area
    float dwidth = 12.5f;   // m, scale width of the smoothed edge
    float h      = 25.0f;   // m, vertical decay scale in Q ~ exp(-z/h)
    float t_ramp = 10.0f;   // s, ramp-up in Q ~ tanh(t/t_ramp)
    float theta0 = 300.0f;  // K, ambient potential temperature
    float rho    = 1.177f;  // kg/m^3 at 300 K, 1000 hPa
    float cp     = 1005.0f; // J/(kg K)
    float g      = 9.81f;   // m/s^2
    float cd_a   = 0.025f;  // Cd*a = 0.1 * 0.25 1/m, lowest cell level only
};

// Inflow and outflow both carry U(z), so mass balances exactly; the lateral and
// top faces are free-slip walls. This is NOT what Cunningham used -- they put a
// non-reflecting Orlanski outflow on the lateral and downstream faces and a
// damping layer under the lid. The solver has no outflow condition, so the
// downstream face is a prescribed profile instead. Keep the measurement plane
// well clear of it.
void SetPlumeBcAsync(ofm::OFM& solver, const PlumeSpec& spec, cudaStream_t stream);
void SetPlumeInitialVelocityAsync(ofm::OFM& solver, const PlumeSpec& spec, cudaStream_t stream);

// theta is the potential-temperature anomaly, cell centred, Prod(tile_dim)*512.
void AddPlumeHeatAsync(ofm::DHMemory<float>& theta, int3 tile_dim, float3 grid_origin, float dx,
                       const PlumeSpec& spec, float t, float dt, cudaStream_t stream);

// Writes the whole external-force field: f_z = g*theta/theta0 on the z faces,
// and the canopy drag -Cd*a*|u_h|*u_i on the lowest cell level, zero elsewhere.
void SetBuoyancyAndDragAsync(ofm::OFM& solver, const ofm::DHMemory<float>& theta,
                             const PlumeSpec& spec, cudaStream_t stream);

struct PlumeDiag {
    double max_theta;   // K, over the whole field
    double plume_top;   // m, highest cell centre with theta > 0.25 K
    double w_max;       // m/s, over the whole field
    double omega_pos;   // 1/s, strongest cyclonic vertical vorticity on the plane
    double omega_neg;   // 1/s, strongest anticyclonic
    double y_pos;       // m, where omega_pos sits
    double y_neg;       // m
    double split_width; // m, |y_pos - y_neg|
    // Where the pair is actually strongest, found by scanning every x plane at
    // the same height. A plume that never bends over leaves the requested plane
    // empty, and then these are the only numbers that say anything.
    double best_x;      // m
    double best_omega;  // 1/s, max |omega_z| on that plane
    double best_split;  // m
    double plane_theta; // K, peak anomaly on the requested plane
    double u_max;       // m/s, peak horizontal speed in the domain
    bool   valid;
};

// Measured on the y-z plane nearest plane_x, at the height nearest cvp_z, from
// the cell-centred velocity. Cunningham report positive vertical vorticity on
// the right-hand side looking downstream and negative on the left.
PlumeDiag MeasurePlume(ofm::OFM& solver, ofm::DHMemory<float>& theta,
                       float plane_x, float cvp_z, cudaStream_t stream);

} // namespace selfcheck
