#pragma once

// Shared setup and measurement helpers for the D1 solver self-check cases.

#include "ofm.h"
#include "timer.h"

#include <cstdio>
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
    // Cap on the multigrid levels, 0 for the solver's own choice (see ofm.h).
    int max_levels   = 0;
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

// Abort before any measurement if the kernels cannot run on this device. The
// build carries cubin for sm_75 (gpu-rtx6k) and sm_89 (gpu-l40, gpu-l40s) plus
// sm_89 PTX; on an older architecture -- the ckpt partition mixes several --
// every launch fails with "no kernel image is available for execution on the
// device". Nothing in the solver checks launch status, so without this the run
// finishes in milliseconds with every field still zero and prints that as a
// result: a whole sweep of zeros that looks like physics. Returns false and
// says what it found instead.
bool CheckDeviceUsable();

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
    // Elevated source for localisation: when > 0 the heating is
    // Q0/2 exp(-|z - z_src|/h), centred at z_src with the same total heat as
    // the ground-based profile, so the plume's root sits away from the wall.
    float z_src  = 0.0f;
    float t_ramp = 10.0f;   // s, ramp-up in Q ~ tanh(t/t_ramp)
    float theta0 = 300.0f;  // K, ambient potential temperature
    float rho    = 1.177f;  // kg/m^3 at 300 K, 1000 hPa
    float cp     = 1005.0f; // J/(kg K)
    float g      = 9.81f;   // m/s^2
    float cd_a   = 0.025f;  // Cd*a = 0.1 * 0.25 1/m, lowest cell level only
    // Which faces are open, and how (see ofm.h): 0 keeps the first cut's closed
    // box with U(z) prescribed on both x faces; 1 and 2 make the downstream face,
    // or the downstream and both lateral faces, zero-gauge pressure outlets;
    // 3 and 4 make the same faces convective. The three-face set is the one
    // Cunningham put their outflow condition on.
    int outflow  = 0;
    // Convective faces at the reinitialization's projection read the last
    // projected velocity, not the impulse-reconstructed one (see ofm.h).
    bool conv_face_projected = false;
    // Thermal diffusion of theta at kappa = mu / (rho Pr). Cunningham's direct
    // runs tie the conductivity to the viscosity through Pr = 0.7; 0 turns the
    // term off, which is what the first two cuts ran.
    float pr     = 0.7f;
    // Rayleigh damping layer under the lid, in the form Wang et al. 2023 give
    // (after Klemp & Lilly 1978): beta(zeta) = (20 dt)^-1 sin^2(pi/2 zeta/zeta0)
    // over the top zeta0 = sponge_depth * Lz, relaxing u to U(z), v and w to 0
    // and theta to 0. Cunningham report such a layer but not its form.
    bool sponge  = false;
    float sponge_depth = 0.1f;
};

// The inflow face carries U(z). With outflow = 0 the downstream face carries the
// same profile, so mass balances exactly and the box is closed with free-slip
// walls elsewhere -- the first cut's configuration, and NOT what Cunningham
// used. With outflow = 1 or 2 the downstream, or downstream and lateral, faces
// are left open: the pressure is pinned to zero beyond them and the projection
// sets their normal velocity, so the plume leaves (or entrains) at whatever
// rate the interior asks for. This is the pressure-outlet condition of Barata
// et al. 2024/2025 and FDS's OPEN boundary, not Cunningham's Orlanski radiation
// condition, and is verified on its own by the translating-vortex case below.
// The top and bottom stay free-slip walls in every mode.
void SetPlumeBcAsync(ofm::OFM& solver, const PlumeSpec& spec, cudaStream_t stream);
void SetPlumeInitialVelocityAsync(ofm::OFM& solver, const PlumeSpec& spec, cudaStream_t stream);

// theta is the potential-temperature anomaly, cell centred, Prod(tile_dim)*512.
void AddPlumeHeatAsync(ofm::DHMemory<float>& theta, int3 tile_dim, float3 grid_origin, float dx,
                       const PlumeSpec& spec, float t, float dt, cudaStream_t stream);

// Writes the whole external-force field: f_z = g*theta/theta0 on the z faces,
// and the canopy drag -Cd*a*|u_h|*u_i on the lowest cell level, zero elsewhere.
// The three staggered components of one velocity field. init_u_ is only current
// at a reinitialization-cycle boundary, so at reinit_every_ > 1 nothing in the
// Stage A step may read it: the buoyancy, the drag and the theta advection all
// have to take the cycle's own history instead. mid_u_[i] holds the projected
// velocity of step i of the cycle, which sits at that step's midpoint.
struct PlumeVelocity {
    ofm::DHMemory<float>* x;
    ofm::DHMemory<float>* y;
    ofm::DHMemory<float>* z;
};

// The latest projected velocity, for use before AdvanceAsync: the cycle's start
// on its first step, the previous step's velocity after that.
PlumeVelocity PlumeVelocityBefore(ofm::OFM& solver);

// The velocity of the step AdvanceAsync has just taken, for use after it. Being
// a midpoint velocity, this is the right one to advect theta across that step.
PlumeVelocity PlumeVelocityAfter(ofm::OFM& solver);

void SetBuoyancyAndDragAsync(ofm::OFM& solver, const ofm::DHMemory<float>& theta,
                             const PlumeSpec& spec, PlumeVelocity u, float dt, cudaStream_t stream);

// Explicit thermal diffusion of the cell-centred theta over one step:
// dst = src + (kappa dt / dx^2) * seven-point Laplacian, zero-gradient at the
// domain faces. Stable for kappa dt / dx^2 < 1/6.
void DiffuseThetaAsync(ofm::DHMemory<float>& dst, const ofm::DHMemory<float>& src, int3 tile_dim,
                       float kappa, float dx, float dt, cudaStream_t stream);

// Air entering through an open face is ambient air. The semi-Lagrangian step
// cannot know that -- its backtrace is clamped to the domain, so a cell next to
// an open face that the flow is entering through would keep its own theta and
// re-import the plume's buoyancy from outside. This replaces the fraction of
// such a cell swept in over the step, |u_n| dt / dx, with ambient theta = 0.
// Applied after the advection, with the same velocity.
void AmbientInflowThetaAsync(ofm::DHMemory<float>& theta, int3 tile_dim, PlumeVelocity u, int outflow,
                             float dx, float dt, cudaStream_t stream);

// The sponge's relaxation of theta towards the ambient over one step, applied in
// place; the velocity part goes through the external force in
// SetBuoyancyAndDragAsync.
void SpongeThetaAsync(ofm::DHMemory<float>& theta, int3 tile_dim, float3 grid_origin, float dx,
                      const PlumeSpec& spec, float dt, cudaStream_t stream);

// Advect theta one step. theta is the field every one of the paper's criteria is
// read from, and in the first cut it was the only field in the case carrying no
// error compensation at all: the velocity goes through the flow map and the
// solver's BFECC pass, while theta got a bare semi-Lagrangian step whose
// trilinear interpolation is first order in space. With bfecc set this applies
// the same correction the solver applies to velocity, in the same order --
// advect, advect back, subtract, advect the error, correct, clamp -- reusing
// BfeccClampAsync for the clamp. fwd and err are scratch, cell centred and the
// same size as theta; dst must be neither src nor scratch.
void AdvectThetaAsync(ofm::DHMemory<float>& dst, ofm::DHMemory<float>& fwd, ofm::DHMemory<float>& err,
                      int3 tile_dim, ofm::DHMemory<float>& src, PlumeVelocity u,
                      float dx, float dt, bool bfecc, bool clamp, cudaStream_t stream);

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
    // What Cunningham actually measured. Their Fig. 6 is the potential-temperature
    // cross section on this plane, contoured every 0.25 K starting at 300.25 K,
    // and both statements the figure supports -- wider bifurcation for the deeper
    // shear layer, wider for the weaker source -- are read off those contours.
    // The omega_z extrema above are only loosely related to them, so comparing
    // the two is not a fair test of either. The plane is reduced to the column
    // maximum P(y) = max_z theta(y,z) and measured there.
    double theta_width;  // m, between the outermost P = 0.25 K crossings
    double theta_split;  // m, between the two branch maxima; 0 when single-lobed
    double theta_left;   // m, left branch
    double theta_right;  // m, right branch
    double theta_saddle; // K, the minimum of P between the two branches
    double theta_peak;   // K, the lower of the two branch maxima
    bool   bifurcated;   // a plotted contour level falls between saddle and peak
    // The same measure on the strongest-CVP plane, which is far enough upstream
    // to be clear of the prescribed outflow.
    double best_theta_width;
    double best_theta_split;
    bool   best_bifurcated;
};

// Measured on the y-z plane nearest plane_x, at the height nearest cvp_z, from
// the cell-centred velocity. Cunningham report positive vertical vorticity on
// the right-hand side looking downstream and negative on the left.
// ---------------------------------------------------------------------------
// Verification of the open (pressure-outlet) boundary: a columnar vortex with
// the Gaussian vorticity omega = Gamma/(pi a^2) exp(-r^2/a^2) -- the profile of
// Tohidi et al. 2018 Eq. 7 that the D1 and D3 cases already use -- carried by a
// uniform stream U along +x through the downstream face. In an unbounded inviscid
// fluid it translates unchanged, so the solution is known at every time. The
// inflow and lateral faces prescribe that solution's normal velocity, updated
// every step, and the top and bottom are free-slip walls the z-uniform field
// satisfies exactly; only the downstream face carries the condition under test.
// A vortex core has a low pressure, and the outlet pins the pressure to zero, so
// the condition is not exact as the core crosses the face: the case measures
// how much that costs, and a run in a domain twice as long, where the vortex
// never reaches the face, supplies the solver's own error for comparison.
struct TranslatingVortexSpec {
    float u_stream    = 1.0f;
    float x0          = 0.75f; // centre at t = 0
    float y0          = 0.5f;
    float core        = 0.05f; // a, 6.4 cells at dx = 1/128
    float circulation = 0.25f; // peak swirl about half the stream
    int outflow       = 1;     // 1: pressure outlet; 2: convective; 0: prescribed at u_stream (control)
};

// Stream plus vortex, written into the solver's velocity state.
void SetTranslatingVortexAsync(ofm::OFM& solver, const TranslatingVortexSpec& spec, cudaStream_t stream);

// A Gaussian vortex column in a planar shear u_x = S (y - y0), the
// known-answer case for the flow map's reconstruction under accumulated
// strain. The flow is two-dimensional and inviscid, so omega_z is conserved
// along particles and the circulation of any fixed circle that keeps the
// whole core inside it is exactly Gamma - S pi r^2 for all time. The strain
// the map accumulates per cycle is S n dt, uniform, which is the quantity
// the adaptive reinitialization bounds. x faces prescribe the shear profile
// (net flux zero), y and z faces are free-slip walls.
struct ShearVortexSpec {
    float shear       = 1.0f;  // S, 1/s
    float x0          = 0.5f;
    float y0          = 0.5f;
    float core        = 0.05f;
    float circulation = 0.25f;
};
void SetShearVortexAsync(ofm::OFM& solver, const ShearVortexSpec& spec, cudaStream_t stream);
void SetShearVortexBcAsync(ofm::OFM& solver, const ShearVortexSpec& spec, cudaStream_t stream);

// The exact normal velocity at time t on the Dirichlet faces. Call with build
// true once, so the Poisson coefficients match the face marks.
void SetTranslatingVortexBcAsync(ofm::OFM& solver, const TranslatingVortexSpec& spec, float t, bool build, cudaStream_t stream);

// Vertical vorticity at the cell corners from the face velocities directly, so
// the sum over the corner set is the exact discrete circulation around the
// rectangle half a cell in from the domain edge, and the analytic value is the
// Gaussian's integral over that same rectangle.
struct TranslatingVortexDiag {
    double gamma_in;         // circulation inside the rectangle
    double gamma_in_exact;
    // The same sum restricted to a box of half-width six core radii about the
    // exact centre, clipped to the domain. Vorticity that appears along the
    // lateral walls counts in gamma_in but not here, which is how the two
    // are told apart.
    double gamma_core;
    double gamma_core_exact;
    double min_omega;        // most negative corner vorticity; the exact field has none
    // Where the circulation outside the core box sits: the two corner rows
    // next to each lateral wall, the two columns next to the inflow face, and
    // everything else. Exact values are all zero to round-off once the core
    // box holds the vortex.
    double gamma_walls, gamma_inflow, gamma_elsewhere;
    double peak_omega;       // largest corner vorticity
    double peak_omega_exact; // Gamma / (pi a^2)
    double peak_x, peak_y;   // where the peak sits; the exact centre is (x0 + U t, y0)
    double l2_all;           // ||omega - omega_exact|| / ||omega_exact(t = 0)||, whole rectangle
    double l2_interior;      // the same restricted to x < x_out - margin
    double max_w;            // largest |w|, which the exact solution has at zero
    bool valid;
};
// x_out is the face under test: the sums run over the corners with x < x_out,
// so a longer box is measured on exactly the nodes of the short one and the
// difference between the two is the boundary's doing alone.
TranslatingVortexDiag MeasureTranslatingVortex(ofm::OFM& solver, const TranslatingVortexSpec& spec, float t,
                                               float x_out, float interior_margin, cudaStream_t stream);

PlumeDiag MeasurePlume(ofm::OFM& solver, ofm::DHMemory<float>& theta,
                       float plane_x, float cvp_z, cudaStream_t stream);

// Append the theta anomaly on the y-z plane nearest plane_x to a raw binary
// stream, so the paper's Fig. 6 can be drawn from the run and compared
// panel for panel. Layout: once, the header "OFMSLICE", int32 ny, int32 nz,
// float dx, float y0, float z0 (cell-centre origin), float plane_x actually
// used; then per call float time followed by ny*nz floats, j outer, k inner.
// Call after MeasurePlume, which leaves the host copy of theta current.
void WritePlumeSlice(FILE* f, ofm::OFM& solver, ofm::DHMemory<float>& theta,
                     float plane_x, float time, bool header, cudaStream_t stream);

// Largest velocity-gradient component in the domain (1/s), the per-step
// strain bound the adaptive reinitialization integrates. Synchronizes the stream.
float MaxVelocityGradient(ofm::OFM& solver, const PlumeVelocity& u, cudaStream_t stream);

// Horizontal means of w and theta over the column [x0, x1] x [y0, y1], one line
// per z level: "time z w_mean theta_mean w_max". Reads the host copies that
// MeasurePlume has just refreshed (call after it). For finding where a plume
// loses its updraught: at the wall or aloft.
void WritePlumeProfile(FILE* f, ofm::OFM& solver, ofm::DHMemory<float>& theta,
                       float x0, float x1, float y0, float y1, float time);

// The x-z section through the plume's centreline (the j plane nearest
// plane_y): theta anomaly, cell-centred u and w, so the transverse (omega_y)
// structure on the plume's upstream face can be drawn. Layout: once, the
// header "OFMSLXZ", int32 nx, int32 nz, float dx, float x0, float z0 (cell
// centres), float y actually used; then per call float time followed by
// theta, u, w, each nx*nz floats, i outer, k inner. Call after MeasurePlume,
// which leaves the host copies current.
void WritePlumeSliceXZ(FILE* f, ofm::OFM& solver, ofm::DHMemory<float>& theta,
                       float plane_y, float time, bool header);

// Point probes for time series: u, v, w (from the staggered faces) and theta
// at each of n cell positions, gathered on the device into one small buffer
// and copied back, so they can be sampled every step. For the shedding
// criterion: a wake probe's v(t) spectrum gives the shedding frequency.
struct ProbeSet {
    std::vector<int3> cells;
    float* d_buf = nullptr;     // 4 floats per probe
    std::vector<float> h_buf;
};
void SetupProbes(ProbeSet& p, ofm::OFM& solver, const std::vector<float3>& positions);
// Reads u from the given velocity buffers (init_u_ at a cycle start, mid_u_ otherwise).
void SampleProbes(ProbeSet& p, ofm::OFM& solver, const PlumeVelocity& u, ofm::DHMemory<float>& theta, cudaStream_t stream);

// Largest |div u| over the cells (1/s) of the staggered velocity init_u_, i.e.
// what the last projection left behind. Downloads the three components.
float MaxDivergence(ofm::OFM& solver, cudaStream_t stream);
// The same divergence, summarised: where the maximum sits, the rms over all
// cells and over the interior (two or more cells from every boundary), and
// the share of the sum of squares in the one-cell boundary layer. The 2-norm
// is what AMGPCG's tolerance test sees; the maximum is what --log-div reports.
struct DivStats {
    float max = 0.0f;
    int3 at = { 0, 0, 0 };
    float rms = 0.0f, rms_interior = 0.0f;
    double boundary_share = 0.0;   // fraction of sum(div^2) within one cell of a boundary
    double l2 = 0.0;               // sqrt(sum div^2) over all cells, the solver's norm up to a scale
    double mean = 0.0;             // signed mean of div over all cells: a net flux imbalance the
                                   // pure-Neumann recentering drops from the right-hand side
    float rms_zero_mean = 0.0f;    // rms of div - mean, what the solver can act on
};
DivStats DivergenceStats(ofm::OFM& solver, cudaStream_t stream);

} // namespace selfcheck
