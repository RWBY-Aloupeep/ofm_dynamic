// D5 solver self-check suite.
//
// Establishes quantitative baselines for the shipped OFM solver on the vortex
// preservation cases LFM uses. OFM's own thesis reports only visual comparisons,
// so these numbers do not exist upstream; they are what later work (in particular
// the low-Mach extension) has to be checked against for regression.

#include "harness.h"

#include "ofm_util.h"
#include "util.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Cluster {
    double x;
    double r;
    double weight;
    bool valid;
};

// Vorticity-weighted 2-means in the (axial, radial) half plane. Seeded from the
// previous step so cluster identity follows each ring through a leap instead of
// being reassigned by position.
std::vector<Cluster> TrackTwoRings(const selfcheck::HostField& vor, float dx, float axis_y, float axis_z,
                                   std::vector<Cluster> seed, float rel_threshold)
{
    float max_w = 0.0f;
    for (float v : vor.data)
        if (v > max_w)
            max_w = v;

    const float threshold = rel_threshold * max_w;
    if (max_w <= 0.0f)
        return { { 0, 0, 0, false }, { 0, 0, 0, false } };

    // Collect the significant vorticity as (x, r, w) samples.
    struct Sample {
        double x, r, w;
    };
    std::vector<Sample> samples;
    samples.reserve(1 << 16);
    for (int i = 0; i < vor.dim.x; i++)
        for (int j = 0; j < vor.dim.y; j++)
            for (int k = 0; k < vor.dim.z; k++) {
                const float w = vor.At(i, j, k);
                if (w < threshold)
                    continue;
                const double y = (j + 0.5) * dx - axis_y;
                const double z = (k + 0.5) * dx - axis_z;
                samples.push_back({ (i + 0.5) * dx, std::sqrt(y * y + z * z), static_cast<double>(w) });
            }

    if (samples.size() < 2)
        return { { 0, 0, 0, false }, { 0, 0, 0, false } };

    std::vector<Cluster> c = std::move(seed);
    for (int iter = 0; iter < 20; iter++) {
        double sx[2] = { 0, 0 }, sr[2] = { 0, 0 }, sw[2] = { 0, 0 };
        for (const Sample& s : samples) {
            // Distance in the (x, r) plane; both coordinates are physical lengths.
            const double d0 = (s.x - c[0].x) * (s.x - c[0].x) + (s.r - c[0].r) * (s.r - c[0].r);
            const double d1 = (s.x - c[1].x) * (s.x - c[1].x) + (s.r - c[1].r) * (s.r - c[1].r);
            const int a     = (d0 <= d1) ? 0 : 1;
            sx[a] += s.x * s.w;
            sr[a] += s.r * s.w;
            sw[a] += s.w;
        }
        bool moved = false;
        for (int a = 0; a < 2; a++) {
            if (sw[a] <= 0.0)
                continue;
            const double nx = sx[a] / sw[a];
            const double nr = sr[a] / sw[a];
            if (std::fabs(nx - c[a].x) > 1e-9 || std::fabs(nr - c[a].r) > 1e-9)
                moved = true;
            c[a].x      = nx;
            c[a].r      = nr;
            c[a].weight = sw[a];
            c[a].valid  = true;
        }
        if (!moved)
            break;
    }
    return c;
}


// Reports where the vorticity actually lives, so the ring tracker can be checked
// against the field rather than trusted blindly.
void DescribeVorticity(const selfcheck::HostField& vor, float dx, float axis_y, float axis_z, const char* label)
{
    float max_w = 0.0f;
    int3 argmax = { 0, 0, 0 };
    for (int i = 0; i < vor.dim.x; i++)
        for (int j = 0; j < vor.dim.y; j++)
            for (int k = 0; k < vor.dim.z; k++)
                if (vor.At(i, j, k) > max_w) {
                    max_w  = vor.At(i, j, k);
                    argmax = { i, j, k };
                }

    // Peak vorticity in radial shells about the ring axis. Peaks rather than sums:
    // outer shells hold far more cells, so a sum hides a thin, intense core.
    const int nbin = 10;
    double shell[nbin];
    for (int b = 0; b < nbin; b++)
        shell[b] = 0.0;
    double total = 0.0;
    for (int i = 0; i < vor.dim.x; i++)
        for (int j = 0; j < vor.dim.y; j++)
            for (int k = 0; k < vor.dim.z; k++) {
                const double y = (j + 0.5) * dx - axis_y;
                const double z = (k + 0.5) * dx - axis_z;
                const double r = std::sqrt(y * y + z * z);
                const int b    = std::min(nbin - 1, static_cast<int>(r / 0.07));
                shell[b] = std::max(shell[b], static_cast<double>(vor.At(i, j, k)));
                total += vor.At(i, j, k);
            }

    printf("[%s] max|w|=%.3f at cell (%d,%d,%d) = (%.3f,%.3f,%.3f), r=%.3f\n",
           label, max_w, argmax.x, argmax.y, argmax.z,
           (argmax.x + 0.5f) * dx, (argmax.y + 0.5f) * dx, (argmax.z + 0.5f) * dx,
           std::sqrt(((argmax.y + 0.5f) * dx - axis_y) * ((argmax.y + 0.5f) * dx - axis_y)
                     + ((argmax.z + 0.5f) * dx - axis_z) * ((argmax.z + 0.5f) * dx - axis_z)));
    printf("[%s] peak |w| by radial shell:", label);
    for (int b = 0; b < nbin; b++)
        printf(" r<%.2f:%.1f", (b + 1) * 0.07, shell[b]);
    printf("\n");
}

struct RingParams {
    float radius      = 0.15f;  // ~19 cells at 256x128x128
    float core        = 0.04f;  // ~5 cells, enough to survive advection
    float circulation = 0.03f;
    float spacing     = 0.15f;
    float x_start     = 0.30f;
};

int RunLeapfrogRings(int total_steps, int diag_every, const char* csv_path, RingParams rp, float dt, int reinit_every, int rk_order)
{
    cudaStream_t stream = 0;
    selfcheck::SolverConfig config;
    config.tile_dim = { 32, 16, 16 }; // 256 x 128 x 128, matching LFM Table 4 for Figure 14
    config.len_y    = 1.0f;
    config.cg_iter  = 15;             // LFM Table 4 reports 15 CG iterations for this case
    config.reinit_every = reinit_every;
    config.rk_order     = rk_order;

    ofm::OFM solver;
    GPUTimer profiler(64);
    selfcheck::SetupSolver(solver, config, profiler, stream);

    const float dx     = solver.dx_;
    const float axis_y = 0.5f;
    const float axis_z = 0.5f;

    // Geometry is our own choice: LFM reports the resolution, reinitialization
    // interval and CG iteration count for this figure but not the ring radius,
    // core size, spacing or circulation.
    const float radius      = rp.radius;
    const float core        = rp.core;
    const float circulation = rp.circulation;
    const float x_trail     = rp.x_start;
    const float x_lead      = rp.x_start + rp.spacing;

    std::vector<selfcheck::RingSpec> rings = {
        { { x_trail, axis_y, axis_z }, { 1.0f, 0.0f, 0.0f }, radius, core, circulation },
        { { x_lead, axis_y, axis_z }, { 1.0f, 0.0f, 0.0f }, radius, core, circulation },
    };
    selfcheck::AddVortexRingsAsync(solver, rings, 256, true, stream);
    cudaStreamSynchronize(stream);

    FILE* csv = fopen(csv_path, "w");
    if (csv == nullptr) {
        printf("could not open %s for writing\n", csv_path);
        return 1;
    }
    fprintf(csv, "step,time,kinetic_energy,max_speed,max_vorticity,x_a,r_a,x_b,r_b,separation,leaps\n");

    std::vector<Cluster> clusters = {
        { x_trail, radius, 0.0, true },
        { x_lead, radius, 0.0, true },
    };

    int leaps       = 0;
    int prev_sign   = 0;
    bool merged     = false;
    int merge_step  = -1;
    // Hysteresis: a crossing only counts once the rings have first pulled apart
    // axially, so tracker jitter around x_a == x_b cannot inflate the count.
    double peak_abs_dx   = 0.0;
    int consecutive_close = 0;
    const double separate_tol = 0.3 * radius;
    const double merge_tol    = 0.35 * radius;
    const int merge_samples   = 5;

    {
        const selfcheck::HostField v0 = selfcheck::DownloadVorticityNorm(solver, stream);
        DescribeVorticity(v0, dx, axis_y, axis_z, "init");
    }
    const selfcheck::FieldStats initial = selfcheck::ComputeFieldStats(solver, stream);
    printf("initial: KE=%.6e  max|u|=%.4f  max|w|=%.4f\n",
           initial.kinetic_energy, initial.max_speed, initial.max_vorticity);
    printf("rings: R=%.3f core=%.3f Gamma=%.3f at x=%.3f and x=%.3f, dx=%.5f (%.1f cells per radius)\n",
           radius, core, circulation, x_trail, x_lead, dx, radius / dx);

    // One reinitialization cycle is reinit_every advection steps followed by a single
    // reinitialization. Between reinitializations the solver's reported velocity is
    // stale -- init_u_ is only refreshed by ReinitAsync -- so diagnostics are sampled
    // at cycle boundaries. The loop counter stays in advection steps so that the
    // physical time axis is identical for every reinitialization interval.
    const int cycle_steps = config.reinit_every;
    int step              = 0;
    int next_diag         = 0;
    int diag_count        = 0;
    while (step < total_steps) {
        profiler.beginFrame();
        for (int i = 0; i < cycle_steps; i++)
            solver.AdvanceAsync(dt, stream);
        solver.ReinitAsync(dt, stream);
        cudaStreamSynchronize(stream);
        step += cycle_steps;

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error at step %d: %s\n", step, cudaGetErrorString(err));
            fclose(csv);
            return 1;
        }

        if (step < next_diag)
            continue;
        next_diag = step + diag_every;
        diag_count++;

        const selfcheck::FieldStats stats = selfcheck::ComputeFieldStats(solver, stream);
        if (!stats.finite) {
            printf("non-finite field at step %d -- aborting\n", step);
            fclose(csv);
            return 1;
        }

        const selfcheck::HostField vor = selfcheck::DownloadVorticityNorm(solver, stream);
        clusters = TrackTwoRings(vor, dx, axis_y, axis_z, clusters, 0.25f);

        // Once a ring nears the outflow wall it expands against it, which looks
        // like a radial excursion and would be miscounted as a leap.
        const double wall_guard = 0.9 * (vor.dim.x * dx);
        if (!merged && (clusters[0].x > wall_guard || clusters[1].x > wall_guard)) {
            merged     = true;
            merge_step = step;
            printf("stopped counting at step %d: rings reached the wall region (x > %.2f)\n", step, wall_guard);
        }

        double separation = 0.0;
        if (clusters[0].valid && clusters[1].valid) {
            const double dxc = clusters[0].x - clusters[1].x;
            const double drc = clusters[0].r - clusters[1].r;
            separation       = std::sqrt(dxc * dxc + drc * drc);

            if (std::fabs(dxc) > peak_abs_dx)
                peak_abs_dx = std::fabs(dxc);

            const int sign = (dxc > 0.0) ? 1 : -1;
            if (!merged) {
                if (prev_sign != 0 && sign != prev_sign && peak_abs_dx > separate_tol) {
                    leaps++;
                    peak_abs_dx = 0.0;
                }
                prev_sign = sign;
            }

            // A leap briefly puts both rings at the same x, but their radii differ
            // most at that instant. Only when the pair stays close in x AND r for
            // several consecutive samples have they actually merged.
            if (separation < merge_tol)
                consecutive_close++;
            else
                consecutive_close = 0;
            if (!merged && consecutive_close >= merge_samples && step > 0) {
                merged     = true;
                merge_step = step;
            }
        }

        fprintf(csv, "%d,%.5f,%.8e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
                step, step * dt, stats.kinetic_energy, stats.max_speed, stats.max_vorticity,
                clusters[0].x, clusters[0].r, clusters[1].x, clusters[1].r, separation, leaps);

        if (diag_count % 20 == 1)
            printf("step %5d  t=%6.3f  KE=%.4e  max|w|=%7.3f  A=(%.3f,%.3f) B=(%.3f,%.3f) sep=%.4f leaps=%d%s\n",
                   step, step * dt, stats.kinetic_energy, stats.max_vorticity,
                   clusters[0].x, clusters[0].r, clusters[1].x, clusters[1].r, separation, leaps,
                   merged ? " [merged]" : "");
    }

    fclose(csv);
    printf("\n=== 3D leapfrog vortex rings ===\n");
    printf("leaps counted before merger: %d\n", leaps);
    if (merged)
        printf("counting stopped at step %d (t=%.3f)\n", merge_step, merge_step * dt);
    else
        printf("no merger detected within %d steps\n", total_steps);
    printf("csv: %s\n", csv_path);
    return 0;
}

} // namespace

struct BurgersParams {
    float core        = 0.06f;   // b_w at t = 0, ~7.7 cells at 128^3
    float circulation = 0.05f;
    float viscosity   = 2.25e-4f;
};

// Verification of the source-term path integral against an exact solution.
//
// A columnar vortex carrying the Burgers profile (Tohidi et al. 2018, Eq. 7)
// with no imposed axial strain spreads by viscosity alone, and its core obeys
//
//     b(t)^2 = b(0)^2 + 4 nu t
//
// exactly. Because the peak vorticity of that profile is Gamma / (pi b^2) and
// circulation is conserved, the core ratio can be read straight off the peak
// vorticity without needing Gamma at all:
//
//     b(t)^2 / b(0)^2 = omega_max(0) / omega_max(t)
//
// so the viscosity the solver actually applied comes back as
//
//     nu_eff = b(0)^2 (omega_max(0)/omega_max(t) - 1) / (4 t)
//
// and the test is whether nu_eff recovers the nu that was asked for. Running the
// same case at nu = 0 measures the numerical dissipation floor in the same
// units, which is what nu_eff has to stand clear of for the result to mean
// anything. The radius of peak azimuthal velocity is reported alongside as an
// independent estimator, since Burgers puts it at r = 1.12091 b_w.
int RunBurgersViscous(int total_steps, int diag_every, const char* csv_path, BurgersParams bp,
                      float dt, int reinit_every, int rk_order, int res_tiles)
{
    cudaStream_t stream = 0;
    selfcheck::SolverConfig config;
    config.tile_dim     = { res_tiles, res_tiles, res_tiles }; // res_tiles*8 cubed, unit cube
    config.len_y        = 1.0f;
    config.cg_iter      = 15;
    config.reinit_every = reinit_every;
    config.rk_order     = rk_order;

    ofm::OFM solver;
    GPUTimer profiler(64);
    selfcheck::SetupSolver(solver, config, profiler, stream);

    solver.use_source_term_ = true;
    solver.viscosity_       = bp.viscosity;

    const float dx       = solver.dx_;
    const float centre_x = 0.5f;
    const float centre_y = 0.5f;

    const selfcheck::ColumnVortexSpec spec = { centre_x, centre_y, bp.core, bp.circulation };
    selfcheck::AddColumnVortexAsync(solver, spec, true, stream);
    cudaStreamSynchronize(stream);

    // Explicit diffusion is applied over the advection interval, which the
    // leapfrog branch of the cycle stretches to 2*dt. Warn rather than fail: the
    // run may still be usable, but the number should not be trusted silently.
    const float diffusion_number = bp.viscosity * 2.0f * dt / (dx * dx);
    printf("burgers: b0=%.4f (%.1f cells)  Gamma=%.3f  nu=%.3e  dt=%.5f  n=%d  rk=%d\n",
           bp.core, bp.core / dx, bp.circulation, bp.viscosity, dt, reinit_every, rk_order);
    printf("         explicit diffusion number nu*2dt/dx^2 = %.4f%s\n",
           diffusion_number, diffusion_number > 0.16f ? "  [WARNING: above the ~1/6 stability limit]" : "");

    const selfcheck::ColumnDiag d0 = selfcheck::MeasureColumnVortex(solver, centre_x, centre_y, stream);
    if (!d0.valid || d0.max_vorticity <= 0.0f) {
        printf("initial measurement failed\n");
        return 1;
    }
    printf("         seeded: max|w|=%.4f  r_peak=%.4f  r_peak/1.12091=%.4f (b0=%.4f)\n\n",
           d0.max_vorticity, d0.r_peak, d0.r_peak / 1.12091f, bp.core);

    FILE* csv = fopen(csv_path, "w");
    if (!csv) {
        printf("cannot open %s\n", csv_path);
        return 1;
    }
    fprintf(csv, "step,time,max_vorticity,r_peak,b_from_vorticity,b_from_r_peak,b_analytic,nu_eff,nu_rel_err\n");

    printf("%6s %8s %10s %10s %10s %10s %12s %10s\n",
           "step", "t", "max|w|", "b_vor", "b_rpeak", "b_exact", "nu_eff", "err");

    const int cycle_steps = config.reinit_every;
    int step              = 0;
    int next_diag         = 0;
    double last_rel_err   = 0.0;
    double last_nu_eff    = 0.0;

    while (step < total_steps) {
        profiler.beginFrame();
        for (int i = 0; i < cycle_steps; i++)
            solver.AdvanceAsync(dt, stream);
        solver.ReinitAsync(dt, stream);
        cudaStreamSynchronize(stream);
        step += cycle_steps;

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error at step %d: %s\n", step, cudaGetErrorString(err));
            fclose(csv);
            return 1;
        }

        if (step < next_diag)
            continue;
        next_diag = step + diag_every;

        const selfcheck::ColumnDiag d = selfcheck::MeasureColumnVortex(solver, centre_x, centre_y, stream);
        if (!d.valid || d.max_vorticity <= 0.0f) {
            printf("non-finite field at step %d -- aborting\n", step);
            fclose(csv);
            return 1;
        }

        const double t          = step * static_cast<double>(dt);
        const double b0_sq      = static_cast<double>(bp.core) * bp.core;
        const double b_vor_sq   = b0_sq * d0.max_vorticity / d.max_vorticity;
        const double b_vor      = std::sqrt(b_vor_sq);
        const double b_rpeak    = d.r_peak / 1.12091;
        const double b_exact    = std::sqrt(b0_sq + 4.0 * bp.viscosity * t);
        const double nu_eff     = (b_vor_sq - b0_sq) / (4.0 * t);
        const double rel_err    = (bp.viscosity > 0.0f) ? (nu_eff / bp.viscosity - 1.0) : 0.0;
        last_rel_err            = rel_err;
        last_nu_eff             = nu_eff;

        fprintf(csv, "%d,%.5f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6f\n",
                step, t, d.max_vorticity, d.r_peak, b_vor, b_rpeak, b_exact, nu_eff, rel_err);
        printf("%6d %8.3f %10.4f %10.5f %10.5f %10.5f %12.4e %9.2f%%\n",
               step, t, d.max_vorticity, b_vor, b_rpeak, b_exact, nu_eff, 100.0 * rel_err);
    }

    fclose(csv);
    printf("\n=== Burgers viscous channel ===\n");
    if (bp.viscosity > 0.0f) {
        printf("nu asked for : %.6e\n", bp.viscosity);
        printf("nu recovered : %.6e   (%.2f%% relative error at the last sample)\n", last_nu_eff, 100.0 * last_rel_err);
    } else {
        printf("nu = 0 run: the recovered %.6e is the numerical dissipation floor,\n", last_nu_eff);
        printf("expressed as an equivalent viscosity. A viscous run is only meaningful\n");
        printf("well above this number.\n");
    }
    printf("csv: %s\n", csv_path);
    return 0;
}

// D1: circulation budget and attribution, checked two ways against an exact solution.
//
// For a material loop C, the flow-map solution gives the circulation directly:
//
//     Gamma(t) = closed_integral_C u.dl = closed_integral_C0 [ m0 + sum_k sum_i dt F^T s_k ] . dX
//              = Gamma(0) + sum_k dGamma_k
//
// because u = m - grad(phi) and the gradient integrates to zero around a loop.
// So each source channel's accumulator, integrated around the loop's preimage,
// IS that channel's contribution to the circulation. That separation is what an
// Eulerian solver cannot do and the flow map gives for free.
//
// The test case is the diffusing columnar Burgers vortex again, for which every
// leg of that identity is known in closed form. Its radial velocity is zero, so a
// circle of fixed radius is a material loop and stays the loop's own preimage at
// every cycle start -- which is exactly the frame the accumulators live in.
//
//     Gamma(r,t) = Gamma_inf * (1 - exp(-r^2 / b(t)^2)),   b(t)^2 = b0^2 + 4 nu t
//
// Three quantities are compared, all as changes since t = 0 so that the
// discretisation of the seeded field cancels:
//
//     direct    measured closed_integral u.dl on the loop
//     budget    sum over cycles and channels of the accumulator line integrals
//     analytic  the expression above
//
// budget vs direct is the dual-path check; budget vs analytic is the attribution
// error the plan puts a 1% bound on.
int RunAttribution(int total_steps, int diag_every, const char* csv_path, BurgersParams bp,
                   float dt, int reinit_every, int rk_order, int res_tiles, float loop_radius)
{
    cudaStream_t stream = 0;
    selfcheck::SolverConfig config;
    config.tile_dim     = { res_tiles, res_tiles, res_tiles };
    config.len_y        = 1.0f;
    config.cg_iter      = 15;
    config.reinit_every = reinit_every;
    config.rk_order     = rk_order;

    ofm::OFM solver;
    GPUTimer profiler(64);
    selfcheck::SetupSolver(solver, config, profiler, stream);

    solver.use_source_term_  = true;
    solver.viscosity_        = bp.viscosity;
    solver.track_attribution_ = true;

    const float dx       = solver.dx_;
    const float centre_x = 0.5f;
    const float centre_y = 0.5f;
    const float r_loop   = (loop_radius > 0.0f) ? loop_radius : bp.core;
    const int samples    = 512;

    const selfcheck::ColumnVortexSpec spec = { centre_x, centre_y, bp.core, bp.circulation };
    selfcheck::AddColumnVortexAsync(solver, spec, true, stream);
    cudaStreamSynchronize(stream);

    const double b0_sq = static_cast<double>(bp.core) * bp.core;
    auto gamma_exact = [&](double t) {
        const double b_sq = b0_sq + 4.0 * bp.viscosity * t;
        return bp.circulation * (1.0 - std::exp(-static_cast<double>(r_loop) * r_loop / b_sq));
    };

    const double gamma0 = selfcheck::CirculationOnCircle(
        solver, *solver.init_u_x_, *solver.init_u_y_, *solver.init_u_z_,
        centre_x, centre_y, r_loop, samples, stream);

    printf("attribution: grid %d^3  b0=%.4f (%.1f cells)  Gamma=%.4f  nu=%.3e  n=%d\n",
           res_tiles * 8, bp.core, bp.core / dx, bp.circulation, bp.viscosity, reinit_every);
    printf("             loop radius %.4f (%.2f b0)   seeded Gamma(loop) = %.6f  (exact %.6f)\n\n",
           r_loop, r_loop / bp.core, gamma0, gamma_exact(0.0));

    FILE* csv = fopen(csv_path, "w");
    if (!csv) {
        printf("cannot open %s\n", csv_path);
        return 1;
    }
    fprintf(csv, "step,time,gamma_direct,gamma_budget,gamma_exact,d_direct,d_budget,d_exact,"
                 "g_viscous,g_external,dual_path_err,attribution_err\n");

    printf("%6s %8s %11s %11s %11s %11s %11s\n",
           "step", "t", "dG direct", "dG budget", "dG exact", "dual-path", "attrib");

    double acc_gamma[ofm::kChanNum] = { 0.0, 0.0 };
    const int cycle_steps = config.reinit_every;
    int step = 0, next_diag = 0;
    double last_dual = 0.0, last_attr = 0.0;

    while (step < total_steps) {
        profiler.beginFrame();
        for (int i = 0; i < cycle_steps; i++)
            solver.AdvanceAsync(dt, stream);
        solver.ReinitAsync(dt, stream);
        cudaStreamSynchronize(stream);
        step += cycle_steps;

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error at step %d: %s\n", step, cudaGetErrorString(err));
            fclose(csv);
            return 1;
        }

        // The accumulators are cleared at every reinitialization, so their line
        // integrals have to be taken every cycle, not only on diagnostic steps.
        for (int k = 0; k < ofm::kChanNum; k++)
            acc_gamma[k] += selfcheck::CirculationOnCircle(
                solver, *solver.acc_x_[k], *solver.acc_y_[k], *solver.acc_z_[k],
                centre_x, centre_y, r_loop, samples, stream);

        if (step < next_diag)
            continue;
        next_diag = step + diag_every;

        const double t        = step * static_cast<double>(dt);
        const double direct   = selfcheck::CirculationOnCircle(
            solver, *solver.init_u_x_, *solver.init_u_y_, *solver.init_u_z_,
            centre_x, centre_y, r_loop, samples, stream);
        const double budget_d = acc_gamma[ofm::kChanViscous] + acc_gamma[ofm::kChanExternal];
        const double exact_d  = gamma_exact(t) - gamma_exact(0.0);

        const double d_direct = direct - gamma0;
        const double scale    = std::fabs(exact_d) > 1e-12 ? std::fabs(exact_d) : 1.0;
        const double dual_err = (budget_d - d_direct) / scale;
        const double attr_err = (budget_d - exact_d) / scale;
        last_dual = dual_err;
        last_attr = attr_err;

        fprintf(csv, "%d,%.5f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.6f,%.6f\n",
                step, t, direct, gamma0 + budget_d, gamma_exact(t),
                d_direct, budget_d, exact_d,
                acc_gamma[ofm::kChanViscous], acc_gamma[ofm::kChanExternal], dual_err, attr_err);
        printf("%6d %8.3f %11.6f %11.6f %11.6f %10.2f%% %10.2f%%\n",
               step, t, d_direct, budget_d, exact_d, 100.0 * dual_err, 100.0 * attr_err);
    }

    fclose(csv);
    printf("\n=== D1 circulation budget ===\n");
    printf("channel split at the last sample:  viscous %.8f   external %.8f\n",
           acc_gamma[ofm::kChanViscous], acc_gamma[ofm::kChanExternal]);
    printf("dual-path (budget vs direct)    : %.2f%%\n", 100.0 * last_dual);
    printf("attribution (budget vs analytic): %.2f%%\n", 100.0 * last_attr);
    printf("csv: %s\n", csv_path);
    return 0;
}

// D1, second half: stretching and tilting reported separately.
//
// The fire whirl and VLS literature argues that vertical vorticity comes from
// tilting ambient horizontal vorticity rather than from stretching, but nobody has
// measured the two terms against each other. Splitting them is a pointwise
// diagnostic of the vertical vorticity equation,
//
//     D w_z / Dt = w_x d_x w + w_y d_y w  +  w_z d_z w
//                  \_____tilting_____/       \_stretching_/
//
// and this calibrates the operator rather than the solver: an analytic field goes
// in, no time stepping happens, and what comes out is compared with the closed-form
// answer. The field is linear, so central differences are exact on it and anything
// beyond round-off is a bug in the operator, not truncation error.
int RunTiltingStretching(int res_tiles, float core, float circulation)
{
    cudaStream_t stream = 0;
    selfcheck::SolverConfig config;
    config.tile_dim = { res_tiles, res_tiles, res_tiles };
    config.len_y    = 1.0f;
    config.cg_iter  = 15;

    ofm::OFM solver;
    GPUTimer profiler(64);
    selfcheck::SetupSolver(solver, config, profiler, stream);

    printf("tilting/stretching operator check, grid %d^3\n\n", res_tiles * 8);

    // --- case 1: analytic shear field, both terms non-zero and uniform ---
    const selfcheck::ShearSpec spec = { 1.0f, 0.7f, 0.5f, 0.3f, 0.0f };
    const double tilt_exact    = static_cast<double>(spec.c) * spec.s;
    const double stretch_exact = 2.0 * static_cast<double>(spec.omega) * spec.g;

    selfcheck::SetShearFieldAsync(solver, spec, stream);
    cudaStreamSynchronize(stream);
    const selfcheck::TiltStretch a = selfcheck::MeasureTiltingStretching(solver, stream);
    if (!a.valid) {
        printf("case 1: non-finite result\n");
        return 1;
    }

    const double tilt_err    = (a.tilting_mean - tilt_exact) / tilt_exact;
    const double stretch_err = (a.stretching_mean - stretch_exact) / stretch_exact;

    printf("case 1 -- analytic shear field (omega=%.2f c=%.2f s=%.2f g=%.2f)\n",
           spec.omega, spec.c, spec.s, spec.g);
    printf("  tilting     measured %+.8f   exact %+.8f   rel err %+.3e\n",
           a.tilting_mean, tilt_exact, tilt_err);
    printf("  stretching  measured %+.8f   exact %+.8f   rel err %+.3e\n",
           a.stretching_mean, stretch_exact, stretch_err);
    printf("  tilting share of the two: %.1f%%  (exact %.1f%%)\n\n",
           100.0 * a.ratio, 100.0 * tilt_exact / (tilt_exact + stretch_exact));

    // --- case 2: a purely columnar vortex has no vertical velocity at all, so
    //     both terms must vanish; this catches spurious coupling between them ---
    solver.init_u_x_->ClearDevAsync(stream);
    solver.init_u_y_->ClearDevAsync(stream);
    solver.init_u_z_->ClearDevAsync(stream);
    const selfcheck::ColumnVortexSpec col = { 0.5f, 0.5f, core, circulation };
    selfcheck::AddColumnVortexAsync(solver, col, true, stream);
    cudaStreamSynchronize(stream);
    const selfcheck::TiltStretch b = selfcheck::MeasureTiltingStretching(solver, stream);
    if (!b.valid) {
        printf("case 2: non-finite result\n");
        return 1;
    }
    printf("case 2 -- columnar vortex, w = 0 so both terms must vanish\n");
    printf("  mean |tilting|    %.3e\n", b.tilting_abs_mean);
    printf("  mean |stretching| %.3e\n\n", b.stretching_abs_mean);

    const bool pass = std::fabs(tilt_err) < 1e-4 && std::fabs(stretch_err) < 1e-4;
    printf("=== D1 tilting/stretching ===\n");
    printf("%s: the split reproduces the analytic terms to %.1e and %.1e\n",
           pass ? "PASS" : "FAIL", std::fabs(tilt_err), std::fabs(stretch_err));
    return pass ? 0 : 1;
}

int main(int argc, char** argv)
{
    std::string test        = "leapfrog3d";
    int total_steps         = 2000; // LFM Table 4: 28.2 s total at 14.1 ms/step
    int diag_every          = 10;
    std::string csv_path    = "leapfrog3d.csv";
    RingParams rp;
    BurgersParams bp;
    bool dt_set      = false;
    float dt         = 1.0f / 60.0f;
    int reinit_every = 1; // 1 = one-step (OFM); LFM runs its Figure 14 at 10
    int res_tiles    = 16; // burgers only: 16 tiles per side = 128^3
    float loop_radius = 0.0f; // attribution only: 0 means use the core radius
    int rk_order     = 3; // TVD-RK3, the order OFM shipped with

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--test" && i + 1 < argc)
            test = argv[++i];
        else if (arg == "--steps" && i + 1 < argc)
            total_steps = std::atoi(argv[++i]);
        else if (arg == "--diag-every" && i + 1 < argc)
            diag_every = std::atoi(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc)
            csv_path = argv[++i];
        else if (arg == "--radius" && i + 1 < argc)
            rp.radius = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--core" && i + 1 < argc) {
            rp.core = static_cast<float>(std::atof(argv[++i]));
            bp.core = rp.core;
        }
        else if (arg == "--circulation" && i + 1 < argc) {
            rp.circulation = static_cast<float>(std::atof(argv[++i]));
            bp.circulation = rp.circulation;
        }
        else if (arg == "--spacing" && i + 1 < argc)
            rp.spacing = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--reinit-every" && i + 1 < argc)
            reinit_every = std::atoi(argv[++i]);
        else if (arg == "--rk-order" && i + 1 < argc)
            rk_order = std::atoi(argv[++i]);
        else if (arg == "--res-tiles" && i + 1 < argc)
            res_tiles = std::atoi(argv[++i]);
        else if (arg == "--loop-radius" && i + 1 < argc)
            loop_radius = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--nu" && i + 1 < argc)
            bp.viscosity = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--dt" && i + 1 < argc) {
            dt     = static_cast<float>(std::atof(argv[++i]));
            dt_set = true;
        }
        else if (arg == "--help") {
            printf("usage: selfcheck [--test leapfrog3d] [--steps N] [--diag-every N] [--csv PATH]\n"
                   "                 [--radius R] [--core S] [--circulation G] [--spacing D]\n"
                   "                 [--reinit-every N] [--rk-order 2|3|4] [--dt DT]\n"
                   "       selfcheck --test burgers [--nu NU] [--core B0] [--circulation G]\n"
                   "                 [--steps N] [--diag-every N] [--reinit-every N] [--rk-order 2|3|4]\n"
                   "                 [--res-tiles T]  (T tiles per side, 8T cells; default 16 = 128^3)\n"
                   "       selfcheck --test attribution [--nu NU] [--loop-radius R] [--reinit-every N]\n"
                   "                 [--res-tiles T] [--steps N] [--diag-every N]\n"
                   "       selfcheck --test tilting [--res-tiles T]\n");
            return 0;
        }
    }

    if (test == "leapfrog3d")
        return RunLeapfrogRings(total_steps, diag_every, csv_path.c_str(), rp, dt, reinit_every, rk_order);

    if (test == "tilting")
        return RunTiltingStretching(res_tiles, bp.core, bp.circulation);

    if (test == "attribution") {
        if (!dt_set)
            dt = 1.0f / 120.0f;
        if (csv_path == "leapfrog3d.csv")
            csv_path = "attribution.csv";
        return RunAttribution(total_steps, diag_every, csv_path.c_str(), bp, dt, reinit_every, rk_order, res_tiles, loop_radius);
    }

    if (test == "burgers") {
        // The viscous case needs a finer step than the ring case to keep explicit
        // diffusion stable; only override when the user did not ask for one.
        if (!dt_set)
            dt = 1.0f / 120.0f;
        if (csv_path == "leapfrog3d.csv")
            csv_path = "burgers.csv";
        return RunBurgersViscous(total_steps, diag_every, csv_path.c_str(), bp, dt, reinit_every, rk_order, res_tiles);
    }

    printf("unknown test: %s\n", test.c_str());
    return 1;
}
