// D1 solver self-check suite.
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

// D2: circulation budget and attribution, checked two ways against an exact solution.
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
    printf("\n=== D2 circulation budget ===\n");
    printf("channel split at the last sample:  viscous %.8f   external %.8f\n",
           acc_gamma[ofm::kChanViscous], acc_gamma[ofm::kChanExternal]);
    printf("dual-path (budget vs direct)    : %.2f%%\n", 100.0 * last_dual);
    printf("attribution (budget vs analytic): %.2f%%\n", 100.0 * last_attr);
    printf("csv: %s\n", csv_path);
    return 0;
}

// D2, second half: stretching and tilting reported separately.
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
    printf("=== D2 tilting/stretching ===\n");
    printf("%s: the split reproduces the analytic terms to %.1e and %.1e\n",
           pass ? "PASS" : "FAIL", std::fabs(tilt_err), std::fabs(stretch_err));
    return pass ? 0 : 1;
}

// D3: the Burgers core-radius fitter, and the three-radius ordering.
//
// Tohidi et al. 2018 report that the Burgers model is the best fit for a
// quasi-steady on-source fire whirl, that the azimuthal velocity of that model
// peaks at r = 1.12091 b_w (Sec. 4.1, Eq. 7), and that three core radii can be
// defined -- b_A from the axial velocity, b_T from the excess temperature, b_w
// from the azimuthal velocity -- which satisfy b_A > b_T > b_w throughout the
// height of a fire whirl (Sec. 4.2, Fig. 7, after Lei et al. 2015b).
//
// This calibrates the estimators, not the solver: fields with known radii go in,
// no time stepping happens, and what comes out is compared with the closed form.
int RunCoreRadii(int res_tiles, float core, float circulation, const char* csv_path)
{
    cudaStream_t stream = 0;
    selfcheck::SolverConfig config;
    config.tile_dim = { res_tiles, res_tiles, res_tiles };
    config.len_y    = 1.0f;
    config.cg_iter  = 15;

    ofm::OFM solver;
    GPUTimer profiler(64);
    selfcheck::SetupSolver(solver, config, profiler, stream);

    const float dx       = solver.dx_;
    const float centre_x = 0.5f;
    const float centre_y = 0.5f;
    const float r_max    = 0.45f;

    const double peak_const = selfcheck::BurgersPeakConstant();

    printf("core-radius estimators, grid %d^3  (dx = %.6f)\n", res_tiles * 8, dx);
    printf("  peak constant from the model root exp(-x)(2x+1) = 1: %.6f"
           "   (Tohidi et al. 2018 quote 1.12091)\n\n", peak_const);

    // --- case 1: the fitter alone, on an exactly sampled analytic profile ---
    //
    // Same radial bins the grid produces, but the values are the closed form, so
    // the only error left is the fit itself.
    selfcheck::RadialProfile exact;
    {
        const double pi = 3.14159265358979323846;
        const int bins  = static_cast<int>(r_max / dx);
        for (int b = 0; b < bins; b++) {
            const double r = (b + 0.5) * dx;
            exact.r.push_back(r);
            exact.v.push_back(circulation / (2.0 * pi * r) * (1.0 - std::exp(-r * r / (static_cast<double>(core) * core))));
        }
    }
    const selfcheck::BurgersFit f1 = selfcheck::FitBurgers(exact);
    const double e1_b = (f1.b_w - core) / core;
    const double e1_g = (f1.gamma_inf - circulation) / circulation;

    printf("case 1 -- fitter on the analytic profile (b_w = %.4f, Gamma_inf = %.4f)\n", core, circulation);
    printf("  b_w          fitted %.8f   exact %.8f   rel err %+.3e\n", f1.b_w, core, e1_b);
    printf("  Gamma_inf    fitted %.8f   exact %.8f   rel err %+.3e\n", f1.gamma_inf, circulation, e1_g);
    printf("  r_peak / b_w        %.6f   model %.6f   rel err %+.3e\n",
           f1.ratio, peak_const, (f1.ratio - peak_const) / peak_const);
    printf("  residual RMS / peak %.3e   (%d golden-section iterations)\n\n", f1.rms_rel, f1.iters);

    // --- case 2: the whole pipeline, on the field the solver actually holds ---
    solver.init_u_x_->ClearDevAsync(stream);
    solver.init_u_y_->ClearDevAsync(stream);
    solver.init_u_z_->ClearDevAsync(stream);
    const selfcheck::ColumnVortexSpec col = { centre_x, centre_y, core, circulation };
    selfcheck::AddColumnVortexAsync(solver, col, true, stream);
    cudaStreamSynchronize(stream);

    const selfcheck::RadialProfile p2 = selfcheck::MeasureRadialProfile(solver, selfcheck::kProfileAzimuthal, centre_x, centre_y, r_max, stream);
    const selfcheck::BurgersFit f2 = selfcheck::FitBurgers(p2);
    if (!f2.ok) {
        printf("case 2: fit failed\n");
        return 1;
    }
    const double e2_b = (f2.b_w - core) / core;
    const double e2_r = (f2.ratio - peak_const) / peak_const;

    printf("case 2 -- seeded and projected on the grid, fitted from U_theta(r)\n");
    printf("  b_w          fitted %.8f   seeded %.8f   rel err %+.3e   (%.1f cells)\n",
           f2.b_w, core, e2_b, core / dx);
    printf("  Gamma_inf    fitted %.8f   seeded %.8f   rel err %+.3e\n",
           f2.gamma_inf, circulation, (f2.gamma_inf - circulation) / circulation);
    printf("  r_peak / b_w        %.6f   model %.6f   rel err %+.3e\n", f2.ratio, peak_const, e2_r);
    printf("  residual RMS / peak %.3e\n\n", f2.rms_rel);

    // --- case 3: the ordering b_A > b_T > b_w ---
    //
    // b_A comes from a Gaussian axial jet and b_T from an analytic excess
    // temperature binned the same way; the solver carries no temperature field
    // until the low-Mach extension, so that leg exercises the estimator rather
    // than a simulated field. Scales are chosen so the ordering is strict.
    const double scale_a = 2.6 * core; // axial velocity Gaussian scale
    const double scale_t = 1.8 * core; // excess temperature Gaussian scale
    const double sqrt_ln2 = std::sqrt(std::log(2.0));

    solver.init_u_x_->ClearDevAsync(stream);
    solver.init_u_y_->ClearDevAsync(stream);
    solver.init_u_z_->ClearDevAsync(stream);
    selfcheck::AddColumnVortexAsync(solver, col, false, stream); // analytic, not projected
    const selfcheck::AxialJetSpec jet = { centre_x, centre_y, static_cast<float>(scale_a), 1.0f };
    selfcheck::SetAxialJetAsync(solver, jet, stream);
    cudaStreamSynchronize(stream);

    const selfcheck::RadialProfile pw = selfcheck::MeasureRadialProfile(solver, selfcheck::kProfileAzimuthal, centre_x, centre_y, r_max, stream);
    const selfcheck::RadialProfile pa = selfcheck::MeasureRadialProfile(solver, selfcheck::kProfileAxial, centre_x, centre_y, r_max, stream);
    const selfcheck::RadialProfile pt = selfcheck::GaussianProfileOnGrid(solver, centre_x, centre_y, r_max, 1.0, scale_t);

    const selfcheck::BurgersFit f3 = selfcheck::FitBurgers(pw);
    const double b_w_m  = f3.b_w;
    const double b_a_m  = selfcheck::RadiusAtFraction(pa, 0.5);
    const double b_t_m  = selfcheck::RadiusAtFraction(pt, 0.5);
    const double b_a_fx = selfcheck::RadiusFromFluxes(pa);

    const double b_a_exact = scale_a * sqrt_ln2;
    const double b_t_exact = scale_t * sqrt_ln2;

    printf("case 3 -- the three radii, and their ordering\n");
    printf("  b_A  half-max of U_z     %.8f   exact %.8f   rel err %+.3e\n",
           b_a_m, b_a_exact, (b_a_m - b_a_exact) / b_a_exact);
    printf("  b_A  flux form Q/sqrt(M) %.8f   exact %.8f   rel err %+.3e\n",
           b_a_fx, scale_a, (b_a_fx - scale_a) / scale_a);
    printf("  b_T  half-max of dT      %.8f   exact %.8f   rel err %+.3e\n",
           b_t_m, b_t_exact, (b_t_m - b_t_exact) / b_t_exact);
    printf("  b_w  Burgers fit         %.8f   exact %.8f   rel err %+.3e\n",
           b_w_m, static_cast<double>(core), (b_w_m - core) / core);
    const bool ordered = (b_a_m > b_t_m) && (b_t_m > b_w_m);
    printf("  ordering b_A > b_T > b_w : %s   (%.4f > %.4f > %.4f)\n\n",
           ordered ? "holds" : "FAILS", b_a_m, b_t_m, b_w_m);

    if (csv_path) {
        FILE* csv = fopen(csv_path, "w");
        if (csv) {
            fprintf(csv, "r,u_theta,u_z,dT,burgers_fit\n");
            const double pi = 3.14159265358979323846;
            for (size_t i = 0; i < pw.r.size(); i++) {
                const double r   = pw.r[i];
                const double fit = f3.gamma_inf / (2.0 * pi * r) * (1.0 - std::exp(-r * r / (f3.b_w * f3.b_w)));
                fprintf(csv, "%.8f,%.8f,%.8f,%.8f,%.8f\n",
                        r, pw.v[i], i < pa.v.size() ? pa.v[i] : 0.0, i < pt.v.size() ? pt.v[i] : 0.0, fit);
            }
            fclose(csv);
            printf("csv: %s\n", csv_path);
        }
    }

    // The criterion is Tohidi's: the peak sits at 1.12091 b_w, and the ordering
    // holds. 1% is the tolerance the rest of Stage 0 is held to.
    const bool pass = f2.ok && std::fabs(e2_r) < 0.01 && std::fabs(e2_b) < 0.01 && ordered;
    printf("=== D3 core radii ===\n");
    printf("%s: r_peak/b_w = %.5f (model %.5f, %+.2f%%), b_w to %+.2f%%, ordering %s\n",
           pass ? "PASS" : "FAIL", f2.ratio, peak_const, 100.0 * e2_r, 100.0 * e2_b,
           ordered ? "holds" : "fails");
    return pass ? 0 : 1;
}

// D4: the vortex-flame Damkohler number.
//
// Linan, Vera & Sanchez 2015 (Sec. 7) define the strain a vortex imposes on a
// flame as A_Gamma = Gamma / (2 r0^2) and the vortex Damkohler number as
// Da_Gamma = A_e / A_Gamma, the ratio of the vortex turnover time to the chemical
// time, and state that local extinction is to be expected for Da_Gamma <~ 1.
// This is the self-check on the prescribed-heat-source assumption: where
// Da_Gamma falls below one, that assumption has no support.
//
// Evaluated pointwise by taking Gamma and r0 at the same radius. On an analytic
// Burgers vortex the Da_Gamma = 1 contour is a circle whose radius solves
// (1 - exp(-x))/x = 2 A_e b_w^2 / Gamma_inf with x = (r/b_w)^2, which is what the
// measured contour is compared against.
int RunDamkohler(int res_tiles, float core, float circulation, const char* csv_path)
{
    cudaStream_t stream = 0;
    selfcheck::SolverConfig config;
    config.tile_dim = { res_tiles, res_tiles, res_tiles };
    config.len_y    = 1.0f;
    config.cg_iter  = 15;

    ofm::OFM solver;
    GPUTimer profiler(64);
    selfcheck::SetupSolver(solver, config, profiler, stream);

    const float dx       = solver.dx_;
    const float centre_x = 0.5f;
    const float centre_y = 0.5f;

    const selfcheck::ColumnVortexSpec col = { centre_x, centre_y, core, circulation };
    selfcheck::AddColumnVortexAsync(solver, col, true, stream);
    cudaStreamSynchronize(stream);

    printf("vortex-flame Damkohler number, grid %d^3  (dx = %.6f)\n", res_tiles * 8, dx);
    printf("  Burgers column: b_w = %.4f (%.1f cells), Gamma_inf = %.4f\n",
           core, core / dx, circulation);
    printf("  A_Gamma = Gamma(r) / (2 r^2),  Da_Gamma = A_e / A_Gamma"
           "   (Linan, Vera & Sanchez 2015, Sec. 7)\n\n");

    // Radii the contour is searched over. The sweep starts one cell out, where
    // the loop integral first has cells to interpolate from.
    const double r_min = 2.0 * dx;
    const double r_max = 0.40;
    const int radius_samples = 400;
    const int loop_samples   = 512;

    // Extinction rates chosen so the analytic contour lands at a known multiple
    // of the core radius, plus one case with no contour at all.
    const double targets[] = { 0.25, 1.0, 2.25, 4.0 };
    const int case_num = 4;

    FILE* csv = csv_path ? fopen(csv_path, "w") : nullptr;
    if (csv)
        fprintf(csv, "case,a_extinction,r_exact,r_measured,rel_err\n");

    printf("%10s %12s %12s %12s %10s\n", "r*/b_w", "A_e", "r* exact", "r* measured", "rel err");
    double worst = 0.0;
    bool all_found = true;
    for (int c = 0; c < case_num; c++) {
        const double x   = targets[c];
        const double a_e = selfcheck::ExtinctionRateForContour(circulation, core, x);
        const double r_exact = selfcheck::BurgersDamkohlerContour(circulation, core, a_e);

        const selfcheck::DamkohlerCurve curve =
            selfcheck::MeasureDamkohler(solver, centre_x, centre_y, a_e, r_min, r_max, radius_samples, loop_samples, stream);
        if (!curve.valid) {
            printf("case %d: non-finite result\n", c);
            return 1;
        }
        if (curve.r_contour < 0.0) {
            printf("%10.2f %12.4f %12.6f %12s %10s\n", std::sqrt(x), a_e, r_exact, "none", "-");
            all_found = false;
            continue;
        }
        const double err = (curve.r_contour - r_exact) / r_exact;
        if (std::fabs(err) > worst)
            worst = std::fabs(err);
        printf("%10.2f %12.4f %12.6f %12.6f %+9.3e\n", std::sqrt(x), a_e, r_exact, curve.r_contour, err);
        if (csv)
            fprintf(csv, "%d,%.8f,%.8f,%.8f,%.6e\n", c, a_e, r_exact, curve.r_contour, err);
    }

    // A_Gamma is largest on the axis, so Da_Gamma is smallest there and the
    // extinction region is the core. Raising A_e past Gamma_inf / (2 b_w^2) lifts
    // the whole curve above one and there is no such region at all.
    const double a_none = 1.2 * circulation / (2.0 * static_cast<double>(core) * core);
    const double r_none_exact = selfcheck::BurgersDamkohlerContour(circulation, core, a_none);
    const selfcheck::DamkohlerCurve none =
        selfcheck::MeasureDamkohler(solver, centre_x, centre_y, a_none, r_min, r_max, radius_samples, loop_samples, stream);
    printf("\nno-contour case: A_e = %.4f gives kappa = 1.2 > 1\n", a_none);
    printf("  analytic: %s     measured: %s\n",
           r_none_exact < 0.0 ? "no contour" : "contour",
           none.r_contour < 0.0 ? "no contour" : "contour");
    const bool none_ok = (r_none_exact < 0.0) && (none.r_contour < 0.0);

    if (csv) {
        fclose(csv);
        printf("csv: %s\n", csv_path);
    }

    const bool pass = all_found && none_ok && worst < 0.01;
    printf("\n=== D4 vortex-flame Damkohler ===\n");
    printf("%s: the Da_Gamma = 1 contour matches the analytic radius to %.2f%% over "
           "r*/b_w = 0.5 to 2.0%s\n",
           pass ? "PASS" : "FAIL", 100.0 * worst,
           none_ok ? ", and the no-contour regime is reported correctly" : "");
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Stage A: buoyant plume from a surface heat source in a sheared cross flow,
// after Cunningham et al. 2005. See harness.h for what the Boussinesq reduction
// of their compressible model drops, and RESULTS.md for the boundary-condition
// deviation. This is the first cut: it establishes whether the counter-rotating
// vortex pair forms at all and how the split width moves with z0 and Q0.
struct PlumeRun {
    int3 tiles       = { 23, 15, 19 }; // 184 x 120 x 152 cells; dx = 10 m
    float len_y      = 1200.0f;        // m, fixes dx = len_y / (8*tiles.y)
    float z0         = 100.0f;         // m
    float q0         = 1000.0f;        // W/m^3
    float mu         = 4.0f;           // kg/(m s); nu = mu/rho
    float dt         = 0.25f;          // s
    int steps        = 2400;           // 600 s, the paper's quasi-steady time
    int diag_every   = 240;
    float plane_x    = 1750.0f;        // m, the paper's cross-section plane
    float cvp_z      = 25.0f;          // m, height at which the CVP is measured
    int reinit_every = 1;
    int rk_order     = 3;
    bool theta_bfecc = true;  // error-compensate the theta advection
    bool theta_clamp = true;  // and clamp it, as the solver does for velocity
};

int RunPlume(const PlumeRun& run, const char* csv_path)
{
    cudaStream_t stream = 0;
    selfcheck::SolverConfig config;
    config.tile_dim     = run.tiles;
    config.len_y        = run.len_y;
    config.cg_iter      = 15;
    config.reinit_every = run.reinit_every;
    config.rk_order     = run.rk_order;

    ofm::OFM solver;
    GPUTimer profiler(64);
    selfcheck::SetupSolver(solver, config, profiler, stream);

    selfcheck::PlumeSpec spec;
    spec.z0 = run.z0;
    spec.q0 = run.q0;

    solver.use_source_term_ = true;
    solver.viscosity_       = run.mu / spec.rho;

    const float dx = solver.dx_;
    const int3 td  = solver.tile_dim_;
    const float lx = td.x * 8 * dx, ly = td.y * 8 * dx, lz = td.z * 8 * dx;

    printf("plume: %d x %d x %d cells, dx = %.2f m, domain %.0f x %.0f x %.0f m\n",
           td.x * 8, td.y * 8, td.z * 8, dx, lx, ly, lz);
    printf("       U0 = %.2f m/s, z0 = %.0f m, Q0 = %.0f W/m^3, mu = %.4f kg/(m s) -> nu = %.4f m^2/s\n",
           spec.u0, spec.z0, spec.q0, run.mu, solver.viscosity_);
    printf("       dt = %.3f s, %d steps = %.0f s, n = %d, theta advection = %s\n",
           run.dt, run.steps, run.steps * run.dt, run.reinit_every,
           run.theta_bfecc ? (run.theta_clamp ? "BFECC + clamp" : "BFECC") : "plain semi-Lagrangian");
    const float diffusion_number = solver.viscosity_ * 2.0f * run.dt / (dx * dx);
    const float cfl              = spec.u0 * run.dt / dx;
    printf("       explicit diffusion number = %.4f%s, cross-flow CFL = %.3f\n",
           diffusion_number, diffusion_number > 0.16f ? "  [WARNING: above the ~1/6 limit]" : "", cfl);
    // Diagnostics read init_u_, which ReinitAsync leaves current only at the end
    // of a cycle. Anywhere else in the cycle it still holds the cycle's start.
    if (run.diag_every % run.reinit_every != 0) {
        printf("       [ERROR: --diag-every %d is not a multiple of n = %d; init_u_ is only\n"
               "        current at a cycle boundary, so the measurement would read a stale field.]\n",
               run.diag_every, run.reinit_every);
        return 1;
    }
    if (run.steps % run.reinit_every != 0)
        printf("       [WARNING: %d steps is not a whole number of n = %d cycles; the last partial\n"
               "        cycle is never reinitialized and is dropped from the diagnostics.]\n",
               run.steps, run.reinit_every);

    selfcheck::SetPlumeInitialVelocityAsync(solver, spec, stream);
    selfcheck::SetPlumeBcAsync(solver, spec, stream);
    selfcheck::ProjectCurrentVelocityAsync(solver, stream);
    cudaStreamSynchronize(stream);

    const int cell_num = ofm::Prod(td) * 512;
    ofm::DHMemory<float> theta_a(cell_num);
    ofm::DHMemory<float> theta_b(cell_num);
    // Scratch for the BFECC pass: the uncorrected forward step and the error.
    ofm::DHMemory<float> theta_fwd(cell_num);
    ofm::DHMemory<float> theta_err(cell_num);
    theta_a.ClearDevAsync(stream);
    theta_b.ClearDevAsync(stream);
    theta_fwd.ClearDevAsync(stream);
    theta_err.ClearDevAsync(stream);
    cudaStreamSynchronize(stream);
    ofm::DHMemory<float>* theta = &theta_a;
    ofm::DHMemory<float>* next  = &theta_b;

    FILE* csv = fopen(csv_path, "w");
    if (!csv) {
        printf("cannot open %s\n", csv_path);
        return 1;
    }
    fprintf(csv, "step,time,max_theta,plume_top,w_max,u_max,plane_theta,omega_pos,omega_neg,y_pos,y_neg,"
                 "split_width,best_x,best_omega,best_split,theta_width,theta_split,theta_saddle,theta_peak,"
                 "bifurcated,best_theta_width,best_theta_split,best_bifurcated\n");
    printf("\n%6s %8s %10s %10s %8s %8s %10s %10s %12s\n",
           "step", "t", "max_dT", "top", "w_max", "u_max", "w_z(+)", "w_z(-)", "split");

    selfcheck::PlumeDiag last;
    last.valid = false;
    for (int step = 0; step < run.steps; step++) {
        const float t = step * run.dt;
        selfcheck::AddPlumeHeatAsync(*theta, td, solver.grid_origin_, dx, spec, t, run.dt, stream);
        // The buoyancy and the drag are built on the latest projected velocity,
        // and theta is then advected across the step by the velocity of the step
        // just taken -- a midpoint velocity, so the semi-Lagrangian update gets
        // its second-order transport. Neither reads init_u_, which is what lets
        // this run at n > 1.
        selfcheck::SetBuoyancyAndDragAsync(solver, *theta, spec, selfcheck::PlumeVelocityBefore(solver), stream);
        solver.AdvanceAsync(run.dt, stream);
        const selfcheck::PlumeVelocity step_u = selfcheck::PlumeVelocityAfter(solver);
        selfcheck::AdvectThetaAsync(*next, theta_fwd, theta_err, td, *theta, step_u, dx, run.dt,
                                    run.theta_bfecc, run.theta_clamp, stream);
        std::swap(theta, next);
        // Once per cycle, not once per step: ReinitAsync re-marches the flow map
        // through the whole cycle's velocity history.
        if ((step + 1) % run.reinit_every == 0)
            solver.ReinitAsync(run.dt, stream);

        // Only ever at a cycle boundary, which the diag_every check above
        // guarantees; the final step is added when it happens to be one.
        if ((step + 1) % run.diag_every == 0
            || (step + 1 == run.steps && (step + 1) % run.reinit_every == 0)) {
            const selfcheck::PlumeDiag d = selfcheck::MeasurePlume(solver, *theta, run.plane_x, run.cvp_z, stream);
            printf("%6d %8.1f %10.3f %10.1f %8.2f %8.2f %10.4f %10.4f %12.1f | best x=%6.0f |w|=%.4f split=%6.1f%s\n",
                   step + 1, (step + 1) * run.dt, d.max_theta, d.plume_top, d.w_max, d.u_max,
                   d.omega_pos, d.omega_neg, d.split_width,
                   d.best_x, d.best_omega, d.best_split, d.valid ? "" : "  [no CVP on plane]");
            printf("       theta: width %6.1f m, split %6.1f m%s (peaks %.3f K over saddle %.3f K) "
                   "| best plane: width %6.1f m, split %6.1f m%s\n",
                   d.theta_width, d.theta_split, d.bifurcated ? "" : " [single lobe]",
                   d.theta_peak, d.theta_saddle,
                   d.best_theta_width, d.best_theta_split, d.best_bifurcated ? "" : " [single lobe]");
            fprintf(csv, "%d,%.3f,%.6f,%.3f,%.6f,%.6f,%.6f,%.6e,%.6e,%.3f,%.3f,%.3f,%.3f,%.6e,%.3f,"
                         "%.3f,%.3f,%.6f,%.6f,%d,%.3f,%.3f,%d\n",
                    step + 1, (step + 1) * run.dt, d.max_theta, d.plume_top, d.w_max, d.u_max, d.plane_theta,
                    d.omega_pos, d.omega_neg, d.y_pos, d.y_neg, d.split_width,
                    d.best_x, d.best_omega, d.best_split,
                    d.theta_width, d.theta_split, d.theta_saddle, d.theta_peak, d.bifurcated ? 1 : 0,
                    d.best_theta_width, d.best_theta_split, d.best_bifurcated ? 1 : 0);
            fflush(csv);
            last = d;
        }
    }
    fclose(csv);

    printf("\nwrote %s\n", csv_path);
    if (!last.valid) {
        printf("FAIL: no counter-rotating pair on the x = %.0f m plane at z = %.0f m\n", run.plane_x, run.cvp_z);
        return 1;
    }
    printf("PASS: counter-rotating pair present, omega_z extrema %.1f m apart "
           "(w_z = %+.4f at y = %.0f m, %+.4f at y = %.0f m)\n",
           last.split_width, last.omega_pos, last.y_pos, last.omega_neg, last.y_neg);
    printf("      theta bifurcation on the same plane: %s, width %.1f m, split %.1f m\n",
           last.bifurcated ? "present" : "absent (single lobe)", last.theta_width, last.theta_split);
    return 0;
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
    PlumeRun plume;       // stage A only

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
        else if (arg == "--theta-advection" && i + 1 < argc) {
            const std::string mode = argv[++i];
            plume.theta_bfecc      = (mode != "plain");
            plume.theta_clamp      = (mode == "bfecc-clamp");
            if (mode != "plain" && mode != "bfecc" && mode != "bfecc-clamp") {
                printf("--theta-advection takes plain, bfecc or bfecc-clamp\n");
                return 1;
            }
        }
        else if (arg == "--reinit-every" && i + 1 < argc)
            reinit_every = std::atoi(argv[++i]);
        else if (arg == "--rk-order" && i + 1 < argc)
            rk_order = std::atoi(argv[++i]);
        else if (arg == "--res-tiles" && i + 1 < argc)
            res_tiles = std::atoi(argv[++i]);
        else if (arg == "--loop-radius" && i + 1 < argc)
            loop_radius = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--z0" && i + 1 < argc)
            plume.z0 = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--q0" && i + 1 < argc)
            plume.q0 = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--mu" && i + 1 < argc)
            plume.mu = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--plane-x" && i + 1 < argc)
            plume.plane_x = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--cvp-z" && i + 1 < argc)
            plume.cvp_z = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--tiles" && i + 3 < argc) {
            plume.tiles.x = std::atoi(argv[++i]);
            plume.tiles.y = std::atoi(argv[++i]);
            plume.tiles.z = std::atoi(argv[++i]);
        }
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
                   "       selfcheck --test tilting [--res-tiles T]\n"
                   "       selfcheck --test coreradii [--res-tiles T] [--core B0] [--circulation G]\n"
                   "       selfcheck --test damkohler [--res-tiles T] [--core B0] [--circulation G]\n"
                   "       selfcheck --test plume [--z0 Z] [--q0 Q] [--mu MU] [--dt DT] [--steps N]\n"
                   "                 [--diag-every N] [--plane-x X] [--cvp-z Z] [--tiles TX TY TZ]\n"
                   "                 [--reinit-every N] [--theta-advection plain|bfecc|bfecc-clamp]\n");
            return 0;
        }
    }

    // Every case below reports numbers, and a device the kernels cannot launch on
    // would let all of them report zeros. Check once, here, before any of them run.
    if (!selfcheck::CheckDeviceUsable())
        return 2;

    if (test == "leapfrog3d")
        return RunLeapfrogRings(total_steps, diag_every, csv_path.c_str(), rp, dt, reinit_every, rk_order);

    if (test == "tilting")
        return RunTiltingStretching(res_tiles, bp.core, bp.circulation);

    if (test == "coreradii") {
        if (csv_path == "leapfrog3d.csv")
            csv_path = "coreradii.csv";
        return RunCoreRadii(res_tiles, bp.core, bp.circulation, csv_path.c_str());
    }

    if (test == "damkohler") {
        if (csv_path == "leapfrog3d.csv")
            csv_path = "damkohler.csv";
        return RunDamkohler(res_tiles, bp.core, bp.circulation, csv_path.c_str());
    }

    if (test == "attribution") {
        if (!dt_set)
            dt = 1.0f / 120.0f;
        if (csv_path == "leapfrog3d.csv")
            csv_path = "attribution.csv";
        return RunAttribution(total_steps, diag_every, csv_path.c_str(), bp, dt, reinit_every, rk_order, res_tiles, loop_radius);
    }

    if (test == "plume") {
        if (csv_path == "leapfrog3d.csv")
            csv_path = "plume.csv";
        if (dt_set)
            plume.dt = dt;
        if (total_steps != 2000)
            plume.steps = total_steps;
        if (diag_every != 10)
            plume.diag_every = diag_every;
        plume.reinit_every = reinit_every;
        plume.rk_order     = rk_order;
        return RunPlume(plume, csv_path.c_str());
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
