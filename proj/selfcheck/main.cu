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

int main(int argc, char** argv)
{
    std::string test        = "leapfrog3d";
    int total_steps         = 2000; // LFM Table 4: 28.2 s total at 14.1 ms/step
    int diag_every          = 10;
    std::string csv_path    = "leapfrog3d.csv";
    RingParams rp;
    float dt         = 1.0f / 60.0f;
    int reinit_every = 1; // 1 = one-step (OFM); LFM runs its Figure 14 at 10
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
        else if (arg == "--core" && i + 1 < argc)
            rp.core = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--circulation" && i + 1 < argc)
            rp.circulation = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--spacing" && i + 1 < argc)
            rp.spacing = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--reinit-every" && i + 1 < argc)
            reinit_every = std::atoi(argv[++i]);
        else if (arg == "--rk-order" && i + 1 < argc)
            rk_order = std::atoi(argv[++i]);
        else if (arg == "--dt" && i + 1 < argc)
            dt = static_cast<float>(std::atof(argv[++i]));
        else if (arg == "--help") {
            printf("usage: selfcheck [--test leapfrog3d] [--steps N] [--diag-every N] [--csv PATH]\n"
                   "                 [--radius R] [--core S] [--circulation G] [--spacing D]\n"
                   "                 [--reinit-every N] [--rk-order 2|3|4] [--dt DT]\n");
            return 0;
        }
    }

    if (test == "leapfrog3d")
        return RunLeapfrogRings(total_steps, diag_every, csv_path.c_str(), rp, dt, reinit_every, rk_order);

    printf("unknown test: %s\n", test.c_str());
    return 1;
}
