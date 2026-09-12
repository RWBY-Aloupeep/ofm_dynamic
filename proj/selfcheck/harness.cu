#include "harness.h"
#include <cub/cub.cuh>
#include <cstring>
#include <cstdint>

#include "ofm_util.h"
#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace selfcheck {

// Grid indexing helpers live in the solver's namespace.
using ofm::IjkToIdx;
using ofm::Prod;
using ofm::TileIdxToIjk;
using ofm::VoxelIdxToIjk;

namespace {

__host__ __device__ inline float3 Normalize(float3 v)
{
    const float n = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return { v.x / n, v.y / n, v.z / n };
}

__host__ __device__ inline float3 Cross(float3 a, float3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

// Any unit vector orthogonal to n.
__host__ __device__ inline float3 OrthoBasis(float3 n)
{
    const float3 pick = (fabsf(n.x) < 0.9f) ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 1.0f, 0.0f);
    return Normalize(Cross(n, pick));
}

// Regularized Biot-Savart velocity of one circular filament, evaluated by
// midpoint quadrature over num_segments straight pieces. The (|r|^2 + core^2)
// denominator is the Rosenhead-Moore smoothing that keeps the integrand finite
// on the filament itself.
__device__ float3 RingVelocity(float3 pos, RingSpec ring, int num_segments)
{
    const float3 axis = Normalize(ring.axis);
    const float3 e1   = OrthoBasis(axis);
    const float3 e2   = Cross(axis, e1);

    const float two_pi   = 6.28318530718f;
    const float d_theta  = two_pi / static_cast<float>(num_segments);
    const float core_sq  = ring.core * ring.core;
    const float prefactor = ring.circulation / (4.0f * 3.14159265359f);

    float3 vel = { 0.0f, 0.0f, 0.0f };
    for (int s = 0; s < num_segments; s++) {
        const float theta = (static_cast<float>(s) + 0.5f) * d_theta;
        const float ct    = cosf(theta);
        const float st    = sinf(theta);

        // Point on the filament and its tangent.
        const float3 p = { ring.center.x + ring.radius * (ct * e1.x + st * e2.x),
                           ring.center.y + ring.radius * (ct * e1.y + st * e2.y),
                           ring.center.z + ring.radius * (ct * e1.z + st * e2.z) };
        const float3 t = { -st * e1.x + ct * e2.x, -st * e1.y + ct * e2.y, -st * e1.z + ct * e2.z };

        const float3 r      = { pos.x - p.x, pos.y - p.y, pos.z - p.z };
        const float r_sq    = r.x * r.x + r.y * r.y + r.z * r.z + core_sq;
        const float inv_r3  = rsqrtf(r_sq * r_sq * r_sq);
        const float3 dl_x_r = Cross(t, r);
        const float w       = ring.radius * d_theta * inv_r3;

        vel.x += dl_x_r.x * w;
        vel.y += dl_x_r.y * w;
        vel.z += dl_x_r.z * w;
    }
    return { prefactor * vel.x, prefactor * vel.y, prefactor * vel.z };
}

// component: 0 = x faces, 1 = y faces, 2 = z faces.
__global__ void AddRingVelocityKernel(float* u_axis, int3 axis_tile_dim, int component,
                                      float3 grid_origin, float dx, RingSpec ring, int num_segments)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(axis_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;

    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        const int idx        = tile_idx * 512 + voxel_idx;

        // Staggered sample positions, matching ResetToIdentity{X,Y,Z}Kernel.
        float3 offset;
        if (component == 0)
            offset = { 0.0f, 0.5f, 0.5f };
        else if (component == 1)
            offset = { 0.5f, 0.0f, 0.5f };
        else
            offset = { 0.5f, 0.5f, 0.0f };

        const float3 pos = { grid_origin.x + (ijk.x + offset.x) * dx,
                             grid_origin.y + (ijk.y + offset.y) * dx,
                             grid_origin.z + (ijk.z + offset.z) * dx };

        const float3 vel = RingVelocity(pos, ring, num_segments);
        const float v    = (component == 0) ? vel.x : ((component == 1) ? vel.y : vel.z);
        u_axis[idx] += v;
    }
}

// Burgers profile, evaluated on one staggered component. Uniform along z.
__global__ void AddColumnVelocityKernel(float* u_axis, int3 axis_tile_dim, int component,
                                        float3 grid_origin, float dx, selfcheck::ColumnVortexSpec spec)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(axis_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;

    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        const int idx        = tile_idx * 512 + voxel_idx;

        float3 offset;
        if (component == 0)
            offset = { 0.0f, 0.5f, 0.5f };
        else if (component == 1)
            offset = { 0.5f, 0.0f, 0.5f };
        else
            offset = { 0.5f, 0.5f, 0.0f };

        const float px = grid_origin.x + (ijk.x + offset.x) * dx;
        const float py = grid_origin.y + (ijk.y + offset.y) * dx;

        const float rx = px - spec.centre_x;
        const float ry = py - spec.centre_y;
        const float r2 = rx * rx + ry * ry;
        const float r  = sqrtf(r2);

        float v = 0.0f;
        if (component != 2 && r > 1e-6f) {
            const float b2      = spec.core * spec.core;
            const float u_theta = spec.circulation / (2.0f * 3.14159265359f * r) * (1.0f - expf(-r2 / b2));
            // theta_hat = (-ry, rx) / r
            v = (component == 0) ? (-u_theta * ry / r) : (u_theta * rx / r);
        }
        u_axis[idx] += v;
    }
}

// Trilinear sample of a cell-centred, tile-ordered field. Cell (i,j,k) is centred
// at grid_origin + (i+0.5)*dx, so the fractional index is offset by half a cell.
float3 SampleCentred(const float3* data, int3 td, float3 grid_origin, float dx, double px, double py, double pz)
{
    const int nx = td.x * 8, ny = td.y * 8, nz = td.z * 8;
    const double fx = (px - grid_origin.x) / dx - 0.5;
    const double fy = (py - grid_origin.y) / dx - 0.5;
    const double fz = (pz - grid_origin.z) / dx - 0.5;

    const int i0 = static_cast<int>(std::floor(fx));
    const int j0 = static_cast<int>(std::floor(fy));
    const int k0 = static_cast<int>(std::floor(fz));
    const double wx = fx - i0, wy = fy - j0, wz = fz - k0;

    auto clamp = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };

    float3 acc = { 0.0f, 0.0f, 0.0f };
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < 2; b++)
            for (int c = 0; c < 2; c++) {
                const double w = (a ? wx : 1.0 - wx) * (b ? wy : 1.0 - wy) * (c ? wz : 1.0 - wz);
                const int3 ijk = { clamp(i0 + a, nx - 1), clamp(j0 + b, ny - 1), clamp(k0 + c, nz - 1) };
                const float3 v = data[IjkToIdx(td, ijk)];
                acc.x += static_cast<float>(w * v.x);
                acc.y += static_cast<float>(w * v.y);
                acc.z += static_cast<float>(w * v.z);
            }
    return acc;
}

__global__ void SetShearVelocityKernel(float* u_axis, int3 axis_tile_dim, int component,
                                       float3 grid_origin, float dx, selfcheck::ShearSpec spec)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(axis_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;

    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        const int idx        = tile_idx * 512 + voxel_idx;

        float3 offset;
        if (component == 0)
            offset = { 0.0f, 0.5f, 0.5f };
        else if (component == 1)
            offset = { 0.5f, 0.0f, 0.5f };
        else
            offset = { 0.5f, 0.5f, 0.0f };

        const float px = grid_origin.x + (ijk.x + offset.x) * dx;
        const float py = grid_origin.y + (ijk.y + offset.y) * dx;
        const float pz = grid_origin.z + (ijk.z + offset.z) * dx;

        float v;
        if (component == 0)
            v = -spec.omega * py - 0.5f * spec.g * px;
        else if (component == 1)
            v = spec.omega * px - spec.c * pz - 0.5f * spec.g * py;
        else
            v = spec.w0 + spec.s * px + spec.g * pz;
        u_axis[idx] = v;
    }
}

} // namespace

void SetupSolver(ofm::OFM& solver, const SolverConfig& config, GPUTimer& profiler, cudaStream_t stream)
{
    solver.max_level_num_ = config.max_levels;
    solver.Alloc(config.tile_dim, config.reinit_every);
    solver.SetProfilier(&profiler);

    solver.step_        = 0;
    solver.cycle_len_   = 0;
    solver.rk_order_    = config.rk_order;
    solver.dx_          = config.len_y / static_cast<float>(8 * config.tile_dim.y);
    solver.grid_origin_ = { 0.0f, 0.0f, 0.0f };
    solver.inlet_norm_  = config.inlet_norm;
    solver.inlet_angle_ = config.inlet_angle;

    solver.use_bfecc_clamp_   = config.bfecc_clamp;
    solver.use_dynamic_solid_ = false;

    solver.init_u_x_->ClearDevAsync(stream);
    solver.init_u_y_->ClearDevAsync(stream);
    solver.init_u_z_->ClearDevAsync(stream);

    const float pi           = 3.14159265359f;
    const float radian_angle = config.inlet_angle / 180.0f * pi;
    const float3 neg_bc_val  = { config.inlet_norm * cosf(radian_angle), config.inlet_norm * sinf(radian_angle), 0.0f };
    const float3 pos_bc_val  = neg_bc_val;
    ofm::SetWallBcAsync(*solver.is_bc_x_, *solver.is_bc_y_, *solver.is_bc_z_,
                        *solver.bc_val_x_, *solver.bc_val_y_, *solver.bc_val_z_,
                        config.tile_dim, neg_bc_val, pos_bc_val, stream);

    ofm::SetCoefByIsBcAsync(*(solver.amgpcg_.poisson_vector_[0].is_dof_),
                            *(solver.amgpcg_.poisson_vector_[0].a_diag_),
                            *(solver.amgpcg_.poisson_vector_[0].a_x_),
                            *(solver.amgpcg_.poisson_vector_[0].a_y_),
                            *(solver.amgpcg_.poisson_vector_[0].a_z_),
                            config.tile_dim, *solver.is_bc_x_, *solver.is_bc_y_, *solver.is_bc_z_, stream);
    solver.amgpcg_.BuildAsync(6.0f, -1.0f, stream);
    solver.amgpcg_.solve_by_tol_ = false;
    solver.amgpcg_.max_iter_     = config.cg_iter;
}

void ProjectCurrentVelocityAsync(ofm::OFM& solver, cudaStream_t stream)
{
    // ProjectAsync operates in place on tmp_u_*, so stage the current state there
    // and copy the projected result back.
    const int3 td = solver.tile_dim_;
    const int nx  = Prod({ td.x + 1, td.y, td.z }) * 512;
    const int ny  = Prod({ td.x, td.y + 1, td.z }) * 512;
    const int nz  = Prod({ td.x, td.y, td.z + 1 }) * 512;

    cudaMemcpyAsync(solver.tmp_u_x_->dev_ptr_, solver.init_u_x_->dev_ptr_, nx * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(solver.tmp_u_y_->dev_ptr_, solver.init_u_y_->dev_ptr_, ny * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(solver.tmp_u_z_->dev_ptr_, solver.init_u_z_->dev_ptr_, nz * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    solver.ProjectAsync(stream);

    cudaMemcpyAsync(solver.init_u_x_->dev_ptr_, solver.tmp_u_x_->dev_ptr_, nx * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(solver.init_u_y_->dev_ptr_, solver.tmp_u_y_->dev_ptr_, ny * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(solver.init_u_z_->dev_ptr_, solver.tmp_u_z_->dev_ptr_, nz * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

void AddVortexRingsAsync(ofm::OFM& solver, const std::vector<RingSpec>& rings, int num_segments, bool project, cudaStream_t stream)
{
    const int3 td           = solver.tile_dim_;
    const int3 x_tile_dim   = { td.x + 1, td.y, td.z };
    const int3 y_tile_dim   = { td.x, td.y + 1, td.z };
    const int3 z_tile_dim   = { td.x, td.y, td.z + 1 };

    for (const RingSpec& ring : rings) {
        AddRingVelocityKernel<<<Prod(x_tile_dim), 128, 0, stream>>>(
            solver.init_u_x_->dev_ptr_, x_tile_dim, 0, solver.grid_origin_, solver.dx_, ring, num_segments);
        AddRingVelocityKernel<<<Prod(y_tile_dim), 128, 0, stream>>>(
            solver.init_u_y_->dev_ptr_, y_tile_dim, 1, solver.grid_origin_, solver.dx_, ring, num_segments);
        AddRingVelocityKernel<<<Prod(z_tile_dim), 128, 0, stream>>>(
            solver.init_u_z_->dev_ptr_, z_tile_dim, 2, solver.grid_origin_, solver.dx_, ring, num_segments);
    }

    if (project)
        ProjectCurrentVelocityAsync(solver, stream);
}

FieldStats ComputeFieldStats(ofm::OFM& solver, cudaStream_t stream)
{
    const int3 td = solver.tile_dim_;
    ofm::GetCenteralVecAsync(*(solver.u_), td, *(solver.init_u_x_), *(solver.init_u_y_), *(solver.init_u_z_), stream);
    ofm::GetVorNormAsync(*(solver.vor_norm_), td, *(solver.u_), solver.dx_, stream);

    solver.u_->DevToHostAsync(stream);
    solver.vor_norm_->DevToHostAsync(stream);
    cudaStreamSynchronize(stream);

    FieldStats stats = { 0.0, 0.0f, 0.0f, true };
    const int n      = Prod(td) * 512;
    const float cell_volume = solver.dx_ * solver.dx_ * solver.dx_;

    for (int i = 0; i < n; i++) {
        const float3 v = solver.u_->host_ptr_[i];
        const float speed_sq = v.x * v.x + v.y * v.y + v.z * v.z;
        if (!isfinite(speed_sq)) {
            stats.finite = false;
            continue;
        }
        stats.kinetic_energy += 0.5 * static_cast<double>(speed_sq) * cell_volume;
        const float speed = sqrtf(speed_sq);
        if (speed > stats.max_speed)
            stats.max_speed = speed;

        const float w = solver.vor_norm_->host_ptr_[i];
        if (!isfinite(w))
            stats.finite = false;
        else if (w > stats.max_vorticity)
            stats.max_vorticity = w;
    }
    return stats;
}

void AddColumnVortexAsync(ofm::OFM& solver, const ColumnVortexSpec& spec, bool project, cudaStream_t stream)
{
    const int3 td         = solver.tile_dim_;
    const int3 x_tile_dim = { td.x + 1, td.y, td.z };
    const int3 y_tile_dim = { td.x, td.y + 1, td.z };
    const int3 z_tile_dim = { td.x, td.y, td.z + 1 };

    AddColumnVelocityKernel<<<Prod(x_tile_dim), 128, 0, stream>>>(
        solver.init_u_x_->dev_ptr_, x_tile_dim, 0, solver.grid_origin_, solver.dx_, spec);
    AddColumnVelocityKernel<<<Prod(y_tile_dim), 128, 0, stream>>>(
        solver.init_u_y_->dev_ptr_, y_tile_dim, 1, solver.grid_origin_, solver.dx_, spec);
    AddColumnVelocityKernel<<<Prod(z_tile_dim), 128, 0, stream>>>(
        solver.init_u_z_->dev_ptr_, z_tile_dim, 2, solver.grid_origin_, solver.dx_, spec);

    if (project)
        ProjectCurrentVelocityAsync(solver, stream);
}

ColumnDiag MeasureColumnVortex(ofm::OFM& solver, float centre_x, float centre_y, cudaStream_t stream)
{
    const int3 td = solver.tile_dim_;
    ofm::GetCenteralVecAsync(*(solver.u_), td, *(solver.init_u_x_), *(solver.init_u_y_), *(solver.init_u_z_), stream);
    ofm::GetVorNormAsync(*(solver.vor_norm_), td, *(solver.u_), solver.dx_, stream);
    solver.u_->DevToHostAsync(stream);
    solver.vor_norm_->DevToHostAsync(stream);
    cudaStreamSynchronize(stream);

    const int nx = td.x * 8, ny = td.y * 8, nz = td.z * 8;
    const float dx = solver.dx_;

    ColumnDiag diag = { 0.0f, 0.0f, 0.0f, true };

    // Peak vorticity. vor_norm_ is written in plain x-major order.
    for (int i = 0; i < nx * ny * nz; i++) {
        const float w = solver.vor_norm_->host_ptr_[i];
        if (!isfinite(w)) {
            diag.valid = false;
            break;
        }
        if (w > diag.max_vorticity)
            diag.max_vorticity = w;
    }
    if (!diag.valid)
        return diag;

    // Azimuthal velocity binned by radius, averaged over azimuth and over the
    // whole column. solver.u_ is in tiled order, so it has to be indexed through
    // IjkToIdx rather than read straight through.
    const float r_max      = 0.45f;
    const int bin_num      = static_cast<int>(r_max / dx);
    std::vector<double> sum(bin_num, 0.0);
    std::vector<int> count(bin_num, 0);

    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            const float px = solver.grid_origin_.x + (i + 0.5f) * dx;
            const float py = solver.grid_origin_.y + (j + 0.5f) * dx;
            const float rx = px - centre_x;
            const float ry = py - centre_y;
            const float r  = sqrtf(rx * rx + ry * ry);
            const int bin  = static_cast<int>(r / dx);
            if (bin >= bin_num || r < 1e-6f)
                continue;
            for (int k = 0; k < nz; k++) {
                const float3 v = solver.u_->host_ptr_[IjkToIdx(td, { i, j, k })];
                if (!isfinite(v.x) || !isfinite(v.y)) {
                    diag.valid = false;
                    return diag;
                }
                sum[bin] += (-v.x * ry + v.y * rx) / r;
                count[bin]++;
            }
        }

    for (int b = 0; b < bin_num; b++) {
        if (count[b] == 0)
            continue;
        const float mean = static_cast<float>(sum[b] / count[b]);
        if (mean > diag.u_theta_peak) {
            diag.u_theta_peak = mean;
            diag.r_peak       = (b + 0.5f) * dx;
        }
    }
    return diag;
}

void SetShearFieldAsync(ofm::OFM& solver, const ShearSpec& spec, cudaStream_t stream)
{
    const int3 td         = solver.tile_dim_;
    const int3 x_tile_dim = { td.x + 1, td.y, td.z };
    const int3 y_tile_dim = { td.x, td.y + 1, td.z };
    const int3 z_tile_dim = { td.x, td.y, td.z + 1 };

    SetShearVelocityKernel<<<Prod(x_tile_dim), 128, 0, stream>>>(
        solver.init_u_x_->dev_ptr_, x_tile_dim, 0, solver.grid_origin_, solver.dx_, spec);
    SetShearVelocityKernel<<<Prod(y_tile_dim), 128, 0, stream>>>(
        solver.init_u_y_->dev_ptr_, y_tile_dim, 1, solver.grid_origin_, solver.dx_, spec);
    SetShearVelocityKernel<<<Prod(z_tile_dim), 128, 0, stream>>>(
        solver.init_u_z_->dev_ptr_, z_tile_dim, 2, solver.grid_origin_, solver.dx_, spec);
}

__global__ void AddAxialJetKernel(float* u_z, int3 z_tile_dim, float3 grid_origin, float dx, selfcheck::AxialJetSpec spec)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(z_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;

    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        const int idx        = tile_idx * 512 + voxel_idx;

        // The z component sits on the face centre in x and y.
        const float px = grid_origin.x + (ijk.x + 0.5f) * dx;
        const float py = grid_origin.y + (ijk.y + 0.5f) * dx;
        const float rx = px - spec.centre_x;
        const float ry = py - spec.centre_y;
        const float r2 = rx * rx + ry * ry;

        u_z[idx] += spec.w_peak * expf(-r2 / (spec.scale * spec.scale));
    }
}

void SetAxialJetAsync(ofm::OFM& solver, const AxialJetSpec& spec, cudaStream_t stream)
{
    const int3 td         = solver.tile_dim_;
    const int3 z_tile_dim = { td.x, td.y, td.z + 1 };
    AddAxialJetKernel<<<Prod(z_tile_dim), 128, 0, stream>>>(
        solver.init_u_z_->dev_ptr_, z_tile_dim, solver.grid_origin_, solver.dx_, spec);
}

TiltStretch MeasureTiltingStretching(ofm::OFM& solver, cudaStream_t stream)
{
    const int3 td = solver.tile_dim_;
    ofm::GetCenteralVecAsync(*(solver.u_), td, *(solver.init_u_x_), *(solver.init_u_y_), *(solver.init_u_z_), stream);
    solver.u_->DevToHostAsync(stream);
    cudaStreamSynchronize(stream);

    const int nx = td.x * 8, ny = td.y * 8, nz = td.z * 8;
    const double inv_2dx = 1.0 / (2.0 * solver.dx_);
    const float3* u = solver.u_->host_ptr_;

    TiltStretch out = { 0.0, 0.0, 0.0, 0.0, 0.0, true };
    long long count = 0;

    auto at = [&](int i, int j, int k) { return u[IjkToIdx(td, { i, j, k })]; };

    // Two layers in from every face: the vorticity needs one neighbour and the
    // gradient of w another, and one-sided differences at the wall would bias the
    // average without saying anything about the operator.
    for (int i = 2; i < nx - 2; i++)
        for (int j = 2; j < ny - 2; j++)
            for (int k = 2; k < nz - 2; k++) {
                const float3 xp = at(i + 1, j, k), xm = at(i - 1, j, k);
                const float3 yp = at(i, j + 1, k), ym = at(i, j - 1, k);
                const float3 zp = at(i, j, k + 1), zm = at(i, j, k - 1);

                const double w_x = (yp.z - ym.z) * inv_2dx - (zp.y - zm.y) * inv_2dx;
                const double w_y = (zp.x - zm.x) * inv_2dx - (xp.z - xm.z) * inv_2dx;
                const double w_z = (xp.y - xm.y) * inv_2dx - (yp.x - ym.x) * inv_2dx;

                const double dwdx = (xp.z - xm.z) * inv_2dx;
                const double dwdy = (yp.z - ym.z) * inv_2dx;
                const double dwdz = (zp.z - zm.z) * inv_2dx;

                const double tilting    = w_x * dwdx + w_y * dwdy;
                const double stretching = w_z * dwdz;

                if (!isfinite(tilting) || !isfinite(stretching)) {
                    out.valid = false;
                    return out;
                }
                out.tilting_mean += tilting;
                out.stretching_mean += stretching;
                out.tilting_abs_mean += std::fabs(tilting);
                out.stretching_abs_mean += std::fabs(stretching);
                count++;
            }

    if (count == 0) {
        out.valid = false;
        return out;
    }
    out.tilting_mean /= count;
    out.stretching_mean /= count;
    out.tilting_abs_mean /= count;
    out.stretching_abs_mean /= count;
    const double denom = out.tilting_abs_mean + out.stretching_abs_mean;
    out.ratio = denom > 0.0 ? out.tilting_abs_mean / denom : 0.0;
    return out;
}

double CirculationOnCircle(ofm::OFM& solver,
                           const ofm::DHMemory<float>& field_x, const ofm::DHMemory<float>& field_y, const ofm::DHMemory<float>& field_z,
                           float centre_x, float centre_y, float radius, int samples, cudaStream_t stream)
{
    const int3 td = solver.tile_dim_;
    ofm::GetCenteralVecAsync(*(solver.u_), td, field_x, field_y, field_z, stream);
    solver.u_->DevToHostAsync(stream);
    cudaStreamSynchronize(stream);

    const float dx  = solver.dx_;
    const double cz = solver.grid_origin_.z + 0.5 * (td.z * 8) * dx;
    const double pi = 3.14159265358979323846;
    const double dtheta = 2.0 * pi / samples;

    // Midpoint rule around the circle. The tangent of a counter-clockwise
    // traversal at angle theta is (-sin theta, cos theta, 0).
    double total = 0.0;
    for (int s = 0; s < samples; s++) {
        const double th = (s + 0.5) * dtheta;
        const double px = centre_x + radius * std::cos(th);
        const double py = centre_y + radius * std::sin(th);
        const float3 v  = SampleCentred(solver.u_->host_ptr_, td, solver.grid_origin_, dx, px, py, cz);
        total += (-v.x * std::sin(th) + v.y * std::cos(th)) * radius * dtheta;
    }
    return total;
}

std::vector<double> CirculationRadialSweep(ofm::OFM& solver,
                                           const ofm::DHMemory<float>& field_x, const ofm::DHMemory<float>& field_y, const ofm::DHMemory<float>& field_z,
                                           float centre_x, float centre_y, const std::vector<double>& radii, int samples, cudaStream_t stream)
{
    const int3 td = solver.tile_dim_;
    ofm::GetCenteralVecAsync(*(solver.u_), td, field_x, field_y, field_z, stream);
    solver.u_->DevToHostAsync(stream);
    cudaStreamSynchronize(stream);

    const float dx  = solver.dx_;
    const double cz = solver.grid_origin_.z + 0.5 * (td.z * 8) * dx;
    const double pi = 3.14159265358979323846;
    const double dtheta = 2.0 * pi / samples;

    std::vector<double> out(radii.size(), 0.0);
    for (size_t i = 0; i < radii.size(); i++) {
        const double radius = radii[i];
        double total = 0.0;
        for (int s = 0; s < samples; s++) {
            const double th = (s + 0.5) * dtheta;
            const double px = centre_x + radius * std::cos(th);
            const double py = centre_y + radius * std::sin(th);
            const float3 v  = SampleCentred(solver.u_->host_ptr_, td, solver.grid_origin_, dx, px, py, cz);
            total += (-v.x * std::sin(th) + v.y * std::cos(th)) * radius * dtheta;
        }
        out[i] = total;
    }
    return out;
}

// ---------------------------------------------------------------------------
// D3: core radii
// ---------------------------------------------------------------------------

RadialProfile MeasureRadialProfile(ofm::OFM& solver, int kind, float centre_x, float centre_y, float r_max, cudaStream_t stream)
{
    const int3 td = solver.tile_dim_;
    ofm::GetCenteralVecAsync(*(solver.u_), td, *(solver.init_u_x_), *(solver.init_u_y_), *(solver.init_u_z_), stream);
    solver.u_->DevToHostAsync(stream);
    cudaStreamSynchronize(stream);

    const int nx = td.x * 8, ny = td.y * 8, nz = td.z * 8;
    const float dx = solver.dx_;

    RadialProfile out;
    const int bin_num = static_cast<int>(r_max / dx);
    std::vector<double> sum(bin_num, 0.0);
    std::vector<long long> count(bin_num, 0);

    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            const float px = solver.grid_origin_.x + (i + 0.5f) * dx;
            const float py = solver.grid_origin_.y + (j + 0.5f) * dx;
            const float rx = px - centre_x;
            const float ry = py - centre_y;
            const float r  = sqrtf(rx * rx + ry * ry);
            const int bin  = static_cast<int>(r / dx);
            if (bin >= bin_num || r < 1e-6f)
                continue;
            for (int k = 0; k < nz; k++) {
                const float3 v = solver.u_->host_ptr_[IjkToIdx(td, { i, j, k })];
                if (!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z)) {
                    out.valid = false;
                    return out;
                }
                sum[bin] += (kind == kProfileAzimuthal) ? ((-v.x * ry + v.y * rx) / r) : v.z;
                count[bin]++;
            }
        }

    for (int b = 0; b < bin_num; b++) {
        if (count[b] == 0)
            continue;
        out.r.push_back((b + 0.5) * dx);
        out.v.push_back(sum[b] / count[b]);
    }
    return out;
}

RadialProfile GaussianProfileOnGrid(const ofm::OFM& solver, float centre_x, float centre_y, float r_max,
                                    double amplitude, double scale)
{
    const int3 td = solver.tile_dim_;
    const int nx = td.x * 8, ny = td.y * 8;
    const float dx = solver.dx_;

    RadialProfile out;
    const int bin_num = static_cast<int>(r_max / dx);
    std::vector<double> sum(bin_num, 0.0);
    std::vector<long long> count(bin_num, 0);

    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            const float px = solver.grid_origin_.x + (i + 0.5f) * dx;
            const float py = solver.grid_origin_.y + (j + 0.5f) * dx;
            const double rx = px - centre_x;
            const double ry = py - centre_y;
            const double r  = std::sqrt(rx * rx + ry * ry);
            const int bin   = static_cast<int>(r / dx);
            if (bin >= bin_num || r < 1e-6)
                continue;
            sum[bin] += amplitude * std::exp(-r * r / (scale * scale));
            count[bin]++;
        }

    for (int b = 0; b < bin_num; b++) {
        if (count[b] == 0)
            continue;
        out.r.push_back((b + 0.5) * dx);
        out.v.push_back(sum[b] / count[b]);
    }
    return out;
}

namespace {

// The Burgers shape function, with Gamma_inf factored out.
double BurgersBasis(double r, double b)
{
    const double pi = 3.14159265358979323846;
    return (1.0 - std::exp(-r * r / (b * b))) / (2.0 * pi * r);
}

// Residual of the best fit at a fixed core radius. Gamma_inf enters linearly, so
// it is eliminated by its own normal equation rather than searched over.
double BurgersResidual(const RadialProfile& p, double b, double* gamma_out)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < p.r.size(); i++) {
        const double phi = BurgersBasis(p.r[i], b);
        num += phi * p.v[i];
        den += phi * phi;
    }
    const double gamma = (den > 0.0) ? num / den : 0.0;
    if (gamma_out)
        *gamma_out = gamma;

    double s = 0.0;
    for (size_t i = 0; i < p.r.size(); i++) {
        const double d = p.v[i] - gamma * BurgersBasis(p.r[i], b);
        s += d * d;
    }
    return s;
}

// Peak of a sampled profile, refined by the vertex of the parabola through the
// maximum bin and its two neighbours. Without this the peak radius is quantised
// at the grid spacing, which is coarser than the quantity being tested.
double RefinedPeakRadius(const RadialProfile& p, double* peak_value)
{
    size_t best = 0;
    for (size_t i = 1; i < p.v.size(); i++)
        if (p.v[i] > p.v[best])
            best = i;
    if (peak_value)
        *peak_value = p.v[best];
    if (best == 0 || best + 1 >= p.v.size())
        return p.r[best];

    const double y0 = p.v[best - 1], y1 = p.v[best], y2 = p.v[best + 1];
    const double denom = y0 - 2.0 * y1 + y2;
    if (std::fabs(denom) < 1e-30)
        return p.r[best];
    const double shift = 0.5 * (y0 - y2) / denom; // in bins, within [-0.5, 0.5]
    const double dr    = p.r[best] - p.r[best - 1];
    return p.r[best] + shift * dr;
}

} // namespace

BurgersFit FitBurgers(const RadialProfile& profile)
{
    BurgersFit fit = { 0.0, 0.0, 0.0, 0.0, 0.0, 0, false };
    if (!profile.valid || profile.r.size() < 8)
        return fit;

    const double r_lo = profile.r.front();
    const double r_hi = profile.r.back();

    // Coarse scan first: the residual is smooth in b but the bracket has to be
    // found before a golden-section search can be trusted to be unimodal.
    double best_b = r_lo, best_s = 1e300;
    const int scan = 400;
    for (int i = 0; i <= scan; i++) {
        const double b = r_lo * 0.25 + (r_hi - r_lo * 0.25) * i / scan;
        if (b <= 0.0)
            continue;
        const double s = BurgersResidual(profile, b, nullptr);
        if (s < best_s) {
            best_s = s;
            best_b = b;
        }
    }

    const double step = (r_hi - r_lo * 0.25) / scan;
    double lo = best_b - step, hi = best_b + step;
    if (lo <= 0.0)
        lo = 1e-6;

    const double gr = 0.6180339887498949;
    double c = hi - gr * (hi - lo), d = lo + gr * (hi - lo);
    double fc = BurgersResidual(profile, c, nullptr);
    double fd = BurgersResidual(profile, d, nullptr);
    int iters = 0;
    while (hi - lo > 1e-12 && iters < 200) {
        if (fc < fd) {
            hi = d; d = c; fd = fc;
            c  = hi - gr * (hi - lo);
            fc = BurgersResidual(profile, c, nullptr);
        } else {
            lo = c; c = d; fc = fd;
            d  = lo + gr * (hi - lo);
            fd = BurgersResidual(profile, d, nullptr);
        }
        iters++;
    }

    const double b = 0.5 * (lo + hi);
    double gamma = 0.0;
    const double s = BurgersResidual(profile, b, &gamma);

    double peak_value = 0.0;
    const double r_peak = RefinedPeakRadius(profile, &peak_value);

    fit.b_w       = b;
    fit.gamma_inf = gamma;
    fit.r_peak    = r_peak;
    fit.ratio     = (b > 0.0) ? r_peak / b : 0.0;
    fit.rms_rel   = (peak_value > 0.0) ? std::sqrt(s / profile.r.size()) / peak_value : 0.0;
    fit.iters     = iters;
    fit.ok        = true;
    return fit;
}

double BurgersPeakConstant()
{
    // d/dr [ (1 - exp(-r^2/b^2)) / r ] = 0 reduces, with x = r^2/b^2, to
    // exp(-x)(2x + 1) = 1, whose non-trivial root is near x = 1.2564.
    auto f = [](double x) { return std::exp(-x) * (2.0 * x + 1.0) - 1.0; };
    double lo = 0.5, hi = 5.0;
    for (int i = 0; i < 200; i++) {
        const double mid = 0.5 * (lo + hi);
        if (f(lo) * f(mid) <= 0.0)
            hi = mid;
        else
            lo = mid;
    }
    return std::sqrt(0.5 * (lo + hi));
}

double RadiusAtFraction(const RadialProfile& profile, double fraction)
{
    if (!profile.valid || profile.r.empty())
        return -1.0;

    size_t best = 0;
    for (size_t i = 1; i < profile.v.size(); i++)
        if (profile.v[i] > profile.v[best])
            best = i;

    const double target = fraction * profile.v[best];
    for (size_t i = best + 1; i < profile.v.size(); i++) {
        if (profile.v[i] <= target) {
            const double v0 = profile.v[i - 1], v1 = profile.v[i];
            if (std::fabs(v0 - v1) < 1e-30)
                return profile.r[i];
            const double t = (v0 - target) / (v0 - v1);
            return profile.r[i - 1] + t * (profile.r[i] - profile.r[i - 1]);
        }
    }
    return -1.0; // never falls that far inside the sampled range
}

double RadiusFromFluxes(const RadialProfile& axial)
{
    if (!axial.valid || axial.r.size() < 2)
        return -1.0;

    // Trapezoidal quadrature of Q_hat = int u_z r dr and M_hat = int u_z^2 r dr.
    double q = 0.0, m = 0.0;
    for (size_t i = 1; i < axial.r.size(); i++) {
        const double dr = axial.r[i] - axial.r[i - 1];
        const double a0 = axial.v[i - 1] * axial.r[i - 1];
        const double a1 = axial.v[i] * axial.r[i];
        const double b0 = axial.v[i - 1] * axial.v[i - 1] * axial.r[i - 1];
        const double b1 = axial.v[i] * axial.v[i] * axial.r[i];
        q += 0.5 * (a0 + a1) * dr;
        m += 0.5 * (b0 + b1) * dr;
    }
    if (m <= 0.0)
        return -1.0;
    return q / std::sqrt(m);
}

// ---------------------------------------------------------------------------
// D4: vortex-flame Damkohler number
// ---------------------------------------------------------------------------

DamkohlerCurve MeasureDamkohler(ofm::OFM& solver, float centre_x, float centre_y,
                                double a_extinction, double r_min, double r_max, int radius_samples,
                                int loop_samples, cudaStream_t stream)
{
    DamkohlerCurve out;
    out.r_contour = -1.0;
    out.valid     = true;

    out.r.resize(radius_samples);
    for (int i = 0; i < radius_samples; i++)
        out.r[i] = r_min + (r_max - r_min) * i / (radius_samples - 1);

    out.gamma = CirculationRadialSweep(solver, *(solver.init_u_x_), *(solver.init_u_y_), *(solver.init_u_z_),
                                       centre_x, centre_y, out.r, loop_samples, stream);

    out.a_gamma.resize(radius_samples);
    out.da.resize(radius_samples);
    for (int i = 0; i < radius_samples; i++) {
        if (!isfinite(out.gamma[i])) {
            out.valid = false;
            return out;
        }
        out.a_gamma[i] = out.gamma[i] / (2.0 * out.r[i] * out.r[i]);
        out.da[i]      = (out.a_gamma[i] != 0.0) ? a_extinction / out.a_gamma[i] : 1e300;
    }

    // Da_Gamma rises with radius for a Burgers vortex, so the extinction region
    // is the disc inside the first upward crossing of unity.
    for (int i = 1; i < radius_samples; i++) {
        if ((out.da[i - 1] - 1.0) <= 0.0 && (out.da[i] - 1.0) > 0.0) {
            const double d0 = out.da[i - 1] - 1.0, d1 = out.da[i] - 1.0;
            const double t  = d0 / (d0 - d1);
            out.r_contour   = out.r[i - 1] + t * (out.r[i] - out.r[i - 1]);
            break;
        }
    }
    return out;
}

double BurgersDamkohlerContour(double gamma_inf, double b_w, double a_extinction)
{
    const double kappa = 2.0 * a_extinction * b_w * b_w / gamma_inf;
    if (kappa >= 1.0)
        return -1.0; // Da_Gamma >= 1 everywhere: no extinction region

    auto g = [](double x) { return (1.0 - std::exp(-x)) / x; };
    double lo = 1e-12, hi = 1.0;
    while (g(hi) > kappa && hi < 1e12)
        hi *= 2.0;
    for (int i = 0; i < 300; i++) {
        const double mid = 0.5 * (lo + hi);
        if (g(mid) > kappa)
            lo = mid;
        else
            hi = mid;
    }
    return b_w * std::sqrt(0.5 * (lo + hi));
}

double ExtinctionRateForContour(double gamma_inf, double b_w, double x)
{
    const double kappa = (1.0 - std::exp(-x)) / x;
    return kappa * gamma_inf / (2.0 * b_w * b_w);
}

HostField DownloadVorticityNorm(ofm::OFM& solver, cudaStream_t stream)
{
    const int3 td = solver.tile_dim_;
    ofm::GetCenteralVecAsync(*(solver.u_), td, *(solver.init_u_x_), *(solver.init_u_y_), *(solver.init_u_z_), stream);
    ofm::GetVorNormAsync(*(solver.vor_norm_), td, *(solver.u_), solver.dx_, stream);
    solver.vor_norm_->DevToHostAsync(stream);
    cudaStreamSynchronize(stream);

    HostField field;
    field.dim = { td.x * 8, td.y * 8, td.z * 8 };
    field.data.resize(static_cast<size_t>(field.dim.x) * field.dim.y * field.dim.z);

    // GetVorNormKernel reads the velocity through the tiled index but writes the
    // vorticity norm with ijk.x * (ny * nz) + ijk.y * nz + ijk.z, i.e. plain
    // x-major order, not the tiled layout. Copy it straight through.
    std::memcpy(field.data.data(), solver.vor_norm_->host_ptr_, field.data.size() * sizeof(float));
    return field;
}

// ---------------------------------------------------------------------------
// Stage A: buoyant plume in a sheared cross flow (Cunningham et al. 2005).

namespace {

__host__ __device__ inline float PlumeInletU(float z, float u0, float z0)
{
    return u0 * tanhf(z / z0);
}

// Eq. (10): uniform inside r1, smoothly tapered to zero by r2, decaying with
// height as exp(-z/h) and ramped in time as tanh(t/t_ramp).
__host__ __device__ inline float PlumeHeatShape(float px, float py, const selfcheck::PlumeSpec& s)
{
    const float ddx = px - s.src_x;
    const float ddy = py - s.src_y;
    const float r   = sqrtf(ddx * ddx + ddy * ddy);
    if (r <= s.r1)
        return 1.0f;
    if (r >= s.r2)
        return 0.0f;
    const float rc = 0.5f * (s.r1 + s.r2);
    return 0.5f * (1.0f - tanhf((r - rc) / s.dwidth));
}

__global__ void SetPlumeInletXKernel(uint8_t* is_bc_x, float* bc_val_x, int3 x_tile_dim,
                                     float3 origin, float dx, selfcheck::PlumeSpec spec)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(x_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    const int3 grid_dim = { (x_tile_dim.x - 1) * 8, x_tile_dim.y * 8, x_tile_dim.z * 8 };
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (ijk.x == 0 || ijk.x >= grid_dim.x) {
            const float pz = origin.z + (ijk.z + 0.5f) * dx;
            is_bc_x[idx]   = 1;
            bc_val_x[idx]  = PlumeInletU(pz, spec.u0, spec.z0);
        } else {
            is_bc_x[idx] = 0;
        }
    }
}

__global__ void SetPlumeVelocityKernel(float* u_axis, int3 axis_tile_dim, int component,
                                       float3 origin, float dx, selfcheck::PlumeSpec spec)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(axis_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (component != 0) {
            u_axis[idx] = 0.0f;
            continue;
        }
        const float pz = origin.z + (ijk.z + 0.5f) * dx; // x faces sit at cell-centre height
        u_axis[idx]    = PlumeInletU(pz, spec.u0, spec.z0);
    }
}

__global__ void AddPlumeHeatKernel(float* theta, int3 tile_dim, float3 origin, float dx,
                                   selfcheck::PlumeSpec spec, float t, float dt)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    const float ramp    = tanhf(t / spec.t_ramp);
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        const float px = origin.x + (ijk.x + 0.5f) * dx;
        const float py = origin.y + (ijk.y + 0.5f) * dx;
        const float pz = origin.z + (ijk.z + 0.5f) * dx;
        const float shape = PlumeHeatShape(px, py, spec);
        if (shape <= 0.0f)
            continue;
        const float q = spec.z_src > 0.0f
                            ? 0.5f * spec.q0 * ramp * expf(-fabsf(pz - spec.z_src) / spec.h) * shape
                            : spec.q0 * ramp * expf(-pz / spec.h) * shape;
        theta[idx] += q / (spec.rho * spec.cp) * dt;
    }
}

__global__ void BuoyancyZKernel(float* f_z, int3 z_tile_dim, int3 tile_dim, const float* theta,
                                float g_over_theta0)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(z_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    const int3 grid_dim = { z_tile_dim.x * 8, z_tile_dim.y * 8, (z_tile_dim.z - 1) * 8 };
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (ijk.z == 0 || ijk.z >= grid_dim.z) {
            f_z[idx] = 0.0f; // the wall faces are Dirichlet anyway
            continue;
        }
        const int3 below = { ijk.x, ijk.y, ijk.z - 1 };
        const int3 above = { ijk.x, ijk.y, ijk.z };
        f_z[idx] = g_over_theta0 * 0.5f * (theta[IjkToIdx(tile_dim, below)] + theta[IjkToIdx(tile_dim, above)]);
    }
}

// Eq. (6) with the canopy confined to the lowest cell level, as in the paper.
// The speed is the horizontal one, taken from the cell-centred velocity either
// side of the face.
__global__ void CanopyDragKernel(float* f_axis, int3 axis_tile_dim, int component,
                                 const float3* u_c, int3 tile_dim, const float* u_axis, float cd_a)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(axis_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    const int3 cell_dim = { tile_dim.x * 8, tile_dim.y * 8, tile_dim.z * 8 };
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (ijk.z != 0 || ijk.x >= cell_dim.x + (component == 0 ? 1 : 0) || ijk.y >= cell_dim.y + (component == 1 ? 1 : 0)) {
            f_axis[idx] = 0.0f;
            continue;
        }
        int3 a = ijk, b = ijk;
        if (component == 0) {
            a.x = ijk.x - 1;
        } else {
            a.y = ijk.y - 1;
        }
        a.x = a.x < 0 ? 0 : (a.x > cell_dim.x - 1 ? cell_dim.x - 1 : a.x);
        a.y = a.y < 0 ? 0 : (a.y > cell_dim.y - 1 ? cell_dim.y - 1 : a.y);
        b.x = b.x > cell_dim.x - 1 ? cell_dim.x - 1 : b.x;
        b.y = b.y > cell_dim.y - 1 ? cell_dim.y - 1 : b.y;
        const float3 ua = u_c[IjkToIdx(tile_dim, a)];
        const float3 ub = u_c[IjkToIdx(tile_dim, b)];
        const float va  = sqrtf(ua.x * ua.x + ua.y * ua.y);
        const float vb  = sqrtf(ub.x * ub.x + ub.y * ub.y);
        f_axis[idx]     = -cd_a * 0.5f * (va + vb) * u_axis[idx];
    }
}

// The damping coefficient of the sponge at height z: Wang et al. 2023's
// beta = (20 dt)^-1 sin^2(pi/2 zeta/zeta0) over the top zeta0 of the domain,
// zero below it.
__host__ __device__ inline float SpongeBeta(float z, float lz, float depth_frac, float dt)
{
    const float zeta0 = depth_frac * lz;
    const float zeta  = z - (lz - zeta0);
    if (zeta <= 0.0f)
        return 0.0f;
    const float sn = sinf(0.5f * 3.14159265359f * fminf(zeta / zeta0, 1.0f));
    return sn * sn / (20.0f * dt);
}

// Adds -beta(z) (u_i - U_i) to the force on one staggered axis, with U = (U(z), 0, 0).
__global__ void SpongeVelocityKernel(float* f_axis, int3 axis_tile_dim, int component, const float* u_axis,
                                     float3 origin, float dx, float lz, selfcheck::PlumeSpec spec, float dt)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(axis_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        const float pz       = origin.z + (ijk.z + (component == 2 ? 0.0f : 0.5f)) * dx;
        const float beta     = SpongeBeta(pz, lz, spec.sponge_depth, dt);
        if (beta <= 0.0f)
            continue;
        const float target = component == 0 ? PlumeInletU(pz, spec.u0, spec.z0) : 0.0f;
        f_axis[idx] -= beta * (u_axis[idx] - target);
    }
}

__global__ void SpongeThetaKernel(float* theta, int3 tile_dim, float3 origin, float dx, float lz,
                                  selfcheck::PlumeSpec spec, float dt)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        const float pz       = origin.z + (ijk.z + 0.5f) * dx;
        const float beta     = SpongeBeta(pz, lz, spec.sponge_depth, dt);
        if (beta > 0.0f)
            theta[idx] *= (1.0f - beta * dt);
    }
}

__global__ void AmbientInflowThetaKernel(float* theta, int3 tile_dim, const float* u_x, const float* u_y,
                                         int outflow, float dt_over_dx)
{
    const int tile_idx    = blockIdx.x;
    const int3 tile_ijk   = TileIdxToIjk(tile_dim, tile_idx);
    const int t_id        = threadIdx.x;
    const int3 cell_dim   = { tile_dim.x * 8, tile_dim.y * 8, tile_dim.z * 8 };
    const int3 x_tile_dim = { tile_dim.x + 1, tile_dim.y, tile_dim.z };
    const int3 y_tile_dim = { tile_dim.x, tile_dim.y + 1, tile_dim.z };
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        float inflow         = 0.0f; // inward normal speed summed over the cell's open faces
        const bool open_x    = outflow >= 1;
        const bool open_y    = outflow == 2 || outflow == 4 || outflow == 5;
        if (open_x && ijk.x == cell_dim.x - 1)
            inflow += fmaxf(0.0f, -u_x[IjkToIdx(x_tile_dim, { ijk.x + 1, ijk.y, ijk.z })]);
        if (open_y && ijk.y == 0)
            inflow += fmaxf(0.0f, u_y[IjkToIdx(y_tile_dim, ijk)]);
        if (open_y && ijk.y == cell_dim.y - 1)
            inflow += fmaxf(0.0f, -u_y[IjkToIdx(y_tile_dim, { ijk.x, ijk.y + 1, ijk.z })]);
        if (inflow <= 0.0f)
            continue;
        const float frac = fminf(1.0f, inflow * dt_over_dx);
        theta[idx] *= (1.0f - frac);
    }
}

// Seven-point Laplacian of a cell-centred field with the neighbours beyond the
// domain faces clamped onto the centre, i.e. zero-gradient there.
__global__ void DiffuseCentredKernel(float* dst, int3 tile_dim, const float* src, float coef)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    const int3 cell_dim = { tile_dim.x * 8, tile_dim.y * 8, tile_dim.z * 8 };
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        const float c        = src[idx];
        float lap            = 0.0f;
        const int3 offs[6]   = { { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } };
        for (int n = 0; n < 6; n++) {
            const int3 nb = { ijk.x + offs[n].x, ijk.y + offs[n].y, ijk.z + offs[n].z };
            const bool inside = nb.x >= 0 && nb.x < cell_dim.x && nb.y >= 0 && nb.y < cell_dim.y && nb.z >= 0 && nb.z < cell_dim.z;
            lap += (inside ? src[IjkToIdx(tile_dim, nb)] : c) - c;
        }
        dst[idx] = c + coef * lap;
    }
}

} // namespace

void SetPlumeBcAsync(ofm::OFM& solver, const PlumeSpec& spec, cudaStream_t stream)
{
    const int3 td         = solver.tile_dim_;
    const int3 x_tile_dim = { td.x + 1, td.y, td.z };
    // Start from closed free-slip walls everywhere, then overwrite the x faces
    // with the shear profile so that the inflow and outflow carry the same mass.
    const float3 zero = { 0.0f, 0.0f, 0.0f };
    ofm::SetWallBcAsync(*solver.is_bc_x_, *solver.is_bc_y_, *solver.is_bc_z_,
                        *solver.bc_val_x_, *solver.bc_val_y_, *solver.bc_val_z_,
                        td, zero, zero, stream);
    // AdvanceAsync would otherwise overwrite these planes with a uniform inflow.
    solver.use_uniform_inlet_ = false;
    SetPlumeInletXKernel<<<Prod(x_tile_dim), 128, 0, stream>>>(
        solver.is_bc_x_->dev_ptr_, solver.bc_val_x_->dev_ptr_, x_tile_dim,
        solver.grid_origin_, solver.dx_, spec);
    // Open faces: clear the Dirichlet mark, which leaves the exterior pressure
    // pinned at zero (ofm.h). The operator is then non-singular, so the pure
    // Neumann recentering must be off.
    if (spec.outflow == 1 || spec.outflow == 2)
        ofm::SetDomainFaceAsync(*solver.is_bc_x_, *solver.bc_val_x_, td, 0, 1, false, 0.0f, stream);
    if (spec.outflow == 2) {
        ofm::SetDomainFaceAsync(*solver.is_bc_y_, *solver.bc_val_y_, td, 1, 0, false, 0.0f, stream);
        ofm::SetDomainFaceAsync(*solver.is_bc_y_, *solver.bc_val_y_, td, 1, 1, false, 0.0f, stream);
    }
    solver.amgpcg_.pure_neumann_ = !(spec.outflow == 1 || spec.outflow == 2);
    // Convective faces keep their marks; the projection refreshes their values.
    solver.convective_face_[1] = (spec.outflow >= 3);
    solver.convective_face_[2] = solver.convective_face_[3] = (spec.outflow == 4 || spec.outflow == 5);
    // Mode 5: lateral faces convective too, but only the downstream face
    // absorbs the net-flux correction.
    for (int f = 0; f < 6; f++)
        solver.flux_correct_face_[f] = false;
    solver.flux_correct_face_[1] = (spec.outflow == 5);

    ofm::SetCoefByIsBcAsync(*(solver.amgpcg_.poisson_vector_[0].is_dof_),
                            *(solver.amgpcg_.poisson_vector_[0].a_diag_),
                            *(solver.amgpcg_.poisson_vector_[0].a_x_),
                            *(solver.amgpcg_.poisson_vector_[0].a_y_),
                            *(solver.amgpcg_.poisson_vector_[0].a_z_),
                            td, *solver.is_bc_x_, *solver.is_bc_y_, *solver.is_bc_z_, stream);
    solver.amgpcg_.BuildAsync(6.0f, -1.0f, stream);
}

void SetPlumeInitialVelocityAsync(ofm::OFM& solver, const PlumeSpec& spec, cudaStream_t stream)
{
    const int3 td         = solver.tile_dim_;
    const int3 x_tile_dim = { td.x + 1, td.y, td.z };
    const int3 y_tile_dim = { td.x, td.y + 1, td.z };
    const int3 z_tile_dim = { td.x, td.y, td.z + 1 };
    SetPlumeVelocityKernel<<<Prod(x_tile_dim), 128, 0, stream>>>(
        solver.init_u_x_->dev_ptr_, x_tile_dim, 0, solver.grid_origin_, solver.dx_, spec);
    SetPlumeVelocityKernel<<<Prod(y_tile_dim), 128, 0, stream>>>(
        solver.init_u_y_->dev_ptr_, y_tile_dim, 1, solver.grid_origin_, solver.dx_, spec);
    SetPlumeVelocityKernel<<<Prod(z_tile_dim), 128, 0, stream>>>(
        solver.init_u_z_->dev_ptr_, z_tile_dim, 2, solver.grid_origin_, solver.dx_, spec);
}

void AddPlumeHeatAsync(ofm::DHMemory<float>& theta, int3 tile_dim, float3 grid_origin, float dx,
                       const PlumeSpec& spec, float t, float dt, cudaStream_t stream)
{
    AddPlumeHeatKernel<<<Prod(tile_dim), 128, 0, stream>>>(
        theta.dev_ptr_, tile_dim, grid_origin, dx, spec, t, dt);
}

// AdvanceAsync stores the step it has just taken in mid_u_[cycle_len_] and then
// increments cycle_len_, so the cycle index is read differently on the two
// sides of the call. On the first step of a cycle there is no history yet and
// the cycle's start, init_u_, is the answer.
PlumeVelocity PlumeVelocityBefore(ofm::OFM& solver)
{
    const int cycle_step = solver.cycle_len_;
    if (cycle_step == 0)
        return { solver.init_u_x_.get(), solver.init_u_y_.get(), solver.init_u_z_.get() };
    return { solver.mid_u_x_[cycle_step - 1].get(), solver.mid_u_y_[cycle_step - 1].get(),
             solver.mid_u_z_[cycle_step - 1].get() };
}

PlumeVelocity PlumeVelocityAfter(ofm::OFM& solver)
{
    const int cycle_step = solver.cycle_len_ - 1;
    return { solver.mid_u_x_[cycle_step].get(), solver.mid_u_y_[cycle_step].get(),
             solver.mid_u_z_[cycle_step].get() };
}

void SetBuoyancyAndDragAsync(ofm::OFM& solver, const ofm::DHMemory<float>& theta,
                             const PlumeSpec& spec, PlumeVelocity u, float dt, cudaStream_t stream)
{
    const int3 td         = solver.tile_dim_;
    const int3 x_tile_dim = { td.x + 1, td.y, td.z };
    const int3 y_tile_dim = { td.x, td.y + 1, td.z };
    const int3 z_tile_dim = { td.x, td.y, td.z + 1 };

    ofm::GetCenteralVecAsync(*solver.u_, td, *u.x, *u.y, *u.z, stream);

    BuoyancyZKernel<<<Prod(z_tile_dim), 128, 0, stream>>>(
        solver.f_z_->dev_ptr_, z_tile_dim, td, theta.dev_ptr_, spec.g / spec.theta0);
    CanopyDragKernel<<<Prod(x_tile_dim), 128, 0, stream>>>(
        solver.f_x_->dev_ptr_, x_tile_dim, 0, solver.u_->dev_ptr_, td, u.x->dev_ptr_, spec.cd_a);
    CanopyDragKernel<<<Prod(y_tile_dim), 128, 0, stream>>>(
        solver.f_y_->dev_ptr_, y_tile_dim, 1, solver.u_->dev_ptr_, td, u.y->dev_ptr_, spec.cd_a);

    if (spec.sponge) {
        const float lz = td.z * 8 * solver.dx_;
        SpongeVelocityKernel<<<Prod(x_tile_dim), 128, 0, stream>>>(
            solver.f_x_->dev_ptr_, x_tile_dim, 0, u.x->dev_ptr_, solver.grid_origin_, solver.dx_, lz, spec, dt);
        SpongeVelocityKernel<<<Prod(y_tile_dim), 128, 0, stream>>>(
            solver.f_y_->dev_ptr_, y_tile_dim, 1, u.y->dev_ptr_, solver.grid_origin_, solver.dx_, lz, spec, dt);
        SpongeVelocityKernel<<<Prod(z_tile_dim), 128, 0, stream>>>(
            solver.f_z_->dev_ptr_, z_tile_dim, 2, u.z->dev_ptr_, solver.grid_origin_, solver.dx_, lz, spec, dt);
    }
}

void DiffuseThetaAsync(ofm::DHMemory<float>& dst, const ofm::DHMemory<float>& src, int3 tile_dim,
                       float kappa, float dx, float dt, cudaStream_t stream)
{
    DiffuseCentredKernel<<<Prod(tile_dim), 128, 0, stream>>>(dst.dev_ptr_, tile_dim, src.dev_ptr_, kappa * dt / (dx * dx));
}

void AmbientInflowThetaAsync(ofm::DHMemory<float>& theta, int3 tile_dim, PlumeVelocity u, int outflow,
                             float dx, float dt, cudaStream_t stream)
{
    if (outflow <= 0)
        return;
    AmbientInflowThetaKernel<<<Prod(tile_dim), 128, 0, stream>>>(theta.dev_ptr_, tile_dim, u.x->dev_ptr_, u.y->dev_ptr_, outflow, dt / dx);
}

void SpongeThetaAsync(ofm::DHMemory<float>& theta, int3 tile_dim, float3 grid_origin, float dx,
                      const PlumeSpec& spec, float dt, cudaStream_t stream)
{
    const float lz = tile_dim.z * 8 * dx;
    SpongeThetaKernel<<<Prod(tile_dim), 128, 0, stream>>>(theta.dev_ptr_, tile_dim, grid_origin, dx, lz, spec, dt);
}

namespace {
__global__ void DeviceProbeKernel(int* out)
{
    *out = 1;
}
} // namespace

bool CheckDeviceUsable()
{
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        printf("ERROR: no usable CUDA device.\n");
        return false;
    }

    int* probe = nullptr;
    if (cudaMalloc(&probe, sizeof(int)) != cudaSuccess) {
        printf("ERROR: cannot allocate on %s.\n", prop.name);
        return false;
    }
    cudaMemset(probe, 0, sizeof(int));
    DeviceProbeKernel<<<1, 1>>>(probe);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
        err = cudaDeviceSynchronize();
    int host = 0;
    if (err == cudaSuccess)
        err = cudaMemcpy(&host, probe, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(probe);

    if (err != cudaSuccess || host != 1) {
        printf("ERROR: kernels do not run on %s (sm_%d%d): %s.\n"
               "       Add this architecture to add_cugencodes in proj/selfcheck/xmake.lua,\n"
               "       or use gpu-rtx6k (sm_75) or gpu-l40s (sm_89). Every field would have\n"
               "       stayed zero and the run would have reported that as a measurement.\n",
               prop.name, prop.major, prop.minor,
               err != cudaSuccess ? cudaGetErrorString(err) : "the probe kernel wrote nothing");
        return false;
    }
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    return true;
}

namespace {

// Cunningham's Fig. 6 plots the first contour at 300.25 K over a 300 K base and
// steps every 0.25 K, so this single level fixes both the plume outline the
// figure shows and the depth a dip must have before two lobes read as separate
// branches of a bifurcation.
constexpr double kThetaContour = 0.25;

struct ThetaProfile {
    double width      = 0.0;
    double split      = 0.0;
    double left       = 0.0;
    double right      = 0.0;
    double saddle     = 0.0;
    double lower_peak = 0.0;
    bool   bifurcated = false;
};

// Sub-cell peak position from the parabola through a local maximum and its two
// neighbours. Without it every position is a multiple of dx, which at Stage A's
// resolutions quantizes the answer more coarsely than the physics does.
double RefinePeak(const std::vector<double>& p, int j, double y0, double dx)
{
    if (j <= 0 || j + 1 >= static_cast<int>(p.size()))
        return y0 + j * dx;
    const double denom = p[j - 1] - 2.0 * p[j] + p[j + 1];
    if (denom >= 0.0)
        return y0 + j * dx;
    const double shift = 0.5 * (p[j - 1] - p[j + 1]) / denom;
    return y0 + (j + std::max(-0.5, std::min(0.5, shift))) * dx;
}

// Reduce one column-maximum profile P(y) = max_z theta(y,z) to the two numbers
// Fig. 6 is read for: the lateral extent of the first contour, and the
// separation of the two branches of the bifurcation.
ThetaProfile AnalyseThetaProfile(const std::vector<double>& p, double y0, double dx)
{
    ThetaProfile out;
    const int n = static_cast<int>(p.size());
    int lo = -1, hi = -1;
    for (int j = 0; j < n; j++)
        if (p[j] >= kThetaContour) {
            if (lo < 0)
                lo = j;
            hi = j;
        }
    if (lo < 0)
        return out;

    // Interpolate the contour crossing on each flank, so the width is not
    // quantized to the grid the way the omega_z extrema separation was.
    double left_edge  = y0 + lo * dx;
    double right_edge = y0 + hi * dx;
    if (lo > 0 && p[lo] > p[lo - 1])
        left_edge = y0 + (lo - 1 + (kThetaContour - p[lo - 1]) / (p[lo] - p[lo - 1])) * dx;
    if (hi + 1 < n && p[hi] > p[hi + 1])
        right_edge = y0 + (hi + (p[hi] - kThetaContour) / (p[hi] - p[hi + 1])) * dx;
    out.width = right_edge - left_edge;

    // The two tallest local maxima above the contour are the branches. Taking the
    // two outermost instead would let a single grid-scale bump on a flank stand in
    // for a branch, and the saddle test below would then reject the real pair.
    int first = -1, last = -1;
    for (int j = lo; j <= hi; j++) {
        const double l = (j > 0) ? p[j - 1] : -1.0;
        const double r = (j + 1 < n) ? p[j + 1] : -1.0;
        if (!(p[j] > l && p[j] >= r))
            continue;
        if (first < 0 || p[j] > p[first]) {
            last  = first;
            first = j;
        }
        else if (last < 0 || p[j] > p[last]) {
            last = j;
        }
    }
    if (first < 0 || last < 0)
        return out;
    if (first > last)
        std::swap(first, last);

    double saddle = p[first];
    for (int j = first; j <= last; j++)
        saddle = std::min(saddle, p[j]);
    out.saddle    = saddle;
    out.lower_peak = std::min(p[first], p[last]);

    // Fig. 6 contours every 0.25 K from 0.25 K up, so the two lobes are drawn as
    // two separately closed contours exactly when a plotted level falls between
    // the saddle and the lower of the two peaks. That is the figure's own
    // resolution for "bifurcated", and it is weaker than demanding a full
    // interval of relief: peaks at 0.93 K over a 0.68 K saddle close separately
    // at 0.75 K even though the dip is a quarter of a kelvin short of an
    // interval. Requiring the full interval would report that plume as single
    // lobed when the paper's own figure would show it split.
    const double next_contour = std::floor(saddle / kThetaContour + 1.0) * kThetaContour;
    if (next_contour > out.lower_peak)
        return out;
    out.left       = RefinePeak(p, first, y0, dx);
    out.right      = RefinePeak(p, last, y0, dx);
    out.split      = out.right - out.left;
    out.bifurcated = true;
    return out;
}

} // namespace

void AdvectThetaAsync(ofm::DHMemory<float>& dst, ofm::DHMemory<float>& fwd, ofm::DHMemory<float>& err,
                      int3 tile_dim, ofm::DHMemory<float>& src, PlumeVelocity u,
                      float dx, float dt, bool bfecc, bool clamp, cudaStream_t stream)
{
    // The uncorrected step, which is also what the clamp below bounds against --
    // the same role u_ plays in the solver's own BFECC.
    ofm::AdvectN2CAsync(fwd, tile_dim, src, *u.x, *u.y, *u.z, dx, dt, stream);
    if (!bfecc) {
        ofm::AddFieldsAsync(dst, tile_dim, fwd, fwd, 0.0f, stream);
        return;
    }
    // Back one step, difference against where it started: that is the error the
    // forward step made, expressed at the departure points.
    ofm::AdvectN2CAsync(err, tile_dim, fwd, *u.x, *u.y, *u.z, dx, -dt, stream);
    ofm::AddFieldsAsync(err, tile_dim, err, src, -1.0f, stream);
    // Carry the error forward the same way and take off half of it.
    ofm::AdvectN2CAsync(dst, tile_dim, err, *u.x, *u.y, *u.z, dx, dt, stream);
    ofm::AddFieldsAsync(dst, tile_dim, fwd, dst, -0.5f, stream);
    if (clamp) {
        const int3 max_ijk = { tile_dim.x * 8 - 1, tile_dim.y * 8 - 1, tile_dim.z * 8 - 1 };
        ofm::BfeccClampAsync(dst, tile_dim, max_ijk, fwd, stream);
    }
}

PlumeDiag MeasurePlume(ofm::OFM& solver, ofm::DHMemory<float>& theta,
                       float plane_x, float cvp_z, cudaStream_t stream)
{
    const int3 td = solver.tile_dim_;
    ofm::GetCenteralVecAsync(*(solver.u_), td, *solver.init_u_x_, *solver.init_u_y_, *solver.init_u_z_, stream);
    solver.u_->DevToHostAsync(stream);
    theta.DevToHostAsync(stream);
    cudaStreamSynchronize(stream);

    const float dx    = solver.dx_;
    const float3 org  = solver.grid_origin_;
    const int nx      = td.x * 8, ny = td.y * 8, nz = td.z * 8;
    const float* th   = theta.host_ptr_;
    const float3* uc  = solver.u_->host_ptr_;
    auto at = [&](int i, int j, int k) { const int3 ijk = { i, j, k }; return IjkToIdx(td, ijk); };

    PlumeDiag d;
    d.max_theta = 0.0;
    d.w_max     = 0.0;
    d.plume_top = 0.0;
    d.omega_pos = -1.0e30;
    d.omega_neg = 1.0e30;
    d.y_pos = d.y_neg = 0.0;
    d.split_width = 0.0;
    d.best_x = d.best_omega = d.best_split = 0.0;
    d.plane_theta = 0.0;
    d.u_max       = 0.0;
    d.valid       = false;
    d.theta_width = d.theta_split = d.theta_left = d.theta_right = 0.0;
    d.theta_saddle = d.theta_peak = 0.0;
    d.bifurcated  = false;
    d.best_theta_width = d.best_theta_split = 0.0;
    d.best_bifurcated  = false;

    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++) {
                const int id  = at(i, j, k);
                const double t = th[id];
                if (t > d.max_theta)
                    d.max_theta = t;
                const double w = uc[id].z;
                if (w > d.w_max)
                    d.w_max = w;
                if (t > 0.25) {
                    const double pz = org.z + (k + 0.5) * dx;
                    if (pz > d.plume_top)
                        d.plume_top = pz;
                }
            }

    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    const int ip = clampi(static_cast<int>(std::floor((plane_x - org.x) / dx - 0.5)), 1, nx - 2);
    const int kp = clampi(static_cast<int>(std::floor((cvp_z - org.z) / dx - 0.5)), 1, nz - 2);

    d.u_max = 0.0;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++) {
                const float3 v = uc[at(i, j, k)];
                const double sp = std::sqrt(static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y);
                if (sp > d.u_max)
                    d.u_max = sp;
            }

    // Vertical vorticity from central differences on the cell-centred velocity,
    // on every x plane at the same height. The requested plane is reported as
    // asked for; the strongest plane says where the pair actually is.
    d.best_x = d.best_omega = d.best_split = 0.0;
    d.plane_theta = 0.0;
    int best_i    = -1;
    for (int i = 1; i < nx - 1; i++) {
        double wp = -1.0e30, wn = 1.0e30, yp = 0.0, yn = 0.0, th_max = 0.0;
        for (int j = 1; j < ny - 1; j++) {
            const double dvdx = (uc[at(i + 1, j, kp)].y - uc[at(i - 1, j, kp)].y) / (2.0 * dx);
            const double dudy = (uc[at(i, j + 1, kp)].x - uc[at(i, j - 1, kp)].x) / (2.0 * dx);
            const double wz   = dvdx - dudy;
            const double py   = org.y + (j + 0.5) * dx;
            if (wz > wp) { wp = wz; yp = py; }
            if (wz < wn) { wn = wz; yn = py; }
            for (int k = 0; k < nz; k++) {
                const double t = th[at(i, j, k)];
                if (t > th_max)
                    th_max = t;
            }
        }
        const double strength = wp > -wn ? wp : -wn;
        if (wp > 0.0 && wn < 0.0 && strength > d.best_omega) {
            d.best_omega = strength;
            d.best_x     = org.x + (i + 0.5) * dx;
            d.best_split = std::fabs(yp - yn);
            best_i       = i;
        }
        if (i == ip) {
            d.omega_pos   = wp;
            d.omega_neg   = wn;
            d.y_pos       = yp;
            d.y_neg       = yn;
            d.plane_theta = th_max;
        }
    }
    d.split_width = std::fabs(d.y_pos - d.y_neg);
    // A counter-rotating pair needs one sign on each side and enough amplitude to
    // be a structure rather than round-off: the ambient flow carries no omega_z
    // at all, so 1e-3 1/s is already well clear of the noise.
    d.valid = d.omega_pos > 1.0e-3 && d.omega_neg < -1.0e-3;

    // The potential-temperature bifurcation, which is what Fig. 6 shows and what
    // the paper's two ordering results are read off. Reported on the requested
    // plane, and again on the strongest-CVP plane, which is far enough upstream
    // to be clear of the prescribed outflow.
    const double y0 = org.y + 0.5 * dx;
    auto column_max = [&](int i) {
        std::vector<double> p(ny, 0.0);
        for (int j = 0; j < ny; j++) {
            double m = 0.0;
            for (int k = 0; k < nz; k++) {
                const double t = th[at(i, j, k)];
                if (t > m)
                    m = t;
            }
            p[j] = m;
        }
        return p;
    };

    const ThetaProfile pp = AnalyseThetaProfile(column_max(ip), y0, dx);
    d.theta_width  = pp.width;
    d.theta_split  = pp.split;
    d.theta_left   = pp.left;
    d.theta_right  = pp.right;
    d.theta_saddle = pp.saddle;
    d.theta_peak   = pp.lower_peak;
    d.bifurcated   = pp.bifurcated;
    if (best_i >= 0) {
        const ThetaProfile bp = AnalyseThetaProfile(column_max(best_i), y0, dx);
        d.best_theta_width = bp.width;
        d.best_theta_split = bp.split;
        d.best_bifurcated  = bp.bifurcated;
    }
    return d;
}


// ---------------------------------------------------------------------------
// Outflow verification: the translating Gaussian vortex column.

namespace {

__host__ __device__ inline float2 GaussianVortexVelocity(float px, float py, float cx, float cy, float a, float gamma)
{
    const float rx = px - cx;
    const float ry = py - cy;
    const float r2 = rx * rx + ry * ry;
    const float r  = sqrtf(r2);
    if (r < 1e-6f)
        return { 0.0f, 0.0f };
    const float u_theta = gamma / (2.0f * 3.14159265359f * r) * (1.0f - expf(-r2 / (a * a)));
    return { -u_theta * ry / r, u_theta * rx / r };
}

// Stream plus vortex on one staggered axis, everywhere.
__global__ void SetTranslatingVortexKernel(float* u_axis, int3 axis_tile_dim, int component,
                                           float3 origin, float dx, selfcheck::TranslatingVortexSpec spec, float t)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(axis_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    const float cx      = spec.x0 + spec.u_stream * t;
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (component == 2) {
            u_axis[idx] = 0.0f;
            continue;
        }
        const float px = origin.x + (ijk.x + (component == 0 ? 0.0f : 0.5f)) * dx;
        const float py = origin.y + (ijk.y + (component == 1 ? 0.0f : 0.5f)) * dx;
        const float2 v = GaussianVortexVelocity(px, py, cx, spec.y0, spec.core, spec.circulation);
        u_axis[idx]    = component == 0 ? spec.u_stream + v.x : v.y;
    }
}

// The exact normal velocity on the Dirichlet domain faces of one axis at time t.
// x-: stream plus vortex. x+: with outflow off, the stream alone -- the mean
// profile, which is what the plume's closed box prescribes -- and otherwise
// untouched, since the face is open. y-, y+: the vortex alone.
__global__ void SetTranslatingVortexFaceKernel(uint8_t* is_bc_axis, float* bc_val_axis, int3 axis_tile_dim, int component,
                                               float3 origin, float dx, selfcheck::TranslatingVortexSpec spec, float t)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(axis_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    const float cx      = spec.x0 + spec.u_stream * t;
    int3 grid_dim       = { axis_tile_dim.x * 8, axis_tile_dim.y * 8, axis_tile_dim.z * 8 };
    if (component == 0)
        grid_dim.x -= 8;
    else if (component == 1)
        grid_dim.y -= 8;
    else
        grid_dim.z -= 8;
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (component == 0) {
            if (ijk.x == 0) {
                const float py = origin.y + (ijk.y + 0.5f) * dx;
                const float2 v = GaussianVortexVelocity(origin.x, py, cx, spec.y0, spec.core, spec.circulation);
                is_bc_axis[idx]  = 1;
                bc_val_axis[idx] = spec.u_stream + v.x;
            } else if (ijk.x == grid_dim.x) {
                if (spec.outflow == 0) {
                    is_bc_axis[idx]  = 1;
                    bc_val_axis[idx] = spec.u_stream;
                } else if (spec.outflow == 1)
                    is_bc_axis[idx] = 0;
                else
                    is_bc_axis[idx] = 1; // convective: the projection writes the value
            }
        } else if (component == 1) {
            if (ijk.y == 0 || ijk.y == grid_dim.y) {
                const float px = origin.x + (ijk.x + 0.5f) * dx;
                const float py = origin.y + ijk.y * dx;
                const float2 v = GaussianVortexVelocity(px, py, cx, spec.y0, spec.core, spec.circulation);
                is_bc_axis[idx]  = 1;
                bc_val_axis[idx] = v.y;
            }
        }
        // z faces stay the free-slip walls SetWallBcAsync wrote.
    }
}

__global__ void SetShearVortexKernel(float* u_axis, int3 axis_tile_dim, int component,
                                     float3 origin, float dx, selfcheck::ShearVortexSpec spec)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(axis_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (component == 2) {
            u_axis[idx] = 0.0f;
            continue;
        }
        const float px = origin.x + (ijk.x + (component == 0 ? 0.0f : 0.5f)) * dx;
        const float py = origin.y + (ijk.y + (component == 1 ? 0.0f : 0.5f)) * dx;
        const float2 v = GaussianVortexVelocity(px, py, spec.x0, spec.y0, spec.core, spec.circulation);
        u_axis[idx]    = component == 0 ? spec.shear * (py - spec.y0) + v.x : v.y;
    }
}

// x faces: the shear profile alone (the vortex's induced velocity there is the
// image-effect price of a closed box). Other faces keep SetWallBcAsync's walls.
__global__ void SetShearVortexFaceKernel(uint8_t* is_bc_x, float* bc_val_x, int3 x_tile_dim,
                                         float3 origin, float dx, selfcheck::ShearVortexSpec spec)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(x_tile_dim, tile_idx);
    const int t_id      = threadIdx.x;
    const int nx        = x_tile_dim.x * 8 - 8;
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = t_id + i * 128;
        const int idx        = tile_idx * 512 + voxel_idx;
        const int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        const int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (ijk.x == 0 || ijk.x == nx) {
            const float py   = origin.y + (ijk.y + 0.5f) * dx;
            is_bc_x[idx]     = 1;
            bc_val_x[idx]    = spec.shear * (py - spec.y0);
        }
    }
}

} // namespace

void SetShearVortexAsync(ofm::OFM& solver, const ShearVortexSpec& spec, cudaStream_t stream)
{
    const int3 td         = solver.tile_dim_;
    const int3 x_tile_dim = { td.x + 1, td.y, td.z };
    const int3 y_tile_dim = { td.x, td.y + 1, td.z };
    const int3 z_tile_dim = { td.x, td.y, td.z + 1 };
    SetShearVortexKernel<<<Prod(x_tile_dim), 128, 0, stream>>>(solver.init_u_x_->dev_ptr_, x_tile_dim, 0, solver.grid_origin_, solver.dx_, spec);
    SetShearVortexKernel<<<Prod(y_tile_dim), 128, 0, stream>>>(solver.init_u_y_->dev_ptr_, y_tile_dim, 1, solver.grid_origin_, solver.dx_, spec);
    SetShearVortexKernel<<<Prod(z_tile_dim), 128, 0, stream>>>(solver.init_u_z_->dev_ptr_, z_tile_dim, 2, solver.grid_origin_, solver.dx_, spec);
}

void SetShearVortexBcAsync(ofm::OFM& solver, const ShearVortexSpec& spec, cudaStream_t stream)
{
    const int3 td         = solver.tile_dim_;
    const int3 x_tile_dim = { td.x + 1, td.y, td.z };
    const float3 zero     = { 0.0f, 0.0f, 0.0f };
    ofm::SetWallBcAsync(*solver.is_bc_x_, *solver.is_bc_y_, *solver.is_bc_z_,
                        *solver.bc_val_x_, *solver.bc_val_y_, *solver.bc_val_z_, td, zero, zero, stream);
    solver.use_uniform_inlet_ = false;
    SetShearVortexFaceKernel<<<Prod(x_tile_dim), 128, 0, stream>>>(
        solver.is_bc_x_->dev_ptr_, solver.bc_val_x_->dev_ptr_, x_tile_dim, solver.grid_origin_, solver.dx_, spec);
    solver.amgpcg_.pure_neumann_ = true;
    ofm::SetCoefByIsBcAsync(*(solver.amgpcg_.poisson_vector_[0].is_dof_),
                            *(solver.amgpcg_.poisson_vector_[0].a_diag_),
                            *(solver.amgpcg_.poisson_vector_[0].a_x_),
                            *(solver.amgpcg_.poisson_vector_[0].a_y_),
                            *(solver.amgpcg_.poisson_vector_[0].a_z_),
                            td, *solver.is_bc_x_, *solver.is_bc_y_, *solver.is_bc_z_, stream);
    solver.amgpcg_.BuildAsync(6.0f, -1.0f, stream);
}

void SetTranslatingVortexAsync(ofm::OFM& solver, const TranslatingVortexSpec& spec, cudaStream_t stream)
{
    const int3 td         = solver.tile_dim_;
    const int3 x_tile_dim = { td.x + 1, td.y, td.z };
    const int3 y_tile_dim = { td.x, td.y + 1, td.z };
    const int3 z_tile_dim = { td.x, td.y, td.z + 1 };
    SetTranslatingVortexKernel<<<Prod(x_tile_dim), 128, 0, stream>>>(
        solver.init_u_x_->dev_ptr_, x_tile_dim, 0, solver.grid_origin_, solver.dx_, spec, 0.0f);
    SetTranslatingVortexKernel<<<Prod(y_tile_dim), 128, 0, stream>>>(
        solver.init_u_y_->dev_ptr_, y_tile_dim, 1, solver.grid_origin_, solver.dx_, spec, 0.0f);
    SetTranslatingVortexKernel<<<Prod(z_tile_dim), 128, 0, stream>>>(
        solver.init_u_z_->dev_ptr_, z_tile_dim, 2, solver.grid_origin_, solver.dx_, spec, 0.0f);
}

void SetTranslatingVortexBcAsync(ofm::OFM& solver, const TranslatingVortexSpec& spec, float t, bool build, cudaStream_t stream)
{
    const int3 td         = solver.tile_dim_;
    const int3 x_tile_dim = { td.x + 1, td.y, td.z };
    const int3 y_tile_dim = { td.x, td.y + 1, td.z };
    if (build) {
        const float3 zero = { 0.0f, 0.0f, 0.0f };
        ofm::SetWallBcAsync(*solver.is_bc_x_, *solver.is_bc_y_, *solver.is_bc_z_,
                            *solver.bc_val_x_, *solver.bc_val_y_, *solver.bc_val_z_, td, zero, zero, stream);
        solver.use_uniform_inlet_ = false;
    }
    SetTranslatingVortexFaceKernel<<<Prod(x_tile_dim), 128, 0, stream>>>(
        solver.is_bc_x_->dev_ptr_, solver.bc_val_x_->dev_ptr_, x_tile_dim, 0, solver.grid_origin_, solver.dx_, spec, t);
    SetTranslatingVortexFaceKernel<<<Prod(y_tile_dim), 128, 0, stream>>>(
        solver.is_bc_y_->dev_ptr_, solver.bc_val_y_->dev_ptr_, y_tile_dim, 1, solver.grid_origin_, solver.dx_, spec, t);
    if (build) {
        solver.amgpcg_.pure_neumann_ = (spec.outflow != 1);
        solver.convective_face_[1]   = (spec.outflow == 2);
        ofm::SetCoefByIsBcAsync(*(solver.amgpcg_.poisson_vector_[0].is_dof_),
                                *(solver.amgpcg_.poisson_vector_[0].a_diag_),
                                *(solver.amgpcg_.poisson_vector_[0].a_x_),
                                *(solver.amgpcg_.poisson_vector_[0].a_y_),
                                *(solver.amgpcg_.poisson_vector_[0].a_z_),
                                td, *solver.is_bc_x_, *solver.is_bc_y_, *solver.is_bc_z_, stream);
        solver.amgpcg_.BuildAsync(6.0f, -1.0f, stream);
    }
}

TranslatingVortexDiag MeasureTranslatingVortex(ofm::OFM& solver, const TranslatingVortexSpec& spec, float t,
                                               float x_out, float interior_margin, cudaStream_t stream)
{
    const int3 td         = solver.tile_dim_;
    const int3 x_tile_dim = { td.x + 1, td.y, td.z };
    const int3 y_tile_dim = { td.x, td.y + 1, td.z };
    const int3 z_tile_dim = { td.x, td.y, td.z + 1 };
    solver.init_u_x_->DevToHostAsync(stream);
    solver.init_u_y_->DevToHostAsync(stream);
    solver.init_u_z_->DevToHostAsync(stream);
    cudaStreamSynchronize(stream);
    const float* ux = solver.init_u_x_->host_ptr_;
    const float* uy = solver.init_u_y_->host_ptr_;
    const float* uz = solver.init_u_z_->host_ptr_;

    const int nx_box = td.x * 8, ny = td.y * 8, nz = td.z * 8;
    const double dx = solver.dx_;
    const float3 org = solver.grid_origin_;
    // Corners with x < x_out: i from 1 to nx - 1 where nx * dx = x_out.
    const int nx     = std::min(nx_box, static_cast<int>(std::lround((x_out - org.x) / dx)));
    const double lx = nx * dx, ly = ny * dx;
    const double cx = spec.x0 + spec.u_stream * t;
    const double cy = spec.y0;
    const double a  = spec.core;
    const double pi = 3.14159265358979323846;

    TranslatingVortexDiag d;
    d.valid            = true;
    d.gamma_in         = 0.0;
    d.gamma_core       = 0.0;
    d.gamma_walls = d.gamma_inflow = d.gamma_elsewhere = 0.0;
    d.min_omega        = 1.0e30;
    d.peak_omega       = -1.0e30;
    const double half  = 6.0 * a;
    d.peak_x = d.peak_y = 0.0;
    d.max_w            = 0.0;
    double err_all = 0.0, err_int = 0.0;

    for (int i = 1; i < nx; i++)
        for (int j = 1; j < ny; j++) {
            const double px = org.x + i * dx;
            const double py = org.y + j * dx;
            double w = 0.0;
            for (int k = 0; k < nz; k++) {
                const float vy_r = uy[IjkToIdx(y_tile_dim, { i, j, k })];
                const float vy_l = uy[IjkToIdx(y_tile_dim, { i - 1, j, k })];
                const float ux_u = ux[IjkToIdx(x_tile_dim, { i, j, k })];
                const float ux_d = ux[IjkToIdx(x_tile_dim, { i, j - 1, k })];
                w += ((vy_r - vy_l) - (ux_u - ux_d)) / dx;
            }
            w /= nz;
            if (!std::isfinite(w)) {
                d.valid = false;
                return d;
            }
            const double rx = px - cx, ry = py - cy;
            const double w_exact = spec.circulation / (pi * a * a) * std::exp(-(rx * rx + ry * ry) / (a * a));
            d.gamma_in += w * dx * dx;
            const bool in_core = std::fabs(rx) < half && std::fabs(ry) < half;
            if (in_core)
                d.gamma_core += w * dx * dx;
            else if (j <= 2 || j >= ny - 2)
                d.gamma_walls += w * dx * dx;
            else if (i <= 2)
                d.gamma_inflow += w * dx * dx;
            else
                d.gamma_elsewhere += w * dx * dx;
            if (w < d.min_omega)
                d.min_omega = w;
            const double e2 = (w - w_exact) * (w - w_exact) * dx * dx;
            err_all += e2;
            if (px < lx - interior_margin)
                err_int += e2;
            if (w > d.peak_omega) {
                d.peak_omega = w;
                d.peak_x     = px;
                d.peak_y     = py;
            }
        }
    for (int k = 0; k <= nz; k++)
        for (int i = 0; i < nx_box; i++)
            for (int j = 0; j < ny; j++) {
                const double w = std::fabs(uz[IjkToIdx(z_tile_dim, { i, j, k })]);
                if (w > d.max_w)
                    d.max_w = w;
            }

    // The Gaussian's integral over the corner rectangle [dx, L-dx] x [dx, Ly-dx].
    const double ex = 0.5 * (std::erf((lx - dx - cx) / a) - std::erf((dx - cx) / a));
    const double ey = 0.5 * (std::erf((ly - dx - cy) / a) - std::erf((dx - cy) / a));
    d.gamma_in_exact   = spec.circulation * ex * ey;
    {
        // The core box, clipped to the corner rectangle.
        const double bx0 = std::max(dx, cx - half), bx1 = std::min(lx - dx, cx + half);
        const double by0 = std::max(dx, cy - half), by1 = std::min(ly - dx, cy + half);
        const double cex = bx1 > bx0 ? 0.5 * (std::erf((bx1 - cx) / a) - std::erf((bx0 - cx) / a)) : 0.0;
        const double cey = by1 > by0 ? 0.5 * (std::erf((by1 - cy) / a) - std::erf((by0 - cy) / a)) : 0.0;
        d.gamma_core_exact = spec.circulation * cex * cey;
    }
    d.peak_omega_exact = spec.circulation / (pi * a * a);
    // ||omega_exact||_2 over the plane: Gamma / (a sqrt(2 pi)).
    const double norm0 = spec.circulation / (a * std::sqrt(2.0 * pi));
    d.l2_all      = std::sqrt(err_all) / norm0;
    d.l2_interior = std::sqrt(err_int) / norm0;
    return d;
}


void WritePlumeSlice(FILE* f, ofm::OFM& solver, ofm::DHMemory<float>& theta,
                     float plane_x, float time, bool header, cudaStream_t stream)
{
    theta.DevToHostAsync(stream);
    cudaStreamSynchronize(stream);
    const int3 td    = solver.tile_dim_;
    const float dx   = solver.dx_;
    const float3 org = solver.grid_origin_;
    const int nx = td.x * 8, ny = td.y * 8, nz = td.z * 8;
    int ip = static_cast<int>(std::lround((plane_x - org.x) / dx - 0.5f));
    ip = std::max(0, std::min(nx - 1, ip));
    if (header) {
        const char magic[8] = { 'O', 'F', 'M', 'S', 'L', 'I', 'C', 'E' };
        fwrite(magic, 1, 8, f);
        const int32_t dims[2] = { ny, nz };
        fwrite(dims, sizeof(int32_t), 2, f);
        const float geo[4] = { dx, org.y + 0.5f * dx, org.z + 0.5f * dx, org.x + (ip + 0.5f) * dx };
        fwrite(geo, sizeof(float), 4, f);
    }
    fwrite(&time, sizeof(float), 1, f);
    std::vector<float> row(nz);
    const float* th = theta.host_ptr_;
    for (int j = 0; j < ny; j++) {
        for (int k = 0; k < nz; k++)
            row[k] = th[IjkToIdx(td, { ip, j, k })];
        fwrite(row.data(), sizeof(float), nz, f);
    }
    fflush(f);
}


// Largest velocity-gradient component in the domain, from the staggered
// velocity: along-axis differences of neighbouring faces for the diagonal
// entries, differences of neighbouring faces across the other two axes for the
// off-diagonal ones (both over one dx). This bounds the strain the flow map
// accumulates per unit time, which is what the adaptive reinitialization
// integrates: ||F - I|| <= exp(int S dt) - 1.
__global__ void MaxVelocityGradientKernel(unsigned int* _out, int3 _tile_dim, const float* _u_x, const float* _u_y, const float* _u_z, float _inv_dx)
{
    const int tile_idx  = blockIdx.x;
    const int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    const int3 xd = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    const int3 yd = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    const int3 zd = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    const int nx = _tile_dim.x * 8, ny = _tile_dim.y * 8, nz = _tile_dim.z * 8;
    float best = 0.0f;
    for (int i = 0; i < 4; i++) {
        const int voxel_idx  = threadIdx.x + i * 128;
        const int3 v         = VoxelIdxToIjk(voxel_idx);
        const int3 c         = { tile_ijk.x * 8 + v.x, tile_ijk.y * 8 + v.y, tile_ijk.z * 8 + v.z };
        const int3 cx = { c.x + 1, c.y, c.z }, cy = { c.x, c.y + 1, c.z }, cz = { c.x, c.y, c.z + 1 };
        // diagonal: face differences across the cell
        const float ux = _u_x[IjkToIdx(xd, c)], uxr = _u_x[IjkToIdx(xd, cx)];
        const float uy = _u_y[IjkToIdx(yd, c)], uyu = _u_y[IjkToIdx(yd, cy)];
        const float uz = _u_z[IjkToIdx(zd, c)], uzf = _u_z[IjkToIdx(zd, cz)];
        best = fmaxf(best, fabsf(uxr - ux));
        best = fmaxf(best, fabsf(uyu - uy));
        best = fmaxf(best, fabsf(uzf - uz));
        // off-diagonal: the same face against its neighbour along the other axes
        if (c.y + 1 < ny) best = fmaxf(best, fabsf(_u_x[IjkToIdx(xd, { c.x, c.y + 1, c.z })] - ux));
        if (c.z + 1 < nz) best = fmaxf(best, fabsf(_u_x[IjkToIdx(xd, { c.x, c.y, c.z + 1 })] - ux));
        if (c.x + 1 < nx) best = fmaxf(best, fabsf(_u_y[IjkToIdx(yd, { c.x + 1, c.y, c.z })] - uy));
        if (c.z + 1 < nz) best = fmaxf(best, fabsf(_u_y[IjkToIdx(yd, { c.x, c.y, c.z + 1 })] - uy));
        if (c.x + 1 < nx) best = fmaxf(best, fabsf(_u_z[IjkToIdx(zd, { c.x + 1, c.y, c.z })] - uz));
        if (c.y + 1 < ny) best = fmaxf(best, fabsf(_u_z[IjkToIdx(zd, { c.x, c.y + 1, c.z })] - uz));
    }
    best *= _inv_dx;
    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ typename BlockReduce::TempStorage temp;
    const float block_max = BlockReduce(temp).Reduce(best, cub::Max());
    if (threadIdx.x == 0)
        atomicMax(_out, __float_as_uint(fmaxf(block_max, 0.0f)));
}

float MaxVelocityGradient(ofm::OFM& solver, const PlumeVelocity& u, cudaStream_t stream)
{
    static unsigned int* d_out = nullptr;
    if (!d_out)
        cudaMalloc(&d_out, sizeof(unsigned int));
    cudaMemsetAsync(d_out, 0, sizeof(unsigned int), stream);
    MaxVelocityGradientKernel<<<Prod(solver.tile_dim_), 128, 0, stream>>>(
        d_out, solver.tile_dim_, u.x->dev_ptr_, u.y->dev_ptr_, u.z->dev_ptr_, 1.0f / solver.dx_);
    unsigned int bits = 0;
    cudaMemcpyAsync(&bits, d_out, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    float v;
    std::memcpy(&v, &bits, sizeof(float));
    return v;
}


void WritePlumeProfile(FILE* f, ofm::OFM& solver, ofm::DHMemory<float>& theta,
                       float x0, float x1, float y0, float y1, float time)
{
    const int3 td    = solver.tile_dim_;
    const float dx   = solver.dx_;
    const float3 org = solver.grid_origin_;
    const int nx = td.x * 8, ny = td.y * 8, nz = td.z * 8;
    const float* th  = theta.host_ptr_;
    const float3* uc = solver.u_->host_ptr_;
    for (int k = 0; k < nz; k++) {
        double sw = 0.0, st = 0.0, wmax = -1e30;
        int n = 0;
        for (int i = 0; i < nx; i++) {
            const float px = org.x + (i + 0.5f) * dx;
            if (px < x0 || px > x1)
                continue;
            for (int j = 0; j < ny; j++) {
                const float py = org.y + (j + 0.5f) * dx;
                if (py < y0 || py > y1)
                    continue;
                const int id = IjkToIdx(td, { i, j, k });
                sw += uc[id].z;
                st += th[id];
                wmax = std::max(wmax, double(uc[id].z));
                n++;
            }
        }
        fprintf(f, "%.1f %.1f %.5f %.5f %.5f\n", time, org.z + (k + 0.5f) * dx, n ? sw / n : 0.0, n ? st / n : 0.0, n ? wmax : 0.0);
    }
    fflush(f);
}


float MaxDivergence(ofm::OFM& solver, cudaStream_t stream)
{
    const int3 td = solver.tile_dim_;
    const int3 xd = { td.x + 1, td.y, td.z }, yd = { td.x, td.y + 1, td.z }, zd = { td.x, td.y, td.z + 1 };
    solver.init_u_x_->DevToHostAsync(stream);
    solver.init_u_y_->DevToHostAsync(stream);
    solver.init_u_z_->DevToHostAsync(stream);
    cudaStreamSynchronize(stream);
    const float* ux = solver.init_u_x_->host_ptr_;
    const float* uy = solver.init_u_y_->host_ptr_;
    const float* uz = solver.init_u_z_->host_ptr_;
    const int nx = td.x * 8, ny = td.y * 8, nz = td.z * 8;
    const float inv_dx = 1.0f / solver.dx_;
    float best = 0.0f;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++) {
                const float d = (ux[IjkToIdx(xd, { i + 1, j, k })] - ux[IjkToIdx(xd, { i, j, k })]
                               + uy[IjkToIdx(yd, { i, j + 1, k })] - uy[IjkToIdx(yd, { i, j, k })]
                               + uz[IjkToIdx(zd, { i, j, k + 1 })] - uz[IjkToIdx(zd, { i, j, k })]) * inv_dx;
                best = std::max(best, std::fabs(d));
            }
    return best;
}

} // namespace selfcheck
