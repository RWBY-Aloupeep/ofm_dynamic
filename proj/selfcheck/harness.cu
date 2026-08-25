#include "harness.h"

#include "ofm_util.h"
#include "util.h"

#include <cmath>
#include <cstdio>
#include <cstring>

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

} // namespace

void SetupSolver(ofm::OFM& solver, const SolverConfig& config, GPUTimer& profiler, cudaStream_t stream)
{
    solver.Alloc(config.tile_dim, config.reinit_every);
    solver.SetProfilier(&profiler);

    solver.step_        = 0;
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

} // namespace selfcheck
