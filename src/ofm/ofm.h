#pragma once

#include "amgpcg.h"
#include "timer.h"

#include <vector>

namespace ofm {
class OFM {
public:
    // domain
    int3 tile_dim_;
    float dx_;
    float3 grid_origin_;

    // simulation parameters
    int step_;
    // Number of steps in one reinitialization cycle. 1 reproduces the one-step
    // (OFM) scheme exactly; larger values restore the LFM cycle, which keeps the
    // projected velocity of every step in the cycle and marches the flow map
    // through that history at reinitialization.
    int reinit_every_ = 1;
    // Order of the flow-map marching scheme: 2, 4, or anything else for TVD-RK3
    // (the order OFM shipped with, kept as the default).
    int rk_order_ = 3;

    // boundary
    float inlet_norm_;
    float inlet_angle_;
    std::shared_ptr<DHMemory<uint8_t>> is_bc_x_;
    std::shared_ptr<DHMemory<uint8_t>> is_bc_y_;
    std::shared_ptr<DHMemory<uint8_t>> is_bc_z_;
    std::shared_ptr<DHMemory<float>> bc_val_x_;
    std::shared_ptr<DHMemory<float>> bc_val_y_;
    std::shared_ptr<DHMemory<float>> bc_val_z_;

    // backward flow map
    std::shared_ptr<DHMemory<float3>> T_x_;
    std::shared_ptr<DHMemory<float3>> T_y_;
    std::shared_ptr<DHMemory<float3>> T_z_;
    std::shared_ptr<DHMemory<float3>> psi_x_;
    std::shared_ptr<DHMemory<float3>> psi_y_;
    std::shared_ptr<DHMemory<float3>> psi_z_;

    /// forward flow map
    std::shared_ptr<DHMemory<float3>> F_x_;
    std::shared_ptr<DHMemory<float3>> F_y_;
    std::shared_ptr<DHMemory<float3>> F_z_;
    std::shared_ptr<DHMemory<float3>> phi_x_;
    std::shared_ptr<DHMemory<float3>> phi_y_;
    std::shared_ptr<DHMemory<float3>> phi_z_;

    // velocity storage
    std::shared_ptr<DHMemory<float3>> u_;
    std::shared_ptr<DHMemory<float>> u_x_;
    std::shared_ptr<DHMemory<float>> u_y_;
    std::shared_ptr<DHMemory<float>> u_z_;
    std::shared_ptr<DHMemory<float>> init_u_x_;
    std::shared_ptr<DHMemory<float>> init_u_y_;
    std::shared_ptr<DHMemory<float>> init_u_z_;
    std::shared_ptr<DHMemory<float>> tmp_u_x_;
    std::shared_ptr<DHMemory<float>> tmp_u_y_;
    std::shared_ptr<DHMemory<float>> tmp_u_z_;
    std::shared_ptr<DHMemory<float>> err_u_x_;
    std::shared_ptr<DHMemory<float>> err_u_y_;
    std::shared_ptr<DHMemory<float>> err_u_z_;
    // One entry per step of the reinitialization cycle.
    std::vector<std::shared_ptr<DHMemory<float>>> mid_u_x_;
    std::vector<std::shared_ptr<DHMemory<float>>> mid_u_y_;
    std::vector<std::shared_ptr<DHMemory<float>>> mid_u_z_;

    // vorticity
    std::shared_ptr<DHMemory<float>> vor_norm_;

    // solver
    AMGPCG amgpcg_;

    // Source terms: viscosity and external force, carried the way LFM's
    // Algorithm 1 carries them. Off by default, which leaves the solver exactly
    // as it shipped -- inviscid, with no path integral evaluated.
    bool use_source_term_ = false;
    float viscosity_      = 0.0f; // kinematic, i.e. mu/rho
    // External force per unit mass. Zeroed by Alloc; write into it to add
    // buoyancy, baroclinic generation or drag.
    std::shared_ptr<DHMemory<float>> f_x_;
    std::shared_ptr<DHMemory<float>> f_y_;
    std::shared_ptr<DHMemory<float>> f_z_;
    // Total source nu*lap(u) + f, rebuilt whenever it is needed.
    std::shared_ptr<DHMemory<float>> src_x_;
    std::shared_ptr<DHMemory<float>> src_y_;
    std::shared_ptr<DHMemory<float>> src_z_;
    // The advected velocity with the source already added (u-dagger in Algorithm 1).
    std::shared_ptr<DHMemory<float>> star_u_x_;
    std::shared_ptr<DHMemory<float>> star_u_y_;
    std::shared_ptr<DHMemory<float>> star_u_z_;
    // One quadrature sample of the Eq. (8) path integral, per staggered axis.
    std::shared_ptr<DHMemory<float>> s_axis_x_;
    std::shared_ptr<DHMemory<float>> s_axis_y_;
    std::shared_ptr<DHMemory<float>> s_axis_z_;

    // bfecc clamp
    bool use_bfecc_clamp_;

    // voxelized solid
    bool use_dynamic_solid_;
    cudaSurfaceObject_t voxel_tex_;
    cudaSurfaceObject_t velocity_tex_;
    float voxelized_velocity_scaler_;

    // profiler
    GPUTimer* profiler_ = nullptr;

    OFM() = default;
    OFM(int3 _tile_dim, int _reinit_every = 1);
    void Alloc(int3 _tile_dim, int _reinit_every = 1);
    void SetProfilier(GPUTimer* _profiler);
    void UpdateBoundary(cudaStream_t _stream);
    void AdvanceAsync(float _dt, cudaStream_t _stream);
    void ReinitAsync(float _dt, cudaStream_t _stream);
    void ResetForwardFlowMapAsync(cudaStream_t _stream);
    void ResetBackwardFlowMapAsync(cudaStream_t _stream);
    void ProjectAsync(cudaStream_t _stream);
    // Fills src_{x,y,z}_ with nu*lap(u) + f for the velocity passed in.
    void ComputeSourceAsync(const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z, cudaStream_t _stream);
};
}
