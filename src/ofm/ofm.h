#pragma once

#include "amgpcg.h"
#include "timer.h"

#include <vector>

namespace ofm {

// Source channels. They are integrated together for the physics, but kept apart
// so that the circulation budget can say how much of Gamma each one contributed.
// Adding a channel means adding an entry here and a case in ComputeSourceChannelAsync.
enum SourceChannel : int {
    kChanViscous  = 0, // nu * lap(u), built from the velocity
    kChanExternal = 1, // f_{x,y,z}_, written by the caller
    kChanNum      = 2
};

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
    // AdvanceAsync rewrites the inlet and outlet planes of bc_val_{x,y}_ from
    // inlet_norm_/inlet_angle_ on every step, which only expresses a uniform
    // inflow. Set this false to keep whatever the caller wrote there instead --
    // a sheared profile, say. Default true, so existing cases are unchanged.
    bool use_uniform_inlet_ = true;
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

    // Circulation attribution (D2). When on, each source channel's contribution to
    // the impulse is accumulated separately over the cycle, in the frame of the
    // cycle's start. A line integral of acc_[k] around a material loop's preimage
    // is that channel's contribution to the loop's circulation over the cycle.
    // Reset at the start of every cycle; the physics is unaffected either way.
    bool track_attribution_ = false;
    std::vector<std::shared_ptr<DHMemory<float>>> acc_x_;
    std::vector<std::shared_ptr<DHMemory<float>>> acc_y_;
    std::vector<std::shared_ptr<DHMemory<float>>> acc_z_;

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
    // Fills src_{x,y,z}_ with a single channel's contribution to the source.
    void ComputeSourceChannelAsync(int _channel, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z, cudaStream_t _stream);
};
}
