#include "ofm.h"
#include "ofm_util.h"
#include <cub/cub.cuh>

namespace ofm {
OFM::OFM(int3 _tile_dim, int _reinit_every)
{
    Alloc(_tile_dim, _reinit_every);
}

void OFM::Alloc(int3 _tile_dim, int _reinit_every)
{
    tile_dim_     = _tile_dim;
    reinit_every_ = _reinit_every < 1 ? 1 : _reinit_every;

    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    int voxel_num   = Prod(tile_dim_) * 512;
    int x_voxel_num = Prod(x_tile_dim) * 512;
    int y_voxel_num = Prod(y_tile_dim) * 512;
    int z_voxel_num = Prod(z_tile_dim) * 512;

    // boundary
    if (flux_sum_ == nullptr)
        cudaMalloc(&flux_sum_, 2 * sizeof(double));
    is_bc_x_  = std::make_shared<DHMemory<uint8_t>>(x_voxel_num);
    is_bc_y_  = std::make_shared<DHMemory<uint8_t>>(y_voxel_num);
    is_bc_z_  = std::make_shared<DHMemory<uint8_t>>(z_voxel_num);
    bc_val_x_ = std::make_shared<DHMemory<float>>(x_voxel_num);
    bc_val_y_ = std::make_shared<DHMemory<float>>(y_voxel_num);
    bc_val_z_ = std::make_shared<DHMemory<float>>(z_voxel_num);

    // backward flow map
    T_x_   = std::make_shared<DHMemory<float3>>(x_voxel_num);
    T_y_   = std::make_shared<DHMemory<float3>>(y_voxel_num);
    T_z_   = std::make_shared<DHMemory<float3>>(z_voxel_num);
    psi_x_ = std::make_shared<DHMemory<float3>>(x_voxel_num);
    psi_y_ = std::make_shared<DHMemory<float3>>(y_voxel_num);
    psi_z_ = std::make_shared<DHMemory<float3>>(z_voxel_num);

    // forward flow map
    F_x_   = std::make_shared<DHMemory<float3>>(x_voxel_num);
    F_y_   = std::make_shared<DHMemory<float3>>(y_voxel_num);
    F_z_   = std::make_shared<DHMemory<float3>>(z_voxel_num);
    phi_x_ = std::make_shared<DHMemory<float3>>(x_voxel_num);
    phi_y_ = std::make_shared<DHMemory<float3>>(y_voxel_num);
    phi_z_ = std::make_shared<DHMemory<float3>>(z_voxel_num);

    // velocity storage
    u_        = std::make_shared<DHMemory<float3>>(voxel_num);
    u_x_      = std::make_shared<DHMemory<float>>(x_voxel_num);
    u_y_      = std::make_shared<DHMemory<float>>(y_voxel_num);
    u_z_      = std::make_shared<DHMemory<float>>(z_voxel_num);
    init_u_x_ = std::make_shared<DHMemory<float>>(x_voxel_num);
    init_u_y_ = std::make_shared<DHMemory<float>>(y_voxel_num);
    init_u_z_ = std::make_shared<DHMemory<float>>(z_voxel_num);
    mid_u_x_.resize(reinit_every_);
    mid_u_y_.resize(reinit_every_);
    mid_u_z_.resize(reinit_every_);
    for (int i = 0; i < reinit_every_; i++) {
        mid_u_x_[i] = std::make_shared<DHMemory<float>>(x_voxel_num);
        mid_u_y_[i] = std::make_shared<DHMemory<float>>(y_voxel_num);
        mid_u_z_[i] = std::make_shared<DHMemory<float>>(z_voxel_num);
    }
    tmp_u_x_  = std::make_shared<DHMemory<float>>(x_voxel_num);
    tmp_u_y_  = std::make_shared<DHMemory<float>>(y_voxel_num);
    tmp_u_z_  = std::make_shared<DHMemory<float>>(z_voxel_num);
    err_u_x_  = std::make_shared<DHMemory<float>>(x_voxel_num);
    err_u_y_  = std::make_shared<DHMemory<float>>(y_voxel_num);
    err_u_z_  = std::make_shared<DHMemory<float>>(z_voxel_num);

    // source terms
    f_x_      = std::make_shared<DHMemory<float>>(x_voxel_num);
    f_y_      = std::make_shared<DHMemory<float>>(y_voxel_num);
    f_z_      = std::make_shared<DHMemory<float>>(z_voxel_num);
    src_x_    = std::make_shared<DHMemory<float>>(x_voxel_num);
    src_y_    = std::make_shared<DHMemory<float>>(y_voxel_num);
    src_z_    = std::make_shared<DHMemory<float>>(z_voxel_num);
    star_u_x_ = std::make_shared<DHMemory<float>>(x_voxel_num);
    star_u_y_ = std::make_shared<DHMemory<float>>(y_voxel_num);
    star_u_z_ = std::make_shared<DHMemory<float>>(z_voxel_num);
    s_axis_x_ = std::make_shared<DHMemory<float>>(x_voxel_num);
    s_axis_y_ = std::make_shared<DHMemory<float>>(y_voxel_num);
    s_axis_z_ = std::make_shared<DHMemory<float>>(z_voxel_num);
    acc_x_.resize(kChanNum);
    acc_y_.resize(kChanNum);
    acc_z_.resize(kChanNum);
    for (int k = 0; k < kChanNum; k++) {
        acc_x_[k] = std::make_shared<DHMemory<float>>(x_voxel_num);
        acc_y_[k] = std::make_shared<DHMemory<float>>(y_voxel_num);
        acc_z_[k] = std::make_shared<DHMemory<float>>(z_voxel_num);
    }
    // The force field is read every step once source terms are on, so it must
    // start at zero rather than at whatever the allocation happened to contain.
    cudaMemset(f_x_->dev_ptr_, 0, x_voxel_num * sizeof(float));
    cudaMemset(f_y_->dev_ptr_, 0, y_voxel_num * sizeof(float));
    cudaMemset(f_z_->dev_ptr_, 0, z_voxel_num * sizeof(float));

    // vorticity
    vor_norm_ = std::make_shared<DHMemory<float>>(voxel_num);

    // solver
    int min_dim   = tile_dim_.x;
    min_dim       = min_dim > tile_dim_.y ? tile_dim_.y : min_dim;
    min_dim       = min_dim > tile_dim_.z ? tile_dim_.z : min_dim;
    int level_num = (int)log2(min_dim) + 1;
    if (max_level_num_ > 0 && level_num > max_level_num_)
        level_num = max_level_num_;
    amgpcg_.Alloc(_tile_dim, level_num);
}

void OFM::SetProfilier(GPUTimer* _profiler) {
    profiler_ = _profiler;
}


void OFM::UpdateBoundary(cudaStream_t _stream)
{
    if (use_dynamic_solid_) {
        {
            CUDA_PROFILE_SCOPE(*profiler_, _stream, "UpdateBoundaryCondition");
            SetBcBySurfaceAsync(*is_bc_x_, *is_bc_y_, *is_bc_z_, *bc_val_x_, *bc_val_y_, *bc_val_z_, tile_dim_, voxel_tex_, velocity_tex_, voxelized_velocity_scaler_, _stream);
            SetCoefByIsBcAsync(*(amgpcg_.poisson_vector_[0].is_dof_), *(amgpcg_.poisson_vector_[0].a_diag_), *(amgpcg_.poisson_vector_[0].a_x_), *(amgpcg_.poisson_vector_[0].a_y_),
                                       *(amgpcg_.poisson_vector_[0].a_z_), tile_dim_, *is_bc_x_, *is_bc_y_, *is_bc_z_, _stream);
        }

        {
            CUDA_PROFILE_SCOPE(*profiler_, _stream, "Rebuild Projection Matrix")
            amgpcg_.BuildAsync(6.0f, -1.0f, _stream);
        }
    }
}

void OFM::AdvanceAsync(float _dt, cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };

    // Leapfrog schedule, following Algorithm 1 of the LFM paper: the first two steps
    // of a reinitialization cycle start the integrator (a half step, then a full
    // step), and every step after that advects the velocity from two steps back
    // across 2*dt using the velocity of the previous step. With reinit_every_ == 1
    // only the first branch is ever reached, which is exactly the one-step scheme
    // OFM shipped; the leapfrog steps proper require a cycle of at least three.
    int cycle_step = cycle_len_;
    float mid_dt;
    std::shared_ptr<DHMemory<float>> last_proj_u_x;
    std::shared_ptr<DHMemory<float>> last_proj_u_y;
    std::shared_ptr<DHMemory<float>> last_proj_u_z;
    std::shared_ptr<DHMemory<float>> src_u_x;
    std::shared_ptr<DHMemory<float>> src_u_y;
    std::shared_ptr<DHMemory<float>> src_u_z;
    if (cycle_step == 0) {
        mid_dt        = 0.5f * _dt;
        last_proj_u_x = init_u_x_;
        last_proj_u_y = init_u_y_;
        last_proj_u_z = init_u_z_;
        src_u_x       = init_u_x_;
        src_u_y       = init_u_y_;
        src_u_z       = init_u_z_;
    } else if (cycle_step == 1) {
        mid_dt        = _dt;
        last_proj_u_x = mid_u_x_[0];
        last_proj_u_y = mid_u_y_[0];
        last_proj_u_z = mid_u_z_[0];
        src_u_x       = mid_u_x_[0];
        src_u_y       = mid_u_y_[0];
        src_u_z       = mid_u_z_[0];
    } else {
        mid_dt        = 2.0f * _dt;
        last_proj_u_x = mid_u_x_[cycle_step - 1];
        last_proj_u_y = mid_u_y_[cycle_step - 1];
        last_proj_u_z = mid_u_z_[cycle_step - 1];
        src_u_x       = mid_u_x_[cycle_step - 2];
        src_u_y       = mid_u_y_[cycle_step - 2];
        src_u_z       = mid_u_z_[cycle_step - 2];
    }

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Advection");
        // Algorithm 1 lines 1, 6 and 12: the advected velocity picks up the source
        // over the same interval it is advected across, evaluated on the velocity
        // that transports it. All three of those lines are this one expression.
        if (use_source_term_) {
            ComputeSourceAsync(*last_proj_u_x, *last_proj_u_y, *last_proj_u_z, _stream);
            AddFieldsAsync(*star_u_x_, x_tile_dim, *src_u_x, *src_x_, mid_dt, _stream);
            AddFieldsAsync(*star_u_y_, y_tile_dim, *src_u_y, *src_y_, mid_dt, _stream);
            AddFieldsAsync(*star_u_z_, z_tile_dim, *src_u_z, *src_z_, mid_dt, _stream);
            src_u_x = star_u_x_;
            src_u_y = star_u_y_;
            src_u_z = star_u_z_;
        }
        AdvectN2XAsync(*tmp_u_x_, tile_dim_, *src_u_x, *last_proj_u_x, *last_proj_u_y, *last_proj_u_z, dx_, mid_dt, _stream);
        AdvectN2YAsync(*tmp_u_y_, tile_dim_, *src_u_y, *last_proj_u_x, *last_proj_u_y, *last_proj_u_z, dx_, mid_dt, _stream);
        AdvectN2ZAsync(*tmp_u_z_, tile_dim_, *src_u_z, *last_proj_u_x, *last_proj_u_y, *last_proj_u_z, dx_, mid_dt, _stream);
    }

    if (use_uniform_inlet_)
        SetInletAsync(*bc_val_x_, *bc_val_y_, tile_dim_, inlet_norm_, inlet_angle_, _stream);

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Projection 1");
        ProjectAsync(_stream);
    }

    mid_u_x_[cycle_step].swap(tmp_u_x_);
    mid_u_y_[cycle_step].swap(tmp_u_y_);
    mid_u_z_[cycle_step].swap(tmp_u_z_);

    step_++;
    cycle_len_++;
}

void OFM::ReinitAsync(float _dt, cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };

    ResetForwardFlowMapAsync(_stream);
    ResetBackwardFlowMapAsync(_stream);

    // The accumulators live in the frame of this cycle's start, so they reset with
    // the map. Their running total across cycles is the caller's to keep: it is a
    // scalar per material loop, not a field.
    if (use_source_term_ && track_attribution_)
        for (int k = 0; k < kChanNum; k++) {
            acc_x_[k]->ClearDevAsync(_stream);
            acc_y_[k]->ClearDevAsync(_stream);
            acc_z_[k]->ClearDevAsync(_stream);
        }

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Marching Backward flowmap");
        // Walk the cycle's velocity history backwards in time.
        const int cycle = cycle_len_ > 0 ? cycle_len_ : reinit_every_;
        for (int i = cycle - 1; i >= 0; i--) {
            RKAxisAsync(rk_order_, *psi_x_, *T_x_, tile_dim_, x_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, _dt, _stream);
            RKAxisAsync(rk_order_, *psi_y_, *T_y_, tile_dim_, y_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, _dt, _stream);
            RKAxisAsync(rk_order_, *psi_z_, *T_z_, tile_dim_, z_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, _dt, _stream);
        }
    }

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Marching Forward flowmap");
        // ... and forwards for the forward map, so the two meet at the cycle's ends.
        //
        // With source terms on, this loop also evaluates the path integral of
        // Eq. (8) -- Algorithm 1 lines 5, 10 and 16. The quadrature points are the
        // step midpoints, so each step is marched in two halves and the source is
        // contracted with the forward Jacobian in between, then accumulated into
        // the initial-time impulse. That accumulation has to happen here, before
        // the pullback below reads init_u_.
        const float half_dt = 0.5f * _dt;
        const int cycle = cycle_len_ > 0 ? cycle_len_ : reinit_every_;
        for (int i = 0; i < cycle; i++) {
            if (!use_source_term_) {
                RKAxisAsync(rk_order_, *phi_x_, *F_x_, tile_dim_, x_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -_dt, _stream);
                RKAxisAsync(rk_order_, *phi_y_, *F_y_, tile_dim_, y_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -_dt, _stream);
                RKAxisAsync(rk_order_, *phi_z_, *F_z_, tile_dim_, z_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -_dt, _stream);
                continue;
            }

            RKAxisAsync(rk_order_, *phi_x_, *F_x_, tile_dim_, x_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -half_dt, _stream);
            RKAxisAsync(rk_order_, *phi_y_, *F_y_, tile_dim_, y_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -half_dt, _stream);
            RKAxisAsync(rk_order_, *phi_z_, *F_z_, tile_dim_, z_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -half_dt, _stream);

            // Without attribution the channels are summed once and contracted once.
            // With it, each channel is contracted on its own so that its share of the
            // impulse -- and so of the circulation -- can be read off separately. The
            // sum reaching init_u_ is identical either way.
            const int channel_num = track_attribution_ ? kChanNum : 1;
            for (int k = 0; k < channel_num; k++) {
                if (track_attribution_)
                    ComputeSourceChannelAsync(k, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], _stream);
                else
                    ComputeSourceAsync(*mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], _stream);

                ContractSourceAxisAsync(*s_axis_x_, tile_dim_, x_tile_dim, *phi_x_, *F_x_, *src_x_, *src_y_, *src_z_, grid_origin_, dx_, _stream);
                ContractSourceAxisAsync(*s_axis_y_, tile_dim_, y_tile_dim, *phi_y_, *F_y_, *src_x_, *src_y_, *src_z_, grid_origin_, dx_, _stream);
                ContractSourceAxisAsync(*s_axis_z_, tile_dim_, z_tile_dim, *phi_z_, *F_z_, *src_x_, *src_y_, *src_z_, grid_origin_, dx_, _stream);

                AddFieldsAsync(*init_u_x_, x_tile_dim, *init_u_x_, *s_axis_x_, _dt, _stream);
                AddFieldsAsync(*init_u_y_, y_tile_dim, *init_u_y_, *s_axis_y_, _dt, _stream);
                AddFieldsAsync(*init_u_z_, z_tile_dim, *init_u_z_, *s_axis_z_, _dt, _stream);

                if (track_attribution_) {
                    AddFieldsAsync(*acc_x_[k], x_tile_dim, *acc_x_[k], *s_axis_x_, _dt, _stream);
                    AddFieldsAsync(*acc_y_[k], y_tile_dim, *acc_y_[k], *s_axis_y_, _dt, _stream);
                    AddFieldsAsync(*acc_z_[k], z_tile_dim, *acc_z_[k], *s_axis_z_, _dt, _stream);
                }
            }

            RKAxisAsync(rk_order_, *phi_x_, *F_x_, tile_dim_, x_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -half_dt, _stream);
            RKAxisAsync(rk_order_, *phi_y_, *F_y_, tile_dim_, y_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -half_dt, _stream);
            RKAxisAsync(rk_order_, *phi_z_, *F_z_, tile_dim_, z_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -half_dt, _stream);
        }
    }

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Impulse reconstruction");
        PullbackAxisAsync(*u_x_, tile_dim_, x_tile_dim, *init_u_x_, *init_u_y_, *init_u_z_, *psi_x_, *T_x_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*u_y_, tile_dim_, y_tile_dim, *init_u_x_, *init_u_y_, *init_u_z_, *psi_y_, *T_y_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*u_z_, tile_dim_, z_tile_dim, *init_u_x_, *init_u_y_, *init_u_z_, *psi_z_, *T_z_, grid_origin_, dx_, _stream);
    }

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "BFECC");
        PullbackAxisAsync(*err_u_x_, tile_dim_, x_tile_dim, *u_x_, *u_y_, *u_z_, *phi_x_, *F_x_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*err_u_y_, tile_dim_, y_tile_dim, *u_x_, *u_y_, *u_z_, *phi_y_, *F_y_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*err_u_z_, tile_dim_, z_tile_dim, *u_x_, *u_y_, *u_z_, *phi_z_, *F_z_, grid_origin_, dx_, _stream);
        AddFieldsAsync(*err_u_x_, x_tile_dim, *err_u_x_, *init_u_x_, -1.0f, _stream);
        AddFieldsAsync(*err_u_y_, y_tile_dim, *err_u_y_, *init_u_y_, -1.0f, _stream);
        AddFieldsAsync(*err_u_z_, z_tile_dim, *err_u_z_, *init_u_z_, -1.0f, _stream);
        PullbackAxisAsync(*init_u_x_, tile_dim_, x_tile_dim, *err_u_x_, *err_u_y_, *err_u_z_, *psi_x_, *T_x_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*init_u_y_, tile_dim_, y_tile_dim, *err_u_x_, *err_u_y_, *err_u_z_, *psi_y_, *T_y_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*init_u_z_, tile_dim_, z_tile_dim, *err_u_x_, *err_u_y_, *err_u_z_, *psi_z_, *T_z_, grid_origin_, dx_, _stream);
        AddFieldsAsync(*tmp_u_x_, x_tile_dim, *u_x_, *init_u_x_, -0.5f, _stream);
        AddFieldsAsync(*tmp_u_y_, y_tile_dim, *u_y_, *init_u_y_, -0.5f, _stream);
        AddFieldsAsync(*tmp_u_z_, z_tile_dim, *u_z_, *init_u_z_, -0.5f, _stream);
        if (use_bfecc_clamp_) {
            int3 x_max_ijk = { tile_dim_.x * 8, tile_dim_.y * 8 - 1, tile_dim_.z * 8 - 1 };
            int3 y_max_ijk = { tile_dim_.x * 8 - 1, tile_dim_.y * 8, tile_dim_.z * 8 - 1 };
            int3 z_max_ijk = { tile_dim_.x * 8 - 1, tile_dim_.y * 8 - 1, tile_dim_.z * 8 };
            BfeccClampAsync(*tmp_u_x_, x_tile_dim, x_max_ijk, *u_x_, _stream);
            BfeccClampAsync(*tmp_u_y_, y_tile_dim, y_max_ijk, *u_y_, _stream);
            BfeccClampAsync(*tmp_u_z_, z_tile_dim, z_max_ijk, *u_z_, _stream);
        }
    }

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Projection 2");
        ProjectAsync(_stream);
    }

    init_u_x_.swap(tmp_u_x_);
    init_u_y_.swap(tmp_u_y_);
    init_u_z_.swap(tmp_u_z_);
    cycle_len_ = 0;
}

void OFM::ResetForwardFlowMapAsync(cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    ResetToIdentityXASync(*phi_x_, *F_x_, x_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityYASync(*phi_y_, *F_y_, y_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityZASync(*phi_z_, *F_z_, z_tile_dim, grid_origin_, dx_, _stream);
}

void OFM::ResetBackwardFlowMapAsync(cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    ResetToIdentityXASync(*psi_x_, *T_x_, x_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityYASync(*psi_y_, *T_y_, y_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityZASync(*psi_z_, *T_z_, z_tile_dim, grid_origin_, dx_, _stream);
}

void OFM::ComputeSourceAsync(const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z, cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    int3 x_max_ijk  = { tile_dim_.x * 8, tile_dim_.y * 8 - 1, tile_dim_.z * 8 - 1 };
    int3 y_max_ijk  = { tile_dim_.x * 8 - 1, tile_dim_.y * 8, tile_dim_.z * 8 - 1 };
    int3 z_max_ijk  = { tile_dim_.x * 8 - 1, tile_dim_.y * 8 - 1, tile_dim_.z * 8 };

    LaplacianAxisAsync(*src_x_, x_tile_dim, x_max_ijk, _u_x, dx_, viscosity_, _stream);
    LaplacianAxisAsync(*src_y_, y_tile_dim, y_max_ijk, _u_y, dx_, viscosity_, _stream);
    LaplacianAxisAsync(*src_z_, z_tile_dim, z_max_ijk, _u_z, dx_, viscosity_, _stream);

    // src = f + nu * lap(u). Writing back into src_ is safe: AddFieldsKernel
    // reads and writes the same index.
    AddFieldsAsync(*src_x_, x_tile_dim, *f_x_, *src_x_, 1.0f, _stream);
    AddFieldsAsync(*src_y_, y_tile_dim, *f_y_, *src_y_, 1.0f, _stream);
    AddFieldsAsync(*src_z_, z_tile_dim, *f_z_, *src_z_, 1.0f, _stream);
}

void OFM::ComputeSourceChannelAsync(int _channel, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z, cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    int3 x_max_ijk  = { tile_dim_.x * 8, tile_dim_.y * 8 - 1, tile_dim_.z * 8 - 1 };
    int3 y_max_ijk  = { tile_dim_.x * 8 - 1, tile_dim_.y * 8, tile_dim_.z * 8 - 1 };
    int3 z_max_ijk  = { tile_dim_.x * 8 - 1, tile_dim_.y * 8 - 1, tile_dim_.z * 8 };

    if (_channel == kChanViscous) {
        LaplacianAxisAsync(*src_x_, x_tile_dim, x_max_ijk, _u_x, dx_, viscosity_, _stream);
        LaplacianAxisAsync(*src_y_, y_tile_dim, y_max_ijk, _u_y, dx_, viscosity_, _stream);
        LaplacianAxisAsync(*src_z_, z_tile_dim, z_max_ijk, _u_z, dx_, viscosity_, _stream);
    } else {
        // The external force is already the source; copy it so that the caller can
        // contract src_ whichever channel it asked for.
        cudaMemcpyAsync(src_x_->dev_ptr_, f_x_->dev_ptr_, Prod(x_tile_dim) * 512 * sizeof(float), cudaMemcpyDeviceToDevice, _stream);
        cudaMemcpyAsync(src_y_->dev_ptr_, f_y_->dev_ptr_, Prod(y_tile_dim) * 512 * sizeof(float), cudaMemcpyDeviceToDevice, _stream);
        cudaMemcpyAsync(src_z_->dev_ptr_, f_z_->dev_ptr_, Prod(z_tile_dim) * 512 * sizeof(float), cudaMemcpyDeviceToDevice, _stream);
    }
}

void OFM::ProjectAsync(cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };

    bool any_convective = false;
    for (int f = 0; f < 6; f++)
        any_convective = any_convective || convective_face_[f];
    if (any_convective) {
        DHMemory<float>* bc_val[3] = { bc_val_x_.get(), bc_val_y_.get(), bc_val_z_.get() };
        DHMemory<float>* tmp_u[3]  = { tmp_u_x_.get(), tmp_u_y_.get(), tmp_u_z_.get() };
        for (int f = 0; f < 6; f++)
            if (convective_face_[f])
                ConvectiveFaceUpdateAsync(*bc_val[f / 2], *tmp_u[f / 2], tile_dim_, f / 2, f % 2, _stream);
        bool any_absorb = false;
        for (int f = 0; f < 6; f++)
            any_absorb = any_absorb || flux_correct_face_[f];
        bool absorb[6];
        for (int f = 0; f < 6; f++)
            absorb[f] = any_absorb ? (flux_correct_face_[f] && convective_face_[f]) : convective_face_[f];
        cudaMemsetAsync(flux_sum_, 0, 2 * sizeof(double), _stream);
        // The flux is summed over every face; the count -- the area the
        // correction is spread over -- only over the faces that absorb it.
        DomainFluxAsync(flux_sum_, tile_dim_, *is_bc_x_, *is_bc_y_, *is_bc_z_, *bc_val_x_, *bc_val_y_, *bc_val_z_,
                        *tmp_u_x_, *tmp_u_y_, *tmp_u_z_, absorb, _stream);
        for (int f = 0; f < 6; f++)
            if (absorb[f])
                ConvectiveFaceCorrectAsync(*bc_val[f / 2], tile_dim_, f / 2, f % 2, flux_sum_, _stream);
    }

    SetBcAxisAsync(*tmp_u_x_, x_tile_dim, *is_bc_x_, *bc_val_x_, _stream);
    SetBcAxisAsync(*tmp_u_y_, y_tile_dim, *is_bc_y_, *bc_val_y_, _stream);
    SetBcAxisAsync(*tmp_u_z_, z_tile_dim, *is_bc_z_, *bc_val_z_, _stream);

    CalcDivAsync(*(amgpcg_.b_), tile_dim_, *(amgpcg_.poisson_vector_[0].is_dof_), *tmp_u_x_, *tmp_u_y_, *tmp_u_z_, _stream);

    amgpcg_.SolveAsync(_stream);

    ApplyPressureAsync(*tmp_u_x_, *tmp_u_y_, *tmp_u_z_, tile_dim_, *(amgpcg_.x_), *is_bc_x_, *is_bc_y_, *is_bc_z_, _stream);
}
}
