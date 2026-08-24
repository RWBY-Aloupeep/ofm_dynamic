// Headless smoke test: bring up the OFM solver with no renderer attached.
#include "ofm.h"
#include "ofm_util.h"
#include "timer.h"
#include "util.h"

#include <cstdio>

int main()
{
    cudaStream_t stream = 0;

    ofm::OFM solver;
    const int3 tile_dim = { 8, 8, 8 }; // 64^3 cells
    solver.Alloc(tile_dim);

    GPUTimer profiler(64);
    solver.SetProfilier(&profiler);

    solver.step_        = 0;
    solver.dx_          = 1.0f / static_cast<float>(8 * tile_dim.y);
    solver.grid_origin_ = { 0.0f, 0.0f, 0.0f };
    solver.inlet_norm_  = 1.0f;
    solver.inlet_angle_ = 0.0f;

    solver.use_bfecc_clamp_  = true;
    solver.use_dynamic_solid_ = false;

    solver.init_u_x_->ClearDevAsync(stream);
    solver.init_u_y_->ClearDevAsync(stream);
    solver.init_u_z_->ClearDevAsync(stream);

    const float3 neg_bc_val = { solver.inlet_norm_, 0.0f, 0.0f };
    const float3 pos_bc_val = neg_bc_val;
    ofm::SetWallBcAsync(*solver.is_bc_x_, *solver.is_bc_y_, *solver.is_bc_z_,
                        *solver.bc_val_x_, *solver.bc_val_y_, *solver.bc_val_z_,
                        tile_dim, neg_bc_val, pos_bc_val, stream);

    ofm::SetCoefByIsBcAsync(*(solver.amgpcg_.poisson_vector_[0].is_dof_),
                            *(solver.amgpcg_.poisson_vector_[0].a_diag_),
                            *(solver.amgpcg_.poisson_vector_[0].a_x_),
                            *(solver.amgpcg_.poisson_vector_[0].a_y_),
                            *(solver.amgpcg_.poisson_vector_[0].a_z_),
                            tile_dim, *solver.is_bc_x_, *solver.is_bc_y_, *solver.is_bc_z_, stream);
    solver.amgpcg_.BuildAsync(6.0f, -1.0f, stream);
    solver.amgpcg_.solve_by_tol_ = false;
    solver.amgpcg_.max_iter_     = 6;

    cudaStreamSynchronize(stream);
    printf("setup complete\n");

    const float dt = 1.0f / 60.0f;
    for (int step = 0; step < 3; step++) {
        profiler.beginFrame();
        solver.AdvanceAsync(dt, stream);
        solver.ReinitAsync(dt, stream);
        cudaStreamSynchronize(stream);

        const cudaError_t err = cudaGetLastError();
        printf("step %d: %s\n", step, cudaGetErrorString(err));
        if (err != cudaSuccess)
            return 1;
    }

    printf("headless smoke test passed\n");
    return 0;
}
