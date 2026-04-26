#include "ofm.h"
#include "ofm_init.h"
#include "ofm_util.h"

#include <cuda_runtime.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "core/config/config.h"
#include "data_io.h"

namespace {

struct HeadlessOptions {
    int steps = 100;
    int save_interval = 10;
    std::string output_dir = "./output";
};

HeadlessOptions ParseArgs(int argc, char** argv)
{
    HeadlessOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--steps" && i + 1 < argc) {
            options.steps = std::stoi(argv[++i]);
        } else if (arg == "--save_interval" && i + 1 < argc) {
            options.save_interval = std::stoi(argv[++i]);
        } else if (arg == "--output_dir" && i + 1 < argc) {
            options.output_dir = argv[++i];
        }
    }

    if (options.steps < 0) {
        options.steps = 0;
    }
    if (options.save_interval <= 0) {
        options.save_interval = 1;
    }

    return options;
}

std::string BuildVorticityPath(const std::string& output_dir, int step)
{
    std::ostringstream oss;
    oss << output_dir << "/vorticity_" << std::setfill('0') << std::setw(6) << step << ".npy";
    return oss.str();
}

}

int main(int argc, char** argv) {
    HeadlessOptions options = ParseArgs(argc, argv);

    cudaStream_t stream{};
    cudaStreamCreate(&stream);

    Configuration config = load("./config/dynamic_obstacle.json");

    OFMConfiguration ofm_cfg = static_cast<OFMConfiguration>(config.at("ofm"));
    ofm_cfg.use_dynamic_solid = false;

    ofm::OFM sim;
    ofm::InitOFMAsync(sim, ofm_cfg, stream);

    std::filesystem::create_directories(options.output_dir);

    auto save_vorticity = [&](int step) {
        ofm::GetCenteralVecAsync(*(sim.u_), sim.tile_dim_, *(sim.init_u_x_), *(sim.init_u_y_), *(sim.init_u_z_), stream);
        ofm::GetVorNormAsync(*(sim.vor_norm_), sim.tile_dim_, *(sim.u_), sim.dx_, stream);
        sim.vor_norm_->DevToHostAsync(stream);
        cudaStreamSynchronize(stream);

        const int nx = sim.tile_dim_.x * 8;
        const int ny = sim.tile_dim_.y * 8;
        const int nz = sim.tile_dim_.z * 8;

        WriteNpy<float>(BuildVorticityPath(options.output_dir, step), sim.vor_norm_->host_ptr_, {nx, ny, nz});
    };

    float dt = 1.0f / 24.0f;

    save_vorticity(0);

    for (int i = 0; i < options.steps; ++i) {
        sim.AdvanceAsync(dt, stream);
        sim.ReinitAsync(dt, stream);

        const int step = i + 1;
        if (step % options.save_interval == 0) {
            save_vorticity(step);
        } else {
            cudaStreamSynchronize(stream);
        }

        std::cout << "step " << step << " done" << std::endl;
    }

    cudaStreamDestroy(stream);
    return 0;
}