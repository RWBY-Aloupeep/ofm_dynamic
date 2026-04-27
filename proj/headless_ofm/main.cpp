#include "ofm.h"
#include "ofm_init.h"

#include <cuda_runtime.h>
#include <iostream>

#include "core/config/config.h"

int main() {
    cudaStream_t stream{};
    cudaStreamCreate(&stream);

    Configuration config = load("./config/dynamic_obstacle.json");

    OFMConfiguration ofm_cfg = static_cast<OFMConfiguration>(config.at("ofm"));

    ofm_cfg.use_dynamic_solid = false;

    ofm::OFM sim;
    ofm::InitOFMAsync(sim, ofm_cfg, stream);

    int total_steps = 100;
    float dt = 1.0f / 24.0f; 

    for (int i = 0; i < total_steps; ++i) {
        sim.AdvanceAsync(dt, stream);
        sim.ReinitAsync(dt, stream);

        cudaStreamSynchronize(stream);
        std::cout << "step " << i + 1 << " done" << std::endl;
    }

    cudaStreamDestroy(stream);
    return 0;
}