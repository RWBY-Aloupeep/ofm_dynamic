#pragma once

#include "ofm.h"

void AddPlumeSourceAsync(
    ofm::OFM& solver,
    float plume_strength,
    float plume_radius,
    float swirl_strength,
    cudaStream_t stream);
