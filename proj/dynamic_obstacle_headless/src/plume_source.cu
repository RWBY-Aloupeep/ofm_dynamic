#include <algorithm>
#include "plume_source.h"

#include <cmath>

namespace {
__global__ void AddPlumeSourceKernel(
    float3* u,
    int nx,
    int ny,
    int nz,
    float plume_strength,
    float radius,
    float swirl_strength,
    float cx,
    float cy,
    float cz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = nx * ny * nz;
    if (idx >= count) {
        return;
    }

    const int yz = ny * nz;
    const int x = idx / yz;
    const int rem = idx % yz;
    const int y = rem / nz;
    const int z = rem % nz;

    const float dx = static_cast<float>(x) - cx;
    const float dy = static_cast<float>(y) - cy;
    const float dz = static_cast<float>(z) - cz;
    const float r2 = dx * dx + dy * dy + dz * dz;
    const float w = expf(-r2 / (radius * radius));

    float3 vel = u[idx];
    vel.y += plume_strength * w;
    vel.x += -swirl_strength * dz * w;
    vel.z += swirl_strength * dx * w;
    u[idx] = vel;
}
}

void AddPlumeSourceAsync(
    ofm::OFM& solver,
    float plume_strength,
    float plume_radius,
    float swirl_strength,
    cudaStream_t stream)
{
    const int nx = solver.tile_dim_.x * 8;
    const int ny = solver.tile_dim_.y * 8;
    const int nz = solver.tile_dim_.z * 8;

    const int min_dim = std::min(nx, std::min(ny, nz));
    const float radius = fmaxf(plume_radius * static_cast<float>(min_dim), 1e-4f);
    const float cx = 0.5f * static_cast<float>(nx);
    const float cy = 0.15f * static_cast<float>(ny);
    const float cz = 0.5f * static_cast<float>(nz);

    const int count = nx * ny * nz;
    constexpr int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    AddPlumeSourceKernel<<<grid_size, block_size, 0, stream>>>(
        solver.u_->dev_ptr_,
        nx,
        ny,
        nz,
        plume_strength,
        radius,
        swirl_strength,
        cx,
        cy,
        cz);
}
