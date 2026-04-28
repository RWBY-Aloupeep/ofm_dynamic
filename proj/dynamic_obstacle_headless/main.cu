#include "data_io.h"
#include "ofm.h"
#include "ofm_init.h"
#include "ofm_util.h"

#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct HeadlessOptions {
    int steps = 1000;
    int save_interval = 10;
    std::string output_dir = "./outputs/headless";
    int device = 0;
    int3 resolution = { 256, 128, 128 };
    float inlet_norm = 0.05f;
    float inlet_angle = 90.0f;
    float voxelized_velocity_scaler = 1.8f;
    float len_y = 1.0f;
    bool use_bfecc_clamp = true;
};

int ParseIntArg(const char* value, const std::string& arg_name)
{
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error("Invalid integer for " + arg_name + ": " + value);
    }
}

void PrintUsage(const char* program)
{
    std::cout << "Usage: " << program << " [options]\n"
              << "Options:\n"
              << "  --steps <int>           Number of simulation steps (default: 1000)\n"
              << "  --save_interval <int>   Save vorticity every N steps (default: 10)\n"
              << "  --output_dir <path>     Output directory for .npy files\n"
              << "  --resolution <x,y,z>    Grid resolution in voxels (default: 256,128,128)\n"
              << "  --device <int>          CUDA device index (default: 0)\n"
              << "  --help                  Print this help\n";
}

int3 ParseResolutionArg(const std::string& value)
{
    int3 resolution = { 0, 0, 0 };
    char comma0, comma1;
    std::stringstream ss(value);
    ss >> resolution.x >> comma0 >> resolution.y >> comma1 >> resolution.z;
    if (!ss || comma0 != ',' || comma1 != ',' || resolution.x <= 0 || resolution.y <= 0 || resolution.z <= 0) {
        throw std::runtime_error("Invalid --resolution format. Expected x,y,z with positive integers.");
    }
    if (resolution.x % 8 != 0 || resolution.y % 8 != 0 || resolution.z % 8 != 0) {
        throw std::runtime_error("Resolution must be divisible by 8 in each dimension.");
    }
    return resolution;
}

HeadlessOptions ParseOptions(int argc, char** argv)
{
    HeadlessOptions opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for argument: " + arg);
        }

        const char* value = argv[++i];
        if (arg == "--steps") {
            opts.steps = ParseIntArg(value, arg);
        } else if (arg == "--save_interval") {
            opts.save_interval = ParseIntArg(value, arg);
        } else if (arg == "--output_dir") {
            opts.output_dir = value;
        } else if (arg == "--resolution") {
            opts.resolution = ParseResolutionArg(value);
        } else if (arg == "--device") {
            opts.device = ParseIntArg(value, arg);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (opts.steps <= 0) {
        throw std::runtime_error("--steps must be > 0");
    }
    if (opts.save_interval <= 0) {
        throw std::runtime_error("--save_interval must be > 0");
    }
    return opts;
}

OFMConfiguration BuildOFMConfiguration(const HeadlessOptions& options)
{
    OFMConfiguration cfg;
    cfg.len_y = options.len_y;
    cfg.tile_dim = { options.resolution.x / 8, options.resolution.y / 8, options.resolution.z / 8 };
    cfg.grid_origin = { 0.0f, 0.0f, 0.0f };
    cfg.inlet_norm = options.inlet_norm;
    cfg.inlet_angle = options.inlet_angle;
    cfg.voxelized_velocity_scaler = options.voxelized_velocity_scaler;
    cfg.use_bfecc_clamp = options.use_bfecc_clamp;
    cfg.use_static_solid = false;
    cfg.use_dynamic_solid = false;
    cfg.solid_sdf_path = "";
    return cfg;
}

void WriteNpyFloat32(const fs::path& output_path, const float* data, int nx, int ny, int nz)
{
    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + output_path.string());
    }

    constexpr char magic[] = "\x93NUMPY";
    constexpr unsigned char major = 1;
    constexpr unsigned char minor = 0;
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    header += std::to_string(nx) + ", " + std::to_string(ny) + ", " + std::to_string(nz) + "), }";
    const std::size_t preamble_size = 6 + 2 + 2;
    const std::size_t unpadded_size = preamble_size + header.size() + 1;
    const std::size_t padding = (16 - (unpadded_size % 16)) % 16;
    header.append(padding, ' ');
    header.push_back('\n');

    const std::uint16_t header_len = static_cast<std::uint16_t>(header.size());

    out.write(magic, 6);
    out.put(static_cast<char>(major));
    out.put(static_cast<char>(minor));
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    const std::size_t count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
    if (!out) {
        throw std::runtime_error("Failed to write npy data to: " + output_path.string());
    }
}

void SaveVorticityField(const ofm::OFM& solver, const fs::path& output_dir, int step, cudaStream_t stream)
{
    ofm::GetCenteralVecAsync(*(solver.u_), solver.tile_dim_, *(solver.init_u_x_), *(solver.init_u_y_), *(solver.init_u_z_), stream);
    ofm::GetVorNormAsync(*(solver.vor_norm_), solver.tile_dim_, *(solver.u_), solver.dx_, stream);
    solver.vor_norm_->DevToHostAsync(stream);
    cudaStreamSynchronize(stream);

    const int nx = solver.tile_dim_.x * 8;
    const int ny = solver.tile_dim_.y * 8;
    const int nz = solver.tile_dim_.z * 8;

    std::ostringstream filename;
    filename << "vorticity_" << std::setfill('0') << std::setw(6) << step << ".npy";
    const fs::path output_path = output_dir / filename.str();

    WriteNpyFloat32(output_path, solver.vor_norm_->host_ptr_, nx, ny, nz);
    std::cout << "[OFM Headless] Saved vorticity to " << output_path.string() << "\n";
}

int main(int argc, char** argv)
{
    try {
        const HeadlessOptions options = ParseOptions(argc, argv);

        cudaSetDevice(options.device);
        fs::create_directories(options.output_dir);

        ofm::OFM solver;
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        const OFMConfiguration cfg = BuildOFMConfiguration(options);
        ofm::InitOFMAsync(solver, cfg, stream);
        cudaStreamSynchronize(stream);

        const auto wall_start = std::chrono::steady_clock::now();
        std::cout << "[OFM Headless] Simulation starts\n";
        std::cout << "[OFM Headless] Resolution: " << options.resolution.x << "x" << options.resolution.y << "x" << options.resolution.z << "\n";
        std::cout << "[OFM Headless] Steps: " << options.steps << ", save interval: " << options.save_interval << "\n";

        const float dt = 1.0f / 30.0f;
        for (int step = 0; step < options.steps; ++step) {
            solver.inlet_angle_ = options.inlet_angle;
            solver.inlet_norm_ = options.inlet_norm;
            solver.voxelized_velocity_scaler_ = options.voxelized_velocity_scaler;

            if (step > 0) {
                solver.UpdateBoundary(stream);
            }
            solver.AdvanceAsync(dt, stream);
            solver.ReinitAsync(dt, stream);

            if (step % options.save_interval == 0) {
                SaveVorticityField(solver, options.output_dir, step, stream);
            }

            if (step % 50 == 0 || step + 1 == options.steps) {
                std::cout << "[OFM Headless] Step " << step + 1 << " / " << options.steps << "\n";
            }
        }

        cudaStreamSynchronize(stream);
        const auto wall_end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(wall_end - wall_start).count();
        std::cout << "[OFM Headless] Finished. Total runtime: " << seconds << " s\n";

        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[OFM Headless] Error: " << e.what() << "\n";
        return 1;
    }
}
