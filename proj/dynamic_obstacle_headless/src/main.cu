#include "data_io.h"
#include "ofm.h"
#include "ofm_init.h"
#include "ofm_util.h"
#include "plume_source.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
static std::string last_stage;
static int3 last_resolution = { 0, 0, 0 };
static int3 last_tile_dim = { 0, 0, 0 };
static std::size_t last_nx = 0;
static std::size_t last_ny = 0;
static std::size_t last_nz = 0;
static std::size_t last_count = 0;

static void SetStage(const std::string& s)
{
    last_stage = s;
    std::cerr << "[STAGE] " << s << std::endl;
}

struct HeadlessOptions {
    int steps = 1000;
    int save_interval = 10;
    std::string output_dir = "outputs";
    int device = 0;
    int3 resolution = { 128, 128, 128 };
    float inlet_norm = 0.05f;
    float inlet_angle = 90.0f;
    float voxelized_velocity_scaler = 1.8f;
    float len_y = 1.0f;
    bool use_bfecc_clamp = true;
    bool plume = true;
    float plume_strength = 0.2f;
    float plume_radius = 0.12f;
    float swirl_strength = 0.05f;
};

int ParseIntArg(const char* value, const std::string& arg_name)
{
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error("Invalid integer for " + arg_name + ": " + value);
    }
}

std::string BuildTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm {};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string FormatFloatTag(float value)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    std::string s = oss.str();
    while (!s.empty() && s.back() == '0') {
        s.pop_back();
    }
    if (!s.empty() && s.back() == '.') {
        s.pop_back();
    }
    return s.empty() ? "0" : s;
}

std::optional<std::string> TryGetGitCommitHash()
{
    FILE* pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
    if (!pipe) {
        return std::nullopt;
    }
    char buffer[128];
    std::string result;
    if (fgets(buffer, sizeof(buffer), pipe)) {
        result = buffer;
    }
    pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    if (result.empty()) {
        return std::nullopt;
    }
    return result;
}

void WriteConfigJson(const fs::path& config_path, const HeadlessOptions& options, const OFMConfiguration& cfg, const fs::path& run_dir, const std::string& timestamp)
{
    std::ofstream out(config_path);
    if (!out) {
        throw std::runtime_error("Failed to open config file: " + config_path.string());
    }
    out << "{\n";
    out << "  \"resolution\": [" << options.resolution.x << ", " << options.resolution.y << ", " << options.resolution.z << "],\n";
    out << "  \"tile_dim\": [" << cfg.tile_dim[0] << ", " << cfg.tile_dim[1] << ", " << cfg.tile_dim[2] << "],\n";
    out << "  \"steps\": " << options.steps << ",\n";
    out << "  \"save_interval\": " << options.save_interval << ",\n";
    out << "  \"device\": " << options.device << ",\n";
    out << "  \"plume_strength\": " << options.plume_strength << ",\n";
    out << "  \"plume_radius\": " << options.plume_radius << ",\n";
    out << "  \"swirl_strength\": " << options.swirl_strength << ",\n";
    out << "  \"output_dir\": \"" << options.output_dir << "\",\n";
    out << "  \"run_dir\": \"" << run_dir.string() << "\",\n";
    out << "  \"timestamp\": \"" << timestamp << "\"";
    const std::optional<std::string> commit_hash = TryGetGitCommitHash();
    if (commit_hash.has_value()) {
        out << ",\n  \"git_commit_hash\": \"" << commit_hash.value() << "\"\n";
    } else {
        out << "\n";
    }
    out << "}\n";
}

float ParseFloatArg(const char* value, const std::string& arg_name)
{
    try {
        return std::stof(value);
    } catch (...) {
        throw std::runtime_error("Invalid float for " + arg_name + ": " + value);
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
              << "  --plume                 Enable plume source forcing (default: off)\n"
              << "  --plume_strength <f>    Upward plume strength (default: 0.2)\n"
              << "  --plume_radius <f>      Plume radius as domain fraction (default: 0.12)\n"
              << "  --swirl_strength <f>    Swirl strength around y-axis (default: 0.05)\n"
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
        if (arg == "--plume") {
            opts.plume = true;
            continue;
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
        } else if (arg == "--plume_strength") {
            opts.plume_strength = ParseFloatArg(value, arg);
        } else if (arg == "--plume_radius") {
            opts.plume_radius = ParseFloatArg(value, arg);
        } else if (arg == "--swirl_strength") {
            opts.swirl_strength = ParseFloatArg(value, arg);
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
    if (opts.plume_radius <= 0.0f) {
        throw std::runtime_error("--plume_radius must be > 0");
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
    SetStage("before save frame " + std::to_string(step));
    ofm::GetCenteralVecAsync(*(solver.u_), solver.tile_dim_, *(solver.init_u_x_), *(solver.init_u_y_), *(solver.init_u_z_), stream);
    ofm::GetVorNormAsync(*(solver.vor_norm_), solver.tile_dim_, *(solver.u_), solver.dx_, stream);
    solver.vor_norm_->DevToHostAsync(stream);
    std::cerr << "[DEBUG] before cudaStreamSynchronize\n" << std::flush;
    SetStage("before save cudaStreamSynchronize frame " + std::to_string(step));
    cudaStreamSynchronize(stream);
    std::cerr << "[DEBUG] after cudaStreamSynchronize\n" << std::flush;
    SetStage("after save cudaStreamSynchronize frame " + std::to_string(step));

    const int nx = solver.tile_dim_.x * 8;
    const int ny = solver.tile_dim_.y * 8;
    const int nz = solver.tile_dim_.z * 8;

    std::ostringstream filename;
    filename << "frame_" << std::setfill('0') << std::setw(6) << step << ".npy";
    const fs::path output_path = output_dir / filename.str();
    const std::size_t count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    const double mb = static_cast<double>(count * sizeof(float)) / 1024.0 / 1024.0;
    std::cerr << "[DEBUG] save frame=" << step
              << " path=" << output_path.string()
              << " shape=(" << nx << "," << ny << "," << nz << ")"
              << " count=" << count
              << " estimated_mb=" << mb << "\n" << std::flush;

    WriteNpyFloat32(output_path, solver.vor_norm_->host_ptr_, nx, ny, nz);
    std::cout << "[OFM Headless] Saved vorticity to " << output_path.string() << "\n" << std::flush;
    SetStage("after save frame " + std::to_string(step));
}

int main(int argc, char** argv)
{
    try {
        SetStage("cli parsing");
        const HeadlessOptions options = ParseOptions(argc, argv);
        last_resolution = options.resolution;
        std::cerr << "[DEBUG] cli resolution=" << options.resolution.x << "x" << options.resolution.y << "x" << options.resolution.z << "\n";
        std::cerr << "[DEBUG] cli steps=" << options.steps << "\n";
        std::cerr << "[DEBUG] cli save_interval=" << options.save_interval << "\n";
        std::cerr << "[DEBUG] cli output_dir=" << options.output_dir << "\n";
        std::cerr << "[DEBUG] cli plume_strength=" << options.plume_strength << "\n";
        std::cerr << "[DEBUG] cli plume_radius=" << options.plume_radius << "\n";
        std::cerr << "[DEBUG] cli swirl_strength=" << options.swirl_strength << "\n";
        std::cerr << "[DEBUG] cli device=" << options.device << "\n" << std::flush;

        cudaSetDevice(options.device);
        cudaDeviceProp device_prop {};
        cudaGetDeviceProperties(&device_prop, options.device);
        const std::string timestamp = BuildTimestamp();
        const fs::path resolution_dir = fs::path(options.output_dir) / "plume" / ("res" + std::to_string(options.resolution.x));
        const fs::path run_dir = resolution_dir
            / ("ps" + FormatFloatTag(options.plume_strength) + "_pr" + FormatFloatTag(options.plume_radius)
                + "_sw" + FormatFloatTag(options.swirl_strength) + "_" + timestamp);
        const fs::path vorticity_dir = run_dir / "vorticity";
        const fs::path preview_dir = run_dir / "preview";
        const fs::path logs_dir = run_dir / "logs";
        const fs::path stats_dir = run_dir / "stats";
        fs::create_directories(vorticity_dir);
        fs::create_directories(preview_dir);
        fs::create_directories(logs_dir);
        fs::create_directories(stats_dir);

        std::cerr << "[DEBUG] before constructing OFM\n" << std::flush;
        SetStage("before constructing OFM");
        ofm::OFM solver;
        std::cerr << "[DEBUG] after constructing OFM\n" << std::flush;
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        const OFMConfiguration cfg = BuildOFMConfiguration(options);
        WriteConfigJson(run_dir / "config.json", options, cfg, run_dir, timestamp);
        SetStage("before InitOFMAsync");
        ofm::InitOFMAsync(solver, cfg, stream);
        SetStage("before init cudaStreamSynchronize");
        cudaStreamSynchronize(stream);
        SetStage("after init cudaStreamSynchronize");
        const std::size_t nx = static_cast<std::size_t>(solver.tile_dim_.x) * 8ULL;
        const std::size_t ny = static_cast<std::size_t>(solver.tile_dim_.y) * 8ULL;
        const std::size_t nz = static_cast<std::size_t>(solver.tile_dim_.z) * 8ULL;
        const std::size_t count = nx * ny * nz;
        last_tile_dim = solver.tile_dim_;
        last_nx = nx;
        last_ny = ny;
        last_nz = nz;
        last_count = count;
        const double velocity_mb = static_cast<double>(count * sizeof(float3)) / 1024.0 / 1024.0;
        const double scalar_mb = static_cast<double>(count * sizeof(float)) / 1024.0 / 1024.0;
        std::cerr << "[DEBUG] solver.tile_dim=" << solver.tile_dim_.x << "x" << solver.tile_dim_.y << "x" << solver.tile_dim_.z << "\n";
        std::cerr << "[DEBUG] voxel dims nx/ny/nz=" << nx << "/" << ny << "/" << nz << "\n";
        std::cerr << "[DEBUG] voxel count=" << count << "\n";
        std::cerr << "[DEBUG] estimated velocity memory MB=" << velocity_mb << "\n";
        std::cerr << "[DEBUG] estimated scalar memory MB=" << scalar_mb << "\n" << std::flush;

        const auto wall_start = std::chrono::steady_clock::now();
        const float dt = 1.0f / 30.0f;
        const int expected_saved_frames = (options.steps - 1) / options.save_interval + 1;
        std::cout << "[OFM Headless] Simulation starts\n";
        std::cout << "[OFM Headless] Resolution: " << options.resolution.x << "x" << options.resolution.y << "x" << options.resolution.z << "\n";
        std::cout << "[OFM Headless] tile_dim: " << solver.tile_dim_.x << "x" << solver.tile_dim_.y << "x" << solver.tile_dim_.z << "\n";
        std::cout << "[OFM Headless] dx: " << solver.dx_ << "\n";
        std::cout << "[OFM Headless] dt: " << dt << "\n";
        std::cout << "[OFM Headless] steps: " << options.steps << "\n";
        std::cout << "[OFM Headless] save_interval: " << options.save_interval << "\n";
        std::cout << "[OFM Headless] output_dir: " << options.output_dir << "\n";
        std::cout << "[OFM Headless] RUN_DIR: " << run_dir.string() << "\n";
        std::cout << "[OFM Headless] Vorticity frames:\n  " << (vorticity_dir / "frame_%06d.npy").string() << "\n";
        std::cout << "[OFM Headless] Log file path (use shell tee):\n  " << (logs_dir / "run.log").string() << "\n";
        std::cout << "[OFM Headless] CUDA device: [" << options.device << "] " << device_prop.name << "\n";
        std::cout << "[OFM Headless] plume: " << (options.plume ? "on" : "off") << "\n";
        std::cout << "[OFM Headless] plume_strength: " << options.plume_strength << "\n";
        std::cout << "[OFM Headless] plume_radius: " << options.plume_radius << "\n";
        std::cout << "[OFM Headless] swirl_strength: " << options.swirl_strength << "\n";

        for (int step = 0; step < options.steps; ++step) {
            solver.inlet_angle_ = options.inlet_angle;
            solver.inlet_norm_ = options.inlet_norm;
            solver.voxelized_velocity_scaler_ = options.voxelized_velocity_scaler;

            if (step > 0) {
                SetStage("before UpdateBoundary step " + std::to_string(step));
                std::cerr << "[DEBUG] before UpdateBoundary\n" << std::flush;
                solver.UpdateBoundary(stream);
                std::cerr << "[DEBUG] after UpdateBoundary\n" << std::flush;
                SetStage("after UpdateBoundary step " + std::to_string(step));
            }
            if (options.plume) {
                SetStage("before AddPlumeSourceAsync step " + std::to_string(step));
                const std::size_t pnx = static_cast<std::size_t>(solver.tile_dim_.x) * 8ULL;
                const std::size_t pny = static_cast<std::size_t>(solver.tile_dim_.y) * 8ULL;
                const std::size_t pnz = static_cast<std::size_t>(solver.tile_dim_.z) * 8ULL;
                const std::size_t pcount = pnx * pny * pnz;
                const int min_dim = std::min({ static_cast<int>(pnx), static_cast<int>(pny), static_cast<int>(pnz) });
                const float radius = fmaxf(options.plume_radius * static_cast<float>(min_dim), 1e-4f);
                const float cx = 0.5f * static_cast<float>(pnx);
                const float cy = 0.15f * static_cast<float>(pny);
                const float cz = 0.5f * static_cast<float>(pnz);
                std::cerr << "[DEBUG] plume params strength=" << options.plume_strength
                          << " radius_norm=" << options.plume_radius
                          << " swirl=" << options.swirl_strength << "\n";
                std::cerr << "[DEBUG] plume dims nx/ny/nz/count=" << pnx << "/" << pny << "/" << pnz << "/" << pcount << "\n";
                std::cerr << "[DEBUG] plume radius/cx/cy/cz=" << radius << "/" << cx << "/" << cy << "/" << cz << "\n" << std::flush;
                AddPlumeSourceAsync(
                    solver,
                    options.plume_strength,
                    options.plume_radius,
                    options.swirl_strength,
                    stream);
                SetStage("after AddPlumeSourceAsync step " + std::to_string(step));
            }
            SetStage("before AdvanceAsync step " + std::to_string(step));
            std::cerr << "[DEBUG] before AdvanceAsync\n" << std::flush;
            solver.AdvanceAsync(dt, stream);
            std::cerr << "[DEBUG] after AdvanceAsync\n" << std::flush;
            SetStage("after AdvanceAsync step " + std::to_string(step));
            SetStage("before ReinitAsync step " + std::to_string(step));
            std::cerr << "[DEBUG] before ReinitAsync\n" << std::flush;
            solver.ReinitAsync(dt, stream);
            std::cerr << "[DEBUG] after ReinitAsync\n" << std::flush;
            SetStage("after ReinitAsync step " + std::to_string(step));

            if (step % options.save_interval == 0) {
                SaveVorticityField(solver, vorticity_dir, step, stream);
            }

            if (step % 50 == 0 || step + 1 == options.steps) {
                std::cout << "[OFM Headless] Step " << step + 1 << " / " << options.steps << "\n";
            }
        }

        SetStage("before final cudaStreamSynchronize");
        std::cerr << "[DEBUG] before cudaStreamSynchronize\n" << std::flush;
        cudaStreamSynchronize(stream);
        std::cerr << "[DEBUG] after cudaStreamSynchronize\n" << std::flush;
        SetStage("after final cudaStreamSynchronize");
        const auto wall_end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(wall_end - wall_start).count();
        const double avg_seconds_per_step = seconds / static_cast<double>(options.steps);
        std::cout << "[OFM Headless] Finished.\n";
        std::cout << "[OFM Headless] total runtime: " << seconds << " s\n";
        std::cout << "[OFM Headless] saved frames: " << expected_saved_frames << "\n";
        std::cout << "[OFM Headless] average seconds per step: " << avg_seconds_per_step << " s\n";


        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::bad_alloc& e) {
        std::cerr << "[OFM Headless] std::bad_alloc caught\n";
        std::cerr << "what: " << e.what() << "\n";
        std::cerr << "Last stage: " << last_stage << "\n";
        std::cerr << "resolution: " << last_resolution.x << "x" << last_resolution.y << "x" << last_resolution.z << "\n";
        std::cerr << "tile_dim: " << last_tile_dim.x << "x" << last_tile_dim.y << "x" << last_tile_dim.z << "\n";
        std::cerr << "voxel dims: " << last_nx << "x" << last_ny << "x" << last_nz << "\n";
        std::cerr << "count: " << last_count << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[OFM Headless] Error: " << e.what() << "\n";
        return 1;
    }
}
