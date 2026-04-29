# dynamic_obstacle_headless

## Purpose
`dynamic_obstacle_headless` is a standalone, headless OFM executable for reproducible data generation on HPC/CI systems.

It focuses on producing simulation outputs (vorticity norm `.npy`) without any GUI stack.

## Why this bypasses xmake/Vulkan/GUI
This pipeline uses a CMake-only standalone build in `proj/dynamic_obstacle_headless` and intentionally avoids GUI/render dependencies to keep runs reliable in non-graphical environments:
- no Vulkan
- no GLFW
- no ImGui
- no render engine startup path

## Build
From repository root:

```bash
cmake -S proj/dynamic_obstacle_headless -B proj/dynamic_obstacle_headless/build -DCMAKE_BUILD_TYPE=Release
cmake --build proj/dynamic_obstacle_headless/build -j
```

## Run
Smoke test preset:

```bash
proj/dynamic_obstacle_headless/run_smoke_test.sh
```

Larger preset:

```bash
proj/dynamic_obstacle_headless/run_64_test.sh
```

Direct executable example:

```bash
./proj/dynamic_obstacle_headless/build/dynamic_obstacle_headless \
  --steps 20 \
  --save_interval 1 \
  --resolution 32,32,32 \
  --output_dir ./proj/dynamic_obstacle_headless/build/outputs/headless_smoke
```

## Validation
Numerical checks for NaNs/Infs and summary stats:

```bash
python3 proj/dynamic_obstacle_headless/check_npy.py ./proj/dynamic_obstacle_headless/build/outputs/headless_smoke
```

Visualization and optional GIF:

```bash
python3 proj/dynamic_obstacle_headless/visualize_vorticity.py ./proj/dynamic_obstacle_headless/build/outputs/headless_smoke --axis z --index mid --log --gif
```

## Output format
Each saved frame is written as:
- file pattern: `vorticity_XXXXXX.npy`
- field: vorticity norm
- dtype: `float32`
- shape: `(Nx, Ny, Nz)`

## Current limitations
- Minimal smoke-test-focused flow.
- No GUI path in this standalone executable.
- No full dynamic obstacle visualization pipeline yet.
- Output currently stores vorticity norm only (not full vorticity vector components).
