# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

**Everything written into this repository must be in English** — code comments, commit messages,
documentation, README content, config comments, log/error strings, identifiers, and any file added
later. No Chinese characters anywhere in the repo, even in scratch or TODO notes. (Conversation with
the project's author may be in Chinese; the repo contents may not.)

## What this project is

This repo is a research project on **circulation attribution in fire whirls**: using impulse/
covector flow-map fluid solvers to compute where a fire whirl's vertical vorticity actually comes
from (ambient horizontal vorticity tilted vertical, baroclinic generation at the fire line, surface
drag, viscosity), something the existing wildfire literature only asserts but has never quantified.

The repo is a fork of **OFM** (One-Step Flow Maps, Yutong Sun 2025, Georgia Tech MS thesis, built
on Sun et al. 2025's Leapfrog Flow Maps / LFM, ACM TOG 44(4)) — `origin` is this project's own repo,
`upstream` (`Mr-222/ofm_dynamic`) is the original author's, kept as a remote so changes there can
still be pulled in. At the point of forking, none of the wildfire-specific physics (variable density,
baroclinic/drag source terms, the circulation-attribution accumulator) had been added yet — what's
in the codebase is still the unmodified OFM/LFM reference solver.

The upstream authorship is not incidental and should be preserved: essentially all existing solver
code (`src/ofm`, the voxelization pipeline, the `proj/` example apps) is Yutong Sun's work, the
AMGPCG solver and leapfrog integrator come from the LFM authors, and the repo inherits their MIT
LICENSE. Keep the Acknowledgments section of README.md accurate as the code diverges, keep the
LICENSE file intact, and don't present upstream code as this project's contribution.

The full research plan (literature audit of 24 papers, the three concrete gaps it identifies, the
Stage 0–D implementation roadmap, and open risks) is **not duplicated in this repo** — it lives in
the "Fire-Whirl Flow Maps" Artifact (owned by the project's author) and is mirrored into this
session's persistent memory. Read that before planning any new implementation work; it is versioned
and gets updated independently of this repo's code.

**Literature access:** all source papers are tagged `proj:wildfire` in a personal Zotero library,
reachable via the Zotero Web API (credentials in persistent memory, not in this repo). When a task
needs a specific paper's claim, fetch and read the actual PDF via that API rather than answering
from general recall — citation-level precision is load-bearing for this project's methodology.

## Build

```
git submodule update --init --recursive   # required — src/engine and src/AMGPCG_Pybind_Torch
                                            # are submodules and start out empty after a plain clone
cd proj
xmake build
```

Toolchain: xmake, C++20, CUDA 12.6 (`compute_75` codegen), Vulkan, VTK 9.3.1, GLFW/GLM/Dear ImGui
(fetched by xmake via `add_requires`). The original author only verified this on Windows 11 + RTX
4080 laptop — CUDA/Vulkan availability on this Linux/HPC environment is unconfirmed, and
`proj/xmake.lua` includes a `sim_render` subproject directory that does not exist in this repo
(only `dynamic_obstacle` and `voxelization` do) — that stray include will need removing or a build
target selected explicitly (`xmake build dynamic_obstacle`) before a full build succeeds.

## Run

```
./build/dynamic_obstacle     # in proj/dynamic_obstacle — dynamic rigid-body obstacle demo
./build/voxelization         # in proj/voxelization — standalone voxelization-pipeline demo
```

Each app is config-driven from a JSON file at `proj/<app>/config/<app>.json` (window/camera setup,
voxelizer grid resolution, etc.), loaded in that app's `main.cpp`.

## Architecture

The solver (`src/ofm/ofm.h`, `ofm.cu`) is a single `ofm::OFM` class holding, per grid tile
(`tile_dim_`, 8³ voxels/tile, spacing `dx_`): staggered-grid boundary condition buffers per axis
(`is_bc_{x,y,z}_`, `bc_val_{x,y,z}_`); a **backward** flow map (`psi_{x,y,z}_`, `T_{x,y,z}_`) and a
**forward** flow map (`phi_{x,y,z}_`, `F_{x,y,z}_`), each stored per staggered axis; velocity
buffers (`u_{x,y,z}_` plus `init_`/`mid_`/`tmp_`/`err_` variants used across the BFECC pipeline);
and an `AMGPCG` (algebraic multigrid–preconditioned CG) Poisson solver instance for pressure
projection. Numerical kernels (advection, RK flow-map marching, pullback, projection, BFECC,
boundary setup) are free functions in `src/ofm/ofm_util.cu` / declared in `ofm_util.h` — that's
where the actual per-kernel math lives, not in `ofm.cu`.

Per-frame control flow (driven by `PhysicsEngineUser::step()` in `proj/*/physics.cu`, one call each
per frame):
1. **`OFM::AdvanceAsync`** — midpoint-advects the previous frame's projected velocity by itself
   (`AdvectN2{X,Y,Z}Async`), applies the inlet BC, and pressure-projects once ("Projection 1") to
   get a half-step velocity (`mid_u_{x,y,z}_`).
2. **`OFM::ReinitAsync`** — the flow map is **reset to identity and re-marched from scratch every
   single frame** (this is the "one-step", `n=1`, design the OFM thesis trades accuracy for
   real-time speed with — see the project memory on OFM/LFM for why this matters for the
   attribution work: there is no long-range map to preserve state across). It RK-marches both the
   backward map (ψ, T) and forward map (φ, F) using the half-step velocity, reconstructs velocity
   at the new time by pulling the *initial impulse* back through `T` (impulse/covector
   reconstruction, not direct velocity advection), runs one BFECC error-compensation pass (forward
   pull through φ/F, subtract, backward-pull the error through T, apply a half correction, optional
   clamp), and pressure-projects again ("Projection 2").
3. Dynamic solid boundaries (`use_dynamic_solid_`) are handled by voxelizing a moving mesh each
   frame into `voxel_tex_`/`velocity_tex_`, sampling that surface for BCs (`SetBcBySurfaceAsync`),
   and rebuilding the AMGPCG coefficient matrix (`amgpcg_.BuildAsync`) whenever the boundary moves.

Submodule boundaries: `src/AMGPCG_Pybind_Torch/` supplies the `common`/`solver` xmake targets (the
AMGPCG Poisson solver) that `src/ofm` depends on; `src/engine/` supplies the app framework
(`Engine`, `RenderEngine`, and the `CudaEngine`/`UIEngineUser` base classes each `proj/*` app
subclasses — e.g. `proj/dynamic_obstacle/physics.h`'s `PhysicsEngineUser : public CudaEngine` owns
an `ofm::OFM` and drives it).

**Two facts about this codebase that directly bound the wildfire adaptation work** (see project
memory for the full reasoning): the solver is **incompressible, constant-density** throughout (one
scalar pressure field, no density field anywhere in `OFM`/`ofm_util.cu`) — wildfire combustion needs
`∇·u ≠ 0` from thermal expansion, which will mean touching `AdvanceAsync`/`ReinitAsync`/
`ProjectAsync` and the divergence/pressure kernels, not just adding a new app on top. And the flow
map genuinely has no persistent state across frames (confirmed by `ReinitAsync` resetting to
identity every call) — any cross-frame circulation accumulator the attribution work needs has to be
built as new state, not recovered from the existing map.
