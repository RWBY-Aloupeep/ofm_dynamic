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

### The plan comes from the Artifact, not from this repo and not from you

The full research plan (literature audit, the three concrete gaps it identifies, the Stage 0–D
roadmap, per-stage pass criteria, and open risks) is **not duplicated in this repo**. It lives in
the "Fire-Whirl Flow Maps" Artifact, owned by the project's author, and is mirrored into persistent
memory. It is versioned (v2.1 as of 2026-09-11) and is updated independently of this code.

**It is authoritative.** Read it (`Artifact action:"read"`) before planning any implementation work,
and take the objective, the stage ordering, and the pass criteria from it rather than proposing your
own. Where the Artifact and the memory mirror disagree, the Artifact wins — the mirror can be stale.
Where the Artifact and your own judgment disagree, say so explicitly and let the author decide; do
not quietly substitute a different plan, a different stage order, or a different pass criterion.

When work shows something in the Artifact to be wrong, that is a finding to report, and the fix is
to update the Artifact — it carries an errata appendix ("Corrections") for exactly this, newest
version first. Do not leave the repo and the Artifact disagreeing.

### Literature is closed to the Zotero tag

All source papers are tagged `proj:wildfire` in the author's personal Zotero library, reachable via
the Zotero Web API (credentials in persistent memory, not in this repo). Two rules, both
load-bearing:

**Read, don't recall.** When a task needs a specific paper's claim, fetch and read the actual PDF
through the API. Citation-level precision is load-bearing for this project's methodology, and a
half-remembered number is worse than no number.

**Do not go outside the tag.** The corpus is deliberately bounded. Benchmarks, validation cases,
pass criteria, and cited claims must come from papers under `proj:wildfire` (plus the small,
individually documented out-of-library set the Artifact lists in its §2 H group). This rules out
reaching for standard CFD material that is not in the library — a textbook test case, a canonical
correlation, a benchmark "everyone uses". If the library genuinely lacks what a task needs, say so
and ask; adding a paper to the corpus is the author's call, and anything admitted gets recorded in
the Artifact with its acquisition status. Do not introduce an outside source and then justify it
after the fact.

A related consequence, stated because it has already been decided: **purely qualitative pass
criteria are not accepted.** "Reproduces the phenomenon" is not a criterion; a relative error, a
percentage, or a measured quantity against an analytic or published value is.

## Compute environment (UW Hyak klone)

All work happens on **UW Hyak `klone`**. Documentation: https://hyak.uw.edu/docs

**Do complex work on a compute node, never on the login node.** `klone-login01` and its siblings
are shared, and are for editing, git, and job submission only. Compilation, simulation runs,
voxelization, and any long or memory-hungry analysis must be submitted to a compute node — either
interactively with `salloc` or as a batch job with `sbatch`.

Run `hyakalloc` to see the current allocation and what is free right now. As of 2026-08-24 this
account can submit to two Slurm accounts:

| Account | Partition | Resources |
|---|---|---|
| `amath` | `cpu-g2` | 320 CPUs, 2519G |
| `amath` | `gpu-l40s` | 32 CPUs, 377G, **2 GPUs** |
| `amath` | `gpu-rtx6k` | 40 CPUs, 376G, **8 GPUs** |
| `stf` | `compute`, `compute-hugemem`, `cpu-g2`, `cpu-g2-mem2x` | CPU-only, up to 680 CPUs / 5038G |
| `stf` | `gpu-l40`, `gpu-l40s` | **8 and 10 GPUs** |

The `ckpt` partition additionally offers idle cluster-wide capacity, but jobs there are
preemptible — fine for restartable sweeps, not for a long single run.

Typical interactive GPU session for this project:

```
salloc -A amath -p gpu-l40s --gpus=1 -c 8 --mem=64G --time=4:00:00
module load cuda/12.6.3
```

`cuda/12.6.3` matches the CUDA version this codebase targets; `module avail` lists the rest.

Two hardware notes that matter for the solver:

- GPU generations differ across partitions. `gpu-rtx6k` is Turing (`sm_75`), which is exactly what
  `add_cugencodes("compute_75")` in the xmake files targets; `gpu-l40` and `gpu-l40s` are Ada
  (`sm_89`). Verify the codegen flags before assuming a build runs on the L40S nodes — the
  gencode list likely needs `sm_89` added rather than relying on PTX JIT.
- The example applications are interactive GLFW/Vulkan/ImGui windows, which will not open on a
  headless compute node as-is. Upstream has a `headless-export` branch that may be a useful
  starting point for producing field output without a display.

Storage: this repo lives under `/gscratch/amath/diwenxu` (GPFS, mounted at `/mmfs1`). Simulation
output belongs there or under scratch, not in the repository — see `.gitignore`.

## Build

Build on a compute node, not the login node (see above).

```
git submodule update --init --recursive   # required — src/engine and src/AMGPCG_Pybind_Torch
                                            # are submodules and start out empty after a plain clone
module load cuda/12.6.3
cd proj
xmake build
```

Toolchain: xmake, C++20, CUDA 12.6 (`compute_75` codegen), Vulkan, VTK 9.3.1, GLFW/GLM/Dear ImGui
(fetched by xmake via `add_requires`). CUDA 12.6 is available on klone as a module; Vulkan
availability on the compute nodes is still unconfirmed, and that matters because the current
applications are windowed. The original author only verified this on Windows 11 + RTX 4080 laptop.
Also, `proj/xmake.lua` includes a `sim_render` subproject directory that does not exist in this repo
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
1. **`OFM::AdvanceAsync`** — advects the previous step's projected velocity, applies the inlet BC,
   and pressure-projects once ("Projection 1"), storing the result in `mid_u_{x,y,z}_[step %
   reinit_every_]`. Which velocity is advected, and over what interval, follows the leapfrog
   schedule of LFM's Algorithm 1: a half step at the start of a reinitialization cycle, a full step
   next, and thereafter the velocity from *two* steps back advected across `2*dt` by the velocity of
   the previous step. At `reinit_every_ == 1` only the first branch is ever reached.
2. **`OFM::ReinitAsync`** — called once per reinitialization cycle (**not** once per step unless
   `reinit_every_ == 1`). The flow map is reset to identity and re-marched through the cycle's
   stored velocity history — backward map (ψ, T) walking the history in reverse, forward map (φ, F)
   walking it forwards. `reinit_every_` defaults to 1, which reproduces the "one-step" (`n=1`)
   scheme the OFM thesis trades accuracy for real-time speed with; larger values restore LFM's
   multi-step cycle. `rk_order_` selects the marching order (2, 4, or TVD-RK3 by default). Note that
   drivers must call `AdvanceAsync` `reinit_every_` times per `ReinitAsync`, and that the solver's
   velocity state (`init_u_`) is only current at a cycle boundary. It reconstructs velocity
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
map has no persistent state across reinitializations (`ReinitAsync` resets it to identity every
call) — any circulation accumulator the attribution work needs has to be built as new state, not
recovered from the existing map. The channel source terms must enter through is the path integral of
LFM's Algorithm 1 / Equation (8); `RKAxisAccumulateForceAsync` implements its integrand (note it
*assigns* rather than accumulates, so the running sum is the caller's job) but is called by nothing,
here or in LFM's own release, so both solvers as shipped are inviscid.
