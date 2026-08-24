# wildfire-sim

**Circulation attribution in fire whirls, using impulse/covector flow maps.**

Where does the vertical vorticity of a fire whirl actually come from? The wildfire literature agrees
on the mechanism chain — buoyancy generates *horizontal* vorticity, which must be tilted into the
vertical and then concentrated by stretching — and it agrees that tilting and stretching cannot
create vorticity, only redistribute it. Three sources can actually generate it: ambient horizontal
vorticity (ground shear, terrain separation), baroclinic generation where `∇ρ` and `∇p` are not
parallel at the fire line, and surface drag. Different papers name different ones as dominant.

None of them computes the split. This project does, by decomposing the circulation on a material
loop into per-source contributions:

```
Γ_C(t) = Γ_ambient + Γ_baroclinic + Γ_drag + Γ_viscous
```

The approach exploits a property of flow-map solvers: in the impulse (gauge) formulation, tilting
and stretching are carried by the flow map's Jacobian rather than appearing as discretized source
terms, the pullback of the impulse covector is exact on a material loop (the discrete counterpart of
Kelvin's theorem), and the pressure projection drops out of the circulation budget entirely, since
`∮ ∇φ · dx = 0` on any closed loop. In an Eulerian solver these contributions are mixed together and
cannot be separated; under a flow-map representation they come apart naturally.

## Status

Early. The codebase is currently the unmodified upstream OFM solver (see *Acknowledgments*); none of
the wildfire-specific physics or diagnostics has been added yet. Planned work, roughly in order:

1. **Diagnostics** — circulation budget with tilting and stretching reported separately, Burgers-vortex
   fitting, a curvilinear-grid vorticity operator, a vortex–flame Damköhler number to flag where the
   prescribed-heat-source assumption breaks down, and the LFM vortex-preservation self-check suite
   (2D leapfrog, Kármán vortex street, 3D leapfrog vortex rings).
2. **Low-Mach extension** — the upstream solver is incompressible and constant-density; wildfire
   combustion drives `∇·u ≠ 0` through thermal expansion. This is the main open methodological risk.
3. **Cross-step attribution** — the flow map is reinitialized every time step, so the per-source
   circulation budget has to be accumulated across reinitializations as new persistent state.
4. **Validation stages** — buoyant-plume-in-crossflow and fire-line-with-slope configurations drawn
   from the literature, a lee-slope vorticity-driven lateral spread case, and quantitative comparison
   against published wildfire benchmark datasets.

## Build

```
git submodule update --init --recursive
cd proj
xmake build
```

Requires xmake, C++20, CUDA 12.6, Vulkan, and VTK; GLFW, GLM, and Dear ImGui are fetched by xmake.
`src/engine` and `src/AMGPCG_Pybind_Torch` are submodules and are empty after a plain clone, so the
submodule step is not optional.

Upstream was developed and verified on Windows 11 with an NVIDIA RTX 4080 laptop GPU. Note that
`proj/xmake.lua` includes a `sim_render` directory that is not present in this repository; build a
specific target (`xmake build dynamic_obstacle`) or remove that include until it is cleaned up.

## Run

```
./build/dynamic_obstacle     # from proj/dynamic_obstacle
./build/voxelization         # from proj/voxelization
```

Each application reads its settings from `proj/<app>/config/<app>.json`.

<div align="center">
    <img src="image/rotating_octa.png"/>
</div>
<div align="center">
    <img src="image/demo.gif"?raw=true/>
</div>

## Acknowledgments

This repository is a fork of **[ofm_dynamic](https://github.com/Mr-222/ofm_dynamic)** by Yutong Sun,
the reference implementation of the master's thesis *One-Step Flow Maps for Real-Time Fluid
Simulation with Dynamic Boundaries* (Georgia Institute of Technology, 2025). The entire flow-map
solver in `src/ofm`, the real-time mesh voxelization pipeline for dynamic boundaries, and the
example applications in `proj/` are that author's work. This project builds its wildfire physics and
circulation diagnostics on top of that foundation, and it would not be feasible to start from
scratch.

OFM in turn builds on **[Leapfrog Flow Maps](https://yuchen-sun-cg.github.io/projects/lfm/)** (Sun
et al., *ACM Transactions on Graphics* 44(4), 2025), which contributed the hybrid velocity–impulse
leapfrog integrator and the matrix-free AMGPCG GPU Poisson solver that this code depends on.

Upstream remains available as the `upstream` git remote:

```
git remote -v
git fetch upstream
```

## License

MIT, inherited from upstream — see [LICENSE](LICENSE). Copyright for the original solver code
remains with Yutong Sun and the Leapfrog Flow Maps authors.
