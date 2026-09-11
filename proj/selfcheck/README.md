# Solver self-check harness

A headless, console-only driver for the OFM solver. It exists so the solver can be
built and exercised on a compute node with no display, which the interactive
applications in `proj/dynamic_obstacle` and `proj/voxelization` cannot do because
they open a GLFW/Vulkan window.

Those applications are untouched. This directory is a **separate xmake project**
that is deliberately not referenced from `proj/xmake.lua`, so the GUI build path
keeps its Vulkan/GLFW/ImGui/VTK dependencies and this one links only CUDA.

## Build and run

```
module load cuda/12.6.3
cd proj/selfcheck
xmake -P . -y
./build/selfcheck
```

The harness drives the solver in reinitialization cycles: `--reinit-every N` runs `N`
advection steps per reinitialization (`N = 1`, the default, is the one-step scheme OFM
ships with; LFM runs its leapfrog figure at 10), and `--rk-order 2|3|4` selects the
flow-map marching order. Diagnostics are sampled at cycle boundaries, because the solver's
velocity state is only current there.

Every case starts by launching a probe kernel and checking that it ran, and prints
the device it is on. The build carries cubin for `sm_75` (`gpu-rtx6k`) and `sm_89`
(`gpu-l40`, `gpu-l40s`) plus `compute_75` PTX for anything newer; on an older
device -- the `ckpt` partition mixes generations, down to P100 -- every launch
would otherwise fail unreported and the run would print a full table of zeros as
if it were a measurement. Pin all the cases of one sweep to a single GPU
generation; `ckpt` has `rtx6k` nodes, reachable with `--constraint=rtx6k`, which
is the way in when the owned partition is at its GPU limit.

`-P .` matters: without it xmake walks up, finds `proj/xmake.lua`, and tries to
resolve the GUI packages. Compute nodes on klone have no outbound network, so any
package that is not already cached will fail there; configure on the login node if
xmake ever needs to download something.

## Cases

`--test` selects one; `--help` lists the flags. The Stage 0 diagnostics are
`burgers` (D1), `attribution` and `tilting` (D2), `coreradii` (D3) and
`damkohler` (D4); `leapfrog3d` is the relative regression on the LFM ring pair.
`plume` is the Stage A case after Cunningham et al. 2005, whose boundaries are
chosen with `--outflow closed|x|xy` (the first cut's closed box, the downstream
face open, or the downstream and both lateral faces open), whose `theta` diffuses
at `kappa = mu/(rho Pr)` with `--pr` (0.7 by default, 0 for none), and which takes
a Rayleigh damping layer under the lid with `--sponge`. `outflow` is the
verification of the open face on a translating Gaussian vortex column, in an
`--outflow open` box, an `--outflow closed` control, and a `--long` box the vortex
never leaves; `analyse_outflow.py` reads the three together. What each case
measured, and against what, is in `RESULTS.md`.

The `*.sbatch` and `*.sh` files are the sweeps that produced the recorded
numbers; submit the `.sbatch` ones from this directory, since they build here on
the compute node before running.

## How it links against the solver

Only the numerical stack is compiled: `src/ofm/ofm.cu`, `src/ofm/ofm_util.cu`, and
the `common`/`solver` sources from `src/AMGPCG_Pybind_Torch`. Two accommodations
are needed, both contained here so that neither `src/ofm` nor the submodules are
modified:

- `src/ofm/ofm_init.cu` is excluded. It is the only solver file that depends on the
  engine's JSON `Configuration` type; the harness configures the solver directly in
  code instead, following the same sequence as `ofm::InitOFMAsync`.
- `compat/core/tool/logger.h` replaces the engine's spdlog logger, which
  `AMGPCG_Pybind_Torch/common/timer.cu` includes for its `INFO_ALL`/`ERROR_ALL`
  macros.

`xmake.lua` also force-includes `<cstdint>`, because `common/mem.cc` uses `uint8_t`
without including it and relies on a transitive include that GCC here does not
provide.
