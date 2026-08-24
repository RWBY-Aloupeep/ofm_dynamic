# D5 baseline: vortex preservation of the solver as shipped

Measured 2026-08-24 on klone (`gpu-l40s`, NVIDIA L40S, CUDA 12.6.3), against the
unmodified OFM solver in `src/ofm`. These numbers are the regression baseline the
low-Mach extension has to be checked against.

## Why a baseline had to be measured rather than assumed

The plan takes its pass criterion from LFM: two leapfrogging vortex rings should
survive **5 leaps** before merging (LFM/NFM 5, PFM 4, Covector Fluids 3, BiMocq²
2). Reading LFM §6.2 alongside Table 4 turns up two things that stop that number
from being used directly here.

1. **LFM's ladder is measured with a reinitialization interval of 10.** The paper
   states plainly that "all methods reinitialize every 10 steps", and Table 4 lists
   the leapfrog figure at `n = 10`. OFM's whole design point is `n = 1`: it resets
   the flow map at the start of every time step. The two are therefore not the same
   measurement, and OFM's thesis says outright that the one-step scheme is more
   diffusive and that its own validation against LFM was visual only.
2. **LFM does not publish the geometry of the test.** Table 4 gives resolution
   (256 × 128 × 128), `n`, CG iterations (15) and total runtime for Figure 14, but
   no ring radius, core size, spacing or circulation. The absolute leap count
   depends on all four, so an exact reproduction is not available from the paper.

So "≥ 5" is LFM's number for LFM's configuration. What follows is OFM's number for
a configuration documented here.

## Configuration

| | |
|---|---|
| Grid | 256 × 128 × 128 (`tile_dim = {32,16,16}`), `dx = 1/128`, domain 2 × 1 × 1 |
| Rings | two coaxial along x, radius `R = 0.15` (19.2 cells), core `σ = 0.04` (5.1 cells) |
| | circulation `Γ = 0.08`, spacing `d = 0.12` (`d/R = 0.8`), leading ring at `x = 0.42` |
| Time | `dt = 1/60`, 900 steps |
| Poisson | 15 CG iterations, matching LFM Table 4 |
| Boundaries | closed box (`SetWallBcAsync` with zero inlet) |

Initial condition is a regularized (Rosenhead–Moore) Biot–Savart field for each
filament, superposed and then projected divergence free.

## Result: 1 leap

The rings complete exactly one exchange and then stop leapfrogging.

| step | pair centre x | r_a | r_b | \|Δr\| |
|---|---|---|---|---|
| 0 | 0.362 | 0.149 | 0.151 | 0.0020 |
| 40 | 0.478 | 0.116 | 0.178 | 0.0624 |
| 50 | 0.514 | 0.121 | 0.183 | crossing |
| 80 | 0.595 | 0.150 | 0.172 | 0.0219 |
| 120 | 0.696 | 0.166 | 0.166 | 0.0008 |
| 200 | 0.884 | 0.167 | 0.167 | 0.0008 |
| 360 | 1.267 | 0.171 | 0.170 | 0.0009 |

The radial separation is the leapfrog signature: one ring must contract and speed
up while the other expands and slows. It peaks at 0.062 around step 40, the rings
cross at step 50, and by step 120 it has collapsed to below 0.001 and never
recovers. From there the two vorticity maxima travel together, 0.06 apart in x, at
identical radius — a co-travelling pair, not a leapfrogging one. Peak vorticity
falls from 15.0 to 11.3 by step 400 (−25%).

This is well before the rings approach the far wall, so the enclosure is not what
ends the leapfrogging.

## The result is not an artefact of the chosen parameters

| Γ | d/R | σ (cells) | dt | leaps |
|---|---|---|---|---|
| 0.03 | 1.0 | 5.1 | 1/60 | 1 |
| 0.08 | 1.0 | 5.1 | 1/60 | 1 |
| 0.08 | 1.0 | 5.1 | 1/120 | 1 |
| 0.08 | 0.4 | 5.1 | 1/60 | 0 |
| 0.08 | 0.6 | 5.1 | 1/60 | 1 |
| 0.08 | 0.8 | 5.1 | 1/60 | **1** |
| 0.08 | 0.8 | 7.7 | 1/60 | 0 |
| 0.08 | 0.8 | 10.2 | 1/60 | 0 |

Halving the time step changes nothing, so this is not time-step accuracy.
Widening the core makes it worse rather than better, so it is not simply that the
core is under-resolved at 5 cells. One leap is the best this configuration
produces.

## Open questions before treating 1 as *the* number

- **Boundary conditions.** The harness runs a closed box. LFM's boundary treatment
  for this figure is not stated, and an enclosure imposes a return flow that an
  open or periodic domain would not.
- **Geometry.** Since LFM's ring parameters are unpublished, a configuration that
  favours leapfrogging more strongly may exist. The sweep above covers a
  reasonable neighbourhood but is not exhaustive.
- **No independent verification of the harness.** The initial condition and the
  tracker were checked against each other (the tracker locates both rings at their
  seeded positions to three decimals, and vorticity appears only in the shell
  containing the ring radius), but the harness has not been validated against a
  case with a known answer.

## Separate finding: the source-term channel is not wired up

`RKAxisAccumulateForceAsync` — the path integral that carries external forces and
viscosity along the flow map, in both RK2 and RK4 variants — is fully implemented
in `src/ofm/ofm_util.cu` and declared in `ofm_util.h`, but **nothing calls it**.
It does not appear in `OFM::ReinitAsync`, anywhere else in `ofm.cu`, or in either
application.

This matters beyond the self-check suite. It is the same channel LFM used to
introduce viscosity and thereby produce its Kármán vortex street, and it is the
channel the baroclinic and vegetation-drag source terms are meant to enter
through. The mechanism exists and is validated in LFM, and the kernels are present
here, but the shipped solver does not use them. The Kármán case cannot be run
until that call is added, and adding it changes `src/ofm`, so it is left as a
decision rather than done here.

## Reproducing

```
module load cuda/12.6.3
cd proj/selfcheck && xmake -P . -y
./build/selfcheck --test leapfrog3d --steps 900 --diag-every 10 \
                  --circulation 0.08 --spacing 0.12 --core 0.04 --csv run.csv
```

`--radius`, `--core`, `--circulation`, `--spacing`, `--dt` and `--steps` are all
settable. The CSV carries per-sample kinetic energy, peak speed, peak vorticity,
both ring positions in the (axial, radial) plane, their separation and the running
leap count.
