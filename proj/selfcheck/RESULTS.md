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

## Follow-up: OFM is LFM at `n = 1`, and what changes when `n` is raised

The caveat above -- that LFM's ladder is measured at a reinitialization interval of 10
while OFM resets every step -- turned out to be testable rather than merely a caveat.
Reading the LFM reference implementation (<https://github.com/yuchen-sun-cg/lfm>) against
`src/ofm` shows the two solvers are the same code: `*_util.cu` is ~92% identical, and the
differences are the marching-order dispatch, one cosmetic signature, and OFM's added
dynamic-boundary kernel. `OFM::ReinitAsync` and `LFM::ReinitAsync` agree line for line
after the map march. **OFM is LFM specialised to `reinit_every = 1`, plus dynamic
boundaries.**

Two mechanisms are lost in that specialisation. The flow map spans one step instead of
`n`, and -- less obviously -- LFM's `AdvanceAsync` implements a leapfrog time integrator
whose third branch advects the velocity from two steps back across `2*dt`. That branch is
reached only when `step % n >= 2`, so at `n = 1` the solver runs the integrator's start-up
half-step forever and never takes a leapfrog step at all. The LFM paper ablates exactly
this (Section 6.2, Fig. 10b: "we replaced the leapfrog method in LFM with directly
advecting the midpoint velocities sequentially. The modified method can not preserve the
vortex pairs well").

`reinit_every_` and `rk_order_` are now parameters of `ofm::OFM`, so both schemes run in
this harness with everything else held fixed: same rings, same tracker, same grid, same 15
CG iterations, same `dt = 1/60`, and the loop counted in advection steps so the physical
time axis is identical. `reinit_every = 1` with TVD-RK3 is the default and reproduces the
baseline above.

| `n` | RK | leaps | peak w @200 | @400 | @560 | loss @400 | max abs dr | at step |
|---|---|---|---|---|---|---|---|---|
| 1 | TVD-RK3 | 1 | 11.10 | 11.31 | 11.59 | 26.1% | 0.0626 | 51 |
| 1 | RK4 | 1 | 11.10 | 11.31 | 11.59 | 26.1% | 0.0626 | 51 |
| 2 | RK4 | 1 | 12.36 | 11.89 | 12.27 | 22.3% | 0.0643 | 42 |
| 5 | RK4 | 1 | 14.29 | 12.93 | 12.97 | 15.5% | 0.0656 | 45 |
| 10 | RK4 | 1 | 16.07 | 13.74 | 13.45 | **10.2%** | 0.0645 | 50 |
| 10 | TVD-RK3 | 1 | 16.07 | 13.74 | 13.45 | 10.2% | 0.0645 | 50 |

Seeded peak vorticity is 15.30 in every run. The window stops at step 560, before the
rings reach the wall guard at step ~575.

**Vorticity preservation improves strongly and monotonically with `n`.** Loss at step 400
falls from 26.1% to 10.2%, a factor of 2.6, and at step 200 the `n = 10` run is still
above its seeded peak. Kinetic-energy loss falls from 3.3% to 2.0%. This is the result
that matters for the attribution work: a direct measurement of the numerical dissipation
floor as a function of the reinitialization interval. Both solvers are inviscid, so all of
it is numerical.

**The leap count does not move.** It is 1 at every interval, and more tellingly the radial
separation that drives leapfrogging is nearly identical across all six runs: it peaks at
0.063-0.066 around step 42-51 and then collapses below 0.0013 in every case. Dissipation
varies by a factor of 2.6 across these runs while the separation history barely moves at
all. So in this configuration the leapfrog mechanism is not ended by dissipation, and the
expectation that the baseline's "1 leap" was a consequence of `n = 1` is wrong. What ends
it is the ring configuration -- which makes the "geometry" open question above the binding
one, not the boundary conditions and not the scheme.

**Marching order is irrelevant here.** RK4 and TVD-RK3 agree to the digits shown at both
`n = 1` and `n = 10`. At `n = 1` that is expected, since the map is marched one step from
identity.

**Long intervals cost stability.** Both `n = 10` runs go non-finite at step 860, well
after the rings have hit the far wall (step ~575) and the field has become violent -- peak
vorticity is already 160 at step 810. `n = 5` and below complete all 900 steps. Nothing
here affects the leapfrog window itself, but it does bound how far `n` can be pushed
inside an enclosure.

### What this changes

The reinitialization interval is now the main accuracy knob available, and it is cheap:
`n = 5` reinitializes once per five steps instead of once per step, so it is *faster* than
`n = 1` per unit of simulated time while dissipating 40% less vorticity. For the
circulation-attribution work the relevant target is the dissipation floor rather than the
leap count, and this table is the first measurement of it.

Raising the leap count, if it is wanted as a published comparison, needs a different ring
configuration rather than a different scheme. That search is not done here.

## Separate finding: the source-term channel is not wired up

`RKAxisAccumulateForceAsync` — the path integral that carries external forces and
viscosity along the flow map, in both RK2 and RK4 variants — is fully implemented
in `src/ofm/ofm_util.cu` and declared in `ofm_util.h`, but **nothing calls it**.
It does not appear in `OFM::ReinitAsync`, anywhere else in `ofm.cu`, or in either
application.

This matters beyond the self-check suite. It is the channel the baroclinic and
vegetation-drag source terms are meant to enter through.

The LFM paper is explicit that this is how it introduced viscosity: Algorithm 1
carries the viscous and external-force terms as
`u_0 += (dt/rho) F^T_{0,i+1/2} (mu*lap(u) + f)(Phi_{0,i+1/2})` accumulated over the
cycle, Equation (8) gives that midpoint quadrature of the path integral, and Section
6.2 states the Karman vortex street (Fig. 8) was produced "by incorporating viscosity
through the path integral during forward marching".

What the LFM source shows is that **the released code does not implement those lines
either**. `RKAxisAccumulateForceAsync` has no caller in LFM's repository, and LFM
contains no viscosity term at all; the public code is the inviscid subset of
Algorithm 1. So the wiring exists in neither repository and has to be written, with
Algorithm 1 and Equation (8) as the spec. Note also that the kernel *assigns*
`f_axis = f . T` rather than accumulating it, so the running sum over the cycle is the
caller's responsibility, and `ofm::OFM` owns no force buffer to sum into. Both solvers
as shipped being inviscid is also why the vorticity losses measured above are entirely
numerical.

## Reproducing

```
module load cuda/12.6.3
cd proj/selfcheck && xmake -P . -y
./build/selfcheck --test leapfrog3d --steps 900 --diag-every 10 \
                  --circulation 0.08 --spacing 0.12 --core 0.04 --csv run.csv
```

`--radius`, `--core`, `--circulation`, `--spacing`, `--dt` and `--steps` are all
settable, as are `--reinit-every N` (1 = the one-step scheme, the default) and
`--rk-order 2|3|4` (3 = TVD-RK3, the default). The interval sweep above is that same
command with `--reinit-every` set to 1, 2, 5 and 10; `--steps` counts advection steps, so
the simulated time is the same for every interval. The CSV carries per-sample kinetic energy, peak speed, peak vorticity,
both ring positions in the (axial, radial) plane, their separation and the running
leap count.
