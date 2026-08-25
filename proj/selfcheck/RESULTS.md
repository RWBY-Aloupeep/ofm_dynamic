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

## D5: the source-term channel, wired and verified

`RKAxisAccumulateForceAsync` having no caller in either upstream repository left
the baroclinic, drag and buoyancy terms with no way into the solver, so the channel
was written against LFM's Algorithm 1 and Eq. (8). This section is the verification.

### What was implemented

Algorithm 1 puts the source in two places, and both are needed:

- **`OFM::AdvanceAsync`** (lines 1, 6, 12) adds the source to the velocity being
  advected, over the same interval it is advected across, evaluated on the velocity
  that transports it. Those three lines differ only in their interval, which the
  leapfrog schedule already carries, so they collapse to one expression.
- **`OFM::ReinitAsync`** (lines 5, 10, 16) accumulates the path integral into the
  initial-time impulse. This has to happen inside the forward march, before the
  pullback reads `init_u_`.

Eq. (8) places the quadrature points at step **midpoints**. The upstream kernel
samples at the *start* of a step and marches a whole step in the same pass, which
is a left-endpoint rule, so it is not used here; the solver marches the forward map
in half steps and contracts the source with the forward Jacobian in between. Two
kernels were added for that: a staggered seven-point Laplacian whose out-of-range
neighbours clamp onto the centre sample (a free-slip wall, matching what
`SetWallBc*Kernel` imposes on the tangential components), and a contraction of the
source with the map Jacobian at a given quadrature point.

`use_source_term_` is off and `viscosity_` is zero by default, so the solver behaves
exactly as it did before. The leapfrog ring case still measures 1 leap.

### How it is verified

A columnar vortex carrying the Burgers profile with no imposed axial strain spreads
by viscosity alone and its core obeys `b(t)^2 = b(0)^2 + 4*nu*t` exactly. The Burgers
profile is the model the fire whirl literature fits to measured cores -- several
studies report it as the best fit for a quasi-steady on-source fire whirl, its
normalised profile is self-similar, and its azimuthal velocity peaks at
`r = 1.12091 b_w` (Tohidi et al. 2018, Eq. 6-7).

The estimator needs no circulation input. Peak vorticity of the profile is
`Gamma / (pi b^2)` and circulation is conserved, so

    b(t)^2 / b(0)^2 = w_max(0) / w_max(t)

and the viscosity the solver actually applied comes back as

    nu_eff = b(0)^2 * (w_max(0)/w_max(t) - 1) / (4t)

Running the same case at `nu = 0` measures the numerical dissipation floor in the
same units, which is what `nu_eff` has to stand clear of. The radius of peak
azimuthal velocity is reported alongside as an independent estimator.

This is **numerical verification** -- whether the code solves the equations it
claims to -- not physical validation. Tohidi records that some studies dispute the
Burgers profile as a description of real fire whirls; that bears on physical
similarity, not on its value as an exact solution of the equations being solved.

### Result: the channel recovers the viscosity it is given

Measured on klone (`gpu-l40s`), unit cube, `b(0) = 0.06`, `Gamma = 0.05`, `n = 1`:

| grid | dt | simulated | `nu = 0` floor | `nu` asked | `nu` recovered | error |
|---|---|---|---|---|---|---|
| 128^3 (b0 = 7.7 cells) | 1/480 | 1.0 s | 7.02e-5 | 1e-3 | 1.017e-3 | +1.69% |
| 256^3 (b0 = 15.4 cells) | 1/960 | 1.0 s | 1.15e-5 | 1e-3 | 9.970e-4 | **-0.30%** |

**The 256^3 run passes the 1% criterion**, and the error is steady across the whole
run (-0.40% at the first sample, -0.30% at the last) rather than drifting. The second
estimator agrees there too: `r_peak / 1.12091` tracks the analytic core to about 1%
(0.06796 vs 0.06785, 0.08189 vs 0.08127).

**The residual is the numerical dissipation floor, not the channel.** The floor is
7.0% of the imposed `nu` at 128^3 and 1.1% at 256^3, and the error follows it down,
1.69% to 0.30%. Refining the grid by two cuts the floor by 6.1x.

A coarser earlier sweep at `nu = 2.25e-4` on 128^3 shows the same thing from the
other side. There the floor is 7.8% of `nu`, and the error is +2.33% at `n = 1` and
-3.74% at `n = 5`; both drift toward each other as the run proceeds (from +3.72% and
-5.83%), which is a startup transient rather than a wrong rate. That configuration
does not pass 1%, and should not be expected to at that ratio of floor to signal.

### A side finding: numerical dissipation scales with reinitialization count

The `nu = 0` runs give a controlled measurement that the leapfrog case could not.
At the same grid and the same simulated time of 1 s on 128^3, taking 480 steps
instead of 120 -- four times as many reinitializations -- raises the floor from
1.98e-5 to 7.02e-5, a factor of 3.55.

So at `n = 1` the numerical dissipation is charged per reinitialization, not per unit
of simulated time, and **refining the time step makes the solution more diffusive,
not less**. That explains the earlier observation that halving `dt` changed nothing
in the leapfrog case, and it is a second reason to raise the reinitialization
interval rather than lower `dt`.

### Reproducing

```
./build/selfcheck --test burgers --res-tiles 32 --nu 1e-3 --dt 0.00104167 \
                  --steps 960 --diag-every 240 --reinit-every 1 --csv burgers.csv
```

`--nu 0` gives the floor for the same configuration. `--res-tiles T` sets the grid to
`8T` cells per side.

## D1: the circulation budget, and what it says the residual is

With the source channel in place, the circulation attribution follows from the
flow-map solution rather than needing a separate mechanism. For a material loop C
whose preimage at the start of a cycle is C0,

    Gamma(t) = closed_integral_C u.dl = closed_integral_C0 [ m0 + sum_k sum_i dt F^T s_k ] . dX
             = Gamma(0) + sum_k dGamma_k

because `u = m - grad(phi)` and a gradient integrates to zero around a closed loop.
So **each source channel's accumulator, integrated around the loop's preimage, is
that channel's contribution to the circulation**. Nothing else is needed: the
separation the wildfire literature has never been able to make is a line integral
of a field the solver already builds.

### What was added

`SourceChannel` splits the source into channels (currently viscous and external
force; baroclinic and drag get their own once there is a density field). Each
channel accumulates its own `sum_i dt F^T s_k` into `acc_[k]` over the cycle.

Two properties matter and are enforced by construction:

- **Attribution does not perturb the physics.** Every channel's contribution still
  lands in `init_u_`; splitting only changes how many times the contraction runs.
  With `track_attribution_` off the code takes the original combined path.
- **The accumulators reset with the map.** They live in the frame of the cycle's
  start, so they are cleared at every reinitialization. The running total across
  cycles is a *scalar per loop*, not a field, and is the caller's to keep — this is
  the cross-reinitialization accumulator the plan identifies as this project's own
  contribution. It also means the line integrals have to be taken every cycle, not
  only on diagnostic steps.

`CirculationOnCircle` in the harness integrates a staggered field around a circle
by trilinear interpolation of the cell-centred form. Applied to the velocity it
gives the loop's circulation; applied to `acc_[k]` it gives that channel's share.

### The test case, and why the loop is legitimate

The diffusing columnar Burgers vortex again, because every leg of the identity is
known in closed form. Its radial velocity is zero, so **a circle of fixed radius is
a material loop**, and it is still its own preimage at every cycle start — exactly
the frame the accumulators live in. The loop is taken at `r = b0`.

    Gamma(r,t) = Gamma_inf (1 - exp(-r^2 / b(t)^2)),   b(t)^2 = b0^2 + 4 nu t

Three quantities are compared, all as changes since t = 0 so the discretisation of
the seeded field cancels: **direct** (measured line integral of the velocity),
**budget** (summed accumulator line integrals), **analytic**.

### Result

| grid | n | budget vs analytic | budget vs direct |
|---|---|---|---|
| 128^3 (b0 = 7.7 cells) | 1 | 4.04% | 6.39% |
| 128^3 | 5 | 2.68% | 1.39% |
| 256^3 (b0 = 15.4 cells) | 5 | **0.58%** | **0.11%** |

**The 256^3 run passes both halves of the criterion** — attribution error under 1%,
and the dual paths agreeing to 0.11%, steady across the entire run with no drift.

The channel split is correct in the way that matters: with only viscosity active,
the budget assigns **all** of the circulation change to the viscous channel and
**exactly zero** to the external one.

### The dual-path gap is the residual, not just a consistency check

The two error columns measure different things, and separating them is the useful
part:

- **budget vs analytic** is the discretisation error of `nu*lap(u)` itself. At
  7.7 cells across the core the discrete Laplacian underestimates the true one, so
  the budget falls short of the exact answer; refining the grid fixes it.
- **budget vs direct** is the part of the measured circulation change that **no
  modelled source accounts for**. The velocity field really did lose more
  circulation than the physics asked for, and the excess is numerical dissipation.

That second column is the number this project actually lives or dies by. An
attribution result of the form "X% from tilting, Y% from baroclinic" only means
something if the unattributed remainder is small next to the terms being compared.
Here it is 6.39% at `n = 1`, 1.39% at `n = 5`, and 0.11% at 256^3 — so the
reinitialization interval buys attribution credibility, not just speed, which is a
third independent argument for the default of 5.

### Regression

With source terms off, the D5 Burgers case recovers `nu = 1.018662e-03` — bit for
bit what it returned before the channel split. The refactor is numerically neutral
when attribution is not asked for.

### Reproducing

```
./build/selfcheck --test attribution --res-tiles 32 --nu 2.25e-4 --dt 0.00416667 \
                  --steps 960 --diag-every 120 --reinit-every 5 --rk-order 4
```

`--loop-radius` overrides the loop radius, which defaults to the seeded core.

### Stretching and tilting, reported separately

The other half of D1. The fire whirl and VLS literature argues that vertical
vorticity comes from tilting ambient horizontal vorticity rather than from
stretching, but no paper in the corpus measures the two against each other:

    D w_z / Dt = w_x d_x w + w_y d_y w  +  w_z d_z w
                 \_____tilting_____/       \_stretching_/

This is a pointwise diagnostic of the vorticity equation, not part of the
circulation budget, so it is calibrated against an analytic field rather than
against the solver: the field goes in, no time stepping happens, and the output is
compared with the closed form. The field

    u = ( -W y - (g/2) x,  W x - c z - (g/2) y,  w0 + s x + g z )

is divergence free with vorticity `(c, -s, 2W)` and `grad(w) = (s, 0, g)`, so
`tilting = c s` and `stretching = 2 W g`, both uniform in space and independently
tunable. It is also **linear**, on which central differences are exact — so
anything above round-off is an error in the operator, not truncation.

| grid | tilting rel. err | stretching rel. err |
|---|---|---|
| 128^3 | -2.4e-08 | -2.5e-09 |
| 64^3 | +1.3e-08 | -3.4e-08 |

Both at float round-off, and independent of resolution as the linear field
requires. A second case seeds a purely columnar vortex, where `w = 0` everywhere
and both terms must vanish: mean `|tilting|` comes back at 2e-15 and mean
`|stretching|` at 3.8e-11, so the two terms are not leaking into each other.

```
./build/selfcheck --test tilting --res-tiles 16
```

## The source-term channel: how it was found missing

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
