# D1 baseline: vortex preservation of the solver as shipped

> **Numbering note (2026-08-26).** The Stage 0 diagnostics were renumbered so that the
> number is the execution order. Old -> new: D5 -> D1 (source-term path integral),
> D1 -> D2 (circulation attribution), D2 -> D3 (core-radius fitter), D4 unchanged.
> The curvilinear-grid vorticity operator, withdrawn together with Stage C, no longer
> holds a number. Commit messages written before this date use the old numbers.

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

## D1: the source-term channel, wired and verified

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
`r = 1.12091 b_w` (Tohidi et al. 2018, Eq. 7 -- Eq. 6 on the same page is the
Rankine model, not this one).

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

## D2: the circulation budget, and what it says the residual is

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

With source terms off, the D1 Burgers case recovers `nu = 1.018662e-03` — bit for
bit what it returned before the channel split. The refactor is numerically neutral
when attribution is not asked for.

### Reproducing

```
./build/selfcheck --test attribution --res-tiles 32 --nu 2.25e-4 --dt 0.00416667 \
                  --steps 960 --diag-every 120 --reinit-every 5 --rk-order 4
```

`--loop-radius` overrides the loop radius, which defaults to the seeded core.

### Stretching and tilting, reported separately

The other half of D2. The fire whirl and VLS literature argues that vertical
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

## D3: the Burgers core-radius fitter, and the three-radius ordering

Tohidi et al. 2018 report that the Burgers model is the best fit for a
quasi-steady on-source fire whirl, that its azimuthal velocity peaks at
`r = 1.12091 b_w` (Section 4.1, Equation 7), and that three core radii can be
defined which satisfy `b_A > b_T > b_w` throughout the height of a fire whirl
(Section 4.2, Figure 7, after Lei et al. 2015b):

- `b_w`, from the azimuthal velocity — the Burgers parameter, fitted here.
- `b_T`, "the radial location where the excess temperature declines to half of
  the maximum recorded value at that height".
- `b_A`, "the radial distance ... at which the local axial velocity `U_z` has
  declined to a fraction of the maximum recorded value at the same height", the
  fraction being 0.5 in the continuous flame region. Equation 9 gives a second,
  integral form, `b_A = Q_hat / sqrt(M_hat)` from the specific mass and axial
  momentum fluxes, which needs no top-hat assumption.

Note that Section 4.2 characterises `b_w` as the radius at which the tangential
velocity is maximum, whereas in the Burgers model of Equation 7 the maximum sits
at `1.12091 b_w`. The two conventions differ by that factor. The fitter reports
the Equation 7 parameter, which is why the criterion is stated as a ratio.

### What was implemented

`FitBurgers` does a least-squares fit of

    U_theta(r) = (Gamma_inf / 2 pi r) (1 - exp(-r^2 / b_w^2))

to a measured radial profile. The model is **linear in `Gamma_inf`**, so that
parameter is eliminated by its own normal equation and only `b_w` is searched
over — a coarse scan to bracket, then golden section. No initial guess, no
Levenberg-Marquardt, and the result is deterministic.

The peak radius is taken from the data, not from the fitted model: reading the
peak off the model and dividing by the model's own `b_w` would return 1.12091 by
construction and test nothing. It is located by the vertex of the parabola
through the maximum bin and its two neighbours, which is needed because the bins
are one cell wide and that is coarser than the quantity being tested.

`BurgersPeakConstant` solves `exp(-x)(2x + 1) = 1` — the stationarity condition
of Equation 7 — and returns `sqrt(x)`, so the literature constant is checked
rather than transcribed. It returns **1.1209064**, against the 1.12091 quoted.

`RadiusAtFraction` and `RadiusFromFluxes` implement the two `b_A` forms and the
`b_T` form. `MeasureRadialProfile` bins the solver's own field; the excess
temperature has no field to come from until the low-Mach extension, so
`GaussianProfileOnGrid` supplies that leg from an analytic profile put through
the identical binning.

### Result

Three cases, at three resolutions. Unit cube, `b_w = 0.06`, `Gamma_inf = 0.05`,
no time stepping — these calibrate the estimators, not the solver.

**Case 1, the fitter alone**, on the closed-form profile sampled at the grid's
own bins:

| grid | `b_w` rel. err | `Gamma_inf` rel. err | `r_peak/b_w` rel. err | residual RMS/peak |
|---|---|---|---|---|
| 256^3 | -4.4e-12 | -2.5e-12 | +9.7e-04 | 1.4e-12 |
| 128^3 | -3.4e-13 | -2.0e-13 | +4.6e-03 | 1.1e-13 |
| 64^3 | -3.8e-12 | -2.2e-12 | +1.9e-02 | 1.3e-12 |

The fit recovers both parameters to round-off at every resolution, and the
residual is round-off too, so **the fitter contributes no error**. What does vary
is the peak ratio, and it varies with resolution alone — that column is the
error of locating the peak on one-cell bins, not of the fit. It falls 1.9e-2 ->
4.6e-3 -> 9.7e-4 for successive halvings of `dx`, ratios of 4.15 and 4.72, so the
parabolic refinement is converging at second order.

**Case 2, the whole pipeline**: seed the columnar Burgers vortex, project, and
fit the azimuthal profile measured back off the grid.

| grid | `b_w` in cells | `b_w` rel. err | `Gamma_inf` rel. err | `r_peak/b_w` | rel. err |
|---|---|---|---|---|---|
| 256^3 | 15.4 | -2.9e-03 | -1.3e-03 | 1.12325 | +0.21% |
| 128^3 | 7.7 | -7.9e-03 | -3.9e-03 | 1.12861 | +0.69% |
| 64^3 | 3.8 | -1.9e-02 | -9.8e-03 | 1.13817 | +1.54% |

**256^3 and 128^3 pass the 1% criterion; 64^3 does not**, at 3.8 cells across the
core. This is the same resolution binding the D1 result has, and for the same
reason.

**Case 3, the ordering.** A Gaussian axial jet of scale `2.6 b_w` supplies `U_z`,
an analytic excess temperature of scale `1.8 b_w` supplies `dT`, and the Burgers
column supplies `U_theta`. For a Gaussian of scale `a`, the half-maximum radius is
`a sqrt(ln 2)` and the flux form of Equation 9 returns `a` exactly, so all three
estimators have closed-form targets.

| grid | `b_A` half-max | `b_A` flux | `b_T` half-max | `b_w` fit | ordering |
|---|---|---|---|---|---|
| 256^3 | -1.6e-03 | -4.0e-04 | -8.9e-05 | -2.9e-03 | holds |
| 128^3 | -2.9e-03 | -6.4e-04 | +1.2e-03 | -8.0e-03 | holds |
| 64^3 | -6.8e-03 | -1.4e-03 | +5.8e-03 | -1.9e-02 | holds |

Relative errors against the closed form. **The ordering `b_A > b_T > b_w` is
reported correctly at every resolution**, including the one where the individual
radii miss 1%, because it is an inequality between numbers separated by 45% and
50% — far more than the estimator errors.

**The flux form of `b_A` is the better-conditioned estimator**, by a factor of
4-5 against the fraction-of-maximum form at every resolution (-4.0e-04 against
-1.6e-03 at 256^3). That is what one would expect of an integral against a local
interpolation, and it is worth preferring Equation 9 when the profile is noisy.

### Scope limit

`b_T` is measured from an analytic excess temperature, not a simulated one,
because the solver is constant-density and carries no temperature field. That
leg therefore calibrates the estimator, and connecting it to a simulated
temperature field waits on the low-Mach extension. `b_w` and `b_A` are both
measured from fields the solver holds.

### Reproducing

```
./build/selfcheck --test coreradii --res-tiles 32 --csv coreradii.csv
```

`--core` and `--circulation` set `b_w` and `Gamma_inf`. The CSV carries the three
measured profiles and the fitted Burgers curve on the same radial bins.

## D4: the vortex-flame Damkohler number

Linan, Vera & Sanchez 2015 (Section 7) characterise a vortex-flame interaction
by the strain the vortex imposes on the flame,

    A_Gamma = Gamma / (2 r0^2),

and the vortex Damkohler number

    Da_Gamma = A_e / A_Gamma,

"defined as the ratio of the characteristic vortex turnover time, `1/A_Gamma`, to
the characteristic chemical time, `1/A_e`", with `A_e` the critical strain rate at
extinction. They state that "local flame extinction should be expected for
`Da_Gamma <~ 1`". This is the self-check on the prescribed-heat-source
assumption: where `Da_Gamma` falls below one, that assumption has no support.

**The plan's `A_Gamma ~ Gamma / r0^2` is missing the factor of two.** The paper
writes `A_Gamma = Gamma / (2 r0^2)` as a definition. The factor moves the
`Da_Gamma = 1` contour by `sqrt(2)` in the strain that produces it, so it matters
for a criterion stated as a contour position. The paper's form is what is
implemented here.

### What was implemented

The paper's `Gamma` and `r0` are the vortex's circulation and characteristic
radius. Evaluating the same expression with the circulation enclosed at radius
`r` gives a pointwise field that reduces to the paper's definition at `r = r0`:

    A_Gamma(r) = Gamma(r) / (2 r^2),   Da_Gamma(r) = A_e / A_Gamma(r)

`Gamma(r)` is the D2 line integral, so the two diagnostics share their
measurement of circulation. `CirculationRadialSweep` was added because placing a
contour needs hundreds of loops and `CirculationOnCircle` pulls the whole field
back to the host on each call — at 256^3 that is minutes of transfer per contour.
The sweep downloads once and integrates every loop on the host.

For a Burgers vortex the contour is a circle whose radius follows in closed form.
With `x = (r/b_w)^2`, `Da_Gamma = 1` reduces to

    (1 - exp(-x)) / x = 2 A_e b_w^2 / Gamma_inf = kappa

whose left side falls monotonically from 1 to 0. So there is exactly one root
when `kappa < 1` and none otherwise, and `A_Gamma` is largest on the axis, which
puts the extinction region **inside** the contour.

### Result

Unit cube, `b_w = 0.06`, `Gamma_inf = 0.05`, seeded and projected. Four
extinction rates chosen to place the analytic contour at known multiples of the
core radius, plus one case with no contour at all.

| `r*/b_w` | `A_e` | `r*` exact | 128^3 measured | rel. err | 256^3 measured | rel. err |
|---|---|---|---|---|---|---|
| 0.50 | 6.1444 | 0.030000 | 0.029182 | -2.73% | 0.029794 | -0.69% |
| 1.00 | 4.3897 | 0.060000 | 0.059692 | -0.51% | 0.059917 | -0.14% |
| 1.50 | 2.7611 | 0.090000 | 0.089875 | -0.14% | 0.089966 | -0.038% |
| 2.00 | 1.7043 | 0.120000 | 0.119961 | -0.033% | 0.119990 | -0.0086% |

**256^3 passes the 1% criterion across the whole range.** At 128^3 three of the
four contours are within 0.51% and only the tightest fails, at 2.73%.

The error is set by how many cells the contour spans, not by the grid as such:
`r* = 0.060` is 7.7 cells at 128^3 and gives -0.51%, while `r* = 0.030` is 7.7
cells at 256^3 and gives -0.69%. Contours wider than about 10 cells are accurate
to better than 0.15% at either resolution. The far-field cases converge fastest
because nearly all the circulation is enclosed by then, so `Gamma(r)` is
insensitive to where exactly the loop sits.

**The no-contour regime is reported correctly.** Setting `A_e` to `1.2 *
Gamma_inf / (2 b_w^2)` puts `kappa = 1.2 > 1`, so `Da_Gamma > 1` everywhere and
no extinction is predicted. Both resolutions report no contour, matching the
analytic answer. This matters because a diagnostic that always finds a contour
would look like it was working while telling you nothing.

### What the test does and does not establish

It establishes that the implemented diagnostic reproduces the closed-form
contour of the definition on a discretely represented field: the measured side
goes through seeding, projection, cell-centre interpolation, loop quadrature and
a root find, none of which the analytic side sees. It does not establish that
`Da_Gamma <~ 1` is where a real flame extinguishes — that is Linan's physical
claim, cited, not tested here.

### Reproducing

```
./build/selfcheck --test damkohler --res-tiles 32 --csv damkohler.csv
```

The CSV carries one row per extinction rate with the exact and measured contour
radii.

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

The other cases are `--test burgers` (D1, the source-term channel), `--test
attribution` and `--test tilting` (D2), and `--test coreradii` and `--test
damkohler` (D3 and D4). Each section above gives its own command line. Every case
returns a non-zero exit code when it misses its criterion, so a batch script can
gate on them.

## Stage A: a buoyant plume in a sheared cross flow, first cut

The configuration is Cunningham, Goodrick, Hussaini & Linn 2005 (Int. J. Wildland
Fire 14, 61-75). The numbers below were read from the paper, not from the plan
Artifact, and three of them differ from what the plan carried.

| | Paper |
|---|---|
| Governing equations | three-dimensional **compressible** flow in a density-stratified atmosphere, forced by a prescribed volumetric heat source; solved with WRF (split-explicit, RK3 advection, fifth-order upwind) |
| Base state | ambient potential temperature **uniform at 300 K** -- neutral |
| Cross flow | `U(z) = U0 tanh(z/z0)`, `U0 = 4.5 m/s`, `z0` in {50, 100, 150} m |
| Heat source | `Q = Q0 tanh(t/t0) exp(-z/h) * shape`, `t0 = 10 s`, `h = 25 m` (a vertical decay scale, not a height) |
| Circular source | smoothed top-hat, `R1 = 75 m`, `R2 = 150 m`, `d = 12.5 m`, centred at `(450, 600) m`; `Q0 = 1 kW/m^3` gives 1000 MW |
| Elliptical source | `A = 200 m`, `B = 100 m`; `Q0 = 1 kW/m^3` gives 1700 MW |
| Canopy drag | `D_i = rho Cd a V u_i`, `Cd = 0.1`, `a = 0.25 1/m`, **lowest grid level only** |
| Direct runs | `mu` = 4, 1, 0.15, 0.0015 kg/(m s), `Pr = 0.7` |
| LES | `Ck = 0.1`, `Ce = 0.93`, turbulent `Pr = 1/3` |
| Domain | 1800 x 1200 x 1500 m, uniform 10 m spacing |
| Boundaries | lateral and downstream **non-reflecting outflow** (Orlanski 1976); top and bottom solid-wall **free-slip**; damping layer under the lid |
| Timing | quasi-steady after ~600 s; wake shedding period ~200 s |
| Strouhal | `St = n D / U0` with **`D = 2 R_c`**, `R_c = (R1+R2)/2 = 112.5 m`, so `D = 225 m`; a 200 s period gives `St = 0.25` |
| Cross-sections | potential temperature on the y-z plane at **x = 1750 m** |
| Results | larger `z0` gives a wider bifurcation; for a given cross flow the **weaker** source gives a wider bifurcation; the cross-section is not self-similar Gaussian in any run |

Three corrections to what the plan carried: the base state is neutral (the plan
did not say, and it is what makes a Boussinesq reduction defensible at all);
`h = 25 m` is the vertical decay scale of `Q`, not a source height; and `St` is
built on `D = 2 R_c`, not `2 R2` -- using `2 R2` turns 0.25 into 0.33 and would
fail a correct run.

### Two gaps between that configuration and this solver

**Boundary conditions.** `SetWallBcAsync` prescribes the normal velocity on all
six faces. The solver has no outflow condition and no damping layer, so this
case prescribes `U(z)` on both x faces -- inflow and outflow carry the same
mass -- and leaves the lateral and top faces as free-slip walls. That is not
Orlanski. The paper's measurement plane at `x = 1750 m` sits 50 m from the
outflow, well inside the influence of a prescribed-profile face, so a split
width measured there is contaminated. Either move the plane upstream, lengthen
the box, or add an outflow condition; the last is the real fix and needs its own
verification case before it can be trusted.

**Thermal expansion.** The solver is incompressible and constant density.
Buoyancy enters as `g theta' / theta0` written into `f_z_` and carried by the
source-term path integral, which is what that field was built for. The neutral
base state makes the Boussinesq reduction reasonable in the far field, but at
`Q0 = 1 kW/m^3` the parcel-following anomaly reaches tens of K, so `dT/T0` is
O(0.1) near the source and the expansion term `Q/(rho cp T)` is not negligible
there. Plume structure should survive that; absolute widths should not be
quoted against the paper until the low-Mach extension lands.

Neither gap blocks the run. Whether a counter-rotating pair forms, and which way
its width moves with `z0` and `Q0`, is a vorticity-dynamics question that the
present solver can answer. Both gaps bound what the numbers mean.

### What was implemented

`--test plume` in `proj/selfcheck`, with the physics in the harness rather than
in `ofm::OFM`, since `f_{x,y,z}_` exists precisely so a caller can add buoyancy
and drag:

- `PlumeSpec` carries the paper's parameters.
- `SetPlumeBcAsync` starts from closed free-slip walls and overwrites the x
  faces with `U(z)`, then rebuilds the Poisson coefficients.
- `AddPlumeHeatAsync` integrates `theta += Q/(rho cp) dt` on the cell centres.
- `SetBuoyancyAndDragAsync` writes `f_z = g theta/theta0` on the z faces and
  `-Cd a |u_h| u_i` on the lowest cell level, zero elsewhere.
- `theta` is advected with `AdvectN2CAsync`, the cell-centred advection kernel
  that shipped with the solver and, like `RKAxisAccumulateForceAsync` before it,
  had no caller anywhere.
- `MeasurePlume` reports the peak anomaly, plume top, `w_max`, and the two
  extrema of `omega_z` on the measurement plane with the distance between them.
  It also reports the potential-temperature bifurcation, which is the quantity
  the paper actually plots -- see below.

The buoyancy and the `theta` advection originally both read `init_u_`, which is
only current at a cycle boundary, so the first cut ran at `reinit_every = 1` and
warned otherwise. Both now read the cycle's own velocity history instead, which
is what `n > 1` needs -- also below.

Building it also turned up a latent break: `src/ofm/ofm_util.h` uses `uint8_t`
without including `<cstdint>`. It only surfaced when a comment edit invalidated
the `src/ofm` build cache and forced a full rebuild.

### A third gap, found by running it: the inlet is hard-coded uniform

The first run held a cross-flow of at most 2.7 m/s where the profile asks for
4.5 m/s, and the plume went nearly straight up. `AdvanceAsync` calls

    SetInletAsync(*bc_val_x_, *bc_val_y_, tile_dim_, inlet_norm_, inlet_angle_, stream);

on **every step**, which rewrites the inlet and outlet planes of `bc_val_{x,y}_`
from a single scalar speed and angle. A sheared `U(z)` written before the first
step is erased by it. Nothing upstream of Stage A noticed, because D1 to D4 all
run closed boxes with `inlet_norm_ = 0`, where overwriting the planes with zero
is what those cases want anyway.

Fixed with `use_uniform_inlet_`, default `true`: when it is false `AdvanceAsync`
leaves `bc_val_{x,y}_` alone and the caller owns them. The default keeps every
existing case bit-identical, which the D1 Burgers regression confirms.

So the solver as it stands assumes the inflow is uniform, the box is closed, and
the fluid has one density. Stage A needs all three relaxed; two are now optional
flags and the third -- an outflow condition -- is still missing.

### First run, before the inlet fix

Coarse grid, 96 x 64 x 80 cells at dx = 18.75 m over the paper's exact
1800 x 1200 x 1500 m domain, `z0 = 100 m`, `Q0 = 1 kW/m^3`, `mu = 4 kg/(m s)`,
`dt = 0.5 s`, 600 s. This is the run that exposed the inlet problem: the plume
rose nearly vertically, hit the 1500 m lid at `t ~ 360 s`, and the only
counter-rotating pair sat at `x ~ 350 m`, on the source itself. The plane at
`x = 1750 m` carried nothing at all.

### The same run with the inlet fixed

Coarse grid, 96 x 64 x 80 cells at dx = 18.75 m over the paper's exact
1800 x 1200 x 1500 m domain, `z0 = 100 m`, `Q0 = 1 kW/m^3`, `mu = 4 kg/(m s)`,
`dt = 0.5 s`, 600 s:

| t (s) | max dT (K) | top (m) | w_max | u_max | omega_z(+) at x=1750 | omega_z(-) | split (m) | strongest plane |
|---|---|---|---|---|---|---|---|---|
| 100 | 7.79 | 272 | 2.72 | 4.79 | ~0 | ~0 | -- | x = 609 m, 0.011 1/s |
| 200 | 7.68 | 441 | 3.94 | 5.33 | ~0 | ~0 | -- | x = 666 m, 0.033 |
| 300 | 7.48 | 591 | 3.68 | 5.96 | +0.0005 | -0.0005 | -- | x = 741 m, 0.031 |
| 400 | 7.42 | 703 | 4.11 | 6.16 | +0.0009 | -0.0009 | -- | x = 816 m, 0.025 |
| 500 | 7.39 | 816 | 4.59 | 6.07 | +0.0044 | -0.0045 | 281 | x = 722 m, 0.022 |
| 600 | 7.40 | 759 | 4.84 | 5.85 | +0.0075 | -0.0076 | 206 | x = 703 m, 0.022 |

Three things this establishes, and one it does not.

- **The cross flow is there.** `u_max` sits at 4.8 to 6.2 m/s against the 4.5 m/s
  the profile asks for above the shear layer, the excess being the plume-induced
  acceleration.
- **The plume bends over instead of standing up.** The top climbs to about 815 m
  at 500 s and settles near 760 m, well clear of the 1500 m lid, so neither the
  lid nor the missing damping layer is setting the trajectory any more.
- **A counter-rotating pair forms, and its handedness matches the paper.** At
  `x = 1750 m` the positive `omega_z` sits at `y = 497 m` and the negative at
  `y = 703 m`, either side of the source axis at `y = 600 m`. With x streamwise
  and z up that is positive on the right-hand side looking downstream and
  negative on the left, which is what Cunningham report and attribute to the
  tilting of ambient cross-flow vorticity.
- **The split width is not yet a number to quote.** 206 m at the last sample, but
  the plane it is measured on is 50 m from a prescribed-profile outflow, and the
  value is still moving between samples. The ordering across `z0` and `Q0` is the
  criterion worth testing first: an ordering survives a boundary offset that an
  absolute width does not.

Running the sweep also needs `module load cuda/12.6.3 gcc/11.2.0` and a login
shell. `gcc/11.2.0` is what supplies the `GLIBCXX_3.4.29` the binary links
against; the cuda module only adds its own `lib64`, and `module` itself is a
shell function that a non-login `sbatch` shell does not have.

### The z0 and Q0 sweep

Same coarse grid, `mu = 4 kg/(m s)`, 600 s, split width read on the `x = 1750 m`
plane at `z = 25 m`. Widths are quantised to `dx = 18.75 m`, so the useful
content here is the ordering, not the value.

| `z0` \ `Q0` | 1000 W/m^3 | 500 W/m^3 |
|---|---|---|
| 50 m | 131.2 m | 112.5 m |
| 100 m | 206.2 m | 131.2 m |
| 150 m | 243.8 m | 206.2 m |

**The `z0` ordering reproduces.** Both columns are monotone: a deeper cross-flow
shear layer gives a wider split, which is the paper's principal result from this
set.

**The `Q0` ordering does not.** Cunningham report a *wider* bifurcation for the
*weaker* source; every row here has the weaker source narrower. Before reading
that as a physics failure, note two things.

First, the two are not the same measurement. The paper's Fig. 6 shows the
bifurcation of the **potential-temperature** cross-section; this table is the
spanwise separation of the two **`omega_z` extrema**. Those track each other
loosely at best, and comparing them as if they were the same number is not a
fair test. The second cut measures the theta bifurcation directly; see below for
what that changes and what it does not.

Second, the runs do support the mechanism the paper conjectures for it -- that
the width is set by how long a buoyant parcel takes to rise through the shear
layer. The weak-source pairs form measurably further downstream: `x` = 797, 853
and 1059 m against 703, 722 and 703 m for the strong source. A slower rise puts
the pair further down the domain, which is exactly the picture. What that does
to the width at one fixed plane is a different question.

### The viscosity sweep says nothing here -- but the measure was part of why

`z0 = 100 m`, `Q0 = 1 kW/m^3`, `mu` = 4, 1, 0.15, 0.0015 kg/(m s) -- a range of
2700 in the physical viscosity:

| `mu` | split @ 1750 m | strongest plane | `|omega_z|` there |
|---|---|---|---|
| 4 | 206.2 m | x = 703 m | 0.0216 1/s |
| 1 | 206.2 m | x = 703 m | 0.0244 |
| 0.15 | 206.2 m | x = 703 m | 0.0253 |
| 0.0015 | 206.2 m | x = 703 m | 0.0255 |

Identical split, identical plane, and `|omega_z|` moving by 18% across a factor
of 2700. The paper's runs go from a laminar symmetric bifurcation at `mu = 4` to
a turbulent asymmetric plume at `mu = 0.0015`; nothing like that happens here.

**Read the "identical split" column with care.** It is quantised to `dx`, and
the second cut shows that quantisation was hiding a response that was present
all along -- the interpolated `theta` width moves monotonically over the same
four runs. The floor conclusion below survives; the claim that the collapse
needed no further evidence does not.

The reason is the one D1 made measurable: **the numerical dissipation floor sits
above all but the largest `mu`**. At `dx = 18.75 m` the scheme's own dissipation
is what sets the effective Reynolds number, so asking for a smaller `mu` changes
nothing. A sweep whose parameter is below the floor is not a sweep. The four identical
rows are suggestive rather than conclusive, though: see the second cut, where
the same four runs measured on `theta` give a monotone, saturating series that
locates the floor instead of merely asserting it.

This is a resolution statement, not a solver defect, and it is the same shape as
the D1 and D3/D4 results: the criterion is bound to the grid. Reproducing the
paper's Reynolds-number progression needs the 10 m grid at least, and probably
the same `nu = 0` control run D1 uses to put a number on the floor.

## Stage A, second cut: measuring what the paper measures

The first cut left four things to do, in order. Items 1 and 4 are done here;
item 2 is the resolution the runs below are at; item 3, an outflow condition,
is separate work with its own verification case and is not attempted here.

### The theta bifurcation, read the way Fig. 6 is read

The `Q0` comparison failed above because the two sides were not the same
quantity. Going back to the paper settles what the right one is. Fig. 6's
caption reads: *cross sections of potential temperature in the y-z plane at
x = 1750 m ... contour interval is 0.25 K with the first plotted contour equal
to 300.25 K.* Both orderings the paper draws from that figure -- wider for the
deeper shear layer, wider for the weaker source -- are read off those contours,
and off nothing else.

So the diagnostic measures the same thing:

- The plane is reduced to the column maximum `P(y) = max_z theta(y, z)`.
- **`theta_width`** is the distance between the outermost `P = 0.25 K`
  crossings, the paper's first plotted contour. The crossings are linearly
  interpolated, so the width is no longer quantised to `dx` the way the
  `omega_z` extrema separation was -- that quantisation is what made four
  viscosity runs report identical widths to the metre.
- **`theta_split`** is the distance between the two outermost maxima of `P`,
  with the peak positions refined by the parabola through each maximum and its
  neighbours.
- Two lobes count as a **bifurcation** only when the saddle between them lies at
  least one contour interval -- 0.25 K, the figure's own interval -- below the
  lower of the two peaks. That is the level at which Fig. 6 would draw them as
  two closed contours rather than one lobed one, so it is the paper's own
  threshold rather than an invented one. A single-lobed plane is reported as
  such instead of returning a number that looks like a width.

All of it is reported twice: on the requested plane, and again on the plane
where the counter-rotating pair is strongest, which sits far enough upstream to
be clear of the prescribed outflow.

### The buoyancy and the theta advection no longer read `init_u_`

`ReinitAsync` leaves `init_u_` current only at the end of a cycle; anywhere else
inside one it still holds the cycle's start. That is why the first cut was
pinned to `reinit_every = 1`. Two selectors fix it, since `AdvanceAsync` stores
the step it has just taken in `mid_u_[step_ % reinit_every_]` and *then*
increments `step_`, so the cycle index reads differently on the two sides of the
call:

- `PlumeVelocityBefore` -- the latest projected velocity, for building the
  buoyancy and the drag: `init_u_` on the cycle's first step, `mid_u_[i-1]`
  after that.
- `PlumeVelocityAfter` -- the velocity of the step just taken, `mid_u_[i]`, for
  advecting `theta` across it.

The second is not just a workaround. `ReinitAsync` marches the flow map across a
full `dt` per step using `mid_u_[i]`, which is to say the solver already treats
those buffers as step-midpoint velocities; using the same buffer for the
semi-Lagrangian `theta` update gives that update its second-order transport,
where the old code used the end-of-step velocity. So the change moves the `n = 1`
answer slightly as well, and the sweep below was re-run rather than carried over.

`ReinitAsync` is now also called once per cycle rather than once per step, which
is what it was always meant for, and the driver refuses a `--diag-every` that is
not a multiple of `n` -- otherwise a measurement mid-cycle would read a stale
`init_u_` and report it without complaint.

### theta was the only field in the case with no error compensation

Measuring the bifurcation is what exposed this. The velocity in this solver is
carried by the flow map and then corrected by a BFECC pass with a clamp. `theta`
was advected by a bare call to `AdvectN2CAsync` -- an RK2 backtrace with
trilinear interpolation and nothing else. That interpolation is first order in
space, and it is applied 2400 times over a 600 s run.

So the field every one of Cunningham's criteria is read from was the least
accurate field in the simulation, by a wide margin, and the first thing the new
diagnostic reported was a single lobe at every resolution and every case: two
branches that exist in `omega_z` but are smeared into one hump in `theta`.

`AdvectThetaAsync` applies to `theta` the same correction the solver applies to
velocity, in the same order -- advect, advect back, difference against the
start, advect the error, subtract half of it, clamp against the uncorrected
result -- reusing the solver's own `BfeccClampAsync` for the last step.
`--theta-advection plain|bfecc|bfecc-clamp` selects it, so the uncorrected runs
above stay reproducible as the control.

### A run that reports zeros is worse than a run that fails

Trying to use the idle `ckpt` capacity turned up a failure mode worth recording.
`proj/selfcheck/xmake.lua` built cubin for `sm_75` and `sm_89` only, and `ckpt`
mixes GPU generations. On a node outside that pair every kernel launch fails
with *no kernel image is available for execution on the device* -- and nothing
in `src/ofm` or the harness checks launch status. A 200-step plume run therefore
finished in 0.3 s with every field exactly zero, and printed those zeros as
diagnostic rows. The only symptom was the wall time.

For a project whose criteria are all relative errors against published or
analytic values, a diagnostic that silently reports zeros is the worst possible
failure: a whole sweep of them would tabulate cleanly. Two changes:

- `CheckDeviceUsable` runs a one-line probe kernel at start-up, checks the
  launch, and aborts with the device name and compute capability if it did not
  run. Every case now prints the device it ran on, which also makes the logs
  self-documenting.
- `compute_75` PTX is added to the gencode list as a JIT fallback, so any device
  from Turing on runs rather than failing. PTX only JITs forwards, so anything
  older is still caught by the probe.

A sweep should still be pinned to a single partition. An ordering result read
off cases that ran on different hardware is not one.

### The paper's grid, and what the theta measure says there

`dx = 10 m` (184 x 120 x 152), `dt = 0.25 s`, 600 s, `n = 1`, plain semi-Lagrangian
`theta` -- the same configuration as the coarse runs above, at the paper's own
resolution. The 10 m grid is much less damped: peak `|omega_z|` rises from
0.021 to 0.054 1/s and the peak anomaly from 7.4 to 12.4 K.

| `z0` | `theta` width, `Q0` = 1000 | `Q0` = 500 |
|---|---|---|
| 50 m | 515.2 m | 357.4 m |
| 100 m | 523.5 m | 331.0 m |
| 150 m | 553.4 m | 342.1 m |

At `Q0` = 1 kW/m^3 those three are monotone -- 515.2, 523.5, 553.4 m -- and it is
tempting to call the `z0` ordering reproduced. **It is not, and the reason is in
the next section: a single snapshot is not a measurement here.** At
`Q0` = 0.5 kW/m^3 the same three are not even monotone: 357.4, 331.0, 342.1 m.

**But none of these six is bifurcated.** Every one is a single lobe: the tallest
two local maxima are separated by a dip of a few hundredths of a kelvin where
one contour interval is 0.25 K. So `theta_width` above is the width of the
plume's 0.25 K outline, not the separation of two branches, and it carries a
confound the branch separation would not: an absolute 0.25 K contour encloses
less of a half-strength source no matter what the bifurcation does. **The `Q0`
result cannot be tested until the bifurcation itself is resolved**, and that is
a sharper statement of the failure than the first cut's -- which blamed a
mismatched quantity, correctly, but could not say what the right quantity would
show.

The coarse grid does not bifurcate either, and there the two lobes are usually
not even distinguishable: `theta_peak` and `theta_saddle` come back as one
maximum in most cases.

### One snapshot is not a measurement: the plume is still unsteady at 600 s

The paper reads Fig. 6 at "the time at which the flows are essentially steady
(typically achieved after approximately 600 s)", and the case was built to match
that. Comparing consecutive diagnostic samples shows these runs are not steady
there. Between `t` = 450 s and `t` = 600 s the `theta` width moves by:

| | range over the cases in that sweep |
|---|---|
| `n` = 1 | 4.1% to 19.7% |
| `n` = 5 | 27.7% to 82.8% |

**The between-case differences the orderings were read from are smaller than the
within-case variation of the individual numbers.** The `z0` spread at
`Q0` = 1 kW/m^3 is 7.4% end to end while its three members individually move
4.1%, 17.1% and 11.0% between the last two samples. Nothing can be concluded
from single snapshots here, in either direction.

This is not a defect of the theta measure -- the `omega_z` widths move as much,
it is just that quantising them to `dx` made them look stable. It is a statement
about the flow: Cunningham themselves report a quasi-periodic oscillation of the
bifurcation with a period near 200 s, so sampling one instant near 600 s samples
one phase of it.

Diagnostics cost far less than the step -- a whole 600 s case is about 90 s of
wall clock -- so sampling every 30 s instead of every 150 s is nearly free. The
sweep was re-run that way, and the next section is what it says.

### The measurement that settles both criteria

Six Fig. 6 cases at the paper's grid, sampled every 30 s, averaged over the last
200 s (7 samples), run twice: once with the bare semi-Lagrangian `theta` step
and once with BFECC + clamp. Everything else identical, so the pair isolates the
scheme. Uncertainties are the standard error of the 7 samples.

**What the scheme changes.** It is not a detail:

| case | width plain | width BFECC | peak dT plain | BFECC | bifurcated, plain -> BFECC |
|---|---|---|---|---|---|
| `z0` 50, `Q0` 1000 | 539.5 | 738.2 | 11.80 K | 26.77 K | 0.00 -> 1.00 |
| `z0` 50, `Q0` 500 | 375.2 | 579.8 | 6.00 | 15.20 | 0.00 -> 1.00 |
| `z0` 100, `Q0` 1000 | 580.1 | 943.2 | 12.39 | 27.94 | 0.00 -> 0.86 |
| `z0` 100, `Q0` 500 | 356.0 | 616.0 | 6.30 | 16.66 | 0.00 -> 1.00 |
| `z0` 150, `Q0` 1000 | 612.4 | 1125.4 | 12.77 | 28.10 | 0.00 -> 1.00 |
| `z0` 150, `Q0` 500 | 323.7 | 678.8 | 6.62 | 17.41 | 0.00 -> 1.00 |

The uncompensated step was destroying more than half the plume's thermal
amplitude, and with it the buoyancy that drives the whole flow. **With `theta`
error-compensated the plume bifurcates in essentially every sample of every
case; without it, in none.** The first cut's "no bifurcation anywhere" was a
statement about the advection scheme, not about the physics.

The measurement also gets much steadier: the width's sample scatter falls from
6-14% to 1-5%, because the field being measured is no longer being ground down
between samples.

**`z0`: the paper's principal result reproduces.** Bifurcation split, BFECC:

| `z0` | `Q0` = 1 kW/m^3 | `Q0` = 0.5 kW/m^3 |
|---|---|---|
| 50 m | 477.5 +- 3.0 m | 386.5 +- 3.1 m |
| 100 m | 565.3 +- 10.9 m | 421.9 +- 6.3 m |
| 150 m | 668.1 +- 20.0 m | 464.9 +- 29.2 m |

Both columns are monotone. End to end the strong-source column gains 190.6 m,
**9.4x the combined standard error**; the weak-source column gains 78.4 m,
2.7x. Deeper cross-flow shear gives a wider bifurcation, measured on the
quantity Fig. 6 plots, at the grid Fig. 6 was computed on, with an error bar.

**`Q0`: it does not, and now that is a real result.** The paper reports a wider
bifurcation for the weaker source (their Fig. 6e against 6f). Every case here is
the other way, on the same quantity:

| `z0` | weak minus strong, split | in standard errors |
|---|---|---|
| 50 m | -91.0 m | 21.2x |
| 100 m | -143.3 m | 11.4x |
| 150 m | -203.2 m | 5.7x |

This is no longer a mismatched-quantity problem, and it is far outside the
scatter. Either the reduction drops something the effect needs -- the volume
expansion is the obvious candidate, since the peak anomaly is now 28 K and
`dT/T0` is O(0.09) -- or the prescribed outflow is distorting the plane. Both
are named gaps; this is the first Stage A result that puts weight on which.

**One caveat on the widest case.** `z0` = 150 m at `Q0` = 1 kW/m^3 has a 0.25 K
outline 1125 m across in a domain 1200 m wide, so its *width* is close to the
free-slip lateral walls, where Cunningham had Orlanski outflow. The *split* is
an interior measurement and much less exposed, which is one more reason to read
the split rather than the width -- but the `z0` = 150 row should be re-run in a
wider domain before it is quoted on its own.

### `n` = 5 is markedly less dissipative, as the D2 residual argued

The rewiring makes `reinit_every = 5` runnable, and it does what raising the
reinitialization interval is supposed to do. At the fine grid, peak `|omega_z|`
on the strongest plane:

| case | `n` = 1 | `n` = 5 |
|---|---|---|
| `z0` = 50 m, `Q0` = 1000 | 0.0541 | 0.0695 (+28%) |
| `z0` = 100 m, `Q0` = 1000 | 0.0539 | 0.0703 (+30%) |
| `z0` = 150 m, `Q0` = 1000 | 0.0534 | 0.0731 (+37%) |

That is the same direction as what D1 measured on the Burgers vortex, now on a
real configuration. The cost is that the livelier flow is also the noisier one
to sample, which is the previous section's problem in a sharper form.

### The viscosity sweep at 10 m: it responds, and it still saturates

The first cut read four identical rows and concluded the sweep measured nothing.
That was half right. The `omega_z` extrema separation it tabulated is quantised
to `dx`, which hid a response that was there; the interpolated `theta` measure
shows it at both resolutions.

| `mu` kg/(m s) | peak `|omega_z|`, 18.75 m | 10 m | `theta` width, 10 m |
|---|---|---|---|
| 4 | 0.0215 | 0.0539 | 524.3 m |
| 1 | 0.0244 | 0.0654 | 494.9 m |
| 0.15 | 0.0253 | 0.0694 | 499.3 m |
| 0.0015 | 0.0254 | 0.0700 | 487.1 m |

The shape is the same at both grids and it is the shape a dissipation floor
makes: the response is real between `mu` = 4 and 1, weaker from 1 to 0.15, and
gone below that. Step by step the coarse column gains 13.5%, 3.7%, 0.4%; the
fine column gains 21.3%, 6.1%, 0.9%. End to end that is 18.1% coarse against
**29.9% fine**.

So refining the grid by 1.9x moved the floor down and roughly doubled how much
of the sweep is above it -- but **the floor at `dx = 10 m` still sits above
`mu` = 0.15**. Cunningham's progression from a laminar symmetric bifurcation at
`mu` = 4 to a turbulent asymmetric plume at `mu` = 0.0015 needs the bottom of
that range to be the physical viscosity, and here it still is not. The paper's
own grid is not sufficient for the paper's own Reynolds-number series in this
solver.

This is a sharper claim than the first cut's, and it comes with a number rather
than an assertion. It does not need a separate `nu = 0` run to support it: the
saturation between consecutive `mu` values is the floor being crossed.

### Run-to-run reproducibility

The AMGPCG solve accumulates with atomics, so two runs of the same case are not
bit-identical. There are two independent measurements of the scatter:

- The coarse sweep was run twice on the same node. Of ten cases, nine returned
  the same `theta_width` to the printed 0.1 m and one moved by 0.1 m in 477.5 m,
  0.02%. Every `omega_z` extrema separation was identical.
- The fine sweep contains an accidental replicate: `shear_z100_q1000` and
  `visc_mu4` are the same configuration, run separately. They give 523.5 m and
  524.3 m, a spread of **0.15%**.

The fine-grid figure is the one to read the orderings against. The `z0` spread at
`Q0` = 1 kW/m^3 is 7.4% from end to end, comfortably outside it; the `Q0` = 0.5
spread is 8%, but non-monotone, so its middle point is the one at issue rather
than its size. No number here is quoted past a tenth of a metre.

### Regression

`use_uniform_inlet_` defaults to `true`, so `AdvanceAsync` behaves exactly as
before for every case that does not ask otherwise. Nothing in `src/ofm` changed
in the second cut either -- the rewiring is all on the harness side -- and the
D1 Burgers case confirms both: on the recorded 256^3 configuration it returns
`-0.40%`, `-0.38%`, `-0.33%`, `-0.30%` at its four samples, which is the
recorded table row for row.

### Where Stage A stands

| the paper's claim | status |
|---|---|
| counter-rotating pair, positive `omega_z` on the right looking downstream | reproduced (first cut) |
| plume cross-section bifurcates | reproduced, but only with `theta` error-compensated |
| deeper shear layer -> wider bifurcation | **reproduced**, 9.4x sem at `Q0` = 1 kW/m^3 |
| weaker source -> wider bifurcation | **fails**, opposite by 5.7-21.2x sem |
| laminar-to-turbulent progression over `mu` | not reachable: the dissipation floor at 10 m is still above `mu` = 0.15 |
| `St` ~ 0.25 shedding | not tested |
| cross-section not Gaussian | not tested |

### What is next, in order

1. **A wider domain for the `z0` = 150 m cases**, whose 0.25 K outline is within
   75 m of the free-slip lateral wall. Cheap, and it either confirms that row or
   removes it.
2. **An outflow condition.** Everything measured at `x = 1750 m` is 50-90 m from
   a prescribed-profile face. It needs its own verification case, which is why
   it is separate work rather than part of Stage A -- but the `Q0` failure now
   gives a concrete reason to want it.
3. **The CVP's vorticity attribution.** Cunningham only assert that tilting
   dominates; D2 exists to put a number on it. The first of the two things
   Stage A is meant to deliver that the paper does not have.
4. **The horseshoe vortex**, which Cunningham and Barata 2024 each missed for
   the same near-wall resolution reason. The second deliverable, and the one
   that will need the finest grid.

## Is there an optimal reinitialization interval? Two errors, charged two ways

D1 §8 concluded that "numerical dissipation is charged per reinitialization, not
per unit of simulated time", and drew from it the rule *raise `reinit_every`
rather than lower `dt`*. That conclusion rested on two measurements taken on
different cases with different metrics: coarsening `dt` at `n = 1` (1/120 to
1/480, four times fewer reinitializations) moved the floor by 3.55x, while
raising `n` at fixed `dt` (1 to 10, ten times fewer) moved the ring case's peak
vorticity by only 2.56x. Written as `E ~ N_reinit^alpha` with
`N_reinit = T/(n dt)`, those give `alpha = 0.91` and `alpha = 0.41`, which would
mean the two knobs are not interchangeable and no one-line rule exists.

They are interchangeable. The disagreement was an artefact of comparing two
cases and two metrics, and the sweep below replaces both with one surface
measured on one case with one metric.

### The sweep

D1's columnar Burgers vortex, the only case here with a closed-form answer
(`b(t)^2 = b0^2 + 4 nu t`, read back self-normalised as
`nu_eff = b0^2 (w0/w - 1)/(4t)`), on 128^3, with the simulated time held at
`T = 1 s` in every run so that "per unit time" is the same axis throughout.
Every `n` divides every step count, so the last diagnostic lands exactly on
`t = 1` rather than at the first cycle boundary past it -- D1 §8's own two floor
numbers were sampled at `t = 1.01` and `t = 0.75`, which is why the 3.55x there
is not a clean ratio.

Four sets: the floor at `nu = 0` over `dt` in {1/240, 1/480, 1/960, 1/1920} and
`n` in {1, 2, 4, 8, 16, 48, 120, 240}; the same grid at `nu = 1e-3` and at
`nu = 3e-4`; the same `n` ladder at twice the circulation; and a timing ladder
with diagnostics off. 112 runs, 52 minutes on one RTX 6000.

### The dissipation floor is a function of `n*dt` alone

Configurations sharing a cycle length agree to better than half a percent, over
a 200x range of cycle lengths:

| `n*dt` | `(n, 1/dt)` | `nu_eff` |
|---|---|---|
| 1/480 | (1, 480) / (2, 960) / (4, 1920) | 7.0246e-5 / 7.0260e-5 / 7.0412e-5 |
| 1/240 | (1, 240) / (2, 480) / (4, 960) / (8, 1920) | 3.6008 / 3.6043 / 3.6042 / 3.6131 e-5 |
| 1/60 | (4, 240) / (8, 480) / (16, 960) | 1.0660 / 1.0689 / 1.0643 e-5 |

`nu_eff * n * dt` stays between 1.40e-7 and 1.90e-7 while `N_reinit` varies by a
factor of 1000. So the floor is very nearly a fixed charge per
reinitialization, and **raising `n` and coarsening `dt` are the same lever**;
only their product matters. `n = 1`, `dt = 1/480` returns 7.0246e-5, which
reproduces D1 §8's 7.02e-5 exactly and makes this a regression on that number as
well.

Doubling the circulation -- hence the velocity and the strain rate -- at fixed
`dt` leaves the floor unchanged to within 1-4% for `n` up to 120. The floor
therefore follows the cycle's *duration*, not the distance the map is marched
across it. Of the two rules that coincide at fixed `dx` and fixed flow,
`n*dt = const` (which is what LFM's driver enforces, `dt = 1/(frame_rate *
reinit_every)`) is the one the floor obeys; `n*sigma = const` is not.

The collapse degrades at the two longest cycles (`n` = 120 and 240 at the
coarsest `dt`), where the peak vorticity changes by only parts in 10^5 over the
whole run and the estimator is closer to its own resolution. Those corners are
reported but not leaned on.

### The source-term channel is charged the opposite way: per sub-step

The `nu = 0` floor cannot see the one error that grows with cycle length. The
path integral accumulates the source into the impulse through the forward map,
so a longer cycle pulls it through a longer composition. Splitting the viscous
error into the floor, which pushes `nu_eff` up, and whatever is left, which
pulls it down --

    rel_err = floor/nu - deficit

-- gives a deficit that is flat in `dt` and grows with `n`:

| `n` | `dt`=1/240 | 1/480 | 1/960 | 1/1920 | spread |
|---|---|---|---|---|---|
| 4 | 5.03% | 5.23% | 5.49% | 6.30% | 25% |
| 8 | 5.67% | 5.81% | 6.00% | 6.26% | 10% |
| 16 | 7.31% | 7.27% | 7.38% | 7.55% | **4%** |
| 48 | 14.68% | 13.95% | 13.65% | 13.60% | 8% |
| 120 | 32.05% | 29.20% | 27.84% | 27.19% | 18% |
| 240 | 52.86% | 48.51% | 46.20% | 44.97% | 18% |

Eight times more simulated time per cycle changes the deficit by a few percent
of itself; four times more sub-steps roughly doubles it. The growth is described
by a per-sub-step fractional loss, `deficit = base + 1 - (1 - eps)^n`. Fitting
`eps` to the `n` = 48 and 240 pair at `dt` = 1/1920 gives **`eps` = 0.224% at
`nu` = 1e-3 and 0.227% at `nu` = 3e-4 -- the same rate at both viscosities**,
which is what a property of the scheme rather than of the case should look like.
The fit is a two-point one, so the check is the rows it was not fitted to: it
puts `n` = 120 at 27.0% against 27.2% measured and 31.9% against 33.5%, and
`n` = 16 at 6.9% against 7.6% and 11.6% against 10.2%. Right shape, one to two
points of slack.

The `base` offsets differ (3.4% and 8.1%) and are the unreliable part:
subtracting a floor measured at `nu = 0` from a run at `nu > 0` is only
approximate, because the two are not the same field. The `n` = 1 and 2 rows,
where `floor/nu` reaches 27%, are unreliable for the same reason and are omitted
above.

So the two error terms are charged on different axes. The floor is a fixed
price per reinitialization and falls as `1/(n dt)`. The source deficit is a
fixed price per sub-step of the cycle and grows as `1 - (1-eps)^n`. Nothing
collapses the pair onto one parameter, and `dt` and `n` are interchangeable for
the first but not the second.

### The error zero is a cancellation, not an optimum

The two terms have opposite signs, so the total error passes through zero. It is
tempting to read that zero as the optimal interval, and at one viscosity it even
looks like a law: the crossing sits at a fixed cycle *duration*, so `n*` scales
as `1/dt`.

| | `nu` = 1e-3 | `nu` = 3e-4 |
|---|---|---|
| `dt` = 1/240 | below `n` = 1 | `n*` = 2.10, `n* dt` = 0.0087 |
| `dt` = 1/480 | `n*` = 1.41, `n* dt` = 0.0029 | `n*` = 3.85, 0.0080 |
| `dt` = 1/960 | `n*` = 2.59, 0.0027 | `n*` = 6.89, 0.0072 |
| `dt` = 1/1920 | `n*` = 4.66, 0.0024 | `n*` = 11.60, 0.0060 |

Each column is nearly constant, which is the law. But the two columns are not
the same law: dropping `nu` by 3.33x moves the crossing by 2.8x, which is what
`floor/nu ~ 1/(nu n dt)` predicts if the deficit does not depend on `nu`. **The
zero is where numerical dissipation happens to cancel the source deficit, and it
moves with `nu`.** It is not a property of the scheme and must not be quoted as
an optimal interval.

What is a property of the scheme is the pair of scalings. Given a target, they
say what to do: the floor is bought down by lengthening the cycle in *time*, by
either knob; the source channel is paid for in *sub-steps*, so at a fixed cycle
duration the cheaper configuration is the one with fewer, longer steps.

### No stability limit appeared, which was the other expectation

The leapfrog ring case had both `n` = 10 runs go non-finite at step 860, which
suggested that `n` is bounded above by stability and that a cheap error-based
controller would spend its time avoiding that bound. Nothing here hit it. All
112 runs completed and produced a finite diagnostic at `t` = 1, including
`n` = 240 at `dt` = 1/240, which is one cycle spanning the entire run: the flow
map is marched 240 sub-steps from identity and never reset. It is accurate to
0.002% on the floor and wrong by 53% on the source term, but it is stable.

So the ceiling on `n` in this case is set by the source channel's accuracy, not
by stability. The ring case's blow-up came after the rings had hit the far wall
and the peak vorticity had reached 160; it is a property of that violent field,
not of long cycles as such. Whether a Stage A or Stage B configuration has a
stability ceiling low enough to matter is still open, and it is a different
question from this one.

### Cost saturates by `n` = 8

Wall clock for 1 s of simulated time, diagnostics off:

| `n` | 1 | 2 | 4 | 8 | 16 | 48 | 120 | 240 |
|---|---|---|---|---|---|---|---|---|
| s | 19 | 16 | 14 | 13 | 13 | 12 | 13 | 13 |

A 32% saving, all of it collected by `n` = 8. That is the shape the structure
predicts: `ReinitAsync` marches O(`n`) sub-steps per cycle
(`ofm.cu:233,251`), so the marching work per unit simulated time does not depend
on `n`, and what raising `n` removes is one Projection 2, one map reset and one
BFECC pass per cycle -- a term that decays as `1/n` onto a floor. The timer has
1 s resolution, so the last three columns are one value, not three.

### What this changes

1. **D1 §8's rule "raise `reinit_every` rather than lower `dt`" is wrong as a
   statement about the floor.** The floor depends only on `n*dt`, so the two are
   the same lever. What is true is the weaker statement D1 actually measured:
   refining `dt` at fixed `n` makes the solution more diffusive, because it
   shortens the cycle. Refining `dt` while holding `n*dt` fixed does not.
2. **Raising `n` has a cost that none of the three arguments for
   `reinit_every = 5` counted.** They were dissipation, speed, and the D2
   dual-path residual, all of them on the floor side. The source deficit is on
   the other side: at 128^3 it charges 0.224% per sub-step, so `n` = 5 gives up
   1.1% of the source term. Since the deficit is an interpolation error it falls
   with `dx` -- D1's own `n` = 1 numbers imply about 5.3% at 128^3 against about
   1.5% at 256^3, a factor of 3.7 for a doubling, so close to second order -- so
   at 256^3 `n` = 5 costs about 0.3%. That does not overturn `n` = 5, but it is a term that
   belongs in the ledger, and it is the term that will decide how far above 5 it
   is safe to go.
3. **There is no viscosity-independent optimal `n` to compute.** The question
   "what is the best interval" has no answer without naming which error is
   being minimised and at what resolution.

### Reproducing

```
sbatch proj/selfcheck/reinit_dt_sweep.sbatch          # all four sets
SET=nuvisclo BUILD=0 sbatch proj/selfcheck/reinit_dt_sweep.sbatch
python3 proj/selfcheck/reinit_dt_summary.py /gscratch/amath/diwenxu/wildfire-sim-runs/reinit-dt
```

In a fresh worktree, `git submodule update --init --recursive` first, from the
login node: the submodules start out empty and the build dies on a missing
`amgpcg.h`.

Output, including the tabled summary, is under
`/gscratch/amath/diwenxu/wildfire-sim-runs/reinit-dt`.

## Stage A, third cut: an open boundary, verified on its own

The second cut ended with two named reasons to want an outflow condition: the
paper's measurement plane sits 50-90 m from a face on which the first two cuts
prescribed the cross-flow profile, and the failed `Q0` ordering had that face as
one of its two candidate explanations. This cut adds the condition, verifies it
on an exact solution as the project's rules require, and then re-reads the Fig.
6 orderings with it. It also adds two things Cunningham's direct runs have that
the case did not: thermal diffusion of `theta` at `Pr = 0.7`, and the damping
layer under the lid.

### The condition was already in the Poisson solver, hidden

Reading `SetCoefByIsBcKernel`, `ApplyPressureXKernel` and the AMGPCG matvec
together: a staggered face marked in `is_bc_*` carries a prescribed normal
velocity and the pressure sees a homogeneous Neumann condition across it -- the
diagonal count drops by one. A domain face that is *not* marked keeps its
diagonal entry, the matvec finds no neighbour beyond it, and `ApplyPressure`
reads the exterior pressure as zero. That is a Dirichlet `p = 0` cell beyond the
face: the projection sets the face's normal velocity from the pressure gradient
and mass leaves or enters at whatever rate the interior asks for. It is the
zero-gauge **pressure outlet** that Barata et al. 2024 (Sec. 2.1) and 2025 (Sec.
2.1) put on their lateral, top and outlet faces in FLUENT, and the `p_ext = 0`
form of the FDS OPEN boundary (FDS Technical Reference Vol. 1, Sec. 4.3.3; FDS
pins its `H = p/rho + |u|^2/2` rather than `p`, but with `p_ext = 0` the
outgoing pressure is the same zero). The graphics solvers this code descends
from never use it because they mark all six faces as walls.

So the solver-side change is small: `SetDomainFaceAsync` clears the mark on one
whole domain face, `ofm.h` documents the exterior-pressure convention, and the
one trap is `AMGPCG::pure_neumann_`, which defaults to `true` and subtracts the
mean of the right-hand side before every V-cycle. That is right for the
singular all-Neumann operator and wrong the moment the pressure level is
pinned; it is switched off whenever a face is open.

What this is **not** is Cunningham's Orlanski (1976) radiation condition, which
extrapolates a phase speed from the interior and advects the boundary value at
it. Orlanski 1976 is not in the corpus, so it was not implemented from memory;
the pressure outlet is the condition two corpus papers use for exactly this
kind of plume. The tangential velocity at an open face is whatever the
semi-Lagrangian step lands there -- its backtrace is clamped to the domain, so
this is the zero-gradient condition FDS Sec. 4.4.5 describes for outflow.

### The verification case: a vortex column leaving through the face

A boundary condition is verified by a flow that crosses it with a known answer.
The case is the Gaussian vortex column of D1 and D3 (Tohidi et al. 2018 Eq. 7,
`omega = Gamma/(pi a^2) exp(-r^2/a^2)`), uniform in `z` between free-slip
walls, carried by a uniform stream `U` along `+x` through the downstream face.
In an unbounded inviscid fluid it translates unchanged, so the solution is
known at every time. The inflow and lateral faces prescribe that solution's
normal velocity, updated every step from the exact centre position, so only the
downstream face carries the condition under test.

The condition is not exact for this flow, and that is the point: a vortex core
is a pressure minimum, and the outlet pins the pressure to zero, so as the core
crosses the face the boundary must do something the exact solution does not.
The case measures how much. Three runs make the measurement:

| run | downstream face | role |
|---|---|---|
| open | `p = 0` | the condition under test |
| closed | `u = U` prescribed | what the first two cuts did to the plume |
| long | twice as far away, never reached | the solver's own error on the same nodes |

Box `2 x 1 x 1/8` at `dx = 1/128` (256 x 128 x 16), `U = 1`, `a = 0.05`
(6.4 cells), `Gamma = 0.25` (peak swirl about `U/2`), centre starting at
`x = 0.75`, `dt = 1/384`, run to `t = 1.75` when the exact centre is ten core
radii past the face. Vorticity is taken at the cell corners from the face
velocities directly, so the sum over the corner set is the exact discrete
circulation around the rectangle half a cell inside the boundary, and the
analytic value is the Gaussian's integral over that same rectangle. Reported
per sample: circulation still inside against exact, the L2 error of the
vorticity field over the whole rectangle and over the interior `x < x_out - 4a`
(normalised by the exact field's norm `Gamma/(a sqrt(2 pi))`), the peak, and
the largest `|w|`, which the exact solution has at zero.

The pass criteria below are **proposed** here, not taken from the plan
Artifact, which asked for a verification case without naming one; they are
written so that the closed box, the treatment being replaced, must fail them.

### Result: transparent to what stays inside, late by a fraction of a core on the way out

Runs at `n` = 1 and `n` = 5 (steps and sampling adjusted to whole cycles at
`n` = 5), on `gpu-rtx6k`. First, what the solver does to this vortex with no
boundary in the way -- the long box, on the short box's nodes:

| `n` | peak vorticity when the centre reaches `x = 2` | interior L2 error there | circulation *outside* the core box before the face |
|---|---|---|---|
| 1 | 0.830 of exact | 0.106 | +0.105 `Gamma` |
| 5 | 0.926 of exact | 0.045 | +0.103 `Gamma` |

The core is conserved: summed over a box of half-width `6a` about the exact
centre the circulation stays at `0.25000` to four places until the face is
reached, in every run. The peak decay and the L2 error are the dissipation
floor D1 measured, again smaller at `n` = 5. The third column is new and is
**not** the outflow's: it is identical in the open, closed and long boxes, it
grows from the first step, and it stays behind (`+0.029 Gamma`, still creeping
up) after the vortex has gone. It lives outside the core box, so along the
walls or the inflow; the third verification run below localises it. It is a
solver finding about the free-slip lateral walls with a prescribed normal
velocity, recorded here and taken out of every comparison by reading the open
and closed boxes against the long box on the same nodes rather than against
the exact field.

Then the boundary itself, as open minus long at every sample:

| `n` | criterion | open | closed control |
|---|---|---|---|
| 1 | interior L2 error added while the core is in transit (`x < x_out - 4a`) | **+0.007** | +0.062 |
| 1 | circulation left behind after exit | **-0.0002 `Gamma`** | -0.0001 `Gamma` |
| 1 | peak vorticity left behind after exit | **0.0022** of the initial peak | 0.0022 |
| 1 | largest `|Gamma_in - Gamma_in(long)|` during the crossing | 0.070 `Gamma` | 0.025 `Gamma` |
| 1 | peak / long-box peak as the centre reaches the face | 1.095 | 1.049 |
| 5 | interior L2 error added while in transit | +0.025 | +0.104 |
| 5 | circulation left behind after exit | **+0.0000 `Gamma`** | +0.0002 `Gamma` |
| 5 | peak vorticity left behind after exit | **0.0021** | 0.0021 |
| 5 | largest `|Gamma_in - Gamma_in(long)|` during the crossing | 0.108 `Gamma` | 0.034 `Gamma` |
| 5 | peak / long-box peak as the centre reaches the face | 1.099 | 1.035 |

Read row by row:

- **Nothing reflects.** After the vortex has left, the open box holds the same
  circulation as the long box to `2e-4 Gamma` and the same residual peak
  (0.2% of the initial, the wall-row noise). The closed box also holds nothing
  afterwards, but only because it never let anything through: its vortex is
  ground down against the prescribed face instead.
- **The interior is untouched at `n` = 1** (+0.7% of the field norm, against
  +6.2% for the closed box), and lightly touched at `n` = 5 (+2.5%, against
  +10.4%). The closed box's contamination is what the second cut's measurement
  plane, 5 cells from such a face, was sitting in.
- **The crossing is late.** As the exact centre reaches the face the long box
  has 0.140 `Gamma` left inside the measured rectangle, the open box 0.158
  (`n` = 1); the core is held for a fraction of a core radius and its peak
  rises by 10% as it is pulled through -- the cost of pinning the pressure to
  zero across a pressure minimum. Both boxes are late; the closed box less so
  on this one number, and worse on every other.

Against the proposed thresholds (1% interior, 1% residual, 2% on the crossing
curve), the open face passes the first two at `n` = 1, fails the interior one
narrowly at `n` = 5, and fails the crossing one at both. The crossing
threshold does not separate the treatments -- the closed box "passes" it more
nearly while distorting the interior ten times more -- so it is the wrong
number to gate on, and the plan Artifact should say which of these it adopts.
The two that discriminate are the interior contamination and the residual;
on those the open face is transparent and the closed box is not. What the
crossing lag means for the plume is not a matter for this case: the plume's
own long-domain reference below measures it directly, on the quantity the
orderings are read from.

### The pressure outlet fails the plume, and why

With the face verified on the vortex, the six Fig. 6 cases were run with the
downstream face open, and again with the lateral faces open too. Every
`Q0` = 1 kW/m^3 case, and the widest `Q0` = 0.5 case with the lateral faces
open, went non-finite between 390 s and 450 s. The trigger is visible in the
diagnostics: the plume top reaches the 1500 m lid at about 360 s (it does in
the closed and long boxes too), and within 30-60 s the largest speed goes
from 10 m/s to several hundred. The weak-source cases, whose plume stays
below the lid at the face, ran through. Replacing the air that enters through
an open face with ambient air (the semi-Lagrangian backtrace is clamped to the
domain, so without that a cell the flow enters through keeps its own `theta`
and re-imports the plume's buoyancy from outside) was necessary but not
sufficient: the same case still failed at the same time.

The reason is what the vortex case already showed in miniature. A zero-gauge
outlet assumes the fluid beyond the face is ambient. In the Boussinesq
pressure a buoyant column under a lid carries a hydrostatic part -- `dp/dz =
g theta'/theta0`, so the pressure just under the lid is high by
`g (theta'/theta0) H` -- and when that column reaches a face where `p` is
pinned to zero, the whole of it becomes a horizontal gradient across one cell.
Cunningham's own words are that the outflow "occurs entirely on the lateral
and downstream boundaries", that is, the plume leaves through them; the
exterior there is plume, not ambient air. The pressure outlet is the wrong
class of condition for a flow that carries its own pressure field out of the
box, and this is the case Stage A needs. It is kept in the code as
`--outflow x|xy`, verified and documented, and not used for the orderings.

### The convective face

The condition Cunningham cite, Orlanski's, is a radiation condition: the
boundary value is advected out at a phase speed estimated from the interior,
and nothing is said about the pressure. The projection scheme here computes
the first half of that already -- the semi-Lagrangian step lands on every
open face the value `u(x - u dt)`, which is `du/dt + u_n du/dn = 0` with the
local normal velocity as the phase speed -- and the second half is a choice
about the projection. `OFM::convective_face_` makes it: the face keeps its
Dirichlet mark, but at every projection its prescribed value is refreshed
from the advected velocity, the set of convective faces is then shifted by
one constant so that the flux through the whole domain boundary nets to zero
(the pure-Neumann operator needs that, and it is the only global coupling the
condition has), and the pressure sees Neumann across the face as it does at a
wall. Nothing is pinned, so the plume's hydrostatic pressure crosses the face
with the plume. Orlanski 1976 is not in the corpus and the adaptive phase
speed was not implemented; this is the local-speed convective condition, and
it is named as such.

On the vortex case it behaves as the pressure outlet did for what stays
inside, and better on the way out (table below). On the plume, the case that
failed at 390 s under the pressure outlet runs to 600 s: the plume top reaches
the lid at 360 s, the warm layer under it is flushed through the downstream
face over the next 150 s, the largest speed never exceeds 11 m/s, and the
counter-rotating pair is on the plane at the end.

The vortex case, convective face against the long box on the same nodes, next
to the pressure outlet:

| `n` | criterion | convective | pressure outlet | closed control |
|---|---|---|---|---|
| 1 | interior L2 error added while the core is in transit | **+0.006** | +0.007 | +0.062 |
| 1 | circulation left behind after exit | **-0.0001 `Gamma`** | -0.0002 | -0.0001 |
| 1 | peak vorticity left behind after exit | **0.0022** | 0.0022 | 0.0022 |
| 1 | largest `|Gamma_in - Gamma_in(long)|` during the crossing | **0.038 `Gamma`** | 0.070 | 0.025 |
| 1 | peak / long-box peak as the centre reaches the face | 0.929 | 1.095 | 1.049 |
| 5 | interior L2 error added while in transit | **+0.008** | +0.025 | +0.104 |
| 5 | circulation left behind after exit | **+0.0001 `Gamma`** | +0.0000 | +0.0002 |
| 5 | peak vorticity left behind after exit | **0.0021** | 0.0021 | 0.0021 |
| 5 | largest `|Gamma_in - Gamma_in(long)|` during the crossing | **0.041 `Gamma`** | 0.108 | 0.034 |
| 5 | peak / long-box peak as the centre reaches the face | 0.980 | 1.099 | 1.035 |

It passes the interior and residual criteria at both `n`, adds less than 1% to
the interior at `n` = 5 where the pressure outlet added 2.5%, and its crossing
lag is half the pressure outlet's; the core is slightly damped on the way out
(peak 0.93 of the long box's at `n` = 1) rather than stretched. Runs in
`/gscratch/amath/diwenxu/wildfire-sim-runs/outflow-v4/`, read by
`analyse_outflow.py`.

### A multigrid trap found on the way: some tile counts give a non-finite solve

The wider reference box (23 x 30 x 19 tiles, so that the plume's flanks are
600 m further from the walls) returned a field that was non-finite after the
very first projection, with no CUDA error reported. Probing tile counts and
multigrid depths (`OFM::Alloc` takes `log2(min tile count) + 1` levels; each
level halves the tile count, rounding up):

| tiles | levels | coarsest level | initial projection |
|---|---|---|---|
| 23 x 15 x 19 (the paper's grid) | 4 (default) | 3 x 2 x 3 | fine |
| 23 x 15 x 19 | 5 | 2 x 1 x 2 | fine |
| 46 x 15 x 19 (long box) | 4 (default) | 6 x 2 x 3 | fine |
| 23 x 14 x 19 | 4 (default) | 3 x 2 x 3 | fine |
| 23 x 16 x 19 | 5 (default) | 2 x 1 x 2 | **non-finite** |
| 23 x 16 x 19 | 4 | 3 x 2 x 3 | fine |
| 23 x 17 x 19 | 4 | 3 x 3 x 3 | **illegal memory access** |
| 23 x 30 x 19 (wide box) | 5 (default) | 2 x 2 x 2 | **non-finite** |
| 23 x 30 x 19 | 4 | 3 x 4 x 3 | **non-finite** |
| 23 x 30 x 19 | 3 | 6 x 8 x 5 | fine |
| 24 x 30 x 20 | 4 | 3 x 4 x 3 | **non-finite** |
| 24 x 16 x 19, 23 x 16 x 20, 24 x 16 x 20, 23 x 24 x 19 | 5 (default) | | **non-finite** |
| 16^3, 32^3, 32 x 16 x 16, 32 x 16 x 2 | default | | fine |

No single rule fits (2 x 1 x 2 is fine from 15 y-tiles and not from 16; 36
coarse tiles are fine as 6 x 2 x 3 and not as 3 x 4 x 3), so this is recorded
as an open AMGPCG submodule issue rather than diagnosed here. What the harness
does about it: `OFM::max_level_num_` caps the depth (`--max-levels`), the wide
box runs with three levels, and because a shallower hierarchy is a weaker
preconditioner for the same fixed 15 CG iterations, its narrow control is run
with three levels too so the two are compared at equal solver settings. The
plume driver also now refuses to continue when the initial projection leaves
the field non-finite, or when a diagnostic finds a non-finite value or a speed
above 100 m/s -- the same lesson as the device probe: fail loudly, never print
zeros.

### The plume with the convective outflow, against a reference with the outflow far away

The six Fig. 6 cases at the paper's grid, `theta` diffusing at `Pr` = 0.7,
sampled every 30 s and averaged over the last 200 s, under five boundary
treatments; and the same six in a box twice as long (46 x 15 x 19 tiles,
3600 m), where the prescribed downstream face is 1850 m from the measurement
plane, as the reference for what that plane looks like with no downstream
boundary near it. Bifurcation split at `x` = 1750 m, in metres, with the
standard error of the seven samples, and the difference from the reference in
combined standard errors:

| case | long reference | closed (first two cuts) | downstream convective | downstream + lateral convective |
|---|---|---|---|---|
| `z0` 50, `Q0` 1000 | 420.1 +- 5.1 | 392.3 (-27.8, **3.8x**) | 403.5 (-16.6, 1.9x) | 391.6 (-28.5, 3.9x) |
| `z0` 50, `Q0` 500 | 323.3 +- 10.3 | 312.2 (-11.1, 0.9x) | 312.4 (-10.9, 0.8x) | 304.5 (-18.8, 1.4x) |
| `z0` 100, `Q0` 1000 | 479.9 +- 4.0 | 467.8 (-12.1, 1.9x) | 478.7 (-1.2, 0.2x) | 455.0 (-24.9, **4.6x**) |
| `z0` 100, `Q0` 500 | 364.9 +- 9.1 | 351.1 (-13.8, 1.2x) | 356.6 (-8.3, 0.6x) | 342.3 (-22.6, 1.8x) |
| `z0` 150, `Q0` 1000 | 596.3 +- 32.1 | 603.1 (+6.8, 0.1x) | 608.6 (+12.3, 0.3x) | 567.1 (-29.2, 0.6x) |
| `z0` 150, `Q0` 500 | 388.5 +- 32.9 | 374.2 (-14.3, 0.3x) | 380.8 (-7.7, 0.2x) | 371.9 (-16.6, 0.4x) |

**The prescribed face was contaminating the plane**, by up to 28 m (7%) and
3.8 standard errors in the strongest, shallowest case, and always in the same
direction (narrower). **With the downstream face convective the plane agrees
with the far-outflow reference in every case**, within 2 standard errors, and
the systematic sign is gone. That is the check the boundary was built for,
on the quantity the orderings are read from, and it passes.

Opening the lateral faces as well -- Cunningham's actual set -- narrows the
split by a further 10-25 m in every case, 4-5 standard errors in two of them.
The long reference cannot judge that, since it keeps the lateral walls; the
wide box is the reference for it, and is reported below.

**What this does to the two orderings.** Deeper shear still gives a wider
bifurcation, in every treatment: with the downstream face convective the
strong-source column runs 403.5 -> 478.7 -> 608.6 m, 6.6 combined standard
errors end to end. **The weaker source is still narrower in every treatment**,
4.8 to 12.1 standard errors with the downstream face convective, 4.9 to 12.2
with the lateral faces open too, and 4.5 to 11.6 in the long reference where
no boundary is near the plane. The second cut named the prescribed outflow as
one of two candidate explanations for that failure; **it is not the
explanation**. The remaining named candidate is the Boussinesq reduction.

**Thermal diffusion** at `Pr` = 0.7 (`kappa` = 4.86 m^2/s at `mu` = 4)
lowers the peak anomaly from 27-28.5 K to 24-25 K and narrows the split by
40-65 m (10-14%), so it belongs in any comparison with the paper's direct
runs, which have it. It changes no ordering. Nor does the **sponge** change
the split (within 1-2 standard errors of the same runs without it), but it
does cut the 0.25 K *width* by up to 30% in the `z0` = 150 m strong-source
case, whose plume top sits in the damped layer -- one more reason the split,
not the width, is the number to read. **`n` = 5** with the convective faces
runs to the end in all six cases (the pressure outlet lost every one at
`n` = 5 before 150 s), reproduces the `z0` ordering, and scatters far more
between samples (standard errors of 30-75 m against 4-35 m at `n` = 1), as the
second cut found for the closed box; it is not used for the orderings here.

**Regression.** The closed box with diffusion off, re-run on this code,
matches the second cut's six splits to within 0.5 standard errors and its
six widths to within 0.3 (largest split difference 12.6 m on the
`z0` = 150 m strong-source case, whose standard error is 19 m).

### The lateral walls, judged by the wide box

The wide box (23 x 30 x 19 tiles, 2400 m across, the heat source recentred at
`y` = 1200 m, three multigrid levels because of the trap above) against the
paper's box with the same three levels, both closed:

| case | narrow, closed | wide, closed | difference |
|---|---|---|---|
| `z0` 50, `Q0` 1000 | 387.0 | 382.0 | -5.0 (0.4x) |
| `z0` 50, `Q0` 500 | 306.3 | 314.5 | +8.2 (0.8x) |
| `z0` 100, `Q0` 1000 | 478.7 | 457.2 | -21.5 (1.9x) |
| `z0` 100, `Q0` 500 | 348.6 | 345.8 | -2.8 (0.2x) |
| `z0` 150, `Q0` 1000 | 591.2 | 567.4 | -23.8 (0.5x) |
| `z0` 150, `Q0` 500 | 371.7 | 366.1 | -5.7 (0.1x) |

and the same pair with the downstream face convective: differences of -19 to
+34 m, none above 1.3 standard errors. **Moving the lateral walls 600 m
further out does not change the split**, including in the `z0` = 150 m
strong-source case whose 0.25 K outline the second cut found within 75 m of
the wall; that row can be quoted. (The three-level and four-level solvers
agree on the split to within 1.3 standard errors and on the width to within
1.4, with the width differing by up to 80 m -- the fixed 15 CG iterations are
not fully converged, and the width is the more sensitive of the two.)

Which leaves the lateral *convective* faces as the odd one out: they narrow
the split by 4 to 32 m relative to the narrow closed box, 5.0 standard errors
in the `z0` = 100 m strong-source case, when removing the wall altogether does
nothing. The likely cause is in the flux correction, which shifts every
convective face by the same constant, so the lateral faces carry a share of
whatever the downstream face's advected flux falls short of the inflow by; a
correction confined to the downstream face is the obvious next thing to try.
Until that is done, the configuration for the orderings is **downstream face
convective, lateral faces free-slip walls**, which the wide box shows to be
far enough away.

### Where Stage A stands after the third cut

| the paper's claim | status |
|---|---|
| counter-rotating pair, positive `omega_z` on the right looking downstream | reproduced (first cut) |
| plume cross-section bifurcates | reproduced, with `theta` error-compensated |
| deeper shear layer -> wider bifurcation | **reproduced** with the verified outflow: 6.6x sem at `Q0` = 1 kW/m^3 |
| weaker source -> wider bifurcation | **fails**, opposite by 4.8-12.1x sem, and the outflow is now excluded as the cause |
| laminar-to-turbulent progression over `mu` | not reachable at 10 m (second cut) |
| `St` ~ 0.25 shedding | not tested |
| cross-section not Gaussian | not tested |

What is next, in order: the `Q0` failure now has one named candidate left,
the Boussinesq reduction, and the low-Mach question the plan lists first
under Sec. 7 is where that goes; the lateral-face flux correction above;
then the CVP attribution and the horseshoe vortex, unchanged from the second
cut's list.

## Stage A, fourth cut: the paper's figure read by pixel, `n` = 5 tried as the default, and the floor pinned with a `mu` = 0 control

Four things were on the list after the third cut, none of them a new
boundary condition: the `Q0` criterion was being compared against a sentence
of the paper rather than against its figure; `n` = 5 was argued for as the
default on dissipation grounds and had never been run past 600 s; the
numerical-dissipation floor at 10 m had been located by saturation between
sweep points but never by the `nu` = 0 control D1 uses; and nothing had ever
drawn the paper's Fig. 6 from a run. All six Fig. 6 cases and the viscosity
ladder were rerun to 1600 s, sampled every 30 s and averaged over the last
1000 s (`END=1600 WINDOW=1000 analyse_stage_a.py`), with the third cut's
configuration otherwise: 10 m grid, downstream convective face, lateral
free-slip walls, `Pr` = 0.7, `theta` BFECC + clamp. Runs live under
`stage-a-4-*`; `stage_a_n5.sbatch` is the sweep.

### What Fig. 6 actually shows, measured

The paper's `Q0` statement is one sentence in Results: "for a given cross
flow, bifurcation is wider for the weak heat source than the strong heat
source (compare Fig. 6e with Fig. 6f)", followed by "based on this limited
set of results we conjecture". It cites one pair of panels. `read_fig6.py`
renders the scanned page and measures every panel: the extent of the first
plotted contour along `y`, and the distance between the two lobes' deepest-
nested closed contours, which is the figure's counterpart of `theta_split`.
Scan and contour crowding put a few tens of metres on each number.

| panel | `z0` | `Q0` | outer width | peak split |
|---|---|---|---|---|
| a | 50 | 1.0 | 625 | 448 |
| b | 50 | 0.5 | 511 | 313 |
| c | 100 | 1.0 | 748 | 431 |
| d | 100 | 0.5 | 684 | 465 |
| e | 150 | 1.0 | 753 | 439 |
| f | 150 | 0.5 | 765 | 532 |

The sentence holds for the pair the paper cites, (e) against (f): 532 against
439 on the split, and the outer widths are equal. It does not hold at
`z0` = 50 m, where the weak source is narrower on both measures (313 against
448), and at `z0` = 100 m the split differs by 34 m with the outer width the
other way. The paper's Fig. 6 supports "weaker is wider" at one shear depth
of three. The visual impression that (f) is narrower than (e) is the lobes
themselves: they are smaller (`z` extent 350-585 m against 469-816 m) and
have five contours instead of eleven, while their centres sit further apart.

### The orderings at 1600 s, `n` = 1

`theta_split` in metres, mean over the last 1000 s of a 1600 s run,
+- the standard error of that mean:

| `z0` | `Q0` = 1 kW/m^3 | `Q0` = 0.5 kW/m^3 |
|---|---|---|
| 50 m | 368.8 +- 0.4 | 295.3 +- 0.4 |
| 100 m | 439.7 +- 1.1 | 314.5 +- 0.4 |
| 150 m | 526.5 +- 3.7 | 348.5 +- 1.6 |

The longer window brings the standard errors down to metres: the plume is
steady to within a few metres over the last 1000 s at `n` = 1. Deeper shear
layer -> wider: monotone in both columns, 42.7x the pooled standard error
end to end in the strong column and 32.0x in the weak. Weaker source ->
wider: opposite at every `z0`, by 73, 125 and 178 m, 44 to 127 standard
errors. The `theta_width` column agrees on every row.

Against the figure's own numbers the disagreement narrows to one place. At
`z0` = 50 m the run gives 369 against 295 and the figure gives 448 against
313: the same direction, the same 20-25% narrowing. At `z0` = 100 m the
figure is flat and the run is not. At `z0` = 150 m the figure widens (439 ->
532) and the run narrows (527 -> 349), and the row that moved is the strong
source: 527 in the run against 439 in the figure, while the weak source's 349
sits closer to the figure's 313-465 range than the strong one does. So what
the Boussinesq candidate has to explain is a strong-source, deep-shear plume
that bifurcates too widely, not a weak-source plume that bifurcates too
narrowly. That is a narrower target than the third cut's "opposite at every
`z0`".

### The numerical floor at 10 m, with the `nu` = 0 control

The third cut's four-point sweep (`mu` = 4, 1, 0.15, 0.0015) saturated
between consecutive points, and the floor was read as "between 0.15 and 1".
Refitting those four points showed the danger of four points: the increments
were also consistent with a response linear in `mu` all the way down, which
would have meant a floor above 4. The ladder was extended to `mu` = 40, 12
and 0, run to 1600 s:

| `mu` kg/(m s) | peak abs `omega_z` | `theta_width` | `theta_split` | peak dT | bifurcated |
|---|---|---|---|---|---|
| 40 | 0.0126 | 597 | -- | 13.0 | 0.00 |
| 12 | 0.0573 | 639 | 400 | 20.1 | 1.00 |
| 4 | 0.0947 | 676 | 440 | 24.6 | 1.00 |
| 1 | 0.1195 | 700 | 520 | 27.1 | 1.00 |
| 0.15 | 0.1280 | 730 | 442 | 27.8 | 0.94 |
| 0.0015 | 0.1286 | 739 | 406 | 27.9 | 0.91 |
| 0 | 0.1292 | 737 | 455 | 27.9 | 1.00 |

Ratios of consecutive peak vorticities: 4.5, 1.65, 1.26, 1.071, 1.005,
1.005. The response is large above 4, real between 4 and 1, marginal from 1
to 0.15, and gone below 0.15: the three lowest rows agree to 0.5%, which is
the sampling noise of a 1000 s mean. The `mu` = 0 control reproduces
`mu` = 0.0015. **The floor at 10 m, `n` = 1, lies between 0.15 and 1
kg/(m s)**, as the third cut read it; the linear alternative is excluded by
the `mu` = 12 and 40 rows, which bend. `mu` = 40 is a different flow: the
plume never bifurcates and its peak anomaly halves, so the top row is a
regime boundary, not a point on the same curve.

For scale, air is 1.8e-5 kg/(m s). The paper's smallest `mu`, 0.0015, is 80x
that; the solver's floor at this grid is four orders above it. The paper's
Fig. 6 cases at `mu` = 4 sit above the floor by a factor of 4-25, which is
why they are reproducible here at all.

### `n` = 5 collapses the weak-source plume, and is not the default

The fourth cut ran every case at `n` = 5 as well, on the argument that it
dissipates a third less vorticity (second cut) and costs less per unit of
simulated time. Over 600 s it had looked fine. Over 1600 s it does not: the
weak-source, deep-shear plume loses its buoyancy.

| `z0` = 150 m, `Q0` = 0.5 kW/m^3 | t = 300 s | 600 | 900 | 1200 | 1500 |
|---|---|---|---|---|---|
| `n` = 1: plume top (m) / `w_max` (m/s) / peak dT (K) | 915 / 7.6 / 14.9 | 605 / 5.8 / 14.8 | 635 / 5.8 / 14.8 | 635 / 5.8 / 14.8 | 635 / 5.8 / 14.8 |
| `n` = 5: same | 865 / 7.9 / 13.6 | 575 / 4.6 / 11.9 | 375 / 3.0 / 9.9 | 145 / 2.1 / 9.2 | 95 / 1.6 / 8.8 |

A 9 K anomaly with a 1.6 m/s updraught and a top below 100 m is not a plume
that has become steadier; it is a plume whose buoyancy is not reaching the
velocity. The strong-source cases hold (peak dT 25 K, top 815 m at 1500 s)
and the `z0` = 50 m weak case only drifts (top 775 -> 565 m), so the effect
scales with how bent-over and how weakly forced the plume is. The viscosity
ladder at `n` = 5 is worse: `mu` = 1 goes non-finite at 570 s (`u_max`
419 m/s), `mu` = 0.0015 loses its pair, and the rest send the plume top to
the lid. The second and third cuts' `n` = 5 runs stopped at 600 s, inside the
window where the collapse is only beginning, which is why "z0 ordering
reproduced at `n` = 5" could be written then.

`stage_a_n5_locate.sbatch` runs the collapsing case with one thing changed
at a time, to 1500 s:

| variant | peak dT at 1500 s | plume top | `w_max` |
|---|---|---|---|
| `n` = 2 | 14.4 | 605 | 5.4 |
| `n` = 3 | 12.0 | 445 | 4.1 |
| `n` = 5, closed box | 9.0 | 95 | 1.9 |
| `n` = 5, `mu` = 0 | 11.1 | 465 | 6.8 |
| `n` = 5, drag off | 8.6 | 175 | 2.9 |
| `n` = 5, drag off and `mu` = 0 | 10.1 | 435 | 5.6 |

The collapse survives closing the box, removing the viscous source and
removing the drag, so it is not the open face, the viscous term or the
velocity-dependent force. It grows with the cycle length: `n` = 2 is as
steady as `n` = 1, `n` = 3 decays, `n` = 5 collapses. What is left is the
buoyancy itself, the one source that remains, and the way the cycle carries
it. The (`n`, `dt`) sweep measured a per-sub-step source deficit of 0.22% on
the Burgers vortex, where the source is smooth; that is 1.1% at `n` = 5 and
cannot do this. The difference here is the source's shape: the heating decays
as `exp(-z/h)` with `h` = 25 m, two and a half cells, so the buoyancy the
path integral contracts along the forward map is a field that changes by a
factor `e` across 2.5 cells, and its interpolation at the marched positions
is a different problem from a smooth source. That is a hypothesis; the test
is a D1-style case with a body force confined to a few cells, or the plume
with `h` widened at `n` = 5. Until it is run, `n` = 1 is the configuration
the criteria are read on, and `n` = 5 is not a default.

What the `n` = 5 runs do still say is what the second cut said: the peak
vorticity is 70% higher at `n` = 5 in the strong-source cases (0.165 against
0.095 at `z0` = 100 m). The floor is real and it is large; the lever to lower
it is not yet usable on this case.

### Reproducing

```
sbatch --array=0-25 stage_a_n5.sbatch        # Fig. 6 six + viscosity seven, at n = 1 and n = 5
sbatch --array=0-5  stage_a_n5_locate.sbatch # the collapsing case, one thing changed at a time
END=1600 WINDOW=1000 python3 analyse_stage_a.py n1=stage-a-4-fig6-n1 n5=stage-a-4-fig6-n5
python3 read_fig6.py cunningham2005.pdf      # the PDF from Zotero, not committed
```

### Fig. 6, drawn from the run

`--slice PATH` writes the `x` = 1750 m theta section at every diagnostic, and
`plot_fig6.py` contours the time-mean section every 0.25 K from 0.25 K on the
paper's axes and panel order; with `--pdf` it sets each of the paper's
scanned panels beside ours. The figures are filed with the runs
(`stage-a-4-slices-n1/fig6_ours.png`, `fig6_ours_vs_paper.png`,
`fig6_ours_t600.png`). What they show, panel by panel:

- The morphology is the paper's: two lobes, each with a tail hooking up and
  inward from its top, the hook on the same side in every panel; the lobes
  sit at the same `y`; the strong-source lobes carry ten to twelve contours
  and the weak-source lobes five to seven, as in the scan.
- Ours are smoother. The paper's strong-source lobes are tightly wound
  spirals, the trace of a rolled-up mixing interface; ours are nested ovals.
  That is the 10 m grid's dissipation floor drawn as a picture.
- The strong-source lobes sit about 100 m lower than the paper's (centres
  near `z` = 450-500 m against 550-650 m); the weak-source lobes sit at the
  paper's height or slightly above.
- In (a), (b) and (d) our first contour arches over between the lobes, a
  bridge the scan shows only in (b).
- The 600 s snapshot is indistinguishable from the 600-1600 s mean at
  `n` = 1: the flow is steady, and the third cut's 200 s window was not too
  short for that reason but because the plume was still settling before
  600 s (next paragraph).

### The 600 s window was inside the transient

The third cut read the orderings over 400-600 s. The fourth cut's
600-1600 s means are lower on every row: 369 / 440 / 527 m for the strong
column against 403.5 / 478.7 / 608.6, and 295 / 315 / 349 against 312.4 /
356.6 / 380.8 for the weak. The plume is steady to a few metres after about
700 s, so the third cut's numbers carried the tail of the settling. Both
orderings, and the reference-box comparison that excluded the outflow, were
read as differences between cases sampled over the same window, and none of
them changes; the absolute splits do, by 15 to 80 m, and the 1600 s values
are the ones to quote.

### How far from a Gaussian

The paper says the sections, even time-averaged, are not self-similar
Gaussians, and gives no number. `analyse_slices.py` fits one 2-D Gaussian
(amplitude, centre, two widths) to the 600-1600 s mean section and reports
the residual RMS as a fraction of the peak; a sum of two Gaussians is
fitted alongside.

| case | peak K | 1 Gaussian | 2 Gaussians | ratio |
|---|---|---|---|---|
| `z0` 50, `Q0` 1 | 3.03 | 0.079 | 0.021 | 3.8 |
| `z0` 50, `Q0` 0.5 | 1.71 | 0.076 | 0.024 | 3.2 |
| `z0` 100, `Q0` 1 | 2.88 | 0.082 | 0.020 | 4.1 |
| `z0` 100, `Q0` 0.5 | 1.74 | 0.079 | 0.021 | 3.8 |
| `z0` 150, `Q0` 1 | 2.73 | 0.086 | 0.022 | 3.9 |
| `z0` 150, `Q0` 0.5 | 1.70 | 0.083 | 0.021 | 3.9 |

A single Gaussian misses by 8% of the peak on every case and a pair by 2%,
so the section is a pair of lobes and not one bump, four times over. No
pass threshold is applied: the paper supplies none, and setting one is the
author's decision. The scorecard row becomes "quantified, criterion pending".

### Where Stage A stands after the fourth cut

| the paper's claim | status |
|---|---|
| counter-rotating pair, positive `omega_z` on the right looking downstream | reproduced (first cut) |
| plume cross-section bifurcates | reproduced, with `theta` error-compensated; Fig. 6 redrawn from the run, same morphology including the hooked tails |
| deeper shear layer -> wider bifurcation | **reproduced**: 42.7x sem at `Q0` = 1 kW/m^3 over the last 1000 s of 1600 s |
| weaker source -> wider bifurcation | **the paper's figure supports it at `z0` = 150 m only**; our runs are opposite at every `z0` (44-127x sem), agree with the figure at `z0` = 50 m, and differ at `z0` = 150 m on the strong source (527 vs 439 m) |
| laminar-to-turbulent progression over `mu` | not reachable at 10 m: floor between `mu` = 0.15 and 1, now with the `mu` = 0 control |
| `St` ~ 0.25 shedding | not tested (needs a configuration that sheds) |
| cross-section not Gaussian | quantified: single-Gaussian residual 8% of peak, pair 2%; threshold pending |

`n` = 5 is not a default: it collapses the weak-source plume (above). What
is next, in order: localise the `n` > 1 buoyancy deficit (a body force
confined to a few cells, in a case with a known answer); the lateral-face
flux correction; the Boussinesq question, now aimed at the strong-source,
deep-shear case; then the CVP attribution and the horseshoe vortex.

### The collapse localised: it is the reinitialisation cycle's duration, not `n`, and not the force channel

Three more rounds of one-change-at-a-time runs on the collapsing case
(`z0` = 150 m, `Q0` = 0.5 kW/m^3), all to 1500 s, with two switches added to
the driver for the purpose: `--h` (the heating's vertical decay scale) and
`--no-velocity-clamp`, then `--direct-force`, which bypasses the impulse-form
path integral and adds `dt*f` to the cycle-start velocity the way a
velocity-form solver would (`n` = 1 only; the viscous channel goes with it).
`stage_a_n5_locate{2,3,4}.sbatch`.

| variant | cycle `n*dt` | peak dT at 1500 s | top | `w_max` | verdict |
|---|---|---|---|---|---|
| `n` = 1, `dt` = 0.25 (baseline) | 0.25 s | 14.8 | 635 | 5.8 | holds |
| `n` = 2, `dt` = 0.25 | 0.5 s | 14.4 | 605 | 5.4 | holds |
| `n` = 5, `dt` = 0.125 | 0.625 s | 12.6 | 625 | 5.3 | holds, slow drift |
| `n` = 10, `dt` = 0.0625 | 0.625 s | 11.7 | 735 | 5.7 | holds |
| `n` = 3, `dt` = 0.25 | 0.75 s | 12.0 | 445 | 4.1 | decays |
| `n` = 5, `dt` = 0.25 | 1.25 s | 8.8 | 95 | 1.6 | collapses |
| `n` = 2, `dt` = 0.625 | 1.25 s | 9.7 | 75 | 1.4 | collapses |
| **`n` = 1, `dt` = 1.25** | 1.25 s | 10.1 | 85 | 1.3 | **collapses** |
| `n` = 1, `dt` = 1.25, `mu` = 0 | 1.25 s | 12.4 | 185 | 3.4 | collapses |
| **`n` = 1, `dt` = 1.25, direct force** | 1.25 s | 10.5 | 285 | 4.0 | **collapses** |
| `n` = 1, `dt` = 0.25, direct force | 0.25 s | 14.9 | 655 | 6.2 | holds |
| `n` = 5, `dt` = 0.25, velocity clamp off | 1.25 s | 8.9 | 95 | 2.0 | collapses |
| `n` = 5, `dt` = 0.25, `h` = 50 m, `Q0` = 250 (same heat) | 1.25 s | 7.4 | 165 | 2.4 | collapses |
| `n` = 5, `dt` = 0.25, `h` = 100 m, `Q0` = 125 (same heat) | 1.25 s | 4.8 | 265 | 2.7 | decays |
| `n` = 1, `dt` = 0.25, `h` = 100 m, `Q0` = 125 | 0.25 s | 6.8 | 685 | 4.5 | holds |
| `n` = 5, `dt` = 0.25, `Q0` = 750 | 1.25 s | 18.5 | 665 | 6.5 | mild decay |

Read down the cycle column: everything at 0.625 s or less holds, 0.75 s
decays, everything at 1.25 s collapses, whatever `n` is. `n` = 1 at
`dt` = 1.25 collapses with no sub-steps, no leapfrog branch and no
multi-step path integral in play at all. The direct-force run collapses
too, so the loss is not in how the body force enters the impulse; the
force channel at `dt` = 0.25 agrees with the impulse route to within what
dropping the viscous term explains (peak `|omega_z|` 0.090 against 0.065,
the same 30-40% the `mu` = 0 rows of the ladder show). A thicker heating
layer at the same total heat does not help, which retires the thin-source
hypothesis the previous section proposed; the earlier "h = 100 holds" was
the 4x extra heat, since `Q = Q0 exp(-z/h)` integrates to `Q0*h`. The clamp
is not it. `Q0` = 750 only drifts, so at a 1.25 s cycle the threshold sits
between 500 and 750 W/m^3.

What is left is the flow-map transport and reconstruction over the cycle
itself: the pullback through `T` and `psi`, and the BFECC pass, over an
interval the weak plume cannot survive. That is the "numerical instability
inherent in long-time flow map evolution" OFM gives as its reason for
reinitialising every step (OFM Sec. 3.2); on this configuration a 1.25 s map
is already long. Why a buoyant plume in shear degrades the map that fast,
and why the weak one goes first, is not localised further here: it needs a
case with a known answer, and none of the analytic ones so far carries a
body force in a wall-bounded shear flow.

**The practical consequence.** The lever that lowers the floor is the
cycle's duration `n*dt`, and it is capped here at about 0.6 s. That still
buys a factor of 2.5 over `n` = 1, `dt` = 0.25: `n` = 5 at `dt` = 0.125
(or `n` = 10 at `dt` = 0.0625), for about 1.5-2x the wall clock per unit of
simulated time. Whether the orderings read the same there is the next
table. The mild drift at 0.625 s (peak dT 12.6 against 14.8, though with a
20% higher peak vorticity) is recorded and not explained.

### The six cases at `n` = 5, `dt` = 0.125: the orderings survive, the thermal field does not quite

`stage_a_n5_dt0125.sbatch`, 1600 s, last 1000 s averaged, otherwise the
criterion configuration. `theta_split` in metres; `n` = 1 in brackets.

| `z0` | `Q0` = 1 kW/m^3 | `Q0` = 0.5 kW/m^3 | peak dT strong / weak (K) | peak abs `omega_z` strong / weak |
|---|---|---|---|---|
| 50 m | 417.3 +- 1.4 (368.8) | 286.5 +- 0.6 (295.3) | 22.1 / 12.0 (23.5 / 13.1) | 0.137 / 0.081 (0.094 / 0.062) |
| 100 m | 509.4 +- 3.4 (439.7) | 290.4 +- 0.7 (314.5) | 23.0 / 12.4 (24.6 / 14.2) | 0.139 / 0.080 (0.095 / 0.063) |
| 150 m | 583.1 +- 9.2 (526.5) | 303.2 +- 1.7 (348.5) | 23.3 / 12.8 (24.8 / 14.8) | 0.140 / 0.080 (0.096 / 0.065) |

Both orderings read the same: deeper shear -> wider, monotone in both
columns (17.9x and 9.3x sem end to end); weaker -> wider, opposite at every
`z0` (30-87x sem). All six bifurcate (rate 0.91-1.00). Peak vorticity is
45% higher on every row, which is the floor coming down by the expected
2.5x in cycle duration.

But the peak anomaly is 1.5-2 K lower on every row, strong and weak alike
(6-13%), the strong splits are 40-70 m wider and the weak splits 5-45 m
narrower than at `n` = 1. Set beside the localisation table, the weak
case's peak anomaly falls monotonically with the cycle's duration -- 14.8 K
at 0.25 s, 12.6 at 0.625 s, 8.8 at 1.25 s -- so the collapse is not a
threshold phenomenon but a continuous loss that the 0.625 s cycle already
pays a small dose of. Which of the two settings is closer to the truth
cannot be read from the plume alone; the paper's figure is a scan of a
different solver. The criteria therefore stay at `n` = 1, `dt` = 0.25, and
`n` = 5, `dt` = 0.125 is recorded as what the floor lever costs: 45% more
vorticity, 6-13% less thermal anomaly, orderings intact. Understanding the
long-map loss on a case with a known answer is the first item in the
queue; until then the floor is what `n` = 1 gives.

### Adaptive reinitialization: the cycle ends when the accumulated strain bound reaches eps

The lever that lowers the floor is the cycle duration, and what breaks the
weak plume is also the cycle duration, so the cycle length should be set by
what the flow is doing rather than fixed. `OFM` now keeps `cycle_len_`, the
steps taken since the last reinitialization; `AdvanceAsync`'s leapfrog
schedule and `ReinitAsync`'s marches read it instead of
`step_ % reinit_every_`, so a caller may reinitialize early, and
`reinit_every_` becomes the longest cycle the velocity history holds.
Calling `ReinitAsync` every `reinit_every_` steps reproduces the fixed
cycle exactly (regression below). The maps are only marched at
reinitialization, so there is no `F` to read mid-cycle; the driver
integrates the bound instead: every step it takes the largest
velocity-gradient component in the domain (`MaxVelocityGradient`, one
block-reduced kernel and one 4-byte copy) and accumulates
`sum(max|grad u|) dt`, which bounds `||F - I||` through
`exp(int S dt) - 1`. `--adaptive-reinit EPS` ends the cycle when the sum
reaches `EPS`, when the buffer is full, or at a diagnostic step;
`--log-strain` reports the sum per cycle under fixed cycles.

Calibration on the collapsing case, the largest per-cycle sum over the last
1000 s:

| fixed cycle | `n*dt` | max `sum(S dt)` per cycle | max `S` (1/s) | plume |
|---|---|---|---|---|
| `n` = 1, `dt` = 0.25 | 0.25 s | 0.043 | 0.17 | holds |
| `n` = 5, `dt` = 0.125 | 0.625 s | 0.095 | 0.16 | holds, drifts |
| `n` = 5, `dt` = 0.25 | 1.25 s | 0.25-0.28 | 0.32 | collapses |

The controller at `dt` = 0.25 with a ten-step buffer, three guesses of
`eps`, 1500 s:

| `eps` | cycle it settles to | peak dT at 1500 s | top | `w_max` | peak abs `omega_z` | `theta_split` |
|---|---|---|---|---|---|---|
| (`n` = 1 fixed) | 1 step, 0.25 s | 14.8 | 635 | 5.85 | 0.065 | 346 |
| 0.05 | 2 steps, 0.5 s | 14.3 | 615 | 5.40 | 0.077 | 329 |
| 0.10 | 3 steps, 0.75 s | 12.3 | 465 | 4.31 | 0.078 | 212 |
| 0.20 | 5 steps, 1.2 s | 8.9 | 85 | 1.45 | 0.081 | -- |

`eps` = 0.05 holds: 3% less peak anomaly, 17% more peak vorticity, the
split 5% narrower. It settles to two-step cycles here because the near-
source strain is what the global maximum sees, and it ran 5-6 step cycles
during the first 300 s while the strain was small, which is the point of
the controller: the cycle follows the flow. `eps` = 0.10 is the same slow
loss the 0.625 s fixed cycle showed and 0.20 collapses, so the threshold on
the bound sits near 0.05, and the loss is already under way at 0.1.

### The inference tested where the answer is known: the shear vortex

The mechanism proposed for the collapse was that the covector
reconstruction `T^T u0(psi)` amplifies interpolation error by the strain
the map has accumulated, which would make the loss a function of `S n dt`.
`--test shear` isolates that: a Gaussian vortex column (core 0.05, 6.4
cells, `Gamma` 0.25) in a planar shear `u_x = S (y - 1/2)`, inviscid,
128 x 128 x 16, x faces prescribing the shear profile. The flow is
two-dimensional, `omega_z` is conserved along particles, and the
circulation of the fixed `r` = 0.3 circle about the core is exactly
`Gamma - S pi r^2` as long as the core stays inside it. Error in units of
the vortex's circulation, at `t` = 4 s, `dt` = 1/384:

| `S` | `n` = 1 | 4 | 16 | 64 | 128 |
|---|---|---|---|---|---|
| `S n dt` at `S` = 1 | 0.0026 | 0.010 | 0.042 | 0.167 | 0.333 |
| 0 | -0.00005 | -0.00003 | -0.00003 | -0.00002 | -0.00003 |
| 1 | +0.0062 | -0.0015 | -0.0017 | -0.0049 | -0.0084 |
| 4 | -0.929 | -0.920 | -0.908 | -0.916 | -0.922 |

`S` = 4 is not a valid row: the loss is the same at every `n`, including
`n` = 1, because at `S / omega_peak` = 0.125 the shear strips the vortex
and its vorticity leaves the circle -- physics, not the map, and the test
telling the two apart is what it is for. `S` = 0 loses nothing at any `n`:
plain dissipation does not move the circulation of a compact vortex.
`S` = 1 is the measurement: the loss grows with the strain accumulated per
cycle, 0.17% at 0.04, 0.49% at 0.17, 0.84% at 0.33, roughly linearly once
past `n` = 16 -- and it is small. Per cycle it is about 0.07% at
`S n dt` = 0.33.

So the direction of the inference holds and its magnitude does not. The
plume's collapsing cycles carry a strain bound of 0.25-0.28, at which the
two-dimensional shear test loses a tenth of a percent per cycle; over the
480 cycles of a 600 s collapse that compounds to tens of percent only if
nothing else intervenes, and the plume is a forced, feedback-bearing flow,
so the order of magnitude is not absurd -- but the shear test alone does
not reproduce a collapse, and it has none of the things the plume has: a
bottom wall with the heating layer against it, vertical transport through
the map, a body force. The next place to look is the near-wall layer: the
map's backtrace for cells in the heated bottom rows, where the plume is
fed, and what the clamp to the wall does to it over a longer cycle.

### Regression of the `cycle_len_` refactor

The Burgers `nu` = 0 floor with fixed cycles, `n` = 1 at `dt` = 1/480 and
`n` = 4 at `dt` = 1/1920 (the (n, dt) sweep's pair that must agree): the
refactored solver returns 8.5873e-5 and 8.5588e-5, agreeing with each
other to 0.3% as the `n*dt` law says they should. The pre-refactor binary
run on the identical command gives the identical 8.587309e-5 and 8.558805e-5, bit for bit: the fixed-cycle path is unchanged. (The sweep's
own 7.0246e-5 was measured on the `feat/reinit-dt-sweep` branch; the
difference between branches, not between binaries, is what the control
run separates.)

### The cause of the collapse: the projection is not converged

Two probes after the shear test, both on the weak case. First the wall:
`--z-src` centres the heating at a height with the same total heat
(`Q0/2 exp(-|z - z_src|/h)`), so the plume's root sits away from the
ground. The elevated source at the 1.25 s cycle still decays (top 665 ->
425 m over 900 s, peak anomaly 5.8 -> 5.2 K, against 735 -> 645 and 5.9 ->
5.9 at 0.25 s), so the wall clamp of near-ground backtraces is not it.
Second the profile: `--profile` writes horizontal means of `w` and `theta`
over the source column (x 300-600 m, y 450-750 m) per level. The
horizontally averaged `w` there is negative -- above a bent-over plume the
column sees compensating subsidence -- and at the 0.25 s cycle it holds at
about -0.5 m/s; at the 1.25 s cycle it grows monotonically, -0.5 at 300 s,
-1.2 at 600 s, -1.85 m/s at 900 s, nearly uniform from 100 m to 600 m.
That is a field of mass sinks over the source, not a transport error.

A residual divergence would do exactly that, and the pressure solve is
run for a fixed 15 AMGPCG iterations (`SetupSolver`: `solve_by_tol_ =
false`, `max_iter_ = cg_iter = 15`), not to a tolerance. The impulse's
gauge part grows with the cycle, so the right-hand side the projection has
to remove grows with it, and 15 iterations leave more behind. `--log-div`
reports max |div u| over the cells after the last projection, `--cg-iter`
sets the count. Weak case, 1.25 s cycle, 900 s:

| iterations | max abs div u (1/s), 300 / 600 / 900 s | peak dT at 900 s | top | `w_max` | peak abs `omega_z` | `theta_split` |
|---|---|---|---|---|---|---|
| 15 | 0.09 / 0.29 / 0.38 | 9.9 | 345 | 2.9 | 0.084 | 168 |
| 30 | 0.09 / 0.20 / 0.15 | 14.86 | 615 | 6.0 | 0.107 | 384 |
| 60 | 0.06 / 0.08 / 0.06 | 14.75 | 605 | 6.3 | 0.114 | 392 |
| 120 | 0.04 / 0.03 / 0.03 | 14.75 | 605 | 6.5 | 0.113 | 395 |
| `n` = 1, 15 | 0.04 / 0.06 / 0.06 | 14.76 | 635 | 5.8 | 0.065 | 346 |
| `n` = 1, 60 | 0.015 / 0.013 / 0.014 | 14.66 | 605 | 6.1 | 0.063 | 371 |

At 15 iterations the residual divergence at the 1.25 s cycle reaches
0.38 1/s -- 4 m/s across a 10 m cell -- and the plume collapses. At 30 it
holds; at 60 and 120 the 1.25 s cycle reproduces `n` = 1's peak anomaly to
the second decimal and carries 75% more peak vorticity, which is the floor
gain the longer cycle was supposed to buy, intact. Doubling the count from
60 to 120 changes nothing but the residual, so 60 is converged for this
case at this cycle.

That is the cause. It explains every localisation result: the loss followed
the cycle's duration because the gauge part does; it survived closing the
box, `mu` = 0, drag off, the clamp off and the direct force because none of
those touch the solve; `n` = 1 at `dt` = 1.25 collapsed because a long
step is a long cycle; the elevated source did not help because the sinks
sit wherever the gauge is large, not at the wall; the two-dimensional shear
vortex could not see it because an irrotational residual leaves the
circulation of a loop alone -- which is also why the `S` = 1 rows there
were so small; and the weak plume went first because a fixed sink field
costs it a larger fraction of its updraught.

The fixed 15 is upstream's choice (OFM ships `cg_iter = 15`), and it is
enough for the flows OFM shows. Here even the criterion configuration
carries a 0.06 1/s residual, and converging it moves the weak `z0` = 150 m
split from 346 to 371 m (7%) with the peak anomaly 0.1 K lower: **the
criterion numbers reported above were read with an unconverged projection**
and are re-read below with 60 iterations. D1-D4 and the (n, dt) floor sweep
were all run at 15; whether any of their numbers move is a regression to
run (a columnar vortex has no gauge growth to speak of, so probably not, but
it has to be measured, not assumed).

`--cg-tol REL` (with `--cg-iter` as the cap) switches the solve to a
relative residual, so a long cycle's larger gauge part gets the iterations
it needs and a short one does not pay for them.

### The six cases with the projection converged: the paper's pattern appears

`stage_a_cg60.sbatch`: the six Fig. 6 cases at 60 iterations, 1600 s, last
1000 s averaged, at the criterion cycle (`n` = 1) and at the 1.25 s cycle
(`n` = 5, `dt` = 0.25) that collapsed under 15. `theta_split` in metres.

| `z0` | `n` = 1, 15 it. | `n` = 1, 60 it. | 1.25 s cycle, 60 it. | paper's Fig. 6 (pixel) |
|---|---|---|---|---|
| 50, strong / weak | 369 / 295 | 355 / 278 | 394 / 270 | 448 / 313 |
| 100, strong / weak | 440 / 315 | 430 / 310 | 397 / 329 | 431 / 465 |
| 150, strong / weak | 527 / 349 | 516 / 372 | 409 / 394 | 439 / 532 |
| peak abs `omega_z`, strong / weak | 0.095 / 0.063 | 0.093 / 0.063 | 0.17-0.18 / 0.10-0.11 | -- |
| max abs div u (1/s) | 0.06 | 0.014 | 0.06-0.07 | -- |

Converging the projection at `n` = 1 moves the splits by -14 to +23 m and
nothing else: both orderings read as before (32.1x and 19.9x sem for `z0`;
weak narrower at every `z0`, 21-441x), the peak anomalies agree to 0.2 K.
The numbers quoted for the criteria from here on are the 60-iteration
ones.

The 1.25 s cycle with the projection converged is a different picture.
The strong column goes flat -- 394, 397, 409 m, 7.2x sem end to end -- and
the weak column rises steeply, 270 -> 329 -> 394 m (28.5x). That is the
shape of the paper's own figure: its strong column is flat too (448, 431,
439) and its weak column rises (313, 465, 532). At `z0` = 150 m the weak
source is 15 m narrower than the strong one, 3.1x sem, where the 15-
iteration `n` = 1 runs had it 178 m narrower at 44x. The peak vorticity is
80% higher than at `n` = 1, the residual divergence is the same 0.06 the
`n` = 1, 15-iteration criterion runs carried, and the peak anomalies match
`n` = 1 to 0.5 K.

What changed between the two columns of this solver is only how much
vorticity the scheme dissipates: the physics, the grid and the boundary
conditions are identical. So the third cut's "weaker source is narrower at
every `z0`, opposite to the paper" was in large part a statement about the
one-step scheme's dissipation, and the Boussinesq reduction it named as the
only remaining candidate is not the only one -- it may not be needed at
all. The residual disagreement at `z0` = 150 m (the weak source 15 m
narrower here against 93 m wider in the figure) is what remains to be
explained, and the cycle-length sweep below says the lever is not
exhausted.

### How long can the cycle be, once the projection is converged

`stage_a_cycle_cg60.sbatch`, weak case, `dt` = 0.25, 1500 s:

| cycle | iterations | max abs div u (1/s), late | peak dT | top | `w_max` | peak abs `omega_z` | `theta_split` |
|---|---|---|---|---|---|---|---|
| 0.25 s (`n` = 1) | 60 | 0.014 | 14.7 | 605 | 6.1 | 0.063 | 372 |
| 1.25 s (`n` = 5) | 60 | 0.06 | 14.75 | 605 | 6.3 | 0.114 | 392 |
| 2.5 s (`n` = 10) | 60 | 0.13 | 14.96 | 635 | 6.5 | 0.140 | 337 |
| 2.5 s (`n` = 10) | 120 | 0.055 | 14.95 | 635 | 6.7 | 0.136 | 341 |
| 5 s (`n` = 20) | 60 | 0.27 | 15.05 | 705 | 6.9 | 0.177 | 256 |
| 10 s (`n` = 40) | 60 | 0.56-0.76 | 14.8 | 775 | 7.9 | 0.28 | 215 |

Nothing collapses any more, at any cycle: the thermal field holds to 15 K
throughout. But the residual divergence at a fixed 60 iterations climbs
with the cycle -- 0.13 at 2.5 s, 0.27 at 5 s, 0.7 at 10 s -- because the
gauge part the projection has to remove grows with it, and 120 iterations
at 2.5 s bring it back to 0.055. The vorticity keeps rising with the cycle
(0.28 at 10 s, 4.4x the `n` = 1 value) while the split narrows, and at
0.7 1/s of residual neither number is trustworthy. The usable regime with
a fixed count is a cycle of about 2.5 s at 120 iterations, residual
0.055, peak vorticity 2.2x `n` = 1's. The right control is not a count
at all but a residual tolerance (`--cg-tol`, below), so that the
iterations follow the gauge.

### Stage 0 does not move with the projection count

`regress_cg.sbatch`: D1 (viscous Burgers, 128^3, `n` = 1 and 5) and D2
(attribution, 128^3, `n` = 5) at 15 and at 60 iterations return the same
numbers to every printed digit (D1 `nu` recovered 1.027296e-3 and
9.407688e-4; D2 dual-path 1.66%, attribution 3.79%). A columnar vortex
grows no gauge part to speak of, so 15 iterations were already converged
there. The Stage 0 results stand; the projection count only bites where
the impulse's gauge part is large, which the sheared buoyant plume is the
first case here to have.

### The residual-tolerance mode works, and costs 3-5x a fixed 60

`stage_a_cgtol.sbatch`, weak case, 900 s, `/usr/bin/time` wall clock on
one RTX 6000:

| cycle | mode | max abs div u, late | peak dT at 900 s | wall |
|---|---|---|---|---|
| 1.25 s | fixed 15 | 0.38 | 9.9 (collapsed) | 175 s |
| 1.25 s | fixed 60 | 0.06 | 14.75 | 382 s |
| 1.25 s | fixed 120 | 0.03 | 14.75 | 677 s |
| 1.25 s | tol 1e-2, cap 200 | 0.02 | 14.75 | 1130 s |
| 1.25 s | tol 1e-3, cap 200 | 0.02 | 14.75 | 1136 s |
| 1.25 s | tol 1e-4, cap 400 | 0.012 | 14.78 | 2131 s |
| 0.25 s | fixed 15 | 0.06 | 14.76 | 239 s |
| 0.25 s | tol 1e-3, cap 200 | 0.005 | 14.67 | 1811 s |

The tolerance mode converges further than any fixed count tried, and the
plume is the same one at every setting from fixed 60 on. But it is
expensive: 1e-2 and 1e-3 cost the same and three times a fixed 60, which
says the solver is running to its cap rather than stopping at the
tolerance, and at `n` = 1 it costs eight times the fixed 15. The
tolerance is applied to the squared residual (`tol = max(abs_tol_,
rel_tol_ * initial_rTr)`, `amgpcg.cu`), and the per-iteration check
synchronizes the host; which of the two makes it slow, or whether the
AMG cycle stalls on this pure-Neumann problem after the first decade, is
not investigated here. For now the criterion runs use a fixed count, 60 at
the 1.25 s cycle and 120 at 2.5 s, with `--log-div` reporting what it
leaves; the tolerance mode is the right design and the cost is a solver
question to take up separately.

### The six cases at the 1.25 s cycle with 120 iterations

`CG=120 sbatch --array=6-11 stage_a_cg60.sbatch`. Against 60 iterations
the splits move by -8 to +10 m, the peak anomalies by 0.1 K, and the
residual halves (0.03 1/s):

| `z0` | strong | weak | weak - strong |
|---|---|---|---|
| 50 m | 385.9 +- 0.3 | 276.3 +- 0.1 | -110 (389x sem) |
| 100 m | 395.3 +- 0.7 | 339.7 +- 1.8 | -56 (28x) |
| 150 m | 411.9 +- 2.5 | 403.4 +- 4.3 | -8.5 (1.7x) |

Strong column flat and rising slightly (10.2x end to end), weak column
rising steeply (29.6x), and at `z0` = 150 m the two sources are 1.7
standard errors apart, which is not a difference. The paper's figure has
the weak source 93 m wider there. Everything else about the two columns
is the figure's own shape. This is the configuration the Stage A
criteria are read on from here: the 1.25 s cycle, 120 iterations,
residual 0.03 1/s, peak vorticity 1.8x the one-step scheme's.

### The 2.5 s cycle is not usable for the strong source yet

`stage_a_c25.sbatch` 0-5: the six cases at `n` = 10, `dt` = 0.25, 120
iterations. Residual 0.05-0.06 1/s on every case, the same level as the
weak case's calibration. But `z0` = 50 m, `Q0` = 1 kW/m^3 goes non-finite
at 570 s (`u_max` 860 m/s), and the strong-source splits that survive are
140-170 m narrower than at the 1.25 s cycle (254 and 244 m against 395
and 412) with the peak anomaly unchanged (24.6 K) and the peak vorticity
at 0.22-0.26 -- while the weak sources move by 9-54 m. With the strong
column narrowed like that the weak source reads wider than the strong at
`z0` = 100 and 150 m (+60 and +106 m), which is the paper's ordering, but
a column that narrows by a third when the cycle doubles, in a set where
one member blows up, is not a reading; whether the strong plume at 2.5 s
is oscillating (Cunningham's 200 s period), has an unsteady pair the
1000 s mean smears, or is at the edge of the map's stability is not
settled here. The criterion configuration stays at the 1.25 s cycle. The
2.5 s row is recorded as "holds for the weak source, not for the strong".

### The lateral-face tail is not the flux correction

`stage_a_lateral.sbatch`: six cases at the criterion configuration with
the lateral faces convective, the correction shared over all three open
faces (`cxy`) and the correction on the downstream face only (`cxy-d`,
`OFM::flux_correct_face_`), against lateral walls (`cx`):

| case | `cx` | `cxy` | `cxy-d` |
|---|---|---|---|
| `z0` 50, strong / weak | 385.9 / 276.3 | 385.7 / 275.5 | 380.4 / 274.8 |
| `z0` 100, strong / weak | 395.3 / 339.7 | 374.3 / 336.8 | 375.7 / 336.3 |
| `z0` 150, strong / weak | 411.9 / 403.4 | 407.8 / 394.0 | 407.2 / 395.2 |

`cxy` and `cxy-d` agree to 6 m on every row: where the correction goes
does not matter, and the third cut's suspect is cleared. Against the walls
the lateral faces move five of six splits by 0-9 m and one, `z0` = 100 m
strong, by 21 m (17 sem); at the unconverged one-step configuration the
same comparison gave 10-25 m on every row. The lateral convective face is
usable for Stage B with that 21 m on record.

### The viscosity ladder at the criterion configuration reaches the paper's transition

`stage_a_c25.sbatch` 6-12: the ladder at the 1.25 s cycle, 120 iterations
(`z0` = 100 m, `Q0` = 1 kW/m^3), last 1000 s of 1600 s:

| `mu` kg/(m s) | peak abs `omega_z` | ratio to next | `theta_width` | `theta_split` | bifurcated | plume top | `w_max` |
|---|---|---|---|---|---|---|---|
| 40 | 0.014 | | 581 | -- | 0.00 | 789 | 4.1 |
| 12 | 0.085 | 5.9 | 591 | 348 | 1.00 | 834 | 7.4 |
| 4 | 0.169 | 2.0 | 674 | 395 | 1.00 | 730 | 8.6 |
| 1 | 0.239 | 1.42 | 534 | 152 | 0.77 | 1341 | 13.4 |
| 0.15 | 0.274 | 1.14 | 742 | 122 | 0.51 | 1502 | 16.0 |
| 0 | 0.275 | 1.006 | 656 | 114 | 0.54 | 1460 | 16.5 |

(`mu` = 0.0015 is a failed run -- the pair left the plane and the driver
stopped it; rerun.) Two things the one-step ladder could not show. The
response to `mu` now reaches further down: 4 -> 1 is +42% (was +26%),
1 -> 0.15 is +14% (was +7%), and only 0.15 -> 0 is flat, so the floor at
this configuration sits between 0.15 and 1 as before but lower within
that decade. And below `mu` = 4 the plume changes character: at `mu` = 1
and 0.15 the plume top reaches the lid, `w_max` doubles, the bifurcation
is present in only half the samples and the split narrows to 120-150 m --
an unsteady plume (whether it is also asymmetric is measured in the next
section: it is not). That is the transition Cunningham describe between
their `mu` = 4 and `mu` = 1 direct runs (transverse vortices at the top
of the laminar base, then an asymmetric wake); the one-step scheme never
got there. Whether the unsteadiness is the paper's, with its
200 s period, is what the shedding criterion (`St` ~ 0.25) now has a
configuration to be measured on.

### The unsteady plume at `mu` = 1: unsteady yes, asymmetric no, and not the lid

`stage_a_mu1.sbatch`, criterion configuration, `z0` = 100 m,
`Q0` = 1 kW/m^3, 1600 s. Sample-to-sample scatter over the last 1000 s
(34 samples, 30 s apart) is what "unsteady" means here:

| `mu` | `theta_split` sd | `theta_width` sd | `w_max` sd | bifurcated | plume top |
|---|---|---|---|---|---|
| 12 | 4 m | 6 m | 0.01 m/s | 1.00 | 834 m |
| 4 | 4 m | 12 m | 0.01 | 1.00 | 730 |
| 1 | 69 m | 59 m | 0.09 | 0.77 | 1341 |
| 0.15 | 88 m | 145 m | 0.65 | 0.51 | 1502 |
| 0 | 108 m | 77 m | 0.72 | 0.54 | 1460 |

At `mu` >= 4 every diagnostic is steady to a few metres; at `mu` <= 1 the
split wanders by 70-110 m between samples and the section is two-lobed in
only half to three quarters of them. The `mu` = 1 sections (`--slice`,
`mu1_sections.png`) show a convoluted, mushroom-like plume between 800 and
1250 m at `x` = 1750 m that changes shape every 120 s -- the plume at this
viscosity rises to twice the height it does at `mu` = 4 and arrives at the
plane still churning.

Two things the ladder alone could not say. The asymmetry Cunningham
describe below `mu` = 4 is not there: the ratio of the left and right
lobe peaks of `P(y)` over the last 1000 s is 1.01 +- 0.04 (range 0.91 to
1.08); the plume is unsteady and symmetric. Their asymmetric wake comes
with vertically oriented wake vortices further downstream and at lower
`mu`; whether we reach it is the `mu` = 0.15 section's question. And it
is not the lid: with the Rayleigh sponge in the top 10% (`--sponge`) the
scatter (sd 52 against 55 m), the bifurcation rate (0.89 against 0.83)
and the top (1355 against 1343 m) are the same. The plume top sits just
below the sponge zone (1368 m) either way.

`mu` = 0.0015 at this configuration goes non-finite at 570 s (`u_max`
4e17), the second time this member fails while `mu` = 0 runs to 1600 s;
the lowest-viscosity rows at the 1.25 s cycle are at a stability edge that
is not understood and is on the list.

### The transition seen in the centreline section: rollers at the top of a laminar base

`stage_a_xz.sbatch`: `--slice-xz` writes the centreline `x`-`z` section
(`theta`, `u`, `w`) every 30 s; `plot_xz.py` draws `omega_y` = du/dz -
dw/dx (red is clockwise seen from +`y`, the sense of the shear between
the rising plume and the faster air above it) under `theta` contours,
and `xz_rollers.py` counts the rollers: the local maxima along `x` of the
strongest clockwise `omega_y` inside the 0.5 K contour, above 0.15 1/s
and 40% of the strongest. Criterion configuration, `z0` = 100 m, `Q0` =
1 kW/m^3, 900 s; the figure is `xz_transition.png` in the run directory
(rows `mu` = 4, 1, 0.15; columns 300, 600, 900 s):

| `mu` | rollers (450 / 600 / 750 / 900 s) | onset `x` | plume upper edge at onset | mean spacing | strongest `omega_y` |
|---|---|---|---|---|---|
| 4 | 0 / 0 / 0 / 0 | -- | -- | -- | 0.07-0.10 1/s |
| 1 | 6 / 4 / 5 / 8 | 1035-1065 m | 605-665 m | 100-150 m | 0.29-0.70 |
| 0.15 | 13 / 8 / 6 / 9 | 845-1065 m | 505-725 m | 67-170 m | 0.43-1.29 |

At `mu` = 4 the plume is a smooth sheet at every time after the starting
head has left (the 300 s column still shows the head, a large clockwise /
anticlockwise pair at 1000-1200 m, the transient the criteria's window
excludes); its shear layer carries a tenth of the vorticity of the
others' rollers and never rolls up. At `mu` = 1 the plume is the same
smooth sheet up to `x` ~ 1040 m, where its upper edge is at 600-650 m, and
from there on its upper face carries a chain of discrete clockwise
rollers 100-150 m apart that grow downstream, with the `theta` contours
below them folded into the mushroom lobes the `y`-`z` sections showed;
the onset sits at the same `x` to within 30 m at all four times, i.e. it
is a spatial instability of the sheared upper face, not a temporal one of
the whole plume. At `mu` = 0.15 the onset moves upstream (845 m at 450 s)
and lower, the rollers are closer, stronger and less regular, and the
plume above them is broken into structures on every scale down to the
grid. This is the picture Cunningham et al. describe for the transition
between their `mu` = 4 and `mu` = 1 runs -- transverse (spanwise) vortices
at the top of a laminar base, and a turbulent plume above them at lower
viscosity -- with the laminar base's height and the roller spacing now
measured. The rollers are also what the wake probe in the plume at
(900, 600, 305) m sees at `mu` = 1: a 50 s period in `v` and `theta`
(`St` = 1.0 on the source diameter), which is the roller passage, not
the 200 s wake shedding the `St` criterion looks for; the shedding
reading is in the next section.

### The shedding criterion: no wake line at `St` 0.25, at either viscosity

`stage_a_sec8.sbatch` 0-1: criterion configuration, `z0` = 100 m, `Q0` =
1 kW/m^3, `mu` = 1 and 0.15, 3000 s, with `--probes` writing `u`, `v`,
`w`, `theta` every step at 16 points: the centreline at `x` = 700, 900,
1100, 1300 m and `z` = 55, 155, 305 m, and two off-centre pairs at `x` =
1000 m, `y` = 450 and 750 m, `z` = 55 and 155 m. `analyse_probes.py`
detrends each record over 1000-3000 s, averages Hann-windowed
periodograms over 800 s segments (Welch; resolution 1/800 Hz, `St`
0.06), and reports the spectral peak in 1/600-1/20 Hz. Cunningham et
al.'s shedding is `St` = f D / U ~ 0.25 with `D` = 225 m, `U` = 4.5 m/s:
f = 0.005 Hz, a 200 s period.

| | `mu` = 1 | `mu` = 0.15 |
|---|---|---|
| rms `v` at the 12 wake probes | 0.003-0.016 m/s | 0.016-0.045 m/s |
| rms `v` as a fraction of `U` | 0.1-0.4% | 0.4-1.0% |
| peak of the `v` spectrum | lowest bin (400 s) at 8 probes, 160 s (`St` 0.31) at 4 near-ground probes | lowest bin at all 12 |
| a line at 0.005 Hz | none | none |

The spectra are red: the power sits in the lowest bin and falls
monotonically, with one slow excursion of the whole wake between 1750
and 2300 s (visible at every probe, once, in `probes_mu1.png`), not a
periodic signal. The 160 s peak at the `mu` = 1 near-ground probes is at
5-9 mm/s rms, 0.1-0.2% of `U`. The only lines in the records are at
0.020-0.025 Hz (40-50 s, `St` 1.0-1.1) in `w` and `theta` at the probes
inside the plume ((900, 600, 305) m and (1100, 600, 305) m): that is the
passage of the transverse rollers of the previous section, 100-150 m
apart, carried at 2-3 m/s. So the wake behind this plume does not shed
at the paper's frequency, at `mu` = 1 or at 0.15, on the 10 m grid in the
long box; whether their shedding needs the lower viscosities of their
direct runs (the `mu` = 0.0015 row is the one that goes non-finite here)
or a feature this configuration lacks is not settled. The `St` criterion
is recorded as not met, with the lateral velocity in the wake at or
below 1% of the wind.

The per-step records also show what the cycle-end diagnostics cannot: a
component locked to the reinitialization cycle. At the near-ground probe
(700, 600, 55) m the fast part of `w` (the record minus its 5 s running
mean) has rms 0.09 m/s at `mu` = 1, its spectral peak is at 0.800 Hz --
the 1.25 s cycle -- and its mean by sub-step within the cycle is +0.09,
-0.16, -0.01, -0.02, +0.10 m/s (`mu` = 0.15: +0.10, -0.19, -0.01, -0.02,
+0.11). At the plume's base (900, 600, 305) m the same pattern is at
0.07 m/s in `w` and 0.22 m/s in `u`. The per-step velocities are the
leapfrog's intermediate states (`mid_u_`, `PlumeVelocityAfter`): the
first sub-step is advected over half a step from the cycle start, the
second over a full step, the rest over two steps from two back, and each
carries its own phase of error. It is zero-mean over the cycle, 2-4% of
`U` where the shear is strongest, and it is what the `theta` advection
and any per-step sampling see; the criteria, sampled at cycle ends, do
not.

### The 2.5 s strong-source blow-up is the downstream face, and a projected face value removes it

`stage_a_sec8.sbatch` 3-4 rerun the member that went non-finite (`z0` =
50 m, `Q0` = 1 kW/m^3, 2.5 s cycle) at 120 and 200 iterations with the
residual, the source-column profile and the centreline section every
30 s. Both blow up, the 200-iteration run 30 s earlier (`u_max` 63 m/s
at 510 s against 53 m/s at 540 s), so the projection count is not the
cause. The residual divergence is flat at 0.04-0.09 1/s until 480 s and
rises only once the velocity has: it is a consequence. What the sections
show is the starting plume's head reaching the lid at 420 s (top 1515 m)
and the downstream face at 480-510 s, and the first cells to run away
are in the last column of the domain, `x` = 1775-1835 m, `z` = 825-1295 m
(`c25_blowup.png`); from there a block of vorticity of either sign
spreads back into the plume within 30 s. (The harness also missed the
end: once the field is non-finite every maximum it reports is 0, because
the reductions use `fmaxf`, which drops NaN; the guard now treats `u_max`
= 0 after the first step as non-finite.)

The convective face takes its value, at every projection, from the
velocity the advection delivers to it (`ConvectiveFaceUpdateKernel`:
`bc_val = u_axis` on the face). At the reinitialization's projection that
velocity is the impulse pulled back through the cycle's flow map, which
carries a gauge part -- a gradient the projection removes from the
interior but cannot remove from a face whose normal velocity is
prescribed. Over a 1.25 s cycle that part is small; over 2.5 s, where
the strong plume's head arrives at the face, it is not, and it is
written into the boundary condition. `convective_face_from_projected_`
(`--face-projected`) has that projection take the face value from the
last per-step projected velocity of the cycle (`mid_u_[cycle_len_ - 1]`)
instead, which is divergence-free and already convectively updated; the
per-step projections are unchanged. `stage_a_faceproj.sbatch` 0, the same
member with the flag:

| | face from the impulse (120 it) | face from the projected velocity |
|---|---|---|
| outcome | non-finite at 570 s | 900 s, pair present |
| plume top | 1515 m at 420 s, then runaway | 1515 m at 420 s (the head), 805 m from 600 s |
| max abs div u, 600-900 s | -- | 0.04-0.09 1/s |
| `theta_split` at 900 s | -- | 261 m |
| `theta_width` | -- | 524 m |
| peak abs `omega_z` | -- | 0.016 |
| `w_max` | -- | 9.3 m/s |

The head leaves through the face and the plume settles. But against the
criterion configuration's reading of the same case at 900 s (top 705 m,
`theta_split` 386 m, `theta_width` 598 m, peak `omega_z` 0.0071,
`w_max` 8.0), the 2.5 s cycle still gives a plume 100 m taller, a pair
twice as strong and a column a third narrower -- the same narrowing the
survivors of the 2.5 s set showed. The blow-up and the narrowing are
therefore two things: the face is fixed, the strong source's dependence
on the cycle length between 1.25 and 2.5 s is not understood, and the 2.5
s row stays "runs, not converged in cycle length for the strong source".

The flag at the criterion configuration (`stage_a_faceproj.sbatch` 1-2,
`z0` = 100 m, `Q0` = 1 kW/m^3, 1600 s, last 1000 s, mean +- standard
error over 35 samples):

| | `theta_split` | `theta_width` | plume top | `w_max` | peak `omega_z` | peak `P(y)` |
|---|---|---|---|---|---|---|
| criterion run (build-v16) | 395.3 +- 0.7 | 674.4 +- 2.2 | 730.4 +- 1.2 | 8.635 | 0.00490 | 2.275 |
| control, rebuilt binary, flag off | 395.4 +- 0.7 | 674.0 +- 2.1 | 731.0 +- 1.2 | 8.632 | 0.00490 | 2.275 |
| flag on | 397.2 +- 0.5 | 674.1 +- 2.0 | 722.7 +- 0.8 | 8.603 | 0.00503 | 2.288 |

The rebuild is regression-clean (the control reproduces the criterion
run to the last digit). The flag itself moves the reading: +2 m on the
split, -8 m on the top, +3% on the pair strength, +0.013 K on the peak --
small, systematic, and not nothing, so the criterion configuration keeps
the face as it was, and the flag is used where it is needed, the 2.5 s
cycle. Which face value is the right one for a plume leaving through the
boundary is a boundary-condition question the long box (`--long`) can
arbitrate, not settled here.

### Why the tolerance mode runs to its cap: the iteration does not reduce the residual past a factor of 2-3

`stage_a_sec8.sbatch` 2 prints the AMGPCG residual at every iteration
(`--cg-verbose`; `|residual|_2` is sqrt(rTr) of the solver's own
residual vector), and `stage_a_divstats.sbatch` puts the projected
field's divergence statistics next to it (`--log-div` now reports the
maximum and its cell, the rms, the interior rms two or more cells from
every boundary, the 2-norm, the share of the sum of squares in the
one-cell boundary layer, and the signed mean). First the two agree: on
the last logged projection the solver's final residual is the field's
divergence 2-norm times `dx` (ratios 9.5, 10.0, 9.7, 9.4, 10.0 in five
runs), so what the solver prints is what the field carries. Then what the
iteration does with it, weak case, 1.25 s cycle, the reinitialization's
projection at step 40:

| count | residual at iteration 1 | its minimum (at) | at the last iteration | last / first | field: max abs div, l2 |
|---|---|---|---|---|---|
| 15 | 31.6 | 31.6 (1) | 49.0 | 1.55 | 0.057, 5.16 |
| 120 | 14.6 | 11.8 (34) | 23.5 | 1.61 | 0.024, 2.35 |
| 120, restart every 30 | 11.8 | 8.1 (94) | 14.2 | 1.20 | 0.015, 1.47 |
| 120, restart every 15 | 11.9 | 9.7 (108) | 13.3 | 1.12 | 0.013, 1.41 |
| 120, restart every 1 (steepest descent) | 20.4 | 20.4 (1) | 24.3 | 1.19 | 0.020, 2.43 |
| `n` = 1, 15 (80 projections) | -- | at iteration 1 in 79 of 80 | -- | 1.5-1.75 | 0.027, 2.42 |

In every projection of every run the residual norm is at or near its
minimum after the first iteration or within the first few dozen, and
higher at the end than at the start; over 199 iterations in tolerance
mode it never falls below 0.35 of its first value (49 projections, the
reinitialization's the worst at 0.8-1.0), so a relative tolerance of
1e-2 on rTr (0.1 on the norm) is never met and the mode always runs to
its cap. That is the whole of the cost measured in the tolerance section.
The count still matters between runs because each projection starts from
the field the previous ones left: at 15 iterations the divergence rms
over 600 s grows to 1.2e-2 1/s (max 0.35-0.50), at 120 it holds at
1.0-1.7e-3 (max 0.03-0.07) -- a factor of ten -- and the 600 s plume
differs accordingly (peak anomaly 11.8 K against 14.8 K).

What the residual is not. It is not a net flux imbalance: the signed
mean of the divergence is 1e-11 1/s (`v20/stage-a-4-divstats`), so the
pure-Neumann recentering is not hiding a constant. It is not at the
boundaries: the boundary layer's share of the sum of squares is
0.03-0.05, the same as that layer's share of the cells (0.041), and the
interior rms equals the rms. It is not cured by restarting the recurrence
(restart every 30 iterations: 1.6x lower on one projection, no
difference in the 600 s divergence trend or the plume, `stage-a-4-
cgrestart`), and steepest descent, which needs no conjugacy, is worse.
The residual is a zero-mean field of rms 1e-3 1/s spread through the
whole domain, peaking 30x higher in the plume, that the multigrid-
preconditioned iteration reduces by a factor of 2-3 and then lets grow.
`AMGPCG::restart_every_` (`--cg-restart`) stays in as a knob that does
not help; the criterion configuration keeps its fixed 120 iterations.

What it is: the preconditioner is not symmetric. `--test amg-sym`
(`stage_a_amgsym.sbatch`) builds the plume's Poisson problem, draws two
random zero-mean vectors `r1`, `r2`, applies one V-cycle from zero to
each (`z = M r`, what every CG iteration applies) and the Laplacian
(`A r`), and compares the cross products:

| operator | `r2 . (op r1)` | `r1 . (op r2)` | relative asymmetry | `r . (op r)` |
|---|---|---|---|---|
| `A`, convective face | -1.945158e+04 | -1.945158e+04 | 4.0e-8 | 2.0e+07, positive |
| `M`, convective face | -2.220e+03 | -1.723e+03 | 0.22 | 7.4e+05, positive |
| `A`, closed box | -1.945158e+04 | -1.945158e+04 | 4.0e-8 | positive |
| `M`, closed box | -1.628e+03 | -1.475e+03 | 0.094 | positive |

The Laplacian is symmetric to single precision; the multigrid V-cycle is
not symmetric at all (the down-sweep smooths in one colour order and the
up-sweep, `ProlongGaussSeidelDot`, in the same order, so the cycle is not
its own transpose), and conjugate gradient with a non-symmetric
preconditioner has no convergence theory. The same test then runs the
solver's own loop on two right-hand sides: on white noise CG still
reduces the residual tenfold in 120 iterations (the V-cycle's smoothing
does that on its own); on a smooth right-hand side -- one low mode, what
a divergence field looks like -- the residual is lowest after the first
iteration and 2.8x higher after 120. That is the projection's behaviour
on the plume, reproduced without the plume. The remedy is either a
symmetric cycle (reverse the colour order on the way up) or a Krylov
method that tolerates a variable preconditioner; the flexible variant
of CG (Polak-Ribiere `beta` = `z_new . (r_new - r_old) / (z_old . r_old)`,
one extra vector and one dot per iteration) is the cheaper test and is
tried next.

### Flexible conjugate gradient converges where the standard recurrence diverged, and what that does to the criteria

`AMGPCG::flexible_` (`--cg-flexible`): `beta` = `z_new . (r_new - r_old) /
(z_old . r_old)` (Polak-Ribiere), one vector copy and one dot per
iteration; identical to the standard `beta` for a symmetric
preconditioner. `stage_a_cgflex.sbatch`:

| test | standard CG, last / first | flexible CG, last / first |
|---|---|---|
| `amg-sym`, random right-hand side, 120 it | 0.103 | 0.036 |
| `amg-sym`, smooth right-hand side, 120 it | 2.81 (minimum at iteration 1) | 0.70 (minimum at 117) |
| weak case, reinitialization projection at step 40, 120 it | 1.61; field l2 2.35 | 1.03; field l2 1.12 |
| the same at 60 it | (15 it: 1.55; l2 5.16) | 1.15; l2 1.50 |

The recurrence is fixed: the residual no longer rises within a
projection, and 60 flexible iterations leave less than 120 standard
ones. But it converges slowly on the smooth right-hand side (0.70 in 120
iterations) and over 600 s the divergence trend at 120 flexible (rms
1.2e-3, max 0.03-0.10) is no better than 120 standard (1.1e-3, 0.03-
0.07). The reason is in the test's own header: the multigrid has 4
levels on the plume grid, the coarsest level is 3 x 2 x 3 tiles = 9216
cells, and it is "solved" by `bottom_smoothing_` = 10 Gauss-Seidel
sweeps -- which cannot converge the smoothest modes of a 9000-unknown
problem, and those are the domain-wide, zero-mean, boundary-indifferent
residual the statistics found. `--amg-bottom N` raises the count; the
test is the next section.

The criteria feel the projection. Criterion configuration (`z0` = 100
m, `Q0` = 1 kW/m^3, 1600 s, last 1000 s), `stage-a-4-cgflex` against the
control:

| projection | residual max / rms, late | `theta_split` | `theta_width` | plume top | peak `omega_z` | peak `P(y)` |
|---|---|---|---|---|---|---|
| standard 120 (criterion run) | 0.023 / -- | 395.4 +- 0.7 | 674 +- 2 | 731 +- 1 | 0.00490 | 2.275 |
| flexible 120 | 0.0064 / 2.9e-4 | 408.8 +- 1.5 | 686 +- 2 | 727 +- 2 | 0.00534 | 2.329 |
| flexible 60 | 0.023 / 8e-4 | 420.6 +- 2.5 | 700 +- 2 | 753 +- 22 | 0.00567 | 2.313 |

Converging the projection 3.5x further moves the split by +13 m (3%,
9 standard errors), the width by +12 m, the pair strength by +9% and the
peak by +0.05 K; and 60 flexible iterations, which leave the same
maximum residual as 120 standard but a different residual field, give
yet another reading (+25 m, and an unsteady top). So the criterion
readings carry a projection-dependent uncertainty of order 15-25 m on
the split and 10-15% on the pair strength that the sampling standard
errors (0.5-2.5 m) do not show. The orderings survive it (the `z0`
column's gaps are 56-127 m); the `Q0` gap at `z0` = 150 m (8.5 m) was
already below it. The criterion configuration is not changed here --
the reading to compare across cuts stays the standard 120 -- but the
six cases must be re-read once the projection converges properly, and
the numbers above are the size of the correction to expect.

### The coarsest level's sweep count is not the bottleneck

`stage_a_amgbottom.sbatch`: `--amg-bottom N` sets `AMGPCG::bottom_smoothing_`
(the coarsest level, 3 x 2 x 3 tiles = 9216 cells on the plume grid, gets
`N` Gauss-Seidel sweeps instead of 10). `--test amg-sym`, flexible CG, 120
iterations:

| coarsest sweeps | `M` asymmetry | random right-hand side, last / first | smooth right-hand side, last / first |
|---|---|---|---|
| 10 | 0.225 | 0.036 | 0.80 |
| 100 | 0.199 | 0.038 | 0.79 |
| 400 | 0.094 | 0.036 | 0.75 |

Forty times more work on the coarsest level halves the preconditioner's
asymmetry (the incomplete bottom solve was half of it; the rest is the
smoother's colour order) and moves the smooth mode's reduction from 0.80
to 0.75. The V-cycle's coarse correction does not act on the lowest
modes of this pure-Neumann problem, whatever the bottom solve; the
residual the plume runs leave (domain-wide, zero-mean, smooth) is what
that looks like, and no count of fine-level iterations or coarse sweeps
tried here reaches it. That is a property of the trimmed-multigrid cycle
as shipped (the coarse operator or its transfer, not investigated here)
and a solver-development item, not a Stage A one.

On the plume the same: flexible 120 with 400 coarse sweeps against 10,
weak case (reinitialization projection at step 40: last / first 0.99
against 1.03, field l2 1.00 against 1.12; 600 s trend rms 1.7-2.8e-3,
max 0.03-0.11, no better) and the criterion configuration (`theta_split`
405.8 +- 1.3 against 408.8 +- 1.5, `theta_width` 683 against 686, peak
`omega_z` 0.00532 against 0.00534, residual max 0.010, rms 3.8e-4). The
two flexible readings agree with each other to within their standard
errors and both sit 10-13 m above the standard 120's 395.4 m: the
projection correction to the criteria is reproducible, and it is the
recurrence, not the coarse solve, that produced it.
