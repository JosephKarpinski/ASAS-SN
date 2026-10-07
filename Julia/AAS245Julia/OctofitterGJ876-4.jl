# ==============================================================================
# OctofitterGJ876.jl
# (Re-)Discovering the GJ876 Planets  --  Octofitter.jl v9 radial-velocity tutorial
#
# Script version of the AAS245Julia notebook 03-exoplanets.jl (a Pluto notebook
# written for Octofitter 5.2.1), migrated to the Octofitter v9 API.
#
# Run from VSCode ("Execute active file in REPL") or:  julia OctofitterGJ876.jl
#
# What it does
#   1. Loads the HARPS RVBank radial velocities for GJ876 (downloaded on first use)
#   2. Plots the data and a Lomb-Scargle periodogram
#   3. Fits a 1-planet and a 2-planet RV model with Pigeons (parallel tempering)
#   4. Compares them with the log Bayesian evidence (stepping-stone estimate)
#   5. Optional switches (default on): the notebook's two exercises -- an eccentric
#      one-planet fit, and a third planet near 123 d
#   6. Saves every plot as a PNG in ./figures_gj876 next to this script
#
# Changelog
#   v1.0.0  2026-10-07  First version. Changes from the notebook (v5 -> v9 API):
#     - `@planet ... RadialVelocityOrbit` / `@system`  ->  `Body(...)` + `System(...)`
#       with `@variables` blocks
#     - `MarginalizedStarAbsoluteRVLikelihood(rv_data, instrument_name=, jitter=)`
#       ->  `MarginalizedRVObs(rv_data; target=A, ref=Barycentre, name=, variables=)`
#     - Periods are in DAYS (notebook used years); masses are in solar masses
#       (`mjup` converts); the semi-major axis `a` is now derived automatically
#     - Orbital phase: notebook had `tau ~ Uniform(0, 2pi)` with
#       `tp = tau*P*365.256 + 55000`. In v9 tau is a phase in [0,1), so this uses
#       `tau ~ UniformCircular(1.0)` and `tp = tau*P + 55000`
#     - `Octofitter.rvpostplot(...)` -> `rvplot` (one draw) + `octoplot` (many draws)
#     - `octocorner(model_one_planet, chain1, chain2)` -> pass the TWO-planet model,
#       because octocorner resolves column names through the model it is given
#     - the notebook's second `display(chain_one_planet)` (in the two-planet
#       section) is `chain_two_planet` here
#     - the notebook's `scatter!(ax, x, y, sigma)` -> scatter + errorbars
#     - the Lomb-Scargle grid is set explicitly (1-200 d) instead of the default
#   v1.0.1  2026-10-07  Lomb-Scargle now uses `fast = false`: with AppleAccelerate
#     loaded, its FFT planner rejects the non-power-of-2 size LombScargle needs
#     ("vDSP FFT requires power-of-2 dimensions"). The periodogram step is also
#     guarded so a failure there cannot stop the run before the sampling.
#   v1.0.2  2026-10-07  Fixes from the first full run:
#     - the periodogram came back empty in Plot 2 and in the final summary: values
#       assigned inside a `try` block stayed local. They are now returned from a
#       function and assigned outside the `try`
#     - the periodogram peak list ignores periods below 1.5 d (daily-sampling aliases)
#     - plot titles no longer clip: the figure grows by the height of the title row
#   v1.1.0  2026-10-07  Added the notebook's two exercises as optional switches
#     (`RUN_ECCENTRIC_EXERCISE`, `RUN_THIRD_PLANET_EXERCISE`, both default on):
#     - Exercise 1: one-planet fit with planet b's eccentricity free in [0, 0.5]
#       (Plots 8-9, evidence compared with the circular one-planet fit)
#     - Exercise 2: a third planet near the ~123 d residual periodogram peak
#       (Plots 10-11, evidence compared with the two-planet fit)
#     They run after the core analysis, and a failure in either is caught so it
#     cannot take the rest of the run down. Each adds one more Pigeons run (the
#     three-planet fit is the slowest); set the switch to false to skip it.
# ==============================================================================

import Pkg
using Dates
using Printf

# ------------------------------------------------------------------ settings
const USE_TEMP_ENV  = true      # true: throw-away Octofitter v9 env; false: keep one in ./.octofitter_v9_env
const USE_APPLE_ACCELERATE = true   # Apple Silicon only. Note it replaces the FFT planner with a
                                    # power-of-2-only one; set false if some other FFT call fails
const RUN_ECCENTRIC_EXERCISE    = true   # Exercise 1: 1-planet fit with free eccentricity (adds Plots 8-9)
const RUN_THIRD_PLANET_EXERCISE = true   # Exercise 2: add a 3rd planet near 123 d (adds Plots 10-11; slowest)
const N_ROUNDS      = 10        # Pigeons rounds (each round doubles the work; 10 matches the notebook)
const HARPS_TARGET  = "GJ876"   # name in the HARPS RVBank catalogue
const FIGDIR        = joinpath(@__DIR__, "figures_gj876")
const T_START       = time()

dbg(args...) = println("[DEBUG ", Dates.format(now(), "HH:MM:SS"), "] ", args...)
section(title) = println("\n", "="^78, "\n", title, "\n", "="^78)

mkpath(FIGDIR)

# ------------------------------------------------------------------ environment
section("0. Environment")
if USE_TEMP_ENV
    Pkg.activate(temp = true)
else
    Pkg.activate(joinpath(@__DIR__, ".octofitter_v9_env"))
end
dbg("Julia ", VERSION, "  |  active project: ", Base.active_project())

const DEPS = Pkg.PackageSpec[
    Pkg.PackageSpec(name = "Octofitter", version = "9"),
    Pkg.PackageSpec(name = "OctofitterRadialVelocity", version = "9"),
    Pkg.PackageSpec(name = "PlanetOrbits"),
    Pkg.PackageSpec(name = "CairoMakie"),
    Pkg.PackageSpec(name = "PairPlots"),
    Pkg.PackageSpec(name = "Distributions"),
    Pkg.PackageSpec(name = "Pigeons"),
    Pkg.PackageSpec(name = "LombScargle"),
]
(Sys.isapple() && USE_APPLE_ACCELERATE) && push!(DEPS, Pkg.PackageSpec(name = "AppleAccelerate"))

let have = keys(Pkg.project().dependencies)
    todo = filter(s -> s.name ∉ have, DEPS)
    if isempty(todo)
        dbg("All packages already present in this environment")
    else
        dbg("Installing: ", join((s.name for s in todo), ", "))
        Pkg.add(todo)
    end
end

# Apple Silicon: load AppleAccelerate early so BLAS/LAPACK calls use it
if Sys.isapple() && USE_APPLE_ACCELERATE
    using AppleAccelerate
end

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"   # let DataDeps download the HARPS catalogue without prompting

using Octofitter
using OctofitterRadialVelocity
using PlanetOrbits
using CairoMakie
using PairPlots
using Distributions
using Pigeons
using LombScargle
using Statistics

CairoMakie.activate!(type = "png")

for (_, info) in Pkg.dependencies()
    info.name in ("Octofitter", "OctofitterRadialVelocity", "PlanetOrbits",
                  "Pigeons", "CairoMakie", "PairPlots", "LombScargle") &&
        dbg(rpad(info.name, 26), "v", info.version)
end

# ------------------------------------------------------------------ helpers
"Run `f`; on failure print the error and carry on (so one bad plot cannot lose a long sampling run)."
function safe(f, label)
    try
        f()
        dbg(label, " OK")
    catch err
        @warn "$label FAILED -- continuing" exception = (err, catch_backtrace())
    end
    return nothing
end

"""
Add a wrapped title in a new row above everything in `fig`, and grow the figure by
exactly that row's height so the existing panels keep their size. (Without the
resize, figures sized tightly to their content, like the corner plots, get their
top or bottom clipped.)
"""
function add_plot_title!(fig, title; fontsize = 14)
    try
        dims = try size(fig.scene) catch; nothing end     # (width, height) in pixels
        w = dims === nothing ? 900 : dims[1]
        chars_per_line = max(20, floor(Int, (w - 40) / (fontsize * 0.62)))
        nlines = max(1, ceil(Int, length(title) / chars_per_line))
        title_h = nlines * fontsize * 1.4 + 10
        Label(fig[0, :], title; fontsize, font = :bold, halign = :left,
              justification = :left, word_wrap = true, tellwidth = false,
              tellheight = true, height = title_h, padding = (8, 8, 6, 2))
        dims === nothing || Makie.resize!(fig, Int(w), round(Int, dims[2] + title_h + 8))   # + default row gap
    catch err
        @warn "Could not add the plot title (saving without it)" exception = err
    end
    return fig
end

"Add a title, save as PNG into FIGDIR, and return the path. Accepts a Figure or an OctoPlotResult."
function save_plot!(obj, title, filename)
    fig = obj isa OctoPlotResult ? obj.figure : obj
    add_plot_title!(fig, title)
    path = joinpath(FIGDIR, filename)
    Makie.save(path, fig; px_per_unit = 2)
    dbg("saved ", path)
    return path
end

chain_vec(chain, name::AbstractString) = Float64.(chain[name][:])

"Median and 16/84 percent range of a chain column, as a short string."
function describe_var(chain, name::AbstractString; scale = 1.0, digits = 4)
    v = chain_vec(chain, name) .* scale
    q = quantile(v, [0.16, 0.5, 0.84])
    return "$(round(q[2]; digits)) +$(round(q[3] - q[2]; digits)) / -$(round(q[2] - q[1]; digits))"
end

function bf_interpretation(lnbf; more = "the two-planet model", less = "the one-planet model")
    isfinite(lnbf) || return "evidence not available"
    a = abs(lnbf)
    a == 0 && return "no evidence either way"
    strength = a > 3.0 ? "Extreme" : a > 1.61 ? "Very strong" : a > 1.10 ? "Strong" :
               a > 0.69 ? "Moderate" : "Anecdotal"
    hyp = lnbf > 0 ? more : less
    return "$strength evidence for $hyp"
end

# ------------------------------------------------------------------ 1. data
section("1. Loading the HARPS RVBank data for $HARPS_TARGET")
# Returns a Table with epoch [MJD], rv [m/s], sigma_rv [m/s]. The first call
# downloads the catalogue (~38 MB); credit the source it prints.
rv_data = OctofitterRadialVelocity.HARPS_RVBank_rvs(HARPS_TARGET)

dbg("RV measurements: ", length(rv_data.epoch))
dbg("Epoch range [MJD]: ", round(minimum(rv_data.epoch); digits = 1), " to ",
    round(maximum(rv_data.epoch); digits = 1), "  (baseline ",
    round(maximum(rv_data.epoch) - minimum(rv_data.epoch); digits = 1), " d)")
dbg("RV range [m/s]: ", round(minimum(rv_data.rv); digits = 1), " to ",
    round(maximum(rv_data.rv); digits = 1))
dbg("Median RV uncertainty [m/s]: ", round(median(rv_data.σ_rv); digits = 2))

safe("Plot 1") do
    fig = Figure(size = (900, 450))
    ax = Axis(fig[1, 1], xlabel = "epoch [MJD]", ylabel = "RV [m/s]")
    Makie.scatter!(ax, rv_data.epoch, rv_data.rv, color = :black, markersize = 5)
    Makie.errorbars!(ax, rv_data.epoch, rv_data.rv, rv_data.σ_rv, color = (:black, 0.4))
    save_plot!(fig, "Plot 1. HARPS RVBank radial velocities of $HARPS_TARGET (error bars are the " *
                    "pipeline uncertainties).", "plot01_harps_rv_data.png")
end

# ------------------------------------------------------------------ 2. periodogram
section("2. Lomb-Scargle periodogram")
function compute_periodogram(data)
    # `fast = false` skips LombScargle's FFT-based approximation. With
    # AppleAccelerate loaded, the FFT planner only accepts power-of-2 sizes
    # and LombScargle needs an arbitrary one (error: "vDSP FFT requires
    # power-of-2 dimensions"). The exact method is instant for ~300 points.
    pgram = lombscargle(data.epoch, data.rv, data.σ_rv;
                        minimum_frequency = 1 / 200, maximum_frequency = 1.0,
                        samples_per_peak = 10, fast = false)
    per, pow = LombScargle.periodpower(pgram)

    # Strongest local maxima between 1.5 and 200 days. Peaks near 1 d are
    # aliases of the once-per-night sampling, not real signals.
    peak_idx = [i for i in 2:length(pow)-1 if pow[i] > pow[i-1] && pow[i] >= pow[i+1] && 1.5 <= per[i] <= 200.0]
    sort!(peak_idx, by = i -> -pow[i])
    return per, pow, peak_idx[1:min(5, length(peak_idx))]
end

# The result is assigned outside the `try`: a `try` block is its own scope, so
# assigning `per`/`pow` inside it would create locals and leave these globals empty.
per, pow, top_peaks = try
    compute_periodogram(rv_data)
catch err
    @warn "Lomb-Scargle periodogram FAILED -- continuing without it" exception = (err, catch_backtrace())
    (Float64[], Float64[], Int[])
end
dbg("Periodogram grid points: ", length(per))
for i in top_peaks
    dbg("  LS peak: period = ", round(per[i]; digits = 2), " d, power = ", round(pow[i]; digits = 3))
end

safe("Plot 2") do
    fig = Figure(size = (900, 450))
    ax = Axis(fig[1, 1], xlabel = "period [d]", ylabel = "power")
    Makie.lines!(ax, per, pow, color = :black)
    # the notebook had these three guide lines commented out
    Makie.vlines!(ax, [30, 61, 123], color = (:crimson, 0.5), linestyle = :dash)
    Makie.xlims!(ax, 0, 200)
    save_plot!(fig, "Plot 2. Lomb-Scargle periodogram of the HARPS RVs. Dashed lines mark 30, 61 and " *
                    "123 d.", "plot02_lomb_scargle.png")
end

# ------------------------------------------------------------------ 3. model pieces
section("3. Model definition")

# Host star. Priors from the notebook: M = 0.346 +/- 0.007 solar masses.
A = Body(
    name = "A",
    variables = @variables begin
        mass ~ truncated(Normal(0.346, 0.007), lower = 0.1)   # M_sun
    end
)

# Planet b -- the ~61 d signal. Circular orbit, period prior 61.1057 +/- 5 d.
b = Body(
    name = "b",
    about = A,
    variables = @variables begin
        # RV-only fit: inclination and node are unconstrained, so fix them.
        # With i = pi/2 the fitted `mass` is a minimum mass (m sin i).
        i = pi / 2
        Ω = 0.0
        e = 0.0
        ω = 0.0

        P ~ Uniform(61.1057 - 5, 61.1057 + 5)          # days (the notebook gave this in years)

        τ ~ UniformCircular(1.0)                        # orbital phase in [0, 1)
        tp = τ * P + 55000                              # reference epoch (MJD) near the data

        mass ~ LogUniform(0.001mjup, 10mjup)            # minimum mass, in M_sun
    end
)

# Planet c -- the ~30 d signal. Eccentric, period prior 30 +/- 5 d.
# Both planets orbit the star directly (about = A), as in the notebook. The
# Jacobi form `about=(A, b)` is only appropriate when c is the OUTER planet,
# and here c (30 d) is inside b (61 d).
c = Body(
    name = "c",
    about = A,
    variables = @variables begin
        i = pi / 2
        Ω = 0.0
        e ~ Uniform(0, 0.7)
        ω ~ Uniform(0, 2pi)

        P ~ Uniform(30 - 5, 30 + 5)                     # days

        τ ~ UniformCircular(1.0)
        tp = τ * P + 55000

        mass ~ LogUniform(0.001mjup, 10mjup)
    end
)

# One RV series with its zero point marginalized out and a free jitter term.
rv_likelihood = MarginalizedRVObs(
    rv_data;
    target = A, ref = Barycentre,
    name = "HARPS",
    variables = @variables begin
        jitter ~ LogUniform(0.1, 100)                   # m/s
    end
)
dbg("Built bodies A, b, c and the HARPS likelihood")

# ------------------------------------------------------------------ 4. one planet
section("4. One-planet model (b)")
sys_one_planet = System(
    name = "GJ876_one_planet",
    bodies = [A, b],
    observations = [rv_likelihood],
)
model_one_planet = Octofitter.LogDensityModel(sys_one_planet)
dbg("One-planet LogDensityModel compiled; sampling with Pigeons, n_rounds = ", N_ROUNDS)

t0 = time()
chain_one_planet, pt_one_planet = octofit_pigeons(model_one_planet, n_rounds = N_ROUNDS)
dbg("One-planet sampling took ", round(time() - t0; digits = 1), " s")
safe("Chain column listing") do
    dbg("Chain columns: ", names(chain_one_planet))
end

display(chain_one_planet)

safe("Plot 3") do
    res = rvplot(model_one_planet, chain_one_planet)
    save_plot!(res, "Plot 3. One-planet fit (planet b): RV curve for the maximum-posterior draw, " *
                    "residuals, and the phase-folded signal.", "plot03_one_planet_rvplot.png")
end

# ------------------------------------------------------------------ 5. two planets
section("5. Two-planet model (b + c)")
sys_two_planet = System(
    name = "GJ876_two_planet",
    bodies = [A, b, c],
    observations = [rv_likelihood],
)
model_two_planet = Octofitter.LogDensityModel(sys_two_planet)
dbg("Two-planet LogDensityModel compiled; sampling with Pigeons, n_rounds = ", N_ROUNDS)

t0 = time()
chain_two_planet, pt_two_planet = octofit_pigeons(model_two_planet, n_rounds = N_ROUNDS)
dbg("Two-planet sampling took ", round(time() - t0; digits = 1), " s")
safe("Chain column listing") do
    dbg("Chain columns: ", names(chain_two_planet))
end

display(chain_two_planet)

safe("Plot 4") do
    res = rvplot(model_two_planet, chain_two_planet)
    save_plot!(res, "Plot 4. Two-planet fit (b + c): RV curve for the maximum-posterior draw, " *
                    "residuals, and phase-folded signals.", "plot04_two_planet_rvplot.png")
end

safe("Plot 5") do
    res = octoplot(model_two_planet, chain_two_planet)
    save_plot!(res, "Plot 5. Two-planet fit: spread of orbits allowed by the posterior over the " *
                    "HARPS data.", "plot05_two_planet_octoplot.png")
end

# ------------------------------------------------------------------ 6. model comparison
section("6. Model comparison: log Bayesian evidence")
lnZ_one = Pigeons.stepping_stone(pt_one_planet)
lnZ_two = Pigeons.stepping_stone(pt_two_planet)
ln_BF   = lnZ_two - lnZ_one
dbg("ln Z (one planet) = ", round(lnZ_one; digits = 2))
dbg("ln Z (two planets) = ", round(lnZ_two; digits = 2))
dbg("ln BF (two vs one) = ", round(ln_BF; digits = 2), "  ->  ", bf_interpretation(ln_BF))

# ------------------------------------------------------------------ 7. corner plots
section("7. Corner plots")
safe("Plot 6") do
    # Pass the two-planet model: octocorner classifies every chain column against the
    # model it is given, and the two-planet model contains all the columns of both chains.
    fig = octocorner(model_two_planet, chain_one_planet, chain_two_planet, small = true)
    save_plot!(fig, "Plot 6. How adding planet c changes the other parameters. Series 1 (blue) is " *
                    "the one-planet fit, series 2 (orange) the two-planet fit.",
               "plot06_corner_one_vs_two.png")
end

safe("Plot 7") do
    fig = octocorner(model_two_planet, chain_two_planet, small = true)
    save_plot!(fig, "Plot 7. Corner plot of the two-planet posterior, showing parameter covariances.",
               "plot07_corner_two_planet.png")
end

# ------------------------------------------------------------------ 8. exercise 1
"Sample `model` with Pigeons and return (chain, pt, lnZ), or `nothing` if anything fails."
function sample_optional(label, model)
    try
        t0 = time()
        chain, pt = octofit_pigeons(model, n_rounds = N_ROUNDS)
        dbg(label, " sampling took ", round(time() - t0; digits = 1), " s")
        return (chain, pt, Pigeons.stepping_stone(pt))
    catch err
        @warn "$label FAILED -- continuing" exception = (err, catch_backtrace())
        return nothing
    end
end

chain_ecc = nothing
lnZ_ecc   = NaN
if RUN_ECCENTRIC_EXERCISE
    section("8. Exercise 1: one-planet model with eccentricity")
    # Same as planet b above, but e is free in [0, 0.5] and omega is free.
    # (Literal numbers here: the @variables block is best kept to plain values.)
    b_ecc = Body(
        name = "b",
        about = A,
        variables = @variables begin
            i = pi / 2
            Ω = 0.0
            e ~ Uniform(0, 0.5)
            ω ~ Uniform(0, 2pi)

            P ~ Uniform(61.1057 - 5, 61.1057 + 5)       # days

            τ ~ UniformCircular(1.0)
            tp = τ * P + 55000

            mass ~ LogUniform(0.001mjup, 10mjup)
        end
    )
    sys_ecc = System(name = "GJ876_one_planet_ecc", bodies = [A, b_ecc], observations = [rv_likelihood])
    model_ecc = Octofitter.LogDensityModel(sys_ecc)
    dbg("Eccentric one-planet model compiled; sampling with Pigeons, n_rounds = ", N_ROUNDS)
    res_ecc = sample_optional("Eccentric one-planet", model_ecc)
    if res_ecc !== nothing
        chain_ecc, _, lnZ_ecc = res_ecc
        display(chain_ecc)
        dbg("ln Z (eccentric one planet) = ", round(lnZ_ecc; digits = 2),
            "   ln BF (eccentric vs circular) = ", round(lnZ_ecc - lnZ_one; digits = 2), "  ->  ",
            bf_interpretation(lnZ_ecc - lnZ_one; more = "the eccentric model", less = "the circular model"))
        safe("Plot 8") do
            res = rvplot(model_ecc, chain_ecc)
            save_plot!(res, "Plot 8. Exercise 1: one-planet fit with free eccentricity (0 to 0.5): RV curve " *
                            "for the maximum-posterior draw, residuals, and phase-folded signal.",
                       "plot08_one_planet_eccentric_rvplot.png")
        end
        safe("Plot 9") do
            # chain_one_planet has no free e or omega; the eccentric model has every column of both.
            fig = octocorner(model_ecc, chain_one_planet, chain_ecc, small = true)
            save_plot!(fig, "Plot 9. Effect of allowing eccentricity on planet b. Series 1 (blue) is the " *
                            "circular fit, series 2 (orange) the eccentric fit.",
                       "plot09_corner_circular_vs_eccentric.png")
        end
    end
end

# ------------------------------------------------------------------ 9. exercise 2
chain_three = nothing
lnZ_three   = NaN
if RUN_THIRD_PLANET_EXERCISE
    section("9. Exercise 2: three-planet model (b + c + d)")
    # The two-planet residuals still show a signal near 123 d (see the dashed line in
    # Plot 2). Search 100-150 d with a free eccentricity, like planet c.
    d = Body(
        name = "d",
        about = A,
        variables = @variables begin
            i = pi / 2
            Ω = 0.0
            e ~ Uniform(0, 0.7)
            ω ~ Uniform(0, 2pi)

            P ~ Uniform(100, 150)                       # days

            τ ~ UniformCircular(1.0)
            tp = τ * P + 55000

            mass ~ LogUniform(0.001mjup, 10mjup)
        end
    )
    sys_three = System(name = "GJ876_three_planet", bodies = [A, b, c, d], observations = [rv_likelihood])
    model_three = Octofitter.LogDensityModel(sys_three)
    dbg("Three-planet model compiled; sampling with Pigeons, n_rounds = ", N_ROUNDS,
        " (this is the slowest fit)")
    res_three = sample_optional("Three-planet", model_three)
    if res_three !== nothing
        chain_three, _, lnZ_three = res_three
        display(chain_three)
        dbg("ln Z (three planets) = ", round(lnZ_three; digits = 2),
            "   ln BF (three vs two) = ", round(lnZ_three - lnZ_two; digits = 2), "  ->  ",
            bf_interpretation(lnZ_three - lnZ_two; more = "the three-planet model", less = "the two-planet model"))
        safe("Plot 10") do
            res = rvplot(model_three, chain_three)
            save_plot!(res, "Plot 10. Exercise 2: three-planet fit (b + c + d): RV curve for the " *
                             "maximum-posterior draw, residuals, and phase-folded signals.",
                       "plot10_three_planet_rvplot.png")
        end
        safe("Plot 11") do
            fig = octocorner(model_three, chain_two_planet, chain_three, small = true)
            save_plot!(fig, "Plot 11. Effect of adding planet d. Series 1 (blue) is the two-planet fit, " *
                             "series 2 (orange) the three-planet fit.",
                       "plot11_corner_two_vs_three.png")
        end
    end
end

# ------------------------------------------------------------------ 10. analysis summary
section("10. Analysis summary")
safe("Summary") do
    println("Data: ", length(rv_data.epoch), " HARPS RVs over ",
            round(maximum(rv_data.epoch) - minimum(rv_data.epoch); digits = 0), " days")
    println("Strongest periodogram peaks (days): ",
            join((round(per[i]; digits = 2) for i in top_peaks), ", "))
    println()
    println("One-planet fit (median, 16/84 percent range):")
    println("  b  period [d]        = ", describe_var(chain_one_planet, "b_P"))
    println("  b  min. mass [Mjup]  = ", describe_var(chain_one_planet, "b_mass"; scale = 1 / mjup))
    println("  HARPS jitter [m/s]   = ", describe_var(chain_one_planet, "HARPS_jitter"))
    println()
    println("Two-planet fit:")
    println("  b  period [d]        = ", describe_var(chain_two_planet, "b_P"))
    println("  b  min. mass [Mjup]  = ", describe_var(chain_two_planet, "b_mass"; scale = 1 / mjup))
    println("  c  period [d]        = ", describe_var(chain_two_planet, "c_P"))
    println("  c  eccentricity      = ", describe_var(chain_two_planet, "c_e"))
    println("  c  min. mass [Mjup]  = ", describe_var(chain_two_planet, "c_mass"; scale = 1 / mjup))
    println("  HARPS jitter [m/s]   = ", describe_var(chain_two_planet, "HARPS_jitter"))
    println()
    Δjit = median(chain_vec(chain_two_planet, "HARPS_jitter")) - median(chain_vec(chain_one_planet, "HARPS_jitter"))
    println("Adding planet c changes the fitted jitter by ", round(Δjit; digits = 2),
            " m/s (median); a large drop means c explains real signal rather than noise.")
    println()
    println("Evidence: ln Z(1 planet) = ", round(lnZ_one; digits = 2),
            ", ln Z(2 planets) = ", round(lnZ_two; digits = 2))
    println("          ln BF(2 vs 1) = ", round(ln_BF; digits = 2), "  ->  ", bf_interpretation(ln_BF))
    if chain_ecc !== nothing
        println()
        println("Exercise 1 -- one planet with free eccentricity:")
        println("  b  period [d]        = ", describe_var(chain_ecc, "b_P"))
        println("  b  eccentricity      = ", describe_var(chain_ecc, "b_e"))
        println("  b  min. mass [Mjup]  = ", describe_var(chain_ecc, "b_mass"; scale = 1 / mjup))
        println("  HARPS jitter [m/s]   = ", describe_var(chain_ecc, "HARPS_jitter"))
        println("  ln Z = ", round(lnZ_ecc; digits = 2), ", ln BF(eccentric vs circular) = ",
                round(lnZ_ecc - lnZ_one; digits = 2), "  ->  ",
                bf_interpretation(lnZ_ecc - lnZ_one; more = "the eccentric model", less = "the circular model"))
        println("  For comparison, the second planet c gave ln BF = ", round(ln_BF; digits = 2), ".")
    end
    if chain_three !== nothing
        println()
        println("Exercise 2 -- three planets:")
        println("  b  period [d]        = ", describe_var(chain_three, "b_P"))
        println("  c  period [d]        = ", describe_var(chain_three, "c_P"))
        println("  d  period [d]        = ", describe_var(chain_three, "d_P"))
        println("  d  eccentricity      = ", describe_var(chain_three, "d_e"))
        println("  d  min. mass [Mjup]  = ", describe_var(chain_three, "d_mass"; scale = 1 / mjup))
        println("  HARPS jitter [m/s]   = ", describe_var(chain_three, "HARPS_jitter"))
        println("  ln Z = ", round(lnZ_three; digits = 2), ", ln BF(3 vs 2) = ",
                round(lnZ_three - lnZ_two; digits = 2), "  ->  ",
                bf_interpretation(lnZ_three - lnZ_two; more = "the three-planet model", less = "the two-planet model"))
    end
end

section("Done")
dbg("Total run time: ", round((time() - T_START) / 60; digits = 1), " min")
dbg("Figures saved in: ", FIGDIR)
for f in sort(readdir(FIGDIR))
    endswith(f, ".png") && dbg("  ", f)
end
