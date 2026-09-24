#=
================================================================================
 OctofitterObsPrior.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Observable-Based Priors"
 (O'Neil et al. 2019, AJ 158, 4 — please cite if you use this prior):
   https://sefffal.github.io/Octofitter.jl/dev/rel-astrom-obs/

 Pipeline:
   Body A (star) + Body b (period P ~ LogUniform, θ from θ_x/θ_y at epoch)
   -> RelAstromObs (8 epochs ra/dec with correlations; jitter/northangle/
      platescale fixed by default)
   -> ObsPriorONeil2019(obs)  [ONLY the wrapper goes in observations=]
   -> System "TutoriaPrime" -> LogDensityModel -> initialize! -> octofit
   Baseline for comparison (not defined on the tutorial page): identical
   model WITHOUT the wrapper, i.e. uniform-in-elements priors
   -> System "TutoriaUniform" -> same steps
   -> orbit plots for both, corner plot comparing both, prior-sensitivity
      histograms (P, derived a, e), side-by-side median table, FITS chains

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `obsprior_result`)
   Shell:    julia OctofitterObsPrior.jl
   Library:  ENV["OCTOOP_AUTORUN"] = "false"; include("OctofitterObsPrior.jl")
             res = OctofitterObsPrior.run_obsprior(iterations=2000)

 Environment: bootstrap as in OctofitterRelAstrom.jl (temporary v9 env by
 default; ENV["OCTOOP_ENV_MODE"] = "local" uses ./octofitter_v9_env, shared
 with the other tutorial modules). Needs Octofitter v9, Distributions,
 CairoMakie, PairPlots, MCMCChains.

 Outputs (prefix "obsprior_") go to the directory containing this file.

 Changelog
 ---------
 v1.0.1 2026-09-24 Histogram/trace colours now match the corner plot
                   (uniform = blue, observable = orange; previously swapped).
                   Trace-plot legend moved outside the axis.
 v1.0  2026-09-24  Initial version on the OctofitterRelAstrom v1.0
                   scaffolding ([OOP +t] debug stages, soft optional stages,
                   env bootstrap, v9 guard, @__DIR__ outputs, dark theme with
                   light corner plots). Adds: P/θ_x/θ_y parameterisation;
                   ObsPriorONeil2019 wrapper; uniform-prior baseline model
                   built identically minus the wrapper; derived a from
                   Kepler's third law; comparison table + histograms;
                   `free_systematics` switch for jitter/northangle/platescale.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "Distributions", "CairoMakie", "PairPlots", "MCMCChains"),
    mode     = lowercase(get(ENV, "OCTOOP_ENV_MODE", "temp"))

    direct_deps() = Dict(info.name => info.version
                         for info in values(Pkg.dependencies()) if info.is_direct_dep)
    missing_deps() = [n for n in required if !haskey(direct_deps(), n)]
    function env_ok()
        v = get(direct_deps(), "Octofitter", nothing)
        v !== nothing && v >= v"9" && isempty(missing_deps())
    end

    loaded = [m for (id, m) in Base.loaded_modules if id.name == "Octofitter"]

    if !isempty(loaded)
        v = pkgversion(first(loaded))
        if v !== nothing && v < v"9"
            error("""
            Octofitter v$v is already loaded in this Julia session, and a loaded
            package can't be swapped for another version. Restart the REPL
            (VSCode: "Julia: Restart REPL") and execute this file again; the
            bootstrap will then set up Octofitter v9 before anything is loaded.""")
        end
        miss = filter(!=("Octofitter"), missing_deps())
        if isempty(miss)
            println("[OOP] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[OOP] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(miss; preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[OOP] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[OOP] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[OOP] Existing v9 environment found at ", Base.active_project())
        else
            specs = [n == "Octofitter" ? Pkg.PackageSpec(name=n, version="9") :
                                         Pkg.PackageSpec(name=n) for n in required]
            Pkg.add(specs)
        end
        println("[OOP] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[OOP] Loading Octofitter, Distributions, CairoMakie, PairPlots, MCMCChains ",
        "(first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterObsPrior

const LOAD_T0 = time()

using Octofitter
using Distributions
using CairoMakie
using PairPlots
import MCMCChains
using Printf
import Statistics

export run_obsprior, build_star, build_planet, astrometry_table, build_obs,
       build_tutorial_system, derived_a, set_plot_theme!

const VERSION_STRING = "1.0.1"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterObsPrior needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up Octofitter v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const DAYS_PER_YEAR = 365.25

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterObsPrior.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[OOP +%7.1fs] ", time() - T0[]), msg...)
    flush(stdout)
    return nothing
end

"Run `f()` with timing; on error report the stage and rethrow."
function stage(f, name::AbstractString)
    dbg("▶ ", name)
    t = time()
    result = try
        f()
    catch err
        dbg(@sprintf("✖ %s FAILED after %.2f s: ", name, time() - t),
            sprint(showerror, err))
        rethrow()
    end
    dbg(@sprintf("✔ %s (%.2f s)", name, time() - t))
    return result
end

"Like `stage`, but for optional steps: on error, log it and return `nothing`."
function soft_stage(f, name::AbstractString)
    dbg("▶ ", name, "  (optional)")
    t = time()
    try
        result = f()
        dbg(@sprintf("✔ %s (%.2f s)", name, time() - t))
        return result
    catch err
        dbg(@sprintf("⚠ %s skipped after %.2f s: ", name, time() - t),
            sprint(showerror, err))
        return nothing
    end
end

function print_env_info(outdir)
    DEBUG[] || return nothing
    dbg("OctofitterObsPrior v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | threads ", Threads.nthreads())
    dbg("project    = ", Base.active_project())
    dbg("pwd()      = ", pwd())
    dbg("SCRIPT_DIR = ", SCRIPT_DIR)
    dbg("outdir     = ", abspath(outdir))
    dbg("interactive = ", isinteractive(),
        " | Makie backend = ", CairoMakie.Makie.current_backend())
    return nothing
end

colvec(chain, p::Symbol) = vec(Array(chain[p]))
colmat(chain, p::Symbol) = reshape(Array(chain[p]), size(chain, 1), :)
haspar(chain, p::Symbol) = p in names(chain)

"""
    derived_a(chain; host=:A_mass, planet=:b)

Semi-major axis [AU] per sample. Uses the chain's `b_a` column if present,
otherwise Kepler's third law with M = host mass (planet mass fixed at 0):
a³ = M P², P in years.
"""
function derived_a(chain; host::Symbol=:A_mass, planet::Symbol=:b)
    pa = Symbol(planet, "_a")
    haspar(chain, pa) && return colvec(chain, pa)
    P = colvec(chain, Symbol(planet, "_P")) ./ DAYS_PER_YEAR
    return cbrt.(colvec(chain, host) .* P .^ 2)
end

function print_divergences(chain; label="chain")
    DEBUG[] || return nothing
    try
        m = colmat(chain, :numerical_error)
        for c in axes(m, 2)
            n = Int(sum(m[:, c]))
            dbg(@sprintf("  %s: divergences %d / %d (%.1f%%)", label, n,
                         size(m, 1), 100n / size(m, 1)))
        end
    catch
        dbg("  (numerical_error not found in ", label, ")")
    end
    try
        dbg(@sprintf("  %s mean acceptance rate = %.3f", label,
                     Statistics.mean(colvec(chain, :acceptance_rate))))
    catch
    end
    return nothing
end

qline(x) = Statistics.quantile(x, (0.16, 0.5, 0.84))

function print_posterior_summary(chain; label="chain")
    DEBUG[] || return nothing
    dbg("Posterior median [16th, 84th] — ", label, " (", size(chain, 1), " iter):")
    params = Symbol[:plx, :A_mass, :b_P, :b_e, :b_i, :b_ω, :b_Ω, :b_θ]
    append!(params, filter(n -> occursin(r"_(jitter|northangle|platescale)$", string(n)),
                           names(chain)))
    for p in params
        haspar(chain, p) || continue
        q = qline(colvec(chain, p))
        dbg(@sprintf("    %-40s %12.4f  [%12.4f, %12.4f]", p, q[2], q[1], q[3]))
    end
    q = qline(colvec(chain, :b_P) ./ DAYS_PER_YEAR)
    dbg(@sprintf("    %-40s %12.4f  [%12.4f, %12.4f]", "P [yr] (derived)", q[2], q[1], q[3]))
    q = qline(derived_a(chain))
    dbg(@sprintf("    %-40s %12.4f  [%12.4f, %12.4f]", "a [AU] (Kepler, derived)", q[2], q[1], q[3]))
    return nothing
end

function print_rhat(chain; tol=0.01)
    ss = MCMCChains.summarystats(chain)
    DEBUG[] && display(ss)
    try
        bad = [(p, r) for (p, r) in zip(ss[:, :parameters], ss[:, :rhat])
               if isfinite(r) && abs(r - 1) > tol]
        if isempty(bad)
            dbg(@sprintf("  All finite R̂ within 1 ± %.2f", tol))
        else
            dbg(@sprintf("  %d parameter(s) with |R̂-1| > %.2f:", length(bad), tol))
            for (p, r) in bad
                dbg(@sprintf("    %-40s R̂ = %.4f", p, r))
            end
        end
    catch
        dbg("  (couldn't extract the rhat column; see table above)")
    end
    return ss
end

"Side-by-side medians [16th, 84th] for the two fits."
function print_comparison(chain_u, chain_op)
    DEBUG[] || return nothing
    rows = [
        ("P [yr]",      c -> colvec(c, :b_P) ./ DAYS_PER_YEAR),
        ("a [AU]",      derived_a),
        ("e",           c -> colvec(c, :b_e)),
        ("i [deg]",     c -> rad2deg.(colvec(c, :b_i))),
        ("A_mass [M⊙]", c -> colvec(c, :A_mass)),
        ("plx [mas]",   c -> colvec(c, :plx)),
    ]
    dbg("Prior comparison — median [16th, 84th]:")
    dbg(@sprintf("    %-12s %-34s %-34s", "", "uniform priors", "observable priors (O'Neil 2019)"))
    for (lab, f) in rows
        qu = qline(f(chain_u)); qo = qline(f(chain_op))
        dbg(@sprintf("    %-12s %9.3f [%9.3f, %9.3f]   %9.3f [%9.3f, %9.3f]",
                     lab, qu[2], qu[1], qu[3], qo[2], qo[1], qo[3]))
    end
    return nothing
end

function list_outputs(paths)
    DEBUG[] || return nothing
    dbg("Output files:")
    for p in paths
        isfile(p) ? dbg(@sprintf("    %8.1f kB  %s", filesize(p) / 1024, p)) :
                    dbg("    MISSING    ", p)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Plot theme and figures
# -----------------------------------------------------------------------------
"""
    set_plot_theme!(; dark=true)

Dark publication theme (#111111 / #EEEEEE) or Makie default when `dark=false`.
"""
function set_plot_theme!(; dark::Bool=true)
    if dark
        fg = "#EEEEEE"
        set_theme!(merge(
            Theme(
                backgroundcolor = "#111111",
                textcolor = fg,
                Axis = (
                    backgroundcolor = "#111111",
                    xgridcolor = (:white, 0.08), ygridcolor = (:white, 0.08),
                    leftspinecolor = fg, rightspinecolor = fg,
                    topspinecolor = fg, bottomspinecolor = fg,
                    xtickcolor = fg, ytickcolor = fg,
                    xlabelcolor = fg, ylabelcolor = fg,
                    xticklabelcolor = fg, yticklabelcolor = fg,
                    titlecolor = fg,
                ),
                Colorbar = (tickcolor = fg, ticklabelcolor = fg, labelcolor = fg),
                Legend = (backgroundcolor = "#111111", labelcolor = fg, framecolor = fg),
            ),
            theme_dark(),
        ))
    else
        set_theme!()
    end
    dbg("Plot theme: ", dark ? "dark (#111111 / #EEEEEE)" : "Makie default (light)")
    return nothing
end

corner(model, chains...; dark::Bool=false, small::Bool=true) =
    dark ? octocorner(model, chains...; small=small) :
           with_theme(() -> octocorner(model, chains...; small=small), Theme())

# Match PairPlots' series order in the corner plot (1st = blue, 2nd = orange),
# since the comparison corner is drawn as (uniform, observable).
const C_UNIF = "#0072B2"   # blue
const C_OBSP = "#E69F00"   # orange

"Histograms of log10 P, derived a and e for both fits."
function comparison_figure(chain_u, chain_op)
    fig = Figure(size=(1300, 420))
    specs = [
        ("log₁₀ P [days]", c -> log10.(colvec(c, :b_P))),
        ("a [AU] (derived)", derived_a),
        ("e", c -> colvec(c, :b_e)),
    ]
    for (k, (lab, f)) in enumerate(specs)
        ax = Axis(fig[1, k], xlabel=lab, ylabel=k == 1 ? "density" : "")
        xu, xo = f(chain_u), f(chain_op)
        if lab == "a [AU] (derived)"
            # clip the long uniform-prior tail at its 99th percentile for readability
            hi = max(Statistics.quantile(xu, 0.99), Statistics.quantile(xo, 0.99))
            xu, xo = filter(<=(hi), xu), filter(<=(hi), xo)
        end
        hist!(ax, xu; bins=60, normalization=:pdf, color=(C_UNIF, 0.45), label="uniform priors")
        hist!(ax, xo; bins=60, normalization=:pdf, color=(C_OBSP, 0.45), label="observable priors")
        k == 1 && axislegend(ax, position=:rt)
    end
    Label(fig[0, :], "Prior sensitivity: uniform vs O'Neil et al. 2019 observable priors",
          fontsize=18, font=:bold)
    return fig
end

"Trace of log10 P for both fits."
function trace_figure(chain_u, chain_op)
    fig = Figure(size=(1000, 380))
    ax = Axis(fig[1, 1], xlabel="iteration", ylabel="log₁₀ P [days]", title="Trace: b_P")
    lines!(ax, log10.(colvec(chain_u, :b_P)), color=C_UNIF, linewidth=0.6, label="uniform priors")
    lines!(ax, log10.(colvec(chain_op, :b_P)), color=C_OBSP, linewidth=0.6, label="observable priors")
    Legend(fig[1, 2], ax, framevisible=false)   # outside the axis: traces fill it
    return fig
end

# -----------------------------------------------------------------------------
# Model components (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
"Tutorial data: 8 epochs of ra/dec [mas] with per-epoch correlations."
function astrometry_table()
    return Table(;
        epoch = Float64[50000, 50120, 50240, 50360, 50480, 50600, 50720, 50840],  # MJD
        ra    = [-494.4, -495.0, -493.7, -490.4, -485.2, -478.1, -469.1, -458.3],
        dec   = [-76.7, -44.9, -12.9, 19.1, 51.0, 82.8, 114.3, 145.3],
        σ_ra  = [12.6, 10.4, 9.9, 8.7, 8.0, 6.9, 5.8, 4.2],
        σ_dec = [12.6, 10.4, 9.9, 8.7, 8.0, 6.9, 5.8, 4.2],
        cor   = [0.2, 0.5, 0.1, -0.8, 0.3, -0.0, 0.1, -0.2],
    )
end

"""
    build_star(; name="A", mass_prior=truncated(Normal(1.2, 0.1), lower=0.1))
"""
function build_star(; name::AbstractString="A",
                      mass_prior::Distribution=truncated(Normal(1.2, 0.1), lower=0.1))
    dbg("  star '", name, "': mass ~ ", mass_prior)
    vars = @eval @variables begin
        mass ~ $mass_prior           # [M⊙]
    end
    return Body(name=name, variables=vars)
end

"""
    build_planet(host; name="b", epoch=50420.0, mass=0.0,
                 P_prior=LogUniform(35, 55_000), e_prior=Uniform(0.0, 0.5))

Planet with period prior `P_prior` [days] (results are sensitive to it) and
orbital phase θ = atan(θ_y, θ_x) at `epoch` [MJD].
"""
function build_planet(host; name::AbstractString="b",
                            epoch::Real=50420.0,
                            mass::Real=0.0,
                            P_prior::Distribution=LogUniform(35, 55_000),
                            e_prior::Distribution=Uniform(0.0, 0.5))
    ep = Float64(epoch)
    m  = Float64(mass)
    dbg("  planet '", name, "': epoch = ", ep, " MJD (", mjd2date(ep), "), mass = ", m,
        ", P ~ ", P_prior, " [d], e ~ ", e_prior)
    vars = @eval @variables begin
        mass = $m
        e ~ $e_prior
        i ~ Sine()
        ω ~ UniformCircular()
        Ω ~ UniformCircular()
        P ~ $P_prior                 # period [days]
        θ_x ~ Normal()
        θ_y ~ Normal()
        θ = atan(θ_y, θ_x)
        epoch = $ep                  # reference epoch for θ [MJD]
    end
    return Body(name=name, about=host, variables=vars)
end

"""
    build_obs(table, target, ref; name="obs_prior_example", free_systematics=false)

`RelAstromObs`. By default jitter = 0, northangle = 0, platescale = 1 (fixed,
as in the tutorial); `free_systematics=true` gives them the tutorial's
suggested priors instead.
"""
function build_obs(table, target, ref; name::AbstractString="obs_prior_example",
                   free_systematics::Bool=false)
    dbg("  obs '", name, "': ", length(table.epoch), " epochs, MJD ",
        minimum(table.epoch), "–", maximum(table.epoch),
        free_systematics ? ", free jitter/northangle/platescale" :
                           ", fixed jitter=0 northangle=0 platescale=1")
    vars = if free_systematics
        jd = Uniform(0, 10)
        nd = Normal(0, deg2rad(1))
        pd = truncated(Normal(1, 0.01), lower=0)
        @eval @variables begin
            jitter ~ $jd             # [mas]
            northangle ~ $nd         # [rad]
            platescale ~ $pd
        end
    else
        @eval @variables begin
            jitter = 0               # [mas]
            northangle = 0           # [rad]
            platescale = 1
        end
    end
    return RelAstromObs(table; target=target, ref=ref, name=name, variables=vars)
end

"""
    build_tutorial_system(; obs_prior=true, name, epoch=50420.0,
                          P_prior=LogUniform(35, 55_000), free_systematics=false,
                          plx_prior=truncated(Normal(50.0, 0.02), lower=0.1))

Build the full system. With `obs_prior=true` the observation is wrapped in
`ObsPriorONeil2019` and ONLY the wrapper is listed in `observations=`.
Each call builds fresh Body/observation objects so the two models share nothing.
"""
function build_tutorial_system(; obs_prior::Bool=true,
                                 name::AbstractString = obs_prior ? "TutoriaPrime" : "TutoriaUniform",
                                 epoch::Real=50420.0,
                                 P_prior::Distribution=LogUniform(35, 55_000),
                                 free_systematics::Bool=false,
                                 plx_prior::Distribution=truncated(Normal(50.0, 0.02), lower=0.1))
    A = build_star()
    b = build_planet(A; epoch=epoch, P_prior=P_prior)
    obs = build_obs(astrometry_table(), b, A; free_systematics=free_systematics)
    if obs_prior
        obs = ObsPriorONeil2019(obs)
        dbg("  wrapped observation in ObsPriorONeil2019 (orbit defaults to target = b)")
    end
    dbg("  system '", name, "': plx ~ ", plx_prior,
        obs_prior ? "  [observable priors]" : "  [uniform priors]")
    vars = @eval @variables begin
        plx ~ $plx_prior             # [mas]
    end
    return System(name=name, bodies=[A, b], observations=[obs], variables=vars)
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_obsprior(; iterations=5000, compare=true, free_systematics=false,
                   P_prior=LogUniform(35, 55_000), epoch=50420.0,
                   outdir=SCRIPT_DIR, prefix="obsprior_", dark=true,
                   corner_dark=false, show_plots=true)

Fit with observable priors (the tutorial), then — if `compare` — the same
model with uniform priors, and compare. Returns a NamedTuple of results.
"""
function run_obsprior(; iterations::Integer=5000,
                        compare::Bool=true,
                        free_systematics::Bool=false,
                        P_prior::Distribution=LogUniform(35, 55_000),
                        epoch::Real=50420.0,
                        outdir::AbstractString=SCRIPT_DIR,
                        prefix::AbstractString="obsprior_",
                        dark::Bool=true,
                        corner_dark::Bool=false,
                        show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    print_env_info(outdir)
    set_plot_theme!(; dark)

    outputs = String[]
    function savefig!(fig, name)
        p = joinpath(outdir, prefix * name)
        save(p, fig)
        push!(outputs, p)
        dbg("  saved ", p)
        show_plots && display(fig)
        return fig
    end
    inout(f) = cd(f, outdir)

    # ---- Observable-prior model (the tutorial) ------------------------------
    sys_op = stage("Build system with observable priors") do
        build_tutorial_system(; obs_prior=true, epoch, P_prior, free_systematics)
    end
    DEBUG[] && display(sys_op)

    model_op = stage("Compile LogDensityModel (observable priors)") do
        Octofitter.LogDensityModel(sys_op)
    end
    DEBUG[] && display(model_op)

    init_op = stage("Initialize starting points (no guesses; global optimisation)") do
        initialize!(model_op)
    end

    stage("Plot initial guess") do
        savefig!(inout(() -> octoplot(model_op, init_op)), "init.png")
    end

    chain_op = stage("Sample posterior, observable priors (HMC, $iterations iterations)") do
        octofit(model_op; iterations=iterations)
    end
    print_divergences(chain_op; label="obs-prior chain")
    print_posterior_summary(chain_op; label="observable priors")
    soft_stage("R-hat (observable priors)") do
        print_rhat(chain_op)
    end

    stage("Orbit plot (observable priors)") do
        savefig!(inout(() -> octoplot(model_op, chain_op)), "orbits_obsprior.png")
    end

    # ---- Uniform-prior baseline ---------------------------------------------
    model_u = nothing
    chain_u = nothing
    if compare
        model_u = soft_stage("Build + compile baseline with uniform priors") do
            s = build_tutorial_system(; obs_prior=false, epoch, P_prior, free_systematics)
            DEBUG[] && display(s)
            Octofitter.LogDensityModel(s)
        end
        if model_u !== nothing
            chain_u = soft_stage("Initialize + sample, uniform priors (HMC, $iterations iterations)") do
                initialize!(model_u)
                octofit(model_u; iterations=iterations)
            end
        end
        if chain_u !== nothing
            print_divergences(chain_u; label="uniform chain")
            print_posterior_summary(chain_u; label="uniform priors")

            soft_stage("Orbit plot (uniform priors)") do
                savefig!(inout(() -> octoplot(model_u, chain_u)), "orbits_uniform.png")
            end
            soft_stage("Corner plot: uniform vs observable priors") do
                savefig!(inout(() -> corner(model_op, chain_u, chain_op; dark=corner_dark)),
                         "corner_compare.png")
            end
            soft_stage("Prior-sensitivity histograms (P, a, e)") do
                savefig!(comparison_figure(chain_u, chain_op), "compare_hist.png")
            end
            soft_stage("Trace plot b_P (both fits)") do
                savefig!(trace_figure(chain_u, chain_op), "trace_P.png")
            end
            print_comparison(chain_u, chain_op)
        end
    else
        soft_stage("Corner plot (observable priors)") do
            savefig!(inout(() -> corner(model_op, chain_op; dark=corner_dark)), "corner.png")
        end
    end

    # ---- Saving ------------------------------------------------------------
    path_op = joinpath(outdir, prefix * "chain_obsprior.fits")
    stage("Save observable-prior chain -> FITS") do
        Octofitter.savechain(path_op, chain_op)
    end
    push!(outputs, path_op)
    soft_stage("Verify FITS reload (observable priors)") do
        c2 = Octofitter.loadchain(path_op; model=model_op)
        dbg("  reloaded chain size = ", size(c2),
            size(c2) == size(chain_op) ? "  (matches)" : "  (DIFFERS from saved)")
    end

    path_u = nothing
    if chain_u !== nothing
        path_u = joinpath(outdir, prefix * "chain_uniform.fits")
        soft_stage("Save uniform-prior chain -> FITS") do
            Octofitter.savechain(path_u, chain_u)
            push!(outputs, path_u)
            c2 = Octofitter.loadchain(path_u; model=model_u)
            dbg("  reloaded chain size = ", size(c2),
                size(c2) == size(chain_u) ? "  (matches)" : "  (DIFFERS from saved)")
        end
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f s", time() - T0[]))

    return (; system=sys_op, model=model_op, init_chain=init_op, chain=chain_op,
              model_uniform=model_u, chain_uniform=chain_u,
              outputs, chain_path=path_op, chain_path_uniform=path_u)
end

println(@sprintf("[OOP] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterObsPrior

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia OctofitterObsPrior.jl`               -> runs
#   - VSCode "Execute active File in REPL"         -> runs (interactive session)
#   - ENV["OCTOOP_AUTORUN"] = "false"; include()   -> loads module only
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOOP_AUTORUN", "true")) != "false"
    if as_script || autorun
        global obsprior_result = OctofitterObsPrior.run_obsprior(
            iterations=5000, compare=true)
        println("[OOP] Result stored in `obsprior_result` (fields: system, model, ",
                "init_chain, chain, model_uniform, chain_uniform, outputs, chain_path, ",
                "chain_path_uniform)")
    end
end

nothing
