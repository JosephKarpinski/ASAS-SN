#=
================================================================================
 OctofitterQuickStart.jl
================================================================================
 Module wrapping the Octofitter.jl v9 Quick Start example:
   "Fit a Single Planet Orbit to Relative Astrometry"
   https://sefffal.github.io/Octofitter.jl/dev/quick-start/

 Pipeline:  Body (star) -> Body (planet) -> RelAstromObs -> System
            -> LogDensityModel -> initialize! -> octofit (HMC)
            -> octoplot / octocorner -> savechain / loadchain

 Requires:  Octofitter (v9+), Distributions, CairoMakie, PairPlots

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (loads the module AND runs the fit; result in `quickstart_result`)
   Shell:    julia --project=. OctofitterQuickStart.jl
   Library:  ENV["OCTOQS_AUTORUN"] = "false"; include("OctofitterQuickStart.jl")
             res = OctofitterQuickStart.run_quickstart(iterations=500)

 Outputs are written to the directory containing this file (not pwd()).

 Environment: if the active environment lacks Octofitter v9, the bootstrap
 block at the top creates one automatically (temporary by default;
 ENV["OCTOQS_ENV_MODE"] = "local" keeps it in ./octofitter_v9_env).
 The session stays in that environment after the run.

 Changelog
 ---------
 v1.4.0 2026-09-24 First clean end-to-end run (Octofitter 9.0.0). Corner plot
                   now rendered with Makie's default light theme (PairPlots
                   draws single-series contours/titles in black, invisible on
                   #111111); `corner_dark=true` restores dark. Legend theme
                   uses `backgroundcolor` (fixes Makie `bgcolor` deprecation
                   warnings).
 v1.3.0 2026-09-24 Environment bootstrap before any `using`: reuses a loaded
                   v9 or an active env that has v9; otherwise activates a
                   temporary env (or ./octofitter_v9_env with
                   OCTOQS_ENV_MODE=local) and adds Octofitter@9 +
                   Distributions, CairoMakie, PairPlots. Clear error if v8
                   is already loaded in the session (needs REPL restart).
 v1.2.0 2026-09-24 Load-time guard: error with upgrade instructions if the
                   active environment has Octofitter < 9 (v8 lacks Body,
                   System(bodies=...), RelAstromObs(target=, ref=)). Env
                   banner also prints the active project path.
 v1.1.3 2026-09-24 Fix "invalid assignment to constant Main._OQS_LOAD_T0" when
                   re-running in a REPL that had loaded v1.1.2. No helper
                   globals in Main any more: load timer lives inside the
                   module (replaced on each run) and the auto-run guard is in
                   a `let` block; only `quickstart_result` is set in Main.
 v1.1.2 2026-09-24 Fix ParseError in auto-run guard: bare `@__FILE__ ||`
                   parses `||` as a macro argument. Macros now called with
                   explicit parens (`@__FILE__()`, `@__DIR__()`); guard split
                   into named booleans.
 v1.1.1 2026-09-24 Fix LoadError: `@sprintf` used in Main (outside the module,
                   where Printf isn't loaded) for the "Module loaded" line.
 v1.1  2026-09-24  Auto-run block after the module so VSCode "Execute active
                   File in REPL" (and `julia file.jl`) actually runs the fit;
                   OCTOQS_AUTORUN=false to load only. Timestamped debug output
                   ([OQS +t]) with per-stage timing, failure location, env
                   info, posterior summary and output-file listing; toggle
                   via OctofitterQuickStart.DEBUG[]. Default outdir is the
                   script's own directory (@__DIR__); plotting runs inside
                   cd(outdir) so any auto-saved Octofitter files land there
                   too. Figures displayed in the VSCode plot pane.
 v1.0  2026-09-24  Initial version. Quick-start steps split into standalone
                   functions; priors/epoch configurable via keywords; dark
                   (#111111 / #EEEEEE) Makie theme with light-theme toggle;
                   plots + FITS chain written to an output directory;
                   MJD date helpers re-exported.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
#
# Ensures Octofitter v9 is what gets loaded:
#   1. Octofitter already loaded in this session?
#        v9+ -> reuse it;  v8 -> error (packages can't be unloaded; restart REPL)
#   2. Active environment already has Octofitter v9 + deps -> use it as-is
#   3. Otherwise create/activate a v9 environment and add the packages:
#        OCTOQS_ENV_MODE = "temp"  (default) throwaway env via Pkg.activate(temp=true)
#        OCTOQS_ENV_MODE = "local" persistent env in ./octofitter_v9_env next to
#                                  this file (faster reruns; delete folder to remove)
# Package downloads and precompile caches live in ~/.julia and are reused, so
# only the first bootstrap is slow.
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "Distributions", "CairoMakie", "PairPlots"),
    mode     = lowercase(get(ENV, "OCTOQS_ENV_MODE", "temp"))

    # Direct deps of the active environment -> Dict(name => version)
    direct_deps() = Dict(info.name => info.version
                         for info in values(Pkg.dependencies()) if info.is_direct_dep)
    function env_ok()
        d = direct_deps()
        v = get(d, "Octofitter", nothing)
        v !== nothing && v >= v"9" && all(n -> haskey(d, n), required)
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
        println("[OQS] Octofitter v$v already loaded; reusing session environment ",
                Base.active_project())
    elseif env_ok()
        println("[OQS] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], ": ", Base.active_project())
    else
        println("[OQS] Active environment ", Base.active_project(),
                " lacks Octofitter v9; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[OQS] Existing v9 environment found at ", Base.active_project())
        else
            Pkg.add([Pkg.PackageSpec(name="Octofitter", version="9"),
                     Pkg.PackageSpec(name="Distributions"),
                     Pkg.PackageSpec(name="CairoMakie"),
                     Pkg.PackageSpec(name="PairPlots")])
        end
        println("[OQS] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[OQS] Loading Octofitter, Distributions, CairoMakie, PairPlots ",
        "(first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterQuickStart

const LOAD_T0 = time()

using Octofitter
using Distributions
using CairoMakie
using PairPlots
using Printf
import Statistics

export build_star, build_planet, example_astrometry_table, build_astrometry,
       build_system, compile_model, initialize_chain, sample_posterior,
       plot_results, save_chain, load_chain, set_plot_theme!, run_quickstart,
       mjd, years2mjd, mjd2date

const VERSION_STRING = "1.4.0"

# This module uses the Octofitter v9 API (Body / System(bodies=...) /
# RelAstromObs(target=, ref=)). v8 has none of these, so fail fast.
const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterQuickStart needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up Octofitter v9.
    """)
end

# Directory of this source file; falls back to pwd() for unsaved buffers.
const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterQuickStart.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[OQS +%7.1fs] ", time() - T0[]), msg...)
    flush(stdout)
    return nothing
end

"""
    stage(f, name)

Run `f()` with start/finish/timing debug lines; on error, report which stage
failed and rethrow.
"""
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

function print_env_info(outdir)
    DEBUG[] || return nothing
    octo_ver = try string(pkgversion(Octofitter)) catch; "unknown" end
    dbg("OctofitterQuickStart v", VERSION_STRING,
        " | Julia ", VERSION, " | Octofitter ", octo_ver,
        " | threads ", Threads.nthreads())
    dbg("project    = ", Base.active_project())
    dbg("pwd()      = ", pwd())
    dbg("SCRIPT_DIR = ", SCRIPT_DIR)
    dbg("outdir     = ", abspath(outdir))
    dbg("interactive = ", isinteractive(),
        " | Makie backend = ", CairoMakie.Makie.current_backend())
    return nothing
end

function print_posterior_summary(chain;
        params=(:plx, :A_mass, :b_a, :b_e, :b_i, :b_ω, :b_Ω, :b_θ))
    DEBUG[] || return nothing
    dbg("Chain size (iter × vars × chains) = ", size(chain))
    try
        acc = Statistics.mean(vec(chain[:acceptance_rate]))
        dbg(@sprintf("Mean acceptance rate = %.3f", acc))
    catch
        dbg("(acceptance_rate not found in chain)")
    end
    try
        nerr = sum(vec(chain[:numerical_error]))
        dbg("Numerical errors (divergences) = ", Int(nerr))
    catch
    end
    dbg("Posterior median [16th, 84th]:")
    for p in params
        try
            x = vec(chain[p])
            q = Statistics.quantile(x, (0.16, 0.5, 0.84))
            dbg(@sprintf("    %-8s %10.4f  [%10.4f, %10.4f]", p, q[2], q[1], q[3]))
        catch
            dbg(@sprintf("    %-8s (not in chain)", p))
        end
    end
    return nothing
end

function list_outputs(paths)
    DEBUG[] || return nothing
    dbg("Output files:")
    for p in paths
        if isfile(p)
            dbg(@sprintf("    %8.1f kB  %s", filesize(p) / 1024, p))
        else
            dbg("    MISSING    ", p)
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Plot theme
# -----------------------------------------------------------------------------
"""
    set_plot_theme!(; dark=true)

Dark publication theme (#111111 background, #EEEEEE text) or Makie's default
light theme when `dark=false`.
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

# -----------------------------------------------------------------------------
# Model components
#
# Priors and fixed values are interpolated into the @variables block with @eval
# so the block sees concrete values rather than function-local bindings (the
# macro may compile derived `=` expressions outside the caller's local scope).
# -----------------------------------------------------------------------------
"""
    build_star(; name="A", mass_prior=truncated(Normal(1.2, 0.1), lower=0.1))

Host star as a root `Body`. Mass in M⊙.
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
    build_planet(host; name="b", epoch=50000.0, mass=0.0,
                 a_prior=Uniform(0, 100), e_prior=Uniform(0.0, 0.5))

Planet `Body` orbiting `host`. Orbit phase is set by position angle `θ` at
reference `epoch` [MJD] — set `epoch` to your most constraining data epoch.
Planet mass is fixed (default 0) since relative astrometry rarely constrains it.
"""
function build_planet(host; name::AbstractString="b",
                            epoch::Real=50000.0,
                            mass::Real=0.0,
                            a_prior::Distribution=Uniform(0, 100),
                            e_prior::Distribution=Uniform(0.0, 0.5))
    ep = Float64(epoch)
    m  = Float64(mass)
    dbg("  planet '", name, "': epoch = ", ep, " MJD (", mjd2date(ep), "), mass = ", m,
        ", a ~ ", a_prior, ", e ~ ", e_prior)
    vars = @eval @variables begin
        mass = $m                    # [M⊙]
        a ~ $a_prior                 # Semi-major axis [AU]
        e ~ $e_prior                 # Eccentricity
        i ~ Sine()                   # Inclination [rad]
        ω ~ UniformCircular()        # Argument of periastron [rad]
        Ω ~ UniformCircular()        # Longitude of ascending node [rad]
        θ ~ UniformCircular()        # Position angle at reference epoch [rad]
        epoch = $ep                  # Reference epoch for θ [MJD]
    end
    return Body(name=name, about=host, variables=vars)
end

"""
    example_astrometry_table()

The three-epoch relative astrometry from the quick-start (mas, MJD).
"""
function example_astrometry_table()
    return Table(
        epoch = [50000.0, 50120.0, 50240.0],  # [MJD]
        ra    = [-505.7, -502.5, -498.2],     # [mas] East positive
        dec   = [-66.9, -37.4, -7.9],         # [mas] North positive
        σ_ra  = [10.0, 10.0, 10.0],           # [mas]
        σ_dec = [10.0, 10.0, 10.0],           # [mas]
        cor   = [0.0, 0.0, 0.0],              # RA/Dec correlation
    )
end

"""
    build_astrometry(table, target, ref; name="GPI astrom")

`RelAstromObs` of `target` measured relative to `ref`. `table` needs columns
epoch, ra, dec, σ_ra, σ_dec, cor.
"""
function build_astrometry(table, target, ref; name::AbstractString="GPI astrom")
    dbg("  astrometry '", name, "': ", length(table.epoch), " epochs, MJD ",
        minimum(table.epoch), "–", maximum(table.epoch))
    return RelAstromObs(table; target=target, ref=ref, name=name)
end

"""
    build_system(bodies, observations; name="HD1234",
                 plx_prior=truncated(Normal(50.0, 0.02), lower=0.1))

Assemble a `System`. Parallax [mas] is required for relative astrometry.
"""
function build_system(bodies::AbstractVector, observations::AbstractVector;
                      name::AbstractString="HD1234",
                      plx_prior::Distribution=truncated(Normal(50.0, 0.02), lower=0.1))
    dbg("  system '", name, "': ", length(bodies), " bodies, ",
        length(observations), " observations, plx ~ ", plx_prior)
    vars = @eval @variables begin
        plx ~ $plx_prior             # Parallax [mas]
    end
    return System(name=name, bodies=bodies, observations=observations, variables=vars)
end

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
"""
    compile_model(sys)

Compile a `System` into an `Octofitter.LogDensityModel`.
"""
compile_model(sys) = Octofitter.LogDensityModel(sys)

"""
    initialize_chain(model, start=nothing)

Find starting points (variational approximation). `start` is an optional
NamedTuple, e.g. `(; plx=50.001, bodies=(; A=(; mass=1.18), b=(; a=10.0)))`.
"""
initialize_chain(model, start=nothing) =
    start === nothing ? initialize!(model) : initialize!(model, start)

"""
    sample_posterior(model; iterations=1000, kwargs...)

HMC sampling via `octofit`; extra keywords are forwarded.
"""
sample_posterior(model; iterations::Integer=1000, kwargs...) =
    octofit(model; iterations=iterations, kwargs...)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
"""
    plot_results(model, chain; outdir=SCRIPT_DIR, prefix="quickstart",
                 small_corner=true, corner_dark=false, show_plots=true)

Orbit plot (`octoplot`) and corner plot (`octocorner`), saved as PNG in
`outdir`. The corner plot uses Makie's default light theme unless
`corner_dark=true`: PairPlots draws a single series (histograms, contours,
value titles) in black, which is unreadable on a dark background. Plotting runs inside `cd(outdir)` so any files Octofitter saves on
its own also land there. Returns `(orbit=fig, corner=fig, paths=[...])`.
"""
function plot_results(model, chain; outdir::AbstractString=SCRIPT_DIR,
                      prefix::AbstractString="quickstart", small_corner::Bool=true,
                      corner_dark::Bool=false, show_plots::Bool=true)
    outdir = abspath(outdir)
    mkpath(outdir)
    p_orbit  = joinpath(outdir, "$(prefix)_orbits.png")
    p_corner = joinpath(outdir, "$(prefix)_corner.png")
    fig_orbit, fig_corner = cd(outdir) do
        fo = octoplot(model, chain)
        fc = corner_dark ? octocorner(model, chain, small=small_corner) :
             with_theme(() -> octocorner(model, chain, small=small_corner), Theme())
        save(p_orbit, fo)
        save(p_corner, fc)
        fo, fc
    end
    dbg("  saved ", p_orbit)
    dbg("  saved ", p_corner, corner_dark ? " (dark theme)" : " (light theme)")
    if show_plots
        display(fig_orbit)
        display(fig_corner)
    end
    return (orbit=fig_orbit, corner=fig_corner, paths=[p_orbit, p_corner])
end

"""
    save_chain(path, chain)

Write chain to FITS.
"""
save_chain(path::AbstractString, chain) = Octofitter.savechain(path, chain)

"""
    load_chain(path; model=nothing)

Read chain from FITS. Passing `model` validates columns against the model
instead of silently returning `missing` for mismatches.
"""
load_chain(path::AbstractString; model=nothing) =
    model === nothing ? Octofitter.loadchain(path) : Octofitter.loadchain(path; model)

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_quickstart(; iterations=1000, outdir=SCRIPT_DIR, dark=true,
                     astrom_table=example_astrometry_table(),
                     epoch=first(astrom_table.epoch), plot_init=true,
                     show_plots=true, corner_dark=false, verify_reload=true)

Run the full quick-start end to end. Returns a NamedTuple with the system,
model, initial chain, posterior chain, figures and output paths.
"""
function run_quickstart(; iterations::Integer=1000,
                          outdir::AbstractString=SCRIPT_DIR,
                          dark::Bool=true,
                          astrom_table=example_astrometry_table(),
                          epoch::Real=first(astrom_table.epoch),
                          plot_init::Bool=true,
                          show_plots::Bool=true,
                          corner_dark::Bool=false,
                          verify_reload::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    print_env_info(outdir)
    set_plot_theme!(; dark)

    sys = stage("Build bodies, observations, system") do
        A      = build_star()
        b      = build_planet(A; epoch=epoch)
        astrom = build_astrometry(astrom_table, b, A)
        build_system([A, b], [astrom])
    end
    DEBUG[] && display(sys)

    model = stage("Compile LogDensityModel") do
        compile_model(sys)
    end
    DEBUG[] && display(model)

    init_chain = stage("Initialize starting points") do
        initialize_chain(model, (;
            plx = 50.001,
            bodies = (;
                A = (; mass = 1.18),
                b = (; a = 10.0, e = 0.01),
            ),
        ))
    end

    outputs = String[]

    # Sanity-check the data entry against the starting-point orbits
    fig_init = nothing
    if plot_init
        fig_init = stage("Plot initial guess") do
            p = joinpath(outdir, "quickstart_init.png")
            f = cd(() -> octoplot(model, init_chain), outdir)
            save(p, f)
            push!(outputs, p)
            show_plots && display(f)
            f
        end
    end

    chain = stage("Sample posterior (HMC, $iterations iterations)") do
        sample_posterior(model; iterations=iterations)
    end
    print_posterior_summary(chain)

    figs = stage("Plot posterior (octoplot + octocorner)") do
        plot_results(model, chain; outdir=outdir, show_plots=show_plots,
                     corner_dark=corner_dark)
    end
    append!(outputs, figs.paths)

    chain_path = joinpath(outdir, "quickstart_chain.fits")
    stage("Save chain -> FITS") do
        save_chain(chain_path, chain)
    end
    push!(outputs, chain_path)

    if verify_reload
        stage("Verify FITS reload against model") do
            c2 = load_chain(chain_path; model)
            dbg("  reloaded chain size = ", size(c2),
                size(c2) == size(chain) ? "  (matches)" : "  (DIFFERS from original)")
        end
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f s", time() - T0[]))

    return (; system=sys, model, init_chain, chain,
              figures=(init=fig_init, orbit=figs.orbit, corner=figs.corner),
              outputs, chain_path)
end

println(@sprintf("[OQS] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterQuickStart

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia OctofitterQuickStart.jl`            -> runs
#   - VSCode "Execute active File in REPL"        -> runs (interactive session)
#   - ENV["OCTOQS_AUTORUN"] = "false"; include()  -> loads module only
# Calls are module-qualified (no `using`) so re-executing the file after edits
# doesn't trigger binding-conflict warnings when the module is replaced.
# -----------------------------------------------------------------------------
let this_file   = @__FILE__(),
    as_script   = abspath(PROGRAM_FILE) == this_file,
    autorun     = isinteractive() &&
                  lowercase(get(ENV, "OCTOQS_AUTORUN", "true")) != "false"
    if as_script || autorun
        global quickstart_result = OctofitterQuickStart.run_quickstart(iterations=1000)
        println("[OQS] Result stored in `quickstart_result` ",
                "(fields: system, model, init_chain, chain, figures, outputs, chain_path)")
    end
end

nothing
