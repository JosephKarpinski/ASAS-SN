#=
================================================================================
 OctofitterRelAstrom.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Basic Astrometry Fit":
   https://sefffal.github.io/Octofitter.jl/dev/rel-astrom/

 Pipeline:
   Body A (star) + Body b (planet, θ at epoch)
   -> RelAstromObs #1 (ra/dec, 8 epochs) + RelAstromObs #2 (sep/pa, 1 epoch),
      each optionally with jitter / northangle / platescale systematics
   -> System -> LogDensityModel -> initialize! -> octofit (HMC)
   -> diagnostics: divergences, trace plot, autocorrelation, R-hat,
      extra chains + chainscat + Gelman-Rubin
   -> octoplot (orbits), octocorner (posterior; posterior vs initial guess),
      optional predicted RV / proper-motion panels
   -> predicted planet offsets on future dates (construct_system/orbitsolve)
   -> savechain/loadchain (FITS), serialize/deserialize model

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (sets up env if needed, loads module, runs; result in
             `relastrom_result`)
   Shell:    julia OctofitterRelAstrom.jl
   Library:  ENV["OCTORA_AUTORUN"] = "false"; include("OctofitterRelAstrom.jl")
             res = OctofitterRelAstrom.run_relastrom(iterations=500, n_chains=2)

 Environment: if the active environment lacks Octofitter v9 (+ Distributions,
 CairoMakie, PairPlots, StatsBase, MCMCChains), the bootstrap block below
 creates one (temporary by default; ENV["OCTORA_ENV_MODE"] = "local" keeps it
 in ./octofitter_v9_env, shared with OctofitterQuickStart.jl). If Octofitter
 v9 is already loaded (e.g. after running OctofitterQuickStart in the same
 REPL), missing packages are added to the active env without changing loaded
 versions. If v8 is loaded, restart the REPL.

 Outputs (prefix "relastrom_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-24  Initial version, built on the OctofitterQuickStart v1.4.0
                   scaffolding (env bootstrap, v9 guard, [ORA +t] debug
                   stages, @__DIR__ outputs, dark theme with light corner
                   plots, auto-run guard in a `let`). Adds: two RelAstromObs
                   formats (ra/dec and sep/pa) with optional instrument
                   systematics; trace + autocorrelation plots; R-hat table;
                   multi-chain chainscat + gelmandiag; optional predicted
                   radvel/pmra panels; posterior predicted offsets on future
                   dates; model serialize/deserialize round-trip; posterior
                   vs initial-guess corner plot. Non-essential steps run as
                   "soft" stages that log a failure and continue.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "Distributions", "CairoMakie", "PairPlots",
                "StatsBase", "MCMCChains"),
    mode     = lowercase(get(ENV, "OCTORA_ENV_MODE", "temp"))

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
            println("[ORA] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[ORA] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(miss; preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[ORA] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[ORA] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[ORA] Existing v9 environment found at ", Base.active_project())
        else
            specs = [n == "Octofitter" ? Pkg.PackageSpec(name=n, version="9") :
                                         Pkg.PackageSpec(name=n) for n in required]
            Pkg.add(specs)
        end
        println("[ORA] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[ORA] Loading Octofitter, Distributions, CairoMakie, PairPlots, StatsBase, ",
        "MCMCChains (first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterRelAstrom

const LOAD_T0 = time()

using Octofitter
using Distributions
using CairoMakie
using PairPlots
import StatsBase
import MCMCChains
using Serialization
using Printf
import Statistics

export run_relastrom, build_star, build_planet, astrometry_table_radec,
       astrometry_table_seppa, build_rel_astrom_obs, build_system,
       predict_offsets, set_plot_theme!

const VERSION_STRING = "1.0"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterRelAstrom needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up Octofitter v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterRelAstrom.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[ORA +%7.1fs] ", time() - T0[]), msg...)
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
    dbg("OctofitterRelAstrom v", VERSION_STRING, " | Julia ", VERSION,
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

"Column as a plain Vector (all chains stacked)."
colvec(chain, p::Symbol) = vec(Array(chain[p]))
"Column as an iterations × chains Matrix."
colmat(chain, p::Symbol) = reshape(Array(chain[p]), size(chain, 1), :)
haspar(chain, p::Symbol) = p in names(chain)

function print_divergences(chain; label="chain")
    DEBUG[] || return nothing
    try
        m = colmat(chain, :numerical_error)
        for c in axes(m, 2)
            n = Int(sum(m[:, c]))
            dbg(@sprintf("  %s[%d]: divergences %d / %d (%.1f%%)", label, c, n,
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

function print_posterior_summary(chain, params)
    DEBUG[] || return nothing
    dbg("Posterior median [16th, 84th]  (", size(chain, 1), " iter × ",
        size(chain, 3), " chains):")
    for p in params
        haspar(chain, p) || continue
        q = Statistics.quantile(colvec(chain, p), (0.16, 0.5, 0.84))
        dbg(@sprintf("    %-26s %11.4f  [%11.4f, %11.4f]", p, q[2], q[1], q[3]))
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
# Plot theme
# -----------------------------------------------------------------------------
"""
    set_plot_theme!(; dark=true)

Dark publication theme (#111111 background, #EEEEEE text) or Makie's default
light theme when `dark=false`. Corner plots are always drawn light unless
`corner_dark=true` (PairPlots draws single series in black).
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

# -----------------------------------------------------------------------------
# Model components (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
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
                 a_prior=Uniform(0, 100), e_prior=Uniform(0.0, 0.5))

Planet orbiting `host`; phase set by θ at `epoch` [MJD]. Mass fixed (default 0).
"""
function build_planet(host; name::AbstractString="b",
                            epoch::Real=50420.0,
                            mass::Real=0.0,
                            a_prior::Distribution=Uniform(0, 100),
                            e_prior::Distribution=Uniform(0.0, 0.5))
    ep = Float64(epoch)
    m  = Float64(mass)
    dbg("  planet '", name, "': epoch = ", ep, " MJD (", mjd2date(ep), "), mass = ", m,
        ", a ~ ", a_prior, ", e ~ ", e_prior)
    vars = @eval @variables begin
        mass = $m                    # [M⊙]
        a ~ $a_prior                 # [AU]
        e ~ $e_prior
        i ~ Sine()                   # [rad]
        ω ~ UniformCircular()        # [rad]
        Ω ~ UniformCircular()        # [rad]
        θ ~ UniformCircular()        # [rad] position angle at `epoch`
        epoch = $ep                  # [MJD]
    end
    return Body(name=name, about=host, variables=vars)
end

"Tutorial dataset #1: 8 epochs of ra/dec offsets [mas]."
function astrometry_table_radec()
    return Table(;
        epoch = Float64[50000, 50120, 50240, 50360, 50480, 50600, 50720, 50840], # MJD
        ra    = [-505.764, -502.57, -498.209, -492.678, -485.977, -478.11, -469.08, -458.896],
        dec   = [-66.9298, -37.4722, -7.92755, 21.6356, 51.1472, 80.5359, 109.729, 138.651],
        σ_ra  = fill(10.0, 8),
        σ_dec = fill(10.0, 8),
        cor   = fill(0.0, 8),
    )
end

"Tutorial dataset #2: one epoch of separation [mas] / position angle [rad]."
function astrometry_table_seppa()
    return Table(
        epoch = [42000.0],                # MJD
        sep   = [505.7637580573554],      # mas
        pa    = [deg2rad(24.1)],          # rad
        σ_sep = [70.0],
        σ_pa  = [deg2rad(10.2)],
    )
end

"""
    build_rel_astrom_obs(table, target, ref; name, systematics=true,
                         jitter_max=10.0, northangle_sigma_deg=1.0,
                         platescale_sigma=0.01)

`RelAstromObs` of `target` relative to `ref`. With `systematics=true`, adds
per-instrument `jitter ~ Uniform(0, jitter_max)` [mas], `northangle ~
Normal(0, σ)` [rad] and `platescale ~ truncated(Normal(1, σ_ps), lower=0)`.
"""
function build_rel_astrom_obs(table, target, ref; name::AbstractString,
                              systematics::Bool=true,
                              jitter_max::Real=10.0,
                              northangle_sigma_deg::Real=1.0,
                              platescale_sigma::Real=0.01)
    fmt = hasproperty(table, :ra) ? "ra/dec" : "sep/pa"
    dbg("  obs '", name, "': ", length(table.epoch), " epoch(s), ", fmt, ", MJD ",
        minimum(table.epoch), "–", maximum(table.epoch),
        systematics ? " + jitter/northangle/platescale" : "")
    if systematics
        jd = Uniform(0, Float64(jitter_max))
        nd = Normal(0, deg2rad(Float64(northangle_sigma_deg)))
        pd = truncated(Normal(1, Float64(platescale_sigma)), lower=0)
        vars = @eval @variables begin
            jitter ~ $jd             # [mas]
            northangle ~ $nd         # [rad]
            platescale ~ $pd         # relative
        end
        return RelAstromObs(table; target=target, ref=ref, name=name, variables=vars)
    else
        return RelAstromObs(table; target=target, ref=ref, name=name)
    end
end

"""
    build_system(bodies, observations; name="Tutoria",
                 plx_prior=truncated(Normal(50.0, 0.02), lower=0.1))
"""
function build_system(bodies::AbstractVector, observations::AbstractVector;
                      name::AbstractString="Tutoria",
                      plx_prior::Distribution=truncated(Normal(50.0, 0.02), lower=0.1))
    dbg("  system '", name, "': ", length(bodies), " bodies, ",
        length(observations), " observations, plx ~ ", plx_prior)
    vars = @eval @variables begin
        plx ~ $plx_prior             # [mas]
    end
    return System(name=name, bodies=bodies, observations=observations, variables=vars)
end

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
"Trace plot of `param`, one line per chain."
function trace_figure(chain, param::Symbol; ylabel=string(param))
    m = colmat(chain, param)
    fig = Figure(size=(900, 350))
    ax = Axis(fig[1, 1], xlabel="iteration", ylabel=ylabel, title="Trace: $param")
    for c in axes(m, 2)
        lines!(ax, m[:, c], linewidth=0.8, label="chain $c")
    end
    size(m, 2) > 1 && axislegend(ax, position=:rt)
    return fig
end

"Autocorrelation of `param` vs lag, one line per chain; returns (fig, lag where |ρ|<0.05)."
function autocor_figure(chain, param::Symbol; maxlag::Integer=500)
    m = colmat(chain, param)
    lags = 1:min(maxlag, size(m, 1) - 1)
    fig = Figure(size=(700, 350))
    ax = Axis(fig[1, 1], xlabel="lag", ylabel="autocorrelation",
              title="Autocorrelation: $param")
    decorr = Int[]
    for c in axes(m, 2)
        ρ = StatsBase.autocor(m[:, c], lags)
        lines!(ax, collect(lags), ρ, label="chain $c")
        k = findfirst(x -> abs(x) < 0.05, ρ)
        push!(decorr, k === nothing ? -1 : lags[k])
    end
    hlines!(ax, [0.0], color=(:gray, 0.6), linestyle=:dash)
    size(m, 2) > 1 && axislegend(ax, position=:rt)
    return fig, decorr
end

"R-hat table (MCMCChains.summarystats) and flag parameters with |R̂-1| > tol."
function print_rhat(chain; tol=0.01)
    ss = MCMCChains.summarystats(chain)
    DEBUG[] && display(ss)
    try
        ps = ss[:, :parameters]
        rh = ss[:, :rhat]
        bad = [(p, r) for (p, r) in zip(ps, rh) if isfinite(r) && abs(r - 1) > tol]
        if isempty(bad)
            dbg(@sprintf("  All finite R̂ within 1 ± %.2f", tol))
        else
            dbg(@sprintf("  %d parameter(s) with |R̂-1| > %.2f:", length(bad), tol))
            for (p, r) in bad
                dbg(@sprintf("    %-26s R̂ = %.4f", p, r))
            end
        end
    catch
        dbg("  (couldn't extract the rhat column; see table above)")
    end
    return ss
end

# -----------------------------------------------------------------------------
# Predictions
# -----------------------------------------------------------------------------
"""
    predict_offsets(model, chain; dates=("2025-01-01", "2030-01-01"),
                    ndraws=200, target=:b, ref=:A)

Predicted ra/dec offsets [mas] of `target` relative to `ref` on `dates`,
using `construct_system` + `orbitsolve` for up to `ndraws` posterior draws.
Prints draw #1 (as in the tutorial) and median [16th, 84th].
"""
function predict_offsets(model, chain; dates=("2025-01-01", "2030-01-01"),
                         ndraws::Integer=200, target::Symbol=:b, ref::Symbol=:A)
    n = size(chain, 1)
    idx = unique(round.(Int, range(1, n, length=min(ndraws, n))))
    epochs = [mjd(d) for d in dates]
    ra  = zeros(length(idx), length(epochs))
    dec = similar(ra)
    for (j, k) in enumerate(idx)
        posys = construct_system(model, chain, k)
        traj = orbitsolve(posys, epochs)
        for t in eachindex(epochs)
            ra[j, t]  = raoff(traj[t], target, ref)
            dec[j, t] = decoff(traj[t], target, ref)
        end
    end
    dbg("  Predicted $(target) vs $(ref) offsets [mas] from $(length(idx)) draws:")
    for t in eachindex(epochs)
        qr = Statistics.quantile(ra[:, t], (0.16, 0.5, 0.84))
        qd = Statistics.quantile(dec[:, t], (0.16, 0.5, 0.84))
        dbg(@sprintf("    %s  draw#1 (%8.2f, %8.2f)   Δα* %8.2f [%8.2f, %8.2f]   Δδ %8.2f [%8.2f, %8.2f]",
                     dates[t], ra[1, t], dec[1, t], qr[2], qr[1], qr[3], qd[2], qd[1], qd[3]))
    end
    return (; dates, epochs, draws=idx, ra, dec)
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_relastrom(; iterations=1000, n_chains=3, systematics=true,
                    outdir=SCRIPT_DIR, prefix="relastrom_", dark=true,
                    corner_dark=false, show_plots=true, epoch=50420.0,
                    plot_predictions=false, predict_dates=("2025-01-01", "2030-01-01"),
                    save_model=true)

Run the Basic Astrometry Fit tutorial end to end. `n_chains` counts the
first chain; extra chains are merged with `chainscat` for Gelman-Rubin and
the final plots. `plot_predictions` adds the predicted radvel/pmra panels
(flat unless planet mass is non-zero). Returns a NamedTuple of results.
"""
function run_relastrom(; iterations::Integer=1000,
                         n_chains::Integer=3,
                         systematics::Bool=true,
                         outdir::AbstractString=SCRIPT_DIR,
                         prefix::AbstractString="relastrom_",
                         dark::Bool=true,
                         corner_dark::Bool=false,
                         show_plots::Bool=true,
                         epoch::Real=50420.0,
                         plot_predictions::Bool=false,
                         predict_dates=("2025-01-01", "2030-01-01"),
                         save_model::Bool=true)
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
    # octoplot/octocorner run inside outdir so any auto-saved files land there
    inout(f) = cd(f, outdir)

    # ---- Model -------------------------------------------------------------
    names_obs = systematics ? ("GPI astrom", "SPHERE astrom") : ("relastrom", "relastrom2")

    sys = stage("Build bodies, observations, system") do
        A = build_star()
        b = build_planet(A; epoch=epoch)
        o1 = build_rel_astrom_obs(astrometry_table_radec(), b, A;
                                  name=names_obs[1], systematics=systematics)
        o2 = build_rel_astrom_obs(astrometry_table_seppa(), b, A;
                                  name=names_obs[2], systematics=systematics)
        build_system([A, b], [o1, o2])
    end
    DEBUG[] && display(sys)

    model = stage("Compile LogDensityModel") do
        Octofitter.LogDensityModel(sys)
    end
    DEBUG[] && display(model)

    init_chain = stage("Initialize starting points") do
        initialize!(model, (;
            plx = 50,
            bodies = (;
                A = (; mass = 1.21),
                b = (; a = 10.0, e = 0.01),
            ),
        ))
    end

    stage("Plot initial guess") do
        savefig!(inout(() -> octoplot(model, init_chain)), "init.png")
    end

    # ---- Sampling ----------------------------------------------------------
    chain = stage("Sample posterior, chain 1 (HMC, $iterations iterations)") do
        octofit(model; iterations=iterations)
    end
    print_divergences(chain; label="chain")

    # Parameters to summarise: core orbit + any instrument systematics
    obs_prefixes = [replace(n, " " => "_") for n in names_obs]
    params = Symbol[:plx, :A_mass, :b_a, :b_e, :b_i, :b_ω, :b_Ω, :b_θ]
    for op in obs_prefixes, v in ("jitter", "northangle", "platescale")
        push!(params, Symbol(op, "_", v))
    end
    print_posterior_summary(chain, params)

    # ---- Diagnostics -------------------------------------------------------
    soft_stage("Trace plot (b_a)") do
        savefig!(trace_figure(chain, :b_a; ylabel="semi-major axis (AU)"), "trace_b_a.png")
    end

    soft_stage("Autocorrelation plot (b_e)") do
        fig, decorr = autocor_figure(chain, :b_e)
        dbg("  |ρ| < 0.05 first reached at lag ", join(decorr, ", "),
            " (per chain; -1 = never within range)")
        savefig!(fig, "autocor_b_e.png")
    end

    soft_stage("R-hat (chain 1)") do
        print_rhat(chain)
    end

    extra = Any[]
    for k in 2:n_chains
        c = stage("Sample posterior, chain $k (HMC, $iterations iterations)") do
            octofit(model; iterations=iterations)
        end
        push!(extra, c)
    end

    merged = if isempty(extra)
        chain
    else
        stage("Merge $(n_chains) chains (chainscat)") do
            MCMCChains.chainscat(chain, extra...)
        end
    end
    if n_chains > 1
        print_divergences(merged; label="merged")
        soft_stage("Gelman-Rubin diagnostic") do
            g = MCMCChains.gelmandiag(merged)
            DEBUG[] && display(g)
            g
        end
        soft_stage("Trace plot, all chains (b_a)") do
            savefig!(trace_figure(merged, :b_a; ylabel="semi-major axis (AU)"),
                     "trace_b_a_all.png")
        end
        print_posterior_summary(merged, params)
    end

    # ---- Analysis plots ----------------------------------------------------
    stage("Orbit plot (octoplot, merged chains)") do
        savefig!(inout(() -> octoplot(model, merged)), "orbits.png")
    end

    if plot_predictions
        soft_stage("Predicted radial velocity panel") do
            savefig!(inout(() -> octoplot(model, merged; channels=radvel, show_sky=false)),
                     "pred_radvel.png")
        end
        soft_stage("Predicted proper-motion panel") do
            savefig!(inout(() -> octoplot(model, merged; channels=pmra, show_sky=false)),
                     "pred_pmra.png")
        end
    end

    stage("Corner plot (octocorner, merged chains)") do
        savefig!(inout(() -> corner(model, merged; dark=corner_dark)), "corner.png")
    end

    soft_stage("Corner plot: posterior vs initial guess") do
        savefig!(inout(() -> corner(model, chain, init_chain; dark=corner_dark)),
                 "corner_vs_init.png")
    end

    predictions = soft_stage("Predict future offsets (construct_system/orbitsolve)") do
        predict_offsets(model, chain; dates=predict_dates)
    end

    # ---- Saving ------------------------------------------------------------
    chain_path = joinpath(outdir, prefix * "chain.fits")
    stage("Save merged chain -> FITS") do
        Octofitter.savechain(chain_path, merged)
    end
    push!(outputs, chain_path)

    soft_stage("Verify FITS reload against model") do
        c2 = Octofitter.loadchain(chain_path; model)
        dbg("  reloaded chain size = ", size(c2),
            size(c2) == size(merged) ? "  (matches)" : "  (DIFFERS from saved)")
    end

    model_path = joinpath(outdir, prefix * "model.jls")
    if save_model
        soft_stage("Serialize model -> .jls (same machine/versions only)") do
            serialize(model_path, model)
            push!(outputs, model_path)
            m2 = deserialize(model_path)
            dbg("  deserialized: ", typeof(m2).name.name,
                " (dimension ", try m2.D catch; "?" end, ")")
        end
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f s", time() - T0[]))

    return (; system=sys, model, init_chain, chain, extra_chains=extra, merged,
              predictions, outputs, chain_path)
end

println(@sprintf("[ORA] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterRelAstrom

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia OctofitterRelAstrom.jl`              -> runs
#   - VSCode "Execute active File in REPL"         -> runs (interactive session)
#   - ENV["OCTORA_AUTORUN"] = "false"; include()   -> loads module only
# Module-qualified call (no `using`) and no helper globals in Main, so
# re-executing the file in the same REPL is clean.
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTORA_AUTORUN", "true")) != "false"
    if as_script || autorun
        global relastrom_result = OctofitterRelAstrom.run_relastrom(
            iterations=1000, n_chains=3, systematics=true)
        println("[ORA] Result stored in `relastrom_result` (fields: system, model, ",
                "init_chain, chain, extra_chains, merged, predictions, outputs, chain_path)")
    end
end

nothing
