#=
================================================================================
 OctofitterRV1.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Basic RV Fit" (K2-131 b,
 reproducing the RadVel example):
   https://sefffal.github.io/Octofitter.jl/dev/rv-1/

 Data: RadVel's example file k2-131.txt (HARPS-N + PFS), downloaded once and
 cached next to this file as rv1_k2-131.txt.

 Model: star A (mass prior); planet b on a circular orbit with i = 90°,
 Ω = ω = 0 fixed (RVs alone can't constrain i or Ω, so `mass` is m·sin i),
 P ~ truncated Normal(0.3693038, 9.1e-6) d, phase τ ~ UniformCircular(1),
 tp = τ·P + 57782, mass ~ LogUniform(0.001, 10) M_jup. One
 RadialVelocityObs per instrument (target=A, ref=Barycentre) with its own
 offset and jitter.

 Pipeline:
   download/cache -> per-instrument tables -> System -> LogDensityModel
   -> initialize! -> octoplot(init) -> octofit(Xoshiro(0), model)
   -> rvplot (single draw, all instruments + phase fold), octoplot (many
      draws, one panel per instrument), octocorner
   -> derived m·sin i [M⊕, M_jup] and semi-amplitude K [m/s]
   -> "Simulating RV Data" section with PlanetOrbits (stellar reflex and
      relative RV curves) -> FITS chain

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `rv1_result`)
   Shell:    julia OctofitterRV1.jl
   Library:  ENV["OCTORV_AUTORUN"] = "false"; include("OctofitterRV1.jl")
             res = OctofitterRV1.run_rv1(iterations=2000, seed=1)

 Threads: the tutorial's HMC fit is a single chain, so it runs in the REPL.
 ENV["OCTORV_SAMPLER"] = "pigeons" switches to parallel tempering; in a
 1-thread REPL that relaunches in a child `julia --threads=auto` process
 (OCTORV_RELAUNCH, OCTORV_THREADS control this), as in the other modules.

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; ENV["OCTORV_ENV_MODE"] = "local" uses ./octofitter_v9_env). Needs
 Octofitter v9, OctofitterRadialVelocity v9, Distributions, CairoMakie,
 PairPlots, PlanetOrbits, CSV, DataFrames, Pigeons, MCMCChains.

 Outputs (prefix "rv1_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-24  Initial version on the OctofitterThieleInnes v1.0.1
                   scaffolding ([ORV +t] debug stages, soft optional stages,
                   env bootstrap with v9 guards for Octofitter and
                   OctofitterRadialVelocity, @__DIR__ outputs, dark theme with
                   light corner plots, optional threaded Pigeons relaunch).
                   Adds: cached data download; per-instrument observations
                   built from the telescope column; seeded HMC; rvplot;
                   derived m·sin i and K; PlanetOrbits RV simulation.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "OctofitterRadialVelocity", "Distributions", "CairoMakie",
                "PairPlots", "PlanetOrbits", "CSV", "DataFrames", "Pigeons", "MCMCChains"),
    min_version = Dict("Octofitter" => v"9", "OctofitterRadialVelocity" => v"9"),
    mode = lowercase(get(ENV, "OCTORV_ENV_MODE", "temp"))

    direct_deps() = Dict(info.name => info.version
                         for info in values(Pkg.dependencies()) if info.is_direct_dep)
    missing_deps() = [n for n in required if !haskey(direct_deps(), n)]
    function versions_ok()
        d = direct_deps()
        all(!haskey(d, n) || (d[n] !== nothing && d[n] >= v) for (n, v) in min_version)
    end
    env_ok() = isempty(missing_deps()) && versions_ok()

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
            println("[ORV] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[ORV] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            specs = [n == "OctofitterRadialVelocity" ? Pkg.PackageSpec(name=n, version="9") :
                                                       Pkg.PackageSpec(name=n) for n in miss]
            Pkg.add(specs; preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[ORV] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[ORV] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[ORV] Existing v9 environment found at ", Base.active_project())
        else
            specs = [haskey(min_version, n) ? Pkg.PackageSpec(name=n, version="9") :
                                              Pkg.PackageSpec(name=n) for n in required]
            Pkg.add(specs)
        end
        println("[ORV] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ", OctofitterRadialVelocity v",
                direct_deps()["OctofitterRadialVelocity"], ")")
    end
    flush(stdout)
end

println("[ORV] Loading Octofitter, OctofitterRadialVelocity, PlanetOrbits, CairoMakie, ",
        "PairPlots, CSV, DataFrames, Distributions, Pigeons, MCMCChains ",
        "(first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterRV1

const LOAD_T0 = time()

using Octofitter
using OctofitterRadialVelocity
using PlanetOrbits
using CairoMakie
using PairPlots
using CSV
using DataFrames
using Distributions
using Pigeons            # enables octofit_pigeons (optional sampler)
import MCMCChains
import Downloads
import Random
import Statistics
using Printf

export run_rv1, load_k2_131, build_star, build_planet, build_rv_observations,
       build_rv_system, simulate_rv_example, set_plot_theme!

const VERSION_STRING = "1.0"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
const ORV_VERSION = pkgversion(OctofitterRadialVelocity)
if (OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9") ||
   (ORV_VERSION !== nothing && ORV_VERSION < v"9")
    error("""
    OctofitterRV1 needs Octofitter v9+ and OctofitterRadialVelocity v9+, but the
    active environment has Octofitter v$(OCTOFITTER_VERSION) and
    OctofitterRadialVelocity v$(ORV_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const K2_131_URL = "https://raw.githubusercontent.com/California-Planet-Search/radvel/master/example_data/k2-131.txt"
const TP_REF_MJD = 57782.0             # reference epoch for the phase τ (tutorial)

# Physical constants (SI) for the derived semi-amplitude
const G_SI    = 6.67430e-11
const MSUN_KG = 1.98847e30
const MSUN_IN_MEARTH = 332946.0487

# Tutorial's per-instrument offset priors [m/s]; other telescopes get
# Normal(median rv, 100).
const OFFSET_PRIORS = Dict("harps-n" => Normal(-6693, 100), "pfs" => Normal(0, 100))

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterRV1.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[ORV +%7.1fs] ", time() - T0[]), msg...)
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

function print_env_info(outdir, sampler)
    DEBUG[] || return nothing
    dbg("OctofitterRV1 v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | OctofitterRadialVelocity ", something(ORV_VERSION, "unknown"),
        " | sampler ", sampler, " | threads ", Threads.nthreads())
    dbg("project    = ", Base.active_project())
    dbg("pwd()      = ", pwd())
    dbg("SCRIPT_DIR = ", SCRIPT_DIR)
    dbg("outdir     = ", abspath(outdir))
    dbg("interactive = ", isinteractive(),
        " | Makie backend = ", CairoMakie.Makie.current_backend())
    return nothing
end

colvec(chain, p::Symbol) = vec(Array(chain[p]))
haspar(chain, p::Symbol) = p in names(chain)
qline(x) = Statistics.quantile(filter(isfinite, x), (0.16, 0.5, 0.84))

"""
    semi_amplitude(chain)

Stellar reflex semi-amplitude K [m/s] per sample:
K = (2πG/P)^(1/3) · m sin i / (M★ + m)^(2/3) / √(1 − e²).
"""
function semi_amplitude(chain)
    n = size(chain, 1) * size(chain, 3)
    P = colvec(chain, :b_P) .* 86400
    m = colvec(chain, :b_mass) .* MSUN_KG
    M = colvec(chain, :A_mass) .* MSUN_KG
    e = haspar(chain, :b_e) ? colvec(chain, :b_e) : zeros(n)
    sini = haspar(chain, :b_i) ? sin.(colvec(chain, :b_i)) : ones(n)
    return (2π * G_SI ./ P) .^ (1 / 3) .* m .* sini ./ (M .+ m) .^ (2 / 3) ./ sqrt.(1 .- e .^ 2)
end

function print_summary(chain, label)
    DEBUG[] || return nothing
    dbg("Posterior median [16th, 84th] — ", label, " (", size(chain, 1), " samples):")
    row(lab, x) = (q = qline(x);
        dbg(@sprintf("    %-24s %14.6f  [%14.6f, %14.6f]", lab, q[2], q[1], q[3])))
    row("A_mass [M⊙]", colvec(chain, :A_mass))
    row("b_P [d]", colvec(chain, :b_P))
    haspar(chain, :b_τ)  && row("b_τ (phase)", colvec(chain, :b_τ))
    haspar(chain, :b_tp) && row("b_tp [MJD]", colvec(chain, :b_tp))
    m = colvec(chain, :b_mass)
    row("m sin i [M⊕]", m .* MSUN_IN_MEARTH)
    row("m sin i [M_jup]", m ./ mjup)
    row("K [m/s] (derived)", semi_amplitude(chain))
    for p in names(chain)
        occursin(r"_(offset|jitter)$", string(p)) || continue
        row(string(p, " [m/s]"), colvec(chain, p))
    end
    return nothing
end

function print_divergences(chain; label="chain")
    DEBUG[] || return nothing
    try
        n = Int(sum(colvec(chain, :numerical_error)))
        dbg(@sprintf("  %s: divergences %d / %d (%.1f%%), mean acceptance %.3f", label, n,
                     size(chain, 1), 100n / size(chain, 1),
                     Statistics.mean(colvec(chain, :acceptance_rate))))
    catch
        dbg("  (", label, ": no HMC diagnostics columns — e.g. a Pigeons chain)")
    end
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
                dbg(@sprintf("    %-24s R̂ = %.4f", p, r))
            end
        end
    catch
        dbg("  (couldn't extract the rhat column; see table above)")
    end
    return ss
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

const C1 = "#56B4E9"
const C2 = "#E69F00"

"Histograms of the derived m sin i [M⊕] and K [m/s]."
function derived_figure(chain)
    fig = Figure(size=(1000, 400))
    ax1 = Axis(fig[1, 1], xlabel="m sin i [M⊕]", ylabel="density")
    hist!(ax1, colvec(chain, :b_mass) .* MSUN_IN_MEARTH; bins=50, normalization=:pdf,
          color=(C1, 0.7))
    ax2 = Axis(fig[1, 2], xlabel="semi-amplitude K [m/s]")
    hist!(ax2, semi_amplitude(chain); bins=50, normalization=:pdf, color=(C2, 0.7))
    Label(fig[0, :], "K2-131 b: derived minimum mass and RV semi-amplitude",
          fontsize=18, font=:bold)
    return fig
end

"Raw RVs per instrument vs time (before any offsets are fitted)."
function data_figure(tables)
    fig = Figure(size=(1000, 300 * length(tables) + 60))
    for (k, (tel, t)) in enumerate(tables)
        ax = Axis(fig[k, 1], ylabel="RV [m/s]", title="$tel ($(length(t.epoch)) points)",
                  xlabel=k == length(tables) ? "epoch [MJD]" : "")
        errorbars!(ax, t.epoch, t.rv, t.σ_rv, color=(k == 1 ? C1 : C2, 0.6))
        scatter!(ax, t.epoch, t.rv, color=k == 1 ? C1 : C2, markersize=7)
    end
    return fig
end

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
"""
    load_k2_131(; cache_dir=SCRIPT_DIR, url=K2_131_URL, force=false)

Download (once) and parse RadVel's k2-131.txt. Returns
`(raw, dat, tels)` where `dat` has columns epoch [MJD], rv, σ_rv [m/s].
"""
function load_k2_131(; cache_dir::AbstractString=SCRIPT_DIR, url::AbstractString=K2_131_URL,
                     force::Bool=false)
    path = joinpath(cache_dir, "rv1_k2-131.txt")
    if force || !isfile(path)
        dbg("  downloading ", url)
        Downloads.download(url, path)
        dbg(@sprintf("  cached → %s (%.1f kB)", path, filesize(path) / 1024))
    else
        dbg("  using cached ", path)
    end
    raw = CSV.read(path, DataFrame; delim=' ', ignorerepeated=true)
    dbg("  columns: ", join(names(raw), ", "), "; ", nrow(raw), " rows")
    dat = DataFrame()
    dat.epoch = jd2mjd.(raw.time)
    dat.rv    = raw.mnvel
    dat.σ_rv  = raw.errvel
    tels = sort(unique(String.(raw.tel)))
    for tel in tels
        sel = String.(raw.tel) .== tel
        e = dat.epoch[sel]
        dbg(@sprintf("  %-8s n=%3d  MJD %.2f–%.2f (%s → %s)  rv median %.1f, rms %.1f m/s, median σ %.2f m/s",
                     tel, count(sel), minimum(e), maximum(e),
                     string(mjd2date(minimum(e)))[1:10], string(mjd2date(maximum(e)))[1:10],
                     Statistics.median(dat.rv[sel]), Statistics.std(dat.rv[sel]),
                     Statistics.median(dat.σ_rv[sel])))
    end
    return (; raw, dat, tels)
end

# -----------------------------------------------------------------------------
# Model components (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
"""
    build_star(; mass_prior=truncated(Normal(0.82, 0.02), lower=0.1))

K2-131 mass prior from Baines & Armstrong 2011 (as in the tutorial).
"""
function build_star(; mass_prior::Distribution=truncated(Normal(0.82, 0.02), lower=0.1))
    dbg("  star 'A': mass ~ ", mass_prior)
    vars = @eval @variables begin
        mass ~ $mass_prior           # [M⊙]
    end
    return Body(name="A", variables=vars)
end

"""
    build_planet(host; P_prior=truncated(Normal(0.3693038, 0.0000091), lower=0.0001),
                 mass_prior=LogUniform(0.001mjup, 10mjup), tp_ref=TP_REF_MJD)

Circular, edge-on (i = π/2, Ω = ω = 0) planet: `mass` is m·sin i. Phase
τ ~ UniformCircular(1) with tp = τ·P + tp_ref.
"""
function build_planet(host; P_prior::Distribution=truncated(Normal(0.3693038, 0.0000091), lower=0.0001),
                      mass_prior::Distribution=LogUniform(0.001mjup, 10mjup),
                      tp_ref::Real=TP_REF_MJD)
    tr = Float64(tp_ref)
    dbg("  planet 'b': P ~ ", P_prior, " [d], mass ~ ", mass_prior,
        " [M⊙], e = ω = Ω = 0, i = 90°, tp = τ·P + ", tr)
    vars = @eval @variables begin
        # RV-only: i and Ω unconstrained, so fix them (mass is then m sin i)
        i = pi / 2
        Ω = 0.0
        e = 0.0
        ω = 0.0
        P ~ $P_prior                 # [days]
        τ ~ UniformCircular(1.0)     # phase in [0, 1)
        tp = τ * P + $tr             # [MJD]
        mass ~ $mass_prior           # [M⊙]
    end
    return Body(name="b", about=host, variables=vars)
end

"""
    build_rv_observations(A, dat, tel_col, tels; jitter_prior=LogUniform(0.1, 100))

One RadialVelocityObs (target=A, ref=Barycentre) per telescope with its own
offset (tutorial priors for harps-n/pfs, else Normal(median rv, 100)) and jitter.
"""
function build_rv_observations(A, dat, tel_col, tels;
                               jitter_prior::Distribution=LogUniform(0.1, 100))
    obs = Any[]
    for tel in tels
        sel = String.(tel_col) .== tel
        off = get(OFFSET_PRIORS, tel, Normal(Statistics.median(dat.rv[sel]), 100))
        dbg("  obs '", tel, "': ", count(sel), " RVs, offset ~ ", off, ", jitter ~ ", jitter_prior)
        vars = @eval @variables begin
            offset ~ $off            # [m/s]
            jitter ~ $jitter_prior   # [m/s]
        end
        push!(obs, RadialVelocityObs(dat[sel, :]; target=A, ref=Barycentre,
                                     name=tel, variables=vars))
    end
    return obs
end

"""
    build_rv_system(data; kw...)

Fresh bodies + observations + System "k2_131" (no reference frame, as in
the tutorial). `data` is the NamedTuple from `load_k2_131`.
"""
function build_rv_system(data; kw...)
    A = build_star()
    b = build_planet(A; kw...)
    obs = build_rv_observations(A, data.dat, data.raw.tel, data.tels)
    dbg("  system 'k2_131': bodies A, b; ", length(obs), " RV instruments; no frame variables")
    return System(name="k2_131", bodies=[A, b], observations=obs)
end

# -----------------------------------------------------------------------------
# "Simulating RV Data" section
# -----------------------------------------------------------------------------
"""
    simulate_rv_example()

The tutorial's PlanetOrbits example: 1 M⊙ star + 1 M_jup companion
(a = 1 AU, e = 0.1, ω = 0.5, i = 90°, tp = 58000), sampled every 10 d.
Returns epochs, stellar reflex RV (vs barycentre) and relative RV (b vs A).
"""
function simulate_rv_example()
    star = PlanetOrbits.Body(mass=1.0, name=:A)
    comp = PlanetOrbits.Body(mass=mjup, name=:b)
    posys = PlanetOrbits.System(
        (star, comp),
        (PlanetOrbits.Orbit(comp, about=star; a=1.0, e=0.1, ω=0.5, i=pi / 2, Ω=0.0, tp=58000.0),)
    )
    epochs = collect(58000.0:10.0:58400.0)
    traj = orbitsolve(posys, epochs)
    rv_star = [radvel(traj[i], :A, barycentre(posys)) for i in eachindex(epochs)]
    rv_rel  = [radvel(traj[i], :b, :A) for i in eachindex(epochs)]
    return (; epochs, rv_star, rv_rel)
end

function simulation_figure(sim)
    fig = Figure(size=(1000, 560))
    ax1 = Axis(fig[1, 1], ylabel="stellar reflex RV [m/s]",
               title="Simulated: 1 M_jup at 1 AU around 1 M⊙ (e = 0.1)")
    scatterlines!(ax1, sim.epochs, sim.rv_star, color=C1, markersize=6)
    ax2 = Axis(fig[2, 1], xlabel="epoch [MJD]", ylabel="relative RV b vs A [km/s]")
    scatterlines!(ax2, sim.epochs, sim.rv_rel ./ 1000, color=C2, markersize=6)
    linkxaxes!(ax1, ax2)
    return fig
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_rv1(; sampler=:hmc, iterations=1000, n_rounds=10, seed=0, simulate=true,
              outdir=SCRIPT_DIR, prefix="rv1_", dark=true, corner_dark=false,
              show_plots=true, force_download=false)

Run the Basic RV Fit tutorial. Returns a NamedTuple of results.
"""
function run_rv1(; sampler::Symbol=:hmc,
                   iterations::Integer=1000,
                   n_rounds::Integer=10,
                   seed::Integer=0,
                   simulate::Bool=true,
                   outdir::AbstractString=SCRIPT_DIR,
                   prefix::AbstractString="rv1_",
                   dark::Bool=true,
                   corner_dark::Bool=false,
                   show_plots::Bool=true,
                   force_download::Bool=false)
    sampler in (:hmc, :pigeons) || error("sampler must be :hmc or :pigeons")
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    print_env_info(outdir, sampler)
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

    # ---- Data ---------------------------------------------------------------
    data = stage("Load K2-131 RVs (RadVel example data)") do
        load_k2_131(; cache_dir=outdir, force=force_download)
    end
    soft_stage("Plot raw RVs per instrument") do
        tables = [tel => data.dat[String.(data.raw.tel) .== tel, :] for tel in data.tels]
        savefig!(data_figure(tables), "data.png")
    end

    # ---- Model --------------------------------------------------------------
    sys = stage("Build system") do
        build_rv_system(data)
    end
    DEBUG[] && display(sys)

    model = stage("Compile LogDensityModel") do
        Octofitter.LogDensityModel(sys)
    end
    DEBUG[] && display(model)

    init_chain = stage("Initialize starting points (no guesses)") do
        initialize!(model)
    end
    soft_stage("Plot initial guess") do
        savefig!(inout(() -> octoplot(model, init_chain)), "init.png")
    end

    # ---- Sampling -----------------------------------------------------------
    chain = if sampler === :pigeons
        stage("Sample posterior (Pigeons, n_rounds=$n_rounds)") do
            c, pt = octofit_pigeons(model; n_rounds=n_rounds)
            soft_stage("Pigeons log-evidence") do
                dbg(@sprintf("  log Z ≈ %.2f, Λ = %.2f", Pigeons.stepping_stone(pt),
                             Pigeons.global_barrier(pt)))
            end
            c
        end
    else
        stage("Sample posterior (HMC, $iterations iterations, rng = Xoshiro($seed))") do
            octofit(Random.Xoshiro(seed), model; iterations=iterations)
        end
    end
    print_divergences(chain)
    print_summary(chain, "K2-131")
    soft_stage("R-hat") do
        print_rhat(chain)
    end

    # ---- Plots --------------------------------------------------------------
    stage("rvplot (single draw, all instruments, phase-folded)") do
        savefig!(inout(() -> rvplot(model, chain)), "rvplot.png")
    end
    soft_stage("octoplot (many draws, one panel per instrument)") do
        savefig!(inout(() -> octoplot(model, chain)), "octoplot.png")
    end
    soft_stage("Corner plot") do
        savefig!(inout(() -> corner(model, chain; dark=corner_dark)), "corner.png")
    end
    soft_stage("Derived m sin i and K histograms") do
        savefig!(derived_figure(chain), "derived.png")
    end

    # ---- Simulation section -------------------------------------------------
    sim = nothing
    if simulate
        sim = soft_stage("Simulate RVs with PlanetOrbits (tutorial example)") do
            s = simulate_rv_example()
            lo, hi = extrema(s.rv_star)
            dbg(@sprintf("  stellar reflex RV extrema: (%.3f, %.3f) m/s", lo, hi))
            lo2, hi2 = extrema(s.rv_rel)
            dbg(@sprintf("  relative RV (b vs A) extrema: (%.1f, %.1f) m/s", lo2, hi2))
            savefig!(simulation_figure(s), "simulated.png")
            s
        end
    end

    # ---- Saving -------------------------------------------------------------
    chain_path = joinpath(outdir, prefix * "chain.fits")
    soft_stage("Save chain -> FITS (+ reload check)") do
        Octofitter.savechain(chain_path, chain)
        push!(outputs, chain_path)
        c2 = Octofitter.loadchain(chain_path; model)
        dbg("  reloaded chain size = ", size(c2),
            size(c2) == size(chain) ? "  (matches)" : "  (DIFFERS from saved)")
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f s", time() - T0[]))

    return (; data, system=sys, model, init_chain, chain, simulation=sim, outputs, chain_path)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (used only for the Pigeons sampler in a 1-thread session)
# -----------------------------------------------------------------------------
"""
    relaunch_threaded(file; threads="auto", outdir=SCRIPT_DIR, prefix="rv1_")

Run `file` in a child Julia process with `--threads=threads` and the same
active project, streaming its output; then load the FITS chain it wrote.
"""
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString="rv1_")
    T0[] = time()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTORV_CHILD" => "1")
    dbg("Pigeons sampler requested and this session has 1 thread; Julia can't add threads")
    dbg("to a running process. Relaunching with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTORV_CHILD=1]")
    dbg("  Child output follows. Plots go to files only (not the plot pane).")
    t = time()
    ok = try
        run(pipeline(cmd; stdout=stdout, stderr=stderr))
        true
    catch err
        dbg("✖ child process failed or was interrupted: ", sprint(showerror, err))
        false
    end
    dbg(@sprintf("Child process finished in %.1f min (%s)", (time() - t) / 60,
                 ok ? "success" : "FAILED"))
    p = joinpath(outdir, prefix * "chain.fits")
    chain = nothing
    if isfile(p) && mtime(p) >= t
        chain = soft_stage("Load chain written by child") do
            Octofitter.loadchain(p)
        end
    end
    new_files = [joinpath(outdir, f) for f in readdir(outdir)
                 if startswith(f, prefix) && mtime(joinpath(outdir, f)) >= t]
    list_outputs(new_files)
    return (; relaunched=true, success=ok, chain, outputs=new_files)
end

println(@sprintf("[ORV] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterRV1

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia OctofitterRV1.jl`                     -> runs
#   - VSCode "Execute active File in REPL"         -> runs (interactive session)
#   - ENV["OCTORV_AUTORUN"] = "false"; include()   -> loads module only
#   - ENV["OCTORV_SAMPLER"] = "pigeons"            -> parallel tempering; in a
#     1-thread session this relaunches in `julia --threads=auto` (disable with
#     ENV["OCTORV_RELAUNCH"] = "false"; ENV["OCTORV_THREADS"] sets the count)
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTORV_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTORV_CHILD", "") == "1",
    sampler   = Symbol(lowercase(get(ENV, "OCTORV_SAMPLER", "hmc"))),
    relaunch  = !is_child && sampler === :pigeons && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTORV_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global rv1_result = OctofitterRV1.relaunch_threaded(
                this_file; threads=get(ENV, "OCTORV_THREADS", "auto"))
            println("[ORV] Child run finished. `rv1_result` has fields: relaunched, success, ",
                    "chain (loaded from FITS), outputs")
        else
            global rv1_result = OctofitterRV1.run_rv1(
                sampler=sampler, iterations=1000, seed=0, show_plots=!is_child)
            is_child || println("[ORV] Result stored in `rv1_result` (fields: data, system, ",
                "model, init_chain, chain, simulation, outputs, chain_path)")
        end
    end
end

nothing
