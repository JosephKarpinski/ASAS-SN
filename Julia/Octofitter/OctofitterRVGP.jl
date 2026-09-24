#=
================================================================================
 OctofitterRVGP.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Fit Gaussian Process"
 (K2-131 b with stellar activity; reproduces the RadVel GP example):
   https://sefffal.github.io/Octofitter.jl/dev/rv-gp/

 Continues OctofitterRV1.jl: same data (cached rv1_k2-131.txt is reused),
 same star/planet model; each instrument's RadialVelocityObs gets a
 Gaussian process for stellar activity.

 Models (both sampled with Pigeons parallel tempering, as in the tutorial):
   :abstractgps  quasi-periodic kernel
                 η_1² · SqExp(len η_2) · Periodic(r = η_4, period η_3)
                 priors from Dai et al. 2017 (as in the tutorial)
   :celerite     approximate quasi-periodic Celerite kernel
                 (RealTerm + ComplexTerm with B, C, L, Prot);
                 finite-difference gradients (AutoFiniteDiff)

 Pipeline per model:
   System -> LogDensityModel -> initialize! -> octoplot(init)
   -> octofit_pigeons(n_rounds) -> log-evidence, summary incl. m sin i, K,
      GP hyperparameters -> rvplot (GP band), octoplot (per-draw GP curves),
      octoplot(N=50, figscale=1.5), octoplot(show_phase=true), corner -> FITS
 Then: comparison with the no-GP fit (rv1_chain.fits from OctofitterRV1.jl,
 if present — loaded, not refitted): K and m sin i table + histograms.

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `rvgp_result`)
   Shell:    julia --threads=auto OctofitterRVGP.jl
   Library:  ENV["OCTOGP_AUTORUN"] = "false"; include("OctofitterRVGP.jl")
             res = OctofitterRVGP.run_rvgp(models=(:celerite,), n_rounds=7)

 Threads: both fits use Pigeons, which uses threads. In a 1-thread REPL the
 run is relaunched in a child `julia --threads=auto` (output streamed here;
 chains loaded back afterwards). OCTOGP_RELAUNCH=false disables this,
 OCTOGP_THREADS sets the count. The docs report ~420 s for the AbstractGPs
 fit and ~55 s for Celerite at n_rounds=7 on their build machine; expect
 longer on 4 threads.

 Convenience ENV settings (also passed to a relaunched child):
   OCTOGP_MODELS = "abstractgps,celerite" (default) | "celerite" | ...
   OCTOGP_ROUNDS = "7" (default; 2^n_rounds scans, 2^n_rounds samples)

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOGP_ENV_MODE=local uses ./octofitter_v9_env). Needs Octofitter
 v9, OctofitterRadialVelocity v9, PlanetOrbits, CairoMakie, PairPlots, CSV,
 DataFrames, Distributions, Pigeons, AbstractGPs, DifferentiationInterface,
 FiniteDiff, MCMCChains.

 Outputs (prefix "rvgp_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-24  Initial version on the OctofitterRV1 v1.0 scaffolding
                   ([OGP +t] debug stages, soft optional stages, env
                   bootstrap with v9 guards, shared cached data, @__DIR__
                   outputs, dark theme with light corner plots). Adds:
                   AbstractGPs quasi-periodic and Celerite kernels; Pigeons
                   with default threaded relaunch; log-evidence; GP
                   hyperparameter summaries; rvplot/octoplot GP views;
                   comparison with the saved no-GP chain.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "OctofitterRadialVelocity", "PlanetOrbits", "CairoMakie",
                "PairPlots", "CSV", "DataFrames", "Distributions", "Pigeons", "AbstractGPs",
                "DifferentiationInterface", "FiniteDiff", "MCMCChains"),
    min_version = Dict("Octofitter" => v"9", "OctofitterRadialVelocity" => v"9"),
    mode = lowercase(get(ENV, "OCTOGP_ENV_MODE", "temp"))

    direct_deps() = Dict(info.name => info.version
                         for info in values(Pkg.dependencies()) if info.is_direct_dep)
    missing_deps() = [n for n in required if !haskey(direct_deps(), n)]
    function versions_ok()
        d = direct_deps()
        all(!haskey(d, n) || (d[n] !== nothing && d[n] >= v) for (n, v) in min_version)
    end
    env_ok() = isempty(missing_deps()) && versions_ok()
    spec(n) = haskey(min_version, n) ? Pkg.PackageSpec(name=n, version="9") :
                                       Pkg.PackageSpec(name=n)

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
            println("[OGP] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[OGP] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[OGP] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[OGP] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[OGP] Existing v9 environment found at ", Base.active_project())
        else
            Pkg.add(spec.(collect(required)))
        end
        println("[OGP] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ", OctofitterRadialVelocity v",
                direct_deps()["OctofitterRadialVelocity"], ")")
    end
    flush(stdout)
end

println("[OGP] Loading Octofitter, OctofitterRadialVelocity (+ vendored Celerite), ",
        "AbstractGPs, Pigeons, CairoMakie, PairPlots, CSV, DataFrames, Distributions, ",
        "DifferentiationInterface, FiniteDiff (first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterRVGP

const LOAD_T0 = time()

using Octofitter
using OctofitterRadialVelocity
import OctofitterRadialVelocity.Celerite      # vendored Celerite (NOT `using Celerite`)
using PlanetOrbits
using CairoMakie
using PairPlots
using CSV
using DataFrames
using Distributions
using Pigeons
using AbstractGPs
using DifferentiationInterface, FiniteDiff
import MCMCChains
import Downloads
import Statistics
using Printf

export run_rvgp, load_k2_131, build_gp_system, quasiperiodic, quasistatic,
       set_plot_theme!

const VERSION_STRING = "1.0"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
const ORV_VERSION = pkgversion(OctofitterRadialVelocity)
if (OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9") ||
   (ORV_VERSION !== nothing && ORV_VERSION < v"9")
    error("""
    OctofitterRVGP needs Octofitter v9+ and OctofitterRadialVelocity v9+, but the
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
const DATA_CACHE = "rv1_k2-131.txt"          # shared with OctofitterRV1.jl
const NOGP_CHAIN = "rv1_chain.fits"          # written by OctofitterRV1.jl
const TP_REF_MJD = 57782.0

const G_SI = 6.67430e-11
const MSUN_KG = 1.98847e30
const MSUN_IN_MEARTH = 332946.0487

const OFFSET_PRIORS = Dict("harps-n" => Normal(-6693, 100), "pfs" => Normal(0, 100))

# GP hyperparameter prior centres from Dai et al. 2017 (as in the tutorial)
const GP_EXPLENGTH_MEAN = 9.5 * sqrt(2.0)          # sqrt(2)·tau [days]
const GP_EXPLENGTH_UNC  = 1.0 * sqrt(2.0)
const GP_PERLENGTH_MEAN = sqrt(1.0 / (2.0 * 3.32)) # sqrt(1/(2·gamma))
const GP_PERLENGTH_UNC  = 0.019
const GP_PER_MEAN       = 9.64                     # T_bar [days]

const MODEL_LABELS = Dict(:nogp => "no GP (rv1)", :abstractgps => "AbstractGPs quasi-periodic",
                          :celerite => "Celerite quasi-periodic")
const MODEL_COLORS = Dict(:nogp => "#999999", :abstractgps => "#56B4E9", :celerite => "#E69F00")

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterRVGP.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[OGP +%7.1fs] ", time() - T0[]), msg...)
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

function print_env_info(outdir, models, n_rounds)
    DEBUG[] || return nothing
    dbg("OctofitterRVGP v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | OctofitterRadialVelocity ", something(ORV_VERSION, "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("models = ", join(models, ", "), " | n_rounds = ", n_rounds,
        " (", 2^n_rounds, " scans → ", 2^n_rounds, " posterior samples)")
    Threads.nthreads() == 1 &&
        dbg("  NOTE: running Pigeons on 1 thread (relaunch disabled) — expect a long run; start julia with --threads=auto to speed up")
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

"Stellar reflex semi-amplitude K [m/s] per sample."
function semi_amplitude(chain)
    n = size(chain, 1) * size(chain, 3)
    P = colvec(chain, :b_P) .* 86400
    m = colvec(chain, :b_mass) .* MSUN_KG
    M = colvec(chain, :A_mass) .* MSUN_KG
    e = haspar(chain, :b_e) ? colvec(chain, :b_e) : zeros(n)
    sini = haspar(chain, :b_i) ? sin.(colvec(chain, :b_i)) : ones(n)
    return (2π * G_SI ./ P) .^ (1 / 3) .* m .* sini ./ (M .+ m) .^ (2 / 3) ./ sqrt.(1 .- e .^ 2)
end
msini_earth(chain) = colvec(chain, :b_mass) .* MSUN_IN_MEARTH

const OBS_PARAM_RE = r"_(offset|jitter|η_\d|B|C|L|Prot)$"

function print_summary(chain, label)
    DEBUG[] || return nothing
    dbg("Posterior median [16th, 84th] — ", label, " (", size(chain, 1), " samples):")
    row(lab, x) = (q = qline(x);
        dbg(@sprintf("    %-24s %14.6f  [%14.6f, %14.6f]", lab, q[2], q[1], q[3])))
    row("A_mass [M⊙]", colvec(chain, :A_mass))
    row("b_P [d]", colvec(chain, :b_P))
    haspar(chain, :b_tp) && row("b_tp [MJD]", colvec(chain, :b_tp))
    row("m sin i [M⊕]", msini_earth(chain))
    row("K [m/s] (derived)", semi_amplitude(chain))
    for p in names(chain)
        occursin(OBS_PARAM_RE, string(p)) || continue
        row(string(p), colvec(chain, p))
    end
    return nothing
end

function print_comparison(pairs)
    DEBUG[] || return nothing
    dbg("Model comparison — median [16th, 84th]:")
    for (lab, f) in (("K [m/s]", semi_amplitude), ("m sin i [M⊕]", msini_earth))
        for (k, c) in pairs
            q = qline(f(c))
            dbg(@sprintf("    %-14s %-28s %8.2f [%8.2f, %8.2f]  (%d samples)", lab,
                         MODEL_LABELS[k], q[2], q[1], q[3], size(c, 1) * size(c, 3)))
        end
    end
    for p in (:harps_n_jitter, :pfs_jitter)
        for (k, c) in pairs
            haspar(c, p) || continue
            q = qline(colvec(c, p))
            dbg(@sprintf("    %-14s %-28s %8.2f [%8.2f, %8.2f]", string(p), MODEL_LABELS[k],
                         q[2], q[1], q[3]))
        end
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

"K and m sin i histograms for every available fit."
function comparison_figure(pairs)
    fig = Figure(size=(1100, 420))
    ax1 = Axis(fig[1, 1], xlabel="semi-amplitude K [m/s]", ylabel="density")
    ax2 = Axis(fig[1, 2], xlabel="m sin i [M⊕]")
    for (k, c) in pairs
        col = (MODEL_COLORS[k], 0.5)
        hist!(ax1, semi_amplitude(c); bins=40, normalization=:pdf, color=col, label=MODEL_LABELS[k])
        hist!(ax2, msini_earth(c); bins=40, normalization=:pdf, color=col)
    end
    axislegend(ax1, position=:rt)
    Label(fig[0, :], "K2-131 b: effect of modelling stellar activity", fontsize=18, font=:bold)
    return fig
end

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
"""
    load_k2_131(; cache_dir=SCRIPT_DIR, force=false)

Download (once) and parse RadVel's k2-131.txt; reuses OctofitterRV1's cache.
Returns `(raw, dat, tels)`.
"""
function load_k2_131(; cache_dir::AbstractString=SCRIPT_DIR, force::Bool=false)
    path = joinpath(cache_dir, DATA_CACHE)
    if force || !isfile(path)
        dbg("  downloading ", K2_131_URL)
        Downloads.download(K2_131_URL, path)
        dbg(@sprintf("  cached → %s (%.1f kB)", path, filesize(path) / 1024))
    else
        dbg("  using cached ", path)
    end
    raw = CSV.read(path, DataFrame; delim=' ', ignorerepeated=true)
    dat = DataFrame()
    dat.epoch = jd2mjd.(raw.time)
    dat.rv    = raw.mnvel
    dat.σ_rv  = raw.errvel
    tels = sort(unique(String.(raw.tel)))
    for tel in tels
        sel = String.(raw.tel) .== tel
        dbg(@sprintf("  %-8s n=%3d  rv median %.1f, rms %.1f m/s, median σ %.2f m/s", tel,
                     count(sel), Statistics.median(dat.rv[sel]), Statistics.std(dat.rv[sel]),
                     Statistics.median(dat.σ_rv[sel])))
    end
    return (; raw, dat, tels)
end

# -----------------------------------------------------------------------------
# GP kernels (as in the tutorial)
# -----------------------------------------------------------------------------
"AbstractGPs quasi-periodic kernel from observation variables η_1..η_4."
quasiperiodic(θ_obs) = GP(
    θ_obs.η_1^2 *
    (SqExponentialKernel() ∘ ScaleTransform(1 / (θ_obs.η_2))) *
    (PeriodicKernel(r=[θ_obs.η_4]) ∘ ScaleTransform(1 / (θ_obs.η_3)))
)

"Celerite approximate quasi-periodic kernel from B, C, L, Prot."
quasistatic(θ_obs) = Celerite.CeleriteGP(
    Celerite.RealTerm(
        #=log_a=# log(θ_obs.B * (1 + θ_obs.C) / (2 + θ_obs.C)),
        #=log_c=# log(1 / θ_obs.L)
    ) + Celerite.ComplexTerm(
        #=log_a=# log(θ_obs.B / (2 + θ_obs.C)),
        #=log_b=# -Inf,
        #=log_c=# log(1 / θ_obs.L),
        #=log_d=# log(2pi / θ_obs.Prot)
    )
)

# -----------------------------------------------------------------------------
# Model components (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
function build_star(; mass_prior::Distribution=truncated(Normal(0.82, 0.02), lower=0.1))
    dbg("  star 'A': mass ~ ", mass_prior)
    vars = @eval @variables begin
        mass ~ $mass_prior
    end
    return Body(name="A", variables=vars)
end

function build_planet(host; P_prior::Distribution=truncated(Normal(0.3693038, 0.0000091), lower=0.0001),
                      mass_prior::Distribution=LogUniform(0.001mjup, 10mjup),
                      tp_ref::Real=TP_REF_MJD)
    tr = Float64(tp_ref)
    dbg("  planet 'b': P ~ ", P_prior, ", mass ~ ", mass_prior, ", circular, i = 90°")
    vars = @eval @variables begin
        i = pi / 2
        Ω = 0.0
        e = 0.0
        ω = 0.0
        P ~ $P_prior
        τ ~ UniformCircular(1.0)
        tp = τ * P + $tr
        mass ~ $mass_prior
    end
    return Body(name="b", about=host, variables=vars)
end

"Observation variables (offset, jitter, GP hyperparameters) for one instrument."
function gp_obs_variables(kind::Symbol, off::Distribution)
    jit = LogUniform(0.1, 100)
    if kind === :abstractgps
        p1 = truncated(Normal(25, 10), lower=0.1, upper=100)
        p2 = truncated(Normal(GP_EXPLENGTH_MEAN, GP_EXPLENGTH_UNC), lower=5, upper=100)
        p3 = truncated(Normal(GP_PER_MEAN, 1), lower=2, upper=100)
        p4 = truncated(Normal(GP_PERLENGTH_MEAN, GP_PERLENGTH_UNC), lower=0.2, upper=10)
        return @eval @variables begin
            offset ~ $off            # [m/s]
            jitter ~ $jit            # [m/s]
            η_1 ~ $p1                # GP amplitude [m/s]
            η_2 ~ $p2                # exponential decay length [d]
            η_3 ~ $p3                # periodic (rotation) period [d]
            η_4 ~ $p4                # periodic length scale
        end
    elseif kind === :celerite
        pB = Uniform(0.00001, 2000000)
        pC = Uniform(0.00001, 200)
        pL = Uniform(2, 200)
        pP = Uniform(8.5, 20)
        return @eval @variables begin
            offset ~ $off
            jitter ~ $jit
            B ~ $pB
            C ~ $pC
            L ~ $pL                  # decay length [d]
            Prot ~ $pP               # rotation period [d]
        end
    end
    error("kind must be :abstractgps or :celerite, got $kind")
end

"""
    build_gp_system(kind, data)

Fresh star, planet and one GP-enabled RadialVelocityObs per telescope.
`kind` is :abstractgps or :celerite.
"""
function build_gp_system(kind::Symbol, data)
    A = build_star()
    b = build_planet(A)
    gp = kind === :abstractgps ? quasiperiodic : quasistatic
    obs = Any[]
    for tel in data.tels
        sel = String.(data.raw.tel) .== tel
        off = get(OFFSET_PRIORS, tel, Normal(Statistics.median(data.dat.rv[sel]), 100))
        vars = gp_obs_variables(kind, off)
        dbg("  obs '", tel, "': ", count(sel), " RVs, GP = ", gp, ", offset ~ ", off)
        push!(obs, RadialVelocityObs(data.dat[sel, :]; target=A, ref=Barycentre, name=tel,
                                     gaussian_process=gp, variables=vars))
    end
    name = kind === :abstractgps ? "k2_131" : "k2_131_celerite"
    dbg("  system '", name, "': ", length(obs), " GP-enabled RV instruments")
    return System(name=name, bodies=[A, b], observations=obs)
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_rvgp(; models=(:abstractgps, :celerite), n_rounds=7, compare_nogp=true,
               outdir=SCRIPT_DIR, prefix="rvgp_", dark=true, corner_dark=false,
               show_plots=true, force_download=false)

Fit K2-131 with GP activity models using Pigeons. Returns a NamedTuple.
"""
function run_rvgp(; models=(:abstractgps, :celerite),
                    n_rounds::Integer=7,
                    compare_nogp::Bool=true,
                    outdir::AbstractString=SCRIPT_DIR,
                    prefix::AbstractString="rvgp_",
                    dark::Bool=true,
                    corner_dark::Bool=false,
                    show_plots::Bool=true,
                    force_download::Bool=false)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    print_env_info(outdir, models, n_rounds)
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

    data = stage("Load K2-131 RVs") do
        load_k2_131(; cache_dir=outdir, force=force_download)
    end

    systems = Dict{Symbol,Any}(); modelsd = Dict{Symbol,Any}()
    chains  = Dict{Symbol,Any}(); logZ = Dict{Symbol,Float64}()

    for kind in models
        tag = string(kind)
        sys = stage("[$tag] Build system") do
            build_gp_system(kind, data)
        end
        DEBUG[] && display(sys)
        systems[kind] = sys

        model = stage("[$tag] Compile LogDensityModel" *
                      (kind === :celerite ? " (autodiff = AutoFiniteDiff)" : "")) do
            kind === :celerite ? Octofitter.LogDensityModel(sys, autodiff=AutoFiniteDiff()) :
                                 Octofitter.LogDensityModel(sys)
        end
        DEBUG[] && display(model)
        modelsd[kind] = model

        init = stage("[$tag] Initialize starting points (no guesses)") do
            initialize!(model)
        end
        soft_stage("[$tag] Plot initial guess") do
            savefig!(inout(() -> octoplot(model, init)), "$(tag)_init.png")
        end

        chain = stage("[$tag] Pigeons parallel tempering (n_rounds=$n_rounds → $(2^n_rounds) scans)") do
            c, pt = octofit_pigeons(model; n_rounds=n_rounds)
            z = soft_stage("[$tag] Log-evidence (stepping stone)") do
                zz = Pigeons.stepping_stone(pt)
                dbg(@sprintf("  [%s] log Z ≈ %.2f, Λ = %.2f", tag, zz, Pigeons.global_barrier(pt)))
                zz
            end
            z === nothing || (logZ[kind] = z)
            c
        end
        chains[kind] = chain
        dbg("  [$tag] chain size = ", size(chain))
        print_summary(chain, tag)

        stage("[$tag] rvplot (single draw; GP conditioned on its residuals)") do
            savefig!(inout(() -> rvplot(model, chain)), "$(tag)_rvplot.png")
        end
        soft_stage("[$tag] octoplot (many draws, each with its own conditioned GP)") do
            savefig!(inout(() -> octoplot(model, chain)), "$(tag)_octoplot.png")
        end
        soft_stage("[$tag] octoplot(N=50, figscale=1.5)") do
            savefig!(inout(() -> octoplot(model, chain; N=50, figscale=1.5)), "$(tag)_octoplot_N50.png")
        end
        soft_stage("[$tag] octoplot(show_phase=true) — Keplerian signal folded") do
            savefig!(inout(() -> octoplot(model, chain; show_phase=true)), "$(tag)_octoplot_phase.png")
        end
        soft_stage("[$tag] Corner plot") do
            savefig!(inout(() -> corner(model, chain; dark=corner_dark)), "$(tag)_corner.png")
        end

        p = joinpath(outdir, prefix * "$(tag)_chain.fits")
        soft_stage("[$tag] Save chain -> FITS (+ reload check)") do
            Octofitter.savechain(p, chain)
            push!(outputs, p)
            c2 = Octofitter.loadchain(p; model)
            dbg("  reloaded chain size = ", size(c2),
                size(c2) == size(chain) ? "  (matches)" : "  (DIFFERS from saved)")
        end
    end

    # ---- Comparison with the no-GP fit and between GP backends --------------
    pairs = Pair{Symbol,Any}[]
    if compare_nogp
        nogp = soft_stage("Load no-GP chain from OctofitterRV1 ($NOGP_CHAIN)") do
            p = joinpath(outdir, NOGP_CHAIN)
            isfile(p) || error("$p not found — run OctofitterRV1.jl first to include it")
            Octofitter.loadchain(p)
        end
        nogp === nothing || push!(pairs, :nogp => nogp)
    end
    for k in models
        haskey(chains, k) && push!(pairs, k => chains[k])
    end
    if length(pairs) > 1
        print_comparison(pairs)
        soft_stage("Comparison plot (K, m sin i)") do
            savefig!(comparison_figure(pairs), "compare.png")
        end
    end
    if haskey(logZ, :abstractgps) && haskey(logZ, :celerite)
        dbg(@sprintf("  log Z(AbstractGPs) − log Z(Celerite) = %.2f  (different kernels AND priors; ", 
                     logZ[:abstractgps] - logZ[:celerite]),
            "short Pigeons runs make this rough)")
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f s", time() - T0[]))
    return (; data, systems, models=modelsd, chains, logZ, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default for this module when the REPL has 1 thread)
# -----------------------------------------------------------------------------
"""
    relaunch_threaded(file; threads="auto", outdir=SCRIPT_DIR, prefix="rvgp_")

Run `file` in a child Julia with `--threads=threads` and the same active
project (OCTOGP_* settings are inherited), stream its output, then load the
FITS chains it wrote.
"""
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString="rvgp_")
    T0[] = time()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOGP_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOGP_CHILD=1]")
    dbg("  Child output follows. Plots go to files only (not the plot pane).")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOGP_RELAUNCH\"]=\"false\".")
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
    chains = Dict{Symbol,Any}()
    for kind in (:abstractgps, :celerite)
        p = joinpath(outdir, prefix * "$(kind)_chain.fits")
        (isfile(p) && mtime(p) >= t) || continue
        c = soft_stage("Load $(kind) chain written by child") do
            Octofitter.loadchain(p)
        end
        c === nothing || (chains[kind] = c)
    end
    new_files = [joinpath(outdir, f) for f in readdir(outdir)
                 if startswith(f, prefix) && mtime(joinpath(outdir, f)) >= t]
    list_outputs(new_files)
    return (; relaunched=true, success=ok, chains, outputs=new_files)
end

"Parse OCTOGP_MODELS (comma-separated) into a tuple of Symbols."
function models_from_env()
    s = get(ENV, "OCTOGP_MODELS", "abstractgps,celerite")
    ms = Tuple(Symbol(strip(lowercase(x))) for x in split(s, ',') if !isempty(strip(x)))
    all(in((:abstractgps, :celerite)), ms) ||
        error("OCTOGP_MODELS must list abstractgps and/or celerite, got \"$s\"")
    return ms
end

println(@sprintf("[OGP] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterRVGP

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterRVGP.jl`      -> runs
#   - VSCode "Execute active File in REPL"          -> runs; with 1 thread it
#     relaunches in `julia --threads=auto` (OCTOGP_RELAUNCH=false to disable,
#     OCTOGP_THREADS to set the count)
#   - ENV["OCTOGP_AUTORUN"] = "false"; include()    -> loads module only
#   - OCTOGP_MODELS, OCTOGP_ROUNDS select models / Pigeons rounds
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOGP_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOGP_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOGP_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global rvgp_result = OctofitterRVGP.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOGP_THREADS", "auto"))
            println("[OGP] Child run finished. `rvgp_result` has fields: relaunched, success, ",
                    "chains (loaded from FITS), outputs")
        else
            global rvgp_result = OctofitterRVGP.run_rvgp(
                models=OctofitterRVGP.models_from_env(),
                n_rounds=parse(Int, get(ENV, "OCTOGP_ROUNDS", "7")),
                show_plots=!is_child)
            is_child || println("[OGP] Result stored in `rvgp_result` (fields: data, systems, ",
                "models, chains, logZ, outputs)")
        end
    end
end

nothing
