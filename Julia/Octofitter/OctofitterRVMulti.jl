#=
================================================================================
 OctofitterRVMulti.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Multi-Planet RV Fits":
   https://sefffal.github.io/Octofitter.jl/dev/rv-multi-planet/

 Data: simulated with PlanetOrbits exactly as on the page (Random.seed!(1)):
   star 1 M⊙; b: 0.25e-3 M⊙, a = 1 AU, e = 0.05; c: 1.0e-3 M⊙, a = 5 AU,
   e = 0.40 (Jacobi, about A+b); ω = π/4, i = 90°. Two instruments ("DATA 1",
   "DATA 2", the latter with a +7 m/s zero point), each a MarginalizedRVObs
   (zero point marginalised analytically) with its own jitter.

 Models (all sampled with Pigeons, which also gives the log evidence):
   :p1     one planet (b only)
   :p2     two planets, identical broad priors (labels can swap)
   :p2v2   two planets, P = P_nom · P_ratio with P_ratio_b ∈ (0, 0.5),
           P_ratio_c ∈ (0.5, 1) so c is always the outer planet
 Plus: prior-only runs for log Z0 (normalisation check), evidence table with
 the page's interpretation scale, label-switching histogram for :p2, MAP-draw
 rvplot and period-ratio histogram for :p2v2, posterior vs simulation truth.

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `rvmulti_result`)
   Shell:    julia --threads=auto OctofitterRVMulti.jl
   Library:  ENV["OCTOMP_AUTORUN"] = "false"; include("OctofitterRVMulti.jl")
             res = OctofitterRVMulti.run_rvmulti(models=(:p1, :p2v2), n_rounds=9)

 Threads: every fit uses Pigeons. In a 1-thread REPL the run is relaunched in
 a child `julia --threads=auto` (output streamed here; chains loaded back
 afterwards). OCTOMP_RELAUNCH=false disables this; OCTOMP_THREADS sets the
 count. (The page's `cores=8` option uses MPI workers; not used here.)

 Convenience ENV settings (also passed to a relaunched child):
   OCTOMP_MODELS = "p1,p2,p2v2" (default)   OCTOMP_ROUNDS = "10" (default)
   OCTOMP_PRIOR_Z0 = "true" (default) — prior-only evidence runs

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOMP_ENV_MODE=local uses ./octofitter_v9_env). Needs Octofitter
 v9, OctofitterRadialVelocity v9, CairoMakie, PairPlots, Distributions,
 PlanetOrbits, Pigeons.

 Outputs (prefix "rvmulti_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-24  Initial version on the OctofitterRVGP v1.0.1 scaffolding
                   ([OMP +t] debug stages, soft optional stages, env bootstrap
                   with v9 guards, @__DIR__ outputs, dark theme with light
                   corner plots, default threaded relaunch). Adds: seeded
                   PlanetOrbits simulation; MarginalizedRVObs; 1p/2p/2p-v2
                   models; evidence table with interpretation; prior-only Z0
                   check; label-switch diagnostics; MAP rvplot; period ratio;
                   posterior-vs-truth table.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "OctofitterRadialVelocity", "CairoMakie", "PairPlots",
                "Distributions", "PlanetOrbits", "Pigeons"),
    min_version = Dict("Octofitter" => v"9", "OctofitterRadialVelocity" => v"9"),
    mode = lowercase(get(ENV, "OCTOMP_ENV_MODE", "temp"))

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
            println("[OMP] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[OMP] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[OMP] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[OMP] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[OMP] Existing v9 environment found at ", Base.active_project())
        else
            Pkg.add(spec.(collect(required)))
        end
        println("[OMP] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ", OctofitterRadialVelocity v",
                direct_deps()["OctofitterRadialVelocity"], ")")
    end
    flush(stdout)
end

println("[OMP] Loading Octofitter, OctofitterRadialVelocity, PlanetOrbits, Pigeons, ",
        "CairoMakie, PairPlots, Distributions (first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterRVMulti

const LOAD_T0 = time()

using Octofitter
using OctofitterRadialVelocity
using CairoMakie
using PairPlots
using Distributions
using PlanetOrbits
using Pigeons
import Random
import Statistics
using Printf

export run_rvmulti, simulate_data, build_rv_system, set_plot_theme!

const VERSION_STRING = "1.0"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
const ORV_VERSION = pkgversion(OctofitterRadialVelocity)
if (OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9") ||
   (ORV_VERSION !== nothing && ORV_VERSION < v"9")
    error("""
    OctofitterRVMulti needs Octofitter v9+ and OctofitterRadialVelocity v9+, but the
    active environment has Octofitter v$(OCTOFITTER_VERSION) and
    OctofitterRadialVelocity v$(ORV_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const TP_REF_MJD = 58400.0

# Simulation truth (as on the tutorial page)
const TRUTH = (
    M_A = 1.0,
    b = (mass = 0.25e-3, a = 1.0, e = 0.05, ω = pi / 4, tp = 58800.0),
    c = (mass = 1.0e-3,  a = 5.0, e = 0.40, ω = pi / 4, tp = 59800.0),
)
"Kepler period [yr] for a [AU] about total mass M [M⊙]."
kepler_P_yr(a, M) = sqrt(a^3 / M)
truth_P_yr(which) = which === :b ?
    kepler_P_yr(TRUTH.b.a, TRUTH.M_A + TRUTH.b.mass) :
    kepler_P_yr(TRUTH.c.a, TRUTH.M_A + TRUTH.b.mass + TRUTH.c.mass)   # Jacobi

const MODEL_LABELS = Dict(:p1 => "1 planet", :p2 => "2 planets (symmetric priors)",
                          :p2v2 => "2 planets (ordered periods)")
const MODEL_ORDER = (:p1, :p2, :p2v2)

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterRVMulti.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[OMP +%7.1fs] ", time() - T0[]), msg...)
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
    dbg("OctofitterRVMulti v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | OctofitterRadialVelocity ", something(ORV_VERSION, "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("models = ", join(models, ", "), " | n_rounds = ", n_rounds,
        " (", 2^n_rounds, " scans → ", 2^n_rounds, " posterior samples each)")
    Threads.nthreads() == 1 &&
        dbg("  NOTE: running Pigeons on 1 thread (relaunch disabled) — expect a long run")
    dbg("project    = ", Base.active_project())
    dbg("SCRIPT_DIR = ", SCRIPT_DIR, " | outdir = ", abspath(outdir))
    dbg("interactive = ", isinteractive(),
        " | Makie backend = ", CairoMakie.Makie.current_backend())
    return nothing
end

colvec(chain, p::Symbol) = vec(Array(chain[p]))
haspar(chain, p::Symbol) = p in names(chain)
qline(x) = Statistics.quantile(filter(isfinite, x), (0.16, 0.5, 0.84))

"Rows: (label, column, transform, truth or nothing)."
function summary_rows()
    rows = Any[]
    for (pl, t) in ((:b, TRUTH.b), (:c, TRUTH.c))
        push!(rows, ("$(pl)_P [yr]",        Symbol(pl, "_P"),    x -> x ./ year2day_julian, truth_P_yr(pl)))
        push!(rows, ("$(pl)_e",             Symbol(pl, "_e"),    identity,                  t.e))
        push!(rows, ("$(pl)_ω [rad]",       Symbol(pl, "_ω"),    identity,                  t.ω))
        push!(rows, ("$(pl)_mass [M_jup]",  Symbol(pl, "_mass"), x -> x ./ mjup,            t.mass / mjup))
    end
    push!(rows, ("P_nom [yr]", :P_nom,     x -> x ./ year2day_julian, nothing))
    push!(rows, ("P_ratio_b",  :P_ratio_b, identity, nothing))
    push!(rows, ("P_ratio_c",  :P_ratio_c, identity, nothing))
    return rows
end

function print_summary(chain, label)
    DEBUG[] || return nothing
    dbg("Posterior median [16th, 84th] — ", label, " (", size(chain, 1), " samples):")
    for (lab, p, f, tru) in summary_rows()
        haspar(chain, p) || continue
        q = qline(f(colvec(chain, p)))
        dbg(@sprintf("    %-20s %12.4f  [%12.4f, %12.4f]%s", lab, q[2], q[1], q[3],
                     tru === nothing ? "" : @sprintf("   truth %.4f", tru)))
    end
    for p in names(chain)
        occursin(r"_jitter$", string(p)) || continue
        q = qline(colvec(chain, p))
        dbg(@sprintf("    %-20s %12.4f  [%12.4f, %12.4f]", string(p, " [m/s]"), q[2], q[1], q[3]))
    end
    if haspar(chain, :b_P) && haspar(chain, :c_P)
        Pb, Pc = colvec(chain, :b_P), colvec(chain, :c_P)
        dbg(@sprintf("    fraction of samples with P_b > P_c (label swap): %.1f%%",
                     100 * Statistics.mean(Pb .> Pc)))
        q = qline(max.(Pb, Pc) ./ min.(Pb, Pc))
        dbg(@sprintf("    outer/inner period ratio: %.3f [%.3f, %.3f]   truth %.3f", q[2], q[1],
                     q[3], truth_P_yr(:c) / truth_P_yr(:b)))
    end
    return nothing
end

"Interpretation of ln BF (the scale printed on the tutorial page)."
function interpret(lnbf)
    a = abs(lnbf)
    s = a > 3.00 ? "extreme" : a > 1.61 ? "very strong" : a > 1.10 ? "strong" :
        a > 0.69 ? "moderate" : a > 0 ? "anecdotal" : "no"
    return lnbf == 0 ? "no evidence" : string(s, " evidence for ", lnbf > 0 ? "H_A" : "H_B")
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

function data_figure(sim)
    fig = Figure(size=(1000, 450))
    ax = Axis(fig[1, 1], xlabel="epoch [MJD]", ylabel="RV [m/s]",
              title="Simulated RVs (b: 1 AU, $(round(TRUTH.b.mass / mjup, digits=2)) M_jup; " *
                    "c: 5 AU, $(round(TRUTH.c.mass / mjup, digits=2)) M_jup)")
    for (t, col, lab) in ((sim.t1, C1, "DATA 1"), (sim.t2, C2, "DATA 2 (+7 m/s zero point)"))
        errorbars!(ax, t.epoch, t.rv, t.σ_rv, color=(col, 0.5))
        scatter!(ax, t.epoch, t.rv, color=col, markersize=7, label=lab)
    end
    ep = range(minimum(sim.t1.epoch), maximum(sim.t2.epoch), length=1500)
    lines!(ax, ep, reflex_rv(sim.truth, ep), color=(:white, 0.5), label="true reflex RV")
    axislegend(ax, position=:lb)
    return fig
end

function period_hist_figure(chain; title="Planet b period (symmetric priors: labels swap)")
    fig = Figure(size=(900, 380))
    ax = Axis(fig[1, 1], xlabel="b_P [yr]", ylabel="count", title=title)
    hist!(ax, colvec(chain, :b_P) ./ year2day_julian; bins=100, color=(C1, 0.8))
    vlines!(ax, [truth_P_yr(:b), truth_P_yr(:c)], color=(:white, 0.6), linestyle=:dash)
    return fig
end

function period_ratio_figure(chain)
    r = colvec(chain, :c_P) ./ colvec(chain, :b_P)
    fig = Figure(size=(900, 380))
    ax = Axis(fig[1, 1], xlabel="period ratio P_c / P_b", ylabel="density",
              title="Period ratio (ordered-period model)")
    hist!(ax, r; bins=50, normalization=:pdf, color=(C2, 0.8))
    vlines!(ax, [truth_P_yr(:c) / truth_P_yr(:b)], color=(:white, 0.7), linestyle=:dash)
    return fig
end

# -----------------------------------------------------------------------------
# Simulation (as on the tutorial page)
# -----------------------------------------------------------------------------
"The simulation-truth PlanetOrbits.System."
function truth_system()
    star = PlanetOrbits.Body(mass=TRUTH.M_A,    name=:A)
    p_b  = PlanetOrbits.Body(mass=TRUTH.b.mass, name=:b)
    p_c  = PlanetOrbits.Body(mass=TRUTH.c.mass, name=:c)
    return PlanetOrbits.System(
        (star, p_b, p_c),
        (
            PlanetOrbits.Orbit(p_b, about=star;        a=TRUTH.b.a, e=TRUTH.b.e, ω=TRUTH.b.ω,
                               i=pi / 2, Ω=0.0, tp=TRUTH.b.tp),
            PlanetOrbits.Orbit(p_c, about=(star, p_b); a=TRUTH.c.a, e=TRUTH.c.e, ω=TRUTH.c.ω,
                               i=pi / 2, Ω=0.0, tp=TRUTH.c.tp),
        )
    )
end

"Star's reflex velocity against the barycentre [m/s]."
function reflex_rv(truth, epochs)
    traj = orbitsolve(truth, collect(epochs))
    return [radvel(traj[k], :A, barycentre(truth)) for k in eachindex(epochs)]
end

"""
    simulate_data(; seed=1)

Reproduce the tutorial's simulated two-instrument data set. The RNG calls
follow the page's order, so with the same seed and Julia version the data
should match the documentation's.
"""
function simulate_data(; seed::Integer=1)
    Random.seed!(seed)
    truth = truth_system()
    epochs1 = (58400:150:69400) .+ 10 .* randn.()
    rv1 = reflex_rv(truth, epochs1)
    t1 = Table(epoch=epochs1, rv=rv1 .+ 4 .* randn.(),
               σ_rv=[4 * abs(randn()) + 1 for _ in eachindex(epochs1)])
    epochs2 = (65400:100:71400) .+ 10 .* randn.()
    rv2 = reflex_rv(truth, epochs2)
    t2 = Table(epoch=epochs2, rv=rv2 .+ 2 .* randn.() .+ 7,
               σ_rv=[2 * abs(randn()) + 1 for _ in eachindex(epochs2)])
    dbg(@sprintf("  DATA 1: %d epochs, MJD %.0f–%.0f, median σ %.2f m/s", length(t1.epoch),
                 minimum(t1.epoch), maximum(t1.epoch), Statistics.median(t1.σ_rv)))
    dbg(@sprintf("  DATA 2: %d epochs, MJD %.0f–%.0f, median σ %.2f m/s (+7 m/s zero point)",
                 length(t2.epoch), minimum(t2.epoch), maximum(t2.epoch), Statistics.median(t2.σ_rv)))
    dbg(@sprintf("  truth: P_b = %.3f yr, P_c = %.3f yr (ratio %.3f); m_b = %.3f, m_c = %.3f M_jup",
                 truth_P_yr(:b), truth_P_yr(:c), truth_P_yr(:c) / truth_P_yr(:b),
                 TRUTH.b.mass / mjup, TRUTH.c.mass / mjup))
    return (; truth, t1, t2)
end

# -----------------------------------------------------------------------------
# Model components (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
function rv_observations(sim)
    jit = LogUniform(0.1, 100)
    obs = Any[]
    for (t, name) in ((sim.t1, "DATA 1"), (sim.t2, "DATA 2"))
        vars = @eval @variables begin
            jitter ~ $jit            # [m/s]; zero point marginalised analytically
        end
        push!(obs, MarginalizedRVObs(t; target=:A, ref=Barycentre, name=name, variables=vars))
    end
    return obs
end

"Planet body. `ordered=true` takes P from system.P_nom · system.P_ratio_<name>."
function rv_planet(name::AbstractString, about; ordered::Bool)
    mp = Uniform(0, 10mjup)
    Pp = Uniform(0, 100year2day_julian)
    ratio = Symbol("P_ratio_", name)
    tr = TP_REF_MJD
    vars = if ordered
        @eval @variables begin
            i = pi / 2
            Ω = 0.0
            e ~ Uniform(0, 0.999999)
            ω ~ Uniform(0, 2pi)
            mass ~ $mp
            P = system.P_nom * system.$ratio
            τ ~ Uniform(0, 1.0)
            tp = τ * P + $tr
        end
    else
        @eval @variables begin
            i = pi / 2
            Ω = 0.0
            e ~ Uniform(0, 0.999999)
            ω ~ Uniform(0, 2pi)
            mass ~ $mp
            P ~ $Pp
            τ ~ Uniform(0, 1.0)
            tp = τ * P + $tr
        end
    end
    dbg("  planet '", name, "': about ", about isa Tuple ? "A+b barycentre (Jacobi)" : "A",
        ordered ? ", P = P_nom·$(ratio)" : ", P ~ U(0, 100 yr)", ", mass ~ U(0, 10 M_jup)")
    return Body(name=name, about=about, variables=vars)
end

"""
    build_rv_system(kind, sim)

Fresh bodies and observations for model `kind` ∈ (:p1, :p2, :p2v2).
"""
function build_rv_system(kind::Symbol, sim)
    A_vars = @eval @variables begin
        mass = 1.0                   # [M⊙], fixed
    end
    A = Body(name="A", variables=A_vars)
    obs = rv_observations(sim)
    if kind === :p1
        b = rv_planet("b", A; ordered=false)
        return System(name="sim_1p", bodies=[A, b], observations=obs)
    elseif kind === :p2
        b = rv_planet("b", A; ordered=false)
        c = rv_planet("c", (A, b); ordered=false)
        return System(name="sim_2p", bodies=[A, b, c], observations=obs)
    elseif kind === :p2v2
        b = rv_planet("b", A; ordered=true)
        c = rv_planet("c", (A, b); ordered=true)
        Pn = Uniform(0, 100year2day_julian)
        vars = @eval @variables begin
            P_nom ~ $Pn
            P_ratio_b ~ Uniform(0, 0.5)
            P_ratio_c ~ Uniform(0.5, 1)
        end
        dbg("  system: P_nom ~ U(0, 100 yr), P_ratio_b ~ U(0, 0.5), P_ratio_c ~ U(0.5, 1)")
        return System(name="sim_2p_v2", bodies=[A, b, c], observations=obs, variables=vars)
    end
    error("kind must be :p1, :p2 or :p2v2, got $kind")
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_rvmulti(; models=(:p1, :p2, :p2v2), n_rounds=10, prior_z0=true, seed=1,
                  outdir=SCRIPT_DIR, prefix="rvmulti_", dark=true,
                  corner_dark=false, show_plots=true)

Simulate the data, fit each model with Pigeons, compare evidences.
"""
function run_rvmulti(; models=MODEL_ORDER,
                       n_rounds::Integer=10,
                       prior_z0::Bool=true,
                       seed::Integer=1,
                       outdir::AbstractString=SCRIPT_DIR,
                       prefix::AbstractString="rvmulti_",
                       dark::Bool=true,
                       corner_dark::Bool=false,
                       show_plots::Bool=true)
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

    sim = stage("Simulate data (Random.seed!($seed), as on the tutorial page)") do
        simulate_data(; seed)
    end
    soft_stage("Plot simulated data") do
        savefig!(data_figure(sim), "data.png")
    end

    modelsd = Dict{Symbol,Any}(); chains = Dict{Symbol,Any}()
    logZ = Dict{Symbol,Float64}(); logZ0 = Dict{Symbol,Float64}()

    for kind in models
        tag = string(kind)
        sys = stage("[$tag] Build system — $(MODEL_LABELS[kind])") do
            build_rv_system(kind, sim)
        end
        DEBUG[] && display(sys)
        model = stage("[$tag] Compile LogDensityModel") do
            Octofitter.LogDensityModel(sys)
        end
        DEBUG[] && display(model)
        modelsd[kind] = model

        chain = stage("[$tag] Pigeons parallel tempering (n_rounds=$n_rounds → $(2^n_rounds) scans)") do
            c, pt = octofit_pigeons(model; n_rounds=n_rounds)
            z = soft_stage("[$tag] Log-evidence ratio (stepping stone)") do
                zz = Pigeons.stepping_stone(pt)
                dbg(@sprintf("  [%s] log(Z₁/Z₀) ≈ %.3f, Λ = %.2f", tag, zz, Pigeons.global_barrier(pt)))
                zz
            end
            z === nothing || (logZ[kind] = z)
            c
        end
        chains[kind] = chain
        print_summary(chain, MODEL_LABELS[kind])

        if prior_z0
            soft_stage("[$tag] Prior-only run for log Z0") do
                pm = Octofitter.LogDensityModel(Octofitter.prior_only_model(sys, exclude_all=true))
                _, ptp = octofit_pigeons(pm; n_rounds=n_rounds)
                z0 = Pigeons.stepping_stone(ptp)
                logZ0[kind] = z0
                dbg(@sprintf("  [%s] log Z0 ≈ %.4f  (≈0 means the priors are properly normalised)", tag, z0))
            end
        end

        stage("[$tag] rvplot (one draw)") do
            savefig!(inout(() -> rvplot(model, chain)), "$(tag)_rvplot.png")
        end
        soft_stage("[$tag] octoplot (many draws)") do
            savefig!(inout(() -> octoplot(model, chain)), "$(tag)_octoplot.png")
        end
        soft_stage("[$tag] Corner plot") do
            savefig!(inout(() -> corner(model, chain; dark=corner_dark)), "$(tag)_corner.png")
        end
        if kind === :p2
            soft_stage("[$tag] Histogram of b_P (label switching)") do
                savefig!(period_hist_figure(chain), "$(tag)_bP_hist.png")
            end
        end
        if kind === :p2v2
            soft_stage("[$tag] rvplot of the maximum-a-posteriori draw") do
                i_map = argmax(colvec(chain, :logpost))
                dbg("  MAP draw index = ", i_map, @sprintf(", logpost = %.3f", colvec(chain, :logpost)[i_map]))
                savefig!(inout(() -> rvplot(model, chain, i_map)), "$(tag)_rvplot_map.png")
            end
            soft_stage("[$tag] Period-ratio histogram") do
                savefig!(period_ratio_figure(chain), "$(tag)_period_ratio.png")
            end
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

    # ---- Evidence comparison ------------------------------------------------
    if !isempty(logZ)
        dbg("Log evidence (stepping stone):")
        for k in MODEL_ORDER
            haskey(logZ, k) || continue
            z0 = get(logZ0, k, NaN)
            dbg(@sprintf("    %-30s log(Z/Z0) = %10.3f   log Z0 = %8.4f   log Z = %10.3f",
                         MODEL_LABELS[k], logZ[k], z0, isnan(z0) ? logZ[k] : logZ[k] - z0))
        end
        for (a, b) in ((:p2, :p1), (:p2v2, :p1), (:p2v2, :p2))
            (haskey(logZ, a) && haskey(logZ, b)) || continue
            lnbf = logZ[a] - logZ[b]
            dbg(@sprintf("    ln BF(%s vs %s) = %8.3f  → %s (H_A = %s)", a, b, lnbf,
                         interpret(lnbf), MODEL_LABELS[a]))
        end
        dbg("    (docs: log Z ≈ -461.76, -411.31, -409.78 for 1p, 2p, 2p_v2)")
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f s", time() - T0[]))
    return (; sim, models=modelsd, chains, logZ, logZ0, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default for this module when the REPL has 1 thread)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString="rvmulti_")
    T0[] = time()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOMP_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOMP_CHILD=1]")
    dbg("  Child output follows. Plots go to files only (not the plot pane).")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOMP_RELAUNCH\"]=\"false\".")
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
    for kind in MODEL_ORDER
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

"Parse OCTOMP_MODELS (comma-separated) into a tuple of Symbols in canonical order."
function models_from_env()
    s = get(ENV, "OCTOMP_MODELS", "p1,p2,p2v2")
    req = Set(Symbol(strip(lowercase(x))) for x in split(s, ',') if !isempty(strip(x)))
    issubset(req, Set(MODEL_ORDER)) ||
        error("OCTOMP_MODELS must list p1, p2 and/or p2v2, got \"$s\"")
    return Tuple(k for k in MODEL_ORDER if k in req)
end

println(@sprintf("[OMP] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterRVMulti

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterRVMulti.jl`   -> runs
#   - VSCode "Execute active File in REPL"          -> runs; with 1 thread it
#     relaunches in `julia --threads=auto` (OCTOMP_RELAUNCH=false to disable,
#     OCTOMP_THREADS to set the count)
#   - ENV["OCTOMP_AUTORUN"] = "false"; include()    -> loads module only
#   - OCTOMP_MODELS, OCTOMP_ROUNDS, OCTOMP_PRIOR_Z0 select what runs
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOMP_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOMP_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOMP_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global rvmulti_result = OctofitterRVMulti.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOMP_THREADS", "auto"))
            println("[OMP] Child run finished. `rvmulti_result` has fields: relaunched, success, ",
                    "chains (loaded from FITS), outputs")
        else
            global rvmulti_result = OctofitterRVMulti.run_rvmulti(
                models=OctofitterRVMulti.models_from_env(),
                n_rounds=parse(Int, get(ENV, "OCTOMP_ROUNDS", "10")),
                prior_z0=lowercase(get(ENV, "OCTOMP_PRIOR_Z0", "true")) != "false",
                show_plots=!is_child)
            is_child || println("[OMP] Result stored in `rvmulti_result` (fields: sim, models, ",
                "chains, logZ, logZ0, outputs)")
        end
    end
end

nothing
