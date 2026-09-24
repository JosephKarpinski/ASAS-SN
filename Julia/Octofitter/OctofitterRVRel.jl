#=
================================================================================
 OctofitterRVRel.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Fit Relative RV Data":
   https://sefffal.github.io/Octofitter.jl/dev/fit-rv-rel/

 Relative RV = velocity of the companion minus velocity of the host
 (positive = companion receding faster). Same RadialVelocityObs type as for
 stellar reflex RVs; only `ref` changes:
     RadialVelocityObs(tab; target=A, ref=Barycentre)   # star's reflex motion
     RadialVelocityObs(tab; target=b, ref=A)            # companion vs star  <- here
 No instrument zero point is needed (spectra are differenced).

 Data: the page's 25 simulated relative RVs, MJD 55000–57400 every 100 d,
 σ = 15 km/s. Model: star A (mass ~ truncated Normal(1.2, 0.1)); companion b
 as a test particle (mass = 0) with a ~ U(0, 10) AU, e ~ U(0, 0.5), i ~ Sine(),
 ω, Ω, M0 ~ U(0, 2π), epoch = 60000; jitter ~ LogUniform(0.1, 1000) m/s.

 Pipeline:
   data plot -> System -> LogDensityModel -> initialize! -> octoplot(init)
   -> octofit(Xoshiro(123), model) (as in the tutorial) -> diagnostics
   -> summary incl. derived P (Kepler), K_rel and a·sin i
   -> octoplot, rvplot, corner, posterior-predictive curves from
      construct_system/orbitsolve/radvel(:b, :A), data folded on median P
   -> FITS chain

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `rvrel_result`)
   Shell:    julia OctofitterRVRel.jl
   Library:  ENV["OCTORR_AUTORUN"] = "false"; include("OctofitterRVRel.jl")
             res = OctofitterRVRel.run_rvrel(iterations=2000, seed=1)

 Threads: the tutorial's HMC fit is one chain (~4 s), so it runs in the REPL.
 ENV["OCTORR_SAMPLER"] = "pigeons" switches to parallel tempering (useful if
 you suspect period aliases); in a 1-thread REPL that relaunches in a child
 `julia --threads=auto` (OCTORR_RELAUNCH, OCTORR_THREADS control this).

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTORR_ENV_MODE=local uses ./octofitter_v9_env). Needs Octofitter
 v9, Distributions, CairoMakie, PairPlots, PlanetOrbits, Pigeons, MCMCChains.

 Outputs (prefix "rvrel_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-24  Initial version on the OctofitterRVMulti v1.0.1
                   scaffolding ([ORR +t] debug stages, soft optional stages,
                   env bootstrap with v9 guard, @__DIR__ outputs, dark theme
                   with light corner plots, initialize! before Pigeons,
                   optional threaded Pigeons relaunch). Adds: relative-RV
                   observation (target=b, ref=A); derived P, K_rel, a·sin i;
                   posterior-predictive relative-RV curves; folded data plot.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "Distributions", "CairoMakie", "PairPlots",
                "PlanetOrbits", "Pigeons", "MCMCChains"),
    mode = lowercase(get(ENV, "OCTORR_ENV_MODE", "temp"))

    direct_deps() = Dict(info.name => info.version
                         for info in values(Pkg.dependencies()) if info.is_direct_dep)
    missing_deps() = [n for n in required if !haskey(direct_deps(), n)]
    function env_ok()
        v = get(direct_deps(), "Octofitter", nothing)
        v !== nothing && v >= v"9" && isempty(missing_deps())
    end
    spec(n) = n == "Octofitter" ? Pkg.PackageSpec(name=n, version="9") : Pkg.PackageSpec(name=n)

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
            println("[ORR] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[ORR] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[ORR] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[ORR] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[ORR] Existing v9 environment found at ", Base.active_project())
        else
            Pkg.add(spec.(collect(required)))
        end
        println("[ORR] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[ORR] Loading Octofitter, Distributions, CairoMakie, PairPlots, PlanetOrbits, ",
        "Pigeons, MCMCChains (first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterRVRel

const LOAD_T0 = time()

using Octofitter
using Distributions
using CairoMakie
using PairPlots
using PlanetOrbits
using Pigeons
import MCMCChains
import Random
import Statistics
using Printf

export run_rvrel, relative_rv_table, build_rvrel_system, set_plot_theme!

const VERSION_STRING = "1.0"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterRVRel needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up Octofitter v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const AU_M = 1.495978707e11
const DAYS_PER_YEAR = 365.25

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterRVRel.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[ORR +%7.1fs] ", time() - T0[]), msg...)
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
    dbg("OctofitterRVRel v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | sampler ", sampler, " | threads ", Threads.nthreads())
    dbg("project    = ", Base.active_project())
    dbg("SCRIPT_DIR = ", SCRIPT_DIR, " | outdir = ", abspath(outdir))
    dbg("interactive = ", isinteractive(),
        " | Makie backend = ", CairoMakie.Makie.current_backend())
    return nothing
end

colvec(chain, p::Symbol) = vec(Array(chain[p]))
haspar(chain, p::Symbol) = p in names(chain)
qline(x) = Statistics.quantile(filter(isfinite, x), (0.16, 0.5, 0.84))

"Orbital period [days] per sample (test particle: M = host mass)."
period_days(chain) = sqrt.(colvec(chain, :b_a) .^ 3 ./ colvec(chain, :A_mass)) .* DAYS_PER_YEAR

"""
    k_rel(chain)

Relative-RV semi-amplitude [m/s] per sample:
K_rel = 2π a sin i / (P √(1 − e²)) — what the data constrain directly.
"""
function k_rel(chain)
    a = colvec(chain, :b_a) .* AU_M
    P = period_days(chain) .* 86400
    sini = sin.(colvec(chain, :b_i))
    e = colvec(chain, :b_e)
    return 2π .* a .* sini ./ (P .* sqrt.(1 .- e .^ 2))
end

function print_summary(chain, label)
    DEBUG[] || return nothing
    dbg("Posterior median [16th, 84th] — ", label, " (", size(chain, 1), " samples):")
    row(lab, x) = (q = qline(x);
        dbg(@sprintf("    %-24s %12.4f  [%12.4f, %12.4f]", lab, q[2], q[1], q[3])))
    row("A_mass [M⊙]", colvec(chain, :A_mass))
    row("b_a [AU]", colvec(chain, :b_a))
    row("b_e", colvec(chain, :b_e))
    row("b_i [deg]", rad2deg.(colvec(chain, :b_i)))
    row("b_ω [deg]", rad2deg.(colvec(chain, :b_ω)))
    row("b_Ω [deg]  (RV-insensitive)", rad2deg.(colvec(chain, :b_Ω)))
    row("b_M0 [deg]", rad2deg.(colvec(chain, :b_M0)))
    row("P [d] (derived)", period_days(chain))
    row("a·sin i [AU] (derived)", colvec(chain, :b_a) .* sin.(colvec(chain, :b_i)))
    row("K_rel [km/s] (derived)", k_rel(chain) ./ 1000)
    for p in names(chain)
        occursin(r"_jitter$", string(p)) && row(string(p, " [m/s]"), colvec(chain, p))
    end
    return nothing
end

function print_diagnostics(chain)
    DEBUG[] || return nothing
    try
        n = Int(sum(colvec(chain, :numerical_error)))
        dbg(@sprintf("  divergences %d / %d (%.1f%%), mean acceptance %.3f", n, size(chain, 1),
                     100n / size(chain, 1), Statistics.mean(colvec(chain, :acceptance_rate))))
    catch
        dbg("  (no HMC diagnostics columns — e.g. a Pigeons chain)")
    end
    try
        ss = MCMCChains.summarystats(chain)
        bad = [(p, r) for (p, r) in zip(ss[:, :parameters], ss[:, :rhat])
               if isfinite(r) && abs(r - 1) > 0.01]
        isempty(bad) ? dbg("  All finite R̂ within 1 ± 0.01") :
            foreach(((p, r),) -> dbg(@sprintf("  R̂(%s) = %.4f", p, r)), bad)
    catch
        dbg("  (R̂ not available)")
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

const C1 = "#56B4E9"
const C2 = "#E69F00"

function data_figure(t)
    fig = Figure(size=(1000, 420))
    ax = Axis(fig[1, 1], xlabel="epoch [MJD]", ylabel="relative RV (b − A) [km/s]",
              title="Relative RV data ($(length(t.epoch)) epochs, σ = $(Int(t.σ_rv[1] / 1000)) km/s)")
    errorbars!(ax, t.epoch, t.rv ./ 1000, t.σ_rv ./ 1000, color=(C1, 0.4))
    scatter!(ax, t.epoch, t.rv ./ 1000, color=C1, markersize=9)
    return fig
end

"""
    predictive_figure(model, chain, t; ndraws=40)

Relative-RV curves for `ndraws` posterior draws (construct_system →
orbitsolve → radvel(:b, :A)) over the data, plus the data folded on the
median period.
"""
function predictive_figure(model, chain, t; ndraws::Integer=40)
    n = size(chain, 1)
    idx = unique(round.(Int, range(1, n, length=min(ndraws, n))))
    tgrid = collect(range(minimum(t.epoch) - 100, maximum(t.epoch) + 100, length=1200))
    fig = Figure(size=(1000, 760))
    ax1 = Axis(fig[1, 1], xlabel="epoch [MJD]", ylabel="relative RV [km/s]",
               title="Posterior draws ($(length(idx))) vs data")
    for k in idx
        posys = construct_system(model, chain, k)
        traj = orbitsolve(posys, tgrid)
        rv = [radvel(traj[j], :b, :A) for j in eachindex(tgrid)]
        lines!(ax1, tgrid, rv ./ 1000, color=(C2, 0.15))
    end
    errorbars!(ax1, t.epoch, t.rv ./ 1000, t.σ_rv ./ 1000, color=(C1, 0.4))
    scatter!(ax1, t.epoch, t.rv ./ 1000, color=C1, markersize=8)

    P = Statistics.median(period_days(chain))
    ax2 = Axis(fig[2, 1], xlabel="phase (P = $(round(P, digits=2)) d, median)",
               ylabel="relative RV [km/s]", title="Data folded on the median period")
    ph = mod.(t.epoch .- t.epoch[1], P) ./ P
    errorbars!(ax2, ph, t.rv ./ 1000, t.σ_rv ./ 1000, color=(C1, 0.4))
    scatter!(ax2, ph, t.rv ./ 1000, color=C1, markersize=8)
    return fig
end

# -----------------------------------------------------------------------------
# Data and model (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
"The tutorial's 25 relative RVs [m/s] at MJD 55000:100:57400, σ = 15 km/s."
function relative_rv_table()
    rv = [-24022.74, -18571.33, 14221.56, 26076.89, -459.26, -26319.26, -13430.96,
          19230.96, 23580.26, -6786.28, -27161.78, -7548.58, 23177.95, 19780.94,
          -12738.39, -26503.74, -1249.19, 25844.47, 14888.83, -17986.76, -24381.49,
          5119.22, 27083.2, 9174.18, -22241.45]
    t = Table(epoch=collect(55000.0:100:57400), rv=rv, σ_rv=fill(15000.0, 25))
    dbg(@sprintf("  %d epochs, MJD %.0f–%.0f every 100 d; rv range %.1f to %.1f km/s; σ = %.0f km/s",
                 length(t.epoch), minimum(t.epoch), maximum(t.epoch),
                 minimum(rv) / 1000, maximum(rv) / 1000, t.σ_rv[1] / 1000))
    return t
end

"""
    build_rvrel_system(t; epoch=60000.0)

Star A (mass prior), test-particle companion b, and a relative-RV
observation with `target=b, ref=A`.
"""
function build_rvrel_system(t; epoch::Real=60000.0)
    mA = truncated(Normal(1.2, 0.1), lower=0.1)
    A_vars = @eval @variables begin
        mass ~ $mA                   # [M⊙]; carries the whole gravitating mass
    end
    A = Body(name="A", variables=A_vars)
    dbg("  star 'A': mass ~ ", mA)

    ep = Float64(epoch)
    b_vars = @eval @variables begin
        mass = 0.0                   # test particle
        a ~ Uniform(0, 10)           # [AU]
        e ~ Uniform(0.0, 0.5)
        i ~ Sine()
        ω ~ Uniform(0, 2pi)
        Ω ~ Uniform(0, 2pi)
        M0 ~ Uniform(0, 2pi)         # mean anomaly at `epoch`
        epoch = $ep                  # [MJD]
    end
    b = Body(name="b", about=A, variables=b_vars)
    dbg("  planet 'b': test particle; a ~ U(0, 10) AU, e ~ U(0, 0.5), i ~ Sine(), ",
        "ω, Ω, M0 ~ U(0, 2π), epoch = ", ep)

    jit = LogUniform(0.1, 1000)
    o_vars = @eval @variables begin
        jitter ~ $jit                # [m/s]
    end
    obs = RadialVelocityObs(t; target=b, ref=A, name="simulated data", variables=o_vars)
    dbg("  obs 'simulated data': RELATIVE RV (target=b, ref=A), jitter ~ ", jit,
        "; no zero-point offset needed")
    return System(name="Example_System", bodies=[A, b], observations=[obs])
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_rvrel(; sampler=:hmc, iterations=1000, n_rounds=10, seed=123,
                outdir=SCRIPT_DIR, prefix="rvrel_", dark=true, corner_dark=false,
                show_plots=true)

Run the Relative RV tutorial. Returns a NamedTuple of results.
"""
function run_rvrel(; sampler::Symbol=:hmc,
                     iterations::Integer=1000,
                     n_rounds::Integer=10,
                     seed::Integer=123,
                     outdir::AbstractString=SCRIPT_DIR,
                     prefix::AbstractString="rvrel_",
                     dark::Bool=true,
                     corner_dark::Bool=false,
                     show_plots::Bool=true)
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

    t = stage("Relative RV data (from the tutorial page)") do
        relative_rv_table()
    end
    soft_stage("Plot data") do
        savefig!(data_figure(t), "data.png")
    end

    sys = stage("Build system") do
        build_rvrel_system(t)
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
    print_diagnostics(chain)
    print_summary(chain, "relative RV fit")

    stage("octoplot (posterior)") do
        savefig!(inout(() -> octoplot(model, chain)), "octoplot.png")
    end
    soft_stage("rvplot (one draw)") do
        savefig!(inout(() -> rvplot(model, chain)), "rvplot.png")
    end
    soft_stage("Corner plot") do
        savefig!(inout(() -> corner(model, chain; dark=corner_dark)), "corner.png")
    end
    soft_stage("Posterior-predictive relative-RV curves + folded data") do
        savefig!(predictive_figure(model, chain, t), "predictive.png")
    end

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
    return (; data=t, system=sys, model, init_chain, chain, outputs, chain_path)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (Pigeons sampler in a 1-thread session only)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString="rvrel_")
    T0[] = time()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTORR_CHILD" => "1")
    dbg("Pigeons requested and this session has 1 thread; relaunching with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTORR_CHILD=1]")
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

println(@sprintf("[ORR] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterRVRel

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia OctofitterRVRel.jl`                   -> runs
#   - VSCode "Execute active File in REPL"         -> runs (interactive session)
#   - ENV["OCTORR_AUTORUN"] = "false"; include()   -> loads module only
#   - ENV["OCTORR_SAMPLER"] = "pigeons"            -> parallel tempering; in a
#     1-thread session this relaunches in `julia --threads=auto`
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTORR_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTORR_CHILD", "") == "1",
    sampler   = Symbol(lowercase(get(ENV, "OCTORR_SAMPLER", "hmc"))),
    relaunch  = !is_child && sampler === :pigeons && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTORR_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global rvrel_result = OctofitterRVRel.relaunch_threaded(
                this_file; threads=get(ENV, "OCTORR_THREADS", "auto"))
            println("[ORR] Child run finished. `rvrel_result` has fields: relaunched, success, ",
                    "chain (loaded from FITS), outputs")
        else
            global rvrel_result = OctofitterRVRel.run_rvrel(
                sampler=sampler, iterations=1000, seed=123, show_plots=!is_child)
            is_child || println("[ORR] Result stored in `rvrel_result` (fields: data, system, ",
                "model, init_chain, chain, outputs, chain_path)")
        end
    end
end

nothing
