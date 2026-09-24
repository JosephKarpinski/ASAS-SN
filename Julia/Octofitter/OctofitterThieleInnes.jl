#=
================================================================================
 OctofitterThieleInnes.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Fit with a Thiele-Innes Basis":
   https://sefffal.github.io/Octofitter.jl/dev/thiele-innes/

 The planet's size/orientation are sampled as the Thiele-Innes constants
 A, B, F, G [mas] (priors Normal(0, 1000)); the Campbell elements a, i, ω, Ω
 are derived per sample via PlanetOrbits.ThieleInnes(; A, B, F, G, plx).
 This avoids the ω/Ω/tp singularities as e → 0 or i → 0, but implies a very
 different prior than uniform-in-Campbell. Note: (ω, Ω) and (ω+π, Ω+π) give
 identical constants, so ThieleInnes returns the Ω ∈ [0°, 180°) branch.

 Pipeline:
   Thiele-Innes model -> initialize! -> octoplot(init) -> octofit (HMC, as in
   the tutorial; or Pigeons) -> octoplot, octocorner (all variables),
   Campbell-element pairplot from the derived columns,
   construct_system for draw 1 (a, e, i, P) + round-trip check with
   PlanetOrbits.thieleinnes -> FITS chain
 Extras (not on the tutorial page):
   - prior-implied a and i under Thiele-Innes vs Campbell priors (Monte Carlo,
     no sampling) — shows "quite a different prior"
   - Campbell-basis baseline on the same data (a ~ U(0,100), i ~ Sine(),
     ω, Ω ~ UniformCircular) and a posterior comparison (Ω folded mod 180°)

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `thieleinnes_result`)
   Shell:    julia --threads=auto OctofitterThieleInnes.jl
   Library:  ENV["OCTOTI_AUTORUN"] = "false"; include("OctofitterThieleInnes.jl")
             res = OctofitterThieleInnes.run_thieleinnes(iterations=2000)

 Threads: the tutorial's HMC fit is a single chain and takes seconds, so it
 runs in the REPL. With ENV["OCTOTI_SAMPLER"] = "pigeons" (parallel tempering,
 which does use threads) and a 1-thread REPL, the run is relaunched in a
 child `julia --threads=auto` process as in OctofitterCoplanar.jl
 (OCTOTI_RELAUNCH, OCTOTI_THREADS control this).

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; ENV["OCTOTI_ENV_MODE"] = "local" uses ./octofitter_v9_env). Needs
 Octofitter v9, Distributions, CairoMakie, PairPlots, PlanetOrbits, Pigeons,
 MCMCChains.

 Outputs (prefix "thieleinnes_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-24  Initial version on the OctofitterCoplanar v1.1.3
                   scaffolding ([OTI +t] debug stages, soft optional stages,
                   env bootstrap, v9 guard, @__DIR__ outputs, dark theme with
                   light corner/pair plots, PairPlots-matched colours, Ω
                   folded mod 180°, threaded child relaunch for Pigeons).
                   Adds: Thiele-Innes basis; derived-Campbell pairplot;
                   draw-1 construct_system check and thieleinnes round trip;
                   Monte Carlo prior-implied a/i comparison; Campbell-basis
                   baseline fit and posterior comparison; HMC or Pigeons.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "Distributions", "CairoMakie", "PairPlots",
                "PlanetOrbits", "Pigeons", "MCMCChains"),
    mode     = lowercase(get(ENV, "OCTOTI_ENV_MODE", "temp"))

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
            println("[OTI] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[OTI] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(miss; preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[OTI] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[OTI] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[OTI] Existing v9 environment found at ", Base.active_project())
        else
            specs = [n == "Octofitter" ? Pkg.PackageSpec(name=n, version="9") :
                                         Pkg.PackageSpec(name=n) for n in required]
            Pkg.add(specs)
        end
        println("[OTI] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[OTI] Loading Octofitter, Distributions, CairoMakie, PairPlots, PlanetOrbits, ",
        "Pigeons, MCMCChains (first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterThieleInnes

const LOAD_T0 = time()

using Octofitter
using Distributions
using CairoMakie
using PairPlots
using PlanetOrbits
using Pigeons            # enables octofit_pigeons (optional sampler)
import MCMCChains
using Printf
import Statistics
import Random

export run_thieleinnes, astrometry_table, build_star, build_planet,
       build_tutorial_system, prior_implied_samples, set_plot_theme!

const VERSION_STRING = "1.0"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterThieleInnes needs Octofitter v9+, but the active environment has
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
"Set `OctofitterThieleInnes.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[OTI +%7.1fs] ", time() - T0[]), msg...)
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
    dbg("OctofitterThieleInnes v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
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
fold180(x_deg) = mod.(x_deg, 180)

"Orbital period [days] per sample from Kepler's third law (planet mass 0)."
period_days(chain) = sqrt.(colvec(chain, :b_a) .^ 3 ./ colvec(chain, :A_mass)) .* DAYS_PER_YEAR

"Summary rows: (label, column, transform); missing columns are skipped."
const SUMMARY_ROWS = [
    ("plx [mas]",           :plx,    identity),
    ("A_mass [M⊙]",         :A_mass, identity),
    ("b_A [mas]",           :b_A,    identity),
    ("b_B [mas]",           :b_B,    identity),
    ("b_F [mas]",           :b_F,    identity),
    ("b_G [mas]",           :b_G,    identity),
    ("b_a [AU]",            :b_a,    identity),
    ("b_e",                 :b_e,    identity),
    ("b_i [deg]",           :b_i,    x -> rad2deg.(x)),
    ("b_ω [deg]",           :b_ω,    x -> rad2deg.(x)),
    ("b_Ω mod 180° [deg]",  :b_Ω,    x -> fold180(rad2deg.(x))),
    ("b_θ [deg]",           :b_θ,    x -> rad2deg.(x)),
]

function print_summary(chain, label)
    DEBUG[] || return nothing
    dbg("Posterior median [16th, 84th] — ", label, " (", size(chain, 1), " samples):")
    for (lab, p, f) in SUMMARY_ROWS
        haspar(chain, p) || continue
        q = qline(f(colvec(chain, p)))
        dbg(@sprintf("    %-20s %12.4f  [%12.4f, %12.4f]", lab, q[2], q[1], q[3]))
    end
    if haspar(chain, :b_a) && haspar(chain, :A_mass)
        q = qline(period_days(chain) ./ DAYS_PER_YEAR)
        dbg(@sprintf("    %-20s %12.4f  [%12.4f, %12.4f]", "P [yr] (Kepler)", q[2], q[1], q[3]))
    end
    return nothing
end

function print_divergences(chain; label="chain")
    DEBUG[] || return nothing
    try
        n = Int(sum(colvec(chain, :numerical_error)))
        dbg(@sprintf("  %s: divergences %d / %d (%.1f%%)", label, n, size(chain, 1),
                     100n / size(chain, 1)))
        dbg(@sprintf("  %s mean acceptance rate = %.3f", label,
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
                dbg(@sprintf("    %-20s R̂ = %.4f", p, r))
            end
        end
    catch
        dbg("  (couldn't extract the rhat column; see table above)")
    end
    return ss
end

const COMPARE_ROWS = [
    ("a [AU]",            c -> colvec(c, :b_a)),
    ("e",                 c -> colvec(c, :b_e)),
    ("i [deg]",           c -> rad2deg.(colvec(c, :b_i))),
    ("ω [deg]",           c -> rad2deg.(colvec(c, :b_ω))),
    ("Ω mod 180° [deg]",  c -> fold180(rad2deg.(colvec(c, :b_Ω)))),
    ("P [yr]",            c -> period_days(c) ./ DAYS_PER_YEAR),
    ("A_mass [M⊙]",       c -> colvec(c, :A_mass)),
]

function print_comparison(chain_c, chain_ti)
    DEBUG[] || return nothing
    dbg("Basis comparison — median [16th, 84th]:")
    dbg(@sprintf("    %-18s %-34s %-34s", "", "Campbell basis", "Thiele-Innes basis"))
    for (lab, f) in COMPARE_ROWS
        qc = qline(f(chain_c)); qt = qline(f(chain_ti))
        dbg(@sprintf("    %-18s %9.3f [%9.3f, %9.3f]   %9.3f [%9.3f, %9.3f]",
                     lab, qc[2], qc[1], qc[3], qt[2], qt[1], qt[3]))
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

light(f) = with_theme(f, Theme())

corner(model, chains...; dark::Bool=false, small::Bool=true) =
    dark ? octocorner(model, chains...; small=small) :
           light(() -> octocorner(model, chains...; small=small))

# Match PairPlots' series order (1st = blue, 2nd = orange): corner is (Campbell, TI)
const C_CAMP = "#0072B2"
const C_TI   = "#E69F00"

"Pair plot of the derived Campbell elements from a Thiele-Innes chain (as in the tutorial)."
campbell_pairplot(chain) = light() do
    tbl = (; b_a = colvec(chain, :b_a),
             b_e = colvec(chain, :b_e),
             b_i_deg = rad2deg.(colvec(chain, :b_i)))
    pairplot(tbl)
end

"Two-histogram panels (Campbell vs Thiele-Innes) for a list of (label, xc, xt)."
function overlay_figure(title, panels; clip_q=0.99)
    fig = Figure(size=(430 * length(panels), 420))
    for (k, (lab, xc, xt)) in enumerate(panels)
        ax = Axis(fig[1, k], xlabel=lab, ylabel=k == 1 ? "density" : "")
        xc, xt = filter(isfinite, xc), filter(isfinite, xt)
        if clip_q < 1
            hi = max(Statistics.quantile(xc, clip_q), Statistics.quantile(xt, clip_q))
            xc, xt = filter(<=(hi), xc), filter(<=(hi), xt)
        end
        hist!(ax, xc; bins=60, normalization=:pdf, color=(C_CAMP, 0.5), label="Campbell")
        hist!(ax, xt; bins=60, normalization=:pdf, color=(C_TI, 0.5), label="Thiele-Innes")
        k == 1 && axislegend(ax, position=:rt)
    end
    Label(fig[0, :], title, fontsize=18, font=:bold)
    return fig
end

# -----------------------------------------------------------------------------
# Data and model components (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
"Tutorial data: 8 epochs of ra/dec [mas], σ = 10 mas, no correlations."
function astrometry_table()
    return Table(;
        epoch = Float64[50000, 50120, 50240, 50360, 50480, 50600, 50720, 50840],
        ra    = [-505.7637580573554, -502.570356287689, -498.2089148883798, -492.67768482682357,
                 -485.9770335870402, -478.1095526888573, -469.0801731788123, -458.89628893460525],
        dec   = [-66.92982418533026, -37.47217527025044, -7.927548139010479, 21.63557115669823,
                 51.147204404903704, 80.53589069730698, 109.72870493064629, 138.65128697876773],
        σ_ra  = fill(10.0, 8),
        σ_dec = fill(10.0, 8),
        cor   = zeros(8),
    )
end

"""
    build_star(; mass_prior=truncated(Normal(1.2, 0.1), lower=0.1))
"""
function build_star(; mass_prior::Distribution=truncated(Normal(1.2, 0.1), lower=0.1))
    dbg("  star 'A': mass ~ ", mass_prior)
    vars = @eval @variables begin
        mass ~ $mass_prior
    end
    return Body(name="A", variables=vars)
end

"""
    build_planet(host, basis; ti_prior=Normal(0, 1000), a_prior=Uniform(0, 100),
                 e_prior=Uniform(0.0, 0.5), epoch=50000.0)

`basis = :thieleinnes` samples A, B, F, G [mas] ~ `ti_prior` and derives
a, i, ω, Ω via PlanetOrbits.ThieleInnes (needs system.plx). `basis =
:campbell` samples a ~ `a_prior`, i ~ Sine(), ω, Ω ~ UniformCircular().
Both share e ~ `e_prior`, θ ~ UniformCircular() at `epoch`, mass = 0.
"""
function build_planet(host, basis::Symbol; ti_prior::Distribution=Normal(0, 1000),
                      a_prior::Distribution=Uniform(0, 100),
                      e_prior::Distribution=Uniform(0.0, 0.5),
                      epoch::Real=50000.0)
    ep = Float64(epoch)
    if basis === :thieleinnes
        dbg("  planet 'b' [Thiele-Innes]: A, B, F, G ~ ", ti_prior, " [mas], e ~ ", e_prior,
            ", epoch = ", ep)
        vars = @eval @variables begin
            mass = 0.0
            e ~ $e_prior
            A ~ $ti_prior            # [mas]
            B ~ $ti_prior            # [mas]
            F ~ $ti_prior            # [mas]
            G ~ $ti_prior            # [mas]
            # Campbell elements derived from the constants (plx converts mas -> AU)
            ti = PlanetOrbits.ThieleInnes(; A, B, F, G, plx=system.plx)
            a = ti.a
            i = ti.i
            ω = ti.ω
            Ω = ti.Ω
            θ ~ UniformCircular()
            epoch = $ep
        end
    elseif basis === :campbell
        dbg("  planet 'b' [Campbell]: a ~ ", a_prior, " [AU], e ~ ", e_prior,
            ", i ~ Sine(), ω/Ω ~ UniformCircular(), epoch = ", ep)
        vars = @eval @variables begin
            mass = 0.0
            a ~ $a_prior
            e ~ $e_prior
            i ~ Sine()
            ω ~ UniformCircular()
            Ω ~ UniformCircular()
            θ ~ UniformCircular()
            epoch = $ep
        end
    else
        error("basis must be :thieleinnes or :campbell, got $basis")
    end
    return Body(name="b", about=host, variables=vars)
end

"""
    build_tutorial_system(basis; plx_prior=truncated(Normal(50.0, 0.02), lower=0.1), kw...)

Fresh star, planet (in `basis`), observation ("GPI", fixed jitter = 0,
northangle = 0, platescale = 1) and system. Extra keywords go to `build_planet`.
"""
function build_tutorial_system(basis::Symbol;
                               plx_prior::Distribution=truncated(Normal(50.0, 0.02), lower=0.1),
                               kw...)
    A = build_star()
    b = build_planet(A, basis; kw...)
    tbl = astrometry_table()
    dbg("  obs 'GPI': ", length(tbl.epoch), " epochs, fixed jitter/northangle/platescale")
    obs_vars = @eval @variables begin
        jitter = 0
        northangle = 0
        platescale = 1
    end
    obs = RelAstromObs(tbl; target=b, ref=A, name="GPI", variables=obs_vars)
    name = basis === :thieleinnes ? "TutoriaPrime" : "TutoriaCampbell"
    dbg("  system '", name, "': plx ~ ", plx_prior)
    sys_vars = @eval @variables begin
        plx ~ $plx_prior
    end
    return System(name=name, bodies=[A, b], observations=[obs], variables=sys_vars)
end

# -----------------------------------------------------------------------------
# Prior-implied Campbell elements (Monte Carlo; no sampling)
# -----------------------------------------------------------------------------
"""
    prior_implied_samples(; n=20_000, ti_prior=Normal(0, 1000),
                          plx_prior=truncated(Normal(50.0, 0.02), lower=0.1),
                          a_prior=Uniform(0, 100), rng=Random.Xoshiro(1))

Draw A, B, F, G, plx from their priors and convert with
PlanetOrbits.ThieleInnes to get the prior-implied a [AU] and i [deg]; also
draw the Campbell priors a ~ a_prior, i ~ Sine() for comparison.
"""
function prior_implied_samples(; n::Integer=20_000, ti_prior::Distribution=Normal(0, 1000),
                               plx_prior::Distribution=truncated(Normal(50.0, 0.02), lower=0.1),
                               a_prior::Distribution=Uniform(0, 100),
                               rng=Random.Xoshiro(1))
    a_ti = Vector{Float64}(undef, n)
    i_ti = Vector{Float64}(undef, n)
    for k in 1:n
        A, B, F, G = rand(rng, ti_prior, 4)
        ti = PlanetOrbits.ThieleInnes(; A, B, F, G, plx=rand(rng, plx_prior))
        a_ti[k] = ti.a
        i_ti[k] = rad2deg(ti.i)
    end
    a_c = rand(rng, a_prior, n)
    i_c = rad2deg.(acos.(rand(rng, Uniform(-1, 1), n)))   # Sine() prior on i ∈ [0, π]
    return (; a_ti, i_ti, a_c, i_c)
end

# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------
"Sample with HMC (tutorial) or Pigeons; returns the chain."
function run_sampler(model; sampler::Symbol, iterations::Integer, n_rounds::Integer, tag)
    if sampler === :pigeons
        chain, pt = octofit_pigeons(model; n_rounds=n_rounds)
        soft_stage("[$tag] Pigeons log-evidence") do
            dbg(@sprintf("  [%s] log Z ≈ %.2f, Λ = %.2f", tag,
                         Pigeons.stepping_stone(pt), Pigeons.global_barrier(pt)))
        end
        return chain
    else
        return octofit(model; iterations=iterations)
    end
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_thieleinnes(; sampler=:hmc, iterations=1000, n_rounds=10,
                      compare_campbell=true, prior_mc=true, ti_sigma=1000.0,
                      outdir=SCRIPT_DIR, prefix="thieleinnes_", dark=true,
                      show_plots=true)

Run the Thiele-Innes tutorial; optionally the Campbell baseline and the
prior-implied comparison. Returns a NamedTuple of results.
"""
function run_thieleinnes(; sampler::Symbol=:hmc,
                           iterations::Integer=1000,
                           n_rounds::Integer=10,
                           compare_campbell::Bool=true,
                           prior_mc::Bool=true,
                           ti_sigma::Real=1000.0,
                           outdir::AbstractString=SCRIPT_DIR,
                           prefix::AbstractString="thieleinnes_",
                           dark::Bool=true,
                           show_plots::Bool=true)
    sampler in (:hmc, :pigeons) || error("sampler must be :hmc or :pigeons")
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    print_env_info(outdir, sampler)
    set_plot_theme!(; dark)
    ti_prior = Normal(0, Float64(ti_sigma))

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

    # ---- Prior-implied comparison (cheap, before any sampling) ---------------
    pri = nothing
    if prior_mc
        pri = soft_stage("Prior-implied a, i: Thiele-Innes vs Campbell (Monte Carlo)") do
            s = prior_implied_samples(; ti_prior)
            qa = qline(s.a_ti); qi = qline(s.i_ti)
            dbg(@sprintf("  TI prior → a [AU]: %.1f [%.1f, %.1f];  i [deg]: %.1f [%.1f, %.1f]",
                         qa[2], qa[1], qa[3], qi[2], qi[1], qi[3]))
            dbg("  Campbell prior → a ~ U(0, 100) AU; i ~ Sine() (median 90°)")
            savefig!(overlay_figure("Prior-implied elements (no data)",
                        [("a [AU]", s.a_c, s.a_ti), ("i [deg]", s.i_c, s.i_ti)]; clip_q=0.995),
                     "prior_implied.png")
            s
        end
    end

    # ---- Thiele-Innes model (the tutorial) ----------------------------------
    sys_ti = stage("[TI] Build system (Thiele-Innes basis)") do
        build_tutorial_system(:thieleinnes; ti_prior)
    end
    DEBUG[] && display(sys_ti)

    model_ti = stage("[TI] Compile LogDensityModel") do
        Octofitter.LogDensityModel(sys_ti)
    end
    DEBUG[] && display(model_ti)

    init_ti = stage("[TI] Initialize starting points (no guesses)") do
        initialize!(model_ti)
    end
    soft_stage("[TI] Plot initial guess") do
        savefig!(inout(() -> octoplot(model_ti, init_ti)), "ti_init.png")
    end

    chain_ti = stage("[TI] Sample posterior ($(sampler === :hmc ? "HMC, $iterations iterations" :
                                               "Pigeons, n_rounds=$n_rounds"))") do
        run_sampler(model_ti; sampler, iterations, n_rounds, tag="TI")
    end
    print_divergences(chain_ti; label="TI chain")
    print_summary(chain_ti, "Thiele-Innes basis")
    soft_stage("[TI] R-hat") do
        print_rhat(chain_ti)
    end

    stage("[TI] Orbit plot") do
        savefig!(inout(() -> octoplot(model_ti, chain_ti)), "ti_orbits.png")
    end
    soft_stage("[TI] Corner plot (all variables, small=false)") do
        savefig!(inout(() -> corner(model_ti, chain_ti; small=false)), "ti_corner.png")
    end
    soft_stage("[TI] Pair plot of derived Campbell elements (a, e, i)") do
        savefig!(campbell_pairplot(chain_ti), "ti_campbell_pairplot.png")
    end

    soft_stage("[TI] construct_system for draw 1 + Thiele-Innes round trip") do
        posys = construct_system(model_ti, chain_ti, 1)
        el = (; a = semimajoraxis(posys, 1), e = eccentricity(posys, 1),
                i = rad2deg(inclination(posys, 1)), P_days = period(posys, 1))
        dbg(@sprintf("  draw 1: a = %.4f AU, e = %.4f, i = %.3f°, P = %.1f d (%.2f yr)",
                     el.a, el.e, el.i, el.P_days, el.P_days / DAYS_PER_YEAR))
        dbg(@sprintf("  chain row 1: b_a = %.4f, b_e = %.4f, b_i = %.3f°",
                     colvec(chain_ti, :b_a)[1], colvec(chain_ti, :b_e)[1],
                     rad2deg(colvec(chain_ti, :b_i)[1])))
        plx1 = colvec(chain_ti, :plx)[1]
        abfg = PlanetOrbits.thieleinnes(posys, 1; plx=plx1)
        chain_abfg = [colvec(chain_ti, p)[1] for p in (:b_A, :b_B, :b_F, :b_G)]
        dbg("  thieleinnes(posys, 1; plx=", round(plx1; digits=4), ") = ", abfg)
        dbg("  chain row 1 (A, B, F, G)          = ", round.(chain_abfg; digits=6))
        try
            dmax = maximum(abs.(collect(Float64, values(abfg)) .- chain_abfg))
            dbg(@sprintf("  max |Δ| = %.3e mas  %s", dmax,
                         dmax < 1e-6 * max(1.0, maximum(abs.(chain_abfg))) ?
                         "(round trip OK)" : "(check: differs)"))
        catch
            dbg("  (couldn't compare element-wise; see the two lines above)")
        end
        el
    end

    path_ti = joinpath(outdir, prefix * "chain_ti.fits")
    soft_stage("[TI] Save chain -> FITS (+ reload check)") do
        Octofitter.savechain(path_ti, chain_ti)
        push!(outputs, path_ti)
        c2 = Octofitter.loadchain(path_ti; model=model_ti)
        dbg("  reloaded chain size = ", size(c2),
            size(c2) == size(chain_ti) ? "  (matches)" : "  (DIFFERS from saved)")
    end

    # ---- Campbell baseline --------------------------------------------------
    model_c = nothing
    chain_c = nothing
    if compare_campbell
        model_c = soft_stage("[Campbell] Build + compile baseline") do
            s = build_tutorial_system(:campbell)
            DEBUG[] && display(s)
            Octofitter.LogDensityModel(s)
        end
        if model_c !== nothing
            chain_c = soft_stage("[Campbell] Initialize + sample") do
                initialize!(model_c)
                run_sampler(model_c; sampler, iterations, n_rounds, tag="Campbell")
            end
        end
        if chain_c !== nothing
            print_divergences(chain_c; label="Campbell chain")
            print_summary(chain_c, "Campbell basis")
            soft_stage("[Campbell] Orbit plot") do
                savefig!(inout(() -> octoplot(model_c, chain_c)), "campbell_orbits.png")
            end
            print_comparison(chain_c, chain_ti)
            soft_stage("Posterior comparison histograms (a, e, i, Ω mod 180°)") do
                panels = [(lab, f(chain_c), f(chain_ti)) for (lab, f) in COMPARE_ROWS[1:5]
                          if lab != "ω [deg]"]
                savefig!(overlay_figure("Posterior: Campbell vs Thiele-Innes basis", panels),
                         "compare_posterior.png")
            end
            soft_stage("Corner plot: Campbell vs Thiele-Innes") do
                savefig!(inout(() -> corner(model_ti, chain_c, chain_ti)), "corner_compare.png")
            end
            path_c = joinpath(outdir, prefix * "chain_campbell.fits")
            soft_stage("[Campbell] Save chain -> FITS") do
                Octofitter.savechain(path_c, chain_c)
                push!(outputs, path_c)
            end
        end
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f s", time() - T0[]))

    return (; system=sys_ti, model=model_ti, init_chain=init_ti, chain=chain_ti,
              model_campbell=model_c, chain_campbell=chain_c, prior_samples=pri,
              outputs, chain_path=path_ti)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (used only for the Pigeons sampler in a 1-thread session)
# -----------------------------------------------------------------------------
"""
    relaunch_threaded(file; threads="auto", outdir=SCRIPT_DIR, prefix="thieleinnes_")

Run `file` in a child Julia process with `--threads=threads` and the same
active project, streaming its output; then load the FITS chains it wrote.
"""
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR,
                           prefix::AbstractString="thieleinnes_")
    T0[] = time()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOTI_CHILD" => "1")
    dbg("Pigeons sampler requested and this session has 1 thread; Julia can't add threads")
    dbg("to a running process. Relaunching with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOTI_CHILD=1]")
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
    chains = Dict{Symbol,Any}()
    for (kind, fname) in ((:ti, "chain_ti.fits"), (:campbell, "chain_campbell.fits"))
        p = joinpath(outdir, prefix * fname)
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

println(@sprintf("[OTI] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterThieleInnes

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia OctofitterThieleInnes.jl`              -> runs
#   - VSCode "Execute active File in REPL"          -> runs (interactive session)
#   - ENV["OCTOTI_AUTORUN"] = "false"; include()    -> loads module only
#   - ENV["OCTOTI_SAMPLER"] = "pigeons"             -> parallel tempering; in a
#     1-thread session this relaunches in `julia --threads=auto` (disable with
#     ENV["OCTOTI_RELAUNCH"] = "false"; ENV["OCTOTI_THREADS"] sets the count)
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOTI_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOTI_CHILD", "") == "1",
    sampler   = Symbol(lowercase(get(ENV, "OCTOTI_SAMPLER", "hmc"))),
    relaunch  = !is_child && sampler === :pigeons && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOTI_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global thieleinnes_result = OctofitterThieleInnes.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOTI_THREADS", "auto"))
            println("[OTI] Child run finished. `thieleinnes_result` has fields: relaunched, ",
                    "success, chains (loaded from FITS), outputs")
        else
            global thieleinnes_result = OctofitterThieleInnes.run_thieleinnes(
                sampler=sampler, iterations=1000, compare_campbell=true,
                prior_mc=true, show_plots=!is_child)
            is_child || println("[OTI] Result stored in `thieleinnes_result` (fields: system, ",
                "model, init_chain, chain, model_campbell, chain_campbell, prior_samples, ",
                "outputs, chain_path)")
        end
    end
end

nothing
