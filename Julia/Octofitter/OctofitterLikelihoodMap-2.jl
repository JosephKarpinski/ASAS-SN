#=
================================================================================
 OctofitterLikelihoodMap.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Fitting Likelihood Maps":
   https://sefffal.github.io/Octofitter.jl/dev/fit-likemap/

 A 2-D map of log-likelihood vs (Δα*, Δδ) — from cleaned interferometry, a
 spectral-cube fit, or anything else — is fitted with an orbit: the
 likelihood reads the map (bilinear interpolation) at the planet's predicted
 position in every epoch (LogLikelihoodMapObs, OctofitterImages).

 Data (as on the page, synthetic): a template orbit (PlanetOrbits system,
 M = 1 M⊙, a = 1 AU, e = 0.1, i = 0, ω = 0, Ω = 0.5, tp = 0, plx = 50 mas)
 gives the true position at 2024-01-30 and 2024-02-29; each epoch's map is
 the log-pdf of a 3-component Gaussian mixture (weights 0.5 / 0.3 / 0.2), the
 heaviest peak at the true position and two decoys 8–13 mas away, over a
 ±50 mas patch at 1 mas/pixel with offset X/Y coordinates.

 Model (as on the page): A mass ~ N(1.0, 0.1) M⊙; b: a ~ U(0, 10) AU,
 e ~ U(0, 0.5), i ~ Sine, ω, Ω, θ ~ UniformCircular, θ epoch MJD 60339;
 LogLikelihoodMapObs "GRAVITY" (target :b, ref :A, platescale 1,
 northangle 0); plx ~ N(50, 0.02) mas. Sampling: initialize!, then
 octofit_pigeons(n_rounds=10), then the page's resume:
 increment_n_rounds!(pt, 2) and octofit_pigeons(pt).

 Plots (page, plus extras marked +): the two likelihood maps (+ with the
 three peak centres and the truth marked), octoplot, corner (small=true), the
 page's posterior-predictive pairplots at both epochs, + predicted positions
 over each map, corner (small=false) after the resume.
 Log: + truth comparison with percentiles; + which peak the posterior picks
 at each epoch (and jointly), before and after the resume; + log(Z₁/Z₀), Λ,
 min(α) before and after the resume.

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `lm_result`)
   Shell:    julia --threads=auto OctofitterLikelihoodMap.jl
   Library:  ENV["OCTOLM_AUTORUN"] = "false"; include("OctofitterLikelihoodMap.jl")
             res = OctofitterLikeMap.run_likemap()

 Threads: Pigeons runs multithreaded. In a 1-thread REPL the run is relaunched
 in a child `julia --threads=auto` (output streamed here; the chain is loaded
 back and the saved plots are shown in the plot pane afterwards).
 OCTOLM_RELAUNCH=false disables this; OCTOLM_THREADS sets the count.

 Settings (ENV, also passed to a relaunched child):
   OCTOLM_ROUNDS      = "10" (page) — first Pigeons run
   OCTOLM_EXTRA       = "2"  (page) — rounds added by increment_n_rounds!; 0 skips
   OCTOLM_CHAINS      = ""   (Pigeons default) — n_chains
   OCTOLM_FLIP        = "false" (page) — "true" builds the maps with map X = −Δα*,
                        the frame LogLikelihoodMapObs reads (see v1.0.1 below)
   OCTOLM_SHOW_PLOTS  = "true" — show the child's PNGs here

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOLM_ENV_MODE=local uses ./octofitter_v9_env). OctofitterImages is
 unregistered and is added from the Octofitter repository at a pinned commit.
 Also needs AstroImages, CairoMakie, PairPlots, Distributions, Pigeons.

 Outputs (prefix "lm_") go to the directory containing this file. No data
 files are downloaded: the maps are generated in the module, as on the page.

 Changelog
 ---------
 v1.0  2026-09-26  Initial version on the OctofitterInterferometryTutorial
                   v1.0.1 scaffolding ([LM +t] debug stages, soft optional
                   stages, env bootstrap with v9 guards and a pinned
                   unregistered-package install, @__DIR__ outputs, dark theme
                   with light corner and pair plots, threaded relaunch with
                   exit-code-only failure report and PNGs shown in the
                   parent's plot pane, @eval-built @variables, min(α) from
                   Pigeons.swap_prs). Adds: truth comparison with percentiles,
                   per-epoch peak assignment, predicted positions over the
                   maps, diagnostics before/after the page's resume.
 v1.0.1 2026-09-26 First run (3.0 min): peak assignment said 100% off-peak and
                   the posterior overlay was empty, because the posterior sits
                   at Δα* ≈ −46 mas while the template is at +46. Cause: the
                   page builds maps with X = +Δα*, but LogLikelihoodMapObs
                   reads map X as −Δα* (pixel_position; "+RA at the left"), so
                   the maps encode the template mirrored in RA (same a, e, M,
                   plx; i → 180° − i — hence i ≈ 137°, not ≈ 0°). Fixes: peak
                   assignment and overlay now work in the map frame; the truth
                   table uses the encoded orbit (i = 180° in page mode); the
                   log explains the mirroring; new OCTOLM_FLIP=true builds the
                   maps in the likelihood's frame. Run 1 numbers: log(Z₁/Z₀)
                   −21.03 (resume Δ −0.003), Λ 3.8, min(α) 0.71; a 1.52
                   (1.09–2.43) AU vs 1.0, e 0.26 vs 0.1, M 1.01, plx 50.00.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let octo_commit = "8c6daaaf5292b6cbdb0aa4f4b1e9a1a5f0bd21f3",
    required = ("Octofitter", "OctofitterImages", "AstroImages", "CairoMakie", "PairPlots",
                "Distributions", "Pigeons"),
    min_version = Dict("Octofitter" => v"9"),
    mode = lowercase(get(ENV, "OCTOLM_ENV_MODE", "temp"))

    direct_deps() = Dict(info.name => info.version
                         for info in values(Pkg.dependencies()) if info.is_direct_dep)
    missing_deps() = [n for n in required if !haskey(direct_deps(), n)]
    function versions_ok()
        d = direct_deps()
        all(!haskey(d, n) || (d[n] !== nothing && d[n] >= v) for (n, v) in min_version)
    end
    env_ok() = isempty(missing_deps()) && versions_ok()
    function spec(n)
        n == "OctofitterImages" &&
            return Pkg.PackageSpec(url="https://github.com/sefffal/Octofitter.jl",
                                   subdir="OctofitterImages", rev=octo_commit)
        haskey(min_version, n) ? Pkg.PackageSpec(name=n, version="9") : Pkg.PackageSpec(name=n)
    end
    note(miss) = "OctofitterImages" in miss &&
        println("[LM] OctofitterImages is unregistered: cloning it from the Octofitter repository ",
                "(commit ", first(octo_commit, 8), "); this takes a minute or two the first time.")

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
            println("[LM] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[LM] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            note(miss)
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[LM] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[LM] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[LM] Existing v9 environment found at ", Base.active_project())
        else
            note(missing_deps())
            Pkg.add(spec.(collect(required)))
        end
        println("[LM] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[LM] Loading Octofitter, OctofitterImages, AstroImages, Pigeons, CairoMakie, PairPlots, ",
        "Distributions (first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterLikeMap

const LOAD_T0 = time()

using Octofitter
using OctofitterImages
import AstroImages
using CairoMakie
using PairPlots
using Distributions
using Pigeons
import Statistics
using Printf

export run_likemap, set_plot_theme!

const VERSION_STRING = "1.0.1"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterLikeMap needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

# PlanetOrbits' own Body/Orbit/System (Octofitter exports model types of the
# same names, so the page qualifies them).
const PO = Octofitter.PlanetOrbits

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const PREFIX = "lm_"

# The page's template orbit (the truth) and epochs
const TRUTH = (M=1.0, a=1.0, e=0.1, i=0.0, ω=0.0, Ω=0.5, tp=0.0, plx=50.0)
const EPOCH_DATES = ("2024-01-30", "2024-02-29")

# The page's peaks: (offset from the true position [mas], covariance, weight)
const PEAKS = (
    ( ((0.0, 0.0), [5.0 0.2; 0.2 8.0], 0.5),
      ((8.0, 4.0), [5.0 0.6; 0.6 5.0], 0.3),
      ((9.0, -10.0), [6.0 0.6; 0.6 6.0], 0.2) ),
    ( ((0.0, 0.0), [5.0 0.2; 0.2 8.0], 0.5),
      ((10.0, 0.0), [5.0 0.6; 0.6 5.0], 0.3),
      ((-4.0, -10.0), [6.0 0.6; 0.6 6.0], 0.2) ),
)
const HALF = 50          # map half-width [pixels = mas]

const PRIORS = (
    mass_A = truncated(Normal(1.0, 0.1), lower=0.1),
    a      = Uniform(0, 10),
    e      = Uniform(0.0, 0.5),
    θ_ep   = 60339.0,
    plx    = truncated(Normal(50.0, 0.02), lower=0.1),
)

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterLikeMap.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[LM +%7.1fs] ", time() - T0[]), msg...)
    flush(stdout)
    return nothing
end

function stage(f, name::AbstractString)
    dbg("▶ ", name)
    t = time()
    result = try
        f()
    catch err
        dbg(@sprintf("✖ %s FAILED after %.2f s: ", name, time() - t),
            first(sprint(showerror, err), 2000))
        rethrow()
    end
    dbg(@sprintf("✔ %s (%.2f s)", name, time() - t))
    return result
end

function soft_stage(f, name::AbstractString)
    dbg("▶ ", name, "  (optional)")
    t = time()
    try
        result = f()
        dbg(@sprintf("✔ %s (%.2f s)", name, time() - t))
        return result
    catch err
        dbg(@sprintf("⚠ %s skipped after %.2f s: ", name, time() - t),
            first(sprint(showerror, err), 1000))
        return nothing
    end
end

envint(k, default) = parse(Int, strip(get(ENV, k, default)))
envbool(k, default) = lowercase(strip(get(ENV, k, default))) in ("true", "1", "yes")
function envopt(k)
    s = strip(get(ENV, k, ""))
    return isempty(s) ? nothing : String(s)
end

function print_env_info(outdir, n_rounds, extra, n_chains)
    DEBUG[] || return nothing
    dbg("OctofitterLikeMap v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | OctofitterImages ", something(pkgversion(OctofitterImages), "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("Pigeons: n_rounds = ", n_rounds, " (page 10), then +", extra,
        " via increment_n_rounds! (page 2) | n_chains = ",
        n_chains === nothing ? "Pigeons default" : n_chains)
    Threads.nthreads() == 1 &&
        dbg("  NOTE: running Pigeons on 1 thread (relaunch disabled) — expect a long run")
    dbg("project    = ", Base.active_project())
    dbg("SCRIPT_DIR = ", SCRIPT_DIR, " | outdir = ", abspath(outdir))
    dbg("interactive = ", isinteractive(),
        " | Makie backend = ", CairoMakie.Makie.current_backend())
    return nothing
end

# ---- Chain helpers -----------------------------------------------------------
colvec(chain, p::Symbol) = vec(Array(chain[p]))
haspar(chain, p::Symbol) = p in names(chain)
qline(x) = (y = filter(isfinite, x); isempty(y) ? (NaN, NaN, NaN) :
                                     Statistics.quantile(y, (0.16, 0.5, 0.84)))

"Posterior summary with the truth, its σ offset and its percentile."
function truth_table(chain; flip::Bool=false)
    DEBUG[] || return nothing
    n = size(chain, 1) * size(chain, 3)
    # Page maps (flip=false) are the template mirrored in RA as the likelihood reads
    # them, so the orbit they encode has i → 180° − i; a, e, M, plx are unchanged.
    i_eff = flip ? rad2deg(TRUTH.i) : 180 - rad2deg(TRUTH.i)
    dbg("Posterior median [16th, 84th] vs the orbit encoded in the maps (", n, " samples)",
        flip ? ":" : " — the template mirrored in RA, so i = 180° − i_template:")
    rows = (("A mass [M⊙]", :A_mass, identity, TRUTH.M),
            ("b a [AU]", :b_a, identity, TRUTH.a),
            ("b e", :b_e, identity, TRUTH.e),
            ("b i [deg]", :b_i, x -> rad2deg.(x), i_eff),
            ("plx [mas]", :plx, identity, TRUTH.plx))
    for (lab, p, f, tv) in rows
        haspar(chain, p) || continue
        x = filter(isfinite, f(colvec(chain, p)))
        q = qline(x)
        σ = (q[3] - q[1]) / 2
        pct = 100 * Statistics.mean(x .< tv)
        flag = (pct < 2.5 || pct > 97.5) ? "  ← truth outside the central 95%" : ""
        dbg(@sprintf("    %-16s %10.4f  [%10.4f, %10.4f]   truth %8.4f  (%+.2fσ, at %.1f%%)%s",
                     lab, q[2], q[1], q[3], tv, σ > 0 ? (tv - q[2]) / σ : NaN, pct, flag))
    end
    if haspar(chain, :b_i)
        dbg(@sprintf("    note: the encoded orbit is face-on (i = %.0f°), where Sine() has almost no prior volume;", i_eff))
        dbg("          a posterior pulled away from it is the prior, not a failure of the fit")
    end
    if haspar(chain, :b_a) && haspar(chain, :A_mass)
        a = colvec(chain, :b_a)
        q = qline(sqrt.(a .^ 3 ./ colvec(chain, :A_mass)))
        dbg(@sprintf("    %-16s %10.4f  [%10.4f, %10.4f]   truth %8.4f", "b P [yr]", q[2], q[1], q[3],
                     sqrt(TRUTH.a^3 / TRUTH.M)))
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
const PEAK_COLORS = ("#009E73", "#E69F00", "#CC79A7")

datestr(ep) = first(string(mjd2date(ep)), 10)

# -----------------------------------------------------------------------------
# Synthetic data (page section: template orbit, three peaks per epoch)
# -----------------------------------------------------------------------------
"The page's template: a PlanetOrbits system (not an Octofitter model)."
function template_system()
    star = PO.Body(mass=TRUTH.M, name=:A)
    planet = PO.Body(mass=0.0, name=:b)
    return PO.System(
        (star, planet),
        (PO.Orbit(planet; about=star, a=TRUTH.a, e=TRUTH.e, i=TRUTH.i, ω=TRUTH.ω,
                  Ω=TRUTH.Ω, tp=TRUTH.tp),);
        plx = TRUTH.plx,
    )
end

"""
One epoch: true position, the page's mixture, and its log-pdf map on a
±50 mas patch (1 mas pixels) wrapped in an AstroImage with offset X/Y axes.
"""
function make_map(orbit, epoch, peaks; flip::Bool=false)
    sol = orbitsolve(orbit, epoch)
    x0, y0 = raoff(sol, :b, :A), decoff(sol, :b, :A)
    comps = [MvNormal([x0 + o[1], y0 + o[2]], Σ) for (o, Σ, _) in peaks]
    d = MixtureModel(comps, [w for (_, _, w) in peaks])
    # Map X coordinate u. The page uses u = +Δα* (s = +1); LogLikelihoodMapObs
    # reads u as −Δα* (pixel_position), which flip=true builds (s = −1).
    s = flip ? -1 : 1
    xs = s * x0 .+ (-HALF:HALF)
    ys = y0 .+ (-HALF:1:HALF)
    lm = broadcast(xs, ys') do u, y
        logpdf(d, [s * u, y])
    end
    img = AstroImages.AstroImage(lm, (AstroImages.X(xs), AstroImages.Y(ys)))
    centres = [(s * (x0 + o[1]), y0 + o[2]) for (o, _, _) in peaks]      # map frame
    return (; img, lm, xs, ys, truth=(s * x0, y0), truth_sky=(x0, y0), centres,
              weights=[w for (_, _, w) in peaks])
end

"The two maps as on the page, with the truth and peak centres per epoch."
function synthetic_maps(; flip::Bool=false)
    orbit = template_system()
    epochs = [mjd(d) for d in EPOCH_DATES]
    maps = [make_map(orbit, epochs[k], PEAKS[k]; flip) for k in 1:2]
    for (k, m) in enumerate(maps)
        dbg(@sprintf("  epoch %d (%s, MJD %.1f): template position (Δα*, Δδ) = (%.2f, %.2f) mas, sep %.2f mas; map %d×%d, logL %.1f…%.2f",
                     k, EPOCH_DATES[k], epochs[k], m.truth_sky[1], m.truth_sky[2], hypot(m.truth_sky...),
                     size(m.lm, 1), size(m.lm, 2), minimum(m.lm), maximum(m.lm)))
        for (j, c) in enumerate(m.centres)
            dbg(@sprintf("      peak %d at map (X, Y) = (%.2f, %.2f), weight %.1f%s", j, c[1], c[2], m.weights[j],
                         j == 1 ? "  (template position)" : ""))
        end
    end
    if flip
        dbg("  maps built with map X = −Δα* (OCTOLM_FLIP=true), the frame LogLikelihoodMapObs reads:")
        dbg("  the fit should recover the template orbit itself")
    else
        dbg("  NOTE: as on the page, map X = +Δα* of the template, but LogLikelihoodMapObs reads map X")
        dbg("  as −Δα* (OctofitterImages pixel_position: x = −Δα*, the page's own '+RA at the left').")
        dbg("  The maps therefore encode the template mirrored in RA: the fit recovers the same a, e,")
        dbg("  M and plx with i → 180° − i and Δα* → −Δα*. OCTOLM_FLIP=true builds the maps in the")
        dbg("  likelihood's frame instead.")
    end
    dbg(@sprintf("  motion between epochs: %.2f mas in %.0f d",
                 hypot((maps[2].truth .- maps[1].truth)...), epochs[2] - epochs[1]))
    return (; orbit, epochs, maps)
end

"The page's imview(10 .^ map) as a heatmap, + truth and peak centres."
function maps_figure(data; posterior=nothing)
    fig = Figure(size=(1300, 600))
    local hm
    for (k, m) in enumerate(data.maps)
        ax = Axis(fig[1, k], aspect=DataAspect(),
                  xlabel="map X [mas] (read by the likelihood as −Δα*)", ylabel="map Y = Δδ [mas]",
                  title=@sprintf("epoch %d — %s", k, EPOCH_DATES[k]))
        z = 10 .^ (m.lm .- maximum(m.lm))     # page: 10 .^ logL (scaled to a peak of 1)
        hm = heatmap!(ax, collect(m.xs), collect(m.ys), z; colormap=:magma)
        if posterior !== nothing
            # predicted positions in the map frame: X = −Δα*
            scatter!(ax, .-posterior[k][1], posterior[k][2]; color=(:cyan, 0.35), markersize=3)
        end
        for (j, c) in enumerate(m.centres)
            scatter!(ax, [c[1]], [c[2]]; marker=:cross, color=PEAK_COLORS[j], markersize=14,
                     label=@sprintf("peak %d (w %.1f)", j, m.weights[j]))
        end
        scatter!(ax, [m.truth[1]], [m.truth[2]]; marker=:circle, color=:transparent,
                 strokecolor=:white, strokewidth=2, markersize=18, label="injected (template) position")
        limits!(ax, m.xs[1], m.xs[end], m.ys[1], m.ys[end])
        k == 1 && axislegend(ax, position=:lt, labelsize=11)
    end
    Colorbar(fig[1, 3], hm, label=posterior === nothing ? "likelihood / max (page: 10^logL)" :
                                  "likelihood / max; cyan = predicted positions")
    return fig
end

# -----------------------------------------------------------------------------
# Model (page)
# -----------------------------------------------------------------------------
function build_system(data)
    likemap_dat = Table(;
        epoch = data.epochs,
        map = [m.img for m in data.maps],
        platescale = [1.0, 1.0],          # mas / pixel of the map
    )
    obs_vars = @eval @variables begin
        platescale = 1.0                  # platescale multiplier
        northangle = 0.0                  # north angle offset [rad]
    end
    loglikemap = LogLikelihoodMapObs(
        likemap_dat,
        target = :b,                      # the companion the map describes
        ref    = :A,                      # the point index [0,0] of the map sits on
        name   = "GRAVITY",
        variables = obs_vars,
    )
    mA = PRIORS.mass_A
    A_vars = @eval @variables begin
        mass ~ $mA                        # M⊙
    end
    A = Body(name="A", variables=A_vars)
    ap, ep, θep = PRIORS.a, PRIORS.e, PRIORS.θ_ep
    b_vars = @eval @variables begin
        a ~ $ap
        e ~ $ep
        i ~ Sine()
        ω ~ UniformCircular()
        Ω ~ UniformCircular()
        θ ~ UniformCircular()
        epoch = $θep                      # reference epoch for θ
    end
    b = Body(name="b", about=A, variables=b_vars)
    pp = PRIORS.plx
    sys_vars = @eval @variables begin
        plx ~ $pp
    end
    sys = System(name="Tutoria", bodies=[A, b], observations=[loglikemap], variables=sys_vars)
    return sys, loglikemap
end

# -----------------------------------------------------------------------------
# Posterior predictive positions and peak assignment
# -----------------------------------------------------------------------------
"Predicted (Δα*, Δδ) of b at each map epoch for every draw (page's construct_system)."
function predicted_positions(model, chain, epochs)
    posteriors = construct_system(model, chain)
    return [([raoff(orbitsolve(p, ep), :b, :A) for p in posteriors],
             [decoff(orbitsolve(p, ep), :b, :A) for p in posteriors]) for ep in epochs]
end

"""
Which peak each draw lands on at each epoch: the nearest peak centre if it
is within 3σ of that peak (σ from the covariance's larger axis), otherwise
'off'. Prints per-epoch fractions and the joint table.
"""
function peak_assignment(pos, data; label="")
    labs = Vector{Vector{Int}}()
    for (k, m) in enumerate(data.maps)
        x = .-pos[k][1]            # map frame: X = −Δα*
        y = pos[k][2]
        r3 = [3 * sqrt(maximum(eigvals_sym(Σ))) for (_, Σ, _) in PEAKS[k]]
        push!(labs, map(eachindex(x)) do i
            d = [hypot(x[i] - c[1], y[i] - c[2]) for c in m.centres]
            j = argmin(d)
            d[j] <= r3[j] ? j : 0
        end)
    end
    n = length(labs[1])
    dbg("Which likelihood peak the posterior picks", isempty(label) ? "" : " — $label",
        " (", n, " draws; peak 1 = true position):")
    for k in eachindex(labs)
        f = [count(==(j), labs[k]) / n for j in 0:3]
        dbg(@sprintf("    epoch %d: peak 1 %.1f%% | peak 2 %.1f%% | peak 3 %.1f%% | off-peak %.1f%%   (map weights 50/30/20%%)",
                     k, 100 * f[2], 100 * f[3], 100 * f[4], 100 * f[1]))
    end
    combos = Dict{Tuple{Int,Int},Int}()
    for i in 1:n
        key = (labs[1][i], labs[2][i])
        combos[key] = get(combos, key, 0) + 1
    end
    top = sort(collect(combos); by=last, rev=true)
    dbg("    joint (epoch 1, epoch 2) peak pairs, most frequent first (0 = off-peak):")
    for (key, c) in first(top, min(6, length(top)))
        dbg(@sprintf("      (%d, %d): %5.1f%%%s", key[1], key[2], 100 * c / n,
                     key == (1, 1) ? "  ← the injected orbit" : ""))
    end
    return labs
end

"Eigenvalues of a symmetric 2×2 matrix (no LinearAlgebra import needed)."
function eigvals_sym(Σ)
    a, b, d = Σ[1, 1], Σ[1, 2], Σ[2, 2]
    t, s = (a + d) / 2, sqrt(((a - d) / 2)^2 + b^2)
    return (t - s, t + s)
end

"The page's posterior-predictive pairplot at one epoch (light theme)."
function predictive_pairplot(x, y)
    return light() do
        pairplot(
            (; x, y),
            axis=(
                x = (; lims=(low=100, high=-100)),
                y = (; lims=(low=-100, high=100)),
            ),
        )
    end
end

function pigeons_diag(pt, label)
    z = Pigeons.stepping_stone(pt)
    λ = Pigeons.global_barrier(pt)
    mα = minimum(Pigeons.swap_prs(pt))
    dbg(@sprintf("  %s: log(Z₁/Z₀) ≈ %.3f | Λ = %.2f | min(α) = %.3g%s", label, z, λ, mα,
                 mα < 1e-3 ? "  ⚠ chains barely swap: add n_chains or rounds" : ""))
    return (; logZ=z, Λ=λ, min_α=mα)
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_likemap(; n_rounds=10, extra_rounds=2, n_chains=nothing,
                  outdir=SCRIPT_DIR, prefix="lm_", dark=true, show_plots=true)
"""
function run_likemap(; n_rounds::Integer=envint("OCTOLM_ROUNDS", "10"),
                       extra_rounds::Integer=envint("OCTOLM_EXTRA", "2"),
                       flip::Bool=envbool("OCTOLM_FLIP", "false"),
                       n_chains=(c = envopt("OCTOLM_CHAINS"); c === nothing ? nothing : parse(Int, c)),
                       outdir::AbstractString=SCRIPT_DIR,
                       prefix::AbstractString=PREFIX,
                       dark::Bool=true,
                       show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    print_env_info(outdir, n_rounds, extra_rounds, n_chains)
    dbg("map frame: ", flip ? "map X = −Δα* (OCTOLM_FLIP=true; the likelihood's convention)" :
                              "map X = +Δα* (page; the likelihood reads it mirrored in RA)")
    set_plot_theme!(; dark)

    outputs = String[]
    function savefig!(fig, name)
        fig === nothing && return nothing
        f = fig isa Octofitter.OctoPlotResult ? fig.figure : fig
        p = joinpath(outdir, prefix * name)
        save(p, f)
        push!(outputs, p)
        dbg("  saved ", p)
        show_plots && display(f)
        return f
    end
    inout(f) = cd(f, outdir)

    # ---- Synthetic data and model -------------------------------------------------------
    data = stage("Synthetic likelihood maps (template orbit, 3 peaks per epoch; page)") do
        synthetic_maps(; flip)
    end
    soft_stage("Likelihood maps (page's imview of 10^logL)") do
        savefig!(maps_figure(data), "maps.png")
    end
    sys, loglikemap = stage("Bodies, LogLikelihoodMapObs \"GRAVITY\", System") do
        build_system(data)
    end
    DEBUG[] && display(sys)
    model = stage("Compile LogDensityModel") do
        Octofitter.LogDensityModel(sys)
    end
    DEBUG[] && display(model)
    stage("initialize!(model) (page)") do
        Octofitter.initialize!(model)
    end

    # ---- Sampling (page) ------------------------------------------------------------------
    chain, pt = stage("octofit_pigeons (n_rounds=$n_rounds" *
                      (n_chains === nothing ? "" : ", n_chains=$n_chains") * "; page)") do
        n_chains === nothing ? octofit_pigeons(model; n_rounds) :
                               octofit_pigeons(model; n_rounds, n_chains)
    end
    diag1 = soft_stage("Pigeons diagnostics") do
        pigeons_diag(pt, "after $n_rounds rounds")
    end
    DEBUG[] && display(chain)
    truth_table(chain; flip)

    # ---- Analysis (page) ------------------------------------------------------------------
    soft_stage("octoplot (page)") do
        savefig!(inout(() -> octoplot(model, chain)), "octoplot.png")
    end
    soft_stage("Corner plot (small=true; page; light theme)") do
        savefig!(inout(() -> light(() -> octocorner(model, chain; small=true))), "corner_small.png")
    end
    pos = soft_stage("Posterior predictive positions at both epochs (page)") do
        predicted_positions(model, chain, data.epochs)
    end
    if pos !== nothing
        for k in 1:2
            soft_stage("Posterior predictive pairplot, epoch $k (page)") do
                savefig!(predictive_pairplot(pos[k]...), "predictive_epoch$k.png")
            end
        end
        soft_stage("Predicted positions over the likelihood maps") do
            savefig!(maps_figure(data; posterior=pos), "maps_posterior.png")
        end
        soft_stage("Peak assignment") do
            peak_assignment(pos, data; label="after $n_rounds rounds")
        end
    end
    p1 = joinpath(outdir, prefix * "chain.fits")
    soft_stage("Save chain -> FITS (+ reload check)") do
        Octofitter.savechain(p1, chain)
        push!(outputs, p1)
        c2 = Octofitter.loadchain(p1; model)
        dbg("  reloaded chain size = ", size(c2), size(c2) == size(chain) ? "  (matches)" : "  (DIFFERS)")
    end

    # ---- Resume sampling (page) ------------------------------------------------------------
    diag2 = nothing
    if extra_rounds > 0
        chain2 = soft_stage("Resume: increment_n_rounds!(pt, $extra_rounds) + octofit_pigeons(pt) (page)") do
            Pigeons.increment_n_rounds!(pt, extra_rounds)
            c, pt2 = octofit_pigeons(pt)
            diag2 = pigeons_diag(pt2, "after $(n_rounds + extra_rounds) rounds")
            c
        end
        if chain2 !== nothing
            truth_table(chain2; flip)
            soft_stage("Peak assignment after the resume") do
                peak_assignment(predicted_positions(model, chain2, data.epochs), data;
                                label="after $(n_rounds + extra_rounds) rounds")
            end
            soft_stage("Corner plot after the resume (small=false; page; light theme)") do
                savefig!(inout(() -> light(() -> octocorner(model, chain2; small=false))),
                         "corner_full_resumed.png")
            end
            p2 = joinpath(outdir, prefix * "chain_resumed.fits")
            soft_stage("Save resumed chain -> FITS") do
                Octofitter.savechain(p2, chain2)
                push!(outputs, p2)
            end
        end
    end

    # ---- Summary ------------------------------------------------------------------------------
    lines = String[]
    for (lab, d) in (("first run", diag1), ("after resume", diag2))
        d === nothing && continue
        push!(lines, @sprintf("Pigeons %-13s log(Z₁/Z₀) = %.3f, Λ = %.2f, min(α) = %.3g",
                              lab * ":", d.logZ, d.Λ, d.min_α))
    end
    if diag1 !== nothing && diag2 !== nothing
        push!(lines, @sprintf("Δlog(Z₁/Z₀) from the resume = %+.3f (page: add rounds until this stops drifting)",
                              diag2.logZ - diag1.logZ))
    end
    foreach(l -> dbg(l), lines)
    soft_stage("Write summary") do
        ps = joinpath(outdir, prefix * "summary.txt")
        write(ps, join(lines, "\n") * "\n")
        push!(outputs, ps)
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f min", (time() - T0[]) / 60))
    return (; model, chain, pt, data, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default when the REPL has 1 thread)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString=PREFIX)
    T0[] = time()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOLM_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOLM_CHILD=1]")
    dbg("  Child output follows; its plots are shown here when it finishes.")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOLM_RELAUNCH\"]=\"false\".")
    t = time()
    ok = try
        run(pipeline(cmd; stdout=stdout, stderr=stderr))
        true
    catch err
        msg = err isa ProcessFailedException ?
              "exit code " * join((string(p.exitcode) for p in err.procs), ", ") :
              err isa InterruptException ? "interrupted (Ctrl+C)" :
              first(sprint(showerror, err), 300)
        dbg("✖ child process failed: ", msg, " — see the ✖ stage line in its output above")
        false
    end
    dbg(@sprintf("Child process finished in %.1f min (%s)", (time() - t) / 60,
                 ok ? "success" : "FAILED"))
    chain = nothing
    p = joinpath(outdir, prefix * "chain.fits")
    if isfile(p) && mtime(p) >= t
        chain = soft_stage("Load chain written by child") do
            Octofitter.loadchain(p)
        end
    end
    sm = joinpath(outdir, prefix * "summary.txt")
    if isfile(sm) && mtime(sm) >= t
        foreach(l -> dbg(l), eachline(sm))
    end
    new_files = [joinpath(outdir, f) for f in readdir(outdir)
                 if startswith(f, prefix) && isfile(joinpath(outdir, f)) &&
                    mtime(joinpath(outdir, f)) >= t]
    list_outputs(new_files)
    pngs = filter(p -> endswith(lowercase(p), ".png"), new_files)
    if isempty(pngs)
        dbg("No new plots were written by the child",
            ok ? "." : " (it failed before the plotting stages — see the ✖ line above).")
    elseif lowercase(get(ENV, "OCTOLM_SHOW_PLOTS", "true")) != "false"
        soft_stage("Show $(length(pngs)) saved plot(s) in the plot pane") do
            show_saved_pngs(pngs)
        end
    end
    return (; relaunched=true, success=ok, chain, outputs=new_files)
end

function png_rank(p)
    f = basename(p)
    for (k, key) in enumerate(("maps.png", "octoplot", "corner_small", "predictive_epoch1",
                               "predictive_epoch2", "maps_posterior", "corner_full"))
        occursin(key, f) && return k
    end
    return 99
end

function show_saved_pngs(pngs; max_width::Integer=1400)
    for p in sort(pngs; by=png_rank)
        img = CairoMakie.Makie.FileIO.load(p)
        h, w = size(img)
        s = min(1.0, max_width / w)
        fig = Figure(size=(round(Int, w * s), round(Int, h * s)), figure_padding=0,
                     backgroundcolor=:white)
        ax = Axis(fig[1, 1], aspect=DataAspect())
        image!(ax, rotr90(img))
        hidedecorations!(ax)
        hidespines!(ax)
        dbg("  showing ", basename(p), " (", w, "×", h, ")")
        display(fig)
    end
    return nothing
end

println(@sprintf("[LM] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterLikeMap

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterLikelihoodMap.jl` -> runs
#   - VSCode "Execute active File in REPL" -> runs; with 1 thread it relaunches
#     in `julia --threads=auto` (OCTOLM_RELAUNCH=false to disable,
#     OCTOLM_THREADS to set the count)
#   - ENV["OCTOLM_AUTORUN"] = "false"; include() -> loads module only
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOLM_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOLM_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOLM_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global lm_result = OctofitterLikeMap.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOLM_THREADS", "auto"))
            println("[LM] Child run finished. `lm_result` has fields: relaunched, success, ",
                    "chain (loaded from FITS), outputs")
        else
            global lm_result = OctofitterLikeMap.run_likemap(show_plots=!is_child)
            is_child || println("[LM] Result stored in `lm_result` (fields: model, chain, pt, ",
                                "data, outputs)")
        end
    end
end

nothing
