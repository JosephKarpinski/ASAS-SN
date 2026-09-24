#=
================================================================================
 OctofitterCoplanar.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial
 "Hierarchical Co-Planar, Near-Resonant Model" (HR 8799 b & c):
   https://sefffal.github.io/Octofitter.jl/dev/fit-coplanar/

 Data: HR 8799 b (17 epochs) and c (13 epochs) relative astrometry
 (collated by J. Wang, whereistheplanet.com).

 Models (Jacobi hierarchy: c orbits A, b orbits the A+c barycentre;
 circular orbits, e = ω = 0; P_c = P_nominal·P_mul_c, P_b = 2·P_nominal·P_mul_b):
   :exact   — exactly co-planar: one system-level i and Ω shared by both planets
   :approx  — approximately co-planar: separate i, Ω per planet, plus a prior
              on the derived mutual inclination, truncated Normal(0, 10°)
 Optional dynamical priors (tutorial's last section, off by default):
   OrbitOrderPrior(c, b), NonCrossingPrior, HillStabilityPrior

 Pipeline per model:
   System -> LogDensityModel -> initialize! (tutorial guesses) -> octoplot(init)
   -> octofit_pigeons (parallel tempering) -> octoplot, octocorner,
      period-ratio histogram (+ mutual-inclination histogram for :approx)
   -> Pigeons log-evidence (stepping stone) -> FITS chain
 Then: comparison of the two models (period ratio overlay, summary table,
 log-evidence difference).

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `coplanar_result`)
   Shell:    julia --threads=auto OctofitterCoplanar.jl
   Library:  ENV["OCTOCP_AUTORUN"] = "false"; include("OctofitterCoplanar.jl")
             res = OctofitterCoplanar.run_coplanar(models=(:approx,), n_rounds=9)

 Pigeons parallel tempering is much faster with threads. Julia can't add
 threads to a running session, so when the REPL has only 1 thread the file
 automatically re-runs itself in a child `julia --threads=auto` process
 (same v9 environment; output streams into the REPL; chains are loaded back
 afterwards). ENV["OCTOCP_RELAUNCH"] = "false" disables this;
 ENV["OCTOCP_THREADS"] = "8" sets the count. For interactive work with
 threads in the REPL itself, set VSCode's "Julia: Num Threads" to auto.

 Parallax prior: `gaia_plx(gaia_id=2832463659640297472)` queries the Gaia
 archive (needs network). If that fails, a STAND-IN prior
 truncated(Normal(24.4549, 0.05), lower=0.1) centred on the tutorial's
 starting value is used and flagged loudly — it is not a catalogue value.

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; ENV["OCTOCP_ENV_MODE"] = "local" uses ./octofitter_v9_env). Needs
 Octofitter v9, Distributions, CairoMakie, PairPlots, PlanetOrbits, Pigeons,
 MCMCChains.

 Outputs (prefix "coplanar_") go to the directory containing this file.

 Changelog
 ---------
 v1.1  2026-09-24  Automatic threaded relaunch: in a 1-thread session the
                   auto-run starts a child `julia --threads=auto` with the
                   same project, streams its output, then loads the FITS
                   chains it wrote into `coplanar_result`. Child runs with
                   show_plots=false. Controls: OCTOCP_RELAUNCH, OCTOCP_THREADS.
 v1.0  2026-09-24  Initial version on the OctofitterObsPrior v1.0.1
                   scaffolding ([OCP +t] debug stages, soft optional stages,
                   env bootstrap, v9 guard, @__DIR__ outputs, dark theme with
                   light corner plots, colours matched to PairPlots order).
                   Adds: two-planet Jacobi hierarchy; system-level shared
                   i/Ω/P_nominal; mutual-inclination prior; optional
                   dynamical-stability priors; gaia_plx with flagged
                   fallback; Pigeons parallel tempering with evidence
                   report; data-preview, period-ratio and mutual-inclination
                   figures; exact-vs-approx comparison.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "Distributions", "CairoMakie", "PairPlots",
                "PlanetOrbits", "Pigeons", "MCMCChains"),
    mode     = lowercase(get(ENV, "OCTOCP_ENV_MODE", "temp"))

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
            println("[OCP] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[OCP] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(miss; preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[OCP] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[OCP] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[OCP] Existing v9 environment found at ", Base.active_project())
        else
            specs = [n == "Octofitter" ? Pkg.PackageSpec(name=n, version="9") :
                                         Pkg.PackageSpec(name=n) for n in required]
            Pkg.add(specs)
        end
        println("[OCP] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[OCP] Loading Octofitter, Distributions, CairoMakie, PairPlots, PlanetOrbits, ",
        "Pigeons, MCMCChains (first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterCoplanar

const LOAD_T0 = time()

using Octofitter
using Distributions
using CairoMakie
using PairPlots
using PlanetOrbits
using Pigeons            # loads Octofitter's Pigeons extension (octofit_pigeons)
import MCMCChains
using Printf
import Statistics

export run_coplanar, astrometry_table_b, astrometry_table_c, build_star,
       build_planet, build_observations, build_coplanar_system, parallax_prior,
       set_plot_theme!

const VERSION_STRING = "1.1"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterCoplanar needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up Octofitter v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const HR8799_GAIA_ID = 2832463659640297472
const REF_EPOCH = 59454.231          # MJD, reference epoch for θ (tutorial)

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterCoplanar.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[OCP +%7.1fs] ", time() - T0[]), msg...)
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
    dbg("OctofitterCoplanar v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    Threads.nthreads() == 1 &&
        dbg("  NOTE: 1 thread — Pigeons will be slow. Set VSCode 'Julia: Num Threads' ",
            "to auto (or julia --threads=auto) and restart the REPL.")
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

"Summary rows: (label, column, transform). Missing columns are skipped."
const SUMMARY_ROWS = [
    ("plx [mas]",       :plx,         identity),
    ("A_mass [M⊙]",     :A_mass,      identity),
    ("P_nominal [yr]",  :P_nominal,   identity),
    ("c_mass [M_jup]",  :c_mass,      x -> x ./ mjup),
    ("b_mass [M_jup]",  :b_mass,      x -> x ./ mjup),
    ("c_P_mul",         :c_P_mul,     identity),
    ("b_P_mul",         :b_P_mul,     identity),
    ("c_P [yr]",        :c_P,         x -> x ./ year2day_julian),
    ("b_P [yr]",        :b_P,         x -> x ./ year2day_julian),
    ("i [deg] (shared)", :i,          x -> rad2deg.(x)),
    ("Ω [deg] (shared)", :Ω,          x -> rad2deg.(x)),
    ("c_i [deg]",       :c_i,         x -> rad2deg.(x)),
    ("b_i [deg]",       :b_i,         x -> rad2deg.(x)),
    ("c_Ω [deg]",       :c_Ω,         x -> rad2deg.(x)),
    ("b_Ω [deg]",       :b_Ω,         x -> rad2deg.(x)),
    ("mut_inc_b_c [deg]", :mut_inc_b_c, x -> rad2deg.(x)),
]

period_ratio(chain) = colvec(chain, :b_P) ./ colvec(chain, :c_P)

function print_summary(chain, label)
    DEBUG[] || return nothing
    dbg("Posterior median [16th, 84th] — ", label, " (", size(chain, 1), " samples):")
    for (lab, p, f) in SUMMARY_ROWS
        haspar(chain, p) || continue
        q = qline(f(colvec(chain, p)))
        dbg(@sprintf("    %-20s %12.4f  [%12.4f, %12.4f]", lab, q[2], q[1], q[3]))
    end
    if haspar(chain, :b_P) && haspar(chain, :c_P)
        q = qline(period_ratio(chain))
        dbg(@sprintf("    %-20s %12.4f  [%12.4f, %12.4f]", "P_b / P_c", q[2], q[1], q[3]))
    end
    return nothing
end

function print_comparison(chains)          # iterable of kind => chain pairs
    DEBUG[] || return nothing
    chains = collect(chains)
    ks = first.(chains)
    lookup = Dict(chains)
    dbg("Model comparison — median [16th, 84th]:")
    dbg("    ", rpad("", 20), join((rpad(string(k), 34) for k in ks)))
    rows = vcat(SUMMARY_ROWS, [("P_b / P_c", :__ratio, identity)])
    for (lab, p, f) in rows
        cells = String[]
        any_present = false
        for k in ks
            c = lookup[k]
            x = p === :__ratio ? (haspar(c, :b_P) && haspar(c, :c_P) ? period_ratio(c) : nothing) :
                (haspar(c, p) ? f(colvec(c, p)) : nothing)
            if x === nothing
                push!(cells, rpad("—", 34))
            else
                any_present = true
                q = qline(x)
                push!(cells, rpad(@sprintf("%9.3f [%9.3f, %9.3f]", q[2], q[1], q[3]), 34))
            end
        end
        any_present && dbg("    ", rpad(lab, 20), join(cells))
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

# Match PairPlots' series order (1st = blue, 2nd = orange)
const MODEL_COLORS = Dict(:exact => "#0072B2", :approx => "#E69F00")
const MODEL_LABELS = Dict(:exact => "exactly co-planar", :approx => "approx. co-planar")
const C_B = "#56B4E9"
const C_C = "#E69F00"

"Sky-plane preview of the data (east left, as in octoplot)."
function data_figure(tb, tc)
    fig = Figure(size=(760, 720))
    ax = Axis(fig[1, 1], aspect=DataAspect(), xreversed=true,
              xlabel="Δα* [mas]", ylabel="Δδ [mas]",
              title="HR 8799 b & c relative astrometry")
    errorbars!(ax, tb.ra, tb.dec, tb.σ_dec, color=(C_B, 0.6))
    errorbars!(ax, tb.ra, tb.dec, tb.σ_ra, direction=:x, color=(C_B, 0.6))
    errorbars!(ax, tc.ra, tc.dec, tc.σ_dec, color=(C_C, 0.6))
    errorbars!(ax, tc.ra, tc.dec, tc.σ_ra, direction=:x, color=(C_C, 0.6))
    scatter!(ax, tb.ra, tb.dec, color=C_B, markersize=8, label="b ($(length(tb.epoch)) epochs)")
    scatter!(ax, tc.ra, tc.dec, color=C_C, markersize=8, label="c ($(length(tc.epoch)) epochs)")
    scatter!(ax, [0.0], [0.0], marker=:star5, markersize=28, color=:white, label="HR 8799 A")
    axislegend(ax, position=:lb)
    return fig
end

"Histogram of P_b/P_c for each fitted model, with the nominal 2:1 marked."
function period_ratio_figure(chains)      # iterable of kind => chain pairs
    fig = Figure(size=(820, 420))
    ax = Axis(fig[1, 1], xlabel="period ratio P_b / P_c", ylabel="density",
              title="Period ratio")
    for (k, c) in chains
        hist!(ax, period_ratio(c); bins=50, normalization=:pdf,
              color=(get(MODEL_COLORS, k, "#999999"), 0.5),
              label=get(MODEL_LABELS, k, string(k)))
    end
    vlines!(ax, [2.0], color=(:white, 0.7), linestyle=:dash, label="2:1")
    axislegend(ax, position=:rt)
    return fig
end

"Histogram of the derived mutual inclination with the Normal(0, σ) prior scale marked."
function mutual_inclination_figure(chain; sigma_deg=10.0)
    x = rad2deg.(colvec(chain, :mut_inc_b_c))
    fig = Figure(size=(820, 420))
    ax = Axis(fig[1, 1], xlabel="mutual inclination [deg]", ylabel="density",
              title="Mutual inclination (prior: truncated Normal(0, $(sigma_deg)°))")
    hist!(ax, x; bins=50, normalization=:pdf, color=(MODEL_COLORS[:approx], 0.6))
    vlines!(ax, [sigma_deg], color=(:white, 0.6), linestyle=:dash)
    return fig
end

# -----------------------------------------------------------------------------
# Data and model components (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
"HR 8799 b relative astrometry [mas], MJD epochs."
function astrometry_table_b()
    return Table(;
        epoch = [53200.0, 54314.0, 54398.0, 54727.0, 55042.0, 55044.0, 55136.0, 55390.0, 55499.0, 55763.0, 56130.0, 56226.0, 56581.0, 56855.0, 58798.03906, 59453.245, 59454.231],
        ra    = [1471.0, 1504.0, 1500.0, 1516.0, 1526.0, 1531.0, 1524.0, 1532.0, 1535.0, 1541.0, 1545.0, 1549.0, 1545.0, 1560.0, 1611.002, 1622.924, 1622.872],
        dec   = [887.0, 837.0, 836.0, 818.0, 797.0, 794.0, 795.0, 783.0, 766.0, 762.0, 747.0, 743.0, 724.0, 725.0, 604.893, 570.534, 571.296],
        σ_ra  = [6.0, 3.0, 7.0, 4.0, 4.0, 7.0, 10.0, 5.0, 15.0, 5.0, 5.0, 4.0, 22.0, 13.0, 0.133, 0.32, 0.204],
        σ_dec = [6.0, 3.0, 7.0, 4.0, 4.0, 7.0, 10.0, 5.0, 15.0, 5.0, 5.0, 4.0, 22.0, 13.0, 0.199, 0.296, 0.446],
        cor   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.406, -0.905, -0.79],
    )
end

"HR 8799 c relative astrometry [mas], MJD epochs."
function astrometry_table_c()
    return Table(;
        epoch = [53200.0, 54314.0, 54398.0, 54727.0, 55042.0, 55136.0, 55390.0, 55499.0, 55763.0, 56130.0, 56226.0, 56581.0, 56855.0],
        ra    = [-739.0, -683.0, -678.0, -663.0, -639.0, -636.0, -619.0, -607.0, -595.0, -578.0, -572.0, -542.0, -540.0],
        dec   = [612.0, 671.0, 678.0, 693.0, 712.0, 720.0, 728.0, 744.0, 747.0, 761.0, 768.0, 784.0, 799.0],
        σ_ra  = [6.0, 4.0, 7.0, 3.0, 4.0, 9.0, 4.0, 12.0, 4.0, 5.0, 3.0, 22.0, 12.0],
        σ_dec = [6.0, 4.0, 7.0, 3.0, 4.0, 9.0, 4.0, 12.0, 4.0, 5.0, 3.0, 22.0, 12.0],
        cor   = zeros(13),
    )
end

"""
    parallax_prior(; source=:gaia, gaia_id=HR8799_GAIA_ID)

`source=:gaia` queries the Gaia archive via `gaia_plx` (network). On failure,
or with `source=:standin`, returns truncated(Normal(24.4549, 0.05), lower=0.1),
a stand-in centred on the tutorial's starting value (NOT a catalogue value).
"""
function parallax_prior(; source::Symbol=:gaia, gaia_id::Integer=HR8799_GAIA_ID)
    standin = truncated(Normal(24.4549, 0.05), lower=0.1)
    if source === :gaia
        try
            d = gaia_plx(; gaia_id=gaia_id)
            dbg("  plx prior from Gaia (source_id ", gaia_id, "): ", d)
            return d
        catch err
            dbg("  ⚠ gaia_plx failed (", sprint(showerror, err), ")")
            dbg("  ⚠ USING STAND-IN plx prior ", standin,
                " — centred on the tutorial's start value, not a catalogue value")
            return standin
        end
    end
    dbg("  plx prior (stand-in, by request): ", standin)
    return standin
end

"""
    build_star(; mass_prior=truncated(Normal(1.5, 0.02), lower=0.1))
"""
function build_star(; mass_prior::Distribution=truncated(Normal(1.5, 0.02), lower=0.1))
    dbg("  star 'A': mass ~ ", mass_prior)
    vars = @eval @variables begin
        mass ~ $mass_prior           # [M⊙]
    end
    return Body(name="A", variables=vars)
end

"""
    build_planet(name, about; coplanar, period_factor, P_mul_prior,
                 mass_max_mjup=12, epoch=REF_EPOCH)

Circular (e = ω = 0) planet with P = period_factor · system.P_nominal · P_mul
[days]. `coplanar=true` takes i and Ω from the system block; otherwise the
planet gets its own i ~ Sine(), Ω ~ UniformCircular().
"""
function build_planet(name::AbstractString, about; coplanar::Bool,
                      period_factor::Real, P_mul_prior::Distribution,
                      mass_max_mjup::Real=12, epoch::Real=REF_EPOCH)
    mp = Uniform(0, mass_max_mjup * mjup)
    pf = Float64(period_factor)
    ep = Float64(epoch)
    dbg("  planet '", name, "': about ",
        about isa Tuple ? "barycentre of $(length(about)) bodies (Jacobi)" : "host star",
        ", mass ~ U(0, ", mass_max_mjup, " M_jup), P = ", pf, "·P_nominal·P_mul, P_mul ~ ",
        P_mul_prior, coplanar ? ", i/Ω shared (system)" : ", own i/Ω")
    vars = if coplanar
        @eval @variables begin
            mass ~ $mp                   # [M⊙]
            e = 0.0
            ω = 0.0
            i = system.i
            Ω = system.Ω
            P_mul ~ $P_mul_prior
            P = $pf * system.P_nominal * P_mul * year2day_julian   # [days]
            θ ~ UniformCircular()
            epoch = $ep
        end
    else
        @eval @variables begin
            mass ~ $mp
            e = 0.0
            ω = 0.0
            i ~ Sine()
            Ω ~ UniformCircular()
            P_mul ~ $P_mul_prior
            P = $pf * system.P_nominal * P_mul * year2day_julian
            θ ~ UniformCircular()
            epoch = $ep
        end
    end
    return Body(name=name, about=about, variables=vars)
end

"""
    build_observations(A, c, b; fixed_systematics=true, stability_priors=false)

Relative astrometry for b and c versus A. `fixed_systematics` adds the
tutorial's fixed jitter = 0, northangle = 0, platescale = 1 block (the
exact-model form; numerically identical to omitting it). `stability_priors`
appends OrbitOrderPrior, NonCrossingPrior and HillStabilityPrior.
"""
function build_observations(A, c, b; fixed_systematics::Bool=true,
                            stability_priors::Bool=false)
    tb, tc = astrometry_table_b(), astrometry_table_c()
    dbg("  obs 'GPI_b': ", length(tb.epoch), " epochs; 'GPI_c': ", length(tc.epoch), " epochs")
    mk(tbl, tgt, nm) = if fixed_systematics
        vars = @eval @variables begin
            jitter = 0
            northangle = 0
            platescale = 1
        end
        RelAstromObs(tbl; target=tgt, ref=A, name=nm, variables=vars)
    else
        RelAstromObs(tbl; target=tgt, ref=A, name=nm)
    end
    obs = Any[mk(tb, b, "GPI_b"), mk(tc, c, "GPI_c")]
    if stability_priors
        push!(obs, OrbitOrderPrior(c, b))                       # keeps c interior to b
        push!(obs, NonCrossingPrior(bodies=(b, c)))             # apsides may not cross
        push!(obs, HillStabilityPrior(bodies=(b, c)))           # basic stability
        dbg("  + OrbitOrderPrior, NonCrossingPrior, HillStabilityPrior")
    end
    return obs
end

"""
    build_coplanar_system(kind; plx_prior, P_nominal_prior=Uniform(50, 300),
                          mutinc_sigma_deg=10.0, stability_priors=false,
                          mass_max_mjup=12, epoch=REF_EPOCH)

`kind = :exact` (shared i, Ω) or `:approx` (own i, Ω + mutual-inclination
prior). Fresh bodies/observations are built on every call.
"""
function build_coplanar_system(kind::Symbol; plx_prior,
                               P_nominal_prior::Distribution=Uniform(50, 300),
                               mutinc_sigma_deg::Real=10.0,
                               stability_priors::Bool=false,
                               mass_max_mjup::Real=12,
                               epoch::Real=REF_EPOCH)
    kind in (:exact, :approx) || error("kind must be :exact or :approx, got $kind")
    exact = kind === :exact
    A = build_star()
    c = build_planet("c", A; coplanar=exact, period_factor=1,
                     P_mul_prior=truncated(Normal(1, 0.1), lower=0.1),
                     mass_max_mjup=mass_max_mjup, epoch=epoch)
    b = build_planet("b", (A, c); coplanar=exact, period_factor=2,
                     P_mul_prior=Normal(1, 0.1),
                     mass_max_mjup=mass_max_mjup, epoch=epoch)
    obs = build_observations(A, c, b; fixed_systematics=exact,
                             stability_priors=stability_priors)
    vars = if exact
        @eval @variables begin
            plx ~ $plx_prior
            i ~ Sine()                       # shared inclination
            Ω ~ UniformCircular()            # shared ascending node
            P_nominal ~ $P_nominal_prior     # [Julian years], nominal P_c
        end
    else
        mi = truncated(Normal(0, deg2rad(Float64(mutinc_sigma_deg))), lower=0)
        dbg("  mutual-inclination prior: ", mi, " [rad]")
        @eval @variables begin
            plx ~ $plx_prior
            # Mutual inclination between the two orbital planes
            mut_inc_b_c = acos(
                cos(b.i) * cos(c.i) +
                sin(b.i) * sin(c.i) * cos(b.Ω - c.Ω)
            )
            mut_inc_b_c ~ $mi
            P_nominal ~ $P_nominal_prior
        end
    end
    name = exact ? "HR8799_res_co" : "HR8799_approx_res_co"
    dbg("  system '", name, "': bodies A, c, b; ", length(obs), " observation terms; ",
        "P_nominal ~ ", P_nominal_prior)
    return System(name=name, bodies=[A, c, b], observations=obs, variables=vars)
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_coplanar(; models=(:exact, :approx), n_rounds=10, plx_source=:gaia,
                   stability_priors=false, mutinc_sigma_deg=10.0,
                   outdir=SCRIPT_DIR, prefix="coplanar_", dark=true,
                   corner_dark=false, show_plots=true)

Run the tutorial's models with Pigeons parallel tempering (`n_rounds` rounds
→ 2^n_rounds scans) and compare them. Returns a NamedTuple of results.
"""
function run_coplanar(; models=(:exact, :approx),
                        n_rounds::Integer=10,
                        plx_source::Symbol=:gaia,
                        stability_priors::Bool=false,
                        mutinc_sigma_deg::Real=10.0,
                        outdir::AbstractString=SCRIPT_DIR,
                        prefix::AbstractString="coplanar_",
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

    soft_stage("Data preview plot") do
        savefig!(data_figure(astrometry_table_b(), astrometry_table_c()), "data.png")
    end

    plx_prior = stage("Parallax prior") do
        parallax_prior(; source=plx_source)
    end

    systems  = Dict{Symbol,Any}()
    modelsd  = Dict{Symbol,Any}()
    inits    = Dict{Symbol,Any}()
    chains   = Dict{Symbol,Any}()   # insertion order irrelevant; we iterate `models`
    pts      = Dict{Symbol,Any}()
    logZ     = Dict{Symbol,Float64}()
    paths    = Dict{Symbol,String}()

    for kind in models
        tag = string(kind)
        sys = stage("[$tag] Build system") do
            build_coplanar_system(kind; plx_prior, mutinc_sigma_deg, stability_priors)
        end
        DEBUG[] && display(sys)
        systems[kind] = sys

        model = stage("[$tag] Compile LogDensityModel") do
            Octofitter.LogDensityModel(sys)
        end
        DEBUG[] && display(model)
        modelsd[kind] = model

        init = stage("[$tag] Initialize starting points (tutorial guesses)") do
            initialize!(model, (;
                plx = 24.4549,
                P_nominal = 230,
                bodies = (;
                    A = (; mass = 1.48),
                    b = (; mass = 5.73 * mjup),
                    c = (; mass = 5.14 * mjup),
                ),
            ))
        end
        inits[kind] = init

        soft_stage("[$tag] Plot initial guess") do
            savefig!(inout(() -> octoplot(model, init)), "$(tag)_init.png")
        end

        res = stage("[$tag] Pigeons parallel tempering (n_rounds=$n_rounds → $(2^n_rounds) scans)") do
            octofit_pigeons(model; n_rounds=n_rounds)
        end
        chain, pt = res
        chains[kind] = chain
        pts[kind] = pt
        dbg("  [$tag] chain size = ", size(chain))

        lz = soft_stage("[$tag] Log-evidence (stepping stone)") do
            z = Pigeons.stepping_stone(pt)
            dbg(@sprintf("  [%s] log Z ≈ %.2f", tag, z))
            z
        end
        lz !== nothing && (logZ[kind] = lz)
        soft_stage("[$tag] Global communication barrier Λ") do
            dbg(@sprintf("  [%s] Λ = %.2f  (more chains than ~2Λ recommended)", tag,
                         Pigeons.global_barrier(pt)))
        end

        print_summary(chain, tag)

        stage("[$tag] Orbit plot") do
            savefig!(inout(() -> octoplot(model, chain)), "$(tag)_orbits.png")
        end
        soft_stage("[$tag] Corner plot") do
            savefig!(inout(() -> corner(model, chain; dark=corner_dark)), "$(tag)_corner.png")
        end
        soft_stage("[$tag] Period-ratio histogram") do
            savefig!(period_ratio_figure([kind => chain]), "$(tag)_period_ratio.png")
        end
        if kind === :approx
            soft_stage("[$tag] Mutual-inclination histogram") do
                q = qline(rad2deg.(colvec(chain, :mut_inc_b_c)))
                dbg(@sprintf("  mutual inclination: %.2f° [%.2f°, %.2f°] — note a half-normal(0, %g°) ",
                             q[2], q[1], q[3], mutinc_sigma_deg),
                    "prior need not peak at 0 once combined with isotropic orbit planes")
                savefig!(mutual_inclination_figure(chain; sigma_deg=mutinc_sigma_deg),
                         "$(tag)_mutual_inclination.png")
            end
        end

        p = joinpath(outdir, prefix * "$(tag)_chain.fits")
        soft_stage("[$tag] Save chain -> FITS (+ reload check)") do
            Octofitter.savechain(p, chain)
            push!(outputs, p)
            paths[kind] = p
            c2 = Octofitter.loadchain(p; model)
            dbg("  reloaded chain size = ", size(c2),
                size(c2) == size(chain) ? "  (matches)" : "  (DIFFERS from saved)")
        end
    end

    # ---- Comparison ---------------------------------------------------------
    if length(chains) > 1
        ordered = [k => chains[k] for k in models if haskey(chains, k)]
        print_comparison(ordered)
        soft_stage("Period-ratio comparison plot") do
            savefig!(period_ratio_figure(ordered), "compare_period_ratio.png")
        end
        if haskey(logZ, :exact) && haskey(logZ, :approx)
            Δ = logZ[:exact] - logZ[:approx]
            dbg(@sprintf("  log Z(exact) − log Z(approx) = %.2f  (positive favours exact co-planarity; ", Δ),
                "sensitive to priors and to Pigeons run length)")
        end
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f s", time() - T0[]))

    return (; systems, models=modelsd, init_chains=inits, chains, pts, logZ,
              outputs, chain_paths=paths, plx_prior)
end

# -----------------------------------------------------------------------------
# Threaded relaunch
# -----------------------------------------------------------------------------
"""
    relaunch_threaded(file; threads="auto", outdir=SCRIPT_DIR, prefix="coplanar_")

Julia's thread count is fixed at startup, so a 1-thread REPL can't gain
threads. This runs `file` in a child Julia process with `--threads=threads`
and the same active project (the v9 env), streaming its output here. The
child writes plots/chains to `outdir`; afterwards the FITS chains it wrote
are loaded back into this session. Returns a NamedTuple.
"""
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR,
                           prefix::AbstractString="coplanar_")
    T0[] = time()
    cmd = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(cmd, "OCTOCP_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", cmd)
    dbg("  Child output follows. Plots go to files only (not the plot pane).")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOCP_RELAUNCH\"]=\"false\".")
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
    paths  = Dict{Symbol,String}()
    for kind in (:exact, :approx)
        p = joinpath(outdir, prefix * "$(kind)_chain.fits")
        (isfile(p) && mtime(p) >= t) || continue
        c = soft_stage("Load $(kind) chain written by child") do
            Octofitter.loadchain(p)
        end
        if c !== nothing
            chains[kind] = c
            paths[kind] = p
            dbg("  ", kind, " chain size = ", size(c))
        end
    end
    new_files = [joinpath(outdir, f) for f in readdir(outdir)
                 if startswith(f, prefix) && mtime(joinpath(outdir, f)) >= t]
    list_outputs(new_files)
    return (; relaunched=true, success=ok, chains, chain_paths=paths, outputs=new_files)
end

println(@sprintf("[OCP] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterCoplanar

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterCoplanar.jl`  -> runs
#   - VSCode "Execute active File in REPL"          -> runs (interactive session)
#   - ENV["OCTOCP_AUTORUN"] = "false"; include()    -> loads module only
# With 1 thread, the run is relaunched in a child `julia --threads=auto`
# process (same v9 env); ENV["OCTOCP_RELAUNCH"]="false" runs in-process instead,
# ENV["OCTOCP_THREADS"]="8" (etc.) picks the thread count.
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOCP_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOCP_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOCP_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global coplanar_result = OctofitterCoplanar.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOCP_THREADS", "auto"))
            println("[OCP] Child run finished. `coplanar_result` has fields: relaunched, ",
                    "success, chains (loaded from FITS), chain_paths, outputs")
        else
            global coplanar_result = OctofitterCoplanar.run_coplanar(
                models=(:exact, :approx), n_rounds=10, show_plots=!is_child)
            is_child || println("[OCP] Result stored in `coplanar_result` (fields: systems, ",
                "models, init_chains, chains, pts, logZ, outputs, chain_paths, plx_prior)")
        end
    end
end

nothing
