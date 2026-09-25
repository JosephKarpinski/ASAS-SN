#=
================================================================================
 OctofitterDR4Sim.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Simulating and Fitting Gaia
 DR4 Data":
   https://sefffal.github.io/Octofitter.jl/dev/gaia-dr4-simulation/

 Part 1 (:single, default) — the page's main example. A real star (Gaia DR3
 5064625130502952704, G = 6.94) sets the realism: its GOST scan forecast gives
 the transit epochs, scan angles (degrees) and parallax factors, and its G23H
 row gives the calibrated along-scan noise (g23h_scan_uncertainty: σ_AL,
 σ_att per CCD; σ_calib per transit; σ_transit_true ≈ 0.153 mas). A dark
 companion is injected with generate_from_params (page truth: m_b = 10 M_jup,
 a = 2 AU, e = 0.01, i = 0.01 rad, Ω = 0.6, θ = 3.52 at the mean transit
 epoch, plx = 16.139 mas) and recovered with Pigeons (n_rounds = 9).

 Part 1b (:formal, optional) — the page's "what a real DR4 table would quote"
 note, carried out: same simulated abscissae, but centroid_pos_error_al set
 to σ_transit_formal and astrometric_jitter free; jitter should come back
 at ≈ σ_calib (0.149 mas).

 Part 2 (:binary, optional) — "Fitting multiple bound sources": the GJ 15
 A/B wide M-dwarf pair (3.56 pc, 34″), one GaiaDR4AstromObs per source with
 a shared frame anchored on star A (anchor_offsets on system_interim), the
 relative orbit as a Cartesian state vector, per-source G23H noise and GOST
 geometry, truth m_A = 0.38, m_B = 0.15 M⊙, z = 60 AU, NUTS (octofit,
 1000 + 1000). Needs the network (two DR3 queries, two GOST queries).

 Plots: simulated abscissae and scan geometry, starting-point octoplot,
 octoplot, single-draw octoplot, gaiastarplot, the page's posterior-vs-truth
 orbit tracks, light corner, marginals with truth lines; for :binary the
 page's mass-fraction and z–M_tot figure. Posterior-vs-truth tables printed.

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `dr4sim_result`)
   Shell:    julia --threads=auto OctofitterDR4Sim.jl
   Library:  ENV["OCTOSIM_AUTORUN"] = "false"; include("OctofitterDR4Sim.jl")
             res = OctofitterDR4Sim.run_dr4sim(parts=(:single, :formal))

 Threads: the Pigeons fits relaunch a 1-thread REPL in a child
 `julia --threads=auto` (output streamed here; chains loaded back and saved
 plots shown in the plot pane afterwards). OCTOSIM_RELAUNCH=false disables
 this; OCTOSIM_THREADS sets the count.

 Convenience ENV settings (also passed to a relaunched child):
   OCTOSIM_PARTS   = "single" (default) | e.g. "single,formal,binary"
   OCTOSIM_ROUNDS  = "9" (default, page) — Pigeons rounds for single/formal
   OCTOSIM_TRUTH   = "page" (default: the page's manual parameters) | "prior"
                     (drawfrompriors, seeded)
   OCTOSIM_SEED    = "1" (default) — RNG seed for noise (and prior truth)
   OCTOSIM_CHAINS  = "" (default: Octofitter's n_chains) — Pigeons n_chains for
                     single/formal; raise (e.g. 32) if min(α) stays at 0
   OCTOSIM_NUTS    = "1000" (default, page) — adaptation = iterations for :binary
   OCTOSIM_CATALOG = "page" (default: the G23H noise rows the docs build uses)
                     | "full" (the 14 GB G23H DataDep)
   OCTOSIM_SHOW_PLOTS = "true" (default)   OCTOSIM_HEARTBEAT = "30" (default)

 Data: the page's cached GOST forecast for the part-1 star
 (GOST-42.03733343244703--31.42348623214663-dr4.csv) is downloaded once from
 the Octofitter repo (pinned commit) next to this file, where
 gaia_dr4_transit_template finds it (it looks up GOST-<ra>-<dec>-dr4.csv in
 the working directory). The DR3 solution is queried from the Gaia archive
 (cached); if that fails, part 1 uses the coordinates of the cached forecast.
 GOST answers for GJ 15 A/B are queried live and cached next to this file.

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOSIM_ENV_MODE=local uses ./octofitter_v9_env). Needs Octofitter
 v9, CairoMakie, PairPlots, Distributions, Pigeons.

 Outputs (prefix "dr4sim_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-25  Initial version on the OctofitterDR4 v1.0.2 scaffolding
                   ([DR4S +t] debug stages, soft optional stages, env bootstrap
                   with v9 guards, @__DIR__ outputs, dark theme with light
                   corner plots, threaded relaunch with exit-code-only failure
                   report and PNGs shown in the parent's plot pane, heartbeat
                   for network steps, pinned-commit data downloads, @eval-built
                   @variables, initialize! before sampling, single literal
                   @sprintf formats). Adds: G23H noise model and GOST transit
                   template; generate_from_params injection; posterior-vs-truth
                   tables and orbit tracks; formal-error/jitter variant; GJ 15
                   two-source anchored binary with NUTS.
 v1.0.1 2026-09-25 First run (part 1, 2.8 min): 191 transits, σ_transit_true
                   0.1525 mas (page 0.153); plx, a, P, m_b, pm and offsets
                   within ~1.2σ of the truth; i = 36 (+9/−14)° vs 0.57° truth,
                   reported as "+3.11σ". That is the Sine() prior's lack of
                   volume near i = 0 (P(i < 10°) = 1.5%), not a failed fit, so
                   the truth table now also gives the truth's percentile in the
                   posterior (flags outside the central 95%) and explains the
                   face-on case. OCTOSIM_CHAINS sets Pigeons n_chains
                   (min(α) stayed 0 at the default).
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "CairoMakie", "PairPlots", "Distributions", "Pigeons"),
    min_version = Dict("Octofitter" => v"9"),
    mode = lowercase(get(ENV, "OCTOSIM_ENV_MODE", "temp"))

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
            println("[DR4S] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[DR4S] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[DR4S] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[DR4S] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[DR4S] Existing v9 environment found at ", Base.active_project())
        else
            Pkg.add(spec.(collect(required)))
        end
        println("[DR4S] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[DR4S] Loading Octofitter, Pigeons, CairoMakie, PairPlots, Distributions ",
        "(first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterDR4Sim

const LOAD_T0 = time()

using Octofitter
using CairoMakie
using PairPlots
using Distributions
using Pigeons
import Downloads
import Random
import Statistics
using Printf

export run_dr4sim, set_plot_theme!

const VERSION_STRING = "1.0.1"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterDR4Sim needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const REF_EPOCH_MJD = 57936.375    # Gaia DR4 reference epoch, J2017.5
const DR3_EPOCH_MJD = 57388.5      # Gaia DR3 reference epoch, J2016.0

# ---- Part 1: the page's star, its cached GOST forecast and G23H noise row ----
const GAIA_ID = 5064625130502952704
const RA_GOST, DEC_GOST = 42.03733343244703, -31.42348623214663   # from the cached file's name
const OCTO_COMMIT = "8c6daaaf5292b6cbdb0aa4f4b1e9a1a5f0bd21f3"
const OCTO_RAW = "https://raw.githubusercontent.com/sefffal/Octofitter.jl/$(OCTO_COMMIT)/docs/src/"
const GOST_FILE = "GOST-42.03733343244703--31.42348623214663-dr4.csv"

"The G23H row columns the noise model reads (as substituted by the docs build)."
const CATALOG_ROW = (;
    gaia_source_id = GAIA_ID,
    phot_g_mean_mag_dr3 = 6.941057,
    sig_AL = 0.05813228279803717,
    sig_att_radec = 0.07765105456802447,
    sig_cal = 0.14901214069992602,
    astrometric_n_good_obs_al_dr3 = 882,
    astrometric_matched_transits_dr3 = 100,
)

const PRIORS = (
    m_b  = LogUniform(0.01mjup, 1000mjup),   # [M⊙]
    a_b  = LogUniform(0.01, 100),            # [AU]
    e_b  = Uniform(0, 0.99),
    plx  = Uniform(0.01, 100),               # [mas]
    jit  = LogUniform(0.00001, 10),          # [mas]
    off  = Normal(0, 10000),                 # [mas]
    pm   = Uniform(-1000, 1000),             # [mas/yr]
)

"The page's manual simulation truth (epoch filled in from the transits)."
page_truth(orbit_ref_epoch) = (
    plx = 16.138978209522527,
    bodies = (
        A = (mass = 1.0, flux = 1.0),
        b = (mass = 10.0 * Octofitter.mjup, a = 2.0, e = 0.01, ω = 0.01, i = 0.01, Ω = 0.6,
             θ = 3.5226970272017826, epoch = orbit_ref_epoch, flux = 0.0),
    ),
    observations = (
        GaiaDR4 = (astrometric_jitter = 0.015590772157762368, ra_offset_mas = 0.05413791355838311,
                   dec_offset_mas = 0.0889816167388366, pmra = 5.301074192615374,
                   pmdec = -24.188882826919325, ref_epoch = REF_EPOCH_MJD),
    ),
)

# ---- Part 2: GJ 15 A/B --------------------------------------------------------
const GJ15 = (
    id_A = 385334230892516480, id_B = 385334196532776576,
    cat_A = (; gaia_source_id = 385334230892516480, phot_g_mean_mag_dr3 = 7.218567,
              sig_AL = 0.07187400179146576, sig_att_radec = 0.07929606239862443,
              sig_cal = 0.14753503330357615, astrometric_n_good_obs_al_dr3 = 513,
              astrometric_matched_transits_dr3 = 59),
    cat_B = (; gaia_source_id = 385334196532776576, phot_g_mean_mag_dr3 = 9.686646,
              sig_AL = 0.05131486951008116, sig_att_radec = 0.07928952757954605,
              sig_cal = 0.20633005978862995, astrometric_n_good_obs_al_dr3 = 496,
              astrometric_matched_transits_dr3 = 57),
    mass_A = 0.38, mass_B = 0.15, z = 60.0,
)

const PART_LABELS = Dict(:single => "single star + dark companion (page part 1)",
                         :formal => "same data, formal errors + free jitter",
                         :binary => "GJ 15 A/B two-source fit (page part 2)")
const PART_ORDER = (:single, :formal, :binary)

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterDR4Sim.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[DR4S +%7.1fs] ", time() - T0[]), msg...)
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
            first(sprint(showerror, err), 2000))
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
            first(sprint(showerror, err), 1000))
        return nothing
    end
end

envbool(k, default) = lowercase(get(ENV, k, default)) != "false"
envint(k, default) = parse(Int, strip(get(ENV, k, default)))

"Run `f()` while a background task prints a line every `every` seconds."
function with_heartbeat(f, what::AbstractString;
                        every::Real=parse(Float64, get(ENV, "OCTOSIM_HEARTBEAT", "30")))
    done = Threads.Atomic{Bool}(false)
    t0 = time()
    hb = Threads.@spawn begin
        next = t0 + every
        while !done[]
            sleep(1)
            done[] && break
            time() < next && continue
            next += every
            dbg(@sprintf("  … still in %s after %.0f s", what, time() - t0))
        end
    end
    try
        return f()
    finally
        done[] = true
        try
            wait(hb)
        catch
        end
    end
end

function print_env_info(outdir, parts, n_rounds, truth, seed, catalog)
    DEBUG[] || return nothing
    dbg("OctofitterDR4Sim v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("parts = ", join(parts, ", "), " | n_rounds = ", n_rounds, " (", 2^n_rounds,
        " samples) | truth = ", truth, " | seed = ", seed, " | G23H noise rows = ", catalog)
    Threads.nthreads() == 1 &&
        dbg("  NOTE: running Pigeons on 1 thread (relaunch disabled) — expect a longer run")
    dbg("project    = ", Base.active_project())
    dbg("SCRIPT_DIR = ", SCRIPT_DIR, " | outdir = ", abspath(outdir))
    dbg("interactive = ", isinteractive(),
        " | Makie backend = ", CairoMakie.Makie.current_backend())
    return nothing
end

# ---- Chain helpers -----------------------------------------------------------
colvec(chain, p::Symbol) = vec(Array(chain[p]))
haspar(chain, p::Symbol) = p in names(chain)
qline(x) = Statistics.quantile(filter(isfinite, x), (0.16, 0.5, 0.84))

"First chain column whose name ends with `suffix`."
function findcol(chain, suffix::AbstractString)
    for n in names(chain)
        endswith(string(n), suffix) && return n
    end
    return nothing
end

"""
Posterior median [16, 84] next to a truth value: the offset in posterior σ and
the truth's percentile within the posterior. The percentile is the honest
number when the posterior is skewed or piles against a boundary (e.g. i near
0°, where the Sine() prior has almost no volume) — a Gaussian σ is not.
"""
function truth_line(lab, x, tru)
    x = filter(isfinite, x)
    q = Statistics.quantile(x, (0.16, 0.5, 0.84))
    s = (q[3] - q[1]) / 2
    z = s > 0 ? (q[2] - tru) / s : NaN
    pct = 100 * Statistics.mean(x .<= tru)
    flag = (pct < 2.5 || pct > 97.5) ? "  ← truth outside the central 95%" : ""
    dbg(@sprintf("    %-24s %12.5g  [%12.5g, %12.5g]   truth %12.5g   (%+.2fσ, truth at %5.1f%%)",
                 lab, q[2], q[1], q[3], tru, z, pct), flag)
    return nothing
end

"Posterior vs injected truth for the single-star parts."
function print_truth_table(chain, truth, label)
    DEBUG[] || return nothing
    b = truth.bodies.b
    dbg("Posterior vs truth — ", label, " (", size(chain, 1) * size(chain, 3), " samples):")
    truth_line("plx [mas]", colvec(chain, :plx), truth.plx)
    truth_line("b mass [M_jup]", colvec(chain, :b_mass) ./ mjup, b.mass / mjup)
    truth_line("b a [AU]", colvec(chain, :b_a), b.a)
    truth_line("b e", colvec(chain, :b_e), b.e)
    truth_line("b i [deg]", rad2deg.(colvec(chain, :b_i)), rad2deg(b.i))
    Pd = sqrt.(colvec(chain, :b_a) .^ 3 ./ (truth.bodies.A.mass .+ colvec(chain, :b_mass))) .* 365.25
    truth_line("b P [d] (derived)", Pd, sqrt(b.a^3 / (truth.bodies.A.mass + b.mass)) * 365.25)
    o = truth.observations.GaiaDR4
    for (lab, key, tv) in (("pmra [mas/yr]", "_pmra", o.pmra), ("pmdec [mas/yr]", "_pmdec", o.pmdec),
                           ("ra offset [mas]", "_ra_offset_mas", o.ra_offset_mas),
                           ("dec offset [mas]", "_dec_offset_mas", o.dec_offset_mas),
                           ("jitter [mas]", "_astrometric_jitter", o.astrometric_jitter))
        c = findcol(chain, key)
        c === nothing || truth_line(lab, colvec(chain, c), tv)
    end
    if abs(b.i) < 0.1
        dbg("    (i ≈ 0: nearly face-on, so ω and Ω are degenerate and not listed. The Sine() prior on i")
        dbg(@sprintf("     has P(i < 10°) = %.1f%%, so the posterior cannot sit at the truth; i and mass are", 100 * (1 - cosd(10))))
        dbg("     correlated, which pulls the mass up with it — a prior-volume effect, not a failed fit)")
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

"Download `url` to `path` once (via a .part file); reuse it afterwards."
function fetch_once(url, path)
    if isfile(path) && filesize(path) > 0
        dbg(@sprintf("  reusing %s (%.1f kB)", path, filesize(path) / 1024))
        return path
    end
    dbg("  downloading ", url)
    tmp = path * ".part"
    Downloads.download(url, tmp)
    mv(tmp, path; force=true)
    dbg(@sprintf("  saved %s (%.1f kB)", path, filesize(path) / 1024))
    return path
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

const CMP1 = "#0072B2"   # PairPlots series 1
const CMP2 = "#E69F00"   # PairPlots series 2
const CTRUTH = "#D55E00"

"Simulated abscissae and the scan geometry for one transit table."
function sim_data_figure(tab, title)
    fig = Figure(size=(1150, 760))
    ax1 = Axis(fig[1, 1], xlabel="epoch [MJD]", ylabel="simulated centroid_pos_al [mas]", title=title)
    errorbars!(ax1, tab.epoch, tab.centroid_pos_al, tab.centroid_pos_error_al; color=(:white, 0.4))
    sc = scatter!(ax1, tab.epoch, tab.centroid_pos_al; color=tab.scan_pos_angle, colormap=:twilight,
                  colorrange=(-180, 180), markersize=7)
    Colorbar(fig[1, 2], sc, label="scan angle ψ [deg]")
    ax2 = Axis(fig[2, 1], xlabel="epoch [MJD]", ylabel="parallax factor (AL)",
               title="GOST parallax factors ($(length(tab.epoch)) transits)")
    scatter!(ax2, tab.epoch, tab.parallax_factor_al; color=CMP2, markersize=6)
    rowsize!(fig.layout, 1, Relative(0.6))
    return fig
end

"The page's figure: 50 posterior tracks of b about A vs the injected truth."
function truth_orbit_figure(sim_model, chain, truth, epochs; n=50, seed=1)
    ts = range(minimum(epochs), minimum(epochs) + 4000, length=300)
    fig = Figure(size=(560, 560))
    ax = Axis(fig[1, 1], xlabel="Δα⋆ [mas]", ylabel="Δδ [mas]", aspect=DataAspect(),
              xreversed=true, title="Posterior orbits of b (grey) vs truth")
    rng = Random.Xoshiro(seed)
    for i in rand(rng, 1:size(chain, 1), n)
        posys = construct_system(sim_model, chain, i)
        traj = orbitsolve(posys, ts)
        lines!(ax, [raoff(traj[k], :b, :A) for k in eachindex(ts)],
                   [decoff(traj[k], :b, :A) for k in eachindex(ts)], color=(:white, 0.12))
    end
    true_sys = construct_system(sim_model, truth)
    tt = orbitsolve(true_sys, ts)
    lines!(ax, [raoff(tt[k], :b, :A) for k in eachindex(ts)],
               [decoff(tt[k], :b, :A) for k in eachindex(ts)], color=CTRUTH, linewidth=3, label="truth")
    scatter!(ax, [0], [0], color=CMP2, markersize=12)
    axislegend(ax)
    return fig
end

function marginals_figure(chain, truth, label)
    b = truth.bodies.b
    fig = Figure(size=(1100, 700))
    specs = (
        (log10.(colvec(chain, :b_mass) ./ mjup), log10(b.mass / mjup), "log₁₀ b mass [M_jup]"),
        (colvec(chain, :b_a), b.a, "b a [AU]"),
        (colvec(chain, :b_e), b.e, "b e"),
        (colvec(chain, :plx), truth.plx, "plx [mas]"),
    )
    for (k, (x, tv, lab)) in enumerate(specs)
        ax = Axis(fig[fldmod1(k, 2)...], xlabel=lab, ylabel="density")
        hist!(ax, x; bins=60, normalization=:pdf, color=(CMP1, 0.85))
        vlines!(ax, [tv]; color=CTRUTH, linewidth=2)
    end
    Label(fig[0, 1:2], "Key marginals — $(label) (red: truth)", fontsize=16)
    return fig
end

"The page's GJ 15 figure: mass-fraction prior vs posterior, and z vs M_tot."
function binary_figure(chain, geom; seed=1)
    rng = Random.Xoshiro(seed)
    mA = rand(rng, Uniform(0.05, 0.9), 200_000)
    mB = rand(rng, Uniform(0.02, 0.6), 200_000)
    fprior = mB ./ (mA .+ mB)
    fpost = colvec(chain, :B_mass) ./ (colvec(chain, :A_mass) .+ colvec(chain, :B_mass))
    Mtot = colvec(chain, :A_mass) .+ colvec(chain, :B_mass)
    ftrue = GJ15.mass_B / (GJ15.mass_A + GJ15.mass_B)
    fig = Figure(size=(1000, 420))
    ax1 = Axis(fig[1, 1], xlabel="mass fraction  mᴮ / (mᴬ + mᴮ)", ylabel="density",
               limits=(0, 1, nothing, nothing))
    hist!(ax1, fprior, bins=60, normalization=:pdf, color=(:gray, 0.45), label="prior")
    hist!(ax1, fpost, bins=40, normalization=:pdf, color=(CMP1, 0.8), label="posterior")
    vlines!(ax1, [ftrue], color=CTRUTH, linewidth=2, label="truth")
    axislegend(ax1, position=:rt)
    ρ = hypot(geom.x_cat, geom.y_cat)
    zs = range(-170, 170, length=200)
    ax2 = Axis(fig[1, 2], xlabel="line-of-sight separation z [AU]", ylabel="total mass [M⊙]",
               limits=(nothing, nothing, 0.2, 1.5))
    scatter!(ax2, colvec(chain, :B_z), Mtot, markersize=3, color=(:white, 0.2))
    lines!(ax2, zs, (GJ15.mass_A + GJ15.mass_B) .* ((ρ^2 .+ zs .^ 2) ./ (ρ^2 + GJ15.z^2)) .^ 1.5,
           color=CTRUTH, linestyle=:dash)
    scatter!(ax2, [GJ15.z], [GJ15.mass_A + GJ15.mass_B], color=CTRUTH, marker=:star5, markersize=18)
    return fig
end

# -----------------------------------------------------------------------------
# Part 1: noise model, transit template, model, simulation
# -----------------------------------------------------------------------------
"g23h_scan_uncertainty from the page's row (or the full catalog), printed."
function noise_model(gaia_id, row; catalog_mode="page", label="")
    σ = catalog_mode == "full" ? g23h_scan_uncertainty(; gaia_id) :
                                 g23h_scan_uncertainty(; gaia_id, catalog=row)
    dbg("  noise model ", label, ":")
    for k in propertynames(σ)
        v = getproperty(σ, k)
        v isa Real && dbg(@sprintf("    %-18s %.5f", string(k), v))
    end
    return σ
end

"Gaia DR3 solution (network, cached); `nothing` if the query fails."
function dr3_solution(gaia_id; outdir=SCRIPT_DIR)
    return soft_stage("Gaia DR3 solution for $gaia_id") do
        s = with_heartbeat("Gaia DR3 query") do
            cd(() -> gaia_dr3_solution(; gaia_id), outdir)
        end
        g(k) = hasproperty(s, k) ? getproperty(s, k) : missing
        dbg("  DR3: ra = ", g(:ra), ", dec = ", g(:dec), ", plx = ", g(:parallax),
            ", pmra = ", g(:pmra), ", pmdec = ", g(:pmdec), ", G = ", g(:phot_g_mean_mag),
            ", RV = ", g(:radial_velocity))
        s
    end
end

"GOST transit template; cached files GOST-<ra>-<dec>-dr4.csv are read from `outdir`."
function transit_template(ra, dec, σ_al; outdir=SCRIPT_DIR, label="")
    t = with_heartbeat("GOST transit template $label") do
        cd(() -> gaia_dr4_transit_template(; ra, dec, σ_al, baseline=:dr4), outdir)
    end
    dbg(@sprintf("  %s: %d transits, MJD %.1f–%.1f (%.2f yr), σ_al = %.4f mas",
                 label, length(t.epoch), minimum(t.epoch), maximum(t.epoch),
                 (maximum(t.epoch) - minimum(t.epoch)) / 365.25, Statistics.median(t.centroid_pos_error_al)))
    return t
end

function single_bodies(orbit_ref_epoch)
    mb, ab, eb = PRIORS.m_b, PRIORS.a_b, PRIORS.e_b
    A_vars = @eval @variables begin
        mass = 1.0             # [M⊙]
        flux = 1.0             # sets the flux scale the photocentre is weighted by
    end
    A = Body(name="A", variables=A_vars)
    b_vars = @eval @variables begin
        mass ~ $mb             # [M⊙]
        flux = 0.0             # dark companion
        a ~ $ab                # [AU]
        e ~ $eb
        ω ~ Uniform(0, 2pi)
        i ~ Sine()
        Ω ~ Uniform(0, 2pi)
        θ ~ Uniform(0, 2pi)
        epoch = $orbit_ref_epoch
    end
    return A, Body(name="b", about=A, variables=b_vars)
end

function single_obs(tab)
    jit, off, pm, re = PRIORS.jit, PRIORS.off, PRIORS.pm, REF_EPOCH_MJD
    vars = @eval @variables begin
        astrometric_jitter ~ $jit      # mas
        ra_offset_mas  ~ $off
        dec_offset_mas ~ $off
        pmra ~ $pm                     # mas/yr
        pmdec ~ $pm
        ref_epoch = $re
    end
    return GaiaDR4AstromObs(tab; target=Photocentre, ref=Barycentre, name="GaiaDR4", variables=vars)
end

"Template system for part 1 (fresh bodies + observation)."
function single_system(tab)
    orbit_ref_epoch = Statistics.mean(tab.epoch)
    A, b = single_bodies(orbit_ref_epoch)
    obs = single_obs(tab)
    pp = PRIORS.plx
    vars = @eval @variables begin
        plx ~ $pp                      # keep physically plausible
    end
    return System(name="target_1", bodies=[A, b], observations=[obs], variables=vars), orbit_ref_epoch
end

"Pigeons with the page's defaults (+ n_chains from OCTOSIM_CHAINS); returns (chain, logZ, Λ)."
function fit_pigeons(model, tag, n_rounds)
    nc = strip(get(ENV, "OCTOSIM_CHAINS", ""))
    kw = isempty(nc) ? (;) : (; n_chains = parse(Int, nc))
    dbg("  octofit_pigeons: n_rounds = ", n_rounds, isempty(nc) ? " (default n_chains)" : ", n_chains = $nc")
    c, pt = octofit_pigeons(model; n_rounds=n_rounds, kw...)
    z = soft_stage("[$tag] Log-evidence ratio (stepping stone)") do
        zz, λ = Pigeons.stepping_stone(pt), Pigeons.global_barrier(pt)
        dbg(@sprintf("  [%s] log(Z₁/Z₀) ≈ %.3f, Λ = %.2f", tag, zz, λ))
        (zz, λ)
    end
    return c, something(z, (NaN, NaN))...
end

# -----------------------------------------------------------------------------
# Part 2: GJ 15
# -----------------------------------------------------------------------------
"Catalogue state vector of B relative to A at the DR4 reference epoch (page)."
function gj15_geometry(dr3_A, dr3_B)
    Δt_yr = (REF_EPOCH_MJD - DR3_EPOCH_MJD) / 365.25
    plx0 = (dr3_A.parallax + dr3_B.parallax) / 2
    Δpmra, Δpmdec = dr3_B.pmra - dr3_A.pmra, dr3_B.pmdec - dr3_A.pmdec
    sep_ra_mas = (dr3_B.ra - dr3_A.ra) * cosd(dr3_A.dec) * 3.6e6 + Δpmra * Δt_yr
    sep_dec_mas = (dr3_B.dec - dr3_A.dec) * 3.6e6 + Δpmdec * Δt_yr
    x_cat, y_cat = sep_ra_mas / plx0, sep_dec_mas / plx0
    vx_cat, vy_cat = Δpmra / plx0, Δpmdec / plx0
    vz_cat = (dr3_B.radial_velocity - dr3_A.radial_velocity) / 4.740470446
    g = (; plx0, Δpmra, Δpmdec, sep_ra_mas, sep_dec_mas, x_cat, y_cat, vx_cat, vy_cat, vz_cat)
    dbg(@sprintf("  GJ 15: separation %.2f″, plx0 = %.3f mas; x, y = %.2f, %.2f AU; vx, vy, vz = %.4f, %.4f, %.4f AU/yr",
                 hypot(sep_ra_mas, sep_dec_mas) / 1000, plx0, x_cat, y_cat, vx_cat, vy_cat, vz_cat))
    return g
end

function gj15_system(tA, tB, geom, dr3_A)
    re = REF_EPOCH_MJD
    xc, yc, vxc, vyc, vzc = geom.x_cat, geom.y_cat, geom.vx_cat, geom.vy_cat, geom.vz_cat
    A_vars = @eval @variables begin
        mass ~ Uniform(0.05, 0.9)          # M⊙
    end
    A = Body(name="A", variables=A_vars)
    B_vars = @eval @variables begin
        mass ~ Uniform(0.02, 0.6)          # M⊙
        x  ~ Normal($xc, 1.0)              # AU, east
        y  ~ Normal($yc, 1.0)              # AU, north
        z  ~ Uniform(-250, 250)            # AU, away from us — nothing measures this
        vx ~ Normal($vxc, 0.02)            # AU / julian year
        vy ~ Normal($vyc, 0.02)
        vz ~ Normal($vzc, 0.02)
        epoch = $re
    end
    B = Body(name="B", about=A, variables=B_vars)
    frame_variables() = @eval @variables begin
        astrometric_jitter = 0.0
        ra_offset_mas  = system.bary_ra_offset_mas
        dec_offset_mas = system.bary_dec_offset_mas
        pmra  = system.bary_pmra
        pmdec = system.bary_pmdec
        ref_epoch = $re
    end
    obs_A = GaiaDR4AstromObs(tA; target=A, ref=Barycentre, name="DR4_A", variables=frame_variables())
    obs_B = GaiaDR4AstromObs(tB; target=B, ref=Barycentre, name="DR4_B", variables=frame_variables())
    pmA, pmdA = dr3_A.pmra, dr3_A.pmdec
    sys_vars = @eval @variables begin
        plx ~ Uniform(200, 350)                    # mas — one parallax for both sources
        ra_offset_A_mas  ~ Normal(0, 500)          # star A's own position…
        dec_offset_A_mas ~ Normal(0, 500)
        pmra_A  ~ Normal($pmA, 50)                 # …and star A's own proper motion
        pmdec_A ~ Normal($pmdA, 50)
        Δ_A = anchor_offsets(system_interim, :A, $re)
        bary_ra_offset_mas  = ra_offset_A_mas  - Δ_A.ra_cosdec
        bary_dec_offset_mas = dec_offset_A_mas - Δ_A.dec
        bary_pmra  = pmra_A  - Δ_A.pmra
        bary_pmdec = pmdec_A - Δ_A.pmdec
    end
    return System(name="GJ15", bodies=[A, B], observations=[obs_A, obs_B], variables=sys_vars)
end

"The page's injected GJ 15 truth (frame spelled out: generate_from_params reads it verbatim)."
function gj15_truth(geom, dr3_A)
    f_B = GJ15.mass_B / (GJ15.mass_A + GJ15.mass_B)
    shared = (; astrometric_jitter = 0.0,
               ra_offset_mas = f_B * geom.sep_ra_mas, dec_offset_mas = f_B * geom.sep_dec_mas,
               pmra = dr3_A.pmra + f_B * geom.Δpmra, pmdec = dr3_A.pmdec + f_B * geom.Δpmdec,
               ref_epoch = REF_EPOCH_MJD)
    return (plx = geom.plx0, ra_offset_A_mas = 0.0, dec_offset_A_mas = 0.0,
            pmra_A = dr3_A.pmra, pmdec_A = dr3_A.pmdec,
            bodies = (A = (; mass = GJ15.mass_A),
                      B = (; mass = GJ15.mass_B, x = geom.x_cat, y = geom.y_cat, z = GJ15.z,
                             vx = geom.vx_cat, vy = geom.vy_cat, vz = geom.vz_cat, epoch = REF_EPOCH_MJD)),
            observations = (DR4_A = shared, DR4_B = shared))
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_dr4sim(; parts=(:single,), n_rounds=9, truth="page", seed=1, nuts=1000,
                 catalog="page", outdir=SCRIPT_DIR, prefix="dr4sim_", dark=true,
                 corner_dark=false, show_plots=true)
"""
function run_dr4sim(; parts=(:single,),
                      n_rounds::Integer=envint("OCTOSIM_ROUNDS", "9"),
                      truth::AbstractString=lowercase(get(ENV, "OCTOSIM_TRUTH", "page")),
                      seed::Integer=envint("OCTOSIM_SEED", "1"),
                      nuts::Integer=envint("OCTOSIM_NUTS", "1000"),
                      catalog::AbstractString=lowercase(get(ENV, "OCTOSIM_CATALOG", "page")),
                      outdir::AbstractString=SCRIPT_DIR,
                      prefix::AbstractString="dr4sim_",
                      dark::Bool=true,
                      corner_dark::Bool=false,
                      show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    print_env_info(outdir, parts, n_rounds, truth, seed, catalog)
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
    savechain!(chain, model, tag) = soft_stage("[$tag] Save chain -> FITS (+ reload check)") do
        p = joinpath(outdir, prefix * "$(tag)_chain.fits")
        Octofitter.savechain(p, chain)
        push!(outputs, p)
        c2 = Octofitter.loadchain(p; model)
        dbg("  reloaded chain size = ", size(c2), size(c2) == size(chain) ? "  (matches)" : "  (DIFFERS)")
    end

    chains = Dict{Symbol,Any}(); models = Dict{Symbol,Any}()
    logZ = Dict{Symbol,Float64}(); Λs = Dict{Symbol,Float64}()
    sim = nothing

    if :single in parts || :formal in parts
        # ---- Part 1 ------------------------------------------------------------
        σ = stage("[single] G23H noise model (g23h_scan_uncertainty)") do
            noise_model(GAIA_ID, CATALOG_ROW; catalog_mode=catalog, label="Gaia DR3 $GAIA_ID")
        end
        stage("[single] Cached GOST forecast from the Octofitter repo") do
            fetch_once(OCTO_RAW * GOST_FILE, joinpath(outdir, GOST_FILE))
        end
        dr3 = dr3_solution(GAIA_ID; outdir)
        # The cached file is found by name, so use the exact coordinates it was made with.
        if dr3 !== nothing && !(dr3.ra == RA_GOST && dr3.dec == DEC_GOST)
            dbg(@sprintf("  note: DR3 ra/dec differ from the cached forecast's (%.12g, %.12g) by %.3g/%.3g mas; using the cached coordinates",
                         RA_GOST, DEC_GOST, (dr3.ra - RA_GOST) * 3.6e6, (dr3.dec - DEC_GOST) * 3.6e6))
        end
        transits = stage("[single] Transit template (σ_al = σ_transit_true)") do
            transit_template(RA_GOST, DEC_GOST, σ.σ_transit_true; outdir, label="part-1 star")
        end
        sys, orbit_ref_epoch = stage("[single] Template system") do
            single_system(transits)
        end
        DEBUG[] && display(sys)
        model = stage("[single] Compile template LogDensityModel (verbosity = 4)") do
            Octofitter.LogDensityModel(sys; verbosity=4)
        end

        Random.seed!(seed)
        params = stage("[single] Simulation truth (mode = $truth)") do
            p = truth == "prior" ? Octofitter.drawfrompriors(model.system) : page_truth(orbit_ref_epoch)
            DEBUG[] && display(p)
            p
        end
        sim_system = stage("[single] generate_from_params(add_noise = true), seed $seed") do
            Random.seed!(seed)
            Octofitter.generate_from_params(model.system, params; add_noise=true)
        end
        sim_tab = sim_system.observations[1].table
        sim = (; transits, sim_tab, params, σ)
        soft_stage("[single] Simulated-data figure") do
            savefig!(sim_data_figure(sim_tab, "Simulated DR4 abscissae ($(length(sim_tab.epoch)) transits, truth = $truth)"),
                     "single_data.png")
        end

        for kind in (:single, :formal)
            kind in parts || continue
            tag = string(kind)
            sim_model = stage("[$tag] Build and compile model — $(PART_LABELS[kind])") do
                if kind === :single
                    Octofitter.LogDensityModel(sim_system)
                else
                    # Fresh bodies/observation; the simulated abscissae with formal errors.
                    tab2 = Table(sim_tab; centroid_pos_error_al = fill(σ.σ_transit_formal, length(sim_tab.epoch)))
                    dbg(@sprintf("  centroid_pos_error_al: %.4f (true) → %.4f mas (formal); expect jitter ≈ σ_calib = %.4f mas",
                                 σ.σ_transit_true, σ.σ_transit_formal, σ.σ_calib))
                    s2, _ = single_system(tab2)
                    Octofitter.LogDensityModel(s2)
                end
            end
            models[kind] = sim_model
            init_chain = stage("[$tag] initialize!") do
                Octofitter.initialize!(sim_model)
            end
            soft_stage("[$tag] octoplot of the starting point") do
                savefig!(inout(() -> octoplot(sim_model, init_chain)), "$(tag)_init_octoplot.png")
            end
            chain = stage("[$tag] Pigeons (n_rounds = $n_rounds)") do
                c, z, λ = fit_pigeons(sim_model, tag, n_rounds)
                logZ[kind] = z; Λs[kind] = λ
                c
            end
            chains[kind] = chain
            DEBUG[] && display(chain)
            soft_stage("[$tag] Posterior vs truth") do
                print_truth_table(chain, params, PART_LABELS[kind])
                if kind === :formal
                    c = findcol(chain, "_astrometric_jitter")
                    c === nothing || truth_line("jitter vs σ_calib [mas]", colvec(chain, c), σ.σ_calib)
                end
            end
            soft_stage("[$tag] octoplot") do
                savefig!(inout(() -> octoplot(sim_model, chain)), "$(tag)_octoplot.png")
            end
            soft_stage("[$tag] octoplot of one draw") do
                savefig!(inout(() -> octoplot(Octofitter.PosteriorSeries(sim_model, chain; ii=[1]))),
                         "$(tag)_octoplot_draw.png")
            end
            soft_stage("[$tag] gaiastarplot (draw 1)") do
                savefig!(inout(() -> Octofitter.gaiastarplot(sim_model, chain, 1)), "$(tag)_gaiastarplot.png")
            end
            soft_stage("[$tag] Posterior orbits vs truth (page)") do
                savefig!(truth_orbit_figure(sim_model, chain, params, transits.epoch; seed), "$(tag)_truth_orbits.png")
            end
            soft_stage("[$tag] Corner plot (small=true)") do
                savefig!(inout(() -> corner(sim_model, chain; dark=corner_dark)), "$(tag)_corner.png")
            end
            soft_stage("[$tag] Key-marginal histograms with truth") do
                savefig!(marginals_figure(chain, params, PART_LABELS[kind]), "$(tag)_marginals.png")
            end
            savechain!(chain, sim_model, tag)
        end
    end

    if :binary in parts
        # ---- Part 2: GJ 15 -----------------------------------------------------
        dr3_A = dr3_solution(GJ15.id_A; outdir)
        dr3_B = dr3_solution(GJ15.id_B; outdir)
        if dr3_A === nothing || dr3_B === nothing
            dbg("⚠ [binary] skipped: the GJ 15 DR3 solutions could not be fetched (network?)")
        else
            soft_stage("[binary] GJ 15 two-source fit") do
                σA = noise_model(GJ15.id_A, GJ15.cat_A; catalog_mode=catalog, label="GJ 15 A")
                σB = noise_model(GJ15.id_B, GJ15.cat_B; catalog_mode=catalog, label="GJ 15 B")
                tA = stage("[binary] GOST transits, A (live query, cached next to the script)") do
                    transit_template(dr3_A.ra, dr3_A.dec, σA.σ_transit_true; outdir, label="GJ 15 A")
                end
                tB = stage("[binary] GOST transits, B") do
                    transit_template(dr3_B.ra, dr3_B.dec, σB.σ_transit_true; outdir, label="GJ 15 B")
                end
                geom = gj15_geometry(dr3_A, dr3_B)
                sysb = stage("[binary] Template system (frame anchored on A)") do
                    gj15_system(tA, tB, geom, dr3_A)
                end
                DEBUG[] && display(sysb)
                mb = stage("[binary] Compile template model") do
                    Octofitter.LogDensityModel(sysb)
                end
                tb = gj15_truth(geom, dr3_A)
                simb = stage("[binary] generate_from_params (Random.seed!(4), as on the page)") do
                    Random.seed!(4)
                    Octofitter.generate_from_params(mb.system, tb; add_noise=true)
                end
                sim_model = Octofitter.LogDensityModel(simb)
                models[:binary] = sim_model
                stage("[binary] initialize! (catalogue values for the well-known quantities)") do
                    Octofitter.initialize!(Random.Xoshiro(1), sim_model, (;
                        plx = geom.plx0, ra_offset_A_mas = 0.0, dec_offset_A_mas = 0.0,
                        pmra_A = dr3_A.pmra, pmdec_A = dr3_A.pmdec,
                        bodies = (B = (; x = geom.x_cat, y = geom.y_cat, vx = geom.vx_cat, vy = geom.vy_cat),),
                    ))
                end
                chain = stage("[binary] NUTS (octofit, adaptation = iterations = $nuts)") do
                    octofit(Random.Xoshiro(1), sim_model; adaptation=nuts, iterations=nuts)
                end
                chains[:binary] = chain
                DEBUG[] && display(chain)
                soft_stage("[binary] What came out (page)") do
                    fB = colvec(chain, :B_mass) ./ (colvec(chain, :A_mass) .+ colvec(chain, :B_mass))
                    dbg("Posterior vs truth — GJ 15 (", length(fB), " samples):")
                    truth_line("mass fraction f_B", fB, GJ15.mass_B / (GJ15.mass_A + GJ15.mass_B))
                    truth_line("total mass [M⊙]", colvec(chain, :A_mass) .+ colvec(chain, :B_mass),
                               GJ15.mass_A + GJ15.mass_B)
                    truth_line("parallax [mas]", colvec(chain, :plx), geom.plx0)
                    truth_line("z [AU]", colvec(chain, :B_z), GJ15.z)
                    dbg("    (M_tot and z are degenerate along M ∝ r³ — see the figure)")
                end
                soft_stage("[binary] Mass fraction and z–M_tot figure (page)") do
                    savefig!(binary_figure(chain, geom; seed), "binary_mass.png")
                end
                soft_stage("[binary] octoplot (one panel per source)") do
                    savefig!(inout(() -> octoplot(sim_model, chain)), "binary_octoplot.png")
                end
                soft_stage("[binary] Corner plot (small=true)") do
                    savefig!(inout(() -> corner(sim_model, chain; dark=corner_dark)), "binary_corner.png")
                end
                savechain!(chain, sim_model, "binary")
            end
        end
    end

    if !isempty(logZ)
        lines = String["Log evidence (stepping stone), n_rounds = $(n_rounds):"]
        for k in (:single, :formal)
            haskey(logZ, k) || continue
            push!(lines, @sprintf("    %-44s log(Z₁/Z₀) = %10.3f   Λ = %6.2f", PART_LABELS[k], logZ[k], Λs[k]))
        end
        foreach(l -> dbg(l), lines)
        soft_stage("Write evidence table") do
            p = joinpath(outdir, prefix * "evidence.txt")
            write(p, join(lines, "\n") * "\n")
            push!(outputs, p)
        end
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f s", time() - T0[]))
    return (; sim, models, chains, logZ, Λ=Λs, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default when the REPL has 1 thread)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString="dr4sim_")
    T0[] = time()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOSIM_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOSIM_CHILD=1]")
    dbg("  Child output follows; its plots are shown here when it finishes.")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOSIM_RELAUNCH\"]=\"false\".")
    t = time()
    ok = try
        run(pipeline(cmd; stdout=stdout, stderr=stderr))
        true
    catch err
        # Never showerror a ProcessFailedException: it prints setenv(...) with
        # the whole environment.
        msg = err isa ProcessFailedException ?
              "exit code " * join((string(p.exitcode) for p in err.procs), ", ") :
              err isa InterruptException ? "interrupted (Ctrl+C)" :
              first(sprint(showerror, err), 300)
        dbg("✖ child process failed: ", msg, " — see the ✖ stage line in its output above")
        false
    end
    dbg(@sprintf("Child process finished in %.1f min (%s)", (time() - t) / 60,
                 ok ? "success" : "FAILED"))
    chains = Dict{Symbol,Any}()
    for kind in PART_ORDER
        p = joinpath(outdir, prefix * "$(kind)_chain.fits")
        (isfile(p) && mtime(p) >= t) || continue
        c = soft_stage("Load $(kind) chain written by child") do
            Octofitter.loadchain(p)
        end
        c === nothing || (chains[kind] = c)
    end
    ev = joinpath(outdir, prefix * "evidence.txt")
    if isfile(ev) && mtime(ev) >= t
        foreach(l -> dbg(l), eachline(ev))
    end
    new_files = [joinpath(outdir, f) for f in readdir(outdir)
                 if startswith(f, prefix) && mtime(joinpath(outdir, f)) >= t]
    list_outputs(new_files)
    pngs = filter(p -> endswith(lowercase(p), ".png"), new_files)
    if isempty(pngs)
        dbg("No new plots were written by the child",
            ok ? "." : " (it failed before the plotting stages — see the ✖ line above).")
    elseif envbool("OCTOSIM_SHOW_PLOTS", "true")
        soft_stage("Show $(length(pngs)) saved plot(s) in the plot pane") do
            show_saved_pngs(pngs)
        end
    end
    return (; relaunched=true, success=ok, chains, outputs=new_files)
end

"Plot order for display: data, then single, formal, binary."
function png_rank(p)
    f = basename(p)
    m = startswith(f, "dr4sim_single") ? 10 : startswith(f, "dr4sim_formal") ? 30 :
        startswith(f, "dr4sim_binary") ? 50 : 70
    for (k, key) in enumerate(("_data", "init_octoplot", "octoplot_draw", "octoplot", "gaiastarplot",
                               "truth_orbits", "corner", "marginals", "mass"))
        occursin(key, f) && return m + k
    end
    return m + 9
end

"""
Load saved PNGs and display each in the plot pane at its own aspect ratio
(the relaunched child can't reach the parent's plot pane).
"""
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

"Parse OCTOSIM_PARTS (comma-separated) into a tuple of Symbols in canonical order."
function parts_from_env()
    s = get(ENV, "OCTOSIM_PARTS", "single")
    req = Set(Symbol(strip(lowercase(x))) for x in split(s, ',') if !isempty(strip(x)))
    issubset(req, Set(PART_ORDER)) ||
        error("OCTOSIM_PARTS must list single, formal and/or binary, got \"$s\"")
    return Tuple(k for k in PART_ORDER if k in req)
end

println(@sprintf("[DR4S] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterDR4Sim

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterDR4Sim.jl`    -> runs
#   - VSCode "Execute active File in REPL"          -> runs; with 1 thread it
#     relaunches in `julia --threads=auto` (OCTOSIM_RELAUNCH=false to disable,
#     OCTOSIM_THREADS to set the count)
#   - ENV["OCTOSIM_AUTORUN"] = "false"; include()   -> loads module only
#   - OCTOSIM_PARTS, _ROUNDS, _TRUTH, _SEED, _NUTS, _CATALOG select what runs
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOSIM_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOSIM_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOSIM_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global dr4sim_result = OctofitterDR4Sim.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOSIM_THREADS", "auto"))
            println("[DR4S] Child run finished. `dr4sim_result` has fields: relaunched, success, ",
                    "chains (loaded from FITS), outputs")
        else
            global dr4sim_result = OctofitterDR4Sim.run_dr4sim(
                parts=OctofitterDR4Sim.parts_from_env(),
                show_plots=!is_child)
            is_child || println("[DR4S] Result stored in `dr4sim_result` (fields: sim, models, chains, ",
                "logZ, Λ, outputs)")
        end
    end
end

nothing
