#=
================================================================================
 OctofitterHipparcos.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Hipparcos IAD":
   https://sefffal.github.io/Octofitter.jl/dev/hipparcos/

 Target: HIP 21547 = c Eri = 51 Eri, host of the directly imaged planet
 51 Eri b. Data: the Hipparcos intermediate astrometric data (IAD, van Leeuwen
 & Michalik 2021 Java-tool release) via `HipparcosIADObs`, plus 14 epochs of
 relative astrometry (GPI/SPHERE, collated on whereistheplanet.com) as on the
 page.

 Models (all sampled with Pigeons, which also gives the log evidence):
   :straight   "Nielsen test" (Nielsen et al. 2020): star only, no orbit; the
               fit must recover the published Hipparcos five-parameter
               solution (plx, Δα*, Δδ, μα*, μδ). n_rounds = 7 by default.
   :mass       51 Eri b with a free mass: Hipparcos IAD (star blended with b)
               + VLT/SPHERE relative astrometry; θ/epoch orbit parametrisation;
               starting point plx = 34 and zero IAD frame offsets; Pigeons with
               SliceSampler explorer (page). n_rounds = 8 by default.
 Plus: IAD data figure (along-scan residuals, scan angles, rejected scans)
 and CSV snapshot; relative-astrometry sky plot and CSV; Nielsen-test
 comparison figure (posterior vs catalog Normal) and table vs the docs'
 values with z-scores; hipparcosplot for each model; octoplot; light-theme
 corner plots; key-marginal histograms for the mass model; b mass in M_jup
 (16/50/84 %, as on the page); Ω folded into 180°; optional prior-only
 log Z0; evidence table.

 recalibrate: HipparcosIADObs defaults to recalibrate=false (published data
 as-is). OCTOHIP_RECAL=true applies the Brandt, Michalik & Brandt shift
 (+0.140 mas, 2.25 mas extra dispersion), which is what G23HObs's :iad_hip
 channel uses.

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `hip_result`)
   Shell:    julia --threads=auto OctofitterHipparcos.jl
   Library:  ENV["OCTOHIP_AUTORUN"] = "false"; include("OctofitterHipparcos.jl")
             res = OctofitterHipparcos.run_hipparcos(models=(:straight,))

 Threads: every fit uses Pigeons. In a 1-thread REPL the run is relaunched in
 a child `julia --threads=auto` (output streamed here; chains loaded back and
 saved plots shown in the plot pane afterwards). OCTOHIP_RELAUNCH=false
 disables this; OCTOHIP_THREADS sets the count.

 Convenience ENV settings (also passed to a relaunched child):
   OCTOHIP_MODELS          = "straight,mass" (default)
   OCTOHIP_ROUNDS_STRAIGHT = "7"  (page)     OCTOHIP_ROUNDS_MASS = "8" (page)
   OCTOHIP_EXPLORER        = "slice" (default, page) or "auto" (Octofitter's
                             default explorer) for the mass model
   OCTOHIP_RECAL           = "false" (default) — recalibrate the IAD
   OCTOHIP_PRIOR_Z0        = "false" (default) — prior-only evidence runs
   OCTOHIP_SHOW_PLOTS      = "true" (default) — show the child's PNGs here
   OCTOHIP_CHAINS          = "" (default: Octofitter's choice) — Pigeons
                             n_chains; more chains help the isolated mass mode
   OCTOHIP_HEARTBEAT       = "30" (default) — seconds between "still waiting"
                             lines during the IAD load
   OCTOHIP_ACCEPT_DOWNLOADS = "true" (default) — sets DATADEPS_ALWAYS_ACCEPT

 Data: the Hipparcos_IAD DataDep (~332 MB zip, 826 MB unpacked) is already
 installed on this Mac by the PMA module (~/.julia/scratchspaces/<uuid>/
 datadeps/Hipparcos_IAD); otherwise it downloads once. Empty DataDep folders
 left by interrupted downloads are reported with the path to delete. The
 relative astrometry (inline, from the page) and a snapshot of the HIP 21547
 IAD rows are written as CSVs next to this file.

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOHIP_ENV_MODE=local uses ./octofitter_v9_env). Needs Octofitter
 v9, CairoMakie, PairPlots, Distributions, Pigeons, CSV.

 Outputs (prefix "hip_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-25  Initial version on the OctofitterPMA v1.0.3 scaffolding
                   ([HIP +t] debug stages, soft optional stages, env bootstrap
                   with v9 guards, @__DIR__ outputs, dark theme with light
                   corner plots, threaded relaunch with exit-code-only failure
                   report and PNGs shown in the parent's plot pane, heartbeat
                   and stale-DataDep checks, initialize! before every Pigeons
                   run, @eval-built @variables). Adds: HipparcosIADObs Nielsen
                   test with catalog comparison figure/table; 51 Eri b mass
                   model with SliceSampler and page starting point; IAD and
                   relative-astrometry data figures and CSVs; hipparcosplot;
                   recalibrate switch.
 v1.0.1 2026-09-25 First run (2.8 min, 4 threads): Nielsen test within 0.11σ
                   of the catalog on all five parameters. Fixes: star-only
                   corner plot used small=true, which drops every column
                   (no orbital elements) — now small=false for that model;
                   mass-mode report (light vs prior-edge ~100 M_jup mode with
                   its a, e, i and IAD Δδ); OCTOHIP_CHAINS sets Pigeons
                   n_chains; the octofit_pigeons keywords are logged.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "CairoMakie", "PairPlots", "Distributions", "Pigeons", "CSV"),
    min_version = Dict("Octofitter" => v"9"),
    mode = lowercase(get(ENV, "OCTOHIP_ENV_MODE", "temp"))

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
            println("[HIP] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[HIP] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[HIP] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[HIP] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[HIP] Existing v9 environment found at ", Base.active_project())
        else
            Pkg.add(spec.(collect(required)))
        end
        println("[HIP] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[HIP] Loading Octofitter, Pigeons, CairoMakie, PairPlots, Distributions, CSV ",
        "(first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterHipparcos

const LOAD_T0 = time()

using Octofitter
using CairoMakie
using PairPlots
using Distributions
using Pigeons
import CSV
import Random
import Statistics
using Printf

export run_hipparcos, build_system, set_plot_theme!

const VERSION_STRING = "1.0.1"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterHipparcos needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

# Target, data and priors (as on the tutorial page)
const TARGET = "HIP 21547 (c Eri = 51 Eri)"
const HIP_ID = 21547

const ASTROM = (
    epoch = [57009.1, 57052.1, 57053.1, 57054.3, 57266.4, 57332.2, 57374.2, 57376.2,
             57415.0, 57649.4, 57652.4, 57739.1, 58068.3, 58442.2],
    sep   = [454.24, 451.81, 456.8, 461.5, 455.1, 452.88, 455.91, 455.01, 454.46,
             454.81, 451.43, 449.39, 447.54, 434.22],
    σ_sep = [1.88, 2.06, 2.57, 23.9, 2.23, 5.41, 6.23, 3.03, 6.03, 2.02, 2.67, 2.15,
             3.02, 2.01],
    pa    = [2.98835, 2.96723, 2.97038, 2.97404, 2.91994, 2.89934, 2.89131, 2.89184,
             2.8962, 2.82394, 2.82272, 2.79357, 2.70927, 2.61171],
    σ_pa  = [0.00401426, 0.00453786, 0.00523599, 0.0523599, 0.00453786, 0.00994838,
             0.00994838, 0.00750492, 0.00890118, 0.00453786, 0.00541052, 0.00471239,
             0.00680678, 0.00401426],
)

const PRIORS = (
    plx_straight = Uniform(10, 100),                      # [mas]
    plx_mass     = Uniform(20, 40),                       # [mas]
    M_A          = truncated(Normal(1.75, 0.05), lower=0.03),   # [M⊙]
    m_b          = LogUniform(0.1mjup, 100mjup),          # [M⊙]
    a_b          = truncated(Normal(10, 1), lower=0.1),   # [AU]
    e_b          = Uniform(0, 0.99),
    θ_epoch      = 58442.2,                               # [MJD]
)

# Docs' 500-sample Nielsen-test result and the catalog values it compares to
const DOCS_NIELSEN = Dict(
    "plx"       => (post=(33.99, 0.36),  cat=(33.98, 0.34)),
    "iad_Δra"   => (post=(-0.00, 0.28),  cat=(0.0, 0.29)),
    "iad_Δdec"  => (post=(0.01, 0.19),   cat=(0.0, 0.19)),
    "iad_pmra"  => (post=(44.21, 0.33),  cat=(44.22, 0.34)),
    "iad_pmdec" => (post=(-64.39, 0.26), cat=(-64.39, 0.27)),
)

const MODEL_LABELS = Dict(:straight => "straight line (Nielsen test)",
                          :mass     => "51 Eri b mass (IAD + rel. astrometry)")
const MODEL_ORDER = (:straight, :mass)
const MODEL_ROUNDS_ENV = Dict(:straight => ("OCTOHIP_ROUNDS_STRAIGHT", "7"),
                              :mass     => ("OCTOHIP_ROUNDS_MASS", "8"))

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterHipparcos.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[HIP +%7.1fs] ", time() - T0[]), msg...)
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

"Let DataDeps download without an interactive y/n prompt (unless disabled)."
function accept_downloads!()
    envbool("OCTOHIP_ACCEPT_DOWNLOADS", "true") && (ENV["DATADEPS_ALWAYS_ACCEPT"] = "true")
    return nothing
end

# ---- DataDeps folders, heartbeat ---------------------------------------------
"""
Existing DataDeps storage folders: ~/.julia/datadeps and
~/.julia/scratchspaces/<uuid>/datadeps (where current DataDeps writes).
"""
function datadeps_dirs()
    depot = first(DEPOT_PATH)
    ds = String[joinpath(depot, "datadeps")]
    ss = joinpath(depot, "scratchspaces")
    if isdir(ss)
        for u in readdir(ss)
            push!(ds, joinpath(ss, u, "datadeps"))
        end
    end
    return filter(isdir, ds)
end

"Total size in bytes of all files under `d` (0 if missing; errors ignored)."
function dir_bytes(d)
    isdir(d) || return 0
    n = 0
    try
        for (root, _, files) in walkdir(d)
            for f in files
                p = joinpath(root, f)
                n += isfile(p) ? filesize(p) : 0
            end
        end
    catch
    end
    return n
end

mb(n) = n / 1024^2

"Top-level DataDeps entries with sizes (skips walking the 236k IAD files)."
function list_datadeps()
    ds = datadeps_dirs()
    isempty(ds) && (dbg("  no DataDeps folders yet"); return nothing)
    for d in ds
        dbg("  DataDeps folder: ", d)
        for e in sort(readdir(d))
            p = joinpath(d, e)
            dbg("    ", e, isdir(p) ? "/" : "")
        end
    end
    return nothing
end

"""
Report DataDep folders left empty by an interrupted download (DataDeps treats
an existing folder as already downloaded). Only the Hipparcos_IAD folder is
checked in depth (for its unpacked ResRec_JavaTool_2014 directory); the G23H
folders aren't used here and only get a note.
"""
function check_stale_datadeps()
    stale = String[]
    for d in datadeps_dirs(), e in readdir(d)
        p = joinpath(d, e)
        isdir(p) || continue
        bad = if e == "Hipparcos_IAD"
            !isdir(joinpath(p, "ResRec_JavaTool_2014"))
        else
            isempty(readdir(p))
        end
        bad || continue
        if startswith(e, "G23H_")
            dbg("  note: empty DataDep folder ", p, " (not used by this module)")
        else
            push!(stale, p)
            dbg("  ⚠ incomplete DataDep folder (interrupted download?): ", p)
            dbg("    delete it so DataDeps downloads again:  rm(\"", p, "\"; recursive=true)")
        end
    end
    isempty(stale) && dbg("  no incomplete DataDep folders that this module uses")
    return stale
end

"""
    with_heartbeat(f, what; every, watch, outdir)

Run `f()` while a background task prints a line every `every` seconds with the
elapsed time and how much the watched folders have grown.
"""
function with_heartbeat(f, what::AbstractString;
                        every::Real=parse(Float64, get(ENV, "OCTOHIP_HEARTBEAT", "30")),
                        watch=(datadeps_dirs()...,), outdir::AbstractString=SCRIPT_DIR)
    done = Threads.Atomic{Bool}(false)
    t0 = time()
    sizes() = Dict(d => dir_bytes(d) for d in watch)
    base = Ref{Any}(nothing)            # measured lazily: the IAD tree is large
    hb = Threads.@spawn begin
        next = t0 + every
        while !done[]
            sleep(1)
            done[] && break
            time() < next && continue
            next += every
            base[] === nothing && (base[] = sizes(); continue)
            now = sizes()
            growth = join((@sprintf("%s +%.1f MB", d == outdir ? "script folder" : "datadeps",
                                    mb(now[d] - get(base[], d, 0))) for d in watch), ", ")
            dbg(@sprintf("  … still in %s after %.0f s (%s)", what, time() - t0, growth))
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

function print_env_info(outdir, models, rounds, explorer, recal)
    DEBUG[] || return nothing
    dbg("OctofitterHipparcos v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("target = ", TARGET, " | relative astrometry: ", length(ASTROM.epoch), " epochs, MJD ",
        minimum(ASTROM.epoch), "–", maximum(ASTROM.epoch))
    dbg("models = ", join((string(m, " (n_rounds=", rounds[m], " → ", 2^rounds[m], " samples)")
                           for m in models), ", "))
    dbg("mass-model explorer = ", explorer, " | recalibrate = ", recal)
    Threads.nthreads() == 1 &&
        dbg("  NOTE: running Pigeons on 1 thread (relaunch disabled) — expect a longer run")
    dbg("project    = ", Base.active_project())
    dbg("SCRIPT_DIR = ", SCRIPT_DIR, " | outdir = ", abspath(outdir))
    dbg("DATADEPS_ALWAYS_ACCEPT = ", get(ENV, "DATADEPS_ALWAYS_ACCEPT", "(unset)"))
    dbg("interactive = ", isinteractive(),
        " | Makie backend = ", CairoMakie.Makie.current_backend())
    return nothing
end

# ---- Chain helpers -----------------------------------------------------------
colvec(chain, p::Symbol) = vec(Array(chain[p]))
haspar(chain, p::Symbol) = p in names(chain)
qline(x) = Statistics.quantile(filter(isfinite, x), (0.16, 0.5, 0.84))

"First chain column whose name ends with `suffix` (IAD columns carry the
observation-name prefix, e.g. `Hipparcos_IAD_iad_Δra`)."
function findcol(chain, suffix::AbstractString)
    for n in names(chain)
        endswith(string(n), suffix) && return n
    end
    return nothing
end

"""
Fold an angle sample (radians) into a 180° window centred on its axial mean.
Relative astrometry can't tell Ω from Ω+180°; a plain `mod π` would split a
peak sitting near 0°/180°.
"""
function fold_axial(x)
    c = atan(Statistics.mean(sin.(2 .* x)), Statistics.mean(cos.(2 .* x))) / 2
    return mod.(x .- c .+ pi / 2, pi) .- pi / 2 .+ c
end

"Rows: (label, column or suffix, transform)."
summary_rows() = (
    ("plx [mas]",              "plx",            identity),
    ("A mass [M⊙]",            "A_mass",         identity),
    ("b mass [M_jup]",         "b_mass",         x -> x ./ mjup),
    ("b a [AU]",               "b_a",            identity),
    ("b e",                    "b_e",            identity),
    ("b i [deg]",              "b_i",            x -> rad2deg.(x)),
    ("b Ω [deg] (folded)",     "b_Ω",            x -> rad2deg.(fold_axial(x))),
    ("b θ [deg]",              "b_θ",            x -> rad2deg.(x)),
    ("IAD Δα* [mas]",          "iad_Δra",        identity),
    ("IAD Δδ [mas]",           "iad_Δdec",       identity),
    ("IAD μα* [mas/yr]",       "iad_pmra",       identity),
    ("IAD μδ [mas/yr]",        "iad_pmdec",      identity),
    ("IAD jitter [mas]",       "hip_iad_jitter", identity),
)

function colfor(chain, key)
    haspar(chain, Symbol(key)) && return Symbol(key)
    return startswith(key, "iad_") || startswith(key, "hip_") ? findcol(chain, key) : nothing
end

function print_summary(chain, label)
    DEBUG[] || return nothing
    dbg("Posterior median [16th, 84th] — ", label, " (", size(chain, 1) * size(chain, 3),
        " samples):")
    for (lab, key, f) in summary_rows()
        c = colfor(chain, key)
        c === nothing && continue
        q = qline(f(colvec(chain, c)))
        dbg(@sprintf("    %-22s %12.4f  [%12.4f, %12.4f]", lab, q[2], q[1], q[3]))
    end
    if haspar(chain, :b_a) && haspar(chain, :b_mass)
        M = haspar(chain, :A_mass) ? colvec(chain, :A_mass) : fill(1.75, length(colvec(chain, :b_a)))
        P = sqrt.(colvec(chain, :b_a) .^ 3 ./ (M .+ colvec(chain, :b_mass)))
        q = qline(P)
        dbg(@sprintf("    %-22s %12.4f  [%12.4f, %12.4f]   (derived)", "b P [yr]", q[2], q[1], q[3]))
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

const CMP1 = "#0072B2"   # PairPlots series 1
const CMP2 = "#E69F00"   # PairPlots series 2
const CSKY = "#56B4E9"

"IAD figure: along-scan residuals vs epoch (coloured by scan angle) and scan angles."
function iad_figure(tbl)
    fig = Figure(size=(1100, 720))
    ok = hasproperty(tbl, :reject) ? .!tbl.reject : trues(length(tbl.epoch))
    σ = hasproperty(tbl, :sres_renorm) ? tbl.sres_renorm : tbl.sres
    ψ = rad2deg.(tbl.scanAngle_rad)
    ax1 = Axis(fig[1, 1], xlabel="epoch [MJD]", ylabel="along-scan residual [mas]",
               title="HIP $(HIP_ID) Hipparcos IAD — $(count(ok)) scans used, " *
                     "$(count(.!ok)) rejected")
    errorbars!(ax1, tbl.epoch[ok], tbl.res[ok], σ[ok]; color=(:white, 0.35))
    sc = scatter!(ax1, tbl.epoch[ok], tbl.res[ok]; color=ψ[ok], colormap=:twilight,
                  colorrange=(-180, 180), markersize=7)
    any(.!ok) && scatter!(ax1, tbl.epoch[.!ok], tbl.res[.!ok]; color=:red, marker=:xcross,
                          markersize=9, label="rejected")
    any(.!ok) && axislegend(ax1, position=:rt)
    Colorbar(fig[1, 2], sc, label="scan angle [deg]")
    ax2 = Axis(fig[2, 1], xlabel="epoch [MJD]", ylabel="scan angle [deg]",
               title="Scan angles and along-scan parallax factors")
    scatter!(ax2, tbl.epoch, ψ; color=tbl.parallaxFactorAlongScan, colormap=:balance,
             markersize=7)
    Colorbar(fig[2, 2]; colormap=:balance,
             limits=extrema(tbl.parallaxFactorAlongScan), label="parallax factor (along scan)")
    return fig
end

"Relative astrometry on the sky (east left), sized to the data's shape."
function astrom_figure()
    x = ASTROM.sep .* sin.(ASTROM.pa)      # Δα* [mas], east positive
    y = ASTROM.sep .* cos.(ASTROM.pa)      # Δδ  [mas]
    σx = @. sqrt((ASTROM.σ_sep * sin(ASTROM.pa))^2 + (ASTROM.sep * ASTROM.σ_pa * cos(ASTROM.pa))^2)
    σy = @. sqrt((ASTROM.σ_sep * cos(ASTROM.pa))^2 + (ASTROM.sep * ASTROM.σ_pa * sin(ASTROM.pa))^2)
    pad = 25.0
    xr = (minimum(x) - pad, maximum(x) + pad)
    yr = (minimum(y) - pad, maximum(y) + pad)
    w = 700
    h = clamp(round(Int, w * (yr[2] - yr[1]) / (xr[2] - xr[1])) + 120, 400, 1100)
    fig = Figure(size=(w + 160, h))
    ax = Axis(fig[1, 1], xlabel="Δα* [mas]", ylabel="Δδ [mas]", aspect=DataAspect(),
              xreversed=true, title="51 Eri b relative astrometry (14 epochs)")
    errorbars!(ax, x, y, σx; direction=:x, color=(:white, 0.5))
    errorbars!(ax, x, y, σy; direction=:y, color=(:white, 0.5))
    sc = scatter!(ax, x, y; color=ASTROM.epoch, colormap=:viridis, markersize=10)
    limits!(ax, xr[1], xr[2], yr[1], yr[2])    # xreversed flips the display
    Colorbar(fig[1, 2], sc, label="epoch [MJD]")
    return fig
end

"Nielsen test: posterior histograms vs the catalog Normal for the five parameters."
function nielsen_figure(chain, hip_sol)
    comps = nielsen_comparisons(chain, hip_sol)
    fig = Figure(size=(1080, 720))
    ax = nothing
    for (k, c) in enumerate(comps)
        r, col = fldmod1(k, 3)
        ax = Axis(fig[r, col], xlabel=c.label, ylabel="density")
        hist!(ax, c.x; bins=55, normalization=:pdf, color=(CMP1, 0.85), label="posterior")
        n = Normal(c.μ, c.σ)
        xs = range(quantile(n, 1e-4), quantile(n, 1 - 1e-4), length=200)
        lines!(ax, xs, pdf.(n, xs); color="#EEEEEE", linewidth=2, label="Hipparcos catalog")
    end
    Legend(fig[2, 3], ax, tellwidth=false)
    Label(fig[0, 1:3], "Nielsen test — HIP $(HIP_ID) five-parameter solution recovered from the IAD",
          fontsize=16)
    return fig
end

"The five comparisons (docs' block): posterior sample, catalog μ, σ, label, docs key."
function nielsen_comparisons(chain, hip_sol)
    spec = (
        ("plx",       :plx,        hip_sol.plx,   hip_sol.e_plx,  "plx [mas]"),
        ("iad_Δra",   nothing,     0.0,           hip_sol.e_ra,   "Δα* [mas]"),
        ("iad_Δdec",  nothing,     0.0,           hip_sol.e_de,   "Δδ [mas]"),
        ("iad_pmra",  nothing,     hip_sol.pm_ra, hip_sol.e_pmra, "μα* [mas/yr]"),
        ("iad_pmdec", nothing,     hip_sol.pm_de, hip_sol.e_pmde, "μδ [mas/yr]"),
    )
    out = NamedTuple[]
    for (key, sym, μ, σ, lab) in spec
        c = sym === nothing ? findcol(chain, key) : sym
        c === nothing && error("chain has no column ending in $key; columns: $(names(chain))")
        push!(out, (; key, col=c, x=colvec(chain, c), μ=Float64(μ), σ=Float64(σ), label=lab))
    end
    return out
end

function print_nielsen_table(chain, hip_sol)
    dbg("Nielsen test — posterior mean ± sd vs Hipparcos catalog (z = Δ/σ_cat) [docs posterior]:")
    for c in nielsen_comparisons(chain, hip_sol)
        m, s = Statistics.mean(c.x), Statistics.std(c.x)
        d = DOCS_NIELSEN[c.key].post
        dbg(@sprintf("    %-14s %9.3f ± %.3f   catalog %9.3f ± %.3f   z = %+5.2f   [docs %7.2f ± %.2f]",
                     c.label, m, s, c.μ, c.σ, (m - c.μ) / c.σ, d[1], d[2]))
    end
    return nothing
end

function marginals_figure(chain, label)
    fig = Figure(size=(1100, 700))
    specs = (
        (:b_mass, x -> log10.(x ./ mjup), "log₁₀ b mass [M_jup]"),
        (:b_a,    identity,               "b a [AU]"),
        (:b_e,    identity,               "b e"),
        (:plx,    identity,               "plx [mas]"),
    )
    for (k, (p, f, lab)) in enumerate(specs)
        haspar(chain, p) || continue
        ax = Axis(fig[fldmod1(k, 2)...], xlabel=lab, ylabel="density")
        hist!(ax, f(colvec(chain, p)); bins=60, normalization=:pdf, color=(CMP1, 0.85))
    end
    Label(fig[0, 1:2], "Key marginals — $(label)", fontsize=16)
    return fig
end

# -----------------------------------------------------------------------------
# Model components (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
function straight_bodies()
    vars = @eval @variables begin
        mass = 1.0                  # host mass is not important for this model
    end
    return (Body(name="A", variables=vars),)
end

function mass_bodies()
    mA, mp, ap, ep, θep = PRIORS.M_A, PRIORS.m_b, PRIORS.a_b, PRIORS.e_b, PRIORS.θ_epoch
    A_vars = @eval @variables begin
        mass ~ $mA                  # [M⊙]
    end
    A = Body(name="A", variables=A_vars)
    b_vars = @eval @variables begin
        mass ~ $mp                  # [M⊙]; mjup is a constant
        a ~ $ap                     # [AU]
        e ~ $ep
        ω ~ Uniform(0, 2pi)
        i ~ Sine()
        Ω ~ Uniform(0, 2pi)
        θ ~ Uniform(0, 2pi)         # position angle at `epoch`
        epoch = $θep
    end
    dbg("  companion 'b': mass ~ LogUniform(0.1, 100) M_jup, a ~ N(10, 1) AU, θ at MJD ", θep)
    return (A, Body(name="b", about=A, variables=b_vars))
end

"HipparcosIADObs for HIP 21547 (first use downloads the Hipparcos_IAD DataDep)."
function iad_obs(target; blends=(), recal::Bool=false)
    obs = Octofitter.HipparcosIADObs(
        hip_id = HIP_ID,
        target = target,
        blends = blends,
        ref = Barycentre,
        renormalize = true,         # default: true
        recalibrate = recal,        # default: false
    )
    try                              # debug only; never fail the stage here
        s = obs.hip_sol
        dbg(@sprintf("  catalog solution: plx %.2f ± %.2f mas, μα* %.2f ± %.2f, μδ %.2f ± %.2f mas/yr",
                     s.plx, s.e_plx, s.pm_ra, s.e_pmra, s.pm_de, s.e_pmde))
        t = obs.table
        nrej = hasproperty(t, :reject) ? count(t.reject) : 0
        dbg("  IAD: ", length(t.epoch), " scans (", nrej, " rejected), MJD ",
            round(minimum(t.epoch), digits=1), "–", round(maximum(t.epoch), digits=1),
            " | observation name \"", obs.name, "\"")
    catch err
        dbg("  (IAD debug print skipped: ", first(sprint(showerror, err), 200), ")")
    end
    return obs
end

function astrom_obs()
    t = Table(; ASTROM...)
    vars = @eval @variables begin
        jitter = 0                  # mas
        northangle = 0              # rad
        platescale = 1              # relative
    end
    return RelAstromObs(t; target=:b, ref=:A, name="VLT/SPHERE", variables=vars)
end

"""
    build_system(kind; recal=false)

Fresh bodies and observations for `kind` ∈ (:straight, :mass).
Returns `(sys, hip_obs)`.
"""
function build_system(kind::Symbol; recal::Bool=false)
    if kind === :straight
        (A,) = straight_bodies()
        hip = iad_obs(A; recal)
        pp = PRIORS.plx_straight
        vars = @eval @variables begin
            plx ~ $pp               # [mas]
        end
        return System(name="c_Eri_straight_line", bodies=[A], observations=[hip],
                      variables=vars), hip
    elseif kind === :mass
        A, b = mass_bodies()
        hip = iad_obs(A; blends=(b,), recal)
        pp = PRIORS.plx_mass
        vars = @eval @variables begin
            plx ~ $pp               # [mas]
        end
        return System(name="cEri", bodies=[A, b], observations=[hip, astrom_obs()],
                      variables=vars), hip
    end
    error("kind must be :straight or :mass, got $kind")
end

"Page's starting point for the mass model (falls back to automatic initialisation)."
function initialize_mass!(model)
    start = (;
        plx = 34.0,
        observations = (;
            Hipparcos_IAD = (;
                iad_Δra = 0.0,
                iad_Δdec = 0.0,
                iad_Δpmra = 0.0,
                iad_Δpmdec = 0.0,
            ),
        ),
    )
    try
        return Octofitter.initialize!(model, start)
    catch err
        dbg("  ⚠ initialize! with the page's starting point failed (",
            first(sprint(showerror, err), 300), "); using automatic initialisation")
        return Octofitter.initialize!(model)
    end
end

function write_iad_csv(p, tbl)
    cols = (:iorb, :epoch, :epoch_yrs, :res, :sres, :sres_renorm, :scanAngle_rad,
            :parallaxFactorAlongScan, :reject)
    keep = [c for c in cols if hasproperty(tbl, c)]
    CSV.write(p, NamedTuple{Tuple(keep)}(Tuple(collect(getproperty(tbl, c)) for c in keep)))
    return p
end

write_astrom_csv(p) = (CSV.write(p, ASTROM); p)

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
rounds_from_env() = Dict(k => parse(Int, get(ENV, v[1], v[2])) for (k, v) in MODEL_ROUNDS_ENV)

"""
Extra octofit_pigeons keywords: SliceSampler explorer for the mass model (page)
and, if OCTOHIP_CHAINS is set, the number of tempered chains (Pigeons n_chains).
"""
function pigeons_kwargs(use_slice::Bool)
    kw = Pair{Symbol,Any}[]
    use_slice && push!(kw, :explorer => Pigeons.SliceSampler())
    nc = strip(get(ENV, "OCTOHIP_CHAINS", ""))
    isempty(nc) || push!(kw, :n_chains => parse(Int, nc))
    return (; kw...)
end

"""
The mass posterior has a light mode and a narrow mode piled against the prior's
100 M_jup upper edge (the IAD frame offsets absorb the star's reflex position,
so 2.5 yr of Hipparcos curvature can't exclude it). Report both.
"""
function print_mass_modes(chain; cut_mjup=50.0)
    m = colvec(chain, :b_mass) ./ mjup
    heavy = m .> cut_mjup
    f = Statistics.mean(heavy)
    dbg(@sprintf("  samples with b mass > %.0f M_jup: %.1f%% (%d of %d)", cut_mjup, 100f,
                 count(heavy), length(m)))
    for (lab, sel) in (("light (≤ $(Int(cut_mjup)) M_jup)", .!heavy), ("heavy (> $(Int(cut_mjup)) M_jup)", heavy))
        count(sel) < 3 && continue
        med(p, g=identity) = Statistics.median(g.(colvec(chain, p)[sel]))
        dΔδ = findcol(chain, "iad_Δdec")
        dbg(@sprintf("    %-22s mass %6.1f M_jup | a %5.2f AU | e %.2f | i %5.1f° | IAD Δδ %7.2f mas",
                     lab, Statistics.median(m[sel]), med(:b_a), med(:b_e), med(:b_i, rad2deg),
                     dΔδ === nothing ? NaN : med(dΔδ)))
    end
    f > 0.05 && dbg("  NOTE: the heavy mode is isolated (see the Pigeons min(α) column); its weight ",
                    "may not be reliable. Check it with OCTOHIP_ROUNDS_MASS=10 and/or OCTOHIP_CHAINS=32.")
    return f
end
explorer_from_env() = lowercase(get(ENV, "OCTOHIP_EXPLORER", "slice"))

"""
    run_hipparcos(; models=(:straight, :mass), rounds=rounds_from_env(),
                    explorer="slice", recal=false, prior_z0=false,
                    outdir=SCRIPT_DIR, prefix="hip_", dark=true,
                    corner_dark=false, show_plots=true)

Build each model, sample with Pigeons, plot and summarise.
"""
function run_hipparcos(; models=MODEL_ORDER,
                         rounds=rounds_from_env(),
                         explorer::AbstractString=explorer_from_env(),
                         recal::Bool=envbool("OCTOHIP_RECAL", "false"),
                         prior_z0::Bool=false,
                         outdir::AbstractString=SCRIPT_DIR,
                         prefix::AbstractString="hip_",
                         dark::Bool=true,
                         corner_dark::Bool=false,
                         show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    accept_downloads!()
    print_env_info(outdir, models, rounds, explorer, recal)
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
    watchdirs() = (datadeps_dirs()..., outdir)

    soft_stage("Check DataDep folders for interrupted downloads") do
        list_datadeps()
        check_stale_datadeps()
    end
    soft_stage("Relative astrometry: CSV + sky plot") do
        push!(outputs, write_astrom_csv(joinpath(outdir, prefix * "51eri_relastrom.csv")))
        savefig!(astrom_figure(), "relastrom_data.png")
    end

    modelsd = Dict{Symbol,Any}(); chains = Dict{Symbol,Any}()
    logZ = Dict{Symbol,Float64}(); logZ0 = Dict{Symbol,Float64}()
    iad_done = false

    for kind in models
        tag = string(kind)
        nr = rounds[kind]
        sys, hip = stage("[$tag] Build system — $(MODEL_LABELS[kind]) (IAD load; heartbeat every " *
                         get(ENV, "OCTOHIP_HEARTBEAT", "30") * " s)") do
            with_heartbeat("HipparcosIADObs"; watch=watchdirs(), outdir) do
                inout(() -> build_system(kind; recal))
            end
        end
        DEBUG[] && display(sys)

        if !iad_done
            soft_stage("IAD snapshot CSV") do
                push!(outputs, write_iad_csv(joinpath(outdir, prefix * "iad_$(HIP_ID).csv"), hip.table))
            end
            soft_stage("Plot IAD residuals and scan angles") do
                savefig!(iad_figure(hip.table), "iad_data.png")
            end
            iad_done = true
        end

        model = stage("[$tag] Compile LogDensityModel") do
            Octofitter.LogDensityModel(sys)
        end
        DEBUG[] && display(model)
        modelsd[kind] = model

        stage("[$tag] initialize!" * (kind === :mass ? " (page starting point: plx = 34, zero IAD offsets)" : "")) do
            kind === :mass ? initialize_mass!(model) : Octofitter.initialize!(model)
        end
        use_slice = kind === :mass && explorer == "slice"
        chain = stage("[$tag] Pigeons parallel tempering (n_rounds=$nr → $(2^nr) scans" *
                      (use_slice ? ", SliceSampler explorer)" : ")")) do
            kw = pigeons_kwargs(use_slice)
            dbg("  octofit_pigeons kwargs: n_rounds = ", nr,
                isempty(kw) ? "" : string(", ", join(("$k = $v" for (k, v) in pairs(kw)), ", ")))
            c, pt = octofit_pigeons(model; n_rounds=nr, kw...)
            z = soft_stage("[$tag] Log-evidence ratio (stepping stone)") do
                zz = Pigeons.stepping_stone(pt)
                dbg(@sprintf("  [%s] log(Z₁/Z₀) ≈ %.3f, Λ = %.2f", tag, zz, Pigeons.global_barrier(pt)))
                zz
            end
            z === nothing || (logZ[kind] = z)
            c
        end
        chains[kind] = chain
        DEBUG[] && display(chain)          # MCMCChains table: check rhat ≈ 1, ess
        print_summary(chain, MODEL_LABELS[kind])

        if kind === :straight
            soft_stage("[$tag] Nielsen-test table vs catalog and docs") do
                print_nielsen_table(chain, hip.hip_sol)
            end
            soft_stage("[$tag] Nielsen-test comparison figure") do
                savefig!(nielsen_figure(chain, hip.hip_sol), "straight_nielsen.png")
            end
        else
            soft_stage("[$tag] b mass [M_jup] quantiles (page)") do
                q = Statistics.quantile(colvec(chain, :b_mass) ./ mjup, (0.16, 0.5, 0.84))
                dbg("  b mass [Mjup]: ", round.(q, digits=2))
            end
            soft_stage("[$tag] Mass modes (light vs prior-edge heavy)") do
                print_mass_modes(chain)
            end
            soft_stage("[$tag] Key-marginal histograms") do
                savefig!(marginals_figure(chain, MODEL_LABELS[kind]), "$(tag)_marginals.png")
            end
        end

        if prior_z0
            soft_stage("[$tag] Prior-only run for log Z0") do
                pm = Octofitter.LogDensityModel(Octofitter.prior_only_model(sys, exclude_all=true))
                Octofitter.initialize!(pm)
                _, ptp = octofit_pigeons(pm; n_rounds=nr)
                z0 = Pigeons.stepping_stone(ptp)
                logZ0[kind] = z0
                dbg(@sprintf("  [%s] log Z0 ≈ %.4f  (≈0 means the priors are properly normalised)", tag, z0))
            end
        end

        soft_stage("[$tag] hipparcosplot (one draw)") do
            savefig!(inout(() -> Octofitter.hipparcosplot(model, chain)), "$(tag)_hipparcosplot.png")
        end
        soft_stage("[$tag] octoplot") do
            savefig!(inout(() -> octoplot(model, chain)), "$(tag)_octoplot.png")
        end
        soft_stage("[$tag] Corner plot") do
            # small=true keeps only orbital elements; the star-only model has
            # none, so octocorner would drop every column. Show them all.
            small = kind !== :straight
            savefig!(inout(() -> corner(model, chain; dark=corner_dark, small)), "$(tag)_corner.png")
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

    # ---- Evidence ------------------------------------------------------------
    if !isempty(logZ)
        lines = String[]
        push!(lines, "Log evidence (stepping stone), recalibrate = $(recal):")
        for k in MODEL_ORDER
            haskey(logZ, k) || continue
            z0s = haskey(logZ0, k) ? @sprintf("%8.4f", logZ0[k]) : "not run (OCTOHIP_PRIOR_Z0=true)"
            push!(lines, @sprintf("    %-40s n_rounds = %2d   log(Z/Z0) = %10.3f   log Z0 = %s",
                                  MODEL_LABELS[k], rounds[k], logZ[k], z0s))
        end
        push!(lines, "    (the two models use different data, so their evidences are not compared)")
        foreach(l -> dbg(l), lines)
        soft_stage("Write evidence table") do
            p = joinpath(outdir, prefix * "evidence.txt")
            write(p, join(lines, "\n") * "\n")
            push!(outputs, p)
        end
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f s", time() - T0[]))
    return (; models=modelsd, chains, logZ, logZ0, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default when the REPL has 1 thread)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString="hip_")
    T0[] = time()
    accept_downloads!()                  # inherited by the child
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOHIP_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOHIP_CHILD=1]")
    dbg("  Child output follows; its plots are shown here when it finishes.")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOHIP_RELAUNCH\"]=\"false\".")
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
    for kind in MODEL_ORDER
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
    elseif envbool("OCTOHIP_SHOW_PLOTS", "true")
        soft_stage("Show $(length(pngs)) saved plot(s) in the plot pane") do
            show_saved_pngs(pngs)
        end
    end
    return (; relaunched=true, success=ok, chains, outputs=new_files)
end

"Plot order for display: data first, then straight-line, then mass model."
function png_rank(p)
    f = basename(p)
    occursin("_data", f) && return 0
    m = startswith(f, "hip_straight") ? 10 : startswith(f, "hip_mass") ? 20 : 30
    for (k, key) in enumerate(("nielsen", "hipparcosplot", "octoplot", "corner", "marginals"))
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

"Parse OCTOHIP_MODELS (comma-separated) into a tuple of Symbols in canonical order."
function models_from_env()
    s = get(ENV, "OCTOHIP_MODELS", "straight,mass")
    req = Set(Symbol(strip(lowercase(x))) for x in split(s, ',') if !isempty(strip(x)))
    issubset(req, Set(MODEL_ORDER)) ||
        error("OCTOHIP_MODELS must list straight and/or mass, got \"$s\"")
    return Tuple(k for k in MODEL_ORDER if k in req)
end

println(@sprintf("[HIP] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterHipparcos

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterHipparcos.jl`  -> runs
#   - VSCode "Execute active File in REPL"          -> runs; with 1 thread it
#     relaunches in `julia --threads=auto` (OCTOHIP_RELAUNCH=false to disable,
#     OCTOHIP_THREADS to set the count)
#   - ENV["OCTOHIP_AUTORUN"] = "false"; include()   -> loads module only
#   - OCTOHIP_MODELS, OCTOHIP_ROUNDS_*, OCTOHIP_EXPLORER, OCTOHIP_RECAL,
#     OCTOHIP_PRIOR_Z0 select what runs
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOHIP_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOHIP_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOHIP_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global hip_result = OctofitterHipparcos.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOHIP_THREADS", "auto"))
            println("[HIP] Child run finished. `hip_result` has fields: relaunched, success, ",
                    "chains (loaded from FITS), outputs")
        else
            global hip_result = OctofitterHipparcos.run_hipparcos(
                models=OctofitterHipparcos.models_from_env(),
                prior_z0=lowercase(get(ENV, "OCTOHIP_PRIOR_Z0", "false")) != "false",
                show_plots=!is_child)
            is_child || println("[HIP] Result stored in `hip_result` (fields: models, chains, ",
                "logZ, logZ0, outputs)")
        end
    end
end

nothing
