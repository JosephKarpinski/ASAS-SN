#=
================================================================================
 OctofitterG23H.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Joint Gaia-Hipparcos
 Astrometry (G23H)":
   https://sefffal.github.io/Octofitter.jl/dev/g23h/

 Target: HIP 384 (Gaia DR3 2738776816458107136), a star with a known
 companion. Data: the G23H catalog (Thompson et al. 2026) via `G23HObs`,
 combining Hipparcos proper motions + IAD, the Hipparcos–Gaia long-term
 proper motion, calibrated Gaia DR2, the DR3−DR2 scaled position difference,
 Gaia DR3 proper motions and the DR3 astrometric excess noise (RUWE/UEVA).
 Gaia RV variability is left out (include_rv=false), as on the page.

 Models (all sampled with Pigeons parallel tempering, page settings:
 n_chains=32, SliceSampler, no variational reference, multithreaded):
   :full   the page's model: every G23H channel for this source, companion b
           with P ~ LogUniform(1 d, 10 000 yr), mass ratio q ~ LogUniform(1e-5, 1),
           mass = q·M_pri (M_pri = 1 M⊙), M0 at epoch 57388.5; dark in G and Hp.
   :hgca   optional extra: the page's `restricted` observation (the HGCA
           channel set: Hipparcos, Hipparcos–Gaia, DR3; ueva_mode = :none),
           fitted with the same bodies, for comparison with :full.
 Plus: channel table + CSV of the catalog proper motions, a figure of those
 proper motions with their averaging windows, the page's quick starting-point
 guess (guess_starting_position with threaded Kepler solving), optional extra
 rounds via increment_n_rounds! (the page's convergence advice), posterior
 summary with Ω folded into 180°, octoplot (sky track, catalog proper motions,
 Hipparcos abscissae), light-theme corner, dotplot (period and separation),
 the page's custom period–mass pairplot, key-marginal histograms, full-vs-HGCA
 comparison when both run, evidence table.

 freeze_epochs: `true` (default, as on the page) fixes the Gaia epoch
 selection — fast but approximate. OCTOG23H_FREEZE=false samples it (slow).

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `g23h_result`)
   Shell:    julia --threads=auto OctofitterG23H.jl
   Library:  ENV["OCTOG23H_AUTORUN"] = "false"; include("OctofitterG23H.jl")
             res = OctofitterG23H.run_g23h(models=(:full, :hgca), n_rounds=8)

 Threads: every fit uses Pigeons. In a 1-thread REPL the run is relaunched in
 a child `julia --threads=auto` (output streamed here; chains loaded back and
 saved plots shown in the plot pane afterwards). OCTOG23H_RELAUNCH=false
 disables this; OCTOG23H_THREADS sets the count.

 Convenience ENV settings (also passed to a relaunched child):
   OCTOG23H_MODELS       = "full" (default) or "full,hgca"
   OCTOG23H_ROUNDS       = "6" (default, page; ~12 for production)
   OCTOG23H_EXTRA_ROUNDS = "0" (default) — after the first fit, continue with
                           increment_n_rounds!(pt, k) (page's convergence route)
   OCTOG23H_CHAINS       = "32" (default, page) — Pigeons n_chains
   OCTOG23H_INIT         = "page" (default: guess_starting_position with 10 000
                           prior draws) or "initialize" (Octofitter.initialize!)
   OCTOG23H_FREEZE       = "true" (default) — freeze_epochs
   OCTOG23H_CATALOG      = "subset" (default) | "full" | "/path/to/G23H-v1.0.feather"
   OCTOG23H_SHOW_PLOTS   = "true" (default) — show the child's PNGs here
   OCTOG23H_HEARTBEAT    = "30" (default) — seconds between "still waiting" lines
   OCTOG23H_ACCEPT_DOWNLOADS = "true" (default) — sets DATADEPS_ALWAYS_ACCEPT

 Catalog data (as in OctofitterPMA): "subset" uses what the docs build uses —
 the 4-row test catalog `G23H-test-subset.feather` (HIP 384 is one of its
 rows, all 115 columns incl. the DR2 sidecar column) and the cached GOST
 forecast `GOST-1.1927097109938027-1.5368044203832403-dr3.csv`, downloaded
 once from the Octofitter repo (pinned commit) next to this file and shared
 with the other modules. "full" uses the ~14 GB G23H DataDep + ~300 MB DR2
 sidecar + live GOST query; delete the empty G23H_Catalog DataDep folder
 first (the module reports it). The Hipparcos IAD DataDep (installed) is used
 for the :iad_hip channel.

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOG23H_ENV_MODE=local uses ./octofitter_v9_env). Needs Octofitter
 v9, CairoMakie, PairPlots, Distributions, Pigeons, Arrow, CSV.

 Outputs (prefix "g23h_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-25  Initial version on the OctofitterHipparcos v1.0.1 /
                   OctofitterPMA v1.0.3 scaffolding ([G23H +t] debug stages,
                   soft optional stages, env bootstrap with v9 guards,
                   @__DIR__ outputs, dark theme with light corner plots,
                   threaded relaunch with exit-code-only failure report and
                   PNGs shown in the parent's plot pane, heartbeat and
                   stale-DataDep checks, docs' offline catalog subset + GOST
                   forecast, @eval-built @variables). Adds: G23HObs full and
                   HGCA-restricted models; page's starting-point guess and
                   Pigeons settings; optional increment_n_rounds!; channel
                   table/CSV/figure; custom period–mass pairplot; comparison.
 v1.0.1 2026-09-25 First run (3.0 min, 4 threads, n_rounds=6 → 64 samples):
                   12 channels, page starting-point guess worked (log post
                   −1718 → sampled), Λ = 10.8 and still rising, log(Z₁/Z₀)
                   = −265.2 stable from round 4. Adds a log₁₀-axes version of
                   the page's period–mass pairplot (the linear one squeezes
                   the bulk into a corner).
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "CairoMakie", "PairPlots", "Distributions", "Pigeons",
                "Arrow", "CSV"),
    min_version = Dict("Octofitter" => v"9"),
    mode = lowercase(get(ENV, "OCTOG23H_ENV_MODE", "temp"))

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
            println("[G23H] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[G23H] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[G23H] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[G23H] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[G23H] Existing v9 environment found at ", Base.active_project())
        else
            Pkg.add(spec.(collect(required)))
        end
        println("[G23H] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[G23H] Loading Octofitter, Pigeons, CairoMakie, PairPlots, Distributions, Arrow, CSV ",
        "(first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterG23H

const LOAD_T0 = time()

using Octofitter
using CairoMakie
using PairPlots
using Distributions
using Pigeons
import Arrow
import CSV
import Downloads
import Random
import Statistics
using Printf

export run_g23h, build_system, set_plot_theme!

const VERSION_STRING = "1.0.1"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterG23H needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

# Target (as on the tutorial page)
const TARGET = "HIP 384"
const GAIA_ID = 2738776816458107136

# Offline inputs used by the docs build (pinned to a known Octofitter commit)
const OCTO_COMMIT = "8c6daaaf5292b6cbdb0aa4f4b1e9a1a5f0bd21f3"
const OCTO_RAW = "https://raw.githubusercontent.com/sefffal/Octofitter.jl/$(OCTO_COMMIT)/"
const SUBSET_FILE = "G23H-test-subset.feather"
const SUBSET_URL  = OCTO_RAW * "test/" * SUBSET_FILE
const GOST_FILE   = "GOST-1.1927097109938027-1.5368044203832403-dr3.csv"
const GOST_URL    = OCTO_RAW * "docs/src/" * GOST_FILE

const PRIORS = (
    M_pri   = 1.0,                                      # [M⊙], fixed (page)
    P_b     = LogUniform(1.0, 10000 * year2day_julian), # [d]
    q_b     = LogUniform(1e-5, 1),
    e_b     = Uniform(0, 0.9),
    M0_ep   = 57388.5,                                  # [MJD]
    pm_half = 10.0,                                     # ±10 mas/yr around DR3
)

const HGCA_CHANNELS = (:ra_hip, :dec_hip, :ra_hg, :dec_hg, :ra_dr3, :dec_dr3)
const PM_KINDS = r"^(ra|dec)_(hip|hg|dr2|dr32|dr3)$"

const MODEL_LABELS = Dict(:full => "G23H, all channels (page model)",
                          :hgca => "G23H restricted to HGCA channels")
const MODEL_ORDER = (:full, :hgca)

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterG23H.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[G23H +%7.1fs] ", time() - T0[]), msg...)
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

"Let DataDeps download without an interactive y/n prompt (unless disabled)."
function accept_downloads!()
    envbool("OCTOG23H_ACCEPT_DOWNLOADS", "true") && (ENV["DATADEPS_ALWAYS_ACCEPT"] = "true")
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

"Top-level DataDeps entries (sizes skipped: the IAD tree has 236k files)."
function list_datadeps()
    ds = datadeps_dirs()
    isempty(ds) && (dbg("  no DataDeps folders yet"); return nothing)
    for d in ds
        dbg("  DataDeps folder: ", d)
        for e in sort(readdir(d))
            dbg("    ", e, isdir(joinpath(d, e)) ? "/" : "")
        end
    end
    return nothing
end

"OCTOG23H_CATALOG: \"subset\" (default), \"full\", or a path to a G23H feather."
function catalog_mode()
    s = strip(get(ENV, "OCTOG23H_CATALOG", "subset"))
    l = lowercase(s)
    return l in ("subset", "full") ? l : String(s)
end

"""
Report DataDep folders left empty by an interrupted download (DataDeps treats
an existing folder as already downloaded). G23H folders only matter in full
mode; Hipparcos_IAD must contain its unpacked ResRec_JavaTool_2014 directory.
"""
function check_stale_datadeps()
    stale = String[]
    full = catalog_mode() == "full"
    for d in datadeps_dirs(), e in readdir(d)
        p = joinpath(d, e)
        isdir(p) || continue
        bad = e == "Hipparcos_IAD" ? !isdir(joinpath(p, "ResRec_JavaTool_2014")) :
              e == "G23H_Catalog"  ? !isfile(joinpath(p, "G23H-v1.0.feather")) :
                                     isempty(readdir(p))
        bad || continue
        if startswith(e, "G23H_") && !full
            dbg("  note: empty DataDep folder ", p, " (not used in subset mode; delete it before ",
                "OCTOG23H_CATALOG=full:  rm(\"", p, "\"; recursive=true))")
        else
            push!(stale, p)
            dbg("  ⚠ incomplete DataDep folder (interrupted download?): ", p)
            dbg("    delete it so DataDeps downloads again:  rm(\"", p, "\"; recursive=true)")
        end
    end
    isempty(stale) && dbg("  no incomplete DataDep folders that this run uses")
    return stale
end

"""
    with_heartbeat(f, what; every, watch, outdir)

Run `f()` while a background task prints a line every `every` seconds with the
elapsed time and how much the watched folders have grown.
"""
function with_heartbeat(f, what::AbstractString;
                        every::Real=parse(Float64, get(ENV, "OCTOG23H_HEARTBEAT", "30")),
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

function print_env_info(outdir, models, n_rounds, extra, n_chains, init, freeze)
    DEBUG[] || return nothing
    dbg("OctofitterG23H v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("target = ", TARGET, " (Gaia DR3 ", GAIA_ID, ")")
    dbg("models = ", join(models, ", "), " | n_rounds = ", n_rounds, " (", 2^n_rounds,
        " samples)", extra > 0 ? " + $extra extra round(s) via increment_n_rounds!" : "",
        " | n_chains = ", n_chains, " | init = ", init)
    dbg("freeze_epochs = ", freeze, freeze ? " (fast approximation)" : " (sampled epoch selection; slow)",
        " | include_rv = false")
    dbg("catalog mode = ", catalog_mode(),
        catalog_mode() == "subset" ? " (docs' 4-row G23H subset + cached GOST forecast)" :
        catalog_mode() == "full" ? " (14 GB G23H DataDep + live GOST query)" :
                                   " (local G23H file + live GOST query)")
    Threads.nthreads() == 1 &&
        dbg("  NOTE: running Pigeons on 1 thread (relaunch disabled) — expect a long run")
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

"First chain column whose name ends with `suffix` (observation variables carry
the observation-name prefix, e.g. `G23H_σ_AL`)."
function findcol(chain, suffix::AbstractString)
    for n in names(chain)
        endswith(string(n), suffix) && return n
    end
    return nothing
end

"""
Fold an angle sample (radians) into a 180° window centred on its axial mean.
Astrometry alone can't tell Ω from Ω+180°; a plain `mod π` would split a peak
sitting near 0°/180°.
"""
function fold_axial(x)
    c = atan(Statistics.mean(sin.(2 .* x)), Statistics.mean(cos.(2 .* x))) / 2
    return mod.(x .- c .+ pi / 2, pi) .- pi / 2 .+ c
end

"Rows: (label, column name or suffix, transform)."
summary_rows() = (
    ("b P [yr]",            "b_P",     x -> x ./ year2day_julian),
    ("b q (mass ratio)",    "b_q",     identity),
    ("b mass [M_jup]",      "b_mass",  x -> x ./ mjup),
    ("b e",                 "b_e",     identity),
    ("b i [deg]",           "b_i",     x -> rad2deg.(x)),
    ("b Ω [deg] (folded)",  "b_Ω",     x -> rad2deg.(fold_axial(x))),
    ("plx [mas]",           "plx",     identity),
    ("pmra [mas/yr]",       "pmra",    identity),
    ("pmdec [mas/yr]",      "pmdec",   identity),
    ("σ_AL [mas]",          "_σ_AL",   identity),
    ("σ_att [mas]",         "_σ_att",  identity),
    ("σ_calib [mas]",       "_σ_calib", identity),
)

colfor(chain, key) = startswith(key, "_") ? findcol(chain, key) :
                     haspar(chain, Symbol(key)) ? Symbol(key) : nothing

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
    if haspar(chain, :b_mass)
        m = colvec(chain, :b_mass) ./ mjup
        dbg(@sprintf("    b mass: %.1f%% < 13 M_jup (planet), %.1f%% 13–80 (brown dwarf), %.1f%% > 80 (star)",
                     100 * Statistics.mean(m .< 13), 100 * Statistics.mean(13 .<= m .<= 80),
                     100 * Statistics.mean(m .> 80)))
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

const CAT_COLORS = Dict("hip" => "#56B4E9", "hg" => "#E69F00", "dr2" => "#CC79A7",
                        "dr32" => "#F0E442", "dr3" => "#009E73")
const CAT_NAMES  = Dict("hip" => "Hipparcos", "hg" => "Hipparcos–Gaia (long-term)",
                        "dr2" => "Gaia DR2", "dr32" => "DR3−DR2 (scaled Δpos)",
                        "dr3" => "Gaia DR3")
const CMP1 = "#0072B2"   # PairPlots series 1
const CMP2 = "#E69F00"   # PairPlots series 2

"Split `kind` (e.g. :ra_dr32) into (axis, catalog)."
function split_kind(k)
    s = string(k)
    i = findfirst('_', s)
    i === nothing && return (s, s)
    return (s[1:i-1], s[i+1:end])
end

"Indices of the proper-motion rows of the channel table."
pm_rows(tbl) = findall(k -> occursin(PM_KINDS, string(k)), tbl.kind)

function pm_figure(tbl)
    rows = pm_rows(tbl)
    fig = Figure(size=(1150, 480))
    for (col, (axis, ylab)) in enumerate((("ra", "μ_α* [mas/yr]"), ("dec", "μ_δ [mas/yr]")))
        ax = Axis(fig[1, col], xlabel="epoch [MJD]", ylabel=ylab,
                  title="$(TARGET) catalog proper motions — $(axis == "ra" ? "RA" : "Dec")")
        for j in rows
            a, cat = split_kind(tbl.kind[j])
            a == axis || continue
            c = get(CAT_COLORS, cat, "#EEEEEE")
            if hasproperty(tbl, :start_epoch)
                rangebars!(ax, [tbl.pm[j]], [tbl.start_epoch[j]], [tbl.stop_epoch[j]];
                           direction=:x, color=(c, 0.7), whiskerwidth=8)
            end
            errorbars!(ax, [tbl.epoch[j]], [tbl.pm[j]], [tbl.σ_pm[j]]; color=c, whiskerwidth=8)
            scatter!(ax, [tbl.epoch[j]], [tbl.pm[j]]; color=c, markersize=10,
                     label=get(CAT_NAMES, cat, cat))
        end
        col == 1 && axislegend(ax, position=:rt, labelsize=11, unique=true)
    end
    Label(fig[2, 1:2], "Horizontal bars: catalog averaging window (not an epoch uncertainty)",
          fontsize=11)
    return fig
end

function write_pm_csv(p, tbl)
    rows = pm_rows(tbl)
    cols = (:kind, :epoch, :start_epoch, :stop_epoch, :pm, :σ_pm)
    keep = [c for c in cols if hasproperty(tbl, c)]
    CSV.write(p, NamedTuple{Tuple(keep)}(Tuple(
        c === :kind ? string.(tbl.kind[rows]) : collect(getproperty(tbl, c)[rows]) for c in keep)))
    return p
end

"The page's custom pairplot of period [yr] vs mass [M_jup] (light theme)."
function period_mass_pairplot(chain)
    data = (; P_yr = colvec(chain, :b_P) ./ 365.25, mass = colvec(chain, :b_mass) ./ mjup)
    return with_theme(Theme()) do
        pairplot(
            data => (
                PairPlots.Scatter(markersize=4),
                PairPlots.MarginHist(),
                PairPlots.MarginQuantileText(),
            ),
            labels = Dict(:P_yr => "period [yr]", :mass => "mass [Mⱼᵤₚ]"),
        )
    end
end

"""
Same pairplot on log₁₀ axes: the posterior spans days to millennia and
10⁻² to 10³ M_jup, so the page's linear version squeezes the bulk into a corner.
"""
function period_mass_pairplot_log(chain)
    data = (; logP = log10.(colvec(chain, :b_P) ./ 365.25),
              logm = log10.(colvec(chain, :b_mass) ./ mjup))
    return with_theme(Theme()) do
        pairplot(
            data => (
                PairPlots.Scatter(markersize=4),
                PairPlots.MarginHist(),
                PairPlots.MarginQuantileText(),
            ),
            labels = Dict(:logP => "log₁₀ period [yr]", :logm => "log₁₀ mass [Mⱼᵤₚ]"),
        )
    end
end

function marginals_figure(chain, label)
    fig = Figure(size=(1100, 700))
    specs = (
        (:b_mass, x -> log10.(x ./ mjup),            "log₁₀ b mass [M_jup]"),
        (:b_P,    x -> log10.(x ./ year2day_julian), "log₁₀ b P [yr]"),
        (:b_e,    identity,                          "b e"),
        (:b_i,    x -> rad2deg.(x),                  "b i [deg]"),
    )
    for (k, (p, f, lab)) in enumerate(specs)
        haspar(chain, p) || continue
        ax = Axis(fig[fldmod1(k, 2)...], xlabel=lab, ylabel="density")
        hist!(ax, f(colvec(chain, p)); bins=60, normalization=:pdf, color=(CMP1, 0.85))
    end
    Label(fig[0, 1:2], "Key marginals — $(label)", fontsize=16)
    return fig
end

function compare_figure(ch1, ch2)
    fig = Figure(size=(1100, 420))
    specs = ((:b_mass, x -> log10.(x ./ mjup),            "log₁₀ b mass [M_jup]"),
             (:b_P,    x -> log10.(x ./ year2day_julian), "log₁₀ b P [yr]"))
    for (k, (p, f, lab)) in enumerate(specs)
        ax = Axis(fig[1, k], xlabel=lab, ylabel="density")
        haspar(ch1, p) && hist!(ax, f(colvec(ch1, p)); bins=60, normalization=:pdf,
                                color=(CMP1, 0.6), label="all channels")
        haspar(ch2, p) && hist!(ax, f(colvec(ch2, p)); bins=60, normalization=:pdf,
                                color=(CMP2, 0.6), label="HGCA channels")
        k == 1 && axislegend(ax, position=:rt)
    end
    return fig
end

# -----------------------------------------------------------------------------
# Catalog inputs
# -----------------------------------------------------------------------------
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

"""
    load_catalog_inputs(; outdir=SCRIPT_DIR, mode=catalog_mode())

Keyword inputs for G23HObs: `(; catalog, forecast_table)` for "subset",
`(; catalog=path)` for a local G23H file, `(;)` for "full" (DataDep + GOST).
"""
function load_catalog_inputs(; outdir::AbstractString=SCRIPT_DIR, mode=catalog_mode())
    if mode == "subset"
        cpath = fetch_once(SUBSET_URL, joinpath(outdir, SUBSET_FILE))
        gpath = fetch_once(GOST_URL, joinpath(outdir, GOST_FILE))
        cat = Arrow.Table(cpath)
        ids = collect(cat.gaia_source_id)
        idx = findfirst(==(GAIA_ID), ids)
        idx === nothing && error("Gaia DR3 $GAIA_ID is not in $cpath (rows: $(ids))")
        dbg("  subset catalog: ", length(ids), " rows, ", length(propertynames(cat)),
            " columns; ", TARGET, " is row ", idx, " (HIP ", cat.hip_id[idx], ")")
        dbg(@sprintf("  row: plx %.4f ± %.4f mas | pmra hip %.2f, hg %.2f, dr2 %.2f, dr3 %.2f mas/yr",
                     cat.parallax[idx], cat.parallax_error[idx], cat.pmra_hip[idx],
                     cat.pmra_hg[idx], cat.pmra_dr2[idx], cat.pmra_dr3[idx]))
        gost = CSV.read(gpath, Table; normalizenames=true)
        forecast = Table(
            epoch = Octofitter.jd2mjd.(gost.ObservationTimeAtBarycentre_BarycentricJulianDateInTCB_),
            scanAngle_rad = gost.scanAngle_rad_,
            parallaxFactorAlongScan = gost.parallaxFactorAlongScan,
        )
        dbg(@sprintf("  GOST forecast: %d transits, MJD %.0f–%.0f", length(forecast.epoch),
                     minimum(forecast.epoch), maximum(forecast.epoch)))
        return (; catalog=cat, forecast_table=forecast)
    elseif mode == "full"
        dbg("  full mode: first use downloads the ~14 GB G23H catalog and ~300 MB DR2 sidecar;")
        dbg("  GOST is queried live.")
        stale = check_stale_datadeps()
        any(p -> occursin("G23H_", p), stale) &&
            error("Incomplete G23H DataDep folder(s) found (see ⚠ lines above); delete them " *
                  "and run again, or use OCTOG23H_CATALOG=subset.")
        return (;)
    else
        isfile(mode) || error("OCTOG23H_CATALOG=\"$mode\" is not \"subset\", \"full\", or an existing file")
        dbg(@sprintf("  local G23H catalog: %s (%.2f GB); GOST queried live", mode, filesize(mode) / 1024^3))
        return (; catalog=mode)
    end
end

# -----------------------------------------------------------------------------
# Model components (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
function make_bodies()
    A_vars = @eval @variables begin
        mass = system.M_pri       # primary mass [M⊙], declared in the system block
        flux_G  = 1.0             # the target defines the contrast scale in each band
        flux_Hp = 1.0
    end
    A = Body(name="A", variables=A_vars)
    Pp, qp, ep, M0ep = PRIORS.P_b, PRIORS.q_b, PRIORS.e_b, PRIORS.M0_ep
    b_vars = @eval @variables begin
        P ~ $Pp                   # [d] (1 d to 10 000 yr)
        q ~ $qp                   # mass ratio
        mass = q * system.M_pri   # [M⊙]
        e ~ $ep
        ω ~ Uniform(0, 2pi)
        i ~ Sine()
        Ω ~ Uniform(0, 2pi)
        M0 ~ Uniform(0, 2pi)
        epoch = $M0ep
        flux_G  = 0.0             # dark to Gaia
        flux_Hp = 0.0             # …and to Hipparcos
    end
    dbg("  companion 'b': P ~ LogUniform(1 d, 10 000 yr), q ~ LogUniform(1e-5, 1), mass = q·M_pri, dark")
    return A, Body(name="b", about=A, variables=b_vars)
end

"G23HObs for model `kind`; `inputs` are the catalog/forecast keywords."
function make_obs(kind::Symbol, A, b; freeze::Bool=true, inputs=(;))
    obs = if kind === :full
        G23HObs(;
            gaia_id = GAIA_ID,
            target = A,
            blends = (b,),
            ref = Barycentre,
            include_rv = false,
            freeze_epochs = freeze,
            inputs...,
        )
    else
        G23HObs(;
            gaia_id = GAIA_ID,
            target = A, blends = (b,), ref = Barycentre,
            channels = HGCA_CHANNELS,
            ueva_mode = :none,
            include_rv = false,
            freeze_epochs = freeze,
            inputs...,
        )
    end
    try                                  # debug only; never fail the stage here
        ks = string.(obs.table.kind)
        counts = Dict{String,Int}()
        foreach(k -> counts[k] = get(counts, k, 0) + 1, ks)
        dbg("  channels (", length(ks), " rows): ",
            join((k * (counts[k] > 1 ? "×$(counts[k])" : "") for k in unique(ks)), ", "))
        c = obs.catalog
        g(k) = hasproperty(c, k) && !ismissing(getproperty(c, k)) ? Float64(getproperty(c, k)) : NaN
        dbg(@sprintf("  catalog row: ra %.6f°, dec %.6f°, plx %.4f ± %.4f mas, RV %.3f km/s, RUWE %.3f",
                     g(:ra), g(:dec), g(:parallax), g(:parallax_error), g(:radial_velocity), g(:ruwe_dr3)))
        dbg("  Hipparcos IAD rows: ", length(obs.hip_table), " | Gaia forecast pool: ",
            length(obs.gaia_table), " | observation name \"", obs.name, "\"")
    catch err
        dbg("  (observation debug print skipped: ", first(sprint(showerror, err), 200), ")")
    end
    return obs
end

"The page's system block, with catalog values interpolated."
function make_system(kind::Symbol, A, b, obs)
    c = obs.catalog
    plx0, eplx = Float64(c.parallax), Float64(c.parallax_error)
    plx_prior = truncated(Normal(plx0, eplx), lower=max(0, plx0 - 10eplx))
    h = PRIORS.pm_half
    pmra_p = Uniform(c.pmra_dr3 - h, c.pmra_dr3 + h)
    pmdec_p = Uniform(c.pmdec_dr3 - h, c.pmdec_dr3 + h)
    ra0, dec0 = c.ra, c.dec
    rv_raw = hasproperty(c, :radial_velocity) ? c.radial_velocity : missing
    rv0 = (ismissing(rv_raw) || isnan(rv_raw)) ? 0.0 : rv_raw * 1e3    # [m/s]
    refep = Octofitter.meta_gaia_DR3.ref_epoch_mjd
    Mp = PRIORS.M_pri
    vars = @eval @variables begin
        M_pri = $Mp               # primary mass [M⊙]
        plx ~ $plx_prior          # catalog value, truncated
        pmra ~ $pmra_p            # wide (±10 mas/yr) — the data constrain these
        pmdec ~ $pmdec_p
        ra = $ra0                 # fixed coordinates and RV from the catalog
        dec = $dec0
        rv = $rv0                 # [m/s]
        ref_epoch = $refep
    end
    dbg(@sprintf("  system: plx ~ N(%.4f, %.4f) truncated, pmra ~ U(%.2f ± %g), pmdec ~ U(%.2f ± %g), rv = %.1f m/s",
                 plx0, eplx, c.pmra_dr3, h, c.pmdec_dr3, h, rv0))
    return System(name = kind === :full ? "HIP384" : "HIP384_hgca",
                  bodies=[A, b], observations=[obs], variables=vars)
end

"""
    build_system(kind; freeze=true, inputs=load_catalog_inputs())

Fresh bodies, observation and system for `kind` ∈ (:full, :hgca).
Returns `(sys, obs)`.
"""
function build_system(kind::Symbol; freeze::Bool=true, inputs=load_catalog_inputs())
    kind in MODEL_ORDER || error("kind must be :full or :hgca, got $kind")
    A, b = make_bodies()
    obs = make_obs(kind, A, b; freeze, inputs)
    return make_system(kind, A, b, obs), obs
end

"""
The page's quick starting point: the best of `n` prior draws (Kepler solving
threaded), linked and copied as every chain's start. Falls back to
`Octofitter.initialize!` if that fails.
"""
function page_initialize!(model; n::Integer=10_000)
    try
        Octofitter._kepsolve_use_threads[] = true
        initial_θ = collect(Octofitter.guess_starting_position(model, n)[1])
        model.starting_points = fill(collect(model.link(initial_θ)), 100)
        lp = model.ℓπcallback(model.starting_points[1])
        dbg(@sprintf("  best of %d prior draws: log posterior = %.3f (copied to 100 starting points)", n, lp))
        return lp
    catch err
        dbg("  ⚠ page starting-point guess failed (", first(sprint(showerror, err), 300),
            "); using Octofitter.initialize!")
        return Octofitter.initialize!(model)
    end
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_g23h(; models=(:full,), n_rounds=6, extra_rounds=0, n_chains=32,
               init="page", freeze=true, outdir=SCRIPT_DIR, prefix="g23h_",
               dark=true, corner_dark=false, show_plots=true)

Build each model, sample with Pigeons (page settings), plot and summarise.
"""
function run_g23h(; models=(:full,),
                    n_rounds::Integer=envint("OCTOG23H_ROUNDS", "6"),
                    extra_rounds::Integer=envint("OCTOG23H_EXTRA_ROUNDS", "0"),
                    n_chains::Integer=envint("OCTOG23H_CHAINS", "32"),
                    init::AbstractString=lowercase(get(ENV, "OCTOG23H_INIT", "page")),
                    freeze::Bool=envbool("OCTOG23H_FREEZE", "true"),
                    outdir::AbstractString=SCRIPT_DIR,
                    prefix::AbstractString="g23h_",
                    dark::Bool=true,
                    corner_dark::Bool=false,
                    show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    accept_downloads!()
    print_env_info(outdir, models, n_rounds, extra_rounds, n_chains, init, freeze)
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
    inputs = stage("Catalog inputs (mode = $(catalog_mode()))") do
        with_heartbeat("catalog inputs"; watch=watchdirs(), outdir) do
            load_catalog_inputs(; outdir)
        end
    end

    modelsd = Dict{Symbol,Any}(); chains = Dict{Symbol,Any}()
    logZ = Dict{Symbol,Float64}(); Λs = Dict{Symbol,Float64}()
    data_done = false

    for kind in models
        tag = string(kind)
        dbg("[$tag] Building — ", MODEL_LABELS[kind])
        A, b = stage("[$tag] Bodies (A, b)") do
            make_bodies()
        end
        obs = stage("[$tag] G23HObs (Hipparcos IAD + Gaia epochs; heartbeat every " *
                    get(ENV, "OCTOG23H_HEARTBEAT", "30") * " s)") do
            with_heartbeat("G23HObs"; watch=watchdirs(), outdir) do
                inout(() -> make_obs(kind, A, b; freeze, inputs))
            end
        end
        sys = stage("[$tag] Assemble System (catalog priors)") do
            make_system(kind, A, b, obs)
        end
        DEBUG[] && display(sys)

        if !data_done
            soft_stage("Channel table") do
                DEBUG[] && display(obs.table)
            end
            soft_stage("Catalog proper motions: CSV + figure") do
                push!(outputs, write_pm_csv(joinpath(outdir, prefix * "pm_$(GAIA_ID).csv"), obs.table))
                savefig!(pm_figure(obs.table), "pm_data.png")
            end
            data_done = true
        end

        model = stage("[$tag] Compile LogDensityModel") do
            Octofitter.LogDensityModel(sys)
        end
        DEBUG[] && display(model)
        modelsd[kind] = model

        stage("[$tag] Starting points (" * (init == "page" ? "page: guess_starting_position, 10 000 draws" :
                                                            "Octofitter.initialize!") * ")") do
            init == "page" ? page_initialize!(model) : Octofitter.initialize!(model)
        end

        chain = stage("[$tag] Pigeons (n_rounds=$n_rounds, n_chains=$n_chains, SliceSampler, " *
                      "no variational reference)") do
            c, pt = octofit_pigeons(model;
                n_chains = n_chains,
                n_rounds = n_rounds,
                explorer = Pigeons.SliceSampler(),
                n_chains_variational = 0,
                variational = nothing,
                multithreaded = true,
            )
            report(pt, r) = soft_stage("[$tag] log(Z₁/Z₀) and Λ after round $r") do
                zz = Pigeons.stepping_stone(pt)
                λ = Pigeons.global_barrier(pt)
                dbg(@sprintf("  [%s] round %d: log(Z₁/Z₀) ≈ %.3f, Λ = %.2f", tag, r, zz, λ))
                logZ[kind] = zz; Λs[kind] = λ
            end
            report(pt, n_rounds)
            if extra_rounds > 0
                c2 = soft_stage("[$tag] increment_n_rounds!(pt, $extra_rounds) and continue") do
                    Pigeons.increment_n_rounds!(pt, extra_rounds)
                    cc, pt2 = octofit_pigeons(pt)
                    report(pt2, n_rounds + extra_rounds)
                    cc
                end
                c2 === nothing ? dbg("  keeping the first-pass chain") : (c = c2)
            end
            c
        end
        chains[kind] = chain
        DEBUG[] && display(chain)          # MCMCChains table: check rhat ≈ 1, ess
        print_summary(chain, MODEL_LABELS[kind])

        soft_stage("[$tag] octoplot (sky track, proper motions, Hipparcos abscissae)") do
            savefig!(inout(() -> octoplot(model, chain)), "$(tag)_octoplot.png")
        end
        soft_stage("[$tag] Corner plot (small=true)") do
            savefig!(inout(() -> corner(model, chain; dark=corner_dark)), "$(tag)_corner.png")
        end
        soft_stage("[$tag] dotplot (mass vs period)") do
            savefig!(inout(() -> Octofitter.dotplot(model, chain; mode=:period)),
                     "$(tag)_dotplot_period.png")
        end
        soft_stage("[$tag] dotplot (mass vs separation)") do
            savefig!(inout(() -> Octofitter.dotplot(model, chain)), "$(tag)_dotplot.png")
        end
        soft_stage("[$tag] Period–mass pairplot (page)") do
            savefig!(period_mass_pairplot(chain), "$(tag)_period_mass.png")
        end
        soft_stage("[$tag] Period–mass pairplot, log₁₀ axes") do
            savefig!(period_mass_pairplot_log(chain), "$(tag)_period_mass_log.png")
        end
        soft_stage("[$tag] Key-marginal histograms") do
            savefig!(marginals_figure(chain, MODEL_LABELS[kind]), "$(tag)_marginals.png")
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

    if haskey(chains, :full) && haskey(chains, :hgca)
        soft_stage("All-channels vs HGCA-channels comparison") do
            savefig!(compare_figure(chains[:full], chains[:hgca]), "compare_full_hgca.png")
        end
    end

    # ---- Evidence ------------------------------------------------------------
    if !isempty(logZ)
        lines = String[]
        push!(lines, "Log evidence (stepping stone), freeze_epochs = $(freeze), " *
                     "n_rounds = $(n_rounds + extra_rounds), n_chains = $(n_chains):")
        for k in MODEL_ORDER
            haskey(logZ, k) || continue
            push!(lines, @sprintf("    %-38s log(Z₁/Z₀) = %10.3f   Λ = %6.2f", MODEL_LABELS[k],
                                  logZ[k], Λs[k]))
        end
        push!(lines, "    (the two models use different data, so their evidences are not compared;")
        push!(lines, "     for convergence, add rounds until log(Z₁/Z₀) and Λ stop drifting)")
        foreach(l -> dbg(l), lines)
        soft_stage("Write evidence table") do
            p = joinpath(outdir, prefix * "evidence.txt")
            write(p, join(lines, "\n") * "\n")
            push!(outputs, p)
        end
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f s", time() - T0[]))
    return (; models=modelsd, chains, logZ, Λ=Λs, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default when the REPL has 1 thread)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString="g23h_")
    T0[] = time()
    accept_downloads!()                  # inherited by the child
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOG23H_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOG23H_CHILD=1]")
    dbg("  Child output follows; its plots are shown here when it finishes.")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOG23H_RELAUNCH\"]=\"false\".")
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
    elseif envbool("OCTOG23H_SHOW_PLOTS", "true")
        soft_stage("Show $(length(pngs)) saved plot(s) in the plot pane") do
            show_saved_pngs(pngs)
        end
    end
    return (; relaunched=true, success=ok, chains, outputs=new_files)
end

"Plot order for display: data first, then full model, HGCA model, comparison."
function png_rank(p)
    f = basename(p)
    occursin("_data", f) && return 0
    occursin("compare", f) && return 99
    m = startswith(f, "g23h_full") ? 10 : startswith(f, "g23h_hgca") ? 20 : 30
    for (k, key) in enumerate(("octoplot", "corner", "dotplot_period", "dotplot", "period_mass_log",
                               "period_mass", "marginals"))
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

"Parse OCTOG23H_MODELS (comma-separated) into a tuple of Symbols in canonical order."
function models_from_env()
    s = get(ENV, "OCTOG23H_MODELS", "full")
    req = Set(Symbol(strip(lowercase(x))) for x in split(s, ',') if !isempty(strip(x)))
    issubset(req, Set(MODEL_ORDER)) ||
        error("OCTOG23H_MODELS must list full and/or hgca, got \"$s\"")
    return Tuple(k for k in MODEL_ORDER if k in req)
end

println(@sprintf("[G23H] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterG23H

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterG23H.jl`      -> runs
#   - VSCode "Execute active File in REPL"          -> runs; with 1 thread it
#     relaunches in `julia --threads=auto` (OCTOG23H_RELAUNCH=false to disable,
#     OCTOG23H_THREADS to set the count)
#   - ENV["OCTOG23H_AUTORUN"] = "false"; include()  -> loads module only
#   - OCTOG23H_MODELS, _ROUNDS, _EXTRA_ROUNDS, _CHAINS, _INIT, _FREEZE, _CATALOG
#     select what runs
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOG23H_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOG23H_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOG23H_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global g23h_result = OctofitterG23H.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOG23H_THREADS", "auto"))
            println("[G23H] Child run finished. `g23h_result` has fields: relaunched, success, ",
                    "chains (loaded from FITS), outputs")
        else
            global g23h_result = OctofitterG23H.run_g23h(
                models=OctofitterG23H.models_from_env(),
                show_plots=!is_child)
            is_child || println("[G23H] Result stored in `g23h_result` (fields: models, chains, ",
                "logZ, Λ, outputs)")
        end
    end
end

nothing
