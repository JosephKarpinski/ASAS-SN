#=
================================================================================
 OctofitterPMA.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Proper Motion Anomaly":
   https://sefffal.github.io/Octofitter.jl/dev/pma/

 Target: HD 91312 A & B (SCExAO discovery, arXiv:2109.12124), Gaia DR3 source
 756291174721509376. Data: the Hipparcos–Gaia Catalog of Accelerations (HGCA,
 arXiv:2105.11662) via `HGCAObs` — six proper motions (Hipparcos, the long-term
 Hipparcos–Gaia difference, Gaia DR3; RA and Dec each), each averaged over its
 catalog window. The companion is modelled as a photocentre offset about the
 barycentre; the barycentre's own track (ra, dec, plx, pmra, pmdec, rv at the
 DR3 reference epoch) is propagated in 3-D by Octofitter.

 Models (all sampled with Pigeons, which also gives the log evidence):
   :dark       the page's model: companion dark in G and Hp (flux_G = flux_Hp = 0)
   :luminous   optional extra: flux_G, flux_Hp ~ U(0, 1) contrast vs the host
               (the page notes this for luminous unresolved companions)
 Plus: HGCA data figure and CSV snapshot, MCMCChains summary (rhat/ess),
 posterior summary with Ω folded into a 180° window, octoplot proper-motion
 panels, light-theme corner plot, dotplot (mass vs separation and vs period),
 key-marginal histograms, optional prior-only log Z0, evidence table, and a
 dark-vs-luminous comparison when both models run.

 freeze_epochs: `true` (default, as on the page) fixes the Gaia epoch
 selection — fast but approximate. OCTOPMA_FREEZE=false samples the selection
 as nuisance parameters (production fit, much slower).

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `pma_result`)
   Shell:    julia --threads=auto OctofitterPMA.jl
   Library:  ENV["OCTOPMA_AUTORUN"] = "false"; include("OctofitterPMA.jl")
             res = OctofitterPMA.run_pma(models=(:dark, :luminous), n_rounds=9)

 Threads: every fit uses Pigeons. In a 1-thread REPL the run is relaunched in
 a child `julia --threads=auto` (output streamed here; chains loaded back
 afterwards). OCTOPMA_RELAUNCH=false disables this; OCTOPMA_THREADS sets the
 count.

 Convenience ENV settings (also passed to a relaunched child):
   OCTOPMA_MODELS   = "dark" (default) or "dark,luminous"
   OCTOPMA_ROUNDS   = "10" (default)
   OCTOPMA_FREEZE   = "true" (default) — freeze_epochs for HGCAObs
   OCTOPMA_PRIOR_Z0 = "false" (default) — prior-only evidence runs
   OCTOPMA_SHOW_PLOTS = "true" (default) — after a relaunched child run, show
             the saved PNGs in the parent's plot pane
   OCTOPMA_HEARTBEAT = "30" (default) — seconds between "still waiting" lines
             during download/query steps
   OCTOPMA_CATALOG  = "subset" (default) | "full" | "/path/to/G23H-v1.0.feather"
             where the HGCA numbers come from (see "Catalog data" below)
   OCTOPMA_ACCEPT_DOWNLOADS = "true" (default) — sets DATADEPS_ALWAYS_ACCEPT
             so one-time DataDeps downloads don't wait for a y/n answer that a
             child process can't receive

 Catalog data. In Octofitter v9, HGCAObs is a wrapper around G23HObs, which
 reads the G23H catalog (Thompson et al. 2026; ~14 GB DataDep, plus a ~300 MB
 DR2 sidecar) and queries the GOST Gaia scan forecast over the network.
   "subset" (default) — what the docs build does: the 4-row test catalog
             `G23H-test-subset.feather` (contains HD 91312, all 115 columns,
             including the DR2 sidecar column) and the cached GOST forecast
             `GOST-158.30707896392835-40.42555422701387-dr3.csv`, both
             downloaded once from the Octofitter GitHub repo (pinned commit)
             into this file's folder and reused by later runs/modules.
   "full"   — the 14 GB G23H DataDep + live GOST query (production).
   a path   — a local copy of the full G23H feather; GOST queried live.
 In every mode the Hipparcos IAD DataDep (~332 MB zip from ESA) is downloaded
 once, and gaia_plx queries the Gaia TAP service (falls back to the catalog
 row's parallax if that fails). DataDeps are stored under
 ~/.julia/scratchspaces/<uuid>/datadeps; a folder left behind by an
 interrupted download is detected and reported with the path to delete.

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOPMA_ENV_MODE=local uses ./octofitter_v9_env). Needs Octofitter
 v9, CairoMakie, PairPlots, Distributions, Pigeons, Arrow, CSV.

 Outputs (prefix "pma_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-25  Initial version on the OctofitterRVMulti v1.0.1 scaffolding
                   ([PMA +t] debug stages, soft optional stages, env bootstrap
                   with v9 guards, @__DIR__ outputs, dark theme with light
                   corner plots, default threaded relaunch, initialize! before
                   every Pigeons run, @eval-built @variables). Adds: HGCAObs
                   with blends and freeze_epochs switch; gaia_plx prior; dark /
                   luminous companion models; HGCA window figure + CSV;
                   dotplot; Ω folding; evidence file read back by the parent.
 v1.0.1 2026-09-25 First run sat silently in HGCAObs (first-run downloads) and
                   showed no plots in the pane. Now: system build split into
                   separately timed stages (bodies / HGCAObs / gaia_plx /
                   System); a heartbeat prints every 30 s during network steps
                   with the growth of ~/.julia/datadeps and the script folder
                   so a slow download is distinguishable from a hang; datadeps
                   contents listed after the HGCA step; after a relaunched
                   child finishes, the parent loads the saved PNGs and shows
                   them in the plot pane (OCTOPMA_SHOW_PLOTS=false disables).
 v1.0.2 2026-09-25 HGCAObs failed: "G23H-v1.0.feather: No such file". v9's
                   HGCAObs needs the 14 GB G23H catalog; the v1.0 run began
                   that download silently and the interrupted DataDep folder
                   was then treated as complete. Now: OCTOPMA_CATALOG switch,
                   default "subset" = the docs' one-row catalog + cached GOST
                   forecast (downloaded next to the script, pinned commit);
                   stale/empty DataDep folders detected with the path to
                   delete; heartbeat and listing watch the real DataDeps
                   location (scratchspaces); gaia_plx falls back to the
                   catalog-row parallax; child failure reports only the exit
                   code (no environment dump); Arrow + CSV added.
 v1.0.3 2026-09-25 First complete run (9.7 min, 4 threads): Λ = 9.06 and
                   log(Z₁/Z₀) = -24.8 vs docs 9.06 / -24.1. Cosmetic: evidence
                   table says "not run" instead of NaN for log Z0 and prints
                   the docs reference values; an empty G23H DataDep folder is
                   a note (not ⚠) in subset mode, since it isn't used there.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "CairoMakie", "PairPlots", "Distributions", "Pigeons",
                "Arrow", "CSV"),
    min_version = Dict("Octofitter" => v"9"),
    mode = lowercase(get(ENV, "OCTOPMA_ENV_MODE", "temp"))

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
            println("[PMA] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[PMA] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[PMA] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[PMA] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[PMA] Existing v9 environment found at ", Base.active_project())
        else
            Pkg.add(spec.(collect(required)))
        end
        println("[PMA] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[PMA] Loading Octofitter, Pigeons, CairoMakie, PairPlots, Distributions, Arrow, CSV ",
        "(first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterPMA

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

export run_pma, build_pma_system, set_plot_theme!

const VERSION_STRING = "1.0.3"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterPMA needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

# Target and priors (as on the tutorial page)
const TARGET = "HD 91312"
const GAIA_ID = 756291174721509376

# Offline inputs used by the docs build (pinned to a known Octofitter commit)
const OCTO_COMMIT = "8c6daaaf5292b6cbdb0aa4f4b1e9a1a5f0bd21f3"
const OCTO_RAW = "https://raw.githubusercontent.com/sefffal/Octofitter.jl/$(OCTO_COMMIT)/"
const SUBSET_FILE = "G23H-test-subset.feather"
const SUBSET_URL  = OCTO_RAW * "test/" * SUBSET_FILE
const GOST_FILE   = "GOST-158.30707896392835-40.42555422701387-dr3.csv"
const GOST_URL    = OCTO_RAW * "docs/src/" * GOST_FILE
const PRIORS = (
    M_A    = truncated(Normal(1.61, 0.1), lower=0.1),   # [M⊙]
    m_b    = LogUniform(0.5mjup, 1000mjup),             # [M⊙]
    a_b    = LogUniform(0.1, 100.0),                    # [AU]
    e_b    = Uniform(0, 0.9),
    tp_b   = Uniform(50000, 60000),                     # [MJD]
    pmra   = Uniform(-137 - 100, -137 + 100),           # [mas/yr]
    pmdec  = Uniform(2 - 100, 2 + 100),                 # [mas/yr]
    flux   = Uniform(0, 1),                             # luminous model only
)

const MODEL_LABELS = Dict(:dark => "dark companion (page model)",
                          :luminous => "luminous companion (flux_G, flux_Hp ~ U(0,1))")
const MODEL_ORDER = (:dark, :luminous)

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterPMA.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[PMA +%7.1fs] ", time() - T0[]), msg...)
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

"Let DataDeps download without an interactive y/n prompt (unless disabled)."
function accept_downloads!()
    if lowercase(get(ENV, "OCTOPMA_ACCEPT_DOWNLOADS", "true")) != "false"
        ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    end
    return nothing
end

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

"Folder label for heartbeat lines."
watch_label(d, outdir) = d == outdir ? "script folder" :
                         occursin("scratchspaces", d) ? "datadeps(scratch)" : "datadeps"

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

"List the top-level entries of every DataDeps folder with sizes."
function list_datadeps()
    ds = datadeps_dirs()
    isempty(ds) && (dbg("  no DataDeps folders yet"); return nothing)
    for d in ds
        dbg("  DataDeps folder: ", d)
        for e in sort(readdir(d))
            p = joinpath(d, e)
            dbg(@sprintf("    %10.1f MB  %s", mb(isdir(p) ? dir_bytes(p) : filesize(p)), e))
        end
    end
    return nothing
end

"Files a complete DataDep folder must contain (others: just non-empty)."
const DATADEP_EXPECT = Dict("G23H_Catalog" => "G23H-v1.0.feather")

"""
Report DataDep folders left incomplete by an interrupted download (DataDeps
treats an existing folder as already downloaded, so the next run fails with
"No such file"). Returns the list of stale folders.
"""
function check_stale_datadeps()
    stale = String[]
    for d in datadeps_dirs(), e in readdir(d)
        p = joinpath(d, e)
        isdir(p) || continue
        need = get(DATADEP_EXPECT, e, nothing)
        bad = need === nothing ? dir_bytes(p) == 0 : !isfile(joinpath(p, need))
        bad || continue
        push!(stale, p)
        g23h = startswith(e, "G23H_")
        if g23h && catalog_mode() == "subset"
            dbg("  note: empty DataDep folder ", p, " (not used in subset mode; delete it before ",
                "OCTOPMA_CATALOG=full:  rm(\"", p, "\"; recursive=true))")
        else
            dbg("  ⚠ incomplete DataDep folder (interrupted download?): ", p)
            dbg("    delete it so DataDeps downloads again:  rm(\"", p, "\"; recursive=true)")
        end
    end
    isempty(stale) && dbg("  no incomplete DataDep folders found")
    return stale
end

"""
    with_heartbeat(f, what; every, watch)

Run `f()` while a background task prints a line every `every` seconds with the
elapsed time and how much the watched folders have grown. It separates a slow
first-run download (sizes growing) from a stalled network request (no growth).
"""
function with_heartbeat(f, what::AbstractString;
                        every::Real=parse(Float64, get(ENV, "OCTOPMA_HEARTBEAT", "30")),
                        watch=(datadeps_dirs()...,), outdir::AbstractString=SCRIPT_DIR)
    done = Threads.Atomic{Bool}(false)
    t0 = time()
    base = Dict(d => dir_bytes(d) for d in watch)
    hb = Threads.@spawn begin
        next = t0 + every
        while !done[]
            sleep(1)
            done[] && break
            time() < next && continue
            next += every
            growth = join((@sprintf("%s +%.1f MB", watch_label(d, outdir),
                                    mb(dir_bytes(d) - get(base, d, 0))) for d in watch), ", ")
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

function print_env_info(outdir, models, n_rounds, freeze_epochs)
    DEBUG[] || return nothing
    dbg("OctofitterPMA v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("target = ", TARGET, " (Gaia DR3 ", GAIA_ID, ")")
    dbg("models = ", join(models, ", "), " | n_rounds = ", n_rounds,
        " (", 2^n_rounds, " scans → ", 2^n_rounds, " posterior samples each)",
        " | freeze_epochs = ", freeze_epochs,
        freeze_epochs ? " (fast approximation)" : " (sampled epoch selection; slow)")
    Threads.nthreads() == 1 &&
        dbg("  NOTE: running Pigeons on 1 thread (relaunch disabled) — expect a long run")
    dbg("project    = ", Base.active_project())
    dbg("SCRIPT_DIR = ", SCRIPT_DIR, " | outdir = ", abspath(outdir))
    dbg("catalog mode = ", catalog_mode(),
        catalog_mode() == "subset" ? " (docs' one-row G23H subset + cached GOST forecast)" :
        catalog_mode() == "full" ? " (14 GB G23H DataDep + live GOST query)" :
                                   " (local G23H file + live GOST query)")
    dbg("DATADEPS_ALWAYS_ACCEPT = ", get(ENV, "DATADEPS_ALWAYS_ACCEPT", "(unset)"))
    for d in datadeps_dirs()
        dbg(@sprintf("  DataDeps folder %s (%.1f MB)", d, mb(dir_bytes(d))))
    end
    dbg("interactive = ", isinteractive(),
        " | Makie backend = ", CairoMakie.Makie.current_backend())
    return nothing
end

colvec(chain, p::Symbol) = vec(Array(chain[p]))
haspar(chain, p::Symbol) = p in names(chain)
qline(x) = Statistics.quantile(filter(isfinite, x), (0.16, 0.5, 0.84))

"""
Fold an angle sample (radians) into a 180° window centred on its axial mean.
Astrometry alone can't tell Ω from Ω+180°; a plain `mod π` would split a peak
sitting near 0°/180°.
"""
function fold_axial(x)
    c = atan(Statistics.mean(sin.(2 .* x)), Statistics.mean(cos.(2 .* x))) / 2
    return mod.(x .- c .+ pi / 2, pi) .- pi / 2 .+ c
end

"Star-mass column (name differs between Octofitter versions)."
star_mass(chain) = haspar(chain, :A_mass) ? colvec(chain, :A_mass) :
                   haspar(chain, :M) ? colvec(chain, :M) :
                   fill(Statistics.mean(PRIORS.M_A), size(chain, 1) * size(chain, 3))

"Rows: (label, column, transform)."
summary_rows() = (
    ("A mass [M⊙]",          :A_mass,    identity),
    ("b mass [M_jup]",       :b_mass,    x -> x ./ mjup),
    ("b a [AU]",             :b_a,       identity),
    ("b e",                  :b_e,       identity),
    ("b i [deg]",            :b_i,       x -> rad2deg.(x)),
    ("b Ω [deg] (folded)",   :b_Ω,       x -> rad2deg.(fold_axial(x))),
    ("b tp [MJD]",           :b_tp,      identity),
    ("b flux_G",             :b_flux_G,  identity),
    ("b flux_Hp",            :b_flux_Hp, identity),
    ("plx [mas]",            :plx,       identity),
    ("pmra [mas/yr]",        :pmra,      identity),
    ("pmdec [mas/yr]",       :pmdec,     identity),
)

function print_summary(chain, label)
    DEBUG[] || return nothing
    dbg("Posterior median [16th, 84th] — ", label, " (", size(chain, 1) * size(chain, 3),
        " samples):")
    for (lab, p, f) in summary_rows()
        haspar(chain, p) || continue
        q = qline(f(colvec(chain, p)))
        dbg(@sprintf("    %-22s %12.4f  [%12.4f, %12.4f]", lab, q[2], q[1], q[3]))
    end
    if haspar(chain, :b_a) && haspar(chain, :b_mass)
        P = sqrt.(colvec(chain, :b_a) .^ 3 ./ (star_mass(chain) .+ colvec(chain, :b_mass)))
        q = qline(P)
        dbg(@sprintf("    %-22s %12.4f  [%12.4f, %12.4f]   (derived)", "b P [yr]", q[2], q[1], q[3]))
        m = colvec(chain, :b_mass) ./ mjup
        dbg(@sprintf("    fraction with b mass > 80 M_jup (stellar): %.1f%%",
                     100 * Statistics.mean(m .> 80)))
    end
    return nothing
end

"Interpretation of ln BF (Jeffreys-type scale used in the Octofitter docs)."
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

const CAT_COLORS = Dict("hip" => "#56B4E9", "hg" => "#E69F00", "dr3" => "#009E73")
const CAT_NAMES  = Dict("hip" => "Hipparcos", "hg" => "Hipparcos–Gaia (long-term)",
                        "dr3" => "Gaia DR3")
const CMP1 = "#0072B2"   # PairPlots series 1
const CMP2 = "#E69F00"   # PairPlots series 2

"The observation's data table (rows: epoch, start_epoch, stop_epoch, pm, σ_pm, kind)."
function obs_table(obs)
    hasproperty(obs, :table) || error("observation has no `.table` field")
    return obs.table
end

"Split `kind` (e.g. \"ra_hip\") into (axis, catalog)."
function split_kind(k)
    s = string(k)
    i = findfirst('_', s)
    i === nothing && return (s, s)
    return (s[1:i-1], s[i+1:end])
end

function hgca_figure(tbl)
    fig = Figure(size=(1100, 460))
    kinds = split_kind.(tbl.kind)
    for (col, (axis, ylab)) in enumerate((("ra", "μ_α* [mas/yr]"), ("dec", "μ_δ [mas/yr]")))
        ax = Axis(fig[1, col], xlabel="epoch [MJD]", ylabel=ylab,
                  title="$(TARGET) HGCA — $(axis == "ra" ? "RA" : "Dec") proper motions")
        for (j, (a, cat)) in enumerate(kinds)
            a == axis || continue
            c = get(CAT_COLORS, cat, "#EEEEEE")
            rangebars!(ax, [tbl.pm[j]], [tbl.start_epoch[j]], [tbl.stop_epoch[j]];
                       direction=:x, color=(c, 0.7), whiskerwidth=8)
            errorbars!(ax, [tbl.epoch[j]], [tbl.pm[j]], [tbl.σ_pm[j]]; color=c, whiskerwidth=8)
            scatter!(ax, [tbl.epoch[j]], [tbl.pm[j]]; color=c, markersize=10,
                     label=get(CAT_NAMES, cat, cat))
        end
        col == 1 && axislegend(ax, position=:rt, labelsize=11)
    end
    Label(fig[2, 1:2], "Horizontal bars: catalog averaging window (not an epoch uncertainty)",
          fontsize=11)
    return fig
end

function marginals_figure(chain, label)
    fig = Figure(size=(1100, 700))
    specs = (
        (:b_mass, x -> log10.(x ./ mjup), "log₁₀ b mass [M_jup]"),
        (:b_a,    x -> log10.(x),         "log₁₀ b a [AU]"),
        (:b_e,    identity,               "b e"),
        (:b_i,    x -> rad2deg.(x),       "b i [deg]"),
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
    specs = ((:b_mass, x -> log10.(x ./ mjup), "log₁₀ b mass [M_jup]"),
             (:b_a,    x -> log10.(x),         "log₁₀ b a [AU]"))
    for (k, (p, f, lab)) in enumerate(specs)
        ax = Axis(fig[1, k], xlabel=lab, ylabel="density")
        haspar(ch1, p) && hist!(ax, f(colvec(ch1, p)); bins=60, normalization=:pdf,
                                color=(CMP1, 0.6), label="dark")
        haspar(ch2, p) && hist!(ax, f(colvec(ch2, p)); bins=60, normalization=:pdf,
                                color=(CMP2, 0.6), label="luminous")
        k == 1 && axislegend(ax, position=:rt)
    end
    return fig
end

# -----------------------------------------------------------------------------
# Model components (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
function star_body()
    mA = PRIORS.M_A
    vars = @eval @variables begin
        mass ~ $mA            # [M⊙]
        flux_G  = 1.0         # the host sets the contrast scale in each band
        flux_Hp = 1.0
    end
    return Body(name="A", variables=vars)
end

function companion_body(A; luminous::Bool)
    mp, ap, ep, tpp, fp = PRIORS.m_b, PRIORS.a_b, PRIORS.e_b, PRIORS.tp_b, PRIORS.flux
    vars = if luminous
        @eval @variables begin
            mass ~ $mp            # [M⊙]
            a ~ $ap               # [AU]
            e ~ $ep
            ω ~ Uniform(0, 2pi)
            i ~ Sine()
            Ω ~ Uniform(0, 2pi)
            tp ~ $tpp             # [MJD]
            flux_G  ~ $fp         # contrast vs host
            flux_Hp ~ $fp
        end
    else
        @eval @variables begin
            mass ~ $mp
            a ~ $ap
            e ~ $ep
            ω ~ Uniform(0, 2pi)
            i ~ Sine()
            Ω ~ Uniform(0, 2pi)
            tp ~ $tpp
            flux_G  = 0.0         # dark to Gaia…
            flux_Hp = 0.0         # …and to Hipparcos
        end
    end
    dbg("  companion 'b': mass ~ LogUniform(0.5, 1000) M_jup, a ~ LogUniform(0.1, 100) AU, ",
        luminous ? "flux_G, flux_Hp ~ U(0, 1)" : "dark (flux 0)")
    return Body(name="b", about=A, variables=vars)
end

"""
    build_pma_system(kind; freeze_epochs=true)

Fresh bodies and HGCA observation for model `kind` ∈ (:dark, :luminous).
Returns `(sys, hgca)`. Call with the working directory where query caches
should go (`run_pma` uses the script folder).
"""
function build_pma_system(kind::Symbol; freeze_epochs::Bool=true, inputs=load_catalog_inputs())
    kind in MODEL_ORDER || error("kind must be :dark or :luminous, got $kind")
    A, b = make_bodies(kind)
    hgca = make_hgca(A, b; freeze_epochs, inputs)
    plx_prior = make_plx_prior(hgca)
    return assemble_system(kind, A, b, hgca, plx_prior), hgca
end

# ---- Catalog inputs ---------------------------------------------------------
"OCTOPMA_CATALOG: \"subset\" (default), \"full\", or a path to a G23H feather."
function catalog_mode()
    s = strip(get(ENV, "OCTOPMA_CATALOG", "subset"))
    l = lowercase(s)
    return l in ("subset", "full") ? l : String(s)
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

"""
    load_catalog_inputs(; outdir=SCRIPT_DIR, mode=catalog_mode())

Keyword inputs for HGCAObs: `(; catalog, forecast_table)` for "subset",
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
        dbg("  subset catalog: ", length(ids), " rows, ", length(Tables_colnames(cat)),
            " columns; HD 91312 is row ", idx, " (HIP ", cat.hip_id[idx], ")")
        dbg(@sprintf("  row: pmra_hip %.3f, pmra_hg %.3f, pmra_dr3 %.3f | pmdec_hip %.3f, pmdec_hg %.3f, pmdec_dr3 %.3f mas/yr",
                     cat.pmra_hip[idx], cat.pmra_hg[idx], cat.pmra_dr3[idx],
                     cat.pmdec_hip[idx], cat.pmdec_hg[idx], cat.pmdec_dr3[idx]))
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
        dbg("  GOST is queried live. Stale folders from interrupted downloads:")
        stale = check_stale_datadeps()
        any(p -> endswith(p, "G23H_Catalog") || endswith(p, "G23H_DR2Transits"), stale) &&
            error("Incomplete G23H DataDep folder(s) found (see ⚠ lines above); delete them " *
                  "and run again, or use OCTOPMA_CATALOG=subset.")
        return (;)
    else
        isfile(mode) || error("OCTOPMA_CATALOG=\"$mode\" is not \"subset\", \"full\", or an existing file")
        dbg(@sprintf("  local G23H catalog: %s (%.2f GB); GOST queried live", mode, filesize(mode) / 1024^3))
        return (; catalog=mode)
    end
end

Tables_colnames(t) = propertynames(t)

"Fresh star and companion bodies for model `kind`."
function make_bodies(kind::Symbol)
    A = star_body()
    b = companion_body(A; luminous = kind === :luminous)
    return A, b
end

"""
HGCA observation. `inputs` comes from `load_catalog_inputs` (catalog /
forecast_table keywords, possibly empty). The Hipparcos IAD DataDep (~332 MB)
downloads on first use in every mode.
"""
function make_hgca(A, b; freeze_epochs::Bool=true, inputs=(;))
    hgca = HGCAObs(;
        gaia_id = GAIA_ID,
        target = A,
        blends = (b,),
        ref = Barycentre,
        freeze_epochs = freeze_epochs,
        inputs...,
    )
    try                                  # debug only; never fail the stage here
        c = hgca.catalog
        g(k) = hasproperty(c, k) ? Float64(getproperty(c, k)) : NaN
        dbg(@sprintf("  HGCA catalog row: ra = %.6f°, dec = %.6f°, plx = %.4f ± %.4f mas",
                     g(:ra), g(:dec), g(:parallax), g(:parallax_error)))
        dbg("  channels: ", join(string.(hgca.table.kind), ", "),
            " | Hipparcos IAD rows: ", length(hgca.hip_table), " | Gaia forecast rows: ",
            length(hgca.gaia_table))
    catch err
        dbg("  (catalog debug print skipped: ", first(sprint(showerror, err), 200), ")")
    end
    return hgca
end

"""
Parallax prior from the Gaia DR3 TAP service via `gaia_plx`; if the query
fails, the same truncated Normal built from the catalog row's parallax.
"""
function make_plx_prior(hgca)
    p = try
        gaia_plx(gaia_id=GAIA_ID)
    catch err
        dbg("  ⚠ gaia_plx failed (", first(sprint(showerror, err), 200),
            "); using the catalog row's parallax instead")
        μ, σ = Float64(hgca.catalog.parallax), Float64(hgca.catalog.parallax_error)
        truncated(Normal(μ, σ), lower=μ - 10σ, upper=μ + 10σ)
    end
    dbg("  plx prior: ", p)
    return p
end

function assemble_system(kind::Symbol, A, b, hgca, plx_prior)
    ra0, dec0 = hgca.catalog.ra, hgca.catalog.dec
    pmra_p, pmdec_p = PRIORS.pmra, PRIORS.pmdec
    refep = Octofitter.meta_gaia_DR3.ref_epoch_mjd
    vars = @eval @variables begin
        plx ~ $plx_prior
        pmra ~ $pmra_p             # barycentre proper motion [mas/yr]
        pmdec ~ $pmdec_p
        ra = $ra0                  # rest of the absolute frame
        dec = $dec0
        rv = 0.0                   # barycentre RV [m/s] for perspective terms
        ref_epoch = $refep
    end
    dbg("  system: pmra ~ U(-237, -37), pmdec ~ U(-98, 102) mas/yr, ref_epoch = MJD ", refep)
    return System(name="HD91312_pma_$(kind)", bodies=[A, b], observations=[hgca],
                  variables=vars)
end

function write_hgca_csv(p, tbl)
    open(p, "w") do io
        println(io, "# HGCA rows loaded by Octofitter HGCAObs for ", TARGET,
                " (Gaia DR3 ", GAIA_ID, "); pm and sigma_pm in mas/yr, epochs in MJD")
        println(io, "kind,epoch,start_epoch,stop_epoch,pm,sigma_pm")
        for j in eachindex(tbl.epoch)
            println(io, join((string(tbl.kind[j]), tbl.epoch[j], tbl.start_epoch[j],
                              tbl.stop_epoch[j], tbl.pm[j], tbl.σ_pm[j]), ","))
        end
    end
    return p
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_pma(; models=(:dark,), n_rounds=10, freeze_epochs=true, prior_z0=false,
              outdir=SCRIPT_DIR, prefix="pma_", dark=true, corner_dark=false,
              show_plots=true)

Load the HGCA for HD 91312, fit each model with Pigeons, plot, compare evidences.
"""
function run_pma(; models=(:dark,),
                   n_rounds::Integer=10,
                   freeze_epochs::Bool=true,
                   prior_z0::Bool=false,
                   outdir::AbstractString=SCRIPT_DIR,
                   prefix::AbstractString="pma_",
                   dark::Bool=true,
                   corner_dark::Bool=false,
                   show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    accept_downloads!()
    print_env_info(outdir, models, n_rounds, freeze_epochs)
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

    modelsd = Dict{Symbol,Any}(); chains = Dict{Symbol,Any}()
    logZ = Dict{Symbol,Float64}(); logZ0 = Dict{Symbol,Float64}()
    data_done = false
    hb = get(ENV, "OCTOPMA_HEARTBEAT", "30")
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

    for kind in models
        tag = string(kind)
        dbg("[$tag] Building system — ", MODEL_LABELS[kind])
        A, b = stage("[$tag] Bodies (A, b)") do
            make_bodies(kind)
        end
        hgca = stage("[$tag] HGCAObs (first run downloads the Hipparcos IAD, ~332 MB; heartbeat every $hb s)") do
            with_heartbeat("HGCAObs"; watch=watchdirs(), outdir) do
                inout(() -> make_hgca(A, b; freeze_epochs, inputs))
            end
        end
        soft_stage("List DataDeps contents") do
            list_datadeps()
        end
        plx_prior = stage("[$tag] Parallax prior (gaia_plx, fallback: catalog row)") do
            with_heartbeat("gaia_plx"; watch=watchdirs(), outdir) do
                inout(() -> make_plx_prior(hgca))
            end
        end
        sys = stage("[$tag] Assemble System") do
            assemble_system(kind, A, b, hgca, plx_prior)
        end
        DEBUG[] && display(sys)

        if !data_done
            tbl = soft_stage("HGCA data table") do
                t = obs_table(hgca)
                DEBUG[] && display(t)
                t
            end
            if tbl !== nothing
                soft_stage("Save HGCA snapshot CSV") do
                    push!(outputs, write_hgca_csv(joinpath(outdir, prefix * "hgca_$(GAIA_ID).csv"), tbl))
                end
                soft_stage("Plot HGCA proper motions with averaging windows") do
                    savefig!(hgca_figure(tbl), "hgca_data.png")
                end
            end
            data_done = true
        end

        model = stage("[$tag] Compile LogDensityModel") do
            Octofitter.LogDensityModel(sys)
        end
        DEBUG[] && display(model)
        modelsd[kind] = model

        stage("[$tag] initialize! (Pathfinder starting points)") do
            Octofitter.initialize!(model)
        end
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
        DEBUG[] && display(chain)          # MCMCChains table: check rhat ≈ 1, ess
        print_summary(chain, MODEL_LABELS[kind])

        if prior_z0
            soft_stage("[$tag] Prior-only run for log Z0") do
                pm = Octofitter.LogDensityModel(Octofitter.prior_only_model(sys, exclude_all=true))
                Octofitter.initialize!(pm)
                _, ptp = octofit_pigeons(pm; n_rounds=n_rounds)
                z0 = Pigeons.stepping_stone(ptp)
                logZ0[kind] = z0
                dbg(@sprintf("  [%s] log Z0 ≈ %.4f  (≈0 means the priors are properly normalised)", tag, z0))
            end
        end

        soft_stage("[$tag] octoplot (proper-motion panels)") do
            savefig!(inout(() -> octoplot(model, chain)), "$(tag)_octoplot.png")
        end
        soft_stage("[$tag] Corner plot") do
            savefig!(inout(() -> corner(model, chain; dark=corner_dark)), "$(tag)_corner.png")
        end
        soft_stage("[$tag] dotplot (mass vs separation)") do
            savefig!(inout(() -> Octofitter.dotplot(model, chain)), "$(tag)_dotplot.png")
        end
        soft_stage("[$tag] dotplot (mass vs period)") do
            savefig!(inout(() -> Octofitter.dotplot(model, chain; mode=:period)),
                     "$(tag)_dotplot_period.png")
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

    if haskey(chains, :dark) && haskey(chains, :luminous)
        soft_stage("Dark vs luminous posterior comparison") do
            savefig!(compare_figure(chains[:dark], chains[:luminous]), "compare_dark_luminous.png")
        end
    end

    # ---- Evidence ------------------------------------------------------------
    if !isempty(logZ)
        lines = String[]
        push!(lines, "Log evidence (stepping stone), freeze_epochs = $(freeze_epochs), n_rounds = $(n_rounds):")
        for k in MODEL_ORDER
            haskey(logZ, k) || continue
            z0s = haskey(logZ0, k) ? @sprintf("%8.4f", logZ0[k]) : "not run (OCTOPMA_PRIOR_Z0=true)"
            push!(lines, @sprintf("    %-46s log(Z/Z0) = %10.3f   log Z0 = %s", MODEL_LABELS[k],
                                  logZ[k], z0s))
        end
        push!(lines, "    (docs page, dark model, freeze_epochs=true: log(Z₁/Z₀) ≈ -24.1, Λ ≈ 9.06)")
        if haskey(logZ, :luminous) && haskey(logZ, :dark)
            lnbf = logZ[:luminous] - logZ[:dark]
            push!(lines, @sprintf("    ln BF(luminous vs dark) = %8.3f  → %s (H_A = luminous)",
                                  lnbf, interpret(lnbf)))
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
    return (; models=modelsd, chains, logZ, logZ0, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default when the REPL has 1 thread)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString="pma_")
    T0[] = time()
    accept_downloads!()                  # inherited by the child
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOPMA_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOPMA_CHILD=1]")
    dbg("  Child output follows. Plots go to files only (not the plot pane).")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOPMA_RELAUNCH\"]=\"false\".")
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
    elseif lowercase(get(ENV, "OCTOPMA_SHOW_PLOTS", "true")) != "false"
        soft_stage("Show $(length(pngs)) saved plot(s) in the plot pane") do
            show_saved_pngs(pngs)
        end
    end
    return (; relaunched=true, success=ok, chains, outputs=new_files)
end

"Plot order for display: data first, then per-model figures, comparison last."
function png_rank(p)
    f = basename(p)
    occursin("hgca_data", f) && return 0
    occursin("compare", f) && return 9
    for (k, key) in enumerate(("octoplot", "corner", "dotplot_period", "dotplot", "marginals"))
        occursin(key, f) && return k
    end
    return 8
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

"Parse OCTOPMA_MODELS (comma-separated) into a tuple of Symbols in canonical order."
function models_from_env()
    s = get(ENV, "OCTOPMA_MODELS", "dark")
    req = Set(Symbol(strip(lowercase(x))) for x in split(s, ',') if !isempty(strip(x)))
    issubset(req, Set(MODEL_ORDER)) ||
        error("OCTOPMA_MODELS must list dark and/or luminous, got \"$s\"")
    return Tuple(k for k in MODEL_ORDER if k in req)
end

println(@sprintf("[PMA] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterPMA

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterPMA.jl`       -> runs
#   - VSCode "Execute active File in REPL"          -> runs; with 1 thread it
#     relaunches in `julia --threads=auto` (OCTOPMA_RELAUNCH=false to disable,
#     OCTOPMA_THREADS to set the count)
#   - ENV["OCTOPMA_AUTORUN"] = "false"; include()   -> loads module only
#   - OCTOPMA_MODELS, OCTOPMA_ROUNDS, OCTOPMA_FREEZE, OCTOPMA_PRIOR_Z0 select what runs
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOPMA_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOPMA_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOPMA_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global pma_result = OctofitterPMA.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOPMA_THREADS", "auto"))
            println("[PMA] Child run finished. `pma_result` has fields: relaunched, success, ",
                    "chains (loaded from FITS), outputs")
        else
            global pma_result = OctofitterPMA.run_pma(
                models=OctofitterPMA.models_from_env(),
                n_rounds=parse(Int, get(ENV, "OCTOPMA_ROUNDS", "10")),
                freeze_epochs=lowercase(get(ENV, "OCTOPMA_FREEZE", "true")) != "false",
                prior_z0=lowercase(get(ENV, "OCTOPMA_PRIOR_Z0", "false")) != "false",
                show_plots=!is_child)
            is_child || println("[PMA] Result stored in `pma_result` (fields: models, chains, ",
                "logZ, logZ0, outputs)")
        end
    end
end

nothing
