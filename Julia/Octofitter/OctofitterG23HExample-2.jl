#=
================================================================================
 OctofitterG23HExample.jl
================================================================================
 Module wrapping the Octofitter.jl v9 page "Full G23H Example Script":
   https://sefffal.github.io/Octofitter.jl/dev/g23h-example/

 The page is a command-line script (ArgParse) for joint Gaia–Hipparcos (G23H)
 astrometry, optionally with radial velocities. This module keeps its model
 and workflow and adds the usual scaffolding:
   - G23HObs from `--hip` or `--gaia`, restricted to the chosen channels
     (`--obs-types`, applied with likeobj_from_epoch_subset as on the page),
     optional Gaia DR3 RV variability channel (`--gaia-rv`).
   - 1..N dark companions (`--n-planets`), all coplanar (shared system i, Ω),
     a ~ LogUniform(0.001, 1024) AU, mass′ ~ LogUniform(0.01 M_jup, M_host),
     M0 at the mean Gaia epoch. The page marginalises over the number of
     companions: n_planets ~ truncated(NegativeBinomial(), upper=N) and
     mass = mass′ · (n_planets ≥ k). With N > 1: OrbitOrderPrior and
     NonCrossingPrior.
   - RVs from DACE (`--dace-rv`; PythonCall + CondaPkg + dace-query) and/or
     RDB files (`--rdb-rv`, repeatable), per-instrument MAD clipping, one
     RadialVelocityObs per instrument with offset, jitter and a shared linear
     trend `m`; optional quasi-periodic Celerite GP (`--gp`).
   - Pigeons: 32 chains, SliceSampler, no variational leg, multithreaded;
     round 1 then increment_n_rounds!(pt, 1) per round, with a FITS
     checkpoint (tagged with the log-evidence ratio) and the Pigeons summary
     CSV after every round, as on the page.
   - Plots: catalog proper motions, RV data, per-round convergence,
     octoplot, light-theme corner, dotplot (separation and period), the
     page's mass–sma pairplot plus a log₁₀ version for draws with a companion,
     n_planets prior vs posterior, rvplot (when there are RVs).

 Two page details, handled explicitly:
   * `--circular` is parsed on the page but never used. Here it fixes e = 0
     and ω = 0 for every companion.
   * The page's clip width is 1.4826 · StatsBase.mad(rv), but StatsBase.mad
     is already normalised (×1.4826), so "3σ" is really ≈ 4.45 normalised-MAD
     σ. Kept as on the page (OCTOG23X_CLIP sets the multiplier); the log says so.
   * DACE/RDB epochs are rjd = BJD − 2 400 000; the page uses them as MJD
     (0.5 d off). Here they are converted: MJD = rjd − 0.5.

 Default target: Gaia DR3 756291174721509376 = HIP 51658 = HD 91312 (the
 page's own `--gaia` example), host mass 1.61 M⊙ (as in the PMA tutorial).
 It is a row of the docs' 4-row G23H subset and its GOST forecast is cached,
 so the default run needs no large downloads. HIP 384 (Gaia DR3
 2738776816458107136, 1.0 M⊙) works offline too.

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `g23hx_result`)
   Shell:    julia --threads=auto OctofitterG23HExample.jl --gaia 756291174721509376 \
                   --host-mass 1.61 --n-rounds 8
             (the page's arguments are accepted and mapped to the ENV settings)
   Library:  ENV["OCTOG23X_AUTORUN"] = "false"; include("OctofitterG23HExample.jl")
             res = OctofitterG23HExample.run_g23hx()

 Threads: Pigeons runs multithreaded. In a 1-thread REPL the run is relaunched
 in a child `julia --threads=auto` (output streamed here; the final chain is
 loaded back and the saved plots are shown in the plot pane afterwards).
 OCTOG23X_RELAUNCH=false disables this; OCTOG23X_THREADS sets the count.

 Settings (ENV, also passed to a relaunched child) ↔ page arguments:
   OCTOG23X_GAIA        --gaia        Gaia DR3 source id (default 756291174721509376
                                      when neither id is given)
   OCTOG23X_HIP         --hip         Hipparcos id (not together with GAIA)
   OCTOG23X_HOST_MASS   --host-mass   [M⊙]; defaults known for the subset targets
   OCTOG23X_N_PLANETS   --n-planets   "1" (default)
   OCTOG23X_ROUNDS      --n-rounds    "8" (default here; the page's default is 12,
                                      which is ~16× longer)
   OCTOG23X_OBS_TYPES   --obs-types   e.g. "ra_hip,dec_hip,ra_hg,dec_hg,ueva_dr3"
   OCTOG23X_GAIA_RV     --gaia-rv     "false" (default)
   OCTOG23X_DACE        --dace-rv     "false" (default); installs PythonCall/CondaPkg
   OCTOG23X_RDB         --rdb-rv      comma-separated RDB paths
   OCTOG23X_GP          --gp          "false" (default)
   OCTOG23X_CIRCULAR    --circular    "false" (default)
   Extra (not on the page):
   OCTOG23X_CHAINS      --n-chains    "32" (default, page)
   OCTOG23X_FREEZE      --freeze      "true" (default here: the page's speed tip);
                                      "false" is the page script's own setting
                                      (G23HObs default) — sampled epoch selection, slow
   OCTOG23X_CATALOG     --catalog     "subset" (default) | "full" | "/path/G23H-v1.0.feather"
   OCTOG23X_INIT        "page" (default: guess_starting_position, 10 000 draws)
                        | "initialize" (Octofitter.initialize!)
   OCTOG23X_CLIP        "3.0" — RV clip multiplier (page semantics, see above)
   OCTOG23X_VERBOSITY   "4" (page) — LogDensityModel verbosity
   OCTOG23X_SHOW_PLOTS  "true" — show the child's PNGs here
   OCTOG23X_HEARTBEAT   "30" — seconds between "still waiting" lines
   OCTOG23X_ACCEPT_DOWNLOADS "true" — sets DATADEPS_ALWAYS_ACCEPT

 Catalog data (as in OctofitterPMA / OctofitterG23H): "subset" uses the docs'
 4-row test catalog `G23H-test-subset.feather` (Gaia DR3 2738776816458107136 =
 HIP 384, 756291174721509376 = HIP 51658, 5164707970261890560 = HIP 16537,
 6166183842771027328 = HIP 65808) and, when cached, the GOST forecast for the
 target. The two cached forecasts are HIP 384 and HIP 51658; the other two
 rows query GOST live (network). "full" uses the ~14 GB G23H DataDep.

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOG23X_ENV_MODE=local uses ./octofitter_v9_env). Needs Octofitter
 and OctofitterRadialVelocity v9, CairoMakie, PairPlots, Distributions,
 Pigeons, Arrow, CSV, FiniteDiff, DifferentiationInterface (+ PythonCall,
 CondaPkg when DACE is requested).

 Outputs (prefix "g23hx_") go to the directory containing this file; the
 per-round checkpoints go to its subfolder "g23hx_checkpoints".

 Changelog
 ---------
 v1.0  2026-09-25  Initial version on the OctofitterDR4Sim v1.0.1 /
                   OctofitterG23H v1.0.1 scaffolding ([G23X +t] debug stages,
                   soft optional stages, env bootstrap with v9 guards,
                   @__DIR__ outputs, dark theme with light corner plots,
                   threaded relaunch with exit-code-only failure report and
                   PNGs shown in the parent's plot pane, heartbeat and
                   stale-DataDep checks, docs' offline catalog subset + GOST
                   forecast, @eval-built @variables). Adds: the page's CLI
                   arguments mapped to ENV; channel selection; DACE/RDB RV
                   loading with MAD clipping; optional GP; n_planets
                   marginalisation with prior/posterior odds; round-by-round
                   sampling with checkpoints and a convergence plot.
 v1.0.1 2026-09-25 First run (4.7 min, 4 threads, 8 rounds → 256 samples):
                   HIP 51658, all 12 channels, log(Z₁/Z₀) settled at −476 ± 3
                   from round 5, Λ = 12.8 and still rising slowly, min(α)
                   0.09 at round 8; n_planets = 1 in 256/256 samples; b ≈ 0.9
                   (0.3–1.4) M⊙ at 77 (55–96) AU. Fix: the "mixing is suspect"
                   note was printed regardless of min(α); min(α) is now read
                   from Pigeons.swap_prs and shown per round, in the summary
                   and in rounds.csv, and the note appears only when < 1e-3.
================================================================================
=#

# -----------------------------------------------------------------------------
# The page's command-line arguments -> ENV (before the bootstrap, because
# --dace-rv changes which packages are needed). No packages used here.
# -----------------------------------------------------------------------------
let cli = copy(ARGS),
    valued = Dict("--hip" => "HIP", "--gaia" => "GAIA", "--host-mass" => "HOST_MASS",
                  "--n-planets" => "N_PLANETS", "--n-rounds" => "ROUNDS",
                  "--obs-types" => "OBS_TYPES", "--n-chains" => "CHAINS",
                  "--catalog" => "CATALOG", "--freeze" => "FREEZE", "--init" => "INIT"),
    switches = Dict("--gaia-rv" => "GAIA_RV", "--dace-rv" => "DACE", "--gp" => "GP",
                    "--circular" => "CIRCULAR")
    rdb = String[]
    set = String[]
    i = 1
    while i <= length(cli)
        a = cli[i]
        k, v = occursin('=', a) ? (String.(split(a, '='; limit=2))...,) : (a, nothing)
        if haskey(switches, k)
            ENV["OCTOG23X_" * switches[k]] = "true"
            push!(set, k)
            i += 1
        elseif haskey(valued, k) || k == "--rdb-rv"
            if v === nothing
                i < length(cli) || error("[G23X] argument $k needs a value")
                v = cli[i+1]
                i += 2
            else
                i += 1
            end
            k == "--rdb-rv" ? push!(rdb, v) : (ENV["OCTOG23X_" * valued[k]] = v)
            push!(set, "$k $v")
        else
            error("[G23X] unknown argument \"$a\". Accepted: " *
                  join(sort(vcat(collect(keys(valued)), collect(keys(switches)), ["--rdb-rv"])), ", "))
        end
    end
    isempty(rdb) || (ENV["OCTOG23X_RDB"] = join(rdb, ","))
    isempty(set) || println("[G23X] command-line arguments mapped to OCTOG23X_* settings: ",
                            join(set, " | "))
end

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let dace = lowercase(get(ENV, "OCTOG23X_DACE", "false")) == "true",
    required = ("Octofitter", "OctofitterRadialVelocity", "CairoMakie", "PairPlots",
                "Distributions", "Pigeons", "Arrow", "CSV", "FiniteDiff",
                "DifferentiationInterface",
                (dace ? ("PythonCall", "CondaPkg") : ())...),
    min_version = Dict("Octofitter" => v"9", "OctofitterRadialVelocity" => v"9"),
    mode = lowercase(get(ENV, "OCTOG23X_ENV_MODE", "temp"))

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
            println("[G23X] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[G23X] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[G23X] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[G23X] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[G23X] Existing v9 environment found at ", Base.active_project())
        else
            Pkg.add(spec.(collect(required)))
        end
        println("[G23X] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[G23X] Loading Octofitter, OctofitterRadialVelocity, Pigeons, CairoMakie, PairPlots, ",
        "Distributions, Arrow, CSV, FiniteDiff (first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterG23HExample

const LOAD_T0 = time()

using Octofitter
using OctofitterRadialVelocity
import OctofitterRadialVelocity.Celerite
using CairoMakie
using PairPlots
using Distributions
using Pigeons
import Arrow
import CSV
import Downloads
import DelimitedFiles
import Logging
import Random
import Statistics
import FiniteDiff
using DifferentiationInterface: AutoFiniteDiff
using Printf

export run_g23hx, set_plot_theme!

const VERSION_STRING = "1.0.1"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterG23HExample needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const PREFIX = "g23hx_"

# Offline inputs used by the docs build (pinned to a known Octofitter commit)
const OCTO_COMMIT = "8c6daaaf5292b6cbdb0aa4f4b1e9a1a5f0bd21f3"
const OCTO_RAW = "https://raw.githubusercontent.com/sefffal/Octofitter.jl/$(OCTO_COMMIT)/"
const SUBSET_FILE = "G23H-test-subset.feather"
const SUBSET_URL  = OCTO_RAW * "test/" * SUBSET_FILE

"Cached GOST forecasts in the repo's docs/src (Gaia DR3 id => file name)."
const GOST_FILES = Dict(
    756291174721509376  => "GOST-158.30707896392835-40.42555422701387-dr3.csv",   # HIP 51658
    2738776816458107136 => "GOST-1.1927097109938027-1.5368044203832403-dr3.csv",  # HIP 384
)

"Host masses [M⊙] used by the other tutorials for the subset targets."
const KNOWN_HOSTS = Dict(
    756291174721509376  => (1.61, "HD 91312 = HIP 51658 (PMA tutorial)"),
    2738776816458107136 => (1.0,  "HIP 384 (G23H tutorial)"),
)

const DEFAULT_GAIA = 756291174721509376          # the page's --gaia example

"The page's default channel list (plus :rv_dr3 with --gaia-rv)."
const DEFAULT_OBS = [:iad_hip, :ra_hip, :dec_hip, :ra_hg, :dec_hg, :ra_dr2, :dec_dr2,
                     :ra_dr32, :dec_dr32, :ra_dr3, :dec_dr3, :ueva_dr3]
const PM_KINDS = r"^(ra|dec)_(hip|hg|dr2|dr32|dr3)$"

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterG23HExample.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[G23X +%7.1fs] ", time() - T0[]), msg...)
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

envbool(k, default) = lowercase(strip(get(ENV, k, default))) in ("true", "1", "yes")
envint(k, default) = parse(Int, strip(get(ENV, k, default)))
envfloat(k, default) = parse(Float64, strip(get(ENV, k, default)))
function envopt(k)
    s = strip(get(ENV, k, ""))
    return isempty(s) ? nothing : String(s)
end

"Let DataDeps download without an interactive y/n prompt (unless disabled)."
function accept_downloads!()
    lowercase(get(ENV, "OCTOG23X_ACCEPT_DOWNLOADS", "true")) != "false" &&
        (ENV["DATADEPS_ALWAYS_ACCEPT"] = "true")
    return nothing
end

# ---- Configuration -------------------------------------------------------------
"""
    config_from_env()

The page's arguments (plus a few extras), read from OCTOG23X_* settings.
"""
function config_from_env()
    hip  = envopt("OCTOG23X_HIP")
    gaia = envopt("OCTOG23X_GAIA")
    hip !== nothing && gaia !== nothing && error("Cannot specify both --hip and --gaia " *
        "(OCTOG23X_HIP=$hip, OCTOG23X_GAIA=$gaia)")
    hip === nothing && gaia === nothing && (gaia = string(DEFAULT_GAIA))
    hm = envopt("OCTOG23X_HOST_MASS")
    obs_types = envopt("OCTOG23X_OBS_TYPES")
    rdb = envopt("OCTOG23X_RDB")
    return (;
        hip        = hip === nothing ? nothing : parse(Int, hip),
        gaia       = gaia === nothing ? nothing : parse(Int64, gaia),
        host_mass  = hm === nothing ? nothing : parse(Float64, hm),
        n_planets  = envint("OCTOG23X_N_PLANETS", "1"),
        n_rounds   = envint("OCTOG23X_ROUNDS", "8"),
        obs_types  = obs_types === nothing ? nothing :
                     [Symbol(strip(s)) for s in split(obs_types, ',') if !isempty(strip(s))],
        gaia_rv    = envbool("OCTOG23X_GAIA_RV", "false"),
        dace       = envbool("OCTOG23X_DACE", "false"),
        rdb        = rdb === nothing ? String[] :
                     [String(strip(s)) for s in split(rdb, ',') if !isempty(strip(s))],
        gp         = envbool("OCTOG23X_GP", "false"),
        circular   = envbool("OCTOG23X_CIRCULAR", "false"),
        n_chains   = envint("OCTOG23X_CHAINS", "32"),
        freeze     = envbool("OCTOG23X_FREEZE", "true"),
        init       = lowercase(get(ENV, "OCTOG23X_INIT", "page")),
        clip       = envfloat("OCTOG23X_CLIP", "3.0"),
        verbosity  = envint("OCTOG23X_VERBOSITY", "4"),
    )
end

# ---- DataDeps folders, heartbeat ---------------------------------------------
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

"OCTOG23X_CATALOG: \"subset\" (default), \"full\", or a path to a G23H feather."
function catalog_mode()
    s = strip(get(ENV, "OCTOG23X_CATALOG", "subset"))
    l = lowercase(s)
    return l in ("subset", "full") ? l : String(s)
end

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
                "OCTOG23X_CATALOG=full:  rm(\"", p, "\"; recursive=true))")
        else
            push!(stale, p)
            dbg("  ⚠ incomplete DataDep folder (interrupted download?): ", p)
            dbg("    delete it so DataDeps downloads again:  rm(\"", p, "\"; recursive=true)")
        end
    end
    isempty(stale) && dbg("  no incomplete DataDep folders that this run uses")
    return stale
end

function with_heartbeat(f, what::AbstractString;
                        every::Real=parse(Float64, get(ENV, "OCTOG23X_HEARTBEAT", "30")),
                        watch=(datadeps_dirs()...,), outdir::AbstractString=SCRIPT_DIR)
    done = Threads.Atomic{Bool}(false)
    t0 = time()
    sizes() = Dict(d => dir_bytes(d) for d in watch)
    base = Ref{Any}(nothing)
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

function print_env_info(cfg, outdir)
    DEBUG[] || return nothing
    dbg("OctofitterG23HExample v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | OctofitterRadialVelocity ", something(pkgversion(OctofitterRadialVelocity), "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("target = ", cfg.hip !== nothing ? "HIP $(cfg.hip)" : "Gaia DR3 $(cfg.gaia)",
        " | host mass = ", cfg.host_mass === nothing ? "(from the known-target table)" :
                           "$(cfg.host_mass) M⊙",
        " | n_planets ≤ ", cfg.n_planets, cfg.circular ? " (circular: e = 0, ω = 0)" : "")
    dbg("channels = ", cfg.obs_types === nothing ? "page default (all 12)" : join(cfg.obs_types, ","),
        " | gaia_rv = ", cfg.gaia_rv, " | freeze_epochs = ", cfg.freeze,
        cfg.freeze ? " (page's speed tip; the script's own default is false)" : " (sampled; slow)")
    dbg("RV: DACE = ", cfg.dace, " | RDB files = ", isempty(cfg.rdb) ? "none" : join(cfg.rdb, ", "),
        " | GP = ", cfg.gp, " | clip = ", cfg.clip, " × 1.4826 × StatsBase.mad")
    dbg("Pigeons: n_rounds = ", cfg.n_rounds, " (last round ", 2^cfg.n_rounds, " samples; page default 12)",
        " | n_chains = ", cfg.n_chains, " | SliceSampler | no variational leg | init = ", cfg.init)
    dbg("catalog mode = ", catalog_mode(),
        catalog_mode() == "subset" ? " (docs' 4-row G23H subset + cached GOST forecast when available)" :
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
qline(x) = (y = filter(isfinite, x); isempty(y) ? (NaN, NaN, NaN) :
                                     Statistics.quantile(y, (0.16, 0.5, 0.84)))

"Chain columns whose name ends with `suffix`."
findcols(chain, suffix::AbstractString) = [n for n in names(chain) if endswith(string(n), suffix)]

function fold_axial(x)
    c = atan(Statistics.mean(sin.(2 .* x)), Statistics.mean(cos.(2 .* x))) / 2
    return mod.(x .- c .+ pi / 2, pi) .- pi / 2 .+ c
end

function qprint(label, x)
    q = qline(x)
    dbg(@sprintf("    %-30s %12.4f  [%12.4f, %12.4f]", label, q[2], q[1], q[3]))
end

"""
Posterior summary. Companion quantities are conditioned on the companion being
present (n_planets ≥ k); P is Kepler's law with M_host + m (inner companions
ignored), since the page parametrises by a.
"""
function print_summary(chain, cfg, planet_names, host_mass)
    DEBUG[] || return nothing
    n = size(chain, 1) * size(chain, 3)
    dbg("Posterior median [16th, 84th] (", n, " samples):")
    for (lab, p, f) in (("plx [mas]", :plx, identity), ("pmra [mas/yr]", :pmra, identity),
                        ("pmdec [mas/yr]", :pmdec, identity),
                        ("i (shared) [deg]", :i, x -> rad2deg.(x)),
                        ("Ω (shared) [deg] (folded)", :Ω, x -> rad2deg.(fold_axial(x))),
                        ("RV trend m [m/s/d]", :m, identity),
                        ("GP B", :gp_B, identity), ("GP C", :gp_C, identity),
                        ("GP L [d]", :gp_L, identity), ("GP Prot [d]", :gp_Prot, identity))
        haspar(chain, p) && qprint(lab, f(colvec(chain, p)))
    end
    npl = haspar(chain, :n_planets) ? colvec(chain, :n_planets) : nothing
    for (k, pn) in enumerate(planet_names)
        s = string(pn)
        haspar(chain, Symbol(s, "_a")) || continue
        mask = npl === nothing ? trues(n) : (npl .>= k)
        nk = count(mask)
        dbg(@sprintf("  companion %s — present (n_planets ≥ %d) in %d of %d samples (%.1f%%):",
                     s, k, nk, n, 100 * nk / n))
        nk == 0 && continue
        a = colvec(chain, Symbol(s, "_a"))[mask]
        m = colvec(chain, Symbol(s, "_mass"))[mask]
        qprint("$s a [AU]", a)
        qprint("$s mass [M_jup]", m ./ mjup)
        qprint("$s log₁₀ mass [M_jup]", log10.(m ./ mjup))
        qprint("$s P ≈ √(a³/(M_host+m)) [yr]", sqrt.(a .^ 3 ./ (host_mass .+ m)))
        haspar(chain, Symbol(s, "_e")) && qprint("$s e", colvec(chain, Symbol(s, "_e"))[mask])
        mm = m ./ mjup
        dbg(@sprintf("    %s mass: %.1f%% < 13 M_jup (planet), %.1f%% 13–80 (brown dwarf), %.1f%% > 80 (star)",
                     s, 100 * Statistics.mean(mm .< 13), 100 * Statistics.mean(13 .<= mm .<= 80),
                     100 * Statistics.mean(mm .> 80)))
    end
    for suf in ("_σ_AL", "_σ_att", "_σ_calib", "_offset", "_jitter")
        for c in findcols(chain, suf)
            qprint(string(c), colvec(chain, c))
        end
    end
    return nothing
end

"""
Prior vs posterior probability of each companion count, and the Bayes factor
for k vs k−1 companions (posterior odds / prior odds).
"""
function n_planets_table(chain, N; min_α=NaN)
    haspar(chain, :n_planets) || return nothing
    x = round.(Int, colvec(chain, :n_planets))
    n = length(x)
    prior = truncated(NegativeBinomial(), upper=N)
    pr = [pdf(prior, k) for k in 0:N]
    po = [count(==(k), x) / n for k in 0:N]
    lines = String[]
    push!(lines, "Companion count: prior truncated(NegativeBinomial(), upper=$N) vs posterior ($n samples)")
    for k in 0:N
        push!(lines, @sprintf("    n_planets = %d   prior %.3f   posterior %.4f  (%d samples)",
                              k, pr[k+1], po[k+1], count(==(k), x)))
    end
    for k in 1:N
        pk, pk1 = po[k+1], po[k]
        prior_odds = pr[k+1] / pr[k]
        s = if pk > 0 && pk1 > 0
            bf = (pk / pk1) / prior_odds
            @sprintf("    BF(%d vs %d) = %.3g  (log₁₀ %.2f)", k, k - 1, bf, log10(bf))
        elseif pk1 == 0 && pk > 0
            lb = (pk / (1 / n)) / prior_odds
            @sprintf("    BF(%d vs %d) > %.3g  (no samples at n_planets = %d)", k, k - 1, lb, k - 1)
        elseif pk == 0 && pk1 > 0
            ub = ((1 / n) / pk1) / prior_odds
            @sprintf("    BF(%d vs %d) < %.3g  (no samples at n_planets = %d)", k, k - 1, ub, k)
        else
            @sprintf("    BF(%d vs %d): no samples at either count", k, k - 1)
        end
        push!(lines, s)
    end
    push!(lines, @sprintf("    (counts from %d samples: a bound like \"> %d\" is the resolution limit, not a measurement)",
                          n, round(Int, 2n)))
    if isfinite(min_α) && min_α < 1e-3
        push!(lines, @sprintf("    ⚠ final min(α) = %.2g: the chains barely swap, so the count's mixing is suspect — add rounds/chains",
                              min_α))
    elseif isfinite(min_α)
        push!(lines, @sprintf("    final min(α) = %.3f: the temperature ladder is connected", min_α))
    end
    return (; lines, prior=pr, post=po)
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

corner(model, chain; dark::Bool=false, small::Bool=true) =
    dark ? octocorner(model, chain; small=small) :
           with_theme(() -> octocorner(model, chain; small=small), Theme())

const CAT_COLORS = Dict("hip" => "#56B4E9", "hg" => "#E69F00", "dr2" => "#CC79A7",
                        "dr32" => "#F0E442", "dr3" => "#009E73")
const CAT_NAMES  = Dict("hip" => "Hipparcos", "hg" => "Hipparcos–Gaia (long-term)",
                        "dr2" => "Gaia DR2", "dr32" => "DR3−DR2 (scaled Δpos)",
                        "dr3" => "Gaia DR3")
const CMP1 = "#0072B2"   # PairPlots series 1
const CMP2 = "#E69F00"   # PairPlots series 2

function split_kind(k)
    s = string(k)
    i = findfirst('_', s)
    i === nothing && return (s, s)
    return (s[1:i-1], s[i+1:end])
end

pm_rows(tbl) = findall(k -> occursin(PM_KINDS, string(k)), tbl.kind)

function pm_figure(tbl, target)
    rows = pm_rows(tbl)
    fig = Figure(size=(1150, 480))
    for (col, (axis, ylab)) in enumerate((("ra", "μ_α* [mas/yr]"), ("dec", "μ_δ [mas/yr]")))
        ax = Axis(fig[1, col], xlabel="epoch [MJD]", ylabel=ylab,
                  title="$(target) catalog proper motions — $(axis == "ra" ? "RA" : "Dec")")
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
        col == 1 && !isempty(rows) && axislegend(ax, position=:rt, labelsize=11, unique=true)
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

"RV data after clipping, one colour per instrument (the page's per-source figure)."
function rv_data_figure(d, insts, title)
    fig = Figure(size=(1100, 440))
    ax = Axis(fig[1, 1], xlabel="epoch [MJD]", ylabel="RV − instrument median [m/s]", title=title)
    cols = CairoMakie.Makie.wong_colors()
    for (j, inst) in enumerate(insts)
        m = d.ins_name .== inst
        c = cols[mod1(j, length(cols))]
        errorbars!(ax, d.epoch[m], d.rv[m], d.rv_err[m]; color=(c, 0.7))
        scatter!(ax, d.epoch[m], d.rv[m]; color=c, markersize=6, label="$inst (n = $(count(m)))")
    end
    axislegend(ax, position=:rt, labelsize=11)
    return fig
end

"Per-round log-evidence ratio and Λ (the page's advice: watch them settle)."
function rounds_figure(rows)
    fig = Figure(size=(1100, 420))
    r = [x.round for x in rows]
    for (k, (key, lab)) in enumerate(((:logZ, "log(Z₁/Z₀) (stepping stone)"), (:Λ, "Λ (global barrier)")))
        ax = Axis(fig[1, k], xlabel="round", ylabel=lab, xticks=r)
        y = [getproperty(x, key) for x in rows]
        scatterlines!(ax, r, y; color=k == 1 ? CMP1 : CMP2, markersize=10)
    end
    Label(fig[0, 1:2], "Convergence by round: both should level off", fontsize=15)
    return fig
end

"The page's mass–sma pairplot (all samples; b_mass = 0 where b is absent)."
function mass_sma_pairplot(chain; pn="b")
    data = (; a = colvec(chain, Symbol(pn, "_a")), mass = colvec(chain, Symbol(pn, "_mass")) ./ mjup)
    return with_theme(Theme()) do
        pairplot(
            data => (
                PairPlots.Scatter(markersize=4),
                PairPlots.MarginHist(),
                PairPlots.MarginQuantileText(),
            ),
            labels = Dict(:a => "semi-major axis [au]", :mass => "mass [Mⱼᵤₚ]"),
        )
    end
end

"log₁₀ version restricted to samples where the companion is present."
function mass_sma_pairplot_log(chain; pn="b", k=1)
    mask = haspar(chain, :n_planets) ? colvec(chain, :n_planets) .>= k :
                                       trues(size(chain, 1) * size(chain, 3))
    count(mask) >= 10 || error("only $(count(mask)) samples with companion $pn present")
    data = (; loga = log10.(colvec(chain, Symbol(pn, "_a"))[mask]),
              logm = log10.(colvec(chain, Symbol(pn, "_mass"))[mask] ./ mjup))
    return with_theme(Theme()) do
        pairplot(
            data => (
                PairPlots.Scatter(markersize=4),
                PairPlots.MarginHist(),
                PairPlots.MarginQuantileText(),
            ),
            labels = Dict(:loga => "log₁₀ semi-major axis [au]", :logm => "log₁₀ mass [Mⱼᵤₚ]"),
        )
    end
end

function n_planets_figure(tab, N)
    fig = Figure(size=(700, 420))
    ax = Axis(fig[1, 1], xlabel="n_planets", ylabel="probability", xticks=0:N,
              title="Companion count: prior vs posterior")
    k = collect(0:N)
    barplot!(ax, k .- 0.18, tab.prior; width=0.34, color=(CMP2, 0.8), label="prior")
    barplot!(ax, k .+ 0.18, tab.post;  width=0.34, color=(CMP1, 0.9), label="posterior")
    ylims!(ax, 0, 1.05)
    axislegend(ax, position=:rt)
    return fig
end

# -----------------------------------------------------------------------------
# Catalog inputs
# -----------------------------------------------------------------------------
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

finite_hip(h) = !ismissing(h) && h isa Real && isfinite(h) && h > 0

"""
    load_catalog_inputs(cfg; outdir)

Keyword inputs for G23HObs plus the resolved Gaia DR3 id (or `nothing` in full
mode with --hip, resolved later by G23HObs).
"""
function load_catalog_inputs(cfg; outdir::AbstractString=SCRIPT_DIR, mode=catalog_mode())
    if mode == "subset"
        cpath = fetch_once(SUBSET_URL, joinpath(outdir, SUBSET_FILE))
        cat = Arrow.Table(cpath)
        ids = collect(cat.gaia_source_id)
        hips = collect(cat.hip_id)
        dbg("  subset catalog rows (", length(propertynames(cat)), " columns):")
        for (j, g) in enumerate(ids)
            dbg("    Gaia DR3 ", g, " | HIP ", finite_hip(hips[j]) ? round(Int, hips[j]) : "—",
                " | cached GOST: ", haskey(GOST_FILES, g) ? "yes" : "no (live query)",
                haskey(KNOWN_HOSTS, g) ? " | host $(KNOWN_HOSTS[g][1]) M⊙" : "")
        end
        idx = cfg.hip !== nothing ?
              findfirst(h -> finite_hip(h) && h == cfg.hip, hips) :
              findfirst(==(cfg.gaia), ids)
        idx === nothing && error((cfg.hip !== nothing ? "HIP $(cfg.hip)" : "Gaia DR3 $(cfg.gaia)") *
            " is not in the 4-row subset catalog (listed above). Use one of those, or " *
            "OCTOG23X_CATALOG=full (14 GB download) / a local G23H feather path.")
        gid = ids[idx]
        dbg(@sprintf("  target row %d: Gaia DR3 %d, plx %.4f ± %.4f mas, pmra_dr3 %.3f, pmdec_dr3 %.3f mas/yr",
                     idx, gid, cat.parallax[idx], cat.parallax_error[idx],
                     cat.pmra_dr3[idx], cat.pmdec_dr3[idx]))
        gfile = get(GOST_FILES, gid, nothing)
        if gfile === nothing
            dbg("  no cached GOST forecast for this source: G23HObs queries GOST live (needs network)")
            return (; kw=(; catalog=cat), gaia_id=gid)
        end
        gpath = fetch_once(OCTO_RAW * "docs/src/" * gfile, joinpath(outdir, gfile))
        gost = CSV.read(gpath, Table; normalizenames=true)
        forecast = Table(
            epoch = Octofitter.jd2mjd.(gost.ObservationTimeAtBarycentre_BarycentricJulianDateInTCB_),
            scanAngle_rad = gost.scanAngle_rad_,
            parallaxFactorAlongScan = gost.parallaxFactorAlongScan,
        )
        dbg(@sprintf("  GOST forecast: %d transits, MJD %.0f–%.0f", length(forecast.epoch),
                     minimum(forecast.epoch), maximum(forecast.epoch)))
        return (; kw=(; catalog=cat, forecast_table=forecast), gaia_id=gid)
    elseif mode == "full"
        dbg("  full mode: first use downloads the ~14 GB G23H catalog and ~300 MB DR2 sidecar;")
        dbg("  GOST is queried live.")
        stale = check_stale_datadeps()
        any(p -> occursin("G23H_", p), stale) &&
            error("Incomplete G23H DataDep folder(s) found (see ⚠ lines above); delete them " *
                  "and run again, or use OCTOG23X_CATALOG=subset.")
        return (; kw=(;), gaia_id=cfg.gaia)
    else
        isfile(mode) || error("OCTOG23X_CATALOG=\"$mode\" is not \"subset\", \"full\", or an existing file")
        dbg(@sprintf("  local G23H catalog: %s (%.2f GB); GOST queried live", mode, filesize(mode) / 1024^3))
        return (; kw=(; catalog=mode), gaia_id=cfg.gaia)
    end
end

# -----------------------------------------------------------------------------
# G23H observation (page section "CREATE G23H OBSERVATION OBJECT")
# -----------------------------------------------------------------------------
"""
The page's G23HObs: `hip_id` or `gaia_id`, target :A, blends = the companion
names, ref = Barycentre, include_rv = --gaia-rv.
"""
function make_g23h(cfg, planet_names, kw)
    common = (; target=:A, blends=Tuple(planet_names), ref=Barycentre,
                include_rv=cfg.gaia_rv, freeze_epochs=cfg.freeze, kw...)
    obs = cfg.hip !== nothing ? G23HObs(; hip_id=cfg.hip, common...) :
                                G23HObs(; gaia_id=cfg.gaia, common...)
    c = obs.catalog
    sysname′ = finite_hip(c.hip_id) ? "HIP$(round(Int, c.hip_id))" : "GDR3$(c.gaia_source_id)"
    return obs, sysname′
end

"The page's channel filter via likeobj_from_epoch_subset."
function select_channels(obs, cfg)
    kinds = Symbol.(obs.table.kind)
    want = cfg.obs_types === nothing ? copy(DEFAULT_OBS) : copy(cfg.obs_types)
    cfg.gaia_rv && !(:rv_dr3 in want) && push!(want, :rv_dr3)
    avail = unique(kinds)
    missing_ = setdiff(want, avail)
    isempty(missing_) || dbg("  requested but not available for this source: ", join(missing_, ", "))
    dropped = setdiff(avail, want)
    isempty(dropped) || dbg("  available but not selected: ", join(dropped, ", "))
    idx = findall(in(want), kinds)
    isempty(idx) && error("No channel left after --obs-types filtering. Available: " * join(avail, ","))
    sub = Octofitter.likeobj_from_epoch_subset(obs, idx)
    dbg("  channels kept (", length(idx), " of ", length(kinds), " rows): ",
        join(string.(sub.table.kind), ", "))
    return sub
end

function describe_obs(obs)
    try
        c = obs.catalog
        g(k) = hasproperty(c, k) && !ismissing(getproperty(c, k)) ? Float64(getproperty(c, k)) : NaN
        dbg(@sprintf("  catalog row: ra %.6f°, dec %.6f°, plx %.4f ± %.4f mas, RV %.3f km/s, RUWE %.3f",
                     g(:ra), g(:dec), g(:parallax), g(:parallax_error), g(:radial_velocity), g(:ruwe_dr3)))
        dbg("  Hipparcos IAD rows: ", length(obs.hip_table), " | Gaia forecast pool: ",
            length(obs.gaia_table), " | observation name \"", obs.name, "\"")
    catch err
        dbg("  (observation debug print skipped: ", first(sprint(showerror, err), 200), ")")
    end
end

# -----------------------------------------------------------------------------
# RV data (page sections "RV HELPER FUNCTIONS" and "LOAD RV DATA")
# -----------------------------------------------------------------------------
"The page's quasi-periodic Celerite kernel (RealTerm + ComplexTerm)."
function gp_quasi_periodic(θ)
    B, C, L, Prot = θ.gp_B, θ.gp_C, θ.gp_L, θ.gp_Prot
    kernel = Celerite.RealTerm(
        log(B * (1 + C) / (2 + C)),  # log_a
        log(1 / L)                   # log_c
    ) + Celerite.ComplexTerm(
        log(B / (2 + C)),            # log_a
        -Inf,                        # log_b
        log(1 / L),                  # log_c
        log(2π / Prot)               # log_d
    )
    return Celerite.CeleriteGP(kernel)
end

"One RadialVelocityObs per instrument, as on the page (composable variable blocks)."
function create_rv_likelihood(tbl, name, mean_epoch, use_gp)
    base_vars = @eval @variables begin
        offset ~ Uniform(-1000, 1000)
        jitter ~ LogUniform(0.1, 100)
        m = system.m
    end
    gp_vars = @eval @variables begin
        gp_B = system.gp_B
        gp_C = system.gp_C
        gp_L = system.gp_L
        gp_Prot = system.gp_Prot
    end
    vars = use_gp ? vcat(base_vars, gp_vars) : base_vars
    me = Float64(mean_epoch)
    trend = (θ_obs, epoch) -> θ_obs.m * (epoch - me)
    return RadialVelocityObs(
        tbl;
        target = :A,
        ref = Barycentre,
        name = name,
        gaussian_process = use_gp ? gp_quasi_periodic : nothing,
        variables = vars,
        trend_function = trend,
    )
end

"Instrument name usable as an observation (and chain column) name."
clean_name(s) = replace(String(s), r"[^A-Za-z0-9_]" => "_")

"""
The page's per-instrument MAD clipping. Note: StatsBase.mad is already
normalised, so σ = 1.4826·mad(rv) is 1.4826² × the raw MAD (page behaviour).
"""
function reject_rv_outliers(d, insts; clip=3.0)
    parts = NamedTuple[]
    tot0 = 0; tot = 0
    for inst in insts
        m = findall(==(inst), d.ins_name)
        rv = d.rv[m]
        tot0 += length(m)
        σ = 1.4826 * Octofitter.StatsBase.mad(rv)
        med = Statistics.median(rv)
        keep = m[abs.(rv .- med) .< clip * σ]
        dbg(@sprintf("  %s: kept %d/%d (rejected %d); clip = %.2f × %.2f m/s", inst, length(keep),
                     length(m), length(m) - length(keep), clip, σ))
        if length(keep) < 3
            dbg("  ⚠ too few measurements for $inst after clipping (< 3); keeping all for this instrument")
            keep = m
        end
        rvk = d.rv[keep]
        push!(parts, (; epoch=d.epoch[keep], rv=rvk .- Statistics.median(rvk),
                        rv_err=d.rv_err[keep], ins_name=d.ins_name[keep]))
        tot += length(keep)
    end
    dbg("  total: kept $tot/$tot0 measurements across all instruments")
    return (; epoch=reduce(vcat, [p.epoch for p in parts]), rv=reduce(vcat, [p.rv for p in parts]),
              rv_err=reduce(vcat, [p.rv_err for p in parts]),
              ins_name=reduce(vcat, [p.ins_name for p in parts]))
end

"Clipped data -> one likelihood per instrument (the page's process_rv_data)."
function rv_likelihoods(d, insts, use_gp)
    map(insts) do inst
        m = d.ins_name .== inst
        o = sortperm(d.epoch[m])
        tbl = Table(epoch=d.epoch[m][o], rv=d.rv[m][o] .- Statistics.median(d.rv[m]),
                    σ_rv=d.rv_err[m][o])
        dbg(@sprintf("  RadialVelocityObs \"%s\": %d epochs, MJD %.1f–%.1f, median σ %.2f m/s",
                     clean_name(inst), length(tbl.epoch), minimum(tbl.epoch), maximum(tbl.epoch),
                     Statistics.median(tbl.σ_rv)))
        create_rv_likelihood(tbl, clean_name(inst), Statistics.mean(tbl.epoch), use_gp)
    end
end

"rjd (BJD − 2 400 000) or JD or MJD column -> MJD, by header/magnitude."
function to_mjd(x, header)
    if Statistics.median(x) > 2.4e6
        return x .- 2400000.5, "JD → MJD (−2400000.5)"
    elseif occursin("rjd", lowercase(header)) || occursin("bjd", lowercase(header))
        return x .- 0.5, "rjd → MJD (−0.5 d)"
    else
        return x, "assumed MJD"
    end
end

"Read one RDB file (tab-separated, 2 header lines; columns rjd, vrad, svrad, …)."
function load_rdb(path)
    header = first(eachline(path))
    raw = DelimitedFiles.readdlm(path, '\t', skipstart=2)
    ok = [all(x -> x isa Real, raw[r, 1:3]) for r in axes(raw, 1)]
    raw = raw[ok, :]
    t, how = to_mjd(Float64.(raw[:, 1]), split(header, '\t')[1])
    inst = replace(basename(path), r"\.rdb$" => "")
    dbg(@sprintf("  %s: %d rows, first column \"%s\" (%s)", path, size(raw, 1),
                 split(header, '\t')[1], how))
    return (; epoch=t, rv=Float64.(raw[:, 2]), rv_err=Float64.(raw[:, 3]),
              ins_name=fill(inst, size(raw, 1))), inst
end

"""
DACE query through PythonCall (imported at run time so the module loads
without it). Installs dace-query into the CondaPkg environment when missing.
"""
function load_dace(hip)
    Base.eval(@__MODULE__, :(import PythonCall, CondaPkg))
    return Base.invokelatest(_load_dace, hip)
end

function _load_dace(hip)
    PC = getfield(@__MODULE__, :PythonCall)
    CP = getfield(@__MODULE__, :CondaPkg)
    sp = try
        PC.pyimport("dace_query.spectroscopy")
    catch
        dbg("  dace_query not importable; CondaPkg.add_pip(\"dace-query\") (one-time install)")
        CP.add_pip("dace-query")
        PC.pyimport("dace_query.spectroscopy")
    end
    res = sp.Spectroscopy.get_timeseries(target="HIP $hip", sorted_by_instrument=false,
                                         output_format="pandas")
    n = PC.pyconvert(Int, PC.pybuiltins.len(res))
    dbg("  DACE returned ", n, " rows for HIP ", hip)
    n == 0 && return nothing
    col(k, T) = PC.pyconvert(Vector{T}, res[k].tolist())
    rjd = col("rjd", Float64); rv = col("rv", Float64); err = col("rv_err", Float64)
    ins = col("ins_name", String)
    ok = isfinite.(rjd) .& isfinite.(rv) .& isfinite.(err) .& (err .> 0)
    dbg("  finite rows: ", count(ok), " | instruments: ", join(unique(ins[ok]), ", "))
    return (; epoch=rjd[ok] .- 0.5, rv=rv[ok], rv_err=err[ok], ins_name=ins[ok])
end

# -----------------------------------------------------------------------------
# Model (page sections "DEFINE PLANET MODEL(S)" and "DEFINE SYSTEM MODEL")
# -----------------------------------------------------------------------------
function make_host(host_mass)
    hm = Float64(host_mass)
    vars = @eval @variables begin
        mass = $hm           # M⊙
        flux_G  = 1.0
        flux_Hp = 1.0
    end
    return Body(name="A", variables=vars)
end

function make_planet(k, pn, host, host_mass, ep, circular)
    mp = LogUniform(0.01mjup, Float64(host_mass))
    vars = if circular
        @eval @variables begin
            planet_i = $k
            a ~ LogUniform(0.001, 1024)
            planet_present = system.n_planets >= planet_i
            mass′ ~ $mp
            mass = mass′ * planet_present
            e = 0.0              # --circular (unused on the page)
            ω = 0.0
            i = system.i         # coplanar: shared inclination…
            Ω = system.Ω         # …and node
            M0 ~ Uniform(0, 2pi)
            epoch = $ep
            flux_G  = 0.0        # dark companion
            flux_Hp = 0.0
        end
    else
        @eval @variables begin
            planet_i = $k
            a ~ LogUniform(0.001, 1024)
            planet_present = system.n_planets >= planet_i
            mass′ ~ $mp
            mass = mass′ * planet_present
            e ~ Uniform(0, 0.9)
            ω ~ Uniform(0, 2pi)
            i = system.i
            Ω = system.Ω
            M0 ~ Uniform(0, 2pi)
            epoch = $ep
            flux_G  = 0.0
            flux_Hp = 0.0
        end
    end
    return Body(name=pn, about=host, variables=vars)
end

"The page's placeholder companion for --n-planets 0 (massless, fixed orbit)."
function make_null_planet(host)
    vars = @eval @variables begin
        a = 1.0
        mass = 0.0
        e = 0.0
        ω = 0.0
        i = 0.0
        Ω = 0.0
        tp = 0.0
        flux_G  = 0.0
        flux_Hp = 0.0
    end
    return Body(name=:b, about=host, variables=vars)
end

function make_system_vars(obs, N, has_rv, use_gp)
    c = obs.catalog
    plx0, eplx = Float64(c.parallax), Float64(c.parallax_error)
    plx_p = truncated(Normal(plx0, eplx), lower=plx0 / 2)
    pmra_p = Uniform((Float64(c.pmra_dr3) .+ (-10, 10))...)
    pmdec_p = Uniform((Float64(c.pmdec_dr3) .+ (-10, 10))...)
    ra0, dec0 = Float64(c.ra), Float64(c.dec)
    rvraw = hasproperty(c, :radial_velocity) ? c.radial_velocity : missing
    rv0 = (ismissing(rvraw) || isnan(rvraw)) ? 0.0 : Float64(rvraw) * 1e3     # m/s
    refep = Octofitter.meta_gaia_DR3.ref_epoch_mjd
    base = if N >= 1
        np_p = truncated(NegativeBinomial(), upper=N)
        @eval @variables begin
            n_planets ~ $np_p
            ref_epoch = $refep
            plx ~ $plx_p
            pmra ~ $pmra_p
            pmdec ~ $pmdec_p
            dec = $dec0
            ra = $ra0
            rv = $rv0
            i ~ Sine()
            Ω ~ Uniform(0, 2pi)
        end
    else
        # --n-planets 0: the page keeps n_planets ~ truncated(NegativeBinomial(), upper=0),
        # a point mass at 0; it is fixed here instead (same model, no discrete sampling).
        @eval @variables begin
            n_planets = 0
            ref_epoch = $refep
            plx ~ $plx_p
            pmra ~ $pmra_p
            pmdec ~ $pmdec_p
            dec = $dec0
            ra = $ra0
            rv = $rv0
            i ~ Sine()
            Ω ~ Uniform(0, 2pi)
        end
    end
    dbg(@sprintf("  system: plx ~ N(%.4f, %.4f) truncated at %.4f, pmra ~ U(%.2f ± 10), pmdec ~ U(%.2f ± 10), rv = %.1f m/s",
                 plx0, eplx, plx0 / 2, c.pmra_dr3, c.pmdec_dr3, rv0))
    N >= 1 && dbg("  n_planets ~ truncated(NegativeBinomial(), upper=$N): prior P(k) = ",
                  join((@sprintf("%d: %.3f", k, pdf(truncated(NegativeBinomial(), upper=N), k)) for k in 0:N), ", "))
    vars = base
    if has_rv
        trend = @eval @variables begin
            m ~ Normal(0, 0.1)  # m/s/d trend
        end
        vars = vcat(vars, trend)
    end
    if use_gp
        gpv = @eval @variables begin
            gp_B ~ Uniform(0.00001, 2000000)
            gp_C ~ Uniform(0.00001, 200)
            gp_L ~ Uniform(40, 2000)
            gp_Prot ~ Uniform(35, 45)
        end
        vars = vcat(vars, gpv)
    end
    return vars
end

# -----------------------------------------------------------------------------
# Starting points and sampling
# -----------------------------------------------------------------------------
function page_initialize!(model; n::Integer=10_000)
    try
        Octofitter._kepsolve_use_threads[] = true
        initial_θ = collect(Octofitter.guess_starting_position(model, n)[1])
        model.starting_points = fill(collect(model.link(initial_θ)), 100)
        t = time()
        lp = model.ℓπcallback(model.starting_points[1])
        dbg(@sprintf("  best of %d prior draws: log posterior = %.3f (one evaluation %.3f ms; copied to 100 starting points)",
                     n, lp, 1000 * (time() - t)))
        return lp
    catch err
        dbg("  ⚠ page starting-point guess failed (", first(sprint(showerror, err), 300),
            "); using Octofitter.initialize!")
        return Octofitter.initialize!(model)
    end
end

"Rounds completed so far (the page's test, with a fallback)."
function rounds_done(pt)
    try
        return length(pt.shared.reports.summary.last_round_max_time)
    catch
        return pt.inputs.n_rounds
    end
end

function round_record(pt, r, t_round)
    zz = try Pigeons.stepping_stone(pt) catch; NaN end
    λ = try Pigeons.global_barrier(pt) catch; NaN end
    zpair = try Pigeons.stepping_stone_pair(pt) catch; (NaN, NaN) end
    mα = try Float64(minimum(Pigeons.swap_prs(pt))) catch; NaN end      # Pigeons' min(α) column
    return (; round=r, samples=2^r, logZ=Float64(zz), logZ_fwd=Float64(zpair[1]),
              logZ_bwd=Float64(zpair[2]), Λ=Float64(λ), min_α=mα, seconds=t_round)
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_g23hx(; cfg=config_from_env(), outdir=SCRIPT_DIR, prefix="g23hx_",
                dark=true, corner_dark=false, show_plots=true)

The page's script: G23H (+ RV) model, round-by-round Pigeons with checkpoints,
then the page's plots and summaries.
"""
function run_g23hx(; cfg=config_from_env(),
                     outdir::AbstractString=SCRIPT_DIR,
                     prefix::AbstractString=PREFIX,
                     dark::Bool=true,
                     corner_dark::Bool=false,
                     show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    ckdir = joinpath(outdir, prefix * "checkpoints")
    mkpath(ckdir)
    accept_downloads!()
    print_env_info(cfg, outdir)
    set_plot_theme!(; dark)
    cfg.n_planets >= 0 || error("--n-planets must be ≥ 0")
    cfg.n_rounds >= 1 || error("--n-rounds must be ≥ 1")

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
    watchdirs() = (datadeps_dirs()..., outdir)

    N = cfg.n_planets
    planet_names = Symbol.(Char.(Int('b') .+ (0:N-1)))      # page
    dbg("Companions: ", N == 0 ? "none (page's massless placeholder b)" : join(planet_names, ", "))

    soft_stage("Check DataDep folders for interrupted downloads") do
        list_datadeps()
        check_stale_datadeps()
    end
    inputs = stage("Catalog inputs (mode = $(catalog_mode()))") do
        with_heartbeat("catalog inputs"; watch=watchdirs(), outdir) do
            load_catalog_inputs(cfg; outdir)
        end
    end

    # ---- Host mass (page: --host-mass, required) --------------------------------
    host_mass = if cfg.host_mass !== nothing
        cfg.host_mass
    elseif inputs.gaia_id !== nothing && haskey(KNOWN_HOSTS, inputs.gaia_id)
        hm, who = KNOWN_HOSTS[inputs.gaia_id]
        dbg("host mass not given: using ", hm, " M⊙ for ", who, " (set OCTOG23X_HOST_MASS / --host-mass)")
        hm
    else
        error("--host-mass is required for this target (OCTOG23X_HOST_MASS)")
    end

    # ---- G23H observation ---------------------------------------------------------
    absastrom, sysname′ = stage("G23HObs (" * (cfg.hip !== nothing ? "hip_id=$(cfg.hip)" : "gaia_id=$(cfg.gaia)") *
                                "; heartbeat every " * get(ENV, "OCTOG23X_HEARTBEAT", "30") * " s)") do
        with_heartbeat("G23HObs"; watch=watchdirs(), outdir) do
            inout(() -> make_g23h(cfg, N == 0 ? Symbol[] : planet_names, inputs.kw))
        end
    end
    describe_obs(absastrom)
    ra, dec = absastrom.catalog.ra, absastrom.catalog.dec
    dbg("Found target: RA=", ra, ", Dec=", dec, " (", sysname′, ")")
    soft_stage("Channel table (all rows)") do
        DEBUG[] && display(absastrom.table)
    end
    absastrom = stage("Select observation types (likeobj_from_epoch_subset)") do
        select_channels(absastrom, cfg)
    end
    sysname = "$sysname′-" * join(string.(absastrom.table.kind), "-")
    dbg("page's sysname (used in checkpoint names): ", sysname)
    soft_stage("Catalog proper motions: CSV + figure") do
        push!(outputs, write_pm_csv(joinpath(outdir, prefix * "pm_catalog.csv"), absastrom.table))
        isempty(pm_rows(absastrom.table)) && error("no proper-motion channels selected")
        savefig!(pm_figure(absastrom.table, sysname′), "pm_data.png")
    end

    # ---- RV data --------------------------------------------------------------------
    rvlikes = Any[]
    rv_sources = String[]
    if cfg.dace
        hip = cfg.hip !== nothing ? cfg.hip :
              finite_hip(absastrom.catalog.hip_id) ? round(Int, absastrom.catalog.hip_id) : nothing
        res = if hip === nothing
            dbg("⚠ DACE needs a HIP number and this source has none; skipping DACE")
            nothing
        else
            soft_stage("DACE RV query (HIP $hip; PythonCall/CondaPkg)") do
                with_heartbeat("DACE query"; watch=(outdir,), outdir) do
                    load_dace(hip)
                end
            end
        end
        if res === nothing
            dbg("⚠ NO RVs from DACE (see above) — continuing without them")
        else
            insts = unique(res.ins_name)
            d = stage("Per-instrument outlier rejection (DACE)") do
                reject_rv_outliers(res, insts; clip=cfg.clip)
            end
            soft_stage("DACE RV figure") do
                savefig!(rv_data_figure(d, insts, "DACE (clipped)"), "rvs-dace.png")
            end
            append!(rvlikes, stage("RV likelihoods (DACE)") do
                rv_likelihoods(d, insts, cfg.gp)
            end)
            push!(rv_sources, "DACE")
        end
    end
    for f in cfg.rdb
        p = isabspath(f) ? f : (isfile(f) ? abspath(f) : joinpath(outdir, f))
        if !isfile(p)
            dbg("⚠ RDB file not found: ", f, " (also looked in ", outdir, ")")
            continue
        end
        d0, inst = stage("Load RDB file $(basename(p))") do
            load_rdb(p)
        end
        d = stage("Outlier rejection ($inst)") do
            reject_rv_outliers(d0, [inst]; clip=cfg.clip)
        end
        soft_stage("RV figure ($inst)") do
            savefig!(rv_data_figure(d, [inst], inst * " (clipped)"), "rvs-$(lowercase(clean_name(inst))).png")
        end
        append!(rvlikes, stage("RV likelihood ($inst)") do
            rv_likelihoods(d, [inst], cfg.gp)
        end)
        push!(rv_sources, inst)
    end
    has_rv = !isempty(rvlikes)
    has_rv || dbg("No RV data loaded", cfg.gp ? " — --gp ignored (it only applies to RVs)" : "")
    use_gp = cfg.gp && has_rv

    # ---- Bodies and system ----------------------------------------------------------
    ref_planet_pos = Statistics.mean(absastrom.gaia_table.epoch)
    dbg(@sprintf("M0 reference epoch = mean Gaia forecast epoch = MJD %.2f", ref_planet_pos))
    host = stage("Host body A (mass $(host_mass) M⊙, flux_G = flux_Hp = 1)") do
        make_host(host_mass)
    end
    planets = stage("Companion bodies") do
        N == 0 ? Body[make_null_planet(host)] :
                 Body[make_planet(k, planet_names[k], host, host_mass, ref_planet_pos, cfg.circular)
                      for k in 1:N]
    end
    N >= 1 && dbg("  each companion: a ~ LogUniform(0.001, 1024) AU, mass′ ~ LogUniform(0.01 M_jup, ",
                  host_mass, " M⊙), mass = mass′·(n_planets ≥ k), ",
                  cfg.circular ? "e = 0, ω = 0" : "e ~ U(0, 0.9), ω ~ U(0, 2π)",
                  ", i and Ω shared (system), M0 ~ U(0, 2π), dark")
    observations = Any[absastrom]
    append!(observations, rvlikes)
    if N > 1
        stage("OrbitOrderPrior + NonCrossingPrior (N = $N)") do
            push!(observations, OrbitOrderPrior(planets...))
            push!(observations, NonCrossingPrior(bodies=Tuple(planets)))
        end
    end
    system_vars = stage("System variables") do
        make_system_vars(absastrom, N, has_rv, use_gp)
    end
    sys = stage("Assemble System") do
        System(; name=replace(sysname′, r"[^A-Za-z0-9_]" => "_"), bodies=[host; planets],
               observations, variables=system_vars)
    end
    DEBUG[] && display(sys)

    model = stage("Compile LogDensityModel (verbosity=$(cfg.verbosity), autodiff=AutoFiniteDiff())") do
        Octofitter.LogDensityModel(sys; verbosity=cfg.verbosity, autodiff=AutoFiniteDiff())
    end
    DEBUG[] && display(model)

    quiet(f) = use_gp ? Logging.with_logger(f, Logging.ConsoleLogger(stderr, Logging.Error)) : f()
    stage("Starting points (" * (cfg.init == "page" ? "page: guess_starting_position, 10 000 draws" :
                                                       "Octofitter.initialize!") * ")") do
        quiet(() -> cfg.init == "page" ? page_initialize!(model) : Octofitter.initialize!(model))
    end

    # ---- Round 1, then one round at a time with checkpoints --------------------------
    rows = NamedTuple[]
    chain, pt = stage("Pigeons round 1 (n_chains=$(cfg.n_chains), SliceSampler, no variational leg)") do
        t = time()
        c, p = octofit_pigeons(model;
            n_chains = cfg.n_chains,
            n_chains_variational = 0,
            variational = nothing,
            n_rounds = 1,
            explorer = Pigeons.SliceSampler(),
            multithreaded = true,
        )
        push!(rows, round_record(p, 1, time() - t))
        (c, p)
    end
    while rounds_done(pt) < cfg.n_rounds
        r = rounds_done(pt) + 1
        chain, pt = stage(@sprintf("Pigeons round %d of %d (%d samples)", r, cfg.n_rounds, 2^r)) do
            t = time()
            Pigeons.increment_n_rounds!(pt, 1)
            c, p = octofit_pigeons(pt)
            push!(rows, round_record(p, r, time() - t))
            (c, p)
        end
        rec = rows[end]
        dbg(@sprintf("  round %d: log(Z₁/Z₀) ≈ %.3f (pair %.3f / %.3f), Λ = %.2f, min(α) = %.3g, %.1f s",
                     rec.round, rec.logZ, rec.logZ_fwd, rec.logZ_bwd, rec.Λ, rec.min_α, rec.seconds))
        soft_stage("Checkpoint after round $r") do
            chain = Octofitter.MCMCChains.setinfo(chain, (; chain.info..., logevidence_ratio=rec.logZ_bwd))
            p = joinpath(ckdir, "$sysname-post-round$(pt.inputs.n_rounds).fits")
            Octofitter.savechain(p, chain)
            dbg("  checkpoint ", p)
            CSV.write(joinpath(outdir, prefix * "pigeons-summary.csv"), pt.shared.reports.summary)
            CSV.write(joinpath(outdir, prefix * "rounds.csv"), [r for r in rows])
        end
    end
    push!(outputs, joinpath(outdir, prefix * "rounds.csv"))
    isfile(joinpath(outdir, prefix * "pigeons-summary.csv")) &&
        push!(outputs, joinpath(outdir, prefix * "pigeons-summary.csv"))
    if cfg.n_rounds == 1
        soft_stage("Round table") do
            CSV.write(joinpath(outdir, prefix * "rounds.csv"), [r for r in rows])
        end
    end

    final = joinpath(outdir, prefix * "post.fits")
    soft_stage("Save final chain -> FITS (+ reload check)") do
        Octofitter.savechain(final, chain)
        push!(outputs, final)
        c2 = Octofitter.loadchain(final; model)
        dbg("  reloaded chain size = ", size(c2), size(c2) == size(chain) ? "  (matches)" : "  (DIFFERS)")
    end

    # ---- Analysis -----------------------------------------------------------------
    dbg("="^60); dbg("FINAL RESULTS"); dbg("="^60)
    DEBUG[] && display(chain)
    print_summary(chain, cfg, N == 0 ? Symbol[] : planet_names, host_mass)
    nt = N >= 1 ? n_planets_table(chain, N; min_α=rows[end].min_α) : nothing
    nt === nothing || foreach(l -> dbg(l), nt.lines)

    soft_stage("Per-round convergence figure") do
        length(rows) >= 2 || error("only one round")
        savefig!(rounds_figure(rows), "rounds.png")
    end
    soft_stage("Corner plot (small=true, light theme)") do
        savefig!(inout(() -> corner(model, chain; dark=corner_dark)), "corner.png")
    end
    soft_stage("octoplot (orbits; RV panels when RVs are in the model)") do
        savefig!(inout(() -> octoplot(model, chain)), "orbits.png")
    end
    if N >= 1
        soft_stage("dotplot (mass vs separation)") do
            savefig!(inout(() -> Octofitter.dotplot(model, chain)), "mass-sep.png")
        end
        soft_stage("dotplot (mass vs period)") do
            savefig!(inout(() -> Octofitter.dotplot(model, chain; mode=:period)), "mass-period.png")
        end
        soft_stage("Mass–sma pairplot (page; all samples)") do
            savefig!(mass_sma_pairplot(chain), "mass-sma.png")
        end
        soft_stage("Mass–sma pairplot, log₁₀ axes, samples with b present") do
            savefig!(mass_sma_pairplot_log(chain), "mass-sma-log.png")
        end
        if nt !== nothing
            soft_stage("n_planets prior vs posterior figure") do
                savefig!(n_planets_figure(nt, N), "n_planets.png")
            end
        end
    end
    if has_rv
        soft_stage("rvplot (single draw, all instruments)") do
            savefig!(inout(() -> Octofitter.rvplot(model, chain)), "rvplot.png")
        end
    end

    # ---- Evidence summary --------------------------------------------------------------
    lines = String[]
    push!(lines, "Target $(sysname′) (host $(host_mass) M⊙), N ≤ $N companion(s)" *
                 (cfg.circular ? " (circular)" : "") * ", RV: " *
                 (has_rv ? join(rv_sources, ", ") : "none") * (use_gp ? " + GP" : ""))
    push!(lines, "Channels: " * join(string.(absastrom.table.kind), ", ") *
                 " | freeze_epochs = $(cfg.freeze)")
    push!(lines, "Per round (n_chains = $(cfg.n_chains)):")
    for r in rows
        push!(lines, @sprintf("    round %2d  %5d samples  log(Z₁/Z₀) %10.3f  Λ %6.2f  min(α) %9.3g  %7.1f s",
                              r.round, r.samples, r.logZ, r.Λ, r.min_α, r.seconds))
    end
    nt === nothing || append!(lines, nt.lines)
    foreach(l -> dbg(l), lines)
    soft_stage("Write summary") do
        p = joinpath(outdir, prefix * "summary.txt")
        write(p, join(lines, "\n") * "\n")
        push!(outputs, p)
    end

    list_outputs(outputs)
    dbg("Checkpoints in ", ckdir, " (", length(readdir(ckdir)), " files)")
    dbg("Done. Total ", @sprintf("%.1f min", (time() - T0[]) / 60))
    return (; model, chain, pt, rounds=rows, n_planets=nt, sysname, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default when the REPL has 1 thread)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString=PREFIX)
    T0[] = time()
    accept_downloads!()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOG23X_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOG23X_CHILD=1]")
    dbg("  Child output follows; its plots are shown here when it finishes.")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOG23X_RELAUNCH\"]=\"false\".")
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
    p = joinpath(outdir, prefix * "post.fits")
    if isfile(p) && mtime(p) >= t
        chain = soft_stage("Load final chain written by child") do
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
    elseif lowercase(get(ENV, "OCTOG23X_SHOW_PLOTS", "true")) != "false"
        soft_stage("Show $(length(pngs)) saved plot(s) in the plot pane") do
            show_saved_pngs(pngs)
        end
    end
    return (; relaunched=true, success=ok, chain, outputs=new_files)
end

"Plot order for display."
function png_rank(p)
    f = basename(p)
    for (k, key) in enumerate(("pm_data", "rvs-", "rounds", "orbits", "corner", "mass-sep",
                               "mass-period", "mass-sma-log", "mass-sma", "n_planets", "rvplot"))
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

println(@sprintf("[G23X] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterG23HExample

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterG23HExample.jl [page args]` -> runs
#   - VSCode "Execute active File in REPL" -> runs; with 1 thread it relaunches
#     in `julia --threads=auto` (OCTOG23X_RELAUNCH=false to disable,
#     OCTOG23X_THREADS to set the count)
#   - ENV["OCTOG23X_AUTORUN"] = "false"; include() -> loads module only
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOG23X_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOG23X_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOG23X_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global g23hx_result = OctofitterG23HExample.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOG23X_THREADS", "auto"))
            println("[G23X] Child run finished. `g23hx_result` has fields: relaunched, success, ",
                    "chain (loaded from FITS), outputs")
        else
            global g23hx_result = OctofitterG23HExample.run_g23hx(show_plots=!is_child)
            is_child || println("[G23X] Result stored in `g23hx_result` (fields: model, chain, pt, ",
                                "rounds, n_planets, sysname, outputs)")
        end
    end
end

nothing
