#=
================================================================================
 OctofitterDR4.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Fitting Gaia DR4 Pre-Release
 Epoch Astrometry":
   https://sefffal.github.io/Octofitter.jl/dev/gaia-dr4-prerelease/

 Target: Gaia-4 (Gaia DR3 1457486023639239296, G = 11.91), host of Gaia-4b,
 a ~11.8 M_jup companion on a ~571 d orbit (Stefansson et al. 2025), fitted
 from the real Gaia DR4 pre-release along-scan epoch astrometry alone.
 Data: the CCD-level CSV extract shipped with the Octofitter docs
 (gaia4_epoch_astrometry.csv; © ESA/Gaia/DPAC — see the ESA DR4 pre-release
 page for licence, acknowledgement and citation instructions).

 Reduction (the page's "bootstrap reduction", re-implemented without
 DataFrames): keep AGIS-used CCD rows; per transit take the median along-scan
 position, its error from 256 bootstrap medians (RNG Xoshiro(source_id),
 transits in sorted order — deterministic), floored at median(ipd_error)/√N.
 Expected for Gaia-4: 93 transits, MJD 57038.6–58843.2 (4.94 yr).

 Models (Pigeons, page settings: n_chains=16, n_rounds=8, no variational ref):
   :barycentric  the page's model: GaiaDR4AstromObs (target = Photocentre,
                 ref = Barycentre) with jitter, frame zero-point offsets and
                 barycentre proper motion in the observation block; plx ~ U(0,100).
   :anchored     optional: the page's "anchoring the frame to the star"
                 variant — sample the star's plx/pm/offsets and derive the
                 barycentre's with anchor_offsets(system_interim, :A, epoch).
                 Same data and likelihood, so its log Z should match :barycentric.
 Bodies: A mass ~ N(0.644, 0.02) M⊙, flux 1; b mass ~ LogUniform(0.3, 100) M_jup,
 dark (flux 0), a ~ U(0, 10) AU, e ~ U(0, 0.99), ω, Ω, θ ~ UniformCircular,
 i ~ Sine, θ at the mean transit epoch.
 Starting point: the page's initialize! guess seeded with the Gaia DR3 solution
 (queried via Octofitter._query_gaia_dr3) — or the NSS solution
 (initialize_from_nss!) or plain initialize!.

 Outputs: CCD vs transit data figure, reduction diagnostics, transit CSV,
 page results block vs Stefansson et al. 2025 (P = 571.3 ± 1.4 d,
 m_b = 11.8 ± 0.7 M_jup), octoplot, single-draw octoplot, gaiastarplot (MAP),
 the page's 3×3 period-quantile gaiastarplot grid, light corner, marginals,
 optional NSS comparison, evidence table. OCTODR4_OTHERS=true also downloads,
 reduces and plots the Gaia BH3 and HD 114762 extracts (data only; the page
 gives no model for them).

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `dr4_result`)
   Shell:    julia --threads=auto OctofitterDR4.jl
   Library:  ENV["OCTODR4_AUTORUN"] = "false"; include("OctofitterDR4.jl")
             res = OctofitterDR4.run_dr4(models=(:barycentric, :anchored))

 Threads: every fit uses Pigeons. In a 1-thread REPL the run is relaunched in
 a child `julia --threads=auto` (output streamed here; chains loaded back and
 saved plots shown in the plot pane afterwards). OCTODR4_RELAUNCH=false
 disables this; OCTODR4_THREADS sets the count.

 Convenience ENV settings (also passed to a relaunched child):
   OCTODR4_MODELS  = "barycentric" (default) or "barycentric,anchored"
   OCTODR4_ROUNDS  = "8" (default, page)      OCTODR4_CHAINS = "16" (page)
   OCTODR4_INIT    = "page" (default: DR3-seeded initialize!) | "nss" | "auto"
   OCTODR4_NBOOT   = "256" (default, page) — bootstrap draws per transit
   OCTODR4_NSS     = "true" (default) — query the NSS solution and compare
   OCTODR4_OTHERS  = "false" (default) — also reduce/plot BH3 and HD 114762
   OCTODR4_SHOW_PLOTS = "true" (default) — show the child's PNGs here
   OCTODR4_HEARTBEAT  = "30" (default) — seconds between "still waiting" lines

 Data files (downloaded once from the Octofitter repo at a pinned commit and
 reused): gaia4_epoch_astrometry.csv (+ gaia_bh3_…, hd114762_… with
 OCTODR4_OTHERS). Gaia archive queries (DR3 solution, NSS) cache next to this
 file. Angles stay in degrees (v9 GaiaDR4AstromObs converts internally).

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTODR4_ENV_MODE=local uses ./octofitter_v9_env). Needs Octofitter
 v9, CairoMakie, PairPlots, Distributions, Pigeons, CSV.

 Outputs (prefix "dr4_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-25  Initial version on the OctofitterG23H v1.0.1 scaffolding
                   ([DR4 +t] debug stages, soft optional stages, env bootstrap
                   with v9 guards, @__DIR__ outputs, dark theme with light
                   corner plots, threaded relaunch with exit-code-only failure
                   report and PNGs shown in the parent's plot pane, heartbeat
                   for network steps, pinned-commit data downloads, @eval-built
                   @variables, initialize! before Pigeons). Adds: DataFrames-
                   free bootstrap transit reduction; GaiaDR4AstromObs
                   barycentric and anchored models; DR3/NSS starting points;
                   page results vs Stefansson et al.; period-quantile grid.
 v1.0.1 2026-09-25 Load failed: "First argument to `@sprintf` must be a format
                   string" — a format built with "…" * "…" in the evidence
                   block. Now a single literal, with the tail appended after.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "CairoMakie", "PairPlots", "Distributions", "Pigeons", "CSV"),
    min_version = Dict("Octofitter" => v"9"),
    mode = lowercase(get(ENV, "OCTODR4_ENV_MODE", "temp"))

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
            println("[DR4] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[DR4] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[DR4] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[DR4] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[DR4] Existing v9 environment found at ", Base.active_project())
        else
            Pkg.add(spec.(collect(required)))
        end
        println("[DR4] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[DR4] Loading Octofitter, Pigeons, CairoMakie, PairPlots, Distributions, CSV ",
        "(first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterDR4

const LOAD_T0 = time()

using Octofitter
using CairoMakie
using PairPlots
using Distributions
using Pigeons
import CSV
import Downloads
import Random
import Statistics
using Printf

export run_dr4, reduce_transits, build_system, set_plot_theme!

const VERSION_STRING = "1.0.1"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterDR4 needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

# Target, data files and published values (as on the tutorial page)
const GAIA4_SOURCE_ID = 1457486023639239296
const REF_EPOCH_MJD   = 57936.375          # Gaia DR4 reference epoch, J2017.5
const PUBLISHED = (P_day = (571.3, 1.4), m_mjup = (11.8, 0.7))   # Stefansson et al. 2025

const OCTO_COMMIT = "8c6daaaf5292b6cbdb0aa4f4b1e9a1a5f0bd21f3"
const OCTO_RAW = "https://raw.githubusercontent.com/sefffal/Octofitter.jl/$(OCTO_COMMIT)/docs/src/"
const SOURCES = (
    gaia4    = (file="gaia4_epoch_astrometry.csv",    sid=1457486023639239296, label="Gaia-4"),
    bh3      = (file="gaia_bh3_epoch_astrometry.csv", sid=4318465066420528000, label="Gaia BH3"),
    hd114762 = (file="hd114762_epoch_astrometry.csv", sid=3937211745905473024, label="HD 114762"),
)

const PRIORS = (
    M_A  = truncated(Normal(0.644, 0.02), lower=0.1),   # [M⊙] (Stefansson et al. 2025)
    m_b  = LogUniform(0.3mjup, 100mjup),                # [M⊙]
    a_b  = Uniform(0, 10),                              # [AU] (Gaia-4b ~1.17 AU)
    e_b  = Uniform(0, 0.99),
    plx  = Uniform(0, 100),                             # [mas]
    pm   = Uniform(-1000, 1000),                        # [mas/yr]
    off  = Normal(0, 1000),                             # [mas]
    jit  = LogUniform(0.00001, 10),                     # [mas]
)

const MODEL_LABELS = Dict(:barycentric => "barycentric frame (page model)",
                          :anchored    => "frame anchored to the star")
const MODEL_ORDER = (:barycentric, :anchored)

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterDR4.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[DR4 +%7.1fs] ", time() - T0[]), msg...)
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

"""
    with_heartbeat(f, what; every)

Run `f()` while a background task prints a line every `every` seconds, so a
slow network query (Gaia TAP, NSS) is visibly still running.
"""
function with_heartbeat(f, what::AbstractString;
                        every::Real=parse(Float64, get(ENV, "OCTODR4_HEARTBEAT", "30")))
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

function print_env_info(outdir, models, n_rounds, n_chains, init, nboot)
    DEBUG[] || return nothing
    dbg("OctofitterDR4 v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("target = Gaia-4 (Gaia DR3 ", GAIA4_SOURCE_ID, ") | DR4 ref epoch MJD ", REF_EPOCH_MJD)
    dbg("models = ", join(models, ", "), " | n_rounds = ", n_rounds, " (", 2^n_rounds,
        " samples) | n_chains = ", n_chains, " | init = ", init, " | n_boot = ", nboot)
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
qline(x) = Statistics.quantile(filter(isfinite, x), (0.16, 0.5, 0.84))

"First chain column whose name ends with `suffix` (observation variables carry
the observation-name prefix, e.g. `GaiaDR4_astrometric_jitter`)."
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

"Period [d] per draw from a, A_mass, b_mass (the page's derivation)."
period_days(chain) = sqrt.(colvec(chain, :b_a) .^ 3 ./
                           (colvec(chain, :A_mass) .+ colvec(chain, :b_mass))) .* 365.25

"Page's results block, plus a comparison with Stefansson et al. (2025)."
function print_results(chain, label)
    DEBUG[] || return nothing
    q(v) = round.(Statistics.quantile(v, (0.16, 0.5, 0.84)), sigdigits=5)
    Pday = period_days(chain)
    mb = colvec(chain, :b_mass) ./ mjup
    dbg("Results — ", label, " (", length(Pday), " samples; 16/50/84 %):")
    dbg("    period  [day] : ", q(Pday), "   (Stefansson et al. 2025: 571.3 ± 1.4)")
    dbg("    a       [AU]  : ", q(colvec(chain, :b_a)))
    dbg("    e             : ", q(colvec(chain, :b_e)))
    dbg("    i       [deg] : ", round.(rad2deg.(Statistics.quantile(colvec(chain, :b_i),
                                                                  (0.16, 0.5, 0.84))), digits=1))
    dbg("    mass_b  [Mjup]: ", q(mb), "   (Stefansson et al. 2025: 11.8 ± 0.7)")
    dbg(@sprintf("    period offset from published: %+.2f d = %+.2fσ(pub) | mass offset: %+.2f M_jup = %+.2fσ(pub)",
                 Statistics.median(Pday) - PUBLISHED.P_day[1],
                 (Statistics.median(Pday) - PUBLISHED.P_day[1]) / PUBLISHED.P_day[2],
                 Statistics.median(mb) - PUBLISHED.m_mjup[1],
                 (Statistics.median(mb) - PUBLISHED.m_mjup[1]) / PUBLISHED.m_mjup[2]))
    for (lab, key, f) in (("plx [mas]", "plx", identity),
                          ("A mass [M⊙]", "A_mass", identity),
                          ("b Ω [deg] (folded)", "b_Ω", x -> rad2deg.(fold_axial(x))),
                          ("jitter [mas]", "astrometric_jitter", identity),
                          ("pmra [mas/yr]", "pmra", identity),
                          ("pmdec [mas/yr]", "pmdec", identity))
        c = haspar(chain, Symbol(key)) ? Symbol(key) : findcol(chain, "_" * key)
        c === nothing && continue
        v = qline(f(colvec(chain, c)))
        dbg(@sprintf("    %-18s %12.4f  [%12.4f, %12.4f]   (%s)", lab, v[2], v[1], v[3], c))
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
# Data: download, read, bootstrap reduction
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

"CCD-level table as a NamedTuple of plain vectors (all rows; `#` header skipped)."
function read_ccd(path)
    f = CSV.File(path; comment="#")
    col(s) = collect(getproperty(f, s))
    used = col(:used_by_agis_al)
    used = eltype(used) <: Bool ? used : lowercase.(string.(used)) .== "true"
    return (; source_id = Int64.(col(:source_id)), transit_id = Int64.(col(:transit_id)),
              epoch = Float64.(col(:epoch)), scan_pos_angle = Float64.(col(:scan_pos_angle)),
              parallax_factor_al = Float64.(col(:parallax_factor_al)),
              centroid_pos_al = Float64.(col(:centroid_pos_al)),
              centroid_pos_error_al = Float64.(col(:centroid_pos_error_al)),
              ipd_error_al = Float64.(col(:ipd_error_al)), used_by_agis_al = Bool.(used))
end

"Keep the rows AGIS used (Gaia's own CCD-level outlier rejection)."
agis_only(c) = map(v -> v[c.used_by_agis_al], c)

"""
    reduce_transits(ccd; n_boot=256) -> Dict{Int64,Table}

The page's bootstrap reduction, without DataFrames: one row per transit with
the median along-scan position, the std of `n_boot` bootstrap medians floored
at median(ipd_error_al)/√N (N = 1: the IPD error), the mean epoch, and the
transit's scan angle [deg] and parallax factor. RNG Xoshiro(source_id), rows
sorted stably by transit_id, so the result is deterministic.
"""
function reduce_transits(ccd; n_boot::Integer=256)
    out = Dict{Int64,Any}()
    for sid in sort(unique(ccd.source_id))
        rng = Random.Xoshiro(sid)
        rows_s = findall(==(sid), ccd.source_id)
        rows_s = rows_s[sortperm(ccd.transit_id[rows_s]; alg=MergeSort)]   # stable
        tids = ccd.transit_id[rows_s]
        cols = (transit_id=Int64[], epoch=Float64[], scan_pos_angle=Float64[],
                parallax_factor_al=Float64[], centroid_pos_al=Float64[],
                centroid_pos_error_al=Float64[], n_ccd=Int[], floor_err=Float64[])
        i = 1
        while i <= length(rows_s)
            j = i
            while j < length(rows_s) && tids[j+1] == tids[i]
                j += 1
            end
            r = rows_s[i:j]
            pos = ccd.centroid_pos_al[r]
            n = length(pos)
            med = Statistics.median(pos)
            floor_err = Statistics.median(ccd.ipd_error_al[r]) / sqrt(n)
            err = if n == 1
                ccd.ipd_error_al[r[1]]
            else
                meds = [Statistics.median(rand(rng, pos, n)) for _ in 1:n_boot]
                max(Statistics.std(meds), floor_err)
            end
            push!(cols.transit_id, tids[i]); push!(cols.epoch, Statistics.mean(ccd.epoch[r]))
            push!(cols.scan_pos_angle, ccd.scan_pos_angle[r[1]])
            push!(cols.parallax_factor_al, ccd.parallax_factor_al[r[1]])
            push!(cols.centroid_pos_al, med); push!(cols.centroid_pos_error_al, err)
            push!(cols.n_ccd, n); push!(cols.floor_err, floor_err)
            i = j + 1
        end
        out[sid] = Table(; cols...)
    end
    return out
end

"Download, read and reduce one source; debug counts vs the page."
function load_source(key::Symbol; outdir=SCRIPT_DIR, n_boot=256)
    s = getproperty(SOURCES, key)
    path = fetch_once(OCTO_RAW * s.file, joinpath(outdir, s.file))
    ccd = read_ccd(path)
    used = agis_only(ccd)
    dbg("  ", s.label, ": ", length(ccd.epoch), " CCD rows over ", length(unique(ccd.transit_id)),
        " transits; ", length(used.epoch), " used by AGIS over ", length(unique(used.transit_id)), " transits")
    tl = reduce_transits(used; n_boot)[s.sid]
    span = (maximum(tl.epoch) - minimum(tl.epoch)) / 365.25
    dbg(@sprintf("  %s: %d transits, MJD %.1f–%.1f (%.2f yr); median σ %.3f mas, floor-limited %d of %d",
                 s.label, length(tl.epoch), minimum(tl.epoch), maximum(tl.epoch), span,
                 Statistics.median(tl.centroid_pos_error_al),
                 count(tl.centroid_pos_error_al .<= tl.floor_err .* (1 + 1e-12)), length(tl.epoch)))
    return (; key, label=s.label, sid=s.sid, ccd, used, tl, path)
end

function write_transit_csv(p, tl)
    CSV.write(p, (; transit_id=tl.transit_id, epoch=tl.epoch, scan_pos_angle=tl.scan_pos_angle,
                    parallax_factor_al=tl.parallax_factor_al, centroid_pos_al=tl.centroid_pos_al,
                    centroid_pos_error_al=tl.centroid_pos_error_al, n_ccd=tl.n_ccd))
    return p
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

"CCD rows (grey: rejected by AGIS) with the transit-level medians and errors."
function data_figure(src)
    c, tl = src.ccd, src.tl
    fig = Figure(size=(1150, 820))
    ax1 = Axis(fig[1, 1], xlabel="epoch [MJD]", ylabel="centroid_pos_al [mas]",
               title="$(src.label) DR4 pre-release: CCD observations and transit medians")
    rej = .!c.used_by_agis_al
    scatter!(ax1, c.epoch[rej], c.centroid_pos_al[rej]; color=(:gray, 0.5), markersize=4,
             marker=:xcross, label="CCD, not used by AGIS")
    scatter!(ax1, c.epoch[.!rej], c.centroid_pos_al[.!rej]; color=(CMP1, 0.35), markersize=4,
             label="CCD, used by AGIS")
    errorbars!(ax1, tl.epoch, tl.centroid_pos_al, tl.centroid_pos_error_al; color=CMP2)
    scatter!(ax1, tl.epoch, tl.centroid_pos_al; color=CMP2, markersize=7, label="transit median")
    axislegend(ax1, position=:lt, labelsize=11)
    ax2 = Axis(fig[2, 1], xlabel="epoch [MJD]", ylabel="scan angle [deg]",
               title="Scan angles (degrees, as published) and along-scan parallax factors")
    sc = scatter!(ax2, tl.epoch, tl.scan_pos_angle; color=tl.parallax_factor_al,
                  colormap=:balance, markersize=8)
    Colorbar(fig[2, 2], sc, label="parallax factor (AL)")
    rowsize!(fig.layout, 1, Relative(0.6))
    return fig
end

"Reduction diagnostics: bootstrap error vs its floor, and CCDs per transit."
function reduction_figure(src)
    tl = src.tl
    fig = Figure(size=(1100, 420))
    ax1 = Axis(fig[1, 1], xlabel="floor: median(ipd_error)/√N [mas]",
               ylabel="adopted transit σ [mas]", xscale=log10, yscale=log10,
               title="Bootstrap error vs floor ($(length(tl.epoch)) transits)")
    scatter!(ax1, tl.floor_err, tl.centroid_pos_error_al; color=tl.n_ccd, colormap=:viridis,
             markersize=8)
    lo = minimum(vcat(tl.floor_err, tl.centroid_pos_error_al)) * 0.8
    hi = maximum(vcat(tl.floor_err, tl.centroid_pos_error_al)) * 1.25
    lines!(ax1, [lo, hi], [lo, hi]; color=(:white, 0.5), linestyle=:dash)
    ax2 = Axis(fig[1, 2], xlabel="AGIS-used CCD observations per transit", ylabel="transits")
    hist!(ax2, tl.n_ccd; bins=0.5:1:(maximum(tl.n_ccd) + 0.5), color=(CMP1, 0.85))
    return fig
end

function marginals_figure(chain, label)
    fig = Figure(size=(1100, 700))
    Pday = period_days(chain)
    specs = (
        (Pday,                               "period [d]"),
        (colvec(chain, :b_mass) ./ mjup,     "b mass [M_jup]"),
        (colvec(chain, :b_e),                "b e"),
        (rad2deg.(colvec(chain, :b_i)),      "b i [deg]"),
    )
    pubs = (PUBLISHED.P_day[1], PUBLISHED.m_mjup[1], nothing, nothing)
    for (k, ((x, lab), pub)) in enumerate(zip(specs, pubs))
        ax = Axis(fig[fldmod1(k, 2)...], xlabel=lab, ylabel="density")
        hist!(ax, x; bins=60, normalization=:pdf, color=(CMP1, 0.85))
        pub === nothing || vlines!(ax, [pub]; color=(:white, 0.8), linestyle=:dash)
    end
    Label(fig[0, 1:2], "Key marginals — $(label) (dashed: Stefansson et al. 2025)", fontsize=16)
    return fig
end

"""
The page's 3×3 grid: gaiastarplot! for the draws nearest the 5th–95th
percentiles of the period marginal, shared scale.
"""
function quantile_grid(model, chain)
    Pday = period_days(chain)
    probs = range(0.05, 0.95, length=9)
    idxs = [argmin(abs.(Pday .- Statistics.quantile(Pday, p))) for p in probs]
    dbg("  period-quantile draws: ", join((@sprintf("%.1f d", Pday[i]) for i in idxs), ", "))
    fig = Figure(size=(620, 1000))   # each panel is DataAspect: size the figure to match
    axes = Axis[]
    for (k, idx) in enumerate(idxs)
        i, j = fldmod1(k, 3)
        ax = Octofitter.gaiastarplot!(fig[i, j], model, chain, idx;
            axis=(; title="P = $(round(Pday[idx], digits=1)) d", titlesize=13,
                    xlabel="", ylabel=""))
        push!(axes, ax)
    end
    linkaxes!(axes...)
    for (k, ax) in enumerate(axes)
        i, j = fldmod1(k, 3)
        hidexdecorations!(ax; ticks=false, grid=false)
        hideydecorations!(ax; ticks=false, grid=false)
        i == 3 && (ax.xticklabelsvisible = true)
        j == 1 && (ax.yticklabelsvisible = true)
    end
    Label(fig[4, 1:3], "Δα* [mas]")
    Label(fig[1:3, 0], "Δδ [mas]", rotation=pi / 2)
    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 8)
    return fig
end

"Stacked histograms of period and mass: this fit (blue) vs NSS draws (orange)."
function nss_compare_figure(chain, nss_chain)
    fig = Figure(size=(1100, 420))
    for (k, (f, lab)) in enumerate(((period_days, "period [d]"),
                                    (c -> colvec(c, :b_mass) ./ mjup, "b mass [M_jup]")))
        ax = Axis(fig[1, k], xlabel=lab, ylabel="density")
        hist!(ax, f(chain); bins=60, normalization=:pdf, color=(CMP1, 0.6), label="this fit")
        x2 = try f(nss_chain) catch; nothing end
        x2 === nothing || hist!(ax, filter(isfinite, x2); bins=60, normalization=:pdf,
                                color=(CMP2, 0.6), label="Gaia DR3 NSS")
        k == 1 && axislegend(ax, position=:rt)
    end
    return fig
end

# -----------------------------------------------------------------------------
# Model components (values interpolated into @variables via @eval)
# -----------------------------------------------------------------------------
function make_bodies(orbit_ref_epoch)
    mA, mb, ab, eb = PRIORS.M_A, PRIORS.m_b, PRIORS.a_b, PRIORS.e_b
    A_vars = @eval @variables begin
        mass ~ $mA                 # host mass [M⊙] (Stefansson et al. 2025)
        flux = 1.0                 # the host sets the photocentre's flux scale
    end
    A = Body(name="A", variables=A_vars)
    b_vars = @eval @variables begin
        mass ~ $mb                 # companion mass [M⊙]
        flux = 0.0                 # dark companion => the photocentre is the host
        a ~ $ab                    # [AU]
        e ~ $eb
        ω ~ UniformCircular()
        i ~ Sine()
        Ω ~ UniformCircular()
        θ ~ UniformCircular()      # position angle at `epoch`
        epoch = $orbit_ref_epoch
    end
    dbg(@sprintf("  bodies: A mass ~ N(0.644, 0.02) M⊙; b mass ~ LogUniform(0.3, 100) M_jup, a ~ U(0, 10) AU, θ at MJD %.3f",
                 orbit_ref_epoch))
    return A, Body(name="b", about=A, variables=b_vars)
end

"GaiaDR4AstromObs; `anchored` takes the frame from the system block."
function make_obs(tl; anchored::Bool=false)
    jit, off, pm, re = PRIORS.jit, PRIORS.off, PRIORS.pm, REF_EPOCH_MJD
    vars = if anchored
        @eval @variables begin
            astrometric_jitter ~ $jit
            ra_offset_mas  = system.bary_ra_offset_mas
            dec_offset_mas = system.bary_dec_offset_mas
            pmra  = system.bary_pmra
            pmdec = system.bary_pmdec
            ref_epoch = $re
        end
    else
        @eval @variables begin
            astrometric_jitter ~ $jit           # mas
            ra_offset_mas  ~ $off               # frame zero-point (absorbs DR3↔DR4 offset)
            dec_offset_mas ~ $off
            pmra  ~ $pm                         # mas/yr
            pmdec ~ $pm
            ref_epoch = $re
        end
    end
    obs = GaiaDR4AstromObs(tl; target=Photocentre, ref=Barycentre, name="GaiaDR4",
                           variables=vars)
    dbg("  GaiaDR4AstromObs: ", length(obs.table.epoch), " transits, target = Photocentre, ref = Barycentre",
        anchored ? ", frame from the system block (anchored)" : "")
    return obs
end

function make_system(kind::Symbol, A, b, obs)
    if kind === :barycentric
        pp = PRIORS.plx
        vars = @eval @variables begin
            plx ~ $pp
        end
    else
        pp, pm, off, re = PRIORS.plx, PRIORS.pm, PRIORS.off, REF_EPOCH_MJD
        vars = @eval @variables begin
            # Sample the star's own solution — the quantities the data measure…
            plx_A   ~ $pp
            pmra_A  ~ $pm
            pmdec_A ~ $pm
            ra_offset_A_mas  ~ $off
            dec_offset_A_mas ~ $off
            # …and derive the barycentre's by subtracting A's modelled motion about it.
            plx_interim = plx_A
            Δ = anchor_offsets(system_interim, :A, $re)
            plx = barycentre_parallax(plx_A, Δ.dz)
            bary_pmra  = pmra_A  - Δ.pmra
            bary_pmdec = pmdec_A - Δ.pmdec
            bary_ra_offset_mas  = ra_offset_A_mas  - Δ.ra_cosdec
            bary_dec_offset_mas = dec_offset_A_mas - Δ.dec
        end
    end
    return System(name="Gaia4", bodies=[A, b], observations=[obs], variables=vars)
end

"""
    build_system(kind, tl) -> (sys, obs)

Fresh bodies, observation and system for `kind` ∈ (:barycentric, :anchored).
"""
function build_system(kind::Symbol, tl)
    kind in MODEL_ORDER || error("kind must be :barycentric or :anchored, got $kind")
    orbit_ref_epoch = Statistics.mean(tl.epoch)
    A, b = make_bodies(orbit_ref_epoch)
    obs = make_obs(tl; anchored = kind === :anchored)
    return make_system(kind, A, b, obs), obs
end

"Gaia DR3 single-star solution (network query, cached next to the script)."
function dr3_solution(; outdir=SCRIPT_DIR)
    sol = with_heartbeat("Gaia DR3 query") do
        cd(() -> Octofitter._query_gaia_dr3(gaia_id=GAIA4_SOURCE_ID), outdir)
    end
    dbg(@sprintf("  DR3 solution: plx = %.4f mas, pmra = %.4f, pmdec = %.4f mas/yr, RUWE = %s",
                 sol.parallax, sol.pmra, sol.pmdec,
                 hasproperty(sol, :ruwe) ? string(round(sol.ruwe, digits=3)) : "n/a"))
    return sol
end

"The page's starting point, seeded from the DR3 solution (names follow `kind`)."
function page_start(kind::Symbol, sol)
    bodies = (A = (; mass = 0.644), b = (; mass = 11.8mjup, a = 1.17, e = 0.1, i = 1.0))
    if kind === :barycentric
        return (; plx = sol.parallax, bodies,
                  observations = (GaiaDR4 = (astrometric_jitter = 0.1, ra_offset_mas = 0.0,
                                             dec_offset_mas = 0.0, pmra = sol.pmra,
                                             pmdec = sol.pmdec),))
    else
        return (; plx_A = sol.parallax, pmra_A = sol.pmra, pmdec_A = sol.pmdec,
                  ra_offset_A_mas = 0.0, dec_offset_A_mas = 0.0, bodies,
                  observations = (GaiaDR4 = (astrometric_jitter = 0.1,),))
    end
end

"initialize! following OCTODR4_INIT, falling back to plain initialize!."
function initialize_model!(model, kind, init, sol; outdir=SCRIPT_DIR)
    try
        if init == "nss"
            return with_heartbeat("NSS query + initialize!") do
                cd(() -> initialize_from_nss!(model; gaia_id=GAIA4_SOURCE_ID, body=:b), outdir)
            end
        elseif init == "page" && sol !== nothing
            return Octofitter.initialize!(model, page_start(kind, sol))
        end
    catch err
        dbg("  ⚠ ", init, " initialisation failed (", first(sprint(showerror, err), 300),
            "); using plain initialize!")
    end
    init == "page" && sol === nothing && dbg("  (no DR3 solution available; plain initialize!)")
    return Octofitter.initialize!(model)
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_dr4(; models=(:barycentric,), n_rounds=8, n_chains=16, init="page",
              n_boot=256, nss=true, others=false, outdir=SCRIPT_DIR,
              prefix="dr4_", dark=true, corner_dark=false, show_plots=true)

Download and reduce the Gaia-4 DR4 pre-release data, fit each model with
Pigeons (page settings), plot and summarise.
"""
function run_dr4(; models=(:barycentric,),
                   n_rounds::Integer=envint("OCTODR4_ROUNDS", "8"),
                   n_chains::Integer=envint("OCTODR4_CHAINS", "16"),
                   init::AbstractString=lowercase(get(ENV, "OCTODR4_INIT", "page")),
                   n_boot::Integer=envint("OCTODR4_NBOOT", "256"),
                   nss::Bool=envbool("OCTODR4_NSS", "true"),
                   others::Bool=envbool("OCTODR4_OTHERS", "false"),
                   outdir::AbstractString=SCRIPT_DIR,
                   prefix::AbstractString="dr4_",
                   dark::Bool=true,
                   corner_dark::Bool=false,
                   show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    print_env_info(outdir, models, n_rounds, n_chains, init, n_boot)
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

    # ---- Data ----------------------------------------------------------------
    src = stage("Gaia-4: download + bootstrap reduction (n_boot = $n_boot)") do
        load_source(:gaia4; outdir, n_boot)
    end
    dbg("  (page: 1077 CCD rows / 109 transits → 93 transits, MJD 57038.6–58843.2, 4.94 yr)")
    soft_stage("Transit-level CSV") do
        push!(outputs, write_transit_csv(joinpath(outdir, prefix * "gaia4_transits.csv"), src.tl))
    end
    soft_stage("Data figure (CCD rows, transit medians, scan angles)") do
        savefig!(data_figure(src), "gaia4_data.png")
    end
    soft_stage("Reduction diagnostics figure") do
        savefig!(reduction_figure(src), "gaia4_reduction.png")
    end
    if others
        for key in (:bh3, :hd114762)
            soft_stage("$(getproperty(SOURCES, key).label): download + reduction + data figure (no fit)") do
                s = load_source(key; outdir, n_boot)
                push!(outputs, write_transit_csv(joinpath(outdir, prefix * "$(key)_transits.csv"), s.tl))
                savefig!(data_figure(s), "$(key)_data.png")
            end
        end
    end

    sol = nothing                       # (no do-block inside a ternary)
    if init == "page"
        sol = soft_stage("Gaia DR3 solution for the starting point") do
            dr3_solution(; outdir)
        end
    end

    modelsd = Dict{Symbol,Any}(); chains = Dict{Symbol,Any}()
    logZ = Dict{Symbol,Float64}(); Λs = Dict{Symbol,Float64}()

    for kind in models
        tag = string(kind)
        sys, obs = stage("[$tag] Build system — $(MODEL_LABELS[kind])") do
            build_system(kind, src.tl)
        end
        DEBUG[] && display(sys)

        model = stage("[$tag] Compile LogDensityModel (verbosity = 4, as on the page)") do
            Octofitter.LogDensityModel(sys; verbosity=4)
        end
        DEBUG[] && display(model)
        modelsd[kind] = model

        init_chain = stage("[$tag] initialize! (mode = $init)") do
            initialize_model!(model, kind, init, sol; outdir)
        end
        soft_stage("[$tag] octoplot of the starting point") do
            savefig!(inout(() -> octoplot(model, init_chain)), "$(tag)_init_octoplot.png")
        end

        chain = stage("[$tag] Pigeons (n_rounds=$n_rounds, n_chains=$n_chains, no variational reference)") do
            c, pt = octofit_pigeons(model;
                n_chains = n_chains,
                n_rounds = n_rounds,
                n_chains_variational = 0,
                variational = nothing,
            )
            soft_stage("[$tag] Log-evidence ratio (stepping stone)") do
                zz = Pigeons.stepping_stone(pt)
                λ = Pigeons.global_barrier(pt)
                dbg(@sprintf("  [%s] log(Z₁/Z₀) ≈ %.3f, Λ = %.2f", tag, zz, λ))
                logZ[kind] = zz; Λs[kind] = λ
            end
            c
        end
        chains[kind] = chain
        DEBUG[] && display(chain)          # MCMCChains table: check rhat ≈ 1, ess
        soft_stage("[$tag] Results vs Stefansson et al. 2025") do
            print_results(chain, MODEL_LABELS[kind])
        end

        i_map = argmax(colvec(chain, :logpost))
        dbg("  MAP draw index = ", i_map, @sprintf(", logpost = %.3f", colvec(chain, :logpost)[i_map]))
        soft_stage("[$tag] octoplot") do
            savefig!(inout(() -> octoplot(model, chain)), "$(tag)_octoplot.png")
        end
        soft_stage("[$tag] octoplot of a single (MAP) draw vs the along-scan data") do
            savefig!(inout(() -> octoplot(Octofitter.PosteriorSeries(model, chain; ii=[i_map]))),
                     "$(tag)_octoplot_map.png")
        end
        soft_stage("[$tag] gaiastarplot (MAP draw)") do
            savefig!(inout(() -> Octofitter.gaiastarplot(model, chain, i_map)), "$(tag)_gaiastarplot.png")
        end
        soft_stage("[$tag] Period-quantile gaiastarplot grid (page)") do
            savefig!(quantile_grid(model, chain), "$(tag)_period_grid.png")
        end
        soft_stage("[$tag] Corner plot (small=true)") do
            savefig!(inout(() -> corner(model, chain; dark=corner_dark)), "$(tag)_corner.png")
        end
        soft_stage("[$tag] Key-marginal histograms") do
            savefig!(marginals_figure(chain, MODEL_LABELS[kind]), "$(tag)_marginals.png")
        end
        if nss && kind === first(models)
            soft_stage("[$tag] Gaia DR3 NSS solution: model/chain + comparison") do
                nss_sol = with_heartbeat("NSS query") do
                    inout(() -> query_nss(gaia_id=GAIA4_SOURCE_ID))
                end
                nss_sol === nothing && error("no NSS solution for this source")
                nss_model, nss_chain = nss_to_model_chain(nss_sol)
                dbg("  NSS chain: ", size(nss_chain))
                soft_stage("[$tag] NSS octoplot") do
                    savefig!(inout(() -> octoplot(nss_model, nss_chain)), "nss_octoplot.png")
                end
                savefig!(nss_compare_figure(chain, nss_chain), "$(tag)_vs_nss.png")
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

    # ---- Evidence ------------------------------------------------------------
    if !isempty(logZ)
        lines = String["Log evidence (stepping stone), n_rounds = $(n_rounds), n_chains = $(n_chains):"]
        for k in MODEL_ORDER
            haskey(logZ, k) || continue
            push!(lines, @sprintf("    %-34s log(Z₁/Z₀) = %10.3f   Λ = %6.2f", MODEL_LABELS[k],
                                  logZ[k], Λs[k]))
        end
        if haskey(logZ, :barycentric) && haskey(logZ, :anchored)
            # @sprintf needs ONE literal format string (no "a" * "b" concatenation)
            push!(lines, @sprintf("    difference anchored − barycentric = %.3f", logZ[:anchored] - logZ[:barycentric]) *
                         " (same data and likelihood; should be ≈ 0 up to sampling noise)")
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
    return (; data=src, models=modelsd, chains, logZ, Λ=Λs, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default when the REPL has 1 thread)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString="dr4_")
    T0[] = time()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTODR4_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTODR4_CHILD=1]")
    dbg("  Child output follows; its plots are shown here when it finishes.")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTODR4_RELAUNCH\"]=\"false\".")
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
    elseif envbool("OCTODR4_SHOW_PLOTS", "true")
        soft_stage("Show $(length(pngs)) saved plot(s) in the plot pane") do
            show_saved_pngs(pngs)
        end
    end
    return (; relaunched=true, success=ok, chains, outputs=new_files)
end

"Plot order for display: data first, then barycentric, anchored, NSS."
function png_rank(p)
    f = basename(p)
    occursin("_data", f) && return 0
    occursin("reduction", f) && return 1
    occursin("nss", f) && return 99
    m = startswith(f, "dr4_barycentric") ? 10 : startswith(f, "dr4_anchored") ? 30 : 50
    for (k, key) in enumerate(("init_octoplot", "octoplot_map", "octoplot", "gaiastarplot",
                               "period_grid", "corner", "marginals"))
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

"Parse OCTODR4_MODELS (comma-separated) into a tuple of Symbols in canonical order."
function models_from_env()
    s = get(ENV, "OCTODR4_MODELS", "barycentric")
    req = Set(Symbol(strip(lowercase(x))) for x in split(s, ',') if !isempty(strip(x)))
    issubset(req, Set(MODEL_ORDER)) ||
        error("OCTODR4_MODELS must list barycentric and/or anchored, got \"$s\"")
    return Tuple(k for k in MODEL_ORDER if k in req)
end

println(@sprintf("[DR4] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterDR4

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterDR4.jl`       -> runs
#   - VSCode "Execute active File in REPL"          -> runs; with 1 thread it
#     relaunches in `julia --threads=auto` (OCTODR4_RELAUNCH=false to disable,
#     OCTODR4_THREADS to set the count)
#   - ENV["OCTODR4_AUTORUN"] = "false"; include()   -> loads module only
#   - OCTODR4_MODELS, _ROUNDS, _CHAINS, _INIT, _NBOOT, _NSS, _OTHERS select what runs
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTODR4_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTODR4_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTODR4_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global dr4_result = OctofitterDR4.relaunch_threaded(
                this_file; threads=get(ENV, "OCTODR4_THREADS", "auto"))
            println("[DR4] Child run finished. `dr4_result` has fields: relaunched, success, ",
                    "chains (loaded from FITS), outputs")
        else
            global dr4_result = OctofitterDR4.run_dr4(
                models=OctofitterDR4.models_from_env(),
                show_plots=!is_child)
            is_child || println("[DR4] Result stored in `dr4_result` (fields: data, models, chains, ",
                "logZ, Λ, outputs)")
        end
    end
end

nothing
