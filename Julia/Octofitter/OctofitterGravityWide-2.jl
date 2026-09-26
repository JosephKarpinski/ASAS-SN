#=
================================================================================
 OctofitterGravityWide.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Fit GRAVITY-WIDE Data":
   https://sefffal.github.io/Octofitter.jl/dev/fit-grav-wide/

 GRAVITY-WIDE closure phases are mapped to non-redundant kernel phases (3 per
 spectral channel for the 4-telescope array), every channel modelled
 separately, with fibre-coupling losses (GRAVITYWideKPObs = InterferometryObs
 with kernel_phases=true, fiber_coupling=true). The page fits a single epoch
 with a fixed-position companion: b is a face-on circular "orbit" at
 separation `sep` [mas] and position angle `pa` at the exposure epoch.

 Data. The page's code is not executed by the docs build and reads a
 GRAVITY file it does not ship (./GRAVI.2025-01-01T00:11:11.111_dualscivis.fits).
 This module therefore has two modes:
   sim   (default) GRAVITY-WIDE-like exposures are simulated: VLT UT1–UT4
         station positions (6 baselines, the 4 closure triangles of GRAVITY's
         design), OCTOGW_NLAMBDA channels over the page's 2.025–2.15 μm window,
         OCTOGW_NEXP exposures at the page's epoch with the baselines rotated
         by the sky rotation during the night, and a companion of known
         contrast/separation/PA injected with the page's own model through
         generate_from_params (noise σ_CP per channel). Everything downstream
         is the page's workflow, and the fit is checked against the truth.
   file  OCTOGW_FILE=/path/to/GRAVI…dualscivis.fits (+ OCTOGW_EPOCH = mean MJD
         of the exposure) runs the page exactly as written on your own data.

 Model (as on the page): A mass = 1 M⊙, flux_K = 1 (host is a source);
 b: flux_K ~ U(0, 1), sep ~ U(0, 10) mas, pa ~ U(0, 2π), a = sep/plx, e = i =
 ω = Ω = 0, θ = pa, epoch = OBS_EPOCH (60676.00776748842); GRAVITYWideKPObs
 (targets (A, b), ref A, band :K, kp_jit ~ U(0, 180) deg, kp_Cy ~ U(−1, 1));
 plx = 173.5740 mas. Sampling: initialize!, then octofit_pigeons(n_chains=8,
 n_chains_variational=0, n_rounds=9, explorer=SliceSampler()) as on the page.
 Then the page's single-epoch grid search over (x, y) positions with fixed
 contrast(s), kernel-phase jitter and correlation (here: the posterior
 medians), and the page's map of the result.
 The page's closing "b_orbit" variant (a real Keplerian orbit) needs multiple
 epochs and is not run.

 Plots: + u-v coverage, + closure phases vs wavelength (data, and the model at
 the MAP draw), posterior positions on the sky (page), grid-search map (page,
 + truth and posterior overlaid), + contrast histogram, + corner (light).
 Log: parameter order check for ℓπcallback (page), truth comparison with
 percentiles and the share of draws within 1 mas of the true position (sim),
 the grid-search peak, Pigeons diagnostics.

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `gw_result`)
   Shell:    julia --threads=auto OctofitterGravityWide.jl
   Library:  ENV["OCTOGW_AUTORUN"] = "false"; include("OctofitterGravityWide.jl")
             res = OctofitterGravWide.run_gravwide()

 Threads: Pigeons and the grid search run multithreaded. In a 1-thread REPL
 the run is relaunched in a child `julia --threads=auto` (output streamed
 here; the chain is loaded back and the plots are shown here afterwards).
 OCTOGW_RELAUNCH=false disables this; OCTOGW_THREADS sets the count.

 Settings (ENV, also passed to a relaunched child):
   OCTOGW_FILE        = ""      — a real GRAVITY dualscivis OI-FITS (file mode)
   OCTOGW_EPOCH       = "60676.00776748842" — exposure mean MJD (page)
   OCTOGW_ROUNDS      = "9" (page)     OCTOGW_CHAINS = "8" (page)
   OCTOGW_FIBER       = "photocentre" (page default) | "host" (fiber_pointing = A,
                        the usual GRAVITY-WIDE setup per the page's note)
   OCTOGW_GRID_K      = "" — contrasts for the grid search (sim default: truth
                        and ×½, ×2; file default: the page's 0.5)
   OCTOGW_GRID_STEP   = "0.25" (page) — grid step [mas]
   Simulation (sim mode):
   OCTOGW_SEP  = "6.0" mas   OCTOGW_PA = "50" deg   OCTOGW_CONTRAST = "0.02"
   OCTOGW_SIGMA_CP = "0.7" deg per channel   OCTOGW_NLAMBDA = "20"
   OCTOGW_NEXP = "3"   OCTOGW_SEED = "42"
   OCTOGW_SHOW_PLOTS  = "true" — show the child's PNGs here

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOGW_ENV_MODE=local uses ./octofitter_v9_env). OctofitterInter-
 ferometry is unregistered and is added from the Octofitter repository at a
 pinned commit. Also needs CairoMakie, PairPlots, Distributions, Pigeons.

 Outputs (prefix "gw_") go to the directory containing this file.

 Changelog
 ---------
 v1.0.1 2026-09-26 Chain lookups resolve observation variables by suffix: the chain
                   stores kp_jit/kp_Cy as GRAVITY_WIDE_kp_jit/_kp_Cy (v1.0 failed
                   there after sampling). λ range printed with 3 decimals.
 v1.0  2026-09-26  Initial version on the OctofitterLikelihoodMap v1.0.1 /
                   OctofitterInterferometryTutorial v1.0.1 scaffolding ([GW
                   +t] debug stages, soft optional stages, env bootstrap with
                   v9 guards and a pinned unregistered-package install,
                   @__DIR__ outputs, dark theme with light corner plots,
                   threaded relaunch with exit-code-only failure report and
                   PNGs shown in the parent's plot pane, @eval-built
                   @variables, min(α) from Pigeons.swap_prs). Adds: simulated
                   GRAVITY-WIDE exposures with a known companion (the page
                   ships no data), file mode for real data, truth comparison,
                   MAP closure-phase model, parameter-order check.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let octo_commit = "8c6daaaf5292b6cbdb0aa4f4b1e9a1a5f0bd21f3",
    required = ("Octofitter", "OctofitterInterferometry", "CairoMakie", "PairPlots",
                "Distributions", "Pigeons"),
    min_version = Dict("Octofitter" => v"9"),
    mode = lowercase(get(ENV, "OCTOGW_ENV_MODE", "temp"))

    direct_deps() = Dict(info.name => info.version
                         for info in values(Pkg.dependencies()) if info.is_direct_dep)
    missing_deps() = [n for n in required if !haskey(direct_deps(), n)]
    function versions_ok()
        d = direct_deps()
        all(!haskey(d, n) || (d[n] !== nothing && d[n] >= v) for (n, v) in min_version)
    end
    env_ok() = isempty(missing_deps()) && versions_ok()
    function spec(n)
        n == "OctofitterInterferometry" &&
            return Pkg.PackageSpec(url="https://github.com/sefffal/Octofitter.jl",
                                   subdir="OctofitterInterferometry", rev=octo_commit)
        haskey(min_version, n) ? Pkg.PackageSpec(name=n, version="9") : Pkg.PackageSpec(name=n)
    end
    note(miss) = "OctofitterInterferometry" in miss &&
        println("[GW] OctofitterInterferometry is unregistered: cloning it from the Octofitter ",
                "repository (commit ", first(octo_commit, 8), "); this takes a minute or two the first time.")

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
            println("[GW] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[GW] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            note(miss)
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[GW] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[GW] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[GW] Existing v9 environment found at ", Base.active_project())
        else
            note(missing_deps())
            Pkg.add(spec.(collect(required)))
        end
        println("[GW] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[GW] Loading Octofitter, OctofitterInterferometry, Pigeons, CairoMakie, PairPlots, ",
        "Distributions (first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterGravWide

const LOAD_T0 = time()

using Octofitter
using OctofitterInterferometry
using CairoMakie
using PairPlots
using Distributions
using Pigeons
import Random
import Statistics
using Printf

export run_gravwide, set_plot_theme!

const VERSION_STRING = "1.0.1"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterGravWide needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const PREFIX = "gw_"
const PLX = 173.5740                         # mas (page)
const λ_MIN, λ_MAX = 2.025e-6, 2.15e-6       # page's wavelength cut [m]

# VLT unit-telescope stations (east, north) [m]; baselines in GRAVITY's order
# (1,2) (1,3) (1,4) (2,3) (2,4) (3,4); the 4 closure triangles match
# GRAVITY_T3_DESIGN (the package tests use the same index vectors).
const UT_EN = ((-9.925, -20.335), (14.887, 30.502), (44.915, 66.183), (103.306, 43.999))
const PAIRS = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
const IDX_CPS1 = [1, 1, 2, 4]
const IDX_CPS2 = [4, 5, 6, 6]
const IDX_CPS3 = [2, 3, 3, 5]

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterGravWide.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[GW +%7.1fs] ", time() - T0[]), msg...)
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
envfloat(k, default) = parse(Float64, strip(get(ENV, k, default)))
function envopt(k)
    s = strip(get(ENV, k, ""))
    return isempty(s) ? nothing : String(s)
end

"All settings, read once."
function config_from_env()
    file = envopt("OCTOGW_FILE")
    kgrid = envopt("OCTOGW_GRID_K")
    return (;
        file,
        mode       = file === nothing ? :sim : :file,
        epoch      = envfloat("OCTOGW_EPOCH", "60676.00776748842"),
        n_rounds   = envint("OCTOGW_ROUNDS", "9"),
        n_chains   = envint("OCTOGW_CHAINS", "8"),
        fiber      = lowercase(get(ENV, "OCTOGW_FIBER", "photocentre")),
        grid_k     = kgrid === nothing ? nothing :
                     [parse(Float64, strip(x)) for x in split(kgrid, ',') if !isempty(strip(x))],
        grid_step  = envfloat("OCTOGW_GRID_STEP", "0.25"),
        sep        = envfloat("OCTOGW_SEP", "6.0"),
        pa_deg     = envfloat("OCTOGW_PA", "50"),
        contrast   = envfloat("OCTOGW_CONTRAST", "0.02"),
        σ_cp       = envfloat("OCTOGW_SIGMA_CP", "0.7"),
        n_λ        = envint("OCTOGW_NLAMBDA", "20"),
        n_exp      = envint("OCTOGW_NEXP", "3"),
        seed       = envint("OCTOGW_SEED", "42"),
    )
end

function print_env_info(cfg, outdir)
    DEBUG[] || return nothing
    dbg("OctofitterGravWide v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | OctofitterInterferometry ", something(pkgversion(OctofitterInterferometry), "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    if cfg.mode === :sim
        dbg("mode = sim (the page ships no GRAVITY file): ", cfg.n_exp, " exposure(s) × ", cfg.n_λ,
            " channels over ", @sprintf("%.3f–%.3f", 1e6λ_MIN, 1e6λ_MAX), " μm; injected companion: contrast ",
            cfg.contrast, ", sep ", cfg.sep, " mas, PA ", cfg.pa_deg, "°; σ_CP = ", cfg.σ_cp,
            "° per channel; seed ", cfg.seed)
    else
        dbg("mode = file: ", cfg.file, " at MJD ", cfg.epoch, " (page)")
    end
    dbg("Pigeons: n_chains = ", cfg.n_chains, ", n_rounds = ", cfg.n_rounds,
        " (page: 8, 9), SliceSampler, no variational leg | fiber_pointing = ",
        cfg.fiber == "host" ? "A (host)" : "Photocentre(:K) (page default)")
    Threads.nthreads() == 1 &&
        dbg("  NOTE: running on 1 thread (relaunch disabled) — expect a long run")
    dbg("project    = ", Base.active_project())
    dbg("SCRIPT_DIR = ", SCRIPT_DIR, " | outdir = ", abspath(outdir))
    dbg("interactive = ", isinteractive(),
        " | Makie backend = ", CairoMakie.Makie.current_backend())
    return nothing
end

# ---- Chain helpers -----------------------------------------------------------
# Observation-level variables are stored with the observation's name as a prefix
# ("GRAVITY-WIDE" → GRAVITY_WIDE_kp_jit), so resolve short names by suffix.
function colname(chain, p::Symbol)
    ns = names(chain)
    p in ns && return p
    hits = filter(n -> endswith(String(n), "_" * String(p)), ns)
    length(hits) == 1 && return only(hits)
    error("chain column $p not found (candidates: $(hits); columns: $(ns))")
end
colvec(chain, p::Symbol) = vec(Array(chain[colname(chain, p)]))
haspar(chain, p::Symbol) = try colname(chain, p); true catch; false end
qline(x) = (y = filter(isfinite, x); isempty(y) ? (NaN, NaN, NaN) :
                                     Statistics.quantile(y, (0.16, 0.5, 0.84)))

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
const COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
"""
One simulated GRAVITY-WIDE exposure: UT baselines rotated by `rot` degrees
(the sky rotation between exposures), `n_λ` channels over the page's window,
placeholder closure phases (replaced by generate_from_params) with σ_cp.
"""
function sim_exposure(epoch, rot, n_λ, σ_cp)
    eff_wave = collect(range(λ_MIN, λ_MAX, length=n_λ))
    c, s = cosd(rot), sind(rot)
    u = zeros(length(PAIRS), n_λ)
    v = zeros(length(PAIRS), n_λ)
    for (b, (i, j)) in enumerate(PAIRS)
        dE = UT_EN[j][1] - UT_EN[i][1]
        dN = UT_EN[j][2] - UT_EN[i][2]
        E, N = c * dE - s * dN, s * dE + c * dN
        for l in 1:n_λ
            u[b, l] = E / eff_wave[l]          # inverse wavelengths
            v[b, l] = N / eff_wave[l]
        end
    end
    return (; epoch, eff_wave, u, v,
              cps_data=zeros(4, n_λ), dcps=fill(σ_cp, 4, n_λ),
              index_cps1=IDX_CPS1, index_cps2=IDX_CPS2, index_cps3=IDX_CPS3,
              use_vis2=false, jitter=:kp_jit, kp_Cy=:kp_Cy)
end

"The observation's input: simulated exposures, or the page's file row."
function data_input(cfg)
    if cfg.mode === :sim
        rots = cfg.n_exp == 1 ? [0.0] : collect(range(-20.0, 20.0, length=cfg.n_exp))
        # exposures a few minutes apart around the page's epoch (position unchanged)
        eps = cfg.epoch .+ (rots ./ 360) .* 0.5
        rows = [sim_exposure(eps[k], rots[k], cfg.n_λ, cfg.σ_cp) for k in eachindex(rots)]
        for (k, r) in enumerate(rows)
            B = [hypot(r.u[b, 1], r.v[b, 1]) * r.eff_wave[1] for b in eachindex(PAIRS)]
            dbg(@sprintf("  exposure %d: MJD %.5f, rotation %+.1f°, baselines %.1f–%.1f m, %d channels",
                         k, r.epoch, rots[k], minimum(B), maximum(B), length(r.eff_wave)))
        end
        return Table(rows)
    else
        isfile(cfg.file) || error("OCTOGW_FILE = \"$(cfg.file)\" does not exist")
        return Table([(;
            filename = abspath(cfg.file),
            epoch = cfg.epoch,
            wavelength_min_meters = λ_MIN,
            wavelength_max_meters = λ_MAX,
            jitter = :kp_jit,
            kp_Cy = :kp_Cy,
        )])
    end
end

# -----------------------------------------------------------------------------
# Model (page)
# -----------------------------------------------------------------------------
function build_system(input, cfg)
    A_vars = @eval @variables begin
        mass = 1.0            # M⊙
        flux_K = 1.0          # the host is an ordinary source; pinning it to 1
    end                       # makes every other body's flux a contrast ratio
    A = Body(name="A", variables=A_vars)
    ep = cfg.epoch
    b_vars = @eval @variables begin
        flux_K ~ Uniform(0, 1)   # contrast ratio against the host
        sep ~ Uniform(0, 10)     # mas
        pa ~ Uniform(0, 2pi)     # radians
        # A face-on circular orbit placed at (sep, pa) at the epoch of the
        # exposure: a fixed position, written as a degenerate orbit.
        a = sep / system.plx
        e = 0.0
        i = 0.0
        ω = 0.0
        Ω = 0.0
        θ = pa
        epoch = $ep
    end
    b = Body(name="b", about=A, variables=b_vars)
    obs_vars = @eval @variables begin
        kp_jit ~ Uniform(0, 180)   # kernel phase jitter, degrees
        kp_Cy  ~ Uniform(-1, 1)    # spectral correlation parameter
    end
    vis_obs = if cfg.fiber == "host"
        GRAVITYWideKPObs(input; targets=(A, b), ref=A, band=:K, fiber_pointing=A,
                         variables=obs_vars)
    else
        GRAVITYWideKPObs(input; targets=(A, b), ref=A, band=:K, variables=obs_vars)
    end
    pp = PLX
    sys_vars = @eval @variables begin
        plx = $pp
    end
    return System(name="sys", bodies=[A, b], observations=[vis_obs], variables=sys_vars)
end

"""
The page's statement: ℓπcallback's flat vector is (b_flux_K, b_sep, b_pa,
kp_jit, kp_Cy). Checked here by round-tripping a vector through arr2nt.
"""
function check_order(model)
    probe = [0.123, 4.56, 1.234, 7.89, 0.321]
    length(model.sample_priors(Random.Xoshiro(0))) == 5 ||
        error("expected 5 free parameters, got $(length(model.sample_priors(Random.Xoshiro(0))))")
    nt = model.arr2nt(probe)
    bb = nt.bodies.b
    obs = first(values(nt.observations))
    got = (bb.flux_K, bb.sep, bb.pa, obs.kp_jit, obs.kp_Cy)
    ok = all(isapprox.(got, Tuple(probe)))
    dbg("  flat order (b_flux_K, b_sep, b_pa, kp_jit, kp_Cy): ", ok ? "confirmed ✓" :
        "DIFFERENT — arr2nt gave $(got) for $(probe)")
    ok || error("parameter order differs from the page; the grid search would be wrong")
    return true
end

function describe_obs(obs)
    t = obs.table
    for k in eachindex(t.epoch)
        cp = t.cps_data[k]
        dbg(@sprintf("  exposure %d: MJD %.5f | %d baselines × %d channels (%.3f–%.3f μm) | %d closure phases, rms %.2f°, median σ %.2f°",
                     k, t.epoch[k], size(t.u[k], 1), size(t.u[k], 2),
                     1e6 * minimum(t.eff_wave[k]), 1e6 * maximum(t.eff_wave[k]), length(cp),
                     sqrt(Statistics.mean(abs2, cp)), Statistics.median(t.dcps[k])))
        χ² = sum(abs2, vec(cp) ./ vec(t.dcps[k])); n = length(cp)
        dbg(@sprintf("             χ² vs no companion = %.1f for %d CPs (p = %.2g)", χ², n,
                     ccdf(Chisq(n), χ²)))
    end
end

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
function uv_figure(obs)
    fig = Figure(size=(700, 650))
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="u [Mλ]", ylabel="v [Mλ]",
              title="u-v coverage (all channels)")
    for k in eachindex(obs.table.epoch)
        u = vec(obs.table.u[k]) ./ 1e6; v = vec(obs.table.v[k]) ./ 1e6
        c = COLORS[mod1(k, length(COLORS))]
        scatter!(ax, vcat(u, -u), vcat(v, -v); color=(c, 0.8), markersize=4, label="exposure $k")
    end
    axislegend(ax, position=:rt, unique=true)
    return fig
end

"Closure phases vs wavelength per triangle; + the model at the MAP draw."
function cp_figure(obs; model_obs=nothing)
    t = obs.table
    fig = Figure(size=(1200, 700))
    for tri in 1:4
        ax = Axis(fig[fldmod1(tri, 2)...], xlabel="λ [μm]", ylabel="closure phase [deg]",
                  title="triangle $tri")
        for k in eachindex(t.epoch)
            c = COLORS[mod1(k, length(COLORS))]
            λ = 1e6 .* t.eff_wave[k]
            errorbars!(ax, λ, t.cps_data[k][tri, :], t.dcps[k][tri, :]; color=(c, 0.5))
            scatter!(ax, λ, t.cps_data[k][tri, :]; color=c, markersize=5, label="exposure $k")
            if model_obs !== nothing
                lines!(ax, λ, model_obs.table.cps_data[k][tri, :]; color=c, linewidth=2)
            end
        end
        tri == 1 && axislegend(ax, position=:lt, labelsize=10, unique=true)
    end
    Label(fig[0, 1:2], model_obs === nothing ? "Closure phases (data)" :
                       "Closure phases: data (points) and the model at the MAP draw (lines)", fontsize=15)
    return fig
end

"The page's posterior figure: positions on the sky, + the injected truth."
function sky_figure(chain; truth=nothing)
    fig = Figure(size=(700, 650))
    ax = Axis(fig[1, 1], xreversed=true, autolimitaspect=1, xlabel="Δα* [mas]", ylabel="Δδ [mas]",
              title="Posterior position of b")
    xlims!(ax, 10, -10)
    ylims!(ax, -10, 10)
    x = colvec(chain, :b_sep) .* sin.(colvec(chain, :b_pa))
    y = colvec(chain, :b_sep) .* cos.(colvec(chain, :b_pa))
    scatter!(ax, x, y; color=(COLORS[1], 0.5), markersize=4)
    scatter!(ax, [0.0], [0.0]; marker=:star5, color=:white, markersize=14)
    if truth !== nothing
        scatter!(ax, [truth[1]], [truth[2]]; marker=:circle, color=:transparent,
                 strokecolor=COLORS[2], strokewidth=2, markersize=18, label="injected")
        axislegend(ax, position=:rt)
    end
    return fig
end

"The page's grid-search map (max over the contrast grid), + truth and posterior."
function grid_figure(xs, ys, M, title; truth=nothing, post=nothing)
    fig = Figure(size=(800, 700))
    ax = Axis(fig[1, 1], xreversed=true, autolimitaspect=1, backgroundcolor="#222",
              xlabel="Δα* [mas]", ylabel="Δδ [mas]", title=title)
    fin = filter(isfinite, M)
    h = heatmap!(ax, xs, ys, M; colormap=:magma,
                 colorrange=(Statistics.quantile(fin, 0.85), maximum(fin)))
    if post !== nothing
        scatter!(ax, post[1], post[2]; color=(:cyan, 0.25), markersize=2)
    end
    if truth !== nothing
        scatter!(ax, [truth[1]], [truth[2]]; marker=:circle, color=:transparent,
                 strokecolor=:white, strokewidth=2, markersize=18)
    end
    Colorbar(fig[1, 2], h, label="log posterior density (max over contrast grid)")
    return fig
end

function flux_figure(chain; truth=nothing)
    f = colvec(chain, :b_flux_K)
    fig = Figure(size=(900, 440))
    ax = Axis(fig[1, 1], xlabel="b_flux_K (contrast to the host)", ylabel="density",
              title="Companion contrast")
    hist!(ax, f; bins=50, normalization=:pdf, color=(COLORS[1], 0.85))
    truth === nothing || vlines!(ax, [truth]; color=COLORS[2], linewidth=2)
    return fig
end

# -----------------------------------------------------------------------------
# Grid search (page)
# -----------------------------------------------------------------------------
"""
The page's detection map: log posterior at every (x, y) on a grid for each
contrast in `ks`, with the kernel-phase jitter and correlation held fixed.
"""
function grid_search(model, ks, jit, Cy; step=0.25)
    xs = (-10:step:10) .+ 1e-6
    ys = (-10:step:10)
    LL = fill(NaN, length(xs), length(ys), length(ks))
    ci = CartesianIndices(LL)
    Threads.@threads for n in eachindex(LL)
        I = ci[n]
        x, y, K = xs[I[1]], ys[I[2]], ks[I[3]]
        sep = hypot(x, y)
        sep > 10 && continue
        pa = rem2pi(atan(x, y), RoundDown)
        θ = model.link([K, sep, pa, jit, Cy])
        LL[n] = model.ℓπcallback(θ; sampled=false)
    end
    return xs, ys, LL
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_gravwide(; cfg=config_from_env(), outdir=SCRIPT_DIR, prefix="gw_",
                   dark=true, show_plots=true)
"""
function run_gravwide(; cfg=config_from_env(),
                        outdir::AbstractString=SCRIPT_DIR,
                        prefix::AbstractString=PREFIX,
                        dark::Bool=true,
                        show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    print_env_info(cfg, outdir)
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

    # ---- Data and model -------------------------------------------------------------
    input = stage(cfg.mode === :sim ? "Simulated GRAVITY-WIDE exposures (UT1–UT4 geometry)" :
                                      "GRAVITY OI-FITS input row (page)") do
        data_input(cfg)
    end
    sys0 = stage("Bodies, GRAVITYWideKPObs (kernel phases, fibre coupling), System (page)") do
        build_system(input, cfg)
    end
    model0 = stage("Compile LogDensityModel (page: verbosity=4)") do
        Octofitter.LogDensityModel(sys0; verbosity=4)
    end
    stage("Check the flat parameter order used by the grid search (page)") do
        check_order(model0)
    end

    truth_xy = nothing
    sys, model = sys0, model0
    if cfg.mode === :sim
        pa = deg2rad(cfg.pa_deg)
        truth_xy = (cfg.sep * sin(pa), cfg.sep * cos(pa))
        sys = stage("Inject the companion: generate_from_params(sys, θ_truth; add_noise=true)") do
            θ = model0.arr2nt([cfg.contrast, cfg.sep, pa, 0.0, 0.0])
            Random.seed!(cfg.seed)
            generate_from_params(sys0, θ; add_noise=true)
        end
        dbg(@sprintf("  injected: contrast %.4g (Δmag %.2f), sep %.2f mas, PA %.1f° → (Δα*, Δδ) = (%.2f, %.2f) mas",
                     cfg.contrast, -2.5log10(cfg.contrast), cfg.sep, cfg.pa_deg, truth_xy...))
        model = stage("Compile LogDensityModel for the simulated data") do
            Octofitter.LogDensityModel(sys; verbosity=4)
        end
    end
    obs = sys.observations[1]
    describe_obs(obs)
    DEBUG[] && display(sys)
    soft_stage("u-v coverage") do
        savefig!(uv_figure(obs), "uv_coverage.png")
    end
    soft_stage("Closure phases vs wavelength (data)") do
        savefig!(cp_figure(obs), "closure_phases.png")
    end

    stage("initialize!(model) (page)") do
        Octofitter.initialize!(model)
    end

    # ---- Sampling (page) -------------------------------------------------------------
    chain, pt = stage("octofit_pigeons (n_chains=$(cfg.n_chains), n_chains_variational=0, " *
                      "n_rounds=$(cfg.n_rounds), SliceSampler; page)") do
        octofit_pigeons(model; n_chains=cfg.n_chains, n_chains_variational=0,
                        n_rounds=cfg.n_rounds, explorer=Pigeons.SliceSampler())
    end
    diag = soft_stage("Pigeons diagnostics") do
        z = Pigeons.stepping_stone(pt); λ = Pigeons.global_barrier(pt)
        mα = minimum(Pigeons.swap_prs(pt))
        dbg(@sprintf("  log(Z₁/Z₀) ≈ %.3f | Λ = %.2f | min(α) = %.3g%s", z, λ, mα,
                     mα < 1e-3 ? "  ⚠ chains barely swap: add n_chains or rounds" : ""))
        (; logZ=z, Λ=λ, min_α=mα)
    end
    DEBUG[] && display(chain)

    # ---- Summary and truth ---------------------------------------------------------------
    lines = String[]
    x = colvec(chain, :b_sep) .* sin.(colvec(chain, :b_pa))
    y = colvec(chain, :b_sep) .* cos.(colvec(chain, :b_pa))
    rows = (("contrast b_flux_K", colvec(chain, :b_flux_K), cfg.mode === :sim ? cfg.contrast : NaN),
            ("sep [mas]", colvec(chain, :b_sep), cfg.mode === :sim ? cfg.sep : NaN),
            ("PA [deg]", rad2deg.(colvec(chain, :b_pa)), cfg.mode === :sim ? cfg.pa_deg : NaN),
            ("Δα* [mas]", x, truth_xy === nothing ? NaN : truth_xy[1]),
            ("Δδ [mas]", y, truth_xy === nothing ? NaN : truth_xy[2]),
            ("kp_jit [deg]", colvec(chain, :kp_jit), cfg.mode === :sim ? 0.0 : NaN),
            ("kp_Cy", colvec(chain, :kp_Cy), cfg.mode === :sim ? 0.0 : NaN))
    push!(lines, "Posterior median [16th, 84th]" * (cfg.mode === :sim ? " vs the injected truth:" : ":"))
    for (lab, v, tv) in rows
        q = qline(v)
        if isfinite(tv)
            pct = 100 * Statistics.mean(v .< tv)
            flag = (pct < 2.5 || pct > 97.5) ? "  ← truth outside the central 95%" : ""
            push!(lines, @sprintf("    %-18s %10.4g  [%10.4g, %10.4g]   truth %8.4g (at %.1f%%)%s",
                                  lab, q[2], q[1], q[3], tv, pct, flag))
        else
            push!(lines, @sprintf("    %-18s %10.4g  [%10.4g, %10.4g]", lab, q[2], q[1], q[3]))
        end
    end
    if truth_xy !== nothing
        d = hypot.(x .- truth_xy[1], y .- truth_xy[2])
        push!(lines, @sprintf("    %.1f%% of draws within 1 mas of the injected position (median offset %.2f mas)",
                              100 * Statistics.mean(d .< 1), Statistics.median(d)))
        push!(lines, "    (kp_jit truth is 0: the simulated noise is σ_CP only, at the prior's lower edge)")
    end
    f = colvec(chain, :b_flux_K)
    push!(lines, @sprintf("    contrast SNR (mean/std) = %.1f", Statistics.mean(f) / Statistics.std(f)))
    foreach(l -> dbg(l), lines)

    p = joinpath(outdir, prefix * "chain.fits")
    soft_stage("Save chain -> FITS (+ reload check)") do
        Octofitter.savechain(p, chain)
        push!(outputs, p)
        c2 = Octofitter.loadchain(p; model)
        dbg("  reloaded chain size = ", size(c2), size(c2) == size(chain) ? "  (matches)" : "  (DIFFERS)")
    end

    # ---- Plots (page + extras) ------------------------------------------------------------
    soft_stage("Posterior position of b on the sky (page)") do
        savefig!(sky_figure(chain; truth=truth_xy), "sky.png")
    end
    soft_stage("Contrast histogram") do
        savefig!(flux_figure(chain; truth=cfg.mode === :sim ? cfg.contrast : nothing), "contrast.png")
    end
    soft_stage("Corner plot (light theme)") do
        savefig!(inout(() -> light(() -> octocorner(model, chain))), "corner.png")
    end
    soft_stage("Closure phases with the model at the MAP draw") do
        lp = colvec(chain, :logpost)
        θmap = Octofitter.mcmcchain2result(model, chain, argmax(lp))
        msys = generate_from_params(sys, θmap; add_noise=false)
        savefig!(cp_figure(obs; model_obs=msys.observations[1]), "closure_phases_model.png")
    end

    # ---- Grid search (page) ----------------------------------------------------------------
    grid = soft_stage("Grid search over positions (page; threaded)") do
        ks = cfg.grid_k !== nothing ? cfg.grid_k :
             cfg.mode === :sim ? [cfg.contrast / 2, cfg.contrast, 2cfg.contrast] : [0.5]
        jit = Statistics.median(colvec(chain, :kp_jit))
        Cy = Statistics.median(colvec(chain, :kp_Cy))
        dbg(@sprintf("  contrasts %s | kp_jit = %.3g°, kp_Cy = %.3g (posterior medians; the page fixed 12.6 and 0.02 for its data)",
                     string(ks), jit, Cy))
        t = time()
        xs, ys, LL = grid_search(model, ks, jit, Cy; step=cfg.grid_step)
        dbg(@sprintf("  %d evaluations in %.1f s (%.3f ms each, %d threads)", length(LL), time() - t,
                     1000 * (time() - t) * Threads.nthreads() / length(LL), Threads.nthreads()))
        M = maximum(LL, dims=3)[:, :]
        I = argmax(replace(M, NaN => -Inf))
        kbest = ks[argmax(LL[I[1], I[2], :])]
        dbg(@sprintf("  grid peak at (Δα*, Δδ) = (%.2f, %.2f) mas, sep %.2f, PA %.1f°, contrast %.3g",
                     xs[I[1]], ys[I[2]], hypot(xs[I[1]], ys[I[2]]),
                     mod(rad2deg(atan(xs[I[1]], ys[I[2]])), 360), kbest))
        truth_xy === nothing ||
            dbg(@sprintf("  offset of the grid peak from the injected position: %.2f mas",
                         hypot(xs[I[1]] - truth_xy[1], ys[I[2]] - truth_xy[2])))
        savefig!(grid_figure(xs, ys, M, @sprintf("Grid search: kp_jit %.2g°, kp_Cy %.2g", jit, Cy);
                             truth=truth_xy, post=(x, y)), "grid_map.png")
        (; xs, ys, LL, ks)
    end

    if diag !== nothing
        push!(lines, @sprintf("Pigeons: log(Z₁/Z₀) = %.3f, Λ = %.2f, min(α) = %.3g", diag.logZ, diag.Λ, diag.min_α))
    end
    soft_stage("Write summary") do
        ps = joinpath(outdir, prefix * "summary.txt")
        write(ps, join(lines, "\n") * "\n")
        push!(outputs, ps)
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f min", (time() - T0[]) / 60))
    return (; model, chain, pt, sys, grid, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default when the REPL has 1 thread)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString=PREFIX)
    T0[] = time()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOGW_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOGW_CHILD=1]")
    dbg("  Child output follows; its plots are shown here when it finishes.")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOGW_RELAUNCH\"]=\"false\".")
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
    elseif lowercase(get(ENV, "OCTOGW_SHOW_PLOTS", "true")) != "false"
        soft_stage("Show $(length(pngs)) saved plot(s) in the plot pane") do
            show_saved_pngs(pngs)
        end
    end
    return (; relaunched=true, success=ok, chain, outputs=new_files)
end

function png_rank(p)
    f = basename(p)
    for (k, key) in enumerate(("uv_coverage", "closure_phases.png", "sky", "grid_map", "contrast",
                               "closure_phases_model", "corner"))
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

println(@sprintf("[GW] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterGravWide

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterGravityWide.jl` -> runs
#   - VSCode "Execute active File in REPL" -> runs; with 1 thread it relaunches
#     in `julia --threads=auto` (OCTOGW_RELAUNCH=false to disable,
#     OCTOGW_THREADS to set the count)
#   - ENV["OCTOGW_AUTORUN"] = "false"; include() -> loads module only
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOGW_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOGW_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOGW_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global gw_result = OctofitterGravWide.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOGW_THREADS", "auto"))
            println("[GW] Child run finished. `gw_result` has fields: relaunched, success, ",
                    "chain (loaded from FITS), outputs")
        else
            global gw_result = OctofitterGravWide.run_gravwide(show_plots=!is_child)
            is_child || println("[GW] Result stored in `gw_result` (fields: model, chain, pt, ",
                                "sys, grid, outputs)")
        end
    end
end

nothing
