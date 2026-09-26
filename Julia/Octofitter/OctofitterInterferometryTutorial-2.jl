#=
================================================================================
 OctofitterInterferometryTutorial.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Fitting Interferometric
 Observables":
   https://sefffal.github.io/Octofitter.jl/dev/fit-interfere/

 Three simulated JWST NIRISS-AMI exposures (OI-FITS; the docs' examples/
 AMI_data, dated 2023-06-01, 2023-08-15, 2024-06-01) are fitted with a point-
 source model: V(u,v) = Σ f_j exp(−2πi(u Δα*_j + v Δδ_j)) / Σ f_j, using
 closure phases (use_vis2 = false, as on the page).

 Model (as on the page): host A (mass 1.5 ± 0.01 M⊙, flux_K = 1, an ordinary
 source in the sum); companion b (massless, flux_K ~ N(0, 0.1) truncated at 0
 = contrast, a ~ N(2, 0.1) AU, e ~ N(0, 0.05) truncated to 0–0.9, i ~ Sine,
 ω, Ω, θ ~ UniformCircular, θ epoch MJD 60171); InterferometryObs
 "NIRISS-AMI" with targets = (A, b), ref = A, band = :K, platescale = 1,
 northangle = 0, σ_cp_jitter = 0; system "Tutoria", plx ~ N(100, 0.1) mas.
 Sampling: initialize!, then octofit_pigeons(n_rounds=10) as on the page.

 Plots (page, plus extras marked +): closure phases per epoch, +u-v coverage,
 flux histogram (+ with the prior), octoplot, positions of b at each epoch,
 corner (page's full octocorner, + a small=true version with flux_K).
 Log: the page's SNR = mean/std of b_flux_K, + median/IQR and what the prior
 alone gives, + separation and position angle of b at each epoch.

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `ifm_result`)
   Shell:    julia --threads=auto OctofitterInterferometryTutorial.jl
   Library:  ENV["OCTOIFM_AUTORUN"] = "false"; include("OctofitterInterferometryTutorial.jl")
             res = OctofitterInterfere.run_interfere()

 Threads: Pigeons runs multithreaded. In a 1-thread REPL the run is relaunched
 in a child `julia --threads=auto` (output streamed here; the chain is loaded
 back and the saved plots are shown in the plot pane afterwards).
 OCTOIFM_RELAUNCH=false disables this; OCTOIFM_THREADS sets the count.

 Settings (ENV, also passed to a relaunched child):
   OCTOIFM_ROUNDS      = "10" (page) — Pigeons n_rounds (2^n samples)
   OCTOIFM_CHAINS      = ""   (Pigeons default) — n_chains
   OCTOIFM_VIS2        = "false" (page) — also fit squared visibilities
   OCTOIFM_SHOW_PLOTS  = "true" — show the child's PNGs here

 Data: the three OI-FITS files (43 kB each) are downloaded once from the
 Octofitter repository (pinned commit) next to this file.

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOIFM_ENV_MODE=local uses ./octofitter_v9_env). OctofitterInter-
 ferometry is unregistered and is added with
   Pkg.PackageSpec(url="https://github.com/sefffal/Octofitter.jl",
                   subdir="OctofitterInterferometry", rev=<pinned commit>)
 Also needs CairoMakie, PairPlots, Distributions, Pigeons.

 Note: file and module are not named `OctofitterInterferometry`, which is
 the name of the package they load. Module: `OctofitterInterfere`.

 Outputs (prefix "ifm_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-26  Initial version on the OctofitterMassPhotometry v1.0.1 /
                   OctofitterImagesTutorial v1.0.1 scaffolding ([IFM +t]
                   debug stages, soft optional stages, env bootstrap with v9
                   guards and a pinned unregistered-package install, @__DIR__
                   outputs, dark theme with light corner plots, threaded
                   relaunch with exit-code-only failure report and PNGs shown
                   in the parent's plot pane, @eval-built @variables, min(α)
                   from Pigeons.swap_prs). Adds: absolute OI-FITS paths, u-v
                   coverage, flux prior overlay, per-epoch separation/PA,
                   small corner with the contrast.
 v1.0.1 2026-09-26 First run (2.8 min, 4 threads): contrast 4.6e-4 ± 0.6e-4
                   (Δmag 8.35, page SNR 7.2), a 2.07 AU, e 0.03, i 34°, P 2.4 yr,
                   min(α) 0.55, Λ 6.1, log(Z₁/Z₀) 176. Fixes: flux figure now
                   two panels (the prior made a linear axis 100× too wide);
                   u-v markers sized per epoch (the simulated epochs share
                   identical coverage, so later ones hid earlier ones) and the
                   log says so; the per-epoch "rms/σ ≫ 1" line replaced by a
                   χ² test against no companion; dates without T00:00:00.
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
    mode = lowercase(get(ENV, "OCTOIFM_ENV_MODE", "temp"))

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
        println("[IFM] OctofitterInterferometry is unregistered: cloning it from the Octofitter ",
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
            println("[IFM] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[IFM] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            note(miss)
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[IFM] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[IFM] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[IFM] Existing v9 environment found at ", Base.active_project())
        else
            note(missing_deps())
            Pkg.add(spec.(collect(required)))
        end
        println("[IFM] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[IFM] Loading Octofitter, OctofitterInterferometry, Pigeons, CairoMakie, PairPlots, ",
        "Distributions (first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterInterfere

const LOAD_T0 = time()

using Octofitter
using OctofitterInterferometry
using CairoMakie
using PairPlots
using Distributions
using Pigeons
import Downloads
import Statistics
using Printf

export run_interfere, set_plot_theme!

const VERSION_STRING = "1.0.1"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterInterfere needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const PREFIX = "ifm_"

const OCTO_COMMIT = "8c6daaaf5292b6cbdb0aa4f4b1e9a1a5f0bd21f3"
const OCTO_RAW = "https://raw.githubusercontent.com/sefffal/Octofitter.jl/$(OCTO_COMMIT)/"

# The page's three exposures (file, date)
const EXPOSURES = (
    ("Sim_data_2023_1_.oifits", "2023-06-01"),
    ("Sim_data_2023_2_.oifits", "2023-08-15"),
    ("Sim_data_2024_1_.oifits", "2024-06-01"),
)

const PRIORS = (
    mass_A = truncated(Normal(1.5, 0.01), lower=0.1),     # M⊙
    flux_K = truncated(Normal(0, 0.1), lower=0),           # contrast to the host
    a      = truncated(Normal(2, 0.1), lower=0.1),         # AU
    e      = truncated(Normal(0, 0.05), lower=0, upper=0.90),
    θ_ep   = 60171.0,                                      # MJD
    plx    = truncated(Normal(100.0, 0.1), lower=0.1),     # mas
)

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterInterfere.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[IFM +%7.1fs] ", time() - T0[]), msg...)
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

envbool(k, default) = lowercase(strip(get(ENV, k, default))) in ("true", "1", "yes")
envint(k, default) = parse(Int, strip(get(ENV, k, default)))
function envopt(k)
    s = strip(get(ENV, k, ""))
    return isempty(s) ? nothing : String(s)
end

function print_env_info(outdir, n_rounds, n_chains, vis2)
    DEBUG[] || return nothing
    dbg("OctofitterInterfere v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | OctofitterInterferometry ", something(pkgversion(OctofitterInterferometry), "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("Pigeons: n_rounds = ", n_rounds, " (", 2^n_rounds, " samples; page 10) | n_chains = ",
        n_chains === nothing ? "Pigeons default" : n_chains,
        " | data: closure phases", vis2 ? " + squared visibilities" : " only (use_vis2 = false, page)")
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

function fold_axial(x)
    c = atan(Statistics.mean(sin.(2 .* x)), Statistics.mean(cos.(2 .* x))) / 2
    return mod.(x .- c .+ pi / 2, pi) .- pi / 2 .+ c
end

function qprint(label, x)
    q = qline(x)
    dbg(@sprintf("    %-30s %12.5g  [%12.5g, %12.5g]", label, q[2], q[1], q[3]))
end

function print_summary(chain)
    DEBUG[] || return nothing
    n = size(chain, 1) * size(chain, 3)
    dbg("Posterior median [16th, 84th] (", n, " samples):")
    for (lab, p, f) in (("b flux_K (contrast)", :b_flux_K, identity),
                        ("b a [AU]", :b_a, identity),
                        ("b e", :b_e, identity),
                        ("b i [deg]", :b_i, x -> rad2deg.(x)),
                        ("b Ω [deg] (folded 180°)", :b_Ω, x -> rad2deg.(fold_axial(x))),
                        ("b ω [deg]", :b_ω, x -> rad2deg.(x)),
                        ("b θ at MJD 60171 [deg]", :b_θ, x -> rad2deg.(x)),
                        ("A mass [M⊙]", :A_mass, identity),
                        ("plx [mas]", :plx, identity))
        haspar(chain, p) && qprint(lab, f(colvec(chain, p)))
    end
    if haspar(chain, :b_a) && haspar(chain, :A_mass)
        a = colvec(chain, :b_a)
        qprint("b P = √(a³/M) [yr] (derived)", sqrt.(a .^ 3 ./ colvec(chain, :A_mass)))
        haspar(chain, :plx) && qprint("b a [mas] = a·plx (derived)", a .* colvec(chain, :plx))
    end
    return nothing
end

"The page's SNR, plus median/IQR and what the flux prior alone gives."
function detection_lines(chain)
    f = colvec(chain, :b_flux_K)
    μ, σ = Statistics.mean(f), Statistics.std(f)
    q25, q50, q75 = Statistics.quantile(f, (0.25, 0.5, 0.75))
    pμ, pσ = mean(PRIORS.flux_K), std(PRIORS.flux_K)
    lines = String[]
    push!(lines, "Detection (b_flux_K, contrast to the host):")
    push!(lines, @sprintf("    page SNR  = mean/std          = %.4g / %.4g = %.1f", μ, σ, μ / σ))
    push!(lines, @sprintf("    alt. SNR  = median/IQR        = %.4g / %.4g = %.1f", q50, q75 - q25, q50 / (q75 - q25)))
    push!(lines, @sprintf("    prior only: mean/std          = %.4g / %.4g = %.2f  (half-normal, σ = 0.1)", pμ, pσ, pμ / pσ))
    push!(lines, @sprintf("    contrast %.4g ≈ Δmag %.2f", μ, -2.5 * log10(μ)))
    return lines
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

const CMP1 = "#0072B2"
const CMP2 = "#E69F00"
const EPOCH_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")

light(f) = with_theme(f, Theme())

# -----------------------------------------------------------------------------
# Data
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

"The page's data table, with absolute paths (InterferometryObs reads the files)."
function data_table(outdir; vis2::Bool=false)
    rows = map(EXPOSURES) do (file, date)
        p = fetch_once(OCTO_RAW * "examples/AMI_data/" * file, joinpath(outdir, file))
        (; filename=p, epoch=mjd(date), use_vis2=vis2)
    end
    return Table(collect(rows))
end

"Date part of an MJD (mjd2date gives a DateTime, which prints T00:00:00)."
datestr(ep) = first(string(mjd2date(ep)), 10)

"""
Per-epoch data summary and a model-free signal test: χ² of the closure phases
against zero (a single point source has all-zero closure phases). Each closure
phase is only slightly above its error bar, so the test that matters is this
combined χ² — and, beyond it, the coherent pattern the orbit fit exploits.
"""
function describe_obs(obs)
    t = obs.table
    χ²tot, ntot = 0.0, 0
    for k in eachindex(t.epoch)
        cp = vec(t.cps_data[k]); dcp = vec(t.dcps[k])
        nλ = hasproperty(t, :eff_wave) ? length(t.eff_wave[k]) : 0
        λ = nλ > 0 ? @sprintf("%.3f μm", 1e6 * Statistics.mean(t.eff_wave[k])) : "?"
        dbg(@sprintf("  epoch %d: MJD %.1f (%s) | %d baselines × %d channel(s) at %s | %d closure phases, rms %.3f°, median σ %.3f°",
                     k, t.epoch[k], datestr(t.epoch[k]), size(t.u[k], 1), size(t.u[k], 2), λ,
                     length(cp), sqrt(Statistics.mean(abs2, cp)), Statistics.median(dcp)))
        χ² = sum(abs2, cp ./ dcp); n = length(cp)
        χ²tot += χ²; ntot += n
        dbg(@sprintf("           χ² vs no companion (all CPs = 0) = %.1f for %d CPs (χ²/N %.2f, p = %.2g)",
                     χ², n, χ² / n, ccdf(Chisq(n), χ²)))
    end
    dbg(@sprintf("  all epochs: χ² = %.1f for %d CPs (χ²/N %.2f, p = %.2g) — each CP is near the noise; the orbit fit adds the coherent pattern",
                 χ²tot, ntot, χ²tot / ntot, ccdf(Chisq(ntot), χ²tot)))
    for k in 2:length(t.epoch)
        du = maximum(abs, vec(t.u[k]) .- vec(t.u[1])) / max(maximum(abs, t.u[1]), eps())
        du < 1e-6 && dbg("  epoch $k has the same u-v coverage as epoch 1 (identical pupil orientation)")
    end
end

"The page's closure-phase stem plot, one series per epoch."
function cp_figure(obs)
    fig = Figure(size=(1100, 480))
    ax = Axis(fig[1, 1], xlabel="index", ylabel="closure phase [deg]", title="Closure phases (page)")
    for k in eachindex(obs.table.epoch)
        stem!(ax, vec(obs.table.cps_data[k]); color=EPOCH_COLORS[mod1(k, end)],
              stemcolor=EPOCH_COLORS[mod1(k, end)], label="epoch $k")
    end
    Legend(fig[1, 2], ax)
    return fig
end

"+ u-v coverage of each exposure (in units of 10⁶ wavelengths), with the point reflections."
function uv_figure(obs)
    fig = Figure(size=(700, 650))
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="u [Mλ]", ylabel="v [Mλ]", title="u-v coverage")
    n = length(obs.table.epoch)
    for k in eachindex(obs.table.epoch)
        u = vec(obs.table.u[k]) ./ 1e6; v = vec(obs.table.v[k]) ./ 1e6
        c = EPOCH_COLORS[mod1(k, end)]
        # decreasing sizes, so epochs with identical coverage stay visible as rings
        scatter!(ax, vcat(u, -u), vcat(v, -v); color=(c, 0.9), markersize=6 + 5 * (n - k),
                 label="epoch $k ($(datestr(obs.table.epoch[k])))")
    end
    axislegend(ax, position=:rt, unique=true)
    return fig
end

# -----------------------------------------------------------------------------
# Model (page section "Build the model")
# -----------------------------------------------------------------------------
function build_system(data)
    mA = PRIORS.mass_A
    A_vars = @eval @variables begin
        mass ~ $mA            # M⊙
        flux_K = 1.0          # the host is an ordinary source
    end
    A = Body(name="A", variables=A_vars)
    fK, ap, ep, θep = PRIORS.flux_K, PRIORS.a, PRIORS.e, PRIORS.θ_ep
    b_vars = @eval @variables begin
        mass = 0.0
        flux_K ~ $fK          # contrast ratio against the host
        a ~ $ap
        e ~ $ep
        i ~ Sine()
        ω ~ UniformCircular()
        Ω ~ UniformCircular()
        θ ~ UniformCircular()
        epoch = $θep          # reference epoch for θ
    end
    b = Body(name="b", about=A, variables=b_vars)
    obs_vars = @eval @variables begin
        platescale = 1.0      # platescale divisor
        northangle = 0.0      # north angle offset [rad]
        σ_cp_jitter = 0.0     # closure phase jitter [deg]
    end
    vis_obs = InterferometryObs(
        data,
        targets = (A, b),     # every source in the visibility sum, host included
        ref     = A,          # phase centre
        band    = :K,
        name    = "NIRISS-AMI",
        variables = obs_vars,
    )
    pp = PRIORS.plx
    sys_vars = @eval @variables begin
        plx ~ $pp
    end
    sys = System(name="Tutoria", bodies=[A, b], observations=[vis_obs], variables=sys_vars)
    return sys, vis_obs
end

# -----------------------------------------------------------------------------
# Figures after sampling
# -----------------------------------------------------------------------------
"""
Page's flux histogram (left, zoomed on the posterior), and + posterior vs
prior on a log₁₀ contrast axis (right): the prior spans decades the posterior
never visits, so on a linear axis one of the two is always invisible.
"""
function flux_figure(chain)
    f = colvec(chain, :b_flux_K)
    fig = Figure(size=(1300, 460))
    ax1 = Axis(fig[1, 1], xlabel="flux (K band, relative to host)", ylabel="density",
               title="b_flux_K (page)")
    hist!(ax1, f; bins=50, normalization=:pdf, color=(CMP1, 0.85))
    ax2 = Axis(fig[1, 2], xlabel="log₁₀ contrast", ylabel="density",
               title="Posterior vs prior N(0, 0.1), f ≥ 0")
    prior = rand(PRIORS.flux_K, 200_000)
    lp = log10.(prior[prior .> 0])
    hist!(ax2, lp; bins=120, normalization=:pdf, color=(CMP2, 0.6), label="prior")
    hist!(ax2, log10.(f); bins=40, normalization=:pdf, color=(CMP1, 0.85), label="posterior")
    xlims!(ax2, max(-6, minimum(lp)), 0)
    axislegend(ax2, position=:lt)
    return fig
end

"""
Page's figure of b's position at each epoch (one PlanetOrbits system per draw),
plus the median separation and position angle per epoch.
"""
function positions_figure(model, chain, obs)
    posteriors = construct_system(model, chain)
    fig = Figure(size=(820, 700))
    ax = Axis(fig[1, 1], autolimitaspect=1, xreversed=true,
              xlabel="ΔR.A. (mas)", ylabel="ΔDec. (mas)", title="Position of b at each epoch")
    stats = NamedTuple[]
    for (k, epoch) in enumerate(obs.table.epoch)
        sols = [orbitsolve(p, epoch) for p in posteriors]
        x = [raoff(s, :b, :A) for s in sols]
        y = [decoff(s, :b, :A) for s in sols]
        scatter!(ax, x, y; markersize=1.5, color=(EPOCH_COLORS[mod1(k, end)], 0.6),
                 label=datestr(epoch))
        sep = sqrt.(x .^ 2 .+ y .^ 2)
        pa = mod.(rad2deg.(atan.(x, y)), 360)
        push!(stats, (; epoch, sep=qline(sep), pa=qline(pa)))
    end
    scatter!(ax, [0.0], [0.0]; marker=:star5, color=:white, markersize=14)
    Legend(fig[1, 2], ax, "date")
    return fig, stats
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_interfere(; n_rounds=10, n_chains=nothing, vis2=false,
                    outdir=SCRIPT_DIR, prefix="ifm_", dark=true, show_plots=true)
"""
function run_interfere(; n_rounds::Integer=envint("OCTOIFM_ROUNDS", "10"),
                         n_chains=(c = envopt("OCTOIFM_CHAINS"); c === nothing ? nothing : parse(Int, c)),
                         vis2::Bool=envbool("OCTOIFM_VIS2", "false"),
                         outdir::AbstractString=SCRIPT_DIR,
                         prefix::AbstractString=PREFIX,
                         dark::Bool=true,
                         show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    print_env_info(outdir, n_rounds, n_chains, vis2)
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

    # ---- Data and model --------------------------------------------------------------
    data = stage("OI-FITS files (download once, pinned commit)") do
        data_table(outdir; vis2)
    end
    sys, vis_obs = stage("Bodies, InterferometryObs \"NIRISS-AMI\" (reads the OI-FITS files), System") do
        build_system(data)
    end
    describe_obs(vis_obs)
    DEBUG[] && display(sys)
    soft_stage("Closure phases (page)") do
        savefig!(cp_figure(vis_obs), "closure_phases.png")
    end
    soft_stage("u-v coverage") do
        savefig!(uv_figure(vis_obs), "uv_coverage.png")
    end

    model = stage("Compile LogDensityModel") do
        Octofitter.LogDensityModel(sys)
    end
    DEBUG[] && display(model)
    stage("initialize!(model) (page)") do
        Octofitter.initialize!(model)
    end

    # ---- Sampling -------------------------------------------------------------------
    chain, pt = stage("octofit_pigeons (n_rounds=$n_rounds" *
                      (n_chains === nothing ? "" : ", n_chains=$n_chains") * "; page)") do
        n_chains === nothing ? octofit_pigeons(model; n_rounds) :
                               octofit_pigeons(model; n_rounds, n_chains)
    end
    diag = soft_stage("Pigeons diagnostics") do
        z = Pigeons.stepping_stone(pt)
        λ = Pigeons.global_barrier(pt)
        mα = minimum(Pigeons.swap_prs(pt))
        dbg(@sprintf("  log(Z₁/Z₀) ≈ %.3f | Λ = %.2f | min(α) = %.3g%s", z, λ, mα,
                     mα < 1e-3 ? "  ⚠ chains barely swap: add n_chains or rounds" : ""))
        (; logZ=z, Λ=λ, min_α=mα)
    end
    DEBUG[] && display(chain)
    print_summary(chain)

    p = joinpath(outdir, prefix * "chain.fits")
    soft_stage("Save chain -> FITS (+ reload check)") do
        Octofitter.savechain(p, chain)
        push!(outputs, p)
        c2 = Octofitter.loadchain(p; model)
        dbg("  reloaded chain size = ", size(c2), size(c2) == size(chain) ? "  (matches)" : "  (DIFFERS)")
    end

    # ---- Analysis (page) ------------------------------------------------------------------
    soft_stage("b_flux_K histogram (page) with prior") do
        savefig!(flux_figure(chain), "flux_K.png")
    end
    soft_stage("octoplot (page)") do
        savefig!(inout(() -> octoplot(model, chain)), "octoplot.png")
    end
    pos = soft_stage("Position of b at each epoch (page; construct_system per draw)") do
        fig, stats = positions_figure(model, chain, vis_obs)
        savefig!(fig, "positions.png")
        for s in stats
            dbg(@sprintf("  %s: separation %.2f mas [%.2f, %.2f], PA %.1f° [%.1f, %.1f]",
                         datestr(s.epoch), s.sep[2], s.sep[1], s.sep[3],
                         s.pa[2], s.pa[1], s.pa[3]))
        end
        stats
    end
    soft_stage("Corner plot (page: full octocorner, light theme)") do
        savefig!(inout(() -> light(() -> octocorner(model, chain))), "corner.png")
    end
    soft_stage("Corner plot (small=true + b_flux_K, light theme)") do
        savefig!(inout(() -> light(() -> octocorner(model, chain; small=true,
                                                     includecols=[:b_flux_K]))), "corner_small.png")
    end

    # ---- Detection summary --------------------------------------------------------------
    lines = String[]
    haspar(chain, :b_flux_K) && append!(lines, detection_lines(chain))
    if diag !== nothing
        push!(lines, @sprintf("Pigeons: n_rounds = %d, log(Z₁/Z₀) = %.3f, Λ = %.2f, min(α) = %.3g",
                              n_rounds, diag.logZ, diag.Λ, diag.min_α))
    end
    if pos !== nothing
        for s in pos
            push!(lines, @sprintf("b at %s: sep %.2f mas, PA %.1f°", datestr(s.epoch),
                                  s.sep[2], s.pa[2]))
        end
    end
    foreach(l -> dbg(l), lines)
    soft_stage("Write summary") do
        ps = joinpath(outdir, prefix * "summary.txt")
        write(ps, join(lines, "\n") * "\n")
        push!(outputs, ps)
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f min", (time() - T0[]) / 60))
    return (; model, chain, pt, obs=vis_obs, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default when the REPL has 1 thread)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString=PREFIX)
    T0[] = time()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOIFM_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOIFM_CHILD=1]")
    dbg("  Child output follows; its plots are shown here when it finishes.")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOIFM_RELAUNCH\"]=\"false\".")
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
    elseif lowercase(get(ENV, "OCTOIFM_SHOW_PLOTS", "true")) != "false"
        soft_stage("Show $(length(pngs)) saved plot(s) in the plot pane") do
            show_saved_pngs(pngs)
        end
    end
    return (; relaunched=true, success=ok, chain, outputs=new_files)
end

function png_rank(p)
    f = basename(p)
    for (k, key) in enumerate(("closure_phases", "uv_coverage", "flux_K", "octoplot", "positions",
                               "corner_small", "corner"))
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

println(@sprintf("[IFM] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterInterfere

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterInterferometryTutorial.jl` -> runs
#   - VSCode "Execute active File in REPL" -> runs; with 1 thread it relaunches
#     in `julia --threads=auto` (OCTOIFM_RELAUNCH=false to disable,
#     OCTOIFM_THREADS to set the count)
#   - ENV["OCTOIFM_AUTORUN"] = "false"; include() -> loads module only
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOIFM_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOIFM_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOIFM_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global ifm_result = OctofitterInterfere.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOIFM_THREADS", "auto"))
            println("[IFM] Child run finished. `ifm_result` has fields: relaunched, success, ",
                    "chain (loaded from FITS), outputs")
        else
            global ifm_result = OctofitterInterfere.run_interfere(show_plots=!is_child)
            is_child || println("[IFM] Result stored in `ifm_result` (fields: model, chain, pt, ",
                                "obs, outputs)")
        end
    end
end

nothing
