#=
================================================================================
 OctofitterMassPhotometry.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Connecting Mass with
 Photometry":
   https://sefffal.github.io/Octofitter.jl/dev/mass-photometry/

 A body's `flux_<band>` is an ordinary model variable, so it can be derived
 from its mass through an evolutionary model. The page uses the Sonora Bobcat
 grids shipped with Octofitter (auto-downloaded DataDep, ~1 MiB):
     (age, mass) --cooling--> T_eff --atmosphere--> absolute magnitude
 and turns the magnitude difference to the host into a linear contrast
 ratio, 10^(−0.4 Δmag), which PhotometryObs compares with measurements.

 Parts:
   toy     the page's first example: flux_H = sqrt(mass) as a derived
           variable; checked here by drawing from the priors (optional).
   page    the page's model: host A (1.0 ± 0.1 M⊙, flux_H = flux_L = 1),
           companion b with mass ~ LogUniform(2, 25) M_jup, T_eff from the
           cooling track at the system age (40 ± 5 Myr), H and L′ contrasts
           against host absolute magnitudes 3.4 / 3.3; data NIRC2_H
           2.4e-4 ± 4.0e-5 and NIRC2_L 7.5e-4 ± 1.2e-4; plx 12.0 ± 0.01 mas.
           Sampled as on the page: initialize!, then octofit (HMC)
           with iterations = adaptation = 2000.
   clamped the page's "Out-of-grid masses" recipe made runnable: the same
           model with a prior that deliberately runs past the grid
           (mass ~ LogUniform(1, 150) M_jup) and the page's clamp —
           NaN magnitude → 8.2 (absurdly bright) above 20 M_jup,
           16.7 (absurdly dim) below. Applied to H as on the page and, with
           the same numbers, to L′.

 Plots: Sonora curves (T_eff and H/L′ contrast vs mass at the page's age,
 with the measured contrasts), companion-mass histogram (page) with the prior,
 corner (small=true + age, T_eff, contrasts), contrast predictive check,
 page-vs-clamped comparison. Log: grid coverage at 40 Myr, the page's median
 H/L′ contrast and temperature, posterior summaries.

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `mp_result`)
   Shell:    julia OctofitterMassPhotometry.jl
   Library:  ENV["OCTOMP_AUTORUN"] = "false"; include("OctofitterMassPhotometry.jl")
             res = OctofitterMassPhot.run_massphot(models=(:page, :clamped))

 Sampler: the page uses HMC (`octofit`), a single chain that gains nothing
 from threads, so by default this runs in the REPL. OCTOMP_SAMPLER=pigeons
 uses octofit_pigeons instead and, in a 1-thread REPL, relaunches in a child
 `julia --threads=auto` (plots shown here afterwards).

 Settings (ENV):
   OCTOMP_MODELS      = "page,clamped" (default) | "page" | "clamped"
   OCTOMP_TOY         = "true"  — run the toy prior check
   OCTOMP_SAMPLER     = "hmc" (default, page) | "pigeons"
   OCTOMP_ITERATIONS  = "2000" (page) — HMC iterations
   OCTOMP_ADAPTATION  = "2000" (page) — HMC adaptation steps
   OCTOMP_ROUNDS      = "10"  — Pigeons n_rounds (sampler=pigeons)
   OCTOMP_SHOW_PLOTS  = "true" — show a child's PNGs here (pigeons only)
   OCTOMP_HEARTBEAT   = "30"  — seconds between "still waiting" lines
   OCTOMP_ACCEPT_DOWNLOADS = "true" — sets DATADEPS_ALWAYS_ACCEPT

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOMP_ENV_MODE=local uses ./octofitter_v9_env). Needs Octofitter
 v9, CairoMakie, PairPlots, Distributions, Pigeons.

 Outputs (prefix "mp_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-26  Initial version on the OctofitterImagesTutorial v1.0.1
                   scaffolding ([MP +t] debug stages, soft optional stages,
                   env bootstrap with v9 guards, @__DIR__ outputs, dark theme
                   with light corner plots, heartbeat and stale-DataDep check,
                   @eval-built @variables with interpolated interpolators,
                   HMC in the REPL / Pigeons via threaded child). Adds: Sonora
                   grid-coverage scan, Sonora curves figure with the measured
                   contrasts, the page's clamp recipe as a runnable model, a
                   contrast predictive check and a page-vs-clamped comparison.
 v1.0.1 2026-09-26 First run (1.7 min): b mass 11.8 (11.5–12.0) M_jup,
                   T_eff 1378 K, H/L′ contrasts −0.71σ / +0.34σ from the data;
                   the clamped model agrees (H grid at 40 Myr: 0.5–35 M_jup,
                   0% of samples clamped). Fixes: `display(sys)` printed the
                   interpolators' captured grids (>1 MiB, so the start of the
                   log was lost) — they are now wrapped in `Labelled`, which
                   prints by name; the corner plot leaves out the prior-only
                   a, e, i.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let required = ("Octofitter", "CairoMakie", "PairPlots", "Distributions", "Pigeons"),
    min_version = Dict("Octofitter" => v"9"),
    mode = lowercase(get(ENV, "OCTOMP_ENV_MODE", "temp"))

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
            println("[MP] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[MP] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[MP] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[MP] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[MP] Existing v9 environment found at ", Base.active_project())
        else
            Pkg.add(spec.(collect(required)))
        end
        println("[MP] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[MP] Loading Octofitter, Pigeons, CairoMakie, PairPlots, Distributions ",
        "(first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterMassPhot

const LOAD_T0 = time()

using Octofitter
using CairoMakie
using PairPlots
using Distributions
using Pigeons
import Random
import Statistics
using Printf

export run_massphot, set_plot_theme!

const VERSION_STRING = "1.0.1"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterMassPhot needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const PREFIX = "mp_"

# The page's numbers
const PAGE = (
    age        = truncated(Normal(40, 5), lower=1),        # Myr
    age0       = 40.0,
    mass_A     = truncated(Normal(1.0, 0.1), lower=0.1),   # M⊙
    host_H     = 3.4,                                      # host absolute magnitudes
    host_L     = 3.3,
    plx        = truncated(Normal(12.0, 0.01), lower=0.1), # mas
    H          = (phot=2.4e-4, σ=4.0e-5),                  # NIRC2 contrasts
    L          = (phot=7.5e-4, σ=1.2e-4),
    mass_page  = (2.0, 25.0),                              # M_jup, LogUniform bounds
    mass_wide  = (1.0, 150.0),                             # M_jup, clamped model
    clamp_hi   = 8.2,                                      # page: off the top (bright)
    clamp_lo   = 16.7,                                     # page: off the bottom (dim)
    clamp_at   = 20.0,                                     # M_jup threshold (page)
)

const MODEL_LABELS = Dict(:page => "page model (mass 2–25 M_jup)",
                          :clamped => "clamped (mass 1–150 M_jup, page's out-of-grid recipe)")
const MODEL_ORDER = (:page, :clamped)

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterMassPhot.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[MP +%7.1fs] ", time() - T0[]), msg...)
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

function accept_downloads!()
    lowercase(get(ENV, "OCTOMP_ACCEPT_DOWNLOADS", "true")) != "false" &&
        (ENV["DATADEPS_ALWAYS_ACCEPT"] = "true")
    return nothing
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

"The Sonora folder must hold evolution_tables/ and photometry_tables/."
function check_sonora_datadep()
    for d in datadeps_dirs()
        p = joinpath(d, "SonoraBobcatEvoPhot")
        isdir(p) || continue
        ok = isdir(joinpath(p, "evolution_tables")) && isdir(joinpath(p, "photometry_tables"))
        if ok
            dbg("  Sonora DataDep present: ", p)
        else
            dbg("  ⚠ incomplete Sonora DataDep folder (interrupted download?): ", p)
            dbg("    delete it so DataDeps downloads again:  rm(\"", p, "\"; recursive=true)")
        end
        return ok
    end
    dbg("  Sonora DataDep not downloaded yet (≈1 MiB from zenodo.org/record/5063476; first use fetches it)")
    return false
end

function with_heartbeat(f, what::AbstractString;
                        every::Real=parse(Float64, get(ENV, "OCTOMP_HEARTBEAT", "30")))
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

function print_env_info(outdir, models, sampler, iters, adapt, rounds)
    DEBUG[] || return nothing
    dbg("OctofitterMassPhot v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("models = ", join(models, ", "), " | sampler = ", sampler,
        sampler == "hmc" ? " (octofit, iterations=$iters, adaptation=$adapt; page)" :
                           " (octofit_pigeons, n_rounds=$rounds)")
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

function qprint(label, x)
    q = qline(x)
    dbg(@sprintf("    %-30s %12.5g  [%12.5g, %12.5g]", label, q[2], q[1], q[3]))
end

function print_summary(chain, label)
    DEBUG[] || return nothing
    n = size(chain, 1) * size(chain, 3)
    dbg("Posterior median [16th, 84th] — ", label, " (", n, " samples):")
    for (lab, p, f) in (("b mass [M_jup]", :b_mass, x -> x ./ mjup),
                        ("system age [Myr]", :age, identity),
                        ("b T_eff [K]", :b_tempK, identity),
                        ("b abs mag H", :b_abs_mag_H, identity),
                        ("b abs mag L′", :b_abs_mag_L, identity),
                        ("b contrast H", :b_flux_H, identity),
                        ("b contrast L′", :b_flux_L, identity),
                        ("A mass [M⊙]", :A_mass, identity),
                        ("b a [AU] (prior only)", :b_a, identity))
        haspar(chain, p) && qprint(lab, f(colvec(chain, p)))
    end
    if haspar(chain, :b_flux_H) && haspar(chain, :b_flux_L)
        fH, fL = Statistics.median(colvec(chain, :b_flux_H)), Statistics.median(colvec(chain, :b_flux_L))
        dbg(@sprintf("    contrast check: median H %.3g vs data %.3g ± %.1g (%+.2fσ); L′ %.3g vs %.3g ± %.1g (%+.2fσ)",
                     fH, PAGE.H.phot, PAGE.H.σ, (fH - PAGE.H.phot) / PAGE.H.σ,
                     fL, PAGE.L.phot, PAGE.L.σ, (fL - PAGE.L.phot) / PAGE.L.σ))
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

const CMP1 = "#0072B2"
const CMP2 = "#E69F00"
const CMP3 = "#009E73"

corner(model, chain; dark::Bool=false, small::Bool=true, includecols=Symbol[], excludecols=Symbol[]) =
    dark ? octocorner(model, chain; small, includecols, excludecols) :
           with_theme(() -> octocorner(model, chain; small, includecols, excludecols), Theme())

# -----------------------------------------------------------------------------
# Sonora grids
# -----------------------------------------------------------------------------
"""
A callable that prints as its name. The Sonora interpolators are closures, and
Julia prints a closure together with everything it captured — here the full
T_eff and magnitude grids — so `display(sys)`, which shows each derived
expression, dumped over 1 MiB of numbers. Wrapped, the expression reads
`sonora_cooling(system.age, mass)`.
"""
struct Labelled{F}
    name::String
    f::F
end
@inline (l::Labelled)(args...) = l.f(args...)
Base.show(io::IO, l::Labelled) = print(io, l.name)
Base.show(io::IO, ::MIME"text/plain", l::Labelled) = print(io, l.name)

"The page's three interpolators (cooling, MKO H, Keck L′), wrapped to print by name."
function load_sonora()
    cooling = Labelled("sonora_cooling", Octofitter.sonora_cooling_interpolator())
    absmag_H = Labelled("sonora_absmag_MKO_H", Octofitter.sonora_photometry_interpolator(:MKO_H))
    absmag_L = Labelled("sonora_absmag_Keck_L′", Octofitter.sonora_photometry_interpolator(:Keck_L′))
    return (; cooling, absmag_H, absmag_L)
end

"Contrast against the host from an absolute magnitude."
contrast(absmag, host) = 10^(-0.4 * (absmag - host))

"""
Scan masses 0.1–300 M_jup at `age`: where the cooling track and each
photometric grid are finite. Returns (masses, T, mH, mL) for plotting.
"""
function grid_scan(s; age=PAGE.age0)
    m = exp.(range(log(0.1), log(300.0), length=800))          # M_jup
    T = [s.cooling(age, mj * mjup) for mj in m]
    mH = [isfinite(t) ? s.absmag_H(t, mj * mjup) : NaN for (t, mj) in zip(T, m)]
    mL = [isfinite(t) ? s.absmag_L(t, mj * mjup) : NaN for (t, mj) in zip(T, m)]
    rng(v) = (i = findall(isfinite, v); isempty(i) ? (NaN, NaN) : (m[first(i)], m[last(i)]))
    for (lab, v) in (("cooling T_eff", T), ("abs mag H", mH), ("abs mag L′", mL))
        lo, hi = rng(v)
        dbg(@sprintf("  %-14s finite for %.2f–%.1f M_jup at %.0f Myr", lab, lo, hi, age))
    end
    return (; m, T, mH, mL, finite=rng(mH))
end

"""
The mass the page's data point to at the page's age: where the predicted
contrast crosses each measurement (and its ±1σ).
"""
function mass_from_contrast(scan)
    out = Dict{Symbol,Any}()
    for (band, mags, host, d) in ((:H, scan.mH, PAGE.host_H, PAGE.H), (:L, scan.mL, PAGE.host_L, PAGE.L))
        c = [isfinite(x) ? contrast(x, host) : NaN for x in mags]
        cross(target) = begin
            j = findfirst(k -> isfinite(c[k]) && isfinite(c[k+1]) &&
                               (c[k] - target) * (c[k+1] - target) <= 0, 1:length(c)-1)
            j === nothing ? NaN :
                exp(log(scan.m[j]) + (log(target) - log(c[j])) / (log(c[j+1]) - log(c[j])) *
                    (log(scan.m[j+1]) - log(scan.m[j])))
        end
        out[band] = (cross(d.phot - d.σ), cross(d.phot), cross(d.phot + d.σ))
        dbg(@sprintf("  %s contrast %.2g ± %.1g ↔ mass %.1f M_jup (%.1f–%.1f) at %.0f Myr",
                     band == :H ? "H " : "L′", d.phot, d.σ, out[band][2], out[band][1], out[band][3],
                     PAGE.age0))
    end
    return out
end

function sonora_figure(s, scan)
    fig = Figure(size=(1300, 520))
    ax1 = Axis(fig[1, 1], xscale=log10, xlabel="mass [M_jup]", ylabel="T_eff [K]",
               title="Sonora Bobcat cooling tracks")
    for (age, col) in ((30.0, CMP3), (40.0, CMP1), (50.0, CMP2))
        T = [s.cooling(age, mj * mjup) for mj in scan.m]
        lines!(ax1, scan.m, T; color=col, label=@sprintf("%.0f Myr", age))
    end
    vspan!(ax1, PAGE.mass_page..., color=(:white, 0.07))
    axislegend(ax1, position=:lt)
    ax2 = Axis(fig[1, 2], xscale=log10, yscale=log10, xlabel="mass [M_jup]",
               ylabel="contrast to host", title="Predicted contrast at 40 Myr vs the page's data")
    cH = [isfinite(x) ? contrast(x, PAGE.host_H) : NaN for x in scan.mH]
    cL = [isfinite(x) ? contrast(x, PAGE.host_L) : NaN for x in scan.mL]
    lines!(ax2, scan.m, cH; color=CMP1, label="MKO H")
    lines!(ax2, scan.m, cL; color=CMP2, label="Keck L′")
    for (d, col) in ((PAGE.H, CMP1), (PAGE.L, CMP2))
        hspan!(ax2, d.phot - d.σ, d.phot + d.σ; color=(col, 0.25))
        hlines!(ax2, [d.phot]; color=col, linestyle=:dash)
    end
    vspan!(ax2, PAGE.mass_page..., color=(:white, 0.07))
    axislegend(ax2, position=:lt)
    Label(fig[2, 1:2], "Shaded vertical band: the page's mass prior (2–25 M_jup); horizontal bands: measured contrast ± 1σ",
          fontsize=12)
    return fig
end

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
"The page's toy: flux_H = sqrt(mass), checked by drawing from the priors."
function toy_check(; n=5)
    H_band_model(mass) = sqrt(mass)
    A_vars = @eval @variables begin
        mass ~ truncated(Normal(1.0, 0.1), lower=0.1)
    end
    A_toy = Body(name="A", variables=A_vars)
    b_vars = @eval @variables begin
        mass ~ Uniform(0, 1)                 # M⊙
        flux_H = $H_band_model(mass)         # derived from the mass!
        a ~ Normal(16, 3)
        e ~ truncated(Normal(0.2, 0.2), lower=0, upper=0.99)
        ω ~ Normal(0.6, 0.2)
        i ~ Normal(0.5, 0.2)
        Ω ~ Normal(0.0, 0.2)
        tp ~ Uniform(50000, 60000)
    end
    b_toy = Body(name="b", about=A_toy, variables=b_vars)
    pp = PAGE.plx
    sys_vars = @eval @variables begin
        plx ~ $pp
    end
    sys = System(name="toy", bodies=[A_toy, b_toy], observations=(), variables=sys_vars)
    model = Octofitter.LogDensityModel(sys; verbosity=0)
    rng = Random.Xoshiro(1)
    for k in 1:n
        θ = model.arr2nt(model.sample_priors(rng))
        m, f = θ.bodies.b.mass, θ.bodies.b.flux_H
        dbg(@sprintf("    draw %d: b mass %.4f M⊙ → flux_H %.4f (sqrt(mass) = %.4f)%s", k, m, f, sqrt(m),
                     isapprox(f, sqrt(m)) ? "  ✓" : "  ✗"))
    end
    return true
end

"Companion variables for the page model or the clamped variant."
function make_b(A, s, kind::Symbol)
    ct, aH, aL = s.cooling, s.absmag_H, s.absmag_L
    if kind === :page
        mp = LogUniform(PAGE.mass_page[1] * mjup, PAGE.mass_page[2] * mjup)
        vars = @eval @variables begin
            mass ~ $mp                           # M⊙
            # Evolutionary model: age + mass -> temperature -> absolute magnitude
            tempK     = $ct(system.age, mass)
            abs_mag_H = $aH(tempK, mass)
            abs_mag_L = $aL(tempK, mass)
            # Absolute magnitude -> contrast ratio against the host
            flux_H = 10^(-0.4 * (abs_mag_H - system.host_abs_mag_H))
            flux_L = 10^(-0.4 * (abs_mag_L - system.host_abs_mag_L))
            a ~ truncated(Normal(16, 3), lower=0)
            e ~ truncated(Normal(0.2, 0.2), lower=0, upper=0.99)
            ω ~ Normal(0.6, 0.2)
            i ~ Normal(0.5, 0.2)
            Ω ~ Normal(0.0, 0.2)
            tp ~ Uniform(50000, 60000)
        end
    else
        mp = LogUniform(PAGE.mass_wide[1] * mjup, PAGE.mass_wide[2] * mjup)
        hi, lo, cut = PAGE.clamp_hi, PAGE.clamp_lo, PAGE.clamp_at * mjup
        vars = @eval @variables begin
            mass ~ $mp
            tempK     = $ct(system.age, mass)
            abs_mag_H = $aH(tempK, mass)
            abs_mag_L = $aL(tempK, mass)
            # The page's clamp: NaN outside the grid -> an absurd but finite magnitude
            # (`oftype` keeps the number type, so ForwardDiff sees no Union)
            abs_mag_H′ = isfinite(abs_mag_H) ? abs_mag_H : oftype(abs_mag_H, mass > $cut ? $hi : $lo)
            abs_mag_L′ = isfinite(abs_mag_L) ? abs_mag_L : oftype(abs_mag_L, mass > $cut ? $hi : $lo)
            flux_H = 10^(-0.4 * (abs_mag_H′ - system.host_abs_mag_H))
            flux_L = 10^(-0.4 * (abs_mag_L′ - system.host_abs_mag_L))
            a ~ truncated(Normal(16, 3), lower=0)
            e ~ truncated(Normal(0.2, 0.2), lower=0, upper=0.99)
            ω ~ Normal(0.6, 0.2)
            i ~ Normal(0.5, 0.2)
            Ω ~ Normal(0.0, 0.2)
            tp ~ Uniform(50000, 60000)
        end
    end
    return Body(name="b", about=A, variables=vars)
end

"Fresh bodies, photometry and system for `kind` ∈ (:page, :clamped)."
function build_system(s, kind::Symbol)
    mA = PAGE.mass_A
    A_vars = @eval @variables begin
        mass ~ $mA              # M⊙
        # Fixing the host's flux to 1 in each band makes every other body's
        # flux a contrast ratio against the host.
        flux_H = 1.0
        flux_L = 1.0
    end
    A = Body(name="A", variables=A_vars)
    b = make_b(A, s, kind)
    H_band_data = PhotometryObs(Table(phot=[PAGE.H.phot], σ_phot=[PAGE.H.σ]);
                                target=b, band=:H, name="NIRC2_H")
    L_band_data = PhotometryObs(Table(phot=[PAGE.L.phot], σ_phot=[PAGE.L.σ]);
                                target=b, band=:L, name="NIRC2_L")
    pp, ap, hH, hL = PAGE.plx, PAGE.age, PAGE.host_H, PAGE.host_L
    sys_vars = @eval @variables begin
        plx ~ $pp
        age ~ $ap                 # Myr
        host_abs_mag_H = $hH      # host absolute magnitudes
        host_abs_mag_L = $hL
    end
    return System(name=kind === :page ? "HD12345" : "HD12345_clamped", bodies=[A, b],
                  observations=[H_band_data, L_band_data], variables=sys_vars)
end

# -----------------------------------------------------------------------------
# Figures after sampling
# -----------------------------------------------------------------------------
"The page's mass histogram, with the prior density overlaid."
function mass_figure(chain, kind)
    m = colvec(chain, :b_mass) ./ mjup
    lo, hi = kind === :page ? PAGE.mass_page : PAGE.mass_wide
    fig = Figure(size=(900, 460))
    ax = Axis(fig[1, 1], xlabel="companion mass [M_jup]", ylabel="density",
              title="b mass — " * MODEL_LABELS[kind])
    hist!(ax, m; bins=60, normalization=:pdf, color=(CMP1, 0.85), label="posterior")
    xs = range(max(lo, minimum(m) * 0.8), min(hi, maximum(m) * 1.2), length=300)
    lines!(ax, xs, pdf.(LogUniform(lo, hi), xs); color=CMP2, linewidth=2,
           label=@sprintf("prior LogUniform(%g, %g)", lo, hi))
    axislegend(ax, position=:rt)
    return fig
end

"Posterior contrasts against the measurements (Gaussian ±1σ curves)."
function contrast_check_figure(chain)
    fig = Figure(size=(1200, 440))
    for (k, (p, d, lab, col)) in enumerate(((:b_flux_H, PAGE.H, "H contrast", CMP1),
                                             (:b_flux_L, PAGE.L, "L′ contrast", CMP2)))
        ax = Axis(fig[1, k], xlabel=lab, ylabel="density")
        x = colvec(chain, p)
        hist!(ax, x; bins=60, normalization=:pdf, color=(col, 0.8), label="posterior")
        xs = range(min(minimum(x), d.phot - 4d.σ), max(maximum(x), d.phot + 4d.σ), length=300)
        lines!(ax, xs, pdf.(Normal(d.phot, d.σ), xs); color=:white, linewidth=2,
               label="measurement ± σ")
        axislegend(ax, position=:rt)
    end
    Label(fig[0, 1:2], "Contrast check: the model's contrasts vs the page's NIRC2 data", fontsize=15)
    return fig
end

function compare_figure(ch1, ch2)
    fig = Figure(size=(1200, 440))
    ax1 = Axis(fig[1, 1], xlabel="b mass [M_jup]", ylabel="density", xscale=log10,
               title="Companion mass: page vs clamped (wide prior)")
    ax2 = Axis(fig[1, 2], xlabel="b T_eff [K]", ylabel="density", title="Temperature")
    for (ch, col, lab) in ((ch1, CMP1, "page (2–25 M_jup)"), (ch2, CMP2, "clamped (1–150 M_jup)"))
        m = colvec(ch, :b_mass) ./ mjup
        hist!(ax1, m; bins=exp.(range(log(minimum(m)), log(maximum(m)), length=60)),
              normalization=:pdf, color=(col, 0.55), label=lab)
        hist!(ax2, colvec(ch, :b_tempK); bins=60, normalization=:pdf, color=(col, 0.55), label=lab)
    end
    axislegend(ax1, position=:rt)
    return fig
end

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
"""
    run_massphot(; models=(:page, :clamped), toy=true, sampler="hmc",
                   iterations=2000, adaptation=2000, n_rounds=10,
                   outdir=SCRIPT_DIR, prefix="mp_", dark=true, show_plots=true)
"""
function run_massphot(; models=models_from_env(),
                        toy::Bool=envbool("OCTOMP_TOY", "true"),
                        sampler::AbstractString=lowercase(get(ENV, "OCTOMP_SAMPLER", "hmc")),
                        iterations::Integer=envint("OCTOMP_ITERATIONS", "2000"),
                        adaptation::Integer=envint("OCTOMP_ADAPTATION", "2000"),
                        n_rounds::Integer=envint("OCTOMP_ROUNDS", "10"),
                        outdir::AbstractString=SCRIPT_DIR,
                        prefix::AbstractString=PREFIX,
                        dark::Bool=true,
                        corner_dark::Bool=false,
                        show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    accept_downloads!()
    sampler in ("hmc", "pigeons") || error("OCTOMP_SAMPLER must be hmc or pigeons, got $sampler")
    print_env_info(outdir, models, sampler, iterations, adaptation, n_rounds)
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

    # ---- Sonora grids ---------------------------------------------------------------
    soft_stage("Check the Sonora DataDep") do
        check_sonora_datadep()
    end
    s = stage("Sonora interpolators: cooling, MKO H, Keck L′ (download + gridding; heartbeat)") do
        with_heartbeat("Sonora download / gridding") do
            load_sonora()
        end
    end
    dbg(@sprintf("  page check: cooling_tracks(40.0, 12mjup) = %.1f K", s.cooling(40.0, 12mjup)))
    dbg(@sprintf("  page check: absmag_H(T, 12mjup) = %.3f mag  → contrast %.3g",
                 s.absmag_H(s.cooling(40.0, 12mjup), 12mjup),
                 contrast(s.absmag_H(s.cooling(40.0, 12mjup), 12mjup), PAGE.host_H)))
    scan = stage("Grid coverage scan at $(PAGE.age0) Myr") do
        grid_scan(s)
    end
    soft_stage("Masses implied by each measured contrast at $(PAGE.age0) Myr") do
        mass_from_contrast(scan)
    end
    soft_stage("Sonora curves figure") do
        savefig!(sonora_figure(s, scan), "sonora_curves.png")
    end
    if toy
        soft_stage("Toy model: flux_H = sqrt(mass), prior draws") do
            toy_check()
        end
    end

    # ---- Models ------------------------------------------------------------------------
    chains = Dict{Symbol,Any}()
    for kind in models
        tag = string(kind)
        dbg("[$tag] ", MODEL_LABELS[kind])
        sys = stage("[$tag] Bodies, PhotometryObs (H, L′), System") do
            build_system(s, kind)
        end
        DEBUG[] && display(sys)
        model = stage("[$tag] Compile LogDensityModel") do
            Octofitter.LogDensityModel(sys)
        end
        stage("[$tag] initialize!(model) (page)") do
            Octofitter.initialize!(model)
        end
        chain = if sampler == "hmc"
            stage("[$tag] octofit (HMC, iterations=$iterations, adaptation=$adaptation; page)") do
                octofit(model; iterations, adaptation)
            end
        else
            stage("[$tag] octofit_pigeons (n_rounds=$n_rounds)") do
                c, pt = octofit_pigeons(model; n_rounds)
                soft_stage("[$tag] Pigeons diagnostics") do
                    dbg(@sprintf("  log(Z₁/Z₀) ≈ %.3f | Λ = %.2f | min(α) = %.3g",
                                 Pigeons.stepping_stone(pt), Pigeons.global_barrier(pt),
                                 minimum(Pigeons.swap_prs(pt))))
                end
                c
            end
        end
        chains[kind] = chain
        DEBUG[] && display(chain)
        print_summary(chain, MODEL_LABELS[kind])
        if kind === :page
            # The page's three printed numbers
            dbg("  H contrast: ", Statistics.median(colvec(chain, :b_flux_H)))
            dbg("  L contrast: ", Statistics.median(colvec(chain, :b_flux_L)))
            dbg("  Temperature: ", Statistics.median(colvec(chain, :b_tempK)), " K")
        else
            soft_stage("[$tag] Where the wide prior landed") do
                m = colvec(chain, :b_mass) ./ mjup
                lo, hi = scan.finite
                dbg(@sprintf("  %.1f%% of samples inside the H grid (%.1f–%.0f M_jup); %.1f%% clamped bright (above), %.1f%% clamped dim (below)",
                             100 * Statistics.mean(lo .< m .< hi), lo, hi,
                             100 * Statistics.mean(m .>= hi), 100 * Statistics.mean(m .<= lo)))
            end
        end

        soft_stage("[$tag] Mass histogram (page) with prior") do
            savefig!(mass_figure(chain, kind), "$(tag)_mass.png")
        end
        soft_stage("[$tag] Contrast check vs the measurements") do
            savefig!(contrast_check_figure(chain), "$(tag)_contrast_check.png")
        end
        # a, e, i are prior-only here (no orbit data), so they are left out
        soft_stage("[$tag] Corner (age, masses, T_eff, contrasts; light theme)") do
            savefig!(inout(() -> corner(model, chain; dark=corner_dark,
                                        includecols=[:age, :b_tempK, :b_flux_H, :b_flux_L],
                                        excludecols=[:b_a, :b_e, :b_i])),
                     "$(tag)_corner.png")
        end
        p = joinpath(outdir, prefix * "$(tag)_chain.fits")
        soft_stage("[$tag] Save chain -> FITS (+ reload check)") do
            Octofitter.savechain(p, chain)
            push!(outputs, p)
            c2 = Octofitter.loadchain(p; model)
            dbg("  reloaded chain size = ", size(c2), size(c2) == size(chain) ? "  (matches)" : "  (DIFFERS)")
        end
    end

    if haskey(chains, :page) && haskey(chains, :clamped)
        soft_stage("Page vs clamped comparison") do
            savefig!(compare_figure(chains[:page], chains[:clamped]), "compare.png")
        end
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f min", (time() - T0[]) / 60))
    return (; chains, sonora=s, scan, outputs)
end

"OCTOMP_MODELS (comma-separated) → tuple of Symbols in canonical order."
function models_from_env()
    str = get(ENV, "OCTOMP_MODELS", "page,clamped")
    req = Set(Symbol(strip(lowercase(x))) for x in split(str, ',') if !isempty(strip(x)))
    issubset(req, Set(MODEL_ORDER)) ||
        error("OCTOMP_MODELS must list page and/or clamped, got \"$str\"")
    return Tuple(k for k in MODEL_ORDER if k in req)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (only for OCTOMP_SAMPLER=pigeons in a 1-thread REPL)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString=PREFIX)
    T0[] = time()
    accept_downloads!()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOMP_CHILD" => "1")
    dbg("OCTOMP_SAMPLER=pigeons in a 1-thread session: relaunching with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOMP_CHILD=1]")
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
    chains = Dict{Symbol,Any}()
    for kind in MODEL_ORDER
        p = joinpath(outdir, prefix * "$(kind)_chain.fits")
        (isfile(p) && mtime(p) >= t) || continue
        c = soft_stage("Load $(kind) chain written by child") do
            Octofitter.loadchain(p)
        end
        c === nothing || (chains[kind] = c)
    end
    new_files = [joinpath(outdir, f) for f in readdir(outdir)
                 if startswith(f, prefix) && isfile(joinpath(outdir, f)) &&
                    mtime(joinpath(outdir, f)) >= t]
    list_outputs(new_files)
    pngs = filter(p -> endswith(lowercase(p), ".png"), new_files)
    if !isempty(pngs) && lowercase(get(ENV, "OCTOMP_SHOW_PLOTS", "true")) != "false"
        soft_stage("Show $(length(pngs)) saved plot(s) in the plot pane") do
            show_saved_pngs(pngs)
        end
    end
    return (; relaunched=true, success=ok, chains, outputs=new_files)
end

function png_rank(p)
    f = basename(p)
    occursin("sonora", f) && return 0
    occursin("compare", f) && return 99
    m = startswith(f, "mp_page") ? 10 : 20
    for (k, key) in enumerate(("mass", "contrast_check", "corner"))
        occursin(key, f) && return m + k
    end
    return m + 9
end

function show_saved_pngs(pngs; max_width::Integer=1400)
    for p in sort(pngs; by=png_rank)
        img = CairoMakie.Makie.FileIO.load(p)
        h, w = size(img)
        sc = min(1.0, max_width / w)
        fig = Figure(size=(round(Int, w * sc), round(Int, h * sc)), figure_padding=0,
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

println(@sprintf("[MP] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterMassPhot

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia OctofitterMassPhotometry.jl`          -> runs
#   - VSCode "Execute active File in REPL"         -> runs in the REPL (HMC);
#     with OCTOMP_SAMPLER=pigeons and 1 thread it relaunches in
#     `julia --threads=auto` (OCTOMP_RELAUNCH=false to disable)
#   - ENV["OCTOMP_AUTORUN"] = "false"; include()   -> loads module only
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOMP_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOMP_CHILD", "") == "1",
    pigeons   = lowercase(get(ENV, "OCTOMP_SAMPLER", "hmc")) == "pigeons",
    relaunch  = pigeons && !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOMP_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global mp_result = OctofitterMassPhot.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOMP_THREADS", "auto"))
            println("[MP] Child run finished. `mp_result` has fields: relaunched, success, ",
                    "chains (loaded from FITS), outputs")
        else
            global mp_result = OctofitterMassPhot.run_massphot(show_plots=!is_child)
            is_child || println("[MP] Result stored in `mp_result` (fields: chains, sonora, scan, outputs)")
        end
    end
end

nothing
