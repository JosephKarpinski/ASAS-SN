#=
================================================================================
 OctofitterImagesTutorial.jl
================================================================================
 Module wrapping the Octofitter.jl v9 tutorial "Fitting Images" (de-orbiting):
   https://sefffal.github.io/Octofitter.jl/dev/images/

 Five reduced coronagraphic images (the docs' `image-examples-1.fits`, a
 multi-extension FITS file) are fitted directly with an orbit: the likelihood
 reads the image flux at the planet's predicted position in every epoch, so a
 planet that is faint in single frames can be found from its Keplerian motion.

 Model (as on the page): host A with mass ~ N(2.0, 0.1) M⊙; planet b with
 flux_H ~ N(3.8, 0.5) (image units), a ~ N(13, 4) AU truncated to 0.1–100,
 e ~ U(0, 0.5), i ~ Sine(), ω, Ω, θ ~ UniformCircular(), θ epoch = MJD 57238.6;
 ImageObs "SPHERE" (targets = b, ref = A, band = :H, platescale = 1,
 northangle = 0); system HD82134, plx ~ N(45, 0.02) mas. Epochs:
 56000 + [1238.6, 1584.7, 3220.0, 7495.9, 7610.4], 10 mas/pixel.

 Sampling: octofit_pigeons(model, n_rounds=10) as on the page, after
 initialize!(model) (the module convention: Pathfinder starting points).

 Plots (as on the page, plus extras marked +):
   image preview (page's imview clims −1…4), +contrast curves per image,
   trace of b_a, autocorrelation of b_e, octoplot, orbits over image 2 and over
   the max-stack of all images, light-theme corner (small=true), b_flux_H
   histogram, +per-epoch stamps with the predicted positions of b,
   +flux posterior vs prior.
 Detection summary: the page's SNR = mean/std of b_flux_H and median/IQR.

 Running
 -------
   VSCode:   open this file -> "Julia: Execute active File in REPL"
             (result in `img_result`)
   Shell:    julia --threads=auto OctofitterImagesTutorial.jl
   Library:  ENV["OCTOIMG_AUTORUN"] = "false"; include("OctofitterImagesTutorial.jl")
             res = OctofitterImagesTutorial.run_images()

 Threads: Pigeons runs multithreaded. In a 1-thread REPL the run is relaunched
 in a child `julia --threads=auto` (output streamed here; the chain is loaded
 back and the saved plots are shown in the plot pane afterwards).
 OCTOIMG_RELAUNCH=false disables this; OCTOIMG_THREADS sets the count.

 Settings (ENV, also passed to a relaunched child):
   OCTOIMG_ROUNDS      = "10" (page) — Pigeons n_rounds (2^n samples)
   OCTOIMG_CHAINS      = ""   (Pigeons default) — n_chains
   OCTOIMG_INIT        = "true" — initialize!(model) before sampling
   OCTOIMG_IMAGE_IDX   = "2" (page) — image drawn under the orbits
   OCTOIMG_DRAWS       = "150" — posterior draws marked on the per-epoch stamps
   OCTOIMG_SHOW_PLOTS  = "true" — show the child's PNGs here

 Data: `image-examples-1.fits` (1.6 MB) is downloaded once from the Octofitter
 repository (pinned commit) next to this file.

 Environment: bootstrap as in the other tutorial modules (temporary v9 env by
 default; OCTOIMG_ENV_MODE=local uses ./octofitter_v9_env). OctofitterImages
 is unregistered: it is added with
   Pkg.PackageSpec(url="https://github.com/sefffal/Octofitter.jl",
                   subdir="OctofitterImages", rev=<pinned commit>)
 (the docs' installation page, pinned). Also needs AstroImages, CairoMakie,
 PairPlots, Distributions, Pigeons.

 Note: file and module are named `OctofitterImagesTutorial`, because
 `OctofitterImages` is the name of the package they load.

 Outputs (prefix "img_") go to the directory containing this file.

 Changelog
 ---------
 v1.0  2026-09-25  Initial version on the OctofitterG23HExample v1.0.1
                   scaffolding ([IMG +t] debug stages, soft optional stages,
                   env bootstrap with v9 guards, @__DIR__ outputs, dark theme
                   with light corner plots, threaded relaunch with exit-code-
                   only failure report and PNGs shown in the parent's plot
                   pane, @eval-built @variables, min(α) from Pigeons.swap_prs).
                   Adds: pinned OctofitterImages install; image overlays that
                   follow octoplot's mas/arcsec axis switch; NaN-safe max
                   stack; contrast curves; per-epoch stamps of predicted
                   positions; flux prior-vs-posterior comparison.
================================================================================
=#

# -----------------------------------------------------------------------------
# Environment bootstrap (runs before any `using`)
# -----------------------------------------------------------------------------
import Pkg

let octo_commit = "8c6daaaf5292b6cbdb0aa4f4b1e9a1a5f0bd21f3",
    required = ("Octofitter", "OctofitterImages", "AstroImages", "CairoMakie", "PairPlots",
                "Distributions", "Pigeons"),
    min_version = Dict("Octofitter" => v"9"),
    mode = lowercase(get(ENV, "OCTOIMG_ENV_MODE", "temp"))

    direct_deps() = Dict(info.name => info.version
                         for info in values(Pkg.dependencies()) if info.is_direct_dep)
    missing_deps() = [n for n in required if !haskey(direct_deps(), n)]
    function versions_ok()
        d = direct_deps()
        all(!haskey(d, n) || (d[n] !== nothing && d[n] >= v) for (n, v) in min_version)
    end
    env_ok() = isempty(missing_deps()) && versions_ok()
    function spec(n)
        n == "OctofitterImages" &&
            return Pkg.PackageSpec(url="https://github.com/sefffal/Octofitter.jl",
                                   subdir="OctofitterImages", rev=octo_commit)
        haskey(min_version, n) ? Pkg.PackageSpec(name=n, version="9") : Pkg.PackageSpec(name=n)
    end
    note(miss) = "OctofitterImages" in miss &&
        println("[IMG] OctofitterImages is unregistered: cloning it from the Octofitter repository ",
                "(commit ", first(octo_commit, 8), "); this takes a minute or two the first time.")

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
            println("[IMG] Octofitter v$v already loaded; reusing session environment ",
                    Base.active_project())
        else
            println("[IMG] Octofitter v$v already loaded; adding ", join(miss, ", "),
                    " to ", Base.active_project(), " (keeping loaded versions)")
            note(miss)
            Pkg.add(spec.(miss); preserve=Pkg.PRESERVE_ALL)
        end
    elseif env_ok()
        println("[IMG] Active environment already has Octofitter v",
                direct_deps()["Octofitter"], " and all packages: ", Base.active_project())
    else
        println("[IMG] Active environment ", Base.active_project(),
                " lacks Octofitter v9 or required packages; setting up a ",
                mode == "local" ? "local" : "temporary", " v9 environment...")
        if mode == "local"
            Pkg.activate(joinpath(@__DIR__(), "octofitter_v9_env"))
        else
            Pkg.activate(; temp=true)
        end
        if env_ok()
            println("[IMG] Existing v9 environment found at ", Base.active_project())
        else
            miss = missing_deps()
            note(miss)
            Pkg.add(spec.(collect(required)))
        end
        println("[IMG] Using environment ", Base.active_project(),
                " (Octofitter v", direct_deps()["Octofitter"], ")")
    end
    flush(stdout)
end

println("[IMG] Loading Octofitter, OctofitterImages, AstroImages, Pigeons, CairoMakie, PairPlots, ",
        "Distributions (first run precompiles and can take several minutes)...")
flush(stdout)

module OctofitterImagesTutorial

const LOAD_T0 = time()

using Octofitter
using OctofitterImages
import AstroImages
using CairoMakie
using PairPlots
using Distributions
using Pigeons
import Downloads
import Statistics
using Printf

export run_images, set_plot_theme!

const VERSION_STRING = "1.0"

const OCTOFITTER_VERSION = pkgversion(Octofitter)
if OCTOFITTER_VERSION !== nothing && OCTOFITTER_VERSION < v"9"
    error("""
    OctofitterImagesTutorial needs Octofitter v9+, but the active environment has
    Octofitter v$(OCTOFITTER_VERSION) (from $(Base.active_project())).
    Restart the REPL and execute this file again so the environment
    bootstrap at the top of the file can set up v9.
    """)
end

const SCRIPT_DIR = let d = @__DIR__()
    (d === nothing || isempty(d)) ? pwd() : d
end

const PREFIX = "img_"

const OCTO_COMMIT = "8c6daaaf5292b6cbdb0aa4f4b1e9a1a5f0bd21f3"
const OCTO_RAW = "https://raw.githubusercontent.com/sefffal/Octofitter.jl/$(OCTO_COMMIT)/"
const FITS_FILE = "image-examples-1.fits"
const FITS_URL = OCTO_RAW * "docs/" * FITS_FILE

# The page's data table and priors
const EPOCHS = 56000 .+ [1238.6, 1584.7, 3220.0, 7495.9, 7610.4]   # MJD (offset labels)
const PLATESCALE = 10.0                                            # mas / pixel
const PRIORS = (
    mass_A = truncated(Normal(2.0, 0.1), lower=0.1),               # M⊙
    flux_H = Normal(3.8, 0.5),                                     # image units
    a      = truncated(Normal(13, 4), lower=0.1, upper=100),       # AU
    e      = Uniform(0.0, 0.5),
    θ_ep   = 57238.6,                                              # MJD
    plx    = truncated(Normal(45.0, 0.02), lower=0.1),             # mas
)

# -----------------------------------------------------------------------------
# Debug output
# -----------------------------------------------------------------------------
"Set `OctofitterImagesTutorial.DEBUG[] = false` to silence debug output."
const DEBUG = Ref(true)
const T0 = Ref(time())

function dbg(msg...)
    DEBUG[] || return nothing
    println(@sprintf("[IMG +%7.1fs] ", time() - T0[]), msg...)
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
function envopt(k)
    s = strip(get(ENV, k, ""))
    return isempty(s) ? nothing : String(s)
end

function print_env_info(outdir, n_rounds, n_chains, init)
    DEBUG[] || return nothing
    dbg("OctofitterImagesTutorial v", VERSION_STRING, " | Julia ", VERSION,
        " | Octofitter ", something(OCTOFITTER_VERSION, "unknown"),
        " | OctofitterImages ", something(pkgversion(OctofitterImages), "unknown"),
        " | AstroImages ", something(pkgversion(AstroImages), "unknown"),
        " | Pigeons ", something(pkgversion(Pigeons), "unknown"),
        " | threads ", Threads.nthreads())
    dbg("Pigeons: n_rounds = ", n_rounds, " (", 2^n_rounds, " samples; page 10) | n_chains = ",
        n_chains === nothing ? "Pigeons default" : n_chains, " | initialize! = ", init)
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
    dbg(@sprintf("    %-30s %12.4f  [%12.4f, %12.4f]", label, q[2], q[1], q[3]))
end

function print_summary(chain)
    DEBUG[] || return nothing
    n = size(chain, 1) * size(chain, 3)
    dbg("Posterior median [16th, 84th] (", n, " samples):")
    rows = (("A mass [M⊙]", :A_mass, identity),
            ("b flux_H [image units]", :b_flux_H, identity),
            ("b a [AU]", :b_a, identity),
            ("b e", :b_e, identity),
            ("b i [deg]", :b_i, x -> rad2deg.(x)),
            ("b Ω [deg] (folded 180°)", :b_Ω, x -> rad2deg.(fold_axial(x))),
            ("b ω [deg]", :b_ω, x -> rad2deg.(x)),
            ("b θ at MJD 57238.6 [deg]", :b_θ, x -> rad2deg.(x)),
            ("plx [mas]", :plx, identity))
    for (lab, p, f) in rows
        haspar(chain, p) && qprint(lab, f(colvec(chain, p)))
    end
    if haspar(chain, :b_a) && haspar(chain, :A_mass)
        a = colvec(chain, :b_a); M = colvec(chain, :A_mass)
        qprint("b P = √(a³/M) [yr] (derived)", sqrt.(a .^ 3 ./ M))
        haspar(chain, :plx) && qprint("b a [mas] = a·plx (derived)", a .* colvec(chain, :plx))
    end
    return nothing
end

"""
The page's detection measures, plus a comparison with what the prior alone
would give (the flux prior N(3.8, 0.5) has 'SNR' 7.6 by itself).
"""
function detection_lines(chain)
    f = colvec(chain, :b_flux_H)
    μ, σ = Statistics.mean(f), Statistics.std(f)
    q25, q50, q75 = Statistics.quantile(f, (0.25, 0.5, 0.75))
    pμ, pσ = mean(PRIORS.flux_H), std(PRIORS.flux_H)
    lines = String[]
    push!(lines, "Detection (b_flux_H):")
    push!(lines, @sprintf("    page SNR  = mean/std          = %.3f / %.3f = %.2f", μ, σ, μ / σ))
    push!(lines, @sprintf("    alt. SNR  = median/IQR        = %.3f / %.3f = %.2f", q50, q75 - q25, q50 / (q75 - q25)))
    push!(lines, @sprintf("    prior only: mean/std          = %.3f / %.3f = %.2f", pμ, pσ, pμ / pσ))
    push!(lines, @sprintf("    posterior std / prior std     = %.2f  (< 1: the images constrain the flux)", σ / pσ))
    push!(lines, @sprintf("    posterior mean − prior mean   = %+.3f  (%+.2f prior σ)", μ - pμ, (μ - pμ) / pσ))
    return lines
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

corner(model, chain; dark::Bool=false, small::Bool=true) =
    dark ? octocorner(model, chain; small=small) :
           with_theme(() -> octocorner(model, chain; small=small), Theme())

const CMP1 = "#0072B2"
const CMP2 = "#E69F00"

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

"All HDUs of the multi-extension FITS file (the page's `load(file, :)`)."
function load_images(path)
    images = AstroImages.load(path, :)
    dbg("  ", length(images), " images")
    for (k, im) in enumerate(images)
        v = filter(isfinite, collect(Float64, im))
        dbg(@sprintf("    image %d: %d×%d, %s, finite %.1f%%, min %.2f, median %.3f, max %.2f",
                     k, size(im, 1), size(im, 2), eltype(im), 100 * length(v) / length(im),
                     minimum(v), Statistics.median(v), maximum(v)))
    end
    return images
end

"The page's table: recentred images, page epochs, 10 mas/pixel."
function image_table(images)
    length(images) == length(EPOCHS) ||
        error("expected $(length(EPOCHS)) images, found $(length(images))")
    return Table(;
        epoch = collect(EPOCHS),
        image = [AstroImages.recenter(images[k]) for k in eachindex(EPOCHS)],
        platescale = fill(PLATESCALE, length(EPOCHS)),
    )
end

"Image `k` flipped as on the page (RA increasing to the left), with axes in mas."
function sky_image(image_dat, k)
    ps = image_dat.platescale[k]
    img = AstroImages.recenter(AstroImages.AstroImage(collect(image_dat.image[k])[end:-1:begin, :]))
    x = Float64.(collect(AstroImages.dims(img, 1))) .* ps
    y = Float64.(collect(AstroImages.dims(img, 2))) .* ps
    return x, y, Float64.(collect(img))
end

"NaN-safe per-pixel maximum over all images (the page's max stack)."
function max_stack(image_dat)
    A = stack([collect(Float64, im) for im in image_dat.image])
    m = map(CartesianIndices(axes(A)[1:2])) do I
        v = filter(isfinite, view(A, I[1], I[2], :))
        isempty(v) ? NaN : maximum(v)
    end
    ps = image_dat.platescale[1]
    img = AstroImages.recenter(AstroImages.AstroImage(m[end:-1:begin, :]))
    x = Float64.(collect(AstroImages.dims(img, 1))) .* ps
    y = Float64.(collect(AstroImages.dims(img, 2))) .* ps
    return x, y, Float64.(collect(img))
end

"The page's preview: every image with clims (−1, 4), side by side."
function preview_figure(image_dat)
    n = length(image_dat.image)
    fig = Figure(size=(300 * n + 80, 380))
    local hm
    for k in 1:n
        x, y, z = sky_image(image_dat, k)
        ax = Axis(fig[1, k], aspect=DataAspect(), xreversed=true,
                  title=@sprintf("image %d — MJD %.1f", k, image_dat.epoch[k]),
                  xlabel="Δα* [mas]", ylabel=k == 1 ? "Δδ [mas]" : "")
        hm = heatmap!(ax, x, y, z; colormap=:magma, colorrange=(-1.0, 4.0), nan_color=:transparent)
        scatter!(ax, [0.0], [0.0]; marker=:star5, color=:white, markersize=12)
    end
    Colorbar(fig[1, n+1], hm, label="image flux (clims −1…4, as on the page)")
    return fig
end

"+ 1σ and 5σ contrast curves measured from each image (what ImageObs uses)."
function contrast_figure(image_dat)
    fig = Figure(size=(900, 480))
    ax = Axis(fig[1, 1], xlabel="separation [mas]", ylabel="flux [image units]", yscale=log10,
              title="Contrast measured from each image (annulus std; solid 1σ, dashed 5σ)")
    cols = CairoMakie.Makie.wong_colors()
    for k in eachindex(image_dat.image)
        c = OctofitterImages.contrast(image_dat.image[k])
        ok = isfinite.(c.contrast) .& (c.contrast .> 0)
        sep = c.separation[ok] .* image_dat.platescale[k]
        col = cols[mod1(k, length(cols))]
        lines!(ax, sep, c.contrast[ok]; color=col, label=@sprintf("image %d", k))
        lines!(ax, sep, 5 .* c.contrast[ok]; color=col, linestyle=:dash)
    end
    hlines!(ax, [mean(PRIORS.flux_H)]; color=:white, linestyle=:dot)
    text!(ax, 0.98, 0.95; text="dotted: prior mean flux_H = 3.8", space=:relative, align=(:right, :top),
          fontsize=11)
    axislegend(ax, position=:lb)
    return fig
end

# -----------------------------------------------------------------------------
# Model (page section "Build the model")
# -----------------------------------------------------------------------------
function build_system(image_dat)
    mA = PRIORS.mass_A
    A_vars = @eval @variables begin
        mass ~ $mA # M⊙
    end
    A = Body(name="A", variables=A_vars)
    fH, ap, ep, θep = PRIORS.flux_H, PRIORS.a, PRIORS.e, PRIORS.θ_ep
    b_vars = @eval @variables begin
        # Brightness of this planet in the H band, in image units.
        flux_H ~ $fH
        a ~ $ap
        e ~ $ep
        i ~ Sine()
        ω ~ UniformCircular()
        Ω ~ UniformCircular()
        θ ~ UniformCircular()       # position angle at `epoch`
        epoch = $θep                # reference epoch for θ; near the first image
    end
    b = Body(name="b", about=A, variables=b_vars)
    obs_vars = @eval @variables begin
        # Platescale uncertainty multiplier [could use: platescale ~ truncated(Normal(1, 0.01), lower=0)]
        platescale = 1.0
        # North angle offset in radians [could use: northangle ~ Normal(0, deg2rad(1))]
        northangle = 0.0
    end
    image_obs = ImageObs(
        image_dat,
        targets = (b,),
        ref     = A,
        band    = :H,
        name    = "SPHERE",
        variables = obs_vars,
    )
    pp = PRIORS.plx
    sys_vars = @eval @variables begin
        plx ~ $pp
    end
    sys = System(name="HD82134", bodies=[A, b], observations=[image_obs], variables=sys_vars)
    return sys, image_obs
end

# -----------------------------------------------------------------------------
# Figures after sampling
# -----------------------------------------------------------------------------
"Page's trace plot of b_a."
function trace_figure(chain)
    fig = Figure(size=(1000, 380))
    ax = Axis(fig[1, 1], xlabel="iteration", ylabel="semi-major axis (AU)", title="Trace of b_a")
    lines!(ax, colvec(chain, :b_a); color=CMP1)
    return fig
end

"Page's autocorrelation of b_e (lags 1:500, capped at n−1)."
function autocor_figure(chain)
    x = colvec(chain, :b_e)
    lags = 1:min(500, length(x) - 1)
    ac = Octofitter.StatsBase.autocor(x, lags)
    fig = Figure(size=(1000, 380))
    ax = Axis(fig[1, 1], xlabel="lag", ylabel="autocorrelation", title="Autocorrelation of b_e")
    lines!(ax, collect(lags), ac; color=CMP2)
    hlines!(ax, [0.0, 0.1]; color=(:white, 0.4), linestyle=:dash)
    k = findfirst(<(0.1), ac)
    return fig, (k === nothing ? nothing : lags[k])
end

"Sky axis of an octoplot result, and the factor from mas to its units."
function sky_axis(res)
    ax = res.axes.sky.sky
    lab = try string(ax.xlabel[]) catch; "" end
    scale = occursin("arcsec", lab) ? 1e-3 : 1.0
    return ax, scale, lab
end

"The page's overlay: octoplot with an image drawn underneath the orbits."
function orbits_over(model, chain, x, y, z, label)
    res = octoplot(model, chain)
    fig = res.figure
    ax, scale, lab = sky_axis(res)
    dbg("  sky axis label \"", lab, "\" → image axes scaled by ", scale)
    h = heatmap!(ax, x .* scale, y .* scale, z; colormap=:greys, nan_color=:transparent)
    CairoMakie.Makie.translate!(h, 0, 0, -1)       # send the image behind the orbits
    Colorbar(fig[1, 2], h, label=label)
    CairoMakie.Makie.resize_to_layout!(fig)
    return fig
end

"""
+ Per-epoch stamps: each image with the positions of b predicted by `n` posterior
draws (construct_system → orbitsolve → raoff/decoff) and by the MAP draw.
"""
function stamps_figure(model, chain, image_dat; n=150, half=nothing)
    N = size(chain, 1) * size(chain, 3)
    ii = unique(round.(Int, range(1, N, length=min(n, N))))
    lp = haspar(chain, :logpost) ? colvec(chain, :logpost) : zeros(N)
    imap = argmax(lp)
    ep = collect(Float64, image_dat.epoch)
    pos = map(vcat(ii, imap)) do i
        s = construct_system(model, chain, i)
        traj = orbitsolve(s, ep)
        (raoff.(traj, :b, :A), decoff.(traj, :b, :A))
    end
    ras = reduce(hcat, first.(pos))       # epochs × draws (last column = MAP)
    decs = reduce(hcat, last.(pos))
    sep = sqrt.(ras .^ 2 .+ decs .^ 2)
    h = half === nothing ? max(400.0, 1.3 * Statistics.quantile(vec(sep), 0.98)) : half
    nimg = length(ep)
    fig = Figure(size=(300 * nimg + 80, 400))
    local hm
    for k in 1:nimg
        x, y, z = sky_image(image_dat, k)
        ax = Axis(fig[1, k], aspect=DataAspect(), xreversed=true,
                  title=@sprintf("image %d — MJD %.1f", k, ep[k]),
                  xlabel="Δα* [mas]", ylabel=k == 1 ? "Δδ [mas]" : "")
        hm = heatmap!(ax, x, y, z; colormap=:magma, colorrange=(-1.0, 4.0), nan_color=:transparent)
        scatter!(ax, ras[k, 1:end-1], decs[k, 1:end-1]; color=(:cyan, 0.35), markersize=4)
        scatter!(ax, [ras[k, end]], [decs[k, end]]; color=:transparent, strokecolor=:lime,
                 strokewidth=2, markersize=16, marker=:circle)
        scatter!(ax, [0.0], [0.0]; marker=:star5, color=:white, markersize=10)
        limits!(ax, h, -h, -h, h)        # reversed x limits keep RA increasing to the left
    end
    Colorbar(fig[1, nimg+1], hm, label="image flux")
    Label(fig[2, 1:nimg], "cyan: predicted position of b in $(length(ii)) posterior draws; " *
          "green ring: maximum-a-posteriori draw", fontsize=12)
    med = [Statistics.median(sep[k, 1:end-1]) for k in 1:nimg]
    return fig, med, (ras[:, end], decs[:, end])
end

"+ Flux posterior histogram (page) with the prior overlaid."
function flux_figure(chain)
    f = colvec(chain, :b_flux_H)
    fig = Figure(size=(900, 460))
    ax = Axis(fig[1, 1], xlabel="flux", ylabel="density", title="b_flux_H: posterior vs prior")
    hist!(ax, f; bins=50, normalization=:pdf, color=(CMP1, 0.8), label="posterior")
    xs = range(minimum(f) - 1, maximum(f) + 1, length=300)
    lines!(ax, xs, pdf.(PRIORS.flux_H, xs); color=CMP2, linewidth=2, label="prior N(3.8, 0.5)")
    axislegend(ax, position=:rt)
    return fig
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
# Driver
# -----------------------------------------------------------------------------
"""
    run_images(; n_rounds=10, n_chains=nothing, init=true, image_idx=2, n_draws=150,
                 outdir=SCRIPT_DIR, prefix="img_", dark=true, show_plots=true)

The page's image fit and plots, plus the extras listed in the file header.
"""
function run_images(; n_rounds::Integer=envint("OCTOIMG_ROUNDS", "10"),
                      n_chains=(c = envopt("OCTOIMG_CHAINS"); c === nothing ? nothing : parse(Int, c)),
                      init::Bool=envbool("OCTOIMG_INIT", "true"),
                      image_idx::Integer=envint("OCTOIMG_IMAGE_IDX", "2"),
                      n_draws::Integer=envint("OCTOIMG_DRAWS", "150"),
                      outdir::AbstractString=SCRIPT_DIR,
                      prefix::AbstractString=PREFIX,
                      dark::Bool=true,
                      corner_dark::Bool=false,
                      show_plots::Bool=true)
    T0[] = time()
    outdir = abspath(outdir)
    mkpath(outdir)
    print_env_info(outdir, n_rounds, n_chains, init)
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

    # ---- Data --------------------------------------------------------------------
    path = stage("Example images (download once, pinned commit)") do
        fetch_once(FITS_URL, joinpath(outdir, FITS_FILE))
    end
    images = stage("Load all FITS extensions (AstroImages.load(file, :))") do
        load_images(path)
    end
    image_dat = stage("Image table (recentred, page epochs, $(PLATESCALE) mas/pixel)") do
        image_table(images)
    end
    dbg("  epochs: ", join((@sprintf("%.1f", e) for e in image_dat.epoch), ", "),
        @sprintf(" (span %.1f yr)", (maximum(EPOCHS) - minimum(EPOCHS)) / 365.25))
    let im = image_dat.image[1]
        d1, d2 = AstroImages.dims(im, 1), AstroImages.dims(im, 2)
        dbg(@sprintf("  recentred pixel ranges: x %d…%d, y %d…%d → ±%.0f mas",
                     first(d1), last(d1), first(d2), last(d2), PLATESCALE * last(d1)))
    end
    soft_stage("Image preview (page's imview, clims −1…4)") do
        savefig!(preview_figure(image_dat), "preview.png")
    end
    soft_stage("Contrast curves (1σ, 5σ) per image") do
        savefig!(contrast_figure(image_dat), "contrast.png")
    end

    # ---- Model ----------------------------------------------------------------------
    sys, image_obs = stage("Bodies, ImageObs \"SPHERE\" (measures each image's contrast), System") do
        build_system(image_dat)
    end
    DEBUG[] && display(sys)
    model = stage("Compile LogDensityModel") do
        Octofitter.LogDensityModel(sys)
    end
    DEBUG[] && display(model)
    if init
        stage("initialize!(model) (Pathfinder starting points)") do
            Octofitter.initialize!(model)
        end
    end

    # ---- Sampling -------------------------------------------------------------------
    chain, pt = stage("octofit_pigeons (n_rounds=$n_rounds" *
                      (n_chains === nothing ? "" : ", n_chains=$n_chains") * ")") do
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

    # ---- Diagnostics (page) ------------------------------------------------------------
    soft_stage("Trace plot of b_a (page)") do
        savefig!(trace_figure(chain), "trace_a.png")
    end
    thin = soft_stage("Autocorrelation of b_e (page)") do
        fig, k = autocor_figure(chain)
        savefig!(fig, "autocor_e.png")
        dbg("  autocorrelation of b_e first drops below 0.1 at lag ",
            k === nothing ? "> max lag (strongly correlated)" : k,
            k === nothing ? "" : " → thinning by ≈$k gives nearly independent samples")
        k
    end

    # ---- Analysis (page) ------------------------------------------------------------------
    soft_stage("octoplot") do
        savefig!(inout(() -> octoplot(model, chain)), "octoplot.png")
    end
    soft_stage("Orbits over image $image_idx (page)") do
        x, y, z = sky_image(image_dat, image_idx)
        savefig!(inout(() -> orbits_over(model, chain, x, y, z, "image $image_idx flux")),
                 "orbits_image$(image_idx).png")
    end
    soft_stage("Orbits over the max-stack of all images (page; NaN-safe)") do
        x, y, z = max_stack(image_dat)
        savefig!(inout(() -> orbits_over(model, chain, x, y, z, "max over images")),
                 "orbits_maxstack.png")
    end
    soft_stage("Corner plot (small=true, light theme; includes flux_H)") do
        savefig!(inout(() -> corner(model, chain; dark=corner_dark)), "corner.png")
    end
    soft_stage("b_flux_H posterior vs prior") do
        savefig!(flux_figure(chain), "flux_H.png")
    end
    stamps = soft_stage("Per-epoch stamps with predicted positions of b ($n_draws draws)") do
        fig, med, mp = stamps_figure(model, chain, image_dat; n=n_draws)
        savefig!(fig, "stamps.png")
        for k in eachindex(med)
            dbg(@sprintf("  image %d (MJD %.1f): median predicted separation %.0f mas; MAP at (Δα*, Δδ) = (%.0f, %.0f) mas",
                         k, image_dat.epoch[k], med[k], mp[1][k], mp[2][k]))
        end
        med
    end

    # ---- Detection -------------------------------------------------------------------------
    lines = String[]
    haspar(chain, :b_flux_H) && append!(lines, detection_lines(chain))
    if diag !== nothing
        push!(lines, @sprintf("Pigeons: n_rounds = %d, log(Z₁/Z₀) = %.3f, Λ = %.2f, min(α) = %.3g",
                              n_rounds, diag.logZ, diag.Λ, diag.min_α))
    end
    thin === nothing || push!(lines, "Autocorrelation of b_e < 0.1 at lag $(thin)")
    foreach(l -> dbg(l), lines)
    soft_stage("Write summary") do
        ps = joinpath(outdir, prefix * "summary.txt")
        write(ps, join(lines, "\n") * "\n")
        push!(outputs, ps)
    end

    list_outputs(outputs)
    dbg("Done. Total ", @sprintf("%.1f min", (time() - T0[]) / 60))
    return (; model, chain, pt, image_dat, outputs)
end

# -----------------------------------------------------------------------------
# Threaded relaunch (default when the REPL has 1 thread)
# -----------------------------------------------------------------------------
function relaunch_threaded(file::AbstractString; threads::AbstractString="auto",
                           outdir::AbstractString=SCRIPT_DIR, prefix::AbstractString=PREFIX)
    T0[] = time()
    base = `$(Base.julia_cmd()) --threads=$threads --project=$(Base.active_project()) $file`
    cmd = addenv(base, "OCTOIMG_CHILD" => "1")
    dbg("This session has 1 thread, and Julia can't add threads to a running process.")
    dbg("Relaunching in a child process with --threads=", threads,
        threads == "auto" ? " (→ $(Sys.CPU_THREADS) on this machine)" : "")
    dbg("  ", join(base.exec, " "), "   [+ OCTOIMG_CHILD=1]")
    dbg("  Child output follows; its plots are shown here when it finishes.")
    dbg("  Ctrl+C here interrupts the child too. Disable with ENV[\"OCTOIMG_RELAUNCH\"]=\"false\".")
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
    elseif lowercase(get(ENV, "OCTOIMG_SHOW_PLOTS", "true")) != "false"
        soft_stage("Show $(length(pngs)) saved plot(s) in the plot pane") do
            show_saved_pngs(pngs)
        end
    end
    return (; relaunched=true, success=ok, chain, outputs=new_files)
end

function png_rank(p)
    f = basename(p)
    for (k, key) in enumerate(("preview", "contrast", "trace", "autocor", "octoplot", "orbits_image",
                               "orbits_maxstack", "stamps", "corner", "flux_H"))
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

println(@sprintf("[IMG] Module loaded in %.1f s", time() - LOAD_T0))

end # module OctofitterImagesTutorial

# -----------------------------------------------------------------------------
# Auto-run
#   - `julia --threads=auto OctofitterImagesTutorial.jl`   -> runs
#   - VSCode "Execute active File in REPL"         -> runs; with 1 thread it
#     relaunches in `julia --threads=auto` (OCTOIMG_RELAUNCH=false to disable,
#     OCTOIMG_THREADS to set the count)
#   - ENV["OCTOIMG_AUTORUN"] = "false"; include()  -> loads module only
# -----------------------------------------------------------------------------
let this_file = @__FILE__(),
    as_script = abspath(PROGRAM_FILE) == this_file,
    autorun   = isinteractive() &&
                lowercase(get(ENV, "OCTOIMG_AUTORUN", "true")) != "false",
    is_child  = get(ENV, "OCTOIMG_CHILD", "") == "1",
    relaunch  = !is_child && Threads.nthreads() == 1 &&
                lowercase(get(ENV, "OCTOIMG_RELAUNCH", "true")) != "false"
    if as_script || autorun
        if relaunch
            global img_result = OctofitterImagesTutorial.relaunch_threaded(
                this_file; threads=get(ENV, "OCTOIMG_THREADS", "auto"))
            println("[IMG] Child run finished. `img_result` has fields: relaunched, success, ",
                    "chain (loaded from FITS), outputs")
        else
            global img_result = OctofitterImagesTutorial.run_images(show_plots=!is_child)
            is_child || println("[IMG] Result stored in `img_result` (fields: model, chain, pt, ",
                                "image_dat, outputs)")
        end
    end
end

nothing
