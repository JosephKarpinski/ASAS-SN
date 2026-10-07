# ==============================================================================
# OctofitterRVBank.jl
# Generic radial-velocity planet search/fit for stars in the HARPS RVBank,
# using Octofitter.jl v9.
#
# Run from VSCode ("Execute active file in REPL") or from a shell:
#     julia OctofitterRVBank.jl            # default: first star in the list (GJ 876)
#     julia OctofitterRVBank.jl 3          # star number 3 in the list
#     julia OctofitterRVBank.jl HD10700    # any list entry, or any RVBank target name
#
# ABOUT THE DATA
#   The HARPS RVBank data for GJ 876 is a high-precision, systematically corrected
#   radial-velocity (RV) dataset for the nearby M-dwarf star Gliese 876 (GJ 876).
#   [1]
#   The data originates from HARPS-RVBank, a massive public archive compiled by
#   astronomers (originally published by Trifonov et al.) that re-processes all
#   raw public spectra from the HARPS spectrograph (located at the ESO 3.6 m
#   telescope at La Silla Observatory, Chile). [2]
#
#   The same catalogue holds RVs for hundreds of other stars (this script uses
#   the same column for all of them), so everything below works for any target.
#
#   [1] https://exo-restart.com/wp-content/uploads/Stellar_parameters/HARPS_RVBank.html
#   [2] https://www.aanda.org/articles/aa/abs/2020/04/aa36686-19/aa36686-19.html
#   Catalogue:  https://www2.mpia-hd.mpg.de/homes/trifonov/HARPS_RVBank.html
#   2023 update (ESO/HARPS RV catalogue): https://arxiv.org/abs/2312.06586
#   License CC0-1.0. Please credit Trifonov et al. when you use the data; the
#   catalogue also prints its citation the first time it is downloaded (~38 MB).
#
# WHAT THIS SCRIPT DOES
#   1. Prints the list of 25 stars and marks the one selected (the first by default)
#   2. Loads that star's RVs (binned per night if there are very many points)
#   3. Finds candidate planet periods by Lomb-Scargle "prewhitening": take the
#      strongest periodogram peak, subtract a sinusoid at that period, repeat
#   4. Fits models with 1..N_PLANETS planets with Pigeons (parallel tempering), each
#      planet's period prior being a window around its periodogram peak
#   5. Compares the models with the log Bayesian evidence and saves all plots
#      as PNGs in ./figures_rvbank_<star> next to this script
#   6. Two optional extra tests (switches below, default on; they correspond to the
#      two exercises in OctofitterGJ876.jl):
#        - eccentricity test: refit the N_PLANETS model with the opposite eccentricity
#          setting (circular orbits if FREE_ECCENTRICITY is true) and compare the evidence
#        - extra-planet test: add one more planet at the next periodogram peak and
#          compare the evidence with the N_PLANETS model
#
# THE LIST
#   It is a curated list, not a ranking computed from the catalogue: 25 well-known
#   planet hosts that have at least ~180 HARPS RVs in RVBank ver02, with GJ 876
#   first. The number of RVs and baseline per star were read from the catalogue file
#   (HARPS_RVBank_ver02.csv); the script prints the real count again after loading.
#   Stellar masses are rough values (used only to turn RV amplitudes into planet
#   masses) -- check them against the literature before quoting any planet mass.
#   The one-line notes are reminders, not authoritative. To add a star, append a
#   line to STAR_LIST (the first field must match RVBank's `target` column exactly).
#
# CAVEATS
#   * Periods come from a sinusoid search, so a strongly eccentric or closely
#     interacting system may need a different PERIOD_WINDOW or N_PLANETS.
#   * Planets are named b, c, d... in the order they were detected (strongest
#     signal first), not by period. This keeps names consistent between the
#     1-, 2-, ... planet fits.
#   * Masses are minimum masses (m sin i): RVs cannot give the inclination.
#
# Changelog
#   v1.0.0  2026-10-07  First version. Built from OctofitterGJ876.jl v1.1.0.
#   v1.1.0  2026-10-07  Added the two tests the GJ876 script had as exercises, as
#     switches that default to true: RUN_ECCENTRICITY_TEST (Plots 7-8) and
#     RUN_EXTRA_PLANET_TEST (Plots 9-10). v1.0.0 had only the core analysis
#     (Plots 1-6). Each test adds one Pigeons run, the extra-planet one being the
#     slowest. A failure in either is caught and the rest of the run continues.
# ==============================================================================

import Pkg
using Dates
using Printf

# ------------------------------------------------------------------ settings
const TARGET_INDEX      = 1       # which star in STAR_LIST to use (1 = GJ 876)
const TARGET_OVERRIDE   = ""      # or any RVBank target name, e.g. "HD10700" (beats TARGET_INDEX)
const N_PLANETS         = 2       # fit models with 1..N_PLANETS planets (GJ 876: 2 reproduces OctofitterGJ876.jl)
const FIT_ALL_SUBMODELS = true    # true: fit 1..N_PLANETS and compare; false: only the N_PLANETS model
const FREE_ECCENTRICITY = true    # true: every planet gets e ~ U(0, 0.7); false: circular orbits
const RUN_ECCENTRICITY_TEST = true  # also fit the N_PLANETS model with the opposite eccentricity setting (adds Plots 7-8)
const RUN_EXTRA_PLANET_TEST = true  # also fit N_PLANETS+1 planets, the extra one at the next periodogram peak (adds Plots 9-10)
const PERIOD_WINDOW     = 0.10    # period prior = peak period +/- this fraction
const MIN_PERIOD        = 1.5     # [d] ignore shorter periodogram peaks (1 d peaks are daily-sampling aliases)
const BIN_NIGHTLY       = :auto   # :auto (bin only if > MAX_POINTS RVs), true, or false
const MAX_POINTS        = 1000
const CUSTOM_STAR_MASS  = 1.0     # [Msun] used only for a star that is not in STAR_LIST
const N_ROUNDS          = 10      # Pigeons rounds; each extra round doubles the work (9 is a quick look)
const USE_TEMP_ENV      = true    # true: throw-away Octofitter v9 env; false: keep one in ./.octofitter_v9_env
const USE_APPLE_ACCELERATE = true # Apple Silicon only; set false if some FFT call complains
const T_START           = time()

# ------------------------------------------------------------------ the list
# (RVBank name, common name, star mass [Msun], mass uncertainty, number of RVs, baseline [yr], note)
const STAR_LIST = [
    ("GJ876",    "GJ 876",             0.346, 0.007,  270, 15.7, "M4; two giant planets in a 2:1 resonance (61 d, 30 d) plus smaller ones"),
    ("GJ581",    "GJ 581",             0.30,  0.03,   250,  8.0, "M3; several low-mass planets (some signals debated)"),
    ("GJ667",    "GJ 667 C",           0.33,  0.03,   272, 15.3, "M1.5; super-Earths, in a triple-star system"),
    ("GJ436",    "GJ 436",             0.45,  0.045,  192, 14.2, "M2.5; hot Neptune (2.6 d)"),
    ("GJ674",    "GJ 674",             0.35,  0.035,  256, 15.4, "M2.5; Neptune-mass planet at 4.7 d"),
    ("GJ163",    "GJ 163",             0.40,  0.04,   207, 18.2, "M3; multiple low-mass planets"),
    ("GJ551",    "Proxima Cen",        0.12,  0.012,  414, 15.3, "M5.5; nearest star, planets b and d"),
    ("GJ699",    "Barnard's Star",     0.16,  0.016,  335, 12.5, "M4; sub-Earth planet candidates"),
    ("HD10180",  "HD 10180",           1.06,  0.1,    336, 13.8, "G1; rich system of several planets"),
    ("HD20794",  "HD 20794 (82 Eri)",  0.81,  0.08,  6824, 18.2, "G8; super-Earth candidates; very many RVs (binned per night)"),
    ("HD85512",  "HD 85512",           0.69,  0.07,  1184, 17.9, "K6; low-mass planet candidate"),
    ("HD10700",  "tau Ceti",           0.78,  0.08, 11632, 18.1, "G8; very quiet star, weak signals; very many RVs (binned per night)"),
    ("HD136352", "HD 136352 (nu2 Lup)",0.87,  0.09,   677, 13.2, "G; three small planets"),
    ("HD1461",   "HD 1461",            1.03,  0.1,    487, 17.8, "G0; super-Earth"),
    ("HD134060", "HD 134060",          1.10,  0.11,   339, 13.2, "G0; two small planets"),
    ("HD204961", "GJ 832 (HD 204961)", 0.45,  0.045,  183, 16.1, "M1.5; cold giant planet plus a candidate"),
    ("HD82943",  "HD 82943",           1.18,  0.12,   255, 13.4, "G; two giant planets in 2:1 resonance"),
    ("GJ3293",   "GJ 3293",            0.42,  0.04,   217,  6.3, "M2.5; several planets"),
    ("GJ1061",   "GJ 1061",            0.12,  0.012,  184, 15.3, "M5.5; three small planets"),
    ("GJ273",    "Luyten's Star",      0.29,  0.03,   318, 12.8, "M3.5; two small planets"),
    ("GJ54.1",   "YZ Ceti",            0.13,  0.013,  334, 14.8, "M4.5; small planets"),
    ("HD45184",  "HD 45184",           1.03,  0.1,    355, 17.9, "G2; two small planets"),
    ("HD96700",  "HD 96700",           0.90,  0.09,   366, 13.5, "G0; several small planets"),
    ("HD134606", "HD 134606",          1.00,  0.1,    290, 12.8, "G3; three small planets"),
    ("GJ785",    "HD 192310 (GJ 785)", 0.78,  0.08,  1821, 12.9, "K3; Neptune-mass planet plus a candidate; many RVs (binned per night)"),
]

dbg(args...) = println("[DEBUG ", Dates.format(now(), "HH:MM:SS"), "] ", args...)
section(title) = println("\n", "="^78, "\n", title, "\n", "="^78)

# ------------------------------------------------------------------ choose the star
normname(s) = lowercase(replace(string(s), r"[\s_\-]" => ""))

function resolve_target()
    spec = TARGET_OVERRIDE != "" ? TARGET_OVERRIDE :
           !isempty(ARGS)        ? ARGS[1] : TARGET_INDEX
    spec = string(spec)
    idx = tryparse(Int, spec)
    if idx !== nothing
        1 <= idx <= length(STAR_LIST) || error("Star number $idx is outside 1..$(length(STAR_LIST))")
        return idx, STAR_LIST[idx]
    end
    for (i, s) in enumerate(STAR_LIST)
        if normname(spec) in (normname(s[1]), normname(s[2]))
            return i, s
        end
    end
    # not in the list: use the name as given as an RVBank target
    return 0, (spec, spec, CUSTOM_STAR_MASS, 0.1 * CUSTOM_STAR_MASS, 0, 0.0, "custom target (not in STAR_LIST)")
end

section("HARPS RVBank target list")
const _SEL_PAIR = resolve_target()
const SEL_IDX = _SEL_PAIR[1]
const SEL     = _SEL_PAIR[2]
println(rpad("#", 4), rpad("RVBank name", 11), rpad("Star", 21), lpad("RVs", 6), lpad("yr", 6),
        lpad("Msun", 7), "  Notes")
for (i, s) in enumerate(STAR_LIST)
    mark = i == SEL_IDX ? "->" : "  "
    println(mark, rpad(string(i), 2), rpad(s[1], 11), rpad(s[2], 21), lpad(string(s[5]), 6),
            lpad(@sprintf("%.1f", s[6]), 6), lpad(@sprintf("%.2f", s[3]), 7), "  ", s[7])
end
SEL_IDX == 0 && println("->   ", rpad(SEL[1], 11), "(custom target; star mass ", CUSTOM_STAR_MASS, " Msun assumed)")

const TARGET   = SEL[1]
const STARNAME = SEL[2]
const M_STAR   = SEL[3]
const M_STAR_ERR = SEL[4]
const FIGDIR   = joinpath(@__DIR__, "figures_rvbank_" * replace(TARGET, r"[^A-Za-z0-9._-]" => "_"))
mkpath(FIGDIR)
println("\nSelected: ", STARNAME, "   (RVBank target \"", TARGET, "\")")
println("To pick another star: change TARGET_INDEX / TARGET_OVERRIDE above, or pass a number or name when running.")

# ------------------------------------------------------------------ environment
section("0. Environment")
if USE_TEMP_ENV
    Pkg.activate(temp = true)
else
    Pkg.activate(joinpath(@__DIR__, ".octofitter_v9_env"))
end
dbg("Julia ", VERSION, "  |  active project: ", Base.active_project())

const DEPS = Pkg.PackageSpec[
    Pkg.PackageSpec(name = "Octofitter", version = "9"),
    Pkg.PackageSpec(name = "OctofitterRadialVelocity", version = "9"),
    Pkg.PackageSpec(name = "PlanetOrbits"),
    Pkg.PackageSpec(name = "CairoMakie"),
    Pkg.PackageSpec(name = "PairPlots"),
    Pkg.PackageSpec(name = "Distributions"),
    Pkg.PackageSpec(name = "Pigeons"),
    Pkg.PackageSpec(name = "LombScargle"),
]
(Sys.isapple() && USE_APPLE_ACCELERATE) && push!(DEPS, Pkg.PackageSpec(name = "AppleAccelerate"))

let have = keys(Pkg.project().dependencies)
    todo = filter(s -> s.name ∉ have, DEPS)
    if isempty(todo)
        dbg("All packages already present in this environment")
    else
        dbg("Installing: ", join((s.name for s in todo), ", "))
        Pkg.add(todo)
    end
end

if Sys.isapple() && USE_APPLE_ACCELERATE
    using AppleAccelerate
end

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"   # let DataDeps download the HARPS catalogue without prompting

using Octofitter
using OctofitterRadialVelocity
using PlanetOrbits
using CairoMakie
using PairPlots
using Distributions
using Pigeons
using LombScargle
using Statistics

CairoMakie.activate!(type = "png")

for (_, info) in Pkg.dependencies()
    info.name in ("Octofitter", "OctofitterRadialVelocity", "PlanetOrbits",
                  "Pigeons", "CairoMakie", "PairPlots", "LombScargle") &&
        dbg(rpad(info.name, 26), "v", info.version)
end

# ------------------------------------------------------------------ helpers
"Run `f`; on failure print the error and carry on (so one bad plot cannot lose a long sampling run)."
function safe(f, label)
    try
        f()
        dbg(label, " OK")
    catch err
        @warn "$label FAILED -- continuing" exception = (err, catch_backtrace())
    end
    return nothing
end

"""
Add a wrapped title in a new row above everything in `fig`, and grow the figure by
exactly that row's height so the existing panels keep their size.
"""
function add_plot_title!(fig, title; fontsize = 14)
    try
        dims = try size(fig.scene) catch; nothing end     # (width, height) in pixels
        w = dims === nothing ? 900 : dims[1]
        chars_per_line = max(20, floor(Int, (w - 40) / (fontsize * 0.62)))
        nlines = max(1, ceil(Int, length(title) / chars_per_line))
        title_h = nlines * fontsize * 1.4 + 10
        Label(fig[0, :], title; fontsize, font = :bold, halign = :left,
              justification = :left, word_wrap = true, tellwidth = false,
              tellheight = true, height = title_h, padding = (8, 8, 6, 2))
        dims === nothing || Makie.resize!(fig, Int(w), round(Int, dims[2] + title_h + 8))
    catch err
        @warn "Could not add the plot title (saving without it)" exception = err
    end
    return fig
end

"Add a title, save as PNG into FIGDIR, and return the path. Accepts a Figure or an OctoPlotResult."
function save_plot!(obj, title, filename)
    fig = obj isa OctoPlotResult ? obj.figure : obj
    add_plot_title!(fig, title)
    path = joinpath(FIGDIR, filename)
    Makie.save(path, fig; px_per_unit = 2)
    dbg("saved ", path)
    return path
end

has_col(chain, name::AbstractString) = Symbol(name) in names(chain)
chain_vec(chain, name::AbstractString) = Float64.(chain[name][:])

"Median and 16/84 percent range of a chain column, as a short string."
function describe_var(chain, name::AbstractString; scale = 1.0, digits = 4)
    v = chain_vec(chain, name) .* scale
    q = quantile(v, [0.16, 0.5, 0.84])
    return "$(round(q[2]; digits)) +$(round(q[3] - q[2]; digits)) / -$(round(q[2] - q[1]; digits))"
end

function bf_interpretation(lnbf; more = "the larger model", less = "the smaller model")
    isfinite(lnbf) || return "evidence not available"
    a = abs(lnbf)
    a == 0 && return "no evidence either way"
    strength = a > 3.0 ? "Extreme" : a > 1.61 ? "Very strong" : a > 1.10 ? "Strong" :
               a > 0.69 ? "Moderate" : "Anecdotal"
    return "$strength evidence for " * (lnbf > 0 ? more : less)
end

# ------------------------------------------------------------------ 1. data
section("1. Loading the HARPS RVBank data for $STARNAME")
# Table with epoch [MJD], rv [m/s], sigma_rv [m/s]. The first call downloads the
# catalogue (~38 MB); credit the source it prints.
rv_raw = OctofitterRadialVelocity.HARPS_RVBank_rvs(TARGET)
if isempty(rv_raw.epoch)
    # HARPS_RVBank_rvs silently returns nothing for an unknown name; this call
    # prints the closest names the catalogue does have.
    try OctofitterRadialVelocity.HARPS_RVBank_observations(TARGET) catch; end
    error("No RVs found in HARPS RVBank for target \"$TARGET\". See the similar names above.")
end

# Drop rows with missing/non-positive values, then sort by time.
let ok = findall(i -> isfinite(rv_raw.epoch[i]) && isfinite(rv_raw.rv[i]) &&
                      isfinite(rv_raw.σ_rv[i]) && rv_raw.σ_rv[i] > 0, eachindex(rv_raw.epoch))
    global rv_clean = Table(epoch = rv_raw.epoch[ok][sortperm(rv_raw.epoch[ok])],
                            rv = rv_raw.rv[ok][sortperm(rv_raw.epoch[ok])],
                            σ_rv = rv_raw.σ_rv[ok][sortperm(rv_raw.epoch[ok])])
end

"Inverse-variance mean of all RVs taken on the same (UT) night."
function bin_nightly(d)
    groups = Dict{Int,Vector{Int}}()
    for i in eachindex(d.epoch)
        push!(get!(groups, floor(Int, d.epoch[i]), Int[]), i)
    end
    ep = Float64[]; rv = Float64[]; er = Float64[]
    for night in sort!(collect(keys(groups)))
        idx = groups[night]
        w = 1 ./ d.σ_rv[idx] .^ 2
        push!(ep, sum(w .* d.epoch[idx]) / sum(w))
        push!(rv, sum(w .* d.rv[idx]) / sum(w))
        push!(er, 1 / sqrt(sum(w)))
    end
    return Table(epoch = ep, rv = rv, σ_rv = er)
end

do_bin = BIN_NIGHTLY === :auto ? length(rv_clean.epoch) > MAX_POINTS : BIN_NIGHTLY === true
rv_data = do_bin ? bin_nightly(rv_clean) : rv_clean

const NPTS     = length(rv_data.epoch)
const BASELINE = maximum(rv_data.epoch) - minimum(rv_data.epoch)
const T_REF    = round(mean(rv_data.epoch))     # reference epoch for the orbital phases

dbg("RVs in the catalogue for this star: ", length(rv_raw.epoch), "  (usable: ", length(rv_clean.epoch), ")")
do_bin && dbg("Binned per night: ", length(rv_clean.epoch), " -> ", NPTS, " points")
NPTS > MAX_POINTS && @warn "$NPTS points is a lot for the sampler; consider BIN_NIGHTLY = true or a lower N_ROUNDS"
dbg("Epoch range [MJD]: ", round(minimum(rv_data.epoch); digits = 1), " to ",
    round(maximum(rv_data.epoch); digits = 1), "  (baseline ", round(BASELINE; digits = 1), " d = ",
    round(BASELINE / 365.25; digits = 1), " yr)")
dbg("RV range [m/s]: ", round(minimum(rv_data.rv); digits = 1), " to ", round(maximum(rv_data.rv); digits = 1))
dbg("Median RV uncertainty [m/s]: ", round(median(rv_data.σ_rv); digits = 2))
dbg("RV scatter (robust, from the median absolute deviation) [m/s]: ",
    round(1.4826 * median(abs.(rv_data.rv .- median(rv_data.rv))); digits = 2))

safe("Plot 1") do
    fig = Figure(size = (900, 450))
    ax = Axis(fig[1, 1], xlabel = "epoch [MJD]", ylabel = "RV [m/s]")
    Makie.scatter!(ax, rv_data.epoch, rv_data.rv, color = :black, markersize = 5)
    Makie.errorbars!(ax, rv_data.epoch, rv_data.rv, rv_data.σ_rv, color = (:black, 0.4))
    save_plot!(fig, "Plot 1. HARPS RVBank radial velocities of $STARNAME ($NPTS points" *
                    (do_bin ? ", binned per night" : "") * "; error bars are the pipeline uncertainties).",
               "plot01_rv_data.png")
end

# ------------------------------------------------------------------ 2. candidate periods
section("2. Candidate periods by periodogram prewhitening")
const MAX_PERIOD = BASELINE / 2        # need at least two cycles to call something a period
const K_MAX = N_PLANETS + (RUN_EXTRA_PLANET_TEST ? 1 : 0)   # periods to find (the extra one is for the extra-planet test)

"""
Run Lomb-Scargle on `y`; return the period grid, the power, and the period of the
strongest local maximum between MIN_PERIOD and MAX_PERIOD.
`fast = false` skips LombScargle's FFT approximation: with AppleAccelerate loaded the
FFT planner only accepts power-of-2 sizes ("vDSP FFT requires power-of-2 dimensions").
"""
function periodogram_stage(t, y, σ)
    pg = lombscargle(t, y, σ; minimum_frequency = 1 / MAX_PERIOD, maximum_frequency = 1 / MIN_PERIOD,
                     samples_per_peak = 10, fast = false)
    per, pow = LombScargle.periodpower(pg)
    peaks = [i for i in 2:length(pow)-1 if pow[i] > pow[i-1] && pow[i] >= pow[i+1]]
    best = isempty(peaks) ? argmax(pow) : peaks[argmax(pow[peaks])]
    return per, pow, per[best]
end

"Find `n` periods: strongest peak, subtract a best-fit sinusoid there, repeat on the residuals."
function find_periods(t, y, σ, n)
    resid = copy(y)
    stages = NamedTuple[]
    for k in 1:n
        per, pow, P = periodogram_stage(t, resid, σ)
        push!(stages, (; per, pow, P))
        resid = resid .- LombScargle.model(t, resid, σ, 1 / P)   # the model includes the mean, which is fine
    end
    return stages
end

# Returned from a function and assigned outside the `try`: a `try` block is its own
# scope, so assigning inside it would leave these globals empty.
stages = try
    find_periods(rv_data.epoch, rv_data.rv, rv_data.σ_rv, K_MAX)
catch err
    @warn "Periodogram prewhitening FAILED -- cannot choose period priors" exception = (err, catch_backtrace())
    NamedTuple[]
end
isempty(stages) && error("No candidate periods could be found; see the warning above.")
const PERIOD_GUESSES = [s.P for s in stages]
const PLANET_NAMES = ["b", "c", "d", "e", "f", "g", "h"]
K_MAX <= length(PLANET_NAMES) || error("N_PLANETS (+1 for the extra-planet test) must be at most $(length(PLANET_NAMES))")

for (k, s) in enumerate(stages)
    dbg("Planet ", PLANET_NAMES[k], k > N_PLANETS ? " (extra-planet test)" : "", ": strongest peak at P = ",
        round(s.P; digits = 3), " d (power ", round(maximum(s.pow); digits = 3), ")")
end

safe("Plot 2") do
    fig = Figure(size = (900, 260 * length(stages) + 40))
    for (k, s) in enumerate(stages)
        ax = Axis(fig[k, 1], xlabel = k == length(stages) ? "period [d]" : "", ylabel = "power",
                  xscale = log10,
                  title = k == 1 ? "data" : "residuals after removing " *
                                    join(("$(round(PERIOD_GUESSES[j]; digits = 2)) d" for j in 1:k-1), ", "))
        Makie.lines!(ax, s.per, s.pow, color = :black)
        Makie.vlines!(ax, [s.P], color = (:crimson, 0.6), linestyle = :dash)
        Makie.text!(ax, s.P, maximum(s.pow); text = " $(PLANET_NAMES[k]): $(round(s.P; digits = 2)) d",
                    align = (:left, :top), color = :crimson)
    end
    save_plot!(fig, "Plot 2. Lomb-Scargle prewhitening of $STARNAME. Each panel shows the periodogram " *
                    "of what is left after removing the sinusoids found above it; the dashed line is " *
                    "the strongest peak, used as the centre of that planet's period prior.",
               "plot02_periodogram_prewhitening.png")
end

# ------------------------------------------------------------------ 3. model pieces
section("3. Model definition")

# Host star: mass prior from STAR_LIST (only scales the planet masses).
A = Body(
    name = "A",
    variables = @variables begin
        mass ~ truncated(Normal(M_STAR, M_STAR_ERR), lower = 0.05)   # M_sun
    end
)

"One planet orbiting the star, with its period prior centred on the periodogram peak."
function make_planet(name, P_guess; free_e = FREE_ECCENTRICITY)
    P_lo, P_hi = P_guess * (1 - PERIOD_WINDOW), P_guess * (1 + PERIOD_WINDOW)
    tref = T_REF
    if free_e
        return Body(
            name = name,
            about = A,
            variables = @variables begin
                i = pi / 2             # RV-only: inclination and node are unconstrained,
                Ω = 0.0                # so with i = pi/2 the fitted mass is a minimum mass
                e ~ Uniform(0, 0.7)
                ω ~ Uniform(0, 2pi)
                P ~ Uniform(P_lo, P_hi)                  # days
                τ ~ UniformCircular(1.0)                 # orbital phase in [0, 1)
                tp = τ * P + $tref                       # periastron time, MJD
                mass ~ LogUniform(0.0001mjup, 20mjup)    # minimum mass, M_sun
            end
        )
    else
        return Body(
            name = name,
            about = A,
            variables = @variables begin
                i = pi / 2
                Ω = 0.0
                e = 0.0
                ω = 0.0
                P ~ Uniform(P_lo, P_hi)
                τ ~ UniformCircular(1.0)
                tp = τ * P + $tref
                mass ~ LogUniform(0.0001mjup, 20mjup)
            end
        )
    end
end

# One RV series with its zero point marginalized out and a free jitter term.
rv_likelihood = MarginalizedRVObs(
    rv_data;
    target = A, ref = Barycentre,
    name = "HARPS",
    variables = @variables begin
        jitter ~ LogUniform(0.1, 100)                    # m/s
    end
)
planets = [make_planet(PLANET_NAMES[k], PERIOD_GUESSES[k]) for k in 1:K_MAX]
# same planets with the opposite eccentricity setting, for the eccentricity test
planets_alt = RUN_ECCENTRICITY_TEST ?
    [make_planet(PLANET_NAMES[k], PERIOD_GUESSES[k]; free_e = !FREE_ECCENTRICITY) for k in 1:N_PLANETS] : Body[]
dbg("Built the star, ", K_MAX, " planet(s) (", N_PLANETS, " for the main fit) and the HARPS likelihood")
dbg("Period priors [d]: ", join(("$(PLANET_NAMES[k]): $(round(PERIOD_GUESSES[k] * (1 - PERIOD_WINDOW); digits = 2))" *
                                 "-$(round(PERIOD_GUESSES[k] * (1 + PERIOD_WINDOW); digits = 2))" for k in 1:K_MAX), ",  "))

# ------------------------------------------------------------------ 4. fits
section("4. Fits with Pigeons")
"Fit the model with the first `k` planets; returns (k, model, chain, pt, lnZ) or `nothing` if it fails."
function fit_planets(k, plist = planets; label = "")
    try
        sys = System(name = "$(replace(TARGET, r"[^A-Za-z0-9]" => ""))_$(k)_planet" * (k == 1 ? "" : "s") *
                            (label == "" ? "" : "_" * label),
                     bodies = [A, plist[1:k]...], observations = [rv_likelihood])
        model = Octofitter.LogDensityModel(sys)
        dbg("$k-planet model compiled; sampling with Pigeons, n_rounds = ", N_ROUNDS)
        t0 = time()
        chain, pt = octofit_pigeons(model, n_rounds = N_ROUNDS)
        dbg("$k-planet sampling took ", round(time() - t0; digits = 1), " s")
        return (; k, model, chain, pt, lnZ = Pigeons.stepping_stone(pt))
    catch err
        @warn "$k-planet fit FAILED -- continuing" exception = (err, catch_backtrace())
        return nothing
    end
end

fits = NamedTuple[]
for k in (FIT_ALL_SUBMODELS ? (1:N_PLANETS) : (N_PLANETS:N_PLANETS))
    section("4.$k  $k-planet model")
    f = fit_planets(k)
    f === nothing && continue
    push!(fits, f)
    display(f.chain)
    dbg("ln Z ($k planet" * (k == 1 ? "" : "s") * ") = ", round(f.lnZ; digits = 2))
end
isempty(fits) && error("Every fit failed; see the warnings above.")
final = fits[end]
const K_FINAL = final.k

safe("Plot 3") do
    res = rvplot(final.model, final.chain)
    save_plot!(res, "Plot 3. $(K_FINAL)-planet fit of $STARNAME: RV curve for the maximum-posterior " *
                    "draw, residuals, and the phase-folded signal of each planet.",
               "plot03_rvplot_$(K_FINAL)_planets.png")
end
safe("Plot 4") do
    res = octoplot(final.model, final.chain)
    save_plot!(res, "Plot 4. $(K_FINAL)-planet fit of $STARNAME: spread of orbits allowed by the " *
                    "posterior over the HARPS data.", "plot04_octoplot_$(K_FINAL)_planets.png")
end

# ------------------------------------------------------------------ 5. model comparison
section("5. Model comparison: log Bayesian evidence")
println(rpad("planets", 9), rpad("ln Z", 12), "ln BF vs the previous model")
for (j, f) in enumerate(fits)
    if j == 1
        println(rpad(f.k, 9), rpad(round(f.lnZ; digits = 2), 12), "-")
    else
        p = fits[j-1]
        lnbf = f.lnZ - p.lnZ
        println(rpad(f.k, 9), rpad(round(f.lnZ; digits = 2), 12), round(lnbf; digits = 2), "  ->  ",
                bf_interpretation(lnbf; more = "$(f.k) planets", less = "$(p.k) planet" * (p.k == 1 ? "" : "s")))
    end
end
println("(Stepping-stone estimates are only good to a few tenths in ln Z, so differences of about 1 or less are a tie.)")

# ------------------------------------------------------------------ 6. corner plots
section("6. Corner plots")
safe("Plot 5") do
    fig = octocorner(final.model, final.chain, small = true)
    save_plot!(fig, "Plot 5. Corner plot of the $(K_FINAL)-planet posterior for $STARNAME, showing parameter covariances.",
               "plot05_corner_$(K_FINAL)_planets.png")
end
if length(fits) >= 2
    safe("Plot 6") do
        # the larger model contains every column of the smaller one
        fig = octocorner(fits[end].model, fits[end-1].chain, fits[end].chain, small = true)
        save_plot!(fig, "Plot 6. How adding planet $(PLANET_NAMES[K_FINAL]) changes the other parameters. " *
                        "Series 1 (blue) is the $(fits[end-1].k)-planet fit, series 2 (orange) the $(K_FINAL)-planet fit.",
                   "plot06_corner_$(fits[end-1].k)_vs_$(K_FINAL)_planets.png")
    end
end

# ------------------------------------------------------------------ 7. eccentricity test
fit_alt = nothing
if RUN_ECCENTRICITY_TEST && K_FINAL == N_PLANETS
    alt_name = FREE_ECCENTRICITY ? "circular" : "free-eccentricity"
    section("7. Eccentricity test: the same $(N_PLANETS)-planet model with $alt_name orbits")
    fit_alt = fit_planets(N_PLANETS, planets_alt; label = alt_name == "circular" ? "circ" : "ecc")
    if fit_alt !== nothing
        display(fit_alt.chain)
        free_fit = FREE_ECCENTRICITY ? final : fit_alt
        circ_fit = FREE_ECCENTRICITY ? fit_alt : final
        lnbf_ecc = free_fit.lnZ - circ_fit.lnZ
        dbg("ln Z ($alt_name, $(N_PLANETS) planets) = ", round(fit_alt.lnZ; digits = 2))
        dbg("ln BF (free eccentricity vs circular) = ", round(lnbf_ecc; digits = 2), "  ->  ",
            bf_interpretation(lnbf_ecc; more = "free eccentricity", less = "circular orbits"))
        safe("Plot 7") do
            res = rvplot(fit_alt.model, fit_alt.chain)
            save_plot!(res, "Plot 7. Eccentricity test: $(N_PLANETS)-planet fit of $STARNAME with $alt_name orbits: " *
                            "RV curve for the maximum-posterior draw, residuals, and phase-folded signals.",
                       "plot07_rvplot_$(N_PLANETS)_planets_$(alt_name).png")
        end
        safe("Plot 8") do
            # the free-eccentricity model has every column of the circular one
            fig = octocorner(free_fit.model, circ_fit.chain, free_fit.chain, small = true)
            save_plot!(fig, "Plot 8. Effect of allowing eccentricity on the $(N_PLANETS)-planet fit of $STARNAME. " *
                            "Series 1 (blue) is the circular fit, series 2 (orange) the free-eccentricity fit.",
                       "plot08_corner_circular_vs_eccentric.png")
        end
    end
end

# ------------------------------------------------------------------ 8. extra-planet test
fit_extra = nothing
if RUN_EXTRA_PLANET_TEST && K_FINAL == N_PLANETS
    k_extra = N_PLANETS + 1
    section("8. Extra-planet test: $(k_extra) planets (planet $(PLANET_NAMES[k_extra]) at the next periodogram peak, " *
            "$(round(PERIOD_GUESSES[k_extra]; digits = 2)) d)")
    fit_extra = fit_planets(k_extra)
    if fit_extra !== nothing
        display(fit_extra.chain)
        lnbf_extra = fit_extra.lnZ - final.lnZ
        dbg("ln Z ($(k_extra) planets) = ", round(fit_extra.lnZ; digits = 2))
        dbg("ln BF ($(k_extra) vs $(N_PLANETS) planets) = ", round(lnbf_extra; digits = 2), "  ->  ",
            bf_interpretation(lnbf_extra; more = "$(k_extra) planets", less = "$(N_PLANETS) planet" * (N_PLANETS == 1 ? "" : "s")))
        safe("Plot 9") do
            res = rvplot(fit_extra.model, fit_extra.chain)
            save_plot!(res, "Plot 9. Extra-planet test: $(k_extra)-planet fit of $STARNAME (planets " *
                            "$(join(PLANET_NAMES[1:k_extra], ", "))): RV curve for the maximum-posterior draw, " *
                            "residuals, and phase-folded signals.",
                       "plot09_rvplot_$(k_extra)_planets.png")
        end
        safe("Plot 10") do
            fig = octocorner(fit_extra.model, final.chain, fit_extra.chain, small = true)
            save_plot!(fig, "Plot 10. Effect of adding planet $(PLANET_NAMES[k_extra]) to $STARNAME. Series 1 (blue) " *
                            "is the $(N_PLANETS)-planet fit, series 2 (orange) the $(k_extra)-planet fit.",
                       "plot10_corner_$(N_PLANETS)_vs_$(k_extra)_planets.png")
        end
    end
end

# ------------------------------------------------------------------ 9. analysis summary
section("9. Analysis summary")
safe("Summary") do
    println("Star: ", STARNAME, " (RVBank \"", TARGET, "\"), assumed mass ", M_STAR, " +/- ", M_STAR_ERR, " Msun")
    println("Data: ", NPTS, " RVs", do_bin ? " (nightly bins)" : "", " over ", round(BASELINE; digits = 0), " days")
    println("Periodogram peaks used as period guesses (days): ",
            join((round(p; digits = 2) for p in PERIOD_GUESSES), ", "))
    println()
    function planet_block(chain, planet_idx)
        for k in planet_idx
            nm = PLANET_NAMES[k]
            println("  ", nm, "  period [d]        = ", describe_var(chain, "$(nm)_P"))
            (has_col(chain, "$(nm)_e") && std(chain_vec(chain, "$(nm)_e")) > 0) &&
                println("  ", nm, "  eccentricity      = ", describe_var(chain, "$(nm)_e"))
            println("  ", nm, "  min. mass [Mjup]  = ", describe_var(chain, "$(nm)_mass"; scale = 1 / mjup),
                    "   (", round(median(chain_vec(chain, "$(nm)_mass")) / mjup * 317.83; digits = 1), " Earth masses)")
        end
        println("  HARPS jitter [m/s]   = ", describe_var(chain, "HARPS_jitter"))
    end
    println("$(K_FINAL)-planet fit (median, 16/84 percent range):")
    planet_block(final.chain, 1:K_FINAL)
    println()
    if length(fits) >= 2
        println("Jitter by number of planets (median, m/s): ",
                join(("$(f.k): $(round(median(chain_vec(f.chain, "HARPS_jitter")); digits = 2))" for f in fits), ",  "),
                "   (a large drop means the extra planet explains real signal, not noise)")
        println("Evidence: ", join(("ln Z($(f.k)) = $(round(f.lnZ; digits = 2))" for f in fits), ",  "))
    end
    if fit_alt !== nothing
        free_fit = FREE_ECCENTRICITY ? final : fit_alt
        circ_fit = FREE_ECCENTRICITY ? fit_alt : final
        lnbf_ecc = free_fit.lnZ - circ_fit.lnZ
        println()
        println("Eccentricity test (", N_PLANETS, " planets): ln Z(circular) = ", round(circ_fit.lnZ; digits = 2),
                ", ln Z(free e) = ", round(free_fit.lnZ; digits = 2), ", ln BF = ", round(lnbf_ecc; digits = 2),
                "  ->  ", bf_interpretation(lnbf_ecc; more = "free eccentricity", less = "circular orbits"))
        println("  Free-eccentricity fit:")
        for k in 1:N_PLANETS
            has_col(free_fit.chain, "$(PLANET_NAMES[k])_e") &&
                println("    ", PLANET_NAMES[k], "  eccentricity = ", describe_var(free_fit.chain, "$(PLANET_NAMES[k])_e"))
        end
        println("  (An eccentricity that is only an upper bound near 0 means a circular orbit is enough; an eccentricity ",
                "found with too few planets can instead be an unmodelled planet.)")
    end
    if fit_extra !== nothing
        lnbf_extra = fit_extra.lnZ - final.lnZ
        println()
        println("Extra-planet test: ", N_PLANETS + 1, " planets, ln Z = ", round(fit_extra.lnZ; digits = 2),
                ", ln BF vs ", N_PLANETS, " = ", round(lnbf_extra; digits = 2), "  ->  ",
                bf_interpretation(lnbf_extra; more = "$(N_PLANETS + 1) planets", less = "$(N_PLANETS) planet" * (N_PLANETS == 1 ? "" : "s")))
        planet_block(fit_extra.chain, N_PLANETS+1:N_PLANETS+1)
        println("  Jitter: ", round(median(chain_vec(final.chain, "HARPS_jitter")); digits = 2), " -> ",
                round(median(chain_vec(fit_extra.chain, "HARPS_jitter")); digits = 2), " m/s")
    end
    println()
    println("Median RV uncertainty is ", round(median(rv_data.σ_rv); digits = 2),
            " m/s; if the fitted jitter is far above that, signal is left over (more planets, or stellar activity / planet-planet interactions).")
end

section("Done")
dbg("Total run time: ", round((time() - T_START) / 60; digits = 1), " min")
dbg("Figures saved in: ", FIGDIR)
for f in sort(readdir(FIGDIR))
    endswith(f, ".png") && dbg("  ", f)
end
