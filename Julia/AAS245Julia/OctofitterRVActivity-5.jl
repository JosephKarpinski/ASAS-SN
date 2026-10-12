# ==============================================================================
# OctofitterRVActivity.jl
# One script for a HARPS RV planet fit AND the Ca II H&K activity analysis of the same
# star, with a combined analysis and a candidate summary at the end.
# Combines OctofitterRVBank.jl (v1.1.0) and OctofitterCaHK.jl (v1.0.1).
#
# Run from VSCode ("Execute active file in REPL") or from a shell:
#     julia OctofitterRVActivity.jl            # default: first star in the list (GJ 876)
#     julia OctofitterRVActivity.jl 3          # star number 3 in the list
#     julia OctofitterRVActivity.jl HD10700    # any list entry, or any target name in the table
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
#   Perdelwitz et al. (2024) added to it a time series of the relative chromospheric
#   emission in the Ca II H&K lines, R'HK, measured from the same ~250 000 HARPS
#   spectra (R'HK = flux in the H and K line cores relative to the bolometric flux,
#   with the photospheric part removed; ~1e-5, i.e. log R'HK ~ -5, for a quiet star and
#   larger for an active one). Each spectrum has one R'HK value, so for every RV
#   there is an activity measurement taken at the same instant.
#
#   One VizieR table holds both: J/A+A/683/A125/table4, "Individual radial velocity
#   measurements and auxiliary data for all targets" (corrected version 20-Apr-2026,
#   252615 rows). The script downloads it ONCE into ./cahk_cache (about 120 MB) and
#   reads that local copy on every later run, so other stars and selections can be tried
#   without the network. If VizieR cannot be reached it falls back to the RVBank CSV
#   on GitHub (via DataDeps; it may pre-date the April 2026 correction).
#
#   [1] https://exo-restart.com/wp-content/uploads/Stellar_parameters/HARPS_RVBank.html
#   [2] https://www.aanda.org/articles/aa/abs/2020/04/aa36686-19/aa36686-19.html
#   Catalogue:  https://www2.mpia-hd.mpg.de/homes/trifonov/HARPS_RVBank.html
#   Ca II H&K paper: Perdelwitz et al. 2024, A&A 683, A125  (arXiv:2312.06586)
#   License CC0-1.0. Please credit Trifonov et al. and Perdelwitz et al. when you use the data.
#
# WHAT THIS SCRIPT DOES
#   Data    Prints the list of 25 stars and marks the one selected; loads the table
#           (VizieR cache, downloaded on the first run), and picks that star's rows.
#   Part A  (RV planet fit, as OctofitterRVBank.jl; plots rv_plot01...)
#           Lomb-Scargle prewhitening gives candidate periods; Octofitter + Pigeons fit
#           1..N_PLANETS planets, compare the log Bayesian evidence, and run the optional
#           eccentricity and extra-planet tests. This part takes tens of minutes.
#   Part B  (Ca II H&K activity, as OctofitterCaHK.jl; plots hk_plot01...)
#           Cleans R'HK, averages per night, plots it, finds rotation and activity-cycle
#           periods, compares season means, correlates R'HK with the RV and the other
#           indicators, and compares with the activity-jitter limit. About a minute.
#   Part C  (new; plots combined_plot01...) Removes the fitted planets from the RVs and
#           tests the RESIDUALS against R'HK and the other indicators, overlays the
#           periodograms, draws the noise budget, then prints where the two analyses
#           agree and differ and vets every candidate signal (activity, annual alias,
#           harmonic or resonance of a stronger planet). The text is also saved as
#           summary_<star>.txt next to the plots in ./figures_rvact_<star>.
#   Set RUN_PLANET_FIT = false for a quick activity-only run, or RUN_ACTIVITY = false for
#   the planet fit alone.
#
# THE LIST
#   A curated list, not a ranking computed from the catalogue: 25 well-known planet
#   hosts with at least ~180 HARPS RVs, GJ 876 first. Counts and baselines were read from
#   the RVBank CSV. Stellar masses are rough (they only turn RV amplitudes into planet
#   masses): check them before quoting a mass. The note's first letter is read as the
#   spectral class. To add a star, append a line to STAR_LIST (the first field must match
#   the table's `Target` column up to spaces and case).
#
# CAVEATS
#   * Periods come from a sinusoid search; a strongly eccentric or interacting system may
#     need a different PERIOD_WINDOW or N_PLANETS. Planets are named b, c, d... in the
#     order they were detected (strongest first). Masses are minimum masses (m sin i).
#   * R'HK is derived with PHOENIX model fluxes and stellar parameters from the same
#     spectra; entries flagged by vsinflag / metflag may be biased. M dwarfs have no v sin i
#     in the table, so for them every row is flagged and the flags carry no information.
#   * The activity-jitter relations (Perdelwitz et al. 2024, Eqs. 1-3) were fitted to F, G
#     and K stars only, and are lower limits.
#   * The residuals in Part C come from a harmonic stand-in for the Keplerians (each planet
#     as a fundamental plus first harmonic at the fitted period), not from the Octofitter
#     model itself. In a resonant system the Keplerian model is itself approximate.
#   * R'HK sees chromospheric activity only; spots and convective blueshift can move the
#     RV without moving R'HK. The Part C verdicts are screening heuristics, not detections.
#
# Changelog
#   v1.0.0  2026-10-11  First combined version.
#   v1.0.1  2026-10-11  vet(): rotation-harmonic match alone no longer flags activity (needs R'HK power);
#                       precession-rate note for harmonic candidates; note on letter naming.
#   v1.1.2  2026-10-12  vet(): without a planet fit the harmonic (eccentricity) amplitude test cannot run, so a
#                       2:1, 3:1 or 4:1 candidate is no longer listed as passing every check; says so.
#   v1.1.1  2026-10-12  vsini limit only from a real measurement (error > 0 and vsini > 2 km/s; GJ 876 has a
#                       placeholder 2.00 +/- 0.00); catalogue R'HK compared only if its error is small;
#                       note when logg/[Fe/H] look like defaults.
#   v1.1.0  2026-10-12  VizieR table1 (stellar parameters of ~3600 targets, cached): spectral class from Teff,
#                       vsini limit on the rotation period, name lookup via SimbadName, parameter printout.
# ==============================================================================

import Pkg
using Dates
using Printf

# ------------------------------------------------------------------ settings
# --- the star
const TARGET_INDEX      = 1       # which star in STAR_LIST to use (1 = GJ 876)
const TARGET_OVERRIDE   = ""      # or any table target name, e.g. "HD10700" (beats TARGET_INDEX)
# --- what to run
const RUN_ACTIVITY      = true    # Part B and C: Ca II H&K analysis and the combined analysis (about a minute)
const RUN_PLANET_FIT    = true    # Part A: Octofitter planet fits (tens of minutes); false = quick run without the fits
# --- data
const DATA_SOURCE       = :auto   # :auto (VizieR cache, then GitHub CSV), :vizier, or :github
const CACHE_DIR         = joinpath(@__DIR__, "cahk_cache")   # the full VizieR table is saved here once
const CACHE_FILE        = joinpath(CACHE_DIR, "J_AA_683_A125_table4.csv")
const REFRESH_CACHE     = false   # true: download the VizieR table again
const N_ROWS_EXPECTED   = 252615  # rows in the corrected table (20-Apr-2026)
const CHUNK             = 50000   # rows per VizieR request
const USE_STAR_TABLE    = true    # also use VizieR table1: Teff, logg, [Fe/H], vsini, SimbadName for ~3600 stars (one small download)
const STAR_CACHE_FILE   = joinpath(CACHE_DIR, "J_AA_683_A125_table1.csv")
const N_STARS_EXPECTED  = 3612    # rows in table1
const VSINI_MIN_RELIABLE = 2.0    # [km/s] a vsini at or below this is an upper limit/placeholder, not a measurement; no rotation limit is derived
# --- Part A: planet fits
const N_PLANETS         = 2       # fit models with 1..N_PLANETS planets
const FIT_ALL_SUBMODELS = true    # true: fit 1..N_PLANETS and compare; false: only the N_PLANETS model
const FREE_ECCENTRICITY = true    # true: every planet gets e ~ U(0, 0.7); false: circular orbits
const RUN_ECCENTRICITY_TEST = true  # also fit the N_PLANETS model with the opposite eccentricity setting
const RUN_EXTRA_PLANET_TEST = true  # also fit N_PLANETS+1 planets, the extra one at the next periodogram peak
const PERIOD_WINDOW     = 0.10    # period prior = peak period +/- this fraction
const MIN_PERIOD        = 1.5     # [d] ignore shorter periodogram peaks (1 d peaks are daily-sampling aliases)
const RV_BIN_NIGHTLY    = :auto   # RVs: :auto (bin only if > MAX_POINTS), true, or false
const MAX_POINTS        = 1000
const CUSTOM_STAR_MASS  = 1.0     # [Msun] used only for a star that is not in STAR_LIST
const N_ROUNDS          = 10      # Pigeons rounds; each extra round doubles the work (9 is a quick look)
# --- Part B: activity
const ACT_BIN_NIGHTLY   = true    # average the spectra of each night (recommended)
const REQUIRE_UNFLAGGED = false   # true: drop rows with vsinflag > 0 or metflag > 0 (ignored if every row is flagged)
const LOGRHK_RANGE      = (-7.0, -3.0)  # plausible log10 R'HK; values outside are treated as garbage
const MAX_REL_ERR       = 1.0     # drop spectra whose R'HK error is larger than this times R'HK
const ROT_RANGE         = (2.0, 150.0)   # [d] "rotation" window for the folded plot / peak table
const CYCLE_MIN         = 500.0   # [d] periods longer than this are "activity cycle" candidates
const MIN_NIGHTS        = 15      # need at least this many nights for the periodograms
const SEASON_MIN_NIGHTS = 3       # nights needed for a season (calendar year) to get a mean
# --- environment
const USE_TEMP_ENV      = true    # true: throw-away environment (reinstalls every run); false: keep one in ./.octofitter_v9_env
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

section("HARPS target list")
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
const SPTYPE_NOTE = uppercase(first(strip(SEL[7]), 1))  # M, K, G, F or ? from the hand-typed note (used if table1 has no Teff)
const FIGDIR   = joinpath(@__DIR__, "figures_rvact_" * replace(TARGET, r"[^A-Za-z0-9._-]" => "_"))
mkpath(FIGDIR)
println("\nSelected: ", STARNAME, "   (table target \"", TARGET, "\", spectral class from the note: ", SPTYPE_NOTE, ")")
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
    Pkg.PackageSpec(name = "CSV"),
    Pkg.PackageSpec(name = "DataFrames"),
    Pkg.PackageSpec(name = "DataDeps"),
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

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"   # let DataDeps download the catalogue without prompting

using Octofitter
using OctofitterRadialVelocity
using PlanetOrbits
using CairoMakie
import CairoMakie.Makie
using PairPlots
using Distributions
using Pigeons
using LombScargle
using Statistics
using CSV
using DataFrames
using DataDeps
using Downloads

CairoMakie.activate!(type = "png")

for (_, info) in Pkg.dependencies()
    info.name in ("Octofitter", "OctofitterRadialVelocity", "PlanetOrbits", "Pigeons", "CairoMakie",
                  "PairPlots", "LombScargle", "CSV", "DataFrames", "DataDeps") &&
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


# --- small numeric helpers (NaN marks a missing value everywhere below)
hk_tofloat(x) = x === missing ? NaN :
             x isa Number ? Float64(x) :
             (v = tryparse(Float64, strip(string(x))); v === nothing ? NaN : v)
hk_nanify(v) = (isfinite(v) && v > -9.0e6) ? v : NaN        # -9999999 is the "no value" code of the GitHub CSV
hk_nanmean(v) = (g = filter(isfinite, v); isempty(g) ? NaN : mean(g))
hk_nanmedian(v) = (g = filter(isfinite, v); isempty(g) ? NaN : median(g))
hk_robust_sigma(v) = (g = filter(isfinite, v); isempty(g) ? NaN : 1.4826 * median(abs.(g .- median(g))))
hk_decyear(mjd) = 2000.0 .+ (mjd .- 51544.5) ./ 365.25
function hk_ranks(x)
    p = sortperm(x); r = zeros(length(x)); r[p] = 1:length(x); return r
end
function hk_paired(x, y)
    ok = findall(i -> isfinite(x[i]) && isfinite(y[i]), eachindex(x))
    return x[ok], y[ok]
end
hk_pearson(x, y)  = ((xx, yy) = hk_paired(x, y); length(xx) < 5 ? NaN : cor(xx, yy))
hk_spearman(x, y) = ((xx, yy) = hk_paired(x, y); length(xx) < 5 ? NaN : cor(hk_ranks(xx), hk_ranks(yy)))
hk_npair(x, y) = length(first(hk_paired(x, y)))
"Approximate significance in sigma of a correlation coefficient r from n pairs."
hk_corr_sigma(r, n) = (isfinite(r) && n > 3) ? abs(r) * sqrt(n - 1) : NaN
function hk_linfit(x, y)
    xx, yy = hk_paired(x, y); n = length(xx)
    xm = mean(xx); ym = mean(yy); sxx = sum((xx .- xm) .^ 2)
    b = sum((xx .- xm) .* (yy .- ym)) / sxx; a = ym - b * xm
    s = sqrt(sum((yy .- (a .+ b .* xx)) .^ 2) / max(n - 2, 1))
    return a, b, s / sqrt(sxx)
end
hk_corr_words(r, n) = (s = hk_corr_sigma(r, n); !isfinite(s) ? "not enough data" :
                    s > 5 ? "strong" : s > 3 ? "significant" : s > 2 ? "marginal" : "no significant correlation")

# ------------------------------------------------------------------ 1. data
section("Data: loading the HARPS table (RVs and Ca II H&K) for $STARNAME")

# Standard column name => name in the GitHub CSV / name in the VizieR table.
const STD = (:bjd, :rv, :erv, :crx, :dlw, :halpha, :nad1, :bis, :fwhm, :snr, :rhk, :drhk, :vsin, :met)
const GH_COLS = ("BJD", "RV_mlc_nzp", "e_RV_mlc_nzp", "CRX", "dLW", "Halpha", "NaD1", "BIS", "FWHM_DRS",
                 "SNR_DRS", "RHKp", "dRHKp", "vsinflag", "metflag")
const VZ_COLS = ("BJD", "DRVmlcnzp", "e_DRVmlcnzp", "CRX", "dLW", "Halpha", "NaD1", "BIS", "FWHMDRS",
                 "SNRDRS", "RHKp", "e_RHKp", "vsinflag", "metflag")

"Find a column by name, ignoring case; `nothing` if absent."
function hk_findcol(df, name)
    for n in names(df)
        lowercase(n) == lowercase(name) && return n
    end
    return nothing
end

"Turn a DataFrame with the source's column names into a NamedTuple of Float64 vectors with standard names."
function hk_standardize(df, srcnames)
    cols = Vector{Float64}[]
    for (k, nm) in zip(STD, srcnames)
        c = hk_findcol(df, nm)
        if c === nothing
            dbg("  column ", nm, " not found -> filled with missing values")
            push!(cols, fill(NaN, nrow(df)))
        else
            push!(cols, hk_nanify.(hk_tofloat.(df[!, c])))
        end
    end
    t = NamedTuple{STD}(Tuple(cols))
    # a catalogue that stores MJD instead of BJD
    return all(x -> !isfinite(x) || x < 1.0e6, t.bjd) ? merge(t, (; bjd = t.bjd .+ 2400000.5)) : t
end

function hk_urlencode(s::AbstractString)
    io = IOBuffer()
    for b in codeunits(s)
        c = Char(b)
        if (b < 128 && (isletter(c) || isdigit(c))) || c in "-_.~"
            write(io, b)
        else
            write(io, '%', uppercase(string(b, base = 16, pad = 2)))
        end
    end
    return String(take!(io))
end

"""
Download the WHOLE VizieR table (all columns, 252615 rows, ~150 MB as CSV) once, in
chunks of CHUNK rows by record number, and keep it in CACHE_FILE. Later runs read the
local file, so other stars, columns and selections can be tried without the network.
Delete the file or set REFRESH_CACHE = true to download it again.
"""
function hk_download_vizier_table()
    mkpath(CACHE_DIR)
    tmp = CACHE_FILE * ".part"
    open(tmp, "w") do io
        first_chunk = true
        for lo in 1:CHUNK:N_ROWS_EXPECTED
            hi = lo + CHUNK - 1
            q = "SELECT * FROM \"J/A+A/683/A125/table4\" WHERE recno BETWEEN $lo AND $hi"
            url = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&MAXREC=$(CHUNK + 10)&QUERY=" * hk_urlencode(q)
            dbg("VizieR TAP: records ", lo, " to ", hi, " ...")
            f = Downloads.download(url, tempname(); timeout = 600)
            lines = readlines(f); rm(f; force = true)
            (isempty(lines) || !occursin(r"recno"i, lines[1])) &&
                error("VizieR returned something other than a table: " * first(join(lines, " "), 200))
            for (k, l) in enumerate(lines)
                (k == 1 && !first_chunk) && continue
                println(io, l)
            end
            first_chunk = false
            dbg("  ", length(lines) - 1, " rows")
        end
    end
    mv(tmp, CACHE_FILE; force = true)
    return CACHE_FILE
end

"""
Read the cached full VizieR table (downloading it first if needed) into the global
VIZIER_TABLE (a DataFrame with every VizieR column, handy for other selections in the
REPL), and return one star as a NamedTuple (see `hk_standardize`), or `nothing` on failure.
"""
function hk_load_vizier(names)
    try
        if REFRESH_CACHE || !isfile(CACHE_FILE)
            hk_download_vizier_table()
        end
        dbg("Reading ", CACHE_FILE, " (", round(filesize(CACHE_FILE) / 1e6; digits = 0), " MB)")
        df = CSV.read(CACHE_FILE, DataFrame; missingstring = ["", "NA", "NaN", "null"],
                      types = Dict("Target" => String))
        global VIZIER_TABLE = df
        nrow(df) == N_ROWS_EXPECTED || @warn "VizieR cache has $(nrow(df)) rows, expected $N_ROWS_EXPECTED -- set REFRESH_CACHE = true to download again"
        tcol = hk_findcol(df, "Target")
        tcol === nothing && error("no Target column; columns: " * join(names(df), ","))
        key = normname.(df[!, tcol])
        for nm in unique(names)
            sub = df[key .== normname(nm), :]
            dbg("  Target = '", nm, "': ", nrow(sub), " rows")
            nrow(sub) > 0 && return hk_standardize(sub, VZ_COLS)
        end
    catch err
        dbg("VizieR path failed: ", sprint(showerror, err))
        isfile(CACHE_FILE * ".part") && rm(CACHE_FILE * ".part"; force = true)
    end
    return nothing
end

"Read the RVBank CSV (downloaded once by DataDeps, ~38 MB zip) and pick one star."
function hk_load_github(names)
    try register(DataDep("HARPS_RVBank",
        """
        Dataset:     A public HARPS radial velocity database corrected for systematic errors
        Author:      Trifonov et al.;  R'HK columns added by Perdelwitz et al. (2024)
        License:     CC0-1.0
        File size:   38MiB (zip), 144 MB unpacked
        """,
        "https://github.com/3fon3fonov/HARPS_RVBank/raw/master/HARPS_RVBank_ver02.csv.zip";
        post_fetch_method = unpack)) catch; end   # already registered by OctofitterRadialVelocity
    dir = resolve("HARPS_RVBank", @__FILE__)
    file = joinpath(dir, "HARPS_RVBank_ver02.csv")
    isfile(file) || error("HARPS_RVBank_ver02.csv not found in $dir")
    dbg("Reading ", file, " (this takes a few seconds)")
    df = CSV.read(file, DataFrame; select = vcat(["target"], collect(GH_COLS)),
                  missingstring = ["", "NA", "NaN"], types = Dict("target" => String))
    for nm in unique(names)
        sub = df[df.target .== nm, :]
        nrow(sub) > 0 && return hk_standardize(sub, GH_COLS)
    end
    nms = unique(df.target)
    close_names = sort(filter(n -> occursin(normname(TARGET)[1:min(end, 3)], normname(n)), nms))[1:min(end, 8)]
    error("Target \"$TARGET\" not found in the table. Names that start alike: $(close_names)")
end


# ------------------------------------------------------------------ B0. stellar parameters (VizieR table1)
"""
Download VizieR J/A+A/683/A125/table1 (stellar parameters, one row per target, ~3600 rows,
well under 1 MB) once into STAR_CACHE_FILE. Later runs read the local file.
"""
function hk_download_star_table()
    mkpath(CACHE_DIR)
    q = "SELECT * FROM \"J/A+A/683/A125/table1\""
    url = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&MAXREC=$(N_STARS_EXPECTED + 500)&QUERY=" * hk_urlencode(q)
    dbg("VizieR TAP: stellar-parameter table (table1) ...")
    f = Downloads.download(url, tempname(); timeout = 300)
    lines = readlines(f)
    if isempty(lines) || !occursin(r"Target"i, lines[1])
        rm(f; force = true)
        error("VizieR returned something other than a table: " * first(join(lines, " "), 200))
    end
    mv(f, STAR_CACHE_FILE; force = true)
    dbg("  ", length(lines) - 1, " stars")
    return STAR_CACHE_FILE
end

"Read table1 (downloading it once); a DataFrame, or nothing if switched off or unavailable."
function hk_load_star_table()
    USE_STAR_TABLE || return nothing
    try
        (REFRESH_CACHE || !isfile(STAR_CACHE_FILE)) && hk_download_star_table()
        df = CSV.read(STAR_CACHE_FILE, DataFrame; missingstring = ["", "NA", "NaN", "null"],
                      types = Dict("Target" => String, "SimbadName" => String))
        nrow(df) == N_STARS_EXPECTED || dbg("  note: table1 has ", nrow(df), " rows, expected ", N_STARS_EXPECTED)
        return df
    catch err
        dbg("Stellar-parameter table unavailable: ", sprint(showerror, err))
        isfile(STAR_CACHE_FILE) && filesize(STAR_CACHE_FILE) == 0 && rm(STAR_CACHE_FILE; force = true)
        return nothing
    end
end

"Row index of a star in table1: match the given names against Target, then against SimbadName."
function hk_star_row(df, names)
    df === nothing && return nothing
    simple(x) = normname(replace(string(x), r"^(\*+|V\*|NAME)\s+" => ""))
    for colname in ("Target", "SimbadName")
        c = hk_findcol(df, colname)
        c === nothing && continue
        keys = [ismissing(x) ? "" : simple(x) for x in df[!, c]]
        for nm in unique(names)
            i = findfirst(==(simple(nm)), keys)
            i === nothing || return (i, colname)
        end
    end
    return nothing
end

"One star's parameters from table1 as a NamedTuple (NaN where missing)."
function hk_star_params(df, i)
    function num(names...)
        for nm in names
            c = hk_findcol(df, nm)
            c === nothing && continue
            x = df[i, c]
            x isa Real && return Float64(x)
            x isa AbstractString && return something(tryparse(Float64, x), NaN)
        end
        return NaN
    end
    txt(name) = (c = hk_findcol(df, name); c === nothing || ismissing(df[i, c]) ? "" : string(df[i, c]))
    return (; target = txt("Target"), simbad = txt("SimbadName"),
            ra = num("RAJ2000"), dec = num("DEJ2000"),
            rhk = num("RHKp"), erhk = num("e_RHKp"), snr = num("S/N", "S_N"),
            sourflag = num("Sourflag"), rvflag = num("RVflag"), vflag = num("vsinflag"), mflag = num("metaflag", "metflag"),
            teff = num("Teff"), eteff = num("e_Teff"), logg = num("logg"), elogg = num("e_logg"),
            feh = num("[Fe/H]", "__Fe_H_", "FeH"), efeh = num("e_[Fe/H]", "e_Fe_H_", "e_FeH"),
            vsini = num("vsini"), evsini = num("e_vsini"))
end

"F, G, K or M from Teff [K] (A for hotter stars); `fallback` if Teff is missing."
function hk_class_from_teff(teff, fallback)
    isfinite(teff) || return fallback
    return teff >= 7500 ? "A" : teff >= 6000 ? "F" : teff >= 5300 ? "G" : teff >= 3900 ? "K" : "M"
end

"Formatted number, or n/a."
hk_f(x, fmt = "%.2f") = isfinite(x) ? Printf.format(Printf.Format(fmt), x) : "n/a"

"Stellar radius [Rsun] from logg (cgs) and mass [Msun]."
hk_radius(logg, mass) = sqrt(6.674e-8 * mass * 1.989e33 / 10.0^logg) / 6.957e10

"""
Upper limit on the rotation period [d] from vsini: P_rot = 2 pi R / v_eq and v_eq >= vsini, so
P_rot <= 2 pi R / vsini. Uses vsini - 2 sigma to be conservative. NaN if there is no usable vsini
(flag 1 = none; below VSINI_MIN_RELIABLE km/s the number is at the measurement limit).
"""
function hk_prot_max(p, mass)
    (p !== nothing && isfinite(p.vsini) && isfinite(p.evsini) && isfinite(p.logg) && p.vflag in (0.0, 2.0)) || return NaN
    p.evsini > 0 || return NaN                  # 'vsini +/- 0' is a default value, not a measurement
    p.vsini > VSINI_MIN_RELIABLE || return NaN  # at or below the resolution limit it is only an upper limit
    vlo = p.vsini - 2 * (isfinite(p.evsini) ? p.evsini : 0.0)
    vlo > 0 || return NaN
    return 2π * hk_radius(p.logg, mass) * 6.957e5 / vlo / 86400    # R in km, v in km/s
end

section("B0. Stellar parameters (VizieR J/A+A/683/A125/table1)")
const STAR_TABLE = hk_load_star_table()
const _SROW = hk_star_row(STAR_TABLE, [TARGET, STARNAME, replace(TARGET, r"^([A-Za-z]+)(\d)" => s"\1 \2")])
const STAR_P = _SROW === nothing ? nothing : hk_star_params(STAR_TABLE, _SROW[1])
const SPTYPE = hk_class_from_teff(STAR_P === nothing ? NaN : STAR_P.teff, SPTYPE_NOTE)
const PROT_MAX = hk_prot_max(STAR_P, M_STAR)
if STAR_P === nothing
    dbg(STAR_TABLE === nothing ? "table1 not available" : "\"" * TARGET * "\" not found in table1 (neither as Target nor as SimbadName)",
        "; spectral class from the note: ", SPTYPE)
else
    println("  matched via ", _SROW[2], ":  Target = ", STAR_P.target, isempty(STAR_P.simbad) ? "" : "   SimbadName = " * STAR_P.simbad)
    println("  Teff = ", hk_f(STAR_P.teff, "%.0f"), " +/- ", hk_f(STAR_P.eteff, "%.0f"), " K   logg = ", hk_f(STAR_P.logg), " +/- ", hk_f(STAR_P.elogg),
            "   [Fe/H] = ", hk_f(STAR_P.feh, "%+.2f"), " +/- ", hk_f(STAR_P.efeh))
    println("  vsini = ", hk_f(STAR_P.vsini, "%.2f"), " +/- ", hk_f(STAR_P.evsini), " km/s   (vsinflag ", hk_f(STAR_P.vflag, "%.0f"),
            ": 0 measured, 1 none/unreliable, 2 from Gaia/TIC;  metflag ", hk_f(STAR_P.mflag, "%.0f"), ")")
    println("  catalogue R'HK = ", hk_f(STAR_P.rhk, "%.3g"), " +/- ", hk_f(STAR_P.erhk, "%.2g"), "   S/N = ", hk_f(STAR_P.snr, "%.0f"),
            "   Sourflag ", hk_f(STAR_P.sourflag, "%.0f"), "   RVflag ", hk_f(STAR_P.rvflag, "%.0f"))
    println("  spectral class from Teff: ", SPTYPE, SPTYPE != SPTYPE_NOTE && SPTYPE_NOTE != "?" ? "   (the hand-typed note says " * SPTYPE_NOTE * "; Teff is used)" : "")
    if STAR_P.mflag == 1 && isfinite(STAR_P.efeh) && STAR_P.efeh >= 0.4
        println("  note: metflag 1 with an [Fe/H] error of ", hk_f(STAR_P.efeh), " means [Fe/H] (and probably logg) are default values, not measurements")
    end
    if isfinite(STAR_P.logg)
        println("  radius from logg and the assumed mass (", M_STAR, " Msun): ", hk_f(hk_radius(STAR_P.logg, M_STAR)), " Rsun")
    end
    if isfinite(PROT_MAX)
        println("  rotation limit from vsini: P_rot <= ", hk_f(PROT_MAX, "%.1f"), " d  (uses vsini - 2 sigma and sin i <= 1)")
    elseif isfinite(STAR_P.vsini) && (STAR_P.vsini <= VSINI_MIN_RELIABLE || !(STAR_P.evsini > 0))
        println("  vsini = ", hk_f(STAR_P.vsini), " +/- ", hk_f(STAR_P.evsini), " km/s looks like a placeholder or an upper limit (error 0, or value <= ",
                VSINI_MIN_RELIABLE, " km/s): no rotation limit derived")
    else
        println("  no usable vsini (flag ", hk_f(STAR_P.vflag, "%.0f"), "): no rotation limit derived")
    end
end
SPTYPE != "?" || dbg("Spectral class unknown: no activity-jitter floor will be computed")

const NAME_TRIES = unique(filter(!isempty, [TARGET, STARNAME, replace(TARGET, r"^([A-Za-z]+)(\d)" => s"\1 \2"),
                                             STAR_P === nothing ? "" : STAR_P.target]))

raw = nothing
SOURCE_USED = ""
if DATA_SOURCE in (:auto, :vizier)
    global raw = hk_load_vizier(NAME_TRIES)
    raw === nothing || (global SOURCE_USED = "VizieR J/A+A/683/A125/table4")
    raw === nothing && DATA_SOURCE === :vizier && error("VizieR gave no data and DATA_SOURCE = :vizier")
end
if raw === nothing
    DATA_SOURCE === :auto && dbg("Falling back to the GitHub RVBank CSV (may pre-date the April 2026 VizieR correction)")
    global raw = hk_load_github(NAME_TRIES)
    global SOURCE_USED = "GitHub HARPS_RVBank_ver02.csv"
end
dbg("Data source: ", SOURCE_USED, "  |  rows for this star: ", length(raw.bjd))


# ------------------------------------------------------------------ Part A, data: the RVs of this star
section("A1. RV data for $STARNAME")
# Same table as the activity analysis uses: epoch [MJD], RV [m/s], uncertainty [m/s].
rv_raw = Table(epoch = raw.bjd .- 2400000.5, rv = raw.rv, σ_rv = raw.erv)
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

do_bin = RV_BIN_NIGHTLY === :auto ? length(rv_clean.epoch) > MAX_POINTS : RV_BIN_NIGHTLY === true
rv_data = do_bin ? bin_nightly(rv_clean) : rv_clean

const NPTS     = length(rv_data.epoch)
const BASELINE = maximum(rv_data.epoch) - minimum(rv_data.epoch)
const T_REF    = round(mean(rv_data.epoch))     # reference epoch for the orbital phases

dbg("RVs in the catalogue for this star: ", length(rv_raw.epoch), "  (usable: ", length(rv_clean.epoch), ")")
do_bin && dbg("Binned per night: ", length(rv_clean.epoch), " -> ", NPTS, " points")
NPTS > MAX_POINTS && @warn "$NPTS points is a lot for the sampler; consider RV_BIN_NIGHTLY = true or a lower N_ROUNDS"
dbg("Epoch range [MJD]: ", round(minimum(rv_data.epoch); digits = 1), " to ",
    round(maximum(rv_data.epoch); digits = 1), "  (baseline ", round(BASELINE; digits = 1), " d = ",
    round(BASELINE / 365.25; digits = 1), " yr)")
dbg("RV range [m/s]: ", round(minimum(rv_data.rv); digits = 1), " to ", round(maximum(rv_data.rv); digits = 1))
dbg("Median RV uncertainty [m/s]: ", round(median(rv_data.σ_rv); digits = 2))
dbg("RV scatter (robust, from the median absolute deviation) [m/s]: ",
    round(1.4826 * median(abs.(rv_data.rv .- median(rv_data.rv))); digits = 2))

safe("RV Plot 1") do
    fig = Figure(size = (900, 450))
    ax = Axis(fig[1, 1], xlabel = "epoch [MJD]", ylabel = "RV [m/s]")
    Makie.scatter!(ax, rv_data.epoch, rv_data.rv, color = :black, markersize = 5)
    Makie.errorbars!(ax, rv_data.epoch, rv_data.rv, rv_data.σ_rv, color = (:black, 0.4))
    save_plot!(fig, "RV Plot 1. HARPS RVBank radial velocities of $STARNAME ($NPTS points" *
                    (do_bin ? ", binned per night" : "") * "; error bars are the pipeline uncertainties).",
               "rv_plot01_rv_data.png")
end

# ------------------------------------------------------------------ 2. candidate periods
section("A2. Candidate periods by periodogram prewhitening")
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

safe("RV Plot 2") do
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
    save_plot!(fig, "RV Plot 2. Lomb-Scargle prewhitening of $STARNAME. Each panel shows the periodogram " *
                    "of what is left after removing the sinusoids found above it; the dashed line is " *
                    "the strongest peak, used as the centre of that planet's period prior.",
               "rv_plot02_periodogram_prewhitening.png")
end


# ------------------------------------------------------------------ results shared with Part C
fits = NamedTuple[]
final = nothing
fit_alt = nothing
fit_extra = nothing

# ==============================================================================
# PART B. Ca II H&K activity
# ==============================================================================
if RUN_ACTIVITY

# --- cleaning, with a count of what each step removes
let
    n0 = length(raw.bjd)
    ok = trues(n0)
    function step(label, mask)
        before = count(ok); ok .&= mask
        dbg(rpad(label, 52), "removed ", before - count(ok), "   left ", count(ok))
    end
    r = raw.rhk
    step("no epoch", isfinite.(raw.bjd))
    step("no R'HK value (missing / -9999999 code)", isfinite.(r) .& (r .> 0))
    lr = [x > 0 ? log10(x) : NaN for x in r]
    step("log R'HK outside $(LOGRHK_RANGE)", (lr .>= LOGRHK_RANGE[1]) .& (lr .<= LOGRHK_RANGE[2]))
    step("R'HK error missing or > $(MAX_REL_ERR) x R'HK",
         isfinite.(raw.drhk) .& (raw.drhk .<= MAX_REL_ERR .* r))
    if REQUIRE_UNFLAGGED
        flagged = (raw.vsin .> 0) .| (raw.met .> 0)
        if all(flagged[ok])
            dbg("Every usable row is flagged (M dwarfs have no v sin i), so REQUIRE_UNFLAGGED is ignored")
        else
            step("vsinflag > 0 or metflag > 0", .!flagged)
        end
    end
    global CLEAN = NamedTuple{STD}(Tuple(getfield(raw, k)[ok] for k in STD))
    global N_FLAGGED = count(i -> (CLEAN.vsin[i] > 0) || (CLEAN.met[i] > 0), eachindex(CLEAN.bjd))
end
length(CLEAN.bjd) >= MIN_NIGHTS || error("Only $(length(CLEAN.bjd)) usable R'HK values for $STARNAME; need at least $MIN_NIGHTS.")
dbg("Usable spectra: ", length(CLEAN.bjd), "   of which flagged (vsinflag>0 or metflag>0): ", N_FLAGGED)
dbg("Median R'HK error / R'HK per spectrum: ", round(median(CLEAN.drhk ./ CLEAN.rhk); digits = 3))

"Average everything per night (night = BJD day, which starts at noon UT, so a Chilean night is never split)."
function hk_bin_nights(d)
    order = sortperm(d.bjd)
    night = floor.(Int, d.bjd[order])
    starts = [1; findall(i -> night[i] != night[i-1], 2:length(night)) .+ 1]
    stops = [starts[2:end] .- 1; length(night)]
    mjd = Float64[]; lr = Float64[]; elr = Float64[]; rv = Float64[]; erv = Float64[]
    crx = Float64[]; dlw = Float64[]; ha = Float64[]; na = Float64[]; bis = Float64[]; fw = Float64[]
    nsp = Int[]; flg = Bool[]
    for (a, b) in zip(starts, stops)
        i = order[a:b]
        n = length(i)
        rbar = mean(d.rhk[i])
        push!(mjd, mean(d.bjd[i]) - 2400000.5)
        push!(lr, log10(rbar))
        push!(elr, sqrt(sum(d.drhk[i] .^ 2)) / n / (rbar * log(10)))      # propagated, in dex
        g = filter(j -> isfinite(d.rv[j]) && isfinite(d.erv[j]) && d.erv[j] > 0, i)
        if isempty(g)
            push!(rv, NaN); push!(erv, NaN)
        else
            w = 1 ./ d.erv[g] .^ 2
            push!(rv, sum(w .* d.rv[g]) / sum(w)); push!(erv, 1 / sqrt(sum(w)))
        end
        push!(crx, hk_nanmean(d.crx[i])); push!(dlw, hk_nanmean(d.dlw[i])); push!(ha, hk_nanmean(d.halpha[i]))
        push!(na, hk_nanmean(d.nad1[i]));  push!(bis, hk_nanmean(d.bis[i]));  push!(fw, hk_nanmean(d.fwhm[i]))
        push!(nsp, n); push!(flg, any(j -> (d.vsin[j] > 0) || (d.met[j] > 0), i))
    end
    return (; mjd, lr, elr, rv, erv, crx, dlw, ha, na, bis, fw, nsp, flg)
end

function hk_unbinned(d)
    return (; mjd = d.bjd .- 2400000.5, lr = log10.(d.rhk), elr = d.drhk ./ (d.rhk .* log(10)),
            rv = d.rv, erv = d.erv, crx = d.crx, dlw = d.dlw, ha = d.halpha, na = d.nad1, bis = d.bis,
            fw = d.fwhm, nsp = ones(Int, length(d.bjd)),
            flg = [(d.vsin[i] > 0) || (d.met[i] > 0) for i in eachindex(d.bjd)])
end

N = ACT_BIN_NIGHTLY ? hk_bin_nights(CLEAN) : hk_unbinned(CLEAN)
const NN       = length(N.mjd)
const HK_BASELINE = maximum(N.mjd) - minimum(N.mjd)
const YEARS    = hk_decyear(N.mjd)
const HAS_RV   = count(isfinite, N.rv) >= MIN_NIGHTS
NN >= MIN_NIGHTS || error("Only $NN nights with R'HK; need at least $MIN_NIGHTS.")
dbg("Nights: ", NN, "  |  baseline ", round(HK_BASELINE; digits = 1), " d = ", round(HK_BASELINE / 365.25; digits = 1), " yr",
    "  |  years ", round(minimum(YEARS); digits = 1), " to ", round(maximum(YEARS); digits = 1))
dbg("Nights with RV: ", count(isfinite, N.rv))
dbg("log R'HK per night: median ", round(median(N.lr); digits = 3), "  robust scatter ", round(hk_robust_sigma(N.lr); digits = 3),
    "  16-84 % range ", round(quantile(N.lr, 0.16); digits = 3), " to ", round(quantile(N.lr, 0.84); digits = 3),
    "  median error ", round(median(N.elr); digits = 3), " dex")
const LOGR_MED = median(N.lr)
const RV_SIGMA = HAS_RV ? hk_robust_sigma(N.rv) : NaN
HAS_RV && dbg("RV scatter per night (robust) [m/s]: ", round(RV_SIGMA; digits = 2),
              "   median RV error ", round(hk_nanmedian(N.erv); digits = 2))

# Season (calendar year) means, used in Plot 5 and the summary.
function hk_season_means(N)
    yr = floor.(Int, YEARS)
    out = (; year = Float64[], lr = Float64[], elr = Float64[], rv = Float64[], erv = Float64[], n = Int[])
    for y in sort(unique(yr))
        i = findall(==(y), yr)
        length(i) >= SEASON_MIN_NIGHTS || continue
        push!(out.year, mean(YEARS[i])); push!(out.lr, mean(N.lr[i])); push!(out.elr, std(N.lr[i]) / sqrt(length(i)))
        g = filter(j -> isfinite(N.rv[j]), i)
        push!(out.rv, isempty(g) ? NaN : mean(N.rv[g]))
        push!(out.erv, length(g) < 2 ? NaN : std(N.rv[g]) / sqrt(length(g)))
        push!(out.n, length(i))
    end
    return out
end
SEAS = hk_season_means(N)
dbg("Seasons with >= ", SEASON_MIN_NIGHTS, " nights: ", length(SEAS.year))

# Plot 1 ----------------------------------------------------------------------
safe("HK Plot 1") do
    fig = Figure(size = (950, HAS_RV ? 640 : 380))
    ax1 = Axis(fig[1, 1], ylabel = "log R'HK", xlabel = HAS_RV ? "" : "year")
    some_flagged = any(N.flg) && !all(N.flg)     # if every row is flagged (M dwarfs), colouring them says nothing
    ok = some_flagged ? .!N.flg : trues(length(N.flg))
    Makie.scatter!(ax1, YEARS[ok], N.lr[ok], color = :black, markersize = 5, label = "nightly mean")
    some_flagged && Makie.scatter!(ax1, YEARS[N.flg], N.lr[N.flg], color = :darkorange, markersize = 6, marker = :utriangle,
                                 label = "flagged (vsinflag/metflag > 0)")
    Makie.errorbars!(ax1, YEARS, N.lr, N.elr, color = (:black, 0.25))
    length(SEAS.year) > 1 && Makie.scatterlines!(ax1, SEAS.year, SEAS.lr, color = :crimson, linewidth = 2, markersize = 9,
                                                 label = "calendar-year mean")
    Makie.hlines!(ax1, [LOGR_MED], color = (:gray, 0.7), linestyle = :dash)
    axislegend(ax1, position = :rt, framevisible = false)
    if HAS_RV
        ax2 = Axis(fig[2, 1], xlabel = "year", ylabel = "RV [m/s]")
        linkxaxes!(ax1, ax2)
        g = isfinite.(N.rv)
        Makie.scatter!(ax2, YEARS[g], N.rv[g], color = :black, markersize = 5)
        Makie.errorbars!(ax2, YEARS[g], N.rv[g], N.erv[g], color = (:black, 0.3))
        length(SEAS.year) > 1 && Makie.scatterlines!(ax2, SEAS.year[isfinite.(SEAS.rv)], SEAS.rv[isfinite.(SEAS.rv)],
                                                     color = :crimson, linewidth = 2, markersize = 9)
        hidexdecorations!(ax1, grid = false)
        rowsize!(fig.layout, 2, Relative(0.45))
    end
    save_plot!(fig, "HK Plot 1. $STARNAME: Ca II H&K activity (top, log R'HK per night) and radial velocity (bottom, " *
                    "HARPS RVBank) on the same time axis. Red: calendar-year means. $NN nights" *
                    (ACT_BIN_NIGHTLY ? ", spectra averaged per night" : "") * ".",
               "hk_plot01_timeseries.png")
end

# Plot 2 ----------------------------------------------------------------------
safe("HK Plot 2") do
    fig = Figure(size = (800, 420))
    ax = Axis(fig[1, 1], xlabel = "log R'HK", ylabel = "nights")
    Makie.hist!(ax, N.lr, bins = clamp(round(Int, sqrt(NN)), 8, 60), color = (:steelblue, 0.8), strokewidth = 1, strokecolor = :white)
    q = quantile(N.lr, [0.16, 0.5, 0.84])
    Makie.vlines!(ax, [q[2]], color = :crimson, linewidth = 2, label = "median $(round(q[2]; digits = 2))")
    Makie.vlines!(ax, [q[1], q[3]], color = :crimson, linestyle = :dash, label = "16 / 84 %")
    axislegend(ax, position = :rt, framevisible = false)
    save_plot!(fig, "HK Plot 2. $STARNAME: distribution of the nightly log R'HK. A wide or skewed distribution means " *
                    "real activity variation (cycle or flares); a narrow one close to the measurement error ($(round(median(N.elr); digits = 2)) dex) " *
                    "means a quiet, stable star.",
               "hk_plot02_histogram.png")
end

# ------------------------------------------------------------------ 2. periodograms
section("B2. Periodograms of log R'HK (and of the RV)")
const HK_MAX_PERIOD = HK_BASELINE / 2

"Lomb-Scargle (unweighted, so RV and R'HK are treated alike). `fast = false` is required with AppleAccelerate loaded."
function hk_ls_run(t, y)
    pg = lombscargle(t, y; minimum_frequency = 1 / HK_MAX_PERIOD, maximum_frequency = 1 / MIN_PERIOD,
                     samples_per_peak = 10, fast = false)
    per, pow = LombScargle.periodpower(pg)
    return pg, per, pow
end

"The `n` strongest local maxima, skipping peaks closer than 10 % in period to a stronger one."
function hk_top_peaks(per, pow; n = 5)
    idx = [i for i in 2:length(pow)-1 if pow[i] > pow[i-1] && pow[i] >= pow[i+1]]
    idx = idx[sortperm(pow[idx]; rev = true)]
    keep = Int[]
    for i in idx
        all(j -> abs(per[i] - per[j]) / per[j] > 0.10, keep) && push!(keep, i)
        length(keep) >= n && break
    end
    return keep
end

# Returned from a function and assigned outside the `try`: a `try` block is its own scope.
function hk_run_ls(t, y)
    try
        return hk_ls_run(t, y)
    catch err
        @warn "Periodogram FAILED" exception = (err, catch_backtrace())
        return nothing
    end
end
LS_R = hk_run_ls(N.mjd, N.lr)
LS_R === nothing && error("The R'HK periodogram failed; see the warning above.")
pgR, perR, powR = LS_R
pkR = hk_top_peaks(perR, powR)
fapR(i) = LombScargle.fap(pgR, powR[i])
hk_fmt_fap(f) = f < 1e-10 ? "<1e-10" : @sprintf("%.1e", f)

println("\nlog R'HK periodogram, strongest peaks (FAP = Baluev false-alarm probability; the sampling is seasonal, so also look for the 1-yr alias):")
println(rpad("  period [d]", 16), rpad("power", 9), rpad("FAP", 11), "kind")
for i in pkR
    P = perR[i]
    kind = P >= CYCLE_MIN ? "cycle?" : (ROT_RANGE[1] <= P <= ROT_RANGE[2] ? "rotation range" : "")
    abs(P - 365.25) / 365.25 < 0.05 && (kind = "1-yr alias?")
    println("  ", rpad(@sprintf("%.2f", P), 14), rpad(@sprintf("%.3f", powR[i]), 9), rpad(hk_fmt_fap(fapR(i)), 11), kind)
end
# Best peak in the rotation window, for Plot 4.
rot_idx = [i for i in eachindex(perR) if ROT_RANGE[1] <= perR[i] <= min(ROT_RANGE[2], HK_MAX_PERIOD)]
ROT_I = isempty(rot_idx) ? 0 : rot_idx[argmax(powR[rot_idx])]
ROT_P = ROT_I == 0 ? NaN : perR[ROT_I]
ROT_FAP = ROT_I == 0 ? NaN : fapR(ROT_I)
dbg("Best peak in the rotation window ", ROT_RANGE, " d: P = ", round(ROT_P; digits = 2), " d, FAP ", hk_fmt_fap(ROT_FAP))
# a rotation period longer than the vsini limit cannot be the rotation of this star
ROT_EXCEEDS = isfinite(PROT_MAX) && isfinite(ROT_P) && ROT_P > 1.2 * PROT_MAX
ROT_EXCEEDS && dbg("!! this peak (", round(ROT_P; digits = 1), " d) is longer than the vsini limit (", round(PROT_MAX; digits = 1), " d): it is probably not the rotation period")
cyc_idx = [i for i in eachindex(perR) if perR[i] >= CYCLE_MIN]
CYC_I = isempty(cyc_idx) ? 0 : cyc_idx[argmax(powR[cyc_idx])]
CYC_P = CYC_I == 0 ? NaN : perR[CYC_I]
CYC_FAP = CYC_I == 0 ? NaN : fapR(CYC_I)
CYC_I == 0 ? dbg("No periods >= ", CYCLE_MIN, " d are probed (baseline/2 = ", round(HK_MAX_PERIOD), " d); no cycle search.") :
             dbg("Best peak at P >= ", CYCLE_MIN, " d: P = ", round(CYC_P; digits = 0), " d (", round(CYC_P / 365.25; digits = 1), " yr), FAP ", hk_fmt_fap(CYC_FAP))

# RV periodogram on the nights that have an RV (all of them have an R'HK, since those were selected above).
LS_V = HAS_RV ? hk_run_ls(N.mjd[isfinite.(N.rv)], N.rv[isfinite.(N.rv)]) : nothing
if LS_V !== nothing
    pgV, perV, powV = LS_V
    pkV = hk_top_peaks(perV, powV; n = 4)
    println("\nRV periodogram, strongest peaks (unweighted, no model removed; for the planet fit use OctofitterRVBank.jl):")
    println(rpad("  period [d]", 16), rpad("power", 9), rpad("FAP", 11), "at an R'HK peak?")
    for i in pkV
        P = perV[i]
        matches = [string(round(perR[j]; digits = 2)) for j in pkR if abs(perR[j] - P) / P < 0.05]
        harm = [string(round(perR[j]; digits = 2)) for j in pkR if abs(perR[j] / 2 - P) / P < 0.05]
        note = !isempty(matches) ? "yes: R'HK peak at " * join(matches, ", ") * " d" :
               !isempty(harm)    ? "first harmonic of the R'HK peak at " * join(harm, ", ") * " d" : "no"
        println("  ", rpad(@sprintf("%.2f", P), 14), rpad(@sprintf("%.3f", powV[i]), 9),
                rpad(hk_fmt_fap(LombScargle.fap(pgV, powV[i])), 11), note)
    end
end

# Plot 3 ----------------------------------------------------------------------
safe("HK Plot 3") do
    fig = Figure(size = (950, 420))
    ax = Axis(fig[1, 1], xlabel = "period [d]", ylabel = "Lomb-Scargle power", xscale = log10)
    Makie.lines!(ax, perR, powR, color = :black)
    for (p, ls, lab) in ((0.01, :dash, "FAP 1 %"), (0.001, :dot, "FAP 0.1 %"))
        try
            Makie.hlines!(ax, [LombScargle.fapinv(pgR, p)], color = (:crimson, 0.7), linestyle = ls, label = lab)
        catch
        end
    end
    Makie.vspan!(ax, ROT_RANGE[1], min(ROT_RANGE[2], HK_MAX_PERIOD), color = (:steelblue, 0.10))
    Makie.vlines!(ax, [365.25], color = (:gray, 0.6), linestyle = :dashdot)
    for i in pkR[1:min(3, end)]
        Makie.text!(ax, perR[i], powR[i]; text = @sprintf(" %.1f d", perR[i]), align = (:left, :bottom), fontsize = 12)
    end
    axislegend(ax, position = :rt, framevisible = false)
    save_plot!(fig, "HK Plot 3. $STARNAME: Lomb-Scargle periodogram of the nightly log R'HK. Red lines: false-alarm " *
                    "levels; shaded band: rotation window $(ROT_RANGE[1])-$(ROT_RANGE[2]) d; grey dash-dot line: 1 year " *
                    "(seasonal sampling alias). Strongest peaks are labelled.",
               "hk_plot03_periodogram_rhk.png")
end

# Plot 4: folded on the best rotation-range period ------------------------------
safe("HK Plot 4") do
    isfinite(ROT_P) || error("no rotation-window period")
    ph = mod.(N.mjd, ROT_P) ./ ROT_P
    fig = Figure(size = (800, 450))
    ax = Axis(fig[1, 1], xlabel = "phase at P = $(round(ROT_P; digits = 2)) d", ylabel = "log R'HK")
    # remove the slow trend first so a cycle does not blur the fold
    yr_med = [hk_nanmedian(N.lr[abs.(N.mjd .- t) .< 180]) for t in N.mjd]
    resid = N.lr .- yr_med
    Makie.scatter!(ax, vcat(ph, ph .+ 1), vcat(resid, resid), color = (:black, 0.5), markersize = 5)
    nb = 10
    edges = range(0, 1; length = nb + 1)
    cen = Float64[]; med = Float64[]
    for k in 1:nb
        i = findall(x -> edges[k] <= x < edges[k+1], ph)
        isempty(i) && continue
        push!(cen, (edges[k] + edges[k+1]) / 2); push!(med, mean(resid[i]))
    end
    Makie.scatterlines!(ax, vcat(cen, cen .+ 1), vcat(med, med), color = :crimson, linewidth = 2, markersize = 9)
    xlims!(ax, 0, 2)
    sig = ROT_FAP < 0.01 ? "significant (FAP $(hk_fmt_fap(ROT_FAP)))" : "NOT significant (FAP $(hk_fmt_fap(ROT_FAP))), so do not read this as a rotation period"
    save_plot!(fig, "HK Plot 4. $STARNAME: log R'HK folded on the best period in the rotation window, P = $(round(ROT_P; digits = 2)) d, " *
                    "after subtracting a +/-180 d running median (removes the long-term cycle). Red: ten-bin means. The peak is " * sig * ".",
               "hk_plot04_rotation_fold.png")
end

# Plot 5: seasonal means --------------------------------------------------------
section("B3. Does the RV follow the activity?")
OK_SEAS = HAS_RV && count(isfinite, SEAS.rv) >= 4
if OK_SEAS
    r_s = hk_pearson(SEAS.lr, SEAS.rv); n_s = hk_npair(SEAS.lr, SEAS.rv)
    rho_s = hk_spearman(SEAS.lr, SEAS.rv)
    a_s, b_s, eb_s = hk_linfit(SEAS.lr, SEAS.rv)
    dbg("Seasonal means (", n_s, " seasons): Pearson r = ", round(r_s; digits = 2), ", Spearman rho = ", round(rho_s; digits = 2),
        "; slope ", round(b_s; digits = 1), " +/- ", round(eb_s; digits = 1), " m/s per dex")
    safe("HK Plot 5") do
        fig = Figure(size = (950, 420))
        ax1 = Axis(fig[1, 1], xlabel = "year", ylabel = "mean log R'HK", yticklabelcolor = :steelblue)
        ax2 = Axis(fig[1, 1], ylabel = "mean RV [m/s]", yaxisposition = :right, yticklabelcolor = :crimson)
        hidespines!(ax2); hidexdecorations!(ax2)
        Makie.errorbars!(ax1, SEAS.year, SEAS.lr, SEAS.elr, color = (:steelblue, 0.5))
        Makie.scatterlines!(ax1, SEAS.year, SEAS.lr, color = :steelblue, markersize = 9)
        g = isfinite.(SEAS.rv)
        Makie.errorbars!(ax2, SEAS.year[g], SEAS.rv[g], SEAS.erv[g], color = (:crimson, 0.5))
        Makie.scatterlines!(ax2, SEAS.year[g], SEAS.rv[g], color = :crimson, markersize = 9)
        linkxaxes!(ax1, ax2)
        save_plot!(fig, "HK Plot 5. $STARNAME: calendar-year means of log R'HK (blue) and RV (red). Pearson r = " *
                        "$(round(r_s; digits = 2)) over $n_s seasons. If the two curves move together the long-term RV " *
                        "variation may be activity -- but the raw yearly RV means of a planet host mostly reflect which orbital phases each season sampled (see the residual test in Combined Plots 2-3).",
                   "hk_plot05_seasonal_means.png")
    end
else
    dbg("Plot 5 skipped: fewer than 4 seasons with both R'HK and RV")
end

# Plot 6: RV vs R'HK ------------------------------------------------------------
r_p = NaN; rho_p = NaN; n_p = 0; b_p = NaN; eb_p = NaN
if HAS_RV
    r_p = hk_pearson(N.lr, N.rv); rho_p = hk_spearman(N.lr, N.rv); n_p = hk_npair(N.lr, N.rv)
    _, b_p, eb_p = hk_linfit(N.lr, N.rv)
    dbg("Nightly RV vs log R'HK (", n_p, " nights): Pearson r = ", round(r_p; digits = 2), ", Spearman rho = ", round(rho_p; digits = 2),
        " (~", round(hk_corr_sigma(rho_p, n_p); digits = 1), " sigma: ", hk_corr_words(rho_p, n_p), "); slope ", round(b_p; digits = 1), " +/- ", round(eb_p; digits = 1), " m/s per dex")
    safe("HK Plot 6") do
        fig = Figure(size = (800, 560))
        ax = Axis(fig[1, 1], xlabel = "log R'HK", ylabel = "RV [m/s]")
        g = isfinite.(N.rv)
        sc = Makie.scatter!(ax, N.lr[g], N.rv[g], color = YEARS[g], colormap = :viridis, markersize = 6)
        Colorbar(fig[1, 2], sc, label = "year")
        a, b, _ = hk_linfit(N.lr, N.rv)
        xs = collect(range(minimum(N.lr[g]), maximum(N.lr[g]); length = 20))
        Makie.lines!(ax, xs, a .+ b .* xs, color = :crimson, linewidth = 2)
        save_plot!(fig, "HK Plot 6. $STARNAME: nightly RV against log R'HK, coloured by year. Spearman rho = $(round(rho_p; digits = 2)) " *
                        "($n_p nights, ~$(round(hk_corr_sigma(rho_p, n_p); digits = 1)) sigma); least-squares slope " *
                        "$(round(b_p; digits = 1)) +/- $(round(eb_p; digits = 1)) m/s per dex (red). A planet signal does not correlate with R'HK; stellar activity often does.",
                   "hk_plot06_rv_vs_rhk.png")
    end
end

# Plot 7: correlation with the other indicators ---------------------------------
INDS = [("RV", N.rv), ("CRX", N.crx), ("dLW", N.dlw), ("H-alpha", N.ha), ("Na D1", N.na), ("BIS", N.bis), ("FWHM", N.fw)]
rhos = [hk_spearman(N.lr, v) for (_, v) in INDS]
ns = [hk_npair(N.lr, v) for (_, v) in INDS]
println("\nSpearman rank correlation of nightly log R'HK with:")
for (k, (nm, _)) in enumerate(INDS)
    println("  ", rpad(nm, 9), "rho = ", lpad(isfinite(rhos[k]) ? @sprintf("%+.2f", rhos[k]) : "n/a", 6), "   n = ", lpad(ns[k], 5),
            "   ~", lpad(isfinite(rhos[k]) ? @sprintf("%.1f", hk_corr_sigma(rhos[k], ns[k])) : "-", 5), " sigma   ", hk_corr_words(rhos[k], ns[k]))
end
safe("HK Plot 7") do
    ok = findall(isfinite, rhos)
    isempty(ok) && error("no correlations to plot")
    fig = Figure(size = (800, 430))
    ax = Axis(fig[1, 1], ylabel = "Spearman rho with log R'HK", xticks = (1:length(ok), [INDS[k][1] for k in ok]))
    Makie.barplot!(ax, 1:length(ok), rhos[ok], color = [r > 0 ? :steelblue : :darkorange for r in rhos[ok]])
    nmin = minimum(ns[ok])
    Makie.hlines!(ax, [3 / sqrt(nmin), -3 / sqrt(nmin)], color = (:gray, 0.7), linestyle = :dash)
    Makie.hlines!(ax, [0], color = :black)
    ylims!(ax, -1, 1)
    save_plot!(fig, "HK Plot 7. $STARNAME: rank correlation of nightly log R'HK with the RV and with the other activity indicators " *
                    "in the table (CRX, dLW, H-alpha, Na D1, BIS, FWHM). Dashed lines: roughly 3 sigma for the smallest sample. " *
                    "Indicators that track R'HK but not the RV show the activity is not what drives the RV.",
               "hk_plot07_indicator_correlations.png")
end

# Plot 8: periodogram comparison ------------------------------------------------
safe("HK Plot 8") do
    LS_V === nothing && error("no RV periodogram")
    fig = Figure(size = (950, 520))
    ax = Axis(fig[1, 1], xlabel = "period [d]", ylabel = "power / max power", xscale = log10)
    Makie.lines!(ax, perR, powR ./ maximum(powR), color = :steelblue, linewidth = 1.8, label = "log R'HK")
    Makie.lines!(ax, perV, powV ./ maximum(powV), color = :crimson, linewidth = 1.8, label = "RV")
    for i in pkV
        Makie.vlines!(ax, [perV[i]], color = (:crimson, 0.35), linestyle = :dash)
    end
    Makie.vlines!(ax, [365.25], color = (:gray, 0.6), linestyle = :dashdot)
    axislegend(ax, position = :rt, framevisible = false)
    save_plot!(fig, "HK Plot 8. $STARNAME: periodograms of log R'HK (blue) and of the RV (red), each divided by its maximum. " *
                    "Dashed red lines mark the four strongest RV peaks. An RV peak that coincides with an R'HK peak (or half its " *
                    "period) is more likely stellar activity than a planet. Grey dash-dot: 1 year.",
               "hk_plot08_periodogram_comparison.png")
end

# ------------------------------------------------------------------ 3. jitter floor
section("B4. Activity-jitter lower limit (Perdelwitz et al. 2024, Eqs. 1-3)")
const JITTER_FIT = Dict("F" => (0.81, 4.55), "G" => (0.91, 4.91), "K" => (0.57, 3.04))
JITTER_FLOOR = NaN
if haskey(JITTER_FIT, SPTYPE)
    a, b = JITTER_FIT[SPTYPE]
    global JITTER_FLOOR = 10.0^(a * LOGR_MED + b)
    dbg("Spectral class ", SPTYPE, ": log sigma_RV [m/s] = ", a, " x log R'HK + ", b, ", log R'HK = ", round(LOGR_MED; digits = 2),
        "  ->  lower limit on the RV jitter ", round(JITTER_FLOOR; digits = 2), " m/s")
    HAS_RV && dbg("Observed RV scatter (robust, per night, all signals included): ", round(RV_SIGMA; digits = 2), " m/s")
else
    dbg("Spectral class ", SPTYPE, ": the relations were fitted to F, G and K stars only, so no jitter limit is computed.")
end

# ------------------------------------------------------------------ 4. summary
section("B5. Summary for $STARNAME")
println("Data:  ", SOURCE_USED, "; ", length(CLEAN.bjd), " usable spectra on ", NN, " nights (",
        round(HK_BASELINE / 365.25; digits = 1), " yr).")
println("Level: median log R'HK = ", round(LOGR_MED; digits = 2), " (R'HK = ", @sprintf("%.1e", 10.0^LOGR_MED),
        "), 16-84 % range ", round(quantile(N.lr, 0.16); digits = 2), " to ", round(quantile(N.lr, 0.84); digits = 2),
        ". Nightly error ", round(median(N.elr); digits = 2), " dex.")
if hk_robust_sigma(N.lr) < 1.5 * median(N.elr)
    println("       The spread is close to the measurement error, so no activity variation is detectable above the noise.")
end
if isfinite(ROT_P)
    println("Rotation window: strongest period ", round(ROT_P; digits = 2), " d, FAP ", hk_fmt_fap(ROT_FAP),
            ROT_FAP < 0.01 ? " -> a rotation signal is detected (could also be a harmonic)." :
                             " -> not significant; no rotation period from R'HK alone.")
end
if isfinite(PROT_MAX) && isfinite(ROT_P)
    println("vsini check:     P_rot <= ", round(PROT_MAX; digits = 1), " d from vsini = ", hk_f(STAR_P.vsini), " km/s; the R'HK peak at ", round(ROT_P; digits = 1), " d is ",
            ROT_EXCEEDS ? "LONGER than that, so it is probably not the rotation period." : "consistent with that limit.")
end
if STAR_P !== nothing && isfinite(STAR_P.rhk) && STAR_P.rhk != 0
    cat_log = STAR_P.rhk < 0 ? STAR_P.rhk : log10(STAR_P.rhk)
    cat_ok = isfinite(STAR_P.erhk) && abs(STAR_P.erhk) < 0.5 * abs(STAR_P.rhk)   # an error comparable to the value is no measurement
    println("Catalogue check: table1 R'HK gives log R'HK = ", round(cat_log; digits = 2), " (error ", hk_f(STAR_P.erhk, "%.2g"), "), this analysis ", round(LOGR_MED; digits = 2),
            !cat_ok ? "  (the catalogue error is as large as the value, so it is not a useful comparison; Sourflag " * hk_f(STAR_P.sourflag, "%.0f") * ")." :
            abs(cat_log - LOGR_MED) > 0.15 ? "  (differ by more than 0.15 dex: different cleaning/averaging)." : "  (agree).")
end
if isfinite(CYC_P)
    println("Long periods:    strongest period ", round(CYC_P / 365.25; digits = 1), " yr, FAP ", hk_fmt_fap(CYC_FAP),
            CYC_FAP < 0.01 ? " -> a long-term activity trend or cycle is present." : " -> no significant long-term cycle.")
end
if HAS_RV
    println("RV vs activity: Spearman rho = ", @sprintf("%+.2f", rho_p), " on ", n_p, " nights (", hk_corr_words(rho_p, n_p), ")",
            isfinite(b_p) ? @sprintf(", slope %.1f +/- %.1f m/s per dex", b_p, eb_p) : "", ".")
    OK_SEAS && println("                seasonal means: r = ", @sprintf("%+.2f", r_s), " over ", n_s, " seasons (raw RV: dominated by planet phase sampling; see Part C).")
    if isfinite(JITTER_FLOOR)
        println("Jitter floor:   ", round(JITTER_FLOOR; digits = 1), " m/s expected at least, observed RV scatter ",
                round(RV_SIGMA; digits = 1), " m/s",
                RV_SIGMA < JITTER_FLOOR ? " (below the floor: this star is quieter than the typical lower limit)." :
                                          " (the scatter is above the floor; planets or other noise add to it).")
    end
end
println("\nCaution: correlations of ~", round(3 / sqrt(max(n_p, 10)); digits = 2), " or less are not distinguishable from zero for this sample size.")

end  # if RUN_ACTIVITY

# ==============================================================================
# PART A (continued). Planet fits with Octofitter
# ==============================================================================
if RUN_PLANET_FIT

# ------------------------------------------------------------------ 3. model pieces
section("A3. Model definition")

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
section("A4. Fits with Pigeons")
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
    section("A4.$k  $k-planet model")
    f = fit_planets(k)
    f === nothing && continue
    push!(fits, f)
    display(f.chain)
    dbg("ln Z ($k planet" * (k == 1 ? "" : "s") * ") = ", round(f.lnZ; digits = 2))
end
isempty(fits) && error("Every fit failed; see the warnings above.")
final = fits[end]
const K_FINAL = final.k

safe("RV Plot 3") do
    res = rvplot(final.model, final.chain)
    save_plot!(res, "RV Plot 3. $(K_FINAL)-planet fit of $STARNAME: RV curve for the maximum-posterior " *
                    "draw, residuals, and the phase-folded signal of each planet.",
               "rv_plot03_rvplot_$(K_FINAL)_planets.png")
end
safe("RV Plot 4") do
    res = octoplot(final.model, final.chain)
    save_plot!(res, "RV Plot 4. $(K_FINAL)-planet fit of $STARNAME: spread of orbits allowed by the " *
                    "posterior over the HARPS data.", "rv_plot04_octoplot_$(K_FINAL)_planets.png")
end

# ------------------------------------------------------------------ 5. model comparison
section("A5. Model comparison: log Bayesian evidence")
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
section("A6. Corner plots")
safe("RV Plot 5") do
    fig = octocorner(final.model, final.chain, small = true)
    save_plot!(fig, "RV Plot 5. Corner plot of the $(K_FINAL)-planet posterior for $STARNAME, showing parameter covariances.",
               "rv_plot05_corner_$(K_FINAL)_planets.png")
end
if length(fits) >= 2
    safe("RV Plot 6") do
        # the larger model contains every column of the smaller one
        fig = octocorner(fits[end].model, fits[end-1].chain, fits[end].chain, small = true)
        save_plot!(fig, "RV Plot 6. How adding planet $(PLANET_NAMES[K_FINAL]) changes the other parameters. " *
                        "Series 1 (blue) is the $(fits[end-1].k)-planet fit, series 2 (orange) the $(K_FINAL)-planet fit.",
                   "rv_plot06_corner_$(fits[end-1].k)_vs_$(K_FINAL)_planets.png")
    end
end

# ------------------------------------------------------------------ 7. eccentricity test
fit_alt = nothing
if RUN_ECCENTRICITY_TEST && K_FINAL == N_PLANETS
    alt_name = FREE_ECCENTRICITY ? "circular" : "free-eccentricity"
    section("A7. Eccentricity test: the same $(N_PLANETS)-planet model with $alt_name orbits")
    fit_alt = fit_planets(N_PLANETS, planets_alt; label = alt_name == "circular" ? "circ" : "ecc")
    if fit_alt !== nothing
        display(fit_alt.chain)
        free_fit = FREE_ECCENTRICITY ? final : fit_alt
        circ_fit = FREE_ECCENTRICITY ? fit_alt : final
        lnbf_ecc = free_fit.lnZ - circ_fit.lnZ
        dbg("ln Z ($alt_name, $(N_PLANETS) planets) = ", round(fit_alt.lnZ; digits = 2))
        dbg("ln BF (free eccentricity vs circular) = ", round(lnbf_ecc; digits = 2), "  ->  ",
            bf_interpretation(lnbf_ecc; more = "free eccentricity", less = "circular orbits"))
        safe("RV Plot 7") do
            res = rvplot(fit_alt.model, fit_alt.chain)
            save_plot!(res, "RV Plot 7. Eccentricity test: $(N_PLANETS)-planet fit of $STARNAME with $alt_name orbits: " *
                            "RV curve for the maximum-posterior draw, residuals, and phase-folded signals.",
                       "rv_plot07_rvplot_$(N_PLANETS)_planets_$(alt_name).png")
        end
        safe("RV Plot 8") do
            # the free-eccentricity model has every column of the circular one
            fig = octocorner(free_fit.model, circ_fit.chain, free_fit.chain, small = true)
            save_plot!(fig, "RV Plot 8. Effect of allowing eccentricity on the $(N_PLANETS)-planet fit of $STARNAME. " *
                            "Series 1 (blue) is the circular fit, series 2 (orange) the free-eccentricity fit.",
                       "rv_plot08_corner_circular_vs_eccentric.png")
        end
    end
end

# ------------------------------------------------------------------ 8. extra-planet test
fit_extra = nothing
if RUN_EXTRA_PLANET_TEST && K_FINAL == N_PLANETS
    k_extra = N_PLANETS + 1
    section("A8. Extra-planet test: $(k_extra) planets (planet $(PLANET_NAMES[k_extra]) at the next periodogram peak, " *
            "$(round(PERIOD_GUESSES[k_extra]; digits = 2)) d)")
    fit_extra = fit_planets(k_extra)
    if fit_extra !== nothing
        display(fit_extra.chain)
        lnbf_extra = fit_extra.lnZ - final.lnZ
        dbg("ln Z ($(k_extra) planets) = ", round(fit_extra.lnZ; digits = 2))
        dbg("ln BF ($(k_extra) vs $(N_PLANETS) planets) = ", round(lnbf_extra; digits = 2), "  ->  ",
            bf_interpretation(lnbf_extra; more = "$(k_extra) planets", less = "$(N_PLANETS) planet" * (N_PLANETS == 1 ? "" : "s")))
        safe("RV Plot 9") do
            res = rvplot(fit_extra.model, fit_extra.chain)
            save_plot!(res, "RV Plot 9. Extra-planet test: $(k_extra)-planet fit of $STARNAME (planets " *
                            "$(join(PLANET_NAMES[1:k_extra], ", "))): RV curve for the maximum-posterior draw, " *
                            "residuals, and phase-folded signals.",
                       "rv_plot09_rvplot_$(k_extra)_planets.png")
        end
        safe("RV Plot 10") do
            fig = octocorner(fit_extra.model, final.chain, fit_extra.chain, small = true)
            save_plot!(fig, "RV Plot 10. Effect of adding planet $(PLANET_NAMES[k_extra]) to $STARNAME. Series 1 (blue) " *
                            "is the $(N_PLANETS)-planet fit, series 2 (orange) the $(k_extra)-planet fit.",
                       "rv_plot10_corner_$(N_PLANETS)_vs_$(k_extra)_planets.png")
        end
    end
end

# ------------------------------------------------------------------ 9. analysis summary
section("A9. Analysis summary")
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


end  # if RUN_PLANET_FIT

# ==============================================================================
# PART C. The two analyses together
# ==============================================================================
if RUN_ACTIVITY

section("C1. RV residuals after the planet model, compared with the activity indicators")

# The planet model used to make the residuals: the planets of the final fit (or the
# periodogram peaks if the fits were switched off), each as a fundamental plus its
# first harmonic. Two harmonics follow an orbit with e up to ~0.3 to a few percent of its
# amplitude, so this is a good stand-in for the Keplerians without needing the fit internals.
const C_K = final === nothing ? min(N_PLANETS, length(PERIOD_GUESSES)) : final.k
const C_PERIODS = Float64[
    (final !== nothing && has_col(final.chain, "$(PLANET_NAMES[k])_P")) ?
        median(chain_vec(final.chain, "$(PLANET_NAMES[k])_P")) : PERIOD_GUESSES[k]
    for k in 1:C_K]
const C_SRC = final === nothing ? "the periodogram peaks (the planet fits were not run)" :
              "the medians of the $(C_K)-planet fit"
dbg("Planet model for the residuals: ", C_K, " planet(s) at P = ", join(round.(C_PERIODS; digits = 3), ", "),
    " d, from ", C_SRC)

function harmonic_fit(t, y, σ, periods; nharm = 2)
    t0 = mean(t)
    cols = Vector{Float64}[ones(length(t))]
    for P in periods, h in 1:nharm
        ph = 2π * h .* (t .- t0) ./ P
        push!(cols, sin.(ph)); push!(cols, cos.(ph))
    end
    X = reduce(hcat, cols)
    w = 1 ./ σ
    β = (X .* w) \ (y .* w)
    mdl = X * β
    return y .- mdl, mdl
end

const RV_T = Float64.(collect(rv_data.epoch))
const RV_Y = Float64.(collect(rv_data.rv))
const RV_S = Float64.(collect(rv_data.σ_rv))
length(RV_T) > 4 * C_K + 10 || error("Too few RVs for the residual model")
const _HF = harmonic_fit(RV_T, RV_Y, RV_S, C_PERIODS)
const RES_ALL = _HF[1]
const MDL_ALL = _HF[2]
const RES_RMS = std(RES_ALL)
const JIT_FIT = final === nothing ? NaN : median(chain_vec(final.chain, "HARPS_jitter"))
dbg("RV scatter before: ", round(std(RV_Y); digits = 1), " m/s;  after removing the planets: ", round(RES_RMS; digits = 1),
    " m/s;  median RV error ", round(median(RV_S); digits = 2), " m/s",
    isfinite(JIT_FIT) ? ";  fitted jitter " * string(round(JIT_FIT; digits = 1)) * " m/s" : "")

"Residual of the RV point nearest to each night (NaN when none is within 0.6 d)."
function align_to_nights(res_t, res_y, night_t)
    out = fill(NaN, length(night_t))
    for (i, t) in enumerate(night_t)
        j = argmin(abs.(res_t .- t))
        abs(res_t[j] - t) < 0.6 && (out[i] = res_y[j])
    end
    return out
end
const RES_N = align_to_nights(RV_T, RES_ALL, N.mjd)
dbg("Residuals matched to ", count(isfinite, RES_N), " of the ", NN, " activity nights")

const IND_C = [("log R'HK", N.lr), ("CRX", N.crx), ("dLW", N.dlw), ("H-alpha", N.ha),
               ("Na D1", N.na), ("BIS", N.bis), ("FWHM", N.fw)]
const RHO_RAW = [hk_spearman(v, N.rv) for (_, v) in IND_C]
const RHO_RES = [hk_spearman(v, RES_N) for (_, v) in IND_C]
const N_RES   = [hk_npair(v, RES_N) for (_, v) in IND_C]
println("\nSpearman correlation of each activity indicator with the RV, before and after removing the planets:")
println(rpad("  indicator", 13), rpad("raw RV", 10), rpad("residual RV", 14), "n(resid)  significance of the residual correlation")
for (k, (nm, _)) in enumerate(IND_C)
    println("  ", rpad(nm, 11), rpad(isfinite(RHO_RAW[k]) ? @sprintf("%+.2f", RHO_RAW[k]) : "n/a", 10),
            rpad(isfinite(RHO_RES[k]) ? @sprintf("%+.2f", RHO_RES[k]) : "n/a", 14), rpad(N_RES[k], 9),
            hk_corr_words(RHO_RES[k], N_RES[k]), isfinite(RHO_RES[k]) ? @sprintf(" (~%.1f sigma)", hk_corr_sigma(RHO_RES[k], N_RES[k])) : "")
end
const RHO_RES_RHK = RHO_RES[1]
const N_RES_RHK   = N_RES[1]
_, B_RES, EB_RES = N_RES_RHK >= 5 ? hk_linfit(N.lr, RES_N) : (0.0, NaN, NaN)
const ACT_RV  = isfinite(B_RES) ? abs(B_RES) * hk_robust_sigma(N.lr) : NaN       # RV rms the R'HK slope would explain
const ACT_RVE = isfinite(EB_RES) ? EB_RES * hk_robust_sigma(N.lr) : NaN
dbg("Residual RV vs log R'HK: slope ", round(B_RES; digits = 1), " +/- ", round(EB_RES; digits = 1),
    " m/s per dex  ->  activity-correlated RV of about ", round(ACT_RV; digits = 1), " +/- ", round(ACT_RVE; digits = 1),
    " m/s rms, against a residual scatter of ", round(RES_RMS; digits = 1), " m/s")

# Periodogram of the residuals, on the same nights as the R'HK one.
const OK_RES = isfinite.(RES_N)
const LS_RES = count(OK_RES) >= MIN_NIGHTS ? hk_run_ls(N.mjd[OK_RES], RES_N[OK_RES]) : nothing
if LS_RES !== nothing
    pgRes, perRes, powRes = LS_RES
    pkRes = hk_top_peaks(perRes, powRes; n = 4)
    println("\nPeriodogram of the RV residuals, strongest peaks (what is left after the planet model):")
    println(rpad("  period [d]", 16), rpad("power", 9), rpad("FAP", 11), "R'HK has power there?")
    for i in pkRes
        P = perRes[i]
        j = findall(p -> abs(p - P) / P < 0.03, perR)
        pr = isempty(j) ? NaN : maximum(powR[j])
        sig = !isempty(j) && LombScargle.fap(pgR, pr) < 0.01
        println("  ", rpad(@sprintf("%.2f", P), 14), rpad(@sprintf("%.3f", powRes[i]), 9),
                rpad(hk_fmt_fap(LombScargle.fap(pgRes, powRes[i])), 11),
                sig ? "yes (FAP " * hk_fmt_fap(LombScargle.fap(pgR, pr)) * ")" : "no")
    end
end

# --- Combined plots -------------------------------------------------------------
safe("Combined Plot 1") do
    fig = Figure(size = (950, 860))
    ax1 = Axis(fig[1, 1], ylabel = "log R'HK")
    ax2 = Axis(fig[2, 1], ylabel = "RV [m/s]")
    ax3 = Axis(fig[3, 1], ylabel = "RV residual [m/s]", xlabel = "year")
    linkxaxes!(ax1, ax2, ax3)
    Makie.scatter!(ax1, YEARS, N.lr, color = :black, markersize = 5)
    yv = hk_decyear(RV_T .- 0.0)
    Makie.scatter!(ax2, yv, RV_Y, color = :black, markersize = 5)
    Makie.scatter!(ax2, yv, MDL_ALL, color = (:crimson, 0.7), markersize = 3)
    Makie.scatter!(ax3, yv, RES_ALL, color = :black, markersize = 5)
    Makie.hlines!(ax3, [0], color = :crimson)
    hidexdecorations!(ax1, grid = false); hidexdecorations!(ax2, grid = false)
    save_plot!(fig, "Combined Plot 1. $STARNAME on one time axis: log R'HK (top), the RVs with the planet model (red dots, " *
                    "middle), and the residuals after removing the planets (bottom). If activity were driving what is left, " *
                    "the bottom panel would follow the top one.",
               "combined_plot01_timeseries_residuals.png")
end

safe("Combined Plot 2") do
    fig = Figure(size = (800, 560))
    ax = Axis(fig[1, 1], xlabel = "log R'HK", ylabel = "RV residual after the planet model [m/s]")
    g = isfinite.(RES_N)
    sc = Makie.scatter!(ax, N.lr[g], RES_N[g], color = YEARS[g], colormap = :viridis, markersize = 6)
    Colorbar(fig[1, 2], sc, label = "year")
    if isfinite(B_RES)
        a, b, _ = hk_linfit(N.lr, RES_N)
        xs = collect(range(minimum(N.lr[g]), maximum(N.lr[g]); length = 20))
        Makie.lines!(ax, xs, a .+ b .* xs, color = :crimson, linewidth = 2)
    end
    save_plot!(fig, "Combined Plot 2. $STARNAME: RV residuals (planets removed) against log R'HK. Spearman rho = " *
                    "$(round(RHO_RES_RHK; digits = 2)) ($N_RES_RHK nights, ~$(round(hk_corr_sigma(RHO_RES_RHK, N_RES_RHK); digits = 1)) sigma); " *
                    "slope $(round(B_RES; digits = 1)) +/- $(round(EB_RES; digits = 1)) m/s per dex, i.e. about $(round(ACT_RV; digits = 1)) m/s " *
                    "of activity-correlated RV against a residual scatter of $(round(RES_RMS; digits = 1)) m/s.",
               "combined_plot02_residual_vs_rhk.png")
end

safe("Combined Plot 3") do
    nan0(x) = isfinite(x) ? x : 0.0
    n = length(IND_C)
    fig = Figure(size = (900, 460))
    ax = Axis(fig[1, 1], ylabel = "Spearman rho", xticks = (1:n, [nm for (nm, _) in IND_C]))
    xs = vcat(1:n, 1:n)
    grp = vcat(fill(1, n), fill(2, n))
    vals = vcat(nan0.(RHO_RAW), nan0.(RHO_RES))
    Makie.barplot!(ax, xs, vals; dodge = grp, color = [g == 1 ? :steelblue : :crimson for g in grp])
    Makie.hlines!(ax, [0], color = :black)
    nmin = max(minimum(filter(>(0), N_RES); init = 20), 5)
    Makie.hlines!(ax, [3 / sqrt(nmin), -3 / sqrt(nmin)], color = (:gray, 0.7), linestyle = :dash)
    ylims!(ax, -1, 1)
    Legend(fig[1, 2], [PolyElement(color = :steelblue), PolyElement(color = :crimson)], ["raw RV", "RV residuals"])
    save_plot!(fig, "Combined Plot 3. $STARNAME: Spearman correlation of log R'HK and the other indicators with the RV before " *
                    "(blue) and after (red) removing the planets. The first pair is log R'HK itself. Dashed lines: roughly 3 sigma. " *
                    "Removing a large planet signal lets a small activity correlation show; a correlation that was there before and " *
                    "vanishes afterwards was the planet's phase sampling.",
               "combined_plot03_indicator_correlations.png")
end

safe("Combined Plot 4") do
    fig = Figure(size = (1000, 580))
    ax = Axis(fig[1, 1], xlabel = "period [d]", ylabel = "power / max power", xscale = log10)
    Makie.lines!(ax, perR, powR ./ maximum(powR), color = :steelblue, linewidth = 1.6, label = "log R'HK")
    LS_V !== nothing && Makie.lines!(ax, perV, powV ./ maximum(powV), color = (:black, 0.6), linewidth = 1.3, label = "RV (raw)")
    LS_RES !== nothing && Makie.lines!(ax, perRes, powRes ./ maximum(powRes), color = :crimson, linewidth = 1.6, label = "RV residuals")
    for (k, P) in enumerate(C_PERIODS)
        Makie.vlines!(ax, [P], color = (:black, 0.6), linestyle = :dash)
        Makie.text!(ax, P, 1.02; text = " " * PLANET_NAMES[k], align = (:left, :bottom), fontsize = 13)
    end
    if length(PERIOD_GUESSES) > C_K
        Makie.vlines!(ax, [PERIOD_GUESSES[C_K+1]], color = (:darkorange, 0.8), linestyle = :dot)
    end
    if isfinite(ROT_P) && ROT_FAP < 0.01
        Makie.vlines!(ax, [ROT_P / n for n in 1:3], color = (:steelblue, 0.6), linestyle = :dot)
    end
    Makie.vlines!(ax, [365.25], color = (:gray, 0.6), linestyle = :dashdot)
    ylims!(ax, 0, 1.12)
    axislegend(ax, position = :lt, framevisible = false)
    save_plot!(fig, "Combined Plot 4. $STARNAME: periodograms of log R'HK (blue), the raw RV (black) and the RV residuals (red). " *
                    "Dashed black lines: fitted planet periods; orange dotted line: the next periodogram peak (extra-planet candidate); " *
                    "blue dotted lines: the rotation period and its half and third (when the rotation peak is significant); grey: 1 year. " *
                    "Residual power that lines up with blue peaks points to activity.",
               "combined_plot04_periodograms.png")
end

safe("Combined Plot 5") do
    labels = String[]; vals = Float64[]
    add!(l, v) = (isfinite(v) && v > 0) && (push!(labels, l); push!(vals, v))
    add!("raw RV scatter (robust)", RV_SIGMA)
    add!("median RV error", median(RV_S))
    for f in fits
        add!("fitted jitter, $(f.k) planet" * (f.k == 1 ? "" : "s"), median(chain_vec(f.chain, "HARPS_jitter")))
    end
    fit_extra === nothing || add!("fitted jitter, $(fit_extra.k) planets (extra-planet test)", median(chain_vec(fit_extra.chain, "HARPS_jitter")))
    add!("scatter left after the harmonic planet model", RES_RMS)
    add!("RV that tracks R'HK (|slope| x scatter of log R'HK)", ACT_RV)
    add!("activity-jitter lower limit (Perdelwitz et al. 2024)", JITTER_FLOOR)
    n = length(vals)
    fig = Figure(size = (950, 140 + 50 * n))
    ax = Axis(fig[1, 1], xscale = log10, xlabel = "m/s", yticks = (1:n, labels), yreversed = true)
    Makie.scatter!(ax, vals, 1:n, color = :crimson, markersize = 12)
    for i in 1:n
        Makie.text!(ax, vals[i], i; text = @sprintf("  %.3g", vals[i]), align = (:left, :center), fontsize = 12)
    end
    save_plot!(fig, "Combined Plot 5. $STARNAME: the RV noise budget. The raw scatter is dominated by the planets; the fitted jitter and the " *
                    "residual scatter show what remains, and the R'HK-tracking amplitude shows how much of that activity could explain.",
               "combined_plot05_noise_budget.png")
end

# ------------------------------------------------------------------ C2. agreement and differences
section("C2. Where the RV and the activity analyses agree and differ")
const SUMLINES = String[]
function emit(s = "")
    println(s); push!(SUMLINES, String(s)); return nothing
end
const PLANET_LIST = [(PLANET_NAMES[k], C_PERIODS[k]) for k in 1:C_K]

"Explain a periodogram peak in terms of the fitted planets: the planet itself, its 1-year alias, or its half period."
function explain_peak(P)
    for (nm, Pj) in PLANET_LIST
        abs(P - Pj) / Pj < 0.03 && return "planet $nm ($(round(Pj; digits = 2)) d)"
    end
    for (nm, Pj) in PLANET_LIST, s in (-1, 1)
        abs(1 / P - (1 / Pj + s / 365.25)) * P < 0.03 && return "annual alias of planet $nm"
    end
    for (nm, Pj) in PLANET_LIST
        abs(P - Pj / 2) / P < 0.03 && return "half the period of planet $nm"
    end
    return "not explained by the planet model"
end

emit("Star: $STARNAME   data: $SOURCE_USED")
if STAR_P !== nothing
    emit("Stellar parameters (VizieR table1): Teff = " * hk_f(STAR_P.teff, "%.0f") * " K, logg = " * hk_f(STAR_P.logg) * ", [Fe/H] = " * hk_f(STAR_P.feh, "%+.2f") *
         ", vsini = " * hk_f(STAR_P.vsini) * " km/s, class " * SPTYPE)
    isfinite(PROT_MAX) && emit("Rotation limit from vsini: P_rot <= " * hk_f(PROT_MAX, "%.1f") * " d" *
        (ROT_EXCEEDS ? "; the R'HK peak at " * hk_f(ROT_P, "%.1f") * " d exceeds it, so it is not used as a rotation period in the candidate vetting." : "."))
end
emit("Planet model: $(C_K) planet(s), periods " * join((string(PLANET_NAMES[k], "=", round(C_PERIODS[k]; digits = 2), " d") for k in 1:C_K), ", ") * " (" * C_SRC * ")")
emit()
emit("AGREEMENT")
if LS_V !== nothing
    emit("  RV periodogram peaks (raw RV) explained by the planet fit:")
    for i in pkV
        emit("    P = " * rpad(@sprintf("%.2f d", perV[i]), 10) * " -> " * explain_peak(perV[i]))
    end
end
emit("  The R'HK periodogram peaks against the planet periods:")
for i in pkR
    P = perR[i]
    hit = [nm for (nm, Pj) in PLANET_LIST if abs(P - Pj) / Pj < 0.05]
    rot = isfinite(ROT_P) && ROT_FAP < 0.01 && !ROT_EXCEEDS && any(n -> abs(P - ROT_P / n) / P < 0.05, 1:4)
    emit("    P = " * rpad(@sprintf("%.2f d", P), 10) * "FAP " * rpad(hk_fmt_fap(fapR(i)), 9) *
         (isempty(hit) ? "no planet at this period" : "COINCIDES with planet " * join(hit, ", ")) *
         (rot ? "; rotation harmonic" : ""))
end
emit()
emit("DIFFERENCES")
emit(@sprintf("  Raw RV scatter %.0f m/s  vs  after the planets %.1f m/s  vs  median RV error %.2f m/s", RV_SIGMA, RES_RMS, median(RV_S)) *
     (isfinite(JIT_FIT) ? @sprintf("  vs  fitted jitter %.1f m/s", JIT_FIT) : ""))
emit(@sprintf("  log R'HK varies by %.2f dex (robust sigma) with measurement error %.3f dex, so the activity signal is %s.",
              hk_robust_sigma(N.lr), median(N.elr), hk_robust_sigma(N.lr) > 3 * median(N.elr) ? "real" : "barely above the noise"))
emit(@sprintf("  Raw RV vs log R'HK: rho = %+.2f;  residual RV vs log R'HK: rho = %+.2f (~%.1f sigma, %s)",
              RHO_RAW[1], RHO_RES_RHK, hk_corr_sigma(RHO_RES_RHK, N_RES_RHK), hk_corr_words(RHO_RES_RHK, N_RES_RHK)))
emit(@sprintf("  A line through residual RV vs R'HK explains about %.1f +/- %.1f m/s rms, i.e. %s of the leftover scatter.",
              ACT_RV, ACT_RVE, isfinite(ACT_RV) ? @sprintf("%.0f%%", 100 * (ACT_RV / RES_RMS)^2) : "n/a"))
emit("  The seasonal-mean correlation in HK Plot 5 uses raw RV, so it mostly reflects which orbital phases each season sampled; the residual test above is the fair one.")

# ------------------------------------------------------------------ C3. candidate vetting
section("C3. Candidate signals: planets or activity?")

"Stronger/earlier planets only, with their periods."
function ecc_of(chain, nm)
    (chain !== nothing && has_col(chain, "$(nm)_e")) ? chain_vec(chain, "$(nm)_e") : Float64[0.0]
end
function mass_of(chain, nm)
    (chain !== nothing && has_col(chain, "$(nm)_mass")) ? median(chain_vec(chain, "$(nm)_mass")) / mjup : NaN
end
function rhk_power_near(P; tol = 0.03)
    idx = findall(p -> abs(p - P) / P <= tol, perR)
    isempty(idx) && return (NaN, NaN)
    i = idx[argmax(powR[idx])]
    return powR[i], LombScargle.fap(pgR, powR[i])
end

"""
Vet one candidate against everything the two analyses know. `earlier` are the (name, period,
mass, e84) of the planets before it. A verdict is a heuristic: it names a reason to doubt a
signal, not a proof, and a clean verdict does not prove a planet.
"""
function vet(nm, P, mass, earlier, chain_extra)
    notes = String[]
    verdict = "no activity, alias or harmonic match"
    # activity
    pw, fp = rhk_power_near(P)
    act = [round(perR[i]; digits = 1) for i in pkR if abs(perR[i] - P) / P < 0.05 && fapR(i) < 0.01]
    rot = ""
    if isfinite(ROT_P) && ROT_FAP < 0.01 && !ROT_EXCEEDS
        for n in 1:4
            abs(P - ROT_P / n) / P < 0.05 && (rot = n == 1 ? "the rotation period" : "P_rot/$n")
        end
        abs(P - 2 * ROT_P) / P < 0.05 && (rot = "twice the rotation period")
    end
    isfinite(fp) && fp < 0.01 && push!(notes, "R'HK has significant power near this period (FAP " * hk_fmt_fap(fp) * ")")
    isempty(act) || push!(notes, "R'HK peak at " * join(string.(act), ", ") * " d")
    if rot != ""
        if isfinite(fp) && fp < 0.01
            push!(notes, "matches " * rot * " and R'HK has power here")
        else
            push!(notes, "period is close to " * rot * ", but R'HK shows no significant power at this period, so this is treated as a coincidence")
        end
    end
    # a rotation-harmonic match alone is not enough: the activity indicator must also show the period
    activity_flag = !isempty(act) || (isfinite(fp) && fp < 0.01)
    # aliases and harmonics of earlier planets
    alias_flag = false; harm_flag = false; res_flag = false; untested = false
    for (en, eP, em, ee) in earlier
        for s in (-1, 1)
            if abs(1 / P - (1 / eP + s / 365.25)) * P < 0.02
                alias_flag = true; push!(notes, "annual alias of planet $en")
            end
        end
        for (a, b) in ((2, 1), (3, 1), (4, 1), (3, 2), (5, 2), (5, 3), (4, 3))
            r = max(P, eP) / min(P, eP)
            if abs(r - a / b) / (a / b) < 0.03
                res_flag = true; push!(notes, "period ratio $(a):$(b) with planet $en (resonance or harmonic)")
                # could it merely be a harmonic of an eccentric orbit? its RV amplitude would be about e x K of the stronger one
                if b == 1 && a >= 2
                    dpre = 360.0 * 365.25 * (1 / P - a / eP)
                    push!(notes, @sprintf("offset from the exact harmonic %d/P_%s would be matched by apsidal precession of about %+.0f deg/yr (compare with dynamical models for resonant pairs)", a, en, dpre))
                end
                if P < eP && b == 1 && !(isfinite(mass) && isfinite(em))
                    untested = true
                    push!(notes, "harmonic test not run: it needs the planet fits (amplitude against e x K of planet $en)")
                end
                if P < eP && isfinite(mass) && isfinite(em)
                    kratio = (mass / em) * (eP / P)^(1 / 3)
                    if kratio < 2 * (ee + 0.02)
                        harm_flag = true
                        push!(notes, @sprintf("its amplitude (%.2f of planet %s) is no more than an eccentric orbit of planet %s (e up to %.2f) would put at the harmonic", kratio, en, en, ee))
                    else
                        push!(notes, @sprintf("its amplitude (%.2f of planet %s) is too large for just the harmonic of planet %s (e up to %.2f)", kratio, en, en, ee))
                    end
                end
            end
        end
    end
    if occursin("876", string(STARNAME)) && abs(P - 1.9376) / 1.9376 < 0.03
        push!(notes, "matches the published inner planet GJ 876 d (P = 1.94 d)")
    end
    verdict = activity_flag ? "ACTIVITY-SUSPECT" :
              alias_flag    ? "ALIAS-SUSPECT" :
              harm_flag     ? "HARMONIC-SUSPECT (may be an unmodelled eccentricity or interaction of an earlier planet)" :
              (res_flag && untested) ? "UNTESTED harmonic (2:1/3:1/4:1 with an earlier planet; run the planet fits to test)" :
              res_flag      ? "resonant partner: planet-planet interaction makes the Keplerian model approximate" :
                              "no activity, alias or harmonic match"
    return (; nm, P, mass, pw, fp, notes, verdict)
end

cands = NamedTuple[]
let earlier = Tuple{String,Float64,Float64,Float64}[]
    for k in 1:C_K
        nm = PLANET_NAMES[k]
        m = mass_of(final === nothing ? nothing : final.chain, nm)
        e84 = quantile(ecc_of(final === nothing ? nothing : final.chain, nm), 0.84)
        push!(cands, vet(nm, C_PERIODS[k], m, copy(earlier), nothing))
        push!(earlier, (nm, C_PERIODS[k], m, e84))
    end
    if fit_extra !== nothing && has_col(fit_extra.chain, "$(PLANET_NAMES[fit_extra.k])_P")
        nm = PLANET_NAMES[fit_extra.k]
        P = median(chain_vec(fit_extra.chain, "$(nm)_P"))
        push!(cands, vet(nm, P, mass_of(fit_extra.chain, nm), copy(earlier), fit_extra.chain))
    elseif length(PERIOD_GUESSES) > C_K
        nm = PLANET_NAMES[C_K+1]
        push!(cands, vet(nm, PERIOD_GUESSES[C_K+1], NaN, copy(earlier), nothing))
    end
end

function lnbf_for(c)
    k = findfirst(==(c.nm), PLANET_NAMES)
    k === nothing && return NaN
    if fit_extra !== nothing && k == fit_extra.k && final !== nothing && k == final.k + 1
        return fit_extra.lnZ - final.lnZ
    end
    ia = findfirst(f -> f.k == k, fits); ib = findfirst(f -> f.k == k - 1, fits)
    (ia === nothing || ib === nothing) ? NaN : fits[ia].lnZ - fits[ib].lnZ
end

emit()
emit("CANDIDATE SIGNALS")
emit(rpad("  planet", 9) * rpad("P [d]", 10) * rpad("m sin i [Mjup]", 16) * rpad("ln BF vs one fewer", 20) * "verdict")
emit("  (Letters follow the order of detection in this run, so a new signal may carry a letter already used in the literature; see the notes.)")
for c in cands
    lb = lnbf_for(c)
    emit("  " * rpad(c.nm, 7) * rpad(@sprintf("%.3f", c.P), 10) * rpad(isfinite(c.mass) ? @sprintf("%.3f", c.mass) : "-", 16) *
         rpad(isfinite(lb) ? @sprintf("%+.1f", lb) : "-", 20) * c.verdict)
    for n in c.notes
        emit("           - " * n)
    end
end

emit()
emit("OVERALL")
cand_ok(c) = startswith(c.verdict, "no activity") || startswith(c.verdict, "resonant")
emit("  Signals that pass every check run here (no activity, alias or harmonic match): " *
     join((c.nm for c in cands if cand_ok(c)), ", "))
SUSPECTS = [c for c in cands if !cand_ok(c)]
if !isempty(SUSPECTS)
    emit("  Signals with a reason for doubt: " * join((c.nm * " (" * first(split(c.verdict)) * ")" for c in SUSPECTS), ", "))
end
if abs(hk_corr_sigma(RHO_RES_RHK, N_RES_RHK)) > 3
    emit(@sprintf("  The residual RVs DO correlate with log R'HK (rho = %+.2f); part of what is left is activity, about %.1f m/s rms.", RHO_RES_RHK, ACT_RV))
else
    emit(@sprintf("  The residual RVs do not correlate with log R'HK (rho = %+.2f, ~%.1f sigma), so activity explains at most about %.1f m/s of the %.1f m/s left over.",
                  RHO_RES_RHK, hk_corr_sigma(RHO_RES_RHK, N_RES_RHK), isfinite(ACT_RV) ? ACT_RV : NaN, RES_RMS))
    emit("  The leftover scatter is therefore more likely planet-planet interaction, an unmodelled planet, or instrumental noise than chromospheric activity.")
end
emit("  Limits: R'HK sees chromospheric (plage) activity only; spots and convective blueshift can move the RV without moving R'HK. The residuals come from a")
emit("  harmonic stand-in for the Keplerian fit, and in a resonant system the Keplerian model itself is approximate.")
emit("  These verdicts are screening heuristics, not a detection claim.")

summary_path = joinpath(FIGDIR, "summary_" * replace(TARGET, r"[^A-Za-z0-9._-]" => "_") * ".txt")
try
    open(summary_path, "w") do io
        println(io, "Combined RV + Ca II H&K summary, generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"))
        for l in SUMLINES
            println(io, l)
        end
    end
    dbg("Wrote the summary to ", summary_path)
catch err
    @warn "Could not write the summary file" exception = err
end

end  # if RUN_ACTIVITY

section("Done")
dbg("Total run time: ", round((time() - T_START) / 60; digits = 1), " min")
dbg("Figures saved in: ", FIGDIR)
for f in sort(readdir(FIGDIR))
    (endswith(f, ".png") || endswith(f, ".txt")) && dbg("  ", f)
end
