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
#   From a shell the script restarts itself with --threads=auto (setting THREADS) so the Pigeons chains run side by side. In the VSCode REPL set the
#   Julia extension setting julia.NumThreads to "auto" and restart the REPL; the script prints the thread count in section 0.
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
#   Data    Prints the list of 100 stars and marks the one selected; loads the table
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
#   100 stars, sorted by catalogue name (GJ, then HD, then HIP; numbers in numerical order
#   within each). 25 are hand-picked planet hosts with notes and masses (GJ 876, GJ 581,
#   HD 10180, ...). The other 75 were chosen by a rule, not by science: the best-sampled
#   HARPS stars not already listed (at least 100 nights with an RV, a baseline of at least
#   5 yr). Their notes carry no planet claims, and their masses are class defaults. Counts
#   and baselines were read from the RVBank CSV. Stellar masses are rough (they only turn
#   RV amplitudes into planet masses, and give the radius for the vsini limit): check them
#   before quoting a mass. The note's first letter is read as the spectral class. To add a star, append a line to STAR_LIST (the first field must match
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
#   * The residuals in Part C come from Keplerian shapes (P, e and periastron time of the
#     maximum-posterior draw, amplitudes refitted linearly) when the planet fits ran, and from
#     a harmonic stand-in (fundamental plus first harmonic) when they did not. They are not
#     the Octofitter model itself. In a resonant system the Keplerian model is itself approximate.
#   * R'HK sees chromospheric activity only; spots and convective blueshift can move the
#     RV without moving R'HK. The Part C verdicts are screening heuristics, not detections.
#
# Changelog
#   v1.0.0  2026-10-11  First combined version.
#   v1.0.1  2026-10-11  vet(): rotation-harmonic match alone no longer flags activity (needs R'HK power);
#                       precession-rate note for harmonic candidates; note on letter naming.
#   v1.1.13 2026-10-14  SPLIT_HARPS: the RVs before and after STEP_MJD (the 2015 fibre upgrade) are two HARPS instruments in the planet fit, each with its own
#                       zero point (marginalized) and its own jitter, instead of one series. HD 82943: the planet-removed residuals differed by +4.4 +/- 0.65 m/s
#                       (6.7 sigma) between the eras (C1c, v1.1.12), which the single-series fit had to absorb in its jitter and perhaps in an extra planet.
#                       The Keplerian residuals (A8 seed, C1) now include a per-era constant when the split is on; C1c reports the fitted offset;
#                       jitters are reported per era and pooled (quadrature mean weighted by RV count). SPLIT_HARPS = false gives the v1.1.12 model.
#   v1.1.12 2026-10-14  (a) THREADS: run from a shell, the script restarts itself with --threads=auto (Julia fixes its thread count at start-up; every log
#                       said "Julia was started with only one thread" while Pigeons reported a multi-threaded sampler); in the VSCode REPL it prints how to
#                       set julia.NumThreads instead; the thread count is printed in section 0;
#                       (b) A8 seeds the extra planet from the periodogram of the Keplerian residuals of the fitted planets (EXTRA_SEED = :residual, skipping peaks within
#                       SEED_EXCLUDE of a fitted planet or at its P/2..P/6) and says when the prewhitening peak is a harmonic of a fitted planet (HD 82943: the 109.84 d
#                       peak that became "planet d" is P_b/2); A3 marks the extra planet's prior as provisional;
#                       (c) new A8b: periastron and apastron of every fitted planet and a flag when two orbits overlap in radius outside a low-order resonance
#                       (HD 82943 d at e = 0.61 reaches b's periastron; b and c overlap too, but sit at 2:1 and are not flagged); C3 gives such a planet the verdict
#                       ORBIT-OVERLAP; not a stability test;
#                       (d) new C1c: planet-removed residuals before and after MJD 57170 (offset in nightly means) and each indicator's correlation with them pooled and
#                       within each era, to tell a correlation that holds in both eras from one produced by an offset between them; summarised in OVERALL;
#                       (e) Combined Plot 5 labels the residual scatter "Keplerian" (it said "harmonic" for the Keplerian residuals).
#   v1.1.11 2026-10-13  (a) the peak FAP shown in C3 (and the WEAK flag) is now computed at the candidate's own period, after removing sinusoids
#                       at the fitted periods of the earlier planets; before, it was looked up by planet letter from the prewhitening stage, so with
#                       PERIOD_OVERRIDE planet c showed the FAP of the 183 d peak and planet d that of the 45 d peak;
#                       (b) new section C1b: fixed sinusoids at 1/n yr (n = 1..4) fitted to the planet-removed RV residuals and to log R'HK, with
#                       amplitudes and S/N, to show whether a signal near a seasonal period is carried by both (HD 20794: 183 d sits 0.1 resolution
#                       elements from 0.5 yr, and R'HK peaks at 184 and 366 d); OVERALL names such terms;
#                       (c) the A8 header names the PERIOD_OVERRIDE window instead of the 45 d peak; the broken sentence in OVERALL is fixed.
#   v1.1.10 2026-10-13  (a) C1 residuals come from Keplerian shapes (P, e and periastron time of the maximum-posterior draw of the final fit;
#                       amplitude and phase refitted linearly) instead of two harmonics, which cannot follow an orbit with e ~ 0.4
#                       (HD 82943 b: the 73 d residual peak was P_b/3 left behind by the stand-in, and its BIS correlation was contaminated);
#                       the harmonic stand-in remains when the planet fits are off;
#                       (b) the alias test covers 1/n yr for n = 1..4: 1 yr and 0.5 yr set ALIAS-SUSPECT, 1/3 yr and 0.25 yr only add a note
#                       (flagging them would mark HD 10180 d and GJ 3293 c, whose periods lie within 0.5 resolution elements of 1/3 yr);
#                       (c) the note on a P_planet/n residual peak says what it means for each kind of model.
#   v1.1.9  2026-10-13  (a) the stellar-mass prior is bounded on both sides (+/- 6 sigma): with only a lower bound the sampler stepped the mass to
#                       ~exp(300) in the 1- and 3-planet fits of HD 20794 ("invalid log density Inf"), so those fits failed;
#                       (b) the alias test also covers 0.5 yr (HD 20794 c at 180.9 d sits 0.35 resolution elements from half a year);
#                       (c) PERIOD_OVERRIDE: a hand-set period prior [d] for a named planet, to test a period the periodogram did not pick
#                       (e.g. 85-95 d instead of 180 d, whose first harmonic mimics 90 d); compare ln Z between the two runs;
#                       (d) the Proxima Cen stop message gives the counts below and above the accepted R'HK range and no longer blames "faint M dwarfs";
#                       (e) the class comes from the hand-typed note when table1 shows its placeholder pattern (logg 4.90 +/- 0.01, vsini 2.00 +/- 0.00,
#                       vsinflag 1: GJ 581, GJ 667, GJ 785), also for a one-class difference.
#   v1.1.8  2026-10-13  (a) annual-alias test against a fixed 1-yr period: a signal with |1/P - 1/365.25| < ANNUAL_RES_ELEMENTS/T (T = RV baseline)
#                       is flagged ALIAS-SUSPECT even when no earlier planet is near it (missed GJ 699 b, HD 204961 b/c/d in v1.1.7);
#                       (b) the spectral class from table1 Teff is overridden by the hand-typed note when they differ by more than one class
#                       (GJ 581 and GJ 667 got G and F jitter floors); a one-class difference only prints a warning;
#                       (c) too few R'HK values (Proxima Cen: 9 left) prints the R'HK range and stops cleanly instead of raising an error.
#   v1.1.7  2026-10-13  STAR_LIST grows from 25 to 100 stars and is sorted by catalogue name (so the star numbers change; names still work).
#   v1.1.6  2026-10-13  the candidate peak FAPs (A2 and C3) include a white jitter added to the RV errors; v1.1.5 used the bare errors, which made
#                       tau Ceti's weak peaks look like FAP < 1e-10 (the scatter is 1.4 m/s against 0.23 m/s errors).
#   v1.1.5  2026-10-12  C3 lists the Baluev FAP of each candidate's periodogram peak and marks a peak with FAP > 0.01 as WEAK; a weak peak no
#                       longer counts as "passing every check" (tau Ceti's three candidates, with no significant peak, had).
#   v1.1.4  2026-10-12  RV step correction across the 2015 HARPS fibre upgrade (STEP_MJD, STEP_WINDOW, STEP_MIN_SIGMA). HD 85512 shows a
#                       ~-10 m/s step that the prewhitening read as 2260/1489/1040 d "planets". The offset is measured and removed, and printed.
#   v1.1.3  2026-10-12  prints a version line at the start of every run (v1.1.2 was re-sent several times with added fixes: star-not-found
#                       line, outlier clip, ARGS fix, precession cap, rms comparison; v1.1.3 has all of them).
#   v1.1.2  2026-10-12  command-line star choice ignores the socket path VSCode's REPL puts in ARGS; RV outliers beyond 10 robust sigma are dropped (HD 85512 has two commissioning-night points at 1250 and 2160 m/s); a harmonic/resonance note needs an apsidal-precession offset below 60 deg/yr (tau Ceti -113, HD 1461 -226 were
#                       coincidences); the summary says when the star is not in table1; raw RV scatter in the comparison is now the rms (the robust MAD value is ~1.5x the rms for a planet-dominated
#                       RV and could exceed the residual rms); small velocities print with 2 decimals; residual-peak table labels P_planet/n harmonics; precession note only for a candidate shorter than the
#                       planet it is compared with (c at 2 x P_b gave a meaningless -897 deg/yr); other indicators that
#                       correlate with the residuals (>3 sigma) are reported in OVERALL; vet(): without a planet fit the harmonic (eccentricity) amplitude test cannot run, so a
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
const TARGET_INDEX      = 47      # which star in STAR_LIST to use (47 = GJ 876; the list is alphabetical, so look the number up in the printed list)
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
const EXTRA_SEED        = :residual # where the extra planet's period prior is centred: :residual = strongest peak of the periodogram of the RV residuals of the N_PLANETS fit (Keplerian shapes), skipping peaks that sit at a fitted planet or its P/2..P/6; :prewhitening = the sinusoid-prewhitening peak (v1.1.11 and earlier)
const SEED_EXCLUDE      = 0.15    # a residual peak within this fraction of a fitted planet's period is skipped as a seed for the extra planet (a second planet that close would need a very special orbit)
const PERIOD_WINDOW     = 0.10    # period prior = peak period +/- this fraction
const PERIOD_OVERRIDE   = Dict{String,Tuple{Float64,Float64}}()   # hand-set period prior [d] per planet letter, e.g. Dict("c" => (85.0, 95.0)); empty = use the periodogram peak +/- PERIOD_WINDOW
const MIN_PERIOD        = 1.5     # [d] ignore shorter periodogram peaks (1 d peaks are daily-sampling aliases)
const RV_BIN_NIGHTLY    = :auto   # RVs: :auto (bin only if > MAX_POINTS), true, or false
const MAX_POINTS        = 1000
const CUSTOM_STAR_MASS  = 1.0     # [Msun] used only for a star that is not in STAR_LIST
const N_ROUNDS          = 10      # Pigeons rounds; each extra round doubles the work (9 is a quick look)
const OUTLIER_SIGMA     = 10.0    # drop RVs further than this many robust sigma (MAD) from the median; 0 = keep everything
const STEP_MJD          = 57170.0 # HARPS fibre upgrade (the commissioning RVs of HD 85512 sit here): fit and remove an RV offset across it; 0 = off
const SPLIT_HARPS       = true    # true: the RVs before and after STEP_MJD are two HARPS instruments in the planet fit (own zero point and jitter each); false: one series
const SPLIT_MIN         = 10      # the split needs at least this many RVs (after any nightly binning) on each side of STEP_MJD
const STEP_WINDOW       = 400.0   # [d] RVs within this many days on each side of STEP_MJD give the offset
const STEP_MIN_SIGMA    = 3.0     # apply the offset only if it is at least this many sigma from zero (and >= 10 RVs on each side)
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
const THREADS           = "auto"  # Julia threads for the Pigeons chains: "auto" (one per CPU thread), a number such as "4", or "1" for no restart. Julia fixes its thread count at start-up, so from a shell the script restarts itself with --threads=THREADS; in the VSCode REPL it cannot (it prints how to set julia.NumThreads instead)
const T_START           = time()

# ------------------------------------------------------------------ the list
# (RVBank name, common name, star mass [Msun], mass uncertainty, number of RVs, baseline [yr], note)
const STAR_LIST = [
    ("GJ17",    "GJ 17",                1.0,      0.1,     3671, 18.2, "G; 382 nights (added by RV count; mass is a class default)"),
    ("GJ29",    "GJ 29",                1.0,      0.1,      253, 13.9, "G; 139 nights (added by RV count; mass is a class default)"),
    ("GJ54.1",  "YZ Ceti",              0.13,   0.013,    334, 14.8, "M4.5; small planets"),
    ("GJ87",    "GJ 87",                0.4,    0.04,     166, 18.0, "M; 156 nights (added by RV count; mass is a class default)"),
    ("GJ138",   "GJ 138",               1.0,      0.1,      401, 17.9, "G; 149 nights (added by RV count; mass is a class default)"),
    ("GJ150.0", "GJ 150.0",             0.75,   0.075,   1117, 13.4, "K; 150 nights (added by RV count; mass is a class default)"),
    ("GJ163",   "GJ 163",               0.4,    0.04,     207, 18.2, "M3; multiple low-mass planets"),
    ("GJ191",   "GJ 191",               0.4,    0.04,     378, 18.0, "M; 217 nights (added by RV count; mass is a class default)"),
    ("GJ219.0", "GJ 219.0",             1.0,      0.1,     6158, 16.4, "G; 184 nights (added by RV count; mass is a class default)"),
    ("GJ253",   "GJ 253",               1.0,      0.1,      227, 13.7, "G; 221 nights (added by RV count; mass is a class default)"),
    ("GJ273",   "Luyten's Star",        0.29,   0.03,     318, 12.8, "M3.5; two small planets"),
    ("GJ275",   "GJ 275",               1.0,      0.1,      531, 14.5, "G; 164 nights (added by RV count; mass is a class default)"),
    ("GJ294A",  "GJ 294A",              1.0,      0.1,      952, 15.5, "G; 299 nights (added by RV count; mass is a class default)"),
    ("GJ302",   "GJ 302",               0.75,   0.075,    728, 15.3, "K; 278 nights (added by RV count; mass is a class default)"),
    ("GJ309",   "GJ 309",               0.75,   0.075,    531, 15.3, "K; 168 nights (added by RV count; mass is a class default)"),
    ("GJ312",   "GJ 312",               1.0,      0.1,      230, 15.3, "G; 165 nights (added by RV count; mass is a class default)"),
    ("GJ367",   "GJ 367",               0.4,    0.04,     332, 17.6, "M; 158 nights (added by RV count; mass is a class default)"),
    ("GJ393",   "GJ 393",               0.4,    0.04,     181, 13.5, "M; 175 nights (added by RV count; mass is a class default)"),
    ("GJ436",   "GJ 436",               0.45,   0.045,    192, 14.2, "M2.5; hot Neptune (2.6 d)"),
    ("GJ447",   "GJ 447",               0.4,    0.04,     307, 15.6, "M; 247 nights (added by RV count; mass is a class default)"),
    ("GJ472",   "GJ 472",               0.75,   0.075,    864, 14.2, "K; 443 nights (added by RV count; mass is a class default)"),
    ("GJ506",   "GJ 506",               1.0,      0.1,     1242, 17.5, "G; 239 nights (added by RV count; mass is a class default)"),
    ("GJ514",   "GJ 514",               0.4,    0.04,     162, 11.9, "M; 154 nights (added by RV count; mass is a class default)"),
    ("GJ536",   "GJ 536",               0.4,    0.04,     196, 11.9, "M; 190 nights (added by RV count; mass is a class default)"),
    ("GJ551",   "Proxima Cen",          0.12,   0.012,    414, 15.3, "M5.5; nearest star, planets b and d"),
    ("GJ559B",  "GJ 559B",              0.75,   0.075,  18283, 14.0, "K; 367 nights (added by RV count; mass is a class default)"),
    ("GJ581",   "GJ 581",               0.3,    0.03,     250,  8.0, "M3; several low-mass planets (some signals debated)"),
    ("GJ613.0", "GJ 613.0",             0.75,   0.075,    449, 14.7, "K; 273 nights (added by RV count; mass is a class default)"),
    ("GJ616",   "GJ 616",               1.0,      0.1,     5500, 15.3, "G; 211 nights (added by RV count; mass is a class default)"),
    ("GJ628",   "GJ 628",               0.4,    0.04,     193, 11.3, "M; 184 nights (added by RV count; mass is a class default)"),
    ("GJ654",   "GJ 654",               0.4,    0.04,     195,  8.5, "M; 183 nights (added by RV count; mass is a class default)"),
    ("GJ656",   "GJ 656",               0.75,   0.075,    835, 13.2, "K; 432 nights (added by RV count; mass is a class default)"),
    ("GJ667",   "GJ 667 C",             0.33,   0.03,     272, 15.3, "M1.5; super-Earths, in a triple-star system"),
    ("GJ674",   "GJ 674",               0.35,   0.035,    256, 15.4, "M2.5; Neptune-mass planet at 4.7 d"),
    ("GJ691",   "GJ 691",               1.0,      0.1,     3012, 11.2, "G; 165 nights (added by RV count; mass is a class default)"),
    ("GJ693",   "GJ 693",               0.4,    0.04,     183, 12.3, "M; 177 nights (added by RV count; mass is a class default)"),
    ("GJ699",   "Barnard's Star",       0.16,   0.016,    335, 12.5, "M4; sub-Earth planet candidates"),
    ("GJ701",   "GJ 701",               0.4,    0.04,     154, 12.3, "M; 149 nights (added by RV count; mass is a class default)"),
    ("GJ752",   "GJ 752",               0.4,    0.04,     149, 12.3, "M; 146 nights (added by RV count; mass is a class default)"),
    ("GJ754",   "GJ 754",               0.4,    0.04,     197, 12.3, "M; 185 nights (added by RV count; mass is a class default)"),
    ("GJ776",   "GJ 776",               1.0,      0.1,      659, 13.8, "G; 261 nights (added by RV count; mass is a class default)"),
    ("GJ780",   "GJ 780",               1.0,      0.1,     5720, 12.0, "G; 441 nights (added by RV count; mass is a class default)"),
    ("GJ785",   "HD 192310 (GJ 785)",   0.78,   0.08,    1821, 12.9, "K3; Neptune-mass planet plus a candidate; many RVs (binned per night)"),
    ("GJ812.1", "GJ 812.1",             1.0,      0.1,     1175, 18.0, "G; 181 nights (added by RV count; mass is a class default)"),
    ("GJ838",   "GJ 838",               1.0,      0.1,      487, 17.9, "G; 158 nights (added by RV count; mass is a class default)"),
    ("GJ845",   "GJ 845",               0.75,   0.075,   4634, 18.1, "K; 205 nights (added by RV count; mass is a class default)"),
    ("GJ876",   "GJ 876",               0.346,  0.007,    270, 15.7, "M4; two giant planets in a 2:1 resonance (61 d, 30 d) plus smaller ones"),
    ("GJ880",   "GJ 880",               0.4,    0.04,     140, 12.8, "M; 140 nights (added by RV count; mass is a class default)"),
    ("GJ887",   "GJ 887",               0.4,    0.04,     715, 15.5, "M; 184 nights (added by RV count; mass is a class default)"),
    ("GJ1061",  "GJ 1061",              0.12,   0.012,    184, 15.3, "M5.5; three small planets"),
    ("GJ1214",  "GJ 1214",              0.4,    0.04,     166, 10.2, "M; 153 nights (added by RV count; mass is a class default)"),
    ("GJ2046",  "GJ 2046",              0.75,   0.075,    478, 15.4, "K; 260 nights (added by RV count; mass is a class default)"),
    ("GJ3138",  "GJ 3138",              0.4,    0.04,     206, 12.0, "M; 186 nights (added by RV count; mass is a class default)"),
    ("GJ3293",  "GJ 3293",              0.42,   0.04,     217,  6.3, "M2.5; several planets"),
    ("GJ3323",  "GJ 3323",              0.4,    0.04,     157, 11.9, "M; 150 nights (added by RV count; mass is a class default)"),
    ("GJ3436",  "GJ 3436",              1.0,      0.1,      237, 11.4, "G; 173 nights (added by RV count; mass is a class default)"),
    ("GJ4291",  "GJ 4291",              0.75,   0.075,    371, 12.8, "K; 330 nights (added by RV count; mass is a class default)"),
    ("GJ9008",  "GJ 9008",              1.0,      0.1,      212, 13.9, "G; 151 nights (added by RV count; mass is a class default)"),
    ("GJ9121",  "GJ 9121",              1.0,      0.1,      214, 13.9, "G; 144 nights (added by RV count; mass is a class default)"),
    ("GJ9191",  "GJ 9191",              0.75,   0.075,    277, 13.8, "K; 274 nights (added by RV count; mass is a class default)"),
    ("GJ9592",  "GJ 9592",              0.4,    0.04,     150, 12.3, "M; 145 nights (added by RV count; mass is a class default)"),
    ("GJ9774",  "GJ 9774",              1.0,      0.1,      257, 18.0, "G; 148 nights (added by RV count; mass is a class default)"),
    ("HD1461",  "HD 1461",              1.03,   0.1,      487, 17.8, "G0; super-Earth"),
    ("HD4308",  "HD 4308",              1.0,      0.1,      359, 17.8, "G; 143 nights (added by RV count; mass is a class default)"),
    ("HD10180", "HD 10180",             1.06,   0.1,      336, 13.8, "G1; rich system of several planets"),
    ("HD10700", "tau Ceti",             0.78,   0.08,   11632, 18.1, "G8; very quiet star, weak signals; very many RVs (binned per night)"),
    ("HD13808", "HD 13808",             0.75,   0.075,    253, 12.2, "K; 231 nights (added by RV count; mass is a class default)"),
    ("HD20003", "HD 20003",             1.0,      0.1,      203, 13.7, "G; 198 nights (added by RV count; mass is a class default)"),
    ("HD20781", "HD 20781",             0.75,   0.075,    237, 13.7, "K; 227 nights (added by RV count; mass is a class default)"),
    ("HD20794", "HD 20794 (82 Eri)",    0.81,   0.08,    6824, 18.2, "G8; super-Earth candidates; very many RVs (binned per night)"),
    ("HD21693", "HD 21693",             1.0,      0.1,      221, 13.9, "G; 219 nights (added by RV count; mass is a class default)"),
    ("HD31128", "HD 31128",             1.0,      0.1,      261, 11.4, "G; 197 nights (added by RV count; mass is a class default)"),
    ("HD31527", "HD 31527",             1.0,      0.1,      268, 13.8, "G; 255 nights (added by RV count; mass is a class default)"),
    ("HD32564", "HD 32564",             1.0,      0.1,      205,  6.3, "G; 191 nights (added by RV count; mass is a class default)"),
    ("HD39091", "HD 39091",             1.0,      0.1,      553, 16.9, "G; 209 nights (added by RV count; mass is a class default)"),
    ("HD41248", "HD 41248",             1.0,      0.1,      231, 14.2, "G; 164 nights (added by RV count; mass is a class default)"),
    ("HD42581", "HD 42581",             0.4,    0.04,     200, 12.8, "M; 193 nights (added by RV count; mass is a class default)"),
    ("HD45184", "HD 45184",             1.03,   0.1,      355, 17.9, "G2; two small planets"),
    ("HD47186", "HD 47186",             1.0,      0.1,      171, 13.8, "G; 145 nights (added by RV count; mass is a class default)"),
    ("HD71835", "HD 71835",             1.0,      0.1,      153, 13.4, "G; 152 nights (added by RV count; mass is a class default)"),
    ("HD82943", "HD 82943",             1.18,   0.12,     255, 13.4, "G; two giant planets in 2:1 resonance"),
    ("HD85512", "HD 85512",             0.69,   0.07,    1184, 17.9, "K6; low-mass planet candidate"),
    ("HD87838", "HD 87838",             1.0,      0.1,      228, 13.5, "G; 158 nights (added by RV count; mass is a class default)"),
    ("HD93385", "HD 93385",             1.0,      0.1,      249, 13.4, "G; 236 nights (added by RV count; mass is a class default)"),
    ("HD95456", "HD 95456",             1.0,      0.1,      302, 13.5, "G; 149 nights (added by RV count; mass is a class default)"),
    ("HD96700", "HD 96700",             0.9,    0.09,     366, 13.5, "G0; several small planets"),
    ("HD114613","HD 114613",            1.0,      0.1,      732, 15.2, "G; 157 nights (added by RV count; mass is a class default)"),
    ("HD134060","HD 134060",            1.1,    0.11,     339, 13.2, "G0; two small planets"),
    ("HD134606","HD 134606",            1.0,      0.1,      290, 12.8, "G3; three small planets"),
    ("HD136352","HD 136352 (nu2 Lup)",  0.87,   0.09,     677, 13.2, "G; three small planets"),
    ("HD154088","HD 154088",            0.75,   0.075,    280, 11.3, "K; 188 nights (added by RV count; mass is a class default)"),
    ("HD161098","HD 161098",            1.0,      0.1,      215, 15.5, "G; 170 nights (added by RV count; mass is a class default)"),
    ("HD176986","HD 176986",            0.75,   0.075,    243, 17.3, "K; 181 nights (added by RV count; mass is a class default)"),
    ("HD181433","HD 181433",            0.75,   0.075,    201, 13.8, "K; 189 nights (added by RV count; mass is a class default)"),
    ("HD204961","GJ 832 (HD 204961)",   0.45,   0.045,    183, 16.1, "M1.5; cold giant planet plus a candidate"),
    ("HD220507","HD 220507",            1.0,      0.1,      210, 14.0, "G; 205 nights (added by RV count; mass is a class default)"),
    ("HD223171","HD 223171",            1.0,      0.1,      267, 15.0, "G; 145 nights (added by RV count; mass is a class default)"),
    ("HIP9786", "HIP 9786",             0.4,    0.04,     182,  7.8, "M; 176 nights (added by RV count; mass is a class default)"),
    ("HIP103039","HIP 103039",           0.4,    0.04,     155, 12.3, "M; 147 nights (added by RV count; mass is a class default)"),
    ("HIP112414","HIP 112414",           1.0,      0.1,      267, 13.5, "G; 145 nights (added by RV count; mass is a class default)"),
]

dbg(args...) = println("[DEBUG ", Dates.format(now(), "HH:MM:SS"), "] ", args...)
section(title) = println("\n", "="^78, "\n", title, "\n", "="^78)

# ------------------------------------------------------------------ choose the star
normname(s) = lowercase(replace(string(s), r"[\s_\-]" => ""))

function resolve_target()
    # VSCode's REPL puts its own socket path in ARGS; only a plain star number or name counts as a command-line choice
    cli = [a for a in ARGS if !occursin(r"[/\\]", a) && !startswith(a, "vsc-")]
    spec = TARGET_OVERRIDE != "" ? TARGET_OVERRIDE :
           !isempty(cli)         ? cli[1] : TARGET_INDEX
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

# ------------------------------------------------------------------ threads
# Julia fixes its number of threads at start-up, so a running script cannot add any. Pigeons runs its parallel-tempering chains on threads; with one
# thread they run one after another (every earlier log said "Julia was started with only one thread"). From a shell (`julia OctofitterRVActivity.jl ...`)
# the script therefore starts a fresh Julia with --threads=THREADS and runs itself there. In the VSCode REPL (or any interactive session) that would
# close your REPL, so it only prints a note. Nothing happens if you already set threads (--threads, -t or JULIA_NUM_THREADS) or THREADS = "1".
if Threads.nthreads() == 1 && THREADS != "1" && !haskey(ENV, "JULIA_NUM_THREADS") && get(ENV, "OCTORVACT_CHILD", "") == ""
    if !isinteractive() && PROGRAM_FILE != "" && isfile(abspath(PROGRAM_FILE))
        println("Julia has 1 thread; restarting with --threads=", THREADS, " (set THREADS = \"1\" at the top of the script to stop this).")
        _code = try
            withenv("OCTORVACT_CHILD" => "1") do
                run(ignorestatus(`$(Base.julia_cmd()) --threads=$(THREADS) $(abspath(PROGRAM_FILE)) $(ARGS)`)).exitcode
            end
        catch err
            @warn "Could not restart Julia with threads; continuing with one thread" exception = err
            -1
        end
        _code >= 0 && exit(_code)
    else
        println("\nNOTE: this Julia session has 1 thread, so the Pigeons chains run one after another. To use more: in VSCode set the Julia extension setting")
        println("      julia.NumThreads to \"auto\" (or a number) and restart the Julia REPL; in a shell run  julia --threads=auto OctofitterRVActivity.jl")
        println("      (or set JULIA_NUM_THREADS=auto). Check with  Threads.nthreads()  at the julia> prompt.")
    end
end

println("OctofitterRVActivity.jl v1.1.13 (RV outlier clip: ", OUTLIER_SIGMA > 0 ? "on, " * string(OUTLIER_SIGMA) * " sigma" : "off",
        "; RV step at MJD ", STEP_MJD > 0 ? string(STEP_MJD) : "off", ")")
section("HARPS target list")
const _SEL_PAIR = resolve_target()
const SEL_IDX = _SEL_PAIR[1]
const SEL     = _SEL_PAIR[2]
println(rpad("#", 5), rpad("RVBank name", 12), rpad("Star", 21), lpad("RVs", 6), lpad("yr", 6),
        lpad("Msun", 7), "  Notes")
for (i, s) in enumerate(STAR_LIST)
    mark = i == SEL_IDX ? "->" : "  "
    println(mark, rpad(string(i), 3), rpad(s[1], 12), rpad(s[2], 21), lpad(string(s[5]), 6),
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
dbg("Threads: ", Threads.nthreads(), "  (this computer reports ", Sys.CPU_THREADS, " CPU threads)",
    Threads.nthreads() == 1 ? "  -> the Pigeons chains run one after another" : "  -> the Pigeons chains can run side by side")

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

"""
Spectral class used for the jitter floor. The note wins when table1 shows its placeholder pattern. Otherwise the Teff class wins unless the hand-typed note is a valid class and differs from it by
more than one class (a Teff of 5900 K for the M dwarf GJ 581 is a table1 mismatch, not a G star). Returns (class, message or "").
"""
function hk_resolve_class(teff, note, p = nothing)
    ct = hk_class_from_teff(teff, note)
    order = ["M", "K", "G", "F", "A"]
    # table1 placeholder pattern (GJ 581, GJ 667, GJ 785): logg 4.90 +/- 0.01, vsini 2.00 +/- 0.00 with vsinflag 1 means Teff, logg and vsini are defaults
    if p !== nothing && note in order && ct != note && isfinite(p.logg) && abs(p.logg - 4.90) < 0.011 &&
       isfinite(p.vsini) && abs(p.vsini - 2.0) < 1e-6 && p.vflag == 1
        return (note, "table1 shows its placeholder pattern (logg 4.90, vsini 2.00, vsinflag 1), so Teff = $(round(Int, teff)) K is not trusted: the hand-typed note ($note) is used")
    end
    (note in order && ct in order && ct != note) || return (ct, "")
    d = abs(findfirst(==(ct), order) - findfirst(==(note), order))
    d > 1 && return (note, "Teff = $(round(Int, teff)) K gives class $ct, but the hand-typed note says $note (differs by $d classes): the note is used, and the table1 Teff is probably for a different star or a blend")
    return (ct, "the hand-typed note says $note; Teff is used (differs by one class: check the jitter floor if it matters)")
end

# Velocity in m/s: two decimals below 10, one above.
hk_ms(x) = !isfinite(x) ? "n/a" : (abs(x) < 10 ? @sprintf("%.2f", x) : @sprintf("%.1f", x))
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
const SPTYPE, SPTYPE_MSG = hk_resolve_class(STAR_P === nothing ? NaN : STAR_P.teff, SPTYPE_NOTE, STAR_P)
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
    println("  spectral class used: ", SPTYPE, isempty(SPTYPE_MSG) ? "   (from Teff)" : "")
    isempty(SPTYPE_MSG) || println("  !! ", SPTYPE_MSG)
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

# Outlier RVs (e.g. the commissioning nights around the 2015 HARPS fibre upgrade, which give 1000+ m/s for HD 85512) would
# dominate every scatter, correlation and periodogram. Their RV is set to NaN; the rows and their R'HK stay.
if OUTLIER_SIGMA > 0
    let rv = raw.rv, ok = findall(isfinite, raw.rv)
        if length(ok) >= 20
            med = median(rv[ok]); sc = max(1.4826 * median(abs.(rv[ok] .- med)), 1.0)
            bad = [i for i in ok if abs(rv[i] - med) > OUTLIER_SIGMA * sc]
            if !isempty(bad)
                dbg("Removed ", length(bad), " RV outlier(s) beyond ", OUTLIER_SIGMA, " robust sigma (", round(OUTLIER_SIGMA * sc; digits = 1),
                    " m/s from the median ", round(med; digits = 1), "); MJD ",
                    join(round.(raw.bjd[bad] .- 2400000.5; digits = 1), ", "), "; RV ", join(round.(rv[bad]; digits = 1), ", "), " m/s")
                rv[bad] .= NaN
            end
        end
    end
end

# The HARPS fibre upgrade (2015) moved the RVs of some stars by several m/s. A periodogram reads a step as a long-period
# planet, so the offset is measured from the RVs just before and after STEP_MJD and removed from the later ones.
if STEP_MJD > 0
    let rv = raw.rv, t = raw.bjd .- 2400000.5
        before = findall(i -> isfinite(rv[i]) && isfinite(t[i]) && STEP_MJD - STEP_WINDOW <= t[i] < STEP_MJD, eachindex(rv))
        after  = findall(i -> isfinite(rv[i]) && isfinite(t[i]) && STEP_MJD <= t[i] <= STEP_MJD + STEP_WINDOW, eachindex(rv))
        if length(before) >= 10 && length(after) >= 10
            mb, ma = median(rv[before]), median(rv[after])
            eb = 1.2533 * 1.4826 * median(abs.(rv[before] .- mb)) / sqrt(length(before))
            ea = 1.2533 * 1.4826 * median(abs.(rv[after]  .- ma)) / sqrt(length(after))
            off = ma - mb; err = sqrt(eb^2 + ea^2)
            dbg("RV step at MJD ", STEP_MJD, ": median ", round(mb; digits = 2), " m/s before (", length(before), " RVs) and ",
                round(ma; digits = 2), " m/s after (", length(after), " RVs) within +/-", STEP_WINDOW, " d  ->  offset ",
                round(off; digits = 2), " +/- ", round(err; digits = 2), " m/s")
            if abs(off) >= STEP_MIN_SIGMA * max(err, 1e-9)
                for i in eachindex(rv)
                    isfinite(rv[i]) && isfinite(t[i]) && t[i] >= STEP_MJD && (rv[i] -= off)
                end
                dbg("Removed the offset from all RVs on or after MJD ", STEP_MJD, ". Set STEP_MJD = 0 to keep the original RVs.")
            else
                dbg("The offset is under ", STEP_MIN_SIGMA, " sigma: not applied.")
            end
        else
            dbg("RV step at MJD ", STEP_MJD, ": fewer than 10 RVs within +/-", STEP_WINDOW, " d on one side (", length(before), " before, ",
                length(after), " after): no correction.")
        end
    end
end


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
const N_PRE    = STEP_MJD > 0 ? count(t -> t < STEP_MJD, rv_data.epoch) : 0     # RVs before / on or after the fibre upgrade
const N_POST   = STEP_MJD > 0 ? count(t -> t >= STEP_MJD, rv_data.epoch) : 0
const SPLIT_ON = SPLIT_HARPS && STEP_MJD > 0 && N_PRE >= SPLIT_MIN && N_POST >= SPLIT_MIN

dbg("RVs in the catalogue for this star: ", length(rv_raw.epoch), "  (usable: ", length(rv_clean.epoch), ")")
dbg("RVs before / after MJD ", STEP_MJD, ": ", N_PRE, " / ", N_POST, "  ->  ",
    SPLIT_ON ? "the planet fit uses two HARPS instruments (own zero point and jitter each)" :
    SPLIT_HARPS && STEP_MJD > 0 ? "fewer than $SPLIT_MIN on one side: one HARPS series" : "one HARPS series (SPLIT_HARPS off)")
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
    return per, pow, per[best], LombScargle.fap(pg, pow[best])   # Baluev false-alarm probability of the strongest peak
end

"Find `n` periods: strongest peak, subtract a best-fit sinusoid there, repeat on the residuals."
function find_periods(t, y, σ, n)
    resid = copy(y)
    stages = NamedTuple[]
    for k in 1:n
        per, pow, P, fap = periodogram_stage(t, resid, σ)
        push!(stages, (; per, pow, P, fap, r = copy(resid)))
        resid = resid .- LombScargle.model(t, resid, σ, 1 / P)   # the model includes the mean, which is fine
    end
    # The Baluev FAP assumes the RV errors are the whole noise. HARPS RVs of a quiet star scatter several times more than their errors
    # (stellar jitter, instrument), which would make every peak look overwhelmingly significant. So the FAP below is computed with a
    # white jitter added in quadrature, taken from what is left after all the sinusoids: jitter^2 = var(left over) - mean(error^2).
    jit = sqrt(max(var(resid) - mean(σ .^ 2), 0.0))
    σe = sqrt.(σ .^ 2 .+ jit^2)
    return [(; s.per, s.pow, s.P, fap = stage_fap(t, s.r, σe, s.P, s.fap), jit, s.r) for s in stages]
end

"FAP of the peak at period P, with the (jitter-inflated) errors σe; falls back to the plain FAP if that fails."
function stage_fap(t, r, σe, P, fallback)
    try
        pg = lombscargle(t, r, σe; minimum_frequency = 1 / MAX_PERIOD, maximum_frequency = 1 / MIN_PERIOD,
                         samples_per_peak = 10, fast = false)
        per, pow = LombScargle.periodpower(pg)
        i = argmin(abs.(per .- P))
        lo = max(i - 3, 1); hi = min(i + 3, length(pow))
        return LombScargle.fap(pg, maximum(pow[lo:hi]))
    catch
        return fallback
    end
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

dbg("Peak FAPs below include a white jitter of ", round(stages[1].jit; digits = 2),
    " m/s added in quadrature to the RV errors (the scatter left after the sinusoids exceeds the errors; plain Baluev FAP would be far too optimistic)")
for (k, s) in enumerate(stages)
    dbg("Planet ", PLANET_NAMES[k], k > N_PLANETS ? " (extra-planet test)" : "", ": strongest peak at P = ",
        round(s.P; digits = 3), " d (power ", round(maximum(s.pow); digits = 3), ", FAP ",
        s.fap < 1e-10 ? "<1e-10" : @sprintf("%.1e", s.fap), ")")
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


# ------------------------------------------------------------------ helpers shared by Part A (extra-planet seed, orbit geometry) and Part C
"True anomaly [rad] at times t for period P, eccentricity e and periastron time tp (Newton solution of Kepler's equation)."
function true_anomaly(t, P, e, tp)
    M = mod.(2π .* (t .- tp) ./ P, 2π)
    E = e < 0.8 ? copy(M) : fill(float(π), length(M))
    for _ in 1:60
        E .-= (E .- e .* sin.(E) .- M) ./ (1 .- e .* cos.(E))
    end
    return 2 .* atan.(sqrt(1 + e) .* sin.(E ./ 2), sqrt(1 - e) .* cos.(E ./ 2))
end

"""
Keplerian stand-in: for each planet the shape (P, e, periastron time) is fixed, and the RV is a cos(nu) + b sin(nu) + const, so the amplitude and
the argument of periastron come out of a weighted linear fit (this avoids any convention for omega). Returns (residual, model).
"""
const ERA_STEP = Ref(NaN)   # offset between the eras fitted by the last keplerian_fit (NaN: no per-era constant)
function keplerian_fit(t, y, σ, shapes)
    cols = Vector{Float64}[ones(length(t))]
    # the planet fit has a zero point per era (SPLIT_HARPS), so this stand-in gets one too
    era = SPLIT_ON && any(x -> x < STEP_MJD, t) && any(x -> x >= STEP_MJD, t)
    era && push!(cols, Float64.(t .>= STEP_MJD))
    for (P, e, tp) in shapes
        ν = true_anomaly(t, P, e, tp)
        push!(cols, cos.(ν)); push!(cols, sin.(ν))
    end
    X = reduce(hcat, cols)
    w = 1 ./ σ
    β = (X .* w) \ (y .* w)
    mdl = X * β
    ERA_STEP[] = era ? β[2] : NaN
    return y .- mdl, mdl
end

"(P, e, periastron time) of every planet at the maximum-posterior draw of `fit`, or nothing if the chain lacks a column."
function kepler_shapes(fit, K)
    (fit === nothing || !has_col(fit.chain, "logpost")) && return nothing
    all(k -> all(c -> has_col(fit.chain, "$(PLANET_NAMES[k])_$c"), ("P", "e", "tp")), 1:K) || return nothing
    i = argmax(chain_vec(fit.chain, "logpost"))
    return [(chain_vec(fit.chain, "$(PLANET_NAMES[k])_P")[i], chain_vec(fit.chain, "$(PLANET_NAMES[k])_e")[i],
             chain_vec(fit.chain, "$(PLANET_NAMES[k])_tp")[i]) for k in 1:K]
end

"Labels such as \"P_b/2\" for every planet period in `periods` (planet k is PLANET_NAMES[k]) that P lies within `tol` of when divided by 2..6."
function harmonic_of(P, periods; tol = 0.03)
    out = String[]
    for (k, Pk) in enumerate(periods), n in 2:6
        abs(P - Pk / n) / P < tol && push!(out, "P_$(PLANET_NAMES[k])/$n")
    end
    return out
end

"The low-order resonance (as \"2:1\") a period ratio is within `tol` of, or \"\"."
function near_resonance(P1, P2; tol = 0.03)
    r = max(P1, P2) / min(P1, P2)
    for (a, b) in ((2, 1), (3, 1), (4, 1), (3, 2), (5, 2), (5, 3), (4, 3))
        abs(r - a / b) / (a / b) < tol && return "$a:$b"
    end
    return ""
end

fap_txt(f) = !isfinite(f) ? "n/a" : (f < 1e-10 ? "<1e-10" : @sprintf("%.1e", f))

"""
Where to centre the extra planet's period prior. Looks at the periodogram of the RV residuals of the `K`-planet `fit` (Keplerian shapes of the
maximum-posterior draw, amplitudes refitted) and takes the strongest local maximum that is neither within SEED_EXCLUDE of a fitted planet's period nor
at P/2..P/6 of one. Returns (seed or nothing, lines): `lines` is the printed account of the strongest peaks and why each was kept or skipped.
"""
function residual_seed(fit, K)
    shapes = kepler_shapes(fit, K)
    shapes === nothing && return (nothing, ["(the Keplerian shapes of the $(K)-planet fit are not available, so the prewhitening peak is used)"])
    t = Float64.(collect(rv_data.epoch)); y = Float64.(collect(rv_data.rv)); s = Float64.(collect(rv_data.σ_rv))
    res, _ = keplerian_fit(t, y, s, shapes)
    jit = sqrt(max(var(res) - mean(s .^ 2), 0.0))
    σe = sqrt.(s .^ 2 .+ jit^2)
    pg = lombscargle(t, res, σe; minimum_frequency = 1 / MAX_PERIOD, maximum_frequency = 1 / MIN_PERIOD, samples_per_peak = 10, fast = false)
    per, pow = LombScargle.periodpower(pg)
    peaks = [i for i in 2:length(pow)-1 if pow[i] > pow[i-1] && pow[i] >= pow[i+1]]
    order = peaks[sortperm(pow[peaks]; rev = true)]
    fitted = [p[1] for p in shapes]
    lines = String[]
    seed = nothing
    for i in order
        P = per[i]
        why = ""
        for (k, Pk) in enumerate(fitted)
            abs(P - Pk) / Pk < SEED_EXCLUDE && (why = @sprintf("within %.0f %% of planet %s (%.2f d)", 100 * abs(P - Pk) / Pk, PLANET_NAMES[k], Pk))
        end
        h = harmonic_of(P, fitted)
        isempty(h) || (why = "at " * join(h, ", ") * " of a fitted planet")
        fp = stage_fap(t, res, σe, P, NaN)
        tag = why != "" ? "skipped: " * why : (seed === nothing ? "<- seed" : "")
        push!(lines, @sprintf("    P = %9.2f d   power %.3f   FAP %-8s %s", P, pow[i], fap_txt(fp), tag))
        why == "" && seed === nothing && (seed = P)
        length(lines) >= 6 && break
    end
    return (seed, lines)
end

"""
Orbit geometry from the posterior of `chain` for planets 1..K: period, semi-major axis a [AU] from Kepler's third law with the star's mass (the planet
masses are neglected), eccentricity (median and 16th/84th percentiles), periastron q = a(1-e) and apastron Q = a(1+e) [AU] at the median e.
"""
function orbit_table(chain, K)
    Ms = has_col(chain, "A_mass") ? median(chain_vec(chain, "A_mass")) : M_STAR
    out = NamedTuple[]
    for k in 1:K
        nm = PLANET_NAMES[k]
        (has_col(chain, "$(nm)_P") && has_col(chain, "$(nm)_e")) || continue
        P = median(chain_vec(chain, "$(nm)_P")); ev = chain_vec(chain, "$(nm)_e")
        e = median(ev); e16 = quantile(ev, 0.16); e84 = quantile(ev, 0.84)
        a = cbrt(Ms * (P / 365.25)^2)
        push!(out, (; nm, P, a, e, e16, e84, q = a * (1 - e), Q = a * (1 + e)))
    end
    return out
end

"Pairs whose radial ranges overlap (apastron of the inner planet beyond the periastron of the outer one). `robust`: still true at the 16th-percentile eccentricities."
function orbit_crossings(tab)
    out = NamedTuple[]
    for x in tab, y in tab
        x.a < y.a || continue                       # x is the inner planet
        x.Q > y.q || continue
        push!(out, (; inner = x.nm, outer = y.nm, Q = x.Q, q = y.q, ratio = y.P / x.P,
                    resonance = near_resonance(x.P, y.P), robust = x.a * (1 + x.e16) > y.a * (1 - y.e16)))
    end
    return out
end

const CROSS_FLAG = Dict{String,String}()    # later-detected planet letter => reason (filled in A8b, read by vet() in C3)
const SEED_NOTES = String[]                 # how the extra planet's period prior was chosen (filled in A8, added to its notes in C3)

# ------------------------------------------------------------------ results shared with Part C
fits = NamedTuple[]
final = nothing
fit_alt = nothing
fit_extra = nothing

# ==============================================================================
# PART B. Ca II H&K activity
# ==============================================================================
if RUN_ACTIVITY

# --- too few R'HK values: say why and stop cleanly (a shell loop goes on to the next star; in the REPL it raises the error)
function hk_stop(msg)
    lr = [x > 0 ? log10(x) : NaN for x in raw.rhk]
    g = filter(isfinite, lr)
    println("\n!! ", msg)
    isempty(g) || println("   log R'HK of all ", length(raw.bjd), " spectra: min ", round(minimum(g); digits = 2), ", median ", round(median(g); digits = 2),
                          ", max ", round(maximum(g); digits = 2), "; accepted range ", LOGRHK_RANGE,
                          " (", count(x -> x < LOGRHK_RANGE[1] || x > LOGRHK_RANGE[2], g), " outside).")
    if !isempty(g)
        nlo = count(x -> x < LOGRHK_RANGE[1], g); nhi = count(x -> x > LOGRHK_RANGE[2], g)
        println("   Below the range: ", nlo, " spectra; above it: ", nhi, " (of ", length(g), ").")
        nhi > nlo && println("   Most rejected values are TOO HIGH for a chromosphere, so they are not real activity levels for this star; the catalogue R'HK is unreliable here.")
        nlo > nhi && println("   Most rejected values are too low (below the photospheric floor): there is no measurable Ca II H&K emission above the noise.")
    end
    println("   The jitter relations used here are fitted to F, G and K stars only (class used: ", SPTYPE, ").")
    println("   Part A (RV) ran; the R'HK analysis (Part B) and the combined Part C are skipped for this star.")
    if isinteractive()
        error(msg)
    else
        dbg("Total run time: ", round((time() - T_START) / 60; digits = 1), " min")
        exit(0)
    end
end

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
length(CLEAN.bjd) >= MIN_NIGHTS || hk_stop("Only $(length(CLEAN.bjd)) usable R'HK values for $STARNAME; need at least $MIN_NIGHTS.")
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
NN >= MIN_NIGHTS || hk_stop("Only $NN nights with R'HK; need at least $MIN_NIGHTS.")
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
        mass ~ truncated(Normal(M_STAR, max(M_STAR_ERR, 0.01 * M_STAR)), lower = max(0.05, M_STAR - 6 * max(M_STAR_ERR, 0.01 * M_STAR)), upper = M_STAR + 6 * max(M_STAR_ERR, 0.01 * M_STAR))   # M_sun, bounded both sides (a one-sided bound let the sampler run to exp(300))
    end
)

"Period prior [d] of planet `name`: PERIOD_OVERRIDE if set, else the periodogram peak +/- PERIOD_WINDOW."
period_prior(name, P_guess) = haskey(PERIOD_OVERRIDE, name) ? PERIOD_OVERRIDE[name] : (P_guess * (1 - PERIOD_WINDOW), P_guess * (1 + PERIOD_WINDOW))

"One planet orbiting the star, with its period prior centred on the periodogram peak (or set by PERIOD_OVERRIDE)."
function make_planet(name, P_guess; free_e = FREE_ECCENTRICITY)
    P_lo, P_hi = period_prior(name, P_guess)
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
function make_rv_obs(d, name)
    return MarginalizedRVObs(
        d;
        target = A, ref = Barycentre,
        name = name,
        variables = @variables begin
            jitter ~ LogUniform(0.1, 100)                    # m/s
        end
    )
end
rv_subset(m) = Table(epoch = rv_data.epoch[m], rv = rv_data.rv[m], σ_rv = rv_data.σ_rv[m])
# SPLIT_ON: two HARPS instruments split at the fibre upgrade (STEP_MJD), each with its own marginalized zero point and its own jitter
const RV_LIKES = SPLIT_ON ? [make_rv_obs(rv_subset(rv_data.epoch .< STEP_MJD), "HARPS_pre"),
                             make_rv_obs(rv_subset(rv_data.epoch .>= STEP_MJD), "HARPS_post")] :
                            [make_rv_obs(rv_data, "HARPS")]
const JIT_NAMES = SPLIT_ON ? ["HARPS_pre_jitter", "HARPS_post_jitter"] : ["HARPS_jitter"]
"Jitter draws [m/s]: the HARPS jitter, or (two eras) the quadrature mean of the two jitters weighted by the number of RVs in each."
function jitter_draws(chain)
    SPLIT_ON || return chain_vec(chain, "HARPS_jitter")
    return sqrt.((N_PRE .* chain_vec(chain, "HARPS_pre_jitter") .^ 2 .+ N_POST .* chain_vec(chain, "HARPS_post_jitter") .^ 2) ./ (N_PRE + N_POST))
end
planets = [make_planet(PLANET_NAMES[k], PERIOD_GUESSES[k]) for k in 1:K_MAX]
# same planets with the opposite eccentricity setting, for the eccentricity test
planets_alt = RUN_ECCENTRICITY_TEST ?
    [make_planet(PLANET_NAMES[k], PERIOD_GUESSES[k]; free_e = !FREE_ECCENTRICITY) for k in 1:N_PLANETS] : Body[]
dbg("Built the star, ", K_MAX, " planet(s) (", N_PLANETS, " for the main fit) and the HARPS likelihood", SPLIT_ON ? "s (before / after the fibre upgrade)" : "")
dbg("Period priors [d]: ", join(("$(PLANET_NAMES[k]): $(round(period_prior(PLANET_NAMES[k], PERIOD_GUESSES[k])[1]; digits = 2))" *
                                 "-$(round(period_prior(PLANET_NAMES[k], PERIOD_GUESSES[k])[2]; digits = 2))" *
                                 (haskey(PERIOD_OVERRIDE, PLANET_NAMES[k]) ? " (PERIOD_OVERRIDE)" :
                                  (k > N_PLANETS && EXTRA_SEED == :residual) ? " (provisional: A8 re-seeds it from the residuals)" : "") for k in 1:K_MAX), ",  "))

# ------------------------------------------------------------------ 4. fits
section("A4. Fits with Pigeons")
"Fit the model with the first `k` planets; returns (k, model, chain, pt, lnZ) or `nothing` if it fails."
function fit_planets(k, plist = planets; label = "")
    try
        sys = System(name = "$(replace(TARGET, r"[^A-Za-z0-9]" => ""))_$(k)_planet" * (k == 1 ? "" : "s") *
                            (label == "" ? "" : "_" * label),
                     bodies = [A, plist[1:k]...], observations = RV_LIKES)
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
    # Where to centre the extra planet's period prior. The sinusoid prewhitening of A2 cannot remove the harmonics of an eccentric orbit, so its next
    # peak can be P/2 of a planet that the Keplerian fit already contains (HD 82943: 109.84 d = P_b/2 for e_b = 0.42). The default therefore uses the
    # periodogram of the residuals of the fitted Keplerian planets instead.
    P_prew = PERIOD_GUESSES[k_extra]
    fitted_periods = [median(chain_vec(final.chain, "$(PLANET_NAMES[k])_P")) for k in 1:N_PLANETS]
    seed_harm = harmonic_of(P_prew, fitted_periods)
    # (the try is evaluated as a value and assigned outside it: a `try` block is its own scope, so assigning inside would leave these globals unchanged)
    _rs = if EXTRA_SEED == :residual
        try
            residual_seed(final, N_PLANETS)
        catch err
            @warn "The residual seed failed; using the prewhitening peak" exception = (err, catch_backtrace())
            (nothing, String[])
        end
    else
        (nothing, String[])
    end
    seed_lines = _rs[2]
    P_seed = _rs[1] === nothing ? P_prew : _rs[1]
    seed_src = _rs[1] === nothing ? "the sinusoid-prewhitening peak" : "the strongest usable peak of the residual periodogram"
    PERIOD_GUESSES[k_extra] = P_seed
    _pp = period_prior(PLANET_NAMES[k_extra], P_seed)
    section("A8. Extra-planet test: $(k_extra) planets (planet $(PLANET_NAMES[k_extra]) " *
            (haskey(PERIOD_OVERRIDE, PLANET_NAMES[k_extra]) ?
                 "with its period prior set by PERIOD_OVERRIDE, $(round(_pp[1]; digits = 2))-$(round(_pp[2]; digits = 2)) d)" :
                 "seeded at $(round(P_seed; digits = 2)) d, $(seed_src))"))
    if !isempty(seed_harm)
        println("!! The prewhitening peak, ", round(P_prew; digits = 2), " d, is ", join(seed_harm, ", "), " of a fitted planet: the harmonic of an eccentric orbit, ",
                "which a sinusoid cannot remove but the Keplerian planet already contains. ",
                (EXTRA_SEED == :residual && P_seed != P_prew) ? "It is not used as the seed." :
                "It is kept as the seed (EXTRA_SEED = :prewhitening or no residual peak): this test then mostly asks whether the harmonic leaves anything.")
        push!(SEED_NOTES, "the prewhitening peak at " * @sprintf("%.2f", P_prew) * " d that suggested this planet is " * join(seed_harm, ", ") *
                          " of a fitted planet: the harmonic of an eccentric orbit, which the Keplerian planet already contains")
    end
    if !isempty(seed_lines)
        println("Peaks of the periodogram of the ", N_PLANETS, "-planet Keplerian residuals, strongest first (FAP with the residual jitter added to the errors):")
        foreach(println, seed_lines)
    end
    isempty(seed_lines) || push!(SEED_NOTES, "period prior centred on " * @sprintf("%.2f", P_seed) * " d, " * seed_src)
    println("Period prior for planet ", PLANET_NAMES[k_extra], ": ", round(_pp[1]; digits = 2), "-", round(_pp[2]; digits = 2), " d")
    planets_x = [planets[1:N_PLANETS]..., make_planet(PLANET_NAMES[k_extra], P_seed)]
    fit_extra = fit_planets(k_extra, planets_x)
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

# ------------------------------------------------------------------ 8b. orbit geometry
section("A8b. Orbit geometry of the fitted planets: do any orbits overlap in radius?")
safe("Orbit geometry") do
    gfit = fit_extra !== nothing ? fit_extra : final
    tab = orbit_table(gfit.chain, gfit.k)
    isempty(tab) && (println("  (no period and eccentricity columns in the chain)"); return)
    println("From the posterior medians of the ", gfit.k, "-planet fit; a from Kepler's third law with the star's mass (planet masses neglected).")
    println(rpad("  planet", 10), rpad("P [d]", 11), rpad("a [AU]", 9), rpad("e (median, 16-84 %)", 24), rpad("periastron [AU]", 17), "apastron [AU]")
    for r in tab
        println("  ", rpad(r.nm, 8), rpad(@sprintf("%.2f", r.P), 11), rpad(@sprintf("%.3f", r.a), 9),
                rpad(@sprintf("%.2f (%.2f-%.2f)", r.e, r.e16, r.e84), 24), rpad(@sprintf("%.3f", r.q), 17), @sprintf("%.3f", r.Q))
    end
    cr = orbit_crossings(tab)
    if isempty(cr)
        println("  No two orbits overlap in radius (the apastron of every inner planet lies inside the periastron of the next).")
    end
    for c in cr
        ia = findfirst(==(c.inner), PLANET_NAMES); ib = findfirst(==(c.outer), PLANET_NAMES)
        later = PLANET_NAMES[max(ia, ib)]; other = later == c.inner ? c.outer : c.inner
        detail = @sprintf("apastron of %s %.2f AU is beyond periastron of %s %.2f AU (period ratio %.3f)%s", c.inner, c.Q, c.outer, c.q, c.ratio,
                          c.robust ? "; still so at the 16th-percentile eccentricities" : "")
        if c.resonance != ""
            println("  ", c.inner, " and ", c.outer, " overlap in radius but are near the ", c.resonance, " resonance, where the orbits can be phase-protected: ", detail, ".")
        else
            println("  !! ", c.inner, " and ", c.outer, " overlap in radius and are not near a low-order resonance: ", detail, ". Two such planets would encounter each other unless",
                    " the phases are special; for a giant planet this is hard to sustain.")
            CROSS_FLAG[later] = "orbit overlaps that of planet " * other * " in radius, outside a low-order resonance: " * detail
        end
    end
    println("  (Not a dynamical stability test: the orbits are not integrated, and the medians can hide a stable corner of the posterior.)")
end

# ------------------------------------------------------------------ 9. analysis summary
section("A9. Analysis summary")
safe("Summary") do
    println("Star: ", STARNAME, " (RVBank \"", TARGET, "\"), assumed mass ", M_STAR, " +/- ", M_STAR_ERR, " Msun")
    println("Data: ", NPTS, " RVs", do_bin ? " (nightly bins)" : "", " over ", round(BASELINE; digits = 0), " days")
    println("Period guesses (days; periodogram peaks, and for the extra planet the seed chosen in A8): ",
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
        for jn in JIT_NAMES
            println("  ", rpad(replace(jn, "_jitter" => "") * " jitter [m/s]", 20), " = ", describe_var(chain, jn))
        end
        SPLIT_ON && println("  pooled jitter [m/s]  = ", round(median(jitter_draws(chain)); digits = 2), "   (quadrature mean over the two eras)")
    end
    println("$(K_FINAL)-planet fit (median, 16/84 percent range):")
    planet_block(final.chain, 1:K_FINAL)
    println()
    if length(fits) >= 2
        println("Jitter by number of planets (median, m/s): ",
                join(("$(f.k): $(round(median(jitter_draws(f.chain)); digits = 2))" for f in fits), ",  "),
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
        println("  Jitter: ", round(median(jitter_draws(final.chain)); digits = 2), " -> ",
                round(median(jitter_draws(fit_extra.chain)); digits = 2), " m/s")
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
const C_SHAPES = try
    kepler_shapes(final, C_K)
catch err
    @warn "Keplerian shapes unavailable; using the harmonic stand-in" exception = err
    nothing
end
const C_MODEL = C_SHAPES === nothing ? "harmonic" : "keplerian"
const _HF = C_SHAPES === nothing ? harmonic_fit(RV_T, RV_Y, RV_S, C_PERIODS) : keplerian_fit(RV_T, RV_Y, RV_S, C_SHAPES)
dbg("Residual model: ", C_MODEL == "keplerian" ?
    "Keplerian shapes (P, e, periastron time of the maximum-posterior draw: " *
    join((@sprintf("%s: P = %.3f d, e = %.2f", PLANET_NAMES[k], C_SHAPES[k][1], C_SHAPES[k][2]) for k in 1:C_K), "; ") * "), amplitudes refitted" :
    "harmonic stand-in (fundamental + first harmonic at the fitted periods)")
const RES_ALL = _HF[1]
const MDL_ALL = _HF[2]
const RES_RMS = std(RES_ALL)
const JIT_FIT = final === nothing ? NaN : median(jitter_draws(final.chain))
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
    println(rpad("  period [d]", 16), rpad("power", 9), rpad("FAP", 11), rpad("R'HK has power there?", 24), "harmonic of a fitted planet?")
    # label like "P_b/3" when P is within 1.5 % of a fitted planet period divided by 2..6
    function resid_harmonic(P)
        out = String[]
        for k in 1:C_K, n in 2:6
            abs(P - C_PERIODS[k] / n) / P < 0.015 && push!(out, "P_$(PLANET_NAMES[k])/$n")
        end
        return isempty(out) ? "" : join(out, ", ")
    end
    for i in pkRes
        P = perRes[i]
        j = findall(p -> abs(p - P) / P < 0.03, perR)
        pr = isempty(j) ? NaN : maximum(powR[j])
        sig = !isempty(j) && LombScargle.fap(pgR, pr) < 0.01
        println("  ", rpad(@sprintf("%.2f", P), 14), rpad(@sprintf("%.3f", powRes[i]), 9),
                rpad(hk_fmt_fap(LombScargle.fap(pgRes, powRes[i])), 11),
                rpad(sig ? "yes (FAP " * hk_fmt_fap(LombScargle.fap(pgR, pr)) * ")" : "no", 24), resid_harmonic(P))
    end
    println(C_MODEL == "keplerian" ?
        "  (the residuals are from Keplerian shapes, so a peak at P_planet/n is not the harmonic of an eccentric orbit; it points to an interaction, a wrong eccentricity or another signal.)" :
        "  (a residual peak at P_planet/n is what an eccentric orbit leaves behind when it is fitted with a sinusoid; the planet fits would absorb it.)")
end

# --- C1b. calendar-year terms ---------------------------------------------------
section("C1b. Calendar-year terms in the planet-removed RV residuals and in log R'HK")
"Fixed-period sinusoids fitted together (weighted linear least squares); returns (amplitude, error) per period. Errors ignore the amplitude-phase covariance."
function sine_terms(t, y, σ, periods)
    t0 = mean(t)
    cols = Vector{Float64}[ones(length(t))]
    for P in periods
        ph = 2π .* (t .- t0) ./ P
        push!(cols, sin.(ph)); push!(cols, cos.(ph))
    end
    X = reduce(hcat, cols)
    w = 1 ./ σ
    Xw = X .* w
    β = Xw \ (y .* w)
    C = inv(Xw' * Xw)
    out = Tuple{Float64,Float64}[]
    for j in eachindex(periods)
        a, b = β[2j], β[2j+1]
        A = hypot(a, b)
        push!(out, (A, A > 0 ? sqrt((a^2 * C[2j, 2j] + b^2 * C[2j+1, 2j+1]) / A^2) : NaN))
    end
    return out
end
const CAL_PERIODS = [365.25 / n for n in 1:4]
const CAL_LABELS = ["1 yr", "1/2 yr", "1/3 yr", "1/4 yr"]
const CAL_BOTH = String[]
try
    jit_c = isfinite(JIT_FIT) ? JIT_FIT : stages[1].jit
    tv = sine_terms(RV_T, RES_ALL, sqrt.(RV_S .^ 2 .+ jit_c^2), CAL_PERIODS)
    okh = isfinite.(N.lr)
    hv = sine_terms(Float64.(N.mjd[okh]), Float64.(N.lr[okh]), fill(std(N.lr[okh]), count(okh)), CAL_PERIODS)
    println("Fixed-period sinusoids at 1/n yr, fitted together to the RV residuals (jitter ", round(jit_c; digits = 2), " m/s added to the errors) and to nightly log R'HK:")
    println(rpad("  term", 9), rpad("P [d]", 9), rpad("RV amp [m/s]", 20), rpad("RV S/N", 9), rpad("R'HK amp [dex]", 20), "R'HK S/N")
    for j in 1:4
        sr = tv[j][1] / tv[j][2]; sh = hv[j][1] / hv[j][2]
        println("  ", rpad(CAL_LABELS[j], 7), rpad(@sprintf("%.2f", CAL_PERIODS[j]), 9),
                rpad(@sprintf("%.2f +/- %.2f", tv[j][1], tv[j][2]), 20), rpad(@sprintf("%.1f", sr), 9),
                rpad(@sprintf("%.4f +/- %.4f", hv[j][1], hv[j][2]), 20), @sprintf("%.1f", sh))
        sr > 3 && sh > 3 && push!(CAL_BOTH, @sprintf("%s (%.1f d)", CAL_LABELS[j], CAL_PERIODS[j]))
    end
    isempty(CAL_BOTH) || println("  Both series have a term > 3 sigma at: ", join(CAL_BOTH, ", "), ". A planet at such a period cannot be told from an observing-season systematic with this data.")
    println("  (An amplitude is biased high at low S/N. The fit is linear, so it is a quick screen; the planet-fit evidence for a signal at these periods is not affected by it.)")
catch err
    @warn "C1b (calendar-year terms) failed -- continuing" exception = (err, catch_backtrace())
end

# --- C1c. before / after the 2015 fibre upgrade ---------------------------------------
section("C1c. Planet-removed RV residuals before and after the 2015 HARPS fibre upgrade (MJD $(STEP_MJD))")
# The fibre upgrade shifted FWHM, BIS and CRX (and can shift the RV by a few m/s). A residual offset between the two eras, together with an indicator
# that has also shifted, produces a pooled rank correlation by itself. Splitting the data tells that apart from a correlation that holds within each era.
const ERA_LINES = String[]
try
    STEP_MJD > 0 || error("STEP_MJD = 0, so there is no split")
    # (1) offset of the residuals, one mean per night so that a night with many exposures does not dominate
    night = floor.(Int, RV_T)
    nm_t = Float64[]; nm_r = Float64[]
    for d in sort(unique(night))
        i = findall(==(d), night)
        push!(nm_t, mean(RV_T[i])); push!(nm_r, mean(RES_ALL[i]))
    end
    pre = nm_t .< STEP_MJD; post = .!pre
    (count(pre) >= 8 && count(post) >= 8) || error("fewer than 8 RV nights on one side ($(count(pre)) before, $(count(post)) after)")
    m1 = mean(nm_r[pre]);  e1 = std(nm_r[pre]) / sqrt(count(pre))
    m2 = mean(nm_r[post]); e2 = std(nm_r[post]) / sqrt(count(post))
    off = m2 - m1; eoff = hypot(e1, e2); soff = abs(off) / max(eoff, 1e-9)
    println(@sprintf("Residual RV (nightly means): %+.2f +/- %.2f m/s before MJD %.0f (%d nights), %+.2f +/- %.2f m/s after (%d nights)  ->  offset %+.2f +/- %.2f m/s (%.1f sigma).",
                     m1, e1, STEP_MJD, count(pre), m2, e2, count(post), off, eoff, soff))
    absorbed = SPLIT_ON && C_MODEL == "keplerian" && isfinite(ERA_STEP[])
    absorbed && println(@sprintf("The planet fit has a separate zero point for each era (SPLIT_HARPS), and the Keplerian stand-in used for these residuals has one too, so the offset above should be\nnear zero: the fitted offset across MJD %.0f is %+.2f m/s.", STEP_MJD, ERA_STEP[]))
    absorbed ? push!(ERA_LINES, @sprintf("  Fibre-upgrade split (C1c): the planet fit has its own zero point and jitter for each era; the fitted offset across MJD %.0f is %+.2f m/s and is already removed from the residuals (their remaining before/after difference is %+.2f +/- %.2f m/s).",
                                         STEP_MJD, ERA_STEP[], off, eoff)) :
    push!(ERA_LINES, @sprintf("  Fibre-upgrade split (C1c): the planet-removed residuals average %+.2f m/s before MJD %.0f and %+.2f m/s after (%d and %d nights), an offset of %+.2f +/- %.2f m/s (%.1f sigma)%s",
                              m1, STEP_MJD, m2, count(pre), count(post), off, eoff, soff,
                              soff > 3 ? ": a shift of this size is what an RV offset across the upgrade (or a slow second signal) leaves behind, and it can create indicator correlations by itself." :
                                         ": no significant offset between the eras."))
    # (2) the indicator correlations, pooled and within each era
    pre_n = N.mjd .< STEP_MJD
    fmt(r, n) = isfinite(r) ? @sprintf("%+.2f (n=%d)", r, n) : "n/a (n=$n)"
    println("\nSpearman correlation of each indicator with the residual RV, pooled and within each era (nights with a residual):")
    println(rpad("  indicator", 13), rpad("pooled", 17), rpad("before", 17), rpad("after", 17), "reading (shown for a pooled correlation > 3 sigma)")
    held = String[]; part = String[]; lost = String[]
    for (k, (nm, v)) in enumerate(IND_C)
        ra = RHO_RES[k]; na = N_RES[k]
        r1 = hk_spearman(v[pre_n], RES_N[pre_n]);   n1 = hk_npair(v[pre_n], RES_N[pre_n])
        r2 = hk_spearman(v[.!pre_n], RES_N[.!pre_n]); n2 = hk_npair(v[.!pre_n], RES_N[.!pre_n])
        sa = hk_corr_sigma(ra, na); s1 = hk_corr_sigma(r1, n1); s2 = hk_corr_sigma(r2, n2)
        sig_a = isfinite(sa) && sa > 3
        ok1 = isfinite(s1) && s1 > 2 && sign(r1) == sign(ra)
        ok2 = isfinite(s2) && s2 > 2 && sign(r2) == sign(ra)
        reading = !sig_a ? "" : (ok1 && ok2) ? "holds within both eras" : (ok1 || ok2) ? "holds within one era only" : "pooled only: follows the offset between the eras"
        sig_a && ((ok1 && ok2) ? push!(held, nm) : (ok1 || ok2) ? push!(part, nm) : push!(lost, nm))
        println("  ", rpad(nm, 11), rpad(fmt(ra, na), 17), rpad(fmt(r1, n1), 17), rpad(fmt(r2, n2), 17), reading)
    end
    if !(isempty(held) && isempty(part) && isempty(lost))
        push!(ERA_LINES, "  Of the indicators that correlate with the residuals (> 3 sigma, pooled): " *
              join(filter(!isempty, [isempty(held) ? "" : "within both eras: " * join(held, ", "),
                                     isempty(part) ? "" : "within one era only: " * join(part, ", "),
                                     isempty(lost) ? "" : "only in the pooled data, so following the offset between the eras: " * join(lost, ", ")]), "; ") * " (C1c).")
    end
    println("(Each side needs > 2 sigma and the pooled sign to count as holding. A step already removed in A1 is not in these residuals.)")
catch err
    println("C1c not run: ", sprint(showerror, err))
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
    add!("raw RV scatter (rms)", std(RV_Y))
    add!("median RV error", median(RV_S))
    for f in fits
        add!("fitted jitter, $(f.k) planet" * (f.k == 1 ? "" : "s"), median(jitter_draws(f.chain)))
    end
    fit_extra === nothing || add!("fitted jitter, $(fit_extra.k) planets (extra-planet test)", median(jitter_draws(fit_extra.chain)))
    add!(C_MODEL == "keplerian" ? "scatter left after the Keplerian planet model" : "scatter left after the harmonic planet model", RES_RMS)
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
else
    emit("Stellar parameters: " * (STAR_TABLE === nothing ? "VizieR table1 not available" : "star not found in VizieR table1") *
         "; spectral class " * SPTYPE * " from the target-list note, no vsini limit.")
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
emit("  Raw RV rms " * hk_ms(std(RV_Y)) * " m/s  vs  after the planets " * hk_ms(RES_RMS) * " m/s  vs  median RV error " * hk_ms(median(RV_S)) * " m/s" *
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

# Largest apsidal precession rate (deg/yr) that is accepted as the reason a candidate sits near a harmonic of an earlier planet.
# GJ 876 b-c, among the most strongly interacting known pairs, precesses at about -41 deg/yr (Correia et al. 2010).
const MAX_PRECESSION = 60.0
# A signal closer to 1 yr or 0.5 yr than this many resolution elements (1/T) is flagged as a possible annual alias.
const ANNUAL_RES_ELEMENTS = 1.5

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
                if b == 1 && a >= 2 && P < eP
                    dpre0 = 360.0 * 365.25 * (1 / P - a / eP)
                    if abs(dpre0) > MAX_PRECESSION
                        push!(notes, @sprintf("period ratio is near %d:1 with planet %s, but the offset would need %+.0f deg/yr of apsidal precession (more than %.0f), too much for a precessing harmonic: treated as a coincidence", a, en, dpre0, MAX_PRECESSION))
                        continue
                    end
                end
                res_flag = true; push!(notes, "period ratio $(a):$(b) with planet $en (resonance or harmonic)")
                # could it merely be a harmonic of an eccentric orbit? its RV amplitude would be about e x K of the stronger one
                if b == 1 && a >= 2 && P < eP
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
    # a fixed 1-yr alias: a seasonally sampled series cannot separate P from 365.25 d when the frequency offset is below
    # a few resolution elements 1/T
    for (n, lab) in ((1, "1 yr"), (2, "0.5 yr"), (3, "1/3 yr"), (4, "0.25 yr"))
        res = abs(1 / P - n / 365.25) * BASELINE
        res < ANNUAL_RES_ELEMENTS || continue
        if n <= 2
            alias_flag = true
            push!(notes, @sprintf("within %.1f resolution elements (1/T, T = %.1f yr) of %s: cannot be separated from the %s sampling alias; check the window function and the BERV%s",
                                  res, BASELINE / 365.25, lab, n == 1 ? "annual" : "half-year",
                                  n == 1 ? "" : " (a seasonal systematic can also show up here and in R'HK)"))
        else
            push!(notes, @sprintf("within %.1f resolution elements of %s: a seasonal systematic could contribute at this period (not flagged: the test would also mark real planets)", res, lab))
        end
    end
    if occursin("876", string(STARNAME)) && abs(P - 1.9376) / 1.9376 < 0.03
        push!(notes, "matches the published inner planet GJ 876 d (P = 1.94 d)")
    end
    # an orbit that overlaps another planet's in radius, outside a low-order resonance (A8b), is dynamically doubtful
    cross_flag = haskey(CROSS_FLAG, nm)
    cross_flag && push!(notes, CROSS_FLAG[nm])
    verdict = activity_flag ? "ACTIVITY-SUSPECT" :
              alias_flag    ? "ALIAS-SUSPECT" :
              harm_flag     ? "HARMONIC-SUSPECT (may be an unmodelled eccentricity or interaction of an earlier planet)" :
              cross_flag    ? "ORBIT-OVERLAP (crosses another planet's orbit outside a resonance; dynamically doubtful, see A8b)" :
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
        for n in SEED_NOTES
            push!(cands[end].notes, n)
        end
    elseif length(PERIOD_GUESSES) > C_K
        nm = PLANET_NAMES[C_K+1]
        push!(cands, vet(nm, PERIOD_GUESSES[C_K+1], NaN, copy(earlier), nothing))
    end
end

# Significance of each candidate's periodogram peak (prewhitening stage k, Baluev FAP). A signal can match no alias, harmonic or
# activity peak and still be noise; the verdict says nothing about that, so a weak peak is named separately.
WEAK_FAP = 0.01
# FAP of the periodogram peak at the candidate's own period c.P, after removing a sinusoid at the fitted period of every earlier planet
# (same jitter-inflated errors as A2). Falls back to the prewhitening stage of that letter if this fails.
const FAP_CACHE = Dict{String,Float64}()
function peak_fap(c)
    haskey(FAP_CACHE, c.nm) && return FAP_CACHE[c.nm]
    k = findfirst(==(c.nm), PLANET_NAMES)
    k === nothing && return NaN
    fb = k > length(stages) ? NaN : stages[k].fap
    f = try
        r = copy(RV_Y)
        for P in C_PERIODS[1:min(k - 1, length(C_PERIODS))]
            r = r .- LombScargle.model(RV_T, r, RV_S, 1 / P)
        end
        stage_fap(RV_T, r, sqrt.(RV_S .^ 2 .+ stages[1].jit^2), c.P, fb)
    catch
        fb
    end
    FAP_CACHE[c.nm] = f
    return f
end
fap_str(f) = !isfinite(f) ? "n/a" : (f < 1e-10 ? "<1e-10" : @sprintf("%.1e", f))
is_weak(c) = isfinite(peak_fap(c)) && peak_fap(c) > WEAK_FAP

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
         rpad(isfinite(lb) ? @sprintf("%+.1f", lb) : "-", 20) * c.verdict * (is_weak(c) ? "; WEAK peak" : ""))
    emit("           - periodogram peak FAP " * fap_str(peak_fap(c)) *
         (is_weak(c) ? " (above $(WEAK_FAP): not significant on its own; the planet fit and its ln BF decide)" : ""))
    for n in c.notes
        emit("           - " * n)
    end
end

emit()
emit("OVERALL")
cand_ok(c) = startswith(c.verdict, "no activity") || startswith(c.verdict, "resonant")
emit("  Signals that pass every check run here (no activity, alias or harmonic match, peak FAP <= $(WEAK_FAP)): " *
     join((c.nm for c in cands if cand_ok(c) && !is_weak(c)), ", "))
WEAKS = [c for c in cands if is_weak(c)]
isempty(WEAKS) || emit("  Signals whose periodogram peak is not significant (FAP > $(WEAK_FAP)): " *
                       join((c.nm * " (FAP " * fap_str(peak_fap(c)) * ")" for c in WEAKS), ", "))
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
OTHER_SIG = [(nm, RHO_RES[k], hk_corr_sigma(RHO_RES[k], N_RES[k])) for (k, (nm, _)) in enumerate(IND_C)
             if k > 1 && isfinite(RHO_RES[k]) && abs(hk_corr_sigma(RHO_RES[k], N_RES[k])) > 3]
if !isempty(OTHER_SIG)
    emit("  Other indicators DO correlate with the residual RVs: " *
         join((@sprintf("%s rho = %+.2f (~%.1f sigma)", nm, r, sg) for (nm, r, sg) in OTHER_SIG), "; ") * ".")
    emit("  Possible causes: activity that R'HK does not show, a change of instrument state (the HARPS fibre upgrade of 2015 shifted FWHM, BIS and CRX), or a poor planet model")
    emit(C_MODEL == "keplerian" ?
         "  (the residuals use Keplerian shapes fixed at the best draw, and the model error can follow the sampling). C1c splits the data before and after 2015." :
         "  (sinusoids stand in for eccentric Keplerians here, and the model error can follow the sampling). Check by fitting the planets; C1c splits the data before and after 2015.")
end
for l in ERA_LINES
    emit(l)
end
if !isempty(CAL_BOTH)
    emit("  Calendar-year terms: RV residuals and R'HK both carry a fixed-period term (> 3 sigma) at " * join(CAL_BOTH, ", ") *
         " (C1b). A signal at such a period is more likely an observing-season systematic than a planet; the data alone cannot separate them.")
end
emit("  Limits: R'HK sees chromospheric (plage) activity only; spots and convective blueshift can move the RV without moving R'HK. The residuals come from")
emit(C_MODEL == "keplerian" ?
     "  Keplerian shapes of the best draw (amplitudes refitted), and in a resonant system the Keplerian model itself is approximate." :
     "  a harmonic stand-in for the Keplerian fit, and in a resonant system the Keplerian model itself is approximate.")
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
