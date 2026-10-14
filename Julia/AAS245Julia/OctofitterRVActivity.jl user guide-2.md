# OctofitterRVActivity.jl: a guide for first-time users

Oct 14, 2026 · @Joseph

## What the script does

OctofitterRVActivity.jl takes one star from the HARPS archive, fits planets to its radial velocities (RVs), measures its Ca II H&K activity, and then asks whether each planet candidate could instead be stellar activity, a seasonal alias or a harmonic of a stronger planet. One run covers one star and writes a log, a text summary and up to 23 PNG plots.

- **Part A, planet fit.** Lomb-Scargle prewhitening picks starting periods. Octofitter (v9) then fits 1 to N planets (default N = 2) with Pigeons parallel tempering and compares the models by their log Bayesian evidence (ln Z). It also tests free against circular eccentricity, tests one extra planet, and checks whether any fitted orbits overlap in radius. This part takes tens of minutes per star.
- **Part B, activity.** It cleans the Ca II H&K index (log R'HK) of every spectrum, averages it per night, and looks for a rotation period and a long activity cycle. It compares R'HK with the RV and with six other indicators (CRX, dLW, H-alpha, Na D1, BIS, FWHM), and compares the RV scatter with the activity-jitter floor of Perdelwitz et al. (2024). This part takes about a minute.
- **Part C, both together.** It removes the fitted planets from the RVs and tests the residuals against the activity indicators. It splits them before and after the 2015 HARPS fibre upgrade, looks for calendar-year terms, and vets every candidate. The verdicts are printed and saved as summary\_\<star>.txt.

The star list holds 100 HARPS stars sorted by name. Twenty-five are hand-picked planet hosts such as GJ 876, GJ 581 and HD 10180. The other 75 are the best-sampled HARPS stars, chosen by a rule and not by science, and they carry class-default masses. Any other target name in the data table also works.

The verdicts are screening heuristics. A clean verdict does not prove a planet, and a flag names a reason for doubt, not a rejection.

## Before you start

You need Julia, an internet connection for the first run, and a folder you can write to. The script was written and run on Julia 1.12.7 on an Apple Silicon Mac; other Julia versions, Windows and Linux are untested.

- **Where to put it.** Save the script in one folder, for example /Volumes/SSD2/MyJulia/AAS245 Julia Scripts. It creates a data cache (cahk\_cache) and one figure folder per star (figures\_rvact\_\<star>) next to itself.
- **File name.** The examples below use OctofitterRVActivity.jl. If you saved it under another name, such as OctofitterRVActivity-18.jl, use that name instead.
- **Apple Silicon.** The script loads AppleAccelerate for faster FFTs. On another machine set USE\_APPLE\_ACCELERATE = false.

**What happens on the first run**

1. Julia builds a throw-away package environment, installs the packages listed in the files section, and precompiles them. On the author's iMac this took about 3 minutes.
2. The script downloads the VizieR table of RVs and activity indices for all HARPS stars (about 120 MB) and saves it in cahk\_cache.
3. It downloads a small VizieR table of stellar parameters (Teff, logg, \[Fe/H\], vsini) for about 3,600 stars.

Later runs read the cached tables and need no network, except for any package work Julia repeats. On the author's machine Julia also set up a Python environment (CondaPkg) during the first precompile; this came from the global Julia environment, not from the script's own packages.

## Running it

The script runs one star per Julia process, and you can start it from a terminal, from the Julia prompt or from VSCode. You need to cd to the script's folder so Julia can find the file; the data cache and figures always go in the script's own folder, wherever you start from.

The star comes from, in order of priority: TARGET\_OVERRIDE in the settings (a name such as "HD82943"), then a number or name given on the command line, then TARGET\_INDEX (default 47, GJ 876). The script prints the numbered list of 100 stars and marks the one selected; HD 82943 is number 81 in the current list.

**From a terminal (the simplest way)**

```bash
cd "/Volumes/SSD2/MyJulia/AAS245 Julia Scripts"
julia OctofitterRVActivity.jl            # the star set by TARGET_INDEX
julia OctofitterRVActivity.jl 81         # star number 81 in the printed list
julia OctofitterRVActivity.jl HD82943    # by name: a list entry or any target name in the table
julia --threads=auto OctofitterRVActivity.jl HD82943   # explicit threads (v1.1.12 adds them itself)
```

**From the Julia prompt**

```julia
cd("/Volumes/SSD2/MyJulia/AAS245 Julia Scripts")
include("OctofitterRVActivity.jl")       # the star comes from TARGET_INDEX / TARGET_OVERRIDE
```

The REPL cannot hand a star number to include, so to pick a star you either edit TARGET\_OVERRIDE or start a fresh Julia process from the prompt. The second way also gives a clean session each time:

```julia
cd("/Volumes/SSD2/MyJulia/AAS245 Julia Scripts")
run(`julia --threads=auto OctofitterRVActivity.jl HD82943`)
```

Use include, not a paste of the file. The REPL keeps an in-memory copy of what it has loaded, so after you edit a setting, restart the REPL or use a fresh process; Julia may otherwise warn about redefined constants and keep the old values.

**From VSCode**

Open the script's folder in VSCode (File, Open Folder), open the file, and run it with the Julia extension's "Execute active file in REPL" command. This is the include route above, so the star comes from the settings. VSCode's REPL starts with one Julia thread; the script cannot change that from inside the REPL. To use threads, set the Julia extension setting julia.NumThreads to auto (or a number), then restart the Julia REPL, and check with Threads.nthreads() at the julia> prompt.

**Threads**

Pigeons runs its parallel-tempering chains on threads, so with one thread they run one after another. From a terminal, v1.1.12 restarts itself with --threads=THREADS (default "auto"). It does nothing if you already set threads with --threads or JULIA\_NUM\_THREADS, or if THREADS = "1". Section 0 of the log prints the thread count and the number of CPU threads the computer reports; that count can be lower than what the machine can use (see the 6-thread test below). On one iMac with 4 CPU threads, HD 82943 took about 35 minutes on one thread and 13.5 to 14.2 minutes with 4 threads (two runs). The runs were not timed under controlled conditions, so treat the speed-up as roughly 2.5 times, not a precise figure. The ln Z values from the threaded runs matched the single-thread ones to within 0.03. A later run with --threads=6, although Julia reported 4 CPU threads, took 13.3 minutes against 14.4 with 4 threads (sampling 653 s against 716 s, about 8% less), and Activity Monitor showed the Julia process rising from about 330% to 486% CPU. Every ln Z, the extra-planet seed and the fitted offset were identical at 4 and 6 threads. More threads than the reported count gave a small gain on this machine, so try it with --threads=N and compare the "sampling took" lines; 'auto' is the safe default.

## Running many stars

Start one fresh Julia process per star and run the stars one after another. Each star writes its own figure folder, so nothing is overwritten, and a failing star does not stop the loop if you wrap the call in ignorestatus.

**From the Julia prompt**

```julia
cd("/Volumes/SSD2/MyJulia/AAS245 Julia Scripts")
for i in 1:25
    println("\n######## star $i ########")
    run(ignorestatus(`$(Base.julia_cmd()) --threads=auto OctofitterRVActivity.jl $i`))
end
```

Change the range for the next chunk: 26:50, 51:75, 76:100. To loop over names instead of numbers:

```julia
for s in ["HD10700", "HD20794", "GJ876"]
    run(ignorestatus(`$(Base.julia_cmd()) --threads=auto OctofitterRVActivity.jl $s`))
end
```

**With one log file per star (Julia prompt)**

```julia
for i in 1:25
    run(pipeline(ignorestatus(`$(Base.julia_cmd()) --threads=auto OctofitterRVActivity.jl $i`),
                 stdout = "log_star_$i.txt", stderr = "err_star_$i.txt"))
end
```

**From a terminal**

```bash
cd "/Volumes/SSD2/MyJulia/AAS245 Julia Scripts"
for i in $(seq 1 25); do
  echo "######## star $i ########"
  julia --threads=auto OctofitterRVActivity.jl $i 2>&1 | tee log_star_$i.txt
done
```

**Time per star**

- With RUN\_PLANET\_FIT = false a star takes about a minute, so 100 stars in four chunks of 25 is practical.
- With the planet fits on, HD 82943 took 35 minutes on one thread, of which about 30 were sampling. Stars with more RVs take longer, so 100 stars with fits would take days. Fit only the stars you care about.
- Do not run several stars at once with --threads=auto; the processes compete for the same cores.
- Each new process rebuilds its throw-away environment. For a long loop, set USE\_TEMP\_ENV = false in the settings so one environment (.octofitter\_v9\_env, next to the script) is kept and reused.

## Settings you can change

The settings are the `const` lines near the top of the script (about lines 161–208). The star is chosen by TARGET\_OVERRIDE, then a command-line argument, then TARGET\_INDEX.

| Setting | Default | What it does |
| --- | --- | --- |
| TARGET\_INDEX | 47 | Number of the star in the built-in list (47 = GJ 876, 81 = HD 82943). The list is alphabetical and is printed at the start of every run. |
| TARGET\_OVERRIDE | "" | A star name from the table, e.g. "HD10700". Beats TARGET\_INDEX and the command-line argument. |
| RUN\_ACTIVITY | true | Parts B and C (Ca II H&K and combined analysis). About a minute. |
| RUN\_PLANET\_FIT | true | Part A, the Octofitter planet fits (tens of minutes). false gives the quick run; RV Plots 1–2 are still made. |
| DATA\_SOURCE | :auto | :auto uses the VizieR cache, then the GitHub CSV; or force :vizier or :github. |
| REFRESH\_CACHE | false | true downloads the VizieR table again. |
| USE\_STAR\_TABLE | true | Also use VizieR table1 (Teff, logg, \[Fe/H\], vsini) for the star. |
| VSINI\_MIN\_RELIABLE | 2.0 | A vsini at or below this km/s is treated as an upper limit, so no rotation limit is derived from it. |
| N\_PLANETS | 2 | Fit models with 1 to N planets. |
| FIT\_ALL\_SUBMODELS | true | true fits 1…N and compares; false fits only the N-planet model. |
| FREE\_ECCENTRICITY | true | Each planet gets e \~ Uniform(0, 0.7); false forces circular orbits. |
| RUN\_ECCENTRICITY\_TEST | true | Also fit the N-planet model with the opposite eccentricity setting. |
| RUN\_EXTRA\_PLANET\_TEST | true | Also fit N+1 planets, the extra one seeded from the residuals. |
| EXTRA\_SEED | :residual | Where the extra planet's period prior is centred: :residual (strongest residual peak) or :prewhitening (the v1.1.11 behaviour). |
| SEED\_EXCLUDE | 0.15 | A residual peak within this fraction of a fitted planet's period is skipped as a seed. |
| PERIOD\_WINDOW | 0.10 | Period prior = peak period ± this fraction. |
| PERIOD\_OVERRIDE | empty | Hand-set period prior per planet letter, e.g. Dict("c" => (85.0, 95.0)) in days. |
| MIN\_PERIOD | 1.5 | Days. Shorter periodogram peaks are ignored (1 d peaks are sampling aliases). |
| RV\_BIN\_NIGHTLY | :auto | Bin RVs by night: :auto (only if more than MAX\_POINTS), true or false. |
| MAX\_POINTS | 1000 | Threshold for the :auto binning. |
| CUSTOM\_STAR\_MASS | 1.0 | Solar masses; used only for a star not in the built-in list. |
| N\_ROUNDS | 10 | Pigeons rounds. Each extra round doubles the work; 9 is a quick look. |
| OUTLIER\_SIGMA | 10.0 | Drop RVs beyond this many robust sigma (MAD) from the median; 0 keeps everything. |
| STEP\_MJD | 57170.0 | HARPS fibre upgrade. An RV offset across it is fitted and removed; 0 turns this off. |
| STEP\_WINDOW | 400.0 | Days on each side of STEP\_MJD used to measure the offset. |
| STEP\_MIN\_SIGMA | 3.0 | The offset is applied only if it is at least this many sigma from zero (and at least 10 RVs lie on each side). |
| ACT\_BIN\_NIGHTLY | true | Average the activity spectra of each night. |
| REQUIRE\_UNFLAGGED | false | true drops rows with vsinflag or metflag above 0 (ignored if every row is flagged). |
| LOGRHK\_RANGE | (-7.0, -3.0) | Plausible log10 R'HK; values outside are treated as garbage. |
| MAX\_REL\_ERR | 1.0 | Drop spectra whose R'HK error exceeds this multiple of R'HK. |
| ROT\_RANGE | (2.0, 150.0) | Days. The "rotation" window for the folded plot and peak table. |
| CYCLE\_MIN | 500.0 | Days. Longer periods are activity-cycle candidates. |
| MIN\_NIGHTS | 15 | Minimum nights for the periodograms. |
| SEASON\_MIN\_NIGHTS | 3 | Nights needed for a calendar-year season to get a mean. |
| USE\_TEMP\_ENV | true | true builds a throw-away package environment every run; false keeps one in .octofitter\_v9\_env next to the script. |
| USE\_APPLE\_ACCELERATE | true | Apple Silicon only. Set false if an FFT call complains. |
| THREADS | "auto" | Threads for the Pigeons chains: "auto", a number such as "4", or "1" for no restart. |
| SPLIT\_HARPS | true | The RVs before and after STEP\_MJD (the 2015 fibre upgrade) become two HARPS instruments in the planet fit, each with its own zero point and jitter. false gives one series (the v1.1.12 model). Added in v1.1.13. |
| SPLIT\_MIN | 10 | The split needs at least this many RVs (after any nightly binning) on each side of STEP\_MJD; otherwise the fit uses one series. |

**How to change a setting**

The simplest way is to edit the const line in the script and save. From the Julia prompt, restart Julia first if you have already run the script in that session, because the REPL keeps the old constants.

To leave your master copy alone, make a changed copy in the same folder (the cache and figure folders sit next to the script, so the copy must stay there). In a terminal:

```bash
cd "/Volumes/SSD2/MyJulia/AAS245 Julia Scripts"
sed -E 's/^const RUN_PLANET_FIT( *)= *true/const RUN_PLANET_FIT\1= false/' \
    OctofitterRVActivity.jl > OctofitterRVActivity-quick.jl
julia OctofitterRVActivity-quick.jl 47
```

The same idea works for any setting by changing its name and value, for example N\_ROUNDS to 9:

```bash
sed -E 's/^const N_ROUNDS( *)= *[0-9]+/const N_ROUNDS\1= 9/' OctofitterRVActivity.jl > OctofitterRVActivity-r9.jl
```

I checked that these two sed lines make the intended edit on the script. I have not run the resulting copies in Julia.

## Files downloaded and written

**Data (downloaded once, then cached)**

| File | Size | Source and use |
| --- | --- | --- |
| cahk\_cache/J\_AA\_683\_A125\_table4.csv | about 120 MB, 252,615 rows | VizieR J/A+A/683/A125 table4 (Perdelwitz et al. 2024): HARPS RVs, activity indicators and R'HK for every spectrum. Saved in the folder next to the script. |
| cahk\_cache/J\_AA\_683\_A125\_table1.csv | small, 3,612 rows | VizieR table1: Teff, logg, \[Fe/H\], vsini and the SIMBAD name for each star. Used only if USE\_STAR\_TABLE = true. |
| HARPS\_RVBank\_ver02.csv | about 38 MB zip, 144 MB unpacked | Fallback from GitHub (HARPS-RVBank, Trifonov et al. 2020), fetched through DataDeps into Julia's own data folder. It may pre-date the April 2026 correction of the VizieR table. |

To force a fresh VizieR download, set REFRESH\_CACHE = true once.

**Julia packages (installed automatically on the first run)**

Octofitter 9, OctofitterRadialVelocity 9, PlanetOrbits, CairoMakie, PairPlots, Distributions, Pigeons, LombScargle, CSV, DataFrames and DataDeps, plus AppleAccelerate on Apple Silicon. By default they go into a throw-away environment, so the first run of every star reinstalls them. With USE\_TEMP\_ENV = false they are kept in .octofitter\_v9\_env next to the script.

**Output (one folder per star)**

- figures\_rvact\_\<STAR>/ next to the script, holding the PNG plots described in the plot guide, at twice the screen pixel density.
- summary\_\<STAR>.txt in the same folder: the verdict tables and conclusions from the end of the log, so you can read the result without scrolling the whole run.
- The full log goes to the screen only. To keep it, use the per-star log loop in the "Running many stars" section, or tee in a terminal.

If disk space matters, the cache and package files are the large items; the PNGs and summaries are small.

## Reading the log

The log is divided into numbered sections. The order in a run is: environment, data loading, B0, A1, A2, B2–B5, A3–A9, then C1–C3 and Done.

| Section | What it shows |
| --- | --- |
| 0. Environment | Package setup, the Julia thread count and the CPU threads the computer reports. |
| Data | Which table was loaded and how many spectra and nights the star has. |
| B0. Stellar parameters | Teff, logg, \[Fe/H\] and vsini from table1, and the longest rotation period they allow. |
| A1. RV data | RV count, baseline, outlier removal and the fibre-upgrade offset, if one was applied. |
| A2. Candidate periods | Periodogram prewhitening: the strongest periods in turn, with false-alarm probabilities. These set the period priors. |
| B2–B5. Activity | Periodograms of log R'HK and the RV, whether the RV follows the activity, the activity-jitter lower limit (Eqs. 1–3 of Perdelwitz et al. 2024) and a summary. |
| A3. Model definition | The priors and the period range of each planet (the extra planet is marked provisional). |
| A4. Fits | One Pigeons run per model; this is where the time goes. |
| A5. Model comparison | ln Z of each model and the ln Bayes factor against the previous one. |
| A6. Corner plots | The 2 vs 3 planet corner plot. |
| A7. Eccentricity test | The same N-planet model with the opposite eccentricity setting. |
| A8. Extra-planet test | N+1 planets, seeded at the strongest peak of the residual periodogram. Prints the peaks considered and warns if the seed is a harmonic of a fitted planet. |
| A8b. Orbit geometry | P, a, e, periastron and apastron of each planet, and whether any orbits overlap in radius. |
| A9. Analysis summary | The period guesses used and the conclusions of Part A. With SPLIT\_HARPS on, the jitter is reported for each era and pooled. |
| C1. Residuals | The RV residuals after the planet model, compared with each activity indicator. |
| C1b. Calendar-year terms | Yearly signals in the residuals and in log R'HK. |
| C1c. Fibre-upgrade check | The residual offset across the 2015 upgrade, and each indicator's Spearman correlation with the residuals, pooled and within each era. With SPLIT\_HARPS on, it also gives the fitted offset between the eras, which is already removed from the residuals. |
| C2. Agreement | Where the RV and activity analyses agree and differ. |
| C3. Candidate signals | One verdict per fitted planet; the key result of the run. |

**Verdict labels in C3**

- **no activity, alias or harmonic match:** none of the checks flagged the signal. This is not a confirmation.
- **ACTIVITY-SUSPECT:** the period matches an activity period (rotation, its harmonics or a cycle) seen in R'HK or an indicator.
- **ALIAS-SUSPECT:** the period matches a known alias, for instance a calendar-year or sampling alias.
- **HARMONIC-SUSPECT:** the period is a harmonic of an earlier planet, so it may be an unmodelled eccentricity or interaction.
- **ORBIT-OVERLAP:** the orbit crosses another planet's outside a low-order resonance, so it is dynamically doubtful (see A8b). Near-resonant pairs (within 3% of 2:1, 3:1, 4:1, 3:2, 5:2, 5:3 or 4:3) are exempt.
- **UNTESTED harmonic:** a 2:1, 3:1 or 4:1 relation to an earlier planet, but the planet fits were not run, so it is untested.
- **resonant partner:** planet-planet interaction makes the Keplerian model approximate.
- **WEAK:** the periodogram false-alarm probability is above 0.01.

**Reading ln Z and ln BF**

ln Z is the log Bayesian evidence from Pigeons' stepping-stone estimate. It is good to a few tenths, so a ln Bayes factor of about 1 or less is a tie. Larger values favour the model with the higher ln Z, but they carry the usual cautions: the Occam factor depends on the priors, and an extra planet seeded at the strongest residual peak has been chosen by looking, so the Bayes factor is not corrected for that look-elsewhere effect. Treat a high ln BF together with a weak periodogram peak (large FAP) or an ORBIT-OVERLAP verdict as unconfirmed.

## Plot guide

Every plot is a PNG in figures\_rvact\_\<STAR>/ with its caption written across the top, so each image explains itself. A star's full set has 10 RV plots, 8 HK plots and 5 combined plots. Which ones appear depends on your settings and the star's data, as noted below. In the file names, N is the number of planets in the fit.

### RV plots (Part A)

**RV Plot 1, rv\_plot01\_rv\_data.png.** The HARPS radial velocities against time, with the pipeline uncertainties as error bars. If the star has more than MAX\_POINTS RVs they are binned per night and the caption says so. Use it to check the baseline, the gaps, any outliers and the jump at the 2015 fibre upgrade. Made even with RUN\_PLANET\_FIT = false.

**RV Plot 2, rv\_plot02\_periodogram\_prewhitening.png.** A stack of Lomb-Scargle periodograms. The first is the RVs; each later one is what is left after removing the sinusoids found above it. The dashed line marks the strongest peak of each panel, which centres that planet's period prior. Made even with RUN\_PLANET\_FIT = false.

**RV Plot 3, rv\_plot03\_rvplot\_N\_planets.png.** The final N-planet fit at its maximum-posterior draw: the RV curve over the data, the residuals, and the phase-folded signal of each planet. Planet fits only.

**RV Plot 4, rv\_plot04\_octoplot\_N\_planets.png.** The same fit drawn as many posterior orbits over the HARPS data. The width of the bundle shows how well the data pin the orbits down.

**RV Plot 5, rv\_plot05\_corner\_N\_planets.png.** A corner plot of the N-planet posterior: each parameter's marginal distribution on the diagonal and the covariances between pairs below it. Elongated or banana-shaped blobs mean degenerate parameters; a piled-up eccentricity at the 0.7 prior cap suggests the data want a more eccentric orbit than the prior allows.

**RV Plot 6, rv\_plot06\_corner\_K-1\_vs\_K\_planets.png.** A two-series corner plot. Blue is the previous model, orange the final model, so you can see how adding the last planet moves the other planets' parameters. If the blue and orange distributions agree, the added planet is not disturbing the rest.

**RV Plot 7, rv\_plot07\_rvplot\_N\_planets\_circular.png (or …\_eccentric.png).** The eccentricity test: the same N-planet fit as Plot 3 but with the opposite eccentricity setting, so the default free-eccentricity run draws the circular fit. Made when RUN\_ECCENTRICITY\_TEST = true.

**RV Plot 8, rv\_plot08\_corner\_circular\_vs\_eccentric.png.** A two-series corner plot. Blue is the circular fit, orange the free-eccentricity fit. It shows what allowing eccentricity does to the periods, masses and the rest.

**RV Plot 9, rv\_plot09\_rvplot\_K\_planets.png.** The extra-planet test: the (N+1)-planet fit drawn like Plot 3, with the extra planet seeded at the strongest residual peak. Made when RUN\_EXTRA\_PLANET\_TEST = true.

**RV Plot 10, rv\_plot10\_corner\_N\_vs\_K\_planets.png.** A two-series corner plot. Blue is the N-planet fit, orange the (N+1)-planet fit. Check whether the extra planet leaves the first N planets unchanged.

### HK plots (Part B, Ca II H&K activity)

**HK Plot 1, hk\_plot01\_timeseries.png.** Nightly log R'HK (top) and the RV (bottom) on the same time axis, with calendar-year means in red. A slow trend or cycle in R'HK shows as a rising and falling line.

**HK Plot 2, hk\_plot02\_histogram.png.** The distribution of the nightly log R'HK. A wide or skewed histogram means real activity variation (a cycle or flares); one close to the measurement error means a quiet star. The caption gives the median error in dex.

**HK Plot 3, hk\_plot03\_periodogram\_rhk.png.** A Lomb-Scargle periodogram of the nightly log R'HK. Red lines are false-alarm levels, the shaded band is the rotation window (ROT\_RANGE), the grey dash-dot line is one year (a seasonal sampling alias), and the strongest peaks are labelled.

**HK Plot 4, hk\_plot04\_rotation\_fold.png.** log R'HK folded on the best period in the rotation window, after a ±180 d running median is removed to take out the long-term cycle. The red points are ten-bin means. A clean modulation means a real rotation signal; the caption says whether the peak is significant.

**HK Plot 5, hk\_plot05\_seasonal\_means.png.** Calendar-year means of log R'HK (blue) and RV (red), with the Pearson r over the seasons. Be careful: the yearly RV means of a planet host mostly reflect which orbital phases each season sampled. The residual test in the combined plots is the better check. Made only if enough seasons have means.

**HK Plot 6, hk\_plot06\_rv\_vs\_rhk.png.** Nightly RV against log R'HK, coloured by year, with the Spearman rho, its approximate significance and a least-squares slope in m/s per dex. A planet signal does not correlate with R'HK; activity often does. Made only if enough nights have both.

**HK Plot 7, hk\_plot07\_indicator\_correlations.png.** Bars of the rank correlation of log R'HK with the RV and with the other indicators in the table (CRX, dLW, H-alpha, Na D1, BIS, FWHM). Dashed lines mark about 3 sigma for the smallest sample. Indicators that track R'HK but not the RV show that activity is not what drives the RV.

**HK Plot 8, hk\_plot08\_periodogram\_comparison.png.** The periodograms of log R'HK (blue) and the RV (red), each divided by its maximum, with dashed red lines at the four strongest RV peaks. An RV peak that coincides with an R'HK peak, or with half its period, is more likely activity than a planet.

### Combined plots (Part C, RV and activity together)

**Combined Plot 1, combined\_plot01\_timeseries\_residuals.png.** One time axis with three panels: log R'HK (top), the RVs with the planet model (middle) and the residuals after removing the planets (bottom). If activity drove what is left, the bottom panel would follow the top one.

**Combined Plot 2, combined\_plot02\_residual\_vs\_rhk.png.** The planet-removed RV residuals against log R'HK, with the Spearman rho, nights and significance, and the slope. The caption turns the slope into the amount of activity-correlated RV in m/s next to the residual scatter.

**Combined Plot 3, combined\_plot03\_indicator\_correlations.png.** Spearman correlations of log R'HK and the other indicators with the RV before (blue) and after (red) removing the planets. Dashed lines mark about 3 sigma. Removing a large planet signal lets a small activity correlation show; a correlation that was there before and vanishes afterwards was the planets' phase sampling.

**Combined Plot 4, combined\_plot04\_periodograms.png.** Periodograms of log R'HK (blue), the raw RV (black) and the residuals (red). Dashed black lines are the fitted planet periods, an orange dotted line is the next periodogram peak (the extra-planet candidate), blue dotted lines are the rotation period and its half and third (when the rotation peak is significant), and grey is one year. Residual power that lines up with blue peaks points to activity.

**Combined Plot 5, combined\_plot05\_noise\_budget.png.** The RV noise budget: the raw scatter (dominated by the planets), the fitted jitter, the residual scatter and the R'HK-tracking amplitude, which shows how much of the residual activity could explain.

## Limits and troubleshooting

**What the verdicts can and cannot tell you**

- The checks are screening heuristics. They flag signals that look like activity, aliases, harmonics or dynamically doubtful orbits; they do not prove that a signal is a planet or that it is not.
- The planets are fitted as independent Keplerians. In a resonant system (GJ 876 is the standard example) planet-planet interaction makes that approximate, so residuals can contain real dynamical signal.
- The extra-planet ln Bayes factor is not corrected for the look-elsewhere effect, and a stepping-stone ln Z is good to a few tenths. A positive ln BF with a weak peak or an ORBIT-OVERLAP verdict is a lead to follow up, not a detection. On HD 82943, planets b and c came out solid. Splitting HARPS at the fibre upgrade raised ln Z(2 planets) from -739.6 to -649.5 and cut the pooled jitter from 3.5 to 2.3 m/s (fitted offset +10.3 m/s). A third planet seeded from the residuals stayed unconfirmed: its ln BF moved with the seed (+5.7, +9.2, +14.1) and its eccentricity hit the 0.7 cap.
- The eccentricity prior is Uniform(0, 0.7). If a posterior piles up at 0.7, the cap is shaping the answer.
- Version 1.1.13 has been run to completion on one star, HD 82943, and every section produced output. Other stars have not been tested, so check the first run of each new feature against your own judgement.

**Problems you may see**

| Symptom | What it means and what to do |
| --- | --- |
| A warning about MethodError(copy, SplittableRandom) from Pathfinder | Harmless. The sampler falls back to another initialisation (BBO) and carries on. |
| "invalid log density Inf" during a fit | The mass prior is now bounded to prevent this. If it still appears, send the log. |
| The star is not found in table1 | The run continues without table1 values; the rotation limit then comes from the hand-typed spectral type and mass. |
| Too few nights of R'HK | The run stops the activity part with a message (the minimum is MIN\_NIGHTS). Check the star has Ca II H&K data in the table. |
| Changed a setting but nothing changed, or warnings about redefined constants | You are at the Julia prompt and the REPL kept the old values. Quit Julia and start again, or run from a terminal. |
| One thread despite THREADS = "auto" | Interactive sessions cannot add threads. Set julia.NumThreads in VSCode, start with julia --threads=auto, or set JULIA\_NUM\_THREADS. See "Running it". |
| First run is slow | About 3 minutes of package precompilation and the 120 MB table download. Later runs reuse the cache. |
| An FFT or AppleAccelerate error | Set USE\_APPLE\_ACCELERATE = false. |
| Packages reinstall on every star | Set USE\_TEMP\_ENV = false. |
| A star name is not found | Use TARGET\_INDEX with the number from the printed list, or TARGET\_OVERRIDE with a name from the table. |

Anything not listed here: keep the full log (see "Running many stars") and send it along with the summary file.

## References

These citations were written from memory and have not been checked against the publications. Please verify authors, volume and page numbers before you cite them.

**Data**

- Trifonov, T. et al. 2020, A&A 636, A74. HARPS-RVBank: radial velocities re-derived with SERVAL from the public HARPS archive.
- Perdelwitz, V. et al. 2024, A&A 683, A125 (arXiv:2312.06586). Ca II H&K activity (R'HK) for the HARPS archive; VizieR catalogue J/A+A/683/A125, whose table4 and table1 the script uses.
- Ochsenbein, F., Bauer, P., Marcout, J. 2000, A&AS 143, 23. The VizieR catalogue service.

**Orbit fitting**

- Thompson, W. et al. 2023, PASA. Octofitter: a Julia package for Bayesian orbit fitting.
- Surjanovic, N. et al. 2023, arXiv:2308.09769. Pigeons.jl: parallel tempering sampling in Julia.
- Xie, W. et al. 2011, Systematic Biology. Marginal likelihood by stepping-stone sampling.

**Periodograms**

- Lomb, N. R. 1976, Ap&SS 39, 447.
- Scargle, J. D. 1982, ApJ 263, 835.
- Baluev, R. V. 2008, MNRAS 385, 1279. False-alarm probability for the generalised Lomb-Scargle periodogram.

**System-specific**

- Correia, A. C. M. et al. 2010, A&A. Cited in the script for the precession of the GJ 876 system.

The script's header comments and changelog give the version history and the reasoning behind each heuristic.
