# Octofitter tutorial modules — required conventions & fixes

Attach this file **together with the latest module (`OctofitterInterferometryTutorial.jl` v1.0.1)** as the template; it already contains every fix below (use `OctofitterRVMulti.jl` v1.0.1 as the reference for RV-specific items 13 and 21).

## Context

- macOS, 4 cores, VSCode Insiders, Julia 1.12.7 (juliaup).
- The global environment has Octofitter **v8.2.4**, but the dev docs describe **v9**.
- Scripts live in `/Volumes/SSD2/MyJulia/Octofitter.jl-main/`. Don't activate that folder as an environment.
- Each tutorial becomes one self-contained `.jl` file that runs with VSCode's **"Execute active File in REPL"**.

## Environment

1. **Set up the environment before any `using`.** Put `import Pkg` at top level, not inside a `let`. The set-up block then works like this:
   - If Octofitter v8 is already loaded, stop and tell the user to restart the REPL.
   - If Octofitter v9 is already loaded, add any missing packages with `Pkg.add(...; preserve=Pkg.PRESERVE_ALL)`.
   - If the active environment already has v9 and every required package, use it.
   - Otherwise run `Pkg.activate(; temp=true)`, or `./octofitter_v9_env` when `ENV["<PFX>_ENV_MODE"]="local"`. Then add `Pkg.PackageSpec(name="Octofitter", version="9")`, and the same for `OctofitterRadialVelocity` when it's needed.
2. **Check versions inside the module:** stop with a clear message if `pkgversion(Octofitter) < v"9"`.

## Julia and REPL pitfalls

3. Use no Printf macros outside the module; `using Printf` is only inside it.
4. Call file-location macros with parentheses: `@__FILE__()` and `@__DIR__()`. Without them, `@__FILE__ ||` fails to parse.
5. **Define nothing constant in Main.** Re-running the file then fails with an "invalid assignment to constant" error. Keep the load timer inside the module, put the auto-run check in a `let`, and set only the result with `global result = ...`.
6. **Auto-run block after the module:**
   - Run when `abspath(PROGRAM_FILE) == @__FILE__()` or `isinteractive()`, unless `ENV["<PFX>_AUTORUN"]="false"`.
   - Call the module qualified (`Mod.run_x()`), not through `using .Mod`.
   - End the file with `nothing`.
   - Without this block, "Execute active File in REPL" only loads the module and runs nothing.
7. **Other syntax traps:**
   - Don't put a `do` block inside a ternary expression.
   - Don't name a helper function `sample`; that name clashes with the one exported by StatsBase and AbstractMCMC.
   - Use `import StatsBase`, `import MCMCChains` and `import Statistics`, and qualify their functions.
   - `@sprintf` needs one literal format string. Joining two strings with `*` inside the call fails when the file loads ("First argument to `@sprintf` must be a format string"). Append any extra text after the call instead.

## Building models

8. **Build every `@variables` block with `@eval` and `$` interpolation**, including blocks containing only fixed values. It has worked in every module so far:
   ```julia
   vars = @eval @variables begin
       a ~ $a_prior
       epoch = $ep
   end
   ```
9. Build fresh `Body` and observation objects for each `System`, and don't share them between models.
10. Don't read `Body` fields such as `.name` in debug prints.

## Sampling

11. **Threads can't be added to a running Julia session.** For Pigeons fits in a one-thread REPL, start a child process instead:
    - The command is `julia --threads=auto --project=$(Base.active_project()) file`, run through `addenv(cmd, "<PFX>_CHILD"=>"1")`.
    - Stream the child's output with `run(pipeline(cmd; stdout, stderr))`. The child runs with `show_plots=false`.
    - Afterwards the parent loads the saved FITS chains back into `result.chains`.
    - Print only `join(base.exec, " ")`, never the `setenv(...)` form, which dumped the whole environment into the log.
    - The same applies when the child fails: never `showerror` a `ProcessFailedException` (it prints `setenv(...)` too). Report `p.exitcode` for each of `err.procs`.
    - After the child finishes, the parent loads the saved PNGs (`Makie.FileIO.load`, `image!(ax, rotr90(img))`, `DataAspect()`, hidden decorations) and `display`s each one, so the plots reach the VSCode plot pane. Setting `<PFX>_SHOW_PLOTS=false` turns this off.
    - HMC-only tutorials stay in the REPL, since a single chain gains nothing from threads. Their Pigeons option, set through `<PFX>_SAMPLER="pigeons"`, uses the child process.
12. **Call `initialize!(model)` before every `octofit_pigeons`.** Otherwise Pathfinder fails with `MethodError(copy, SplittableRandom)` and every chain starts from a single fallback point. Do the same for prior-only models.
13. **Celerite:**
    - Use `import OctofitterRadialVelocity.Celerite`, never `using Celerite`.
    - Build the model with `LogDensityModel(sys, autodiff=AutoFiniteDiff())`, which needs DifferentiationInterface and FiniteDiff.
    - Run its `initialize!` inside `Logging.with_logger(Logging.ConsoleLogger(stderr, Logging.Error))`. The warning otherwise prints a type signature of about 100 kB.
14. **Evidence:** use `Pigeons.stepping_stone(pt)` and `Pigeons.global_barrier(pt)`. For log Z0, use `LogDensityModel(Octofitter.prior_only_model(sys, exclude_all=true))`.

## Output and plots

15. **Output location:** save everything to `SCRIPT_DIR = @__DIR__()`, falling back to `pwd()`, with a file prefix per module. Run `octoplot` and `octocorner` inside `cd(f, outdir)`.
16. **Dark theme:** `#111111` background and `#EEEEEE` text. In the Legend theme, use `backgroundcolor`; `bgcolor` is deprecated.
17. **Corner and pair plots use the light theme**, via `with_theme(() -> octocorner(...), Theme())`. PairPlots draws single-series contours and labels in black, which vanish on a dark background.
18. **Match PairPlots' colour order** in any custom comparison plot: first series blue `#0072B2`, second orange `#E69F00`.
19. **Report Ω folded into a 180° window centred on its average direction.** Relative astrometry can't distinguish Ω from Ω+180°, and a plain `mod 180` splits a peak that sits near 0°/180°.
20. **Figures using `DataAspect()`** should be sized to the data's shape.
21. **When comparing with a simulated truth**, match each fitted planet to the true planet with the nearest period.

## Debugging and structure

22. **Debug helpers:**
    - `dbg` prints `[TAG +t]` lines.
    - `stage` times a required step, names it if it fails, and rethrows.
    - `soft_stage` logs ⚠ on failure and continues; use it for all optional steps.
    - Also include an environment banner (versions, threads, project, pwd, SCRIPT_DIR), a posterior summary, and a list of output files.
    - Save chains with `savechain` and check them with `loadchain(p; model)`.
23. **File header:** a versioned block describing the file, with a changelog. Settings use `ENV["<PFX>_AUTORUN|ENV_MODE|RELAUNCH|THREADS|SAMPLER|MODELS|ROUNDS"]`.
24. **Downloaded data** is saved next to the script and reused, shared between modules where possible (for example `rv1_k2-131.txt`).
25. **Heartbeat for silent network steps:** wrap downloads and queries in `with_heartbeat` (a `Threads.@spawn` task that prints a line every 30 s with the elapsed time and how much the DataDeps folders and the script folder have grown). Growing sizes mean a slow download; no growth means a stuck request.
26. **Split construction into timed stages** (bodies, observation, priors, `System`), so the log names the exact step that stalls or fails.

## Absolute astrometry (Gaia/Hipparcos): data sources

27. **In v9, `HGCAObs` is a wrapper around `G23HObs`.** By default it reads the G23H catalog (~14 GB DataDep), a DR2 sidecar (~300 MB) and a live GOST query. Don't let a tutorial trigger that download by accident.
    - **Default "subset" mode:** do what the docs build does. Pass `catalog = Arrow.Table("G23H-test-subset.feather")` and `forecast_table = Table(epoch=jd2mjd.(gost.ObservationTimeAtBarycentre_BarycentricJulianDateInTCB_), scanAngle_rad=gost.scanAngle_rad_, parallaxFactorAlongScan=gost.parallaxFactorAlongScan)`, where `gost = CSV.read("GOST-<ra>-<dec>-dr3.csv", Table; normalizenames=true)`. Both files come from the Octofitter repo (`test/` and `docs/src/`) at a pinned commit. They're downloaded once next to the script and shared between modules.
    - The subset file has 4 rows: Gaia DR3 2738776816458107136 (HIP 384), 756291174721509376 (HIP 51658, HD 91312), 5164707970261890560 (HIP 16537) and 6166183842771027328 (HIP 65808). It has all 115 columns, including the DR2 sidecar column, so no sidecar download is needed.
    - A `<PFX>_CATALOG` setting chooses between `"subset"`, `"full"` and a local path.
    - Add `Arrow` and `CSV` to the bootstrap's `required` list. Load `Downloads` with `import`.
28. **Stale DataDep folders:** DataDeps treats any existing folder as a finished download, so an interrupted download fails later with "No such file". The folders are in `~/.julia/scratchspaces/124859b0-ceae-595e-8997-d05f6a7a8dfe/datadeps/`, not `~/.julia/datadeps`. Check for empty or incomplete folders at startup and print the `rm(...)` command to clear them. The G23H_Catalog folder on this Mac is empty; delete it before running any fit that uses `OCTOPMA_CATALOG=full`.
29. **The Hipparcos IAD DataDep** (~332 MB zip, 826 MB unpacked) downloads on the first `HGCAObs`/`G23HObs` call, in every mode. It's now installed on this Mac, along with DE440_Ephemeris (114 MB).
30. **`gaia_plx` queries the Gaia TAP service.** If the query fails, fall back to `truncated(Normal(μ, σ), lower=μ-10σ, upper=μ+10σ)` built from the catalog row's `parallax` and `parallax_error`.
31. **Useful `G23HObs` fields:** `.table` (epoch, start_epoch, stop_epoch, pm, σ_pm, kind), `.catalog` (a NamedTuple row), `.hip_table` and `.gaia_table`. `dotplot(model, chain; mode=:separation|:period)` works as expected.
32. **`HipparcosIADObs`:** the default observation name is `"Hipparcos IAD"`, so chain columns look like `Hipparcos_IAD_iad_Δra` (find them by suffix). Starting values use the key `observations=(; Hipparcos_IAD=(; iad_Δra=0.0, …))`. Useful fields are `.hip_sol` (`plx`, `e_plx`, `pm_ra`, `pm_de`, `e_ra`, `e_de`, `e_pmra`, `e_pmde`) and `.table` (`epoch`, `res`, `sres_renorm`, `scanAngle_rad`, `parallaxFactorAlongScan`, `reject`). Use `Octofitter.hipparcosplot(model, chain)` for the residual plot. Pass the explorer as `explorer=Pigeons.SliceSampler()`.
33. **`octocorner(...; small=true)` keeps only orbital elements.** A model with no orbit (star only) loses every column and fails, so use `small=false` for such a model.
34. **Isolated modes:** when Pigeons' `min(α)` column is ≈ 0, the weight of an isolated mode (for example 51 Eri b piled at the 100 M_jup prior edge) isn't reliable. Report each mode separately, and offer more rounds and an `n_chains` setting through ENV.
35. **`G23HObs` with every channel:** for HIP 384 this gives 12 rows (`iad_hip`, `ra/dec_hip`, `_hg`, `_dr2`, `_dr32`, `_dr3`, `ueva_dr3`), with the observation named `"G23H"` so columns are `G23H_σ_AL` and so on. The `iad_hip` row has NaN in `pm` and `σ_pm`, so filter the proper-motion rows with `r"^(ra|dec)_(hip|hg|dr2|dr32|dr3)$"`. The page's shortcut, `model.starting_points = fill(collect(model.link(θ)), 100)` with `θ` from `guess_starting_position(model, 10_000)[1]`, works in v9, and `octofit_pigeons(model; n_chains=32, explorer=Pigeons.SliceSampler(), n_chains_variational=0, variational=nothing, multithreaded=true)` runs. Use `Octofitter.jd2mjd` for the GOST epochs. HIP 384's GOST file is `GOST-1.1927097109938027-1.5368044203832403-dr3.csv` in the repo's `docs/src/`.
36. **Linear period–mass pairplots are unreadable when the posterior spans decades.** Add a log₁₀ version next to the page's linear one.
37. **Gaia DR4 pre-release (`GaiaDR4AstromObs`):**
    - The page is not executed when the docs build (plain `julia` blocks, not `@example`), so check each API against the source (`src/likelihoods/gaia-dr4.jl`, `src/nss.jl`, `ext/OctofitterMakieExt.jl`). All of them worked in v9.
    - Required columns are `epoch`, `scan_pos_angle` (**degrees**), `parallax_factor_al`, `centroid_pos_al` and `centroid_pos_error_al`. The observation accepts any Tables.jl table.
    - The page's bootstrap reduction needs no DataFrames: sort stably by `transit_id`, use `Random.Xoshiro(source_id)`, and take `median(rand(rng, pos, n))` 256 times.
    - Use `target=Photocentre, ref=Barycentre`, with `flux = 1.0` on the host and `0.0` on the companion. The frame variables live in the observation block.
    - Starting values go in `initialize!(model, (; plx, bodies=(A=(;mass), b=(;mass, a, e, i)), observations=(GaiaDR4=(; …),)))`. Use `Octofitter._query_gaia_dr3(gaia_id=…)` for the DR3 seed.
    - Pigeons settings: `n_chains=16, n_chains_variational=0, variational=nothing`.
    - `gaiastarplot!(fig[i,j], model, chain, idx; axis=(;…))` draws into one cell of a grid.
    - The data are the CSVs in `docs/src/` (`gaia4_…`, `gaia_bh3_…`, `hd114762_…`).
38. **`nss_to_model_chain` builds a photocentre-only model:** the host carries all the mass, `b_mass` is 0, `b_a` is α/plx, and there is no `P` column. Compare the NSS row's `period` and `eccentricity` (± their `_error`) and the photocentre α with the fit. Don't compare its `b_mass`, or a period derived from its `a` and mass.
39. **Tensions against published values:** state them in combined σ (the fit's own spread and the published error), not in units of the published error alone.
40. **Simulating DR4 (`gaia_dr4_transit_template`, `g23h_scan_uncertainty`, `generate_from_params`):**
    - The GOST cache is looked up by name, `GOST-<ra>-<dec>-dr4.csv`, in the working directory, with `ra` and `dec` printed exactly as passed. Download the repo's file and call the function with the file's own coordinates inside `cd(outdir)`.
    - `g23h_scan_uncertainty(; gaia_id, catalog=<NamedTuple row>)` accepts the page's hidden row. It returns `σ_AL`, `σ_att`, `σ_calib`, `σ_formal`, `n_ccd`, `σ_transit_formal` and `σ_transit_true`.
    - `generate_from_params(sys, θ; add_noise=true)` reads `θ.observations` verbatim, and missing entries become 0. Seed it with `Random.seed!` so reruns give the same data.
    - `construct_system(model, θ)` and `construct_system(model, chain, i)` exist, and PlanetOrbits is re-exported (`raoff`, `decoff`, `orbitsolve`).
41. **When comparing a fit with an injected truth, report the truth's percentile in the posterior as well as the σ offset.** Near a prior boundary (for example i ≈ 0°, where `Sine()` gives P(i < 10°) = 1.5%), a Gaussian σ offset overstates the disagreement.

42. **Pages written as command-line scripts (ArgParse), such as the G23H Full Example:** don't add ArgParse. A small top-level `let` before the bootstrap maps the page's flags to `<PFX>_*` ENV settings (for example `--host-mass 1.61` → `OCTOG23X_HOST_MASS`, repeatable `--rdb-rv` → a comma-separated list). It runs before the bootstrap because a flag can change which packages are needed (`--dace-rv` adds PythonCall and CondaPkg). A relaunched child inherits the ENV. The page's placeholder IDs (HIP 12345) aren't real, so default to a target in the 4-row subset that has a cached GOST file.
43. **Marginalising over the companion count** (`n_planets ~ truncated(NegativeBinomial(), upper=N)`, `mass = mass′ * (system.n_planets >= k)`): Octofitter handles the discrete variable. Report the prior and posterior probability of each count and the Bayes factor (posterior odds ÷ prior odds), and condition each companion's summary on its presence, since `b_mass` is 0 in the absent draws. `Octofitter.MCMCChains.setinfo` and `Octofitter.StatsBase` can be reached without adding either package.
44. **Page quirks in the G23H Full Example:**
    - `--circular` is parsed but never used; the module fixes e = 0 and ω = 0.
    - The RV clip is `1.4826 * StatsBase.mad(rv)`, but `mad` is already normalised, so "3σ" is really about 4.45σ. The module keeps this behaviour and logs it.
    - DACE and RDB `rjd` is BJD − 2 400 000, so MJD = rjd − 0.5; the page treats rjd as MJD.
    - `RadialVelocityObs` is in the Octofitter core; call `Octofitter.rvplot` qualified. `octoplot` and `rvplot` return an `Octofitter.OctoPlotResult`, so save `.figure`.
45. **No Julia in the build workspace** (the Julia download servers are blocked). Before delivering, check the file structurally: block openers must balance their `end`s once strings, comments and brackets are stripped, and every `@sprintf` must have a literal format and the right number of arguments. Check APIs against a shallow clone of the Octofitter repository, which github.com allows.

46. **Reading Pigeons diagnostics in code:** `min(α)` is `minimum(Pigeons.swap_prs(pt))`, Λ is `Pigeons.global_barrier(pt)`, and log Z is `Pigeons.stepping_stone(pt)`. Base any "mixing is suspect" note on the actual value; a caution printed on every run misleads. Two warnings seen when continuing a run with `increment_n_rounds!` are harmless: Pigeons' "The set of successful reports changed", and "arr2nt is not type-stable", which comes from the discrete `n_planets` variable in the page's model.

47. **Unregistered extension packages** (OctofitterImages, OctofitterInterferometry): add them in the bootstrap with `Pkg.PackageSpec(url="https://github.com/sefffal/Octofitter.jl", subdir="OctofitterImages", rev=<pinned commit>)`. The pinned commit carries Octofitter 9.0.0, matching the registered version. The first install clones the repository. Don't name the script after the package: the file and module are `OctofitterImagesTutorial`.
48. **Drawing images on octoplot's sky panel** (`res.axes.sky.sky`): the panel switches from mas to arcsec once its extent passes 1500 mas, so read `ax.xlabel[]` and scale the image axes (×1e-3 for arcsec). RA increases to the left: flip the image rows (`[end:-1:begin, :]`) and `recenter` again, as the page does. For your own axes, pass reversed limits (`limits!(ax, h, -h, -h, h)`) with `xreversed=true`, because `xlims!` given low-then-high silently clears `xreversed`. Take the maximum across images with NaN skipped, since a plain `maximum` spreads NaN from masked areas.
49. **Predicted positions from a posterior:** `s = construct_system(model, chain, i)`, `traj = orbitsolve(s, epochs)`, then `raoff.(traj, :b, :A)` and `decoff.(traj, :b, :A)` (in mas). With AstroImages: `load(file, :)` reads every extension, `recenter(img)` puts index [0,0] at the centre, `dims(img, 1)` gives pixel offsets, and `OctofitterImages.contrast(img)` gives the annulus 1σ curve that `ImageObs` uses.
50. **Structural check:** `begin` inside an index (`[end:-1:begin, :]`) is not a block opener; the balance checker skips it.

51. **`octocorner(...; small=true)` keeps only the body variables a, e, i and mass.** Add others back with `includecols=[:b_flux_H]` (for example, the photometry the images page shows in its corner plot), and use `excludecols` to drop columns. Harmless warnings in image fits: "This model has priors that cannot be sampled IID" (from `UniformCircular`), OptimizationBase's "Unrecognized stop reason" during `initialize!`, and the parent's "precompiled but different versions are currently loaded" after the bootstrap's `Pkg.add`.

52. **Deriving a flux from the mass (Sonora Bobcat):**
    - `Octofitter.sonora_cooling_interpolator()` maps (age in Myr, mass in M⊙) to T_eff. `Octofitter.sonora_photometry_interpolator(:MKO_H)` (or `:Keck_L′` and others) maps (T_eff, mass in M⊙) to an **absolute magnitude**.
    - The grids are a ~1 MiB DataDep called `SonoraBobcatEvoPhot`, which holds `evolution_tables/` and `photometry_tables/`.
    - Interpolate the functions into `@eval @variables` with `$` (`tempK = $ct(system.age, mass)`), and convert to a linear contrast with `10^(-0.4*(mag − host))`.
    - Outside the grid they return a typed NaN. When clamping, wrap the replacement value in `oftype(mag, …)` so ForwardDiff doesn't get a Union type.
    - A system with no data uses `observations=()`, not `[]`.
    - For quick prior draws, `model.arr2nt(model.sample_priors(rng))` returns a nested named tuple; read a body's values from `.bodies.b.<var>`.
    - `PhotometryObs(Table(phot=[…], σ_phot=[…]); target=b, band=:H, name=…)`.
53. **`@sprintf` check:** remove `%%` before counting specifiers; otherwise text like "100%% of" is counted as a format specifier.

54. **A closure interpolated into `@variables` makes `display(sys)` print all the data it captured,** because a closure prints with its captured values. The Sonora interpolators printed their full grids, over 1 MiB, and VSCode cut the log. Wrap such functions in a callable that prints as its name (`struct Labelled{F}; name::String; f::F; end`, `(l::Labelled)(args...) = l.f(args...)`, `show` gives the name). The system display then reads `sonora_cooling(system.age, mass)`. In models with no orbit data, leave the prior-only a, e and i out of `octocorner` with `excludecols`.

55. **`InterferometryObs`** (OctofitterInterferometry, unregistered; add it as in item 47 with `subdir="OctofitterInterferometry"`):
    - It reads the OI-FITS files named in a `filename` column when it is built, so pass absolute paths.
    - The table gains `u`, `v` (inverse wavelengths, baseline × channel), `cps_data`, `dcps` (degrees), `eff_wave`, and `vis2_data`/`dvis2` when `use_vis2` is set.
    - The host is an ordinary source: give it `flux_K = 1.0` and list it in `targets=(A, b)`, with `ref=A` and `band=:K`.
    - `platescale` is now a divisor.
    - `construct_system(model, chain)` with no index returns one system per draw.
    - `mjd2date` comes from PlanetOrbits 1.0, re-exported by Octofitter.
    - The data are `examples/AMI_data/Sim_data_20{23_1,23_2,24_1}_.oifits` (43 kB each).
    - Balance checker: `end` inside any `[ ]`, even within a call like `x[mod1(k, end)]`, is an index, not a block end.

56. **Plot and log details from the interferometry run:**
    - When a prior spans decades that the posterior never reaches (a contrast prior N(0, 0.1) against a posterior near 5e-4), a shared linear axis hides one of the two. Show the page's histogram zoomed in, plus log₁₀ contrast with the prior histogrammed from `rand`.
    - Simulated exposures can have identical u-v coverage. Size markers per epoch so they show as rings, and log when coverage is identical.
    - A per-exposure "rms/σ" ratio near 1 does not mean "no signal". Use the χ² of the closure phases against zero (`ccdf(Chisq(N), χ²)`) across all epochs.
    - `mjd2date` returns a DateTime; take the first 10 characters for labels.

## Status

- **Done (18):** QuickStart, RelAstrom, ObsPrior, Coplanar, ThieleInnes, RV1, RVGP, RVMulti, RVRel, PMA, Hipparcos, G23H, DR4, DR4Sim, G23HExample, Images, MassPhotometry, Interferometry.
  - Hipparcos: 2.8 min. The catalog check (Nielsen test) lands within 0.11σ of the catalog on all 5 parameters. For 51 Eri b, the page's b mass line reads 0.67 / 10.67 / 94.0 M_jup (16/50/84%); the share of samples above 50 M_jup changed from about 25% to 1.6% between two identical runs (min(α) ≈ 1e-30 both times), so that heavy group's weight isn't reliable at n_rounds=8. The second run gives 0.73 / 11.18 / 37.4 M_jup.
  - PMA: 9.7 min with 4 threads. Λ = 9.06 and log(Z₁/Z₀) = −24.8; the docs give 9.06 and −24.1.
  - G23H: 3.0 min at the page's n_rounds=6 (64 samples). There are 12 channels; Λ = 10.8 and still rising, while log(Z₁/Z₀) = −265.2 was stable from round 4. HIP 384 b is poorly constrained: median mass 27 M_jup (0.3–95), P 11 yr (2.4–384).
  - DR4 (Gaia-4b): 2.8 min, 93 transits as on the page. P = 578.0 (574.4–581.7) d and m_b = 11.0 (10.4–11.6) M_jup, against Stefansson's 571.3 ± 1.4 d and 11.8 ± 0.7 M_jup. min(α) ≈ 1e-57, so more rounds are needed.
  - DR4Sim (part 1): 2.8 min, 191 transits, σ_transit_true 0.1525 mas. Everything is recovered within about 1.2σ except i (36° against 0.57°, a prior-volume effect). min(α) stayed at 0, so try OCTOSIM_CHAINS=32. Parts `formal` and `binary` haven't been run yet.
  - G23HExample (HIP 51658, 12 channels, 8 rounds): 4.7 min. log(Z₁/Z₀) settled at about −476 from round 5; Λ reached 12.8 and was still rising slowly; min(α) = 0.09 at the end. n_planets = 1 in all 256 samples (BF > 512, the sample-count limit). b is about 0.9 (0.34–1.48) M⊙ at 77 (55–96) AU with e 0.50. That sits inside the PMA fit's wider range (444 [158–654] M_jup, 19 [1.9–80] AU). The mass posterior reaches the mass′ prior cap (M_host = 1.61 M⊙). v1.0.1 now reads min(α) instead of always printing the caution.
  - Images (de-orbiting): 2.4 min with 4 threads, 1024 samples. min(α) = 0.63, Λ = 4.5, log(Z₁/Z₀) = 65.0. The best-fit position sits on the planet's blob in all 5 images. a = 14.4 (13.7–15.6) AU, e = 0.18, i = 38 ± 3°, P = 39 yr. flux_H = 4.68 ± 0.31, 1.8 prior σ above the prior's 3.8, while the per-epoch readouts are about 5–5.8, so the informative prior pulls the flux down. The page's SNR is 15.2, against 7.6 from the prior alone. The b_e autocorrelation drops below 0.1 at lag 2, much better than the page's warning suggests. v1.0.1 adds flux_H to the corner plot.
  - MassPhotometry: 1.7 min (HMC, 2000 + 2000). b mass = 11.8 (11.5–12.0) M_jup, T_eff = 1378 K, age = 39 ± 5 Myr. The fitted contrasts sit −0.71σ (H) and +0.34σ (L′) from the data. The posterior cuts off sharply above about 12.3 M_jup, where the cooling track jumps near 13 M_jup (deuterium burning). At 40 Myr the H grid covers 0.5–35 M_jup; the clamped model (1–150 M_jup prior) gives the same result, with 0% of samples clamped. v1.0.1 stops the grids from being printed into the log.
  - Interferometry (NIRISS-AMI, simulated): 2.8 min with 4 threads (Pigeons 58 s). Contrast 4.6e-4 ± 0.6e-4 (Δmag 8.35), page SNR 7.2 (median/IQR 5.3). a = 2.07 ± 0.05 AU, e = 0.03, i = 34 ± 6°, P = 2.4 yr. min(α) = 0.55, Λ = 6.1, log(Z₁/Z₀) = 176. Ω is bimodal (about 40° and 220°), as expected for relative positions with no RV. Separation and PA: 189 mas at 267°, 176 mas at 300°, 204 mas at 55°. θ at MJD 60171 = 299.8°, matching the epoch-2 PA. Each closure phase is only about 1.2σ from zero; the v1.0.1 run measured χ² = 162 for 105 phases (p = 3e-4), with the three epochs at p = 0.02 each. The three simulated exposures have identical u-v coverage. v1.0.1 was confirmed in a second run (same results).
- **Next:** Likelihood Map (https://sefffal.github.io/Octofitter.jl/dev/fit-likemap/). After that: fit-grav-wide.
