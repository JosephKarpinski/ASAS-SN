# Octofitter tutorial modules — required conventions & fixes

Attach this file **together with the latest module (`OctofitterG23H.jl` v1.0.1)** as the template; it already contains every fix below (use `OctofitterRVMulti.jl` v1.0.1 as the reference for RV-specific items 13 and 21).

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

## Status

- **Done (12):** QuickStart, RelAstrom, ObsPrior, Coplanar, ThieleInnes, RV1, RVGP, RVMulti, RVRel, PMA, Hipparcos, G23H.
  - Hipparcos: 2.8 min. The catalog check (Nielsen test) lands within 0.11σ of the catalog on all 5 parameters. For 51 Eri b, the page's b mass line reads 0.67 / 10.67 / 94.0 M_jup (16/50/84%); the share of samples above 50 M_jup changed from about 25% to 1.6% between two identical runs (min(α) ≈ 1e-30 both times), so that heavy group's weight isn't reliable at n_rounds=8. The second run gives 0.73 / 11.18 / 37.4 M_jup.
  - PMA: 9.7 min with 4 threads. Λ = 9.06 and log(Z₁/Z₀) = −24.8; the docs give 9.06 and −24.1.
  - G23H: 3.0 min at the page's n_rounds=6 (64 samples). There are 12 channels; Λ = 10.8 and still rising, while log(Z₁/Z₀) = −265.2 was stable from round 4. HIP 384 b is poorly constrained: median mass 27 M_jup (0.3–95), P 11 yr (2.4–384).
- **Next:** Gaia DR4 (https://sefffal.github.io/Octofitter.jl/dev/gaia-dr4-prerelease/), then Gaia DR4 Simulation and the G23H Full Example. Check each page's hidden `# hide` setup (docs source `docs/src/<page>.md`) for the offline `catalog=`/`forecast_table=`/`hipparcos=` inputs before writing the module.
