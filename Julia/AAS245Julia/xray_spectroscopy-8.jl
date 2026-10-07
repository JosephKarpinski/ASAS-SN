# X-ray Spectral Analysis Using SpectralFitting.jl
# Script version of the AAS245Julia notebook 05-X-ray-spectroscopy
# (SpectralFitting.jl equivalent of the XSPEC "Walk through XSPEC").
#
# Run from a terminal:   julia xray_spectroscopy.jl
# Or from the REPL:      include("xray_spectroscopy.jl")
#
# Plots are saved as PNGs into ./figures (nothing relies on an interactive display).
# The first run installs packages into a project in this folder and can take a while.

# ------------------------------------------------------------------ setup
import Pkg
Pkg.activate(@__DIR__)

Pkg.Registry.add(Pkg.RegistrySpec(url = "https://github.com/astro-group-bristol/AstroRegistry"))

const DEPS = ["SpectralFitting", "XSPECModels", "Plots", "StatsPlots", "Turing",
              "CodecZlib", "Tar", "Downloads"]
let have = keys(Pkg.project().dependencies)
    todo = filter(d -> d ∉ have, DEPS)
    isempty(todo) || Pkg.add(todo)
end

using SpectralFitting, XSPECModels, Plots
using Downloads, CodecZlib, Tar
SpectralFitting.download_all_model_data()

const FIGDIR = joinpath(@__DIR__, "figures")
mkpath(FIGDIR)
savefig_(p, name) = (savefig(p, joinpath(FIGDIR, name)); p)

# Fit statistic as a single number (newer SpectralFitting returns a per-dataset vector)
chi2(r) = sum(r.stats)

# ------------------------------------------------------------------ data
url = "https://heasarc.gsfc.nasa.gov/docs/xanadu/xspec/"
xrayfile = "walkthrough.tar.gz"
tmpdir = mktempdir(tempdir(), prefix = "s54405_")
Downloads.download(joinpath(url, xrayfile), joinpath(tmpdir, xrayfile))
datadir = Tar.extract(GzipDecompressorStream(open(joinpath(tmpdir, xrayfile))),
                      mktempdir(tempdir(), prefix = "s54405_"))
data = OGIPDataset(joinpath(datadir, "walkthrough", "s54405.pha"))
@show data
# (older SpectralFitting versions also had `data.paths`; newer ones do not)

p = plot(data, xlims = (0.5, 70), xscale = :log10, label = "1E 1048.1-5937")
savefig_(p, "01_raw_spectrum.png")

# Convert to counts s⁻¹ keV⁻¹ and drop bad channels
normalize!(data)
drop_bad_channels!(data)
p = plot(data, ylims = (0.001, 2.0), yscale = :log10, xscale = :log10,
         label = "1E 1048.1-5937")
savefig_(p, "02_normalized_spectrum.png")

# ------------------------------------------------------------------ power law
model1 = PhotoelectricAbsorption() * PowerLaw()
@show model1

example = PhotoelectricAbsorption() * PowerLaw(a = FitParam(3.0))

prob = FittingProblem(model1 => data)
result0 = SpectralFitting.fit(prob, LevenbergMarquadt())
@show result0

p = plot(data, ylims = (0.001, 2.0), xscale = :log10, yscale = :log10,
         label = "1E 1048.1-5937")
plot!(p, result0, label = "Powerlaw: χ² = $(round(chi2(result0)))")
savefig_(p, "03_powerlaw_full_range.png")

# Ignore the poorly-fit high-energy range and refit (0 to 15 keV)
mask_energies!(data, 0, 15)
result1 = SpectralFitting.fit(prob, LevenbergMarquadt())
@show result1

p = plot(data, ylims = (0.001, 2.0), xscale = :log10, yscale = :log10,
         label = "1E 1048.1-5937")
plot!(p, result1, label = "PowerLaw")
savefig_(p, "04_powerlaw_0_15keV.png")

update_model!(model1, result1)
@show model1

# ------------------------------------------------------------------ goodness of fit
# Newer SpectralFitting returns (percent better, fit statistic of each simulated trial)
pcent, spread = goodness(result1; N = 1000, seed = 42,
                         exposure_time = data.spectrum.exposure_time)
println("goodness(): percent of simulations with a better fit statistic = ", pcent)
p = histogram(spread, ylims = (0, 300), label = "Simulated")
vline!(p, [chi2(result1)], label = "Best fit")
savefig_(p, "05_goodness.png")

pct_better = count(<(chi2(result1)), spread) * 100 / length(spread)
println("Percent of simulations with a fit statistic better than the data: ", pct_better)

# ------------------------------------------------------------------ flux
calc_flux = XS_CalculateFlux(
    E_min = FitParam(0.2, frozen = true),
    E_max = FitParam(2.0, frozen = true),
    log10Flux = FitParam(-10.3, lower_limit = -100, upper_limit = 100),
)
flux_model1 = model1.m1 * calc_flux(model1.a1)

# The flux model scales the power law, so freeze its normalization
flux_model1.a1.K.frozen = true
@show flux_model1

flux_problem = FittingProblem(flux_model1 => data)
flux_result1 = SpectralFitting.fit(flux_problem, LevenbergMarquadt())
@show flux_result1

p = plot(data, ylims = (0.001, 2.0), xscale = :log10, yscale = :log10,
         label = "1E 1048.1-5937")
plot!(p, flux_result1, label = "powerlaw")
vspan!(p, [flux_model1.c1.E_min.value, flux_model1.c1.E_max.value], alpha = 0.5)
savefig_(p, "06_flux_region.png")

# ------------------------------------------------------------------ alternative models
model2 = PhotoelectricAbsorption() * XS_BlackBody()
prob2 = FittingProblem(model2 => data)
result2 = SpectralFitting.fit!(prob2, LevenbergMarquadt())
@show result2

dp = plot(data, ylims = (0.001, 2.0), xscale = :log10, yscale = :log10,
          legend = :bottomleft, label = "1E 1048.1-5937")
plot!(dp, result1, label = "PowerLaw: χ² = $(round(chi2(result1)))")
plot!(dp, result2, label = "BlackBody: χ² = $(round(chi2(result2)))")

model3 = PhotoelectricAbsorption() * XS_BremsStrahlung()
prob3 = FittingProblem(model3 => data)
result3 = SpectralFitting.fit(prob3, LevenbergMarquadt())
@show result3

plot!(dp, result3, label = "Brems: χ² = $(round(chi2(result3)))")
savefig_(dp, "07_model_comparison.png")

# ------------------------------------------------------------------ residuals
function calc_residuals(result)
    r = result[1]   # only one result here; generalizes to multi-model fits
    # Newer SpectralFitting API (replaces invoke_result / r.objective / r.variance)
    y = calculate_objective!(r, r.u)
    obj, var = get_objective(r), get_objective_variance(r)
    @. (obj - y) / sqrt(var)
end

domain = SpectralFitting.plotting_domain(data)
rp = hline([0], linestyle = :dash, legend = false)
plot!(rp, domain, calc_residuals(result1), seriestype = :stepmid)
plot!(rp, domain, calc_residuals(result2), seriestype = :stepmid)
plot!(rp, domain, calc_residuals(result3), seriestype = :stepmid)
savefig_(rp, "08_residuals.png")

p = plot(dp, rp, layout = grid(2, 1, heights = [0.7, 0.3]), link = :x, xscale = :linear)
savefig_(p, "09_models_and_residuals.png")

# Same figure in one call. `plotresult` no longer exists in newer SpectralFitting;
# the current docs define this helper using residualplot!.
function plot_result(data, results...)
    p1 = plot(data,
        ylims = (0.001, 2.0),
        xscale = :log10,
        yscale = :log10,
        legend = :bottomleft,
    )
    p2 = plot(xscale = :log10)
    for r in results
        plot!(p1, r)
        residualplot!(p2, r)
    end
    plot(p1, p2, link = :x, layout = @layout [top{0.75h}; bottom{0.25h}])
end

p = plot_result(data, result1, result2, result3)
savefig_(p, "10_plotresult.png")

# ------------------------------------------------------------------ black body + power law
# deepcopy so the new model's parameters are not aliased to model2's
bbpl_model = model2.m1 * (PowerLaw() + model2.a1) |> deepcopy

# Freeze the hydrogen column density to the galactic value
# (newer parameter naming from the current docs: bbpl_model.m1.ηH, not ηH_1)
bbpl_model.m1.ηH.value = 4
bbpl_model.m1.ηH.frozen = true
@show bbpl_model

bbpl_result = SpectralFitting.fit(FittingProblem(bbpl_model => data), LevenbergMarquadt())
@show bbpl_result

p = plot(data, ylims = (0.001, 2.0), xscale = :log10, yscale = :log10,
         legend = :bottomleft)
plot!(p, bbpl_result, label = "Blackbody+Powerlaw: χ² = $(round(chi2(bbpl_result)))")

# Update model, then fix the black body temperature to 2 keV
update_model!(bbpl_model, bbpl_result)
bbpl_model.a2.T.value = 2.0
bbpl_model.a2.T.frozen = true
@show bbpl_model

bbpl_result2 = SpectralFitting.fit(FittingProblem(bbpl_model => data), LevenbergMarquadt())
@show bbpl_result2

plot!(p, bbpl_result2, label = "Blackbody+Powerlaw (T fixed): χ² = $(round(chi2(bbpl_result2)))")
savefig_(p, "11_bbpl.png")

# ------------------------------------------------------------------ MCMC with Turing.jl
using StatsPlots, Turing

# Newer SpectralFitting API (per the current SpectralFitting.jl docs walkthrough):
# the model function takes the data, standard deviation and a wrapper `f`
# that accepts the parameters as separate arguments.
@model function mcmc_model(objective, stddev, f)
    K ~ Normal(20.0, 1.0)
    a ~ Normal(2.2, 0.3)
    ηH ~ truncated(Normal(0.5, 0.1); lower = 0)
    # The wrapper takes parameters in model order: ηH (absorption), then K, a (power law).
    # Verified: f(result1.u...) with u = [ηH, K, a] reproduces the least-squares χ².
    pred = f(ηH, K, a)
    return objective ~ MvNormal(pred, stddev)
end

# NOTE: the original notebook wrote `model => data` here, but no variable named
# `model` exists; the text says "go back to our first model", so use model1.
config = FittingConfig(FittingProblem(model1 => data))
mm = mcmc_model(
    get_objective_single(config),
    sqrt.(get_objective_variance_single(config)),
    get_invoke_wrapper_single(config),
)

chain = sample(mm, NUTS(), 5_000)
display(chain)

p = plot(chain)
savefig_(p, "12_mcmc_chains.png")

println("\nDone. Figures are in: ", FIGDIR)
