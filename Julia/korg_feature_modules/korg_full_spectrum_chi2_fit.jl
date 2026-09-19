# Full-Spectrum chi^2 Fitting Example (Julia)
# Synthesizes a fake "observed" spectrum at known parameters, adds noise,
# then uses Korg.Fit.fit_spectrum to recover the parameters from a
# deliberately offset initial guess.

println(">>> [1/9] Loading packages...")
using Korg
using PyPlot
using LinearAlgebra   # for diag(), used to pull uncertainties off the covariance matrix
println(">>> [1/9] Packages loaded: Korg, PyPlot, LinearAlgebra")

# ---------------------------------------------------------------------------
# Shared wavelength range and linelist
# ---------------------------------------------------------------------------
λmin = 6000.0
λmax = 6020.0
println(">>> [2/9] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()  # `get_linelist` does not exist
println(">>> [2/9] Linelist loaded: $(length(linelist)) lines")

# ---------------------------------------------------------------------------
# 1. Create a synthetic "observed" spectrum (so the example is runnable)
# ---------------------------------------------------------------------------
true_params = Dict(
    "Teff"  => 5650.0,
    "logg"  => 4.30,
    "M_H"   => -0.15,
    "vmic"  => 1.2,
    "vsini" => 3.0,
)
println(">>> [3/9] True parameters: ", true_params)

println(">>> [4/9] Synthesizing the 'observed' spectrum at the true parameters...")
obs_wls, obs_flux, _ = Korg.synth(
    Teff  = true_params["Teff"],
    logg  = true_params["logg"],
    M_H   = true_params["M_H"],
    vmic  = true_params["vmic"],
    vsini = true_params["vsini"],
    linelist = linelist,
    wavelengths = (λmin, λmax)   # NOT λmin=/λmax= keywords
)
println(">>> [4/9] Synthesized $(length(obs_wls)) points.")

# Add small noise to simulate real data, and keep a matching per-pixel
# uncertainty vector -- fit_spectrum requires obs_err as an actual argument.
noise_level = 0.01
obs_flux = obs_flux .+ noise_level .* randn(length(obs_flux))
obs_err  = fill(noise_level, length(obs_flux))
println(">>> [4/9] Added Gaussian noise (σ=$(noise_level)) and built obs_err vector.")

# ---------------------------------------------------------------------------
# 2. Initial guess for fitting (deliberately offset from the truth)
# ---------------------------------------------------------------------------
initial_guess = Dict(
    "Teff"  => 5500.0,
    "logg"  => 4.0,
    "M_H"   => -0.3,
    "vmic"  => 1.0,
    "vsini" => 2.0,
)
println(">>> [5/9] Initial guess: ", initial_guess)

# Nothing is held fixed -- all 5 parameters above are free -- but
# fit_spectrum's signature requires a fixed_params Dict positionally
# regardless, so pass an empty one.
fixed_params = Dict{String, Float64}()

# ---------------------------------------------------------------------------
# 3. Full-spectrum chi^2 fitting
# ---------------------------------------------------------------------------
# `Korg.fit_spectrum` does not exist -- it's `Korg.Fit.fit_spectrum`, and it
# requires obs_wls, obs_flux, obs_err, linelist, initial_guess, fixed_params
# ALL as positional arguments (fixed_params has no default), not a single
# `initial=` NamedTuple.
println(">>> [6/9] Running Korg.Fit.fit_spectrum (this can take a while)...")
result = Korg.Fit.fit_spectrum(
    obs_wls,
    obs_flux,
    obs_err,
    linelist,
    initial_guess,
    fixed_params;
    # R has NO default -- fit_spectrum errors outright if it's omitted
    # (unless LSF_matrix + synthesis_wls are given instead). It also can't
    # be Inf (that produced NaNs via a stray 0*Inf internally). Since our
    # synthetic "observed" spectrum has no instrumental broadening baked
    # in, use a large-but-finite R so the LSF's FWHM (~λ/R) is small
    # compared to the lines' intrinsic (vmic/vsini/thermal) widths --
    # effectively negligible broadening without hitting floating-point Inf.
    R = 300_000.0
)
println(">>> [6/9] Fit complete. Result type: ", typeof(result))

# The exact field/property names on the returned result aren't something
# we've independently confirmed, and earlier scripts in this series turned
# out to return simpler things than their docstrings implied -- so check
# what's actually there rather than assume.
best = if hasproperty(result, :best_fit_params)
    println(">>>       Using result.best_fit_params")
    result.best_fit_params
elseif result isa AbstractDict
    println(">>>       Result itself is a Dict of best-fit parameters")
    result
else
    println(">>>       Unrecognized result shape -- dumping for inspection:")
    println(result)
    error("Unexpected return type from fit_spectrum: $(typeof(result))")
end

# result.covariance is (parameter_names, covariance_matrix) -- the diagonal
# gives each parameter's variance, so sqrt(diag(...)) gives 1-sigma
# uncertainties. Build a name -> uncertainty Dict so we can look values up
# by the same keys as `best`, regardless of what order covariance lists them in.
uncertainties = if hasproperty(result, :covariance)
    cov_names, cov_matrix = result.covariance
    sigmas = sqrt.(diag(cov_matrix))
    Dict(name => sigma for (name, sigma) in zip(cov_names, sigmas))
else
    println(">>>       No `covariance` field on result -- uncertainties unavailable.")
    Dict{String, Float64}()
end

println(">>> [7/9] Best-fit parameters (with 1σ uncertainties where available):")
for k in ["Teff", "logg", "M_H", "vmic", "vsini"]
    σ = get(uncertainties, k, NaN)
    println(">>>       $(k) = ", best[k], " ± ", σ)
end

println(">>> [7/9] Recovered vs. true:")
for k in ["Teff", "logg", "M_H", "vmic", "vsini"]
    σ = get(uncertainties, k, NaN)
    Δ = best[k] - true_params[k]
    n_sigma = isnan(σ) || σ == 0 ? NaN : Δ / σ
    println(">>>       $(k): fit=$(best[k]) ± $(σ), true=$(true_params[k]), Δ=$(Δ) ($(n_sigma)σ)")
end

# ---------------------------------------------------------------------------
# 4. Plot observed vs best-fit synthetic spectrum
# ---------------------------------------------------------------------------
println(">>> [8/9] Synthesizing best-fit spectrum for the comparison plot...")
fit_wls, fit_flux, _ = Korg.synth(
    Teff  = best["Teff"],
    logg  = best["logg"],
    M_H   = best["M_H"],
    vmic  = best["vmic"],
    vsini = best["vsini"],
    linelist = linelist,
    wavelengths = (λmin, λmax)
)
println(">>> [8/9] Best-fit spectrum synthesized.")

println(">>> [9/9] Building plot...")
figure(figsize=(10, 4))
plot(obs_wls, obs_flux, label="Observed", color="black", linewidth=0.8)
plot(fit_wls, fit_flux, label="Best-fit", color="red", alpha=0.7, linewidth=0.8)
xlabel("λ [Å]")
ylabel("Flux")
title("Full-Spectrum χ² Fit")
legend()
tight_layout()

outpath = "korg_full_spectrum_chi2_fit.png"
savefig(outpath, dpi=150)
println(">>> [9/9] Plot saved to: $(abspath(outpath))")
println(">>> Done.")
