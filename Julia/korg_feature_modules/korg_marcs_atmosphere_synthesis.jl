# Model Atmosphere Interpolation Example (Julia)
# Interpolates a MARCS atmosphere at arbitrary stellar parameters, then
# synthesizes a spectrum from that specific atmosphere object.

println(">>> [1/6] Loading packages...")
using Korg
using PyPlot
println(">>> [1/6] Packages loaded: Korg, PyPlot")

# --- Stellar parameters ------------------------------------------------------
Teff = 4800.0
logg = 2.2
M_H  = -0.7
println(">>> [2/6] Stellar parameters set: Teff=$(Teff), logg=$(logg), [M/H]=$(M_H)")

# --- Abundance vector ---------------------------------------------------------
# Build A(X) once and reuse it for both the atmosphere and the synthesis, so
# the two stay self-consistent (Korg's recommended approach, rather than
# passing M_H directly to interpolate_marcs and separately to synthesis).
A_X = Korg.format_A_X(M_H)

# --- Interpolate a MARCS atmosphere ------------------------------------------
# `interpolate_marcs` takes Teff, logg, A_X as POSITIONAL arguments:
# interpolate_marcs(Teff, logg, A_X; kwargs...). Passing M_H directly instead
# (interpolate_marcs(Teff, logg, M_H=...)) also works but triggers a
# "not recommended" warning, since it's less self-consistent.
println(">>> [3/6] Interpolating MARCS atmosphere...")
atm = Korg.interpolate_marcs(Teff, logg, A_X)
println(">>> [3/6] Atmosphere interpolated: $(length(atm.layers)) layers")

# An atmosphere doesn't expose flat `.T` / `.depth` arrays directly — its data
# lives in `atm.layers` (a vector of layer structs with .tau_ref, .z, .temp,
# .electron_number_density, .number_density). Korg provides convenience
# accessors for exactly this:
temps = Korg.get_temps(atm)
println(">>>       First few temperatures (K): ", temps[1:5])
println(">>>       Temperature range: $(minimum(temps)) - $(maximum(temps)) K")

# --- Linelist -----------------------------------------------------------------
println(">>> [4/6] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()
println(">>> [4/6] Linelist loaded: $(length(linelist)) lines")

# --- Synthesize a spectrum using the interpolated atmosphere -----------------
# Korg.synth has NO `atmosphere` keyword — it always builds its own atmosphere
# internally from Teff/logg/M_H. To actually synthesize from an atmosphere you
# interpolated yourself, use the lower-level `synthesize` function instead.
# It needs an A(X) abundance vector (format_A_X builds one from [M/H]), and it
# returns un-rectified physical flux, so we normalize by the continuum
# ourselves to get a spectrum comparable to what `synth` would hand back.
println(">>> [5/6] Running Korg.synthesize with the interpolated atmosphere...")
λmin = 5000.0
λmax = 5100.0
result = Korg.synthesize(atm, linelist, A_X, (λmin, λmax))

wls  = result.wavelengths
cont = result.cntm
flux = result.flux ./ result.cntm   # rectify (continuum-normalize) manually

println(">>> [5/6] Synthesis complete.")
println(">>>       wls   : $(length(wls)) points, range $(first(wls)) - $(last(wls)) Å")
println(">>>       flux  : min=$(minimum(flux)), max=$(maximum(flux))")
println(">>>       cont  : min=$(minimum(cont)), max=$(maximum(cont))")

# --- Plot -----------------------------------------------------------------
println(">>> [6/6] Building plot...")
figure(figsize=(10, 4))
plot(wls, flux, color="black", linewidth=0.8)
xlabel("λ [Å]")
ylabel("Rectified Flux")
title("Spectrum from Interpolated MARCS Atmosphere | Teff=$(Teff), logg=$(logg), [M/H]=$(M_H)")
tight_layout()

outpath = "korg_marcs_atmosphere_synthesis.png"
savefig(outpath, dpi=150)
println(">>> [6/6] Plot saved to: $(abspath(outpath))")
println(">>> Done.")
