# Abundance Response Example (Julia)
# Computes the finite-difference abundance response function for Fe:
# ∂Flux/∂A(Fe), holding the atmosphere fixed and only perturbing the
# abundance fed into the radiative transfer -- the standard way response
# functions are computed (MOOG/SME-style), and it avoids conflating a real
# abundance signal with spurious atmosphere-recomputation noise.

println(">>> [1/7] Loading packages...")
using Korg
using PyPlot
println(">>> [1/7] Packages loaded: Korg, PyPlot")

# ---------------------------------------------------------------------------
# 1. Stellar parameters
# ---------------------------------------------------------------------------
Teff = 5800.0
logg = 4.40
M_H  = 0.0

A_Fe = 7.50   # baseline solar Fe abundance (ABSOLUTE A(X))
dA   = 0.10   # abundance perturbation (dex)

# ---------------------------------------------------------------------------
# 2. Wavelength range + linelist
# ---------------------------------------------------------------------------
λmin = 6000.0
λmax = 6020.0

println(">>> [2/7] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()  # `get_linelist` does not exist
println(">>> [2/7] Linelist loaded: $(length(linelist)) lines")

# ---------------------------------------------------------------------------
# 3. Build baseline + perturbed A_X vectors, and ONE shared atmosphere
# ---------------------------------------------------------------------------
# `synth`'s `abundances=Dict(...)` keyword doesn't exist, and even its real
# per-element keywords (e.g. `Fe=...`) are OFFSETS added on top of the
# metallicity-scaled default -- not absolute A(X) substitutions. Since
# A_Fe/A_Fe+dA here are meant as absolute abundances, build the A_X vectors
# by hand (as in the custom-solar-scale and NLTE-correction fixes) and use
# the low-level `synthesize`. Both syntheses share the SAME atmosphere
# (built once at the baseline composition) so the finite difference below
# reflects only the abundance's effect on line opacity, not incidental
# differences from re-interpolating the atmosphere twice.
const ELEMENT_SYMBOLS = ["H","He","Li","Be","B","C","N","O","F","Ne",
                          "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
                          "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"]
atomic_number(sym::AbstractString) = findfirst(==(sym), ELEMENT_SYMBOLS)
Z_Fe = atomic_number("Fe")

solar_ref    = Korg.format_A_X()          # default solar reference vector
A_X_baseline = copy(solar_ref); A_X_baseline[Z_Fe] = A_Fe
A_X_plus     = copy(solar_ref); A_X_plus[Z_Fe]     = A_Fe + dA

println(">>> [3/7] Interpolating shared atmosphere at Teff=$(Teff), " *
        "logg=$(logg), [M/H]=$(M_H)...")
atm = Korg.interpolate_marcs(Teff, logg, A_X_baseline;
                              solar_abundances=solar_ref,
                              clamp_abundances=true)
println(">>> [3/7] Atmosphere built: $(length(atm.layers)) layers")

# ---------------------------------------------------------------------------
# 4. Synthesize baseline and perturbed spectra
# ---------------------------------------------------------------------------
println(">>> [4/7] Synthesizing baseline spectrum (A(Fe)=$(A_Fe))...")
result0 = Korg.synthesize(atm, linelist, A_X_baseline, (λmin, λmax))
wls   = result0.wavelengths
flux0 = result0.flux ./ result0.cntm   # rectified flux -- fine here, since
                                        # the response function is defined
                                        # in terms of normalized flux/dex,
                                        # unlike the photometry case.

println(">>> [5/7] Synthesizing perturbed spectrum (A(Fe)=$(A_Fe + dA))...")
result_plus = Korg.synthesize(atm, linelist, A_X_plus, (λmin, λmax))
flux_plus = result_plus.flux ./ result_plus.cntm

# ---------------------------------------------------------------------------
# 5. Compute abundance response
# ---------------------------------------------------------------------------
println(">>> [6/7] Computing finite-difference abundance response...")
response = (flux_plus .- flux0) ./ dA

println(">>> [6/7] Computed abundance response for Fe.")
println(">>> [6/7] First few values: ", response[1:5])

# ---------------------------------------------------------------------------
# 6. Plot abundance response
# ---------------------------------------------------------------------------
println(">>> [7/7] Building plot...")
figure(figsize=(10, 4))
plot(wls, response, color="red", linewidth=0.8)
xlabel("λ [Å]")
ylabel("∂Flux / ∂A(Fe)")
title("Abundance Response Function for Fe (A(Fe)=$(A_Fe), dA=$(dA))")
tight_layout()

outpath = "korg_abundance_response.png"
savefig(outpath, dpi=150)
println(">>> [7/7] Plot saved to: $(abspath(outpath))")
println(">>> [7/7] Done.")
