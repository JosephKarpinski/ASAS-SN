# Full AD Spectrum Derivative Example (Julia)
# Uses ForwardDiff to compute d(flux)/d(Teff) across a wavelength window,
# exercising Korg's support for automatic differentiation through synthesis.

println(">>> [1/6] Loading packages...")
using Korg
using ForwardDiff
using PyPlot
println(">>> [1/6] Packages loaded: Korg, ForwardDiff, PyPlot")

# --- Fixed inputs shared across all synth calls -------------------------------
logg = 4.40
M_H  = 0.0
λmin = 6000.0
λmax = 6020.0
println(">>> [2/6] Fixed parameters: logg=$(logg), [M/H]=$(M_H), " *
        "λ range=$(λmin)-$(λmax) Å")

# There is no `get_linelist` function in Korg (same bug as the earlier basic
# synthesis example). The linelist also doesn't depend on Teff, so load it
# ONCE here rather than inside the differentiated function -- otherwise
# ForwardDiff would reload the ~42,000-line VALD list on every evaluation.
println(">>> [2/6] Loading VALD solar linelist (once, outside the AD call)...")
linelist = Korg.get_VALD_solar_linelist()
println(">>> [2/6] Linelist loaded: $(length(linelist)) lines")

# --- Define a wrapper that returns flux at a given Teff -----------------------
# `synth`'s wavelength keyword is `wavelengths` (a tuple), not λmin/λmax.
function flux_at_Teff(Teff)
    wls, flux, cont = Korg.synth(
        Teff = Teff,
        logg = logg,
        M_H  = M_H,
        linelist = linelist,
        wavelengths = (λmin, λmax)
    )
    return flux
end

# --- Compute derivative d(flux)/d(Teff) at Teff = 5777 K ----------------------
Teff0 = 5777.0
println(">>> [3/6] Computing d(flux)/d(Teff) at Teff0=$(Teff0) via ForwardDiff...")
dflux_dTeff = ForwardDiff.derivative(flux_at_Teff, Teff0)
println(">>> [3/6] Derivative computed.")
println(">>>       Derivative array length = ", length(dflux_dTeff))
println(">>>       First few derivative values = ", dflux_dTeff[1:5])
println(">>>       min=$(minimum(dflux_dTeff)), max=$(maximum(dflux_dTeff))")

# --- Compute the reference spectrum at Teff0 (for the plot's x-axis/context) --
println(">>> [4/6] Computing reference spectrum at Teff0 for plotting...")
wls0, flux0, _ = Korg.synth(
    Teff = Teff0,
    logg = logg,
    M_H  = M_H,
    linelist = linelist,
    wavelengths = (λmin, λmax)
)
println(">>> [4/6] Reference spectrum computed: $(length(wls0)) points.")

# --- Plot derivative -----------------------------------------------------------
println(">>> [5/6] Building plot...")
figure(figsize=(10, 4))
plot(wls0, dflux_dTeff, color="red", linewidth=0.8)
xlabel("λ [Å]")
ylabel("∂Flux / ∂Teff")
title("Automatic Differentiation of Spectrum w.r.t. Teff at Teff=$(Teff0)")
tight_layout()

outpath = "korg_ad_spectrum_derivative.png"
savefig(outpath, dpi=150)
println(">>> [5/6] Plot saved to: $(abspath(outpath))")
println(">>> [6/6] Done.")
