# vmic + vmac Example (Julia)
# Demonstrates microturbulence (a real Korg `synth` keyword) alongside
# macroturbulence (NOT a Korg keyword -- applied manually as a post-hoc
# Gaussian convolution, which is the standard way it's handled anyway).

println(">>> [1/6] Loading packages...")
using Korg
using PyPlot
using Statistics   # for `mean`, used in the macroturbulence kernel below
println(">>> [1/6] Packages loaded: Korg, PyPlot, Statistics")

# ---------------------------------------------------------------------------
# 1. Define stellar parameters
# ---------------------------------------------------------------------------
Teff = 5800.0
logg = 4.40
M_H  = 0.0

vmic  = 1.2   # microturbulence (km/s) -- a real `synth` keyword
vmac  = 4.0   # macroturbulence (km/s) -- NOT a `synth` keyword; see below
vsini = 2.0   # rotational broadening (km/s) -- a real `synth` keyword

# ---------------------------------------------------------------------------
# 2. Wavelength range and linelist
# ---------------------------------------------------------------------------
λmin = 6000.0
λmax = 6020.0

println(">>> [2/6] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()  # `get_linelist` does not exist
println(">>> [2/6] Linelist loaded: $(length(linelist)) lines")

# ---------------------------------------------------------------------------
# 3. Synthesize spectrum with vmic + vsini (vmac is NOT passed here)
# ---------------------------------------------------------------------------
# `synth` has no `λmin=`/`λmax=` keywords -- must be `wavelengths=(λmin,λmax)`.
# `synth` also has no `vmac=` keyword at all: microturbulence is a physical
# velocity field that widens each line's intrinsic profile during radiative
# transfer, so Korg needs it as an input to `synth`/`synthesize`. Macroturbulence
# is conventionally just a Gaussian smoothing of the emergent spectrum (an
# approximation to disk-integrated large-scale velocity cells), so it's
# applied as a separate convolution step after synthesis, not inside it.
println(">>> [3/6] Running Korg.synth (vmic + vsini only; vmac applied below)...")
wls, flux, cont = Korg.synth(
    Teff  = Teff,
    logg  = logg,
    M_H   = M_H,
    vmic  = vmic,
    vsini = vsini,
    linelist    = linelist,
    wavelengths = (λmin, λmax)
)
println(">>> [3/6] Synthesis complete: $(length(wls)) points")

# ---------------------------------------------------------------------------
# 4. Apply macroturbulent broadening as a Gaussian convolution
# ---------------------------------------------------------------------------
# Converts a macroturbulent velocity (km/s) to a Gaussian sigma in Å at a
# representative wavelength, then convolves the flux with that kernel.
# This is a simple direct-sum convolution (fine for a ~2000-point window);
# for very large spectra you'd want an FFT-based convolution instead.
function gaussian_broaden(wls::Vector{Float64}, flux::Vector{Float64}, vmac_kms::Float64)
    c_kms = 2.99792458e5
    dλ = mean(diff(wls))                # assume roughly uniform sampling
    λ0 = mean(wls)
    σ_λ   = vmac_kms / c_kms * λ0       # Gaussian sigma in Å
    σ_pix = σ_λ / dλ                    # Gaussian sigma in pixels

    half_width = ceil(Int, 4 * σ_pix)
    offsets = -half_width:half_width
    kernel  = [exp(-0.5 * (Δ / σ_pix)^2) for Δ in offsets]
    kernel ./= sum(kernel)

    n = length(flux)
    broadened = similar(flux)
    for i in 1:n
        s = 0.0
        for (k, Δ) in zip(kernel, offsets)
            j = clamp(i + Δ, 1, n)      # edge-clamp instead of zero-padding
            s += flux[j] * k
        end
        broadened[i] = s
    end
    return broadened
end

println(">>> [4/6] Applying macroturbulent (vmac=$(vmac) km/s) broadening...")
flux_broadened = gaussian_broaden(wls, flux, vmac)
println(">>> [4/6] Broadening complete.")

# ---------------------------------------------------------------------------
# 5. Plot the result
# ---------------------------------------------------------------------------
println(">>> [5/6] Building plot...")
figure(figsize=(10, 4))
plot(wls, flux,            color="gray",  linewidth=0.7, alpha=0.6, label="vmic+vsini only")
plot(wls, flux_broadened,  color="black", linewidth=0.9, label="+ vmac (Gaussian)")
xlabel("λ [Å]")
ylabel("Flux")
title("Spectrum with vmic=$(vmic) km/s, vmac=$(vmac) km/s, vsini=$(vsini) km/s")
legend()
tight_layout()

outpath = "korg_vmic_vmac_broadening.png"
savefig(outpath, dpi=150)
println(">>> [6/6] Plot saved to: $(abspath(outpath))")
println(">>> [6/6] Done.")
