# Synthetic Photometry Example (Julia)
# Synthesizes a spectrum, integrates it against a filter transmission
# curve, and computes a synthetic AB magnitude.

println(">>> [1/8] Loading packages...")
using Korg
using DelimitedFiles
using Interpolations
using PyPlot
println(">>> [1/8] Packages loaded: Korg, DelimitedFiles, Interpolations, PyPlot")

# ---------------------------------------------------------------------------
# 1. Load a filter transmission curve (example: SDSS g-band)
# ---------------------------------------------------------------------------
# `readdlm(path, delim, T1, T2)` doesn't exist -- there's no two-Type-args
# overload. `readdlm(path, delim, T)` returns a single Matrix{T} with one
# column per field; split it into the two columns you actually want.
filter_path = joinpath(@__DIR__, "sdss_g_filter.dat")

if isfile(filter_path)
    println(">>> [2/8] Loading filter transmission curve: $(filter_path)")
    filter_data = readdlm(filter_path, '\t', Float64)
    λfilt = filter_data[:, 1]
    Tfilt = filter_data[:, 2]
else
    # No file on disk -- fall back to a rough Gaussian approximation of the
    # SDSS g-band response (real curve peaks ~4770 Å, spans ~3900-5600 Å;
    # for an authentic curve, grab one from the SVO Filter Profile Service).
    println(">>> [2/8] $(filter_path) not found -- using an approximate " *
            "Gaussian SDSS g-band curve instead (replace with the real " *
            "SVO transmission file for anything quantitative).")
    λfilt = collect(3900.0:5.0:5600.0)
    λ0, σ = 4770.0, 400.0
    Tfilt = exp.(-0.5 .* ((λfilt .- λ0) ./ σ) .^ 2)
end
println(">>> [2/8] Filter curve: $(length(λfilt)) points, " *
        "range $(minimum(λfilt)) - $(maximum(λfilt)) Å")

# ---------------------------------------------------------------------------
# 2. Synthesize a spectrum with Korg
# ---------------------------------------------------------------------------
# `synth` has no `λmin=`/`λmax=` keywords -- needs `wavelengths=(λmin,λmax)`.
# `get_linelist(:vald)` doesn't exist -- use `get_VALD_solar_linelist()`.
# A small buffer around the filter's exact bounds avoids the interpolation
# below landing right on (or just past) the synthesis grid's edge.
buffer = 5.0
λmin = minimum(λfilt) - buffer
λmax = maximum(λfilt) + buffer

println(">>> [3/8] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()
println(">>> [3/8] Linelist loaded: $(length(linelist)) lines")

println(">>> [4/8] Running Korg.synth over $(λmin) - $(λmax) Å...")
wls, flux, cont = Korg.synth(
    Teff = 5800.0,
    logg = 4.40,
    M_H  = 0.0,
    wavelengths = (λmin, λmax),
    linelist    = linelist
)
println(">>> [4/8] Synthesis complete: $(length(wls)) points")

# ---------------------------------------------------------------------------
# 3. Interpolate spectrum onto filter grid
# ---------------------------------------------------------------------------
# IMPORTANT: `synth`'s `flux` is RECTIFIED (continuum-normalized, ~1 outside
# lines) by default -- it's a dimensionless ratio, not a physical flux
# density. Feeding that straight into an AB-magnitude calculation (as the
# original script did) would run without error but silently produce a
# meaningless magnitude, since the AB zeropoint assumes real erg/s/cm^2/Å
# units. The fix: reconstruct physical flux as `flux .* cont` (rectified
# flux times the continuum it was normalized by) before doing any
# photometry -- the rectified version is fine for the display plot below,
# but not for the magnitude integral.
println(">>> [5/8] Reconstructing physical flux (flux .* continuum) " *
        "for photometry...")
flux_physical = flux .* cont

itp = LinearInterpolation(wls, flux_physical; extrapolation_bc=Line())
Fλ = itp.(λfilt)

# ---------------------------------------------------------------------------
# 4. Compute synthetic AB magnitude
# ---------------------------------------------------------------------------
c = 2.99792458e18  # speed of light in Å/s

# Convert Fλ (erg/s/cm^2/Å) -> Fν (erg/s/cm^2/Hz)
Fν = Fλ .* (λfilt .^ 2) ./ c

# Filter-weighted mean Fν
num = sum(Fν .* Tfilt)
den = sum(Tfilt)
Fν_eff = num / den

m_AB = -2.5 * log10(Fν_eff / 3.631e-20)   # AB zeropoint: 3631 Jy = 3.631e-20 erg/s/cm^2/Hz

println(">>> [6/8] Synthetic SDSS g magnitude (AB) = ", m_AB)
println(">>>       (this is a surface/Eddington-flux-based magnitude -- " *
        "not diluted by (R/d)^2 -- useful for internal comparisons/colors, " *
        "not a true apparent magnitude without further scaling.)")

# ---------------------------------------------------------------------------
# 5. Plot spectrum + filter curve
# ---------------------------------------------------------------------------
println(">>> [7/8] Building plot...")
figure(figsize=(10, 4))
plot(wls, flux ./ maximum(flux), color="black", linewidth=0.6, label="Spectrum (rectified, for display)")
plot(λfilt, Tfilt, color="tab:green", linewidth=1.5, label="Filter (scaled)")
xlabel("λ [Å]")
ylabel("Flux / Transmission")
title("Synthetic Photometry Example (m_AB = $(round(m_AB, digits=3)))")
legend()
tight_layout()

outpath = "korg_synthetic_photometry.png"
savefig(outpath, dpi=150)
println(">>> [8/8] Plot saved to: $(abspath(outpath))")
println(">>> [8/8] Done.")
