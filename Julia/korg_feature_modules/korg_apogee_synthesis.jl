#=
High-resolution APOGEE synthesis example using Korg's built-in APOGEE DR17
linelist, APOGEE wavelength grid, and resolution R ~ 22,500.
=#

println(">>> [1/7] Loading packages...")
using Korg
# `PythonPlot` (PythonCall-based) is a different package from `PyPlot`
# (PyCall-based) -- it has its own separate Python bridge, and its
# initialization is what's crashing here, before any Korg code even runs.
# Using the already-working `PyPlot` avoids the conflict entirely.
using PyPlot
println(">>> [1/7] Packages loaded: Korg, PyPlot")

# ------------------------------------------------------------
# 1. APOGEE wavelength grid (log-λ, R ~ 22,500)
# ------------------------------------------------------------
delLog = 6e-6
apowls = 10 .^ range(
    start  = 4.179 - 125 * delLog,
    step   = delLog,
    length = 8575 + 125
)
println(">>> [2/7] APOGEE wavelength grid built: $(length(apowls)) points, " *
        "range $(first(apowls)) - $(last(apowls)) Å")

# ------------------------------------------------------------
# 2. APOGEE DR17 linelist (H-band, includes water if desired)
# ------------------------------------------------------------
println(">>> [3/7] Loading APOGEE DR17 linelist (with water lines)...")
apolines = Korg.get_APOGEE_DR17_linelist(; include_water=true)
println(">>> [3/7] Linelist loaded: $(length(apolines)) lines")

# ------------------------------------------------------------
# 3. High-res synthetic spectrum on Korg's native grid
# ------------------------------------------------------------
# This has to run BEFORE building the LSF matrix -- the matrix maps FROM
# this native synthesis grid TO the APOGEE grid, so it needs the real `wls`
# array `synth` produces, not just the wavelength bounds.
println(">>> [4/7] Running Korg.synth over 15,000-17,000 Å " *
        "(this is a wide window at default resolution -- it may take a while)...")
wls, flux, cont = Korg.synth(
    linelist    = apolines,
    wavelengths = (15_000, 17_000),
    Teff        = 4800.0,
    logg        = 2.3,
    M_H         = -0.3,
    vmic        = 1.8
)
println(">>> [4/7] Native synthesis complete: $(length(wls)) points")

# ------------------------------------------------------------
# 4. APOGEE LSF matrix at R ≈ 22,500
# ------------------------------------------------------------
# `compute_LSF_matrix` needs the ACTUAL native synthesis wavelength array
# (`wls`, just produced above) as its source grid, and `apowls` as the
# target/observed grid -- not the raw (λmin, λmax) bounds tuple, which
# isn't a wavelength array at all.
println(">>> [5/7] Building APOGEE LSF matrix (native grid -> APOGEE grid)...")
LSF = Korg.compute_LSF_matrix(wls, apowls, 22_500)
println(">>> [5/7] LSF matrix built: size $(size(LSF))")

# ------------------------------------------------------------
# 5. Convolve with APOGEE LSF and resample to APOGEE grid
# ------------------------------------------------------------
println(">>> [6/7] Applying LSF and resampling to APOGEE grid...")
flux_apo = LSF * flux           # apply instrumental profile + resample
flux_apo ./= maximum(flux_apo)  # simple renormalization
println(">>> [6/7] Resampled flux: $(length(flux_apo)) points, " *
        "min=$(minimum(flux_apo)), max=$(maximum(flux_apo))")

# ------------------------------------------------------------
# 6. Plot: high-res APOGEE-like synthetic spectrum
# ------------------------------------------------------------
println(">>> [7/7] Building plot...")
figure(figsize=(10, 4))
plot(apowls, flux_apo, color="black", linewidth=0.6)
xlabel("λ [Å]")
ylabel("Flux (APOGEE resolution)")
title("High-res APOGEE Synthetic Spectrum (Teff=4800, logg=2.3, [M/H]=-0.3)")
tight_layout()

outpath = "korg_apogee_synthesis.png"
savefig(outpath, dpi=150)
println(">>> [7/7] Plot saved to: $(abspath(outpath))")
println(">>> [7/7] Done.")
