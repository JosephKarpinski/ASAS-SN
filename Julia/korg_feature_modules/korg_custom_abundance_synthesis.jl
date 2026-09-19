# Custom Abundance Pattern Example (Julia)
# Synthesizes a spectrum with arbitrary per-element abundance overrides
# (e.g. a CEMP-like, alpha-enhanced metal-poor star).

println(">>> [1/5] Loading packages...")
using Korg
using PyPlot
println(">>> [1/5] Packages loaded: Korg, PyPlot")

# --- Stellar parameters ------------------------------------------------------
Teff = 5100.0
logg = 2.5
M_H  = -2.0
println(">>> [2/5] Stellar parameters set: Teff=$(Teff), logg=$(logg), [M/H]=$(M_H)")

# --- Wavelength range (Å) ----------------------------------------------------
λmin = 4300.0
λmax = 4350.0
println(">>> [2/5] Wavelength range set: $(λmin) - $(λmax) Å")

# --- Custom abundance pattern -------------------------------------------------
# `Korg.synth` has no `abundances=Dict(...)` keyword. Instead, individual
# element abundances ([X/H], solar-relative) are passed as their OWN keyword
# arguments — e.g. `C=1.2, N=0.8` — which override M_H for that element only.
# Rather than typing each one out, build a Dict{Symbol,Float64} (note: Symbol
# keys, e.g. :C, not "C") and splat it into the call with `...`. This only
# works as KEYWORD splatting if the call has a leading `;` before the keyword
# list (see the `synth(;` below) — without it, `abund...` splats as bare
# positional `Pair` objects instead, which is a MethodError.
abund = Dict(
    :C  => +1.2,   # Carbon-enhanced (CEMP-like)
    :N  => +0.8,   # Nitrogen-enhanced
    :O  => +0.4,   # Mild alpha-enhancement
    :Mg => +0.3,
    :Si => +0.2,
    :Ca => +0.2,
    :Ti => +0.1,
    :Fe => -1.0    # Metal-poor baseline, overriding the global M_H for Fe specifically
)
println(">>> [2/5] Custom abundance overrides: ", abund)

# --- Linelist -----------------------------------------------------------------
println(">>> [3/5] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()
println(">>> [3/5] Linelist loaded: $(length(linelist)) lines")

# --- Synthesize spectrum -----------------------------------------------------
println(">>> [4/5] Running Korg.synth with custom abundances...")
wls, flux, cont = Korg.synth(;
    Teff = Teff,
    logg = logg,
    M_H  = M_H,
    linelist = linelist,
    wavelengths = (λmin, λmax),
    abund...
)
println(">>> [4/5] Synthesis complete.")
println(">>>       wls   : $(length(wls)) points, range $(first(wls)) - $(last(wls)) Å")
println(">>>       flux  : min=$(minimum(flux)), max=$(maximum(flux))")
println(">>>       cont  : min=$(minimum(cont)), max=$(maximum(cont))")

# --- Plot -------------------------------------------------------------------
println(">>> [5/5] Building plot...")
figure(figsize=(10, 4))
plot(wls, flux, color="black", linewidth=0.8)
xlabel("λ [Å]")
ylabel("Flux")
title("Custom Abundance Pattern | Teff=$(Teff), logg=$(logg), [M/H]=$(M_H)")
tight_layout()

outpath = "korg_custom_abundance_synthesis.png"
savefig(outpath, dpi=150)
println(">>> [5/5] Plot saved to: $(abspath(outpath))")
println(">>> Done.")
