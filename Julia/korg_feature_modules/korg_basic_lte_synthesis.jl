# Basic LTE spectrum synthesis with Korg.jl
# Computes a spectrum from Teff/logg/[M/H] and a built-in linelist, then plots it.

println(">>> [1/5] Loading packages...")
using Korg
using PyPlot
println(">>> [1/5] Packages loaded: Korg, PyPlot")

# --- Stellar parameters ------------------------------------------------------
Teff = 5777.0        # Effective temperature (K)
logg = 4.44          # Surface gravity (cgs)
M_H  = 0.0           # Metallicity [M/H]  (Korg >=1.0 uses M_H; m_H was removed)
println(">>> [2/5] Stellar parameters set: Teff=$(Teff), logg=$(logg), [M/H]=$(M_H)")

# --- Wavelength range (Å) ----------------------------------------------------
λmin = 5000.0
λmax = 5100.0
println(">>> [2/5] Wavelength range set: $(λmin) - $(λmax) Å")

# --- Linelist -----------------------------------------------------------------
# Korg has no generic `get_linelist` function — that was the original bug.
# Use one of the built-in getters, or Korg.read_linelist(path; format=...) for
# your own file. The solar VALD linelist covers 3000-9000 Å, so it's fine here.
#
# Other built-ins if you need them later:
#   Korg.get_APOGEE_DR17_linelist()  # ~15,000-17,000 Å
#   Korg.get_GALAH_DR3_linelist()    # ~4,675-7,930 Å
#   Korg.get_GES_linelist()          # Gaia-ESO, large; include_molecules=false to speed up
println(">>> [3/5] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()
println(">>> [3/5] Linelist loaded: $(length(linelist)) lines")

# --- Synthesize ---------------------------------------------------------------
# `synth` takes a `wavelengths` tuple (or vector of tuples), not λmin/λmax kwargs.
println(">>> [4/5] Running Korg.synth (this can take a few seconds)...")
wls, flux, cont = Korg.synth(
    Teff = Teff,
    logg = logg,
    M_H  = M_H,
    linelist = linelist,
    wavelengths = (λmin, λmax)
)
println(">>> [4/5] Synthesis complete.")
println(">>>       wls   : $(length(wls)) points, range $(first(wls)) - $(last(wls)) Å")
println(">>>       flux  : $(length(flux)) points, min=$(minimum(flux)), max=$(maximum(flux))")
println(">>>       cont  : $(length(cont)) points, min=$(minimum(cont)), max=$(maximum(cont))")

# --- Plot -----------------------------------------------------------------
println(">>> [5/5] Building plot...")
figure(figsize=(10, 4))
plot(wls, flux, color="black", linewidth=0.8)
xlabel("λ [Å]")
ylabel("Flux")
title("Basic LTE Synthesis | Teff=$(Teff), logg=$(logg), [M/H]=$(M_H)")
tight_layout()

# Explicitly save AND show the figure. In a non-interactive Julia session
# (e.g. running a .jl file from a terminal, rather than a notebook or the
# VSCode plot pane), PyPlot won't display anything unless you call show(),
# and the window can close instantly once the script ends — savefig gives
# you a persistent file regardless of the backend/display situation.
outpath = "korg_basic_lte_synthesis.png"
savefig(outpath, dpi=150)
println(">>> [5/5] Plot saved to: $(abspath(outpath))")

show()
println(">>> Done.")
