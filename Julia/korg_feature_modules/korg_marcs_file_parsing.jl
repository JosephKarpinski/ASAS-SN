# MARCS Atmosphere File Parsing Example (Julia)
# Reads a raw MARCS .mod file directly from disk (as opposed to interpolating
# one from Teff/logg/M_H), inspects it, and synthesizes a spectrum from it.

println(">>> [1/7] Loading packages...")
using Korg
using PyPlot
println(">>> [1/7] Packages loaded: Korg, PyPlot")

# ---------------------------------------------------------------------------
# 1. Load a raw MARCS atmosphere file
# ---------------------------------------------------------------------------
# `read_marcs` does not exist. The function for reading a MARCS .mod file
# straight off disk is `Korg.read_model_atmosphere`.
atm_path = joinpath(@__DIR__, "marcs_5000_2.5_-1.0.mod")  # replace with your file's actual path

if !isfile(atm_path)
    error("Atmosphere file not found at $(atm_path). " *
          "Make sure your .mod file is actually saved there, or update atm_path.")
end

println(">>> [2/7] Reading MARCS atmosphere file: $(atm_path)")
atm = Korg.read_model_atmosphere(atm_path)
println(">>> [2/7] Atmosphere loaded: $(length(atm.layers)) layers")

# An atmosphere doesn't expose flat `.depth`/`.T`/`.Pgas` arrays -- its data
# lives in `atm.layers` (a vector of layer structs with .tau_ref, .z, .temp,
# .electron_number_density, .number_density). Korg provides a convenience
# accessor for temperature; there's no built-in gas-pressure accessor at
# all, so we derive it ourselves via the ideal gas law (P = n_total * k_B * T),
# using total particle number density (neutrals/ions + electrons).
temps = Korg.get_temps(atm)
const K_BOLTZMANN_CGS = 1.380649e-16  # erg/K
Pgas = [(l.number_density + l.electron_number_density) * K_BOLTZMANN_CGS * l.temp
        for l in atm.layers]

println(">>> [3/7] First few temperatures (K): ", temps[1:5])
println(">>> [3/7] First few gas pressures (dyn/cm^2, derived via ideal gas law): ", Pgas[1:5])
println(">>> [3/7] Temperature range: $(minimum(temps)) - $(maximum(temps)) K")

# ---------------------------------------------------------------------------
# 2. Synthesize a spectrum using the parsed atmosphere
# ---------------------------------------------------------------------------
λmin = 5000.0
λmax = 5100.0

println(">>> [4/7] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()  # `get_linelist` does not exist
println(">>> [4/7] Linelist loaded: $(length(linelist)) lines")

# `synth` has no `atmosphere=` keyword -- it always builds its own
# atmosphere internally from Teff/logg/M_H. To synthesize from an
# atmosphere read from a file, use the lower-level `synthesize` function,
# which needs an A(X) abundance vector too. This file's name suggests
# [M/H] = -1.0 (Teff=5000, logg=2.5) -- adjust to match your actual file's
# metallicity if it differs; a mismatch here doesn't break anything, but
# the line strengths won't reflect the atmosphere's real composition.
M_H_assumed = -1.0
println(">>> [5/7] Building A_X assuming [M/H]=$(M_H_assumed) from the filename " *
        "(adjust to match your file's actual metallicity)...")
A_X = Korg.format_A_X(M_H_assumed)

println(">>> [5/7] Running Korg.synthesize with the parsed atmosphere...")
result = Korg.synthesize(atm, linelist, A_X, (λmin, λmax))

wls  = result.wavelengths
cont = result.cntm
flux = result.flux ./ result.cntm   # rectify manually, as with the interpolated-atmosphere script

println(">>> [5/7] Synthesis complete.")
println(">>>       wls   : $(length(wls)) points, range $(first(wls)) - $(last(wls)) Å")
println(">>>       flux  : min=$(minimum(flux)), max=$(maximum(flux))")

# ---------------------------------------------------------------------------
# 3. Plot the result
# ---------------------------------------------------------------------------
println(">>> [6/7] Building plot...")
figure(figsize=(10, 4))
plot(wls, flux, color="black", linewidth=0.8)
xlabel("λ [Å]")
ylabel("Rectified Flux")
title("Spectrum from Parsed MARCS Atmosphere File")
tight_layout()

outpath = "korg_marcs_file_parsing.png"
savefig(outpath, dpi=150)
println(">>> [6/7] Plot saved to: $(abspath(outpath))")
println(">>> [7/7] Done.")
