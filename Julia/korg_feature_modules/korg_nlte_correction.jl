# NLTE Correction Example (Julia)
# Applies a user-supplied NLTE correction function on top of an LTE
# abundance derived from a measured equivalent width, then re-synthesizes
# using the NLTE-corrected abundance.

println(">>> [1/8] Loading packages...")
using Korg
using PyPlot
println(">>> [1/8] Packages loaded: Korg, PyPlot")

# ---------------------------------------------------------------------------
# 1. Example NLTE correction function (unchanged -- this is just your own
#    plain Julia function, nothing Korg-specific about it)
# ---------------------------------------------------------------------------
function nlte_correction(element::String, Teff, logg, M_H)
    if element == "Fe"
        return 0.10 * exp(-(Teff - 5000) / 1500) + 0.02 * (4.5 - logg) + 0.03 * M_H
    elseif element == "Mg"
        return -0.05 * exp(-(Teff - 4800) / 1200)
    else
        return 0.0
    end
end

# ---------------------------------------------------------------------------
# 2. Build the line
# ---------------------------------------------------------------------------
# `Korg.Line(λ=,species=,loggf=,χ=)` (keyword form) doesn't exist -- the
# constructor is strictly POSITIONAL: Line(wl_angstrom, log_gf, species, χ),
# and `species` must be a `Korg.Species` object, not a bare string.
line = Korg.Line(5528.4, -0.498, Korg.Species("Mg I"), 4.34)

EW_obs = 120.0   # mÅ

Teff = 5000.0
logg = 2.5
M_H  = -1.0

# ---------------------------------------------------------------------------
# 3. LTE abundance from EW
# ---------------------------------------------------------------------------
# `Korg.fit_abundance_from_EW` doesn't exist anywhere in Korg. The real
# function operates on a full atmosphere + A(X) vector + a list of
# lines/EWs at once: `Korg.Fit.ews_to_abundances(atm, linelist, A_X,
# measured_EWs)`. It returns a plain Vector{Float64} (one abundance per
# line), not a scalar and not a (abundances, slopes) tuple.
println(">>> [2/8] Building baseline atmosphere and A_X for Teff=$(Teff), " *
        "logg=$(logg), [M/H]=$(M_H)...")
A_X_baseline = Korg.format_A_X(M_H)
atm = Korg.interpolate_marcs(Teff, logg, A_X_baseline)
println(">>> [2/8] Atmosphere built: $(length(atm.layers)) layers")

println(">>> [3/8] Fitting LTE abundance from EW_obs=$(EW_obs) mÅ...")
A_Mg_LTE_vec = Korg.Fit.ews_to_abundances(atm, [line], A_X_baseline, [EW_obs])
A_Mg_LTE = A_Mg_LTE_vec[1]

println(">>> [3/8] LTE abundance A(Mg) = ", A_Mg_LTE)

# ---------------------------------------------------------------------------
# 4. Apply NLTE correction
# ---------------------------------------------------------------------------
ΔNLTE = nlte_correction("Mg", Teff, logg, M_H)
A_Mg_NLTE = A_Mg_LTE + ΔNLTE

println(">>> [4/8] NLTE correction ΔNLTE = ", ΔNLTE)
println(">>> [4/8] NLTE abundance A(Mg) = ", A_Mg_NLTE)

# ---------------------------------------------------------------------------
# 5. Synthesize spectrum using the NLTE-corrected abundance
# ---------------------------------------------------------------------------
# `synth` has no `abundances=Dict(...)` keyword, no `λmin=`/`λmax=`
# keywords, and its per-element keywords (e.g. `Mg=...`) are OFFSETS added
# on top of the metallicity-scaled default -- NOT absolute A(X) values
# (this is exactly what caused the [M/H] range-explosion bug in the custom
# solar-scale script: passing an absolute-looking number there gets added
# on top of the existing abundance, not substituted for it).
#
# Since A_Mg_NLTE IS meant to be an absolute A(X) value, the robust way to
# set it is the same one used to fix that earlier script: build the full
# 92-element A_X vector by hand, starting from the star's baseline
# composition, and override just the Mg entry.
println(">>> [5/8] Building final A_X vector with NLTE-corrected A(Mg)...")
const ELEMENT_SYMBOLS = ["H","He","Li","Be","B","C","N","O","F","Ne",
                          "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
                          "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"]
atomic_number(sym::AbstractString) = findfirst(==(sym), ELEMENT_SYMBOLS)

A_X_final = copy(A_X_baseline)
A_X_final[atomic_number("Mg")] = A_Mg_NLTE

λmin = 5520.0
λmax = 5535.0

println(">>> [6/8] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()  # `get_linelist` does not exist
println(">>> [6/8] Linelist loaded: $(length(linelist)) lines")

# The atmosphere itself is essentially insensitive to a single trace
# element's abundance, so we reuse `atm` from step 2 rather than
# re-interpolating -- only the A_X vector fed to `synthesize` changes.
println(">>> [7/8] Running Korg.synthesize with NLTE-corrected A_X...")
result = Korg.synthesize(atm, linelist, A_X_final, (λmin, λmax))

wls  = result.wavelengths
flux = result.flux ./ result.cntm   # rectify manually

println(">>> [7/8] Synthesis complete: $(length(wls)) points, " *
        "range $(first(wls)) - $(last(wls)) Å")

# ---------------------------------------------------------------------------
# 6. Plot the result
# ---------------------------------------------------------------------------
println(">>> [8/8] Building plot...")
figure(figsize=(10, 4))
plot(wls, flux, color="black")
xlabel("λ [Å]")
ylabel("Flux")
title("Spectrum with NLTE-corrected Mg abundance " *
      "(A(Mg)_LTE=$(round(A_Mg_LTE, digits=3)), " *
      "ΔNLTE=$(round(ΔNLTE, digits=3)))")
tight_layout()

outpath = "korg_nlte_correction.png"
savefig(outpath, dpi=150)
println(">>> [8/8] Plot saved to: $(abspath(outpath))")
println(">>> [8/8] Done.")
