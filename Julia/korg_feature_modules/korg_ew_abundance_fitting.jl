# Full Equivalent-Width Abundance Fitting Example (Julia)
# Solves for the abundance of Mg that reproduces a measured equivalent width,
# given stellar parameters and a line's atomic data.

println(">>> [1/6] Loading packages...")
using Korg
println(">>> [1/6] Packages loaded: Korg")

# --- Define a spectral line ---------------------------------------------------
# Korg.Line takes POSITIONAL arguments: (wl, log_gf, species, E_lower, ...).
# There is no keyword constructor (λ=, species=, loggf= all fail), and
# `species` must be an actual Korg.Species object, built from the string.
println(">>> [2/6] Defining the Mg I line...")
mg_species = Korg.Species("Mg I")
line = Korg.Line(5528.4, -0.498, mg_species, 4.34)  # wl [Å], log_gf, species, χ [eV]
println(">>> [2/6] Line defined: λ=5528.4 Å, species=Mg I, log_gf=-0.498, χ=4.34 eV")

# --- Observed equivalent width -------------------------------------------------
EW_obs = 120.0   # mÅ
println(">>> [2/6] Observed EW: $(EW_obs) mÅ")

# --- Stellar parameters ---------------------------------------------------------
Teff = 5000.0
logg = 2.5
M_H  = -1.0
println(">>> [3/6] Stellar parameters set: Teff=$(Teff), logg=$(logg), [M/H]=$(M_H)")

# --- Build abundance vector and atmosphere ---------------------------------------
println(">>> [4/6] Building A_X and interpolating MARCS atmosphere...")
A_X = Korg.format_A_X(M_H)
atm = Korg.interpolate_marcs(Teff, logg, A_X)
println(">>> [4/6] Atmosphere interpolated: $(length(atm.layers)) layers")

# --- Fit abundance from EW -------------------------------------------------------
# There is no `fit_abundance_from_EW` function in Korg. The actual API for
# this "measured EW -> abundance" inversion is Korg.Fit.ews_to_abundances,
# which takes a model atmosphere, a (wavelength-sorted) linelist, an A_X
# vector, and a VECTOR of measured EWs (mÅ) -- it's built to process many
# lines against one atmosphere in a single call, not just one.
# Note: despite what the docstring describes, this installed version returns
# a single Vector{Float64} of per-line abundances (not an (abundances,
# slopes) tuple) -- so capture it as one variable, not two.
println(">>> [5/6] Fitting abundance from EW via Korg.Fit.ews_to_abundances...")
linelist = [line]  # must be sorted by wavelength; trivially true for one line
abundances = Korg.Fit.ews_to_abundances(atm, linelist, A_X, [EW_obs])
println(">>> [5/6] Fit complete.")

println(">>> [6/6] Best-fit Mg abundance A(Mg) = ", abundances[1])
println(">>> Done.")
