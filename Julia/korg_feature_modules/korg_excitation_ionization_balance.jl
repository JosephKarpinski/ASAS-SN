# Full Excitation-Ionization Balance Example (Julia)
# Solves for Teff (Fe I excitation balance), logg (Fe I/Fe II ionization
# balance), vmic, and [M/H] from a set of measured Fe I/Fe II equivalent
# widths -- the classical spectroscopic parameter-determination method.

println(">>> [1/6] Loading packages...")
using Korg
println(">>> [1/6] Packages loaded: Korg")

# ---------------------------------------------------------------------------
# Example Fe I and Fe II lines (normally you would load these from VALD/MOOG)
# ---------------------------------------------------------------------------
# Korg.Line takes POSITIONAL arguments (wl, log_gf, species, χ), and `species`
# must be a Korg.Species object, not a raw string.
println(">>> [2/6] Defining Fe I / Fe II lines...")

fe1 = Korg.Species("Fe I")
fe2 = Korg.Species("Fe II")

raw_lines = [
    (Korg.Line(5247.1, -1.05, fe1, 0.09), 70.0),
    (Korg.Line(5328.0, -1.47, fe1, 0.92), 55.0),
    (Korg.Line(5415.2, -0.62, fe1, 4.39), 30.0),
    (Korg.Line(4923.9, -1.32, fe2, 2.89), 40.0),
    (Korg.Line(5018.4, -1.22, fe2, 2.89), 35.0),
]

# `ews_to_stellar_parameters` requires the linelist to be sorted by
# wavelength -- these five aren't, so sort the (line, EW) pairs together.
sort!(raw_lines, by = pair -> pair[1].wl)
fe_lines = [pair[1] for pair in raw_lines]
fe_EWs   = [pair[2] for pair in raw_lines]

println(">>> [2/6] Lines defined and sorted by wavelength:")
for (line, ew) in zip(fe_lines, fe_EWs)
    # Korg.Line stores wavelength internally in cm; convert back to Å for display.
    println(">>>       λ=$(line.wl * 1e8) Å, EW=$(ew) mÅ")
end

# ---------------------------------------------------------------------------
# Initial guess for stellar parameters
# ---------------------------------------------------------------------------
Teff0 = 5600.0
logg0 = 4.3
M_H0  = -0.2
println(">>> [3/6] Initial guess: Teff0=$(Teff0), logg0=$(logg0), [M/H]0=$(M_H0)")

# ---------------------------------------------------------------------------
# Solve excitation + ionization balance
# ---------------------------------------------------------------------------
# There is no `solve_excitation_ionization_balance` function in Korg. The
# actual function is Korg.Fit.ews_to_stellar_parameters(linelist,
# measured_EWs; Teff0=..., logg0=..., vmic0=..., M_H0=...) -- initial guesses
# are separate keywords, not a single `initial=` NamedTuple.
println(">>> [4/6] Solving excitation-ionization balance via Korg.Fit.ews_to_stellar_parameters...")
result = Korg.Fit.ews_to_stellar_parameters(
    fe_lines,
    fe_EWs;
    Teff0 = Teff0,
    logg0 = logg0,
    M_H0  = M_H0,
    # With only 5 lines (3 Fe I, 2 Fe II), the default max_iterations=30 and
    # tight tolerances often won't converge -- the fit is poorly constrained
    # regardless. Loosening these gives the solver more room, but won't fix
    # the underlying sparsity; a real analysis needs far more lines.
    max_iterations = 100,
    tolerances = [1e-2, 1e-2, 1e-3, 1e-2]
)
println(">>> [4/6] Solve complete.")

# The docstring describes a (params, uncertainties) pair, but earlier
# functions in this API turned out to return something simpler than
# documented -- so check what actually came back rather than assume.
println(">>> [5/6] Result type: ", typeof(result))

if result isa Tuple && length(result) == 2
    params, uncertainties = result
    Teff_fit, logg_fit, vmic_fit, MH_fit = params
    println(">>> [6/6] Excitation-Ionization Balance Solution:")
    println(">>>       Teff  = ", Teff_fit, " ± ", uncertainties[1])
    println(">>>       logg  = ", logg_fit, " ± ", uncertainties[2])
    println(">>>       vmic  = ", vmic_fit, " ± ", uncertainties[3])
    println(">>>       [M/H] = ", MH_fit,   " ± ", uncertainties[4])
else
    params = result
    Teff_fit, logg_fit, vmic_fit, MH_fit = params
    println(">>> [6/6] Excitation-Ionization Balance Solution:")
    println(">>>       Teff  = ", Teff_fit)
    println(">>>       logg  = ", logg_fit)
    println(">>>       vmic  = ", vmic_fit)
    println(">>>       [M/H] = ", MH_fit)
end
println(">>> Done.")
