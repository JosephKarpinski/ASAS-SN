# Round-trip validation of the EW -> stellar parameters pipeline.
#
# Rather than trusting hand-picked EWs (which may not correspond to any real
# star, as the previous script's non-convergence suggested), this generates
# EWs that ARE self-consistent with a known atmosphere, then checks whether
# Korg.Fit.ews_to_stellar_parameters can recover the parameters that
# generated them. This isolates "is the workflow correct?" from "is this
# particular EW data physical?".

println(">>> [1/8] Loading packages...")
using Korg
println(">>> [1/8] Packages loaded: Korg")

# ---------------------------------------------------------------------------
# "True" stellar parameters used to generate the synthetic EWs
# ---------------------------------------------------------------------------
Teff_true = 5750.0
logg_true = 4.30
M_H_true  = -0.30
vmic_true = 1.20
println(">>> [2/8] True parameters: Teff=$(Teff_true), logg=$(logg_true), " *
        "[M/H]=$(M_H_true), vmic=$(vmic_true)")

# ---------------------------------------------------------------------------
# Fe I / Fe II line list (same atomic data as before, sorted by wavelength)
# ---------------------------------------------------------------------------
println(">>> [3/8] Defining Fe I / Fe II lines...")
fe1 = Korg.Species("Fe I")
fe2 = Korg.Species("Fe II")

raw_lines = [
    Korg.Line(4923.9, -1.32, fe2, 2.89),
    Korg.Line(5018.4, -1.22, fe2, 2.89),
    Korg.Line(5247.1, -1.05, fe1, 0.09),
    Korg.Line(5328.0, -1.47, fe1, 0.92),
    Korg.Line(5415.2, -0.62, fe1, 4.39),
]
fe_lines = sort(raw_lines, by = line -> line.wl)
println(">>> [3/8] $(length(fe_lines)) lines defined, sorted by wavelength.")

# ---------------------------------------------------------------------------
# Build the "true" atmosphere and generate self-consistent EWs from it
# ---------------------------------------------------------------------------
println(">>> [4/8] Building A_X and interpolating the TRUE atmosphere...")
A_X_true = Korg.format_A_X(M_H_true)
atm_true = Korg.interpolate_marcs(Teff_true, logg_true, A_X_true)
println(">>> [4/8] True atmosphere interpolated: $(length(atm_true.layers)) layers")

println(">>> [5/8] Computing self-consistent EWs from the true atmosphere...")
true_EWs = try
    Korg.Fit.calculate_EWs(atm_true, fe_lines, A_X_true; vmic = vmic_true)
catch e
    if e isa MethodError
        println(">>>       `vmic` keyword not accepted by calculate_EWs in " *
                "this Korg version -- falling back to its default vmic (1.0 km/s).")
        Korg.Fit.calculate_EWs(atm_true, fe_lines, A_X_true)
    else
        rethrow(e)
    end
end
println(">>> [5/8] True (self-consistent) EWs:")
for (line, ew) in zip(fe_lines, true_EWs)
    println(">>>       λ=$(line.wl * 1e8) Å -> EW=$(round(ew, digits=2)) mÅ")
end

# ---------------------------------------------------------------------------
# Deliberately offset initial guess -- NOT the true values -- to test recovery
# ---------------------------------------------------------------------------
Teff0 = Teff_true + 250.0   # off by +250 K
logg0 = logg_true + 0.3     # off by +0.3 dex
M_H0  = M_H_true  + 0.2     # off by +0.2 dex
println(">>> [6/8] Offset initial guess for the solver: Teff0=$(Teff0), " *
        "logg0=$(logg0), [M/H]0=$(M_H0)")

# ---------------------------------------------------------------------------
# Attempt to recover the true parameters from the self-consistent EWs
# ---------------------------------------------------------------------------
println(">>> [7/8] Running Korg.Fit.ews_to_stellar_parameters on the self-consistent EWs...")
result = Korg.Fit.ews_to_stellar_parameters(
    fe_lines,
    true_EWs;
    Teff0 = Teff0,
    logg0 = logg0,
    M_H0  = M_H0
)
println(">>> [7/8] Solve complete. Result type: ", typeof(result))

if result isa Tuple && length(result) == 2
    params, uncertainties = result
else
    params = result
    uncertainties = fill(NaN, length(params))
end
Teff_fit, logg_fit, vmic_fit, MH_fit = params

# ---------------------------------------------------------------------------
# Compare recovered parameters to the ground truth
# ---------------------------------------------------------------------------
println(">>> [8/8] Recovered vs. true parameters:")
println(">>>       Teff  : recovered=$(Teff_fit), true=$(Teff_true), Δ=$(Teff_fit - Teff_true)")
println(">>>       logg  : recovered=$(logg_fit), true=$(logg_true), Δ=$(logg_fit - logg_true)")
println(">>>       vmic  : recovered=$(vmic_fit), true=$(vmic_true), Δ=$(vmic_fit - vmic_true)")
println(">>>       [M/H] : recovered=$(MH_fit), true=$(M_H_true), Δ=$(MH_fit - M_H_true)")
println(">>> Done.")
