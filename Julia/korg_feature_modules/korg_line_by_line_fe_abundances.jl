# Line-by-line Fe I abundance determination from measured equivalent widths.
# Fits A(Fe) for each line independently (against one shared atmosphere),
# then reports the mean and scatter across lines.

println(">>> [1/7] Loading packages...")
using Korg
using Statistics   # mean() and std() live here, not in Base
println(">>> [1/7] Packages loaded: Korg, Statistics")

# ---------------------------------------------------------------------------
# Example Fe I lines (normally loaded from VALD, Kurucz, or MOOG)
# ---------------------------------------------------------------------------
# Korg.Line takes POSITIONAL arguments (wl, log_gf, species, χ), and
# `species` must be a Korg.Species object, not a raw string.
println(">>> [2/7] Defining Fe I lines...")
fe1 = Korg.Species("Fe I")

fe_lines = [
    Korg.Line(5247.1, -1.05, fe1, 0.09),
    Korg.Line(5328.0, -1.47, fe1, 0.92),
    Korg.Line(5415.2, -0.62, fe1, 4.39),
    Korg.Line(5434.5, -2.12, fe1, 1.01),
    Korg.Line(5576.1, -1.00, fe1, 3.43),
]
# Already ascending in wavelength (5247.1 < 5328.0 < ... < 5576.1), which
# Korg.Fit.ews_to_abundances requires -- no sort needed here, but worth
# checking whenever you build a linelist by hand.

fe_EWs = [70.0, 55.0, 30.0, 25.0, 40.0]  # mÅ, one per line above

println(">>> [2/7] $(length(fe_lines)) lines defined:")
for (line, ew) in zip(fe_lines, fe_EWs)
    println(">>>       λ=$(line.wl * 1e8) Å, EW=$(ew) mÅ")
end

# ---------------------------------------------------------------------------
# Stellar parameters
# ---------------------------------------------------------------------------
Teff = 5600.0
logg = 4.3
M_H  = -0.2
println(">>> [3/7] Stellar parameters: Teff=$(Teff), logg=$(logg), [M/H]=$(M_H)")

# ---------------------------------------------------------------------------
# Build the (shared) atmosphere and abundance vector once
# ---------------------------------------------------------------------------
println(">>> [4/7] Building A_X and interpolating MARCS atmosphere...")
A_X = Korg.format_A_X(M_H)
atm = Korg.interpolate_marcs(Teff, logg, A_X)
println(">>> [4/7] Atmosphere interpolated: $(length(atm.layers)) layers")

# ---------------------------------------------------------------------------
# Fit abundance for each line
# ---------------------------------------------------------------------------
# There is no `fit_abundance_from_EW` function, and no need to loop
# line-by-line calling it repeatedly (which would also rebuild the
# atmosphere every time). Korg.Fit.ews_to_abundances takes the atmosphere,
# the FULL linelist, and the FULL vector of EWs in one call, returning one
# abundance per line directly.
println(">>> [5/7] Fitting per-line abundances via Korg.Fit.ews_to_abundances...")
abundances = Korg.Fit.ews_to_abundances(atm, fe_lines, A_X, fe_EWs)
println(">>> [5/7] Fit complete.")

println(">>> [6/7] Line-by-line Fe abundances:")
for (i, A) in enumerate(abundances)
    println(">>>       Line $(i) (λ=$(fe_lines[i].wl * 1e8) Å): A(Fe) = ", A)
end

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
mean_A = mean(abundances)
std_A  = std(abundances)
println(">>> [7/7] Mean Fe abundance A(Fe) = ", mean_A)
println(">>> [7/7] Stddev across lines     = ", std_A)
println(">>> Done.")
