# Custom Solar Scale Example (Julia)
# Demonstrates overriding individual elements' A(X) = log10(N_X/N_H) + 12
# abundances relative to Korg's built-in solar reference.

println(">>> [1/7] Loading packages...")
using Korg
using PyPlot
println(">>> [1/7] Packages loaded: Korg, PyPlot")

# ---------------------------------------------------------------------------
# 1. Define a custom solar abundance scale
#    Values are ABSOLUTE A(X) = log10(N_X / N_H) + 12
# ---------------------------------------------------------------------------
custom_solar = Dict(
    :C  => 8.43,
    :N  => 7.83,
    :O  => 8.69,
    :Mg => 7.60,
    :Si => 7.51,
    :Ca => 6.34,
    :Ti => 4.95,
    :Fe => 7.50,
)

println(">>> [2/7] Custom solar scale loaded.")
println(">>> [2/7] Solar Fe abundance = ", custom_solar[:Fe])

# ---------------------------------------------------------------------------
# 2. IMPORTANT: `synth`'s per-element keywords (e.g. `Fe=...`) are NOT
#    absolute A(X) overrides -- they're OFFSETS added on top of the default
#    solar scale. Passing Fe=7.50 there means "+7.50 dex on top of solar Fe",
#    not "set A(Fe) to 7.50". That's exactly what blew up: Korg's internal
#    M_H-inference (used to pick a point in the MARCS grid) saw an effective
#    metallicity of ~+7.86, miles outside the grid's [-2.5, 1.0] range.
#
#    To set ABSOLUTE A(X) values -- i.e. an actual custom solar reference
#    scale -- you have to build the full 92-element A(X) vector yourself and
#    drop to the low-level `interpolate_marcs`/`synthesize` calls, which is
#    the same pattern used for reading MARCS files directly from disk.
# ---------------------------------------------------------------------------

# Minimal Z lookup for the elements we need to override (extend as needed).
const ELEMENT_SYMBOLS = ["H","He","Li","Be","B","C","N","O","F","Ne",
                          "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
                          "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"]
atomic_number(sym::AbstractString) = findfirst(==(sym), ELEMENT_SYMBOLS)

# `Korg.format_A_X()` with no arguments (M_H=0, no overrides) returns exactly
# Korg's default solar reference vector -- this is both our starting point
# AND the reference we tell `interpolate_marcs` to measure metallicity
# against, so the two stay consistent.
solar_ref = Korg.format_A_X()

custom_A_X = copy(solar_ref)
for (elem, absA) in custom_solar
    Z = atomic_number(String(elem))
    custom_A_X[Z] = absA
end

println(">>> [3/7] Built custom A_X vector (92 elements), overriding: " *
        join(sort(collect(keys(custom_solar))), ", "))

# Note: these particular custom_solar values happen to already match Korg's
# built-in defaults (Grevesse 2007), so this run should reproduce the
# default spectrum almost exactly -- change a value above to actually see a
# difference (e.g. custom_solar[:Fe] = 6.50 for a strongly Fe-poor scale).

# ---------------------------------------------------------------------------
# 3. Synthesize using the custom scale
# ---------------------------------------------------------------------------
Teff = 5777.0
logg = 4.44

λmin = 6000.0
λmax = 6020.0

println(">>> [4/7] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()  # `get_linelist` does not exist
println(">>> [4/7] Linelist loaded: $(length(linelist)) lines")

println(">>> [5/7] Interpolating MARCS atmosphere against the custom scale...")
# `solar_abundances=` and `clamp_abundances=` are real keywords here (as
# revealed by the earlier error's own stack trace) -- passing our own
# `solar_ref` as the comparison point keeps the inferred grid metallicity
# sane, and `clamp_abundances=true` is cheap insurance against edge cases.
atm = Korg.interpolate_marcs(Teff, logg, custom_A_X;
                              solar_abundances=solar_ref,
                              clamp_abundances=true)

println(">>> [6/7] Running Korg.synthesize with the custom A_X vector...")
result = Korg.synthesize(atm, linelist, custom_A_X, (λmin, λmax))

wls  = result.wavelengths
flux = result.flux ./ result.cntm   # rectify manually

println(">>> [6/7] Synthesis complete: $(length(wls)) points, " *
        "range $(first(wls)) - $(last(wls)) Å")
println(">>> [6/7] flux: min=$(minimum(flux)), max=$(maximum(flux))")

# ---------------------------------------------------------------------------
# 4. Plot the result
# ---------------------------------------------------------------------------
println(">>> [7/7] Building plot...")

# Build a compact "A(X)=value" label for each overridden element, sorted by
# atomic number so the order is stable/readable, and fold it into a second
# title line so the plot is self-documenting about which scale made it.
overrides_str = join(["$(elem)=$(custom_solar[elem])"
                       for elem in sort(collect(keys(custom_solar));
                                        by=e -> atomic_number(String(e)))],
                      ", ")

figure(figsize=(10, 4))
plot(wls, flux, color="black", linewidth=0.8)
xlabel("λ [Å]")
ylabel("Flux")
title("Spectrum Using Custom Solar Abundance Scale\n" *
      "A(X) overrides: $(overrides_str)", fontsize=10)
tight_layout()

outpath = "korg_custom_solar_scale.png"
savefig(outpath, dpi=150)
println(">>> [7/7] Plot saved to: $(abspath(outpath))")
println(">>> [7/7] Done.")
