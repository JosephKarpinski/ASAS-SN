# Full Custom Linelist Loading Example (Julia)
# Loads a user-supplied multi-element linelist file and synthesizes a
# spectrum with it.
#
# NOTE ON FORMAT: Korg's built-in `read_linelist(...; format="moog")` parser
# errored on this file's layout (an internal `+(::Nothing, ::Int64)`
# MethodError, likely a fixed-width column-slicing assumption that doesn't
# match this whitespace-separated file). Rather than reverse-engineer
# Korg's exact expected column positions, we parse it ourselves (its
# columns are simple and documented in its own header: wavelength, species
# code, E_low, loggf) and build `Korg.Line` objects directly.

println(">>> [1/8] Loading packages...")
using Korg
using PyPlot
println(">>> [1/8] Packages loaded: Korg, PyPlot")

# ---------------------------------------------------------------------------
# Element symbols by atomic number, needed to turn a MOOG species code like
# 26.0 into "Fe" (Korg needs a species STRING like "Fe I", not a numeric code).
# Covers every element in this 40-line file (Mg=12, Si=14, Ca=20, Ti=22,
# Fe=26) plus some headroom.
# ---------------------------------------------------------------------------
const ELEMENT_SYMBOLS = [
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
]
const ROMAN_IONIZATION = ["I", "II", "III", "IV", "V"]

"""
    moog_species_to_string(code)

Convert a MOOG-style species code (e.g. 26.0 = Fe I, 26.1 = Fe II) into the
"<Symbol> <Roman ionization>" string Korg.Species expects.
"""
function moog_species_to_string(code::Real)
    Z = Int(floor(code))
    ion_stage = round(Int, (code - Z) * 10)  # 0 -> neutral, 1 -> singly ionized, ...
    symbol = ELEMENT_SYMBOLS[Z]
    return "$(symbol) $(ROMAN_IONIZATION[ion_stage + 1])"
end

# ---------------------------------------------------------------------------
# 1. Parse the custom linelist file ourselves
# ---------------------------------------------------------------------------
linelist_path = joinpath(@__DIR__, "my_extended_linelist.txt")

if !isfile(linelist_path)
    error("Linelist file not found at $(linelist_path). " *
          "Make sure you've saved the linelist content to an actual file " *
          "at that path (or update linelist_path to point at it).")
end

println(">>> [2/8] Parsing custom linelist: $(linelist_path)")
linelist = Korg.Line[]
species_labels = String[]   # parallel array, same order as `linelist`, for the summary below
for raw_line in eachline(linelist_path)
    line = strip(raw_line)
    (isempty(line) || startswith(line, "#")) && continue  # skip blanks/comments

    fields = split(line)
    wl_A, species_code, chi, log_gf = parse.(Float64, fields[1:4])

    species_str = moog_species_to_string(species_code)
    species = Korg.Species(species_str)
    push!(linelist, Korg.Line(wl_A, log_gf, species, chi))
    push!(species_labels, species_str)
end

# Sort lines and labels together by wavelength -- required for some Korg.Fit
# functions, and good practice in general.
order = sortperm(linelist, by = l -> l.wl)
linelist = linelist[order]
species_labels = species_labels[order]

println(">>> [2/8] Number of lines loaded: ", length(linelist))

# ---------------------------------------------------------------------------
# Per-species summary: count and wavelength range for each element/ion,
# so you can confirm the file parsed the way you expect at a glance.
# ---------------------------------------------------------------------------
println(">>> [3/8] Per-species summary:")
for species_str in unique(species_labels)
    idxs = findall(==(species_str), species_labels)
    wls_A = [linelist[i].wl * 1e8 for i in idxs]
    println(">>>       $(species_str): $(length(idxs)) lines, " *
            "$(round(minimum(wls_A), digits=2))-$(round(maximum(wls_A), digits=2)) Å")
end

# ---------------------------------------------------------------------------
# 2. Define stellar parameters
# ---------------------------------------------------------------------------
Teff = 5000.0
logg = 2.5
M_H  = -1.0
println(">>> [4/8] Stellar parameters: Teff=$(Teff), logg=$(logg), [M/H]=$(M_H)")

# ---------------------------------------------------------------------------
# 3. Synthesize spectrum using the custom linelist
# ---------------------------------------------------------------------------
# Lines now span 4923.9-6145.0 Å (Fe II up through Si I), so the window is
# widened to bracket all of them. Note this is a much bigger window than
# earlier scripts (>1200 Å vs ~20-100 Å), so synthesis will take noticeably
# longer at the default 0.01 Å step (~122,000 points).
λmin = 4900.0
λmax = 6160.0
println(">>> [5/8] Synthesizing spectrum over $(λmin)-$(λmax) Å " *
        "with the extended custom linelist (this will take a while)...")
wls, flux, cont = Korg.synth(
    Teff = Teff,
    logg = logg,
    M_H  = M_H,
    linelist = linelist,
    wavelengths = (λmin, λmax)
)
println(">>> [5/8] Synthesis complete: $(length(wls)) points.")
println(">>>       flux: min=$(minimum(flux)), max=$(maximum(flux))")

# ---------------------------------------------------------------------------
# 4. Plot the result
# ---------------------------------------------------------------------------
println(">>> [6/8] Building plot...")
figure(figsize=(14, 4))
plot(wls, flux, color="black", linewidth=0.6)
xlabel("λ [Å]")
ylabel("Flux")
title("Spectrum Using Extended Custom Linelist (Fe I/II, Mg I, Ca I, Ti I, Si I)")
tight_layout()

outpath = "korg_extended_linelist_loading.png"
savefig(outpath, dpi=150)
println(">>> [7/8] Plot saved to: $(abspath(outpath))")
println(">>> [8/8] Done.")
