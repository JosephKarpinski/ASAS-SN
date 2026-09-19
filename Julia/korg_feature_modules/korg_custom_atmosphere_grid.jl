# Custom Atmosphere Grid Example (Julia)
#
# The original version of this script tried to: glob a directory of MARCS
# files, read them all into a Dict, then call `Korg.interpolate_atmosphere`
# on that Dict to get an atmosphere at off-grid parameters. That function
# doesn't exist -- Korg has no public API for interpolating over a
# user-assembled collection of atmospheres. Its real interpolation function,
# `Korg.interpolate_marcs` (used successfully in earlier scripts), already
# does off-grid Teff/logg/[M/H] interpolation, but against Korg's own
# BUNDLED MARCS grid data, not files you download and read yourself.
#
# So rather than reading potentially thousands of files just to reimplement
# something Korg already provides, this version does the interpolation the
# way Korg actually supports it. The file-loading section is kept below,
# commented out, in case you specifically need atmospheres from a
# non-standard/custom grid Korg doesn't ship (e.g. your own model grid) --
# but note that would require writing your own interpolation logic, since
# Korg doesn't have a built-in for it.

println(">>> [1/6] Loading packages...")
using Korg
using PyPlot
println(">>> [1/6] Packages loaded: Korg, PyPlot")

# ---------------------------------------------------------------------------
# (Reference only -- NOT run) Loading a directory of your own atmosphere
# files, for cases where you truly have a custom/non-standard grid:
# ---------------------------------------------------------------------------
# using Glob
# atm_dir = "my_marcs_grid/"
# files = glob("*.mod.gz", atm_dir)
# grid = Dict{Tuple{Float64,Float64,Float64}, Any}()   # `Korg.Atmosphere`
#                                                        # isn't a real type
#                                                        # name -- use `Any`
#                                                        # (or leave the Dict
#                                                        # untyped) instead.
# for f in files
#     # Real MARCS filenames look like:
#     #   p5000_g+3.0_m0.0_t01_st_z-1.00_a+0.40_c+0.00_n+0.00_o+0.40_r+0.00_s+0.00.mod.gz
#     # -- adjust this regex to match YOUR actual filenames; the toy pattern
#     # in the original script (marcs_5000_2.5_-1.0.mod) won't match real
#     # downloaded MARCS files at all.
#     m = match(r"...", basename(f))
#     ...
#     atm = Korg.read_model_atmosphere(f)   # NOT `read_marcs`
#     grid[(Teff, logg, M_H)] = atm
# end
# # Interpolating over `grid` from here would need hand-written
# # multilinear interpolation -- Korg has no built-in for this.

# ---------------------------------------------------------------------------
# 1. Define target parameters
# ---------------------------------------------------------------------------
target_Teff = 4950.0
target_logg = 2.3
target_MH   = -0.3

# ---------------------------------------------------------------------------
# 2. Interpolate the atmosphere using Korg's own built-in grid
# ---------------------------------------------------------------------------
# This is what actually accomplishes "get an atmosphere at off-grid
# Teff/logg/[M/H]" in Korg -- no need to read any files from disk at all,
# since Korg ships its own MARCS grid data internally.
println(">>> [2/6] Interpolating atmosphere at Teff=$(target_Teff), " *
        "logg=$(target_logg), [M/H]=$(target_MH) via Korg's built-in grid...")
A_X = Korg.format_A_X(target_MH)
atm_interp = Korg.interpolate_marcs(target_Teff, target_logg, A_X)
println(">>> [2/6] Interpolated atmosphere: $(length(atm_interp.layers)) layers")

# `atm_interp.T` doesn't exist -- temperature lives in `atm_interp.layers`,
# accessed via `Korg.get_temps`.
temps = Korg.get_temps(atm_interp)
println(">>> [2/6] First few T layers: ", temps[1:5])

# ---------------------------------------------------------------------------
# 3. Synthesize spectrum using the interpolated atmosphere
# ---------------------------------------------------------------------------
# `synth` has no `atmosphere=` keyword (it always builds its own internally)
# and no `λmin=`/`λmax=` keywords. To synthesize from an atmosphere object
# directly, drop to the low-level `synthesize`.
λmin = 6000.0
λmax = 6020.0

println(">>> [3/6] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()  # `get_linelist` does not exist
println(">>> [3/6] Linelist loaded: $(length(linelist)) lines")

println(">>> [4/6] Running Korg.synthesize with the interpolated atmosphere...")
result = Korg.synthesize(atm_interp, linelist, A_X, (λmin, λmax))

wls  = result.wavelengths
flux = result.flux ./ result.cntm   # rectify manually

println(">>> [4/6] Synthesis complete: $(length(wls)) points")

# ---------------------------------------------------------------------------
# 4. Plot result
# ---------------------------------------------------------------------------
println(">>> [5/6] Building plot...")
figure(figsize=(10, 4))
plot(wls, flux, color="black", linewidth=0.8)
xlabel("λ [Å]")
ylabel("Flux")
title("Spectrum from Korg-Interpolated Atmosphere " *
      "(Teff=$(target_Teff), logg=$(target_logg), [M/H]=$(target_MH))")
tight_layout()

outpath = "korg_custom_atmosphere_grid.png"
savefig(outpath, dpi=150)
println(">>> [6/6] Plot saved to: $(abspath(outpath))")
println(">>> [6/6] Done.")
