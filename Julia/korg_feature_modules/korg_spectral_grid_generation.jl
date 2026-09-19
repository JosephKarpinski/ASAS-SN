# Full Spectral Grid Generation Example (Julia)
# Generates a small grid of synthetic spectra over Teff x logg x [M/H] x vmic.

println(">>> [1/4] Loading packages...")
using Korg
using PyPlot
println(">>> [1/4] Packages loaded: Korg, PyPlot")

# ---------------------------------------------------------------------------
# 1. Define grid axes
# ---------------------------------------------------------------------------
Teffs = [4800.0, 5000.0, 5200.0]
loggs = [2.0, 2.5]
MHs   = [-0.5, 0.0]
vmics = [1.0, 1.5]

λmin = 6000.0
λmax = 6020.0

n_combos = length(Teffs) * length(loggs) * length(MHs) * length(vmics)
println(">>> [1/4] Grid axes defined: $(n_combos) total combinations")

# ---------------------------------------------------------------------------
# 2. Load the linelist ONCE, outside the loop
# ---------------------------------------------------------------------------
# `get_linelist(:vald)` doesn't exist -- use `get_VALD_solar_linelist()`.
# Loading it here (rather than inside the loop) avoids re-parsing ~42k VALD
# lines on every one of the 24 grid points below -- the same fix applied to
# the ForwardDiff-derivative script earlier.
println(">>> [2/4] Loading VALD solar linelist...")
linelist = Korg.get_VALD_solar_linelist()
println(">>> [2/4] Linelist loaded: $(length(linelist)) lines")

# ---------------------------------------------------------------------------
# 3. Storage container for the grid
# ---------------------------------------------------------------------------
grid = Dict{Tuple{Float64,Float64,Float64,Float64}, Tuple{Vector{Float64},Vector{Float64}}}()

# ---------------------------------------------------------------------------
# 4. Generate the grid
# ---------------------------------------------------------------------------
# `synth` has no `λmin=`/`λmax=` keywords -- needs `wavelengths=(λmin,λmax)`.
println(">>> [3/4] Generating grid ($(n_combos) syntheses; this will take a while)...")
i = 0
for Teff in Teffs
    for logg in loggs
        for M_H in MHs
            for vmic in vmics
                global i += 1
                local wls, flux
                wls, flux, _ = Korg.synth(
                    Teff        = Teff,
                    logg        = logg,
                    M_H         = M_H,
                    vmic        = vmic,
                    wavelengths = (λmin, λmax),
                    linelist    = linelist
                )

                key = (Teff, logg, M_H, vmic)
                grid[key] = (wls, flux)

                println(">>>   [$(i)/$(n_combos)] Generated spectrum for ", key)
            end
        end
    end
end

println(">>> [4/4] Grid size = ", length(grid))

# ---------------------------------------------------------------------------
# 5. Quick sanity-check plot: overlay a few grid members
# ---------------------------------------------------------------------------
# The grid itself only lives in memory (`grid`) for the rest of this Julia
# session -- if you want to reuse it later without re-synthesizing, save it
# to disk (e.g. with JLD2.jl) rather than relying on it staying in the REPL.
println(">>> [4/4] Building sanity-check plot (3 grid members)...")
figure(figsize=(10, 4))
sample_keys = [(4800.0, 2.0, -0.5, 1.0), (5000.0, 2.5, 0.0, 1.0), (5200.0, 2.5, 0.0, 1.5)]
for key in sample_keys
    local wls, flux
    wls, flux = grid[key]
    plot(wls, flux, linewidth=0.7, label="Teff=$(key[1]), logg=$(key[2]), [M/H]=$(key[3]), vmic=$(key[4])")
end
xlabel("λ [Å]")
ylabel("Flux")
title("Sample Spectra from the Generated Grid")
legend(fontsize=7)
tight_layout()

outpath = "korg_spectral_grid_generation.png"
savefig(outpath, dpi=150)
println(">>> [4/4] Plot saved to: $(abspath(outpath))")
println(">>> [4/4] Done.")
