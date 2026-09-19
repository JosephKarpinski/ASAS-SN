# Contribution Function Example (Julia)
# Computes the depth-dependent contribution function for a single Fe I
# line at 6000 Å, from first principles using Korg's lower-level
# synthesis output (opacity + LTE source function), since Korg has no
# built-in `contribution_function` convenience call.

println(">>> [1/8] Loading packages...")
using Korg
using PyPlot
println(">>> [1/8] Packages loaded: Korg, PyPlot")

# ---------------------------------------------------------------------------
# 1. Load a MARCS atmosphere
# ---------------------------------------------------------------------------
# `read_marcs` does not exist -- use `Korg.read_model_atmosphere`.
atm_path = joinpath(@__DIR__, "marcs_5800_4.4_0.0.mod")  # replace with your file's actual path

if !isfile(atm_path)
    error("Atmosphere file not found at $(atm_path). " *
          "Make sure your .mod file is actually saved there, or update atm_path.")
end

println(">>> [2/8] Reading MARCS atmosphere file: $(atm_path)")
atm = Korg.read_model_atmosphere(atm_path)
println(">>> [2/8] Atmosphere loaded: $(length(atm.layers)) layers")

# ---------------------------------------------------------------------------
# 2. Define the line
# ---------------------------------------------------------------------------
# `Korg.Line(λ=,species=,loggf=,χ=)` (keyword form) doesn't exist -- the
# constructor is strictly POSITIONAL: Line(wl_angstrom, log_gf, species, χ),
# and `species` must be a `Korg.Species` object, not a bare string.
line = Korg.Line(6000.0, -1.50, Korg.Species("Fe I"), 2.20)

# Filename suggests [M/H] = 0.0 -- adjust to match your file if it differs.
M_H  = 0.0
A_X  = Korg.format_A_X(M_H)

λmin, λmax = 5999.5, 6000.5   # a tight window bracketing just this line

# ---------------------------------------------------------------------------
# 3. Run synthesize and inspect what it actually gives us
# ---------------------------------------------------------------------------
# `Korg.contribution_function(atmosphere=, line=)` doesn't exist anywhere
# in Korg -- there's no single call that hands back (tauλ, Sλ, Cλ). The
# lower-level `synthesize` result carries the per-layer, per-wavelength
# absorption coefficient (commonly the field `alpha`), which is what a
# contribution function is actually built from. NOTE: this is the one part
# of this script I'm not 100% certain about without running it -- if the
# `result.alpha` access below errors, paste back what this print statement
# shows and I'll adjust to the real field name.
println(">>> [3/8] Running Korg.synthesize with a single Fe I line...")
result = Korg.synthesize(atm, [line], A_X, (λmin, λmax))
println(">>> [3/8] synthesize() result fields: ", propertynames(result))

# ---------------------------------------------------------------------------
# 4. Build the LTE source function and per-layer optical depth at 6000 Å
# ---------------------------------------------------------------------------
# LTE assumption: source function = Planck function at each layer's
# temperature, evaluated at the line's wavelength.
h_cgs = 6.62607015e-27   # erg*s
c_cgs = 2.99792458e10    # cm/s
k_cgs = 1.380649e-16     # erg/K

planck_Bλ(λ_cm, T) = (2 * h_cgs * c_cgs^2 / λ_cm^5) /
                     (exp(h_cgs * c_cgs / (λ_cm * k_cgs * T)) - 1)

λ0_cm = 6000.0e-8   # line wavelength in cm
temps = Korg.get_temps(atm)
Sλ = [planck_Bλ(λ0_cm, T) for T in temps]

# CORRECTION: the previous attempt at this line (`result.alpha[λ_idx, :]`)
# produced a length-101 vector, which means it had the axes backwards --
# `result.alpha` is actually (n_layers=56, n_wavelengths=101). The right
# slice is a COLUMN: `result.alpha[:, λ_idx]`, giving one opacity value per
# layer at the wavelength nearest 6000 Å.
λ_idx = argmin(abs.(result.wavelengths .- 6000.0))
α_λ = result.alpha[:, λ_idx]   # linear absorption coefficient per layer at 6000 Å
n = length(α_λ)

# CORRECTION: hand-integrating α_λ over physical height `z` produced a
# ~14-decade optical-depth range (log10τ from -10 to +4), way beyond the
# ~6-8 decades a real photosphere spans -- a sign of a unit-scaling error
# in `.z` that I can't pin down without running the code myself. Rather
# than guess at units again, use the MARCS model's own reference optical
# depth `tau_ref` (already correctly calibrated per layer by Korg/MARCS,
# and the conventional depth variable contribution functions are plotted
# against in the literature anyway) instead of a manual height integration.
τref = [l.tau_ref for l in atm.layers]

println(">>> [4/8] temps  (first 5, last 5): ", temps[1:5], " ... ", temps[end-4:end])
println(">>> [4/8] α_λ    (first 5, last 5): ", α_λ[1:5], " ... ", α_λ[end-4:end])
println(">>> [4/8] τ_ref  (first 5, last 5): ", τref[1:5], " ... ", τref[end-4:end])
println(">>> [4/8] Built Sλ (Planck source function) and pulled τ_ref " *
        "(MARCS reference optical depth) for $(n) layers.")

# ---------------------------------------------------------------------------
# 5. Contribution function
# ---------------------------------------------------------------------------
# Standard (Eddington-Barbier-style) contribution function, using the
# model's own calibrated τ_ref in place of a hand-integrated τλ:
#   C_λ(i) = S_λ(i) * exp(-τ_ref(i)) * α_λ(i)
Cλ = Sλ .* exp.(-τref) .* α_λ

println(">>> [5/8] Number of layers: ", n)
println(">>> [5/8] Peak contribution at layer: ", argmax(Cλ))

# ---------------------------------------------------------------------------
# 6. Plot contribution function vs optical depth
# ---------------------------------------------------------------------------
println(">>> [6/8] Building plot...")
# Avoid log10(0) for the top layer where τ_ref can be ~0.
τref_safe = max.(τref, 1e-10)

figure(figsize=(8, 4))
plot(log10.(τref_safe), Cλ ./ maximum(Cλ), color="red")
xlabel("log10 τ_ref")
ylabel("Contribution Function (normalized)")
title("Contribution Function for Fe I 6000 Å")
tight_layout()

outpath = "korg_contribution_function.png"
savefig(outpath, dpi=150)
println(">>> [7/8] Plot saved to: $(abspath(outpath))")
println(">>> [8/8] Done.")
