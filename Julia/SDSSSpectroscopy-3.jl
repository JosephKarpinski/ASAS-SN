# =============================================================================
# SDSSSpectroscopy.jl
#
# Standalone Julia module based on the JuliaAstro tutorial
#   "Spectroscopy with SDSS"
#   https://learn.juliaastro.org/tutorials/spectroscopy-sdss/
#   (authors: Aditya Kumar Pandey, Chris Garling, Ian Weaver)
#
# Sections (same as the tutorial):
#   1. Load a real SDSS DR14 spectrum (plate 1323, MJD 52797, fiber 12)
#   2. Physical unit conversions
#   3. Dust extinction / dereddening (SFD98 map)
#   4. Blackbody spectra and stellar luminosity density
#   5. Cosmological redshift and surface-brightness dimming
#
# HOW TO RUN (VSCode): open this file and press the "Run file" (▷) tab.
#   * The working directory is set to the folder containing this file.
#   * All files are read/written there: FITS file, Project.toml/Manifest.toml,
#     and every PNG plot.
#   * First run creates a Julia environment in this folder and installs the
#     packages (this can take several minutes). Later runs skip that step.
#
# NOTE: like the tutorial, this uses development branches of several
# JuliaAstro packages plus Makie 0.25 (unit-aware axes via DQConversion).
# If those branches have since been merged/renamed, edit `PACKAGE_SPECS` below.
# =============================================================================

using Dates
import Pkg

# ----------------------------------------------------------------------------
# Folder handling: everything lives next to this file
# ----------------------------------------------------------------------------
const SCRIPT_DIR = @__DIR__
cd(SCRIPT_DIR)

# ----------------------------------------------------------------------------
# Debug printing (available before the module is loaded)
# ----------------------------------------------------------------------------
const T_START = time()

function dbg(msg::AbstractString)
    elapsed = round(time() - T_START; digits = 1)
    println("[DEBUG $(Dates.format(now(), "HH:MM:SS")) | +$(elapsed)s] $msg")
    flush(stdout)
    return nothing
end

dbg("Script started. Working directory set to: $(pwd())")

# ----------------------------------------------------------------------------
# Environment bootstrap (runs BEFORE the module so `using` finds the packages)
# ----------------------------------------------------------------------------
const PACKAGE_SPECS = [
    Pkg.PackageSpec(; name = "MathTeXEngine"),
    Pkg.PackageSpec(;
        url = "https://github.com/MakieOrg/Makie.jl",
        subdir = "Makie",
        rev = "ff/breaking-0.25",
    ),
    Pkg.PackageSpec(;
        url = "https://github.com/MakieOrg/Makie.jl",
        subdir = "ComputePipeline",
        rev = "ff/breaking-0.25",
    ),
    Pkg.PackageSpec(;
        url = "https://github.com/MakieOrg/Makie.jl",
        subdir = "CairoMakie",
        rev = "ff/breaking-0.25",
    ),
    Pkg.PackageSpec(;
        url = "https://github.com/icweaver/Measurements.jl",
        rev = "makie-v0.25.0",
    ),
    Pkg.PackageSpec(;
        url = "https://github.com/JuliaAstro/SpectrumBase.jl",
        rev = "units",
    ),
    Pkg.PackageSpec(;
        url = "https://github.com/JuliaAstro/DustExtinction.jl",
        rev = "units",
    ),
    Pkg.PackageSpec(;
        url = "https://github.com/JuliaAstro/SkyCoords.jl",
        rev = "makie-v0.25",
    ),
    Pkg.PackageSpec(;
        url = "https://github.com/JuliaAstro/Cosmology.jl",
        rev = "units",
    ),
    Pkg.PackageSpec(; url = "https://github.com/JuliaPhysics/DynamicQuantities.jl"),
    Pkg.PackageSpec(; url = "https://github.com/JuliaAstro/FITSFiles.jl"),
]

function bootstrap_environment()
    dbg("Activating project environment in: $(SCRIPT_DIR)")
    Pkg.activate(SCRIPT_DIR)

    marker = joinpath(SCRIPT_DIR, ".sdss_env_ready")
    if isfile(marker)
        dbg("Environment marker found ($(marker)); skipping package installation.")
        dbg("(Delete that marker file to force a reinstall.)")
    else
        dbg("First run: installing $(length(PACKAGE_SPECS)) packages. This may take several minutes...")
        try
            Pkg.add(PACKAGE_SPECS)
            dbg("Pkg.add finished.")
            write(marker, string(now()))
        catch err
            dbg("ERROR while installing packages: $(sprint(showerror, err))")
            dbg("Check the branch/rev names in PACKAGE_SPECS; upstream branches may have changed.")
            rethrow()
        end
    end

    dbg("Instantiating / precompiling environment...")
    Pkg.instantiate()
    dbg("Environment ready.")
    return nothing
end

bootstrap_environment()

# =============================================================================
# The module
# =============================================================================
module SDSSSpectroscopy

using Dates: now, format
using Downloads: download

# Analysis
using DustExtinction, Cosmology, SkyCoords, FITSFiles, SpectrumBase

# Units and uncertainties
using DynamicQuantities, Measurements

# Plotting
using CairoMakie
using MathTeXEngine: set_texfont_family!, FontFamily

export main

# -----------------------------------------------------------------------------
# Paths: everything is read from / written to the folder holding this file
# -----------------------------------------------------------------------------
const OUTDIR = @__DIR__
const SDSS_FILE = joinpath(OUTDIR, "sdss_example.fits")

# The tutorial notebook actually ships this FITS file pre-bundled in its repo
# (the `dr14.sdss.org/.../view/data/...` URL in its source is commented-out
# dead code) -- that "view" web service returns 404/403 for direct downloads.
# The real, stable way to fetch this exact spectrum (plate 1323, MJD 52797,
# fiber 12) is SDSS's static bulk-data mirror (the "SAS"), which uses the
# path pattern: sas/<DR>/sdss/spectro/redux/26/spectra/<plate>/spec-<plate>-<mjd>-<fiber, 4 digits>.fits
# We try a few data-release mirrors in case one is temporarily down.
const SDSS_URLS = [
    "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/1323/spec-1323-52797-0012.fits",
    "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/1323/spec-1323-52797-0012.fits",
    "https://data.sdss.org/sas/dr14/sdss/spectro/redux/26/spectra/1323/spec-1323-52797-0012.fits",
    "https://data.sdss.org/sas/dr12/sdss/spectro/redux/26/spectra/1323/spec-1323-52797-0012.fits",
]

# Fallback dust-map coordinates (degrees), only used if the FITS header's own
# RA/DEC can't be read. These are the tutorial's original placeholder values.
const RA_DEG = 178.90417
const DEC_DEG = 0.66278

# -----------------------------------------------------------------------------
# Debug helpers
# -----------------------------------------------------------------------------
const T0 = Ref(time())

function dbg(msg::AbstractString)
    elapsed = round(time() - T0[]; digits = 1)
    println("[DEBUG $(format(now(), "HH:MM:SS")) | +$(elapsed)s] $msg")
    flush(stdout)
    return nothing
end

"""
    step(f, name)

Run `f()` with start/finish debug messages and timing. If it throws, the error
is logged and `nothing` is returned so later sections can be skipped cleanly.
"""
function step(f, name::AbstractString)
    dbg(">>> START: $name")
    t = time()
    try
        result = f()
        dbg("<<< DONE:  $name ($(round(time() - t; digits = 2)) s)")
        return result
    catch err
        dbg("!!! FAILED: $name -> $(sprint(showerror, err))")
        for (i, frame) in enumerate(stacktrace(catch_backtrace())[1:min(8, end)])
            println("      [$i] $frame")
        end
        flush(stdout)
        return nothing
    end
end

function save_png(fig, filename::AbstractString)
    path = joinpath(OUTDIR, filename)
    dbg("Saving plot -> $path")
    save(path, fig)
    dbg("Saved ($(round(filesize(path) / 1024; digits = 1)) KiB)")
    return path
end

# -----------------------------------------------------------------------------
# 1. Loading a real SDSS spectrum
# -----------------------------------------------------------------------------
"""
    looks_like_fits(path)

A real FITS file's first 6 bytes are always the literal ASCII "SIMPLE" (the
start of the mandatory SIMPLE = T header card). Some servers return HTTP 200
with an HTML error/redirect page instead of a 404, which `download` would
otherwise treat as success -- this catches that case before it gets fed to
FITSFiles.jl.
"""
function looks_like_fits(path)
    isfile(path) || return false
    filesize(path) < 2880 && return false # smaller than one FITS header block
    header = open(path) do io
        read(io, 6)
    end
    return String(header) == "SIMPLE"
end

function ensure_data()
    if isfile(SDSS_FILE) && looks_like_fits(SDSS_FILE)
        dbg("FITS file already present: $SDSS_FILE ($(filesize(SDSS_FILE)) bytes)")
        return SDSS_FILE
    elseif isfile(SDSS_FILE)
        dbg("Existing file at $SDSS_FILE doesn't look like a FITS file; re-downloading.")
        rm(SDSS_FILE; force = true)
    end

    for (i, url) in enumerate(SDSS_URLS)
        dbg("Downloading SDSS spectrum (source $i of $(length(SDSS_URLS))): $url")
        try
            download(url, SDSS_FILE)
            if looks_like_fits(SDSS_FILE)
                dbg("Download complete and looks like a valid FITS file: $(filesize(SDSS_FILE)) bytes")
                return SDSS_FILE
            else
                dbg("Download from source $i returned something that isn't a FITS file (likely an HTML error page); trying next source.")
                isfile(SDSS_FILE) && rm(SDSS_FILE; force = true)
            end
        catch err
            dbg("Download from source $i failed: $(sprint(showerror, err))")
            isfile(SDSS_FILE) && rm(SDSS_FILE; force = true)
        end
    end
    error("Could not download the SDSS spectrum from any of $(length(SDSS_URLS)) sources. Place a valid FITS file at $SDSS_FILE manually (e.g. download " *
          "spec-1323-52797-0012.fits for plate 1323 / MJD 52797 / fiber 12 from https://www.sdss.org and rename it).")
end

function load_hdus(path)
    dbg("Opening FITS file with FITSFiles.jl: $path")
    hdus = fits(path)
    dbg("FITS opened. Number of HDUs: $(length(hdus))")
    for (i, h) in enumerate(hdus)
        dbg("  HDU $i: $(typeof(h))")
    end
    return hdus
end

function build_spectrum(hdus)
    dbg("Reading COADD table (HDU 2): loglam, flux, ivar")
    hdu = hdus[2].data
    loglam = hdu["loglam"]          # log10 wavelength
    ivar = hdu["ivar"]              # inverse variance, σ_flux = 1/√ivar
    flux_err = inv.(sqrt.(ivar))
    dbg("  n pixels = $(length(loglam))")
    dbg("  wavelength range = $(round(10^first(loglam); digits = 1)) – $(round(10^last(loglam); digits = 1)) Å")
    dbg("  pixels with ivar == 0 (masked): $(count(iszero, ivar))")

    # NOTE: SDSS stores flux in units of 1e-17 erg/s/cm^2/Å. As in the original
    # tutorial the numbers are attached to erg/s/cm^2/Å directly, which is why
    # the plot label reads "Flux density * 1e-17".
    spec = spectrum(
        exp10.(loglam) * u"Å",
        (hdu["flux"] .± flux_err) * u"erg/s/cm^2/Å",
    )
    dbg("Spectrum object built: $(typeof(spec))")
    return spec
end

function plot_spectrum(spec)
    wav, flux = spec.spectral_axis, spec.flux_axis

    dbg("Creating band + line plot (zoom around Hα)")
    fig, ax, p = band(
        wav,
        flux;
        axis = (;
            xlabel = "Wavelength",
            ylabel = "Flux density * 1e-17",
            title = "SDSS Galaxy — plate 1323, fiber 12",
            dim1_conversion = Makie.DQConversion(us"Å"),
            dim2_conversion = Makie.DQConversion(us"erg/Å/cm^2/s"),
        ),
        color = :orange,
    )
    lines!(ax, wav, flux)
    xlims!(ax, 6450u"Å", 6800u"Å")

    return save_png(fig, "01_sdss_spectrum_halpha.png")
end

# -----------------------------------------------------------------------------
# 2. Physical unit conversions
# -----------------------------------------------------------------------------
function unit_conversions()
    λ_Hα = 6563.0us"Å"
    dbg("Hα rest wavelength: $λ_Hα")
    dbg("  in nm : $(uconvert(us"nm", λ_Hα))")
    dbg("  in μm : $(uconvert(us"μm", λ_Hα))")
    E = uconvert(us"erg", (u"Constants.c * Constants.h" / λ_Hα))
    dbg("  photon energy (hc/λ) : $E")
    return nothing
end

# -----------------------------------------------------------------------------
# 3. Dust extinction and dereddening
# -----------------------------------------------------------------------------
function deredden_spectrum(spec, hdus)
    # The original tutorial hardcodes a placeholder RA/Dec that its own authors
    # note doesn't match this FITS file's header -- we use the header's actual
    # pointing instead, since that's the real sky position of this spectrum
    # and therefore the correct one to look up dust extinction for.
    ra_deg, dec_deg = try
        ra = Float64(hdus[1].cards["RA"])
        dec = Float64(hdus[1].cards["DEC"])
        dbg("Using FITS header RA/DEC (ICRS, degrees): RA=$(ra)°, Dec=$(dec)°")
        ra, dec
    catch err
        dbg("Could not read RA/DEC from primary header ($(sprint(showerror, err))); " *
            "falling back to the tutorial's placeholder coordinates.")
        dbg("Tutorial placeholder coordinates (ICRS): RA=$(RA_DEG)°, Dec=$(DEC_DEG)°")
        RA_DEG, DEC_DEG
    end

    eq = ICRSCoords(ra_deg * u"°", dec_deg * u"°")
    gal = convert(GalCoords, eq)
    dbg("Galactic coordinates: $gal")

    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    dbg("Loading SFD98 dust map (first use downloads it via DataDeps; may take a while)...")
    dustmap = SFD98Map()
    dbg("Dust map loaded.")

    ebv = dustmap(gal.l, gal.b)
    dbg("E(B-V) = $ebv")
    Av = 3.1 * ebv
    dbg("A_V = 3.1 * E(B-V) = $Av")

    spec_dered = SpectrumBase.deredden(spec, Av)
    dbg("Dereddened spectrum computed.")

    fig, ax, p = lines(
        spec.spectral_axis, spec.flux_axis;
        axis = (;
            xlabel = "Wavelength",
            ylabel = "Flux density",
            title = "CCM89 Dust Dereddening",
            dim1_conversion = Makie.DQConversion(us"Å"),
            dim2_conversion = Makie.DQConversion(us"erg/Å/cm^2/s"),
        ),
        label = "original",
    )
    lines!(ax, spec_dered.spectral_axis, spec_dered.flux_axis; label = "dereddened")
    axislegend(ax)

    return save_png(fig, "02_dust_dereddening.png")
end

# -----------------------------------------------------------------------------
# 4. Blackbody spectra and stellar luminosity density
#    L_λ(λ, T) = 4π² R² B(λ, T)
# -----------------------------------------------------------------------------
L(λ, T; R = u"Constants.R_sun") = 4 * π^2 * R^2 * blackbody(λ, T).flux_axis

function blackbody_section()
    wav_bb = range(3_000, 20_000; length = 500)u"Å"
    dbg("Wavelength grid: $(length(wav_bb)) points, $(first(wav_bb)) – $(last(wav_bb))")

    spec_solar = blackbody(wav_bb, 5778.0u"K")
    B_solar = spec_solar.flux_axis
    L_solar = L(wav_bb, 5778.0u"K")

    dbg("Solar (5778 K) peak radiance : $(uconvert(us"erg/s/cm^2/sr/Å", maximum(B_solar)))")
    dbg("Solar (5778 K) peak L_λ      : $(uconvert(us"erg/s/Å", maximum(L_solar)))")

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        yscale = log10,
        yticks = LogTicks(LinearTicks(5)),
        xlabel = "Wavelength",
        ylabel = "Flux density",
        title = "Blackbody Spectra",
        dim1_conversion = Makie.DQConversion(us"Å"),
        dim2_conversion = Makie.DQConversion(us"erg/Å/cm^2/s"),
    )

    temps_bb = [30_000, 10_000, 5778.0, 3_000]u"K"
    colors_bb = [:purple, :steelblue, :orange, :red]
    for (T, color) in zip(temps_bb, colors_bb)
        dbg("  plotting blackbody T = $T")
        B = blackbody(wav_bb, T).flux_axis
        lines!(ax, wav_bb, B; color, label = string(T))
    end
    axislegend(ax)

    path = save_png(fig, "03_blackbody_spectra.png")
    return wav_bb, path
end

# -----------------------------------------------------------------------------
# 5. Cosmological redshift and surface-brightness dimming
#    F_obs(λ_obs, z) = L_λ(λ_rest) / ((1 + z) 4π d_L²)
# -----------------------------------------------------------------------------
function F_obs(λ_obs, z, cosmo; T = 5778.0u"K")
    λ_rest = λ_obs * inv(1 + z)
    d_L = luminosity_dist(cosmo, z)
    return L(λ_rest, T) / ((1 + z) * 4 * π * d_L^2)
end

function redshift_section(wav_bb)
    cosmo = cosmology()
    dbg("Cosmology object created: $(typeof(cosmo))")

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        yticks = LinearTicks(5),
        xlabel = "Wavelength (observed)",
        ylabel = "Flux density",
        title = "Cosmological Surface-Brightness Dimming",
        dim1_conversion = Makie.DQConversion(us"Å"),
        dim2_conversion = Makie.DQConversion(us"erg/Å/cm^2/s"),
    )

    redshifts = [0.4 => :black, 0.5 => :blue, 1.0 => :green, 2.0 => :red]
    for (z, color) in redshifts
        dbg("  computing observed flux for z = $z")
        wav_obs = wav_bb * (1 + z)
        lines!(ax, wav_obs, F_obs(wav_obs, z, cosmo); label = "z = $(z)", color)
    end

    # The tutorial leaves the legend commented out; try it but don't fail on it.
    try
        axislegend(ax; position = :rt)
    catch err
        dbg("Legend skipped: $(sprint(showerror, err))")
    end

    return save_png(fig, "04_redshift_dimming.png")
end

# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
function main()
    T0[] = time()
    dbg("=== SDSSSpectroscopy: start ===")
    dbg("Julia $(VERSION); output/input folder: $OUTDIR")
    cd(OUTDIR)

    step("Set TeX font family") do
        set_texfont_family!(FontFamily("TeXGyreHeros"))
    end

    # --- 1. Load spectrum ----------------------------------------------------
    path = step(ensure_data, "1a. Get SDSS FITS file")
    hdus = path === nothing ? nothing : step(() -> load_hdus(path), "1b. Load FITS HDUs")
    spec = hdus === nothing ? nothing : step(() -> build_spectrum(hdus), "1c. Build spectrum object")
    if spec !== nothing
        step(() -> plot_spectrum(spec), "1d. Plot spectrum (Hα zoom)")
    else
        dbg("Skipping spectrum plot (no spectrum).")
    end

    # --- 2. Unit conversions -------------------------------------------------
    step(unit_conversions, "2. Unit conversions")

    # --- 3. Dust extinction --------------------------------------------------
    if spec !== nothing
        step(() -> deredden_spectrum(spec, hdus), "3. Dust dereddening")
    else
        dbg("Skipping dereddening (no spectrum).")
    end

    # --- 4. Blackbodies ------------------------------------------------------
    bb = step(blackbody_section, "4. Blackbody spectra")

    # --- 5. Redshift dimming -------------------------------------------------
    if bb !== nothing
        wav_bb, _ = bb
        step(() -> redshift_section(wav_bb), "5. Cosmological redshift dimming")
    else
        dbg("Skipping redshift section (blackbody grid unavailable).")
    end

    pngs = sort(filter(f -> endswith(f, ".png"), readdir(OUTDIR)))
    dbg("PNG files in $OUTDIR: $(join(pngs, ", "))")
    dbg("=== SDSSSpectroscopy: finished ===")
    return nothing
end

end # module SDSSSpectroscopy

# ----------------------------------------------------------------------------
# Run automatically when the file is executed (VSCode "Run file" / `julia file`)
# ----------------------------------------------------------------------------
dbg("Module loaded. Calling SDSSSpectroscopy.main() ...")
SDSSSpectroscopy.main()
