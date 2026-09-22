# =============================================================================
# FITSImages.jl
#
# Standalone Julia module based on the JuliaAstro tutorial
#   "FITS images"
#   https://learn.juliaastro.org/tutorials/fits-images/
#   (adapted from https://learn.astropy.org/tutorials/FITS-images.html;
#    original authors: Lia Corrales, Kris Stern, Stephanie T. Douglas,
#    Kelle Cruz, Lúthien Liu, Zihao Chen, Saima Siddiqui)
#
# Sections (same as the tutorial):
#   1. Open a FITS image (Horsehead Nebula) and inspect it
#   2. Custom `imview`-style rendering (stretch/colormap/contrast/bias)
#   3. Basic statistics (extrema, mean, median, std)
#   4. Pixel-value histogram
#   5. `implotview` (labeled Makie plot with colorbar)
#   6. Log-scaled Makie plot with a custom colorbar
#   7. Image stacking: download & sum 5 exposures of M13
#   8. Save the stacked image back out as a FITS file
#
# HOW TO RUN (VSCode): open this file and press the "Run file" (▷) tab.
#   * The working directory is set to the folder containing this file.
#   * All files are read/written there: downloaded FITS files, the
#     Project.toml/Manifest.toml, every PNG plot, and the stacked FITS output.
#   * First run creates a Julia environment in this folder and installs the
#     packages (this can take several minutes). Later runs skip that step.
#
# NOTE: like the tutorial, this uses development branches of several
# JuliaAstro/Makie packages. If those branches have since been merged/renamed,
# edit `PACKAGE_SPECS` below.
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
        rev = "makie",
        url = "https://github.com/JuliaAstro/AstroImages.jl",
    ),
    Pkg.PackageSpec(; url = "https://github.com/JuliaAstro/AstroAngles.jl"),
    Pkg.PackageSpec(; url = "https://github.com/JuliaAstro/FITSFiles.jl"),
    Pkg.PackageSpec(; url = "https://github.com/JuliaAstro/FITSWCS.jl"),
    # DimensionalData is handled separately below (see `develop_patched_dimensionaldata`)
    # because its "makie-0.25" branch still declares `Makie = "0.20 - 0.24"` in its own
    # Project.toml -- the branch name promises 0.25 support but the compat bound was
    # never bumped, which makes a plain `Pkg.add` of it unresolvable against the
    # Makie 0.25 dev branches above (ERROR: Unsatisfiable requirements... restricted
    # to versions 0.20 - 0.24 by DimensionalData). We clone it ourselves and patch
    # just that one compat line before registering it as a dev dependency.
]

const DIMENSIONALDATA_GIT_URL = "https://github.com/icweaver/DimensionalData.jl"
const DIMENSIONALDATA_BRANCH = "makie-0.25"
const DIMENSIONALDATA_DEV_PATH = joinpath(SCRIPT_DIR, ".dev_pkgs", "DimensionalData")

"""
    compat_makie_span(proj_text)

Find the `Makie = "..."` line, but ONLY inside the `[compat]` table -- the
same Project.toml also has a `[weakdeps]` table with a line `Makie = "<uuid>"`
(a package UUID, not a version bound), which comes first in the file and must
not be touched. Returns `(range_of_whole_line, captured_value)`, or `nothing`
if the `[compat]` table or its `Makie` entry can't be found.
"""
function compat_makie_span(proj_text::AbstractString)
    compat_hdr = findfirst("[compat]", proj_text)
    compat_hdr === nothing && return nothing
    section_start = last(compat_hdr) + 1

    next_hdr = findnext(r"\n\[", proj_text, section_start)
    section_end = next_hdr === nothing ? lastindex(proj_text) : first(next_hdr)

    section = proj_text[section_start:section_end]
    m = match(r"(?m)^Makie\s*=\s*\"([^\"]*)\"", section)
    m === nothing && return nothing

    # Re-anchor the match's offsets back onto the *full* string.
    whole_start = section_start + first(m.offset) - 1
    whole_range = whole_start:(whole_start + ncodeunits(m.match) - 1)
    return whole_range, m.captures[1]
end

"""
    develop_patched_dimensionaldata()

Clone `icweaver/DimensionalData.jl`@`makie-0.25` locally, patch its Project.toml
so the `Makie` compat entry also allows 0.25 (the branch's actual code targets
Makie 0.25 -- only the declared compat bound is stale), then register it with
`Pkg.develop` so the resolver picks up the corrected bound from disk.
"""
function develop_patched_dimensionaldata()
    # Always start from a pristine clone rather than trying to detect/repair a
    # possibly-corrupted one left over from an earlier failed attempt (a stray
    # `.dev_pkgs/DimensionalData` from before this patch step existed could have
    # a broken [weakdeps] OR [compat] entry, or both) -- a shallow clone is cheap,
    # and this function only runs once per environment anyway (guarded by the
    # `.fits_images_env_ready` marker in `bootstrap_environment`).
    if isdir(DIMENSIONALDATA_DEV_PATH)
        dbg("Removing existing dev clone at $DIMENSIONALDATA_DEV_PATH before re-cloning fresh (avoids building on a possibly-corrupted leftover from an earlier failed run).")
        rm(DIMENSIONALDATA_DEV_PATH; force = true, recursive = true)
    end

    mkpath(dirname(DIMENSIONALDATA_DEV_PATH))
    dbg("Cloning $DIMENSIONALDATA_GIT_URL @ $DIMENSIONALDATA_BRANCH -> $DIMENSIONALDATA_DEV_PATH")
    try
        run(`git clone --depth 1 -b $DIMENSIONALDATA_BRANCH $DIMENSIONALDATA_GIT_URL $DIMENSIONALDATA_DEV_PATH`)
    catch err
        dbg("`git clone` failed or `git` isn't on PATH ($(sprint(showerror, err))). Falling back to a plain " *
            "Pkg.add of DimensionalData@$DIMENSIONALDATA_BRANCH -- this will likely hit the same " *
            "unresolvable-Makie-compat error as before, since it skips the patch step.")
        Pkg.add(Pkg.PackageSpec(; rev = DIMENSIONALDATA_BRANCH, url = DIMENSIONALDATA_GIT_URL))
        return nothing
    end

    proj_path = joinpath(DIMENSIONALDATA_DEV_PATH, "Project.toml")
    proj_text = read(proj_path, String)
    span = compat_makie_span(proj_text)
    if span === nothing
        dbg("WARNING: could not find a `Makie = \"...\"` line inside [compat] in $proj_path; leaving it unpatched (Pkg.develop may still fail).")
    else
        line_range, current_bound = span
        if occursin("0.25", current_bound)
            dbg("DimensionalData's [compat] Makie bound already allows 0.25 ($current_bound); no patch needed.")
        else
            new_line = "Makie = \"$(current_bound), 0.25\""
            dbg("Patching $proj_path [compat]: `$(proj_text[line_range])` -> `$new_line`")
            patched = proj_text[1:prevind(proj_text, first(line_range))] * new_line * proj_text[nextind(proj_text, last(line_range)):end]
            write(proj_path, patched)
        end
    end

    dbg("Registering patched DimensionalData as a dev dependency (Pkg.develop)")
    Pkg.develop(Pkg.PackageSpec(; path = DIMENSIONALDATA_DEV_PATH))
    return nothing
end

function bootstrap_environment()
    dbg("Activating project environment in: $(SCRIPT_DIR)")
    Pkg.activate(SCRIPT_DIR)

    marker = joinpath(SCRIPT_DIR, ".fits_images_env_ready")
    if isfile(marker)
        dbg("Environment marker found ($(marker)); skipping package installation.")
        dbg("(Delete that marker file to force a reinstall.)")
    else
        dbg("First run: installing $(length(PACKAGE_SPECS) + 1) packages. This may take several minutes...")
        try
            develop_patched_dimensionaldata()
            Pkg.add(PACKAGE_SPECS)
            dbg("Pkg.add finished.")
            write(marker, string(now()))
        catch err
            dbg("ERROR while installing packages: $(sprint(showerror, err))")
            dbg("Check the branch/rev names in PACKAGE_SPECS (and DIMENSIONALDATA_* above); upstream branches may have changed.")
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
module FITSImages

using Dates: now, format
using Downloads: download
using Printf: @sprintf
using Statistics: mean, median, std

using AstroImages
using CairoMakie

export main

# -----------------------------------------------------------------------------
# Paths: everything is read from / written to the folder holding this file
# -----------------------------------------------------------------------------
const OUTDIR = @__DIR__

# The tutorial's data lives on data.astropy.org over plain http; we try https
# first (works on most mirrors) and fall back to http if that's refused.
const HORSEHEAD_URLS = [
    "https://data.astropy.org/tutorials/FITS-images/HorseHead.fits",
    "http://data.astropy.org/tutorials/FITS-images/HorseHead.fits",
]
const HORSEHEAD_FILE = joinpath(OUTDIR, "HorseHead.fits")

function m13_urls(i::Integer)
    name = @sprintf("M13_blue_%04d.fits", i)
    return [
        "https://data.astropy.org/tutorials/FITS-images/$name",
        "http://data.astropy.org/tutorials/FITS-images/$name",
    ]
end
m13_file(i::Integer) = joinpath(OUTDIR, @sprintf("M13_blue_%04d.fits", i))

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

"""
    looks_like_fits(path)

A real FITS file's first 6 bytes are always the literal ASCII "SIMPLE" (the
start of the mandatory SIMPLE = T header card). Some servers return HTTP 200
with an HTML error/redirect page instead of a 404, which `download` would
otherwise treat as success -- this catches that case before it gets fed to
the FITS reader.
"""
function looks_like_fits(path)
    isfile(path) || return false
    filesize(path) < 2880 && return false # smaller than one FITS header block
    header_bytes = open(path) do io
        read(io, 6)
    end
    return String(header_bytes) == "SIMPLE"
end

"""
    ensure_fits(urls, dest; desc)

Download `dest` from the first working URL in `urls`, validating that the
result actually looks like a FITS file. Skips the download entirely if a
valid file is already sitting at `dest`.
"""
function ensure_fits(urls, dest::AbstractString; desc::AbstractString = basename(dest))
    if isfile(dest) && looks_like_fits(dest)
        dbg("$desc already present: $dest ($(filesize(dest)) bytes)")
        return dest
    elseif isfile(dest)
        dbg("Existing file at $dest doesn't look like a FITS file; re-downloading.")
        rm(dest; force = true)
    end

    for (i, url) in enumerate(urls)
        dbg("Downloading $desc (source $i of $(length(urls))): $url")
        try
            download(url, dest)
            if looks_like_fits(dest)
                dbg("Download complete and looks like a valid FITS file: $(filesize(dest)) bytes")
                return dest
            else
                dbg("Download from source $i returned something that isn't a FITS file (likely an HTML error page); trying next source.")
                isfile(dest) && rm(dest; force = true)
            end
        catch err
            dbg("Download from source $i failed: $(sprint(showerror, err))")
            isfile(dest) && rm(dest; force = true)
        end
    end
    error("Could not download $desc from any of $(length(urls)) sources. Place a valid FITS file at $dest manually.")
end

# -----------------------------------------------------------------------------
# 1. Opening the Horsehead Nebula FITS image
# -----------------------------------------------------------------------------
function load_horsehead()
    path = ensure_fits(HORSEHEAD_URLS, HORSEHEAD_FILE; desc = "Horsehead Nebula image")
    dbg("Opening image with AstroImages.jl: $path")
    img = AstroImage(path; scale = false)
    dbg("Image size: $(size(img)); eltype: $(eltype(img))")

    hdr_str = try
        s = string(header(img))
        lines = split(s, '\n')
        first_lines = join(lines[1:min(12, length(lines))], '\n')
        length(lines) > 12 ? first_lines * "\n  ... ($(length(lines) - 12) more header lines)" : first_lines
    catch err
        "<could not stringify header: $(sprint(showerror, err))>"
    end
    dbg("Header (first lines):\n$hdr_str")

    return img
end

# -----------------------------------------------------------------------------
# 2. Custom imview-style rendering (stretch/colormap/contrast/bias)
# -----------------------------------------------------------------------------
"""
    figure_of(result)

`implotview` (and Makie recipes generally) can return either a bare `Figure`
or a tuple whose first element is the `Figure` (commonly `(fig, ax)` or
`(fig, ax, plot)`, depending on the recipe/version). Rather than assume a
fixed arity -- which broke here because `implotview` returns a 2-element
result, not the 3-element `(fig, ax, plot)` some other Makie recipes use --
just take whatever the first element is when the result is a `Tuple`, and
the result itself otherwise.
"""
figure_of(result) = result isa Tuple ? first(result) : result

function plot_custom_view(img)
    dbg("Rendering with custom imview options: clims=Percent(99.5), stretch=identity, cmap=:magma")
    result = implotview(
        img;
        clims = Percent(99.5),
        stretch = identity,
        cmap = :magma,
        contrast = 1.0,
        bias = 0.5,
    )
    fig = figure_of(result)
    return save_png(fig, "01_horsehead_custom_view.png")
end

# -----------------------------------------------------------------------------
# 3. Basic statistics
# -----------------------------------------------------------------------------
function log_stats(img)
    lo, hi = extrema(img)
    dbg("extrema = ($lo, $hi)")
    dbg("mean    = $(mean(img))")
    dbg("median  = $(median(img))")
    dbg("std     = $(std(img))")
    return nothing
end

# -----------------------------------------------------------------------------
# 4. Pixel-value histogram
# -----------------------------------------------------------------------------
function plot_histogram(img, filename)
    dbg("Building pixel-value histogram (50 bins)")
    fig, ax, p = stephist(
        vec(img);
        bins = 50,
        axis = (; xlabel = "Pixel value", ylabel = "Counts"),
    )
    return save_png(fig, filename)
end

# -----------------------------------------------------------------------------
# 5. implotview: labeled Makie plot with colorbar (default options)
# -----------------------------------------------------------------------------
function plot_implotview(img, filename; kwargs...)
    dbg("Rendering implotview (default look) -> $filename")
    result = implotview(img; kwargs...)
    fig = figure_of(result)
    return save_png(fig, filename)
end

# -----------------------------------------------------------------------------
# 6. Log-scaled Makie plot with a custom colorbar
# -----------------------------------------------------------------------------
function plot_log_colorbar(img)
    dbg("Rendering log10-scaled plot with custom colorbar ticks")
    fig, ax, p = plot(
        img;
        colorscale = log10,
        colormap = :greys,
        colorbar = (
            ticks = [4.0e3, 5.0e3, 6.0e4, 1.0e4, 2.0e4],
            minorticksvisible = true,
            minorticks = IntervalsBetween(9),
        ),
    )
    return save_png(fig, "04_horsehead_log_colorbar.png")
end

# -----------------------------------------------------------------------------
# 7. Image stacking: download & sum 5 exposures of M13
# -----------------------------------------------------------------------------
function load_and_stack_m13()
    imgs = map(1:5) do i
        path = ensure_fits(m13_urls(i), m13_file(i); desc = "M13 exposure $i/5")
        dbg("Loading M13 exposure $i/5: $path")
        load(path)
    end
    dbg("Stacking $(length(imgs)) exposures with sum(...)")
    img_stacked = sum(imgs)
    dbg("Stacked image size: $(size(img_stacked))")
    return img_stacked
end

# -----------------------------------------------------------------------------
# 8. Save the stacked image back out as a FITS file
# -----------------------------------------------------------------------------
function save_stacked_fits(img_stacked)
    path = joinpath(OUTDIR, "m13_stacked.fits")
    dbg("Saving stacked image + header data -> $path")
    save(path, img_stacked)
    dbg("Saved ($(round(filesize(path) / 1024; digits = 1)) KiB)")
    return path
end

# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
function main()
    T0[] = time()
    dbg("=== FITSImages: start ===")
    dbg("Julia $(VERSION); output/input folder: $OUTDIR")
    cd(OUTDIR)

    # --- 1. Load the Horsehead Nebula image ---------------------------------
    img = step(load_horsehead, "1. Load Horsehead Nebula FITS image")

    if img !== nothing
        # --- 2. Custom imview-style rendering --------------------------------
        step(() -> plot_custom_view(img), "2. Custom imview-style rendering")

        # --- 3. Basic statistics ----------------------------------------------
        step(() -> log_stats(img), "3. Basic statistics")

        # --- 4. Pixel-value histogram ------------------------------------------
        step(() -> plot_histogram(img, "02_horsehead_histogram.png"), "4. Pixel-value histogram")

        # --- 5. implotview (default look) --------------------------------------
        step(() -> plot_implotview(img, "03_horsehead_implotview.png"), "5. implotview (default look)")

        # --- 6. Log-scaled plot with custom colorbar ----------------------------
        step(() -> plot_log_colorbar(img), "6. Log-scaled plot with custom colorbar")
    else
        dbg("Skipping steps 2-6 (no Horsehead image available).")
    end

    # --- 7. Image stacking: download & sum 5 M13 exposures --------------------
    img_stacked = step(load_and_stack_m13, "7. Download & stack 5 M13 exposures")

    if img_stacked !== nothing
        step(() -> plot_histogram(img_stacked, "05_m13_stack_histogram.png"), "7b. Stacked-image histogram")
        step(
            () -> plot_implotview(img_stacked, "06_m13_stack_implotview.png"; clims = (2.0e3, 3.0e3)),
            "7c. Stacked-image implotview (clims = (2e3, 3e3))",
        )

        # --- 8. Save the stacked image back out as a FITS file ------------------
        step(() -> save_stacked_fits(img_stacked), "8. Save stacked image as FITS")
    else
        dbg("Skipping steps 7b-8 (no stacked M13 image available).")
    end

    pngs = sort(filter(f -> endswith(f, ".png"), readdir(OUTDIR)))
    dbg("PNG files in $OUTDIR: $(join(pngs, ", "))")
    dbg("=== FITSImages: finished ===")
    return nothing
end

end # module FITSImages

# ----------------------------------------------------------------------------
# Run automatically when the file is executed (VSCode "Run file" / `julia file`)
# ----------------------------------------------------------------------------
dbg("Module loaded. Calling FITSImages.main() ...")
FITSImages.main()
