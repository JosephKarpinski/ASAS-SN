"""
    JWSTScaleBar

Download a JWST release image (default: the Carina nebula, NIRCam F187N),
display it with world coordinates, and overlay an angular scale bar.

Based on the JuliaAstro tutorial "JWST image with scale bar".

Dependencies: AstroImages, Plots (plus the stdlibs Downloads and LinearAlgebra).

    pkg> add AstroImages Plots

# Running it

Executing the file (`julia JWSTScaleBar.jl`, `include`, or VS Code's "Execute
active file in REPL") loads the module and, because `AUTORUN_JWSTSCALEBAR` at
the bottom of the file is `true`, runs the whole pipeline and writes
`carina_scalebar.png`. Set it to `false` to only load the module and call the
functions yourself:

```julia
include("JWSTScaleBar.jl")
using .JWSTScaleBar

run_demo()                       # full pipeline with debug output
# or step by step:
fname = download_carina()        # saved in pwd(); reused on later runs
full  = load_image(fname)
print_header_key(full, "CTYPE1") # inspect a WCS header key (comment + value)
quicklook(full)                  # quick imview preview, no WCS axes
small = downsample(full; step = 10)
p     = plot_with_scalebar(small, full; arcmin = 1)
savefig(p, "carina_scalebar.png")
```

Debug messages are on by default; silence them with `set_debug!(false)`.
"""
module JWSTScaleBar

using Downloads
using LinearAlgebra
using AstroImages
using Plots

export CARINA_URL,
       CARINA_FILENAME,
       download_carina,
       load_image,
       downsample,
       angular_separation,
       arcmin_in_pixels,
       header_value,
       header_comment,
       print_header_key,
       quicklook,
       plot_with_world_coords,
       plot_with_scalebar,
       run_demo,
       set_debug!

# ---------------------------------------------------------------------------
# Debug output
# ---------------------------------------------------------------------------

const DEBUG = Ref(true)
const T0 = Ref(time())

"""
    set_debug!(on::Bool = true)

Turn the step-by-step debug messages on or off.
"""
function set_debug!(on::Bool = true)
    DEBUG[] = on
    return on
end

# Print "[DEBUG +12.3s] Module.function: message" and flush immediately so
# progress is visible even during long-running steps.
function _dbg(where_::AbstractString, msg...)
    DEBUG[] || return nothing
    t = round(time() - T0[]; digits = 1)
    println("[DEBUG +", t, "s] JWSTScaleBar.", where_, ": ", msg...)
    flush(stdout)
    return nothing
end

# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

"""URL of the JWST Carina nebula NIRCam F187N image on MAST (over 3 GB)."""
const CARINA_URL =
    "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/jw02731-o001_t017_nircam_clear-f187n_i2d.fits"

"""Default local file name for the Carina image."""
const CARINA_FILENAME = "jw02731-o001_t017_nircam_clear-f187n_i2d.fits"

# Build a `progress(total, now)` callback for `Downloads.download` that prints
# percent, GB downloaded, and speed at most once every `every` seconds.
# `total` is 0 when the server does not report a size; then only GB is shown.
function _progress_printer(; every::Real = 3.0)
    t_start = time()
    t_last  = Ref(t_start - every)     # so the first update prints promptly
    return function (total::Integer, now::Integer)
        t = time()
        finished = total > 0 && now >= total
        (t - t_last[] >= every || finished) || return nothing
        t_last[] = t
        gb    = round(now / 1e9; digits = 2)
        speed = round(now / 1e6 / max(t - t_start, 1e-3); digits = 1)
        if total > 0
            pct     = round(100 * now / total; digits = 1)
            tot_gb  = round(total / 1e9; digits = 2)
            eta_s   = speed > 0 ? round(Int, (total - now) / 1e6 / speed) : 0
            _dbg("download_carina", "progress: ", pct, "% (", gb, " / ", tot_gb,
                 " GB) at ", speed, " MB/s, ETA ~", eta_s, " s")
        else
            _dbg("download_carina", "progress: ", gb, " GB at ", speed, " MB/s")
        end
        return nothing
    end
end

"""
    download_carina(; url = CARINA_URL, dir = pwd(), filename = CARINA_FILENAME,
                      force = false) -> String

Return the local path of the JWST image, downloading it into `dir` (the
current directory by default) only if it is not already there.

* An existing non-empty file is reused; the download is skipped.
* `force = true` re-downloads even if the file exists.
* The download goes to a temporary `.part` file first and is renamed on
  success, so an interrupted download is never mistaken for a complete file.

!!! warning
    The image is over 3 GB. Make sure you have enough disk space.
"""
function download_carina(; url::AbstractString = CARINA_URL,
                          dir::AbstractString = pwd(),
                          filename::AbstractString = CARINA_FILENAME,
                          force::Bool = false,
                          progress_every::Real = 120.0)
    _dbg("download_carina", "start (dir=", dir, ", force=", force, ")")
    mkpath(dir)
    path = joinpath(dir, filename)
    _dbg("download_carina", "target path = ", path)

    if !force && isfile(path) && filesize(path) > 0
        sz = round(filesize(path) / 1e9; digits = 2)
        _dbg("download_carina", "file already present (", sz, " GB), skipping download")
        return path
    end

    part = path * ".part"
    if isfile(part)
        _dbg("download_carina", "removing stale partial download ", part)
        rm(part)
    end
    _dbg("download_carina", "downloading from MAST (>3 GB, this may take a while)...")
    try
        Downloads.download(url, part; progress = _progress_printer(; every = progress_every))
        _dbg("download_carina", "download finished, renaming ", basename(part), " -> ", basename(path))
        mv(part, path; force = true)
    catch
        _dbg("download_carina", "download FAILED, cleaning up partial file")
        isfile(part) && rm(part)
        rethrow()
    end
    _dbg("download_carina", "done")
    return path
end

"""
    load_image(fname) -> AstroImage

Load a FITS image with AstroImages.jl (the first image HDU is used).
"""
function load_image(fname::AbstractString)
    _dbg("load_image", "loading ", fname, " (large files can take a minute)...")
    img = AstroImages.load(fname)
    _dbg("load_image", "loaded image with size ", size(img))
    return img
end

"""
    downsample(img; step = 10)

Keep every `step`-th pixel along both axes. The result retains the original
pixel coordinates, so overlays can still be specified in full-resolution
pixel units.
"""
function downsample(img; step::Integer = 10)
    _dbg("downsample", "keeping every ", step, "th pixel of image size ", size(img))
    out = img[begin:step:end, begin:step:end]
    _dbg("downsample", "result size ", size(out))
    return out
end

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

"""
    angular_separation(a, b) -> Float64

Great-circle separation in degrees between two sky positions `a = (ra, dec)`
and `b = (ra, dec)`, given in degrees, using the spherical law of cosines:

    cos γ = cos(90°-δa) cos(90°-δb) + sin(90°-δa) sin(90°-δb) cos(αa-αb)
"""
function angular_separation(a, b)
    αa, δa = deg2rad.(a)
    αb, δb = deg2rad.(b)
    pa, pb = π / 2 - δa, π / 2 - δb
    c = cos(pa) * cos(pb) + sin(pa) * sin(pb) * cos(αa - αb)
    γ = rad2deg(acos(clamp(c, -1.0, 1.0)))
    _dbg("angular_separation", "a=", a, " b=", b, " -> ", γ, " deg")
    return γ
end

"""
    arcmin_in_pixels(img_full; at = [1000, 100], arcmin = 1.0) -> Float64

Length in pixels of an angular span of `arcmin` arcminutes, measured by
stepping in declination from pixel position `at` of the full-resolution image.

The WCS is warped, so the value depends slightly on where in the image the
measurement is made.
"""
function arcmin_in_pixels(img_full; at = [1000, 100], arcmin::Real = 1.0)
    _dbg("arcmin_in_pixels", "measuring ", arcmin, " arcmin starting at pixel ", at)
    a_px  = collect(Float64, at)
    a_deg = pix_to_world(img_full, a_px)
    _dbg("arcmin_in_pixels", "start in world coords (deg) = ", a_deg)
    b_deg = a_deg .+ [0, arcmin / 60]          # 1 arcmin = 1/60 degree
    b_px  = world_to_pix(img_full, b_deg)
    _dbg("arcmin_in_pixels", "end in pixel coords = ", b_px)
    len = norm(b_px .- a_px)
    _dbg("arcmin_in_pixels", arcmin, " arcmin = ", len, " pixels")
    return len
end

# ---------------------------------------------------------------------------
# Header inspection
# ---------------------------------------------------------------------------

"""
    header_value(img, key)

Return the FITS header value for `key` (e.g. `"CTYPE1"`).
"""
function header_value(img, key::AbstractString)
    v = img[key]
    _dbg("header_value", key, " = ", v)
    return v
end

"""
    header_comment(img, key)

Return the FITS header comment for `key` (e.g. `"CTYPE1"` → `"Axis 1 type"`).
"""
function header_comment(img, key::AbstractString)
    c = img[key, Comment]
    _dbg("header_comment", key, " comment = ", c)
    return c
end

"""
    print_header_key(img, key)

Print a header key's comment and value, mirroring the tutorial's
`println(carina["CTYPE1", Comment]); println(carina["CTYPE1"])`.
"""
function print_header_key(img, key::AbstractString)
    println(header_comment(img, key))
    println(header_value(img, key))
    return nothing
end

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

"""
    quicklook(img; clims = Percent(98), kwargs...)

Simple preview with [`AstroImages.imview`](@extref) — no world-coordinate
axes, just the image with adjusted display limits. This is the tutorial's
first display step, useful for a fast look before adding WCS axes or a
scale bar.
"""
function quicklook(img; clims = Percent(98), kwargs...)
    _dbg("quicklook", "previewing image with imview (clims=", clims, ")")
    return imview(img; clims = clims, kwargs...)
end

"""
    plot_with_world_coords(img; clims = Percent(98), grid = false, kwargs...)

Plot `img` with RA/Dec axes and a colourbar (MJy/sr for JWST images).
"""
function plot_with_world_coords(img; clims = Percent(98), grid = false, kwargs...)
    _dbg("plot_with_world_coords", "plotting image with WCS axes")
    return implot(img; clims = clims, grid = grid, kwargs...)
end

"""
    plot_with_scalebar(img, img_full;
                       arcmin = 1.0,
                       bar_origin = (11000, 1000),
                       measure_at = [1000, 100],
                       label = nothing,
                       clean = false,
                       clims = Percent(98),
                       color = :white,
                       linewidth = 5,
                       fontsize = 18,
                       size = (1600, 1000),
                       dpi = 200)

Plot the (downsampled) image `img` and overlay a horizontal scale bar spanning
`arcmin` arcminutes.

* `img_full`    – full-resolution image, used for the WCS pixel↔world conversion.
* `bar_origin`  – left end of the bar, in full-resolution pixel coordinates.
* `measure_at`  – pixel position where the arcminute length is measured.
* `label`       – annotation text; defaults to `" 1' "`-style text.
* `clean`       – if `true`, hide ticks, colourbar, frame, and labels so only
                  the scale (not the location) is communicated.
* `size`        – figure size in pixels, `(width, height)`. This is what
                  controls how large the saved PNG is — bump it for a bigger
                  image (e.g. `(2400, 1500)`).
* `dpi`         – resolution used when saving to a raster format like PNG.

Returns the `Plots.Plot`; save it with `savefig(p, "output.png")`.
"""
function plot_with_scalebar(img, img_full;
                            arcmin::Real = 1.0,
                            bar_origin = (11000, 1000),
                            measure_at = [1000, 100],
                            label::Union{Nothing,AbstractString} = nothing,
                            clean::Bool = false,
                            clims = Percent(98),
                            color = :white,
                            linewidth::Real = 5,
                            fontsize::Integer = 18,
                            size = (1600, 1000),
                            dpi::Integer = 200)

    _dbg("plot_with_scalebar", "start (arcmin=", arcmin, ", clean=", clean, ", size=", size, ", dpi=", dpi, ")")
    len_px = arcmin_in_pixels(img_full; at = measure_at, arcmin = arcmin)
    x0, y0 = bar_origin

    _dbg("plot_with_scalebar", "drawing base image...")
    if clean
        p = implot(
            img[10:end-10, 10:end-10];   # crop slightly
            grid = false,
            ticks = false,
            colorbar = false,
            clims = clims,
            xlabel = "",
            ylabel = "",
            framestyle = :none,
            background_outside = :transparent,
            size = size,
            dpi = dpi,
        )
    else
        p = implot(img; grid = false, clims = clims, size = size, dpi = dpi)
    end

    _dbg("plot_with_scalebar", "overplotting bar from x=", x0, " to x=", x0 + len_px, " at y=", y0)
    plot!(p, [x0, x0 + len_px], [y0, y0];
          color = color, linewidth = linewidth, label = "")

    txt = label === nothing ? " $(arcmin isa Integer ? arcmin : round(arcmin; digits = 2))' " : label
    _dbg("plot_with_scalebar", "adding annotation ", repr(txt))
    annotate!(p, (x0 + len_px / 2, y0 + 100,
                  text(txt, fontsize, color, :center, :bottom)))
    _dbg("plot_with_scalebar", "done")
    return p
end

# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------

"""
    run_demo(; dir = pwd(), step = 10, arcmin = 1.0, output = "carina_scalebar.png",
               clean = false, force = false, progress_every = 120.0,
               size = (1600, 1000), dpi = 200) -> Plots.Plot

Run the whole pipeline: download (or reuse) the image, load it, downsample,
plot with a scale bar, and save the figure to `joinpath(dir, output)`.

`progress_every` controls how often (in seconds) the download prints a
progress update; the default is every 2 minutes. `size` and `dpi` control
the dimensions and resolution of the saved PNG — increase them for a larger
image (e.g. `size = (2400, 1500), dpi = 300`).
"""
function run_demo(; dir::AbstractString = pwd(),
                   step::Integer = 10,
                   arcmin::Real = 1.0,
                   output::AbstractString = "carina_scalebar.png",
                   clean::Bool = false,
                   force::Bool = false,
                   progress_every::Real = 120.0,
                   size = (1600, 1000),
                   dpi::Integer = 200)
    T0[] = time()
    _dbg("run_demo", "=== starting pipeline ===")

    _dbg("run_demo", "[1/5] locate or download data")
    fname = download_carina(; dir = dir, force = force, progress_every = progress_every)

    _dbg("run_demo", "[2/5] load FITS image")
    full = load_image(fname)

    _dbg("run_demo", "[3/5] downsample for display")
    small = downsample(full; step = step)

    _dbg("run_demo", "[4/5] plot with scale bar")
    p = plot_with_scalebar(small, full; arcmin = arcmin, clean = clean, size = size, dpi = dpi)

    _dbg("run_demo", "[5/5] save figure")
    outpath = joinpath(dir, output)
    savefig(p, outpath)
    _dbg("run_demo", "saved ", outpath)

    _dbg("run_demo", "=== pipeline complete ===")
    return p
end

end # module

# Auto-run switch. VS Code's "Julia: Execute active file in REPL" (and
# `include`) does not set PROGRAM_FILE, so a script-only guard never fires
# there. Set this to `false` if you want to load the module without running
# the pipeline (e.g. to call the functions yourself).
AUTORUN_JWSTSCALEBAR = true

if AUTORUN_JWSTSCALEBAR || abspath(PROGRAM_FILE) == @__FILE__
    println("[JWSTScaleBar] module loaded, starting run_demo() ...")
    flush(stdout)
    JWSTScaleBar.run_demo()
end
