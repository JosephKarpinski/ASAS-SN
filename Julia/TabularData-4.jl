"""
    TabularData

Load the Hipparcos-GAIA Catalog of Accelerations (HGCA, Brandt et al. 2021)
from a FITS table into a `DataFrame`, filter it to nearby stars, and plot
their sky positions (plain and Aitoff-projected), colouring by the
significance of their anomalous acceleration.

Based on the JuliaAstro tutorial "Working with tabular data".

Dependencies: FITSIO, DataFrames, Plots, AstroLib.

    pkg> add FITSIO DataFrames Plots AstroLib

Optional, for the interactive plot: GLMakie. It is *not* a hard dependency —
`plot_aitoff_interactive` loads it lazily on first use, so the rest of the
module works fine without it. Install it yourself first if you want it:

    pkg> add GLMakie

# Getting the data

Unlike the JWST tutorial, the HGCA catalog has no fixed, scriptable download
URL — the paper's page only offers a manual link for "Table 4". So this
module does not try to auto-download it. Instead:

  1. Open the paper: https://iopscience.iop.org/article/10.3847/1538-4365/abf93c
  2. Find Table 4 and download its FITS data file.
  3. Uncompress it and save it as `HGCA_vEDR3.fits` in your working directory
     (or pass `dir=`/`filename=` to point elsewhere).

Run once with the file missing and you'll get a `MissingCatalogError` with
these same instructions and the exact path expected. If you do have a
direct URL (e.g. a mirror you trust), pass `url=` and it will be downloaded
and cached exactly like the JWST module does.

# Running it

Executing the file (`julia TabularData.jl`, `include`, or VS Code's "Execute
active file in REPL") loads the module and, because `AUTORUN_TABULARDATA` at
the bottom of the file is `true`, runs the whole static pipeline and writes
`hgca_aitoff.png`. Right after that, because `AUTORUN_INTERACTIVE` is also
`true`, it opens the interactive GLMakie window too (see the note above about
it sometimes opening behind your editor — check Mission Control / Cmd+Tab).
Set either switch to `false` to skip that step; `AUTORUN_TABULARDATA = false`
skips both and just loads the module so you can call the functions yourself:

```julia
include("TabularData.jl")
using .TabularData

run_demo()                              # full pipeline with debug output
# or step by step:
path    = locate_catalog()              # errors with instructions if missing
df      = load_dataframe(path)
summary = describe_table(df)
nearby  = filter_nearby(df; min_parallax = 50.0)
p       = plot_aitoff(nearby)
savefig(p, "hgca_aitoff.png")

# interactive (loads GLMakie on first call; needs a display):
fig = plot_aitoff_interactive(nearby)   # or: run_interactive()
```

Debug messages are on by default; silence them with `set_debug!(false)`.
"""
module TabularData

using Downloads
using FITSIO
using DataFrames
using Plots
using AstroLib

export HGCA_INFO_URL,
       CATALOG_FILENAME,
       MissingCatalogError,
       locate_catalog,
       load_dataframe,
       describe_table,
       filter_nearby,
       aitoff_projection,
       plot_sky,
       plot_aitoff,
       plot_aitoff_interactive,
       run_demo,
       run_interactive,
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
    println("[DEBUG +", t, "s] TabularData.", where_, ": ", msg...)
    flush(stdout)
    return nothing
end

# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

"""Paper page describing the catalog and linking to Table 4 (manual download)."""
const HGCA_INFO_URL = "https://iopscience.iop.org/article/10.3847/1538-4365/abf93c"

"""Default local file name for the HGCA catalog."""
const CATALOG_FILENAME = "HGCA_vEDR3.fits"

"""
    MissingCatalogError(path)

Thrown by [`locate_catalog`](@ref) when the catalog isn't present locally
and no `url` was given to download it from.
"""
struct MissingCatalogError <: Exception
    path::String
end

function Base.showerror(io::IO, e::MissingCatalogError)
    print(io,
        "MissingCatalogError: catalog not found at\n    ", e.path, "\n\n",
        "The HGCA catalog has no stable direct-download URL, so it must be ",
        "fetched by hand:\n",
        "  1. Open ", HGCA_INFO_URL, "\n",
        "  2. Find Table 4 and download its FITS data file\n",
        "  3. Uncompress it and save it as\n         ", e.path, "\n",
        "  4. Re-run this script (or call run_demo() again)\n\n",
        "Alternatively, if you already know a working direct URL, pass it:\n",
        "    run_demo(url = \"https://.../HGCA_vEDR3.fits\")",
    )
end

# Build a `progress(total, now)` callback for `Downloads.download` that prints
# percent, GB downloaded, and speed at most once every `every` seconds.
# `total` is 0 when the server does not report a size; then only GB is shown.
function _progress_printer(; every::Real = 30.0)
    t_start = time()
    t_last  = Ref(t_start - every)     # so the first update prints promptly
    return function (total::Integer, now::Integer)
        t = time()
        finished = total > 0 && now >= total
        (t - t_last[] >= every || finished) || return nothing
        t_last[] = t
        mb    = round(now / 1e6; digits = 2)
        speed = round(now / 1e6 / max(t - t_start, 1e-3); digits = 1)
        if total > 0
            pct    = round(100 * now / total; digits = 1)
            tot_mb = round(total / 1e6; digits = 2)
            _dbg("locate_catalog", "progress: ", pct, "% (", mb, " / ", tot_mb, " MB) at ", speed, " MB/s")
        else
            _dbg("locate_catalog", "progress: ", mb, " MB at ", speed, " MB/s")
        end
        return nothing
    end
end

"""
    locate_catalog(; dir = pwd(), filename = CATALOG_FILENAME, url = nothing,
                     force = false, progress_every = 30.0) -> String

Return the local path of the HGCA catalog.

* An existing non-empty file is reused; nothing is downloaded.
* If the file is missing and `url` is given, it is downloaded into `dir`
  (via a `.part` file, renamed on success, exactly like the JWST module).
* If the file is missing and no `url` is given, throws
  [`MissingCatalogError`](@ref) with instructions for the manual download —
  see the module docstring.
* `force = true` re-downloads (requires `url`) even if the file exists.
"""
function locate_catalog(; dir::AbstractString = pwd(),
                         filename::AbstractString = CATALOG_FILENAME,
                         url::Union{Nothing,AbstractString} = nothing,
                         force::Bool = false,
                         progress_every::Real = 30.0)
    _dbg("locate_catalog", "start (dir=", dir, ", force=", force, ")")
    mkpath(dir)
    path = joinpath(dir, filename)
    _dbg("locate_catalog", "target path = ", path)

    if !force && isfile(path) && filesize(path) > 0
        sz = round(filesize(path) / 1e6; digits = 1)
        _dbg("locate_catalog", "file already present (", sz, " MB), skipping download")
        return path
    end

    if url === nothing
        _dbg("locate_catalog", "not found locally and no url given")
        throw(MissingCatalogError(path))
    end

    part = path * ".part"
    isfile(part) && rm(part)
    _dbg("locate_catalog", "downloading from ", url, " ...")
    try
        Downloads.download(url, part; progress = _progress_printer(; every = progress_every))
        _dbg("locate_catalog", "download finished, renaming ", basename(part), " -> ", basename(path))
        mv(part, path; force = true)
    catch
        _dbg("locate_catalog", "download FAILED, cleaning up partial file")
        isfile(part) && rm(part)
        rethrow()
    end
    _dbg("locate_catalog", "done")
    return path
end

# ---------------------------------------------------------------------------
# Loading and examining the table
# ---------------------------------------------------------------------------

"""
    load_dataframe(fname; hdu = 2) -> DataFrame

Open the FITS file at `fname`, read the table in HDU `hdu` (2 by default,
matching the HGCA layout: HDU 1 is an empty image, HDU 2 is the table), and
return it as a `DataFrame`. The file is closed before returning.
"""
function load_dataframe(fname::AbstractString; hdu::Integer = 2)
    _dbg("load_dataframe", "opening ", fname, " (large files can take a moment)...")
    df = FITS(fname) do f
        _dbg("load_dataframe", "file has ", length(f), " HDU(s); reading HDU ", hdu)
        table_fits = f[hdu]
        DataFrame(table_fits)
    end
    _dbg("load_dataframe", "loaded DataFrame with ", nrow(df), " rows and ", ncol(df), " columns")
    return df
end

"""
    describe_table(df) -> DataFrame

Print and return `describe(df)`: variable, mean, min, median, max, and
missing count for every column.
"""
function describe_table(df::DataFrame)
    _dbg("describe_table", "summarizing ", ncol(df), " columns over ", nrow(df), " rows")
    summary = describe(df)
    show(summary; allrows = true, allcols = true)
    println()
    return summary
end

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

"""
    filter_nearby(df; column = :parallax_gaia, min_parallax = 50.0) -> DataFrame

Return the rows of `df` where `column` exceeds `min_parallax` (milliarcseconds
of parallax by default — i.e. nearby stars).
"""
function filter_nearby(df::DataFrame; column::Symbol = :parallax_gaia, min_parallax::Real = 50.0)
    _dbg("filter_nearby", "filtering ", column, " > ", min_parallax, " on ", nrow(df), " rows")
    nearby = filter(column => >(min_parallax), df)
    _dbg("filter_nearby", "kept ", nrow(nearby), " of ", nrow(df), " rows")
    return nearby
end

# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

"""
    aitoff_projection(df; ra_col = :gaia_ra, dec_col = :gaia_dec) -> (x, y)

Apply [`AstroLib.aitoff`](@extref) to the given RA/Dec columns (degrees) and
return separate `x` and `y` vectors ready for plotting.
"""
function aitoff_projection(df::DataFrame; ra_col::Symbol = :gaia_ra, dec_col::Symbol = :gaia_dec)
    _dbg("aitoff_projection", "projecting ", nrow(df), " points (", ra_col, ", ", dec_col, ")")
    newpoints = aitoff.(df[:, ra_col], df[:, dec_col])
    x = getindex.(newpoints, 1)
    y = getindex.(newpoints, 2)
    _dbg("aitoff_projection", "done")
    return x, y
end

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

"""
    plot_sky(df; ra_col = :gaia_ra, dec_col = :gaia_dec, color_col = :chisq,
             size = (1600, 1000), dpi = 200, kwargs...)

Plain RA/Dec scatter plot (no map projection), coloured by `log10(color_col)`.
This is the tutorial's first, un-projected plot.
"""
function plot_sky(df::DataFrame; ra_col::Symbol = :gaia_ra, dec_col::Symbol = :gaia_dec,
                   color_col::Symbol = :chisq, size = (1600, 1000), dpi::Integer = 200, kwargs...)
    _dbg("plot_sky", "plotting ", nrow(df), " points, size=", size, ", dpi=", dpi)
    return scatter(
        df[:, ra_col], df[:, dec_col];
        marker_z = log10.(df[:, color_col]),
        colorbartitle = "log10 χ²",
        label = "",
        xlabel = "right ascension (°)",
        ylabel = "declination (°)",
        size = size,
        dpi = dpi,
        kwargs...,
    )
end

"""
    plot_aitoff(df; ra_col = :gaia_ra, dec_col = :gaia_dec, color_col = :chisq,
                color = :turbo, size = (1600, 1000), dpi = 200, kwargs...)

Aitoff-projected sky scatter plot, coloured by `log10(color_col)`, styled to
match the tutorial's final plot (transparent background, gray frame, no
marker outline).

Returns the `Plots.Plot`; save it with `savefig(p, "output.png")`.
"""
function plot_aitoff(df::DataFrame; ra_col::Symbol = :gaia_ra, dec_col::Symbol = :gaia_dec,
                      color_col::Symbol = :chisq, color = :turbo,
                      size = (1600, 1000), dpi::Integer = 200, kwargs...)
    _dbg("plot_aitoff", "start (n=", nrow(df), ", size=", size, ", dpi=", dpi, ")")
    x, y = aitoff_projection(df; ra_col = ra_col, dec_col = dec_col)
    _dbg("plot_aitoff", "drawing scatter plot...")
    p = scatter(
        x, y;
        marker_z = log10.(df[:, color_col]),
        color = color,
        colorbartitle = "log10 χ²",
        label = "",
        xlabel = "right ascension (°)",
        ylabel = "declination (°)",
        background = :transparent,
        foreground = :gray,
        framestyle = :box,
        markerstrokewidth = 0,
        grid = :none,
        size = size,
        dpi = dpi,
        kwargs...,
    )
    _dbg("plot_aitoff", "done")
    return p
end

# ---------------------------------------------------------------------------
# Interactive plotting (optional — GLMakie, loaded lazily)
# ---------------------------------------------------------------------------

# Keep the most recent interactive figure/screen alive. GLMakie windows are
# tied to their Screen object; if nothing keeps a reference to it (e.g. you
# call `run_interactive()` at the REPL without assigning the result), it can
# become eligible for GC and the window can be torn down. Pinning it here
# avoids that regardless of what the caller does with the return value.
const _LAST_FIGURE = Ref{Any}(nothing)
const _LAST_SCREEN = Ref{Any}(nothing)

# `using GLMakie` inside this module the first time it's actually needed,
# rather than as a top-level dependency of the module. This keeps GLMakie
# (a heavy package that needs a display / OpenGL) entirely optional: nothing
# else here — including `run_demo` and AUTORUN_TABULARDATA — requires it.
function _ensure_glmakie!()
    if !isdefined(@__MODULE__, :GLMakie)
        _dbg("plot_aitoff_interactive", "loading GLMakie (first use — may take a moment to precompile)...")
        Base.eval(@__MODULE__, :(using GLMakie))
        _dbg("plot_aitoff_interactive", "GLMakie loaded")
    end
    return nothing
end

"""
    plot_aitoff_interactive(df; ra_col = :gaia_ra, dec_col = :gaia_dec,
                             color_col = :chisq, colormap = :turbo,
                             markersize = 10, size = (1200, 800))

Interactive Aitoff-projected sky scatter plot using GLMakie: scroll to zoom,
drag to pan, hover a point to see its HIP / Gaia source ID and log10(χ²).

GLMakie is loaded on first call rather than being a hard dependency of this
module — install it yourself first with `pkg> add GLMakie` if you haven't.
This needs an actual display; it won't work in a headless session.

Returns the `Makie.Figure` (also opened in an interactive window).
"""
function plot_aitoff_interactive(df::DataFrame; ra_col::Symbol = :gaia_ra, dec_col::Symbol = :gaia_dec,
                                  color_col::Symbol = :chisq, colormap = :turbo,
                                  markersize::Real = 10, size = (1200, 800))
    _ensure_glmakie!()
    return Base.invokelatest(_plot_aitoff_interactive_impl, df;
                              ra_col = ra_col, dec_col = dec_col, color_col = color_col,
                              colormap = colormap, markersize = markersize, size = size)
end

# Split out so `plot_aitoff_interactive` can dispatch to it via
# `Base.invokelatest` — GLMakie's methods were just defined at runtime by
# `_ensure_glmakie!`, and calling straight into them from a function whose
# world age predates that `using` would otherwise raise a world-age error.
function _plot_aitoff_interactive_impl(df::DataFrame; ra_col::Symbol, dec_col::Symbol, color_col::Symbol,
                                        colormap, markersize::Real, size)
    _dbg("plot_aitoff_interactive", "start (n=", nrow(df), ", size=", size, ")")

    x, y = aitoff_projection(df; ra_col = ra_col, dec_col = dec_col)
    colorvals = log10.(df[:, color_col])

    has_hip  = "hip_id" in names(df)
    has_gaia = "gaia_source_id" in names(df)

    GLMakie.activate!()
    fig = GLMakie.Figure(; size = size)
    ax  = GLMakie.Axis(fig[1, 1]; xlabel = "right ascension (°)", ylabel = "declination (°)")

    label_fn = (self, i, p) -> begin
        lines = String[]
        has_hip  && push!(lines, "HIP $(df.hip_id[i])")
        has_gaia && push!(lines, "Gaia $(df.gaia_source_id[i])")
        push!(lines, "log10 χ² = $(round(colorvals[i]; digits = 2))")
        join(lines, "\n")
    end

    sc = GLMakie.scatter!(ax, x, y;
        color = colorvals,
        colormap = colormap,
        markersize = markersize,
        inspector_label = label_fn,
    )
    GLMakie.Colorbar(fig[1, 2], sc; label = "log10 χ²")
    GLMakie.DataInspector(fig)

    _dbg("plot_aitoff_interactive", "displaying figure (scroll to zoom, drag to pan, hover for details)")
    screen = GLMakie.display(fig)
    _LAST_FIGURE[] = fig
    _LAST_SCREEN[] = screen
    _dbg("plot_aitoff_interactive",
         "window opened. If you don't see it: check other desktops/Spaces ",
         "(Mission Control, F3) or Cmd+Tab through open apps — GLMakie ",
         "windows can open behind the current one on macOS.")
    _dbg("plot_aitoff_interactive", "done")
    return fig
end

"""
    run_interactive(; dir = pwd(), filename = CATALOG_FILENAME, url = nothing,
                       force = false, min_parallax = 50.0, kwargs...) -> Makie.Figure

Locate the catalog, load it, filter to nearby stars, and open the
interactive Aitoff plot ([`plot_aitoff_interactive`](@ref)). `kwargs...` are
passed through to it (e.g. `markersize`, `colormap`).

Not run by `AUTORUN_TABULARDATA` — call this yourself when you want the
interactive window; the default auto-run pipeline stays headless-safe.
"""
function run_interactive(; dir::AbstractString = pwd(),
                          filename::AbstractString = CATALOG_FILENAME,
                          url::Union{Nothing,AbstractString} = nothing,
                          force::Bool = false,
                          min_parallax::Real = 50.0,
                          kwargs...)
    _dbg("run_interactive", "=== starting interactive pipeline ===")
    path = locate_catalog(; dir = dir, filename = filename, url = url, force = force)
    df = load_dataframe(path)
    nearby = filter_nearby(df; min_parallax = min_parallax)
    fig = plot_aitoff_interactive(nearby; kwargs...)
    _dbg("run_interactive", "=== interactive pipeline complete ===")
    return fig
end

# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------

"""
    run_demo(; dir = pwd(), filename = CATALOG_FILENAME, url = nothing,
               force = false, min_parallax = 50.0, output = "hgca_aitoff.png",
               size = (1600, 1000), dpi = 200, progress_every = 30.0) -> Plots.Plot

Run the whole pipeline: locate (or download, if `url` is given) the catalog,
load it into a `DataFrame`, print a summary, filter to nearby stars, make the
Aitoff-projected plot, and save it to `joinpath(dir, output)`.

Throws [`MissingCatalogError`](@ref) if the catalog isn't present and no
`url` was given — see the module docstring for how to fetch it manually.
"""
function run_demo(; dir::AbstractString = pwd(),
                   filename::AbstractString = CATALOG_FILENAME,
                   url::Union{Nothing,AbstractString} = nothing,
                   force::Bool = false,
                   min_parallax::Real = 50.0,
                   output::AbstractString = "hgca_aitoff.png",
                   size = (1600, 1000),
                   dpi::Integer = 200,
                   progress_every::Real = 30.0)
    T0[] = time()
    _dbg("run_demo", "=== starting pipeline ===")

    _dbg("run_demo", "[1/6] locate catalog")
    path = locate_catalog(; dir = dir, filename = filename, url = url,
                           force = force, progress_every = progress_every)

    _dbg("run_demo", "[2/6] load table into DataFrame")
    df = load_dataframe(path)

    _dbg("run_demo", "[3/6] summarize table")
    describe_table(df)

    _dbg("run_demo", "[4/6] filter to nearby stars (parallax > ", min_parallax, " mas)")
    nearby = filter_nearby(df; min_parallax = min_parallax)

    _dbg("run_demo", "[5/6] plot Aitoff projection")
    p = plot_aitoff(nearby; size = size, dpi = dpi)

    _dbg("run_demo", "[6/6] save figure")
    outpath = joinpath(dir, output)
    savefig(p, outpath)
    _dbg("run_demo", "saved ", outpath)

    _dbg("run_demo", "=== pipeline complete ===")
    return p
end

end # module

# Auto-run switches. VS Code's "Julia: Execute active file in REPL" (and
# `include`) does not set PROGRAM_FILE, so a script-only guard never fires
# there. Set AUTORUN_TABULARDATA to `false` if you want to load the module
# without running anything (e.g. to call the functions yourself).
AUTORUN_TABULARDATA = true

# Also open the interactive GLMakie window after the static pipeline runs.
# Set to `false` to skip it (e.g. in a headless session, or if you don't
# want GLMakie's first-time precompile on every run).
AUTORUN_INTERACTIVE = true

if AUTORUN_TABULARDATA || abspath(PROGRAM_FILE) == @__FILE__
    println("[TabularData] module loaded, starting run_demo() ...")
    flush(stdout)
    try
        TabularData.run_demo()
    catch err
        if err isa TabularData.MissingCatalogError
            println(sprint(showerror, err))
        else
            rethrow()
        end
    end

    if AUTORUN_INTERACTIVE
        println("[TabularData] starting run_interactive() ...")
        flush(stdout)
        try
            TabularData.run_interactive()
        catch err
            if err isa TabularData.MissingCatalogError
                println(sprint(showerror, err))
            else
                println("[TabularData] interactive plot failed (set AUTORUN_INTERACTIVE = false to skip it):")
                println(sprint(showerror, err))
            end
        end
    end
end
