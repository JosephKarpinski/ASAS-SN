
module KorgApogeeDemo_v1

using FITSIO
using DataFrames
using Korg
using PyPlot

export run_apogee_demo

# ------------------------------------------------------------
# Read APOGEE DR17 FITS (allStarLite-dr17-synspec_rev1.fits)
# ------------------------------------------------------------
function read_apogee_fits(path)
    f = FITS(path)

    best_df = nothing
    max_rows = 0

    for h in f
        # APOGEE DR17 main catalog is a binary table.
        # FITSIO.jl has no `isbinary` function - check the HDU type instead.
        # (TableHDU = binary table, ASCIITableHDU = ASCII table)
        if isa(h, TableHDU)
            # TableHDU implements the Tables.jl interface, so DataFrame(h)
            # works directly - no need to read raw columns/dicts by hand.
            df_tmp = DataFrame(h)
            nrows = nrow(df_tmp)

            if nrows > max_rows
                max_rows = nrows
                best_df = df_tmp
            end
        end
    end

    best_df === nothing && error("No binary table HDU found in APOGEE FITS file")

    close(f)
    return best_df
end





# ------------------------------------------------------------
# Extract APOGEE abundances into a Dict for Korg
# ------------------------------------------------------------
function extract_abundances(row)
    # APOGEE FITS columns are named like NA_H, MG_H, AL_H (all uppercase),
    # but Korg.synth expects proper element symbols as keyword names
    # (e.g. Na, Mg, Al - not NA, MG, AL).
    elems = [
        "C", "N", "O", "Na", "Mg", "Al", "Si", "P", "S", "K",
        "Ca", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ce", "Yb"
    ]

    abund = Dict{Symbol,Float64}()

    for el in elems
        col = Symbol(uppercase(el) * "_H")   # e.g. Symbol("NA_H") matches the FITS column
        if hasproperty(row, col)
            val = getproperty(row, col)
            if !ismissing(val)
                abund[Symbol(el)] = Float64(val)
            end
        end
    end

    return abund
end

# ------------------------------------------------------------
# Synthesize APOGEE star spectrum using Korg
# ------------------------------------------------------------
function synthesize_apogee_star(row, linelist)
    Teff  = Float64(row.TEFF)
    logg  = Float64(row.LOGG)
    M_H   = Float64(row.M_H)
    vmic  = Float64(row.VMICRO)
    vmac  = Float64(row.VMACRO)   # not applied below - see note
    vsini = Float64(row.VSINI)

    abund = extract_abundances(row)

    # NOTE: Korg.synth has no `vmac` keyword - it does not implement
    # macroturbulent broadening (only vsini/rotation and vmic/microturbulence
    # are supported). If you want VMACRO applied, convolve `flux` with a
    # macroturbulence kernel (e.g. Gray's radial-tangential profile, or a
    # simple Gaussian in velocity space) after this call.
    #
    # NOTE: Korg.synth also has no `abundances=` keyword - individual
    # element abundances are passed as separate keywords (e.g. Fe=-0.2),
    # collected internally via a keyword splat. Splat the dict in instead.
    wls, flux, cont = Korg.synth(;
        Teff = Teff,
        logg = logg,
        M_H  = M_H,
        vmic = vmic,
        vsini = vsini,
        wavelengths = (15000.0, 17000.0),
        linelist = linelist,
        abund...
    )

    return wls, flux, cont
end

# ------------------------------------------------------------
# Quick plot helper
# ------------------------------------------------------------
function plot_quick(wls, flux; plot_title="")
    fig = figure(figsize=(10,4))
    plot(wls, flux, color="black", linewidth=0.8)
    xlabel("λ [Å]")
    ylabel("Flux")
    title(plot_title)
    tight_layout()
    return fig
end

# ------------------------------------------------------------
# Main APOGEE demo
# ------------------------------------------------------------
function run_apogee_demo(fits_path)

    println("[run_apogee_demo] reading APOGEE FITS: $fits_path")
    df = read_apogee_fits(fits_path)
    println("[run_apogee_demo] loaded $(nrow(df)) rows")

    println("[run_apogee_demo] loading APOGEE DR17 linelist")
    linelist = Korg.get_APOGEE_DR17_linelist()

    N = min(5, nrow(df))

    for i in 1:N
        println("\n[run_apogee_demo] ----- ROW $i -----")
        row = df[i, :]

        try
            wls, flux, cont = synthesize_apogee_star(row, linelist)
            println("[run_apogee_demo] row $i spectrum length = $(length(wls))")

            # Title lines
            title_line1 = "GaiaDR3=$(row.GAIAEDR3_SOURCE_ID) | APOGEE=$(row.APOGEE_ID)"
            title_line2 = "Teff=$(row.TEFF) | logg=$(row.LOGG) | [M/H]=$(row.M_H)"

            title = title_line1 * "\n" * title_line2

            fig = plot_quick(wls, flux; plot_title=title)

            # Output directory
            outdir = "Korg_Apogee_Star_Spectra"
            isdir(outdir) || mkpath(outdir)

            # Filename
            gaia_id    = row.GAIAEDR3_SOURCE_ID
            apogee_id  = row.APOGEE_ID

            outfile = joinpath(outdir, "gaia_$(gaia_id)_apogee_$(apogee_id)_row$(lpad(i, 5, '0')).png")
            fig.savefig(outfile, dpi=150)

        catch err
            println("[run_apogee_demo] ERROR on row $i: $err")
        end
    end

    println("[run_apogee_demo] DONE")
end

end # module

# Auto-run when executed directly
#
# `Base.invokelatest` is needed here (per Julia 1.12's stricter world-age
# rules) because the module definition above and this call are being
# evaluated together as one top-level chunk (e.g. via an editor's "run
# file" command) rather than as separate top-level statements. Without
# it, `run_apogee_demo` is looked up in a world that predates its own
# definition.
println("[KorgApogeeDemo_v1] Running demo because file was executed directly.")
Base.invokelatest(
    KorgApogeeDemo_v1.run_apogee_demo,
    "/Volumes/SSD2/MyJulia/Apogee/allStarLite-dr17-synspec_rev1.fits",
)

