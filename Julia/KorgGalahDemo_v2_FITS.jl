
module KorgGalahDemo_v2_FITS

using CSV, DataFrames
using Korg
using PythonPlot

export run_galah_demo


using FITSIO
using DataFrames

function read_galah_fits(path)
    f = FITS(path)
    t = f[2]  # GALAH DR4 table is in HDU 2
    df = DataFrame(t)
    close(f)
    return df
end



# ------------------------------------------------------------------
# MARCS grid ranges
# ------------------------------------------------------------------

const MARCS_GRID_RANGES = (
    Teff=(2500.0, 8000.0),
    logg=(-0.5, 5.5),
    M_H=(-2.5, 1.0),
)

function _check_marcs_range(Teff, logg, M_H)
    println("[_check_marcs_range] Teff=$Teff logg=$logg M_H=$M_H")
    return nothing
end

# ------------------------------------------------------------------
# SPECTRUM SYNTHESIS
# ------------------------------------------------------------------

function synthesize_star(Teff, logg, M_H;
    abundances=Dict{String,Float64}(),
    wavelengths=(5850.0, 5900.0),
    linelist=nothing,
    kwargs...
)

    println("[synthesize_star] START Teff=$Teff logg=$logg M_H=$M_H")
    println("[synthesize_star] abundances = $abundances")
    println("[synthesize_star] extra kwargs = $(kwargs)")

    linelist = linelist === nothing ? Korg.get_GALAH_DR3_linelist() : linelist

    abundance_kwargs = (; (Symbol(el) => val for (el,val) in abundances)...)

    println("[synthesize_star] calling Korg.synth with forwarded kwargs")

    wls, flux, cont = Korg.synth(; 
        Teff=Teff,
        logg=logg,
        M_H=M_H,
        linelist=linelist,
        wavelengths=wavelengths,
        abundance_kwargs...,
        kwargs...      # <-- REQUIRED FIX
    )

    println("[synthesize_star] DONE, spectrum length = $(length(wls))")

    return collect(wls), collect(flux), collect(cont)
end


# ------------------------------------------------------------------
# GALAH ROW PARSING (updated for your CSV)
# ------------------------------------------------------------------

function star_from_galah_row(row)
    println("[star_from_galah_row] START")

    Teff = row.teff
    logg = row.logg
    M_H  = row.fe_h

    println("[star_from_galah_row] Teff=$Teff logg=$logg M_H=$M_H")

    abundances = Dict{String,Float64}()

    # Use columns that actually exist in your file
    for el in ["Fe", "Mg", "Ba"]
        col = Symbol(lowercase(el)*"_fe")
        if hasproperty(row, col) && !ismissing(getproperty(row,col))
            abundances[el] = getproperty(row,col) + M_H
            println("[star_from_galah_row] [$(el)/H] = $(abundances[el])")
        else
            println("[star_from_galah_row] missing $col, skipping $el")
        end
    end

    println("[star_from_galah_row] DONE abundances = $abundances")

    return Teff, logg, M_H, abundances
end

# ------------------------------------------------------------------
# GALAH STAR SYNTHESIS
# ------------------------------------------------------------------

function synthesize_galah_star(row; wavelengths=(5850.0,5900.0))
    println("[synthesize_galah_star] START")

    Teff, logg, M_H, abundances = star_from_galah_row(row)

    vmic  = hasproperty(row,:vmic)  && !ismissing(row.vmic)  ? row.vmic  : nothing
    vsini = hasproperty(row,:vsini) && !ismissing(row.vsini) ? row.vsini : nothing

    println("[synthesize_galah_star] vmic=$vmic vsini=$vsini")

    extra = Dict{Symbol,Any}()
    if vmic !== nothing;  extra[:vmic] = vmic;  end
    if vsini !== nothing; extra[:vsini] = vsini; end

    wls, flux, cont = synthesize_star(Teff, logg, M_H;
                                      abundances=abundances,
                                      wavelengths=wavelengths,
                                      extra...)

    println("[synthesize_galah_star] DONE")

    return wls, flux, cont
end

# ------------------------------------------------------------------
# DEMO DRIVER
# ------------------------------------------------------------------

function run_galah_demo(fits_path)

    # --- Load GALAH DR4 FITS -------------------------------------------------
    println("[run_galah_demo] reading FITS: $fits_path")
    df = read_galah_fits(fits_path)
    println("[run_galah_demo] loaded $(nrow(df)) rows")

    N = min(5, nrow(df))

    for i in 1:N
        println("\n[run_galah_demo] ----- ROW $i -----")
        row = df[i, :]

        try
            wls, flux, cont = synthesize_galah_star(row)
            println("[run_galah_demo] row $i spectrum length = $(length(wls))")

            # --- Title lines --------------------------------------------------
            ra_str  = round(row.ra,  digits=4)
            dec_str = round(row.dec, digits=4)

            title_line1 = "GaiaDR3=$(row.gaiadr3_source_id) | Survey=$(row.survey_name) | " *
                          "RA=$(ra_str) | Dec=$(dec_str)"

            title_line2 = "Teff=$(row.teff) | logg=$(row.logg) | [Fe/H]=$(row.fe_h)"

            title = title_line1 * "\n" * title_line2

            fig = plot_quick(wls, flux; title=title)

            # --- Output directory --------------------------------------------
            outdir = "Korg_Galah_Star_Spectra"
            isdir(outdir) || mkpath(outdir)

            # --- Filename: gaia_<id>_sobject_<id>.png ------------------------
            gaia_id    = row.gaiadr3_source_id
            sobject_id = row.sobject_id

            outfile = joinpath(outdir, "gaia_$(gaia_id)_sobject_$(sobject_id).png")
            fig.savefig(outfile, dpi=150)

        catch err
            println("[run_galah_demo] ERROR on row $i: $err")
        end
    end

    println("[run_galah_demo] DONE")
end


  

end # module

# -------------------------------------------------------------------
# Auto-run when the file is executed (including VS Code "Run File")
# -------------------------------------------------------------------
println("[KorgGalahDemo_v2_FITS] Running demo because file was executed directly.")
KorgGalahDemo_v2_FITS.run_galah_demo("/Volumes/SSD2/MyJulia/Galah/galah_dr4_allstar_240705.fits")



