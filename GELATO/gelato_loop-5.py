#!/usr/bin/env python
"""
Run GELATO on a slice of the galaxies in SDSS_Galaxies_J2000.csv and make the
GELATO example-notebook plots (spectrum + model components, BPT bootstraps).

Selection (see settings below):
  * The class / subClass types present in the CSV are printed first.
  * Rows are optionally filtered on subClass (e.g. 'STARFORMING').
  * ROW_RANGE = (start, stop) then picks a slice of the filtered rows, like
    Python's [start:stop]:  (0, 5), (5, 10), (10, 15), ...

Per galaxy:
  1. Look up plate/mjd/fiberID from the CSV row
  2. Query SDSS SpecPhoto for z, ra, dec (astroquery SQL snippet)
  3. Download the spec-PPPP-MJJJJJ-FFFF.fits file into Spectra/
  4. Run gelato.gelato(...) with NProcess = 1
  5. Plot the data + model components and the BPT diagram, save PNGs

Needs in the working directory: ExampleParameters.json (from the GELATO repo).
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astroquery.sdss import SDSS
from colorama import Fore, Style

import gelato

mpl.rcParams['font.size'] = 20

# ----------------------------------------------------------------- settings
CSV_FILE    = 'SDSS_Galaxies_J2000.csv'
PATH_PARAMS = './ExampleParameters.json'
SPEC_DIR    = 'Spectra'
RESULT_DIR  = 'Results'
PLOT_DIR    = 'Plots'
SHOW_PLOTS  = True       # set False to only save PNGs

# --- which galaxies -----------------------------------------------------------
SUBCLASS    = 'STARFORMING'   # exact subClass to keep, e.g. 'STARBURST', 'AGN BROADLINE',
                              # 'MISSING' (rows with no subClass); None = no filter
ROW_RANGE   = (0, 5)          # slice of the filtered rows: (0,5), (5,10), ...
LIST_TYPES_ONLY = False       # True = just print the class/subClass counts, then stop

# --- BPT plot -----------------------------------------------------------------
MIN_VALID_FRAC  = 0.5         # skip the BPT unless this fraction of the bootstraps have
                              # all four lines (OIII, NII, Halpha, Hbeta) > 0
BPT_ERRORBARS   = 'percentile'  # 'percentile' = median with 16-84% range (log-axis safe)
                                # 'std' = mean-style std bars like the GELATO notebook

for d in (SPEC_DIR, RESULT_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------------ helpers
def get_col(row, *names):
    """Case-insensitive column lookup (plate, mjd, fiberID / fiberid ...)."""
    lower = {c.lower(): c for c in row.index}
    for n in names:
        if n.lower() in lower:
            return row[lower[n.lower()]]
    raise KeyError(f'None of {names} found in CSV columns: {list(row.index)}')


def find_column(df, name):
    """Return the actual column name matching `name` case-insensitively, or None."""
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return None


MISSING_KEYS = ('MISSING', '<MISSING>')   # accepted as SUBCLASS to select blank rows


def missing_mask(series):
    """True where a value is NaN/None or just blank (these are the '<missing>' rows)."""
    return series.isna() | (series.astype(str).str.strip() == '')


def show_types(df):
    """Print how many rows there are of each class and subClass."""
    for name in ('class', 'subClass'):
        col = find_column(df, name)
        if col is None:
            print(f'(no "{name}" column in the CSV)')
            continue
        s = df[col].astype(str).str.strip().where(~missing_mask(df[col]), '<missing>')
        print(f'--- {col} types ---')
        print(s.value_counts().to_string(), '\n')


def select_galaxies(df, subclass, row_range):
    """Filter on subClass (case-insensitive), then take rows [start:stop].

    The original CSV index is kept so galaxy numbers match the file.
    """
    sel = df
    if subclass is not None:
        col = find_column(df, 'subClass')
        if col is None:
            raise KeyError('No subClass column in the CSV; set SUBCLASS = None')
        is_missing = missing_mask(df[col])
        if subclass.strip().upper() in MISSING_KEYS:
            mask = is_missing                      # rows with no subClass
        else:                                      # exact match, so 'STARFORMING'
            mask = (~is_missing) & (               # does not include 'STARFORMING BROADLINE'
                df[col].astype(str).str.strip().str.upper() == subclass.strip().upper())
        sel = df[mask]
        print(f'{len(sel)} of {len(df)} rows have subClass == {subclass!r}')
        if len(sel) == 0:
            raise ValueError(f'No rows with subClass {subclass!r}; '
                             'check the list of types printed above')
    start, stop = row_range
    out = sel.iloc[start:stop]
    print(f'Processing rows {start}:{stop} of the selection '
          f'({len(out)} galaxies)\n')
    return out


def subclass_tag(row):
    """subClass of this row as a filename-safe tag, e.g. 'STARFORMING',
    'AGN_BROADLINE'; 'MISSING' if the value is missing."""
    import re
    col = find_column(row.to_frame().T, 'subClass')
    val = row[col] if col is not None else None
    if val is None or pd.isna(val) or str(val).strip() == '':
        return 'MISSING'
    return re.sub(r'\W+', '_', str(val).strip()).upper()


def plot_basename(name, tag=None):
    """spec-0270-51909-0212 + STARFORMING -> spec-0270-51909-0212_STARFORMING"""
    return f'{name}_{tag}' if tag else name


def query_specphoto(plate, mjd, fiberID):
    """SDSS SpecPhoto lookup -> one-row DataFrame, or None."""
    sql_query = f"""
    SELECT spPhoto.plate, spPhoto.mjd, spPhoto.fiberID,
           spPhoto.z, spPhoto.ra, spPhoto.dec
    FROM SpecPhoto as spPhoto
    WHERE (spPhoto.plate = {plate})
    AND (spPhoto.mjd = {mjd})
    AND (spPhoto.fiberID = {fiberID})
    """
    result = SDSS.query_sql(sql_query)
    if result is None or len(result) == 0:
        return None
    return result.to_pandas()


def download_spectrum(plate, mjd, fiberID):
    """Download the SDSS spectrum and save as spec-PPPP-MJJJJJ-FFFF.fits."""
    path = os.path.join(SPEC_DIR, f'spec-{plate:04d}-{mjd}-{fiberID:04d}.fits')
    if os.path.exists(path):
        return path
    hdus = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiberID)
    if not hdus:
        return None
    hdus[0].writeto(path, overwrite=True)
    return path


def plot_spectrum_and_model(path_spec, summary, name, tag=None):
    spectrum = Table.read(path_spec)
    ivar = spectrum['ivar']
    good = ivar > 0                       # GELATO only uses nonzero weights
    wavl = 10**spectrum['loglam'][good]
    flux = spectrum['flux'][good]
    ivar = ivar[good]
    sig = 3 / np.sqrt(ivar)               # 3-sigma band

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.fill_between(wavl, flux - sig, flux + sig, color='gray')
    ax.step(wavl, flux, where='mid', c='k', lw=0.5, label='Data')
    lam = 10**summary['loglam']
    # Plot whichever model components GELATO actually wrote for this galaxy
    # (e.g. the power-law column can be missing when that component isn't fit).
    components = [('MODEL', 'r', 'Total Model'),
                  ('SSP',   'g', 'SSP Cont.'),
                  ('PL',    'b', 'Power-Law Cont.'),
                  ('LINE',  'y', 'Emission Lines')]
    missing = [c for c, _, _ in components if c not in summary.colnames]
    if missing:
        print(Fore.YELLOW + f'  note: SUMMARY has no {missing} column(s); '
              f'available: {summary.colnames}' + Style.RESET_ALL)
    for col, color, label in components:
        if col in summary.colnames:
            ax.step(lam, summary[col], where='mid', c=color, label=label)
    ax.legend()
    ax.set(xlim=[wavl.min(), wavl.max()], ylim=[0, flux.max()],
           xlabel=r'Obs. Wavelength [$\mathrm{\AA}$]', ylabel=r'$F_\lambda$',
           title=name)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'{plot_basename(name, tag)}_fit.png'), dpi=150)
    return fig


def find_flux(tab, pattern):
    """Return the flux column (as a float array) whose name matches a regex.

    GELATO column names carry the line wavelength (which differs between
    GELATO versions / air vs vacuum), so match on the pattern instead of a
    hard-coded name. Broad and Outflow components never match the patterns
    because they have extra tokens in the name.
    """
    import re
    matches = [c for c in tab.colnames if re.match(pattern, c)]
    if len(matches) != 1:
        raise KeyError(f'Expected one column matching {pattern}, found {matches}')
    return np.asarray(tab[matches[0]], dtype=float)


def plot_bpt(params_tab, name, tag=None, z=None):
    import re
    patterns = {'[OIII]': r'^AGN_\[OIII\]_500\d(\.\d+)?_Flux$',
                '[NII]':  r'^AGN_\[NII\]_658\d(\.\d+)?_Flux$',
                'Halpha': r'^Balmer_HI_656\d(\.\d+)?_Flux$',
                'Hbeta':  r'^Balmer_HI_486\d(\.\d+)?_Flux$'}
    missing = [k for k, p in patterns.items()
               if not any(re.match(p, c) for c in params_tab.colnames)]
    if missing:
        # GELATO leaves a line out of its output when it isn't fitted, e.g. when it is
        # redshifted beyond the end of the spectrum (SDSS spectra stop near 9200 A).
        msg = 'no flux column in the GELATO output for ' + ', '.join(missing)
        if z is not None:
            msg += (f' (z = {z}: Halpha would be at {6564.6 * (1 + z):.0f} A, '
                    f'[NII] at {6585.3 * (1 + z):.0f} A)')
        raise ValueError(msg)
    oiii, nii, ha, hb = (find_flux(params_tab, patterns[k]) for k in patterns)
    lines = {'[OIII]': oiii, '[NII]': nii, 'Halpha': ha, 'Hbeta': hb}
    ok = np.ones(len(oiii), dtype=bool)
    for v in lines.values():
        ok &= np.isfinite(v) & (v > 0)
    n_total, n_valid = len(ok), int(ok.sum())
    if n_valid == 0 or n_valid < MIN_VALID_FRAC * n_total:
        # Usually weak or undetected lines: their fitted flux is 0 in many bootstraps
        counts = ', '.join(f'{k} {int(np.sum(np.isfinite(v) & (v > 0)))}'
                           for k, v in lines.items())
        raise ValueError(f'only {n_valid} of {n_total} bootstraps have all four BPT '
                         f'lines > 0 (need at least {MIN_VALID_FRAC:.0%}); '
                         f'bootstraps with flux > 0: {counts}')
    oiii, nii, ha, hb = oiii[ok], nii[ok], ha[ok], hb[ok]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Kauffmann+03 (dashed) -- star-forming / composite boundary
    x = np.logspace(-1.5, 0.05, 100)
    y = 10**(0.61 / (np.log10(x) - 0.05) + 1.3)
    ax.plot(x, y, color='gray', ls='--', label='Kauffmann+03')

    # Kewley+01 (solid) -- maximum starburst line
    x = np.logspace(-1.5, 0.47, 100)
    y = 10**(0.61 / (np.log10(x) - 0.47) + 1.18)
    ax.plot(x, y, color='gray', ls='-', label='Kewley+01')

    xr, yr = nii / ha, oiii / hb
    xm, ym = np.median(xr), np.median(yr)
    if BPT_ERRORBARS == 'std':                      # as in the GELATO example notebook
        xerr, yerr, err_label = np.std(xr), np.std(yr), 'Average'
    else:                                           # 16th-84th percentile, safe on log axes
        (xlo, xhi), (ylo, yhi) = np.percentile(xr, [16, 84]), np.percentile(yr, [16, 84])
        xerr, yerr = [[xm - xlo], [xhi - xm]], [[ym - ylo], [yhi - ym]]
        err_label = 'Median (16-84%)'
    ax.scatter(xr, yr, color='k', label=f'Bootstraps ({n_valid}/{n_total})',
               edgecolors='none', alpha=0.1)
    ax.errorbar(xm, ym, xerr=xerr, yerr=yerr, color='r', label=err_label)
    ax.legend()
    ax.set(xlim=[1e-1, 1e1], ylim=[1e-1, 2e1],
           xscale='log', yscale='log',
           xlabel=r'[NII]/H$\alpha$', ylabel=r'[OIII]/H$\beta$',
           title=name)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'{plot_basename(name, tag)}_BPT.png'), dpi=150)
    return fig


# --------------------------------------------------------------------- main
SDSS_Galaxies = pd.read_csv(CSV_FILE)
print(f'Loaded {len(SDSS_Galaxies)} galaxies\n')

show_types(SDSS_Galaxies)
if LIST_TYPES_ONLY:
    raise SystemExit

selected = select_galaxies(SDSS_Galaxies, SUBCLASS, ROW_RANGE)

for i, row in selected.iterrows():       # i = row number in the original CSV
    try:
        plate   = int(get_col(row, 'plate'))
        mjd     = int(get_col(row, 'mjd'))
        fiberID = int(get_col(row, 'fiberID', 'fiber'))
        print(Fore.CYAN + f'=== Galaxy {i}: plate {plate}, MJD {mjd}, '
              f'fiber {fiberID} ===' + Style.RESET_ALL)

        # --- SDSS SpecPhoto query (redshift, coordinates) -------------------
        df = query_specphoto(plate, mjd, fiberID)
        if df is None:
            print(Fore.RED + 'No SDSS data returned from query; skipping.'
                  + Style.RESET_ALL)
            continue
        print(Fore.YELLOW + 'SDSS query match:\n\n'
              + df.to_string(index=False) + Style.RESET_ALL)

        plate   = int(df['plate'].iloc[0])
        mjd     = int(df['mjd'].iloc[0])
        fiberID = int(df['fiberID'].iloc[0])
        z       = round(df['z'].iloc[0], 4)
        ra      = df['ra'].iloc[0]
        dec     = df['dec'].iloc[0]
        print(f'\nz = {z}   RA = {ra:.5f}   Dec = {dec:.5f}')

        # --- spectrum -------------------------------------------------------
        path_spec = download_spectrum(plate, mjd, fiberID)
        if path_spec is None:
            print(Fore.RED + 'Could not download spectrum; skipping.'
                  + Style.RESET_ALL)
            continue
        name = os.path.basename(path_spec).replace('.fits', '')
        tag  = subclass_tag(row)          # goes into the plot filenames

        # --- run GELATO (single process so it returns the model) -----------
        params = gelato.ConstructParams.construct(PATH_PARAMS)
        params['NProcess'] = 1
        model = gelato.gelato(params, path_spec, z)

        # --- load results ---------------------------------------------------
        path_res = os.path.join(RESULT_DIR, f'{name}-results.fits')
        with fits.open(path_res) as results:
            summary    = Table(results['SUMMARY'].data)
            params_tab = Table(results['PARAMS'].data)

        # --- plots ----------------------------------------------------------
        # Each plot has its own try so one failing doesn't block the other.
        for plot_fn, data in ((plot_spectrum_and_model, summary),
                              (plot_bpt, params_tab)):
            try:
                if plot_fn is plot_spectrum_and_model:
                    plot_fn(path_spec, data, name, tag)
                else:
                    plot_fn(data, name, tag, z)
            except Exception as e:
                print(Fore.RED + f'  {plot_fn.__name__} skipped for {name}: '
                      f'{type(e).__name__}: {e}' + Style.RESET_ALL)
        if SHOW_PLOTS:
            plt.show()
        plt.close('all')

    except Exception as e:
        print(Fore.RED + f'Galaxy {i} failed: {type(e).__name__}: {e}'
              + Style.RESET_ALL)
        plt.close('all')
        continue

print('\nDone. Plots in ./' + PLOT_DIR)
print("SDSS Plate Browser: https://skyserver.sdss.org/dr16/en/tools/getimg/plate.aspx")
print("SDSS SQL Query:     https://skyserver.sdss.org/dr12/en/tools/search/sql.aspx")
