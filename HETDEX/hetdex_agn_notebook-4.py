"""
hetdex_agn_notebook.py
======================
Jupyter-compatible version of the HETDEX AGN spectrum viewer.
Argparse replaced with a CONFIG cell at the top — edit the variables
there, then Run All.

HETDEX AGN FITS structure
--------------------------
HDU 1  basic_info          BinTableHDU   5322 rows x 159 cols  (one row per AGN)
HDU 2  flux_array          ImageHDU      (1036, 5322)           best-det spectra
HDU 3  error_array         ImageHDU      (1036, 5322)           errors for HDU2
HDU 4  repeat_info         BinTableHDU   6004 rows x 3 cols     repeat metadata
HDU 5  flux_array_repeat   ImageHDU      (1036, 6004)           repeat spectra
HDU 6  error_array_repeat  ImageHDU      (1036, 6004)           errors for HDU5

Requirements:  pip install astropy numpy matplotlib scipy
Data:          hetdex_agn.fits
               https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/hetdex_source_catalog_1/
               MD5: 29f08a675a818b07f2dc444a2324dee0
"""

# =============================================================================
# CELL 1 — CONFIGURATION  (edit these, then Run All)
# =============================================================================

CATALOG_PATH = "hetdex_agn.fits"   # path to hetdex_agn.fits

# --- Which AGN to plot? (only one of the four options is used, priority order)
AGNID      = None    # int  : plot this agnid        (set to None to skip)
ROW        = None    # int  : plot this row index    (set to None to skip)
ZRANGE     = None    # tuple: e.g. (1.8, 2.2)        (set to None to skip)
# If all three are None, the highest-S/N AGN is chosen automatically.

# --- Output
SAVE_PATH  = None    # str  : e.g. "agn9558.png"     (None = display inline)
OVERVIEW_N = None    # int  : show grid of N AGN     (None = single-AGN mode)

# =============================================================================
# CELL 2 — IMPORTS & CONSTANTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.ndimage import gaussian_filter1d
from scipy.signal  import savgol_filter

from astropy.io    import fits
from astropy.table import Table
import astropy.units as u
from astropy.cosmology import Planck18

# Jupyter inline display
try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
        matplotlib.rcParams["figure.dpi"] = 120
except Exception:
    pass

# HETDEX VIRUS wavelength grid (1036 pixels, 3470-5540 AA)
WAVE_START = 3470.0
WAVE_STEP  = (5540.0 - 3470.0) / (1036 - 1)   # ~1.9978 AA/px
N_PIX      = 1036
WAVE_GRID  = WAVE_START + WAVE_STEP * np.arange(N_PIX)
FLUX_UNIT  = r"$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$"

# AGN rest-frame emission lines (AA)
LINES = {
    r"Ly$\alpha$" : 1215.67,
    "N V"         : 1240.81,
    "C IV"        : 1549.48,
    "He II"       : 1640.40,
    "C III]"      : 1908.73,
    "Mg II"       : 2799.12,
    "[O II]"      : 3727.09,
    r"H$\delta$"  : 4101.73,
    r"H$\gamma$"  : 4340.47,
    r"H$\beta$"   : 4861.33,
    "[O III]"     : 4958.91,
    "[O III]*"    : 5006.84,
}
LABELS_SHORT = {
    r"Ly$\alpha$": "Lya",   "N V": "NV",      "C IV": "CIV",
    "He II": "HeII",         "C III]": "CIII]","Mg II": "MgII",
    "[O II]": "OII",         r"H$\delta$": "Hd",
    r"H$\gamma$": "Hg",     r"H$\beta$": "Hb",
    "[O III]": "OIII",       "[O III]*": "OIII*",
}

cosmo = Planck18

print("Imports OK.  WAVE_GRID:", WAVE_GRID[0], "–", WAVE_GRID[-1], "AA  |",
      N_PIX, "pixels  |  step =", round(WAVE_STEP, 4), "AA/px")

# =============================================================================
# CELL 3 — SYNTHETIC DATA (runs when FITS file is absent — delete cell if real)
# =============================================================================

def make_synthetic_agn(n_agn=5322, seed=42):
    """Mimics hetdex_agn.fits structure exactly for testing."""
    rng = np.random.default_rng(seed)
    z_arr    = np.sort(rng.uniform(0.05, 3.5, n_agn))[::-1]
    agnid    = np.arange(1, n_agn + 1, dtype=np.int64)
    ra       = rng.uniform(130.0, 230.0, n_agn).astype(np.float32)
    dec      = rng.uniform(43.0,  55.0,  n_agn).astype(np.float32)
    gmag     = rng.uniform(17.0,  24.0,  n_agn).astype(np.float32)
    sn       = rng.uniform(5.0,   40.0,  n_agn).astype(np.float32)
    roff     = rng.uniform(0.0,   1.5,   n_agn).astype(np.float32)

    flux_arr = np.zeros((N_PIX, n_agn), dtype=np.float32)
    err_arr  = np.zeros((N_PIX, n_agn), dtype=np.float32)

    line_waves    = np.array(list(LINES.values()))
    line_strengths= np.array([8, 2, 5, 1, 3, 4, 2, 0.5, 0.8, 2, 3, 5])

    for i in range(n_agn):
        z_i   = z_arr[i]
        scale = 10.0**((20.0 - gmag[i]) / 2.5) * 0.3
        cont  = np.clip(scale * (WAVE_GRID / 4500.0)**(-1.5), 0, None)
        spec  = cont.copy()
        for w_rest, strength in zip(line_waves, line_strengths):
            w_obs = w_rest * (1.0 + z_i)
            if WAVE_GRID[0] < w_obs < WAVE_GRID[-1]:
                sigma_aa = rng.uniform(1.5, 4.0) * WAVE_STEP
                amp      = strength * scale * rng.uniform(0.3, 2.5)
                spec    += amp * np.exp(-0.5 * ((WAVE_GRID - w_obs) / sigma_aa)**2)
        noise_level      = np.clip(np.abs(cont) * rng.uniform(0.08, 0.25) + 0.02*scale, 0.01, None)
        flux_arr[:, i]   = (spec + rng.normal(0, noise_level)).astype(np.float32)
        err_arr[:, i]    = noise_level.astype(np.float32)

    # Repeat observations
    n_repeat  = int(n_agn * 1.13)
    rep_agnid = rng.choice(agnid, size=n_repeat, replace=True)
    rep_shotid= rng.integers(20170000000, 20230000000, n_repeat)
    rep_sep   = rng.uniform(0.0, 2.0, n_repeat).astype(np.float32)
    flux_rep  = np.zeros((N_PIX, n_repeat), dtype=np.float32)
    err_rep   = np.zeros((N_PIX, n_repeat), dtype=np.float32)
    for j, aid in enumerate(rep_agnid):
        idx          = int(aid) - 1
        scale_j      = rng.uniform(0.7, 1.4)
        noise_j      = rng.normal(0, err_arr[:, idx] * rng.uniform(0.9, 1.3))
        flux_rep[:, j] = (flux_arr[:, idx] * scale_j + noise_j).astype(np.float32)
        err_rep[:, j]  = (err_arr[:, idx] * 1.1).astype(np.float32)

    return dict(z=z_arr.astype(np.float32), agnid=agnid, ra=ra, dec=dec,
                gmag=gmag, sn=sn, roff=roff,
                flux=flux_arr, err=err_arr,
                rep_agnid=rep_agnid, rep_shotid=rep_shotid, rep_sep=rep_sep,
                flux_rep=flux_rep, err_rep=err_rep)

# =============================================================================
# CELL 4 — LOAD CATALOG
# =============================================================================

def load_catalog(path):
    try:
        hdul = fits.open(path, memmap=True)
        info = Table(hdul[1].data)
        info.rename_columns(info.colnames, [c.lower() for c in info.colnames])

        print(f"HDU1 has {len(info.colnames)} columns.")
        print(f"First 30: {info.colnames[:30]}")

        # ── Required columns (raise if absent) ────────────────────────────────
        def col(names, required=True):
            for n in names:
                if n in info.colnames:
                    return np.array(info[n])
            if required:
                raise KeyError(
                    f"None of {names} found in HDU1.\n"
                    f"All columns: {info.colnames}"
                )
            return None   # optional

        # ── Optional columns: fill with NaN array if absent ───────────────────
        def col_or_nan(names, n_rows):
            v = col(names, required=False)
            return v.astype(np.float32) if v is not None \
                   else np.full(n_rows, np.nan, dtype=np.float32)

        n_agn = len(info)

        # gmag: not in the AGN FITS HDU1 — derive a proxy from fpl0 (power-law
        # normalisation at 4500 AA, units 1e-17 erg/s/cm2/AA) if present,
        # otherwise fill NaN.  fpl0 ~ flux density -> AB mag conversion:
        # m_AB = -2.5*log10(f_nu) - 48.6  with f_nu from f_lambda * lambda^2/c
        def gmag_proxy(info, n_agn):
            if "gmag" in info.colnames:
                return np.array(info["gmag"]).astype(np.float32)
            if "fpl0" in info.colnames:
                fpl0 = np.array(info["fpl0"]).astype(float)
                lam  = 4770.0   # SDSS g-band effective wavelength AA
                c_aa = 2.998e18 # speed of light AA/s
                fnu  = np.abs(fpl0) * 1e-17 * lam**2 / c_aa   # erg/s/cm2/Hz
                fnu  = np.clip(fnu, 1e-35, None)
                return (-2.5 * np.log10(fnu) - 48.6).astype(np.float32)
            return np.full(n_agn, np.nan, dtype=np.float32)

        # sn: may be stored as peak S/N, line S/N, or not at all
        # Fall back to nshots (number of observations) as a quality proxy
        def sn_proxy(info, n_agn):
            for name in ["sn", "snr", "sn_line", "sn_peak", "peak_sn"]:
                if name in info.colnames:
                    return np.array(info[name]).astype(np.float32)
            # Use nshots as a rough stand-in (more shots = better data)
            if "nshots" in info.colnames:
                print("  'sn' column absent — using nshots as S/N proxy.")
                return np.array(info["nshots"]).astype(np.float32)
            return np.full(n_agn, 1.0, dtype=np.float32)

        # ── Spectral arrays: normalise to shape (N_PIX, N_sources) ───────────
        # The FITS may be stored as (N_PIX, N_sources) or (N_sources, N_PIX).
        # We detect orientation by checking which axis matches N_PIX=1036.
        def orient(arr, n_sources, label):
            s = arr.shape
            print(f"  {label} raw shape: {s}")
            if s == (N_PIX, n_sources):
                return arr          # already correct
            if s == (n_sources, N_PIX):
                return arr.T        # transpose to (N_PIX, N_sources)
            # Ambiguous (e.g. n_sources == N_PIX): use heuristic — spectra
            # should have larger variance along the wavelength axis
            if s[0] == N_PIX:
                return arr
            if s[1] == N_PIX:
                return arr.T
            raise ValueError(
                f"{label}: cannot resolve orientation for shape {s}. "
                f"Expected one axis = N_PIX={N_PIX}, other = N_sources={n_sources}."
            )

        flux_raw = hdul[2].data
        err_raw  = hdul[3].data
        flux2d   = orient(flux_raw, n_agn, "HDU2 flux_array")
        err2d    = orient(err_raw,  n_agn, "HDU3 error_array")

        cat = dict(
            z     = col(["z", "redshift", "z_hetdex"]).astype(np.float32),
            agnid = col(["agnid", "agn_id", "source_id"]).astype(np.int64),
            ra    = col(["ra", "ra_best"]).astype(np.float32),
            dec   = col(["dec", "dec_best"]).astype(np.float32),
            roff  = col_or_nan(["roff", "r_off", "separation"], n_agn),
            gmag  = gmag_proxy(info, n_agn),
            sn    = sn_proxy(info, n_agn),
            flux  = flux2d,   # (N_PIX, N_agn)
            err   = err2d,
            _info = info,     # full table for Cell 9 column explorer
        )

        # ── Repeat table (HDU4) ────────────────────────────────────────────────
        rep = Table(hdul[4].data)
        rep.rename_columns(rep.colnames, [c.lower() for c in rep.colnames])
        print(f"HDU4 (repeat_info) columns: {rep.colnames}")

        def rcol(names):
            for n in names:
                if n in rep.colnames:
                    return np.array(rep[n])
            raise KeyError(f"None of {names} found in HDU4.\n"
                           f"All HDU4 columns: {rep.colnames}")

        n_repeat         = len(rep)
        cat["rep_agnid"] = rcol(["agnid", "agn_id"]).astype(np.int64)
        cat["rep_shotid"]= rcol(["shotid", "shot_id", "obsid"]).astype(np.int64)
        # Third column of HDU4 is nshots per repeat group
        cat["rep_sep"]   = np.array(
            rep[rep.colnames[2]] if len(rep.colnames) >= 3
            else np.zeros(n_repeat)
        ).astype(np.float32)

        flux_rep_raw = hdul[5].data
        err_rep_raw  = hdul[6].data
        cat["flux_rep"] = orient(flux_rep_raw, n_repeat, "HDU5 flux_array_repeat")
        cat["err_rep"]  = orient(err_rep_raw,  n_repeat, "HDU6 error_array_repeat")
        hdul.close()

        synthetic = False
        print(f"\nLoaded {path}:")
        print(f"  {n_agn:,} AGN         flux shape: {cat['flux'].shape}")
        print(f"  {n_repeat:,} repeat obs   flux_rep shape: {cat['flux_rep'].shape}")
        print(f"  gmag proxy: {'fpl0' in info.colnames or 'gmag' in info.colnames}")

    except FileNotFoundError:
        print(f"'{path}' not found — running with synthetic demo data.")
        print("Set CATALOG_PATH in Cell 1 to point at hetdex_agn.fits.")
        cat       = make_synthetic_agn()
        synthetic = True
        print(f"Synthetic:  {len(cat['agnid']):,} AGN  |  "
              f"{cat['flux_rep'].shape[1]:,} repeat obs")

    return cat, synthetic


cat, SYNTHETIC = load_catalog(CATALOG_PATH)

# Confirm orientations
print(f"flux shape    : {cat['flux'].shape}   (should be ({N_PIX}, N_agn))")
print(f"flux_rep shape: {cat['flux_rep'].shape}   (should be ({N_PIX}, N_repeat))")

# =============================================================================
# CELL 5 — RESOLVE WHICH AGN TO PLOT
# =============================================================================

def resolve_row(cat, agnid=None, row=None, zrange=None):
    n = len(cat["agnid"])
    if agnid is not None:
        hits = np.where(cat["agnid"] == agnid)[0]
        if len(hits) == 0:
            raise ValueError(f"agnid={agnid} not found. "
                             f"Valid range: {cat['agnid'].min()}–{cat['agnid'].max()}")
        r = int(hits[0])
        print(f"agnid={agnid} -> row {r}")
        return r
    if row is not None:
        if not (0 <= row < n):
            raise ValueError(f"row={row} out of range (0–{n-1})")
        print(f"Using row {row}  (agnid={cat['agnid'][row]})")
        return row
    if zrange is not None:
        z1, z2 = zrange
        cands = np.where((cat["z"] >= z1) & (cat["z"] <= z2))[0]
        if len(cands) == 0:
            raise ValueError(f"No AGN with z in [{z1}, {z2}]")
        r = int(np.random.choice(cands))
        print(f"Random AGN in z=[{z1},{z2}]: row {r}  "
              f"(agnid={cat['agnid'][r]}, z={cat['z'][r]:.4f})")
        return r
    # Default: highest S/N
    r = int(np.argmax(cat["sn"]))
    print(f"Highest S/N AGN: row {r}  "
          f"(agnid={cat['agnid'][r]}, z={cat['z'][r]:.4f}, "
          f"S/N={cat['sn'][r]:.1f})")
    return r


TARGET_ROW = resolve_row(cat, agnid=AGNID, row=ROW, zrange=ZRANGE)

# Quick summary
_a = cat["agnid"][TARGET_ROW]
_z = cat["z"][TARGET_ROW]
_n_rep = int((cat["rep_agnid"] == _a).sum())
_n_rep = min(_n_rep, cat["flux_rep"].shape[1])  # clip to actual HDU5 columns
print(f"\nTarget:  agnid={_a}  z={_z:.4f}  "
      f"g={cat['gmag'][TARGET_ROW]:.1f}  "
      f"S/N={cat['sn'][TARGET_ROW]:.1f}  "
      f"repeat_obs={_n_rep}")

# =============================================================================
# CELL 6 — HELPER FUNCTIONS
# =============================================================================

def get_spectrum(cat, row):
    """Best-detection spectrum for HDU1 row index."""
    flux = cat["flux"][:, row].astype(float)
    err  = np.clip(cat["err"][:, row].astype(float), 1e-5, None)
    return flux, err


def get_repeats(cat, agnid):
    """
    All repeat spectra (HDU5/6) for a given agnid.

    HDU4 has one row per repeat-spectrum entry; each row's 'agnid' field
    identifies which AGN it belongs to.  HDU5 rows correspond 1-to-1 with
    HDU4 rows, so we find the HDU4 rows matching our agnid and index HDU5
    with those same row numbers.

    Returns (list_of_flux_arrays, list_of_err_arrays, shotid_array).
    Returns empty lists if no repeats exist for this agnid.
    """
    mask = cat["rep_agnid"] == agnid
    idx  = np.where(mask)[0]

    if len(idx) == 0:
        return [], [], np.array([], dtype=np.int64)

    # Clip to valid column range of flux_rep
    n_rep_cols = cat["flux_rep"].shape[1]
    idx        = idx[idx < n_rep_cols]

    fluxes  = [cat["flux_rep"][:, j].astype(float) for j in idx]
    errs    = [np.clip(cat["err_rep"][:, j].astype(float), 1e-5, None)
               for j in idx]
    shotids = cat["rep_shotid"][mask][:len(idx)]
    return fluxes, errs, shotids


def smooth(flux, window=7, poly=3):
    try:
        return savgol_filter(flux, window_length=window, polyorder=poly)
    except Exception:
        return gaussian_filter1d(flux, sigma=2)


def ylims(flux, lo_pct=1.0, hi_pct=99.2, pad_frac=0.12):
    f = flux[np.isfinite(flux)]
    lo = np.percentile(f, lo_pct)
    hi = np.percentile(f, hi_pct)
    p  = pad_frac * (hi - lo)
    return lo - p, hi + p


def draw_lines(ax, z, ylo, yhi, alpha=0.7, fontsize=7.5):
    for name, w_rest in LINES.items():
        w_obs = w_rest * (1.0 + z)
        if WAVE_GRID[0] + 10 < w_obs < WAVE_GRID[-1] - 10:
            ax.axvline(w_obs, color="#ffa657", lw=0.8, ls="--", alpha=alpha)
            ax.text(w_obs + 3, ylo + 0.87*(yhi - ylo),
                    LABELS_SHORT.get(name, name),
                    color="#ffa657", fontsize=fontsize,
                    rotation=90, va="top", ha="left", alpha=0.88)


# Style helper
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

def style_ax(ax, title, xl, yl):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, which="both", direction="in",
                   top=True, right=True, labelsize=9)
    ax.set_xlabel(xl, color=TEXT, fontsize=10)
    ax.set_ylabel(yl, color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=11,
                 fontweight="bold", loc="left", pad=6)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def mleg(ax, **kw):
    return ax.legend(fontsize=8.5, facecolor="#21262d",
                     edgecolor=SPINE, labelcolor=TEXT, **kw)

print("Helper functions defined.")

# =============================================================================
# CELL 7 — MAIN PLOT: single AGN, four panels
# =============================================================================

row    = TARGET_ROW
agnid  = int(cat["agnid"][row])
z      = float(cat["z"][row])
gmag   = float(cat["gmag"][row])
sn_val = float(cat["sn"][row])
ra     = float(cat["ra"][row])
dec    = float(cat["dec"][row])

flux_best, err_best = get_spectrum(cat, row)
rep_fluxes, rep_errs, shotids = get_repeats(cat, agnid)

all_fluxes  = [flux_best] + rep_fluxes
stack_med   = np.nanmedian(np.vstack(all_fluxes), axis=0)
smooth_best = smooth(flux_best)
wave_rest   = WAVE_GRID / (1.0 + z)

OBS_XLABEL  = r"Observed wavelength  $\lambda_{\rm obs}$  (Å)"
REST_XLABEL = r"Rest-frame wavelength  $\lambda_{\rm rest}$  (Å)"

fig  = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor(BG)
gs_outer = gridspec.GridSpec(3, 1, figure=fig,
                              hspace=0.42,
                              left=0.08, right=0.97,
                              top=0.91,  bottom=0.06)
gs_bot   = gridspec.GridSpecFromSubplotSpec(
               1, 2, subplot_spec=gs_outer[2], wspace=0.30)

ax_obs  = fig.add_subplot(gs_outer[0])
ax_rest = fig.add_subplot(gs_outer[1])
ax_rep  = fig.add_subplot(gs_bot[0])
ax_sn   = fig.add_subplot(gs_bot[1])

# ── Panel 1: Observed-frame spectrum ─────────────────────────────────────────
syn_tag = "  [SYNTHETIC]" if SYNTHETIC else ""
style_ax(ax_obs,
         f"agnid = {agnid}  |  z = {z:.4f}  |  "
         f"g = {gmag:.1f}  |  S/N = {sn_val:.1f}  |  "
         f"RA = {ra:.4f}  Dec = {dec:.4f}{syn_tag}",
         OBS_XLABEL, FLUX_UNIT)

ylo, yhi = ylims(flux_best)

ax_obs.fill_between(WAVE_GRID,
                    flux_best - err_best,
                    flux_best + err_best,
                    color="#58a6ff", alpha=0.20,
                    label=r"$\pm1\sigma$ error")
ax_obs.plot(WAVE_GRID, flux_best,
            color="#58a6ff", lw=0.6, alpha=0.55,
            label="Best-detection spectrum")
ax_obs.plot(WAVE_GRID, smooth_best,
            color="#e6edf3", lw=1.5, alpha=0.92,
            label="Smoothed (SavGol)")
ax_obs.plot(WAVE_GRID, stack_med,
            color="#3fb950", lw=1.1, alpha=0.78, ls="--",
            label=f"Epoch median  (N = {len(all_fluxes)})")
ax_obs.axhline(0, color=SPINE, lw=0.8, ls=":")

draw_lines(ax_obs, z, ylo, yhi)

ax_obs.set_xlim(WAVE_GRID[0], WAVE_GRID[-1])
ax_obs.set_ylim(ylo, yhi)
mleg(ax_obs, loc="upper right", ncol=2)

# ── Panel 2: Rest-frame spectrum ──────────────────────────────────────────────
style_ax(ax_rest,
         f"Rest-frame  (z = {z:.4f})",
         REST_XLABEL, FLUX_UNIT)

ylo_r, yhi_r = ylims(flux_best)

ax_rest.fill_between(wave_rest,
                     flux_best - err_best,
                     flux_best + err_best,
                     color="#d2a8ff", alpha=0.20)
ax_rest.plot(wave_rest, flux_best,
             color="#d2a8ff", lw=0.6, alpha=0.55,
             label="Spectrum (rest-frame)")
ax_rest.plot(wave_rest, smooth_best,
             color="#e6edf3", lw=1.5, alpha=0.92,
             label="Smoothed")
ax_rest.axhline(0, color=SPINE, lw=0.8, ls=":")

# Rest-frame line markers (no redshift shift needed)
for name, w_rest_val in LINES.items():
    if wave_rest[0] + 5 < w_rest_val < wave_rest[-1] - 5:
        ax_rest.axvline(w_rest_val, color="#ffa657",
                        lw=0.85, ls="--", alpha=0.75)
        ax_rest.text(w_rest_val + 2,
                     ylo_r + 0.86*(yhi_r - ylo_r),
                     LABELS_SHORT.get(name, name),
                     color="#ffa657", fontsize=7.5,
                     rotation=90, va="top", ha="left", alpha=0.88)

ax_rest.set_xlim(wave_rest[0], wave_rest[-1])
ax_rest.set_ylim(ylo_r, yhi_r)
mleg(ax_rest, loc="upper right")

# ── Panel 3: Repeat epochs ────────────────────────────────────────────────────
style_ax(ax_rep,
         f"Repeat epochs  (N = {len(rep_fluxes) + 1})",
         OBS_XLABEL, FLUX_UNIT)

ax_rep.plot(WAVE_GRID, flux_best,
            color="#e6edf3", lw=1.0, alpha=0.88,
            label="Best detection", zorder=10)

epoch_colors = plt.cm.plasma(np.linspace(0.15, 0.90, max(len(rep_fluxes), 1)))
for j, (rf, sid, ec) in enumerate(zip(rep_fluxes, shotids, epoch_colors)):
    lbl = str(sid) if j < 6 else ("..." if j == 6 else None)
    ax_rep.plot(WAVE_GRID, rf,
                color=ec, lw=0.55, alpha=0.52, label=lbl)

ax_rep.plot(WAVE_GRID, stack_med,
            color="#3fb950", lw=1.8, alpha=0.90, ls="--",
            label="Epoch median", zorder=9)
ax_rep.axhline(0, color=SPINE, lw=0.7, ls=":")

all_v = np.concatenate([flux_best] + rep_fluxes) if rep_fluxes else flux_best
ax_rep.set_xlim(WAVE_GRID[0], WAVE_GRID[-1])
ax_rep.set_ylim(*ylims(all_v))
ax_rep.legend(fontsize=6.5, facecolor="#21262d",
              edgecolor=SPINE, labelcolor=TEXT,
              loc="upper right", ncol=2, handlelength=1.2)

# ── Panel 4: S/N spectrum ─────────────────────────────────────────────────────
style_ax(ax_sn, "Signal-to-Noise per pixel",
         OBS_XLABEL, "S/N")

snr        = flux_best / np.clip(err_best, 1e-10, None)
snr_smooth = smooth(snr, window=11)

ax_sn.fill_between(WAVE_GRID, 0, snr,
                   where=(snr >= 0), interpolate=True,
                   color="#58a6ff", alpha=0.32)
ax_sn.fill_between(WAVE_GRID, 0, snr,
                   where=(snr < 0), interpolate=True,
                   color="#f78166", alpha=0.32)
ax_sn.plot(WAVE_GRID, snr,
           color=MUTED, lw=0.5, alpha=0.55)
ax_sn.plot(WAVE_GRID, snr_smooth,
           color="#e6edf3", lw=1.4, alpha=0.92,
           label="Smoothed S/N")
ax_sn.axhline(0,  color=SPINE,    lw=0.8, ls=":")
ax_sn.axhline(5,  color="#3fb950",lw=0.9, ls="--", alpha=0.65, label="5σ")
ax_sn.axhline(-2, color="#f78166",lw=0.8, ls="--", alpha=0.55, label="−2σ")

sn_vals = snr[np.isfinite(snr)]
ax_sn.set_xlim(WAVE_GRID[0], WAVE_GRID[-1])
ax_sn.set_ylim(np.percentile(sn_vals, 0.5) - 1,
               np.percentile(sn_vals, 99.5) + 2)
mleg(ax_sn, loc="upper right")

fig.suptitle("HETDEX AGN Spectrum Viewer",
             color=TEXT, fontsize=14, fontweight="bold", y=0.975)

# Save first (before plt.show() clears the figure from memory in Jupyter),
# then display inline regardless.
if SAVE_PATH:
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 8 — OPTIONAL: Overview grid of multiple AGN
# =============================================================================

if OVERVIEW_N is not None and OVERVIEW_N > 0:
    n_show  = min(OVERVIEW_N, len(cat["agnid"]))
    ov_rows = sorted(np.random.choice(len(cat["agnid"]),
                                      size=n_show, replace=False))
    ncols   = 3
    nrows   = int(np.ceil(n_show / ncols))

    fig2, axes = plt.subplots(nrows, ncols,
                               figsize=(6*ncols, 3.5*nrows),
                               facecolor=BG)
    axes = np.array(axes).flatten()

    for ax, r in zip(axes, ov_rows):
        z_i   = float(cat["z"][r])
        aid_i = int(cat["agnid"][r])
        f, e  = get_spectrum(cat, r)
        sm    = smooth(f)
        ax.set_facecolor(AX_BG)
        for sp in ax.spines.values():
            sp.set_color(SPINE)
        ax.tick_params(colors=MUTED, which="both",
                       direction="in", labelsize=7)
        ylo_i, yhi_i = ylims(f)
        ax.fill_between(WAVE_GRID, f - e, f + e,
                        color="#58a6ff", alpha=0.18)
        ax.plot(WAVE_GRID, f,  color="#58a6ff", lw=0.5, alpha=0.50)
        ax.plot(WAVE_GRID, sm, color="#e6edf3", lw=1.1)
        ax.axhline(0, color=SPINE, lw=0.7, ls=":")
        draw_lines(ax, z_i, ylo_i, yhi_i, alpha=0.45, fontsize=6.5)
        ax.set_xlim(WAVE_GRID[0], WAVE_GRID[-1])
        ax.set_ylim(ylo_i, yhi_i)
        ax.set_title(f"agnid={aid_i}  z={z_i:.3f}",
                     color=TEXT, fontsize=8, fontweight="bold", pad=4)

    for ax in axes[n_show:]:
        ax.set_visible(False)

    syn_tag2 = "  [SYNTHETIC]" if SYNTHETIC else ""
    fig2.suptitle(f"HETDEX AGN Overview  (N = {n_show}){syn_tag2}",
                  color=TEXT, fontsize=13, fontweight="bold", y=1.01)
    fig2.tight_layout()

    if SAVE_PATH:
        ov_path = SAVE_PATH.replace(".png", "_overview.png")
        fig2.savefig(ov_path, dpi=130, bbox_inches="tight", facecolor=BG)
        print(f"Saved -> {ov_path}")
    plt.show()

# =============================================================================
# CELL 9 — NUMERICAL SUMMARY + HDU1 COLUMN EXPLORER
# =============================================================================

flux_b, err_b = get_spectrum(cat, TARGET_ROW)
snr_all       = flux_b / np.clip(err_b, 1e-10, None)
n_rep_tgt     = (cat["rep_agnid"] == cat["agnid"][TARGET_ROW]).sum()

def _fmt(val):
    """Format a scalar gracefully even if NaN."""
    if np.isnan(val):
        return "n/a (column absent in HDU1)"
    return f"{val:.4f}"

print("=" * 62)
print("  HETDEX AGN — Spectral Summary")
print("=" * 62)
print(f"  agnid            : {cat['agnid'][TARGET_ROW]}")
print(f"  Redshift (z)     : {cat['z'][TARGET_ROW]:.6f}")
print(f"  RA               : {cat['ra'][TARGET_ROW]:.6f} deg")
print(f"  Dec              : {cat['dec'][TARGET_ROW]:.6f} deg")
print(f"  g magnitude      : {_fmt(cat['gmag'][TARGET_ROW])}")
print(f"  Catalog S/N      : {_fmt(cat['sn'][TARGET_ROW])}")
print(f"  roff             : {_fmt(cat['roff'][TARGET_ROW])} arcsec")
print(f"  Repeat obs       : {n_rep_tgt}")
print(f"  Wavelength range : {WAVE_GRID[0]:.1f} – {WAVE_GRID[-1]:.1f} AA")
print(f"  Dispersion       : {WAVE_STEP:.4f} AA/px")
print(f"  Pixels           : {N_PIX}")
print(f"  Median pixel S/N : {np.nanmedian(snr_all):.2f}")
print(f"  Peak   pixel S/N : {np.nanmax(snr_all):.2f}")
print(f"  Flux range       : {flux_b.min():.3g} – {flux_b.max():.3g}  (1e-17 cgs)")

# Emission lines visible in VIRUS band
print(f"\n  Lines in VIRUS band (z = {cat['z'][TARGET_ROW]:.4f}):")
for name, w_rest in LINES.items():
    w_obs = w_rest * (1.0 + cat["z"][TARGET_ROW])
    if WAVE_GRID[0] < w_obs < WAVE_GRID[-1]:
        print(f"    {LABELS_SHORT.get(name, name):8s}  "
              f"rest = {w_rest:.2f} AA  ->  obs = {w_obs:.2f} AA")
print("=" * 62)

# ── HDU1 column explorer (real catalog only) ──────────────────────────────────
if not SYNTHETIC and "_info" in cat:
    info_tab = cat["_info"]
    print(f"\n  HDU1 — all {len(info_tab.colnames)} columns for this AGN:")
    print(f"  {'Column':<28}  {'Value':>14}")
    print(f"  {'-'*28}  {'-'*14}")
    for cname in info_tab.colnames:
        val = info_tab[cname][TARGET_ROW]
        # Truncate long array-valued cells
        if hasattr(val, "__len__") and len(val) > 4:
            display_val = f"[array len={len(val)}]"
        elif isinstance(val, (float, np.floating)):
            display_val = f"{val:.6g}"
        else:
            display_val = str(val)
        print(f"  {cname:<28}  {display_val:>14}")
