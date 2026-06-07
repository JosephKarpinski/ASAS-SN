"""
hetdex_lae_stack.py
===================
Load HETDEX SC2 spectral FITS data, shift individual spectra to rest-frame
using z_hetdex, and create a median-stacked composite spectrum for LAEs in
a user-defined redshift bin.

FITS structure  (hetdex_sc2_spec_v1.5.fits)
--------------------------------------------
  HDU 1  INFO       BinTableHDU   Source Observation Table
  HDU 2  SPEC       ImageHDU      (1036, N_sources)  flux  [1e-17 cgs/AA]
  HDU 3  SPEC_ERR   ImageHDU      (1036, N_sources)  1-sigma errors
  HDU 4  WAVELENGTH ImageHDU      (1036,)             observed wavelength grid [AA]

Stacking method
---------------
1. Select LAEs in the requested z-bin with quality cuts.
2. For each source: interpolate flux & error onto a common rest-frame
   wavelength grid using the source redshift z_hetdex.
3. Normalise each spectrum to its median flux in a continuum window
   adjacent to Lyα (1260–1350 Å rest) before stacking, so bright
   sources do not dominate.
4. Stack with numpy.nanmedian.  Uncertainty estimated via bootstrap
   (N_BOOTSTRAP resamples) and the MAD-based "median absolute deviation"
   propagation:  σ_stack = 1.4826 * MAD / sqrt(N_contributing).
5. Measure Lyα equivalent width (EW_rest) and FWHM on the stacked profile
   by fitting a Gaussian to the Lyα line.

Key columns used from INFO table
----------------------------------
  source_type, z_hetdex, p_conf, p_cnn, sn, field, logL_lya

Requirements
------------
  pip install astropy numpy matplotlib scipy

Data
----
  hetdex_sc2_spec_v1.5.fits
  https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/hetdex_source_catalog_2/
"""

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

SPEC_PATH   = "hetdex_sc2_spec_v1.5.fits"   # update to local path
SAVE_PATH   = "hetdex_lae_stack.png"         # None = display inline only
CSV_PATH    = "hetdex_lae_stack.csv"         # stacked spectrum output

# Redshift bins to stack  (can be a list — one panel per bin)
Z_BINS = [
    (2.0, 2.4),
    (2.4, 2.8),
    (2.8, 3.2),
]

# Quality cuts
MIN_SN      = 5.5     # line S/N
MIN_P_CONF  = 0.5     # RF classifier
MIN_P_CNN   = 0.5     # CNN classifier
MAX_SOURCES = 5000    # cap per bin (random draw) to keep runtime fast
                      # set to None to use all

# Rest-frame output grid
REST_WAVE_MIN  = 1050.0   # AA  (covers OVI, Lyβ, Lyα, NV)
REST_WAVE_MAX  = 1700.0   # AA  (covers CIV, HeII)
REST_WAVE_STEP = 1.0      # AA  — finer than native 2 AA for smooth stacking

# Normalisation window (rest-frame AA) — continuum red-ward of Lyα
NORM_WAVE_MIN  = 1260.0
NORM_WAVE_MAX  = 1350.0

# Bootstrap uncertainty estimation
N_BOOTSTRAP = 200   # number of resample iterations (set 0 to skip)

BAD = -999.0

# =============================================================================
# CELL 2 — IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker  import AutoMinorLocator, MultipleLocator
from scipy.interpolate  import interp1d
from scipy.optimize     import curve_fit
from scipy.stats        import median_abs_deviation

from astropy.io    import fits
from astropy.table import Table
import astropy.units as u
from astropy.cosmology import Planck18

try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
        matplotlib.rcParams["figure.dpi"] = 120
except Exception:
    pass

cosmo  = Planck18
LYA_AA = 1215.67

print("Imports OK.")

# =============================================================================
# CELL 3 — SYNTHETIC DATA GENERATOR
# =============================================================================

def make_synthetic_spec(n_sources=4000, seed=17):
    """
    Synthetic hetdex_sc2_spec_v1.5.fits equivalent.
    Produces realistic power-law + Lya + UV line spectra for LAEs across
    the HETDEX redshift range, with noise.
    """
    rng  = np.random.default_rng(seed)
    N_PIX = 1036
    wave_obs = np.linspace(3470.0, 5540.0, N_PIX)   # AA

    # Source properties
    z_arr  = rng.uniform(1.90, 3.50, n_sources).astype(np.float32)
    stype  = np.where(z_arr > 1.87, "lae", "oii")
    sn_arr = np.abs(rng.lognormal(2.0, 0.6, n_sources)).astype(np.float32)
    logL   = rng.uniform(42.0, 44.0, n_sources).astype(np.float32)
    p_conf = np.clip(rng.beta(4, 1.5, n_sources), 0, 1).astype(np.float32)
    p_cnn  = np.clip(rng.beta(3.5, 1.3, n_sources), 0, 1).astype(np.float32)
    field  = rng.choice(["dex-spring","dex-fall","cosmos","goods-n"],
                         n_sources, p=[0.55,0.30,0.10,0.05])

    # Rest-frame UV lines (wavelength AA, strength relative to continuum)
    UV_LINES = [
        (1215.67, 8.0, 5.0),    # Lya  — strong, asymmetric (use Gaussian approx)
        (1240.81, 0.8, 2.5),    # NV
        (1302.17, 0.5, 2.0),    # OI
        (1335.31, 0.6, 2.0),    # CII
        (1549.48, 2.5, 3.5),    # CIV
        (1640.40, 0.8, 2.5),    # HeII
        (1908.73, 1.5, 3.0),    # CIII]
    ]

    spec_arr = np.zeros((N_PIX, n_sources), dtype=np.float32)
    err_arr  = np.zeros((N_PIX, n_sources), dtype=np.float32)

    for i in range(n_sources):
        z_i   = float(z_arr[i])
        scale = 10.0**((logL[i] - 43.0) * 0.4) * 0.5  # rough flux scale

        # Power-law continuum
        cont = scale * (wave_obs / 4500.0)**(-1.5)
        cont = np.clip(cont, 0, None)
        spec = cont.copy()

        # Add UV lines at observed wavelengths
        for w_rest, amp_fac, sigma_aa in UV_LINES:
            w_obs_i = w_rest * (1.0 + z_i)
            if wave_obs[0] < w_obs_i < wave_obs[-1]:
                amp   = amp_fac * scale * rng.uniform(0.5, 2.0)
                sigma = sigma_aa * (1.0 + z_i)   # observed sigma
                spec += amp * np.exp(-0.5 * ((wave_obs - w_obs_i) / sigma)**2)

        # Noise
        noise_lev = np.abs(cont) * rng.uniform(0.10, 0.30) + 0.01 * scale
        noise_lev = np.clip(noise_lev, 1e-4, None)
        spec     += rng.normal(0, noise_lev)

        spec_arr[:, i] = spec.astype(np.float32)
        err_arr[:, i]  = noise_lev.astype(np.float32)

    # Pack into astropy tables / arrays matching the real HDU structure
    info = Table({
        "source_id"  : np.arange(n_sources, dtype=np.int64),
        "source_type": stype,
        "z_hetdex"   : z_arr,
        "sn"         : sn_arr,
        "logL_lya"   : logL,
        "p_conf"     : p_conf,
        "p_cnn"      : p_cnn,
        "field"      : field,
    })
    return info, spec_arr, err_arr, wave_obs.astype(np.float32)


# =============================================================================
# CELL 4 — LOAD SPECTRAL FITS
# =============================================================================

def load_spec_fits(path):
    """
    Load hetdex_sc2_spec_v1.5.fits.
    Returns (info_table, spec_2d, err_2d, wave_1d) with orientation
    normalised to spec_2d.shape == (N_PIX, N_sources).
    """
    try:
        hdul = fits.open(path, memmap=True)
        info = Table(hdul[1].data)
        info.rename_columns(info.colnames,
                            [c.lower() for c in info.colnames])

        spec_raw = np.array(hdul[2].data, dtype=np.float32)
        err_raw  = np.array(hdul[3].data, dtype=np.float32)
        wave_raw = np.array(hdul[4].data, dtype=np.float32).ravel()
        hdul.close()

        n_sources = len(info)
        n_pix     = len(wave_raw)

        # Normalise to (N_PIX, N_sources)
        def orient(arr, label):
            if arr.shape == (n_pix, n_sources):
                return arr
            if arr.shape == (n_sources, n_pix):
                print(f"  {label}: transposing {arr.shape} -> "
                      f"({n_pix}, {n_sources})")
                return arr.T
            raise ValueError(f"{label} shape {arr.shape} unrecognised. "
                             f"Expected ({n_pix},{n_sources}) or "
                             f"({n_sources},{n_pix})")

        spec_2d = orient(spec_raw, "SPEC")
        err_2d  = orient(err_raw,  "SPEC_ERR")

        synthetic = False
        print(f"Loaded {path}:")
        print(f"  Sources  : {n_sources:,}")
        print(f"  Pixels   : {n_pix}")
        print(f"  Wave range: {wave_raw[0]:.1f} – {wave_raw[-1]:.1f} AA")
        print(f"  Columns  : {info.colnames}")

    except FileNotFoundError:
        print(f"'{path}' not found — using synthetic demo data.")
        info, spec_2d, err_2d, wave_raw = make_synthetic_spec()
        synthetic = True
        print(f"  Synthetic: {len(info):,} sources, "
              f"{spec_2d.shape[0]} pixels")

    return info, spec_2d, err_2d, wave_raw, synthetic


info, spec_2d, err_2d, wave_obs, SYNTHETIC = load_spec_fits(SPEC_PATH)

# =============================================================================
# CELL 5 — HELPER FUNCTIONS
# =============================================================================

def getcol(tab, *cands):
    lc = {c.lower(): c for c in tab.colnames}
    for c in cands:
        if c.lower() in lc:
            return lc[c.lower()]
    raise KeyError(f"None of {cands} in table. Have: {tab.colnames[:20]}")

# Build rest-frame output grid (common to all sources)
rest_wave = np.arange(REST_WAVE_MIN, REST_WAVE_MAX + REST_WAVE_STEP,
                      REST_WAVE_STEP)
N_REST    = len(rest_wave)

def shift_to_restframe(flux_obs, err_obs, wave_obs_arr, z):
    """
    Shift one observed spectrum to rest-frame by dividing wavelengths by (1+z).
    Interpolate onto the common rest_wave grid.

    Physical note: the flux density transforms as
        f_lambda_rest = f_lambda_obs * (1+z)
    (energy is redshifted, but wavelength interval contracts by same factor).
    We apply this (1+z) factor here.

    Returns (flux_rest, err_rest) interpolated onto rest_wave.
    NaN outside the coverage range.
    """
    wave_rest_src = wave_obs_arr / (1.0 + z)
    flux_factor   = 1.0 + z    # cosmological (1+z) brightening correction

    # Only interpolate within the observed coverage
    in_range = (rest_wave >= wave_rest_src[0]) & \
               (rest_wave <= wave_rest_src[-1])

    flux_out = np.full(N_REST, np.nan, dtype=np.float64)
    err_out  = np.full(N_REST, np.nan, dtype=np.float64)

    if in_range.sum() < 5:
        return flux_out, err_out

    # Linear interpolation is fast and accurate for 2AA pixels
    f_interp = interp1d(wave_rest_src, flux_obs,
                        kind="linear", bounds_error=False,
                        fill_value=np.nan)
    e_interp = interp1d(wave_rest_src, err_obs,
                        kind="linear", bounds_error=False,
                        fill_value=np.nan)

    flux_out[in_range] = f_interp(rest_wave[in_range]) * flux_factor
    err_out[in_range]  = e_interp(rest_wave[in_range]) * flux_factor

    return flux_out, err_out


def normalise_spectrum(flux, wave, w_min=NORM_WAVE_MIN, w_max=NORM_WAVE_MAX):
    """
    Normalise by the median flux in the continuum window [w_min, w_max].
    Returns (flux_norm, norm_factor).
    NaN flux or zero continuum -> returns (flux, 1.0).
    """
    win   = (wave >= w_min) & (wave <= w_max)
    f_win = flux[win]
    f_win = f_win[np.isfinite(f_win)]
    if len(f_win) < 3:
        return flux, 1.0
    median_cont = float(np.nanmedian(f_win))
    if not np.isfinite(median_cont) or median_cont <= 0:
        return flux, 1.0
    return flux / median_cont, median_cont


def gaussian(x, amp, cen, sigma, offset):
    return amp * np.exp(-0.5 * ((x - cen) / sigma)**2) + offset


def fit_lya(wave, flux, err=None, win_aa=40.0):
    """
    Fit a Gaussian to the Lyα line in rest-frame spectrum.
    Returns dict with amp, cen, sigma, fwhm, ew_rest, or None on failure.
    """
    cen0    = LYA_AA
    win_lo  = cen0 - win_aa
    win_hi  = cen0 + win_aa
    mask    = (wave >= win_lo) & (wave <= win_hi) & np.isfinite(flux)
    if mask.sum() < 8:
        return None
    x_fit   = wave[mask]
    y_fit   = flux[mask]
    sigma_f = err[mask] if err is not None else None

    # Estimate continuum from flanks
    flank   = ((wave >= win_lo) & (wave < cen0 - 10)) | \
               ((wave > cen0 + 20) & (wave <= win_hi))
    cont0   = float(np.nanmedian(flux[flank & np.isfinite(flux)])) \
              if flank.sum() > 2 else 0.0
    amp0    = float(np.nanmax(y_fit)) - cont0

    try:
        p0     = [amp0, cen0, 5.0, cont0]
        bounds = ([0, cen0-15, 0.5, -np.inf],
                  [np.inf, cen0+15, 30, np.inf])
        popt, pcov = curve_fit(
            gaussian, x_fit, y_fit, p0=p0,
            sigma=sigma_f, absolute_sigma=(sigma_f is not None),
            bounds=bounds, maxfev=4000,
        )
        perr  = np.sqrt(np.diag(pcov))
        fwhm  = 2.355 * abs(popt[2])   # AA rest-frame
        # EW_rest = integral(line) / continuum
        # For Gaussian: integral = amp * sigma * sqrt(2pi)
        integral = popt[0] * abs(popt[2]) * np.sqrt(2 * np.pi)
        cont_val = max(abs(popt[3]), 1e-10)
        ew_rest  = integral / cont_val
        return {
            "amp"   : popt[0], "amp_err" : perr[0],
            "cen"   : popt[1], "cen_err" : perr[1],
            "sigma" : popt[2], "sigma_err": perr[2],
            "fwhm"  : fwhm,
            "ew_rest": ew_rest,
            "cont"  : popt[3],
        }
    except Exception:
        return None


# =============================================================================
# CELL 6 — BUILD STACKED SPECTRA PER Z-BIN
# =============================================================================

# Resolve column names
STYPE_COL  = getcol(info, "source_type")
Z_COL      = getcol(info, "z_hetdex")
SN_COL     = getcol(info, "sn")
PCONF_COL  = getcol(info, "p_conf")
PCNN_COL   = getcol(info, "p_cnn")

z_arr    = np.array(info[Z_COL],    dtype=float)
stype_arr= np.array(info[STYPE_COL],dtype=str)
sn_arr   = np.array(info[SN_COL],   dtype=float)
pconf_arr= np.array(info[PCONF_COL],dtype=float)
pcnn_arr = np.array(info[PCNN_COL], dtype=float)

stacks = []    # one dict per z-bin

for z_lo, z_hi in Z_BINS:
    print(f"\n--- z = [{z_lo}, {z_hi}) ---")

    # Selection mask
    sel = (
        (np.array([s.strip().lower() for s in stype_arr]) == "lae") &
        (z_arr  >= z_lo)   & (z_arr  < z_hi)  &
        (sn_arr >= MIN_SN)  & (sn_arr != BAD)  &
        (pconf_arr >= MIN_P_CONF) &
        (pcnn_arr  >= MIN_P_CNN)  &
        np.isfinite(z_arr)
    )
    idx_sel = np.where(sel)[0]
    print(f"  Selected: {len(idx_sel):,} LAEs")

    if len(idx_sel) == 0:
        print("  *** No sources — skipping this bin ***")
        stacks.append(None)
        continue

    # Optional random downsample
    if MAX_SOURCES is not None and len(idx_sel) > MAX_SOURCES:
        rng_sub = np.random.default_rng(len(idx_sel))
        idx_sel = rng_sub.choice(idx_sel, size=MAX_SOURCES, replace=False)
        print(f"  Downsampled to {len(idx_sel):,}")

    n_stack = len(idx_sel)
    z_med   = float(np.median(z_arr[idx_sel]))

    # ── Shift & normalise each spectrum ──────────────────────────────────────
    norm_cube = np.full((N_REST, n_stack), np.nan, dtype=np.float32)
    err_cube  = np.full((N_REST, n_stack), np.nan, dtype=np.float32)
    norm_factors = np.zeros(n_stack, dtype=np.float64)

    n_good = 0
    for j, src_idx in enumerate(idx_sel):
        z_src    = float(z_arr[src_idx])
        flux_src = spec_2d[:, src_idx].astype(float)
        err_src  = np.clip(err_2d[:, src_idx].astype(float), 1e-6, None)

        # Replace bad values
        flux_src[~np.isfinite(flux_src)] = np.nan
        err_src[~np.isfinite(err_src)]   = np.nan

        # Shift to rest-frame
        flux_rest, err_rest = shift_to_restframe(
            flux_src, err_src, wave_obs, z_src)

        # Normalise to continuum window
        flux_norm, nfac = normalise_spectrum(flux_rest, rest_wave)
        err_norm        = err_rest / nfac if nfac > 0 else err_rest

        norm_cube[:, j] = flux_norm.astype(np.float32)
        err_cube[:, j]  = err_norm.astype(np.float32)
        norm_factors[j] = nfac
        if np.isfinite(flux_norm).sum() > 20:
            n_good += 1

    print(f"  Spectra with valid continuum: {n_good:,} / {n_stack:,}")

    # ── Median stack ─────────────────────────────────────────────────────────
    stack_flux = np.nanmedian(norm_cube, axis=1)
    n_contrib  = np.sum(np.isfinite(norm_cube), axis=1)

    # Uncertainty: MAD-based  σ = 1.4826 * MAD / sqrt(N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mad_per_pix = median_abs_deviation(
            norm_cube, axis=1, nan_policy="omit", scale=1.0)
    stack_err = 1.4826 * mad_per_pix / np.sqrt(np.maximum(n_contrib, 1))

    # ── Bootstrap uncertainty (optional) ─────────────────────────────────────
    if N_BOOTSTRAP > 0 and n_good >= 5:
        rng_bs   = np.random.default_rng(42)
        bs_stacks= np.zeros((N_BOOTSTRAP, N_REST), dtype=np.float32)
        for b in range(N_BOOTSTRAP):
            bs_idx       = rng_bs.integers(0, n_stack, n_stack)
            bs_stacks[b] = np.nanmedian(norm_cube[:, bs_idx], axis=1)
        stack_err_bs = np.nanstd(bs_stacks, axis=0)
        # Use the larger of the two error estimates
        stack_err = np.maximum(stack_err, stack_err_bs)
        print(f"  Bootstrap ({N_BOOTSTRAP} resamples) complete.")

    # ── Fit Lyα ──────────────────────────────────────────────────────────────
    lya_fit = fit_lya(rest_wave, stack_flux, err=stack_err)
    if lya_fit:
        print(f"  Lya fit:  cen={lya_fit['cen']:.2f} AA  "
              f"FWHM={lya_fit['fwhm']:.1f} AA  "
              f"EW_rest={lya_fit['ew_rest']:.1f} AA")
    else:
        print("  Lya fit failed.")

    stacks.append({
        "z_lo"      : z_lo,
        "z_hi"      : z_hi,
        "z_med"     : z_med,
        "n_stack"   : n_stack,
        "n_good"    : n_good,
        "flux"      : stack_flux,
        "err"       : stack_err,
        "n_contrib" : n_contrib,
        "lya_fit"   : lya_fit,
    })

# =============================================================================
# CELL 7 — SAVE STACKED SPECTRA TO CSV
# =============================================================================

import pandas as pd
rows = []
for st in stacks:
    if st is None:
        continue
    for i, (w, f, e, nc) in enumerate(
            zip(rest_wave, st["flux"], st["err"], st["n_contrib"])):
        rows.append({
            "z_lo"     : st["z_lo"],
            "z_hi"     : st["z_hi"],
            "wave_rest": round(w, 3),
            "flux_norm": round(float(f), 6) if np.isfinite(f) else None,
            "err_norm" : round(float(e), 6) if np.isfinite(e) else None,
            "n_contrib": int(nc),
        })
if rows and CSV_PATH:
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"\nStacked spectra saved -> {CSV_PATH}  ({len(rows):,} rows)")

# =============================================================================
# CELL 8 — PLOTTING
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

# Colours per z-bin
BIN_COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]

# Rest-frame emission lines to mark
UV_LINES_MARK = {
    r"Ly$\beta$"  : 1025.7,
    r"Ly$\alpha$" : 1215.7,
    "N V"         : 1240.8,
    "O I"         : 1302.2,
    "C II"        : 1335.3,
    "Si IV"       : 1393.8,
    "C IV"        : 1549.5,
    "He II"       : 1640.4,
    "C III]"      : 1908.7,
}

n_bins_valid = sum(1 for s in stacks if s is not None)
if n_bins_valid == 0:
    print("No valid stacks to plot.")
else:
    fig = plt.figure(figsize=(17, 5 * n_bins_valid + 4))
    fig.patch.set_facecolor(BG)

    gs_outer = gridspec.GridSpec(
        n_bins_valid + 1, 1, figure=fig,
        hspace=0.45,
        left=0.07, right=0.97,
        top=0.94,  bottom=0.05,
        height_ratios=[2.5] * n_bins_valid + [1.2],
    )

    valid_stacks = [s for s in stacks if s is not None]

    # ── One stacked spectrum per z-bin ────────────────────────────────────────
    ax_prev = None
    for panel_idx, (st, color) in enumerate(
            zip(valid_stacks, BIN_COLORS)):

        ax = fig.add_subplot(gs_outer[panel_idx],
                             sharey=ax_prev if panel_idx > 0 else None)
        ax.set_facecolor(AX_BG)
        for sp in ax.spines.values():
            sp.set_color(SPINE)
        ax.tick_params(colors=MUTED, which="both", direction="in",
                       top=True, right=True, labelsize=9)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        flux = st["flux"]
        err  = st["err"]
        nc   = st["n_contrib"]

        # Only plot pixels with enough contributors
        MIN_CONTRIB = max(5, st["n_good"] // 20)
        good_pix = nc >= MIN_CONTRIB

        # Error shading
        ax.fill_between(
            rest_wave[good_pix],
            (flux - err)[good_pix],
            (flux + err)[good_pix],
            color=color, alpha=0.18, zorder=2,
        )
        # Raw stacked spectrum (thin)
        ax.plot(rest_wave[good_pix], flux[good_pix],
                color=color, lw=0.7, alpha=0.50, zorder=3)
        # Smoothed (5-pixel boxcar)
        kernel = np.ones(5) / 5.0
        flux_smooth = np.convolve(
            np.where(good_pix, flux, np.nan), kernel, mode="same")
        ax.plot(rest_wave[good_pix], flux_smooth[good_pix],
                color=TEXT, lw=1.6, alpha=0.92, zorder=4,
                label="Smoothed median stack")

        # Gaussian Lya fit
        lf = st["lya_fit"]
        if lf is not None:
            x_fit  = np.linspace(LYA_AA - 50, LYA_AA + 50, 300)
            y_fit  = gaussian(x_fit, lf["amp"], lf["cen"],
                              lf["sigma"], lf["cont"])
            ax.plot(x_fit, y_fit, "--",
                    color="#ffa657", lw=1.8, alpha=0.85, zorder=5,
                    label=(rf"Gaussian fit: FWHM={lf['fwhm']:.1f} Å  "
                           rf"EW$_{{rest}}$={lf['ew_rest']:.0f} Å"))

        # Continuum normalisation window
        ax.axvspan(NORM_WAVE_MIN, NORM_WAVE_MAX,
                   color="#3fb950", alpha=0.05, zorder=1)
        ax.axhline(1.0, color=MUTED, lw=0.8, ls=":", alpha=0.60)
        ax.axhline(0.0, color=SPINE, lw=0.7, ls=":")

        # Emission line markers
        ylims = ax.get_ylim()
        y_lo, y_hi = -0.3, 4.0  # fixed scale for comparability
        for lname, lwave in UV_LINES_MARK.items():
            if REST_WAVE_MIN < lwave < REST_WAVE_MAX:
                ax.axvline(lwave, color="#d2a8ff",
                           lw=0.7, ls="--", alpha=0.55, zorder=2)
                ax.text(lwave + 2, y_hi * 0.88,
                        lname, color="#d2a8ff",
                        fontsize=7.0, rotation=90,
                        va="top", ha="left", alpha=0.80)

        ax.set_xlim(REST_WAVE_MIN, REST_WAVE_MAX)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("Normalised flux", color=TEXT, fontsize=10)
        if panel_idx == n_bins_valid - 1:
            ax.set_xlabel(r"Rest-frame wavelength  $\lambda_{\rm rest}$  (Å)",
                          color=TEXT, fontsize=10)

        # Title with key statistics
        fit_str = (f"   |   Lya FWHM={lf['fwhm']:.1f} Å"
                   rf"   EW$_\mathrm{{rest}}$={lf['ew_rest']:.0f} Å"
                   if lf else "")
        ax.set_title(
            (f"z = [{st['z_lo']}, {st['z_hi']})   "
             f"z_med = {st['z_med']:.3f}   "
             f"N = {st['n_stack']:,}   "
             f"N_good = {st['n_good']:,}{fit_str}"),
            color=TEXT, fontsize=10, fontweight="bold",
            loc="left", pad=6,
        )

        ax.legend(fontsize=8.5, facecolor="#21262d",
                  edgecolor=SPINE, labelcolor=TEXT,
                  loc="upper right")

        # N_contributing pixels (right y-axis)
        ax_r = ax.twinx()
        ax_r.fill_between(rest_wave[good_pix],
                          0, nc[good_pix],
                          color=color, alpha=0.08, step="mid")
        ax_r.set_ylabel("N contributing", color=MUTED, fontsize=8)
        ax_r.tick_params(colors=MUTED, labelsize=7)
        ax_r.spines["right"].set_color(SPINE)
        ax_r.set_ylim(0, st["n_good"] * 1.3)

        ax_prev = ax

    # ── Bottom panel: overlay all bins ───────────────────────────────────────
    ax_all = fig.add_subplot(gs_outer[n_bins_valid])
    ax_all.set_facecolor(AX_BG)
    for sp in ax_all.spines.values():
        sp.set_color(SPINE)
    ax_all.tick_params(colors=MUTED, which="both", direction="in",
                       top=True, right=True, labelsize=9)
    ax_all.xaxis.set_minor_locator(AutoMinorLocator())
    ax_all.yaxis.set_minor_locator(AutoMinorLocator())

    for st, color in zip(valid_stacks, BIN_COLORS):
        nc   = st["n_contrib"]
        good = nc >= max(5, st["n_good"] // 20)
        flux_sm = np.convolve(
            np.where(good, st["flux"], np.nan),
            np.ones(5)/5, mode="same")
        label = (f"z=[{st['z_lo']},{st['z_hi']})  "
                 f"N={st['n_good']:,}")
        ax_all.plot(rest_wave[good], flux_sm[good],
                    color=color, lw=1.5, alpha=0.88, label=label)

    for lname, lwave in UV_LINES_MARK.items():
        if REST_WAVE_MIN < lwave < REST_WAVE_MAX:
            ax_all.axvline(lwave, color="#d2a8ff",
                           lw=0.6, ls="--", alpha=0.40)
            ax_all.text(lwave + 2, 3.5, lname,
                        color="#d2a8ff", fontsize=6.5,
                        rotation=90, va="top", ha="left", alpha=0.70)

    ax_all.axhline(1.0, color=MUTED, lw=0.7, ls=":", alpha=0.50)
    ax_all.axhline(0.0, color=SPINE, lw=0.6, ls=":")
    ax_all.set_xlim(REST_WAVE_MIN, REST_WAVE_MAX)
    ax_all.set_ylim(-0.3, 4.2)
    ax_all.set_xlabel(
        r"Rest-frame wavelength  $\lambda_{\rm rest}$  (Å)",
        color=TEXT, fontsize=10)
    ax_all.set_ylabel("Normalised flux", color=TEXT, fontsize=10)
    ax_all.set_title("All z-bins overlaid",
                     color=TEXT, fontsize=10, fontweight="bold",
                     loc="left", pad=6)
    ax_all.legend(fontsize=8.5, facecolor="#21262d",
                  edgecolor=SPINE, labelcolor=TEXT,
                  loc="upper right", ncol=len(valid_stacks))

    syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
    fig.suptitle(
        rf"HETDEX SC2 — Stacked LAE Rest-frame Spectra{syn_tag}",
        color=TEXT, fontsize=13, fontweight="bold", y=0.975,
    )

    if SAVE_PATH:
        fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight",
                    facecolor=BG)
        print(f"\nSaved -> {SAVE_PATH}")
    plt.show()

# =============================================================================
# CELL 9 — NUMERICAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("  HETDEX SC2 — LAE Stacked Spectrum Summary")
print("=" * 70)
for st in stacks:
    if st is None:
        continue
    lf    = st["lya_fit"]
    print(f"\n  z = [{st['z_lo']}, {st['z_hi']})   "
          f"z_med = {st['z_med']:.3f}   "
          f"N_stack = {st['n_stack']:,}   N_good = {st['n_good']:,}")
    # Peak S/N of stacked Lya line
    lya_win = (rest_wave > LYA_AA - 10) & (rest_wave < LYA_AA + 10)
    if lya_win.sum() > 0:
        peak_flux = float(np.nanmax(st["flux"][lya_win]))
        med_err   = float(np.nanmedian(st["err"][lya_win]))
        sn_stack  = peak_flux / med_err if med_err > 0 else np.nan
        print(f"    Lya peak S/N in stack : {sn_stack:.1f}")
    if lf:
        print(f"    Lya centroid          : {lf['cen']:.2f} +/- "
              f"{lf['cen_err']:.2f} AA")
        print(f"    Lya FWHM              : {lf['fwhm']:.2f} AA  "
              f"(sigma={lf['sigma']:.2f} AA)")
        print(f"    EW_rest               : {lf['ew_rest']:.1f} AA")
    else:
        print("    Lya Gaussian fit      : failed")
    print(f"    N contributing @ Lya  : "
          f"{int(st['n_contrib'][np.argmin(abs(rest_wave-LYA_AA))]):,}")
print("=" * 70)
