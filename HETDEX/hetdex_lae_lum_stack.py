"""
hetdex_lae_lum_stack.py
=======================
Stack HETDEX SC2 LAE spectra in bins of logL_lya WITHIN fixed redshift
slices, separating the luminosity dependence of Lyα EW from redshift
evolution.

Design
------
For each z-slice we divide the LAE population into N_L_BINS luminosity
quartiles (or custom edges).  Within each (z, L) cell we:
  1. Shift spectra to rest-frame (with (1+z) flux correction).
  2. Normalise to the 1260–1350 Å continuum.
  3. Median-stack with bootstrap uncertainty.
  4. Fit a Gaussian to the Lyα line → EW_rest, FWHM, centroid.
  5. Fit a UV power-law β to the stacked continuum (1300–1500 Å).

The key figure is a 2D grid: rows = z-slices, columns = logL bins.
Below the grid, summary panels show EW_rest(L) per z-slice and
β(L) per z-slice — the two quantities that diagnose whether EW
evolution is driven by luminosity selection or genuine redshift evolution.

Physical motivation
-------------------
The EW–L anti-correlation (brighter LAEs have smaller EW) is well-known
from narrow-band surveys (e.g. Ando+06, Ouchi+08).  HETDEX's blind
spectroscopy and large N lets us measure this at fixed z, free from
photometric pre-selection.  Any residual EW(z) trend AFTER controlling
for L is genuine redshift evolution of the Lyα escape fraction or CGM
opacity.

Columns used
------------
  source_type, z_hetdex, logL_lya, p_conf, p_cnn, sn

Requirements
------------
  pip install astropy numpy matplotlib scipy pandas

Data
----
  hetdex_sc2_spec_v1.5.fits
  https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/hetdex_source_catalog_2/
"""

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

SPEC_PATH  = "hetdex_sc2_spec_v1.5.fits"
SAVE_PATH  = "hetdex_lae_lum_stack.png"
CSV_PATH   = "hetdex_lae_lum_stack.csv"

# Fixed redshift slices (kept narrow to minimise z-evolution within slice)
Z_SLICES = [
    (2.0, 2.4),
    (2.4, 2.8),
    (2.8, 3.2),
]

# Luminosity binning within each z-slice
# Set L_BIN_EDGES to None to use data-driven quartiles, or give explicit edges
L_BIN_EDGES = [42.0, 42.5, 43.0, 43.5, 44.5]   # log10(L_lya / erg/s)
# If None, N_L_BINS equal-count (quartile) bins are used per z-slice
N_L_BINS   = 4    # used only when L_BIN_EDGES is None

# Quality cuts
MIN_SN     = 5.5
MIN_P_CONF = 0.5
MIN_P_CNN  = 0.5
MAX_PER_BIN = 3000   # cap per (z, L) cell; None = use all

# Rest-frame grid
REST_WAVE_MIN  = 1050.0
REST_WAVE_MAX  = 1700.0
REST_WAVE_STEP = 1.0

# Continuum normalisation window (rest AA, red of Lya forest, blue of CIV)
NORM_WIN = (1260.0, 1350.0)

# UV slope β fit window (rest AA)
BETA_WIN = (1300.0, 1500.0)

# Bootstrap
N_BOOTSTRAP = 150

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
from matplotlib.ticker  import AutoMinorLocator, LogLocator
from matplotlib.lines   import Line2D
from scipy.interpolate  import interp1d
from scipy.optimize     import curve_fit
from scipy.stats        import median_abs_deviation
import pandas as pd

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

def make_synthetic_spec(n_sources=8000, seed=31):
    """
    Synthetic spec FITS — same structure as make_synthetic_spec() in
    hetdex_lae_stack.py but with a realistic EW–L anti-correlation built in:
        EW_rest ~ 300 * (L / L_pivot)^{-0.5}   (Ando+06 scaling)
    so the stacking can recover it.
    """
    rng    = np.random.default_rng(seed)
    N_PIX  = 1036
    wave_obs = np.linspace(3470.0, 5540.0, N_PIX)

    z_arr  = rng.uniform(1.95, 3.45, n_sources).astype(np.float32)
    logL   = rng.uniform(41.8, 44.2, n_sources).astype(np.float32)
    sn_arr = np.abs(rng.lognormal(2.1, 0.5, n_sources)).astype(np.float32)
    p_conf = np.clip(rng.beta(4, 1.5, n_sources), 0, 1).astype(np.float32)
    p_cnn  = np.clip(rng.beta(3.5, 1.3, n_sources), 0, 1).astype(np.float32)
    field  = rng.choice(["dex-spring","dex-fall","cosmos","goods-n"],
                         n_sources, p=[0.55,0.30,0.10,0.05])

    # EW–L anti-correlation: EW_rest = 250 * 10^{-0.4*(logL-42.5)} AA
    L_pivot = 42.5
    ew_true = 250.0 * 10.0**(-0.4 * (logL - L_pivot))
    ew_true = np.clip(ew_true, 10, 500)

    # UV continuum slope β: steeper (bluer) at lower L
    beta_true = -1.2 - 0.6 * (logL - L_pivot) / 2.0
    beta_true = np.clip(beta_true, -3, -0.5)

    spec_arr = np.zeros((N_PIX, n_sources), dtype=np.float32)
    err_arr  = np.zeros((N_PIX, n_sources), dtype=np.float32)

    UV_LINES = [
        (1215.67, None, 5.0),   # Lya — amplitude from EW below
        (1240.81, 0.05, 2.5),   # NV
        (1302.17, 0.03, 2.0),   # OI
        (1335.31, 0.04, 2.0),   # CII
        (1393.80, 0.03, 2.0),   # SiIV
        (1549.48, 0.10, 3.5),   # CIV
        (1640.40, 0.05, 2.5),   # HeII
        (1908.73, 0.08, 3.0),   # CIII]
    ]

    for i in range(n_sources):
        z_i   = float(z_arr[i])
        lL    = float(logL[i])
        scale = 10.0**((lL - 43.0) * 0.4) * 0.4
        beta  = float(beta_true[i])

        # Power-law continuum with UV slope beta
        cont = scale * (wave_obs / 4500.0)**beta
        cont = np.clip(cont, 0, None)
        spec = cont.copy()

        # Lya: set amplitude from EW_rest
        # EW_obs = EW_rest * (1+z);  EW = integral/cont_at_lya * sigma*sqrt(2pi)
        # amp_lya = EW_rest * cont_at_lya_rest / (sigma_rest * sqrt(2pi))
        lya_obs  = LYA_AA * (1.0 + z_i)
        cont_lya = float(scale * (lya_obs / 4500.0)**beta)
        ew_obs   = float(ew_true[i]) * (1.0 + z_i)
        sig_obs  = float(UV_LINES[0][2]) * (1.0 + z_i)
        amp_lya  = ew_obs * cont_lya / (sig_obs * np.sqrt(2 * np.pi))

        for w_rest, amp_fac, sigma_aa in UV_LINES:
            w_obs_i = w_rest * (1.0 + z_i)
            if not (wave_obs[0] < w_obs_i < wave_obs[-1]):
                continue
            if w_rest == LYA_AA:
                amp   = amp_lya * rng.lognormal(0, 0.15)
                sigma = sig_obs
            else:
                amp   = amp_fac * scale * rng.lognormal(0, 0.3)
                sigma = sigma_aa * (1.0 + z_i)
            spec += amp * np.exp(-0.5 * ((wave_obs - w_obs_i) / sigma)**2)

        # Lyman forest absorption blueward of Lya
        lya_obs_i = LYA_AA * (1 + z_i)
        forest_mask = wave_obs < lya_obs_i - 5
        tau_eff = 0.0037 * (1 + z_i)**3.2    # Madau 1995
        spec[forest_mask] *= np.exp(-tau_eff)

        # Noise
        noise_lev = np.abs(cont) * rng.uniform(0.12, 0.30) + 0.005 * scale
        noise_lev = np.clip(noise_lev, 1e-4, None)
        spec     += rng.normal(0, noise_lev)

        spec_arr[:, i] = spec.astype(np.float32)
        err_arr[:, i]  = noise_lev.astype(np.float32)

    info = Table({
        "source_id"  : np.arange(n_sources, dtype=np.int64),
        "source_type": np.where(z_arr > 1.87, "lae", "oii"),
        "z_hetdex"   : z_arr,
        "logl_lya"   : logL,
        "sn"         : sn_arr,
        "p_conf"     : p_conf,
        "p_cnn"      : p_cnn,
        "field"      : field,
    })
    print(f"  Synthetic: {n_sources:,} sources, EW range "
          f"{ew_true.min():.0f}–{ew_true.max():.0f} AA (EW~L^-0.5 built in)")
    return info, spec_arr, err_arr, wave_obs.astype(np.float32)


# =============================================================================
# CELL 4 — LOAD SPECTRAL FITS
# =============================================================================

def load_spec_fits(path):
    try:
        hdul = fits.open(path, memmap=True)
        info = Table(hdul[1].data)
        info.rename_columns(info.colnames,
                            [c.lower() for c in info.colnames])
        spec_raw = np.array(hdul[2].data, dtype=np.float32)
        err_raw  = np.array(hdul[3].data, dtype=np.float32)
        wave_raw = np.array(hdul[4].data, dtype=np.float32).ravel()
        hdul.close()
        n_src, n_pix = len(info), len(wave_raw)

        def orient(arr, lbl):
            if arr.shape == (n_pix, n_src):  return arr
            if arr.shape == (n_src, n_pix):
                print(f"  {lbl}: transposing {arr.shape}")
                return arr.T
            raise ValueError(f"{lbl} shape {arr.shape} unrecognised")

        synthetic = False
        spec_2d = orient(spec_raw, "SPEC")
        err_2d  = orient(err_raw,  "SPEC_ERR")
        print(f"Loaded {path}: {n_src:,} sources, {n_pix} pixels, "
              f"{wave_raw[0]:.0f}–{wave_raw[-1]:.0f} AA")
    except FileNotFoundError:
        print(f"'{path}' not found — using synthetic demo data.")
        info, spec_2d, err_2d, wave_raw = make_synthetic_spec()
        synthetic = True
    return info, spec_2d, err_2d, wave_raw, synthetic


info, spec_2d, err_2d, wave_obs, SYNTHETIC = load_spec_fits(SPEC_PATH)

# =============================================================================
# CELL 5 — SHARED HELPERS (identical to hetdex_lae_stack.py)
# =============================================================================

def getcol(tab, *cands):
    lc = {c.lower(): c for c in tab.colnames}
    for c in cands:
        if c.lower() in lc:
            return lc[c.lower()]
    raise KeyError(f"None of {cands} in table. Have: {list(tab.colnames)[:25]}")

rest_wave = np.arange(REST_WAVE_MIN, REST_WAVE_MAX + REST_WAVE_STEP,
                      REST_WAVE_STEP)
N_REST = len(rest_wave)


def shift_to_restframe(flux_obs, err_obs, wave_obs_arr, z):
    wave_rest_src = wave_obs_arr / (1.0 + z)
    flux_factor   = 1.0 + z
    in_range = ((rest_wave >= wave_rest_src[0]) &
                (rest_wave <= wave_rest_src[-1]))
    flux_out = np.full(N_REST, np.nan)
    err_out  = np.full(N_REST, np.nan)
    if in_range.sum() < 5:
        return flux_out, err_out
    f_itp = interp1d(wave_rest_src, flux_obs, kind="linear",
                     bounds_error=False, fill_value=np.nan)
    e_itp = interp1d(wave_rest_src, err_obs,  kind="linear",
                     bounds_error=False, fill_value=np.nan)
    flux_out[in_range] = f_itp(rest_wave[in_range]) * flux_factor
    err_out[in_range]  = e_itp(rest_wave[in_range]) * flux_factor
    return flux_out, err_out


def normalise_spectrum(flux, wave, wmin=NORM_WIN[0], wmax=NORM_WIN[1]):
    win = (wave >= wmin) & (wave <= wmax)
    fv  = flux[win]
    fv  = fv[np.isfinite(fv)]
    if len(fv) < 3:
        return flux, 1.0
    med = float(np.nanmedian(fv))
    if not np.isfinite(med) or med <= 0:
        return flux, 1.0
    return flux / med, med


def gaussian(x, amp, cen, sigma, offset):
    return amp * np.exp(-0.5 * ((x - cen) / sigma)**2) + offset


def fit_lya(wave, flux, err=None, win_aa=40.0):
    mask = (wave >= LYA_AA - win_aa) & (wave <= LYA_AA + win_aa) & np.isfinite(flux)
    if mask.sum() < 8:
        return None
    x, y = wave[mask], flux[mask]
    sig  = err[mask] if err is not None else None
    flank = ((wave >= LYA_AA - win_aa) & (wave < LYA_AA - 10)) | \
            ((wave > LYA_AA + 20)       & (wave <= LYA_AA + win_aa))
    cont0 = float(np.nanmedian(flux[flank & np.isfinite(flux)])) \
            if flank.sum() > 2 else 0.0
    amp0  = float(np.nanmax(y)) - cont0
    try:
        p0  = [amp0, LYA_AA, 5.0, cont0]
        bds = ([0, LYA_AA-15, 0.5, -np.inf],
               [np.inf, LYA_AA+15, 30, np.inf])
        popt, pcov = curve_fit(gaussian, x, y, p0=p0,
                               sigma=sig, absolute_sigma=(sig is not None),
                               bounds=bds, maxfev=4000)
        perr = np.sqrt(np.diag(pcov))
        fwhm = 2.355 * abs(popt[2])
        integral = popt[0] * abs(popt[2]) * np.sqrt(2 * np.pi)
        ew  = integral / max(abs(popt[3]), 1e-10)
        return {"amp": popt[0], "amp_err": perr[0],
                "cen": popt[1], "cen_err": perr[1],
                "sigma": popt[2], "sigma_err": perr[2],
                "fwhm": fwhm, "ew_rest": ew, "cont": popt[3]}
    except Exception:
        return None


def fit_beta(wave, flux, err=None, wmin=BETA_WIN[0], wmax=BETA_WIN[1]):
    """
    Fit UV power-law:  f_lambda ∝ lambda^beta
    Returns (beta, beta_err) or (nan, nan) on failure.
    """
    mask = (wave >= wmin) & (wave <= wmax) & np.isfinite(flux) & (flux > 0)
    if mask.sum() < 10:
        return np.nan, np.nan
    x = np.log10(wave[mask])
    y = np.log10(flux[mask])
    w = (1.0 / err[mask]**2) if err is not None else None
    try:
        if w is not None:
            w = np.where(np.isfinite(w), w, 0)
            coeffs, cov = np.polyfit(x, y, 1, w=w, cov=True)
        else:
            coeffs, cov = np.polyfit(x, y, 1, cov=True)
        beta     = coeffs[0]
        beta_err = float(np.sqrt(cov[0, 0]))
        return float(beta), beta_err
    except Exception:
        return np.nan, np.nan


# =============================================================================
# CELL 6 — BUILD PARENT LAE SELECTION
# =============================================================================

STYPE_COL = getcol(info, "source_type")
Z_COL     = getcol(info, "z_hetdex")
SN_COL    = getcol(info, "sn")
PCONF_COL = getcol(info, "p_conf")
PCNN_COL  = getcol(info, "p_cnn")
LOGL_COL  = getcol(info, "logl_lya", "logl_lya")

z_arr    = np.array(info[Z_COL],    dtype=float)
stype_arr= np.array([s.strip().lower()
                     for s in info[STYPE_COL]], dtype=str)
sn_arr   = np.array(info[SN_COL],   dtype=float)
pconf_arr= np.array(info[PCONF_COL],dtype=float)
pcnn_arr = np.array(info[PCNN_COL], dtype=float)
logL_arr = np.array(info[LOGL_COL], dtype=float)
logL_arr[logL_arr == BAD] = np.nan

# Base LAE quality mask (applied before z/L splitting)
base_mask = (
    (stype_arr == "lae") &
    (sn_arr >= MIN_SN)   & (sn_arr != BAD) &
    (pconf_arr >= MIN_P_CONF) &
    (pcnn_arr  >= MIN_P_CNN)  &
    np.isfinite(z_arr) &
    np.isfinite(logL_arr)
)
print(f"\nBase LAE selection: {base_mask.sum():,} sources")
print(f"logL_lya range: {logL_arr[base_mask].min():.2f} – "
      f"{logL_arr[base_mask].max():.2f}")

# Determine luminosity bin edges (shared across z-slices for comparability)
if L_BIN_EDGES is not None:
    l_edges = np.array(L_BIN_EDGES)
else:
    # Data-driven quartiles from the full LAE population
    l_edges = np.nanpercentile(
        logL_arr[base_mask],
        np.linspace(0, 100, N_L_BINS + 1)
    )
    l_edges[0]  -= 0.01
    l_edges[-1] += 0.01

l_bin_centres = 0.5 * (l_edges[:-1] + l_edges[1:])
N_L           = len(l_edges) - 1
print(f"Luminosity bin edges: {l_edges}")
print(f"N_L = {N_L} bins")


# =============================================================================
# CELL 7 — STACKING ENGINE
# =============================================================================

def build_stack(idx_src, label=""):
    """
    Stack all spectra at indices idx_src.
    Returns dict with flux, err, n_contrib, lya_fit, beta, n_good, z_med, logL_med.
    """
    n = len(idx_src)
    if n == 0:
        return None

    # Downsample if needed
    if MAX_PER_BIN is not None and n > MAX_PER_BIN:
        rng_sub = np.random.default_rng(n)
        idx_src = rng_sub.choice(idx_src, size=MAX_PER_BIN, replace=False)
        n = len(idx_src)

    norm_cube = np.full((N_REST, n), np.nan, dtype=np.float32)
    err_cube  = np.full((N_REST, n), np.nan, dtype=np.float32)
    n_good    = 0

    for j, si in enumerate(idx_src):
        z_s   = float(z_arr[si])
        f_s   = spec_2d[:, si].astype(float)
        e_s   = np.clip(err_2d[:, si].astype(float), 1e-6, None)
        f_s[~np.isfinite(f_s)] = np.nan

        fr, er = shift_to_restframe(f_s, e_s, wave_obs, z_s)
        fn, nf = normalise_spectrum(fr, rest_wave)
        en     = er / nf if nf > 0 else er

        norm_cube[:, j] = fn.astype(np.float32)
        err_cube[:, j]  = en.astype(np.float32)
        if np.isfinite(fn).sum() > 20:
            n_good += 1

    if n_good < 3:
        return None

    stack_f    = np.nanmedian(norm_cube, axis=1)
    n_contrib  = np.sum(np.isfinite(norm_cube), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mad = median_abs_deviation(norm_cube, axis=1,
                                   nan_policy="omit", scale=1.0)
    stack_e = 1.4826 * mad / np.sqrt(np.maximum(n_contrib, 1))

    # Bootstrap
    if N_BOOTSTRAP > 0 and n_good >= 5:
        rng_bs = np.random.default_rng(42)
        bs_arr = np.zeros((N_BOOTSTRAP, N_REST), dtype=np.float32)
        for b in range(N_BOOTSTRAP):
            bi         = rng_bs.integers(0, n, n)
            bs_arr[b]  = np.nanmedian(norm_cube[:, bi], axis=1)
        stack_e = np.maximum(stack_e, np.nanstd(bs_arr, axis=0))

    lya_fit = fit_lya(rest_wave, stack_f, err=stack_e)
    beta_v, beta_e = fit_beta(rest_wave, stack_f)

    return {
        "flux"      : stack_f,
        "err"       : stack_e,
        "n_contrib" : n_contrib,
        "lya_fit"   : lya_fit,
        "beta"      : beta_v,
        "beta_err"  : beta_e,
        "n_stack"   : n,
        "n_good"    : n_good,
        "z_med"     : float(np.median(z_arr[idx_src])),
        "logL_med"  : float(np.median(logL_arr[idx_src])),
        "logL_lo"   : float(np.min(logL_arr[idx_src])),
        "logL_hi"   : float(np.max(logL_arr[idx_src])),
    }


# ── Run grid: (z_slice) × (L_bin) ─────────────────────────────────────────────
print("\nBuilding (z, L) stacking grid ...")
grid = []   # grid[iz][il] = stack dict or None

for iz, (z_lo, z_hi) in enumerate(Z_SLICES):
    row = []
    for il in range(N_L):
        l_lo, l_hi = l_edges[il], l_edges[il + 1]
        cell_mask  = (base_mask &
                      (z_arr    >= z_lo) & (z_arr    < z_hi) &
                      (logL_arr >= l_lo) & (logL_arr < l_hi))
        idx = np.where(cell_mask)[0]
        label = f"z=[{z_lo},{z_hi})  logL=[{l_lo:.2f},{l_hi:.2f})"
        print(f"  {label}: {len(idx):,}", end="  ")
        st = build_stack(idx, label=label)
        if st and st["lya_fit"]:
            lf = st["lya_fit"]
            print(f"EW={lf['ew_rest']:.0f}AA  β={st['beta']:.2f}")
        elif st:
            print(f"stack OK, Lya fit failed  β={st['beta']:.2f}")
        else:
            print("SKIP (too few)")
        row.append(st)
    grid.append(row)

# =============================================================================
# CELL 8 — SAVE CSV
# =============================================================================

rows_csv = []
for iz, (z_lo, z_hi) in enumerate(Z_SLICES):
    for il in range(N_L):
        st = grid[iz][il]
        if st is None:
            continue
        lf = st["lya_fit"]
        for i_w, (w, f, e, nc) in enumerate(
                zip(rest_wave, st["flux"], st["err"], st["n_contrib"])):
            rows_csv.append({
                "z_lo"     : z_lo, "z_hi": z_hi,
                "logL_lo"  : round(l_edges[il], 3),
                "logL_hi"  : round(l_edges[il+1], 3),
                "logL_med" : round(st["logL_med"], 3),
                "z_med"    : round(st["z_med"],  4),
                "wave_rest": round(w, 2),
                "flux_norm": round(float(f), 6) if np.isfinite(f) else None,
                "err_norm" : round(float(e), 6) if np.isfinite(e) else None,
                "n_contrib": int(nc),
            })
if rows_csv and CSV_PATH:
    pd.DataFrame(rows_csv).to_csv(CSV_PATH, index=False)
    print(f"\nCSV saved -> {CSV_PATH}  ({len(rows_csv):,} rows)")

# =============================================================================
# CELL 9 — PLOTTING
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

# Colour ramp for luminosity bins (faint=cool, bright=warm)
L_CMAP  = plt.cm.plasma
l_cols  = [L_CMAP(0.15 + 0.70 * il / max(N_L - 1, 1)) for il in range(N_L)]

# UV line markers
UV_LINES_MARK = {
    r"Ly$\alpha$": 1215.7, "N V": 1240.8,  "O I": 1302.2,
    "C II": 1335.3, "Si IV": 1393.8, "C IV": 1549.5,
    "He II": 1640.4,
}

N_Z = len(Z_SLICES)

# Layout: N_Z rows of stacked panels + 2 summary rows
fig = plt.figure(figsize=(5 * N_L, 4.5 * N_Z + 5.5))
fig.patch.set_facecolor(BG)

gs_top = gridspec.GridSpec(
    N_Z, N_L, figure=fig,
    hspace=0.42, wspace=0.18,
    left=0.06, right=0.97,
    top=0.93,  bottom=0.30,
)
gs_bot = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.42, wspace=0.30,
    left=0.06, right=0.97,
    top=0.27,  bottom=0.04,
)

ax_ew_l   = [fig.add_subplot(gs_bot[0, iz]) for iz in range(N_Z)]
ax_beta_l = [fig.add_subplot(gs_bot[1, iz]) for iz in range(N_Z)]

def style_ax(ax, title, xl, yl):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, which="both", direction="in",
                   top=True, right=True, labelsize=8)
    ax.set_xlabel(xl, color=TEXT, fontsize=9)
    ax.set_ylabel(yl, color=TEXT, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=9,
                 fontweight="bold", loc="left", pad=5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def mleg(ax, **kw):
    return ax.legend(fontsize=7.5, facecolor="#21262d",
                     edgecolor=SPINE, labelcolor=TEXT, **kw)

# ── Grid panels: one per (z, L) cell ──────────────────────────────────────────
for iz, (z_lo, z_hi) in enumerate(Z_SLICES):
    for il in range(N_L):
        st    = grid[iz][il]
        color = l_cols[il]
        ax    = fig.add_subplot(gs_top[iz, il])
        ax.set_facecolor(AX_BG)
        for sp in ax.spines.values():
            sp.set_color(SPINE)
        ax.tick_params(colors=MUTED, which="both", direction="in",
                       top=True, right=True, labelsize=7)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        y_lo_ax, y_hi_ax = -0.3, 4.5

        if st is None or st["n_good"] < 3:
            ax.text(0.5, 0.5, "insufficient\ndata",
                    ha="center", va="center",
                    transform=ax.transAxes, color=MUTED, fontsize=8)
        else:
            nc   = st["n_contrib"]
            MIN_C = max(3, st["n_good"] // 20)
            good = nc >= MIN_C
            flux_sm = np.convolve(
                np.where(good, st["flux"], np.nan),
                np.ones(5) / 5, mode="same")

            # Error shading
            ax.fill_between(
                rest_wave[good],
                (st["flux"] - st["err"])[good],
                (st["flux"] + st["err"])[good],
                color=color, alpha=0.18, zorder=2)
            # Raw stack
            ax.plot(rest_wave[good], st["flux"][good],
                    color=color, lw=0.5, alpha=0.40, zorder=3)
            # Smoothed
            ax.plot(rest_wave[good], flux_sm[good],
                    color=TEXT, lw=1.3, alpha=0.90, zorder=4)

            # Lya Gaussian fit
            lf = st["lya_fit"]
            if lf:
                x_g = np.linspace(LYA_AA - 50, LYA_AA + 50, 300)
                y_g = gaussian(x_g, lf["amp"], lf["cen"],
                               lf["sigma"], lf["cont"])
                ax.plot(x_g, y_g, "--", color="#ffa657",
                        lw=1.4, alpha=0.85, zorder=5)
                ax.text(0.97, 0.95,
                        f"EW={lf['ew_rest']:.0f}Å",
                        transform=ax.transAxes,
                        color="#ffa657", fontsize=7.5,
                        ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor=AX_BG,
                                  edgecolor=SPINE, alpha=0.8))

            # β annotation
            if np.isfinite(st["beta"]):
                ax.text(0.03, 0.95,
                        f"β={st['beta']:.2f}",
                        transform=ax.transAxes,
                        color="#58a6ff", fontsize=7.5,
                        ha="left", va="top")

            ax.axhline(1.0, color=MUTED, lw=0.7, ls=":", alpha=0.55)
            ax.axhline(0.0, color=SPINE, lw=0.5, ls=":")

        # Emission line markers
        for lname, lwave in UV_LINES_MARK.items():
            if REST_WAVE_MIN < lwave < REST_WAVE_MAX:
                ax.axvline(lwave, color="#d2a8ff",
                           lw=0.6, ls="--", alpha=0.45)
                if iz == 0:
                    ax.text(lwave + 1.5, y_hi_ax * 0.92,
                            lname, color="#d2a8ff", fontsize=5.5,
                            rotation=90, va="top", ha="left", alpha=0.75)

        ax.set_xlim(REST_WAVE_MIN, REST_WAVE_MAX)
        ax.set_ylim(y_lo_ax, y_hi_ax)

        # Column header (logL bin) — top row only
        if iz == 0:
            ax.set_title(
                f"logL=[{l_edges[il]:.2f},{l_edges[il+1]:.2f})",
                color=color, fontsize=8.5, fontweight="bold",
                loc="center", pad=5)

        # Row label (z bin) — leftmost column only
        if il == 0:
            ax.set_ylabel(f"z=[{z_lo},{z_hi})\nNorm. flux",
                          color=TEXT, fontsize=8)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        # x-axis label — bottom row only
        if iz == N_Z - 1:
            ax.set_xlabel(r"$\lambda_{\rm rest}$ (Å)", color=TEXT, fontsize=8)
        else:
            ax.tick_params(labelbottom=False)

        # N label
        if st:
            ax.text(0.50, 0.03,
                    f"N={st['n_good']:,}",
                    transform=ax.transAxes,
                    color=MUTED, fontsize=6.5, ha="center", va="bottom")

# ── Summary panel A: EW_rest vs logL per z-slice ──────────────────────────────
for iz, (z_lo, z_hi) in enumerate(Z_SLICES):
    ax = ax_ew_l[iz]
    style_ax(ax,
             f"EW$_{{\\rm rest}}$(L)  z=[{z_lo},{z_hi})",
             r"$\log_{10}\ L_{\rm Ly\alpha}$  [erg/s]",
             r"EW$_{\rm rest}$  (Å)")

    logL_meds, ews, ew_errs = [], [], []
    for il in range(N_L):
        st = grid[iz][il]
        if st is None:
            continue
        lf = st["lya_fit"]
        if lf is None:
            continue
        logL_meds.append(st["logL_med"])
        ews.append(lf["ew_rest"])
        # EW uncertainty from Gaussian fit amplitude error
        ew_err = (lf["ew_rest"] * lf["amp_err"]
                  / max(lf["amp"], 1e-10))
        ew_errs.append(ew_err)

    if len(logL_meds) < 2:
        ax.text(0.5, 0.5, "insufficient data",
                ha="center", va="center",
                transform=ax.transAxes, color=MUTED)
        continue

    lm  = np.array(logL_meds)
    ew  = np.array(ews)
    ewe = np.array(ew_errs)

    ax.errorbar(lm, ew, yerr=ewe,
                fmt="o", color="#ffa657", ms=7, lw=1.5,
                capsize=3, elinewidth=1.2, zorder=5)

    # Power-law fit: log EW = a * logL + b
    try:
        coeffs = np.polyfit(lm, np.log10(np.clip(ew, 1, None)), 1)
        xl = np.linspace(lm.min() - 0.1, lm.max() + 0.1, 100)
        ax.plot(xl, 10**np.polyval(coeffs, xl),
                "--", color="#ffa657", lw=1.3, alpha=0.75,
                label=f"slope={coeffs[0]:.2f}")
        mleg(ax, loc="upper right")
    except Exception:
        pass

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))

# ── Summary panel B: β vs logL per z-slice ────────────────────────────────────
for iz, (z_lo, z_hi) in enumerate(Z_SLICES):
    ax = ax_beta_l[iz]
    style_ax(ax,
             f"UV slope β(L)  z=[{z_lo},{z_hi})",
             r"$\log_{10}\ L_{\rm Ly\alpha}$  [erg/s]",
             r"UV slope  $\beta$")

    logL_meds, betas, beta_errs = [], [], []
    for il in range(N_L):
        st = grid[iz][il]
        if st is None or not np.isfinite(st["beta"]):
            continue
        logL_meds.append(st["logL_med"])
        betas.append(st["beta"])
        beta_errs.append(st["beta_err"])

    if len(logL_meds) < 2:
        ax.text(0.5, 0.5, "insufficient data",
                ha="center", va="center",
                transform=ax.transAxes, color=MUTED)
        continue

    ax.errorbar(np.array(logL_meds), np.array(betas),
                yerr=np.array(beta_errs),
                fmt="s", color="#58a6ff", ms=7, lw=1.5,
                capsize=3, elinewidth=1.2, zorder=5)

    # Trend line
    try:
        coeffs = np.polyfit(logL_meds, betas, 1)
        xl = np.linspace(min(logL_meds) - 0.1, max(logL_meds) + 0.1, 100)
        ax.plot(xl, np.polyval(coeffs, xl),
                "--", color="#58a6ff", lw=1.3, alpha=0.75,
                label=f"slope={coeffs[0]:.2f}")
        mleg(ax, loc="upper right")
    except Exception:
        pass

    ax.axhline(-2.0, color=MUTED, lw=0.8, ls=":", alpha=0.55,
               label=r"β = −2 (reference)")

# ── Super-title ───────────────────────────────────────────────────────────────
syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
fig.suptitle(
    "HETDEX SC2 — LAE Stacks:  logL_Lya bins within fixed z-slices" + syn_tag,
    color=TEXT, fontsize=13, fontweight="bold", y=0.975,
)

# Luminosity colourbar
sm  = plt.cm.ScalarMappable(
    cmap=L_CMAP,
    norm=matplotlib.colors.Normalize(
        vmin=l_edges[0], vmax=l_edges[-1]))
sm.set_array([])
cbar_ax = fig.add_axes([0.91, 0.30, 0.012, 0.63])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label(r"$\log_{10}\ L_{\rm Ly\alpha}$  [erg/s]",
             color=MUTED, fontsize=9)
cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
cb.outline.set_edgecolor(SPINE)

if SAVE_PATH:
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 10 — NUMERICAL SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 88)
print("  HETDEX SC2 — EW_rest and β by (z, logL) cell")
print("=" * 88)
hfmt = "  {:>12}  {:>14}  {:>6}  {:>7}  {:>8}  {:>7}  {:>7}  {:>6}"
rfmt = "  {:>12}  {:>14}  {:>6}  {:>7.3f}  {:>8.1f}  {:>7.2f}  {:>7.2f}  {:>6}"
print(hfmt.format("z-slice","logL-bin","N","z_med",
                  "EW_rest","FWHM","beta","beta_e"))
print("  " + "-"*86)
for iz, (z_lo, z_hi) in enumerate(Z_SLICES):
    for il in range(N_L):
        st = grid[iz][il]
        if st is None:
            continue
        lf = st["lya_fit"]
        ew_s   = f"{lf['ew_rest']:.1f}" if lf else "n/a"
        fwhm_s = f"{lf['fwhm']:.2f}"    if lf else "n/a"
        b_s    = f"{st['beta']:.2f}"     if np.isfinite(st["beta"]) else "n/a"
        be_s   = f"{st['beta_err']:.2f}" if np.isfinite(st["beta_err"]) else "n/a"
        zbin_s = f"[{z_lo},{z_hi})"
        lbin_s = f"[{l_edges[il]:.2f},{l_edges[il+1]:.2f})"
        print(rfmt.format(
            zbin_s, lbin_s, st["n_good"],
            st["z_med"],
            float(ew_s) if ew_s != "n/a" else 0,
            float(fwhm_s) if fwhm_s != "n/a" else 0,
            float(b_s) if b_s != "n/a" else 0,
            be_s,
        ))
    print()
print("=" * 88)

print("\nPhysical interpretation:")
print("  EW slope < 0  -->  anti-correlation (brighter LAEs have smaller EW)")
print("  beta  < 0     -->  falling UV continuum (standard for star-forming galaxies)")
print("  If EW(L) slope is consistent across z-slices, the EW evolution")
print("  seen in the z-only stacks is entirely a luminosity selection effect.")
print("  If the EW-L relation shifts vertically between z-slices, that is")
print("  genuine redshift evolution of the Lya escape fraction or CGM opacity.")
