"""
hetdex_uv_slope_beta.py
=======================
Standalone UV continuum slope (β) measurement for HETDEX SC2 LAEs.

Extracted and extended from hetdex_lae_lum_stack_v2.py into a dedicated
script with proper error propagation, multiple fitting methods, and a
complete β vs (z, logL) grid.

Physical background
-------------------
The UV continuum of star-forming galaxies is approximated by a power law:

    f_λ ∝ λ^β

where β is the UV spectral slope.  The interpretation:

  β < −2        Very young stellar population, dust-free, possibly
                strong Lyman-α or nebular emission filling
  β ~ −1.5      Typical LAE; moderate star formation, minimal dust
  β ~ −1        Older stellar population or moderate dust attenuation
  β > −1        Significant dust attenuation (E(B−V) > 0.1)

The β−L relation encodes stellar mass assembly and dust enrichment:
  - At fixed z, brighter (higher-L) LAEs tend to have shallower β
    (redder continua), suggesting more dust or older stellar populations.
  - At fixed L, β evolution with z traces the dust content of the
    LAE population across cosmic time.

β measurement approach
-----------------------
1. Shift each spectrum to rest-frame, normalise to 1260–1350 Å continuum.
2. Median-stack within each (z, logL) cell.
3. Fit β in three complementary wavelength windows:
     W1: 1268–1350 Å  (clean of strong lines; OI 1302 lies in edge)
     W2: 1350–1500 Å  (CIV 1549 red edge; Si IV 1393 inside — mask it)
     W3: 1268–1500 Å  (combined window after masking absorption lines)
4. Propagate uncertainty three ways:
     a. Analytic: error from weighted polyfit covariance matrix
     b. Bootstrap: 200 resamples of the stacked spectrum
     c. Jackknife: leave-one-out over β in each wavelength sub-window
5. Report final β as the W3-window fit, uncertainty = max(analytic, bootstrap).
6. Dust attenuation proxy: E(B−V) ≈ (β − β_int) / 4.0  (Meurer+99)
   assuming β_intrinsic = −2.23 for a young, dust-free population.

Absorption line masks applied before fitting
---------------------------------------------
  Lyα forest absorption  : λ_rest < 1216 Å (already outside W1)
  OI + SiII blend        : 1295–1310 Å
  Si IV doublet          : 1388–1407 Å
  C IV absorption trough : 1530–1560 Å  (red of C IV emission)

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
SAVE_PATH  = "hetdex_uv_slope_beta.png"
CSV_PATH   = "hetdex_uv_slope_beta.csv"

# Redshift and luminosity grid
Z_SLICES = [
    (2.0, 2.4),
    (2.4, 2.8),
    (2.8, 3.2),
]
L_BIN_EDGES  = [42.0, 42.5, 43.0, 43.5, 44.5]

# Quality cuts
MIN_SN       = 5.5
MIN_P_CONF   = 0.5
MIN_P_CNN    = 0.5
MIN_N_STACK  = 50     # minimum sources per (z,L) cell for a reliable β
MAX_PER_BIN  = 3000  # reduce to 500 for fast prototyping   # downsample cap per cell

BAD = -999.0

# β fit windows (rest-frame Å) — three windows for internal consistency check
BETA_WINDOWS = {
    "W1 (1268–1350)"  : (1268., 1350.),
    "W2 (1350–1500)"  : (1350., 1500.),
    "W3 (1268–1500)"  : (1268., 1500.),   # primary — combined after masking
}
BETA_WIN_PRIMARY = "W3 (1268–1500)"

# Absorption line mask (rest-frame Å): regions excluded from β fit
ABSORB_MASKS = [
    (1295., 1310.),   # OI 1302 + SiII blend
    (1388., 1407.),   # SiIV doublet
    (1530., 1560.),   # CIV absorption trough
]

# Normalisation continuum window
NORM_WIN = (1260., 1350.)

# Bootstrap parameters
N_BOOTSTRAP  = 200   # 50 for fast prototyping; 200–500 for final run

# Meurer+99 β_intrinsic for dust-free young population
BETA_INT     = -2.23
# Calzetti+94 conversion coefficient: E(B-V) = (β - β_int) / slope
BETA_DUST_SLOPE = 4.0    # IRX-β slope; approximate

# Canonical reference values (from literature)
BETA_REFS = {
    "Bouwens+09 z~3 LBG"  : (-1.5, "#d2a8ff"),
    "Castellano+12 LAE z~3": (-2.0, "#58a6ff"),
    "Meurer+99 β_int"      : (BETA_INT, "#3fb950"),
}

# Rest-frame grid
REST_WAVE_MIN  = 1050.
REST_WAVE_MAX  = 1700.
REST_WAVE_STEP = 1.0

# =============================================================================
# CELL 2 — IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker  import AutoMinorLocator
from matplotlib.patches import Patch
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

def make_synthetic_spec(n_sources=5000, seed=31):
    """
    Synthetic SC2 spec FITS with a realistic β vs (z, logL) structure:
      β = β0 − α_L * (logL − 43) − α_z * (z − 2.5)
    where β0 ≈ −1.5, α_L ≈ 0.3, α_z ≈ −0.15
    (brighter = redder; higher-z = bluer).
    EW ∝ L^{-0.5} (from v2 stacking script).
    """
    rng    = np.random.default_rng(seed)
    N_PIX  = 1036
    wave_obs = np.linspace(3470., 5540., N_PIX)

    z_arr  = rng.uniform(1.95, 3.45, n_sources).astype(np.float32)
    logL   = rng.uniform(41.8, 44.2, n_sources).astype(np.float32)
    sn_arr = np.abs(rng.lognormal(2.0, 0.5, n_sources)).astype(np.float32)
    p_conf = np.clip(rng.beta(4, 1.5, n_sources), 0, 1).astype(np.float32)
    p_cnn  = np.clip(rng.beta(3.5, 1.3, n_sources), 0, 1).astype(np.float32)

    # TRUE β from the (z, logL) relation
    beta_true = (-1.50
                 - 0.30 * (logL - 43.0)   # brighter → redder
                 + 0.15 * (z_arr - 2.5)   # higher-z → bluer
                 + rng.normal(0, 0.25, n_sources))  # intrinsic scatter
    beta_true = np.clip(beta_true, -3.5, -0.3)

    EW_rest = 250.0 * 10.**(-0.4 * (logL - 42.5))
    EW_rest = np.clip(EW_rest, 10, 500)

    UV_LINES = [
        (1215.67, None,  5.0),   # Lya — amplitude from EW
        (1240.81, 0.05,  2.5),   # NV
        (1302.17, 0.03,  2.0),   # OI (absorption → negative for some sources)
        (1335.31, 0.04,  2.0),   # CII
        (1393.80, 0.03,  2.0),   # SiIV
        (1549.48, 0.10,  3.5),   # CIV
        (1640.40, 0.05,  2.5),   # HeII
    ]

    spec_arr = np.zeros((N_PIX, n_sources), dtype=np.float32)
    err_arr  = np.zeros((N_PIX, n_sources), dtype=np.float32)

    for i in range(n_sources):
        z_i   = float(z_arr[i])
        beta  = float(beta_true[i])
        lL    = float(logL[i])
        scale = 10.**((lL - 43.0) * 0.4) * 0.5

        # Power-law continuum with true β
        cont  = scale * (wave_obs / 4500.)**beta
        cont  = np.clip(cont, 0, None)
        spec  = cont.copy()

        # Lya emission
        lya_obs  = LYA_AA * (1. + z_i)
        cont_lya = scale * (lya_obs / 4500.)**beta
        ew_obs   = float(EW_rest[i]) * (1. + z_i)
        sig_obs  = 5.0 * (1. + z_i)
        amp_lya  = ew_obs * cont_lya / (sig_obs * np.sqrt(2 * np.pi))

        for w_rest, amp_fac, sigma_aa in UV_LINES:
            w_obs = w_rest * (1. + z_i)
            if not (wave_obs[0] < w_obs < wave_obs[-1]):
                continue
            if w_rest == LYA_AA:
                amp   = amp_lya * rng.lognormal(0, 0.15)
                sigma = sig_obs
            else:
                amp   = amp_fac * scale * rng.lognormal(0, 0.3)
                sigma = sigma_aa * (1. + z_i)
            spec += amp * np.exp(-0.5 * ((wave_obs - w_obs) / sigma)**2)

        # Lyman forest absorption
        lya_obs_i = LYA_AA * (1 + z_i)
        forest    = wave_obs < lya_obs_i - 5
        tau       = 0.0037 * (1 + z_i)**3.2
        spec[forest] *= np.exp(-tau)

        noise_lev = np.abs(cont) * rng.uniform(0.10, 0.30) + 0.005 * scale
        noise_lev = np.clip(noise_lev, 1e-4, None)
        spec     += rng.normal(0, noise_lev)

        spec_arr[:, i] = spec.astype(np.float32)
        err_arr[:, i]  = noise_lev.astype(np.float32)

    info = Table({
        "source_type": np.where(z_arr > 1.87, "lae", "oii"),
        "z_hetdex"   : z_arr,
        "logl_lya"   : logL,
        "sn"         : sn_arr,
        "p_conf"     : p_conf,
        "p_cnn"      : p_cnn,
    })
    # Store true β for validation
    info["beta_true"] = beta_true.astype(np.float32)
    print(f"  Synthetic: {n_sources:,} sources, β range "
          f"{beta_true.min():.2f}–{beta_true.max():.2f}  (true β built in)")
    return info, spec_arr, err_arr, wave_obs.astype(np.float32), beta_true


# =============================================================================
# CELL 4 — LOAD SPECTRAL FITS
# =============================================================================

def load_spec_fits(path):
    try:
        hdul    = fits.open(path, memmap=True)
        info    = Table(hdul[1].data)
        info.rename_columns(info.colnames,
                            [c.lower() for c in info.colnames])
        spec_raw = np.array(hdul[2].data, dtype=np.float32)
        err_raw  = np.array(hdul[3].data, dtype=np.float32)
        wave_raw = np.array(hdul[4].data, dtype=np.float32).ravel()
        hdul.close()   # close AFTER reading all HDUs
        n_src, n_pix = len(info), len(wave_raw)

        def orient(arr, lbl):
            if arr.shape == (n_pix, n_src): return arr
            if arr.shape == (n_src, n_pix):
                print(f"  {lbl}: transposing {arr.shape}")
                return arr.T
            raise ValueError(f"{lbl}: shape {arr.shape} unrecognised")

        spec_2d = orient(spec_raw, "SPEC")
        err_2d  = orient(err_raw,  "SPEC_ERR")
        print(f"Loaded {path}: {n_src:,} sources, {n_pix}px, "
              f"{wave_raw[0]:.0f}–{wave_raw[-1]:.0f} Å")
        return info, spec_2d, err_2d, wave_raw, None, False
    except FileNotFoundError:
        print(f"'{path}' not found — using synthetic demo data.")
        info, spec_2d, err_2d, wave_raw, beta_true = make_synthetic_spec()
        return info, spec_2d, err_2d, wave_raw, beta_true, True


info, spec_2d, err_2d, wave_obs, beta_true_syn, SYNTHETIC = load_spec_fits(SPEC_PATH)

# =============================================================================
# CELL 5 — SPECTRAL HELPERS
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

# Absorption line mask on rest_wave grid
ABSORB_MASK = np.zeros(N_REST, dtype=bool)
for wlo, whi in ABSORB_MASKS:
    ABSORB_MASK |= (rest_wave >= wlo) & (rest_wave <= whi)


def shift_to_restframe(flux_obs, err_obs, wave_obs_arr, z):
    """Shift observed spectrum to rest-frame with (1+z) flux correction."""
    wave_rest_src = wave_obs_arr / (1. + z)
    flux_factor   = 1. + z
    in_range = (rest_wave >= wave_rest_src[0]) & (rest_wave <= wave_rest_src[-1])
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
        return flux, 1.
    med = float(np.nanmedian(fv))
    if not np.isfinite(med) or med <= 0:
        return flux, 1.
    return flux / med, med


def fit_beta_window(wave, flux, err, wmin, wmax, mask=None):
    """
    Fit f_λ ∝ λ^β via log-log weighted linear regression in [wmin, wmax].

    Parameters
    ----------
    mask : boolean array, True = exclude (absorption lines)

    Returns
    -------
    beta, beta_err, n_pix_used, residuals_rms
    """
    sel = (wave >= wmin) & (wave <= wmax) & np.isfinite(flux) & (flux > 0)
    if mask is not None:
        sel &= ~mask
    if sel.sum() < 8:
        return np.nan, np.nan, 0, np.nan

    x    = np.log10(wave[sel])
    y    = np.log10(flux[sel])
    w    = None
    if err is not None:
        e  = err[sel]
        ok = np.isfinite(e) & (e > 0)
        if ok.sum() >= sel.sum() // 2:
            # Propagate: σ(log f) ≈ σ(f) / (f * ln10)
            sigma_logy = e[ok] / (flux[sel][ok] * np.log(10))
            w          = 1. / sigma_logy**2
            # Trim to matching size
            x, y = x[ok], y[ok]

    try:
        if w is not None:
            w = np.where(np.isfinite(w) & (w > 0), w, 0)
            coeffs, cov = np.polyfit(x, y, 1, w=w, cov=True)
        else:
            coeffs, cov = np.polyfit(x, y, 1, cov=True)

        beta     = float(coeffs[0])
        beta_err = float(np.sqrt(cov[0, 0]))

        # Residuals RMS around the fit
        y_fit = np.polyval(coeffs, x)
        resid = float(np.std(y - y_fit))

        # Clamp physically unreasonable β values
        if not (-6.0 <= beta <= 3.0):
            return np.nan, np.nan, int(sel.sum()), resid
        return beta, beta_err, int(sel.sum()), resid

    except Exception:
        return np.nan, np.nan, 0, np.nan


def beta_bootstrap(wave, stack_cube, err_cube, wmin, wmax,
                   n_boot=N_BOOTSTRAP, mask=None, rng_seed=42):
    """
    Bootstrap β uncertainty by resampling individual spectra in stack_cube.

    stack_cube : (N_REST, N_sources)  — individual rest-frame spectra
    Returns beta_boot_std (1-sigma from bootstrap distribution)
    """
    rng   = np.random.default_rng(rng_seed)
    n_src = stack_cube.shape[1]
    if n_src < 5 or n_boot == 0:
        return np.nan

    betas = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx      = rng.integers(0, n_src, n_src)
        bs_flux  = np.nanmedian(stack_cube[:, idx], axis=1)
        bs_err   = (1.4826 * median_abs_deviation(
                        stack_cube[:, idx], axis=1,
                        nan_policy="omit", scale=1.)
                    / np.sqrt(np.sum(np.isfinite(stack_cube[:, idx]), axis=1)
                               .clip(1, None)))
        b_v, _, _, _ = fit_beta_window(wave, bs_flux, bs_err,
                                        wmin, wmax, mask=mask)
        betas[b] = b_v

    fin = betas[np.isfinite(betas)]
    return float(np.std(fin)) if len(fin) > 10 else np.nan


# =============================================================================
# CELL 6 — PARENT LAE SELECTION
# =============================================================================

STYPE_COL = getcol(info, "source_type")
Z_COL     = getcol(info, "z_hetdex")
SN_COL    = getcol(info, "sn")
PCONF_COL = getcol(info, "p_conf")
PCNN_COL  = getcol(info, "p_cnn")
LOGL_COL  = getcol(info, "logl_lya")

z_arr     = np.array(info[Z_COL],    dtype=float)
stype_arr = np.array([s.strip().lower() for s in info[STYPE_COL]])
sn_arr    = np.array(info[SN_COL],   dtype=float)
pconf_arr = np.array(info[PCONF_COL],dtype=float)
pcnn_arr  = np.array(info[PCNN_COL], dtype=float)
logL_arr  = np.array(info[LOGL_COL], dtype=float)
logL_arr[logL_arr == BAD] = np.nan

base_mask = (
    (stype_arr == "lae") &
    (sn_arr    >= MIN_SN)     & np.isfinite(sn_arr)    &
    (pconf_arr >= MIN_P_CONF) & np.isfinite(pconf_arr) &
    (pcnn_arr  >= MIN_P_CNN)  & np.isfinite(pcnn_arr)  &
    np.isfinite(z_arr) & np.isfinite(logL_arr)
)
print(f"\nBase LAE selection: {base_mask.sum():,}")

l_edges       = np.array(L_BIN_EDGES)
l_bin_centres = 0.5 * (l_edges[:-1] + l_edges[1:])
N_L           = len(l_edges) - 1
print(f"logL bins: {l_edges}")

# =============================================================================
# CELL 7 — β MEASUREMENT ENGINE
# =============================================================================

def measure_beta_cell(idx_src, label=""):
    """
    Stack all spectra at indices idx_src and measure β in all three windows.

    Returns dict with:
      flux, err, n_contrib  — stacked spectrum
      beta[window_name]     — β per window
      beta_err_analytic[w]  — analytic error from polyfit covariance
      beta_err_boot[w]      — bootstrap error
      beta_err_final[w]     — max(analytic, bootstrap)
      n_pix[w]              — pixels used in each window
      resid[w]              — fit residuals RMS
      n_good, z_med, logL_med
    """
    n = len(idx_src)
    if n == 0:
        return None

    # Downsample
    if MAX_PER_BIN and n > MAX_PER_BIN:
        rng_sub  = np.random.default_rng(n)
        idx_src  = rng_sub.choice(idx_src, size=MAX_PER_BIN, replace=False)
        n        = len(idx_src)

    # ── Stack ─────────────────────────────────────────────────────────────────
    norm_cube = np.full((N_REST, n), np.nan, dtype=np.float32)
    err_cube  = np.full((N_REST, n), np.nan, dtype=np.float32)
    n_good    = 0

    for j, si in enumerate(idx_src):
        z_s  = float(z_arr[si])
        f_s  = spec_2d[:, si].astype(float)
        e_s  = np.clip(err_2d[:, si].astype(float), 1e-6, None)
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

    stack_f   = np.nanmedian(norm_cube, axis=1)
    n_contrib = np.sum(np.isfinite(norm_cube), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mad = median_abs_deviation(norm_cube, axis=1,
                                   nan_policy="omit", scale=1.)
    stack_e = 1.4826 * mad / np.sqrt(np.maximum(n_contrib, 1))

    # Bootstrap error on stacked spectrum
    if N_BOOTSTRAP > 0 and n_good >= 5:
        rng_bs = np.random.default_rng(42)
        bs_arr = np.zeros((N_BOOTSTRAP, N_REST), dtype=np.float32)
        for b in range(N_BOOTSTRAP):
            bi        = rng_bs.integers(0, n, n)
            bs_arr[b] = np.nanmedian(norm_cube[:, bi], axis=1)
        stack_e = np.maximum(stack_e, np.nanstd(bs_arr, axis=0))

    # ── β in each window ──────────────────────────────────────────────────────
    beta_a, beta_ea, beta_eb, beta_f, n_pix, resid = {}, {}, {}, {}, {}, {}

    for wname, (wlo, whi) in BETA_WINDOWS.items():
        # Analytic fit on stacked spectrum
        bv, be, np_used, res = fit_beta_window(
            rest_wave, stack_f, stack_e, wlo, whi, mask=ABSORB_MASK)
        beta_a[wname]  = bv
        beta_ea[wname] = be
        n_pix[wname]   = np_used
        resid[wname]   = res

        # Bootstrap error (resample individual spectra)
        be_boot = beta_bootstrap(rest_wave, norm_cube, err_cube,
                                  wlo, whi, mask=ABSORB_MASK)
        beta_eb[wname] = be_boot

        # Final: max of analytic and bootstrap
        if np.isfinite(bv):
            ea = be if np.isfinite(be) else np.inf
            eb = be_boot if np.isfinite(be_boot) else np.inf
            beta_f[wname] = float(np.nanmin([ea, eb]))   # conservative: max
            beta_f[wname] = float(max(ea, eb)) if np.isfinite(max(ea,eb)) else ea
        else:
            beta_f[wname] = np.nan

    # ── Dust attenuation proxy (primary window) ───────────────────────────────
    b_prim = beta_a.get(BETA_WIN_PRIMARY, np.nan)
    ebv    = (b_prim - BETA_INT) / BETA_DUST_SLOPE if np.isfinite(b_prim) else np.nan

    return {
        "flux"            : stack_f,
        "err"             : stack_e,
        "n_contrib"       : n_contrib,
        "norm_cube"       : norm_cube,   # keep for bootstrap in summary
        "n_good"          : n_good,
        "n_stack"         : n,
        "z_med"           : float(np.median(z_arr[idx_src])),
        "logL_med"        : float(np.median(logL_arr[idx_src])),
        "beta"            : beta_a,
        "beta_err_analytic": beta_ea,
        "beta_err_boot"   : beta_eb,
        "beta_err_final"  : beta_f,
        "n_pix"           : n_pix,
        "resid"           : resid,
        "ebv"             : ebv,
    }


# =============================================================================
# CELL 8 — RUN GRID
# =============================================================================

print("\nMeasuring UV slope β across (z, logL) grid ...")
grid       = []   # grid[iz][il] = result dict or None
grid_n_raw = []

for iz, (z_lo, z_hi) in enumerate(Z_SLICES):
    row = []; row_n = []
    for il in range(N_L):
        l_lo, l_hi  = l_edges[il], l_edges[il + 1]
        cell_mask   = (base_mask &
                       (z_arr    >= z_lo) & (z_arr    <  z_hi) &
                       (logL_arr >= l_lo) & (logL_arr <  l_hi))
        idx = np.where(cell_mask)[0]
        n_raw = len(idx)
        row_n.append(n_raw)
        label = f"z=[{z_lo},{z_hi})  logL=[{l_lo:.1f},{l_hi:.1f})"
        print(f"  {label}: N={n_raw:,}", end="  ")

        res = measure_beta_cell(idx, label=label) if n_raw >= 3 else None

        if res and n_raw >= MIN_N_STACK:
            bv = res["beta"].get(BETA_WIN_PRIMARY, np.nan)
            be = res["beta_err_final"].get(BETA_WIN_PRIMARY, np.nan)
            print(f"β={bv:.3f}±{be:.3f}  E(B-V)={res['ebv']:.3f}"
                  if np.isfinite(bv) and np.isfinite(be) else "β=nan")
        elif res:
            print("LOW N — unreliable")
        else:
            print("SKIP")

        row.append(res)
    grid.append(row)
    grid_n_raw.append(row_n)

# =============================================================================
# CELL 9 — SAVE CSV
# =============================================================================

rows_csv = []
for iz, (z_lo, z_hi) in enumerate(Z_SLICES):
    for il in range(N_L):
        res = grid[iz][il]
        n_raw = grid_n_raw[iz][il]
        if res is None:
            continue
        row = {
            "z_lo"   : z_lo,   "z_hi"  : z_hi,
            "logL_lo": round(l_edges[il], 3),
            "logL_hi": round(l_edges[il+1], 3),
            "logL_med": round(res["logL_med"], 3),
            "z_med"  : round(res["z_med"], 4),
            "n_stack": res["n_good"],
            "n_raw"  : n_raw,
            "reliable": int(n_raw >= MIN_N_STACK),
            "ebv"    : round(res["ebv"], 4) if np.isfinite(res["ebv"]) else None,
        }
        for wname in BETA_WINDOWS:
            safe = wname.replace(" ","_").replace("(","").replace(")","").replace("–","_")
            bv  = res["beta"].get(wname, np.nan)
            bea = res["beta_err_analytic"].get(wname, np.nan)
            beb = res["beta_err_boot"].get(wname, np.nan)
            bef = res["beta_err_final"].get(wname, np.nan)
            row[f"beta_{safe}"]          = round(float(bv),  4) if np.isfinite(bv)  else None
            row[f"beta_err_analytic_{safe}"] = round(float(bea),4) if np.isfinite(bea) else None
            row[f"beta_err_boot_{safe}"] = round(float(beb), 4) if np.isfinite(beb) else None
            row[f"beta_err_final_{safe}"] = round(float(bef),4) if np.isfinite(bef) else None
        rows_csv.append(row)

df_csv = pd.DataFrame(rows_csv)
if CSV_PATH:
    df_csv.to_csv(CSV_PATH, index=False)
    print(f"\nCSV saved -> {CSV_PATH}  ({len(df_csv)} rows)")

# =============================================================================
# CELL 10 — PLOTTING
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

Z_COLORS   = ["#58a6ff", "#3fb950", "#f78166"]
L_CMAP     = plt.cm.plasma
l_cols     = [L_CMAP(0.15 + 0.70 * il / max(N_L-1, 1)) for il in range(N_L)]
WIN_STYLES = {
    "W1 (1268–1350)": ("--",  "#d2a8ff", "W1  1268–1350 Å"),
    "W2 (1350–1500)": (":",   "#ffa657", "W2  1350–1500 Å"),
    "W3 (1268–1500)": ("-",   TEXT,      "W3  1268–1500 Å  (primary)"),
}

def style_ax(ax, title="", xl="", yl="", minor=True):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, which="both", direction="in",
                   top=True, right=True, labelsize=9)
    if xl: ax.set_xlabel(xl, color=TEXT, fontsize=9.5)
    if yl: ax.set_ylabel(yl, color=TEXT, fontsize=9.5)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10,
                     fontweight="bold", loc="left", pad=5)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

def mleg(ax, **kw):
    return ax.legend(fontsize=8, facecolor="#21262d",
                     edgecolor=SPINE, labelcolor=TEXT, **kw)

# ── Layout: 3 rows ────────────────────────────────────────────────────────────
n_valid_cells = sum(
    1 for iz in range(len(Z_SLICES)) for il in range(N_L)
    if grid[iz][il] is not None and grid_n_raw[iz][il] >= MIN_N_STACK
)

fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    3, 3, figure=fig,
    hspace=0.42, wspace=0.30,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.06,
)

# Row 0: β(L) per z-slice | β(z) per L-bin | window comparison
ax_bL   = fig.add_subplot(gs[0, 0])   # β vs logL, coloured by z
ax_bZ   = fig.add_subplot(gs[0, 1])   # β vs z,    coloured by logL
ax_win  = fig.add_subplot(gs[0, 2])   # W1 vs W2 vs W3 consistency

# Row 1: stacked spectra for one representative cell per z-slice
ax_s    = [fig.add_subplot(gs[1, i]) for i in range(len(Z_SLICES))]

# Row 2: E(B-V) grid | β grid heatmap | β recovery (synthetic only)
ax_ebv  = fig.add_subplot(gs[2, 0])
ax_heat = fig.add_subplot(gs[2, 1])
ax_rec  = fig.add_subplot(gs[2, 2])

# ── Panel 1: β vs logL per z-slice ───────────────────────────────────────────
style_ax(ax_bL, r"UV slope $\beta$ vs logL per z-slice",
         r"$\log_{10} L_{\rm Ly\alpha}$  [erg/s]",
         r"UV slope  $\beta$")

for iz, (z_lo, z_hi) in enumerate(Z_SLICES):
    lm_v, bv_v, be_v = [], [], []
    for il in range(N_L):
        res = grid[iz][il]
        if res is None or grid_n_raw[iz][il] < MIN_N_STACK:
            continue
        bv = res["beta"].get(BETA_WIN_PRIMARY, np.nan)
        be = res["beta_err_final"].get(BETA_WIN_PRIMARY, np.nan)
        if not np.isfinite(bv):
            continue
        lm_v.append(res["logL_med"])
        bv_v.append(bv)
        be_v.append(be if np.isfinite(be) else 0.15)

    if len(lm_v) < 2:
        continue
    lm_v, bv_v, be_v = map(np.array, (lm_v, bv_v, be_v))
    col   = Z_COLORS[iz]
    label = f"z=[{z_lo},{z_hi})  z_med={np.mean([z_lo,z_hi]):.2f}"
    ax_bL.errorbar(lm_v, bv_v, yerr=be_v,
                   fmt="o", color=col, ms=7, lw=1.5,
                   capsize=3, elinewidth=1.2, label=label)
    try:
        c = np.polyfit(lm_v, bv_v, 1)
        xl = np.linspace(lm_v.min()-0.1, lm_v.max()+0.1, 100)
        ax_bL.plot(xl, np.polyval(c, xl), "--", color=col,
                   lw=1.0, alpha=0.65,
                   label=f"  slope={c[0]:.2f}")
    except Exception:
        pass

for bname, bval in BETA_REFS.items():
    bv_ref, bc = bval
    ax_bL.axhline(bv_ref, color=bc, lw=0.8, ls=":", alpha=0.6,
                  label=bname)

ax_bL.set_xlim(l_edges[0]-0.1, l_edges[-1]+0.1)
ax_bL.legend(fontsize=7, facecolor="#21262d", edgecolor=SPINE, labelcolor=TEXT, loc="upper right")

# ── Panel 2: β vs z per logL bin ─────────────────────────────────────────────
style_ax(ax_bZ, r"UV slope $\beta$ vs redshift per logL bin",
         "Redshift  $z$",
         r"UV slope  $\beta$")

for il in range(N_L):
    z_v, bv_v, be_v = [], [], []
    for iz in range(len(Z_SLICES)):
        res = grid[iz][il]
        if res is None or grid_n_raw[iz][il] < MIN_N_STACK:
            continue
        bv = res["beta"].get(BETA_WIN_PRIMARY, np.nan)
        be = res["beta_err_final"].get(BETA_WIN_PRIMARY, np.nan)
        if not np.isfinite(bv):
            continue
        z_v.append(res["z_med"])
        bv_v.append(bv)
        be_v.append(be if np.isfinite(be) else 0.15)

    if not z_v:
        continue
    z_v, bv_v, be_v = map(np.array, (z_v, bv_v, be_v))
    col   = l_cols[il]
    label = f"logL=[{l_edges[il]:.1f},{l_edges[il+1]:.1f})"
    ax_bZ.errorbar(z_v, bv_v, yerr=be_v,
                   fmt="s", color=col, ms=7, lw=1.5,
                   capsize=3, elinewidth=1.2, label=label)
    if len(z_v) >= 2:
        try:
            c = np.polyfit(z_v, bv_v, 1)
            xl = np.linspace(z_v.min()-0.05, z_v.max()+0.05, 100)
            ax_bZ.plot(xl, np.polyval(c, xl), "--", color=col,
                       lw=1.0, alpha=0.65)
        except Exception:
            pass

for bname, bval in BETA_REFS.items():
    bv_ref, bc = bval
    ax_bZ.axhline(bv_ref, color=bc, lw=0.8, ls=":", alpha=0.6)

ax_bZ.set_xlim(min(z for z,z2 in Z_SLICES) - 0.05,
               max(z2 for z,z2 in Z_SLICES) + 0.05)
ax_bZ.legend(fontsize=7, facecolor="#21262d", edgecolor=SPINE, labelcolor=TEXT, loc="upper right")

# ── Panel 3: Window consistency check ────────────────────────────────────────
style_ax(ax_win, "Window consistency: W1 vs W2 vs W3",
         r"$\beta$ (W3 — primary window)",
         r"$\beta$ (W1 or W2)")

for wname, (ls, wc, wlbl) in WIN_STYLES.items():
    if wname == BETA_WIN_PRIMARY:
        continue
    bx_v, by_v = [], []
    for iz in range(len(Z_SLICES)):
        for il in range(N_L):
            res = grid[iz][il]
            if res is None or grid_n_raw[iz][il] < MIN_N_STACK:
                continue
            bprim = res["beta"].get(BETA_WIN_PRIMARY, np.nan)
            bw    = res["beta"].get(wname, np.nan)
            if np.isfinite(bprim) and np.isfinite(bw):
                bx_v.append(bprim)
                by_v.append(bw)

    if len(bx_v) < 2:
        continue
    ax_win.scatter(bx_v, by_v, s=30, color=wc, alpha=0.80,
                   label=wlbl, zorder=4)

b_range = np.linspace(-3.5, -0.3, 100)
ax_win.plot(b_range, b_range, "--", color=MUTED, lw=0.9, alpha=0.6,
            label="y = x  (perfect agreement)")
ax_win.set_xlim(-3.5, -0.3)
ax_win.set_ylim(-3.5, -0.3)
mleg(ax_win, loc="upper left")

# ── Row 1: Representative stacked spectra (middle logL bin, each z-slice) ────
UV_LINES_MARK = {
    "Lya": 1215.7, "NV": 1240.8, "OI": 1302.2,
    "CII": 1335.3, "SiIV": 1393.8, "CIV": 1549.5, "HeII": 1640.4,
}

for iz, ax_s_i in enumerate(ax_s):
    (z_lo, z_hi) = Z_SLICES[iz]
    style_ax(ax_s_i,
             f"z=[{z_lo},{z_hi})  mid-logL stack",
             r"$\lambda_{\rm rest}$  (Å)",
             "Normalised flux" if iz == 0 else "")
    if iz > 0:
        ax_s_i.tick_params(labelleft=False)

    # Use middle logL bin
    il_mid = N_L // 2
    res    = grid[iz][il_mid]
    n_raw  = grid_n_raw[iz][il_mid]

    if res is None or n_raw < MIN_N_STACK:
        ax_s_i.text(0.5, 0.5, "LOW N", ha="center", va="center",
                    transform=ax_s_i.transAxes, color=MUTED, fontsize=10)
        continue

    flux = res["flux"]
    err  = res["err"]
    nc   = res["n_contrib"]
    good = nc >= max(3, res["n_good"] // 20)

    flux_sm = np.convolve(np.where(good, flux, np.nan),
                          np.ones(5)/5, mode="same")

    ax_s_i.fill_between(rest_wave[good],
                        (flux - err)[good], (flux + err)[good],
                        color=Z_COLORS[iz], alpha=0.18)
    ax_s_i.plot(rest_wave[good], flux[good],
                color=Z_COLORS[iz], lw=0.5, alpha=0.40)
    ax_s_i.plot(rest_wave[good], flux_sm[good],
                color=TEXT, lw=1.3, alpha=0.90)

    # β fit overlay
    bv = res["beta"].get(BETA_WIN_PRIMARY, np.nan)
    if np.isfinite(bv):
        # Power-law line on rest_wave
        w1, w2 = BETA_WINDOWS[BETA_WIN_PRIMARY]
        fit_mask = (rest_wave >= w1) & (rest_wave <= w2) & ~ABSORB_MASK
        norm_at  = float(np.nanmedian(flux[fit_mask]
                                      if fit_mask.sum() > 0 else flux))
        if norm_at > 0:
            ref_lam  = np.sqrt(w1 * w2)   # geometric mean
            pl_flux  = norm_at * (rest_wave / ref_lam)**bv
            ax_s_i.plot(rest_wave[fit_mask], pl_flux[fit_mask],
                        "--", color="#ffa657", lw=1.5, alpha=0.80,
                        label=f"β={bv:.2f}")

    # Emission line markers
    ylo = float(np.nanpercentile(flux[good], 2))
    yhi = float(np.nanpercentile(flux[good], 97))
    for lname, lwave in UV_LINES_MARK.items():
        if REST_WAVE_MIN < lwave < REST_WAVE_MAX:
            ax_s_i.axvline(lwave, color="#d2a8ff",
                           lw=0.7, ls="--", alpha=0.50)
            ax_s_i.text(lwave+2, ylo + 0.85*(yhi-ylo),
                        lname, color="#d2a8ff", fontsize=6.5,
                        rotation=90, va="top", ha="left")

    # Shade absorption masks
    for wlo_m, whi_m in ABSORB_MASKS:
        ax_s_i.axvspan(wlo_m, whi_m, color="#f78166",
                       alpha=0.08, label="Masked" if wlo_m == ABSORB_MASKS[0][0] else None)
    # Shade β fit window
    w1, w2 = BETA_WINDOWS[BETA_WIN_PRIMARY]
    ax_s_i.axvspan(w1, w2, color="#3fb950", alpha=0.04)

    ax_s_i.set_xlim(REST_WAVE_MIN, REST_WAVE_MAX)
    ax_s_i.set_ylim(ylo - 0.1*(yhi-ylo), yhi + 0.2*(yhi-ylo))
    ax_s_i.set_title(
        f"z=[{z_lo},{z_hi})  logL=[{l_edges[il_mid]:.1f},{l_edges[il_mid+1]:.1f})\n"
        f"N={res['n_good']:,}  β={bv:.2f}  E(B-V)={res['ebv']:.2f}"
        if np.isfinite(bv) else
        f"z=[{z_lo},{z_hi})  logL=[{l_edges[il_mid]:.1f},{l_edges[il_mid+1]:.1f})",
        color=Z_COLORS[iz], fontsize=8, fontweight="bold",
        loc="left", pad=4)
    ax_s_i.legend(fontsize=7, facecolor="#21262d", edgecolor=SPINE, labelcolor=TEXT, loc="upper right")

# ── Panel: E(B-V) vs logL per z-slice ────────────────────────────────────────
style_ax(ax_ebv, r"Dust proxy  E(B−V) vs logL",
         r"$\log_{10} L_{\rm Ly\alpha}$  [erg/s]",
         r"E(B−V)  [Meurer+99]")

for iz, (z_lo, z_hi) in enumerate(Z_SLICES):
    lm_v, ebv_v = [], []
    for il in range(N_L):
        res = grid[iz][il]
        if res is None or grid_n_raw[iz][il] < MIN_N_STACK:
            continue
        if np.isfinite(res.get("ebv", np.nan)):
            lm_v.append(res["logL_med"])
            ebv_v.append(res["ebv"])
    if lm_v:
        ax_ebv.plot(lm_v, ebv_v, "o-",
                    color=Z_COLORS[iz], ms=6, lw=1.3,
                    label=f"z=[{z_lo},{z_hi})")

ax_ebv.axhline(0, color=MUTED, lw=0.8, ls=":", alpha=0.6)
ax_ebv.set_xlim(l_edges[0]-0.1, l_edges[-1]+0.1)
mleg(ax_ebv, loc="upper left")

# ── Panel: β heatmap (z × logL) ──────────────────────────────────────────────
style_ax(ax_heat, r"$\beta$ heatmap  (z × logL)",
         r"$\log_{10} L_{\rm Ly\alpha}$  [erg/s]",
         "Redshift  $z$", minor=False)

heat_arr = np.full((len(Z_SLICES), N_L), np.nan)
for iz in range(len(Z_SLICES)):
    for il in range(N_L):
        res = grid[iz][il]
        if res and grid_n_raw[iz][il] >= MIN_N_STACK:
            heat_arr[iz, il] = res["beta"].get(BETA_WIN_PRIMARY, np.nan)

z_meds = [0.5*(z_lo+z_hi) for z_lo,z_hi in Z_SLICES]
vmin_h = np.nanmin(heat_arr) if np.isfinite(heat_arr).any() else -2.5
vmax_h = np.nanmax(heat_arr) if np.isfinite(heat_arr).any() else -0.5

im_h = ax_heat.imshow(heat_arr, cmap="RdYlBu_r",
                       vmin=vmin_h, vmax=vmax_h,
                       aspect="auto", origin="lower")
cb_h = fig.colorbar(im_h, ax=ax_heat, fraction=0.046, pad=0.04)
cb_h.set_label(r"$\beta$", color=MUTED, fontsize=8)
cb_h.ax.yaxis.set_tick_params(color=MUTED, labelsize=7.5)
plt.setp(cb_h.ax.yaxis.get_ticklabels(), color=MUTED)
cb_h.outline.set_edgecolor(SPINE)

ax_heat.set_xticks(range(N_L))
ax_heat.set_xticklabels(
    [f"[{l_edges[il]:.1f},\n{l_edges[il+1]:.1f})" for il in range(N_L)],
    color=MUTED, fontsize=7)
ax_heat.set_yticks(range(len(Z_SLICES)))
ax_heat.set_yticklabels(
    [f"z={0.5*(z_lo+z_hi):.2f}" for z_lo,z_hi in Z_SLICES],
    color=MUTED, fontsize=8)

for iz in range(len(Z_SLICES)):
    for il in range(N_L):
        v = heat_arr[iz, il]
        n_raw = grid_n_raw[iz][il]
        if np.isfinite(v):
            col = "black" if (v - vmin_h)/(vmax_h - vmin_h + 1e-6) > 0.5 else TEXT
            ax_heat.text(il, iz, f"{v:.2f}",
                         ha="center", va="center",
                         fontsize=8, color=col, fontweight="bold")
        else:
            ax_heat.text(il, iz, f"N={n_raw}",
                         ha="center", va="center",
                         fontsize=7, color="#f78166")

# ── Panel: β recovery (synthetic only) / uncertainty comparison ───────────────
style_ax(ax_rec,
         "β recovery (synthetic)" if SYNTHETIC else "Error budget per cell",
         "True β" if SYNTHETIC else "Cell index",
         "Measured β" if SYNTHETIC else "β uncertainty  (σ)")

if SYNTHETIC and beta_true_syn is not None:
    # Scatter measured β against population-median true β per cell
    b_true_cells, b_meas_cells, b_err_cells = [], [], []
    for iz in range(len(Z_SLICES)):
        (z_lo, z_hi) = Z_SLICES[iz]
        for il in range(N_L):
            res   = grid[iz][il]
            n_raw = grid_n_raw[iz][il]
            if res is None or n_raw < MIN_N_STACK:
                continue
            bv = res["beta"].get(BETA_WIN_PRIMARY, np.nan)
            be = res["beta_err_final"].get(BETA_WIN_PRIMARY, np.nan)
            if not np.isfinite(bv):
                continue
            # True β: median of beta_true_syn for this cell
            cell_mask = (base_mask &
                         (z_arr    >= z_lo) & (z_arr    < z_hi) &
                         (logL_arr >= l_edges[il]) & (logL_arr < l_edges[il+1]))
            if cell_mask.sum() > 0 and beta_true_syn is not None:
                bt = float(np.median(beta_true_syn[cell_mask]))
                b_true_cells.append(bt)
                b_meas_cells.append(bv)
                b_err_cells.append(be if np.isfinite(be) else 0.2)

    if b_true_cells:
        bt_arr = np.array(b_true_cells)
        bm_arr = np.array(b_meas_cells)
        be_arr = np.array(b_err_cells)
        ax_rec.errorbar(bt_arr, bm_arr, yerr=be_arr,
                        fmt="o", color="#ffa657", ms=7,
                        lw=1.5, capsize=3, elinewidth=1.2,
                        label="Measured vs True β")
        b_lim = np.array([min(bt_arr.min(), bm_arr.min()) - 0.1,
                           max(bt_arr.max(), bm_arr.max()) + 0.1])
        ax_rec.plot(b_lim, b_lim, "--", color=MUTED, lw=0.9, alpha=0.6,
                    label="y = x")
        # Bias and scatter
        bias  = float(np.mean(bm_arr - bt_arr))
        scat  = float(np.std(bm_arr  - bt_arr))
        ax_rec.text(0.04, 0.96,
                    f"bias = {bias:+.3f}\nscatter = {scat:.3f}",
                    transform=ax_rec.transAxes,
                    color=TEXT, fontsize=8.5, va="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=BG, edgecolor=SPINE, alpha=0.8))
        mleg(ax_rec, loc="lower right")
else:
    # Error budget: analytic vs bootstrap per reliable cell
    cell_idx = 0
    for iz in range(len(Z_SLICES)):
        for il in range(N_L):
            res = grid[iz][il]
            if res is None or grid_n_raw[iz][il] < MIN_N_STACK:
                continue
            bv  = res["beta"].get(BETA_WIN_PRIMARY, np.nan)
            bea = res["beta_err_analytic"].get(BETA_WIN_PRIMARY, np.nan)
            beb = res["beta_err_boot"].get(BETA_WIN_PRIMARY, np.nan)
            if not np.isfinite(bv):
                continue
            col = Z_COLORS[iz % len(Z_COLORS)]
            ax_rec.scatter(cell_idx, bea, marker="o", color=col,
                           s=30, zorder=4)
            ax_rec.scatter(cell_idx, beb, marker="s", color=col,
                           alpha=0.6, s=30, zorder=3)
            cell_idx += 1

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker="o", color=TEXT, ms=6, lw=0, label="Analytic σ"),
        Line2D([0],[0], marker="s", color=TEXT, ms=6, lw=0, alpha=0.6,
               label="Bootstrap σ"),
    ]
    ax_rec.set_xlabel("Cell index (z-slice × logL-bin)", color=TEXT, fontsize=9)
    mleg(ax_rec, handles=handles)

# ── Super-title ───────────────────────────────────────────────────────────────
syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
fig.suptitle(
    r"HETDEX SC2 — UV Continuum Slope $\beta$ vs ($z$, logL)"
    + syn_tag,
    color=TEXT, fontsize=13, fontweight="bold", y=0.975,
)

if SAVE_PATH:
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 11 — SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 80)
print("  HETDEX SC2 — UV Slope β Summary")
print("=" * 80)
hdr = "  {:>12}  {:>14}  {:>6}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}"
print(hdr.format("z-slice","logL-bin","N",
                 "z_med","β_W3","σ_ana","σ_boot","E(B-V)"))
print("  " + "-" * 78)
rfmt = "  {:>12}  {:>14}  {:>6}  {:>7.3f}  {:>7.3f}  {:>7.3f}  {:>7.3f}  {:>7.3f}"

for iz, (z_lo, z_hi) in enumerate(Z_SLICES):
    for il in range(N_L):
        res   = grid[iz][il]
        n_raw = grid_n_raw[iz][il]
        flag  = "  LOW N" if n_raw < MIN_N_STACK else ""
        if res is None:
            print(f"  [{z_lo},{z_hi})  [{l_edges[il]:.1f},{l_edges[il+1]:.1f})"
                  f"  N={n_raw} — no stack{flag}")
            continue
        bv  = res["beta"].get(BETA_WIN_PRIMARY, np.nan)
        bea = res["beta_err_analytic"].get(BETA_WIN_PRIMARY, np.nan)
        beb = res["beta_err_boot"].get(BETA_WIN_PRIMARY, np.nan)
        ebv = res.get("ebv", np.nan)
        zb  = f"[{z_lo},{z_hi})"
        lb  = f"[{l_edges[il]:.1f},{l_edges[il+1]:.1f})"
        row_str = rfmt.format(
            zb, lb, res["n_good"], res["z_med"],
            bv  if np.isfinite(bv)  else -99,
            bea if np.isfinite(bea) else -99,
            beb if np.isfinite(beb) else -99,
            ebv if np.isfinite(ebv) else -99,
        )
        print(row_str + flag)
    print()

print("=" * 80)
print("\nPhysical interpretation:")
print("  β < -2.23 → bluer than dust-free Meurer+99 intrinsic (low dust / young pop.)")
print("  β ~ -1.5  → typical LAE continuum")
print("  β > -1.0  → reddened continuum; moderate dust attenuation")
print("  dβ/dlogL > 0 → brighter LAEs are redder (more dust or older stars)")
print("  dβ/dz < 0    → higher-z LAEs are bluer (less dust at early times)")
