"""
hetdex_desi_xmatch.py
=====================
Cross-validate HETDEX SC2 source classifications and redshifts against
the DESI DR1 HETDEX Follow-up Value Added Catalog (Landriau et al. 2025).

Data
----
  DESI_HETDEX_SPEC_v1.6.fits
  https://data.desi.lbl.gov/public/dr1/vac/dr1/hetdex/v1.6/DESI_HETDEX_SPEC_v1.6.fits
  (or on NERSC: /global/cfs/cdirs/desi/public/dr1/vac/dr1/hetdex/)
  File size: 168 MB  |  2374 sources  |  Landriau et al. 2025 arXiv:2503.02229

  hetdex_sc2_detinfo_v1.5.fits  (for p_conf, p_cnn, plya_classification)

FITS structure of DESI_HETDEX_SPEC_v1.6.fits
----------------------------------------------
  HDU 1  INFO              BinTableHDU  34 columns (one row per source)
  HDU 2  HETDEX_WAVE       ImageHDU     HETDEX wavelength grid
  HDU 3  HETDEX_SPEC       ImageHDU     HETDEX spectra
  HDU 4  HETDEX_SPEC_ERR   ImageHDU     HETDEX errors
  HDU 5  DESI_WAVE         ImageHDU     DESI wavelength grid
  HDU 6  DESI_WAVE_VACUUM  ImageHDU     DESI vacuum wavelengths
  HDU 7  DESI_SPEC         ImageHDU     DESI spectra
  HDU 8  DESI_SPEC_ERR     ImageHDU     DESI errors

Key INFO columns used
---------------------
  DETECTID            HETDEX detection identifier → join key to SC2
  VI_Z                DESI visual-inspection redshift (ground truth)
  VI_QUALITY          DESI VI quality flag (higher = more reliable)
  SOURCE_TYPE         DESI classification: LAE | OII | AGN | other
  Z_BEST_0PT5         HETDEX pipeline redshift at p(Lya)≥0.5 threshold
  Z_BEST_0PT3         HETDEX pipeline redshift at p(Lya)≥0.3 threshold
  Z_BEST_0PT4         HETDEX pipeline redshift at p(Lya)≥0.4 threshold
  SN_HETDEX           HETDEX line S/N
  WAVE_HETDEX         HETDEX observed emission-line wavelength
  FLUX_HETDEX         HETDEX line flux
  SIGMA_HETDEX        HETDEX line width (σ)
  WAVE_DESI           DESI observed emission-line wavelength
  FLUX_DESI           DESI line flux
  SEP                 Angular separation HETDEX–DESI  [arcsec]

Science goals
-------------
1. Redshift validation:  z_hetdex vs VI_Z — bias, scatter, catastrophic rate
2. Classification confusion matrix:  HETDEX source_type vs DESI SOURCE_TYPE
3. p_conf / p_cnn calibration curves:  P(DESI=LAE | p_conf) reliability diagrams
4. Line flux comparison:  FLUX_HETDEX vs FLUX_DESI — systematic offsets
5. σ linewidth comparison:  SIGMA_HETDEX vs SIGMA_DESI
6. Dual-spectrum overlay:  HETDEX + DESI on same axes for disagreement cases
7. Export:  ambiguous sources (HETDEX≠DESI classification) for follow-up

Requirements
------------
  pip install astropy numpy matplotlib scipy pandas requests

Download (if you don't have the file locally)
---------------------------------------------
  import requests
  url = "https://data.desi.lbl.gov/public/dr1/vac/dr1/hetdex/v1.6/DESI_HETDEX_SPEC_v1.6.fits"
  with requests.get(url, stream=True) as r:
      with open("DESI_HETDEX_SPEC_v1.6.fits", "wb") as f:
          for chunk in r.iter_content(chunk_size=8192):
              f.write(chunk)
"""

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

VAC_PATH     = "DESI_HETDEX_SPEC_v1.6.fits"      # DESI×HETDEX VAC
DETINFO_PATH = "hetdex_sc2_detinfo_v1.5.fits"     # for p_conf, p_cnn, plya
SAVE_PATH    = "hetdex_desi_xmatch.png"
CSV_PATH     = "hetdex_desi_ambiguous.csv"

# Download VAC automatically if absent?
AUTO_DOWNLOAD = True

# Quality cuts
MIN_VI_QUALITY  = 2      # DESI visual-inspection quality (1-4; ≥2 = reliable)
MIN_SN_HETDEX   = 4.5    # HETDEX line S/N
MAX_SEP_ARCSEC  = 2.0    # max HETDEX–DESI angular separation

# p_conf threshold tiers for calibration curves

# Catastrophic redshift failure threshold
DZ_CATAS = 0.1      # |Δz| > this = catastrophic failure
DZ_CLOSE = 0.01     # |Δz| < this = excellent agreement

LYA_AA  = 1215.67
OII_AA  = 3727.09
BAD     = -999.0

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
from matplotlib.ticker  import AutoMinorLocator, FixedLocator
from matplotlib.patches import Patch
from scipy.ndimage      import gaussian_filter1d
from scipy.stats        import binned_statistic

from astropy.io    import fits
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
import astropy.units as u

try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
        matplotlib.rcParams["figure.dpi"] = 120
except Exception:
    pass

PCONF_BINS = np.arange(0, 1.05, 0.10)

print("Imports OK.")

# =============================================================================
# CELL 3 — DOWNLOAD VAC IF NEEDED
# =============================================================================

import os
if not os.path.exists(VAC_PATH) and AUTO_DOWNLOAD:
    try:
        import requests
        url = ("https://data.desi.lbl.gov/public/dr1/vac/dr1/hetdex/"
               "v1.6/DESI_HETDEX_SPEC_v1.6.fits")
        print(f"Downloading {url} ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done  = 0
            with open(VAC_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done/total*100:.0f}%", end="", flush=True)
        print(f"\nDownloaded -> {VAC_PATH}")
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Set VAC_PATH to the local file path or place the file here.")

# =============================================================================
# CELL 4 — SYNTHETIC DATA GENERATOR
# =============================================================================

def make_synthetic_vac(n=2374, seed=42):
    """
    Realistic synthetic DESI_HETDEX_SPEC_v1.6.fits equivalent.
    Builds a matched HETDEX–DESI catalog with:
    - ~60% LAE, ~25% OII, ~8% AGN, ~7% other
    - Realistic redshift scatter and confusion
    - p_conf, p_cnn, plya columns for calibration analysis
    - Paired HETDEX + DESI 1D spectra
    """
    rng = np.random.default_rng(seed)

    # True DESI classifications
    desi_types = rng.choice(
        ["LAE", "OII", "AGN", "other"],
        size=n, p=[0.60, 0.25, 0.08, 0.07]
    )

    # True redshifts
    vi_z = np.zeros(n)
    vi_z[desi_types == "LAE"]   = rng.uniform(1.9, 3.5,
                                               (desi_types=="LAE").sum())
    vi_z[desi_types == "OII"]   = rng.uniform(0.0, 0.5,
                                               (desi_types=="OII").sum())
    vi_z[desi_types == "AGN"]   = rng.uniform(0.5, 3.0,
                                               (desi_types=="AGN").sum())
    vi_z[desi_types == "other"] = rng.uniform(0.0, 3.5,
                                               (desi_types=="other").sum())

    # DESI VI quality (2-4; most have ≥2 by construction)
    vi_quality = rng.choice([2, 3, 4], n, p=[0.20, 0.45, 0.35])

    # HETDEX WAVE_HETDEX: observed wavelength of detected line
    wave_h = np.where(
        desi_types == "LAE",
        LYA_AA * (1 + vi_z),
        OII_AA * (1 + vi_z)
    ).astype(np.float64)
    # Clamp to VIRUS band
    wave_h = np.clip(wave_h, 3470, 5540)

    # HETDEX line properties
    sn_h     = np.abs(rng.lognormal(2.0, 0.6, n))
    flux_h   = rng.lognormal(1.2, 0.7, n).astype(np.float64)
    sigma_h  = rng.lognormal(0.8, 0.3, n).astype(np.float64)
    cont_h   = rng.lognormal(-0.5, 0.8, n).astype(np.float64)

    # HETDEX pipeline redshift: mostly right, some confusion
    z_hetdex = np.zeros(n)
    for i in range(n):
        zt = vi_z[i]
        dt = desi_types[i]
        if dt == "LAE":
            if rng.uniform() < 0.90:
                z_hetdex[i] = zt + rng.normal(0, 0.008)
            else:
                # Confused as OII
                z_hetdex[i] = wave_h[i] / OII_AA - 1
        elif dt == "OII":
            if rng.uniform() < 0.85:
                z_hetdex[i] = zt + rng.normal(0, 0.008)
            else:
                # Confused as LAE
                z_hetdex[i] = wave_h[i] / LYA_AA - 1
        else:
            z_hetdex[i] = zt + rng.normal(0, 0.015)

    # HETDEX source_type (what HETDEX called it)
    hetdex_type = []
    for i in range(n):
        zt_h = z_hetdex[i]
        if zt_h > 1.87:
            hetdex_type.append("lae")
        elif 0 <= zt_h <= 0.5:
            hetdex_type.append("oii")
        elif zt_h > 0.5:
            hetdex_type.append("agn")
        else:
            hetdex_type.append("none")
    hetdex_type = np.array(hetdex_type)

    # ELiXer / classifier scores (correlated with true type)
    p_conf = np.where(
        desi_types == "LAE",
        np.clip(rng.beta(7, 1.5, n), 0, 1),
        np.clip(rng.beta(1.5, 7, n), 0, 1)
    )
    p_cnn  = p_conf * rng.uniform(0.85, 1.15, n)
    p_cnn  = np.clip(p_cnn, 0, 1)
    plya   = np.where(
        desi_types == "LAE",
        np.clip(rng.beta(6, 1.5, n), 0, 1),
        np.clip(rng.beta(1.5, 6, n), 0, 1)
    )

    # DESI line measurements (correlated with HETDEX)
    wave_d  = wave_h + rng.normal(0, 0.5, n)
    flux_d  = flux_h * rng.lognormal(0, 0.15, n)
    sigma_d = sigma_h * rng.lognormal(0, 0.12, n)
    cont_d  = cont_h  * rng.lognormal(0, 0.20, n)

    # Build synthetic spectra
    N_PIX_H = 1036
    N_PIX_D = 7781  # DESI covers 3600–9800 AA at 0.8 AA/px
    hetdex_wave_grid = np.linspace(3470, 5540, N_PIX_H)
    desi_wave_grid   = np.linspace(3600, 9824, N_PIX_D)

    hetdex_spec = np.zeros((n, N_PIX_H), dtype=np.float32)
    hetdex_err  = np.zeros((n, N_PIX_H), dtype=np.float32)
    desi_spec   = np.zeros((n, N_PIX_D), dtype=np.float32)
    desi_err    = np.zeros((n, N_PIX_D), dtype=np.float32)

    for i in range(n):
        # HETDEX spectrum
        cont    = float(cont_h[i])
        h_spec  = cont + rng.normal(0, max(cont*0.1, 0.02), N_PIX_H)
        sig_obs = float(sigma_h[i])
        amp     = float(flux_h[i]) / (sig_obs * np.sqrt(2*np.pi))
        h_spec += amp * np.exp(
            -0.5*((hetdex_wave_grid - wave_h[i]) / sig_obs)**2)
        hetdex_spec[i] = h_spec.astype(np.float32)
        hetdex_err[i]  = (np.abs(h_spec)*0.12 + 0.01*cont).astype(np.float32)

        # DESI spectrum (wider wavelength range)
        cont_d_i = float(cont_d[i]) * 0.5
        d_spec   = cont_d_i + rng.normal(0, max(cont_d_i*0.08, 0.01), N_PIX_D)
        sig_d    = float(sigma_d[i])
        amp_d    = float(flux_d[i]) / (sig_d * np.sqrt(2*np.pi))
        d_spec  += amp_d * np.exp(
            -0.5*((desi_wave_grid - wave_d[i]) / sig_d)**2)
        desi_spec[i] = d_spec.astype(np.float32)
        desi_err[i]  = (np.abs(d_spec)*0.08 + 0.005*cont_d_i).astype(np.float32)

    info = Table({
        "TARGETID"        : np.arange(39_000_000_000, 39_000_000_000+n, dtype=np.int64),
        "TARGET_RA"       : rng.uniform(130, 235, n).astype(np.float64),
        "TARGET_DEC"      : rng.uniform(42,   58, n).astype(np.float64),
        "TILEID"          : rng.choice([80869, 80870], n).astype(np.int64),
        "RA_HETDEX"       : rng.uniform(130, 235, n).astype(np.float64),
        "DEC_HETDEX"      : rng.uniform(42,   58, n).astype(np.float64),
        "SEP"             : rng.exponential(0.5, n).astype(np.float64),
        "DETECTID"        : np.arange(2_100_000_000,
                                       2_100_000_000+n, dtype=np.int64),
        "VI_Z"            : vi_z.astype(np.float64),
        "VI_QUALITY"      : vi_quality.astype(np.int64),
        "SOURCE_TYPE"     : desi_types,
        "DEX_FLAG"        : rng.integers(0, 4, n).astype(np.int64),
        "Z_BEST_0PT5"     : z_hetdex.astype(np.float64),
        "Z_BEST_0PT4"     : (z_hetdex + rng.normal(0,0.002,n)).astype(np.float64),
        "Z_BEST_0PT3"     : (z_hetdex + rng.normal(0,0.004,n)).astype(np.float64),
        "AV"              : rng.exponential(0.2, n).astype(np.float64),
        "GMAG"            : rng.uniform(18, 25, n).astype(np.float64),
        "SN_HETDEX"       : sn_h.astype(np.float64),
        "WAVE_HETDEX"     : wave_h,
        "WAVE_ERR_HETDEX" : (wave_h * 0.001).astype(np.float64),
        "FLUX_HETDEX"     : flux_h,
        "FLUX_ERR_HETDEX" : (flux_h * 0.15).astype(np.float64),
        "SIGMA_HETDEX"    : sigma_h,
        "SIGMA_ERR_HETDEX": (sigma_h * 0.10).astype(np.float64),
        "CONT_HETDEX"     : cont_h,
        "CONT_ERR_HETDEX" : (cont_h * 0.12).astype(np.float64),
        "WAVE_DESI"       : wave_d.astype(np.float64),
        "WAVE_ERR_DESI"   : np.full(n, 0.3),
        "FLUX_DESI"       : flux_d.astype(np.float64),
        "FLUX_ERR_DESI"   : (flux_d * 0.10).astype(np.float64),
        "SIGMA_DESI"      : sigma_d.astype(np.float64),
        "SIGMA_ERR_DESI"  : (sigma_d * 0.08).astype(np.float64),
        "CONT_DESI"       : cont_d.astype(np.float64),
        "CONT_ERR_DESI"   : (cont_d * 0.10).astype(np.float64),
        # Extra columns from detinfo (added during join step)
        "p_conf"          : p_conf.astype(np.float32),
        "p_cnn"           : p_cnn.astype(np.float32),
        "plya_classification": plya.astype(np.float32),
        "source_type_hetdex" : hetdex_type,
    })
    return (info,
            hetdex_wave_grid.astype(np.float32),
            hetdex_spec, hetdex_err,
            desi_wave_grid.astype(np.float32),
            desi_spec, desi_err)

# =============================================================================
# CELL 5 — LOAD VAC
# =============================================================================

def getcol(tab, *cands, required=True):
    lc = {c.lower(): c for c in tab.colnames}
    for c in cands:
        if c.lower() in lc:
            return lc[c.lower()]
    if required:
        raise KeyError(f"None of {cands} found. Have: {list(tab.colnames)[:25]}")
    return None

print("Loading DESI × HETDEX VAC ...")
try:
    hdul = fits.open(VAC_PATH)
    # Print HDU list so user can verify structure
    print("  HDU structure:")
    for i, h in enumerate(hdul):
        print(f"    [{i}] {h.name:<22} {type(h).__name__:<16} "
              f"{h.data.shape if h.data is not None else '()'}")

    # HDU indices per Landriau+2025 Table 5 and confirmed VAC docs:
    #  0 PRIMARY  1 INFO  2 HETDEX_WAVE  3 HETDEX_SPEC  4 HETDEX_SPEC_ERR
    #  5 DESI_WAVE  6 DESI_WAVE_VACUUM  7 DESI_SPEC  8 DESI_SPEC_ERR
    # Resolve by EXTNAME in case order differs in a future version
    extnames = {h.name.upper(): i for i, h in enumerate(hdul)}
    def hdu_idx(name, fallback):
        return extnames.get(name.upper(), fallback)

    info        = Table(hdul[1].data)
    h_wave_grid = np.array(
        hdul[hdu_idx("HETDEX_WAVE",    2)].data, dtype=np.float32).ravel()
    h_spec_all  = np.array(
        hdul[hdu_idx("HETDEX_SPEC",    3)].data, dtype=np.float32)
    h_err_all   = np.array(
        hdul[hdu_idx("HETDEX_SPEC_ERR",4)].data, dtype=np.float32)
    d_wave_grid = np.array(
        hdul[hdu_idx("DESI_WAVE",      5)].data, dtype=np.float32).ravel()
    d_spec_all  = np.array(
        hdul[hdu_idx("DESI_SPEC",      7)].data, dtype=np.float32)
    d_err_all   = np.array(
        hdul[hdu_idx("DESI_SPEC_ERR",  8)].data, dtype=np.float32)
    hdul.close()
    SYNTHETIC = False
    print(f"  {len(info):,} sources, {len(info.colnames)} columns")
    print(f"  HETDEX spec shape: {h_spec_all.shape}")
    print(f"  DESI   spec shape: {d_spec_all.shape}")
    print(f"  Columns: {list(info.colnames)}")
except FileNotFoundError:
    print(f"  '{VAC_PATH}' not found — using synthetic demo data.")
    (info, h_wave_grid, h_spec_all, h_err_all,
     d_wave_grid, d_spec_all, d_err_all) = make_synthetic_vac()
    SYNTHETIC = True

# ── Normalise column names to upper-case for consistency ─────────────────────
info.rename_columns(info.colnames, [c.upper() for c in info.colnames])
n_total = len(info)
print(f"\nVAC: {n_total:,} total sources")

# =============================================================================
# CELL 6 — JOIN SC2 DETINFO CLASSIFIER SCORES
# =============================================================================

# The VAC already has HETDEX pipeline redshifts (Z_BEST_0PT5 etc.) but
# not p_conf, p_cnn, or plya_classification — join from detinfo on DETECTID.

pconf_arr  = np.full(n_total, np.nan, dtype=np.float32)
pcnn_arr   = np.full(n_total, np.nan, dtype=np.float32)
plya_arr   = np.full(n_total, np.nan, dtype=np.float32)
httype_arr = np.full(n_total, "none", dtype="U10")

if not SYNTHETIC:
    try:
        print(f"\nJoining detinfo classifier scores from {DETINFO_PATH} ...")
        det_hdul = fits.open(DETINFO_PATH, memmap=True)
        det_tab  = Table(det_hdul[1].data)
        det_hdul.close()
        det_tab.rename_columns(det_tab.colnames,
                               [c.lower() for c in det_tab.colnames])

        # Build detectid lookup dict
        det_did   = np.array(det_tab["detectid"],           dtype=np.int64)
        det_pc    = np.array(det_tab["p_conf"],             dtype=np.float32)
        det_pn    = np.array(det_tab["p_cnn"],              dtype=np.float32)
        det_plya  = np.array(det_tab["plya_classification"],dtype=np.float32)
        det_stype = np.array([s.strip().lower()
                               for s in det_tab["source_type"]], dtype=str)
        did_to_idx = {int(d): i for i, d in enumerate(det_did)}

        vac_did = np.array(info["DETECTID"], dtype=np.int64)
        n_joined = 0
        for j, did in enumerate(vac_did):
            idx = did_to_idx.get(int(did))
            if idx is not None:
                pconf_arr[j]  = det_pc[idx]
                pcnn_arr[j]   = det_pn[idx]
                plya_arr[j]   = det_plya[idx]
                httype_arr[j] = det_stype[idx]
                n_joined += 1
        print(f"  Joined {n_joined:,} / {n_total:,} sources")
    except FileNotFoundError:
        print(f"  '{DETINFO_PATH}' not found — classifier columns will be NaN")
else:
    # Synthetic already has these columns
    pconf_arr  = np.array(info["P_CONF"],               dtype=np.float32)
    pcnn_arr   = np.array(info["P_CNN"],                dtype=np.float32)
    plya_arr   = np.array(info["PLYA_CLASSIFICATION"], dtype=np.float32)
    httype_arr = np.array([s.strip().lower()
                            for s in info["SOURCE_TYPE_HETDEX"]], dtype=str)

# =============================================================================
# CELL 7 — QUALITY FILTER & DERIVED QUANTITIES
# =============================================================================

vi_z      = np.array(info["VI_Z"],        dtype=float)
vi_qual   = np.array(info["VI_QUALITY"],  dtype=int)
desi_type = np.array([s.strip().upper()   for s in info["SOURCE_TYPE"]], dtype=str)
sn_h      = np.array(info["SN_HETDEX"],   dtype=float)
sep       = np.array(info["SEP"],         dtype=float)
z_h5      = np.array(info["Z_BEST_0PT5"], dtype=float)
z_h4      = np.array(info["Z_BEST_0PT4"], dtype=float)
z_h3      = np.array(info["Z_BEST_0PT3"], dtype=float)
wave_h    = np.array(info["WAVE_HETDEX"], dtype=float)
flux_h    = np.array(info["FLUX_HETDEX"], dtype=float)
flux_d    = np.array(info["FLUX_DESI"],   dtype=float)
sigma_h   = np.array(info["SIGMA_HETDEX"],dtype=float)
sigma_d   = np.array(info["SIGMA_DESI"],  dtype=float)

for arr in [vi_z, sn_h, sep, z_h5, flux_h, flux_d, sigma_h, sigma_d]:
    arr[arr == BAD] = np.nan

# Quality mask
qmask = (
    (vi_qual >= MIN_VI_QUALITY) &
    (sn_h    >= MIN_SN_HETDEX)  &
    (sep      < MAX_SEP_ARCSEC) &
    np.isfinite(vi_z) & np.isfinite(z_h5)
)

print(f"\nQuality mask  (VI_QUAL≥{MIN_VI_QUALITY}, "
      f"SN≥{MIN_SN_HETDEX}, sep<{MAX_SEP_ARCSEC}\"): "
      f"{qmask.sum():,} / {n_total:,}")

# Δz quantities (using p(Lya)≥0.5 threshold as default)
dz       = z_h5 - vi_z                         # signed offset
dz_abs   = np.abs(dz)
catas    = dz_abs > DZ_CATAS
close    = dz_abs < DZ_CLOSE
dz_norm  = dz / (1 + vi_z)                     # normalised Δz

print(f"\nRedshift statistics (quality-filtered, N={qmask.sum():,}):")
dz_q = dz[qmask]
print(f"  Bias  ⟨Δz⟩          : {np.nanmean(dz_q):+.5f}")
print(f"  σ(Δz)               : {np.nanstd(dz_q):.5f}")
print(f"  Catastrophic (>{DZ_CATAS}): "
      f"{catas[qmask].sum():,}  "
      f"({100*catas[qmask].mean():.1f}%)")
print(f"  Excellent   (<{DZ_CLOSE}): "
      f"{close[qmask].sum():,}  "
      f"({100*close[qmask].mean():.1f}%)")

# =============================================================================
# CELL 8 — CONFUSION ANALYSIS
# =============================================================================

# Map HETDEX type → comparable DESI type for confusion matrix
HTDEX_NORM = {"lae":"LAE","oii":"OII","agn":"AGN",
              "star":"STAR","lzg":"OTHER","none":"OTHER","cont":"OTHER"}
ht_norm = np.array([HTDEX_NORM.get(s, "OTHER") for s in httype_arr])

CLASSES = ["LAE","OII","AGN","OTHER"]
n_cls   = len(CLASSES)

# Confusion matrix (HETDEX rows, DESI columns)
cm = np.zeros((n_cls, n_cls), dtype=int)
for i, h_cls in enumerate(CLASSES):
    for j, d_cls in enumerate(CLASSES):
        sel = qmask & (ht_norm == h_cls) & (desi_type == d_cls)
        cm[i, j] = sel.sum()

# Per-class statistics
print("\nConfusion matrix  (HETDEX → DESI):")
print(f"  {'HETDEX\\DESI':>14}", end="")
for d in CLASSES:
    print(f"  {d:>8}", end="")
print(f"  {'TOTAL':>8}")
for i, h in enumerate(CLASSES):
    row_tot = cm[i].sum()
    print(f"  {h:>14}", end="")
    for j in range(n_cls):
        print(f"  {cm[i,j]:>8,}", end="")
    print(f"  {row_tot:>8,}")

# LAE/OII confusion rates
lae_as_oii = cm[CLASSES.index("LAE"), CLASSES.index("OII")]
oii_as_lae = cm[CLASSES.index("OII"), CLASSES.index("LAE")]
n_lae_total = cm[CLASSES.index("LAE")].sum()
n_oii_total = cm[CLASSES.index("OII")].sum()
print(f"\n  LAE→OII confusion : {lae_as_oii:,} / {n_lae_total:,}  "
      f"({100*lae_as_oii/max(n_lae_total,1):.1f}%)")
print(f"  OII→LAE confusion : {oii_as_lae:,} / {n_oii_total:,}  "
      f"({100*oii_as_lae/max(n_oii_total,1):.1f}%)")

# =============================================================================
# CELL 9 — p_conf CALIBRATION CURVES
# =============================================================================

def calibration_curve(scores, true_labels, bins):
    """
    Compute reliability diagram: mean(score) and fraction_positive per bin.
    Returns (bin_centres, mean_score, frac_pos, n_per_bin).
    """
    bin_centres, frac_pos, n_per_bin = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi) & np.isfinite(scores)
        if mask.sum() < 3:
            continue
        mean_s = float(np.nanmean(scores[mask]))
        f_pos  = float(true_labels[mask].mean())
        bin_centres.append(mean_s)
        frac_pos.append(f_pos)
        n_per_bin.append(int(mask.sum()))
    return (np.array(bin_centres), np.array(frac_pos),
            np.array(n_per_bin, dtype=int))

# True positive = DESI classified as LAE
true_lae = (desi_type == "LAE") & qmask & np.isfinite(pconf_arr)

cal_pc  = calibration_curve(pconf_arr[qmask & np.isfinite(pconf_arr)],
                              true_lae[qmask & np.isfinite(pconf_arr)],
                              PCONF_BINS)
cal_pn  = calibration_curve(pcnn_arr[qmask & np.isfinite(pcnn_arr)],
                              true_lae[qmask & np.isfinite(pcnn_arr)],
                              PCONF_BINS)
cal_pl  = calibration_curve(plya_arr[qmask & np.isfinite(plya_arr)],
                              true_lae[qmask & np.isfinite(plya_arr)],
                              PCONF_BINS)

# =============================================================================
# CELL 10 — AMBIGUOUS SOURCE CATALOGUE
# =============================================================================

disagree = qmask & (ht_norm != desi_type)
df_ambig = pd.DataFrame({
    "detectid"       : np.array(info["DETECTID"], dtype=np.int64)[disagree],
    "ra_hetdex"      : np.array(info["RA_HETDEX"],  dtype=float)[disagree],
    "dec_hetdex"     : np.array(info["DEC_HETDEX"], dtype=float)[disagree],
    "hetdex_type"    : httype_arr[disagree],
    "desi_type"      : desi_type[disagree],
    "vi_z"           : vi_z[disagree],
    "z_hetdex_0pt5"  : z_h5[disagree],
    "delta_z"        : dz[disagree],
    "sn_hetdex"      : sn_h[disagree],
    "wave_hetdex"    : wave_h[disagree],
    "flux_hetdex"    : flux_h[disagree],
    "flux_desi"      : flux_d[disagree],
    "p_conf"         : pconf_arr[disagree],
    "p_cnn"          : pcnn_arr[disagree],
    "plya_class"     : plya_arr[disagree],
    "vi_quality"     : vi_qual[disagree],
    "sep_arcsec"     : sep[disagree],
}).sort_values("sn_hetdex", ascending=False).reset_index(drop=True)

if CSV_PATH:
    df_ambig.to_csv(CSV_PATH, index=False, float_format="%.5f")
    print(f"\nAmbiguous catalog -> {CSV_PATH}  ({len(df_ambig):,} sources)")

# =============================================================================
# CELL 11 — PLOTTING
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

DESI_TYPE_COL = {"LAE":"#58a6ff","OII":"#3fb950",
                 "AGN":"#f78166","OTHER":"#8b949e","STAR":"#d2a8ff"}

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

fig = plt.figure(figsize=(18, 15))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(
    3, 3, figure=fig,
    hspace=0.42, wspace=0.30,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.06,
)

ax_zdiff  = fig.add_subplot(gs[0, 0])   # z_hetdex vs vi_z
ax_cm     = fig.add_subplot(gs[0, 1])   # confusion matrix heatmap
ax_cal    = fig.add_subplot(gs[0, 2])   # calibration curves
ax_flux   = fig.add_subplot(gs[1, 0])   # flux comparison
ax_sigma  = fig.add_subplot(gs[1, 1])   # σ comparison
ax_thresh = fig.add_subplot(gs[1, 2])   # accuracy vs p_conf threshold
ax_dzq    = fig.add_subplot(gs[2, 0])   # Δz histogram by DESI type
ax_spec1  = fig.add_subplot(gs[2, 1])   # representative LAE spectrum pair
ax_spec2  = fig.add_subplot(gs[2, 2])   # representative disagreement pair

# ── Panel 1: z_hetdex vs vi_z ─────────────────────────────────────────────────
style_ax(ax_zdiff,
         r"Redshift: $z_{\rm HETDEX}$ vs $z_{\rm DESI\,VI}$",
         r"$z_{\rm DESI\,VI}$  (ground truth)",
         r"$z_{\rm HETDEX}$  (Z\_BEST\_0PT5)")

for dtype in ["LAE","OII","AGN","OTHER"]:
    sel = qmask & (desi_type == dtype)
    if sel.sum() == 0: continue
    ax_zdiff.scatter(vi_z[sel], z_h5[sel],
                     s=4, c=DESI_TYPE_COL[dtype], alpha=0.55,
                     linewidths=0, rasterized=True,
                     label=f"{dtype}  (N={sel.sum():,})",
                     zorder=4)

z_range = np.array([max(vi_z[qmask].min()-0.1, -0.1),
                    vi_z[qmask].max()+0.1])
ax_zdiff.plot(z_range, z_range, "--", color=MUTED, lw=1.0, alpha=0.6,
              label="y = x")
ax_zdiff.plot(z_range, z_range + DZ_CATAS, ":", color="#ffa657",
              lw=0.8, alpha=0.5)
ax_zdiff.plot(z_range, z_range - DZ_CATAS, ":", color="#ffa657",
              lw=0.8, alpha=0.5, label=f"|Δz| = {DZ_CATAS}")

# Annotate statistics
bias = float(np.nanmean(dz[qmask]))
sig  = float(np.nanstd(dz[qmask]))
pct_catas = 100 * catas[qmask].mean()
ax_zdiff.text(0.03, 0.97,
              f"bias = {bias:+.5f}\nσ(Δz) = {sig:.5f}\n"
              f"catas. = {pct_catas:.1f}%",
              transform=ax_zdiff.transAxes,
              color=TEXT, fontsize=8, va="top",
              fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.35",
                        facecolor=BG, edgecolor=SPINE, alpha=0.85))
mleg(ax_zdiff, loc="lower right", markerscale=3, ncol=2)

# ── Panel 2: Confusion matrix heatmap ────────────────────────────────────────
style_ax(ax_cm, "Classification confusion  (row-normalised)",
         "DESI SOURCE_TYPE", "HETDEX source_type", minor=False)

cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
cmap_cm = plt.cm.Blues
im_cm   = ax_cm.imshow(cm_norm, cmap=cmap_cm, vmin=0, vmax=1, aspect="auto")
for i in range(n_cls):
    for j in range(n_cls):
        val = cm_norm[i, j]
        raw = cm[i, j]
        col = "black" if val > 0.55 else TEXT
        ax_cm.text(j, i, f"{val:.2f}\n({raw:,})",
                   ha="center", va="center",
                   fontsize=8, color=col,
                   fontweight="bold" if i == j else "normal")

ax_cm.set_xticks(range(n_cls))
ax_cm.set_yticks(range(n_cls))
ax_cm.set_xticklabels(CLASSES, color=TEXT, fontsize=9)
ax_cm.set_yticklabels(CLASSES, color=TEXT, fontsize=9)
for tick, cls in zip(ax_cm.get_xticklabels(), CLASSES):
    tick.set_color(DESI_TYPE_COL.get(cls, TEXT))
for tick, cls in zip(ax_cm.get_yticklabels(), CLASSES):
    tick.set_color(DESI_TYPE_COL.get(cls, TEXT))

cb_cm = fig.colorbar(im_cm, ax=ax_cm, fraction=0.046, pad=0.04)
cb_cm.set_label("Row fraction", color=MUTED, fontsize=8)
cb_cm.ax.yaxis.set_tick_params(color=MUTED, labelsize=7.5)
plt.setp(cb_cm.ax.yaxis.get_ticklabels(), color=MUTED)
cb_cm.outline.set_edgecolor(SPINE)

# ── Panel 3: Calibration curves ───────────────────────────────────────────────
style_ax(ax_cal,
         "Classifier reliability  (DESI VI = ground truth)",
         "Classifier score",
         "P(DESI = LAE | score bin)")

for (bc, fp, nb), label, color in [
    (cal_pc, "p_conf  (RF)",     "#58a6ff"),
    (cal_pn, "p_cnn   (CNN)",    "#3fb950"),
    (cal_pl, "plya_class (ELiXer)","#ffa657"),
]:
    if len(bc) == 0: continue
    ax_cal.errorbar(bc, fp,
                    yerr=np.sqrt(fp*(1-fp)/np.maximum(nb,1)),
                    fmt="o-", color=color, ms=6, lw=1.5,
                    capsize=3, elinewidth=1.0, label=label)
    for x, y, n_b in zip(bc, fp, nb):
        ax_cal.text(x, y + 0.03, str(n_b),
                    ha="center", color=color, fontsize=6, alpha=0.7)

ax_cal.plot([0,1],[0,1],"--", color=MUTED, lw=0.9, alpha=0.55,
            label="Perfect calibration")
ax_cal.set_xlim(-0.02, 1.02)
ax_cal.set_ylim(-0.05, 1.05)
mleg(ax_cal, loc="upper left")

# ── Panel 4: Flux comparison ──────────────────────────────────────────────────
style_ax(ax_flux,
         "Line flux: HETDEX vs DESI",
         r"$F_{\rm HETDEX}$  [10$^{-17}$ cgs]",
         r"$F_{\rm DESI}$    [10$^{-17}$ cgs]")

for dtype in ["LAE","OII","AGN"]:
    sel = qmask & (desi_type == dtype) & np.isfinite(flux_h) & np.isfinite(flux_d)
    if sel.sum() < 3: continue
    ax_flux.scatter(flux_h[sel], flux_d[sel],
                    s=5, c=DESI_TYPE_COL[dtype], alpha=0.55,
                    linewidths=0, rasterized=True,
                    label=f"{dtype} (N={sel.sum():,})", zorder=4)

# 1:1 line + median ratio
fsel = qmask & np.isfinite(flux_h) & np.isfinite(flux_d) & (flux_h > 0)
if fsel.sum() > 5:
    f_range = np.array([np.nanpercentile(flux_h[fsel],1),
                        np.nanpercentile(flux_h[fsel],99)])
    ax_flux.plot(f_range, f_range, "--", color=MUTED, lw=0.9, alpha=0.6,
                 label="y = x")
    ratio = np.nanmedian(flux_d[fsel] / flux_h[fsel])
    ax_flux.text(0.03, 0.97,
                 f"median F_DESI/F_HETDEX = {ratio:.3f}",
                 transform=ax_flux.transAxes,
                 color=TEXT, fontsize=8, va="top",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=BG, edgecolor=SPINE, alpha=0.8))
    ax_flux.set_xscale("log"); ax_flux.set_yscale("log")
mleg(ax_flux, loc="lower right", markerscale=3)

# ── Panel 5: σ linewidth comparison ──────────────────────────────────────────
style_ax(ax_sigma,
         "Line width σ: HETDEX vs DESI",
         r"$\sigma_{\rm HETDEX}$  (Å)",
         r"$\sigma_{\rm DESI}$    (Å)")

for dtype in ["LAE","OII","AGN"]:
    sel = qmask & (desi_type == dtype) & np.isfinite(sigma_h) & np.isfinite(sigma_d)
    if sel.sum() < 3: continue
    ax_sigma.scatter(sigma_h[sel], sigma_d[sel],
                     s=5, c=DESI_TYPE_COL[dtype], alpha=0.55,
                     linewidths=0, rasterized=True,
                     label=f"{dtype} (N={sel.sum():,})", zorder=4)

sigs = qmask & np.isfinite(sigma_h) & np.isfinite(sigma_d)
if sigs.sum() > 3:
    s_range = np.array([np.nanpercentile(sigma_h[sigs],1),
                        np.nanpercentile(sigma_h[sigs],99)])
    ax_sigma.plot(s_range, s_range, "--", color=MUTED, lw=0.9, alpha=0.6)
    ax_sigma.set_xscale("log"); ax_sigma.set_yscale("log")
mleg(ax_sigma, loc="lower right", markerscale=3)

# ── Panel 6: Accuracy vs p_conf threshold ────────────────────────────────────
style_ax(ax_thresh,
         "LAE purity & completeness vs p_conf threshold",
         "p_conf threshold",
         "Fraction")

thresholds = np.linspace(0.0, 0.95, 50)
purity, completeness, n_above = [], [], []
lae_mask_q = qmask & np.isfinite(pconf_arr)
true_lae_q = (desi_type == "LAE") & lae_mask_q

for t in thresholds:
    above   = lae_mask_q & (pconf_arr >= t)
    if above.sum() == 0:
        purity.append(np.nan); completeness.append(np.nan)
        n_above.append(0); continue
    purity.append(float((true_lae_q & above).sum() / above.sum()))
    completeness.append(float(
        (true_lae_q & above).sum() / max(true_lae_q.sum(), 1)))
    n_above.append(int(above.sum()))

ax_thresh.plot(thresholds, purity,
               color="#58a6ff", lw=1.8, label="Purity  P(LAE|pred)")
ax_thresh.plot(thresholds, completeness,
               color="#3fb950", lw=1.8, label="Completeness")
ax_thresh.axhline(0.98, color="#ffa657", lw=0.7, ls=":",
                  alpha=0.55, label="HETDEX req: 98% purity")
ax_thresh.axhline(0.96, color="#3fb950", lw=0.7, ls=":",
                  alpha=0.55, label="HETDEX req: 96% completeness")
ax_thresh.axvline(0.5, color=MUTED, lw=0.9, ls="--", alpha=0.6,
                  label="Default threshold (0.5)")
ax_thresh.set_xlim(0, 1); ax_thresh.set_ylim(0, 1.05)

ax_thresh_r = ax_thresh.twinx()
ax_thresh_r.plot(thresholds, n_above, color="#ffa657",
                 lw=1.2, ls=":", alpha=0.65)
ax_thresh_r.set_ylabel("N above threshold", color="#ffa657", fontsize=8.5)
ax_thresh_r.tick_params(colors="#ffa657", labelsize=8)
ax_thresh_r.spines["right"].set_color(SPINE)
mleg(ax_thresh, loc="upper left")

# ── Panel 7: Δz histogram by DESI type ───────────────────────────────────────
style_ax(ax_dzq,
         r"$\Delta z = z_{\rm HETDEX} - z_{\rm VI}$ distribution",
         r"$\Delta z$", "Normalised density")

dz_bins = np.linspace(-0.5, 0.5, 80)
for dtype in ["LAE","OII","AGN"]:
    sel = qmask & (desi_type == dtype) & np.isfinite(dz)
    if sel.sum() < 5: continue
    ax_dzq.hist(dz[sel], bins=dz_bins, density=True,
                color=DESI_TYPE_COL[dtype], alpha=0.45,
                histtype="stepfilled", label=dtype)
    ax_dzq.hist(dz[sel], bins=dz_bins, density=True,
                color=DESI_TYPE_COL[dtype], lw=1.2, histtype="step")
    med = np.nanmedian(dz[sel])
    ax_dzq.axvline(med, color=DESI_TYPE_COL[dtype],
                   lw=0.9, ls="--", alpha=0.8)

ax_dzq.axvline(0, color=MUTED, lw=0.8, ls=":", alpha=0.6)
ax_dzq.set_xlim(-0.5, 0.5)
mleg(ax_dzq, loc="upper right")

# ── Panels 8–9: Spectral overlays ────────────────────────────────────────────
def plot_spec_pair(ax, row_idx, title, zoom=None):
    """
    Plot HETDEX + DESI spectra for source at row_idx in the VAC.
    zoom: (w_lo, w_hi) observed Å to zoom into (None = full HETDEX range)
    """
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, which="both", direction="in",
                   labelsize=8, top=True, right=True)
    ax.set_title(title, color=TEXT, fontsize=8.5,
                 fontweight="bold", loc="left", pad=4)
    ax.set_xlabel(r"Observed $\lambda$ (Å)", color=TEXT, fontsize=8.5)
    ax.set_ylabel(r"$f_\lambda$ [10$^{-17}$ cgs]", color=TEXT, fontsize=8.5)

    # Orientate spec arrays: shape might be (n_src, n_pix) or (n_pix, n_src)
    def get_spec(arr, idx, n_pix):
        if arr.shape[0] == n_pix:
            return arr[:, idx].astype(float)
        return arr[idx, :].astype(float)

    h_sp = get_spec(h_spec_all, row_idx, len(h_wave_grid))
    h_er = get_spec(h_err_all,  row_idx, len(h_wave_grid))
    d_sp = get_spec(d_spec_all, row_idx, len(d_wave_grid))
    d_er = get_spec(d_err_all,  row_idx, len(d_wave_grid))

    # HETDEX
    ax.fill_between(h_wave_grid, h_sp-h_er, h_sp+h_er,
                    color="#58a6ff", alpha=0.18)
    ax.plot(h_wave_grid, h_sp, color="#58a6ff",
            lw=0.8, alpha=0.65, label="HETDEX")
    # DESI (scale to HETDEX flux range for visual clarity)
    d_scale = np.nanmedian(np.abs(h_sp)) / max(np.nanmedian(np.abs(d_sp)), 1e-10)
    ax.fill_between(d_wave_grid, (d_sp-d_er)*d_scale, (d_sp+d_er)*d_scale,
                    color="#3fb950", alpha=0.15)
    ax.plot(d_wave_grid, d_sp*d_scale, color="#3fb950",
            lw=0.8, alpha=0.65, label="DESI (rescaled)")

    if zoom:
        ax.set_xlim(*zoom)
    else:
        ax.set_xlim(h_wave_grid[0], h_wave_grid[-1])

    ax.axhline(0, color=SPINE, lw=0.6, ls=":")

    # Mark HETDEX detected wavelength
    wh = float(wave_h[row_idx])
    if np.isfinite(wh):
        ax.axvline(wh, color="#ffa657", lw=1.0, ls="--", alpha=0.7,
                   label=f"λ_HETDEX={wh:.1f}Å")
    ax.legend(fontsize=7, facecolor="#21262d",
              edgecolor=SPINE, labelcolor=TEXT,
              loc="upper right")

# Find a good LAE and a good disagreement example
lae_agree_idxs = np.where(
    qmask & (desi_type=="LAE") & (ht_norm=="LAE") &
    (dz_abs < DZ_CLOSE)
)[0]
disagree_idxs = np.where(
    qmask & (desi_type=="OII") & (ht_norm=="LAE")
)[0]

if len(lae_agree_idxs) > 0:
    ri = lae_agree_idxs[0]
    wh = float(wave_h[ri])
    zoom_lo = max(wh - 60, h_wave_grid[0])
    zoom_hi = min(wh + 60, h_wave_grid[-1])
    plot_spec_pair(ax_spec1, ri,
                   f"AGREE: LAE  VI_Z={vi_z[ri]:.4f}  z_h={z_h5[ri]:.4f}",
                   zoom=(zoom_lo, zoom_hi))
else:
    ax_spec1.text(0.5, 0.5, "No agree-LAE examples",
                  ha="center", va="center",
                  transform=ax_spec1.transAxes, color=MUTED)

if len(disagree_idxs) > 0:
    ri2 = disagree_idxs[0]
    wh2 = float(wave_h[ri2])
    plot_spec_pair(ax_spec2, ri2,
                   f"DISAGREE: HETDEX=LAE / DESI=OII\n"
                   f"VI_Z={vi_z[ri2]:.4f}  z_h={z_h5[ri2]:.4f}  "
                   f"Δz={dz[ri2]:+.4f}",
                   zoom=(max(wh2-80, h_wave_grid[0]),
                         min(wh2+80, h_wave_grid[-1])))
else:
    ax_spec2.text(0.5, 0.5, "No HETDEX=LAE/DESI=OII examples",
                  ha="center", va="center",
                  transform=ax_spec2.transAxes, color=MUTED)

# ── Super-title ───────────────────────────────────────────────────────────────
syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
fig.suptitle(
    "HETDEX SC2 × DESI DR1 — Redshift Validation & Classification Cross-check"
    + syn_tag,
    color=TEXT, fontsize=12, fontweight="bold", y=0.975,
)

if SAVE_PATH:
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 12 — FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 68)
print("  HETDEX × DESI DR1 — Validation Summary")
print("=" * 68)
q = qmask
print(f"  Sources in VAC               : {n_total:,}")
print(f"  After quality filter         : {q.sum():,}")
print(f"  LAE/OII/AGN/other (DESI VI) :")
for dtype in CLASSES:
    n_d = (desi_type[q]==dtype).sum()
    print(f"    {dtype:<8}: {n_d:>6,}  ({100*n_d/max(q.sum(),1):.1f}%)")

print(f"\n  Redshift validation (z_HETDEX vs VI_Z):")
print(f"    Bias ⟨Δz⟩         : {np.nanmean(dz[q]):+.6f}")
print(f"    Scatter σ(Δz)     : {np.nanstd(dz[q]):.6f}")
print(f"    Catastrophic rate : {100*catas[q].mean():.2f}%  "
      f"(|Δz|>{DZ_CATAS})")
print(f"    Excellent rate    : {100*close[q].mean():.2f}%  "
      f"(|Δz|<{DZ_CLOSE})")

print(f"\n  Classification accuracy:")
for i, h in enumerate(CLASSES):
    row_tot = cm[i].sum()
    correct = cm[i, i]
    if row_tot > 0:
        print(f"    HETDEX={h:<6} → DESI correct: "
              f"{correct:>5,}/{row_tot:>5,}  "
              f"({100*correct/row_tot:.1f}%)")

print(f"\n  LAE/OII confusion rates:")
print(f"    HETDEX says LAE → DESI says OII : "
      f"{lae_as_oii:,}/{n_lae_total:,}  "
      f"({100*lae_as_oii/max(n_lae_total,1):.1f}%)")
print(f"    HETDEX says OII → DESI says LAE : "
      f"{oii_as_lae:,}/{n_oii_total:,}  "
      f"({100*oii_as_lae/max(n_oii_total,1):.1f}%)")

print(f"\n  Ambiguous sources (HETDEX≠DESI) : {len(df_ambig):,}")
print("=" * 68)
