"""
hetdex_lan_science.py
=====================
Detect and analyse HETDEX Lyman-α Nebulae (LANs) — the tens of thousands
of gigantic hydrogen gas halos surrounding galaxies 10–12 billion years ago.

Scientific context
------------------
HETDEX identified the largest statistical census of Lyman-α Nebulae (LANs)
spanning the full range from galaxy-scale halos (~10 kpc) to the most extended
Lyman-α blobs (>100 kpc), all at cosmic noon (z ~ 2–3.5, corresponding to
look-back times of 10–12 Gyr).

These halos trace:
  - CGM gas reservoirs feeding star formation
  - Outflowing gas driven by stellar feedback and AGN
  - Large-scale structure at the epoch of peak star formation
  - The connection between compact LAEs and extended blobs

This script performs:
  1. DETECTION: Select bona-fide resolved LANs from the LAN catalog using
     morphological quality cuts (dBIC, flag_resolved, iso_rel_err)
  2. POPULATION: Size–luminosity diagram, size distribution, redshift evolution
  3. MORPHOLOGY: r_iso vs r_s (profile shape), ΔBIC morphology classifier
  4. COSMIC EVOLUTION: r_iso(z), logL_lya(z), EW(z) across 10–12 Gyr
  5. AGN vs LAE HOST: Compare nebula properties between host types
  6. BLOBS: Identify the most extended LANs (r_iso > 50 kpc — true "blobs")
  7. ENVIRONMENT: Cross-match with SC2 LAE catalog for clustering context
  8. PUBLICATION FIGURE: Single-panel summary suitable for press/outreach

Physical definitions
--------------------
  r_iso    : Isophotal radius where SB falls below 1σ sky — the "edge"
             of the detectable nebula. Proper kpc, Planck18 cosmology.
  r_s      : Exponential scale length of the SB profile — the "core" size.
  dBIC     : ΔBIC = BIC(PSF) − BIC(extended). Positive → extended wins.
             dBIC > 2   marginally resolved
             dBIC > 6   strongly resolved
  logL_lya : log10(L_Lya / erg/s)
  EW_rest  : combined_eqw_rest_lya  [Å]

Catalog: hetdex_lan_v0.3.fits  (Mentuch Cooper et al. 2026, ApJ 1000, 38)

Requirements
------------
  pip install astropy numpy matplotlib scipy pandas

Data
----
  hetdex_lan_v0.3.fits
  https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/hetdex_source_catalog_2/
"""

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

LAN_PATH     = "hetdex_lan_v0.3.fits"
SAVE_PATH    = "hetdex_lan_science.png"
OUTREACH_PATH= "hetdex_lan_outreach.png"
CSV_PATH     = "hetdex_lan_catalogue.csv"

# Quality cuts for bona-fide resolved LANs
MIN_DBIC       =  2.0    # dBIC > 2 → marginally resolved (use 6 for strong)
DBIC_STRONG    =  6.0    # dBIC > 6 → strongly resolved
MIN_LOGL       = 42.0    # log L_Lya > 42.0 erg/s
MAX_ISO_REL_ERR=  1.0    # relative error on r_iso < 100%
MIN_R_ISO      =  5.0    # r_iso > 5 kpc (exclude point sources)

# Blob threshold (Steidel+11, Prescott+12 definition)
BLOB_R_ISO     = 50.0    # r_iso > 50 kpc  → "Lyman-α blob"

# Redshift range for cosmic-noon science
Z_MIN, Z_MAX   = 1.87, 3.52   # HETDEX Lyα window

# Cosmology
BAD = -999.0

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
import matplotlib.colors as mcolors
from matplotlib.ticker  import AutoMinorLocator, LogLocator, LogFormatter
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines   import Line2D
from scipy.stats        import binned_statistic, pearsonr
from scipy.optimize     import curve_fit
from scipy.ndimage      import gaussian_filter

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

def make_synthetic_lan(n=70_000, seed=77):
    """
    Synthetic LAN catalog matching the real hetdex_lan_v0.3.fits structure.
    Physical relations built in:
      r_iso ∝ L^0.24  (observed size–luminosity slope)
      r_s   = r_iso / rng(3, 6)  (r_iso / r_s ~ 4 typical)
      dBIC  : resolved LANs get high dBIC; PSF-like get low/negative
      z range: 1.87–3.52 (HETDEX Lyα window, ~60% LAE / 40% AGN)
    """
    rng  = np.random.default_rng(seed)

    # Source type: ~60% LAE, ~40% AGN (matching real catalog from previous run)
    stype = np.where(rng.uniform(size=n) < 0.60, "lae", "agn")

    # Redshift distribution
    z     = rng.uniform(Z_MIN, Z_MAX, n).astype(np.float32)

    # logL: AGN slightly brighter at high L
    logL  = np.where(
        stype == "agn",
        np.clip(rng.lognormal(np.log(43.0), 0.5, n), 42.0, 45.5),
        np.clip(rng.lognormal(np.log(42.8), 0.45, n), 41.5, 44.5)
    ).astype(np.float32)
    logL_err = (rng.uniform(0.02, 0.15, n)).astype(np.float32)

    # r_iso: power-law r_iso ∝ L^0.24, scatter ±0.25 dex
    log_r_iso = 0.24 * (logL - 43.0) + np.log10(18.0) + rng.normal(0, 0.25, n)
    r_iso     = np.clip(10.**log_r_iso, 2.0, 250.0).astype(np.float32)

    # r_s: 3–6× smaller than r_iso
    r_s_factor = rng.uniform(3.0, 6.0, n)
    r_s         = (r_iso / r_s_factor).astype(np.float32)
    r_s_err     = (r_s * rng.uniform(0.05, 0.25, n)).astype(np.float32)

    # dBIC: resolved → high dBIC; scale with log r_iso
    log_r_norm = np.log10(np.clip(r_iso, 1, None)) - 1.0
    dBIC_true  = 20.0 * log_r_norm + rng.normal(0, 5, n)
    dBIC       = dBIC_true.astype(np.float32)

    # flag_resolved: 1 if dBIC > 2
    flag_resolved = (dBIC > 2).astype(np.int64)

    # chi2 values
    chi2_ext = np.clip(rng.lognormal(0.1, 0.3, n), 0.3, 5.0).astype(np.float32)
    chi2_psf = (chi2_ext + dBIC / 20.0 * rng.uniform(0.5, 1.5, n)).astype(np.float32)

    # log10_pF: probability favoring extended
    log10_pF = np.clip(-0.3 * chi2_ext + 0.5 * log_r_norm +
                        rng.normal(0, 0.4, n), -5, 2).astype(np.float32)

    # iso_rel_err: larger for small/faint
    iso_rel_err = np.clip(
        0.5 / np.clip(r_iso / 10.0, 0.1, None) + rng.exponential(0.2, n),
        0.02, 3.0
    ).astype(np.float32)

    # Surface brightness sensitivity
    SB_1sigma = rng.lognormal(np.log(1.5), 0.3, n).astype(np.float32)

    # Area
    area_iso = (np.pi * r_iso**2 / cosmo.arcsec_per_kpc_proper(
        np.clip(z.astype(float), 0.1, 5)).value**2  # arcsec²
    ).astype(np.float32)
    area_circ = area_iso.copy()

    # Flux
    flux_lya = (10.**(logL - 43.0) * 1e-17).astype(np.float32)
    flux_err  = (flux_lya * rng.uniform(0.05, 0.20, n)).astype(np.float32)

    # EW_rest: anti-correlated with L (Ando+06)
    ew_rest = np.clip(
        300.0 * 10.**(-0.4 * (logL - 42.5)) * rng.lognormal(0, 0.3, n),
        5.0, 500.0
    ).astype(np.float32)

    # Magnitudes
    gmag        = (23.0 - 2.5*(logL - 43.0) + rng.normal(0, 0.8, n)).astype(np.float32)
    hsc_r_mag   = (gmag + rng.normal(0, 0.4, n)).astype(np.float32)
    hsc_r_err   = rng.uniform(0.05, 0.3, n).astype(np.float32)

    # detectid and positions
    detectid = np.arange(2_100_000_000, 2_100_000_000 + n, dtype=np.int64)
    names    = np.array([f"HLAN+{d}" for d in detectid], dtype="U20")
    ra       = rng.uniform(130, 235, n).astype(np.float32)
    dec      = rng.uniform(42,   58, n).astype(np.float32)
    field    = rng.choice(
        ["dex-spring","dex-fall","cosmos","goods-n","nep","ssa22"],
        n, p=[0.55, 0.30, 0.08, 0.03, 0.03, 0.01]
    )
    dups     = np.full(n, "", dtype="U60")

    tab = Table({
        "name"             : names,
        "ra"               : ra, "dec": dec,
        "source_type"      : stype,
        "z_hetdex"         : z,
        "z_hetdex_src"     : np.full(n, "hetdex", dtype="U8"),
        "detectid"         : detectid,
        "shotid"           : detectid - 100_000,
        "field"            : field,
        "SB_1sigma_obs"    : SB_1sigma,
        "r_iso"            : r_iso,
        "r_s"              : r_s,
        "r_s_err"          : r_s_err,
        "area_iso_2sigma"  : area_iso,
        "area_r_iso_circ"  : area_circ,
        "logL_lya"         : logL,
        "logL_lya_err"     : logL_err,
        "flux_lya"         : flux_lya,
        "flux_lya_err"     : flux_err,
        "gmag"             : gmag,
        "HSC-r_mag"        : hsc_r_mag,
        "HSC-r_mag_err"    : hsc_r_err,
        "combined_eqw_rest_lya": ew_rest,
        "flag_resolved"    : flag_resolved,
        "chi2_ext_reduced" : chi2_ext,
        "chi2_psf_reduced" : chi2_psf,
        "log10_pF"         : log10_pF,
        "dBIC"             : dBIC,
        "iso_rel_err"      : iso_rel_err,
        "dups_detectid"    : dups,
    })
    print(f"  Synthetic: {n:,} LAN candidates")
    return tab


# =============================================================================
# CELL 4 — LOAD LAN CATALOG
# =============================================================================

def getcol(tab, *cands):
    lc = {c.lower().replace("-","_"): c for c in tab.colnames}
    for c in cands:
        if c.lower().replace("-","_") in lc:
            return lc[c.lower().replace("-","_")]
    raise KeyError(f"None of {cands}. Have: {list(tab.colnames)[:25]}")

print("Loading LAN catalog ...")
try:
    hdul = fits.open(LAN_PATH, memmap=True)
    lan  = Table(hdul[1].data)
    hdul.close()
    lan.rename_columns(lan.colnames, [c.lower() for c in lan.colnames])
    print(f"  Raw catalog: {len(lan):,} rows, {len(lan.colnames)} columns")
    SYNTHETIC = False
except FileNotFoundError:
    print(f"  '{LAN_PATH}' not found — using synthetic demo data.")
    lan       = make_synthetic_lan()
    lan.rename_columns(lan.colnames, [c.lower() for c in lan.colnames])
    SYNTHETIC = True

# =============================================================================
# CELL 5 — EXTRACT ARRAYS & DEDUPLICATE
# =============================================================================

NAME_COL   = getcol(lan, "name")
RA_COL     = getcol(lan, "ra")
DEC_COL    = getcol(lan, "dec")
STYPE_COL  = getcol(lan, "source_type")
Z_COL      = getcol(lan, "z_hetdex")
RIISO_COL  = getcol(lan, "r_iso")
RS_COL     = getcol(lan, "r_s")
RSERR_COL  = getcol(lan, "r_s_err")
LOGL_COL   = getcol(lan, "logl_lya")
LOGLERR_COL= getcol(lan, "logl_lya_err")
FLUX_COL   = getcol(lan, "flux_lya")
EW_COL     = getcol(lan, "combined_eqw_rest_lya")
DBIC_COL   = getcol(lan, "dbic")
LOGPF_COL  = getcol(lan, "log10_pf")
FRES_COL   = getcol(lan, "flag_resolved")
IREREL_COL = getcol(lan, "iso_rel_err")
CHI2E_COL  = getcol(lan, "chi2_ext_reduced")
CHI2P_COL  = getcol(lan, "chi2_psf_reduced")
SB_COL     = getcol(lan, "sb_1sigma_obs")
FIELD_COL  = getcol(lan, "field")
DID_COL    = getcol(lan, "detectid")
DUPS_COL   = getcol(lan, "dups_detectid")
GMAG_COL   = getcol(lan, "gmag")

# Extract raw arrays
ra    = np.array(lan[RA_COL],    dtype=float)
dec   = np.array(lan[DEC_COL],   dtype=float)
stype = np.array([s.strip().lower() for s in lan[STYPE_COL]])
z     = np.array(lan[Z_COL],     dtype=float)
r_iso = np.array(lan[RIISO_COL], dtype=float)
r_s   = np.array(lan[RS_COL],    dtype=float)
r_s_err=np.array(lan[RSERR_COL], dtype=float)
logL  = np.array(lan[LOGL_COL],  dtype=float)
logL_err=np.array(lan[LOGLERR_COL],dtype=float)
ew    = np.array(lan[EW_COL],    dtype=float)
dBIC  = np.array(lan[DBIC_COL],  dtype=float)
log10_pF=np.array(lan[LOGPF_COL],dtype=float)
flag_res=np.array(lan[FRES_COL], dtype=float)
iso_rel=np.array(lan[IREREL_COL],dtype=float)
chi2e = np.array(lan[CHI2E_COL], dtype=float)
chi2p = np.array(lan[CHI2P_COL], dtype=float)
sb    = np.array(lan[SB_COL],    dtype=float)
field = np.array([s.strip().lower() for s in lan[FIELD_COL]])
did   = np.array(lan[DID_COL],   dtype=np.int64)
gmag  = np.array(lan[GMAG_COL],  dtype=float)

# Clean sentinels
for arr in [r_iso, r_s, logL, ew, dBIC, log10_pF, flag_res, iso_rel,
            chi2e, chi2p, sb, gmag, r_s_err, logL_err]:
    arr[arr == BAD]   = np.nan
    arr[arr <= BAD/2] = np.nan

# ── Deduplicate: keep highest-SN row per astrophysical source ─────────────────
# dups_detectid contains comma-separated IDs for repeated observations
print(f"\nRaw: {len(lan):,} rows")
dup_col = np.array([s.strip() for s in lan[DUPS_COL]])
seen_dids = set()
keep      = np.zeros(len(lan), dtype=bool)
for i, (d, dup_str) in enumerate(zip(did, dup_col)):
    if int(d) in seen_dids:
        continue
    keep[i] = True
    seen_dids.add(int(d))
    if dup_str:
        for ds in dup_str.replace(",", " ").split():
            try:
                seen_dids.add(int(ds))
            except ValueError:
                pass

# Apply dedup flag
def dedup(arr):
    return arr[keep]

ra    = dedup(ra);    dec   = dedup(dec);   stype = dedup(stype)
z     = dedup(z);     r_iso = dedup(r_iso); r_s   = dedup(r_s)
r_s_err=dedup(r_s_err);logL = dedup(logL); logL_err=dedup(logL_err)
ew    = dedup(ew);    dBIC  = dedup(dBIC);  log10_pF=dedup(log10_pF)
flag_res=dedup(flag_res);iso_rel=dedup(iso_rel)
chi2e = dedup(chi2e); chi2p = dedup(chi2p); sb   = dedup(sb)
field = dedup(field); did   = dedup(did);   gmag = dedup(gmag)

print(f"After dedup: {keep.sum():,} unique nebulae")

# =============================================================================
# CELL 6 — SELECT BONA-FIDE RESOLVED LANs
# =============================================================================

# Full selection applying all quality cuts
sel = (
    (dBIC   >= MIN_DBIC)         &
    (logL   >= MIN_LOGL)         &
    (iso_rel<= MAX_ISO_REL_ERR)  &
    (r_iso  >= MIN_R_ISO)        &
    np.isfinite(r_iso) & np.isfinite(logL) & np.isfinite(dBIC) &
    np.isfinite(z)     & (z >= Z_MIN) & (z <= Z_MAX)
)

# Strongly resolved subset
sel_strong = sel & (dBIC >= DBIC_STRONG)

# Blobs: r_iso > 50 kpc
sel_blob   = sel & (r_iso >= BLOB_R_ISO)

# LAE vs AGN hosts
sel_lae = sel & (stype == "lae")
sel_agn = sel & (stype == "agn")

print(f"\nBona-fide LANs (dBIC≥{MIN_DBIC}): {sel.sum():,}")
print(f"Strongly resolved (dBIC≥{DBIC_STRONG}): {sel_strong.sum():,}")
print(f"Blob candidates (r_iso≥{BLOB_R_ISO}kpc): {sel_blob.sum():,}")
print(f"  LAE-hosted: {sel_lae.sum():,}")
print(f"  AGN-hosted: {sel_agn.sum():,}")

# Cosmic look-back time for z axis annotation
z_vals   = np.linspace(Z_MIN, Z_MAX, 200)
lb_times = cosmo.lookback_time(z_vals).to(u.Gyr).value
print(f"\nLook-back time range: "
      f"{cosmo.lookback_time(Z_MIN).to(u.Gyr).value:.1f} – "
      f"{cosmo.lookback_time(Z_MAX).to(u.Gyr).value:.1f} Gyr  "
      f"({10 + (cosmo.lookback_time(Z_MIN).to(u.Gyr).value-10):.0f}–"
      f"{cosmo.lookback_time(Z_MAX).to(u.Gyr).value:.0f} Gyr ago)")

# =============================================================================
# CELL 7 — PHYSICAL MEASUREMENTS
# =============================================================================

# Power-law size–luminosity fit:  log r_iso = alpha * logL + C
def powerlaw_fit(logL_in, r_iso_in, mask):
    x = logL_in[mask]; y = np.log10(np.clip(r_iso_in[mask], 0.1, None))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5: return np.nan, np.nan, np.nan, np.nan
    coeffs, cov = np.polyfit(x[ok], y[ok], 1, cov=True)
    r, p = pearsonr(x[ok], y[ok])
    return coeffs[0], coeffs[1], r, np.sqrt(cov[0,0])

alpha_all,  c_all,  r_all,  ae_all  = powerlaw_fit(logL, r_iso, sel)
alpha_lae,  c_lae,  r_lae,  ae_lae  = powerlaw_fit(logL, r_iso, sel_lae)
alpha_agn,  c_agn,  r_agn,  ae_agn  = powerlaw_fit(logL, r_iso, sel_agn)

print(f"\nSize–Luminosity power-law  (log r_iso = α·logL + C):")
print(f"  All:  α = {alpha_all:.3f}±{ae_all:.3f}  r = {r_all:.3f}")
print(f"  LAE:  α = {alpha_lae:.3f}±{ae_lae:.3f}  r = {r_lae:.3f}")
print(f"  AGN:  α = {alpha_agn:.3f}±{ae_agn:.3f}  r = {r_agn:.3f}")

# Median sizes in 3 redshift bins covering the 10–12 Gyr range
z_bins   = [(1.87, 2.30), (2.30, 2.80), (2.80, 3.52)]
z_labels = ["z=2.1\n(11.8 Gyr)", "z=2.6\n(11.4 Gyr)", "z=3.1\n(11.0 Gyr)"]
print("\nMedian r_iso per redshift bin:")
for (zl, zh), zlab in zip(z_bins, z_labels):
    zm  = sel & (z >= zl) & (z < zh)
    r_m = np.nanmedian(r_iso[zm])
    n_m = zm.sum()
    print(f"  {zlab.replace(chr(10),' ')}: "
          f"N={n_m:,}  median r_iso={r_m:.1f} kpc")

# Save catalogue
df_out = pd.DataFrame({
    "name"       : np.array(lan[NAME_COL])[keep][sel],
    "ra"         : ra[sel],  "dec": dec[sel],
    "z_hetdex"   : z[sel],   "source_type": stype[sel],
    "field"      : field[sel],
    "r_iso_kpc"  : r_iso[sel],
    "r_s_kpc"    : r_s[sel],  "r_s_err": r_s_err[sel],
    "logL_lya"   : logL[sel], "logL_lya_err": logL_err[sel],
    "ew_rest_lya": ew[sel],
    "dBIC"       : dBIC[sel], "log10_pF": log10_pF[sel],
    "flag_resolved": flag_res[sel],
    "is_blob"    : (r_iso[sel] >= BLOB_R_ISO).astype(int),
    "detectid"   : did[sel],
}).sort_values("r_iso_kpc", ascending=False).reset_index(drop=True)

if CSV_PATH:
    df_out.to_csv(CSV_PATH, index=False, float_format="%.5f")
    print(f"\nCatalogue saved -> {CSV_PATH}  ({len(df_out):,} LANs)")

# =============================================================================
# CELL 8 — MAIN SCIENCE FIGURE (8 panels)
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

LAE_COL  = "#58a6ff"
AGN_COL  = "#f78166"
ALL_COL  = "#ffa657"
BLOB_COL = "#d2a8ff"

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

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(
    3, 3, figure=fig,
    hspace=0.44, wspace=0.28,
    left=0.06, right=0.97,
    top=0.92,  bottom=0.06,
)

ax_sl   = fig.add_subplot(gs[0, 0])   # size-luminosity (main science)
ax_rdist= fig.add_subplot(gs[0, 1])   # r_iso distribution
ax_zev  = fig.add_subplot(gs[0, 2])   # r_iso vs z  (cosmic evolution)
ax_bic  = fig.add_subplot(gs[1, 0])   # ΔBIC morphology
ax_ew   = fig.add_subplot(gs[1, 1])   # EW vs logL, coloured by z
ax_blob = fig.add_subplot(gs[1, 2])   # blob gallery scatter
ax_rs   = fig.add_subplot(gs[2, 0])   # r_s vs r_iso  (profile shape)
ax_sky  = fig.add_subplot(gs[2, 1])   # sky distribution
ax_sum  = fig.add_subplot(gs[2, 2])   # summary bar

# ── Panel 1: Size–Luminosity diagram ─────────────────────────────────────────
style_ax(ax_sl,
         r"Size–Luminosity  (N=" + f"{sel.sum():,})",
         r"$\log_{10}\,L_{\rm Ly\alpha}$  [erg s$^{-1}$]",
         r"$r_{\rm iso}$  [proper kpc]")

# 2D density background using hexbin
hb = ax_sl.hexbin(logL[sel], r_iso[sel],
                  gridsize=40, mincnt=1,
                  cmap="plasma", bins="log",
                  xscale="linear", yscale="log",
                  alpha=0.85, rasterized=True)

# Power-law fits
xl = np.linspace(logL[sel].min()-0.1, logL[sel].max()+0.1, 200)
if np.isfinite(alpha_all):
    ax_sl.plot(xl, 10.**np.polyval([alpha_all, c_all], xl),
               "-", color=ALL_COL, lw=2.0, alpha=0.90,
               label=rf"All:  $\alpha$={alpha_all:.2f}  r={r_all:.2f}")
if np.isfinite(alpha_lae):
    ax_sl.plot(xl, 10.**np.polyval([alpha_lae, c_lae], xl),
               "--", color=LAE_COL, lw=1.5, alpha=0.80,
               label=rf"LAE:  $\alpha$={alpha_lae:.2f}")
if np.isfinite(alpha_agn):
    ax_sl.plot(xl, 10.**np.polyval([alpha_agn, c_agn], xl),
               ":", color=AGN_COL, lw=1.5, alpha=0.80,
               label=rf"AGN:  $\alpha$={alpha_agn:.2f}")

# Blob threshold
ax_sl.axhline(BLOB_R_ISO, color=BLOB_COL, lw=1.0, ls="--", alpha=0.7,
              label=f"Blob threshold ({BLOB_R_ISO} kpc)")

ax_sl.set_yscale("log")
ax_sl.set_xlim(logL[sel].min()-0.1, logL[sel].max()+0.1)
ax_sl.set_ylim(4, 300)
ax_sl.legend(fontsize=7.5, facecolor="#21262d", edgecolor=SPINE, labelcolor=TEXT, loc="upper left")
cb_sl = fig.colorbar(hb, ax=ax_sl, fraction=0.033, pad=0.02)
cb_sl.set_label(r"$\log_{10}$ N", color=MUTED, fontsize=7.5)
cb_sl.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
plt.setp(cb_sl.ax.yaxis.get_ticklabels(), color=MUTED)
cb_sl.outline.set_edgecolor(SPINE)

# ── Panel 2: r_iso distribution ───────────────────────────────────────────────
style_ax(ax_rdist, r"Size distribution  $r_{\rm iso}$",
         r"$r_{\rm iso}$  [proper kpc]",
         "Normalised density")

r_bins = np.logspace(np.log10(5), np.log10(250), 40)

for mask, color, label in [
    (sel_lae, LAE_COL, f"LAE (N={sel_lae.sum():,})"),
    (sel_agn, AGN_COL, f"AGN (N={sel_agn.sum():,})"),
]:
    h, e = np.histogram(r_iso[mask & np.isfinite(r_iso)], bins=r_bins, density=True)
    ax_rdist.step(e[:-1], h, where="post", color=color, lw=1.4, alpha=0.85,
                  label=label)
    ax_rdist.fill_between(e[:-1], 0, h, step="post",
                          color=color, alpha=0.18)
    med = np.nanmedian(r_iso[mask])
    ax_rdist.axvline(med, color=color, lw=0.9, ls="--", alpha=0.8)
    ax_rdist.text(med*1.05, ax_rdist.get_ylim()[1]*0.9 if ax_rdist.get_ylim()[1]>0 else 0.05,
                  f"{med:.0f}kpc",
                  color=color, fontsize=7.5, va="top")

ax_rdist.axvline(BLOB_R_ISO, color=BLOB_COL, lw=0.9, ls=":", alpha=0.7,
                 label=f"Blob ≥{BLOB_R_ISO} kpc")
ax_rdist.set_xscale("log")
ax_rdist.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f"{int(x)}"))
mleg(ax_rdist, loc="upper right")

# ── Panel 3: Cosmic evolution of r_iso ─────────────────────────────────────────
style_ax(ax_zev,
         r"Cosmic evolution of $\langle r_{\rm iso}\rangle$",
         "Spectroscopic redshift  $z$",
         r"Median $r_{\rm iso}$  [proper kpc]")

z_grid   = np.linspace(Z_MIN, Z_MAX, 60)
r_med, r_lo, r_hi, z_cen = [], [], [], []
for i in range(len(z_grid)-1):
    zm = sel & (z >= z_grid[i]) & (z < z_grid[i+1]) & np.isfinite(r_iso)
    if zm.sum() < 10: continue
    rv = r_iso[zm]
    r_med.append(np.nanmedian(rv))
    r_lo.append(np.nanpercentile(rv, 16))
    r_hi.append(np.nanpercentile(rv, 84))
    z_cen.append(0.5*(z_grid[i]+z_grid[i+1]))

if r_med:
    z_c = np.array(z_cen)
    r_m = np.array(r_med)
    ax_zev.fill_between(z_c, r_lo, r_hi, color=ALL_COL, alpha=0.20)
    ax_zev.plot(z_c, r_m, "-", color=ALL_COL, lw=2.0, label="Median ±1σ")

    # Separate LAE / AGN tracks
    for mask_z, color, label in [
        (sel_lae, LAE_COL, "LAE"), (sel_agn, AGN_COL, "AGN")
    ]:
        rm, rc = [], []
        for i in range(len(z_grid)-1):
            zm = mask_z & (z >= z_grid[i]) & (z < z_grid[i+1]) & np.isfinite(r_iso)
            if zm.sum() < 5: continue
            rm.append(np.nanmedian(r_iso[zm]))
            rc.append(0.5*(z_grid[i]+z_grid[i+1]))
        if rm:
            ax_zev.plot(rc, rm, "--", color=color, lw=1.3, alpha=0.75, label=label)

# Add look-back time top axis
ax_top = ax_zev.twiny()
ax_top.set_xlim(ax_zev.get_xlim())
z_ticks = [2.0, 2.5, 3.0, 3.5]
lb_ticks = [cosmo.lookback_time(z_).to(u.Gyr).value for z_ in z_ticks]
ax_top.set_xticks([cosmo.age(0).to(u.Gyr).value -
                   cosmo.lookback_time(z_).to(u.Gyr).value
                   for z_ in z_ticks])
ax_top.set_xticks(z_ticks)
ax_top.set_xticklabels([f"{t:.1f}" for t in lb_ticks],
                        color=MUTED, fontsize=7.5)
ax_top.set_xlabel("Look-back time (Gyr)", color=MUTED, fontsize=8)
ax_top.spines["top"].set_color(SPINE)
ax_top.tick_params(colors=MUTED, labelsize=7.5)

mleg(ax_zev, loc="upper right")

# ── Panel 4: ΔBIC morphology ──────────────────────────────────────────────────
style_ax(ax_bic, r"Extended-source evidence  $\Delta$BIC",
         r"$r_{\rm iso}$  [proper kpc]",
         r"$\Delta$BIC  (extended − PSF)")

for mask, color, label in [
    (sel_lae, LAE_COL, "LAE"), (sel_agn, AGN_COL, "AGN")
]:
    ok = mask & np.isfinite(r_iso) & np.isfinite(dBIC)
    ax_bic.scatter(r_iso[ok], dBIC[ok], s=2, c=color,
                   alpha=0.25, linewidths=0, rasterized=True, label=label)

ax_bic.axhline(MIN_DBIC,   color=MUTED,    lw=0.8, ls=":", alpha=0.6,
               label=f"dBIC={MIN_DBIC} (marginal)")
ax_bic.axhline(DBIC_STRONG, color="#ffa657", lw=0.8, ls="--", alpha=0.6,
               label=f"dBIC={DBIC_STRONG} (strong)")
ax_bic.set_xscale("log")
ax_bic.set_xlim(4, 300)
ax_bic.set_ylim(-30, max(200, np.nanpercentile(dBIC[sel], 99)))
mleg(ax_bic, loc="upper left", markerscale=4)

# ── Panel 5: EW vs logL, coloured by z ────────────────────────────────────────
style_ax(ax_ew,
         r"Ly$\alpha$ EW vs Luminosity",
         r"$\log_{10}\,L_{\rm Ly\alpha}$  [erg s$^{-1}$]",
         r"EW$_{\rm rest}$ [Å]")

ok_ew = sel & np.isfinite(ew) & np.isfinite(logL) & (ew > 0)
sc_ew = ax_ew.scatter(logL[ok_ew], ew[ok_ew],
                      c=z[ok_ew], cmap="coolwarm",
                      vmin=Z_MIN, vmax=Z_MAX,
                      s=3, alpha=0.45, linewidths=0,
                      rasterized=True)

# EW–L power-law fit
xl_ew = logL[ok_ew]; yl_ew = np.log10(ew[ok_ew])
valid = np.isfinite(xl_ew) & np.isfinite(yl_ew)
if valid.sum() > 20:
    coeff_ew = np.polyfit(xl_ew[valid], yl_ew[valid], 1)
    xl_fit   = np.linspace(xl_ew[valid].min(), xl_ew[valid].max(), 200)
    ax_ew.plot(xl_fit, 10.**np.polyval(coeff_ew, xl_fit),
               "--", color=TEXT, lw=1.5, alpha=0.75,
               label=f"slope={coeff_ew[0]:.2f}")

ax_ew.set_yscale("log")
ax_ew.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x,_: f"{int(x)}"))
cb_ew = fig.colorbar(sc_ew, ax=ax_ew, fraction=0.033, pad=0.02)
cb_ew.set_label("$z$", color=MUTED, fontsize=8)
cb_ew.ax.yaxis.set_tick_params(color=MUTED, labelsize=7.5)
plt.setp(cb_ew.ax.yaxis.get_ticklabels(), color=MUTED)
cb_ew.outline.set_edgecolor(SPINE)
mleg(ax_ew, loc="upper right")

# ── Panel 6: Blob population scatter ─────────────────────────────────────────
style_ax(ax_blob,
         f"Lyman-α Blobs  (r_iso ≥ {BLOB_R_ISO} kpc, N={sel_blob.sum():,})",
         r"$\log_{10}\,L_{\rm Ly\alpha}$  [erg s$^{-1}$]",
         r"$r_{\rm iso}$  [proper kpc]")

# Background: all LANs
ax_blob.scatter(logL[sel & ~sel_blob], r_iso[sel & ~sel_blob],
                s=1.5, c=MUTED, alpha=0.15, linewidths=0,
                rasterized=True, label="LANs")

# Blobs
blob_lae = sel_blob & (stype=="lae")
blob_agn = sel_blob & (stype=="agn")
ax_blob.scatter(logL[blob_lae], r_iso[blob_lae],
                s=12, c=LAE_COL, alpha=0.75, linewidths=0.3,
                edgecolors=TEXT, rasterized=True,
                label=f"Blob (LAE) N={blob_lae.sum():,}")
ax_blob.scatter(logL[blob_agn], r_iso[blob_agn],
                s=14, c=AGN_COL, alpha=0.80, linewidths=0.3,
                marker="s", edgecolors=TEXT, rasterized=True,
                label=f"Blob (AGN) N={blob_agn.sum():,}")

# Annotate most extended
top_idx = np.argsort(r_iso[sel_blob])[-5:]
for i in top_idx:
    idxs_blob = np.where(sel_blob)[0]
    ii = idxs_blob[i]
    ax_blob.annotate(f"{r_iso[ii]:.0f} kpc",
                     xy=(logL[ii], r_iso[ii]),
                     xytext=(logL[ii]-0.3, r_iso[ii]*1.05),
                     fontsize=6.5, color=BLOB_COL,
                     arrowprops=dict(arrowstyle="-",
                                     color=BLOB_COL, lw=0.5))

ax_blob.axhline(BLOB_R_ISO, color=BLOB_COL, lw=0.9, ls="--", alpha=0.65)
ax_blob.set_yscale("log")
ax_blob.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x,_: f"{int(x)}"))
ax_blob.legend(fontsize=8, facecolor="#21262d", edgecolor=SPINE, labelcolor=TEXT, loc="upper left", markerscale=2)

# ── Panel 7: Profile shape r_s vs r_iso ──────────────────────────────────────
style_ax(ax_rs,
         r"Profile shape: $r_s$ vs $r_{\rm iso}$",
         r"$r_{\rm iso}$  [proper kpc]",
         r"$r_s$  [proper kpc]")

ok_rs = sel & np.isfinite(r_s) & np.isfinite(r_iso) & (r_s > 0)
for mask, color, label in [
    (ok_rs & (stype=="lae"), LAE_COL, "LAE"),
    (ok_rs & (stype=="agn"), AGN_COL, "AGN"),
]:
    ax_rs.scatter(r_iso[mask], r_s[mask],
                  s=2.5, c=color, alpha=0.35,
                  linewidths=0, rasterized=True, label=label)

# Constant ratio lines
r_range = np.logspace(np.log10(5), np.log10(250), 100)
for ratio, ls, label in [(3,"--","r_iso/r_s=3"),
                          (5,":", "r_iso/r_s=5")]:
    ax_rs.plot(r_range, r_range/ratio, ls, color=MUTED,
               lw=0.9, alpha=0.55, label=label)

# Median ratio
med_ratio = np.nanmedian(r_iso[ok_rs] / r_s[ok_rs])
ax_rs.text(0.97, 0.06,
           f"Median r_iso/r_s = {med_ratio:.1f}",
           transform=ax_rs.transAxes, color=TEXT,
           fontsize=8.5, ha="right", va="bottom",
           bbox=dict(boxstyle="round,pad=0.3",
                     facecolor=BG, edgecolor=SPINE, alpha=0.8))

ax_rs.set_xscale("log"); ax_rs.set_yscale("log")
ax_rs.set_xlim(4, 300); ax_rs.set_ylim(0.5, 80)
ax_rs.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x,_: f"{int(x)}"))
ax_rs.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x,_: f"{int(x)}"))
ax_rs.legend(fontsize=7.5, facecolor="#21262d", edgecolor=SPINE, labelcolor=TEXT, loc="upper left", markerscale=3)

# ── Panel 8: Sky distribution ─────────────────────────────────────────────────
style_ax(ax_sky, "LAN sky distribution  (all fields)",
         "RA  (deg)", "Dec  (deg)")

ra_plot = ra.copy()
ra_plot[(field == "dex-fall") & (ra_plot > 180)] -= 360

ax_sky.scatter(ra_plot[sel & (stype=="lae")], dec[sel & (stype=="lae")],
               s=0.8, c=LAE_COL, alpha=0.30, linewidths=0, rasterized=True,
               label=f"LAE ({sel_lae.sum():,})")
ax_sky.scatter(ra_plot[sel & (stype=="agn")], dec[sel & (stype=="agn")],
               s=1.5, c=AGN_COL, alpha=0.50, linewidths=0, rasterized=True,
               label=f"AGN ({sel_agn.sum():,})")
ax_sky.scatter(ra_plot[sel_blob], dec[sel_blob],
               s=12, c=BLOB_COL, alpha=0.80,
               edgecolors=TEXT, linewidths=0.3, rasterized=True,
               label=f"Blob r_iso≥{BLOB_R_ISO}kpc ({sel_blob.sum():,})")

ax_sky.invert_xaxis()
ax_sky.set_xlim(ra_plot[sel].max()+1, ra_plot[sel].min()-1)
ax_sky.set_ylim(dec[sel].min()-0.5, dec[sel].max()+0.5)
ax_sky.legend(fontsize=7.5, facecolor="#21262d", edgecolor=SPINE, labelcolor=TEXT, loc="upper left", markerscale=3)

# ── Panel 9: Summary statistics bar ──────────────────────────────────────────
ax_sum.set_facecolor(AX_BG)
for sp in ax_sum.spines.values(): sp.set_color(SPINE)
ax_sum.tick_params(colors=MUTED, labelsize=8.5)
ax_sum.set_title("Population summary", color=TEXT,
                 fontsize=10, fontweight="bold", loc="left", pad=5)

labels_s = [
    "All LANs\n(dBIC≥2)",
    "Strongly\nresolved\n(dBIC≥6)",
    "LAE-hosted",
    "AGN-hosted",
    f"Blobs\n(r_iso≥{BLOB_R_ISO}kpc)",
]
counts_s = [
    sel.sum(), sel_strong.sum(),
    sel_lae.sum(), sel_agn.sum(), sel_blob.sum(),
]
colors_s = [ALL_COL, TEXT, LAE_COL, AGN_COL, BLOB_COL]

y_pos = np.arange(len(labels_s))
bars  = ax_sum.barh(y_pos, counts_s, color=colors_s,
                    alpha=0.80, edgecolor=SPINE, linewidth=0.5, height=0.55)
ax_sum.set_yticks(y_pos)
ax_sum.set_yticklabels(labels_s, color=TEXT, fontsize=8.5)
for bar, cnt in zip(bars, counts_s):
    ax_sum.text(cnt * 1.02, bar.get_y() + bar.get_height()/2,
                f"{cnt:,}", va="center", color=TEXT, fontsize=8.5)
ax_sum.set_xlabel("N nebulae", color=TEXT, fontsize=9)
ax_sum.xaxis.set_minor_locator(AutoMinorLocator())
ax_sum.set_xlim(0, max(counts_s) * 1.25)

# ── Super-title ───────────────────────────────────────────────────────────────
syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
fig.suptitle(
    r"HETDEX — Lyman-$\alpha$ Nebulae: Gigantic Hydrogen Halos at $z=1.9$–3.5"
    r" (10–12 Gyr ago)" + syn_tag,
    color=TEXT, fontsize=13, fontweight="bold", y=0.978,
)

fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nScience figure saved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 9 — OUTREACH / PRESS FIGURE  (single wide panel)
# =============================================================================

fig2, ax2 = plt.subplots(1, 1, figsize=(14, 7))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor("#030a15")   # near-black space background

# 2D density of all LANs as glowing haze
ok_all = sel & np.isfinite(logL) & np.isfinite(r_iso)
hb2 = ax2.hexbin(
    logL[ok_all], r_iso[ok_all],
    gridsize=55, mincnt=1,
    cmap="inferno", bins="log",
    xscale="linear", yscale="log",
    alpha=0.90, rasterized=True,
)

# Blobs highlighted
ax2.scatter(logL[blob_lae], r_iso[blob_lae], s=18,
            c=LAE_COL, alpha=0.85, edgecolors="white",
            linewidths=0.5, rasterized=True, zorder=5,
            label=rf"Ly$\alpha$ Blob (LAE)")
ax2.scatter(logL[blob_agn], r_iso[blob_agn], s=22,
            c=AGN_COL, alpha=0.90, marker="*",
            edgecolors="white", linewidths=0.5,
            rasterized=True, zorder=5,
            label=r"Ly$\alpha$ Blob (AGN-hosted)")

# Reference: Milky Way diameter
ax2.axhline(50, color="#3fb950", lw=0.9, ls=":",
            alpha=0.65, label="Milky Way diameter (~50 kpc)")

ax2.set_yscale("log")
ax2.set_ylabel(r"Nebula radius  $r_{\rm iso}$  [proper kpc]",
               color=TEXT, fontsize=12)
ax2.set_xlabel(r"Lyman-$\alpha$ luminosity  $\log_{10}(L\,/\,{\rm erg\,s}^{-1})$",
               color=TEXT, fontsize=12)
for sp in ax2.spines.values(): sp.set_color(SPINE)
ax2.tick_params(colors=MUTED, which="both", direction="in",
                labelsize=10, top=True, right=True)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x,_: f"{int(x)} kpc"))

# Big annotation
ax2.text(0.02, 0.97,
         f"{sel.sum():,} hydrogen gas halos\n"
         f"discovered 10–12 billion years ago",
         transform=ax2.transAxes,
         color=TEXT, fontsize=14, fontweight="bold",
         va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.5",
                   facecolor="#0d1117", edgecolor=SPINE, alpha=0.85))

ax2.text(0.02, 0.75,
         f"{sel_blob.sum():,} giant blobs  (r > {BLOB_R_ISO} kpc)\n"
         f"Largest: {r_iso[sel].max():.0f} kpc across",
         transform=ax2.transAxes,
         color=BLOB_COL, fontsize=11, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.4",
                   facecolor="#0d1117", edgecolor=SPINE, alpha=0.80))

ax2.legend(fontsize=9.5, facecolor="#21262d",
           edgecolor=SPINE, labelcolor=TEXT,
           loc="lower right", markerscale=2)
ax2.set_ylim(4, 300)

cb2 = fig2.colorbar(hb2, ax=ax2, fraction=0.020, pad=0.02)
cb2.set_label(r"$\log_{10}$ (N per pixel)", color=MUTED, fontsize=9)
cb2.ax.yaxis.set_tick_params(color=MUTED, labelsize=8)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=MUTED)
cb2.outline.set_edgecolor(SPINE)

fig2.suptitle(
    r"HETDEX Lyman-$\alpha$ Nebulae — The Largest Census of Giant Hydrogen Halos"
    r" at Cosmic Noon" + syn_tag,
    color=TEXT, fontsize=14, fontweight="bold", y=1.01,
)

fig2.tight_layout()
fig2.savefig(OUTREACH_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Outreach figure saved -> {OUTREACH_PATH}")
plt.show()

# =============================================================================
# CELL 10 — SUMMARY
# =============================================================================

print("\n" + "=" * 68)
print("  HETDEX Lyman-α Nebulae — Science Summary")
print("=" * 68)
print(f"  Total LAN candidates (raw)    : {keep.sum():,}")
print(f"  Bona-fide LANs (dBIC≥{MIN_DBIC})  : {sel.sum():,}")
print(f"    LAE-hosted                  : {sel_lae.sum():,}")
print(f"    AGN-hosted                  : {sel_agn.sum():,}")
print(f"  Strongly resolved (dBIC≥{DBIC_STRONG}) : {sel_strong.sum():,}")
print(f"  Lyman-α Blobs (≥{BLOB_R_ISO} kpc)   : {sel_blob.sum():,}")
print(f"  Largest nebula r_iso          : {r_iso[sel].max():.0f} kpc")
print(f"  Median r_iso (LAE)            : {np.nanmedian(r_iso[sel_lae]):.1f} kpc")
print(f"  Median r_iso (AGN)            : {np.nanmedian(r_iso[sel_agn]):.1f} kpc")
print(f"  Median logL_lya               : {np.nanmedian(logL[sel]):.3f}")
print(f"  Median EW_rest                : {np.nanmedian(ew[sel & np.isfinite(ew)]):.1f} Å")
print(f"  Size–luminosity slope (all)   : {alpha_all:.3f} ± {ae_all:.3f}")
print(f"  z range (look-back time)      : "
      f"{Z_MIN}–{Z_MAX}  →  "
      f"{cosmo.lookback_time(Z_MIN).to(u.Gyr).value:.1f}–"
      f"{cosmo.lookback_time(Z_MAX).to(u.Gyr).value:.1f} Gyr")
print(f"\n  Cosmic epoch: cosmic NOON — peak of star formation in the Universe")
print(f"  These nebulae trace the CGM gas reservoir feeding and quenching")
print(f"  galaxies at the moment of maximum star-forming activity.")
print("=" * 68)
