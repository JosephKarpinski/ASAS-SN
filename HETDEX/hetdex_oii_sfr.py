"""
hetdex_oii_sfr.py
=================
Star Formation Rate analysis for HETDEX [OII] emitters
using the Kennicutt (1998) calibration.

Kennicutt (1998) ApJ, 498, 541:
    SFR [M_sun/yr] = (1.4e-41) * L_OII [erg/s]

Requires:
    pip install astropy numpy matplotlib scipy

Data:
    hetdex_sc2_v1.5.fits  (or hetdex_sc1_vX.fits)
    from https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/hetdex_source_catalog_2/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings("ignore")

from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import Planck18
import astropy.units as u

# ── Configuration ──────────────────────────────────────────────────────────────
CATALOG_PATH = "hetdex_sc2_v1.5.fits"   # update to your local path
# CATALOG_PATH = "hetdex_sc1_v3.2.fits"  # SC1 alternative

# Quality cuts
MIN_SN        = 4.5      # minimum line S/N
MAX_CHI2      = 3.0      # maximum reduced chi2 of line fit (detinfo table)
MIN_P_CONF    = 0.5      # RF classifier confidence (SC2 only)
MIN_P_CNN     = 0.5      # CNN classifier confidence (SC2 only)
BAD_VALUE     = -999.0   # sentinel for missing floats in SC2

# Kennicutt (1998) OII calibration constant
# SFR [M_sun/yr] = K_OII * L_OII [erg/s]
K_OII = 1.4e-41

# ── Cosmology ──────────────────────────────────────────────────────────────────
cosmo = Planck18

# ── Load catalog ───────────────────────────────────────────────────────────────
print(f"Loading {CATALOG_PATH} …")
try:
    hdul = fits.open(CATALOG_PATH)
    tab  = Table(hdul[1].data)
    hdul.close()
    print(f"  Loaded {len(tab):,} rows")
except FileNotFoundError:
    # ── Synthetic demo dataset ─────────────────────────────────────────────────
    # Reproduces realistic HETDEX OII distributions so the script runs
    # without the actual FITS file.  Remove this block once you have the data.
    print("  *** Catalog not found — generating synthetic demo data ***")
    rng = np.random.default_rng(42)
    N   = 18_000

    # Redshift: OII emitters observed in HETDEX VIRUS bandpass (3470–5540 Å)
    # [OII] 3727 Å → z = (wave/3727) - 1 → z ∈ [−0.07, 0.49]
    # but HETDEX selects z > 0.05 for useful OII coverage
    z_oii = rng.uniform(0.05, 0.48, N)

    # Flux: log-normal centred on realistic HETDEX OII flux
    # typical detected flux ~1–30 × 1e-17 erg/s/cm2
    log_flux_centre = np.log(8.0) + 1.5 * z_oii          # brighter at high-z: survey bias
    flux_oii_raw    = rng.lognormal(log_flux_centre, 0.7, N)   # 1e-17 erg/s/cm2

    # S/N: correlate with flux
    sn_raw = flux_oii_raw / rng.lognormal(0.6, 0.4, N) * 2.5
    sn_raw = np.clip(sn_raw, 0.5, 60.0)

    # chi2: most good, some bad
    chi2_raw = rng.lognormal(-0.15, 0.55, N)

    # Classifiers (SC2)
    p_conf = np.clip(rng.beta(3, 1.2, N), 0, 1)
    p_cnn  = np.clip(rng.beta(2.8, 1.1, N), 0, 1)

    # flag_aper: 30% use aperture flux
    flag_aper = rng.integers(0, 2, N)

    # Aperture correction factor (log-normal ~1)
    aper_factor = rng.lognormal(0.07, 0.12, N)
    flux_aper_  = flux_oii_raw * aper_factor

    tab = Table({
        "source_type": np.where(rng.uniform(size=N) < 0.94, "oii", "lzg"),
        "z_hetdex"   : z_oii.astype(np.float32),
        "flux_oii"   : flux_oii_raw.astype(np.float32),
        "flux_oii_err": (flux_oii_raw / sn_raw).astype(np.float32),
        "flux_aper"  : flux_aper_.astype(np.float32),
        "flux_aper_err": (flux_aper_ / sn_raw).astype(np.float32),
        "flag_aper"  : flag_aper.astype(np.int32),
        "sn"         : sn_raw.astype(np.float32),
        "p_conf"     : p_conf.astype(np.float32),
        "p_cnn"      : p_cnn.astype(np.float32),
        "field"      : rng.choice(
                           ["dex-spring","dex-fall","cosmos","goods-n"],
                           N, p=[0.55,0.30,0.10,0.05]).astype("U12"),
        "logL_oii"   : np.full(N, BAD_VALUE, dtype=np.float32),  # will recompute
    })

# ── Select OII emitters ─────────────────────────────────────────────────────────
mask_type = tab["source_type"] == "oii"
print(f"  OII sources before cuts: {mask_type.sum():,}")

# Quality cuts (handle both SC1 and SC2 column availability gracefully)
mask_sn = tab["sn"] > MIN_SN

mask_qual = mask_type & mask_sn

if "p_conf" in tab.colnames:
    mask_qual &= (tab["p_conf"] >= MIN_P_CONF)
    mask_qual &= (tab["p_cnn"]  >= MIN_P_CNN)

# Guard against bad sentinel values
if "flux_oii" in tab.colnames:
    mask_qual &= (tab["flux_oii"] > 0) & (tab["flux_oii"] != BAD_VALUE)

oii = tab[mask_qual].copy()
print(f"  OII sources after cuts  : {len(oii):,}")

# ── Best flux: aperture (resolved) or PSF ──────────────────────────────────────
# flag_aper == 1 → use flux_aper (resolved elliptical aperture)
# flag_aper == 0 → use flux     (PSF point-source flux)
if "flag_aper" in oii.colnames:
    use_aper   = oii["flag_aper"] == 1
    flux_best  = np.where(use_aper, oii["flux_aper"],  oii["flux_oii"])
    ferr_best  = np.where(use_aper, oii["flux_aper_err"], oii["flux_oii_err"])
else:
    flux_best  = np.array(oii["flux_oii"],     dtype=float)
    ferr_best  = np.array(oii["flux_oii_err"], dtype=float)

flux_best  = np.array(flux_best,  dtype=float)
ferr_best  = np.array(ferr_best,  dtype=float)
z          = np.array(oii["z_hetdex"], dtype=float)

# Unit conversion: catalog flux is in 1e-17 erg/s/cm2
FLUX_SCALE = 1e-17   # erg/s/cm2 per catalog unit

flux_cgs   = flux_best * FLUX_SCALE    # erg/s/cm2
ferr_cgs   = ferr_best * FLUX_SCALE

# ── Luminosity distance ────────────────────────────────────────────────────────
z_clipped  = np.clip(z, 1e-4, None)
DL         = cosmo.luminosity_distance(z_clipped).to(u.cm).value   # cm

# ── [OII] Luminosity ──────────────────────────────────────────────────────────
# L_OII = 4 π D_L^2 F_OII
L_oii      = 4.0 * np.pi * DL**2 * flux_cgs          # erg/s
L_oii_err  = 4.0 * np.pi * DL**2 * ferr_cgs

log_L      = np.log10(np.clip(L_oii, 1e30, None))

# ── SFR: Kennicutt (1998) ─────────────────────────────────────────────────────
# SFR [M_sun/yr] = 1.4e-41 * L_OII [erg/s]
SFR        = K_OII * L_oii                             # M_sun/yr
SFR_err    = K_OII * L_oii_err
log_SFR    = np.log10(np.clip(SFR, 1e-6, None))

# ── Cosmological volume per z-bin (for SFR density SFRD) ──────────────────────
z_bins     = np.linspace(z.min(), z.max(), 12)
z_cen      = 0.5 * (z_bins[:-1] + z_bins[1:])

# Comoving volume of shell between z_bins[i] and z_bins[i+1]
# using the full HETDEX area
HETDEX_AREA_DEG2   = 540.0   # approximate total survey footprint
HETDEX_AREA_SR     = HETDEX_AREA_DEG2 * (np.pi/180)**2

def dVc_shell(z1, z2, area_sr):
    """Comoving volume [Mpc^3] of a shell between z1 and z2."""
    V1 = cosmo.comoving_volume(z1).to(u.Mpc**3).value
    V2 = cosmo.comoving_volume(z2).to(u.Mpc**3).value
    return (V2 - V1) * (area_sr / (4.0*np.pi))

bin_idx    = np.digitize(z, z_bins) - 1
bin_idx    = np.clip(bin_idx, 0, len(z_bins)-2)

Vc_shells  = np.array([dVc_shell(z_bins[i], z_bins[i+1], HETDEX_AREA_SR)
                        for i in range(len(z_bins)-1)])

# SFRD per bin [M_sun/yr/Mpc^3]
SFRD       = np.zeros(len(z_cen))
SFRD_err   = np.zeros(len(z_cen))
counts     = np.zeros(len(z_cen), dtype=int)

for i in range(len(z_cen)):
    sel = bin_idx == i
    counts[i] = sel.sum()
    if sel.sum() > 0 and Vc_shells[i] > 0:
        SFRD[i]     = SFR[sel].sum()     / Vc_shells[i]
        SFRD_err[i] = np.sqrt(np.sum(SFR_err[sel]**2)) / Vc_shells[i]

# ── Plotting ───────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(16, 14))
fig.patch.set_facecolor("#0d1117")

gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                         left=0.09, right=0.97, top=0.93, bottom=0.07)

AX_COLOR   = "#161b22"
SPINE_COL  = "#30363d"
TEXT_COL   = "#e6edf3"
MUTED_COL  = "#8b949e"
ACCENT     = "#58a6ff"
ACCENT2    = "#3fb950"
ACCENT3    = "#f78166"

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(AX_COLOR)
    for sp in ax.spines.values():
        sp.set_color(SPINE_COL)
    ax.tick_params(colors=MUTED_COL, which="both", direction="in",
                   top=True, right=True)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.set_title(title, color=TEXT_COL, fontsize=12, pad=8, loc="left",
                 fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

# ── Panel 1: 2D hex density  SFR vs z ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "SFR vs Redshift",
         "Spectroscopic redshift  $z$",
         r"$\log_{10}$ SFR  [$M_\odot\,\mathrm{yr}^{-1}$]")

hb = ax1.hexbin(z, log_SFR, gridsize=55, cmap="YlOrRd",
                mincnt=2, linewidths=0.2,
                vmin=2, vmax=None)
cb = fig.colorbar(hb, ax=ax1, pad=0.01)
cb.set_label("Counts per hex", color=MUTED_COL, fontsize=9)
cb.ax.yaxis.set_tick_params(color=MUTED_COL)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED_COL)
cb.outline.set_edgecolor(SPINE_COL)

# Running median
z_med_bins = np.linspace(z.min(), z.max(), 20)
z_meds, sfr_meds, sfr_p16, sfr_p84 = [], [], [], []
for i in range(len(z_med_bins)-1):
    sel = (z >= z_med_bins[i]) & (z < z_med_bins[i+1])
    if sel.sum() > 10:
        vals = log_SFR[sel]
        z_meds.append(0.5*(z_med_bins[i]+z_med_bins[i+1]))
        sfr_meds.append(np.median(vals))
        sfr_p16.append(np.percentile(vals, 16))
        sfr_p84.append(np.percentile(vals, 84))

z_meds   = np.array(z_meds)
sfr_meds = np.array(sfr_meds)
ax1.plot(z_meds, sfr_meds, color=ACCENT, lw=2.0, label="Median")
ax1.fill_between(z_meds, sfr_p16, sfr_p84,
                 color=ACCENT, alpha=0.18, label="16–84th pct.")
ax1.legend(fontsize=9, facecolor="#21262d", edgecolor=SPINE_COL,
           labelcolor=TEXT_COL)

# ── Panel 2: log L_OII vs z ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "[OII] Luminosity vs Redshift",
         "Spectroscopic redshift  $z$",
         r"$\log_{10}$ L$_{\rm [OII]}$  [erg s$^{-1}$]")

hb2 = ax2.hexbin(z, log_L, gridsize=55, cmap="plasma",
                  mincnt=2, linewidths=0.2)
cb2 = fig.colorbar(hb2, ax=ax2, pad=0.01)
cb2.set_label("Counts per hex", color=MUTED_COL, fontsize=9)
cb2.ax.yaxis.set_tick_params(color=MUTED_COL)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=MUTED_COL)
cb2.outline.set_edgecolor(SPINE_COL)

# ── Panel 3: SFR histogram by field ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "SFR Distribution by Survey Field",
         r"$\log_{10}$ SFR  [$M_\odot\,\mathrm{yr}^{-1}$]",
         "Normalized count")

field_colors = {
    "dex-spring": "#58a6ff",
    "dex-fall"  : "#3fb950",
    "cosmos"    : "#f78166",
    "goods-n"   : "#d2a8ff",
    "nep"       : "#ffa657",
    "ssa22"     : "#79c0ff",
}
sfr_bins = np.linspace(log_SFR.min(), log_SFR.max(), 50)

if "field" in oii.colnames:
    fields = np.array(oii["field"], dtype=str)
    unique_fields = [f for f in field_colors if f in np.unique(fields)]
    for field in unique_fields:
        fsel   = fields == field
        if fsel.sum() < 5:
            continue
        color  = field_colors.get(field, "#888")
        ax3.hist(log_SFR[fsel], bins=sfr_bins, density=True,
                 alpha=0.55, color=color, label=field, histtype="stepfilled")
        ax3.hist(log_SFR[fsel], bins=sfr_bins, density=True,
                 color=color, histtype="step", lw=1.2)
else:
    ax3.hist(log_SFR, bins=sfr_bins, density=True, color=ACCENT,
             alpha=0.6, histtype="stepfilled")

ax3.legend(fontsize=9, facecolor="#21262d", edgecolor=SPINE_COL,
           labelcolor=TEXT_COL, ncol=2)

# ── Panel 4: SFRD vs z (cosmic star formation history) ────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Cosmic SFR Density (HETDEX [OII])",
         "Redshift  $z$",
         r"SFRD  [$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3}$]")

good_bins = counts > 20
ax4.errorbar(z_cen[good_bins], SFRD[good_bins],
             yerr=SFRD_err[good_bins],
             fmt="o", color=ACCENT2, ms=6, lw=1.4, capsize=3,
             ecolor=ACCENT2, alpha=0.9, label="HETDEX OII (this work)")

# Madau & Dickinson (2014) reference curve over z=0–0.5
z_md = np.linspace(0.0, 0.50, 100)
# MD14 Eq 15: ψ(z) = 0.015 (1+z)^2.7 / [1 + ((1+z)/2.9)^5.6]  M_sun/yr/Mpc^3
psi_md = 0.015 * (1+z_md)**2.7 / (1 + ((1+z_md)/2.9)**5.6)
ax4.plot(z_md, psi_md, "--", color=ACCENT3, lw=1.8, alpha=0.85,
         label="Madau & Dickinson (2014)")

ax4.set_yscale("log")
ax4.legend(fontsize=9, facecolor="#21262d", edgecolor=SPINE_COL,
           labelcolor=TEXT_COL)

# ── Super-title + annotation ───────────────────────────────────────────────────
fig.suptitle(
    "HETDEX [OII] Emitters — Star Formation Rates  "
    r"[Kennicutt 1998: SFR = $1.4\times10^{-41}\ L_{\rm [OII]}$]",
    color=TEXT_COL, fontsize=13, fontweight="bold", y=0.97
)

stats_str = (
    f"N = {len(oii):,} OII emitters   |   "
    f"S/N > {MIN_SN}   |   "
    f"z ∈ [{z.min():.2f}, {z.max():.2f}]   |   "
    f"Planck18 cosmology"
)
fig.text(0.5, 0.005, stats_str, ha="center", fontsize=9, color=MUTED_COL)

plt.savefig("hetdex_oii_sfr.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → hetdex_oii_sfr.png")
plt.show()

# ── Summary statistics ─────────────────────────────────────────────────────────
print("\n── SFR Summary ──────────────────────────────────────────────")
print(f"  N OII emitters (after cuts) : {len(oii):,}")
print(f"  Redshift range              : {z.min():.3f} – {z.max():.3f}")
print(f"  Median log L_OII [erg/s]    : {np.median(log_L):.2f}")
print(f"  Median log SFR [Msun/yr]    : {np.median(log_SFR):.2f}")
print(f"  SFR range [Msun/yr]         : {SFR.min():.3g} – {SFR.max():.3g}")
print(f"  Median SFR [Msun/yr]        : {np.median(SFR):.2f}")
print("─────────────────────────────────────────────────────────────")
