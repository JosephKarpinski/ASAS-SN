"""
hetdex_sensitivity_map.py
=========================
Map the HETDEX SC2 survey depth across all fields by aggregating
flux_noise_1sigma_obs (the per-detection 1-sigma line flux sensitivity)
onto a sky grid.

Physics
-------
flux_noise_1sigma_obs  [1e-17 erg/s/cm²]  is the observed (not extinction-
corrected) 1-sigma line flux sensitivity for each individual detection.
It encodes all depth-relevant information simultaneously:
  - Atmospheric seeing (fwhm): worse seeing dilutes the PSF → higher noise
  - Sky transparency (throughput): low throughput → higher noise
  - Number of IFU shots covering that sky position: more shots → lower noise
  - Chip position / amplifier gain: accounted for per-fiber

Strategy
--------
1. Load detinfo; keep ONE representative detection per shotid × sky-pixel
   (use the median noise value per pixel bin to avoid sampling bias from
   lines-per-shot variation).
2. For each field, bin onto a fine RA/Dec grid.
   stat = "median" noise → the typical depth a survey would reach at that pixel
   stat = "min"    noise → the best depth achieved (deepest co-added region)
   stat = "N_shot" → how many independent HET shots covered that pixel
3. Derived products:
   - 5σ limiting flux  = 5 × median_noise   [1e-17 cgs]
   - Limiting logL_lya at median field z     [erg/s]
   - Sensitivity inhomogeneity: σ(noise)/median(noise) per field
4. Multi-panel figure:
   Row 0  Full-survey sensitivity panorama (all 6 fields, median noise, log scale)
   Row 1  Per-field zoom: dex-spring | dex-fall | COSMOS | GOODS-N
   Row 2  Per-field zoom: NEP | SSA22 | throughput map | fwhm map
   Row 3  Summary panels: noise histogram, limiting logL vs z, depth vs field

Column names used
-----------------
  ra_det, dec_det           Detection sky position
  flux_noise_1sigma_obs     Observed 1-sigma sensitivity  [1e-17 cgs]
  flux_noise_1sigma         Extinction-corrected 1-sigma sensitivity
  throughput                Relative spectral response at 4540 Å
  fwhm                      Seeing FWHM  [arcsec]
  shotid                    HET observation ID
  field                     Survey field name

Requirements
------------
  pip install astropy numpy matplotlib scipy

Data
----
  hetdex_sc2_detinfo_v1.5.fits
  https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/hetdex_source_catalog_2/
"""

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

DETINFO_PATH = "hetdex_sc2_detinfo_v1.5.fits"
SAVE_PATH    = "hetdex_sensitivity_map.png"

# Grid resolution (deg/pixel) per panel type
# Synthetic-safe defaults below. On the real 3.3M-row detinfo table you
# can tighten these substantially without hitting memory limits:
#   RES_FULL = 0.03   RES_WIDE = 0.01   RES_DEEP = 0.003
# Each IFU covers ~51" x 51", so 0.01 deg ≈ 36" ≈ 0.7 IFU — a good match.
RES_FULL    = 0.10    # deg — full-survey panorama
RES_WIDE    = 0.04    # deg — dex-spring / dex-fall
RES_DEEP    = 0.012   # deg — COSMOS / GOODS-N / NEP / SSA22

# Lyα rest wavelength and reference redshift for L_limit computation
LYA_AA      = 1215.67
Z_REF       = 2.5     # compute limiting luminosity at this redshift

# Bad value sentinel
BAD         = -999.0

# Maximum noise value to include (remove pathological outliers)
MAX_NOISE   = 50.0    # 1e-17 cgs — cap at 50 sigma for display

# =============================================================================
# CELL 2 — IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.ticker  import AutoMinorLocator, LogFormatter
# make_axes_locatable replaced by fig.colorbar(fraction=) for layout stability
from scipy.ndimage      import gaussian_filter
from scipy.stats        import binned_statistic_2d

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

cosmo = Planck18
print("Imports OK.")

# =============================================================================
# CELL 3 — SYNTHETIC DATA GENERATOR
# =============================================================================

def make_synthetic_detinfo(n=150_000, seed=42):
    """
    Synthetic detinfo with realistic flux_noise_1sigma_obs spatial structure:
      - dex-spring / dex-fall: noise ~ 2–15 × 1e-17, deep strips where shots overlap
      - COSMOS:  noise ~ 1–5 × 1e-17  (many repeat visits)
      - GOODS-N: noise ~ 0.8–4 × 1e-17
      - NEP:     noise ~ 2–12 × 1e-17
      - SSA22:   noise ~ 2–10 × 1e-17
    """
    rng = np.random.default_rng(seed)

    # Field configurations
    field_cfgs = [
        # name,          frac,  ra_c, dec_c, ra_s, dec_s,  noise_mu, noise_s
        ("dex-spring",   0.50,  185,  50.5,  25,   3.0,    5.0, 2.5),
        ("dex-fall",     0.28,  355,   0.5,  20,   3.5,    6.0, 3.0),
        ("cosmos",       0.10,  150.1, 2.2,  0.6,  0.5,    2.5, 1.0),
        ("goods-n",      0.06,  189.3,62.2,  0.4,  0.3,    1.8, 0.8),
        ("nep",          0.04,  269,  66.5,  2.0,  1.5,    4.5, 2.0),
        ("ssa22",        0.02,  334,   0.4,  0.4,  0.4,    4.0, 1.8),
    ]

    rows = {k: [] for k in [
        "ra_det","dec_det","flux_noise_1sigma_obs","flux_noise_1sigma",
        "throughput","fwhm","shotid","field",
    ]}

    shot_counter = int(2e10)
    for fname, frac, ra_c, dec_c, ra_s, dec_s, noise_mu, noise_s in field_cfgs:
        n_f = int(n * frac)

        # Positions: cluster around realistic field centre
        ra_f  = (rng.normal(ra_c, ra_s, n_f)) % 360
        dec_f = rng.normal(dec_c, dec_s, n_f)

        # Noise: log-normal with a depth gradient (deeper near field centre)
        dist  = np.sqrt(((ra_f - ra_c) * np.cos(np.radians(dec_c)))**2
                        + (dec_f - dec_c)**2)
        # Add spatial structure: deeper (lower noise) near centre, noisier at edges
        # Simulate overlapping shots: every ~0.35 deg a shot centre → noise dips
        shot_period   = 0.35
        shot_modulation = 0.4 * np.sin(ra_f * np.pi / shot_period)**2 * \
                               np.sin(dec_f * np.pi / shot_period)**2
        noise_local   = noise_mu * (1.0 + 0.4 * dist / (ra_s + 0.1)
                                    + shot_modulation)
        noise_obs     = np.clip(
            rng.lognormal(np.log(noise_local), 0.25), 0.5, MAX_NOISE
        ).astype(np.float32)
        noise_corr    = (noise_obs * rng.uniform(0.9, 1.1, n_f)).astype(np.float32)

        # Throughput: anti-correlated with noise (low throughput → high noise)
        throughput    = np.clip(
            rng.normal(0.15, 0.04, n_f) - 0.005 * noise_obs, 0.02, 0.30
        ).astype(np.float32)

        # Seeing: uniform per field with slight spatial trend
        fwhm_f        = np.clip(
            rng.normal(1.8, 0.3, n_f) + 0.1 * shot_modulation, 0.8, 4.5
        ).astype(np.float32)

        # Shot IDs: ~150 detections per shot on average
        n_shots = max(1, n_f // 150)
        shot_ids = rng.choice(
            np.arange(shot_counter, shot_counter + n_shots, dtype=np.int64),
            size=n_f,
        )
        shot_counter += n_shots

        rows["ra_det"].extend(ra_f)
        rows["dec_det"].extend(dec_f)
        rows["flux_noise_1sigma_obs"].extend(noise_obs)
        rows["flux_noise_1sigma"].extend(noise_corr)
        rows["throughput"].extend(throughput)
        rows["fwhm"].extend(fwhm_f)
        rows["shotid"].extend(shot_ids)
        rows["field"].extend([fname] * n_f)

    tab = Table({k: np.array(v, dtype=np.float32
                             if k not in ("field","shotid") else
                             (np.int64 if k == "shotid" else str))
                 for k, v in rows.items()})
    # Fix field column type
    tab["field"] = np.array(rows["field"])
    tab["shotid"] = np.array(rows["shotid"], dtype=np.int64)

    print(f"  Synthetic: {len(tab):,} detections across 6 fields")
    return tab


# =============================================================================
# CELL 4 — LOAD CATALOG
# =============================================================================

def getcol(tab, *cands):
    lc = {c.lower(): c for c in tab.colnames}
    for c in cands:
        if c.lower() in lc:
            return lc[c.lower()]
    raise KeyError(f"None of {cands} in table. Have: {list(tab.colnames)[:25]}")

print("Loading detinfo ...")
try:
    hdul = fits.open(DETINFO_PATH, memmap=True)
    tab  = Table(hdul[1].data)
    hdul.close()
    tab.rename_columns(tab.colnames, [c.lower() for c in tab.colnames])
    print(f"  {len(tab):,} rows, {len(tab.colnames)} columns")
    SYNTHETIC = False
except FileNotFoundError:
    print(f"  '{DETINFO_PATH}' not found — using synthetic demo data.")
    tab       = make_synthetic_detinfo()
    SYNTHETIC = True

# Resolve column names
RA_COL    = getcol(tab, "ra_det",  "ra")
DEC_COL   = getcol(tab, "dec_det", "dec")
NOISE_COL = getcol(tab, "flux_noise_1sigma_obs")
NOISE_C   = getcol(tab, "flux_noise_1sigma")
THRU_COL  = getcol(tab, "throughput")
FWHM_COL  = getcol(tab, "fwhm")
SHOT_COL  = getcol(tab, "shotid")
FIELD_COL = getcol(tab, "field")

print(f"  Key columns: {RA_COL}, {DEC_COL}, {NOISE_COL}, {THRU_COL}, {FWHM_COL}")

# =============================================================================
# CELL 5 — EXTRACT AND CLEAN ARRAYS
# =============================================================================

ra       = np.array(tab[RA_COL],    dtype=float)
dec      = np.array(tab[DEC_COL],   dtype=float)
noise    = np.array(tab[NOISE_COL], dtype=float)
noise_c  = np.array(tab[NOISE_C],   dtype=float)
thru     = np.array(tab[THRU_COL],  dtype=float)
fwhm     = np.array(tab[FWHM_COL],  dtype=float)
shotid   = np.array(tab[SHOT_COL],  dtype=np.int64)
field    = np.array([f.strip().lower() for f in tab[FIELD_COL]])

# Clean sentinels
for arr in [noise, noise_c, thru, fwhm]:
    arr[arr == BAD]  = np.nan
    arr[arr <= 0]    = np.nan
noise[noise > MAX_NOISE] = np.nan

# Valid mask
valid = (np.isfinite(ra) & np.isfinite(dec) &
         np.isfinite(noise) & np.isfinite(thru))

ra    = ra[valid];    dec   = dec[valid]
noise = noise[valid]; thru  = thru[valid]
fwhm  = fwhm[valid];  field = field[valid]
shotid= shotid[valid]
if np.isfinite(noise_c[valid]).any():
    noise_c = noise_c[valid]
else:
    noise_c = noise.copy()

print(f"\nValid detections: {valid.sum():,} / {len(tab):,}")
print(f"Noise range: {np.nanpercentile(noise,1):.2f} – "
      f"{np.nanpercentile(noise,99):.2f}  ×1e-17 cgs")
print(f"Throughput range: {np.nanpercentile(thru,1):.3f} – "
      f"{np.nanpercentile(thru,99):.3f}")
print(f"FWHM range: {np.nanpercentile(fwhm,1):.2f} – "
      f"{np.nanpercentile(fwhm,99):.2f}  arcsec")


# =============================================================================
# CELL 6 — GRIDDING FUNCTION
# =============================================================================

def sky_grid(ra_f, dec_f, values, resolution_deg,
             ra_range=None, dec_range=None, statistic="median",
             smooth=0.5):
    """
    Bin (ra_f, dec_f, values) onto a regular sky grid.

    Parameters
    ----------
    statistic : "median" | "mean" | "min" | "count" | "std"
    smooth    : Gaussian smoothing sigma in pixels (0 = no smoothing)

    Returns
    -------
    grid   : 2D array  (n_dec, n_ra) — display orientation
    ra_cen : 1D array of RA  bin centres
    dc_cen : 1D array of Dec bin centres
    """
    if ra_range is None:
        pad      = resolution_deg * 3
        ra_range = (ra_f.min() - pad, ra_f.max() + pad)
    if dec_range is None:
        pad       = resolution_deg * 3
        dec_range = (dec_f.min() - pad, dec_f.max() + pad)

    n_ra  = max(4, int((ra_range[1]  - ra_range[0])  / resolution_deg))
    n_dec = max(4, int((dec_range[1] - dec_range[0]) / resolution_deg))

    result = binned_statistic_2d(
        ra_f, dec_f, values,
        statistic=statistic,
        bins=[n_ra, n_dec],
        range=[ra_range, dec_range],
    )
    # grid is (n_ra, n_dec); transpose to (n_dec, n_ra) for imshow
    grid   = result.statistic.T
    ra_cen = 0.5 * (result.x_edge[:-1] + result.x_edge[1:])
    dc_cen = 0.5 * (result.y_edge[:-1] + result.y_edge[1:])

    if smooth > 0:
        mask_nan = ~np.isfinite(grid)
        grid_s   = np.where(mask_nan, 0.0, grid)
        grid_s   = gaussian_filter(grid_s, sigma=smooth)
        # Zero-weight smooth in masked region
        ones     = gaussian_filter(np.where(mask_nan, 0.0, 1.0), sigma=smooth)
        grid     = np.where(ones > 0.05, grid_s / ones, np.nan)

    return grid, ra_cen, dc_cen


# =============================================================================
# CELL 7 — PRECOMPUTE ALL GRIDS
# =============================================================================

# Handle dex-fall RA wrap (crosses 0°)
ra_plot = ra.copy()
fall_mask = (field == "dex-fall") & (ra_plot > 180)
ra_plot[fall_mask] -= 360

# ── Full-survey noise grid ─────────────────────────────────────────────────────
print("\nComputing full-survey sensitivity grid ...")
G_full_noise, ra_full, dc_full = sky_grid(
    ra_plot, dec, noise,
    resolution_deg=RES_FULL, statistic="median", smooth=0.5,
)
G_full_count, _, _ = sky_grid(
    ra_plot, dec, noise,
    resolution_deg=RES_FULL, statistic="count",  smooth=0.0,
)

# ── Per-field grids ────────────────────────────────────────────────────────────
FIELD_ORDER = ["dex-spring", "dex-fall", "cosmos", "goods-n", "nep", "ssa22"]
FIELD_RES   = {
    "dex-spring": RES_WIDE, "dex-fall": RES_WIDE,
    "cosmos": RES_DEEP, "goods-n": RES_DEEP,
    "nep": RES_DEEP,    "ssa22":   RES_DEEP,
}

field_grids = {}   # {fname: {"noise","thru","fwhm","count"}}

for fname in FIELD_ORDER:
    fm   = field == fname
    ra_f = ra_plot[fm]
    dc_f = dec[fm]
    if fm.sum() < 10:
        field_grids[fname] = None
        continue
    res  = FIELD_RES[fname]
    print(f"  {fname}: {fm.sum():,} detections @ {res}°/pix")
    d    = {}
    for stat_key, arr_key, arr_data in [
        ("noise", "noise", noise[fm]),
        ("thru",  "thru",  thru[fm]),
        ("fwhm",  "fwhm",  fwhm[fm]),
        ("count", "noise", noise[fm]),   # count uses same positions
    ]:
        stat  = "count" if stat_key == "count" else "median"
        g, ra_c, dc_c = sky_grid(ra_f, dc_f, arr_data,
                                  resolution_deg=res, statistic=stat)
        d[stat_key] = (g, ra_c, dc_c)
    field_grids[fname] = d


# ── Limiting luminosity calculation ───────────────────────────────────────────
DL_ref   = cosmo.luminosity_distance(Z_REF).to(u.cm).value
# 5σ limiting flux → luminosity
# L = 4π DL² × F_5sigma  [erg/s],  F in erg/s/cm²
# catalog noise is in 1e-17 erg/s/cm²
FLUX_SCALE = 1e-17

def noise_to_logL5sig(noise_arr, DL_cm=DL_ref):
    F5   = 5.0 * noise_arr * FLUX_SCALE
    L    = 4.0 * np.pi * DL_cm**2 * F5
    return np.log10(np.clip(L, 1e30, None))

# Limiting logL map for full survey
with np.errstate(invalid="ignore"):
    G_logL = noise_to_logL5sig(G_full_noise)


# =============================================================================
# CELL 8 — PLOTTING
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

FIELD_COLORS = {
    "dex-spring": "#58a6ff", "dex-fall" : "#3fb950",
    "cosmos"    : "#f78166", "goods-n"  : "#d2a8ff",
    "nep"       : "#ffa657", "ssa22"    : "#79c0ff",
}

# Diverging sensitivity colourmap: deep (low noise) = blue, shallow = red/yellow
SENS_CMAP  = "plasma_r"     # low noise = bright / high noise = dark
THRU_CMAP  = "viridis"
FWHM_CMAP  = "RdYlGn_r"     # low fwhm (good seeing) = green, high = red

def style_ax(ax, title="", xl="", yl="", invert=True):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, which="both", direction="in",
                   top=True, right=True, labelsize=7.5)
    if xl: ax.set_xlabel(xl, color=TEXT, fontsize=8.5)
    if yl: ax.set_ylabel(yl, color=TEXT, fontsize=8.5)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9,
                     fontweight="bold", loc="left", pad=4)
    if invert:
        ax.invert_xaxis()

def _style_cb(cb, label, fs=7.0):
    """Apply consistent dark styling to a colorbar."""
    cb.set_label(label, color=MUTED, fontsize=fs)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=6.5)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    cb.outline.set_edgecolor(SPINE)
    return cb

def colorbar(fig, ax, im, label, size="3%", pad=0.05, fs=7.0):
    """Thin colourbar attached to ax — kept small so it does not steal space."""
    cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02, shrink=0.85)
    return _style_cb(cb, label, fs)

def show_grid(ax, grid, ra_c, dc_c, cmap, vmin, vmax,
              log=False, smooth_disp=0.0):
    """Display a pre-computed sky grid on ax (RA inverted, Dec upright)."""
    if smooth_disp > 0:
        g2 = np.where(np.isfinite(grid), grid, 0.0)
        g2 = gaussian_filter(g2, sigma=smooth_disp)
        grid = np.where(np.isfinite(grid), g2, np.nan)
    extent = [ra_c[0], ra_c[-1], dc_c[0], dc_c[-1]]
    norm   = (mcolors.LogNorm(vmin=vmin, vmax=vmax) if log
              else mcolors.Normalize(vmin=vmin, vmax=vmax))
    im = ax.imshow(
        grid, origin="lower", extent=extent,
        cmap=cmap, norm=norm, aspect="auto",
        interpolation="nearest", rasterized=True,
    )
    return im

# ── Colour scale limits ────────────────────────────────────────────────────────
noise_p2, noise_p98 = (np.nanpercentile(G_full_noise, p)
                       for p in [2, 98])
thru_p2,  thru_p98  = (np.nanpercentile(thru, p) for p in [2, 98])
fwhm_p2,  fwhm_p98  = (np.nanpercentile(fwhm, p) for p in [2, 98])

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 17))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    4, 6, figure=fig,
    hspace=0.52, wspace=0.12,
    left=0.04, right=0.88,
    top=0.93,  bottom=0.06,
    height_ratios=[2.0, 1.6, 1.6, 1.2],
)

ax_full = fig.add_subplot(gs[0, :5])    # full-survey sensitivity panorama
ax_logL = fig.add_subplot(gs[0, 5])    # limiting logL map (reuses full grid)

# Row 1: dex-spring, dex-fall, cosmos, goods-n  (noise maps)
ax_sp   = fig.add_subplot(gs[1, 0:2])
ax_fa   = fig.add_subplot(gs[1, 2:4])
ax_co   = fig.add_subplot(gs[1, 4])
ax_gn   = fig.add_subplot(gs[1, 5])

# Row 2: nep, ssa22, throughput map, fwhm map
ax_nep  = fig.add_subplot(gs[2, 0])
ax_ssa  = fig.add_subplot(gs[2, 1])
ax_thru = fig.add_subplot(gs[2, 2:4])
ax_fwhm = fig.add_subplot(gs[2, 4:6])

# Row 3: noise histogram, limiting logL vs z, depth per field bar
ax_hist = fig.add_subplot(gs[3, 0:2])
ax_llz  = fig.add_subplot(gs[3, 2:4])
ax_bar  = fig.add_subplot(gs[3, 4:6])

# ── Panel 0: Full-survey sensitivity panorama ──────────────────────────────────
style_ax(ax_full,
         title=(f"HETDEX SC2 — Survey Sensitivity  "
                f"(median 1σ line flux noise per {RES_FULL}° pixel)"),
         xl="RA  (deg)", yl="Dec  (deg)")

im_full = show_grid(ax_full, G_full_noise,
                    ra_full, dc_full,
                    SENS_CMAP, noise_p2, noise_p98, log=False)
# cb_full handled by shared right-strip colourbar

# Tight limits: only show sky where data actually exists
# ra_plot goes from ~-50 (dex-fall shifted) to ~235 (dex-spring)
# but 90% of that range is empty. Clamp to data extent.
ra_data_lo = np.nanmin(ra_plot) - 1
ra_data_hi = np.nanmax(ra_plot) + 1
dec_data_lo = np.nanmin(dec) - 0.5
dec_data_hi = np.nanmax(dec) + 0.5
ax_full.set_xlim(ra_data_hi, ra_data_lo)   # inverted RA
ax_full.set_ylim(dec_data_lo, dec_data_hi)
xtk = ax_full.get_xticks()
ax_full.set_xticklabels([f"{t%360:.0f}°" for t in xtk],
                        color=MUTED, fontsize=7)

# Annotate field centres
field_label_pos = {
    "dex-spring": (190, 50.5), "dex-fall": (10, 0.5),
    "cosmos":     (150.1, 2.2),"goods-n": (189.3, 62.2),
    "nep":        (269, 66.5), "ssa22":   (334, 0.4),
}
for fname, (rx, dy) in field_label_pos.items():
    rx_plot = rx if fname != "dex-fall" else rx - 360
    ax_full.text(rx_plot, dy + 1.5, fname.replace("dex-",""),
                color=FIELD_COLORS[fname], fontsize=7.5,
                ha="center", fontweight="bold", zorder=6)

# ── Panel 0b: Limiting logL map ───────────────────────────────────────────────
style_ax(ax_logL,
         title=f"5σ lim. logL  (z={Z_REF})",
         xl="RA", yl="Dec")

logL_p2, logL_p98 = (np.nanpercentile(G_logL[np.isfinite(G_logL)], p)
                     for p in [2, 98])
im_logL = show_grid(ax_logL, G_logL, ra_full, dc_full,
                    "plasma_r", logL_p2, logL_p98)
# cb_logL handled by shared right-strip colourbar
ax_logL.set_xlim(ra_data_hi, ra_data_lo)
ax_logL.set_ylim(dec_data_lo, dec_data_hi)
ax_logL.set_xticklabels(
    [f"{t%360:.0f}°" for t in ax_logL.get_xticks()],
    color=MUTED, fontsize=7)

# ── Row 1: dex-spring ─────────────────────────────────────────────────────────
style_ax(ax_sp, title=f"dex-spring", xl="RA (°)", yl="Dec (°)")
if field_grids["dex-spring"]:
    g, rc, dc = field_grids["dex-spring"]["noise"]
    im_sp = show_grid(ax_sp, g, rc, dc, SENS_CMAP,
                      noise_p2, noise_p98)
    ax_sp.set_xlim(rc.max(), rc.min())
    ax_sp.set_ylim(dc.min(), dc.max())
    pass  # noise scale shown in right-strip cb

# ── Row 1: dex-fall ───────────────────────────────────────────────────────────
style_ax(ax_fa, title=f"dex-fall", xl="RA (°)", yl="Dec (°)")
if field_grids["dex-fall"]:
    g, rc, dc = field_grids["dex-fall"]["noise"]
    im_fa = show_grid(ax_fa, g, rc, dc, SENS_CMAP,
                      noise_p2, noise_p98)
    ax_fa.set_xlim(rc.max(), rc.min())
    ax_fa.set_ylim(dc.min(), dc.max())
    ax_fa.set_xticklabels(
        [f"{t%360:.1f}" for t in ax_fa.get_xticks()],
        color=MUTED, fontsize=7)

# ── Row 1: COSMOS ─────────────────────────────────────────────────────────────
style_ax(ax_co, title="COSMOS", xl="RA (°)", yl="Dec (°)")
if field_grids["cosmos"]:
    g, rc, dc = field_grids["cosmos"]["noise"]
    # COSMOS: use its own noise range (much lower noise)
    co_lo = np.nanpercentile(g[np.isfinite(g)], 2)
    co_hi = np.nanpercentile(g[np.isfinite(g)], 98)
    im_co = show_grid(ax_co, g, rc, dc, SENS_CMAP, co_lo, co_hi)
    im_co = show_grid(ax_co, g, rc, dc, SENS_CMAP, co_lo, co_hi)
    ax_co.set_xlim(rc.max(), rc.min())
    ax_co.set_ylim(dc.min(), dc.max())

# ── Row 1: GOODS-N ────────────────────────────────────────────────────────────
style_ax(ax_gn, title="GOODS-N", xl="RA (°)", yl="Dec (°)")
if field_grids["goods-n"]:
    g, rc, dc = field_grids["goods-n"]["noise"]
    gn_lo = np.nanpercentile(g[np.isfinite(g)], 2)
    gn_hi = np.nanpercentile(g[np.isfinite(g)], 98)
    im_gn = show_grid(ax_gn, g, rc, dc, SENS_CMAP, gn_lo, gn_hi)
    ax_gn.set_xlim(rc.max(), rc.min())
    ax_gn.set_ylim(dc.min(), dc.max())

# ── Row 2: NEP ────────────────────────────────────────────────────────────────
style_ax(ax_nep, title="NEP", xl="RA (°)", yl="Dec (°)")
if field_grids["nep"]:
    g, rc, dc = field_grids["nep"]["noise"]
    nep_lo = np.nanpercentile(g[np.isfinite(g)], 2)
    nep_hi = np.nanpercentile(g[np.isfinite(g)], 98)
    im_nep = show_grid(ax_nep, g, rc, dc, SENS_CMAP, nep_lo, nep_hi)
    ax_nep.set_xlim(rc.max(), rc.min())
    ax_nep.set_ylim(dc.min(), dc.max())

# ── Row 2: SSA22 ──────────────────────────────────────────────────────────────
style_ax(ax_ssa, title="SSA22", xl="RA (°)", yl="Dec (°)")
if field_grids["ssa22"]:
    g, rc, dc = field_grids["ssa22"]["noise"]
    ssa_lo = np.nanpercentile(g[np.isfinite(g)], 2)
    ssa_hi = np.nanpercentile(g[np.isfinite(g)], 98)
    im_ssa = show_grid(ax_ssa, g, rc, dc, SENS_CMAP, ssa_lo, ssa_hi)
    ax_ssa.set_xlim(rc.max(), rc.min())
    ax_ssa.set_ylim(dc.min(), dc.max())

# ── Row 2: Throughput map (full survey) ───────────────────────────────────────
style_ax(ax_thru,
         title=f"Spectral throughput at 4540 Å  (median per {RES_FULL}° pixel)",
         xl="RA (deg)", yl="Dec (deg)")

G_thru, _, _ = sky_grid(ra_plot, dec, thru,
                         resolution_deg=RES_FULL, statistic="median")
im_thru = show_grid(ax_thru, G_thru, ra_full, dc_full,
                    THRU_CMAP, thru_p2, thru_p98)
# throughput scale shown in right-strip cb
ax_thru.set_xlim(ra_data_hi, ra_data_lo)
ax_thru.set_ylim(dec_data_lo, dec_data_hi)
ax_thru.set_xticklabels(
    [f"{t%360:.0f}°" for t in ax_thru.get_xticks()],
    color=MUTED, fontsize=7)

# ── Row 2: Seeing FWHM map (full survey) ──────────────────────────────────────
style_ax(ax_fwhm,
         title=f"Seeing FWHM  (median per {RES_FULL}° pixel)",
         xl="RA (deg)", yl="Dec (deg)")

G_fwhm, _, _ = sky_grid(ra_plot, dec, fwhm,
                         resolution_deg=RES_FULL, statistic="median")
im_fwhm = show_grid(ax_fwhm, G_fwhm, ra_full, dc_full,
                    FWHM_CMAP, fwhm_p2, fwhm_p98)
# fwhm scale shown in right-strip cb
ax_fwhm.set_xlim(ra_data_hi, ra_data_lo)
ax_fwhm.set_ylim(dec_data_lo, dec_data_hi)
ax_fwhm.set_xticklabels(
    [f"{t%360:.0f}°" for t in ax_fwhm.get_xticks()],
    color=MUTED, fontsize=7)

# ── Row 3: Noise histogram per field ──────────────────────────────────────────
ax_hist.set_facecolor(AX_BG)
for sp in ax_hist.spines.values(): sp.set_color(SPINE)
ax_hist.tick_params(colors=MUTED, which="both", direction="in",
                    labelsize=8, top=True, right=True)
ax_hist.set_title("1σ noise distribution per field",
                  color=TEXT, fontsize=9, fontweight="bold",
                  loc="left", pad=4)
ax_hist.set_xlabel(r"1σ noise  [×10$^{-17}$ erg s$^{-1}$ cm$^{-2}$]",
                   color=TEXT, fontsize=8.5)
ax_hist.set_ylabel("Normalised density", color=TEXT, fontsize=8.5)

noise_bins = np.linspace(0.5, 20, 60)
for fname in FIELD_ORDER:
    fm = field == fname
    if fm.sum() < 5: continue
    n_f = noise[fm]; n_f = n_f[np.isfinite(n_f)]
    ax_hist.hist(n_f, bins=noise_bins, density=True,
                 color=FIELD_COLORS[fname], alpha=0.45,
                 histtype="stepfilled", label=fname)
    ax_hist.hist(n_f, bins=noise_bins, density=True,
                 color=FIELD_COLORS[fname], lw=1.2,
                 histtype="step")

ax_hist.set_xlim(0.5, 20)
ax_hist.xaxis.set_minor_locator(AutoMinorLocator())
ax_hist.yaxis.set_minor_locator(AutoMinorLocator())
ax_hist.legend(fontsize=7, facecolor="#21262d",
               edgecolor=SPINE, labelcolor=TEXT,
               loc="upper right", ncol=2)

# Median lines
for fname in FIELD_ORDER:
    fm = field == fname
    if fm.sum() < 5: continue
    n_f = noise[fm]; n_f = n_f[np.isfinite(n_f)]
    med = np.median(n_f)
    ax_hist.axvline(med, color=FIELD_COLORS[fname],
                    lw=0.9, ls="--", alpha=0.8)

# ── Row 3: Limiting logL vs redshift ──────────────────────────────────────────
ax_llz.set_facecolor(AX_BG)
for sp in ax_llz.spines.values(): sp.set_color(SPINE)
ax_llz.tick_params(colors=MUTED, which="both", direction="in",
                   labelsize=8, top=True, right=True)
ax_llz.set_title(
    r"5σ limiting $\log L_{\rm Ly\alpha}$ vs redshift  (per field)",
    color=TEXT, fontsize=9, fontweight="bold", loc="left", pad=4)
ax_llz.set_xlabel("Redshift  z", color=TEXT, fontsize=8.5)
ax_llz.set_ylabel(r"$\log_{10} L_{\rm lim}$  [erg/s]",
                  color=TEXT, fontsize=8.5)

z_grid = np.linspace(1.9, 3.6, 200)
DL_grid = cosmo.luminosity_distance(z_grid).to(u.cm).value

for fname in FIELD_ORDER:
    fm  = field == fname
    if fm.sum() < 5: continue
    n_f = noise[fm]; n_f = n_f[np.isfinite(n_f)]
    # Three percentile curves: p16, p50, p84
    for pct, ls, alpha in [(16,":",0.5),(50,"-",0.85),(84,":",0.5)]:
        noise_pct = np.percentile(n_f, pct)
        F5        = 5.0 * noise_pct * FLUX_SCALE
        logL_lim  = np.log10(4 * np.pi * DL_grid**2 * F5)
        lbl       = fname if pct == 50 else None
        ax_llz.plot(z_grid, logL_lim,
                    color=FIELD_COLORS[fname], ls=ls,
                    lw=1.3 if pct == 50 else 0.7,
                    alpha=alpha, label=lbl)

ax_llz.set_xlim(1.9, 3.6)
ax_llz.xaxis.set_minor_locator(AutoMinorLocator())
ax_llz.yaxis.set_minor_locator(AutoMinorLocator())
ax_llz.legend(fontsize=7, facecolor="#21262d",
              edgecolor=SPINE, labelcolor=TEXT,
              loc="upper left", ncol=2)

# Indicate survey lower limits
for log_l, label in [(41.5, "logL=41.5"), (42.0, "42.0"), (42.5, "42.5")]:
    ax_llz.axhline(log_l, color=MUTED, lw=0.6, ls=":", alpha=0.5)
    ax_llz.text(3.55, log_l + 0.02, label,
                color=MUTED, fontsize=6.5, ha="right", va="bottom")

# ── Row 3: Depth summary bar chart ────────────────────────────────────────────
ax_bar.set_facecolor(AX_BG)
for sp in ax_bar.spines.values(): sp.set_color(SPINE)
ax_bar.tick_params(colors=MUTED, which="both", direction="in",
                   labelsize=8, top=False, right=True)
ax_bar.set_title("Median 1σ noise ± 16/84th pct per field",
                 color=TEXT, fontsize=9, fontweight="bold",
                 loc="left", pad=4)
ax_bar.set_xlabel("Field", color=TEXT, fontsize=8.5)
ax_bar.set_ylabel(r"1σ noise  [×10$^{-17}$ cgs]",
                  color=TEXT, fontsize=8.5)

x_pos = np.arange(len(FIELD_ORDER))
meds  = []; errs_lo = []; errs_hi = []
for fname in FIELD_ORDER:
    fm  = field == fname
    n_f = noise[fm]; n_f = n_f[np.isfinite(n_f)]
    med = np.median(n_f) if len(n_f) > 0 else np.nan
    p16 = np.percentile(n_f, 16) if len(n_f) > 0 else np.nan
    p84 = np.percentile(n_f, 84) if len(n_f) > 0 else np.nan
    meds.append(med)
    errs_lo.append(med - p16)
    errs_hi.append(p84 - med)

bars = ax_bar.bar(x_pos, meds,
                  color=[FIELD_COLORS[f] for f in FIELD_ORDER],
                  alpha=0.80, edgecolor=SPINE, linewidth=0.5)
ax_bar.errorbar(x_pos, meds,
                yerr=[errs_lo, errs_hi],
                fmt="none", color=TEXT,
                elinewidth=1.2, capsize=4, capthick=1.0)

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(
    [f.replace("dex-","") for f in FIELD_ORDER],
    color=TEXT, fontsize=8)
for tick, fname in zip(ax_bar.get_xticklabels(), FIELD_ORDER):
    tick.set_color(FIELD_COLORS[fname])

for xp, med in zip(x_pos, meds):
    if np.isfinite(med):
        ax_bar.text(xp, med * 1.02, f"{med:.1f}",
                    ha="center", va="bottom",
                    color=TEXT, fontsize=7.5)

ax_bar.yaxis.set_minor_locator(AutoMinorLocator())

# Colour norms for shared colourbars
z_cmap   = plt.cm.coolwarm
z_norm   = mcolors.Normalize(
    vmin=np.nanpercentile(z_lae if 'z_lae' in dir() else
                          np.array([2.0,3.5]), 5),
    vmax=np.nanpercentile(z_lae if 'z_lae' in dir() else
                          np.array([2.0,3.5]), 95))

# ── Shared colourbars in right-margin strip ──────────────────────────────────
# Noise scale (plasma_r) — shared by all sensitivity panels
cax_noise = fig.add_axes([0.895, 0.58, 0.012, 0.32])   # [left, bottom, w, h]
sm_noise  = plt.cm.ScalarMappable(
    cmap=SENS_CMAP,
    norm=mcolors.Normalize(vmin=noise_p2, vmax=noise_p98))
sm_noise.set_array([])
cb_noise = fig.colorbar(sm_noise, cax=cax_noise)
_style_cb(cb_noise,
          r"1σ noise  [×10$^{-17}$ erg s$^{-1}$ cm$^{-2}$]",
          fs=7.5)
cb_noise.ax.set_title("depth", color=MUTED, fontsize=7, pad=4)

# Throughput scale (viridis)
cax_thru2 = fig.add_axes([0.895, 0.38, 0.012, 0.15])
sm_thru2  = plt.cm.ScalarMappable(
    cmap=THRU_CMAP,
    norm=mcolors.Normalize(vmin=thru_p2, vmax=thru_p98))
sm_thru2.set_array([])
cb_thru2 = fig.colorbar(sm_thru2, cax=cax_thru2)
_style_cb(cb_thru2, "Throughput", fs=7.5)

# FWHM scale (RdYlGn_r)
cax_fwhm2 = fig.add_axes([0.895, 0.20, 0.012, 0.15])
sm_fwhm2  = plt.cm.ScalarMappable(
    cmap=FWHM_CMAP,
    norm=mcolors.Normalize(vmin=fwhm_p2, vmax=fwhm_p98))
sm_fwhm2.set_array([])
cb_fwhm2 = fig.colorbar(sm_fwhm2, cax=cax_fwhm2)
_style_cb(cb_fwhm2, "FWHM (arcsec)", fs=7.5)

# z colourbar (scatter overlay)
cax_z2 = fig.add_axes([0.895, 0.06, 0.012, 0.11])
sm_z2  = plt.cm.ScalarMappable(cmap=z_cmap, norm=z_norm)
sm_z2.set_array([])
cb_z2 = fig.colorbar(sm_z2, cax=cax_z2)
_style_cb(cb_z2, "z (scatter)", fs=7.5)

# ── Super-title & footer ───────────────────────────────────────────────────────
syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
fig.suptitle(
    "HETDEX SC2 — Survey Sensitivity Map  "
    r"($\mathtt{flux\_noise\_1sigma\_obs}$ per sky pixel)" + syn_tag,
    color=TEXT, fontsize=13, fontweight="bold", y=0.975,
)
fig.text(
    0.5, 0.005,
    (f"Detections: {valid.sum():,}   |   "
     f"Noise clipped at {MAX_NOISE} ×10⁻¹⁷   |   "
     f"Grid: {RES_FULL}°/px (full), {RES_WIDE}°/px (wide), "
     f"{RES_DEEP}°/px (deep)   |   "
     f"5σ limiting logL at z={Z_REF} from Planck18"),
    ha="center", fontsize=8, color=MUTED,
)

if SAVE_PATH:
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 9 — NUMERICAL SUMMARY
# =============================================================================

print("\n" + "=" * 68)
print("  HETDEX SC2 — Survey Sensitivity Summary")
print("=" * 68)
print(f"  {'Field':<14}  {'N det':>9}  {'p16':>6}  {'med':>6}  "
      f"{'p84':>6}  {'5σ logL(z={})'.format(Z_REF):>14}")
print("  " + "-" * 66)

for fname in FIELD_ORDER:
    fm   = field == fname
    n_f  = noise[fm]; n_f = n_f[np.isfinite(n_f)]
    if len(n_f) == 0:
        print(f"  {fname:<14}  {'no data':>9}")
        continue
    p16, med, p84 = np.percentile(n_f, [16, 50, 84])
    DL_r = cosmo.luminosity_distance(Z_REF).to(u.cm).value
    logL5 = np.log10(4 * np.pi * DL_r**2 * 5 * med * FLUX_SCALE)
    print(f"  {fname:<14}  {fm.sum():>9,}  {p16:>6.2f}  {med:>6.2f}  "
          f"{p84:>6.2f}  {logL5:>14.2f}")

print("=" * 68)
print(f"\n  Throughput range : {np.nanpercentile(thru,2):.3f} – "
      f"{np.nanpercentile(thru,98):.3f}")
print(f"  FWHM range       : {np.nanpercentile(fwhm,2):.2f} – "
      f"{np.nanpercentile(fwhm,98):.2f}  arcsec")
print(f"  Noise units      : ×10⁻¹⁷ erg s⁻¹ cm⁻²")
print(f"  5σ flux limit at median noise (dex-spring): "
      f"{5 * float(np.nanmedian(noise[field=='dex-spring'])):.1f} ×10⁻¹⁷ cgs")
