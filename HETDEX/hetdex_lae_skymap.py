"""
hetdex_lae_skymap.py
====================
Sky density map of HETDEX SC2 LAE sources using matplotlib and astropy.
Plots RA/DEC in all survey fields with 2D histograms, per-field zoom panels,
and a redshift-coloured scatter overlay for the compact fields.

HETDEX survey fields
---------------------
  dex-spring   Primary spring field   RA ~130–235°, Dec ~42–58°   (~360 deg²)
  dex-fall     Primary fall field     RA ~320–40°,  Dec ~-2–4°    (~150 deg²)
  cosmos       COSMOS deep field      RA ~150°,     Dec ~+2°       (~3 deg²)
  goods-n      GOODS-N deep field     RA ~189°,     Dec ~+62°      (~0.05 deg²)
  nep          North Ecliptic Pole    RA ~266–272°, Dec ~65–68°    (~9 deg²)
  ssa22        SSA22 proto-cluster    RA ~334°,     Dec ~+0°       (~1 deg²)

Strategy
--------
1. Full-survey panel: 2D histogram of all LAEs in equatorial coordinates,
   RA plotted right-to-left (astronomical convention), coloured by log10
   source density, with per-field ellipse annotations.
2. Six per-field zoom panels: individual 2D histograms at native field
   resolution, with logL_lya colour scatter overlay.
3. Redshift evolution stripe: N(RA) marginal histogram for dex-spring,
   the largest field, coloured by mean redshift per RA slice.
4. Summary statistics panel: source count, sky area, and surface density
   per field.

Column names used (from SC2 source observation table, hetdex_sc2_v1.5.fits)
  RA, DEC, source_type, field, z_hetdex, logL_lya, sn, p_conf, p_cnn

Requirements
------------
  pip install astropy numpy matplotlib scipy

Data
----
  hetdex_sc2_v1.5.fits
  https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/hetdex_source_catalog_2/
"""

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

CATALOG_PATH = "hetdex_sc2_v1.5.fits"
SAVE_PATH    = "hetdex_lae_skymap.png"

# Quality cuts
MIN_SN     = 5.5
MIN_P_CONF = 0.5
MIN_P_CNN  = 0.5
BAD        = -999.0

# Histogram resolution per panel
FULL_BINS_RA  = 400    # full-survey panorama
FULL_BINS_DEC = 200
FIELD_BINS    = 80     # per-field zoom panels

# Known survey field footprints (ra_min, ra_max, dec_min, dec_max)
# Used for annotations — not for filtering (field column handles that)
FIELD_BOXES = {
    "dex-spring": (130.0, 235.0,  42.0,  58.0),
    "dex-fall"  : (320.0,  40.0,  -3.0,   5.0),   # wraps RA=0
    "cosmos"    : (149.5, 150.7,   1.7,   2.7),
    "goods-n"   : (188.9, 189.7,  62.0,  62.4),
    "nep"       : (264.0, 274.0,  64.5,  68.5),
    "ssa22"     : (333.5, 334.5,   0.0,   0.8),
}

FIELD_COLORS = {
    "dex-spring": "#58a6ff",
    "dex-fall"  : "#3fb950",
    "cosmos"    : "#f78166",
    "goods-n"   : "#d2a8ff",
    "nep"       : "#ffa657",
    "ssa22"     : "#79c0ff",
}

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
import matplotlib.patches as mpatches
from matplotlib.ticker  import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

from astropy.io    import fits
from astropy.table import Table
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

print("Imports OK.")

# =============================================================================
# CELL 3 — SYNTHETIC DATA GENERATOR
# =============================================================================

def make_synthetic_sc2(n=200_000, seed=42):
    """
    Realistic synthetic SC2 catalog mirroring the real survey footprint,
    class proportions, and redshift distributions.
    """
    rng = np.random.default_rng(seed)

    # Field populations (approximate real fractions)
    fields_cfg = [
        ("dex-spring", 0.50, 160, 50.0, 25,  3.0, (2.0, 3.4)),
        ("dex-fall",   0.33, 355,  0.5, 20,  3.5, (2.0, 3.4)),
        ("cosmos",     0.10, 150.1, 2.2, 0.6,0.5, (1.9, 3.5)),
        ("goods-n",    0.04, 189.3,62.2, 0.4,0.3, (1.9, 3.5)),
        ("nep",        0.02, 269.0,66.5, 2.0,1.5, (2.0, 3.2)),
        ("ssa22",      0.01, 334.0, 0.4, 0.4,0.4, (2.0, 3.0)),
    ]

    ra_all   = []
    dec_all  = []
    fld_all  = []
    z_all    = []
    sn_all   = []
    logL_all = []
    pc_all   = []
    pn_all   = []

    for fname, frac, ra_c, dec_c, ra_s, dec_s, z_range in fields_cfg:
        n_f   = int(n * frac)
        ra_f  = (rng.normal(ra_c, ra_s, n_f)) % 360
        dec_f = rng.normal(dec_c, dec_s, n_f)
        z_f   = rng.uniform(*z_range, n_f)
        sn_f  = np.abs(rng.lognormal(2.0, 0.5, n_f))
        logL_f= rng.uniform(42.0, 44.2, n_f)
        pc_f  = np.clip(rng.beta(4, 1.5, n_f), 0, 1)
        pn_f  = np.clip(rng.beta(3.5, 1.3, n_f), 0, 1)

        ra_all.extend(ra_f);   dec_all.extend(dec_f)
        fld_all.extend([fname]*n_f)
        z_all.extend(z_f);     sn_all.extend(sn_f)
        logL_all.extend(logL_f); pc_all.extend(pc_f)
        pn_all.extend(pn_f)

    tab = Table({
        "RA"          : np.array(ra_all,   dtype=np.float32),
        "DEC"         : np.array(dec_all,  dtype=np.float32),
        "source_type" : np.array(["lae"] * len(ra_all)),
        "field"       : np.array(fld_all),
        "z_hetdex"    : np.array(z_all,    dtype=np.float32),
        "logL_lya"    : np.array(logL_all, dtype=np.float32),
        "sn"          : np.array(sn_all,   dtype=np.float32),
        "p_conf"      : np.array(pc_all,   dtype=np.float32),
        "p_cnn"       : np.array(pn_all,   dtype=np.float32),
    })
    print(f"  Synthetic: {len(tab):,} sources across "
          f"{len(set(fld_all))} fields")
    return tab


# =============================================================================
# CELL 4 — LOAD CATALOG
# =============================================================================

def getcol(tab, *cands):
    lc = {c.lower(): c for c in tab.colnames}
    for c in cands:
        if c.lower() in lc:
            return lc[c.lower()]
    raise KeyError(f"None of {cands} found. "
                   f"Available: {list(tab.colnames)[:20]}")

print("Loading catalog ...")
try:
    hdul = fits.open(CATALOG_PATH, memmap=True)
    tab  = Table(hdul[1].data)
    hdul.close()
    tab.rename_columns(tab.colnames,
                       [c.lower() for c in tab.colnames])
    print(f"  Loaded {len(tab):,} rows, {len(tab.colnames)} columns")
    SYNTHETIC = False
except FileNotFoundError:
    print(f"  '{CATALOG_PATH}' not found — using synthetic data.")
    tab       = make_synthetic_sc2()
    SYNTHETIC = True
    tab.rename_columns(tab.colnames,
                       [c.lower() for c in tab.colnames])

# Resolve column names
RA_COL    = getcol(tab, "ra")
DEC_COL   = getcol(tab, "dec")
STYPE_COL = getcol(tab, "source_type")
FIELD_COL = getcol(tab, "field")
Z_COL     = getcol(tab, "z_hetdex")
LOGL_COL  = getcol(tab, "logl_lya")
SN_COL    = getcol(tab, "sn")
PCONF_COL = getcol(tab, "p_conf")
PCNN_COL  = getcol(tab, "p_cnn")

# =============================================================================
# CELL 5 — SELECT LAEs
# =============================================================================

stype = np.array([s.strip().lower() for s in tab[STYPE_COL]])
ra    = np.array(tab[RA_COL],    dtype=float)
dec   = np.array(tab[DEC_COL],   dtype=float)
z     = np.array(tab[Z_COL],     dtype=float)
logL  = np.array(tab[LOGL_COL],  dtype=float)
sn    = np.array(tab[SN_COL],    dtype=float)
pconf = np.array(tab[PCONF_COL], dtype=float)
pcnn  = np.array(tab[PCNN_COL],  dtype=float)
field = np.array([f.strip().lower() for f in tab[FIELD_COL]])

# Replace bad sentinels
for arr in [z, logL, sn, pconf, pcnn]:
    arr[arr == BAD] = np.nan

# LAE quality mask
mask = (
    (stype == "lae") &
    (sn    >= MIN_SN)    & np.isfinite(sn)    &
    (pconf >= MIN_P_CONF)& np.isfinite(pconf) &
    (pcnn  >= MIN_P_CNN) & np.isfinite(pcnn)  &
    np.isfinite(ra) & np.isfinite(dec)
)

ra_lae    = ra[mask];    dec_lae  = dec[mask]
z_lae     = z[mask];    logL_lae = logL[mask]
field_lae = field[mask]; sn_lae   = sn[mask]

# Identify unique fields present in the data
unique_fields = [f for f in
                 ["dex-spring","dex-fall","cosmos","goods-n","nep","ssa22"]
                 if f in np.unique(field_lae)]

print(f"\nLAEs selected: {mask.sum():,} / {len(tab):,}")
print(f"Fields present: {unique_fields}")
for f in unique_fields:
    n_f = (field_lae == f).sum()
    area = ((FIELD_BOXES[f][1] - FIELD_BOXES[f][0]) *
             (FIELD_BOXES[f][3] - FIELD_BOXES[f][2])
             if f != "dex-fall" else
             ((FIELD_BOXES[f][1] + 360 - FIELD_BOXES[f][0]) *
              (FIELD_BOXES[f][3] - FIELD_BOXES[f][2])))
    area_approx = abs(area) * np.cos(np.radians(
        0.5*(FIELD_BOXES[f][2]+FIELD_BOXES[f][3])))
    print(f"  {f:12s}: {n_f:8,} LAEs")

# =============================================================================
# CELL 6 — HELPER: 2D HISTOGRAM FOR ONE FIELD
# =============================================================================

def field_hist2d(ra_f, dec_f, n_bins=FIELD_BINS,
                 ra_range=None, dec_range=None, smooth=0.8):
    """
    Compute a 2D histogram of (RA, Dec) for one field.
    Returns (H, ra_edges, dec_edges) where H is source counts per pixel.
    Optionally Gaussian-smooth H for display.
    """
    if ra_range is None:
        pad = max(0.1, (ra_f.max() - ra_f.min()) * 0.05)
        ra_range = (ra_f.min() - pad, ra_f.max() + pad)
    if dec_range is None:
        pad = max(0.05, (dec_f.max() - dec_f.min()) * 0.05)
        dec_range = (dec_f.min() - pad, dec_f.max() + pad)

    H, ra_e, dec_e = np.histogram2d(
        ra_f, dec_f, bins=n_bins,
        range=[ra_range, dec_range]
    )
    if smooth > 0:
        H = gaussian_filter(H.astype(float), sigma=smooth)
    return H, ra_e, dec_e


def draw_hist2d(ax, H, ra_e, dec_e, cmap="plasma",
                log_scale=True, vmin=None, vmax=None,
                alpha=1.0):
    """
    Display 2D histogram on ax using pcolormesh.
    H is shaped (n_ra, n_dec); axes are ra (x) and dec (y).
    """
    ra_c  = 0.5 * (ra_e[:-1]  + ra_e[1:])
    dec_c = 0.5 * (dec_e[:-1] + dec_e[1:])

    if log_scale:
        H_plot = np.where(H > 0, np.log10(H), np.nan)
        label  = r"$\log_{10}$ (counts per pixel)"
    else:
        H_plot = np.where(H > 0, H, np.nan)
        label  = "Counts per pixel"

    RA_MESH, DEC_MESH = np.meshgrid(ra_c, dec_c, indexing="ij")
    im = ax.pcolormesh(RA_MESH, DEC_MESH, H_plot,
                       cmap=cmap, shading="auto",
                       vmin=vmin, vmax=vmax, alpha=alpha,
                       rasterized=True)
    return im, label


# =============================================================================
# CELL 7 — PLOTTING
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

def style_ax(ax, title="", xl="", yl="", invert_x=True):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, which="both", direction="in",
                   top=True, right=True, labelsize=8)
    ax.set_xlabel(xl, color=TEXT, fontsize=9)
    ax.set_ylabel(yl, color=TEXT, fontsize=9)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9,
                     fontweight="bold", loc="left", pad=5)
    if invert_x:
        ax.invert_xaxis()

def add_colorbar(fig, ax, im, label, fontsize=7.5):
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="3%", pad=0.06)
    cb      = fig.colorbar(im, cax=cax)
    cb.set_label(label, color=MUTED, fontsize=fontsize)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    cb.outline.set_edgecolor(SPINE)
    return cb

# ── Figure layout ─────────────────────────────────────────────────────────────
# Row 0: Full-survey panorama (wide) + statistics table
# Row 1: Per-field zoom panels (6 panels)
# Row 2: dex-spring RA marginal + redshift stripe

fig = plt.figure(figsize=(19, 15))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    3, 6, figure=fig,
    hspace=0.50, wspace=0.38,
    left=0.06, right=0.97,
    top=0.93,  bottom=0.05,
    height_ratios=[2.2, 1.5, 1.0],
)

ax_full  = fig.add_subplot(gs[0, :5])   # full-survey panorama
ax_stats = fig.add_subplot(gs[0, 5])    # statistics text panel

# Per-field zoom (row 1)
field_axes = {}
for i, fname in enumerate(["dex-spring","dex-fall","cosmos",
                            "goods-n","nep","ssa22"]):
    field_axes[fname] = fig.add_subplot(gs[1, i])

# Bottom row: RA profile + N(z) per field
ax_ra_prof = fig.add_subplot(gs[2, :3])  # RA marginal profile
ax_nz      = fig.add_subplot(gs[2, 3:])  # N(z) per field

# ── Panel 1: Full-survey panorama ─────────────────────────────────────────────
style_ax(ax_full,
         title=f"HETDEX SC2 LAE Sky Density  "
               f"(N = {mask.sum():,}  |  S/N > {MIN_SN}  |  "
               f"p_conf & p_cnn ≥ {MIN_P_CONF})",
         xl="RA  (deg)", yl="Dec  (deg)")

# Handle RA wrap for dex-fall (crosses RA=0)
ra_plot = ra_lae.copy()
# Shift dex-fall sources > 300° to negative so histogram is contiguous
fall_mask = (field_lae == "dex-fall") & (ra_plot > 180)
ra_plot[fall_mask] -= 360

ra_lo  = ra_plot.min() - 1
ra_hi  = ra_plot.max() + 1
dec_lo = dec_lae.min() - 0.5
dec_hi = dec_lae.max() + 0.5

H_full, ra_e_full, dec_e_full = field_hist2d(
    ra_plot, dec_lae,
    n_bins=[FULL_BINS_RA, FULL_BINS_DEC],
    ra_range=(ra_lo, ra_hi),
    dec_range=(dec_lo, dec_hi),
    smooth=0.5,
)

# Global log-density colour scale
h_max = np.nanmax(np.log10(np.where(H_full > 0, H_full, np.nan)))
im_full, lbl_full = draw_hist2d(
    ax_full, H_full, ra_e_full, dec_e_full,
    cmap="plasma", log_scale=True,
    vmin=0, vmax=h_max,
)

# Field boundary ellipses + labels
for fname in unique_fields:
    fb   = FIELD_BOXES[fname]
    fc   = FIELD_COLORS[fname]
    ra_c = fb[0] + (fb[1]-fb[0])/2 if fname != "dex-fall" else -20
    dec_c= (fb[2]+fb[3])/2
    ra_w = abs(fb[1]-fb[0]) if fname != "dex-fall" \
           else (fb[1]+360-fb[0])
    dec_w= fb[3]-fb[2]
    ell  = mpatches.Ellipse(
        (ra_c, dec_c),
        width=ra_w * np.cos(np.radians(dec_c)),
        height=dec_w,
        angle=0, edgecolor=fc, facecolor="none",
        lw=1.2, ls="--", alpha=0.80, zorder=5,
    )
    ax_full.add_patch(ell)
    ax_full.text(ra_c, dec_c + dec_w/2 + 0.8,
                 fname.replace("dex-",""),
                 color=fc, fontsize=7.5,
                 ha="center", va="bottom",
                 fontweight="bold", zorder=6)

ax_full.set_xlim(ra_hi, ra_lo)   # RA increases R→L
ax_full.set_ylim(dec_lo, dec_hi)

cb_full = add_colorbar(fig, ax_full, im_full, lbl_full)

# Custom RA tick labels (handle negative values back to > 180)
xticks = ax_full.get_xticks()
ax_full.set_xticklabels(
    [f"{t % 360:.0f}°" for t in xticks],
    color=MUTED, fontsize=7.5
)

# ── Panel 2: Statistics table ──────────────────────────────────────────────────
ax_stats.set_facecolor(AX_BG)
for sp in ax_stats.spines.values():
    sp.set_color(SPINE)
ax_stats.set_title("Field statistics", color=TEXT,
                   fontsize=9, fontweight="bold", loc="left", pad=5)
ax_stats.set_xticks([])
ax_stats.set_yticks([])
ax_stats.tick_params(colors=MUTED)

col_headers = ["Field", "N LAE", "med z", "med logL"]
col_x       = [0.02, 0.28, 0.60, 0.82]
row_y_start = 0.88
row_h       = 0.12

ax_stats.text(col_x[0], row_y_start + 0.04, col_headers[0],
              transform=ax_stats.transAxes, color=TEXT,
              fontsize=7.0, fontweight="bold", va="top")
for j, (hdr, cx) in enumerate(zip(col_headers[1:], col_x[1:])):
    ax_stats.text(cx, row_y_start + 0.04, hdr,
                  transform=ax_stats.transAxes, color=TEXT,
                  fontsize=7.0, fontweight="bold", va="top",
                  ha="right")

ax_stats.axhline(1 - (row_y_start - 0.01 + 0.04) * 0 - (1 - row_y_start + 0.01),
                 color=SPINE, lw=0.8)

for row_i, fname in enumerate(unique_fields):
    fm   = field_lae == fname
    n_f  = fm.sum()
    z_f  = z_lae[fm];    z_f  = z_f[np.isfinite(z_f)]
    lL_f = logL_lae[fm]; lL_f = lL_f[np.isfinite(lL_f)]
    z_med  = float(np.median(z_f))  if len(z_f)  > 0 else np.nan
    lL_med = float(np.median(lL_f)) if len(lL_f) > 0 else np.nan
    fc     = FIELD_COLORS.get(fname, TEXT)
    y_row  = row_y_start - (row_i + 1) * row_h

    ax_stats.text(col_x[0], y_row,
                  fname.replace("-"," "),
                  transform=ax_stats.transAxes,
                  color=fc, fontsize=7.0, va="top",
                  fontweight="bold")
    ax_stats.text(col_x[1], y_row,
                  f"{n_f:,}",
                  transform=ax_stats.transAxes,
                  color=MUTED, fontsize=7.0, va="top", ha="right")
    ax_stats.text(col_x[2], y_row,
                  f"{z_med:.3f}" if np.isfinite(z_med) else "—",
                  transform=ax_stats.transAxes,
                  color=MUTED, fontsize=7.0, va="top", ha="right")
    ax_stats.text(col_x[3], y_row,
                  f"{lL_med:.2f}" if np.isfinite(lL_med) else "—",
                  transform=ax_stats.transAxes,
                  color=MUTED, fontsize=7.0, va="top", ha="right")

# Total row
y_total = row_y_start - (len(unique_fields) + 1.3) * row_h
ax_stats.add_artist(__import__("matplotlib.lines", fromlist=["Line2D"]).Line2D(
    [0, 1], [y_total + 0.03, y_total + 0.03],
    transform=ax_stats.transAxes,
    color=SPINE, lw=0.6, solid_capstyle="butt"))
ax_stats.text(col_x[0], y_total,
              "TOTAL", transform=ax_stats.transAxes,
              color=TEXT, fontsize=7.0, va="top", fontweight="bold")
ax_stats.text(col_x[1], y_total,
              f"{mask.sum():,}",
              transform=ax_stats.transAxes,
              color=TEXT, fontsize=7.0, va="top", ha="right",
              fontweight="bold")

# ── Panel 3: Per-field zoom panels ─────────────────────────────────────────────
z_norm = matplotlib.colors.Normalize(
    vmin=np.nanpercentile(z_lae, 5),
    vmax=np.nanpercentile(z_lae, 95)
)
z_cmap = plt.cm.coolwarm

for fname in ["dex-spring","dex-fall","cosmos","goods-n","nep","ssa22"]:
    ax = field_axes[fname]
    fc = FIELD_COLORS[fname]
    fm = field_lae == fname

    style_ax(ax,
             title=f"{fname}  (N={fm.sum():,})",
             xl="RA (°)", yl="Dec (°)",
             invert_x=True)

    if fm.sum() < 3:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color=MUTED, fontsize=8)
        continue

    ra_f   = ra_lae[fm];  dec_f = dec_lae[fm]
    z_f    = z_lae[fm];   lL_f  = logL_lae[fm]

    # RA range: handle dex-fall wrap
    if fname == "dex-fall":
        ra_f_plot = ra_f.copy()
        ra_f_plot[ra_f_plot > 180] -= 360
    else:
        ra_f_plot = ra_f

    ra_rng  = (ra_f_plot.min() - 0.2, ra_f_plot.max() + 0.2)
    dec_rng = (dec_f.min() - 0.1,     dec_f.max() + 0.1)

    H_f, ra_ef, dec_ef = field_hist2d(
        ra_f_plot, dec_f,
        n_bins=FIELD_BINS,
        ra_range=ra_rng,
        dec_range=dec_rng,
        smooth=0.6,
    )

    im_f, _ = draw_hist2d(ax, H_f, ra_ef, dec_ef,
                           cmap="plasma", log_scale=True,
                           vmin=0)

    # Scatter overlay: colour by redshift for compact fields
    max_scatter = 5000
    if fm.sum() > max_scatter:
        rng_sub = np.random.default_rng(42)
        sub_idx = rng_sub.choice(fm.sum(), size=max_scatter, replace=False)
        ra_sc, dec_sc, z_sc = (ra_f_plot[sub_idx],
                                dec_f[sub_idx], z_f[sub_idx])
    else:
        ra_sc, dec_sc, z_sc = ra_f_plot, dec_f, z_f

    sc = ax.scatter(ra_sc, dec_sc, c=z_sc,
                    cmap=z_cmap, norm=z_norm,
                    s=1.2, alpha=0.35, linewidths=0,
                    rasterized=True, zorder=4)

    ax.set_xlim(ra_rng[1], ra_rng[0])   # inverted RA
    ax.set_ylim(*dec_rng)

    # RA tick labels: handle negative back to > 180
    xticks = ax.get_xticks()
    ax.set_xticklabels(
        [f"{t%360:.1f}" for t in xticks], color=MUTED, fontsize=6.5
    )

    # Small per-field colourbar
    cb_f = add_colorbar(fig, ax, im_f,
                        r"$\log_{10}$ N", fontsize=6.5)

# Shared z colourbar (right of last field panel)
sm_z = plt.cm.ScalarMappable(cmap=z_cmap, norm=z_norm)
sm_z.set_array([])
# Place it at right edge of the nep panel
ax_nep = field_axes["nep"]
divider_nep = make_axes_locatable(ax_nep)
cax_z = fig.add_axes([0.942, 0.38, 0.010, 0.18])
cb_z  = fig.colorbar(sm_z, cax=cax_z)
cb_z.set_label("z", color=MUTED, fontsize=8)
cb_z.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
plt.setp(cb_z.ax.yaxis.get_ticklabels(), color=MUTED)
cb_z.outline.set_edgecolor(SPINE)

# ── Panel 4: RA marginal profile for dex-spring ───────────────────────────────
ax_ra_prof.set_facecolor(AX_BG)
for sp in ax_ra_prof.spines.values():
    sp.set_color(SPINE)
ax_ra_prof.tick_params(colors=MUTED, which="both", direction="in",
                       top=True, right=True, labelsize=8)
ax_ra_prof.set_title(
    "dex-spring  |  N(RA) marginal  coloured by mean redshift per strip",
    color=TEXT, fontsize=9, fontweight="bold", loc="left", pad=5)
ax_ra_prof.set_xlabel("RA  (deg)", color=TEXT, fontsize=9)
ax_ra_prof.set_ylabel("N LAEs per strip", color=TEXT, fontsize=9)

sp_mask = field_lae == "dex-spring"
ra_sp   = ra_lae[sp_mask]
z_sp    = z_lae[sp_mask]

ra_strip_edges = np.arange(ra_sp.min(), ra_sp.max() + 0.5, 0.5)
n_strips = len(ra_strip_edges) - 1
strip_counts = np.zeros(n_strips)
strip_z_med  = np.full(n_strips, np.nan)

for i in range(n_strips):
    in_strip = ((ra_sp >= ra_strip_edges[i]) &
                (ra_sp <  ra_strip_edges[i+1]))
    strip_counts[i] = in_strip.sum()
    if in_strip.sum() > 0:
        zs = z_sp[in_strip]
        strip_z_med[i] = float(np.median(zs[np.isfinite(zs)])
                               if np.isfinite(zs).sum() > 0 else np.nan)

strip_centers = 0.5 * (ra_strip_edges[:-1] + ra_strip_edges[1:])

z_bar_norm = matplotlib.colors.Normalize(
    vmin=np.nanpercentile(strip_z_med, 5),
    vmax=np.nanpercentile(strip_z_med, 95)
)
bar_colors = plt.cm.coolwarm(z_bar_norm(strip_z_med))

for i in range(n_strips):
    ax_ra_prof.bar(
        strip_centers[i], strip_counts[i],
        width=0.48,
        color=bar_colors[i], alpha=0.85,
        linewidth=0,
    )
ax_ra_prof.invert_xaxis()
ax_ra_prof.xaxis.set_minor_locator(AutoMinorLocator())
ax_ra_prof.yaxis.set_minor_locator(AutoMinorLocator())

# Colourbar for the RA strip
sm_ra = plt.cm.ScalarMappable(cmap="coolwarm", norm=z_bar_norm)
sm_ra.set_array([])
cax_ra = make_axes_locatable(ax_ra_prof).append_axes(
    "right", size="2%", pad=0.06)
cb_ra = fig.colorbar(sm_ra, cax=cax_ra)
cb_ra.set_label("Median z", color=MUTED, fontsize=7.5)
cb_ra.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
plt.setp(cb_ra.ax.yaxis.get_ticklabels(), color=MUTED)
cb_ra.outline.set_edgecolor(SPINE)

# ── Panel 5: N(z) per field ───────────────────────────────────────────────────
ax_nz.set_facecolor(AX_BG)
for sp in ax_nz.spines.values():
    sp.set_color(SPINE)
ax_nz.tick_params(colors=MUTED, which="both", direction="in",
                  top=True, right=True, labelsize=8)
ax_nz.set_title("Redshift distribution per field",
                color=TEXT, fontsize=9, fontweight="bold",
                loc="left", pad=5)
ax_nz.set_xlabel("Spectroscopic redshift  z", color=TEXT, fontsize=9)
ax_nz.set_ylabel("Normalised count", color=TEXT, fontsize=9)

z_bins = np.linspace(1.85, 3.55, 50)
bottom = np.zeros(len(z_bins) - 1)

for fname in unique_fields:
    fm   = field_lae == fname
    z_f  = z_lae[fm]
    z_f  = z_f[np.isfinite(z_f)]
    if len(z_f) < 5:
        continue
    h, _ = np.histogram(z_f, bins=z_bins)
    h_norm = h / max(h.max(), 1)
    ax_nz.step(z_bins[:-1], h_norm,
               where="post", color=FIELD_COLORS[fname],
               lw=1.4, alpha=0.85,
               label=f"{fname}  (N={fm.sum():,})")

ax_nz.set_xlim(z_bins[0], z_bins[-1])
ax_nz.set_ylim(0, 1.12)
ax_nz.xaxis.set_minor_locator(AutoMinorLocator())
ax_nz.yaxis.set_minor_locator(AutoMinorLocator())
ax_nz.legend(fontsize=7.5, facecolor="#21262d",
             edgecolor=SPINE, labelcolor=TEXT,
             loc="upper right", ncol=2)

# ── Super-title ───────────────────────────────────────────────────────────────
syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
fig.suptitle(
    "HETDEX SC2 — LAE Sky Density Map" + syn_tag,
    color=TEXT, fontsize=13, fontweight="bold", y=0.975,
)

if SAVE_PATH:
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 8 — SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("  HETDEX SC2 LAE Sky Density — Summary")
print("=" * 60)
print(f"  Total LAEs (all fields)  : {mask.sum():,}")
print(f"  S/N > {MIN_SN}  |  "
      f"p_conf ≥ {MIN_P_CONF}  |  p_cnn ≥ {MIN_P_CNN}")
print(f"\n  {'Field':<14}  {'N LAE':>8}  "
      f"{'frac':>6}  {'med z':>7}  {'med logL':>9}")
print("  " + "-"*50)
for fname in unique_fields:
    fm   = field_lae == fname
    n_f  = fm.sum()
    frac = n_f / mask.sum()
    z_f  = z_lae[fm];    z_f  = z_f[np.isfinite(z_f)]
    lL_f = logL_lae[fm]; lL_f = lL_f[np.isfinite(lL_f)]
    z_med  = float(np.median(z_f))  if len(z_f)  else np.nan
    lL_med = float(np.median(lL_f)) if len(lL_f) else np.nan
    print(f"  {fname:<14}  {n_f:>8,}  {frac:>6.3f}  "
          f"{z_med:>7.3f}  {lL_med:>9.3f}")
print("=" * 60)
