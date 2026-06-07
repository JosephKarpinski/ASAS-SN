"""
hetdex_wave_groups.py
=====================
Load the HETDEX SC2 Detection Information Table, identify the largest
3D Friends-of-Friends (FOF) emission-line groups via wave_group_id, and
produce a multi-panel visualization of their sky positions and group extent.

wave_group columns in detinfo
------------------------------
  wave_group_id    int   FOF group identifier (-999 = ungrouped singleton)
  wave_group_a     arcsec  semi-major axis of the group ellipse on sky
  wave_group_b     arcsec  semi-minor axis
  wave_group_pa    deg     position angle (E of N)
  wave_group_ra    deg     mean RA of group members (ICRS)
  wave_group_dec   deg     mean Dec of group members (ICRS)
  wave_group_wave  AA      mean wavelength of group members

Physics
-------
HETDEX runs a 3D FOF linking algorithm on (RA, Dec, wavelength) space,
connecting detections that lie within a linking length in all three
dimensions simultaneously.  Large groups (many members, large wave_group_a)
are candidates for:
  - Extended emission systems / proto-clusters
  - Lya blobs with multiple IFU detections
  - AGN with spatially extended narrow-line regions
  - Foreground OII overdensities at low z

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

DETINFO_PATH = "hetdex_sc2_detinfo_v1.5.fits"   # update to local path

# How many top groups to highlight in the sky maps
N_TOP = 15

# Minimum group member count to be considered a "real" group
MIN_MEMBERS = 3

# Quality filter on individual detections within groups
MIN_SN = 4.5          # S/N threshold
BAD    = -999.0       # sentinel for missing float values

# Save figure?
SAVE_PATH = "hetdex_wave_groups.png"   # None = display inline only

# =============================================================================
# CELL 2 — IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.ticker  import AutoMinorLocator
from matplotlib.colors  import LogNorm, Normalize
import matplotlib.cm as cm
from scipy.stats import gaussian_kde

from astropy.io    import fits
from astropy.table import Table, vstack
from astropy.cosmology import Planck18
import astropy.units as u
from astropy.coordinates import SkyCoord

try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
        matplotlib.rcParams["figure.dpi"] = 120
except Exception:
    pass

cosmo  = Planck18
LYA_AA = 1215.67   # Lya rest wavelength

print("Imports OK.")

# =============================================================================
# CELL 3 — SYNTHETIC DATA GENERATOR
# =============================================================================

def make_synthetic_detinfo(n_rows=80_000, seed=99):
    """
    Generates a synthetic detinfo table that mimics hetdex_sc2_detinfo_v1.5.fits.
    Includes:
      - Realistic (RA, Dec) sky distribution across HETDEX fields
      - Clustered groups with wave_group_id including a handful of large groups
      - Ungrouped singletons (wave_group_id = -999)
      - Wavelength range 3500–5500 AA
    """
    rng = np.random.default_rng(seed)

    # ── Sky positions: concentrate in dex-spring and dex-fall fields ──────────
    field_params = {
        "dex-spring": (185.0, 51.5, 25.0, 3.0, 0.50),  # ra_cen, dec_cen, ra_std, dec_std, frac
        "dex-fall"  : (355.0, 0.5,  20.0, 3.0, 0.33),
        "cosmos"    : (150.1, 2.2,   0.6, 0.5, 0.10),
        "goods-n"   : (189.3, 62.2,  0.5, 0.4, 0.07),
    }
    ra_all   = np.zeros(n_rows)
    dec_all  = np.zeros(n_rows)
    fld_all  = np.empty(n_rows, dtype="U12")
    idx = 0
    for fname, (ra_c, dec_c, ra_s, dec_s, frac) in field_params.items():
        n_f = int(n_rows * frac) if fname != "goods-n" else n_rows - idx
        ra_all[idx:idx+n_f]  = rng.normal(ra_c, ra_s, n_f) % 360
        dec_all[idx:idx+n_f] = rng.normal(dec_c, dec_s, n_f)
        fld_all[idx:idx+n_f] = fname
        idx += n_f

    wave_all   = rng.uniform(3500.0, 5500.0, n_rows).astype(np.float32)
    sn_all     = np.abs(rng.lognormal(1.5, 0.6, n_rows)).astype(np.float32)
    flux_all   = (sn_all * rng.lognormal(0.3, 0.5, n_rows)).astype(np.float32)
    z_all      = (wave_all / LYA_AA - 1.0).astype(np.float32)  # assume Lya

    # Source types
    stypes = rng.choice(["lae","oii","agn","star","lzg","none"], n_rows,
                        p=[0.38,0.28,0.08,0.05,0.10,0.11])

    # ── Build wave_group_id ────────────────────────────────────────────────────
    # Create ~500 groups of various sizes, plus singletons
    wave_group_id  = np.full(n_rows, -999, dtype=np.int64)
    wave_group_a   = np.full(n_rows, BAD, dtype=np.float32)
    wave_group_b   = np.full(n_rows, BAD, dtype=np.float32)
    wave_group_pa  = np.full(n_rows, BAD, dtype=np.float32)
    wave_group_ra  = np.full(n_rows, BAD, dtype=np.float32)
    wave_group_dec = np.full(n_rows, BAD, dtype=np.float32)
    wave_group_wave= np.full(n_rows, BAD, dtype=np.float32)

    # Size distribution: power-law n_members ~ 3..200
    n_groups   = 550
    group_sizes= np.clip(
        (3 * rng.pareto(2.0, n_groups) + 3).astype(int), 3, 220
    )
    # A handful of monster groups
    group_sizes[:8] = rng.integers(80, 220, 8)
    np.random.shuffle(group_sizes)

    row_ptr = 0
    for gid, gsize in enumerate(group_sizes):
        if row_ptr + gsize > n_rows:
            break
        # Group centre on sky
        g_ra  = float(rng.choice(ra_all[::100]))
        g_dec = float(rng.choice(dec_all[::100]))
        g_wave= float(rng.uniform(3600, 5400))
        # Semi-axes: correlated with group size
        g_a   = float(rng.lognormal(np.log(gsize * 0.08 + 1.0), 0.4))
        g_b   = g_a * rng.uniform(0.3, 1.0)
        g_pa  = float(rng.uniform(0, 180))

        wave_group_id[row_ptr:row_ptr+gsize]  = gid
        wave_group_a[row_ptr:row_ptr+gsize]   = g_a
        wave_group_b[row_ptr:row_ptr+gsize]   = g_b
        wave_group_pa[row_ptr:row_ptr+gsize]  = g_pa
        wave_group_ra[row_ptr:row_ptr+gsize]  = g_ra
        wave_group_dec[row_ptr:row_ptr+gsize] = g_dec
        wave_group_wave[row_ptr:row_ptr+gsize]= g_wave

        # Member positions: scatter around group centre
        ra_all[row_ptr:row_ptr+gsize]  = g_ra  + rng.normal(0, g_a/3600, gsize)
        dec_all[row_ptr:row_ptr+gsize] = g_dec + rng.normal(0, g_b/3600, gsize)
        wave_all[row_ptr:row_ptr+gsize]= g_wave + rng.normal(0, 5.0, gsize)
        row_ptr += gsize

    tab = Table({
        "source_id"       : np.arange(n_rows, dtype=np.int64),
        "RA"              : ra_all.astype(np.float32),
        "DEC"             : dec_all.astype(np.float32),
        "RA_det"          : ra_all.astype(np.float32),
        "DEC_det"         : dec_all.astype(np.float32),
        "z_hetdex"        : z_all,
        "source_type"     : stypes,
        "detectid"        : np.arange(2_100_000_000,
                                       2_100_000_000 + n_rows, dtype=np.int64),
        "wave"            : wave_all,
        "flux"            : flux_all,
        "sn"              : sn_all,
        "field"           : fld_all,
        "line_id"         : np.where(z_all > 1.87, "lya", "oii").astype("U8"),
        "p_conf"          : np.clip(
                                rng.beta(3, 1.5, n_rows), 0, 1
                            ).astype(np.float32),
        "p_cnn"           : np.clip(
                                rng.beta(2.8, 1.5, n_rows), 0, 1
                            ).astype(np.float32),
        "wave_group_id"   : wave_group_id,
        "wave_group_a"    : wave_group_a,
        "wave_group_b"    : wave_group_b,
        "wave_group_pa"   : wave_group_pa,
        "wave_group_ra"   : wave_group_ra,
        "wave_group_dec"  : wave_group_dec,
        "wave_group_wave" : wave_group_wave,
    })
    print(f"  Synthetic detinfo: {len(tab):,} rows")
    return tab


# =============================================================================
# CELL 4 — LOAD DETINFO TABLE
# =============================================================================

def load_detinfo(path):
    try:
        hdul = fits.open(path, memmap=True)
        tab  = Table(hdul[1].data)
        hdul.close()
        # Normalise all column names to lower-case
        tab.rename_columns(tab.colnames,
                           [c.lower() for c in tab.colnames])
        synthetic = False
        print(f"Loaded {path}: {len(tab):,} rows, "
              f"{len(tab.colnames)} columns")
    except FileNotFoundError:
        print(f"'{path}' not found — using synthetic demo data.")
        tab       = make_synthetic_detinfo()
        synthetic = True
    return tab, synthetic


det, SYNTHETIC = load_detinfo(DETINFO_PATH)

# Column-name normalisation: handle both 'RA'/'DEC' and 'ra'/'dec'
def getcol(tab, *candidates):
    """Return first matching column name (case-insensitive candidates)."""
    lc = {c.lower(): c for c in tab.colnames}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    raise KeyError(f"None of {candidates} found. "
                   f"Available: {list(tab.colnames)[:25]}")

RA_COL   = getcol(det, "RA_det", "RA", "ra_det", "ra")
DEC_COL  = getcol(det, "DEC_det", "DEC", "dec_det", "dec")
WGI_COL  = getcol(det, "wave_group_id")
WGA_COL  = getcol(det, "wave_group_a")
WGB_COL  = getcol(det, "wave_group_b")
WGPA_COL = getcol(det, "wave_group_pa")
WGRA_COL = getcol(det, "wave_group_ra")
WGDC_COL = getcol(det, "wave_group_dec")
WGW_COL  = getcol(det, "wave_group_wave")
SN_COL   = getcol(det, "sn")
WAVE_COL = getcol(det, "wave")

print(f"Key columns resolved: RA={RA_COL}, DEC={DEC_COL}, "
      f"wave_group_id={WGI_COL}, sn={SN_COL}")

# =============================================================================
# CELL 5 — EXTRACT GROUPED DETECTIONS
# =============================================================================

ra_arr    = np.array(det[RA_COL],   dtype=float)
dec_arr   = np.array(det[DEC_COL],  dtype=float)
wgid_arr  = np.array(det[WGI_COL],  dtype=np.int64)
wga_arr   = np.array(det[WGA_COL],  dtype=float)
wgb_arr   = np.array(det[WGB_COL],  dtype=float)
wgpa_arr  = np.array(det[WGPA_COL], dtype=float)
wgra_arr  = np.array(det[WGRA_COL], dtype=float)
wgdec_arr = np.array(det[WGDC_COL], dtype=float)
wgw_arr   = np.array(det[WGW_COL],  dtype=float)
sn_arr    = np.array(det[SN_COL],   dtype=float)
wave_arr  = np.array(det[WAVE_COL], dtype=float)

# Replace bad-value sentinels
for arr in [wga_arr, wgb_arr, wgpa_arr, wgra_arr, wgdec_arr, wgw_arr]:
    arr[arr == BAD] = np.nan

# Source type and field
stype_col = getcol(det, "source_type")
field_col = getcol(det, "field")
stype_arr = np.array(det[stype_col], dtype=str)
field_arr = np.array(det[field_col], dtype=str)
# Line ID if present
try:
    lid_col = getcol(det, "line_id")
    lid_arr = np.array(det[lid_col], dtype=str)
except KeyError:
    lid_arr = np.full(len(det), "unknown", dtype="U10")

# ── Select grouped detections ────────────────────────────────────────────────
# Exclude sentinel values: -999 (bad), -1 (catch-all group with NaN coords),
# and any gid <= 0 to be safe.
grouped_mask = (wgid_arr > 0) & (sn_arr > MIN_SN) & (sn_arr != BAD)
grouped      = det[grouped_mask]
g_ids        = wgid_arr[grouped_mask]
g_ra         = ra_arr[grouped_mask]
g_dec        = dec_arr[grouped_mask]
g_wga        = wga_arr[grouped_mask]
g_wgra       = wgra_arr[grouped_mask]
g_wgdec      = wgdec_arr[grouped_mask]
g_wgw        = wgw_arr[grouped_mask]
g_sn         = sn_arr[grouped_mask]
g_wave       = wave_arr[grouped_mask]
g_stype      = stype_arr[grouped_mask]
g_field      = field_arr[grouped_mask]
g_lid        = lid_arr[grouped_mask]

print(f"\nTotal detections       : {len(det):,}")
print(f"Grouped (wgid != -999) : {grouped_mask.sum():,}  "
      f"({100*grouped_mask.mean():.1f}%)")

# =============================================================================
# CELL 6 — BUILD GROUP SUMMARY TABLE
# =============================================================================

unique_gids, g_counts = np.unique(g_ids, return_counts=True)

# For each group, collect the ellipse parameters (constant within group)
# and compute aggregate statistics
group_records = []
for gid, n_mem in zip(unique_gids, g_counts):
    if n_mem < MIN_MEMBERS:
        continue
    sel   = g_ids == gid
    # Ellipse params (take first non-nan row)
    wga_v = g_wga[sel][np.isfinite(g_wga[sel])]
    a     = float(wga_v[0]) if len(wga_v) else np.nan

    # Group centre
    cra  = float(np.nanmedian(g_wgra[sel]))
    cdec = float(np.nanmedian(g_wgdec[sel]))
    cw   = float(np.nanmedian(g_wgw[sel]))

    # Redshift assuming Lya
    z_lya = cw / LYA_AA - 1.0

    # Angular spread (arcs) from member RA/Dec scatter
    dra  = (g_ra[sel]  - cra) * np.cos(np.radians(cdec)) * 3600  # arcsec
    ddec = (g_dec[sel] - cdec) * 3600
    ang_spread = float(np.sqrt(dra.var() + ddec.var()))

    # Dominant source type in group
    types, tcounts = np.unique(g_stype[sel], return_counts=True)
    dom_type = types[tcounts.argmax()]

    # Median S/N
    med_sn = float(np.median(g_sn[sel]))

    # Skip groups whose centre coordinates are non-finite
    if not (np.isfinite(cra) and np.isfinite(cdec) and np.isfinite(cw)):
        continue

    group_records.append({
        "gid"        : int(gid),
        "n_mem"      : int(n_mem),
        "cra"        : cra,
        "cdec"       : cdec,
        "cwave"      : cw,
        "z_lya"      : z_lya,
        "semi_a"     : a,
        "ang_spread" : ang_spread,
        "dom_type"   : dom_type,
        "med_sn"     : med_sn,
        "field"      : g_field[sel][0].strip(),
    })

# Sort by n_members descending
group_records.sort(key=lambda r: r["n_mem"], reverse=True)

print(f"\nGroups with >= {MIN_MEMBERS} members: {len(group_records):,}")
print(f"\nTop {min(N_TOP, len(group_records))} groups by member count:")
print(f"  {'Rank':>4}  {'GID':>10}  {'N_mem':>6}  "
      f"{'RA':>9}  {'Dec':>8}  {'wave':>7}  "
      f"{'semi_a':>7}  {'z_lya':>6}  {'type':>6}  field")
print("  " + "-"*90)
for rank, rec in enumerate(group_records[:N_TOP], 1):
    print(f"  {rank:4d}  {rec['gid']:10d}  {rec['n_mem']:6d}  "
          f"{rec['cra']:9.4f}  {rec['cdec']:8.4f}  "
          f"{rec['cwave']:7.2f}  "
          f"{rec['semi_a']:7.2f}  {rec['z_lya']:6.3f}  "
          f"{rec['dom_type']:>6s}  {rec['field']}")

top_gids = [r["gid"] for r in group_records[:N_TOP]]

# =============================================================================
# CELL 7 — DERIVED PHYSICS
# =============================================================================

# For groups where z_lya > 1.87 (Lya in VIRUS band), compute physical sizes
LYA_ZMIN = 3500 / LYA_AA - 1  # ~1.879
for rec in group_records:
    z = rec["z_lya"]
    if 1.87 < z < 3.6 and np.isfinite(rec["semi_a"]):
        # Proper kpc per arcsec at this redshift
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z).to(
            u.kpc / u.arcsec).value
        rec["semi_a_kpc"] = rec["semi_a"] * kpc_per_arcsec
    else:
        rec["semi_a_kpc"] = np.nan

# =============================================================================
# CELL 8 — PLOTTING
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

# Colour maps for source type
TYPE_COL = {
    "lae" : "#58a6ff", "oii": "#3fb950", "agn": "#f78166",
    "star": "#d2a8ff", "lzg": "#ffa657", "none": "#8b949e",
    "cont": "#79c0ff",
}
# Top-group colour ramp for sky maps
TOP_CMAP  = plt.cm.plasma
top_cols  = {gid: TOP_CMAP(i / max(N_TOP - 1, 1))
             for i, gid in enumerate(top_gids)}

def style_ax(ax, title, xl, yl, minor=True):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, which="both", direction="in",
                   top=True, right=True, labelsize=9)
    ax.set_xlabel(xl, color=TEXT, fontsize=10)
    ax.set_ylabel(yl, color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=11,
                 fontweight="bold", loc="left", pad=6)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

def mleg(ax, **kw):
    return ax.legend(fontsize=8, facecolor="#21262d",
                     edgecolor=SPINE, labelcolor=TEXT, **kw)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 15))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    3, 3, figure=fig,
    hspace=0.40, wspace=0.30,
    left=0.07, right=0.97,
    top=0.92,  bottom=0.06,
)

ax_sky   = fig.add_subplot(gs[0, :2])  # wide: full survey sky map
ax_rank  = fig.add_subplot(gs[0, 2])   # group size rank plot
ax_zoom  = fig.add_subplot(gs[1, :2])  # zoom into the largest group
ax_wave  = fig.add_subplot(gs[1, 2])   # wavelength distribution of groups
ax_nsize = fig.add_subplot(gs[2, 0])   # N_members histogram
ax_semia = fig.add_subplot(gs[2, 1])   # semi-major axis distribution
ax_zwave = fig.add_subplot(gs[2, 2])   # group centre wave vs z_lya scatter

# ── Panel 1: Full sky map coloured by group membership ────────────────────────
style_ax(ax_sky,
         f"HETDEX SC2 — wave_group_id sky map  (top {N_TOP} groups highlighted)",
         "RA  (deg)", "Dec  (deg)")

# Background: all ungrouped detections (faint grey)
ung_mask = (wgid_arr == -999) & (sn_arr > MIN_SN)
ax_sky.scatter(ra_arr[ung_mask], dec_arr[ung_mask],
               s=0.3, c="#2d333b", alpha=0.15,
               linewidths=0, rasterized=True, label="Ungrouped")

# Grouped non-top detections (dim teal)
grp_mask = grouped_mask & ~np.isin(wgid_arr, top_gids)
ax_sky.scatter(ra_arr[grp_mask], dec_arr[grp_mask],
               s=0.8, c="#1f4e79", alpha=0.25,
               linewidths=0, rasterized=True, label="Grouped (other)")

# Top groups: each in its own colour with ellipse overlay
for rank, rec in enumerate(group_records[:N_TOP]):
    gid   = rec["gid"]
    color = top_cols[gid]
    sel   = g_ids == gid
    ax_sky.scatter(g_ra[sel], g_dec[sel],
                   s=6, c=[color], alpha=0.75,
                   linewidths=0, zorder=5)

    # Draw the FOF ellipse — semi-axes in arcsec, convert to degrees
    if np.isfinite(rec["semi_a"]):
        ell = Ellipse(
            xy=(rec["cra"], rec["cdec"]),
            width  = 2 * rec["semi_a"] / 3600,   # arcsec -> deg
            height = 2 * rec.get("semi_a", rec["semi_a"]) / 3600,
            angle  = 0,  # simplified — real PA from wave_group_pa
            edgecolor=color, facecolor="none",
            lw=1.2, alpha=0.70, zorder=6,
        )
        ax_sky.add_patch(ell)
    ax_sky.text(rec["cra"], rec["cdec"] + rec["semi_a"]/3600 * 1.4,
                f"#{rank+1}", color=color, fontsize=6.5,
                ha="center", va="bottom", zorder=7)

ax_sky.set_xlim(ra_arr.min() - 1, ra_arr.max() + 1)
ax_sky.set_ylim(dec_arr.min() - 0.5, dec_arr.max() + 0.5)
ax_sky.invert_xaxis()   # RA increases right to left

# Custom legend
legend_elements = [
    mpatches.Patch(color="#2d333b", alpha=0.7, label="Ungrouped"),
    mpatches.Patch(color="#1f4e79", alpha=0.7, label="Grouped (not top)"),
    mpatches.Patch(color=TOP_CMAP(0.1), label=f"Top {N_TOP} groups"),
]
mleg(ax_sky, handles=legend_elements, loc="upper right")

# ── Panel 2: Group size rank (Zipf-like) ──────────────────────────────────────
style_ax(ax_rank, "Group Size Distribution",
         "Rank", "N members per group")

all_sizes = sorted([r["n_mem"] for r in group_records], reverse=True)
ranks     = np.arange(1, len(all_sizes) + 1)
ax_rank.scatter(ranks, all_sizes,
                s=8, c="#58a6ff", alpha=0.6, linewidths=0)
# Highlight top N
ax_rank.scatter(ranks[:N_TOP], all_sizes[:N_TOP],
                s=20,
                c=[TOP_CMAP(i/(N_TOP-1)) for i in range(N_TOP)],
                zorder=5, linewidths=0.3, edgecolors=TEXT)
ax_rank.set_yscale("log")
ax_rank.set_xscale("log")
ax_rank.axhline(MIN_MEMBERS, color=MUTED, lw=0.8, ls="--", alpha=0.6,
                label=f"Min members = {MIN_MEMBERS}")
mleg(ax_rank)

# ── Panel 3: Zoom into the single largest group ───────────────────────────────
top1 = group_records[0]
g1_sel = g_ids == top1["gid"]
_semi_a_top = top1["semi_a"] if np.isfinite(top1["semi_a"]) else 10.0
pad_deg = max(0.01, _semi_a_top / 3600 * 3.5)

style_ax(ax_zoom,
         (f"Largest group  #1  |  "
          f"N={top1['n_mem']}  |  "
          f"GID={top1['gid']}  |  "
          f"wave={top1['cwave']:.1f} Å  |  "
          f"z_lya={top1['z_lya']:.3f}  |  "
          f"{top1['field']}"),
         "RA  (deg)", "Dec  (deg)")

# Background grouped detections in this region
roi = (ra_arr  > top1["cra"]  - pad_deg) & (ra_arr  < top1["cra"]  + pad_deg) & \
      (dec_arr > top1["cdec"] - pad_deg) & (dec_arr < top1["cdec"] + pad_deg)
bg_roi = roi & (wgid_arr != top1["gid"])
ax_zoom.scatter(ra_arr[bg_roi], dec_arr[bg_roi],
                s=4, c=MUTED, alpha=0.20, linewidths=0)

# Members of this group, coloured by source type
for stype_val, sc in TYPE_COL.items():
    tm = g1_sel & (g_stype == stype_val)
    if tm.sum() == 0:
        continue
    ax_zoom.scatter(g_ra[tm], g_dec[tm],
                    s=25, c=sc, alpha=0.80, linewidths=0,
                    label=f"{stype_val}  (N={tm.sum()})", zorder=5)

# FOF ellipse
if np.isfinite(top1["semi_a"]):
    ell1 = Ellipse(
        xy=(top1["cra"], top1["cdec"]),
        width  = 2 * top1["semi_a"] / 3600,
        height = 2 * top1["semi_a"] / 3600,
        angle  = 0,
        edgecolor="#ffa657", facecolor="none",
        lw=1.5, alpha=0.85, ls="--", zorder=6,
    )
    ax_zoom.add_patch(ell1)
    ax_zoom.scatter(top1["cra"], top1["cdec"],
                    marker="+", s=80, c="#ffa657",
                    lw=1.5, zorder=7)

if np.isfinite(top1["cra"]) and np.isfinite(top1["cdec"]) and pad_deg > 0:
    ax_zoom.set_xlim(top1["cra"] - pad_deg, top1["cra"] + pad_deg)
    ax_zoom.set_ylim(top1["cdec"] - pad_deg, top1["cdec"] + pad_deg)
ax_zoom.invert_xaxis()

kpc_str = (f"  ~{top1['semi_a_kpc']:.0f} kpc"
           if np.isfinite(top1.get("semi_a_kpc", np.nan)) else "")
_arcsec_label = 'semi-major axis = {:.2f}"{}'.format(top1["semi_a"], kpc_str)
ax_zoom.text(0.02, 0.97,
             _arcsec_label,
             transform=ax_zoom.transAxes,
             color="#ffa657", fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor=AX_BG, edgecolor=SPINE, alpha=0.8))
mleg(ax_zoom, loc="lower right", ncol=2)

# ── Panel 4: Wavelength distribution of top-group centres ────────────────────
style_ax(ax_wave,
         f"Centre wavelength of top {N_TOP} groups",
         r"$\lambda_{\rm group}$  (Å)", "Group size rank")

for rank, rec in enumerate(group_records[:N_TOP], 1):
    color = top_cols[rec["gid"]]
    z_lya = rec["cwave"] / LYA_AA - 1.0
    ax_wave.scatter(rec["cwave"], rank, s=rec["n_mem"] ** 0.6 * 3,
                    c=[color], zorder=5, linewidths=0)
    ax_wave.text(rec["cwave"] + 8, rank, f"N={rec['n_mem']}",
                 color=color, fontsize=6.5, va="center")

# Lya window shading
ax_wave.axvspan(3500, 5500, alpha=0.05, color="#58a6ff",
                label="VIRUS bandpass")
ax_wave.set_xlim(3400, 5600)
ax_wave.set_ylim(N_TOP + 0.5, 0.5)   # rank 1 at top
ax_wave.set_yticks(range(1, N_TOP + 1))

# Add secondary z_lya axis
ax_wave2 = ax_wave.twiny()
ax_wave2.set_xlim(ax_wave.get_xlim())
z_ticks   = [2.0, 2.5, 3.0, 3.5]
wave_ticks= [LYA_AA * (1 + z) for z in z_ticks]
ax_wave2.set_xticks(wave_ticks)
ax_wave2.set_xticklabels([f"z={z:.1f}" for z in z_ticks],
                          fontsize=7, color=MUTED)
ax_wave2.tick_params(colors=MUTED)
ax_wave2.spines["top"].set_color(SPINE)

mleg(ax_wave, loc="lower right")

# ── Panel 5: N_members histogram ──────────────────────────────────────────────
style_ax(ax_nsize, "Group Member Count Distribution",
         "N members", "Number of groups")

all_n = [r["n_mem"] for r in group_records]
bins  = np.logspace(np.log10(MIN_MEMBERS), np.log10(max(all_n) + 1), 25)
ax_nsize.hist(all_n, bins=bins, color="#58a6ff", alpha=0.70,
              histtype="stepfilled", edgecolor="#1f6feb", lw=0.8)
ax_nsize.set_xscale("log")
ax_nsize.set_yscale("log")
ax_nsize.axvline(group_records[0]["n_mem"], color="#ffa657",
                 lw=1.2, ls="--", alpha=0.80,
                 label=f"Largest: N={group_records[0]['n_mem']}")
mleg(ax_nsize)

# ── Panel 6: Semi-major axis distribution ────────────────────────────────────
style_ax(ax_semia, "FOF Ellipse Semi-major Axis",
         "wave_group_a  (arcsec)", "Number of groups")

semia_vals = [r["semi_a"] for r in group_records
              if np.isfinite(r["semi_a"])]
if semia_vals:
    bins_a = np.logspace(np.log10(min(semia_vals) * 0.9),
                         np.log10(max(semia_vals) * 1.1), 30)
    ax_semia.hist(semia_vals, bins=bins_a,
                  color="#3fb950", alpha=0.70,
                  histtype="stepfilled", edgecolor="#238636", lw=0.8)
    ax_semia.set_xscale("log")
    ax_semia.set_yscale("log")

    # Mark largest group
    ax_semia.axvline(group_records[0]["semi_a"],
                     color="#ffa657", lw=1.2, ls="--", alpha=0.80,
                     label='#1:  {:.1f}"'.format(group_records[0]["semi_a"]))
    mleg(ax_semia)

# ── Panel 7: Group centre wavelength vs redshift  ────────────────────────────
style_ax(ax_zwave,
         r"Group $\lambda$ vs $z_{\rm Ly\alpha}$",
         r"$z_{\rm Ly\alpha}$ (if Lya)",
         r"$\lambda_{\rm group}$  (Å)")

# Scatter all groups coloured by N_members
n_arr  = np.array([r["n_mem"]  for r in group_records], dtype=float)
z_arr2 = np.array([r["cwave"] / LYA_AA - 1.0 for r in group_records])
w_arr2 = np.array([r["cwave"] for r in group_records])

sc = ax_zwave.scatter(z_arr2, w_arr2,
                       c=np.log10(n_arr), cmap="plasma",
                       s=8, alpha=0.60, linewidths=0,
                       vmin=np.log10(MIN_MEMBERS),
                       vmax=np.log10(n_arr.max()))
cb = fig.colorbar(sc, ax=ax_zwave, pad=0.02)
cb.set_label(r"$\log_{10}\ N_{\rm members}$",
             color=MUTED, fontsize=9)
cb.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
cb.outline.set_edgecolor(SPINE)

# Lya line: wave = LYA_AA * (1+z)
z_line = np.linspace(1.8, 3.6, 100)
ax_zwave.plot(z_line, LYA_AA * (1 + z_line),
              color="#58a6ff", lw=1.5, ls="--", alpha=0.65,
              label=r"Ly$\alpha$")
# OII line: wave = 3727 * (1+z)  -> z_lya axis is wrong for OII groups
ax_zwave.plot(z_line, 3727.0 * (1 + z_line) / (1 + z_line)
              * (1 + z_line) / (1 + z_line),  # identity, just for clarity
              color="#3fb950", lw=0, alpha=0)  # placeholder

# Highlight top groups
for rank, rec in enumerate(group_records[:N_TOP], 1):
    z_r = rec["cwave"] / LYA_AA - 1.0
    ax_zwave.scatter(z_r, rec["cwave"],
                     s=25, c=[top_cols[rec["gid"]]],
                     zorder=6, linewidths=0)

ax_zwave.set_xlim(1.7, 3.7)
mleg(ax_zwave, loc="upper left")

# ── Super-title & footer ──────────────────────────────────────────────────────
syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
fig.suptitle(
    rf"HETDEX SC2 — 3D FOF Wave-Group Analysis{syn_tag}",
    color=TEXT, fontsize=14, fontweight="bold", y=0.975,
)
fig.text(
    0.5, 0.005,
    (f"detinfo rows: {len(det):,}   |   "
     f"grouped (S/N>{MIN_SN}): {grouped_mask.sum():,}   |   "
     f"groups (N≥{MIN_MEMBERS}): {len(group_records):,}   |   "
     f"top {N_TOP} highlighted"),
    ha="center", fontsize=8.5, color=MUTED,
)

if SAVE_PATH:
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 9 — SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 95)
print(f"  HETDEX SC2 wave_group Summary  —  Top {min(N_TOP, len(group_records))} groups")
print("=" * 95)
_hdr_cols = ("Rk", "GID", "N", "RA", "Dec",
             "wave(A)", "z_lya", 'semi_a"', "semi_a_kpc", "type", "field")
_hdr_fmt  = "  {:>2}  {:>10}  {:>5}  {:>9}  {:>8}  {:>8}  {:>6}  {:>7}  {:>10}  {:>6}  {}"
print(_hdr_fmt.format(*_hdr_cols))
print("  " + "-"*93)
_row_fmt = "  {:2d}  {:10d}  {:5d}  {:9.4f}  {:8.4f}  {:8.2f}  {:6.3f}  {:7.2f}  {:>10}  {:>6s}  {}"
for rank, rec in enumerate(group_records[:N_TOP], 1):
    _kpc_val = rec.get("semi_a_kpc", np.nan)
    _kpc_str = "{:.1f}".format(_kpc_val) if np.isfinite(_kpc_val) else "n/a"
    print(_row_fmt.format(
        rank, rec["gid"], rec["n_mem"],
        rec["cra"], rec["cdec"],
        rec["cwave"], rec["z_lya"],
        rec["semi_a"], _kpc_str,
        rec["dom_type"], rec["field"]
    ))
print("=" * 95)

print(f"\n  All-group statistics:")
print(f"    Total groups (N>={MIN_MEMBERS})  : {len(group_records):,}")
all_n2 = [r["n_mem"] for r in group_records]
print(f"    Median N_members        : {np.median(all_n2):.1f}")
print(f"    Mean   N_members        : {np.mean(all_n2):.1f}")
print(f"    Largest group           : {max(all_n2):,} members")
sa_fin = [r["semi_a"] for r in group_records if np.isfinite(r["semi_a"])]
if sa_fin:
    print(f"    Median semi_a           : {np.median(sa_fin):.2f} arcsec")
    print(f"    Max    semi_a           : {max(sa_fin):.2f} arcsec")
