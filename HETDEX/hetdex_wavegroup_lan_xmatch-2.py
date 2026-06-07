"""
hetdex_wavegroup_lan_xmatch.py
==============================
Cross-match HETDEX SC2 wave_group emission-line groups against the
HETDEX Lyman-Alpha Nebulae (LAN) catalog.

Strategy
--------
1. Load and deduplicate the LAN catalog (dups_detectid).
2. Load the SC2 detinfo table; build one row per group (group_records)
   using the same logic as hetdex_wave_groups.py.
3. Filter groups to LAE-redshift range (z_lya > 1.87, Lya in VIRUS band)
   and minimum angular size (wave_group_a > MIN_GROUP_A_ARCSEC).
4. Cross-match group centres (wave_group_ra/dec) against LAN centroids
   using astropy SkyCoord.match_to_catalog_sky.
5. Accept matches within MATCH_RADIUS_ARCSEC and flag additional
   redshift agreement (|Δz| < DELTA_Z_MAX).
6. Produce a four-panel figure:
     Panel 1: Sky map of groups, LANs, and matched pairs
     Panel 2: Separation distribution (matched vs random control)
     Panel 3: r_iso vs wave_group_a for matched pairs (size comparison)
     Panel 4: Redshift agreement Δz for matched pairs
7. Write matched catalog to CSV and print console summary.

Column origins
--------------
  From wave_group (detinfo aggregate):
    gid, n_mem, cra, cdec, cwave, z_lya, semi_a [arcsec], semi_a_kpc, field

  From LAN catalog (deduplicated):
    name, ra, dec, z_hetdex, source_type, r_iso [kpc], logl_lya,
    flag_resolved, dbic, r_s, field

Requirements
------------
  pip install astropy numpy matplotlib scipy pandas

Data
----
  hetdex_sc2_detinfo_v1.5.fits
  hetdex_lan_v0.3.fits
  Both at: https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/
"""

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

DETINFO_PATH = "hetdex_sc2_detinfo_v1.5.fits"
LAN_PATH     = "hetdex_lan_v0.3.fits"
SAVE_PATH    = "hetdex_wavegroup_lan_xmatch.png"
CSV_PATH     = "hetdex_wavegroup_lan_xmatch.csv"

# Group pre-filter (applied before crossmatch)
MIN_GROUP_MEMBERS   = 3       # minimum detections per group
MIN_GROUP_A_ARCSEC  = 3.0     # only groups with semi_a > this  [arcsec]
Z_LYA_MIN           = 1.87    # Lya blue limit in VIRUS band
Z_LYA_MAX           = 3.60    # Lya red limit
MIN_SN              = 4.5     # S/N cut on individual detections

# Match criteria
MATCH_RADIUS_ARCSEC = 5.0     # primary positional match radius
DELTA_Z_MAX         = 0.05    # |z_group - z_lan| < this for redshift agreement

# LAN quality cuts
LAN_MIN_DBIC        = -999.0  # set > 0 to restrict to resolved LANs
LAN_MIN_FLAG_RES    = 0       # 1 = resolved only

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
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.ticker  import AutoMinorLocator
from matplotlib.lines   import Line2D
import matplotlib.patches as mpatches

import pandas as pd

from astropy.io         import fits
from astropy.table      import Table
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units    as u
from astropy.cosmology  import Planck18

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
# CELL 3 — SYNTHETIC DATA GENERATORS (used when FITS files are absent)
# =============================================================================

def make_synthetic_detinfo(n_rows=80_000, seed=7):
    rng = np.random.default_rng(seed)
    ra  = np.concatenate([rng.normal(185,  25, int(n_rows*0.50)),
                          rng.normal(355,  20, int(n_rows*0.33)),
                          rng.normal(150.1, 0.6,int(n_rows*0.10)),
                          rng.normal(189.3, 0.5,int(n_rows*0.07))]) % 360
    dec = np.concatenate([rng.normal(51.5,  3, int(n_rows*0.50)),
                          rng.normal( 0.5,  3, int(n_rows*0.33)),
                          rng.normal( 2.2, 0.5,int(n_rows*0.10)),
                          rng.normal(62.2,  0.4,int(n_rows*0.07))])
    n   = len(ra)
    wave= rng.uniform(3500, 5500, n).astype(np.float32)
    sn  = np.abs(rng.lognormal(1.5, 0.6, n)).astype(np.float32)
    fields = rng.choice(["dex-spring","dex-fall","cosmos","goods-n"],
                         n, p=[0.50,0.33,0.10,0.07])

    # Build clustered groups
    wgid   = np.full(n, -1,  dtype=np.int64)
    wga    = np.full(n, BAD, dtype=np.float32)
    wgb    = np.full(n, BAD, dtype=np.float32)
    wgpa   = np.full(n, BAD, dtype=np.float32)
    wgra   = np.full(n, BAD, dtype=np.float32)
    wgdec  = np.full(n, BAD, dtype=np.float32)
    wgwave = np.full(n, BAD, dtype=np.float32)

    # 400 groups of sizes 3–60
    ptr = 0
    for gid in range(1, 401):
        gsz  = int(np.clip(rng.pareto(2.5)*3+3, 3, 60))
        if ptr + gsz > n: break
        cra  = ra[ptr]; cdec = dec[ptr]; cw = wave[ptr]
        ga   = float(rng.uniform(1, 15))
        wgid[ptr:ptr+gsz]   = gid
        wga[ptr:ptr+gsz]    = ga
        wgb[ptr:ptr+gsz]    = ga * rng.uniform(0.4, 1.0)
        wgpa[ptr:ptr+gsz]   = rng.uniform(0, 180)
        wgra[ptr:ptr+gsz]   = cra
        wgdec[ptr:ptr+gsz]  = cdec
        wgwave[ptr:ptr+gsz] = cw
        ra[ptr:ptr+gsz]     = cra  + rng.normal(0, ga/3600, gsz)
        dec[ptr:ptr+gsz]    = cdec + rng.normal(0, ga/3600, gsz)
        wave[ptr:ptr+gsz]   = cw   + rng.normal(0, 5, gsz)
        ptr += gsz

    z = (wave / LYA_AA - 1).astype(np.float32)
    stype = np.where(z > 1.87, "lae", "oii")

    return Table({
        "source_id"       : np.arange(n, dtype=np.int64),
        "ra_det"          : ra.astype(np.float32),
        "dec_det"         : dec.astype(np.float32),
        "z_hetdex"        : z,
        "source_type"     : stype,
        "detectid"        : np.arange(2_100_000_000,
                                       2_100_000_000+n, dtype=np.int64),
        "wave"            : wave,
        "sn"              : sn,
        "field"           : fields,
        "line_id"         : np.where(z > 1.87, "lya", "oii").astype("U8"),
        "wave_group_id"   : wgid,
        "wave_group_a"    : wga,
        "wave_group_b"    : wgb,
        "wave_group_pa"   : wgpa,
        "wave_group_ra"   : wgra,
        "wave_group_dec"  : wgdec,
        "wave_group_wave" : wgwave,
    })


def make_synthetic_lan(n_unique=3000, seed=42):
    """Synthetic LAN catalog seeded near real detinfo group positions."""
    rng = np.random.default_rng(seed)
    z   = rng.uniform(1.9, 3.5, n_unique).astype(np.float32)
    ra  = np.concatenate([
        rng.normal(185, 20, int(n_unique*0.55)),
        rng.normal(355, 15, int(n_unique*0.30)),
        rng.normal(150.1, 0.5, int(n_unique*0.10)),
        rng.normal(189.3, 0.4, int(n_unique-
                                   int(n_unique*0.55)-int(n_unique*0.30)-
                                   int(n_unique*0.10)))
    ]).astype(np.float32)[:n_unique] % 360
    dec = np.concatenate([
        rng.normal(51.5, 2.5, int(n_unique*0.55)),
        rng.normal( 0.5, 2.5, int(n_unique*0.30)),
        rng.normal( 2.2, 0.4, int(n_unique*0.10)),
        rng.normal(62.2, 0.3, int(n_unique-int(n_unique*0.55)
                                  -int(n_unique*0.30)-int(n_unique*0.10)))
    ]).astype(np.float32)[:n_unique]

    r_iso   = (10**( rng.uniform(41.5,43.5,n_unique)/40
                     - 0.5)*rng.lognormal(0,.3,n_unique)).astype(np.float32)
    logL    = rng.uniform(41.5, 43.8, n_unique).astype(np.float32)
    r_s     = (r_iso * rng.uniform(0.2, 0.5, n_unique)).astype(np.float32)
    flag_r  = (rng.uniform(size=n_unique) < 0.35).astype(np.int64)
    dbic    = np.where(flag_r, rng.uniform(2,40,n_unique),
                       rng.uniform(-5,3,n_unique)).astype(np.float32)
    iso_err = rng.uniform(0.05, 0.55, n_unique).astype(np.float32)
    dids    = np.arange(2_100_100_000, 2_100_100_000+n_unique, dtype=np.int64)
    fields  = rng.choice(["dex-spring","dex-fall","cosmos","goods-n"],
                          n_unique, p=[0.55,0.30,0.10,0.05])
    stype   = rng.choice(["lae","agn"], n_unique, p=[0.80,0.20])

    return Table({
        "name"           : [f"HLAN{d}" for d in dids],
        "ra"             : ra,
        "dec"            : dec,
        "source_type"    : stype,
        "z_hetdex"       : z,
        "detectid"       : dids,
        "field"          : fields,
        "r_iso"          : r_iso,
        "r_s"            : r_s,
        "r_s_err"        : (r_s*0.15).astype(np.float32),
        "logl_lya"       : logL,
        "logl_lya_err"   : rng.uniform(0.02,0.12,n_unique).astype(np.float32),
        "flag_resolved"  : flag_r,
        "dbic"           : dbic,
        "iso_rel_err"    : iso_err,
        "sb_1sigma_obs"  : rng.uniform(0.8,5.0,n_unique).astype(np.float32),
        "dups_detectid"  : [""] * n_unique,
    })


# =============================================================================
# CELL 4 — LOAD CATALOGS
# =============================================================================

def load_fits_lower(path):
    hdul = fits.open(path, memmap=True)
    tab  = Table(hdul[1].data)
    hdul.close()
    tab.rename_columns(tab.colnames, [c.lower() for c in tab.colnames])
    return tab

def getcol(tab, *cands):
    lc = {c.lower(): c for c in tab.colnames}
    for c in cands:
        if c.lower() in lc:
            return lc[c.lower()]
    raise KeyError(f"None of {cands} in table. Have: {list(tab.colnames)[:30]}")

# ── Detinfo ───────────────────────────────────────────────────────────────────
print("Loading detinfo ...")
try:
    det = load_fits_lower(DETINFO_PATH)
    print(f"  {len(det):,} rows, {len(det.colnames)} columns")
    SYN_DET = False
except FileNotFoundError:
    print("  Not found — using synthetic data.")
    det = make_synthetic_detinfo()
    SYN_DET = True

# ── LAN catalog ───────────────────────────────────────────────────────────────
print("Loading LAN catalog ...")
try:
    lan_raw = load_fits_lower(LAN_PATH)
    print(f"  {len(lan_raw):,} rows, {len(lan_raw.colnames)} columns")
    SYN_LAN = False
except FileNotFoundError:
    print("  Not found — using synthetic data.")
    lan_raw = make_synthetic_lan()
    SYN_LAN = True

SYNTHETIC = SYN_DET or SYN_LAN

# =============================================================================
# CELL 5 — DEDUPLICATE LAN
# =============================================================================

def deduplicate_lan(tab):
    dids  = np.array(tab["detectid"],      dtype=np.int64)
    dups  = np.array(tab["dups_detectid"], dtype=str)
    ierr  = np.array(tab["iso_rel_err"],   dtype=float)
    n_in  = len(tab)
    keep  = np.ones(n_in, dtype=bool)
    seen  = set()
    for idx in np.argsort(ierr):
        did = int(dids[idx])
        if did in seen:
            keep[idx] = False
            continue
        dup_str = dups[idx].strip()
        group   = ({int(x.strip()) for x in dup_str.replace(',', ' ').split() if x.strip()}
                   if dup_str else {did})
        seen   |= group
    dedup = tab[keep]
    print(f"  LAN dedup: {n_in:,} -> {keep.sum():,} unique sources "
          f"({n_in - keep.sum():,} duplicates removed)")
    return dedup

lan_dedup = deduplicate_lan(lan_raw)

# Apply LAN quality cuts
lan_mask  = (np.array(lan_dedup["flag_resolved"], dtype=int)
             >= LAN_MIN_FLAG_RES)
lan_mask &= (np.array(lan_dedup["dbic"],  dtype=float) >= LAN_MIN_DBIC)
lan_mask &= (np.array(lan_dedup["r_iso"], dtype=float) > 0)
lan_mask &= np.isfinite(np.array(lan_dedup["r_iso"], dtype=float))
lan       = lan_dedup[lan_mask]
print(f"  LAN after quality cuts: {len(lan):,}")

# LAN arrays
lan_ra   = np.array(lan["ra"],        dtype=float)
lan_dec  = np.array(lan["dec"],       dtype=float)
lan_z    = np.array(lan["z_hetdex"],  dtype=float)
lan_riso = np.array(lan["r_iso"],     dtype=float)
lan_logL = np.array(lan["logl_lya"],  dtype=float)
lan_type = np.array(lan["source_type"], dtype=str)
lan_fld  = np.array(lan["field"],     dtype=str)
lan_name = np.array(lan["name"],      dtype=str)
lan_dbic = np.array(lan["dbic"],      dtype=float)
lan_flag = np.array(lan["flag_resolved"], dtype=int)

# =============================================================================
# CELL 6 — BUILD GROUP SUMMARY FROM DETINFO
# =============================================================================

print("\nBuilding group records from detinfo ...")

RA_COL   = getcol(det, "ra_det", "ra")
DEC_COL  = getcol(det, "dec_det", "dec")
WGI_COL  = getcol(det, "wave_group_id")
WGA_COL  = getcol(det, "wave_group_a")
WGB_COL  = getcol(det, "wave_group_b")
WGPA_COL = getcol(det, "wave_group_pa")
WGRA_COL = getcol(det, "wave_group_ra")
WGDC_COL = getcol(det, "wave_group_dec")
WGW_COL  = getcol(det, "wave_group_wave")
SN_COL   = getcol(det, "sn")

ra_arr    = np.array(det[RA_COL],   dtype=float)
dec_arr   = np.array(det[DEC_COL],  dtype=float)
wgid_arr  = np.array(det[WGI_COL],  dtype=np.int64)
wga_arr   = np.array(det[WGA_COL],  dtype=float)
wgra_arr  = np.array(det[WGRA_COL], dtype=float)
wgdec_arr = np.array(det[WGDC_COL], dtype=float)
wgw_arr   = np.array(det[WGW_COL],  dtype=float)
sn_arr    = np.array(det[SN_COL],   dtype=float)
stype_arr = np.array(det[getcol(det,"source_type")], dtype=str)
field_arr = np.array(det[getcol(det,"field")],       dtype=str)

for arr in [wga_arr, wgra_arr, wgdec_arr, wgw_arr]:
    arr[arr == BAD] = np.nan

# Grouped mask: exclude sentinels (-1, -999) and bad S/N
grouped_mask = (wgid_arr > 0) & (sn_arr > MIN_SN) & (sn_arr != BAD)
g_ids        = wgid_arr[grouped_mask]
g_wga        = wga_arr[grouped_mask]
g_wgra       = wgra_arr[grouped_mask]
g_wgdec      = wgdec_arr[grouped_mask]
g_wgw        = wgw_arr[grouped_mask]
g_stype      = stype_arr[grouped_mask]
g_field      = field_arr[grouped_mask]

unique_gids, g_counts = np.unique(g_ids, return_counts=True)

group_records = []
for gid, n_mem in zip(unique_gids, g_counts):
    if n_mem < MIN_GROUP_MEMBERS:
        continue
    sel   = g_ids == gid
    cra   = float(np.nanmedian(g_wgra[sel]))
    cdec  = float(np.nanmedian(g_wgdec[sel]))
    cw    = float(np.nanmedian(g_wgw[sel]))
    if not (np.isfinite(cra) and np.isfinite(cdec) and np.isfinite(cw)):
        continue
    wga_v = g_wga[sel][np.isfinite(g_wga[sel])]
    a     = float(wga_v[0]) if len(wga_v) else np.nan
    z_lya = cw / LYA_AA - 1.0
    types, tcounts = np.unique(g_stype[sel], return_counts=True)
    dom_type = types[tcounts.argmax()]
    group_records.append(dict(
        gid=int(gid), n_mem=int(n_mem),
        cra=cra, cdec=cdec, cwave=cw,
        z_lya=z_lya, semi_a=a,
        dom_type=dom_type,
        field=g_field[sel][0].strip(),
    ))

group_records.sort(key=lambda r: r["n_mem"], reverse=True)
print(f"  Total groups (N>={MIN_GROUP_MEMBERS}): {len(group_records):,}")

# Physical sizes for groups in Lya window
for rec in group_records:
    z = rec["z_lya"]
    if Z_LYA_MIN < z < Z_LYA_MAX and np.isfinite(rec["semi_a"]):
        kpc = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec).value
        rec["semi_a_kpc"] = rec["semi_a"] * kpc
    else:
        rec["semi_a_kpc"] = np.nan

# ── Pre-filter groups for crossmatch ─────────────────────────────────────────
xm_groups = [r for r in group_records
             if (Z_LYA_MIN < r["z_lya"] < Z_LYA_MAX
                 and np.isfinite(r["semi_a"])
                 and r["semi_a"] >= MIN_GROUP_A_ARCSEC)]
print(f"  Groups in Lya window with semi_a>={MIN_GROUP_A_ARCSEC}\": "
      f"{len(xm_groups):,}")

# =============================================================================
# CELL 7 — CROSSMATCH
# =============================================================================

print(f"\nCross-matching {len(xm_groups):,} groups against "
      f"{len(lan):,} LANs ...")
print(f"  Match radius: {MATCH_RADIUS_ARCSEC}\"  |  "
      f"Redshift window: |Δz| < {DELTA_Z_MAX}")

# Build SkyCoord arrays
grp_coords = SkyCoord(
    ra  = [r["cra"]  for r in xm_groups] * u.deg,
    dec = [r["cdec"] for r in xm_groups] * u.deg,
)
lan_coords = SkyCoord(ra=lan_ra * u.deg, dec=lan_dec * u.deg)

# For each group find the nearest LAN
idx_lan, sep2d, _ = grp_coords.match_to_catalog_sky(lan_coords)
sep_arcsec = sep2d.arcsec

# Positional match flag
pos_match = sep_arcsec < MATCH_RADIUS_ARCSEC

# Redshift agreement flag
z_grp_arr  = np.array([r["z_lya"] for r in xm_groups])
z_lan_match= lan_z[idx_lan]
dz         = np.abs(z_grp_arr - z_lan_match)
z_match    = dz < DELTA_Z_MAX

# Combined match
both_match = pos_match & z_match

n_pos_only = pos_match.sum()
n_both     = both_match.sum()
print(f"  Positional matches only    : {n_pos_only:,}")
print(f"  Positional + redshift      : {n_both:,}")

# ── Build matched pair records ────────────────────────────────────────────────
matched_pairs = []
for i, (rec, li, sep, dz_i, pm, zm) in enumerate(
        zip(xm_groups, idx_lan, sep_arcsec, dz, pos_match, z_match)):
    matched_pairs.append({
        # Group columns
        "gid"          : rec["gid"],
        "n_mem"        : rec["n_mem"],
        "grp_ra"       : rec["cra"],
        "grp_dec"      : rec["cdec"],
        "grp_wave"     : rec["cwave"],
        "grp_z"        : rec["z_lya"],
        "semi_a_arcsec": rec["semi_a"],
        "semi_a_kpc"   : rec["semi_a_kpc"],
        "grp_dom_type" : rec["dom_type"],
        "grp_field"    : rec["field"],
        # Nearest LAN columns
        "lan_name"     : str(lan_name[li]),
        "lan_ra"       : lan_ra[li],
        "lan_dec"      : lan_dec[li],
        "lan_z"        : lan_z[li],
        "lan_r_iso_kpc": lan_riso[li],
        "lan_logL"     : lan_logL[li],
        "lan_type"     : str(lan_type[li]),
        "lan_flag_res" : int(lan_flag[li]),
        "lan_dbic"     : float(lan_dbic[li]),
        # Match quality
        "sep_arcsec"   : sep,
        "delta_z"      : dz_i,
        "pos_match"    : bool(pm),
        "z_match"      : bool(zm),
        "full_match"   : bool(pm and zm),
    })

df_all     = pd.DataFrame(matched_pairs)
df_matched = df_all[df_all["full_match"]].copy().reset_index(drop=True)
df_pos     = df_all[df_all["pos_match"]].copy().reset_index(drop=True)

# ── Random control: shuffle LAN positions ────────────────────────────────────
rng_ctrl   = np.random.default_rng(42)
n_ctrl     = len(xm_groups)
ctrl_idx   = rng_ctrl.integers(0, len(lan), n_ctrl)
ctrl_coords= SkyCoord(ra=lan_ra[ctrl_idx]*u.deg, dec=lan_dec[ctrl_idx]*u.deg)
_, ctrl_sep, _ = grp_coords.match_to_catalog_sky(ctrl_coords)
ctrl_sep_arcsec= ctrl_sep.arcsec

print(f"\n  Full matches (pos+z)       : {n_both:,}  "
      f"({100*n_both/max(len(xm_groups),1):.1f}% of xm_groups)")

# =============================================================================
# CELL 8 — SAVE CSV
# =============================================================================

if CSV_PATH:
    df_matched.to_csv(CSV_PATH, index=False, float_format="%.5f")
    print(f"\nMatched catalog saved -> {CSV_PATH}  ({len(df_matched)} rows)")

# =============================================================================
# CELL 9 — PLOTTING
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"
C_GRP = "#58a6ff"   # groups
C_LAN = "#3fb950"   # LANs
C_MAT = "#ffa657"   # matched pairs
C_CTL = "#8b949e"   # control

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
    return ax.legend(fontsize=8.5, facecolor="#21262d",
                     edgecolor=SPINE, labelcolor=TEXT, **kw)

fig = plt.figure(figsize=(17, 14))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(2, 2, figure=fig,
                         hspace=0.38, wspace=0.30,
                         left=0.08, right=0.97,
                         top=0.92,  bottom=0.07)

ax_sky  = fig.add_subplot(gs[0, :])   # wide top: sky map
ax_sep  = fig.add_subplot(gs[1, 0])   # separation CDF
ax_size = fig.add_subplot(gs[1, 1])   # r_iso vs semi_a

# Re-use gs bottom row with 3 columns via inner gridspec
gs_bot  = gridspec.GridSpecFromSubplotSpec(
              1, 3, subplot_spec=gs[1, :], wspace=0.32)
ax_sep  = fig.add_subplot(gs_bot[0])
ax_size = fig.add_subplot(gs_bot[1])
ax_dz   = fig.add_subplot(gs_bot[2])

# ── Panel 1: Sky map ──────────────────────────────────────────────────────────
style_ax(ax_sky,
         f"HETDEX wave_group × LAN cross-match  "
         f"(groups semi_a>={MIN_GROUP_A_ARCSEC}\", z_lya in "
         f"[{Z_LYA_MIN:.2f},{Z_LYA_MAX:.1f}])",
         "RA  (deg)", "Dec  (deg)", minor=False)

# All LANs (background)
ax_sky.scatter(lan_ra, lan_dec,
               s=1.5, c=C_LAN, alpha=0.20,
               linewidths=0, rasterized=True, label="LAN (all)")

# All pre-filtered groups
grp_ra_arr  = np.array([r["cra"]  for r in xm_groups])
grp_dec_arr = np.array([r["cdec"] for r in xm_groups])
ax_sky.scatter(grp_ra_arr, grp_dec_arr,
               s=6, c=C_GRP, alpha=0.45,
               linewidths=0, rasterized=True,
               label=f"Groups (semi_a>={MIN_GROUP_A_ARCSEC}\", N={len(xm_groups):,})")

# Matched pairs: draw connecting lines + highlight both ends
if len(df_matched) > 0:
    for _, row in df_matched.iterrows():
        ax_sky.plot([row["grp_ra"], row["lan_ra"]],
                    [row["grp_dec"], row["lan_dec"]],
                    color=C_MAT, lw=0.8, alpha=0.55, zorder=4)
    ax_sky.scatter(df_matched["grp_ra"],  df_matched["grp_dec"],
                   s=35, c=C_MAT, marker="o", zorder=5, linewidths=0,
                   label=f"Matched group centre (N={len(df_matched):,})")
    ax_sky.scatter(df_matched["lan_ra"],  df_matched["lan_dec"],
                   s=35, c=C_MAT, marker="D", zorder=5, linewidths=0,
                   label="Matched LAN centroid")

ax_sky.invert_xaxis()
ax_sky.set_xlim(np.nanmax([lan_ra.max(),  grp_ra_arr.max() if len(grp_ra_arr) else 360]) + 2,
                np.nanmin([lan_ra.min(),  grp_ra_arr.min() if len(grp_ra_arr) else 0])   - 2)
ax_sky.set_ylim(min(lan_dec.min(), grp_dec_arr.min() if len(grp_dec_arr) else -90) - 1,
                max(lan_dec.max(), grp_dec_arr.max() if len(grp_dec_arr) else  90) + 1)

mleg(ax_sky, loc="lower left", ncol=2)

# ── Panel 2: Separation CDF (real vs random) ──────────────────────────────────
style_ax(ax_sep,
         "Match separation distribution",
         "Angular separation  (arcsec)",
         "Cumulative fraction")

sep_lim    = 60.0   # arcsec display limit
bins_sep   = np.linspace(0, sep_lim, 120)

def cdf(arr, bins):
    h, _ = np.histogram(arr[arr < sep_lim], bins=bins)
    return np.cumsum(h) / max(h.sum(), 1)

cdf_all    = cdf(sep_arcsec,      bins_sep)
cdf_ctrl   = cdf(ctrl_sep_arcsec, bins_sep)

ax_sep.plot(bins_sep[1:], cdf_all,  color=C_GRP,  lw=2.0,
            label=f"Groups vs LANs  (N={len(xm_groups):,})")
ax_sep.plot(bins_sep[1:], cdf_ctrl, color=C_CTL,  lw=1.5, ls="--",
            label="Random control")
ax_sep.axvline(MATCH_RADIUS_ARCSEC, color=C_MAT, lw=1.2, ls=":",
               label=f"Match radius = {MATCH_RADIUS_ARCSEC}\"")

# Mark the matched fraction at the match radius
frac_real = float(cdf_all[np.searchsorted(bins_sep[1:], MATCH_RADIUS_ARCSEC)])
frac_ctrl = float(cdf_ctrl[np.searchsorted(bins_sep[1:], MATCH_RADIUS_ARCSEC)])
ax_sep.annotate(f"{frac_real*100:.1f}%",
                xy=(MATCH_RADIUS_ARCSEC, frac_real),
                xytext=(MATCH_RADIUS_ARCSEC + 3, frac_real + 0.04),
                color=C_GRP, fontsize=8.5,
                arrowprops=dict(arrowstyle="->",
                                color=C_GRP, lw=0.8))
ax_sep.annotate(f"{frac_ctrl*100:.1f}% (ctrl)",
                xy=(MATCH_RADIUS_ARCSEC, frac_ctrl),
                xytext=(MATCH_RADIUS_ARCSEC + 3, frac_ctrl - 0.06),
                color=C_CTL, fontsize=8.5,
                arrowprops=dict(arrowstyle="->",
                                color=C_CTL, lw=0.8))
ax_sep.set_xlim(0, sep_lim)
ax_sep.set_ylim(0, 1.05)
mleg(ax_sep)

# ── Panel 3: r_iso (LAN) vs semi_a (group) ────────────────────────────────────
style_ax(ax_size,
         "LAN size vs group extent  (matched pairs)",
         r"wave_group_a  (arcsec)",
         r"$r_{\rm iso}$  [proper kpc]")

if len(df_matched) > 0:
    sc = ax_size.scatter(
        df_matched["semi_a_arcsec"],
        df_matched["lan_r_iso_kpc"],
        c=df_matched["grp_z"],
        cmap="plasma", s=30, alpha=0.80,
        vmin=Z_LYA_MIN, vmax=Z_LYA_MAX,
        linewidths=0, zorder=5,
    )
    cb = fig.colorbar(sc, ax=ax_size, pad=0.02)
    cb.set_label(r"$z_{\rm Ly\alpha}$", color=MUTED, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    cb.outline.set_edgecolor(SPINE)

    # Physical size conversion: arcsec -> kpc at median z
    z_med = float(df_matched["grp_z"].median())
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_med).to(
        u.kpc/u.arcsec).value
    xlim = ax_size.get_xlim()
    ax_size2 = ax_size.twiny()
    ax_size2.set_xlim(np.array(xlim) * kpc_per_arcsec)
    ax_size2.set_xlabel(f"wave_group_a  [kpc  at z={z_med:.2f}]",
                        color=MUTED, fontsize=8.5)
    ax_size2.tick_params(colors=MUTED, labelsize=8)
    ax_size2.spines["top"].set_color(SPINE)

    # 1:1 line in kpc (assuming r_iso ~ semi_a)
    x_range = np.linspace(xlim[0], xlim[1], 100)
    ax_size.plot(x_range, x_range * kpc_per_arcsec,
                 color=MUTED, lw=1.0, ls="--", alpha=0.6,
                 label=f"1:1 (at z={z_med:.2f})")
    mleg(ax_size, loc="upper left")
else:
    ax_size.text(0.5, 0.5, "No full matches\nwith current thresholds",
                 ha="center", va="center", transform=ax_size.transAxes,
                 color=MUTED, fontsize=11)

# ── Panel 4: Δz distribution ──────────────────────────────────────────────────
style_ax(ax_dz,
         r"Redshift agreement  $\Delta z = |z_{\rm grp} - z_{\rm LAN}|$",
         r"$|\Delta z|$",
         "Number of groups")

bins_dz = np.linspace(0, 0.5, 40)
ax_dz.hist(df_pos["delta_z"] if len(df_pos) else [],
           bins=bins_dz, color=C_GRP, alpha=0.70,
           histtype="stepfilled", edgecolor="#1f6feb", lw=0.8,
           label=f"Pos. matches (N={len(df_pos):,})")
ax_dz.axvline(DELTA_Z_MAX, color=C_MAT, lw=1.5, ls="--",
              label=f"|Δz| = {DELTA_Z_MAX}")
ax_dz.set_xlim(0, 0.5)
mleg(ax_dz)

# ── Super-title & footer ──────────────────────────────────────────────────────
syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
fig.suptitle(
    rf"HETDEX SC2 wave_group × LAN Cross-match{syn_tag}",
    color=TEXT, fontsize=14, fontweight="bold", y=0.975,
)
fig.text(
    0.5, 0.005,
    (f"Groups: {len(xm_groups):,} (semi_a>={MIN_GROUP_A_ARCSEC}\", "
     f"z_lya in [{Z_LYA_MIN:.2f},{Z_LYA_MAX:.1f}])   |   "
     f"LANs: {len(lan):,}   |   "
     f"Positional matches (<{MATCH_RADIUS_ARCSEC}\"): {n_pos_only:,}   |   "
     f"Full matches (pos+Δz<{DELTA_Z_MAX}): {n_both:,}"),
    ha="center", fontsize=8.5, color=MUTED,
)

if SAVE_PATH:
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 10 — SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 100)
print(f"  HETDEX wave_group × LAN — Full Matches  "
      f"(sep < {MATCH_RADIUS_ARCSEC}\", |Δz| < {DELTA_Z_MAX})")
print("=" * 100)

if len(df_matched) == 0:
    print(f"  No matches found with current thresholds.")
    print(f"  Try relaxing MATCH_RADIUS_ARCSEC (currently {MATCH_RADIUS_ARCSEC}\")")
    print(f"  or DELTA_Z_MAX (currently {DELTA_Z_MAX})")
    print(f"  or MIN_GROUP_A_ARCSEC (currently {MIN_GROUP_A_ARCSEC}\")")
else:
    hfmt = "  {:>4}  {:>12}  {:>5}  {:>7}  {:>9}  {:>8}  {:>6}  {:>6}  {:>10}  {:>6}  {:>5}"
    rfmt = "  {:>4}  {:>12}  {:>5}  {:>7.3f}  {:>9.4f}  {:>8.4f}  {:>6.3f}  {:>6.1f}  {:>10.1f}  {:>6.1f}  {:>5.3f}"
    hdr_cols = ("Rank","LAN name","N_mem","sep\"","grp_ra",
                "grp_dec","z_grp","semi_a","r_iso_kpc","logL","dz")
    print(hfmt.format(*hdr_cols))
    print("  " + "-"*98)
    for rank, (_, row) in enumerate(df_matched.iterrows(), 1):
        print(rfmt.format(
            rank,
            row["lan_name"][:12],
            int(row["n_mem"]),
            row["sep_arcsec"],
            row["grp_ra"],
            row["grp_dec"],
            row["grp_z"],
            row["semi_a_arcsec"],
            row["lan_r_iso_kpc"],
            row["lan_logL"],
            row["delta_z"],
        ))

print("=" * 100)
print(f"\n  Statistics:")
print(f"    Groups in Lya window (semi_a>={MIN_GROUP_A_ARCSEC}\"): {len(xm_groups):,}")
print(f"    LANs after quality cuts                     : {len(lan):,}")
print(f"    Positional matches (<{MATCH_RADIUS_ARCSEC}\")           : "
      f"{n_pos_only:,}  ({100*n_pos_only/max(len(xm_groups),1):.1f}%)")
print(f"    Full matches (pos + |Δz|<{DELTA_Z_MAX})         : "
      f"{n_both:,}  ({100*n_both/max(len(xm_groups),1):.1f}%)")
if len(df_matched) > 0:
    print(f"    Median sep (matched)                        : "
          f"{df_matched['sep_arcsec'].median():.2f}\"")
    print(f"    Median |Δz| (matched)                       : "
          f"{df_matched['delta_z'].median():.4f}")
    print(f"    Median r_iso of matched LANs                : "
          f"{df_matched['lan_r_iso_kpc'].median():.1f} kpc")
    print(f"    Median semi_a of matched groups             : "
          f"{df_matched['semi_a_arcsec'].median():.1f}\"")
    print(f"    LAN source types in matched set             : "
          f"{dict(zip(*np.unique(df_matched['lan_type'], return_counts=True)))}")
