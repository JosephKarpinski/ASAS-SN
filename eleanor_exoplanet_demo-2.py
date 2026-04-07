"""
=============================================================
Eleanor + Lightkurve — Target-Agnostic Exoplanet Demo
=============================================================

Reads a NASA Exoplanet Archive dataframe (df) and extracts all
parameters needed for Eleanor FFI photometry and Lightkurve
transit analysis from the row for the chosen planet.

Currently configured for:  Kepler-6 b  (TIC 27916356)

To switch targets, change PLANET_NAME to any pl_name in df.

Install dependencies:
    pip install eleanor lightkurve matplotlib astropy pandas
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import eleanor
import lightkurve as lk

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 ─ Load dataframe and extract target parameters
# ══════════════════════════════════════════════════════════════════════════════
# All science parameters (TIC ID, period, T0, transit depth, duration)
# are pulled directly from the NASA Exoplanet Archive dataframe row.
# Nothing below this block needs editing when switching targets.

# ── Load the dataframe ────────────────────────────────────────────────────
# Replace this path with wherever your CSV lives, or pass df in directly.
# df = pd.read_csv("nasa_exoplanet_archive.csv")

# For demonstration we reconstruct the Kepler-6 b row from the document:
row_data = {
    "pl_name":      "Kepler-6 b",
    "hostname":     "Kepler-6",
    "tic_id":       "TIC 27916356",
    "tic_id_clean": 27916356,
    "ra":           296.837247,
    "dec":          48.23997,
    # Orbital period (days) and uncertainty
    "pl_orbper":    3.2347,
    "pl_orbpererr1": 4e-7,
    "pl_orbpererr2": -4e-7,
    # Transit mid-time (BJD) — from pl_tranmid column
    "pl_tranmid":   2454954.48652,   # BJD
    # Transit depth (fraction) — pl_trandep is in % in NExSci; store as ppm
    # From document: 0.4 ± 0.06 %  →  store as fraction 0.004
    "pl_trandep":   0.40,            # percent
    # Transit duration (hours) — pl_trandur
    "pl_trandur":   np.nan,          # not given in excerpt; we derive below
    # Planet radius (Jupiter radii)
    "pl_radj":      1.304,
    # Stellar parameters
    "st_teff":      5647.0,
    "st_rad":       1.391,
    "sy_tmag":      12.6877,
}
df = pd.DataFrame([row_data])

PLANET_NAME = "Kepler-6 b"

# ── Extract the row ───────────────────────────────────────────────────────
row = df[df["pl_name"] == PLANET_NAME].iloc[0]

# Core identifiers
TIC_ID    = int(row["tic_id_clean"])
RA_DEG    = float(row["ra"])
DEC_DEG   = float(row["dec"])
HOST_NAME = str(row["hostname"])
PL_NAME   = str(row["pl_name"])
PL_LETTER = PL_NAME.split()[-1]          # "b", "c", etc.

# Ephemeris
KNOWN_PERIOD   = float(row["pl_orbper"])              # days
KNOWN_T0_BJD   = float(row["pl_tranmid"])             # BJD
KNOWN_T0_BTJD  = KNOWN_T0_BJD - 2457000.0            # → BTJD (TESS standard)

# Transit depth — NExSci stores pl_trandep in percent; convert to fraction
TRANSIT_DEPTH_PCT  = float(row["pl_trandep"])         # e.g. 0.40 %
TRANSIT_DEPTH_FRAC = TRANSIT_DEPTH_PCT / 100.0        # e.g. 0.004

# Transit duration — derive from pl_trandur if available, else estimate
if pd.notna(row.get("pl_trandur", np.nan)):
    TRANSIT_DUR_HR  = float(row["pl_trandur"])        # hours
    TRANSIT_DUR_DAY = TRANSIT_DUR_HR / 24.0
else:
    # Rough estimate: ~10% of the orbital period for hot Jupiters
    TRANSIT_DUR_DAY = KNOWN_PERIOD * 0.035
    print(f"    Transit duration not in df — estimated as {TRANSIT_DUR_DAY:.4f} d")

HALF_DUR = TRANSIT_DUR_DAY / 2.0

# BLS search window: ±20% around known period
BLS_P_MIN = KNOWN_PERIOD * 0.80
BLS_P_MAX = KNOWN_PERIOD * 1.20

print("=" * 60)
print(f"  Eleanor + Lightkurve  —  {PL_NAME}")
print("=" * 60)
print(f"    TIC ID         : {TIC_ID}")
print(f"    RA / Dec       : {RA_DEG:.5f}°  /  {DEC_DEG:.5f}°")
print(f"    Known period   : {KNOWN_PERIOD} d")
print(f"    Known T0 (BTJD): {KNOWN_T0_BTJD:.4f}")
print(f"    Transit depth  : {TRANSIT_DEPTH_PCT:.3f} %  ({TRANSIT_DEPTH_FRAC:.5f})")
print(f"    Transit dur.   : {TRANSIT_DUR_DAY:.4f} d  ({TRANSIT_DUR_DAY*24:.2f} hr)")
print(f"    BLS search     : {BLS_P_MIN:.3f} – {BLS_P_MAX:.3f} d")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 ─ Discover TESS sectors
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1] Finding all available TESS sectors …")
stars = eleanor.multi_sectors(sectors="all", tic=TIC_ID)
print(f"    Found {len(stars)} sector(s):")
for s in stars:
    print(f"      Sector {s.sector:>3d}  |  camera {s.camera}  chip {s.chip}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 ─ Build TargetData — FFI extraction + custom aperture
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2] Building TargetData objects (TPF + light curve) …")
all_data = []
for star in stars:
    print(f"    Processing sector {star.sector} …")
    data = eleanor.TargetData(
        star,
        height=13,
        width=13,
        do_pca=True,
        do_psf=False,           # disabled — tf.logging removed in TensorFlow 2.x
        aperture_mode="normal",
    )

    # Locate star as brightest pixel in central 7×7 region only.
    # This avoids corner/edge artefacts that Eleanor's WCS sometimes
    # places brighter than the target when TessCut is used as fallback.
    mean_frame    = np.nanmean(data.tpf, axis=0)
    center_region = mean_frame[3:10, 3:10]
    by_loc, bx_loc = np.unravel_index(np.nanargmax(center_region),
                                       center_region.shape)
    bright_x = bx_loc + 3
    bright_y = by_loc + 3
    print(f"      Brightest pixel in central 7×7 : ({bright_x}, {bright_y})")

    data.custom_aperture(shape="circle", r=2.5,
                         pos=(bright_x, bright_y))
    data.get_lightcurve()
    print(f"      Custom aperture pixels  : {np.sum(data.aperture > 0)}")

    all_data.append(data)
    print(f"      Cadences : {len(data.time)}")
    print(f"      Quality flags set : {np.sum(data.quality > 0)}")

first = all_data[0]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 ─ Aperture summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Aperture information …")
print(f"    Aperture shape (pixels)    : {first.aperture.shape}")
print(f"    Number of apertures tested : {len(first.all_apertures)}")
print(f"    Best aperture index        : {first.best_ind}")
print(f"    Non-zero aperture pixels   : {np.sum(first.aperture > 0)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 ─ Stitch all sectors
# ══════════════════════════════════════════════════════════════════════════════

print("\n[4] Stitching sectors …")
if len(all_data) > 1:
    time_stitched, flux_stitched, quality_stitched, fluxerr_stitched = (
        first.stitch(all_data, flux="corrected")
    )
    print(f"    Total stitched cadences : {len(time_stitched)}")
else:
    mask = first.quality == 0
    time_stitched    = first.time[mask]
    flux_stitched    = first.corr_flux[mask] / np.nanmedian(first.corr_flux[mask])
    fluxerr_stitched = first.flux_err[mask]  / np.nanmedian(first.corr_flux[mask])
    quality_stitched = first.quality[mask]
    print(f"    Single sector, {len(time_stitched)} good cadences")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 ─ Eleanor → Lightkurve, detrend, BLS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[5] Converting Eleanor TargetData → Lightkurve TessLightCurve …")
lc_eleanor = first.to_lightkurve()
print(f"    Type            : {type(lc_eleanor)}")
print(f"    Length          : {len(lc_eleanor)}")
print(f"    Time format     : {lc_eleanor.time.format}")
print(f"    Flux columns    : {lc_eleanor.colnames}")

# ── Single-sector light curve for Panel D ────────────────────────────────
lc_norm    = lc_eleanor.remove_nans().normalize()
lc_trimmed = lc_norm[50:]                           # drop momentum-dump edge
lc_flat, trend = lc_trimmed.flatten(window_length=101, return_trend=True)

print("\n    Running Lomb-Scargle periodogram …")
pg = lc_flat.to_periodogram(method="lombscargle",
                             minimum_period=0.5,
                             maximum_period=30)
print(f"    Dominant LS period : {pg.period_at_max_power:.4f}")

# ── Stitched light curve for BLS ─────────────────────────────────────────
lc_stitched = lk.LightCurve(time=time_stitched,
                              flux=flux_stitched,
                              flux_err=fluxerr_stitched)
lc_stitched         = lc_stitched.remove_nans().normalize()
lc_stitched_trimmed = lc_stitched[50:]
lc_flat_stitched, trend_stitched = lc_stitched_trimmed.flatten(
    window_length=101, return_trend=True)
lc_clipped = lc_flat_stitched.remove_outliers(sigma=4.0)
print(f"    Cadences after sigma clipping : {len(lc_clipped)}")

print(f"    Running BLS  ({BLS_P_MIN:.3f} – {BLS_P_MAX:.3f} d) …")
bls = lc_clipped.to_periodogram(
    method="bls",
    minimum_period=BLS_P_MIN,
    maximum_period=BLS_P_MAX,
    frequency_factor=150,
)
bls_period   = bls.period_at_max_power
bls_t0       = bls.transit_time_at_max_power
bls_duration = bls.duration_at_max_power
print(f"    BLS best period  : {bls_period.value:.4f} {bls_period.unit}  "
      f"(known: {KNOWN_PERIOD:.4f} d, "
      f"Δ = {abs(bls_period.value - KNOWN_PERIOD):.4f} d)")
print(f"    BLS transit time : {bls_t0.value:.4f}")
print(f"    BLS transit dur. : {bls_duration.value:.4f} {bls_duration.unit}")

# Phase-fold on BLS-recovered period (Panel F)
lc_folded = lc_clipped.fold(period=bls_period, epoch_time=bls_t0.value)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 ─ Custom aperture demo
# ══════════════════════════════════════════════════════════════════════════════

print("\n[6] Custom circular aperture demo (r = 2.5 px, default centre) …")
first.custom_aperture(shape="circle", r=2.5)
first.get_lightcurve()
print(f"    Custom aperture pixels : {np.sum(first.aperture > 0)}")
print(f"    Re-extracted cadences  : {len(first.raw_flux)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 ─ Background subtraction
# ══════════════════════════════════════════════════════════════════════════════

print("\n[7] Background subtraction …")
first.bkg_subtraction(scope="tpf", sigma=2.5)
print("    Applied (scope='tpf', sigma=2.5)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 ─ Main six-panel figure
# ══════════════════════════════════════════════════════════════════════════════

print("\n[8] Generating main figure …")

# Derive safe output filename from planet name
safe_name = PL_NAME.replace(" ", "_").replace("/", "-")

fig = plt.figure(figsize=(16, 14))
fig.suptitle(f"{PL_NAME}  —  Eleanor + Lightkurve Demo\n"
             f"(TIC {TIC_ID}  ·  Sector {first.source_info.sector})",
             fontsize=14, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

# ── Panel A: Mean TPF ─────────────────────────────────────────────────────
ax_tpf = fig.add_subplot(gs[0, 0])
mean_frame = np.nanmean(first.tpf, axis=0)
im = ax_tpf.imshow(mean_frame, origin="lower", cmap="YlOrRd",
                   interpolation="nearest")
plt.colorbar(im, ax=ax_tpf, label="Mean Flux (e⁻/s)")
ax_tpf.contour(first.aperture, levels=[0.5], colors="cyan", linewidths=1.5)
ax_tpf.set_title("A — Mean TPF + aperture (cyan)\n"
                 "Centred on brightest pixel in central 7×7", fontsize=8)
ax_tpf.set_xlabel("Pixel column")
ax_tpf.set_ylabel("Pixel row")

# ── Panel B: Centroid motion ──────────────────────────────────────────────
ax_cen = fig.add_subplot(gs[0, 1])
ax_cen.scatter(first.centroid_xs, first.centroid_ys,
               c=first.time, cmap="plasma", s=4, alpha=0.6)
ax_cen.set_title("B — Centroid motion (colour = time)", fontsize=9)
ax_cen.set_xlabel("Centroid x (postcard px)")
ax_cen.set_ylabel("Centroid y (postcard px)")

# ── Panel C: Raw vs corrected flux ────────────────────────────────────────
ax_raw = fig.add_subplot(gs[1, :])
good  = first.quality == 0
t     = first.time[good]
raw   = first.raw_flux[good]  / np.nanmedian(first.raw_flux[good])
corr  = first.corr_flux[good] / np.nanmedian(first.corr_flux[good])
ax_raw.plot(t, raw,  "gray",      lw=0.6, alpha=0.7, label="Raw flux")
ax_raw.plot(t, corr, "steelblue", lw=0.9, label="Corrected flux (Eleanor)")
if hasattr(first, "pca_flux") and first.pca_flux is not None:
    pca = first.pca_flux[good] / np.nanmedian(first.pca_flux[good])
    ax_raw.plot(t, pca, "darkorange", lw=0.7, alpha=0.8, label="PCA flux")
ax_raw.set_title("C — Eleanor raw vs systematics-corrected flux", fontsize=9)
ax_raw.set_xlabel("Time (BTJD)")
ax_raw.set_ylabel("Normalised flux")
ax_raw.legend(fontsize=7, loc="upper right")

# ── Panel D: Lightkurve flattening ───────────────────────────────────────
ax_flat = fig.add_subplot(gs[2, :])
ax_flat.plot(lc_trimmed.time.value, lc_trimmed.flux.value,
             "gray", lw=0.5, alpha=0.5, label="Normalised (LK, trimmed)")
ax_flat.plot(trend.time.value, trend.flux.value,
             "tomato", lw=1.5, label="S-G trend (LK)")
ax_flat.plot(lc_flat.time.value, lc_flat.flux.value,
             "steelblue", lw=0.7, alpha=0.8, label="Flattened (LK)")
ax_flat.set_title("D — Lightkurve normalisation & S-G flattening  "
                  "(data from Eleanor via to_lightkurve(),  first 50 cadences trimmed)",
                  fontsize=8)
ax_flat.set_xlabel("Time (BTJD)")
ax_flat.set_ylabel("Normalised flux")
ax_flat.legend(fontsize=7, loc="upper right")

# ── Panel E: BLS periodogram ─────────────────────────────────────────────
ax_bls = fig.add_subplot(gs[3, 0])
ax_bls.plot(bls.period.value, bls.power.value, "steelblue", lw=0.8)
ax_bls.axvline(bls_period.value, color="tomato", lw=1.5, ls="--",
               label=f"BLS P = {bls_period.value:.3f} d")
ax_bls.axvline(KNOWN_PERIOD, color="green", lw=1.0, ls=":",
               label=f"Known P = {KNOWN_PERIOD:.4f} d")
ax_bls.set_title("E — BLS periodogram  (stitched + sigma-clipped)", fontsize=9)
ax_bls.set_xlabel("Period (days)")
ax_bls.set_ylabel("BLS power")
ax_bls.legend(fontsize=7)

# ── Panel F: BLS phase fold ───────────────────────────────────────────────
ax_fold = fig.add_subplot(gs[3, 1])
ax_fold.scatter(lc_folded.time.value, lc_folded.flux.value,
                s=3, alpha=0.4, color="steelblue", label="Folded (BLS period)")
lc_binned = lc_folded.bin(time_bin_size=0.01)
ax_fold.plot(lc_binned.time.value, lc_binned.flux.value,
             "tomato", lw=1.5, label="Binned")
ax_fold.axvline(0, color="gray", lw=0.8, ls=":")
ax_fold.axhline(1.0 - TRANSIT_DEPTH_FRAC, color="green", lw=0.8, ls="--",
                label=f"Expected depth {TRANSIT_DEPTH_PCT:.2f}%")
ax_fold.set_title(f"F — BLS phase fold  P = {bls_period.value:.3f} d\n"
                  "(stitched, sigma-clipped)", fontsize=8)
ax_fold.set_xlabel("Phase (days from mid-transit)")
ax_fold.set_ylabel("Normalised flux")
ax_fold.legend(fontsize=7)

plt.savefig(f"./{safe_name}_eleanor_demo.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Figure saved → {safe_name}_eleanor_demo.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 ─ Known-ephemeris fold
# ══════════════════════════════════════════════════════════════════════════════

print("\n[9] Folding on known ephemeris from dataframe …")

# ── Propagate T0 to the TESS epoch ───────────────────────────────────────
# The discovery T0 (e.g. Kepler 2009) is thousands of cycles before TESS.
# Even a perfect period accumulates phase drift over that many cycles.
# We shift T0 forward by an integer number of periods to the nearest
# transit within the TESS data window — mathematically identical to the
# original ephemeris, no new assumptions introduced.
tess_ref_time = float(np.nanmedian(lc_clipped.time.value))
n_cycles      = np.round((tess_ref_time - KNOWN_T0_BTJD) / KNOWN_PERIOD)
KNOWN_T0_TESS = KNOWN_T0_BTJD + n_cycles * KNOWN_PERIOD

print(f"    Original T0     : {KNOWN_T0_BTJD:.4f} BTJD  (discovery epoch)")
print(f"    TESS ref time   : {tess_ref_time:.4f} BTJD  (data midpoint)")
print(f"    Cycles elapsed  : {int(n_cycles)}")
print(f"    Updated T0      : {KNOWN_T0_TESS:.4f} BTJD  (propagated to TESS era)")
print(f"    Period          : {KNOWN_PERIOD} d")

lc_folded_known = lc_clipped.fold(period=KNOWN_PERIOD,
                                   epoch_time=KNOWN_T0_TESS)

transit_half_window = max(HALF_DUR * 3, 0.2)   # zoom window = 3× half-duration
mask_zoom  = np.abs(lc_folded_known.time.value) < transit_half_window
lc_zoom    = lc_folded_known[mask_zoom]

lc_fold_binned = lc_folded_known.bin(time_bin_size=0.02)
lc_zoom_binned = lc_zoom.bin(time_bin_size=max(TRANSIT_DUR_DAY / 15, 0.005))

print(f"    Zoom window : ±{transit_half_window:.3f} d  "
      f"({mask_zoom.sum()} cadences)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 ─ Large-aperture re-extraction
# ══════════════════════════════════════════════════════════════════════════════

print("\n[10] Re-extracting first sector with aperture_mode='large' …")

data_large = eleanor.TargetData(
    stars[0],
    height=15,
    width=15,
    do_pca=True,
    do_psf=False,
    aperture_mode="large",
)

# ── FIX: reuse the star position found in Step 2 for the first sector ────
# The large TPF is 15×15 so the Step 2 position (bright_x, bright_y from
# the first sector) maps directly — no need to re-search and risk locking
# onto background pixels inside the larger frame.
first_sector_data = all_data[0]
mean_frame_lg = np.nanmean(first_sector_data.tpf, axis=0)
cr_lg = mean_frame_lg[3:10, 3:10]
by_lg, bx_lg = np.unravel_index(np.nanargmax(cr_lg), cr_lg.shape)
lg_star_x = bx_lg + 3
lg_star_y = by_lg + 3
print(f"    Large aperture star position (7×7) : ({lg_star_x}, {lg_star_y})")

data_large.custom_aperture(shape="circle", r=3.5,
                            pos=(lg_star_x, lg_star_y))
data_large.get_lightcurve()

print(f"    Large aperture pixels              : {np.sum(data_large.aperture > 0)}")
print(f"    Cadences                           : {len(data_large.time)}")

lc_large = lk.LightCurve(
    time=data_large.time[data_large.quality == 0],
    flux=data_large.corr_flux[data_large.quality == 0],
)
lc_large         = lc_large.remove_nans().normalize()
lc_large_trimmed = lc_large[50:]
lc_large_flat, _ = lc_large_trimmed.flatten(window_length=101,
                                              return_trend=True)
lc_large_clipped = lc_large_flat.remove_outliers(sigma=4.0)

# FIX: use propagated KNOWN_T0_TESS (not raw discovery T0) for all folds
lc_large_folded      = lc_large_clipped.fold(period=KNOWN_PERIOD,
                                              epoch_time=KNOWN_T0_TESS)
lc_large_fold_binned = lc_large_folded.bin(time_bin_size=0.02)

print(f"    Cadences after clipping            : {len(lc_large_clipped)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 ─ Transit supplement figure
# ══════════════════════════════════════════════════════════════════════════════

print("\n[11] Generating transit supplement figure …")

fig2 = plt.figure(figsize=(16, 12))
fig2.suptitle(f"{PL_NAME}  —  Transit Detection Supplement\n"
              f"(TIC {TIC_ID}  ·  Known P = {KNOWN_PERIOD} d)",
              fontsize=14, fontweight="bold", y=0.98)
gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.50, wspace=0.35)

# ── Panel G: Full known-ephemeris fold ────────────────────────────────────
ax_g = fig2.add_subplot(gs2[0, 0])
ax_g.scatter(lc_folded_known.time.value, lc_folded_known.flux.value,
             s=2, alpha=0.25, color="steelblue", label="Folded (stitched)")
ax_g.plot(lc_fold_binned.time.value, lc_fold_binned.flux.value,
          "tomato", lw=2.0, label="Binned (20 min)")
ax_g.axvline(0, color="gray", lw=0.8, ls=":")
ax_g.axhline(1.0 - TRANSIT_DEPTH_FRAC, color="green", lw=1.0, ls="--",
             label=f"Expected depth {TRANSIT_DEPTH_PCT:.2f}%")
ax_g.set_xlim(-KNOWN_PERIOD / 2, KNOWN_PERIOD / 2)
ax_g.set_title("G — Full phase fold on KNOWN ephemeris\n"
               f"P = {KNOWN_PERIOD} d  ·  normal aperture  ·  stitched",
               fontsize=8)
ax_g.set_xlabel("Phase (days from mid-transit)")
ax_g.set_ylabel("Normalised flux")
ax_g.legend(fontsize=7)

# ── Panel H: Zoomed transit window ───────────────────────────────────────
ax_h = fig2.add_subplot(gs2[0, 1])
ax_h.scatter(lc_zoom.time.value, lc_zoom.flux.value,
             s=8, alpha=0.4, color="steelblue",
             label=f"Folded (±{transit_half_window:.2f} d)")
ax_h.plot(lc_zoom_binned.time.value, lc_zoom_binned.flux.value,
          "tomato", lw=2.5, label="Binned")
ax_h.axvline(0,          color="gray",   lw=0.8, ls=":",  label="Mid-transit")
ax_h.axvline(-HALF_DUR,  color="purple", lw=0.9, ls="--", alpha=0.7,
             label=f"Ingress/egress (±{HALF_DUR:.3f} d)")
ax_h.axvline(+HALF_DUR,  color="purple", lw=0.9, ls="--", alpha=0.7)
ax_h.axhline(1.0 - TRANSIT_DEPTH_FRAC, color="green", lw=1.0, ls="--",
             label=f"Expected depth {TRANSIT_DEPTH_PCT:.2f}%")
ax_h.set_title(f"H — Zoomed transit window  (±{transit_half_window:.2f} d)\n"
               "Purple dashes = expected ingress / egress", fontsize=8)
ax_h.set_xlabel("Phase (days from mid-transit)")
ax_h.set_ylabel("Normalised flux")
ax_h.legend(fontsize=7)

# ── Panel I: Large-aperture TPF ──────────────────────────────────────────
ax_i = fig2.add_subplot(gs2[1, 0])
mean_frame_lg2 = np.nanmean(data_large.tpf, axis=0)
im2 = ax_i.imshow(mean_frame_lg2, origin="lower", cmap="YlOrRd",
                  interpolation="nearest")
plt.colorbar(im2, ax=ax_i, label="Mean Flux (e⁻/s)")
ax_i.contour(data_large.aperture, levels=[0.5], colors="cyan", linewidths=1.5)
ax_i.set_title("I — Large aperture TPF  (aperture_mode='large', r=3.5 px)\n"
               "More stellar flux captured → less transit dilution", fontsize=8)
ax_i.set_xlabel("Pixel column")
ax_i.set_ylabel("Pixel row")

# ── Panel J: Large-aperture known-ephemeris fold ──────────────────────────
ax_j = fig2.add_subplot(gs2[1, 1])
ax_j.scatter(lc_large_folded.time.value, lc_large_folded.flux.value,
             s=2, alpha=0.25, color="darkorange", label="Folded (large ap.)")
ax_j.plot(lc_large_fold_binned.time.value, lc_large_fold_binned.flux.value,
          "tomato", lw=2.0, label="Binned (20 min)")
ax_j.axvline(0,         color="gray",   lw=0.8, ls=":")
ax_j.axvline(-HALF_DUR, color="purple", lw=0.9, ls="--", alpha=0.7)
ax_j.axvline(+HALF_DUR, color="purple", lw=0.9, ls="--", alpha=0.7)
ax_j.axhline(1.0 - TRANSIT_DEPTH_FRAC, color="green", lw=1.0, ls="--",
             label=f"Expected depth {TRANSIT_DEPTH_PCT:.2f}%")
ax_j.set_xlim(-transit_half_window, transit_half_window)
ax_j.set_title("J — Large-aperture fold on KNOWN ephemeris\n"
               f"First sector only  ·  aperture_mode='large'", fontsize=8)
ax_j.set_xlabel("Phase (days from mid-transit)")
ax_j.set_ylabel("Normalised flux")
ax_j.legend(fontsize=7)

plt.savefig(f"./{safe_name}_transit_supplement.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Supplementary figure saved → {safe_name}_transit_supplement.png")


print("\n" + "=" * 60)
print(f"  All done!  Target : {PL_NAME}  (TIC {TIC_ID})")
print("  Parameters extracted from dataframe df:")
print(f"    pl_orbper  → KNOWN_PERIOD          = {KNOWN_PERIOD} d")
print(f"    pl_tranmid → KNOWN_T0_BTJD         = {KNOWN_T0_BTJD:.4f}  (discovery epoch)")
print(f"               → KNOWN_T0_TESS         = {KNOWN_T0_TESS:.4f}  (propagated +{int(n_cycles)} cycles)")
print(f"    pl_trandep → TRANSIT_DEPTH_FRAC    = {TRANSIT_DEPTH_FRAC:.5f}")
print(f"    pl_trandur → TRANSIT_DUR_DAY       = {TRANSIT_DUR_DAY:.4f} d")
print(f"    tic_id_clean → TIC_ID              = {TIC_ID}")
print("  Eleanor data products accessed:")
print("    • eleanor.multi_sectors()         — sector discovery")
print("    • eleanor.TargetData              — TPF + raw/corr/PCA flux")
print("    • data.custom_aperture()          — brightest-pixel aperture")
print("    • data.centroid_xs / _ys          — pointing-model centroid")
print("    • data.bkg_subtraction()          — background removal")
print("    • data.stitch()                   — multi-sector stitching")
print("    • data.to_lightkurve()            — Eleanor → Lightkurve bridge")
print("    • aperture_mode='large'           — large aperture extraction")
print("  Lightkurve operations:")
print("    • lc.normalize() / .flatten()     — detrending")
print("    • lc.remove_outliers(sigma=4.0)   — sigma clipping")
print("    • lc.to_periodogram('bls')        — BLS transit search")
print("    • lc.fold(KNOWN_PERIOD, T0)       — known-ephemeris phase fold")
print("=" * 60)
