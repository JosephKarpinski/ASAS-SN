"""
=============================================================
WASP-39b Light Curve Extraction with Eleanor + Lightkurve
=============================================================

WASP-39b is a well-known hot Jupiter exoplanet (TIC 422756824).
This script demonstrates Eleanor's core features for FFI-based
photometry and shows how Eleanor hands off to Lightkurve for
downstream analysis using the to_lightkurve() bridge.

Install dependencies:
    pip install eleanor lightkurve matplotlib astropy
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.stats import sigma_clip

import eleanor
import lightkurve as lk

# ── Target ────────────────────────────────────────────────────────────────────
# WASP-39 (host star): TIC 422756824
# RA = 217.9490°, Dec = -3.4445°  (J2000)
TIC_ID    = 422756824
RA_DEG    = 217.9490
DEC_DEG   = -3.4445
STAR_NAME = "WASP-39"

print("=" * 60)
print(f"  Eleanor + Lightkurve demo  —  {STAR_NAME}b")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 ─ Discover which TESS sectors observed WASP-39
# ══════════════════════════════════════════════════════════════════════════════
# eleanor.multi_sectors('all') queries every sector the target
# appears in.  Each returned Source knows its sector, camera & chip.

print("\n[1] Finding all available TESS sectors …")
stars = eleanor.multi_sectors(
    sectors="all",
    tic=TIC_ID,
)
print(f"    Found {len(stars)} sector(s):")
for s in stars:
    print(f"      Sector {s.sector:>3d}  |  camera {s.camera}  chip {s.chip}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 ─ Build TargetData objects for every sector
# ══════════════════════════════════════════════════════════════════════════════
# TargetData is Eleanor's primary product: it combines a TESS
# Target Pixel File (TPF) *and* a systematics-corrected light curve
# in one object, extracted from the Full Frame Images (FFIs).
#
# FIX: Instead of trusting tpf_star_x/y (which returns the array centre
# (6,6) and is unreliable when TessCut is used as a fallback), we locate
# the star by finding the brightest pixel inside the CENTRAL 7×7 region
# of the TPF.  This avoids latching onto edge/corner artefacts or
# background contamination that sometimes appears brighter than the target.

print("\n[2] Building TargetData objects (TPF + light curve) …")
all_data = []
for star in stars:
    print(f"    Processing sector {star.sector} …")
    data = eleanor.TargetData(
        star,
        height=13,
        width=13,
        do_pca=True,
        do_psf=False,       # disabled — tf.logging removed in TensorFlow 2.x
        aperture_mode="normal",
    )

    # ── Locate the star in the central 7×7 region only ───────────────────
    # Using the full frame risks locking on to edge artefacts (as seen
    # previously when the brightest pixel was at corner position (1,11)).
    mean_frame = np.nanmean(data.tpf, axis=0)
    center_region = mean_frame[3:10, 3:10]          # inner 7×7 sub-array
    bright_y_local, bright_x_local = np.unravel_index(
        np.nanargmax(center_region), center_region.shape
    )
    bright_x = bright_x_local + 3                   # convert back to full TPF coords
    bright_y = bright_y_local + 3
    print(f"      Brightest pixel in central 7×7 : ({bright_x}, {bright_y})")

    data.custom_aperture(
        shape="circle",
        r=2.5,
        pos=(bright_x, bright_y),   # anchor on actual stellar flux peak
    )
    data.get_lightcurve()
    print(f"      Custom aperture pixels  : {np.sum(data.aperture > 0)}")

    all_data.append(data)
    print(f"      Cadences : {len(data.time)}")
    print(f"      Quality flags set : {np.sum(data.quality > 0)}")

first = all_data[0]   # use the first sector for single-sector demos


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 ─ Inspect the aperture
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Aperture information …")
print(f"    Aperture shape (pixels) : {first.aperture.shape}")
print(f"    Number of apertures tested : {len(first.all_apertures)}")
print(f"    Best aperture index : {first.best_ind}")
print(f"    Non-zero aperture pixels : {np.sum(first.aperture > 0)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 ─ Stitch all sectors into one continuous time series
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
# STEP 5 ─ Eleanor → Lightkurve via to_lightkurve(), then BLS on stitched data
# ══════════════════════════════════════════════════════════════════════════════
# Eleanor has a `to_lightkurve()` method that creates a
# lightkurve.lightcurve.TessLightCurve object from Eleanor's output
# — so the two tools can be used together, with Eleanor handling
# the FFI extraction and Lightkurve handling downstream analysis.
#
# For Panel D we use the single-sector to_lightkurve() object so
# the plot time axis matches Panel C.  For BLS and phase-folding we
# use the full stitched light curve for maximum transit S/N.

print("\n[5] Converting Eleanor TargetData  →  Lightkurve TessLightCurve …")
lc_eleanor = first.to_lightkurve()
print(f"    Type   : {type(lc_eleanor)}")
print(f"    Length : {len(lc_eleanor)}")
print(f"    Time format : {lc_eleanor.time.format}")
print(f"    Flux columns available : {lc_eleanor.colnames}")

# ── Single-sector light curve for Panel D ────────────────────────────────
lc_norm = lc_eleanor.remove_nans().normalize()

# FIX: Trim the first 50 cadences before flattening.
# The S-G filter was previously chasing the sector-start momentum-dump
# artefact, causing a spurious dip to ~0.94 at the beginning of Panel D.
lc_trimmed = lc_norm[50:]
lc_flat, trend = lc_trimmed.flatten(window_length=101, return_trend=True)

# Lomb-Scargle periodogram on single sector (diagnostic only)
print("\n    Running Lomb-Scargle periodogram (Lightkurve) …")
pg = lc_flat.to_periodogram(method="lombscargle",
                             minimum_period=0.5,
                             maximum_period=30)
best_period = pg.period_at_max_power
print(f"    Dominant period : {best_period:.4f}")

# ── Stitched light curve for BLS ─────────────────────────────────────────
# Using all available sectors gives more transits folded → better S/N.
lc_stitched = lk.LightCurve(
    time=time_stitched,
    flux=flux_stitched,
    flux_err=fluxerr_stitched,
)
lc_stitched = lc_stitched.remove_nans().normalize()

# FIX: Trim first 50 cadences of the stitched series for the same reason.
lc_stitched_trimmed = lc_stitched[50:]
lc_flat_stitched, trend_stitched = lc_stitched_trimmed.flatten(
    window_length=101, return_trend=True
)

# FIX: Sigma-clip outliers before BLS.
# A single bad cadence was previously folding on top of itself at phase 0,
# creating a spurious ~40% dip in Panel F.  4-sigma clipping removes it
# without affecting genuine transit signals (~1.2% depth for WASP-39b).
lc_clipped = lc_flat_stitched.remove_outliers(sigma=4.0)
print(f"    Cadences after sigma clipping : {len(lc_clipped)}")

print("    Running BLS transit search on stitched data (Lightkurve) …")
bls = lc_clipped.to_periodogram(
    method="bls",
    minimum_period=3.5,
    maximum_period=5.0,
    frequency_factor=150,
)
bls_period   = bls.period_at_max_power
bls_t0       = bls.transit_time_at_max_power
bls_duration = bls.duration_at_max_power
print(f"    BLS best period   : {bls_period.value:.4f} {bls_period.unit}")
print(f"    BLS transit time  : {bls_t0.value:.4f}")
print(f"    BLS transit dur.  : {bls_duration.value:.4f} {bls_duration.unit}")

# Phase-fold using the clipped stitched light curve
lc_folded = lc_clipped.fold(period=bls_period, epoch_time=bls_t0.value)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 ─ Custom aperture demo (Eleanor feature)
# ══════════════════════════════════════════════════════════════════════════════
# Eleanor lets you define circular or rectangular apertures of any size,
# centred anywhere on the TPF.  Here we demonstrate a default-centred
# circle purely as an API illustration; the science aperture was already
# set in Step 2 using the brightest-pixel position.

print("\n[6] Creating a custom circular aperture (r = 2.5 px) …")
first.custom_aperture(shape="circle", r=2.5)
first.get_lightcurve()
print(f"    Custom aperture pixels : {np.sum(first.aperture > 0)}")
print(f"    Re-extracted raw flux  : {len(first.raw_flux)} cadences")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 ─ Background subtraction (Eleanor feature)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Applying background subtraction …")
first.bkg_subtraction(scope="tpf", sigma=2.5)
print("    Background subtraction applied (scope='tpf', sigma=2.5)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 ─ Comprehensive figure
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] Generating summary figure …")

fig = plt.figure(figsize=(16, 14))
fig.suptitle(f"{STAR_NAME}b  —  Eleanor + Lightkurve Demo\n"
             f"(TIC {TIC_ID}  ·  Sector {first.source_info.sector})",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

# ── Panel A: Mean TPF image ────────────────────────────────────────────────
ax_tpf = fig.add_subplot(gs[0, 0])
mean_frame = np.nanmean(first.tpf, axis=0)
im = ax_tpf.imshow(mean_frame, origin="lower", cmap="YlOrRd",
                   interpolation="nearest")
plt.colorbar(im, ax=ax_tpf, label="Mean Flux (e⁻/s)")
ax_tpf.contour(first.aperture, levels=[0.5], colors="cyan", linewidths=1.5)
ax_tpf.set_title("A — Mean TPF + aperture (cyan)\n"
                 "Aperture centred on brightest pixel in central 7×7", fontsize=8)
ax_tpf.set_xlabel("Pixel column")
ax_tpf.set_ylabel("Pixel row")

# ── Panel B: Centroid motion ───────────────────────────────────────────────
ax_cen = fig.add_subplot(gs[0, 1])
ax_cen.scatter(first.centroid_xs, first.centroid_ys,
               c=first.time, cmap="plasma", s=4, alpha=0.6)
ax_cen.set_title("B — Centroid motion (colour = time)", fontsize=9)
ax_cen.set_xlabel("Centroid x (postcard px)")
ax_cen.set_ylabel("Centroid y (postcard px)")

# ── Panel C: Raw vs corrected flux (Eleanor) ──────────────────────────────
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
# psf_flux omitted — do_psf=False (TF2 incompatibility with eleanor 2.0.5)
ax_raw.set_title("C — Eleanor raw vs systematics-corrected flux", fontsize=9)
ax_raw.set_xlabel("Time (BTJD)")
ax_raw.set_ylabel("Normalised flux")
ax_raw.legend(fontsize=7, loc="upper right")

# ── Panel D: Lightkurve flattened + trend ─────────────────────────────────
# FIX: Plot lc_trimmed (first 50 cadences dropped) so the S-G trend no
# longer chases the momentum-dump artefact at the sector start.
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

# ── Panel E: BLS periodogram ──────────────────────────────────────────────
ax_bls = fig.add_subplot(gs[3, 0])
ax_bls.plot(bls.period.value, bls.power.value, "steelblue", lw=0.8)
ax_bls.axvline(bls_period.value, color="tomato", lw=1.5, ls="--",
               label=f"P = {bls_period.value:.3f} d")
ax_bls.axvline(4.0553, color="green", lw=1.0, ls=":",
               label="Known P = 4.055 d")
ax_bls.set_title("E — BLS periodogram  (stitched + sigma-clipped)", fontsize=9)
ax_bls.set_xlabel("Period (days)")
ax_bls.set_ylabel("BLS power")
ax_bls.legend(fontsize=7)

# ── Panel F: Phase-folded transit ─────────────────────────────────────────
ax_fold = fig.add_subplot(gs[3, 1])
ax_fold.scatter(lc_folded.time.value, lc_folded.flux.value,
                s=3, alpha=0.4, color="steelblue", label="Folded")
lc_binned = lc_folded.bin(time_bin_size=0.01)
ax_fold.plot(lc_binned.time.value, lc_binned.flux.value,
             "tomato", lw=1.5, label="Binned")
ax_fold.axvline(0, color="gray", lw=0.8, ls=":")
# Expected transit depth reference line for WASP-39b (~1.2%)
ax_fold.axhline(1.0 - 0.012, color="green", lw=0.8, ls="--",
                label="Expected depth ~1.2%")
ax_fold.set_title(f"F — Phase-folded transit  P = {bls_period.value:.3f} d\n"
                  "(stitched, sigma-clipped)", fontsize=8)
ax_fold.set_xlabel("Phase (days from mid-transit)")
ax_fold.set_ylabel("Normalised flux")
ax_fold.legend(fontsize=7)

plt.savefig("./wasp39b_eleanor_demo.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Figure saved → wasp39b_eleanor_demo.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 ─ Fold on KNOWN ephemeris (period + t0) to reveal the transit
# ══════════════════════════════════════════════════════════════════════════════
# The BLS recovered P = 4.106 d which is close but not exact.  Folding on
# the wrong period smears the transit across phase bins, suppressing the
# visible dip.  WASP-39b's published ephemeris from Faedi et al. (2011) /
# ExoFOP gives:
#   P   = 4.05527892 days
#   T0  = 2454773.6716 BJD_TDB  →  1373.1716 BTJD  (subtract 2457000.0)
#
# We use the stitched, sigma-clipped light curve built in Step 5 so the
# fold benefits from both sectors (~25 transits total).

print("\n[9] Folding on known WASP-39b ephemeris …")

KNOWN_PERIOD = 4.05527892          # days  (Faedi et al. 2011)
KNOWN_T0_BJD = 2454773.6716        # BJD_TDB
KNOWN_T0_BTJD = KNOWN_T0_BJD - 2457000.0   # convert to BTJD used by TESS

print(f"    Known period : {KNOWN_PERIOD} d")
print(f"    Known T0     : {KNOWN_T0_BTJD:.4f} BTJD")

# Fold the stitched sigma-clipped light curve on the known ephemeris
lc_folded_known = lc_clipped.fold(
    period=KNOWN_PERIOD,
    epoch_time=KNOWN_T0_BTJD,
)

# Zoom into ±0.3 days around mid-transit for the close-up panel
transit_half_window = 0.3   # days
mask_zoom = np.abs(lc_folded_known.time.value) < transit_half_window
lc_zoom = lc_folded_known[mask_zoom]

# Bin the full fold and the zoomed fold for clarity
lc_fold_binned  = lc_folded_known.bin(time_bin_size=0.02)
lc_zoom_binned  = lc_zoom.bin(time_bin_size=0.01)

print(f"    Cadences in zoom window (±{transit_half_window} d) : {mask_zoom.sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 ─ Large aperture re-extraction to reduce dilution
# ══════════════════════════════════════════════════════════════════════════════
# With aperture_mode='large' Eleanor tests apertures bigger than 8 pixels,
# capturing more stellar flux from WASP-39's extended PSF in the FFIs.
# This reduces the flux dilution that suppresses the transit depth.
# We rebuild TargetData for Sector 12 only (faster) as a comparison.

print("\n[10] Re-extracting Sector 12 with aperture_mode='large' …")

data_large = eleanor.TargetData(
    stars[0],               # Sector 12 Source object
    height=15,              # slightly larger cutout to accommodate big aperture
    width=15,
    do_pca=True,
    do_psf=False,
    aperture_mode="large",  # only considers apertures > 8 pixels
)

# Locate star in central region and apply centred aperture
mean_frame_lg = np.nanmean(data_large.tpf, axis=0)
cr_lg = mean_frame_lg[3:12, 3:12]
by_lg, bx_lg = np.unravel_index(np.nanargmax(cr_lg), cr_lg.shape)
data_large.custom_aperture(shape="circle", r=3.5,
                           pos=(bx_lg + 3, by_lg + 3))
data_large.get_lightcurve()

print(f"    Large aperture pixels : {np.sum(data_large.aperture > 0)}")
print(f"    Cadences              : {len(data_large.time)}")

# Build and process the large-aperture light curve
lc_large = lk.LightCurve(
    time=data_large.time[data_large.quality == 0],
    flux=data_large.corr_flux[data_large.quality == 0],
)
lc_large = lc_large.remove_nans().normalize()
lc_large_trimmed = lc_large[50:]
lc_large_flat, _ = lc_large_trimmed.flatten(window_length=101,
                                             return_trend=True)
lc_large_clipped = lc_large_flat.remove_outliers(sigma=4.0)

# Fold the large-aperture light curve on the known ephemeris
lc_large_folded = lc_large_clipped.fold(
    period=KNOWN_PERIOD,
    epoch_time=KNOWN_T0_BTJD,
)
lc_large_fold_binned = lc_large_folded.bin(time_bin_size=0.02)

print(f"    Large-aperture cadences after clipping : {len(lc_large_clipped)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 ─ Supplementary figure  (appended pages / new PNG)
# ══════════════════════════════════════════════════════════════════════════════
# Four-panel figure:
#   G  Full phase fold on known ephemeris  (normal aperture)
#   H  Zoomed transit window               (normal aperture)
#   I  Large-aperture TPF comparison
#   J  Large-aperture phase fold on known ephemeris

print("\n[11] Generating supplementary transit figure …")

fig2 = plt.figure(figsize=(16, 12))
fig2.suptitle(f"{STAR_NAME}b  —  Transit Detection Supplement\n"
              f"(TIC {TIC_ID}  ·  Known P = {KNOWN_PERIOD} d)",
              fontsize=14, fontweight="bold", y=0.98)

gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.50, wspace=0.35)

# ── Panel G: Full phase fold on known ephemeris ───────────────────────────
ax_g = fig2.add_subplot(gs2[0, 0])
ax_g.scatter(lc_folded_known.time.value, lc_folded_known.flux.value,
             s=2, alpha=0.25, color="steelblue", label="Folded (stitched)")
ax_g.plot(lc_fold_binned.time.value, lc_fold_binned.flux.value,
          "tomato", lw=2.0, label="Binned (20 min)")
ax_g.axvline(0, color="gray", lw=0.8, ls=":")
ax_g.axhline(1.0 - 0.012, color="green", lw=1.0, ls="--",
             label="Expected depth ~1.2%")
ax_g.set_xlim(-2.5, 2.5)
ax_g.set_title("G — Full phase fold on KNOWN ephemeris\n"
               f"P = {KNOWN_PERIOD} d  ·  normal aperture  ·  stitched",
               fontsize=8)
ax_g.set_xlabel("Phase (days from mid-transit)")
ax_g.set_ylabel("Normalised flux")
ax_g.legend(fontsize=7)

# ── Panel H: Zoomed transit window ────────────────────────────────────────
ax_h = fig2.add_subplot(gs2[0, 1])
ax_h.scatter(lc_zoom.time.value, lc_zoom.flux.value,
             s=8, alpha=0.4, color="steelblue", label="Folded (±0.3 d)")
ax_h.plot(lc_zoom_binned.time.value, lc_zoom_binned.flux.value,
          "tomato", lw=2.5, label="Binned (10 min)")
ax_h.axvline(0, color="gray", lw=0.8, ls=":", label="Mid-transit")
ax_h.axhline(1.0 - 0.012, color="green", lw=1.0, ls="--",
             label="Expected depth ~1.2%")
# Mark transit ingress / egress (duration ~0.113 d from literature)
half_dur = 0.113 / 2
ax_h.axvline(-half_dur, color="purple", lw=0.9, ls="--", alpha=0.7,
             label=f"Ingress/egress (±{half_dur:.3f} d)")
ax_h.axvline(+half_dur, color="purple", lw=0.9, ls="--", alpha=0.7)
ax_h.set_title("H — Zoomed transit window  (±0.3 d)\n"
               "Purple dashes = expected ingress / egress",
               fontsize=8)
ax_h.set_xlabel("Phase (days from mid-transit)")
ax_h.set_ylabel("Normalised flux")
ax_h.legend(fontsize=7)

# ── Panel I: Large-aperture mean TPF ──────────────────────────────────────
ax_i = fig2.add_subplot(gs2[1, 0])
mean_frame_lg2 = np.nanmean(data_large.tpf, axis=0)
im2 = ax_i.imshow(mean_frame_lg2, origin="lower", cmap="YlOrRd",
                  interpolation="nearest")
plt.colorbar(im2, ax=ax_i, label="Mean Flux (e⁻/s)")
ax_i.contour(data_large.aperture, levels=[0.5], colors="cyan", linewidths=1.5)
ax_i.set_title("I — Large aperture TPF  (aperture_mode='large', r=3.5 px)\n"
               "More stellar flux captured → less transit dilution",
               fontsize=8)
ax_i.set_xlabel("Pixel column")
ax_i.set_ylabel("Pixel row")

# ── Panel J: Large-aperture phase fold on known ephemeris ─────────────────
ax_j = fig2.add_subplot(gs2[1, 1])
ax_j.scatter(lc_large_folded.time.value, lc_large_folded.flux.value,
             s=2, alpha=0.25, color="darkorange", label="Folded (large ap.)")
ax_j.plot(lc_large_fold_binned.time.value, lc_large_fold_binned.flux.value,
          "tomato", lw=2.0, label="Binned (20 min)")
ax_j.axvline(0, color="gray", lw=0.8, ls=":")
ax_j.axhline(1.0 - 0.012, color="green", lw=1.0, ls="--",
             label="Expected depth ~1.2%")
ax_j.axvline(-half_dur, color="purple", lw=0.9, ls="--", alpha=0.7)
ax_j.axvline(+half_dur, color="purple", lw=0.9, ls="--", alpha=0.7)
ax_j.set_xlim(-0.3, 0.3)
ax_j.set_title("J — Large-aperture phase fold on KNOWN ephemeris\n"
               f"Sector 12 only  ·  aperture_mode='large'",
               fontsize=8)
ax_j.set_xlabel("Phase (days from mid-transit)")
ax_j.set_ylabel("Normalised flux")
ax_j.legend(fontsize=7)

plt.savefig("./wasp39b_transit_supplement.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Supplementary figure saved → wasp39b_transit_supplement.png")



print("\n" + "=" * 60)
print("  All done!  Summary of Eleanor data products accessed:")
print("    • eleanor.multi_sectors()       — sector discovery")
print("    • eleanor.TargetData            — TPF + raw/corr/PCA flux")
print("    • data.aperture                 — brightest-pixel custom aperture")
print("    • data.centroid_xs / _ys        — pointing-model centroid trace")
print("    • data.custom_aperture()        — user-defined aperture")
print("    • data.bkg_subtraction()        — background removal")
print("    • data.stitch()                 — multi-sector stitching")
print("    • data.to_lightkurve()          — Eleanor → Lightkurve bridge")
print("    • aperture_mode='large'         — large aperture re-extraction")
print("  Lightkurve operations on Eleanor data:")
print("    • lc.normalize() / .flatten()   — detrending")
print("    • lc.remove_outliers(sigma=4.0) — sigma clipping")
print("    • lc.to_periodogram('bls')      — transit search")
print("    • lc.fold(known_period, known_t0) — known-ephemeris phase fold")
print("=" * 60)
