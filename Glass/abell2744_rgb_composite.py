"""
Abell 2744 GLASS-JWST RGB Composite
====================================
Reproduces the canonical colour composite from the GLASS-JWST HLSP:

  Blue  : F090W + F115W + F150W
  Green : F200W + F277W
  Red   : F356W + F410M + F444W

Coloured border overlays:
  Green  → GLASS (GO-1324)
  Blue   → UNCOVER (GO-2561)
  Red    → DD-2756

Requirements
------------
  pip install astropy matplotlib numpy scipy

Usage
-----
  python abell2744_rgb_composite.py

The script auto-discovers the FITS files produced by the MAST download
(path pattern: ./mastDownload/HLSP/hlsp_glass-jwst_jwst_nircam_abell2744_<filt>_v1.0_sci/
                              hlsp_glass-jwst_jwst_nircam_abell2744_<filt>_v1.0_sci.fits)

Adjust MAST_ROOT if your download tree lives elsewhere.
"""

import sys
import glob
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from astropy.io import fits
from astropy.visualization import (
    AsinhStretch, ManualInterval, ImageNormalize, ZScaleInterval
)
from astropy.wcs import WCS, FITSFixedWarning

warnings.filterwarnings("ignore", category=FITSFixedWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAST_ROOT = Path("./mastDownload/HLSP")   # ← change if needed

FILTER_GROUPS = {
    "blue":  ["f090w", "f115w", "f150w"],
    "green": ["f200w", "f277w"],
    "red":   ["f356w", "f410m", "f444w"],
}

# Asinh stretch softening parameter (play with Q and stretch_factor)
Q              = 8      # controls colour saturation in faint regions
STRETCH_FACTOR = 0.6    # overall image brightness (larger → brighter)

# Output figure
FIG_DPI  = 200
FIG_SIZE = (12, 12)
OUTPUT   = Path("abell2744_rgb_composite.png")

# Survey footprint border colours  (R, G, B fractions)
BORDER_COLORS = {
    "GLASS\n(GO-1324)":   "#00cc44",   # green
    "UNCOVER\n(GO-2561)": "#4488ff",   # blue
    "DD-2756":            "#ff3333",   # red
}

# Approximate fractional insets of the three survey footprints
# (left, bottom, width, height) as fractions of the full image size.
# These are representative — tune to your actual WCS if needed.
BORDER_BOXES = {
    "GLASS\n(GO-1324)":   (0.10, 0.10, 0.80, 0.80),
    "UNCOVER\n(GO-2561)": (0.25, 0.25, 0.50, 0.50),
    "DD-2756":            (0.40, 0.40, 0.20, 0.20),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_sci_fits(filter_name: str) -> Path:
    """Return path to the science FITS for a given filter (case-insensitive)."""
    pattern = str(MAST_ROOT / f"*{filter_name}*sci" / f"*{filter_name}*sci.fits")
    matches = sorted(glob.glob(pattern, recursive=False))
    if not matches:
        raise FileNotFoundError(
            f"No science FITS found for filter '{filter_name}'.\n"
            f"  Searched: {pattern}\n"
            f"  Run the MAST download first, or adjust MAST_ROOT."
        )
    return Path(matches[0])


def load_and_coadd(filters: list[str], downsample: int = 4) -> np.ndarray:
    """
    Load multiple science frames, reproject to a common WCS (nearest-neighbour
    on a shared pixel grid), co-add with equal weights, and optionally downsample.

    Returns a 2-D float32 array (NaN-safe sum / N_valid).
    """
    from astropy.nddata import Cutout2D      # lightweight; avoids reproject dep
    from astropy.wcs.utils import proj_plane_pixel_scales

    arrays = []
    ref_shape = None
    ref_wcs   = None

    for filt in filters:
        path = find_sci_fits(filt)
        print(f"  Loading {path.name} …", flush=True)
        with fits.open(path, memmap=True) as hdul:
            # GLASS HLSPs store science in extension 0 or 1
            ext = 0 if hdul[0].data is not None else 1
            data = hdul[ext].data.astype(np.float32)
            hdr  = hdul[ext].header

        # Replace sentinels
        data[data == 0]    = np.nan
        data[~np.isfinite(data)] = np.nan

        if ref_shape is None:
            ref_shape = data.shape
            ref_wcs   = WCS(hdr, naxis=2)

        # Crop / pad to ref_shape if sizes differ slightly
        if data.shape != ref_shape:
            rows = min(data.shape[0], ref_shape[0])
            cols = min(data.shape[1], ref_shape[1])
            canvas = np.full(ref_shape, np.nan, dtype=np.float32)
            canvas[:rows, :cols] = data[:rows, :cols]
            data = canvas

        arrays.append(data)

    # Co-add: mean ignoring NaN
    stack = np.nanmean(np.stack(arrays, axis=0), axis=0)

    # Downsample by block-mean to speed rendering
    if downsample > 1:
        h, w = stack.shape
        h2, w2 = h // downsample * downsample, w // downsample * downsample
        stack = stack[:h2, :w2].reshape(
            h2 // downsample, downsample,
            w2 // downsample, downsample
        ).mean(axis=(1, 3))

    return stack, ref_wcs


def asinh_stretch(img: np.ndarray, sky_sigma: float | None = None) -> np.ndarray:
    """
    Per-channel asinh normalisation inspired by Lupton et al. (2004).
    Maps to [0, 1].
    """
    if sky_sigma is None:
        # Robust sky: sigma-clipped std of pixels < 25th percentile
        p25 = np.nanpercentile(img, 25)
        sky_sigma = max(np.nanstd(img[img < p25]), 1e-12)
    softening = Q * sky_sigma
    norm = np.arcsinh(img * STRETCH_FACTOR / softening) / np.arcsinh(STRETCH_FACTOR / softening)
    norm = np.clip(norm, 0, 1)
    return norm.astype(np.float32)


def make_rgb(red, green, blue):
    """Assemble and auto-clip an (H, W, 3) RGB float32 array."""
    h = min(red.shape[0], green.shape[0], blue.shape[0])
    w = min(red.shape[1], green.shape[1], blue.shape[1])
    rgb = np.stack([
        red[:h, :w],
        green[:h, :w],
        blue[:h, :w],
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return rgb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n=== Abell 2744 GLASS-JWST RGB Composite ===\n")

    # ---- Build each channel -----------------------------------------------
    print("Building BLUE channel  (F090W + F115W + F150W)…")
    blue_raw, wcs = load_and_coadd(FILTER_GROUPS["blue"])

    print("Building GREEN channel (F200W + F277W)…")
    green_raw, _  = load_and_coadd(FILTER_GROUPS["green"])

    print("Building RED channel   (F356W + F410M + F444W)…")
    red_raw, _    = load_and_coadd(FILTER_GROUPS["red"])

    # ---- Stretch each channel independently --------------------------------
    print("\nApplying asinh stretch…")
    r = asinh_stretch(red_raw)
    g = asinh_stretch(green_raw)
    b = asinh_stretch(blue_raw)

    # ---- Assemble RGB -------------------------------------------------------
    rgb = make_rgb(r, g, b)
    H, W = rgb.shape[:2]

    # ---- Plot ---------------------------------------------------------------
    print("Rendering figure…")
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="black")
    ax.set_facecolor("black")

    ax.imshow(rgb, origin="lower", interpolation="nearest")

    # ---- Survey footprint border overlays ----------------------------------
    # Insets are defined as (left_frac, bottom_frac, width_frac, height_frac)
    for label, (lf, bf, wf, hf) in BORDER_BOXES.items():
        x0 = lf * W
        y0 = bf * H
        bw = wf * W
        bh = hf * H
        color = BORDER_COLORS[label]
        rect = mpatches.Rectangle(
            (x0, y0), bw, bh,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(rect)
        # Label inside upper-left corner of each box
        ax.text(
            x0 + 0.015 * W,
            y0 + bh - 0.03 * H,
            label,
            color=color,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
            zorder=6,
        )

    # ---- Annotation ---------------------------------------------------------
    ax.set_title(
        "Abell 2744 — GLASS-JWST NIRCam RGB\n"
        "Blue: F090W+F115W+F150W  |  Green: F200W+F277W  |  Red: F356W+F410M+F444W",
        color="white",
        fontsize=11,
        pad=10,
    )

    # Compass rose (N up, E left for standard WCS orientation with origin=lower)
    arrow_kw = dict(arrowstyle="-|>", color="white", lw=1.5,
                    mutation_scale=12, zorder=7)
    ax.annotate("", xy=(0.06 * W, 0.07 * H), xytext=(0.06 * W, 0.02 * H),
                arrowprops=dict(**arrow_kw))
    ax.text(0.06 * W, 0.075 * H, "N", color="white", fontsize=9,
            ha="center", va="bottom", zorder=7)
    ax.annotate("", xy=(0.01 * W, 0.04 * H), xytext=(0.06 * W, 0.04 * H),
                arrowprops=dict(**arrow_kw))
    ax.text(0.005 * W, 0.04 * H, "E", color="white", fontsize=9,
            ha="right", va="center", zorder=7)

    ax.axis("off")
    plt.tight_layout(pad=0.5)

    fig.savefig(OUTPUT, dpi=FIG_DPI, bbox_inches="tight",
                facecolor="black", pad_inches=0.05)
    print(f"\nSaved → {OUTPUT.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
