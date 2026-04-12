"""
HD 209458 b JWST Transmission Spectrum — Download & Plot
=========================================================
Paper:
  Xue, Q., Bean, J., Zhang, M., Welbanks, L., Lunine, J. & August, P.
  "JWST Transmission Spectroscopy of HD 209458b: A Supersolar Metallicity,
   a Very Low C/O, and No Evidence of CH4, HCN, or C2H2"
  The Astrophysical Journal Letters, 963, L5 (2024)
  arXiv:  https://arxiv.org/abs/2310.03245
  PDF:    https://arxiv.org/pdf/2310.03245
  DOI:    https://doi.org/10.3847/2041-8213/ad2682

Data:
  Zenodo record 10557924 (v2) — spectra_final.csv
  DOI: https://doi.org/10.5281/zenodo.10557924

Planet summary:
  HD 209458 b — hot Jupiter, Teq ~1450 K, ~0.69 MJ, ~1.36 RJ
  Host star: G0V, V=7.65, d=47 pc
  First transiting exoplanet (Charbonneau+2000, Henry+2000)
  First exoplanet atmospheric detection (Charbonneau+2002, Na)
  JWST detections: H₂O, CO₂, CO (Xue+2024)
  Retrieved: [M/H] = 3×solar, C/O = 0.11 (very sub-solar)

Coverage: 2.3–5.1 µm  (NIRCam F322W2 + F444W)

Requirements:
    pip install astropy matplotlib numpy requests pandas
"""

import warnings
from pathlib import Path

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

OUT_DIR = Path("hd209458b_data")
OUT_DIR.mkdir(exist_ok=True)

# ── Dataset definition ────────────────────────────────────────────────────────
# Zenodo 10557924 (v2, Xue et al. 2024) contains spectra.csv with columns:
#   wavelength (µm), depth (ppm or %), depth_err  — Eureka! + SPARTA reductions
#
# Column name variants seen in Eureka!/SPARTA CSV outputs:
#   wavelength | wave | lambda
#   depth | transit_depth | dppm | (rp/rs)^2
#   depth_err | err | error | e_depth | uncertainty
#
# The read_csv function below tries all of these automatically.
# If column detection fails at runtime, the script prints what IS available
# and switches to demo mode.

DATASET = {
    "record":     "10557924",
    "label":      "NIRCam F322W2+F444W  (Xue et al. 2024)",
    "color":      "#1a5fa8",
    "instrument": "NIRCam",
    # The Zenodo ZIP/file contains spectra.csv — pick first CSV in the archive
    "fmt":        "csv",
}

# Molecular absorption bands in the 2.3–5.1 µm NIRCam window.
# Positions based on Xue+2024 Fig.1, HITRAN cross-sections, and
# the data features visible in the plotted spectrum.
# Each tuple: (wave_start, wave_end, label, colour, show_label)
MOLECULE_BANDS = [
    (2.62, 2.87, "H₂O",     "#0288d1", True),    # H2O 2.7 µm band
    (4.15, 4.55, "CO₂",     "#b8860b", True),    # CO2 4.3 µm band (strong)
    (4.58, 4.98, "CO",      "#c0392b", True),    # CO  4.7 µm band
]

# Y-axis: tight around the actual data range (1.441–1.477 %)
DEPTH_YMIN = 1.436   # %
DEPTH_YMAX = 1.483   # %

# ── Download helpers ──────────────────────────────────────────────────────────

def stream_download(url, dest):
    dest = Path(dest)
    if dest.exists():
        print(f"  cached   {dest.name}")
        return True
    print(f"  fetching {dest.name} ...")
    try:
        with requests.get(url, stream=True, timeout=120,
                          headers={"User-Agent": "hd209458b-plot/1.0"}) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
        print(f"  saved  ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def zenodo_download_url(record_id):
    r = requests.get(
        f"https://zenodo.org/api/records/{record_id}",
        timeout=20,
        headers={"User-Agent": "hd209458b-plot/1.0"},
    )
    r.raise_for_status()
    files = r.json().get("files", [])
    if not files:
        raise RuntimeError(f"No files in Zenodo record {record_id}")
    # Prefer the CSV file if multiple files are present
    for f in files:
        if f["key"].lower().endswith(".csv"):
            return f["links"]["self"], f["key"]
    # Fall back to first file
    return files[0]["links"]["self"], files[0]["key"]


# ── Unit normalisation ────────────────────────────────────────────────────────

def _to_ppm(wave, depth, err):
    """Normalise units to ppm, sort by wavelength, drop non-finite rows."""
    wave  = np.array(wave,  dtype=float)
    depth = np.array(depth, dtype=float)
    err   = np.array(err,   dtype=float)
    mask = np.isfinite(wave) & np.isfinite(depth)
    wave, depth, err = wave[mask], depth[mask], err[mask]
    if len(wave) == 0:
        raise ValueError("No finite data points after masking")
    med = np.nanmedian(depth)
    if 0.5 < med < 10.0:      # % → ppm
        depth *= 1e4; err *= 1e4
    elif med < 0.5:            # fractional (Rp/Rs)² → ppm
        depth *= 1e6; err *= 1e6
    # already ppm if med ~ thousands
    idx = np.argsort(wave)
    return wave[idx], depth[idx], err[idx]


# ── CSV reader ────────────────────────────────────────────────────────────────

# Column name aliases tried in order (case-insensitive).
_WAVE_KEYS  = ["wavelength", "wave", "lambda", "central_wavelength",
               "wave_1d", "wavebin", "bin_wave", "wave_um"]
_BESTFIT_KEYS = ["bestfit", "best_fit", "model", "fit"]
_WAVE_START = ["start", "wave_start", "wl_low", "wavestart", "bin_start"]
_WAVE_END   = ["end",   "wave_end",   "wl_high", "waveend",   "bin_end"]
_DEPTH_KEYS = ["tr_depth", "depth", "transit_depth", "dppm", "fp_fstar",
               "(rp/rs)^2", "rp_rs_2", "rpsrs2", "delta", "td"]
_ERR_KEYS   = ["tr_depth_err", "depth_err", "err", "error", "transit_depth_err",
               "transit_depth_uncertainty", "e_depth", "sigma",
               "uncertainty", "fp_fstar_err", "depth_error",
               "e_transit_depth", "dppm_err"]


def _parse_with_pandas(path):
    """
    Handle the Xue+2024 spectra_final.csv dual-pipeline layout:

        Eureka, , , , Sparta, , ,
        wavelength, depth, depth_err, ..., wavelength, depth, depth_err, ...
        <data rows>

    Astropy reads row-0 as the header (giving 'Eureka', 'col1', '_1' …).
    We instead use pandas with header=1, which treats row-1 as column names.
    We then pick the Eureka! reduction columns (first wavelength/depth/err
    triplet) and return them.  Falls back to header=0 if row-1 is also
    non-numeric.
    """
    import pandas as pd
    import io

    raw = Path(path).read_text()
    print(f"  Raw first 3 lines of CSV:")
    for i, line in enumerate(raw.splitlines()[:3]):
        print(f"    [{i}] {line[:120]}")

    # Strategy A: row 1 is the real header (dual-pipeline layout)
    try:
        df = pd.read_csv(io.StringIO(raw), header=1, comment="#")
        df.columns = [str(c).strip().strip('"').lower() for c in df.columns]
        print(f"  Columns (header=1): {list(df.columns)}")

        # pandas adds '.1', '.2' suffixes to duplicate column names.
        # Use exact match (or name + '.N') so 'depth' won't grab 'depth_err'.
        import re
        def first_col(keys):
            for k in keys:
                pat = re.compile(r'^' + re.escape(k.lower()) + r'(\.\d+)?$')
                for col in df.columns:
                    if pat.match(col):
                        arr = pd.to_numeric(df[col], errors="coerce").values
                        if np.isfinite(arr).sum() > 3:
                            return arr
            return None

        wave  = first_col(_WAVE_KEYS)
        # Bin-edge fallback: compute centre from (start + end) / 2
        if wave is None:
            w0 = first_col(_WAVE_START)
            w1 = first_col(_WAVE_END)
            if w0 is not None and w1 is not None:
                wave = (w0 + w1) / 2.0
                print("  wavelength ← (start + end) / 2")
        depth   = first_col(_DEPTH_KEYS)
        err     = first_col(_ERR_KEYS)
        bestfit = first_col(_BESTFIT_KEYS)
        if wave is not None and depth is not None:
            print(f"  Parsed using header=1 (dual-pipeline layout)")
            print(f"    wave  range : {wave.min():.3f} – {wave.max():.3f} µm")
            print(f"    depth range : {depth.min():.5f} – {depth.max():.5f}")
            print(f"    err   median: {np.nanmedian(err):.5f}" if err is not None else "    err: none")
            if bestfit is not None: print(f"    bestfit     : found")
            return wave, depth, err, bestfit
    except Exception as e:
        print(f"  header=1 attempt failed: {e}")

    # Strategy B: standard single-header CSV
    try:
        df = pd.read_csv(io.StringIO(raw), header=0, comment="#")
        df.columns = [str(c).strip().strip('"').lower() for c in df.columns]
        print(f"  Columns (header=0): {list(df.columns)}")

        def find(keys):
            for k in keys:
                if k.lower() in df.columns:
                    arr = pd.to_numeric(df[k.lower()], errors="coerce").values
                    if np.isfinite(arr).sum() > 5:
                        return arr
            return None

        wave  = find(_WAVE_KEYS)
        if wave is None:
            w0 = find(_WAVE_START)
            w1 = find(_WAVE_END)
            if w0 is not None and w1 is not None:
                wave = (w0 + w1) / 2.0
                print("  wavelength ← (start + end) / 2")
        depth = find(_DEPTH_KEYS)
        err   = find(_ERR_KEYS)
        bestfit = find(_BESTFIT_KEYS)
        if wave is not None and depth is not None:
            print("  Parsed using header=0 (standard layout)")
            return wave, depth, err, bestfit
    except Exception as e:
        print(f"  header=0 attempt failed: {e}")

    return None, None, None, None


def read_csv(path):
    """
    Read a transmission-spectrum CSV, handling both standard and
    dual-pipeline (Eureka! / SPARTA side-by-side) layouts.
    """
    wave, depth, err, bestfit = _parse_with_pandas(path)

    if wave is None:
        raise ValueError(
            "Could not find wavelength data in CSV. "
            "Check the printed column names above and add the correct "
            "name to _WAVE_KEYS at the top of the script."
        )
    if depth is None:
        raise ValueError(
            "Could not find depth data in CSV. "
            "Check the printed column names and add to _DEPTH_KEYS."
        )
    if err is None:
        print("  Warning: no error column found — using zeros")
        err = np.zeros_like(depth)

    wave, depth, err = _to_ppm(
        np.asarray(wave, dtype=float).ravel(),
        np.asarray(depth, dtype=float).ravel(),
        np.asarray(err, dtype=float).ravel(),
    )
    # Normalise bestfit to ppm using same sort order
    if bestfit is not None:
        bf  = np.array(bestfit, dtype=float).ravel()   # copy → writable
        med = np.nanmedian(bf[np.isfinite(bf)])
        if 0.5 < med < 10.0:  bf *= 1e4
        elif med < 0.5:        bf *= 1e6
        idx     = np.argsort(np.array(wave, dtype=float))
        bestfit = bf[idx]
    return wave, depth, err, bestfit


# ── Fetch dataset ─────────────────────────────────────────────────────────────

def fetch_dataset(ds):
    print(f"\n-- {ds['label']} --")
    try:
        dl_url, fname = zenodo_download_url(ds["record"])
    except Exception as e:
        print(f"  Zenodo error: {e}")
        return None

    dest = OUT_DIR / f"NIRCam_{fname}"
    if not stream_download(dl_url, dest):
        return None

    try:
        wave, depth, err, bestfit = read_csv(dest)
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return None

    print(
        f"  OK: {len(wave)} pts | "
        f"{wave.min():.3f}–{wave.max():.3f} µm | "
        f"median {np.median(depth):.0f} ppm"
    )
    return wave, depth, err, bestfit


# ── Demo fallback ─────────────────────────────────────────────────────────────

def make_demo_data():
    """
    Physically motivated synthetic spectrum for HD 209458b.
    Based on Xue et al. 2024 Fig. 1 and retrieved molecular abundances.
    NIRCam F322W2 (2.35–4.00 µm) + F444W (3.90–5.10 µm).
    """
    rng  = np.random.default_rng(7)
    wave = np.concatenate([
        np.linspace(2.35, 4.00, 140),   # F322W2
        np.linspace(4.01, 5.10,  90),   # F444W
    ])

    # Baseline transit depth in ppm  (Rp/Rs = 0.1207 → ~14570 ppm)
    base = np.full_like(wave, 14570.0)

    def gauss(w, c, s, a):
        return a * np.exp(-0.5 * ((w - c) / s) ** 2)

    # H₂O features
    base += gauss(wave, 2.70, 0.14, 220)   # 2.7 µm band
    base += gauss(wave, 3.80, 0.10,  80)   # 3.8 µm shoulder

    # CO₂ feature (strong, ~4.3 µm) — key detection in Xue+2024
    base += gauss(wave, 4.30, 0.13, 350)

    # CO feature (4.7 µm)
    base += gauss(wave, 4.72, 0.11, 180)

    # Slight Rayleigh-like slope at blue end
    base -= 60 * (1.0 / wave - 1.0 / 3.5)

    # Realistic per-point uncertainties (~100–160 ppm, larger at red end)
    sigma = 110 + 50 * ((wave - 2.35) / (5.10 - 2.35))
    err   = np.abs(sigma + rng.normal(0, sigma * 0.12, len(wave)))
    depth = np.clip(base + rng.normal(0, sigma * 0.9, len(wave)), 14_200, 15_100)

    return wave, depth, err, None


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_spectrum(result, demo=False):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#f5f5f0")
    ax.set_facecolor("#fafaf7")

    y_lo, y_hi = DEPTH_YMIN, DEPTH_YMAX
    label_y_hi  = y_hi - 0.0018   # top labels (H2O — left side, clear of legend)
    label_y_lo  = y_lo + 0.0022   # bottom labels (CO2, CO — right side)

    # ── Molecule band shading + labels ───────────────────────────────────
    # H2O label goes at top (left half of plot, no legend clash)
    # CO2 / CO labels go at bottom (right half covered by legend at top)
    bottom_mols = {"CO₂", "CO"}
    for w0, w1, mol, col, show_label in MOLECULE_BANDS:
        ax.axvspan(w0, w1, color=col, alpha=0.22, lw=0, zorder=0)
        if show_label:
            mid = (w0 + w1) / 2
            if mol in bottom_mols:
                ax.text(mid, label_y_lo, mol,
                        color=col, fontsize=8.0, ha="center", va="bottom",
                        alpha=0.95, fontfamily="monospace", fontweight="bold")
            else:
                ax.text(mid, label_y_hi, mol,
                        color=col, fontsize=8.0, ha="center", va="top",
                        alpha=0.95, fontfamily="monospace", fontweight="bold")

    # ── Filter gap: subtle stripe only, no cluttering text ───────────────
    ax.axvspan(3.97, 4.02, color="#333333", alpha=0.20, lw=0, zorder=0)

    # ── Data ─────────────────────────────────────────────────────────────
    if result is not None:
        wave, depth, err, bestfit = result
        ax.errorbar(
            wave, depth / 1e4, yerr=err / 1e4,
            fmt="o", ms=2.5, lw=0, elinewidth=0.65,
            color=DATASET["color"], ecolor=DATASET["color"],
            alpha=0.88, label=DATASET["label"], zorder=3,
        )
        if bestfit is not None:
            ax.plot(wave, bestfit / 1e4, color="#c0392b", lw=1.4,
                    alpha=0.75, zorder=4, label="Best-fit model (Eureka!)")
    else:
        ax.text(0.5, 0.5, "No data loaded.\nTry --demo for offline mode.",
                transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=11)

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_xlim(2.28, 5.08)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Wavelength (µm)", color="#111111", fontsize=12)
    ax.set_ylabel("Transit Depth (%)", color="#111111", fontsize=12)
    ax.tick_params(colors="#222222", labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor("#999")

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.grid(which="major", color="#dddddd", lw=0.5, zorder=1)
    ax.grid(which="minor", color="#eeeeee", lw=0.3, zorder=1)

    ax.legend(loc="upper right", fontsize=9,
              facecolor="#ffffff", edgecolor="#aaaaaa",
              labelcolor="#111111", framealpha=0.95)

    suffix = " [DEMO]" if demo else ""
    ax.set_title(
        f"HD 209458 b — JWST NIRCam Transmission Spectrum{suffix}\n"
        "(Xue, Bean, Zhang, Welbanks, Lunine & August 2024, ApJL 963 L5)",
        color="#111111", fontsize=11, pad=10,
    )

    # ── Reference annotation (bottom-left, outside axes) ────────────────
    ref_text = (
        "Xue et al. 2024, ApJL 963 L5  ·  arXiv:2310.03245  ·  "
        "doi:10.3847/2041-8213/ad2682\n"
        "Data: doi:10.5281/zenodo.10557924"
    )
    fig.text(0.01, 0.005, ref_text,
             fontsize=6.5, color="#666666",
             ha="left", va="bottom", fontfamily="monospace")

    plt.tight_layout()
    # Leave room at bottom for the reference line
    plt.subplots_adjust(bottom=0.10)
    out = OUT_DIR / "hd209458b_jwst_spectrum.png"
    plt.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nFigure saved → {out}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

VERSION = "1.6.0"

if __name__ == "__main__":
    import sys

    demo = "--demo" in sys.argv

    print("=" * 60)
    print(f"  HD 209458 b  —  JWST NIRCam Transmission Spectrum  v{VERSION}")
    print("  Xue et al. 2024  |  Zenodo 10557924")
    print("=" * 60)

    if demo:
        print("\n[DEMO MODE — synthetic data]")
        result = make_demo_data()
        plot_spectrum(result, demo=True)
    else:
        result = fetch_dataset(DATASET)
        if result is None:
            print("\nDownload failed — switching to demo mode.")
            result = make_demo_data()
            plot_spectrum(result, demo=True)
        else:
            plot_spectrum(result)

    print("Done.")
