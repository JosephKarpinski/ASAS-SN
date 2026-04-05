"""
WASP-39 b JWST Transmission Spectra — Download & Plot
======================================================
Data sources (all public, CC-BY):
  NIRISS SOSS    — Feinstein et al. 2023  → GitHub (CSV)
  NIRCam F322W2  — Ahrer et al. 2023      → zenodo 7101283 (ZIP of HDF5)
  NIRSpec G395H  — Alderson et al. 2023   → zenodo 7185300 (ZIP of NetCDF)
  NIRSpec PRISM  — Rustamkulov et al. 2023→ zenodo 7388032 (ZIP of HDF5)

Requirements:
    pip install astropy matplotlib numpy requests h5py netCDF4
"""

import io, zipfile, tempfile, warnings
from pathlib import Path
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

OUT_DIR = Path("wasp39b_data")
OUT_DIR.mkdir(exist_ok=True)

# ── Dataset definitions ───────────────────────────────────────────────────────
# zip_pick: exact substring that uniquely identifies the target member.
#           The FIRST member whose lowercased path contains ALL substrings wins.

DATASETS = [
    {
        "url":        "https://raw.githubusercontent.com/afeinstein20/wasp39b_niriss_paper/main/data/ts/CMADF-WASP_39b_NIRISS_transmission_spectrum_R300.csv",
        "label":      "NIRISS SOSS (0.6-2.8 um)",
        "color":      "#4477AA",
        "instrument": "NIRISS",
        # CSV columns confirmed from paper scripts (figure3.py, edfigure6.py)
        "col_wave":   "wave",
        "col_depth":  "dppm",
        "col_err":    "dppm_err",
    },
    {
        "record":     "7101283",
        "label":      "NIRCam F322W2 (2.4-4.0 um)",
        "color":      "#EE6677",
        "instrument": "NIRCam",
        # From ZIP listing: 2_TRANSMISSION_SPECTRA/LW-transit-spectrum-*-eureka.h5
        "zip_pick":   ["2_transmission_spectra", "lw-transit", "eureka"],
        "fmt":        "h5",
    },
    {
        "record":     "7185300",
        "label":      "NIRSpec G395H (2.8-5.2 um)",
        "color":      "#228833",
        "instrument": "NIRSpec_G395H",
        # From ZIP listing: 3_TRANSMISSION_SPECTRA/transit-spectrum-*-weighted-average.nc
        "zip_pick":   ["3_transmission_spectra", "weighted-average"],
        "fmt":        "nc",
    },
    {
        "record":     "7388032",
        "label":      "NIRSpec PRISM (0.5-5.5 um)",
        "color":      "#CCBB44",
        "instrument": "NIRSpec_PRISM",
        # From ZIP listing: transit_spectra/FIREFLy_transit_spec.h5
        "zip_pick":   ["transit_spectra", "firefly"],
        "fmt":        "h5",
    },
]

MOLECULE_BANDS = [
    (0.589, 0.589, "Na",  "#9370DB"), (0.767, 0.767, "K",   "#FF8C00"),
    (1.15,  1.50,  "H2O", "#4FC3F7"), (1.80,  2.05,  "H2O", "#4FC3F7"),
    (2.30,  2.40,  "CO",  "#FF7043"), (2.65,  2.90,  "CO2", "#FDD835"),
    (2.70,  3.20,  "H2O", "#4FC3F7"), (3.96,  4.10,  "SO2", "#66BB6A"),
    (4.16,  4.58,  "CO2", "#FDD835"), (4.60,  4.96,  "CO",  "#FF7043"),
    (5.00,  5.35,  "H2O", "#4FC3F7"),
]


# ── Download ──────────────────────────────────────────────────────────────────

def stream_download(url, dest):
    dest = Path(dest)
    if dest.exists():
        print(f"  cached   {dest.name}")
        return True
    print(f"  fetching {dest.name} ...")
    try:
        with requests.get(url, stream=True, timeout=120,
                          headers={"User-Agent": "wasp39b-plot/1.0"}) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
        print(f"  saved  ({dest.stat().st_size//1024} KB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if dest.exists(): dest.unlink()
        return False


def zenodo_download_url(record_id):
    r = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=20)
    r.raise_for_status()
    files = r.json().get("files", [])
    if not files:
        raise RuntimeError(f"No files in Zenodo record {record_id}")
    return files[0]["links"]["self"], files[0]["key"]


# ── ZIP extraction ────────────────────────────────────────────────────────────

def extract_from_zip(zip_path, substrings):
    """
    Find the first ZIP member whose lowercased path contains ALL substrings,
    write it to a temp file, return the temp file path.
    The caller is responsible for deleting it.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        print(f"  ZIP members ({len(names)} files):")
        for n in names:
            if not any(s in ["__macosx", ".ds_store"] for s in n.lower().split("/")):
                print(f"    {n}")

        for name in names:
            low = name.lower()
            if all(s in low for s in substrings):
                ext = Path(name).suffix
                tmp = tempfile.NamedTemporaryFile(
                    suffix=ext, delete=False, dir=OUT_DIR)
                tmp.write(zf.read(name))
                tmp.close()
                print(f"  Extracted: {name}")
                return Path(tmp.name)

    raise RuntimeError(
        f"No ZIP member matched substrings: {substrings}\n"
        f"Available: {[n for n in names if '__macosx' not in n.lower()]}")


# ── Format readers ────────────────────────────────────────────────────────────

def _to_ppm(wave, depth, err):
    """Normalise units to ppm."""
    mask = np.isfinite(wave) & np.isfinite(depth)
    wave, depth, err = wave[mask], depth[mask], err[mask]
    if len(wave) == 0:
        raise ValueError("No finite data points")
    med = np.nanmedian(depth)
    if 1.0 < med < 10.0:   depth *= 1e4; err *= 1e4   # % → ppm
    elif med < 0.5:         depth *= 1e6; err *= 1e6   # fractional → ppm
    idx = np.argsort(wave)
    return wave[idx], depth[idx], err[idx]


def read_csv(path, col_wave, col_depth, col_err):
    """Read a plain CSV with named columns."""
    from astropy.table import Table
    tbl  = Table.read(str(path), format="csv", comment="#")
    cols = [c.lower() for c in tbl.colnames]
    def gc(name):
        if name.lower() in cols:
            return tbl.colnames[cols.index(name.lower())]
        raise KeyError(f"Column '{name}' not found. Available: {tbl.colnames}")
    wave  = np.asarray(tbl[gc(col_wave)],  float)
    depth = np.asarray(tbl[gc(col_depth)], float)
    err   = np.asarray(tbl[gc(col_err)],   float)
    return _to_ppm(wave, depth, err)


def read_h5(path):
    """
    Read an HDF5 transmission spectrum.
    Tries common group/dataset naming conventions used by Eureka!, FIREFLy,
    tshirt, and HANSOLO pipelines.
    Prints the file structure so the user can see what's inside.
    """
    import h5py

    WAVE_KEYS  = ["central_wavelength", "wave_1d", "wavelength", "wave", "wavegrid",
                  "bin_wave", "wavebin", "lambda"]
    DEPTH_KEYS = ["fp_fstar", "transit_depth", "depth", "dppm",
                  "rp_rs_2", "(rp/rs)^2", "fp", "delta"]
    ERR_KEYS   = ["transit_depth_uncertainty", "transit_depth_error_down",
                  "fp_fstar_err", "e_transit_depth", "err", "error",
                  "depth_err", "dppm_err", "sigma", "uncertainty",
                  "e_fp_fstar", "ferr"]

    def search(f, target_keys):
        """Recursively find the first dataset whose name matches target_keys."""
        for key in target_keys:
            if key in f:
                v = f[key]
                if hasattr(v, 'shape'):   # it's a Dataset
                    return np.asarray(v, dtype=float).ravel()
        # recurse into groups
        for name in f:
            item = f[name]
            if hasattr(item, 'keys'):    # it's a Group
                result = search(item, target_keys)
                if result is not None:
                    return result
        return None

    def print_h5(f, indent=0):
        for name in f:
            item = f[name]
            if hasattr(item, 'shape'):
                print(f"  {'  '*indent}{name}: shape={item.shape}")
            else:
                print(f"  {'  '*indent}{name}/")
                print_h5(item, indent+1)

    with h5py.File(path, "r") as f:
        print("  HDF5 structure:")
        print_h5(f)

        wave  = search(f, WAVE_KEYS)
        depth = search(f, DEPTH_KEYS)
        err   = search(f, ERR_KEYS)

    if wave is None:
        raise ValueError("Could not find wavelength array in HDF5 file")
    if depth is None:
        raise ValueError("Could not find depth array in HDF5 file")
    if err is None:
        print("  Warning: no error array found, using zeros")
        err = np.zeros_like(depth)

    return _to_ppm(wave, depth, err)


def read_nc(path):
    """
    Read a NetCDF4 transmission spectrum.
    Prints variables so the user can see the structure.
    """
    import netCDF4

    WAVE_KEYS  = ["central_wavelength", "wave_1d", "wavelength", "wave", "wavegrid",
                  "bin_wave", "lambda", "WAVELENGTH"]
    DEPTH_KEYS = ["transit_depth", "depth", "dppm", "fp_fstar",
                  "rp_rs_2", "delta", "TRANSIT_DEPTH"]
    ERR_KEYS   = ["transit_depth_error", "transit_depth_uncertainty",
                  "e_transit_depth", "transit_depth_err", "err", "error",
                  "depth_err", "sigma", "fp_fstar_err", "E_TRANSIT_DEPTH"]

    ds = netCDF4.Dataset(path, "r")
    print("  NetCDF4 variables:")
    for vname, var in ds.variables.items():
        print(f"    {vname}: shape={var.shape}, units={getattr(var,'units','?')}")

    def find(keys):
        vnames_lower = {v.lower(): v for v in ds.variables}
        for k in keys:
            if k.lower() in vnames_lower:
                return np.asarray(ds.variables[vnames_lower[k.lower()]][:],
                                  dtype=float).ravel()
        return None

    wave  = find(WAVE_KEYS)
    depth = find(DEPTH_KEYS)
    err   = find(ERR_KEYS)
    ds.close()

    if wave is None:
        raise ValueError("Could not find wavelength variable in NetCDF file")
    if depth is None:
        raise ValueError("Could not find depth variable in NetCDF file")
    if err is None:
        print("  Warning: no error variable found, using zeros")
        err = np.zeros_like(depth)

    return _to_ppm(wave, depth, err)


# ── Fetch one dataset ─────────────────────────────────────────────────────────

def fetch_dataset(ds):
    instrument = ds["instrument"]
    print(f"\n-- {ds['label']} --")
    tmp_path = None

    try:
        # ── Direct URL (NIRISS CSV) ───────────────────────────────────────
        if "url" in ds:
            fname = ds["url"].split("/")[-1]
            dest  = OUT_DIR / f"{instrument}_{fname}"
            if not stream_download(ds["url"], dest):
                return None
            wave, depth, err = read_csv(
                dest, ds["col_wave"], ds["col_depth"], ds["col_err"])

        # ── Zenodo ZIP ────────────────────────────────────────────────────
        elif "record" in ds:
            try:
                dl_url, fname = zenodo_download_url(ds["record"])
            except Exception as e:
                print(f"  Zenodo error: {e}"); return None

            zip_dest = OUT_DIR / f"{instrument}_{fname}"
            if not stream_download(dl_url, zip_dest):
                return None

            tmp_path = extract_from_zip(zip_dest, ds["zip_pick"])
            fmt = ds.get("fmt", "h5")
            if fmt == "h5":
                wave, depth, err = read_h5(tmp_path)
            elif fmt == "nc":
                wave, depth, err = read_nc(tmp_path)
            else:
                raise ValueError(f"Unknown format: {fmt}")

        else:
            print("  No 'url' or 'record' key."); return None

    except Exception as e:
        print(f"  ERROR: {e}")
        return None
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

    print(f"  OK: {len(wave)} pts | "
          f"{wave.min():.3f}-{wave.max():.3f} um | "
          f"median {np.median(depth):.0f} ppm")
    return wave, depth, err


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_spectra(results, demo=False):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    seen = set()
    for w0, w1, mol, col in MOLECULE_BANDS:
        ax.axvspan(w0, max(w1, w0+0.01), color=col, alpha=0.10, lw=0, zorder=0)
        mid = (w0+w1)/2 if w1 > w0 else w0
        if mol not in seen:
            ax.text(mid, 2.315, mol, color=col, fontsize=7,
                    ha="center", va="bottom", alpha=0.85, fontfamily="monospace")
            seen.add(mol)

    plotted = False
    for ds, res in zip(DATASETS, results):
        if res is None: continue
        wave, depth, err = res
        ax.errorbar(wave, depth/1e4, yerr=err/1e4,
                    fmt="o", ms=2.5, lw=0, elinewidth=0.6,
                    color=ds["color"], ecolor=ds["color"],
                    alpha=0.85, label=ds["label"], zorder=3)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No data loaded.\nTry --demo for offline mode.",
                transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=11)

    ax.set_xlim(0.45, 5.6);  ax.set_ylim(2.04, 2.34)
    ax.set_xlabel("Wavelength (um)", color="white", fontsize=12)
    ax.set_ylabel("Transit Depth (%)", color="white", fontsize=12)
    ax.tick_params(colors="white", labelsize=10)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.grid(which="major", color="#222", lw=0.5, zorder=1)
    ax.grid(which="minor", color="#1a1a1a", lw=0.3, zorder=1)
    ax.legend(loc="upper left", fontsize=9, facecolor="#1a1a2e",
              edgecolor="#555", labelcolor="white", framealpha=0.9)
    suffix = " [DEMO]" if demo else ""
    ax.set_title(
        f"WASP-39 b - JWST ERS-1366 Transmission Spectra{suffix}\n"
        "(Alderson+, Ahrer+, Feinstein+, Rustamkulov+ 2023, Nature 614)",
        color="white", fontsize=11, pad=10)
    plt.tight_layout()
    out = OUT_DIR / "wasp39b_jwst_spectra.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nFigure saved -> {out}")
    plt.show()


# ── Demo fallback ─────────────────────────────────────────────────────────────

def make_demo_data(ds):
    rng = np.random.default_rng(42)
    grids = {"NIRISS": np.linspace(0.63,2.80,280),
             "NIRCam": np.linspace(2.40,4.00,110),
             "NIRSpec_G395H": np.linspace(2.80,5.20,350),
             "NIRSpec_PRISM": np.linspace(0.50,5.50,210)}
    wave = grids[ds["instrument"]]
    base = np.full_like(wave, 2.147e4)
    def g(w,c,s,a): return a*np.exp(-0.5*((w-c)/s)**2)
    for c,s in [(1.15,.12),(1.40,.13),(1.85,.12),(2.70,.20),(2.90,.12),(5.15,.15)]:
        base += g(wave,c,s,500)
    base += (g(wave,2.77,.08,500) + g(wave,4.30,.12,750) +
             g(wave,2.35,.04,250) + g(wave,4.72,.12,300) +
             g(wave,4.05,.06,250) + g(wave,.589,.015,350))
    base -= 80*(1/wave - 1/3.0)
    n = {"NIRISS":120,"NIRCam":150,"NIRSpec_G395H":100,"NIRSpec_PRISM":180}[ds["instrument"]]
    err = np.abs(np.full_like(wave,n) + rng.normal(0,n*.15,len(wave)))
    return wave, np.clip(base+rng.normal(0,n*.8,len(wave)), 2.05e4, 2.35e4), err


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    demo = "--demo" in sys.argv

    print("=" * 60)
    print("  WASP-39 b JWST Transmission Spectra - ERS-1366")
    print("=" * 60)

    if demo:
        print("\n[DEMO MODE]")
        results = [make_demo_data(ds) for ds in DATASETS]
        plot_spectra(results, demo=True)
    else:
        results = [fetch_dataset(ds) for ds in DATASETS]
        n_ok = sum(r is not None for r in results)
        print(f"\n{n_ok}/{len(DATASETS)} datasets loaded.")
        if n_ok == 0:
            print("All downloads failed - switching to demo mode.")
            results = [make_demo_data(ds) for ds in DATASETS]
            plot_spectra(results, demo=True)
        else:
            plot_spectra(results)

    print("Done.")
