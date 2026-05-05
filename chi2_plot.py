"""
extract_and_plot_chi2.py
========================
Extracts (age_gyr, chi2_best) pairs from YaPSI and MIST chi2_scan results
and saves a combined χ² vs Age plot.

Both chi2_scan functions return a list of tuples:
    (age_gyr, chi2, mass, Teff, logg, logL, R)
so extraction is identical — just index [0] and [1].

Usage
-----
Call run_extraction(yapsi_results, mist_results) with the raw output from
each pipeline's chi2_scan, or use the hardcoded demo data below.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json, os

# ---------------------------------------------------------------------------
# 1.  Extraction helpers
# ---------------------------------------------------------------------------

def extract_age_chi2(scan_results):
    """
    Extract parallel (ages, chi2) arrays from chi2_scan output.

    Parameters
    ----------
    scan_results : list of tuples
        Direct return value of chi2_scan().
        Each tuple: (age_gyr, chi2, mass, Teff, logg, logL, R)

    Returns
    -------
    ages  : np.ndarray  — age in Gyr
    chi2s : np.ndarray  — best chi² at that age
    """
    if not scan_results:
        return np.array([]), np.array([])
    arr   = np.array(scan_results, dtype=float)   # shape (N, 7)
    ages  = arr[:, 0]
    chi2s = arr[:, 1]
    return ages, chi2s


def save_params_json(ages_yapsi, chi2_yapsi, ages_mist, chi2_mist,
                     path="chi2_params.json"):
    """Persist the plot-ready arrays to JSON for reproducibility."""
    payload = {
        "yapsi": {"ages_gyr": ages_yapsi.tolist(), "chi2": chi2_yapsi.tolist()},
        "mist":  {"ages_gyr": ages_mist.tolist(),  "chi2": chi2_mist.tolist()},
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  [extract] parameters saved → {path}")
    return payload


# ---------------------------------------------------------------------------
# 2.  Plotting
# ---------------------------------------------------------------------------

def plot_combined_chi2(ages_yapsi, chi2_yapsi, ages_mist, chi2_mist,
                       output_path="combined_chi2_vs_age.png"):
    """
    Produce a publication-quality combined χ² vs Age plot for MIST & YaPSI.
    Two panels: linear scale (top) and log scale (bottom) for full dynamic range.
    """
    min_y = np.argmin(chi2_yapsi)
    min_m = np.argmin(chi2_mist)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 10),
                              gridspec_kw={"hspace": 0.35})

    for ax, yscale, tag in zip(axes, ["linear", "log"], ["Linear scale", "Log scale"]):

        ax.plot(ages_yapsi, chi2_yapsi, "o-",  color="tab:blue",
                lw=1.8, ms=5, label="YaPSI")
        ax.plot(ages_mist,  chi2_mist,  "s--", color="tab:orange",
                lw=1.8, ms=5, label="MIST")

        # Highlight global minima
        ax.plot(ages_yapsi[min_y], chi2_yapsi[min_y], "o",
                color="tab:blue",   ms=11, zorder=5,
                label=f"YaPSI min  ({ages_yapsi[min_y]:.1f} Gyr, χ²={chi2_yapsi[min_y]:.2f})")
        ax.plot(ages_mist[min_m],  chi2_mist[min_m],  "s",
                color="tab:orange", ms=11, zorder=5,
                label=f"MIST min  ({ages_mist[min_m]:.2f} Gyr, χ²={chi2_mist[min_m]:.3f})")

        ax.set_yscale(yscale)
        ax.set_xlabel("Age (Gyr)", fontsize=12)
        ax.set_ylabel(r"$\chi^2$", fontsize=12)
        ax.set_title(rf"Combined $\chi^2$ vs Age — MIST vs YaPSI  [{tag}]",
                     fontsize=13)
        ax.legend(fontsize=9, loc="upper left")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        if yscale == "linear":
            # Clip y-axis so the well-resolved minimum region is visible
            chi2_clip = 200
            ax.set_ylim(-5, chi2_clip)
            ax.annotate("(values > 200 clipped)", xy=(0.98, 0.96),
                        xycoords="axes fraction", ha="right", va="top",
                        fontsize=8, color="gray")

    fig.suptitle("Stellar Isochrone Fit Quality: MIST vs YaPSI", fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [plot]    figure saved → {output_path}")


# ---------------------------------------------------------------------------
# 3.  Main entry point
# ---------------------------------------------------------------------------

def run_extraction(yapsi_scan_results=None, mist_scan_results=None,
                   output_dir="."):
    """
    Full pipeline: extract → save JSON → plot.

    Pass the raw chi2_scan() return values from your notebook.
    If None, the hardcoded reference arrays from the previous run are used.
    """

    # --- Demo / reference data (replace with live scan_results) -----------
    _ages_yapsi_ref = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,
                       6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,
                       12.0,12.5,13.0,13.5,14.0]
    _chi2_yapsi_ref = [230.470,267.862,15.082,21.880,40.550,46.859,28.803,
                       30.297,25.313,20.323,10.962,2.741,32.885,44.603,
                       79.531,245.380,266.957,322.165,476.084,709.212,
                       731.486,801.022,987.320,1474.563,1469.234,1485.898,
                       1555.867,1734.818]
    _ages_mist_ref  = [0.50,0.63,0.79,1.00,1.26,1.58,2.00,2.51,3.16,
                       3.98,5.01,6.31,7.94,10.00,12.59]
    _chi2_mist_ref  = [50.799,29.987,15.643,2.266,3.164,1.935,2.789,1.882,
                       1.846,1.616,1.193,5.340,126.282,471.415,1049.811]
    # -----------------------------------------------------------------------

    if yapsi_scan_results is not None:
        ages_y, chi2_y = extract_age_chi2(yapsi_scan_results)
    else:
        print("  [extract] no YaPSI scan_results supplied — using reference data")
        ages_y = np.array(_ages_yapsi_ref)
        chi2_y = np.array(_chi2_yapsi_ref)

    if mist_scan_results is not None:
        ages_m, chi2_m = extract_age_chi2(mist_scan_results)
    else:
        print("  [extract] no MIST scan_results supplied — using reference data")
        ages_m = np.array(_ages_mist_ref)
        chi2_m = np.array(_chi2_mist_ref)

    json_path = os.path.join(output_dir, "chi2_params.json")
    plot_path = os.path.join(output_dir, "combined_chi2_vs_age.png")

    save_params_json(ages_y, chi2_y, ages_m, chi2_m, path=json_path)
    plot_combined_chi2(ages_y, chi2_y, ages_m, chi2_m, output_path=plot_path)

    # Print summary table
    print("\n  YaPSI best fit:")
    idx = int(np.argmin(chi2_y))
    print(f"    age = {ages_y[idx]:.1f} Gyr,  χ² = {chi2_y[idx]:.3f}")
    print("  MIST best fit:")
    idx = int(np.argmin(chi2_m))
    print(f"    age = {ages_m[idx]:.2f} Gyr,  χ² = {chi2_m[idx]:.3f}")

    return {"ages_yapsi": ages_y, "chi2_yapsi": chi2_y,
            "ages_mist":  ages_m, "chi2_mist":  chi2_m}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out = run_extraction(output_dir="/mnt/user-data/outputs")
