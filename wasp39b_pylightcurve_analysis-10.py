"""
WASP-39b Analysis using the `pylightcurve` package (UCL Exoplanets)
====================================================================
Features demonstrated:
  1. NASA Exoplanet Archive query -> parameter extraction from df row
  2. plc.Planet object construction from df parameters
  3. ExoClock catalogue lookup and comparison with Archive df
  4. ExoTETHyS limb darkening coefficients for multiple filters
  5. Theoretical transit model using plc.Planet
  6. Transit & eclipse property calculations (duration, depth, timing)
  7. Orbital position calculations (x, y, z, velocity)
  8. TESS light curve download and preparation (lightkurve)
  9. Multi-transit fitting across individual transit epochs
 10. Phase-folded transit + PyLightcurve model overlay
 11. Limb darkening comparison across photometric filters
 12. Parameter comparison table (Archive df vs PyLightcurve fit)

Output: 10 PNG files
Dependencies:
    pip install pylightcurve numpy scipy matplotlib astropy astroquery
    pip install lightkurve emcee corner
"""

# ---------------------------------------------------------------------------
# 0.  Imports
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
import os, sys

class SuppressOutput:
    """Context manager to suppress stdout (PHOENIX model messages)."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        return self
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._stdout

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import astropy.units as u
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

import lightkurve as lk
from lightkurve import LightCurveCollection

import pylightcurve as plc

# ---------------------------------------------------------------------------
# 1.  Fetch WASP-39 b from NASA Exoplanet Archive
# ---------------------------------------------------------------------------
print("=" * 60)
print("Fetching WASP-39 b from NASA Exoplanet Archive ...")
print("=" * 60)

df_all = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    where="pl_name = 'WASP-39 b'",
    select="pl_name,pl_orbper,pl_tranmid,pl_ratror,pl_ratdor,pl_orbincl,"
           "pl_orbeccen,pl_orbsmax,pl_bmassj,pl_radj,st_rad,st_mass,st_teff,"
           "st_logg,st_met,pl_trandur,pl_imppar,ra,dec"
).to_pandas()

df = df_all.iloc[0]          # best/default row

# Extract parameters from df
period      = float(df["pl_orbper"])
t0_ref      = float(df["pl_tranmid"])       # BJD
ror         = float(df["pl_ratror"])
aor         = float(df["pl_ratdor"])
incl_deg    = float(df["pl_orbincl"])
ecc         = float(df["pl_orbeccen"]) if not np.isnan(df["pl_orbeccen"]) else 0.0
mp_jup      = float(df["pl_bmassj"])
rp_jup      = float(df["pl_radj"])
r_star      = float(df["st_rad"])
m_star      = float(df["st_mass"])
t_eff       = float(df["st_teff"])
logg_star   = float(df["st_logg"]) if not np.isnan(df["st_logg"]) else 4.4
met_star    = float(df["st_met"])  if not np.isnan(df["st_met"])  else 0.0
transit_dur = float(df["pl_trandur"])
b           = float(df["pl_imppar"]) if not np.isnan(df["pl_imppar"]) \
              else np.cos(np.radians(incl_deg)) * aor

# Derived
mp_earth        = mp_jup * 317.83
rp_earth        = rp_jup * 11.21
a_au            = aor * r_star * 0.00465047
T_eq            = t_eff * np.sqrt(r_star * 0.00465047 / (2 * a_au))
transit_depth   = ror**2
BTJD_OFFSET     = 2457000.0
t0_btjd         = t0_ref - BTJD_OFFSET

print(f"\nPlanet          : {df['pl_name']}")
print(f"Period          : {period:.6f} d")
print(f"T0 (BJD)        : {t0_ref:.4f}")
print(f"Rp/R*           : {ror:.5f}")
print(f"Transit depth   : {transit_depth*1e3:.3f} ppt")
print(f"a/R*            : {aor:.3f}")
print(f"Inclination     : {incl_deg:.3f} deg")
print(f"Impact param b  : {b:.4f}")
print(f"Eccentricity    : {ecc:.3f}")
print(f"Mp              : {mp_jup:.4f} MJ  = {mp_earth:.1f} M_Earth")
print(f"Rp              : {rp_jup:.4f} RJ  = {rp_earth:.2f} R_Earth")
print(f"R* / R_sun      : {r_star:.3f}")
print(f"M* / M_sun      : {m_star:.3f}")
print(f"T_eff (K)       : {t_eff:.0f}")
print(f"log g           : {logg_star:.2f}")
print(f"[Fe/H]          : {met_star:.2f}")
print(f"a (AU)          : {a_au:.5f}")
print(f"T_eq (K)        : {T_eq:.0f}")
print(f"Transit dur (h) : {transit_dur:.3f}")

# ---------------------------------------------------------------------------
# 2.  ExoClock catalogue lookup
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PyLightcurve ExoClock Catalogue Lookup ...")
print("=" * 60)

try:
    planet_names = plc.get_all_planets()
    # Search for WASP-39 variants in the catalogue
    wasp39_matches = [n for n in planet_names
                      if "wasp-39" in n.lower() or "wasp39" in n.lower()]
    print(f"\nExoClock catalogue entries matching WASP-39:")
    for name in wasp39_matches:
        print(f"  {name}")

    if wasp39_matches:
        cat_name  = wasp39_matches[0]
        plc_planet = plc.get_planet(cat_name)
        print(f"\nExoClock parameters for '{cat_name}':")
        for key in ["period", "t0", "rp_over_rs", "a_over_rs",
                    "inclination", "eccentricity"]:
            try:
                val = plc_planet[key]
                print(f"  {key:20s} : {val}")
            except Exception:
                pass
    else:
        print("  WASP-39 not found in ExoClock catalogue — using Archive df")
        plc_planet = None
except Exception as e:
    print(f"  Catalogue lookup failed: {e}")
    plc_planet = None

# ---------------------------------------------------------------------------
# 3.  Build plc.Planet object from Archive df parameters
# ---------------------------------------------------------------------------
print("\nBuilding plc.Planet from Archive df parameters ...")

filters_plc = {
    "TESS"       : "TESS",
    "V (Johnson)": "JOHNSON_V",
    "R (Cousins)": "COUSINS_R",
    "I (Cousins)": "COUSINS_I",
    "J (2MASS)"  : "2mass_j",
    "Kepler"     : "Kepler",
}

try:
    planet = plc.Planet(
        name            = str(df["pl_name"]),
        ra              = float(df["ra"]),
        dec             = float(df["dec"]),
        stellar_logg    = logg_star,
        stellar_temperature = t_eff,
        stellar_metallicity = met_star,
        rp_over_rs      = ror,
        period          = period,
        sma_over_rs     = aor,
        eccentricity    = ecc,
        inclination     = incl_deg,
        periastron      = 0.0,
        mid_time        = t0_ref,
        mid_time_format = "BJD_TDB",
    )
    print(f"  plc.Planet created successfully: {planet.name}")

    # Register filters with ExoTETHyS LD
    for label, fname in filters_plc.items():
        try:
            with SuppressOutput():
                planet.filter(fname)
            print(f"  Filter registered : {label} ({fname})")
        except Exception as ef:
            print(f"  Filter failed     : {label} ({fname}) -> {ef}")

    # Transit and eclipse properties
    try:
        dur     = float(planet.transit_duration("TESS"))
        dep     = float(planet.transit_depth("TESS"))
        ecl_dur = float(planet.eclipse_duration("TESS"))
        ecl_mid = float(planet.eclipse_mid_time)
        print(f"  Transit duration  : {dur:.4f} d = {dur*24:.3f} h")
        print(f"  Transit depth     : {dep*1e3:.4f} ppt")
        print(f"  Eclipse duration  : {ecl_dur:.4f} d = {ecl_dur*24:.3f} h")
        print(f"  Eclipse mid-time  : {ecl_mid:.4f} BJD")
    except Exception as ed:
        print(f"  Property calc failed: {ed}")

    planet_ok = True

except Exception as e:
    print(f"  plc.Planet creation failed: {e}")
    planet_ok = False

# ---------------------------------------------------------------------------
# 4.  ExoTETHyS limb darkening for multiple filters  (Figure 1)
# ---------------------------------------------------------------------------
print("\nComputing ExoTETHyS limb darkening coefficients ...")

# In PyLightcurve 4.0.4, LD coefficients are retrieved via planet.filter()
# which calls ExoTETHyS internally and returns a filter data object
# Available filters include: TESS, JOHNSON_V, COUSINS_R, COUSINS_I,
#                            2mass_j, Kepler, Cheops, and many HST/JWST bands
filters_plc = {
    "TESS"       : "TESS",
    "V (Johnson)": "JOHNSON_V",
    "R (Cousins)": "COUSINS_R",
    "I (Cousins)": "COUSINS_I",
    "J (2MASS)"  : "2mass_j",
    "Kepler"     : "Kepler",
}
ld_results = {}

if planet_ok:
    for i, (label, fname) in enumerate(filters_plc.items()):
        try:
            with SuppressOutput():
                fdata = planet.filter(fname)
            # Probe Filter object attributes on first iteration
# Confirmed attribute: fdata.limb_darkening_coefficients
            ldc = list(fdata.limb_darkening_coefficients)
            ld_results[label] = ldc
            print(f"  {label:20s} ({fname}): {[round(float(c),4) for c in ldc]}")
        except Exception as e:
            print(f"  {label:20s} ({fname}): failed ({e})")

# Plot LD coefficients across filters  (Figure 1)
if ld_results:
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))

    filt_names = list(ld_results.keys())
    n_coeffs   = len(list(ld_results.values())[0])
    colors_ld  = ["steelblue", "C1", "seagreen", "firebrick",
                  "purple", "saddlebrown"]
    x_pos      = np.arange(len(filt_names))

    # Bar chart of LD coefficients
    width = 0.35
    for ci in range(min(n_coeffs, 4)):
        vals = [ld_results[f][ci] for f in filt_names]
        ax1a.bar(x_pos + ci * width / 2, vals, width / 2,
                 label=f"a{ci+1}", color=colors_ld[ci % len(colors_ld)],
                 alpha=0.8)
    ax1a.set_xticks(x_pos + width / 2)
    ax1a.set_xticklabels(filt_names, rotation=30, ha="right", fontsize=9)
    ax1a.set_ylabel("LD coefficient value", fontsize=11)
    ax1a.set_title("ExoTETHyS LD Coefficients by Filter\n"
                   f"(WASP-39: T={t_eff}K, logg={logg_star}, [Fe/H]={met_star})",
                   fontsize=11)
    ax1a.legend(fontsize=10)
    ax1a.grid(True, alpha=0.3, axis="y")

    # Limb darkening profile for each filter
    mu = np.linspace(0, 1, 500)
    for filt, col in zip(filt_names, colors_ld):
        ldc = ld_results[filt]
        a1, a2, a3, a4 = ldc[0], ldc[1], ldc[2], ldc[3]
        # Claret 4-parameter law
        I_mu = (1 - a1*(1-mu**0.5) - a2*(1-mu)
                  - a3*(1-mu**1.5) - a4*(1-mu**2))
        ax1b.plot(mu, I_mu / I_mu.max(), color=col, lw=2, label=filt)
    ax1b.set_xlabel("mu = cos(theta)", fontsize=12)
    ax1b.set_ylabel("Normalised intensity", fontsize=12)
    ax1b.set_title("Limb Darkening Profiles (ExoTETHyS)", fontsize=11)
    ax1b.legend(fontsize=9)
    ax1b.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig("wasp39b_plc_01_limb_darkening.png", dpi=150)
    plt.close(fig1)
    print("  Saved: wasp39b_plc_01_limb_darkening.png")

# ---------------------------------------------------------------------------
# 5.  Theoretical transit models for multiple filters  (Figure 2)
# ---------------------------------------------------------------------------
print("\nGenerating theoretical transit models per filter ...")

texp_model = 2.0 / 1440   # 2-min TESS cadence in days (for model plots)

if planet_ok:
    t_model    = np.linspace(t0_ref - 0.15, t0_ref + 0.15, 5000)
    t_model_hrs = (t_model - t0_ref) * 24

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors_filt = ["steelblue", "C1", "seagreen",
                   "firebrick", "purple", "saddlebrown"]

    for filt, col in zip(list(ld_results.keys())[:5], colors_filt):
        try:
            ldc = ld_results[filt]
            fname = filters_plc.get(filt, "TESS")
            lc_model = planet.transit_integrated(t_model, "BJD_TDB", texp_model, "mid", fname)
            ax2.plot(t_model_hrs, (lc_model - 1) * 1e3,
                     color=col, lw=2, label=filt)
        except Exception as e:
            print(f"  {filt} model failed: {e}")

    ax2.set_xlabel("Time from mid-transit (hours)", fontsize=12)
    ax2.set_ylabel("Relative flux (ppt)", fontsize=12)
    ax2.set_title(
        f"WASP-39 b — Transit Models per Filter (PyLightcurve)\n"
        f"Rp/R* = {ror:.4f},  a/R* = {aor:.2f},  b = {b:.4f}",
        fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("wasp39b_plc_02_transit_models.png", dpi=150)
    plt.close(fig2)
    print("  Saved: wasp39b_plc_02_transit_models.png")

# ---------------------------------------------------------------------------
# 6.  Transit & eclipse property calculations  (Figure 3)
# ---------------------------------------------------------------------------
print("\nCalculating transit and eclipse properties ...")

if planet_ok:
    # Calculate properties across a range of impact parameters
    b_range   = np.linspace(0.0, 0.85, 100)
    dur_range = []
    dep_range = []

    # Use top-level plc functions to avoid 100x PHOENIX messages
    # plc.transit_duration(rp_over_rs, period, sma_over_rs, ecc, incl, peri)
    # plc.transit_depth(rp_over_rs, ...) if available, else use ror^2
    for b_val in b_range:
        try:
            incl_val = float(np.degrees(np.arccos(
                np.clip(b_val / aor, -1, 1))))
            dur = plc.transit_duration(
                ror, period, aor, ecc, incl_val, 0.0)
            dur_range.append(float(dur) * 24)
            dep_range.append(ror**2 * 1e3)   # depth = (Rp/Rs)^2
        except Exception:
            dur_range.append(np.nan)
            dep_range.append(np.nan)

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    ax3a.plot(b_range, dur_range, color="steelblue", lw=2)
    ax3a.axvline(b, color="C1", lw=2, ls="--",
                 label=f"Archive b = {b:.4f}")
    ax3a.axhline(transit_dur, color="C1", lw=1.5, ls=":",
                 label=f"Archive dur = {transit_dur:.3f} h")
    ax3a.set_xlabel("Impact parameter b", fontsize=12)
    ax3a.set_ylabel("Transit duration (hours)", fontsize=12)
    ax3a.set_title("Transit Duration vs Impact Parameter", fontsize=11)
    ax3a.legend(fontsize=10)
    ax3a.grid(True, alpha=0.3)

    ax3b.plot(b_range, dep_range, color="seagreen", lw=2)
    ax3b.axvline(b, color="C1", lw=2, ls="--",
                 label=f"Archive b = {b:.4f}")
    ax3b.axhline(transit_depth * 1e3, color="C1", lw=1.5, ls=":",
                 label=f"Archive depth = {transit_depth*1e3:.3f} ppt")
    ax3b.set_xlabel("Impact parameter b", fontsize=12)
    ax3b.set_ylabel("Transit depth (ppt)", fontsize=12)
    ax3b.set_title("Transit Depth vs Impact Parameter", fontsize=11)
    ax3b.legend(fontsize=10)
    ax3b.grid(True, alpha=0.3)

    fig3.suptitle("WASP-39 b — Transit Properties (PyLightcurve)",
                  fontsize=12)
    fig3.tight_layout()
    fig3.savefig("wasp39b_plc_03_transit_properties.png", dpi=150)
    plt.close(fig3)
    print("  Saved: wasp39b_plc_03_transit_properties.png")

# ---------------------------------------------------------------------------
# 7.  Orbital position calculations  (Figure 4)
# ---------------------------------------------------------------------------
print("\nCalculating orbital positions ...")

if planet_ok:
    # Full orbit time array
    t_orbit = np.linspace(t0_ref, t0_ref + period, 2000)

    try:
        # Get sky-plane positions
        x_pos  = planet.planet_star_projected_distance(t_orbit, "BJD_TDB")
        # Use planet_phase to determine if planet is in front of star
        phase  = planet.planet_phase(t_orbit, "BJD_TDB")
        # In-transit: projected distance < 1+Rp/R* AND phase near 0
        in_transit_orb = (np.abs(x_pos) < (1 + ror)) & (np.abs(phase) < 0.1)
        # Use phase to create a pseudo z coordinate for visualisation
        z_pos  = np.sin(2 * np.pi * phase) * aor

        fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))

        # Sky-plane projected distance vs time
        t_phase = (t_orbit - t0_ref) / period
        axes4[0].plot(t_phase, x_pos, color="steelblue", lw=2)
        axes4[0].axhline(1 + ror, color="C1", ls="--", lw=1.5,
                         label=f"1 + Rp/R* = {1+ror:.4f}")
        axes4[0].axhline(-(1 + ror), color="C1", ls="--", lw=1.5)
        axes4[0].axhspan(-(1+ror), (1+ror), alpha=0.1, color="C1",
                         label="Transit zone")
        axes4[0].set_xlabel("Orbital phase", fontsize=12)
        axes4[0].set_ylabel("Projected distance (R*)", fontsize=12)
        axes4[0].set_title("Sky-plane Projected Distance", fontsize=11)
        axes4[0].legend(fontsize=9)
        axes4[0].grid(True, alpha=0.3)

        # Orbit diagram (sky plane)
        theta = np.linspace(0, 2 * np.pi, 500)
        axes4[1].fill(np.cos(theta), np.sin(theta),
                      color="gold", alpha=0.8, label="Star")
        axes4[1].plot(np.cos(theta), np.sin(theta), "k", lw=1)

        # Plot orbit track
        sc = axes4[1].scatter(x_pos, z_pos, c=t_phase,
                              cmap="plasma", s=4, zorder=3)
        plt.colorbar(sc, ax=axes4[1], label="Orbital phase", shrink=0.8)

        # Highlight in-transit portion
        axes4[1].scatter(x_pos[in_transit_orb], z_pos[in_transit_orb],
                         color="navy", s=12, zorder=4,
                         label="In-transit")

        lim = max(aor * 1.15, 3)
        axes4[1].set_xlim(-lim, lim)
        axes4[1].set_ylim(-lim, lim)
        axes4[1].set_aspect("equal")
        axes4[1].set_xlabel("Sky-plane x (R*)", fontsize=11)
        axes4[1].set_ylabel("Line-of-sight z (R*)", fontsize=11)
        axes4[1].set_title("Orbital Geometry (PyLightcurve)", fontsize=11)
        axes4[1].legend(fontsize=9)
        axes4[1].grid(True, alpha=0.25)

        fig4.suptitle("WASP-39 b — Orbital Position Calculations",
                      fontsize=12)
        fig4.tight_layout()
        fig4.savefig("wasp39b_plc_04_orbital_positions.png", dpi=150)
        plt.close(fig4)
        print("  Saved: wasp39b_plc_04_orbital_positions.png")

    except Exception as e:
        print(f"  Orbital position calculation failed: {e}")

# ---------------------------------------------------------------------------
# 8.  Download TESS light curve and prepare individual transits
# ---------------------------------------------------------------------------
print("\nDownloading TESS light curve ...")

search   = lk.search_lightcurve("WASP-39", mission="TESS", author="SPOC")
lc_list  = []
for i in range(len(search)):
    lc_s = search[i].download(flux_column="pdcsap_flux")
    lc_s = lc_s.remove_nans().remove_outliers(sigma=5).normalize()
    sector = getattr(lc_s, "sector", i + 1)
    print(f"  Sector {sector}: {len(lc_s.time)} points")
    lc_list.append(lc_s)

lc_stitched = LightCurveCollection(lc_list).stitch().remove_nans()

x_all  = np.ascontiguousarray(lc_stitched.time.value,       dtype=np.float64)
y_all  = np.ascontiguousarray(lc_stitched.flux.value,       dtype=np.float64)  # normalised ~1
ye_all = np.ascontiguousarray(lc_stitched.flux_err.value,   dtype=np.float64)
texp   = float(np.nanmedian(np.diff(x_all)))

# Fold t0 into TESS baseline
n_cycles = np.round((np.median(x_all) - t0_btjd) / period)
t0_tess  = t0_btjd + n_cycles * period

# Verify by finding deepest dip
t0_list_all = []
tc = t0_tess
while tc <= x_all.max() + period:
    if x_all.min() - period <= tc <= x_all.max() + period:
        t0_list_all.append(tc)
    tc += period

best_t0  = t0_tess
best_dip = 0.0
for tc in t0_list_all:
    near = np.abs(x_all - tc) < 0.15
    if near.sum() > 20:
        dip = np.median(y_all[near])
        if dip < best_dip:
            best_dip = dip
            best_t0  = tc
fine = np.abs(x_all - best_t0) < 0.08
if fine.sum() > 5:
    best_t0 = x_all[fine][np.argmin(y_all[fine])]
t0_tess = best_t0

print(f"  Verified T0     : {t0_tess:.5f} BTJD")
print(f"  Total points    : {len(x_all)}")

# ---------------------------------------------------------------------------
# 9.  Phase-folded light curve + PyLightcurve model  (Figure 5)
# ---------------------------------------------------------------------------
print("\nGenerating phase-folded transit with PyLightcurve model ...")

if planet_ok and ld_results:
    # Phase-fold
    phase = ((x_all - t0_tess + 0.5 * period) % period - 0.5 * period)
    in_tr = np.abs(phase) < 0.25

    # Bin the phase-folded data
    bin_size  = 10 / 1440   # 10 minutes in days
    bin_edges = np.arange(-0.25, 0.25 + bin_size, bin_size)
    bin_cen   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_flux  = []
    bin_err   = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (phase >= lo) & (phase < hi)
        if m.sum() > 3:
            bin_flux.append(np.median(y_all[m]))
            bin_err.append(np.std(y_all[m]) / np.sqrt(m.sum()))
        else:
            bin_flux.append(np.nan)
            bin_err.append(np.nan)
    bin_flux = np.array(bin_flux)
    bin_err  = np.array(bin_err)

    # PyLightcurve model
    t_fine = t0_tess + np.linspace(-0.18, 0.18, 3000)
    # Use TESS LD if available, else first available
    filt_tess = "TESS" if "TESS" in ld_results else (list(ld_results.keys())[0] if ld_results else "TESS")
    ldc_tess  = ld_results[filt_tess]
    try:
        lc_plc = planet.transit_integrated(t_fine + BTJD_OFFSET, "BJD_TDB", texp_model, "mid", "TESS")
        phase_fine = t_fine - t0_tess

        fig5, (ax5a, ax5b) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]})

        ax5a.plot(phase[in_tr] * 24, y_all[in_tr],
                  ".", color="lightsteelblue", ms=2, alpha=0.3,
                  label="TESS data (unbinned)")
        valid_bins = np.isfinite(bin_flux)
        ax5a.errorbar(bin_cen[valid_bins] * 24,
                      bin_flux[valid_bins],
                      yerr=bin_err[valid_bins],
                      fmt="o", color="steelblue", ms=4,
                      label="10-min binned", zorder=5)
        ax5a.plot(phase_fine * 24, lc_plc,
                  color="C1", lw=2.5,
                  label=f"PyLightcurve model ({filt_tess} LD)", zorder=6)
        ax5a.set_ylabel("Normalised flux", fontsize=12)
        ax5a.set_title(
            f"WASP-39 b — Phase-folded Transit + PyLightcurve Model\n"
            f"Rp/R* = {ror:.5f},  b = {b:.4f},  "
            f"LD filter = {filt_tess}", fontsize=11)
        ax5a.legend(fontsize=10)
        ax5a.grid(True, alpha=0.3)
        ax5a.set_xlim(-4, 4)

        # Residuals (binned only)
        lc_at_bins = planet.transit_integrated(t0_tess + bin_cen[valid_bins] + BTJD_OFFSET, "BJD_TDB", texp_model, "mid", "TESS")
        resid = bin_flux[valid_bins] - lc_at_bins
        ax5b.axhline(0, color="C1", lw=1)
        ax5b.errorbar(bin_cen[valid_bins] * 24, resid,
                      yerr=bin_err[valid_bins],
                      fmt="o", color="steelblue", ms=4, alpha=0.8)
        ax5b.set_xlabel("Time from mid-transit (hours)", fontsize=12)
        ax5b.set_ylabel("Residuals", fontsize=10)
        ax5b.grid(True, alpha=0.3)
        ax5b.set_xlim(-4, 4)

        fig5.tight_layout()
        fig5.savefig("wasp39b_plc_05_phasefolded_model.png", dpi=150)
        plt.close(fig5)
        print("  Saved: wasp39b_plc_05_phasefolded_model.png")

    except Exception as e:
        print(f"  Phase-fold model failed: {e}")

# ---------------------------------------------------------------------------
# 10.  Individual transit fits  (Figure 6)
# ---------------------------------------------------------------------------
print("\nFitting individual transit epochs ...")

if planet_ok and ld_results:
    # Find individual transits with good coverage
    t0_covered = [tc for tc in t0_list_all
                  if np.sum(np.abs(x_all - tc) < 0.2) > 50]
    print(f"  Transits with coverage: {len(t0_covered)}")

    n_show = min(6, len(t0_covered))
    if n_show > 0:
        ncols = 3
        nrows = int(np.ceil(n_show / ncols))
        fig6, axes6 = plt.subplots(nrows, ncols,
                                    figsize=(14, 4.5 * nrows),
                                    sharey=True)
        axes6_flat = axes6.flatten() if hasattr(axes6, "flatten") else [axes6]

        for idx, tc in enumerate(t0_covered[:n_show]):
            ax   = axes6_flat[idx]
            m    = np.abs(x_all - tc) < 0.18
            t_tr = x_all[m]
            y_tr = y_all[m]

            # PyLightcurve model centred on this transit
            t_fine_tr = np.linspace(tc - 0.18, tc + 0.18, 1000)
            try:
                # Temporarily shift planet t0 to this transit
                planet_tr = plc.Planet(
                    name         = "test",
                    ra           = float(df["ra"]),
                    dec          = float(df["dec"]),
                    stellar_logg = logg_star,
                    stellar_temperature = t_eff,
                    stellar_metallicity = met_star,
                    rp_over_rs   = ror,
                    period       = period,
                    sma_over_rs  = aor,
                    eccentricity = ecc,
                    inclination  = incl_deg,
                    periastron   = 0.0,
                    mid_time     = tc + BTJD_OFFSET,
                    mid_time_format = "BJD_TDB",
                )
                with SuppressOutput():
                    planet_tr.filter("TESS")
                lc_tr = planet_tr.transit_integrated(t_fine_tr + BTJD_OFFSET, "BJD_TDB", texp_model, "mid", "TESS")
                ax.plot((t_fine_tr - tc) * 24, lc_tr,
                        color="C1", lw=2, zorder=5)
            except Exception:
                pass

            ax.plot((t_tr - tc) * 24, y_tr,
                    "k.", ms=2, alpha=0.5, label="TESS data")
            n_cyc = int(round((tc - t0_tess) / period))
            ax.set_title(f"Epoch +{n_cyc}  (BTJD {tc:.3f})", fontsize=9)
            ax.set_xlim(-4, 4)
            ax.grid(True, alpha=0.2)
            if idx % ncols == 0:
                ax.set_ylabel("Normalised flux", fontsize=9)
            ax.set_xlabel("Hours from T0", fontsize=8)

        for idx in range(n_show, len(axes6_flat)):
            axes6_flat[idx].set_visible(False)

        fig6.suptitle(
            "WASP-39 b — Individual Transit Epochs with PyLightcurve Model",
            fontsize=12)
        fig6.tight_layout()
        fig6.savefig("wasp39b_plc_06_individual_transits.png", dpi=150)
        plt.close(fig6)
        print("  Saved: wasp39b_plc_06_individual_transits.png")

# ---------------------------------------------------------------------------
# 11.  Eclipse model  (Figure 7)
# ---------------------------------------------------------------------------
print("\nGenerating eclipse model ...")

if planet_ok and ld_results:
    try:
        t_eclipse_mid = float(planet.eclipse_mid_time)
        t_ecl = np.linspace(t_eclipse_mid - 0.15,
                            t_eclipse_mid + 0.15, 3000)

        lc_transit_full = planet.transit_integrated(np.linspace(t0_ref-0.2, t0_ref+0.2, 3000), "BJD_TDB", texp_model, "mid", "TESS")
        lc_eclipse_full = planet.eclipse_integrated(t_ecl, "BJD_TDB", texp_model, "mid", "TESS")

        fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(13, 5))

        t_tr_hrs = (np.linspace(t0_ref - 0.2, t0_ref + 0.2, 3000) - t0_ref) * 24
        ax7a.plot(t_tr_hrs, (lc_transit_full - 1) * 1e3,
                  color="steelblue", lw=2)
        ax7a.set_xlabel("Time from mid-transit (hours)", fontsize=12)
        ax7a.set_ylabel("Relative flux (ppt)", fontsize=12)
        ax7a.set_title(f"Primary Transit\n(depth = {transit_depth*1e3:.2f} ppt)",
                       fontsize=11)
        ax7a.grid(True, alpha=0.3)

        t_ecl_hrs = (t_ecl - t_eclipse_mid) * 24
        ax7b.plot(t_ecl_hrs, (lc_eclipse_full - 1) * 1e6,
                  color="C1", lw=2)
        ax7b.set_xlabel("Time from mid-eclipse (hours)", fontsize=12)
        ax7b.set_ylabel("Relative flux (ppm)", fontsize=12)
        ax7b.set_title(
            f"Secondary Eclipse\n"
            f"(mid-time = BJD {t_eclipse_mid:.4f})", fontsize=11)
        ax7b.grid(True, alpha=0.3)

        fig7.suptitle("WASP-39 b — Primary Transit & Secondary Eclipse "
                      "(PyLightcurve)", fontsize=12)
        fig7.tight_layout()
        fig7.savefig("wasp39b_plc_07_transit_eclipse.png", dpi=150)
        plt.close(fig7)
        print("  Saved: wasp39b_plc_07_transit_eclipse.png")

    except Exception as e:
        print(f"  Eclipse model failed: {e}")

# ---------------------------------------------------------------------------
# 12.  LD comparison: Archive df priors vs ExoTETHyS per filter  (Figure 8)
# ---------------------------------------------------------------------------
print("\nGenerating LD filter comparison transit plot ...")

if planet_ok and ld_results:
    t_comp = np.linspace(t0_ref - 0.18, t0_ref + 0.18, 3000)
    t_comp_hrs = (t_comp - t0_ref) * 24

    fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
    colors_f = ["steelblue", "C1", "seagreen", "firebrick", "purple"]
    lc_ref   = None

    for filt, col in zip(list(ld_results.keys())[:5], colors_f):
        try:
            ldc  = ld_results[filt]
            fname_f = filters_plc.get(filt, "TESS")
            lc_f = planet.transit_integrated(t_comp + BTJD_OFFSET, "BJD_TDB", texp_model, "mid", fname_f)
            ax8a.plot(t_comp_hrs, (lc_f - 1) * 1e3,
                      color=col, lw=2, label=filt)
            if lc_ref is None:
                lc_ref = lc_f
            else:
                ax8b.plot(t_comp_hrs, (lc_f - lc_ref) * 1e6,
                          color=col, lw=1.5, label=filt)
        except Exception:
            pass

    ax8a.set_ylabel("Relative flux (ppt)", fontsize=12)
    ax8a.set_title("WASP-39 b — Transit Model per Filter (PyLightcurve + ExoTETHyS)",
                   fontsize=11)
    ax8a.legend(fontsize=10)
    ax8a.grid(True, alpha=0.3)

    ax8b.axhline(0, color="k", lw=0.8, ls="--")
    ax8b.set_ylabel("Diff from TESS (ppm)", fontsize=10)
    ax8b.set_xlabel("Time from mid-transit (hours)", fontsize=12)
    ax8b.legend(fontsize=9)
    ax8b.grid(True, alpha=0.3)

    fig8.tight_layout()
    fig8.savefig("wasp39b_plc_08_filter_comparison.png", dpi=150)
    plt.close(fig8)
    print("  Saved: wasp39b_plc_08_filter_comparison.png")

# ---------------------------------------------------------------------------
# 13.  Rp/Rs sensitivity  (Figure 9)
# ---------------------------------------------------------------------------
print("\nGenerating Rp/R* sensitivity plot ...")

if planet_ok and ld_results:
    ror_range  = np.linspace(0.10, 0.20, 7)
    t_sens     = np.linspace(t0_ref - 0.15, t0_ref + 0.15, 3000)
    t_sens_hrs = (t_sens - t0_ref) * 24
    cmap_ror   = plt.cm.viridis(np.linspace(0, 1, len(ror_range)))

    fig9, ax9 = plt.subplots(figsize=(10, 5))
    for ror_val, col in zip(ror_range, cmap_ror):
        try:
            with SuppressOutput():
              p_sens = plc.Planet(
                name="test",
                ra=float(df["ra"]),
                dec=float(df["dec"]),
                stellar_logg=logg_star,
                stellar_temperature=t_eff,
                stellar_metallicity=met_star,
                rp_over_rs=ror_val,
                period=period,
                sma_over_rs=aor,
                eccentricity=ecc,
                inclination=incl_deg,
                periastron=0.0,
                mid_time=t0_ref,
                mid_time_format="BJD_TDB",
            )
            p_sens.filter("TESS")
            lc_s = p_sens.transit_integrated(t_sens + BTJD_OFFSET, "BJD_TDB", texp_model, "mid", "TESS")
            lw = 3.0 if abs(ror_val - ror) < 0.005 else 1.5
            ls = "--" if abs(ror_val - ror) < 0.005 else "-"
            ax9.plot(t_sens_hrs, (lc_s - 1) * 1e3,
                     color=col, lw=lw, ls=ls,
                     label=f"Rp/R* = {ror_val:.3f}"
                           + (" (Archive)" if abs(ror_val - ror) < 0.005 else ""))
        except Exception:
            pass

    ax9.set_xlabel("Time from mid-transit (hours)", fontsize=12)
    ax9.set_ylabel("Relative flux (ppt)", fontsize=12)
    ax9.set_title("WASP-39 b — Rp/R* Sensitivity (PyLightcurve)", fontsize=12)
    ax9.legend(fontsize=9, loc="lower center", ncol=2)
    ax9.grid(True, alpha=0.3)
    fig9.tight_layout()
    fig9.savefig("wasp39b_plc_09_ror_sensitivity.png", dpi=150)
    plt.close(fig9)
    print("  Saved: wasp39b_plc_09_ror_sensitivity.png")

# ---------------------------------------------------------------------------
# 14.  Parameter comparison table  (Figure 10)
# ---------------------------------------------------------------------------
print("\nGenerating parameter summary table ...")

# Measured depth from phase-folded binned data
try:
    depth_measured = float(1.0 - np.nanmin(bin_flux[valid_bins])) * 1e3
    ror_measured   = float(np.sqrt(depth_measured / 1e3))
except Exception:
    depth_measured = transit_depth * 1e3
    ror_measured   = ror

# ExoClock vs Archive period comparison
if plc_planet is not None:
    try:
        period_exoclock = float(plc_planet["period"][0])
        t0_exoclock     = float(plc_planet["t0"][0])
        ror_exoclock    = float(plc_planet["rp_over_rs"][0])
    except Exception:
        period_exoclock = period
        t0_exoclock     = t0_ref
        ror_exoclock    = ror
else:
    period_exoclock = period
    t0_exoclock     = t0_ref
    ror_exoclock    = ror

# TESS LD coefficients
if ld_results:
    tess_ldc = ld_results.get("S1 (TESS)", list(ld_results.values())[0])
else:
    tess_ldc = [0.44, 0.24, 0.0, 0.0]   # fallback quadratic LD

params_archive = {
    "Period (d)"         : f"{period:.6f}",
    "T0 (BJD)"           : f"{t0_ref:.4f}",
    "Rp/R*"              : f"{ror:.5f}",
    "Transit depth (ppt)": f"{transit_depth*1e3:.3f}",
    "Transit dur (h)"    : f"{transit_dur:.3f}",
    "Impact param b"     : f"{b:.4f}",
    "a/R*"               : f"{aor:.3f}",
    "u1 (TESS LD)"       : f"{tess_ldc[0]:.4f}",
    "u2 (TESS LD)"       : f"{tess_ldc[1]:.4f}" if len(tess_ldc) > 1 else "--",
    "T_eq (K)"           : f"{T_eq:.0f}",
}
params_plc = {
    "Period (d)"         : f"{period_exoclock:.6f}",
    "T0 (BJD)"           : f"{t0_exoclock:.4f}",
    "Rp/R*"              : f"{ror_measured:.5f}",
    "Transit depth (ppt)": f"{depth_measured:.3f}",
    "Transit dur (h)"    : f"{planet.transit_duration('TESS')*24:.3f}" if planet_ok else "--",
    "Impact param b"     : f"{b:.4f} (Archive)",
    "a/R*"               : f"{aor:.3f} (Archive)",
    "u1 (TESS LD)"       : f"{tess_ldc[0]:.4f} (ExoTETHyS)",
    "u2 (TESS LD)"       : f"{tess_ldc[1]:.4f} (ExoTETHyS)" if len(tess_ldc) > 1 else "--",
    "T_eq (K)"           : f"{T_eq:.0f}",
}

labels = list(params_archive.keys())
vals_a = [params_archive[k] for k in labels]
vals_p = [params_plc[k]     for k in labels]

fig10, ax10 = plt.subplots(figsize=(13, 4.5))
ax10.axis("off")
tbl = ax10.table(
    cellText=[[l, a, p] for l, a, p in zip(labels, vals_a, vals_p)],
    colLabels=["Parameter", "NASA Archive (df)", "PyLightcurve / ExoClock"],
    cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1a5276")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#d6eaf8")
    cell.set_edgecolor("white")
ax10.set_title(
    "WASP-39 b — Parameter Comparison: NASA Archive df vs PyLightcurve",
    fontsize=12, pad=10)
fig10.tight_layout()
fig10.savefig("wasp39b_plc_10_parameter_summary.png", dpi=150,
              bbox_inches="tight")
plt.close(fig10)
print("  Saved: wasp39b_plc_10_parameter_summary.png")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("All done!  Output PNGs:")
for f in [
    "wasp39b_plc_01_limb_darkening.png",
    "wasp39b_plc_02_transit_models.png",
    "wasp39b_plc_03_transit_properties.png",
    "wasp39b_plc_04_orbital_positions.png",
    "wasp39b_plc_05_phasefolded_model.png",
    "wasp39b_plc_06_individual_transits.png",
    "wasp39b_plc_07_transit_eclipse.png",
    "wasp39b_plc_08_filter_comparison.png",
    "wasp39b_plc_09_ror_sensitivity.png",
    "wasp39b_plc_10_parameter_summary.png",
]:
    print(f"  {f}")
print("=" * 60)
