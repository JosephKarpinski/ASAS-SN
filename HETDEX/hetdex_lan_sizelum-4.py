"""
hetdex_lan_sizelum.py
=====================
Load the HETDEX Lyman-Alpha Nebulae catalog (hetdex_lan_v0.3.fits),
deduplicate on dups_detectid, and produce a multi-panel size–luminosity
diagram of r_iso vs logL_lya coloured by source_type.

Catalog columns used
---------------------
  name             HLAN identifier
  detectid         unique detection ID (int64)
  dups_detectid    string listing all detectids for repeat observations
                   of the same source — used to deduplicate
  source_type      'lae' | 'agn' | other
  z_hetdex         spectroscopic redshift
  r_iso            isophotal radius  [proper kpc]
  r_s              exponential scale length  [proper kpc]
  r_s_err          1-sigma uncertainty on r_s
  logL_lya         log10 Lya luminosity  [erg/s]
  logL_lya_err     uncertainty on logL_lya
  flag_resolved    1 = spatially resolved
  dBIC             delta-BIC: extended vs PSF model  (>0 = extended preferred)
  log10_pF         log10 probability favouring extended model
  SB_1sigma_obs    1-sigma surface-brightness sensitivity  [1e-18 cgs]
  field            survey field name

Deduplication strategy
-----------------------
dups_detectid is a space-separated string of all detectids that correspond
to the same astrophysical source.  For each duplicate group we keep only the
row with the LOWEST iso_rel_err (best constrained size measurement).
If dups_detectid is empty/blank, the row has no known duplicate.

Requirements
------------
  pip install astropy numpy matplotlib scipy

Data
----
  hetdex_lan_v0.3.fits
  https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/
  DOI: https://doi.org/10.3847/1538-4357/ae44f3
"""

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

CATALOG_PATH = "hetdex_lan_v0.3.fits"   # path to the LAN FITS file
SAVE_PATH    = "hetdex_lan_sizelum.png" # None = display inline only

# Quality cuts applied after deduplication
# Set to None to disable a cut and see diagnostics first
MIN_FLAG_RESOLVED = 0       # 0 = include all (1 = resolved only); check diagnostics
MIN_DBIC          = -999.0  # set to 2.0 once you confirm dBIC distribution
MIN_LOG_L         = 41.0    # log10 L_Lya lower bound [erg/s]
MAX_ISO_REL_ERR   = 1.0     # relax to 1.0; tighten after seeing distribution

# =============================================================================
# CELL 2 — IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, linregress

from astropy.io    import fits
from astropy.table import Table
import astropy.units as u
from astropy.cosmology import Planck18

try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
        matplotlib.rcParams["figure.dpi"] = 120
except Exception:
    pass

cosmo = Planck18

print("Imports OK.")

# =============================================================================
# CELL 3 — SYNTHETIC DATA GENERATOR (remove when real FITS is available)
# =============================================================================

def make_synthetic_lan(n_total=1800, seed=17):
    """
    Generates a realistic synthetic LAN catalog that mirrors the structure of
    hetdex_lan_v0.3.fits, including repeat observations and dups_detectid.

    Size–luminosity relation follows the observed trend from Ouchi+20 / 
    Kikuta+23: r_iso ~ L^0.25 with scatter, AGN being systematically larger.
    """
    rng = np.random.default_rng(seed)

    # Source types: ~70% LAE, ~20% AGN, ~10% other
    n_unique = int(n_total * 0.72)
    types_u  = rng.choice(["lae", "agn", "none"],
                           size=n_unique,
                           p=[0.70, 0.22, 0.08])

    # Redshifts: LAEs z~2-3.5, AGN slightly broader
    z_u = np.where(
        types_u == "lae",
        rng.uniform(1.9, 3.5, n_unique),
        rng.uniform(1.5, 3.8, n_unique),
    ).astype(np.float32)

    # logL: draw from realistic LF-shaped distribution
    logL_u = np.where(
        types_u == "agn",
        rng.uniform(42.5, 44.2, n_unique),
        rng.uniform(41.5, 43.5, n_unique),
    ).astype(np.float32)

    # Size–luminosity relation: r_iso ~ 10^(0.25*(logL-42)) * type_factor
    type_factor = np.where(types_u == "agn", 1.6,
                  np.where(types_u == "lae", 1.0, 0.8))
    r_iso_u = (type_factor
               * 10.0**(0.25 * (logL_u - 42.0))
               * rng.lognormal(0.0, 0.25, n_unique)).astype(np.float32)
    r_iso_u = np.clip(r_iso_u, 0.3, 120.0)

    # Scale length r_s ~ 0.3 * r_iso with scatter
    r_s_u   = (r_iso_u * rng.uniform(0.20, 0.45, n_unique)).astype(np.float32)
    r_s_err_u = (r_s_u * rng.uniform(0.05, 0.25, n_unique)).astype(np.float32)

    logL_err_u   = rng.uniform(0.02, 0.15, n_unique).astype(np.float32)
    iso_rel_err_u= rng.uniform(0.03, 0.55, n_unique).astype(np.float32)

    # flag_resolved: larger/brighter sources more likely resolved
    p_res = np.clip((r_iso_u / 8.0) * 0.85, 0.05, 0.98)
    flag_res_u = rng.binomial(1, p_res).astype(np.int64)

    # dBIC and log10_pF correlated with flag_resolved
    dBIC_u     = np.where(flag_res_u == 1,
                           rng.uniform(2.0, 40.0, n_unique),
                           rng.uniform(-5.0, 3.0, n_unique)).astype(np.float32)
    log10_pF_u = (dBIC_u / 20.0 + rng.normal(0, 0.3, n_unique)).astype(np.float32)

    SB_u  = rng.uniform(0.8, 5.0, n_unique).astype(np.float32)
    ra_u  = rng.uniform(130.0, 235.0, n_unique).astype(np.float32)
    dec_u = rng.uniform(42.0,  57.0,  n_unique).astype(np.float32)

    fields_u = rng.choice(
        ["dex-spring", "dex-fall", "cosmos", "goods-n"],
        n_unique, p=[0.55, 0.30, 0.10, 0.05]
    )

    detectid_base = np.arange(2_100_000_001,
                               2_100_000_001 + n_unique, dtype=np.int64)

    # ── Build repeat observations ─────────────────────────────────────────────
    # ~28% of sources have 1-3 repeat observations
    rows   = {k: [] for k in [
        "name", "ra", "dec", "source_type", "z_hetdex",
        "detectid", "shotid", "field", "SB_1sigma_obs",
        "r_iso", "r_s", "r_s_err", "logl_lya", "logl_lya_err",
        "flag_resolved", "dbic", "log10_pf", "iso_rel_err",
        "dups_detectid",
    ]}

    shotid_counter = int(2e10)

    for i in range(n_unique):
        n_reps = rng.choice([1, 2, 3], p=[0.72, 0.20, 0.08])
        did_primary = detectid_base[i]
        all_dids = [did_primary]

        # generate extra detectids for repeats
        for _ in range(n_reps - 1):
            all_dids.append(did_primary + rng.integers(1000, 9999))

        dups_str = " ".join(str(d) for d in all_dids) if n_reps > 1 else ""

        for rep_idx, did in enumerate(all_dids):
            # Add slight noise to repeated measurements
            noise = 1.0 + rng.normal(0, 0.04) if rep_idx > 0 else 1.0
            iso_err_rep = float(iso_rel_err_u[i]) * rng.uniform(0.9, 1.1)

            rows["name"].append(f"HLAN{did_primary}")
            rows["ra"].append(float(ra_u[i]))
            rows["dec"].append(float(dec_u[i]))
            rows["source_type"].append(types_u[i])
            rows["z_hetdex"].append(float(z_u[i]))
            rows["detectid"].append(int(did))
            rows["shotid"].append(shotid_counter + i * 10 + rep_idx)
            rows["field"].append(fields_u[i])
            rows["SB_1sigma_obs"].append(float(SB_u[i]))
            rows["r_iso"].append(float(r_iso_u[i]) * noise)
            rows["r_s"].append(float(r_s_u[i]) * noise)
            rows["r_s_err"].append(float(r_s_err_u[i]))
            rows["logl_lya"].append(float(logL_u[i]))
            rows["logl_lya_err"].append(float(logL_err_u[i]))
            rows["flag_resolved"].append(int(flag_res_u[i]))
            rows["dbic"].append(float(dBIC_u[i]))
            rows["log10_pf"].append(float(log10_pF_u[i]))
            rows["iso_rel_err"].append(iso_err_rep)
            rows["dups_detectid"].append(dups_str)

    tab = Table(rows)
    print(f"  Synthetic catalog: {len(tab):,} rows "
          f"({n_unique:,} unique sources + repeats)")
    return tab

# =============================================================================
# CELL 4 — LOAD CATALOG
# =============================================================================

def load_lan(path):
    try:
        hdul = fits.open(path)
        tab  = Table(hdul[1].data)
        hdul.close()
        # Normalise column names to lower-case
        tab.rename_columns(tab.colnames, [c.lower() for c in tab.colnames])
        # HSC-r_mag has a hyphen — rename to safe Python identifier
        for old, new in [("hsc-r_mag", "hsc_r_mag"),
                         ("hsc-r_mag_err", "hsc_r_mag_err")]:
            if old in tab.colnames:
                tab.rename_column(old, new)
        synthetic = False
        print(f"Loaded {path}: {len(tab):,} rows, "
              f"{len(tab.colnames)} columns")
        print(f"Columns: {tab.colnames}")
    except FileNotFoundError:
        print(f"'{path}' not found — using synthetic demo data.")
        tab       = make_synthetic_lan()
        synthetic = True
    return tab, synthetic


raw, SYNTHETIC = load_lan(CATALOG_PATH)
print(f"\nRaw rows: {len(raw):,}")
print(f"source_type values: {np.unique(np.array(raw['source_type'], dtype=str))}")

# =============================================================================
# CELL 5 — DEDUPLICATION
# =============================================================================

def deduplicate_lan(tab):
    """
    Deduplicate the LAN catalog using the dups_detectid column.

    Strategy
    --------
    dups_detectid is a space-separated string of all detectids belonging to
    the same astrophysical source (including the row's own detectid).
    An empty string means the row has no known duplicate.

    For each duplicate group we keep the single row with the smallest
    iso_rel_err (best-constrained size measurement).  Rows with no duplicate
    are kept as-is.

    Returns
    -------
    dedup : Table  with one row per unique source
    n_removed : int
    """
    n_in  = len(tab)
    dids  = np.array(tab["detectid"],      dtype=np.int64)
    dups  = np.array(tab["dups_detectid"], dtype=str)
    ierr  = np.array(tab["iso_rel_err"],   dtype=float)

    keep = np.ones(n_in, dtype=bool)
    seen = set()   # detectids already assigned to a kept row

    # Sort by iso_rel_err ascending so the first time we see a group
    # we're looking at the best-quality row
    order = np.argsort(ierr)

    for idx in order:
        did = int(dids[idx])
        if did in seen:
            keep[idx] = False
            continue

        # Parse the group from dups_detectid
        dup_str = dups[idx].strip()
        if dup_str:
            try:
                group = {int(x.strip()) for x in dup_str.replace(',', ' ').split() if x.strip()}
            except ValueError:
                group = {did}
        else:
            group = {did}

        # Mark all group members as seen; only this row (best quality) is kept
        seen |= group

    dedup     = tab[keep]
    n_removed = n_in - keep.sum()
    print(f"\nDeduplication:")
    print(f"  Input rows      : {n_in:,}")
    print(f"  Duplicate rows  : {n_removed:,}")
    print(f"  Unique sources  : {keep.sum():,}")
    return dedup, n_removed


lan_dedup, n_dup_removed = deduplicate_lan(raw)

# =============================================================================
# CELL 6 — QUALITY CUTS
# =============================================================================

def apply_cuts(tab, min_flag=MIN_FLAG_RESOLVED, min_dbic=MIN_DBIC,
               min_logL=MIN_LOG_L, max_rel_err=MAX_ISO_REL_ERR):
    n = len(tab)

    flag  = np.array(tab["flag_resolved"], dtype=float)
    dbic  = np.array(tab["dbic"],          dtype=float)
    logL  = np.array(tab["logl_lya"],      dtype=float)
    r_iso = np.array(tab["r_iso"],         dtype=float)
    ierr  = np.array(tab["iso_rel_err"],   dtype=float)

    # ── Diagnostic: show distributions before cutting ─────────────────────────
    print("\n  Column diagnostics (pre-cut):")
    for label, arr in [("flag_resolved", flag), ("dbic",     dbic),
                        ("logl_lya",     logL),  ("r_iso",   r_iso),
                        ("iso_rel_err",  ierr)]:
        fin = arr[np.isfinite(arr)]
        if len(fin):
            print(f"    {label:<18} min={fin.min():.3g}  "
                  f"med={np.median(fin):.3g}  max={fin.max():.3g}  "
                  f"N_finite={len(fin):,}")
        else:
            print(f"    {label:<18} all NaN/Inf!")

    # ── Apply cuts one at a time and report survival ──────────────────────────
    m_base  = np.isfinite(logL) & np.isfinite(r_iso) & (r_iso > 0)
    m_flag  = m_base  & (flag >= min_flag)
    m_dbic  = m_flag  & (dbic >= min_dbic)
    m_logL  = m_dbic  & (logL >= min_logL)
    m_ierr  = m_logL  & (ierr <= max_rel_err)

    print(f"\n  Quality cuts (sequential survival):")
    print(f"    finite + r_iso>0          : {m_base.sum():>7,} / {n:,}")
    print(f"    flag_resolved >= {min_flag:<6}  : {m_flag.sum():>7,}")
    print(f"    dbic >= {min_dbic:<10}      : {m_dbic.sum():>7,}")
    print(f"    logl_lya >= {min_logL:<7}    : {m_logL.sum():>7,}")
    print(f"    iso_rel_err <= {max_rel_err:<5}    : {m_ierr.sum():>7,}")

    cut = tab[m_ierr]
    print(f"\n  Final: {len(cut):,} / {n:,} sources pass all cuts")
    if len(cut) == 0:
        print("  *** WARNING: 0 sources remain — relax cuts in CELL 1 ***")
    return cut


lan = apply_cuts(lan_dedup)

# Extract arrays
z        = np.array(lan["z_hetdex"],    dtype=float)
r_iso    = np.array(lan["r_iso"],       dtype=float)   # proper kpc
r_s      = np.array(lan["r_s"],         dtype=float)   # proper kpc
r_s_err  = np.array(lan["r_s_err"],     dtype=float)
logL     = np.array(lan["logl_lya"],    dtype=float)
logL_err = np.array(lan["logl_lya_err"],dtype=float)
dbic     = np.array(lan["dbic"],        dtype=float)
stype    = np.array(lan["source_type"], dtype=str)
field    = np.array(lan["field"],       dtype=str)

# Source-type palette (robust against extra whitespace in string column)
stype_clean = np.array([s.strip().lower() for s in stype])

TYPE_STYLES = {
    "lae" : {"color": "#58a6ff", "marker": "o",  "label": "LAE",   "zorder": 4},
    "agn" : {"color": "#f78166", "marker": "s",  "label": "AGN",   "zorder": 5},
    "none": {"color": "#8b949e", "marker": "^",  "label": "Other", "zorder": 3},
}

unique_types = [t for t in ["lae", "agn", "none"]
                if (stype_clean == t).sum() > 0]

print(f"\nSource counts after cuts:")
for t in unique_types:
    print(f"  {t:8s}: {(stype_clean == t).sum():,}")

# =============================================================================
# CELL 7 — POWER-LAW FIT HELPERS
# =============================================================================

def powerlaw_fit(logL_arr, log_r_arr, weights=None):
    """
    Fit log r_iso = a * logL + b via weighted least squares.
    Returns (slope, intercept, r_value, logL_grid, log_r_fit).
    """
    finite = np.isfinite(logL_arr) & np.isfinite(log_r_arr)
    x, y   = logL_arr[finite], log_r_arr[finite]
    w      = weights[finite] if weights is not None else None

    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.array([]), np.array([])
    if w is not None:
        coeffs = np.polyfit(x, y, 1, w=w)
        slope, intercept = coeffs
        r_val, _ = pearsonr(x, y)
    else:
        slope, intercept, r_val, _, _ = linregress(x, y)

    logL_grid  = np.linspace(x.min() - 0.1, x.max() + 0.1, 200)
    log_r_fit  = slope * logL_grid + intercept
    return slope, intercept, r_val, logL_grid, log_r_fit


def fit_label(slope, intercept, r_val, n):
    sgn = "+" if intercept >= 0 else ""
    return (rf"$\log r = {slope:.2f}\,\log L {sgn}{intercept:.1f}$"
            f"\n$r = {r_val:.2f}$   N = {n:,}")

# =============================================================================
# CELL 8 — MAIN PLOT: five panels
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

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

LOGR_LABEL = r"$r_{\rm iso}$  [proper kpc]"
LOGL_LABEL = r"$\log_{10}\,L_{\rm Ly\alpha}$  [erg s$^{-1}$]"
LOGRS_LABEL= r"$r_s$  [proper kpc]"

fig = plt.figure(figsize=(17, 14))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.38, wspace=0.32,
                        left=0.07, right=0.97,
                        top=0.92,  bottom=0.07)

ax_main  = fig.add_subplot(gs[0, :2])   # wide: main size-lum diagram
ax_hist  = fig.add_subplot(gs[0, 2])    # r_iso histogram by type
ax_rs    = fig.add_subplot(gs[1, 0])    # r_s vs logL
ax_z     = fig.add_subplot(gs[1, 1])    # r_iso vs z coloured by logL
ax_dbic  = fig.add_subplot(gs[1, 2])    # dBIC vs r_iso (morphology)

# ── Panel 1: Main size–luminosity diagram ─────────────────────────────────────
style_ax(ax_main,
         "Size–Luminosity Diagram  (HETDEX Lyman-α Nebulae)",
         LOGL_LABEL, LOGR_LABEL)

log_r = np.log10(np.clip(r_iso, 1e-3, None))

# Scatter per source type
for t in unique_types:
    sel  = stype_clean == t
    st   = TYPE_STYLES.get(t, TYPE_STYLES["none"])
    # error bars on logL
    ax_main.errorbar(
        logL[sel], r_iso[sel],
        xerr=logL_err[sel],
        fmt="none", ecolor=st["color"], elinewidth=0.5,
        alpha=0.25, zorder=st["zorder"] - 1,
    )
    ax_main.scatter(
        logL[sel], r_iso[sel],
        c=st["color"], marker=st["marker"],
        s=18, alpha=0.65, linewidths=0,
        label=f"{st['label']}  (N={sel.sum():,})",
        zorder=st["zorder"],
    )

# Global power-law fit (all types)
w_all = 1.0 / np.clip(logL_err, 0.01, None)**2
sl, ic, rv, lL_fit, lr_fit = powerlaw_fit(logL, log_r, weights=w_all)

# Convert fit back to linear r for the linear-y plot
r_fit_lin = 10**lr_fit
if len(lL_fit) > 0:
    ax_main.plot(lL_fit, r_fit_lin,
                 color="#ffa657", lw=2.0, ls="--", alpha=0.90, zorder=6,
                 label=fit_label(sl, ic, rv, len(logL)))

# LAE-only fit
sel_lae = stype_clean == "lae"
if sel_lae.sum() > 10:
    sl_l, ic_l, rv_l, lL_fl, lr_fl = powerlaw_fit(logL[sel_lae],
                                                    log_r[sel_lae])
    if len(lL_fl) > 0:
        ax_main.plot(lL_fl, 10**lr_fl,
                     color="#58a6ff", lw=1.4, ls=":", alpha=0.80, zorder=5,
                     label=f"LAE fit:  slope={sl_l:.2f}")

# AGN-only fit
sel_agn = stype_clean == "agn"
if sel_agn.sum() > 5:
    sl_a, ic_a, rv_a, lL_fa, lr_fa = powerlaw_fit(logL[sel_agn],
                                                    log_r[sel_agn])
    if len(lL_fa) > 0:
        ax_main.plot(lL_fa, 10**lr_fa,
                     color="#f78166", lw=1.4, ls=":", alpha=0.80, zorder=5,
                     label=f"AGN fit:  slope={sl_a:.2f}")

ax_main.set_yscale("log")
ax_main.set_xlim(logL.min() - 0.2, logL.max() + 0.2)
ax_main.set_ylim(0.2, r_iso.max() * 1.5)
ax_main.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(
    base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
mleg(ax_main, loc="upper left", ncol=2)

# Annotation: halo/blob boundary
BLOB_LIMIT = 30.0   # kpc: conventional Lya blob lower size limit
ax_main.axhline(BLOB_LIMIT, color="#3fb950", lw=1.0, ls="--", alpha=0.55)
ax_main.text(logL.min() - 0.15, BLOB_LIMIT * 1.08,
             "Ly$\\alpha$ blob threshold (30 kpc)",
             color="#3fb950", fontsize=8, alpha=0.80, va="bottom")

# ── Panel 2: r_iso histogram by source type ───────────────────────────────────
style_ax(ax_hist, "r_iso Distribution", LOGR_LABEL, "Normalised count")

r_bins = np.logspace(np.log10(r_iso.min() * 0.9),
                     np.log10(r_iso.max() * 1.1), 30)
for t in unique_types:
    sel = stype_clean == t
    st  = TYPE_STYLES.get(t, TYPE_STYLES["none"])
    ax_hist.hist(r_iso[sel], bins=r_bins, density=True,
                 color=st["color"], alpha=0.50,
                 histtype="stepfilled", label=st["label"])
    ax_hist.hist(r_iso[sel], bins=r_bins, density=True,
                 color=st["color"], alpha=0.90,
                 histtype="step", lw=1.3)

# Median lines
for t in unique_types:
    sel = stype_clean == t
    if sel.sum() == 0:
        continue
    med = np.median(r_iso[sel])
    ax_hist.axvline(med, color=TYPE_STYLES.get(t, TYPE_STYLES["none"])["color"],
                    lw=1.3, ls="--", alpha=0.80)
    ax_hist.text(med * 1.04, ax_hist.get_ylim()[1] * 0.02,
                 f"{med:.1f}",
                 color=TYPE_STYLES.get(t, TYPE_STYLES["none"])["color"],
                 fontsize=7.5, va="bottom")

ax_hist.set_xscale("log")
ax_hist.axvline(BLOB_LIMIT, color="#3fb950", lw=1.0, ls="--", alpha=0.55)
mleg(ax_hist)

# ── Panel 3: r_s vs logL (scale length) ──────────────────────────────────────
style_ax(ax_rs, r"Scale length $r_s$ vs Luminosity", LOGL_LABEL, LOGRS_LABEL)

for t in unique_types:
    sel = stype_clean == t
    st  = TYPE_STYLES.get(t, TYPE_STYLES["none"])
    good = sel & (r_s > 0) & np.isfinite(r_s)
    ax_rs.errorbar(
        logL[good], r_s[good],
        yerr=r_s_err[good],
        fmt=st["marker"], color=st["color"],
        ms=4, lw=0, elinewidth=0.6, alpha=0.55,
        capsize=1.5, ecolor=st["color"],
        label=st["label"], zorder=st["zorder"],
    )

# Overall fit for r_s
good_rs = (r_s > 0) & np.isfinite(r_s) & np.isfinite(logL)
if good_rs.sum() > 5:
    log_rs = np.log10(r_s[good_rs])
    sl_s, ic_s, rv_s, lL_s, lr_s = powerlaw_fit(logL[good_rs], log_rs)
    if len(lL_s) > 0:
        ax_rs.plot(lL_s, 10**lr_s,
                   color="#ffa657", lw=1.8, ls="--", alpha=0.85,
                   label=f"Fit:  slope={sl_s:.2f}, r={rv_s:.2f}")

ax_rs.set_yscale("log")
mleg(ax_rs, loc="upper left")

# ── Panel 4: r_iso vs z coloured by logL ─────────────────────────────────────
style_ax(ax_z, r"$r_{\rm iso}$ vs Redshift",
         "Redshift  $z$", LOGR_LABEL)

sc = ax_z.scatter(z, r_iso,
                   c=logL,
                   cmap="plasma",
                   s=14, alpha=0.65, linewidths=0,
                   vmin=logL.min(), vmax=logL.max(),
                   zorder=4)
cb = fig.colorbar(sc, ax=ax_z, pad=0.02)
cb.set_label(r"$\log L_{\rm Ly\alpha}$", color=MUTED, fontsize=9)
cb.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
cb.outline.set_edgecolor(SPINE)

ax_z.axhline(BLOB_LIMIT, color="#3fb950", lw=1.0, ls="--", alpha=0.55)
ax_z.set_yscale("log")
ax_z.set_ylim(0.2, r_iso.max() * 1.5)

# ── Panel 5: dBIC vs r_iso (morphology indicator) ────────────────────────────
style_ax(ax_dbic,
         r"Extended-source evidence vs $r_{\rm iso}$",
         LOGR_LABEL,
         r"$\Delta$BIC  (extended $-$ PSF)")

for t in unique_types:
    sel = stype_clean == t
    st  = TYPE_STYLES.get(t, TYPE_STYLES["none"])
    ax_dbic.scatter(r_iso[sel], dbic[sel],
                    c=st["color"], marker=st["marker"],
                    s=14, alpha=0.55, linewidths=0,
                    label=st["label"], zorder=st["zorder"])

ax_dbic.axhline(0,   color=SPINE,    lw=0.9, ls=":")
ax_dbic.axhline(2.0, color="#ffa657",lw=1.0, ls="--", alpha=0.60,
                label=r"$\Delta$BIC = 2  (marginal)")
ax_dbic.axhline(6.0, color="#3fb950",lw=1.0, ls="--", alpha=0.60,
                label=r"$\Delta$BIC = 6  (strong)")
ax_dbic.set_xscale("log")
mleg(ax_dbic, loc="upper left")

# ── Super-title & footer ───────────────────────────────────────────────────────
syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
fig.suptitle(
    rf"HETDEX Lyman-$\alpha$ Nebulae — Size–Luminosity Analysis{syn_tag}",
    color=TEXT, fontsize=13, fontweight="bold", y=0.975,
)

n_raw   = len(raw)
n_dedup = len(lan_dedup)
n_final = len(lan)
fig.text(
    0.5, 0.005,
    (f"Raw: {n_raw:,} rows  →  after dedup: {n_dedup:,}  →  "
     f"after quality cuts: {n_final:,}  |  "
     f"flag_resolved≥{MIN_FLAG_RESOLVED}, "
     f"dBIC≥{MIN_DBIC}, "
     f"iso_rel_err≤{MAX_ISO_REL_ERR}  |  "
     f"Planck18 cosmology"),
    ha="center", fontsize=8.5, color=MUTED,
)

if SAVE_PATH:
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 9 — NUMERICAL SUMMARY
# =============================================================================

print("\n" + "=" * 62)
print("  HETDEX LAN — Size–Luminosity Summary")
print("=" * 62)
print(f"  Raw rows (inc. repeats) : {n_raw:,}")
print(f"  Duplicate rows removed  : {n_dup_removed:,}")
print(f"  After deduplication     : {n_dedup:,}")
print(f"  After quality cuts      : {n_final:,}")

print(f"\n  {'Type':<8}  {'N':>6}  "
      f"{'med r_iso':>10}  {'med logL':>10}  {'med z':>8}")
print(f"  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")
for t in unique_types:
    sel = stype_clean == t
    if sel.sum() == 0:
        continue
    print(f"  {t:<8}  {sel.sum():>6}  "
          f"{np.median(r_iso[sel]):>10.2f}  "
          f"{np.median(logL[sel]):>10.3f}  "
          f"{np.median(z[sel]):>8.3f}")

print(f"\n  Global power-law fit (all types):")
print(f"    log r_iso = {sl:.3f} * logL + ({ic:.2f})")
print(f"    Pearson r = {rv:.3f}")
print(f"\n  r_iso range : {r_iso.min():.2f} – {r_iso.max():.2f} kpc")
print(f"  logL range  : {logL.min():.2f} – {logL.max():.2f}")
print(f"  z range     : {z.min():.3f} – {z.max():.3f}")
print(f"\n  Lya blobs (r_iso > 30 kpc): "
      f"{(r_iso > BLOB_LIMIT).sum():,} "
      f"({100*(r_iso > BLOB_LIMIT).mean():.1f}%)")
print("=" * 62)
