"""
hetdex_elixer_diagnose.py
=========================
Analyse and visualise disagreements between the two independent HETDEX
classification pipelines:

  ELiXer    → plya_classification  (0–1 probability line is Lyα)
               z_elixer             (best ELiXer redshift)
               best_pz              (ELiXer redshift confidence)

  Diagnose  → cls_diagnose         (STAR | GALAXY | QSO | UNKNOWN)
               z_diagnose           (Diagnose best-fit redshift)

Both pipelines run independently on the same 1D spectrum + imaging
counterpart data, so their disagreements identify the most ambiguous
detections in the survey.

Classifier definitions
----------------------
ELiXer (Davis et al. 2023):
  Bayesian line-ID and redshift estimator focused on Lya vs OII separation.
  plya_classification ≈ P(line = Lyα | spectrum, imaging).
  Caveat: if no confident assignment is made and the line is not likely Lya,
  ELiXer defaults z to OII-3727 rather than NULL.

Diagnose:
  SED-fitting + spectral-template pipeline classifying each detection as
  STAR / GALAXY / QSO / UNKNOWN and estimating a photometric+spectroscopic
  redshift z_diagnose.

Disagreement categories defined here
--------------------------------------
  AGREE_LAE    plya > 0.5  AND  cls_diagnose in (QSO, GALAXY)  AND  z agree
  AGREE_LOW_Z  plya < 0.5  AND  cls_diagnose in (STAR, GALAXY) AND  z agree
  ELIXER_LAE   plya > 0.5  AND  cls_diagnose == STAR            → likely star?
  ELIXER_LAE2  plya > 0.5  AND  cls_diagnose == GALAXY          AND  Δz large
  DIAG_LAE     plya < 0.5  AND  cls_diagnose == QSO             → ELiXer missed AGN?
  BOTH_UNSURE  plya in (0.3,0.7) AND  cls_diagnose == UNKNOWN

Outputs
-------
  hetdex_elixer_diagnose.png      Main figure
  hetdex_ambiguous_sources.csv    Sources where both pipelines disagree
                                  (candidates for follow-up)

Column names used (detinfo table)
----------------------------------
  plya_classification, z_elixer, best_pz,
  z_diagnose, cls_diagnose,
  source_type, sn, wave, counterpart_mag,
  ra_det, dec_det, field, detectid

Requirements
------------
  pip install astropy numpy matplotlib scipy pandas

Data
----
  hetdex_sc2_detinfo_v1.5.fits
  https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/hetdex_source_catalog_2/
"""

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

DETINFO_PATH  = "hetdex_sc2_detinfo_v1.5.fits"
SAVE_PATH     = "hetdex_elixer_diagnose.png"
CSV_PATH      = "hetdex_ambiguous_sources.csv"

# Thresholds
PLYA_HIGH   = 0.7    # confident ELiXer LAE
PLYA_LOW    = 0.3    # confident ELiXer non-LAE
DZ_MAX      = 0.05   # |z_elixer - z_diagnose| < this → "agree" on redshift
MIN_SN      = 5.0    # minimum line S/N
BAD         = -999.0

# =============================================================================
# CELL 2 — IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.ticker  import AutoMinorLocator
from matplotlib.patches import Patch
from scipy.stats        import gaussian_kde

from astropy.io    import fits
from astropy.table import Table

try:
    from IPython import get_ipython
    _ip = get_ipython()
    if _ip is not None:
        _ip.run_line_magic("matplotlib", "inline")
        matplotlib.rcParams["figure.dpi"] = 120
except Exception:
    pass

print("Imports OK.")

# =============================================================================
# CELL 3 — SYNTHETIC DATA GENERATOR
# =============================================================================

def make_synthetic_detinfo(n=120_000, seed=55):
    """
    Generates a realistic synthetic detinfo table with:
    - plya_classification vs cls_diagnose correlations
    - Realistic disagreement fraction (~8% major disagreements)
    - z_elixer and z_diagnose with correlated errors + occasional
      catastrophic failures (the 'interesting' disagreements)
    """
    rng = np.random.default_rng(seed)

    # ── True source types ────────────────────────────────────────────────────
    # Proportions roughly matching the real SC2 detinfo distribution
    n_lae  = int(n * 0.45)
    n_oii  = int(n * 0.30)
    n_star = int(n * 0.16)
    n_agn  = int(n * 0.05)
    n_unk  = n - n_lae - n_oii - n_star - n_agn

    true_types = (["lae"]  * n_lae  + ["oii"]  * n_oii  +
                  ["star"] * n_star + ["agn"]  * n_agn  +
                  ["none"] * n_unk)

    # ── True redshifts ────────────────────────────────────────────────────────
    z_true = np.zeros(n)
    z_true[:n_lae]                   = rng.uniform(1.9, 3.5, n_lae)
    z_true[n_lae:n_lae+n_oii]        = rng.uniform(0.0, 0.5, n_oii)
    z_true[n_lae+n_oii:n_lae+n_oii+n_star] = rng.uniform(0.0, 0.1, n_star)
    z_true[n_lae+n_oii+n_star:n_lae+n_oii+n_star+n_agn] = rng.uniform(0.5, 3.0, n_agn)
    z_true[n_lae+n_oii+n_star+n_agn:] = rng.uniform(0.0, 3.5, n_unk)

    # Shuffle
    idx    = rng.permutation(n)
    true_t = np.array(true_types)[idx]
    z_t    = z_true[idx]

    # ── ELiXer: plya_classification and z_elixer ──────────────────────────────
    plya = np.zeros(n)
    z_el = np.zeros(n)
    best_pz = np.zeros(n)

    for i in range(n):
        tt = true_t[i]
        zt = z_t[i]

        if tt == "lae":
            # Mostly high plya, occasionally confused with OII at low z
            if rng.uniform() < 0.92:
                plya[i]   = rng.beta(8, 1.5)       # high confidence
                z_el[i]   = zt + rng.normal(0, 0.01)
                best_pz[i]= rng.beta(6, 1.5)
            else:
                # ELiXer confused: thinks OII
                plya[i]   = rng.beta(1.5, 6)
                z_el[i]   = 3727.0 / 1215.67 * (1 + zt) - 1  # OII alias
                best_pz[i]= rng.beta(2, 3)

        elif tt == "oii":
            if rng.uniform() < 0.88:
                plya[i]   = rng.beta(1.5, 7)
                z_el[i]   = zt + rng.normal(0, 0.01)
                best_pz[i]= rng.beta(5, 2)
            else:
                # ELiXer confused: thinks Lya
                plya[i]   = rng.beta(6, 1.5)
                z_el[i]   = 1215.67 / 3727.0 * (1 + zt) - 1  # Lya alias
                best_pz[i]= rng.beta(2, 3)

        elif tt == "star":
            plya[i]   = rng.beta(1.5, 8)
            z_el[i]   = rng.uniform(0.0, 0.1)
            best_pz[i]= rng.beta(3, 2)

        elif tt == "agn":
            # AGN: broad lines, ELiXer often assigns high plya if at z>2
            plya[i]   = (rng.beta(5, 2) if zt > 2 else rng.beta(2, 5))
            z_el[i]   = zt + rng.normal(0, 0.03)
            best_pz[i]= rng.beta(4, 2)

        else:  # none/unknown
            plya[i]   = rng.uniform(0, 1)
            z_el[i]   = rng.uniform(0, 3.5)
            best_pz[i]= rng.beta(1.5, 3)

    # ── Diagnose: cls_diagnose and z_diagnose ─────────────────────────────────
    diag_map = {
        "lae" : ["QSO", "GALAXY", "GALAXY", "QSO", "UNKNOWN"],
        "oii" : ["GALAXY", "GALAXY", "STAR", "GALAXY", "UNKNOWN"],
        "star": ["STAR", "STAR", "STAR", "GALAXY", "UNKNOWN"],
        "agn" : ["QSO", "QSO", "GALAXY", "UNKNOWN", "QSO"],
        "none": ["UNKNOWN", "GALAXY", "STAR", "QSO", "UNKNOWN"],
    }
    cls_diag = []
    z_diag   = np.zeros(n)

    for i in range(n):
        tt = true_t[i]
        zt = z_t[i]
        # Weighted random choice toward correct answer
        choices = diag_map[tt]
        cls_diag.append(rng.choice(choices))
        # z_diagnose: correlated with truth, occasional catastrophic outliers
        if rng.uniform() < 0.90:
            z_diag[i] = zt + rng.normal(0, 0.04)
        else:
            z_diag[i] = rng.uniform(0, 3.5)   # catastrophic failure

    cls_diag = np.array(cls_diag)
    z_diag   = np.clip(z_diag, 0, 5)

    # ── Other columns ─────────────────────────────────────────────────────────
    wave   = np.where(true_t == "lae",
                      1215.67 * (1 + z_t),
                      3727.0  * (1 + z_t))
    wave   = np.clip(wave, 3470, 5540).astype(np.float32)
    sn     = np.abs(rng.lognormal(1.8, 0.6, n)).astype(np.float32)
    cmag   = rng.uniform(18, 27, n).astype(np.float32)
    cmag[rng.uniform(size=n) < 0.15] = BAD    # 15% blank-field

    tab = Table({
        "source_type"       : true_t,
        "detectid"          : np.arange(2_100_000_000,
                                         2_100_000_000 + n, dtype=np.int64),
        "ra_det"            : rng.uniform(130, 235, n).astype(np.float32),
        "dec_det"           : rng.uniform(42,   58, n).astype(np.float32),
        "field"             : rng.choice(
                                  ["dex-spring","dex-fall","cosmos","goods-n"],
                                  n, p=[0.55,0.30,0.10,0.05]).astype("U12"),
        "wave"              : wave,
        "sn"                : sn,
        "counterpart_mag"   : cmag,
        "plya_classification": plya.astype(np.float32),
        "z_elixer"          : z_el.astype(np.float32),
        "best_pz"           : best_pz.astype(np.float32),
        "z_diagnose"        : z_diag.astype(np.float32),
        "cls_diagnose"      : cls_diag.astype("U10"),
    })
    print(f"  Synthetic: {len(tab):,} rows")
    print(f"  cls_diagnose counts: "
          f"{dict(zip(*np.unique(cls_diag, return_counts=True)))}")
    return tab


# =============================================================================
# CELL 4 — LOAD CATALOG
# =============================================================================

def getcol(tab, *cands):
    lc = {c.lower(): c for c in tab.colnames}
    for c in cands:
        if c.lower() in lc:
            return lc[c.lower()]
    raise KeyError(f"None of {cands} found. Have: {list(tab.colnames)[:25]}")

print("Loading detinfo ...")
try:
    hdul = fits.open(DETINFO_PATH, memmap=True)
    tab  = Table(hdul[1].data)
    hdul.close()
    tab.rename_columns(tab.colnames, [c.lower() for c in tab.colnames])
    print(f"  {len(tab):,} rows, {len(tab.colnames)} columns")
    SYNTHETIC = False
except FileNotFoundError:
    print(f"  '{DETINFO_PATH}' not found — using synthetic demo data.")
    tab       = make_synthetic_detinfo()
    SYNTHETIC = True

# =============================================================================
# CELL 5 — EXTRACT ARRAYS
# =============================================================================

PLYA_COL  = getcol(tab, "plya_classification")
ZEL_COL   = getcol(tab, "z_elixer")
PZ_COL    = getcol(tab, "best_pz")
ZDIAG_COL = getcol(tab, "z_diagnose")
CDIAG_COL = getcol(tab, "cls_diagnose")
STYPE_COL = getcol(tab, "source_type")
SN_COL    = getcol(tab, "sn")
WAVE_COL  = getcol(tab, "wave")
DID_COL   = getcol(tab, "detectid")
FIELD_COL = getcol(tab, "field")

try:
    CMAG_COL = getcol(tab, "counterpart_mag")
    cmag_arr = np.array(tab[CMAG_COL], dtype=float)
    cmag_arr[cmag_arr == BAD] = np.nan
except KeyError:
    cmag_arr = np.full(len(tab), np.nan)

plya   = np.array(tab[PLYA_COL],  dtype=float)
z_el   = np.array(tab[ZEL_COL],   dtype=float)
best_pz= np.array(tab[PZ_COL],    dtype=float)
z_diag = np.array(tab[ZDIAG_COL], dtype=float)
cls_d  = np.array([s.strip().upper() for s in tab[CDIAG_COL]], dtype=str)
stype  = np.array([s.strip().lower() for s in tab[STYPE_COL]], dtype=str)
sn_arr = np.array(tab[SN_COL],    dtype=float)
wave   = np.array(tab[WAVE_COL],  dtype=float)
did    = np.array(tab[DID_COL],   dtype=np.int64)
field  = np.array([s.strip().lower() for s in tab[FIELD_COL]], dtype=str)

try:
    ra_arr  = np.array(tab[getcol(tab,"ra_det","ra")],  dtype=float)
    dec_arr = np.array(tab[getcol(tab,"dec_det","dec")], dtype=float)
except KeyError:
    ra_arr  = np.full(len(tab), np.nan)
    dec_arr = np.full(len(tab), np.nan)

# Clean bad values
for arr in [plya, z_el, best_pz, z_diag, sn_arr]:
    arr[arr == BAD] = np.nan
    arr[~np.isfinite(arr)] = np.nan

# Also treat 'n/a' cls_diagnose as UNKNOWN
cls_d[cls_d == "N/A"] = "UNKNOWN"
cls_d[cls_d == ""]    = "UNKNOWN"

# =============================================================================
# CELL 6 — QUALITY FILTER & DISAGREEMENT LABELLING
# =============================================================================

# Base mask: both classifiers must have produced an output
base = (np.isfinite(plya) & np.isfinite(z_el) & np.isfinite(z_diag) &
        (sn_arr >= MIN_SN))

print(f"\nDetections with both classifiers & S/N>{MIN_SN}: {base.sum():,}")

# Δz between the two pipelines
dz = np.abs(z_el - z_diag)

# ELiXer verdict
el_lae     = plya >= PLYA_HIGH
el_not_lae = plya <= PLYA_LOW
el_unsure  = ~el_lae & ~el_not_lae

# Diagnose verdict
DIAG_VALS = ["STAR", "GALAXY", "QSO", "UNKNOWN"]
diag_extragal = np.isin(cls_d, ["GALAXY", "QSO"])
diag_star     = cls_d == "STAR"
diag_qso      = cls_d == "QSO"
diag_unk      = cls_d == "UNKNOWN"

z_agree = dz < DZ_MAX

# ── Six disagreement categories ───────────────────────────────────────────────
cat = np.full(len(tab), "OTHER", dtype="U20")

# Both agree: ELiXer Lya + Diagnose extragalactic + z match
cat[base & el_lae     & diag_extragal & z_agree]  = "AGREE_LAE"
# Both agree: ELiXer non-Lya + Diagnose non-QSO + z match
cat[base & el_not_lae & ~diag_qso     & z_agree]  = "AGREE_LOW_Z"
# ELiXer says Lya but Diagnose says STAR → likely stellar mis-ID by ELiXer
cat[base & el_lae     & diag_star]                 = "EL_LAE_DIAG_STAR"
# ELiXer says Lya but Diagnose says GALAXY + z disagree → OII confusion
cat[base & el_lae     & diag_extragal & ~z_agree]  = "EL_LAE_Z_CLASH"
# ELiXer says NOT Lya but Diagnose says QSO → ELiXer may have missed AGN
cat[base & el_not_lae & diag_qso]                  = "EL_NOTLAE_DIAG_QSO"
# Both unsure
cat[base & el_unsure  & diag_unk]                  = "BOTH_UNSURE"

# ── Summary counts ─────────────────────────────────────────────────────────────
CAT_STYLES = {
    "AGREE_LAE"         : ("#58a6ff", "Agree: LAE"),
    "AGREE_LOW_Z"       : ("#3fb950", "Agree: low-z"),
    "EL_LAE_DIAG_STAR"  : ("#f78166", "ELiXer=LAE / Diagnose=STAR"),
    "EL_LAE_Z_CLASH"    : ("#ffa657", "ELiXer=LAE / z clash"),
    "EL_NOTLAE_DIAG_QSO": ("#d2a8ff", "ELiXer=not-LAE / Diagnose=QSO"),
    "BOTH_UNSURE"       : ("#8b949e", "Both unsure"),
    "OTHER"             : ("#30363d", "Other"),
}

print(f"\nDisagreement category counts:")
for cname, (color, label) in CAT_STYLES.items():
    n_c = (cat == cname).sum()
    pct = 100 * n_c / base.sum() if base.sum() > 0 else 0
    print(f"  {label:<40s}: {n_c:>8,}  ({pct:>5.1f}%)")

# =============================================================================
# CELL 7 — CROSS-TABULATION
# =============================================================================

print("\nCross-tabulation: ELiXer verdict × Diagnose class")
print("  (row = ELiXer Lya probability tier, col = cls_diagnose)")

# ELiXer tiers
def plya_tier(p):
    if np.isnan(p): return "no_data"
    if p >= PLYA_HIGH: return "high (≥{})".format(PLYA_HIGH)
    if p <= PLYA_LOW:  return "low  (≤{})".format(PLYA_LOW)
    return "mid ({}-{})".format(PLYA_LOW, PLYA_HIGH)

el_tier = np.array([plya_tier(p) for p in plya])

# Build cross-tab for base mask only
import pandas as pd
df_xt = pd.DataFrame({
    "el_tier"    : el_tier[base],
    "cls_diagnose": cls_d[base],
})
xtab = pd.crosstab(df_xt["el_tier"], df_xt["cls_diagnose"],
                   margins=True, margins_name="TOTAL")
print(xtab.to_string())

# =============================================================================
# CELL 8 — BUILD AMBIGUOUS SOURCE CATALOGUE
# =============================================================================

# Define ambiguous = ELiXer and Diagnose substantively disagree
# (exclude both-agree and OTHER categories)
ambig_cats = {"EL_LAE_DIAG_STAR","EL_LAE_Z_CLASH","EL_NOTLAE_DIAG_QSO","BOTH_UNSURE"}
ambig_mask = np.isin(cat, list(ambig_cats)) & base

df_ambig = pd.DataFrame({
    "detectid"          : did[ambig_mask],
    "ra"                : ra_arr[ambig_mask],
    "dec"               : dec_arr[ambig_mask],
    "field"             : field[ambig_mask],
    "wave"              : wave[ambig_mask],
    "sn"                : sn_arr[ambig_mask],
    "source_type"       : stype[ambig_mask],
    "plya_classification": plya[ambig_mask],
    "z_elixer"          : z_el[ambig_mask],
    "best_pz"           : best_pz[ambig_mask],
    "z_diagnose"        : z_diag[ambig_mask],
    "cls_diagnose"      : cls_d[ambig_mask],
    "delta_z"           : dz[ambig_mask],
    "counterpart_mag"   : cmag_arr[ambig_mask],
    "disagreement_cat"  : cat[ambig_mask],
}).sort_values("sn", ascending=False).reset_index(drop=True)

if CSV_PATH:
    df_ambig.to_csv(CSV_PATH, index=False, float_format="%.5f")
    print(f"\nAmbiguous sources saved -> {CSV_PATH}  "
          f"({len(df_ambig):,} rows)")

# =============================================================================
# CELL 9 — PLOTTING
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

fig = plt.figure(figsize=(18, 15))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(
    3, 3, figure=fig,
    hspace=0.42, wspace=0.32,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.06,
)

ax_main  = fig.add_subplot(gs[0, :2])   # wide: plya vs z_elixer − z_diagnose
ax_xtab  = fig.add_subplot(gs[0, 2])    # cross-tab heatmap
ax_zdiff = fig.add_subplot(gs[1, 0])    # Δz histogram by disagreement cat
ax_zwav  = fig.add_subplot(gs[1, 1])    # z_elixer vs wave coloured by cat
ax_cmag  = fig.add_subplot(gs[1, 2])    # plya vs counterpart_mag
ax_pie   = fig.add_subplot(gs[2, 0])    # category pie chart
ax_field = fig.add_subplot(gs[2, 1])    # disagreement fraction per field
ax_sn    = fig.add_subplot(gs[2, 2])    # S/N distribution by cat

def style_ax(ax, title, xl="", yl="", minor=True):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, which="both", direction="in",
                   top=True, right=True, labelsize=8.5)
    ax.set_xlabel(xl, color=TEXT, fontsize=9)
    ax.set_ylabel(yl, color=TEXT, fontsize=9)
    ax.set_title(title, color=TEXT, fontsize=10,
                 fontweight="bold", loc="left", pad=5)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

def mleg(ax, **kw):
    return ax.legend(fontsize=8, facecolor="#21262d",
                     edgecolor=SPINE, labelcolor=TEXT, **kw)

# ── Panel 1: Main scatter — plya vs (z_el − z_diag) ─────────────────────────
style_ax(ax_main,
         "ELiXer plya vs redshift offset  Δz = z_ELiXer − z_Diagnose",
         r"$\Delta z = z_{\rm ELiXer} - z_{\rm Diagnose}$",
         r"$p_{\rm Ly\alpha}$  (ELiXer)")

# Background: all detections (sparse, grey)
good = base & (cat == "OTHER")
ax_main.scatter(
    (z_el - z_diag)[good], plya[good],
    s=0.5, c="#21262d", alpha=0.15,
    linewidths=0, rasterized=True,
)

# Foreground: each disagreement category
plot_cats = [c for c in CAT_STYLES if c != "OTHER"]
for cname in plot_cats:
    sel = base & (cat == cname)
    if sel.sum() == 0: continue
    color, label = CAT_STYLES[cname]
    ax_main.scatter(
        (z_el - z_diag)[sel], plya[sel],
        s=4, c=color, alpha=0.65,
        linewidths=0, rasterized=True,
        label=f"{label}  (N={sel.sum():,})",
        zorder=4,
    )

# Reference lines
ax_main.axhline(PLYA_HIGH, color="#ffa657", lw=0.9, ls="--", alpha=0.6)
ax_main.axhline(PLYA_LOW,  color="#ffa657", lw=0.9, ls="--", alpha=0.6)
ax_main.axvline(0,          color=MUTED,    lw=0.9, ls=":",  alpha=0.6)
ax_main.axvline(-DZ_MAX,    color=MUTED,    lw=0.7, ls=":",  alpha=0.4)
ax_main.axvline( DZ_MAX,    color=MUTED,    lw=0.7, ls=":",  alpha=0.4)

ax_main.text(0.99, PLYA_HIGH + 0.015,
             f"p_Lya = {PLYA_HIGH}  (ELiXer confident LAE)",
             color="#ffa657", fontsize=7, ha="right", va="bottom")
ax_main.text(0.99, PLYA_LOW - 0.025,
             f"p_Lya = {PLYA_LOW}  (ELiXer confident non-LAE)",
             color="#ffa657", fontsize=7, ha="right", va="top",
             transform=ax_main.get_yaxis_transform())

ax_main.set_xlim(-1.2, 1.2)
ax_main.set_ylim(-0.02, 1.02)
mleg(ax_main, loc="upper left", ncol=2, markerscale=3)

# ── Panel 2: Cross-tab heatmap ────────────────────────────────────────────────
style_ax(ax_xtab, "ELiXer tier × cls_diagnose",
         "cls_diagnose", "ELiXer p_Lya tier", minor=False)

tiers = [t for t in sorted(df_xt["el_tier"].unique()) if t != "no_data"]
dcols = [c for c in DIAG_VALS if c in df_xt["cls_diagnose"].unique()]

xt_mat = np.zeros((len(tiers), len(dcols)))
for i, tier in enumerate(tiers):
    for j, dcol in enumerate(dcols):
        xt_mat[i, j] = ((df_xt["el_tier"] == tier) &
                         (df_xt["cls_diagnose"] == dcol)).sum()

# Row-normalise
xt_norm = xt_mat / np.maximum(xt_mat.sum(axis=1, keepdims=True), 1)

cmap_xt = plt.cm.YlOrRd
im_xt   = ax_xtab.imshow(xt_norm, cmap=cmap_xt,
                          vmin=0, vmax=1, aspect="auto")
for i in range(len(tiers)):
    for j in range(len(dcols)):
        val = xt_norm[i, j]
        raw = int(xt_mat[i, j])
        col = "black" if val > 0.55 else TEXT
        ax_xtab.text(j, i,
                     f"{val:.2f}\n({raw:,})",
                     ha="center", va="center",
                     fontsize=7, color=col)

ax_xtab.set_xticks(range(len(dcols)))
ax_xtab.set_yticks(range(len(tiers)))
ax_xtab.set_xticklabels(dcols, rotation=25, ha="right",
                         color=TEXT, fontsize=8)
ax_xtab.set_yticklabels(tiers, color=TEXT, fontsize=7.5)
cb_xt = fig.colorbar(im_xt, ax=ax_xtab, fraction=0.046, pad=0.04)
cb_xt.set_label("Row fraction", color=MUTED, fontsize=7)
cb_xt.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
plt.setp(cb_xt.ax.yaxis.get_ticklabels(), color=MUTED)
cb_xt.outline.set_edgecolor(SPINE)

# ── Panel 3: Δz histogram ─────────────────────────────────────────────────────
style_ax(ax_zdiff,
         r"$\Delta z$ distribution by category",
         r"$|z_{\rm ELiXer} - z_{\rm Diagnose}|$",
         "Normalised density")

dz_bins = np.linspace(0, 1.5, 50)
for cname in ["AGREE_LAE", "AGREE_LOW_Z",
              "EL_LAE_DIAG_STAR", "EL_LAE_Z_CLASH",
              "EL_NOTLAE_DIAG_QSO", "BOTH_UNSURE"]:
    sel = base & (cat == cname)
    if sel.sum() < 5: continue
    color, label = CAT_STYLES[cname]
    ax_zdiff.hist(dz[sel], bins=dz_bins, density=True,
                  color=color, alpha=0.45, histtype="stepfilled")
    ax_zdiff.hist(dz[sel], bins=dz_bins, density=True,
                  color=color, lw=1.2, histtype="step",
                  label=label.split("/")[0].strip()[:18])

ax_zdiff.axvline(DZ_MAX, color=MUTED, lw=1.0, ls="--", alpha=0.7,
                 label=f"|Δz| = {DZ_MAX}")
ax_zdiff.set_xlim(0, 1.5)
ax_zdiff.legend(fontsize=6.5, facecolor="#21262d", edgecolor=SPINE, labelcolor=TEXT, loc="upper right")

# ── Panel 4: z_elixer vs observed wavelength ──────────────────────────────────
style_ax(ax_zwav,
         r"$z_{\rm ELiXer}$ vs observed wavelength",
         r"Observed $\lambda$  (Å)",
         r"$z_{\rm ELiXer}$")

# Reference lines: Lya and OII dispersion curves
lam_arr = np.linspace(3470, 5540, 200)
ax_zwav.plot(lam_arr, lam_arr / 1215.67 - 1,
             "--", color="#58a6ff", lw=1.2, alpha=0.70,
             label=r"$z_{\rm Ly\alpha}$")
ax_zwav.plot(lam_arr, lam_arr / 3727.0 - 1,
             "--", color="#3fb950", lw=1.2, alpha=0.70,
             label=r"$z_{\rm [OII]}$")

# Scatter disagreement categories
for cname in ["EL_LAE_DIAG_STAR","EL_LAE_Z_CLASH","EL_NOTLAE_DIAG_QSO","BOTH_UNSURE"]:
    sel = base & (cat == cname)
    if sel.sum() < 3: continue
    color, label = CAT_STYLES[cname]
    ax_zwav.scatter(wave[sel], z_el[sel],
                    s=3, c=color, alpha=0.55,
                    linewidths=0, rasterized=True,
                    label=label.split("/")[0].strip()[:18],
                    zorder=4)

ax_zwav.set_xlim(3450, 5560)
ax_zwav.set_ylim(-0.1, 3.7)
mleg(ax_zwav, loc="upper left", markerscale=3)

# ── Panel 5: plya vs counterpart_mag ──────────────────────────────────────────
style_ax(ax_cmag,
         r"$p_{\rm Ly\alpha}$ vs counterpart magnitude",
         "Counterpart magnitude",
         r"$p_{\rm Ly\alpha}$  (ELiXer)")

for cname in ["AGREE_LAE","AGREE_LOW_Z",
              "EL_LAE_DIAG_STAR","EL_NOTLAE_DIAG_QSO"]:
    sel = base & (cat == cname) & np.isfinite(cmag_arr)
    if sel.sum() < 3: continue
    color, label = CAT_STYLES[cname]
    ax_cmag.scatter(cmag_arr[sel], plya[sel],
                    s=3, c=color, alpha=0.50,
                    linewidths=0, rasterized=True,
                    label=label[:22], zorder=4)

ax_cmag.axhline(PLYA_HIGH, color=MUTED, lw=0.8, ls="--", alpha=0.6)
ax_cmag.axhline(PLYA_LOW,  color=MUTED, lw=0.8, ls="--", alpha=0.6)
ax_cmag.set_ylim(-0.02, 1.02)
mleg(ax_cmag, loc="lower right", markerscale=3)

# ── Panel 6: Pie chart of category fractions ─────────────────────────────────
ax_pie.set_facecolor(AX_BG)
ax_pie.set_title("Category breakdown  (S/N > {})".format(MIN_SN),
                 color=TEXT, fontsize=10, fontweight="bold",
                 loc="left", pad=5)

pie_cats   = [c for c in CAT_STYLES if c != "OTHER"]
pie_counts = [(cat == c).sum() & base.any() for c in pie_cats]
pie_counts = [(cat[base] == c).sum() for c in pie_cats]
pie_colors = [CAT_STYLES[c][0] for c in pie_cats]
pie_labels = [f"{CAT_STYLES[c][1]}\n({(cat[base]==c).sum():,})" for c in pie_cats]

wedges, texts, autotexts = ax_pie.pie(
    pie_counts, labels=None, colors=pie_colors,
    autopct="%1.1f%%", startangle=90,
    pctdistance=0.78, textprops={"color": TEXT, "fontsize": 7},
    wedgeprops={"edgecolor": BG, "linewidth": 1.2},
)
for at in autotexts:
    at.set_fontsize(7)
    at.set_color(BG)

legend_patches = [Patch(color=c, label=l)
                  for c, l in zip(pie_colors, pie_labels)]
ax_pie.legend(handles=legend_patches,
              fontsize=6.5, facecolor="#21262d",
              edgecolor=SPINE, labelcolor=TEXT,
              loc="lower center", bbox_to_anchor=(0.5, -0.30),
              ncol=2)

# ── Panel 7: Disagreement fraction per field ──────────────────────────────────
style_ax(ax_field, "Disagreement fraction per field",
         "Field", "Fraction of base detections")

unique_fields = [f for f in ["dex-spring","dex-fall","cosmos","goods-n","nep","ssa22"]
                 if (field[base] == f).sum() > 10]

field_colors_map = {
    "dex-spring":"#58a6ff","dex-fall":"#3fb950",
    "cosmos":"#f78166","goods-n":"#d2a8ff",
    "nep":"#ffa657","ssa22":"#79c0ff",
}

disagree_mask = np.isin(cat, list(ambig_cats))
x_pos = np.arange(len(unique_fields))

for i, fname in enumerate(unique_fields):
    fm_base  = base & (field == fname)
    fm_dis   = disagree_mask & fm_base
    frac     = fm_dis.sum() / max(fm_base.sum(), 1)
    fc       = field_colors_map.get(fname, MUTED)
    ax_field.bar(i, frac, color=fc, alpha=0.82, edgecolor=SPINE, linewidth=0.5)
    ax_field.text(i, frac + 0.002, f"{frac:.3f}",
                  ha="center", va="bottom", color=TEXT, fontsize=7.5)

ax_field.set_xticks(x_pos)
ax_field.set_xticklabels(
    [f.replace("dex-","") for f in unique_fields],
    color=TEXT, fontsize=8.5, rotation=15)
for tick, fname in zip(ax_field.get_xticklabels(), unique_fields):
    tick.set_color(field_colors_map.get(fname, TEXT))
ax_field.yaxis.set_minor_locator(AutoMinorLocator())

# ── Panel 8: S/N distribution for disagree categories ────────────────────────
style_ax(ax_sn, "S/N distribution: agree vs disagree",
         "S/N", "Normalised density")

sn_bins = np.linspace(MIN_SN, 30, 40)
for cname, label in [("AGREE_LAE",          "Agree: LAE"),
                      ("AGREE_LOW_Z",        "Agree: low-z"),
                      ("EL_LAE_DIAG_STAR",   "EL=LAE / Diag=STAR"),
                      ("EL_LAE_Z_CLASH",     "EL=LAE / z clash"),
                      ("EL_NOTLAE_DIAG_QSO", "EL=notLAE / Diag=QSO")]:
    sel = base & (cat == cname)
    if sel.sum() < 5: continue
    color = CAT_STYLES[cname][0]
    ax_sn.hist(sn_arr[sel], bins=sn_bins, density=True,
               color=color, alpha=0.45, histtype="stepfilled")
    ax_sn.hist(sn_arr[sel], bins=sn_bins, density=True,
               color=color, lw=1.2, histtype="step", label=label)

ax_sn.set_xlim(MIN_SN, 30)
ax_sn.legend(fontsize=6.5, facecolor="#21262d", edgecolor=SPINE, labelcolor=TEXT, loc="upper right")

# ── Super-title ───────────────────────────────────────────────────────────────
syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
fig.suptitle(
    "HETDEX SC2 — ELiXer vs Diagnose Classifier Disagreement" + syn_tag,
    color=TEXT, fontsize=13, fontweight="bold", y=0.975,
)

if SAVE_PATH:
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 10 — SUMMARY
# =============================================================================

print("\n" + "=" * 68)
print("  HETDEX SC2 — ELiXer × Diagnose Disagreement Summary")
print("=" * 68)
print(f"  Total detections                : {len(tab):,}")
print(f"  Both classifiers + S/N>{MIN_SN}   : {base.sum():,}")
print(f"  Ambiguous (disagree) sources    : {ambig_mask.sum():,}  "
      f"({100*ambig_mask.sum()/max(base.sum(),1):.1f}%)")
print(f"\n  Category breakdown:")
total_base = base.sum()
for cname, (color, label) in CAT_STYLES.items():
    n_c  = (cat[base] == cname).sum()
    pct  = 100 * n_c / max(total_base, 1)
    flag = "  ← DISAGREE" if cname in ambig_cats else ""
    print(f"    {label:<40s} {n_c:>8,}  ({pct:>5.1f}%){flag}")

print(f"\n  Top disagreement pairs by Δz:")
print(f"    {'detectid':>14}  {'plya':>6}  {'z_el':>7}  "
      f"{'z_diag':>7}  {'dz':>6}  {'cls_d':>8}  {'cat':>22}")
print("    " + "-"*80)
top_dis = df_ambig.nlargest(10, "delta_z")
for _, row in top_dis.iterrows():
    print(f"    {int(row['detectid']):>14d}  "
          f"{row['plya_classification']:>6.3f}  "
          f"{row['z_elixer']:>7.4f}  "
          f"{row['z_diagnose']:>7.4f}  "
          f"{row['delta_z']:>6.3f}  "
          f"{row['cls_diagnose']:>8s}  "
          f"{row['disagreement_cat']:>22s}")
print("=" * 68)
