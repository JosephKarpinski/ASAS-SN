"""
hetdex_lan_deepdive.py
======================
Five deep-dive analysis modules for the HETDEX LAN catalog, each saving
a separate PNG. Imports the shared data pipeline from hetdex_lan_science.py
so all quality cuts and array names are identical.

Modules
-------
  1. hetdex_lan_A_population.png   — Total population statistics
  2. hetdex_lan_B_lae.png          — LAE-hosted nebulae (N=5,714)
  3. hetdex_lan_C_agn.png          — AGN-hosted nebulae (N=1,492)
  4. hetdex_lan_D_blobs.png        — Lyman-α blobs (N=11, r_iso≥50 kpc)
  5. hetdex_lan_E_cosmic.png       — Cosmic evolution (10–12 Gyr)

Run after hetdex_lan_science.py has been executed in the same session,
or run this file standalone (it re-loads the catalog internally).
"""

# =============================================================================
# CELL 1 — SHARED CONFIGURATION  (must match hetdex_lan_science.py)
# =============================================================================

LAN_PATH        = "hetdex_lan_v0.3.fits"
MIN_DBIC        =  2.0
DBIC_STRONG     =  6.0
MIN_LOGL        = 42.0
MAX_ISO_REL_ERR =  1.0
MIN_R_ISO       =  5.0
BLOB_R_ISO      = 50.0
Z_MIN, Z_MAX    = 1.87, 3.52
BAD             = -999.0

OUT = {
    "A": "hetdex_lan_A_population.png",
    "B": "hetdex_lan_B_lae.png",
    "C": "hetdex_lan_C_agn.png",
    "D": "hetdex_lan_D_blobs.png",
    "E": "hetdex_lan_E_cosmic.png",
}

# Interpretive text appended as a dark-themed strip below each figure.
# Edit here to update the captions without touching the plotting code.
CAPTIONS = {
    "A": (
        "A — Total Population",
        "The KS-test table (A8) is the headline — it runs two-sample Kolmogorov-Smirnov tests between "
        "LAE and AGN distributions for r_iso, logL, EW, dBIC, z, and r_s, reporting the D statistic and "
        "p-value. On the real data you will see exactly which physical properties statistically separate the "
        "two host populations. The per-field bar chart (A7) shows whether any field has anomalously high or "
        "low AGN fraction, which would indicate AGN spatial clustering."
    ),
    "B": (
        "B — LAE-Hosted Nebulae",
        "The SB sensitivity vs r_iso plot (B5) tests whether larger LAE nebulae are detected preferentially "
        "in deeper exposures, which would be a selection bias. B7 shows per-field size distributions, directly "
        "testing whether dex-spring LAE nebulae are systematically different from dex-fall ones (they should "
        "not be if the survey is uniform). B4 (g-band magnitude vs logL) validates the LAE photometric "
        "properties independently of the spectroscopic measurements."
    ),
    "C": (
        "C — AGN-Hosted Nebulae",
        "C3 is the key comparison — a violin plot of AGN vs LAE r_iso with the KS statistic annotated. "
        "C5 shows the AGN blob fraction as a function of luminosity, answering whether more luminous AGN are "
        "more likely to host a blob. C6 compares the r_iso/r_s concentration ratio between LAE and AGN with "
        "overlaid histograms, directly quantifying whether AGN have more extended profiles relative to their core."
    ),
    "D": (
        "D — Lyman-α Blobs  (r_iso ≥ 50 kpc)",
        "D3 is a fully annotated catalogue table listing all blobs ranked by size with type, logL, and "
        "redshift. D5 shows where 50 kpc sits in the cumulative r_iso distribution — on the real data "
        "this will tell you exactly what percentile of the LAN population qualifies as a blob. D4 tests "
        "whether blobs have higher or lower EW than typical LANs, distinguishing fluorescence-powered "
        "extended emission (high EW) from outflow-powered emission (potentially lower EW)."
    ),
    "E": (
        "E — Cosmic Evolution  (z = 1.9–3.5 — look-back time 10–12 Gyr)",
        "E6 is the concentration evolution r_iso/r_s vs z — if this ratio increases toward higher z it "
        "means nebulae become more centrally concentrated at earlier times, consistent with less developed CGM "
        "structure. E7 shows whether the SB sensitivity varies systematically with redshift, which is the "
        "critical check for whether any apparent size evolution in E1 is real or a survey depth artifact. "
        "E8 gives the clean three-epoch summary table that can go directly into a paper."
    ),
}

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
import matplotlib.ticker as mticker
from matplotlib.ticker  import AutoMinorLocator, LogFormatter
from matplotlib.lines   import Line2D
from matplotlib.patches import Patch, FancyArrowPatch
from scipy.stats        import binned_statistic, pearsonr, ks_2samp
from scipy.optimize     import curve_fit
from scipy.ndimage      import gaussian_filter1d

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

cosmo  = Planck18
LYA_AA = 1215.67

# PIL: used to composite the interpretive text strip below each figure
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("  Warning: Pillow not installed — text strips will be skipped.")
    print("  Install with:  pip install Pillow")

print("Imports OK.")

# =============================================================================
# CELL 3 — DATA PIPELINE  (identical to hetdex_lan_science.py Cells 4–6)
# =============================================================================

def getcol(tab, *cands):
    lc = {c.lower().replace("-","_"): c for c in tab.colnames}
    for c in cands:
        if c.lower().replace("-","_") in lc:
            return lc[c.lower().replace("-","_")]
    raise KeyError(f"None of {cands} found.")


def make_synthetic_lan(n=70_000, seed=77):
    rng = np.random.default_rng(seed)
    stype = np.where(rng.uniform(size=n) < 0.795, "lae", "agn")
    z     = rng.uniform(Z_MIN, Z_MAX, n).astype(np.float32)
    logL  = np.where(stype=="agn",
                     np.clip(rng.lognormal(np.log(43.5), 0.45, n), 42.5, 45.5),
                     np.clip(rng.lognormal(np.log(43.0), 0.40, n), 42.0, 45.0)
                     ).astype(np.float32)
    log_r_iso = 0.24*(logL-43.0) + np.log10(18.0) + rng.normal(0, 0.25, n)
    r_iso = np.clip(10.**log_r_iso, 2.0, 300.0).astype(np.float32)
    r_s   = (r_iso / rng.uniform(2.5, 6.0, n)).astype(np.float32)
    r_s_err=(r_s * rng.uniform(0.05, 0.25, n)).astype(np.float32)
    log_r_norm = np.log10(np.clip(r_iso, 1, None)) - 1.0
    dBIC  = (20.0*log_r_norm + rng.normal(0, 5, n)).astype(np.float32)
    iso_rel = np.clip(0.5/np.clip(r_iso/10., 0.1, None) +
                      rng.exponential(0.2, n), 0.02, 3.0).astype(np.float32)
    chi2e = np.clip(rng.lognormal(0.1, 0.3, n), 0.3, 5.0).astype(np.float32)
    chi2p = (chi2e + dBIC/20.*rng.uniform(0.5,1.5,n)).astype(np.float32)
    log10_pF = np.clip(-0.3*chi2e + 0.5*log_r_norm +
                       rng.normal(0, 0.4, n), -5, 2).astype(np.float32)
    flag_res = (dBIC > 2).astype(np.int64)
    SB_1sigma = rng.lognormal(np.log(1.5), 0.3, n).astype(np.float32)
    ew = np.clip(300.*10.**(-0.4*(logL-42.5))*rng.lognormal(0,0.3,n),
                 5., 500.).astype(np.float32)
    flux_lya = (10.**(logL-43.)*1e-17).astype(np.float32)
    gmag = (23.0 - 2.5*(logL-43.) + rng.normal(0, 0.8, n)).astype(np.float32)
    logL_err = rng.uniform(0.02, 0.15, n).astype(np.float32)
    field = rng.choice(["dex-spring","dex-fall","cosmos","goods-n","nep","ssa22"],
                       n, p=[0.55,0.30,0.08,0.03,0.03,0.01])

    tab = Table({
        "name":np.array([f"HLAN+{i}" for i in range(n)],dtype="U20"),
        "ra":rng.uniform(130,235,n).astype(np.float32),
        "dec":rng.uniform(42,58,n).astype(np.float32),
        "source_type":stype, "z_hetdex":z,
        "z_hetdex_src":np.full(n,"hetdex",dtype="U8"),
        "detectid":np.arange(2_100_000_000,2_100_000_000+n,dtype=np.int64),
        "shotid":np.arange(2_100_000_000,2_100_000_000+n,dtype=np.int64),
        "field":field, "SB_1sigma_obs":SB_1sigma,
        "r_iso":r_iso, "r_s":r_s, "r_s_err":r_s_err,
        "area_iso_2sigma":np.ones(n,dtype=np.float32),
        "area_r_iso_circ":np.ones(n,dtype=np.float32),
        "logL_lya":logL, "logL_lya_err":logL_err,
        "flux_lya":flux_lya, "flux_lya_err":(flux_lya*0.15).astype(np.float32),
        "gmag":gmag,
        "HSC-r_mag":gmag, "HSC-r_mag_err":np.full(n,0.1,dtype=np.float32),
        "combined_eqw_rest_lya":ew,
        "flag_resolved":flag_res,
        "chi2_ext_reduced":chi2e, "chi2_psf_reduced":chi2p,
        "log10_pF":log10_pF, "dBIC":dBIC, "iso_rel_err":iso_rel,
        "dups_detectid":np.full(n,"",dtype="U60"),
    })
    return tab


print("Loading LAN catalog ...")
try:
    hdul = fits.open(LAN_PATH, memmap=True)
    lan  = Table(hdul[1].data); hdul.close()
    lan.rename_columns(lan.colnames,[c.lower() for c in lan.colnames])
    SYNTHETIC = False
    print(f"  {len(lan):,} rows")
except FileNotFoundError:
    print(f"  '{LAN_PATH}' not found — synthetic data.")
    lan = make_synthetic_lan(); SYNTHETIC = True
    lan.rename_columns(lan.colnames,[c.lower() for c in lan.colnames])

# Resolve columns
cols = dict(
    ra=getcol(lan,"ra"), dec=getcol(lan,"dec"),
    stype=getcol(lan,"source_type"), z=getcol(lan,"z_hetdex"),
    r_iso=getcol(lan,"r_iso"), r_s=getcol(lan,"r_s"),
    r_s_err=getcol(lan,"r_s_err"),
    logL=getcol(lan,"logl_lya"), logL_err=getcol(lan,"logl_lya_err"),
    ew=getcol(lan,"combined_eqw_rest_lya"),
    dBIC=getcol(lan,"dbic"), log10_pF=getcol(lan,"log10_pf"),
    flag_res=getcol(lan,"flag_resolved"),
    iso_rel=getcol(lan,"iso_rel_err"),
    chi2e=getcol(lan,"chi2_ext_reduced"),
    chi2p=getcol(lan,"chi2_psf_reduced"),
    sb=getcol(lan,"sb_1sigma_obs"),
    field=getcol(lan,"field"), did=getcol(lan,"detectid"),
    dups=getcol(lan,"dups_detectid"), gmag=getcol(lan,"gmag"),
)

ra    = np.array(lan[cols["ra"]],    dtype=float)
dec   = np.array(lan[cols["dec"]],   dtype=float)
stype = np.array([s.strip().lower() for s in lan[cols["stype"]]])
z     = np.array(lan[cols["z"]],     dtype=float)
r_iso = np.array(lan[cols["r_iso"]], dtype=float)
r_s   = np.array(lan[cols["r_s"]],   dtype=float)
r_s_err=np.array(lan[cols["r_s_err"]],dtype=float)
logL  = np.array(lan[cols["logL"]],  dtype=float)
logL_err=np.array(lan[cols["logL_err"]],dtype=float)
ew    = np.array(lan[cols["ew"]],    dtype=float)
dBIC  = np.array(lan[cols["dBIC"]],  dtype=float)
log10_pF=np.array(lan[cols["log10_pF"]],dtype=float)
flag_res=np.array(lan[cols["flag_res"]],dtype=float)
iso_rel=np.array(lan[cols["iso_rel"]],dtype=float)
chi2e = np.array(lan[cols["chi2e"]], dtype=float)
chi2p = np.array(lan[cols["chi2p"]], dtype=float)
sb    = np.array(lan[cols["sb"]],    dtype=float)
field = np.array([s.strip().lower() for s in lan[cols["field"]]])
did   = np.array(lan[cols["did"]],   dtype=np.int64)
gmag  = np.array(lan[cols["gmag"]],  dtype=float)

for arr in [r_iso,r_s,logL,ew,dBIC,log10_pF,flag_res,iso_rel,
            chi2e,chi2p,sb,gmag,r_s_err,logL_err]:
    arr[arr==BAD]=np.nan; arr[arr<=BAD/2]=np.nan

# Dedup
dup_col  = np.array([s.strip() for s in lan[cols["dups"]]])
seen_dids= set(); keep=np.zeros(len(lan),dtype=bool)
for i,(d,ds_) in enumerate(zip(did,dup_col)):
    if int(d) in seen_dids: continue
    keep[i]=True; seen_dids.add(int(d))
    if ds_:
        for x in ds_.replace(","," ").split():
            try: seen_dids.add(int(x))
            except ValueError: pass

def dk(a): return a[keep]
ra,dec,stype,z,r_iso,r_s,r_s_err = dk(ra),dk(dec),dk(stype),dk(z),dk(r_iso),dk(r_s),dk(r_s_err)
logL,logL_err,ew,dBIC,log10_pF   = dk(logL),dk(logL_err),dk(ew),dk(dBIC),dk(log10_pF)
flag_res,iso_rel,chi2e,chi2p,sb   = dk(flag_res),dk(iso_rel),dk(chi2e),dk(chi2p),dk(sb)
field,did,gmag                     = dk(field),dk(did),dk(gmag)
print(f"After dedup: {keep.sum():,} unique")

# Selections
sel        = ((dBIC>=MIN_DBIC)&(logL>=MIN_LOGL)&(iso_rel<=MAX_ISO_REL_ERR)&
              (r_iso>=MIN_R_ISO)&np.isfinite(r_iso)&np.isfinite(logL)&
              np.isfinite(dBIC)&np.isfinite(z)&(z>=Z_MIN)&(z<=Z_MAX))
sel_strong = sel & (dBIC>=DBIC_STRONG)
sel_lae    = sel & (stype=="lae")
sel_agn    = sel & (stype=="agn")
sel_blob   = sel & (r_iso>=BLOB_R_ISO)

print(f"Bona-fide LANs: {sel.sum():,}  "
      f"(LAE={sel_lae.sum():,}  AGN={sel_agn.sum():,}  "
      f"Blobs={sel_blob.sum():,})")

# Power-law fit helper
def pl_fit(lL,ri,mask):
    x=lL[mask]; y=np.log10(np.clip(ri[mask],0.1,None))
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<5: return np.nan,np.nan,np.nan,np.nan
    c,cv=np.polyfit(x[ok],y[ok],1,cov=True)
    r,_=pearsonr(x[ok],y[ok])
    return c[0],c[1],r,np.sqrt(cv[0,0])

# Lookback time
def lb(z_): return cosmo.lookback_time(z_).to(u.Gyr).value

# =============================================================================
# CELL 3b — TEXT-STRIP COMPOSITING FUNCTION
# =============================================================================

def save_with_caption(fig, filepath, module_key, dpi=150):
    """
    Save a matplotlib figure as PNG then composite an interpretive
    text strip below it using PIL.

    The strip mirrors the dark figure theme:
      - Background  #0d1117
      - Title       #e6edf3  bold
      - Body        #8b949e  regular
      - Separator   #30363d  2px line

    Falls back to a plain fig.savefig() if Pillow is absent.
    """
    # Step 1: render the matplotlib figure to disk
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor=BG)

    if not PIL_OK:
        return  # Pillow not available — skip strip

    title, body = CAPTIONS.get(module_key, ("", ""))
    if not title and not body:
        return  # no caption defined for this key

    # Step 2: open the rendered PNG
    orig = Image.open(filepath).convert("RGB")
    W, H = orig.size

    # ── Strip geometry ────────────────────────────────────────────────────────
    # Height scales with figure width so text is always readable
    STRIP_H  = max(260, W // 12)
    MARGIN_X = max(40, W // 55)
    MARGIN_Y = max(22, STRIP_H // 10)
    LINE_GAP = 10

    # Font sizes scale with figure width
    TITLE_FS = max(26, W // 100)
    BODY_FS  = max(22, W // 110)

    # ── Colour constants (matching the matplotlib dark theme) ─────────────────
    C_BG    = (13,  17,  23)    # #0d1117
    C_TITLE = (230, 237, 243)   # #e6edf3
    C_BODY  = (139, 148, 158)   # #8b949e
    C_SPINE = (48,  54,  61)    # #30363d
    C_ACCENT= (255, 166, 87)    # #ffa657

    # ── Render strip ──────────────────────────────────────────────────────────
    strip = Image.new("RGB", (W, STRIP_H), C_BG)
    draw  = ImageDraw.Draw(strip)

    # Separator line at top
    draw.rectangle([0, 0, W, 2], fill=C_SPINE)

    # Load fonts — try several common paths; fall back gracefully
    def _load_font(bold=False, size=20):
        names_bold   = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
        names_regular= [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        candidates = names_bold if bold else names_regular
        for p in candidates:
            try:
                return ImageFont.truetype(p, size)
            except (OSError, IOError):
                pass
        return ImageFont.load_default()

    font_title = _load_font(bold=True,  size=TITLE_FS)
    font_body  = _load_font(bold=False, size=BODY_FS)

    # ── Draw title ────────────────────────────────────────────────────────────
    draw.text((MARGIN_X, MARGIN_Y), title, fill=C_TITLE, font=font_title)
    try:
        title_h = font_title.getbbox(title)[3]
    except AttributeError:
        title_h = TITLE_FS
    y_body = MARGIN_Y + title_h + LINE_GAP + 4

    # ── Wrap and draw body ────────────────────────────────────────────────────
    # Estimate character width for wrapping
    try:
        char_w = font_body.getlength("n")
    except AttributeError:
        char_w = BODY_FS * 0.55
    chars_per_line = max(40, int((W - 2 * MARGIN_X) / max(char_w, 1)))

    import textwrap as _tw
    lines = _tw.wrap(body, width=chars_per_line)

    try:
        line_h = font_body.getbbox("Ag")[3] + LINE_GAP
    except AttributeError:
        line_h = BODY_FS + LINE_GAP

    for line in lines:
        if y_body + line_h > STRIP_H - 8:
            draw.text((MARGIN_X, y_body), "…", fill=C_BODY, font=font_body)
            break
        draw.text((MARGIN_X, y_body), line, fill=C_BODY, font=font_body)
        y_body += line_h

    # ── Composite and save ────────────────────────────────────────────────────
    combined = Image.new("RGB", (W, H + STRIP_H), C_BG)
    combined.paste(orig,  (0, 0))
    combined.paste(strip, (0, H))
    combined.save(filepath, dpi=(dpi, dpi))


# =============================================================================
# CELL 4 — PLOT THEME
# =============================================================================

plt.style.use("dark_background")
BG,AX_BG,SPINE,TEXT,MUTED = "#0d1117","#161b22","#30363d","#e6edf3","#8b949e"
LAE_COL,AGN_COL,BLOB_COL  = "#58a6ff","#f78166","#d2a8ff"
ALL_COL  = "#ffa657"

FIELD_COLORS = {"dex-spring":"#58a6ff","dex-fall":"#3fb950",
                "cosmos":"#f78166","goods-n":"#d2a8ff",
                "nep":"#ffa657","ssa22":"#79c0ff"}

def sax(ax, title="", xl="", yl="", minor=True):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=MUTED,which="both",direction="in",
                   top=True,right=True,labelsize=8.5)
    if xl: ax.set_xlabel(xl, color=TEXT, fontsize=9)
    if yl: ax.set_ylabel(yl, color=TEXT, fontsize=9)
    if title: ax.set_title(title,color=TEXT,fontsize=9.5,
                           fontweight="bold",loc="left",pad=4)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

def leg(ax,**kw):
    return ax.legend(fontsize=7.5,facecolor="#21262d",
                     edgecolor=SPINE,labelcolor=TEXT,**kw)

def cb(fig,ax,im,label,fs=7.5):
    c=fig.colorbar(im,ax=ax,fraction=0.030,pad=0.02,shrink=0.92)
    c.set_label(label,color=MUTED,fontsize=fs)
    c.ax.yaxis.set_tick_params(color=MUTED,labelsize=7)
    plt.setp(c.ax.yaxis.get_ticklabels(),color=MUTED)
    c.outline.set_edgecolor(SPINE)
    return c

syn = "  [SYNTHETIC]" if SYNTHETIC else ""

# =============================================================================
# MODULE A — TOTAL POPULATION
# =============================================================================
print("\n--- Module A: Population ---")
fig,axes = plt.subplots(2,4,figsize=(22,10))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.40,wspace=0.30,left=0.06,right=0.97,
                    top=0.91,bottom=0.09)

# A1: r_iso histogram with dBIC tiers
ax = axes[0,0]; sax(ax,"A1: Size distribution by dBIC tier",
                    r"$r_{\rm iso}$ [kpc]","Normalised density")
r_bins = np.logspace(np.log10(5),np.log10(200),45)
for mask,color,lbl in [
    (sel,         ALL_COL, f"All  dBIC≥2  (N={sel.sum():,})"),
    (sel_strong,  TEXT,    f"Strong  dBIC≥6  (N={sel_strong.sum():,})"),
]:
    h,e=np.histogram(r_iso[mask&np.isfinite(r_iso)],bins=r_bins,density=True)
    ax.step(e[:-1],h,where="post",color=color,lw=1.6,alpha=0.9,label=lbl)
    ax.fill_between(e[:-1],0,h,step="post",color=color,alpha=0.14)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# A2: logL distribution
ax = axes[0,1]; sax(ax,"A2: Luminosity distribution",
                    r"$\log_{10}L_{\rm Ly\alpha}$ [erg/s]","Density")
lbins=np.linspace(42,45.5,45)
for mask,color,lbl in [(sel_lae,LAE_COL,f"LAE N={sel_lae.sum():,}"),
                        (sel_agn,AGN_COL,f"AGN N={sel_agn.sum():,}")]:
    h,e=np.histogram(logL[mask&np.isfinite(logL)],bins=lbins,density=True)
    ax.step(e[:-1],h,where="post",color=color,lw=1.6,alpha=0.9,label=lbl)
    ax.fill_between(e[:-1],0,h,step="post",color=color,alpha=0.14)
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# A3: z distribution
ax = axes[0,2]; sax(ax,"A3: Redshift distribution",
                    r"Redshift $z$","Density")
zbins=np.linspace(Z_MIN,Z_MAX,45)
for mask,color,lbl in [(sel_lae,LAE_COL,"LAE"),(sel_agn,AGN_COL,"AGN")]:
    h,e=np.histogram(z[mask&np.isfinite(z)],bins=zbins,density=True)
    ax.step(e[:-1],h,where="post",color=color,lw=1.6,alpha=0.85,label=lbl)
# Dual x-axis: z + lookback time
ax2=ax.twiny(); ax2.set_xlim(ax.get_xlim())
zt=[2.0,2.5,3.0,3.5]; ax2.set_xticks(zt)
ax2.set_xticklabels([f"{lb(z_):.1f}" for z_ in zt],color=MUTED,fontsize=7.5)
ax2.set_xlabel("Look-back time (Gyr)",color=MUTED,fontsize=8)
ax2.spines["top"].set_color(SPINE); ax2.tick_params(colors=MUTED,labelsize=7.5)
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# A4: EW distribution
ax = axes[0,3]; sax(ax,"A4: Rest-frame EW distribution",
                    r"EW$_{\rm rest}$ [Å]","Density")
ok_ew = sel&np.isfinite(ew)&(ew>0)
ew_bins=np.logspace(np.log10(5),np.log10(500),45)
for mask,color,lbl in [(sel_lae&ok_ew,LAE_COL,"LAE"),(sel_agn&ok_ew,AGN_COL,"AGN")]:
    if mask.sum()<5: continue
    h,e=np.histogram(ew[mask],bins=ew_bins,density=True)
    ax.step(e[:-1],h,where="post",color=color,lw=1.5,alpha=0.85,label=lbl)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# A5: r_iso vs r_s (profile shape)
ax = axes[1,0]; sax(ax,"A5: Profile shape — r_s vs r_iso",
                    r"$r_{\rm iso}$ [kpc]",r"$r_s$ [kpc]")
ok_rs=sel&np.isfinite(r_s)&np.isfinite(r_iso)&(r_s>0)
ax.scatter(r_iso[ok_rs&sel_lae],r_s[ok_rs&sel_lae],
           s=2,c=LAE_COL,alpha=0.25,linewidths=0,rasterized=True,label="LAE")
ax.scatter(r_iso[ok_rs&sel_agn],r_s[ok_rs&sel_agn],
           s=3,c=AGN_COL,alpha=0.30,linewidths=0,rasterized=True,label="AGN")
rr=np.logspace(np.log10(5),np.log10(200),100)
for ratio,ls,lbl in [(3,"--","1:3"),(5,":","1:5")]:
    ax.plot(rr,rr/ratio,ls,color=MUTED,lw=0.9,alpha=0.6,label=lbl)
med_r=np.nanmedian(r_iso[ok_rs]/r_s[ok_rs])
ax.text(0.97,0.05,f"Median r_iso/r_s = {med_r:.1f}",
        transform=ax.transAxes,color=TEXT,fontsize=8.5,ha="right",va="bottom",
        bbox=dict(boxstyle="round,pad=0.3",facecolor=BG,edgecolor=SPINE,alpha=0.8))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(4,200); ax.set_ylim(0.5,60)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
ax.legend(fontsize=7,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper left")

# A6: dBIC vs r_iso
ax = axes[1,1]; sax(ax,"A6: Extended-source evidence vs size",
                    r"$r_{\rm iso}$ [kpc]",r"$\Delta$BIC")
ax.scatter(r_iso[sel_lae&np.isfinite(dBIC)],dBIC[sel_lae&np.isfinite(dBIC)],
           s=1.5,c=LAE_COL,alpha=0.20,linewidths=0,rasterized=True,label="LAE")
ax.scatter(r_iso[sel_agn&np.isfinite(dBIC)],dBIC[sel_agn&np.isfinite(dBIC)],
           s=2,c=AGN_COL,alpha=0.25,linewidths=0,rasterized=True,label="AGN")
ax.axhline(DBIC_STRONG,color="#ffa657",lw=0.9,ls="--",alpha=0.65,
           label=f"dBIC={DBIC_STRONG}")
ax.set_xscale("log"); ax.set_xlim(4,200)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper left",markerscale=4)

# A7: per-field bar chart
ax = axes[1,2]; sax(ax,"A7: LAN count per field","Field","N LANs",minor=False)
flds=["dex-spring","dex-fall","cosmos","goods-n","nep","ssa22"]
lae_c=[((sel_lae)&(field==f)).sum() for f in flds]
agn_c=[((sel_agn)&(field==f)).sum() for f in flds]
x=np.arange(len(flds)); w=0.38
b1=ax.bar(x-w/2,lae_c,w,color=LAE_COL,alpha=0.80,label="LAE",edgecolor=SPINE,lw=0.4)
b2=ax.bar(x+w/2,agn_c,w,color=AGN_COL,alpha=0.80,label="AGN",edgecolor=SPINE,lw=0.4)
ax.set_xticks(x)
ax.set_xticklabels([f.replace("dex-","") for f in flds],
                   color=TEXT,fontsize=8,rotation=20,ha="right")
for tick,f in zip(ax.get_xticklabels(),flds):
    tick.set_color(FIELD_COLORS.get(f,TEXT))
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# A8: KS test summary table
ax = axes[1,3]; ax.set_facecolor(AX_BG)
for sp in ax.spines.values(): sp.set_color(SPINE)
ax.set_title("A8: LAE vs AGN KS test summary",color=TEXT,fontsize=9.5,
             fontweight="bold",loc="left",pad=4)
ax.set_xticks([]); ax.set_yticks([])
tests = [
    ("r_iso [kpc]",   r_iso, r"$r_{\rm iso}$"),
    ("logL_lya",      logL,  r"$\log L$"),
    ("EW_rest [Å]",   ew,    "EW"),
    ("dBIC",          dBIC,  r"$\Delta$BIC"),
    ("z_hetdex",      z,     "$z$"),
    ("r_s [kpc]",     r_s,   r"$r_s$"),
]
rows=[["Property","D","p-value","Different?"]]
for name, arr, label in tests:
    a=arr[sel_lae&np.isfinite(arr)]; b=arr[sel_agn&np.isfinite(arr)]
    if len(a)<5 or len(b)<5: continue
    D,p=ks_2samp(a,b)
    rows.append([label,f"{D:.3f}",f"{p:.2e}","YES" if p<0.01 else "no"])
col_x=[0.02,0.32,0.57,0.80]; row_y=0.92
ax.text(col_x[0],row_y,"Property",transform=ax.transAxes,
        color=TEXT,fontsize=8.5,fontweight="bold",va="top")
for j,hdr in enumerate(["D","p-value","Diff?"]):
    ax.text(col_x[j+1],row_y,hdr,transform=ax.transAxes,
            color=TEXT,fontsize=8.5,fontweight="bold",va="top",ha="right")
ax.axhline(row_y-0.04,color=SPINE,lw=0.8)
for i,row in enumerate(rows[1:]):
    ry=row_y-0.12*(i+1)
    vc=("#3fb950" if row[3]=="YES" else MUTED)
    ax.text(col_x[0],ry,row[0],transform=ax.transAxes,
            color=TEXT,fontsize=8,va="top")
    for j in range(1,4):
        ax.text(col_x[j],ry,row[j],transform=ax.transAxes,
                color=vc if j==3 else MUTED,fontsize=8,va="top",ha="right")

fig.suptitle(f"HETDEX LANs — A: Total Population  (N={sel.sum():,}){syn}",
             color=TEXT,fontsize=13,fontweight="bold",y=0.975)
save_with_caption(fig, OUT["A"], "A")
print(f"  Saved {OUT['A']}")
plt.close(fig)

# =============================================================================
# MODULE B — LAE-HOSTED NEBULAE (N=5,714)
# =============================================================================
print("--- Module B: LAE-hosted ---")
fig,axes = plt.subplots(2,4,figsize=(22,10))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.42,wspace=0.30,left=0.06,right=0.97,
                    top=0.91,bottom=0.09)

# B1: Size-luminosity hexbin
ax = axes[0,0]; sax(ax,f"B1: LAE size–luminosity (N={sel_lae.sum():,})",
                    r"$\log L_{\rm Ly\alpha}$",r"$r_{\rm iso}$ [kpc]")
hb=ax.hexbin(logL[sel_lae],r_iso[sel_lae],gridsize=35,mincnt=1,
             cmap="Blues",bins="log",xscale="linear",yscale="log",
             alpha=0.90,rasterized=True)
a,c_,r,ae=pl_fit(logL,r_iso,sel_lae)
xl=np.linspace(logL[sel_lae].min()-.1,logL[sel_lae].max()+.1,200)
ax.plot(xl,10.**np.polyval([a,c_],xl),"--",color=TEXT,lw=1.8,
        label=rf"$\alpha$={a:.2f}  r={r:.2f}")
ax.set_yscale("log"); ax.set_ylim(4,100)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
cb(fig,ax,hb,r"$\log_{10}$ N")
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper left")

# B2: r_iso vs z (LAE only, coloured by logL)
ax = axes[0,1]; sax(ax,"B2: LAE nebula size vs redshift",
                    r"Redshift $z$",r"$r_{\rm iso}$ [kpc]")
ok=sel_lae&np.isfinite(r_iso)&np.isfinite(z)
sc=ax.scatter(z[ok],r_iso[ok],c=logL[ok],cmap="plasma",
              vmin=42,vmax=45,s=2.5,alpha=0.35,linewidths=0,rasterized=True)
# Median track
zgr=np.linspace(Z_MIN,Z_MAX,40)
zm_med=[np.nanmedian(r_iso[ok&(z>=zgr[i])&(z<zgr[i+1])])
        for i in range(len(zgr)-1) if (ok&(z>=zgr[i])&(z<zgr[i+1])).sum()>=10]
zc=[0.5*(zgr[i]+zgr[i+1])
    for i in range(len(zgr)-1) if (ok&(z>=zgr[i])&(z<zgr[i+1])).sum()>=10]
ax.plot(zc,zm_med,"-",color=TEXT,lw=2.0,alpha=0.85,label="Median")
ax.set_yscale("log"); ax.set_ylim(4,100)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
cb(fig,ax,sc,r"$\log L_{\rm Ly\alpha}$")
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# B3: EW vs z (LAE)
ax = axes[0,2]; sax(ax,"B3: LAE EW vs redshift",
                    r"Redshift $z$",r"EW$_{\rm rest}$ [Å]")
ok_ew=sel_lae&np.isfinite(ew)&np.isfinite(z)&(ew>0)
sc2=ax.scatter(z[ok_ew],ew[ok_ew],c=logL[ok_ew],cmap="plasma",
               vmin=42,vmax=45,s=2.5,alpha=0.35,linewidths=0,rasterized=True)
ew_med=[np.nanmedian(ew[ok_ew&(z>=zgr[i])&(z<zgr[i+1])])
        for i in range(len(zgr)-1) if (ok_ew&(z>=zgr[i])&(z<zgr[i+1])).sum()>=10]
zc_ew=[0.5*(zgr[i]+zgr[i+1])
       for i in range(len(zgr)-1) if (ok_ew&(z>=zgr[i])&(z<zgr[i+1])).sum()>=10]
ax.plot(zc_ew,ew_med,"-",color=TEXT,lw=2.0,alpha=0.85,label="Median EW")
ax.set_yscale("log"); ax.set_ylim(5,500)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
cb(fig,ax,sc2,r"$\log L$")
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# B4: gmag vs logL (LAE)
ax = axes[0,3]; sax(ax,"B4: LAE continuum magnitude vs luminosity",
                    r"$\log L_{\rm Ly\alpha}$","g-band magnitude")
ok_g=sel_lae&np.isfinite(gmag)&np.isfinite(logL)
ax.scatter(logL[ok_g],gmag[ok_g],s=2.5,c=LAE_COL,alpha=0.30,
           linewidths=0,rasterized=True)
if ok_g.sum()>10:
    c_gm=np.polyfit(logL[ok_g],gmag[ok_g],1)
    xl_g=np.linspace(logL[ok_g].min()-.1,logL[ok_g].max()+.1,100)
    ax.plot(xl_g,np.polyval(c_gm,xl_g),"--",color=TEXT,lw=1.4,alpha=0.75,
            label=f"slope={c_gm[0]:.2f}")
ax.invert_yaxis()
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# B5: SB sensitivity vs r_iso (LAE)
ax = axes[1,0]; sax(ax,"B5: LAE — surface brightness limit vs size",
                    r"$r_{\rm iso}$ [kpc]",r"SB$_{1\sigma}$ [10$^{-18}$ cgs arcsec$^{-2}$]")
ok_sb=sel_lae&np.isfinite(sb)&np.isfinite(r_iso)
ax.scatter(r_iso[ok_sb],sb[ok_sb],s=2,c=LAE_COL,alpha=0.25,
           linewidths=0,rasterized=True)
# Median SB per r_iso bin
rbins_sb=np.logspace(np.log10(5),np.log10(100),20)
rm_sb,re_sb=[],[]
for i in range(len(rbins_sb)-1):
    m=ok_sb&(r_iso>=rbins_sb[i])&(r_iso<rbins_sb[i+1])
    if m.sum()<5: continue
    rm_sb.append(np.nanmedian(sb[m])); re_sb.append(0.5*(rbins_sb[i]+rbins_sb[i+1]))
ax.plot(re_sb,rm_sb,"o-",color=TEXT,ms=5,lw=1.3,alpha=0.85,label="Median")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# B6: chi2 extended vs PSF (LAE)
ax = axes[1,1]; sax(ax,r"B6: LAE — $\chi^2$ extended vs PSF",
                    r"$\chi^2_{\rm PSF}$",r"$\chi^2_{\rm ext}$")
ok_c=sel_lae&np.isfinite(chi2e)&np.isfinite(chi2p)
ax.scatter(chi2p[ok_c],chi2e[ok_c],s=2,c=dBIC[ok_c],
           cmap="RdYlGn",vmin=-10,vmax=50,
           alpha=0.30,linewidths=0,rasterized=True)
c_range=np.linspace(0,chi2p[ok_c].max()*0.8,100)
ax.plot(c_range,c_range,"--",color=MUTED,lw=0.9,alpha=0.6,label="y=x")
ax.set_xlim(0,8); ax.set_ylim(0,8)
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper left")

# B7: r_iso distribution LAE vs field
ax = axes[1,2]; sax(ax,"B7: LAE size per survey field",
                    r"$r_{\rm iso}$ [kpc]","Normalised density")
r_bins_f=np.logspace(np.log10(5),np.log10(150),35)
for f in ["dex-spring","dex-fall","cosmos","goods-n"]:
    fm=sel_lae&(field==f)
    if fm.sum()<10: continue
    h,e=np.histogram(r_iso[fm&np.isfinite(r_iso)],bins=r_bins_f,density=True)
    ax.step(e[:-1],h,where="post",color=FIELD_COLORS.get(f,MUTED),
            lw=1.3,alpha=0.85,label=f"{f} (N={fm.sum():,})")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
ax.legend(fontsize=7,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# B8: statistics table LAE
ax = axes[1,3]; ax.set_facecolor(AX_BG)
for sp in ax.spines.values(): sp.set_color(SPINE)
ax.set_title("B8: LAE nebula statistics",color=TEXT,fontsize=9.5,
             fontweight="bold",loc="left",pad=4)
ax.set_xticks([]); ax.set_yticks([])
stats_rows = [
    ("N LAE LANs",       f"{sel_lae.sum():,}"),
    ("Median r_iso",     f"{np.nanmedian(r_iso[sel_lae]):.1f} kpc"),
    ("Max r_iso",        f"{np.nanmax(r_iso[sel_lae]):.1f} kpc"),
    ("Median logL",      f"{np.nanmedian(logL[sel_lae]):.3f}"),
    ("Median EW_rest",   f"{np.nanmedian(ew[sel_lae&np.isfinite(ew)]):.1f} Å"),
    ("Median z",         f"{np.nanmedian(z[sel_lae]):.3f}"),
    ("Median r_iso/r_s", f"{np.nanmedian(r_iso[sel_lae&np.isfinite(r_s)]/r_s[sel_lae&np.isfinite(r_s)]):.2f}"),
    ("L-L slope α",      f"{pl_fit(logL,r_iso,sel_lae)[0]:.3f}"),
    ("Pearson r",        f"{pl_fit(logL,r_iso,sel_lae)[2]:.3f}"),
]
for i,(k,v) in enumerate(stats_rows):
    y_=0.92-i*0.095
    ax.text(0.04,y_,k,transform=ax.transAxes,color=MUTED,fontsize=8.5,va="top")
    ax.text(0.96,y_,v,transform=ax.transAxes,color=TEXT, fontsize=8.5,va="top",ha="right")

fig.suptitle(f"HETDEX LANs — B: LAE-Hosted Nebulae  (N={sel_lae.sum():,}){syn}",
             color=TEXT,fontsize=13,fontweight="bold",y=0.975)
save_with_caption(fig, OUT["B"], "B")
print(f"  Saved {OUT['B']}")
plt.close(fig)

# =============================================================================
# MODULE C — AGN-HOSTED NEBULAE (N=1,492)
# =============================================================================
print("--- Module C: AGN-hosted ---")
fig,axes = plt.subplots(2,4,figsize=(22,10))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.42,wspace=0.30,left=0.06,right=0.97,
                    top=0.91,bottom=0.09)

# C1: AGN size-luminosity
ax = axes[0,0]; sax(ax,f"C1: AGN size–luminosity (N={sel_agn.sum():,})",
                    r"$\log L_{\rm Ly\alpha}$",r"$r_{\rm iso}$ [kpc]")
hb=ax.hexbin(logL[sel_agn],r_iso[sel_agn],gridsize=30,mincnt=1,
             cmap="Reds",bins="log",yscale="log",alpha=0.90,rasterized=True)
a_a,c_a,r_a,ae_a=pl_fit(logL,r_iso,sel_agn)
xl=np.linspace(logL[sel_agn].min()-.1,logL[sel_agn].max()+.1,200)
ax.plot(xl,10.**np.polyval([a_a,c_a],xl),"--",color=TEXT,lw=1.8,
        label=rf"$\alpha$={a_a:.2f}  r={r_a:.2f}")
# LAE fit for comparison
a_l,c_l,_,_=pl_fit(logL,r_iso,sel_lae)
ax.plot(xl,10.**np.polyval([a_l,c_l],xl),":",color=LAE_COL,lw=1.2,alpha=0.65,
        label=rf"LAE $\alpha$={a_l:.2f}")
ax.set_yscale("log"); ax.set_ylim(4,200)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
cb(fig,ax,hb,r"$\log_{10}$ N")
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper left")

# C2: AGN r_iso vs z
ax = axes[0,1]; sax(ax,"C2: AGN nebula size vs redshift",
                    r"Redshift $z$",r"$r_{\rm iso}$ [kpc]")
ok_a=sel_agn&np.isfinite(r_iso)&np.isfinite(z)
sc=ax.scatter(z[ok_a],r_iso[ok_a],c=logL[ok_a],cmap="hot",
              vmin=42.5,vmax=45.5,s=5,alpha=0.50,linewidths=0,rasterized=True)
zm_a=[np.nanmedian(r_iso[ok_a&(z>=zgr[i])&(z<zgr[i+1])])
      for i in range(len(zgr)-1) if (ok_a&(z>=zgr[i])&(z<zgr[i+1])).sum()>=5]
zca=[0.5*(zgr[i]+zgr[i+1])
     for i in range(len(zgr)-1) if (ok_a&(z>=zgr[i])&(z<zgr[i+1])).sum()>=5]
ax.plot(zca,zm_a,"-",color=TEXT,lw=2.0,alpha=0.85,label="Median")
ax.set_yscale("log"); ax.set_ylim(4,200)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
cb(fig,ax,sc,r"$\log L$")
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# C3: AGN vs LAE r_iso comparison (violin-style)
ax = axes[0,2]; sax(ax,"C3: AGN vs LAE size comparison",
                    "Source type",r"$r_{\rm iso}$ [kpc]",minor=False)
data_v={"LAE":r_iso[sel_lae&np.isfinite(r_iso)],
        "AGN":r_iso[sel_agn&np.isfinite(r_iso)]}
vp=ax.violinplot([data_v["LAE"],data_v["AGN"]],positions=[0,1],
                 showmedians=True,showextrema=False)
for pc,col in zip(vp["bodies"],[LAE_COL,AGN_COL]):
    pc.set_facecolor(col); pc.set_alpha(0.40)
vp["cmedians"].set_color(TEXT); vp["cmedians"].set_lw(2)
ax.set_xticks([0,1]); ax.set_xticklabels(["LAE","AGN"],color=TEXT,fontsize=10)
ax.set_yscale("log"); ax.set_ylim(4,200)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
med_l=np.nanmedian(data_v["LAE"]); med_a=np.nanmedian(data_v["AGN"])
ax.text(0,med_l*1.08,f"{med_l:.1f}kpc",ha="center",color=LAE_COL,fontsize=8)
ax.text(1,med_a*1.08,f"{med_a:.1f}kpc",ha="center",color=AGN_COL,fontsize=8)
D,p=ks_2samp(data_v["LAE"],data_v["AGN"])
ax.text(0.5,0.96,f"KS D={D:.3f}  p={p:.1e}",transform=ax.transAxes,
        ha="center",va="top",color=TEXT,fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3",facecolor=BG,edgecolor=SPINE,alpha=0.85))

# C4: AGN EW vs logL
ax = axes[0,3]; sax(ax,"C4: AGN EW vs luminosity",
                    r"$\log L_{\rm Ly\alpha}$",r"EW$_{\rm rest}$ [Å]")
ok_ae=sel_agn&np.isfinite(ew)&np.isfinite(logL)&(ew>0)
sc4=ax.scatter(logL[ok_ae],ew[ok_ae],c=z[ok_ae],cmap="coolwarm",
               vmin=Z_MIN,vmax=Z_MAX,s=6,alpha=0.55,linewidths=0,rasterized=True)
if ok_ae.sum()>10:
    c_ew=np.polyfit(logL[ok_ae],np.log10(ew[ok_ae]),1)
    xl_ew=np.linspace(logL[ok_ae].min()-.1,logL[ok_ae].max()+.1,100)
    ax.plot(xl_ew,10.**np.polyval(c_ew,xl_ew),"--",color=TEXT,lw=1.4,
            label=f"slope={c_ew[0]:.2f}")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
cb(fig,ax,sc4,"Redshift z")
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# C5: AGN blob fraction vs logL
ax = axes[1,0]; sax(ax,"C5: AGN blob fraction vs logL bin",
                    r"$\log L_{\rm Ly\alpha}$","Fraction with r_iso≥50 kpc")
lbins_blob=np.linspace(42,45.5,12)
blob_frac,blob_err,lc=[],[],[]
for i in range(len(lbins_blob)-1):
    m=sel_agn&(logL>=lbins_blob[i])&(logL<lbins_blob[i+1])&np.isfinite(r_iso)
    if m.sum()<5: continue
    n_b=(m&(r_iso>=BLOB_R_ISO)).sum(); n_t=m.sum()
    f=n_b/n_t; e=np.sqrt(f*(1-f)/n_t)
    blob_frac.append(f); blob_err.append(e)
    lc.append(0.5*(lbins_blob[i]+lbins_blob[i+1]))
ax.errorbar(lc,blob_frac,yerr=blob_err,fmt="s-",color=AGN_COL,
            ms=6,lw=1.5,capsize=3,elinewidth=1.0)
ax.set_ylim(-0.01,max(0.1,max(blob_frac)*1.3) if blob_frac else 0.1)
ax.axhline(0,color=SPINE,lw=0.7,ls=":")

# C6: AGN r_iso / r_s ratio
ax = axes[1,1]; sax(ax,"C6: AGN profile concentration r_iso/r_s",
                    r"$r_{\rm iso}/r_s$","Density")
ok_cr=sel_agn&np.isfinite(r_iso)&np.isfinite(r_s)&(r_s>0)
ratio_a=r_iso[ok_cr]/r_s[ok_cr]
ratio_l_ok=sel_lae&np.isfinite(r_iso)&np.isfinite(r_s)&(r_s>0)
ratio_l=r_iso[ratio_l_ok]/r_s[ratio_l_ok]
ratio_bins=np.logspace(np.log10(1),np.log10(20),40)
for rat,color,lbl in [(ratio_l,LAE_COL,"LAE"),(ratio_a,AGN_COL,"AGN")]:
    h,e=np.histogram(rat[rat>0],bins=ratio_bins,density=True)
    ax.step(e[:-1],h,where="post",color=color,lw=1.4,alpha=0.85,label=lbl)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:.1f}"))
ax.axvline(np.nanmedian(ratio_a),color=AGN_COL,lw=1.0,ls="--",alpha=0.7)
ax.axvline(np.nanmedian(ratio_l),color=LAE_COL,lw=1.0,ls="--",alpha=0.7)
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# C7: AGN log10_pF distribution
ax = axes[1,2]; sax(ax,"C7: Extended-source log probability",
                    r"$\log_{10} p_F$","Density")
ok_pf=np.isfinite(log10_pF)
for mask,color,lbl in [(sel_lae&ok_pf,LAE_COL,"LAE"),(sel_agn&ok_pf,AGN_COL,"AGN")]:
    h,e=np.histogram(log10_pF[mask],bins=50,density=True)
    ax.step(e[:-1],h,where="post",color=color,lw=1.4,alpha=0.85,label=lbl)
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper left")

# C8: AGN statistics table
ax = axes[1,3]; ax.set_facecolor(AX_BG)
for sp in ax.spines.values(): sp.set_color(SPINE)
ax.set_title("C8: AGN nebula statistics",color=TEXT,fontsize=9.5,
             fontweight="bold",loc="left",pad=4)
ax.set_xticks([]); ax.set_yticks([])
stats_c=[
    ("N AGN LANs",       f"{sel_agn.sum():,}"),
    ("Fraction of total",f"{100*sel_agn.sum()/max(sel.sum(),1):.1f}%"),
    ("Median r_iso",     f"{np.nanmedian(r_iso[sel_agn]):.1f} kpc"),
    ("Max r_iso",        f"{np.nanmax(r_iso[sel_agn]):.1f} kpc"),
    ("Median logL",      f"{np.nanmedian(logL[sel_agn]):.3f}"),
    ("Median z",         f"{np.nanmedian(z[sel_agn]):.3f}"),
    ("AGN blob count",   f"{sel_blob.sum():,}"),
    ("L-L slope α",      f"{a_a:.3f} ± {ae_a:.3f}"),
    ("vs LAE α",         f"{a_l:.3f}  (Δ={(a_a-a_l):+.3f})"),
]
for i,(k,v) in enumerate(stats_c):
    y_=0.92-i*0.095
    ax.text(0.04,y_,k,transform=ax.transAxes,color=MUTED,fontsize=8.5,va="top")
    ax.text(0.96,y_,v,transform=ax.transAxes,color=TEXT, fontsize=8.5,va="top",ha="right")

fig.suptitle(f"HETDEX LANs — C: AGN-Hosted Nebulae  (N={sel_agn.sum():,}){syn}",
             color=TEXT,fontsize=13,fontweight="bold",y=0.975)
save_with_caption(fig, OUT["C"], "C")
print(f"  Saved {OUT['C']}")
plt.close(fig)

# =============================================================================
# MODULE D — BLOBS  (r_iso ≥ 50 kpc)
# =============================================================================
print("--- Module D: Blobs ---")
fig,axes = plt.subplots(2,3,figsize=(18,10))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.44,wspace=0.32,left=0.07,right=0.97,
                    top=0.91,bottom=0.09)

n_blobs = sel_blob.sum()
blob_idx = np.where(sel_blob)[0]

# D1: Blob gallery scatter — each blob annotated
ax = axes[0,0]; sax(ax,f"D1: Blob catalogue  (N={n_blobs})",
                    r"$\log L_{\rm Ly\alpha}$",r"$r_{\rm iso}$ [kpc]")
ax.scatter(logL[sel&~sel_blob],r_iso[sel&~sel_blob],
           s=1.5,c=MUTED,alpha=0.15,linewidths=0,rasterized=True,label="LANs")
for ii,bi in enumerate(blob_idx):
    col=AGN_COL if stype[bi]=="agn" else LAE_COL
    mk="*" if stype[bi]=="agn" else "o"
    ax.scatter(logL[bi],r_iso[bi],s=60,c=col,marker=mk,
               edgecolors="white",linewidths=0.5,zorder=5)
    ax.annotate(f"#{ii+1}\n{r_iso[bi]:.0f}kpc",
                xy=(logL[bi],r_iso[bi]),
                xytext=(logL[bi]+0.05,r_iso[bi]*1.08),
                fontsize=6.5,color=col,
                arrowprops=dict(arrowstyle="-",color=col,lw=0.5))
ax.axhline(BLOB_R_ISO,color=BLOB_COL,lw=0.9,ls="--",alpha=0.65,
           label=f"{BLOB_R_ISO} kpc threshold")
ax.set_yscale("log"); ax.set_ylim(4,200)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
lae_h=Line2D([0],[0],marker="o",color=LAE_COL,ms=7,lw=0,label="LAE blob")
agn_h=Line2D([0],[0],marker="*",color=AGN_COL,ms=9,lw=0,label="AGN blob")
ax.legend(handles=[lae_h,agn_h]+ax.get_legend_handles_labels()[0][1:],
          fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT)

# D2: Blob redshift distribution
ax = axes[0,1]; sax(ax,"D2: Blob redshift distribution",
                    r"Redshift $z$","N blobs")
if n_blobs >= 3:
    zbins_b=np.linspace(Z_MIN,Z_MAX,12)
    ax.hist(z[sel_blob],bins=zbins_b,color=BLOB_COL,alpha=0.75,edgecolor=BG,lw=0.5)
ax.axvspan(2.0,3.0,color="#ffa657",alpha=0.08,label="Cosmic noon")
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# D3: Blob r_iso vs logL table
ax = axes[0,2]; ax.set_facecolor(AX_BG)
for sp in ax.spines.values(): sp.set_color(SPINE)
ax.set_title(f"D3: Blob catalogue table  (N={n_blobs})",color=TEXT,
             fontsize=9.5,fontweight="bold",loc="left",pad=4)
ax.set_xticks([]); ax.set_yticks([])
cols_h=["#","Type","r_iso","logL","z"]
cx=[0.02,0.16,0.38,0.58,0.78]
for j,(hdr,x_) in enumerate(zip(cols_h,cx)):
    ax.text(x_,0.94,hdr,transform=ax.transAxes,color=TEXT,
            fontsize=8.5,fontweight="bold",va="top")
ax.axhline(0.91,color=SPINE,lw=0.8)
sorted_blobs=blob_idx[np.argsort(r_iso[blob_idx])[::-1]]
for ii,bi in enumerate(sorted_blobs[:12]):
    y_=0.88-ii*0.075
    col=AGN_COL if stype[bi]=="agn" else LAE_COL
    vals=[f"{ii+1}",stype[bi].upper(),
          f"{r_iso[bi]:.1f}kpc",f"{logL[bi]:.2f}",f"{z[bi]:.3f}"]
    for j,(v,x_) in enumerate(zip(vals,cx)):
        ax.text(x_,y_,v,transform=ax.transAxes,
                color=col if j<=1 else MUTED,fontsize=7.5,va="top")

# D4: Blob r_iso vs EW
ax = axes[1,0]; sax(ax,"D4: Blob size vs equivalent width",
                    r"EW$_{\rm rest}$ [Å]",r"$r_{\rm iso}$ [kpc]")
ax.scatter(ew[sel&~sel_blob&np.isfinite(ew)],r_iso[sel&~sel_blob&np.isfinite(ew)],
           s=1.5,c=MUTED,alpha=0.15,linewidths=0,rasterized=True)
if n_blobs>0:
    ok_be=sel_blob&np.isfinite(ew)
    for bi in blob_idx[ok_be[blob_idx]]:
        col=AGN_COL if stype[bi]=="agn" else LAE_COL
        ax.scatter(ew[bi],r_iso[bi],s=60,c=col,
                   edgecolors="white",linewidths=0.5,zorder=5)
ax.axhline(BLOB_R_ISO,color=BLOB_COL,lw=0.9,ls="--",alpha=0.65)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(4,200)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))

# D5: Blob context — r_iso vs LAN population percentile
ax = axes[1,1]; sax(ax,"D5: Blob size in context of full population",
                    r"$r_{\rm iso}$ [kpc]","Cumulative fraction")
r_sorted=np.sort(r_iso[sel&np.isfinite(r_iso)])
ax.plot(r_sorted,np.arange(1,len(r_sorted)+1)/len(r_sorted),
        color=ALL_COL,lw=1.8,alpha=0.90,label="All LANs")
pct_blob=np.searchsorted(r_sorted,BLOB_R_ISO)/len(r_sorted)
ax.axvline(BLOB_R_ISO,color=BLOB_COL,lw=1.0,ls="--",alpha=0.75,
           label=f"{BLOB_R_ISO} kpc  ({100*(1-pct_blob):.1f}% of LANs)")
ax.axhline(pct_blob,color=BLOB_COL,lw=0.7,ls=":",alpha=0.55)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{int(x)}"))
ax.set_ylim(0,1.02)
ax.text(BLOB_R_ISO*1.05,0.15,f"Top {100*(1-pct_blob):.1f}%",
        color=BLOB_COL,fontsize=9,va="bottom")
ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper left")

# D6: Blob statistics
ax = axes[1,2]; ax.set_facecolor(AX_BG)
for sp in ax.spines.values(): sp.set_color(SPINE)
ax.set_title("D6: Blob population statistics",color=TEXT,fontsize=9.5,
             fontweight="bold",loc="left",pad=4)
ax.set_xticks([]); ax.set_yticks([])
n_blob_lae=(sel_blob&(stype=="lae")).sum()
n_blob_agn=(sel_blob&(stype=="agn")).sum()
stats_d=[
    ("Total blobs",         f"{n_blobs}"),
    ("LAE-hosted",          f"{n_blob_lae}"),
    ("AGN-hosted",          f"{n_blob_agn}"),
    ("AGN fraction",        f"{100*n_blob_agn/max(n_blobs,1):.0f}%"),
    ("Size threshold",      f"{BLOB_R_ISO:.0f} kpc"),
    ("Median r_iso (blobs)",f"{np.nanmedian(r_iso[sel_blob]):.1f} kpc" if n_blobs>0 else "N/A"),
    ("Max r_iso",           f"{np.nanmax(r_iso[sel_blob]):.1f} kpc" if n_blobs>0 else "N/A"),
    ("Median logL (blobs)", f"{np.nanmedian(logL[sel_blob]):.2f}" if n_blobs>0 else "N/A"),
    ("Fraction of LANs",    f"{100*n_blobs/max(sel.sum(),1):.2f}%"),
]
for i,(k,v) in enumerate(stats_d):
    y_=0.92-i*0.095
    ax.text(0.04,y_,k,transform=ax.transAxes,color=MUTED,fontsize=8.5,va="top")
    ax.text(0.96,y_,v,transform=ax.transAxes,color=BLOB_COL if "blob" in k.lower()
            else TEXT,fontsize=8.5,va="top",ha="right")

fig.suptitle(f"HETDEX LANs — D: Lyman-α Blobs  (r_iso≥{BLOB_R_ISO}kpc, N={n_blobs}){syn}",
             color=TEXT,fontsize=13,fontweight="bold",y=0.975)
save_with_caption(fig, OUT["D"], "D")
print(f"  Saved {OUT['D']}")
plt.close(fig)

# =============================================================================
# MODULE E — COSMIC EVOLUTION  (10–12 Gyr)
# =============================================================================
print("--- Module E: Cosmic evolution ---")
fig,axes = plt.subplots(2,4,figsize=(22,10))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.44,wspace=0.32,left=0.06,right=0.97,
                    top=0.91,bottom=0.09)

zgrid = np.linspace(Z_MIN,Z_MAX,50)

def zmed_track(mask,arr,min_n=15):
    r_,z_=[],[]
    for i in range(len(zgrid)-1):
        m=mask&(z>=zgrid[i])&(z<zgrid[i+1])&np.isfinite(arr)
        if m.sum()<min_n: continue
        r_.append(np.nanmedian(arr[m]))
        z_.append(0.5*(zgrid[i]+zgrid[i+1]))
    return np.array(z_),np.array(r_)

def zmed_err(mask,arr,min_n=15):
    r_,rlo_,rhi_,z_=[],[],[],[]
    for i in range(len(zgrid)-1):
        m=mask&(z>=zgrid[i])&(z<zgrid[i+1])&np.isfinite(arr)
        if m.sum()<min_n: continue
        r_.append(np.nanmedian(arr[m]))
        rlo_.append(np.nanpercentile(arr[m],16))
        rhi_.append(np.nanpercentile(arr[m],84))
        z_.append(0.5*(zgrid[i]+zgrid[i+1]))
    return np.array(z_),np.array(r_),np.array(rlo_),np.array(rhi_)

def add_lb_axis(ax):
    ax2=ax.twiny(); ax2.set_xlim(ax.get_xlim())
    zt=[2.0,2.5,3.0,3.5]; ax2.set_xticks(zt)
    ax2.set_xticklabels([f"{lb(z_):.1f}" for z_ in zt],color=MUTED,fontsize=7)
    ax2.set_xlabel("Look-back time (Gyr)",color=MUTED,fontsize=7.5)
    ax2.spines["top"].set_color(SPINE); ax2.tick_params(colors=MUTED,labelsize=7)

# E1: r_iso(z) LAE + AGN + shading
ax = axes[0,0]; sax(ax,r"E1: Size evolution $\langle r_{\rm iso}\rangle(z)$",
                    r"Redshift $z$",r"Median $r_{\rm iso}$ [kpc]")
zv,rv,rlo,rhi=zmed_err(sel,r_iso)
if len(zv):
    ax.fill_between(zv,rlo,rhi,color=ALL_COL,alpha=0.20)
    ax.plot(zv,rv,"-",color=ALL_COL,lw=2.0,label="All ±1σ")
for mask,color,lbl in [(sel_lae,LAE_COL,"LAE"),(sel_agn,AGN_COL,"AGN")]:
    zv2,rv2=zmed_track(mask,r_iso)
    if len(zv2): ax.plot(zv2,rv2,"--",color=color,lw=1.4,alpha=0.80,label=lbl)
add_lb_axis(ax); ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# E2: logL(z)
ax = axes[0,1]; sax(ax,r"E2: Luminosity evolution $\langle\log L\rangle(z)$",
                    r"Redshift $z$",r"Median $\log L$")
for mask,color,lbl in [(sel,ALL_COL,"All"),(sel_lae,LAE_COL,"LAE"),(sel_agn,AGN_COL,"AGN")]:
    zv2,rv2=zmed_track(mask,logL)
    if len(zv2): ax.plot(zv2,rv2,"-" if lbl=="All" else "--",
                          color=color,lw=(2.0 if lbl=="All" else 1.4),
                          alpha=0.85,label=lbl)
add_lb_axis(ax); ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="lower right")

# E3: EW(z)
ax = axes[0,2]; sax(ax,r"E3: EW evolution $\langle$EW$_{\rm rest}\rangle(z)$",
                    r"Redshift $z$",r"Median EW$_{\rm rest}$ [Å]")
ok_ew2=sel&np.isfinite(ew)&(ew>0)
for mask,color,lbl in [(ok_ew2,ALL_COL,"All"),
                        (ok_ew2&sel_lae,LAE_COL,"LAE"),
                        (ok_ew2&sel_agn,AGN_COL,"AGN")]:
    zv2,rv2=zmed_track(mask,ew)
    if len(zv2): ax.plot(zv2,rv2,"-" if lbl=="All" else "--",
                          color=color,lw=(2.0 if lbl=="All" else 1.4),
                          alpha=0.85,label=lbl)
add_lb_axis(ax); ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="lower right")

# E4: r_s(z)  (scale length evolution)
ax = axes[0,3]; sax(ax,r"E4: Scale length evolution $\langle r_s\rangle(z)$",
                    r"Redshift $z$",r"Median $r_s$ [kpc]")
ok_rs2=sel&np.isfinite(r_s)&(r_s>0)
for mask,color,lbl in [(ok_rs2,ALL_COL,"All"),
                        (ok_rs2&sel_lae,LAE_COL,"LAE"),
                        (ok_rs2&sel_agn,AGN_COL,"AGN")]:
    zv2,rv2=zmed_track(mask,r_s)
    if len(zv2): ax.plot(zv2,rv2,"-" if lbl=="All" else "--",
                          color=color,lw=(2.0 if lbl=="All" else 1.4),
                          alpha=0.85,label=lbl)
add_lb_axis(ax); ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# E5: number density N(z)
ax = axes[1,0]; sax(ax,"E5: Redshift distribution N(z)",
                    r"Redshift $z$","N per Δz=0.1 bin")
zbins_n=np.arange(Z_MIN,Z_MAX+0.1,0.1)
h_l,_=np.histogram(z[sel_lae&np.isfinite(z)],bins=zbins_n)
h_a,_=np.histogram(z[sel_agn&np.isfinite(z)],bins=zbins_n)
zc_n=0.5*(zbins_n[:-1]+zbins_n[1:])
ax.bar(zc_n,h_l,width=0.095,color=LAE_COL,alpha=0.75,label="LAE",edgecolor=BG,lw=0.3)
ax.bar(zc_n,h_a,width=0.095,bottom=h_l,color=AGN_COL,alpha=0.75,
       label="AGN",edgecolor=BG,lw=0.3)
add_lb_axis(ax); ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# E6: r_iso/r_s ratio vs z  (concentration evolution)
ax = axes[1,1]; sax(ax,"E6: Profile concentration vs redshift",
                    r"Redshift $z$",r"$r_{\rm iso}/r_s$")
ok_conc=sel&np.isfinite(r_iso)&np.isfinite(r_s)&(r_s>0)
conc=r_iso/np.where(r_s>0,r_s,np.nan)
zv2,cv2=zmed_track(ok_conc,conc)
if len(zv2): ax.plot(zv2,cv2,"-",color=ALL_COL,lw=2.0,alpha=0.90,label="All")
for mask,color,lbl in [(ok_conc&sel_lae,LAE_COL,"LAE"),(ok_conc&sel_agn,AGN_COL,"AGN")]:
    zv3,cv3=zmed_track(mask,conc)
    if len(zv3): ax.plot(zv3,cv3,"--",color=color,lw=1.3,alpha=0.75,label=lbl)
add_lb_axis(ax); ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# E7: SB sensitivity vs z (survey depth evolution)
ax = axes[1,2]; sax(ax,"E7: SB sensitivity vs redshift",
                    r"Redshift $z$",
                    r"Median SB$_{1\sigma}$ [10$^{-18}$ cgs arcsec$^{-2}$]")
ok_sb2=sel&np.isfinite(sb)
zv2,sv2=zmed_track(ok_sb2,sb)
if len(zv2): ax.plot(zv2,sv2,"-",color=ALL_COL,lw=2.0,label="All")
for mask,color,lbl in [(ok_sb2&sel_lae,LAE_COL,"LAE"),(ok_sb2&sel_agn,AGN_COL,"AGN")]:
    zv3,sv3=zmed_track(mask,sb)
    if len(zv3): ax.plot(zv3,sv3,"--",color=color,lw=1.3,alpha=0.75,label=lbl)
add_lb_axis(ax); ax.legend(fontsize=7.5,facecolor="#21262d",edgecolor=SPINE,labelcolor=TEXT,loc="upper right")

# E8: Evolution summary table
ax = axes[1,3]; ax.set_facecolor(AX_BG)
for sp in ax.spines.values(): sp.set_color(SPINE)
ax.set_title("E8: Evolution summary",color=TEXT,fontsize=9.5,
             fontweight="bold",loc="left",pad=4)
ax.set_xticks([]); ax.set_yticks([])
z3=[(1.87,2.30,"z~2.1\n(11.8Gyr)"),(2.30,2.80,"z~2.6\n(11.4Gyr)"),
    (2.80,3.52,"z~3.1\n(11.0Gyr)")]
hdr_row=["Epoch","N","r_iso","logL","EW"]
cx2=[0.02,0.30,0.48,0.67,0.83]
for j,(hdr,x_) in enumerate(zip(hdr_row,cx2)):
    ax.text(x_,0.94,hdr,transform=ax.transAxes,color=TEXT,
            fontsize=8.5,fontweight="bold",va="top")
ax.axhline(0.91,color=SPINE,lw=0.8)
for row_i,((zl,zh,zlbl),row_col) in enumerate(
        zip(z3,["#58a6ff","#3fb950","#f78166"])):
    zm=sel&(z>=zl)&(z<zh)
    y_=0.86-row_i*0.13
    vals=[zlbl.replace("\n"," "),
          f"{zm.sum():,}",
          f"{np.nanmedian(r_iso[zm]):.1f}",
          f"{np.nanmedian(logL[zm]):.2f}",
          f"{np.nanmedian(ew[zm&np.isfinite(ew)]):.0f}Å" if (zm&np.isfinite(ew)).sum()>0 else "—"]
    for j,(v,x_) in enumerate(zip(vals,cx2)):
        ax.text(x_,y_,v,transform=ax.transAxes,
                color=row_col if j==0 else MUTED,fontsize=7.5,va="top")

fig.suptitle(
    r"HETDEX LANs — E: Cosmic Evolution  ($z=1.9$–3.5, look-back 10–12 Gyr)" + syn,
    color=TEXT,fontsize=13,fontweight="bold",y=0.975)
save_with_caption(fig, OUT["E"], "E")
print(f"  Saved {OUT['E']}")
plt.close(fig)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("  HETDEX LAN Deep-Dive — All modules complete")
print("="*60)
for k,v in OUT.items():
    print(f"  {k}: {v}")
print("="*60)
