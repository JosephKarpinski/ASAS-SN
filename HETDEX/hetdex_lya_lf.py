"""
hetdex_lya_lf.py
================
Lyman-alpha Luminosity Function from the HETDEX Source Catalog 2 (HPSC2)
using the classical 1/Vmax estimator (Schmidt 1968).

Physics
-------
The 1/Vmax estimator corrects for Malmquist bias: each galaxy is weighted
by the maximum comoving volume in which it COULD have been detected above
the survey flux limit, rather than the volume actually surveyed.

    Phi(L) dL = sum_i  1/Vmax_i    [Mpc^-3 dex^-1]

where Vmax_i = Omega/4pi * [Vc(z_max,i) - Vc(z_min)],
and z_max,i is the redshift at which source i would equal the flux limit.

Catalog columns used
--------------------
  source_type   : select 'lae'
  z_hetdex      : spectroscopic redshift
  flux_lya      : dust-corrected Lya line flux  [1e-17 erg/s/cm2]
  flux_lya_err  : uncertainty
  sn            : line S/N
  p_conf, p_cnn : RF and CNN classifier scores (SC2 only)
  field         : survey field

Requirements
------------
  pip install astropy numpy matplotlib scipy

Data
----
  hetdex_sc2_v1.5.fits
  https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/hetdex_source_catalog_2/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

from astropy.io    import fits
from astropy.table import Table
import astropy.units as u
from astropy.cosmology import Planck18

# ── Survey & quality configuration ────────────────────────────────────────────
CATALOG_PATH  = "hetdex_sc2_v1.5.fits"   # update to local path

# HETDEX VIRUS bandpass window for Lya 1215.67 AA
WAVE_MIN      = 3500.0    # AA  (conservative blue cut)
WAVE_MAX      = 5500.0    # AA
LYA_REST      = 1215.67   # AA
Z_MIN         = WAVE_MIN / LYA_REST - 1   # ~1.879
Z_MAX         = WAVE_MAX / LYA_REST - 1   # ~3.524

# Quality cuts
MIN_SN        = 5.5
MIN_P_CONF    = 0.5    # RF classifier (SC2)
MIN_P_CNN     = 0.5    # CNN classifier (SC2)
BAD_VALUE     = -999.0

# Survey solid angle (~HETDEX DR1 footprint)
HETDEX_AREA_DEG2 = 540.0
OMEGA_SR         = HETDEX_AREA_DEG2 * (np.pi / 180.0)**2

# Flux limit for Vmax: effective 5-sigma detection threshold
# HETDEX VIRUS 1-sigma depth ~4e-18 erg/s/cm2 → 5-sigma ~ 2e-17
F_LIM_CGS     = 2.0e-17    # erg/s/cm2
FLUX_SCALE    = 1e-17      # catalog units -> cgs

# LF binning in log10 luminosity
LUM_BIN_EDGES = np.arange(41.5, 44.1, 0.25)
LUM_BIN_CEN   = 0.5 * (LUM_BIN_EDGES[:-1] + LUM_BIN_EDGES[1:])
DLOG_L        = LUM_BIN_EDGES[1] - LUM_BIN_EDGES[0]

# Redshift bins for evolution panels
Z_BINS        = [(1.9, 2.4), (2.4, 2.9), (2.9, 3.5)]
Z_BIN_LABELS  = ["1.9 < z < 2.4", "2.4 < z < 2.9", "2.9 < z < 3.5"]
Z_BIN_COLORS  = ["#58a6ff", "#3fb950", "#f78166"]

cosmo = Planck18


# ── Physics helpers ───────────────────────────────────────────────────────────

def lum_distance_cm(z):
    return cosmo.luminosity_distance(z).to(u.cm).value


def flux_to_logL(flux_cgs, z):
    DL = lum_distance_cm(z)
    L  = 4.0 * np.pi * DL**2 * flux_cgs
    return np.log10(np.clip(L, 1.0, None))


def compute_vmax(flux_cgs, z_obs, z_min_survey, z_max_survey,
                 F_lim=F_LIM_CGS, n_steps=400):
    """
    Vectorised 1/Vmax: for each source find the redshift z_max at which
    its flux would equal F_lim, then integrate the comoving volume shell.

    Strategy: fix source luminosity L = 4pi DL(z_obs)^2 * flux.
    The predicted flux at trial z is F(z) = L / (4pi DL(z)^2).
    z_max is where F(z_max) = F_lim  =>  DL(z_max) = sqrt(L / 4pi / F_lim).
    We interpolate on a pre-built z vs DL grid.
    """
    z_grid  = np.linspace(z_min_survey, z_max_survey, n_steps)
    DL_grid = cosmo.luminosity_distance(z_grid).to(u.cm).value
    Vc_grid = cosmo.comoving_volume(z_grid).to(u.Mpc**3).value
    Vc_min  = cosmo.comoving_volume(z_min_survey).to(u.Mpc**3).value

    DL_obs  = lum_distance_cm(z_obs)              # (N,)
    L_src   = 4.0 * np.pi * DL_obs**2 * flux_cgs # (N,)  erg/s
    DL_max  = np.sqrt(L_src / (4.0 * np.pi * F_lim))  # (N,) cm

    # Vc at the redshift where DL = DL_max
    Vc_at_zmax = np.interp(DL_max, DL_grid, Vc_grid,
                           left=Vc_grid[0], right=Vc_grid[-1])

    Vmax = OMEGA_SR / (4.0 * np.pi) * (Vc_at_zmax - Vc_min)
    return np.clip(Vmax, 1.0, None)


def lf_1overVmax(log_L, Vmax, bin_edges):
    """
    1/Vmax luminosity function estimator.

    Returns
    -------
    phi      : Phi [Mpc^-3 dex^-1]
    phi_err  : Poisson uncertainty propagated through sum(1/V^2)
    n_gal    : source count per bin
    """
    dlog   = bin_edges[1] - bin_edges[0]
    n_bins = len(bin_edges) - 1
    phi    = np.zeros(n_bins)
    phi_err= np.zeros(n_bins)
    n_gal  = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        sel        = (log_L >= bin_edges[i]) & (log_L < bin_edges[i+1])
        n          = sel.sum()
        n_gal[i]   = n
        if n > 0:
            phi[i]     = np.sum(1.0 / Vmax[sel]) / dlog
            phi_err[i] = np.sqrt(np.sum(1.0 / Vmax[sel]**2)) / dlog

    return phi, phi_err, n_gal


def schechter(log_L, log_phi_star, log_L_star, alpha):
    """
    Schechter (1976) LF in log-log space.
    log Phi(L) = log[ln(10) * Phi* * (L/L*)^(alpha+1) * exp(-L/L*)]
    """
    x   = 10.0**(log_L - log_L_star)
    val = np.log(10.0) * 10.0**log_phi_star * x**(alpha + 1.0) * np.exp(-x)
    return np.log10(np.clip(val, 1e-40, None))


# ── Load catalog ──────────────────────────────────────────────────────────────
print(f"Loading {CATALOG_PATH} ...")
try:
    hdul = fits.open(CATALOG_PATH)
    tab  = Table(hdul[1].data)
    hdul.close()
    print(f"  Loaded {len(tab):,} rows")
    SYNTHETIC = False

except FileNotFoundError:
    print("  *** Catalog not found — generating synthetic LAE demo data ***")
    print("  (reproduces realistic HETDEX Lya LF statistics)")
    SYNTHETIC = True

    rng   = np.random.default_rng(42)
    N_raw = 1_000_000  # pre-detection draws

    z_raw = rng.uniform(Z_MIN + 0.05, Z_MAX - 0.05, N_raw)
    DL_raw = cosmo.luminosity_distance(z_raw).to(u.cm).value

    # Schechter draw with Gronwall+07 / Ouchi+08 parameters
    # log_phi* = -3.1, log_L* = 42.7, alpha = -1.6
    log_L_grid = np.arange(40.5, 44.5, 0.01)
    x_g        = 10.0**(log_L_grid - 42.7)
    pdf_g      = x_g**(-0.6) * np.exp(-x_g)  # alpha+1 = -0.6
    pdf_g     /= pdf_g.sum()

    log_L_true = rng.choice(log_L_grid, size=N_raw, p=pdf_g)
    flux_true  = 10.0**log_L_true / (4.0 * np.pi * DL_raw**2)

    # Measurement noise: 1-sigma ~ F_lim/5 with log-normal scatter
    noise_1sig = (F_LIM_CGS / 5.0) * rng.lognormal(0.0, 0.35, N_raw)
    flux_obs   = flux_true + rng.normal(0.0, noise_1sig)
    sn_obs     = flux_obs / noise_1sig

    # Detection cut
    det = (flux_obs >= F_LIM_CGS) & (sn_obs >= 4.5)
    print(f"  Pre-detection: {N_raw:,}  ->  Detected: {det.sum():,}")

    z_d        = z_raw[det]
    flux_d     = flux_obs[det]
    ferr_d     = noise_1sig[det]
    sn_d       = sn_obs[det]
    p_conf_d   = np.clip(rng.beta(4.0, 1.2, det.sum()), 0.0, 1.0)
    p_cnn_d    = np.clip(rng.beta(3.5, 1.1, det.sum()), 0.0, 1.0)
    fields_d   = rng.choice(
        ["dex-spring", "dex-fall", "cosmos", "goods-n"],
        det.sum(), p=[0.55, 0.30, 0.10, 0.05]
    )

    tab = Table({
        "source_type" : np.full(det.sum(), "lae"),
        "z_hetdex"    : z_d.astype(np.float32),
        "flux_lya"    : (flux_d / FLUX_SCALE).astype(np.float32),
        "flux_lya_err": (ferr_d / FLUX_SCALE).astype(np.float32),
        "sn"          : sn_d.astype(np.float32),
        "p_conf"      : p_conf_d.astype(np.float32),
        "p_cnn"       : p_cnn_d.astype(np.float32),
        "field"       : fields_d.astype("U12"),
    })


# ── Quality selection ─────────────────────────────────────────────────────────
mask  = tab["source_type"] == "lae"
z_col = np.array(tab["z_hetdex"], dtype=float)
mask &= (z_col >= Z_MIN) & (z_col <= Z_MAX)
mask &= np.array(tab["sn"], dtype=float) >= MIN_SN
mask &= np.array(tab["flux_lya"], dtype=float) > 0
mask &= np.array(tab["flux_lya"], dtype=float) != BAD_VALUE

if "p_conf" in tab.colnames:
    mask &= np.array(tab["p_conf"], dtype=float) >= MIN_P_CONF
if "p_cnn" in tab.colnames:
    mask &= np.array(tab["p_cnn"],  dtype=float) >= MIN_P_CNN

lae = tab[mask].copy()
print(f"\n  LAEs after quality cuts : {len(lae):,}")
print(f"  z window : {Z_MIN:.3f} – {Z_MAX:.3f}  "
      f"(Lya in {WAVE_MIN:.0f}–{WAVE_MAX:.0f} AA)")

z_arr    = np.array(lae["z_hetdex"],    dtype=float)
flux_arr = np.array(lae["flux_lya"],    dtype=float) * FLUX_SCALE
ferr_arr = np.array(lae["flux_lya_err"],dtype=float) * FLUX_SCALE
log_L    = flux_to_logL(flux_arr, z_arr)

if "field" in lae.colnames:
    field_arr = np.array(lae["field"], dtype=str)
else:
    field_arr = np.full(len(lae), "unknown")


# ── Global 1/Vmax LF ──────────────────────────────────────────────────────────
print("\nComputing global 1/Vmax ...")
Vmax_all                      = compute_vmax(flux_arr, z_arr, Z_MIN, Z_MAX)
phi_all, phi_err_all, n_all   = lf_1overVmax(log_L, Vmax_all, LUM_BIN_EDGES)
good_global                   = (n_all >= 3) & (phi_all > 0)
log_phi_all   = np.where(phi_all > 0, np.log10(phi_all),     np.nan)
log_phi_err_all = np.where(phi_all > 0,
                           phi_err_all / (phi_all * np.log(10)), np.nan)


# ── Schechter fit ─────────────────────────────────────────────────────────────
fit_mask = good_global & (LUM_BIN_CEN >= 41.8) & (LUM_BIN_CEN <= 43.5)
p0 = [-3.0, 42.7, -1.6]
fit_ok = False
try:
    popt, pcov = curve_fit(
        schechter,
        LUM_BIN_CEN[fit_mask],
        log_phi_all[fit_mask],
        p0=p0,
        sigma=log_phi_err_all[fit_mask],
        absolute_sigma=True,
        maxfev=8000,
    )
    perr   = np.sqrt(np.diag(pcov))
    fit_ok = True
    print(f"\n  Schechter fit:")
    print(f"    log Phi* = {popt[0]:.2f} +/- {perr[0]:.2f}  [Mpc^-3 dex^-1]")
    print(f"    log L*   = {popt[1]:.2f} +/- {perr[1]:.2f}  [erg/s]")
    print(f"    alpha    = {popt[2]:.2f} +/- {perr[2]:.2f}")
except Exception as exc:
    print(f"  Schechter fit failed: {exc}")
    popt, perr = p0, [0, 0, 0]


# ── Redshift-binned LFs ───────────────────────────────────────────────────────
print("\nComputing redshift-binned LFs ...")
zbin_results = []
for z1, z2 in Z_BINS:
    sel = (z_arr >= z1) & (z_arr < z2)
    print(f"  z={z1:.1f}-{z2:.1f}: {sel.sum():,} LAEs")
    if sel.sum() < 10:
        zbin_results.append(None)
        continue
    Vmax_b = compute_vmax(flux_arr[sel], z_arr[sel], z1, z2)
    phi_b, phi_err_b, n_b = lf_1overVmax(log_L[sel], Vmax_b, LUM_BIN_EDGES)
    zbin_results.append({
        "phi": phi_b, "phi_err": phi_err_b,
        "n": n_b, "n_sel": sel.sum(),
    })


# ── Literature reference points ───────────────────────────────────────────────
# Gronwall et al. (2007) z=3.1
gronwall_logL   = np.array([41.60, 41.85, 42.10, 42.35, 42.60, 42.85])
gronwall_logPhi = np.array([-2.22, -2.35, -2.60, -2.92, -3.35, -3.95])
gronwall_err    = np.array([ 0.10,  0.08,  0.10,  0.12,  0.18,  0.30])

# Ouchi et al. (2008) z=3.7
ouchi_logL   = np.array([42.20, 42.45, 42.70, 42.95, 43.20])
ouchi_logPhi = np.array([-2.65, -2.95, -3.30, -3.80, -4.30])
ouchi_err    = np.array([ 0.12,  0.12,  0.15,  0.20,  0.35])


# ── Plotting ───────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
BG     = "#0d1117"
AX_BG  = "#161b22"
SPINE  = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
ACCENT = "#58a6ff"

fig = plt.figure(figsize=(17, 14))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(2, 2, figure=fig,
                         hspace=0.38, wspace=0.30,
                         left=0.09, right=0.97,
                         top=0.92,  bottom=0.07)

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, which="both", direction="in",
                   top=True, right=True, labelsize=10)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=11)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold",
                 loc="left", pad=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def leg(ax, **kw):
    return ax.legend(fontsize=9, facecolor="#21262d",
                     edgecolor=SPINE, labelcolor=TEXT, **kw)

LYA_XLABEL = r"$\log_{10}\ L_{\rm Ly\alpha}\ [\rm erg\,s^{-1}]$"
PHI_YLABEL = r"$\log_{10}\ \Phi\ [\rm Mpc^{-3}\,dex^{-1}]$"

# ── Panel 1: Global LF + Schechter ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Global Lya Luminosity Function", LYA_XLABEL, PHI_YLABEL)

ax1.errorbar(
    LUM_BIN_CEN[good_global], log_phi_all[good_global],
    yerr=log_phi_err_all[good_global],
    fmt="o", color=ACCENT, ms=6.5, lw=1.5,
    capsize=3, ecolor=ACCENT, elinewidth=1.2,
    label=f"HETDEX SC2  (N={n_all[good_global].sum():,})",
    zorder=5,
)

if fit_ok:
    L_fit = np.linspace(41.2, 44.2, 400)
    ax1.plot(
        L_fit, schechter(L_fit, *popt),
        "--", color="#f78166", lw=2.0, alpha=0.88,
        label=(
            f"Schechter fit\n"
            f"  log$\\Phi^*$={popt[0]:.2f}\n"
            f"  log$L^*$={popt[1]:.2f}\n"
            f"  $\\alpha$={popt[2]:.2f}"
        ),
    )

ax1.errorbar(gronwall_logL, gronwall_logPhi, yerr=gronwall_err,
             fmt="s", color="#d2a8ff", ms=5, capsize=2.5,
             elinewidth=1.0, alpha=0.75,
             label="Gronwall+07  (z=3.1)")
ax1.errorbar(ouchi_logL, ouchi_logPhi, yerr=ouchi_err,
             fmt="^", color="#ffa657", ms=5, capsize=2.5,
             elinewidth=1.0, alpha=0.75,
             label="Ouchi+08  (z=3.7)")

ax1.set_xlim(41.2, 44.2)
ax1.set_ylim(-5.8, -1.2)
leg(ax1)

# ── Panel 2: LF in redshift bins ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Lya LF: Redshift Evolution", LYA_XLABEL, PHI_YLABEL)

for res, label, color in zip(zbin_results, Z_BIN_LABELS, Z_BIN_COLORS):
    if res is None:
        continue
    good = (res["n"] >= 3) & (res["phi"] > 0)
    lp   = np.where(res["phi"] > 0, np.log10(res["phi"]), np.nan)
    lpe  = np.where(res["phi"] > 0,
                    res["phi_err"] / (res["phi"] * np.log(10)), np.nan)
    ax2.errorbar(LUM_BIN_CEN[good], lp[good], yerr=lpe[good],
                 fmt="o", color=color, ms=5.5, lw=1.3,
                 capsize=2.5, elinewidth=1.0,
                 label=f"z = {label}  (N={res['n_sel']:,})")

ax2.set_xlim(41.2, 44.2)
ax2.set_ylim(-5.8, -1.2)
leg(ax2)

# ── Panel 3: N(z) stacked by field + limiting luminosity ─────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Redshift Distribution of LAEs",
         "Spectroscopic redshift  $z$",
         "Count per 0.05 dex bin")

z_hist_bins = np.arange(Z_MIN, Z_MAX + 0.051, 0.05)
field_palette = {
    "dex-spring": "#58a6ff", "dex-fall": "#3fb950",
    "cosmos": "#f78166",     "goods-n": "#d2a8ff",
    "nep": "#ffa657",        "ssa22": "#79c0ff",
}
bottom = np.zeros(len(z_hist_bins) - 1)
for fname, fc in field_palette.items():
    fsel = field_arr == fname
    if fsel.sum() < 2:
        continue
    h, _ = np.histogram(z_arr[fsel], bins=z_hist_bins)
    ax3.bar(z_hist_bins[:-1], h, width=0.05, bottom=bottom,
            color=fc, alpha=0.82, label=fname,
            align="edge", linewidth=0)
    bottom += h

# Shade the three z bins
for (z1, z2), c in zip(Z_BINS, Z_BIN_COLORS):
    ax3.axvspan(z1, z2, alpha=0.07, color=c)

ax3.set_xlim(Z_MIN, Z_MAX)
leg(ax3, ncol=2)

# Twin axis: limiting log L vs z (the Malmquist ramp)
ax3b = ax3.twinx()
z_grid   = np.linspace(Z_MIN, Z_MAX, 300)
DL_grid  = cosmo.luminosity_distance(z_grid).to(u.cm).value
logL_lim = np.log10(4.0 * np.pi * DL_grid**2 * F_LIM_CGS)
ax3b.plot(z_grid, logL_lim, "--", color="#ffa657",
          lw=1.8, alpha=0.75, label=r"$L_{\rm lim}(z)$")
ax3b.set_ylabel(r"$\log_{10}\ L_{\rm lim}\ [\rm erg\,s^{-1}]$",
                color=MUTED, fontsize=10)
ax3b.tick_params(colors=MUTED, labelsize=9)
ax3b.legend(fontsize=9, facecolor="#21262d", edgecolor=SPINE,
            labelcolor=TEXT, loc="upper left")

# ── Panel 4: Vmax vs luminosity (bias diagnostic) ─────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4,
         r"$V_{\rm max}$ Diagnostic",
         LYA_XLABEL,
         r"$\log_{10}\ V_{\rm max}\ [\rm Mpc^{3}]$")

hb = ax4.hexbin(log_L, np.log10(Vmax_all),
                gridsize=52, cmap="YlOrRd",
                mincnt=3, linewidths=0.2)
cb = fig.colorbar(hb, ax=ax4, pad=0.01)
cb.set_label("Counts per hex", color=MUTED, fontsize=9)
cb.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
cb.outline.set_edgecolor(SPINE)

# Full survey volume (bright sources that reach z_max = Z_MAX)
Vc_full = cosmo.comoving_volume(Z_MAX).to(u.Mpc**3).value
Vc_min  = cosmo.comoving_volume(Z_MIN).to(u.Mpc**3).value
logV_survey = np.log10(OMEGA_SR / (4*np.pi) * (Vc_full - Vc_min))
ax4.axhline(logV_survey, color="#3fb950", lw=1.6, ls="--", alpha=0.75,
            label=r"$V_{\rm survey}$ ceiling (bright sources)")

# Flux-limit locus: trace the boundary Vmax = V(z_obs)
z_locus = np.linspace(Z_MIN + 0.05, Z_MAX - 0.05, 200)
DL_locus = cosmo.luminosity_distance(z_locus).to(u.cm).value
logL_locus = np.log10(4*np.pi * DL_locus**2 * F_LIM_CGS)
Vc_locus = cosmo.comoving_volume(z_locus).to(u.Mpc**3).value
logVmax_locus = np.log10(OMEGA_SR / (4*np.pi) * (Vc_locus - Vc_min))
ax4.plot(logL_locus, logVmax_locus, "-", color=ACCENT,
         lw=1.4, alpha=0.7, label="Flux-limit locus")

leg(ax4)

# ── Super-title & footer ───────────────────────────────────────────────────────
fig.suptitle(
    r"HETDEX SC2 — Ly$\alpha$ Luminosity Function  [$1/V_{\rm max}$ estimator, Schmidt 1968]",
    color=TEXT, fontsize=13, fontweight="bold", y=0.97,
)
fig.text(
    0.5, 0.005,
    (f"N_LAE = {len(lae):,}   |   S/N > {MIN_SN}   |   "
     f"z in [{Z_MIN:.2f}, {Z_MAX:.2f}]   |   "
     f"F_lim = {F_LIM_CGS/1e-17:.0f}e-17 erg/s/cm2   |   Planck18"),
    ha="center", fontsize=9, color=MUTED,
)

plt.savefig("hetdex_lya_lf.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("\nSaved -> hetdex_lya_lf.png")
plt.show()

# ── Console table ─────────────────────────────────────────────────────────────
print("\n-- Global LF ---------------------------------------------------")
print(f"  {'log L':>8}  {'log Phi':>9}  {'sigma':>8}  {'N':>6}")
for i in range(len(LUM_BIN_CEN)):
    if n_all[i] > 0 and phi_all[i] > 0:
        lp  = np.log10(phi_all[i])
        lpe = phi_err_all[i] / (phi_all[i] * np.log(10))
        print(f"  {LUM_BIN_CEN[i]:8.2f}  {lp:9.3f}  {lpe:8.3f}  {n_all[i]:6d}")
print("----------------------------------------------------------------")
if fit_ok:
    print(f"\n  Best-fit Schechter parameters (1-sigma):")
    print(f"    log Phi* = {popt[0]:.3f} +/- {perr[0]:.3f}")
    print(f"    log L*   = {popt[1]:.3f} +/- {perr[1]:.3f}")
    print(f"    alpha    = {popt[2]:.3f} +/- {perr[2]:.3f}")
