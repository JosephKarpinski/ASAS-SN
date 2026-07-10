"""
SPT-3G Main Survey Cluster Catalog - Analysis & Visualization Suite
Bleem et al. 2026 (arXiv:2607.01175)

Reads spt3g.csv and produces a set of dark-themed diagnostic figures
covering: sky distribution, significance/purity, mass-redshift space,
SZ signal analysis, richness-mass relations, redshift source breakdown,
dusty/synchrotron contamination flags, line-of-sight structure,
and strong lens candidates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# ----------------------------------------------------------------------
# Dark theme setup
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "savefig.facecolor": "#0d1117",
    "axes.edgecolor": "#8b949e",
    "axes.labelcolor": "#e6edf3",
    "text.color": "#e6edf3",
    "xtick.color": "#c9d1d9",
    "ytick.color": "#c9d1d9",
    "grid.color": "#30363d",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

CSV_PATH = "/mnt/user-data/uploads/spt3g.csv"
OUT_DIR = "/mnt/user-data/outputs"

df = pd.read_csv(CSV_PATH)

# Derived / convenience columns -----------------------------------------------
df["CONFIRMED"] = df["REDSHIFT"].notna()
confirmed = df[df["CONFIRMED"]].copy()
unconfirmed = df[~df["CONFIRMED"]].copy()

print(f"Total candidates: {len(df)}")
print(f"Confirmed clusters: {len(confirmed)}")
print(f"Unconfirmed candidates: {len(unconfirmed)}")

accent = "#58a6ff"
accent2 = "#f78166"
accent3 = "#3fb950"
accent4 = "#d29922"
accent5 = "#bc8cff"


# ==============================================================================
# FIGURE 1: Sky distribution + XI significance distribution + purity cuts
# ==============================================================================
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

# --- Sky map (RA/DEC), colored by confirmation status ---
ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(unconfirmed["RA"], unconfirmed["DEC"], s=4, c="#484f58",
            alpha=0.5, label=f"Unconfirmed ({len(unconfirmed)})")
sc = ax1.scatter(confirmed["RA"], confirmed["DEC"], s=5, c=confirmed["REDSHIFT"],
                  cmap="plasma", alpha=0.85, label=f"Confirmed ({len(confirmed)})")
cb = plt.colorbar(sc, ax=ax1, pad=0.01)
cb.set_label("Redshift")
ax1.set_xlabel("Right Ascension (deg)")
ax1.set_ylabel("Declination (deg)")
ax1.set_title("SPT-3G Main Field: Sky Distribution of Cluster Candidates")
ax1.legend(loc="upper right", framealpha=0.3, markerscale=3)
ax1.invert_xaxis()  # RA convention
ax1.grid(alpha=0.2)

# --- XI (detection significance) histogram with purity thresholds ---
ax2 = fig.add_subplot(gs[1, 0])
bins = np.logspace(np.log10(4), np.log10(df["XI"].max()), 40)
ax2.hist(df["XI"], bins=bins, color=accent, alpha=0.8, label="All candidates")
ax2.hist(confirmed["XI"], bins=bins, color=accent3, alpha=0.7, label="Confirmed")
ax2.axvline(4, color=accent4, ls="--", lw=1.5, label=r"$\xi=4$ (>82% pure)")
ax2.axvline(5, color=accent2, ls="--", lw=1.5, label=r"$\xi=5$ (>99% pure)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"Detection significance $\xi$")
ax2.set_ylabel("Number of candidates")
ax2.set_title("Significance Distribution & Purity Thresholds")
ax2.legend(fontsize=8.5, framealpha=0.3)
ax2.grid(alpha=0.2)

# --- Purity subsample bar chart ---
ax3 = fig.add_subplot(gs[1, 1])
n_xi4 = (df["XI"] >= 4).sum()
n_xi45 = (df["XI"] >= 4.5).sum()
n_xi5 = (df["XI"] >= 5).sum()
counts = [n_xi4, n_xi45, n_xi5]
purities = [82, 97, 99]
labels = [r"$\xi\geq4$", r"$\xi\geq4.5$", r"$\xi\geq5$"]
bars = ax3.bar(labels, counts, color=[accent4, accent2, accent3], alpha=0.85)
for b, p in zip(bars, purities):
    ax3.text(b.get_x() + b.get_width()/2, b.get_height()*1.02,
              f"{int(b.get_height())}\n({p}% pure)", ha="center", va="bottom", fontsize=9)
ax3.set_ylabel("Number of candidates")
ax3.set_title("Sample Size vs. Purity Cut")
ax3.grid(alpha=0.2, axis="y")

fig.suptitle("SPT-3G Cluster Catalog — Detection Overview", fontsize=15, y=1.00)
fig.savefig(f"{OUT_DIR}/fig1_sky_and_significance.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================================
# FIGURE 2: Mass-redshift space & lookback-time axis
# ==============================================================================
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.30)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
sc = ax.scatter(confirmed["REDSHIFT"], confirmed["M500C"], s=8,
                 c=confirmed["XI"], cmap="viridis", norm=mpl.colors.LogNorm(),
                 alpha=0.75)
cb = plt.colorbar(sc, ax=ax, pad=0.01)
cb.set_label(r"$\xi$")
ax.set_yscale("log")
ax.set_xlabel("Redshift")
ax.set_ylabel(r"$M_{500c}$  [$10^{14}\,M_\odot/h_{70}$]")
ax.set_title("Mass–Redshift Distribution (Confirmed Clusters)")
ax.grid(alpha=0.2)

# secondary lookback-time axis
z_ticks = np.array([0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
lookback = cosmo.lookback_time(z_ticks).value
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(z_ticks)
ax_top.set_xticklabels([f"{lt:.1f}" for lt in lookback])
ax_top.set_xlabel("Lookback time [Gyr]")

# --- Redshift histogram with high-z shading ---
ax2 = axes[1]
ax2.hist(confirmed["REDSHIFT"], bins=30, color=accent, alpha=0.85, edgecolor="#0d1117")
ax2.axvspan(1.6, confirmed["REDSHIFT"].max()*1.02, color=accent2, alpha=0.15,
            label="z > 1.6 (IR color-degeneracy limit)")
ax2.axvline(confirmed["REDSHIFT"].median(), color=accent4, ls="--",
            label=f"median z = {confirmed['REDSHIFT'].median():.3f}")
ax2.set_xlabel("Redshift")
ax2.set_ylabel("Number of confirmed clusters")
ax2.set_title("Redshift Distribution")
ax2.legend(fontsize=9, framealpha=0.3)
ax2.grid(alpha=0.2)

fig.suptitle("Mass–Redshift Space & Cosmic Epoch", fontsize=15)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_mass_redshift.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================================
# FIGURE 3: SZ signal analysis - Y0 vs Mass & THETA_CORE vs mass/redshift
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
valid = confirmed.dropna(subset=["Y0", "M500C"])
sc = ax.scatter(valid["M500C"], valid["Y0"], s=8, c=valid["REDSHIFT"],
                 cmap="magma", alpha=0.75)
cb = plt.colorbar(sc, ax=ax, pad=0.01)
cb.set_label("Redshift")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$M_{500c}$  [$10^{14}\,M_\odot/h_{70}$]")
ax.set_ylabel(r"$Y_{SZ}^{0.75\prime}$ (integrated Comptonization)")
ax.set_title(r"$Y_{SZ}$–Mass Scaling")
ax.grid(alpha=0.2)

ax2 = axes[1]
sc2 = ax2.scatter(confirmed["REDSHIFT"], confirmed["THETA_CORE"], s=8,
                   c=np.log10(confirmed["M500C"]), cmap="cividis", alpha=0.75)
cb2 = plt.colorbar(sc2, ax=ax2, pad=0.01)
cb2.set_label(r"$\log_{10}(M_{500c})$")
ax2.set_xlabel("Redshift")
ax2.set_ylabel(r"Filter core size $\theta_c$ (arcmin)")
ax2.set_title(r"Matched-Filter Core Size vs. Redshift & Mass")
ax2.grid(alpha=0.2)

fig.suptitle("SZ Signal Diagnostics", fontsize=15)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3_sz_signal.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================================
# FIGURE 4: Optical richness (LAMBDA) vs SZ mass, colored by FCONT purity
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
valid = confirmed.dropna(subset=["LAMBDA", "M500C"])
sc = ax.scatter(valid["M500C"], valid["LAMBDA"], s=8, c=valid["FCONT"],
                 cmap="coolwarm", alpha=0.8, vmin=0, vmax=0.3)
cb = plt.colorbar(sc, ax=ax, pad=0.01)
cb.set_label(r"$f_{\rm cont}$ (contamination)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$M_{500c}$  [$10^{14}\,M_\odot/h_{70}$]")
ax.set_ylabel(r"MCMF Richness $\lambda$")
ax.set_title(r"Optical Richness vs. SZ Mass")
ax.grid(alpha=0.2)

ax2 = axes[1]
ax2.hist(confirmed["FCONT"].dropna(), bins=40, color=accent5, alpha=0.85, edgecolor="#0d1117")
ax2.axvline(0.2, color=accent2, ls="--", label=r"$f_{\rm cont}^{max}=0.2$ threshold")
ax2.set_yscale("log")
ax2.set_xlabel(r"$f_{\rm cont}$")
ax2.set_ylabel("Number of clusters")
ax2.set_title("Contamination Fraction Distribution")
ax2.legend(fontsize=9, framealpha=0.3)
ax2.grid(alpha=0.2)

fig.suptitle("Optical Richness–Mass Relation & Purity Cuts", fontsize=15)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4_richness_mass.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================================
# FIGURE 5: Redshift source classification (pandas groupby)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
src_counts = confirmed["REDSHIFT_SOURCE"].value_counts()
colors = plt.cm.tab10(np.linspace(0, 1, len(src_counts)))
ax.barh(src_counts.index[::-1], src_counts.values[::-1], color=colors[::-1])
ax.set_xscale("log")
ax.set_xlabel("Number of clusters")
ax.set_title("Redshift Source Breakdown")
ax.grid(alpha=0.2, axis="x")

ax2 = axes[1]
specz = confirmed[confirmed["SPECZ"] == 1]
photoz = confirmed[confirmed["SPECZ"] != 1]
ax2.hist(photoz["REDSHIFT"], bins=30, alpha=0.7, color=accent, label=f"Photo-z ({len(photoz)})")
ax2.hist(specz["REDSHIFT"], bins=30, alpha=0.7, color=accent2, label=f"Spec-z ({len(specz)})")
ax2.set_xlabel("Redshift")
ax2.set_ylabel("Number of clusters")
ax2.set_title("Spectroscopic vs. Photometric Redshifts")
ax2.legend(fontsize=9, framealpha=0.3)
ax2.grid(alpha=0.2)

fig.suptitle("Redshift Provenance", fontsize=15)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig5_redshift_source.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================================
# FIGURE 6: Dusty/synchrotron contamination (DUSTY_FLAG, PS_FLAG) vs redshift
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
dusty_flagged = confirmed[confirmed["DUSTY_FLAG"] > 0]
clean = confirmed[confirmed["DUSTY_FLAG"] == 0]
z_bins = np.linspace(0, 2, 21)
ax.hist(clean["REDSHIFT"], bins=z_bins, alpha=0.6, color=accent, label="No dusty flag")
ax.hist(dusty_flagged["REDSHIFT"], bins=z_bins, alpha=0.9, color=accent2,
         label=f"Dusty-flagged ({len(dusty_flagged)})")
ax.set_yscale("log")
ax.set_xlabel("Redshift")
ax.set_ylabel("Number of clusters")
ax.set_title("Dusty Contamination Flags vs. Redshift")
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(alpha=0.2)

ax2 = axes[1]
ps_counts = df["PS_FLAG"].value_counts().sort_index()
ps_labels = {0: "None", 1: "Bright source\n<4' (>6mJy)", 2: r"$Y_{SZ}$ impacted", 3: "Both"}
ax2.bar([ps_labels[i] for i in ps_counts.index], ps_counts.values,
        color=[accent3, accent4, accent2, accent5][:len(ps_counts)])
ax2.set_yscale("log")
ax2.set_ylabel("Number of candidates")
ax2.set_title("Point-Source Flag (PS_FLAG) Breakdown")
ax2.grid(alpha=0.2, axis="y")
for tick in ax2.get_xticklabels():
    tick.set_fontsize(8.5)

fig.suptitle("Contamination Diagnostics: Dusty & Point-Source Flags", fontsize=15)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig6_contamination_flags.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ==============================================================================
# FIGURE 7: Line-of-sight structure (LOS) & Strong Lens candidates
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
los_yes = confirmed[confirmed["LOS"] == 1]
los_no = confirmed[confirmed["LOS"] == 0]
ax.scatter(los_no["REDSHIFT"], los_no["M500C"], s=6, color="#484f58", alpha=0.4,
           label=f"Single structure ({len(los_no)})")
ax.scatter(los_yes["REDSHIFT"], los_yes["M500C"], s=14, color=accent2, alpha=0.9,
           label=f"Multiple LOS structures ({len(los_yes)})")
ax.set_yscale("log")
ax.set_xlabel("Redshift")
ax.set_ylabel(r"$M_{500c}$  [$10^{14}\,M_\odot/h_{70}$]")
ax.set_title("Line-of-Sight Structure Flags")
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(alpha=0.2)

ax2 = axes[1]
lens_counts = confirmed[confirmed["STRONG_LENS"] > 0]["STRONG_LENS"].value_counts().sort_index()
lens_labels = {1: "Visual\n(this work)", 2: "Literature", 3: "Both"}
ax2.bar([lens_labels[i] for i in lens_counts.index], lens_counts.values,
        color=[accent3, accent, accent2][:len(lens_counts)])
ax2.set_ylabel("Number of clusters")
ax2.set_title(f"Strong Lens Candidates (n={int((confirmed['STRONG_LENS']>0).sum())})")
ax2.grid(alpha=0.2, axis="y")

fig.suptitle("Line-of-Sight Structure & Strong Gravitational Lens Candidates", fontsize=15)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig7_los_and_lensing.png", dpi=150, bbox_inches="tight")
plt.close(fig)


print("\nAll figures written to", OUT_DIR)
