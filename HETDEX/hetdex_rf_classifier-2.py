"""
hetdex_rf_classifier.py
=======================
Train a Random Forest classifier on HETDEX SC2 Detection Information Table
features to predict source_type, evaluate with a confusion matrix, and
compare against the existing HETDEX p_conf / p_cnn classifiers.

Source types in SC2
-------------------
  lae   Lyman-alpha emitter             (z ~ 1.9–3.5)
  oii   [OII] emitter                   (z ~ 0–0.5)
  agn   Active galactic nucleus
  star  Stellar spectrum
  lzg   Low-redshift galaxy (continuum-selected)
  none  Unclassified

Features used (all from detinfo table)
---------------------------------------
  PRIMARY (spectroscopic — always available):
    sn              Line S/N
    sigma           Gaussian line-width σ  [AA]
    chi2            Reduced χ² of line fit
    continuum       Local continuum level  [1e-17 cgs/AA]
    wave            Central wavelength of line  [AA]
    flux            Extinction-corrected line flux

  IMAGING COUNTERPART (may be NaN for blank-field detections):
    counterpart_mag    Closest counterpart magnitude
    counterpart_dist   Angular distance to counterpart  [arcsec]
    gmag               HETDEX-spectrum g magnitude

  CLASSIFIER SCORES (existing HETDEX classifiers — used as features
  to see if RF can outperform or recombine them):
    p_conf             RF classifier score (existing HETDEX)
    p_cnn              CNN classifier score (existing HETDEX)
    plya_classification ELiXer Lya probability

  OBSERVATION QUALITY:
    fwhm               Seeing FWHM  [arcsec]
    throughput         Relative throughput at 4540 AA
    apcor              Aperture correction

Strategy
--------
1. Load detinfo FITS; resolve columns robustly.
2. Impute missing values (NaN sentinel → column median for numerics;
   mode for categorical).
3. Encode multi-class target: {lae, oii, agn, star, lzg, none}.
4. Stratified 80/20 train/test split.
5. Train RF with class_weight='balanced' to handle imbalance.
6. Evaluate: confusion matrix, per-class metrics, ROC-OVR.
7. Plot: confusion matrix heatmap, feature importance, per-class
   precision/recall bars, and (if p_conf available) comparison with
   existing HETDEX classifier at matched operating point.

Requirements
------------
  pip install scikit-learn astropy numpy matplotlib scipy pandas

Data
----
  hetdex_sc2_detinfo_v1.5.fits
  https://web.corral.tacc.utexas.edu/hetdex/HETDEX/pdr/pdr1/hetdex_source_catalog_2/
"""

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

DETINFO_PATH = "hetdex_sc2_detinfo_v1.5.fits"
SAVE_PATH    = "hetdex_rf_classifier.png"
MODEL_PATH   = "hetdex_rf_model.pkl"     # saved sklearn model; None = don't save

# Features to use (will gracefully skip any absent from the real catalog)
FEATURE_COLS = [
    # Core spectroscopic
    "sn", "sigma", "chi2", "continuum", "wave", "flux",
    # Imaging counterpart
    "counterpart_mag", "counterpart_dist", "gmag",
    # Existing HETDEX classifier scores (informative priors)
    "p_conf", "p_cnn", "plya_classification",
    # Observation quality
    "fwhm", "throughput", "apcor",
]

TARGET_COL   = "source_type"
BAD          = -999.0

# Class colours for plotting
CLASS_COLORS = {
    "lae" : "#58a6ff",
    "oii" : "#3fb950",
    "agn" : "#f78166",
    "star": "#d2a8ff",
    "lzg" : "#ffa657",
    "none": "#8b949e",
}

# RF hyperparameters
N_ESTIMATORS = 200   # halve for speed; raise to 400 for final run
MAX_DEPTH    = None     # fully grown trees
MIN_SAMPLES_LEAF = 5
N_JOBS       = -1       # use all CPU cores
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Optional: limit rows for speed during prototyping (None = use all)
MAX_ROWS = 300_000   # stratified sample — ~30 sec on M1; set None for full run

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
from matplotlib.ticker  import AutoMinorLocator
from matplotlib.colors  import LogNorm
import matplotlib.patheffects as pe

from astropy.io    import fits
from astropy.table import Table

from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import (train_test_split,
                                        StratifiedKFold,
                                        cross_val_score)
from sklearn.metrics           import (confusion_matrix,
                                        classification_report,
                                        roc_auc_score,
                                        precision_recall_fscore_support,
                                        ConfusionMatrixDisplay)
from sklearn.preprocessing     import LabelEncoder, label_binarize
from sklearn.inspection        import permutation_importance
from sklearn.calibration       import CalibratedClassifierCV
from scipy.stats               import mode as scipy_mode

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

def make_synthetic_detinfo(n=80_000, seed=77):
    """
    Realistic synthetic detinfo with genuine class-feature correlations:
      lae  : high sn, moderate sigma, chi2~1, faint continuum, no counterpart
      oii  : moderate sn, broad sigma, bright counterpart, low wave
      agn  : high sn, broad sigma, bright gmag, intermediate wave
      star : moderate sn, narrow sigma, bright counterpart, flat chi2
      lzg  : low sn, broad sigma, bright continuum, bright counterpart
      none : mixed, low sn
    """
    rng    = np.random.default_rng(seed)
    N_CLS  = {"lae":25000,"oii":28000,"agn":5000,"star":8000,"lzg":9000,"none":5000}
    rows   = []

    for cls, n_c in N_CLS.items():
        if cls == "lae":
            sn   = rng.lognormal(2.0, 0.5, n_c)
            sig  = rng.lognormal(0.8, 0.3, n_c)
            chi2 = rng.lognormal(0.1, 0.3, n_c)
            cont = rng.lognormal(-1.5, 0.8, n_c)
            wave = rng.uniform(3700, 5400, n_c)
            flux = rng.lognormal(1.5, 0.7, n_c)
            cmag = rng.uniform(22, 27, n_c)
            cdist= rng.exponential(3.0, n_c)
            gmag = rng.uniform(21, 26, n_c)
            pconf= np.clip(rng.beta(5, 1.5, n_c), 0, 1)
            pcnn = np.clip(rng.beta(4.5, 1.5, n_c), 0, 1)
            plya = np.clip(rng.beta(5, 1.2, n_c), 0, 1)
            # ~15% missing counterpart
            miss = rng.uniform(size=n_c) < 0.15
            cmag[miss] = np.nan;  cdist[miss] = np.nan

        elif cls == "oii":
            sn   = rng.lognormal(1.8, 0.6, n_c)
            sig  = rng.lognormal(1.5, 0.4, n_c)
            chi2 = rng.lognormal(0.2, 0.4, n_c)
            cont = rng.lognormal(0.5, 0.6, n_c)
            wave = rng.uniform(3470, 4100, n_c)   # [OII] 3727 in VIRUS band
            flux = rng.lognormal(1.2, 0.7, n_c)
            cmag = rng.uniform(18, 24, n_c)
            cdist= rng.exponential(1.0, n_c)
            gmag = rng.uniform(17, 23, n_c)
            pconf= np.clip(rng.beta(1.5, 5, n_c), 0, 1)
            pcnn = np.clip(rng.beta(1.5, 5, n_c), 0, 1)
            plya = np.clip(rng.beta(1.2, 6, n_c), 0, 1)
            miss = rng.uniform(size=n_c) < 0.05
            cmag[miss] = np.nan;  cdist[miss] = np.nan

        elif cls == "agn":
            sn   = rng.lognormal(2.5, 0.6, n_c)
            sig  = rng.lognormal(1.8, 0.5, n_c)
            chi2 = rng.lognormal(0.3, 0.5, n_c)
            cont = rng.lognormal(1.0, 0.7, n_c)
            wave = rng.uniform(3500, 5400, n_c)
            flux = rng.lognormal(2.0, 0.8, n_c)
            cmag = rng.uniform(18, 23, n_c)
            cdist= rng.exponential(0.5, n_c)
            gmag = rng.uniform(17, 22, n_c)
            pconf= np.clip(rng.beta(3, 2, n_c), 0, 1)
            pcnn = np.clip(rng.beta(3, 2, n_c), 0, 1)
            plya = np.clip(rng.beta(3, 3, n_c), 0, 1)
            miss = rng.uniform(size=n_c) < 0.03
            cmag[miss] = np.nan;  cdist[miss] = np.nan

        elif cls == "star":
            sn   = rng.lognormal(1.5, 0.6, n_c)
            sig  = rng.lognormal(0.4, 0.3, n_c)   # narrow lines
            chi2 = rng.lognormal(-0.1, 0.3, n_c)
            cont = rng.lognormal(2.0, 0.6, n_c)   # bright continuum
            wave = rng.uniform(3500, 5400, n_c)
            flux = rng.lognormal(0.8, 0.7, n_c)
            cmag = rng.uniform(14, 21, n_c)        # bright counterpart
            cdist= rng.exponential(0.3, n_c)
            gmag = rng.uniform(14, 21, n_c)
            pconf= np.clip(rng.beta(1, 5, n_c), 0, 1)
            pcnn = np.clip(rng.beta(1, 5, n_c), 0, 1)
            plya = np.clip(rng.beta(1, 8, n_c), 0, 1)
            miss = rng.uniform(size=n_c) < 0.01
            cmag[miss] = np.nan;  cdist[miss] = np.nan

        elif cls == "lzg":
            sn   = rng.lognormal(1.3, 0.6, n_c)
            sig  = rng.lognormal(2.0, 0.5, n_c)   # broad
            chi2 = rng.lognormal(0.4, 0.5, n_c)
            cont = rng.lognormal(1.5, 0.6, n_c)   # bright continuum
            wave = rng.uniform(3470, 5200, n_c)
            flux = rng.lognormal(0.5, 0.8, n_c)
            cmag = rng.uniform(17, 23, n_c)
            cdist= rng.exponential(0.8, n_c)
            gmag = rng.uniform(17, 23, n_c)
            pconf= np.clip(rng.beta(1.5, 4, n_c), 0, 1)
            pcnn = np.clip(rng.beta(1.5, 4, n_c), 0, 1)
            plya = np.clip(rng.beta(1, 6, n_c), 0, 1)
            miss = rng.uniform(size=n_c) < 0.04
            cmag[miss] = np.nan;  cdist[miss] = np.nan

        else:  # none
            sn   = rng.lognormal(1.2, 0.7, n_c)
            sig  = rng.lognormal(1.0, 0.6, n_c)
            chi2 = rng.lognormal(0.3, 0.6, n_c)
            cont = rng.lognormal(0.0, 1.0, n_c)
            wave = rng.uniform(3470, 5540, n_c)
            flux = rng.lognormal(0.8, 0.9, n_c)
            cmag = rng.uniform(19, 27, n_c)
            cdist= rng.exponential(2.0, n_c)
            gmag = rng.uniform(18, 26, n_c)
            pconf= np.clip(rng.beta(2, 2, n_c), 0, 1)
            pcnn = np.clip(rng.beta(2, 2, n_c), 0, 1)
            plya = np.clip(rng.beta(2, 4, n_c), 0, 1)
            miss = rng.uniform(size=n_c) < 0.20
            cmag[miss] = np.nan;  cdist[miss] = np.nan

        fwhm       = rng.uniform(1.2, 2.8, n_c)
        throughput = rng.uniform(0.05, 0.25, n_c)
        apcor      = rng.uniform(0.8, 1.3, n_c)

        for i in range(n_c):
            rows.append({
                "source_type"        : cls,
                "sn"                 : float(sn[i]),
                "sigma"              : float(sig[i]),
                "chi2"               : float(chi2[i]),
                "continuum"          : float(cont[i]),
                "wave"               : float(wave[i]),
                "flux"               : float(flux[i]),
                "counterpart_mag"    : float(cmag[i]) if not np.isnan(cmag[i]) else np.nan,
                "counterpart_dist"   : float(cdist[i]) if not np.isnan(cdist[i]) else np.nan,
                "gmag"               : float(gmag[i]),
                "p_conf"             : float(pconf[i]),
                "p_cnn"              : float(pcnn[i]),
                "plya_classification": float(plya[i]),
                "fwhm"               : float(fwhm[i]),
                "throughput"         : float(throughput[i]),
                "apcor"              : float(apcor[i]),
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  Synthetic: {len(df):,} rows  |  "
          f"class counts: {df['source_type'].value_counts().to_dict()}")
    return df


# =============================================================================
# CELL 4 — LOAD CATALOG
# =============================================================================

def getcol(cols, *cands):
    lc = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in lc:
            return lc[c.lower()]
    return None

def load_detinfo(path, max_rows=None):
    try:
        hdul = fits.open(path, memmap=True)
        tab  = Table(hdul[1].data)
        hdul.close()
        tab.rename_columns(tab.colnames,
                           [c.lower() for c in tab.colnames])
        df = tab.to_pandas()
        if max_rows and max_rows < len(df):
            # Stratified sample: preserve class proportions
            from sklearn.model_selection import train_test_split as _tts
            _, df = _tts(df, test_size=min(max_rows, len(df)) / len(df),
                         stratify=df[tgt_col].astype(str).str.strip().str.lower()
                                  .where(lambda s: s.isin(
                                      {"lae","oii","agn","star","lzg","none"}),
                                      other="none"),
                         random_state=RANDOM_STATE)
            df = df.reset_index(drop=True)
            print(f"  Stratified sample: {len(df):,} rows")
        print(f"Loaded {path}: {len(df):,} rows, "
              f"{len(df.columns)} columns")
        return df, False
    except FileNotFoundError:
        print(f"'{path}' not found — using synthetic demo data.")
        return make_synthetic_detinfo(), True


df_raw, SYNTHETIC = load_detinfo(DETINFO_PATH, max_rows=MAX_ROWS)
df_raw.columns = df_raw.columns.str.lower()

# =============================================================================
# CELL 5 — FEATURE ENGINEERING & PREPROCESSING
# =============================================================================

# Resolve available feature columns
avail_feats = [f for f in FEATURE_COLS
               if f.lower() in df_raw.columns]
missing_feats = [f for f in FEATURE_COLS
                 if f.lower() not in df_raw.columns]
if missing_feats:
    print(f"  Columns not in catalog (skipped): {missing_feats}")
print(f"  Using {len(avail_feats)} features: {avail_feats}")

# Resolve target column
tgt_col = next((c for c in df_raw.columns
                if c.lower() == TARGET_COL.lower()), None)
if tgt_col is None:
    raise KeyError(f"Target column '{TARGET_COL}' not found")

# Build working DataFrame
df = df_raw[avail_feats + [tgt_col]].copy()
df.rename(columns={tgt_col: "source_type"}, inplace=True)

# Clean target: strip whitespace, lower-case
df["source_type"] = df["source_type"].astype(str).str.strip().str.lower()
# Drop rows with unknown/empty target
valid_types = {"lae","oii","agn","star","lzg","none"}
df = df[df["source_type"].isin(valid_types)].copy()
print(f"\nAfter target filter: {len(df):,} rows")

# Replace BAD sentinels with NaN
df[avail_feats] = df[avail_feats].replace(BAD, np.nan)

# Derived features (always computable from primary columns)
if "flux" in df.columns and "continuum" in df.columns:
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ew_obs"] = (df["flux"] /
                        df["continuum"].replace(0, np.nan)).clip(-100, 2000)

if "sn" in df.columns and "chi2" in df.columns:
    df["sn_over_chi2"] = (df["sn"] /
                          df["chi2"].replace(0, np.nan)).clip(-100, 1000)

if "counterpart_mag" in df.columns:
    df["has_counterpart"] = df["counterpart_mag"].notna().astype(float)

# Update feature list with derived columns
derived = [c for c in ["ew_obs","sn_over_chi2","has_counterpart"]
           if c in df.columns]
feat_cols = avail_feats + derived
print(f"  + {len(derived)} derived features: {derived}")
print(f"  Total features: {len(feat_cols)}")

# ── Imputation: median for numeric, most-common for object ────────────────────
impute_vals = {}
for col in feat_cols:
    if col not in df.columns:
        continue
    if df[col].dtype == object:
        mv = df[col].mode()
        impute_vals[col] = mv.iloc[0] if len(mv) else "unknown"
    else:
        impute_vals[col] = df[col].median()
df[feat_cols] = df[feat_cols].fillna(impute_vals)

# Log-transform right-skewed features
log_feats = ["sn","sigma","chi2","flux","continuum",
             "counterpart_dist","ew_obs","sn_over_chi2"]
for f in log_feats:
    if f in df.columns:
        log_col = f"log_{f}"
        df[log_col] = np.log1p(np.clip(df[f], 0, None))
        feat_cols.append(log_col)

feat_cols = [f for f in feat_cols if f in df.columns]
feat_cols = list(dict.fromkeys(feat_cols))   # deduplicate preserving order

print(f"\nClass distribution:")
vc = df["source_type"].value_counts()
for cls, cnt in vc.items():
    print(f"  {cls:6s}: {cnt:8,}  ({100*cnt/len(df):.1f}%)")

# ── Encode target ─────────────────────────────────────────────────────────────
le  = LabelEncoder()
y   = le.fit_transform(df["source_type"])
X   = df[feat_cols].values.astype(np.float32)
classes = le.classes_

print(f"\nFeature matrix shape: {X.shape}")
print(f"Classes: {list(classes)}")

# =============================================================================
# CELL 6 — TRAIN / TEST SPLIT AND RF TRAINING
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain: {len(X_train):,}   Test: {len(X_test):,}")

print("Training Random Forest ...")
rf = RandomForestClassifier(
    n_estimators     = N_ESTIMATORS,
    max_depth        = MAX_DEPTH,
    min_samples_leaf = MIN_SAMPLES_LEAF,
    class_weight     = "balanced",
    n_jobs           = N_JOBS,
    random_state     = RANDOM_STATE,
    oob_score        = True,
)
rf.fit(X_train, y_train)

y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)   # (N_test, N_classes)

print(f"  OOB accuracy: {rf.oob_score_:.4f}")
print(f"  Test accuracy: {(y_pred == y_test).mean():.4f}")

# =============================================================================
# CELL 7 — METRICS
# =============================================================================

cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(classes)))

print("\n" + classification_report(
    y_test, y_pred,
    target_names=classes,
    digits=3
))

# Per-class precision, recall, F1
prec, rec, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=np.arange(len(classes)),
    average=None, zero_division=0
)

# Macro ROC-AUC (one-vs-rest)
y_test_bin = label_binarize(y_test, classes=np.arange(len(classes)))
try:
    roc_auc = roc_auc_score(y_test_bin, y_proba,
                             multi_class="ovr", average="macro")
    print(f"Macro ROC-AUC (OVR): {roc_auc:.4f}")
except Exception:
    roc_auc = np.nan

# Comparison with existing p_conf classifier (if available)
pconf_available = "p_conf" in avail_feats
if pconf_available:
    # p_conf is binary (lae/oii vs rest): evaluate on lae detection
    lae_idx = list(classes).index("lae") if "lae" in classes else None
    if lae_idx is not None:
        lae_mask_test = (y_test == lae_idx)
        rf_lae_prec   = prec[lae_idx]
        rf_lae_rec    = rec[lae_idx]
        # p_conf threshold at 0.5 for comparison
        p_conf_test = df.loc[df.index[
            df.index.isin(
                pd.RangeIndex(len(df))[
                    pd.Index(range(len(df))).isin(
                        np.where(
                            np.ones(len(df), bool)
                        )[0][-len(X_test):]
                    )
                ]
            )
        ], "p_conf"] if "p_conf" in df.columns else None

# ── Feature importances ───────────────────────────────────────────────────────
feat_imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(f"\nTop 15 feature importances:")
for fname, imp in feat_imp.head(15).items():
    bar = "█" * int(imp * 200)
    print(f"  {fname:25s}  {imp:.4f}  {bar}")

# Save model
if MODEL_PATH:
    import pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": rf, "label_encoder": le, "features": feat_cols}, f)
    print(f"\nModel saved -> {MODEL_PATH}")

# =============================================================================
# CELL 8 — PLOTTING
# =============================================================================

plt.style.use("dark_background")
BG    = "#0d1117"
AX_BG = "#161b22"
SPINE = "#30363d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    3, 3, figure=fig,
    hspace=0.42, wspace=0.32,
    left=0.07, right=0.97,
    top=0.93,  bottom=0.06,
)
ax_cm    = fig.add_subplot(gs[0, :2])   # wide: confusion matrix
ax_diag  = fig.add_subplot(gs[0, 2])   # diagonal metrics
ax_fi    = fig.add_subplot(gs[1, :])   # full-width: feature importances
ax_pr    = fig.add_subplot(gs[2, 0])   # precision/recall/F1 bars
ax_supp  = fig.add_subplot(gs[2, 1])   # support (class count)
ax_cmp   = fig.add_subplot(gs[2, 2])   # p_conf comparison or ROC-AUC

def style_ax(ax, title, xl="", yl=""):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, which="both", direction="in",
                   top=False, right=False, labelsize=9)
    ax.set_xlabel(xl, color=TEXT, fontsize=10)
    ax.set_ylabel(yl, color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=11,
                 fontweight="bold", loc="left", pad=6)

def mleg(ax, **kw):
    return ax.legend(fontsize=8.5, facecolor="#21262d",
                     edgecolor=SPINE, labelcolor=TEXT, **kw)

# ── Panel 1: Confusion matrix heatmap ─────────────────────────────────────────
style_ax(ax_cm, "Confusion Matrix  (test set, row-normalised)",
         "Predicted label", "True label")

# Row-normalise
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

colors = [CLASS_COLORS.get(c, "#8b949e") for c in classes]
cmap   = matplotlib.colors.LinearSegmentedColormap.from_list(
    "hetdex", ["#0d1117", "#1f6feb", "#58a6ff", "#e6edf3"], N=256)

im = ax_cm.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
fig.colorbar(im, ax=ax_cm, pad=0.01, fraction=0.02,
             label="Fraction").ax.yaxis.set_tick_params(color=MUTED)

for i in range(len(classes)):
    for j in range(len(classes)):
        val   = cm_norm[i, j]
        count = cm[i, j]
        color = "black" if val > 0.55 else TEXT
        ax_cm.text(j, i, f"{val:.2f}\n({count:,})",
                   ha="center", va="center",
                   fontsize=7.5 if len(classes) <= 6 else 6,
                   color=color, fontweight="bold" if i == j else "normal")

ax_cm.set_xticks(range(len(classes)))
ax_cm.set_yticks(range(len(classes)))
ax_cm.set_xticklabels(classes, rotation=30, ha="right",
                       color=TEXT, fontsize=9)
ax_cm.set_yticklabels(classes, color=TEXT, fontsize=9)

# Colour class labels
for tick, cls in zip(ax_cm.get_xticklabels(), classes):
    tick.set_color(CLASS_COLORS.get(cls, TEXT))
for tick, cls in zip(ax_cm.get_yticklabels(), classes):
    tick.set_color(CLASS_COLORS.get(cls, TEXT))

# ── Panel 2: Per-class diagonal accuracy + key stats ─────────────────────────
style_ax(ax_diag, "Per-class diagonal accuracy")

diag_acc = cm_norm.diagonal()
y_pos    = np.arange(len(classes))
bars     = ax_diag.barh(y_pos, diag_acc,
                         color=[CLASS_COLORS.get(c, MUTED) for c in classes],
                         alpha=0.85, edgecolor=SPINE, linewidth=0.5)
ax_diag.set_yticks(y_pos)
ax_diag.set_yticklabels(classes, color=TEXT, fontsize=9)
for tick, cls in zip(ax_diag.get_yticklabels(), classes):
    tick.set_color(CLASS_COLORS.get(cls, TEXT))
ax_diag.set_xlim(0, 1.05)
ax_diag.set_xlabel("True-positive rate", color=TEXT, fontsize=9)
ax_diag.axvline(0.9, color=MUTED, lw=0.8, ls="--", alpha=0.6,
                label="90% threshold")
for bar, val in zip(bars, diag_acc):
    ax_diag.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", color=TEXT, fontsize=8)
mleg(ax_diag)

# Key stats annotation
stats_txt = (
    f"OOB accuracy :  {rf.oob_score_:.4f}\n"
    f"Test accuracy:  {(y_pred==y_test).mean():.4f}\n"
    f"Macro ROC-AUC:  {roc_auc:.4f}\n"
    f"N estimators :  {N_ESTIMATORS}\n"
    f"N features   :  {len(feat_cols)}\n"
    f"Train N      :  {len(X_train):,}\n"
    f"Test N       :  {len(X_test):,}"
)
ax_diag.text(0.02, 0.02, stats_txt,
             transform=ax_diag.transAxes,
             color=MUTED, fontsize=7.5, va="bottom",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor=BG, edgecolor=SPINE, alpha=0.8))

# ── Panel 3: Feature importances ──────────────────────────────────────────────
style_ax(ax_fi,
         "Random Forest Feature Importances  (mean decrease in impurity)",
         "Feature", "Importance")

top_n   = min(20, len(feat_imp))
fi_top  = feat_imp.head(top_n)
fi_cols = fi_top.index.tolist()
fi_vals = fi_top.values

# Colour by feature group
def feat_group_color(fname):
    if fname in ("sn","sigma","chi2","continuum","wave","flux",
                 "log_sn","log_sigma","log_chi2","log_flux",
                 "log_continuum","sn_over_chi2","log_sn_over_chi2","ew_obs","log_ew_obs"):
        return "#58a6ff"   # spectroscopic
    if fname in ("counterpart_mag","counterpart_dist","gmag",
                 "log_counterpart_dist","has_counterpart"):
        return "#3fb950"   # imaging
    if fname in ("p_conf","p_cnn","plya_classification"):
        return "#ffa657"   # existing classifiers
    return "#8b949e"       # observational

bar_colors = [feat_group_color(f) for f in fi_cols]
ax_fi.bar(range(top_n), fi_vals, color=bar_colors, alpha=0.85,
          edgecolor=SPINE, linewidth=0.5)
ax_fi.set_xticks(range(top_n))
ax_fi.set_xticklabels(fi_cols, rotation=40, ha="right",
                       color=TEXT, fontsize=8)
for tick, fc in zip(ax_fi.get_xticklabels(), bar_colors):
    tick.set_color(fc)
ax_fi.set_ylabel("Importance", color=TEXT, fontsize=9)

# Legend for colour groups
from matplotlib.patches import Patch
fi_legend = [
    Patch(color="#58a6ff", label="Spectroscopic"),
    Patch(color="#3fb950", label="Imaging counterpart"),
    Patch(color="#ffa657", label="Existing HETDEX classifiers"),
    Patch(color="#8b949e", label="Observational quality"),
]
mleg(ax_fi, handles=fi_legend, loc="upper right")

# ── Panel 4: Precision / Recall / F1 per class ────────────────────────────────
style_ax(ax_pr,
         "Precision / Recall / F1  per class",
         "Class", "Score")

x_pos   = np.arange(len(classes))
w       = 0.26
ax_pr.bar(x_pos - w, prec,    width=w, color="#58a6ff", alpha=0.85,
          label="Precision", edgecolor=SPINE, linewidth=0.5)
ax_pr.bar(x_pos,     rec,     width=w, color="#3fb950", alpha=0.85,
          label="Recall",    edgecolor=SPINE, linewidth=0.5)
ax_pr.bar(x_pos + w, f1,      width=w, color="#ffa657", alpha=0.85,
          label="F1",        edgecolor=SPINE, linewidth=0.5)
ax_pr.set_xticks(x_pos)
ax_pr.set_xticklabels(classes, color=TEXT, fontsize=9, rotation=15)
for tick, cls in zip(ax_pr.get_xticklabels(), classes):
    tick.set_color(CLASS_COLORS.get(cls, TEXT))
ax_pr.set_ylim(0, 1.08)
ax_pr.axhline(0.9, color=MUTED, lw=0.7, ls=":", alpha=0.6)
mleg(ax_pr, loc="lower right")

# ── Panel 5: Support (class sizes in test set) ────────────────────────────────
style_ax(ax_supp, "Test-set class support", "Class", "N sources")

ax_supp.bar(x_pos, support,
            color=[CLASS_COLORS.get(c, MUTED) for c in classes],
            alpha=0.85, edgecolor=SPINE, linewidth=0.5)
ax_supp.set_xticks(x_pos)
ax_supp.set_xticklabels(classes, color=TEXT, fontsize=9, rotation=15)
for tick, cls in zip(ax_supp.get_xticklabels(), classes):
    tick.set_color(CLASS_COLORS.get(cls, TEXT))
ax_supp.set_yscale("log")
for xp, sp in zip(x_pos, support):
    ax_supp.text(xp, sp * 1.15, f"{sp:,}",
                 ha="center", va="bottom", color=TEXT, fontsize=7.5)

# ── Panel 6: p_conf vs RF LAE probability scatter / ROC comparison ───────────
style_ax(ax_cmp,
         "RF LAE probability vs HETDEX p_conf",
         "p_conf  (HETDEX RF)", "RF P(lae)")

if "p_conf" in df.columns and "lae" in classes:
    lae_idx_cls = list(classes).index("lae")
    # Get test-set rows
    test_indices_in_df = np.where(np.ones(len(df), bool))[0][-len(X_test):]
    p_conf_vals  = df["p_conf"].values[test_indices_in_df]
    rf_lae_proba = y_proba[:, lae_idx_cls]
    true_lae     = (y_test == lae_idx_cls)

    # Scatter: colour by true class
    for cls in classes:
        cls_idx = list(classes).index(cls)
        sel     = (y_test == cls_idx)
        if sel.sum() == 0:
            continue
        ax_cmp.scatter(
            p_conf_vals[sel], rf_lae_proba[sel],
            s=2, c=CLASS_COLORS.get(cls, MUTED), alpha=0.30,
            linewidths=0, rasterized=True,
            label=f"{cls} (N={sel.sum():,})"
        )
    ax_cmp.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, alpha=0.6,
                label="y = x")
    ax_cmp.set_xlim(-0.02, 1.02)
    ax_cmp.set_ylim(-0.02, 1.02)
    mleg(ax_cmp, loc="upper left", markerscale=4, ncol=2)
else:
    # Fallback: F1 vs class bar if p_conf not available
    ax_cmp.bar(x_pos, f1,
               color=[CLASS_COLORS.get(c, MUTED) for c in classes],
               alpha=0.85, edgecolor=SPINE, linewidth=0.5)
    ax_cmp.set_xticks(x_pos)
    ax_cmp.set_xticklabels(classes, color=TEXT, fontsize=9)
    ax_cmp.set_ylabel("F1 score", color=TEXT, fontsize=9)
    ax_cmp.set_ylim(0, 1.08)
    ax_cmp.set_title("F1 scores per class", color=TEXT,
                      fontsize=11, fontweight="bold", loc="left")

# ── Super-title ───────────────────────────────────────────────────────────────
syn_tag = "  [SYNTHETIC DATA]" if SYNTHETIC else ""
fig.suptitle(
    "HETDEX SC2 — Random Forest Source Classifier" + syn_tag,
    color=TEXT, fontsize=13, fontweight="bold", y=0.975,
)

if SAVE_PATH:
    fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved -> {SAVE_PATH}")
plt.show()

# =============================================================================
# CELL 9 — CROSS-VALIDATION AND FINAL REPORT
# =============================================================================

print("\n" + "=" * 65)
print("  HETDEX SC2 Random Forest Classifier — Summary")
print("=" * 65)
print(f"  OOB accuracy     : {rf.oob_score_:.4f}")
print(f"  Test accuracy    : {(y_pred==y_test).mean():.4f}")
print(f"  Macro ROC-AUC    : {roc_auc:.4f}")
print(f"\n  Per-class metrics (test set):")
hdr = "  {:6s}  {:>9}  {:>9}  {:>9}  {:>8}"
row = "  {:6s}  {:>9.3f}  {:>9.3f}  {:>9.3f}  {:>8,}"
print(hdr.format("class","precision","recall","F1","N"))
print("  " + "-" * 50)
for i, cls in enumerate(classes):
    print(row.format(cls, prec[i], rec[i], f1[i], support[i]))

print(f"\n  Top 10 features by importance:")
for fname, imp in feat_imp.head(10).items():
    grp = ("spectroscopic" if feat_group_color(fname) == "#58a6ff" else
           "imaging" if feat_group_color(fname) == "#3fb950" else
           "existing_classifier" if feat_group_color(fname) == "#ffa657" else
           "observational")
    print(f"    {fname:25s}  {imp:.4f}  [{grp}]")
print("=" * 65)
print("\nPhysical interpretation:")
print("  High p_conf/p_cnn importance -> RF is largely re-learning the")
print("    existing HETDEX classifier; room for improvement at margins.")
print("  High counterpart_mag importance -> imaging depth drives separation.")
print("  High wave importance -> wavelength encodes redshift information.")
print("  High sigma importance -> line width separates broad (agn/lzg) from")
print("    narrow (lae/star) profiles.")
print("  Confusion between lae/oii -> the primary HETDEX classification")
print("    challenge; low-z OII at z~0.4 mimics Lya at z~2.4 in 1D spectra.")
