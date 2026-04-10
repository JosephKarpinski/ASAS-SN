"""
WASP-39 Stellar Characterization -- YaPSI 2016 Grid
====================================================
Spada, Demarque, Kim, Boyajian & Brewer (2017), ApJ, 838, 161
http://www.astro.yale.edu/yapsi/

REAL FILE FORMAT (confirmed from screenshot of M0p15_X0p602357_Z0p027643_A1p91804.trk):

  Directory layout:
    yapsi_tracks/
      X0p602357_Z0p027643/     <- X=0.602357, Z=0.027643, Y=0.370 -> [Fe/H]=+0.36
        M0p15_X0p602357_Z0p027643_A1p91804.trk
        M0p93_...
        ...
      X0p615836_Z0p014164/     <- [Fe/H]=+0.06
        ...

  Track file structure:
    # comment header lines (mass and X,Z extracted from here)
    # Model_no  Shells  Age(Gyr)  Y_center  Z_center  log(L/Lsun)  log(R/Rsun)  log g
    # ...further column descriptions...
    [data: each model point = 3 wrapped sub-lines, ~29 columns total]
    [IMPORTANT: column count varies between low-mass and high-mass tracks]

  CONFIRMED COLUMN MAPPING (0-based, invariant across all masses):
    idx 0 : Model_no      (integer)
    idx 1 : Shells        (integer)
    idx 2 : Age_Gyr       -- age in GYR directly (NOT years, NOT log years)
    idx 3 : Y_center      -- central helium (or X_center, confirmed by X~0.60 match)
    idx 4 : Z_center
    idx 5 : log(L/Lsun)  -- log luminosity
    idx 6 : log(R/Rsun)  -- log radius
    idx 7 : log g         -- surface gravity
    idx 19: r_bcz         (if present)
    idx 20: tau_c(d)      (if present)

  NO Teff column stored -- derived via:
    log_Teff = log10(5778) + (log_L - 2 * log_R) / 4

  PARSER STRATEGY: line-by-line detection.
    A model's first sub-line is identified by: first token = positive integer (model_no),
    second token = positive integer (shells > 10), third token = float in (0, 25) Gyr.
    Continuation sub-lines are appended to form the full column vector.
    This handles variable column counts robustly without assuming a fixed NCOLS.

USAGE
-----
  1. Set TRACK_DIR to your yapsi_tracks/ directory.
  2. Run:  python wasp39_yapsi2016.py
  3. Outputs: printed chi-sq table + wasp39_yapsi_fit.png

WASP-39 adopted parameters (Faedi+2011, Mancini+2018, Fischer+2016):
  Teff = 5400 +/- 150 K    log g = 4.49 +/- 0.05
  [Fe/H] = -0.12 +/- 0.10  Mass  = 0.93 +/- 0.04 Msun
"""

import re, sys, math, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

TRACK_DIR    = "./yapsi_tracks"
FIT_AGES_GYR = np.arange(1.0, 13.5, 0.5)
OUT_FIGURE   = "wasp39_yapsi_fit.png"

OBS = dict(
    Teff  = 5400.0,  Teff_e = 150.0,
    logg  = 4.49,    logg_e = 0.05,
    feh   = -0.12,   feh_e  = 0.10,
    mass  = 0.93,    mass_e = 0.04,
)

# Column indices within the grouped row (after stripping model_no+shells from each sub-line).
# CONFIRMED from real YaPSI 2016 raw file (yapsi_rawdump.py output):
#   35 columns per line, single physical line per model point.
#   After stripping model_no(col1) and shells(col2):
#     row[0]=Age(Gyr)    row[1]=X_center  row[2]=Y_center  row[3]=Z_center
#     row[4]=log(L/Lsun) row[5]=log(R/Rsun) row[6]=log_g  row[7]=log(Teff)
#     row[8]=m_core/M   row[9]=m_envp/M  row[10]=r_bcz   row[11]=tau_c(d) ...
# Change ONLY if yapsi_colmap.py shows a different layout for your download.
COL_AGE    = 0   # Age(Gyr)
COL_LOGL   = 4   # log(L/Lsun)
COL_LOGR   = 5   # log(R/Rsun)
COL_LOGG   = 6   # log g
COL_LOGTEFF= 7   # log(Teff) -- stored in file, no Stefan-Boltzmann needed

X_SUN    = 0.7158
Z_SUN    = 0.0142
LOG_TSUN = math.log10(5778.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 -- FOLDER AND FILENAME DECODING
# ─────────────────────────────────────────────────────────────────────────────

def decode_p(token):
    """'0p602357' -> 0.602357"""
    return float(token.replace("p", "."))


def parse_folder_composition(name):
    """
    Parse X0p602357_Z0p027643 -> (X, Z, Y, feh) or None.
    """
    m = re.match(r"X([\dp]+)_Z([\dp]+)$", name, re.I)
    if not m:
        return None
    X   = decode_p(m.group(1))
    Z   = decode_p(m.group(2))
    Y   = round(1.0 - X - Z, 8)
    feh = math.log10(Z / X) - math.log10(Z_SUN / X_SUN)
    return X, Z, Y, feh


def parse_track_mass(filepath):
    """M0p93_... -> 0.93,  M1p20_... -> 1.20"""
    stem = Path(filepath).stem
    m = re.match(r"M([\dp]+)[_\-]", stem, re.I)
    if m:
        return decode_p(m.group(1))
    m2 = re.search(r"\bM([\dp]+)\b", stem, re.I)
    if m2:
        return decode_p(m2.group(1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 -- GRID DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

# Target helium for composition selection. Y=0.28 is the solar-calibrated
# value in YaPSI 2016 and the correct choice for most field star applications.
# The real grid ships 5 Y values per [Fe/H] level (0.25, 0.28, 0.31, 0.34, 0.38).
# We select the folder whose Y is closest to this target.
Y_TARGET = 0.28


def discover_grid(track_dir):
    """
    Find all X0p..._Z0p... composition sub-folders and catalogue them.

    The YaPSI 2016 grid has MULTIPLE FOLDERS per [Fe/H] level, one per
    helium abundance Y (0.25, 0.28, 0.31, 0.34, 0.38).  We group by
    round([Fe/H], 2) and select the folder whose Y is closest to Y_TARGET
    (default 0.28, solar-calibrated).  This avoids the previous bug where
    filesystem-order determined which Y was used.

    Returns dict: round(feh, 2) -> {X, Z, Y, feh, folder, files[]}
    """
    track_dir = Path(track_dir)
    if not track_dir.exists():
        sys.exit(f"\nERROR: '{track_dir}' not found.\n"
                 f"  Set TRACK_DIR to your yapsi_tracks/ directory.\n")

    # Collect ALL valid composition folders
    all_comps = []
    subdirs   = [d for d in track_dir.iterdir() if d.is_dir()]
    for d in subdirs:
        comp = parse_folder_composition(d.name)
        if comp is None:
            continue
        X, Z, Y, feh = comp
        files = sorted(d.glob("*.trk")) or sorted(d.glob("*.dat"))
        if not files:
            continue
        all_comps.append(dict(X=X, Z=Z, Y=Y, feh=feh, folder=d, files=files))

    if not all_comps:
        # Flat layout fallback: all .trk in one directory
        files_all = sorted(track_dir.glob("*.trk")) + sorted(track_dir.glob("*.dat"))
        if not files_all:
            sys.exit(f"\nERROR: No .trk/.dat files found in '{track_dir}'.\n")
        groups = {}
        for fp in files_all:
            zm  = re.search(r"Z([\dp]+)", fp.stem, re.I)
            xm  = re.search(r"X([\dp]+)", fp.stem, re.I)
            Z   = decode_p(zm.group(1)) if zm else Z_SUN
            X   = decode_p(xm.group(1)) if xm else X_SUN
            Y   = round(1 - X - Z, 8)
            feh = math.log10(Z / X) - math.log10(Z_SUN / X_SUN)
            key = round(feh, 2)
            if key not in groups:
                groups[key] = dict(X=X, Z=Z, Y=Y, feh=feh,
                                   folder=track_dir, files=[])
            groups[key]["files"].append(fp)
        if not groups:
            sys.exit(f"\nERROR: No YaPSI compositions found in '{track_dir}'.\n")
        return dict(sorted(groups.items()))

    # Group by round(feh, 2) and select best Y within each group
    from collections import defaultdict
    groups = defaultdict(list)
    for comp in all_comps:
        key = round(comp["feh"], 2)
        groups[key].append(comp)

    catalogue = {}
    for key, comps in groups.items():
        # Pick the composition with Y closest to Y_TARGET
        best = min(comps, key=lambda c: abs(c["Y"] - Y_TARGET))
        catalogue[key] = best

    if not catalogue:
        sys.exit(f"\nERROR: No YaPSI composition folders found in '{track_dir}'.\n")

    return dict(sorted(catalogue.items()))


def bracket_feh(target, available):
    avail = np.sort(list(available))
    if target <= avail[0]:
        return avail[0], avail[0], 0.0
    if target >= avail[-1]:
        return avail[-1], avail[-1], 0.0
    idx = np.searchsorted(avail, target) - 1
    flo, fhi = avail[idx], avail[idx + 1]
    return flo, fhi, (target - flo) / (fhi - flo)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 -- PARSE A SINGLE TRACK FILE  (linewise, variable-column-safe)
# ─────────────────────────────────────────────────────────────────────────────

def parse_track_file(filepath):
    """
    Parse one YaPSI 2016 .trk file by grouping lines on model_no.

    CONFIRMED REAL FORMAT (from diagnostic output):
      Every sub-line of a model point starts with  model_no  shells
      Sub-line 1:  model_no  shells  age_gyr  Y_c  Z_c  log_L  log_R  log_g  ...
      Sub-line 2:  model_no  shells  M_Hsh  DM_Hsh  M_He  ...
      Sub-line 3:  model_no  shells  m_core_M  m_env_M  r_bcz  ...

    The parser groups all lines sharing the same model_no, strips the leading
    model_no+shells from each sub-line, then reads columns from the concatenated
    remainder.  Column indices within the concatenated row (after stripping):
      [0] age_gyr   [1] Y_center  [2] Z_center
      [3] log_L     [4] log_R     [5] log_g   ...rest of sub-line 1...
      then sub-line 2 data, then sub-line 3 data

    This approach also handles the alternative format where sub-lines 2 & 3
    do NOT repeat model_no+shells (in that case they are treated as orphan lines
    and simply ignored; the 6 columns from sub-line 1 are sufficient).

    Teff derived: log_Teff = log10(5778) + (log_L - 2*log_R) / 4
    """
    fp   = Path(filepath)
    mass = parse_track_mass(fp)
    meta = {}

    try:
        text = fp.read_text(errors="replace")
    except OSError:
        return None

    # model_no -> list of floats (accumulated from all sub-lines)
    model_data = {}

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue

        # Header comment -- mine for mass and composition
        if s.startswith("#"):
            mm = re.search(r"Mtot/Msun\s*=\s*([\d.E+\-]+)", s, re.I)
            if mm and mass is None:
                try:
                    mass = float(mm.group(1))
                except ValueError:
                    pass
            if "Inertial" in s or "inertial" in s:
                xm = re.search(r"X\s*=\s*([\d.E+\-]+)", s)
                zm = re.search(r"Z\s*=\s*([\d.E+\-]+)", s)
                if xm:
                    try:
                        meta["X"] = float(xm.group(1))
                    except ValueError:
                        pass
                if zm:
                    try:
                        meta["Z"] = float(zm.group(1))
                    except ValueError:
                        pass
            continue

        # Parse numeric tokens from data line
        vals = []
        for t in s.split():
            try:
                vals.append(float(t))
            except ValueError:
                pass
        if len(vals) < 3:
            continue

        # Check if line starts with model_no + shells
        v0, v1 = vals[0], vals[1]
        if (v0 == int(v0) and int(v0) >= 1 and
                v1 == int(v1) and int(v1) > 10):
            model_no = int(v0)
            # Strip model_no + shells; keep the rest
            data_vals = vals[2:]
            if model_no not in model_data:
                model_data[model_no] = list(data_vals)
            else:
                model_data[model_no].extend(data_vals)
        # Lines that don't match model_no+shells are silently ignored
        # (they are continuation sub-lines in the alternative format that
        #  don't carry additional columns we need)

    if not model_data:
        return None

    if mass is None:
        mass = meta.get("mass_header")
    if mass is None:
        return None

    # Extract physical columns using confirmed indices from raw file analysis
    # Real YaPSI 2016 format: 35 cols/line, after stripping model_no+shells:
    #   row[0]=age  row[1]=X_c  row[2]=Y_c  row[3]=Z_c
    #   row[4]=log_L  row[5]=log_R  row[6]=log_g  row[7]=log_Teff  row[8]=m_core ...
    age_list   = []
    logL_list  = []
    logR_list  = []
    logg_list  = []
    logTeff_list = []   # stored log(Teff) from file
    rbcz_list  = []
    tauc_list  = []

    for model_no in sorted(model_data.keys()):
        row = model_data[model_no]
        if len(row) <= COL_LOGG:
            continue
        age_gyr = row[COL_AGE]
        log_L   = row[COL_LOGL]
        log_R   = row[COL_LOGR]
        log_g   = row[COL_LOGG]
        # Stored log(Teff) -- available in real YaPSI 2016 files at row[7]
        log_Tf  = row[COL_LOGTEFF] if len(row) > COL_LOGTEFF else None
        # r_bcz and tau_c at confirmed positions (cols 13 and 14, row idx 10 and 11)
        r_bcz   = row[10] if len(row) > 10 else np.nan
        tau_c   = row[11] if len(row) > 11 else np.nan

        # Physical sanity filters
        if not (0.0 < age_gyr < 25.0): continue
        if not (-8.0 < log_L < 8.0):   continue
        if not (-1.5 < log_R < 3.0):   continue
        if not (0.5  < log_g < 7.0):   continue

        age_list.append(age_gyr)
        logL_list.append(log_L)
        logR_list.append(log_R)
        logg_list.append(log_g)
        logTeff_list.append(log_Tf)
        rbcz_list.append(r_bcz)
        tauc_list.append(tau_c)

    if len(age_list) < 2:
        return None

    age_arr  = np.array(age_list)
    log_L    = np.array(logL_list)
    log_R    = np.array(logR_list)
    log_g    = np.array(logg_list)
    R        = 10.0 ** np.clip(log_R, -1.0, 2.5)

    # Prefer stored log(Teff); fall back to Stefan-Boltzmann
    sb_logT  = LOG_TSUN + (log_L - 2.0 * log_R) / 4.0
    stored   = [v for v in logTeff_list if v is not None]
    if stored and 2.5 < float(np.nanmean(stored)) < 5.5:
        log_Teff = np.array([v if (v is not None and 2.5 < v < 5.5)
                              else sb_logT[i]
                              for i, v in enumerate(logTeff_list)], dtype=float)
    else:
        log_Teff = sb_logT
    Teff = 10.0 ** np.clip(log_Teff, 2.5, 5.5)

    X   = meta.get("X", X_SUN)
    Z   = meta.get("Z", Z_SUN)
    Y   = round(1.0 - X - Z, 8)
    feh = math.log10(Z / X) - math.log10(Z_SUN / X_SUN)

    return {
        "age_gyr":  age_arr,
        "log_L":    log_L,
        "log_R":    log_R,
        "log_g":    log_g,
        "Teff":     Teff,
        "log_Teff": log_Teff,
        "R":        R,
        "r_bcz":    np.array(rbcz_list),
        "tau_c_d":  np.array(tauc_list),
        "_mass":    mass,
        "_X": X, "_Z": Z, "_Y": Y, "_feh": feh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 -- BUILD ISOCHRONES FROM TRACKS
# ─────────────────────────────────────────────────────────────────────────────

def build_isochrones(track_files, target_ages_gyr, label=""):
    """
    Load all tracks and interpolate to target ages (direct Gyr).
    Returns dict: age_gyr -> {mass, Teff, log_L, log_g, R, ...}
    """
    tracks = []
    for fp in track_files:
        data = parse_track_file(fp)
        if data is not None:
            tracks.append((data["_mass"], data))

    if not tracks:
        return {}

    tracks.sort(key=lambda t: t[0])
    masses = [t[0] for t in tracks]
    print(f"  {label}Loaded {len(tracks)} tracks  "
          f"({masses[0]:.3f} – {masses[-1]:.3f} Msun)")

    sample = tracks[len(tracks) // 2][1]
    a      = sample["age_gyr"]
    print(f"  {label}Age range (sample): {a.min():.5f} – {a.max():.2f} Gyr")

    isochrones = {}
    n_built    = 0

    for age_gyr in target_ages_gyr:
        pts = {k: [] for k in ["mass", "Teff", "log_Teff", "log_L", "log_g", "R"]}

        for mass, data in tracks:
            age_arr = data["age_gyr"]
            if len(age_arr) < 2:
                continue
            if age_gyr < age_arr.min() or age_gyr > age_arr.max():
                continue

            def _interp(col):
                if col not in data or not np.any(np.isfinite(data[col])):
                    return np.nan
                try:
                    return float(interp1d(age_arr, data[col], kind="linear",
                                          bounds_error=False,
                                          fill_value=np.nan)(age_gyr))
                except Exception:
                    return np.nan

            pts["mass"].append(mass)
            pts["Teff"].append(_interp("Teff"))
            pts["log_Teff"].append(_interp("log_Teff"))
            pts["log_L"].append(_interp("log_L"))
            pts["log_g"].append(_interp("log_g"))
            pts["R"].append(_interp("R"))

        if len(pts["mass"]) < 3:
            continue

        for k in pts:
            pts[k] = np.array(pts[k], dtype=float)

        good = (np.isfinite(pts["Teff"]) & np.isfinite(pts["log_g"])
                & (pts["Teff"] > 2000) & (pts["Teff"] < 50000)
                & (pts["log_g"] > 0.5) & (pts["log_g"] < 6.5))
        if good.sum() < 3:
            continue
        for k in pts:
            pts[k] = pts[k][good]

        order = np.argsort(pts["mass"])
        for k in pts:
            pts[k] = pts[k][order]

        isochrones[age_gyr] = pts
        n_built += 1

    print(f"  {label}Constructed {n_built}/{len(target_ages_gyr)} isochrones")
    return isochrones


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 -- [Fe/H] INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_feh(isos_lo, isos_hi, weight):
    """
    Linear interpolation in [Fe/H] between two isochrone sets.

    Uses the INTERSECTION of both mass grids to avoid extrapolation artifacts.
    fill_value=np.nan (not 'extrapolate') ensures no garbage outside coverage.
    Drops any rows where key columns are NaN after interpolation.
    """
    if weight == 0.0 or not isos_hi:
        return isos_lo
    if weight == 1.0 or not isos_lo:
        return isos_hi

    result = {}
    for age in sorted(set(isos_lo) & set(isos_hi)):
        lo, hi = isos_lo[age], isos_hi[age]
        mass_lo, mass_hi = lo["mass"], hi["mass"]

        # Common mass range = strict intersection of both grids
        mass_min = max(mass_lo.min(), mass_hi.min())
        mass_max = min(mass_lo.max(), mass_hi.max())
        if mass_max <= mass_min:
            result[age] = lo
            continue

        common_mask = (mass_lo >= mass_min) & (mass_lo <= mass_max)
        if common_mask.sum() < 3:
            result[age] = lo
            continue

        mass_common = mass_lo[common_mask]
        merged = {"mass": mass_common}

        for col in ["Teff", "log_Teff", "log_L", "log_g", "R"]:
            lo_col = lo.get(col)
            hi_col = hi.get(col)
            if lo_col is None:
                continue
            lo_vals = lo_col[common_mask]
            if hi_col is None or len(hi_col) < 2:
                merged[col] = lo_vals
                continue
            try:
                # NaN outside hi range -- never extrapolate
                hi_vals = interp1d(mass_hi, hi_col, kind="linear",
                                   bounds_error=False,
                                   fill_value=np.nan)(mass_common)
            except Exception:
                merged[col] = lo_vals
                continue
            both = np.isfinite(lo_vals) & np.isfinite(hi_vals)
            out  = np.where(both,
                            (1 - weight) * lo_vals + weight * hi_vals,
                            np.nan)
            merged[col] = out

        # Drop rows where Teff or log_g are NaN
        good = (np.isfinite(merged.get("Teff",  np.array([np.nan]))) &
                np.isfinite(merged.get("log_g", np.array([np.nan]))))
        if good.sum() < 3:
            continue
        for col in list(merged.keys()):
            v = merged[col]
            if isinstance(v, np.ndarray) and len(v) == len(good):
                merged[col] = v[good]
        result[age] = merged

    for age in set(isos_lo) - set(isos_hi):
        result[age] = isos_lo[age]
    for age in set(isos_hi) - set(isos_lo):
        result[age] = isos_hi[age]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 -- CHI-SQUARED FITTING
# ─────────────────────────────────────────────────────────────────────────────

def chi2_scan(isochrones, obs):
    results = []
    for age_gyr, iso in sorted(isochrones.items()):
        mass_arr = iso["mass"]
        Teff_arr = iso["Teff"]
        logg_arr = iso["log_g"]
        logL_arr = iso.get("log_L", np.full_like(mass_arr, np.nan))
        R_arr    = iso.get("R",     np.full_like(mass_arr, np.nan))

        if len(mass_arr) < 3:
            continue

        # Restrict to MS/subgiant regime relevant for WASP-39
        # (avoid PMS, RGB, and white dwarf branches in the fit)
        ms_mask = (
            (Teff_arr > 3500) & (Teff_arr < 8000) &
            (logg_arr > 3.5)  & (logg_arr < 5.5) &
            (mass_arr > 0.3)  & (mass_arr < 2.5)
        )
        if ms_mask.sum() < 3:
            continue   # no MS-range points at this age -- skip

        chi2_arr = (
            ((Teff_arr - obs["Teff"]) / obs["Teff_e"])**2 +
            ((logg_arr - obs["logg"]) / obs["logg_e"])**2 +
            ((mass_arr - obs["mass"]) / obs["mass_e"])**2
        )
        chi2_arr[~ms_mask] = np.inf

        best = int(np.nanargmin(chi2_arr))
        if not np.isfinite(chi2_arr[best]):
            continue

        results.append((
            float(age_gyr),
            float(chi2_arr[best]),
            float(mass_arr[best]),
            float(Teff_arr[best]),
            float(logg_arr[best]),
            float(logL_arr[best]) if np.isfinite(logL_arr[best]) else np.nan,
            float(R_arr[best])    if np.isfinite(R_arr[best])    else np.nan,
        ))

    return sorted(results, key=lambda r: r[0])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 -- PLOTTING  (improved)
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(isochrones, results, obs, best, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")

    best_age  = best[0]
    best_logL = best[5] if not np.isnan(best[5]) else -0.15
    best_Teff = best[3]

    sorted_ages = sorted(isochrones)
    age_min, age_max = sorted_ages[0], sorted_ages[-1]
    cmap = plt.cm.plasma

    def age_color(age):
        t = (age - age_min) / max(age_max - age_min, 1)
        return cmap(0.1 + 0.8 * t)

    # ── Left: HR diagram tight G-star MS window ────────────────────────────
    ax = axes[0]
    Teff_lo, Teff_hi = 4800, 6500
    logL_lo,  logL_hi = -0.55, 0.55
    label_ages = {2, 4, 6, 8, 10, 12}

    for age in sorted_ages:
        iso  = isochrones[age]
        Teff = iso.get("Teff", np.array([]))
        logL = iso.get("log_L", np.full_like(Teff, np.nan))
        ms_win = (np.isfinite(Teff) & np.isfinite(logL) &
                  (Teff > Teff_lo) & (Teff < Teff_hi) &
                  (logL > logL_lo) & (logL < logL_hi))
        if ms_win.sum() < 2:
            continue
        is_best = np.isclose(age, best_age, atol=0.3)
        color   = "#00e5a0" if is_best else age_color(age)
        lw      = 2.8 if is_best else 0.75
        alpha_v = 1.0 if is_best else 0.50
        label   = (f"{age:.0f} Gyr (best)" if is_best
                   else (f"{age:.0f} Gyr" if round(age) in label_ages else None))
        ax.plot(Teff[ms_win], logL[ms_win], color=color, lw=lw,
                alpha=alpha_v, label=label, zorder=6 if is_best else 2)
        if label and not is_best and round(age) in label_ages:
            idx_hot = int(np.argmax(Teff[ms_win]))
            ax.text(Teff[ms_win][idx_hot] + 25, logL[ms_win][idx_hot],
                    f"{age:.0f}", fontsize=6.5, color=color,
                    alpha=0.85, ha="left", va="center")

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=age_min, vmax=age_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.82)
    cbar.set_label("Isochrone age (Gyr)", color="#c9d1d9", fontsize=8.5)
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e", fontsize=7.5)

    ax.scatter([5778], [0.0], marker="*", s=260, color="#F9CB42",
               zorder=7, label="Sun", edgecolors="white", linewidths=0.5)

    # WASP-39 shaded error box + point
    import matplotlib.patches as mpatches
    rect = mpatches.FancyBboxPatch(
        (obs["Teff"] - obs["Teff_e"], -0.22),
        2 * obs["Teff_e"], 0.44,
        boxstyle="square,pad=0", linewidth=0,
        facecolor="#ff4d6d", alpha=0.12, zorder=4)
    ax.add_patch(rect)
    ax.errorbar(obs["Teff"], best_logL,
                xerr=obs["Teff_e"], yerr=0.05,
                fmt="o", color="#ff4d6d", ms=10, capsize=4, capthick=1.5,
                ecolor="#ff8fa3", elinewidth=1.5, label="WASP-39", zorder=8)
    # Best-fit model diamond
    ax.plot(best_Teff, best_logL, "D", color="#00e5a0", ms=7,
            markeredgecolor="#0d1117", markeredgewidth=0.5, zorder=9,
            label=f"Best fit ({best_age:.0f} Gyr)")

    ax.annotate(
        (f"WASP-39  G8V\n"
         f"Teff = {obs['Teff']:.0f} K\n"
         f"log g = {obs['logg']:.2f}\n"
         f"[Fe/H] = {obs['feh']:+.2f}"),
        xy=(obs["Teff"], best_logL), xytext=(6280, 0.42),
        fontsize=8, color="#ff8fa3", ha="right",
        arrowprops=dict(arrowstyle="->", color="#ff8fa3", lw=0.9,
                        connectionstyle="arc3,rad=0.15"),
        bbox=dict(boxstyle="round,pad=0.4", fc="#161b22",
                  ec="#ff4d6d", alpha=0.90, lw=1.2))

    ax.set_xlabel("Effective temperature  Teff (K)", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Log luminosity  (L / Lsun)", color="#c9d1d9", fontsize=11)
    ax.set_title(f"YaPSI 2016 isochrones  |  WASP-39  [Fe/H] = {obs['feh']:+.2f}",
                 color="#e6edf3", fontsize=10.5, pad=8)
    ax.invert_xaxis()
    ax.set_xlim(Teff_hi + 50, Teff_lo - 50)
    ax.set_ylim(logL_lo, logL_hi)
    ax.tick_params(colors="#8b949e", which="both")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(300))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, which="major", color="#21262d", lw=0.6, alpha=0.9)
    ax.grid(True, which="minor", color="#1c2128", lw=0.3, alpha=0.5)
    h, la = ax.get_legend_handles_labels()
    labeled = [(hh, ll) for hh, ll in zip(h, la) if ll]
    if labeled:
        hh, ll = zip(*labeled)
        ax.legend(hh, ll, framealpha=0.35, facecolor="#161b22",
                  edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=7.5,
                  loc="lower left", ncol=2, columnspacing=0.8, handlelength=1.2)

    # ── Right: chi^2 vs Age ─────────────────────────────────────────────────
    ax2      = axes[1]
    age_arr  = np.array([r[0] for r in results])
    chi2_arr = np.array([r[1] for r in results])
    min_chi2 = chi2_arr.min()
    chi2_range = chi2_arr.max() - min_chi2

    ax2.plot(age_arr, chi2_arr, color="#3d4451", lw=1.2, zorder=2)
    for age, chi2 in zip(age_arr, chi2_arr):
        ax2.plot(age, chi2, "o", color=age_color(age), ms=8,
                 markeredgecolor="#0d1117", markeredgewidth=0.5, zorder=4)

    age_1sig = age_arr[chi2_arr <= min_chi2 + 1.0]
    sig_lo = age_1sig.min() if len(age_1sig) else best_age
    sig_hi = age_1sig.max() if len(age_1sig) else best_age

    ax2.fill_between(age_arr, min_chi2, chi2_arr,
                     where=(chi2_arr <= min_chi2 + 1),
                     alpha=0.25, color="#00e5a0", zorder=1,
                     label=f"1-sigma: {sig_lo:.1f}-{sig_hi:.1f} Gyr")
    ax2.axvline(best_age, color="#00e5a0", lw=2.0, ls="--", zorder=5,
                label=f"Best fit: {best_age:.1f} Gyr")
    ax2.axhline(min_chi2 + 1, color="#f4a261", lw=1.2, ls=":", alpha=0.9,
                label="Delta chi^2 = 1  (1-sigma)")
    ax2.plot(best_age, min_chi2, "*", color="#00e5a0", ms=18,
             markeredgecolor="#0d1117", markeredgewidth=0.5, zorder=6)

    ax2.annotate(
        (f"Age = {best_age:.1f} Gyr\n"
         f"M = {best[2]:.3f} Msun\n"
         f"Teff = {best[3]:.0f} K\n"
         f"log g = {best[4]:.3f}\n"
         f"log L = {best[5]:.3f}"),
        xy=(best_age, min_chi2),
        xytext=(best_age + 2.2, min_chi2 + chi2_range * 0.28),
        fontsize=8.5, color="#00e5a0",
        arrowprops=dict(arrowstyle="->", color="#00e5a0", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.45", fc="#0d1117",
                  ec="#00e5a0", alpha=0.92, lw=1.2))

    ax2.set_xlabel("Isochrone age  (Gyr)", color="#c9d1d9", fontsize=11)
    ax2.set_ylabel("chi-squared  (Teff + log g + mass)", color="#c9d1d9", fontsize=11)
    ax2.set_title("YaPSI 2016 age constraint  |  WASP-39",
                  color="#e6edf3", fontsize=10.5, pad=8)
    ax2.set_xlim(age_arr.min() - 0.3, age_arr.max() + 0.3)
    ax2.set_ylim(bottom=max(0, min_chi2 - chi2_range * 0.05))
    ax2.tick_params(colors="#8b949e", which="both")
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")
    ax2.grid(True, which="major", color="#21262d", lw=0.6, alpha=0.9)
    ax2.grid(True, which="minor", color="#1c2128", lw=0.3, alpha=0.5)
    ax2.legend(framealpha=0.35, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=9, loc="upper left")

    plt.tight_layout(pad=2.5)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n  Figure saved -> {out_path}")



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 -- MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  WASP-39  x  YaPSI 2016 Isochrone Fitting")
    print("  Spada et al. (2017), ApJ 838, 161")
    print("=" * 65)
    print(f"\n  Track directory : {TRACK_DIR}")
    print(f"  Target [Fe/H]   : {OBS['feh']:+.2f}")
    print()

    # 1. Discover all compositions
    catalogue = discover_grid(TRACK_DIR)
    feh_nodes = sorted(catalogue.keys())
    print(f"  Compositions found ({len(feh_nodes)} nodes):")
    for key, info in sorted(catalogue.items()):
        print(f"    [Fe/H] = {info['feh']:+.3f}"
              f"  (X={info['X']:.6f}, Z={info['Z']:.6f}, Y={info['Y']:.4f})"
              f"  ->  {len(info['files'])} tracks")

    # 2. Bracket WASP-39 [Fe/H]
    feh_lo, feh_hi, feh_w = bracket_feh(OBS["feh"], feh_nodes)
    print(f"\n  [Fe/H] bracketing : [{feh_lo:+.3f}, {feh_hi:+.3f}]"
          f"  weight = {feh_w:.3f}")

    # 3. Build isochrones
    print(f"\n  Building isochrones for [Fe/H] = {feh_lo:+.3f} ...")
    isos_lo = build_isochrones(catalogue[feh_lo]["files"], FIT_AGES_GYR,
                               label=f"[{feh_lo:+.3f}] ")

    if feh_w > 0 and feh_lo != feh_hi:
        print(f"\n  Building isochrones for [Fe/H] = {feh_hi:+.3f} ...")
        isos_hi = build_isochrones(catalogue[feh_hi]["files"], FIT_AGES_GYR,
                                   label=f"[{feh_hi:+.3f}] ")
    else:
        isos_hi = isos_lo
        feh_w   = 0.0

    # 4. Interpolate in [Fe/H]
    print(f"\n  Interpolating to [Fe/H] = {OBS['feh']:+.2f} ...")
    isochrones = interpolate_feh(isos_lo, isos_hi, feh_w)

    if not isochrones:
        sys.exit(
            "\nERROR: No isochrones constructed.\n"
            "  - Check that track files parse correctly (try a single file manually).\n"
            "  - Ensure FIT_AGES_GYR overlaps the track age ranges.\n"
            "  - Ensure mass range covers ~0.93 Msun.\n"
        )

    ages_built = sorted(isochrones)
    print(f"  Final isochrones  : {len(isochrones)}"
          f"  ({ages_built[0]:.1f} – {ages_built[-1]:.1f} Gyr)")

    # 5. Chi-squared scan
    obs_fit = dict(
        Teff   = OBS["Teff"],  Teff_e = OBS["Teff_e"],
        logg   = OBS["logg"],  logg_e = OBS["logg_e"],
        mass   = OBS["mass"],  mass_e = OBS["mass_e"],
    )
    # Diagnostic: sample a few isochrones to verify the data looks physical
    print("  Isochrone spot-check (at key ages):")
    for age_check in [1.0, 5.0, 10.0]:
        if age_check in isochrones:
            ic = isochrones[age_check]
            n_ms = int(np.sum(
                (ic["Teff"]>3500) & (ic["Teff"]<8000) &
                (ic["log_g"]>3.5) & (ic["log_g"]<5.5) &
                (ic["mass"]>0.3)  & (ic["mass"]<2.5)
            ))
            # Find point nearest to WASP-39
            idx = int(np.argmin(np.abs(ic["mass"] - OBS["mass"])))
            print(f"    {age_check:.0f} Gyr: N={len(ic['mass'])}  N_MS={n_ms}"
                  f"  |  at M={ic['mass'][idx]:.3f}:"
                  f"  Teff={ic['Teff'][idx]:.0f} K"
                  f"  log_g={ic['log_g'][idx]:.3f}"
                  f"  log_L={ic['log_L'][idx]:.3f}")
    print()

    results = chi2_scan(isochrones, obs_fit)

    if not results:
        # Extended error with diagnostic information
        print("\nDIAGNOSTIC -- Isochrone data at age=5 Gyr:")
        if 5.0 in isochrones:
            ic = isochrones[5.0]
            print(f"  N points   : {len(ic['mass'])}")
            print(f"  Mass range : {ic['mass'].min():.3f} – {ic['mass'].max():.3f} Msun")
            print(f"  Teff range : {ic['Teff'].min():.0f} – {ic['Teff'].max():.0f} K")
            print(f"  log_g range: {ic['log_g'].min():.3f} – {ic['log_g'].max():.3f}")
            ms = ((ic["Teff"]>3500) & (ic["Teff"]<8000) &
                  (ic["log_g"]>3.5) & (ic["log_g"]<5.5) &
                  (ic["mass"]>0.3)  & (ic["mass"]<2.5))
            print(f"  MS points  : {ms.sum()}")
            if ms.sum() == 0:
                print("\n  CAUSE: No points pass the MS filter.")
                print("  Check that log_g values are ~4.0-4.9 for solar-type MS stars.")
                print("  If log_g values are <3.5, the tracks may have a column shift.")
                print("  Run yapsi_diagnostic.py to inspect individual track files.")
        sys.exit(
            "\nERROR: Chi-squared scan returned no results.\n"
            "  Run yapsi_diagnostic.py first to verify track file parsing.\n"
            "  Common causes:\n"
            "  1. Column count differs from 29 -> parser misaligns log_g (col 8)\n"
            "  2. Y-helium selection: wrong composition folder was chosen\n"
            "  3. Age units: check that Age column is in Gyr (not years or log)\n"
        )

    best = min(results, key=lambda r: r[1])
    best_age, best_chi2, best_mass, best_Teff, best_logg, best_logL, best_R = best

    # 6. Print table
    chi2_arr = np.array([r[1] for r in results])
    age_arr  = np.array([r[0] for r in results])
    age_1sig = age_arr[chi2_arr <= best_chi2 + 1.0]

    print("\n" + "-" * 70)
    print(f"  {'Age (Gyr)':>10}  {'chi^2':>10}  {'Mass':>6}  "
          f"{'Teff':>6}  {'log g':>6}  {'log L':>7}")
    print("  " + "-" * 65)
    for age, chi2, mass, Teff, logg, logL, R in results:
        flag     = "  <- BEST" if np.isclose(age, best_age) else ""
        logL_str = f"{logL:7.3f}" if not np.isnan(logL) else "    ---"
        print(f"  {age:>10.1f}  {chi2:>10.3f}  {mass:>6.3f}  "
              f"{Teff:>6.0f}  {logg:>6.3f}  {logL_str}{flag}")

    print("\n" + "=" * 65)
    print("  DERIVED STELLAR PARAMETERS  (YaPSI 2016 best fit)")
    print("=" * 65)
    print(f"  Best-fit age       :  {best_age:.1f} Gyr")
    if len(age_1sig) >= 2:
        print(f"  1-sigma age range  :  {age_1sig.min():.1f} – {age_1sig.max():.1f} Gyr"
              f"  (delta chi^2 <= 1)")
    print(f"  Best-fit mass      :  {best_mass:.3f} Msun"
          f"   (obs: {OBS['mass']:.3f} +/- {OBS['mass_e']:.3f})")
    print(f"  Best-fit Teff      :  {best_Teff:.0f} K"
          f"       (obs: {OBS['Teff']:.0f} +/- {OBS['Teff_e']:.0f})")
    print(f"  Best-fit log g     :  {best_logg:.3f}"
          f"         (obs: {OBS['logg']:.3f} +/- {OBS['logg_e']:.3f})")
    if not np.isnan(best_logL):
        print(f"  Best-fit log L     :  {best_logL:.3f}  (L/L_sun)")
    if not np.isnan(best_R):
        print(f"  Best-fit R         :  {best_R:.3f} Rsun")
    print(f"  [Fe/H] interpolated:  {OBS['feh']:+.2f}")
    print(f"  Min chi^2          :  {best_chi2:.3f}")
    print()
    print(f"  Method notes:")
    print(f"    Teff from stored log(Teff) column (col 7 in real YaPSI 2016 files).")
    print(f"    Age read directly in Gyr from col 3 of each track file.")
    print(f"    Parser uses line-by-line model detection (variable column count safe).")
    print()

    # 7. Plot
    make_plot(isochrones, results, OBS, best, OUT_FIGURE)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
