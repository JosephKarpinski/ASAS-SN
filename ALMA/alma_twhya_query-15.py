"""
Query, download, and display ALMA archive data for TW Hydrae.

Equivalent to the ALMA Archive Query UI at:
https://almascience.nrao.edu/aq/?sourceNameResolver=TW%20Hydrae

Requires: astroquery, astropy, matplotlib, numpy, scipy
    pip install astroquery astropy matplotlib numpy scipy --break-system-packages

Notes:
- TW Hya is the closest known protoplanetary disk to Earth (~60 pc) and has
  ~80 ALMA observations across many bands/projects (see the archive UI
  screenshot: multiple bands, ~54 projects, 96 publications). Unlike HL Tau's
  2014 Long Baseline Campaign, TW Hya has no single standalone "science
  verification" .tgz reference-image package published outside the normal
  archive — so this script uses the general query -> inspect -> filter ->
  download path all the way through, rather than a hardcoded external URL.
- The most famous TW Hya continuum image (the sharp, multi-ring "ALMA's best
  disk image" release, Andrews et al. 2016, ApJL) comes from project
  2015.1.00686.S (Band 7, high angular resolution). Pass that proposal_id to
  target it specifically; leave it as None to browse everything.
- As with the HL Tau script, this avoids downloading raw/calibrated
  tarballs wholesale where possible. For projects like this one, ALMA only
  exposes whole .tar packages (not individual FITS URLs) through the
  datalink listing, so the script downloads the smallest relevant product
  tarball and extracts just the FITS members from it, deleting the tar
  afterward, rather than keeping the full multi-GB package.
"""

import os
import re
import tarfile
import time
import traceback
import urllib.request
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
from astropy.wcs import WCS, FITSFixedWarning
from astroquery.alma import Alma

# Older ALMA FITS headers use legacy PCi_j / 4-axis (RA/Dec/Freq/Stokes) WCS
# conventions that astropy "fixes" automatically and warns about on every
# load. The fix is correct and harmless, so silence the warnings.
warnings.filterwarnings("ignore", category=FITSFixedWarning)

CACHE_DIR = "./alma_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Set True to print step-by-step checkpoints inside the radial-profile
# functions -- useful for tracing exactly where execution stops if a cell
# silently produces no output/plot.
DEBUG = True


def _dbg(*args):
    if DEBUG:
        print("[DEBUG]", *args)

# ALMA's datalink/TAP service (used by get_data_info) intermittently throws
# transient 502/503 "Proxy Error" responses under load, even for perfectly
# valid UIDs -- this is server-side flakiness, not a query bug. astroquery
# lets you point at different regional mirrors that all serve the same
# archive, so we retry with backoff and rotate mirrors if one is down.
ALMA_MIRRORS = [
    "https://almascience.nrao.edu",
    "https://almascience.eso.org",
    "https://almascience.nao.ac.jp",
]

# Literature substructure radii for TW Hya, for visual/numeric sanity-checking
# our own detections. Reported by Tsukagoshi et al. 2016 (ApJL 829, L35),
# consistent with Andrews et al. 2016 (ApJL 820, L40). These papers assumed
# a pre-Gaia distance of ~54 pc; scale to whatever distance_pc this pipeline
# used (e.g. the Gaia DR2/DR3 value ~60.1 pc) before comparing, since a
# ~11% distance difference is a ~11% radius difference, not a detection
# discrepancy.
TSUKAGOSHI2016_TWHYA = {
    "distance_pc": 54.0,
    "central_hole_au": 1.0,
    "deep_gaps_au": [22.0, 37.0],
    "shallow_gaps_au": [6.0, 28.0, 44.0],
}


def literature_features_au(target_distance_pc):
    """
    Return TW Hya literature gap radii (Tsukagoshi et al. 2016 / Andrews et
    al. 2016) rescaled from their assumed distance to target_distance_pc,
    as a list of (radius_au, label, kind) tuples for plotting/comparison.
    kind is one of "hole", "deep", "shallow".
    """
    lit = TSUKAGOSHI2016_TWHYA
    scale = target_distance_pc / lit["distance_pc"]
    features = [(lit["central_hole_au"] * scale, "central hole (lit.)", "hole")]
    features += [(r * scale, "deep gap (lit.)", "deep") for r in lit["deep_gaps_au"]]
    features += [(r * scale, "shallow gap (lit.)", "shallow") for r in lit["shallow_gaps_au"]]
    return features


def robust_get_data_info(alma, member_ous_uid, expand_tarfiles=True,
                          max_attempts=3, base_delay=5):
    """
    Wrapper around alma.get_data_info() that retries transient server
    errors (502/503/DALServiceError) with exponential backoff, and falls
    back to alternate ALMA mirrors if a given mirror keeps failing.
    """
    last_exc = None
    for mirror in ALMA_MIRRORS:
        alma.archive_url = mirror
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"  [mirror={mirror}, attempt {attempt}/{max_attempts}] "
                      f"fetching data info for {member_ous_uid} ...")
                return alma.get_data_info(member_ous_uid,
                                           expand_tarfiles=expand_tarfiles)
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                transient = any(code in msg for code in
                                 ("502", "503", "504", "DALServiceError",
                                  "Proxy Error", "Gateway"))
                if not transient or attempt == max_attempts:
                    break
                delay = base_delay * attempt
                print(f"  Transient error ({exc.__class__.__name__}); "
                      f"retrying in {delay}s ...")
                time.sleep(delay)
        print(f"  Mirror {mirror} did not succeed, trying next mirror ...")
    raise RuntimeError(
        f"get_data_info failed for {member_ous_uid} across all mirrors "
        f"{ALMA_MIRRORS}. Last error: {last_exc}"
    )


def query_tw_hya(public=True, science=True, proposal_id=None):
    """
    Query the ALMA archive for TW Hydrae observations, filtered strictly to
    TW Hya (query_object is a loose sky-position match and can pull in
    nearby/related fields, so we mask down to rows that actually target it).

    proposal_id: if given (e.g. "2015.1.00686.S"), further restrict results
        to that specific proposal — useful for grabbing the well-known
        Andrews et al. 2016 high-resolution Band 7 ring image rather than
        whatever UID happens to come up first among TW Hya's ~80 rows.
    """
    alma = Alma()
    alma.cache_location = CACHE_DIR

    print("Querying ALMA archive for source: TW Hydrae ...")
    results = alma.query_object("TW Hydrae", public=public, science=science)
    print(f"Found {len(results)} raw rows (query_object can return nearby/related fields too).")

    name_col = "target_name" if "target_name" in results.colnames else "obs_id"
    name_mask = np.array([
        ("tw_hya" in str(val).lower().replace(" ", "_")
         or "twhya" in str(val).lower().replace(" ", "_"))
        for val in results[name_col]
    ])
    results = results[name_mask]
    print(f"Filtered to {len(results)} rows matching TW Hydrae.")

    if proposal_id is not None and "proposal_id" in results.colnames:
        prop_mask = np.array([
            str(val).strip().upper() == proposal_id.strip().upper()
            for val in results["proposal_id"]
        ])
        results = results[prop_mask]
        print(f"Further filtered to {len(results)} rows for proposal {proposal_id}.")

    print()
    cols = [c for c in [
        "obs_id", "target_name", "member_ous_uid", "band_list", "frequency",
        "spatial_resolution", "velocity_resolution", "t_exptime",
        "proposal_id", "obs_release_date", "science_keyword",
    ] if c in results.colnames]

    print(results[cols])
    return alma, results


def inspect_products(alma, member_ous_uid, expand_tarfiles=True):
    """
    List the individual data files available for one ALMA project (UID).

    expand_tarfiles=True asks the archive to list files *inside* the
    delivered .tar packages (e.g. calibrated FITS images) instead of just
    the tar itself — this is what lets us grab a single FITS image without
    downloading the whole (often multi-GB) tarball.
    """
    print(f"\nFetching file listing for {member_ous_uid} (expand_tarfiles={expand_tarfiles}) ...")
    data_info = robust_get_data_info(alma, member_ous_uid,
                                      expand_tarfiles=expand_tarfiles)
    print(data_info["access_url", "content_length", "content_type"])
    return data_info


def pick_product_tarball(data_info):
    """
    Some ALMA deliveries (like this TW Hya Band 7 project) don't expose
    individual FITS files through the datalink listing at all -- even with
    expand_tarfiles=True, get_data_info only returns whole .tar packages:
    the main pipeline-products tar, an "auxiliary" tar, raw ASDM tars, and
    an "external_ari_l" tar (reprocessed ARI-L products). In that case we
    can't filter by ".fits" URL suffix; we have to pick the right tarball
    and extract FITS members from it after downloading.

    Preference order:
      1. The plain main product tar (contains calibrated pipeline images):
         "<proposal>_uid___...__NNN_of_NNN.tar", excluding auxiliary/asdm.
      2. The external_ari_l tar (ARI-L reprocessed images), as a fallback.
    Raw ASDM tars and the auxiliary tar are skipped -- they hold
    uncalibrated/ancillary data, not science-ready images, and are much
    larger than needed just to grab a continuum image.
    """
    urls = [str(u) for u in data_info["access_url"]]

    def is_main_product_tar(u):
        lu = u.lower()
        return (lu.endswith(".tar")
                and "auxiliary" not in lu
                and "asdm" not in lu
                and "external_ari_l" not in lu
                and "readme" not in lu)

    def is_ari_l_tar(u):
        lu = u.lower()
        return lu.endswith(".tar") and "external_ari_l" in lu

    main_candidates = [u for u in urls if is_main_product_tar(u)]
    if main_candidates:
        return main_candidates[0]

    ari_l_candidates = [u for u in urls if is_ari_l_tar(u)]
    if ari_l_candidates:
        print("No plain main-product tar found; falling back to the "
              "external_ari_l (reprocessed) tarball.")
        return ari_l_candidates[0]

    print("No suitable product tarball found -- only auxiliary/raw ASDM "
          "tars are available for this UID. Inspect data_info manually.")
    return None


def download_and_extract_fits(alma, tarball_url, dest_dir=CACHE_DIR,
                                fits_regex=r".*\.fits$", delete_tar=True):
    """
    Download a single ALMA product tarball and extract only the FITS
    members from it, then delete the tarball.

    NOTE: astroquery's own Alma.download_and_extract_files() unconditionally
    scrapes an old "cycle 0 tarfile contents" HTML page to classify the
    tarball before doing anything else -- and as of this writing that page's
    table markup has changed, so the scrape returns None and the method
    crashes with AttributeError even for perfectly normal cycle 3+ data
    (this proposal is 2015.1.00686.S, nowhere near cycle 0). So we sidestep
    that method entirely: plain urllib download + tarfile extraction of
    just the FITS members, same approach as the HL Tau script's SV
    .tgz downloader.
    """
    tarball_name = os.path.basename(tarball_url)
    tar_local = os.path.join(dest_dir, tarball_name)
    extract_dir = os.path.join(
        dest_dir, os.path.splitext(tarball_name)[0] + "_extracted"
    )
    os.makedirs(extract_dir, exist_ok=True)

    # Skip re-downloading/re-extracting if we already have FITS here.
    existing = [
        os.path.join(root, f)
        for root, _, files in os.walk(extract_dir)
        for f in files if f.lower().endswith(".fits")
    ]
    if existing:
        print(f"Already have {len(existing)} extracted FITS file(s), skipping download.")
        return existing

    if not os.path.exists(tar_local):
        print(f"\nDownloading {tarball_url} ...")
        urllib.request.urlretrieve(tarball_url, tar_local)
        print("Download complete.")
    else:
        print(f"Found cached tarball at {tar_local}.")

    print("Extracting FITS members ...")
    pattern = re.compile(fits_regex, re.IGNORECASE)
    extracted_paths = []
    with tarfile.open(tar_local, "r") as tf:
        members = [m for m in tf.getmembers() if pattern.match(m.name)]
        if not members:
            print("No FITS members found inside this tarball.")
        for m in members:
            tf.extract(m, path=extract_dir)
            extracted_paths.append(os.path.join(extract_dir, m.name))

    if delete_tar and os.path.exists(tar_local):
        os.remove(tar_local)
        print(f"Deleted {tarball_name} after extraction to save space.")

    return extracted_paths


def download_file(alma, url, dest_dir=CACHE_DIR):
    """Download a single ALMA data file."""
    print(f"\nDownloading: {url}")
    filename = alma.download_files([url], cache=True, savedir=dest_dir)
    return filename[0] if isinstance(filename, list) else filename


def display_fits_image(filepath, cutout_arcsec=6.0, vmax_percentile=99.5, asinh_a=0.01):
    """
    Display a 2D (or 2D-slice of a) FITS image with WCS axes.

    cutout_arcsec: size (in arcsec) of a square region to crop around the
        image center before display.
    vmax_percentile: upper clip percentile for the color scale. TW Hya's
        compact central source is much brighter than the surrounding disk
        rings, so ZScale (built for background-dominated images) sets vmax
        too low and the core just saturates. A percentile interval keeps
        vmax below the true peak so ring structure doesn't wash out.
    asinh_a: softening parameter for the asinh stretch. Smaller values
        compress the bright core harder, pulling up faint disk/ring
        structure relative to it. Try 0.005-0.05.

    Note: TW Hya's disk is only ~1-2" in radius at ~60 pc, and its rings
    (as first resolved at high resolution by Andrews et al. 2016) are on
    sub-arcsec scales, so a tight cutout (a few arcsec) is needed to
    actually resolve the ring structure rather than showing empty sky.
    """
    with fits.open(filepath) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        data = np.squeeze(hdu.data)
        while data.ndim > 2:
            data = data[0]
        wcs = WCS(hdu.header, naxis=2)

    if cutout_arcsec is not None:
        from astropy.nddata import Cutout2D

        ny, nx = data.shape
        center_pix = (nx // 2, ny // 2)  # TW Hya is centered in these deliveries

        pix_scale_deg = np.abs(wcs.proj_plane_pixel_scales()[0].value)
        pix_scale_arcsec = pix_scale_deg * 3600.0
        size_pix = int(cutout_arcsec / pix_scale_arcsec)
        size_pix = max(size_pix, 10)

        cutout = Cutout2D(data, position=center_pix, size=size_pix, wcs=wcs)
        data = cutout.data
        wcs = cutout.wcs

    from astropy.visualization import PercentileInterval

    interval = PercentileInterval(vmax_percentile)
    stretch = AsinhStretch(a=asinh_a)
    norm = ImageNormalize(data, interval=interval, stretch=stretch)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection=wcs)
    im = ax.imshow(data, origin="lower", cmap="inferno", norm=norm)
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_title(f"TW Hydrae — {os.path.basename(filepath)}")
    fig.colorbar(im, ax=ax, label="Flux density")
    plt.tight_layout()
    plt.show()


def extract_radial_profile(filepath, incl_deg=7.0, pa_deg=155.0,
                            distance_pc=60.1, max_radius_arcsec=3.0,
                            n_bins=60, center=None):
    """
    Azimuthally averaged (deprojected) radial flux profile of a disk image.

    TW Hya is viewed nearly face-on, so this is much simpler than the HL
    Tau case: inclination is small (i ~ 7 deg, Andrews et al. 2016; PA is
    poorly constrained at such low inclination -- ~155 deg is a commonly
    quoted literature value, but deprojection barely matters here since
    cos(7 deg) ~ 0.99). Defaults are approximate literature values, not a
    fit to this specific image -- override incl_deg/pa_deg if you have
    better constraints (e.g. from a Keplerian rotation fit to the CO cube
    also extracted from this tarball).

    incl_deg, pa_deg: disk inclination and position angle (deg), used to
        deproject the image into a face-on frame before radial binning.
        PA is measured east of north for the disk's major axis.
    distance_pc: system distance, used to convert arcsec to au.
    max_radius_arcsec: outer radius of the profile.
    n_bins: number of radial annuli.
    center: (x, y) pixel coordinates of the star. If None, uses the
        brightest pixel in the image (fine for a centrally-peaked
        continuum image like this one).

    Returns a dict with radius_arcsec, radius_au, mean_flux, std_flux,
    plus the pixel scale and center used, so it can be re-plotted or fed
    into peak-finding without re-reading the FITS file.
    """
    with fits.open(filepath) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        data = np.squeeze(hdu.data)
        while data.ndim > 2:
            data = data[0]
        wcs = WCS(hdu.header, naxis=2)
        header = hdu.header

    _dbg("extract_radial_profile: loaded", filepath)
    _dbg("extract_radial_profile: data shape", data.shape, "dtype", data.dtype,
         "nan count", int(np.isnan(data).sum()), "finite count", int(np.isfinite(data).sum()))

    # Synthesized beam size (BMAJ/BMIN are in degrees per FITS convention).
    # This matters for ring/gap detection: any "feature" at a radius
    # comparable to or smaller than the beam FWHM is very likely the PSF of
    # the unresolved central source, not real disk substructure, and
    # should be excluded from peak-finding rather than trusted.
    bmaj_deg = header.get("BMAJ")
    bmin_deg = header.get("BMIN")
    bpa_deg = header.get("BPA")
    if bmaj_deg is not None and bmin_deg is not None:
        bmaj_arcsec = bmaj_deg * 3600.0
        bmin_arcsec = bmin_deg * 3600.0
        bmaj_au = bmaj_arcsec * distance_pc
        bmin_au = bmin_arcsec * distance_pc
        print(f"Synthesized beam: {bmaj_arcsec:.4f}\" x {bmin_arcsec:.4f}\" "
              f"(PA={bpa_deg}) = {bmaj_au:.2f} au x {bmin_au:.2f} au at {distance_pc} pc")
        _dbg("extract_radial_profile: BMAJ/BMIN/BPA (deg) =", bmaj_deg, bmin_deg, bpa_deg)
    else:
        bmaj_arcsec = bmin_arcsec = bmaj_au = bmin_au = None
        print("WARNING: no BMAJ/BMIN found in this FITS header -- beam size "
              "unknown, can't distinguish beam-scale artifacts from real "
              "substructure near the core by beam size alone.")

    pix_scale_deg = np.abs(wcs.proj_plane_pixel_scales()[0].value)
    pix_scale_arcsec = pix_scale_deg * 3600.0
    _dbg("extract_radial_profile: pix_scale_arcsec =", pix_scale_arcsec)

    ny, nx = data.shape
    if center is None:
        # Brightest pixel -- reasonable for a centrally-peaked disk image
        # like this one, where the star dominates the continuum peak.
        cy, cx = np.unravel_index(np.nanargmax(data), data.shape)
    else:
        cx, cy = center
    _dbg("extract_radial_profile: center pixel (cx, cy) =", (cx, cy),
         "image shape (ny, nx) =", (ny, nx))

    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = (xx - cx) * pix_scale_arcsec
    dy = (yy - cy) * pix_scale_arcsec

    # Deproject: rotate so the disk's major axis aligns with x, then
    # stretch the (foreshortened) minor axis back out by 1/cos(incl).
    pa_rad = np.deg2rad(pa_deg)
    incl_rad = np.deg2rad(incl_deg)
    x_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
    y_rot = (-dx * np.sin(pa_rad) + dy * np.cos(pa_rad)) / np.cos(incl_rad)
    radius_arcsec_map = np.sqrt(x_rot**2 + y_rot**2)

    bin_edges = np.linspace(0, max_radius_arcsec, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean_flux = np.full(n_bins, np.nan)
    std_flux = np.full(n_bins, np.nan)

    empty_bins = 0
    for i in range(n_bins):
        mask = (radius_arcsec_map >= bin_edges[i]) & (radius_arcsec_map < bin_edges[i + 1])
        if mask.any():
            mean_flux[i] = np.nanmean(data[mask])
            std_flux[i] = np.nanstd(data[mask])
        else:
            empty_bins += 1

    _dbg("extract_radial_profile: bins =", n_bins, "empty bins =", empty_bins,
         "valid mean_flux entries =", int(np.sum(~np.isnan(mean_flux))))
    if np.all(np.isnan(mean_flux)):
        print("WARNING: every radial bin is NaN -- check that max_radius_arcsec "
              "and the pixel scale are compatible (radius map may not overlap "
              "the requested bin range at all).")

    result = {
        "radius_arcsec": bin_centers,
        "radius_au": bin_centers * distance_pc,
        "mean_flux": mean_flux,
        "std_flux": std_flux,
        "pix_scale_arcsec": pix_scale_arcsec,
        "center_pix": (cx, cy),
        "incl_deg": incl_deg,
        "pa_deg": pa_deg,
        "bmaj_arcsec": bmaj_arcsec,
        "bmin_arcsec": bmin_arcsec,
        "bmaj_au": bmaj_au,
        "bmin_au": bmin_au,
        "distance_pc": distance_pc,
    }
    _dbg("extract_radial_profile: returning profile dict with keys", list(result.keys()))
    return result


def find_rings_and_gaps(profile, ring_prominence=None, gap_prominence=None):
    """
    Locate local maxima (rings) and minima (gaps) in a radial flux
    profile using scipy.signal.find_peaks.

    prominence=None lets scipy auto-derive a reasonable threshold from the
    profile's own scatter (a few times the median std_flux); pass an
    explicit value to override if the auto threshold is too strict/loose.
    """
    flux = profile["mean_flux"]
    radius = profile["radius_au"]
    valid = ~np.isnan(flux)
    flux_v = flux[valid]
    radius_v = radius[valid]

    _dbg("find_rings_and_gaps: valid profile points =", flux_v.size, "/", flux.size)

    if ring_prominence is None:
        ring_prominence = 3 * np.nanmedian(profile["std_flux"])
    if gap_prominence is None:
        gap_prominence = ring_prominence
    _dbg("find_rings_and_gaps: ring_prominence =", ring_prominence,
         "gap_prominence =", gap_prominence)

    ring_idx, ring_props = find_peaks(flux_v, prominence=ring_prominence)
    gap_idx, gap_props = find_peaks(-flux_v, prominence=gap_prominence)
    _dbg("find_rings_and_gaps: raw ring_idx =", ring_idx, "raw gap_idx =", gap_idx)

    rings = [
        {"radius_au": radius_v[i], "flux": flux_v[i], "prominence": ring_props["prominences"][j]}
        for j, i in enumerate(ring_idx)
    ]
    gaps = [
        {"radius_au": radius_v[i], "flux": flux_v[i], "prominence": gap_props["prominences"][j]}
        for j, i in enumerate(gap_idx)
    ]
    return rings, gaps


def find_rings_and_gaps_residual(profile, window_frac=0.22, polyorder=3,
                                  ring_prominence=None, gap_prominence=None,
                                  prominence_sigma=1.2, min_radius_beams=2.5,
                                  min_abs_residual_frac=0.3):
    """
    Locate ring/gap substructure using a smooth-baseline-subtraction
    approach (as in e.g. Huang et al. 2018's DSHARP substructure analysis),
    rather than find_peaks on the raw profile.

    Why this is needed: find_peaks only flags genuine local maxima/minima.
    When a disk's radial profile is dominated by a steep, monotonically
    declining envelope (a bright unresolved central source falling off with
    radius, as here), real ring/gap substructure often shows up only as
    "shoulders" -- slope changes riding on top of the decline -- which
    never actually turn over into a local max/min and so are invisible to
    plain find_peaks (this is exactly what happened on the first pass:
    0 rings, 0 gaps despite visible shoulders in the profile plot).

    This function fits a smooth Savitzky-Golay baseline to the profile
    (capturing just the broad monotonic falloff, not the substructure),
    subtracts it to get a residual, and runs find_peaks on the residual
    instead -- turning shoulders into detectable bumps/dips.

    window_frac: Savitzky-Golay window as a fraction of the number of
        valid profile bins (must end up odd; auto-adjusted). Larger =
        smoother baseline = more sensitive to broad shoulders; smaller =
        baseline tracks the data more closely = less sensitive.
    polyorder: polynomial order for the Savitzky-Golay fit within each
        window (3 is a reasonable default for a smoothly declining
        profile with a few features).
    prominence_sigma: multiplier on the residual's standard deviation used
        to auto-derive ring_prominence/gap_prominence when those are None.
        Lower values (e.g. 0.6-0.8) recover weaker, few-percent features
        (like the shallow gaps at ~6 au and ~44 au reported for TW Hya by
        Tsukagoshi et al. 2016 and Andrews et al. 2016) at the cost of
        being more susceptible to noise-driven false positives -- always
        sanity-check low-sigma detections against the plot and against
        literature values rather than trusting them blindly.
    min_radius_beams: exclude detections within this many synthesized-beam
        FWHMs (using BMAJ, the larger axis) of the center before reporting
        rings/gaps. CLEAN deconvolution of a very bright, compact,
        high-dynamic-range core commonly leaves small negative-bowl /
        restoration artifacts within the first few beam-widths of the
        peak, which can masquerade as a "gap" even when they're well
        outside a single beam FWHM (as happened here: a beam of ~1.9 au
        still left a spurious-looking gap at ~4 au, about 2 beams out).
        The baseline fit itself still uses the full profile (excluding
        points would distort the fit near the boundary); only the
        peak-finding *results* are filtered by radius. Set to 0 to disable.
    min_abs_residual_frac: additionally require |residual| at the detected
        point to be at least this fraction of the residual's overall
        standard deviation, independent of find_peaks' prominence measure.
        Prominence is a *relative* height (vs. neighboring valleys/peaks),
        so a detection can have high prominence while still sitting right
        on the noise floor in absolute terms -- e.g. a residual of ~1e-7
        next to a std of ~2e-5 can still register as "prominent" if its
        neighbors are even flatter. This catches that case. Set to 0 to
        disable.
    """
    flux = profile["mean_flux"]
    radius = profile["radius_au"]
    valid = ~np.isnan(flux)
    flux_v = flux[valid]
    radius_v = radius[valid]

    n = flux_v.size
    window = max(int(round(window_frac * n)), polyorder + 2)
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    _dbg("find_rings_and_gaps_residual: n_valid =", n, "savgol window =", window,
         "polyorder =", polyorder)

    baseline = savgol_filter(flux_v, window_length=window, polyorder=polyorder)
    residual = flux_v - baseline
    _dbg("find_rings_and_gaps_residual: residual std =", np.std(residual),
         "residual min/max =", residual.min(), residual.max())

    if ring_prominence is None:
        ring_prominence = prominence_sigma * np.std(residual)
    if gap_prominence is None:
        gap_prominence = prominence_sigma * np.std(residual)
    _dbg("find_rings_and_gaps_residual: ring_prominence =", ring_prominence,
         "gap_prominence =", gap_prominence)

    ring_idx, ring_props = find_peaks(residual, prominence=ring_prominence)
    gap_idx, gap_props = find_peaks(-residual, prominence=gap_prominence)
    _dbg("find_rings_and_gaps_residual: raw ring_idx =", ring_idx,
         "raw gap_idx =", gap_idx)

    bmaj_au = profile.get("bmaj_au")
    min_radius_au = min_radius_beams * bmaj_au if (bmaj_au and min_radius_beams > 0) else 0.0
    _dbg("find_rings_and_gaps_residual: bmaj_au =", bmaj_au,
         "min_radius_beams =", min_radius_beams, "-> min_radius_au =", min_radius_au)

    residual_std = np.std(residual)
    min_abs_residual = min_abs_residual_frac * residual_std
    _dbg("find_rings_and_gaps_residual: min_abs_residual_frac =", min_abs_residual_frac,
         "-> min_abs_residual =", min_abs_residual)

    def keep(i):
        return radius_v[i] >= min_radius_au and abs(residual[i]) >= min_abs_residual

    ring_idx_kept = [i for i in ring_idx if keep(i)]
    gap_idx_kept = [i for i in gap_idx if keep(i)]

    dropped_radius = [i for i in list(ring_idx) + list(gap_idx) if radius_v[i] < min_radius_au]
    dropped_noise = [
        i for i in list(ring_idx) + list(gap_idx)
        if radius_v[i] >= min_radius_au and abs(residual[i]) < min_abs_residual
    ]
    if dropped_radius:
        print(f"Excluded {len(dropped_radius)} detection(s) within {min_radius_au:.2f} au "
              f"({min_radius_beams}x beam FWHM) of center as likely beam/CLEAN artifacts:")
        for i in dropped_radius:
            print(f"    r = {radius_v[i]:.1f} au, residual = {residual[i]:.4e} Jy/beam "
                  f"({residual[i] / residual_std:+.2f} sigma)")
    if dropped_noise:
        print(f"Excluded {len(dropped_noise)} detection(s) with |residual| below "
              f"{min_abs_residual:.2e} Jy/beam ({min_abs_residual_frac}x residual std) "
              f"as likely noise/boundary artifacts, despite passing the prominence test:")
        for i in dropped_noise:
            print(f"    r = {radius_v[i]:.1f} au, residual = {residual[i]:.4e} Jy/beam "
                  f"({residual[i] / residual_std:+.2f} sigma)")

    rings = [
        {"radius_au": radius_v[i], "flux": flux_v[i], "residual": residual[i],
         "prominence": ring_props["prominences"][j]}
        for j, i in enumerate(ring_idx) if i in ring_idx_kept
    ]
    gaps = [
        {"radius_au": radius_v[i], "flux": flux_v[i], "residual": residual[i],
         "prominence": gap_props["prominences"][j]}
        for j, i in enumerate(gap_idx) if i in gap_idx_kept
    ]
    return rings, gaps, {"radius_au": radius_v, "flux": flux_v,
                          "baseline": baseline, "residual": residual,
                          "min_radius_au": min_radius_au,
                          "min_abs_residual": min_abs_residual,
                          "distance_pc": profile.get("distance_pc")}


def report_residual_near(residual_data, target_radius_au, window_au=3.0):
    """
    Print the raw residual value(s) near a specific radius, independent of
    whether find_peaks flagged anything there at all. Useful for checking
    a feature that a prominence/noise-floor filter may have dropped (or
    that never registered as a local extremum in the first place) --
    e.g. "is there a real few-sigma dip near the literature's 44 au gap
    even though nothing survived filtering there?" This never applies any
    threshold; it just reports what the residual actually does.
    """
    radius_au = residual_data["radius_au"]
    residual = residual_data["residual"]
    residual_std = np.std(residual)

    mask = np.abs(radius_au - target_radius_au) <= window_au
    if not mask.any():
        print(f"No profile bins found within {window_au} au of {target_radius_au} au.")
        return

    print(f"\nRaw residual near r = {target_radius_au} au "
          f"(+/-{window_au} au, residual std = {residual_std:.4e} Jy/beam):")
    for r, res in zip(radius_au[mask], residual[mask]):
        marker = " <-- closest" if abs(r - target_radius_au) == np.abs(radius_au[mask] - target_radius_au).min() else ""
        print(f"    r = {r:.2f} au, residual = {res:.4e} Jy/beam "
              f"({res / residual_std:+.2f} sigma){marker}")

    closest_i = np.argmin(np.abs(radius_au - target_radius_au))
    print(f"  Closest bin: r = {radius_au[closest_i]:.2f} au, "
          f"residual = {residual[closest_i]:.4e} Jy/beam "
          f"({residual[closest_i] / residual_std:+.2f} sigma)")


def plot_radial_profile(profile, rings=None, gaps=None, title="TW Hydrae radial profile"):
    """Plot azimuthally averaged flux vs. radius, with rings/gaps marked."""
    _dbg("plot_radial_profile: called with", len(rings or []), "rings and",
         len(gaps or []), "gaps")
    radius_au = profile["radius_au"]
    flux = profile["mean_flux"]
    std = profile["std_flux"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(radius_au, flux, color="tab:orange", lw=1.5, label="Azimuthal mean")
    ax.fill_between(radius_au, flux - std, flux + std, color="tab:orange", alpha=0.2,
                     label="±1σ scatter")

    if rings:
        for r in rings:
            ax.axvline(r["radius_au"], color="gold", ls="--", lw=1)
        ax.scatter([r["radius_au"] for r in rings], [r["flux"] for r in rings],
                    color="gold", marker="^", zorder=5, label="Ring")
    if gaps:
        for g in gaps:
            ax.axvline(g["radius_au"], color="navy", ls="--", lw=1)
        ax.scatter([g["radius_au"] for g in gaps], [g["flux"] for g in gaps],
                    color="navy", marker="v", zorder=5, label="Gap")

    ax.set_xlabel("Deprojected radius [au]")
    ax.set_ylabel("Flux density [Jy/beam]")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    _dbg("plot_radial_profile: about to call plt.show()")
    plt.show()
    _dbg("plot_radial_profile: plt.show() returned")
    return fig


def plot_residual_profile(residual_data, rings=None, gaps=None,
                           title="TW Hydrae substructure (baseline-subtracted)",
                           show_literature=True):
    """
    Two-panel plot: top shows the raw radial profile with its fitted
    Savitzky-Golay baseline overlaid; bottom shows the residual
    (raw - baseline) with detected rings (gold ^) and gaps (navy v)
    marked. This is the view that actually reveals substructure hidden
    inside a steep monotonic decline -- see find_rings_and_gaps_residual
    docstring for why the raw-profile view alone can miss it.

    show_literature: if True and residual_data carries a distance_pc,
        overlay literature gap/hole radii from Tsukagoshi et al. 2016 /
        Andrews et al. 2016 (see literature_features_au docstring),
        rescaled to this pipeline's assumed distance, as light dotted
        vertical lines -- so every run is visually self-checking against
        published values rather than requiring a manual lookup each time.
        This is specific to TW Hya; set False for other targets.
    """
    _dbg("plot_residual_profile: called with", len(rings or []), "rings and",
         len(gaps or []), "gaps")
    radius_au = residual_data["radius_au"]
    flux = residual_data["flux"]
    baseline = residual_data["baseline"]
    residual = residual_data["residual"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [1.2, 1]})

    min_radius_au = residual_data.get("min_radius_au", 0.0)
    distance_pc = residual_data.get("distance_pc")

    ax1.plot(radius_au, flux, color="tab:orange", lw=1.5, label="Azimuthal mean")
    ax1.plot(radius_au, baseline, color="black", lw=1.2, ls="--", label="Smooth baseline")
    if min_radius_au:
        ax1.axvspan(0, min_radius_au, color="gray", alpha=0.15,
                    label=f"Excluded (<{min_radius_au:.1f} au, beam/CLEAN artifacts)")
    ax1.set_ylabel("Flux density [Jy/beam]")
    ax1.set_title(title)

    ax2.axhline(0, color="gray", lw=0.8)
    ax2.plot(radius_au, residual, color="tab:blue", lw=1.5, label="Residual")
    if min_radius_au:
        ax2.axvspan(0, min_radius_au, color="gray", alpha=0.15)

    if show_literature and distance_pc:
        lit_features = literature_features_au(distance_pc)
        _dbg("plot_residual_profile: literature features (au) =", lit_features)
        style_by_kind = {
            "hole": dict(color="gray", ls=":", lw=1.0, alpha=0.7),
            "deep": dict(color="crimson", ls=":", lw=1.4, alpha=0.8),
            "shallow": dict(color="crimson", ls=":", lw=0.9, alpha=0.5),
        }
        seen_kinds = set()
        for radius_lit, label, kind in lit_features:
            style = style_by_kind[kind]
            for ax in (ax1, ax2):
                ax.axvline(radius_lit, **style,
                           label=label if kind not in seen_kinds else None)
            seen_kinds.add(kind)

    ax1.legend(fontsize=8, loc="upper right")

    if rings:
        for r in rings:
            ax2.axvline(r["radius_au"], color="gold", ls="--", lw=1)
        ax2.scatter([r["radius_au"] for r in rings], [r["residual"] for r in rings],
                    color="gold", marker="^", zorder=5, s=60, label="Ring")
    if gaps:
        for g in gaps:
            ax2.axvline(g["radius_au"], color="navy", ls="--", lw=1)
        ax2.scatter([g["radius_au"] for g in gaps], [g["residual"] for g in gaps],
                    color="navy", marker="v", zorder=5, s=60, label="Gap")

    ax2.set_xlabel("Deprojected radius [au]")
    ax2.set_ylabel("Residual [Jy/beam]")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    _dbg("plot_residual_profile: about to call plt.show()")
    plt.show()
    _dbg("plot_residual_profile: plt.show() returned")
    return fig


if __name__ == "__main__":
    # Target the well-known Andrews et al. 2016 high-resolution Band 7
    # continuum project. Set to None to browse all ~80 TW Hya observations
    # instead and pick a different member_ous_uid by hand.
    PROPOSAL_ID = "2015.1.00686.S"

    alma, results = query_tw_hya(proposal_id=PROPOSAL_ID)

    if len(results) == 0:
        raise SystemExit(
            "No rows matched that proposal_id. Re-run with PROPOSAL_ID=None "
            "to see all available TW Hya projects and pick a member_ous_uid "
            "from the printed table."
        )

    # Take the first matching member OUS (a proposal can have multiple).
    # The 4 rows returned for 2015.1.00686.S above are per-spectral-window
    # entries that all share the same member_ous_uid, which is expected.
    member_ous_uid = str(results["member_ous_uid"][0]).strip()

    data_info = inspect_products(alma, member_ous_uid, expand_tarfiles=True)

    tarball_url = pick_product_tarball(data_info)
    if tarball_url is None:
        raise SystemExit(
            "Could not find a usable product tarball for this UID. Print "
            "data_info['access_url'] in full and pick one manually, or try "
            "a different member_ous_uid / proposal_id."
        )

    extracted_files = download_and_extract_fits(alma, tarball_url)
    if not extracted_files:
        raise SystemExit(
            "Download succeeded but no FITS files were extracted -- this "
            "tarball may not contain FITS images (e.g. it's a raw ASDM or "
            "measurement-set-only package). Try the external_ari_l tarball "
            "instead, or inspect the tar contents manually."
        )

    print(f"\nExtracted {len(extracted_files)} FITS file(s):")
    for f in extracted_files:
        print(f"  {f}")

    # Prefer a continuum / pbcor image if multiple FITS files were extracted.
    def rank(f):
        lf = f.lower()
        if "cont" in lf and "pbcor" in lf:
            return 0
        if "cont" in lf:
            return 1
        if "pbcor" in lf:
            return 2
        return 3

    local_path = sorted(extracted_files, key=rank)[0]
    print(f"\nUsing: {local_path}")

    # High-resolution Band 7 data resolve sub-arcsec rings around a disk
    # that's only ~1-2" across, so crop tight and stretch hard.
    display_fits_image(local_path, cutout_arcsec=3.0, asinh_a=0.01, vmax_percentile=99.5)

    # Azimuthally averaged radial profile to pull out ring/gap locations
    # numerically, rather than eyeballing them off the 2D image. TW Hya is
    # nearly face-on (i ~ 7 deg) so deprojection has little effect here,
    # unlike HL Tau's much more inclined disk.
    _dbg("main: entering radial profile block, local_path =", local_path)
    try:
        profile = extract_radial_profile(
            local_path, incl_deg=7.0, pa_deg=155.0, distance_pc=60.1,
            max_radius_arcsec=2.5, n_bins=120,
        )
        rings, gaps = find_rings_and_gaps(profile)

        print(f"\nRadial profile center (pixel): {profile['center_pix']}")
        print(f"Pixel scale: {profile['pix_scale_arcsec']:.4f} arcsec/pix")

        print(f"\nDetected {len(rings)} ring(s):")
        for r in rings:
            print(f"  r = {r['radius_au']:.1f} au, "
                  f"flux = {r['flux']:.4e} Jy/beam, prominence = {r['prominence']:.2e}")

        print(f"\nDetected {len(gaps)} gap(s):")
        for g in gaps:
            print(f"  r = {g['radius_au']:.1f} au, "
                  f"flux = {g['flux']:.4e} Jy/beam, prominence = {g['prominence']:.2e}")

        plot_radial_profile(profile, rings=rings, gaps=gaps,
                             title=f"TW Hydrae Band 7 continuum radial profile — {os.path.basename(local_path)}")

        # Raw find_peaks on the profile above found 0 rings/0 gaps because
        # the flux declines monotonically overall -- real substructure here
        # shows up as shoulders on that decline, not local max/min. Fit out
        # the smooth envelope and look at the residual instead.
        # Literature (Tsukagoshi et al. 2016; Andrews et al. 2016) reports
        # two additional weak (few-%) gaps at ~6 au and ~44 au beyond the
        # two strong ones we already recovered at prominence_sigma=1.2 --
        # lower the threshold to try to pull those out too. Finer n_bins
        # above also helps separate the ~6 au feature from the beam
        # exclusion zone (~4.7 au) rather than blending into it.
        rings_r, gaps_r, residual_data = find_rings_and_gaps_residual(
            profile, prominence_sigma=0.7
        )
        print(f"\n[baseline-subtracted] Detected {len(rings_r)} ring(s):")
        for r in rings_r:
            print(f"  r = {r['radius_au']:.1f} au, "
                  f"residual = {r['residual']:.4e} Jy/beam, prominence = {r['prominence']:.2e}")
        print(f"\n[baseline-subtracted] Detected {len(gaps_r)} gap(s):")
        for g in gaps_r:
            print(f"  r = {g['radius_au']:.1f} au, "
                  f"residual = {g['residual']:.4e} Jy/beam, prominence = {g['prominence']:.2e}")

        # The literature's shallow ~44 au gap (Tsukagoshi 2016 / Andrews
        # 2016, scaled to our distance_pc) landed near a spot that got
        # filtered out by min_abs_residual_frac in a previous run. Rather
        # than loosening that threshold globally (risking noise elsewhere),
        # check the raw, unfiltered residual there directly.
        lit_shallow_44au_scaled = 44.0 * (profile["distance_pc"] / TSUKAGOSHI2016_TWHYA["distance_pc"])
        report_residual_near(residual_data, lit_shallow_44au_scaled, window_au=4.0)

        plot_residual_profile(
            residual_data, rings=rings_r, gaps=gaps_r,
            title=f"TW Hydrae Band 7 continuum substructure — {os.path.basename(local_path)}"
        )
        _dbg("main: radial profile block completed without exception")
    except Exception:
        print("ERROR: radial profile block raised an exception:")
        traceback.print_exc()
        raise
