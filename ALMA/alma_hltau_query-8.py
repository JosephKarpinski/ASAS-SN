"""
Query, download, and display ALMA archive data for HL Tau.

Equivalent to the ALMA Archive Query UI at:
https://almascience.nrao.edu/aq/?sourceNameResolver=HL%20Tau

Requires: astroquery, astropy, matplotlib, numpy
    pip install astroquery astropy matplotlib numpy --break-system-packages

Notes:
- HL Tau has many ALMA projects (incl. the famous 2014 Band 6/7 ring-gap
  image). Raw + calibrated products can be tens to hundreds of GB per
  project, so this script queries first, lets you inspect what's
  available, and only downloads a filtered subset (e.g. published
  "member.uid" continuum FITS images) rather than everything.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
from astropy.wcs import WCS, FITSFixedWarning
from astroquery.alma import Alma

# These older ALMA SV FITS headers use a legacy PCi_j / 4-axis (RA/Dec/Freq/
# Stokes) WCS convention that astropy "fixes" automatically and warns about
# on every load. The fix is correct and harmless, so silence the warnings.
warnings.filterwarnings("ignore", category=FITSFixedWarning)

CACHE_DIR = "./alma_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def query_hl_tau(public=True, science=True, proposal_id=None):
    """
    Query the ALMA archive for HL Tau observations, filtered strictly to HL Tau.

    proposal_id: if given (e.g. "2011.0.00015.SV"), further restrict results
        to that specific proposal — useful for grabbing a known dataset like
        the 2014 Long Baseline Campaign showcase image rather than whatever
        UID happens to come up first.
    """
    alma = Alma()
    alma.cache_location = CACHE_DIR

    print("Querying ALMA archive for source: HL Tau ...")
    results = alma.query_object("HL Tau", public=public, science=science)
    print(f"Found {len(results)} raw rows (query_object can return nearby/related fields too).")

    # query_object is a loose match — restrict to rows that actually target HL Tau.
    name_col = "target_name" if "target_name" in results.colnames else "obs_id"
    name_mask = np.array([
        "hl_tau" in str(val).lower().replace(" ", "_")
        for val in results[name_col]
    ])
    results = results[name_mask]
    print(f"Filtered to {len(results)} rows matching HL Tau.")

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
    data_info = alma.get_data_info(member_ous_uid, expand_tarfiles=expand_tarfiles)
    print(data_info["access_url", "content_length", "content_type"])
    return data_info


def pick_small_fits(data_info, keyword="cont", max_bytes=1_000_000_000):
    """
    Filter the file listing down to FITS images (preferring continuum /
    pbcor images) rather than measurement sets or auxiliary tars.
    """
    urls = [str(u) for u in data_info["access_url"]]
    is_fits = np.array([u.lower().endswith(".fits") for u in urls])

    if not is_fits.any():
        print("No individual FITS files found in the listing — the archive "
              "may only be serving whole tarballs for this UID (try "
              "expand_tarfiles=True, or a different UID/proposal).")
        return data_info[0:0]  # empty table, same schema

    candidates = data_info[is_fits]
    urls_fits = [str(u) for u in candidates["access_url"]]

    keyword_mask = np.array([keyword.lower() in u.lower() for u in urls_fits])
    if keyword_mask.any():
        candidates = candidates[keyword_mask]

    sizes = np.array(candidates["content_length"], dtype=float)
    small = candidates[sizes <= max_bytes]
    return small if len(small) > 0 else candidates


import tarfile
import urllib.request

# The 2014 Long Baseline Campaign science verification images are NOT served
# through the normal archive query/data-portal path used elsewhere in this
# script — proposal 2011.0.00015.SV only exposes raw ASDM + calibration
# tarballs there. ALMA instead publishes the finished reference images as
# standalone .tgz packages on dedicated pages. See:
# https://almascience.nrao.edu/alma-data/science-verification
LBC_REFERENCE_IMAGES = {
    "band3": {
        "tgz_url": "https://almascience.nrao.edu/almadata/sciver/HLTauBand3/HLTau_Band3_ReferenceImages.tgz",
        "fits_name": "HLTau_B3.contms_ap_big.image.pbcor.fits",
    },
    "band6": {
        "tgz_url": "https://almascience.nrao.edu/almadata/sciver/HLTauBand6/HLTau_Band6_ReferenceImages.tgz",
        "fits_name": "HLTau_B6cont_mscale_ap.image.fits",
    },
    "band7": {
        "tgz_url": "https://almascience.nrao.edu/almadata/sciver/HLTauBand7/HLTau_Band7_ReferenceImages.tgz",
        "fits_name": "HLTau_B7cont_mscale_ap.image.pbcor.fits",
    },
}


def download_lbc_reference_image(band="band6", dest_dir=CACHE_DIR):
    """
    Download and extract the 2014 Long Baseline Campaign continuum FITS
    image for HL Tau directly from ALMA's science verification page.

    band: one of "band3", "band6", "band7"
    Returns the local path to the extracted FITS file.
    """
    if band not in LBC_REFERENCE_IMAGES:
        raise ValueError(f"band must be one of {list(LBC_REFERENCE_IMAGES)}")

    info = LBC_REFERENCE_IMAGES[band]
    tgz_url = info["tgz_url"]
    fits_name = info["fits_name"]

    tgz_local = os.path.join(dest_dir, os.path.basename(tgz_url))
    extract_dir = os.path.join(dest_dir, f"HLTau_{band}_extracted")
    os.makedirs(extract_dir, exist_ok=True)

    # Skip re-downloading if we've already got the extracted FITS file
    for root, _, files in os.walk(extract_dir):
        if fits_name in files:
            print(f"Already have {fits_name}, skipping download.")
            return os.path.join(root, fits_name)

    if not os.path.exists(tgz_local):
        print(f"Downloading {tgz_url} ...")
        urllib.request.urlretrieve(tgz_url, tgz_local)
        print("Download complete.")
    else:
        print(f"Found cached tarball at {tgz_local}.")

    print(f"Extracting {fits_name} from {os.path.basename(tgz_local)} ...")
    with tarfile.open(tgz_local, "r:gz") as tf:
        member = next((m for m in tf.getmembers() if m.name.endswith(fits_name)), None)
        if member is None:
            available = [m.name for m in tf.getmembers()][:20]
            raise FileNotFoundError(
                f"Could not find '{fits_name}' inside {tgz_local}. "
                f"First files in archive: {available}"
            )
        tf.extract(member, path=extract_dir)
        extracted_path = os.path.join(extract_dir, member.name)

    print(f"Extracted to {extracted_path}")
    return extracted_path
    """Download a single ALMA data file."""
    print(f"\nDownloading: {url}")
    filename = alma.download_files([url], cache=True, savedir=dest_dir)
    return filename[0] if isinstance(filename, list) else filename


def display_fits_image(filepath, cutout_arcsec=8.0, vmax_percentile=99.0, asinh_a=0.02):
    """
    Display a 2D (or 2D-slice of a) FITS image with WCS axes.

    cutout_arcsec: size (in arcsec) of a square region to crop around the
        image center before display (see prior note on pbcor edge noise).
    vmax_percentile: upper clip percentile for the color scale. HL Tau's
        compact central source is much brighter than the surrounding disk,
        so ZScale (built for background-dominated images) sets vmax too
        low and the core just saturates. A percentile interval keeps vmax
        below the true peak so structure in the core doesn't wash out.
    asinh_a: softening parameter for the asinh stretch. Smaller values
        compress the bright core harder, pulling up faint disk/ring
        structure relative to it. Try 0.005-0.05.
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
        center_pix = (nx // 2, ny // 2)  # HL Tau is centered in these deliveries

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
    ax.set_title(f"HL Tau — {os.path.basename(filepath)}")
    fig.colorbar(im, ax=ax, label="Flux density")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # The 2014 Long Baseline Campaign (2011.0.00015.SV) reference images are
    # published directly by ALMA rather than through the archive query/data
    # portal — see download_lbc_reference_image() docstring for why.
    BAND = "band6"  # "band3", "band6", or "band7"

    local_path = download_lbc_reference_image(band=BAND)

    # This dataset's beam is 35x22 mas (vs. arcsecond-scale for the regular
    # archive continuum images we looked at earlier), and the disk itself is
    # only ~1.5-2" across at HL Tau's distance. The wide-field defaults used
    # for the lower-resolution UID earlier would show mostly empty sky here,
    # so we crop tighter and stretch harder to actually resolve the rings.
    display_fits_image(local_path, cutout_arcsec=2.0, asinh_a=0.005, vmax_percentile=99.5)
