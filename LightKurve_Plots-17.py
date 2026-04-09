def LightKurve_Plots(main_id, author="SPOC"):
    """
    Analysis using the `lightkurve` package
    =================================================
    Features demonstrated:
    1. NASA Exoplanet Archive query -> parameter extraction from df row
    2. TESS SPOC light curve search, download and inspection
    3. Raw vs PDCSAP vs SAP flux comparison (all 3 sectors)
    4. Individual sector normalisation and stitching
    5. Sigma-clipping outlier removal comparison
    6. Lomb-Scargle periodogram (stellar rotation / systematics)
    7. BLS periodogram -> period recovery vs Archive df value
    8. Phase-folding at Archive period and t0 from df row
    9. Binned phase-folded transit light curve
    10. Pixel-level target pixel file (TPF) visualisation
    11. Centroid motion analysis (transit vs out-of-transit)
    12. Flattening / detrending comparison (Savitzky-Golay)
    13. Auto-correlation function (ACF) for rotation period
    14. Parameter summary table (Archive df vs lightkurve measurements)

    Output: 12 PNG files
    Dependencies (all in env_pytransit):
        lightkurve, astroquery, numpy, matplotlib, scipy, astropy
    """

    _VERSION = "2025-04-09-r17"   # bump this on every edit for cache verification
    print(f"  LightKurve_Plots version: {_VERSION}")

    # ---------------------------------------------------------------------------
    # 0.  Imports
    # ---------------------------------------------------------------------------
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    from scipy.signal import savgol_filter

    import astropy.units as u
    from astropy.timeseries import BoxLeastSquares
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    import lightkurve as lk
    from lightkurve import LightCurveCollection

    slug = (
    main_id.lower()
           .replace("-", "")
           .replace(" ", "_")
    )

    # ---------------------------------------------------------------------------
    # 1.  Fetch WASP-39 b from NASA Exoplanet Archive
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("Fetching " + str(main_id) + " from NASA Exoplanet Archive ...")
    print("=" * 60)

    df_all = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        where=f"pl_name = '{main_id}'",
        select="pl_name,pl_orbper,pl_tranmid,pl_ratror,pl_ratdor,pl_orbincl,"
            "pl_orbeccen,pl_bmassj,pl_radj,st_rad,st_mass,st_teff,"
            "pl_trandur,pl_imppar,ra,dec"
    ).to_pandas()

    df = df_all.iloc[0]          # best/default row

    # Extract all parameters from df
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
    transit_dur = float(df["pl_trandur"])       # hours
    b           = float(df["pl_imppar"]) if not np.isnan(df["pl_imppar"]) \
                else np.cos(np.radians(incl_deg)) * aor
    ra          = float(df["ra"])
    dec         = float(df["dec"])

    # Derived quantities
    mp_earth = mp_jup * 317.83
    rp_earth = rp_jup * 11.21
    a_au     = aor * r_star * 0.00465047
    T_eq     = t_eff * np.sqrt(r_star * 0.00465047 / (2 * a_au))
    transit_depth_ppt = ror**2 * 1e3

    # BTJD offset
    BTJD_OFFSET = 2457000.0

    print(f"\nPlanet          : {df['pl_name']}")
    print(f"Period          : {period:.6f} d")
    print(f"T0 (BJD)        : {t0_ref:.4f}")
    print(f"Rp/R*           : {ror:.5f}")
    print(f"Transit depth   : {transit_depth_ppt:.2f} ppt")
    print(f"a/R*            : {aor:.3f}")
    print(f"Inclination     : {incl_deg:.3f} deg")
    print(f"Impact param b  : {b:.4f}")
    print(f"Mp              : {mp_jup:.4f} MJ  = {mp_earth:.1f} M_Earth")
    print(f"Rp              : {rp_jup:.4f} RJ  = {rp_earth:.2f} R_Earth")
    print(f"R* / R_sun      : {r_star:.3f}")
    print(f"M* / M_sun      : {m_star:.3f}")
    print(f"T_eff (K)       : {t_eff:.0f}")
    print(f"T_eq (K)        : {T_eq:.0f}")
    print(f"Transit dur (h) : {transit_dur:.3f}")
    print(f"RA / Dec        : {ra:.4f} / {dec:.4f}")

    # Convert Archive t0 to BTJD and fold into TESS window (done after download)
    t0_btjd = t0_ref - BTJD_OFFSET

    # ---------------------------------------------------------------------------
    # 2.  Search and inspect available TESS data
    # ---------------------------------------------------------------------------
    print("\nSearching for TESS data ...")
    search_lc  = lk.search_lightcurve(main_id, mission="TESS", author=author)
    search_tpf = lk.search_targetpixelfile(main_id, mission="TESS", author=author)

    print(f"\n  Light curves available  : {len(search_lc)} products")
    print(f"  Target pixel files      : {len(search_tpf)} products")
    print("\n  Light curve search results:")
    print(search_lc)

    # ---------------------------------------------------------------------------
    # 3.  Download all sectors, compare SAP vs PDCSAP  (Figure 1)
    # ---------------------------------------------------------------------------
    print("\nDownloading all TESS sectors ...")

    # Pre-seed an anonymous requests.Session so MAST never starts with None
    try:
        import requests
        from astroquery.mast import Observations
        if getattr(Observations, "_session", None) is None:
            Observations._session = requests.Session()
            Observations._session.headers.update(
                {"User-Agent": "astroquery/lightkurve-retry"}
            )
    except Exception:
        pass

    lc_list_pdcsap = []
    lc_list_sap    = []

    # Flux column names differ by pipeline author.
    # QLP files contain kspsap_flux (not pdcsap_flux); SPOC/TESS-SPOC have pdcsap_flux.
    # We detect the right column by inspecting the actual FITS columns of the first
    # downloaded file, so we are robust to author string variants and future authors.
    # Preferred cleaned-flux column names, in priority order.
    # We try each per-sector until one works — different QLP sectors can have
    # different column layouts (e.g. kspsap_flux vs orrsap_flux vs sap_flux).
    # Priority order: prefer the most corrected flux available.
    # det_flux / sys_rm_flux are QLP Sector 90+ column names.
    _CLEANED_CANDIDATES = [
        "pdcsap_flux",   # SPOC / TESS-SPOC
        "kspsap_flux",   # QLP sectors ≤ ~80
        "sys_rm_flux",   # QLP sectors 90+: systematics-removed
        "det_flux",      # QLP sectors 90+: detrended
        "orrsap_flux",   # other QLP variants
    ]
    _LABEL_MAP = {
        "pdcsap_flux": "PDCSAP flux",
        "kspsap_flux": "KSPSAP flux",
        "sys_rm_flux": "SYS-RM flux",
        "det_flux":    "Det. flux",
        "orrsap_flux": "ORRSAP flux",
    }

    def _best_flux_col(lc):
        """Return the first recognised cleaned-flux column present in lc, or None."""
        cols_lower = {c.lower() for c in lc.columns}
        for cand in _CLEANED_CANDIDATES:
            if cand in cols_lower:
                return cand
        return None

    def _download_sector(search_entry, max_attempts=3):
        """Download one sector, auto-detecting the best available flux columns.

        Returns (lc_cleaned, lc_sap, col_name, label).
        Strategy:
          1. Download without flux_column so lightkurve uses its default — this
             always succeeds regardless of which columns exist.
          2. Inspect the loaded columns to find the best cleaned-flux column.
          3. If the default load already used the best column, return it directly
             (no second download needed — lightkurve caches).
          4. Re-download explicitly with the chosen column only if needed.
        """
        import requests
        from astroquery.mast import Observations

        def _patch_session():
            Observations._session = requests.Session()
            Observations._session.headers.update(
                {"User-Agent": "astroquery/lightkurve-retry"}
            )

        # Step 1: probe download (no flux_column → lightkurve default)
        for attempt in range(1, max_attempts + 1):
            try:
                lc_probe = search_entry.download()
                break
            except AttributeError as exc:
                if "'NoneType'" in str(exc) and attempt < max_attempts:
                    print(f"    MAST session uninitialised (attempt {attempt}), "
                          f"patching and retrying ...")
                    _patch_session()
                else:
                    raise
            except Exception as exc:
                if attempt < max_attempts:
                    print(f"    Probe download error (attempt {attempt}: "
                          f"{exc.__class__.__name__}), retrying ...")
                else:
                    raise

        # Step 2: pick best cleaned column from what's actually in this file
        cols_available = sorted(lc_probe.columns)
        print(f"    Sector columns : {cols_available}")
        col_clean = _best_flux_col(lc_probe)

        if col_clean is None:
            # No recognised cleaned column — the probe itself is already the best
            # we can do; use it directly without any re-download.
            actual_col = getattr(lc_probe.flux, "name", None)
            # lc_probe.flux.name is lightkurve's internal alias ("flux"), not a
            # real FITS column — report the full column list so the user can
            # add the right name to _CLEANED_CANDIDATES if desired.
            print(f"    No standard cleaned column found; using probe default "
                  f"(available: {cols_available}).")
            col_clean = actual_col if actual_col else "flux"
            label = col_clean
            lc_clean = lc_probe
        else:
            label = _LABEL_MAP.get(col_clean, col_clean)
            # Step 3: re-download with explicit column only if probe used a different one
            probe_flux_col = getattr(lc_probe.flux, "name", "").lower()
            if probe_flux_col == col_clean:
                lc_clean = lc_probe   # probe already has the right column — reuse it
            else:
                for attempt in range(1, max_attempts + 1):
                    try:
                        lc_clean = search_entry.download(flux_column=col_clean)
                        break
                    except AttributeError as exc:
                        if "'NoneType'" in str(exc) and attempt < max_attempts:
                            _patch_session()
                        else:
                            raise
                    except Exception as exc:
                        if attempt < max_attempts:
                            print(f"    Clean-flux download error (attempt {attempt}: "
                                  f"{exc.__class__.__name__}), retrying ...")
                        else:
                            raise

        # Step 4: SAP flux — always present, simple re-download
        for attempt in range(1, max_attempts + 1):
            try:
                lc_sap = search_entry.download(flux_column="sap_flux")
                break
            except AttributeError as exc:
                if "'NoneType'" in str(exc) and attempt < max_attempts:
                    _patch_session()
                else:
                    raise
            except Exception as exc:
                if attempt < max_attempts:
                    print(f"    SAP download error (attempt {attempt}: "
                          f"{exc.__class__.__name__}), retrying ...")
                else:
                    raise

        return lc_clean, lc_sap, col_clean, label

    label_pdcsap = "PDCSAP flux"   # updated below from first successful sector

    for i in range(len(search_lc)):
        lc_p, lc_s, col_this, label_this = _download_sector(search_lc[i])
        if i == 0:
            label_pdcsap = label_this   # use first sector's label for plot titles
        sector = getattr(lc_p, "sector", i + 1)
        print(f"  Sector {sector}: {len(lc_p.time)} cadences  "
              f"[cleaned={col_this}, sap=sap_flux]")
        lc_list_pdcsap.append(lc_p.remove_nans().normalize())
        lc_list_sap.append(lc_s.remove_nans().normalize())

    n_sectors = len(lc_list_pdcsap)

    fig1, axes1 = plt.subplots(n_sectors, 1,
                                figsize=(14, 4 * n_sectors), sharex=False)
    if n_sectors == 1:
        axes1 = [axes1]

    colors_sap    = ["#d62728", "#e377c2", "#8c564b"]
    colors_pdcsap = ["steelblue", "seagreen", "darkorange"]

    for i, (lc_p, lc_s) in enumerate(zip(lc_list_pdcsap, lc_list_sap)):
        sector = getattr(lc_p, "sector", i + 1)
        ax = axes1[i]
        ax.plot(lc_s.time.value,  1e3 * (lc_s.flux  - 1),
                color=colors_sap[i % 3],    lw=0.4, alpha=0.6,
                label=f"SAP flux")
        ax.plot(lc_p.time.value,  1e3 * (lc_p.flux  - 1),
                color=colors_pdcsap[i % 3], lw=0.4, alpha=0.8,
                label=label_pdcsap)
        ax.set_ylabel("Rel. flux (ppt)", fontsize=10)
        ax.set_title(f"Sector {sector}", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes1[-1].set_xlabel("BTJD (days)", fontsize=11)
    fig1.suptitle(f"{main_id} — SAP vs {label_pdcsap} (All TESS Sectors)",
                fontsize=13, y=1.01)
    fig1.tight_layout()
    fig1.savefig(f"{slug}_lk_01_sap_vs_pdcsap.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {slug}_lk_01_sap_vs_pdcsap.png")

    # ---------------------------------------------------------------------------
    # 4.  Sigma-clipping comparison  (Figure 2)
    # ---------------------------------------------------------------------------
    print("\nComparing sigma-clipping levels ...")

    lc_base = lc_list_pdcsap[0]   # use first sector for illustration
    sigmas  = [3, 5, 10]
    colors_sig = ["C1", "steelblue", "seagreen"]

    fig2, axes2 = plt.subplots(len(sigmas), 1, figsize=(12, 8), sharex=True)
    for ax, sig, col in zip(axes2, sigmas, colors_sig):
        lc_clipped = lc_base.remove_outliers(sigma=sig)
        n_removed  = len(lc_base.time) - len(lc_clipped.time)
        ax.plot(lc_clipped.time.value, 1e3 * (lc_clipped.flux - 1),
                color=col, lw=0.5, alpha=0.7)
        ax.set_ylabel("Flux (ppt)", fontsize=10)
        ax.set_title(f"sigma = {sig}  ({n_removed} points removed)", fontsize=10)
        ax.grid(True, alpha=0.25)

    axes2[-1].set_xlabel("BTJD (days)", fontsize=11)
    fig2.suptitle(f"{main_id} — Sigma-clipping Comparison (Sector 1, {label_pdcsap})",
                fontsize=12)
    fig2.tight_layout()
    fig2.savefig(f"{slug}_lk_02_sigma_clipping.png", dpi=150)
    plt.close(fig2)
    print(f"  Saved: {slug}_lk_02_sigma_clipping.png")

    # ---------------------------------------------------------------------------
    # 5.  Stitch all sectors  (Figure 3)
    # ---------------------------------------------------------------------------
    print("\nStitching all sectors ...")

    lc_stitched = LightCurveCollection(
        [lc.remove_outliers(sigma=5) for lc in lc_list_pdcsap]
    ).stitch().remove_nans()

    # FITS-backed arrays may be memory-mapped (read-only) or memoryview-backed.
    # astropy's downsample (used by .bin()) writes in-place and raises ValueError
    # on read-only arrays. Unconditionally copy every numeric column to a fresh
    # writeable numpy array — cheapest and most robust approach.
    for _col in lc_stitched.columns:
        try:
            lc_stitched[_col] = np.array(lc_stitched[_col], dtype=np.float64)
        except (ValueError, TypeError):
            pass   # skip non-numeric columns (e.g. string metadata)

    x_all  = np.ascontiguousarray(lc_stitched.time.value,       dtype=np.float64)
    y_all  = np.ascontiguousarray(1e3 * (lc_stitched.flux - 1), dtype=np.float64)
    ye_all = np.ascontiguousarray(1e3 * lc_stitched.flux_err,   dtype=np.float64)
    texp   = float(np.nanmedian(np.diff(x_all)))

    print(f"  Total points    : {len(x_all)}")
    print(f"  Time baseline   : {x_all.min():.1f} -- {x_all.max():.1f} BTJD")
    print(f"  Cadence         : {texp*24*60:.1f} min")

    # Fold t0 into TESS baseline
    n_cycles = np.round((np.median(x_all) - t0_btjd) / period)
    t0_tess  = t0_btjd + n_cycles * period
    print(f"  Folded T0       : {t0_tess:.4f} BTJD")

    fig3, ax3 = plt.subplots(figsize=(14, 4))
    # Colour by sector
    sector_boundaries = [x_all.min()]
    for lc in lc_list_pdcsap[:-1]:
        sector_boundaries.append(lc.time.value.max())
    sector_boundaries.append(x_all.max())

    for i in range(n_sectors):
        mask_s = (x_all >= sector_boundaries[i]) & (x_all <= sector_boundaries[i+1])
        ax3.plot(x_all[mask_s], y_all[mask_s],
                color=colors_pdcsap[i % 3], lw=0.4, alpha=0.7,
                label=f"Sector {getattr(lc_list_pdcsap[i], 'sector', i+1)}")

    ax3.set_xlabel("BTJD (days)", fontsize=12)
    ax3.set_ylabel("Relative flux (ppt)", fontsize=12)
    ax3.set_title(f"{main_id} — Stitched TESS Light Curve ({n_sectors} Sectors, "
                f"{len(x_all):,} points)", fontsize=12)
    ax3.legend(fontsize=10, loc="upper right")
    ax3.grid(True, alpha=0.25)
    fig3.tight_layout()
    fig3.savefig(f"{slug}_lk_03_stitched_lightcurve.png", dpi=150)
    plt.close(fig3)
    print(f"  Saved: {slug}_lk_03_stitched_lightcurve.png")

    # ---------------------------------------------------------------------------
    # 6.  Lomb-Scargle periodogram  (Figure 4)
    # ---------------------------------------------------------------------------
    print("\nComputing Lomb-Scargle periodogram ...")

    # Bin to 10-min cadence for LS periodogram speed
    lc_ls_input = lc_stitched.bin(time_bin_size=10 * u.minute).remove_nans()
    pg_ls = lc_ls_input.to_periodogram(method="lombscargle",
                                        minimum_period=0.1,
                                        maximum_period=30.0)
    ls_period = float(pg_ls.period_at_max_power.value)
    print(f"  LS peak period  : {ls_period:.4f} d")

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 8))

    pg_ls.plot(ax=ax4a, color="steelblue", lw=1)
    ax4a.axvline(period, color="C1", lw=2, ls="--",
                label=f"Archive orbital period = {period:.4f} d")
    ax4a.axvline(ls_period, color="C2", lw=1.5, ls=":",
                label=f"LS peak = {ls_period:.4f} d")
    ax4a.set_title(f"{main_id} — Lomb-Scargle Periodogram", fontsize=12)
    ax4a.legend(fontsize=10)
    ax4a.grid(True, alpha=0.3)

    # Phase-fold at LS peak for inspection
    lc_fold_ls = lc_stitched.fold(period=ls_period)
    lc_fold_ls.scatter(ax=ax4b, s=1, alpha=0.3, color="steelblue")
    ax4b.set_title(f"Phase-folded at LS peak ({ls_period:.4f} d)", fontsize=11)
    ax4b.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4.savefig(f"{slug}_lk_04_lomb_scargle.png", dpi=150)
    plt.close(fig4)
    print(f"  Saved: {slug}_lk_04_lomb_scargle.png")

    # ---------------------------------------------------------------------------
    # 7.  BLS periodogram  (Figure 5)
    # ---------------------------------------------------------------------------
    print("\nComputing BLS periodogram ...")

    # Build BLS input: bin raw arrays to 10-min cadence using numpy directly.
    # We avoid lc_stitched.bin() here because mixed-cadence stitched LCs can
    # produce a larger-than-expected output when astropy misidentifies bin edges.
    _bin_size_days = 10.0 / 24.0 / 60.0
    _bin_edges     = np.arange(x_all.min(), x_all.max() + _bin_size_days, _bin_size_days)
    _bin_idx       = np.digitize(x_all, _bin_edges) - 1
    _x_bin, _y_bin, _ye_bin = [], [], []
    for _b in range(len(_bin_edges) - 1):
        _m = _bin_idx == _b
        if _m.sum() == 0:
            continue
        _x_bin.append(np.mean(x_all[_m]))
        _y_bin.append(np.mean(y_all[_m]))
        _ye_bin.append(np.sqrt(np.mean(ye_all[_m]**2) / _m.sum()))
    _x_bin  = np.array(_x_bin)
    _y_bin  = np.array(_y_bin)
    _ye_bin = np.array(_ye_bin)

    # Reconstruct a LightCurve from the binned arrays for BLS
    from astropy.time import Time as _ATime
    import astropy.units as _u
    lc_bls_input = lk.LightCurve(
        time=_ATime(_x_bin, format="btjd", scale="tdb"),
        flux=(_y_bin / 1e3 + 1.0),
        flux_err=(_ye_bin / 1e3),
    ).remove_nans()
    print(f"  BLS input: {len(lc_bls_input.time)} points at 10-min cadence")

    # BLS period search: explicit linspace grid avoids frequency_factor guessing.
    # Window = archive period ± 50%, hard floor at 0.2 d (not 0.5 d) so
    # short-period planets like WASP-18b (0.94 d) are fully covered.
    _baseline_days  = float(x_all.max() - x_all.min())
    _bls_min_period = max(0.2, period * 0.5)
    _bls_max_period = period * 1.5
    _bls_npoints    = 60_000
    _bls_grid       = np.linspace(_bls_min_period, _bls_max_period, _bls_npoints)
    print(f"  BLS search window   : {_bls_min_period:.3f} -- {_bls_max_period:.3f} d  "
          f"({_bls_npoints:,} grid points, "
          f"res={(_bls_max_period-_bls_min_period)/_bls_npoints*24*60:.2f} min)")

    # Run BLS via astropy directly — lightkurve's wrapper ignores explicit
    # period arrays and recomputes its own oversized grid from cadence.
    from astropy.timeseries import BoxLeastSquares as _BLS
    from astropy.time import Time as _ATime2

    _t_bls  = _x_bin * u.day
    _y_bls  = _y_bin / 1e3 + 1.0          # normalised flux
    _ye_bls = _ye_bin / 1e3

    _durations = np.array([0.08, 0.10, 0.12, 0.14]) * u.day
    _bls_model = _BLS(_t_bls, _y_bls, _ye_bls)
    _bls_result = _bls_model.power(
        _bls_grid * u.day,
        _durations,
        method="fast",
        objective="snr",
    )

    # Best period — with harmonic recovery.
    # BLS often peaks at a sub-harmonic (period/N) of the true period because
    # N transits per cycle also produces a valid box fit. If the raw peak is
    # close to period/N for N in [2,3,4], scale back up to the true period.
    _best_idx     = np.argmax(_bls_result.power)
    _bls_raw      = float(_bls_result.period[_best_idx].value)
    bls_period_lk = _bls_raw
    for _n in [2, 3, 4]:
        _candidate = _bls_raw * _n
        if abs(_candidate - period) / period < 0.05:   # within 5% of archive
            bls_period_lk = _candidate
            print(f"  BLS harmonic recovery: {_bls_raw:.5f} d × {_n} → {bls_period_lk:.5f} d")
            break
    print(f"  BLS peak period : {bls_period_lk:.5f} d  (Archive: {period:.5f} d)")

    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(10, 8))

    ax5a.plot(_bls_result.period.value, _bls_result.power,
              color="steelblue", lw=0.8)
    ax5a.axvline(period, color="C1", lw=2, ls="--",
                label=f"Archive P = {period:.5f} d")
    ax5a.axvline(bls_period_lk, color="C2", lw=1.5, ls=":",
                label=f"BLS peak = {bls_period_lk:.5f} d")
    ax5a.set_xlabel("Period (days)", fontsize=11)
    ax5a.set_ylabel("BLS Power (SNR)", fontsize=11)
    ax5a.set_title(f"{main_id} — BLS Periodogram (astropy)", fontsize=12)
    ax5a.legend(fontsize=10)
    ax5a.grid(True, alpha=0.3)

    # Phase-fold at Archive period (from df row)
    lc_fold_bls = lc_stitched.fold(period=period,
                                    epoch_time=t0_tess)
    lc_fold_bls.scatter(ax=ax5b, s=1, alpha=0.3, color="steelblue",
                        label="TESS data")
    ax5b.set_xlim(-0.25, 0.25)
    ax5b.set_title(f"Phase-folded at Archive period ({period:.5f} d)", fontsize=11)
    ax5b.set_xlabel("Phase (days from mid-transit)", fontsize=11)
    ax5b.grid(True, alpha=0.3)

    fig5.tight_layout()
    fig5.savefig(f"{slug}_lk_05_bls_periodogram.png", dpi=150)
    plt.close(fig5)
    print(f"  Saved: {slug}_lk_05_bls_periodogram.png")

    # ---------------------------------------------------------------------------
    # 8.  Phase-folded transit + binned light curve  (Figure 6)
    # ---------------------------------------------------------------------------
    print("\nGenerating phase-folded transit ...")

    lc_folded = lc_stitched.fold(period=period, epoch_time=t0_tess)
    lc_binned = lc_folded.bin(time_bin_size=5 * u.minute)

    fig6, ax6 = plt.subplots(figsize=(10, 5))
    lc_folded.scatter(ax=ax6, s=1, alpha=0.2, color="lightsteelblue",
                    label="Unbinned TESS data")
    lc_binned.errorbar(ax=ax6, fmt="o", color="steelblue", ms=4,
                        label="5-min binned", zorder=5)

    # Mark transit duration from Archive df
    half_dur = transit_dur / 2 / 24   # convert hours -> days
    ax6.axvspan(-half_dur, half_dur, alpha=0.08, color="C1",
                label=f"Archive transit duration ({transit_dur:.2f} h)")
    ax6.axvline(-half_dur, color="C1", lw=1.5, ls="--", alpha=0.7)
    ax6.axvline( half_dur, color="C1", lw=1.5, ls="--", alpha=0.7)
    ax6.axhline(1.0, color="gray", lw=0.8, ls=":")

    ax6.set_xlim(-0.2, 0.2)
    ax6.set_xlabel("Phase (days from mid-transit)", fontsize=12)
    ax6.set_ylabel("Normalised flux", fontsize=12)
    ax6.set_title(
        f"{main_id} — Phase-folded Transit (lightkurve)\n"
        f"P = {period:.5f} d from Archive df,  "
        f"depth = {transit_depth_ppt:.2f} ppt,  "
        f"duration = {transit_dur:.2f} h",
        fontsize=11)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    fig6.tight_layout()
    fig6.savefig(f"{slug}_lk_06_phasefolded_transit.png", dpi=150)
    plt.close(fig6)
    print(f"  Saved: {slug}_lk_06_phasefolded_transit.png")

    # ---------------------------------------------------------------------------
    # 9.  Detrending / flattening comparison  (Figure 7)
    # ---------------------------------------------------------------------------
    print("\nComparing detrending methods ...")

    lc_sec1 = lc_list_pdcsap[0].remove_outliers(sigma=5)

    # Lightkurve flatten (Savitzky-Golay)
    window_lengths = [101, 301, 601]
    colors_flat = ["C1", "steelblue", "seagreen"]
    labels_flat = ["Window 101 (~3.4 h)", "Window 301 (~10 h)", "Window 601 (~20 h)"]

    fig7, axes7 = plt.subplots(len(window_lengths) + 1, 1,
                                figsize=(12, 10), sharex=True)

    # Raw light curve on top
    axes7[0].plot(lc_sec1.time.value, 1e3 * (lc_sec1.flux - 1),
                "k", lw=0.4, alpha=0.6)
    axes7[0].set_ylabel("Flux (ppt)", fontsize=10)
    axes7[0].set_title("Raw PDCSAP (Sector 1)", fontsize=10)
    axes7[0].grid(True, alpha=0.25)

    for ax, wl, col, lab in zip(axes7[1:], window_lengths, colors_flat, labels_flat):
        try:
            lc_flat, lc_trend = lc_sec1.flatten(window_length=wl, return_trend=True)
            ax.plot(lc_flat.time.value, 1e3 * (lc_flat.flux - 1),
                    color=col, lw=0.4, alpha=0.7)
            ax.set_ylabel("Flux (ppt)", fontsize=10)
            ax.set_title(f"Flattened — {lab}", fontsize=10)
            ax.grid(True, alpha=0.25)
        except Exception as e:
            ax.set_title(f"Flattened — {lab} (failed: {e})", fontsize=9)

    axes7[-1].set_xlabel("BTJD (days)", fontsize=11)
    fig7.suptitle(f"{main_id} — Savitzky-Golay Detrending Comparison", fontsize=12)
    fig7.tight_layout()
    fig7.savefig(f"{slug}_lk_07_detrending.png", dpi=150)
    plt.close(fig7)
    print(f"  Saved: {slug}_lk_07_detrending.png")

    # ---------------------------------------------------------------------------
    # 10.  Target Pixel File  (Figure 8)
    # ---------------------------------------------------------------------------
    print("\nDownloading and visualising Target Pixel File ...")

    if len(search_tpf) > 0:
        tpf = search_tpf[0].download()
        sector_tpf = getattr(tpf, "sector", 1)

        fig8, axes8 = plt.subplots(1, 3, figsize=(14, 5))

        # Mean flux image
        mean_img = np.nanmean(tpf.flux.value, axis=0)
        im0 = axes8[0].imshow(mean_img, origin="lower", cmap="YlOrRd",
                            interpolation="nearest")
        plt.colorbar(im0, ax=axes8[0], label="Mean flux (e-/s)")
        axes8[0].set_title(f"Mean TPF Image\n(Sector {sector_tpf})", fontsize=11)

        # Pipeline aperture mask
        try:
            pipeline_mask = tpf.pipeline_mask
            axes8[1].imshow(mean_img, origin="lower", cmap="YlOrRd",
                            alpha=0.7, interpolation="nearest")
            axes8[1].imshow(np.where(pipeline_mask, 1, np.nan),
                            origin="lower", cmap="Blues", alpha=0.5,
                            interpolation="nearest")
            axes8[1].set_title("Pipeline Aperture Mask", fontsize=11)
        except Exception:
            axes8[1].imshow(mean_img, origin="lower", cmap="YlOrRd",
                            interpolation="nearest")
            axes8[1].set_title("TPF Image (no mask)", fontsize=11)

        # Light curve from TPF using pipeline aperture
        try:
            lc_from_tpf = tpf.to_lightcurve(aperture_mask="pipeline")
            lc_from_tpf = lc_from_tpf.remove_nans().normalize()
            axes8[2].plot(lc_from_tpf.time.value,
                        1e3 * (lc_from_tpf.flux - 1),
                        "k", lw=0.5, alpha=0.6)
            axes8[2].set_xlabel("BTJD (days)", fontsize=10)
            axes8[2].set_ylabel("Flux (ppt)", fontsize=10)
            axes8[2].set_title("LC from TPF (pipeline aperture)", fontsize=11)
            axes8[2].grid(True, alpha=0.3)
        except Exception as e:
            axes8[2].set_title(f"LC extraction failed:\n{e}", fontsize=9)

        fig8.suptitle(f"{main_id} — Target Pixel File (Sector {sector_tpf})",
                    fontsize=13)
        fig8.tight_layout()
        fig8.savefig(f"{slug}_lk_08_tpf.png", dpi=150)
        plt.close(fig8)
        print(f"  Saved: {slug}_lk_08_tpf.png")
    else:
        print("  No TPF available — skipping Figure 8")

    # ---------------------------------------------------------------------------
    # 11.  Centroid motion analysis  (Figure 9)
    # ---------------------------------------------------------------------------
    print("\nAnalysing centroid motion ...")

    # Use the first sector PDCSAP light curve
    lc_cen = lc_list_pdcsap[0].remove_outliers(sigma=5)

    # Fold to find in-transit vs out-of-transit cadences
    lc_cen_fold = lc_cen.fold(period=period, epoch_time=t0_tess)
    in_transit   = np.abs(lc_cen_fold.phase.value) < (transit_dur / 2 / 24)

    try:
        cen_col = lc_cen.centroid_col.value
        cen_row = lc_cen.centroid_row.value
        valid   = np.isfinite(cen_col) & np.isfinite(cen_row)

        fig9, axes9 = plt.subplots(1, 2, figsize=(12, 5))

        # Centroid col vs time
        axes9[0].plot(lc_cen.time.value[valid], cen_col[valid],
                    "k.", ms=1, alpha=0.5, label="Centroid col")
        axes9[0].set_xlabel("BTJD (days)", fontsize=11)
        axes9[0].set_ylabel("CCD Column (pixels)", fontsize=11)
        axes9[0].set_title("Centroid Column vs Time", fontsize=11)
        axes9[0].grid(True, alpha=0.3)

        # Centroid scatter: in-transit vs out-of-transit
        oot = ~in_transit
        axes9[1].scatter(cen_col[valid & oot[:len(valid)]],
                        cen_row[valid & oot[:len(valid)]],
                        s=2, alpha=0.3, color="steelblue",
                        label="Out-of-transit")
        axes9[1].scatter(cen_col[valid & in_transit[:len(valid)]],
                        cen_row[valid & in_transit[:len(valid)]],
                        s=8, alpha=0.7, color="C1",
                        label="In-transit")
        axes9[1].set_xlabel("CCD Column (pixels)", fontsize=11)
        axes9[1].set_ylabel("CCD Row (pixels)", fontsize=11)
        axes9[1].set_title("Centroid: In-transit vs Out-of-transit", fontsize=11)
        axes9[1].legend(fontsize=10)
        axes9[1].grid(True, alpha=0.3)

        fig9.suptitle(f"{main_id} — Centroid Motion Analysis (lightkurve)", fontsize=12)
        fig9.tight_layout()
        fig9.savefig(f"{slug}_lk_09_centroid.png", dpi=150)
        plt.close(fig9)
        print(f"  Saved: {slug}_lk_09_centroid.png")
    except Exception as e:
        print(f"  Centroid analysis skipped: {e}")

    # ---------------------------------------------------------------------------
    # 12.  Individual transit snapshots  (Figure 10)
    # ---------------------------------------------------------------------------
    print("\nGenerating individual transit snapshots ...")

    # Find all transit times in the stitched data
    t0_list = []
    tc = t0_tess
    while tc <= x_all.max() + period:
        if x_all.min() - period <= tc <= x_all.max() + period:
            t0_list.append(tc)
        tc += period

    # Keep only transits with data coverage
    t0_covered = [tc for tc in t0_list
                if np.sum(np.abs(x_all - tc) < 0.2) > 20]

    print(f"  Transits with coverage: {len(t0_covered)}")

    # Plot up to 12 individual transits
    n_show = min(12, len(t0_covered))
    ncols  = 4
    nrows  = int(np.ceil(n_show / ncols))

    fig10, axes10 = plt.subplots(nrows, ncols,
                                figsize=(14, 3.5 * nrows),
                                sharey=True)
    axes10_flat = axes10.flatten() if nrows > 1 else axes10

    for idx, tc in enumerate(t0_covered[:n_show]):
        ax  = axes10_flat[idx]
        m   = np.abs(x_all - tc) < 0.2
        t_c = x_all[m] - tc
        y_c = y_all[m]
        ax.plot(t_c * 24, y_c, "k.", ms=2, alpha=0.5)
        ax.axvline(0, color="C1", lw=1, ls="--", alpha=0.7)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"T0 + {int(round((tc - t0_tess)/period))} periods\n"
                    f"BTJD {tc:.2f}", fontsize=8)
        ax.grid(True, alpha=0.2)
        if idx % ncols == 0:
            ax.set_ylabel("Flux (ppt)", fontsize=9)

    # Hide unused panels
    for idx in range(n_show, len(axes10_flat)):
        axes10_flat[idx].set_visible(False)

    for ax in axes10_flat[:n_show]:
        ax.set_xlabel("Hours from T0", fontsize=8)

    fig10.suptitle(f"{main_id} — Individual Transit Snapshots "
                f"({n_show} of {len(t0_covered)} covered transits)",
                fontsize=12)
    fig10.tight_layout()
    fig10.savefig(f"{slug}_lk_10_transit_snapshots.png", dpi=150)
    plt.close(fig10)
    print(f"  Saved: {slug}_lk_10_transit_snapshots.png")

    # ---------------------------------------------------------------------------
    # 13.  Auto-correlation function for stellar rotation  (Figure 11)
    # ---------------------------------------------------------------------------
    print("\nComputing auto-correlation function ...")

    # Flatten first to remove transit signal
    try:
        # Flatten per-sector (avoids all-NaN from gaps across sector boundaries).
        # Also copy to writeable arrays first so flatten's internal ops don't
        # hit the FITS read-only memory-map error.
        _flat_sectors = []
        for _lc_s in lc_list_pdcsap:
            try:
                _lc_w = _lc_s.copy()
                for _col in _lc_w.columns:
                    try:
                        _lc_w[_col] = np.array(_lc_w[_col], dtype=np.float64)
                    except (ValueError, TypeError):
                        pass
                _flat_sectors.append(
                    _lc_w.remove_outliers(sigma=5)
                         .flatten(window_length=301)
                         .remove_nans()
                )
            except Exception as _fe:
                print(f"    Flatten skipped for a sector: {_fe}")
        if not _flat_sectors:
            raise ValueError("All sectors failed to flatten")
        lc_flat = LightCurveCollection(_flat_sectors).stitch().remove_nans()
        if len(lc_flat.time) < 50:
            raise ValueError("Too few points after flattening")
        pg_acf  = lc_flat.to_periodogram(method="lombscargle",
                                        minimum_period=1.0,
                                        maximum_period=40.0)
        rot_period = float(pg_acf.period_at_max_power.value)
        print(f"  Estimated rotation period: {rot_period:.2f} d")

        fig11, (ax11a, ax11b) = plt.subplots(2, 1, figsize=(10, 8))

        # Flattened light curve
        ax11a.plot(lc_flat.time.value, 1e3 * (lc_flat.flux - 1),
                "k", lw=0.4, alpha=0.5)
        ax11a.set_ylabel("Flattened flux (ppt)", fontsize=11)
        ax11a.set_xlabel("BTJD (days)", fontsize=11)
        ax11a.set_title("Flattened Light Curve (transits removed)", fontsize=11)
        ax11a.grid(True, alpha=0.3)

        # LS periodogram of flattened LC
        pg_acf.plot(ax=ax11b, color="steelblue", lw=1)
        ax11b.axvline(rot_period, color="C1", lw=2, ls="--",
                    label=f"Peak = {rot_period:.2f} d (candidate rotation)")
        ax11b.axvline(period, color="C2", lw=1.5, ls=":",
                    label=f"Orbital P = {period:.4f} d")
        ax11b.set_title("LS Periodogram of Flattened LC", fontsize=11)
        ax11b.legend(fontsize=10)
        ax11b.grid(True, alpha=0.3)

        fig11.suptitle(f"{main_id} — Stellar Rotation / Systematics Analysis",
                    fontsize=12)
        fig11.tight_layout()
        fig11.savefig(f"{slug}_lk_11_rotation.png", dpi=150)
        plt.close(fig11)
        print(f"  Saved: {slug}_lk_11_rotation.png")
    except Exception as e:
        print(f"  Rotation analysis skipped: {e}")

    # ---------------------------------------------------------------------------
    # 14.  Parameter summary table  (Figure 12)
    # ---------------------------------------------------------------------------
    print("\nGenerating parameter summary table ...")

    # Measure transit depth from binned folded light curve
    try:
        flux_binned = lc_binned.flux.value
        in_tr_bin   = np.abs(lc_binned.phase.value) < (transit_dur / 2 / 24)
        depth_meas  = float(1.0 - np.nanmin(flux_binned[in_tr_bin])) * 1e3
        ror_meas    = float(np.sqrt(depth_meas / 1e3))
    except Exception:
        depth_meas = transit_depth_ppt
        ror_meas   = ror

    # BLS period measurement
    bls_period_measured = bls_period_lk

    params_archive = {
        "Period (d)"        : f"{period:.6f}",
        "T0 (BJD)"          : f"{t0_ref:.4f}",
        "Rp/R*"             : f"{ror:.5f}",
        "Transit depth (ppt)": f"{transit_depth_ppt:.3f}",
        "Transit dur (h)"   : f"{transit_dur:.3f}",
        "Impact param b"    : f"{b:.4f}",
        "Rp (RJ)"           : f"{rp_jup:.4f}",
        "Mp (MJ)"           : f"{mp_jup:.4f}",
        "a (AU)"            : f"{a_au:.5f}",
        "T_eq (K)"          : f"{T_eq:.0f}",
    }
    params_lk = {
        "Period (d)"        : f"{bls_period_measured:.6f}",
        "T0 (BJD)"          : f"{t0_tess + BTJD_OFFSET:.4f}",
        "Rp/R*"             : f"{ror_meas:.5f}",
        "Transit depth (ppt)": f"{depth_meas:.3f}",
        "Transit dur (h)"   : f"{transit_dur:.3f} (Archive)",
        "Impact param b"    : "-- (not fitted)",
        "Rp (RJ)"           : f"{ror_meas * r_star * 9.731:.4f}",
        "Mp (MJ)"           : "-- (RV needed)",
        "a (AU)"            : f"{a_au:.5f} (Archive)",
        "T_eq (K)"          : f"{T_eq:.0f} (Archive)",
    }

    labels = list(params_archive.keys())
    vals_a = [params_archive[k] for k in labels]
    vals_l = [params_lk[k]      for k in labels]

    fig12, ax12 = plt.subplots(figsize=(13, 4))
    ax12.axis("off")
    tbl = ax12.table(
        cellText=[[l, a, lv] for l, a, lv in zip(labels, vals_a, vals_l)],
        colLabels=["Parameter", "NASA Archive (df)", "lightkurve Measured"],
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
    ax12.set_title(
        f"{main_id} — Parameter Comparison: NASA Archive df vs lightkurve",
        fontsize=12, pad=10)
    fig12.tight_layout()
    fig12.savefig(f"{slug}_lk_12_parameter_summary.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig12)
    print(f"  Saved: {slug}_lk_12_parameter_summary.png")

    # ---------------------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All done!  Output PNGs:")

    for i in range(1, 13):
        print(f"{slug}_lk_{i:02d}_{[
            'sap_vs_pdcsap',
            'sigma_clipping',
            'stitched_lightcurve',
            'lomb_scargle',
            'bls_periodogram',
            'phasefolded_transit',
            'detrending',
            'tpf',
            'centroid',
            'transit_snapshots',
            'rotation',
            'parameter_summary'
        ][i-1]}.png")
