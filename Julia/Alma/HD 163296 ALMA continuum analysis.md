# HD 163296: ALMA continuum analysis

Oct 18, 2026 · @Joseph

## Summary

The archival ALMA Band 6 image of HD 163296 shows two well-measured rings (67.6 and 100.8 au) and two gaps (about 43 and 84 au), and the planet-mass estimates from the gaps are weak, and probably low. Everything below is statistical error only; the code is unvalidated beyond comparison with the published radii.

- **Rings and gaps:** rings at 67.6 ± 0.2 and 100.8 ± 0.2 au (S/N 30 and 24); gaps at 42.5 and 83.8 au. All four match published radii (DSHARP) within one beam (4.6 au).
- **Geometry:** the ridge fit gives i = 46.5 ± 0.3°, PA = 132.9 ± 0.4°, centre shift 5 mas (about 0.5 au). The catalogue values are confirmed.
- **Asymmetry:** one strong crescent at 55.6 au (+137% of the local mean, 12σ). A centre error cannot explain it (it would need 21 au; the fit gives 0.5 au).
- **Temperature:** T\_b is 11 K at the 68 au ring and 8 K at the 101 au ring, with absorption-only optical depth 0.45 and 0.38. Scattering would raise the true optical depth.
- **Planet masses:** the 84 au gap gives 0.07–0.23 Jupiter masses from its width and 0.18–0.57 from its depth, depending on α. The 45 au gap measurement is unreliable. Published values are roughly 0.5–2 Jupiter masses.

## Data and method

The analysis uses the DSHARP Band 6 continuum image of HD 163296 at 239 GHz (1.25 mm), made with the Julia ALMA target explorer (v7).

| Item | Value |
| --- | --- |
| Project / member OUS | 2016.1.00484.L / uid://A001/X8c5/X96 |
| Exposure | 3230 s (0.90 h) |
| Beam | 0.0454 × 0.0347 arcsec = 4.6 × 3.5 au |
| Pixel scale | 0.0060 arcsec |
| Distance | 101.0 pc |
| Geometry used | i = 46.7°, PA = 133.3° (catalogue); ridge fit gives 46.5°, 132.9° |
| Image noise | 0.0348 mJy/beam (robust rms) |
| Star | L\* = 17 L☉, M\* = 2.0 M☉ (approximate; verify) |
| Profile | azimuthal mean, 132 bins of 2.3 au out to 303 au |
| Inner exclusion | inside 11 au (2.5 beams), as beam and CLEAN artifacts |

The peak of the azimuthal mean is 2.68 mJy/beam at 3.4 au. 90% of the flux lies within 102 au (22 beams), so the disk is well resolved. Large disks can lose their largest scales without short spacings, so the faint outer emission may be filtered.

## Rings and gaps

The 68 and 101 au rings and the 43 and 84 au gaps are secure and match the literature; the 15 au ring is secure; the 32 au component is a shoulder, not a confirmed ring.

Gaussian fits (tapered power law plus 5 Gaussians, χ²\_red = 3.1, errors scaled by √χ²\_red):

| r \[au\] | Amplitude \[mJy/beam\] | S/N | Intrinsic FWHM \[au\] | Note |
| --- | --- | --- | --- | --- |
| 1.8 ± 0.6 | 2.19 ± 0.71 | 3.1 | 9.7 ± 3.0 | unresolved core, inside the excluded zone |
| 15.3 ± 1.8 | 1.71 ± 0.09 | 18.6 | 19.5 ± 9.2 | radius sits at its bound |
| 31.5 ± 1.5 | 0.76 ± 0.38 | 2.0 | 11.8 ± 1.9 | below 3σ, unconfirmed |
| 67.6 ± 0.2 | 0.354 ± 0.012 | 30.1 | 15.6 ± 0.7 | secure |
| 100.8 ± 0.2 | 0.187 ± 0.008 | 23.6 | 12.6 ± 0.7 | secure |

After baseline subtraction the profile shows rings at 33, 65 and 100 au and gaps at 42.5 and 83.8 au, all above 23σ (statistical). Two of the five have no peak or dip within one beam in the raw profile: the 33 au ring is a shoulder, and the 42.5 au gap sits 4.6 au inside the raw minimum at 47.1 au, so its position depends on the baseline.

Comparison with published features (DSHARP, rescaled to 101 pc; tolerance one beam, 4.6 au):

| Published | Found | Offset |
| --- | --- | --- |
| gap 44.8 au | gap 47.1 au (raw profile) | +2.3 au |
| ring 67.0 au | ring 65.4 au (baseline-subtracted); 67.6 au (Gaussian) | −1.6 au; +0.6 au |
| gap 86.6 au | gap 86.1 au (raw profile) | −0.5 au |
| ring 99.7 au | ring 99.9 au | +0.2 au |
| ring 14.4 au | not in the baseline-subtracted profile; Gaussian ring at 15.3 au | +0.9 au |
| gap 10.0 au | inside the excluded zone | not testable |
| gap 140.6 au, ring 158.7 au | not detected (−1.3σ and +0.5σ) | not recovered |

Four of seven testable features were recovered by the baseline-subtracted or raw profile. The outer gap and ring at 141 and 159 au are fainter than this image's sensitivity or are filtered.

## Azimuthal asymmetry

The one robust non-axisymmetric feature is a crescent at 55.6 au; the Fourier amplitudes at the rings are mostly small and several are dominated by the beam.

Geometry check from the ring ridges (68 and 100 au, 72 ridge points, χ²\_red = 20, formal errors only):

| Parameter | Catalogue | Fitted |
| --- | --- | --- |
| Centre dE \[mas\] | 0 | −2.1 ± 1.9 |
| Centre dN \[mas\] | 0 | +4.6 ± 1.6 |
| Inclination \[°\] | 46.7 | 46.5 ± 0.3 |
| Position angle \[°\] | 133.3 | 132.9 ± 0.4 |

The 100 au ring has a small residual eccentricity, e = 0.008 ± 0.002 (3.9σ); the 68 au ring has none (e = 0.005 ± 0.005). The window-mean Fourier amplitudes barely change with the fitted geometry, as expected, because they are blind to small geometry errors at first order.

Fourier amplitudes relative to the ring amplitude, with the fitted geometry in parentheses:

| r \[au\] | m=1 \[%\] | m=2 \[%\] | m=2 expected from the beam alone \[%\] |
| --- | --- | --- | --- |
| 1.8 | 7.8 (8.6) | 4.9 (5.6) | 3.8 |
| 15.3 | 0.5 (1.1) | 4.0 (3.8) | 1.1 |
| 67.6 | 5.7 (5.5) | 3.8 (3.7) | 1.6 |
| 100.8 | 5.1 (5.3) | 11.9 (12.7) | 2.4 |

The m=2 orientations differ from ring to ring (R = 0.18, random is about 0.50), so no common geometry error explains them, and the 101 au value (12%, 7.4σ, excess 13 ± 1.6% over the beam) is the one worth following up. The m=1 maxima at 2, 68 and 101 au point in different directions (R = 0.37, random about 0.58).

Candidate arcs (beam-smoothed residual above 4σ, at least one beam of area):

- **#1, 55.6 au, +137%, 12σ, 36° span, 12 beams of area:** the only strong, extended positive arc. It lies 12 au inside the 68 au ring and needs a 21 au centre error against the 0.5 au fitted, so it is not a geometry artefact. Its negative counterpart (#3, 56.3 au) is only 10° wide and a third of the area, so they do not form a pair. The sky PA of 125° may differ from published crescent position angles because of how each is defined.
- **Arcs at 12, 18.5 and 66 au (#4, #5, #7):** on ring crests, where a centre error has no first-order effect, so they cannot be tested this way and are not informative.
- **Remaining arcs (#2, 6, 8, 9, 10):** 5–7σ but small (1.5 to 5.7 beams); treat as tentative.

The noise for the arcs is the scatter of the smoothed residual itself, so real structure inflates it and the thresholds are conservative. The residual image also shows stripes along the beam, probably from the CLEAN and uv-coverage structure, which these estimators do not model.

## Brightness temperature and optical depth

The disk is cold and the continuum is at most moderately optically thick: T\_b stays well below the passive-disk dust temperature everywhere, and the absorption-only optical depth is below 0.7.

Planck brightness temperature at 239 GHz from the azimuthal mean, against T\_d = (φL\*/8πr²σ\_SB)^¼ with φ = 0.02 and L\* = 17 L☉:

| Feature | r \[au\] | T\_b \[K\] | T\_d \[K\] | T\_b/T\_d | τ | H/r |
| --- | --- | --- | --- | --- | --- | --- |
| ring | 1.8 | 41.5 ± 0.5 | 235.9 | 0.18 | 0.19 | 0.023 |
| ring | 15.3 | 31.1 ± 0.2 | 65.4 | 0.48 | 0.65 | 0.044 |
| gap | 42.5 | 7.5 ± 0.1 | 38.8 | 0.19 | 0.21 | 0.058 |
| ring | 67.6 | 11.2 ± 0.1 | 30.7 | 0.37 | 0.45 | 0.065 |
| gap | 83.8 | 5.2 ± 0.1 | 27.6 | 0.19 | 0.21 | 0.068 |
| ring | 100.8 | 8.0 ± 0.1 | 25.3 | 0.32 | 0.38 | 0.071 |

Caveats:

- T\_b is beam-averaged, so it is a lower bound for features narrower than the beam, which includes the inner core.
- τ assumes absorption only; scattering by large grains lowers T\_b, so the true τ is higher.
- T\_d is an assumed passive-disk value with φ = 0.02. It ignores accretion heating and the difference between surface and midplane temperatures. The error bars on T\_b are statistical.
- The 15 au ring is the most optically thick feature (τ = 0.65) and also the one with the weakest radius constraint (±1.8 au).

## Planet masses

The 84 au gap gives a mass that is 0.07–0.23 Jupiter masses from its width and 0.18–0.57 from its depth (1 Jupiter mass = 18.5 Neptune masses), and the 45 au gap result is unreliable; treat these as order-of-magnitude only.

Gap width uses the Kanagawa et al. 2016 relation, Mp/M\* = 2.1×10⁻³ (Δ/Rp)² (h/r / 0.05)^1.5 (α/10⁻³)^0.5, with M\* = 2.0 M☉ and h/r from the passive-disk temperature. Gap depth uses the Kanagawa et al. 2015 relation with the dust-intensity contrast. Ranges for width run from the deconvolved to the observed width.

| Gap | r\_c \[au\] | h/r | Depth | W\_obs / W\_int \[au\] | Width, α = 10⁻³ | Width, α = 10⁻⁴ | Depth, α = 10⁻³ | Depth, α = 10⁻⁴ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42.5 | 50.8 | 0.060 | 3.22 | 19.5 / 19.1 | 0.83–0.86 M\_Jup | 4.8–5.1 M\_Nep | 0.44 M\_Jup | 2.6 M\_Nep |
| 83.8 | 87.2 | 0.069 | 2.86 | 15.8 / 15.2 | 4–4.3 M\_Nep | 1.3–1.4 M\_Nep | 0.57 M\_Jup | 3.3 M\_Nep |

Caveats attached to these numbers:

- **45 au gap is unreliable.** Its measured centre is 50.8 au against about 44.8 au published, and W/r is 0.385. The outer "peak" is probably the shoulder of the 55 au crescent and the inner side is the core's slope, so the width, and hence the mass, is biased. The 0.83 M\_Jup does not stand.
- **The two methods disagree for the 84 au gap.** The width relation gives 0.07–0.23 Jupiter masses (1.3–4.3 Neptune masses) and the depth relation 0.18–0.57, depending on α. A 1 M\_Jup planet would need a gap about 33 au wide at α = 10⁻³ and h/r = 0.069, against the 15.8 au observed.
- **The widths sit below published masses.** Hydrodynamic and kinematic work quotes roughly 0.5–2 M\_Jup for the 87 au gap (DSHARP IX; I read this from a page summary, so check the table before quoting). The width-based values for the 84 au gap, 0.07–0.23 M\_Jup, are about 2–7× below that range, so these are likely low; the depth-based 0.57 M\_Jup at α = 10⁻³ falls inside it.
- **The relations are for gas gaps.** Dust gaps from large grains are wider, so width masses are upper limits on that count; the depth is diluted by the beam (lower bound on contrast) while dust gaps are deeper than gas gaps (upper bound on mass). Zhang et al. 2018 dust calibrations, which depend on grain size, are not applied.
- **α and h/r are assumptions.** α is unknown and the two values bracket the usual range; h/r comes from an assumed temperature.
- **Errors are statistical only.** Several gaps may be carved by one planet, or by other mechanisms.

## Dust trapping and gap spacing

Only the 101 au ring gives a usable trapping limit; the other rings are wider than the gas scale height and give none.

Following Dullemond et al. 2018, a dust ring in a Gaussian gas pressure bump of width w has width w\_d = w√(α/(α+St)), so α/St = 1/((w/w\_d)² − 1). The largest α/St comes from w = H; the smallest from w equal to the distance to the nearest gap.

| r \[au\] | σ\_d \[au\] | H \[au\] | σ\_d/H | α/St range | St (α = 10⁻³) | St (α = 10⁻⁴) |
| --- | --- | --- | --- | --- | --- | --- |
| 1.8 | 4.1 ± 1.3 | 0.0 | 86 | no limit | n/a | n/a |
| 15.3 | 8.3 ± 3.9 | 0.7 | 12 | no limit | n/a | n/a |
| 67.6 | 6.6 ± 0.3 | 4.4 | 1.5 | no limit | n/a | n/a |
| 100.8 | 5.3 ± 0.3 | 7.2 | 0.74 | 0.110–1.21 (max 0.95–1.59) | 8.2×10⁻⁴ to 9.1×10⁻³ | 8.2×10⁻⁵ to 9.1×10⁻⁴ |

The 101 au ring is narrower than the gas scale height (σ\_d/H = 0.74), so it needs dust trapping in a pressure maximum, with α/St between 0.11 and about 1.2. The ring width is the intensity width; for optically thick rings it overestimates the dust width, so true trapping would be stronger. The 15 au value has a large width error (±3.9 au) and the 1.8 au ring is the unresolved core.

Gap spacing: the only measured pair, 42.5 and 83.8 au, has a radius ratio of 1.97 and an orbital period ratio of 2.77, 7.6% from 3:1. That is weak evidence on its own, since a ±3% match to one of eight low-order ratios happens by chance for about 40% of values between 1.1 and 3.2. The 42.5 au centre is also uncertain (see Planet masses).

## Open issues

- **45 au gap measurement:** the centre (50.8 au) disagrees with the literature (44.8 au), probably because the crescent at 55 au pulls the outer peak. A fit of the gap as a dip on the local baseline would be more robust.
- **Zhang et al. 2018 dust calibrations** are not implemented; masses use the gas relations.
- **Stellar parameters** (L\* = 17 L☉, M\* = 2 M☉) are approximate and set T\_d, h/r and the masses.
- **Ring fit χ²\_red = 3.1** for the Gaussian model and 20 for the ridge fit: widths and amplitudes are less reliable than radii, and all errors are formal.
- **Beam artifacts:** the inner 11 au is excluded; the residual image shows beam-aligned stripes that are not modelled.
- **Unrecovered features:** the published gap at 141 au and ring at 159 au are not detected; the published 14 au ring only appears as a Gaussian component.
- **Literature numbers** (DSHARP radii, crescent position, published planet masses) were read from page summaries and should be checked against the tables before quoting.
- **Gas kinematics**, which constrain planet masses independently, are outside this analysis.
