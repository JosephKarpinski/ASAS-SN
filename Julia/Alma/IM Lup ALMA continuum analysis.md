# IM Lup: ALMA continuum analysis

Oct 18, 2026 · @Joseph

## Summary

IM Lup is a negative control for this code: the azimuthal-mean profile is a smooth, steep decline with no isolated rings or gaps, so no planet masses or trapping limits were obtained, and the code invented none. Errors are statistical only.

- **Rings and gaps:** the raw peak finder finds 0 rings and 0 gaps. After baseline subtraction there is one gap at 13.2 au and one ring at 39.5 au, and neither has a peak or dip in the raw profile, so both are shoulders. The one fitted ring, at 6.0 au with a 17 au width, is the unresolved core inside the excluded zone.
- **Geometry:** the catalogue values (i = 47.5°, PA = 144.5°) were used. The ridge-based check could not run, because there is no isolated ring with S/N ≥ 10.
- **Asymmetry:** ten candidate arcs, at 4.7–5.6σ beyond 30 au and small (1–4 beams). None is a convincing crescent. The 12% m=2 at 6 au is inside 1.3 beams of the centre and is not interpretable.
- **Temperature:** T\_b is 45 K at 6 au, with absorption-only optical depth 1.2, so the inner disk is optically thick.
- **Spirals:** the catalogue lists this disk as having spirals. The code does not model spirals, so the arc list is the only place they could show up. I have not compared it with published spiral positions.

## Data and method

The analysis uses a Band 6 continuum image of IM Lup at 239 GHz (1.25 mm) from project 2016.1.00484.L, made with the Julia ALMA target explorer (v7).

| Item | Value |
| --- | --- |
| Member OUS | uid://A001/X8c5/X72 (public release 2019-02-01) |
| Exposure | 4170 s (1.16 h) |
| Image used | IM\_Lupi\_sci spw0–7 mfs, manually imaged, primary-beam corrected |
| Beam | 0.0304 × 0.0240 arcsec = 4.8 × 3.8 au |
| Pixel scale | 0.0024 arcsec |
| Distance | 158.0 pc |
| Geometry | i = 47.5°, PA = 144.5° (catalogue) |
| Image noise | 0.0158 mJy/beam (robust rms) |
| Star | L\* = 2.57 L☉, M\* = 1.10 M☉ (approximate; verify) |
| Profile | azimuthal mean, 165 bins of 2.4 au out to 395 au |
| Inner exclusion | inside 12 au (2.5 beams), as beam and CLEAN artifacts |

The archive lists 57 observations; the finest (0.0230 arcsec) was used, and the image is the manually imaged product rather than the standard calibrated continuum file used for the other targets. The peak of the azimuthal mean is 2.45 mJy/beam at 1.2 au; half the flux lies within 32 au and 90% within 102 au (about 21 beams), with a total of about 82 mJy.

## Rings and gaps

The profile has no isolated rings or gaps: the raw peak finder finds none, and the two baseline-subtracted features are shoulders.

Gaussian fit (tapered power law plus one Gaussian, χ²\_red = 3.8):

| r \[au\] | Amplitude \[mJy/beam\] | S/N | Intrinsic FWHM \[au\] | Note |
| --- | --- | --- | --- | --- |
| 6.0 ± 2.6 | 0.396 ± 0.089 | 4.5 | 16.9 ± 3.3 | radius at its bound; inside the excluded zone; 30% of the local flux |

The fit also tried a component at 40 au and dropped it (S/N 0.0). A 17 au wide "ring" at 6 au, a little over one beam from the centre, is the unresolved core and not a ring.

After baseline subtraction (re-tuned detection):

| Feature | r \[au\] | Residual | Significance | In the raw profile? |
| --- | --- | --- | --- | --- |
| gap | 13.2 | −22.6% | 31σ | no (shoulder) |
| ring | 39.5 | n/a (baseline near zero) | 30σ | no (shoulder) |

The significances are statistical only and measure a deviation from a smooth baseline, not an isolated ring or gap. The 13 au "gap" is a shoulder near the edge of the excluded zone, and the 40 au "ring" sits on the outer decline. No published radii are stored for this target, so there is no literature comparison. I have not checked the literature for faint rings in this disk.

## Azimuthal structure

The only Fourier signal is at the core, where it cannot be interpreted, and the ten candidate arcs are small and tentative.

The single ring at 6.0 au has m=1 = 1.6 ± 1.3% (1.2σ) and m=2 = 11.8 ± 1.3% (9σ, excess 13.5% over the beam expectation of 1.8%). At 6 au that is 1.3 beams from the centre, inside the excluded zone, where beam and CLEAN structure dominates; I would not interpret it. There is only one ring, so the coherence test cannot run.

Candidate arcs (beam-smoothed residual above 4σ, at least one beam of area):

| # | Sign | r \[au\] | Span \[°\] | S/N | Contrast | Area \[beams\] | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | + | 5.4 | 357 | 13.8 | +9% | 3.2 | core, ring crest, not informative |
| 2 | − | 6.7 | 98 | 9.4 | −7% | 1.6 | core, not informative |
| 3 | − | 7.2 | 91 | 6.6 | −5% | 1.1 | core, not informative |
| 4 | + | 35.2 | 21 | 5.6 | +48% | 3.3 |  |
| 5 | + | 37.2 | 15 | 5.6 | +55% | 2.3 |  |
| 6 | + | 58.8 | 15 | 5.6 | +126% | 3.7 | contrast inflated by a near-zero baseline |
| 7 | − | 30.8 | 30 | 5.2 | −33% | 1.6 |  |
| 8 | − | 39.3 | 21 | 5.1 | −55% | 3.1 |  |
| 9 | − | 61.1 | 8 | 4.9 | −111% | 1.4 | contrast inflated |
| 10 | − | 61.1 | 8 | 4.7 | −106% | 1.3 | contrast inflated |

None is a convincing crescent: all are 4.7–5.6σ, under 4 beams of area, and none forms a comparable positive and negative pair. The outer arcs (#6, #9, #10) show contrasts of about ±100% only because the local mean is near zero, and the "centre error needed" values of 60–170 au for them are meaningless. Positive and negative arcs at 35–40 au (#4, #5, #8) could be parts of a spiral pattern, but the code has no spiral model, and I have not compared them with published spiral positions. At 4.7–5.6σ over a large area, some chance detections are expected.

## Brightness temperature and optical depth

The inner disk is optically thick: T\_b peaks at 77.5 K and is 45 K at 6 au, where the absorption-only optical depth is 1.2.

Planck brightness temperature at 239 GHz from the azimuthal mean, against T\_d = (φL\*/8πr²σ\_SB)^¼ with φ = 0.02 and L\* = 2.57 L☉:

| Feature | r \[au\] | T\_b \[K\] | T\_d \[K\] | T\_b/T\_d | τ | H/r |
| --- | --- | --- | --- | --- | --- | --- |
| core "ring" | 6.0 | 45.0 ± 0.3 | 64.4 | 0.70 | 1.20 | 0.038 |
| "gap" (shoulder) | 13.2 | 24.5 ± 0.2 | 43.4 | 0.56 | 0.83 | 0.046 |

Caveats:

- T\_b is beam-averaged, so it is a lower bound for features narrower than the beam, which includes the core.
- τ assumes absorption only; scattering lowers T\_b, so the true τ is higher.
- T\_d is an assumed passive-disk value with φ = 0.02 and ignores accretion heating. The T\_b errors are statistical.
- Both rows are inside or at the edge of the 12 au excluded zone, and neither is a measured ring or gap.

## Planet masses and dust trapping

No planet mass, trapping limit or gap spacing could be derived, because the profile has no isolated gap or ring to measure.

- **Planet masses:** 0 of 1 gap was usable. The 13 au "gap" has no outer ring peak: the profile does not rise past its minimum at 15.6 au, so neither the width relation nor the depth relation can be applied. This is correct behaviour for a shoulder.
- **Dust trapping:** the only ring, at 6 au with σ\_d = 7.2 ± 1.4 au, is far wider than the gas scale height (H = 0.2 au), so it gives no limit on α/St. It is also the core, inside the excluded zone.
- **Gap spacing:** fewer than two gaps, so nothing to compare.

## Open issues

- **Literature comparison** is missing: no published ring or gap radii are stored for this target. Published ring and spiral positions need to be checked before drawing conclusions about substructure.
- **Spirals:** the code has no spiral model and the azimuthal mean washes them out. If the spiral structure is the interest, a dedicated analysis (spiral tracing in the deprojected residual image) is needed.
- **Geometry** is the catalogue value and the ridge-based check could not run; there is no isolated ring with S/N ≥ 10.
- **Faint outer structure:** the analysis works at a smoothing scale of about one beam. Rings with contrasts of a few per cent of a steeply falling profile may be present and are below what this baseline can recover.
- **Image product:** this is a manually imaged continuum file, not the standard calibrated product, so its calibration and weighting may differ from the other targets.
- **Stellar parameters** (L\* = 2.57 L☉, M\* = 1.10 M☉) are approximate and set T\_d and h/r.
- **A cosmetic bug** printed "beam NaN au" in the planet-mass table text when no gap was usable; it is fixed in the latest notebook.
- **Gas kinematics** are outside this analysis.
