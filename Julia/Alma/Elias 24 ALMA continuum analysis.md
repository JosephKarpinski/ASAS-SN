# Elias 24: ALMA continuum analysis

Oct 18, 2026 · @Joseph

## Summary

Elias 24 has one well-measured feature, a wide gap near 51–60 au with a ring at about 78 au, and a mass estimate for that gap of about half a Jupiter mass that rests on a single width measurement. Errors are statistical only, and the code has not been validated beyond this comparison.

- **Rings and gaps:** the secure ring is at 78.1 ± 1.1 au (S/N 4.6 in the Gaussian fit; 41σ in the baseline-subtracted profile). Four of the six baseline-subtracted features have no peak or dip in the raw profile, so they are shoulders on a steep profile, not isolated rings or gaps. No published radii are stored for this target, so there is no literature comparison.
- **Geometry:** i = 30° and PA = 45° were fitted from the image (about ±5°), not taken from a catalogue. The ridge-based geometry check could not run, because no ring has S/N ≥ 10.
- **Asymmetry:** the 78 au ring has m=1 = 7.2% (7σ) and m=2 = 11.2% (excess 9.3% over the beam, 9σ). The strongest candidate crescent sits on that ring (+36%, 34° span, 6.7 beams). Its position on the ring crest means a centre error cannot be ruled out or confirmed with this test.
- **Temperature:** T\_b is 8.7 K at the 78 au ring and 4.4 K in the 51 au gap, with absorption-only optical depth 0.50 and 0.17.
- **Planet mass:** the 51 au gap (centre 59.5 au, width 21 au) gives 0.49–0.52 Jupiter masses at α = 10⁻³ and about 0.16 at α = 10⁻⁴. There is no depth estimate because the gap floor is at the noise.
- **Trapping:** the 78 au ring is narrower than the gas scale height (σ\_d/H = 0.71), so it needs dust trapping, with α/St of about 0.4–3.

## Data and method

The analysis uses the DSHARP Band 6 continuum image of Elias 24 at 239 GHz (1.25 mm), made with the Julia ALMA target explorer (v7).

| Item | Value |
| --- | --- |
| Project / member OUS | 2016.1.00484.L / uid://A001/X8c5/X4e |
| Exposure | 4064 s (1.13 h) |
| Beam | 0.0458 × 0.0292 arcsec = 6.2 × 4.0 au |
| Pixel scale | 0.0030 arcsec |
| Distance | 136.0 pc |
| Geometry | i = 30°, PA = 45°, fitted from the image (radial-symmetry criterion, about ±5°); not in the catalogue |
| Image noise | 0.0198 mJy/beam (robust rms) |
| Star | L\* = 6 L☉, M\* = 0.78 M☉ (approximate; verify) |
| Profile | azimuthal mean, 79 bins of 3.1 au out to 245 au |
| Inner exclusion | inside 16 au (2.5 beams), as beam and CLEAN artifacts |

The archive lists three observations; the finest (0.0222 arcsec) was used. A purely radial profile explains 98% of the pixel variance (95% if face-on), so the geometry is only weakly constrained at this low inclination. The peak of the azimuthal mean is 3.92 mJy/beam at 1.5 au, and 90% of the flux lies within 125 au (20 beams).

## Rings and gaps

The 78 au ring and a wide gap around 51–60 au are real; most of the other features are shoulders on a steeply falling profile.

Gaussian fits (tapered power law plus 3 Gaussians, χ²\_red = 148, errors scaled by √χ²\_red; the Gaussians do not describe the profile well):

| r \[au\] | Amplitude \[mJy/beam\] | S/N | Intrinsic FWHM \[au\] | Note |
| --- | --- | --- | --- | --- |
| 13.9 ± 7.1 | 0.89 ± 0.21 | 4.2 | < 50 (unresolved) | radius at its bound |
| 37.2 ± 3.2 | 0.45 ± 0.31 | 1.4 | 11.5 ± 6.1 | below 3σ, unconfirmed |
| 78.1 ± 1.1 | 0.211 ± 0.046 | 4.6 | 12.3 ± 2.8 | the one reliable component |

After baseline subtraction (re-tuned detection), the profile shows:

| Feature | r \[au\] | Residual | Significance | In the raw profile? |
| --- | --- | --- | --- | --- |
| ring | 20.1 | +17.5% | 27σ | no (shoulder) |
| gap | 32.5 | +7.1% | 11σ | no (shoulder; the residual is positive) |
| ring | 41.8 | +25.6% | 26σ | no (shoulder) |
| gap | 51.1 | −61.7% | 25σ | yes |
| ring | 75.9 | +75.1% | 41σ | yes |
| gap | 88.3 | −36.4% | 21σ | no (the raw-profile contrast is 0.4σ) |

Significances are statistical only. The raw peak finder sees one ring and one gap, so only the 51 au gap and the 76 au ring are isolated features. The 88 au dip is an inflection on the outer decline: significant after baseline subtraction but only 0.4σ in the raw profile. The radii and the 22 and 34 au ring spacings depend on the baseline filter, so I would not quote the shoulder positions. No published ring or gap radii are stored for this target, so nothing was compared with the literature.

## Azimuthal asymmetry

The 78 au ring carries strong m=1 and m=2 signals and the strongest candidate crescent; none of this can be tied to a geometry error or ruled out, because the geometry check could not run.

Fourier amplitudes relative to the ring amplitude (assumed geometry):

| r \[au\] | m=1 \[%\] | m=1 S/N | m=2 \[%\] | m=2 S/N | m=2 expected from the beam \[%\] | m=2 excess \[%\] (S/N) |
| --- | --- | --- | --- | --- | --- | --- |
| 13.9 | 1.6 ± 0.4 | 3.9 | 2.1 ± 0.4 | 5.1 | 0.6 | 2.6 (6.5) |
| 78.1 | 7.2 ± 1.0 | 7.1 | 11.2 ± 1.0 | 11.0 | 2.3 | 9.3 (9.1) |

At 78 au the m=1 maximum lies within 25° of the minor axis, which can indicate a near/far-side brightness difference from an optically thick, flared ring. The m=1 maxima at the two rings point different ways (sky PA 280° and 159°), and so do the m=2 orientations (48° and 124°), so there is no common geometry signature across rings. With only two rings this test is weak.

Candidate arcs (beam-smoothed residual above 4σ):

- **#2, 78.6 au, +36%, 6.4σ, 34° span, 6.7 beams:** the only large positive arc. It sits on the ring crest, so a centre error has no first-order effect and this test is not informative. Its sky PA of 180° is close to the m=1 maximum of 159°, so they are probably the same structure. Treat it as a candidate crescent.
- **#1, 42.9 au (+21%, 1.5 beams), #3, 75 au (+31%, 1.4 beams) and #4, 83.3 au (−45%, 1.1 beams):** small, 4.7–6.4σ. A centre error would have to be 2.3, 10.9 and 4.9 au to produce them, well above the 1 au or less that a good geometry fit gives, so they are probably not centre errors, but at this size they are tentative.

The geometry is only good to about 5°, and the ridge-based check that would tighten it could not run. The arc noise is the scatter of the smoothed residual itself, which makes the 4σ threshold conservative.

## Brightness temperature and optical depth

The inner disk is moderately optically thick and the outer ring is not: T\_b reaches 68 K at the centre and 25 K at 14 au, and is below 10 K from the 51 au gap outward.

Planck brightness temperature at 239 GHz from the azimuthal mean, against T\_d = (φL\*/8πr²σ\_SB)^¼ with φ = 0.02 and L\* = 6 L☉:

| Feature | r \[au\] | T\_b \[K\] | T\_d \[K\] | T\_b/T\_d | τ | H/r |
| --- | --- | --- | --- | --- | --- | --- |
| ring | 13.9 | 24.9 ± 0.1 | 52.2 | 0.48 | 0.65 | 0.061 |
| gap (shoulder) | 32.5 | 16.5 ± 0.1 | 34.2 | 0.48 | 0.66 | 0.076 |
| gap | 51.1 | 4.4 ± 0.1 | 27.2 | 0.16 | 0.17 | 0.085 |
| ring | 78.1 | 8.7 ± 0.1 | 21.9 | 0.40 | 0.50 | 0.095 |
| gap | 88.3 | 5.4 ± 0.1 | 20.7 | 0.26 | 0.30 | 0.097 |

Caveats:

- T\_b is beam-averaged, so it is a lower bound for features narrower than the beam.
- τ assumes absorption only; scattering lowers T\_b, so the true τ is higher.
- T\_d is an assumed passive-disk value with φ = 0.02 and ignores accretion heating. The T\_b errors are statistical.
- The 32.5 au row is a shoulder on the declining profile, not a true gap. Its T\_b of 16.5 K says only that the profile is still bright there.

## Planet mass

The 51 au gap gives about half a Jupiter mass from its width at α = 10⁻³ and about 0.16 at α = 10⁻⁴, and no depth estimate; the value is order-of-magnitude only.

Gap width uses the Kanagawa et al. 2016 relation, Mp/M\* = 2.1×10⁻³ (Δ/Rp)² (h/r / 0.05)^1.5 (α/10⁻³)^0.5, with M\* = 0.78 M☉ and h/r from the passive-disk temperature. The range runs from the deconvolved to the observed width.

| Gap | r\_c \[au\] | h/r | Depth | W\_obs / W\_int \[au\] | W/r | Width, α = 10⁻³ | Width, α = 10⁻⁴ | Depth mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 51.1 | 59.5 | 0.088 | > 83 (lower limit) | 21.4 / 20.8 | 0.36 | 0.49–0.52 M\_Jup | 2.9–3.1 M\_Nep (about 0.16 M\_Jup) | n/a |

Caveats attached to these numbers:

- **Centre and width are uncertain.** The measured centre (59.5 au) is 8.4 au beyond the detected gap radius (51.1 au), more than one beam. The gap sits on a steeply falling profile, so the inner edge is the slope of the inner disk and the width may be biased. The same kind of bias affected the HD 163296 45 au gap. W/r = 0.36 is large.
- **No depth estimate.** The gap floor is within the noise (contrast > 83, a lower limit), so the depth relation gives no mass.
- **Gap radius from the literature not compared.** No published radii are stored. From memory, I think the published gap is near 55–60 au and published planet-mass estimates are of order 0.5–1 Jupiter masses, but I have not checked these, so verify them before quoting.
- **The relation is for gas gaps.** Dust gaps from large grains are wider, so width masses are upper limits on that count. Zhang et al. 2018 dust calibrations are not applied. On HD 163296 the same relation gave values 2–7× below published masses for an 84 au gap, so this value may be low.
- **α and h/r are assumptions.** α is unknown and the two values bracket the usual range; h/r = 0.088 comes from an assumed temperature and approximate L\* and M\*.
- **Two other gaps were not evaluated.** The 33 au "gap" has no outer peak (the profile never rises past 35.6 au) and is a shoulder. The 88 au gap has a raw-profile contrast of only 0.4σ.

## Dust trapping and gap spacing

The 78 au ring is narrower than the gas scale height, so it needs dust trapping; the inner ring gives no limit, and there is no usable gap-spacing test.

Following Dullemond et al. 2018, α/St = 1/((w/w\_d)² − 1), where w\_d is the beam-deconvolved Gaussian width of the ring and w the width of the gas pressure bump. The largest α/St comes from w = H; the smallest from w equal to the distance to the nearest gap.

| r \[au\] | σ\_d \[au\] | H \[au\] | σ\_d/H | α/St, largest | α/St, smallest | St (α = 10⁻³) | St (α = 10⁻⁴) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 13.9 | < 21.3 | 0.9 | 25 | no limit | n/a | n/a | n/a |
| 78.1 | 5.2 ± 1.2 | 7.4 | 0.71 | 1.01 (0.42–3.12) | 0.360 | 9.9×10⁻⁴ to 2.8×10⁻³ | 9.9×10⁻⁵ to 2.8×10⁻⁴ |

The 78 au ring has σ\_d/H = 0.71 ± about 0.2, so α/St lies between 0.36 and about 1 (with a statistical range up to 3). The ring is only 4.6σ in the Gaussian fit and the fit has χ²\_red = 148, so the width is poorly constrained. The ring width is the intensity width; for optically thick rings it overestimates the dust width, so true trapping would be stronger. The 14 au ring is unresolved.

Gap spacing: the code lists 33→51 au (period ratio 1.97, 1.5% from 2:1) and 51→88 au (2.27, 9.2% from 5:2). Neither counts: the 33 au and 88 au features are not measured gaps, and a ±3% match to a low-order ratio happens by chance for about 40% of values anyway. Only one gap was measured, so there is no spacing result.

## Open issues

- **Geometry** was fitted from the image (about ±5°) and the ridge-based check could not run; a ring with S/N ≥ 10 would be needed. A different PA or inclination would shift the radii and asymmetry numbers.
- **Gap measurement on a sloping profile:** the 51 au gap centre and width are probably biased by the steep inner disk. A flag for asymmetric gaps has since been added to the code, which would mark this gap; the numbers here predate it.
- **Literature comparison** is missing: no published radii are stored for this target. Gap radius and published masses need to be checked against DSHARP and the planet-mass literature.
- **Zhang et al. 2018 dust calibrations** are not implemented; masses use the gas relations.
- **Stellar parameters** (L\* = 6 L☉, M\* = 0.78 M☉) are approximate and set T\_d, h/r and the masses.
- **Ring fit χ²\_red = 148:** widths and amplitudes are unreliable and the formal errors mean little; the 14 au component is unresolved and the 37 au one is below 3σ.
- **Crescent at 78 au** is a candidate only, since its position on the ring crest makes the centre-error test uninformative and the geometry is uncertain.
- **Beam artifacts:** the inner 16 au is excluded, and CLEAN or uv-coverage effects are not modelled.
- **Gas kinematics** are outside this analysis.
