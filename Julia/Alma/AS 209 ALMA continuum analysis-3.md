# AS 209: ALMA continuum analysis

Oct 18, 2026 · @Joseph

## Summary

Two AS 209 rings (74 and 121 au) are well measured and the 121 au ring shows clear dust trapping; the planet masses, asymmetries and spacing are indicative only. Every number below carries its caveat in the section where it appears.

- **Solid:** ring centres at 74.2 and 120.8 au (S/N 27); fitted widths 9.4 and 10.3 au FWHM; the 121 au ring is narrower than the gas scale height (σ\_d/H = 0.43), so the dust is trapped (α/St ≈ 0.2).
- **Moderate:** peak brightness temperature 25.8 K; τ ≈ 0.6–0.74 at the two main rings (absorption only); planet masses of about 1–3 Neptune masses for the 59 and 110 au gaps, uncertain by a factor of 2–3 plus a factor of about 3 from α.
- **Tentative:** the m=2 signal (about 7%) is close to the beam's prediction but exceeds it by 5–6%; the m=1 signal at 307° is unexplained; gap period ratios are consistent with chance.
- **Not found:** no isolated vortex-like arc above 4σ.
- **Not tested:** the Zhang+18 dust-based mass calibrations, and gas kinematics.

## Data and method

The analysis uses the DSHARP Band 6 continuum image at 121 pc; the stellar parameters are approximate and unverified.

| Item | Value | Caveat |
| --- | --- | --- |
| Project / band | 2016.1.00484.L (DSHARP), Band 6, 239.0 GHz | Image `member.uid___A001_X8c5_X62.calibrated_final_cont.pbcor.fits` |
| Beam | 0.0529" × 0.0361" = 6.4 × 4.4 au | Features narrower than about 1 beam are deconvolved, not resolved |
| Image rms | 0.0156 mJy/beam | Robust rms; ignores CLEAN artifacts |
| Catalogue geometry | i = 35.0°, PA = 85.8° | Literature values |
| Fitted geometry | i = 34.9°, PA = 85.6°, centre shift (−1.1, +0.4) mas | Ridge χ²\_red = 12.84, so errors are underestimated; ellipticity 0.4% / 0.3% |
| Stellar parameters | L\* = 1.41 Lsun, M\* = 0.83 Msun | From memory, approximate; verify against the literature before quoting |
| Distance | 121 pc | Catalogue value |

**Method.** The code deprojects the image with the fitted geometry and builds an azimuthal-mean profile. It fits Gaussian rings, estimates Fourier m=1 and m=2 amplitudes in ring windows, converts intensity to brightness temperature, and derives planet masses and trapping limits from the fitted widths. Quoted errors are formal and statistical only.

## Rings and gaps

The 7-component fit (χ²\_red = 6.03, 52 dof) gives two well-measured rings at 74 and 121 au; the rest are blended, weak or unresolved. The smooth disk component collapsed onto the core, so only isolated rings are trustworthy.

| Ring centre (au) | Intrinsic FWHM (au) | S/N | Caveat |
| --- | --- | --- | --- |
| 14.0 | not reported | 1.9 | Blended; inside the beam/CLEAN zone |
| 28.7 | unresolved (< 40 au) | 13.8 | Blended with the core |
| 45.3 | not reported | 1.0 | Kept only to reduce χ² |
| 74.2 ± 0.1 | 9.4 ± 0.4 | 27.2 | Solid |
| 95.7 | 6.9 ± 2.6 | 4.8 | Weak; large error |
| 120.8 ± 0.2 | 10.3 ± 0.5 | 27.8 | Solid |
| 140.3 | 20.0 ± 2.8 | 14.1 | Wide; near the negative CLEAN bowl beyond about 150 au |

**Gaps.** The baseline-subtracted profile has gap detections at 33, 59, 84, 110 and 132 au. The 33 and 132 au gaps are not real dips in the raw profile, so they are baseline artifacts until shown otherwise. Of 11 testable published features, 9 are recovered within 1 beam (6.4 au).

**Caveats.** Errors are formal and statistical. The residuals show a ring-to-ring offset, and the smooth-disk collapse leaves a shoulder degeneracy, so the widths of rings near the core depend on the baseline.

## Azimuthal asymmetry

No isolated vortex-like arc exceeds 4σ; the two candidates are the central dipole only. The ring-window Fourier amplitudes are given relative to the ring amplitude.

| Ring | Mode | Amplitude | Significance | Orientation | Caveat |
| --- | --- | --- | --- | --- | --- |
| 74 au | m=2 | about 6.9% | 8.5σ | 85° | Beam predicts 100°; excess over the beam is 5.5% |
| 121 au | m=2 | about 7.5% | 10.1σ | 84° | Excess over the beam is 6.3% |
| 29 au | m=1 | 2.4% | 6.8σ | 307° | Ring is blended with the core |
| 74 au | m=1 | 3.2% | 3.9σ | 307° | Unexplained |

**Reading.** The m=2 signal is aligned with the elliptical beam, so it is mostly beam shape. The 5–6% excess is not explained by a Gaussian beam and could come from CLEAN or uv-coverage effects, a non-Gaussian PSF, or a small geometry error. It is not evidence of a real asymmetry yet. The m=1 signal at 307° is unexplained and appears at two radii.

**Caveats.** Significances are statistical and ignore CLEAN correlations between pixels. Window-mean amplitudes are insensitive to small geometry errors. The 96 and 140 au rings have large errors and are not quoted.

## Brightness temperature and optical depth

Peak brightness temperature is 25.8 K, and the two main rings are moderately optically thick (τ ≈ 0.6–0.74, T\_b/T\_d ≈ 0.46–0.52).

- **Dust temperature model:** T\_d = (φ L\* / 8π r² σ\_SB)^¼ with φ = 0.02 and a 10 K floor. This is a passive-disk assumption and depends on L\* = 1.41 Lsun, which is unverified.
- **Optical depth:** τ = −ln(1 − T\_b/T\_d) counts absorption only. Scattering would raise the true τ and lower the apparent T\_b, so τ here is a lower bound in that sense.
- **Scale height:** h/r = 0.06–0.09, from the same passive-disk temperature. It is not a measurement, and the planet masses scale as (h/r)^1.5.
- **Beam dilution:** T\_b is a beam-averaged value, so narrow rings are diluted and their true T\_b is higher.

## Planet masses

The 59 and 110 au gaps imply roughly 1–3 Neptune masses (width relation); the 84 au gap is an upper limit, not a planet detection. Masses use the Kanagawa+16 width relation with M\* = 0.83 Msun, and the two values per gap are for α = 10⁻³ / 10⁻⁴.

| Gap (au) | Width, α = 10⁻³ / 10⁻⁴ | Mass (Neptune masses), α = 10⁻³ / 10⁻⁴ | Caveat |
| --- | --- | --- | --- |
| 58.9 (r\_c 61.7) | 14.6 / 13.6 au | 2.8–3.3 / 0.9–1.0 | Range spans beam-deconvolved to observed width |
| 84.4 (r\_c 89.6) | 6.2 / 3.2 au | ≲ 0.3 / 0.1 | Depth 2.31; upper limit only. The depth relation gives 5.1 / 1.6, which disagrees |
| 109.9 (r\_c 106.7) | 15.7 / 14.8 au | 1.4–1.6 / 0.44–0.49 | Inner peak is below the edge level, so width = 2 × outer half-width |

Not evaluated: the 33 and 132 au gaps, which are not dips in the raw profile.

**Caveats.**

- The quoted ranges reflect only the beam deconvolution. The real uncertainty is a factor of 2–3, plus a factor of about 3 from the α bracket (masses scale as α^½).
- Masses scale as (h/r)^1.5, and h/r comes from an assumed passive-disk temperature.
- A dust gap is not a gas gap. The Zhang+18 dust calibrations are not implemented.
- Depth-based masses are a cross-check only; they are suppressed where the gap floor is at the noise level.
- No comparison has been made yet against published Zhang+18 masses for AS 209.
- On HD 163296 the same width relation gave 0.08–0.25 Jupiter masses for the 87 au gap, 2–8× below the roughly 0.5–2 Jupiter masses DSHARP IX quotes from hydrodynamic and kinematic work, but the 45 au gap gave 0.83–0.86 Jupiter masses, which is unreliable (its measured centre, 50.8 au, is 6 au off the published 44.8 au, probably pulled by the 55 au crescent). These AS 209 masses may therefore be low, but the HD 163296 test does not give a reliable correction factor.

## Dust trapping and gap spacing

The 121 au ring is clearly trapped (α/St ≈ 0.2); the 74 au ring gives only a weak constraint. The method follows Dullemond+18: w\_d = w√(α/(α+St)), where w\_d is the Gaussian σ with the beam removed and w is bounded below by the gas scale height H.

| Ring (au) | σ\_d (au) | H (au) | σ\_d/H | α/St | St (α = 10⁻³) | Caveat |
| --- | --- | --- | --- | --- | --- | --- |
| 28.7 | n/a | n/a | n/a | no limit | n/a | Unresolved and blended |
| 74.2 | 4.0 ± 0.2 | 5.6 | 0.71 | 0.18–1.03 (0.88–1.20) | 9.7×10⁻⁴ to 5.5×10⁻³ | Weak constraint; the ring is only slightly narrower than H |
| 95.7 | n/a | n/a | n/a | 0.07–0.17 | n/a | Weak and uncertain ring |
| 120.8 | 4.4 ± 0.2 | 10.3 | 0.43 | 0.19–0.22 | 4.5–5.2×10⁻³ | Clear trapping; for α = 10⁻⁴, St = 4.5–5.2×10⁻⁴ |
| 140.3 | n/a | n/a | n/a | ≤ 0.88 | n/a | No lower bound |

**Caveats.** H comes from the assumed passive-disk temperature and the unverified L\*. Intensity width overestimates the dust-density width for optically thick rings, so true trapping is, if anything, stronger. Converting St to grain size needs the gas surface density, which belongs with the gas-kinematics work.

**Gap spacing.** Period ratios for the 33 → 59 → 84 → 110 → 132 au gaps are 2.34, 1.71, 1.49 and 1.32. Three of the four lie within 3% of 5:3, 3:2 and 4:3, but a chance match at that level occurs for about 40% of ratios, and ≥ 3 of 4 matches arise by chance about 18% of the time. This is not significant, and the 33 and 132 au positions are weak. A table restricted to the measured gaps (62, 90, 107 au) has not been made.

## Open issues

Verify these before quoting any number externally.

- [ ] Check L\* = 1.41 Lsun and M\* = 0.83 Msun against the literature; they feed T\_d, H, and every planet mass and trapping limit.
- [ ] Compare the planet masses with the published Zhang+18 AS 209 values.
- [ ] Test whether the m=2 excess over the beam (5–6%) survives a different CLEAN or a simulated-beam injection.
- [ ] Find the cause of the m=1 signal at 307°, seen at 29 and 74 au.
- [ ] Re-fit with a model that keeps a smooth disk component, to remove the shoulder degeneracy near the core.
- [ ] Treat errors as formal and statistical only; the ridge χ²\_red of 12.84 and the ring-to-ring residual offset mean they are underestimated.
- [ ] Note the negative CLEAN bowl beyond about 150 au, which affects the 140 au ring.
- [ ] Run the second target (see chat) to check the code on a different disk.
