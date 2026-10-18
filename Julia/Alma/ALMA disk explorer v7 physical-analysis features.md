# ALMA disk explorer v7: physical-analysis features

Oct 18, 2026 · @Joseph

## Overview

Version 7 of the Julia ALMA target explorer adds four physical analyses that work on a continuum image and its azimuthal-mean profile, all run from one function, `physical_analysis`. None of them has been validated against independent code; they were tested by running four targets and comparing with published values where I could.

| Analysis | What it measures | Main output |
| --- | --- | --- |
| 1. Azimuthal asymmetry | what the axisymmetric model leaves behind: lopsided rings, arcs, eccentricity, geometry errors | m=1 and m=2 amplitudes per ring, candidate arcs, ring-ridge geometry and ellipticity |
| 2. Brightness temperature and optical depth | T\_b from Jy/beam, compared with a passive-disk dust temperature | T\_b, T\_d, τ per ring and gap |
| 3. Gap-opening planet masses | planet mass from gap width and depth (Kanagawa et al. 2015, 2016) | mass range per gap at α = 10⁻³ and 10⁻⁴ |
| 4. Ring widths, dust trapping, ring spacing | α/St from ring width against the gas scale height (Dullemond et al. 2018); Keplerian period ratios | α/St limits, St ranges, resonance-like ratios |

Inputs: the calibrated continuum FITS image (beam and frequency are read from the header), the distance, and the disk inclination and position angle (from the catalogue, or fitted from the image when the catalogue has none). Analyses 2–4 also need the stellar luminosity L\* and mass M\*; the stored values are approximate (Andrews et al. 2018) and need verifying.

The gas-kinematics analysis in the original plan is a separate project and is not included. Extra diagnostics that went beyond the four analyses (flaring test, round-beam test, vertical-thickness scan) are off by default. Each analysis is isolated: if one fails, the others still run, and the report says why.

## 1. Azimuthal asymmetry

Azimuthal averaging throws away everything that is not axisymmetric; this analysis recovers it by subtracting the mean and reading what is left. A lopsided ring or an arc can point to a dust trap in a vortex (Rossby-wave instability) or to a spiral from an embedded planet, and a ring that is eccentric on its own can point to a planet or a binary. It needs only the image and the geometry.

### Residual image

The residual is each pixel minus the azimuthal mean at the same deprojected radius, using the fitted ring-ridge centre shift when one is available (applied at sub-pixel level). The report shows this as an image, as an azimuth–radius map, and as Fourier amplitudes.

### Fourier amplitudes per ring

For each fitted ring, the mean residual is measured in azimuth wedges within ± one observed ring width (at least half a beam) and decomposed into m=1 and m=2 terms, as a percentage of the ring's fitted amplitude, with a maximum position angle. Errors come from the image noise. The false-alarm probability exp(−S/N²/2) is for pure noise only and for a single ring; geometry errors and CLEAN artifacts are not in it.

- **Elliptical-beam expectation for m=2.** A beam that is not round makes an axisymmetric ring brighter where the ring runs along the beam's short axis, which is a pure m=2 pattern. The code computes this expectation for each ring from its fitted width and reports the excess over it. Without this, a few per cent of m=2 is not evidence of structure.
- **Coherence across rings.** A vortex sits at one radius; a centre, inclination or PA error shows up at every ring with the same orientation. The report gives the resultant length R of the m=1 maxima (and of the m=2 orientations) against the value random orientations would give.
- **Before and after geometry.** Amplitudes are measured with the catalogue geometry and with the fitted one. The window mean is blind to small geometry errors at first order (they move the ring ridge, not its brightness), so amplitudes are expected to survive.

### Candidate arcs

The beam-smoothed residual is thresholded at 4σ of its own scatter (a conservative noise estimate, since real structure inflates it). Each region at least one beam in area and outside one beam from the centre is listed with its radius, azimuth, span, sky PA, S/N, contrast (peak smoothed residual over the local azimuthal mean) and area in beams. Two refinements came from testing:

- **Dipole rule.** A positive and a negative region about 180° apart count as a geometry dipole only if their peak amplitudes agree (ratio ≥ 0.7) and their areas agree (ratio ≥ 0.5). Without this, a real crescent whose opposite lobe was half as bright and a quarter the area was dismissed as a dipole.
- **Implied centre shift.** Each arc gets "a centre error would need X au": the shift whose dipole (shift × local profile slope) equals the arc's amplitude. Compare it with the fitted centre uncertainty, which is about 1 au or less. At a ring crest the slope is near zero and the number is meaningless, so it is suppressed there and the arc is marked "not informative".

### Ring-ridge geometry and eccentricity

A shared centre offset, inclination and PA are fitted to the ridge (the radius of peak brightness in azimuth wedges about one beam wide) of all suitable rings, with a free radius for each ring. Suitable means narrow (observed FWHM ≤ max(2.5 beams, 0.25 r)), isolated (no other ring within 2.5 beams) and amplitude S/N ≥ 10. The errors are formal and scaled by √χ²\_red; the ridge fit's χ²\_red is typically 10–20, so the errors are optimistic.

Each ring also gets a residual ellipticity e (m=1 wobble of the ridge radius over R), with its significance and the azimuth where the ridge radius peaks. A significant e after the shared geometry is removed means that ring is eccentric on its own, or the geometry differs between rings (a warp). A flared, optically thick ring looks shifted along the minor axis, which mimics a centre offset; the formal errors exclude that, and an opt-in flaring test exists for it.

If no ring qualifies, the check is reported as failed with the reason and the catalogue geometry is used. This happened for Elias 24 and IM Lup.

### How to read the result

- One positive arc, tens of degrees wide, at a ring radius, contrast tens of per cent: the signature of a dust trap such as a vortex.
- A comparable positive and negative pair 180° apart in the residual image: a centre, inclination or PA error.
- m=2 with the orientation the beam predicts: nothing real.
- A significant ellipticity in one ring but not others: an eccentric ring.

## 2. Brightness temperature and optical depth

This converts the azimuthal-mean intensity to a brightness temperature and compares it with the temperature a passive disk would have, which gives an optical depth. Where τ is of order 1 or more, a gap or ring in the image can be a change in optical depth and not in surface density.

**Method**

- **Brightness temperature.** T\_b is the inverse Planck function at the observing frequency (read from the header, for example 239 GHz), applied to the intensity per beam divided by the beam solid angle from BMAJ and BMIN. It is the full Planck form, not Rayleigh–Jeans.
- **Dust temperature.** T\_d = (φ L\* / 8π r² σ\_SB)^¼, with flaring factor φ = 0.02 (option `phi_dust`) and a floor of 10 K.
- **Optical depth.** τ = −ln(1 − T\_b/T\_d), absorption only. The plot caps τ at 4.6.
- **Scale height.** H/r = c\_s/v\_K from T\_d and M\*, used again in sections 3 and 4.

The report gives T\_b with its error, T\_d, T\_b/T\_d, τ and H/r at each detected ring and gap, and a figure shows T\_b, T\_d, τ and the feature radii.

**Reading it**

- τ well below 1: optically thin, so intensity traces the dust surface density.
- τ of order 1 or more: partly or fully optically thick. On IM Lup the 6 au region has τ = 1.2; HD 163296 and Elias 24 rings are at 0.4–0.7.

**Caveats**

- T\_b is beam-averaged, so it is a lower bound for any feature narrower than the beam, including the inner core.
- Scattering by large grains lowers T\_b below the absorption-only value, so the true τ is higher.
- T\_d is a passive-disk estimate with an assumed φ; it ignores accretion heating and the difference between the surface layer and the midplane. L\* and M\* are approximate.
- The T\_b errors are statistical only.

## 3. Gap-opening planet masses

Each detected gap is measured on the azimuthal-mean profile and converted to a planet mass range with two relations from the literature, at α = 10⁻³ and 10⁻⁴. The result is order-of-magnitude: it depends on the assumed α and h/r, on the gap measurement, and on relations calibrated for gas gaps.

**Relations**

```latex
\frac{M_p}{M_*} = 2.1\times10^{-3}\left(\frac{\Delta_{gap}}{R_p}\right)^{2}\left(\frac{h/r}{0.05}\right)^{3/2}\left(\frac{\alpha}{10^{-3}}\right)^{1/2}
```

This is the Kanagawa et al. (2016) width relation, Δ\_gap/R\_p = 0.41 (M\_p/M\*)^{1/2} (h/r)^{-3/4} α^{-1/4}, solved for the mass. The depth relation is Kanagawa et al. (2015), Σ\_min/Σ\_0 = 1/(1 + 0.04 K) with K = (M\_p/M\*)² (h/r)^{-5} α^{-1}, taking Σ\_0/Σ\_min as the dust-intensity contrast. The Zhang et al. (2018) dust-specific calibrations, which also depend on grain size, are not implemented, so dust gaps are read with the gas relations. M\* comes from the stored stellar parameters; h/r comes from the passive-disk temperature at the gap centre.

**Gap measurement (`measure_gap`)**

- Start from the lowest profile bin within half a beam of the requested gap radius, then slide to the true local minimum, at most 1.5 beams away (a baseline-subtracted gap can sit about a beam inside the raw minimum).
- The depth is the next ring peak outside the gap divided by the gap minimum. The edge level is (peak + minimum)/2 (Zhang et al. 2018 definitions), and the width is between the two crossings of that level. Each edge is searched only up to the neighbouring ring peak, so two adjacent gaps keep their own edges. If the inner side never reaches the edge level, the width is twice the outer half-width.
- If the gap floor is within 3σ of zero, the depth is only a lower limit and no depth mass is given.
- Gaps with contrast below 3σ are not evaluated.
- The width range runs from the beam-deconvolved width to the observed width (beam removed in quadrature, using the geometric-mean beam FWHM).
- Gaps that cannot be measured are listed under "Not evaluated" with a specific reason: no profile bin near the gap, no local maximum outward of the minimum (the profile does not rise, so it is a shoulder), outer peak not above the minimum, outer edge not found, or contrast below 3σ.

**Report columns:** gap radius asked for, r\_c (measured centre), h/r, depth, W\_obs and W\_int (au), W/r, width mass at both α, depth mass at both α, and notes. Notes flag an inner peak below the edge level, a shallow gap (depth < 2.2, where the width relation is formally invalid), an unresolved width, a noise-floor depth and, since the last change, an asymmetric gap (half-widths differing by 1.5× or more).

**Caveats**

- Mass scales as α^(1/2) for both relations, and α is unknown, hence the two values.
- The width relation is for gas gaps; dust gaps from large grains are wider, so width masses are upper limits on that count. The depth is diluted by the beam (a lower bound on contrast) but dust gaps are deeper than gas gaps (an upper bound on mass). The two effects partly cancel and are not quantified.
- On HD 163296 the width-based masses were 2–7× below published masses for the 84 au gap, so they are likely low.
- Gaps on a steep inner profile or beside an arc can be biased (HD 163296 45 au, Elias 24 51 au). The asymmetry flag marks these but does not correct them. A Gaussian-dip fit I prototyped as a correction disagreed with the edge method on clean synthetic gaps and would turn shoulders into gaps, so it was not adopted.
- Several gaps may be carved by one planet, or by other mechanisms.

## 4. Ring widths, dust trapping and ring spacing

A ring's width against the local gas scale height tells you whether a gas pressure bump can explain it or whether the dust must be trapped, and constrains α/St, the ratio of turbulence to grain size. The ring spacing can also be written as Keplerian period ratios, to look for resonance-like patterns, which is suggestive only.

**Dust trapping.** A dust ring in a Gaussian gas pressure bump of width w has width w\_d = w √(α/(α + St)), so α/St = 1/((w/w\_d)² − 1) (Dullemond et al. 2018, DSHARP VI).

- w\_d is the Gaussian σ of the fitted ring with the beam removed in quadrature.
- A long-lived bump cannot be narrower than about one gas scale height H, so w = H gives the largest α/St the ring allows.
- A bump as wide as the distance to the nearest gap gives the smallest.
- A ring wider than H gives no limit, because the bump could be as wide as the ring.
- The Stokes number range is α divided by the two α/St limits, for α = 10⁻³ and 10⁻⁴.
- A ring narrower than H (σ\_d/H < 1) cannot be a gas density bump alone; it needs dust trapping in a pressure maximum.

The Gaussian ring fit behind it is a tapered power law plus Gaussians seeded from the detected rings, published radii (marked "literature") and unexplained residuals. Components with amplitude S/N below 3 are flagged as unconfirmed, and errors are scaled by √χ²\_red when it exceeds 1.

**Ring spacing.** The code lists the radius ratio, the orbital period ratio (r₂/r₁)^(3/2) and the nearest of eight low-order commensurabilities (6:5, 5:4, 4:3, 3:2, 5:3, 2:1, 5:2, 3:1), flagging matches within 3%. A ±3% match to one of those ratios happens by chance for about 40% of values between 1.1 and 3.2, so a near-resonance is weak evidence on its own. The table now uses only gaps that were actually measured, at their measured centres, so a shoulder that was rejected cannot create a ratio.

**Caveats**

- The ring width is the intensity width. For optically thick rings it overestimates the dust-density width, so true rings are, if anything, narrower and trapping stronger.
- Blended components and rings with large errors are unreliable; trust isolated rings. The ring fit χ²\_red was 3.1 for HD 163296 and 148 for Elias 24, where the Gaussians do not describe the profile and the errors mean little.
- Results depend on the seed positions and on M\* and L\* (through H).

## Running it

Run the pipeline for a target first, then pass its result to the physical analysis:

```julia
physical_analysis(out)                                   # all four analyses with the defaults
physical_analysis(out; lstar = 6.0, mstar = 0.78)        # supply stellar parameters (solar units)
physical_analysis(out; use_fitted_geometry = true)       # adopt the ridge-fitted geometry
physical_analysis(out; extra_seeds_au = [15.0, 28.0])    # extra ring radii to start the fit from
```

The `out` argument is the result of `run_alma_pipeline`; the exact pipeline call for a given target depends on the target menu cells.

| Option | Default | Effect |
| --- | --- | --- |
| `rings`, `gaps` | detections from the pipeline | detections to use; pass retuned ones after retuning |
| `extra_seeds_au` | none | extra ring radii to start the Gaussian fit from |
| `use_literature_seeds` | true | also seed from published ring radii |
| `lstar`, `mstar` | stored values | stellar luminosity and mass in solar units |
| `phi_dust` | 0.02 | flaring factor in T\_d |
| `arc_sigma` | 4 | arc threshold, in units of the smoothed-residual noise |
| `max_added_rings` | 3 | extra ring components allowed for unexplained residuals |
| `do_geom`, `geom_min_snr` | true, 10 | ring-ridge geometry fit and its S/N cut |
| `use_fitted_geometry` | false | recompute the profile with the fitted geometry; by default the asymmetry is measured with both and compared |
| `do_fit`, `do_asym`, `do_temp`, `do_planets`, `do_trapping` | true | switch each analysis off |
| `do_flare`, `do_roundbeam`, `do_thickness` | false | opt-in extras: flaring test, round-beam re-measure of m=2 (takes minutes), vertical-thickness scan |

Outputs are written next to the other pipeline outputs: a text report `<slug>_physical_analysis.txt` and figures for the ring fits, the azimuthal analysis (residual map, azimuth–radius map, Fourier amplitudes), the brightness temperature and the ring-ridge geometry. The report has numbered sections: 1 ring fits, 1b geometry check, 1c flaring test, 2 asymmetry, 3 temperature and optical depth, 4 planet masses, 5 ring widths, trapping and spacing, followed by the caveats.

## Results on four targets

Four DSHARP Band 6 images were run, each with a write-up: [AS 209](https://claude.ai/code/artifact/3130331d-6e78-4d40-96bd-bef9da3ed25a), [HD 163296](https://claude.ai/code/artifact/f30de391-4a36-4aae-90e6-78df0edc0340), [Elias 24](https://claude.ai/code/artifact/346dd463-69e6-4c3a-a15b-89f1ed79cc3a) and [IM Lup](https://claude.ai/code/artifact/b7ab48f8-ff71-45ec-9622-8be58abfd871). Each was chosen to test a different part of the code.

| Target | Role in testing | Outcome |
| --- | --- | --- |
| AS 209 | reference case with several resolved gaps | geometry fit i = 35.0°, PA = 85.6°, centre shift about 1 mas; fitted geometry did not change the m=1 or m=2 amplitudes; three gaps gave masses and one (33 au) was not evaluated; the only two arcs were a central dipole |
| HD 163296 | second target with published values to compare | ring radii at 67.6 and 100.8 au and the 87 au gap centre agree with DSHARP within a beam; geometry i = 46.5°, PA = 132.9°; a real crescent at 55.6 au; width-based masses for the 84 au gap 2–7× below published values; the 45 au gap measurement unreliable |
| Elias 24 | wide-gap case at low inclination | geometry fitted from the image because the catalogue has none, ridge check could not run; one gap giving about 0.5 Jupiter masses from its width; candidate crescent on the 78 au ring; ring fit χ²\_red = 148 |
| IM Lup | negative control | no isolated rings or gaps, so no masses or trapping limits, and the code invented none; optically thick core; ten small tentative arcs |

The tests found real problems, listed in the next section. The agreement of ring and gap radii with the literature is the only independent check so far; the temperatures, asymmetry amplitudes, masses and α/St limits have not been compared with any independent calculation.

## Known limitations and changes during testing

**Changes made after testing on real targets**

| Problem found | Change |
| --- | --- |
| The geometry check failed on HD 163296 because both rings were wider than the "narrow" cut | narrow now means observed FWHM ≤ max(2.5 beams, 0.25 r) |
| A real crescent on HD 163296 was dismissed as a geometry dipole although its opposite lobe was half as bright and a quarter the area | pairing needs comparable amplitude (≥ 0.7) and area (≥ 0.5); arcs now give the implied centre shift |
| "Not evaluated" gaps had a catch-all reason | each failure now has a specific reason |
| The HD 163296 45 au gap failed because the slide to the true minimum was just past a 1-beam cap | the cap is now 1.5 beams |
| "Centre error would need X au" was meaningless at ring crests | suppressed there; the arc is marked not informative |
| Gaps on a steep or arc-contaminated profile gave biased widths without warning | asymmetric gaps (half-widths differing by 1.5× or more) are flagged |
| The spacing table used shoulders that were not measured gaps | it now uses only measured gaps at their centres |
| The planet-mass text printed "beam NaN au" when no gap was usable | it now says "the beam" |
| Extra diagnostics (m=2 round-beam test, flaring, second-observation comparison) went beyond the plan | flaring, round-beam and thickness are opt-in and off by default |

**Limitations that remain**

- **Nothing has been run by me.** There is no Julia in my environment, so every change was checked only by reading the code, a format-string check and, for one design question, a Python prototype. All runs were done by you.
- **Errors are statistical only.** They exclude CLEAN artifacts, baseline-filter ripples, calibration error and geometry error. The ridge fit's χ²\_red is 10–20, so its errors are optimistic.
- **Planet masses are order-of-magnitude.** They depend on α and h/r, use gas-gap relations on dust gaps, and omit the Zhang et al. (2018) dust calibrations. On HD 163296 the width-based masses were likely low.
- **Gap measurement on sloping profiles** is flagged, not corrected. A Gaussian-dip fit on a local linear baseline was prototyped and rejected: it disagreed with the edge method on clean synthetic gaps, was biased by neighbouring rings, and would turn shoulders into gaps.
- **Stellar parameters** (L\*, M\*) are approximate and set T\_d, h/r and every mass.
- **Smooth disks:** the code reports shoulders as features after baseline subtraction (for example on IM Lup and AS 209's 33 au), and the radii of those depend on the filter.
- **The geometry check needs a bright, isolated, narrow ring** (S/N ≥ 10); it does not run for faint or low-contrast disks.
- **No spiral model.** Spirals are averaged out of the profile and appear only as arcs.
- **The inner 2.5 beams are excluded** as beam and CLEAN artifacts, which can hold much of the flux.
- **Published values quoted in the target write-ups** (DSHARP radii, crescent positions, planet masses) were read from page summaries or recalled from memory and need checking against the tables before publication.
- **Gas kinematics** (the original item 5) is not included.
