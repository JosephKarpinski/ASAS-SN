#=

## What the module does

`GalaxyAnalysis` measures the basic structure of galaxy images (Hubble JPEGs such as NGC 1300, NGC 2525 and NGC 6984) with image-processing steps. Nothing in it is specific to one galaxy: the nucleus, background, mask and ellipses are all measured from each image. It has two public functions: `sweep_threshold` (how sensitive the results are to the mask threshold) and `analyze_galaxy` (the full analysis). The run block at the bottom of the file loops over a list of images and calls both.

## Workflow (what happens when you run the file)

For each image in `galaxies`:

1. `sweep_threshold` measures the disk shape, concentration and bar amplitude at a series of `min_rise` thresholds, finds plateaus where the shape stops changing, and suggests a threshold. If no plateau exists, `fallback_min_rise` is used.
2. `analyze_galaxy` runs the full analysis at that threshold.
3. After all images, a summary table is printed and `galaxy_summary.csv` is written.

Per-image settings go in the `overrides` dictionary (for example `"NGC 6984.jpg" => (; center = (856, 1404))`); they are passed to both functions. A failed image is reported and skipped, and the others continue. The start and end banners show the code version, so you can confirm which file actually ran.

## `analyze_galaxy` pipeline

1. **Load and convert:** the colour JPEG becomes a single grayscale (luminance) image with values from 0 to 1.
2. **Remove stars:** pixels at or above `star_level` (0.99) seed a star mask. Each saturated blob is grown to `star_halo` (8) times its core radius, limited to 8-150 px, and masked pixels are filled with a multi-scale normalized convolution (sigma = 15, 40, 80 px) from the surrounding image. This gives a star-free "working image".
3. **Find the nucleus:** the peak of the star-free image smoothed with sigma = 60 px (`nucleus_smooth`). A warning is printed if it lies within 15% of the frame edge, and `center = (x, y)` overrides it.
4. **Estimate the background:** a plane is fitted to strips along the four image borders, with asymmetric sigma clipping so that galaxy light and stars are rejected. The noise sigma comes from the darker half of the border. The noise floor is 5 sigma of the smoothed image, capped at `max_floor` (0.15).
5. **Build the galaxy mask:** the working image is smoothed (sigma = 6 px) and thresholded at max(noise floor, `min_rise`) above the background plane. Only the connected region containing the nucleus is kept.
6. **Fit the disk ellipse:** a direct least-squares ellipse (Halir-Flusser) is fitted to the outer boundary of the smoothed image at several isophote levels (multiples of the threshold, `iso_factors`), using a 4x-downsampled image, hole filling, iterative clipping of outlying points, and exclusion of points on the frame edge. The disk ellipticity is the median over the levels and the PA is a circular median. The area-moment ellipse is kept only for comparison. Printed output shows epsilon and PA versus semi-major axis, the scatter between levels, and a warning if they drift systematically with radius (more than 0.04 in epsilon or 8 degrees in PA, from the innermost to the outermost isophote). The thin-disk inclination is acos(1 - epsilon).
7. **Deproject:** every pixel gets an elliptical "deprojected radius" about the nucleus. The radial profile and r20/r80 use it.
8. **Concentration:** C = 5 log10(r80 / r20), computed two ways. The mask-based value depends on the threshold. The fixed-aperture value uses all pixels inside the isophote ellipse at the lowest usable multiple of the noise floor (1, 1.5, 2, 3, 4 times), and that aperture must be at least 90% inside the frame. Concentration is also evaluated for apertures at 1, 1.5, 2 and 3 times the floor, and the spread is reported as the aperture sensitivity.
9. **Bar strength:** the m = 2 Fourier amplitude A2 is computed in deprojected annuli, using all pixels and only annuli that lie fully inside the frame. The peak is searched only out to `bar_search_frac` (0.7) times r80. Warnings are printed if the peak sits at that limit with higher A2 beyond it (probably arms or a ring), or if the bar direction is within `bar_align_tol` (3 degrees) of the disk axes (probably a deprojection artifact).
10. **Image filters:** difference-of-Gaussians, Laplacian-of-Gaussian (sigma = `log_sigma`), structure-tensor coherence, and a residual image (working image minus a sigma = 20 px smoothed copy), from which the arm strength and contrast are computed.
11. **Report:** plots, images and `analysis_report.txt`, all with the image-derived filename prefix (for example `NGC_1300_`).

## `sweep_threshold`

It repeats the mask, ellipse, concentration and A2 steps for each `min_rise` in `min_rises` (values within 0.02 of the noise floor are replaced by the floor itself). It then:

- finds plateaus: runs of at least 3 thresholds where epsilon and PA change by no more than 0.03 and 3 degrees between neighbours, the total spread stays within 0.02 and 3 degrees, and the mask covers less than 60% of the frame. Rows with fewer than 2 isophote fits are excluded. All non-overlapping plateaus are listed, the best-ranked one is chosen, and the suggested `min_rise` is its interior point closest to the plateau median.
- summarizes the plateau (median and range of epsilon, PA, inclination and concentration) together with a "wider spread" over all rows whose mask covers 10-60% of the frame.
- checks the isophotes of the suggested row for a radial trend and prints the epsilon range and the outermost isophote if there is one.
- reports how many isophote levels are shared between rows (shared levels are not independent measurements).
- checks A2 stability: the fraction of rows whose A2 peak lies at a consistent radius, and how many rows are pinned at the search limit.
- saves `<prefix>threshold_sweep.csv` and `.png`.

## Summary table and flags

The final table gives, per galaxy: epsilon (median plus or minus the larger of the sweep spread and half the isophote scatter, or the full range when there is a radial trend), PA, inclination, the outermost-isophote epsilon, concentration (fixed aperture, with its range), A2, the A2 radius, A2 stability, and flags:

- `shared-levels`: sweep rows reuse the same isophote levels, so the spread understates the uncertainty.
- `eps(r)` (printed as the Greek letter): epsilon and PA drift with radius, so a single disk ellipse is only approximate. Prefer the outermost isophote for the disk.
- `ap-cov<0.9`: every candidate concentration aperture extends beyond the frame, so the concentration is biased.
- `A2-at-limit`: the A2 peak is pinned at the bar search limit with higher A2 beyond it.
- `A2-unstable`: the A2 peak radius is not stable across thresholds, so a bar detection is not supported.
- `bar-aligned`: the bar axis is within tolerance of the disk axes.

`galaxy_summary.csv` holds the same quantities plus the ranges, uncertainties and intermediate values.

## Reading the report (worked example: NGC 1300)

Typical values from a run of this version:

- Image 3787 x 6637 (rows x columns), nucleus (x, y) = (3233, 1803): a pixel position with y counted from the top, not an astrometric position.
- Background noise sigma = 0.0148 (individual pixels), smoothed-image sigma = 0.0069, noise floor 5 sigma = 0.0345. These are on the 0-1 gray scale.
- `min_rise` = 0.26 was suggested by the sweep. The threshold is a brightness cut above the fitted background, not a physical isophote. The mask covered 24% of the frame, which is very sensitive to `min_rise`.
- Flux = 2.8e6 gray-level x pixels: not calibrated (display-stretched JPEG), so compare only between runs of the same image and settings.
- Isophote fits run from a = 1945 px (epsilon 0.396) to a = 2619 px (epsilon 0.447), a systematic outward increase. The plateau median (epsilon 0.40, inclination 53 degrees) therefore describes the inner isophotes. The honest range is epsilon 0.39-0.45, inclination 52-56 degrees, and the outermost isophote is the better estimate of the disk.
- PA is 30 +/- 2 degrees: the major-axis angle from +x toward +y (downward on screen), repeating every 180 degrees.
- Concentration: the mask-based value (2.33) drifts with the threshold, while the fixed-aperture value stays at 2.07-2.08 over all thresholds and apertures. Quote the fixed-aperture value. The ratio is scale-free, so pixel size cancels. The aperture here is 92% inside the frame.
- A2 = 0.36 at about 1300 px with 90% stability across the sweep: a real, robust bar. Its PA of 22 degrees is about 8 degrees from the disk PA. The radius is binned in 10 px steps and only a proxy for the bar half-length. Converting to arcseconds or kpc needs the pixel scale, which is not in the report.
- Arm strength (0.051) and contrast (0.11) measure small-scale clumpiness (H II knots, dust lanes, background galaxies, JPEG noise count as much as spiral arms). Use them only to compare runs.

## Current results (v9, display-stretched JPEGs)

| Galaxy | epsilon | PA | Inclination | Concentration | Bar |
|---|---|---|---|---|---|
| NGC 1300 | 0.39-0.45 | 30 +/- 2 | 52-56 | 2.07 | supported (A2 0.36) |
| NGC 2525 | 0.36 +/- 0.02 | -45.5 +/- 1 | 50 | 1.76 (1.68-1.79) | stable but weak (A2 0.135) |
| NGC 6984 | 0.37 +/- 0.01 | 39 +/- 1 | 51 | 2.07 (2.05-2.07) | undecided (A2 pinned at limit) |

These have not been compared with published catalog values yet; check them (and the bar classifications) before relying on them.

## Choosing settings for a new image

- `min_rise`: let the sweep choose. Faint, low-surface-brightness galaxies may need much lower values. The sweep skips values within 0.02 of the noise floor.
- Pixel-scale parameters are in pixels, not fractions of the image: `nucleus_exclude` (200), the smoothing scales (6, 20, 60, `log_sigma`), the star radii limits (8-150) and the 10 px annuli. They suit images of a few thousand pixels, so scale them for a very different resolution.
- `center`: pass it (through `overrides`) if the smoothed peak is not the nucleus, for example because of a bright foreground star.
- Noisy or hazy backgrounds (such as NGC 2525): the noise floor is capped at 0.15, so the outermost isophote may be only a couple of sigma above structured haze. Expect larger epsilon uncertainty.

## Limits and caveats

- All brightness-based values come from display-stretched, non-linear JPEGs: they are relative, not photometric, and an "isophote" is a level of displayed brightness. Use calibrated FITS data for photometry.
- Radii are in pixels and flux is in gray levels, so flux cannot be compared between galaxies.
- The inclination is a thin-disk estimate (acos of the axis ratio), and the axis ratio is clamped at 0.2, so edge-on galaxies are unreliable.
- Galaxies cut by the frame bias the outline. Check the aperture coverage and `A2 measured out to` lines.
- Ellipticals, irregulars, mergers and cluster fields: the mask and ellipse still run, but deprojection, bar and arm numbers lose their meaning.
- Unbarred galaxies: A2 then measures any two-fold spiral or disk pattern. Use the A2 stability and limit flags before claiming a bar.
- The 0.7 x r80 bar search limit can cut off a long bar. If A2 keeps rising beyond it (as in NGC 6984), the bar question is open.

=#

using Dates

const CODE_VERSION = "2026-10-04 isophote-ellipse + plateau-v9 FINAL (coverage-limited aperture, trend-aware sweep summary)"
const T_START      = time()

println("\n", "#"^70)
println("# galaxy_analysis.jl  STARTED   ", Dates.now())
println("# code version: ", CODE_VERSION)
println("#"^70, "\n")

module GalaxyAnalysis

const APERTURE_MIN_COVERAGE = 0.9     # the concentration aperture must be >= 90 % inside the frame
const ISO_TREND_ELL = 0.04            # sweep-summary radial-trend thresholds (same as analyze_galaxy)
const ISO_TREND_PA  = 8.0

using Images
using ImageFiltering
using ImageMorphology
using Statistics
using LinearAlgebra
using Dates
using Printf
using Plots

export analyze_galaxy, sweep_threshold

###############################################################
# HELPERS
###############################################################

logmsg(msg) = println("[", Dates.now(), "] ", msg)

# Kernel.gaussian needs a tuple of sigmas (one per dimension)
gblur(img, σ) = imfilter(img, Kernel.gaussian((σ, σ)))

# Rescale to [0,1] safely (avoids divide-by-zero on flat images)
function normalize01(A)
    lo, hi = extrema(A)
    hi == lo && return zeros(Float32, size(A))
    return Float32.((A .- lo) ./ (hi - lo))
end

angdiff(a, b) = abs(mod(a - b + 90, 180) - 90)   # position-angle difference, mod 180°

# Robust "circular median" of position angles (degrees, period 180°): the
# member of the set with the smallest summed angular distance to all others.
# A single outlier (e.g. a bar-dominated isophote) cannot pull the result.
function pa_median_deg(pas)
    k = argmin([sum(angdiff(p, q) for q in pas) for p in pas])
    return pas[k]
end

# Concentration of a sweep row: fixed-aperture value when available, else mask-based
conc_of(r) = isnan(r.conc_ap) ? r.conc : r.conc_ap

# Fraction of isophote levels that are (nearly) repeated across a set of sweep rows.
# Rows whose isophotes share levels are not independent measurements, so the
# spread between them understates the true uncertainty.
function level_share(rs; tol = 0.05)
    isempty(rs) && return 0.0
    lv = sort(Float64[x for r in rs for x in r.levels])
    isempty(lv) && return 0.0
    distinct, last = 1, lv[1]
    for x in lv[2:end]
        if x > last * (1 + tol)
            distinct += 1
            last = x
        end
    end
    return 1 - distinct / length(lv)
end

# Prefix for every output file: user-supplied, or derived from the image name
# ("NGC 1300.jpg" -> "NGC_1300_")
function output_prefix(infile, prefix)
    prefix === nothing || return String(prefix)
    stem = splitext(basename(infile))[1]
    return replace(stem, r"[^A-Za-z0-9]+" => "_") * "_"
end

# Warn if the nucleus is suspiciously close to an image edge
function check_nucleus(xn, yn, X, Y, frac)
    d = min(min(xn - 1, X - xn) / X, min(yn - 1, Y - yn) / Y)
    d < frac &&
        println("WARNING: nucleus (", xn, ", ", yn, ") is within ", round(100d, digits = 1),
                " % of an image edge. If this is not the galaxy centre (e.g. a bright ",
                "star was picked), pass `center = (x, y)`; if it is, the galaxy is ",
                "clipped by the frame and shape results are approximate.")
    return nothing
end

###############################################################
# BACKGROUND: plane fit to the image borders
###############################################################

# Fit  A ≈ c1 + c2*(x/X) + c3*(y/Y)  to the border strips, with iterative
# asymmetric sigma-clipping (bright outliers - stars, galaxy arms reaching the
# edge - are rejected harder than dark ones).  Returns the coefficients and a
# noise σ estimated from the DARKER half of the clipped residuals only, so haze
# and halos on the bright side cannot inflate it.
function fit_plane_background(A; frac = 0.02, niter = 5, clip_hi = 2.0, clip_lo = 4.0)
    Y, X = size(A)
    bw = max(10, round(Int, frac * min(X, Y)))
    pts = vcat(vec(CartesianIndices((1:bw, 1:X))),
               vec(CartesianIndices((Y-bw+1:Y, 1:X))),
               vec(CartesianIndices((bw+1:Y-bw, 1:bw))),
               vec(CartesianIndices((bw+1:Y-bw, X-bw+1:X))))
    v = Float64[A[p] for p in pts]
    M = hcat(ones(length(pts)), Float64[p[2] / X for p in pts], Float64[p[1] / Y for p in pts])

    keep = trues(length(v))
    coef = zeros(3)
    r    = similar(v)
    for _ in 1:niter
        coef = M[keep, :] \ v[keep]
        r    = v .- M * coef
        σ    = max(1.4826 * median(abs.(r[keep])), 1e-4)
        keep = (r .< clip_hi * σ) .& (r .> -clip_lo * σ)
    end

    rk   = r[keep]
    μ    = median(rk)
    dark = rk[rk .<= μ]
    σd   = max(1.4826 * median(abs.(dark .- μ)), 1e-4)
    return coef, Float32(σd)
end

plane_image(coef, Y, X) =
    Float32[coef[1] + coef[2] * (i / X) + coef[3] * (j / Y) for j in 1:Y, i in 1:X]

###############################################################
# STAR MASK (halo size scales with the brightness/size of the saturated core)
###############################################################

function build_starmask(gray, star_level, halo_factor, rmin, rmax)
    Y, X = size(gray)
    seeds  = gray .>= star_level
    labels = label_components(seeds)
    nl     = maximum(labels)
    mask   = falses(Y, X)
    nl == 0 && return mask

    area = zeros(Int, nl)
    sx   = zeros(Float64, nl)
    sy   = zeros(Float64, nl)
    @inbounds for j in 1:Y, i in 1:X
        l = labels[j, i]
        if l > 0
            area[l] += 1
            sx[l]   += i
            sy[l]   += j
        end
    end

    for l in 1:nl
        a  = area[l]
        cx = sx[l] / a
        cy = sy[l] / a
        R  = clamp(halo_factor * sqrt(a / π), rmin, rmax)   # core radius x factor
        for j in max(1, floor(Int, cy - R)):min(Y, ceil(Int, cy + R)),
            i in max(1, floor(Int, cx - R)):min(X, ceil(Int, cx + R))
            if (i - cx)^2 + (j - cy)^2 <= R^2
                mask[j, i] = true
            end
        end
    end
    return mask
end

# Fill masked pixels from valid neighbours (normalized convolution), using
# progressively larger scales so large masked discs are filled from the outside in.
function fill_masked(gray, starmask)
    filled    = copy(gray)
    remaining = copy(starmask)
    for σf in (15, 40, 80)
        any(remaining) || break
        valid = Float32.(.!remaining)
        num = gblur(filled .* valid, σf)
        den = gblur(valid, σf)
        ok  = remaining .& (den .> 0.05f0)
        filled[ok] .= (num ./ max.(den, 1f-3))[ok]
        remaining .&= .!ok
    end
    any(remaining) && (filled[remaining] .= median(gray))
    return filled
end

###############################################################
# IMAGE PREPARATION (shared by analysis and sweep)
###############################################################

# Load, mask + fill stars, find the nucleus on the STAR-FREE image (heavily
# smoothed so compact objects cannot win), then subtract the border plane so
# the background is ~0 everywhere.
function prepare_image(infile; star_level, star_halo, star_min_radius,
                       star_max_radius, nucleus_exclude, nucleus_smooth, center)
    gray = Float32.(Gray.(load(infile)))
    Y, X = size(gray)

    starmask = build_starmask(gray, star_level, star_halo, star_min_radius, star_max_radius)
    filled   = fill_masked(gray, starmask)

    if center === nothing
        yn, xn = Tuple(argmax(gblur(filled, nucleus_smooth)))
    else
        xn, yn = round.(Int, center)
    end

    # un-mask the nucleus region (it is legitimately bright); refill if needed
    near = [hypot(i - xn, j - yn) < nucleus_exclude for j in 1:Y, i in 1:X]
    if any(starmask .& near)
        starmask .&= .!near
        filled = fill_masked(gray, starmask)
    end

    coef, σsky = fit_plane_background(filled)
    gwork = filled .- plane_image(coef, Y, X)
    bg = (; mean = coef[1] + coef[2] / 2 + coef[3] / 2,   # plane level at image centre
            grad_x = coef[2], grad_y = coef[3])           # change across full width / height

    return (; gwork, starmask, xn, yn, Y, X, σsky, bg)
end

###############################################################
# SHAPE: AREA MOMENTS (fallback / comparison)
###############################################################

# Weighted centroid + second-moment ellipse
function ellipse_moments(xs, ys, w)
    W  = sum(w)
    xc = sum(xs .* w) / W
    yc = sum(ys .* w) / W
    dx = xs .- xc
    dy = ys .- yc
    Mxx = sum(w .* dx .^ 2) / W
    Myy = sum(w .* dy .^ 2) / W
    Mxy = sum(w .* dx .* dy) / W
    λs    = eigen(Symmetric([Mxx Mxy; Mxy Myy])).values   # ascending
    major = sqrt(λs[2])
    minor = sqrt(max(λs[1], 0.0))
    θ     = 0.5 * atan(2Mxy, Mxx - Myy)   # major-axis angle from +x toward +y (image coords)
    return (; xc, yc, major, minor, ellipticity = 1 - minor / major, θ)
end

###############################################################
# SHAPE: ISOPHOTE ELLIPSE FITS
###############################################################

# Direct least-squares ellipse fit (Halir & Flusser) to the points (x, y).
# Returns (; cx, cy, a, b, θ) with a >= b and θ the major-axis angle from +x
# toward +y (image coords, wrapped to (-π/2, π/2]), or nothing if no valid
# ellipse exists.
function fit_ellipse_direct(x, y)
    n = length(x)
    n < 6 && return nothing
    mx, my = mean(x), mean(y)
    s = max(std(x), std(y))
    s > 0 || return nothing
    xs = (x .- mx) ./ s
    ys = (y .- my) ./ s

    D1 = hcat(xs .^ 2, xs .* ys, ys .^ 2)
    D2 = hcat(xs, ys, ones(n))
    S1 = D1' * D1
    S2 = D1' * D2
    S3 = D2' * D2
    T  = -(S3 \ S2')
    Mt = S1 + S2 * T
    Mm = vcat(Mt[3:3, :] ./ 2, -Mt[2:2, :], Mt[1:1, :] ./ 2)

    ev = try
        eigen(Mm)
    catch
        return nothing
    end

    a1 = nothing
    for k in 1:3
        vc = ev.vectors[:, k]
        maximum(abs.(imag.(vc))) > 1e-8 && continue
        v = real.(vc)
        if 4v[1] * v[3] - v[2]^2 > 0          # ellipse constraint
            a1 = v
            break
        end
    end
    a1 === nothing && return nothing
    a2 = T * a1
    A, B, C = a1
    D, E, F = a2

    Mc = [2A B; B 2C]
    abs(det(Mc)) < 1e-14 && return nothing
    c  = Mc \ [-D, -E]
    F0 = F + (D * c[1] + E * c[2]) / 2
    Q  = [A B/2; B/2 C]
    ee = eigen(Symmetric(Q))
    λ  = ee.values
    (-F0 / λ[1] > 0 && -F0 / λ[2] > 0) || return nothing

    r    = sqrt.(-F0 ./ λ)
    imaj = argmax(r)
    vmaj = ee.vectors[:, imaj]
    θ    = mod(atan(vmaj[2], vmaj[1]) + π / 2, π) - π / 2
    return (; cx = mx + s * c[1], cy = my + s * c[2],
              a = s * r[imaj], b = s * r[3-imaj], θ)
end

# Ellipse fit to the OUTER BOUNDARY of the nucleus-connected region where the
# smoothed image `sm` exceeds `level`.  Interior holes are filled; boundary
# pixels lying on the image edge (frame clipping) are excluded; arm
# protrusions are rejected by iterative residual clipping.
# (xk, yk) is the nucleus position in the coordinates of `sm`.
function isophote_ellipse(sm, level, xk, yk; margin = 2, max_pts = 20000, min_pts = 200)
    labels = label_components(sm .> level)
    lab = labels[yk, xk]
    lab == 0 && return nothing
    mask = labels .== lab
    Yd, Xd = size(mask)

    # fill holes: background components that do not touch the array border
    bg   = label_components(.!mask)
    edge = Set{Int}()
    for i in 1:Xd
        push!(edge, bg[1, i]); push!(edge, bg[Yd, i])
    end
    for j in 1:Yd
        push!(edge, bg[j, 1]); push!(edge, bg[j, Xd])
    end
    delete!(edge, 0)
    filled = .!(in.(bg, Ref(edge)))

    # boundary pixels = filled pixels with at least one 4-neighbour outside
    P = falses(Yd + 2, Xd + 2)
    P[2:end-1, 2:end-1] .= filled
    core     = P[2:end-1, 2:end-1]
    interior = core .& P[1:end-2, 2:end-1] .& P[3:end, 2:end-1] .&
               P[2:end-1, 1:end-2] .& P[2:end-1, 3:end]
    pts = findall(core .& .!interior)
    pts = filter(p -> margin < p[1] <= Yd - margin && margin < p[2] <= Xd - margin, pts)
    length(pts) < min_pts && return nothing
    step = cld(length(pts), max_pts)
    pts  = pts[1:step:end]

    xp = Float64[p[2] for p in pts]
    yp = Float64[p[1] for p in pts]

    keep = trues(length(xp))
    fit  = nothing
    for _ in 1:4
        f = fit_ellipse_direct(xp[keep], yp[keep])
        f === nothing && break
        fit = f
        c, s = cos(f.θ), sin(f.θ)
        u = (xp .- f.cx) .* c .+ (yp .- f.cy) .* s
        v = -(xp .- f.cx) .* s .+ (yp .- f.cy) .* c
        r = sqrt.((u ./ f.a) .^ 2 .+ (v ./ f.b) .^ 2) .- 1
        tol = max(2.5 * 1.4826 * median(abs.(r)), 0.02)
        newkeep = abs.(r) .<= tol
        count(newkeep) < min_pts && break
        keep = newkeep
    end
    fit === nothing && return nothing

    q = fit.b / fit.a
    (0.2 <= q <= 1.0 && fit.a < 3 * max(Xd, Yd)) || return nothing
    return (; ellipticity = 1 - q, θ = fit.θ, a = fit.a, b = fit.b,
              cx = fit.cx, cy = fit.cy, npts = count(keep))
end

# Isophote levels: factors x threshold, never below the noise floor. Levels
# within `merge_tol` (fractional) of each other are merged, so every level is a
# genuinely different isophote. If fewer than `min_levels` remain (threshold at
# or near the floor), levels are extended upward to 1.0-1.5 x the floor.
function iso_levels(thr, floor_t, factors; merge_tol = 0.05, min_levels = 3,
                    extend = (1.0, 1.15, 1.3, 1.5))
    function merged(v)
        out = Float64[]
        for x in sort(v)
            (isempty(out) || x > out[end] * (1 + merge_tol)) && push!(out, x)
        end
        return out
    end
    raw    = max.(Float64(floor_t), Float64(thr) .* collect(Float64, factors))
    levels = merged(raw)
    if length(levels) < min_levels
        levels = merged(vcat(raw, Float64(floor_t) .* collect(Float64, extend)))
    end
    return round.(levels; digits = 5)
end

# Disk ellipse from isophotes at several brightness levels around the mask
# threshold `thr` (levels = factors x thr, never below the noise floor).
# `sm_ds` is the smoothed image sampled every `ds` pixels.  Combined result:
# median ellipticity, circular-median position angle, plus the scatter between
# isophotes.  Returns nothing if fewer than 2 isophotes could be fitted.
function disk_from_isophotes(sm_ds, ds, xn, yn, thr, floor_t; factors)
    xk = round(Int, (xn - 1) / ds) + 1
    yk = round(Int, (yn - 1) / ds) + 1
    Yd, Xd = size(sm_ds)
    (1 <= xk <= Xd && 1 <= yk <= Yd) || return nothing

    levels = iso_levels(thr, floor_t, factors)
    fits = NamedTuple[]
    for L in levels
        f = isophote_ellipse(sm_ds, Float32(L), xk, yk)
        f === nothing && continue
        push!(fits, merge((; level = L), f))
    end
    length(fits) < 2 && return nothing

    εs = [f.ellipticity for f in fits]
    θs = [f.θ for f in fits]
    θ  = deg2rad(pa_median_deg(rad2deg.(θs)))
    return (; ellipticity = median(εs), θ, fits, nvalid = length(fits), ds,
              spread_ell = maximum(εs) - minimum(εs),
              spread_pa  = maximum(angdiff(rad2deg(a), rad2deg(b)) for a in θs for b in θs))
end

###############################################################
# RADII / BAR HELPERS
###############################################################

# Deprojected radii enclosing 20 % / 80 % of the mask flux
function flux_radii(gwork, inds, xn, yn, q, θd)
    Y, X = size(gwork)
    cθ, sθ = cos(θd), sin(θd)
    enc = zeros(ceil(Int, hypot(Y, X) / q) + 2)
    for I in inds
        j, i = I[1], I[2]
        dx = i - xn
        dy = j - yn
        u  =  dx * cθ + dy * sθ
        v  = -dx * sθ + dy * cθ
        enc[floor(Int, hypot(u, v / q)) + 1] += max(gwork[j, i], 0)
    end
    cum = cumsum(enc)
    return max(findfirst(>=(0.20 * cum[end]), cum), 1),
           max(findfirst(>=(0.80 * cum[end]), cum), 1)
end

# Fixed concentration aperture: semi-major axis of the lowest valid isophote at or
# above the noise floor (levels tried: 1, 1.5, 2, 3, 4 x floor). It does not depend
# on the mask threshold, so concentrations measured inside it are comparable
# between thresholds.
function fixed_aperture(sm_ds, ds, xn, yn, floor_t; multipliers = (1.0, 1.5, 2.0, 3.0, 4.0),
                        g = nothing, min_cov = 0.0)
    xk = round(Int, (xn - 1) / ds) + 1
    yk = round(Int, (yn - 1) / ds) + 1
    Yd, Xd = size(sm_ds)
    (1 <= xk <= Xd && 1 <= yk <= Yd) || return nothing
    first_valid = nothing
    for m in multipliers
        L = Float64(floor_t) * m
        f = isophote_ellipse(sm_ds, Float32(L), xk, yk)
        f === nothing && continue
        cand = (; level = L, a = f.a * ds, ellipticity = f.ellipticity, mult = m,
                  coverage = NaN, fallback = false)
        if g === nothing                       # no coverage test requested
            return cand
        end
        # fraction of the aperture (the isophote's own ellipse) inside the frame
        c = aperture_concentration(g, xn, yn, clamp(1 - f.ellipticity, 0.2, 1.0), f.θ, f.a * ds; ds = ds)
        cand = merge(cand, (; coverage = c.coverage))
        first_valid === nothing && (first_valid = cand)
        cand.coverage >= min_cov && return cand
    end
    # no aperture met the coverage requirement: use the first valid one (flagged)
    first_valid === nothing ? nothing : merge(first_valid, (; fallback = true))
end

# r20 / r80 / concentration from the flux inside the deprojected elliptical
# aperture r <= a_ap. Every pixel counts (not only the mask). `ds` > 1 samples
# every ds-th pixel. `coverage` is the fraction of the aperture area inside the frame.
function aperture_concentration(g, xn, yn, q, θd, a_ap; ds = 1)
    Y, X = size(g)
    cθ, sθ = cos(θd), sin(θd)
    enc = zeros(ceil(Int, a_ap / ds) + 1)
    npx = 0
    @inbounds for j in 1:ds:Y, i in 1:ds:X
        dx = i - xn
        dy = j - yn
        u  =  dx * cθ + dy * sθ
        v  = -dx * sθ + dy * cθ
        r  = hypot(u, v / q)
        r >= a_ap && continue
        enc[floor(Int, r / ds) + 1] += g[j, i]
        npx += 1
    end
    cum = cumsum(enc)
    tot = cum[end]
    tot > 0 || return (; r20 = 0, r80 = 0, conc = NaN, coverage = NaN)
    r20 = max(findfirst(>=(0.20 * tot), cum) * ds, 1)      # upper bin edges (px)
    r80 = max(findfirst(>=(0.80 * tot), cum) * ds, 1)
    return (; r20, r80, conc = 5 * log10(r80 / r20),
              coverage = npx * ds^2 / (π * q * a_ap^2))
end

# Concentration versus aperture size: the aperture is set by the isophote at each
# of `multipliers` x the noise floor (higher multiplier = smaller aperture). The
# spread is the real uncertainty of a fixed-aperture concentration.
function aperture_sensitivity(g, sm_ds, ds, xn, yn, q, θd, floor_t; multipliers = (1.0, 1.5, 2.0, 3.0))
    out = NamedTuple[]
    for m in multipliers
        ap = fixed_aperture(sm_ds, ds, xn, yn, floor_t; multipliers = (m,))
        ap === nothing && continue
        c = aperture_concentration(g, xn, yn, q, θd, ap.a; ds = ds)
        isnan(c.conc) && continue
        push!(out, (; mult = m, level = ap.level, a = ap.a, conc = c.conc, coverage = c.coverage))
    end
    return out
end

function print_aperture_sensitivity(sens)
    println("Concentration versus aperture size (higher multiplier = smaller aperture):")
    for s in sens
        @printf("  %.1f x floor (level %.4f): a = %5.0f px, concentration = %.2f, coverage = %.2f%s\n",
                s.mult, s.level, s.a, s.conc, s.coverage,
                s.coverage < APERTURE_MIN_COVERAGE ? "  (partly outside the frame: biased, excluded from the range)" : "")
    end
end

# Is the A2 peak a stable feature of the sweep? Rows whose peak is pinned at the
# search limit (with a higher A2 beyond it) are unreliable; of the others, count
# those peaking within `tol` of the median radius.
function a2_stability(rs; tol = 0.15)
    isempty(rs) && return (; frac = NaN, n_limit = 0, med_radius = NaN)
    free = [r for r in rs if !r.A2_at_limit]
    n_limit = length(rs) - length(free)
    isempty(free) && return (; frac = 0.0, n_limit, med_radius = NaN)
    med = median([Float64(r.A2_radius) for r in free])
    nstable = count(r -> abs(r.A2_radius - med) <= tol * med, free)
    return (; frac = nstable / length(rs), n_limit, med_radius = med)
end

# m = 2 Fourier amplitude A2(r) and phase in deprojected annuli.
# Uses ALL pixels (not just the mask) and only annuli lying fully inside the
# image.  The peak is searched only for r <= rsearch (inner disk); the
# unrestricted maximum is returned too for comparison.
function a2_profile(gwork, xn, yn, q, θd; dr = 10, rmax = 3000, rsearch = Inf)
    Y, X = size(gwork)
    cθ, sθ = cos(θd), sin(θd)

    # half-extents of the r = 1 deprojected ellipse along x and y
    hx = sqrt(cθ^2 + (q * sθ)^2)
    hy = sqrt(sθ^2 + (q * cθ)^2)
    rcomp = min((xn - 1) / hx, (X - xn) / hx, (yn - 1) / hy, (Y - yn) / hy)

    nbar = floor(Int, min(rcomp, Float64(rmax)) / dr)
    nbar >= 2 || error("Nucleus too close to the image edge for an A2 profile.")
    rlim = nbar * dr

    re  = zeros(nbar)
    imv = zeros(nbar)
    si  = zeros(nbar)
    cnt = zeros(Int, nbar)

    x1 = max(1, floor(Int, xn - hx * rlim)); x2 = min(X, ceil(Int, xn + hx * rlim))
    y1 = max(1, floor(Int, yn - hy * rlim)); y2 = min(Y, ceil(Int, yn + hy * rlim))

    @inbounds for j in y1:y2, i in x1:x2
        dx = i - xn
        dy = j - yn
        u  =  dx * cθ + dy * sθ
        v  = -dx * sθ + dy * cθ
        r  = hypot(u, v / q)
        r >= rlim && continue
        b  = floor(Int, r / dr) + 1
        φ  = atan(v / q, u)
        w  = max(gwork[j, i], 0)
        re[b]  += w * cos(2φ)
        imv[b] += w * sin(2φ)
        si[b]  += w
        cnt[b] += 1
    end

    A2      = sqrt.(re .^ 2 .+ imv .^ 2) ./ max.(si, eps())
    phase   = 0.5 .* atan.(imv, re)
    centers = ((1:nbar) .- 0.5) .* dr
    ok      = cnt .>= 100
    any(ok) || error("Too few pixels for the A2 profile.")

    ifull = argmax(ifelse.(ok, A2, -Inf))
    inner = ok .& (centers .<= rsearch)
    ibar  = any(inner) ? argmax(ifelse.(inner, A2, -Inf)) : ifull

    return (; A2, phase, dr, nbar, rlim, centers, ibar,
              strength = A2[ibar], radius = centers[ibar],
              strength_full = A2[ifull], radius_full = centers[ifull])
end

###############################################################
# MAIN ANALYSIS
###############################################################

"""
    analyze_galaxy(filename; outdir, prefix, nsigma, min_rise, max_floor,
                   star_level, star_halo, star_min_radius, star_max_radius,
                   nucleus_exclude, nucleus_smooth, center, edge_warn_frac,
                   iso_factors, iso_downsample, log_sigma, bar_rmax,
                   bar_search_frac, bar_align_tol)

Keywords
- `prefix`: prepended to every output file (default: image name, e.g. `NGC_1300_`)
- `nsigma`, `min_rise`, `max_floor`: mask threshold above the flattened smoothed
  background = max(floor, min_rise), where floor = min(nsigma*σ, max_floor)
- `star_level`: gray level (0-1) above which pixels seed the star mask
- `star_halo`: masked radius = `star_halo` x core radius of each saturated blob,
  clamped to [`star_min_radius`, `star_max_radius`] pixels
- `nucleus_exclude`: radius (px) around the nucleus where stars are NOT masked
- `nucleus_smooth`: Gaussian σ (px) used to locate the nucleus on the star-free image
- `center`: optional `(x, y)` pixel position of the nucleus (overrides the finder)
- `edge_warn_frac`: warn if the nucleus lies within this fraction of an image edge
- `iso_factors`: isophote levels as multiples of the mask threshold; an ellipse is
  fitted to the outer boundary at each level and the results are combined
  (median ellipticity, circular-median PA). Falls back to area moments if < 2 fits work.
  The default levels (0.4-1.0 x threshold) lie at or outside the mask threshold, where
  the disk dominates; higher levels tend to be bar- or bulge-dominated.
- `iso_downsample`: the isophote fits run on every n-th pixel of the smoothed image
- `iso_warn_ell`, `iso_warn_pa`: warn if the scatter between isophotes exceeds these
  (ellipticity, degrees)
- `iso_trend_ell`, `iso_trend_pa`: warn if ellipticity / PA change systematically from
  the innermost to the outermost isophote by more than these (ellipticity, degrees)
- `log_sigma`: scale (px) of the Laplacian-of-Gaussian image
- `bar_rmax`: maximum deprojected radius (px) of the A2 profile
- `bar_search_frac`: the bar peak is searched only for r <= `bar_search_frac` * r80
- `bar_align_tol`: warn if the bar direction is within this many degrees of the
  disk major/minor axis (a sign of imperfect deprojection, not a bar)
"""
function analyze_galaxy(filename = "galaxy.jpg";
                        outdir = @__DIR__,
                        prefix = nothing,
                        nsigma = 5,
                        min_rise = 0.15f0,
                        max_floor = 0.15,
                        star_level = 0.99f0,
                        star_halo = 8,
                        star_min_radius = 8,
                        star_max_radius = 150,
                        nucleus_exclude = 200,
                        nucleus_smooth = 60,
                        center = nothing,
                        edge_warn_frac = 0.15,
                        iso_factors = (0.4, 0.55, 0.7, 0.85, 1.0),
                        iso_downsample = 4,
                        iso_warn_ell = 0.1,
                        iso_warn_pa = 10.0,
                        iso_trend_ell = 0.04,
                        iso_trend_pa = 8.0,
                        log_sigma = 20,
                        bar_rmax = 3000,
                        bar_search_frac = 0.7,
                        bar_align_tol = 3.0)

    infile = isfile(filename) ? filename : joinpath(@__DIR__, filename)
    isfile(infile) || error("Image not found: $filename")
    mkpath(outdir)
    pre = output_prefix(infile, prefix)
    out(name) = joinpath(outdir, pre * name)

    println()
    println("==================================================")
    println(" GALAXY ANALYSIS")
    println("==================================================")
    println("Input:  ", infile)
    println("Output: ", outdir, "  (prefix \"", pre, "\")")
    println()

    ###########################################################
    # LOAD, STARS, NUCLEUS, BACKGROUND
    ###########################################################

    logmsg("Loading image, masking stars, locating nucleus, flattening background...")

    (; gwork, starmask, xn, yn, Y, X, σsky, bg) =
        prepare_image(infile; star_level, star_halo, star_min_radius,
                      star_max_radius, nucleus_exclude, nucleus_smooth, center)

    println("Image dimensions (rows, cols) = ", (Y, X))
    println("Nucleus (x, y) = ", (xn, yn), center === nothing ? "  (found)" : "  (user-supplied)")
    check_nucleus(xn, yn, X, Y, edge_warn_frac)
    println("Star-masked pixels = ", count(starmask),
            " (", round(100 * count(starmask) / length(starmask), digits = 2), " %)")
    println("Background plane: level at centre = ", round(bg.mean, digits = 4),
            ", change across width = ", round(bg.grad_x, digits = 4),
            ", across height = ", round(bg.grad_y, digits = 4))
    println("Background noise σ (dark half of border) = ", σsky)

    save(out("starmask.png"), Gray.(Float32.(starmask)))

    ###########################################################
    # MASK (smoothed threshold, nucleus-connected component only)
    ###########################################################

    logmsg("Creating galaxy mask...")

    smooth6   = gblur(gwork, 6)
    _, σ_s    = fit_plane_background(smooth6)
    floor_raw = nsigma * σ_s
    floor_t   = min(floor_raw, Float32(max_floor))
    threshold = max(floor_t, Float32(min_rise))
    println("Smoothed-image noise σ = ", σ_s, "  (", nsigma, "σ = ", floor_raw, ")")
    floor_raw > max_floor &&
        println("NOTE: noise floor capped at ", max_floor, ".")
    println("Threshold (above background) = ", threshold, "  (min_rise = ", min_rise, ")")
    floor_t > min_rise &&
        println("WARNING: the noise floor (", floor_t, ") exceeds min_rise; ",
                "the threshold is set by the floor, not by min_rise.")

    labels = label_components(smooth6 .> threshold)
    lab    = labels[yn, xn]
    lab == 0 && error("Nucleus is below threshold; lower `min_rise`/`nsigma`.")
    mask = labels .== lab

    galaxy_pixels = count(mask)
    frac = galaxy_pixels / length(mask)
    println("Galaxy pixels = ", galaxy_pixels, " (", round(100frac, digits = 1), " % of image)")
    frac > 0.6 && println("WARNING: mask covers >60 % of the image; raise `min_rise` or `nsigma`.")
    frac < 0.01 &&
        println("WARNING: mask covers <1 % of the image; the 'nucleus' may be a star or ",
                "other compact object (use `center = (x, y)`), or the threshold is too high.")

    save(out("mask.png"), Gray.(Float32.(mask)))

    ###########################################################
    # FLUX
    ###########################################################

    logmsg("Computing integrated flux...")

    flux = sum(max.(gwork[mask], 0))
    println("Integrated Flux (gray-level units) = ", flux)

    ###########################################################
    # DISK ELLIPSE (isophote fits; area moments for comparison / fallback)
    ###########################################################

    logmsg("Fitting disk ellipse...")

    inds = findall(mask)
    xs   = Float64[I[2] for I in inds]
    ys   = Float64[I[1] for I in inds]
    disk_mom = ellipse_moments(xs, ys, ones(length(xs)))

    sm_ds = smooth6[1:iso_downsample:end, 1:iso_downsample:end]
    iso   = disk_from_isophotes(sm_ds, iso_downsample, xn, yn, threshold, floor_t;
                                factors = iso_factors)

    ell_trend    = NaN
    pa_trend     = NaN
    radial_trend = false
    ell_iso_min = ell_iso_max = ell_outer = pa_outer = NaN

    if iso === nothing
        println("NOTE: isophote fits failed; using area moments for the disk ellipse.")
        disk_method = "moments"
        ellipticity = disk_mom.ellipticity
        θd          = disk_mom.θ
        iso_spread_ell = NaN
        iso_spread_pa  = NaN
    else
        disk_method = "isophote"
        ellipticity = iso.ellipticity
        θd          = iso.θ
        iso_spread_ell = iso.spread_ell
        iso_spread_pa  = iso.spread_pa
        println("Isophote fits, inner to outer (ellipticity and PA versus semi-major axis):")
        fs = sort(iso.fits; by = f -> f.a)
        for f in fs
            @printf("  a = %5.0f px (level %.3f): ε = %.3f, PA = %6.1f°, boundary points = %d\n",
                    f.a * iso_downsample, f.level, f.ellipticity, rad2deg(f.θ), f.npts)
        end
        ell_iso_min = minimum(f.ellipticity for f in fs)
        ell_iso_max = maximum(f.ellipticity for f in fs)
        ell_outer   = fs[end].ellipticity
        pa_outer    = rad2deg(fs[end].θ)
        if length(fs) >= 2
            ell_trend = fs[end].ellipticity - fs[1].ellipticity            # outer minus inner
            pa_trend  = mod(rad2deg(fs[end].θ) - rad2deg(fs[1].θ) + 90, 180) - 90
            radial_trend = length(fs) >= 3 &&
                           (abs(ell_trend) > iso_trend_ell || abs(pa_trend) > iso_trend_pa)
            @printf("Trend outward (a = %.0f -> %.0f px): Δε = %+.3f, ΔPA = %+.1f°\n",
                    fs[1].a * iso_downsample, fs[end].a * iso_downsample, ell_trend, pa_trend)
            radial_trend &&
                println("WARNING: ellipticity/PA change systematically with radius, so a single ",
                        "disk ellipse is only approximate; quote a range (see the table above).")
            @printf("Isophote ellipticity range: %.3f-%.3f (inclination %.0f°-%.0f°); outermost isophote: ε = %.3f, PA = %.1f°, inclination %.0f°\n",
                    ell_iso_min, ell_iso_max, acosd(1 - ell_iso_min), acosd(1 - ell_iso_max),
                    ell_outer, pa_outer, acosd(1 - ell_outer))
        end
        println("Scatter between isophotes: Δε = ", round(iso_spread_ell, digits = 3),
                ", ΔPA = ", round(iso_spread_pa, digits = 1), "°")
        (iso_spread_ell > iso_warn_ell || iso_spread_pa > iso_warn_pa) &&
            println("WARNING: the isophotes disagree (Δε = ", round(iso_spread_ell, digits = 3),
                    ", ΔPA = ", round(iso_spread_pa, digits = 1), "°); the inner levels are ",
                    "probably bar- or core-dominated and the disk ellipse is uncertain. ",
                    "Consider a lower `min_rise` or lower `iso_factors`.")
    end

    q = clamp(1 - ellipticity, 0.2, 1.0)          # axis ratio b/a
    inclination = acosd(q)                        # thin-disk approximation

    println("Disk method      = ", disk_method)
    println("Disk ellipticity = ", ellipticity,
            "   (area moments, for comparison: ", round(disk_mom.ellipticity, digits = 3), ")")
    println("Disk PA (deg, image coords) = ", rad2deg(θd),
            "   (area moments: ", round(rad2deg(disk_mom.θ), digits = 1), ")")
    println("Inclination (thin disk) = ", inclination, " deg")

    ###########################################################
    # DEPROJECTED RADIUS MAP (about the nucleus)
    ###########################################################

    logmsg("Building deprojected radius map...")

    cθ, sθ = cos(θd), sin(θd)
    rdep = Array{Float32}(undef, Y, X)
    @inbounds for j in 1:Y, i in 1:X
        dx = i - xn
        dy = j - yn
        u  =  dx * cθ + dy * sθ        # along disk major axis
        v  = -dx * sθ + dy * cθ        # along disk minor axis
        rdep[j, i] = hypot(u, v / q)
    end
    nb = floor(Int, maximum(rdep)) + 1

    ###########################################################
    # RADIAL PROFILE + CONCENTRATION (deprojected)
    ###########################################################

    logmsg("Computing radial profile and concentration...")

    prof_sum = zeros(Float64, nb)
    prof_n   = zeros(Int, nb)
    enc_sum  = zeros(Float64, nb)      # galaxy-mask flux per bin

    @inbounds for j in 1:Y, i in 1:X
        b = floor(Int, rdep[j, i]) + 1
        f = gwork[j, i]
        prof_sum[b] += f
        prof_n[b]   += 1
        if mask[j, i]
            enc_sum[b] += max(f, 0)
        end
    end

    radial_profile = prof_sum ./ max.(prof_n, 1)

    p2 = plot(0:nb-1, radial_profile, lw = 2, legend = false,
              xlabel = "Deprojected radius (pixels)", ylabel = "Brightness above background",
              title = "Galaxy Radial Profile")
    savefig(p2, out("radial_profile.png"))

    cum   = cumsum(enc_sum)
    idx20 = findfirst(>=(0.20 * cum[end]), cum)
    idx80 = findfirst(>=(0.80 * cum[end]), cum)
    r20   = max(idx20, 1)
    r80   = max(idx80, 1)
    concentration = 5 * log10(r80 / r20)

    println("r20 = ", r20, " px")
    println("r80 = ", r80, " px")
    println("Concentration (mask-based, depends on the threshold) = ", concentration)

    # fixed-aperture concentration: independent of the mask threshold
    ap = fixed_aperture(sm_ds, iso_downsample, xn, yn, floor_t;
                        g = gwork, min_cov = APERTURE_MIN_COVERAGE)
    if ap === nothing
        println("NOTE: no valid fixed-aperture isophote; fixed-aperture concentration unavailable.")
        concentration_ap, r20_ap, r80_ap, ap_cov, ap_a = NaN, 0, 0, NaN, NaN
    else
        capr = aperture_concentration(gwork, xn, yn, q, θd, ap.a)
        concentration_ap, r20_ap, r80_ap, ap_cov, ap_a = capr.conc, capr.r20, capr.r80, capr.coverage, ap.a
        println("Fixed aperture: semi-major axis = ", round(Int, ap_a), " px (isophote at level ",
                round(ap.level, digits = 4), " = ", ap.mult, " x floor)")
        ap.mult != 1.0 && !ap.fallback &&
            println("  (smaller apertures were skipped because they extended beyond the frame; ",
                    "coverage must be >= ", APERTURE_MIN_COVERAGE, ")")
        println("  r20 = ", r20_ap, " px, r80 = ", r80_ap, " px, concentration (fixed aperture) = ",
                concentration_ap)
        println("  fraction of the aperture inside the frame = ", round(ap_cov, digits = 2))
        ap_cov < 0.9 &&
            println("WARNING: the aperture extends beyond the frame, so the flux inside it is ",
                    "incomplete and the fixed-aperture concentration is biased.")
    end
    conc_ap_min, conc_ap_max = NaN, NaN
    if ap !== nothing
        sens = aperture_sensitivity(gwork, sm_ds, iso_downsample, xn, yn, q, θd, floor_t)
        if !isempty(sens)
            print_aperture_sensitivity(sens)
            good = filter(c -> c.coverage >= APERTURE_MIN_COVERAGE, sens)
            isempty(good) && (good = sens)
            conc_ap_min = minimum(c.conc for c in good)
            conc_ap_max = maximum(c.conc for c in good)
            @printf("Concentration range over apertures: %.2f-%.2f\n", conc_ap_min, conc_ap_max)
        end
    end

    ###########################################################
    # BAR STRENGTH: m=2 Fourier amplitude A2(r) (deprojected)
    ###########################################################

    logmsg("Computing m=2 Fourier bar amplitude...")

    rsearch = bar_search_frac * r80
    bar = a2_profile(gwork, xn, yn, q, θd; rmax = bar_rmax, rsearch = rsearch)
    bar_strength = bar.strength
    bar_radius   = bar.radius

    # bar direction in the deprojected plane -> back to image coords
    φb = bar.phase[bar.ibar]
    ux, vy = cos(φb), q * sin(φb)
    bar_pa_image = rad2deg(atan(ux * sθ + vy * cθ, ux * cθ - vy * sθ))

    # a peak aligned with the disk major/minor axis suggests imperfect deprojection
    axis_dev    = abs(mod(rad2deg(φb) + 45, 90) - 45)
    bar_aligned = axis_dev < bar_align_tol

    println("A2 measured out to r = ", bar.rlim, " px (full annuli only)")
    bar.rlim < r80 &&
        println("WARNING: the A2 profile ends (", bar.rlim, " px) before r80 (", r80,
                " px): annuli run off the frame, so the galaxy is clipped by the image ",
                "edge or the nucleus is off-centre.")
    println("Bar search limited to r <= ", round(rsearch), " px (", bar_search_frac, " x r80)")
    println("Bar strength A2 (max in search range) = ", bar_strength)
    println("Radius of A2 max                      = ", bar_radius, " px")
    println("Bar PA (deg, image coords)            = ", bar_pa_image)
    if bar.radius_full != bar.radius
        println("(Unrestricted A2 maximum: ", round(bar.strength_full, digits = 3),
                " at r = ", bar.radius_full, " px)")
    end
    bar_at_limit = bar.radius >= 0.95 * rsearch && bar.radius_full != bar.radius
    bar_at_limit &&
        println("WARNING: the A2 peak sits at the search limit (", bar_radius, " of ", round(rsearch),
                " px) and A2 is higher beyond it, so the peak is not a local maximum: probably ",
                "arms or a ring rather than a bar.")
    bar_aligned &&
        println("WARNING: bar direction is within ", round(axis_dev, digits = 1),
                "° of the disk major/minor axis; the A2 peak may reflect imperfect ",
                "deprojection or disk/arm structure rather than a bar.")

    p3 = plot(bar.centers, bar.A2, lw = 2, legend = false,
              xlabel = "Deprojected radius (pixels)", ylabel = "A2",
              title = "m = 2 Fourier amplitude")
    vline!(p3, [rsearch]; ls = :dash, color = :gray)
    savefig(p3, out("a2_profile.png"))

    ###########################################################
    # DIFFERENCE OF GAUSSIANS
    ###########################################################

    logmsg("Running Difference of Gaussians...")

    dog = gblur(gwork, 2) .- gblur(gwork, 8)
    save(out("dog.png"), Gray.(clamp01.(dog .+ 0.5f0)))
    println("Saved: ", pre, "dog.png")

    ###########################################################
    # LAPLACIAN OF GAUSSIAN (Gaussian smooth, then 3x3 Laplacian)
    ###########################################################

    try
        logmsg("Running Laplacian of Gaussian (σ = $log_sigma px)...")

        logimg = abs.(imfilter(gblur(gwork, log_sigma), Kernel.Laplacian()))
        save(out("log.png"), Gray.(normalize01(logimg)))
        println("Saved: ", pre, "log.png")
    catch err
        println("LoG filter skipped:")
        println(err)
    end

    ###########################################################
    # STRUCTURE TENSOR
    ###########################################################

    logmsg("Computing structure tensor...")

    gy, gx = imgradients(gwork, KernelFactors.sobel)
    Jxx = gblur(gx .^ 2, 5)
    Jyy = gblur(gy .^ 2, 5)
    Jxy = gblur(gx .* gy, 5)
    coherence = sqrt.((Jxx .- Jyy) .^ 2 .+ 4 .* Jxy .^ 2) ./ (Jxx .+ Jyy .+ 1f-6)
    save(out("coherence.png"), Gray.(clamp01.(coherence)))
    println("Saved: ", pre, "coherence.png")

    ###########################################################
    # SPIRAL ARM RESIDUAL
    ###########################################################

    logmsg("Computing spiral arm residual...")

    smooth20     = gblur(gwork, 20)
    residual     = gwork .- smooth20
    arm_strength = std(residual[mask])
    arm_contrast = arm_strength / mean(smooth20[mask])   # dimensionless
    println("Arm Strength (std residual)     = ", arm_strength)
    println("Arm Contrast (relative to disk) = ", arm_contrast)

    save(out("residual.png"), Gray.(normalize01(residual)))
    println("Saved: ", pre, "residual.png")

    ###########################################################
    # SAVE SUMMARY REPORT
    ###########################################################

    logmsg("Writing report...")

    open(out("analysis_report.txt"), "w") do io
        println(io, "GALAXY ANALYSIS REPORT")
        println(io)
        println(io, "Image Size = $(Y)x$(X)")
        println(io, "Nucleus (x,y) = ($xn, $yn)")
        println(io, "Background plane level at centre = $(bg.mean), change across width = $(bg.grad_x), across height = $(bg.grad_y)")
        println(io, "Background noise sigma = $σsky; smoothed-image sigma = $σ_s")
        println(io, "min_rise = $min_rise, threshold above background = $threshold")
        println(io, "Galaxy Pixels = $galaxy_pixels")
        println(io, "Flux = $flux")
        println(io, "Disk method = $disk_method")
        println(io, "Disk ellipticity = $ellipticity (area moments: $(disk_mom.ellipticity))")
        println(io, "Disk PA (deg) = $(rad2deg(θd)) (area moments: $(rad2deg(disk_mom.θ)))")
        println(io, "Scatter between isophotes: d_ellipticity = $iso_spread_ell, d_PA (deg) = $iso_spread_pa")
        println(io, "Trend outward (outer minus inner isophote): d_ellipticity = $ell_trend, d_PA (deg) = $pa_trend, systematic = $radial_trend")
        println(io, "Inclination (deg, thin disk) = $inclination")
        println(io, "r20 (deprojected px) = $r20")
        println(io, "r80 (deprojected px) = $r80")
        println(io, "Concentration (mask-based) = $concentration")
        println(io, "Concentration (fixed aperture a = $ap_a px) = $concentration_ap; r20 = $r20_ap, r80 = $r80_ap; aperture coverage = $ap_cov")
        println(io, "A2 profile reaches r = $(bar.rlim) px (r80 = $r80)")
        println(io, "Isophote ellipticity range = $ell_iso_min - $ell_iso_max; outermost isophote: ellipticity = $ell_outer, PA = $pa_outer")
        println(io, "Concentration range over apertures = $conc_ap_min - $conc_ap_max")
        println(io, "A2 peak at search limit = $bar_at_limit")
        println(io, "Bar search radius (px) = $(round(rsearch))")
        println(io, "Bar strength A2 = $bar_strength")
        println(io, "Bar radius at A2 max (px) = $bar_radius")
        println(io, "Bar PA (deg, image coords) = $bar_pa_image")
        println(io, "Bar aligned with disk axis (possible artifact) = $bar_aligned")
        println(io, "Arm Strength = $arm_strength")
        println(io, "Arm Contrast = $arm_contrast")
    end

    println("Saved: ", pre, "analysis_report.txt")
    logmsg("Done.")

    return (; nucleus = (xn, yn), σsky, threshold, galaxy_pixels, flux,
              disk_method, disk_ellipticity = ellipticity,
              disk_ellipticity_moments = disk_mom.ellipticity,
              disk_pa_deg = rad2deg(θd),
              iso_spread_ell, iso_spread_pa, ell_trend, pa_trend, radial_trend,
              ell_iso_min, ell_iso_max, ell_outer, pa_outer,
              conc_ap_min, conc_ap_max, bar_at_limit,
              inclination, r20, r80, concentration,
              concentration_ap, aperture_a = ap_a, r20_ap, r80_ap, aperture_coverage = ap_cov,
              bar_strength, bar_radius, bar_pa_image, bar_aligned,
              bar_search_radius = rsearch,
              arm_strength, arm_contrast)
end

###############################################################
# THRESHOLD SWEEP
###############################################################

"""
    sweep_threshold(filename; min_rises, ...)

Repeats the mask -> isophote ellipse -> deprojection -> A2 analysis for a range
of `min_rise` values and prints mask area fraction, ellipticity, PA, number of
isophote fits, r80 and the peak A2 (searched out to `bar_search_frac` x r80).

The noise floor is min(`nsigma`*σ, `max_floor`). `min_rise` values within
`floor_gap` of the floor give (nearly) the same mask; they are dropped and
replaced by a single row at the floor itself.

Plateau: a run of at least `min_plateau_points` consecutive thresholds where
ellipticity and PA change by less than `tol_ell` / `tol_pa` (deg) between
neighbours, the mask covers < `max_area` of the image, and the TOTAL spread
stays within `spread_ell` / `spread_pa` (deg). Candidate windows are ranked by
`length_weight` x (number of points) minus their normalised total spread (so
flatter windows win, with a mild preference for longer ones); all non-overlapping
plateaus are listed and the top-ranked one is chosen. Rows with fewer than 2
isophote fits (area-moment fallback) are excluded from the plateau test and the
summary. The suggested `min_rise` is the plateau point
closest to the plateau's median ellipticity and (circular-median) PA, using
interior points only when the plateau has 3 or more points.

Summary: median ellipticity and PA (with min-max range / half-range) over the
plateau rows if a plateau exists, otherwise over the rows whose mask area lies
in `range_area`. A wider spread over all rows with mask area in `wide_area` is
reported alongside as a more conservative uncertainty.

Returns `(; rows, suggested, plateau, summary)`; `suggested`/`summary` are
`nothing` when not available.
"""
function sweep_threshold(filename = "galaxy.jpg";
                         min_rises = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30],
                         outdir = @__DIR__,
                         prefix = nothing,
                         nsigma = 5,
                         max_floor = 0.15,
                         floor_gap = 0.02,
                         star_level = 0.99f0,
                         star_halo = 8,
                         star_min_radius = 8,
                         star_max_radius = 150,
                         nucleus_exclude = 200,
                         nucleus_smooth = 60,
                         center = nothing,
                         edge_warn_frac = 0.15,
                         iso_factors = (0.4, 0.55, 0.7, 0.85, 1.0),
                         iso_downsample = 4,
                         bar_rmax = 3000,
                         bar_search_frac = 0.7,
                         tol_ell = 0.03,
                         tol_pa = 3.0,
                         spread_ell = 0.02,
                         spread_pa = 3.0,
                         length_weight = 0.7,
                         min_plateau_points = 3,
                         max_area = 0.6,
                         range_area = (0.25, 0.55),
                         wide_area = (0.10, 0.60))

    infile = isfile(filename) ? filename : joinpath(@__DIR__, filename)
    isfile(infile) || error("Image not found: $filename")
    mkpath(outdir)
    pre = output_prefix(infile, prefix)

    println()
    println("==================================================")
    println(" THRESHOLD SWEEP")
    println("==================================================")

    logmsg("Preparing image (stars, nucleus, background)...")
    (; gwork, xn, yn, Y, X, σsky, bg) =
        prepare_image(infile; star_level, star_halo, star_min_radius,
                      star_max_radius, nucleus_exclude, nucleus_smooth, center)

    smooth6 = gblur(gwork, 6)
    sm_ds   = smooth6[1:iso_downsample:end, 1:iso_downsample:end]
    _, σ_s  = fit_plane_background(smooth6)
    floor_raw = Float64(nsigma * σ_s)
    floor_t   = min(floor_raw, Float64(max_floor))
    println("Nucleus (x, y) = ", (xn, yn), center === nothing ? "  (found)" : "  (user-supplied)")
    check_nucleus(xn, yn, X, Y, edge_warn_frac)
    println("Background noise σ = ", σsky, ", smoothed-image σ = ", σ_s)
    println("Noise floor = ", round(floor_t, digits = 4),
            floor_raw > max_floor ? "  (capped; $(nsigma)σ = $(round(floor_raw, digits = 4)))" :
                                    "  ($(nsigma)σ)")

    ap = fixed_aperture(sm_ds, iso_downsample, xn, yn, floor_t;
                        g = gwork, min_cov = APERTURE_MIN_COVERAGE)
    println(ap === nothing ?
            "Fixed concentration aperture: unavailable (no valid isophote); using mask-based concentration." :
            "Fixed concentration aperture: semi-major axis = $(round(Int, ap.a)) px (isophote at level $(round(ap.level, digits = 4)) = $(ap.mult) x floor" *
            (ap.fallback ? "; WARNING: every candidate aperture extends beyond the frame, coverage $(round(ap.coverage, digits = 2))" :
             ap.mult != 1.0 ? "; smaller apertures skipped, they extended beyond the frame" : "") * ")")

    # drop min_rise values hidden by (or within floor_gap of) the noise floor
    hidden = filter(<=(floor_t + floor_gap), min_rises)
    usable = sort(filter(>(floor_t + floor_gap), min_rises))
    if !isempty(hidden)
        println("Skipping min_rise ", hidden, " (within ", floor_gap,
                " of the floor, near-identical masks); the floor itself is the first row.")
        pushfirst!(usable, floor_t)
    end
    isempty(usable) && error("No usable min_rise values.")
    println()
    @printf("%9s %10s %9s %12s %10s %6s %8s %6s %7s %8s %10s %4s\n",
            "min_rise", "threshold", "area_frac", "ellipticity", "PA_deg", "n_iso", "r80_px", "conc", "conc_ap", "A2_max", "A2_rad_px", "lim")

    rows = NamedTuple[]

    for mr in usable
        thr    = max(floor_t, mr)
        labels = label_components(smooth6 .> thr)
        lab    = labels[yn, xn]
        if lab == 0
            println("Nucleus falls below threshold at min_rise = $mr; stopping sweep.")
            break
        end
        mask = labels .== lab
        inds = findall(mask)

        iso = disk_from_isophotes(sm_ds, iso_downsample, xn, yn, thr, floor_t;
                                  factors = iso_factors)
        if iso === nothing                      # fall back to area moments
            xs = Float64[I[2] for I in inds]
            ys = Float64[I[1] for I in inds]
            dm = ellipse_moments(xs, ys, ones(length(xs)))
            ell, θd, niso = dm.ellipticity, dm.θ, 0
        else
            ell, θd, niso = iso.ellipticity, iso.θ, iso.nvalid
        end
        q = clamp(1 - ell, 0.2, 1.0)

        r20, r80 = flux_radii(gwork, inds, xn, yn, q, θd)
        bar = a2_profile(gwork, xn, yn, q, θd; rmax = bar_rmax, rsearch = bar_search_frac * r80)
        capr = ap === nothing ? nothing :
               aperture_concentration(gwork, xn, yn, q, θd, ap.a; ds = iso_downsample)
        lvls = iso === nothing ? Float64[] : Float64[f.level for f in iso.fits]

        row = (; min_rise = Float64(mr), threshold = Float64(thr),
                 area_frac = length(inds) / (X * Y),
                 ellipticity = ell,
                 pa_deg = rad2deg(θd),
                 n_iso = niso,
                 r20 = r20,
                 r80 = r80,
                 conc = 5 * log10(r80 / r20),
                 conc_ap = capr === nothing ? NaN : capr.conc,
                 ap_cov = capr === nothing ? NaN : capr.coverage,
                 levels = lvls,
                 fits = iso === nothing ? NamedTuple[] :
                        [(; a = f.a * iso_downsample, ell = f.ellipticity, pa = rad2deg(f.θ)) for f in iso.fits],
                 A2 = bar.strength,
                 A2_radius = bar.radius,
                 A2_limit = bar_search_frac * r80,
                 A2_at_limit = bar.radius >= 0.95 * bar_search_frac * r80 && bar.radius_full != bar.radius)
        push!(rows, row)
        @printf("%9.3f %10.4f %9.3f %12.3f %10.2f %6d %8d %6.2f %7.2f %8.3f %10.0f %4s\n",
                row.min_rise, row.threshold, row.area_frac, row.ellipticity,
                row.pa_deg, row.n_iso, row.r80, row.conc, row.conc_ap, row.A2, row.A2_radius,
                row.A2_at_limit ? "*" : "")
    end
    any(r -> r.A2_at_limit, rows) &&
        println("  * = A2 peak pinned at the bar search limit (A2 is higher beyond it)")

    ###########################################################
    # PLATEAU DETECTION (ellipticity and PA only)
    ###########################################################

    # rows with < 2 isophote fits (area-moment fallback) are excluded from the
    # plateau test and the summary (all rows are restored before the output files)
    all_rows = rows
    rows     = filter(r -> r.n_iso >= 2, all_rows)
    nexcl    = length(all_rows) - length(rows)
    nexcl > 0 &&
        println("\nNote: ", nexcl, " row(s) with < 2 isophote fits are excluded from the ",
                "plateau test and summary.")

    n = length(rows)
    stepok = [abs(rows[k].ellipticity - rows[k+1].ellipticity) <= tol_ell &&
              angdiff(rows[k].pa_deg, rows[k+1].pa_deg) <= tol_pa for k in 1:n-1]
    areaok = [r.area_frac < max_area for r in rows]

    function window_spreads(s, e)
        es = [rows[k].ellipticity for k in s:e]
        ps = [rows[k].pa_deg for k in s:e]
        return maximum(es) - minimum(es), maximum(angdiff(a, b) for a in ps for b in ps)
    end

    # all valid windows, ranked by flatness with a mild preference for length
    candidates = NamedTuple[]
    for s in 1:n, e in (s + min_plateau_points - 1):n
        (all(stepok[s:e-1]) && all(areaok[s:e])) || continue
        de, dp = window_spreads(s, e)
        (de <= spread_ell && dp <= spread_pa) || continue
        push!(candidates, (; s, e, de, dp,
                             rank = length_weight * (e - s + 1) - (de / spread_ell + dp / spread_pa)))
    end
    sort!(candidates; by = c -> -c.rank)

    # non-overlapping plateaus, best first
    plateaus = NamedTuple[]
    for c in candidates
        any(!(c.e < p.s || c.s > p.e) for p in plateaus) && continue
        push!(plateaus, c)
    end
    best = isempty(plateaus) ? nothing : (plateaus[1].s, plateaus[1].e)

    println()
    if !isempty(plateaus)
        println("Plateaus (steps <= ", tol_ell, " / ", tol_pa, "°, total spread <= ", spread_ell,
                " / ", spread_pa, "°; ranked by flatness and length):")
        for p in sort(plateaus; by = p -> p.s)
            pr  = rows[p.s:p.e]
            es  = [r.ellipticity for r in pr]
            pas = [r.pa_deg for r in pr]
            pam = pa_median_deg(pas)
            @printf("  min_rise %.3f-%.3f (%d pts): ε = %.3f (%.3f-%.3f), PA = %.1f° ± %.1f°, inclination %.0f°, conc %.2f%s\n",
                    rows[p.s].min_rise, rows[p.e].min_rise, p.e - p.s + 1,
                    median(es), minimum(es), maximum(es), pam,
                    maximum(angdiff(x, pam) for x in pas),
                    acosd(1 - median(es)), median([conc_of(r) for r in pr]),
                    p.s == plateaus[1].s ? "   <- chosen" : "")
        end
    end

    println()
    suggested = nothing
    plateau   = nothing
    if best !== nothing
        lo, hi = best
        # plateau point closest to the plateau's median ellipticity and PA;
        # endpoints are excluded when the plateau has >= 3 points (they border the failure)
        cand   = (hi - lo + 1 >= 3) ? ((lo + 1):(hi - 1)) : (lo:hi)
        pl_ell = median([rows[k].ellipticity for k in lo:hi])
        pl_pa  = pa_median_deg([rows[k].pa_deg for k in lo:hi])
        kbest  = cand[argmin([abs(rows[k].ellipticity - pl_ell) / spread_ell +
                              angdiff(rows[k].pa_deg, pl_pa) / spread_pa for k in cand])]
        suggested = rows[kbest].min_rise
        plateau   = (rows[lo].min_rise, rows[hi].min_rise)
        de, dp    = window_spreads(lo, hi)
        println("Plateau (ellipticity, PA) for min_rise in [", plateau[1], ", ", plateau[2],
                "]  (", hi - lo + 1, " points, Δε ", round(de, digits = 3),
                ", ΔPA ", round(dp, digits = 1), "°)")
        println("Suggested min_rise = ", suggested,
                "  (ε = ", round(rows[kbest].ellipticity, digits = 3),
                ", PA = ", round(rows[kbest].pa_deg, digits = 1),
                "°, A2 = ", round(rows[kbest].A2, digits = 3), ")")
    else
        println("No plateau found (>= $min_plateau_points points with steps <= $tol_ell / $(tol_pa)°, ",
                "total spread <= $spread_ell / $(spread_pa)°).")
        println("Inspect ", pre, "threshold_sweep.png; the shape parameters may drift steadily with threshold.")
    end

    ###########################################################
    # SUMMARY OVER MODERATE MASK AREAS (median and range)
    ###########################################################

    summary = nothing

    # primary rows: the plateau if there is one, otherwise the moderate-area rows
    if best !== nothing
        sel    = rows[best[1]:best[2]]
        source = "plateau"
    else
        sel    = filter(r -> range_area[1] <= r.area_frac <= range_area[2], rows)
        source = "area range"
    end
    # wider, more conservative set: all rows with mask area in `wide_area` (plus the primary rows)
    wide = filter(r -> r in sel || wide_area[1] <= r.area_frac <= wide_area[2], rows)

    if length(sel) >= 2
        es   = [r.ellipticity for r in sel]
        pas  = [r.pa_deg for r in sel]
        pa_med  = pa_median_deg(pas)
        pa_half = maximum(angdiff(p, pa_med) for p in pas)
        ell_med = median(es)
        wes  = [r.ellipticity for r in wide]
        wpas = [r.pa_deg for r in wide]
        cs   = [conc_of(r) for r in sel]
        wcs  = [conc_of(r) for r in wide]
        shared = level_share(sel)
        a2s = a2_stability(rows)
        sens = ap === nothing ? NamedTuple[] :
               aperture_sensitivity(gwork, sm_ds, iso_downsample, xn, yn,
                                    clamp(1 - ell_med, 0.2, 1.0), deg2rad(pa_med), floor_t)
        gsens = filter(c -> c.coverage >= APERTURE_MIN_COVERAGE, sens)
        isempty(gsens) && (gsens = sens)
        csens = [c.conc for c in gsens]
        summary = (; source, n = length(sel),
                     min_rise_lo = minimum(r.min_rise for r in sel),
                     min_rise_hi = maximum(r.min_rise for r in sel),
                     ell_med, ell_min = minimum(es), ell_max = maximum(es),
                     pa_med, pa_half,
                     incl_med = acosd(1 - ell_med),
                     incl_min = acosd(1 - minimum(es)),
                     incl_max = acosd(1 - maximum(es)),
                     wide_n = length(wide),
                     wide_ell_min = minimum(wes), wide_ell_max = maximum(wes),
                     wide_pa_half = maximum(angdiff(p, pa_med) for p in wpas),
                     wide_incl_min = acosd(1 - minimum(wes)),
                     wide_incl_max = acosd(1 - maximum(wes)),
                     conc_med = median(cs),
                     conc_min = minimum(vcat(cs, csens)), conc_max = maximum(vcat(cs, csens)),
                     wide_conc_min = minimum(wcs), wide_conc_max = maximum(wcs),
                     level_share = shared,
                     a2_stab = a2s.frac, a2_n_limit = a2s.n_limit, a2_med_radius = a2s.med_radius)
        println()
        @printf("Disk shape from %s (%d thresholds, min_rise %.3f-%.3f):\n",
                summary.source, summary.n, summary.min_rise_lo, summary.min_rise_hi)
        @printf("  ellipticity = %.3f (%.3f-%.3f)\n", summary.ell_med, summary.ell_min, summary.ell_max)
        @printf("  PA          = %.1f° ± %.1f°\n", summary.pa_med, summary.pa_half)
        @printf("  inclination = %.0f° (%.0f°-%.0f°, thin disk)\n",
                summary.incl_med, summary.incl_min, summary.incl_max)
        # radial trend of the isophotes in the row that will be analysed in full
        kb = suggested === nothing ? nothing : findfirst(r -> r.min_rise == suggested, rows)
        fsw = kb === nothing ? NamedTuple[] : sort(rows[kb].fits; by = f -> f.a)
        if length(fsw) >= 3
            dE = fsw[end].ell - fsw[1].ell
            dP = mod(fsw[end].pa - fsw[1].pa + 90, 180) - 90
            if abs(dE) > ISO_TREND_ELL || abs(dP) > ISO_TREND_PA
                emin = min(summary.ell_min, minimum(f.ell for f in fsw))
                emax = max(summary.ell_max, maximum(f.ell for f in fsw))
                @printf("  RADIAL TREND in the isophotes of the suggested row (Δε = %+.3f, ΔPA = %+.1f°): the plateau median above is not a single disk value.\n",
                        dE, dP)
                @printf("    ε range %.3f-%.3f (inclination %.0f°-%.0f°); outermost isophote: ε = %.3f, PA = %.1f°, inclination %.0f°\n",
                        emin, emax, acosd(1 - emin), acosd(1 - emax),
                        fsw[end].ell, fsw[end].pa, acosd(1 - fsw[end].ell))
            end
        end
        @printf("  wider spread over %d thresholds with mask area %.0f-%.0f %%: ε %.3f-%.3f, PA ± %.1f°, inclination %.0f°-%.0f°\n",
                summary.wide_n, 100wide_area[1], 100wide_area[2], summary.wide_ell_min,
                summary.wide_ell_max, summary.wide_pa_half, summary.wide_incl_min,
                summary.wide_incl_max)
        isempty(sens) || print_aperture_sensitivity(sens)
        @printf("  concentration (fixed aperture) = %.2f, range %.2f-%.2f over thresholds and apertures; wider spread %.2f-%.2f\n",
                summary.conc_med, summary.conc_min, summary.conc_max,
                summary.wide_conc_min, summary.wide_conc_max)
        @printf("  A2 stability: %.0f %% of %d rows peak at a consistent radius (median %.0f px); %d row(s) pinned at the search limit\n",
                100 * summary.a2_stab, length(rows), summary.a2_med_radius, summary.a2_n_limit)
        summary.a2_stab < 0.7 &&
            println("  NOTE: the A2 peak is not stable across thresholds, so a bar detection here is not supported.")
        summary.level_share > 0.5 &&
            @printf("  NOTE: %.0f %% of the isophote levels are repeated across these rows (levels pinned near the noise floor), so they are not independent and the spread above understates the uncertainty.\n",
                    100 * summary.level_share)
    else
        println()
        println("Fewer than 2 rows for a ranged summary (no plateau, and mask area in ",
                range_area, " for < 2 rows).")
    end

    ###########################################################
    # OUTPUT FILES
    ###########################################################

    rows = all_rows          # restore all rows (including excluded ones) for the outputs
    n    = length(rows)

    open(joinpath(outdir, pre * "threshold_sweep.csv"), "w") do io
        println(io, "min_rise,threshold,area_frac,ellipticity,pa_deg,n_iso,r20_px,r80_px,concentration_mask,concentration_fixed_aperture,aperture_coverage,A2,A2_radius_px,A2_search_limit_px,A2_at_limit")
        for r in rows
            println(io, join((r.min_rise, r.threshold, r.area_frac, r.ellipticity,
                              r.pa_deg, r.n_iso, r.r20, r.r80, r.conc, r.conc_ap, r.ap_cov, r.A2, r.A2_radius,
                              r.A2_limit, r.A2_at_limit), ","))
        end
    end

    if n >= 2
        xs = [r.min_rise for r in rows]
        p = plot(
            plot(xs, [r.area_frac for r in rows],   marker = :o, ylabel = "Mask area fraction", legend = false),
            plot(xs, [r.ellipticity for r in rows], marker = :o, ylabel = "Ellipticity",        legend = false),
            plot(xs, [r.pa_deg for r in rows],      marker = :o, ylabel = "PA (deg)",           legend = false),
            plot(xs, [r.A2 for r in rows],          marker = :o, ylabel = "Peak A2",            legend = false),
            layout = (2, 2), size = (1000, 700), xlabel = "min_rise",
        )
        if suggested !== nothing
            for sp in 1:4
                vline!(p[sp], [suggested]; ls = :dash, color = :red, label = "")
            end
        end
        savefig(p, joinpath(outdir, pre * "threshold_sweep.png"))
        println("Saved: ", pre, "threshold_sweep.png, ", pre, "threshold_sweep.csv")
    end

    logmsg("Sweep done.")
    return (; rows, suggested, plateau, plateaus, summary)
end

end # module

###############################################################
# RUN (executes when you use "Execute Active File in REPL")
###############################################################

using .GalaxyAnalysis
using Printf

# Images to analyze (looked up as given, then next to this file)
galaxies = [
    "NGC 1300.jpg",
    "NGC 2525.jpg",
    "NGC 6984.jpg",
]

# Used when the sweep finds no plateau
fallback_min_rise = 0.15f0

# Optional per-image keywords shared by the sweep and the analysis, e.g.
#   "NGC 6984.jpg" => (; center = (856, 1402))
overrides = Dict{String,Any}()

all_results = NamedTuple[]
failed      = String[]

for galaxy in galaxies
    println("\n", "#"^70)
    println("# ", galaxy)
    println("#"^70)

    try
        extra = get(overrides, galaxy, (;))

        sweep = sweep_threshold(galaxy; extra...)

        # use the plateau value if one was found, otherwise the fallback
        mr = sweep.suggested === nothing ? fallback_min_rise : Float32(sweep.suggested)
        println("\nRunning full analysis for ", galaxy, " with min_rise = ", mr)

        res = analyze_galaxy(galaxy; min_rise = mr, extra...)

        push!(all_results,
              merge((; galaxy = galaxy, min_rise = Float64(mr),
                       from_plateau = sweep.suggested !== nothing,
                       sweep_summary = sweep.summary), res))
    catch err
        println("\nFAILED: ", galaxy)
        showerror(stdout, err)
        println()
        push!(failed, galaxy)
    end
end

###############################################################
# SUMMARY TABLE + CSV
###############################################################

# Combined uncertainty: the larger of the sweep spread and the scatter between
# isophote levels. When the isophotes show a systematic radial trend, a symmetric
# ± is misleading, so the full ellipticity range is quoted instead.
function combined_shape(r)
    ie = isnan(r.iso_spread_ell) ? 0.0 : r.iso_spread_ell / 2
    ip = isnan(r.iso_spread_pa)  ? 0.0 : r.iso_spread_pa / 2
    if r.sweep_summary === nothing
        ell, eu, pa, pu, src = r.disk_ellipticity, ie, r.disk_pa_deg, ip, "single"
        lo, hi = ell - eu, ell + eu
        trend = r.radial_trend && !isnan(r.ell_iso_min)
        trend && ((lo, hi) = (r.ell_iso_min, r.ell_iso_max))
    else
        s = r.sweep_summary
        ell, pa, src = s.ell_med, s.pa_med, string(s.source)
        eu = max((s.ell_max - s.ell_min) / 2, ie)
        pu = max(s.pa_half, ip)
        lo, hi = ell - eu, ell + eu
        trend = r.radial_trend && !isnan(r.ell_iso_min)
        trend && ((lo, hi) = (min(s.ell_min, r.ell_iso_min), max(s.ell_max, r.ell_iso_max)))
    end
    lo = clamp(lo, 0.0, 0.95); hi = clamp(hi, 0.0, 0.95)
    return (ell = ell, ell_unc = eu, pa = pa, pa_unc = pu, incl = acosd(1 - ell), source = src,
            trend = trend, ell_lo = lo, ell_hi = hi,
            incl_lo = acosd(1 - lo), incl_hi = acosd(1 - hi),
            ell_outer = r.ell_outer,
            incl_outer = isnan(r.ell_outer) ? NaN : acosd(1 - r.ell_outer))
end

function flag_string(r)
    f = String[]
    r.sweep_summary !== nothing && r.sweep_summary.level_share > 0.5 && push!(f, "shared-levels")
    r.radial_trend && push!(f, "ε(r)")
    !isnan(r.aperture_coverage) && r.aperture_coverage < 0.9 && push!(f, "ap-cov<0.9")
    r.bar_at_limit && push!(f, "A2-at-limit")
    if r.sweep_summary !== nothing && !isnan(r.sweep_summary.a2_stab) && r.sweep_summary.a2_stab < 0.7
        push!(f, "A2-unstable")
    end
    r.bar_aligned && push!(f, "bar-aligned")
    return isempty(f) ? "-" : join(f, ",")
end

# Concentration: fixed-aperture values (plateau median and range when available)
function conc_string(r)
    if r.sweep_summary === nothing
        return @sprintf("%.2f", isnan(r.concentration_ap) ? r.concentration : r.concentration_ap)
    end
    @sprintf("%.2f (%.2f-%.2f)", r.sweep_summary.conc_med, r.sweep_summary.conc_min, r.sweep_summary.conc_max)
end

println("\n", "="^185)
println(" SUMMARY   (ε, PA, inclination: median ± max(sweep spread, half isophote scatter); with a radial trend the full isophote range is shown;")
println("            concentration: fixed aperture, range over apertures and thresholds)")
println("="^185)
@printf("%-14s %-13s %8s %-14s %-12s %-9s %-8s %-17s %6s %7s %6s %8s %6s  %s\n",
        "galaxy", "nucleus", "min_rise", "ellipticity", "PA_deg", "incl_deg", "ε_outer", "concentration", "A2", "A2_rad", "A2_stb", "aligned", "arm_c", "flags")
for r in all_results
    c = combined_shape(r)
    es = c.trend ? @sprintf("%.2f-%.2f", c.ell_lo, c.ell_hi) : @sprintf("%.2f ±%.2f", c.ell, c.ell_unc)
    is = c.trend ? @sprintf("%.0f-%.0f", c.incl_lo, c.incl_hi) : @sprintf("%.0f", c.incl)
    eo = isnan(c.ell_outer) ? "-" : @sprintf("%.2f", c.ell_outer)
    st = r.sweep_summary === nothing || isnan(r.sweep_summary.a2_stab) ? "-" : @sprintf("%.0f%%", 100 * r.sweep_summary.a2_stab)
    @printf("%-14s %-13s %8.3f %-14s %-12s %-9s %-8s %-17s %6.3f %7.0f %6s %8s %6.3f  %s\n",
            r.galaxy, string(r.nucleus), r.min_rise, es,
            @sprintf("%.0f ±%.0f", c.pa, c.pa_unc), is, eo,
            conc_string(r), r.bar_strength, r.bar_radius, st,
            string(r.bar_aligned), r.arm_contrast, flag_string(r))
end
println("flags: shared-levels = sweep rows reuse the same isophote levels (spread understates error); ε(r) = ε/PA drift with radius (range shown, prefer ε_outer for the disk);")
println("       ap-cov<0.9 = concentration aperture partly outside the frame (biased); A2-at-limit = A2 peak pinned at the bar search limit; A2-unstable = A2 peak radius not stable across the sweep;")
println("       bar-aligned = bar axis within tolerance of disk PA (ε/PA may be bar-driven). A2_stb = fraction of sweep rows with a consistent A2 peak radius.")
isempty(failed) || println("\nFailed: ", join(failed, ", "))

sval(r, key) = r.sweep_summary === nothing ? "" : getproperty(r.sweep_summary, key)

open(joinpath(@__DIR__, "galaxy_summary.csv"), "w") do io
    println(io, "galaxy,nucleus_x,nucleus_y,min_rise,from_plateau,disk_method,",
                "ellipticity,ellipticity_moments,pa_deg,inclination_deg,",
                "sweep_source,ell_unc,pa_unc,ell_lo,ell_hi,incl_lo,incl_hi,ell_outer,incl_outer,sweep_level_share,radial_trend,ell_trend,pa_trend,sweep_n,sweep_ell_med,sweep_ell_min,sweep_ell_max,sweep_pa_med,sweep_pa_halfrange,",
                "sweep_incl_med,sweep_incl_min,sweep_incl_max,",
                "wide_n,wide_ell_min,wide_ell_max,wide_pa_halfrange,wide_incl_min,wide_incl_max,",
                "sweep_conc_med,sweep_conc_min,sweep_conc_max,wide_conc_min,wide_conc_max,",
                "concentration_fixed_aperture,aperture_a_px,aperture_coverage,r20_px,r80_px,concentration_mask,bar_A2,bar_radius_px,bar_pa_deg,bar_aligned,",
                "arm_strength,arm_contrast,flux,galaxy_pixels,",
                "a2_stability,a2_rows_at_limit,bar_at_limit,conc_aperture_min,conc_aperture_max")
    for r in all_results
        println(io, join((r.galaxy, r.nucleus[1], r.nucleus[2], r.min_rise, r.from_plateau,
                          r.disk_method, r.disk_ellipticity, r.disk_ellipticity_moments,
                          r.disk_pa_deg, r.inclination,
                          sval(r, :source), combined_shape(r).ell_unc, combined_shape(r).pa_unc,
                          combined_shape(r).ell_lo, combined_shape(r).ell_hi, combined_shape(r).incl_lo,
                          combined_shape(r).incl_hi, combined_shape(r).ell_outer, combined_shape(r).incl_outer,
                          sval(r, :level_share),
                          r.radial_trend, r.ell_trend, r.pa_trend, sval(r, :n), sval(r, :ell_med), sval(r, :ell_min), sval(r, :ell_max),
                          sval(r, :pa_med), sval(r, :pa_half),
                          sval(r, :incl_med), sval(r, :incl_min), sval(r, :incl_max),
                          sval(r, :wide_n), sval(r, :wide_ell_min), sval(r, :wide_ell_max),
                          sval(r, :wide_pa_half), sval(r, :wide_incl_min), sval(r, :wide_incl_max),
                          sval(r, :conc_med), sval(r, :conc_min), sval(r, :conc_max),
                          sval(r, :wide_conc_min), sval(r, :wide_conc_max),
                          r.concentration_ap, r.aperture_a, r.aperture_coverage,
                          r.r20, r.r80, r.concentration, r.bar_strength, r.bar_radius,
                          r.bar_pa_image, r.bar_aligned, r.arm_strength, r.arm_contrast,
                          r.flux, r.galaxy_pixels,
                          sval(r, :a2_stab), sval(r, :a2_n_limit), r.bar_at_limit,
                          r.conc_ap_min, r.conc_ap_max), ","))
    end
end
println("\nSaved: galaxy_summary.csv")

println("\n", "#"^70)
println("# galaxy_analysis.jl  FINISHED  ", Dates.now())
println("# code version: ", CODE_VERSION)
println("# galaxies analyzed: ", length(all_results), " of ", length(galaxies),
        isempty(failed) ? "" : "   (failed: " * join(failed, ", ") * ")")
println("# elapsed: ", round((time() - T_START) / 60, digits = 1), " min")
println("#"^70)
