# galaxy_composite.jl
# ─────────────────────────────────────────────────────────────────────────────
# One-call composite of the best galaxy-feature filters from
# galaxy_laplacian_color_NGC_1300.ipynb, for ANY Hubble galaxy JPG/PNG/TIFF.
#
#   RUN AS-IS: edit IMAGE_PATH_DEFAULT at the bottom, then VS Code "Execute active file in REPL".
#   FROM A NOTEBOOK (never put these lines inside this file):
#       GALAXY_NO_AUTORUN = true;  include("galaxy_composite.jl")
#   res = GalaxyComposite.galaxy_composite("NGC 1300.jpg")   # or an image array
#   res.panel                                         # 2×3 overview
#   res.enhanced                                      # natural-colour enhanced image
#   GalaxyComposite.save_composite(res, "out"; prefix="ngc1300")      # writes PNGs
#
# Command line:  julia galaxy_composite.jl "NGC 1300.jpg" out_dir
#
# Filters used (and why they were kept from the notebook):
#   • Gaussian unsharp / smooth residual  (best overall: dust lanes, clumps)
#   • Mid-scale DoG band 2-8 px           (star-forming knots, cleanest band)
#   • Hessian ridges, FIXED to give both bright (arm skeleton) and dark ridges
#   • Scharr gradient, luminance with original colour (best colour-faithful edges)
#   • Scale-coded DoG (R=8-16, G=4-8, B=2-4 px)
# Dropped: σ=1 LoG (noise/seams), structure tensor, 1-level dwt, radial residual
# (circular model on an inclined barred disk leaves geometry artefacts).
#
# NOTE: all pixel scales are given for a ~3920 px image (the NGC 1300 file the
# notebook used) and are rescaled automatically to your image size (scale=:auto).
# ─────────────────────────────────────────────────────────────────────────────
println("[debug] galaxy_composite.jl started | Julia $(VERSION) | pwd = $(pwd())"); flush(stdout)
println("[debug] loading packages (the first load can take a minute or more)..."); flush(stdout)

module GalaxyComposite

using Images, ImageFiltering, FileIO, ImageIO, Statistics, MosaicViews
println("[debug] packages loaded, defining module GalaxyComposite"); flush(stdout)

export galaxy_composite, save_composite, describe_panel

const REF_SIZE = 3920   # long side (px) the σ values below were tuned for

# ── helpers ──────────────────────────────────────────────────────────────────
gauss(x, σ) = imfilter(Float32, x, Kernel.gaussian((Float64(σ), Float64(σ))))

# fast quantile on a strided subsample (full sort of 14 M pixels is slow)
function qtl(x, p; nmax = 1_500_000)
    v  = vec(x)
    st = max(1, length(v) ÷ nmax)
    return Float32(quantile(collect(@view v[1:st:end]), p))
end

# percentile-normalise a non-negative map into [0,1]
nrm01(x, p) = clamp.(x ./ max(qtl(x, p), 1f-9), 0f0, 1f0)

function load_rgb(src)
    raw = src isa AbstractString ? load(src) : src
    return RGB{Float32}.(raw)
end

# second derivatives by central differences (replicate edges)
function second_derivs(L::AbstractMatrix{Float32})
    H, W = size(L)
    Lxx = similar(L); Lyy = similar(L); Lxy = similar(L)
    @inbounds for j in 1:W
        jm = max(j - 1, 1); jp = min(j + 1, W)
        for i in 1:H
            im = max(i - 1, 1); ip = min(i + 1, H)
            c = L[i, j]
            Lxx[i, j] = L[i, jp] - 2c + L[i, jm]
            Lyy[i, j] = L[ip, j] - 2c + L[im, j]
            Lxy[i, j] = 0.25f0 * (L[ip, jp] - L[ip, jm] - L[im, jp] + L[im, jm])
        end
    end
    return Lxx, Lyy, Lxy
end

# Hessian ridge maps. bright = crest (arm skeleton), dark = trough (dust lane).
# The |λ_small|/|λ_large| weight suppresses round blobs (stars) – a Frangi-style term.
function ridge_maps(L::AbstractMatrix{Float32})
    Lxx, Lyy, Lxy = second_derivs(L)
    bright = zeros(Float32, size(L)); dark = zeros(Float32, size(L))
    @inbounds for k in eachindex(L)
        a = Lxx[k]; d = Lyy[k]; b = Lxy[k]
        t  = (a + d) / 2
        r  = sqrt(((a - d) / 2)^2 + b^2)
        λ1 = t + r          # larger eigenvalue
        λ2 = t - r          # smaller eigenvalue (most negative across a crest)
        if λ2 < 0
            rb = abs(λ1) / (abs(λ2) + 1f-12)
            bright[k] = -λ2 * exp(-rb^2 / 0.5f0)
        end
        if λ1 > 0
            rb = abs(λ2) / (abs(λ1) + 1f-12)
            dark[k] = λ1 * exp(-rb^2 / 0.5f0)
        end
    end
    return bright, dark
end

to_rgb(x) = RGB{Float32}.(x)

# ── panel titles: tiny built-in 5×7 bitmap font (no extra packages needed) ───
const FONT5x7 = Dict{Char,NTuple{7,UInt8}}(
    'A' => (0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11),
    'B' => (0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e),
    'C' => (0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e),
    'D' => (0x1e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1e),
    'E' => (0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f),
    'F' => (0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10),
    'G' => (0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0e),
    'H' => (0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11),
    'I' => (0x0e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0e),
    'J' => (0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0c),
    'K' => (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    'L' => (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f),
    'M' => (0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11),
    'N' => (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    'O' => (0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e),
    'P' => (0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10),
    'Q' => (0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d),
    'R' => (0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11),
    'S' => (0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e),
    'T' => (0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    'U' => (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e),
    'V' => (0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04),
    'W' => (0x11, 0x11, 0x11, 0x15, 0x15, 0x1b, 0x11),
    'X' => (0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11),
    'Y' => (0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04),
    'Z' => (0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f),
    '0' => (0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e),
    '1' => (0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e),
    '2' => (0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f),
    '3' => (0x1f, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0e),
    '4' => (0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02),
    '5' => (0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e),
    '6' => (0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e),
    '7' => (0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    '8' => (0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e),
    '9' => (0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c),
    '-' => (0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00),
    '+' => (0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00),
    '/' => (0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10),
    ':' => (0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00),
    '(' => (0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02),
    ')' => (0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08),
    ',' => (0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08),
    '.' => (0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c),
    '=' => (0x00, 0x00, 0x1f, 0x00, 0x1f, 0x00, 0x00),
    '|' => (0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    '?' => (0x0e, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04),
    ' ' => (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
)

# (title, subtitle) for the six tiles, in panel order: col 1 top/bottom, col 2 top/bottom, col 3 top/bottom
const PANEL_TITLES = [
    ("1 ORIGINAL",      "INPUT IMAGE (REFERENCE)"),
    ("2 ENHANCED",      "DETAIL ADDED TO BRIGHTNESS, COLOUR KEPT"),
    ("3 FEATURE MAP",   "GREEN ARMS  BLUE KNOTS  RED DARK/DUST"),
    ("4 SCHARR EDGES",  "EDGE STRENGTH IN ORIGINAL COLOUR"),
    ("5 RESIDUAL",      "WHITE BRIGHTER  BLACK DARKER"),
    ("6 SCALE-CODED",   "BLUE 2-4 PX  GREEN 4-8  RED 8-16"),
]

function draw_text!(img::AbstractMatrix, text::AbstractString, x0::Int, y0::Int, k::Int, col)
    H, W = size(img)
    x = x0
    for ch in uppercase(text)
        g = get(FONT5x7, ch, FONT5x7['?'])
        for r in 1:7, c in 1:5
            ((g[r] >> (5 - c)) & 0x01) == 0x01 || continue
            for dy in 0:k-1, dx in 0:k-1
                yy = y0 + (r - 1) * k + dy
                xx = x + (c - 1) * k + dx
                (1 <= yy <= H && 1 <= xx <= W) && (img[yy, xx] = col)
            end
        end
        x += 6k
    end
    return img
end

# copy of `tile` with a darkened banner at the top holding a title and a smaller subtitle
function add_title(tile, title::AbstractString, subtitle::AbstractString = "")
    img = RGB{Float32}.(tile)
    H, W = size(img)
    pad = max(2, W ÷ 150)
    kt  = clamp(W ÷ 300, 1, 6)
    kt  = max(1, min(kt, (W - 4pad) ÷ (6 * max(length(title), 1))))
    ks  = max(1, min(max(kt - 1, 1), (W - 4pad) ÷ (6 * max(length(subtitle), 1))))
    bh  = pad + 7kt + pad + (isempty(subtitle) ? 0 : 7ks + pad)
    bh  = min(bh, H)
    img[1:bh, :] .= img[1:bh, :] .* 0.25f0
    draw_text!(img, title, 2pad, pad, kt, RGB{Float32}(1, 1, 1))
    isempty(subtitle) || draw_text!(img, subtitle, 2pad, pad + 7kt + pad, ks, RGB{Float32}(0.85, 0.9, 1.0))
    return img
end

# ── main entry point ─────────────────────────────────────────────────────────
"""
    galaxy_composite(src; kwargs...) -> NamedTuple

`src` is a file path or an image array. Returns
`(enhanced, features, scharr, residual, scales, panel, layers, scale, size)`.

Panel layout (3 columns, each top / bottom):
  column 1: original / enhanced     column 2: features / scharr     column 3: residual / scales
features: R = dust (dark vs surroundings), G = arm ridges, B = star-forming knots.
Call `describe_panel(res)` to print a plain-language guide to every tile.

Keywords (σ values in px at the 3920 px reference size, rescaled by `scale`):
  scale=:auto        number, or :auto = max(H,W)/3920
  unsharp_σ=15       unsharp-mask radius (try 8–30; larger = more global contrast)
  unsharp_amount=1.0 gain of the unsharp detail added to luminance
  knot_amount=0.6    brightening of 2–8 px star-forming knots
  dust_amount=0.5    extra darkening of dust lanes
  resid_σ=50         radius of the smooth-model residual panel
  ridge_σ=15         Hessian scale for the arm skeleton
  dust_σ=8           background radius for the dust map
  scharr_σ=4         pre-blur for the Scharr panel
  pct=0.995          percentile used for every stretch
  maxdim=nothing     downsample so the long side ≤ maxdim (fast previews)
  panel_width=900    width of each panel tile
  titles=true        burn a numbered title + subtitle into each tile of the panel (saved single PNGs stay clean)
"""
function galaxy_composite(src;
        scale = :auto,
        unsharp_σ = 15, unsharp_amount = 1.0,
        knot_amount = 0.6, dust_amount = 0.5,
        resid_σ = 50, ridge_σ = 15, dust_σ = 8, scharr_σ = 4,
        pct = 0.995, maxdim = nothing, panel_width = 900, titles = true, verbose = true)

    t0  = time()
    say(m) = verbose && (println("[galaxy_composite +", round(time() - t0, digits = 1), "s] ", m); flush(stdout))

    say("loading image: $(src isa AbstractString ? src : "array")")
    rgb = load_rgb(src)
    say("loaded $(size(rgb, 2))×$(size(rgb, 1)) px")
    if maxdim !== nothing && max(size(rgb)...) > maxdim
        rgb = RGB{Float32}.(imresize(rgb; ratio = maxdim / max(size(rgb)...)))
    end
    H, W = size(rgb)
    s  = scale === :auto ? max(H, W) / REF_SIZE : Float64(scale)
    sg = x -> max(Float64(x) * s, 0.8)          # σ in px for THIS image, floor 0.8
    say("image $(W)×$(H), scale factor $(round(s, digits=3))")

    lum = Float32.(Gray.(rgb))

    # 1. star mask (so detail layers don't ring around stars) ────────────────
    say("star mask")
    b1 = gauss(lum, sg(1)); b2 = gauss(lum, sg(2))
    fine     = abs.(b1 .- b2)
    starcore = Float32.(fine .> qtl(fine, 0.9995))
    stars    = Float32.(gauss(starcore, sg(4)) .> 0.005f0)
    keep     = 1f0 .- clamp.(gauss(stars, sg(2)) .* 2f0, 0f0, 1f0)      # 1 galaxy, 0 star
    wide     = 1f0 .- clamp.(gauss(stars, sg(8)) .* 8f0, 0f0, 1f0)      # wider, for dust

    # 2. unsharp mask (notebook: "very good") ───────────────────────────────
    say("unsharp / residual")
    un = lum .- gauss(lum, sg(unsharp_σ))

    # 3. mid-scale knots = DoG 2–8 px (bands 3+4 telescope to b2 − b8) ───────
    b8    = gauss(lum, sg(8))
    knots = max.(b2 .- b8, 0f0) .* keep
    kn    = nrm01(knots, pct)

    # 4. ridges (clip very bright pixels first so stars don't dominate) ───────
    say("Hessian ridges")
    cl = min.(lum, qtl(lum, 0.995))
    bright, _ = ridge_maps(gauss(cl, sg(ridge_σ)))
    brn = nrm01(bright .* keep, pct)

    # dust = darker than surroundings, away from stars and bright blobs
    blob  = Float32.(knots .> qtl(knots, 0.99))
    dwide = wide .* (1f0 .- clamp.(gauss(blob, sg(8)) .* 6f0, 0f0, 1f0))
    dust  = max.(-(lum .- b8), 0f0) .* dwide .* Float32.(lum .> 0.03f0)
    dn    = nrm01(dust, pct)

    # 5. natural-colour enhanced image: add detail to luminance, keep chroma ─
    say("compose enhanced image")
    Lnew = lum .+ Float32(unsharp_amount) .* un .* keep .+ Float32(knot_amount) * 0.25f0 .* kn
    Lnew = clamp.(Lnew .* (1f0 .- Float32(dust_amount) .* dn), 0f0, 1f0)
    ratio    = (Lnew .+ 0.02f0) ./ (lum .+ 0.02f0)
    enhanced = clamp01.(rgb .* ratio)

    # 6. feature map: R dust, G arm ridges, B knots ──────────────────────────
    features = RGB{Float32}.(dn .^ 0.7f0, brn .^ 0.7f0, kn .^ 0.7f0)

    # 7. Scharr, luminance with original colour (notebook: best colour-faithful) ─
    say("Scharr")
    gy, gx = imgradients(gauss(lum, sg(scharr_σ)), KernelFactors.scharr)
    m      = hypot.(gy, gx)
    v      = clamp.(m ./ max(qtl(m, pct), 1f-9), 0f0, 1f0)
    chroma = rgb ./ max.(lum, 0.05f0)
    scharr = clamp01.(chroma .* v)

    # 8. smooth-model residual (signed; dark = dust) ─────────────────────────
    say("residual + scale-coded bands")
    res    = lum .- gauss(lum, sg(resid_σ))
    rscale = qtl(abs.(res), pct)
    g      = clamp.(0.5f0 .+ 2f0 .* res ./ max(rscale, 1f-9), 0f0, 1f0)
    residual = RGB{Float32}.(g, g, g)

    # 9. scale-coded DoG: R = 8–16, G = 4–8, B = 2–4 px ──────────────────────
    b4 = gauss(lum, sg(4)); b16 = gauss(lum, sg(16))
    scales = RGB{Float32}.(nrm01(abs.(b8 .- b16), pct),
                           nrm01(abs.(b4 .- b8),  pct),
                           nrm01(abs.(b2 .- b4),  pct))

    # 10. overview panel ─────────────────────────────────────────────────────
    say("building panel")
    r     = min(1.0, panel_width / W)
    thumb = x -> r < 1 ? RGB{Float32}.(imresize(to_rgb(x); ratio = r)) : to_rgb(x)
    tiles = [thumb(x) for x in (rgb, enhanced, features, scharr, residual, scales)]
    titles && (tiles = [add_title(t, ti, su) for (t, (ti, su)) in zip(tiles, PANEL_TITLES)])
    panel = mosaicview(tiles...; nrow = 2, npad = 4, fillvalue = RGB{Float32}(0.08, 0.08, 0.08))

    say("done")
    return (; enhanced, features, scharr, residual, scales, panel,
              layers = (; unsharp = un, knots, ridge = bright, dust, keep),
              scale = s, size = (H, W),
              panel_order = ["col 1 top: original", "col 1 bottom: enhanced",
                             "col 2 top: features (R dust, G arms, B knots)", "col 2 bottom: scharr (colour)",
                             "col 3 top: residual", "col 3 bottom: scale-coded 2-4/4-8/8-16 px"])
end

# Plain-language guide printed under the displayed panel.
function describe_panel(res)
    H, W = res.size
    bar = "═"^98
    println("\n", bar)
    println(" HOW TO READ THE PANEL   (image $(W)×$(H) px, filter scale factor $(round(res.scale, digits = 3)))")
    println(bar)
    println(" Three columns, each with a top and a bottom image.\n")

    println(" COLUMN 1")
    println("  TOP     [1] ORIGINAL   The input image, unchanged. Reference for the other five tiles.")
    println("  BOTTOM  [2] ENHANCED   Same colours with extra detail added to the brightness only:")
    println("                     unsharp mask (arm / dust-lane contrast), brightened 2-8 px star-forming knots,")
    println("                     darkened dust lanes. Stars are masked so they get no dark halos.")
    println("                     Look for: dust lanes along the bar and ring, clumpy clusters along the arms,")
    println("                     and the small spiral at the nucleus.\n")

    println(" COLUMN 2")
    println("  TOP     [3] FEATURE MAP   False colour on black.  GREEN = bright arm / ring ridges (skeleton),")
    println("                        BLUE = compact bright knots (star clusters, HII regions),")
    println("                        RED = regions darker than their surroundings (dust, inter-arm gaps).")
    println("                        Look for: the green ring and arm skeleton, blue dots where clusters sit.")
    println("                        Caution: red is a broad faint haze and partly noise, so read it as 'darker than")
    println("                        local background', not a measured dust column.")
    println("  BOTTOM  [4] SCHARR EDGES (gradient x original colour)   Edge strength after a ~4 px blur, painted in the")
    println("                        original colours: outlines of clusters, shells and dust filaments.")
    println("                        Blue edges = young clusters, rust / brown edges = dust. Interiors and the")
    println("                        smooth bulge and bar stay dark (no edges there).\n")

    println(" COLUMN 3")
    println("  TOP     [5] RESIDUAL   Image minus a heavy (sigma ~50 px) blur. Mid-grey = no change, white = brighter")
    println("                     than the surroundings, black = darker. Removes the smooth bulge / bar glow.")
    println("                     Look for: dust lanes as black ribbons, ring and arm clumps in white, the nuclear")
    println("                     spiral. Bright stars get dark halos here (expected for this method).")
    println("  BOTTOM  [6] SCALE-CODED BANDS   Structure strength by size, as colour: BLUE = 2-4 px, GREEN = 4-8 px,")
    println("                     RED = 8-16 px (pixel sizes at the reference resolution, rescaled to this image).")
    println("                     White / yellow = structure at several sizes, blue / cyan = fine compact detail,")
    println("                     red / orange = larger clumps. Shows size only, not bright vs dark, and the")
    println("                     8-16 px (red) band rings around stars.\n")

    println(" NOTES")
    println("  - Brightness-based display image, not calibrated flux: use it to compare structure, not for photometry.")
    println("  - To tune: unsharp_amount, knot_amount and dust_amount (enhanced tile); unsharp_σ (detail size).")
    println("  - Saved PNGs: *_enhanced, *_features, *_scharr, *_residual, *_scales, *_panel (same tiles, full size).")
    println(bar)
    flush(stdout)
    return nothing
end

function save_composite(res, outdir::AbstractString; prefix = "galaxy")
    mkpath(outdir)
    out(x) = RGB{N0f8}.(clamp01nan.(to_rgb(x)))
    for name in (:enhanced, :features, :scharr, :residual, :scales, :panel)
        p = joinpath(outdir, "$(prefix)_$(name).png")
        save(p, out(getfield(res, name)))
        println("saved ", p)
    end
    return outdir
end

end # module
println("[debug] module GalaxyComposite defined"); flush(stdout)

# NOTE: no `using .GalaxyComposite` here on purpose. Re-running this file in the same REPL redefines the
# module, and a second `using` then clashes with the old exports. Calls below are qualified instead.

# ═════════════════════════════════════════════════════════════════════════════
# AUTO-RUN  (VS Code "Execute active file in REPL", or `julia galaxy_composite.jl [image] [out_dir]`)
# EDIT THESE THREE LINES:
IMAGE_PATH_DEFAULT = "NGC 1300.jpg"   # relative to the current folder, this file's folder, or absolute
OUTDIR_DEFAULT     = "galaxy_out"
MAXDIM_DEFAULT     = 1500             # long side in px for a quick test; set to nothing for full size
#
# To only load the functions (e.g. from a notebook), run   GALAXY_NO_AUTORUN = true   before include().
# ═════════════════════════════════════════════════════════════════════════════
if @isdefined(GALAXY_NO_AUTORUN) && GALAXY_NO_AUTORUN
    println("[debug] GALAXY_NO_AUTORUN set: functions loaded, not running anything")
else
    try
        img_path = length(ARGS) >= 1 ? ARGS[1] : IMAGE_PATH_DEFAULT
        outdir   = length(ARGS) >= 2 ? ARGS[2] : OUTDIR_DEFAULT
        println("[debug] auto-run: image = ", repr(img_path), " | out dir = ", repr(outdir),
                " | maxdim = ", MAXDIM_DEFAULT)

        # look for the image: as given, then next to this script, then any jpg/png in the current folder
        cands = [img_path, joinpath(something(@__DIR__, pwd()), img_path)]
        found = findfirst(isfile, cands)
        if found === nothing
            println("[debug] ERROR: image not found. Looked in:")
            foreach(c -> println("          ", abspath(c)), cands)
            local_imgs = filter(f -> occursin(r"\.(jpe?g|png|tiff?)$"i, f), readdir(pwd()))
            println("[debug] image files in ", pwd(), ": ", isempty(local_imgs) ? "(none)" : local_imgs)
            println("[debug] set IMAGE_PATH_DEFAULT above to one of these (or an absolute path) and run again.")
        else
            path = cands[found]
            println("[debug] found image: ", abspath(path), " (", round(filesize(path) / 1e6, digits = 1), " MB)")
            res = GalaxyComposite.galaxy_composite(path; maxdim = MAXDIM_DEFAULT)
            println("[debug] saving PNGs to ", abspath(outdir))
            GalaxyComposite.save_composite(res, outdir; prefix = splitext(basename(path))[1])
            println("[debug] showing panel (VS Code plot pane / image viewer)")
            try display(res.panel) catch e; println("[debug] display failed (", e, "), open the saved *_panel.png instead") end
            GalaxyComposite.describe_panel(res)
            println("[debug] FINISHED. Result is in the variable `res`  (res.enhanced, res.features, res.panel ...)")
        end
    catch e
        println("[debug] ERROR: ", sprint(showerror, e, catch_backtrace()))
    end
end
