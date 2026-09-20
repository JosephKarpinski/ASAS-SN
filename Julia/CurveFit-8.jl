"""
    CurveFit

Fit synthetic data with a straight line (by hand, via linear algebra), an
arbitrary non-linear model (via Optimization.jl), and finally a Bayesian
model (via Turing.jl), producing plots and (for the Bayesian part) corner
plots at each stage.

Based on the JuliaAstro tutorial "Curve Fitting".

# Dependencies

Hard dependencies (needed for the frequentist half — data, linear algebra
fit, Optimization.jl fits, autodiff+BFGS timing demo):

    pkg> add Plots Optimization OptimizationOptimJL ForwardDiff

(`LinearAlgebra`, `Random`, and `DelimitedFiles` are Julia standard
libraries and don't need to be added.)

Optional, for the Bayesian section only: Turing and PairPlots. Like GLMakie
in the tabular-data module, these are *not* hard dependencies — they're
loaded lazily the first time you call a Bayesian function (`bayesian_fit_*`,
`corner_plot_*`, or `run_bayesian`). Install them yourself first if you want
that part:

    pkg> add Turing PairPlots

Turing is a large package; expect its first load in a session to take a
while to precompile, and NUTS sampling itself takes real time too.

# Getting the data

Unlike the other tutorials, this one has no data to download — it's
synthetic (a noisy weak parabola), generated with a fixed random seed so
results are reproducible. `load_or_generate_data` still follows the same
"check the local directory first" pattern as the other modules: it caches
the generated `(x, y)` to a CSV in your working directory and reuses it on
later runs instead of regenerating, unless you pass `force = true`.

# Running it

Executing the file (`julia CurveFit.jl`, `include`, or VS Code's "Execute
active file in REPL") loads the module and, because `AUTORUN_CURVEFIT` at
the bottom of the file is `true`, runs the frequentist pipeline
(`run_demo()`) and saves its plots. `AUTORUN_BAYESIAN` defaults to `false`
since that pipeline is much slower (Turing's first-time precompile plus
NUTS sampling); set it to `true` if you want `run_bayesian()` to run
automatically too.

```julia
include("CurveFit.jl")
using .CurveFit

run_demo()                              # frequentist pipeline, saves plots
# or step by step:
x, y = load_or_generate_data()          # cached in pwd() after first run
intercept, slope = linear_regression_matrix(x, y)
slope2, intercept2 = fit_linear_optim(x, y)
u, _ = fit_quadratic_optim(x, y)

# Bayesian (loads Turing on first call; can be slow):
run_bayesian()
# or:
chain = bayesian_fit_linear(x, y)
plot_bayesian_linear(x, y, chain)
corner_plot_linear(chain; outpath = "lin-regress-corner.png")
```

Debug messages are on by default; silence them with `set_debug!(false)`.
"""
module CurveFit

using LinearAlgebra
using Random
using DelimitedFiles
using Plots
using Optimization
using OptimizationOptimJL
using ForwardDiff

export DATA_FILENAME,
       data_path,
       generate_data,
       load_or_generate_data,
       plot_data,
       plot_fit,
       linfunc,
       quadfunc,
       linear_regression_matrix,
       fit_linear_optim,
       fit_quadratic_optim,
       fit_linear_autodiff,
       bayesian_fit_linear,
       bayesian_fit_quadratic,
       plot_bayesian_linear,
       plot_bayesian_quadratic,
       corner_plot_linear,
       corner_plot_quadratic,
       run_demo,
       run_bayesian,
       set_debug!

# ---------------------------------------------------------------------------
# Debug output
# ---------------------------------------------------------------------------

const DEBUG = Ref(true)
const T0 = Ref(time())

"""
    set_debug!(on::Bool = true)

Turn the step-by-step debug messages on or off.
"""
function set_debug!(on::Bool = true)
    DEBUG[] = on
    return on
end

# Print "[DEBUG +12.3s] Module.function: message" and flush immediately so
# progress is visible even during long-running steps.
function _dbg(where_::AbstractString, msg...)
    DEBUG[] || return nothing
    t = round(time() - T0[]; digits = 1)
    println("[DEBUG +", t, "s] CurveFit.", where_, ": ", msg...)
    flush(stdout)
    return nothing
end

# ---------------------------------------------------------------------------
# Data generation / caching
# ---------------------------------------------------------------------------

"""Default local file name for the cached synthetic data."""
const DATA_FILENAME = "curvefit_data.csv"

"""
    data_path(; dir = pwd(), filename = DATA_FILENAME) -> String
"""
function data_path(; dir::AbstractString = pwd(), filename::AbstractString = DATA_FILENAME)
    return joinpath(dir, filename)
end

"""
    generate_data(; seed = 1234, xrange = 0:5:100) -> (x, y)

Generate a noisy weak parabola: `y = (x/20 - 0.2)^2 + 2 + randn()`. Seeding
the RNG makes this reproducible — the same `seed` always gives the same data.
"""
function generate_data(; seed::Integer = 1234, xrange = 0:5:100)
    _dbg("generate_data", "seeding RNG with ", seed, " and generating ", length(xrange), " points")
    Random.seed!(seed)
    x = collect(Float64, xrange)
    y = (x ./ 20 .- 0.2) .^ 2 .+ 2 .+ randn(length(x))
    _dbg("generate_data", "done")
    return x, y
end

"""
    load_or_generate_data(; dir = pwd(), filename = DATA_FILENAME, seed = 1234,
                             xrange = 0:5:100, force = false) -> (x, y)

Return the synthetic `(x, y)` data, reusing a cached CSV in `dir` if present
instead of regenerating it (this tutorial has nothing to download, so this
plays the same "check locally first" role `locate_catalog`/`download_carina`
play in the other modules). `force = true` regenerates and overwrites it.
"""
function load_or_generate_data(; dir::AbstractString = pwd(), filename::AbstractString = DATA_FILENAME,
                                seed::Integer = 1234, xrange = 0:5:100, force::Bool = false)
    _dbg("load_or_generate_data", "start (dir=", dir, ", force=", force, ")")
    mkpath(dir)
    path = data_path(; dir = dir, filename = filename)
    _dbg("load_or_generate_data", "target path = ", path)

    if !force && isfile(path) && filesize(path) > 0
        _dbg("load_or_generate_data", "cached data found, loading (pass force=true to regenerate)")
        data_matrix, _header = readdlm(path, ','; header = true)
        x = Float64.(data_matrix[:, 1])
        y = Float64.(data_matrix[:, 2])
        _dbg("load_or_generate_data", "loaded ", length(x), " points")
        return x, y
    end

    x, y = generate_data(; seed = seed, xrange = xrange)
    _dbg("load_or_generate_data", "writing data to ", path)
    open(path, "w") do io
        writedlm(io, ["x" "y"], ',')
        writedlm(io, hcat(x, y), ',')
    end
    _dbg("load_or_generate_data", "done")
    return x, y
end

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

"""
    plot_data(x, y; size = (1600, 1000), dpi = 200, show = true, kwargs...)

Plain scatter plot of the data. If `show` is true (the default), also
`display`s the plot — this is what makes it pop up in VS Code's Julia plot
pane (or a GUI window in a plain terminal); `savefig` alone only writes a
file and never triggers a display.
"""
function plot_data(x, y; size = (1600, 1000), dpi::Integer = 200, show::Bool = true, kwargs...)
    _dbg("plot_data", "plotting ", length(x), " points")
    p = scatter(x, y; xlabel = "x", ylabel = "y", label = "data", size = size, dpi = dpi, kwargs...)
    if show
        _dbg("plot_data", "displaying plot")
        display(p)
    end
    return p
end

"""
    plot_fit(x, y, yfit; fit_label = "best fit", size = (1600, 1000), dpi = 200,
             show = true, kwargs...)

Scatter the data and overlay a fitted curve `yfit` (same length as `x`).
`kwargs...` are passed to the fit line's `plot!` call (e.g. `color`, `alpha`).
If `show` is true (the default), also `display`s the plot — see
[`plot_data`](@ref) for why that's needed to see it in VS Code.
"""
function plot_fit(x, y, yfit; fit_label::AbstractString = "best fit",
                   size = (1600, 1000), dpi::Integer = 200, show::Bool = true, kwargs...)
    _dbg("plot_fit", "plotting data with fit line (", fit_label, ")")
    p = scatter(x, y; xlabel = "x", ylabel = "y", label = "data", size = size, dpi = dpi)
    plot!(p, x, yfit; label = fit_label, kwargs...)
    if show
        _dbg("plot_fit", "displaying plot")
        display(p)
    end
    return p
end

# ---------------------------------------------------------------------------
# Linear regression (from scratch, via linear algebra)
# ---------------------------------------------------------------------------

"""
    linfunc(x; slope, intercept)

`slope * x + intercept`.
"""
linfunc(x; slope, intercept) = slope * x + intercept

"""
    quadfunc(x, u)

`u[1]*x^2 + u[2]*x + u[3]`, for a 3-element coefficient vector `u`.
"""
quadfunc(x, u) = u[1] * x^2 + u[2] * x + u[3]

"""
    linear_regression_matrix(x, y) -> (intercept, slope)

Solve the normal equations for a straight-line fit directly:

    [N     Σx  ] [c1]   [Σy ]
    [Σx    Σx²] [c2] = [Σxy]

via `A \\ b`, where `c1` is the intercept and `c2` is the slope.
"""
function linear_regression_matrix(x, y)
    _dbg("linear_regression_matrix", "building normal equations for n=", length(x), " points")
    A = [
        length(x)  sum(x)
        sum(x)     sum(x .^ 2)
    ]
    b = [
        sum(y)
        sum(y .* x)
    ]
    _dbg("linear_regression_matrix", "A = ", A, ", b = ", b)
    c = A \ b
    intercept, slope = c[1], c[2]
    _dbg("linear_regression_matrix", "solved: intercept=", intercept, ", slope=", slope)
    return intercept, slope
end

# ---------------------------------------------------------------------------
# (Non-)linear curve fit via Optimization.jl
# ---------------------------------------------------------------------------

function _objective_linear(u, data)
    slope, intercept = u
    x, y = data
    residuals = linfunc.(x; slope, intercept) .- y
    return sum(residuals .^ 2)
end

function _objective_quadratic(u, data)
    x, y = data
    model = u[1] .* x .^ 2 .+ u[2] .* x .+ u[3]
    residuals = model .- y
    return sum(residuals .^ 2)
end

"""
    fit_linear_optim(x, y; u0 = [1.0, 1.0], solver = NelderMead()) -> (slope, intercept, sol)

Fit a line to `(x, y)` by minimizing the sum of squared residuals with
Optimization.jl (derivative-free `NelderMead` by default).
"""
function fit_linear_optim(x, y; u0 = [1.0, 1.0], solver = NelderMead())
    _dbg("fit_linear_optim", "start (u0=", u0, ")")
    data = [x, y]
    prob = OptimizationProblem(_objective_linear, u0, data)
    sol = solve(prob, solver)
    slope, intercept = sol.u
    _dbg("fit_linear_optim", "solved: slope=", slope, ", intercept=", intercept, ", retcode=", sol.retcode)
    return slope, intercept, sol
end

"""
    fit_quadratic_optim(x, y; u0 = [1.0, 1.0, 1.0], solver = NelderMead()) -> (u, sol)

Fit `u[1]*x^2 + u[2]*x + u[3]` to `(x, y)` by minimizing the sum of squared
residuals with Optimization.jl.
"""
function fit_quadratic_optim(x, y; u0 = [1.0, 1.0, 1.0], solver = NelderMead())
    _dbg("fit_quadratic_optim", "start (u0=", u0, ")")
    data = [x, y]
    prob = OptimizationProblem(_objective_quadratic, u0, data)
    sol = solve(prob, solver)
    _dbg("fit_quadratic_optim", "solved: u=", sol.u, ", retcode=", sol.retcode)
    return sol.u, sol
end

"""
    fit_linear_autodiff(x, y; u0 = [1.0, 1.0]) -> (slope, intercept, sol)

Same linear fit as [`fit_linear_optim`](@ref), but using forward-mode
automatic differentiation (ForwardDiff) and the higher-order `BFGS`
algorithm instead of a derivative-free method. Prints the solve time.
"""
function fit_linear_autodiff(x, y; u0 = [1.0, 1.0])
    _dbg("fit_linear_autodiff", "start (u0=", u0, ")")
    data = [x, y]
    optf = OptimizationFunction(_objective_linear, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, u0, data)
    t = @elapsed sol = solve(prob, BFGS())
    slope, intercept = sol.u
    _dbg("fit_linear_autodiff", "solved in ", round(t; digits = 4), "s: slope=", slope, ", intercept=", intercept)
    return slope, intercept, sol
end

# ---------------------------------------------------------------------------
# Bayesian models (optional — Turing + PairPlots, loaded lazily)
# ---------------------------------------------------------------------------

# `using Turing` inside this module the first time it's actually needed,
# rather than as a top-level dependency. This keeps Turing (a large package
# with a real precompile cost) entirely optional: nothing else here —
# including `run_demo` and AUTORUN_CURVEFIT — requires it.
function _ensure_turing!()
    if !isdefined(@__MODULE__, :Turing)
        _dbg("bayesian", "loading Turing (first use — a large package, this can take a while to precompile)...")
        Base.eval(@__MODULE__, :(using Turing))
        _dbg("bayesian", "Turing loaded")
    end
    return nothing
end

# Define the `@model`-based Bayesian models the first time they're needed.
# This has to happen via `eval` *after* `using Turing` so the `@model` macro
# exists — same reasoning as `_ensure_glmakie!`/`Base.invokelatest` in the
# tabular-data module, just applied to a macro instead of ordinary functions.
function _ensure_bayesian_models!()
    _ensure_turing!()
    if !isdefined(@__MODULE__, :_bayes_linear_model)
        _dbg("bayesian", "defining Bayesian model functions...")
        Base.eval(@__MODULE__, quote
            @model function _bayes_linear_model(x, y)
                σ₂ ~ truncated(Normal(0, 100), 0, Inf)
                intercept ~ Normal(0, 5)
                slope ~ Normal(0, 10)
                for i in eachindex(x, y)
                    y[i] ~ Normal(x[i] * slope + intercept, sqrt(σ₂))
                end
            end

            @model function _bayes_quad_model(x, y)
                σ₂ ~ truncated(Normal(0, 10), 0, Inf)
                u1 ~ Normal(0, 0.01)
                u2 ~ Normal(0, 0.1)
                u3 ~ Normal(0, 5)
                for i in eachindex(x, y)
                    model_val = u1 * x[i]^2 + u2 * x[i] + u3
                    y[i] ~ Normal(model_val, sqrt(σ₂))
                end
            end
        end)
        _dbg("bayesian", "models defined")
    end
    return nothing
end

"""
    bayesian_fit_linear(x, y; n = 500, target_accept = 0.65, seed = nothing) -> chain

Draw `n` posterior samples (intercept, slope, σ²) for a Bayesian linear
model using Turing's NUTS sampler. Loads Turing on first call — see the
module docstring. Pass `seed` to reseed the RNG immediately before sampling
(for reproducing a specific run, e.g. before a big corner-plot resample).
"""
function bayesian_fit_linear(x, y; n::Integer = 500, target_accept::Real = 0.65,
                              seed::Union{Nothing,Integer} = nothing)
    _ensure_bayesian_models!()
    return Base.invokelatest(_bayesian_fit_linear_impl, x, y; n = n, target_accept = target_accept, seed = seed)
end

function _bayesian_fit_linear_impl(x, y; n::Integer, target_accept::Real, seed)
    seed === nothing || Random.seed!(seed)
    _dbg("bayesian_fit_linear", "sampling ", n, " draws (NUTS, target_accept=", target_accept, ")...")
    model = _bayes_linear_model(x, y)
    chain = sample(model, NUTS(target_accept), n)
    _dbg("bayesian_fit_linear", "done")
    return chain
end

"""
    bayesian_fit_quadratic(x, y; n = 500, target_accept = 0.65, seed = nothing) -> chain

Draw `n` posterior samples (u1, u2, u3, σ²) for a Bayesian quadratic model
using Turing's NUTS sampler. Loads Turing on first call.
"""
function bayesian_fit_quadratic(x, y; n::Integer = 500, target_accept::Real = 0.65,
                                 seed::Union{Nothing,Integer} = nothing)
    _ensure_bayesian_models!()
    return Base.invokelatest(_bayesian_fit_quadratic_impl, x, y; n = n, target_accept = target_accept, seed = seed)
end

function _bayesian_fit_quadratic_impl(x, y; n::Integer, target_accept::Real, seed)
    seed === nothing || Random.seed!(seed)
    _dbg("bayesian_fit_quadratic", "sampling ", n, " draws (NUTS, target_accept=", target_accept, ")...")
    model = _bayes_quad_model(x, y)
    chain = sample(model, NUTS(target_accept), n)
    _dbg("bayesian_fit_quadratic", "done")
    return chain
end

"""
    plot_bayesian_linear(x, y, chain; size = (1600, 1000), dpi = 200, alpha = 0.05, show = true)

Overlay every posterior draw's line (gray, translucent) on the data —
a "spaghetti plot" showing the linear model's uncertainty. If `show` is
true (the default), also `display`s the plot — see [`plot_data`](@ref).

`chain` comes from a lazily-loaded package (Turing/FlexiChains), so — like
the fit and corner-plot functions — this dispatches its actual work via
`Base.invokelatest`, even though it doesn't itself `using` anything. Calling
it straight from a caller whose world age predates Turing being loaded
(e.g. `run_bayesian`) would otherwise raise a world-age `MethodError` when
indexing into `chain`.
"""
function plot_bayesian_linear(x, y, chain; size = (1600, 1000), dpi::Integer = 200,
                               alpha::Real = 0.05, show::Bool = true)
    return Base.invokelatest(_plot_bayesian_linear_impl, x, y, chain;
                              size = size, dpi = dpi, alpha = alpha, show = show)
end

function _plot_bayesian_linear_impl(x, y, chain; size, dpi::Integer, alpha::Real, show::Bool)
    _dbg("plot_bayesian_linear", "plotting posterior draws")
    # chain[:key] may come back as a plain Vector (old MCMCChains.Chains) or as a
    # DimensionalData-wrapped array (newer Turing/FlexiChains, shape (chain, iter)).
    # Strip any dimension labels and flatten to a plain Vector before doing arithmetic,
    # since the two backends disagree on shape/orientation.
    intercept = vec(Array(chain[:intercept]))
    slope = vec(Array(chain[:slope]))
    _dbg("plot_bayesian_linear", "n_draws=", length(slope))
    p = plot(x, x .* slope' .+ intercept'; label = "", color = :gray, alpha = alpha, size = size, dpi = dpi)
    scatter!(p, x, y; xlabel = "x", ylabel = "y", label = "data", color = 1)
    if show
        _dbg("plot_bayesian_linear", "displaying plot")
        display(p)
    end
    return p
end

"""
    plot_bayesian_quadratic(x, y, chain; size = (1600, 1000), dpi = 200, alpha = 0.1, show = true)

Same as [`plot_bayesian_linear`](@ref) but for the quadratic model's
posterior draws — see that docstring for why this is `invokelatest`-wrapped.
"""
function plot_bayesian_quadratic(x, y, chain; size = (1600, 1000), dpi::Integer = 200,
                                  alpha::Real = 0.1, show::Bool = true)
    return Base.invokelatest(_plot_bayesian_quadratic_impl, x, y, chain;
                              size = size, dpi = dpi, alpha = alpha, show = show)
end

function _plot_bayesian_quadratic_impl(x, y, chain; size, dpi::Integer, alpha::Real, show::Bool)
    _dbg("plot_bayesian_quadratic", "plotting posterior draws")
    # See _plot_bayesian_linear_impl: flatten any DimensionalData wrapper to a plain
    # Vector before arithmetic — the shape/orientation differs between MCMCChains and
    # the newer FlexiChains-based chain types.
    u1 = vec(Array(chain[:u1]))
    u2 = vec(Array(chain[:u2]))
    u3 = vec(Array(chain[:u3]))
    _dbg("plot_bayesian_quadratic", "n_draws=", length(u1))
    posterior = u1' .* x .^ 2 .+ u2' .* x .+ u3'
    p = plot(x, posterior; label = "", color = :gray, alpha = alpha, size = size, dpi = dpi)
    scatter!(p, x, y; xlabel = "x", ylabel = "y", label = "data", color = 1)
    if show
        _dbg("plot_bayesian_quadratic", "displaying plot")
        display(p)
    end
    return p
end

# `using PairPlots` (and Makie, for saving the resulting Figure) the first
# time a corner plot is requested, for the same reasons Turing is lazy.
function _ensure_pairplots!()
    if !isdefined(@__MODULE__, :PairPlots)
        _dbg("bayesian", "loading PairPlots (first use)...")
        # Bare `Makie` has no rendering backend of its own — `Makie.save`/`display`
        # need a concrete backend (GLMakie, CairoMakie, WGLMakie) loaded too, or you
        # get "No backend available!". CairoMakie is the right one here: it's a
        # lightweight, no-window (software-rendered) backend, good for saving a
        # static corner-plot PNG and for showing it in the VS Code plot pane —
        # unlike GLMakie, it doesn't need a real GPU window. `using CairoMakie`
        # registers itself as the active backend automatically.
        #
        # `import` rather than `using`: PairPlots/CairoMakie/Makie all export a
        # `plot`/`scatter` binding that collides with `Plots.jl`'s (a hard
        # dependency of this module, `using`-ed at the top), which raised an
        # "ambiguity" UndefVarError the moment `plot(...)` was called anywhere in
        # this module. `import` only binds the module names themselves
        # (`PairPlots`, `CairoMakie`, `Makie`), not their exports, and every call
        # site below already goes through a fully-qualified name
        # (`PairPlots.pairplot`/`.corner`, `Makie.save`), so nothing here needs
        # the exports anyway.
        try
            Base.eval(@__MODULE__, :(import PairPlots, CairoMakie, Makie))
        catch err
            # PairPlots depends on Makie internally, but that doesn't make a backend
            # `import`-able unless it's also a *direct* dependency of the active
            # project — plenty of environments have PairPlots without one. If that's
            # the problem, install it once and retry rather than failing outright.
            msg = sprint(showerror, err)
            if err isa ArgumentError && occursin("not found in current path", msg)
                _dbg("bayesian", "CairoMakie not found in this project; installing via Pkg (one-time)...")
                Base.eval(@__MODULE__, :(import Pkg; Pkg.add("CairoMakie")))
                Base.eval(@__MODULE__, :(import PairPlots, CairoMakie, Makie))
            else
                rethrow()
            end
        end
        _dbg("bayesian", "PairPlots loaded")
    end
    return nothing
end

# PairPlots.jl v2 renamed its main entry point from `corner` to `pairplot`;
# older installs only have `corner`. Pick whichever this version provides
# instead of hard-coding one name.
function _pairplots_plot(table)
    if isdefined(PairPlots, :pairplot)
        return PairPlots.pairplot(table)
    elseif isdefined(PairPlots, :corner)
        return PairPlots.corner(table)
    else
        error("Neither PairPlots.pairplot nor PairPlots.corner is defined — " *
              "unexpected PairPlots version; check `using Pkg; Pkg.status(\"PairPlots\")`.")
    end
end

"""
    corner_plot_linear(chain; outpath = nothing, show = true) -> Figure

Corner (pair) plot of the linear model's posterior: intercept, slope, σ.
If `outpath` is given, saves the figure there. If `show` is true (the
default), also `display`s it — see [`plot_data`](@ref) for why that's
needed to see it in VS Code. Loads PairPlots on first call.
"""
function corner_plot_linear(chain; outpath::Union{Nothing,AbstractString} = nothing, show::Bool = true)
    _ensure_pairplots!()
    return Base.invokelatest(_corner_plot_linear_impl, chain, outpath, show)
end

function _corner_plot_linear_impl(chain, outpath, show)
    _dbg("corner_plot_linear", "building corner plot")
    # PairPlots.corner expects plain vector columns; flatten any DimensionalData
    # wrapper (see _plot_bayesian_linear_impl) before building the table.
    table = (;
        intercept = vec(Array(chain[:intercept])),
        slope = vec(Array(chain[:slope])),
        σ = vec(Array(sqrt.(chain[:σ₂]))),
    )
    fig = _pairplots_plot(table)
    if outpath !== nothing
        Makie.save(outpath, fig)
        _dbg("corner_plot_linear", "saved ", outpath)
    end
    if show
        _dbg("corner_plot_linear", "displaying plot")
        display(fig)
    end
    return fig
end

"""
    corner_plot_quadratic(chain; outpath = nothing, show = true) -> Figure

Corner (pair) plot of the quadratic model's posterior: u1, u2, u3, σ.
If `outpath` is given, saves the figure there. If `show` is true (the
default), also `display`s it. Loads PairPlots on first call.
"""
function corner_plot_quadratic(chain; outpath::Union{Nothing,AbstractString} = nothing, show::Bool = true)
    _ensure_pairplots!()
    return Base.invokelatest(_corner_plot_quadratic_impl, chain, outpath, show)
end

function _corner_plot_quadratic_impl(chain, outpath, show)
    _dbg("corner_plot_quadratic", "building corner plot")
    # See _corner_plot_linear_impl: flatten any DimensionalData wrapper first.
    table = (;
        u_1 = vec(Array(chain[:u1])),
        u_2 = vec(Array(chain[:u2])),
        u_3 = vec(Array(chain[:u3])),
        σ = vec(Array(sqrt.(chain[:σ₂]))),
    )
    fig = _pairplots_plot(table)
    if outpath !== nothing
        Makie.save(outpath, fig)
        _dbg("corner_plot_quadratic", "saved ", outpath)
    end
    if show
        _dbg("corner_plot_quadratic", "displaying plot")
        display(fig)
    end
    return fig
end

# ---------------------------------------------------------------------------
# End-to-end drivers
# ---------------------------------------------------------------------------

"""
    run_demo(; dir = pwd(), seed = 1234, xrange = 0:5:100, force = false,
               size = (1600, 1000), dpi = 200) -> NamedTuple

Run the frequentist pipeline: load/generate data, fit a line by hand
(linear algebra), fit a line and a parabola with Optimization.jl, time an
autodiff+BFGS version, and save all the plots to `dir`. Fast; doesn't touch
Turing/PairPlots — see [`run_bayesian`](@ref) for that.
"""
function run_demo(; dir::AbstractString = pwd(), seed::Integer = 1234, xrange = 0:5:100,
                   force::Bool = false, size = (1600, 1000), dpi::Integer = 200)
    T0[] = time()
    _dbg("run_demo", "=== starting pipeline ===")
    mkpath(dir)

    _dbg("run_demo", "[1/6] load or generate data")
    x, y = load_or_generate_data(; dir = dir, seed = seed, xrange = xrange, force = force)

    _dbg("run_demo", "[2/6] plot raw data")
    savefig(plot_data(x, y; size = size, dpi = dpi), joinpath(dir, "data-scatter.png"))

    _dbg("run_demo", "[3/6] linear regression (linear algebra)")
    intercept, slope = linear_regression_matrix(x, y)
    yfit = linfunc.(x; slope, intercept)
    savefig(plot_fit(x, y, yfit; fit_label = "best fit", size = size, dpi = dpi),
            joinpath(dir, "linear-regression.png"))

    _dbg("run_demo", "[4/6] linear fit via Optimization.jl (NelderMead)")
    slope2, intercept2 = fit_linear_optim(x, y)
    yfit2 = linfunc.(x; slope = slope2, intercept = intercept2)
    savefig(plot_fit(x, y, yfit2; fit_label = "best fit", size = size, dpi = dpi),
            joinpath(dir, "optimization-linear-regression.png"))

    _dbg("run_demo", "[5/6] quadratic fit via Optimization.jl (NelderMead)")
    u, _ = fit_quadratic_optim(x, y)
    yfit3 = quadfunc.(x, Ref(u))
    savefig(plot_fit(x, y, yfit3; fit_label = "quadratic fit", size = size, dpi = dpi),
            joinpath(dir, "optimization-quad-regression.png"))

    _dbg("run_demo", "[6/6] autodiff + BFGS timing demo")
    slope3, intercept3, _ = fit_linear_autodiff(x, y)

    _dbg("run_demo", "=== pipeline complete ===")
    return (; x, y, intercept, slope, slope2, intercept2, u, slope3, intercept3)
end

"""
    run_bayesian(; dir = pwd(), seed = 1234, xrange = 0:5:100, force = false,
                   n = 500, corner_n = 25_000, target_accept = 0.65,
                   size = (1600, 1000), dpi = 200) -> NamedTuple

Run the Bayesian pipeline: reuse (or generate) the same data, fit Bayesian
linear and quadratic models with NUTS, save posterior spaghetti plots, then
redraw each with more samples (`corner_n`) and save corner plots via
PairPlots. Loads Turing and PairPlots on first use. Not fast — expect real
time for first-time precompilation and for the `corner_n`-sample resamples.
"""
function run_bayesian(; dir::AbstractString = pwd(), seed::Integer = 1234, xrange = 0:5:100,
                       force::Bool = false, n::Integer = 500, corner_n::Integer = 25_000,
                       target_accept::Real = 0.65, size = (1600, 1000), dpi::Integer = 200)
    T0[] = time()
    _dbg("run_bayesian", "=== starting Bayesian pipeline ===")
    mkpath(dir)

    _dbg("run_bayesian", "[1/8] load or generate data")
    x, y = load_or_generate_data(; dir = dir, seed = seed, xrange = xrange, force = force)

    _dbg("run_bayesian", "[2/8] fit Bayesian linear model (n=", n, ")")
    chain_lin = bayesian_fit_linear(x, y; n = n, target_accept = target_accept)

    _dbg("run_bayesian", "[3/8] plot linear posterior draws")
    savefig(plot_bayesian_linear(x, y, chain_lin; size = size, dpi = dpi),
            joinpath(dir, "bayesian-lin-regression.png"))

    _dbg("run_bayesian", "[4/8] re-sample linear model for corner plot (n=", corner_n, ")")
    chain_lin_big = bayesian_fit_linear(x, y; n = corner_n, target_accept = target_accept, seed = 1234)

    _dbg("run_bayesian", "[5/8] save linear corner plot")
    corner_plot_linear(chain_lin_big; outpath = joinpath(dir, "lin-regress-corner.png"))

    _dbg("run_bayesian", "[6/8] fit Bayesian quadratic model (n=", n, ")")
    chain_quad = bayesian_fit_quadratic(x, y; n = n, target_accept = target_accept)

    _dbg("run_bayesian", "[7/8] plot quadratic posterior draws")
    savefig(plot_bayesian_quadratic(x, y, chain_quad; size = size, dpi = dpi),
            joinpath(dir, "bayesian-quad-regression.png"))

    _dbg("run_bayesian", "[8/8] re-sample quadratic model and save corner plot (n=", corner_n, ")")
    chain_quad_big = bayesian_fit_quadratic(x, y; n = corner_n, target_accept = target_accept, seed = 1)
    corner_plot_quadratic(chain_quad_big; outpath = joinpath(dir, "quad-regress-corner.png"))

    _dbg("run_bayesian", "=== Bayesian pipeline complete ===")
    return (; chain_lin, chain_lin_big, chain_quad, chain_quad_big)
end

end # module

# Auto-run switches. VS Code's "Julia: Execute active file in REPL" (and
# `include`) does not set PROGRAM_FILE, so a script-only guard never fires
# there. Set AUTORUN_CURVEFIT to `false` if you want to load the module
# without running anything (e.g. to call the functions yourself).
AUTORUN_CURVEFIT = true

# The Bayesian pipeline is much slower (Turing precompile + NUTS sampling),
# so it's opt-in. Set to `true` if you want run_bayesian() to run too.
AUTORUN_BAYESIAN = false

if AUTORUN_CURVEFIT || abspath(PROGRAM_FILE) == @__FILE__
    println("[CurveFit] module loaded, starting run_demo() ...")
    flush(stdout)
    try
        CurveFit.run_demo()
    catch err
        println("[CurveFit] run_demo() failed:")
        println(sprint(showerror, err))
    end

    if AUTORUN_BAYESIAN
        println("[CurveFit] starting run_bayesian() (this can be slow — Turing/PairPlots precompile + NUTS sampling) ...")
        flush(stdout)
        try
            CurveFit.run_bayesian()
        catch err
            println("[CurveFit] run_bayesian() failed (set AUTORUN_BAYESIAN = false to skip it):")
            println(sprint(showerror, err))
        end
    end
end
