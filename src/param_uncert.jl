# activate environment
using Pkg
if isfile("../Project.toml") && isfile("../Manifest.toml")
    Pkg.activate("../")
end

using Random
using JLD
using DataFrames
using GLM
using Plots
using StatsPlots
using Turing

# load models and functions
include("models.jl")
include("functions.jl")

# load data
@load "../data/data.jld" x_small y_small x_new


# quadratic (Non-Bayesian) linear model and prediction
mod = lm(@formula(y ~ x + x^2), DataFrame(x=x_small, y=y_small))
mod_preds = predict(mod, DataFrame(x=x_new), interval=:prediction)


# set seed
Random.seed!(1234)

# MCMC parameters
n_samples = 20_000
n_chains = 3

# fit Bayesian model
post = sample(quad_mod(x_small, y_small),
              NUTS(),
              MCMCThreads(),
              n_samples,
              n_chains)


# predictions for new x-values
post_pred = turing_predict(post=post, x_new=x_new, model="quad_mod")

# set up plot
p = scatter([], [], legend=false, color=:steelblue,
            xlim=(-0.03, 1.03), ylim=(-0.1, 1.6),
            ylab="Clinical outcome", xlab="Assay result")

# add Bayesian results
p = plot!(collect(x_new), post_pred.means, color=:lightgrey, fillalpha=0.75, 
          ribbon=(post_pred.means .- post_pred.q_025,
                  post_pred.q_975 .- post_pred.means))

# add Frequentist results
p = plot!(x_new, mod_preds.prediction, linecolor=:steelblue, linewidth=2)
p = plot!(x_new, mod_preds.lower, linecolor=:black, linestyle=:dash)
p = plot!(x_new, mod_preds.upper, linecolor=:black, linestyle=:dash)

## add points
p = scatter!(x_small, y_small, color=:steelblue, markerstrokecolor=:white,
              markersize=5, markerstrokewidth = 1.25)

# save plot
Plots.pdf(
    plot(p,
         size = (width=400, height=400),
         tick_direction=:out), 
    "../figs/parameter_uncertainty.pdf")
