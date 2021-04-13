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
using Plots.PlotMeasures
using StatsPlots
using Turing

# load models and functions
include("models.jl")
include("functions.jl")

# load data
@load "../data/data.jld" x y x_new


# set seed
Random.seed!(1234)

# MCMC parameters
n_samples = 10_000
n_chains = 3

# fit models
post_quad = sample(quad_mod(x, y),
                   NUTS(),
                   MCMCThreads(),
                   n_samples,
                   n_chains)

post_exp2p = sample(exp2p(x, y),
                    NUTS(),
                    MCMCThreads(),
                    n_samples,
                    n_chains)

post_exp3p = sample(exp3p(x, y),
                    NUTS(),
                    MCMCThreads(),
                    n_samples,
                    n_chains)

# number of new x-values
n_new = size(x_new)[1]


# draw posterior predictive summaries for new data
quad_pred = turing_predict(post=post_quad, x_new=x_new, model="quad_mod")
exp2p_pred = turing_predict(post=post_exp2p, x_new=x_new, model="exp2p")
exp3p_pred = turing_predict(post=post_exp3p, x_new=x_new, model="exp3p")


# calculate (unweighted) model-averaged results
# draw posterior predictive samples for new data
quad_samps = turing_predict(post=post_quad, x_new=x_new, model="quad_mod", summary=false)
exp2p_samps = turing_predict(post=post_exp2p, x_new=x_new, model="exp2p", summary=false)
exp3p_samps = turing_predict(post=post_exp3p, x_new=x_new, model="exp3p", summary=false)

# calculate mean, 2.5% and 97.5% quantiles for model-averaged prediction
mod_avg = mean(vcat(quad_samps, exp2p_samps, exp3p_samps), dims=1)'
mod_avg_025 = [quantile(vcat(quad_samps[:, i], exp2p_samps[:, i], exp3p_samps[:, i]), 0.025)
               for i in 1:n_new]
mod_avg_975 = [quantile(vcat(quad_samps[:, i], exp2p_samps[:, i], exp3p_samps[:, i]), 0.975)
               for i in 1:n_new]


p1 = scatter(x, y, legend=false, color=:grey, markerstrokecolor=:grey, 
             xlim=(-0.03, 2.03), ylim=(-0.1, 1.7),  ylab="Clinical outcome",
             xlab="Assay result", title="A\nQuadratic", titleloc = :left, markersize=3)
p1 = plot!(x_new, quad_pred.means, ribbon=(quad_pred.means .- quad_pred.q_025,
                                        quad_pred.q_975 .- quad_pred.means),
      color=:firebrick, fillalpha=0.25, linewidth=1.5, linecolor=:firebrick)


p2 = scatter(x, y, legend=false, color=:grey, markerstrokecolor=:grey, markersize=3, 
             xlim=(-0.03, 2.03), ylim=(-0.1, 1.7),  ylab="Clinical outcome",
             xlab="Assay result", title="B\n2-Parameter Exponential", titleloc = :left)
p2 = plot!(x_new, exp2p_pred.means, ribbon=(exp2p_pred.means .- exp2p_pred.q_025,
                                         exp2p_pred.q_975 .- exp2p_pred.means),
      color=:green3, fillalpha=0.25, linewidth=1.5, linecolor=:green3)


p3 = scatter(x, y, legend=false, color=:grey, markerstrokecolor=:grey, markersize=3,
             xlim=(-0.03, 2.03), ylim=(-0.1, 1.7),  ylab="Clinical outcome",
             xlab="Assay result", title="C\n3-Parameter exponential", titleloc = :left)
p3 = plot!(x_new, exp3p_pred.means, ribbon=(exp3p_pred.means .- exp3p_pred.q_025,
                                         exp3p_pred.q_975 .- exp3p_pred.means),
           color=:steelblue, fillalpha=0.25, linewidth=1.5, linecolor=:steelblue)

p4 = scatter(x, y, legend=false, color=:grey, markerstrokecolor=:grey, markersize=3,
             xlim=(-0.03, 2.03), ylim=(-0.1, 1.7),  ylab="Clinical outcome",
             xlab="Assay result", title="D\nAll models", titleloc = :left)
p4 = plot!(x_new, quad_pred.means, linewidth=1.75, linecolor=:firebrick)
p4 = plot!(x_new, exp2p_pred.means, linewidth=1.75, linecolor=:green3)
p4 = plot!(x_new, exp3p_pred.means, linewidth=1.75, linecolor=:steelblue)


p5 = scatter(x, y, legend=false, color=:grey, markerstrokecolor=:grey, markersize=3,
             xlim=(-0.03, 2.03), ylim=(-0.1, 1.7),  ylab="Clinical outcome",
             xlab="Assay result", title="E\nModel averaged", titleloc = :left)
p5 = plot!(x_new, mod_avg, ribbon=(mod_avg .- mod_avg_025, mod_avg_975 .- mod_avg),
           color=:black, fillalpha=0.25, linewidth=1.5, linecolor=:black)

p6 = plot(x_new, quad_pred.q_975 - quad_pred.q_025, linewidth=1.5, linecolor=:firebrick,
          ylab="95% Prediction interval width", xlab="Assay result",
          title="F\nUncertainties", titleloc = :left, markersize=3,
          label="Quadratic", legend=:topleft)
p6 = plot!(x_new, exp2p_pred.q_975 - exp2p_pred.q_025, linewidth=1.5, linecolor=:green3,
           label="2-Parameter Exponential")
p6 = plot!(x_new, exp3p_pred.q_975 - exp3p_pred.q_025, linewidth=1.5, linecolor=:steelblue,
           label="3-Parameter Exponential")
p6 = plot!(x_new, mod_avg_975 - mod_avg_025, linewidth=1.5, linecolor=:black,
           label="Model averaged")


Plots.pdf(
    plot(p1, p2, p3, p4, p5, p6,
         size=(width=1050, height=700),
         left_margin=12px, 
         tick_direction=:out),
    "../figs/mod_avg.pdf"
)

