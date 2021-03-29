# activate environment
using Pkg
if isfile("../Project.toml") && isfile("../Manifest.toml")
    Pkg.activate("../")
end

using JLD
using DataFrames
using Plots
using StatsPlots
using Turing

# load models and functions
include("models.jl")
include("functions.jl")

# load data
@load "../data/data.jld" x y x_new y_het



# set seed
Random.seed!(1234)

# MCMC parameters
n_samples = 10000
n_chains = 3

# fit standard model
post_exp3p = sample(exp3p(x, y_het),
                    NUTS(),
                    MCMCThreads(),
                    n_samples,
                    n_chains)

# fit variance model
post_exp3p_het = sample(exp3p_het(x, y_het),
                        NUTS(),
                        MCMCThreads(),
                        n_samples,
                        n_chains)


# draw posterior predictive summaries for new data
post = turing_predict(; post=post_exp3p, x_new=x_new, model="exp3p")
post_het = turing_predict(; post=post_exp3p_het, x_new=x_new, model="exp3p_het")


p0 = scatter(x, y_het, legend=false, color=:steelblue, markerstrokecolor=:steelblue, 
             xlim=(-0.03, 1.35), ylim=(-0.1, 1.8), ylab="Clinical  outcome",
             xlab="Assay result", #size = (width=400, height=400),
             title="A\nConstant variance", titleloc = :left, markersize=3)


p0 = plot!(x_new, post.means,
           ribbon=(post.means .- post.q_025, post.q_975 .- post.means),
           color=:firebrick, fillalpha=0.25, linewidth=1.5, linecolor=:firebrick)


p1 = scatter(x, y_het, legend=false, color=:steelblue, markerstrokecolor=:steelblue, 
             xlim=(-0.03, 1.35), ylim=(-0.1, 1.8),
             ylab="Clinical outcome", xlab="Assay result",
             title="B\nWith variance function", titleloc = :left, markersize=3)

p1 = plot!(x_new, post_het.means,
           ribbon=(post_het.means .- post_het.q_025, post_het.q_975 .- post_het.means),
           color=:firebrick, fillalpha=0.25, linewidth=1.5, linecolor=:firebrick)


Plots.pdf(
    plot(p0, p1,
         size=(width=700, height=350),
         tick_direction=:out),
    "../figs/hetero.pdf"
)

