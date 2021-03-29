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
using LaTeXStrings

# load models and functions
include("models.jl")
include("functions.jl")

# load data
@load "../data/data.jld" x y me_x me_y x1 x2 x3 x4 x5 y1 y2 y3 y4 y5


# set seed
Random.seed!(1234)

# MCMC parameters
n_samples = 10000
n_chains = 3

# fit 3P exponential model (no measurement error)
exp3p_fit0 = sample(exp3p(x, y),
                    NUTS(),
                    MCMCThreads(),
                    n_samples,
                    n_chains)

# prediction ignoring noise in x_new = 0.15
post0_pred = turing_predict(; post=exp3p_fit0,
                            x_new=0.15,
                            model="exp3p", summary=false)

# prediction with noise in x_new
exp3p_pred_err = turing_predict(; post=exp3p_fit0,
                                x_new=rand(Normal(0.15, 0.06), 1000),
                                model="exp3p", summary=false)


# plot data with uncertainty
p1 = scatter(x, y, legend=:bottomright, color=:black, markerstrokecolor=:black, 
             xlim=(-0.05, 1.03), ylim=(-0.1, 1.5), label="Training data", 
             ylab="Clinical outcome", xlab="Assay result",
             title="A\nMeasurement error", titleloc = :left, markersize=3,
             xerror=(me_x, me_x), yerror=(me_y, me_y), arrows=false)
p1 = scatter!([0.15], [0], xerror=([0.06], [0.06]), arrows=false,
               color=:firebrick, markerstrokecolor=:firebrick, label="Test data")


p2 = density(post0_pred, color=:steelblue, fill=0, fillalpha=0.3,
             title="B\nError in test data", titleloc = :left,
             xlabel=L"P(y\, | x = 0.15)", label="x = 0.15",
             xlim=(0, 1.1), ylim=(0, 4))

p2 = density!(vec(exp3p_pred_err), color=:firebrick, fill=0, fillalpha=0.3,
               label="x ~ Normal(0.15, 0.06)")



# account for error in training data
# fit model to each dataset 
exp3p_fit1 = sample(exp3p(x1, y1), NUTS(), MCMCThreads(), n_samples, n_chains)
exp3p_fit2 = sample(exp3p(x2, y2), NUTS(), MCMCThreads(), n_samples, n_chains)
exp3p_fit3 = sample(exp3p(x3, y3), NUTS(), MCMCThreads(), n_samples, n_chains)
exp3p_fit4 = sample(exp3p(x4, y4), NUTS(), MCMCThreads(), n_samples, n_chains)
exp3p_fit5 = sample(exp3p(x5, y5), NUTS(), MCMCThreads(), n_samples, n_chains)


# draw posterior predictive summaries for new data
exp3p_pred1  = turing_predict(; post=exp3p_fit1, x_new=0.15,
                              model="exp3p", summary=false)
exp3p_pred2  = turing_predict(; post=exp3p_fit2, x_new=0.15,
                              model="exp3p", summary=false)
exp3p_pred3  = turing_predict(; post=exp3p_fit3, x_new=0.15,
                              model="exp3p", summary=false)
exp3p_pred4  = turing_predict(; post=exp3p_fit4, x_new=0.15,
                              model="exp3p", summary=false)
exp3p_pred5  = turing_predict(; post=exp3p_fit5, x_new=0.15,
                              model="exp3p", summary=false)

# combine predictions across datasets
combined_pred = vcat(exp3p_pred1, exp3p_pred2, exp3p_pred3, exp3p_pred4, exp3p_pred5)


# original dataset
p3 = density(post0_pred, color=:steelblue, fill=0, fillalpha=0.3,
              title="C\nError in training data",
              titleloc = :left, xlabel=L"P(y\, | x = 0.15)", label="No error model",
              xlim=(0, 1.1), ylim=(0, 4))

# new datasets
p3 = density!(exp3p_pred1, color=:firebrick, label="Dataset 1")
p3 = density!(exp3p_pred2, color=:firebrick, label="Dataset 2")
p3 = density!(exp3p_pred3, color=:firebrick, label="Dataset 3")
p3 = density!(exp3p_pred4, color=:firebrick, label="Dataset 4")
p3 = density!(exp3p_pred5, color=:firebrick, label="Dataset 5")


# original dataset
p4 = density(post0_pred, color=:steelblue, fill=0, fillalpha=0.3,
              title="D\nError in training data",
              titleloc = :left, xlabel=L"P(y\, | x = 0.15)", label="No error model",
              xlim=(0, 1.1), ylim=(0, 4))

# combined new dataset
p4 = density!(combined_pred, color=:firebrick, fill=0, fillalpha=0.3,
                label="Error model")


Plots.pdf(
    plot(p1, p2, p3, p4,
         size=(width=700, height=600),
         tick_direction=:out),
    "../figs/measurement_error.pdf")
