# activate environment
using Pkg
if isfile("../Project.toml") && isfile("../Manifest.toml")
    Pkg.activate("../")
end

using Random
using JLD
using DataFrames
using Plots
using StatsPlots
using Turing

# load models and functions
include("models.jl")
include("functions.jl")

# load data
@load "../data/data.jld" x y x_new


# set seed
Random.seed!(1234)

n_samples = 10000
n_chains = 3

# fit standard model
post_exp2p = sample(exp2p(x, y),
                    NUTS(),
                    MCMCThreads(),
                    n_samples,
                    n_chains)

# fit truncated model
post_exp2p_trunc = sample(exp2p_trunc(x, y),
                          NUTS(),
                          MCMCThreads(),
                          n_samples,
                          n_chains)


# draw posterior predictive summaries for new data
exp2p_pred = turing_predict(; post=post_exp2p, x_new=x_new,
                            model="exp2p", summary=true)

exp2p_pred_trunc = turing_predict(; post=post_exp2p_trunc, x_new=x_new,
                                  model="exp2p_trunc", summary=true)


# draw posterior predictive samples for x=0.05
y_est = turing_predict(; post=post_exp2p, x_new=0.05,
                       model="exp2p", summary=false)

y_est_trunc = turing_predict(; post=post_exp2p_trunc, x_new=0.05,
                             model="exp2p_trunc", summary=false)


p1 = scatter(x, y, legend=false, color=:grey, markerstrokecolor=:grey, markersize=3, 
             xlim=(-0.03, 1), ylim=(-0.3, 1.5),
             ylab="Clinical outcome", xlab="Assay result",
             title="A\n2-P Exponential", titleloc = :left)
p1 = plot!(x_new, exp2p_pred.means,
           ribbon=(exp2p_pred.means .- exp2p_pred.q_025,
                   exp2p_pred.q_975 .- exp2p_pred.means),
           color=:green3, fillalpha=0.25, linewidth=1.5, linecolor=:green3)
p1 = hline!([0], color=:firebrick, linestyle=:dash)


p2 = scatter(x, y, legend=false, color=:grey, markerstrokecolor=:grey, markersize=3, 
             xlim=(-0.03, 1), ylim=(-0.3, 1.5),
             ylab="Clinical outcome", xlab="Assay result",
             title="B\nTruncated 2-P Exponential", titleloc = :left)
p2 = plot!(x_new, exp2p_pred_trunc.means,
           ribbon=(exp2p_pred_trunc.means .- exp2p_pred_trunc.q_025,
                   exp2p_pred_trunc.q_975 .- exp2p_pred_trunc.means),
           color=:orange3, fillalpha=0.25, linewidth=1.5, linecolor=:orange3)
p2 = hline!([0], color=:firebrick, linestyle=:dash)


p3 = density(y_est, color=:green3, fill=0, fillalpha=0.3, title="C",
              titleloc = :left, xlabel="P(y | x = 0.05)", legend=:none,
              xlim=(-0.4, 0.9), ylim=(0, 4))
p3 = vline!([0], color=:firebrick, linestyle=:dash)


p4 = density(y_est_trunc, color=:orange3, fill=0, fillalpha=0.3, title="D",
              titleloc = :left, xlabel="P(y | x = 0.05)", xlim=(-0.4, 0.9),
              legend=:none, ylim=(0, 4))
p4 = vline!([0], color=:firebrick, linestyle=:dash)


Plots.pdf(
    plot(p1, p2, p3, p4,
         size=(width=700, height=600),
         tick_direction=:out),
    "../figs/truncated.pdf"
)

# proportion of the distribution below zero
mean(y_est .< 0) # 9.3%
mean(y_est_trunc .< 0) # 0%
