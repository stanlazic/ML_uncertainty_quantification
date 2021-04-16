# activate environment
using Pkg
if isfile("../Project.toml") && isfile("../Manifest.toml")
    Pkg.activate("../")
end

using Random
using JLD
using DataFrames
using Plots
using Plots.PlotMeasures
using StatsPlots
using Turing

# load models and functions
include("models.jl")
include("functions.jl")

# load data
@load "../data/binary_data.jld"


# set seed
Random.seed!(1234)

n_samples = 10_000
n_chains = 3

# fit model
post = sample(binom(x₁, x₂, grp),
              NUTS(),
              MCMCThreads(),
              n_samples,
              n_chains)


# extract parameters
pars = DataFrame(post)[:, [:θ₀, :θ₁, :θ₂, :θ₃]]

# calculate decision boundary
x1_values = 0:0.01:0.79
boundary_line = boundary(mean(pars.θ₀),
                         mean(pars.θ₁),
                         mean(pars.θ₂),
                         mean(pars.θ₃),
                         x1_values, 0.5)

# calculate 95% CI for decision boundary
boundary_err = Array{Float64}(undef, size(pars)[1], length(x1_values))
for i in 1:size(pars)[1]
    boundary_err[i, :] = boundary(pars.θ₀[i], pars.θ₁[i], pars.θ₂[i], pars.θ₃[i],
                      x1_values, 0.5)
end

q_025 = [quantile(boundary_err[:, i], 0.025) for i in 1:length(x1_values)]
q_975 = [quantile(boundary_err[:, i], 0.975) for i in 1:length(x1_values)]


# calculate prediction
N = length(x₁)
# arrays to store results
pred = Array{Float64}(undef, size(pars)[1], N)

for i in 1:N
    pred[:, i] = @. invlogit(pars.θ₀ +
                             pars.θ₁*x₁[i] +
                             pars.θ₂*x₂[i] +
                             pars.θ₃*x₁[i]*x₂[i])
end


# calculate summary stats
means = [mean(pred[:, i]) for i in 1:N]
sds = [std(pred[:, i]) for i in 1:N]

## get posteriors for these key points
y_new1 = pred[:, 202]
y_new2 = pred[:, 11]


# Calculate aleatoric and epistemic uncertainty for each compound
alea = mean(pred .* (1 .- pred), dims=1)
epi = mean(pred.^2, dims=1) .- mean(pred, dims=1).^2

# ratio of epistemic uncertainty for the two compounds
epi[11] /  epi[202]


# colours for plotting
cols = Array{String}(undef, N)
for i in 1:N
    if sds[i] > 0.08
        cols[i] = "firebrick"
    elseif sds[i] > 0.06
        cols[i] = "orange3"
    else
        cols[i] = "white"
    end
end


# Text labels for plotting
labels = ifelse.(grp .== 0, "Safe", "Toxic")


p1 = scatter(x₁, x₂, ylab="x₂", xlab="x₁", group=labels, 
             legend=:topleft, markersize=5, title="A\nData", titleloc = :left, 
             markershape=ifelse.(grp .== 0, :circle, :utriangle), 
             color=ifelse.(grp .== 0, :steelblue, :grey30),
             markerstrokecolor=ifelse.(grp .== 0, :steelblue, :grey30),
             ylim=(-0.05, 1.025), xlim=(-0.05, 1.025))
p1 = plot!(x1_values, boundary_line, color=:black, linewidth=2, label=:none)

p2 = scatter(means, sds, xlab="μ", ylab="σ",
             title="B\nMean - variance relationship", titleloc = :left, 
             legend=:none, markersize=5, color=cols, ylim=(-0.005, 0.16),
             markershape=ifelse.(grp .== 0, :circle, :utriangle))

p2 = scatter!(means[[11, 202]], sds[[11, 202]], markersize=13, markershape=:circle,
              markeralpha=0.25, markercolor=:green3, 
              markerstrokestyle=:dash)

p3 = scatter([], [], ylab="x₂",  xlab="x₁", legend=:none,
             markersize=5, title="C\nPrediction uncertainty", titleloc = :left,
             color=cols, ylim=(-0.05, 1.025), xlim=(-0.05, 1.025))
p3 = plot!(x1_values, boundary_line, color=:black, linewidth=1.5)

p3 = plot!(x1_values, boundary_line, 
           ribbon=(boundary_line .- q_025, q_975 .- boundary_line), 
           color=:black, fillalpha=0.3)
p3 = scatter!(x₁, x₂, markersize=5, color=cols,
              markershape=ifelse.(grp .== 0, :circle, :utriangle))

p3 = scatter!(x₁[[11, 202]], x₂[[11, 202]], markersize=13, markershape=:circle,
              markeralpha=0.25, markercolor=:green3, 
              markerstrokestyle=:dash)

p4 = density(y_new1, xlim=(0, 1), color=:orange3, fill=0, fillalpha=0.3,
             title="D\nTwo predictions with same mean",
             label="μ = 0.74; σ = 0.06", 
             titleloc = :left, xlabel="Prediction for μ", legend=:topleft)

p4 = density!(y_new2, color=:firebrick, fill=0, fillalpha=0.3,
              label="μ = 0.74; σ = 0.12")

p5 = bar(["Safe", "Toxic"], [0.26, 0.74], color=:orange3, fill=0, fillalpha=0.3,
         ylim=(0, 1), xlim=(-0.5, 2.5), linecolor=:orange3, bar_width=0.5, label=:none, 
         title="E\nMore certain prediction (σ=0.06)", titleloc=:left, ylab="P(Class | Data)")

p6 = bar(["Safe", "Toxic"], [0.26, 0.74], color=:firebrick, fill=0, fillalpha=0.3,
         ylim=(0, 1), xlim=(-0.5, 2.5), linecolor=:firebrick, bar_width=0.5,
         label=:none, title="F\nUncertain prediction (σ=0.12)", titleloc=:left,
         ylab="P(Class | Data)")

p7 = scatter(means, alea', xlab="μ", ylab="Aleatoric uncertainty",
             title="G\nMean - uncertainty relationship", titleloc = :left, 
             legend=:none, markersize=5, color=cols, ylim=(-0.01, 0.4),
             markershape=ifelse.(grp .== 0, :circle, :utriangle))

p7 = scatter!(means[[11, 202]], alea[[11, 202]], markersize=13, markershape=:circle,
              markeralpha=0.25, markercolor=:green3, 
              markerstrokestyle=:dash)

p8 = bar(["σ=0.06"], [alea[202]], color=:orange3, fill=0, fillalpha=0.3, label=:none, 
         ylim=(0, 0.25), xlim=(-0.5, 2.5), linecolor=:orange3, bar_width=0.5,
         title="H\nAleatoric uncertainty", titleloc=:left, ylab="Uncertainty")

p8 = bar!(["σ=0.12"], [alea[11]], color=:firebrick, fill=0, fillalpha=0.3, label=:none,
          linecolor= :firebrick, bar_width=0.5)

p9 = bar(["σ=0.06"], [epi[202]], color=:orange3, fill=0, fillalpha=0.3,
         ylim=(0, 0.02), xlim=(-0.5, 2.5), linecolor=:orange3, bar_width=0.5, label=:none, 
         title="I\nEpistemic uncertainty", titleloc=:left, ylab="Uncertainty")

p9 = bar!(["σ=0.12"], [epi[11]], color=:firebrick, fill=0, fillalpha=0.3, label=:none,
          linecolor= :firebrick, bar_width=0.5)


Plots.pdf(
    plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, 
         left_margin=5px,
         bottom_margin=10px,
         top_margin=15px,
         size=(width=1050, height=1050),
         tick_direction=:out),
    "../figs/binom_uncertainty.pdf")
