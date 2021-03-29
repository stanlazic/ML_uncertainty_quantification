# activate environment
using Pkg
if isfile("../Project.toml") && isfile("../Manifest.toml")
    Pkg.activate("../")
end

using JLD
using Random
using Distributions
using Plots
using StatsPlots
using LaTeXStrings

include("functions.jl")

## ------------------------------------------------------------
## Simulate data
## ------------------------------------------------------------

Random.seed!(1234)

# generate x values uniformly between 0 and 1
N = 100
x = rand(N) |> sort


# define true data generating model
true_model = function(x, θ₁, θ₂)
    θ₂ + (1 - exp(-θ₁ * x)) / (1 + exp(-θ₁ * x))
end

# set model parameters
θ₁ = 3.25
θ₂ = 0.2
σ = 0.1

# generate y values
y_hat = true_model.(x, θ₁, θ₂)

# add noise to generate observed values
y = rand.(Normal.(y_hat, σ))

# new x-values for plotting predictions
x_new = 0:0.01:2



# plot values and label parts
p = scatter(x, y, legend=false, color=:steelblue, markerstrokecolor=:steelblue, 
             xlim=(-0.03, 1.35), ylim=(-0.1, 1.4), ylab=L"\mathrm{Clinical\ outcome}\ (y)",
             xlab=L"\mathrm{Assay\ result}\ (x)")

# add true line
p = plot!(x, true_model.(x, θ₁, θ₂), linewidth=2, color=:orange3)

# add info/annotations
p = plot!([0.8, x[55]], [0.4, y_hat[55]], line=(:arrow), color=:black, linewidth = 2)
p = plot!([1.05, 1.05], [0.917, 1.317], color=:black, linewidth=2)
p = plot!([1.05, 1], [0.917, 0.917], color=:black, linewidth=2)
p = plot!([1.05, 1], [1.317, 1.317], color=:black, linewidth=2)
p = annotate!(0.8, 0.35, L"\mu = f_{\mu}(x; \theta_{\mu})")
p = annotate!(1.1, y_hat[100], L"G(\mu, \sigma)", :left)

Plots.pdf(
    plot(p,
         size = (width=400, height=400),
         tick_direction=:out),
    "../figs/simulated_data.pdf")


## ------------------------------------------------------------
## Parameter uncertainty
## ------------------------------------------------------------

# take a subset --> reduced N highlights parameter uncertainty better
x_small = x[1:8:N]
y_small = y[1:8:N]


## ------------------------------------------------------------
## Data uncertainty
## ------------------------------------------------------------

Random.seed!(1234)

# generate random errors for x uniformly between 0.01 and 0.06
me_x = rand(0.01:0.0001:0.06, N) 

# errors for y are constant
me_y = repeat([0.02], N)


# generate 5 new data sets
Random.seed!(123)
y1 = [rand(Normal(y[i], me_y[i])) for i in 1:N]
x1 = [rand(Normal(x[i], me_x[i])) for i in 1:N]

y2 = [rand(Normal(y[i], me_y[i])) for i in 1:N]
x2 = [rand(Normal(x[i], me_x[i])) for i in 1:N]

y3 = [rand(Normal(y[i], me_y[i])) for i in 1:N]
x3 = [rand(Normal(x[i], me_x[i])) for i in 1:N]

y4 = [rand(Normal(y[i], me_y[i])) for i in 1:N]
x4 = [rand(Normal(x[i], me_x[i])) for i in 1:N]

y5 = [rand(Normal(y[i], me_y[i])) for i in 1:N]
x5 = [rand(Normal(x[i], me_x[i])) for i in 1:N]


## ------------------------------------------------------------
## Variance function uncertainty
## ------------------------------------------------------------

Random.seed!(1234)

# variance function
sigma_hat = 0.01 .+ 0.15 .* x

# add noise
y_het = [rand(Normal(y_hat[i], sigma_hat[i])) for i in 1:N]

# save variables
@save "../data/data.jld" x y N θ₁ θ₂ σ y_hat x_new x_small y_small me_x me_y y_het x1 x2 x3 x4 x5 y1 y2 y3 y4 y5


## ------------------------------------------------------------
## Binary prediction
## ------------------------------------------------------------


Random.seed!(1234)

# number of samples
N = 500

# generate x values uniformly between 0 and 1
x₁ = rand(N)
x₂ = rand(N)

# set model parameters
θ₀  = 0.35
θ₁ = 1.5
θ₂ = -2.3
θ₃ = 2.0

# generate linear predictor 
grp = @. invlogit(θ₀ + θ₁ * x₁ + θ₂ * x₂ + θ₃ * x₁ * x₂)

# generate binary y values
grp = grp .+ rand(Normal(0, 0.075), N) .> 0.6

# filter samples to make x₁ and x₂ correlated
pkeep = 1 .- abs.(x₁ - x₂) .> rand(N) .+ 0.2

x₁ = x₁[pkeep]
x₂ = x₂[pkeep]
grp = grp[pkeep]

# sample size of remaining samples
N = length(x₁)


@save "../data/binary_data.jld" x₁ x₂ grp θ₁ θ₂ θ₃

