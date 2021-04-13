
"""
Quadratic model
"""
@model quad_mod(x, y) = begin
    # priors
    θ₀ ~ Normal(0, 5)
    θ₁ ~ Normal(0, 10)
    θ₂ ~ Normal(0, 10)
    σ ~ truncated(Normal(0, 5), 0, Inf)

    # likelihood
    @. y ~ Normal(θ₀ + θ₁ * x + θ₂ * x^2, σ)
end 


"""
2 parameter exponential model
"""
@model exp2p(x, y) = begin
    # priors
    θ₁ ~ truncated(Normal(1, 5), 0, Inf)
    θ₂ ~ truncated(Normal(0, 5), 0, Inf)
    σ ~ truncated(Normal(0, 5), 0, Inf)

    # likelihood
    @. y ~ Normal(θ₂ * (1 - exp(-θ₁ * x)), σ)
end 


"""
3 parameter exponential model
"""
@model exp3p(x, y) = begin
    # priors
    θ₁ ~ truncated(Normal(1, 5), 0, Inf)
    θ₂ ~ truncated(Normal(0, 5), 0, Inf)
    θ₃ ~ truncated(Normal(0, 2), 0, Inf)
    σ ~ truncated(Normal(0, 5), 0, Inf)

    # likelihood
    @. y ~ Normal(θ₃ + θ₂ * (1 - exp(-θ₁ * x)), σ)
end 


"""
truncated 2-parameter exponential model
"""
@model exp2p_trunc(x, y) = begin
    # priors
    θ₁ ~ truncated(Normal(1, 5), 0, Inf)
    θ₂ ~ truncated(Normal(0, 5), 0, Inf)
    σ ~ truncated(Normal(0, 5), 0, Inf)

    # likelihood
    @. y ~ truncated(Normal(θ₂ * (1 - exp(-θ₁ * x)), σ), 0, Inf)
end 


"""
3 parameter exponential model with variance function
"""
@model exp3p_het(x, y) = begin
    # priors
    θ₁ ~ truncated(Normal(1, 5), 0, Inf)
    θ₂ ~ truncated(Normal(0, 5), 0, Inf)
    θ₃ ~ truncated(Normal(0, 2), 0, Inf)
    σ₀ ~ Normal(0, 10)
    σ₁ ~ Normal(0, 10)

    # likelihood
    μ = θ₃ .+ θ₂ .* (1 .- exp.(-θ₁ .* x))
    σ = log.(1 .+ exp.(σ₀ .+ σ₁ .* μ))
    @. y ~ Normal(μ, σ)

    return θ₁, θ₂, θ₃, μ, σ
end


"""
binomial model
"""
@model binom(x1, x2, y) = begin
    # priors
    θ₀ ~ Normal(0, 5)
    θ₁ ~ Normal(0, 20)
    θ₂ ~ Normal(0, 20)
    θ₃ ~ Normal(0, 20)
    
    # likelihood
    η = @. invlogit(θ₀ + θ₁*x1 + θ₂*x2 + θ₃ * x1 * x2)
    @. y ~ Bernoulli(η)
end 

