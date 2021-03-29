
"""
Inverse logit function.
"""
function invlogit(x)
    1/(1 + exp(-x))
end


"""
Calculates decision boundary for binomial model.

β0, β1, β2, β3 = model parameters
x = new x values for boundary
threshold = decision boundary threshold
"""
function boundary(β0, β1, β2, β3, x, threshold=0.5)
    threshold = -log(threshold / (1 - threshold))
    b = @. (threshold + β0 + β1 * x) / (-1 * β2 - β3 * x)
    return(b)
end


"""
Posterior predictions from a Turing model.

post = posterior samples from a model
x_new = new values for predictor variables 
model = name of a model defined in the models.jl file
summary = if true, return means and 95% PI, otherwise return full
          posterior predictive distribution
"""
function turing_predict(; post, x_new, model, summary=true)
    
    n_xs = length(x_new)

    if model == "quad_mod"
        pars = DataFrame(post)[:, [:θ₀, :θ₁, :θ₂, :σ]]
        pred = Array{Float64}(undef, size(pars)[1], n_xs)

        for i in 1:n_xs
            pred[:, i] = @. rand(Normal(pars.θ₀ + pars.θ₁ * x_new[i] + pars.θ₂ * x_new[i]^2,
                                        pars.σ))
        end

        means = [mean(pred[:, i]) for i in 1:n_xs]
        q_025 = [quantile(pred[:, i], 0.025) for i in 1:n_xs]
        q_975 = [quantile(pred[:, i], 0.975) for i in 1:n_xs]

        if summary
            return(means = means, q_025 = q_025, q_975 = q_975)
        else
            return(pred)
        end

    elseif model == "exp2p"
        pars = DataFrame(post)[:, [:θ₁, :θ₂, :σ]]
        pred = Array{Float64}(undef, size(pars)[1], n_xs)

        for i in 1:n_xs
            pred[:, i] = @. rand(Normal(pars.θ₂ * (1 - exp(-pars.θ₁ * x_new[i])), 
                                      pars.σ))
        end

        means = [mean(pred[:, i]) for i in 1:n_xs]
        q_025 = [quantile(pred[:, i], 0.025) for i in 1:n_xs]
        q_975 = [quantile(pred[:, i], 0.975) for i in 1:n_xs]

        if summary
            return(means = means, q_025 = q_025, q_975 = q_975)
        else
            return(pred)
        end


    elseif model == "exp3p"
        pars = DataFrame(post)[:, [:θ₁, :θ₂, :θ₃, :σ]]
        pred = Array{Float64}(undef, size(pars)[1], n_xs)

        for i in 1:n_xs
            pred[:, i] = @. rand(Normal(pars.θ₃ + pars.θ₂ * (1 - exp(-pars.θ₁ * x_new[i])), 
                                        pars.σ))
        end

        means = [mean(pred[:, i]) for i in 1:n_xs]
        q_025 = [quantile(pred[:, i], 0.025) for i in 1:n_xs]
        q_975 = [quantile(pred[:, i], 0.975) for i in 1:n_xs]

        if summary
            return(means = means, q_025 = q_025, q_975 = q_975)
        else
            return(pred)
        end

        

    elseif model == "exp2p_trunc"
        pars = DataFrame(post)[:, [:θ₁, :θ₂, :σ]]
        pred = Array{Float64}(undef, size(pars)[1], n_xs)

        for i in 1:n_xs
            #pred[:, i] = @. rand(truncated(Normal(pars.θ₂ * (1 - exp(-pars.θ₁ * x_new[i])), 
            #                                      pars.σ)), 0, Inf)
            pred[:, i] = @. rand(
                truncated(Normal(pars.θ₂ * (1 - exp(-pars.θ₁ * x_new[i])), 
                                 pars.σ), 0, Inf))
        end

        means = [mean(pred[:, i]) for i in 1:n_xs]
        q_025 = [quantile(pred[:, i], 0.025) for i in 1:n_xs]
        q_975 = [quantile(pred[:, i], 0.975) for i in 1:n_xs]

        if summary
            return(means = means, q_025 = q_025, q_975 = q_975)
        else
            return(pred)
        end
        
                
    elseif model == "exp3p_het"
        # extract parameters
        pars = DataFrame(post)[:, [:θ₁, :θ₂, :θ₃, :σ₀, :σ₁]]
        
        # arrays to store results
        pred = Array{Float64}(undef, size(pars)[1], n_xs)

        # calculate prediction
        for i in 1:n_xs
            μ = @. pars.θ₃ + pars.θ₂ * (1 - exp(-pars.θ₁ * x_new[i]))
            σ = @. log.(1 + exp.(pars.σ₀ .+ pars.σ₁ .* μ))
            pred[:, i] = @. rand(Normal(μ, σ))
        end
        
        # calculate summary stats
        means = [mean(pred[:, i]) for i in 1:n_xs]
        q_025 = [quantile(pred[:, i], 0.025) for i in 1:n_xs]
        q_975 = [quantile(pred[:, i], 0.975) for i in 1:n_xs]

        if summary
            return(means = means, q_025 = q_025, q_975 = q_975)
        else
            return(pred)
        end

    else
        println("Model not in list!")
    end
end
