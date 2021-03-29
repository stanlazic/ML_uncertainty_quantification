# activate environment
using Pkg
if isfile("../Project.toml") && isfile("../Manifest.toml")
    Pkg.activate("../")
end

using Plots
using StatsPlots
using Distributions
using LaTeXStrings


# generate x-values
x_vals = -4:0.1:4

# pre-allocate output arrays
y_logit = similar(x_vals)
y_probit = similar(x_vals)
y_cll = similar(x_vals)
y_cauchit = similar(x_vals)

@. y_logit  = 1/(1 + exp(-x_vals))
@. y_probit = cdf(Normal(0, pi / sqrt(3)), x_vals)
@. y_cll = 1 - exp(-exp(x_vals))
@. y_cauchit = 0.5 + atan(x_vals)/pi



p1 = plot(x_vals, y_logit, legend=:topleft, label="logit", color=:steelblue,
          linewidth=1.5, xlab=L"\stackrel{{}}{\mu}", ylab=L"\stackrel{l(\mu)}{{}}",
          title="Link functions", titleloc = :left, framestyle = :box, guidefont=(20))
p1 = plot!(x_vals, y_probit, label="probit", color=:firebrick, linewidth=1.5)
p1 = plot!(x_vals, y_cll, label="cloglog", color=:green3, linewidth=1.5)
p1 = plot!(x_vals, y_cauchit, label="cauchit", color=:orange3, linewidth=1.5)

Plots.pdf(
    plot(p1,
         size=(width=400, height=400),
         tick_direction=:out),
    "../figs/link_functions.pdf")

