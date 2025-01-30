module ExampleOneDimensional

using Distributions
using OnlineScenarioOptimization
using Plots
using Random

const D_con = Normal(1, 2)

sample_constraints(N) = [rand(D_con) for _ = 1:N]
solve_problem(con_list) = maximum(con_list)
compute_risk(x) = ccdf(D_con, x)

################################################################################
function generate_risk_samples(N_set, n_sample)
    samples = Vector{RiskSample}(undef, n_sample)
    for k = 1:n_sample
        N = rand(N_set)
        con_list = sample_constraints(N)
        x_opt = solve_problem(con_list)
        ϵ = compute_risk(x_opt)
        samples[k] = RiskSample(N, ϵ, ϵ^2)
    end
    return samples
end

N = 100
n_sample = 100
samples = generate_risk_samples((N,), n_sample)

risk_list = [sample.ϵ for sample in samples]
θ, success = fit_parameter(samples)
@assert success
display(θ)
D_hat = Beta(θ, N - θ + 1)
display(D_hat)
D_Beta = fit_mle(Beta, risk_list)
display(D_Beta)

xs = range(0, 1, length=1000)
plt1 = histogram(risk_list, normalize=:pdf, label=false)
plot!(plt1, xs, pdf.(D_hat, xs), c=:red, label=false)
plot!(plt1, xs, pdf.(D_Beta, xs), c=:green, label=false)

N_set = 64:511
n_sample = 100
samples = generate_risk_samples(N_set, n_sample)

size_list = [sample.N for sample in samples]
risk_list = [sample.ϵ for sample in samples]
θ, success = fit_parameter(samples)
@assert success
display(θ)

plt2 = scatter(size_list, risk_list, label=false)

################################################################################
ϵ_max = 0.1
β = 0.9
size_list = [100]
risk_list = Float64[]
samples = RiskSample[]
n_step = 1000
θ_list = Float64[]

for t = 1:n_step
    con_list = sample_constraints(size_list[t])
    x_opt = solve_problem(con_list)
    ϵ = compute_risk(x_opt)
    push!(risk_list, ϵ)
    push!(samples, RiskSample(size_list[t], ϵ, ϵ^2))
    θ, success = fit_parameter(samples)
    @assert success
    push!(θ_list, θ)
    N, success = find_size(θ, β, ϵ_max, 10, 1000)
    @assert success
    if t < n_step
        push!(size_list, N)
    end
end

plt3 = plot(size_list, label=false)
plt4 = plot(risk_list, label=false)
plt5 = plot(θ_list, label=false)
sort!(risk_list)
cdf_list = collect(1:length(risk_list)) / length(risk_list)
plt6 = plot(risk_list, cdf_list, label=false)
hline!(plt6, [β])
vline!(plt6, [ϵ_max])

θ_max = minimum(size_list) * 1.0
θ_min = 1.0
θ_range = range(θ_min, θ_max, 100)
llh_range = map(θ -> loglikelihood(samples, θ), θ_range)
θ_init = OnlineScenarioOptimization._initial_guess(samples)
plt7 = plot(θ_range, llh_range)
vline!(plt7, [θ_init])
vline!(plt7, [θ_list[end]])

plt = plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, layout=7)
display(plt)

end # module