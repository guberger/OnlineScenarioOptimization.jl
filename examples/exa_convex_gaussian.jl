module ExampleConvexGaussian

using Distributions
using Gurobi
using JuMP
using LinearAlgebra
using OnlineScenarioOptimization
using Plots
using Random

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

const dim_x = 20

sample_constraints(N) = [randn(dim_x) for _ = 1:N]
function solve_problem(con_list)
    model = solver()
    set_silent(model)
    x = @variable(model, [1:dim_x], lower_bound=-100, upper_bound=100)
    for v in con_list
        @constraint(model, dot(v, x) ≤ 1)
    end
    @objective(model, Min, sum(x))
    optimize!(model)
    @assert primal_status(model) == FEASIBLE_POINT
    return value.(x)
end
compute_risk(x) = ccdf(Normal(), 1 / norm(x))

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

plt = plot(plt1, plt2, plt3, plt4, plt5, plt6, layout=6)
display(plt)

end # module