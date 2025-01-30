module ExamplePathlanning

using Distributions
using Gurobi
using JuMP
using OnlineScenarioOptimization
using Plots
using Random

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

const D_con = Normal(1.5, 0.1)
const H = 100
const _M_ = 100
const Dy = 0.3
const Dx = 0.045

sample_constraints(N) = [rand(D_con) for _ = 1:N]
function solve_problem(con_list)
    δ = 0.01
    model = solver()
    x_list = [@variable(model, [1:2]) for _ = 1:(H + 1)]
    for x in x_list
        @constraint(model, x ≥ [0, 0])
        @constraint(model, x ≤ [5, 3])
    end
    @constraint(model, x_list[1] == [0.5, 0.5])
    for x in x_list
        b1 = @variable(model, binary=true)
        b2 = @variable(model, binary=true)
        b3 = @variable(model, binary=true)
        @constraint(model, b1 + b2 + b3 ≥ 1)
        @constraint(model, x[1] ≤ 2 - δ + _M_ * (1 - b1))
        @constraint(model, x[1] ≥ 3 + δ - _M_ * (1 - b2))
        for y in con_list
            @constraint(model, x[2] ≥ y - Dy - _M_ * (1 - b3))
            @constraint(model, x[2] ≤ y + Dy + _M_ * (1 - b3))
        end
    end
    for i = 1:H
        @constraint(model, [Dx; x_list[i] - x_list[i + 1]] in SecondOrderCone())
    end
    for i = 2:H
        a = x_list[i + 1] + x_list[i - 1] - 2 * x_list[i]
        @constraint(model, [0.001; a] in SecondOrderCone())
    end
    e = @variable(model, lower_bound=0)
    xT = [4.5, 0.5]
    @constraint(model, x_list[H + 1] - xT ≤ +e * ones(2))
    @constraint(model, x_list[H + 1] - xT ≥ -e * ones(2))
    @objective(model, Min, sum(e))
    optimize!(model)
    @assert primal_status(model) == FEASIBLE_POINT
    return map(x -> value.(x), x_list)
end
function compute_risk(x_list)
    y_min = +Inf
    y_max = -Inf
    for x in x_list
        if 2 ≤ x[1] ≤ 3
            y_min = min(x[2], y_min)
            y_max = max(x[2], y_max)
        end
    end
    display(y_min)
    display(y_max)
    return min(0.0, cdf(D_con, y_max - Dy) - cdf(D_con, y_min + Dy)) + 1.0
end

con_list = sample_constraints(50)
display(con_list)
x_opt = solve_problem(con_list)
ϵ = compute_risk(x_opt)
display(ϵ)

plt = plot(getindex.(x_opt, 1), getindex.(x_opt, 2), marker=:diamond)
display(plt)

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
@assert false
N = 100
n_sample = 10
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

N_set = 512:1023
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
size_list = [1000]
risk_list = Float64[]
samples = RiskSample[]
n_step = 1000
θ_list = Float64[]

for t = 1:n_step
    con_list = sample_constraints(size_list[t])
    x_opt = solve_problem(con_list)
    ϵ = compute_risk(x_opt)
    push!(risk_list, ϵ)
    push!(samples, RiskSample(size_list[t], ϵ, t * ϵ^2.5))
    θ, success = fit_parameter(samples)
    @assert success
    push!(θ_list, θ)
    N, success = find_size(θ, β, ϵ_max, 500, 5000)
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