module ExamplePathlanning

using Distributions
using Gurobi
using JuMP
using LaTeXStrings
using OnlineScenarioOptimization
using Plots
using Random
pythonplot()

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

const D_con = Normal(1.5, 0.05)
const H = 100
const _M_ = 100
const Dy = 0.3
const Dx = 0.045

sample_constraints(N) = [rand(D_con) for _ = 1:N]
function solve_problem(xobs1, con_list)
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
        @constraint(model, x[1] ≤ xobs1 - 0.5 - δ + _M_ * (1 - b1))
        @constraint(model, x[1] ≥ xobs1 + 0.5 + δ - _M_ * (1 - b2))
        for xobs2 in con_list
            @constraint(model, x[2] ≥ xobs2 - Dy - _M_ * (1 - b3))
            @constraint(model, x[2] ≤ xobs2 + Dy + _M_ * (1 - b3))
        end
    end
    for i = 1:H
        @constraint(model, x_list[i] - x_list[i + 1] ≤ +Dx * ones(2))
        @constraint(model, x_list[i] - x_list[i + 1] ≥ -Dx * ones(2))
    end
    e = @variable(model, lower_bound=0)
    xT = [4.5, 0.5]
    @constraint(model, x_list[H + 1] - xT ≤ +e * ones(2))
    @constraint(model, x_list[H + 1] - xT ≥ -e * ones(2))
    @objective(model, Min, e)
    optimize!(model)
    @assert primal_status(model) == FEASIBLE_POINT
    return map(x -> value.(x), x_list)
end
function compute_risk(xobs1, x_list)
    y_min = +Inf
    y_max = -Inf
    for x in x_list
        if xobs1 - 0.5 ≤ x[1] ≤ xobs1 + 0.5
            y_min = min(x[2], y_min)
            y_max = max(x[2], y_max)
        end
    end
    return min(0.0, cdf(D_con, y_max - Dy) - cdf(D_con, y_min + Dy)) + 1.0
end

################################################################################
function generate_risk_samples(N_set, n_sample)
    samples = Vector{RiskSample}(undef, n_sample)
    for k = 1:n_sample
        N = rand(N_set)
        con_list = sample_constraints(N)
        x_opt = solve_problem(2.5, con_list)
        ϵ = compute_risk(2.5, x_opt)
        samples[k] = RiskSample(N, ϵ, 1)
    end
    return samples
end

N = 20
n_sample = 100
samples = generate_risk_samples((N,), n_sample)

risk_list = [sample.ϵ for sample in samples]
θ, success = fit_parameter(samples)
@assert success
display(θ)
D_hat = Beta(θ, N - θ + 1)

xs = range(0.0, 0.5, length=1000)
plt = histogram(risk_list, normalize=:pdf, label=false)
plot!(plt, xs, pdf.(D_hat, xs), c=:red, label=false)
plot!(plt, xlabel=L"v", ylabel=L"f_\theta(v,N)",
      size=(600, 500), labelfontsize=18, tickfontsize=12, dpi=600)
savefig(plt, "examples/figures/path_planning_histogram.png")

################################################################################
ϵ_max = 0.1
β = 0.9
N0 = 50
size_list = [N0]
risk_list = Float64[]
samples = RiskSample[]
n_step = 100
θ_list = Float64[]

for t = 1:n_step
    con_list = sample_constraints(size_list[t])
    x_opt = solve_problem(2.5, con_list)
    ϵ = compute_risk(2.5, x_opt)
    push!(risk_list, ϵ)
    push!(samples, RiskSample(size_list[t], ϵ, t))
    θ, success = fit_parameter(samples)
    @assert success
    push!(θ_list, θ)
    N, success = find_size(θ, β, ϵ_max, 10, 200)
    @assert success
    if t < n_step
        push!(size_list, N)
    end
end

plt1 = plot(size_list, xlabel=L"t", ylabel=L"N_t", label=false)
plt2 = plot(θ_list, xlabel=L"t", ylabel=L"\theta_t", label=false)
sort!(risk_list)
cdf_list = collect(1:length(risk_list)) / length(risk_list)
plt3 = plot(risk_list, cdf_list,
            xlabel=L"V(x_t)", ylabel="frequency",
            legend=:bottomright, label=false)
hline!(plt3, [β], label=L"\beta")
vline!(plt3, [ϵ_max], label=L"\epsilon")

plt = plot(plt1, plt2, plt3, size=(900,250), layout=(1, 3))
plot!(plt, labelfontsize=18, tickfontsize=12, dpi=600)
savefig(plt, "examples/figures/path_planning.png")

################################################################################
ϵ_max = 0.1
β = 0.9
N0 = 50
size_list = [N0]
risk_list = Float64[]
samples = RiskSample[]
n_step = 100
θ_list = Float64[]

for t = 1:n_step
    con_list = sample_constraints(size_list[t])
    xobs1 = 2.5 + sin(0.1 * t)
    x_opt = solve_problem(xobs1, con_list)
    ϵ = compute_risk(xobs1, x_opt)
    push!(risk_list, ϵ)
    push!(samples, RiskSample(size_list[t], ϵ, t))
    θ, success = fit_parameter(samples)
    @assert success
    push!(θ_list, θ)
    N, success = find_size(θ, β, ϵ_max, 10, 200)
    @assert success
    if t < n_step
        push!(size_list, N)
    end
end

plt1 = plot(size_list, xlabel=L"t", ylabel=L"N_t", label=false)
plt2 = plot(θ_list, xlabel=L"t", ylabel=L"\theta_t", label=false)
sort!(risk_list)
cdf_list = collect(1:length(risk_list)) / length(risk_list)
plt3 = plot(risk_list, cdf_list,
            xlabel=L"V(x_t)", ylabel="frequency",
            legend=:bottomright, label=false)
hline!(plt3, [β], label=L"\beta")
vline!(plt3, [ϵ_max], label=L"\epsilon")

plt = plot(plt1, plt2, plt3, size=(900,250), layout=(1, 3))
plot!(plt, labelfontsize=18, tickfontsize=12, dpi=600)
savefig(plt, "examples/figures/path_planning_shifting.png")
display(plt)

end # module