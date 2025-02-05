module ExampleConvexGaussian

using Distributions
using Gurobi
using JuMP
using LaTeXStrings
using LinearAlgebra
using OnlineScenarioOptimization
using Plots
using Random
pythonplot()

const GUROBI_ENV = Gurobi.Env()
solver() = Model(optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag"=>false
))

const dim_x = 20

sample_constraints(N) = [randn(dim_x) for _ = 1:N]
function solve_problem(con_list)
    model = solver()
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
ϵ_max = 0.1
β = 0.9
N0 = 100
size_list = [N0]
risk_list = Float64[]
samples = RiskSample[]
n_step = 1000
θ_list = Float64[]

for t = 1:n_step
    con_list = sample_constraints(size_list[t])
    x_opt = solve_problem(con_list)
    ϵ = compute_risk(x_opt)
    push!(risk_list, ϵ)
    push!(samples, RiskSample(size_list[t], ϵ, 1))
    θ, success = fit_parameter(samples)
    @assert success
    push!(θ_list, θ)
    N, success = find_size(θ, β, ϵ_max, 10, 1000)
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
savefig(plt, "examples/figures/convex_gaussian.png")
display(plt)

end # module