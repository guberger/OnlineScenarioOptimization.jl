module ExampleOneDimensional

using Distributions
using LaTeXStrings
using OnlineScenarioOptimization
using Plots
using Random
pythonplot()

const D_con = Normal(1, 2)

sample_constraints(N) = [rand(D_con) for _ = 1:N]
solve_problem(con_list) = maximum(con_list)
compute_risk(x) = ccdf(D_con, x)

################################################################################
ϵ_max = 0.1
β = 0.9
N0 = 20
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
savefig(plt, "examples/figures/one_dimensional.png")
display(plt)

end # module