module ExampleGarattiFig9

using Distributions
using LaTeXStrings
using OnlineScenarioOptimization
using Plots
using Random
pythonplot()

const dim_x = 400
const N_valid = 10000

sample_constraints(N) = [randn(dim_x) .+ (rand() < 0.01) * randn() * 2 for _ = 1:N]
solve_problem(con_list) = [maximum(p[i] for p in con_list) for i = 1:dim_x]
compute_risk(x) = count(any(x .< p) for p in sample_constraints(N_valid)) / N_valid

################################################################################
ϵ_max = 0.1
β = 0.9
N0 = 1000
size_list = [N0]
risk_list = Float64[]
samples = RiskSample[]
n_step = 1000
θ_list = Float64[]

_update(sample, ρ) = RiskSample(sample.N, sample.ϵ, sample.w * ρ)

for t = 1:n_step
    x_opt = solve_problem(sample_constraints(size_list[t]))
    ϵ = compute_risk(x_opt)
    push!(risk_list, ϵ)
    push!(samples, RiskSample(size_list[t], ϵ, t^2))
    #
    θ = -1.0
    samples_bis = copy(samples)
    for s = 1:100
        display((t, s))
        θ, success = fit_parameter(samples_bis)
        @assert success
        freq = compute_risk_excess(samples, θ, β)
        freq ≤ 1 - β + (t > 100 ? -1 / t : 1) && break
        for (k, sample) in enumerate(samples_bis)
            ϵ_th = quantile_risk(θ, β, sample.N)
            sample.ϵ ≤ ϵ_th && break
            samples_bis[k] = _update(sample, 2)
        end
    end
    @assert θ > 0
    push!(θ_list, θ)
    #
    N, success = find_size(θ, β, ϵ_max, 500, 5000)
    @assert success
    t == n_step && break
    push!(size_list, N)
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

plt = plot(plt1, plt2, plt3, size=(900,250), layout=(1, 3),
           labelfontsize=18, tickfontsize=12, dpi=600)
savefig(plt, "examples/figures/garatti_fig9.png")
display(plt)

end # module