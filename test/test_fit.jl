module TestFit

using Distributions
using OnlineScenarioOptimization
using Random
using Test

Random.seed!(0)

N_set = 100:199
θ = 20
n_sample = 1000
samples1 = Vector{RiskSample}(undef, n_sample)
samples2 = Vector{RiskSample}(undef, n_sample)
for k = 1:n_sample
    N = rand(N_set)
    ϵ = rand(Beta(θ, N - θ + 1))
    samples1[k] = RiskSample(N, ϵ, 0.5)
    samples2[k] = RiskSample(N, ϵ, ϵ^2)
end

@testset "Fit" begin
    θ_hat, success = fit_parameter(samples1)
    @test success
    @test abs(θ_hat - 19.745) < 1e-3
    θ_hat, success = fit_parameter(samples2)
    @test success
    @test abs(θ_hat - 21.390) < 1e-3
end

N = 100
θ = 20
n_sample = 100
samples = Vector{RiskSample}(undef, n_sample)
for k = 1:n_sample
    ϵ = rand(Beta(θ, N - θ + 1))
    samples[k] = RiskSample(N, ϵ, 1.0)
end

@testset "Fit fail" begin
    _, success = fit_parameter(samples, maxiter=2)
    @test !success
end

samples = [RiskSample(3, 0.5, 2.0), RiskSample(2, 0.5, 1.0)]

@testset "Log likelihood" begin
    ll = loglikelihood(samples, 2.0)
    @test abs(ll - 2 * log(3 / 2) / 3) < 1e-5
end

N_set = 100:199
θ = 20
n_sample = 1000
samples = Vector{RiskSample}(undef, n_sample)
for k = 1:n_sample
    N = rand(N_set)
    ϵ = rand(Beta(θ, N - θ + 1))
    samples[k] = RiskSample(N, ϵ, 3 + sin(0.1 * k))
end

@testset "Accuracy" begin
    freq = compute_risk_excess(samples, θ, 0.9)
    @test abs(freq - 0.109) < 1e-3
    freq = compute_risk_excess(samples, θ, 0.5)
    @test abs(freq - 0.510) < 1e-3
end

end # module