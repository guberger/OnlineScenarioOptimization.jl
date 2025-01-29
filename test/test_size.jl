module TestSize

using Distributions
using OnlineScenarioOptimization
using Test

θ = 20
ϵ_max = 0.1
β = 0.9

@testset "Size normal" begin
    N, success = find_size(θ, β, ϵ_max, 0, 1000)
    @test success
    @test cdf(Beta(θ, N - θ + 1), ϵ_max) ≥ β
    @test cdf(Beta(θ, N - θ + 0), ϵ_max) < β
end

@testset "Size N_min" begin
    N, success = find_size(θ, β, ϵ_max, 500, 1000)
    @test success
    @test N == 500
end

@testset "Size fail" begin
    _, success = find_size(θ, β, ϵ_max, 10, 20)
    @test !success
    _, success = find_size(θ, β, ϵ_max, 50, 100)
    @test !success
end

end # module