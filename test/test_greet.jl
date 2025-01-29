module TestGreet

using OnlineScenarioOptimization
using Test

@testset "Greet" begin
    @test greet() == "Hello World!"
end

end # module