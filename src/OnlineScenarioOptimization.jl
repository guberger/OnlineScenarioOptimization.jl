module OnlineScenarioOptimization

export greet,
    RiskSample,
    fit_parameter,
    loglikelihood,
    compute_risk_excess,
    cdf_risk,
    find_size,
    quantile_risk,
    compute_risk_excess

greet() = "Hello World!"

using Distributions
using LinearAlgebra
using SpecialFunctions

struct RiskSample
    N::Int
    ϵ::Float64
    w::Float64
end

_L(N, ϵ, θ) = (θ - 1) * log(ϵ) + (N - θ) * log(1 - ϵ) - logbeta(θ, N - θ + 1)
_LL(sample, θ) = _L(sample.N, sample.ϵ, θ) * sample.w
_risk(sample) = log(0 + sample.ϵ) * sample.w
_safe(sample) = log(1 - sample.ϵ) * sample.w
_D1(sample, θ) = digamma(θ) * sample.w
_D2(sample, θ) = digamma(sample.N - θ + 1) * sample.w
_H1(sample, θ) = trigamma(θ) * sample.w
_H2(sample, θ) = trigamma(sample.N - θ + 1) * sample.w

function Distributions.loglikelihood(samples, θ)
    w_tot = sum(sample -> sample.w, samples)
    ll = sum(sample -> _LL(sample, θ), samples)
    return ll / w_tot
end

function _initial_guess(samples)
    num = sum(sample -> sample.ϵ * sample.w, samples)
    den = sum(sample -> sample.w / (sample.N + 1), samples)
    return num / den
end

function fit_parameter(samples; maxiter::Int=1000, tol::Float64=1e-14)
    θ = _initial_guess(samples)
    g1 = sum(_risk, samples)
    g2 = sum(_safe, samples)
    t = 0
    while t < maxiter # Newton method
        t += 1
        temp1 = sum(sample -> _D1(sample, θ), samples)
        temp2 = sum(sample -> _D2(sample, θ), samples)
        grad = -(temp1 - temp2) + g1 - g2
        dtemp1 = sum(sample -> _H1(sample, θ), samples)
        dtemp2 = sum(sample -> _H2(sample, θ), samples)
        hess = -(dtemp1 + dtemp2)
        dθ = grad / hess # Newton step
        θ -= dθ
        if dθ^2 < 2*tol # stopping criterion
            return θ, true
        end
    end
    return θ, false
end

cdf_risk(θ, ϵ, N) = cdf(Beta(θ, N - θ + 1), ϵ)

function find_size(θ, β, ϵ_max, N_min::Int, N_max::Int)
    N_min = max(N_min, ceil(Int, θ + 1))
    if N_min > N_max
        return N_max, false
    end
    conf_min = cdf_risk(θ, ϵ_max, N_min)
    if conf_min ≥ β
        return N_min, true
    end
    conf_max = cdf_risk(θ, ϵ_max, N_max)
    if conf_max < β
        return N_max, false
    end
    while N_min < N_max - 1
        N_mid = (N_max + N_min) / 2
        conf = cdf_risk(θ, ϵ_max, N_mid)
        if conf < β
            N_min = floor(Int, N_mid + 0.1)
        else
            N_max = ceil(Int, N_mid - 0.1)
        end
    end
    return N_max, true
end

quantile_risk(θ, β, N) = quantile(Beta(θ, N - θ + 1), β)

function compute_risk_excess(samples, θ, β)
    w_tot = sum(sample -> sample.w, samples)
    excess = 0.0
    for sample in samples
        ϵ_th = quantile_risk(θ, β, sample.N)
        if sample.ϵ > ϵ_th
            excess += sample.w
        end
    end
    return excess / w_tot
end

end # module
