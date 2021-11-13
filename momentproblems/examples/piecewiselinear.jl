# This Julia script is the code for the example of a piecewise-linear
# function.
using HomotopyContinuation
using LinearAlgebra
r = 10; # number of points
d = 3*r; # highest used moment
s = div(d+1, 2); # bound on Fourier coefficients
theseed = 123;

# We generate random input data.
import Distributions: Random, Uniform, Exponential, Normal
Random.seed!(theseed);

t = sort(rand(Uniform(-π,π), r)); # locations
t_ext = [t; 0];
f₀ = [0; rand(Normal(0,1), r-1); 0]; # jumps
f₁ = [0; rand(Normal(0,1), r-1); 0]; # slopes
# distribute slopes equally on both end points of segments
f₀ -= f₁ .* [0; t_ext[2:end] - t_ext[1:end-1]] / 2;
noise = rand(Normal(0,1e-12), 2*s+1);
f₀[8] -= 0.5;

fourierCoeffScaled = k -> ( # scaled by 2π(ik)^2
    @assert(-s ≤ k ≤ s);
    weights = (im * k * (f₀[2:end] - f₀[1:end-1]
                         + f₁[1:end-1].*(t_ext[1:end-1] - t_ext[2:end]))
               + f₁[2:end] - f₁[1:end-1]);
    transpose(weights) * exp.(-im * k * t) + noise[s+k+1]);

moment = k -> (@assert(0 ≤ k ≤ d ≤ 2*s);
               k == s ? 0.0*im : fourierCoeffScaled(k-s));

# We solve the quadratic system given by the moments via homotopy continuation.
@polyvar p[0:r-1];
P = [p;1];
X = [sum(P[i]*P[j] + 0.0 for i=1:(r+1) for j= 1:(r+1) if i+j == k) for k = 2:2*r+2];
momentmat = (a,b) -> hcat([[moment(i+j) for i = 0:a] for j = 0:b]...);
H₀ = momentmat(r-1, 2*r);
H₁ = momentmat(r, 2*r);
sols = solve(H₀ * X, seed=theseed);

# We pick the solution polynomial q that best solves H₁*q^2=0, using one additional moment.
computeErr(q) = (Q² = [map(xk -> xk(p => q), X[1:2*r]); 1];
                 norm(H₁ * Q²));
errs = map(q -> computeErr(q.solution), finite(sols));
@assert(!isempty(errs));
q = finite(sols)[findmin(errs)[2]].solution;

# We compute the roots of the solution polynomial.
using PolynomialRoots
ξ = map(z -> z/norm(z), sort(roots([q; 1]), by=(-) ∘ angle))

# We compute the weights using a confluent Vandermonde system
V = transpose(hcat([vcat([[ξ[j]^k, k * ξ[j]^(k-1)] for j = 1:r]...) for k = 0:d]...));
L = V \ map(moment, 0:d);
λ₀ = L[1:2:end-1];
λ₁ = L[2:2:end];

# We compare the estimated recovered parameters to the original parameters
t_est = real(im * log.(ξ));
f₁_est = real([0; cumsum((λ₀ + s * λ₁ ./ ξ) .* (ξ.^s))]);
f₀_est = real([0; cumsum(-im * λ₁ .* (ξ.^(s-1))
                         - f₁_est[1:end-1] .* (t_est - [t_est; 0][2:end])
                        )]);
# total error
[norm(t - t_est), norm(f₀ - f₀_est), norm(f₁ - f₁_est)]
