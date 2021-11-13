# This is a more complicated example on the torus. Computing the moments
# accurately takes some time. This example nicely separates the components (see
# pictures) using moments of small degree (a bit smaller than what would be
# able to recover the whole variety), but at the expense of using several
# weighted sums instead of just one weighted sum.
from sage.all import *
import numpy as np

from momentproblems.moment_functionals import *
from momentproblems.mainalgorithm import *
from momentproblems.mainalgorithm import _check_weights
from momentproblems.plotting import plot_fun
tol = 1e-8
dim = 2

def example(r=6, D=10):
    weights = matrix.random(CDF, r).columns()
    KK = ComplexBallField(60)
    functionals0 = [mk_moment_trigonometric_curve_example(*cc, prec=60)
                   for cc in [(5/8, 0), (3/7, 1), (5/6, 1/2), (7/2, 5/3), (1, 4), (15/4, 2)]]

    functionals = [modulated(f, (vector(KK, shift)*KK(I*2*pi)).apply_map(exp)) for f, shift
                   in zip(functionals0, [(0,0), (.25,.85), (0.1, 0.4), (1/3, 8/9), (0.5, 0.8), (0.9, 0.2)])]

    dim = 2
    Ts = [moment_functionals.toeplitz(f, dim=dim, d=D).change_ring(CDF) for f in functionals[:r]]
    Ms = [Ts[0].parent().sum(weights[i][j] * Ts[j] for j in range(r)) for i in range(len(weights))]

    δ = 0
    λ_ij = weights_from_moment_matrices_fixed_δ(Ms, d=D-δ, δ=δ, dim=dim, tol=tol, torus=True, algorithm='cluster')
    assert _check_weights(weights, λ_ij, 10*tol), "reconstruction failed"
    Ts_rec = unmixed_moment_matrices(Ms, λ_ij)
    return Ts, Ms, Ts_rec

def generate_plots(r=6, D=10, ε=0.01):
    Ts, Ms, Ts_rec = example(r=r, D=D)
    Gs = []
    Gs += [plot_fun(Mi, ε=ε, d=D, plot_points=1000, colorbar=False) for Mi in Ms]
    Gs += [plot_fun(Ts_reci, ε=ε, d=D, plot_points=1000, colorbar=False) for Ts_reci in Ts_rec]
    Gs += [plot_fun(*Ts_rec, ε=ε, d=D, plot_points=1000, colorbar=False)]
    return Gs
