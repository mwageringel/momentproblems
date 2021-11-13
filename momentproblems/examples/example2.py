# example of curves on torus of max-degree 1.
from sage.all import *
import numpy as np

from momentproblems.moment_functionals import *
from momentproblems.mainalgorithm import *
from momentproblems.mainalgorithm import _check_weights
from momentproblems.plotting import plot_fun
tol = 1e-8

def _functionals(r):
    KK = ComplexBallField(60)
    functionals0 = [mk_moment_trigonometric_curve_example1(*cc, prec=60)
                   for cc in [(RDF**2).random_element() for _ in range(r)]]
    functionals = [modulated(f, (vector(KK, shift)*KK(I*2*pi)).apply_map(exp)) for f, shift
                   in zip(functionals0, [(RDF**2).random_element() for _ in range(r)])]
    return functionals

def _toeplitz_matrices(functionals, D):
    dim = 2
    Ts = [moment_functionals.toeplitz(f, dim=dim, d=D).change_ring(CDF) for f in functionals]
    return Ts

def example(r=6):
    D = r-1
    Ts = _toeplitz_matrices(_functionals(r), D)
    dim = 2
    weights = matrix.random(CDF, r).columns()
    Ms = [Ts[0].parent().sum(weights[i][j] * Ts[j] for j in range(r)) for i in range(len(weights))]

    δ = 0
    λ_ij = weights_from_moment_matrices_fixed_δ(Ms, d=D-δ, δ=δ, dim=dim, tol=tol, torus=True, algorithm='cluster')
    assert _check_weights(weights, λ_ij, 10*tol), "reconstruction failed"
    Ts_rec = unmixed_moment_matrices(Ms, λ_ij)
    return Ts, Ms, Ts_rec

def generate_plots(r=6):
    D = r-1
    Ts, Ms, Ts_rec = example(r=r)
    Gs = []
    Gs += [plot_fun(sum(Ts), ε=0.1, d=D, plot_points=1000, colorbar=False)]
    # Gs += [plot_fun(Mi), ε=0.1, d=D, plot_points=1000, colorbar=False) for Mi in Ms]
    # Gs += [plot_fun(Ts_reci, ε=0.1, d=D, plot_points=1000, colorbar=False) for Ts_reci in Ts_rec]
    Gs += [plot_fun(*Ts_rec, ε=0.1, d=D, plot_points=1000, colorbar=False)]
    return Gs

def generate_plots2(r=3):
    Gs = []
    fs = _functionals(r)
    for D in [5, 10, 20]:
        T = sum(_toeplitz_matrices(fs, D=D))
        Gs += [plot_fun(T, d=D, plot_points=1000, colorbar=False, fun='Q', ε=0.01)]
        Gs += [plot_fun(T, d=D, plot_points=1000, colorbar=False, fun='P')]
        Gs += [plot_fun(T, d=D, plot_points=1000, colorbar=False, fun='P1', ε=1e-4)]
    return Gs
