from sage.all import *
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def _vand(M, d, dim):
    grid = np.linspace(0, CDF(2*pi), M, endpoint=False)
    grid_exp = np.exp(-CDF(i) * grid)
    grid_exp1 = np.vander(grid_exp, d+1, increasing=True) #* (grid_exp ** -d)).T
    if dim == 2:
        return np.kron(grid_exp1, grid_exp1)
    elif dim == 1:
        return grid_exp1
    else:
        raise NotImplementedError("only dimension 1 and 2 supported")

def _eval_Q(HT, d, *, ε, plot_points, regularization='spectralcutoff', dim=2):
    U, Σ, V = (HT / HT.ncols()).SVD()
    M = plot_points
    Vand = _vand(M, d, dim) / RDF(sqrt(HT.ncols()))
    Q1 = Vand @ U.numpy()
    if regularization == 'spectralcutoff':
        Σε = [1/σj if σj > ε else 1/ε for σj in Σ.diagonal()]
    elif regularization == 'tikhonov':
        Σε = [1/(σj + ε) for σj in Σ.diagonal()]
    elif regularization == 'ideallowpass':
        Σε = [0 if σj > ε else 1/ε for σj in Σ.diagonal()]
    else:
        raise ValueError('unknown regularization: %s' % regularization)
    Q2 = (Q1 * Q1.conj()) * np.array(Σε)
    Q_eval = 1 / Q2.sum(axis=1).real
    if dim == 2:
        evals = Q_eval.reshape(M, M)
    else:
        assert dim == 1
        evals = Q_eval
    return fftshift(evals)

def _eval_P(HT, d, *, ε=None, plot_points, one=False, dim=2):
    U, Σ, V = (HT / HT.ncols()).SVD()
    M = plot_points
    Vand = _vand(M, d, dim) / RDF(sqrt(HT.ncols()))
    Q1 = Vand @ U.numpy()
    if one:  # P1
        Q2 = (Q1 * Q1.conj()) * np.array([1 if σj > ε else 0 for σj in Σ.diagonal()])
    else: # P
        Q2 = (Q1 * Q1.conj()) * np.array(Σ.diagonal())
    Q_eval = Q2.sum(axis=1).real
    if dim == 2:
        evals = Q_eval.reshape(M, M)
    else:
        assert dim == 1
        evals = Q_eval
    return fftshift(evals)

def plot_fun(*HTs, d, ε=0.01, plot_points=500, fun='Q', regularization='spectralcutoff',
           **plot_options):  # only two-dimensional, only torus, HT should be positive-semidefinite
    # making ε smaller makes q_ε narrower
    Q_evals = []
    for HT in HTs:
        if fun == 'Q':
            Q_evals.append(_eval_Q(HT, d, ε=ε, plot_points=plot_points, regularization=regularization))
        elif fun == 'P':
            Q_evals.append(_eval_P(HT, d, plot_points=plot_points, one=False))
        elif fun == 'P1':
            Q_evals.append(_eval_P(HT, d, ε=ε, plot_points=plot_points, one=True))
        else:
            raise ValueError('unknown function')

    from matplotlib import rc
    from matplotlib import ticker
    rc('text', usetex=True)
    rc('font', **{'family':'serif','serif':['Computer Modern']})
    rc('ytick', right=True)

    total = np.maximum.reduce([arr/np.max(arr) for arr in Q_evals])  # combine different components using maximum
    return matrix_plot(total, xrange=(0,1), yrange=(0,1), flip_y=False,
        axes_pad=0, ticks=[[0,0.5,1]]*2,
        tick_formatter=[ticker.NullFormatter()]*2,  # no labels
        transparent=True,
        dpi=400,
        figsize=2.8,
        # colorbar=True,
        # title=f"$d = {d}$; $\\varepsilon = {ε:.1g}$",
        **plot_options)
