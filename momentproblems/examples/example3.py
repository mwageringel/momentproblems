from sage.all import *
import numpy as np
import momentproblems.moment_functionals
from momentproblems.plotting import _eval_Q, _eval_P

def generate_plots(ds=(4,8), filename=None):
    from matplotlib import rc
    rc('text', usetex=True)
    rc('font', **{'family':'serif','serif':['Computer Modern']})

    points = [0.2, 0.35, 0.8]
    Λ = matrix.diagonal(RDF, [1.3, 0.7, 1])  # average height 1
    Gs = []
    tt = np.linspace(0, RDF(1), 3000, endpoint=False)
    data = [['tt'] + list(tt)]
    for d in ds:
        col = 'black'
        basis = momentproblems.moment_functionals.monomial_basis_laurent(dim=1, d=d)
        A = momentproblems.moment_functionals.vandermonde([[RDF(-t*2*pi + pi)] for t in points], basis=basis, trigonometric=True)
        HT = A.H * Λ * A
        qq = _eval_Q(HT, d, dim=1, plot_points=len(tt), ε=0.1)
        q2 = _eval_Q(HT, d, dim=1, plot_points=len(tt), ε=0.01)
        pp = _eval_P(HT, d, dim=1, plot_points=len(tt), one=False)
        p1 = _eval_P(HT, d, dim=1, plot_points=len(tt), one=True, ε=1e-4)
        if filename is None:
            G = Graphics()
            G += sum(list_plot(list(zip(tt, arr)),
                    ymin=0, ymax=1.4,
                    aspect_ratio=0.25,
                    title=rf'$d = {d}$',
                    plotjoined=True, linestyle=sty, color=col, frame=True, axes=False, figsize=5.75, thickness=0.4)
                    for arr, sty in zip([p1, pp, qq, q2], [':', '--', '-', '-.']))
            G += sum(point2d([points[j], Λ[j,j]], size=5, color=col) for j in range(len(points)))
            G += sum(point2d([points[j], 1     ], size=5, color=col, marker='x') for j in range(len(points)))
            Gs.append(G)
        else:  # write data file
            data += [[f'p1{d}'] + list(p1),
                    [f'pp{d}'] + list(pp),
                    [f'qq{d}'] + list(qq),
                    [f'q2{d}'] + list(q2)]
    if filename is None:
        return Gs
    else:
        with open(filename, 'w') as f:
            for row in range(len(data[0])):
                f.write(' '.join(str(data[col][row]) for col in range(len(data))) + '\n')
