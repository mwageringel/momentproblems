r"""
Main algorithms for recovering weights as eigenvalues from mixed moment matrices.


************************************************************
Example: Three circles on a line
************************************************************

Imports::

    sage: from momentproblems import moment_functionals
    sage: from momentproblems.mainalgorithm import *
    sage: from momentproblems.mainalgorithm import _check_weights
    sage: tol = 1e-7
    sage: dim = 2

Construction of moment matrices of components and mixtures.
Note that we have more pencils than needed (s > r)::

    sage: functionals = [moment_functionals.mk_moment_ellipse(center=xy, radius=(1,1)) for xy in [(0,0), (-3,3/2), (2,-1)]]
    sage: weights = [[1, 1, 1], [2, 3, 6], [-1, 3, 5], [2, 1, -2]]
    sage: Hs = [moment_functionals.hankel(f, dim=dim, d=6) for f in functionals]
    sage: Ms = [Hs[0].parent().sum(weights[i][j] * Hs[j] for j in range(len(Hs))) for i in range(len(weights))]


Symbolically
============

::

    sage: λ_ij = weights_from_moment_matrices_fixed_δ(Ms, d=4, δ=2, dim=dim, tol=tol)
    sage: assert λ_ij.base_ring().is_exact()
    sage: _check_weights(weights, λ_ij, tol)
    True
    sage: λ_ij = weights_from_moment_matrices(Ms, d=4, dim=dim, tol=tol)
    sage: assert λ_ij.base_ring().is_exact()
    sage: _check_weights(weights, λ_ij, tol)
    True

In the case that ``δ=0``, we find more common eigenvalues than desired::

    sage: Hs2 = [moment_functionals.hankel(f, dim=dim, d=4) for f in functionals]
    sage: Ms2 = [Hs2[0].parent().sum(weights[i][j] * Hs2[j] for j in range(len(Hs2))) for i in range(len(weights))]
    sage: λ_ij = weights_from_moment_matrices_fixed_δ(Ms2, d=4, δ=0, dim=dim, tol=tol)
    sage: λ_ij.ncols() > len(Hs2)
    True

We can compute the eigenvector corresponding to the extraneous eigenvalue which
exists due to the symmetry structure between the components::

    sage: extraneous = ProjectiveSpace(len(Ms2)-1)(*sum(matrix(weights).columns()))
    sage: from momentproblems.commoneigenspaces import common_eigenspaces_symbolic
    sage: eigvals, eigmat = common_eigenspaces_symbolic(Ms2)
    sage: monoms = vector(moment_functionals.monomial_basis(dim=dim, d=4))
    sage: p = eigmat.column(eigvals.index(extraneous)) * monoms; p / p.lc()
    y0^4 + 8*y0^3*y1 + 24*y0^2*y1^2 + 32*y0*y1^3 + 16*y1^4 - 5*y0^2 - 20*y0*y1 - 20*y1^2 + 25/8

For ``δ=1``, the number of common eigenvalues is correct and we recover the
defining ideals::

    sage: Hs2 = [moment_functionals.hankel(f, dim=dim, d=5) for f in functionals]
    sage: Ms2 = [Hs2[0].parent().sum(weights[i][j] * Hs2[j] for j in range(len(Hs2))) for i in range(len(weights))]
    sage: λ_ij = weights_from_moment_matrices_fixed_δ(Ms2, d=4, δ=1, dim=dim, tol=tol)
    sage: λ_ij.ncols() == len(Hs2)
    True
    sage: _check_weights(weights, λ_ij, tol)
    True

    sage: Hs_rec = [Hj/Hj[0,0] for Hj in unmixed_moment_matrices(Ms2, λ_ij)]
    sage: H0_rec, = [Hj for Hj in Hs_rec if Hj[0,1] == 0]
    sage: monoms = vector(moment_functionals.monomial_basis(dim=dim, d=5))
    sage: ker = (H0_rec.right_kernel_matrix() * monoms); ker.column()
    [    -y0^4 - 2*y0^2*y1^2 - y1^4 + 1]
    [-y0^5 - 2*y0^3*y1^2 - y0*y1^4 + y0]
    [-y0^4*y1 - 2*y0^2*y1^3 - y1^5 + y1]
    [          -y0^4 - y0^2*y1^2 + y0^2]
    [        -y0^3*y1 - y0*y1^3 + y0*y1]
    [          -y0^2*y1^2 - y1^4 + y1^2]
    [          -y0^5 - y0^3*y1^2 + y0^3]
    [    -y0^4*y1 - y0^2*y1^3 + y0^2*y1]
    [    -y0^3*y1^2 - y0*y1^4 + y0*y1^2]
    [          -y0^2*y1^3 - y1^5 + y1^3]
    sage: ideal(ker.list()).groebner_basis()
    [y0^2 + y1^2 - 1]


Numerically
===========

::

    sage: Hs = [H.change_ring(RDF) for H in Hs]
    sage: Ms = [Hs[0].parent().sum(weights[i][j] * Hs[j] for j in range(len(Hs))) for i in range(len(weights))]

Reconstruction of weights via two different algorithms.
Ellipses are generated by polynomials of degree ``δ=2``::

    sage: λ_ij = weights_from_moment_matrices_fixed_δ(Ms, d=4, δ=2, dim=dim, tol=tol)
    sage: _check_weights(weights, λ_ij, tol)
    True

We have ``r+1=3`` components::

    sage: λ_ij = weights_from_moment_matrices(Ms, d=4, dim=dim, tol=tol)
    sage: _check_weights(weights, λ_ij, tol)
    True

Convex case
===========

Reconstruction using information about convex hulls in case of
positive-semidefinite matrices::

    sage: weights = [[4, 2, 1], [2, 3, 6], [1, 3, 5], [2, 1, 2]]
    sage: Hs = [moment_functionals.hankel(f, dim=dim, d=6) for f in functionals]
    sage: Ms = [Hs[0].parent().sum(weights[i][j] * Hs[j] for j in range(len(Hs))) for i in range(len(weights))]

Check positive-semidefiniteness, so all eigenvalues are non-negative (Sage 9.3
has a method ``is_positive_semidefinite`` for this)::

    sage: is_positive_semidefinite = lambda M: all(abs(e.imag()) < tol and e.real() > -tol for e in M.change_ring(CDF).eigenvalues())
    sage: all(is_positive_semidefinite(Mi) for Mi in Ms)
    True

The reconstruction (symbolically)::

    sage: λ_ij = weights_from_moment_matrices_convex(Ms, tol=tol)
    sage: _check_weights(weights, λ_ij, tol)
    True

Numerically::

    sage: Hs = [H.change_ring(RDF) for H in Hs]
    sage: Ms = [Hs[0].parent().sum(weights[i][j] * Hs[j] for j in range(len(Hs))) for i in range(len(weights))]
    sage: all(is_positive_semidefinite(Mi) for Mi in Ms)
    True
    sage: λ_ij = weights_from_moment_matrices_convex(Ms, tol=tol)
    sage: _check_weights(weights, λ_ij, tol)  # not tested (numeric convex hull not fully implemented)
    True

With complex weights::

    sage: weights = [[4, CDF(0,2), CDF(1,1)], [CDF(2,-2), 3, CDF(6,-1)], [CDF(1,1), CDF(3,4), 5], [CDF(2,-2), CDF(-1,-4), CDF(-2,2)]]
    sage: Ms = [Hs[0].parent().sum(weights[i][j] * Hs[j] for j in range(len(Hs))) for i in range(len(weights))]

This requires a linear combination of the positive-semidefinite matrices ``Hs``
with positive weights, so that the coordinates for the eigenvalues can be
choosen with respect to this matrix::

    sage: H = sum(Hs).change_ring(CDF)
    sage: λ_ij = weights_from_moment_matrices_convex(Ms, H, tol=tol)
    sage: _check_weights(weights, λ_ij, tol)  # not tested (numeric convex hull not fully implemented)
    True


************************************************************
Example: straight lines on the torus
************************************************************

The components are of degree `(1,1)`, `(1,1)` and `(2,1)`.
Therefore, max-degree 3 is enough to find common eigenspaces.
As the components are of max-degree 2, an upper bound on `δ` is 2,
so `d+δ ≤ 5`::

    sage: functionals = [moment_functionals.mk_moment_torus_line(ratio=q, translate=c) for q, c in
    ....:                [((1, 1), e^(i*pi/4)), ((-1, 1), 1), ((-2, 1), 1)]]  #, ((2, 1), e^(i*1))]]
    sage: D = 5
    sage: Ts = [moment_functionals.toeplitz(f, dim=dim, d=D).change_ring(CDF) for f in functionals]
    sage: weights = [[1, 1, 1], [2, 3, 6], [-1, 3, 5]]
    sage: Ms = [Ts[0].parent().sum(weights[i][j] * Ts[j] for j in range(len(Ts))) for i in range(len(weights))]

    sage: δ = 2
    sage: λ_ij = weights_from_moment_matrices_fixed_δ(Ms, d=D-δ, δ=δ, dim=dim, tol=tol, torus=True)
    sage: _check_weights(weights, λ_ij, tol)
    True

    sage: λ_ij = weights_from_moment_matrices(Ms, d=3, dim=dim, tol=tol, torus=True)
    sage: _check_weights(weights, λ_ij, tol)
    True
    sage: λ_ij.ncols() == len(Ts)
    True

From the weights `λ_ij`, we recover the original Toeplitz matrices (up to
permutation), after normalizing them::

    sage: Ts_rec = [Tj/Tj[0,0] for Tj in unmixed_moment_matrices(Ms, λ_ij)]
    sage: all(any((Tj_rec - Tl).norm() < tol for Tl in Ts) for Tj_rec in Ts_rec)
    True

In the case that ``δ=0``, we find more common eigenvalues than desired::

    sage: λ_ij = weights_from_moment_matrices_fixed_δ(Ms, d=D, δ=0, dim=dim, tol=tol, torus=True)
    sage: λ_ij.ncols() > len(Ts)
    True


************************************************************
Example: affine line segments
************************************************************

::

    sage: functionals = [moment_functionals.mk_moment_affine_line_segment(start, end) for start, end in
    ....:                [[(2,3,1), (3,4,0)], [(1,-3,-5), (0,2,0)], [(-3,-7,2), (3,-4,-4)]]]
    sage: dim, D = 3, 3
    sage: Hs = [moment_functionals.hankel(f, dim=dim, d=D).change_ring(RDF) for f in functionals]
    sage: weights = [[1, 1, 1], [2, 3, 6], [-1, 3, 5]]
    sage: Ms = [Hs[0].parent().sum(weights[i][j] * Hs[j] for j in range(len(Hs))) for i in range(len(weights))]

Each of the three components lives in a variety of degree 1, so total degree 2
is enough to find common eigenspaces (polynomials that vanish on two of three
components). An upper bound on `δ` is 1, so `d+δ ≤ 3`::

    sage: δ = 1
    sage: λ_ij = weights_from_moment_matrices_fixed_δ(Ms, d=D-δ, δ=δ, dim=dim, tol=tol)
    sage: _check_weights(weights, λ_ij, tol)
    True

    sage: λ_ij = weights_from_moment_matrices(Ms, d=2, dim=dim, tol=tol)
    sage: _check_weights(weights, λ_ij, tol)
    True
    sage: λ_ij.ncols() == len(Hs)
    True

In this case, we already find the correct weights when `δ=0`::

    sage: λ_ij = weights_from_moment_matrices_fixed_δ(Ms, d=D, δ=0, dim=dim, tol=tol)
    sage: _check_weights(weights, λ_ij, tol)
    True

Note that picking `d` larger than necessary can introduce numerical errors that
requires increasing the tolerance::

    sage: D = 4
    sage: Hs = [moment_functionals.hankel(f, dim=dim, d=D).change_ring(RDF) for f in functionals]
    sage: Ms = [Hs[0].parent().sum(weights[i][j] * Hs[j] for j in range(len(Hs))) for i in range(len(weights))]
    sage: λ_ij = weights_from_moment_matrices_fixed_δ(Ms, d=3, δ=1, dim=dim, tol=tol)
    sage: _check_weights(weights, λ_ij, tol)
    False
    sage: λ_ij = weights_from_moment_matrices_fixed_δ(Ms, d=3, δ=1, dim=dim, tol=1e2*tol)
    sage: _check_weights(weights, λ_ij, 1e2*tol)
    True


************************************************************
Example: random functionals on some trigonometric varieties
************************************************************

These functionals will not be positive-semidefinite. Recovery works nonetheless.
The components are of bidegree `(1,2)`, `(2,1)`, `(1,1)`.
Thus, max-degree 3 is enough to find common eigenspaces, i.e. there is a
polynomial of degree `(3,3)` vanishing on the first two components, but not on
the third.
As the varieties are defined by some unstructured polynomials, we expect δ=0 to
be sufficient for recovery.  ::

    sage: L.<x,y> = LaurentPolynomialRing(QQ)
    sage: random_fun = lambda: ZZ.random_element(1, 30)
    sage: functionals = [moment_functionals.random_moment_functional(L.ideal([p]), random_fun=random_fun)
    ....:                for p in [y^2 - x*y + 1, x^2 + y + x*y + y, x*y + 3*x + 1]]
    sage: D = 3
    sage: dim = L.ngens()
    sage: Ts = [moment_functionals.toeplitz(f, dim=dim, d=D).change_ring(CDF) for f in functionals]
    sage: weights = [[1, 1, 1], [2, 3, 6], [-1, 3, 5]]
    sage: Ms = [Ts[0].parent().sum(weights[i][j] * Ts[j] for j in range(len(Ts))) for i in range(len(weights))]

    sage: δ = 0
    sage: λ_ij = weights_from_moment_matrices_fixed_δ(Ms, d=D-δ, δ=δ, dim=dim, tol=tol, torus=True)
    sage: _check_weights(weights, λ_ij, tol)
    True

    sage: λ_ij = weights_from_moment_matrices(Ms, d=D, dim=dim, tol=tol, torus=True)
    sage: _check_weights(weights, λ_ij, tol)
    True


************************************************************
Example: tensor decomposition
************************************************************

We demonstrate how to compute a tensor decomposition by computing the
eigenvalues of its slices, in case the rank of the tensor is at most one of the
dimensions. ::

    sage: K = QQ
    sage: n = 7
    sage: r = n-2
    sage: s = r+3
    sage: assert r <= n and r <= s
    sage: x = [vector(K, [ZZ.random_element(1,30) for _ in range(s)]) for _ in range(r)]
    sage: y = [vector(K, [ZZ.random_element(1,30) for _ in range(n)]) for _ in range(r)]
    sage: z = [vector(K, [ZZ.random_element(1,30) for _ in range(n)]) for _ in range(r)]
    sage: λ = [K(ZZ.random_element(1,30)) for _ in range(r)]
    sage: import numpy as np
    sage: from momentproblems.commoneigenspaces import common_eigenspaces_symbolic
    sage: t = sum(λ[j] * np.einsum('i,j,k->ijk', x[j].numpy(int), y[j].numpy(int), z[j].numpy(int)) for j in range(r))

    sage: Xs = [matrix(K, np.ascontiguousarray(t[i,:,:])) for i in range(s)]
    sage: Vx, ev_x = eigenspaces_lifted(Xs, tol=None)[:2]
    sage: YZs = unmixed_moment_matrices(Xs, matrix.column(K, ev_x))

    sage: def check(YZ): # the rank-1 matrices are what we expect
    ....:     u, = [w[0] for e,w,m in YZ.eigenvectors_right() if e != 0]
    ....:     v, = [w[0] for e,w,m in YZ.eigenvectors_left() if e != 0]
    ....:     return any(matrix([u, y[j]]).rank() == 1 == matrix([v, z[j]]).rank() == 1 for j in range(r))

    sage: _check_weights(matrix.column(K, x), matrix.column(K, ev_x), tol=None)
    True
    sage: all(check(YZ) for YZ in YZs)
    True
"""

import itertools
from sage.all import matrix, vector, binomial, ZZ, sqrt, ProjectiveSpace, Polyhedron, real, imag, RDF, CDF
import scipy.linalg
import numpy as np
from numpy.linalg import matrix_rank

def _mk_shift_mat(basis_from, basis_to, K):
    r"""
    maps from R_≤d-δ to R_≤d
    """
    indices = {yγ: k for k, yγ in enumerate(basis_to)}
    def shift_mat(yα):
        mat = matrix(K, len(basis_to), len(basis_from), sparse=False)
        for k, yβ in enumerate(basis_from):
            mat[indices[yα*yβ], k] = K.one()
        return mat
    return shift_mat

# TODO (here we work with d-δ,d instead of d,d+δ)
def shift_invariant_eigenvalue_indices(Ur, eigspaces, d, δ, dim, tol, basis_fun, deg_fun):
    from momentproblems import intersections
    K = Ur.base_ring()
    if any(Vj.base_ring() != K for Vj in eigspaces):
        raise ValueError("all matrices should be defined over same ring: %s" % K)
    if δ == 0:
        # shifting has no effect in this case, so can be skipped
        return range(len(eigspaces))
    basis_d = basis_fun(dim, d)
    basis_dδ = basis_fun(dim, d-δ)
    basis_δ = basis_fun(dim, δ)

    # shifting of the vectors corresponds to multipliciation by monomials of polynomials
    shift_mat = _mk_shift_mat(basis_dδ, basis_d, K)  # maps from R_≤d-δ to R_≤d

    # workaround for trac.sagemath.org/ticket/31234 (fixed in Sage 9.3)
    mat_prod_safe = lambda A, B: A * B if B.ncols() > 0 else matrix(K, A.nrows(), 0)
    # lives in R_≤d-δ
    W = [intersections.column_space_intersection(*[
            mat_prod_safe((Yα := shift_mat(yα)).T,
                          intersections.column_space_intersection(Vj, Yα, tol=tol, orthonormal=True))
                for yα in basis_δ
            ], tol=tol, orthonormal=True) for Vj in eigspaces]

    embedding = shift_mat(basis_d[0].parent().one())  # from R_≤d-δ to R_≤d
    UrH = Ur.H if K.characteristic() == 0 else Ur.T
    good = [j for j in range(len(W)) if W[j].ncols() > 0 and (UrH*embedding*W[j]).norm() > tol]
    return good


def eigenspaces_lifted(Ms, M=None, *, tol, convex=False, **kwds):
    from momentproblems import intersections
    ker_M = intersections.null_space_intersection(*Ms, tol=tol)

    if ker_M.ncols() > 0:
        if not ker_M.base_ring().is_exact():
            Ur = Vr = intersections.null_space(ker_M.H, tol=tol)  # orthogonal complement of kernel
            VrH = Vr.H
            Δs = [VrH * Mi * Vr for Mi in Ms]  # now Δ_i is a regular pencil
        else:
            Ur = Vr = ker_M.H.right_kernel_matrix().T
            VrH = Vr.H if ker_M.base_ring().characteristic() == 0 else Vr.T
            Δs = [VrH * Mi * Vr for Mi in Ms]
    else:
        # common kernel is trivial, so pencil is already regular
        Ur = Vr = matrix.identity(ker_M.base_ring(), Ms[0].ncols())
        VrH = Vr.H if ker_M.base_ring().characteristic() == 0 else Vr.T
        Δs = Ms

    if not convex:
        # TODO this ignores M for now
        Δ = None
    elif M is not None:
        # M should be a positive-semidefinite linear combination of the Ms
        # such that Δ is positive-definite.
        Δ = VrH * M * Vr
    else:
        # In this case, all Δi should be positive-semidefinite.
        # As Δs is a regular pencil, Δ is positive-definite, in particular
        # non-singular. The coordinates of the eigenvalues are then
        # computed with respect to the positive-definite matrix Δ.
        Δ = sum(Δs)

    if not ker_M.base_ring().is_exact():
        from momentproblems.commoneigenspaces import common_eigenspaces_numeric
        QZs = sorted(common_eigenspaces_numeric(Δs, tol=tol, B=Δ, **kwds), key=lambda QZ: QZ[0].ncols(), reverse=True)
        # right eigenspaces, lifted from coordinate ring to ambient ring, including null space (polynomials in the ideal), lives in R_≤d
        V = [(Ur * Z).augment(ker_M) for Q, Z, aa in QZs]
        eigvals = [aa for Q, Z, aa in QZs]
    else:
        from momentproblems.commoneigenspaces import common_eigenspaces_symbolic
        eZs = sorted(common_eigenspaces_symbolic(Δs, extend=False, multiplicities=True, B=Δ), key=lambda eZ: eZ[1].ncols(), reverse=True)
        V = [(Ur * Z).augment(ker_M) for aa, Z in eZs]
        eigvals = [aa for aa, Z in eZs]

    # they are indeed common eigenspaces, so Mi*Z is contained in a space of dimension equal to the multiplicity of the eigenvalue (or possibly lower)
    # assert (all([matrix_rank(matrix.column(flatten([(Mi * Z).columns() for Mi in Ms])), tol) == Z.ncols() - ker_M.ncols() for Z in V]))
    return V, eigvals, Ur

def eigenvalues_from_indices(eigvals, indices, *, exact=False):
    if exact:
        return matrix.column(eigvals)[:,indices]
    else:
        # computation of eigenvalues (using only first eigenvalue of multiple)
        return matrix.column([matrix(eigvals[j]).column(0).normalized() for j in indices])

def _basis_data(torus=False):
    from momentproblems import moment_functionals
    if torus:
        # maximal degree
        deg_max = lambda p: max(abs(a) for aa in p.exponents() for a in aa)
        deg_fun = deg_max
        basis_fun = moment_functionals.monomial_basis_laurent
        return basis_fun, deg_fun
    else:
        # total degree
        deg_fun = lambda p: p.degree()
        basis_fun = moment_functionals.monomial_basis
        return basis_fun, deg_fun

def convex_hull_vertices(eigvals, *, tol, exact):
    if exact:
        p = Polyhedron(vertices=eigvals)
        return p.vertices_list()
    else:
        # In case of multiple eigenvalues, all of them should be numerically
        # equal, so we only use the first.
        ee = matrix.column([matrix(e).column(0) for e in eigvals])
        if ee.ncols() <= 1:
            raise ValueError('found too few eigenvalues')
        if ee.base_ring() == RDF:
            p = Polyhedron(vertices=ee.columns())
            vs = p.vertices_list()
            if len(vs) != p.dimension() + 1:
                # TODO This is very naive and can fail numerically when some
                # eigenvalues that are not vertices lie on the boundary of the convex
                # hull.
                raise ValueError("failed to numerically compute convex hull")
            return vs
        elif ee.base_ring() == CDF:
            from scipy.spatial import ConvexHull
            # view complex vectors as real vectors
            ee2 = matrix(RDF, [row.apply_map(part) for row in ee.rows() for part in [real, imag]])
            eeU, eeS, eeV = ee2.SVD()
            rk = matrix_rank(eeS, tol)
            # project to lower-dimensional space (we arbitrarily drop the first
            # coordinate to go from affine space to subspace, potentially could make a better choice)
            ee3 = (eeU.H * ee2)[1:rk,:]
            # in this lower dimension, the convex hull is not degenerate
            p = ConvexHull(ee3.numpy().T)
            indices = sorted(p.vertices)
            dim = ee3.nrows()
            if len(indices) != dim + 1:
                # TODO This is very naive and can fail numerically when some
                # eigenvalues that are not vertices lie on the boundary of the convex
                # hull.
                raise ValueError("failed to numerically compute convex hull")
            return ee[:,indices].columns()
        else:
            raise AssertionError("base ring should be RDF or CDF")

# === the main algorithm ===
# If this fails with
# LinAlgError: generalized eig algorithm (ggev) did not converge (LAPACK info=66)
# then https://github.com/Reference-LAPACK/lapack/issues/475 is not fixed in
# the installed version of LAPACK.
def weights_from_moment_matrices_fixed_δ(Ms, d, δ, dim, tol, torus=False, **kwds):  # kwds: algorithm (if numeric)
    N = binomial(dim+d+δ, dim) if not torus else ((d+δ)+1) ** dim
    assert all(Mi.nrows() == Mi.ncols() == N for Mi in Ms)
    V, eigvals, Ur = eigenspaces_lifted(Ms, tol=tol, **kwds)
    basis_fun, deg_fun = _basis_data(torus=torus)
    good = shift_invariant_eigenvalue_indices(Ur, V, d=d+δ, δ=δ, dim=dim, tol=tol,
                                              basis_fun=basis_fun, deg_fun=deg_fun)
    # computation of eigenvalues (using only first eigenvalue of multiple)
    λ_ij = eigenvalues_from_indices(eigvals, good, exact=Ur.base_ring().is_exact())
    return λ_ij  # columns of λ_ij are Γ, up to scaling

# assumes that all matrices are positive-semidefinite
def weights_from_moment_matrices_convex(Ms, M=None, *, tol, **kwds):  # kwds: algorithm (if numeric)
    # If M is not None, it must be a positive-semidefinite linear combination
    # of the Ms with kernel not larger than the common kernel of the Ms. Then
    # eigenvalues are represented in coordinates with respect to M.
    V, eigvals, Ur = eigenspaces_lifted(Ms, M, tol=tol, convex=True, **kwds)
    if not eigvals:
        raise ValueError('did not find any eigenvalues')
    vs = convex_hull_vertices(eigvals, tol=tol, exact=Ur.base_ring().is_exact())  # TODO numeric case
    λ_ij = matrix.column(vs)
    return λ_ij


def eigenvalues_dimension(eigvals, *, exact, tol):
    """
    Return the dimension of the projective subspace spanned by the eigenvalues
    """
    if not exact:
        # only consider the first of a multiple eigenvalue
        ee = matrix.column([matrix(e).column(0) for e in eigvals])
        return matrix_rank(ee, tol=tol) - 1
    else:
        ee = matrix.column(eigvals)
        return ee.rank() - 1


# (here we work with d,d+δ)
def weights_from_moment_matrices(Ms0, d, dim, tol, torus=False, δ_MAX=10, **kwds):  # kwds: algorithm (if numeric)
    basis_fun, deg_fun = _basis_data(torus=torus)
    if not torus:
        indices_upto = lambda d1: range(binomial(dim+d1, dim))
    else:
        try:
            d_max = ZZ((Ms0[0].ncols() ** (1/dim) - 1))
        except TypeError:
            raise ValueError("toeplitz matrices expected to be of size (d_max+1)^dim, not %s" % Ms0[0].ncols())
        basis = basis_fun(dim=dim, d=d_max)
        indices_upto = lambda d1: [k for k, yα in enumerate(basis) if deg_fun(yα) <= d1]

    for δ in range(δ_MAX):
        indices = indices_upto(d+δ)
        N = len(indices)
        Ms = [Mi[indices, indices] for Mi in Ms0]
        if not all(Mi.nrows() == Mi.ncols() == N for Mi in Ms):
            raise ValueError(f"moment matrices need to be at least of size {N}")
        V, eigvals, Ur = eigenspaces_lifted(Ms, tol=tol, **kwds)

        r = eigenvalues_dimension(eigvals, exact=Ur.base_ring().is_exact(), tol=tol)
        assert len(eigvals) >= r+1
        if len(eigvals) == r+1:
            good = range(len(eigvals))
        else:
            good = shift_invariant_eigenvalue_indices(Ur, V, d=d+δ, δ=δ, dim=dim, tol=tol,
                                                      basis_fun=basis_fun, deg_fun=deg_fun)
            if len(good) < r+1:
                raise ValueError("found fewer than r+1 common eigenvalues")
            elif len(good) > r+1:
                continue
        # computation of eigenvalues (using only first eigenvalue of multiple)
        λ_ij = eigenvalues_from_indices(eigvals, good, exact=Ur.base_ring().is_exact())
        return λ_ij  # columns of λ_ij are Γ, up to scaling
    raise ValueError(f"δ_MAX = {δ_MAX} was reached")


def unmixed_moment_matrices(Ms, λ_ij):
    Ms_flat = matrix([Mi.list() for Mi in Ms])
    Hs_flat = λ_ij.solve_right(Ms_flat)
    Hs = [matrix(Hj.base_ring(), sqrt(len(Hj)), Hj) for Hj in Hs_flat.rows()]
    return Hs


def _check_weights(weights, λ_ij, tol):
    weights = matrix(weights)
    if weights.dimensions() != λ_ij.dimensions():
        return False
    if λ_ij.base_ring().is_exact():
        P = ProjectiveSpace(λ_ij.base_ring(), λ_ij.nrows()-1)
        aa = set(map(P, map(tuple, λ_ij.columns())))
        bb = set(map(P, map(tuple, weights.columns())))
        return len(aa) == λ_ij.ncols() and aa == bb
    # check that, up to scaling and permutation, the columns are the same
    # pick weight vector pointing in same direction as λ_ij column (singular value close to 0)
    indices = [min(range(weights.ncols()),
                   key=lambda j: v.column().augment(weights[:,j]).singular_values()[1]
                  ) for v in λ_ij.columns()]
    return (len(set(indices)) == len(indices)  # we get a permutation, so the indices must be distinct
            and all(λ_ij[:,j].augment(weights[:,indices[j]]).singular_values()[1] < tol for j in range(λ_ij.ncols())))
