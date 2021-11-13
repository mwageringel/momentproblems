import itertools
from sage.all import matrix, vector, binomial, ZZ, sqrt, ProjectiveSpace, RDF, exp, log, atan2, CDF, sin, cos
from scipy.linalg import ordqz, eigvals, svdvals
import numpy as np
import scipy.cluster

def _is_triangular(T, tol=1e-15):  # or trapezoidal
    return np.all(np.abs(np.tril(T, k=-1)) < tol)

def _eigenspace_right_from_eigenvalue(A, e):
    return (A - e * matrix.identity(A.base_ring(), A.ncols())).right_kernel()

def _eigenspaces_right_no_extend(A):
    es = set(A.eigenvalues(extend=False))
    return [(e, _eigenspace_right_from_eigenvalue(A, e)) for e in es]

def common_eigenspaces_symbolic(As, *, extend=False, max_tries=30, multiplicities=False, B=None):
    r"""
    TESTS::

        sage: from momentproblems.commoneigenspaces import common_eigenspaces_symbolic
        sage: K = GF(101)
        sage: As = [
        ....:     matrix.diagonal(K, [2,2,3,3,4]),
        ....:     matrix.diagonal(K, [3,2,3,3,5]),
        ....:     matrix.identity(K, 5),
        ....:     matrix.diagonal(K, [5,2,4,4,1])]
        sage: eigvals, eigmat = common_eigenspaces_symbolic(As, extend=False)
        sage: eigmat.dimensions()
        (5, 5)
        sage: Γ = sorted(eigvals); Γ
        [(1 : 1 : 51 : 1),
         (4 : 5 : 1 : 1),
         (26 : 26 : 76 : 1),
         (26 : 26 : 76 : 1),
         (61 : 41 : 81 : 1)]
        sage: assert all([all(γ[j] * As[l] * v == γ[l] * As[j] * v
        ....:                 for j in range(1,len(γ)) for l in range(j))
        ....:             for γ, v in zip(eigvals, eigmat.columns())])

        sage: P, Q = [matrix.random(K, As[0].ncols(), algorithm='unimodular') for _ in range(2)]
        sage: Bs = [P * A * Q for A in As]
        sage: eigvals2, eigmat2 = common_eigenspaces_symbolic(Bs, extend=False)
        sage: sorted(eigvals2) == Γ
        True
    """
    # naive implementation
    K = As[0].base_ring()
    assert all(A.base_ring() is K for A in As)
    if not K.is_exact():
        raise ValueError("field is not exact: %s" % K)
    if extend:
        Kbar = K.algebraic_closure()
        if Kbar != K:
            As = [A.change_ring(Kbar) for A in As]
        _eigenspaces = lambda A: A.eigenspaces_right()
    else:
        Kbar = K
        _eigenspaces = _eigenspaces_right_no_extend

    preserve_coordinates = B is not None
    if B is not None:
        # convex case
        if B.rank() != B.ncols():
            raise ValueError("B should be an invertible matrix which is a linear combination of As")
        B2 = B
    else:
        B2 = As[-1]
        if B2.rank() != B2.ncols():
            for _ in range(max_tries):
                B2 = sum(K(ZZ.random_element(len(As))) * A for A in As)
                if B2.rank() == B2.ncols():
                    break
            else:
                raise ValueError("could not find regular linear combination of pencil;"
                                 " pencil might be singular")

    # transform the pencil by the invertible matrix B, so that generalized
    # eigenvalues become ordinary eigenvalues
    # (does not alter eigenvalues or eigenvectors)
    B2inv = ~B2
    As = [B2inv * A for A in As]

    eig_spaces = [([e], V) for e, V in _eigenspaces(As[0])]
    for A1 in As[1:]:
        eig_spaces_new = []
        if all(Vj.dimension() == 1 for ej, Vj in eig_spaces):
            # If all the common eigenspaces are already 1-dimensional, we do
            # not need to compute eigenvalues of A1, but merely need to check
            # if computed eigenvectors are eigenvectors of A1
            # eig_values1 = None
            eig_spaces1 = None
        else:
            # eig_values1 = A1.eigenvalues(extend=extend)
            eig_spaces1 = _eigenspaces(A1)
        for ej, Vj in eig_spaces:
            # possibly split the space, or remove it if there is no common intersection
            if Vj.dimension() == 1:
                vj = Vj.gen(0)
                wj = A1 * vj
                if wj in Vj:
                    ek = wj / vj
                    eig_spaces_new.append((ej + [ek], Vj))
            else:
                # TODO this direct approach does not work yet
                # Vjmat = Vj.basis_matrix().T
                # for ek in eig_values1:
                #     # intersection of Vj with eigenspaces Vk of A1
                #     Wmat = (A1 * Vjmat - ek * Vjmat).right_kernel_matrix().T
                #     if Wmat.ncols() > 0:
                #         Vjk = Vj.subspace_with_basis((Vjmat * Wmat).columns(), check=False)
                #         assert Vjk.dimension() > 0
                #         eig_spaces_new.append((ej + [ek], Vjk))
                for ek, Vk in eig_spaces1:
                    Vjk = Vj.intersection(Vk)
                    if Vjk.dimension() > 0:
                        eig_spaces_new.append((ej + [ek], Vjk))
        eig_spaces = eig_spaces_new

    # as we work symbolically, there is no harm in returning projective
    # coordinates, which normalizes last coordinate,
    # unless we are in the convex case and want to preserve coordinates
    if preserve_coordinates:
        P = lambda x: x
    else:
        P = ProjectiveSpace(Kbar, len(As)-1)
    if multiplicities:
        return list((P(e), V.basis_matrix().T) for e, V in eig_spaces)
    else:
        eigvals = [P(e) for e, V in eig_spaces for _ in range(V.dimension())]
        eigmat = matrix.column(Kbar, len(eigvals), As[0].ncols(), entries=[v for e, V in eig_spaces for v in V.basis_matrix().rows()])
        return eigvals, eigmat

def common_eigenspaces_numeric(Δs, *, tol, B=None, algorithm='fast'):
    """
    TESTS:

    Generate input::

        sage: from momentproblems import *
        sage: from momentproblems.commoneigenspaces import common_eigenspaces_numeric
        sage: tol = 1e-7
        sage: functionals = [moment_functionals.mk_moment_ellipse(center=xy, radius=(1,1)) for xy in [(0,0), (-2,1), (2,-1)]]
        sage: weights = [[4, 2, 1], [2, 3, 6], [1, 3, 5], [2, 1, 2]]
        sage: Hs = [moment_functionals.hankel(f, dim=2, d=6) for f in functionals]
        sage: Ms = [Hs[0].parent().sum(weights[i][j] * Hs[j] for j in range(len(Hs))) for i in range(len(weights))]
        sage: ker_M = intersections.null_space_intersection(*Ms, tol=tol)
        sage: Vr = intersections.null_space(ker_M.H, tol=tol)
        sage: VrH = Vr.H
        sage: Δs = [VrH * Mi * Vr for Mi in Ms]  # now, Δ_i is a regular pencil

    This matrix is non-singular, as it is a positive linear combination of
    positive-semidefinite matrices forming a regular pencil::

        sage: Δ = sum(Δs)

    Check that we have computed eigenvectors and eigenvalues with coordinates
    choosen with respect to Δ::

        sage: eigspaces = list(common_eigenspaces_numeric(Δs, tol=tol, B=Δ, algorithm='fast'))
        sage: len(eigspaces)
        4
        sage: for Q, Z, ee in eigspaces:
        ....:     ee = matrix(ee)
        ....:     for i in range(len(Δs)):
        ....:         for k in range(ee.ncols()):
        ....:             assert ((Δs[i] - ee[i,k] * Δ) * Z[:,k]).norm() < tol

        sage: eigspaces = list(common_eigenspaces_numeric(Δs, tol=tol, B=Δ, algorithm='cluster'))
        sage: len(eigspaces)
        4
        sage: for Q, Z, ee in eigspaces:
        ....:     ee = matrix(ee)
        ....:     for i in range(len(Δs)):
        ....:         for k in range(ee.ncols()):
        ....:             assert ((Δs[i] - ee[i,k] * Δ) * Z[:,k]).norm() < tol

    With unspecified choice of coordinates::

        sage: eigspaces = list(common_eigenspaces_numeric(Δs, tol=tol))
        sage: len(eigspaces)
        4
        sage: for Q, Z, ee in eigspaces:
        ....:     ee = matrix(ee)
        ....:     for i in range(len(Δs)):
        ....:       for j in range(i):
        ....:         for k in range(ee.ncols()):
        ....:             scale = 0.1 / (abs(ee[j,k]) + abs(ee[i,k]))  # TODO an attempt to rescue the accuracy
        ....:             assert (scale * (ee[j,k] * Δs[i] - ee[i,k] * Δs[j]) * Z[:,k]).norm() < tol

    ::

        sage: As = [
        ....:     matrix.diagonal(RDF, [2,2,3,3,4], sparse=False),
        ....:     matrix.diagonal(RDF, [3,2,3,3,5], sparse=False),
        ....:     matrix.diagonal(RDF, [1,1,1,1,1], sparse=False),
        ....:     matrix.diagonal(RDF, [5,2,4,4,1], sparse=False)]
        sage: eigspaces = list(common_eigenspaces_numeric(As, B=matrix.identity(RDF, 5), tol=1e-13, algorithm='cluster'))
        sage: list(map(matrix.column, sorted([matrix(ee).columns() for Q,Z,ee in eigspaces])))
        [
        [2.0]  [2.0]  [3.0 3.0]  [4.0]
        [2.0]  [3.0]  [3.0 3.0]  [5.0]
        [1.0]  [1.0]  [1.0 1.0]  [1.0]
        [2.0], [5.0], [4.0 4.0], [1.0]
        ]
    """
    # If B is passed, a non-singular linear combination of Δs, the coordinates
    # are computed with respect to B, otherwise they are unspecified.
    if B is not None:
        proj_coordinates = False  # in this case, coordinates of eigenvalues are chosen with respect to B
    else:
        proj_coordinates = True
        # Set B to a random linear combination, which is generically regular if
        # the pencil is regular.
        # We currently do not choose random weights because it
        # occasionally non-deterministically leads to test failures, possibly
        # because we need to be more careful about `tol`.
        # weights = vector(RDF, [RDF.random_element(0.5, 1) for _ in range(len(Δs))]).normalized()
        weights = vector(RDF, [RDF(1) for _ in range(len(Δs))])  # not normalizing can sometimes avoid LAPACK convergence issues
        B = Δs[0].parent().sum(weights[i] * Δs[i] for i in range(len(Δs)))

    B_np = B.numpy()
    Δs_np = [Δi.numpy() for Δi in Δs]

    if algorithm == 'cluster':
        yield from _common_eigenspaces_cluster(Δs_np, B_np, tol)
    elif algorithm == 'fast':
        yield from _common_eigenspaces_fast(Δs, Δs_np, B, B_np, tol, proj_coordinates)
    else:
        raise ValueError("algorithm unknown: %s" % algorithm)


def _common_eigenspaces_fast(Δs, Δs_np, B, B_np, tol, proj_coordinates):
    # This naively computes common eigenspaces only if they do not need to be
    # subdivided further and only if eigenvalues of B are not close to zero.
    Ds = [Δi.eigenvalues(B, tol=tol) for Δi in Δs]
    for eig, mult in Ds[0]:
        T0, TB, aa, bb, Q, Z = ordqz(Δs_np[0], B_np, sort=lambda a, b: abs(a / b - eig) < tol)
        # T0, TB, Q, Z = map(matrix, (T0, TB, Q, Z))
        # assert matrix(TB - Q.H * Δs[0] * Z).norm() < tol
        # assert matrix(T0 - Q.H * Δs[1] * Z).norm() < tol
        # assert _is_triangular(TB) and _is_triangular(T0)

        # now, if the leading columns of T1,… are triangular, this should be a common eigenspace
        T1s = [Q.conj().T @ Δi @ Z for Δi in Δs_np[1:]]
        TBdiag = TB.diagonal()[:mult]
        if all(_is_triangular(Ti[:,:mult], tol=tol) for Ti in T1s):
            yield (matrix(np.ascontiguousarray(Q[:,:mult])),
                   matrix(np.ascontiguousarray(Z[:,:mult])),
                   np.array([Ti.diagonal()[:mult] for Ti in [T0] + T1s] if proj_coordinates else
                            [Ti.diagonal()[:mult] / TBdiag for Ti in [T0] + T1s]))


def _common_eigenspaces_cluster(Δs_np, B_np, tol):
    r"""
    Compute common eigenvalues by an approach based on bisection by splitting
    generalized eigenvalues of (Δs_np[0], B_np) into two clusters, and then
    continuing the search with (Δs_np[1], B_np), etc. until only common
    eigenvalues are left.

    This should avoid some numerical issues that can arise if one computes all
    (rather than just two clusters) generalized eigenvalues of a 2-pencil.
    """
    # For clustering, this adhoc transformation continuously maps (a:b) ∈ ℂP^1
    # to some complex number that only depends on ratio without computing ratio.
    # (There is room for improvement as this is not uniform in radial/imaginary
    # direction, but this naive metric still seems to give better results than e.g.
    # Fubini-Study/spherical metric using identification of ℂP^1 with S^2.)
    trafo = lambda a,b: CDF(exp(atan2(CDF(a), CDF(b)) * complex(0,2)))
    trafo_inv = lambda c: vector(CDF, [sin(z := (log(c)/complex(0,2))), cos(z)]).normalized()
    from sage.all import pi
    dist = lambda x,y: abs((ab0 := trafo_inv(x))[0] * (ab1 := trafo_inv(y))[1] - ab0[1]*ab1[0])  # chordal metric (note that ab0, ab1 are normalized)
    # sanity checks:
    # all(abs((trafo(a := CDF.random_element(), b := CDF.random_element())) - trafo(a/b, 1)) < 1e-12 for _ in range(100))
    # all(abs(trafo(*trafo_inv(c := trafo(CDF.random_element(), CDF.random_element()))) - c) < 1e-12 for _ in range(100))
    # all(abs(dist((trafo(*(v := (CDF^2).random_element()).normalized())), trafo(*v))) < 1e-12 for _ in range(100))

    Δs_np_orig = Δs_np
    def loop(Δs_np, B_np, j, Qs, Zs, skip_count):
        j = j % len(Δs_np)
        if B_np.shape[0] == 1 or skip_count == len(Δs_np):
            # no clusters left
            z = np.linalg.multi_dot(Zs) if len(Zs) > 1 else Zs[0]  # may consist of several columns in case of multiple eigenvalues
            Δz = np.concatenate([Δi @ z[:,:1] for Δi in Δs_np_orig], 1)  # TODO only check first column, or all individually?
            svals = svdvals(Δz)
            # if columns of Δz are linearly dependent and not zero, z is a common eigenvector
            if svals[0] > tol and svals[1] < tol:  # if skip_count == len(Δs_np) we may ignore second check?
                q = np.linalg.multi_dot(Qs) if len(Qs) > 1 else Qs[0]
                yield matrix(q), matrix(z), np.array([eigvals(Δi, B_np) for Δi in Δs_np])
            return

        # The following can fail if LAPACK's eigenvalue computation/QZ iteration does not converge.
        # See https://github.com/Reference-LAPACK/lapack/issues/475 (fixed in openblas 0.3.14+)
        aa, bb = eigvals(Δs_np[j], B_np, homogeneous_eigvals=True)
        # transform homogeneous eigenvalues to some comparable complex numbers
        cc = np.array([trafo(a, b) for a,b in zip(aa.tolist(), bb.tolist())])
        cc2 = np.stack([cc.real, cc.imag], axis=1)
        # sorting for more deterministic results
        clusters = sorted([CDF(*c) for c in scipy.cluster.vq.kmeans(cc2, 2)[0]])  # split eigenvalues into 2 clusters that are hopefully well-separated
        if len(clusters) < 2 or dist(*clusters) < tol:
            # no clustering needed for this j, so continue (but not too often)
            # Instead of blindly continuing with j+1, we could choose the
            # coordinate with the best-separated clusters - then the skipping/termination condition must be adjusted -
            # but computing best coordinate is slower.
            yield from loop(Δs_np, B_np, j+1, Qs, Zs, skip_count+1)
            return

        skip_count = 0
        do_label = lambda aa, bb: np.abs((tab := np.apply_along_axis(lambda ab: trafo(*ab), 0, [aa, bb])) - clusters[0]) < np.abs(tab - clusters[1])
        cluster0size = np.sum(do_label(aa, bb))
        for cols, sort in zip([range(cluster0size), range(len(aa)-cluster0size)],
                              [do_label, lambda aa,bb: ~(do_label(aa, bb))]):
            Tj, TB, aa, bb, Q, Z = ordqz(Δs_np[j], B_np, sort=sort)
            Z1 = Z[:,cols]
            Q1 = Q[:,cols]
            Q1H = Q1.T.conj()
            Zs.append(Z1); Qs.append(Q1)
            cols2 = np.ix_(cols, cols)
            yield from loop([Tj[cols2] if i==j else Q1H @ Δs_np[i] @ Z1 for i in range(len(Δs_np))], TB[cols2], j+1, Qs, Zs, skip_count)
            Zs.pop(); Qs.pop()

    yield from loop(Δs_np, B_np, 0, [], [], 0)
