from sage.all import RDF, CDF, matrix, prod
import scipy.linalg
import numpy as np

def column_space_intersection(*As, tol, orthonormal=False):
    r"""
    Return a matrix with orthonormal columns spanning the intersection of the
    column spaces of the given matrices.

    INPUT:

    - ``*As`` -- matrices with a fixed number of rows and linearly independent
      (or orthonormal) columns each
    - ``tol`` -- tolerance for truncating the singular values to determine the
      rank of the intersection
    - ``orthonormal`` -- boolean (default: ``False``); if ``True``, the columns
      of each matrix are assumed to be orthonormal

    ALGORITHM: Golub, van Loan -- Algorithm 12.4.3
    """
    if len(As) < 1:
        raise ValueError("at least one matrix required")
    n = As[0].nrows()
    for A in As:
        if A.nrows() != n:
            raise ValueError("matrices must have same number of rows")
    if all(A.base_ring().is_exact() for A in As):
        V = As[0].column_space()
        for A in As[1:]:
            V = V.intersection(A.column_space())
        return V.basis_matrix().T
    for A in As:
        if A.base_ring() not in (RDF, CDF):
            raise ValueError("only matrices over RDF/CDF or exact fields supported")

    if any(A.ncols() == 0 for A in As):
        return matrix(As[0].base_ring(), n, 0)
    Qs = As if orthonormal else [A.QR()[0][:,:A.ncols()] for A in As]
    if len(As) == 1:
        return Qs[0]

    # for better performance, we switch to numpy
    # Taking slices or hermitian transposes is a bottleneck with double dense matrices in Sage.
    Qs = [Q.numpy() for Q in Qs]

    # C = prod([Qs[0].H] + [Q*Q.H for Q in Qs[1:-1]] + [Qs[-1]])
    # sort Qs such that smallest matrix is last, second smallest first
    Q_last  = Qs.pop(min(range(len(Qs)), key=lambda j: Qs[j].shape[1]))
    Q_first = Qs.pop(min(range(len(Qs)), key=lambda j: Qs[j].shape[1]))
    C = Q_last
    for Q in Qs:  # without Q_last and Q_first
        C = Q @ (Q.conj().T @ C)  # this should be faster than (Q * Q.H) * C, since Q*Q.H is very large
    C = Q_first.conj().T @ C

    Σ, Vh = scipy.linalg.svd(C, overwrite_a=True)[1:]  # we can overwrite, since C involves at least 1 multiplication
    rk = np.sum(1-Σ < tol)
    return matrix(Q_last @ Vh.T[:,:rk].conj())


def null_space_intersection(*As, tol):
    r"""
    Return a matrix with orthonormal columns spanning the intersection of the
    null spaces of the given matrices.

    INPUT:

    - ``*As`` -- matrices with a fixed number of columns
    - ``tol`` -- tolerance for truncating the singular values to determine the
      rank of intermediate results

    ALGORITHM: Golub, van Loan -- Algorithm 12.4.2
    """
    if len(As) < 1:
        raise ValueError("at least one matrix required")
    n = As[0].ncols()
    if all(A.base_ring().is_exact() for A in As):
        ker = As[0].right_kernel()
        for A in As[1:]:
            ker = ker.intersection(A.right_kernel())
        # TODO document that this does not have orthonormal columns
        return ker.basis_matrix().T
    for A in As:
        if A.base_ring() not in (RDF, CDF):
            raise ValueError("only matrices over RDF/CDF or exact rings supported")
        if A.ncols() != n:
            raise ValueError("matrices must have same number of columns")
    Y = None
    for A in As:
        if A.nrows() == 0:
            continue
        C = A * Y if Y is not None else A
        Σ, V = C.SVD()[1:]
        q = len([s for s in Σ.diagonal() if s > tol])
        if q >= C.ncols():
            return matrix(As[0].base_ring(), n, 0)
        X = V[:, q:]
        Y = Y * X if Y is not None else X
    if Y is None:
        # all the matrices have 0 rows
        return matrix.identity(As[0].base_ring(), n)
    else:
        return Y


def null_space(A, tol):
    import numpy
    import scipy.linalg
    if A.nrows() == 0:
        return matrix.identity(A.base_ring(), A.ncols())
    return matrix(numpy.ascontiguousarray(scipy.linalg.null_space(A, rcond=tol)))


def _tests_sage():
    """
    TESTS::

        sage: from momentproblems import intersections
        sage: TestSuite(intersections._tests_sage()).run(skip='_test_pickling')
    """
    from sage.all import SageObject, matrix, RDF, ZZ
    import numpy
    import numpy.linalg
    import scipy.linalg

    class Tests(SageObject):

        def matrices(self):
            # test data
            for _ in range(5):
                for num in range(1, 5):
                    # generate some matrices with few rows, so we can intersect their kernels
                    matrices = [matrix.random(RDF, ZZ.random_element(0, 4), 9) for _ in range(num)]
                    yield matrices

        def matrices2(self):
            # test data
            for _ in range(5):
                for num in range(1, 5):
                    # generate some matrices with few rows, so we can intersect their kernels
                    matrices = [matrix.random(RDF, 9, 9 - ZZ.random_element(0, 4)) for _ in range(num)]
                    yield matrices

        def equal_spaces(self, A, B, tol):
            from numpy.linalg import matrix_rank
            return matrix_rank(A.augment(B), tol) == matrix_rank(A, tol) == matrix_rank(B, tol)

        def _test_null_space_intersection(self, **kwds):
            tol = 1e-10
            for As in self.matrices():
                ker = null_space_intersection(*As, tol=tol)
                assert all([ker.ncols() == 0 or A.nrows() == 0 or (A * ker).norm() < tol for A in As])
                assert max(0, As[0].ncols() - sum([A.nrows() for A in As])) == ker.ncols()  # generically the correct dimension
                # the intersection is also simply the null space of the augmented matrix
                ker2 = null_space(matrix(RDF, [v for A in As for v in A.rows()], ncols=As[0].ncols()), tol)
                assert self.equal_spaces(ker, ker2, tol)

        def _test_column_space_intersection(self, **kwds):
            tol = 1e-10
            for As in self.matrices2():
                B = column_space_intersection(*As, tol=tol)
                assert B.ncols() == max(0, As[0].nrows() - sum([A.nrows() - A.ncols() for A in As]))  # generically the correct dimension
                for A in As:
                    assert self.equal_spaces(A.augment(B), A, tol)  # B is contained in A

        def _test_compatibilty(self, **kwds):
            tol = 1e-10
            for As in self.matrices():
                # computing null space intersection is the same as computing
                # column space intersection of null spaces
                ker = null_space_intersection(*As, tol=tol)
                ker2 = column_space_intersection(*[null_space(A, tol) for A in As], tol=tol, orthonormal=True)
                assert self.equal_spaces(ker, ker2, tol)

    return Tests()
