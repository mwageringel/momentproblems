r"""
Exact moment functionals for circles, ellipses and other varieties with respect
to uniform measure.


EXAMPLES::

    sage: from momentproblems.moment_functionals import *

::

    sage: moment_ellipse = mk_moment_ellipse(center=(1,1), radius=(5,4))
    sage: h0 = hankel(moment_circle, dim=2, d=2)
    sage: h1 = hankel(moment_ellipse, dim=2, d=2)
    sage: h0.right_kernel_matrix()
    [ 1  0  0 -1  0 -1]
    sage: h1.right_kernel_matrix()
    [      1  32/359  50/359 -16/359       0 -25/359]

::

    sage: mom_a = mk_moment_torus_line(ratio=(1, 1), translate=e^(i*pi/4))
    sage: mom_b = mk_moment_torus_line(ratio=(-1, 1), translate=1)
    sage: mom_c = mk_moment_torus_line(ratio=(-2, 1), translate=1)
    sage: mom_d = mk_moment_torus_line(ratio=(2, 1), translate=e^(i*1))
    sage: mom_e = mk_moment_torus_line(ratio=(2, 1), translate=2)

    sage: T_c = toeplitz(mom_c, dim=2, d=3)
    sage: T_c.is_hermitian()
    True
    sage: T_e = toeplitz(mom_e, dim=2, d=3)
    sage: T_e.is_hermitian()
    False

The Toeplitz moment matrices have kernels which allow to recover the expected ideals::

    sage: basis = vector(monomial_basis_laurent(dim=2, d=3))
    sage: ker = T_c.right_kernel_matrix() * basis
    sage: basis.base_ring().ideal(list(ker)).groebner_basis()
    (z0^2*z1 - 1,)

    sage: ker = T_e.right_kernel_matrix() * basis
    sage: basis.base_ring().ideal(list(ker)).groebner_basis()
    (z0^2 - 1/2*z1,)

Affine line segments::

    sage: mk_moment_affine_line_segment((0,0,0), (1,2,3), normalize=False)((1,2,1)).radical_expression()
    12/5*sqrt(14)
    sage: mk_moment_affine_line_segment((0,0,0), (1,2,3))((1,2,1))
    12/5
    sage: mk_moment_affine_line_segment((1,0,0), (2,0,0))((2,0,0)) == 7/3
    True
    sage: mk_moment_affine_line_segment((0,1,0), (0,2,0))((0,2,0)) == 7/3
    True

The kernel of the Hankel matrix recovers the ideal corresponding to the line
defined by start and end points::

    sage: start, end = (2,3,1), (3,4,0)
    sage: mom = mk_moment_affine_line_segment(start, end)
    sage: basis = vector(monomial_basis(dim=3, d=2))
    sage: H = hankel(mom, dim=3, d=2)
    sage: ker = H.right_kernel_matrix() * basis
    sage: gb = basis.base_ring().ideal(list(ker)).groebner_basis(); gb
    [y0 + y2 - 3, y1 + y2 - 4]
    sage: all(f(*pnt).is_zero() for f in gb for pnt in [start, end])
    True

We construct a moment matrix corresponding to points lying on a trigonometric
curve.
We normalize by the number of points, so that the 0-th moment becomes 1::

    sage: R.<x1,x2> = QQ[]
    sage: g = -1/4 + x1 + x2 + x1*x2
    sage: points = list(generate_points_on_trigonometric_variety(R.ideal(g), gridsize=6, parent=RDF))
    sage: basis = monomial_basis_laurent(dim=R.ngens(), d=2)
    sage: A = vandermonde(points, basis, trigonometric=True)
    sage: A.dimensions()
    (18, 9)
    sage: T = A.H * A / len(points)
    sage: T.is_hermitian()
    True
    sage: T.zero_at(1e-8)[:4,:4].n(30)
    [                  1.0000000                 0.069444444                 -0.55381944   0.37037037 + 0.11111111*I]
    [                0.069444444                   1.0000000                 0.069444444 -0.18981481 + 0.027777778*I]
    [                -0.55381944                 0.069444444                   1.0000000 -0.40219907 - 0.097222222*I]
    [  0.37037037 - 0.11111111*I -0.18981481 - 0.027777778*I -0.40219907 + 0.097222222*I                   1.0000000]
    sage: T.zero_at(1e-8)[-4:,-4:].n(30)
    [                  1.0000000 -0.40219907 - 0.097222222*I -0.18981481 + 0.027777778*I   0.37037037 + 0.11111111*I]
    [-0.40219907 + 0.097222222*I                   1.0000000                 0.069444444                 -0.55381944]
    [-0.18981481 - 0.027777778*I                 0.069444444                   1.0000000                 0.069444444]
    [  0.37037037 - 0.11111111*I                 -0.55381944                 0.069444444                   1.0000000]
    sage: T.zero_at(1e-12).diagonal()  # abs tol 1e-12
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

The matrix has one-dimensional kernel, as the relation defined by `g` is in the
kernel::

    sage: from numpy.linalg import matrix_rank
    sage: T.ncols() - matrix_rank(T, tol=1e-13)
    1
    sage: ker = T.SVD()[0][:,-1]
    sage: (ker / ker[0,0]).zero_at(1e-12)  # abs tol 1e-12
    [ 1.0]
    [ 2.0]
    [ 1.0]
    [ 2.0]
    [-1.0]
    [ 2.0]
    [ 1.0]
    [ 2.0]
    [ 1.0]
"""

import itertools
from sage.all import (cached_function, QQ, QQbar, AA, ZZ, RDF, CDF, gamma, cos, acos, pi, gcd, exp,
                      PolynomialRing, Family, vector, LaurentPolynomialRing, Integer,
                      DisjointUnionEnumeratedSets, IntegerVectors, matrix, NN, SR)
import numpy as np

@cached_function(key=tuple)
def moment_circle(aa):
    a,b = map(Integer, aa)
    # explicit formula:
    res = gamma((a+1)/2) * gamma((b+1)/2) / gamma(Integer(1)/2)**2 / gamma((a+b+2)/2) * cos(pi/2*a)**2 * cos(pi/2*b)**2
    assert res.parent() is SR
    return QQ(res)

def eval_functional(moment_functional, p):
    return p.parent().base_ring().sum(
            c * moment_circle(e) for e, c in p.iterator_exp_coeff())

def mk_moment_ellipse(center, radius):
    R = PolynomialRing(QQ, 2, 'y')
    trafo = R.hom([radius[0] * R.gen(0) + center[0], radius[1] * R.gen(1) + center[1]], R)

    @cached_function(key=tuple)
    def moment_ellipse(aa):
        return eval_functional(moment_circle, trafo(R.monomial(*aa)))
    return moment_ellipse

def monomial_basis(dim, d=None):
    R = PolynomialRing(QQ, 'y', dim)
    if d is None:
        basis = DisjointUnionEnumeratedSets(Family(NN, lambda d:
                    IntegerVectors(d, dim))).map(lambda v: R.monomial(*v))
        return basis
    else:
        return list(itertools.takewhile(lambda q: q.degree() <= d, monomial_basis(dim)))

def hankel(moment_functional, dim=None, d=None, basis=None):
    if basis is None:
        if dim is None:
            raise ValueError("basis or dim must be specified")
        basis = monomial_basis(dim)
    mons = list(basis if d is None else itertools.takewhile(lambda yα: yα.degree() <= d, basis))
    def monomial_exp(g):
        exps = g.exponents()
        if len(exps) != 1:
            raise NotImplementedError("only monomial bases supported")
        return exps[0]
    return matrix([[moment_functional(monomial_exp(yα*yβ)) for yα in mons] for yβ in mons])


def mk_moment_torus_line(ratio, translate):
    r"""
    Construct a moment functional of a straight line on the two-dimensional torus.

    ratio = (q1, q2)  -- coprime
    translate = c1    -- non-zero scalar  (if on unit circle, then the Toeplitz moment matrix is Hermitian)

    where::

        q2*t2        == q1*t1        + b1             | exp(i*_)
        exp(i*q2*t2) == exp(i*q1*t1) * exp(i*b1)      | c1 = exp(i*b1)
        z2^q2        == z1^q1        * c1
    """
    assert gcd(*ratio) == 1
    assert translate != 0
    qq = vector([-ratio[0], ratio[1]])  # the smallest point on the line

    def moment_torus_line(aa):
        aa = vector(aa)
        a1, a2 = aa
        if a1 * ratio[1] + a2 * ratio[0] != 0:  # since we only work with lines, we can restrict to a 1-dimensional sublattice
            # i.e. matrix([aa, qq]).det() != 0
            return 0
        distance = ZZ(aa / qq)
        return translate ** distance  # cf. N.1946
    return moment_torus_line

def monomial_basis_laurent(dim, d):
    R = LaurentPolynomialRing(QQ, 'z', dim)
    # We only pick non-negative exponents
    return tuple(R.monomial(*aa) for aa in itertools.product(range(0, d+1), repeat=dim))

def toeplitz(moment_functional, dim, d):
    # Only non-negative exponents
    alphas = [vector(aa, immutable=True) for aa in itertools.product(range(0, d+1), repeat=dim)]
    # The following also involves negative exponents. This is justified since
    # the toeplitz matrix for exponent vectors -α+β is the same for
    # 0 ≤ α_i,β_i ≤ 2d as the one for -d ≤ α_i,β_i ≤ d.
    return matrix([[moment_functional(-a+b) for b in alphas] for a in alphas])


def mk_moment_affine_line_segment(start, end, *, normalize=True):
    start = vector(start)
    end = vector(end)
    K = QQ if normalize else QQbar
    assert start.base_ring().is_subring(K)
    assert end.base_ring().is_subring(K)

    S = K['t']
    t = S.gen(0)
    γ = start + t * (end - start)
    if not normalize:
        γ_norm = abs(γ.derivative(t))
        assert γ_norm.parent() is K
    else:
        # 0-th moment is γ_norm, so we normalize such that 0-th moment becomes
        # 1 for convenience, as it allows us to avoid square roots and stay in
        # QQ.
        γ_norm = 1

    R = PolynomialRing(K, 'x', start.degree())

    @cached_function(key=tuple)
    def moment_affine_line_segment(aa):
        f = R.monomial(*aa)
        return S(f(*γ) * γ_norm).integral(t)(1)  # t ∈ [0,1]
    return moment_affine_line_segment


def mk_moment_trigonometric_curve_example(c1, c2, *, prec=80):
    r"""
    Return the moment functional for the uniform measure on a trigonometric
    example curve.

    The curve depends on two parameters, defining a family of curves. The curve
    is defined in degrees -1..+1.

    If c1 > c2, the curve is connected, if c1 < c2 it has two disconnected
    parts.  The parameters must satisfy c1 != c2 and c1 < c2 + 2.

    EXAMPLES::

        sage: from momentproblems.moment_functionals import *
        sage: mom = mk_moment_trigonometric_curve_example(5/8, 0)
        sage: [mom(a) for a in [(0,0), (1,0), (0,1)]]
        [[12.61257092367025016329 +/- ...],
         [2.753059781992467887758 +/- ...] + [+/- ...]*I,
         [2.753059781992467887758 +/- ...] + [+/- ...]*I]

    ::

        sage: mom = mk_moment_trigonometric_curve_example(1, 5/4)
        sage: [mom(a) for a in [(0,0), (1,0), (0,1)]]
        [[13.448678364566850580435 +/- ...],
         [-0.408926940531123205236 +/- ...] + [+/- ...]*I,
         [0.060306710833167660509 +/- ...] + [+/- ...]*I]
    """
    if not (c1 > 0 and c1 != c2 and 2+c2 > c1):
        raise ValueError("parameter out of bounds")
    from sage.all import ComplexBallField, fast_callable, sqrt
    i = SR.I()
    t1 = SR.var('t1')
    T2 = acos(-1 + ZZ(2)*c1 / (cos(t1) + 1 + c2))
    T1 = acos(-1 - c2 + ZZ(2)*c1 / (cos(t1) + 1))  # t1, t2 coordinates swapped

    KK = ComplexBallField(prec)
    gam = vector([exp(i*t1), exp(i * T2)])
    gam1 = vector([gk.derivative(t1) for gk in gam])

    if c2 > c1:
        # in this case, the curve consists of two smooth pieces above and below x-axis
        def _moment_lebesgue_half(a1, a2):
            h = exp(i*t1*a1 + i*T2*a2)
            fc = fast_callable(h * gam1.norm(), vars=[t1], domain=KK)
            res = KK.integral(lambda t,_: fc(t), KK(-pi), KK(pi))
            return res

        @cached_function(key=tuple)
        def moment_lebesgue(aa):
            a1, a2 = aa
            north = _moment_lebesgue_half(a1,a2)
            south = _moment_lebesgue_half(-a1,-a2)  # could make use of symmetries (conjugation)
            return north + south

    else:
        t1_a = -acos(-ZZ(1)/2*c2 - 1 + ZZ(1)/2*sqrt(8*c1+c2**2))
        t1_b = acos(-ZZ(1)/2*c2 - 1 + ZZ(1)/2*sqrt(8*c1+c2**2))
        if KK(t1_a).is_NaN() or KK(t1_b).is_NaN():
            raise ValueError("parameter out of bounds")

        def _moment_lebesgue_quarter(a1, a2):  # the moment over the top quarter of the curve (from diagonal to diagonal)
            # An accurate and fast implementation using complex ball fields.
            # This allows us to directly compute the complex (path) integral.
            h = exp(i*t1*a1 + i*T2*a2)
            fc = fast_callable(h * gam1.norm(), vars=[t1], domain=KK)
            res = KK.integral(lambda t,_: fc(t), KK(t1_a), KK(t1_b))
            return res

        gam_b = vector([exp(i*t1), exp(i * T1)])
        gam1_b = vector([gk.derivative(t1) for gk in gam_b])

        def _moment_lebesgue_quarter_2(a1, a2):  # the moment over the other quarter
            h = exp(i*t1*a1 + i*T1*a2)
            fc = fast_callable(h * gam1_b.norm(), vars=[t1], domain=KK)
            res = KK.integral(lambda t,_: fc(t), KK(t1_a), KK(t1_b))
            return res

        @cached_function(key=tuple)
        def moment_lebesgue(aa):
            a1, a2 = aa
            # The full moments are the sum over the quarters.
            north = _moment_lebesgue_quarter(a1,a2)
            south = _moment_lebesgue_quarter(-a1,-a2)  # could make use of symmetries (conjugation)
            west = _moment_lebesgue_quarter_2(-a2,a1)
            east = _moment_lebesgue_quarter_2(a2,-a1)
            return north + south + west + east

    return moment_lebesgue

def mk_moment_trigonometric_curve_example1(c1, c2, *, prec=80):
    """
    A parametric family of curves of max-degree 1 on the torus.

    Requirements: c1,c2 real such that c1+c2 != 0.

    Orientation changes at c1==c2.
    """
    from sage.all import ComplexBallField, fast_callable, sqrt, log
    i = SR.I()
    t1 = SR.var('t1')
    T2 = -i * log((-c1 + c2 * exp(i*t1)) / (-c2 + c1 * exp(i*t1)))

    KK = ComplexBallField(prec)
    gam = vector([exp(i*t1), exp(i * T2)])
    gam1 = vector([gk.derivative(t1) for gk in gam])

    @cached_function(key=tuple)
    def moment_lebesgue(aa):
        a1, a2 = aa
        h = exp(i*t1*a1 + i*T2*a2)
        fc = fast_callable(h * gam1.norm(), vars=[t1], domain=KK)
        res = KK.integral(lambda t,_: fc(t), KK(-pi), KK(pi))
        return res

    return moment_lebesgue


def modulated(functional, shift):
    r"""
    Apply a modulation to a moment functional.

    EXAMPLES::

        sage: from momentproblems.moment_functionals import *
        sage: mom = mk_moment_trigonometric_curve_example(5/8, 0)
        sage: K = mom((0,0)).parent()
        sage: mom2 = modulated(mom, vector(K, (exp(2*pi*I*0.1), exp(2*pi*I*0.3))))
        sage: (mom((0,0)) - mom2((0,0))).diameter() < 1e-13
        True
        sage: (mom((3,2)) * K(exp(2*pi*I*(0.1*3 + 0.3*2))) - mom2((3,2))).diameter() < 1e-12
        True
    """
    def modulated_functional(aa):
        bb = tuple(a*z for a, z in zip(aa, shift))
        m = functional(aa)
        K = m.parent()
        return m * m.base_ring().prod(K.coerce(z)**a for z, a in zip(shift, aa))
    return modulated_functional


def generate_points_on_trigonometric_variety(I, gridsize, parent=None):
    r"""
    Find points on trigonometric variety by cutting with (deterministic)
    hyperplanes.

    We assume that the variety can be defined purely in terms of coordinates
    `x_k`, i.e. coordinates of the real part, where

    .. MATH::

        z_k = exp(i t_k) = x_k + i y_k = cos(t_k) + i sin(t_k).

    INPUT:

    - ``gridsize`` -- number of grid points `-1` at which the hyperplanes are
      evaluated
    - ``parent`` -- ring of the coordinates of the result (default: ``SR``)

    OUTPUT:

    Iterator of some points on the variety, given with respect to the
    coordinates `t_k`.

    EXAMPLES::

        sage: from momentproblems.moment_functionals import *
        sage: P.<x0,x1,x2> = QQ[]
        sage: g = -1/4 + x1 + x2 + x1*x2 + x0
        sage: points = list(generate_points_on_trigonometric_variety(P.ideal(g), gridsize=4))
        sage: len(points)
        102
        sage: points[:5]
        [(pi, 1/3*pi, 1/3*pi), (pi, 1/3*pi, -1/3*pi), (pi, -1/3*pi, 1/3*pi), (pi, -1/3*pi, -1/3*pi), (-pi, 1/3*pi, 1/3*pi)]

    ::

        sage: from itertools import islice
        sage: list(islice(generate_points_on_trigonometric_variety(P.ideal(g, x2-1/2), gridsize=5, parent=RealBallField(100)), 5))
        [([3.14159265358979323846264338328 +/- 2.25e-30], [1.04719755119659774615421446109 +/- 5.86e-30], [1.04719755119659774615421446109 +/- 5.86e-30]),
         ([3.14159265358979323846264338328 +/- 2.25e-30], [1.04719755119659774615421446109 +/- 5.86e-30], [-1.04719755119659774615421446109 +/- 5.86e-30]),
         ([3.14159265358979323846264338328 +/- 2.25e-30], [-1.04719755119659774615421446109 +/- 5.86e-30], [1.04719755119659774615421446109 +/- 5.86e-30]),
         ([3.14159265358979323846264338328 +/- 2.25e-30], [-1.04719755119659774615421446109 +/- 5.86e-30], [-1.04719755119659774615421446109 +/- 5.86e-30]),
         ([-3.14159265358979323846264338328 +/- 2.25e-30], [1.04719755119659774615421446109 +/- 5.86e-30], [1.04719755119659774615421446109 +/- 5.86e-30])]
    """
    if parent is None:
        parent = SR
    assert I.ring().is_exact()
    PAA = I.ring().change_ring(AA)  # we switch to AA to find all points symbolically
    I = I.change_ring(PAA)
    grid = [-1 + 2*k/gridsize for k in range(gridsize+1)]  # we pick these points because -1≤cos≤1
    dim = I.dimension()
    # We cut the variety with hyperplanes, hoping to find some points on it.
    # We naively assume that variety sufficiently involves the last variables,
    # so that fixing values for the first ≤dim variables can give a
    # zero-dimensional variety.
    coordinates_fixed = [[PAA.ideal(xk-x) for x in grid] for xk in PAA.gens()[:dim]]
    import itertools
    for Js in itertools.product(*coordinates_fixed):
        Z = list((I + sum(Js)).variety())
        from sage.misc.prandom import shuffle
        shuffle(Z)
        for ξ in Z:
            if all(-1 <= ξ[xk] <= 1 for xk in PAA.gens()[dim:]):
                # We switch coordinates from x_k to t_k where to x_k = cos(t_k).
                # If ξk not zero, then cos(ξk) has two preimages,
                # so this usually multiplies the number of generated points by 2^n
                yield from itertools.product(*(
                    [acos(parent(ξk))] if (ξk := ξ[xk]).is_zero() else
                    [acos(parent(ξk)), -acos(parent(ξk))]
                    for xk in PAA.gens()))


def vandermonde(points, basis, trigonometric=False):
    r"""
    Construct the Vandermonde matrix of size ``len(points) × len(basis)``.

    EXAMPLES::

        sage: from momentproblems.moment_functionals import *
        sage: points = [(1/2, 1/2), (1/3, 2/5)]
        sage: basis = monomial_basis_laurent(dim=2, d=1)
        sage: V = vandermonde(points, basis, trigonometric=True); V.T
        [                                      1.0                                       1.0]
        [ 0.8775825618903728 + 0.479425538604203*I 0.9210609940028851 + 0.3894183423086505*I]
        [ 0.8775825618903728 + 0.479425538604203*I 0.9449569463147377 + 0.3271946967961522*I]
        [0.5403023058681398 + 0.8414709848078965*I  0.742947367824044 + 0.6693498402504662*I]
    """
    if not trigonometric:
        raise NotImplementedError("non-trigonometric case not yet implemented")
    else:
        # the points are given in coordinates t_k, so powers are just multiples
        if not (len(points) or len(basis)):
            raise ValueError("cannot determine dimension of empty Vandermonde matrix")
        dim = len(points[0]) if len(points) else basis[0].parent().ngens()
        if not all(len(m.exponents()) == 1 for m in basis):
            raise ValueError("basis should be monomials")
        alphas = matrix.column(RDF, len(basis), [m.exponents()[0] for m in basis])
        ts = matrix(RDF, len(points), points)
        A = ts * alphas
        return matrix(CDF, np.exp((A.numpy() * 1j)))

# def vandermonde2(points, d, trigonometric=False):
#     r"""
#     A more efficient construction of the Vandermonde matrix of size
#     ``len(points) × len(monomial_basis_laurent(d=d))``.

#     The points are given in coordinates ``(t_1,…,t_n)``.
#     """
#     if not trigonometric:
#         raise NotImplementedError("non-trigonometric case not yet implemented")
#     else:
#         import scipy
#         # note that this implementation assumes that the order of
#         # monomial_basis_laurent is compatible
#         grid_exps = [np.exp(CDF(0,1)*np.array(tis)) for tis in zip(*points)]
#         vands = [np.vander(grid_exp, d+1, increasing=True) for grid_exp in grid_exps]
#         vand = vands[0]
#         for j in range(1, len(vands)):
#             vand = scipy.linalg.khatri_rao(vand.T, vands[j].T).T
#         return matrix(CDF, vand)


def random_positive_definite_matrix(K, n):
    r"""
    EXAMPLES::

        sage: from momentproblems.moment_functionals import *
        sage: random_positive_definite_matrix(QQ, 5).is_positive_semidefinite()
        True
        sage: random_positive_definite_matrix(RDF, 5).is_positive_definite()
        True
        sage: random_positive_definite_matrix(CDF, 5).is_positive_definite()
        True
    """
    V = matrix.random(K, n)  # TODO check that matrix is non-singular, otherwise result is positive-semidefinite
    return V.H * V


def random_moment_functional(ideal, *, random_fun=None):
    r"""
    Return a random functional on a coordinate ring of a variety defined by some ideal.

    EXAMPLES::

        sage: from momentproblems.moment_functionals import *
        sage: R.<x,y,z> = QQ[]
        sage: J = R.ideal(y^2 - x^3 + 1, z^4 - z)
        sage: moment_fun = random_moment_functional(J, random_fun=lambda: ZZ.random_element(1, 10))
        sage: H = hankel(moment_fun, dim=R.ngens(), d=4)
        sage: J == R.ideal(list(H.right_kernel_matrix() * vector(monomial_basis(dim=R.ngens(), d=4))))
        True

    Note that the matrices are not positive-semidefinite::

        sage: H.is_positive_semidefinite()
        False

    Laurent case::

        sage: L.<x,y,z> = LaurentPolynomialRing(QQ)
        sage: J = L.ideal([y^2 - x^3 + 1, z^4 - z])
        sage: moment_fun = random_moment_functional(J, random_fun=lambda: ZZ.random_element(1, 10))
        sage: H = toeplitz(moment_fun, dim=L.ngens(), d=3)
        sage: J == L.ideal(list(H.right_kernel_matrix() * vector(monomial_basis_laurent(dim=L.ngens(), d=3))))
        True
        sage: H.is_positive_semidefinite()
        False
    """
    # A random positive-semidefinite matrix would not have the appropriate Hankel structure.
    # Use e.g. `generate_points_on_trigonometric_variety` instead to create a finite measure with many points on a variety.
    R = ideal.ring()
    K = R.base_ring()
    assert K.is_exact()
    if random_fun is None:
        random_fun = K.random_element
    from collections import defaultdict
    D = defaultdict(random_fun)

    from sage.rings.polynomial.laurent_polynomial_ideal import LaurentPolynomialIdeal
    if isinstance(ideal, LaurentPolynomialIdeal):
        # R is a Laurent ring
        # Construct an equivalent polynomial quotient ring in which we have reduction modulo ideals.
        # The last variable is the inverse of the product of all other variables.
        S = PolynomialRing(R.base_ring(), list(R.variable_names()) + [''.join(R.variable_names()) + "_inv"])
        Q = S.quotient([S.prod(S.gens()) - 1], names=S.variable_names())
        I = Q.defining_ideal() + ideal.polynomial_ideal().change_ring(S)
        phi = R.hom(Q.gens()[:R.ngens()], Q)
        canonicalize = lambda p: I.reduce(phi(p).lift())  # from R to canonical representative in S
    else:
        canonicalize = ideal.reduce

    @cached_function(key=tuple)
    def moment_fun(aa):
        q = canonicalize(R.monomial(*aa))
        return K.sum(c * D[e] for e, c in q.iterator_exp_coeff())
    return moment_fun
