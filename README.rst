##################
``momentproblems``
##################

*******************************************
Research code accompanying the thesis (tba)
*******************************************

This is a `SageMath <SAGE_>`_ package with a proof-of-concept implementation of
algorithms for:

* parameter recovery of certain moment problems
* generalized eigenvalues and eigenspaces of regular matrix pencils `(A_0,…,A_r)`

Installation
============

**Requirements**: A version of `Sage <SAGE_>`_ in which
`LAPACK issue 475 <https://github.com/Reference-LAPACK/lapack/issues/475>`_
is fixed, e.g. with OpenBLAS 0.3.14+ (included in the Sage distribution since
Sage 9.5.beta6+, or can be installed manually).

First, clone the `repository from GitHub <momentproblems_gh_>`_ and then
install the package and run the tests::

    git clone https://github.com/mwageringel/momentproblems.git && cd momentproblems
    make SAGE=sage install test

Alternatively, to install into the Python user install directory (no root
access required), run::

    make SAGE=sage install-user test

.. _SAGE: https://www.sagemath.org/
.. _momentproblems_gh: https://github.com/mwageringel/momentproblems
