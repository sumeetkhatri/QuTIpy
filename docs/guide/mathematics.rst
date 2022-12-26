.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _qutipy-doc-mathematics:


Mathematics
===========

Ullamcorper eget nulla facilisi etiam dignissim diam quis enim. Aliquam sem
fringilla ut morbi tincidunt augue interdum. Gravida dictum fusce ut placerat
orci nulla. Ut tristique et egestas quis ipsum suspendisse. Arcu cursus euismod
quis viverra nibh cras pulvinar mattis. Imperdiet massa tincidunt nunc pulvinar
sapien et ligula. Aliquam malesuada bibendum arcu vitae. Pellentesque sit amet
porttitor eget dolor morbi non arcu risus. At consectetur lorem donec massa sapien
faucibus.


.. _qutipy-doc-hilbert-space:

Hilbert Space
-------------

Hilbert Space is an inner product space that is complete with respect to the norm defined
by the inner product. Mathematically, a Hilbert Space is a vector space :math:`H` with an inner
product :math:`\langle f, g \rangle` such that the norm defined by :math:`|f| = \sqrt{\langle f, f \rangle}`,
turns :math:`H` into a complete metric space. Completeness in this context means that every
Cauchy sequence of elements of the space converges to an element in the space, in the sense
that the norm of differences approaches zero. As a metric space, Hilbert space can be considered
an infinite-dimensional linear topological space. A Hilbert Space is always a Banach space, but
the converse need not hold.

The definition of an inner product space is relatively straight-forward. Given a vector space :math:`V`
over a field :math:`F`, an inner product on :math:`V` is a mapping :math:`\langle\cdot , \cdot\rangle : V \times V \to F`
that is bilinear, symmetric, and positive-definite.
An inner product space is a real or complex vector space with an operation called an inner
product. The inner product of two vectors in the space is a scalar, often denoted with angle brackets
such as in :math:`\langle a, b\rangle`.

.. note::

   An important result in linear algebra is that every finite-dimensional inner product space is complete.
   That is, given any set of vectors in the space, their Euclidean norms will approach zero as the number
   of vectors in the set goes to infinity. Consequently, the only infinite-dimensional normed vector space
   that is complete is an infinite-dimensional Hilbert space.

.. _qutipy-doc-linear-operator:

Linear Operator
---------------

.. note::
   An **operator** is a generalization of the concept of a function applied to a function.
   Whereas a function is a rule for turning one number into another, an operator is a rule
   for turning one function into another.

A mathematical operator with the property that applying it to a linear combination of two
objects yields the same linear combination as the result of applying it to the objects separately.
It is a function that maps one vector onto other vectors. They can be represented by matrices,
which can be thought of as coordinate representations of Linear Operators.

A function :math:`f` is called a linear operator if it has the two properties:

#. :math:`\hspace{1em}` :math:`f(x+y) = f(x) + f(y), \hspace{3em} \forall` :math:`x` & :math:`y`

#. :math:`\hspace{1em}` :math:`f(c \cdot x) = c \cdot f(x), \hspace{4.6em} \forall` :math:`x`, & :math:`c` is a constant.

OR

Which also means,  :math:`f ( a \cdot x + b \cdot y) = a \cdot f(x) + b \cdot f(y), \hspace{3em} \forall` :math:`x` & :math:`y` and :math:`\forall` constants :math:`a` & :math:`b`.

Given a Hilbert space :math:`H_A` with dimension :math:`d_A` and a Hilbert space :math:`H_B`
with dimension :math:`d_B`, a linear operator :math:`X : H_A \rightarrow H_B` is defined to
be a function such that :math:`X( {\alpha |\psi \rangle}_A + {\beta | \phi \rangle}_A ) = {\alpha X |\psi \rangle}_A + {\beta X | \psi \rangle}_A`
for all :math:`\alpha , \beta \in \mathbf{C}` and :math:`{|\psi \rangle}_A, {| \phi \rangle}_A \in H_A`.
For clarity, we sometimes write :math:`X_{A \rightarrow B}`` to explicitly indicate the input and output
Hilbert spaces of the linear operator :math:`X`. We use :math:`\mathbb{1}` to denote the identity operator,
which is defined as the unique linear operator such that :math:`\mathbb{1}|\psi \rangle = |\psi \rangle`
for all vectors :math:`|\psi \rangle`. For clarity, when needed, we write :math:`\mathbb{1}_d` to indicate the identity
operator acting on a :math:`d`-dimensional Hilbert space.

We denote the set of all linear operators from :math:`H_A` to :math:`H_B` by :math:`L(H_A, H_B)`. If
:math:`H_A = H_B`, then :math:`L(H_A) := L(H_A, H_A)`, and we sometimes indicate the input Hilbert
space :math:`H_A` of :math:`X \in L(H_A)` by writing :math:`X_A`. In particular, we often write
:math:`X_{AB}` when referring to linear operators in :math:`L(H_A \otimes H_B)`, i.e., when referring
to linear operators acting on a tensor-product Hilbert space.



Image
*****

The *image* of a linear operator :math:`X \in L(H_A, H_B)`, denoted by :math:`im(X)`, is the set
defined as,

.. math::
   im(X) := \{{|\psi\rangle}_B \in H_B : {|\psi\rangle}_B = X{|\psi\rangle}_A, {|\psi\rangle}_A \in H_A \}.

It is also known as the column space or range of :math:`X`. The image of :math:`X` is a subspace of
:math:`H_B`. The rank of :math:`X`, denoted by :math:`rank(X)`, is defined to be the dimension of :math:`im(X)`.

.. note::
   Note that :math:`rank(X) \leq min\{d_A, d_B \}` for all :math:`X \in L(H_A, H_B)`.

Kernel
******

The kernel of a linear operator :math:`X \in L(H_A, H_B)`, denoted by :math:`ker(X)`, is defined
to be the set of vectors in the input space :math:`H_A` of :math:`X` for which the output is the
zero vector; i.e.,

.. math::
   ker(X) := \{ {|\psi\rangle}_A \in H_A : X{|\psi\rangle}_A = 0\}.

It is also known as the null space of :math:`X`. The following dimension formula holds:

.. math::
   d_A = rank(X) + dim(ker(X)),

and it is known as the rank-nullity theorem (:math:`dim (ker (X))` is called the nullity of :math:`X`).

Support
*******

The support of a linear operator :math:`X \in L(H_A, H_B)`, denoted by :math:`supp(X)`, is defined
to be the orthogonal complement of the kernel:

.. math::
   supp(X) := ker(X)^\bot := \{ |\psi \rangle \in H_A : \langle\psi | \phi\rangle = 0 \hspace{1em} \forall | \psi\rangle \in ker(X)\}

It is also known as the row space or coimage of :math:`X`.

|

.. note::
   The **rank** of a linear operator can also be equivalently defined as the number of its singular values.

|


A linear operator :math:`X \in L(H_A, H_B)` is called **injective** (or one-to-one) if, for all
:math:`|\psi\rangle, |\phi\rangle \in H_A, X|\psi\rangle = X|\phi\rangle` implies
:math:`| \psi\rangle = | \phi\rangle`. A necessary and sufficient condition for :math:`X` to be
injective that the kernel of :math:`X` contains only the zero vector (i.e., the column vector in
which all of the elements are equal to zero), which implies that :math:`dim(ker(X)) = 0`.

A linear operator :math:`X \in L(H_A, H_B)` is called **surjective** (or onto) if, for all
:math:`|\phi\rangle \in H_B`, there exists :math:`|\psi\rangle \in H_A` such that :math:`X|\psi\rangle = |\phi\rangle`.
A necessary and sufficient condition for :math:`X` to be surjective is that :math:`rank(X) = d_B`.

Tensor product
--------------

The tensor product :math:`V \otimes W` of two vector spaces :math:`V` and :math:`W` (over the same field) is itself a vector space,
endowed with the operation of bilinear composition, denoted by :math:`\otimes`, from ordered pairs in the Cartesian product :math:`V \times W`
onto :math:`V \otimes W` in a way that generalizes the outer product. The tensor product of :math:`V` and :math:`W` is the vector space
generated by the symbols :math:`v \otimes w`, with :math:`v \in V` and :math:`w \in W`, in which the relations of bilinearity are imposed
for the product operation :math:`\otimes`, and no other relations are assumed to hold. The tensor product space is thus the "freest"
(or most general) such vector space, in the sense of having the fewest constraints.

The tensor product of (finite dimensional) vector spaces has dimension equal to the product of the dimensions of the two factors:


.. math::
   dim ⁡ ( V \otimes W ) = dim ⁡ V \times dim ⁡ W.




Given two linear operators :math:`X \in L(H_A, H_B)` and :math:`Y \in L(H_A, H_B)`, their tensor
product :math:`X \otimes Y` is a linear operator in :math:`L(H_A \otimes H_{A′} , H_B \otimes H_{B′} )` such that

:math:`(X \otimes Y)({|\psi \rangle}_A \otimes {|\psi \rangle}_{A′} ) = X{|\psi \rangle}_A \otimes Y{|\psi \rangle}_{A′}`

for all :math:`{| \psi \rangle }_A \in {H}_A` and :math:`{| \psi \rangle}_{A′} \in H_{A′}`. The matrix representation
of :math:`X \otimes Y` is the Kronecker product of the matrix representations of :math:`X` and :math:`Y`,
which is a matrix generalization of the “stack-and-multiply” procedure:

.. math::
   {|\psi\rangle}_A \otimes {|\psi\rangle}_B =
   \begin{pmatrix}
      {\alpha}_0 \\ {\alpha}_1
   \end{pmatrix}
   \otimes
   \begin{pmatrix}
      {\beta}_0 \\ {\beta}_1 \\ {\beta}_2
   \end{pmatrix} =
   \begin{pmatrix}
      {\alpha}_0
      \cdot
      \begin{pmatrix}
         {\beta}_0 \\ {\beta}_1 \\ {\beta}_2
      \end{pmatrix}
      \\
      {\alpha}_1
      \cdot
      \begin{pmatrix}
         {\beta}_0 \\ {\beta}_1 \\ {\beta}_2
      \end{pmatrix}
   \end{pmatrix} =
   \begin{pmatrix}
      {\alpha}_0 \cdot {\beta}_0 \\
      {\alpha}_0 \cdot {\beta}_1 \\
      {\alpha}_0 \cdot {\beta}_2 \\
      {\alpha}_1 \cdot {\beta}_0 \\
      {\alpha}_1 \cdot {\beta}_1 \\
      {\alpha}_1 \cdot {\beta}_2
   \end{pmatrix}

Basis Expansion
---------------

A set :math:`B` of vectors in a vector space :math:`V` is called a basis if every element of :math:`V` may be written in a unique way as
a finite linear combination of elements of :math:`B`. The coefficients of this linear combination are referred to as components or coordinates
of the vector with respect to :math:`B`. The elements of a basis are called basis vectors.

Every linearly independent list of vectors in a finite-dimensional vector space :math:`V` can be extended to a basis of :math:`V`.

Singular Value Decomposition
----------------------------

Schmidt Decomposition
---------------------

Let :math:`{|\psi\rangle}_{AB}` be a vector in the tensor-product Hilbert space :math:`H_{AB}`. Let :math:`X_{A \to B}` be the
linear operator with matrix elements :math:`\langle j|_B X |i\rangle_A = \langle i, j| \psi \rangle_{AB}`, and let :math:`r = rank(X)`.

Then, there exist strictly positive *Schmidt coefficients* :math:`\{ \lambda_k \}^r_{k=1}`, and orthonormal
vectors :math:`\{ | e_k \rangle_A \}^r_{k=1}` and :math:`\{ | f_k \rangle_B \}^r_{k=1}`, such that

.. math::
   {|\psi\rangle}_{AB} =  \sum\limits_{k=1}^{r} \sqrt{\lambda_k}  \{ | e_k \rangle_A \}  \otimes \{ | f_k \rangle_B \} .

The quantity :math:`r` is called the **Schmidt rank**, and it holds that :math:`r \leq min\{d_A, d_B\}`.