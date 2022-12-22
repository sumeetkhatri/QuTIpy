.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _qutipy-doc-general-functions:

General Functions
=================

Hilbert Space
-------------

The primary mathematical object in quantum theory is the `Hilbert
space <https://en.wikipedia.org/wiki/Hilbert_space>`__. We consider only
finite-dimensional **Hilbert spaces**, denoted by :math:`\mathcal{H}`.
Although we will be considering *finite-dimensional spaces*
exclusively, we note here that many of the statements and claims extend
directly to the case of separable, *infinite-dimensional Hilbert
spaces*, especially for operationally-defined tasks and information
quantities.

A :math:`d`-dimensional Hilbert space :math:`(1 \le d < \infty)` is defined to
be a complex vector space equipped with an inner product. We use the notation
:math:`{\displaystyle |\psi\rangle}` to denote a vector in :math:`\mathcal{H}`.
More generally, a **Hilbert space** is a “**complete inner product**” space.

.. hint::
   **Completeness** is an issue that pops up only
   in *infinite-dimensional spaces*, so all *finite-dimensional
   inner-product spaces* are **Hilbert spaces**.

Bra-Ket Notation
----------------
A **ket** is of the form :math:`{\displaystyle |v\rangle }`. Mathematically it denotes a
`vector <https://en.wikipedia.org/wiki/Vector_space>`__, :math:`{\displaystyle {\boldsymbol {v}}}`,
in an abstract (complex) `vector space <https://en.wikipedia.org/wiki/Vector_space>`__ :math:`{\displaystyle V}`,
and physically it represents a state of some quantum system. An example of a **Ket** can
be :math:`{\displaystyle |r\rangle } = \begin{bmatrix} x \\ y\\ z\end{bmatrix}` represents
a vector :math:`{\displaystyle \vec{r} } =\begin{bmatrix} x \\ y\\ z\end{bmatrix}`.

A **bra** is of the form :math:`{\displaystyle \langle f|}` . Mathematically it denotes a
`linear form <https://en.wikipedia.org/wiki/Linear_form>`__ :math:`{\displaystyle f:V\to \mathbb {C} }`,
i.e. a `linear map <https://en.wikipedia.org/wiki/Linear_map>`__ that maps each vector
in :math:`{\displaystyle V}` to a number in the complex plane :math:`{\displaystyle \mathbb {C} }`.
Letting the linear functional :math:`{\displaystyle \langle f|}` act on a vector :math:`{\displaystyle |v\rangle }`
is written as :math:`{\displaystyle \langle f|v\rangle \in \mathbb {C} }`.
The **bra** is similar to the **ket**, but the values are in a **row**, and each element is the complex
`conjugate <https://en.wikipedia.org/wiki/Complex_conjugate>`__ of the **ket**\ ’s elements.

In the simple case where we consider the vector space :math:`{\displaystyle \mathbb {C} ^{n}}`,
a **ket** can be identified with a `column vector <https://en.wikipedia.org/wiki/Column_vector>`__,
and a **bra** as a `row vector <https://en.wikipedia.org/wiki/Row_vector>`__.

Meaning :
~~~~~~~~~


:math:`{\displaystyle \langle A| }=\begin{bmatrix}A_1&A_2&A_3&\dots\end{bmatrix} \qquad \& \qquad  {\displaystyle |B\rangle}=\begin{bmatrix}B_1\\B_2\\B_3\\\vdots\end{bmatrix}`

Example:
^^^^^^^^

:math:`{\displaystyle |0\rangle }=\begin{bmatrix}1\\0\end{bmatrix}` , for two dimensional Hilbert Space,
Defining a basis state :math:`{\displaystyle |0\rangle }`, we can use the ``ket`` module like this:

.. code:: python

   from qutipy.general_functions import ket

   # Defining a ket 0 in a 2Dimensional Hilbert space,
   # The first argument takes a dimension of the Hilbert space,
   # while the secind argument takes the ket value.
   v = ket(2,0)

Here we have defined the **ket** v for :math:`{\displaystyle |v\rangle } = \begin{bmatrix} 1 \\ 0 \end{bmatrix}`.
In numpy, defining the same would need one to define the matrix manually, just as shown in the
`Overview section <../getting-started/overview.md#bra-ket-notation>`__.


Partial Trace
-----------------




Partial Transpose
-----------------

The Partial Transpose plays an important role in quantum information
theory due to its connection with entanglement. In fact, it leads to a
sufficient condition for a bipartite state to be entangled.

Given quantum systems :math:`A` and :math:`B`, the partial transpose
on :math:`B`is denoted by :math:`T_B\equiv id_A \otimes T_B`, and it is defined as,

.. math::

   T_B(X_{AB})  :=  \sum\limits^{d_B-1}_{j, j'=0}   (\mathbf{1}_A  \otimes
   |i\rangle \langle{i'}|_B)  X_{AB}  (\mathbf{1}_A  \otimes  |i\rangle \langle{i'}|_B)

``partial_transpose(...)`` is a function that computes the partial
transpose of a matrix. The transposition may be taken on any subset of
the subsystems on which the matrix acts.

Defining a state ``X`` with [ … ]

.. code:: python

   import numpy as np

   X = np.array(
       [
           [ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]
       ]
   )

Now we can apply the ``partial_transpose`` function over our state ``X``:

.. code:: python

   from qutipy.general_functions import partial_transpose

   pt = partial_transpose(X, [1], X.shape)
