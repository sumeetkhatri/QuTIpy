.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _qutipy-doc-quickstart:

Quickstart
==========

.. meta::
    :description lang=en:
        The user guide provides in-depth information on the
        key concepts of QuTIpy with useful background information and explanation.

Welcome to QuTIpy!
------------------

QuTIpy (**Quantum Theory of Information for Python**; pronounced `/cutiɛ paɪ/`) is an open source
Python library that’s used for performing calculations with quantum states, channels and protocols.
While there are many quantum information theory toolboxes that allow the user to perform basic operations
such as the `partial transposition <./general_functions.html#partial-transpose>`_, new tests are
constantly discovered.

The core of QuTIpy package is a suite of mathematical functions that make working with quantum states,
channels and protocols, quick and easy.

Installing QuTIpy
-----------------
To install QuTIpy, we strongly recommend using a scientific Python distribution. If you’re
looking for the full instructions for installing QuTIpy on your operating system, see `Installing QuTIpy <./installation.html>`_ .

If you already have Python, you can install QuTIpy with:

.. code-block:: shell

   $ pip install git+https://github.com/sumeetkhatri/QuTIpy

or

.. code-block:: shell

   $ pip install QuTIpy==0.1.0a0


How to import QuTIpy
--------------------
To access QuTIpy and its functions import it in your Python code like this:

.. code-block:: python

   import qutipy

Once we have imported qutipy, then we can access other sub modules as :code:`qutipy.general_functions`


Bra-Ket Notation
________________

Qutipy currently follows a functional architecture. THis means that any and all tasks willl have a respective function
definition prepared in the library. From building quantum states to applying gates and channels, everything has their
funtions defined.

For an example, suppose we wanted to create a state vector
:math:`{\displaystyle |v\rangle } = \begin{bmatrix} 1 \\ 0 \end{bmatrix}`

We can import the `ket` function from `general_functions` sub-module and create
the vector :math:`{\displaystyle |v\rangle }` like this,

.. code-block:: python

   >>> from qutipy.general_functions import ket
   >>> # Defining a ket 0 in a 2-Dimensional Hilbert space,
   >>> # The first argument takes a dimension of the Hilbert space,
   >>> # while the second argument takes the ket value.
   >>> ket(2,0)

   array([[1.],
          [0.]])



States
______

Pre-defined states can be accessed from :code:`qutipy.states` sub-module. Example:

.. code:: python

   from qutipy.states import (
       Bell_state,
       GHZ_state,
       max_ent,
       MaxMix_state,
       RandomDensityMatrix,
       RandomStateVector,
       Werner_state,
       Werner_twirl_state,
       graph_state,
       isotropic_state,
       isotropic_twirl_state,
       singlet_state
   )

Maximally Entangled State
~~~~~~~~~~~~~~~~~~~~~~~~~

A pure state :math:`\psi_{AB} = |\psi\rangle\langle\psi|_{AB}` , for two systems :math:`A`  and :math:`B`  of the same dimension :math:`d` , is called **Maximally Entangled** if the Schmidt coefficients of :math:`|\psi\rangle_{AB}` are all equal to :math:`\frac{1}{\sqrt{d}}` , with :math:`d` being the Schmidt rank of :math:`|\psi\rangle_{AB}` .

In other words, :math:`\psi_{AB}` \ is called maximally entangled if :math:`|\psi\rangle_{AB}` has the Schmidt decomposition,\

.. math::
   |\psi\rangle_{AB} = \frac{1}{\sqrt{d}}\sum_{k=1}^{d} |e_k\rangle_A \otimes |f_k\rangle_B

for some orthonormal sets :math:`\{ |e_k\rangle_A : 1 \le k \le d \}` and :math:`\{ |f_k\rangle_B : 1 \le k \le d \}` .

In simple terms, the **Maximally Entangled** can be written as
:math:`(\frac{1}{\sqrt{d}})*(|0\rangle|0\rangle+|1\rangle|1\rangle+...+|d-1\rangle|d-1\rangle)` \ and can be created using the ``max_ent`` function.

.. code:: python

   >>> from qutipy.states import max_ent
   >>> # This will create a Maximally Entangled State for a 3 dimensional system.
   >>> # The resultant matrix will be of shape 9x9.
   >>> max_ent(3)


Bell State
~~~~~~~~~~

A `Bell state <https://en.wikipedia.org/wiki/Bell_state>`__ is defined as a `maximally entangled quantum state <states.md#maximally-entangled-state>`__ of two qubits.
It can be described as one of four entangled two qubit quantum states, known collectively as the four “ `Bell states <https://en.wikipedia.org/wiki/Bell_state>`__ ”.

.. math::
   |\phi^{+}\rangle \equiv |\phi_{0, 0}\rangle = \frac{1}{\sqrt{2}} (|0, 0\rangle + |1, 1\rangle)

.. math::
   |\phi^{-}\rangle \equiv |\phi_{1, 0}\rangle = \frac{1}{\sqrt{2}} (|0, 0\rangle - |1, 1\rangle)

.. math::
   |\psi^{+}\rangle \equiv |\phi_{0, 1}\rangle = \frac{1}{\sqrt{2}} (|0, 1\rangle + |1, 0\rangle)

.. math::
   |\psi^{-}\rangle \equiv |\phi_{1, 1}\rangle = \frac{1}{\sqrt{2}} (|0, 1\rangle - |1, 0\rangle)

A generalized version of the above `Bell States <https://en.wikipedia.org/wiki/Bell_state>`__ is explained below,

Using the operators :math:`X`  , :math:`Z`  , and :math:`ZX`  , we define the following set of four entangled two-qubit state vectors :math:`|\phi_{z,x}\rangle = (Z^zX^x \otimes I)|\phi^{+}\rangle`  for :math:`z, x \in {0, 1}`  .
To generates a :math:`d`  -dimensional Bell State with :math:`0 <= z`  , :math:`x <= d-1`  , we can simply call the module ``Bell_state`` that was imported above.

.. code-block:: python

   >>> from qutipy.states import Bell
   >>> # This will create a Bell State for a 2 dimensional system.
   >>> # The resultant matrix will be of shape 4x4.
   >>> Bell(d=2, z=1, x=1)

   array([[ 0.        +0.00000000e+00j],
          [ 0.70710678+0.00000000e+00j],
          [-0.70710678+8.65956056e-17j],
          [ 0.        +0.00000000e+00j]])


This is a generalized version of the above `Bell state`_ , that we defined
using the operators :math:`X` , :math:`Z` , and :math:`ZX` , where the two-qubit
`maximally entangled quantum state`_ vectors is represented as

:math:`\displaystyle |\phi_{z,x}\rangle = (Z^zX^x \otimes I)|\phi^{+}\rangle` for :math:`z, x \in {0, 1}` .

Above, we generated a :math:`d` -dimensional Bell State with :math:`0 <= z` , :math:`x <= d-1` .

Singlet State
~~~~~~~~~~~~~

A singlet state is defined as :math:`\frac{1}{(d^2-d)} \times (I_{(d^2)}-F)` \ where :math:`F` is a Swap Operator.
Generating a singlet state is as easy as writing a single word,

.. code:: python

   >>> from qutipy.states import singlet_state
   >>> # This will create a Singlet State for a 3 dimensional system.
   >>> # The resultant matrix will be of shape 9x9.
   >>> singlet_state(3)

.. _Bell state: https://en.wikipedia.org/wiki/Bell_state
.. _maximally entangled quantum state: https://github.com/arnavdas88/QuTIpy-Tutorials/blob/main/modules/states.md#maximally-entangled-state

Pauli
______

The `Pauli matrices, <https://en.wikipedia.org/wiki/Pauli_matrices>`__
also called the Pauli spin matrices, are complex matrices that arise in
Pauli’s treatment of spin in quantum mechanics. They are defined by:

.. math::
   \sigma_1 = \sigma_x = \begin{bmatrix} 0 && 1 \\ 1 && 0 \end{bmatrix}

.. math::
   \sigma_2 = \sigma_y = \begin{bmatrix} 0 && -i \\ i && 0 \end{bmatrix}

.. math::
   \sigma_3 = \sigma_z = \begin{bmatrix} 1 && 0 \\ 0 && -1 \end{bmatrix}


In `quantum mechanics <https://en.wikipedia.org/wiki/Quantum_mechanics>`__, pauli matrices occur in the `Pauli equation <https://en.wikipedia.org/wiki/Pauli_equation>`__ which takes into account the interaction of the `spin <https://en.wikipedia.org/wiki/Spin_(physics)>`__ of a particle with an external `electromagnetic field <https://en.wikipedia.org/wiki/Electromagnetic_field>`__.

.. hint::
   *Pauli matrices* also represent the interaction states of two polarization filters
   for horizontal / vertical polarization, 45º polarization, and circular polarization.

Each Pauli matrix is `Hermitian <https://en.wikipedia.org/wiki/Hermitian_matrix>`__, and
together with the identity matrix :math:`I` , the Pauli matrices form a
`basis <https://en.wikipedia.org/wiki/Basis_(linear_algebra)>`__ for the
real `vector space <https://en.wikipedia.org/wiki/Vector_space>`__ of 2 :math:`\times` 2 Hermitian matrices.
This means that any 2 :math:`\times` 2 `Hermitian matrix <https://en.wikipedia.org/wiki/Hermitian_matrix>`__ can be
written in a unique way as a linear combination of Pauli matrices, with all coefficients being real numbers.

.. code:: python

   >>> from qutipy.pauli import generate_nQubit_Pauli_X
   >>> # This will create a pauli_x operator
   >>> px = generate_nQubit_Pauli_X([1])
   >>> px
   array([[0, 1],
          [1, 0]])
   >>> # This will create a tensor product of pauli_x operators
   >>> px = generate_nQubit_Pauli_X([0, 1])
   >>> px
   array([[0., 1., 0., 0.],
          [1., 0., 0., 0.],
          [0., 0., 0., 1.],
          [0., 0., 1., 0.]])


.. code:: python

   >>> I = generate_nQubit_Pauli([0])
   >>> I
   array([[1., 0.],
          [0., 1.]])

.. code:: python

   >>> X = generate_nQubit_Pauli([1])
   >>> X
   array([[0., 1.],
          [1., 0.]])

.. code:: python

   >>> Y = generate_nQubit_Pauli([2])
   >>> Y
   array([[0.j, -1.j],
          [1.j,  0.j]])

.. code:: python

   >>> Z = generate_nQubit_Pauli([3])
   >>> Z
   array([[ 1,  0],
          [ 0, -1]])

.. code:: python

   >>> ZZ = generate_nQubit_Pauli([3, 3])
   >>> ZZ
   array([[ 1,  0,  0,  0],
          [ 0, -1,  0,  0],
          [ 0,  0, -1,  0],
          [ 0,  0,  0,  1]])


.. code:: python

   >>> ZI = generate_nQubit_Pauli([3, 1])
   >>> ZI
   array([[ 0,  1,  0,  0],
          [ 1,  0,  0,  0],
          [ 0,  0,  0, -1],
          [ 0,  0, -1,  0]])