.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _qutipy-doc-beginners-guide:

.. meta::
    :description lang=en:
        New to QuTIpy? Check out the Absolute Beginner’s Guide. It contains an
        introduction to QuTIpy’s main concepts and links to additional tutorials.

Beginners Guide
===============

Welcome to the absolute beginner’s guide to QuTIpy! If you have comments or suggestions, please don’t hesitate to reach out!

Welcome to QuTIpy!
------------------


QuTIpy (**Quantum Theory of Information for Python**; pronounced `/cutiɛ paɪ/`) is an open source
Python library that’s used for performing calculations with quantum states, channels and quantum information processing protocols.
While there are many quantum information theory toolboxes that allow the user to perform basic operations
such as the `partial transposition <./general_functions.html#partial-transpose>`_ and `partial trace <./general_functions.html#partial-trace>`__, the focus of QuTIpy is to
allow you perform these operations in a `simple` and `quick` way.


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


Reading the example code
------------------------
If you aren’t already comfortable with reading tutorials that contain a
lot of code, you might not know how to interpret a code block that looks
like this:

.. code-block:: python

   >>> from qutipy.general_functions import ket
   >>> ket(2,0)
   array([[1.],
          [0.]])

If you aren’t familiar with this style, it’s very easy to understand.
If you see :code:`>>>`, you’re looking at input, or the code that you
would enter. Everything that doesn’t have :code:`>>>` in front of it is output,
or the results of running your code. This is the style you see when you
run python on the command line, but if you’re using IPython, you might
see a different style. Note that it is not part of the code and will cause
an error if typed or pasted into the Python shell. It can be safely typed or
pasted into the IPython shell; the :code:`>>>` is ignored.

Quantum Mechanics
-----------------
Quantum mechanics is the backbone for quantum information processing, and many
aspects of it cannot be explained by classical reasoning.

.. note::
   For example, there is no strong classical analogue for pure quantum states or
   entanglement, and this leads to stark differences between what is possible in
   the classical and quantum worlds.

However, at the same time, it is important to emphasize that all of classical
information theory is subsumed by quantum information theory, so that whatever
is possible with classical information processing is also possible with quantum
information processing. As such, quantum information subsumes classical information
while allowing for richer possibilities.

A **quantum system** :math:`A` is associated with a :hoverxreftooltip:`Hilbert space <qutipy-doc-hilbert-space>` :math:`\mathcal{H}_A`.
The state of the system :math:`A` is described by a density operator, which is a unit-
trace, positive semi-definite linear operator acting on :math:`\mathcal{H}_A`.

What are Quantum States?
------------------------
A `quantum state`_ is a mathematical entity that provides a probability
distribution for the outcomes of each possible measurement on a
`system`_. The `state`_ of a `quantum system`_ is described by a
`density operator`_ acting on the underlying `Hilbert space`_ of the
`quantum system`_. Knowledge of the `quantum state`_ together with the
rules for the system’s `evolution in time`_ exhausts all that can be
predicted about the system’s behavior.

.. _quantum state: https://en.wikipedia.org/wiki/Quantum_state
.. _system: states.md#quantum-systems
.. _state: https://en.wikipedia.org/wiki/Quantum_state
.. _quantum system: states.md#quantum-systems
.. _density operator: https://en.wikipedia.org/wiki/Density_matrix#Definition_and_motivation
.. _Hilbert space: general-functions.md#firstheading
.. _evolution in time: https://en.wikipedia.org/wiki/Quantum_channel#Time_evolution

The qubit is perhaps the most fundamental quantum system
and is the quantum analogue of the (classical) bit. Every physical system
with two distinct degrees of freedom obeying the laws of quantum mechanics
can be considered a qubit system. The Hilbert space associated with a qubit
system is :math:`\mathcal{C}^2`, whose standard orthonormal basis is denoted by :math:`{|0〉, |1〉}`. Three
common ways of physically realizing qubit systems are as follows:

#. The two spin states of a spin-:math:`\frac{1}{2}` particle.
#. Two distinct energy levels of an atom, such as the ground state and one of the excited states.
#. Clockwise and counter-clockwise directions of current flow in a superconducting electronic circuit.

How to create a basic Quantum States?
-------------------------------------

*This section covers* ``1D array``, ``2D array``, ``ndarray``, ``vector``, ``matrix``

------

QuTIpy contains the definitions of these states, inside the states sub-module,
and can be imported as such.

.. code-block:: python

   >>> from qutipy.states import Bell

Bell States
***********
A `Bell state`_ is defined as a `maximally entangled quantum state`_ of two qubits.
It can be described as one of four entangled two qubit quantum states,
known collectively as the four "`Bell state`_".

.. _Bell state: https://en.wikipedia.org/wiki/Bell_state
.. _maximally entangled quantum state: https://github.com/arnavdas88/QuTIpy-Tutorials/blob/main/modules/states.md#maximally-entangled-state



:math:`\displaystyle |\phi^{+}\rangle \equiv |\phi_{0, 0}\rangle = \frac{1}{\sqrt{2}} (|0, 0\rangle + |1, 1\rangle)`

:math:`\displaystyle |\phi^{-}\rangle \equiv |\phi_{1, 0}\rangle = \frac{1}{\sqrt{2}} (|0, 0\rangle - |1, 1\rangle)`

:math:`\displaystyle |\psi^{+}\rangle \equiv |\phi_{0, 1}\rangle = \frac{1}{\sqrt{2}} (|0, 1\rangle + |1, 0\rangle)`

:math:`\displaystyle |\psi^{-}\rangle \equiv |\phi_{1, 1}\rangle = \frac{1}{\sqrt{2}} (|0, 1\rangle - |1, 0\rangle)`


A generalized version of the above `Bell state`_ is explained below,

Using the operators :math:`X`, :math:`Z`, and :math:`ZX`, we define the following set of four entangled two-qubit state vectors :math:`\displaystyle |\phi_{z,x}\rangle = (Z^zX^x \otimes \mathbb{1})|\phi^{+}\rangle` for :math:`z, x \in {0, 1}`.

To generates a :math:`d`-dimensional Bell State with :math:`0 \leq z`, :math:`x \leq d-1`, we can simply call the module `Bell_state` that was imported above.

.. code-block:: python

   >>> from qutipy.states import Bell
   >>>
   >>> # This will create a Bell State for a 2 dimensional system.
   >>> # The resultant matrix will be of shape 4x4.
   >>> bell_state = Bell(d=2, z=1, x=1)
   >>>
   >>> bell_state.shape
   (2, 2)


Random Quantum States
*********************

**Density matrices** define classical statistical mixtures of pure quantum states. Whereas,
**State vectors** define pure quantum states of a system, and, for an isolated system.

Random Quantum States, for either case (Density Matrix or State Vectors), can be easily generated
through the ``qutipy.states.random_density_matrix`` definition and ``qutipy.states.random_state_vector`` definition.

For Density Matrix,

.. code-block:: python

   >>> # Import the random_density_matrix definition
   >>> from qutipy.states import random_density_matrix
   >>>
   >>> # Let's create a random density matrix of shape 3 x 3
   >>> A = random_density_matrix(dim = 3)
   >>>
   >>> # The shape of A will be `dim x dim`, i.e. 3 x 3
   >>> A.shape
   (3, 3)


State Vectors can be generated directly as well using the definition ``random_state_vector``.

.. code-block:: python

   >>> # Import the random_density_matrix definition
   >>> from qutipy.states import random_state_vector
   >>>
   >>> # Let's create a pure random state vector of shape 3 x 1
   >>> A = random_state_vector(dim = 3)
   >>>
   >>> # The shape of A will be 3 x 1
   >>> A.shape
   (3, 1)
   >>>
   >>> # One can also define the Schmidt rank like this:
   >>> A = random_state_vector(dim = [2, 4], rank = 2)
   >>> # In this case, the random_state_vector generates the state_vector for 2 systems,
   >>> # one with dimension 2 and one with dimension 4.
   >>>
   >>> # The shape of A will be 8 x 1
   >>> A.shape
   (8, 1)



Unitary Operators
-----------------

These are linear operators :math:`U \in L(H)` whose inverses are
equal to their adjoints, meaning that :math:`U^{\dagger} U = UU^{\dagger} = \mathbb{1}`. Unitary operators
generalize invertible maps or permutations from classical information theory
and describe the noiseless evolution of the state of a quantum system.

Random Unitaries
****************

QuTIpy ships a definition ``qutipy.gates.RandomUnitary`` that generates a Random Unitary of a given specificaiton

.. code-block:: python

   >>> from qutipy.gates import RandomUnitary
   >>>
   >>> # Creates a random unitary of shape 2 x 2
   >>> random_unitary = RandomUnitary(2)
   >>>
   >>> random_unitary.shape
   (2, 2)

Pauli Operators
***************

The `Pauli matrices`_, also called the Pauli spin matrices,
are complex matrices that arise in Pauli's treatment of spin in quantum mechanics. They are defined by:

.. math::

   \sigma_1 = \sigma_x = \begin{bmatrix} 0 && 1 \\ 1 && 0 \end{bmatrix}

   \sigma_2 = \sigma_y = \begin{bmatrix} 0 && -i \\ i && 0 \end{bmatrix}

   \sigma_3 = \sigma_z = \begin{bmatrix} 1 && 0 \\ 0 && -1 \end{bmatrix}

In `quantum mechanics`_, pauli matrices occur in the `Pauli equation`_ which takes into account the
interaction of the `spin`_ of a particle with an external `electromagnetic field`_.

.. note::

   `Pauli matrices`_ also represent the interaction states of two polarization filters for horizontal
   / vertical polarization, 45º polarization, and circular polarization.


Each Pauli matrix is `Hermitian`_, and together with the identity matrix :math:`I`, the Pauli matrices
form a `basis`_ for the real `vector space`_ of :math:`2 \times 2`  Hermitian matrices. This means that
any :math:`2 \times 2`  `Hermitian matrix`_ can be written in a unique way as a linear combination of
Pauli matrices, with all coefficients being real numbers.

.. note::

   The identity matrix :math:`I` is sometimes considered as the zero :math:`^{th}` Pauli matrix or
   :math:`\sigma_0`.


The Pauli matrices are important in the context of quantum mechanics, and `quantum information`_
more generally, as they can be used to describe the quantum states, as well as the evolution of the
quantum states, of 2 Dimensional quantum systems, called `qubits`_. They are also involved in fundamental
quantum information processing protocols such as `quantum teleportation`_.

Suppose an example, for :ref:`qutipy-doc-tensor-product` of 3 Pauli-X (:math:`\sigma_1` or :math:`\sigma_x`)

.. math::

   \sigma_x( 111 ) = \sigma_x \otimes \sigma_x \otimes \sigma_x

To define :math:`X = \sigma_x( 111 )` using numpy, one need to define the entire matrix.

.. code:: python

   >>> import numpy as np
   >>>
   >>> # Define Pauli-X for [ 1 1 1 ] using numpy
   >>> X = np.array([
   >>>       [0, 0, 0, 0, 0, 0, 0, 1],
   >>>       [0, 0, 0, 0, 0, 0, 1, 0],
   >>>       [0, 0, 0, 0, 0, 1, 0, 0],
   >>>       [0, 0, 0, 0, 1, 0, 0, 0],
   >>>       [0, 0, 0, 1, 0, 0, 0, 0],
   >>>       [0, 0, 1, 0, 0, 0, 0, 0],
   >>>       [0, 1, 0, 0, 0, 0, 0, 0],
   >>>       [1, 0, 0, 0, 0, 0, 0, 0]
   >>> ])

These kind of definition for :math:`X = \sigma_x( 111 )` can be obtained easily using QuTIpy.

.. code:: python

   >>> from qutipy.Pauli import generate_nQubit_Pauli_X
   >>>
   >>> # Define Pauli-X for [ 1 1 1 ] using qutipy
   >>> X = generate_nQubit_Pauli_X([1, 1, 1])


.. _Pauli matrices: https://en.wikipedia.org/wiki/Pauli\_matrices
.. _quantum mechanics: https://en.wikipedia.org/wiki/Quantum\_mechanics
.. _Pauli equation: https://en.wikipedia.org/wiki/Pauli\_equation
.. _spin: https://en.wikipedia.org/wiki/Spin\_\(physics\)
.. _electromagnetic field: https://en.wikipedia.org/wiki/Electromagnetic\_field

.. _Hermitian: https://en.wikipedia.org/wiki/Hermitian\_matrix
.. _basis: https://en.wikipedia.org/wiki/Basis\_\(linear\_algebra\)
.. _vector space: https://en.wikipedia.org/wiki/Vector\_space
.. _Hermitian matrix: https://en.wikipedia.org/wiki/Hermitian\_matrix
.. _quantum information: https://en.wikipedia.org/wiki/Quantum\_information
.. _qubits: https://en.wikipedia.org/wiki/Qubit
.. _quantum teleportation: https://en.wikipedia.org/wiki/Quantum\_teleportation

Quantum Channels
----------------

A quantum channel is a communication channel which can transmit quantum information, as well as classical information.
It is a completely positive, trace-preserving linear map from density matrices to density matrices,
:math:`\rho \mapsto \sum\limits_i A_i \rho A^{\dagger}_i` with :math:`\sum\limits_i A^{\dagger}_i A_i = \mathbb{I}`.

An alternative definition of a quantum channel is a partial trace of a unitary transformation on a larger Hilbert space.

Applying a channel on a state is easy. For example, say, a `Depolarizing Channel`, is a channel
defined as a one-qubit `Pauli channel` given by :math:`p_x = p_y = p_z = \frac{p}{3}`, can be easily
implemented as such,

.. code-block:: python

   >>> from qutipy.channels import depolarizing_channel, apply_channel
   >>>
   >>> # The first element of the channel is the Kraus Operator
   >>> kraus_op = depolarizing_channel(0.2)[0]
   >>>
   >>> # Suppose `density_matrix` is a 2 x 2 density matrix,
   >>> # say, density_matrix = random_density_matrix(dim = 2)
   >>> evolved_density_matrix = apply_channel(kraus_op, density_matrix)
   >>>
   >>> evolved_density_matrix.shape
   (2, 2)

The `Depolarizing Channel` is applied as a `Kraus Operator`, which is a ``list`` type object. It will be much easier
to visualize the channel with the `Choi Representation` of the `Kraus Operator`, as such,

.. code-block:: python

   >>> from qutipy.channels import choi_representation
   >>>
   >>> # We represent the Kraus Operator, as Choi representation that will be
   >>> # a 4 x 4 matrix, representing the operator.
   >>> representation = choi_representation(kraus_op, 2)
   >>>
   >>> # representation = array(
   >>> #    [[□, □, □, □],
   >>> #     [□, □, □, □],
   >>> #     [□, □, □, □],
   >>> #     [□, □, □, □]]
   >>> # )
   >>>
   >>> representation.shape
   (4, 4)

For starters, a `random quantum channel` can be created with the definition ``qutipy.channels.random_quantum_channel``,

.. code-block:: python

   >>> from qutipy.channels import random_quantum_channel
   >>>
   >>> # Here we get the  Kraus Operator for a Random Quantum Channel
   >>> kraus_op = random_quantum_channel(2, 2, return_as="kraus")
   >>>
   >>> # Suppose `density_matrix` is a 2 x 2 density matrix,
   >>> # say, density_matrix = random_density_matrix(dim = 2)
   >>> evolved_density_matrix = apply_channel(kraus_op, density_matrix)
   >>>
   >>> evolved_density_matrix.shape
   (2, 2)


Pauli Channel
*************

Pauli channel is one of the most common channel, where the kraus operator are proportional to
that of the the pauli operators, i.e.

.. math::
   \rho 7 → p_I \rho + p_X X\rho X + p_Y Y\rho Y + p_ZZ\rho Z

where :math:`p_I, p_X, p_Y , p_Z \ge 0, p_I + p_X + p_Y + p_Z = 1`

.. code-block:: python

   >>> from qutipy.channels import Pauli_channel
   >>>
   >>> # Here we define the  Kraus Operator for the Pauli Quantum Channel
   >>> kraus_op, _, _ = Pauli_channel(px=0.16, py=0.04, pz=0.16)
   >>>
   >>> # Suppose exists a `density_matrix` of dimension 2, i.e. a 2 x 2 density matrix,
   >>> # let's assume the density_matrix for, `density_matrix = random_density_matrix(dim = 2)`
   >>> # then we can apply the pauli channel on the state as follows,
   >>> evolved_density_matrix = apply_channel(kraus_op, density_matrix)
   >>>
   >>> evolved_density_matrix.shape
   (2, 2)

Amplitude Damping Channel
*************************

Amplitude Damping channel is one of the most common channels. The amplitude damping
channel with decay parameter :math:`\gamma \in [0, 1]` is the channel :math:`\mathcal{A}_\gamma` given by
:math:`\mathcal{A}_\gamma(\rho) = A_1 \rho A_1^{\dagger} + A_2 \rho A_2^{\dagger}`, with the two Kraus
operators :math:`A_1` and :math:`A_2` defined as,

.. math::
   A_1 = \sqrt{\gamma}|0\rangle\langle 1|, \\
   A_2 = |0\rangle\langle 0| + \sqrt{1-\gamma}|1\rangle\langle 1|

It is straightforward to verify that :math:`A_1^{\dagger} A_1 + A_2^{\dagger} A_2 = \mathbb{1}`, so that :math:`\mathcal{A}_\gamma` is indeed trace preserving.

.. code-block:: python

   >>> from qutipy.channels import amplitude_damping_channel
   >>>
   >>> # Here we generate the  Kraus Operator for Amplitude Damping Quantum Channel
   >>> kraus_op = amplitude_damping_channel(0.2)
   >>>
   >>> # Suppose exists a `density_matrix` of dimension 2, i.e. a 2 x 2 density matrix,
   >>> # let's assume the density_matrix for, `density_matrix = random_density_matrix(dim = 2)`
   >>> # then we can apply the amplitude damping channel on the state as follows,
   >>> evolved_density_matrix = apply_channel(kraus_op, density_matrix)
   >>>
   >>> evolved_density_matrix.shape
   (2, 2)