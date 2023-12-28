.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _qutipy-doc-states:

States
======

Quantum mechanics is the backbone for quantum information processing, and many aspects of it cannot be
explained by classical reasoning. **Quantum states are the key mathematical objects in quantum theory.
Quantum states are states of knowledge, representing uncertainty about the real physical state of the
system.**

.. hint::
    For example, there is no strong classical analogue for pure quantum states or
    entanglement, and this leads to stark differences between what is possible in the classical and quantum
    worlds.

However, at the same time, it is important to emphasize that all of classical information theory is subsumed
by quantum information theory, so that whatever is possible with classical information processing is also
possible with quantum information processing. As such, quantum information subsumes classical information while
allowing for richer possibilities.

Quantum Systems
---------------

A quantum system :math:`A` is associated with a `Hilbert space`_ , :math:`H_{A}`.
The state of the system :math:`A` is described by a density operator, which is a unit-trace,
positive semi-definite linear operator acting on :math:`H_{A}`.

Bipartite Quantum Systems
*************************

For distinct quantum systems :math:`A` and :math:`B` with associated `Hilbert space`_
:math:`H_{A}` and :math:`H_{B}` , the composite system :math:`AB` is associated with the `Hilbert space`_
:math:`H_{A} \otimes  H_{B}`. This joint state is described by a bipartite quantum state
:math:`\rho_{AB} \in D(H_{A} \otimes H_{B})`. For brevity, the joint `Hilbert space`_
:math:`H_A \otimes H_B` of the composite system :math:`AB` is denoted by :math:`H_{AB}` .

Measurement
***********

The `Measurement`_ of a `quantum systems`_ :math:`A` is described by a
`Positive Operator Valued Measure (POVM)`_ :math:`\{M_{x}\}_{x \in {X}}`, which
is defined to be a collection of positive semi-definite operators indexed by a finite alphabet
satisfying :math:`\sum_{x \in X} {M_x} = 1_{H_A}`.

If the system is in the state :math:`\rho`, then the probability :math:`Pr[x]` of obtaining the outcome :math:`x` is given by the Born rule as :math:`Pr[x] = Tr[M_x \rho ]`.

Evolution
*********

The `Evolution`_ of the state of a quantum system is described by a quantum channel, which is a linear, completely positive, and trace-preserving map acting on the state of the system.

Quantum States
--------------

A `quantum state`_ is a mathematical entity that provides
a probability distribution for the outcomes of each possible measurement on a `quantum systems`_.
The `quantum state`_ of a `quantum systems`_ is described by a `density operator`_ acting on the
underlying `Hilbert space`_ of the `quantum systems`_. Knowledge of the `quantum state`_ together
with the rules for the system's `evolution in time`_ exhausts all that can be predicted about the
system's behavior.

.. hint::

    A `density operator`_ is a unit-trace, positive semi-definite linear operator. We denote the set of
    density operators on a `Hilbert space`_ :math:`H` as :math:`D(H)`

We typically use the Greek letters :math:`\rho`, :math:`\sigma`, :math:`\tau`, or :math:`\omega` to denote
`quantum state`_.

States in QuTIpy Package
************************

QuTIpy contains the definitions of these states, inside the `states` sub-module, and can be imported as such

.. code:: python

    >>> from qutipy.states import (
    >>>     Bell_state,
    >>>     GHZ_state,
    >>>     MaxEnt_state,
    >>>     MaxMix_state,
    >>>     RandomDensityMatrix,
    >>>     RandomStateVector,
    >>>     Werner_state,
    >>>     Werner_twirl_state,
    >>>     graph_state,
    >>>     isotropic_state,
    >>>     isotropic_twirl_state,
    >>>     singlet_state
    >>> )

Maximally Entangled State
*************************

A pure state :math:`\displaystyle \psi_{AB} = |\psi\rangle\langle\psi|_{AB}`, for two systems :math:`A` and :math:`B` of the same dimension :math:`d`, is called **Maximally Entangled** if the Schmidt coefficients of :math:`\displaystyle |\psi\rangle_{AB}` are all equal to :math:`\frac{1}{\sqrt{d}}` , with :math:`d` being the Schmidt rank of :math:`\displaystyle |\psi\rangle_{AB}`.

In other words, :math:`\psi_{AB}` is called maximally entangled if :math:`\displaystyle |\psi\rangle_{AB}` has the Schmidt decomposition, :math:`\displaystyle |\psi\rangle_{AB} = \frac{1}{\sqrt{d}}\sum_{k=1}^{d} |e_k\rangle_A \otimes |f_k\rangle_B` for some orthonormal sets :math:`\displaystyle \{ |e_k\rangle_A : 1 \le k \le d \}` and :math:`\displaystyle \{ |f_k\rangle_B : 1 \le k \le d \}`.



In simple terms, the **Maximally Entangled** can be written as :math:`\displaystyle (\frac{1}{\sqrt{d}})*(|0\rangle|0\rangle+|1\rangle|1\rangle+...+|d-1\rangle|d-1\rangle)` and can be created using the `MaxEnt_state` function.

.. code:: python

    >>> # This will create a Macimally Entangled State for a 3 dimensional system.
    >>> # The resultant matrix will be of shape 9x9.
    >>> MaxEnt_state(3)

Bell State
**********

A `Bell state`_ is defined as a `maximally entangled quantum state`_ of two qubits. It can be
described as one of four entangled two qubit quantum states, known collectively as the four `Bell state`_.

:math:`\displaystyle |\phi^{+}\rangle \equiv |\phi_{0, 0}\rangle = \frac{1}{\sqrt{2}} (|0, 0\rangle + |1, 1\rangle)`

:math:`\displaystyle |\phi^{-}\rangle \equiv |\phi_{1, 0}\rangle = \frac{1}{\sqrt{2}} (|0, 0\rangle - |1, 1\rangle)`

:math:`\displaystyle |\psi^{+}\rangle \equiv |\phi_{0, 1}\rangle = \frac{1}{\sqrt{2}} (|0, 1\rangle + |1, 0\rangle)`

:math:`\displaystyle |\psi^{-}\rangle \equiv |\phi_{1, 1}\rangle = \frac{1}{\sqrt{2}} (|0, 1\rangle - |1, 0\rangle)`

A generalized version of the above `Bell state`_ is explained below,

Using the operators :math:`X`, :math:`Z`, and :math:`ZX`, we define the following set of
four entangled two-qubit state vectors :math:`\displaystyle |\phi_{z,x}\rangle = (Z^zX^x \otimes I)|\phi^{+}\rangle`
for :math:`z, x \in {0, 1}`.

To generates a :math:`d`-dimensional Bell State with :math:`0 <= z`, :math:`x <= d-1`, we can simply
call the module `Bell_state` that was imported above.

.. code:: python

    >>> # This will create a Bell State for a 2 dimensional system.
    >>> # The resultant matrix will be of shape 4x4.
    >>> Bell_state(d=2, z=1, x=1)

Singlet State
*************

A singlet state is defined as  :math:`\frac{1}{(d^2-d)} \times (I_{(d^2)}-F)` where :math:`F` is a Swap Operator.

Generating a singlet state is as easy as writing a single word,

.. code:: python

    >>> # This will create a Singlet State for a 3 dimensional system.
    >>> # The resultant matrix will be of shape 9x9.
    >>> singlet_state(3)




.. _density operator: https://en.wikipedia.org/wiki/Density_matrix#Definition_and_motivation
.. _quantum state: https://en.wikipedia.org/wiki/Quantum_state
.. _Bell state: https://en.wikipedia.org/wiki/Bell_state
.. _evolution in time: https://en.wikipedia.org/wiki/Quantum_channel#Time_evolution
.. _Hilbert space: https://en.wikipedia.org/wiki/Hilbert_space
.. _Positive Operator Valued Measure (POVM): https://en.wikipedia.org/wiki/POVM
.. _Measurement: https://en.wikipedia.org/wiki/Measurement_in_quantum_mechanics
.. _Evolution: https://en.wikipedia.org/wiki/Quantum_channel#Time_evolution
.. _maximally entangled quantum state: #maximally-entangled-state
.. _quantum systems: #quantum-systems