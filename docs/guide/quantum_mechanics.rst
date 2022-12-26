.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. _qutipy-doc-quantum-mechanics:

Quantum Mechanics
=================

Quantum mechanics is a fundamental theory in physics that provides a description of the physical
properties of nature at the scale of atoms and subatomic particles. It is the foundation of all
quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum
information science.

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

Here, we provide an overview of quantum mechanics, placing particular emphasis on those aspects
of quantum mechanics that are useful for the quantum information processing and communication
protocols.



Quantum system
--------------

A Quantum System is a system which holds conservation laws, and **necessarily** have a
statistical space-time description. A **quantum system** :math:`A` is associated with a
:hoverxreftooltip:`Hilbert space <qutipy-doc-hilbert-space>` :math:`\mathcal{H}_A`. The
state of the system :math:`A` is described by a density operator, which is a unit-trace,
positive semi-definite linear operator acting on :math:`\mathcal{H}_A`.

We mainly consider only finite-dimensional quantum systems, that is, quantum systems described by
finite-dimensional Hilbert spaces.

The **qubit** is perhaps the most fundamental quantum system and is the quantum analogue of the (classical)
bit. Every physical system with two distinct degrees of freedom obeying the laws of quantum mechanics
can be considered a qubit system. The Hilbert space associated with a qubit system is :math:`\mathbb{C}^2`, whose standard
orthonormal basis is denoted by :math:`\{ |0 \rangle , |1 \rangle \}`.

A **qutrit system** is a quantum system consisting of three distinct physical degrees of freedom. The
Hilbert space of a qutrit is :math:`\mathbb{C}^3`, with the standard orthonormal basis denoted by
:math:`\{ |0 \rangle , |1 \rangle , |2 \rangle \}`. Qutrit systems are less commonly considered than
qubit systems for implementations, although one important example of an implementation of a qutrit
system occurs in quantum optical systems, which we briefly discuss below.

A **qudit system** is a quantum system with d distinct degrees of freedom and is described by the
Hilbert space :math:`\mathbb{C}^d`, with the standard orthonormal basis denoted by
:math:`\{ |0 \rangle , |1 \rangle ,  \dots , |d-1 \rangle \}`. The spin states of every spin-:math:`j`
atom can be used to realize a qudit system with :math:`d = 2 j + 1`.

An important quantum system, particularly for the implementation of many quantum communication protocols,
is a **quantum optical system**. By a quantum optical system we mean a physical system, such as an optical
cavity or a fiber-optic cable, in which modes of light, with photons as information carriers, propagate.
A mode of light has a well defined momentum, frequency, polarization, and spatial direction. Formally,
a quantum optical system with d distinct modes is described by the Fock space :math:`\mathcal{F}_B(\mathbb{C}^d)`,
which is a Hilbert space equipped with the orthonormal occupation number basis
:math:`\{ |n_1, n_2,  \dots , n_d \rangle : n_1, n_2,  \dots , n_d \ge 0 \}` where :math:`n_j`,
for :math:`1 \le j \le d`, indicates the number of photons occupied in mode :math:`j`.

Quantum states
**************
The state of a quantum system is described by a density operator acting on
the underlying Hilbert space of the quantum system. A density operator is
a unit-trace, positive semi-definite linear operator. Usually, we identify
a state with its corresponding density operator. We denote the set of
density operators on a Hilbert space :math:`H` as :math:`D(H)`.

he extremal points in the convex set of quantum states are called pure states.
A pure state is a unit-rank projection onto a vector in the Hilbert space. Concretely,
pure states are of the form :math:`|\psi\rangle\langle\psi|` for every normalized vector
:math:`|\psi\rangle\in H`. For convenience, we sometimes denote :math:`|\psi\rangle\langle\psi|`
simply as :math:`\psi`, and refer to the unit vector :math:`|\psi\rangle` as a state vector.
Since every element of a convex set can be written as a convex combination of the extremal points
in the set, every quantum state :math:`\rho` that is not a pure state can be written as

.. math::

   \rho = \sum\limits_{x \in \chi} p(x) |\psi_x\rangle\langle\psi_x|

for some set :math:`\{|\psi_x\rangle\}_{x \in \chi}` of state vectors defined with respect to a
finite alphabet :math:`\chi`, where :math:`p:\chi\rightarrow [0, 1]` is a probability distribution.


States :math:`\rho` that are not pure are called **mixed states**, because they can be thought
of as arising from the lack of knowledge of which pure state from the set :math:`\{|\psi_x\rangle\}_{x \in \chi}`
in the system has been prepared. Note that the decomposition in of a quantum state into pure
states is generally not unique.

A state :math:`\rho` is called **maximally mixed** if the set :math:`\{|\psi_x\rangle\}_{x \in \chi}` consists of
:math:`d` orthonormal states and the probabilities :math:`p(x)` are uniform (i.e., :math:`p(x) = \frac{1}{d}`
for all :math:`x \in \chi`). In this case, it follows that

.. math::

   \rho = \frac{\mathbb{1}_d}{d} =: \pi_d



Measurement
***********
``Born Rule``

The measurement of a quantum system :math:`A`` is described by a Positive Operator-Valued Measure
(POVM) :math:`\{M_x\}_{x \in \chi}`, which is defined to be a collection of positive semi-definite
operators indexed by a finite alphabet satisfying :math:`\sum\limits_{x \in \chi} M_x = \mathbb{1}_{H_A}`.
If the system is in the state :math:`\rho`, then the probability :math:`Pr[x]` of obtaining the outcome
:math:`x` is given by the Born rule as,

.. math::

   Pr[x] = Tr[M_x\rho]


Observables
***********

An observable is a Hermitian linear operator that can operate on system states describing something that can be observed.
Together with a transition state an observable determines the probabilities of the possible outcomes of a measurement of
that observable on the quantum system prepared in the given state.

Furthermore, a physical observable :math:`O` corresponds to a Hermitian operator acting on the underlying Hilbert space.
The observable :math:`O` has a spectral decomposition as,

.. math::

   O = \sum\limits_{\lambda \in spec(O)} \lambda\Pi_\lambda

where :math:`spec(O)` is the set of distinct eigenvalues of :math:`O` and :math:`\Pi_\lambda` is a spectral
projection. A measurement of :math:`O` is described by the POVM :math:`\{\Pi_\lambda\}_λ`, which is
indexed by the distinct eigenvalues :math:`\lambda` of :math:`O`. The expected value :math:`\langle O\rangle_\rho` of the
observable :math:`O` when the state is :math:`\rho` is given by,

.. math::

   \langle O\rangle_\rho := Tr[O\rho]


Evolution
*********
``Unitaries and Channels``

The evolution of the state of a quantum system is described by a
quantum channel, which is a linear, completely positive, and trace-preserving
map acting on the state of the system. Mathematically, the evolution is described by a
quantum channel. As quantum communication necessarily involves the evolution
of quantum systems (such as the evolution of photons when travelling through
an optical fiber), quantum channels are the primary object of study.

The evolution of a (non-relativistic) quantum system is governed by the Schrödinger equation:

.. math::
   i\hslash \frac{\partial }{\partial t} | \psi (t) \rangle = H(t)| \psi (t) \rangle

where :math:`| \psi (t) \rangle` is the state vector of the system at time :math:`t \ge 0` and :math:`H(t)` is the
Hamiltonian operator of the system at time :math:`t`. The Hamiltonian operator is a Hermitian operator that describes
the energy of the system. This describe the evolution of so-called closed quantum systems, and this evolution is
given by unitary maps. In other words, the solution to the Schrödinger equation is :math:`| \psi (t) \rangle = U(t)| \psi_0 \rangle`
for all :math:`t \ge 0`, where :math:`\psi_0` is an initial state vector of the system (at time :math:`t = 0`) and
:math:`U(t)` is a unitary operator.

More generally, we are interested in the evolution of open quantum systems, i.e., quantum systems that interact with
an external environment that is out of our control. For such systems, the same connection as before holds. In fact,
the evolution is given by a joint unitary evolution of the system and environment followed by discarding the state of
the environment. Every completely positive trace-preserving map (i.e., every quantum channel) can be viewed in terms
of a joint unitary evolution with an environment followed by discarding the state of the environment.

Thus, from an abstract, information-theoretic perspective, the evolution of a quantum system is given simply
by a quantum channel, and the details of the actual physical system of interest (which would be given by the
Hamiltonian operator) are unimportant. This viewpoint is powerful, because this helps us realize that virtually
every operation on quantum states, including measurements, is a quantum channel.

