.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _qutipy-doc-beginners-guide:

Beginners Guide
==================================

Welcome to the absolute beginner’s guide to QuTIpy! If you have comments or suggestions, please don’t hesitate to reach out!

Welcome to QuTIpy!
------------------


QuTIpy (**Quantum Theory of Information for Python**; pronounced `/cutiɛ paɪ/`) is an open source
Python library that’s used for performing calculations with quantum states, channels and quantum information processing protocols.
While there are many quantum information theory toolboxes that allow the user to perform basic operations
such as the `partial transposition <../modules/general-functions.md#firstheading>`_, [and partial trace], the focus of QuTIpy is to
allow you perform these operations in a `simple` and `quick` way.


Installing QuTIpy
-----------------
To install QuTIpy, we strongly recommend using a scientific Python distribution. If you’re
looking for the full instructions for installing NumPy on your operating system, see `Installing NumPy <./guide/installation.html>`_ .

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

The mathematical description of quantum systems can be summarized by the
following axioms. Each of these axioms is elaborated upon in the section indicated.


Quantum systems: A quantum system A is associated with a :hoverxreftooltip:`Hilbert space <qutipy-doc-hilbert-space>` :math:`\mathcal{H}_A`.
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

   from qutipy.states import Bell

Bell States
-----------
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

   # This will create a Bell State for a 2 dimensional system.
   # The resultant matrix will be of shape 4x4.

   Bell(d=2, z=1, x=1)


Random Quantum States
-----------------------



Random Unitaries
-----------------------



Random Quantum Channels
--------------------------

