.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _qutipy-doc-whatis-quitpy:

What is QuTIpy ?
================

QuTIpy
-------
QuTIpy stands for Quantum Theory of Information for Python; pronounced "cutie pie". It is a package for performing calculations
with quantum states, channels, and quantum information processing protocols.

The primary focus of QuTIpy is to be `simple`. Indeed, with QuTIpy, you work directly with numpy arrays when it comes to both quantum
states and channels, without having to worry about defining lots of meta-data and without having to work with proprietary data structures.
This means that, with basic knowledge of numpy, you can begin using this package. You can do very simple things very simply and quickly,
such as finding the coefficients of a particular quantum state in the Pauli basis. You can also easily calculate various quantities
of interest, such as norms, entropies, entanglement measures, and even quantities that don't have a closed-form expressions but must
in general be calculated numerically using, e.g., semi-definite programming.

QuTIpy is most comparable to the QETLAB package for MATLAB/Octave: http://www.qetlab.com/Main_Page

