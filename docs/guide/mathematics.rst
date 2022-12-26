.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _qutipy-doc-mathematics:


Mathematics
===========

In this package we work with finite-dimensional quantum systems, which can be described using finite-dimensional Hilbert spaces.
The mathematics of finite-dimensional Hilbert spaces is essentially linear and matrix analysis. In this section, we provide a summary
of some basic facts from linear and matrix analysis...


.. _qutipy-doc-hilbert-space:


Hilbert Space
-------------

Hilbert Space is an inner product space that is complete with respect to the norm defined
by the inner product. Mathematically, A Hilbert Space is a vector space :math:`H` with an inner
product :math:`\langle f, g \rangle` such that the norm defined by :math:`|f| = \sqrt{\langle f, f \rangle}`,
turns :math:`H` into a complete metric space. Completeness in this context means that every
Cauchy sequence of elements of the space converges to an element in the space, in the sense
that the norm of differences approaches zero. A Hilbert Space is always a Banach space, but
the converse need not hold.

.. _qutipy-doc-linear-operator:


Linear Operator
---------------

As a metric space, Hilbert space can be considered an infinite-dimensional linear topological space