.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _qutipy-doc-quickstart:

Quickstart
==========

Welcome to QuTIpy!
------------------

QuTIpy (**Quantum Theory of Information for Python**; pronounced `/cutiɛ paɪ/`) is an open source
Python library that’s used for performing calculations with quantum states, channels and protocols.
While there are many quantum information theory toolboxes that allow the user to perform basic operations
such as the `partial transposition <../modules/general-functions.md#firstheading>`_, new tests are
constantly discovered.

The core of QuTIpy package is a suite of mathematical functions that make working with quantum states,
channels and protocols, quick and easy.

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
using the operators :math:`X`, :math:`Z`, and :math:`ZX`, where the two-qubit
`maximally entangled quantum state`_ vectors is represented as

:math:`\displaystyle |\phi_{z,x}\rangle = (Z^zX^x \otimes I)|\phi^{+}\rangle` for :math:`z, x \in {0, 1}`.

Above, we generated a :math:`d`-dimensional Bell State with :math:`0 <= z`, :math:`x <= d-1`.

.. _Bell state: https://en.wikipedia.org/wiki/Bell_state
.. _maximally entangled quantum state: https://github.com/arnavdas88/QuTIpy-Tutorials/blob/main/modules/states.md#maximally-entangled-state
