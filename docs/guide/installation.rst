.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _qutipy-doc-installations:

Installation
============

If you’d like to be able to update your QuTIpy code occasionally with the latest
bug fixes and improvements, follow these instructions:

Manual Installation
___________________

#. Make sure that you have `Git`_ installed and that you can run its commands from a shell. (Enter `git help` at a shell prompt to test this.)
#. Check out QuTIpy’s main development branch like so:

    .. code-block:: shell

      $ git clone https://github.com/sumeetkhatri/QuTIpy.git

    `This will create a directory "QuTIpy" in your current directory.`

#. Make sure that the Python interpreter can load QuTIpy’s code. The most convenient way to do this is to use a virtual environment and `pip`_. The :ref:`contributing tutorial <qutipy-doc-contribute>` walks through how to create a virtual environment.
#. After setting up and activating the virtual environment, run the following command:

    .. code-block:: shell

      $ python -m pip install -e QuTIpy/

    `This will make QuTIpy’s code importable. In other words, you’re all set!`



When you want to update your copy of the QuTIpy source code, run the command :code:`git pull` from within the `QuTIpy` directory. When you do this, Git will download any new change available.




PIP Installation
________________

#. Install the package directly from `GitHub <https://github.com/sumeetkhatri/QuTIpy>`_ using pip, like so:

    .. code-block:: shell

      $ python -m pip install https://github.com/sumeetkhatri/QuTIpy.git

   Or

   Install the package directly from `PyPI <https://pypi.org/project/QuTIpy>`_ using pip, like so:

    .. code-block:: shell

      $ python -m pip install QuTIpy
|
|

.. note::
   Check if the installation is successfull :
   ------------------------------------------

   Run the shell command ,
    .. code-block:: shell

      $ echo "import qutipy; print(qutipy.version);" | python;

   This should output the version of qutipy installed in your system like this ,
    .. code-block:: shell

      $ 0.1.0


.. _git: https://git-scm.com/
.. _pip: https://pip.pypa.io/