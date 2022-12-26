.. QuTIpy documentation master file, created by
   sphinx-quickstart on Thu Jun  9 22:10:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _qutipy-doc-contribute:

Contribute Guide
================


Reporting issues
----------------

When reporting issues please include as much detail as possible about your operating system,
QuTIpy version and python version. Whenever possible, please also include a brief, self-contained
code example that demonstrates the problem.

If you are reporting a segfault please include a GDB traceback, which you can generate by
following these instructions.

Contributing code
-----------------

Thanks for your interest in contributing code to QuTIpy!

   - If this is your first time contributing to a project on GitHub, please read through our guide to contributing to QuTIpy
   - If you have contributed to other projects on GitHub you can go straight to our `development workflow`_

Either way, please be sure to follow our convention for commit messages.

Suggested ways to work on your development version (compile and run the tests without interfering with system packages) are described in doc/source/dev/development_environment.rst.

A note on feature enhancements/API changes
******************************************

If you are interested in adding a new feature to QuTIpy, consider submitting your feature proposal to the mailing list, which
is the preferred forum for discussing new features and API changes.

Development workflow
--------------------

You already have your own forked copy of the QuTIpy repository, by following Create a
QuTIpy fork, Make the local copy, you have configured `git <https://git-scm.com/>`_, and have
linked the upstream repository as explained in Linking your repository to the upstream repo.

What is described below is a recommended workflow with Git.

Basic workflow
**************

In short:

   - Create a new branch for each set of edits that you do. See below.

   - Hack away! Code your way in !!!

   - When finished, push your feature branch to your own Github repo, and create a pull request.

This way of working helps to keep work well organized and the history as clear as possible.

Making a new feature branch
***************************

First, fetch new commits from the ``upstream`` repository:

.. code-block:: bash

   $ git fetch upstream

Then, create a new branch, say ``my-new-feature``, based on the main branch of the upstream repository:

.. code-block:: bash

   $ git checkout -b my-new-feature upstream/main

Once the **checkout** is done, we can start coding |:tada:| |:tada:| |:tada:| !!!

The editing workflow
********************

Once the coding is done, make sure that the code is not broken and is properly formatted.

.. code-block:: bash

   $ nox -e test # Run the unit tests
   $ nox -e formatting # Formats the code properly

Now the code is ready to be committed and pushed.

.. code-block:: bash

     # code! code! code!
   $ git status # Optional
   $ git diff # Optional
   $ git add modified_file
   $ git commit
     # push the branch to your own Github repo
   $ git push origin my-new-feature


