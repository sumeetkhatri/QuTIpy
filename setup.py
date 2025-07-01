#               This file is part of the QuTIpy package.
#                https://github.com/sumeetkhatri/QuTIpy
#
#                   Copyright (c) 2023 Sumeet Khatri.
#                       --.- ..- - .. .--. -.--
#
#
# SPDX-License-Identifier: AGPL-3.0
#
#  This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#


from setuptools import find_packages, setup

# from sphinx.setup_command import BuildDoc

from version import VERSION

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="QuTIpy",
    version=VERSION,
    description="A package to perform quantum information calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sumeetkhatri/QuTIpy",
    project_urls={
        "GitHub": "https://github.com/sumeetkhatri/QuTIpy",
        "Homepage": "https://arnav-das.gitbook.io/qutipy-quantum-theory-of-information-for-python",
        "PyPi": "https://pypi.org/project/QuTIpy",
        "Author": "https://sumeetkhatri.com",
    },
    author="Sumeet Khatri",
    author_email="khatri6000@gmail.com",
    maintainer="Sumeet Khatri, Arnav Das",
    maintainer_email="khatri6000@gmail.com",
    keywords="qutipy quantum sdk",
    license="LGPLv3",
    packages=find_packages(),
    # cmdclass={"build_sphinx": BuildDoc},
    # these are optional and override conf.py settings
    command_options={
        "build_sphinx": {
            "project": ("setup.py", "QuTIpy"),
            "version": ("setup.py", "0.1.0a"),
            "release": ("setup.py", "0.1"),
            "source_dir": ("setup.py", "docs"),
        }
    },
    install_requires=[
        "numpy>=1.21.3",
        "scipy",
        "sympy",
        "cvxpy",
        "pytest",
    ],
    extras_require={
        "dev": [
            "nox",
            "pytest",
        ],
        "docs": [
            "sphinx",
            "sphinxemoji",
            "pydata-sphinx-theme",
            "numpydoc",
            "sphinx_panels",
            "matplotlib",
            "Ipython",
            "sphinx-hoverxref",
        ],
    },
    classifiers=[
        # License
        "License :: OSI Approved :: GNU Affero General Public License v3",
        # Project Maturity
        "Development Status :: 1 - Planning",
        # Topic
        "Topic :: Communications",
        "Topic :: Scientific/Engineering :: Physics",
        # Intended Audience
        "Intended Audience :: Science/Research",
        # Compatibility
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        # Python Version
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
