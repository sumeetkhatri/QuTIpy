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

import nox

PYTHON_ENV = python = ["3.7", "3.8", "3.9", "3.10"]

SOURCE_FILES = (
    "setup.py",
    "noxfile.py",
    "qutipy/",
    "test/",
)


@nox.session(python=PYTHON_ENV)
def tests(session):
    """Run the test suite."""
    session.install("black")
    session.install("pytest")
    session.install(".")
    session.run("pytest")


@nox.session(python=PYTHON_ENV[-1])
def lint(session):
    """Run the lint suite."""

    session.install("flake8", "black", "mypy", "isort", "types-requests")

    session.run("isort", "--check", "--profile=black", *SOURCE_FILES)
    session.run("black", "--target-version=py39", "--check", *SOURCE_FILES)
    session.run("python", "utils/license-headers.py", "check", *SOURCE_FILES)


@nox.session(python=PYTHON_ENV)
def formatting(session):
    """Run the formatter suite."""
    session.install(
        "black", "isort", "autopep8", "flake8-black", "flake8-bugbear", "flake8-bandit"
    )

    session.run("isort", "--profile=black", *SOURCE_FILES)
    session.run("black", "--target-version=py39", *SOURCE_FILES)
    session.run("stubgen", "-p", "qutipy", external=True)
    session.run("python", "utils/license-headers.py", "fix", *SOURCE_FILES)
