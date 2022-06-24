#               This file is part of the QuTIpy package.
#                https://github.com/sumeetkhatri/QuTIpy
#
#                   Copyright (c) 2022 Sumeet Khatri.
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

import importlib
import inspect
import os

import qutipy


def get_modules(test_files):
    return [importlib.import_module(file.split(".")[0], file) for file in test_files]


def get_all_tests(directory):
    test_files = [
        file
        for file in os.listdir(directory)
        if file.startswith("test_") and file.endswith(".py")
    ]
    test_modules = get_modules(test_files)
    tests = {}
    tests = {
        module.__name__: [
            definition for definition in dir(module) if definition.startswith("test_")
        ]
        for module in test_modules
    }
    return tests


def get_all_objects(module, suffix=None):
    if suffix is None:
        suffix = module.__name__
    members = dict(inspect.getmembers(module))

    namespace = {}

    for member_name, member_object in members.items():
        if callable(member_object):
            # is a function definition
            namespace[member_object] = [".".join([suffix, member_name])]

    for member_name, member_object in members.items():
        if inspect.ismodule(member_object):
            # is a submodule
            if member_object.__package__ == "qutipy":
                # is a part of qutipy
                summodule_obj = get_all_objects(
                    member_object, ".".join([suffix, member_name])
                )
                for key, value in summodule_obj.items():
                    if key not in namespace:
                        namespace[key] = [*value]
                    else:
                        namespace[key] += [*value]
            else:
                # is not a part of qutipy. is an externally imported module.
                pass
    return namespace


def test_integrity():
    tests = get_all_tests("test")
    all_tests_list = [t.lower() for test in tests.values() for t in test]
    objects = get_all_objects(qutipy)
    tests_found = [
        (obj, location)
        for (obj, location) in objects.items()
        if "_".join(["test", obj.__name__.lower()]) in all_tests_list
    ]
    tests_not_found = [
        (obj, location)
        for (obj, location) in objects.items()
        if "_".join(["test", obj.__name__.lower()]) not in all_tests_list
    ]

    assert len(tests_found) > len(tests_not_found)
