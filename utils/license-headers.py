

"""Script which verifies that all source files have a license header.
Has two modes: 'fix' and 'check'. 'fix' fixes problems, 'check' will
error out if 'fix' would have changed the file.
"""

import os
import re
import sys
import datetime
from itertools import chain
from typing import Iterator, List

GIT = 'https://github.com/sumeetkhatri/QuTIpy'
AUTHOR = 'Sumeet Khatri'
YEAR = datetime.date.today().year

lines_to_keep = ["# -*- coding: utf-8 -*-\n", "#!/usr/bin/env python\n"]
license_header_lines = [
    # "# ========================================================================== #\n",
    # "#\n",
    "#               This file is part of the QuTIpy package.\n",
    f"#                {GIT}\n",
    "#\n",
    f"#                   Copyright (c) {YEAR} {AUTHOR}.\n",
    "#                       --.- ..- - .. .--. -.--\n",
    "#\n",
    "#\n",
    "# SPDX-License-Identifier: AGPL-3.0\n",
    "#\n",
    "#  This program is free software: you can redistribute it and/or modify\n",
    "# it under the terms of the GNU General Public License as published by\n",
    "# the Free Software Foundation, version 3.\n",
    "#\n",
    "# This program is distributed in the hope that it will be useful, but\n",
    "# WITHOUT ANY WARRANTY; without even the implied warranty of\n",
    "# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU\n",
    "# General Public License for more details.\n",
    "#\n",
    "# You should have received a copy of the GNU General Public License\n",
    "# along with this program. If not, see <http://www.gnu.org/licenses/>.\n",
    "#\n",
    # "# ========================================================================== #\n",
    "\n",
]


def find_files_to_fix(sources: List[str]) -> Iterator[str]:
    """Iterates over all files and dirs in 'sources' and returns
    only the filepaths that need fixing.
    """
    for source in sources:
        if os.path.isfile(source) and does_file_need_fix(source):
            yield source
        elif os.path.isdir(source):
            for root, _, filenames in os.walk(source):
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    if does_file_need_fix(filepath):
                        yield filepath


def does_file_need_fix(filepath: str) -> bool:
    if not re.search(r"\.pyi?$", filepath):
        return False
    if any([object.startswith('__') or object.startswith('.')
           for object in filepath.split('/')[0:-1]]):
        return False
    with open(filepath, mode="r") as f:
        first_license_line = None
        for line in f:
            if line == license_header_lines[0]:
                first_license_line = line
                break
            elif line not in lines_to_keep:
                return True
        for header_line, line in zip(
            license_header_lines, chain((first_license_line,), f)
        ):
            if line != header_line:
                return True
    return False


def add_header_to_file(filepath: str) -> None:
    with open(filepath, mode="r") as f:
        lines = list(f)
    i = 0
    for i, line in enumerate(lines):
        if line not in lines_to_keep:
            break
    lines = lines[:i] + license_header_lines + lines[i:]
    with open(filepath, mode="w") as f:
        f.truncate()
        f.write("".join(lines))
    print(f"Fixed {os.path.relpath(filepath, os.getcwd())}")


def main():
    mode = sys.argv[1]
    assert mode in ("fix", "check")
    sources = [os.path.abspath(x) for x in sys.argv[2:]]
    files_to_fix = find_files_to_fix(sources)

    if mode == "fix":
        for filepath in files_to_fix:
            add_header_to_file(filepath)
    else:
        no_license_headers = list(files_to_fix)
        if no_license_headers:
            print("No license header found in:")
            cwd = os.getcwd()
            [
                print(f" - {os.path.relpath(filepath, cwd)}")
                for filepath in no_license_headers
            ]
            sys.exit(1)
        else:
            print("All files had license header")


if __name__ == "__main__":
    main()