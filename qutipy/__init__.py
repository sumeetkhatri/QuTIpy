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

import numpy as np
from numpy.linalg import eig, norm
from scipy.linalg import expm, logm

# from qutipy.channels import *
# from qutipy.Clifford import *
# from qutipy.distance_measures import *
# from qutipy.entropies import *
# from qutipy.fermions import *
# from qutipy.fidelities import *
# from qutipy.gates import *
# from qutipy.general_functions import *
# from qutipy.linalg import *
# from qutipy.misc import *
# from qutipy.Pauli import *
# from qutipy.protocols import *
# from qutipy.states import *
# from qutipy.su import *
# from qutipy.Weyl import *

# Core Functions
from . import general_functions as general_functions
from .general_functions import dag, eye, ket, syspermute, tensor

# Toolkit
from . import fidelities as fidelities
from . import protocols as protocols
from . import entropies as entropies
from . import fermions as fermions
from . import channels as channels
from . import distance as distance
from . import linalg as linalg
from . import states as states
from . import gates as gates
from . import misc as misc
from . import su as su

# Name Convention
from . import clifford as Clifford
from . import pauli as Pauli
from . import weyl as Weyl

__version__ = "0.1.0"
__author__ = "Sumeet Khatri"
