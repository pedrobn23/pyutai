"""
Define Element class for portentials
"""

import dataclasses
from typing import Dict, Tuple

IndexSelection = Dict[str, int]
"""Selection of values associated with each variable.

It stores a mapping from the name of a variable to its state.
"""


@dataclasses.dataclass
class Element:
    """
    An element is a pair of a state of the variables of a potential, and the
    associated value.
    """
    state: IndexSelection
    value: float


@dataclasses.dataclass
class TupleElement:
    """
    An element is a pair of a state of the variables of a potential, and the
    associated value.
    """
    state: Tuple[int]
    value: float
