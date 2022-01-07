"""Tree-based potentials structure in Python.

This module contains the different implementations of potentials that are to be
implemented. It is based on the impelmentations done in pgmpy.DiscreteFactor.

Initially it will contains the classes:

- Node: Abstract Base Class for Tree nodes
- BranchNode: Class for non-terminal nodes. It contains a variable, as well as 
    one child for each state of the variable.
- LeafNode: Class for terminal nodes. It contains the associated value.
- Tree: Wrapper class for a tree root node. It performs most of the operations.

Typical usage example:

  # data is read from a numpy ndarray object
  data = np.array(get_data())
  tree = Tree.from_array(data)
  
  # We can perform most of the operations over tree. For example:
  tree.prune()
  tree.access([state_configuration])
 """

import abc
import collections
import copy
import dataclasses

from typing import List, Dict, Iterable

import numpy as np


class Node(abc.ABC):
    """Abstract Base Class for potential tree nodes"""

    def __init__(self):
        pass

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        """is_terminal returns whether a given node is terminal.

        Non-terminal nodes should have a children attribute. 
        Terminal nodes should have a value attribute. 
        """

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass


class BranchNode(Node):
    """BranchNode is a node that is not terminal.

    A branch node is a node that has children. It is associated with
    a variable, and has one child for every state such variable can be."""

    def __init__(self, name: int, children: List[Node]):
        super().__init__()

        if name < 0:
            raise ValueError(f'Name must be non-negative, got: {name}')

        self.name = name
        self.children = children

    def is_terminal(self) -> bool:
        """is_terminal returns False, as branch nodes are never terminal."""
        return False

    def __repr__(self) -> str:
        return f'{self.__class__!r}({self.name!r}, {self.children!r})'


class LeafNode(Node):
    """LeafNode is a node that is terminal.

    A leaf node is a node that has no children. It only contains a value."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def is_terminal(self) -> bool:
        """is_terminal returns True, as leaf nodes are always terminal."""
        return True

    def __repr__(self) -> str:
        return f'{self.__class__!r}({self.value!r})'


# Deberiamos hacer la poda directamente?
# Usamos otra vez tail recursion. Puede ser un problema con
# cantidades grandisimas de variables.
def _from_array(data: np.ndarray, assigned_vars: List[int]) -> Node:
    """TODO"""

    var = len(assigned_vars)  # Next variable to be assigned

    # If every variable is already selected
    if len(data.shape) == var:
        return LeafNode(data[tuple(assigned_vars)])

    else:
        cardinality = data.shape[var]
        children = [
            _from_array(data, assigned_vars + [i]) for i in range(cardinality)
        ]
        return BranchNode(var, children)


@dataclasses.dataclass
class Tree:
    """TODO"""

    root: Node
    n_variables: int
    cardinality: List[int] = dataclasses.field(default_factory=list)

    restraints: Dict[int, int] = dataclasses.field(
        default_factory=collections.defaultdict, init=False)

    @classmethod
    def from_array(cls, data: np.ndarray) -> Node:
        """TODO"""
        if data.size == 0:
            raise ValueError('Array should be non-empty')

        return cls(root=_from_array(data, []),
                   n_variables=len(data.shape),
                   cardinality=data.shape)

    # Consider that you are using tail recursion so it might overload with big files.
    # Suggestion: change it later.
    @classmethod
    def _prune(cls, node: Node):
        """TODO"""

        #node.children = [Tree._prune(node) for node in node.children]
        pass

    @classmethod
    def _access(cls, node: Node, states: Iterable,
                restraints: Dict[int, int]) -> float:

        for var, state in enumerate(states):
            index = restraints[var] if (var in restraints) else state
            if node.is_terminal():
                return node.value
            else:
                node = node.children[index]

        return node.value

    # DUDA: Should I use variadic input instead
    # Obviar variables restrained no?
    def access(self, states: List[int], *, ignore_restraints=False) -> float:
        """ TODO """

        if len(states) != self.n_variables:
            raise ValueError(f'Incorrect number of variables; ' +
                             f'expected: {n_variables}, received {len(states)}')

        for var, (value, bound) in enumerate(zip(states, self.cardinality)):
            if value >= bound:
                raise ValueError(f'Value for variable {var} is out of bound;' +
                                 f'received: {value}, limit : {bound}.')

        if self.restraints and not ignore_restraints:
            logging.warning(
                f'variables {self.restraints.keys} will be ignored as they are restrained.'
            )
            return Tree._access(self.root, states, self.restraints)
        else:
            return Tree._access(self.root, states, {})

    def restraint(self, variable: int, value: int):
        """TODO"""
        if variable >= len(self.cardinality):
            raise ValueError(f'Invalid value {variable} for variable.')

        if value >= (bound := self.cardinality[variable]):
            raise ValueError(f'Value for variable {variable} is out of bound;' +
                             f'received: {value}, limit : {bound}.')

        self.restraint_vars[variable] = state

    def unrestraint(self, variable: int):
        """TODO"""
        self.restraint_vars.pop(variable, None)


if __name__ == "__main__":
    array1 = np.array([[1, 1], [2, 2]])
    array2 = np.array([[[1, 1], [1, 7]], [[2, 34], [3, 23]]])
    tree1 = Tree.from_array(array1)
    tree2 = Tree.from_array(array2)
    print(repr(tree1))
    print(repr(tree2))
    print(tree1.access([0, 0]))
    print(tree2.access([1, 1, 1]))
