"""Tree-based potentials structure in Python.

This module contains the different implementations of potentials that are to be implemented. It is based on the impelmentations done in pgmpy.DiscreteFactor. Initially it will contains the classes:
- Tree: Tree 
- TreePotential: A potential based on a tree structure.
- TreeCPD: A conditional probability distribution based on TreePotential. Analogous to pgmpy.TabularCPD.

  Typical usage example:

  TODO
"""

import abc
import attr
import collections
import copy
import pprint
import typing

import pandas as pd
import numpy as np
import numpy.typing as npt


class Node(abc.ABC):

    def __init__(self, name: int):
        if name < 0:
            raise ValueError(f'Name must be non-negative, got: {name}')
        self.name = name

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    @abc.abstractmethod
    def __deepcopy__(self) -> Node:
        pass


class BranchNode(Node):

    def __init__(self, name: int, children: typing.List[Node]):
        super().__init__(name)
        self.children = children

    def is_terminal(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f'{self.__class__!r}({self.name!r}, {self.children!r})'

    def __deepcopy__(self) -> Node:
        return self.__cls__(self.name, copy.deepcopy(self.children))


class LeafNode(Node):

    def __init__(self, name: int, values: typing.List[float]):
        super().__init__(name)
        self.values = values

    def is_terminal(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f'{self.__class__!r}({self.name!r}, {self.values!r})'

    def __deepcopy__(self) -> Node:
        return self.__cls__(self.name, copy.deepcopy(self.values))


# Deberiamos hacer la poda directamente?
def _from_array(data: np.ndarray, assigned_vars: typing.List[int]) -> Node:
    var = len(assigned_vars)  # Next variable to be assigned
    cardinality = data.shape[var]

    # If this is the last variable
    if len(data.shape) - 1 == var:
        values = []
        for i in range(cardinality):
            values.append(data[tuple(assigned_vars + [i])])
        return LeafNode(var, values)

    # Otherwise assign a new variable to each child
    else:
        children = [
            _from_array(data, assigned_vars + [i]) for i in range(cardinality)
        ]
        return BranchNode(var, children)


@dataclass
class Tree:
    root : Node
    n_variables : int
    cardinality Tupl;e

    restraint_vars = attr.ib(type=typing.Dict[int, int], init=False)

    @classmethod
    def from_array(cls, data: np.ndarray) -> Node:
        if data.size == 0:
            raise ValueError('Array should be non-empty')

        return cls(root=_from_array(data, []),
                   n_variables=len(data.shape),
                   cardinality=data.shape)

    @classmethod
    def _prune(cls, node: Node):
        node.children = [Tree._prune(node) for node in node.children]

    @classmethod
    def _access(cls, node: Node, states: typing.Iterable) -> float:
        for var, state in enumerate(states):
            # If already restricted we choose restricted variable
            index = self.restraint_vars[
                var] if var in self.restraint_vars else state

            if node.is_terminal():
                return node.values[index]
            else:
                node = node.children[index]

    # DUDA: Should I use variadic input instead
    # Obviar variables restrained no?
    def access(self, states: typing.List[int]) -> float:

        if len(states) != self.n_variables:
            raise ValueError(f'Incorrect number of variables; ' +
                             f'expected: {n_variables}, received {len(states)}')

        for var, (value, bound) in enumerate(zip(states, self.cardinality)):
            if value >= bound:
                raise ValueError(f'Value for variable {var} is out of bound;' +
                                 f'received: {value}, limit : {bound}.')

        if self.restrained_variables:
            logging.warning(
                f'variables {self.restrained_variables.keys} will be ignored as they are restrained.'
            )

        return Tree._access(self.root, states)

    # Debería hacerlo inplace o cambiar?
    def restraint(self, variable: int, value: int):
        if variable >= len(self.cardinality):
            raise ValueError(f'Invalid value {variable} for variable.')

        if value >= (bound := self.cardinality[variable]):
            raise ValueError(f'Value for variable {variable} is out of bound;' +
                             f'received: {value}, limit : {bound}.')

        self.restraint_vars[variable] = state

    def unrestraint(self, variable: int):
        self.restraint_vars.pop(variable, None)


if __name__ == "__main__":
    array1 = np.array([[1, 1], [2, 2]])
    array2 = np.array([[[1, 1], [1, 7]], [[2, 34], [3, 23]]])
    tree1 = Tree.from_array(array1)
    tree2 = Tree.from_array(array2)
    print(repr(tree1))
    print(repr(tree2))
    tree1.access([0, 0])
    tree2.access([1, 1, 1])
