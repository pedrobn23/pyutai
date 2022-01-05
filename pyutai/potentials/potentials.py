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


class BranchNode(Node):

    def __init__(self, name: int, children: typing.List[Node]):
        super().__init__(name)
        self.children = children

    def is_terminal(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f'{self.__class__!r}({self.name!r}, {self.children!r})'


class LeafNode(Node):

    def __init__(self, name: int, values: typing.List[float]):
        super().__init__(name)
        self.values = values

    def is_terminal(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f'{self.__class__!r}({self.name!r}, {self.values!r})'


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


@attr.s
class Tree:
    root = attr.ib(type=Node)
    n_variables = attr.ib(type=int)
    cardinality = attr.ib(type=typing.Tuple[int])

    @classmethod
    def from_array(cls, data: np.ndarray) -> Node:
        if data.size == 0:
            raise ValueError('Array should be non-empty')

        return cls(root=_from_array(data, []),
                   n_variables=len(data.shape),
                   cardinality=data.shape)

    @classmethod
    def _access(cls, node: Node, states: typing.Iterable) -> float:
        for index in states:
            if node.is_terminal():
                return node.values[index]
            else:
                node = node.children[index]

    # DUDA: Should I use variadic input instead
    def access(self, states: typing.List[int]) -> float:

        if len(states) != self.n_variables:
            raise ValueError(f'Incorrect number of variables; ' +
                             f'expected: {n_variables}, received {len(states)}')

        for var, (value, bound) in enumerate(zip(states, self.cardinality)):
            if value >= bound:
                raise ValueError(f'Value for variable {var} is out of bound;' +
                                 f'received: {value}, limit : {bound}.')

        return Tree._access(self.root, states)

    @classmethod
    def _retraint(cls, node: Node, variable: int, state: int):
        pass

    # Debería hacerlo inplace o cambiar?
    def restraint(self,
                  variable: int,
                  state: int,
                  *,
                  inplace: bool = False) -> Node or None:

        if inplace:
            Tree._restraint(self.root, variable, state)

        else:
            node = node.copy()
            Tree._restraint(node, variable, state)
            return node


if __name__ == "__main__":
    array1 = np.array([[1, 1], [2, 2]])
    array2 = np.array([[[1, 1], [1, 7]], [[2, 34], [3, 23]]])
    tree1 = Tree.from_array(array1)
    tree2 = Tree.from_array(array2)
    print(repr(tree1))
    print(repr(tree2))
    tree1.access([0, 0])
    tree2.access([1, 1, 1])
