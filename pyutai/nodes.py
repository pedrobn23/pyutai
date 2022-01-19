""" Nodes for Tree-based potentials structure in Python.

This module contains the different implementations of nodes to be used by 
Initially it will contains the classes:

- Node: Abstract Base Class for Tree nodes
- BranchNode: Class for non-terminal nodes. It contains a variable, as well as
    one child for each state of the variable.
- LeafNode: Class for terminal nodes. It contains the associated value.

Typical usage example:

  # data is read from a numpy ndarray object
  data = np.array(get_data())
  tree = Tree.from_array(data)

  # We can perform most of the operations over tree. For example:
  tree.prune()
  tree.access([state_configuration()])
"""

import abc
import collections
import copy
import dataclasses
import logging

from typing import Dict, Iterable, List, Tuple

import numpy as np


class Node(abc.ABC):
    """Abstract Base Class for potential Value-Tree nodes"""

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

    @abc.abstractmethod
    def __eq__(self, other):
        pass

    # have to include memo
    @abc.abstractmethod
    def __deepcopy__(self, memo):
        pass

    # deberíamos memorizarla?
    @abc.abstractmethod
    def size(self):
        """size is the number of nodes that lives under the root.
        """


# should I make hashing nodes.
class BranchNode(Node):
    """BranchNode is a node that has children.

    A branch node is characterized by not being terminal. As we deal
    with value-tree, each branch node has to be associated with a
    variable, and have a children for every state possible for the
    variable.

    Attributes:
        name (int): Name of the variable associated with the node.
        children (List[Node]): each of the node associated with each state of variable name.
    """

    def __init__(self, name: int, children: List[Node]):
        """Initializes BranchNode

        checks that the name is a non-negative value and initializes the node.

        Args:
            name: Name of the variable associated with the node. 
            children: Each of the nodes associated with each state of variable name.
                It should be non-empty.

        Raises:
            ValueError: Whenever name is negative of children is empty.
        """
        super().__init__()

        if not children:
            raise ValueError(
                f'Children must be a non-empty list, got: {children}')

        self.name = name
        self.children = children

    def is_terminal(self) -> bool:
        """is_terminal returns False, as branch nodes are never terminal."""
        return False

    def __repr__(self) -> str:
        return f'{self.__class__}({self.name!r}, {self.children!r})'

    def __eq__(self, other: Node):
        if not other.is_terminal():
            if len(other.children) == len(self.children):
                return all(
                    a == b for a, b in zip(self.children, other.children))

        return False

    # have to include memo
    def __deepcopy__(self, memo):
        return type(self)(self.name, copy.deepcopy(self.children))

    # deberíamos memorizarla?
    def size(self):
        """size is the number of nodes that lives under the root."""
        return sum(child.size() for child in self.children) + 1


class LeafNode(Node):
    """LeafNode is a node that store a value.

    A leaf node is a node that store a value. It has no children and thus it is not terminal.
    It only contains a value.

    Attributes:
        value (float): value of the node.
    """

    def __init__(self, value: float):
        """Initializes LeafNode.

        Args:
            value: value to be stored.
         """
        super().__init__()
        self.value = value

    def is_terminal(self) -> bool:
        """is_terminal returns True, as leaf nodes are always terminal."""
        return True

    def __repr__(self) -> str:
        return f'{self.__class__}({self.value!r})'

    def __eq__(self, other: Node):
        if other.is_terminal():
            return self.value == other.value

        return False

    # have to include memo
    def __deepcopy__(self, memo):
        return type(self)(self.value)

    def size(self):
        """size is the number of nodes that lives under the root."""
        return 1


class MarkedNode(BranchNode):
    """A MarkedNode is a BranchNode with a special mark stored.

    A MarkedNode differs from a BranchNode in the fact that it stored an
    id related with its contents. Then, whenever comparing it with another
    marked node, it will ensure before computing the equality that the ids
    are the same, thus greatly reducing the numbers of comparisons done. 
  
    It creates heavier node than a normal BranchNode, so if no comparisons 
    are to be made, it might not be a good fit.
    """

    @staticmethod
    def _mark(node: Node):
        if node.is_terminal():
            return np.uintc(value)
        if node is type(self):
            return node.mark
        return 0

    def __init__(self, name: int, children: List[Node], *, mark: int = None):
        """Initializes BranchNode

        checks that the variable is a non-negative value and assign

        Args:
            name: Name of the variable associated with the node. It should be non-negative.
            children: Each of the nodes associated with each state of variable name.
                It should be non-negative.

        Raises:
            ValueError: Whenever name is negative of children is empty.
        """

        super().__init__(name, children)
        if not mark:
            self.mark = sum(3**i * MarkedNode._mark(child)
                            for i, child in enumerate(children))

    def __eq__(self, other: Node):
        if other is type(self):
            if not self.mark == other.mark:
                return False

        return super().__eq__(other)

    # have to include memo
    def __deepcopy__(self, memo):
        t = type(self)(self.name, copy.deepcopy(self.children), self.mark)
