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
  tree.access([state_configuration()])
 """

import abc
import collections
import copy
import dataclasses
import logging

from typing import Dict, Iterable, List, Tuple
from pyutai import IndexType

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

        checks that the variable is a non-negative value and assign

        Args:
            name: Name of the variable associated with the node. It should be non-negative.
            children: Each of the nodes associated with each state of variable name.
                It should be non-negative.
        
        Raises:
            ValueError: Whenever name is negative of children is empty.
        """
        super().__init__()

        if name < 0:
            raise ValueError(f'Name must be non-negative, got: {name}')
        if not children:
            raise ValueError(
                f'Children must be a non-empty list, got: {children}')

        self.name = name
        self.children = children

    def is_terminal(self) -> bool:
        """is_terminal returns False, as branch nodes are never terminal."""
        return False

    def __repr__(self) -> str:
        return f'{self.__class__!r}({self.name!r}, {self.children!r})'


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
        return f'{self.__class__!r}({self.value!r})'


@dataclasses.dataclass
class Tree:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. If ``napoleon_attr_annotations``
    is True, types can be specified in the class body using ``PEP 526``
    annotations.

    Attributes:
        root: root node of the tree.
        cardinality: number of states of each variable.
        restraints: restrained variables in the tree.

    """
    root: Node
    cardinality: List[int] = dataclasses.field(default_factory=list)

    restraints: Dict[int, int] = dataclasses.field(
        default_factory=collections.defaultdict, init=False)

    @classmethod
    def _from_array(cls, data: np.ndarray, assigned_vars: List[int]) -> Node:
        """Auxiliar function for tail recursion in from_array method.

        As it uses tail recursion, it may generate stack overflow for big trees."""
        var = len(assigned_vars)  # Next variable to be assigned

        # If every variable is already selected
        if len(data.shape) == var:
            return LeafNode(data[tuple(assigned_vars)])

        else:
            cardinality = data.shape[var]
            children = [
                Tree._from_array(data, assigned_vars + [i])
                for i in range(cardinality)
            ]
            return BranchNode(var, children)

    @classmethod
    def from_array(cls, data: np.ndarray) -> Node:
        """Create a Tree from a numpy.ndarray.

        Read a potential from a given np.ndarray, and store it in a value tree.
        It does not returns a prune tree. Consider pruning the tree after creation.
        Variables are named 0,...,len(data)-1, and as such will be refered for 
        operations like restricting and accessing.

        Args:
            data: table-valued potential.

        Raises:
            ValueError: is provided with an empty table.
        """
        if data.size == 0:
            raise ValueError('Array should be non-empty')

        return cls(root=Tree._from_array(data, []), cardinality=data.shape)

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
    def access(self,
               states: IndexType,
               *,
               ignore_restraints: bool = False) -> float:
        """Returns a value for a given series of states.

        Returns a value for a given state configuration. It receives either a
        list or a tuple of states, with as many states as variables. 
        
        In the case of retrained varaibles, via restraint method, those values
        are ignored unless ignore_restraints is set to True. If no variable is
        restrained, every value is considered.

        In some case, specially in pruned tree, it is not necessary to state 
        the value of every variable to get the value. Nonetheless, for good
        measure, a complete set of states is required.

        Args:
            states: list or tuple of states for each variable.
            ignore_restraints: if set to true, restraints are ignored.
        
        Raises:
            ValueError: if incorrect states are provided. In particular if:
                * Incorrect number of state are provided.
                * An state is out of bound for its variable.
        """

        if len(states) != (n_variables := len(self.cardinality)):
            raise ValueError(f'Incorrect number of variables; expected: ' +
                             f'{n_variables}, received: {len(states)}.')

        for var, (state, bound) in enumerate(zip(states, self.cardinality)):
            if state >= bound or state < 0:
                raise ValueError(f'State for variable {var} is out of bound;' +
                                 f'expected state in interval:[0,{bound}),' +
                                 f'received: {state}.')

        if self.restraints and not ignore_restraints:
            logging.warning(f'variables {list(self.restraints.keys())}' +
                            f'will be ignored as they are restrained.')
            return Tree._access(self.root, states, self.restraints)
        else:
            return Tree._access(self.root, states, {})

    def restrain(self, variable: int, state: int):
        """retraint variable to a particular state.

        Restraint a variable to a particular state. See access for more 
        information.

        Args:
            variable: variable to be restrained. 
            state: state to restrain the variable with.
        
        Raises:
            ValueError: if either variable or state are out of bound.
        """
        if variable >= len(self.cardinality) or variable < 0:
            raise ValueError(f'Invalid value {variable} for variable.')

        if state >= (bound := self.cardinality[variable]):
            raise ValueError(f'State for variable {varriable} is ' +
                             f'out of bound; expected state in ' +
                             f'interval:[0,{bound}),received: {state}.')

        self.restraints[variable] = state

    def unrestrain(self, variable: int):
        """unretraint variable.

        Args:
            variable: variable to be unrestrained. 
        
        Raises:
            ValueError: if variable is out of bound.
        """
        if variable >= len(self.cardinality) or variable < 0:
            raise ValueError(f'Invalid value {variable} for variable.')

        self.restraints.pop(variable, None)
