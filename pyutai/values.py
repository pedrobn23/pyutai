"""Tree-based potentials structure in Python.

This module contains the different implementations of potentials that are to be
implemented. It is based on the implementations done in pgmpy.DiscreteFactor.

Initially it will contains the class Tree, that is a wrapper class for a tree root node. It performs most of the operations.

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

@dataclasses.dataclass
class Element:
    states  : Tuple[int]
    value : float

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

    restraints: Dict[int, int] = dataclasses.field(default_factory=dict,
                                                   init=False)


   @classmethod
    def _from_callable(cls, data : callable, data_shape : List[int], assigned_vars: List[int]) -> Node:
        """Auxiliar function for tail recursion in from_array method.

        As it uses tail recursion, it may generate stack overflow for big trees.
        data_shape is changed from Tuple to list to avoid copying it multiple times,
        due to the inmutability of tuples. """
        var = len(assigned_vars)  # Next variable to be assigned

        # If every variable is already selected
        if len(data_shape) == var:
            return LeafNode(data(tuple(assigned_vars)))

        else:
            cardinality = data.shape[var]
            children = [
                Tree._from_array(data, assigned_vars + [i])
                for i in range(cardinality)
            ]
            return BranchNode(var, children)

    @classmethod
    def from_callable(cls, data : callable, data_shape : Tuple[int]) -> Tree:
        """Create a Tree from a callable.

        Read a potential from a given callable, and store it in a value tree.
        It does not returns a prune tree. Consider pruning the tree after creation.
        Variables are named 0,...,len(data)-1, and as such will be refered for
        operations like restricting and accessing.

        Args:
            data: callable that receibes a variable configuration and returns the
                 corresponing value.
            data_shape: The elements of the shape tuple give the lengths of
                  the corresponding tree variables.
        """
        return cls(root=Tree._from_callable(data, []), data_shape=list(data_shape))
        
    @classmethod
    def from_array(cls, data: np.ndarray) -> Tree:
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

        return cls.from_callable(data.item, data.shape)

    
    # Consider that you are using tail recursion so it might overload with big files.
    # Suggestion: change it later.
    # Should I do a hash-base module pruning?
    @staticmethod
    def _value_prune(node: Node):
        if node.is_terminal():
            return node
        else:
            node.children = [Tree._prune(node) for node in node.children]

            if all(child.is_terminal() for child in node.children):
                if len(set(child.value for child in node.children)) == 1:
                    return LeafNode(node.children[0].value)

            return node        
    
    def prune(self):
        """"Reduces the size of the tree by erasing duplicated branches.

        Tail-recursion function that consider if two childs are equal, in
        which case it unifies them under the same reference."""
        self.root = Tree._value_prune(self.root)

    @staticmethod
    def _access(node: Node, states: Iterable,
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
            logging.warning(f'variables {list(self.restraints.keys())} ' +
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


    def __iter__(self):
        """Returns an iterator over the values of the Tree.

        Returns:
            Element: with the configuration of states variables and the associated value.
        """
        for var in itertools.product(*[range(var) for var in self.cardinality]):
            raise Element(var, self.access(var))

    def size(self):
        return self.root.size()

    
    def SQEuclideanDistance(self, other : Tree) -> float:
        return sum((a.value - b.value)**2 for a,b in zip(self, other))        
    
    def KullbackDistance(self, other : Tree):
        return sum((a.value * (np.log(a.value - b.vlaue)) for a,b in zip(self, other))

    
    # mismo problema, podemos modular el access para que haga unas sumas
    # o podemos devolver otro arbol con una variable menos.
    def marginalize(self, variable: int):
        pass

    def sum(self, other: Tree):
        pass

    def product(self, other: Tree):
        pass
