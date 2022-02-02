"""Tree-based potentials structure in Python.

This module contains the different implementations of potentials that are to be
implemented. It is based on the implementations done in pgmpy.DiscreteFactor.

Initially it will contains the class Tree, that is a wrapper class for a tree root node.

Typical usage example:

  # data is read from a numpy ndarray object
  data = np.array(get_data())
  variables = ['A', 'B', 'C']
  cardinality= {'A':4, 'B':3, 'C':3}

  tree = Tree.from_array(data, variables, cardinality)

  # We can perform most of the operations over tree. For example:
  tree.prune()
  tree.access({'C':1, 'B':2})
 """
from __future__ import annotations

import abc
import collections
import copy
import dataclasses
import itertools
import math
import logging
import sys

from pyutai import nodes
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

IndexType = Dict[str, int]
"""IndexType is the type accepted by access method to retrieve a variable."""

DataAccessor = Callable[[IndexType], int]
"""DataAccesor is the type of the functions that, from a state configuration of
a set of variables, retrieves the associated value"""

VarSelector = Callable[[Dict[str, int]], int]
"""VarSelector is the type of the functions that select which variable to explore
next at the tree creation. It receives a  Dictionary with the variables that have
 already been assigned and select the next variable to explore.

As dict are optimized for lookup, an str to int dict will be passed, with every 
string representing a number.
"""


@dataclasses.dataclass
class Element:
    """
    An element is a pair of a state of the variables of a potential, and the 
    """
    state: Tuple[int]
    value: float


@dataclasses.dataclass
class Tree:
    """Tree stores the value of a  potential over a tree. 

    It encapsulates a root node from pyutai.nodes and perfomr most operations
    required between trees.

    Attributes:
        root: root node of the tree.
        variables: Set of variables name to be used in the tree.
        cardinality: number of states of each variable. Assumed to be a global
            variable shared with other trees, maybe in the same network.
    """
    root: nodes.Node
    variables: Set[str]
    cardinalities: Dict[int, int] = dataclasses.field(default_factory=tuple)

    @classmethod
    def _from_callable(cls,
                       data: DataAccessor,
                       variables: List[str],
                       cardinalities: Dict[str, int],
                       assigned_vars: Dict[str, int],
                       *,
                       next_var: VarSelector = None) -> nodes.Node:
        """Auxiliar function for tail recursion in from_array method.

        As it uses tail recursion, it may generate stack overflow for big trees.
        data_shape is changed from Tuple to list to avoid copying it multiple times,
        due to the immutability of tuples."""

        if next_var is None:
            next_var = lambda assigned_vars: variables[len(assigned_vars)]

        # If every variable is already selected
        if len(variables) == len(assigned_vars):
            return nodes.LeafNode(data(assigned_vars))

        else:
            var = next_var(assigned_vars)
            cardinality = cardinalities[var]

            # propagate the creation
            children = []
            for i in range(cardinality):
                new_vars = dict(assigned_vars, **{var: i})
                child = Tree._from_callable(data, variables, cardinalities,
                                            new_vars)
                children.append(child)

            return nodes.BranchNode(var, children)

    @classmethod
    def from_callable(cls,
                      data: DataAccessor,
                      variables: List[str],
                      cardinalities: Dict[str, int],
                      *,
                      next_var: Callable[[Dict[int, int]], int] = None):
        """Create a Tree from a callable.

        Read a potential from a given callable, and store it in a value tree.
        It does not returns a prune tree. Consider pruning the tree after creation.
        Variables are named 0,...,len(data)-1, and as such will be referred for
        operations like restricting and accessing.

        Args:
            data: callable that receives a variable configuration and returns the
                 corresponding value.
            data_shape: the elements of the shape tuple give the lengths of
                  the corresponding tree variables.
            next_var: callable that receives the already assigned variables and select
                  the next variables to be assigned. If not specified, variables are
                  are assigned in increasing order.
        """
        return cls(root=Tree._from_callable(data=data,
                                            variables=variables,
                                            cardinalities=cardinalities,
                                            assigned_vars={},
                                            next_var=next_var),
                   variables=set(variables),
                   cardinalities=cardinalities)

    @classmethod
    def from_array(cls,
                   data: np.ndarray,
                   variables: List[str],
                   cardinalities: Dict[str, int],
                   *,
                   next_var: Callable[[Dict[int, int]], int] = None):
        """Create a Tree from a numpy.ndarray.

        Read a potential from a given np.ndarray, and store it in a value tree.
        It does not returns a prune tree. Consider pruning the tree after creation.
        Variables are named 0,...,len(data)-1, and as such will be referred for
        operations like restricting and accessing.

        Args:
            data: table-valued potential.

        Raises:
            ValueError: is provided with an empty table, or if Array and cardinalities
                does not match.

        TODO: add error when cardinality len dos not match data.shape len.
        """
        if data.size == 0:
            raise ValueError('Array should be non-empty')
        for index, var in enumerate(variables):
            if data.shape[index] != cardinalities[var]:
                raise ValueError(
                    'Array shape must match cardinalities; In variable ' +
                    f'{var}: received cardinality {cardinalities[var]},' +
                    f'in array {data.shape[index]}.')

        # adapter from IndexType to numpy Index-Tuple
        data_accessor = lambda x: data.item(tuple(x[var] for var in variables))

        return cls.from_callable(
            data_accessor,
            variables,  # has to be a list
            cardinalities,
            next_var=next_var)

    def __iter__(self):
        """Returns an iterator over the values of the Tree.

        Returns:
            Element: with the configuration of states variables and the associated value.
        """
        for states in itertools.product(
                *[range(self.cardinalities[var]) for var in self.variables]):
            indexes = {
                variable: state
                for variable, state in zip(self.variables, states)
            }
            yield Element(indexes, self.access(indexes))

    def __deepcopy__(self, memo):
        """Deepcopy the provided tree. Beaware that cardinalities is assumed to be shared
        globaly within all trees, so it is not copied."""
        return type(self)(root=copy.deepcopy(self.root),
                          variables=self.variables.copy(),
                          cardinalities=self.cardinalities)

    def size(self):
        """Number of nodes contained in the tree. May varies upon pruning."""
        return self.root.size()

    # Consider that you are using tail recursion so it might overload with big files.
    @classmethod
    def _prune(cls, node: nodes.Node):
        if node.is_terminal():
            return node
        else:
            node.children = [cls._prune(node) for node in node.children]

            if all(child.is_terminal() for child in node.children):
                if len(set(child.value for child in node.children)) == 1:
                    return nodes.LeafNode(node.children[0].value)

            return node

    def prune(self):
        """"Reduces the size of the tree by erasing duplicated branches.

        Tail-recursion function that consider if two children are equal, in
        which case it unifies them under the same reference."""
        self.root = type(self)._prune(self.root)

    @staticmethod
    def _access(node: nodes.Node, states: IndexType) -> float:
        """Helper method for access function."""
        while not node.is_terminal():
            var = node.name
            state = states[var]
            node = node.children[state]

        return node.value

    def access(self, states: IndexType) -> float:
        """Returns a value for a given series of states.

        Returns a value for a given state configuration. It receives either a
        list or a tuple of states, with as many states as variables.

        In the case of retrained variables, via restraint method, those values
        are ignored unless ignore_restraints is set to True. If no variable is
        restrained, every value is considered.

        In some case, specially in pruned tree, it is not necessary to state
        the value of every variable to get the value. Nonetheless, for good
        measure, a complete set of states is required.

        Args:
            states: list or tuple of states for each variable.

        Raises:
            ValueError: if incorrect states are provided. In particular if:
                * An state is out of bound for its variable.
        """

        for var in states:
            if (state := states[var]) >= (bound :=
                                          self.cardinalities[var]) or state < 0:
                raise ValueError(f'State for variable {var} is out of bound;' +
                                 f'expected state in interval:[0,{bound}),' +
                                 f'received: {state}.')

        return type(self)._access(self.root, states)

    @classmethod
    def _restrain(cls, node: nodes.Node, restrictions: IntexType):
        #TODO: separate inplace and copy style

        if node.is_terminal():
            return nodes.LeafNode(node.value)

        else:
            if node.name in restrictions:
                state = restrictions[node.name]
                return cls._restrain(node.children[state], restrictions)
            else:
                children = [
                    cls._restrain(child, restrictions)
                    for child in node.children
                ]
                return nodes.BranchNode(node.name, children)

    def restrain(self, restrictions: IndexType, *, inplace: bool = False):
        """restraint variable to a particular state.

        Restraint a variable to a particular state. See access for more
        information.

        Args:
            restrictions: Mapping from variables to state to which 
                restrict them.
            inplace: If true, modifications will be made on the provided
                tree. Otherwise, the operation will return a modified new
                tree.

        Returns: Modified tree.
        Raises:
            ValueError: if either variable or state are out of bound.
        """
        restricted_root = type(self)._restrain(self.root, restrictions)

        if inplace:
            self.root = restricted_root
            self.variables.difference_update(restrictions.keys())
            return self

        else:
            # Fastest way to copy: https://stackoverflow.com/a/26875847.
            variables = self.variables.difference(restrictions.keys())

            return type(self)(root=restricted_root,
                              variables=variables,
                              cardinalities=self.cardinalities)

    @classmethod
    def _product(cls, node, other):

        if node.is_terminal() and other.is_terminal():
            return node * other

        elif node.is_terminal() and not other.is_terminal():
            # Special cases for fast product
            if node.value == 0:
                return nodes.LeafNode(0)
            elif node.value == 1:
                return copy.deepcopy(other)

            # General case - interchange order
            return cls._product(other, node)

        else:  # Whenever node is not terminal
            var = node.name
            children = [
                cls._product(child, cls._restrain(other, {var: i}))
                for i, child in enumerate(node.children)
            ]

            return nodes.BranchNode(var, children)

    def product(self, other: Tree, *, inplace: bool = False):
        """Combines two trees."""

        root = type(self)._product(self.root, other.root)
        variables = self.variables.union(other.variables)
        tree = type(self)(root=root,
                          variables=variables,
                          cardinalities=self.cardinalities)
        if inplace:
            self = tree

        return tree

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return other.product(self, inplace=False)

    def __imul__(self, other):
        return self.product(other, inplace=True)

    @classmethod
    def _sum(cls, node, other):
        """ TODO: make special method for faster sum reduction"""

        if node.is_terminal() and other.is_terminal():
            return node + other

        elif node.is_terminal() and not other.is_terminal():
            # Special cases for fast sum
            if node.value == 0:
                return copy.deepcopy(other)

            # General case - interchange order
            return cls._sum(other, node)

        else:  # Whenever node is not terminal
            var = node.name
            children = [
                cls._sum(child, cls._restrain(other, {var: i}))
                for i, child in enumerate(node.children)
            ]

            return nodes.BranchNode(var, children)

    def sum(self, other: Tree, *, inplace: bool = False):
        """sum two trees"""

        if self.variables != other.variables:
            raise ValueError("Trees needs to have the same variables to be sum")

        root = type(self)._sum(self.root, other.root)
        tree = type(self)(root=root,
                          variables=self.variables.copy(),
                          cardinalities=self.cardinalities)
        if inplace:
            self = tree

        return tree

    def __add__(self, other):
        return self.sum(other, inplace=False)

    def __radd__(self, other):
        return other.sum(self, inplace=False)

    def __iadd__(self, other):
        return self.sum(other, inplace=True)

    @classmethod
    def _marginalize(cls, node: nodes.Node, variable: str):
        if node.is_terminal():
            return nodes.LeafNode(node.value * self.cardinalities[variable])

        else:
            if node.name == variable:
                return reduce(lambda a, b: a._sum(b), node.children)
            else:
                children = [
                    cls._marginalize(child, variable) for child in node.children
                ]
                return nodes.BranchNode(node.name, children)

    def marginalize(self, variable: int, *, inplace: bool = False):
        """Marginalize a variable"""
        root = type(self)._marginalize(self.root, other.root)
        tree = type(self)(root=root,
                          variables=variables,
                          cardinalities=self.cardinalities)
        if inplace:
            self = tree

        return tree

    def SQEuclideanDistance(self, other: Tree) -> float:
        return sum((a.value - b.value)**2 for a, b in zip(self, other))

    def KullbackDistance(self, other: Tree):
        return sum((
            a.value * (np.log(a.value - b.value)) for a, b in zip(self, other)))
