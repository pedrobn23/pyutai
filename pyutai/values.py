"""Tree-based potentials structure in Python.

This module contains the different implementations of potentials that are to be
implemented. It is based on the implementations done in pgmpy.DiscreteFactor.

Initially it will contains the class Tree, that is a wrapper class for a tree root node.

Typical usage example:

  # data is read from a numpy ndarray object
  data = np.array(get_data())
  tree = Tree.from_array(data)

  # We can perform most of the operations over tree. For example:
  tree.prune()
  tree.access([state_configuration()])
 """
from __future__ import annotations

import abc
import collections
import copy
import dataclasses
import itertools
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

VarSelector = Callable[[IndexType], str]
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
        cardinality: number of states of each variable.
        restraints: restrained variables in the tree.
    """
    root: nodes.Node
    variables: List[str]
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
            variables: list with the name of the variables used by the tree.
            cardinalities: map from the name of the variables to its cardinalities. It
                may have 
            next_var: callable that receives a dictionary that maps the already assigned
                  variables to its value and select the next variables to be assigned.
                  If not specified, variables are are assigned in order.
        """
        return cls(root=Tree._from_callable(data=data,
                                            variables=variables,
                                            cardinalities=cardinalities,
                                            assigned_vars={},
                                            next_var=next_var),
                   variables=variables,
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

        return cls.from_callable(data_accessor,
                                 variables,
                                 cardinalities,
                                 next_var=next_var)

    def __iter__(self):
        """Returns an iterator over the values of the Tree.

        Returns:
            Element: with the configuration of states variables and the associated value.
        """
        print(self.variables)
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
                          variables=self.variables[:],
                          cardinalities=self.cardinalities)

    # Consider that you are using tail recursion so it might overload with big files.
    @staticmethod
    def _prune(node: nodes.Node):
        if node.is_terminal():
            return node
        else:
            node.children = [Tree._prune(node) for node in node.children]

            if all(child.is_terminal() for child in node.children):
                if len(set(child.value for child in node.children)) == 1:
                    return nodes.LeafNode(node.children[0].value)

            return node

    def prune(self):
        """"Reduces the size of the tree by erasing duplicated branches.

        Tail-recursion function that consider if two children are equal, in
        which case it unifies them under the same reference."""
        self.root = Tree._prune(self.root)

    @staticmethod
    def _access(node: nodes.Node, states: IndexType) -> float:
        """Helper method for access function."""
        while not node.is_terminal():
            var = node.name
            state = states[var]
            node = node.children[state]

        return node.value

    def access(self,
               states: IndexType,
               *,
               ignore_restraints: bool = False) -> float:
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
            ignore_restraints: if set to true, restraints are ignored.

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

        return Tree._access(self.root, states)

    @classmethod
    def _restrain(cls, node: nodes.Node, restrictons : IntexType, accumulated_card):
        #TODO: separate inplace and copy style
        
        if node.is_terminal():
            return copy.deepcopy(node)*accumulated_card

        else:
            if node.name in states:
                state = restrictions[node.name]
                return cls._restrain(node.children[state], accumulated_card)
            else:
                children = [
                    cls._restrain(child, variable, state)
                    for child in node.children
                ]
                return nodes.BranchNode(node.name, children)
                

    def restrain(self, restrictions : IndexType, *, inplace: bool = False):
        """restraint variable to a particular state.

        Restraint a variable to a particular state. See access for more
        information.

        Args:
            variable: variable to be restrained.
            state: state to restrain the variable with.

        Raises:
            ValueError: if either variable or state are out of bound.
        """

        accumulated_cardinality = math.prod((self.cardinalities[var] for var in restrictions))
        restricted_root = Tree._restrain(self.root, restrictions, accumulated_cardinality)

        if inplace:
            self.root = restricted_root
            self.variables.remove(variable)
            return self
        
        else:
            # Fastest way to copy: https://stackoverflow.com/a/26875847.
            restricted_variables = self.variables[:]
            restricted_variables.remove(variable)

            return type(self)(root=restricted_root, variables = restricted_variable, cardinalities = cardinalities)
            

    def product(self, other: Tree):
        pass

        
    def _marginalize(self, node, variable, assigned_vars) -> node:
        pass
    
    def marginalize(self, variable: int, *, inplace: bool = False):
        pass

    def sum(self, other: Tree):
        pass


    def size(self):
        return self.root.size()

    def SQEuclideanDistance(self, other: Tree) -> float:
        return sum((a.value - b.value)**2 for a, b in zip(self, other))

    def KullbackDistance(self, other: Tree):
        return sum((
            a.value * (np.log(a.value - b.value)) for a, b in zip(self, other)))
