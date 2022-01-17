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
import logging
import sys


from pyutai import nodes
from typing import Callable, Dict, Iterable, List, Tuple


import numpy as np

@dataclasses.dataclass
class Element:
    """
    An element is a pair of a state of the variables of a potential, and the 
    """
    states  : Tuple[int]
    value : float


# Union types are only allowed from python 3.10 onwards [1].
# Typing analisys for access will only be done for List[int]
# for eariler versions.
#
# [1] https://www.python.org/dev/peps/pep-0604/
if sys.version_info.major >= 3 and sys.version_info.major >= 10:
    IndexType = List[int] | Tuple[int]
    """IndexType is the type accepted by access method to retrieve a variable."""
else:
    IndexType = Tuple[int]

DataAccessor = Callable[[Tuple[int]], int]
"""DataAccesor is the type of the functions that, from a state configuration of
a set of variables, retrieves the associated value"""

VarSelector = Callable[[Dict[str, int]], int]
"""VarSelector is the type of the functions that select which variable to explore
next at the tree creation. It receives a  Dictionary with the variables that have
 already been assigned and select the next variable to explore.

As dict are optimized for lookup, an str to int dict will be passed, with every 
string representing a number.
"""



def _tuple_from_dict(data : Dict[str, int]):
    """Helper function to create a tuple from a dictionary of 
    assigned vars. If the dictionary can not be converted to a 
    tuple it will raise IndexError."""
    return tuple([data[str(i)] for i, _ in enumerate(data.keys)])
        

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
    cardinality: List[int] = dataclasses.field(default_factory=list)
    restraints: Dict[int, int] = dataclasses.field(default_factory=dict,
                                                   init=False)

    @classmethod
    def _from_callable(cls, data : DataAccessor, data_shape : List[int],
                       assigned_vars: Dict[int, int], *,
                       next_var : VarSelector = None) -> nodes.Node:
        """Auxiliar function for tail recursion in from_array method.

        As it uses tail recursion, it may generate stack overflow for big trees.
        data_shape is changed from Tuple to list to avoid copying it multiple times,
        due to the immutability of tuples."""
        if next_var is None:
            next_var = len
        
        # If every variable is already selected
        if len(data_shape) == len(assigned_vars):
            return LeafNode(data(_tuple_from_dict(assigned_vars)))

        else:
            var = next_var(assigned_vars)
            cardinality = data_shape[var]
            # produced dict have str keys
            children = [
                Tree._from_callable(data,  data_shape, dict(assigned_vars, var=i))
                for i in range(cardinality)
            ]
            return BranchNode(var, children)

    @classmethod
    def from_callable(cls, data : DataAccessor, data_shape : Tuple[int], *, next_var : Callable[[Dict[int, int]], int] = None):
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
        return cls(root=Tree._from_callable(data=data, data_shape = data_shape, assigned_vars={}), data_shape=list(data_shape))
        
    @classmethod
    def from_array(cls, data: np.ndarray):
        """Create a Tree from a numpy.ndarray.

        Read a potential from a given np.ndarray, and store it in a value tree.
        It does not returns a prune tree. Consider pruning the tree after creation.
        Variables are named 0,...,len(data)-1, and as such will be referred for
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
    @staticmethod
    def _value_prune(node:nodes.Node):
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

        Tail-recursion function that consider if two children are equal, in
        which case it unifies them under the same reference."""
        self.root = Tree._value_prune(self.root)

    @staticmethod
    def _access(node:nodes.Node, states: List[int],
                restraints: Dict[int, int]) -> float:

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
        """restraint variable to a particular state.

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
            raise ValueError(f'State for variable {variable} is ' +
                             f'out of bound; expected state in ' +
                             f'interval:[0,{bound}),received: {state}.')

        self.restraints[variable] = state

    def unrestrain(self, variable: int):
        """unrestrain variable.

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
        return sum((a.value * (np.log(a.value - b.value))
                    for a,b in zip(self, other)) )


    @staticmethod
    def _marginalize(node, variable) -> node:
        pass
                   
    def marginalize(self, variable: int, *, inplace : bool = False):
        pass

    def sum(self, other: Tree):
        pass

    def product(self, other: Tree):
        pass
