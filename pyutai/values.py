"""Tree-based potentials structure in Python.

This module contains the different implementations of potentials that are to be
implemented. It is based on the implementations done in pgmpy.DiscreteFactor.

Initially it will contains the class Tree, that is a wrapper class for a tree root node.

The tree is used as an undrelying data structures to represent a Potential from a set of
finite random variables {X_1,...,X_n} to |R. Each variables is assumed to have a finite
amount of states, represented as 0,...,n.

An IndexSelection stores the is a mapping from variables to states. Sometimes, it will be
refered as a state_configuration.

Typical usage example:

  # data is read from a numpy ndarray object
  data = np.array(get_data())
  variables = ['A', 'B', 'C']
  cardinality= {'A':4, 'B':3, 'C':3}

  tree = Tree.from_array(data, variables, cardinality)

  # We can perform most of the operations over tree. For example:
  tree.prune()

  tree.access([state_configuration()])
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import itertools
import sys
import warnings

from typing import Callable, Dict, List

import numpy as np

from pyutai import nodes

IndexSelection = Dict[str, int]
"""IndexSelection is the type accepted by access method to retrieve a variable.

It stores a mapping from the name of a variable to its state.
"""

DataAccessor = Callable[[IndexSelection], float]
"""DataAccesor is the type of the functions that from a IndexSelection, returns
the value associated with such state."""

VarSelector = Callable[[Dict[str, int]], int]
"""VarSelector is the type of the functions that select which variable to explore
next at tree creation. It receives a  Dictionary with the variables that have
 already been assigned and select the next variable to explore.

As dict are optimized for lookup, an str to int dict will be passed, with every
string representing a number.
"""


@dataclasses.dataclass
class Element:
    """
    An element is a pair of a state of the variables of a potential, and the
    """
    state: IndexSelection
    value: float


@dataclasses.dataclass
class Tree:
    """Tree stores the value of a  potential over a tree.

    It encapsulates a root node from pyutai.nodes and perfomr most operations
    required between trees.

    Attributes:
        root: root node of the tree.
        variables: set of variable names to be used in the tree.
        cardinality: number of states of each variable. Assumed to be a global
            variable shared with other trees, maybe in the same network.
    """
    root: nodes.Node
    variables: Set[str]
    cardinalities: Dict[str, int]
    pruned : bool = False
    
    
    @classmethod
    def _from_callable(cls,
                       data: DataAccessor,
                       variables: List[str],
                       cardinalities: Dict[str, int],
                       assigned_vars: Dict[str, int],
                       *,
                       selector: VarSelector = None) -> nodes.Node:
        """Auxiliar function for tail recursion in from_callable method.

        As it uses tail recursion, it may generate stack overflow for big trees.
        data_shape is changed from Tuple to list to avoid copying it multiple times,
        due to the immutability of tuples."""

        if selector is None:
            selector = lambda assigned_vars: variables[len(assigned_vars)]

        # If every variable is already selected
        if len(variables) == len(assigned_vars):
            return nodes.LeafNode(data(assigned_vars))

        else:
            next_var = selector(assigned_vars)
            n_children = cardinalities[next_var]

            # Tail recursion propagation
            children = []
            for i in range(n_children):
                new_assigned_vars = dict(assigned_vars)
                new_assigned_vars[next_var] = i
                child = cls._from_callable(data=data,
                                            variables=variables,
                                            cardinalities=cardinalities,
                                            assigned_vars=new_assigned_vars,
                                            selector=selector)
                children.append(child)

            return nodes.BranchNode(next_var, children)

    @classmethod
    def from_callable(cls,
                      data: DataAccessor,
                      variables: List[str],
                      cardinalities: Dict[str, int],
                      *,
                      selector: VarSelector = None):
        """Create a Tree from a callable.

        Read a potential from a given callable, and store it in a value tree.
        It does not returns a prune tree. Consider pruning the tree after creation.
        Variables are named 0,...,len(data)-1, and as such will be referred for
        operations like restricting and accessing.

        Args:
            data: callable that receives a variable configuration and returns the
                 corresponding value.
            variables: list with the name of the variables.
            cardinalities: mapping from the variable names to its number of states.
            selector (optional): Var selector to create each node of the tree. See
                See VarSelector type for more information.
        """
        root = cls._from_callable(data=data,
                                   variables=variables,
                                   cardinalities=cardinalities,
                                   assigned_vars={},
                                   selector=selector)
        return cls(root=root,
                   variables=set(variables),
                   cardinalities=cardinalities,
                   pruned=False)

    @classmethod
    def from_array(cls,
                   data: np.ndarray,
                   variables: List[str],
                   cardinalities: Dict[str, int],
                   *,
                   selector: Callable[[Dict[int, int]], int] = None):
        """Create a Tree from a numpy.ndarray.

        Read a potential from a given np.ndarray, and store it in a value tree.
        It does not returns a prune tree. Consider pruning the tree after creation.
        Variables are named 0,...,len(data)-1, and as such will be referred for
        operations like restricting and accessing.

        Args:
            data: table-valued potential.
            variables: list with the name of the variables.
            cardinalities: mapping from the variable names to its number of states.
            selector (optional): Var selector to create each node of the tree. See
                See VarSelector type for more information.

        Raises:
            ValueError: is provided with an empty table, or if Array and cardinalities
                does not match.
        """
        if data.size == 0:
            raise ValueError('Array should be non-empty')


        if len(data.shape) != len(variables):
            raise ValueError(f'Array shape does not match number of variables' +
                             f'provided.\nArray shape: {data}, ' +
                             f'variables: {variables}.')

        for index, var in enumerate(variables):
            if data.shape[index] != cardinalities[var]:
                raise ValueError(
                    'Array shape must match cardinalities; In variable ' +
                    f'{var}: received cardinality {cardinalities[var]},' +
                    f'in array {data.shape[index]}.')

        # adapter from IndexSelection to numpy Index-Tuple
        data_accessor = lambda x: data.item(tuple(x[var] for var in variables))

        return cls.from_callable(
            data_accessor,
            variables,  # has to be a list
            cardinalities,
            selector=selector)

    def __iter__(self):
        """Returns an iterator over the values of the Tree.

        Iteration is done consistently, that is, always in the same order.

        Returns:
            Element: with the configuration of states variables and the associated value.
        """
        # Variables are ordered to ensure consistent iteration.
        variables = list(self.variables)
        variables.sort()

        shape = tuple(self.cardinalities[var] for var in variables)
        
        for states in np.ndindex(shape):
            indexes = dict(zip(variables, states))
            yield Element(indexes, self.access(indexes))


    def array(self):
        variables = list(self.variables)
        variables.sort()

        shape = tuple(self.cardinalities[var] for var in variables)
        result = np.zeros(shape)
        
        for states in np.ndindex(shape):
            indexes = dict(zip(self.variables, states))            
            result[states] = self.access(indexes)

        return result
        
            
    def __deepcopy__(self, memo):
        """Deepcopy the provided tree. Beaware that cardinalities is assumed to be shared
        globaly within all trees, so it is not copied.

        Returns:
            Tree: deepcopy of the tree.
        """
        return type(self)(root=copy.deepcopy(self.root),
                          variables=self.variables.copy(),
                          cardinalities=self.cardinalities)

    def copy(self):
        """Deepcopy the provided tree. Beaware that cardinalities is assumed to be shared
        globaly within all trees, so it is not copied.

        Returns:
            Tree: deepcopy of the tree.
        """
        return self.__deepcopy__({})
    
    def size(self):
        """Number of nodes contained in the tree. May varies upon pruning.

        Returns:
            int: number of nodes in the tree.
        """
        return self.root.size()

    # Consider that you are using tail recursion so it might overload with big files.
    @classmethod
    def _prune(cls, node: nodes.Node):
        """Auxiliar, tail-recursion function that consider if two children are equal, in
        which case it unifies them under the same reference."""
        if node.is_terminal():
            return node
        else:
            node.children = [cls._prune(node) for node in node.children]

            if all(child.is_terminal() for child in node.children):
                if len(set(child.value for child in node.children)) == 1:
                    return nodes.LeafNode(node.children[0].value)

            return node

    
    def prune(self):
        """"Reduces the size of the tree by erasing duplicated branches."""
        self.root = type(self)._prune(self.root)
        self.pruned = True
        
    @classmethod
    def _expand_node(cls, value : float, rest: set, cardinalities : Dict[str, int]):
        """Auxliar function that reexpand a previously prunned node."""
        if rest:
            var = rest.pop()
            children = [cls._expand_node(value, set(rest), cardinalities) for _ in range(cardinalities[var])]
            return nodes.BranchNode(var, children)

        else:
            return nodes.LeafNode(value)

    def _unprune(self, node: nodes.Node, assigned : set):
        """Auxiliar, tail-recursion function that consider if two children are equal, in
        which case it unifies them under the same reference."""
        if not node.is_terminal():
            node.children = [self._unprune(child, assigned.union({node.name}))
                             for child in node.children]
            return node
        elif assigned != self.variables:
            return type(self)._expand_node(node.value, self.variables.difference(assigned), self.cardinalities)
        else:
            return node

    def unprune(self):
        """"Restore tree structure for easier modifications."""
        self.root = self._unprune(self.root, set())
        self.pruned = False
        
    @staticmethod
    def _access(node: nodes.Node, states: IndexSelection) -> float:
        """Helper method for access function."""
        while not node.is_terminal():
            var = node.name
            state = states[var]
            node = node.children[state]

        return node.access(states)

    def access(self, states: IndexSelection) -> float:
        """Value associated with a given series of states.

        Returns a value for a given state configuration.

        In some case, specially in pruned trees, it is not necessary to state
        the value of every variable to get the value. Nonetheless, for good
        measure, a complete set of states is usually required.

        Args:
            states: Dictionary of states for each variable.

        Raises:
            ValueError: if incorrect states are provided. In particular if:
                * An state is out of bound for its variable.
                * There is not enough information in states to retrieve a value.
        """

        for var in states:
            if (state := states[var]) >= (bound :=
                                          self.cardinalities[var]) or state < 0:
                raise ValueError(f'State for variable {var} is out of bound;' +
                                 f'expected state in interval:[0,{bound}),' +
                                 f'received: {state}.')
        try:
            value = type(self)._access(self.root, states)
        except KeyError as key_error:
            raise ValueError(
                f'State configuration {states} does not have' +
                f'enough information to select one value.') from key_error
        return value

    @staticmethod
    def _set(node: nodes.Node, states: IndexSelection, value : float):
        """Helper method for access function."""
        while not node.is_terminal():
            var = node.name
            state = states[var]
            node = node.children[state]
            
        return node.set(value, states)

    
    def set(self, states: IndexSelection, value : float):
        """Set the value associated with a given series of states.

        Set a value for a given state configuration. 

        A complete set of states is required.

        Note that set operation need to modify the values, so it unprune
        the tree before starting if needed. This can affect efficiency, and
        maybe you want to prune the tree again after any needed modification.

        Args:
            states: Dictionary of states for each variable.

        Raises:
            ValueError: if incorrect states are provided. In particular if:
                * An state is out of bound for its variable.
                * There is not enough information in states to retrieve a value.
        """

        for var in states:
            if (state := states[var]) >= (bound :=
                                          self.cardinalities[var]) or state < 0:
                raise ValueError(f'State for variable {var} is out of bound;' +
                                 f'expected state in interval:[0,{bound}),' +
                                 f'received: {state}.')

        if self.pruned:
            self.unprune()
            
        try:
            type(self)._set(self.root, states, value)
        except KeyError as key_error:
            raise ValueError(
                f'State configuration {states} does not have' +
                f'enough information to select one value.') from key_error
    
    #TODO: separate inplace and copy style
    @classmethod
    def _restrict(cls, node: nodes.Node, restrictions: IntexType):
        """Helper method for restrict method. Uses tail recursion."""
        if node.is_terminal():
            return node.restrict(restrictions)

        else:
            if node.name in restrictions:
                state = restrictions[node.name]
                return cls._restrict(node.children[state], restrictions)
            else:
                children = [
                    cls._restrict(child, restrictions)
                    for child in node.children
                ]
                return nodes.BranchNode(node.name, children)

    def restrict(self, restrictions: IndexSelection, *, inplace: bool = False):
        """Restrict a variable to a particular state.

        See access for more information on states.

        Args:
            restrictions: Mapping from variables to state to which
                restrict them.
            inplace: If true, modifications will be made on the provided
                tree. Otherwise, the operation will return a modified new
                tree.

        Returns: Modified tree.
        """
        restricted_root = type(self)._restrict(self.root, restrictions)

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


    def _update_cardinality(self, other):
        """Helper to make cardinality update in old python versions."""

        if sys.version_info.minor >= 9:
            return self.cardinalities | other.cardinalities
        else:
            result = dict(self.cardinalities)
            for var, value in other.cardinalities.items():
                if var not in result:
                    result[var] = value
            return result
        
    @classmethod
    def _product(cls, node, other):
        """Tail-recursion helper."""
        if node.is_terminal() and other.is_terminal():
            return node * other

        elif node.is_terminal() and not other.is_terminal():
            # Special cases for fast product in LeafNodes
            if isinstance(node, nodes.LeafNode)
                if node.value == 0:
                    return nodes.LeafNode(0)
                elif node.value == 1:
                    return copy.deepcopy(other)

                
            # General case - interchange order
            return cls._product(other, node)

        else:  # Whenever node is not terminal
            var = node.name
            children = [
                cls._product(child, cls._restrict(other, {var: i}))
                for i, child in enumerate(node.children)
            ]

            return nodes.BranchNode(var, children)

    def product(self, other: Tree, *, inplace: bool = False):
        """Combines to trees, so that the resulting tree represents the
        product of the two potentials involved.

        Args:
            other: Tree to combine with.
            inplace: If true, modifications will be made on the provided
                tree. Otherwise, the operation will return a modified new
                tree.

        Returns:
            Tree: Product tree.
        """
        # TODO: incluir producto por entero y flotante.

        
        root = type(self)._product(self.root, other.root)
        variables = self.variables.union(other.variables)
        tree = type(self)(root=root,
                          variables=variables,
                          cardinalities=self._update_cardinality(other))
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
        """Tail-recursion helper."""

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
                cls._sum(child, cls._restrict(other, {var: i}))
                for i, child in enumerate(node.children)
            ]

            return nodes.BranchNode(var, children)

    def sum(self, other: Tree, *, inplace: bool = False):
        """Combines to trees, so that the resulting tree represents the
        sum of the two potentials involved.

        Args:
            other: Tree to combine with.
            inplace: If true, modifications will be made on the provided
                tree. Otherwise, the operation will return a modified new
                tree.

        Returns:
            Tree: sum tree.
        """

        if self.variables != other.variables:
            raise ValueError("Trees needs to have the same variables to be sum")

        root = type(self)._sum(self.root, other.root)
        tree = type(self)(root=root,
                          variables=self.variables.copy(),
                          cardinalities=self._update_cardinality(other))
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
    def _marginalize(cls, node: nodes.Node, variable: str, cardinalities : Dict[str, int]):
        if node.is_terminal():
            return node.marginalize(variable, cardinalities)

        else:
            if node.name == variable:
                return functools.reduce(cls._sum, node.children)
            else:
                children = [
                    cls._marginalize(child, variable, cardinalities) for child in node.children
                ]
                return nodes.BranchNode(node.name, children)

    def marginalize(self, variable: str, *, inplace: bool = False):
        """Delete a variable by marginalizing it out.


        Args:
            variable: name of the variable to marginalize.
            inplace: If true, modifications will be made on the provided
                tree. Otherwise, the operation will return a modified new
                tree.

        Returns:
            Tree: Modified tree.
        """

        root = type(self)._marginalize(self.root, variable, self.cardinalities)

        variables = set(self.variables)
        variables.remove(variable)

        tree = type(self)(root=root,
                          variables=variables,
                          cardinalities=self.cardinalities)
        if inplace:
            self.root = tree.root
            self.variables = tree.variables

        return tree

    def __eq__(self, other):
        warnings.warn('This method perform value-wise comprasion, this being a'+
                      'potentially highly inefficient method [O(n^2)].')

        if not isinstance(other, type(self)):
            return False

        if other.variables != self.variables:
            return False

        for variable in self.variables:
            if self.cardinalities[variable] != other.cardinalities[variable]:
                return False

        for element in self:
            if other.access(element.state) != element.value:
                return False

        return True

        

class TableTree(Tree):
    @staticmethod
    def _restrict_table(data: np.ndarray, variables : List[str], assigned: Dict[str, int]):
        """Restrict variables to provided values"""

        filter_ = tuple(
            assigned[var] if var in assigned else slice(None)
            for var in variables)

        new_data = data[filter_]

        new_variables = [var for var in variables if var not in assigned]

        return new_data, new_variables
    
    @classmethod
    def _from_array(cls,
                    data: np.ndarray,
                    variables: List[str],
                    cardinalities: Dict[str, int],
                    assigned_vars: Dict[str, int],
                    *,
                    table_size : int = 5,
                    selector: VarSelector = None) -> nodes.Node:
        """Auxiliar function for tail recursion in from_callable method.

        As it uses tail recursion, it may generate stack overflow for big trees.
        data_shape is changed from Tuple to list to avoid copying it multiple times,
        due to the immutability of tuples."""

        if selector is None:
            selector = lambda assigned_vars: variables[len(assigned_vars)]

        # If every variable is already selected
        if len(variables) <= len(assigned_vars)+table_size:
            node_values, node_variables = cls._restrict_table(data, variables, assigned_vars) 
            return nodes.TableNode(node_values, node_variables)

        else:
            next_var = selector(assigned_vars)
            n_children = cardinalities[next_var]

            # Tail recursion propagation
            children = []
            for i in range(n_children):
                new_assigned_vars = dict(assigned_vars)
                new_assigned_vars[next_var] = i
                child = cls._from_array(data=data,
                                        variables=variables,
                                        cardinalities=cardinalities,
                                        assigned_vars=new_assigned_vars,
                                        table_size=table_size,
                                        selector=selector)
                children.append(child)

            return nodes.BranchNode(next_var, children)

    @classmethod
    def from_array(cls,
                   data: np.ndarray,
                   variables: List[str],
                   *,
                   table_size : int = 1,
                   selector: Callable[[Dict[int, int]], int] = None):
        """Create a Tree from a numpy.ndarray.

        Read a potential from a given np.ndarray, and store it in a value tree.
        It does not returns a prune tree. Consider pruning the tree after creation.
        Variables are named 0,...,len(data)-1, and as such will be referred for
        operations like restricting and accessing.

        Args:
            data: table-valued potential.
            variables: list with the name of the variables.
            cardinalities: mapping from the variable names to its number of states.
            selector (optional): Var selector to create each node of the tree. See
                See VarSelector type for more information.

        Raises:
            ValueError: is provided with an empty table, or data dimension does
                not match variable list.
        """
        if data.size == 0:
            raise ValueError('Array should be non-empty')

        if len(data.shape) != len(variables):
            raise ValueError(f'Array shape does not match number of variables' +
                             f'provided.\nArray shape: {data}, ' +
                             f'variables: {variables}.')

        cardinalities = dict(zip(variables, data.shape))

        root = cls._from_array(data=data,
                               variables=variables,
                               cardinalities=cardinalities,
                               assigned_vars={},
                               table_size=table_size, 
                               selector=selector)
        return cls(root=root,
                   variables=set(variables),
                   cardinalities=cardinalities)

    # Consider that you are using tail recursion so it might overload with big files.
    @classmethod
    def _prune(cls, node: nodes.Node):
        """Auxiliar, tail-recursion function that consider if two children are equal, in
        which case it unifies them under the same reference."""
        if node.is_terminal():
            return node
        else:
            node.children = [cls._prune(node) for node in node.children]

            if not node.children:
                raise ValueError('Children vector should not be empty.')
                
            if all(child.is_terminal() for child in node.children):
                for child in node.children:
                    if node.children[0] != child:
                        return node
                    # In this case every child is equal to children[0] 
                return node.children[0].copy()

            return node
        
    def prune(self):
        """"Reduces the size of the tree by erasing duplicated branches."""
        self.root = self._prune(self.root)
        self.pruned = True
        
    @classmethod
    def _expand_node(cls, node : nodes.TableNode, rest: set, cardinalities : Dict[str, int]):
        """Auxliar function that reexpand a previously prunned node."""
        if rest:
            var = rest.pop()
            children = [cls._expand_node(node, set(rest), cardinalities) for _ in range(cardinalities[var])]
            return nodes.BranchNode(var, children)

        else:
            return node.copy()

        
    def _unprune(self, node: nodes.Node, assigned : set):
        """Auxiliar, tail-recursion function that consider if two children are equal, in
        which case it unifies them under the same reference."""
        if not node.is_terminal():
            node.children = [self._unprune(child, assigned.union({node.name}))
                             for child in node.children]
            return node

        elif len(assigned) + len(node.variables) != self.variables:
            variables_left = self.variables.difference(set(node.variables).union(assigned))            
            return type(self)._expand_node(node, variables_left, self.cardinalities)

        else:
            return node
        
        
    def unprune(self):
        """"Restore tree structure for easier modifications."""
        self.root = self._unprune(self.root, set())
        self.pruned = False
    
def sq_euclidean_distance(tree: Tree, other: Tree) -> float:
    """Square Euclidean distance between two trees.

    Provided trees are assumed to have the same variables.

    Returns:
        float: distance.
    Raises:
        ValueError: If provided tree those not share variables.
    """
    if tree.variables != other.variables:
        raise ValueError('Trees must share variables. Got:' +
                         f'{tree.variables}, {other.variables}.')
    return sum((a.value - b.value)**2 for a, b in zip(tree, other))


def kullback_distance(tree: Tree, other: Tree):
    """KullbackLeibler distance between two trees.

    Provided trees are assumed to have the same variables.

    Returns:
        float: distance.
    """
    if tree.variables != other.variables:
        raise ValueError('Trees must share variables. Got:' +
                         f'{tree.variables}, {other.variables}.')

    return sum(a.value * (np.log(a.value - b.value)) for a, b in zip(tree, other))
