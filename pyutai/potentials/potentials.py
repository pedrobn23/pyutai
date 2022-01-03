"""Tree-based potentials structure in Python.

This module contains the different implementations of potentials that are to be implemented. It is based on the impelmentations done in pgmpy.DiscreteFactor. Initially it will contains the classes:
- Tree: Tree 
- TreePotential: A potential based on a tree structure.
- TreeCPD: A conditional probability distribution based on TreePotential. Analogous to pgmpy.TabularCPD.

  Typical usage example:

  TODO
"""
import attr
import pandas as pd
import numpy as np
import numpy.typing as npt


@attr.s
class Node(ABC):

    def __init__(self, name: int):
        if name < 0:
            raise ValueError(f'Name must be non-negative, got: {name}')
        self.name = name

    @abstractmethod
    def is_terminal(self) -> bool:
        pass


class BranchNode(Node):

    def __init__(self, name: int, children: list[Node]):
        super().__init__(name)
        self.children = children

    def is_terminal(self) -> bool:
        return False

        
    


class LeafNode(Node):

    def __init__(self, name: int, values: list[float]):
        super().__init__(name)
        self.values = values

    def is_terminal(self) -> bool:
        return True


class Tree(pgpmpy.StateNameMixin):

    @classmethod
    def _from_array(cls, data, var):
        # Accede a las variables

        # Si es la ultima variable almacena un nodo terminal.
        
    @classmethod
    def from_array(cls, data):

    

class TreePotential
