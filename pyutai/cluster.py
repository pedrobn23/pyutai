"""
Cluster module implements k-meas cluster reduction of potentials. WIP at this moment.

[1] Wang, Haizhou & Song, Mingzhou. (2011). Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic Programming. The R Journal. 3. 29-33. 10.32614/RJ-2011-015. 
"""
import collections
import dataclasses
import functools

import numpy as np
from typing import Dict, Tuple, Iterable


@dataclasses.dataclass
class ClusterDistance:
    """Following notation from [1]."""
    median : float
    error : float
    

def euclidean_distance_table(elems):
    @functools.cache
    def distance(j,i):
        if j>i: # null case
            return ClusterDistance(0, 0)

        if j == i: # trivial case
            return ClusterDistance(elems(j), 0)

        else: # recursion case
            k = i-j
            cd = distance(j+1, i) 
            error = cd.error + (k) / (k+1) * (elems(i) - cd.median)**2
            median = (elems(i) + (i-1)*cd.median) / i
        

@dataclasses.dataclass
class ClusterPotential:
    clusters: Dict[float, set] = dataclasses.field(
        default_factory=collections.defaultdict(set))

    variables : List[str]
    cardinalities : Dict[str, int]
    
    
    def access(self, indexes: Tuple[int]) -> float:
        for value, index_set in self.clusters.items():
            if indexes in index_set:
                return value

    @classmethod
    def from_iterable(cls, it : Iterable):
        cluster = collections.defaultdict(set)
        for index, value in it:
            cluster[value].append(index)

        return cls(cluster, card)
            
    @classmethod
    def from_array(cls, array: np.ndarray):
        return cls.from_iterable(np.ndenumerate(array))

    @classmethod
    def from_tree(cls, tree):
        return cls.from_iterable(tree)

    

    @staticmethod
    def _update_distance(cls, old_distance, old_medium, new_element_value, length):
        if length == 0:
            return 0, new_element

    @classmethod
    def _compute_minimum(cls, D, j, m):
        minimum = math.inf
        distance = 0
        medium = 0
        length = 0
        index = j

        for i in range(j,m,-1):
            if (value := D[i-1, m-1] + d) > minimum:
                minimum = value
                index = i

            distance, medium, length = cls._update_distance()
            
        
    
    def reduce_cluster(self, goal):

        # Access values in increasing order.
        values =  list(clusters.keys())
        elememts = [(index, value) for index in self.cluster[value] for value in values.sorted()]
                
        
        # Little to do
        if goal >= len(self.cluster):
            return self

        # Dinamic porgraming
        self.D = [[0] * goal] * n_elements
        distance = euclidean_distance(elems)

        for i in range(1, n_elements):
            for m in range(1, goals):
                for j in range(i,m-1, -1):     
                    D[i][m] = min(D[j - 1][m - 1] + self.eu_distance(j, i) for j in )
