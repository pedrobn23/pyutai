import collections
import dataclasses
import numpy as np
from typing import Dict, Tuple
  

@dataclasses.dataclass    
class ClusterPotential:
    clusters :  Dict[int, set] = dataclasses.field(default_factory=collections.defaultdict(set))
    cardinality : Tuple[int] = None
    
    def access(indexes : Tuple[int]) -> float:
        for value, index_set in self.clusters:
            if indexes in index_set:
                return value

    
    @classmethod
    def from_array(cls, array : np.ndarray):
        self.cardinality = arr.shape
        for index, value in np.ndenumerate(array):
            self.cluster[value].append(index)            

    @staticmethod
    def euclidean_distance(*elems : float):
        mean = sum(elems)/len(elems)
        return sum((e-mean)**2 for e in elems)

    def reduce_cluster(self, goal):
        n_elements = np.prod(self.cardinality)
        
        if goal >= len(self.cluster):
            return self
        
        D = [[0]*goal]*n_elements]
        for i in range(1, n_elements):
            for m in range(1, goals):
                D[i][m] = min(D[i-1][j-1]

        
if __name__ == '__main__':
    clusters = {0.3 : {1,2,3}, 0.25 : {4,5}, 0.2 : {6,7,8,0}}
    cardinality = (9,)
    pot = ClusterPotential(clusters=clusters, cardinality=cardinality)

    
    
