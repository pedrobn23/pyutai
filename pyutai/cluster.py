import collections
import dataclasses
import numpy as np
from typing import Dict, Tuple
  

@dataclasses.dataclass    
class ClusterPotential:
    clusters :  Dict[int, set] = dataclasses.field(default_factory=collections.defaultdict(set))
    cardinality : Tuple[int]
        
    @classmethod
    def from_array(cls, array : np.ndarray):
        self.cardinality = arr.shape
        for index, value in np.ndenumerate(array):
            self.cluster[value].append(index)            

if __name__ == '__main__':
    pot = Potential({Cluster({1,2,3}, 0.3), Cluster({4,5}, 0.25)}, Cluster({6,7,8,9}, 0.2)})

    
