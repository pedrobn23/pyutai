"""Cluster Experiment create an enviroment to test cluster reduction
capabilities on real datasets.
"""
import collections
import math
import os
import statistics
import sys

from typing import List

import numpy as np



from experiments import networks, read
from potentials import cluster, element, reductions
from potentials import utils as size_utils

def ordered_elements(array : np.ndarray) -> List[element.TupleElement]:
    res = [element.TupleElement(state=state,value=value)
           for state, value in np.ndenumerate(array)]
    res.sort(key=lambda x: x.value)
    return res

if __name__ == '__main__':
    error = 0.1
    
    for cpd in networks.medical():
        
        original_values = cpd.values
        ordered_elements_ = ordered_elements(original_values)
        threshold = len(np.unique(original_values))
        
        reduction = reductions.Reduction.from_elements(ordered_elements_,
                                                       threshold=threshold,
                                                       interactive=False)

        n_partitions = reduction.min_partitions(0.05)
        reduced_values = reduction.array(n_partitions, cpd.cardinality)

        if n_partitions != len(np.unique(reduced_values)):
            raise AssertionError('This should no happen')
        
        for cls in [cluster.Cluster]:

            print(f'results for {cls} class')

            original = cls.from_array(original_values, cpd.variables)
            original_size = size_utils.total_size(original)
            
            reduced =  cls.from_array(reduced_values, cpd.variables)
            reduced_size = size_utils.total_size(reduced)
            
            print(f'- Original cluster size: {original_size}')
            print(f'- Reduced cluster size: {reduced_size}')
            print(f'- Total reduction: {1 - (reduced_size/original_size):.2f}% ')

            
